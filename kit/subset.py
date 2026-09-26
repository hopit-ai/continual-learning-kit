#!/usr/bin/env python3
"""Drop the training questions the model already solves every time, and keep the rest as a file.

    python subset.py drop-solved --from $WORK/data/spider/train.parquet \\
                                 --stuck $WORK/k3dose/stuck/spider-stuck-a1/stuck.json \\
                                 --out $WORK/k3dose/data/hard-subset-a1

WHY THIS FILE EXISTS. `kit/hints.py stuck` samples every training question 8 times at temperature
1.0 -- the trainer's own sampling -- and records, per question, how many of those attempts the bed's
own checker called right. On Spider's 640 training questions the untrained Qwen3-1.7B solved 300 of
them in ALL 8 attempts and none of 171 (docs/phase2/evidence/k4-gate, receipt 227). A question solved
every time contributes no gradient to GRPO: every sample in its group scores 1, so its advantage is
zero. Nearly half of a stage-A batch is therefore inert, and `hard20` -- one arm of the K3 stage-A
dose probe -- asks what the same 20 steps do when those questions are gone.

WHAT IT DOES, and what it does NOT do. It keeps every row of the input whose question id is not
recorded with `wins == attempts`, in the input's own order, and writes the trainer's parquet and the
jsonl beside it. The rows are the bed's own trainer rows, unchanged: nothing about a prompt, an
answer or a checker is touched, and the model answers them afresh at every step exactly as before.
Sizing the file to a dose is `kit/sequence.py pool`'s job, not this one's.

FOUR REFUSALS, each of which would otherwise be a silently wrong training file:
  - a stuck.json that is not one `kit/hints.py stuck` wrote;
  - a recorded question that is not in the input file (the stuck set and the file are then not the
    same rows, and the subset would be a subset of something else);
  - two input rows carrying one question id;
  - fewer than 32 rows left, which is one batch: a file shorter than that cannot fill a single step.

Standard library, plus pyarrow to write the parquet the trainer reads, as `kit/sequence.py` uses it.
Nothing here trains, scores or downloads anything, and nothing is ever overwritten: an existing
--out is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-subset.v1"
#: One batch of `kit/run_grpo.sh` (data.train_batch_size=32). A file shorter than this cannot fill a step.
BATCH = 32


class SubsetError(ValueError):
    """The inputs cannot support the subset that was asked for; nothing is written."""


# ------------------------------------------------------------------------------------ reading
# read_rows and columns_of are DUPLICATED from kit/sequence.py on purpose, for the reason given at
# the top of that file: a shared helper would let a change there move a number here silently.
# tests/test_kit_k3_dose.py compares the two and fails if either side moves.
def read_rows(path) -> list:
    """Trainer rows from a .jsonl (one object a line) or a .parquet file."""
    path = Path(path)
    if not path.is_file():
        raise SubsetError("no such input file: %s" % path)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq                                      # noqa: PLC0415
        except ImportError:
            raise SubsetError("%s is parquet and pyarrow is not installed; pass the jsonl instead" % path) from None
        rows = pq.read_table(path).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if not rows:
        raise SubsetError("%s has no rows" % path)
    if any(not isinstance(row, dict) for row in rows):
        raise SubsetError("%s has a row that is not a JSON object" % path)
    return rows


def columns_of(rows: list, path) -> tuple:
    """The one column set every row in a file must have. A file that disagrees with itself is refused."""
    first = tuple(sorted(rows[0]))
    for index, row in enumerate(rows):
        if tuple(sorted(row)) != first:
            raise SubsetError("row %d of %s has columns %s, row 0 has %s: one file must be one table"
                              % (index, path, sorted(row), list(first)))
    return first


def question_id(row: dict) -> str:
    """How a training row names its question: the bed's own id, in `extra_info.index`.

    The same rule as kit/hints.py's question_id(), which is what wrote the ids in the stuck.json read
    here. Duplicated rather than imported for the reason above; the test pins the two together.
    """
    index = (row.get("extra_info") or {}).get("index")
    if not isinstance(index, str) or not index:
        raise SubsetError("a training row has no `extra_info.index`: this is not a row one of this "
                          "kit's beds wrote, and it could not be matched to a stuck record")
    return index


def rows_by_id(rows: list) -> dict:
    found: dict = {}
    for row in rows:
        key = question_id(row)
        if key in found:
            raise SubsetError("two training rows carry the question id %s" % key)
        found[key] = row
    return found


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_jsonl(rows: list) -> str:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_stuck(path) -> dict:
    """The record `kit/hints.py stuck` wrote. Anything else is refused rather than half-read."""
    path = Path(path)
    if not path.is_file():
        raise SubsetError("no such stuck file: %s" % path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SubsetError("%s is not JSON: %s" % (path, exc)) from exc
    if not isinstance(record, dict) or record.get("stage") != "stuck":
        raise SubsetError("%s is not a stuck.json kit/hints.py wrote (its `stage` is not \"stuck\")" % path)
    per_question = record.get("per_question")
    if not isinstance(per_question, list) or not per_question:
        raise SubsetError("%s carries no `per_question` list, so it says nothing about any question" % path)
    for entry in per_question:
        if not isinstance(entry, dict) or not isinstance(entry.get("index"), str):
            raise SubsetError("%s has a per_question entry with no question id: %r" % (path, entry))
        for key in ("wins", "attempts"):
            value = entry.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SubsetError("%s: question %s has %s=%r, which is not a count"
                                  % (path, entry["index"], key, value))
        if entry["attempts"] == 0:
            raise SubsetError("%s: question %s was never attempted, so nothing is known about it"
                              % (path, entry["index"]))
        if entry["wins"] > entry["attempts"]:
            raise SubsetError("%s: question %s won %d of %d attempts"
                              % (path, entry["index"], entry["wins"], entry["attempts"]))
    return record


# ------------------------------------------------------------------------------------ the subset
def histogram(entries: list, attempts: int) -> list:
    return [sum(1 for entry in entries if entry["wins"] == k) for k in range(attempts + 1)]


def drop_solved(rows: list, record: dict) -> tuple:
    """(kept rows, facts). Every row whose question was NOT solved in all of its attempts, in order."""
    by_id = rows_by_id(rows)
    per_question = record["per_question"]
    unknown = [entry["index"] for entry in per_question if entry["index"] not in by_id]
    if unknown:
        raise SubsetError("%d of the %d questions in the stuck record are not in this training file "
                          "(e.g. %s): the stuck set and the training file must be the same rows, or "
                          "the subset would be a subset of something else"
                          % (len(unknown), len(per_question), ", ".join(unknown[:3])))
    attempts = max(entry["attempts"] for entry in per_question)
    solved = {entry["index"] for entry in per_question if entry["wins"] == entry["attempts"]}
    kept = [row for row in rows if question_id(row) not in solved]
    if len(kept) < BATCH:
        raise SubsetError("only %d of %d rows are left after dropping the %d questions solved every "
                          "time, and one step needs %d: there is no dose to train on"
                          % (len(kept), len(rows), len(solved), BATCH))
    kept_ids = {question_id(row) for row in kept}
    facts = {"input_rows": len(rows), "questions_recorded": len(per_question), "attempts": attempts,
             "dropped_always_solved": len(solved), "kept": len(kept),
             "rows_with_no_record": len(rows) - len(per_question),
             "wins_histogram": histogram(per_question, attempts),
             "wins_histogram_kept": histogram([e for e in per_question if e["index"] in kept_ids], attempts),
             "never_solved": sum(1 for entry in per_question if entry["wins"] == 0)}
    return kept, facts


def build_table(rows: list):
    """The rows as one arrow table, or None when pyarrow is missing. Built BEFORE anything is written."""
    try:
        import pyarrow as pa                                                   # noqa: PLC0415
    except ImportError:
        return None
    return pa.Table.from_pylist(rows)


def build_subset(source, stuck, out, *, name: str = "train", allow_no_parquet: bool = False) -> dict:
    """The whole job: read, refuse, write. Every refusal happens before the output directory exists."""
    out = Path(out)
    if out.exists():
        raise SubsetError("refusing to overwrite %s: an output is never replaced, choose a new --out" % out)
    rows = read_rows(source)
    columns = columns_of(rows, source)
    record = load_stuck(stuck)
    kept, facts = drop_solved(rows, record)
    table = build_table(kept)
    if table is None and not allow_no_parquet:
        raise SubsetError("pyarrow is not installed, so the parquet the trainer reads cannot be written "
                          "and nothing has been. Install pyarrow, or pass --allow-no-parquet on purpose.")
    out.mkdir(parents=True)
    text = as_jsonl(kept)
    (out / ("%s.jsonl" % name)).write_text(text, encoding="utf-8")
    parquet_sha = None
    if table is not None:
        import pyarrow.parquet as pq                                           # noqa: PLC0415
        pq.write_table(table, out / ("%s.parquet" % name))
        parquet_sha = sha256_file(out / ("%s.parquet" % name))
    manifest = {"schema": SCHEMA, "stage": "drop-solved", "generated_at": now(),
                "what": "every row of the input whose question the model did not solve in ALL of its "
                        "attempts; a question solved every time has a zero advantage in GRPO and "
                        "contributes no gradient",
                **facts, "columns": list(columns),
                "input": {"path": str(Path(source).resolve()), "sha256": sha256_file(source),
                          "rows": len(rows)},
                "stuck": {"path": str(Path(stuck).resolve()), "sha256": sha256_file(stuck),
                          "model": record.get("model"), "data_source": record.get("data_source"),
                          "attempts": record.get("attempts"), "temperature": record.get("temperature"),
                          "questions": record.get("questions"), "stuck_questions": record.get("stuck"),
                          "solved_every_time": record.get("solved_every_time")},
                "outputs": {"jsonl": "%s.jsonl" % name, "jsonl_sha256": sha256_text(text),
                            "parquet": "%s.parquet" % name if parquet_sha else None,
                            "parquet_sha256": parquet_sha}}
    (out / "subset.manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                              encoding="utf-8")
    return manifest


# ------------------------------------------------------------------------------------ the command line
def cmd_drop_solved(args) -> int:
    manifest = build_subset(args.source, args.stuck, args.out, name=args.name,
                            allow_no_parquet=args.allow_no_parquet)
    print("kept %d of %d rows: %d questions were solved in all %d attempts, %d never were; "
          "wins histogram %s"
          % (manifest["kept"], manifest["input_rows"], manifest["dropped_always_solved"],
             manifest["attempts"], manifest["never_solved"], manifest["wins_histogram"]))
    print("wrote", Path(args.out) / (manifest["outputs"]["parquet"] or manifest["outputs"]["jsonl"]))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("drop-solved", help="keep only the questions the model does not solve every time")
    one.add_argument("--from", dest="source", required=True, help="a prepared bed's trainer file")
    one.add_argument("--stuck", required=True, help="the stuck.json written by `kit/hints.py stuck`")
    one.add_argument("--out", required=True, help="a directory; it must not exist")
    one.add_argument("--name", default="train", help="the stem of the written files (default: train)")
    one.add_argument("--allow-no-parquet", action="store_true",
                     help="write the jsonl alone when pyarrow is missing; the trainer reads parquet")
    args = parser.parse_args(argv)
    try:
        return {"drop-solved": cmd_drop_solved}[args.action](args)
    except SubsetError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    sys.exit(main())
