#!/usr/bin/env python3
"""Build a stage-B training file that carries a chosen share of job A's QUESTIONS.

    python mix.py --b /work/data/gsm8k/train.jsonl --a /work/data/spider/train.jsonl \
                  --share 0.10 --seed 0 --out /work/data/mix-b-rehearse10

WHAT REHEARSAL MEANS HERE. The rows this writes are job A's *questions*, in job A's own trainer
format, with job A's own `data_source` and `reward_model`. Nothing about the answers is stored: the
model being trained answers them afresh at every step and job A's checker rewards those answers, so
the rehearsal is as on-policy as the stage-B rows beside it. Replaying stored answers is a different
experiment and is not what this file builds. `kit/beds/rewards.py` is what rewards the mixture: one
entry point, dispatching on `data_source`.

THE SHARE. Every job-B row is kept and job-A rows are ADDED, enough of them that job A is `--share`
of the rows written: k = the whole number that puts k / (rows_B + k) closest to the share. So a share
of 0.10 over 640 job-B rows adds 71 job-A rows and writes 711; 0.30 adds 274 and writes 914. Training
length is therefore set by the mixture, not held fixed -- an arm with more rehearsal sees more rows,
and the manifest records exactly how many of each so a readout can say so.

SPREAD. Job-A rows are placed at even intervals through the file, because the trainer reads it in
order: every step then carries its share of rehearsal, instead of none in one step and six in the next.

DETERMINISM. `--seed` fixes which job-A rows are drawn and the order everything is written in: the
same inputs, share and seed give byte-identical files, and a different seed gives different ones.
Job-A rows are drawn without replacement while they last; if the share asks for more rows than job A
has, the rest are drawn with replacement and the manifest counts them.

REFUSALS. A share outside (0, 0.5]: rehearsal is a minority of the batch, and a file where the old
job is the majority is a different experiment. Inputs whose columns differ, or a file whose own rows
do not all carry the same columns: the trainer reads one table, and a row missing a column is a
silent failure inside it. An existing output: nothing here is ever overwritten.

Standard library, plus pyarrow to write (and read) parquet. The jsonl is the exact record; parquet
unions the nested `extra_info` and `reward_model` keys of the two beds and fills the missing ones
with nulls, which is how one table can hold two beds' rows at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

SCHEMA = "kit-mix.v1"
MAX_SHARE = 0.5
ORDER_SEED_OFFSET = 1_000_003


class MixError(ValueError):
    """The inputs or the request are wrong; nothing may be written."""


# ------------------------------------------------------------------------------------ reading
def read_rows(path) -> list:
    """Trainer rows from a .jsonl (one object a line) or a .parquet file."""
    path = Path(path)
    if not path.is_file():
        raise MixError("no such input file: %s" % path)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq                                      # noqa: PLC0415
        except ImportError:
            raise MixError("%s is parquet and pyarrow is not installed; pass the jsonl instead" % path) from None
        rows = pq.read_table(path).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if not rows:
        raise MixError("%s has no rows" % path)
    if any(not isinstance(row, dict) for row in rows):
        raise MixError("%s has a row that is not a JSON object" % path)
    return rows


def columns_of(rows: list, path) -> tuple:
    """The one column set every row in a file must have. A file that disagrees with itself is refused."""
    first = tuple(sorted(rows[0]))
    for index, row in enumerate(rows):
        if tuple(sorted(row)) != first:
            raise MixError("row %d of %s has columns %s, row 0 has %s: one file must be one table"
                           % (index, path, sorted(row), list(first)))
    return first


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------ the mixture
def rows_to_add(n_b: int, share: float) -> int:
    """How many job-A rows make job A `share` of (n_b + k) rows: the whole number that lands closest."""
    if not isinstance(share, (int, float)) or isinstance(share, bool):
        raise MixError("the share must be a number")
    if not 0 < share <= MAX_SHARE:
        raise MixError("refusing a rehearsal share of %r: it must be greater than 0 and at most %g"
                       % (share, MAX_SHARE))
    if n_b <= 0:
        raise MixError("the job-B file has no rows to mix into")
    exact = share * n_b / (1.0 - share)
    candidates = [k for k in (int(exact), int(exact) + 1) if k >= 1] or [1]
    return min(candidates, key=lambda k: (abs(k / (n_b + k) - share), k))


def draw(n_a: int, k: int, seed: int) -> list:
    """Which job-A rows to add: without replacement while they last, then with. Deterministic in `seed`."""
    rng = random.Random(seed)
    pool = list(range(n_a))
    rng.shuffle(pool)
    if k <= n_a:
        return pool[:k]
    return pool + [rng.randrange(n_a) for _ in range(k - n_a)]


def mix(rows_b: list, rows_a: list, share: float, seed: int) -> tuple:
    """(rows, facts): the shuffled mixture, and the counts a manifest and a readout need."""
    k = rows_to_add(len(rows_b), share)
    chosen = draw(len(rows_a), k, seed)
    # A separate stream for the order, far from the draw's, so that one arm's order is not another arm's draw.
    order = random.Random(seed + ORDER_SEED_OFFSET)
    rows_b, added = list(rows_b), [rows_a[index] for index in chosen]
    order.shuffle(rows_b)
    order.shuffle(added)
    # Job-A rows are SPREAD EVENLY, not scattered: the trainer reads the file in order, 32 questions a
    # step, and a plain shuffle at a 10 percent share leaves some steps with no rehearsal at all and
    # others with six. Row j of k sits in the middle of the j-th of k equal stretches of the file.
    total = len(rows_b) + k
    slots = {int((j + 0.5) * total / k) for j in range(k)}
    assert len(slots) == k, "the share cap keeps the stretches at least two rows long"
    a_iter, b_iter = iter(added), iter(rows_b)
    rows = [next(a_iter) if position in slots else next(b_iter) for position in range(total)]
    facts = {"rows_b": len(rows_b), "rows_a_added": k, "rows_total": len(rows),
             "rows_a_available": len(rows_a), "rows_a_distinct_used": len(set(chosen)),
             "rows_a_drawn_with_replacement": max(0, k - len(rows_a)),
             "target_share": float(share), "achieved_share": round(k / len(rows), 6), "seed": int(seed)}
    facts["share_error"] = round(abs(facts["achieved_share"] - facts["target_share"]), 6)
    return rows, facts


def as_jsonl(rows: list) -> str:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)


def data_source_counts(rows: list) -> dict:
    counts: dict = {}
    for row in rows:
        key = str(row.get("data_source", "-"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# ------------------------------------------------------------------------------------ writing
def write_new(path: Path, text: str) -> str:
    if path.exists():
        raise MixError("refusing to overwrite %s: an output is never replaced, write to a new directory" % path)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_table(rows: list):
    """The rows as one arrow table, or None when pyarrow is not installed.

    Built BEFORE anything is written, so a mixture pyarrow cannot represent is refused while the
    output directory is still empty rather than half written and then blocked by the overwrite rule.
    """
    try:
        import pyarrow as pa                                                  # noqa: PLC0415
    except ImportError:
        return None
    return pa.Table.from_pylist(rows)


def write_parquet(path: Path, table) -> bool:
    if table is None:
        return False
    if path.exists():
        raise MixError("refusing to overwrite %s: an output is never replaced, write to a new directory" % path)
    import pyarrow.parquet as pq                                              # noqa: PLC0415
    pq.write_table(table, path)
    return True


def build(a_path, b_path, share: float, seed: int, out, *, name: str = "train",
          allow_no_parquet: bool = False) -> dict:
    """Write the mixture and its manifest under `out`, and return the manifest."""
    rows_a, rows_b = read_rows(a_path), read_rows(b_path)
    columns_a, columns_b = columns_of(rows_a, a_path), columns_of(rows_b, b_path)
    if columns_a != columns_b:
        raise MixError("the two inputs have different columns: job A has %s, job B has %s. The trainer reads one "
                       "table; prepare both beds with the same bed writer and mix those files."
                       % (list(columns_a), list(columns_b)))
    rows, facts = mix(rows_b, rows_a, share, seed)
    table = build_table(rows)
    if table is None and not allow_no_parquet:
        raise MixError("pyarrow is not installed, so the parquet the trainer reads cannot be written and nothing "
                       "has been. Install pyarrow, or pass --allow-no-parquet on purpose.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "mix.manifest.json"
    if manifest_path.exists():
        raise MixError("refusing to overwrite %s: an output is never replaced, write to a new directory" % manifest_path)
    jsonl_sha = write_new(out / ("%s.jsonl" % name), as_jsonl(rows))
    wrote_parquet = write_parquet(out / ("%s.parquet" % name), table)
    manifest = {"schema": SCHEMA, "rehearsal": "job-A questions answered afresh by the model being trained",
                **facts,
                "columns": list(columns_b), "data_source_counts": data_source_counts(rows),
                "inputs": {"a": {"path": str(Path(a_path).resolve()), "sha256": sha256_file(a_path), "rows": len(rows_a)},
                           "b": {"path": str(Path(b_path).resolve()), "sha256": sha256_file(b_path), "rows": len(rows_b)}},
                "outputs": {"jsonl": "%s.jsonl" % name, "jsonl_sha256": jsonl_sha,
                            "parquet": "%s.parquet" % name if wrote_parquet else None}}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mix job-A questions into a job-B training file for rehearsal.")
    parser.add_argument("--b", required=True, help="the job being learned in stage B (every row is kept)")
    parser.add_argument("--a", required=True, help="the earlier job whose QUESTIONS are rehearsed")
    parser.add_argument("--share", type=float, required=True, help="job A's share of the rows written, in (0, 0.5]")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="a directory; nothing in it is overwritten")
    parser.add_argument("--name", default="train", help="the stem of the written files (default: train)")
    parser.add_argument("--allow-no-parquet", action="store_true",
                        help="write the jsonl alone when pyarrow is missing; the trainer reads parquet")
    args = parser.parse_args(argv)
    try:
        manifest = build(args.a, args.b, args.share, args.seed, args.out, name=args.name,
                         allow_no_parquet=args.allow_no_parquet)
    except MixError as exc:
        raise SystemExit(str(exc))
    print("wrote %s: %d rows = %d job-B + %d job-A (%d distinct, %d drawn twice or more); job A is %.4f of the rows "
          "(asked for %.4f), seed %d%s"
          % (args.out, manifest["rows_total"], manifest["rows_b"], manifest["rows_a_added"],
             manifest["rows_a_distinct_used"], manifest["rows_a_drawn_with_replacement"],
             manifest["achieved_share"], manifest["target_share"], manifest["seed"],
             "" if manifest["outputs"]["parquet"] else " (jsonl only: pyarrow not installed)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
