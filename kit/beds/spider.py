#!/usr/bin/env python3
"""Spider 1.0 text-to-SQL as a training and evaluation bed: prompts, a scoring rule, a reward function.

Spider 1.0 (Yu et al., 2018; CC BY-SA 4.0; yale-lily.github.io/spider) asks a natural-language
question about one of 166 small SQLite databases and expects a single SELECT statement. This bed is
the SQL half of the rehearsal test: a model learns SQL here, then learns maths somewhere else, and
the question is what is left of the SQL.

WHAT IS FROZEN, and why the loader refuses rather than assumes. Phase 1 of this programme measured a
particular 640-question training set and a particular 100-question held-out panel, with prompts
rendered in a particular way. Spider is not ours to redistribute, so `spider-split-v1.json` carries
no question text, no SQL and no gold rows: it carries the question ids, the database ids, and the
sha256 of every prompt and of every SQLite file. Point this module at your own copy of Spider and it
rebuilds each prompt from that copy and refuses unless every hash matches. A different Spider
release, an edited database, a changed prompt: each is a refusal, not a quietly different number.

THE SCORING RULE, as phase 1 applied it. The response's SQL is taken from a ```sql fence (else the
first SELECT statement), executed read-only against the question's own database under a hard
per-query time limit, and compared with the result of the gold query on the same database: the two
result sets are equal when their rows, each cell stringified and the rows sorted, are equal. That is
execution-result equality, not official Spider test-suite accuracy, and it makes no claim of
semantic SQL equivalence: an unordered gold query and a differently ordered answer count as equal,
and a coincidentally equal result counts as correct.

    python3 spider.py prepare --spider-root /path/to/spider_data --out /work/data/spider   # CPU, before any GPU is held
    SPIDER_ROOT=/path/to/spider_data python3 spider.py score --split heldout --responses responses.jsonl

`compute_score` has the signature the SDPO reference expects of a custom reward function. Its
feedback never contains the gold query, the gold rows, or how many rows they are: phase 1's training
feedback did disclose the expected row count, which was recorded as a departure; this bed does not.
A teacher may see that the attempt was wrong, never what right would have been.

Standard library only, plus pyarrow when it happens to be installed. Nothing here imports anything
from this programme's private packages.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

DATA_SOURCE = "spider1_execution"
SPLIT_FILE = Path(__file__).resolve().with_name("spider-split-v1.json")
SPLIT_SCHEMA = "kit-bed-spider-split.v1"
SPLITS = ("train", "heldout")

INSTRUCTION = (
    "Write a single SQLite SELECT statement that answers the question, inside a ```sql fence. "
    "Think briefly first if needed."
)

# One candidate query's wall-clock budget. Ten seconds is the limit phase 1's verifier ran under: of
# 1,567 gold queries in its laboratory 1,552 finished inside five seconds, and one model-written
# aggregate over an unconstrained join never finished at all and cost a training run. A Python signal
# cannot stop SQLite's C loop; `set_progress_handler` can, and a query stopped that way surfaces as an
# ordinary execution error that the caller already handles.
QUERY_TIMEOUT_S = 10.0
# Result sets are compared after truncation, so a low limit would compare two different arbitrary
# prefixes of the same unordered answer and call a correct query wrong.
ROW_LIMIT = 100_000

MEMBER_ID = re.compile(r"^spider-(train_spider|train_others)-([A-Za-z0-9_]+)-(\d+)$")
DATABASE_ID = re.compile(r"^[A-Za-z0-9_]+$")
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
_SELECT = re.compile(r"(SELECT\b.*?)(?:;|$)", re.S | re.I)
_READ_ONLY = re.compile(r"\s*(SELECT|WITH)\b", re.I)

# The only feedback this bed ever returns. Fixed strings: no gold query, no gold rows, no gold row
# count, no executor exception text, and no digits at all, so no count can leak by accident.
FEEDBACK = {
    "correct": "",
    "mismatch": "The query executed but did not return the expected result. Re-check the filter "
                "conditions, the joins, and which columns are selected.",
    "no_sql": "Return a single SQLite SELECT statement inside a ```sql fence.",
    "unsupported_statement": "Only a read-only SELECT query is supported.",
    "error": "The query failed to execute against the database.",
    "timeout": "The query exceeded the execution time limit.",
    "row_limit": "The query returned more rows than the result limit allows.",
}
INCORRECT_FORMAT = {"no_sql", "unsupported_statement"}


class SpiderBedError(Exception):
    """A source, a hash or a request does not match what the split file pins."""


# ----------------------------------------------------------------- the frozen split and its sources
def load_split(path=SPLIT_FILE) -> dict:
    """Read `spider-split-v1.json` and check it is the shape the loader is allowed to trust."""
    path = Path(path)
    try:
        split = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpiderBedError("cannot read the split file %s: %s" % (path, exc)) from exc
    except ValueError as exc:
        raise SpiderBedError("the split file %s is not JSON: %s" % (path, exc)) from exc
    if not isinstance(split, dict) or split.get("schema") != SPLIT_SCHEMA:
        raise SpiderBedError("%s is not a %s file" % (path, SPLIT_SCHEMA))
    if split.get("instruction_sha256") != hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest():
        raise SpiderBedError("the split file was built with a different instruction line than this module renders")
    databases = split.get("databases")
    if not isinstance(databases, dict) or not databases:
        raise SpiderBedError("the split file lists no databases")
    for name in SPLITS:
        members = split.get(name)
        if not isinstance(members, list) or not members:
            raise SpiderBedError("the split file has no %r members" % name)
        for row in members:
            if not isinstance(row, dict) or set(row) != {"id", "db", "prompt_sha256"}:
                raise SpiderBedError("a %r member is not {id, db, prompt_sha256}" % name)
            if row["db"] not in databases:
                raise SpiderBedError("member %s uses database %s, which the split file does not pin" % (row["id"], row["db"]))
    return split


def database_path(spider_root, database: str) -> Path:
    """`<root>/database/<db>/<db>.sqlite`, with the id checked before it becomes a path."""
    if not isinstance(database, str) or not DATABASE_ID.match(database):
        raise SpiderBedError("unsafe Spider database id: %r" % (database,))
    root = Path(spider_root)
    path = root / "database" / database / ("%s.sqlite" % database)
    if not path.is_file():
        raise SpiderBedError("missing Spider database %s under SPIDER_ROOT=%s (expected %s)" % (database, root, path))
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_database(spider_root, database: str, split=None) -> Path:
    """The database file, refused unless its bytes hash to what the split file pins."""
    split = load_split() if split is None else split
    entry = split["databases"].get(database)
    if entry is None:
        raise SpiderBedError("database %s is not pinned by the split file" % database)
    path = database_path(spider_root, database)
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise SpiderBedError("database %s has sha256 %s, the split file pins %s: this is not the Spider release "
                             "phase 1 measured" % (database, actual[:12], entry["sha256"][:12]))
    return path


def sqlite_schema(path) -> str:
    """The database's DDL, exactly as phase 1 put it in the prompt: tables and views, by name."""
    connection = sqlite3.connect("file:%s?mode=ro" % Path(path).resolve(), uri=True)
    try:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    finally:
        connection.close()
    ddl = "\n".join(str(row[0]).strip().rstrip(";") for row in rows if row[0])
    if not ddl:
        raise SpiderBedError("database has no schema to render: %s" % path)
    return ddl


def render_prompt(schema: str, question: str) -> str:
    return "Schema:\n%s\n\nQuestion: %s\n\n%s" % (schema, question, INSTRUCTION)


def _source_rows(spider_root, source: str) -> list:
    path = Path(spider_root) / ("%s.json" % source)
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpiderBedError("missing Spider source file %s (is SPIDER_ROOT right?): %s" % (path, exc)) from exc
    except ValueError as exc:
        raise SpiderBedError("Spider source file %s is not JSON: %s" % (path, exc)) from exc
    if not isinstance(rows, list):
        raise SpiderBedError("Spider source file %s is not an array" % path)
    return rows


def load_members(spider_root, split_name: str, split=None) -> list:
    """Rebuild one split's members from a local Spider copy, refusing every hash that disagrees.

    Returns, per member: id, db, source (train_spider or train_others), source_index, question,
    gold_sql, prompt and prompt_sha256. Every database used is hashed once; every prompt is rebuilt
    and hashed. A mismatch raises: there is no mode in which this returns a prompt phase 1 did not
    render, or a database phase 1 did not execute against.
    """
    if split_name not in SPLITS:
        raise SpiderBedError("unknown split %r: expected one of %s" % (split_name, ", ".join(SPLITS)))
    split = load_split() if split is None else split
    sources, schemas, members = {}, {}, []
    for database in sorted({row["db"] for row in split[split_name]}):
        check_database(spider_root, database, split)
    for row in split[split_name]:
        match = MEMBER_ID.match(row["id"])
        if not match or match.group(2) != row["db"]:
            raise SpiderBedError("split member %r is not spider-<split>-<db>-<index> for database %s" % (row["id"], row["db"]))
        source, database, index = match.group(1), row["db"], int(match.group(3))
        if source not in sources:                            # read each source file and each schema once
            sources[source] = _source_rows(spider_root, source)
        source_rows = sources[source]
        if index >= len(source_rows):
            raise SpiderBedError("%s.json has %d questions, %s needs index %d: this is not the pinned Spider release"
                                 % (source, len(source_rows), row["id"], index))
        entry = source_rows[index] if isinstance(source_rows[index], dict) else {}
        question, gold_sql = str(entry.get("question", "")), str(entry.get("query", ""))
        if entry.get("db_id") != database or not question.strip() or not gold_sql.strip():
            raise SpiderBedError("%s.json[%d] is not the question %s pins" % (source, index, row["id"]))
        if database not in schemas:
            schemas[database] = sqlite_schema(database_path(spider_root, database))
        prompt = render_prompt(schemas[database], question)
        actual = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if actual != row["prompt_sha256"]:
            raise SpiderBedError("the prompt rebuilt for %s has sha256 %s, the split file pins %s: this Spider copy "
                                 "does not render the prompt phase 1 measured" % (row["id"], actual[:12], row["prompt_sha256"][:12]))
        members.append({"id": row["id"], "db": database, "source": source, "source_index": index,
                        "question": question, "gold_sql": gold_sql, "prompt": prompt, "prompt_sha256": actual})
    return members


# ------------------------------------------------------------------------------- answers and scoring
def extract_sql(text: str):
    """The query in a ```sql fence; failing that, the first SELECT statement. None when neither."""
    match = _FENCE.search(text or "")
    if match and re.search(r"\bselect\b", match.group(1), re.I):
        return match.group(1).strip().rstrip(";").strip()
    match = _SELECT.search(text or "")
    return match.group(1).strip() if match else None


def execute(db_path, query: str, timeout_s: float = QUERY_TIMEOUT_S) -> dict:
    """Run one query read-only under a hard wall-clock budget. Never raises on a bad query."""
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= 600:
        raise SpiderBedError("the query time limit must be positive and no greater than 600 seconds")
    connection = sqlite3.connect("file:%s?mode=ro" % Path(db_path).resolve(), uri=True)
    connection.text_factory = lambda value: value.decode("utf-8", "replace")
    started = time.monotonic()
    deadline = started + timeout_s
    connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        rows = connection.execute(query).fetchmany(ROW_LIMIT + 1)
        if len(rows) > ROW_LIMIT:
            return {"ok": False, "kind": "row_limit", "elapsed_s": time.monotonic() - started}
        return {"ok": True, "kind": "correct", "rows": [list(row) for row in rows], "elapsed_s": time.monotonic() - started}
    except sqlite3.Error as exc:
        elapsed = time.monotonic() - started
        return {"ok": False, "kind": "timeout" if elapsed >= timeout_s else "error",
                "elapsed_s": elapsed, "error": "%s: %s" % (type(exc).__name__, exc)}
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def normalise(rows) -> list:
    """Phase 1's comparison: every cell stringified, the rows sorted, the column order kept."""
    return sorted(tuple(str(value) for value in row) for row in rows)


def is_correct(rows, gold_rows) -> bool:
    return normalise(rows) == normalise(gold_rows)


def gold_rows_for(db_path, gold_sql: str, timeout_s: float = QUERY_TIMEOUT_S) -> list:
    """Execute the gold query. A gold query that does not execute is a source problem, so it raises."""
    result = execute(db_path, gold_sql, timeout_s=timeout_s)
    if not result["ok"]:
        raise SpiderBedError("the gold query did not execute (%s) against %s" % (result["kind"], db_path))
    return result["rows"]


@functools.lru_cache(maxsize=4096)
def _cached_gold(db_path: str, gold_sql: str, timeout_s: float) -> tuple:
    return tuple(tuple(row) for row in gold_rows_for(db_path, gold_sql, timeout_s))


@functools.lru_cache(maxsize=256)
def _verified_database(spider_root: str, database: str, split_file: str) -> str:
    """Hash a database once per process, then reuse the path: a reward function is called per rollout."""
    return str(check_database(spider_root, database, load_split(split_file)))


def score_response(response: str, gold_sql: str, db_path, timeout_s: float = QUERY_TIMEOUT_S) -> dict:
    """One response against one gold query: the outcome kind and whether it was correct."""
    sql = extract_sql(response or "")
    if sql is None:
        return {"kind": "no_sql", "correct": False, "pred": ""}
    if not _READ_ONLY.match(sql):
        return {"kind": "unsupported_statement", "correct": False, "pred": sql}
    gold = _cached_gold(str(Path(db_path).resolve()), gold_sql, float(timeout_s))
    result = execute(db_path, sql, timeout_s=timeout_s)
    if not result["ok"]:
        return {"kind": result["kind"], "correct": False, "pred": sql}
    correct = is_correct(result["rows"], gold)
    return {"kind": "correct" if correct else "mismatch", "correct": correct, "pred": sql}


def spider_root_from_env(spider_root=None) -> Path:
    """The database root at scoring time: the argument if given, else SPIDER_ROOT."""
    root = spider_root if spider_root is not None else os.environ.get("SPIDER_ROOT")
    if not root:
        raise SpiderBedError("SPIDER_ROOT is not set: the scorer needs the directory holding Spider's "
                             "database/, train_spider.json and train_others.json")
    path = Path(root)
    if not path.is_dir():
        raise SpiderBedError("SPIDER_ROOT=%s is not a directory" % path)
    return path


def ground_truth_for(member: dict) -> str:
    return json.dumps({"db": member["db"], "gold_sql": member["gold_sql"]}, sort_keys=True)


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None,
                  *, spider_root=None, split_file=SPLIT_FILE, timeout_s: float = QUERY_TIMEOUT_S) -> dict:
    """Reward function in the SDPO reference's shape. The feedback never reveals `ground_truth`.

    `ground_truth` is the JSON string `{"db": ..., "gold_sql": ...}`; the database root comes from
    SPIDER_ROOT unless one is passed, and the database it names is hashed against the split file
    before it is executed against. Identity failures raise: a reward that silently scores against the
    wrong database is worse than a crash.
    """
    if data_source != DATA_SOURCE:
        raise SpiderBedError("unsupported reward data source %r, expected %r" % (data_source, DATA_SOURCE))
    if not isinstance(solution_str, str):
        raise SpiderBedError("the response must be text")
    try:
        reference = json.loads(ground_truth)
    except (TypeError, ValueError) as exc:
        raise SpiderBedError("ground_truth is not a JSON object: %s" % exc) from exc
    if not isinstance(reference, dict) or set(reference) != {"db", "gold_sql"}:
        raise SpiderBedError("ground_truth must be {\"db\": ..., \"gold_sql\": ...}")
    root = spider_root_from_env(spider_root)
    db_path = _verified_database(str(root), reference["db"], str(split_file))
    outcome = score_response(solution_str, reference["gold_sql"], db_path, timeout_s=timeout_s)
    return {"score": float(outcome["correct"]), "acc": float(outcome["correct"]), "pred": outcome["pred"],
            "incorrect_format": int(outcome["kind"] in INCORRECT_FORMAT), "feedback": FEEDBACK[outcome["kind"]]}


def rows_for_trainer(members, split_name: str) -> list:
    return [{"data_source": DATA_SOURCE, "prompt": [{"role": "user", "content": member["prompt"]}], "ability": "sql",
             "reward_model": {"style": "rule", "ground_truth": ground_truth_for(member)},
             "extra_info": {"split": split_name, "index": member["id"], "problem": member["question"],
                            "db": member["db"], "prompt_sha256": member["prompt_sha256"],
                            "description": "", "elo": 0, "achievement_prior": 0}}
            for member in members]


# --------------------------------------------------------------------------------------- subcommands
def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise SpiderBedError("refusing to overwrite an existing output: %s" % path)
    path.write_text(text, encoding="utf-8")


def cmd_prepare(args) -> int:
    root, out = Path(args.spider_root), Path(args.out)
    split = load_split(args.split_file)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa                                                 # noqa: PLC0415
        import pyarrow.parquet as pq                                         # noqa: PLC0415
    except ImportError:
        pa = pq = None
    manifest = {"schema": "kit-bed-spider.v1", "data_source": DATA_SOURCE, "spider_root": str(root),
                "split_file": str(Path(args.split_file)), "split_file_sha256": sha256_file(Path(args.split_file)),
                "instruction_sha256": split["instruction_sha256"], "query_timeout_s": QUERY_TIMEOUT_S,
                "row_limit": ROW_LIMIT, "splits": {}}
    used_sources = set()
    for split_name in SPLITS:
        members = load_members(root, split_name, split)
        if args.limit:
            members = members[: args.limit]
        used_sources.update(member["source"] for member in members)
        unexecutable = []
        for member in members:                               # a gold query that does not execute cannot judge anything
            try:
                gold_rows_for(database_path(root, member["db"]), member["gold_sql"])
            except SpiderBedError:
                unexecutable.append(member["id"])
        if unexecutable:
            raise SpiderBedError("%d gold queries in %s do not execute, e.g. %s"
                                 % (len(unexecutable), split_name, ", ".join(unexecutable[:3])))
        rows = rows_for_trainer(members, split_name)
        text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
        _write_new(out / ("%s.jsonl" % split_name), text)
        if pq is not None:
            parquet = out / ("%s.parquet" % split_name)
            if parquet.exists():
                raise SpiderBedError("refusing to overwrite an existing output: %s" % parquet)
            pq.write_table(pa.Table.from_pylist(rows), parquet)
        manifest["splits"][split_name] = {
            "rows": len(rows), "databases": len({member["db"] for member in members}),
            "jsonl_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "parquet": pq is not None}
    manifest["source_sha256"] = {"%s.json" % source: sha256_file(root / ("%s.json" % source))
                                 for source in sorted(used_sources)}
    _write_new(out / "spider.manifest.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print("prepared Spider in %s: %s%s"
          % (out, {name: entry["rows"] for name, entry in manifest["splits"].items()},
             "" if pq else " (jsonl only: pyarrow not installed)"))
    return 0


def score_responses(members, responses, timeout_s: float = QUERY_TIMEOUT_S, spider_root=None,
                    split_file=SPLIT_FILE) -> dict:
    """Score saved responses for one split. Missing, duplicate or foreign ids are refused, never zero."""
    root = spider_root_from_env(spider_root)
    by_id = {}
    for row in responses:
        member_id = str(row.get("id") or row.get("member_id") or "")
        if member_id in by_id:
            raise SpiderBedError("response file has a duplicate answer for %s" % member_id)
        by_id[member_id] = str(row.get("response") or "")
    expected = [member["id"] for member in members]
    missing = [member_id for member_id in expected if member_id not in by_id]
    extra = sorted(set(by_id) - set(expected))
    if missing or extra:
        raise SpiderBedError("response membership failed: %d missing (e.g. %s), %d unexpected (e.g. %s)"
                             % (len(missing), ", ".join(missing[:3]) or "-", len(extra), ", ".join(extra[:3]) or "-"))
    outcomes, correct = {}, 0
    for member in members:
        db_path = _verified_database(str(root), member["db"], str(split_file))
        result = score_response(by_id[member["id"]], member["gold_sql"], db_path, timeout_s=timeout_s)
        outcomes[result["kind"]] = outcomes.get(result["kind"], 0) + 1
        correct += result["correct"]
    return {"n": len(members), "correct": correct, "accuracy": round(correct / len(members), 6),
            "outcomes": dict(sorted(outcomes.items())), "timeout_s": timeout_s}


def cmd_score(args) -> int:
    root = spider_root_from_env(args.spider_root)
    members = load_members(root, args.split, load_split(args.split_file))
    responses = [json.loads(line) for line in Path(args.responses).read_text(encoding="utf-8").split("\n") if line.strip()]
    if args.panel:
        responses = [row for row in responses if row.get("panel") in (None, args.panel)]
    summary = {"split": args.split, "responses": str(args.responses),
               "responses_sha256": sha256_file(Path(args.responses)),
               **score_responses(members, responses, timeout_s=args.timeout, spider_root=root,
                                 split_file=args.split_file)}
    text = json.dumps(summary, indent=1, sort_keys=True) + "\n"
    if args.out:
        _write_new(Path(args.out), text)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Spider 1.0 bed: prepare trainer files, or score saved responses.")
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--spider-root", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--split-file", default=str(SPLIT_FILE))
    prepare.add_argument("--limit", type=int)
    score = sub.add_parser("score")
    score.add_argument("--responses", required=True)
    score.add_argument("--split", default="heldout", choices=list(SPLITS))
    score.add_argument("--spider-root", default=None, help="default: the SPIDER_ROOT environment variable")
    score.add_argument("--split-file", default=str(SPLIT_FILE))
    score.add_argument("--panel", default=None, help="keep only response rows carrying this panel name")
    score.add_argument("--timeout", type=float, default=QUERY_TIMEOUT_S)
    score.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        return {"prepare": cmd_prepare, "score": cmd_score}[args.action](args)
    except SpiderBedError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    sys.exit(main())
