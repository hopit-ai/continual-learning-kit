#!/usr/bin/env python3
"""The repertoire of a set of SQL answers: which constructs the model reaches for, and how far training moved it.

Receipt 238 found that twenty GRPO steps on Spider change HOW the untrained Qwen3-1.7B writes SQL
before they change how often it is right: in every run the model stops writing JOINs (41 of 100
answers before, 0 to 9 after) and nests subqueries instead, and the questions it loses are the ones
whose gold answer joins tables. One run read +5 on the accuracy count with the same collapse. So the
count alone cannot say whether a recipe kept the model's repertoire; this file says it.

    python repertoire.py compare --reference DIR --trained DIR [--trained DIR ...]

Each DIR is a scoring folder: the `responses.jsonl` beside its `bed-score.json` is read, the SQL is
taken out of each answer exactly as kit/beds/spider.py's `extract_sql` takes it (the ```sql fence,
failing that the first SELECT), and every answer counts in the denominator, so an answer with no SQL
uses no feature. Each feature is a case-insensitive regex, the same as scripts/analyse_spider_flips.py
uses, and its figure is the share of answers (percent) whose SQL matches.

    features(responses)          {feature: percent of answers}, or None for no answers
    shift(reference, trained)    per-feature change in points, the JOIN share of each, and
                                 `style_shift_points`: the mean absolute change over the features

A scoring with no responses.jsonl is UNKNOWN, never a refusal, as in kit/density.py: nothing a report
decides on its counts depends on it. Standard library only; nothing here scores or executes SQL.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "kit-repertoire.v1"
#: The features, in the order every table prints them. scripts/analyse_spider_flips.py's set, less its
#: `double-quoted identifiers`, which is quoting, not a construct.
FEATURES = {
    "JOIN": r"\bJOIN\b",
    "subquery": r"\(\s*SELECT\b",
    "WHERE": r"\bWHERE\b",
    "GROUP BY": r"\bGROUP\s+BY\b",
    "HAVING": r"\bHAVING\b",
    "ORDER BY": r"\bORDER\s+BY\b",
    "LIMIT": r"\bLIMIT\b",
    "DISTINCT": r"\bDISTINCT\b",
    "COUNT(": r"\bCOUNT\s*\(",
    "AVG/SUM/MIN/MAX(": r"\b(AVG|SUM|MIN|MAX)\s*\(",
    "LIKE": r"\bLIKE\b",
    "UNION/INTERSECT/EXCEPT": r"\b(UNION|INTERSECT|EXCEPT)\b",
    "alias AS": r"\bAS\b",
    "SELECT *": r"SELECT\s+\*",
}
_COMPILED = {name: re.compile(pattern, re.I) for name, pattern in FEATURES.items()}
# kit/beds/spider.py's own extraction, copied so this file needs nothing beside it;
# tests/test_kit_repertoire.py holds the two to the same answers.
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
_SELECT = re.compile(r"(SELECT\b.*?)(?:;|$)", re.S | re.I)


def extract_sql(text: str):
    """The query in a ```sql fence; failing that, the first SELECT statement. None when neither."""
    match = _FENCE.search(text or "")
    if match and re.search(r"\bselect\b", match.group(1), re.I):
        return match.group(1).strip().rstrip(";").strip()
    match = _SELECT.search(text or "")
    return match.group(1).strip() if match else None


def _answer(item) -> str:
    """One answer's SQL: `item` is a responses.jsonl row (its `response`) or the answer's text."""
    text = item.get("response") if isinstance(item, dict) else item
    return extract_sql(text if isinstance(text, str) else "") or ""


def features(responses) -> dict | None:
    """{feature: percent of the answers whose SQL uses it}, or None when there are no answers."""
    sql = [_answer(item) for item in responses or []]
    if not sql:
        return None
    return {name: round(100.0 * sum(1 for s in sql if pattern.search(s)) / len(sql), 2)
            for name, pattern in _COMPILED.items()}


def shift(reference: dict | None, trained: dict | None) -> dict:
    """How far `trained`'s repertoire moved from `reference`'s, both from features().

    {change: {feature: points}, join_reference, join_trained, style_shift_points}; every number is
    None when either side is unknown."""
    if not reference or not trained:
        return {"change": {name: None for name in FEATURES}, "join_reference": (reference or {}).get("JOIN"),
                "join_trained": (trained or {}).get("JOIN"), "style_shift_points": None}
    change = {name: round(trained[name] - reference[name], 2) for name in FEATURES}
    return {"change": change, "join_reference": reference["JOIN"], "join_trained": trained["JOIN"],
            "style_shift_points": round(sum(abs(v) for v in change.values()) / len(change), 2)}


def read_responses(directory: Path) -> list | None:
    """The answers in `directory`/responses.jsonl (a scoring folder, or its bed-score.json), or None
    when there is no such file. Unreadable lines are skipped, as kit/k3_dose_report.py skips them."""
    directory = Path(directory)
    if directory.is_file():
        directory = directory.parent
    path = directory / "responses.jsonl"
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "response" in row:
                rows.append(row)
    return rows


def of_scoring(directory: Path) -> dict | None:
    """features() of one scoring folder, or None when its answers are unknown."""
    rows = read_responses(directory)
    return features(rows) if rows is not None else None


def show(value, digits: int = 1) -> str:
    """A percent or a point figure, without a trailing `.0`; `-` when unknown."""
    if value is None:
        return "-"
    text = ("%%.%df" % digits) % value
    return text.rstrip("0").rstrip(".") if "." in text else text


# ------------------------------------------------------------------------------------------ the CLI
def compare(reference: Path, trained: list) -> dict:
    """The comparison the CLI prints: one column per scoring, the reference first."""
    ref_rows = read_responses(reference)
    ref = features(ref_rows) if ref_rows is not None else None
    columns = [{"name": Path(reference).name, "path": str(reference), "role": "reference",
                "answers": len(ref_rows) if ref_rows is not None else None, "features": ref, "shift": None}]
    for path in trained:
        rows = read_responses(path)
        mine = features(rows) if rows is not None else None
        columns.append({"name": Path(path).name, "path": str(path), "role": "trained",
                        "answers": len(rows) if rows is not None else None, "features": mine,
                        "shift": shift(ref, mine)})
    return {"schema": SCHEMA, "columns": columns}


def render(result: dict) -> str:
    columns = result["columns"]
    lines = ["| SQL feature (%% of answers) | %s |" % " | ".join(c["name"] for c in columns),
             "|---|" + "---|" * len(columns),
             "| answers read | %s |" % " | ".join("%d" % c["answers"] if c["answers"] is not None else "unknown"
                                                  for c in columns)]
    for name in FEATURES:
        cells = []
        for column in columns:
            if column["features"] is None:
                cells.append("unknown")
                continue
            cell = show(column["features"][name])
            change = (column["shift"] or {}).get("change", {}).get(name)
            if column["role"] == "trained" and change is not None:
                cell += " (%s%s)" % ("+" if change > 0 else "", show(change))
            cells.append(cell)
        lines.append("| %s | %s |" % (name, " | ".join(cells)))
    lines.append("| **style shift** (mean absolute change, points) | %s |" % " | ".join(
        "-" if c["role"] == "reference" else
        ("**%s**" % show(c["shift"]["style_shift_points"]) if c["shift"]["style_shift_points"] is not None
         else "unknown") for c in columns))
    unknown = [c for c in columns if c["features"] is None]
    if unknown:
        lines += [""] + ["- %s: no responses.jsonl beside bed-score.json in %s, so its repertoire is unknown"
                         % (c["name"], c["path"]) for c in unknown]
    return "\n".join(lines) + "\n"


def cmd_compare(args) -> int:
    result = compare(args.reference, args.trained)
    print(render(result), end="")
    if args.out:
        if args.out.exists():
            print("refusing to overwrite %s" % args.out, file=sys.stderr)
            return 2
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote", args.out)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare", help="the repertoire of trained scorings against a reference scoring")
    c.add_argument("--reference", type=Path, required=True, help="the untrained model's scoring folder")
    c.add_argument("--trained", type=Path, action="append", required=True, help="a trained scoring folder")
    c.add_argument("--out", type=Path, default=None, help="also write the comparison as JSON (never overwritten)")
    c.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
