#!/usr/bin/env python3
"""Subtract one score from another and write the difference as a number a campaign bar can read.

    python delta.py --a /work/k3/base-spider-a1 --b /work/k3/stage-a-spider-a1 --key correct \
                    --out /work/k3/deltas/learned-sql.json

The K3 pilot's two bars are both differences: job A must IMPROVE by at least 5 when it is learned,
and must then FALL by at least 5 when job B is learned without protection. `kit/runner.py` gates on
numbers it finds in a file, so the difference has to BE a number in a file:

    bars:
      - {name: sql-improved, source: "{work}/k3/deltas/learned-sql.json", key: delta, min: 5}
      - {name: same-machine, source: "{work}/k3/deltas/learned-sql.json", key: same_machine_flag, min: 1}

and damage is the same file with `max: -5`. `same_machine_flag` is 1 only when both scorings carry
the same machine-and-mode fingerprint; gate on it, because a difference between two machines is
worth up to 3 points of noise on a 100-question panel before any training has happened (LEARNINGS
37). It is 0 when either side has no fingerprint at all -- a `score` re-scoring of saved answers has
none, and two of those are comparable only because the answers they read were generated once.

`--a` and `--b` are each either a JSON file or a directory holding `bed-score.json` (kit/eval_bed.py)
or `forgetting.json` (kit/score_forgetting.py). The key must be a number in both: a missing key is a
refusal, never a zero. Standard library only. Nothing is overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "kit-delta.v1"
RESULT_FILES = ("bed-score.json", "forgetting.json")


class DeltaError(ValueError):
    """A side is missing, unreadable, or does not carry the key; no difference is written."""


def result_path(where) -> Path:
    """The JSON file a side names: the file itself, or the one result file in the directory."""
    path = Path(where)
    if path.is_file():
        return path
    if path.is_dir():
        for name in RESULT_FILES:
            if (path / name).is_file():
                return path / name
        raise DeltaError("%s holds none of %s" % (path, ", ".join(RESULT_FILES)))
    raise DeltaError("no such result file or directory: %s" % path)


def read_value(where, key: str) -> tuple:
    """(value, machine_id, path): the key as a number, and the fingerprint of the scoring that wrote it."""
    path = result_path(where)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DeltaError("%s is not JSON: %s" % (path, exc)) from exc
    if not isinstance(result, dict):
        raise DeltaError("%s is not a JSON object, so it holds no keys to subtract" % path)
    if key not in result:
        raise DeltaError("%s has no top-level key %r (it has %s)"
                         % (path, key, ", ".join(sorted(result)[:12])))
    value = result[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeltaError("%s[%r] is %r, which is not a number: a bar can only gate on a number"
                         % (path, key, value))
    return float(value), (result.get("machine") or {}).get("id"), path


def delta(a, b, key: str) -> dict:
    a_value, a_machine, a_path = read_value(a, key)
    b_value, b_machine, b_path = read_value(b, key)
    same = a_machine is not None and a_machine == b_machine
    return {"schema": SCHEMA, "key": key, "a": a_value, "b": b_value, "delta": b_value - a_value,
            "same_machine_flag": int(same), "same_machine": same,
            "machine_ids": {"a": a_machine, "b": b_machine},
            "sources": {"a": str(a_path.resolve()), "b": str(b_path.resolve())}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="b - a for one key of two scorings, as a number a bar can read.")
    parser.add_argument("--a", required=True, help="the earlier scoring: a JSON file or its directory")
    parser.add_argument("--b", required=True, help="the later scoring: a JSON file or its directory")
    parser.add_argument("--key", default="correct", help="the key to subtract (default: correct)")
    parser.add_argument("--out", required=True, help="the JSON file to write; never overwritten")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % out)
    try:
        result = delta(args.a, args.b, args.key)
    except DeltaError as exc:
        raise SystemExit(str(exc))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("%s: %g - %g = %+g%s" % (args.key, result["b"], result["a"], result["delta"],
                                   "" if result["same_machine"] else
                                   "  (NOT the same machine-and-mode fingerprint: %s vs %s)"
                                   % (result["machine_ids"]["a"], result["machine_ids"]["b"])))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
