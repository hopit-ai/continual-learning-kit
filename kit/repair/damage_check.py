#!/usr/bin/env python3
"""How much general ability did a model lose against its original? Writes one small JSON a campaign bar can read.

    python damage_check.py --original DIR --before DIR --out damage.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Total points lost between two scorings of the forgetting panels.")
    parser.add_argument("--original", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    original, before = (json.loads((Path(d) / "forgetting.json").read_text()) for d in (args.original, args.before))
    same = (original.get("machine") or {}).get("id") is not None and (original.get("machine") or {}).get("id") == (before.get("machine") or {}).get("id")
    result = {"schema": "kit-damage-check.v1", "original_total": original["total_correct"], "before_total": before["total_correct"],
              "damage_total": original["total_correct"] - before["total_correct"], "same_machine_flag": int(same),
              "median_output_chars_original": {p: v["median_output_chars"] for p, v in original["panels"].items()},
              "median_output_chars_before": {p: v["median_output_chars"] for p, v in before["panels"].items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
