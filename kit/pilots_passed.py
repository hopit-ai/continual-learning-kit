#!/usr/bin/env python3
"""Did the original campaign's pilots pass in THIS work directory?

    python pilots_passed.py --campaign campaigns/k3-replay.yaml --out $WORK/k3/pilots-a1

A follow-up campaign adds seeds to a campaign that has already run here. It must never re-run that
campaign's pilots and never train a pilot of its own, so its first row asks the only question that is
left: did every pilot of the ORIGINAL campaign pass in this same WORK? This file answers it from the
runner's own state -- {work}/campaign/<name>/<pilot row>/attempt-N/verdict.json, latest attempt --
read through kit/runner.py itself, so the paths and the verdict format cannot drift from the runner
that wrote them.

It writes `<out>/pilots-passed.json`, every count a number:

    expected_count   how many rows the original campaign marks `pilot: true`
    passed_count     how many of them have a latest verdict of PASS in this WORK
    failed_count     how many ran here and did not pass
    missing_count    how many never ran here

and ALWAYS exits 0, including when nothing passed and including when the campaign file cannot be
read: the follow-up's own bars judge this row. A refusal then comes from a bar written in advance
rather than from an exit code, and the numbers that produced it are on disk either way.

Standard library plus PyYAML, the same as the runner.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SCHEMA = "kit-pilots-passed.v1"


def _runner():
    """kit/runner.py, beside this file: its paths and its verdict format are the ones we must read."""
    spec = importlib.util.spec_from_file_location("kit_runner_for_pilots",
                                                  Path(__file__).resolve().parent / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def look(campaign_path: Path) -> dict:
    """Every pilot of `campaign_path`, with the latest verdict the runner left for it in this WORK."""
    payload = {"schema": SCHEMA, "campaign_file": str(campaign_path), "campaign": None, "state_dir": None,
               "expected_count": 0, "passed_count": 0, "failed_count": 0, "missing_count": 0,
               "pilots": [], "reason": None}
    try:
        runner = _runner()
        campaign = runner.load_campaign(campaign_path)
        pilots = [row["id"] for row in campaign["rows"] if row.get("pilot")]
        payload["campaign"] = campaign["name"]
        payload["state_dir"] = str(runner.work_root(campaign) / "campaign" / campaign["name"])
    except Exception as exc:        # noqa: BLE001 -- a campaign we cannot read is "nothing passed",
        payload["reason"] = "%s: %s" % (type(exc).__name__, exc)   # and the bar, not a traceback, says so
        return payload
    payload["expected_count"] = len(pilots)
    for row_id in pilots:
        verdict = runner.latest_verdict(campaign, row_id)
        payload["pilots"].append({"row": row_id, "verdict": (verdict or {}).get("verdict") or "NOT_RUN",
                                  "attempt": (verdict or {}).get("attempt"),
                                  "reason": (verdict or {}).get("reason")})
        if verdict is None:
            payload["missing_count"] += 1
        elif verdict["verdict"] == "PASS":
            payload["passed_count"] += 1
        else:
            payload["failed_count"] += 1
    if payload["passed_count"] != payload["expected_count"]:
        payload["reason"] = ("%d of %d pilots of %s passed in this WORK: %d failed, %d never ran. Run the "
                             "original campaign here first; this follow-up adds seeds to it, it does not "
                             "replace it." % (payload["passed_count"], payload["expected_count"],
                                              payload["campaign"], payload["failed_count"],
                                              payload["missing_count"]))
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Did the original campaign's pilots pass in this WORK?")
    parser.add_argument("--campaign", type=Path, required=True,
                        help="the ORIGINAL campaign file, whose pilots gate this follow-up")
    parser.add_argument("--out", type=Path, required=True, help="a directory; pilots-passed.json is written in it")
    args = parser.parse_args(argv)
    payload = look(args.campaign)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pilots-passed.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    print("%d of %d pilots of %s passed in %s"
          % (payload["passed_count"], payload["expected_count"], payload["campaign"] or args.campaign,
             payload["state_dir"] or "an unreadable state directory"))
    for pilot in payload["pilots"]:
        print("  %-24s %s%s" % (pilot["row"], pilot["verdict"], " (%s)" % pilot["reason"] if pilot["reason"] else ""))
    if payload["reason"]:
        print(payload["reason"])
    return 0                    # ALWAYS: this row is judged by its bars, never by an exit code


if __name__ == "__main__":
    sys.exit(main())
