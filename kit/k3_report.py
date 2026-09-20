#!/usr/bin/env python3
"""The K3 readout: what stage B did to the old job, per arm, across seeds.

    python k3_report.py --root $WORK/k3 --out $WORK/k3/report-a1

Reads the folders the campaign wrote under --root:

    eval/<point>-spider-aN/bed-score.json     the old job (Spider held-out 100)
    eval/<point>-gsm8k-aN/bed-score.json      the new job (GSM8K held-out 300)
    forgetting/<point>-aN/forgetting.json     the three general panels

where a point is `base`, `a-seed<S>` (after stage A) or `b-<arm>-seed<S>` (after stage B). The
highest attempt of each is used. For every arm and seed it reports the old job after A and after B,
the new job after B, and the panels; and per arm the mean and sample standard deviation over seeds
of (old job after B - old job after A) and of the new job.

THE COMPARISON RULE. Every number here is a count of correct answers from a deterministic scoring,
and counts are only comparable when they were made on the same machine in the same decoding mode:
across machines a panel moves by up to 3 points before any training (receipts 204 and 209). So this
tool collects the machine-and-mode fingerprint of every scoring it reads and REFUSES to write a
report that mixes two of them -- `--allow-different-machines` overrides that, and is recorded in the
report as the departure it is.

Standard library only. Nothing is overwritten: an existing --out is refused.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "kit-k3-report.v1"
POINT = re.compile(r"^(?:base|a-seed(?P<aseed>\d+)|b-(?P<arm>[a-z0-9]+)-seed(?P<bseed>\d+))$")
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")


class K3ReportError(ValueError):
    """The evidence tree cannot support a report; nothing is written."""


def latest(directory: Path, filename: str) -> dict:
    """{stem: result} for the highest attempt of every <stem>-aN folder holding `filename`."""
    found: dict = {}
    if not directory.is_dir():
        return found
    for child in sorted(directory.iterdir()):
        match = ATTEMPT.match(child.name)
        if not match or not (child / filename).is_file():
            continue
        stem, attempt = match["stem"], int(match["attempt"])
        if stem not in found or attempt > found[stem][0]:
            found[stem] = (attempt, json.loads((child / filename).read_text(encoding="utf-8")))
    return {stem: result for stem, (_, result) in found.items()}


def read_tree(root: Path) -> dict:
    """Every scoring under `root`, keyed by point: {'spider': ..., 'gsm8k': ..., 'panels': ...}."""
    scores = latest(root / "eval", "bed-score.json")
    panels = latest(root / "forgetting", "forgetting.json")
    points: dict = {}
    for stem, result in scores.items():
        for bed in ("spider", "gsm8k"):
            if stem.endswith("-" + bed):
                points.setdefault(stem[: -len(bed) - 1], {})[bed] = result
    for stem, result in panels.items():
        points.setdefault(stem, {})["panels"] = result
    return {point: value for point, value in points.items() if POINT.match(point)}


def fingerprints(points: dict) -> list:
    """Every machine-and-mode fingerprint the scorings carry, as sorted distinct ids (None included)."""
    seen = set()
    for value in points.values():
        for result in value.values():
            seen.add((result.get("machine") or {}).get("id"))
    return sorted(seen, key=lambda item: (item is None, item))


def number(result: dict | None, key: str = "correct"):
    if result is None:
        return None
    value = result.get(key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def panel_scores(result: dict | None) -> dict:
    if not result:
        return {}
    return {name: block["correct"] for name, block in sorted((result.get("panels") or {}).items())}


def spread(values: list) -> dict:
    """Mean and SAMPLE standard deviation; the spread of one seed is not a number, so it is null."""
    clean = [v for v in values if v is not None]
    return {"n": len(clean), "mean": round(statistics.fmean(clean), 4) if clean else None,
            "sd": round(statistics.stdev(clean), 4) if len(clean) > 1 else None}


def build(root: Path, *, allow_different_machines: bool = False) -> dict:
    points = read_tree(root)
    if not points:
        raise K3ReportError("no scorings under %s: expected eval/<point>-<bed>-aN/bed-score.json and "
                            "forgetting/<point>-aN/forgetting.json" % root)
    machines = fingerprints(points)
    comparable = len(machines) == 1 and machines[0] is not None
    if not comparable and not allow_different_machines:
        raise K3ReportError(
            "the scorings under %s carry %d different machine-and-mode fingerprints (%s). A count from "
            "one machine cannot be subtracted from a count on another: a panel moves by up to 3 points "
            "across machines before any training. Re-score on one machine, or pass "
            "--allow-different-machines and say so in the readout."
            % (root, len(machines), ", ".join(str(m) for m in machines)))
    arms, seeds_seen = {}, set()
    for point in points:
        match = POINT.match(point)
        if not match["arm"]:
            continue
        arm, seed = match["arm"], int(match["bseed"])
        seeds_seen.add(seed)
        after_a = points.get("a-seed%d" % seed, {})
        rows = arms.setdefault(arm, {"seeds": {}})
        rows["seeds"][seed] = {
            "job_a_after_a": number(after_a.get("spider")),
            "job_a_after_b": number(points[point].get("spider")),
            "job_b_after_b": number(points[point].get("gsm8k")),
            "job_b_after_a": number(after_a.get("gsm8k")),
            "panels_after_b": panel_scores(points[point].get("panels")),
        }
        entry = rows["seeds"][seed]
        entry["job_a_change"] = (None if entry["job_a_after_a"] is None or entry["job_a_after_b"] is None
                                 else entry["job_a_after_b"] - entry["job_a_after_a"])
    for arm, rows in arms.items():
        ordered = [rows["seeds"][seed] for seed in sorted(rows["seeds"])]
        rows["job_a_change"] = spread([row["job_a_change"] for row in ordered])
        rows["job_b"] = spread([row["job_b_after_b"] for row in ordered])
    base = points.get("base", {})
    report = {"schema": SCHEMA, "root": str(root.resolve()),
              "base": {"job_a": number(base.get("spider")), "job_b": number(base.get("gsm8k")),
                       "panels": panel_scores(base.get("panels"))},
              "after_stage_a": {point: {"job_a": number(value.get("spider")), "job_b": number(value.get("gsm8k")),
                                        "panels": panel_scores(value.get("panels"))}
                                for point, value in sorted(points.items()) if point.startswith("a-seed")},
              "arms": {arm: arms[arm] for arm in sorted(arms)},
              "machine_ids": machines, "comparable": int(comparable),
              "different_machines_allowed": int(bool(allow_different_machines)),
              "arms_reported": len(arms), "seeds_reported": len(seeds_seen), "points_read": len(points)}
    return report


def render(report: dict) -> str:
    arms = report["arms"]
    panels = sorted({name for arm in arms.values() for row in arm["seeds"].values() for name in row["panels_after_b"]})
    lines = ["# K3: does rehearsing the old job's questions protect it?", "",
             "The old job is Spider text-to-SQL (held-out 100), the new job is GSM8K maths (held-out 300). "
             "Every number is a count of correct answers, scored deterministically on one machine.", ""]
    if not report["comparable"]:
        lines += ["**The scorings do not share one machine-and-mode fingerprint (%s), so these differences are "
                  "not comparable.**" % ", ".join(str(m) for m in report["machine_ids"]), ""]
    lines += ["Untrained: old job %s, new job %s." % (report["base"]["job_a"], report["base"]["job_b"]), "",
              "| arm | seeds | old job after A | old job after B | change | new job after B |",
              "|---|---|---|---|---|---|"]
    for arm in sorted(arms):
        rows = arms[arm]
        for seed in sorted(rows["seeds"]):
            row = rows["seeds"][seed]
            lines.append("| %s | %d | %s | %s | %s | %s |"
                         % (arm, seed, row["job_a_after_a"], row["job_a_after_b"],
                            "%+d" % row["job_a_change"] if row["job_a_change"] is not None else "-",
                            row["job_b_after_b"]))
        change, job_b = rows["job_a_change"], rows["job_b"]
        lines.append("| **%s** | mean of %d | | | **%s** (sd %s) | **%s** (sd %s) |"
                     % (arm, change["n"], change["mean"], change["sd"], job_b["mean"], job_b["sd"]))
    if panels:
        lines += ["", "General panels after stage B (a trained model may lose at most 3 per 100 on each):", "",
                  "| arm | seed | " + " | ".join(panels) + " |", "|---|---|" + "---|" * len(panels)]
        for arm in sorted(arms):
            for seed in sorted(arms[arm]["seeds"]):
                row = arms[arm]["seeds"][seed]
                lines.append("| %s | %d | %s |" % (arm, seed, " | ".join(str(row["panels_after_b"].get(p, "-"))
                                                                        for p in panels)))
    lines += ["", "Arms reported: %d over %d seeds, from %d scored points."
              % (report["arms_reported"], report["seeds_reported"], report["points_read"]), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The K3 readout, per arm and seed.")
    parser.add_argument("--root", required=True, help="the campaign's k3 directory")
    parser.add_argument("--out", required=True, help="a new directory; never overwritten")
    parser.add_argument("--allow-different-machines", action="store_true",
                        help="report anyway; recorded in the report as a departure")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % out)
    try:
        report = build(Path(args.root), allow_different_machines=args.allow_different_machines)
    except K3ReportError as exc:
        raise SystemExit(str(exc))
    out.mkdir(parents=True)
    (out / "k3-report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    text = render(report)
    (out / "k3-report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
