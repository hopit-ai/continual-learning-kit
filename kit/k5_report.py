#!/usr/bin/env python3
"""The K5 readout: four jobs learned one after another, joined per order, arm and seed.

    python k5_report.py --root $WORK/k5 --out $WORK/k5/report-a1

Reads the folders the campaign wrote under --root, and nothing else:

    eval/<point>-<bed>-aN/bed-score.json          every bed's held-out set, at every point
    forgetting/<point>-aN/forgetting.json         the three general panels, at every point
    plasticity/<point>-aN/plasticity.json         the probe, at every point
    sequences/<order>-<arm>-aN.json               which job each stage learned (kit/sequence.py)
    scorecards/<order>-<arm>-aN/scorecard.json    the four scorecard numbers (kit/scorecard.py)
    plasticity-compare/<order>-<arm>-seed<S>-aN/plasticity.json   plan 4b's bar, per sequence

where a point is `base` (the untrained model, stage 0 of every sequence) or
`<order>-<arm>-seed<SEED>-stage<STAGE>`. The highest attempt of each is used.

WHAT IT PUTS TOGETHER. For every order and arm it reports, per seed and then averaged over seeds:
what each job scored when it was learned; what it scored at the end of the sequence (the difference
is what the later jobs cost it); what every job scored at the end; and the general panels before and
after. The scorecard's own four numbers and the plasticity verdict are carried through beside them,
not recomputed here -- one number has one owner.

THE COMPARISON RULE. Every number here is a count of correct answers from a deterministic scoring,
and counts are only comparable when they were made on the same machine in the same decoding mode:
across machines a panel moves by up to 3 points before any training (receipts 204 and 209). So this
tool collects the machine-and-mode fingerprint of every scoring it reads and REFUSES to write a
report that mixes two of them -- `--allow-different-machines` overrides that, and is recorded in the
report as the departure it is.

A HOLE IS A HOLE. A missing stage, a missing bed or a missing seed is reported as missing and never
filled in from a neighbour. The manifest (`kit/sequence.py manifest`) is what refuses a grid with a
hole in it; this report says which one it found.

Standard library only. Nothing is overwritten: an existing --out is refused.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "kit-k5-report.v1"
#: The same grammar kit/sequence.py builds the manifest from.
POINT = re.compile(r"^(?:base|(?P<order>[a-z0-9]+)-(?P<arm>[a-z0-9]+)-seed(?P<seed>\d+)-stage(?P<stage>\d+))$")
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
COMPARE = re.compile(r"^(?P<order>[a-z0-9]+)-(?P<arm>[a-z0-9]+)-seed(?P<seed>\d+)$")
SHARED_ARM = "shared"                # stage 1 is one run per order and seed, shared by every arm


class K5ReportError(ValueError):
    """The evidence tree cannot support a report; nothing is written."""


# ------------------------------------------------------------------------------------ reading
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
    return {stem: result for stem, (_attempt, result) in found.items()}


def latest_file(directory: Path) -> dict:
    """{stem: json} for the highest attempt of every <stem>-aN.json file in a directory."""
    found: dict = {}
    if not directory.is_dir():
        return found
    for child in sorted(directory.glob("*.json")):
        match = ATTEMPT.match(child.stem)
        if not match:
            continue
        stem, attempt = match["stem"], int(match["attempt"])
        if stem not in found or attempt > found[stem][0]:
            found[stem] = (attempt, json.loads(child.read_text(encoding="utf-8")))
    return {stem: result for stem, (_attempt, result) in found.items()}


def beds_of(scores: dict) -> list:
    """Every bed scored at the untrained model, in a fixed (sorted) order."""
    beds = sorted(stem[len("base-"):] for stem in scores if stem.startswith("base-"))
    if not beds:
        raise K5ReportError("nothing under eval/base-<bed>-aN/bed-score.json: the untrained model is "
                            "stage 0 of every sequence, and no difference can be read without it")
    return beds


def read_tree(root: Path) -> dict:
    """Every scoring under `root`, keyed by point: {'scores': {...}, 'panels': ..., 'plasticity': ...}."""
    scores = latest(root / "eval", "bed-score.json")
    beds = beds_of(scores)
    points: dict = {}
    for stem, result in scores.items():
        for bed in beds:
            if stem.endswith("-" + bed) and POINT.match(stem[: -len(bed) - 1]):
                points.setdefault(stem[: -len(bed) - 1], {}).setdefault("scores", {})[bed] = result
    for stem, result in latest(root / "forgetting", "forgetting.json").items():
        if POINT.match(stem):
            points.setdefault(stem, {})["panels"] = result
    for stem, result in latest(root / "plasticity", "plasticity.json").items():
        if POINT.match(stem):
            points.setdefault(stem, {})["plasticity"] = result
    return {"beds": beds, "points": points}


def fingerprints(points: dict) -> list:
    """Every machine-and-mode fingerprint the SCORINGS carry, as sorted distinct ids (None included)."""
    seen = set()
    for value in points.values():
        for result in list(value.get("scores", {}).values()) + ([value["panels"]] if "panels" in value else []):
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


# ------------------------------------------------------------------------- the sequences
def sequences_from_manifests(root: Path) -> dict:
    """{(order, arm): [job, ...]} from the manifests kit/sequence.py wrote, which name the stages."""
    found: dict = {}
    for stem, manifest in latest_file(root / "sequences").items():
        source = manifest.get("source") or {}
        order, arm = source.get("order"), source.get("arm")
        jobs = manifest.get("jobs")
        if not (isinstance(order, str) and isinstance(arm, str) and isinstance(jobs, list) and jobs):
            raise K5ReportError("%s/sequences/%s-aN.json names no order, arm and jobs: it is not a "
                                "manifest kit/sequence.py wrote" % (root, stem))
        found[(order, arm)] = list(jobs)
    return found


def point_name(order: str, arm: str, seed: int, stage: int) -> str:
    return "%s-%s-seed%d-stage%d" % (order, SHARED_ARM if stage == 1 else arm, seed, stage)


def seeds_in_tree(points: dict, order: str, arm: str) -> list:
    seeds = set()
    for name in points:
        match = POINT.match(name)
        if match and match["order"] == order and match["arm"] in (arm, SHARED_ARM):
            seeds.add(int(match["seed"]))
    return sorted(seeds)


def read_sequence(points: dict, beds: list, order: str, arm: str, jobs: list, seed: int) -> dict:
    """One seed's sequence: every bed after every stage, the panels, the probe, and what moved."""
    stages, missing = [], []
    for stage in range(0, len(jobs) + 1):
        name = "base" if stage == 0 else point_name(order, arm, seed, stage)
        found = points.get(name, {})
        if not found:
            missing.append(name)
        stages.append({"stage": stage, "point": name, "job": None if stage == 0 else jobs[stage - 1],
                       "scores": {bed: number(found.get("scores", {}).get(bed)) for bed in beds},
                       "accuracy": {bed: number(found.get("scores", {}).get(bed), "accuracy") for bed in beds},
                       "panels": panel_scores(found.get("panels")),
                       "dormant_share": number(found.get("plasticity"), "dormant_share"),
                       "effective_rank": number(found.get("plasticity"), "effective_rank")})
    last = len(jobs)

    def cell(stage: int, bed: str):
        return stages[stage]["scores"].get(bed)

    def difference(after, before):
        return None if after is None or before is None else after - before
    learned = {jobs[i - 1]: difference(cell(i, jobs[i - 1]), cell(i - 1, jobs[i - 1]))
               for i in range(1, last + 1)}
    kept = {jobs[i - 1]: difference(cell(last, jobs[i - 1]), cell(i, jobs[i - 1]))
            for i in range(1, last)}
    panels_before, panels_after = stages[0]["panels"], stages[last]["panels"]
    general = {panel: difference(panels_after.get(panel), panels_before.get(panel))
               for panel in sorted(set(panels_before) | set(panels_after))}
    return {"seed": seed, "stages": stages, "learned": learned, "kept": kept,
            "final": {bed: cell(last, bed) for bed in beds}, "general_delta": general,
            "missing_points": missing}


# ------------------------------------------------------------------------------------ the report
def build(root: Path, *, allow_different_machines: bool = False) -> dict:
    root = Path(root)
    tree = read_tree(root)
    points, beds = tree["points"], tree["beds"]
    if not points:
        raise K5ReportError("no scorings under %s: expected eval/<point>-<bed>-aN/bed-score.json and "
                            "forgetting/<point>-aN/forgetting.json" % root)
    machines = fingerprints(points)
    comparable = len(machines) == 1 and machines[0] is not None
    if not comparable and not allow_different_machines:
        raise K5ReportError(
            "the scorings under %s carry %d different machine-and-mode fingerprints (%s). A count from "
            "one machine cannot be subtracted from a count on another: a panel moves by up to 3 points "
            "across machines before any training. Re-score on one machine, or pass "
            "--allow-different-machines and say so in the readout."
            % (root, len(machines), ", ".join(str(m) for m in machines)))
    sequences = sequences_from_manifests(root)
    if not sequences:
        raise K5ReportError("no sequence manifests under %s/sequences: the report reads which job each "
                            "stage learned from the manifests kit/sequence.py wrote, and will not guess "
                            "it from the scores" % root)
    scorecards = latest(root / "scorecards", "scorecard.json")
    compares = latest(root / "plasticity-compare", "plasticity.json")
    orders: dict = {}
    seeds_seen, arms_seen = set(), set()
    for (order, arm), jobs in sorted(sequences.items()):
        entry = orders.setdefault(order, {"jobs": jobs, "arms": {}})
        if entry["jobs"] != jobs:
            raise K5ReportError("order %s learns %s for arm %s and %s for another: one order is one "
                                "sequence" % (order, " then ".join(jobs), arm, " then ".join(entry["jobs"])))
        by_seed = {}
        for seed in seeds_in_tree(points, order, arm):
            by_seed[seed] = read_sequence(points, beds, order, arm, jobs, seed)
            seeds_seen.add(seed)
        arms_seen.add(arm)
        rows = [by_seed[seed] for seed in sorted(by_seed)]
        card = scorecards.get("%s-%s" % (order, arm))
        plasticity = {}
        for seed in sorted(by_seed):
            stem = "%s-%s-seed%d" % (order, arm, seed)
            if stem in compares:
                plasticity[str(seed)] = {key: compares[stem].get(key) for key in
                                         ("verdict", "plasticity_loss_present", "learning_half_met",
                                          "internal_half_met", "comparable")}
        entry["arms"][arm] = {
            "seeds": {str(seed): by_seed[seed] for seed in sorted(by_seed)},
            "learned": {job: spread([row["learned"].get(job) for row in rows]) for job in jobs},
            "kept": {job: spread([row["kept"].get(job) for row in rows]) for job in jobs[:-1]},
            "final": {bed: spread([row["final"].get(bed) for row in rows]) for bed in beds},
            "general_delta": {panel: spread([row["general_delta"].get(panel) for row in rows])
                              for panel in sorted({p for row in rows for p in row["general_delta"]})},
            "scorecard": None if card is None else {key: card.get(key) for key in
                                                    ("average", "seeds", "jobs", "key", "comparable",
                                                     "cells_read", "job_questions")},
            "plasticity": plasticity,
            "missing_points": sorted({name for row in rows for name in row["missing_points"]}),
            "seeds_reported": len(rows)}
    return {"schema": SCHEMA, "root": str(root.resolve()), "beds": beds,
            "base": {"scores": {bed: number(points.get("base", {}).get("scores", {}).get(bed))
                                for bed in beds},
                     "panels": panel_scores(points.get("base", {}).get("panels"))},
            "orders": orders, "machine_ids": machines, "comparable": int(comparable),
            "different_machines_allowed": int(bool(allow_different_machines)),
            "orders_reported": len(orders), "arms_reported": len(arms_seen),
            "seeds_reported": len(seeds_seen), "points_read": len(points),
            "scorecards_read": len(scorecards), "plasticity_compares_read": len(compares)}


def show(value, digits: int = 0, *, signed: bool = True) -> str:
    if value is None:
        return "-"
    if digits == 0:
        return "%+d" % value if signed else "%d" % value
    return "%+.*f" % (digits, value) if signed else "%.*f" % (digits, value)


def render(report: dict) -> str:
    beds = report["beds"]
    lines = ["# K5: four jobs learned one after another", "",
             "Every number is a count of correct answers on a held-out set, scored deterministically "
             "on one machine after every stage. The beds are %s." % ", ".join(beds), ""]
    if not report["comparable"]:
        lines += ["**The scorings do not share one machine-and-mode fingerprint (%s), so these "
                  "differences are not comparable.**" % ", ".join(str(m) for m in report["machine_ids"]), ""]
    lines += ["Untrained: %s." % ", ".join("%s %s" % (bed, show(report["base"]["scores"][bed], signed=False))
                                           for bed in beds), ""]
    for order, entry in sorted(report["orders"].items()):
        lines += ["## %s: %s" % (order, " then ".join(entry["jobs"])), ""]
        for arm, arm_entry in sorted(entry["arms"].items()):
            lines += ["### arm `%s` (%d seeds)" % (arm, arm_entry["seeds_reported"]), ""]
            if arm_entry["missing_points"]:
                lines += ["**%d points of this sequence were not found and are left empty: %s**"
                          % (len(arm_entry["missing_points"]), ", ".join(arm_entry["missing_points"][:6])), ""]
            lines += ["| after stage | learned | " + " | ".join(beds) + " | panels |",
                      "|---|---|" + "---|" * (len(beds) + 1)]
            seeds = sorted(arm_entry["seeds"], key=int)
            first = arm_entry["seeds"][seeds[0]] if seeds else None
            for stage in range(len(entry["jobs"]) + 1):
                cells = []
                for bed in beds:
                    values = [arm_entry["seeds"][seed]["stages"][stage]["scores"].get(bed) for seed in seeds]
                    summary = spread(values)
                    cells.append("-" if summary["mean"] is None else
                                 ("%.1f" % summary["mean"] if summary["sd"] is None else
                                  "%.1f (sd %.1f)" % (summary["mean"], summary["sd"])))
                panels = [arm_entry["seeds"][seed]["stages"][stage]["panels"] for seed in seeds]
                total = [sum(p.values()) for p in panels if p]
                job = "-" if stage == 0 else entry["jobs"][stage - 1]
                lines.append("| %d | %s | %s | %s |"
                             % (stage, job, " | ".join(cells),
                                "-" if not total else "%.1f" % statistics.fmean(total)))
            lines += ["", "| job | learned when it was its turn | kept to the end of the sequence |",
                      "|---|---|---|"]
            for job in entry["jobs"]:
                learned, kept = arm_entry["learned"][job], arm_entry["kept"].get(job)
                lines.append("| %s | %s (sd %s) | %s |"
                             % (job, show(learned["mean"], 1),
                                "-" if learned["sd"] is None else "%.1f" % learned["sd"],
                                "-" if kept is None or kept["mean"] is None else
                                "%s (sd %s)" % (show(kept["mean"], 1),
                                                "-" if kept["sd"] is None else "%.1f" % kept["sd"])))
            general = arm_entry["general_delta"]
            if general:
                lines += ["", "General panels, end of the sequence minus the untrained model: %s."
                          % ", ".join("%s %s (sd %s)" % (panel, show(general[panel]["mean"], 1),
                                                         "-" if general[panel]["sd"] is None else
                                                         "%.1f" % general[panel]["sd"])
                                      for panel in sorted(general))]
            card = arm_entry["scorecard"]
            if card and card.get("average"):
                average = card["average"]
                lines += ["", "Scorecard (`%s`, %d cells): average accuracy %s, backward transfer %s, "
                          "forward transfer %s."
                          % (card.get("key"), card.get("cells_read") or 0,
                             show((average.get("average_accuracy") or {}).get("mean"), 4, signed=False),
                             show((average.get("backward_transfer") or {}).get("mean"), 4),
                             show((average.get("forward_transfer") or {}).get("mean"), 4))]
            else:
                lines += ["", "No scorecard for this arm yet (kit/scorecard.py has not run over its manifest)."]
            if arm_entry["plasticity"]:
                verdicts = ", ".join("seed %s %s" % (seed, value.get("verdict"))
                                     for seed, value in sorted(arm_entry["plasticity"].items()))
                lines += ["", "Plasticity (plan 4b's bar, measured not gated): %s." % verdicts]
            lines.append("")
            if first is not None:
                lines += ["Per seed, %s at the end of the sequence: %s."
                          % (entry["jobs"][0],
                             ", ".join("seed %s %s" % (seed,
                                                       show(arm_entry["seeds"][seed]["final"].get(entry["jobs"][0]),
                                                            signed=False))
                                       for seed in seeds)), ""]
    lines += ["%d orders, %d arms, %d seeds, %d scored points, %d scorecards, %d plasticity comparisons."
              % (report["orders_reported"], report["arms_reported"], report["seeds_reported"],
                 report["points_read"], report["scorecards_read"], report["plasticity_compares_read"]), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The K5 readout, per order, arm and seed.")
    parser.add_argument("--root", required=True, help="the campaign's k5 directory")
    parser.add_argument("--out", required=True, help="a new directory; never overwritten")
    parser.add_argument("--allow-different-machines", action="store_true",
                        help="report anyway; recorded in the report as a departure")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % out)
    try:
        report = build(Path(args.root), allow_different_machines=args.allow_different_machines)
    except K5ReportError as exc:
        raise SystemExit(str(exc))
    out.mkdir(parents=True)
    (out / "k5-report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    text = render(report)
    (out / "k5-report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
