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
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "kit-k3-report.v1"
POINT = re.compile(r"^(?:base|a-seed(?P<aseed>\d+)|b-(?P<arm>[a-z0-9]+)-seed(?P<bseed>\d+))$")
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
HERE = Path(__file__).resolve().parent
BEDS = ("spider", "gsm8k")


class K3ReportError(ValueError):
    """The evidence tree cannot support a report; nothing is written."""


# ------------------------------------------------------- intelligence density (plan 4c, row Q13)
def _load_density():
    """kit/density.py, loaded by path from this file's own directory, the way kit/beds/rewards.py
    loads a bed: nothing here needs the kit installed, on sys.path or in the working directory. A
    copy of the kit without the file still reports -- every density is then unknown, and an unknown
    density is never a refusal."""
    path = HERE / "density.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("kit_density_for_k3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


density = _load_density()
DENSITY_BAR = density.BAR if density else 1.5
NO_TOKENS = ("No scoring under this root carried token counts -- neither `output_tokens_total` in "
             "the result nor a `responses.jsonl` beside it -- so every figure in this table is `-`. "
             "Nothing else in this report depends on them.")


def cost(result, path, *, panel=None):
    """Tokens per correct answer for one scoring, or None when its tokens are not recorded."""
    if density is None or result is None or path is None:
        return None
    try:
        return density.tokens_per_correct(result, path, panel=panel)
    except density.DensityError:
        return None


def compare_cost(trained, untrained) -> dict:
    """{untrained, trained, ratio, bar, verdict} for one cell of the density table."""
    if density is None:
        return {"untrained": None, "trained": None, "ratio": None, "bar": DENSITY_BAR,
                "verdict": "unknown"}
    return density.compare(trained, untrained, bar=DENSITY_BAR)


def show_cost(value, digits: int = 1) -> str:
    return density.show(value, digits) if density else "-"


def latest_found(directory: Path, filename: str) -> dict:
    """{stem: (path, result)} for the highest attempt of every <stem>-aN folder holding `filename`.

    The PATH is carried beside the result because kit/density.py needs it: a scoring's token counts
    may live in the `responses.jsonl` written beside it rather than inside it.
    """
    found: dict = {}
    if not directory.is_dir():
        return found
    for child in sorted(directory.iterdir()):
        match = ATTEMPT.match(child.name)
        if not match or not (child / filename).is_file():
            continue
        stem, attempt = match["stem"], int(match["attempt"])
        if stem not in found or attempt > found[stem][0]:
            found[stem] = (attempt, child / filename,
                           json.loads((child / filename).read_text(encoding="utf-8")))
    return {stem: (path, result) for stem, (_, path, result) in found.items()}


def latest(directory: Path, filename: str) -> dict:
    """{stem: result} for the highest attempt of every <stem>-aN folder holding `filename`."""
    return {stem: result for stem, (_path, result) in latest_found(directory, filename).items()}


def read_tree(root: Path) -> dict:
    """Every scoring under `root`, keyed by point: {'spider': ..., 'gsm8k': ..., 'panels': ...}."""
    return read_tree_with_paths(root)[0]


def read_tree_with_paths(root: Path) -> tuple:
    """(points, paths): the same tree, and the file every result in it was read from.

    The paths live in their own mapping so a point stays exactly what the rest of this file reads --
    results, and nothing else -- while the density can still find the responses.jsonl beside each.
    """
    scores = latest_found(root / "eval", "bed-score.json")
    panels = latest_found(root / "forgetting", "forgetting.json")
    points: dict = {}
    paths: dict = {}
    for stem, (path, result) in scores.items():
        for bed in BEDS:
            if stem.endswith("-" + bed):
                point = stem[: -len(bed) - 1]
                points.setdefault(point, {})[bed] = result
                paths.setdefault(point, {})[bed] = path
    for stem, (path, result) in panels.items():
        points.setdefault(stem, {})["panels"] = result
        paths.setdefault(stem, {})["panels"] = path
    kept = {point: value for point, value in points.items() if POINT.match(point)}
    return kept, {point: paths.get(point, {}) for point in kept}


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


def build_density(points: dict, paths: dict) -> dict:
    """What a correct answer costs, per arm and seed: the learned beds and the general panels, at
    the untrained model and at the end of stage B, with the ratio between them against the bar.

    Plan section 4c and research row Q13. A scoring that carries no token counts is unknown here and
    changes nothing else in this report: the counts every other table is built on do not depend on
    this one.
    """
    base, base_paths = points.get("base", {}), paths.get("base", {})
    panel_names = sorted((base.get("panels") or {}).get("panels") or {})
    untrained = {bed: cost(base.get(bed), base_paths.get(bed)) for bed in BEDS}
    untrained_panels = {name: cost(base.get("panels"), base_paths.get("panels"), panel=name)
                        for name in panel_names}
    arms: dict = {}
    measured = sum(1 for value in list(untrained.values()) + list(untrained_panels.values())
                   if value is not None)
    for point, value in sorted(points.items()):
        match = POINT.match(point)
        if not match["arm"]:
            continue
        arm, seed = match["arm"], int(match["bseed"])
        after_a, after_a_paths = points.get("a-seed%d" % seed, {}), paths.get("a-seed%d" % seed, {})
        row: dict = {"beds": {}, "panels": {}}
        for bed in BEDS:
            after_b = cost(value.get(bed), paths.get(point, {}).get(bed))
            middle = cost(after_a.get(bed), after_a_paths.get(bed)) if bed == "spider" else None
            row["beds"][bed] = {**compare_cost(after_b, untrained[bed]),
                                "after_a": (middle or {}).get("tokens_per_correct")}
            measured += sum(1 for cell in (after_b, middle) if cell is not None)
        for name in panel_names:
            after_b = cost(value.get("panels"), paths.get(point, {}).get("panels"), panel=name)
            row["panels"][name] = compare_cost(after_b, untrained_panels[name])
            measured += 1 if after_b is not None else 0
        arms.setdefault(arm, {"seeds": {}})["seeds"][seed] = row
    for arm, rows in arms.items():
        ordered = [rows["seeds"][seed] for seed in sorted(rows["seeds"])]
        rows["beds"] = {bed: spread([row["beds"][bed]["ratio"] for row in ordered]) for bed in BEDS}
        rows["panels"] = {name: spread([row["panels"][name]["ratio"] for row in ordered])
                          for name in panel_names}
    return {"bar": DENSITY_BAR, "definition": "output tokens over the whole held-out set, divided "
                                              "by the correct answers",
            "measured": measured, "available": int(measured > 0),
            "note": None if measured else NO_TOKENS,
            "untrained": {"beds": {bed: (untrained[bed] or {}).get("tokens_per_correct") for bed in BEDS},
                          "panels": {name: (untrained_panels[name] or {}).get("tokens_per_correct")
                                     for name in panel_names}},
            "arms": {arm: arms[arm] for arm in sorted(arms)}}


def build(root: Path, *, allow_different_machines: bool = False) -> dict:
    points, paths = read_tree_with_paths(root)
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
              "density": build_density(points, paths),
              "machine_ids": machines, "comparable": int(comparable),
              "different_machines_allowed": int(bool(allow_different_machines)),
              "arms_reported": len(arms), "seeds_reported": len(seeds_seen), "points_read": len(points)}
    return report


def density_summary(arm: str, name: str, cell: dict | None) -> str:
    """One mean row: the mean ratio over seeds, its spread, and the bar's verdict on the mean."""
    mean = None if cell is None else cell["mean"]
    verdict = density.verdict(mean, bar=DENSITY_BAR) if density else "unknown"
    return "| **%s** | mean of %d | %s | | | | **%s** (sd %s) | %s |" % (
        arm, 0 if cell is None else cell["n"], name, show_cost(mean, 2),
        "-" if cell is None or cell["sd"] is None else show_cost(cell["sd"], 2), verdict)


def render_density(report: dict) -> list:
    """One table: what a correct answer costs, per arm and seed, against the untrained model."""
    block = report.get("density") or {}
    lines = ["", "## What a correct answer costs (tokens per correct answer)", "",
             "Plan section 4c, research row Q13. Tokens per correct answer is the output tokens over "
             "the whole held-out set divided by the correct answers, so it is a cost per task and not "
             "a length. A trained model may spend at most %.1f times the untrained model's on the bed "
             "it learned and on the general panel; over the bar the gain is reported at cost."
             % block.get("bar", DENSITY_BAR), ""]
    if not block.get("measured"):
        lines += [block.get("note") or NO_TOKENS, ""]
    lines += ["| arm | seed | bed or panel | untrained | after A | after B | ratio | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    for arm in sorted(block.get("arms") or {}):
        rows = block["arms"][arm]
        for seed in sorted(rows["seeds"]):
            row = rows["seeds"][seed]
            for name, cell in list(row["beds"].items()) + list(row["panels"].items()):
                lines.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
                    arm, seed, name, show_cost(cell["untrained"]),
                    show_cost(cell.get("after_a")), show_cost(cell["trained"]),
                    show_cost(cell["ratio"], 2), cell["verdict"]))
        for name, summary in list(rows["beds"].items()) + list(rows["panels"].items()):
            lines.append(density_summary(arm, name, summary))
    return lines


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
    lines += render_density(report)
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
