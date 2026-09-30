#!/usr/bin/env python3
"""The K2b readout: does the route of damage (GRPO or supervised fine-tuning) decide whether an ability comes back?

    python k2b_report.py --root $WORK/k2b --out $WORK/k2b/report-a1

Reads the folders kit/campaigns/k2b-route.yaml writes under --root:

    forgetting/<subject>-<stage>-aN/forgetting.json   the 300-question panel; stages original, before, after<N>, control300
    forgetting/<subject>-drift-<point>.json            weight distance from the original (kit/repair/drift.py)
    eval/<point>-spider-aN/bed-score.json              Spider held-out 100, the old ability at stake
    eval/<point>-gsm8k-aN/bed-score.json               GSM8K held-out 300, the new job

Subjects are `rl-seed<S>` and `sft-seed<S>`. The healthy original is the untrained Qwen3-1.7B, scored as `q17`,
and the control is the same repair applied to it, `q17-control300`; both are shared by every subject.

First it runs kit/repair/report.py, unchanged and loaded by path, on forgetting/ with an explicit `--alias
<subject>=q17` for every subject (report.py's own campaign reader matches no hyphenated subject name), and keeps
its report under <out>/recovery/. Then, per subject, it computes the same shares on Spider held-out, by the
same rule: recovered share = (after - before) / (original - before), and against the control = (after - before)
/ (control - before), each only where that loss is at least 5 points and only from scorings with the
original's machine-and-mode fingerprint. Per route it gives the mean, the sample standard deviation, the
minimum and the maximum over the repeats of: Spider damage, panel damage (total and per panel), GSM8K gain, the
recovered shares against the original and the control at every repair step, the relative weight distance, and
the density (kit/density.py: tokens per correct answer against the untrained model's).

THE VERDICT, fixed in docs/phase2/k2b-design-20260929.md before any number existed: the claim "the route
decides" HOLDS if the mean share recovered against the control by step 300 is at least 80 percent for RL and
below 80 percent for SFT, and the RL mean is at least 20 points above the SFT mean. The spread is printed
beside it. The rule is applied to Spider held-out, the ability the pilot proves damaged, and that is the
verdict of record; the same rule on the panel total is printed beside it and does not replace it (GSM8K
training feeds the panel's maths third directly, so the panel can gain while Spider falls).

A flag, not a gate: if one route's mean GSM8K gain is not positive, or is more than twice the other's, the
routes did not learn comparable amounts of maths and the comparison is marked as such.

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

SCHEMA = "kit-k2b-report.v1"
HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "campaigns" / "k2b-route.yaml"
ORIGINAL = "q17"
CONTROL = "control300"
ROUTES = ("rl", "sft")
SUBJECT = re.compile(r"^(?P<route>rl|sft)-seed(?P<seed>\d+)$")
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
STEPS = (50, 100, 200, 300)
FINAL = "after300"
BAR, GAP = 0.80, 0.20
LEARNED_RATIO = 2.0


class K2bReportError(ValueError):
    """The evidence tree cannot support a report; nothing is written."""


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repair = _load("kit_repair_report_for_k2b", HERE / "repair" / "report.py")
MIN_DAMAGE = repair.MIN_DAMAGE
density = _load("kit_density_for_k2b", HERE / "density.py") if (HERE / "density.py").is_file() else None


def latest_found(directory: Path, filename: str) -> dict:
    """{stem: (path, result)} for the highest attempt of every <stem>-aN folder holding `filename`."""
    found: dict = {}
    if not directory.is_dir():
        return {}
    for child in sorted(directory.iterdir()):
        match = ATTEMPT.match(child.name)
        if match and (child / filename).is_file():
            stem, attempt = match["stem"], int(match["attempt"])
            if stem not in found or attempt > found[stem][0]:
                found[stem] = (attempt, child / filename, json.loads((child / filename).read_text(encoding="utf-8")))
    return {stem: (path, result) for stem, (_, path, result) in found.items()}


def machine(result) -> str | None:
    return ((result or {}).get("machine") or {}).get("id")


def cost(found):
    """Tokens per correct answer for one scoring, or None when its tokens are not recorded."""
    if density is None or not found:
        return None
    try:
        return density.tokens_per_correct(found[1], Path(found[0]))
    except density.DensityError:
        return None


def summary(values: list) -> dict:
    values = [v for v in values if v is not None]
    return {"n": len(values), "mean": round(statistics.fmean(values), 4) if values else None,
            "sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def shares(scores: dict, comparable: dict) -> dict:
    """The recovered shares of one measure for one subject: {stage: {vs_original, vs_control}}; K2's rule."""
    original, before, control = scores.get("original"), scores.get("before"), scores.get(CONTROL)
    out = {}
    damage = None if original is None or before is None else original - before
    gap = None if control is None or before is None or not comparable.get(CONTROL) else control - before
    for step in STEPS:
        stage = "after%d" % step
        after = scores.get(stage)
        if after is None or not comparable.get(stage):
            continue
        out[stage] = {"vs_original": round((after - before) / damage, 4) if damage is not None and damage >= MIN_DAMAGE else None,
                      "vs_control": round((after - before) / gap, 4) if gap is not None and gap >= MIN_DAMAGE else None}
    return out


def decide(rl: list, sft: list) -> dict:
    """The rule fixed in the design, on the per-repeat shares against the control by step 300."""
    rl, sft = [v for v in rl if v is not None], [v for v in sft if v is not None]
    result = {"rule": "mean(rl) >= %.2f and mean(sft) < %.2f and mean(rl) - mean(sft) >= %.2f" % (BAR, BAR, GAP),
              "rl": summary(rl), "sft": summary(sft)}
    if not rl or not sft:
        result.update(verdict="NOT_DECIDABLE", outcome="a route has no repeat with at least %d points to recover" % MIN_DAMAGE, gap=None)
        return result
    m_rl, m_sft = statistics.fmean(rl), statistics.fmean(sft)
    gap = round(m_rl - m_sft, 6)
    rl_hides, sft_hides = m_rl >= BAR, m_sft >= BAR
    outcome = ("RL hides, SFT overwrites" if rl_hides and not sft_hides else "both hide" if rl_hides and sft_hides
               else "neither hides" if not (rl_hides or sft_hides) else "SFT hides, RL does not")
    holds = rl_hides and not sft_hides and gap >= GAP
    if rl_hides and not sft_hides and not holds:
        outcome += ", but by %d points, under the %d fixed in advance" % (round(100 * gap), round(100 * GAP))
    result.update(verdict="HOLDS" if holds else "DOES_NOT_HOLD", outcome=outcome, gap=round(gap, 4))
    return result


def subject_entry(sub: str, beds: dict, recovery: dict, reference: str | None) -> dict:
    """Everything the readout needs about one trained model."""
    def bed(point, name):
        return beds.get("%s-%s" % (point, name))

    def correct(found):
        return None if not found else found[1].get("correct")

    points = {"original": ORIGINAL, "before": sub, CONTROL: "%s-%s" % (ORIGINAL, CONTROL),
              **{"after%d" % step: "%s-after%d" % (sub, step) for step in STEPS}}
    spider = {stage: correct(bed(point, "spider")) for stage, point in points.items() if bed(point, "spider")}
    spider_same = {stage: reference is not None and machine(bed(point, "spider")[1]) == reference
                   for stage, point in points.items() if bed(point, "spider")}
    panel = recovery.get(sub) or {}
    stages = panel.get("stages") or {}
    original, before = stages.get("original"), stages.get("before")
    gsm8k_before, gsm8k_original = correct(bed(sub, "gsm8k")), correct(bed(ORIGINAL, "gsm8k"))
    entry = {
        "route": SUBJECT.match(sub)["route"], "seed": int(SUBJECT.match(sub)["seed"]),
        "spider": {"scores": spider, "comparable": spider_same,
                   "damage": None if spider.get("original") is None or spider.get("before") is None else spider["original"] - spider["before"],
                   "shares": shares(spider, spider_same),
                   "output_tokens_mean": {stage: bed(point, "spider")[1].get("output_tokens_mean") for stage, point in points.items() if bed(point, "spider")}},
        "panel": {"verdict": panel.get("verdict"), "verdict_vs_control": panel.get("verdict_vs_control"),
                  "damage_total": panel.get("damage_total"),
                  "damage": {p: original["scores"][p] - before["scores"][p] for p in panel.get("panels", [])} if original and before else {},
                  "shares": {stage: {"vs_original": stages[stage].get("recovered_share"), "vs_control": stages[stage].get("recovered_share_vs_control")}
                             for stage in stages if stage.startswith("after")},
                  "median_output_chars": {stage: stages[stage]["median_output_chars"] for stage in ("original", "before") if stage in stages}},
        "gsm8k": {"original": gsm8k_original, "after_training": gsm8k_before,
                  "gain": None if gsm8k_before is None or gsm8k_original is None else gsm8k_before - gsm8k_original},
        "relative_distance_from_original": panel.get("relative_distance_from_original") or {},
        "density": {}}
    for name in ("spider", "gsm8k"):
        trained, untrained = cost(bed(sub, name)), cost(bed(ORIGINAL, name))
        entry["density"][name] = density.compare(trained, untrained) if density else {"ratio": None, "verdict": "unknown"}
    return entry


def route_table(entries: dict, route: str, panels: list) -> dict:
    mine = [e for e in entries.values() if e["route"] == route]
    table = {"repeats": len(mine),
             "spider_damage": summary([e["spider"]["damage"] for e in mine]),
             "panel_damage_total": summary([e["panel"]["damage_total"] for e in mine]),
             "panel_damage": {p: summary([e["panel"]["damage"].get(p) for e in mine]) for p in panels},
             "gsm8k_gain": summary([e["gsm8k"]["gain"] for e in mine]),
             "relative_distance": {point: summary([e["relative_distance_from_original"].get(point) for e in mine])
                                   for point in ["damaged"] + ["after%d" % s for s in STEPS]},
             "density_ratio": {name: summary([e["density"][name]["ratio"] for e in mine]) for name in ("spider", "gsm8k")}}
    for measure in ("spider", "panel"):
        for against in ("vs_original", "vs_control"):
            table["%s_share_%s" % (measure, against)] = {
                "after%d" % s: summary([(e[measure]["shares"].get("after%d" % s) or {}).get(against) for e in mine]) for s in STEPS}
    return table


def pct(value) -> str:
    return "-" if value is None else "%d%%" % round(100 * value)


def num(value, digits: int = 1) -> str:
    return "-" if value is None else ("%%.%df" % digits) % value


def spread(cell: dict, fmt=num) -> str:
    if not cell or cell["mean"] is None:
        return "-"
    return "%s ± %s (n %d)" % (fmt(cell["mean"]), fmt(cell["sd"]) if cell["sd"] is not None else "-", cell["n"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The K2b readout: recovery after damage by RL and by SFT, per route.")
    parser.add_argument("--root", required=True, help="the campaign's k2b folder (holds forgetting/ and eval/)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--campaign", default=str(CAMPAIGN), help="passed to kit/repair/report.py (default: the K2b campaign beside this kit)")
    args = parser.parse_args(argv)
    out, root = Path(args.out), Path(args.root)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    forget, beds = root / "forgetting", latest_found(root / "eval", "bed-score.json")
    try:
        if not forget.is_dir():
            raise K2bReportError("no forgetting/ under %s: this is not a K2b tree" % root)
        subjects = sorted({m["stem"].rsplit("-before", 1)[0] for m in map(ATTEMPT.match, (p.name for p in forget.iterdir())) if m and m["stem"].endswith("-before")
                           and SUBJECT.match(m["stem"].rsplit("-before", 1)[0])}, key=lambda s: (s.split("-")[0], int(SUBJECT.match(s)["seed"])))
        if not subjects:
            raise K2bReportError("no <route>-seed<S>-before scoring under %s" % forget)
        for point in (ORIGINAL, "%s-%s" % (ORIGINAL, CONTROL)):
            if "%s-spider" % point not in beds:
                raise K2bReportError("no Spider scoring of %s under %s/eval: every share is read against it" % (point, root))
    except K2bReportError as exc:
        raise SystemExit("refused: %s" % exc)

    out.mkdir(parents=True)
    aliases = [arg for sub in subjects for arg in ("--alias", "%s=%s" % (sub, ORIGINAL))]
    campaign = ["--campaign", args.campaign] if Path(args.campaign).is_file() else []
    repair.main(["--root", str(forget), "--out", str(out / "recovery"), *campaign, *aliases])
    recovery = json.loads((out / "recovery" / "recovery-report.json").read_text())
    reference = machine(beds["%s-spider" % ORIGINAL][1])
    entries = {sub: subject_entry(sub, beds, recovery["subjects"], reference) for sub in subjects}
    panels = sorted({p for e in entries.values() for p in e["panel"]["damage"]})
    machines = sorted({str(machine(result)) for _, result in beds.values()}
                      | {str(machine(json.loads((d / "forgetting.json").read_text()))) for d in forget.iterdir() if (d / "forgetting.json").is_file()})
    routes = {route: route_table(entries, route, panels) for route in ROUTES}
    verdicts = {measure: decide([(e[measure]["shares"].get(FINAL) or {}).get("vs_control") for e in entries.values() if e["route"] == "rl"],
                                [(e[measure]["shares"].get(FINAL) or {}).get("vs_control") for e in entries.values() if e["route"] == "sft"])
                for measure in ("spider", "panel")}
    gains = [routes[r]["gsm8k_gain"]["mean"] for r in ROUTES]
    comparable_learning = (None if None in gains else
                           all(g > 0 for g in gains) and max(gains) <= LEARNED_RATIO * min(gains))
    report = {"schema": SCHEMA, "root": str(root.resolve()), "original": ORIGINAL, "control": "%s-%s" % (ORIGINAL, CONTROL),
              "bar": BAR, "gap": GAP, "min_damage_points": MIN_DAMAGE, "subjects_reported": len(subjects),
              "subjects": entries, "routes": routes, "verdict_of_record": "spider", "verdicts": verdicts,
              "verdict": verdicts["spider"]["verdict"], "learned_comparably": comparable_learning,
              "machines": machines, "comparable": int(len(machines) == 1 and machines[0] != "None"),
              "recovery_report": "recovery/recovery-report.json"}
    (out / "k2b-report.json").write_text(json.dumps(report, indent=1, sort_keys=True))

    lines = ["# K2b: does the route of damage decide whether an ability comes back?", "",
             "The untrained Qwen3-1.7B learned GSM8K by GRPO (`rl`) or by supervised fine-tuning (`sft`), matched in steps, rows, rate and parameters; "
             "K2's repair was applied to each trained model and once to the untrained model (the control).", "",
             "**Verdict of record (Spider held-out): %s** -- %s. Rule fixed in advance: mean share recovered against the control by step 300 at least %d%% for RL, "
             "below %d%% for SFT, and RL at least %d points above SFT." % (verdicts["spider"]["verdict"], verdicts["spider"]["outcome"], round(100 * BAR), round(100 * BAR), round(100 * GAP)), "",
             "| measure | verdict | RL mean ± sd (n) | SFT mean ± sd (n) | gap |", "|---|---|---|---|---|"]
    for measure in ("spider", "panel"):
        v = verdicts[measure]
        lines.append("| %s%s | %s | %s | %s | %s |" % (measure, " (of record)" if measure == "spider" else "", v["verdict"], spread(v["rl"], pct), spread(v["sft"], pct),
                                                    "-" if v["gap"] is None else "%+d points" % round(100 * v["gap"])))
    lines += ["", "Did the two routes learn comparable maths? %s (GSM8K gain RL %s, SFT %s; flagged when a mean is not positive or one is more than %g times the other)."
              % ({True: "yes", False: "NO: read the comparison with that in mind", None: "unknown"}[comparable_learning],
                 spread(routes["rl"]["gsm8k_gain"]), spread(routes["sft"]["gsm8k_gain"]), LEARNED_RATIO), ""]
    if not report["comparable"]:
        lines += ["**The scorings carry %d machine-and-mode fingerprints (%s): only same-machine shares are counted.**" % (len(machines), ", ".join(machines)), ""]
    lines += ["## Per route, mean ± sd over repeats", "", "| | RL | SFT |", "|---|---|---|"]
    rows = [("Spider damage (points)", "spider_damage", num), ("panel damage, total", "panel_damage_total", num)]
    for label, key, fmt in rows:
        lines.append("| %s | %s | %s |" % (label, spread(routes["rl"][key], fmt), spread(routes["sft"][key], fmt)))
    for p in panels:
        lines.append("| panel damage, %s | %s | %s |" % (p, spread(routes["rl"]["panel_damage"][p]), spread(routes["sft"]["panel_damage"][p])))
    lines.append("| GSM8K gain (points of 300) | %s | %s |" % (spread(routes["rl"]["gsm8k_gain"]), spread(routes["sft"]["gsm8k_gain"])))
    for measure in ("spider", "panel"):
        for against in ("vs_original", "vs_control"):
            for step in (50, 300):
                key = "%s_share_%s" % (measure, against)
                lines.append("| %s recovered %s, step %d | %s | %s |" % (measure, against.replace("_", " "), step,
                                                                        spread(routes["rl"][key]["after%d" % step], pct), spread(routes["sft"][key]["after%d" % step], pct)))
    for point in ("damaged", "after50", "after300"):
        lines.append("| weight distance from the original, %s | %s | %s |" % (point, spread(routes["rl"]["relative_distance"][point], lambda v: num(v, 4)),
                                                                            spread(routes["sft"]["relative_distance"][point], lambda v: num(v, 4))))
    for name in ("spider", "gsm8k"):
        lines.append("| density: tokens per correct, %s, x untrained (bar %s) | %s | %s |" % (name, num(density.BAR if density else 1.5, 1),
                                                                                         spread(routes["rl"]["density_ratio"][name], lambda v: num(v, 2)),
                                                                                         spread(routes["sft"]["density_ratio"][name], lambda v: num(v, 2))))
    lines += ["", "## Per repeat", "",
              "| subject | Spider orig / trained / after50 / after300 / control | Spider share vs control, 50 / 300 | panel damage | panel share vs control, 50 / 300 | GSM8K gain | distance trained / after300 |",
              "|---|---|---|---|---|---|---|"]
    for sub, e in entries.items():
        s, p = e["spider"], e["panel"]
        lines.append("| %s | %s | %s / %s | %s | %s / %s | %s | %s / %s |" % (
            sub, " / ".join(num(s["scores"].get(k), 0) for k in ("original", "before", "after50", FINAL, CONTROL)),
            pct((s["shares"].get("after50") or {}).get("vs_control")), pct((s["shares"].get(FINAL) or {}).get("vs_control")),
            num(p["damage_total"], 0), pct((p["shares"].get("after50") or {}).get("vs_control")), pct((p["shares"].get(FINAL) or {}).get("vs_control")),
            num(e["gsm8k"]["gain"], 0), num(e["relative_distance_from_original"].get("damaged"), 4), num(e["relative_distance_from_original"].get(FINAL), 4)))
    lines += ["", "The panel's own per-stage tables, with median answer lengths, are in recovery/recovery-report.md (kit/repair/report.py, unchanged)."]
    (out / "k2b-report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
