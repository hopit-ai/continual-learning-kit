#!/usr/bin/env python3
"""The recovery report: for each damaged model, how much of its lost general ability a short repair brought back.

    python report.py --root $WORK/k2 --out $WORK/k2/report-a1 [--campaign FILE] [--alias subject=model ...]

Reads the scorer's folders under --root (named <subject>-<stage>-aN: `original`, `before`, `after50` ...
`after300`, and `control300` for the same repair applied to the healthy original) and the drift files,
and writes recovery-report.md and .json. Standard library only.

The campaign scores a subject's healthy original and its control under the MODEL's name (`q3-original`,
`q3-control300` for subject `spider60`), so `original` and `control*` are looked up through an alias read
from the campaign's own damage-check rows, which pair `<model>-original` with `<subject>-before`. `--alias`
adds or overrides pairs for another campaign. `before` and `after*` are always the subject's own.

Recovered share, per panel and overall = (after - before) / (original - before), reported only where the
damage was at least 5 points, because a ratio over a smaller loss is noise. The bar fixed in advance:
at least 80 percent of the lost total back within 300 steps. A scoring is used only if it carries the
same machine-and-mode fingerprint as the subject's original. Beside it, and not replacing it, the share
against the control = (after - before) / (control - before), with the same bar and the same minimum-damage
rule: the repair by itself costs a healthy model points, so the control is as far back as a repaired
model can be expected to come.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BAR = 0.80
MIN_DAMAGE = 5
NAME = re.compile(r"^(?P<subject>.+)-(?P<stage>original|before|after\d+|control\d+)-a(?P<attempt>\d+)$")
CAMPAIGN = Path(__file__).resolve().parents[1] / "campaigns" / "k2-recovery-test.yaml"
# Found by Shivam Chourasia on the first K2 run, 26 September 2026: the join dropped every subject whose
# original is scored under its model's name, so only the pilot was reported. A damage-check row names both.
PAIR = re.compile(r"/(?P<model>[\w.]+)-original-a\*.*?/(?P<subject>[\w.]+)-before-a\*")


def campaign_aliases(path: Path) -> dict:
    """{subject: model} from the campaign text: the folder its damage check reads as `--original` for the subject's `--before`."""
    return {m["subject"]: m["model"] for m in map(PAIR.search, path.read_text().splitlines()) if m and "--original" in m.string}


def latest(root: Path) -> dict:
    found: dict = {}
    for directory in root.iterdir():
        match = NAME.match(directory.name)
        if match and (directory / "forgetting.json").is_file():
            key = (match["subject"], match["stage"])
            if key not in found or int(match["attempt"]) > found[key][0]:
                found[key] = (int(match["attempt"]), json.loads((directory / "forgetting.json").read_text()))
    return {key: value for key, (_, value) in found.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recovery report for the K2 campaign.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--campaign", help="campaign file to read the subject-to-model aliases from (default: the K2 campaign beside this kit, if present)")
    parser.add_argument("--alias", action="append", default=[], metavar="SUBJECT=MODEL", help="score SUBJECT's original and control from MODEL's folders; repeatable")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    campaign = Path(args.campaign) if args.campaign else CAMPAIGN
    if args.campaign and not campaign.is_file():
        raise SystemExit("no campaign file %s" % campaign)
    alias = campaign_aliases(campaign) if campaign.is_file() else {}
    for pair in args.alias:
        subject, sep, model = pair.partition("=")
        if not (sep and subject and model):
            raise SystemExit("--alias wants subject=model, got %r" % pair)
        alias[subject] = model
    root, scored = Path(args.root), latest(Path(args.root))
    subjects = sorted({subject for subject, _ in scored})
    report = {"schema": "kit-recovery-report.v1", "bar": BAR, "min_damage_points": MIN_DAMAGE, "alias": alias, "subjects": {}}
    lines = ["# Recovery report", "", "Bar fixed in advance: at least %d%% of the lost total back within 300 repair steps. Shares are shown only where the damage was at least %d points."
             % (int(BAR * 100), MIN_DAMAGE), "",
             "Recovery is shown twice: against the original (the bar and verdict of record, fixed before any number existed) and against the control, "
             "because the same repair applied to the healthy original costs it points by itself, so the control is as far back as any repaired model can be expected to come.", ""]
    for subject in subjects:
        origin = alias.get(subject, subject)

        def find(stage, subject=subject, origin=origin):
            """`original` and `control*` come from the healthy model's folders; `before` and `after*` are only ever the subject's own."""
            if stage == "original" or stage.startswith("control"):
                return scored.get((origin, stage)) or scored.get((subject, stage))
            return scored.get((subject, stage))

        original, before = find("original"), find("before")
        if not original or not before:
            continue
        machine = (original.get("machine") or {}).get("id")
        stages = sorted({s for (sub, s) in scored if (sub == subject and s.startswith(("after", "control"))) or (sub == origin and s.startswith("control"))},
                        key=lambda s: (s.startswith("control"), int(re.sub(r"\D", "", s))))
        panels = list(original["panels"])
        rows, entry = [], {"panels": panels, "stages": {}}
        if origin != subject:
            entry["original_and_control_from"] = origin
        for stage in ["original", "before"] + stages:
            result = find(stage)
            same = machine is not None and (result.get("machine") or {}).get("id") == machine
            scores = {p: result["panels"][p]["correct"] for p in panels}
            chars = {p: result["panels"][p]["median_output_chars"] for p in panels}
            entry["stages"][stage] = {"scores": scores, "total": sum(scores.values()), "median_output_chars": chars, "comparable": same}
            rows.append((stage, scores, chars, same))
        damage = entry["stages"]["original"]["total"] - entry["stages"]["before"]["total"]
        entry["damage_total"] = damage
        for stage in stages:
            if stage.startswith("after") and damage >= MIN_DAMAGE and entry["stages"][stage]["comparable"]:
                entry["stages"][stage]["recovered_share"] = round((entry["stages"][stage]["total"] - entry["stages"]["before"]["total"]) / damage, 4)
        finals = [s for s in stages if s.startswith("after") and "recovered_share" in entry["stages"][s]]
        final = finals[-1] if finals else None
        entry["verdict"] = ("NO_DAMAGE_TO_RECOVER" if damage < MIN_DAMAGE else "NOT_COMPARABLE" if not final
                            else "RECOVERED" if entry["stages"][final]["recovered_share"] >= BAR else "NOT_RECOVERED")
        control = next((s for s in stages if s.startswith("control")), None)
        if control:
            entry["repair_harms_a_healthy_model_by"] = entry["stages"]["original"]["total"] - entry["stages"][control]["total"]
        if control and entry["stages"][control]["comparable"]:
            gap = entry["stages"][control]["total"] - entry["stages"]["before"]["total"]
            entry["damage_vs_control_total"] = gap
            for stage in stages:
                if stage.startswith("after") and gap >= MIN_DAMAGE and entry["stages"][stage]["comparable"]:
                    entry["stages"][stage]["recovered_share_vs_control"] = round((entry["stages"][stage]["total"] - entry["stages"]["before"]["total"]) / gap, 4)
            finals = [s for s in stages if s.startswith("after") and "recovered_share_vs_control" in entry["stages"][s]]
            entry["verdict_vs_control"] = ("NO_DAMAGE_TO_RECOVER" if gap < MIN_DAMAGE else "NOT_COMPARABLE" if not finals
                                           else "RECOVERED" if entry["stages"][finals[-1]]["recovered_share_vs_control"] >= BAR else "NOT_RECOVERED")
        else:
            entry["verdict_vs_control"] = "NOT_COMPARABLE" if control else "NO_CONTROL"
        drift = {}
        for path in sorted(root.glob("%s-drift-*.json" % subject)):
            drift[path.stem.split("-drift-", 1)[1]] = json.loads(path.read_text())["relative_distance_all"]
        entry["relative_distance_from_original"] = drift
        report["subjects"][subject] = entry
        lines += ["## %s: %s (against the control: %s)" % (subject, entry["verdict"], entry["verdict_vs_control"]), "",
                  "| stage | " + " | ".join(panels) + " | total | recovered | recovered vs control | median answer chars (%s) |" % ", ".join(panels),
                  "|---|" + "---|" * (len(panels) + 4)]
        for stage, scores, chars, same in rows:
            shares = [entry["stages"][stage].get(key) for key in ("recovered_share", "recovered_share_vs_control")]
            lines.append("| %s%s | %s | %d | %s | %s | %s |" % (stage, "" if same else " (other machine: not comparable)", " | ".join(str(scores[p]) for p in panels), sum(scores.values()),
                                                            *("%d%%" % round(100 * share) if share is not None else "-" for share in shares), ", ".join(str(int(chars[p])) for p in panels)))
        if drift:
            lines += ["", "Relative weight distance from the original: " + "; ".join("%s %.4f" % (k, v) for k, v in drift.items()) + "."]
        if control:
            lines += ["", "The same repair applied to the healthy original changes its total by %+d." % -entry["repair_harms_a_healthy_model_by"]]
        lines.append("")
    out.mkdir(parents=True)
    report["subjects_reported"] = len(report["subjects"])
    (out / "recovery-report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    (out / "recovery-report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
