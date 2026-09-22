#!/usr/bin/env python3
"""The K4a readout: each arm and seed beside the K0 control that shares its seed.

    python k4a_report.py --runs $WORK/runs --forgetting $WORK/k4a/forgetting \\
                         --k0 /work/k0-report/report.json --out $WORK/k4a/report-a1

Reads three things and recomputes every number from them:

    --runs         the K4a run directories written by kit/run_sdpo_toolalpaca.sh, named
                   <arm>-seed<S>-aN (and pilot-<arm>-aN, which is reported but never compared)
    --forgetting   <arm>-seed<S>-aN/forgetting.json and base-aN/forgetting.json from
                   kit/score_forgetting.py, all scored on ONE GPU of ONE machine
    --k0           the CONTROL: the partner's finished K0 report.json (kit/make_report.py), whose
                   dose40-seed42/43/44 runs are what every arm is paired against, seed for seed

For every arm, seed and validation step it reports the strict tool-use score, the share of the batch
with no successful attempt, the success share, response length, entropy, and the three general
panels -- each beside the K0 run with the same seed.

TWO RULES IT ENFORCES, because both have already cost this programme a wrong answer:

1. THE STRICT SCORE. Every accuracy here is `val-core/tooluse/acc/mean@16`, the authors'
   all-or-nothing score, recomputed per question from the `acc` field of the validation dumps. The
   `soft` arm's partial credit lands in `score`, never in `acc`, so it cannot reach this number --
   and if a run's logged accuracy and its recomputed accuracy ever disagree, that is a FLAG, not a
   rounding difference.
2. ONE MACHINE. Panel counts are only comparable when scored on the same machine in the same
   decoding mode: across machines a panel moves by up to 3 points before any training (receipt 204).
   Mixed fingerprints are REFUSED unless --allow-different-machines, which is recorded as the
   departure it is.

STRICT SUCCESS, SEPARATELY. `self_distillation/success_group_fraction` and `empty_target_batch` are
computed by the trainer from whatever reward it was given, so under the `soft` arm they count
near-misses and are NOT comparable with K0's. Where the run kept its rollout dumps
(`trainer.rollout_data_dir`, which kit/run_sdpo_toolalpaca.sh always sets) this file also recomputes
a STRICT per-question success share from the `acc` of each attempt, which is comparable across every
arm. Both are reported, side by side, and the table says which is which.

Standard library only. Nothing is overwritten: an existing --out is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-k4a-report.v1"
VAL_KEY = "val-core/tooluse/acc/mean@16"
CONTROL_PREFIX = "dose40-seed"
POINT = re.compile(r"^(?P<arm>[a-z0-9]+)-seed(?P<seed>\d+)$")
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
TRAIN_KEYS = {
    "score": "critic/score/mean",
    "response_tokens": "response_length/mean",
    "success_group_fraction": "self_distillation/success_group_fraction",
    "empty_target_batch": "self_distillation/empty_target_batch",
    "entropy": "actor/entropy",
    "grad_norm": "actor/grad_norm",
    "seconds_per_step": "perf/time_per_step",
    "max_memory_gb": "perf/max_memory_allocated_gb",
}


class K4aReportError(ValueError):
    """The evidence tree cannot support a report; nothing is written."""


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _round(value, digits=4):
    return round(value, digits) if _finite(value) else None


def _jsonl(path: Path) -> list:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _mean(values):
    clean = [v for v in values if _finite(v)]
    return _round(statistics.fmean(clean)) if clean else None


def spread(values: list) -> dict:
    """Mean and SAMPLE standard deviation; the spread of one seed is not a number, so it is null."""
    clean = [v for v in values if _finite(v)]
    return {"n": len(clean), "mean": _round(statistics.fmean(clean)) if clean else None,
            "sd": _round(statistics.stdev(clean)) if len(clean) > 1 else None}


# --------------------------------------------------------------------------- the strict measurements
def per_question_acc(path: Path) -> dict:
    """One dump -> {question id: mean STRICT accuracy over its samples}. Reads `acc`, never `score`."""
    groups: dict = defaultdict(list)
    for row in _jsonl(path):
        if "input" not in row or "acc" not in row:
            continue
        groups[hashlib.sha1(row["input"].encode()).hexdigest()[:12]].append(row)
    return {qid: statistics.fmean(float(r["acc"]) for r in rows) for qid, rows in groups.items()}


def strict_success(path: Path) -> dict | None:
    """A rollout dump -> the STRICT share of questions with at least one correct attempt, and of
    attempts that were correct. Comparable across arms because it never reads the reward."""
    groups: dict = defaultdict(list)
    for row in _jsonl(path):
        if "input" not in row or "acc" not in row:
            continue
        groups[hashlib.sha1(row["input"].encode()).hexdigest()[:12]].append(float(row["acc"]))
    if not groups:
        return None
    attempts = [a for values in groups.values() for a in values]
    return {"questions": len(groups),
            "attempts_per_question": sorted({len(v) for v in groups.values()}),
            "strict_success_group_fraction": _round(
                statistics.fmean(1.0 if max(v) >= 1.0 else 0.0 for v in groups.values())),
            "strict_empty_target_fraction": _round(
                statistics.fmean(0.0 if max(v) >= 1.0 else 1.0 for v in groups.values())),
            "strict_attempt_accuracy": _round(statistics.fmean(attempts))}


def read_run(run: Path) -> dict:
    """One run directory -> validations (strict), per-step training signals, and identity."""
    out: dict = {"name": run.name, "flags": []}
    summary = run / "run-summary.json"
    if summary.is_file():
        try:
            out["summary"] = json.loads(summary.read_text())
        except json.JSONDecodeError:
            out["flags"].append("UNREADABLE run-summary.json")
    metrics = run / "metrics.jsonl"
    training, validations = [], []
    for record in (_jsonl(metrics) if metrics.is_file() else []):
        data = record.get("data") or {}
        if VAL_KEY in data:
            validations.append({"step": record["step"], "accuracy_logged": _round(data[VAL_KEY])})
        if TRAIN_KEYS["score"] in data:
            training.append({"step": record["step"],
                             **{k: _round(data.get(v)) for k, v in TRAIN_KEYS.items()}})
    out["training"] = training
    out["steps_completed"] = max((t["step"] for t in training), default=0)

    dumps = {}
    vdir = run / "validation"
    if vdir.is_dir():
        for path in sorted(vdir.glob("*.jsonl")):
            if path.stem.isdigit():
                dumps[int(path.stem)] = per_question_acc(path)
    for entry in validations:
        found = dumps.get(entry["step"])
        if found:
            entry["accuracy_strict"] = _round(statistics.fmean(found.values()))
            entry["questions"] = len(found)
            if _finite(entry["accuracy_logged"]) and abs(entry["accuracy_logged"] - entry["accuracy_strict"]) > 1e-3:
                out["flags"].append(
                    "MISMATCH at step %s: the trainer logged %.4f but the per-question `acc` gives "
                    "%.4f. The validation metric is not the strict score."
                    % (entry["step"], entry["accuracy_logged"], entry["accuracy_strict"]))
    out["validations"] = validations
    out["per_question"] = {str(step): {q: _round(a) for q, a in found.items()} for step, found in dumps.items()}

    rollouts = run / "rollouts"
    strict = [strict_success(p) for p in sorted(rollouts.glob("*.jsonl"))] if rollouts.is_dir() else []
    strict = [s for s in strict if s]
    if strict:
        out["strict_success"] = {
            "steps_read": len(strict),
            "success_group_fraction": _mean([s["strict_success_group_fraction"] for s in strict]),
            "empty_target_fraction": _mean([s["strict_empty_target_fraction"] for s in strict]),
            "attempt_accuracy": _mean([s["strict_attempt_accuracy"] for s in strict]),
            "attempts_per_question": sorted({n for s in strict for n in s["attempts_per_question"]}),
        }
    else:
        out["flags"].append("NO ROLLOUT DUMPS: the strict success share could not be recomputed, so "
                            "success_group_fraction below is the trainer's own and is not comparable "
                            "across arms with different rewards")
    out["means"] = {key: _mean([t[key] for t in training]) for key in TRAIN_KEYS}
    out["max_memory_gb"] = max([t["max_memory_gb"] for t in training if _finite(t["max_memory_gb"])], default=None)
    merged = sorted(run.glob("hf-step*/config.json"))
    out["merged_checkpoint"] = merged[-1].parent.name if merged else None
    return out


def latest_runs(runs_dir: Path) -> dict:
    """{point: run summary} for the highest attempt of every <point>-aN directory."""
    found: dict = {}
    if not runs_dir.is_dir():
        raise K4aReportError("not a directory: %s" % runs_dir)
    for child in sorted(runs_dir.iterdir()):
        match = ATTEMPT.match(child.name) if child.is_dir() else None
        if not match:
            continue
        stem, attempt = match["stem"], int(match["attempt"])
        if stem not in found or attempt > found[stem][0]:
            found[stem] = (attempt, child)
    return {stem: read_run(path) for stem, (_, path) in found.items()}


def latest_panels(forgetting: Path) -> dict:
    """{stem: forgetting.json} for the highest attempt under the forgetting directory."""
    found: dict = {}
    if not forgetting.is_dir():
        return {}
    for child in sorted(forgetting.iterdir()):
        match = ATTEMPT.match(child.name) if child.is_dir() else None
        if not match or not (child / "forgetting.json").is_file():
            continue
        stem, attempt = match["stem"], int(match["attempt"])
        if stem not in found or attempt > found[stem][0]:
            found[stem] = (attempt, json.loads((child / "forgetting.json").read_text()))
    return {stem: result for stem, (_, result) in found.items()}


# ------------------------------------------------------------------------------------ the control
def read_control(path: Path) -> dict:
    """The K0 report's dose40-seed* runs, keyed by seed. Every number is the authors' strict score."""
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise K4aReportError("cannot read the K0 control report at %s: %s" % (path, exc))
    control = {}
    for run in report.get("runs") or []:
        name = run.get("name") or ""
        if not name.startswith(CONTROL_PREFIX):
            continue
        seed = name[len(CONTROL_PREFIX):]
        if not seed.isdigit():
            continue
        training = run.get("training") or []
        per_question = run.get("per_question") or {}
        last_step = max(per_question, key=lambda step: int(step)) if per_question else None
        control[int(seed)] = {
            "name": name,
            "validations": [{"step": v["step"], "accuracy": _round(v.get("accuracy_logged"))}
                            for v in run.get("validations") or []],
            "final": _round((run.get("validations") or [{}])[-1].get("accuracy_logged")),
            "means": {key: _mean([t.get(key) for t in training])
                      for key in ("empty_target_batch", "success_group_fraction",
                                  "response_tokens", "entropy", "score")},
            # what THIS run looked like over its own first two steps: the only fair thing to hold a
            # two-step pilot against, and the reason those pilot numbers are printed and not gated.
            "first_two_steps": {key: _mean([t.get(key) for t in training[:2]])
                                for key in ("empty_target_batch", "success_group_fraction",
                                            "response_tokens", "entropy", "score")},
            "steps_completed": run.get("steps_completed"),
            # the per-question accuracy at the control's LAST validation: what K0 could and could not
            # do. Question ids are the sha1 of the prompt, so they name the same held-out question on
            # both sides of the comparison (kit/make_report.py:68 and per_question_acc above).
            "per_question": per_question.get(last_step) if last_step is not None else None,
            "per_question_step": int(last_step) if last_step is not None else None,
        }
    if not control:
        raise K4aReportError(
            "the K0 report at %s holds no run named %s<seed>. K4a is paired seed for seed with the "
            "K0 dose-40 runs; without them there is nothing to compare against." % (path, CONTROL_PREFIX))
    return control


# ------------------------------------------------------------------------------------ the build
def fingerprints(panels: dict) -> list:
    seen = {(result.get("machine") or {}).get("id") for result in panels.values()}
    return sorted(seen, key=lambda item: (item is None, item))


def panel_scores(result: dict | None) -> dict:
    if not result:
        return {}
    return {name: block["correct"] for name, block in sorted((result.get("panels") or {}).items())}


def build(runs_dir: Path, forgetting: Path, k0: Path, *, allow_different_machines: bool = False) -> dict:
    runs = latest_runs(runs_dir)
    control = read_control(k0)
    panels = latest_panels(forgetting)
    machines = fingerprints(panels)
    comparable = len(machines) == 1 and machines[0] is not None
    if panels and not comparable and not allow_different_machines:
        raise K4aReportError(
            "the panel scorings under %s carry %d different machine-and-mode fingerprints (%s). A "
            "count from one machine cannot be subtracted from a count on another: a panel moves by "
            "up to 3 points across machines before any training. Re-score on one machine, or pass "
            "--allow-different-machines and say so in the readout."
            % (forgetting, len(machines), ", ".join(str(m) for m in machines)))

    base_panels = panel_scores(panels.get("base"))
    arms: dict = {}
    for point, run in sorted(runs.items()):
        match = POINT.match(point)
        if not match:
            continue                                   # pilot-*, and anything else not in the grid
        arm, seed = match["arm"], int(match["seed"])
        after = panel_scores(panels.get("%s-forget" % point))
        row = {
            "run": run["name"], "seed": seed, "steps_completed": run["steps_completed"],
            "validations": run["validations"],
            "final": (run["validations"] or [{}])[-1].get("accuracy_strict")
                     or (run["validations"] or [{}])[-1].get("accuracy_logged"),
            "means": run["means"], "strict_success": run.get("strict_success"),
            "max_memory_gb": run["max_memory_gb"], "merged_checkpoint": run["merged_checkpoint"],
            "arm_declared": (run.get("summary") or {}).get("arm"),
            "panels": after,
            "panel_change": {name: after[name] - base_panels[name]
                             for name in sorted(set(after) & set(base_panels))},
            "flags": list(run["flags"]),
            "control": control.get(seed),
        }
        if row["control"] and _finite(row["final"]) and _finite(row["control"]["final"]):
            row["change_vs_control"] = _round(row["final"] - row["control"]["final"])
        # split the questions the way the package's question is asked: what K0 could not do
        row["by_k0_difficulty"] = split_by_control(run, control.get(seed))
        # A split over nothing is the quietest way to get this wrong: if the two reports ever named
        # questions differently, every table above would still render and this one would be empty.
        if row["by_k0_difficulty"] and not row["by_k0_difficulty"]["shared_questions"]:
            row["flags"].append(
                "NO SHARED QUESTIONS with the control run %s: this arm's validation dump and the K0 "
                "report name no question the same way, so the stuck/solved split below is empty "
                "rather than zero." % (row["control"] or {}).get("name"))
        arms.setdefault(arm, {"seeds": {}})["seeds"][seed] = row

    for arm, block in arms.items():
        ordered = [block["seeds"][s] for s in sorted(block["seeds"])]
        block["final"] = spread([r["final"] for r in ordered])
        block["change_vs_control"] = spread([r.get("change_vs_control") for r in ordered])
        block["panel_change"] = {name: spread([r["panel_change"].get(name) for r in ordered])
                                 for name in sorted({n for r in ordered for n in r["panel_change"]})}
    return {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_dir": str(runs_dir.resolve()), "forgetting_dir": str(forgetting.resolve()),
        "control_report": str(k0.resolve()),
        "control": control, "control_runs": len(control),
        "base_panels": base_panels,
        "arms": {arm: arms[arm] for arm in sorted(arms)},
        "arms_reported": len(arms),
        "pilots": {name: runs[name] for name in sorted(runs) if name.startswith("pilot-")},
        "pilot_checks": pilot_checks(runs, control),
        "machine_ids": machines, "comparable": int(comparable),
        "different_machines_allowed": int(bool(allow_different_machines)),
    }


#: what each pilot is supposed to move, and the column the reader should compare it with. These are
#: CHECKS, not gates: over two steps every one of them is noisy, and the runner makes every pilot
#: gate every later row, so a bar on a noisy number would stop all three arms on a coin flip. Only
#: `feedback`'s empty_target_batch is gated in the campaign, because the checker writes feedback for
#: every failed attempt and so that number cannot fail by luck.
PILOT_CHECKS = (
    ("empty_target_batch", "share of the batch with nothing to learn from", "lower"),
    ("success_group_fraction", "share of questions with a usable attempt", "higher"),
    ("entropy", "how varied the attempts were", "higher"),
    ("response_tokens", "response length", "watch"),
)


def pilot_checks(runs: dict, control: dict) -> dict:
    """The two-step pilots beside K0's own first two steps. Printed, never judged."""
    pilots = {name: run for name, run in runs.items() if name.startswith("pilot-")}
    if not pilots:
        return {}
    reference = {key: spread([c["first_two_steps"].get(key) for c in control.values()])
                 for key, _, _ in PILOT_CHECKS}
    return {
        "control_first_two_steps": reference,
        "control_runs": sorted(c["name"] for c in control.values()),
        "arms": {name[len("pilot-"):]: {key: run["means"].get(key) for key, _, _ in PILOT_CHECKS}
                 for name, run in sorted(pilots.items())},
    }


def split_by_control(run: dict, control: dict | None) -> dict | None:
    """The package's actual question: did the arm rescue what K0 could not do, and keep what it could?

    Questions the control scored 0 on at its LAST validation are `stuck`; the rest are
    `already_solved`. Both sides are keyed by the sha1 of the prompt, so they name the same held-out
    question, and both sides are the strict `acc`. Only questions present on BOTH sides are counted.
    """
    if not control or not control.get("per_question"):
        return None
    steps = [int(s) for s in run["per_question"]]
    now = run["per_question"].get(str(max(steps))) if steps else None
    if not now:
        return None
    theirs = control["per_question"]
    shared = sorted(set(theirs) & set(now))
    stuck = [q for q in shared if theirs[q] == 0.0]
    solved = [q for q in shared if theirs[q] > 0.0]
    return {
        "control_step": control.get("per_question_step"),
        "shared_questions": len(shared),
        "stuck_questions": len(stuck),
        "stuck_accuracy_now": _mean([now[q] for q in stuck]),
        "already_solved_questions": len(solved),
        "already_solved_accuracy_now": _mean([now[q] for q in solved]),
        "control_stuck_accuracy": 0.0,
        "control_already_solved_accuracy": _mean([theirs[q] for q in solved]),
    }


# ------------------------------------------------------------------------------------ rendering
def _fmt(value, pattern="%.4f"):
    return pattern % value if _finite(value) else "-"


def render(report: dict) -> str:
    lines = ["# K4a: which fix rescues the stuck questions, and what does it cost the rest?", "",
             "Generated %s by kit/k4a_report.py (%s). Every number is recomputed from the raw files."
             % (report["generated_at"], SCHEMA), "",
             "The control is %s: the K0 dose-40 runs at the same seeds, unchanged." % report["control_report"],
             "All accuracies are the authors' STRICT all-or-nothing tool-use score, "
             "`val-core/tooluse/acc/mean@16`, recomputed per question from `acc`.", ""]
    if not report["comparable"]:
        lines += ["**The panel scorings do not share one machine-and-mode fingerprint (%s), so the "
                  "panel differences below are not comparable.**"
                  % ", ".join(str(m) for m in report["machine_ids"]), ""]
    lines += ["## Final score, against the K0 run with the same seed", "",
              "| arm | seed | steps | K0 control | this arm | change | max GB |", "|---|---|---|---|---|---|---|"]
    for arm in sorted(report["arms"]):
        block = report["arms"][arm]
        for seed in sorted(block["seeds"]):
            row = block["seeds"][seed]
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                arm, seed, row["steps_completed"], _fmt((row["control"] or {}).get("final")),
                _fmt(row["final"]), _fmt(row.get("change_vs_control"), "%+.4f"),
                _fmt(row["max_memory_gb"], "%.1f")))
        lines.append("| **%s** | mean of %d | | | **%s** (sd %s) | **%s** (sd %s) | |" % (
            arm, block["final"]["n"], _fmt(block["final"]["mean"]), _fmt(block["final"]["sd"]),
            _fmt(block["change_vs_control"]["mean"], "%+.4f"), _fmt(block["change_vs_control"]["sd"])))

    lines += ["", "## The stuck questions: what each arm did to the batch it could not learn from", "",
              "`strict` is recomputed from the per-attempt `acc` in the rollout dumps and is comparable "
              "across arms. `as trained` is the trainer's own figure, computed from whatever reward "
              "that arm gave it -- under `soft` it counts near-misses, so it is NOT comparable with K0.", "",
              "| arm | seed | empty target (strict) | empty target (as trained) | success group (strict) "
              "| success group (as trained) | K0 empty target | response tokens | entropy |",
              "|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(report["arms"]):
        for seed in sorted(report["arms"][arm]["seeds"]):
            row = report["arms"][arm]["seeds"][seed]
            strict = row.get("strict_success") or {}
            control = (row["control"] or {}).get("means", {})
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                arm, seed, _fmt(strict.get("empty_target_fraction"), "%.3f"),
                _fmt(row["means"].get("empty_target_batch"), "%.3f"),
                _fmt(strict.get("success_group_fraction"), "%.3f"),
                _fmt(row["means"].get("success_group_fraction"), "%.3f"),
                _fmt(control.get("empty_target_batch"), "%.3f"),
                _fmt(row["means"].get("response_tokens"), "%.1f"),
                _fmt(row["means"].get("entropy"), "%.3f")))

    split = [(arm, seed, report["arms"][arm]["seeds"][seed]["by_k0_difficulty"])
             for arm in sorted(report["arms"]) for seed in sorted(report["arms"][arm]["seeds"])
             if report["arms"][arm]["seeds"][seed].get("by_k0_difficulty")]
    if split:
        lines += ["", "## Rescued without hurting the rest?", "",
                  "Held-out questions split by what the K0 run at the same seed could do at its last "
                  "validation. `stuck` is every question K0 scored 0 on.", "",
                  "| arm | seed | stuck questions | this arm on them | already solved | K0 on those | this arm on those |",
                  "|---|---|---|---|---|---|---|"]
        for arm, seed, block in split:
            lines.append("| %s | %s | %d | %s | %d | %s | %s |" % (
                arm, seed, block["stuck_questions"], _fmt(block["stuck_accuracy_now"]),
                block["already_solved_questions"], _fmt(block["control_already_solved_accuracy"]),
                _fmt(block["already_solved_accuracy_now"])))

    if report["base_panels"]:
        names = sorted(report["base_panels"])
        lines += ["", "## Forgetting (a trained model may lose at most 3 per 100 on each panel)", "",
                  "Untrained: " + ", ".join("%s %d" % (n, report["base_panels"][n]) for n in names), "",
                  "| arm | seed | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for arm in sorted(report["arms"]):
            for seed in sorted(report["arms"][arm]["seeds"]):
                row = report["arms"][arm]["seeds"][seed]
                lines.append("| %s | %s | %s |" % (arm, seed, " | ".join(
                    "%s (%+d)" % (row["panels"][n], row["panel_change"][n])
                    if n in row["panels"] and n in row["panel_change"] else "-" for n in names)))

    checks = report.get("pilot_checks") or {}
    if checks.get("arms"):
        lines += ["", "## Did each knob do what its arm is named for? (a CHECK, not a gate)", "",
                  "Each arm's two-step pilot, beside the same two steps of the K0 control runs (%s). "
                  "Over two steps every one of these swings widely -- K0's own per-step success share "
                  "ranges 0.19 to 0.62 -- so only `feedback`'s empty-target share is gated in the "
                  "campaign, where it cannot fail by luck. **Read this table; do not treat it as a "
                  "verdict.**" % ", ".join(checks["control_runs"]), "",
                  "| what it should move | K0's first two steps | " +
                  " | ".join(sorted(checks["arms"])) + " |",
                  "|---|---|" + "---|" * len(checks["arms"])]
        for key, label, direction in PILOT_CHECKS:
            reference = checks["control_first_two_steps"].get(key) or {}
            lines.append("| %s (%s is the intended direction) | %s (sd %s) | %s |" % (
                label, direction, _fmt(reference.get("mean"), "%.3f"), _fmt(reference.get("sd"), "%.3f"),
                " | ".join(_fmt(checks["arms"][arm].get(key), "%.3f") for arm in sorted(checks["arms"]))))

    lines += ["", "## Every validation", "",
              "| arm | seed | " + " | ".join("step %d" % s for s in _validation_steps(report)) + " |",
              "|---|---|" + "---|" * len(_validation_steps(report))]
    for arm in sorted(report["arms"]):
        for seed in sorted(report["arms"][arm]["seeds"]):
            row = report["arms"][arm]["seeds"][seed]
            found = {v["step"]: v.get("accuracy_strict", v.get("accuracy_logged")) for v in row["validations"]}
            lines.append("| %s | %s | %s |" % (arm, seed, " | ".join(
                _fmt(found.get(s)) for s in _validation_steps(report))))
    for seed in sorted(report["control"]):
        found = {v["step"]: v["accuracy"] for v in report["control"][seed]["validations"]}
        lines.append("| **K0 control** | %s | %s |" % (seed, " | ".join(
            _fmt(found.get(s)) for s in _validation_steps(report))))

    flags = [(arm, seed, flag) for arm in sorted(report["arms"])
             for seed in sorted(report["arms"][arm]["seeds"])
             for flag in report["arms"][arm]["seeds"][seed]["flags"]]
    if flags:
        lines += ["", "## Flags", ""] + ["- **%s seed %s** %s" % item for item in flags]
    lines += ["", "%d arms over %d seeds, against %d control runs."
              % (report["arms_reported"], len({s for a in report["arms"].values() for s in a["seeds"]}),
                 report["control_runs"]), ""]
    return "\n".join(lines)


def _validation_steps(report: dict) -> list:
    steps = {v["step"] for arm in report["arms"].values() for row in arm["seeds"].values()
             for v in row["validations"]}
    steps |= {v["step"] for run in report["control"].values() for v in run["validations"]}
    return sorted(steps)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, required=True, help="the K4a run directories")
    parser.add_argument("--forgetting", type=Path, required=True, help="the panel scorings")
    parser.add_argument("--k0", type=Path, required=True, help="the control: K0's report.json")
    parser.add_argument("--out", type=Path, required=True, help="a new directory; never overwritten")
    parser.add_argument("--allow-different-machines", action="store_true",
                        help="report anyway; recorded in the report as a departure")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % args.out)
    try:
        report = build(args.runs, args.forgetting, args.k0,
                       allow_different_machines=args.allow_different_machines)
    except K4aReportError as exc:
        raise SystemExit(str(exc))
    args.out.mkdir(parents=True)
    (args.out / "k4a-report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    text = render(report)
    (args.out / "k4a-report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
