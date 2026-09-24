#!/usr/bin/env python3
"""The K4 readout: per bed and arm, what the student answers UNAIDED, beside the reused `none` runs.

    python k4_report.py --root $WORK/k4 --runs $WORK/runs \\
                        --spider-none-root $K3_ROOT --finqa-none-root $K1C_ROOT \\
                        --finqa-rows $WORK/data/finqa-1600/test.jsonl --out $WORK/k4/report-a1

Reads four kinds of file and recomputes every number from them:

    --root              K4's own tree: `eval/<point>-a<N>/bed-score.json` (the bed, unaided) and
                        `forgetting/<point>-a<N>/forgetting.json` (the three general panels), plus
                        `eval/base-<bed>-a<N>` and `forgetting/base-a<N>` for the untrained model
    --runs              K4's training runs, for the steps each point completed and its training signals
    --*-none-root       THE CONTROL, not rerun: a finished package whose runs already are the `none`
                        arm at the same dose from the same untrained model -- K3 (Spider stage A) and
                        K1c (FinQA part B). Only their scorings are read; nothing is retrained
    --finqa-rows        FinQA's prepared test rows, which carry `extra_info.gold_consistent`: the 8.3
                        percent of items whose own displayed answer disagrees with the value their
                        program executes to are EXCLUDED, because no model can be fairly judged on them
                        (kit/beds/finqa.py, receipt 208, PROCESS 3c)

THE VERDICT IS ONE NUMBER PER BED: the mean over five seeds of `hint` minus `none`, against the design
note's 3-point bar, written before any of these numbers existed.

EVERY NUMBER BELONGS TO A ROUTE, and the report says which. A hint can reach the student's prompt
(GRPO, `kit/run_grpo.sh`) or only the teacher's (SDPO, `kit/run_sdpo_bed.sh`), and each route carries
its own no-hint control, so each route's effect is a difference within ONE trainer:

    GRPO route   `hint` minus `none`                  -- the verdict
    SDPO route   `teacher-hint` minus `teacher-none`  -- one trainer key apart
    the route    `teacher-none` minus `none`          -- SDPO instead of GRPO, with no hint anywhere;
    change                                               a trainer effect, never a hint effect

`teacher-hint` minus `none` crosses both, so it is printed as what it is and read as neither.

THREE RULES IT ENFORCES, because each has already cost this programme a wrong answer:

1. ONE MACHINE. A held-out count from one machine cannot be subtracted from a count on another: across
   machines a panel moves by up to 3 points before any training (receipt 204), which is the size of the
   effect being measured. Here the risk is sharper than usual, because the control was scored in a
   DIFFERENT PACKAGE: if K3, K1c and K4 did not run in the same container on the same machine, the
   fingerprints differ and this file REFUSES unless --allow-different-machines, which it records.
2. A DIFFERENCE IS ONLY READ WITHIN ITS ROUTE. Every arm's gain against `none` is printed in one
   column, but the two SDPO arms' gains against it also change the trainer, so they carry
   `controlled_comparison: 0` and the SDPO route's own difference is computed separately.
   docs/phase2/k4/feasibility.md (d).
3. NO SILENT HOLES. A missing scoring, a missing control seed, a split over no shared questions, a run
   whose recorded feedback switch disagrees with the arm it is filed under: each is a FLAG in the
   report with the path that was looked for, never a dash that reads like a zero.

Standard library only. Nothing is overwritten: an existing --out is refused.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-k4-report.v1"
NONE_ARM = "none"
ARMS = ("none", "hint", "hint-faded", "teacher-none", "teacher-hint")
NEW_ARMS = ("hint", "hint-faded", "teacher-none", "teacher-hint")
BEDS = ("spider", "finqa")
#: Which trainer each arm ran, and where a hint given on that route would be read. An arm is only ever
#: subtracted from an arm on its own route.
ARM_ROUTE = {"none": "grpo", "hint": "grpo", "hint-faded": "grpo",
             "teacher-none": "sdpo", "teacher-hint": "sdpo"}
ROUTE_LABEL = {"grpo": "GRPO (kit/run_grpo.sh): a hint is read by the STUDENT",
               "sdpo": "SDPO (kit/run_sdpo_bed.sh): a hint is read by the TEACHER only"}
#: The three differences worth a line, in order. The first is the verdict; the second is the same
#: question on the other route; the third is the route change itself, with no hint on either side, and
#: it is a TRAINER effect -- naming it here is what stops it being read as a hint effect.
ROUTE_EFFECTS = (
    {"name": "grpo", "route": "grpo", "treated": "hint", "control": "none",
     "effect": "the hint, in the student's prompt", "controlled_comparison": 1},
    {"name": "sdpo", "route": "sdpo", "treated": "teacher-hint", "control": "teacher-none",
     "effect": "the hint, in the teacher's prompt only", "controlled_comparison": 1},
    {"name": "trainer-change", "route": "grpo -> sdpo", "treated": "teacher-none", "control": "none",
     "effect": "SDPO instead of GRPO, with no hint on either side", "controlled_comparison": 1},
)
#: What each arm's `feedback` field must say for it to be the arm it is filed under. The two SDPO arms
#: are one trainer key apart, so a control that ran with the switch on would look like a treatment.
ARM_FEEDBACK = {"teacher-none": 0, "teacher-hint": 1}
#: The design note's bar, in accuracy: at least 3 points on the mean of five seeds, with no larger loss
#: on the general panel. Written before any number existed; it is not a knob.
GAIN_BAR = 0.03
PANEL_FLOOR = -3
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
POINT = re.compile(r"^(?P<bed>spider|finqa)-(?P<arm>hint-faded|teacher-hint|teacher-none|hint)"
                   r"-seed(?P<seed>\d+)$")
#: Where each bed's `none` arm lives inside the package that already ran it, and what the untrained
#: model's scoring is called there. These are the other packages' own names, not ours: K3 writes
#: `eval/a-seed0-spider-a1` and `forgetting/a-seed0-a1`; K1c writes `eval/finqa-seed0-finqa-a1` and
#: `forgetting/finqa-seed0-forget-a1`.
NONE_TREES = {
    "spider": {"package": "K3", "eval": "a-seed%d-spider", "forget": ("a-seed%d",),
               "base_eval": "base-spider", "base_forget": ("base",)},
    "finqa": {"package": "K1c", "eval": "finqa-seed%d-finqa", "forget": ("finqa-seed%d-forget",),
              "base_eval": "base17b-finqa", "base_forget": ("base17b",)},
}
TRAIN_KEYS = {"score": "critic/score/mean", "response_tokens": "response_length/mean",
              "entropy": "actor/entropy", "grad_norm": "actor/grad_norm",
              "seconds_per_step": "perf/time_per_step",
              "max_memory_gb": "perf/max_memory_allocated_gb",
              "success_group_fraction": "self_distillation/success_group_fraction",
              "empty_target_batch": "self_distillation/empty_target_batch"}


class K4ReportError(ValueError):
    """The evidence tree cannot support a report; nothing is written."""


# ------------------------------------------------------------------------------------ small helpers
def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _round(value, digits=6):
    return round(value, digits) if _finite(value) else None


def _mean(values):
    clean = [v for v in values if _finite(v)]
    return _round(statistics.fmean(clean)) if clean else None


def spread(values: list) -> dict:
    """Mean and SAMPLE standard deviation; the spread of one seed is not a number, so it is null."""
    clean = [v for v in values if _finite(v)]
    return {"n": len(clean), "mean": _round(statistics.fmean(clean)) if clean else None,
            "sd": _round(statistics.stdev(clean)) if len(clean) > 1 else None,
            "min": _round(min(clean)) if clean else None, "max": _round(max(clean)) if clean else None}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path) -> list:
    """One object a line; `split` on newlines, never splitlines."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def latest_under(directory: Path, stem: str, filename: str):
    """(path, attempt) of the highest `<stem>-a<N>/<filename>` under a directory, or (None, None).

    The directory layout is the campaigns' own: every output folder is `<row's stem>-a<attempt>`, and
    a report that looked up a name the campaign does not write would render an empty table instead of
    a refusal (receipt 225, the K4a lesson). tests/test_kit_k4.py builds its tree FROM the campaign
    file's own OUT templates so the two cannot drift apart.
    """
    if not directory.is_dir():
        return None, None
    best = (None, None)
    for child in sorted(directory.iterdir()):
        match = ATTEMPT.match(child.name) if child.is_dir() else None
        if not match or match["stem"] != stem:
            continue
        attempt = int(match["attempt"])
        if (child / filename).is_file() and (best[1] is None or attempt > best[1]):
            best = (child / filename, attempt)
    return best


# --------------------------------------------------------------------------- the two measurements
def bed_score(directory: Path, stem: str) -> dict:
    """One `bed-score.json`, as the readout needs it: counts, the per-item verdicts, the machine."""
    path, attempt = latest_under(directory, stem, "bed-score.json")
    if path is None:
        return {"found": 0, "looked_for": str(directory / ("%s-a<N>" % stem)) + "/bed-score.json"}
    result = read_json(path) or {}
    return {"found": 1, "path": str(path), "attempt": attempt, "n": result.get("n"),
            "correct": result.get("correct"), "accuracy": result.get("accuracy"),
            "incorrect_format": result.get("incorrect_format"),
            "per_item": result.get("per_item") or {},
            "machine": ((result.get("machine") or {}).get("id")),
            "model": result.get("model")}


def panels(directory: Path, stems: tuple) -> dict:
    """One `forgetting.json`: {panel: correct}, the total, and the machine fingerprint."""
    for stem in stems:
        path, attempt = latest_under(directory, stem, "forgetting.json")
        if path is None:
            continue
        result = read_json(path) or {}
        return {"found": 1, "path": str(path), "attempt": attempt,
                "panels": {name: block.get("correct") for name, block in
                           sorted((result.get("panels") or {}).items())},
                "total_correct": result.get("total_correct"),
                "machine": ((result.get("machine") or {}).get("id"))}
    return {"found": 0, "looked_for": ", ".join(str(directory / ("%s-a<N>" % stem)) for stem in stems)}


def accuracy_on(per_item: dict, keep: set | None) -> dict:
    """The accuracy over a subset of the items, recomputed from the per-item verdicts."""
    items = {k: v for k, v in (per_item or {}).items() if keep is None or k in keep}
    if not items:
        return {"n": 0, "correct": 0, "accuracy": None}
    correct = sum(1 for value in items.values() if float(value) >= 1.0)
    return {"n": len(items), "correct": correct, "accuracy": _round(correct / len(items))}


def read_run(runs: Path, point: str) -> dict:
    """The training run behind a point: which arm it declared, how many steps it took, its signals."""
    path, attempt = latest_under(runs, point, "train-summary.json")
    if path is None:
        return {"found": 0, "looked_for": str(runs / ("%s-a<N>" % point)) + "/train-summary.json"}
    summary = read_json(path) or {}
    metrics = path.parent / "metrics.jsonl"
    training = []
    for record in (read_jsonl(metrics) if metrics.is_file() else []):
        data = record.get("data") or {}
        if TRAIN_KEYS["score"] in data:
            training.append({"step": record.get("step"),
                             **{key: _round(data.get(name)) for key, name in TRAIN_KEYS.items()}})
    return {"found": 1, "path": str(path), "attempt": attempt, "steps": summary.get("steps"),
            "returncode": summary.get("returncode"), "merged": summary.get("merged"),
            "arm_declared": summary.get("arm"), "hints_served": summary.get("hints"),
            "feedback_declared": summary.get("feedback"),
            "train_file": summary.get("train_file"), "schema": summary.get("schema"),
            "steps_measured": len(training),
            "means": {key: _mean([step[key] for step in training]) for key in TRAIN_KEYS},
            "first_two_steps": {key: _mean([step[key] for step in training[:2]]) for key in TRAIN_KEYS},
            "max_memory_gb": max([step["max_memory_gb"] for step in training
                                  if _finite(step["max_memory_gb"])], default=None)}


# ------------------------------------------------------------------------------------ the build
def gold_consistent_ids(path) -> set | None:
    """FinQA's flagged items: the ids whose gold DOES score, which are the only ones to judge on."""
    if not path:
        return None
    rows = read_jsonl(Path(path))
    keep = {str((row.get("extra_info") or {}).get("index"))
            for row in rows if (row.get("extra_info") or {}).get("gold_consistent")}
    keep.discard("None")
    return keep or None


def seed_row(bed: str, arm: str, seed: int, *, root: Path, runs: Path, none_root: Path | None,
             keep: set | None) -> dict:
    """One bed, arm and seed: where its numbers came from, and the numbers."""
    tree = NONE_TREES[bed]
    missing_tree = None
    if arm == NONE_ARM:
        if none_root is None:
            missing_tree = ("NO CONTROL TREE: pass --%s-none-root with the finished %s package, whose "
                            "runs ARE this arm. Nothing here is a substitute for it."
                            % (bed, tree["package"]))
            evaluation = {"found": 0, "looked_for": "--%s-none-root was not given" % bed}
            block = {"found": 0, "looked_for": "--%s-none-root was not given" % bed}
        else:
            evaluation = bed_score(none_root / "eval", tree["eval"] % seed)
            block = panels(none_root / "forgetting", tuple(stem % seed for stem in tree["forget"]))
        run = {"found": 0, "reused": 1}
    else:
        point = "%s-%s-seed%d" % (bed, arm, seed)
        evaluation = bed_score(root / "eval", point)
        block = panels(root / "forgetting", (point,))
        run = read_run(runs, point)
        # `hint-faded` is two runs: the hinted half and the plain half that restarts from it. The
        # second one's own step count is half the arm's dose, so the total is carried explicitly --
        # a table showing 10 where the arm trained 20 would read as a shorter arm, not a faded one.
        run["steps_total"] = run.get("steps")
        if arm == "hint-faded":
            first = read_run(runs, "%s-first" % point)
            run["first_half"] = first
            if first.get("found") and _finite(first.get("steps")) and _finite(run.get("steps")):
                run["steps_total"] = first["steps"] + run["steps"]
            elif not first.get("found"):
                run["steps_total"] = None
    row = {"seed": seed, "arm": arm, "bed": bed, "route": ARM_ROUTE[arm], "eval": evaluation,
           "panels": block, "run": run, "flags": []}
    if arm == NONE_ARM:
        row["reused_from"] = tree["package"]
    # The two SDPO arms are one trainer key apart, so the run must say which position it ran at. A
    # control that ran with the switch on is not a control, and a table cannot show that by itself.
    if arm in ARM_FEEDBACK and run.get("found"):
        wanted = ARM_FEEDBACK[arm]
        if run.get("feedback_declared") != wanted:
            row["flags"].append("WRONG FEEDBACK SWITCH: this run records feedback=%s, and `%s` is the "
                                "arm the switch at %d makes. %s"
                                % (run.get("feedback_declared"), arm, wanted, run.get("path")))
        if run.get("arm_declared") not in (None, arm):
            row["flags"].append("WRONG ARM: the run under this point's name declares itself `%s`. %s"
                                % (run.get("arm_declared"), run.get("path")))
    if missing_tree:
        row["flags"].append(missing_tree)
    elif not evaluation.get("found"):
        row["flags"].append("NO HELD-OUT SCORING: looked for %s" % evaluation.get("looked_for"))
    if not block.get("found") and not missing_tree:
        row["flags"].append("NO PANEL SCORING: looked for %s" % block.get("looked_for"))
    if arm != NONE_ARM and not run.get("found"):
        row["flags"].append("NO TRAINING SUMMARY: looked for %s" % run.get("looked_for"))
    row["judged"] = accuracy_on(evaluation.get("per_item"), keep)
    row["all_items"] = accuracy_on(evaluation.get("per_item"), None)
    if keep is not None and row["all_items"]["n"] and not row["judged"]["n"]:
        row["flags"].append("NO JUDGED ITEMS: none of the %d scored items is in the excluded-items "
                            "list, so the two files name items differently" % row["all_items"]["n"])
    return row


def split_by_untrained(row: dict, base: dict, keep: set | None) -> dict | None:
    """The package's own question: did the hint rescue what the untrained model could not do?

    Held-out items are split by the UNTRAINED model's verdict at step 0 -- `stuck` is every item it
    got wrong -- exactly as K4a split them by what its control could do. Both sides are the same
    per-item verdicts from kit/eval_bed.py, keyed by the bed's own question id, so they name the same
    question; only items present on both sides and not excluded are counted.
    """
    theirs, ours = base.get("per_item") or {}, row["eval"].get("per_item") or {}
    if not theirs or not ours:
        return None
    shared = sorted(set(theirs) & set(ours) & (keep if keep is not None else set(theirs) | set(ours)))
    stuck = [q for q in shared if float(theirs[q]) < 1.0]
    solved = [q for q in shared if float(theirs[q]) >= 1.0]
    return {"shared_items": len(shared),
            "stuck_items": len(stuck), "stuck_accuracy_now": _mean([float(ours[q]) for q in stuck]),
            "already_solved_items": len(solved),
            "already_solved_accuracy_now": _mean([float(ours[q]) for q in solved]),
            "untrained_stuck_accuracy": 0.0,
            "untrained_already_solved_accuracy": 1.0}


def paired_difference(block: dict, treated: str, control: str, seeds) -> tuple:
    """One arm minus another, seed for seed: `({seed: difference}, spread)`.

    Paired, never a difference of means: the seeds are the same five runs on both sides, and a seed
    whose scoring is missing on either side contributes nothing rather than a zero.
    """
    per_seed = {}
    for seed in seeds:
        mine = block["arms"][treated]["seeds"][str(seed)]["judged"]["accuracy"]
        theirs = block["arms"][control]["seeds"][str(seed)]["judged"]["accuracy"]
        per_seed[str(seed)] = _round(mine - theirs) if _finite(mine) and _finite(theirs) else None
    return per_seed, spread(list(per_seed.values()))


def fingerprints(rows: list, base: dict, panel_base: dict) -> list:
    seen = {row["eval"].get("machine") for row in rows} | {row["panels"].get("machine") for row in rows}
    seen |= {base.get("machine"), panel_base.get("machine")}
    seen.discard(None)
    return sorted(seen)


def build(root: Path, runs: Path, *, none_roots: dict, finqa_rows=None, seeds=(0, 1, 2, 3, 4),
          allow_different_machines: bool = False) -> dict:
    root, runs = Path(root), Path(runs)
    keep_by_bed = {"spider": None, "finqa": gold_consistent_ids(finqa_rows)}
    report = {"schema": SCHEMA,
              "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "root": str(root.resolve()), "runs_dir": str(runs.resolve()),
              "none_roots": {bed: (str(Path(path).resolve()) if path else None)
                             for bed, path in none_roots.items()},
              "gain_bar": GAIN_BAR, "panel_floor": PANEL_FLOOR, "seeds": list(seeds),
              "finqa_rows": str(Path(finqa_rows).resolve()) if finqa_rows else None,
              "beds": {}, "flags": []}
    every_row, base_blocks = [], {}
    for bed in BEDS:
        keep = keep_by_bed[bed]
        tree = NONE_TREES[bed]
        none_root = Path(none_roots[bed]) if none_roots.get(bed) else None
        base_eval = bed_score(root / "eval", "base-%s" % bed)
        base_panels = panels(root / "forgetting", ("base",))
        base_blocks[bed] = (base_eval, base_panels)
        block = {"bed": bed, "reused_control_from": tree["package"],
                 "untrained": {"eval": base_eval, "panels": base_panels,
                               "judged": accuracy_on(base_eval.get("per_item"), keep),
                               "all_items": accuracy_on(base_eval.get("per_item"), None)},
                 "excluded_items": (None if keep is None
                                    else (base_eval.get("n") or 0) - len(
                                        set(keep) & set(base_eval.get("per_item") or {}))),
                 "arms": {}}
        if not base_eval.get("found"):
            report["flags"].append("%s: no untrained held-out scoring, so the stuck/solved split "
                                   "cannot be made (looked for %s)" % (bed, base_eval.get("looked_for")))
        for arm in ARMS:
            rows = []
            for seed in seeds:
                row = seed_row(bed, arm, seed, root=root, runs=runs, none_root=none_root, keep=keep)
                row["split"] = split_by_untrained(row, base_eval, keep)
                if row["split"] is not None and not row["split"]["shared_items"]:
                    row["flags"].append("NO SHARED ITEMS with the untrained scoring: the two files "
                                        "name held-out questions differently, so the split below is "
                                        "empty rather than zero")
                if row["panels"].get("found") and base_panels.get("found"):
                    row["panel_change"] = {name: row["panels"]["panels"][name] - base_panels["panels"][name]
                                           for name in sorted(set(row["panels"]["panels"])
                                                              & set(base_panels["panels"]))}
                    row["worst_panel_change"] = min(row["panel_change"].values()) if row["panel_change"] else None
                else:
                    row["panel_change"], row["worst_panel_change"] = {}, None
                rows.append(row)
                every_row.append(row)
            block["arms"][arm] = {
                "seeds": {str(row["seed"]): row for row in rows},
                "accuracy": spread([row["judged"]["accuracy"] for row in rows]),
                "correct": spread([row["judged"]["correct"] for row in rows]),
                "accuracy_all_items": spread([row["all_items"]["accuracy"] for row in rows]),
                "stuck_accuracy": spread([(row["split"] or {}).get("stuck_accuracy_now") for row in rows]),
                "already_solved_accuracy": spread([(row["split"] or {}).get("already_solved_accuracy_now")
                                                   for row in rows]),
                "worst_panel_change": spread([row["worst_panel_change"] for row in rows]),
                "panel_change": {name: spread([row["panel_change"].get(name) for row in rows])
                                 for name in sorted({n for row in rows for n in row["panel_change"]})},
                "seeds_scored": sum(1 for row in rows if row["eval"].get("found")),
            }
        control = block["arms"][NONE_ARM]["accuracy"]["mean"]
        for arm in NEW_ARMS:
            arm_block = block["arms"][arm]
            per_seed, paired = paired_difference(block, arm, NONE_ARM, seeds)
            arm_block["gain_per_seed"] = per_seed
            arm_block["gain_paired"] = paired
            arm_block["gain_of_means"] = (_round(arm_block["accuracy"]["mean"] - control)
                                          if _finite(arm_block["accuracy"]["mean"]) and _finite(control)
                                          else None)
            # The column above is every arm against the GRPO control. For an SDPO arm that difference
            # also changes the trainer, so it is not a controlled comparison and the SDPO route's own
            # difference is the one below.
            arm_block["controlled_comparison"] = int(ARM_ROUTE[arm] == ARM_ROUTE[NONE_ARM])
            arm_block["compared_against"] = NONE_ARM
        for arm in ARMS:
            block["arms"][arm]["route"] = ARM_ROUTE[arm]
            block["arms"][arm]["route_label"] = ROUTE_LABEL[ARM_ROUTE[arm]]
        block["routes"] = {}
        for effect in ROUTE_EFFECTS:
            per_seed, paired = paired_difference(block, effect["treated"], effect["control"], seeds)
            block["routes"][effect["name"]] = {**effect, "per_seed": per_seed, "paired": paired,
                                               "is_the_verdict": int(effect["name"] == "grpo")}
        block["verdict"] = verdict_of(block)
        report["beds"][bed] = block
    machines = sorted({m for bed in BEDS
                       for m in fingerprints([r for r in every_row if r["bed"] == bed],
                                             base_blocks[bed][0], base_blocks[bed][1])})
    comparable = len(machines) == 1
    if machines and not comparable and not allow_different_machines:
        raise K4ReportError(
            "the scorings read here carry %d different machine-and-mode fingerprints (%s). A held-out "
            "count from one machine cannot be subtracted from a count on another: a panel moves by up "
            "to 3 points across machines before any training, which is the size of the effect K4 "
            "measures. The `none` arm was scored in another package, so this is the thing to check "
            "first: K3, K1c and K4 must have run in the same container on the same machine. Re-score "
            "on one machine, or pass --allow-different-machines and say so in the readout."
            % (len(machines), ", ".join(machines)))
    report["machine_ids"] = machines
    report["comparable"] = int(comparable)
    report["different_machines_allowed"] = int(bool(allow_different_machines))
    report["beds_reported"] = sum(1 for bed in BEDS
                                 if report["beds"][bed]["arms"][NONE_ARM]["seeds_scored"]
                                 and any(report["beds"][bed]["arms"][arm]["seeds_scored"] for arm in NEW_ARMS))
    report["arms_reported"] = len({arm for bed in BEDS for arm in ARMS
                                   if report["beds"][bed]["arms"][arm]["seeds_scored"]})
    report["none_runs"] = sum(report["beds"][bed]["arms"][NONE_ARM]["seeds_scored"] for bed in BEDS)
    report["flags"] += ["%s %s seed %s: %s" % (row["bed"], row["arm"], row["seed"], flag)
                        for row in every_row for flag in row["flags"]]
    return report


def verdict_of(block: dict) -> dict:
    """The design note's verdict for one bed: `hint` minus `none` against the 3-point bar.

    The bar is on the MEAN OF FIVE SEEDS of the paired difference, and it is a bar on `hint` only:
    `hint-faded` answers a second question (does what the hint taught survive its removal?) and
    `teacher-hint` is not a controlled comparison at all.
    """
    hint = block["arms"]["hint"]
    gain = hint["gain_paired"]["mean"]
    worst = hint["worst_panel_change"]["mean"]
    if not _finite(gain) or hint["gain_paired"]["n"] < 1:
        return {"verdict": "NO VERDICT", "reason": "the paired gain could not be computed for any seed",
                "gain": None, "seeds": hint["gain_paired"]["n"]}
    panel_ok = (not _finite(worst)) or worst >= PANEL_FLOOR
    met = gain >= GAIN_BAR and panel_ok
    return {"verdict": "BAR MET" if met else "BAR NOT MET", "gain": gain,
            "seeds": hint["gain_paired"]["n"], "bar": GAIN_BAR,
            "worst_panel_change_mean": worst, "panel_floor": PANEL_FLOOR,
            "panel_ok": int(bool(panel_ok)),
            "reason": None if met else
                      ("the mean gain is %+.4f, under the %.2f bar" % (gain, GAIN_BAR)
                       if gain < GAIN_BAR else
                       "the mean gain is %+.4f but the worst panel fell by %.1f, past the floor of %d"
                       % (gain, worst, PANEL_FLOOR))}


# ------------------------------------------------------------------------------------ rendering
def _fmt(value, pattern="%.4f"):
    return pattern % value if _finite(value) else "-"


def render(report: dict) -> str:
    lines = ["# K4: does a hint on the stuck questions teach the model to answer them unaided?", "",
             "Generated %s by kit/k4_report.py (%s). Every number is recomputed from the raw files."
             % (report["generated_at"], SCHEMA), "",
             "The `none` arm is NOT rerun here: it is %s."
             % ", ".join("%s for %s" % (NONE_TREES[bed]["package"], bed) for bed in BEDS)
             + " Every accuracy is `kit/eval_bed.py` on the bed's held-out questions with NO hint in "
               "the prompt, scored on one GPU of one machine in the deterministic mode.", ""]
    if not report["comparable"]:
        lines += ["**The scorings do not share one machine-and-mode fingerprint (%s), so the "
                  "differences below are not comparable. This is what to check first when the control "
                  "comes from another package.**" % ", ".join(report["machine_ids"]), ""]
    lines += ["## The verdict, one line a bed", "",
              "The bar, written before any of these numbers existed: `hint` beats `none` by at least "
              "%.0f points on the mean of five seeds, with no larger loss on the general panel "
              "(floor %d per 100). Both arms are GRPO, so this is the GRPO route's own difference."
              % (GAIN_BAR * 100, PANEL_FLOOR), "",
              "| bed | `hint` - `none` (paired, 5 seeds) | worst panel | verdict |", "|---|---|---|---|"]
    for bed in BEDS:
        block = report["beds"][bed]
        verdict = block["verdict"]
        paired = block["arms"]["hint"]["gain_paired"]
        lines.append("| %s | %s (sd %s, n %d) | %s | **%s**%s |" % (
            bed, _fmt(paired["mean"], "%+.4f"), _fmt(paired["sd"]), paired["n"],
            _fmt(verdict.get("worst_panel_change_mean"), "%+.1f"), verdict["verdict"],
            "" if not verdict.get("reason") else " -- %s" % verdict["reason"]))

    lines += ["", "## Which route does each number belong to?", "",
              "A hint can be read by the student (GRPO, the control's own trainer) or by the teacher "
              "alone (SDPO). Each route has its own no-hint control, so each route's effect is a "
              "difference within one trainer. The last line is the change of trainer itself, with no "
              "hint on either side: it is a trainer effect and never a hint effect.", "",
              "| bed | route | difference | what it measures | paired mean (5 seeds) | sd | n |",
              "|---|---|---|---|---|---|---|"]
    for bed in BEDS:
        for effect in ROUTE_EFFECTS:
            line = report["beds"][bed]["routes"][effect["name"]]
            paired = line["paired"]
            lines.append("| %s | %s | `%s` - `%s` | %s | %s%s | %s | %d |" % (
                bed, line["route"], line["treated"], line["control"], line["effect"],
                _fmt(paired["mean"], "%+.4f"), " **(the verdict)**" if line["is_the_verdict"] else "",
                _fmt(paired["sd"]), paired["n"]))
    lines += ["", "`teacher-hint` minus `none` crosses both routes -- the hint AND GRPO to SDPO -- so "
              "it is printed in the per-bed tables below for completeness and is not one of the "
              "effects above."]

    for bed in BEDS:
        block = report["beds"][bed]
        untrained = block["untrained"]
        lines += ["", "## %s" % bed.upper(), "",
                  "Untrained Qwen3-1.7B: %s of %s items%s. Control: %s, reused."
                  % (_fmt(untrained["judged"]["correct"], "%s"), _fmt(untrained["judged"]["n"], "%s"),
                     "" if block["excluded_items"] in (None, 0)
                     else " (%d items excluded: FinQA's own answer disagrees with its program)"
                          % block["excluded_items"],
                     block["reused_control_from"]), "",
                  "| arm | route | seed | correct | accuracy | gain vs none | worst panel | steps |",
                  "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            arm_block = block["arms"][arm]
            route = ARM_ROUTE[arm]
            for seed in report["seeds"]:
                row = arm_block["seeds"][str(seed)]
                gain = (arm_block.get("gain_per_seed") or {}).get(str(seed))
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    arm, route, seed, _fmt(row["judged"]["correct"], "%s"),
                    _fmt(row["judged"]["accuracy"]),
                    _fmt(gain, "%+.4f"), _fmt(row["worst_panel_change"], "%+d"),
                    _fmt((row["run"] or {}).get("steps_total"), "%s")))
            lines.append("| **%s** | %s | mean of %d | %s | **%s** (sd %s) | **%s** (sd %s) | %s | |" % (
                arm, route, arm_block["accuracy"]["n"], _fmt(arm_block["correct"]["mean"], "%.1f"),
                _fmt(arm_block["accuracy"]["mean"]), _fmt(arm_block["accuracy"]["sd"]),
                _fmt((arm_block.get("gain_paired") or {}).get("mean"), "%+.4f"),
                _fmt((arm_block.get("gain_paired") or {}).get("sd")),
                _fmt(arm_block["worst_panel_change"]["mean"], "%+.1f")))
        lines += ["", "The `gain vs none` column is every arm against the GRPO control. For the two "
                  "SDPO arms that difference changes the trainer (SDPO, with a teacher) as well as "
                  "the hint, so it is not a controlled comparison and is not the verdict: the hint's "
                  "effect on that route is `teacher-hint` minus `teacher-none`, %s here, in the route "
                  "table above." % _fmt(block["routes"]["sdpo"]["paired"]["mean"], "%+.4f"), "",
                  "### Rescued without hurting the rest?", "",
                  "Held-out items split by what the UNTRAINED model could do. `stuck` is every item it "
                  "got wrong.", "",
                  "| arm | stuck items | this arm on them | already solved | this arm on those |",
                  "|---|---|---|---|---|"]
        for arm in ARMS:
            arm_block = block["arms"][arm]
            first = next((arm_block["seeds"][str(seed)]["split"] for seed in report["seeds"]
                          if arm_block["seeds"][str(seed)]["split"]), None)
            lines.append("| %s | %s | %s (sd %s) | %s | %s (sd %s) |" % (
                arm, _fmt((first or {}).get("stuck_items"), "%s"),
                _fmt(arm_block["stuck_accuracy"]["mean"]), _fmt(arm_block["stuck_accuracy"]["sd"]),
                _fmt((first or {}).get("already_solved_items"), "%s"),
                _fmt(arm_block["already_solved_accuracy"]["mean"]),
                _fmt(arm_block["already_solved_accuracy"]["sd"])))
        names = sorted(block["untrained"]["panels"].get("panels") or {})
        if names:
            lines += ["", "### The general panel (a trained model may lose at most %d per 100)" % -PANEL_FLOOR, "",
                      "Untrained: " + ", ".join("%s %s" % (name, block["untrained"]["panels"]["panels"][name])
                                                for name in names), "",
                      "| arm | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
            for arm in ARMS:
                arm_block = block["arms"][arm]
                lines.append("| %s | %s |" % (arm, " | ".join(
                    _fmt((arm_block["panel_change"].get(name) or {}).get("mean"), "%+.1f")
                    for name in names)))
    if report["flags"]:
        lines += ["", "## Flags", ""] + ["- %s" % flag for flag in report["flags"]]
    lines += ["", "%d beds, %d arms, %d control runs read from other packages."
              % (report["beds_reported"], report["arms_reported"], report["none_runs"]), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="K4's own tree: eval/ and forgetting/")
    parser.add_argument("--runs", type=Path, required=True, help="K4's training runs")
    parser.add_argument("--spider-none-root", default=None,
                        help="the finished K3 package directory: Spider's `none` arm")
    parser.add_argument("--finqa-none-root", default=None,
                        help="the finished K1c package directory: FinQA's `none` arm")
    parser.add_argument("--finqa-rows", default=None,
                        help="FinQA's prepared test.jsonl, to exclude the items whose gold does not score")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", type=Path, required=True, help="a new directory; never overwritten")
    parser.add_argument("--allow-different-machines", action="store_true",
                        help="report anyway; recorded in the report as the departure it is")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % args.out)
    seeds = tuple(int(part) for part in str(args.seeds).split(",") if part.strip())
    try:
        report = build(args.root, args.runs,
                       none_roots={"spider": args.spider_none_root, "finqa": args.finqa_none_root},
                       finqa_rows=args.finqa_rows, seeds=seeds,
                       allow_different_machines=args.allow_different_machines)
    except K4ReportError as exc:
        raise SystemExit(str(exc))
    args.out.mkdir(parents=True)
    (args.out / "k4-report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    text = render(report)
    (args.out / "k4-report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
