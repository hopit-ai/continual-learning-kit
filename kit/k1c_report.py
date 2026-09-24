#!/usr/bin/env python3
"""The K1c readout: what each job looks like when it is learned ALONE, with reward-only GRPO.

    python k1c_report.py --runs $WORK/runs --forgetting $WORK/k1c/forgetting --eval $WORK/k1c/eval \\
                         --k0 /work/k0-report/report.json --out $WORK/k1c/report-a1
    #  --k3 /work/k3/report-a1/k3-report.json   OPTIONAL: adds K3's SQL-alone rows

Two parts, read from the folders the campaign wrote:

    PART A, Qwen3-8B on ToolAlpaca. `--runs`/<arm>-seed<S>-aN with `run-summary.json` and
    `metrics.jsonl` from kit/run_grpo_toolalpaca.sh, plus `hf-step40/fold.json` on a LoRA row, and
    `--forgetting`/<arm>-seed<S>-aN/forgetting.json. The arms are `full` and `lora`.

    PART B, Qwen3-1.7B. `--runs`/<job>-seed<S>-aN with `train-summary.json` and `metrics.jsonl` from
    kit/run_grpo.sh, `--eval`/<job>-seed<S>-<bed>-aN/bed-score.json, and the same forgetting folder.
    The jobs are `gsm8k` and `finqa`.

    THE COMPARATOR, `--k0`: the partner's finished K0 report. Its dose40-seed42/43/44 SDPO runs are
    what Part A is read BESIDE, seed for seed. They are a comparator and not a control: GRPO and SDPO
    are two methods at the same dose on the same data, and the point of this package is to put the
    reward-only number next to the self-distilled one, not to declare a winner.

    SQL ALONE, `--k3`: K3's stage A already IS SQL learned alone at 1.7B, so it is named rather than
    re-measured. Without --k3 the SQL row simply says where to find it.

FOUR RULES IT ENFORCES, each of which has already cost this programme a wrong answer:

1. ONE MACHINE. Panel and bed counts are comparable only when scored on the same machine in the same
   decoding mode: across machines a panel moves by up to 3 points before any training (receipt 204).
   Mixed fingerprints are REFUSED unless --allow-different-machines, which is recorded as the
   departure it is.
2. THE STRICT SCORE. Every Part A accuracy is `val-core/tooluse/acc/mean@16`, the authors'
   all-or-nothing tool-use score, recomputed per question from the `acc` field of the validation
   dumps. A disagreement with the trainer's own logged number is a FLAG, not a rounding difference.
3. A LORA ROW MUST HAVE BEEN FOLDED. `verl.model_merger merge` saves the BASE model and an adapter
   with lora_alpha 0 (base_model_merger.py:268,301-306), so a LoRA row whose `hf-step40` was never
   folded by kit/fold_lora.py would have had the UNTRAINED model scored for forgetting, and would
   read as "LoRA forgets nothing". Any LoRA run without `folded: 1` and a positive
   `fold_changed_tensors` is a FLAG at the top of the report, and its panel numbers are refused.
4. NOTHING IS A CONCLUSION HERE. These are baselines. Every table says what was measured and against
   what; the only comparisons drawn are LoRA against full at the same seed, and each arm against the
   SDPO run at the same seed.

Standard library only. Nothing is overwritten: an existing --out is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-k1c-report.v1"
HERE = Path(__file__).resolve().parent
VAL_KEY = "val-core/tooluse/acc/mean@16"
COMPARATOR_PREFIX = "dose40-seed"
A_ARMS = ("full", "lora")
B_JOBS = {"gsm8k": "gsm8k", "finqa": "finqa"}       # job -> the bed that scores it
A_POINT = re.compile(r"^(?P<arm>%s)-seed(?P<seed>\d+)$" % "|".join(A_ARMS))
B_POINT = re.compile(r"^(?P<job>%s)-seed(?P<seed>\d+)$" % "|".join(sorted(B_JOBS)))
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
TRAIN_KEYS = {
    "score": "critic/score/mean",
    "response_tokens": "response_length/mean",
    "entropy": "actor/entropy",
    "grad_norm": "actor/grad_norm",
    "seconds_per_step": "perf/time_per_step",
    "max_memory_gb": "perf/max_memory_allocated_gb",
}


class K1cReportError(ValueError):
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


def latest_dirs(directory: Path) -> dict:
    """{stem: path} for the highest attempt of every <stem>-aN directory."""
    found: dict = {}
    if not directory.is_dir():
        return found
    for child in sorted(directory.iterdir()):
        match = ATTEMPT.match(child.name) if child.is_dir() else None
        if not match:
            continue
        stem, attempt = match["stem"], int(match["attempt"])
        if stem not in found or attempt > found[stem][0]:
            found[stem] = (attempt, child)
    return {stem: path for stem, (_, path) in found.items()}


def latest_files_found(directory: Path, filename: str) -> dict:
    """{stem: (path, parsed json)} for the highest attempt of every <stem>-aN folder holding `filename`.

    The PATH is carried beside the result because kit/density.py needs it: a scoring's token counts
    may live in the `responses.jsonl` written beside it rather than inside it.
    """
    out = {}
    for stem, path in latest_dirs(directory).items():
        target = path / filename
        if target.is_file():
            try:
                out[stem] = (target, json.loads(target.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    return out


def latest_files(directory: Path, filename: str) -> dict:
    """{stem: parsed json} for the highest attempt of every <stem>-aN folder holding `filename`."""
    return {stem: result for stem, (_path, result) in latest_files_found(directory, filename).items()}


# ------------------------------------------------------- intelligence density (plan 4c, row Q13)
def _load_density():
    """kit/density.py, loaded by path from this file's own directory, the way kit/beds/rewards.py
    loads a bed: nothing here needs the kit installed, on sys.path or in the working directory. A
    copy of the kit without the file still reports -- every density is then unknown, and an unknown
    density is never a refusal."""
    path = HERE / "density.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("kit_density_for_k1c", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


density = _load_density()
DENSITY_BAR = density.BAR if density else 1.5
NO_TOKENS = ("No scoring read here carried token counts -- neither `output_tokens_total` in the "
             "result nor a `responses.jsonl` beside it -- so every figure in this table is `-`. "
             "Nothing else in this report depends on them.")
PART_A_NOTE = ("Part A is left out: ToolAlpaca is scored from the trainer's own validation dumps "
               "and writes no `bed-score.json`, so there is no held-out scoring here to read a "
               "token count from.")


def cost(found, *, panel=None):
    """Tokens per correct answer for one (path, result) pair, or None when its tokens are absent."""
    if density is None or not found:
        return None
    path, result = found
    try:
        return density.tokens_per_correct(result, Path(path), panel=panel)
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


# --------------------------------------------------------------------------- the strict measurements
def per_question_acc(path: Path) -> dict:
    """One dump -> {question id: mean STRICT accuracy over its samples}. Reads `acc`, never `score`.

    The id is the sha1 of the prompt, the same key kit/make_report.py:68 uses, so a question is named
    the same way on both sides of the K0 comparison.
    """
    groups: dict = defaultdict(list)
    for row in _jsonl(path):
        if "input" not in row or "acc" not in row:
            continue
        groups[hashlib.sha1(row["input"].encode()).hexdigest()[:12]].append(float(row["acc"]))
    return {qid: statistics.fmean(values) for qid, values in groups.items()}


def strict_success(path: Path) -> dict | None:
    """A rollout dump -> the STRICT share of questions with at least one correct attempt."""
    groups: dict = defaultdict(list)
    for row in _jsonl(path):
        if "input" not in row or "acc" not in row:
            continue
        groups[hashlib.sha1(row["input"].encode()).hexdigest()[:12]].append(float(row["acc"]))
    if not groups:
        return None
    attempts = [a for values in groups.values() for a in values]
    return {"questions": len(groups),
            "success_group_fraction": _round(
                statistics.fmean(1.0 if max(v) >= 1.0 else 0.0 for v in groups.values())),
            "attempt_accuracy": _round(statistics.fmean(attempts))}


def read_run(run: Path, *, summary_name: str) -> dict:
    """One run directory -> its summary, per-step training signals, validations and flags."""
    out: dict = {"name": run.name, "flags": [], "summary": {}}
    summary = run / summary_name
    if summary.is_file():
        try:
            out["summary"] = json.loads(summary.read_text())
        except json.JSONDecodeError:
            out["flags"].append("UNREADABLE %s" % summary_name)
    else:
        out["flags"].append("NO %s: this run wrote no summary, so what it ran is unrecorded" % summary_name)

    training, validations = [], []
    metrics = run / "metrics.jsonl"
    for record in (_jsonl(metrics) if metrics.is_file() else []):
        data = record.get("data") or {}
        if VAL_KEY in data:
            validations.append({"step": record.get("step"), "accuracy_logged": _round(data[VAL_KEY])})
        if TRAIN_KEYS["score"] in data:
            training.append({"step": record.get("step"),
                             **{k: _round(data.get(v)) for k, v in TRAIN_KEYS.items()}})
    if not metrics.is_file():
        out["flags"].append("NO metrics.jsonl: the per-step numbers below are missing. "
                            "kit/run_grpo.sh needs FILE_LOG=1 to write one")
    out["training"] = training
    out["steps_completed"] = max((t["step"] for t in training if _finite(t["step"])), default=0)
    out["means"] = {key: _mean([t[key] for t in training]) for key in TRAIN_KEYS}
    out["max_memory_gb"] = max([t["max_memory_gb"] for t in training
                                if _finite(t["max_memory_gb"])], default=None)

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
            if _finite(entry["accuracy_logged"]) and \
                    abs(entry["accuracy_logged"] - entry["accuracy_strict"]) > 1e-3:
                out["flags"].append(
                    "MISMATCH at step %s: the trainer logged %.4f but the per-question `acc` gives "
                    "%.4f. The validation metric is not the strict score."
                    % (entry["step"], entry["accuracy_logged"], entry["accuracy_strict"]))
    out["validations"] = validations
    out["per_question"] = {str(step): {q: _round(a) for q, a in found.items()}
                           for step, found in dumps.items()}

    rollouts = run / "rollouts"
    strict = [s for s in ([strict_success(p) for p in sorted(rollouts.glob("*.jsonl"))]
                          if rollouts.is_dir() else []) if s]
    out["strict_success"] = ({"steps_read": len(strict),
                              "success_group_fraction": _mean([s["success_group_fraction"] for s in strict]),
                              "attempt_accuracy": _mean([s["attempt_accuracy"] for s in strict])}
                             if strict else None)

    fold = sorted(run.glob("hf-step*/fold.json"))
    out["fold"] = json.loads(fold[-1].read_text()) if fold else None
    merged = sorted(run.glob("hf-step*/config.json"))
    out["merged_checkpoint"] = merged[-1].parent.name if merged else None
    return out


# ------------------------------------------------------------------------------------ the comparator
def read_comparator(path: Path) -> dict:
    """The K0 report's dose40-seed* SDPO runs, keyed by seed. Every number is the authors' strict score."""
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise K1cReportError("cannot read the K0 comparator report at %s: %s" % (path, exc))
    found = {}
    for run in report.get("runs") or []:
        name = run.get("name") or ""
        if not name.startswith(COMPARATOR_PREFIX):
            continue
        seed = name[len(COMPARATOR_PREFIX):]
        if not seed.isdigit():
            continue
        validations = run.get("validations") or []
        training = run.get("training") or []
        first = next((v for v in validations if v.get("step") == 0), None)
        found[int(seed)] = {
            "name": name,
            "base": _round((first or {}).get("accuracy_logged")),
            "final": _round((validations or [{}])[-1].get("accuracy_logged")),
            "validations": [{"step": v.get("step"), "accuracy": _round(v.get("accuracy_logged"))}
                            for v in validations],
            "means": {"response_tokens": _mean([t.get("response_tokens") for t in training]),
                      "entropy": _mean([t.get("entropy") for t in training]),
                      "score": _mean([t.get("score") for t in training]),
                      "seconds_per_step": _mean([t.get("seconds_per_step") for t in training]),
                      "max_memory_gb": max([t.get("max_memory_gb") for t in training
                                            if _finite(t.get("max_memory_gb"))], default=None)},
            "steps_completed": run.get("steps_completed"),
        }
        if _finite(found[int(seed)]["final"]) and _finite(found[int(seed)]["base"]):
            found[int(seed)]["change"] = _round(found[int(seed)]["final"] - found[int(seed)]["base"])
    if not found:
        raise K1cReportError(
            "the K0 report at %s holds no run named %s<seed>. Part A is read seed for seed beside the "
            "K0 dose-40 SDPO runs; without them there is nothing to put it next to."
            % (path, COMPARATOR_PREFIX))
    return found


def read_sql_alone(path: Path | None) -> dict | None:
    """K3's stage A: SQL learned alone at 1.7B. Named, never re-measured."""
    if path is None:
        return None
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise K1cReportError("cannot read the K3 report at %s: %s" % (path, exc))
    stages = report.get("after_stage_a") or {}
    base = (report.get("base") or {}).get("job_a")
    rows = {}
    for point, value in sorted(stages.items()):
        seed = point[len("a-seed"):] if point.startswith("a-seed") else None
        if seed and seed.isdigit():
            rows[int(seed)] = {"point": point, "job_a": value.get("job_a"),
                               "panels": value.get("panels") or {}}
    if not rows:
        raise K1cReportError("the K3 report at %s holds no a-seed<N> stage-A rows, which is what SQL "
                             "learned alone means here" % path)
    return {"report": str(Path(path).resolve()), "base_job_a": base, "seeds": rows,
            "learned": spread([r["job_a"] - base for r in rows.values()
                               if _finite(r["job_a"]) and _finite(base)]),
            "note": "Spider held-out 100, 20 GRPO steps (one pass over Spider's 640 questions). "
                    "Measured by K3, not re-run here."}


# ------------------------------------------------------------------------------------ the build
def panel_scores(result: dict | None) -> dict:
    if not result:
        return {}
    return {name: block["correct"] for name, block in sorted((result.get("panels") or {}).items())}


def fingerprints(*groups) -> list:
    seen = set()
    for group in groups:
        for result in group.values():
            seen.add((result.get("machine") or {}).get("id"))
    return sorted(seen, key=lambda item: (item is None, item))


def number(result: dict | None, key: str = "correct"):
    if not result:
        return None
    value = result.get(key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def build_part_a(runs: dict, panels: dict, comparator: dict) -> dict:
    base_panels = panel_scores(panels.get("base8b"))
    arms: dict = {}
    flags: list = []
    for point, run in sorted(runs.items()):
        match = A_POINT.match(point)
        if not match:
            continue
        arm, seed = match["arm"], int(match["seed"])
        summary = run["summary"]
        after = panel_scores(panels.get("%s-forget" % point))
        validations = run["validations"]
        final = ((validations or [{}])[-1].get("accuracy_strict")
                 or (validations or [{}])[-1].get("accuracy_logged"))
        first = next((v for v in validations if v.get("step") == 0), {})
        base_score = first.get("accuracy_strict") or first.get("accuracy_logged")
        row = {
            "run": run["name"], "seed": seed, "steps_completed": run["steps_completed"],
            "arm_declared": summary.get("arm"), "lora": summary.get("lora"),
            "learning_rate": summary.get("learning_rate"),
            "validations": validations, "base": base_score, "final": final,
            "change": _round(final - base_score) if _finite(final) and _finite(base_score) else None,
            "means": run["means"], "max_memory_gb": run["max_memory_gb"],
            "strict_success": run["strict_success"],
            "merged_checkpoint": run["merged_checkpoint"],
            "fold": run["fold"], "panels": after,
            "flags": list(run["flags"]),
            "comparator": comparator.get(seed),
        }
        # RULE 3: a LoRA row whose adapter never reached the weights would have had the UNTRAINED model
        # scored for forgetting. Its panels are refused rather than printed.
        if arm == "lora":
            folded = summary.get("folded")
            changed = summary.get("fold_changed_tensors")
            if not (folded == 1 and _finite(changed) and changed > 0):
                row["flags"].append(
                    "NOT FOLDED: run-summary.json says folded=%r, fold_changed_tensors=%r. "
                    "verl.model_merger leaves the BASE model behind with an adapter whose lora_alpha "
                    "is 0, so anything scored from this run is the UNTRAINED model. Its panels are "
                    "withheld from the tables below." % (folded, changed))
                row["panels"] = {}
                flags.append("%s: %s" % (point, row["flags"][-1]))
            elif run["fold"] and run["fold"].get("flags"):
                row["flags"] += ["fold: %s" % f for f in run["fold"]["flags"]]
        row["panel_change"] = {name: row["panels"][name] - base_panels[name]
                               for name in sorted(set(row["panels"]) & set(base_panels))}
        if row["comparator"] and _finite(final) and _finite(row["comparator"]["final"]):
            row["vs_sdpo"] = _round(final - row["comparator"]["final"])
        arms.setdefault(arm, {"seeds": {}})["seeds"][seed] = row

    for arm, block in arms.items():
        ordered = [block["seeds"][s] for s in sorted(block["seeds"])]
        block["final"] = spread([r["final"] for r in ordered])
        block["change"] = spread([r["change"] for r in ordered])
        block["vs_sdpo"] = spread([r.get("vs_sdpo") for r in ordered])
        block["seconds_per_step"] = spread([r["means"].get("seconds_per_step") for r in ordered])
        block["max_memory_gb"] = spread([r["max_memory_gb"] for r in ordered])
        block["panel_change"] = {name: spread([r["panel_change"].get(name) for r in ordered])
                                 for name in sorted({n for r in ordered for n in r["panel_change"]})}
    # The one comparison this part is FOR, drawn seed for seed rather than mean against mean.
    paired = []
    if "full" in arms and "lora" in arms:
        for seed in sorted(set(arms["full"]["seeds"]) & set(arms["lora"]["seeds"])):
            full, lora = arms["full"]["seeds"][seed], arms["lora"]["seeds"][seed]
            entry = {"seed": seed}
            for key in ("final", "change"):
                entry["%s_difference" % key] = (_round(lora[key] - full[key])
                                                if _finite(lora[key]) and _finite(full[key]) else None)
            entry["panel_difference"] = {
                name: lora["panel_change"][name] - full["panel_change"][name]
                for name in sorted(set(lora["panel_change"]) & set(full["panel_change"]))}
            paired.append(entry)
    return {"base_panels": base_panels, "arms": {arm: arms[arm] for arm in sorted(arms)},
            "lora_minus_full": paired,
            "lora_minus_full_final": spread([p["final_difference"] for p in paired]),
            "comparator": comparator, "flags": flags}


def build_part_b(runs: dict, panels: dict, scores: dict, sql_alone: dict | None) -> dict:
    base = {"panels": panel_scores(panels.get("base17b")),
            **{job: number(scores.get("base17b-%s" % job)) for job in sorted(B_JOBS)}}
    base["n"] = {job: number(scores.get("base17b-%s" % job), "n") for job in sorted(B_JOBS)}
    jobs: dict = {}
    for point, run in sorted(runs.items()):
        match = B_POINT.match(point)
        if not match:
            continue
        job, seed = match["job"], int(match["seed"])
        bed = B_JOBS[job]
        score = scores.get("%s-%s" % (point, bed))
        after = panel_scores(panels.get("%s-forget" % point))
        row = {
            "run": run["name"], "seed": seed, "steps_completed": run["steps_completed"],
            "job_score": number(score), "job_n": number(score, "n"),
            "job_accuracy": (score or {}).get("accuracy"),
            "incorrect_format": number(score, "incorrect_format"),
            "base_score": base.get(job),
            "means": run["means"], "max_memory_gb": run["max_memory_gb"],
            "merged_checkpoint": run["merged_checkpoint"],
            "panels": after, "flags": list(run["flags"]),
        }
        row["learned"] = (row["job_score"] - row["base_score"]
                          if _finite(row["job_score"]) and _finite(row["base_score"]) else None)
        row["panel_change"] = {name: after[name] - base["panels"][name]
                               for name in sorted(set(after) & set(base["panels"]))}
        jobs.setdefault(job, {"bed": bed, "seeds": {}})["seeds"][seed] = row
    for job, block in jobs.items():
        ordered = [block["seeds"][s] for s in sorted(block["seeds"])]
        block["job_score"] = spread([r["job_score"] for r in ordered])
        block["learned"] = spread([r["learned"] for r in ordered])
        block["seconds_per_step"] = spread([r["means"].get("seconds_per_step") for r in ordered])
        block["panel_change"] = {name: spread([r["panel_change"].get(name) for r in ordered])
                                 for name in sorted({n for r in ordered for n in r["panel_change"]})}
    return {"base": base, "jobs": {job: jobs[job] for job in sorted(jobs)}, "sql_alone": sql_alone,
            "sql_alone_note": "SQL learned alone at 1.7B is K3's stage A (rows a-seed0/1/2), not "
                              "re-measured here. Pass --k3 with your K3 report to include it.",
            "coding_note": "Coding is out of scope for K1c: the kit has no Modal-free code sandbox, so "
                           "there is no bed on which a coding task can be learned alone."}


def build_density(b_runs: dict, panels: dict, scores: dict) -> dict:
    """What a correct answer costs in Part B, per job and seed, against the untrained model.

    Plan section 4c and research row Q13: the job's own bed and the general panels, at the untrained
    Qwen3-1.7B and after the job was learned alone, with the ratio between them against the bar. A
    scoring that carries no token counts is unknown here and changes nothing else in this report.
    Part A has no bed-score.json at all, and says so rather than showing an empty row.
    """
    base_panels_found = panels.get("base17b")
    panel_names = sorted(((base_panels_found or (None, {}))[1].get("panels") or {}))
    untrained_panels = {name: cost(base_panels_found, panel=name) for name in panel_names}
    untrained = {job: cost(scores.get("base17b-%s" % job)) for job in sorted(B_JOBS)}
    measured = sum(1 for cell in list(untrained.values()) + list(untrained_panels.values())
                   if cell is not None)
    jobs: dict = {}
    for point in sorted(b_runs):
        match = B_POINT.match(point)
        if not match:
            continue
        job, seed = match["job"], int(match["seed"])
        bed = B_JOBS[job]
        trained = cost(scores.get("%s-%s" % (point, bed)))
        after = panels.get("%s-forget" % point)
        row = {"bed": compare_cost(trained, untrained.get(job)),
               "panels": {name: compare_cost(cost(after, panel=name), untrained_panels[name])
                          for name in panel_names}}
        measured += sum(1 for cell in [trained] + [cost(after, panel=name) for name in panel_names]
                        if cell is not None)
        jobs.setdefault(job, {"bed": bed, "seeds": {}})["seeds"][seed] = row
    for job, block in jobs.items():
        ordered = [block["seeds"][seed] for seed in sorted(block["seeds"])]
        block["ratio"] = {"bed": spread([row["bed"]["ratio"] for row in ordered]),
                          "panels": {name: spread([row["panels"][name]["ratio"] for row in ordered])
                                     for name in panel_names}}
        block["untrained"] = {"bed": (untrained.get(job) or {}).get("tokens_per_correct"),
                              "panels": {name: (untrained_panels[name] or {}).get("tokens_per_correct")
                                         for name in panel_names}}
    return {"bar": DENSITY_BAR, "definition": "output tokens over the whole held-out set, divided "
                                              "by the correct answers",
            "measured": measured, "available": int(measured > 0),
            "note": None if measured else NO_TOKENS, "part_a_note": PART_A_NOTE,
            "jobs": {job: jobs[job] for job in sorted(jobs)}}


def build(runs_dir: Path, forgetting: Path, evaluations: Path, k0: Path, k3: Path | None = None, *,
          allow_different_machines: bool = False) -> dict:
    if not runs_dir.is_dir():
        raise K1cReportError("not a directory: %s" % runs_dir)
    a_runs = {stem: read_run(path, summary_name="run-summary.json")
              for stem, path in latest_dirs(runs_dir).items() if A_POINT.match(stem)}
    b_runs = {stem: read_run(path, summary_name="train-summary.json")
              for stem, path in latest_dirs(runs_dir).items() if B_POINT.match(stem)}
    if not a_runs and not b_runs:
        raise K1cReportError(
            "no K1c runs under %s: expected <arm>-seed<S>-aN for %s (Part A) or <job>-seed<S>-aN for "
            "%s (Part B)" % (runs_dir, ", ".join(A_ARMS), ", ".join(sorted(B_JOBS))))
    panels_found = latest_files_found(forgetting, "forgetting.json")
    scores_found = latest_files_found(evaluations, "bed-score.json")
    panels = {stem: result for stem, (_path, result) in panels_found.items()}
    scores = {stem: result for stem, (_path, result) in scores_found.items()}
    machines = fingerprints(panels, scores)
    comparable = len(machines) == 1 and machines[0] is not None
    if (panels or scores) and not comparable and not allow_different_machines:
        raise K1cReportError(
            "the scorings under %s and %s carry %d different machine-and-mode fingerprints (%s). A "
            "count from one machine cannot be subtracted from a count on another: a panel moves by up "
            "to 3 points across machines before any training. Re-score on one machine, or pass "
            "--allow-different-machines and say so in the readout."
            % (forgetting, evaluations, len(machines), ", ".join(str(m) for m in machines)))

    comparator = read_comparator(k0)
    part_a = build_part_a(a_runs, panels, comparator)
    part_b = build_part_b(b_runs, panels, scores, read_sql_alone(k3))
    pilots = {name: read_run(path, summary_name=("run-summary.json" if name.startswith("pilot-") and
                                                 not name.startswith("pilot-b-") else "train-summary.json"))
              for name, path in sorted(latest_dirs(runs_dir).items()) if name.startswith("pilot-")}
    return {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_dir": str(runs_dir.resolve()), "forgetting_dir": str(forgetting.resolve()),
        "eval_dir": str(evaluations.resolve()), "comparator_report": str(k0.resolve()),
        "k3_report": str(Path(k3).resolve()) if k3 else None,
        "part_a": part_a, "part_b": part_b,
        "density": build_density(b_runs, panels_found, scores_found),
        "pilots": {name: {"means": run["means"], "validations": run["validations"],
                          "summary": run["summary"], "flags": run["flags"]}
                   for name, run in pilots.items()},
        "parts_reported": int(bool(part_a["arms"])) + int(bool(part_b["jobs"])),
        "arms_reported": len(part_a["arms"]) + len(part_b["jobs"]),
        "comparator_runs": len(comparator),
        "machine_ids": machines, "comparable": int(comparable),
        "different_machines_allowed": int(bool(allow_different_machines)),
        "flags": part_a["flags"],
    }


# ------------------------------------------------------------------------------------ rendering
def _fmt(value, pattern="%.4f"):
    return pattern % value if _finite(value) else "-"


def _pct(value, digits):
    return ("%%.%df%%%%" % digits) % (100 * value) if _finite(value) else "-"


def render_density(report: dict) -> list:
    """One table: what a correct answer costs in Part B, per job and seed, against the untrained model."""
    block = report.get("density") or {}
    lines = ["", "## What a correct answer costs (tokens per correct answer)", "",
             "Plan section 4c, research row Q13. Tokens per correct answer is the output tokens over "
             "the whole held-out set divided by the correct answers, so it is a cost per task and not "
             "a length. A trained model may spend at most %.1f times the untrained model's on the bed "
             "it learned and on the general panel; over the bar the gain is reported at cost."
             % block.get("bar", DENSITY_BAR), "",
             block.get("part_a_note") or PART_A_NOTE, ""]
    if not block.get("measured"):
        lines += [block.get("note") or NO_TOKENS, ""]
    lines += ["| job | seed | bed or panel | untrained | after learning it alone | ratio | verdict |",
              "|---|---|---|---|---|---|---|"]
    for job in sorted(block.get("jobs") or {}):
        entry = block["jobs"][job]
        for seed in sorted(entry["seeds"]):
            row = entry["seeds"][seed]
            for name, cell in [(entry["bed"], row["bed"])] + sorted(row["panels"].items()):
                lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                    job, seed, name, show_cost(cell["untrained"]), show_cost(cell["trained"]),
                    show_cost(cell["ratio"], 2), cell["verdict"]))
        ratios = entry.get("ratio") or {"bed": spread([]), "panels": {}}
        for name, summary in [(entry["bed"], ratios["bed"])] + sorted(ratios["panels"].items()):
            mean = summary["mean"]
            lines.append("| **%s** | mean of %d | %s | | | **%s** (sd %s) | %s |" % (
                job, summary["n"], name, show_cost(mean, 2),
                "-" if summary["sd"] is None else show_cost(summary["sd"], 2),
                density.verdict(mean, bar=DENSITY_BAR) if density else "unknown"))
    return lines


def render(report: dict) -> str:
    a, b = report["part_a"], report["part_b"]
    lines = ["# K1c: the learned-alone baselines, with reward-only GRPO", "",
             "Generated %s by kit/k1c_report.py (%s). Every number is recomputed from the raw files."
             % (report["generated_at"], SCHEMA), "",
             "Every job here was learned ALONE, from the untrained model, with the authors' own "
             "`--config-name baseline_grpo` -- reward only, no teacher. These are the rows every later "
             "comparison in this programme is read against; nothing here is a conclusion about a "
             "method.", ""]
    if not report["comparable"]:
        lines += ["**The scorings do not share one machine-and-mode fingerprint (%s), so the "
                  "differences below are not comparable.**"
                  % ", ".join(str(m) for m in report["machine_ids"]), ""]
    if report["flags"]:
        lines += ["## Read this first", ""] + ["- **%s**" % flag for flag in report["flags"]] + [""]

    # ---------------------------------------------------------------------------------- Part A
    if a["arms"]:
        lines += ["## Part A: Qwen3-8B on ToolAlpaca, full training against LoRA", "",
                  "K0's task, data and geometry: 40 steps, batch 32, 8 attempts, validation every 5, "
                  "warm-up 10. Accuracies are the authors' STRICT all-or-nothing tool-use score, "
                  "`%s`, recomputed per question from `acc`. The SDPO column is the K0 run at the same "
                  "seed (%s) -- a comparator, not a control." % (VAL_KEY, report["comparator_report"]), "",
                  "| arm | seed | lr | steps | untrained | final | change | SDPO final | GRPO - SDPO | s/step | max GB |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in sorted(a["arms"]):
            block = a["arms"][arm]
            for seed in sorted(block["seeds"]):
                r = block["seeds"][seed]
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    arm, seed, r.get("learning_rate") or "-", r["steps_completed"],
                    _fmt(r["base"]), _fmt(r["final"]), _fmt(r["change"], "%+.4f"),
                    _fmt((r["comparator"] or {}).get("final")), _fmt(r.get("vs_sdpo"), "%+.4f"),
                    _fmt(r["means"].get("seconds_per_step"), "%.1f"),
                    _fmt(r["max_memory_gb"], "%.1f")))
            lines.append("| **%s** | mean of %d | | | | **%s** (sd %s) | **%s** (sd %s) | | **%s** (sd %s) | %s | %s |" % (
                arm, block["final"]["n"], _fmt(block["final"]["mean"]), _fmt(block["final"]["sd"]),
                _fmt(block["change"]["mean"], "%+.4f"), _fmt(block["change"]["sd"]),
                _fmt(block["vs_sdpo"]["mean"], "%+.4f"), _fmt(block["vs_sdpo"]["sd"]),
                _fmt(block["seconds_per_step"]["mean"], "%.1f"),
                _fmt(block["max_memory_gb"]["mean"], "%.1f")))

        if a["lora_minus_full"]:
            lines += ["", "### LoRA minus full, at the same seed", "",
                      "Paired seed for seed, which is the only way these two are comparable: both arms "
                      "read the same questions in the same order, and differ only in their "
                      "parameterisation and learning rate.", "",
                      "| seed | final difference | change difference | panel differences |",
                      "|---|---|---|---|"]
            for entry in a["lora_minus_full"]:
                panels = ", ".join("%s %+d" % (name, value)
                                   for name, value in sorted(entry["panel_difference"].items())) or "-"
                lines.append("| %s | %s | %s | %s |" % (
                    entry["seed"], _fmt(entry["final_difference"], "%+.4f"),
                    _fmt(entry["change_difference"], "%+.4f"), panels))
            lines.append("| **mean of %d** | **%s** (sd %s) | | |" % (
                a["lora_minus_full_final"]["n"], _fmt(a["lora_minus_full_final"]["mean"], "%+.4f"),
                _fmt(a["lora_minus_full_final"]["sd"])))

        folds = [(arm, seed, a["arms"][arm]["seeds"][seed]["fold"])
                 for arm in sorted(a["arms"]) for seed in sorted(a["arms"][arm]["seeds"])
                 if a["arms"][arm]["seeds"][seed]["fold"]]
        if folds:
            lines += ["", "### Did the adapter actually reach the weights?", "",
                      "`verl.model_merger merge` saves the BASE model and writes the adapter beside it "
                      "with `lora_alpha` hard-coded to 0. kit/fold_lora.py folds it and refuses a "
                      "no-op; this table is the receipt.", "",
                      "| arm | seed | r | alpha | modules changed | of | largest weight change | "
                      "adapted weights that moved | change lost to the 16-bit save (median) |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for arm, seed, fold in folds:
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    arm, seed, fold.get("r"), fold.get("alpha_declared"),
                    fold.get("changed_tensors"), fold.get("expected_changed"),
                    _fmt(fold.get("max_abs_delta"), "%.3g"),
                    _pct(fold.get("share_of_adapted_elements_changed"), 1),
                    _pct(fold.get("rounding_error_median"), 0)))
            lines += ["", "If most of the change is lost to the 16-bit save, the LoRA model that was scored is "
                      "closer to the untrained model than the one that trained, and a smaller forgetting number "
                      "for LoRA may be that rounding, not the method."]

        if a["base_panels"]:
            names = sorted(a["base_panels"])
            lines += ["", "### Forgetting, Part A (a trained model may lose at most 3 per 100 on each panel)", "",
                      "Untrained Qwen3-8B: " + ", ".join("%s %d" % (n, a["base_panels"][n]) for n in names), "",
                      "| arm | seed | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
            for arm in sorted(a["arms"]):
                for seed in sorted(a["arms"][arm]["seeds"]):
                    r = a["arms"][arm]["seeds"][seed]
                    lines.append("| %s | %s | %s |" % (arm, seed, " | ".join(
                        "%s (%+d)" % (r["panels"][n], r["panel_change"][n])
                        if n in r["panels"] and n in r["panel_change"] else "-" for n in names)))

    # ---------------------------------------------------------------------------------- Part B
    if b["jobs"]:
        lines += ["", "## Part B: Qwen3-1.7B, one job learned alone", "",
                  "40 steps x 32 questions, which is K3's stage-B dose, so `gsm8k` here is directly "
                  "readable against K3's `b-none` arm (maths learned AFTER SQL) as well as being the "
                  "row rehearsal has to beat. Counts are correct answers on the bed's largest held-out "
                  "set, scored deterministically on one GPU of one machine.", "",
                  "| job | bed | seed | held out | untrained | after | learned | wrong format | s/step |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for job in sorted(b["jobs"]):
            block = b["jobs"][job]
            for seed in sorted(block["seeds"]):
                r = block["seeds"][seed]
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    job, block["bed"], seed, r["job_n"], r["base_score"], r["job_score"],
                    "%+d" % r["learned"] if _finite(r["learned"]) else "-",
                    r["incorrect_format"], _fmt(r["means"].get("seconds_per_step"), "%.1f")))
            lines.append("| **%s** | | mean of %d | | | **%s** (sd %s) | **%s** (sd %s) | | %s |" % (
                job, block["learned"]["n"], _fmt(block["job_score"]["mean"], "%.1f"),
                _fmt(block["job_score"]["sd"], "%.1f"), _fmt(block["learned"]["mean"], "%+.1f"),
                _fmt(block["learned"]["sd"], "%.1f"),
                _fmt(block["seconds_per_step"]["mean"], "%.1f")))

        if b["base"]["panels"]:
            names = sorted(b["base"]["panels"])
            lines += ["", "### Forgetting, Part B", "",
                      "Untrained Qwen3-1.7B: " + ", ".join("%s %d" % (n, b["base"]["panels"][n])
                                                           for n in names), "",
                      "| job | seed | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
            for job in sorted(b["jobs"]):
                for seed in sorted(b["jobs"][job]["seeds"]):
                    r = b["jobs"][job]["seeds"][seed]
                    lines.append("| %s | %s | %s |" % (job, seed, " | ".join(
                        "%s (%+d)" % (r["panels"][n], r["panel_change"][n])
                        if n in r["panels"] and n in r["panel_change"] else "-" for n in names)))

        lines += ["", "### SQL learned alone", ""]
        if b["sql_alone"]:
            sql = b["sql_alone"]
            lines += ["From %s. %s" % (sql["report"], sql["note"]), "",
                      "| seed | K3 row | Spider held-out correct | untrained | learned |",
                      "|---|---|---|---|---|"]
            for seed in sorted(sql["seeds"]):
                entry = sql["seeds"][seed]
                learned = (entry["job_a"] - sql["base_job_a"]
                           if _finite(entry["job_a"]) and _finite(sql["base_job_a"]) else None)
                lines.append("| %s | %s | %s | %s | %s |" % (
                    seed, entry["point"], entry["job_a"], sql["base_job_a"],
                    "%+d" % learned if _finite(learned) else "-"))
            lines.append("| **mean of %d** | | | | **%s** (sd %s) |" % (
                sql["learned"]["n"], _fmt(sql["learned"]["mean"], "%+.1f"), _fmt(sql["learned"]["sd"])))
        else:
            lines += [b["sql_alone_note"]]
        lines += ["", b["coding_note"]]

    pilots = report.get("pilots") or {}
    if pilots:
        lines += ["", "## The pilots (printed, not judged)", "",
                  "Two steps each. Every number here swings widely over two steps, which is why the "
                  "campaign gates these rows on exit codes, a merged model, a length floor and the "
                  "untrained validation band instead.", "",
                  "| pilot | arm | steps | response tokens | entropy | score | untrained validation |",
                  "|---|---|---|---|---|---|---|"]
        for name, run in sorted(pilots.items()):
            first = next((v for v in run["validations"] if v.get("step") == 0), {})
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                name, (run["summary"] or {}).get("arm", "-"), (run["summary"] or {}).get("steps", "-"),
                _fmt(run["means"].get("response_tokens"), "%.1f"),
                _fmt(run["means"].get("entropy"), "%.3f"), _fmt(run["means"].get("score"), "%.3f"),
                _fmt(first.get("accuracy_logged"))))

    lines += render_density(report)

    everything = []
    for part in (a["arms"], b["jobs"]):
        for name in sorted(part):
            for seed in sorted(part[name]["seeds"]):
                for flag in part[name]["seeds"][seed]["flags"]:
                    everything.append(("%s seed %s" % (name, seed), flag))
    for name in sorted(pilots):
        for flag in pilots[name]["flags"]:
            everything.append((name, flag))
    if everything:
        lines += ["", "## Flags", ""] + ["- **%s** %s" % item for item in everything]
    lines += ["", "%d parts, %d arms, against %d SDPO comparator runs."
              % (report["parts_reported"], report["arms_reported"], report["comparator_runs"]), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, required=True, help="the K1c run directories")
    parser.add_argument("--forgetting", type=Path, required=True, help="the panel scorings")
    parser.add_argument("--eval", dest="evaluations", type=Path, required=True,
                        help="the bed scorings (Part B)")
    parser.add_argument("--k0", type=Path, required=True,
                        help="the SDPO comparator: K0's report.json")
    parser.add_argument("--k3", type=Path, default=None,
                        help="optional: K3's report.json, which holds SQL learned alone at 1.7B")
    parser.add_argument("--out", type=Path, required=True, help="a new directory; never overwritten")
    parser.add_argument("--allow-different-machines", action="store_true",
                        help="report anyway; recorded in the report as a departure")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % args.out)
    try:
        report = build(args.runs, args.forgetting, args.evaluations, args.k0, args.k3,
                       allow_different_machines=args.allow_different_machines)
    except K1cReportError as exc:
        raise SystemExit(str(exc))
    args.out.mkdir(parents=True)
    (args.out / "k1c-report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    text = render(report)
    (args.out / "k1c-report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
