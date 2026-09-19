#!/usr/bin/env python3
"""Turn the run directories written by run_sdpo_toolalpaca.sh into one report to send back.

    python make_report.py /work/sdpo-work/runs --out /work/sdpo-work/report

Writes report.md (for a person) and report.json (for our receipts). Standard library only, no GPU,
no network; it reads each run's metrics.jsonl, validation/*.jsonl, env/ and console.log, and never
writes inside a run directory. A run that crashed is reported as such, with the end of its log.

Every number is recomputed from the raw files, so the report can be regenerated at any time and a
second copy made on our side from the same files would be identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-sdpo-report.v1"
VAL_KEY = "val-core/tooluse/acc/mean@16"
CALIBRATION_BAND = (0.555, 0.600)      # our two base readings were 0.5744 and 0.5790
LENGTH_FLOOR_TOKENS = 16
TRAIN_KEYS = {
    "score": "critic/score/mean",
    "response_tokens": "response_length/mean",
    "success_group_fraction": "self_distillation/success_group_fraction",
    "empty_target_batch": "self_distillation/empty_target_batch",
    "grad_norm": "actor/grad_norm",
    "entropy": "actor/entropy",
    "seconds_per_step": "perf/time_per_step",
    "max_memory_gb": "perf/max_memory_allocated_gb",
}


def _read_text(path: Path, limit: int | None = None) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    return text if limit is None else text[-limit:]


def _jsonl(path: Path) -> list:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _round(value, digits=4):
    return round(value, digits) if _finite(value) else None


def validation_by_question(path: Path) -> dict:
    """One validation dump -> {question id: mean accuracy over its samples}, plus format and length."""
    groups: dict = defaultdict(list)
    for row in _jsonl(path):
        qid = hashlib.sha1(row["input"].encode()).hexdigest()[:12]
        groups[qid].append(row)
    per_question = {qid: statistics.fmean(r["acc"] for r in rows) for qid, rows in groups.items()}
    every = [r for rows in groups.values() for r in rows]
    return {
        "questions": len(groups),
        "samples_per_question": sorted({len(rows) for rows in groups.values()}),
        "accuracy": statistics.fmean(per_question.values()) if per_question else None,
        "incorrect_format_rate": statistics.fmean(r.get("incorrect_format", 0) for r in every) if every else None,
        "median_output_chars": statistics.median(len(r["output"]) for r in every) if every else None,
        "per_question": per_question,
    }


def paired_change(first: dict, last: dict) -> dict | None:
    shared = sorted(set(first) & set(last))
    if len(shared) < 2:
        return None
    deltas = [last[q] - first[q] for q in shared]
    mean = statistics.fmean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(len(deltas))
    return {
        "questions": len(shared), "mean_change": mean, "standard_error": se,
        "z": mean / se if se > 0 else None,
        "interval_95": [mean - 1.96 * se, mean + 1.96 * se],
        "better": sum(d > 0 for d in deltas), "worse": sum(d < 0 for d in deltas),
        "unchanged": sum(d == 0 for d in deltas),
    }


def _identity(run: Path) -> dict:
    env = run / "env"
    argv = _read_text(env / "argv.txt")
    overrides = dict(t.split("=", 1) for t in (argv or "").splitlines() if "=" in t)
    freeze = _read_text(env / "pip-freeze.txt") or ""
    versions = {}
    for line in freeze.splitlines():
        name = line.split("==")[0].split(" @")[0].strip().lower()
        if name in ("torch", "vllm", "transformers", "ray", "math-verify", "flash-attn", "flash_attn"):
            versions[name] = line.strip()
    smi = _read_text(env / "nvidia-smi.txt") or ""
    gpus = sorted({part.strip() for line in smi.splitlines() if "NVIDIA" in line and "|" in line
                   for part in line.split("|")[1:2]})
    return {
        "argv_sha256": hashlib.sha256(argv.encode()).hexdigest() if argv else None,
        "seed": overrides.get("data.seed"),
        "total_training_steps": overrides.get("trainer.total_training_steps"),
        "test_freq": overrides.get("trainer.test_freq"),
        "train_batch_size": overrides.get("data.train_batch_size"),
        "rollouts": overrides.get("actor_rollout_ref.rollout.n"),
        "param_offload": overrides.get("actor_rollout_ref.actor.fsdp_config.param_offload"),
        "n_gpus": overrides.get("trainer.n_gpus_per_node"),
        "sdpo_commit": (_read_text(env / "sdpo-commit.txt") or "").strip() or None,
        "sdpo_checkout_dirty": bool((_read_text(env / "sdpo-dirty.txt") or "").strip()),
        "versions": versions, "gpus": gpus,
        "started_at": (_read_text(env / "started-at.txt") or "").strip() or None,
        "finished_at": (_read_text(env / "finished-at.txt") or "").strip() or None,
    }


def summarize_run(run: Path) -> dict:
    out: dict = {"name": run.name, "identity": _identity(run), "flags": []}
    metrics_path = run / "metrics.jsonl"
    rows = _jsonl(metrics_path) if metrics_path.is_file() else []
    training, validations = [], []
    for row in rows:
        data = row.get("data", {})
        if VAL_KEY in data:
            validations.append({"step": row["step"], "accuracy_logged": data[VAL_KEY]})
        if TRAIN_KEYS["score"] in data:
            training.append({"step": row["step"], **{k: _round(data.get(v)) for k, v in TRAIN_KEYS.items()}})
    out["training"] = training
    out["steps_completed"] = max((t["step"] for t in training), default=0)

    dumps = {}
    vdir = run / "validation"
    if vdir.is_dir():
        for path in sorted(vdir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
            if path.stem.isdigit():
                dumps[int(path.stem)] = validation_by_question(path)
    for v in validations:
        d = dumps.get(v["step"])
        if d:
            v.update({"accuracy_from_dump": _round(d["accuracy"]), "questions": d["questions"],
                      "samples_per_question": d["samples_per_question"],
                      "incorrect_format_rate": _round(d["incorrect_format_rate"]),
                      "median_output_chars": d["median_output_chars"]})
    out["validations"] = validations
    out["per_question"] = {str(step): {q: _round(a) for q, a in d["per_question"].items()} for step, d in dumps.items()}
    if len(dumps) >= 2:
        first, last = min(dumps), max(dumps)
        change = paired_change(dumps[first]["per_question"], dumps[last]["per_question"])
        if change:
            out["paired_change"] = {"from_step": first, "to_step": last,
                                    **{k: (_round(v) if not isinstance(v, list) else [_round(x) for x in v])
                                       for k, v in change.items()}}

    want = out["identity"]["total_training_steps"]
    out["complete"] = bool(want) and out["steps_completed"] >= int(want)
    merged = sorted(run.glob("hf-step*/config.json"))
    out["merged_checkpoint"] = str(merged[-1].parent.name) if merged else None

    if validations:
        base = validations[0]["accuracy_logged"]
        if validations[0]["step"] == 0 and not (CALIBRATION_BAND[0] <= base <= CALIBRATION_BAND[1]):
            out["flags"].append("CALIBRATION: base accuracy %.4f is outside %.3f-%.3f" % (base, *CALIBRATION_BAND))
        for v in validations:
            a, b = v["accuracy_logged"], v.get("accuracy_from_dump")
            if b is not None and abs(a - b) > 1e-3:
                out["flags"].append("MISMATCH at step %s: logged %.4f, recomputed %.4f" % (v["step"], a, b))
    short = [t["step"] for t in training if _finite(t["response_tokens"]) and t["response_tokens"] < LENGTH_FLOOR_TOKENS]
    if short:
        out["flags"].append("LENGTH: mean response under %d tokens at steps %s" % (LENGTH_FLOOR_TOKENS, short))
    starved = [t["step"] for t in training if t["success_group_fraction"] == 0]
    if starved:
        out["flags"].append("NO TEACHER SIGNAL (success_group_fraction = 0) at steps %s" % starved)
    bad = [t["step"] for t in training if t["grad_norm"] is None]
    if bad:
        out["flags"].append("NON-FINITE grad norm at steps %s" % bad)
    if not out["complete"]:
        out["flags"].append("INCOMPLETE: %s of %s steps" % (out["steps_completed"], want))
        out["console_tail"] = _read_text(run / "console.log", limit=4000)
    return out


def _fmt(value, pattern="%.4f"):
    return pattern % value if _finite(value) else "-"


def render(report: dict) -> str:
    lines = ["# SDPO on ToolAlpaca: partner run report", "",
             "Generated %s by kit/make_report.py (%s). Every number is recomputed from the raw run files."
             % (report["generated_at"], SCHEMA), "", "## Summary", "",
             "| run | seed | steps done | base | final | change | 95% interval | better / worse / same | flags |",
             "|---|---|---|---|---|---|---|---|---|"]
    for run in report["runs"]:
        vals, change = run["validations"], run.get("paired_change") or {}
        interval = change.get("interval_95") or [None, None]
        lines.append("| %s | %s | %s of %s | %s | %s | %s | %s to %s | %s / %s / %s | %s |" % (
            run["name"], run["identity"]["seed"] or "default", run["steps_completed"],
            run["identity"]["total_training_steps"] or "?",
            _fmt(vals[0]["accuracy_logged"]) if vals else "-", _fmt(vals[-1]["accuracy_logged"]) if len(vals) > 1 else "-",
            _fmt(change.get("mean_change"), "%+.4f"), _fmt(interval[0], "%+.4f"), _fmt(interval[1], "%+.4f"),
            change.get("better", "-"), change.get("worse", "-"), change.get("unchanged", "-"), len(run["flags"])))
    for dose, group in sorted(report["by_dose"].items()):
        lines += ["", "Dose %s steps, %d complete run(s): final accuracy mean %s (sd %s); change mean %s (sd %s)." % (
            dose, group["runs"], _fmt(group["final_mean"]), _fmt(group["final_sd"]),
            _fmt(group["change_mean"], "%+.4f"), _fmt(group["change_sd"]))]
    for run in report["runs"]:
        ident = run["identity"]
        lines += ["", "## %s" % run["name"], "",
                  "- reference commit `%s`%s; %s GPUs (%s); offload %s; versions: %s" % (
                      (ident["sdpo_commit"] or "?")[:12], " **with local modifications**" if ident["sdpo_checkout_dirty"] else "",
                      ident["n_gpus"] or "?", ", ".join(ident["gpus"]) or "not recorded", ident["param_offload"],
                      ", ".join(sorted(ident["versions"].values())) or "not recorded"),
                  "- started %s, finished %s; merged checkpoint: %s; argv sha256 `%s`" % (
                      ident["started_at"], ident["finished_at"], run["merged_checkpoint"] or "none",
                      (ident["argv_sha256"] or "?")[:16])]
        lines += ["- **FLAG** " + flag for flag in run["flags"]]
        if run["validations"]:
            lines += ["", "| validation step | avg@16 logged | recomputed | wrong-format rate | median output chars |", "|---|---|---|---|---|"]
            lines += ["| %s | %s | %s | %s | %s |" % (v["step"], _fmt(v["accuracy_logged"]), _fmt(v.get("accuracy_from_dump")),
                                                     _fmt(v.get("incorrect_format_rate")), v.get("median_output_chars", "-"))
                      for v in run["validations"]]
        if run["training"]:
            lines += ["", "| step | train score | response tokens | success-group fraction | empty target | grad norm | entropy | s/step | max GB |",
                      "|---|---|---|---|---|---|---|---|---|"]
            lines += ["| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                t["step"], _fmt(t["score"]), _fmt(t["response_tokens"], "%.1f"), _fmt(t["success_group_fraction"], "%.3f"),
                _fmt(t["empty_target_batch"], "%.2f"), _fmt(t["grad_norm"], "%.3f"), _fmt(t["entropy"], "%.3f"),
                _fmt(t["seconds_per_step"], "%.1f"), _fmt(t["max_memory_gb"], "%.1f")) for t in run["training"]]
        if run.get("console_tail"):
            lines += ["", "End of console.log:", "", "```", run["console_tail"].strip(), "```"]
    return "\n".join(lines) + "\n"


def build(runs_dir: Path) -> dict:
    runs = [summarize_run(p) for p in sorted(runs_dir.iterdir()) if p.is_dir()]
    by_dose: dict = defaultdict(list)
    for run in runs:
        if run["complete"] and len(run["validations"]) > 1 and run.get("paired_change"):
            by_dose[str(run["identity"]["total_training_steps"])].append(run)
    summary = {}
    for dose, group in by_dose.items():
        finals = [r["validations"][-1]["accuracy_logged"] for r in group]
        changes = [r["paired_change"]["mean_change"] for r in group]
        summary[dose] = {"runs": len(group), "final_mean": _round(statistics.fmean(finals)),
                         "final_sd": _round(statistics.stdev(finals)) if len(finals) > 1 else None,
                         "change_mean": _round(statistics.fmean(changes)),
                         "change_sd": _round(statistics.stdev(changes)) if len(changes) > 1 else None}
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runs_dir": str(runs_dir), "runs": runs, "by_dose": summary}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.runs_dir.is_dir():
        print("not a directory: %s" % args.runs_dir, file=sys.stderr)
        return 2
    report = build(args.runs_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    (args.out / "report.md").write_text(render(report))
    print("wrote %s and %s (%d runs)" % (args.out / "report.md", args.out / "report.json", len(report["runs"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
