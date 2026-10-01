#!/usr/bin/env python3
"""The pre-registered decisions of the small-model control (plan v3, package 2), applied mechanically.

    python control_report.py --work $WORK --serving 2048 --diagnostic 8192 --out $WORK/k1c/control/control-report-a1

THE CONTROL. Qwen3-1.7B learns GSM8K alone for 40 steps in three cells of three runs (seeds 0, 1, 2):
    lr6        learning rate 1e-6, training limit 8,192, no gate
    gate       learning rate 1e-5, training limit 2,048, reward only for an answer that finishes
    lr6-gate   both
kit/campaigns/k1c-control.yaml writes the tree; this file reads exactly this and nothing else:
    <work>/runs/<cell>-seed<S>-a<N>/train-summary.json                    returncode, merged, steps, lr, ...
    <work>/runs/<cell>-seed<S>-a<N>/metrics.jsonl                         {"step", "data": {"response_length/mean", ...}}
    <work>/k1c/control/report-sweep/base17b-gsm8k-a<N>/sweep.json         the untrained model: the reference r
    <work>/k1c/control/report-sweep/<cell>-seed<S>-gsm8k-a<N>/sweep.json  one per run (kit/cap_sweep.py bed)
    <work>/k1c/control/report-prefix/prefix-base17b-gsm8k-a<N>.json       a prefix check of r (kit/cap_sweep.py check)
The highest attempt of a sweep (or check) that holds the file wins. A run's attempt of record is the highest attempt
whose train-summary.json has merged == 1.

LINEAGE. A run's sweep is used only if the end of its `model` path is `runs/<key>-a<N>/hf-step<digits>` with the
run's own key and its attempt of record N; otherwise the run is not reported, the reason starts "lineage:", and it
is listed under `lineage_problems`. A sweep of attempt 1 after attempt 2 merged scored a model that is not the run of
record.

ONE MACHINE. A run whose sweep's machine id differs from the reference's, or is missing, is not reported, unless
--allow-different-machines, which is recorded.

THE RULES (pre-registration section 3), with n the reference's question count and t = ceil(5n/100) (15 of 300):
  per reported run, from kit/budget_report.py's `account` (the identity F = residual + budget + extraction is that
  file's): gain = S_run(B) - S_ref(B); learns = gain >= t; preserves = S_run(B) >= S_ref(B) - t;
  drift_by_length = the mean answer length at the last training step that has it > 2 x the first (None if either is
  missing); drift_by_cost = tokens per strict-correct answer at B > 1.5 x the reference's (None if not defined);
  drifts = True if either is True, False if both are False, None otherwise.
  per cell of three runs (an unreported run counts as unknown):
    free_of_drift  True if at least two runs have drifts False; False if at least two have drifts True or two can no
                   longer be reached; None otherwise
    learns         None unless all three runs are reported; then True iff the mean gain >= t and at least two gains > 0
    preserves      True if at least two runs preserve; False if at least two do not or two can no longer be reached;
                   None otherwise
    reading        the first that applies of READINGS below
  no_cell_learns = True iff every cell's learns is False; False iff some cell's is True; None otherwise.
Thresholds in questions are compared in integers. `control-report.json` carries the integer bar keys `runs_expected`,
`runs_reported`, `cells_decided`, `accounting_exact` and `lineage_ok`.

REFUSED: an existing --out, a --work that is not a folder, B not below H, a missing or unreadable reference sweep, one
that did not read B or H, and a reference prefix check that exists and does not pass (budget_report.read_checks: some
answers compared, text agreement at least QUALIFY_AGREEMENT, one machine, one context length). The check is REQUIRED and must
be of the reference scoring that was swept (its folder name and the two caps): absent, failed or of another scoring,
the report refuses. Standard library only; kit/budget_report.py is loaded by file path. Nothing is
overwritten.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "kit-control-report.v1"
CELLS = ("lr6", "gate", "lr6-gate")
SEEDS = (0, 1, 2)
BASE = "base17b"
BED = "gsm8k"
TOLERANCE_PER_100 = 5              # learns / preserves at ceil(5n/100) questions
LENGTH_FACTOR = 2                  # drift: last mean length > 2 x first
COST_FACTOR = (3, 2)               # drift: tokens per correct > 3/2 x the reference's
LENGTH_KEY = "response_length/mean"
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
PREFIX_FILE = re.compile(r"^prefix-%s-%s-a(?P<attempt>\d+)\.json$" % (BASE, BED))
SUMMARY_FIELDS = ("returncode", "merged", "steps", "lr", "max_response_length", "finish_gate", "n_gpus", "seconds", "model_dir")
DECISIONS = ("free_of_drift", "learns", "preserves")

READ_LR6 = "The drift is recipe-specific to the higher learning rate, subject to the K1c runs being a fair comparison."
READ_GATE = ("The combined training-limit-and-gate change stabilises a single task at the old rate; it cannot say which "
             "of the two parts did it.")
READ_NOT_USEFUL = "Its usefulness is not established; that alone does not show that learning was suppressed."
READ_NOT_DECIDED = "Not decided: "
READ_NONE = "No pre-registered reading applies; the numbers stand as reported."
NO_CELL_LEARNS = ("No tested setting demonstrated learning on GSM8K at this dose; the reason is not established and it "
                  "is not evidence about model size.")
MECHANICAL = "These are the pre-registered rules applied mechanically; they decide nothing by themselves."
ONE_TASK = "This is one task: it says nothing about keeping a task while learning another."


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_control_report_%s" % name, HERE / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


budget_report = _load("budget_report")


class ControlReportError(ValueError):
    """The work tree cannot support a report; nothing is written."""


def run_keys() -> list:
    return ["%s-seed%d" % (cell, seed) for cell in CELLS for seed in SEEDS]


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# ---------------------------------------------------------------------------------------- the tree
def attempts(directory: Path) -> dict:
    """{stem: [(attempt, folder)]} for every `<stem>-a<N>` folder directly under `directory`."""
    found: dict = defaultdict(list)
    if directory.is_dir():
        for child in directory.iterdir():
            match = ATTEMPT.match(child.name)
            if match and child.is_dir():
                found[match["stem"]].append((int(match["attempt"]), child))
    return found


def latest(index: dict, stem: str, filename=None) -> tuple:
    """(folder, attempt) of the highest attempt of `stem`, holding `filename` when one is named; else (None, None)."""
    best = (None, None)
    for attempt, folder in index.get(stem, []):
        if filename and not (folder / filename).is_file():
            continue
        if best[1] is None or attempt > best[1]:
            best = (folder, attempt)
    return best


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_metrics(path: Path) -> tuple:
    """([{"step", "data"}] in step order, unreadable lines). A half-written line is counted, never fatal."""
    rows, bad = [], 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return [], 0
    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if not isinstance(row, dict) or not isinstance(row.get("data"), dict) or not _finite(row.get("step")):
            bad += 1
            continue
        rows.append(row)
    return sorted(rows, key=lambda r: r["step"]), bad


def run_attempt(runs: dict, key: str) -> tuple:
    """(folder, attempt, summary) of the highest attempt of `key` whose train-summary.json has merged == 1."""
    best = (None, None, None)
    for attempt, folder in runs.get(key, []):
        summary = _json(folder / "train-summary.json")
        if isinstance(summary, dict) and summary.get("merged") == 1 and (best[1] is None or attempt > best[1]):
            best = (folder, attempt, summary)
    return best


def lineage_matches(model, key: str, attempt: int) -> bool:
    """Whether the end of a sweep's `model` path is runs/<key>-a<attempt>/hf-step<digits>."""
    if not isinstance(model, str):
        return False
    pattern = r"(?:^|/)runs/%s-a%d/hf-step\d+$" % (re.escape(key), attempt)
    return re.search(pattern, model.replace("\\", "/").rstrip("/")) is not None


def prefix_check_file(root: Path):
    """The highest attempt of the reference's prefix check, or None."""
    best = (None, None)
    folder = root / "k1c" / "control" / "report-prefix"
    if folder.is_dir():
        for child in folder.iterdir():
            match = PREFIX_FILE.match(child.name)
            if match and child.is_file() and (best[1] is None or int(match["attempt"]) > best[1]):
                best = (child, int(match["attempt"]))
    return best[0]


# ------------------------------------------------------------------------------------- the rules
def two_of_three(values: list):
    """True if at least two are True; False if at least two are False or two True can no longer be reached; else None."""
    yes, no, unknown = values.count(True), values.count(False), values.count(None)
    if yes >= 2:
        return True
    if no >= 2 or yes + unknown < 2:
        return False
    return None


def run_drifts(by_length, by_cost):
    if by_length is True or by_cost is True:
        return True
    if by_length is False and by_cost is False:
        return False
    return None


def cell_decisions(cell: str, runs: list, threshold: int) -> dict:
    reported = [r for r in runs if r["reported"]]
    drifts = [r["drifts"] if r["reported"] else None for r in runs]
    free = two_of_three([None if d is None else not d for d in drifts])
    if len(reported) == len(runs):
        gains = [r["gain"] for r in runs]
        learns = sum(gains) >= len(gains) * threshold and sum(1 for g in gains if g > 0) >= 2
        mean_gain = round(sum(gains) / len(gains), 6)
    else:
        learns, mean_gain = None, None
    preserves = two_of_three([r["preserves"] if r["reported"] else None for r in runs])
    decisions = {"free_of_drift": free, "learns": learns, "preserves": preserves}
    return {"cell": cell, "runs": [r["key"] for r in runs], "runs_reported": len(reported), "mean_gain": mean_gain,
            **decisions, "decided": int(all(v is not None for v in decisions.values())),
            "reading": reading(cell, decisions, runs)}


def reading(cell: str, decisions: dict, runs: list) -> str:
    """The first sentence of the fixed table that applies."""
    if cell == "lr6" and decisions["free_of_drift"] is True:
        return READ_LR6
    if cell == "gate" and decisions["free_of_drift"] is True and decisions["preserves"] is True:
        return READ_GATE
    if decisions["learns"] is False or decisions["preserves"] is False:
        return READ_NOT_USEFUL
    undecided = [name for name in DECISIONS if decisions[name] is None]
    if undecided:
        missing = []
        unreported = [r["key"] for r in runs if not r["reported"]]
        if unreported:
            missing.append("runs not reported: %s" % ", ".join(unreported))
        if decisions["free_of_drift"] is None:
            unknown = [r["key"] for r in runs if r["reported"] and r["drifts"] is None]
            if unknown:
                missing.append("drift not known for %s" % ", ".join(unknown))
        return READ_NOT_DECIDED + ", ".join(undecided) + (" (%s)" % "; ".join(missing) if missing else "") + "."
    return READ_NONE


# ------------------------------------------------------------------------------------------- build
def _training(runs: dict, key: str) -> dict:
    folder, attempt, summary = run_attempt(runs, key)
    out = {"run_attempt": attempt, "run": str(folder) if folder else None,
           "attempts_found": sorted(a for a, _ in runs.get(key, []))}
    for field in SUMMARY_FIELDS:
        out[field] = (summary or {}).get(field)
    rows, bad = read_metrics(folder / "metrics.jsonl") if folder else ([], 0)
    # The pre-registered endpoints, exactly: the mean answer length at training step 1 and at the run's last step (the
    # `steps` its summary records: 40). A run whose metrics lack either has no length ratio; the nearest step that
    # happens to be there is NOT used in its place (a missing step 1 could otherwise read as "no drift").
    at = {row["step"]: row["data"][LENGTH_KEY] for row in rows if _finite(row["data"].get(LENGTH_KEY))}
    steps = (summary or {}).get("steps")
    first, last = at.get(1), at.get(steps) if _finite(steps) else None
    out.update({"metrics_unreadable_lines": bad,
                "length_first": first, "length_first_step": 1 if first is not None else None,
                "length_last": last, "length_last_step": steps if last is not None else None,
                "length_ratio": round(last / first, 6) if _finite(first) and _finite(last) and first else None})
    return out


def _sweep_of(sweeps: dict, stem: str) -> tuple:
    """(sweep, attempt, None) or (None, attempt, why)."""
    folder, attempt = latest(sweeps, stem, "sweep.json")
    if folder is None:
        return None, None, None
    try:
        sweep = budget_report.read_sweep(folder)
    except (budget_report.BudgetReportError, KeyError, TypeError) as exc:
        return None, attempt, "%s-a%d cannot be read: %s" % (stem, attempt, exc)
    sweep["_attempt"] = attempt
    return sweep, attempt, None


def build(args) -> dict:
    serving, diagnostic = int(args.serving), int(args.diagnostic)
    if serving >= diagnostic:
        raise ControlReportError("--serving (%d) must be below --diagnostic (%d)" % (serving, diagnostic))
    root = Path(args.work)
    if not root.is_dir():
        raise ControlReportError("no such work tree: %s" % root)
    sweeps = attempts(root / "k1c" / "control" / "report-sweep")
    runs = attempts(root / "runs")

    ref, ref_attempt, why = _sweep_of(sweeps, "%s-%s" % (BASE, BED))
    if ref is None:
        raise ControlReportError(why or "%s holds no sweep of the untrained model (looked for k1c/control/report-sweep/%s-%s-a<N>/"
                                 "sweep.json): every decision is read against it" % (root, BASE, BED))
    try:
        ref_b, ref_h = budget_report.at(ref, serving, "the reference"), budget_report.at(ref, diagnostic, "the reference")
    except budget_report.BudgetReportError as exc:
        raise ControlReportError(str(exc)) from exc
    n = ref.get("n")
    if not _finite(n) or n <= 0:
        raise ControlReportError("the reference sweep %s holds no questions (n %r)" % (ref["_path"], n))
    # The reference's prefix check is scheduled by the campaign and REQUIRED here: absent, failed, or made on another
    # scoring than the one swept, no budget read against the reference can be used (budget_report.read_checks).
    check_file = prefix_check_file(root) or (root / "k1c" / "control" / "report-prefix" / "prefix-base17b-gsm8k-a1.json")
    try:
        check = budget_report.read_checks(["ref=%s" % check_file], {"ref": ref}, serving)["ref"]
    except budget_report.BudgetReportError as exc:
        raise ControlReportError(str(exc)) from exc
    if not check["passed"]:
        raise ControlReportError("the reference's prefix check %s does not qualify it (%s; compared %s, agreement %s, same machine %s, "
                                 "same context length %s): no budget read against it can be used"
                                 % (check["file"], check["why"], check["compared"], check["text_agreement"], check["same_machine"],
                                    check["same_max_model_len"]))
    ref_machine = budget_report.machine_id(ref)
    threshold = budget_report.ceil_frac(n, TOLERANCE_PER_100, 100)
    S_rB = ref_b["correct_strict"]
    ref_tpc = ref_b.get("tokens_per_correct_strict")

    records, lineage_problems, identity_failures = [], [], 0
    for cell in CELLS:
        for seed in SEEDS:
            key = "%s-seed%d" % (cell, seed)
            rec = {"key": key, "cell": cell, "seed": seed, "reported": 0, "reason": None, **_training(runs, key)}
            for name in ("gain", "learns", "preserves", "drift_by_length", "drift_by_cost", "drifts"):
                rec[name] = None
            sweep, sweep_attempt, why = _sweep_of(sweeps, "%s-%s" % (key, BED))
            rec.update({"sweep_attempt": sweep_attempt, "sweep": sweep["_path"] if sweep else None,
                        "sweep_model": sweep.get("model") if sweep else None,
                        "machine_id": budget_report.machine_id(sweep) if sweep else None})
            records.append(rec)
            if sweep is None:
                rec["reason"] = why or ("missing: no sweep at k1c/control/report-sweep/%s-%s-a<N>/sweep.json%s"
                                        % (key, BED, "" if rec["run_attempt"] else ", and no merged training run"))
                continue
            if rec["run_attempt"] is None:
                rec["reason"] = ("lineage: %s-%s-a%d scored %s, but no attempt of %s has a train-summary.json with merged 1"
                                 % (key, BED, sweep_attempt, rec["sweep_model"], key))
                lineage_problems.append({"key": key, "reason": rec["reason"]})
                continue
            if not lineage_matches(rec["sweep_model"], key, rec["run_attempt"]):
                rec["reason"] = ("lineage: %s-%s-a%d scored %s, not runs/%s-a%d/hf-step<N> (the merged attempt of record)"
                                 % (key, BED, sweep_attempt, rec["sweep_model"], key, rec["run_attempt"]))
                lineage_problems.append({"key": key, "reason": rec["reason"]})
                continue
            if sweep.get("bed") != ref.get("bed") or not sweep.get("items_sha256") or sweep.get("items_sha256") != ref.get("items_sha256") \
                    or sweep.get("n") != n:
                rec["reason"] = ("other items: the sweep scored %s (items_sha256 %s, n %s), the reference %s (%s, n %s)"
                                 % (sweep.get("bed"), str(sweep.get("items_sha256"))[:12], sweep.get("n"),
                                    ref.get("bed"), str(ref.get("items_sha256"))[:12], n))
                continue
            rec["same_machine"] = ref_machine is not None and rec["machine_id"] == ref_machine
            if not rec["same_machine"] and not args.allow_different_machines:
                rec["reason"] = ("another machine: the sweep's machine id is %s, the reference's %s (pass --allow-different-machines "
                                 "to compare anyway; it is recorded)" % (rec["machine_id"], ref_machine))
                continue
            try:
                acc = budget_report.account(ref, sweep, serving, diagnostic, key)
            except budget_report.BudgetReportError as exc:
                rec["reason"] = "not compared: %s" % exc
                continue
            except (KeyError, TypeError) as exc:
                rec["reason"] = "not compared: a sweep lacks the count %s" % exc
                continue
            except AssertionError as exc:
                identity_failures += 1
                rec["reason"] = "the accounting identity failed: %s" % exc
                continue
            if not acc["accounting_exact"]:
                identity_failures += 1
            S_aB = acc["scores"]["strict_serving"]
            run_tpc = acc["tokens_per_correct_serving"]["after"]
            gain = S_aB - S_rB
            by_length = (rec["length_last"] > LENGTH_FACTOR * rec["length_first"]
                         if _finite(rec["length_first"]) and _finite(rec["length_last"]) else None)
            # the same None cases as account's ratio; the comparison is made on the unrounded counts
            by_cost = (run_tpc * COST_FACTOR[1] > COST_FACTOR[0] * ref_tpc if ref_tpc and _finite(run_tpc) else None)
            rec.update({"reported": 1, "scores": acc["scores"], "F": acc["F"], "residual": acc["residual"],
                        "budget": acc["budget"], "extraction": acc["extraction"], "accounting_exact": acc["accounting_exact"],
                        "label": acc["label"], "label_canonical": acc["label_canonical"],
                        "cut_share_diagnostic": acc["cut_share_diagnostic"],
                        "tokens_per_correct_serving": run_tpc, "tokens_per_correct_ratio": acc["tokens_per_correct_serving"]["ratio"],
                        "gain": gain, "learns": gain >= threshold, "preserves": S_aB >= S_rB - threshold,
                        "drift_by_length": by_length, "drift_by_cost": by_cost, "drifts": run_drifts(by_length, by_cost)})

    cells = {cell: cell_decisions(cell, [r for r in records if r["cell"] == cell], threshold) for cell in CELLS}
    # three-valued: unknown is not evidence that some cell learned, and it is not evidence that none did
    learns = [c["learns"] for c in cells.values()]
    no_cell_learns = False if any(v is True for v in learns) else (True if all(v is False for v in learns) else None)
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "work": str(root.resolve()), "serving": serving, "diagnostic": diagnostic, "n": n, "threshold": threshold,
            "reference": {"sweep": ref["_path"], "attempt": ref_attempt, "machine_id": ref_machine, "model": ref.get("model"),
                          "scores": {"strict_serving": S_rB, "canonical_serving": ref_b["correct_canonical"],
                                     "strict_diagnostic": ref_h["correct_strict"], "canonical_diagnostic": ref_h["correct_canonical"]},
                          "cut_diagnostic": ref_h.get("cut"), "tokens_per_correct_serving": ref_tpc},
            "reference_prefix_check": check,
            "different_machines_allowed": bool(args.allow_different_machines),
            "runs": records, "cells": cells,
            "not_reported": [{"key": r["key"], "reason": r["reason"]} for r in records if not r["reported"]],
            "lineage_problems": lineage_problems,
            "no_cell_learns": no_cell_learns, "no_cell_learns_sentence": NO_CELL_LEARNS if no_cell_learns else None,
            "runs_expected": len(run_keys()), "runs_reported": sum(r["reported"] for r in records),
            "cells_decided": sum(c["decided"] for c in cells.values()),
            "accounting_exact": 0 if identity_failures else 1, "lineage_ok": 0 if lineage_problems else 1,
            "statements": [MECHANICAL, ONE_TASK]}


# ------------------------------------------------------------------------------------------ render
def _v(value, pattern="%d") -> str:
    return "-" if not _finite(value) else pattern % value


def _yes(value) -> str:
    return "-" if value is None else ("yes" if value else "no")


def render(report: dict) -> str:
    B, H, t = report["serving"], report["diagnostic"], report["threshold"]
    ref = report["reference"]
    s = ref["scores"]
    check = report["reference_prefix_check"]
    lines = ["# Small-model control: the pre-registered decisions", "",
             "Work tree `%s`. Qwen3-1.7B on GSM8K, %d questions; serving budget B = %d, diagnostic H = %d. The reference is "
             "the untrained model (`%s`): S(%d) %d, E(%d) %d, S(%d) %d, E(%d) %d. Threshold t = %d questions (five per 100, "
             "rounded up). Its prefix check: %s. Runs reported: %d of %d; cells decided: %d of %d; accounting identity "
             "exact: %s; lineage clean: %s."
             % (report["work"], report["n"], B, H, ref["sweep"], B, s["strict_serving"], B, s["canonical_serving"],
                H, s["strict_diagnostic"], H, s["canonical_diagnostic"], t,
                "none found (decides nothing)" if check is None else "passed (agreement %s)" % check["text_agreement"],
                report["runs_reported"], report["runs_expected"], report["cells_decided"], len(report["cells"]),
                "yes" if report["accounting_exact"] else "NO", "yes" if report["lineage_ok"] else "NO"), ""]
    if report["different_machines_allowed"]:
        lines += ["--allow-different-machines was given: runs scored on another machine than the reference are compared "
                  "anyway (`same_machine` false on their rows).", ""]
    lines += ["## Runs", "",
              "gain = S(%d) - S_ref(%d); learns: gain >= %d; preserves: S(%d) >= S_ref(%d) - %d. Length is the mean answer "
              "length in training at step 1 and at the run's last step (a dash when the metrics lack that step); drift by length: last > 2 x first. Cost is tokens "
              "per strict-correct answer at %d as a ratio to the reference's; drift by cost: over 1.5. F = S_ref(%d) - S(%d) "
              "= residual + budget + extraction (kit/budget_report.py)." % (B, B, t, B, B, t, B, B, B), "",
              "| cell | seed | S(%d) | E(%d) | S(%d) | E(%d) | gain | learns | preserves | length first | last | ratio | cost ratio "
              "| drifts | F = residual + budget + extraction | label |" % (B, B, H, H),
              "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|---|"]
    for r in report["runs"]:
        if not r["reported"]:
            lines.append("| %s | %d | - | - | - | - | - | - | - | %s | %s | %s | - | - | - | not reported |"
                         % (r["cell"], r["seed"], _v(r["length_first"], "%.1f"), _v(r["length_last"], "%.1f"), _v(r["length_ratio"], "%.2f")))
            continue
        sc = r["scores"]
        drift = "%s (length %s, cost %s)" % (_yes(r["drifts"]), _yes(r["drift_by_length"]), _yes(r["drift_by_cost"]))
        lines.append("| %s | %d | %d | %d | %d | %d | %+d | %s | %s | %s | %s | %s | %s | %s | %+d = %+d + %+d + %+d | %s (canonical %s) |" % (
            r["cell"], r["seed"], sc["strict_serving"], sc["canonical_serving"], sc["strict_diagnostic"], sc["canonical_diagnostic"],
            r["gain"], _yes(r["learns"]), _yes(r["preserves"]), _v(r["length_first"], "%.1f"), _v(r["length_last"], "%.1f"),
            _v(r["length_ratio"], "%.2f"), _v(r["tokens_per_correct_ratio"], "%.2f"), drift,
            r["F"], r["residual"], r["budget"], r["extraction"], r["label"], r["label_canonical"]))
    lines += ["", "## Cells", "",
              "Free of drift: at least two runs do not drift. Learns: all three runs reported, mean gain >= %d and at least "
              "two gains above zero. Preserves: at least two runs preserve. \"-\" is not decided." % t, "",
              "| cell | runs reported | mean gain | free of drift | learns | preserves | reading |", "|---|---:|---:|---|---|---|---|"]
    for c in report["cells"].values():
        lines.append("| %s | %d | %s | %s | %s | %s | %s |" % (c["cell"], c["runs_reported"], _v(c["mean_gain"], "%+.2f"),
                                                            _yes(c["free_of_drift"]), _yes(c["learns"]), _yes(c["preserves"]), c["reading"]))
    if report["no_cell_learns"]:
        lines += ["", report["no_cell_learns_sentence"]]
    lines += ["", "## Not reported", ""]
    lines += ["- %s: %s" % (r["key"], r["reason"]) for r in report["not_reported"]] or ["None: every run is reported."]
    lines += ["", MECHANICAL, "", ONE_TASK]
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The pre-registered decisions of the small-model control, from one work tree.")
    parser.add_argument("--work", required=True, help="the control campaign's work tree (runs/, k1c/control/)")
    parser.add_argument("--serving", type=int, required=True, help="the serving budget B, e.g. 2048")
    parser.add_argument("--diagnostic", type=int, required=True, help="the diagnostic budget H, e.g. 8192")
    parser.add_argument("--allow-different-machines", action="store_true",
                        help="report runs scored on another machine than the reference; recorded")
    parser.add_argument("--out", required=True, help="a new directory")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: choose a new --out" % out)
    try:
        report = build(args)
    except ControlReportError as exc:
        raise SystemExit(str(exc))
    out.mkdir(parents=True)
    (out / "control-report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "control-report.md").write_text(render(report), encoding="utf-8")
    print("runs reported %d of %d; cells decided %d; accounting exact %s; lineage ok %s"
          % (report["runs_reported"], report["runs_expected"], report["cells_decided"], bool(report["accounting_exact"]),
             bool(report["lineage_ok"])))
    for c in report["cells"].values():
        print("%s: %s" % (c["cell"], c["reading"]))
    print("wrote", out / "control-report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
