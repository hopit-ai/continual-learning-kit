#!/usr/bin/env python3
"""The tables of the two-task pilot: one work tree in, one report out, on whatever part of the tree exists.

    python pilot_report.py --work $WORK --serving 2048 --diagnostic 8192 --out $WORK/pilot-report-a1
                           [--allow-different-machines]

THE PILOT. Qwen3-8B is trained on two tasks one after the other, in both orders, under four recipes, and the report
asks whether what looks like forgetting of the first task is the answer no longer fitting a serving token budget.
The tasks are the beds `chemistry` (four-option multiple choice) and `toolalpaca` (tool calls).

THE LAYOUT IS FIXED, and this file reads nothing else (the pilot campaign writes exactly this):
    recipes      g8, g32, sema, sfrz; replicates r1, and r2 for g8 and sema only
    chains       chem      trained on chemistry alone, from the base model
                 tool      trained on toolalpaca alone, from the base model
                 chemtool  trained on toolalpaca STARTING FROM the "<recipe>-chem-<rep>" checkpoint
                 toolchem  trained on chemistry STARTING FROM the "<recipe>-tool-<rep>" checkpoint
    run key      "<recipe>-<chain>-<rep>", e.g. "g8-chemtool-r1"; the untrained model's key is "base8b"
    <work>/runs/<key>-a<N>/                         run-summary.json (`merged`, `model_dir`), metrics.jsonl
    <work>/k8b/report-rollouts/<key>-a<N>/          rollout-stats.json            (kit/rollout_stats.py)
    <work>/k8b/report-sweep/<key>-chemistry-a<N>/   sweep.json of kit/cap_sweep.py bed (also for base8b)
    <work>/k8b/report-sweep/<key>-toolalpaca-a<N>/  the same
    <work>/k8b/report-sweep/<key>-panel-a<N>/       sweep.json of kit/cap_sweep.py panel (also for base8b)
    <work>/k8b/report-prefix/prefix-<key>-<task>-a<N>.json   kit/cap_sweep.py check of a bed sweep's prefix reuse
For a sweep and a prefix check, the highest attempt N wins. A missing piece is reported as missing ("-") and never
fails the report: it is made to be run on trees where only some runs finished.

LINEAGE. Every sweep.json names the model folder it scored (`model`). `lineage(path)` reads `(key, attempt)` from a
path ending in `runs/<key>-a<N>/hf-step<S>`, and None from any other path (the untrained model). The run of record of a
key is `run_attempt(key)`: its highest attempt whose run-summary.json says `merged` 1. A sweep of a run key K is used
only if `lineage(model) == (K, run_attempt(K))`: a retried run is never compared with scorings of an earlier attempt.
A sweep of base8b is used if its `model` is not one of the runs (a base sweep with no `model` is accepted: in this
layout it can only be the untrained model). For retention only, a stage-2 run must have started from the run of record
of its first stage: `lineage(model_dir)` of its run summary at `run_attempt` must be `(K1, run_attempt(K1))`, or the
row is not scored. Every sweep left out and every broken stage link is a line of `lineage_problems`, and the rows that
would have used it say "lineage: <that line>". `lineage_ok` is 1 when there is none. Table 7 reads each run at
`run_attempt` when there is one, else at its highest attempt folder, whatever that holds.

PREFIX QUALIFICATION. Every budget of a sweep is read from a prefix of one long generation; kit/cap_sweep.py check
compares those prefixes with a generation capped short. A check PASSES when it compared at least one answer, the text
agrees on at least 99% of them, and both scorings shared a machine and a max_model_len. The bed sweep of a (key, task)
whose check FAILED is used nowhere, and the rows that would have used it say "prefix reuse failed for <key> on <task>
(agreement <x>)". A missing check leaves nothing out. Files of that folder not named `prefix-...` are not checks.

ONE MACHINE. Bed counts move by several points across machines before any training, so two sweeps whose machine ids
differ, or of which either has none, are not compared anywhere: no acquisition decision, retention row, second-task
difference, panel difference or ranking is computed across machines; each row keeps its own scores and says
"not compared: scored on different machines (<ref id> vs <after id>)". --allow-different-machines restores the
comparison, records `same_machine` false on each such row, and is recorded as `different_machines_allowed`.

THE SCORES are budget_report's throughout: S is the strict count (the bed's own rule), E the canonical count
(kit/canonical.py's gold-blind reading), B the serving budget, H the diagnostic one. Every comparison of two bed
sweeps goes through kit/budget_report.py's `account`, so the identity F = residual + budget + extraction is that
file's, not re-derived here; `accounting_exact` is 1 unless an identity failed. Two sweeps of other items are not
compared (the row says so).

THE TABLES.
  1. BASE. The untrained model per task: the four scores, the share of answers cut at H, tokens per strict-correct
     answer at B.
  2. ACQUISITION (stage 1). Each "<recipe>-chem-<rep>" on chemistry and "<recipe>-tool-<rep>" on toolalpaca against
     the base: the four scores, the gain at S(B) in questions and per 100 questions, `acquired` (gain per 100 at
     least --acquired-per-100: the pre-registered rule, the serving cohort), the cut share at H, tokens per correct at
     B and its ratio to the base's. Beside it, never in its place, the same threshold on the canonical score:
     `gain_serving_canonical` and `passes_canonical_serving` at E(B), `gain_diagnostic_canonical`,
     `gain_diagnostic_per_100` and `acquired_diagnostic` at E(H) (the diagnostic cohort). A row not acquired says
     "gain visible at E(B), not at S(B)" or "gain visible only at E(H)" in its note when that is so. Under the table:
     ToolAlpaca's 68 questions put a paired difference under about 12 per 100 inside noise.
  3. RETENTION (stage 2). chemtool on chemistry: reference "<recipe>-chem-<rep>", after "<recipe>-chemtool-<rep>";
     toolchem on toolalpaca: reference "<recipe>-tool-<rep>", after "<recipe>-toolchem-<rep>". F and its three
     terms, the after scores, whether the first task had been acquired (table 2), and the retained share
     S_after(B) / S_ref(B). `first_task_acquired`, `defined` and `status` are the serving cohort: a row whose first
     task was not acquired is printed and marked "not acquired: retention is not defined", because a task that was
     not acquired cannot be said to be kept; its accounting terms are still printed, its retained share is not.
     `first_task_acquired_diagnostic` and `defined_diagnostic` (scored, and the first task acquired at E(H)) are the
     diagnostic cohort. The two cohorts are never pooled. A defined row is `access_sensitive` (pre-registered) when
     F is at least 5 per 100 questions and budget + extraction is at least half of F.
  4. SECOND TASK. The task trained second, after the chain, against the same task trained alone (same recipe and
     replicate): the four scores of each and the difference at S(B).
  5. RETENTION BY BUDGET, once per cohort (`serving_cohort`, `diagnostic_cohort`). Over the rows of table 3 defined
     in that cohort, at every budget read by all their sweeps: the strict loss S_ref(N) - S_after(N) per row; each
     recipe's mean over ITS OWN rows, marked `comparable` false (two recipes' own rows can be different tasks); and
     for every pair of recipes the mean loss per 100 of each over the (chain, replicate) rows defined for BOTH, which
     of the two has the smaller mean at B and at H ("tie" when equal), and whether that order is strictly reversed.
     `ranking_differs` is true if some pair with shared rows is reversed, false if some pair has shared rows and none
     is, null otherwise. This is descriptive, from at most two replicates, and needs independent confirmation.
  6. PANEL. The general panel per run at B and H, strict, per panel, against the base.
  7. TRAINING. Per run, from metrics.jsonl and run-summary.json: the mean answer length at the first step, at step 20
     and at the last step, its maximum and the step of it, the maximum clip ratio, the mean training score first and
     last, the trainer's own validation accuracy at each validation step (sampled, 16 answers a question, the authors'
     protocol; not comparable with the greedy scores of the other tables), seconds and GPU-hours (seconds x n_gpus /
     3600); and from rollout-stats.json the share of prompt groups with no reward spread at the first and last step.
  8. DECISIONS. The pre-registered rules (plan v3, section 4) applied mechanically, deciding nothing by themselves:
     whether g8 and sema each acquired both tasks at S(B) in their first run (null if a row is not scored); per
     recipe g8 and sema and per chain, whether both runs' retention rows are defined, lose 5 per 100, and are
     access-sensitive; whether any defined row loses 5 per 100 and whether such losses are only in g32; and
     `on_to_package_4`: true iff a published recipe acquires both in its first run AND some recipe and chain of g8 or
     sema loses 5 per 100 in both runs; false with the pre-registered reason when either half is settled false; null
     when the data to settle it are missing (`why` says which).

`pilot-report.json` carries, at its top level, the numbers a campaign bar reads, all ints: `runs_found`,
`stage1_scored`, `stage2_scored`, `retention_rows_defined`, `retention_rows_defined_diagnostic`, `accounting_exact`,
`lineage_ok`, `prefix_checks_found`, `prefix_checks_failed`, `prefix_checks_missing_count`; and `lineage_problems`, `prefix_checks`, `decision`,
`different_machines_allowed`. `pilot-report.md` gives each table with one sentence saying what it is, a section on
the prefix checks before table 1, and two caveats verbatim; it states definitions and numbers, no interpretation.

REFUSED: an existing --out, a --work that is not a folder, a work tree with no base sweep at all (no chemistry,
toolalpaca or panel sweep of base8b), and B not below H. Standard library only; kit/budget_report.py is loaded by file
path, so `kit/` works when exported alone. Nothing is overwritten.
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
SCHEMA = "kit-pilot-report.v1"
RECIPES = ("g8", "g32", "sema", "sfrz")
PUBLISHED = ("g8", "sema")                                      # the recipes the package-4 rule reads
REPLICATES = {"g8": ("r1", "r2"), "g32": ("r1",), "sema": ("r1", "r2"), "sfrz": ("r1",)}
CHAINS = ("chem", "tool", "chemtool", "toolchem")
BASE = "base8b"
TASKS = ("chemistry", "toolalpaca")
STAGE1 = {"chem": "chemistry", "tool": "toolalpaca"}           # chain -> the task it trains, from the base model
# chain -> (the chain that trained the first task, the first task, the chain that trained the second task alone, the second task)
STAGE2 = {"chemtool": ("chem", "chemistry", "tool", "toolalpaca"),
          "toolchem": ("tool", "toolalpaca", "chem", "chemistry")}
DEFAULT_ACQUIRED_PER_100 = 5
LOSS_PER_100 = 5                                                # the pre-registered loss of section 4, fixed
PREFIX_AGREEMENT = 0.99
#: The prefix checks the campaign schedules (kit/campaigns/k8b-pilot.yaml): the untrained model on both tasks, as
#: pilots, and the first run's Chemistry-then-ToolAlpaca checkpoint of the two published recipes on Chemistry. A sweep
#: of one of these is used only with its check; a check that is absent leaves it out exactly as a failed one does.
REQUIRED_PREFIX = (("base8b", "chemistry"), ("base8b", "toolalpaca"), ("g8-chemtool-r1", "chemistry"), ("sema-chemtool-r1", "chemistry"))
VAL_SUFFIX = "/acc/mean@16"
VALIDATION_LABEL = "sampled, 16 answers a question, the authors' protocol; not comparable with the greedy scores above"
LENGTH_STEP = 20
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
LINEAGE = re.compile(r"runs/(?P<key>[^/]+)-a(?P<attempt>\d+)/hf-step(?P<step>\d+)/?$")
PREFIX_FILE = re.compile(r"^prefix-(?P<key>.+)-(?P<task>%s)-a(?P<attempt>\d+)\.json$" % "|".join(TASKS))
NOT_ACQUIRED = "not acquired: retention is not defined"
ACQUISITION_UNKNOWN = "acquisition unknown: retention is not defined"
TOOLALPACA_CAVEAT = ("ToolAlpaca has 68 held-out questions: a paired difference under about 12 per 100 is inside noise, "
                     "and every ToolAlpaca acquisition is to be read with that.")
RECIPE_CAVEAT = ("The four recipes differ in learning rate, minibatch and teacher at once, and have one or two runs each: "
                 "nothing here identifies an effect of an algorithm.")
NO_PUBLISHED_BOTH = "no published recipe acquires both tasks"
NO_REPEATED_LOSS = "no acquired task loses 5 per 100 in both runs of a recipe"
ONLY_G32 = "the only losses are in g32"


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_pilot_report_%s" % name, HERE / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


budget_report = _load("budget_report")


class PilotReportError(ValueError):
    """The work tree cannot support a report; nothing is written."""


def run_keys() -> list:
    return ["%s-%s-%s" % (recipe, chain, rep) for recipe in RECIPES for rep in REPLICATES[recipe] for chain in CHAINS]


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def lineage(path):
    """(key, attempt) of a model folder ending in `runs/<key>-a<N>/hf-step<S>`, else None (the untrained model)."""
    match = LINEAGE.search(path) if isinstance(path, str) else None
    return (match["key"], int(match["attempt"])) if match else None


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


def read_prefix_checks(folder: Path) -> dict:
    """{(key, task): check} from the highest attempt of every `prefix-<key>-<task>-a<N>.json` in `folder`.
    An unreadable file is a check that compared nothing, so it fails."""
    best: dict = {}
    if folder.is_dir():
        for path in folder.iterdir():
            match = PREFIX_FILE.match(path.name)
            if not match or not path.is_file():
                continue
            slot, attempt = (match["key"], match["task"]), int(match["attempt"])
            if slot not in best or attempt > best[slot][0]:
                best[slot] = (attempt, path)
    checks = {}
    for (key, task), (attempt, path) in sorted(best.items()):
        data = _json(path)
        data = data if isinstance(data, dict) else {}
        compared, agreement = data.get("compared"), data.get("text_agreement")
        same_machine, same_length = data.get("same_machine"), data.get("same_max_model_len")
        passed = (_finite(compared) and compared > 0 and _finite(agreement) and agreement >= PREFIX_AGREEMENT
                  and same_machine is True and same_length is True)
        # `long` is the long scoring the check was made on: its folder NAME (not its path, which changes when the tree
        # is packed and moved) must be the folder name of the scoring the sweep read, or the check is of something else.
        long = data.get("long")
        checks[(key, task)] = {"key": key, "task": task, "attempt": attempt, "file": str(path.resolve()),
                               "compared": compared, "text_agreement": agreement, "same_machine": same_machine,
                               "same_max_model_len": same_length, "long_scoring": Path(long).name if isinstance(long, str) and long else None,
                               "short_cap": data.get("short_cap"), "long_cap": data.get("long_cap"), "passed": bool(passed),
                               "bound": None, "used": None}
    return checks


class Work:
    """One work tree: its folders indexed once, each sweep read and admitted once, every unusable piece written down."""

    def __init__(self, root: Path, allow_different_machines: bool = False, serving=None):
        self.root = root
        self.serving = serving                       # the short cap a prefix check must have been made at (None: not checked)
        self.prefix_missing: list = []               # required checks that are absent while their sweep exists
        self.allow_different_machines = bool(allow_different_machines)
        self.runs = attempts(root / "runs")
        self.sweeps = attempts(root / "k8b" / "report-sweep")
        self.rollouts = attempts(root / "k8b" / "report-rollouts")
        self.prefix_checks = read_prefix_checks(root / "k8b" / "report-prefix")
        self.problems: list = []
        self.lineage_problems: list = []
        self.left_out: dict = {}                     # (key, task or "panel") -> why a sweep that exists is not used
        self.identity_failures = 0
        self._bed: dict = {}
        self._panel: dict = {}
        self._merged: dict = {}

    # runs
    def merged_run(self, key: str):
        """(attempt, folder, run summary) of the highest attempt of `key` whose run-summary.json has merged == 1, or None."""
        if key not in self._merged:
            best = None
            for attempt, folder in self.runs.get(key, []):
                summary = _json(folder / "run-summary.json")
                if isinstance(summary, dict) and summary.get("merged") == 1 and (best is None or attempt > best[0]):
                    best = (attempt, folder, summary)
            self._merged[key] = best
        return self._merged[key]

    def run_attempt(self, key: str):
        merged = self.merged_run(key)
        return merged[0] if merged else None

    def lineage_problem(self, key: str, label: str, sweep: dict):
        """Why the sweep of `key` (on task `label`, or "panel") did not score the run of record of `key`, or None."""
        name = "%s %s sweep a%d" % (key, label, sweep["_attempt"])
        model = sweep.get("model")
        found = lineage(model)
        if key == BASE:
            if found is None:
                return None
            return "%s scored %s, a checkpoint of %s attempt %d, not the untrained model" % (name, model, found[0], found[1])
        if not isinstance(model, str) or not model:
            return "%s has no `model` field" % name
        merged = self.run_attempt(key)
        if merged is None:
            return "%s: the run has no merged attempt" % name
        if found is None or found[0] != key:
            return "%s scored %s, which is not a checkpoint of %s" % (name, model, key)
        if found[1] != merged:
            return "%s scored attempt %d of the run; the run's merged attempt is %d" % (name, found[1], merged)
        return None

    def stage_link_problem(self, key: str, first_key: str):
        """None when the run of record of `key` started from the run of record of `first_key`; else why not."""
        merged, first = self.merged_run(key), self.run_attempt(first_key)
        model_dir = merged[2].get("model_dir") if merged else None
        if merged is not None and first is not None and lineage(model_dir) == (first_key, first):
            return None
        return "%s was trained from %s, but the scored attempt of %s is %s" % (
            key, model_dir if isinstance(model_dir, str) and model_dir else "a model its run summary does not name", first_key, first)

    # sweeps
    def _admit(self, key: str, label: str, sweep):
        if sweep is None:
            return None
        problem = self.lineage_problem(key, label, sweep)
        if problem is not None:
            self.lineage_problems.append(problem)
            self.left_out[(key, label)] = "lineage: " + problem
            return None
        # A stage-2 run is admitted into ANY comparison only if it was trained from the scored attempt of its first
        # stage; checked here, on the run summaries alone, so that a missing reference sweep cannot hide it (round 2).
        chain = key.split("-")[1] if key != BASE and len(key.split("-")) == 3 else None
        if chain in STAGE2:
            recipe, _chain, rep = key.split("-")
            link = self.stage_link_problem(key, "%s-%s-%s" % (recipe, STAGE2[chain][0], rep))
            if link is not None:
                if link not in self.lineage_problems:
                    self.lineage_problems.append(link)
                self.left_out[(key, label)] = "lineage: " + link
                return None
        check = self.prefix_checks.get((key, label))
        if check is None:
            if (key, label) in REQUIRED_PREFIX:
                self.prefix_missing.append("%s on %s" % (key, label))
                self.left_out[(key, label)] = "prefix check missing for %s on %s: its curve is not qualified" % (key, label)
                return None
            return sweep
        scoring = Path(sweep["scoring"]).name if isinstance(sweep.get("scoring"), str) and sweep.get("scoring") else None
        check["bound"] = bool(check["long_scoring"] is not None and check["long_scoring"] == scoring and check["long_cap"] == sweep.get("cap")
                              and (self.serving is None or check["short_cap"] == self.serving))
        check["used"] = bool(check["passed"] and check["bound"])
        if not check["passed"]:
            self.left_out[(key, label)] = "prefix reuse failed for %s on %s (agreement %s)" % (key, label, check["text_agreement"])
            return None
        if not check["bound"]:
            self.left_out[(key, label)] = ("the prefix check of %s on %s is of another scoring (%s at caps %s and %s; the sweep read %s at %s, "
                                           "serving %s): its curve is not qualified"
                                           % (key, label, check["long_scoring"], check["short_cap"], check["long_cap"], scoring, sweep.get("cap"), self.serving))
            return None
        return sweep

    def why_not(self, key: str, label: str):
        """Why the sweep of `key` on `label` (a task, or "panel") exists and is not used; None if used or missing."""
        self.panel(key) if label == "panel" else self.bed(key, label)
        return self.left_out.get((key, label))

    def bed(self, key: str, task: str):
        """The bed sweep of `key` on `task` (budget_report.read_sweep), or None (missing, unreadable or not admitted)."""
        if (key, task) not in self._bed:
            self._bed[(key, task)] = self._admit(key, task, self._read_bed(key, task))
        return self._bed[(key, task)]

    def _read_bed(self, key: str, task: str):
        stem = "%s-%s" % (key, task)
        folder, attempt = latest(self.sweeps, stem, "sweep.json")
        if folder is None:
            return None
        try:
            sweep = budget_report.read_sweep(folder)
        except (budget_report.BudgetReportError, KeyError, TypeError) as exc:
            self.problems.append("%s-a%d left out: %s" % (stem, attempt, exc))
            return None
        if sweep.get("bed") not in (None, task):
            self.problems.append("%s-a%d left out: it swept the %s bed, not %s" % (stem, attempt, sweep.get("bed"), task))
            return None
        sweep["_attempt"] = attempt
        return sweep

    def panel(self, key: str):
        """The panel sweep of `key`, with `_by_budget`, or None."""
        if key not in self._panel:
            self._panel[key] = self._admit(key, "panel", self._read_panel(key))
        return self._panel[key]

    def _read_panel(self, key: str):
        stem = "%s-panel" % key
        folder, attempt = latest(self.sweeps, stem, "sweep.json")
        if folder is None:
            return None
        sweep = _json(folder / "sweep.json")
        if not isinstance(sweep, dict) or sweep.get("kind") != "panel":
            self.problems.append("%s-a%d left out: not a panel sweep" % (stem, attempt))
            return None
        sweep["_by_budget"] = {entry.get("budget"): entry for entry in sweep.get("per_budget") or []
                               if isinstance(entry, dict) and isinstance(entry.get("panels"), dict)}
        sweep["_attempt"], sweep["_path"] = attempt, str((folder / "sweep.json").resolve())
        return sweep

    # comparisons
    def machine_refusal(self, ref: dict, after: dict):
        """Why two sweeps are not compared across machines, or None (one machine, or --allow-different-machines)."""
        ref_machine, after_machine = budget_report.machine_id(ref), budget_report.machine_id(after)
        if (ref_machine is not None and ref_machine == after_machine) or self.allow_different_machines:
            return None
        return "not compared: scored on different machines (%s vs %s)" % (ref_machine, after_machine)

    def account(self, ref: dict, after: dict, serving: int, diagnostic: int, name: str) -> tuple:
        """(budget_report.account record, None), or (None, why it could not be made)."""
        if not ref.get("items_sha256") or ref.get("items_sha256") != after.get("items_sha256") or ref.get("n") != after.get("n"):
            return None, "not compared: the two sweeps scored other items (items_sha256 %s vs %s)" % (
                str(ref.get("items_sha256"))[:12], str(after.get("items_sha256"))[:12])
        refused = self.machine_refusal(ref, after)
        if refused is not None:
            return None, refused
        try:
            record = budget_report.account(ref, after, serving, diagnostic, name)
        except budget_report.BudgetReportError as exc:
            return None, "not compared: %s" % exc
        except (KeyError, TypeError) as exc:
            return None, "not compared: a sweep lacks the count %s" % exc
        except AssertionError as exc:
            self.identity_failures += 1
            return None, "the accounting identity failed: %s" % exc
        if not record.get("accounting_exact"):
            self.identity_failures += 1
        ref_machine, after_machine = budget_report.machine_id(ref), budget_report.machine_id(after)
        record["same_machine"] = ref_machine is not None and ref_machine == after_machine
        record["ref_sweep"], record["ref_machine_id"] = ref["_path"], ref_machine
        return record, None


# ------------------------------------------------------------------------------------- one sweep
def scores_of(sweep: dict, serving: int, diagnostic: int) -> dict:
    """The four scores, the cut share at H and tokens per correct at B of one bed sweep; None where a budget is unread."""
    b, h = sweep["_by_budget"].get(serving), sweep["_by_budget"].get(diagnostic)
    n = sweep.get("n")
    return {"found": 1, "sweep": sweep["_path"], "attempt": sweep.get("_attempt"), "n": n, "model": sweep.get("model"),
            "machine_id": budget_report.machine_id(sweep), "budgets": sorted(sweep["_by_budget"]),
            "scores": {"strict_serving": (b or {}).get("correct_strict"), "canonical_serving": (b or {}).get("correct_canonical"),
                       "strict_diagnostic": (h or {}).get("correct_strict"), "canonical_diagnostic": (h or {}).get("correct_canonical")},
            "cut_diagnostic": (h or {}).get("cut"),
            "cut_share_diagnostic": round(h["cut"] / n, 6) if h and _finite(h.get("cut")) and n else None,
            "tokens_per_correct_serving": (b or {}).get("tokens_per_correct_strict")}


def _missing(looked_for: str) -> dict:
    return {"found": 0, "looked_for": looked_for}


def _bed_folder(work: Work, key: str, task: str) -> str:
    return str(work.root / "k8b" / "report-sweep" / ("%s-%s-a<N>" % (key, task))) + "/sweep.json"


def _per_100(count, n):
    return round(100 * count / n, 3) if _finite(count) and n else None


def _absent_note(work: Work, absent: list, task: str):
    """The lineage or prefix reasons of the absent sweeps [(key, sweep)], each missing one named; None if none has a reason."""
    reasons = [(key, work.why_not(key, task)) for key, sweep in absent if sweep is None]
    if not any(reason for _key, reason in reasons):
        return None
    return "; ".join(reason or "missing: " + _bed_folder(work, key, task) for key, reason in reasons)


def loses(row: dict) -> bool:
    """F of at least 5 per 100 questions (section 4)."""
    return row["F"] * 100 >= LOSS_PER_100 * row["n"]


def access_sensitive(row: dict) -> bool:
    """F of at least 5 per 100 questions, and the budget and extraction terms together at least half of F (section 4)."""
    return loses(row) and 2 * (row["budget"] + row["extraction"]) >= row["F"]


# ------------------------------------------------------------------------------------------ tables
def table_base(work: Work, serving: int, diagnostic: int) -> dict:
    out = {}
    for task in TASKS:
        sweep = work.bed(BASE, task)
        out[task] = scores_of(sweep, serving, diagnostic) if sweep else _missing(_bed_folder(work, BASE, task))
        if sweep is None and work.why_not(BASE, task):
            out[task]["note"] = work.why_not(BASE, task)
    return out


def table_acquisition(work: Work, serving: int, diagnostic: int, threshold) -> list:
    rows = []
    for recipe in RECIPES:
        for rep in REPLICATES[recipe]:
            for chain, task in STAGE1.items():
                key = "%s-%s-%s" % (recipe, chain, rep)
                after, ref = work.bed(key, task), work.bed(BASE, task)
                row = {"key": key, "recipe": recipe, "replicate": rep, "chain": chain, "task": task, "scored": 0,
                       "acquired": None, "passes_canonical_serving": None, "acquired_diagnostic": None, "note": None}
                if after is None:
                    row.update(_missing(_bed_folder(work, key, task)))
                    row["note"] = work.why_not(key, task) or "missing"
                    rows.append(row)
                    continue
                row.update(scores_of(after, serving, diagnostic))
                if ref is None:
                    row["note"] = work.why_not(BASE, task) or "no base sweep of %s: the gain is not defined" % task
                    rows.append(row)
                    continue
                record, why = work.account(ref, after, serving, diagnostic, key)
                if record is None:
                    row["note"] = why
                    rows.append(row)
                    continue
                gain, n = -record["F"], record["n"]
                # The same threshold on the canonical score, at B and at H, printed beside `acquired` (the pre-registered
                # rule, the serving cohort) and never used in its place; E(H) defines the diagnostic cohort.
                gain_b = record["scores"]["canonical_serving"] - ref["_by_budget"][serving]["correct_canonical"]
                gain_h = -record["residual"]
                acquired = bool(n) and gain * 100 >= threshold * n
                row.update({"scored": 1, "gain_serving_strict": gain, "gain_per_100": _per_100(gain, n), "acquired": acquired,
                            "gain_serving_canonical": gain_b, "gain_serving_canonical_per_100": _per_100(gain_b, n),
                            "passes_canonical_serving": bool(n) and gain_b * 100 >= threshold * n,
                            "gain_diagnostic_canonical": gain_h, "gain_diagnostic_per_100": _per_100(gain_h, n),
                            "acquired_diagnostic": bool(n) and gain_h * 100 >= threshold * n,
                            "tokens_per_correct_ratio_to_base": record["tokens_per_correct_serving"]["ratio"],
                            "same_machine": record["same_machine"], "accounting_exact": record["accounting_exact"]})
                notes = []
                if not record["same_machine"]:
                    notes.append("scored on another machine than the base (%s vs %s)" % (record["machine_id"], record["ref_machine_id"]))
                if not acquired:
                    if row["passes_canonical_serving"]:
                        notes.append("gain visible at E(%d), not at S(%d)" % (serving, serving))
                    elif row["acquired_diagnostic"]:
                        notes.append("gain visible only at E(%d)" % diagnostic)
                row["note"] = "; ".join(notes) or None
                rows.append(row)
    return rows


def table_retention(work: Work, serving: int, diagnostic: int, acquisition: list) -> list:
    acquired_of = {row["key"]: row["acquired"] for row in acquisition}
    diagnostic_of = {row["key"]: row["acquired_diagnostic"] for row in acquisition}
    rows = []
    for recipe in RECIPES:
        for rep in REPLICATES[recipe]:
            for chain, (first_chain, first_task, _alone, _second) in STAGE2.items():
                key, ref_key = "%s-%s-%s" % (recipe, chain, rep), "%s-%s-%s" % (recipe, first_chain, rep)
                ref, after = work.bed(ref_key, first_task), work.bed(key, first_task)
                acquired, acquired_h = acquired_of.get(ref_key), diagnostic_of.get(ref_key)
                row = {"key": key, "recipe": recipe, "replicate": rep, "chain": chain, "task": first_task,
                       "reference": ref_key, "after": key, "ref_found": int(ref is not None), "after_found": int(after is not None),
                       "first_task_acquired": acquired, "first_task_acquired_diagnostic": acquired_h,
                       "scored": 0, "defined": 0, "defined_diagnostic": 0, "retained_share": None, "access_sensitive": None}
                why, link = None, None
                if ref is not None and after is not None:
                    link = work.stage_link_problem(key, ref_key)        # already refused at admission; kept as a second lock
                    if link is not None:
                        if link not in work.lineage_problems:
                            work.lineage_problems.append(link)
                    else:
                        record, why = work.account(ref, after, serving, diagnostic, key)
                        if record is not None:
                            row.update({"scored": 1, "F": record["F"], "residual": record["residual"], "budget": record["budget"],
                                        "extraction": record["extraction"], "n": record["n"], "scores": record["scores"],
                                        "ref_scores": scores_of(ref, serving, diagnostic)["scores"],
                                        "accounting_exact": record["accounting_exact"], "same_machine": record["same_machine"],
                                        "cut_share_diagnostic": record["cut_share_diagnostic"],
                                        "ref_sweep": record["ref_sweep"], "after_sweep": record["sweep"]})
                absent = _absent_note(work, [(ref_key, ref), (key, after)], first_task)
                if link is not None:
                    status = "lineage: " + link
                elif absent:
                    status = absent
                elif acquired is False:
                    status = NOT_ACQUIRED
                elif ref is None or after is None:
                    status = "missing: %s" % " and ".join(
                        _bed_folder(work, k, first_task) for k, s in ((ref_key, ref), (key, after)) if s is None)
                elif not row["scored"]:
                    status = why
                elif not row["n"]:
                    status = "not compared: the sweeps hold no questions"
                elif acquired is None:
                    status = ACQUISITION_UNKNOWN
                else:
                    status = "defined"
                    row["defined"] = 1
                    ref_b = row["ref_scores"]["strict_serving"]
                    row["retained_share"] = round(row["scores"]["strict_serving"] / ref_b, 4) if ref_b else None
                row["defined_diagnostic"] = int(bool(row["scored"]) and bool(row.get("n")) and acquired_h is True)
                if row["defined"] or row["defined_diagnostic"]:       # the same formula in either cohort; membership stays separate
                    row["access_sensitive"] = access_sensitive(row)
                row["status"] = status
                rows.append(row)
    return rows


def table_second_task(work: Work, serving: int, diagnostic: int) -> list:
    rows = []
    for recipe in RECIPES:
        for rep in REPLICATES[recipe]:
            for chain, (_first, _task, alone_chain, second_task) in STAGE2.items():
                key, alone_key = "%s-%s-%s" % (recipe, chain, rep), "%s-%s-%s" % (recipe, alone_chain, rep)
                chained, alone = work.bed(key, second_task), work.bed(alone_key, second_task)
                row = {"key": key, "task": second_task, "alone": alone_key,
                       "chain_scores": scores_of(chained, serving, diagnostic) if chained else _missing(_bed_folder(work, key, second_task)),
                       "alone_scores": scores_of(alone, serving, diagnostic) if alone else _missing(_bed_folder(work, alone_key, second_task)),
                       "difference_serving_strict": None, "difference_per_100": None, "note": None}
                if chained is None or alone is None:
                    row["note"] = _absent_note(work, [(key, chained), (alone_key, alone)], second_task) or "missing"
                elif not chained.get("items_sha256") or chained.get("items_sha256") != alone.get("items_sha256") or chained.get("n") != alone.get("n"):
                    row["note"] = "not compared: the two sweeps scored other items"
                elif work.machine_refusal(alone, chained) is not None:
                    row["note"] = work.machine_refusal(alone, chained)
                else:
                    notes = []
                    a, b = row["chain_scores"]["scores"]["strict_serving"], row["alone_scores"]["scores"]["strict_serving"]
                    if _finite(a) and _finite(b):
                        row["difference_serving_strict"] = a - b
                        row["difference_per_100"] = _per_100(a - b, chained.get("n"))
                    else:
                        notes.append("not compared: a sweep did not read budget %d" % serving)
                    if budget_report.machine_id(chained) != budget_report.machine_id(alone):
                        notes.append("scored on different machines (%s vs %s)" % (budget_report.machine_id(alone), budget_report.machine_id(chained)))
                    row["note"] = "; ".join(notes) or None
                rows.append(row)
    return rows


def _smaller(a: str, b: str, value_a, value_b) -> str:
    return a if value_a < value_b else b if value_b < value_a else "tie"


def recipe_pair(a: str, b: str, by_recipe: dict, budgets: list, serving: int, diagnostic: int) -> dict:
    """Recipes a and b over the (chain, replicate) rows defined for both: each one's mean loss per 100, the smaller at
    B and at H, and whether that order is strictly reversed (a tie is not a reversal)."""
    mine = {r: {(e["chain"], e["replicate"]): e for e in by_recipe.get(r, [])} for r in (a, b)}
    shared = sorted((k for k in mine[a] if k in mine[b]), key=lambda k: (CHAINS.index(k[0]), k[1]))
    pair = {"a": a, "b": b, "shared": ["%s-%s" % k for k in shared], "n_shared": len(shared),
            "mean_loss_per_100": None, "order_serving": None, "order_diagnostic": None, "reversed": None}
    if shared:
        means = {r: {str(budget): round(sum(mine[r][k]["loss_per_100"][str(budget)] for k in shared) / len(shared), 6)
                     for budget in budgets} for r in (a, b)}
        at_b = _smaller(a, b, means[a][str(serving)], means[b][str(serving)])
        at_h = _smaller(a, b, means[a][str(diagnostic)], means[b][str(diagnostic)])
        pair.update({"mean_loss_per_100": means, "order_serving": at_b, "order_diagnostic": at_h,
                     "reversed": "tie" not in (at_b, at_h) and at_b != at_h})
    return pair


def cohort_by_budget(work: Work, retention: list, serving: int, diagnostic: int, field: str) -> dict:
    """Table 5 over the retention rows with `field` set (`defined` or `defined_diagnostic`)."""
    defined = [row for row in retention if row[field]]
    sweeps = [(row, work.bed(row["reference"], row["task"]), work.bed(row["after"], row["task"])) for row in defined]
    budgets = sorted(set.intersection(*(set(s["_by_budget"]) for _, ref, after in sweeps for s in (ref, after)))) if sweeps else []
    per_row, by_recipe = [], defaultdict(list)
    for row, ref, after in sweeps:
        losses = {budget: ref["_by_budget"][budget]["correct_strict"] - after["_by_budget"][budget]["correct_strict"] for budget in budgets}
        entry = {"key": row["key"], "recipe": row["recipe"], "chain": row["chain"], "replicate": row["replicate"],
                 "task": row["task"], "n": row["n"],
                 "loss": {str(b): v for b, v in losses.items()},
                 "loss_per_100": {str(b): _per_100(v, row["n"]) for b, v in losses.items()}}
        per_row.append(entry)
        by_recipe[row["recipe"]].append(entry)
    recipes = {}
    for recipe in RECIPES:
        mine = by_recipe.get(recipe)
        if not mine:
            continue
        recipes[recipe] = {"rows": len(mine), "keys": [e["key"] for e in mine], "comparable": False,
                           "mean_loss": {str(b): round(sum(e["loss"][str(b)] for e in mine) / len(mine), 6) for b in budgets},
                           "mean_loss_per_100": {str(b): round(sum(e["loss_per_100"][str(b)] for e in mine) / len(mine), 6) for b in budgets}}
    pairs = [recipe_pair(a, b, by_recipe, budgets, serving, diagnostic) for i, a in enumerate(RECIPES) for b in RECIPES[i + 1:]]
    with_rows = [p for p in pairs if p["n_shared"]]
    return {"budgets": budgets, "rows": per_row, "recipes": recipes, "pairs": pairs,
            "ranked_on": "mean strict loss per 100 questions over the rows both recipes share",
            "ranking_differs": any(p["reversed"] for p in with_rows) if with_rows else None}


def table_by_budget(work: Work, retention: list, serving: int, diagnostic: int) -> dict:
    return {"serving_cohort": cohort_by_budget(work, retention, serving, diagnostic, "defined"),
            "diagnostic_cohort": cohort_by_budget(work, retention, serving, diagnostic, "defined_diagnostic")}


def table_panel(work: Work, serving: int, diagnostic: int) -> dict:
    base = work.panel(BASE)
    keys = [BASE] + run_keys()
    names = list((base or {}).get("panels") or [])
    rows = []
    for key in keys:
        sweep = work.panel(key)
        if sweep is None:
            rows.append({"key": key, **_missing(str(work.root / "k8b" / "report-sweep" / ("%s-panel-a<N>" % key)) + "/sweep.json"),
                         "note": work.why_not(key, "panel")})
            continue
        for name in sweep.get("panels") or []:
            if name not in names:
                names.append(name)
        note = None
        comparable = base is not None and key != BASE and base.get("panel_file_sha256") == sweep.get("panel_file_sha256")
        if base is not None and key != BASE:
            if not comparable:
                note = "scored on another panel file than the base: not compared"
            elif work.machine_refusal(base, sweep) is not None:
                comparable, note = False, work.machine_refusal(base, sweep)
            elif budget_report.machine_id(base) != budget_report.machine_id(sweep):
                note = "scored on different machines (%s vs %s)" % (budget_report.machine_id(base), budget_report.machine_id(sweep))
        row = {"key": key, "found": 1, "sweep": sweep["_path"], "attempt": sweep["_attempt"], "model": sweep.get("model"),
               "machine_id": budget_report.machine_id(sweep), "compared_with_base": comparable, "panels": {}}
        for name in list(sweep.get("panels") or []) + ["total"]:
            cell = {}
            for label, budget in (("serving", serving), ("diagnostic", diagnostic)):
                cell[label] = _panel_correct(sweep, budget, name)
                cell[label + "_minus_base"] = None
                if comparable:
                    theirs = _panel_correct(base, budget, name)
                    if _finite(cell[label]) and _finite(theirs):
                        cell[label + "_minus_base"] = cell[label] - theirs
            row["panels"][name] = cell
        if note:
            row["note"] = note
        rows.append(row)
    return {"panels": names, "rows": rows}


def _panel_correct(sweep: dict, budget: int, name: str):
    entry = sweep["_by_budget"].get(budget)
    if not entry:
        return None
    slot = entry.get("total") if name == "total" else entry["panels"].get(name)
    return slot.get("correct") if isinstance(slot, dict) else None


def decision_of(acquisition: list, retention: list, serving: int) -> dict:
    """The pre-registered rules of section 4 applied mechanically to tables 2 and 3. It decides nothing by itself."""
    by_key = {row["key"]: row for row in acquisition}
    by_row = {(row["recipe"], row["chain"], row["replicate"]): row for row in retention}
    why = []
    first = {}
    for recipe in PUBLISHED:
        keys = ["%s-%s-r1" % (recipe, chain) for chain in STAGE1]
        rows = [by_key.get(k) for k in keys]
        unscored = [k for k, r in zip(keys, rows) if r is None or not r.get("scored")]
        # three-valued: a task scored and NOT acquired settles it as false whatever the other is; unknown only when
        # nothing scored says no and something is missing
        if any(r is not None and r.get("scored") and r["acquired"] is False for r in rows):
            first[recipe] = False
        elif unscored:
            first[recipe] = None
            why.append("missing: whether %s acquires both tasks in its first run (%s not scored)" % (recipe, " and ".join(unscored)))
        else:
            first[recipe] = all(r["acquired"] is True for r in rows)
    if any(v is True for v in first.values()):
        acquires_both = True
        why.append("%s acquired both tasks at S(%d) in its first run" % (" and ".join(r for r in PUBLISHED if first[r]), serving))
    elif all(v is False for v in first.values()):
        acquires_both = False
        why.append(NO_PUBLISHED_BOTH)
    else:
        acquires_both = None

    repeats, unsettled = [], []
    for recipe in PUBLISHED:
        for chain in STAGE2:
            rows = [by_row.get((recipe, chain, rep)) for rep in REPLICATES[recipe]]
            both = all(r is not None and r["defined"] for r in rows)
            repeats.append({"recipe": recipe, "chain": chain, "both_defined": both,
                            "both_lose_5_per_100": all(loses(r) for r in rows) if both else None,
                            "both_access_sensitive": all(r["access_sensitive"] for r in rows) if both else None})
            # Settled without both rows: one run's first task was not acquired, or one defined run does not lose.
            if not both and not any(r is not None and (r["first_task_acquired"] is False or (r["defined"] and not loses(r))) for r in rows):
                unsettled.append("%s %s" % (recipe, chain))
    defined = [row for row in retention if row["defined"]]
    losers = [row for row in defined if loses(row)]
    any_loss = bool(losers) if defined else None
    only_g32 = None if any_loss is None else bool(losers) and all(row["recipe"] == "g32" for row in losers)
    repeated = ["%s %s" % (e["recipe"], e["chain"]) for e in repeats if e["both_lose_5_per_100"]]
    if repeated:
        loss_repeats = True
        why.append("an acquired first task loses at least 5 per 100 in both runs of %s" % ", ".join(repeated))
    elif not unsettled:
        loss_repeats = False
        why.append(ONLY_G32 if only_g32 else NO_REPEATED_LOSS)
    else:
        loss_repeats = None
        why.append("missing: whether a loss repeats in %s (a retention row there is neither defined nor marked not acquired)"
                   % ", ".join(unsettled))
    if acquires_both is True and loss_repeats is True:
        on = True
    elif acquires_both is False or loss_repeats is False:
        on = False
    else:
        on = None
    return {"published_acquire_both_first_run": first, "loss_repeats": repeats,
            "any_defined_loss_5_per_100": any_loss, "losses_only_in_g32": only_g32,
            "on_to_package_4": on, "why": why}


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


def _series(rows: list, name: str) -> list:
    return [(row["step"], row["data"][name]) for row in rows if _finite(row["data"].get(name))]


def training_of(work: Work, key: str) -> dict:
    merged = work.merged_run(key)
    folder, attempt = (merged[1], merged[0]) if merged else latest(work.runs, key)
    if folder is None:
        return {"key": key, **_missing(str(work.root / "runs" / ("%s-a<N>" % key)))}
    out = {"key": key, "found": 1, "run": str(folder), "attempt": attempt, "merged_attempt": work.run_attempt(key)}
    summary = _json(folder / "run-summary.json")
    summary = summary if isinstance(summary, dict) else {}
    out["summary_found"] = int(bool(summary))
    for field in ("name", "arm", "steps", "returncode", "merged", "n_gpus", "seconds", "learning_rate", "dataset",
                  "model_dir", "mini_batch", "teacher_update_rate"):
        out[field] = summary.get(field)
    seconds, gpus = summary.get("seconds"), summary.get("n_gpus")
    out["gpu_hours"] = round(seconds * gpus / 3600, 3) if _finite(seconds) and _finite(gpus) else None
    rows, bad = read_metrics(folder / "metrics.jsonl")
    out["metrics_found"], out["metrics_unreadable_lines"] = int((folder / "metrics.jsonl").is_file()), bad
    out["metrics_steps"] = len(rows)
    length = _series(rows, "response_length/mean")
    at_step = dict(length)
    peak = max(length, key=lambda pair: pair[1]) if length else None
    clip = [v for _, v in _series(rows, "response_length/clip_ratio")]
    longest = [v for _, v in _series(rows, "response_length/max")]
    score = _series(rows, "critic/score/mean")
    out.update({"length_mean_first": length[0][1] if length else None, "length_mean_first_step": length[0][0] if length else None,
                "length_mean_step_%d" % LENGTH_STEP: at_step.get(LENGTH_STEP),
                "length_mean_last": length[-1][1] if length else None, "length_mean_last_step": length[-1][0] if length else None,
                "length_mean_max": peak[1] if peak else None, "length_mean_max_step": peak[0] if peak else None,
                "length_max_max": max(longest) if longest else None,
                "clip_ratio_max": max(clip) if clip else None,
                "score_first": score[0][1] if score else None, "score_last": score[-1][1] if score else None})
    out["validation"] = [{"step": row["step"], "key": name, "accuracy": value} for row in rows
                         for name, value in sorted(row["data"].items()) if name.endswith(VAL_SUFFIX) and _finite(value)]
    out["validation_label"] = VALIDATION_LABEL
    # The campaign writes the statistics under <work>/k8b/report-rollouts/<key>-a<N>/ (a row's output folder cannot sit
    # inside another row's run folder); a copy inside the run folder is read too, for a tree built by hand.
    stats_folder, _stats_attempt = latest(work.rollouts, key, "rollout-stats.json")
    stats = _json(stats_folder / "rollout-stats.json") if stats_folder is not None else None
    if not isinstance(stats, dict):
        stats = _json(folder / "report-rollouts" / "rollout-stats.json")
    steps = [s for s in (stats or {}).get("steps") or [] if isinstance(s, dict)] if isinstance(stats, dict) else []
    out["rollout_stats_found"] = int(bool(steps))

    def no_spread(step: dict):
        value = (step.get("groups") or {}).get("same_score_share")
        return value if _finite(value) else None

    out["no_spread_first"] = no_spread(steps[0]) if steps else None
    out["no_spread_last"] = no_spread(steps[-1]) if steps else None
    out["no_spread_first_step"] = steps[0].get("step") if steps else None
    out["no_spread_last_step"] = steps[-1].get("step") if steps else None
    return out


# ------------------------------------------------------------------------------------------- build
def build(args) -> dict:
    serving, diagnostic = int(args.serving), int(args.diagnostic)
    if serving >= diagnostic:
        raise PilotReportError("--serving (%d) must be below --diagnostic (%d)" % (serving, diagnostic))
    root = Path(args.work)
    if not root.is_dir():
        raise PilotReportError("no such work tree: %s" % root)
    work = Work(root, getattr(args, "allow_different_machines", False), serving)
    if (all(work.bed(BASE, task) is None and (BASE, task) not in work.left_out for task in TASKS)
            and work.panel(BASE) is None and (BASE, "panel") not in work.left_out):
        raise PilotReportError("%s holds no sweep of %s at all (looked for %s-chemistry-a<N>, %s-toolalpaca-a<N> and %s-panel-a<N> "
                               "under k8b/report-sweep): every table is read against the untrained model, so there is "
                               "nothing to report" % (root, BASE, BASE, BASE, BASE))
    threshold = args.acquired_per_100
    base = table_base(work, serving, diagnostic)
    acquisition = table_acquisition(work, serving, diagnostic, threshold)
    retention = table_retention(work, serving, diagnostic, acquisition)
    second = table_second_task(work, serving, diagnostic)
    by_budget = table_by_budget(work, retention, serving, diagnostic)
    panel = table_panel(work, serving, diagnostic)
    training = [training_of(work, key) for key in run_keys()]
    decision = decision_of(acquisition, retention, serving)
    machines = sorted({str(m) for m in [budget_report.machine_id(s) for s in work._bed.values() if s]
                       + [budget_report.machine_id(s) for s in work._panel.values() if s] if m is not None})
    for key, task in REQUIRED_PREFIX:                  # admit (or refuse) every required pair, so a check never read is still judged
        work.bed(key, task)
    checks = list(work.prefix_checks.values())
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "work": str(root.resolve()), "serving": serving, "diagnostic": diagnostic, "acquired_per_100": threshold,
            "layout": {"recipes": list(RECIPES), "replicates": {r: list(v) for r, v in REPLICATES.items()},
                       "chains": list(CHAINS), "base": BASE, "tasks": list(TASKS)},
            "runs_expected": len(run_keys()),
            "runs_found": sum(t["found"] for t in training),
            "stage1_scored": sum(r["scored"] for r in acquisition),
            "stage2_scored": sum(r["scored"] for r in retention),
            "retention_rows_defined": sum(r["defined"] for r in retention),
            "retention_rows_defined_diagnostic": sum(r["defined_diagnostic"] for r in retention),
            "accounting_exact": 0 if work.identity_failures else 1,
            "lineage_ok": 0 if work.lineage_problems else 1, "lineage_problems": work.lineage_problems,
            "prefix_checks": checks, "prefix_checks_found": len(checks),
            # failed: did not pass, or passed on another scoring than the one swept (`bound` false)
            "prefix_checks_failed": sum(1 for c in checks if not c["passed"] or c["bound"] is False),
            "prefix_checks_required": ["%s on %s" % pair for pair in REQUIRED_PREFIX],
            "prefix_checks_missing": sorted(set(work.prefix_missing)), "prefix_checks_missing_count": len(set(work.prefix_missing)),
            "different_machines_allowed": int(work.allow_different_machines),
            "machine_ids": machines, "problems": work.problems,
            "base": base, "acquisition": acquisition, "retention": retention, "second_task": second,
            "retention_by_budget": by_budget, "panel": panel, "training": training, "decision": decision,
            "validation_label": VALIDATION_LABEL}


# ------------------------------------------------------------------------------------------ render
def _v(value, pattern="%d") -> str:
    return "-" if not _finite(value) else pattern % value


def _pct(value) -> str:
    return "-" if not _finite(value) else "%.1f%%" % (100 * value)


def _yes(value) -> str:
    return "-" if value is None else ("yes" if value else "no")


def _four(scores) -> str:
    scores = scores or {}
    return " | ".join(_v(scores.get(k)) for k in ("strict_serving", "canonical_serving", "strict_diagnostic", "canonical_diagnostic"))


def _cohort_cell(row: dict) -> str:
    if row["defined_diagnostic"]:
        return "defined"
    if row["first_task_acquired_diagnostic"] is False:
        return "not acquired at E(H)"
    return "-"


def _render_cohort(lines: list, title: str, t: dict, B: int, H: int) -> None:
    budgets = t["budgets"]
    lines += ["### %s" % title, ""]
    if not budgets:
        lines += ["No retention row is defined in this cohort, so there is nothing to tabulate.", ""]
        return
    lines += ["The strict loss per row, in questions:", "",
              "| row | n | " + " | ".join(str(b) for b in budgets) + " |", "|---|---:|" + "---:|" * len(budgets)]
    lines += ["| %s | %s | %s |" % (e["key"], _v(e["n"]), " | ".join(_v(e["loss"][str(b)], "%+d") for b in budgets)) for e in t["rows"]]
    lines += ["", "Each recipe's own mean loss per 100 questions (its number of rows in brackets). A recipe's own mean is "
              "over its own rows and is not comparable across recipes.", "",
              "| recipe (rows) | " + " | ".join(str(b) for b in budgets) + " |", "|---|" + "---:|" * len(budgets)]
    for recipe, v in t["recipes"].items():
        lines.append("| %s (%d) | %s |" % (recipe, v["rows"], " | ".join(_v(v["mean_loss_per_100"][str(b)], "%+.2f") for b in budgets)))
    lines += ["", RECIPE_CAVEAT, "",
              "Each pair of recipes over the rows defined for both (chain-replicate), with each one's mean loss per 100 at "
              "%d and at %d, the recipe with the smaller mean at each (\"tie\" when equal), and whether that order is "
              "strictly reversed:" % (B, H), "",
              "| a | b | shared rows | a at %d | b at %d | a at %d | b at %d | smaller at %d | smaller at %d | reversed |" % (B, B, H, H, B, H),
              "|---|---|---|---:|---:|---:|---:|---|---|---|"]
    for p in t["pairs"]:
        if not p["n_shared"]:
            lines.append("| %s | %s | none | - | - | - | - | - | - | - |" % (p["a"], p["b"]))
            continue
        m = p["mean_loss_per_100"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            p["a"], p["b"], ", ".join(p["shared"]), _v(m[p["a"]][str(B)], "%+.2f"), _v(m[p["b"]][str(B)], "%+.2f"),
            _v(m[p["a"]][str(H)], "%+.2f"), _v(m[p["b"]][str(H)], "%+.2f"), p["order_serving"], p["order_diagnostic"],
            _yes(p["reversed"])))
    lines += ["", "Some pair of recipes ordered strictly the other way at %d than at %d: %s." % (
        H, B, {True: "yes", False: "no", None: "not defined (no pair of recipes shares a defined row)"}[t["ranking_differs"]]), ""]


def render(report: dict) -> str:
    B, H = report["serving"], report["diagnostic"]
    four = "S(%d) | E(%d) | S(%d) | E(%d)" % (B, B, H, H)
    lines = ["# Two-task pilot report", "",
             "Work tree `%s`. Serving budget B = %d new tokens, diagnostic budget H = %d. S is the strict count (the "
             "task's own scoring rule), E the canonical count (the gold-blind reading of kit/canonical.py), both read from "
             "one long generation per model by kit/cap_sweep.py. Found: %d of %d training runs; %d of the stage-1 and %d "
             "of the stage-2 scorings compared; %d retention rows defined at S(%d), %d at E(%d). Accounting identity exact "
             "for every comparison: %s. Every sweep scored the run of record of its model: %s."
             % (report["work"], B, H, report["runs_found"], report["runs_expected"], report["stage1_scored"],
                report["stage2_scored"], report["retention_rows_defined"], B, report["retention_rows_defined_diagnostic"], H,
                "yes" if report["accounting_exact"] else "NO", "yes" if report["lineage_ok"] else "NO"), "",
             RECIPE_CAVEAT, ""]
    if report["different_machines_allowed"]:
        lines += ["--allow-different-machines was given: sweeps from different machines are compared, and each comparison "
                  "records whether its two sweeps share a machine. Bed counts move by several points across machines before "
                  "any training.", ""]
    elif len(report["machine_ids"]) > 1:
        lines += ["The sweeps come from more than one machine (%s); bed counts move by several points across machines "
                  "before any training, so no two sweeps of different machines are compared: such a row keeps its own "
                  "scores and says \"not compared\"." % ", ".join(report["machine_ids"]), ""]
    if report["lineage_problems"]:
        lines += ["**Lineage: these sweeps or rows are not used: %s.**" % "; ".join(report["lineage_problems"]), ""]
    if report["problems"]:
        lines += ["Pieces left out: " + "; ".join(report["problems"]) + ".", ""]

    # prefix checks
    checks = report["prefix_checks"]
    lines += ["## Prefix checks", "",
              "Every budget of a sweep is read from a prefix of one long generation; kit/cap_sweep.py check compares those "
              "prefixes with a generation capped short. A check passes when it compared at least one answer, the text "
              "agrees on at least %d%% of them, and both scorings ran on the same machine with the same max_model_len. "
              "A check counts only for the scoring it was made on (the folder the sweep read, at the serving budget and "
              "the sweep's cap). The sweep of a model and task whose check failed, or is of another scoring, is left out "
              "of every table. The campaign schedules four checks (%s): a sweep of one of those with no check is left out "
              "too. Any other sweep with no check is used." % (round(100 * PREFIX_AGREEMENT), "; ".join(report["prefix_checks_required"])), ""]
    if report["prefix_checks_missing"]:
        lines += ["**Left out of every table because its scheduled prefix check is missing: %s.**" % ", ".join(report["prefix_checks_missing"]), ""]
    if not checks:
        lines += ["No prefix check was found under k8b/report-prefix.", ""]
    else:
        lines += ["| model | task | attempt | compared | text agreement | same machine | same max_model_len | of the swept scoring | verdict |",
                  "|---|---|---:|---:|---:|---|---|---|---|"]
        for c in checks:
            lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
                c["key"], c["task"], c["attempt"], _v(c["compared"]), _v(c["text_agreement"], "%.4f"),
                _yes(c["same_machine"]), _yes(c["same_max_model_len"]), _yes(c["bound"]),
                "FAIL" if not c["passed"] else ("pass" if c["bound"] is not False else "NOT OF THIS SCORING")))
        failed = [c for c in checks if not c["passed"] or c["bound"] is False]
        if failed:
            lines += ["", "**Left out of every table because prefix reuse failed: %s.**"
                      % ", ".join("the %s sweep of %s" % (c["task"], c["key"]) for c in failed)]
        lines.append("")

    # 1
    lines += ["## 1. Base", "",
              "The untrained model (%s) on each task: the four scores, the share of its answers cut at %d, and its tokens "
              "per strict-correct answer at %d." % (BASE, H, B), "",
              "| task | n | %s | cut at %d | tokens per correct at %d |" % (four, H, B), "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task, s in report["base"].items():
        if not s["found"]:
            lines.append("| %s | - | - | - | - | - | - | - |" % task)
        else:
            lines.append("| %s | %s | %s | %s | %s |" % (task, _v(s["n"]), _four(s["scores"]), _pct(s["cut_share_diagnostic"]),
                                                       _v(s["tokens_per_correct_serving"], "%.1f")))
    # 2
    lines += ["", "## 2. Acquisition (stage 1)", "",
              "Each model trained on one task from the base, scored on that task: the four scores, the gain over the base "
              "at S(%d) in questions and per 100 questions, and acquired = a gain of at least %s per 100 at S(%d), the "
              "pre-registered rule. Beside it, never in its place, the same threshold on the canonical score: the gain per "
              "100 at E(%d) and whether it passes, and the gain per 100 at E(%d) and whether it passes (acquired at E(%d), "
              "which defines the diagnostic cohort of tables 3 and 5). Then the share cut at %d and tokens per "
              "strict-correct answer at %d with its ratio to the base's. A row not acquired at S(%d) whose canonical gain "
              "passes says where in its note." % (B, report["acquired_per_100"], B, B, H, H, H, B, B), "",
              "| run | task | %s | gain | per 100 | acquired | E(%d) per 100 | passes | E(%d) per 100 | acquired at E(%d) "
              "| cut at %d | tokens per correct | ratio to base | note |" % (four, B, H, H, H),
              "|---|---|" + "---:|" * 14 + "---|"]
    for r in report["acquisition"]:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["key"], r["task"], _four(r.get("scores")), _v(r.get("gain_serving_strict"), "%+d"), _v(r.get("gain_per_100"), "%+.1f"),
            _yes(r["acquired"]), _v(r.get("gain_serving_canonical_per_100"), "%+.1f"), _yes(r.get("passes_canonical_serving")),
            _v(r.get("gain_diagnostic_per_100"), "%+.1f"), _yes(r.get("acquired_diagnostic")),
            _pct(r.get("cut_share_diagnostic")), _v(r.get("tokens_per_correct_serving"), "%.1f"),
            _v(r.get("tokens_per_correct_ratio_to_base"), "%.2f"), r["note"] or ""))
    lines += ["", TOOLALPACA_CAVEAT]
    # 3
    lines += ["", "## 3. Retention (stage 2)", "",
              "The first task, scored after the second was trained on top of it, against the same task scored just before "
              "the second training (the reference). F = S_ref(%d) - S_after(%d) is the exact sum of three signed accounting "
              "terms, from kit/budget_report.py: residual = E_ref(%d) - E_after(%d); budget = [E_after(%d) - E_after(%d)] - "
              "[E_ref(%d) - E_ref(%d)]; extraction = [E_after(%d) - S_after(%d)] - [E_ref(%d) - S_ref(%d)]. The terms locate "
              "the loss among the four scores; they do not say why it happened. Retained = S_after(%d) / S_ref(%d). A row "
              "whose first task was not acquired in table 2 is marked \"%s\". Access-sensitive (pre-registered, defined "
              "rows only): F at least %d per 100 questions and budget + extraction at least half of F. The column \"E(%d) "
              "cohort\" is the diagnostic cohort: a scored row whose first task was acquired at E(%d) is defined there; the "
              "two cohorts are never pooled."
              % (B, B, H, H, H, B, H, B, B, B, B, B, B, B, NOT_ACQUIRED, LOSS_PER_100, H, H), "",
              "| after | first task | reference | acquired | F | residual | budget | extraction | %s | retained "
              "| access-sensitive | E(%d) cohort | status |" % (four, H),
              "|---|---|---|---|" + "---:|" * 9 + "---|---|---|"]
    for r in report["retention"]:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["after"], r["task"], r["reference"], _yes(r["first_task_acquired"]), _v(r.get("F"), "%+d"), _v(r.get("residual"), "%+d"),
            _v(r.get("budget"), "%+d"), _v(r.get("extraction"), "%+d"), _four(r.get("scores")), _v(r["retained_share"], "%.3f"),
            _yes(r["access_sensitive"]), _cohort_cell(r), r["status"]))
    # 4
    lines += ["", "## 4. Second task", "",
              "The task trained second, scored after the chain, against the same task trained alone from the base with the "
              "same recipe and replicate: the four scores of each and the difference at S(%d) (chain minus alone)." % B, "",
              "| chain run | task | %s | alone run | %s | difference | per 100 | note |" % (four, four),
              "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in report["second_task"]:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["key"], r["task"], _four(r["chain_scores"].get("scores")), r["alone"], _four(r["alone_scores"].get("scores")),
            _v(r["difference_serving_strict"], "%+d"), _v(r["difference_per_100"], "%+.1f"), r["note"] or ""))
    # 5
    rb = report["retention_by_budget"]
    lines += ["", "## 5. Retention by budget", "",
              "The strict loss S_ref(N) - S_after(N) at every budget N read by all the sweeps of the rows of table 3 defined "
              "in a cohort, once for the serving cohort (first task acquired at S(%d)) and once for the diagnostic cohort "
              "(first task acquired at E(%d)), never pooled. Recipes are compared only pairwise, over the rows defined for "
              "both, because a recipe's rows can be a different mix of tasks and orders from another's. This is "
              "descriptive, from at most two replicates per recipe, and needs independent confirmation." % (B, H), ""]
    _render_cohort(lines, "Serving cohort: first task acquired at S(%d)" % B, rb["serving_cohort"], B, H)
    _render_cohort(lines, "Diagnostic cohort: first task acquired at E(%d)" % H, rb["diagnostic_cohort"], B, H)
    # 6
    panel = report["panel"]
    names = panel["panels"] + ["total"]
    lines += ["## 6. Panel", "",
              "The general panel, strict correct per panel at %d / at %d, with the difference from the base in brackets "
              "(at %d, at %d)." % (B, H, B, H), "",
              "| run | " + " | ".join(names) + " | note |", "|---|" + "---:|" * len(names) + "---|"]
    for r in panel["rows"]:
        if not r["found"]:
            lines.append("| %s | %s | %s |" % (r["key"], " | ".join("-" for _ in names), r.get("note") or ""))
            continue
        cells = []
        for name in names:
            c = r["panels"].get(name)
            if not c:
                cells.append("-")
                continue
            cell = "%s / %s" % (_v(c["serving"]), _v(c["diagnostic"]))
            if r["compared_with_base"]:
                cell += " (%s, %s)" % (_v(c["serving_minus_base"], "%+d"), _v(c["diagnostic_minus_base"], "%+d"))
            cells.append(cell)
        lines.append("| %s | %s | %s |" % (r["key"], " | ".join(cells), r.get("note") or ""))
    # 7
    lines += ["", "## 7. Training", "",
              "Per run, from the trainer's metrics.jsonl and run-summary.json of its merged attempt (else its highest "
              "attempt): the mean answer length in tokens at the first step, at step %d and at the last step, its maximum "
              "(at step), the maximum clip ratio, the mean training score at the first and last step, seconds, GPU-hours "
              "(seconds x GPUs / 3600), and from rollout-stats.json the share of prompt groups whose attempts all got the "
              "same score at the first and last step." % LENGTH_STEP, "",
              "| run | attempt | steps | exit | merged | length first | at %d | last | max (step) | clip max | score first | last "
              "| no spread first | last | seconds | GPU-hours |" % LENGTH_STEP,
              "|---|" + "---:|" * 15]
    for t in report["training"]:
        if not t["found"]:
            lines.append("| %s |" % t["key"] + " - |" * 15)
            continue
        peak = "-" if t["length_mean_max"] is None else "%.1f (%s)" % (t["length_mean_max"], t["length_mean_max_step"])
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            t["key"], _v(t["attempt"]), _v(t["metrics_steps"]), _v(t["returncode"]), _v(t["merged"]), _v(t["length_mean_first"], "%.1f"),
            _v(t["length_mean_step_%d" % LENGTH_STEP], "%.1f"), _v(t["length_mean_last"], "%.1f"), peak,
            _v(t["clip_ratio_max"], "%.3f"), _v(t["score_first"], "%.3f"), _v(t["score_last"], "%.3f"),
            _pct(t["no_spread_first"]), _pct(t["no_spread_last"]), _v(t["seconds"]), _v(t["gpu_hours"], "%.2f")))
    lines += ["", "The trainer's own validation accuracy (%s):" % VALIDATION_LABEL, "",
              "| run | step: accuracy |", "|---|---|"]
    for t in report["training"]:
        if t["found"]:
            lines.append("| %s | %s |" % (t["key"], ", ".join("%s: %.4f" % (v["step"], v["accuracy"]) for v in t["validation"]) or "-"))
    # 8
    d = report["decision"]
    first = d["published_acquire_both_first_run"]
    lines += ["", "## 8. Decisions", "",
              "These are the pre-registered rules (plan v3, section 4) applied mechanically to tables 2 and 3; they decide "
              "nothing by themselves.", "",
              "- A published recipe acquired both tasks at S(%d) in its first run: %s." % (
                  B, ", ".join("%s %s" % (recipe, _yes(value)) for recipe, value in first.items())),
              "- Some defined retention row loses at least %d per 100: %s. All such losses are in g32: %s." % (
                  LOSS_PER_100, _yes(d["any_defined_loss_5_per_100"]), _yes(d["losses_only_in_g32"])),
              "- On to package 4: %s." % {True: "yes", False: "no", None: "not decided (data missing)"}[d["on_to_package_4"]],
              "- Why: %s." % "; ".join(d["why"]), "",
              "| recipe | order | both runs defined | both lose %d per 100 | both access-sensitive |" % LOSS_PER_100,
              "|---|---|---|---|---|"]
    lines += ["| %s | %s | %s | %s | %s |" % (e["recipe"], e["chain"], _yes(e["both_defined"]), _yes(e["both_lose_5_per_100"]),
                                             _yes(e["both_access_sensitive"])) for e in d["loss_repeats"]]
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The tables of the two-task pilot, from one work tree.")
    parser.add_argument("--work", required=True, help="the pilot campaign's work tree (runs/, k8b/report-sweep/)")
    parser.add_argument("--serving", type=int, required=True, help="the serving budget B, e.g. 2048")
    parser.add_argument("--diagnostic", type=int, required=True, help="the diagnostic budget H, e.g. 8192")
    parser.add_argument("--acquired-per-100", type=float, default=DEFAULT_ACQUIRED_PER_100,
                        help="the stage-1 gain over the base at S(B), per 100 questions, that counts as acquired (default %d)"
                        % DEFAULT_ACQUIRED_PER_100)
    parser.add_argument("--allow-different-machines", action="store_true",
                        help="compare sweeps scored on different machines (recorded; refused by default)")
    parser.add_argument("--out", required=True, help="a new directory")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: choose a new --out" % out)
    try:
        report = build(args)
    except PilotReportError as exc:
        raise SystemExit(str(exc))
    out.mkdir(parents=True)
    (out / "pilot-report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "pilot-report.md").write_text(render(report), encoding="utf-8")
    print("runs found %d of %d; stage 1 scored %d, stage 2 scored %d; retention rows defined %d (at E(H) %d); accounting "
          "exact %s; lineage ok %s; prefix checks %d, failed %d"
          % (report["runs_found"], report["runs_expected"], report["stage1_scored"], report["stage2_scored"],
             report["retention_rows_defined"], report["retention_rows_defined_diagnostic"], bool(report["accounting_exact"]),
             bool(report["lineage_ok"]), report["prefix_checks_found"], report["prefix_checks_failed"]))
    print("wrote", out / "pilot-report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
