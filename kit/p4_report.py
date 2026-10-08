#!/usr/bin/env python3
"""The package-4 report: the registered rules of the prevention screen and the bridges, applied mechanically.

    python p4_report.py --work $WORK --selection $WORK/k8b4/selection/selection.json --out $WORK/k8b4/report-p4-a<N>
                        [--serving 2048] [--diagnostic 8192] [--blocks prevention,bridges]

Implements, and does not improve:
  docs/phase2/plan-v3-package4-preregistration-20261002.md   ("the registration": 3, 3.1, 3.2, 3.3, 4, 5, 6)
  docs/phase2/plan-v3-package4-supplement1-20261003.md       ("the supplement": S2 to S5; S3 the physical GPU)
  docs/phase2/plan-v3-package4-amendment2-20261004.md        ("the amendment": A1 to A4, finite-panel nominations)
Every bound comes from kit/p4_intervals.py (the one interval method); nothing here re-implements one. The labels that
decide are the amendment's NOMINATIONS, from exact integer counts; the interval-gated results of the registration and
the supplement are computed and printed beside each, under "interval-gated result (reported, does not decide the
nomination)".

THE TREE (under the pilot's WORK folder; area `k8b4`):
    WORK/runs/<key>-a<N>/run-summary.json, env/argv.txt          package-4 training runs (keys below)
    WORK/k8b4/eval/<key>-<bed>-a<N>/bed-score.json                scorings (machine block)
    WORK/k8b4/report-sweep/<key>-<bed>-a<N>/sweep.json + per_item.jsonl
    WORK/k8b4/report-sweep/<key>-panel-a<N>/sweep.json            general panel (reported, no bar)
    WORK/k8b4/report-prefix/prefix-<key>-<bed>-a<N>.json          prefix checks (kit/cap_sweep.py check)
  Prevention keys p4-<lineage>-ctl1/gate1 (seed 101) and ctl2/gate2 (seed 102), lineage r1 and r2. Bridge keys
  p4-br-<cfg>-s101/-s102 for the trained configurations; the reused configuration (`a` for g8, `s` for sema) is
  p4-r1-ctl1 (seed 101) and p4-r1-ctl2 (seed 102).

RECORD RULES. A scoring: its highest attempt holding sweep.json (as kit/pilot_report.py). A package-4 TRAINING run:
its FIRST attempt whose run-summary.json says merged 1 and returncode 0 (registration 6, retries), never the
best-scoring one; a later attempt never replaces it, and a scoring of a later attempt is of another checkpoint. A
pilot run (stage 1, pilot stage 2): the attempt the selection recorded, found in THIS tree as runs/<key>-a<attempt>
(never by the absolute folder the selection quotes) and read only when its run-summary.json has the recorded sha256.

VALIDATION (registration 5, supplement S3 and S6). A bed scoring QUALIFIES when: sweep.json is a bed sweep of that bed
whose `budgets` and `per_budget` hold the serving and diagnostic budgets; per_item.jsonl has exactly `n` rows, unique
ids, and 0/1 `strict` lists aligned with `budgets` whose sums equal `per_budget`'s strict counts at both budgets; its
id set equals the untrained model's sweep of that bed and its `items_sha256` is that sweep's;
`reproduces_scoring_mismatches` is 0 and `decode_equals_text` is n; its `model` is the checkpoint of record of its key
(kit/pilot_report.py `lineage`; the untrained model's is no run checkpoint); and the registered machine id can be
RECOMPUTED from the stored fields gpus, cuda_visible_devices, versions and deterministic (kit/eval_bed.py
machine_fingerprint: sha256 of json.dumps of those four keys with sort_keys=True, first 16 hex) of bed-score.json in
k8b4/eval/<scoring folder name>/ when present, else of the sweep's machine block; the two must agree. The supplied `id`
is recorded and never used. A block needs ONE recomputed id and one (cap, max_model_len) over every scoring entering
its contrasts. A required prefix check qualifies when present, passed (compared > 0, text agreement >= 0.99, same
machine and context length, kit/pilot_report.py `read_prefix_checks`), bound to its own long scoring (folder name,
long cap, short cap = serving) and its short scoring's recomputed machine id equals its long scoring's.
Required checks: base8b, p4-r1-ctl1 and p4-r1-gate1 on both beds (prevention); the same plus p4-br-c-s101 on both
beds (bridges). A failed or missing required check makes the block's registered labels "not evaluated"
(prevention) or "incomplete: no registered label" (bridges); every computable number is still reported.
A package-4 run is evaluated only when its run of record matches its registered settings: steps 40; seed (summary
`seed` when recorded, and the three seed overrides of argv.txt); dataset = task B's; launcher (run-summary schema);
learning rate and minibatch (summary and argv); teacher update rate (SDPO 0.05; none for GRPO); max response length
(gate runs: summary `max_response_length` 2048, and any argv override 2048; controls and bridges: no
`data.max_response_length=` override in argv and 8192 if recorded); finish gate (gate runs: summary `finish_gate` 1;
others 0 if recorded); merged 1, returncode 0; and `model_dir` the selected stage-1 checkpoint of its lineage (all
bridge runs: lineage r1's). An absent key that is required is "not recorded" and the run is not evaluated.

PREVENTION (registration 3.2 as amended by A1). Exact counts throughout, S a strict-correct count at the serving
budget, n_A and n_B the panel sizes, t_B = ceil(0.05 n_B):
  condition 1 (lineage): every fresh control has F_c = S_stage1(A) - S_c(A) with 100 F_c >= 5 n_A;
  condition 2 (control): S_c(B) - S_base(B) >= t_B and S_c(B) - S_stage1(B) >= t_B;
  A1.1: Q = 1/2 F_c - F_i >= 0. With F in questions over the same n_A, Q n_A = F_c / 2 - F_i, so Q >= 0 iff
        F_c - 2 F_i >= 0, i.e. the integer inequality 2 * F_i <= F_c;
  A1.2: S_i(B) - S_base(B) >= t_B and S_i(B) - S_stage1(B) >= t_B;
  A1.3: (S_i(B) - S_c(B)) / n_B > -0.025 = -1/40, i.e. the integer inequality 40 * (S_i(B) - S_c(B)) > -n_B.
A pair whose control fails condition 1 or 2 is "not evaluated" (A1). The screen is "replicated finite-panel
prevention nomination" only if all eight pairs meet A1.1 to A1.3.

BRIDGES (supplement S2 to S5 as amended by A2), within seed, both seeds for a label, t_A = ceil(0.05 n_A):
contrasts D_ab = S_a(A) - S_b(A), D_bc = S_b(A) - S_c(A), D_sc = S_s(A) - S_c(A), G = S_a(A) - S_c(A) nominated in a
direction when the count is >= t_A (or <= -t_A) in both seeds; acquisition per configuration; B-preservation
40 (S_x(B) - S_y(B)) > -n_B in both seeds; half-gaps 2 n_A (D_ab - G/2) = S_a - 2 S_b + S_c and 2 n_A (D_bc - G/2) =
-S_a + 2 S_b - S_c (A); attribution and the combined algorithm sentence exactly as A2. Registered labels need the
whole eight-run block, its scorings and its prefix checks; otherwise "incomplete: no registered label" and the
available estimates and bounds are printed descriptively.

ROUND 2 OF THE SEND-5 REVIEW, AND AMENDMENT 3 (docs/phase2/plan-v3-package4-amendment3-20261004.md), which governs:
  BUDGET (B3). The durable stop records WORK/k8b4/budget/stop-<block>.json and the ledger are read: a stopped block (or
    one whose budget status shows spending over its ceiling) is "budget-incomplete" -- prevention's screen and every
    pair; bridges' status, with no registered label; missing accounting evidence is "budget compliance unavailable".
    Available estimates are still printed. The report row runs after the budget-status row and reads it (--budget-status).
  RUN OF RECORD (B5). The first attempt that is either merged 1 / returncode 0 in its run-summary.json, or the attempt a
    wrapper record (WORK/k8b4/budget/attempts/<row>-a<N>.json, class `reconciled`) kept as a complete export. Without
    its run-summary.json that run is "not evaluated: settings not recorded"; it is never replaced by a later attempt.
  RECIPE (B6). Each run's archived env/argv.txt is compared with its pilot baseline's recorded command (named by the
    frozen recipe-check record WORK/k8b4/recipe-check/baseline.json) under the permitted substitutions of
    kit/p4_recipe.py, and its archived env/resolved-config.yaml with the baseline's reconstructed resolved
    configuration; a difference, or a missing file, makes the run "not evaluated". For c and s every difference of
    command and resolved configuration is REPORTED, and the shared settings of S2 are verified.
  ROLLOUTS (B8). The run of record's reduced rollout records (WORK/k8b4/report-rollouts/<run>/rollout-rows.jsonl.gz)
    must exist; a GATED run's must carry the termination flag and both rewards on every rollout, else "not evaluated"
    (protocol failure). Before/after rewards and no-reward groups are reported with no bar.
  SCORINGS (B8). Registered populations: Chemistry 210, ToolAlpaca 68, and the untrained model's items_sha256 must be the
    pilot's own base sweep's (WORK/k8b/report-sweep/base8b-<bed>-a<N>, highest attempt, in the archive) and its model
    the untrained model the pilot used. The scoring of record is the HIGHEST attempt folder; without its sweep.json it is
    not qualified (no older attempt replaces it). Raw evidence in k8b4/eval/<scoring>/: bed-score.json, responses.jsonl
    and the token file, with the sweep's responses_sha256 / tokens_sha256 equal to the files' (a scoring outside k8b4 is
    refused). One signature across a block: the whole engine, generation and decoding blocks, max_new_tokens,
    max_model_len, canonical_rule and the scorer (bed-score schema and mode, sweep schema). A machine id is recomputed
    only from fields that list at least one GPU. A prefix check: compared == the registered n, text agreement >= 0.99,
    bound to its own long scoring, and its short scoring of the SAME model.
  COMPLETENESS (R7, B8). A block's registered labels need every row of the block (its P4_BLOCK rows in --campaign) to
    have a terminal runner verdict, every row but the statistics rows to have PASSED (qualification, prefix, recipe and
    admission checks included), the reservation and the passed recipe-check record, every checkpoint's three long
    scorings (A, B, panel) qualified, and no budget stop; otherwise prevention is "technically incomplete" and bridges
    "incomplete: no registered label", with every available contrast printed descriptively.
  CONDITION 1 (B8) is judged per pair with that pair's own control; the screen still needs all eight pairs.
  AMENDMENT 3 B1 and B2 are stated in the report: E over the separable pilot pairs, and no pilot scoring reused.

ROUND 3 OF THE SEND-5 REVIEW (the manager's rulings F2 to F6), which governs where it differs from the text above:
  F2  the frozen decision inputs are re-verified FIRST (kit/p4_frozen.py: the pilot's finalization manifest and every
      input in it, the selection, the reservation, the recipe check's frozen record); any difference refuses the report
      ("inputs changed after they were frozen: <which>"), on the partner's node and on an extracted archive alike.
  F3/F4  a run's recipe is its ACTUAL env/argv.txt and env/resolved-config.yaml compared with its frozen pilot baseline
      under the exact substitution table of kit/p4_recipe.py `verify_run` (full key paths, exact old and new values,
      interpolation-derived keys enumerated; missing evidence is a problem).
  F5  every bed and panel sweep records cap 8192, max_model_len 12288 and the registered budgets, and its raw scoring
      the same limits; a panel carries both budget totals and its raw scoring's model is the checkpoint of record; every
      scoring records `scorer_sha256` (kit/eval_bed.py, kit/score_forgetting.py) and its sweep the same value and its own
      `sweep_source_sha256`, one signature across a block; a sweep is CURRENT only if it read the highest scoring attempt
      folder, the runner's latest attempts of the scoring row and the sweep row both passed, the sweep of record is the
      sweep row's latest attempt and that attempt read the scoring row's latest one (its `inputs`); a prefix check's
      short (direct 2,048) scoring is bound like a long one (raw files' sha256 as the check recorded them, its model the
      checkpoint of record, its limits, its scorer, and the prefix, direct and long rows' latest attempts passed and read
      by the prefix row); the two qualification runs are validated from their artifacts (2 steps, seed 101, the
      intervention settings, their task, lineage r1's stage-1 checkpoint, their actual command, their scorings: n == 8,
      limit 8, their own checkpoint, registered limits, a recorded scorer, the latest scoring row passed).
  F6  whole-block labels need a reduced rollout record for EVERY launched training attempt of the block (qualification
      included, failed attempts included: the runner's and the wrapper's records say which launched), its row count
      equal to the rollouts its statistics counted, every gated attempt's rows carrying the cut flag and both rewards;
      a failed statistics row is a terminal failure like any other.

ROUND 4 OF THE SEND-5 REVIEW (the manager's rulings G1, G3, G4, G5), which governs where it differs from the text above:
  G1  a stop for workers not verified terminated, and any attempt whose accounting is unavailable, label the block
      "budget compliance unavailable" (missing accounting evidence, not a demonstrated overrun); no registered label.
  G3  besides the selection, the reservation and the recipe check's frozen document, the recipe check's BASELINE is
      verified through kit/p4_frozen.py (`verify_baseline`: its seal and its link to the frozen document); a corrupted
      or missing baseline leaves no run evaluated: both blocks are reported descriptively (`recipe_baseline` says why).
      Every reconstructed pilot configuration and pilot command that kit/p4_recipe.py `verify_run` reads back is
      first verified against the hash the frozen record stores; a mismatch is a refusal (the run is not evaluated).
  G4  each qualification scoring needs its raw evidence, bound like any other scoring: responses.jsonl and its token
      file holding exactly the first eight registered items of its task in file order (the untrained model's
      scoring of record gives the order), each file with the sha256 the scoring attempt recorded in bed-score.json,
      items_sha256 of those eight, and the block's recomputed machine id, engine/generation/decoding/limit record and
      scorer.
  H2  (round 5) an attempt record holding spending-limit evidence (kit/p4_budget.py `spending_limit_evidence`:
      `ended_by: limit`, its watchdog found the job alive at limit_until, or it ended at or after limit_until) makes its
      block "budget-incomplete" even when its `class` says otherwise and no stop file exists.
  J3  (round 6, replacing round 5's H3) a failed attempt with zero reduced rows counts as "no rollout before the
      failure" only with POSITIVE, durable, attempt-bound evidence: the wrapper never started its launcher, or the
      launcher's own env/stage.json (bound to the attempt by its run name and job marker) ends before `trainer-invoked`
      with the `exited` entry of the launcher's EXIT trap. In every other case -- no stage record, no `exited` entry,
      the trainer invoked with or without metrics, metrics empty or without a step -- its completeness is "rollout
      evidence unavailable" and the block gets no registered label.
  G5  each launched training attempt's reduced records are reconciled with what the trainer did (kit/rollout_stats.py
      `reconcile`): for a completed run (a complete export, or run-summary returncode 0) every step 1..steps of its
      metrics.jsonl needs exactly data.train_batch_size x rollout n rows (from its recorded command); missing or
      partial dumps are "rollout evidence incomplete" and withhold the block's labels; an attempt that failed may have
      no rows only when its metrics record no training step.

Exit 0 when a report was written (whatever the labels), 2 for a refusal (an existing --out, an unreadable or
unentered or altered selection, unregistered budgets, unknown blocks). Standard library only; kit/p4_intervals.py,
kit/pilot_report.py and kit/budget_report.py are loaded by file path, so `kit/` works when exported alone.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import re
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "kit-p4-report.v1"
MANIFEST_SCHEMA = "kit-p4-manifest.v1"
SELECTION_SCHEMA = "kit-p4-selection.v1"
AREA = "k8b4"
BASE = "base8b"
BEDS = ("chemistry", "toolalpaca")
SERVING, DIAGNOSTIC = 2048, 8192
ALPHA = 0.05
MARGIN = Fraction(1, 40)                       # 0.025 in accuracy (registration 3.2 condition 5)
ACQUIRED_PER_100 = 5
LOSS_PER_100 = 5
STEPS = 40
AT_COST = Fraction(3, 2)
MACHINE_FIELDS = ("gpus", "cuda_visible_devices", "versions", "deterministic")
DATASETS = {"chemistry": "datasets/sciknoweval/chemistry", "toolalpaca": "datasets/tooluse"}
GRPO, SDPO = "kit-grpo-toolalpaca-run.v1", "kit-sdpo-run.v1"
CONFIGS = {"a": {"schema": GRPO, "lr": "1e-6", "mini_batch": 8, "teacher": None},
           "b": {"schema": GRPO, "lr": "1e-6", "mini_batch": 32, "teacher": None},
           "c": {"schema": GRPO, "lr": "1e-5", "mini_batch": 32, "teacher": None},
           "s": {"schema": SDPO, "lr": "1e-5", "mini_batch": 32, "teacher": "0.05"}}
RECIPE_CONFIG = {"g8": "a", "sema": "s"}       # the selected recipe's configuration, reused from the prevention controls
SEEDS = (101, 102)
LINEAGES = ("r1", "r2")
GATE_RESPONSE, LONG_RESPONSE = 2048, 8192
PREVENTION_PREFIX = (BASE, "p4-r1-ctl1", "p4-r1-gate1")
BRIDGE_PREFIX = PREVENTION_PREFIX + ("p4-br-c-s101",)
SEED_KEYS = ("data.seed", "actor_rollout_ref.actor.data_loader_seed", "actor_rollout_ref.actor.fsdp_config.seed")
LR_KEY, MB_KEY = "actor_rollout_ref.actor.optim.lr", "actor_rollout_ref.actor.ppo_mini_batch_size"
TEACHER_KEY = "actor_rollout_ref.actor.self_distillation.teacher_update_rate"
RESPONSE_KEY = "data.max_response_length"
MODEL_PATH_KEY = "actor_rollout_ref.model.path"
#: amendment 3 B8: the registered question sets, and the general panel's size (kit/panels/general-v1.jsonl)
REGISTERED_N = {"chemistry": 210, "toolalpaca": 68}
PANEL_N = 300
PREFIX_AGREEMENT = 0.99
#: round 3 (F5): the registered limits every scoring and sweep of package 4 records, and the qualification's own
REGISTERED_CAP, REGISTERED_CONTEXT = 8192, 12288
REGISTERED_BUDGETS = [256, 512, 1024, 2048, 4096, 8192]
QUALIFICATION = {"p4-qual-chem": "chemistry", "p4-qual-tool": "toolalpaca"}
QUAL_STEPS, QUAL_SEED, QUAL_N = 2, 101, 8
BUDGET_INCOMPLETE = "budget-incomplete"
COMPLIANCE_UNAVAILABLE = "budget compliance unavailable"
PROCESS_GROUP_ONLY = "process-group-only execution"
SETTINGS_NOT_RECORDED = "settings not recorded"
AMENDMENT_3 = {"source": "docs/phase2/plan-v3-package4-amendment3-20261004.md (amendment 3)",
               "E": "amendment 3 B1: the reservation's E averages the separable pilot (model, evaluation) pairs that have their "
                    "own runner row (73 in the pilot campaign); the untrained model's two task scorings, bundled inside the "
                    "prefix-pilot rows, are excluded",
               "rescoring": "amendment 3 B2: no pilot scoring is reused; every checkpoint entering a contrast was scored again "
                            "in the package-4 campaign"}

# --- fixed wording (registration 3.3, supplement S4, amendment A1 to A3)
FIXED_STATEMENT = "These are the registered rules applied mechanically; they decide nothing by themselves."
A3_CAVEAT = ("These nominations assert reproducible arithmetic on these panels and fitted checkpoints. They do not "
             "establish population prevention, population non-inferiority, a statistically established attribution, "
             "or reliability over training seeds.")
INTERVAL_HEADING = "interval-gated result (reported, does not decide the nomination)"
NOT_EVALUATED = "not evaluated"
TECHNICALLY_INCOMPLETE = "technically incomplete"
NOT_RUN = "not run"
SCREEN_YES = "replicated finite-panel prevention nomination"
SCREEN_NO = "nomination criteria not met"
PAIR_MEETS = "meets nomination criteria"
PAIR_FAILS = "does not meet nomination criteria"
LOSS_NOT_REPRODUCED = "target loss not reproduced under the registered rule"
CONTROL_NOT_LEARNED = "the control did not learn B during stage 2 (condition 2)"
ACQUISITION_NOT_DEMONSTRATED = "acquisition not demonstrated"
BRIDGE_INCOMPLETE = "incomplete: no registered label"
NO_CONSISTENT = "no consistent five-point difference on the panel"
NO_ATTRIBUTION = "no attribution"
AT_LEAST_HALF, LESS_THAN_HALF = "observed to carry at least half", "observed to carry less than half"
BOTH_ZERO = 'Both half-gap contrasts equal zero in both seeds: both steps receive "at least half".'
ATTRIBUTION_NOTE = ("These labels describe the observed path decomposition. They neither identify a unique responsible "
                    "setting nor supply independent causal evidence.")
COMBINED = ("At equal learning rate and minibatch, SDPO as configured in the pilot was observed to lose less of A than "
            "GRPO from this checkpoint on this panel in both seeds, with its B score within 2.5 points of GRPO's or above it.")
STEP_NAMES = {"minibatch": "minibatch step (D_ab = S_a(A) - S_b(A))", "learning_rate": "learning-rate step (D_bc = S_b(A) - S_c(A))",
              "algorithm": "algorithm step (D_sc = S_s(A) - S_c(A))", "reference_gap": "reference gap of the settings path (G = S_a(A) - S_c(A))"}
NOMINATION_WORDS = {"minibatch": ("observed to raise the loss", "observed to lower the loss"),
                    "learning_rate": ("observed to raise the loss", "observed to lower the loss"),
                    "algorithm": ("SDPO as configured observed to lose less", "SDPO as configured observed to lose more"),
                    "reference_gap": ("reference gap observed", "reference gap observed reversed")}
# the supplement's S4 names, interval-gated (reported beside the nominations, deciding nothing)
INTERVAL_WORDS = {"minibatch": ("raises the loss", "lowers the loss"), "learning_rate": ("raises the loss", "lowers the loss"),
                  "algorithm": ("SDPO as configured loses less", "SDPO as configured loses more"),
                  "reference_gap": ("reference gap established", "reference gap reversed")}
INTERVAL_NO_DIFFERENCE = "no difference shown"
INTERVAL_COMBINED = ("At equal learning rate and minibatch, SDPO as configured in the pilot lost less of A than GRPO from "
                     "this checkpoint, while establishing B noninferiority within the registered 2.5-percentage-point margin")
RULE_WORDS = {"lower_at_least_zero": "pass if the lower bound >= 0; fail if the upper bound < 0; otherwise inconclusive",
              "lower_above_zero": "pass if the lower bound > 0; fail if the upper bound < 0; otherwise inconclusive",
              "non_inferiority": "pass if the lower bound > -0.025; fail if the upper bound < -0.025; otherwise inconclusive"}


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_p4_report_%s" % name, HERE / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


iv = _load("p4_intervals")
pilot_report = _load("pilot_report")
budget_report = pilot_report.budget_report
recipe = _load("p4_recipe")
runner = _load("runner")                       # campaign files (--campaign); loaded here, beside this file, like the rest
budget_tool = _load("p4_budget")               # the runner's and the wrapper's attempt records (which attempts launched)
frozen_inputs = _load("p4_frozen")
rollouts = _load("rollout_stats")              # round-4 ruling G5: the reduced records reconciled with the trainer's metrics


class ReportRefused(ValueError):
    """The inputs cannot support a report; nothing is written."""


# ----------------------------------------------------------------------------------------------------- small helpers
def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is(value, target: int) -> bool:
    return _int(value) and value == target


def _same_int(value, target: int) -> bool:
    """An int, or a string of digits, equal to `target` (argv values are strings)."""
    if _int(value):
        return value == target
    return isinstance(value, str) and value.strip().isdigit() and int(value.strip()) == target


def _same_fraction(value, target: str) -> bool:
    try:
        return value is not None and not isinstance(value, bool) and Fraction(str(value).strip()) == Fraction(target)
    except (ValueError, ZeroDivisionError):
        return False


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha256_file(path: Path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def ceil_share(n: int) -> int:
    """ceil(0.05 n) in integers."""
    return budget_report.ceil_frac(n, ACQUIRED_PER_100, 100)


def recompute_machine_id(machine):
    """(id, None) from the four stored fields by kit/eval_bed.py's formula, or (None, why). The supplied id is not read.
    A block that lists no GPU names no physical GPU (scripts/read_send4.py's rule): no id."""
    if not isinstance(machine, dict):
        return None, "no machine block"
    absent = [k for k in MACHINE_FIELDS if k not in machine]
    if absent:
        return None, "the machine block lacks %s" % ", ".join(absent)
    if not isinstance(machine.get("gpus"), list) or not machine["gpus"]:
        return None, "the machine block lists no GPU, so no physical GPU is identified"
    blob = json.dumps({k: machine[k] for k in MACHINE_FIELDS}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16], None


def read_argv(path: Path):
    """{key: value} of the `key=value` lines of argv.txt (the first '=' splits), or None when the file is absent."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    out = {}
    for line in text.split("\n"):
        line = line.strip()
        if "=" in line and not line.startswith("-"):
            key, _sep, value = line.partition("=")
            out.setdefault(key, []).append(value)
    return out


def _fr(value) -> str:
    return str(Fraction(value))


def _tri(value) -> str:
    return "unavailable" if value is None else ("true" if value else "false")


def _decimal(value: Fraction, places=4) -> str:
    return ("%%.%df" % places) % float(value)


# ------------------------------------------------------------------------------------------------------------ the tree
class Tree:
    """One package-4 tree: every run, scoring and prefix check read once and judged once, every problem written down."""

    def __init__(self, work: Path, selection: dict, serving: int, diagnostic: int, campaign: dict | None = None,
                 baseline_problems: list | None = None):
        self.work, self.area = work, work / AREA
        self.selection, self.serving, self.diagnostic = selection, serving, diagnostic
        self.campaign = campaign
        self.recipe, self.chain = selection["group"]["recipe"], selection["group"]["chain"]
        self.A, self.B = selection["tasks"]["A"], selection["tasks"]["B"]
        self.sweeps = pilot_report.attempts(self.area / "report-sweep")
        self.pilot_sweeps = pilot_report.attempts(work / "k8b" / "report-sweep")
        self.runs_index = pilot_report.attempts(work / "runs")
        self.prefix_index = pilot_report.read_prefix_checks(self.area / "report-prefix")
        # round-4 ruling G3: the recipe check's baseline is used ONLY when kit/p4_frozen.py verified its seal and its link
        # to the frozen document; otherwise no run is evaluated and both blocks are reported descriptively
        self.baseline_problems = list(baseline_problems or [])
        self.recipe_record = None if self.baseline_problems else _json(self.area / "recipe-check" / "baseline.json")
        self.pilot = {}                       # pilot key -> the selection's provenance record
        self.lineage_of = {}                  # key -> lineage
        for lin in LINEAGES:
            for part in ("stage1", "pilot_stage2"):
                record = selection["lineages"][lin][part]
                self.pilot[record["key"]] = record
                self.lineage_of[record["key"]] = lin
        self.stage1 = {lin: selection["lineages"][lin]["stage1"]["key"] for lin in LINEAGES}
        self.pilot_stage2 = {lin: selection["lineages"][lin]["pilot_stage2"]["key"] for lin in LINEAGES}
        self.reused = RECIPE_CONFIG[self.recipe]
        self.roles = {}                       # package-4 key -> its registered settings
        for lin in LINEAGES:
            for index, seed in ((1, 101), (2, 102)):
                for kind in ("ctl", "gate"):
                    key = "p4-%s-%s%d" % (lin, kind, index)
                    self.roles[key] = {"key": key, "kind": "control" if kind == "ctl" else "gate", "seed": seed,
                                       "config": self.reused, "lineage": lin}
                    self.lineage_of[key] = lin
        self.bridge_keys = {}
        for cfg in ("a", "b", "c", "s"):
            for seed in SEEDS:
                if cfg == self.reused:
                    key = "p4-r1-ctl%d" % (1 if seed == 101 else 2)
                else:
                    key = "p4-br-%s-s%d" % (cfg, seed)
                    self.roles[key] = {"key": key, "kind": "bridge", "seed": seed, "config": cfg, "lineage": "r1"}
                    self.lineage_of[key] = "r1"
                self.bridge_keys[(cfg, seed)] = key
        self._run, self._bed, self._panel, self._prefix = {}, {}, {}, {}
        self.contrasts = []

    # ------------------------------------------------------------------------------------------------ training runs
    def kept_attempts(self, key: str) -> dict:
        """{attempt: wrapper record} of the attempts kit/p4_run.py kept as complete exports (class `reconciled`)."""
        kept = {}
        for path in sorted((self.area / "budget" / "attempts").glob("%s-a*.json" % key)):
            tail = path.stem[len(key) + 2:]
            record = _json(path)
            if tail.isdigit() and isinstance(record, dict) and record.get("class") == "reconciled" and _int(record.get("kept_attempt")):
                kept[record["kept_attempt"]] = {"record": path.name, **{k: record.get(k) for k in ("reason", "settings_recorded")}}
        return kept

    def raw_run(self, key: str) -> dict:
        """Every attempt of a package-4 run and its run of record: the FIRST attempt with merged 1 and returncode 0, or
        the first a reconcile record kept as a complete export (amendment 3 B5)."""
        if key in self._run:
            return self._run[key]
        listed, record = [], None
        kept = self.kept_attempts(key)
        for attempt, folder in sorted(self.runs_index.get(key, []), key=lambda pair: pair[0]):
            summary = _json(folder / "run-summary.json")
            entry = {"attempt": attempt, "run": str(folder.resolve()), "run_summary_sha256": sha256_file(folder / "run-summary.json")}
            if attempt in kept:
                entry["kept_by_reconcile"] = kept[attempt]
            if not isinstance(summary, dict):
                valid = attempt in kept
                entry.update({"valid_export": valid, "why": "run-summary.json missing or unreadable" + (
                    "; a reconcile record kept its complete export" if valid else "")})
                summary = None
            else:
                valid = (_is(summary.get("merged"), 1) and _is(summary.get("returncode"), 0)) or attempt in kept
                entry.update({"valid_export": valid, "why": None if valid else "merged %r, returncode %r"
                              % (summary.get("merged"), summary.get("returncode"))})
            if valid and record is None:
                record = {"attempt": attempt, "folder": folder, "summary": summary,
                          "argv": read_argv(folder / "env" / "argv.txt")}
            entry["of_record"] = bool(record is not None and record["attempt"] == attempt)
            listed.append(entry)
        out = {"key": key, "attempts": listed, "record": record,
               "why": None if record else "no attempt of %s under %s has a valid complete export (run-summary.json with merged 1 "
                                          "and returncode 0, or a reconcile record)" % (key, self.work / "runs")}
        role = self.roles.get(key)
        if role is not None:
            out["role"] = role
            out["problems"] = self.settings_problems(out, role)
            out["valid"] = record is not None and not out["problems"]
        self._run[key] = out
        return out

    def stage1_checkpoint(self, lineage: str):
        """(key, attempt) of the selected stage-1 checkpoint of a lineage, or None when the selection records none."""
        record = self.selection["lineages"][lineage]["stage1"]
        if record.get("status") != "present" or not _int(record.get("attempt")):
            return None
        return record["key"], record["attempt"]

    def recipe_problems(self, key: str, role: dict, record: dict) -> list:
        """Amendment 3 B6, repeated from the archive (rounds 2 and 3, F3/F4): the run's ACTUAL command and resolved
        configuration against its frozen pilot baseline under the exact substitution table (kit/p4_recipe.py
        `verify_run`); missing evidence is a problem too (fail closed)."""
        name = "%s-a%d" % (key, record["attempt"])
        doc = self.recipe_record
        if not isinstance(doc, dict) or doc.get("ok") is not True:
            return ["%s: no passed recipe-check record verified (k8b4/recipe-check/baseline.json%s): the recipe cannot be verified"
                    % (name, ": " + "; ".join(self.baseline_problems) if self.baseline_problems else "")]
        verdict, problems = recipe.verify_run(self.work, doc, key, record["folder"], self.area / "recipe-check")
        return [] if verdict is True else problems or ["%s: the recipe cannot be verified" % name]

    def launched_attempts(self, key: str) -> list:
        """[(attempt, run name)] of every attempt of a training row that LAUNCHED its launcher (the wrapper record says
        so; without one, its run folder exists), from the runner's own records (round-3 ruling F6)."""
        if self.campaign is None:
            return [(a, f.name) for a, f in sorted(self.runs_index.get(key, []), key=lambda pair: pair[0])]
        out = []
        for item in budget_tool.attempt_history(self.work, self.campaign["name"], key):
            if item["launched"]:
                out.append((item["attempt"], item.get("name") or "%s-a%d" % (key, item["attempt"])))
        known = {name for _a, name in out}
        out += [(a, f.name) for a, f in sorted(self.runs_index.get(key, []), key=lambda pair: pair[0]) if f.name not in known]
        return sorted(set(out))

    def every_attempt_reduced(self, key: str, gated: bool) -> list:
        """Round-3 ruling F6: a reduced record for EVERY launched attempt of the run (failed ones included), its row
        count equal to the dumped rollouts its statistics counted, and -- for a gated run -- every row carrying the cut
        flag and both rewards. Any gap is a protocol failure: the block gets no registered label."""
        problems = []
        attempts = self.launched_attempts(key)
        if not attempts:
            return ["%s: no launched attempt is recorded" % key]
        for _attempt, name in attempts:
            folder = self.area / "report-rollouts" / name
            stats = _json(folder / "rollout-stats.json")
            rows_path = folder / "rollout-rows.jsonl.gz"
            if not isinstance(stats, dict) or not rows_path.is_file():
                problems.append("%s: no reduced rollout records (k8b4/report-rollouts/%s/rollout-rows.jsonl.gz and rollout-stats.json): "
                                "protocol failure (amendment 3 B8, round-3 F6)" % (name, name))
                continue
            try:
                rows = [json.loads(line) for line in gzip.decompress(rows_path.read_bytes()).decode("utf-8").split("\n") if line.strip()]
            except (OSError, ValueError, EOFError) as exc:
                problems.append("%s: the reduced rollout records cannot be read: %s" % (name, exc))
                continue
            counted = (stats.get("totals") or {}).get("rollouts")
            if stats.get("rows") != len(rows) or (counted is not None and counted != len(rows)) or (counted is None and stats.get("steps_read")):
                problems.append("%s: %d reduced rows, the statistics counted %r dumped rollouts (rows %r): the reduction is incomplete"
                                % (name, len(rows), counted, stats.get("rows")))
            if stats.get("rows_sha256") and sha256_file(rows_path) != stats.get("rows_sha256"):
                problems.append("%s: rollout-rows.jsonl.gz is not the file its statistics wrote (sha256)" % name)
            # round-4 ruling G5: what the reduction holds against what the trainer did (its metrics.jsonl, the recorded
            # prompts x attempts); a completed run's missing or partial dumps withhold the block's labels
            wrapper = _json(self.area / "budget" / "attempts" / ("%s.json" % name)) or {}
            evidence = rollouts.reconcile(self.work / "runs" / name, [r.get("step") for r in rows],
                                          export_complete=wrapper.get("export_complete") == 1 and wrapper.get("class") != "reconciled",
                                          launcher_started=bool(wrapper.get("launched")) if wrapper else None,
                                          job_marker=wrapper.get("job_marker") or None)
            if evidence["status"] not in rollouts.EVIDENCE_OK:
                # G5: missing or partial dumps; J3: a failed attempt with no rows and no positive, attempt-bound evidence of
                # "no rollout" (the launcher's own env/stage.json ending before `trainer-invoked`, or a launcher never started)
                problems.append("%s: %s: %s (%s; amendment 3 B8, round-4 ruling G5, round-6 ruling J3)" % (
                    name, evidence["status"], "; ".join(evidence["problems"]), evidence["completed_why"]))
            if gated:
                missing = sum(1 for r in rows if r.get("truncated") is None or r.get("score_before_gate") is None or r.get("score") is None)
                if missing:
                    problems.append("%s: protocol failure: %d of %d gated rollouts lack the termination flag or a reward before or after the "
                                    "gate (amendment 3 B8)" % (name, missing, len(rows)))
        return problems

    def rollout_problems(self, key: str, role: dict, record: dict) -> list:
        """Amendment 3 B8 and round-3 F6: every launched attempt of the run reduced, a gated run's records complete."""
        return self.every_attempt_reduced(key, role["kind"] == "gate")

    def settings_problems(self, run: dict, role: dict) -> list:
        """Every way the run of record differs from its registered settings (registration 3 and 6, supplement S2)."""
        record = run["record"]
        if record is None:
            return [run["why"]]
        summary, argv, name = record["summary"], record["argv"], "%s-a%d" % (run["key"], record["attempt"])
        if summary is None:
            return ["%s: %s: its complete export was kept by reconcile, but run-summary.json is missing, so the run is not "
                    "evaluated (amendment 3 B5)" % (name, SETTINGS_NOT_RECORDED)]
        config = CONFIGS[role["config"]]
        problems = []
        if argv is None:
            problems.append("%s: env/argv.txt missing: the command cannot be verified" % name)
            argv = {}
        if not _is(summary.get("steps"), STEPS):
            problems.append("%s: steps %r, registered %d" % (name, summary.get("steps"), STEPS))
        seed = role["seed"]
        if "seed" in summary and not _same_int(summary.get("seed"), seed):
            problems.append("%s: run-summary seed %r, registered %d" % (name, summary.get("seed"), seed))
        for key in SEED_KEYS:
            values = argv.get(key)
            if values is None or len(values) != 1 or not _same_int(values[0], seed):
                problems.append("%s: argv %s=%s, registered %d" % (name, key, ",".join(values) if values else "(absent)", seed))
        if summary.get("dataset") != DATASETS[self.B]:
            problems.append("%s: dataset %r, registered %r (task B, %s)" % (name, summary.get("dataset"), DATASETS[self.B], self.B))
        if summary.get("schema") != config["schema"]:
            problems.append("%s: launcher schema %r, registered %r" % (name, summary.get("schema"), config["schema"]))
        if not _same_fraction(summary.get("learning_rate"), config["lr"]):
            problems.append("%s: learning rate %r, registered %s" % (name, summary.get("learning_rate"), config["lr"]))
        lr = argv.get(LR_KEY)
        if lr is None or len(lr) != 1 or not _same_fraction(lr[0], config["lr"]):
            problems.append("%s: argv %s=%s, registered %s" % (name, LR_KEY, ",".join(lr) if lr else "(absent)", config["lr"]))
        mb = argv.get(MB_KEY)
        if mb is None or len(mb) != 1 or not _same_int(mb[0], config["mini_batch"]):
            problems.append("%s: argv %s=%s, registered %d" % (name, MB_KEY, ",".join(mb) if mb else "(absent)", config["mini_batch"]))
        if "mini_batch" in summary and not _same_int(summary.get("mini_batch"), config["mini_batch"]):
            problems.append("%s: run-summary mini_batch %r, registered %d" % (name, summary.get("mini_batch"), config["mini_batch"]))
        teacher = argv.get(TEACHER_KEY)
        if config["teacher"] is None:
            if summary.get("teacher_update_rate") is not None or teacher is not None:
                problems.append("%s: a teacher update rate is recorded (%r, argv %r); the registered configuration has none"
                                % (name, summary.get("teacher_update_rate"), teacher))
        else:
            if not _same_fraction(summary.get("teacher_update_rate"), config["teacher"]):
                problems.append("%s: teacher update rate %r, registered %s" % (name, summary.get("teacher_update_rate"), config["teacher"]))
            if teacher is None or len(teacher) != 1 or not _same_fraction(teacher[0], config["teacher"]):
                problems.append("%s: argv %s=%s, registered %s" % (name, TEACHER_KEY, ",".join(teacher) if teacher else "(absent)",
                                                                  config["teacher"]))
        response = argv.get(RESPONSE_KEY)
        if role["kind"] == "gate":
            if "max_response_length" not in summary:
                problems.append("%s: max_response_length not recorded in run-summary.json (registered %d)" % (name, GATE_RESPONSE))
            elif not _same_int(summary.get("max_response_length"), GATE_RESPONSE):
                problems.append("%s: max_response_length %r, registered %d" % (name, summary.get("max_response_length"), GATE_RESPONSE))
            if response is not None and (len(response) != 1 or not _same_int(response[0], GATE_RESPONSE)):
                problems.append("%s: argv %s=%s, registered %d" % (name, RESPONSE_KEY, ",".join(response), GATE_RESPONSE))
            if "finish_gate" not in summary:
                problems.append("%s: finish_gate not recorded in run-summary.json (registered 1)" % name)
            elif not _same_int(summary.get("finish_gate"), 1):
                problems.append("%s: finish_gate %r, registered 1" % (name, summary.get("finish_gate")))
        else:
            if response is not None:
                problems.append("%s: argv overrides %s=%s; the registered command has no override (8192)"
                                % (name, RESPONSE_KEY, ",".join(response)))
            if "max_response_length" in summary and not _same_int(summary.get("max_response_length"), LONG_RESPONSE):
                problems.append("%s: max_response_length %r, registered %d" % (name, summary.get("max_response_length"), LONG_RESPONSE))
            if "finish_gate" in summary and not _same_int(summary.get("finish_gate"), 0):
                problems.append("%s: finish_gate %r, registered 0" % (name, summary.get("finish_gate")))
        expected = self.stage1_checkpoint(role["lineage"])
        found = pilot_report.lineage(summary.get("model_dir"))
        if expected is None:
            problems.append("%s: the selection records no stage-1 checkpoint of lineage %s, so its ancestry cannot be verified"
                            % (name, role["lineage"]))
        elif found != expected:
            problems.append("%s: trained from %r, not the selected stage-1 checkpoint %s attempt %d of lineage %s"
                            % (name, summary.get("model_dir"), expected[0], expected[1], role["lineage"]))
        problems += self.recipe_problems(run["key"], role, record)
        problems += self.rollout_problems(run["key"], role, record)
        return problems

    def pilot_summary(self, key: str) -> tuple:
        """(run-summary.json of a selected pilot run, None) or (None, why not), found by the layout of THIS tree,
        WORK/runs/<key>-a<attempt>/, never by a folder the selection recorded on the machine that made it
        (registration 5: recomputing from an extracted archive needs no path outside it). Used only when its sha256 is
        the one the selection recorded."""
        record = self.pilot.get(key) or {}
        if record.get("status") != "present" or not _int(record.get("attempt")):
            return None, "the selection records no run of record for %s" % key
        path = self.work / "runs" / ("%s-a%d" % (key, record["attempt"])) / "run-summary.json"
        digest = sha256_file(path)
        if digest is None:
            return None, "missing: %s" % path
        if record.get("run_summary_sha256") and digest != record["run_summary_sha256"]:
            return None, "%s differs from the run-summary.json the selection recorded (sha256 %s, recorded %s)" % (
                path, digest[:12], str(record["run_summary_sha256"])[:12])
        return _json(path), None

    def run_ok(self, key: str):
        """None when `key` is not a package-4 run or its run of record is valid; else why not."""
        if key not in self.roles:
            return None
        run = self.raw_run(key)
        return None if run["valid"] else "run %s not evaluated: %s" % (key, "; ".join(run["problems"]))

    def expected_checkpoint(self, key: str):
        """("base", None) | ((key, attempt), None) | (None, why) for the model a scoring of `key` must be of."""
        if key == BASE:
            return "base", None
        if key in self.pilot:
            record = self.pilot[key]
            if record.get("status") != "present" or not _int(record.get("attempt")):
                return None, "the selection records no run of record for %s" % key
            return (key, record["attempt"]), None
        run = self.raw_run(key)
        if run["record"] is None:
            return None, run["why"]
        return (key, run["record"]["attempt"]), None

    # ----------------------------------------------------------------------------------------------------- scorings
    def _model_problem(self, key: str, model, where: str):
        expected, why = self.expected_checkpoint(key)
        found = pilot_report.lineage(model)
        if expected is None:
            return "%s: %s" % (where, why)
        if expected == "base":
            if not isinstance(model, str) or not model:
                return "%s has no `model` field" % where
            if found is not None:
                return "%s scored %s, a checkpoint of %s attempt %d, not the untrained model" % (where, model, found[0], found[1])
            return None
        if found != expected:
            return "%s scored %r, not the checkpoint of record %s attempt %d" % (where, model, expected[0], expected[1])
        return None

    def _raw_evidence(self, sweep: dict, sweep_path: Path, area_folder: str, result_name: str, problems: list):
        """The scoring a sweep read, as raw evidence inside THIS tree's package-4 area: (result json, folder) or (None, None)."""
        scoring = sweep.get("scoring")
        parts = Path(scoring).parts if isinstance(scoring, str) and scoring else ()
        if len(parts) < 3 or parts[-3:-1] != (AREA, area_folder):
            problems.append("%s: its scoring %r is not in the package-4 area (%s/%s/<scoring>): refused" % (sweep_path, scoring, AREA, area_folder))
            return None, None
        folder = self.area / area_folder / parts[-1]
        result = _json(folder / result_name)
        if not isinstance(result, dict):
            problems.append("missing: %s (the raw scoring the sweep read)" % (folder / result_name))
            return None, None
        self._raw_files(folder, result, {"responses": sweep.get("responses_sha256"), "tokens": sweep.get("tokens_sha256")},
                        "the sweep", sweep_path, problems)
        return result, folder

    @staticmethod
    def _raw_files(folder: Path, result: dict, recorded: dict, by: str, where, problems: list, expected_ids=None) -> None:
        """The raw evidence of one scoring: responses.jsonl and its token file present, each with the sha256 `by`
        recorded; with `expected_ids`, both hold exactly those items, in that order (round-4 ruling G4: the same binding
        for the qualification scorings as for any other)."""
        tokens = folder / (result.get("tokens_file") or "tokens.jsonl")
        for label, path in (("responses", folder / "responses.jsonl"), ("tokens", tokens)):
            digest = sha256_file(path)
            if digest is None:
                problems.append("missing: %s (raw %s of the scoring)" % (path, label))
                continue
            if not recorded.get(label) or digest != recorded[label]:
                problems.append("%s: its %s file %s has sha256 %s, %s recorded %s" % (where, label, path.name, digest[:12], by,
                                                                                   str(recorded.get(label))[:12]))
            if expected_ids is not None:
                try:
                    found = [json.loads(line).get("id") for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
                except (OSError, ValueError, AttributeError):
                    found = None
                if found != list(expected_ids):
                    problems.append("%s: its %s file holds the items %s, not exactly the first %d registered items in file order %s"
                                    % (where, label, (found or [])[:10], len(expected_ids), list(expected_ids)))

    def _pilot_base(self, bed: str, rec: dict, problems: list):
        """Amendment 3 B8: the untrained model's scoring has the registered population, the pilot's own base sweep's items
        and the untrained model the pilot used."""
        folder, attempt = pilot_report.latest(self.pilot_sweeps, "%s-%s" % (BASE, bed))
        pilot = _json(folder / "sweep.json") if folder is not None else None
        if not isinstance(pilot, dict):
            problems.append("the pilot's own sweep of the untrained model on %s (k8b/report-sweep/base8b-%s-a<N>/sweep.json, highest "
                            "attempt) is not in the tree: the registered question set cannot be verified" % (bed, bed))
            return
        rec["pilot_base_sweep"] = "k8b/report-sweep/%s/sweep.json" % folder.name
        if pilot.get("n") != REGISTERED_N[bed]:
            problems.append("the pilot's base sweep of %s has n %r, registered %d" % (bed, pilot.get("n"), REGISTERED_N[bed]))
        if not rec.get("items_sha256") or rec["items_sha256"] != pilot.get("items_sha256"):
            problems.append("items_sha256 %s differs from the pilot's own base sweep's %s (%s)" % (str(rec.get("items_sha256"))[:12],
                                                                                              str(pilot.get("items_sha256"))[:12], bed))
        if not rec.get("model") or rec["model"] != pilot.get("model"):
            problems.append("the untrained model scored here (%r) is not the one the pilot scored (%r)" % (rec.get("model"), pilot.get("model")))
        for lin in LINEAGES:
            summary, why = self.pilot_summary(self.stage1[lin])
            if summary is None:
                problems.append("the stage-1 run %s cannot show which untrained model the pilot trained from: %s" % (self.stage1[lin], why))
            elif summary.get("model_dir") != rec.get("model"):
                problems.append("the stage-1 run %s trained from %r, not the untrained model scored here (%r)"
                                % (self.stage1[lin], summary.get("model_dir"), rec.get("model")))

    def bed(self, key: str, bed: str) -> dict:
        """The bed scoring of `key` on `bed` with every validation problem; `qualified` when it may enter a contrast."""
        if (key, bed) in self._bed:
            return self._bed[(key, bed)]
        rec = {"key": key, "bed": bed, "found": False, "readable": False, "qualified": False, "problems": [],
               "path": None, "attempt": None, "sweep_sha256": None, "per_item_sha256": None, "machine_id": None,
               "supplied_machine_id": None, "n": None, "verdicts": None, "ids": None, "signature": None}
        self._bed[(key, bed)] = rec
        stem = "%s-%s" % (key, bed)
        folder, attempt = pilot_report.latest(self.sweeps, stem)              # the HIGHEST attempt folder, never an older one
        if folder is None:
            rec["problems"].append("missing: %s" % (self.area / "report-sweep" / ("%s-a<N>" % stem) / "sweep.json"))
            return rec
        sweep_path, items_path = folder / "sweep.json", folder / "per_item.jsonl"
        rec.update({"attempt": attempt})
        if not sweep_path.is_file():
            rec["problems"].append("the scoring of record %s (attempt %d, the highest) has no sweep.json: not qualified, and no "
                                   "earlier attempt replaces it (amendment 3 B8)" % (folder, attempt))
            return rec
        rec.update({"found": True, "path": str(sweep_path.resolve()), "sweep_sha256": sha256_file(sweep_path),
                    "per_item_sha256": sha256_file(items_path), "per_item_path": str(items_path.resolve())})
        problems = rec["problems"]
        sweep = _json(sweep_path)
        if not isinstance(sweep, dict) or sweep.get("kind") != "bed" or sweep.get("bed") != bed:
            problems.append("%s is not a bed sweep of %s" % (sweep_path, bed))
            return rec
        n = sweep.get("n")
        budgets = sweep.get("budgets")
        per_budget = {e.get("budget"): e for e in sweep.get("per_budget") or [] if isinstance(e, dict)}
        if not _int(n) or n <= 0:
            problems.append("%s: n is %r" % (sweep_path, n))
            return rec
        if not isinstance(budgets, list) or not all(_int(b) for b in budgets):
            problems.append("%s: `budgets` is not a list of integers" % sweep_path)
            return rec
        for budget in (self.serving, self.diagnostic):
            if budget not in budgets or budget not in per_budget:
                problems.append("%s: budget %d is not among its budgets %s" % (sweep_path, budget, budgets))
        rec.update({"n": n, "budgets": budgets, "cap": sweep.get("cap"), "max_model_len": sweep.get("max_model_len"),
                    "model": sweep.get("model"), "scoring": sweep.get("scoring"), "items_sha256": sweep.get("items_sha256"),
                    "supplied_machine_id": (sweep.get("machine") or {}).get("id") if isinstance(sweep.get("machine"), dict) else None})
        if problems:
            return rec
        b_index, h_index = budgets.index(self.serving), budgets.index(self.diagnostic)
        try:
            text = items_path.read_text(encoding="utf-8")
        except OSError:
            problems.append("missing: %s" % items_path)
            return rec
        rows, seen, duplicates = [], set(), []
        for number, line in enumerate(text.split("\n"), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                problems.append("%s line %d is not JSON" % (items_path, number))
                continue
            strict = row.get("strict") if isinstance(row, dict) else None
            ident = row.get("id") if isinstance(row, dict) else None
            if not isinstance(ident, str) or not ident:
                problems.append("%s line %d has no id" % (items_path, number))
                continue
            if not isinstance(strict, list) or len(strict) != len(budgets) or not all(_int(v) and v in (0, 1) for v in strict):
                problems.append("%s line %d (%s): `strict` is not a 0/1 list aligned with the %d budgets" % (items_path, number, ident, len(budgets)))
                continue
            if ident in seen:
                duplicates.append(ident)
                continue
            seen.add(ident)
            rows.append(row)
        if duplicates:
            problems.append("%s: duplicate question ids %s" % (items_path, ", ".join(sorted(set(duplicates))[:5])))
        total = len(rows) + len(duplicates)
        if total != n:
            problems.append("%s has %d rows; sweep.json says n = %d" % (items_path, total, n))
        if problems:
            return rec
        verdicts = {row["id"]: row["strict"][b_index] for row in rows}
        for budget, index in ((self.serving, b_index), (self.diagnostic, h_index)):
            counted = sum(row["strict"][index] for row in rows)
            if per_budget[budget].get("correct_strict") != counted:
                problems.append("%s: %d strict-correct at %d in per_item.jsonl, %r in sweep.json" % (sweep_path, counted, budget,
                                                                                               per_budget[budget].get("correct_strict")))
        rec.update({"verdicts": verdicts, "ids": sorted(verdicts), "readable": True,
                    "strict_serving": sum(verdicts.values()), "per_budget": {str(b): per_budget[b] for b in (self.serving, self.diagnostic)}})
        if n != REGISTERED_N.get(bed):
            problems.append("%s: n = %d, the registered %s population is %s questions (amendment 3 B8)" % (sweep_path, n, bed, REGISTERED_N.get(bed)))
        problems += self.registered_limits(sweep, sweep_path)
        if sweep.get("reproduces_scoring_mismatches") != 0:
            problems.append("%s: reproduces_scoring_mismatches is %r, not 0" % (sweep_path, sweep.get("reproduces_scoring_mismatches")))
        if sweep.get("decode_equals_text") != n:
            problems.append("%s: decode_equals_text is %r, not n = %d" % (sweep_path, sweep.get("decode_equals_text"), n))
        model_problem = self._model_problem(key, sweep.get("model"), "%s (the sweep's model)" % sweep_path)
        if model_problem:
            problems.append(model_problem)
        machine_id, why = recompute_machine_id(sweep.get("machine"))
        scored, scored_folder = self._raw_evidence(sweep, sweep_path, "eval", "bed-score.json", problems)
        if scored is not None:
            rec["bed_score"] = str((scored_folder / "bed-score.json").resolve())
            scored_id, scored_why = recompute_machine_id(scored.get("machine"))
            if scored_id is None:
                problems.append("%s: %s: the registered machine id cannot be recomputed" % (scored_folder / "bed-score.json", scored_why))
            elif machine_id is not None and machine_id != scored_id:
                problems.append("%s and %s record different machine fields (recomputed %s vs %s)" % (scored_folder / "bed-score.json", sweep_path, scored_id, machine_id))
            machine_id = scored_id
            if scored.get("model") != sweep.get("model"):
                problems.append("%s scored %r, its sweep says %r" % (scored_folder / "bed-score.json", scored.get("model"), sweep.get("model")))
            blocks = {k: scored.get(k) for k in ("engine", "generation", "decoding")}
            rec["raw_signature"] = raw_signature(scored)
            if not all(isinstance(v, dict) and v for v in blocks.values()):
                problems.append("%s: the engine, generation or decoding block is missing: the decoding cannot be compared" % (scored_folder / "bed-score.json"))
            problems += self.raw_limits(scored, scored_folder / "bed-score.json")
            problems += self.scorer_problems(scored, sweep, scored_folder / "bed-score.json", sweep_path)
            rec["scorer_sha256"] = scored.get("scorer_sha256")
            rec["signature"] = json.dumps({**blocks, "max_new_tokens": scored.get("max_new_tokens"), "max_model_len": sweep.get("max_model_len"),
                                           "canonical_rule": sweep.get("canonical_rule"),
                                           "scorer": {"bed_score_schema": scored.get("schema"), "mode": scored.get("mode"),
                                                      "sweep_schema": sweep.get("schema"),
                                                      "sweep_source_sha256": sweep.get("sweep_source_sha256")}}, sort_keys=True)
            problems += self.currentness("%s-%s" % (key, bed), "sweep-%s-%s" % (key, bed), "eval", sweep, attempt)
        elif machine_id is None:
            problems.append("%s: %s: the registered machine id cannot be recomputed" % (sweep_path, why))
        rec["machine_id"] = machine_id
        if key != BASE:
            base = self.bed(BASE, bed)
            if not base["qualified"]:
                problems.append("no expected question set for %s: the untrained model's %s scoring is not qualified" % (bed, bed))
            else:
                if set(verdicts) != set(base["verdicts"]):
                    missing, extra = set(base["verdicts"]) - set(verdicts), set(verdicts) - set(base["verdicts"])
                    problems.append("%s: question ids differ from the untrained model's %s sweep (missing %s; extra %s)"
                                    % (items_path, bed, ", ".join(sorted(missing)[:5]) or "none", ", ".join(sorted(extra)[:5]) or "none"))
                if not rec["items_sha256"] or rec["items_sha256"] != base["items_sha256"]:
                    problems.append("%s: items_sha256 %s differs from the untrained model's %s" % (sweep_path, str(rec["items_sha256"])[:12],
                                                                                              str(base["items_sha256"])[:12]))
        else:
            if not rec["items_sha256"]:
                problems.append("%s records no items_sha256" % sweep_path)
            self._pilot_base(bed, rec, problems)
        rec["qualified"] = not problems
        return rec

    def panel(self, key: str) -> dict:
        """The general-panel sweep of `key`: totals at the serving and diagnostic budgets (reported, no bar), and whether it
        is a complete, identified, CURRENT scoring of the checkpoint of record (`qualified`, needed for block
        completeness): the registered population, cap, context and budgets, both budget totals, raw evidence in the
        package-4 area whose model is the checkpoint of record, the scorer's identity, and the runner's latest attempts
        (round-3 ruling F5)."""
        if key in self._panel:
            return self._panel[key]
        stem = "%s-panel" % key
        folder, attempt = pilot_report.latest(self.sweeps, stem)
        out = {"key": key, "found": False, "qualified": False, "problems": []}
        self._panel[key] = out
        if folder is None or not (folder / "sweep.json").is_file():
            out["note"] = "missing: %s" % (self.area / "report-sweep" / ("%s-a%s" % (stem, attempt or "<N>")) / "sweep.json")
            out["problems"].append(out["note"])
            return out
        sweep_path = folder / "sweep.json"
        sweep = _json(sweep_path)
        out.update({"found": True, "path": str(sweep_path.resolve()), "attempt": attempt, "sweep_sha256": sha256_file(sweep_path)})
        if not isinstance(sweep, dict) or sweep.get("kind") != "panel":
            out["note"] = "not a panel sweep"
            out["problems"].append(out["note"])
            return out
        problem = self._model_problem(key, sweep.get("model"), "%s (the sweep's model)" % sweep_path)
        out["machine_id"] = recompute_machine_id(sweep.get("machine"))[0]
        by_budget = {e.get("budget"): e for e in sweep.get("per_budget") or [] if isinstance(e, dict)}
        for label, budget in (("serving", self.serving), ("diagnostic", self.diagnostic)):
            total = (by_budget.get(budget) or {}).get("total")
            ok = isinstance(total, dict) and _int(total.get("correct")) and _int(total.get("n"))
            out[label] = {"correct": total.get("correct"), "n": total.get("n")} if ok else None
            if not ok:
                out["problems"].append("%s: no budget total at the %s budget %d" % (sweep_path, label, budget))
        out["note"] = problem
        if problem:
            out["problems"].append(problem)
        if sweep.get("n") != PANEL_N:
            out["problems"].append("%s: n %r, the general panel has %d members" % (sweep_path, sweep.get("n"), PANEL_N))
        if out["machine_id"] is None:
            out["problems"].append("%s: the registered machine id cannot be recomputed" % sweep_path)
        out["problems"].extend(self.registered_limits(sweep, sweep_path))
        scored, scored_folder = self._raw_evidence(sweep, sweep_path, "forgetting", "forgetting.json", out["problems"])
        if scored is not None:
            raw_model = scored.get("model")
            raw_model = raw_model.get("path") if isinstance(raw_model, dict) else raw_model
            raw_problem = self._model_problem(key, raw_model, "%s (the raw scoring's model)" % (scored_folder / "forgetting.json"))
            if raw_problem:
                out["problems"].append(raw_problem)
            if raw_model != sweep.get("model"):
                out["problems"].append("%s scored %r, its sweep says %r" % (scored_folder / "forgetting.json", raw_model, sweep.get("model")))
            out["problems"].extend(self.raw_limits(scored, scored_folder / "forgetting.json"))
            out["problems"].extend(self.scorer_problems(scored, sweep, scored_folder / "forgetting.json", sweep_path))
            raw_id = recompute_machine_id(scored.get("machine"))[0]
            if raw_id is None or raw_id != out["machine_id"]:
                out["problems"].append("%s and %s record different or no machine fields" % (scored_folder / "forgetting.json", sweep_path))
            out["scorer_sha256"] = scored.get("scorer_sha256")
        out["problems"].extend(self.currentness(stem, "sweep-%s" % stem, "forgetting", sweep, attempt, scoring_stem=key))
        out["qualified"] = not out["problems"]
        return out

    # ------------------------------------------------------------------------------------- limits, scorer, currentness
    def registered_limits(self, sweep: dict, where: Path) -> list:
        """Round-3 ruling F5: every sweep records the registered cap 8192, context 12288 and budgets."""
        problems = []
        if not _is(sweep.get("cap"), REGISTERED_CAP):
            problems.append("%s: cap %r, registered %d" % (where, sweep.get("cap"), REGISTERED_CAP))
        if not _is(sweep.get("max_model_len"), REGISTERED_CONTEXT):
            problems.append("%s: max_model_len %r, registered %d" % (where, sweep.get("max_model_len"), REGISTERED_CONTEXT))
        if sweep.get("budgets") != REGISTERED_BUDGETS:
            problems.append("%s: budgets %r, registered %r" % (where, sweep.get("budgets"), REGISTERED_BUDGETS))
        return problems

    def raw_limits(self, scored: dict, where: Path, cap: int = REGISTERED_CAP) -> list:
        """The raw scoring's own limits: max_new_tokens and the engine's context."""
        problems = []
        if not _is(scored.get("max_new_tokens"), cap):
            problems.append("%s: max_new_tokens %r, registered %d" % (where, scored.get("max_new_tokens"), cap))
        if not _is((scored.get("engine") or {}).get("max_model_len"), REGISTERED_CONTEXT):
            problems.append("%s: engine max_model_len %r, registered %d" % (where, (scored.get("engine") or {}).get("max_model_len"), REGISTERED_CONTEXT))
        return problems

    def scorer_problems(self, scored: dict, sweep, scored_path: Path, sweep_path) -> list:
        """Round-3 ruling F5: the scoring records its scorer's implementation identity (`scorer_sha256`), and the sweep that
        read it copied the same value and recorded its own source hash."""
        problems = []
        value = scored.get("scorer_sha256")
        if not isinstance(value, str) or len(value) != 64:
            problems.append("%s records no scorer_sha256: the scorer's implementation identity is unknown" % scored_path)
        if sweep is not None:
            if sweep.get("scorer_sha256") != value:
                problems.append("%s: scorer_sha256 %s, its scoring's %s" % (sweep_path, str(sweep.get("scorer_sha256"))[:12], str(value)[:12]))
            if not isinstance(sweep.get("sweep_source_sha256"), str):
                problems.append("%s records no sweep_source_sha256" % sweep_path)
        return problems

    def row_latest(self, row_id: str) -> tuple:
        """(attempt, verdict, inputs) of the runner's LATEST attempt of a campaign row, or (None, None, None)."""
        if self.campaign is None:
            return None, None, None
        base = self.work / "campaign" / self.campaign["name"] / row_id
        found = sorted(int(p.name.split("-")[1]) for p in base.glob("attempt-*") if p.name.split("-")[1].isdigit()) if base.is_dir() else []
        if not found:
            return None, None, None
        verdict = _json(base / ("attempt-%d" % found[-1]) / "verdict.json") or {}
        return found[-1], verdict.get("verdict"), verdict.get("inputs")

    def currentness(self, scoring_row: str, sweep_row: str, area_folder: str, sweep: dict, sweep_attempt, scoring_stem=None) -> list:
        """Round-3 ruling F5: a sweep is CURRENT only if the scoring it read is the HIGHEST scoring attempt folder of its
        key and bed, the runner's latest attempts of the scoring row and of the sweep row both PASSED, the sweep of record
        is the sweep row's latest attempt, and that attempt read the scoring row's latest attempt (its `inputs`)."""
        problems = []
        if self.campaign is None:
            return ["no campaign file was given (--campaign): whether %s is current cannot be verified" % sweep_row]
        stem = scoring_stem or scoring_row
        index = pilot_report.attempts(self.area / area_folder)
        highest, number = pilot_report.latest(index, stem)
        read = Path(sweep.get("scoring")).name if isinstance(sweep.get("scoring"), str) and sweep.get("scoring") else None
        if highest is None or read != highest.name:
            problems.append("%s read the scoring %s, not the highest scoring attempt %s" % (sweep_row, read, highest.name if highest else None))
        s_attempt, s_verdict, _ = self.row_latest(scoring_row)
        w_attempt, w_verdict, w_inputs = self.row_latest(sweep_row)
        if s_verdict != "PASS":
            problems.append("the runner's latest attempt of %s (%s) is %s, not PASS" % (scoring_row, s_attempt, s_verdict))
        elif number is not None and s_attempt != number:
            problems.append("the highest scoring folder of %s is attempt %s, the runner's latest attempt %s" % (scoring_row, number, s_attempt))
        if w_verdict != "PASS":
            problems.append("the runner's latest attempt of %s (%s) is %s, not PASS" % (sweep_row, w_attempt, w_verdict))
        elif sweep_attempt != w_attempt:
            problems.append("the sweep of record of %s is attempt %s, the runner's latest attempt %s" % (sweep_row, sweep_attempt, w_attempt))
        elif not isinstance(w_inputs, dict) or w_inputs.get(scoring_row) != [s_attempt, "PASS"]:
            problems.append("the latest attempt of %s read %s at %r, not its latest attempt %s: the sweep is older than the scoring"
                            % (sweep_row, scoring_row, (w_inputs or {}).get(scoring_row) if isinstance(w_inputs, dict) else None, s_attempt))
        return problems

    # ------------------------------------------------------------------------------------------------ prefix checks
    def prefix_check(self, key: str, bed: str) -> dict:
        """A required prefix check (amendment 3 B4, B8): it compared the bed's FULL registered population, agrees on at
        least 99 percent of the texts, is bound to its own long scoring, and its short scoring is of the SAME model, in
        the package-4 area, at the serving cap, on the same recomputed machine."""
        if (key, bed) in self._prefix:
            return self._prefix[(key, bed)]
        rec = self.bed(key, bed)
        check = self.prefix_index.get((key, bed))
        out = {"key": key, "bed": bed, "found": check is not None, "passed": None, "bound": None, "machine_id": None, "ok": False,
               "file": check["file"] if check else str(self.area / "report-prefix" / ("prefix-%s-%s-a<N>.json" % (key, bed))),
               "text_agreement": check.get("text_agreement") if check else None, "compared": check.get("compared") if check else None,
               "why": None}
        self._prefix[(key, bed)] = out
        if check is None:
            out["why"] = "missing: %s" % out["file"]
            return out
        agreement, compared = check.get("text_agreement"), check.get("compared")
        out["passed"] = bool(_int(compared) and compared == REGISTERED_N.get(bed) and isinstance(agreement, (int, float))
                             and not isinstance(agreement, bool) and agreement >= PREFIX_AGREEMENT
                             and check["same_machine"] is True and check["same_max_model_len"] is True)
        scoring = Path(rec["scoring"]).name if isinstance(rec.get("scoring"), str) and rec.get("scoring") else None
        out["bound"] = bool(rec["found"] and check["long_scoring"] is not None and check["long_scoring"] == scoring
                            and check["long_cap"] == rec.get("cap") and check["short_cap"] == self.serving)
        data = _json(Path(check["file"])) or {}
        short = data.get("short") if isinstance(data, dict) else None
        parts = Path(short).parts if isinstance(short, str) and short else ()
        in_area = len(parts) >= 3 and parts[-3:-1] == (AREA, "eval")
        short_score = self.area / "eval" / parts[-1] / "bed-score.json" if parts else None
        scored = _json(short_score) if short_score is not None and in_area else None
        long_scored = _json(Path(rec["bed_score"])) if rec.get("bed_score") else None
        short_id, short_why = recompute_machine_id(scored.get("machine") if isinstance(scored, dict) else None)
        out["machine_id"] = short_id
        out["short_scoring"] = str(short_score.resolve()) if short_score is not None else None
        same_model = isinstance(scored, dict) and isinstance(long_scored, dict) and bool(scored.get("model")) \
            and scored.get("model") == long_scored.get("model")
        out["short_model_is_the_long_model"] = same_model
        if not out["passed"]:
            out["why"] = ("failed: compared %s of the registered %s, text agreement %s, same machine %s, same max_model_len %s "
                          "(needs the whole panel, agreement >= 0.99, one machine and one context)"
                          % (compared, REGISTERED_N.get(bed), agreement, check["same_machine"], check["same_max_model_len"]))
        elif not out["bound"]:
            out["why"] = ("of another scoring: its long scoring is %s at caps %s/%s; the sweep of record read %s at cap %s, serving %d"
                          % (check["long_scoring"], check["short_cap"], check["long_cap"], scoring, rec.get("cap"), self.serving))
        elif not rec["qualified"]:
            out["why"] = "its long scoring is not qualified: %s" % "; ".join(rec["problems"])
        elif not in_area or not isinstance(scored, dict):
            out["why"] = "its short scoring %r is not a scoring in the package-4 area of this tree" % short
        elif scored.get("max_new_tokens") != self.serving:
            out["why"] = "its short scoring was made at %r new tokens, not the serving %d" % (scored.get("max_new_tokens"), self.serving)
        elif not same_model:
            out["why"] = "its short scoring is of %r, not of the long scoring's model %r" % (scored.get("model"), (long_scored or {}).get("model"))
        elif short_id is None:
            out["why"] = "the short scoring's machine id cannot be recomputed (%s: %s)" % (short_score, short_why)
        elif short_id != rec["machine_id"]:
            out["why"] = "the short scoring ran on another machine (recomputed %s vs the long scoring's %s)" % (short_id, rec["machine_id"])
        else:
            binding = self.short_binding(key, bed, check, data, short_score.parent, scored, rec)
            if binding:
                out["why"] = "; ".join(binding)
            else:
                out["ok"] = True
        return out

    def short_binding(self, key: str, bed: str, check: dict, data: dict, short_folder: Path, scored: dict, rec: dict) -> list:
        """Round-3 ruling F5: the short (direct 2,048) scoring of a prefix check gets the raw-evidence binding of a long
        one: its responses and token files are the ones the check read (sha256), its model is the checkpoint of record,
        its limits are the registered ones, its scorer is the long scoring's; the long scoring's files are the ones the
        sweep read; and the check is current: the runner's latest attempts of the prefix row, the direct row and the long
        scoring row passed, and the prefix row's latest attempt read the other two at their latest attempts."""
        problems = []
        name = Path(check["file"]).name
        for label, path, recorded in (("short responses", short_folder / "responses.jsonl", data.get("short_responses_sha256")),
                                      ("short tokens", short_folder / (scored.get("tokens_file") or "tokens.jsonl"), data.get("short_tokens_sha256")),
                                      ("long responses", Path(rec["bed_score"]).parent / "responses.jsonl", data.get("long_responses_sha256"))):
            digest = sha256_file(path)
            if digest is None or not recorded or digest != recorded:
                problems.append("%s: its %s file %s has sha256 %s, the check recorded %s" % (name, label, path.name, str(digest)[:12], str(recorded)[:12]))
        model_problem = self._model_problem(key, scored.get("model"), "%s (the short scoring's model)" % (short_folder / "bed-score.json"))
        if model_problem:
            problems.append(model_problem)
        problems += self.raw_limits(scored, short_folder / "bed-score.json", cap=self.serving)
        if not scored.get("scorer_sha256") or scored.get("scorer_sha256") != rec.get("scorer_sha256"):
            problems.append("%s: the short scoring's scorer_sha256 %s is not the long scoring's %s" % (name, str(scored.get("scorer_sha256"))[:12],
                                                                                                  str(rec.get("scorer_sha256"))[:12]))
        if self.campaign is None:
            problems.append("no campaign file was given (--campaign): whether %s is current cannot be verified" % name)
            return problems
        prefix_row, direct_row, long_row = "prefix-%s-%s" % (key, bed), "direct-%s-%s" % (key, bed), "%s-%s" % (key, bed)
        p_attempt, p_verdict, p_inputs = self.row_latest(prefix_row)
        d_attempt, d_verdict, _ = self.row_latest(direct_row)
        l_attempt, l_verdict, _ = self.row_latest(long_row)
        direct_index = pilot_report.attempts(self.area / "eval")
        highest, number = pilot_report.latest(direct_index, "direct-%s-%s" % (key, bed))
        if highest is None or short_folder.name != highest.name or number != d_attempt:
            problems.append("%s: its short scoring %s is not the highest direct scoring attempt %s of the runner's latest attempt %s"
                            % (name, short_folder.name, highest.name if highest else None, d_attempt))
        match = re.search(r"-a(\d+)\.json$", name)
        if p_verdict != "PASS" or d_verdict != "PASS" or l_verdict != "PASS":
            problems.append("%s: the runner's latest attempts of %s, %s and %s are %s, %s and %s, not all PASS"
                            % (name, prefix_row, direct_row, long_row, p_verdict, d_verdict, l_verdict))
        elif not match or int(match.group(1)) != p_attempt:
            problems.append("%s is not the prefix row's latest attempt %s" % (name, p_attempt))
        elif not isinstance(p_inputs, dict) or p_inputs.get(direct_row) != [d_attempt, "PASS"] or p_inputs.get(long_row) != [l_attempt, "PASS"]:
            problems.append("%s: the prefix row's latest attempt read %s and %s at %r and %r, not their latest attempts %s and %s"
                            % (name, direct_row, long_row, (p_inputs or {}).get(direct_row) if isinstance(p_inputs, dict) else None,
                               (p_inputs or {}).get(long_row) if isinstance(p_inputs, dict) else None, d_attempt, l_attempt))
        return problems

    # ---------------------------------------------------------------------------------------------- use in contrasts
    def usable(self, key: str, bed: str):
        """(record, None) when the scoring of `key` on `bed` may enter a contrast, else (None, why)."""
        rec = self.bed(key, bed)
        why = []
        if not rec["qualified"]:
            why.append("scoring %s on %s not qualified: %s" % (key, bed, "; ".join(rec["problems"])))
        run = self.run_ok(key)
        if run:
            why.append(run)
        return (rec, None) if not why else (None, "; ".join(why))

    def gather(self, needed: list) -> tuple:
        """({(key, bed): record}, [why]) for every (key, bed) needed; one recomputed machine id is required among them."""
        recs, whys = {}, []
        for key, bed in needed:
            rec, why = self.usable(key, bed)
            if why:
                whys.append(why)
            else:
                recs[(key, bed)] = rec
        ids = sorted({r["machine_id"] for r in recs.values()})
        if len(ids) > 1:
            whys.append("the scorings were made on more than one physical GPU (recomputed machine ids %s)" % ", ".join(ids))
        signatures = {r["signature"] for r in recs.values()}
        scorers = {}
        for (key, bed), r in recs.items():                     # a bed's scorer is its own: one value PER BED (round 3, F5)
            scorers.setdefault(bed, set()).add(r.get("scorer_sha256"))
        if len(signatures) > 1 or any(len(v) > 1 for v in scorers.values()):
            whys.append("the scorings differ in their engine, generation, decoding, limits, canonical rule or scorer")
        return recs, whys

    def block_identity(self, needed: list, prefix_keys: tuple) -> tuple:
        """(problems, prefix checks, machine ids, signatures) of a block: required prefix checks, one machine, and ONE
        scoring signature (the whole engine, generation and decoding blocks, max_new_tokens, max_model_len, canonical
        rule and scorer) over every scoring entering its contrasts."""
        problems, checks = [], []
        for key in prefix_keys:
            for bed in BEDS:
                check = self.prefix_check(key, bed)
                checks.append(check)
                if not check["ok"]:
                    problems.append("required prefix check of %s on %s: %s" % (key, bed, check["why"]))
        recs = [self.bed(k, b) for k, b in needed]
        ids = {r["machine_id"] for r in recs if r["qualified"]} | {c["machine_id"] for c in checks if c["ok"]}
        if len(ids) > 1:
            problems.append("the scorings entering this block's contrasts were made on more than one physical GPU (recomputed "
                            "machine ids %s)" % ", ".join(sorted(ids)))
        signatures = {r["signature"] for r in recs if r["qualified"]}
        if len(signatures) > 1:
            problems.append("the scorings entering this block's contrasts differ in engine, generation, decoding, max_new_tokens, "
                            "max_model_len, canonical rule or scorer (%d different records)" % len(signatures))
        scorers = {}
        for r in recs:
            if r["qualified"]:
                scorers.setdefault(r["bed"], set()).add(r.get("scorer_sha256"))
        for bed, values in sorted(scorers.items()):
            if len(values) > 1:
                problems.append("the scorings entering this block's contrasts differ in engine, generation, decoding or scorer: the %s "
                                "scorings record %d different scorer_sha256 values (one implementation per bed)" % (bed, len(values)))
        return problems, checks, sorted(ids), [json.loads(s) for s in sorted(x for x in signatures if x)]

    # ----------------------------------------------------------------------------------------- completeness and budget
    def row_status(self, row_id: str) -> tuple:
        """(latest attempt, verdict or None) of a campaign row from the runner's records in this tree."""
        base = self.work / "campaign" / self.campaign["name"] / row_id
        found = sorted(int(p.name.split("-")[1]) for p in base.glob("attempt-*") if p.name.split("-")[1].isdigit()) if base.is_dir() else []
        if not found:
            return None, None
        verdict = _json(base / ("attempt-%d" % found[-1]) / "verdict.json")
        return found[-1], (verdict or {}).get("verdict")

    def completeness(self, block: str, checkpoints: list) -> list:
        """Every registered artifact of a whole block (R7; amendment 3 B8). Empty when the block is complete."""
        problems = []
        if self.campaign is None:
            return ["no campaign file was given (--campaign): the runner's statuses of the block cannot be read"]
        rows = [r["id"] for r in self.campaign["rows"] if (r.get("env") or {}).get("P4_BLOCK") == block]
        if block == "bridges":
            # the bridges also need what they reuse from the prevention block's rows: the reservation, the recipe check,
            # the reused controls and their scorings, the untrained model's and the incoming checkpoint's scorings, and
            # the required prefix checks (supplement S2, S5)
            ids = {r["id"] for r in self.campaign["rows"]}
            reused = [self.bridge_keys[(self.reused, seed)] for seed in SEEDS]
            shared = ["reserve", "recipe-check"] + reused
            for key in [BASE, self.stage1["r1"]] + reused:
                shared += ["%s-%s" % (key, t) for t in BEDS + ("panel",)] + ["sweep-%s-%s" % (key, t) for t in BEDS + ("panel",)]
            for key in BRIDGE_PREFIX:
                shared += ["prefix-%s-%s" % (key, bed) for bed in BEDS] + ["direct-%s-%s" % (key, bed) for bed in BEDS]
            rows = [r for r in dict.fromkeys(shared) if r in ids and r not in rows] + rows
        for row_id in rows:
            attempt, verdict = self.row_status(row_id)
            if attempt is None:
                problems.append("row %s never ran" % row_id)
            elif verdict not in ("PASS", "FAIL"):
                problems.append("row %s attempt %d has no terminal verdict" % (row_id, attempt))
            elif verdict != "PASS":                         # a failed statistics row too (round-3 ruling F6)
                problems.append("row %s ended %s (attempt %d): a terminal failure is not a passed check" % (row_id, verdict, attempt))
        if not isinstance(_json(self.area / "budget" / "reservation.json"), dict):
            problems.append("the reservation k8b4/budget/reservation.json is not in the tree")
        if not isinstance(self.recipe_record, dict) or self.recipe_record.get("ok") is not True:
            problems.append("no passed recipe-check record verified (k8b4/recipe-check/baseline.json)%s"
                            % (": " + "; ".join(self.baseline_problems) if self.baseline_problems else ""))
        if block == "bridges":
            folder = self.area / "report-admit"
            found = sorted((int(p.stem.rsplit("-a", 1)[1]), p) for p in folder.glob("bridges-admit-a*.json") if p.stem.rsplit("-a", 1)[1].isdigit()) if folder.is_dir() else []
            admit = _json(found[-1][1]) if found else None
            if not isinstance(admit, dict) or admit.get("may_launch") != 1 or admit.get("prerequisites_present") != 1:
                problems.append("the bridges admission record (k8b4/report-admit/bridges-admit-a<N>.json) does not admit the block")
        for key in checkpoints:
            for task in BEDS:
                if not self.bed(key, task)["qualified"]:
                    problems.append("the long scoring of %s on %s is not qualified" % (key, task))
            if not self.panel(key)["qualified"]:
                problems.append("the general-panel scoring of %s is not qualified: %s" % (key, "; ".join(self.panel(key)["problems"])))
        panel_scorers = {self.panel(key).get("scorer_sha256") for key in checkpoints if self.panel(key)["qualified"]}
        if len(panel_scorers) > 1:
            problems.append("the general-panel scorings of the block record %d different scorer_sha256 values" % len(panel_scorers))
        # round-3 ruling F6: EVERY launched training attempt of the block's runs, the failed ones included, reduced
        runs = [k for k in checkpoints if k in self.roles]
        if block == "prevention":
            runs += list(QUALIFICATION)
            problems += self.qualification_problems(checkpoints)
        for key in runs:
            gated = key in QUALIFICATION or (self.roles.get(key) or {}).get("kind") == "gate"
            problems += self.every_attempt_reduced(key, gated)
        return problems

    def registered_order(self, bed: str):
        """(the registered items of `bed` in file order, None) from the untrained model's qualified long scoring of record
        (its responses.jsonl, bound by its sweep's hash; its item set is the pilot's own), or (None, why)."""
        base = self.bed(BASE, bed)
        if not base["qualified"] or not base.get("bed_score"):
            return None, "the untrained model's %s scoring is not qualified, so the registered item order is unknown" % bed
        try:
            ids = [json.loads(line).get("id") for line in (Path(base["bed_score"]).parent / "responses.jsonl").read_text(encoding="utf-8").split("\n")
                   if line.strip()]
        except (OSError, ValueError, AttributeError) as exc:
            return None, "the untrained model's %s responses cannot be read: %s" % (bed, exc)
        if sorted(ids) != sorted(base["verdicts"]):
            return None, "the untrained model's %s responses are not its sweep's items" % bed
        return ids, None

    def qualification_problems(self, checkpoints: list | None = None) -> list:
        """Registration 6 and round-3 ruling F5: the two qualification runs validated from their ARTIFACTS, not from PASS
        records alone: each run of record's summary (2 steps, seed 101, the selected recipe's configuration with the
        intervention settings, its registered task, lineage r1's selected stage-1 checkpoint), its actual command and
        resolved configuration against its frozen baseline, and its scoring (n == 8, the run's own checkpoint, in the
        package-4 area, the registered limits and a recorded scorer)."""
        problems = []
        config = CONFIGS[self.reused]
        expected = self.stage1_checkpoint("r1")
        for key, task in QUALIFICATION.items():
            run = self.raw_run_plain(key)
            if run is None:
                problems.append("qualification %s: no attempt with a valid export (merged 1, returncode 0)" % key)
                continue
            attempt, folder, summary = run
            name = folder.name
            want = {"steps": QUAL_STEPS, "seed": QUAL_SEED, "max_response_length": GATE_RESPONSE, "finish_gate": 1,
                    "dataset": DATASETS[task], "schema": config["schema"]}
            for field, value in want.items():
                got = summary.get(field)
                if not (got == value or (isinstance(value, int) and _same_int(got, value))):
                    problems.append("qualification %s: run-summary %s %r, registered %r" % (name, field, got, value))
            if not _same_fraction(summary.get("learning_rate"), config["lr"]):
                problems.append("qualification %s: learning rate %r, registered %s" % (name, summary.get("learning_rate"), config["lr"]))
            if expected is None or pilot_report.lineage(summary.get("model_dir")) != expected:
                problems.append("qualification %s: trained from %r, not lineage r1's selected stage-1 checkpoint" % (name, summary.get("model_dir")))
            verdict, why = recipe.verify_run(self.work, self.recipe_record, key, folder, self.area / "recipe-check")
            if verdict is not True:
                problems += ["qualification %s: %s" % (name, w) for w in (why or ["the recipe cannot be verified"])]
            scorings = pilot_report.attempts(self.area / "eval")
            scoring, number = pilot_report.latest(scorings, "%s-%s" % (key, task))
            scored = _json(scoring / "bed-score.json") if scoring is not None else None
            row_attempt, row_verdict, _ = self.row_latest("%s-score" % key)
            if not isinstance(scored, dict):
                problems.append("qualification %s: no scoring k8b4/eval/%s-%s-a<N>/bed-score.json" % (name, key, task))
                continue
            where = scoring / "bed-score.json"
            if not _is(scored.get("n"), QUAL_N) or not _is(scored.get("limit"), QUAL_N):
                problems.append("%s: n %r, limit %r; the qualification scoring is the first %d questions" % (where, scored.get("n"), scored.get("limit"), QUAL_N))
            if scored.get("bed") != task:
                problems.append("%s: bed %r, registered %s" % (where, scored.get("bed"), task))
            if pilot_report.lineage(scored.get("model")) != (key, attempt):
                problems.append("%s scored %r, not the qualification run of record %s" % (where, scored.get("model"), name))
            problems += self.raw_limits(scored, where)
            problems += self.scorer_problems(scored, None, where, None)
            # round-4 ruling G4: raw evidence, bound like any other scoring -- the responses and token records of exactly
            # the first eight registered items of the task in file order, with the sha256 the scoring attempt recorded,
            # and the block's machine, decoding and scorer
            expected_ids, why = self.registered_order(task)
            if expected_ids is None:
                problems.append("qualification %s: %s" % (name, why))
            else:
                first = expected_ids[:QUAL_N]
                self._raw_files(scoring, scored, {"responses": scored.get("responses_sha256"), "tokens": scored.get("tokens_sha256")},
                                "the scoring attempt", where, problems, expected_ids=first)
                digest = hashlib.sha256("".join("%s\n" % i for i in first).encode("utf-8")).hexdigest()
                if scored.get("items_sha256") != digest:
                    problems.append("%s: items_sha256 %s is not that of the first %d registered %s items in file order (%s)"
                                    % (where, str(scored.get("items_sha256"))[:12], QUAL_N, task, digest[:12]))
            block = [self.bed(k, b) for k in (checkpoints or []) for b in BEDS]
            block = [r for r in block if r["qualified"]]
            ids = {r["machine_id"] for r in block}
            machine_id, machine_why = recompute_machine_id(scored.get("machine"))
            if len(ids) != 1:
                problems.append("qualification %s: the block's scorings record %d recomputed machine ids: the qualification scoring "
                                "cannot be bound to the block's machine" % (name, len(ids)))
            elif machine_id not in ids:
                problems.append("%s: recomputed machine id %s (%s), the block's scorings %s" % (where, machine_id, machine_why or "recomputed",
                                                                                            sorted(ids)[0]))
            signatures = {r.get("raw_signature") for r in block if r["bed"] == task}
            if len(signatures) != 1 or raw_signature(scored) not in signatures:
                problems.append("%s: its engine, generation, decoding, limit or scorer mode is not the block's %s scorings'" % (where, task))
            scorers = {r.get("scorer_sha256") for r in block if r["bed"] == task}
            if len(scorers) != 1 or scored.get("scorer_sha256") not in scorers:
                problems.append("%s: scorer_sha256 %s is not the block's %s scorer" % (where, str(scored.get("scorer_sha256"))[:12], task))
            if row_verdict != "PASS" or row_attempt != number:
                problems.append("qualification %s: the runner's latest attempt of %s-score is %s (%s), the scoring of record is attempt %s"
                                % (name, key, row_attempt, row_verdict, number))
        return problems

    def raw_run_plain(self, key: str):
        """(attempt, folder, summary) of a run's FIRST attempt with merged 1 and returncode 0, or None."""
        for attempt, folder in sorted(self.runs_index.get(key, []), key=lambda pair: pair[0]):
            summary = _json(folder / "run-summary.json")
            if isinstance(summary, dict) and _is(summary.get("merged"), 1) and _is(summary.get("returncode"), 0):
                return attempt, folder, summary
        return None

    def budget(self, status_path) -> dict:
        """{block: {label, why, stop}} from the durable stop records, the budget status (--budget-status) and the ledger
        (amendment 3 B3). label: None, "budget-incomplete" or "budget compliance unavailable"."""
        status = _json(Path(status_path)) if status_path else None
        ledger, allowed = [], set()
        try:
            for line in (self.area / "budget" / "ledger.jsonl").read_text().split("\n"):
                if line.strip():
                    entry = json.loads(line)
                    ledger.append(entry)
                    if entry.get("exit") == 0:
                        allowed.add((entry.get("row"), entry.get("attempt")))
        except (OSError, ValueError):
            pass
        unaccounted, attempts = {}, []
        for path in sorted((self.area / "budget" / "attempts").glob("*.json")):
            record = _json(path)
            if isinstance(record, dict):
                attempts.append(record)
            if isinstance(record, dict) and record.get("launched") and (record.get("row"), record.get("attempt")) not in allowed:
                unaccounted.setdefault(record.get("block"), []).append("%s attempt %s" % (record.get("row"), record.get("attempt")))
        out = {}
        for block in ("prevention", "bridges"):
            stop = _json(self.area / "budget" / ("stop-%s.json" % block)) if (self.area / "budget" / ("stop-%s.json" % block)).exists() else None
            entry = (status or {}).get("blocks", {}).get(block) if isinstance(status, dict) else None
            label, why = None, []
            unverified = [r for r in attempts if r.get("block") == block and r.get("accounting") == COMPLIANCE_UNAVAILABLE]
            if stop is not None or (self.area / "budget" / ("stop-%s.json" % block)).exists():
                # round-4 ruling G1: a stop for workers not verified terminated is missing accounting evidence, not an overrun
                label = COMPLIANCE_UNAVAILABLE if ((stop or {}).get("accounting") == "unavailable" or unverified) else BUDGET_INCOMPLETE
                why.append("%s: the block stopped at %s by %s: %s" % (label, (stop or {}).get("time"), (stop or {}).get("row"),
                                                                     (stop or {}).get("reason", "an unreadable stop record")))
            elif unverified:
                label = COMPLIANCE_UNAVAILABLE
                why.append("%s: attempts whose termination was not verified: %s" % (COMPLIANCE_UNAVAILABLE, ", ".join(
                    "%s attempt %s" % (r.get("row"), r.get("attempt")) for r in unverified[:5])))
            # round-5 ruling H2: spending-limit evidence in an attempt record (ended_by limit, a watchdog that found the job
            # alive at limit_until, an end at or after it) makes the block budget-incomplete, whatever its class and
            # whether or not the stop file was written
            limited = [(r.get("row"), r.get("attempt"), budget_tool.spending_limit_evidence(self.work, r)) for r in attempts
                       if r.get("block") == block]
            limited = [x for x in limited if x[2]]
            if limited and label != COMPLIANCE_UNAVAILABLE:
                label = BUDGET_INCOMPLETE
                why.append("%s: spending-limit termination recorded: %s" % (BUDGET_INCOMPLETE, "; ".join(
                    "%s attempt %s (%s)" % x for x in limited[:3])))
            if isinstance(entry, dict) and entry.get("over_ceiling") and label != COMPLIANCE_UNAVAILABLE:
                label = BUDGET_INCOMPLETE
                why.append("%s: the budget status shows spending over the ceiling" % BUDGET_INCOMPLETE)
            if label is None and (status is None or not isinstance(entry, dict) or entry.get("accounting") != "available"):
                label = COMPLIANCE_UNAVAILABLE
                why.append("%s: %s" % (COMPLIANCE_UNAVAILABLE, "no budget status was given (--budget-status)" if status is None else
                                       (entry or {}).get("why") or "the budget status does not account for the block"))
            if label is None and unaccounted.get(block):
                label = COMPLIANCE_UNAVAILABLE
                why.append("%s: launched attempts with no admitting ledger line: %s" % (COMPLIANCE_UNAVAILABLE, ", ".join(unaccounted[block][:5])))
            import importlib.util
            spec = importlib.util.spec_from_file_location('report_contain', HERE / 'p4_contain.py')
            contain = importlib.util.module_from_spec(spec); spec.loader.exec_module(contain)
            gpu_attempts = [r for r in attempts if r.get('block') == block and r.get('launched')]
            containment_errors = [{'row': r.get('row'), 'attempt': r.get('attempt'), 'problems': contain.scheduler_problems(self.work, r)}
                                  for r in gpu_attempts]
            containment_errors = [r for r in containment_errors if r['problems']]
            for r in attempts:
                if r.get('block') == block and r.get('launched') is not True and not (
                        r.get('launched') is False and r.get('state') in ('refused', 'reconciled', 'busy')):
                    containment_errors.append({'row': r.get('row'), 'attempt': r.get('attempt'),
                                               'problems': ['attempt has no conclusive launch/no-launch evidence']})
            if self.campaign:
                recorded = {(r.get('row'), r.get('attempt')) for r in gpu_attempts}
                for row in self.campaign['rows']:
                    env = row.get('env') or {}
                    if env.get('P4_BLOCK') != block or env.get('P4_KIND') not in ('training','qualification','scoring','direct','selftest'):
                        continue
                    for start in (self.work/'campaign'/self.campaign['name']/row['id']).glob('attempt-*/start.json'):
                        doc = _json(start) or {}
                        k = int(start.parent.name.split('-')[1])
                        # A recorded wrapper refusal/reconcile establishes that no GPU launch occurred.
                        wrapper = _json(self.area/'budget/attempts'/('%s-a%d.json' % (row['id'], k)))
                        if wrapper is None or (wrapper.get('launched') is not True and
                                               not (wrapper.get('launched') is False and wrapper.get('state') in ('refused', 'reconciled', 'busy'))):
                            containment_errors.append({'row': row['id'], 'attempt': k, 'problems': ['GPU attempt has no containment record']})
            verified_slurm = bool(gpu_attempts) and not containment_errors
            if not verified_slurm:
                label = COMPLIANCE_UNAVAILABLE
                why += [PROCESS_GROUP_ONLY, contain.DESCRIPTIVE]
            out[block] = {'containment': 'slurm-step' if verified_slurm else PROCESS_GROUP_ONLY,
                          'budget_compliance': 'AVAILABLE' if verified_slurm and label is None else 'UNAVAILABLE',
                          'containment_problems': containment_errors, "label": label, "why": why, "stop": stop, "spent": (entry or {}).get("spent") if isinstance(entry, dict) else None,
                          "refusals": [{"row": e.get("row"), "attempt": e.get("attempt"), "reason": e.get("reason")} for e in ledger
                                       if e.get("block") == block and e.get("exit") == 1]}
        return out

    # ---------------------------------------------------------------------------------------------------- contrasts
    def contrast(self, cid: str, block: str, quantity: str, builder, roles: dict, task: str, kind: str, margin=0,
                 nomination_rule: str = "", nomination_result=None) -> dict:
        """One registered contrast: its manifest entry (coefficients, support, alpha, bounds, the decision inequality in
        words, the interval result) and its finite-panel nomination rule and result. Bounds come from kit/p4_intervals."""
        samples_template = builder(*[{"_": 0} for _ in roles], task=task)
        coefficients = samples_template[0]["coefficients"]
        names = list(samples_template[0]["verdicts"])           # the builder's own role names, in argument order
        mapping = dict(zip(names, roles.values()))
        entry = {"id": cid, "block": block, "quantity": quantity, "task": task,
                 "checkpoints": {mapping[name]: str(coefficients[name]) for name in names},
                 "roles": {name: mapping[name] for name in names},
                 "support": [str(z) for z in iv.support(coefficients)], "alpha": ALPHA,
                 "interval_rule": RULE_WORDS[kind], "margin": str(MARGIN) if kind == "non_inferiority" else None,
                 "nomination_rule": nomination_rule, "nomination_result": nomination_result,
                 "available": False, "why": None, "estimate_exact": None, "lower": None, "upper": None, "interval_result": None}
        recs, whys = self.gather([(key, task) for key in roles.values()])
        if whys:
            entry["why"] = "unavailable: %s" % "; ".join(whys)
        else:
            verdicts = [recs[(key, task)]["verdicts"] for key in roles.values()]
            expected = self.bed(BASE, task)["verdicts"]
            try:
                samples = builder(*verdicts, task=task, expected_ids=list(expected) if expected else None)
                bounds = iv.contrast_bounds(samples, alpha=ALPHA)
                entry.update({"available": True, "estimate_exact": bounds["estimate_exact"], "estimate": bounds["estimate"],
                              "lower": bounds["lower"], "upper": bounds["upper"], "eta": bounds["eta"], "cells": bounds["cells"],
                              "n": bounds["samples"][0]["n"], "counts": bounds["samples"][0]["counts"],
                              "interval_result": iv.decide(bounds, kind, margin)})
            except (iv.IntervalInputError, ArithmeticError, ValueError) as exc:
                entry["why"] = "unavailable: %s" % exc
        self.contrasts.append(entry)
        return entry


def raw_signature(scored: dict) -> str:
    """What every scoring of a block shares in its raw result (round-4 ruling G4): the engine, generation and decoding
    blocks, the new-token limit, the result schema and mode."""
    return json.dumps({**{k: scored.get(k) for k in ("engine", "generation", "decoding", "max_new_tokens")},
                       "schema": scored.get("schema"), "mode": scored.get("mode")}, sort_keys=True)


def _delta(first, second, task, expected_ids=None):
    return iv.delta(first, second, task, expected_ids)


def _q(stage1, control, intervention, task, expected_ids=None):
    return iv.q_retention(stage1, control, intervention, task, expected_ids)


def _hg_mb(a, b, c, task, expected_ids=None):
    return iv.half_gap_minibatch(a, b, c, task, expected_ids)


def _hg_lr(a, b, c, task, expected_ids=None):
    return iv.half_gap_learning_rate(a, b, c, task, expected_ids)


# ------------------------------------------------------------------------------------------------------- prevention
def gains(recs: dict, key: str, base_key: str, stage1_key: str, task: str):
    """(gain over the untrained model, gain over the stage-1 checkpoint, threshold ceil(0.05 n), both >= threshold) or None."""
    try:
        x, b, s = recs[(key, task)], recs[(base_key, task)], recs[(stage1_key, task)]
    except KeyError:
        return None
    n = x["n"]
    t = ceil_share(n)
    over_base, over_stage1 = x["strict_serving"] - b["strict_serving"], x["strict_serving"] - s["strict_serving"]
    return {"over_base": over_base, "over_stage1": over_stage1, "threshold": t, "n": n,
            "holds": over_base >= t and over_stage1 >= t}


def prevention(tree: Tree, budget: dict | None = None) -> dict:
    """Registration 3.2 as amended by A1 and amendment 3, with the interval-gated results of 3.2 and 3.3 beside each pair."""
    A, B = tree.A, tree.B
    runs = ["p4-%s-%s" % (lin, k) for lin in LINEAGES for k in ("ctl1", "gate1", "ctl2", "gate2")]
    keys = [BASE] + [tree.stage1[lin] for lin in LINEAGES] + runs
    needed = [(k, t) for k in keys for t in (A, B) if not (k == BASE and t == A)]
    block_problems, checks, machine_ids, decodings = tree.block_identity(needed, PREVENTION_PREFIX)
    incomplete = tree.completeness("prevention", keys + [tree.pilot_stage2[lin] for lin in LINEAGES])
    budget = (budget or {}).get("prevention") or {"label": None, "why": []}
    lineages, pairs = {}, []
    for lin in LINEAGES:
        s1 = tree.stage1[lin]
        controls = ["p4-%s-ctl1" % lin, "p4-%s-ctl2" % lin]
        gates = ["p4-%s-gate1" % lin, "p4-%s-gate2" % lin]
        cond1, cond2 = {}, {}
        for c in controls:
            recs, whys = tree.gather([(s1, A), (c, A)])
            if whys:
                cond1[c] = {"holds": None, "why": "; ".join(whys)}
            else:
                n = recs[(s1, A)]["n"]
                F = recs[(s1, A)]["strict_serving"] - recs[(c, A)]["strict_serving"]
                cond1[c] = {"holds": 100 * F >= LOSS_PER_100 * n, "F": F, "n": n, "per_100": _fr(Fraction(100 * F, n)),
                            "rule": "100 * F_control >= 5 * n_A"}
            recs, whys = tree.gather([(BASE, B), (s1, B), (c, B)])
            if whys:
                cond2[c] = {"holds": None, "why": "; ".join(whys)}
            else:
                g = gains(recs, c, BASE, s1, B)
                cond2[c] = {**g, "rule": "both gains on B >= ceil(0.05 n_B)"}
        values = [cond1[c]["holds"] for c in controls]
        lineage_cond1 = False if False in values else (True if all(v is True for v in values) else None)
        lineages[lin] = {"stage1": s1, "pilot_stage2": tree.pilot_stage2[lin], "controls": controls, "interventions": gates,
                         "condition_1": {"holds": lineage_cond1, "per_control": cond1,
                                         "label": LOSS_NOT_REPRODUCED if lineage_cond1 is False else None},
                         "condition_2": cond2}
        for c in controls:
            for i in gates:
                pairs.append(prevention_pair(tree, lin, s1, c, i, cond1[c]["holds"], cond1, cond2, block_problems))
    technical = any(p["technical"] for p in pairs)
    if budget["label"]:
        screen, why = budget["label"], list(budget["why"])
        for p in pairs:                                     # a stopped block receives no nomination, whatever its scores
            p["status_without_budget"] = p["status"]
            p["status"] = budget["label"]
            if budget.get('containment') == PROCESS_GROUP_ONLY:
                p['interval']['label'] = 'descriptive bounds only'
    elif block_problems:
        screen, why = NOT_EVALUATED, block_problems
    elif technical or incomplete:
        screen, why = TECHNICALLY_INCOMPLETE, (["%s vs %s: %s" % (p["control"], p["intervention"], "; ".join(p["reasons"]))
                                                for p in pairs if p["technical"]] + incomplete)
    elif all(p["status"] == PAIR_MEETS for p in pairs):
        screen, why = SCREEN_YES, []
    else:
        screen, why = SCREEN_NO, ["%s vs %s: %s" % (p["control"], p["intervention"], p["status"]) for p in pairs if p["status"] != PAIR_MEETS]
    labels = [p["interval"]["label"] for p in pairs]
    if budget.get('containment') == PROCESS_GROUP_ONLY:
        interval_screen = COMPLIANCE_UNAVAILABLE
    elif block_problems or technical or NOT_EVALUATED in labels:
        interval_screen = NOT_EVALUATED if (block_problems or technical) else "not positive"
    else:
        interval_screen = "positive: all eight pairs labelled prevents" if all(l == "prevents" for l in labels) else "not positive"
    return {"block": "prevention", "screen": screen, "screen_why": why, "block_problems": block_problems,
            "completeness_problems": incomplete, "budget": budget,
            "prefix_checks": checks, "machine_ids": machine_ids, "scoring_signatures": decodings,
            "lineages": lineages, "pairs": pairs, "pairs_meeting": sum(p["status"] == PAIR_MEETS for p in pairs),
            "interval_gated": {"heading": INTERVAL_HEADING, "conjunction": interval_screen}}


def prevention_pair(tree: Tree, lin: str, s1: str, c: str, i: str, lineage_cond1, cond1: dict, cond2: dict,
                    block_problems: list) -> dict:
    A, B = tree.A, tree.B
    needed = [(s1, A), (c, A), (i, A), (BASE, B), (s1, B), (c, B), (i, B)]
    _all_recs, whys = tree.gather(needed)
    pid = "prevention/%s/%s-vs-%s" % (lin, c, i)
    conditions, exact = {}, {}
    # A1.1: Q >= 0 <=> 2 * F_i <= F_c (counts over the same n_A)
    recs, q_whys = tree.gather([(s1, A), (c, A), (i, A)])
    if not q_whys:
        n_A = recs[(s1, A)]["n"]
        F_c = recs[(s1, A)]["strict_serving"] - recs[(c, A)]["strict_serving"]
        F_i = recs[(s1, A)]["strict_serving"] - recs[(i, A)]["strict_serving"]
        conditions["Q_at_least_zero"] = 2 * F_i <= F_c
        exact.update({"n_A": n_A, "F_control": F_c, "F_intervention": F_i, "Q": _fr(Fraction(F_c, 2 * n_A) - Fraction(F_i, n_A))})
    else:
        conditions["Q_at_least_zero"] = None
    recs, g_whys = tree.gather([(BASE, B), (s1, B), (i, B)])
    g = gains(recs, i, BASE, s1, B) if not g_whys else None
    conditions["intervention_acquired_B"] = g["holds"] if g else None
    if g:
        exact["intervention_gains_B"] = g
    recs, d_whys = tree.gather([(c, B), (i, B)])
    if not d_whys:
        n_B = recs[(c, B)]["n"]
        d = recs[(i, B)]["strict_serving"] - recs[(c, B)]["strict_serving"]
        conditions["B_difference_above_minus_0.025"] = 40 * d > -n_B
        exact.update({"n_B": n_B, "B_difference_questions": d, "B_difference": _fr(Fraction(d, n_B))})
    else:
        conditions["B_difference_above_minus_0.025"] = None
    q = tree.contrast(pid + "/Q", "prevention", "Q = 1/2 F_control - F_intervention = S_intervention(A) - 1/2 S_stage1(A) - 1/2 S_control(A)",
                      _q, {"stage1": s1, "control": c, "intervention": i}, A, "lower_at_least_zero",
                      nomination_rule="2 * F_intervention <= F_control in questions (exact Q >= 0)",
                      nomination_result=conditions["Q_at_least_zero"])
    dB = tree.contrast(pid + "/delta_B", "prevention", "Delta_B = S_intervention(B) - S_control(B)", _delta,
                       {"first": i, "second": c}, B, "non_inferiority", margin=MARGIN,
                       nomination_rule="40 * (S_intervention(B) - S_control(B)) > -n_B in questions (exact difference > -0.025)",
                       nomination_result=conditions["B_difference_above_minus_0.025"])
    reasons, technical = [], False
    if block_problems:
        reasons.append("block not evaluated: " + "; ".join(block_problems))
    if whys:
        reasons.extend(whys)
        technical = True
    elif lineage_cond1 is False:                        # the pair's OWN control (amendment 3 B8; amendment 2 A1)
        reasons.append("%s (control %s: F %s of %s)" % (LOSS_NOT_REPRODUCED, c, cond1[c].get("F"), cond1[c].get("n")))
    elif lineage_cond1 is None:
        reasons.append("condition 1 unavailable: %s" % cond1[c].get("why"))
        technical = True
    elif cond2[c]["holds"] is False:
        reasons.append(CONTROL_NOT_LEARNED)
    elif cond2[c]["holds"] is None:
        reasons.append("condition 2 unavailable: %s" % cond2[c].get("why"))
        technical = True
    if reasons:
        status = NOT_EVALUATED
    elif all(v is True for v in conditions.values()):
        status = PAIR_MEETS
    else:
        status = PAIR_FAILS
    # the original interval-gated label of registration 3.3, reported beside the nomination
    if status == NOT_EVALUATED or not q["available"] or not dB["available"]:
        label = NOT_EVALUATED
    elif q["interval_result"] == "pass" and conditions["intervention_acquired_B"] and dB["interval_result"] == "pass":
        label = "prevents"
    elif q["interval_result"] == "pass" and dB["interval_result"] == "fail":
        label = "retention with a learning deficit"
    elif q["interval_result"] == "fail":
        label = "does not meet the 50 percent reduction bar"
    else:
        label = "inconclusive"
    acquisition_words = (None if conditions["intervention_acquired_B"] is None
                         else ("held" if conditions["intervention_acquired_B"] else ACQUISITION_NOT_DEMONSTRATED))
    return {"id": pid, "lineage": lin, "control": c, "intervention": i, "seed_control": tree.roles[c]["seed"],
            "seed_intervention": tree.roles[i]["seed"], "status": status, "reasons": reasons, "technical": technical,
            "prerequisites": {"loss_reproduced": lineage_cond1, "control_learned_B": cond2[c]["holds"]},
            "conditions": conditions, "acquisition": acquisition_words, "exact": exact,
            "interval": {"heading": INTERVAL_HEADING, "label": label,
                         "Q": {"estimate": q["estimate_exact"], "lower": q["lower"], "upper": q["upper"],
                               "decision": q["interval_result"], "why": q["why"]},
                         "delta_B": {"estimate": dB["estimate_exact"], "lower": dB["lower"], "upper": dB["upper"],
                                     "decision": dB["interval_result"], "why": dB["why"]},
                         "acquisition": acquisition_words}}


# ---------------------------------------------------------------------------------------------------------- bridges
def _direction(values: list, t: int, words: tuple):
    if any(v is None for v in values):
        return None
    if all(v >= t for v in values):
        return words[0]
    if all(v <= -t for v in values):
        return words[1]
    return NO_CONSISTENT


def _interval_direction(results: list, words: tuple):
    if any(r is None for r in results):
        return None
    if all(r == "pass" for r in results):
        return words[0]
    if all(r == "fail" for r in results):
        return words[1]
    return INTERVAL_NO_DIFFERENCE


def _all(values: list):
    if any(v is False for v in values):
        return False
    return True if all(v is True for v in values) else None


def argv_diff(tree: Tree, first: str, second: str):
    """The mechanical difference of two runs' argv.txt lines (supplement S2 and S6), or None when either is missing."""
    lines = []
    for key in (first, second):
        record = tree.raw_run(key)["record"] if key in tree.roles else None
        if record is None:
            return None
        try:
            lines.append([l.strip() for l in (record["folder"] / "env" / "argv.txt").read_text(encoding="utf-8").split("\n") if l.strip()])
        except OSError:
            return None
    return {"only_in_%s" % first: sorted(set(lines[0]) - set(lines[1])), "only_in_%s" % second: sorted(set(lines[1]) - set(lines[0]))}


def algorithm_step_report(tree: Tree, c_key: str, s_key: str) -> dict:
    """Supplement S2 / amendment 3 B6: every difference of command and resolved configuration between c and s, archived
    as a REPORT (the algorithm's own differences, no equality test), and the shared settings S2 lists, verified."""
    out = {"c": c_key, "s": s_key}
    records = [tree.raw_run(k)["record"] if k in tree.roles else None for k in (c_key, s_key)]
    if any(r is None for r in records):
        out["why"] = "a run of record is missing"
        return out
    argv = [recipe.read_lines(r["folder"] / "env" / "argv.txt") or [] for r in records]
    resolved = [recipe.read_resolved(r["folder"] / "env" / "resolved-config.yaml") for r in records]
    out["command_differences"] = recipe.difference_report(recipe.parse_argv(argv[0])[1], recipe.parse_argv(argv[1])[1])
    out["resolved_differences"] = recipe.difference_report(resolved[0] or {}, resolved[1] or {}) if all(resolved) else None
    out["shared_settings"] = recipe.shared_settings(argv[0], argv[1], resolved[0], resolved[1])
    return out


def bridges(tree: Tree, budget: dict | None = None) -> dict:
    """Supplement S2 to S5 as amended by A2 and amendment 3, with the S4 interval-gated results beside each nomination."""
    A, B = tree.A, tree.B
    s1 = tree.stage1["r1"]
    run_keys = {cfg: {seed: tree.bridge_keys[(cfg, seed)] for seed in SEEDS} for cfg in ("a", "b", "c", "s")}
    all_runs = [run_keys[cfg][seed] for cfg in ("a", "b", "c", "s") for seed in SEEDS]
    needed = [(BASE, B)] + [(k, t) for k in [s1] + all_runs for t in (A, B)]
    problems, checks, machine_ids, decodings = tree.block_identity(needed, BRIDGE_PREFIX)
    budget = (budget or {}).get("bridges") or {"label": None, "why": []}
    problems += tree.completeness("bridges", [BASE, s1] + all_runs)
    algorithm_step = {str(seed): algorithm_step_report(tree, run_keys["c"][seed], run_keys["s"][seed]) for seed in SEEDS}
    for seed, entry in algorithm_step.items():
        shared = entry.get("shared_settings")
        if shared is not None and not shared["ok"]:
            problems.append("c and s (seed %s) differ in a setting supplement S2 holds equal: %s" % (seed, shared["unequal"]))
    for key in all_runs:
        why = tree.run_ok(key)
        if why:
            problems.append(why)
    for key, task in needed:
        rec = tree.bed(key, task)
        if not rec["qualified"]:
            problems.append("scoring %s on %s not qualified: %s" % (key, task, "; ".join(rec["problems"])))
    problems = list(budget["why"]) + problems
    complete = not problems
    # acquisition of B per configuration (both runs, both gains)
    acquisition = {}
    for cfg in ("a", "b", "c", "s"):
        per_seed = {}
        for seed in SEEDS:
            key = run_keys[cfg][seed]
            recs, whys = tree.gather([(BASE, B), (s1, B), (key, B)])
            per_seed[str(seed)] = gains(recs, key, BASE, s1, B) if not whys else {"holds": None, "why": "; ".join(whys)}
        acquisition[cfg] = {"runs": {str(s): run_keys[cfg][s] for s in SEEDS}, "per_seed": per_seed,
                            "acquired": _all([v["holds"] for v in per_seed.values()])}
    definitions = {"minibatch": ("a", "b"), "learning_rate": ("b", "c"), "algorithm": ("s", "c"), "reference_gap": ("a", "c")}
    preservation_of = {"minibatch": ("a", "b"), "learning_rate": ("b", "c"), "algorithm": ("s", "c")}
    base_A, base_B = tree.bed(BASE, A), tree.bed(BASE, B)
    n_A = base_A["n"] if base_A["qualified"] else None
    n_B = base_B["n"] if base_B["qualified"] else None
    t_A = ceil_share(n_A) if n_A else None
    seeds = {}
    for seed in SEEDS:
        k = {cfg: run_keys[cfg][seed] for cfg in run_keys}
        entry = {"runs": k, "n_A": n_A, "n_B": n_B, "threshold_A": t_A, "contrasts": {}, "half_gaps": {}, "preservation": {}}

        def counts(cfgs, task):
            recs, whys = tree.gather([(k[cfg], task) for cfg in cfgs])
            return (None, whys) if whys else ({cfg: recs[(k[cfg], task)]["strict_serving"] for cfg in cfgs}, whys)

        for name, (x, y) in definitions.items():
            S, whys = counts((x, y), A)
            value = S[x] - S[y] if S else None
            c = tree.contrast("bridges/s%d/%s" % (seed, name), "bridges", "%s, seed %d" % (STEP_NAMES[name], seed), _delta,
                              {"first": k[x], "second": k[y]}, A, "lower_above_zero",
                              nomination_rule="signed count >= ceil(0.05 n_A) (or <= -ceil(0.05 n_A)) in both seeds",
                              nomination_result=value)
            entry["contrasts"][name] = {"questions": value, "exact": _fr(Fraction(value, n_A)) if value is not None and n_A else None,
                                        "lower": c["lower"], "upper": c["upper"], "interval_result": c["interval_result"], "why": c["why"]}
        for name, builder, sign in (("minibatch", _hg_mb, (1, -2, 1)), ("learning_rate", _hg_lr, (-1, 2, -1))):
            S, whys = counts(("a", "b", "c"), A)
            twice = sign[0] * S["a"] + sign[1] * S["b"] + sign[2] * S["c"] if S else None
            c = tree.contrast("bridges/s%d/half_gap_%s" % (seed, name), "bridges",
                              ("D_ab - G/2 = 1/2 S_a(A) - S_b(A) + 1/2 S_c(A)" if name == "minibatch" else
                               "D_bc - G/2 = -1/2 S_a(A) + S_b(A) - 1/2 S_c(A)") + ", seed %d" % seed,
                              builder, {"a": k["a"], "b": k["b"], "c": k["c"]}, A, "lower_at_least_zero",
                              nomination_rule="exact half-gap >= 0 (at least half) or < 0 (less than half) in both seeds",
                              nomination_result=twice)
            entry["half_gaps"][name] = {"twice_n_times_value": twice,
                                        "exact": _fr(Fraction(twice, 2 * n_A)) if twice is not None and n_A else None,
                                        "lower": c["lower"], "upper": c["upper"], "interval_result": c["interval_result"], "why": c["why"]}
        for name, (x, y) in preservation_of.items():
            S, whys = counts((x, y), B)
            d = S[x] - S[y] if S else None
            holds = (40 * d > -n_B) if d is not None and n_B else None
            c = tree.contrast("bridges/s%d/preservation_%s" % (seed, name), "bridges",
                              "S_%s(B) - S_%s(B), seed %d (B-preservation of the %s step)" % (x, y, seed, name.replace("_", "-")),
                              _delta, {"first": k[x], "second": k[y]}, B, "non_inferiority", margin=MARGIN,
                              nomination_rule="40 * (S_%s(B) - S_%s(B)) > -n_B in questions (exact difference > -0.025)" % (x, y),
                              nomination_result=holds)
            entry["preservation"][name] = {"questions": d, "exact": _fr(Fraction(d, n_B)) if d is not None and n_B else None,
                                           "holds": holds, "lower": c["lower"], "upper": c["upper"],
                                           "interval_result": c["interval_result"], "why": c["why"]}
        seeds[str(seed)] = entry
    per = [seeds[str(s)] for s in SEEDS]
    nominations, interval = {}, {}
    if complete and t_A is not None:
        for name in definitions:
            nominations[name] = _direction([e["contrasts"][name]["questions"] for e in per], t_A, NOMINATION_WORDS[name])
            interval[name] = _interval_direction([e["contrasts"][name]["interval_result"] for e in per], INTERVAL_WORDS[name])
        preservation = {name: _all([e["preservation"][name]["holds"] for e in per]) for name in preservation_of}
        interval_preservation = {name: (True if all(e["preservation"][name]["interval_result"] == "pass" for e in per) else
                                        False if all(e["preservation"][name]["interval_result"] == "fail" for e in per) else None)
                                 for name in preservation_of}
        acquired_abc = _all([acquisition[cfg]["acquired"] for cfg in ("a", "b", "c")])
        attribution, interval_attribution = {}, {}
        for step in ("minibatch", "learning_rate"):
            reasons = []
            if nominations["reference_gap"] != NOMINATION_WORDS["reference_gap"][0]:
                reasons.append("the reference gap is not observed (%s): there is nothing to attribute" % nominations["reference_gap"])
            if acquired_abc is not True:
                reasons.append("configurations a, b and c did not all acquire B")
            if preservation[step] is not True:
                reasons.append("the %s step's B-preservation prerequisite does not hold in both seeds" % step.replace("_", "-"))
            values = [e["half_gaps"][step]["twice_n_times_value"] for e in per]
            if reasons:
                label = NO_ATTRIBUTION
            elif all(v >= 0 for v in values):
                label = AT_LEAST_HALF
            elif all(v < 0 for v in values):
                label = LESS_THAN_HALF
            else:
                label, reasons = NO_ATTRIBUTION, ["the half-gap contrast differs in sign between the seeds"]
            attribution[step] = {"label": label, "reasons": reasons}
            ireasons = []
            if interval["reference_gap"] != INTERVAL_WORDS["reference_gap"][0]:
                ireasons.append("reference gap not established")
            if acquired_abc is not True:
                ireasons.append("configurations a, b and c did not all acquire B")
            if interval_preservation[step] is not True:
                ireasons.append("the lower bound of the step's B-preservation contrast does not exceed -0.025 in both seeds")
            results = [e["half_gaps"][step]["interval_result"] for e in per]
            if ireasons:
                ilabel = NO_ATTRIBUTION
            elif all(r == "pass" for r in results):
                ilabel = "carries at least half of the reference gap"
            elif all(r == "fail" for r in results):
                ilabel = "carries less than half of the reference gap"
            else:
                ilabel = NO_ATTRIBUTION
            interval_attribution[step] = {"label": ilabel, "reasons": ireasons}
        both_zero = (all(e["half_gaps"][s]["twice_n_times_value"] == 0 for e in per for s in ("minibatch", "learning_rate"))
                     and all(attribution[s]["label"] == AT_LEAST_HALF for s in ("minibatch", "learning_rate")))
        combined_conditions = {"algorithm_nomination": nominations["algorithm"] == NOMINATION_WORDS["algorithm"][0],
                               "c_acquired_B": acquisition["c"]["acquired"], "s_acquired_B": acquisition["s"]["acquired"],
                               "B_preservation_algorithm_step": preservation["algorithm"]}
        combined = COMBINED if all(v is True for v in combined_conditions.values()) else None
        contrast5 = (True if all(e["preservation"]["algorithm"]["interval_result"] == "pass" for e in per) else
                     False if all(e["preservation"]["algorithm"]["interval_result"] == "fail" for e in per) else None)
        contrast5_label = {True: "not inferior on B", False: "inferior on B", None: "inconclusive"}[contrast5]
        interval_combined_conditions = {"contrast_3": interval["algorithm"] == INTERVAL_WORDS["algorithm"][0],
                                        "contrast_5": contrast5_label == "not inferior on B",
                                        "c_acquired_B": acquisition["c"]["acquired"], "s_acquired_B": acquisition["s"]["acquired"]}
        interval_combined = INTERVAL_COMBINED if all(v is True for v in interval_combined_conditions.values()) else None
        status = "complete"
        registered = {"nominations": {name: {"label": nominations[name], "contrast": STEP_NAMES[name]} for name in definitions},
                      "acquired_B": {cfg: acquisition[cfg]["acquired"] for cfg in acquisition},
                      "B_preservation": preservation, "attribution": attribution,
                      "attribution_note": ATTRIBUTION_NOTE, "both_zero_sentence": BOTH_ZERO if both_zero else None,
                      "combined_conditions": combined_conditions, "combined_sentence": combined}
        interval_block = {"heading": INTERVAL_HEADING, "contrasts": interval, "contrast_5": contrast5_label,
                          "B_preservation": interval_preservation, "attribution": interval_attribution,
                          "combined_conditions": interval_combined_conditions, "combined_sentence": interval_combined}
    else:
        status = budget["label"] or BRIDGE_INCOMPLETE       # a stopped block is budget-incomplete, with no registered label
        registered = None
        interval_block = {"heading": INTERVAL_HEADING, "note": "no interval-gated label: the block is incomplete; the bounds "
                                                               "per seed are printed descriptively"}
    return {"block": "bridges", "status": status, "problems": problems, "budget": budget, "recipe": tree.recipe,
            "reused_configuration": tree.reused,
            "reused_runs": {str(s): run_keys[tree.reused][s] for s in SEEDS}, "runs": run_keys, "incoming": s1,
            "prefix_checks": checks, "machine_ids": machine_ids, "scoring_signatures": decodings,
            "acquisition": acquisition, "per_seed": seeds, "registered": registered, "interval_gated": interval_block,
            "algorithm_step_argv_diff": {str(s): argv_diff(tree, run_keys["c"][s], run_keys["s"][s]) for s in SEEDS},
            "algorithm_step_report": algorithm_step}


# --------------------------------------------------------------------------------------------- reported with no bar
def _tokens_per_correct(rec: dict, budget: int):
    """Exact tokens per strict-correct answer at `budget`, "inf" when nothing is correct, None when unread."""
    entry = (rec.get("per_budget") or {}).get(str(budget)) or {}
    tokens, correct = entry.get("tokens_total"), entry.get("correct_strict")
    if not _int(tokens) or not _int(correct):
        return None
    return "inf" if correct == 0 else Fraction(tokens, correct)


def _ratio(x, y):
    """x / y for tokens per correct (Fraction or "inf"); None when not defined (both infinite or either unread)."""
    if x is None or y is None:
        return None
    if x == "inf" and y == "inf":
        return None
    if x == "inf":
        return "inf"
    if y == "inf":
        return Fraction(0)
    return x / y if y else None


def _show(value):
    if value is None or value == "inf":
        return value
    return _decimal(value, 4)


def no_bar(tree: Tree) -> list:
    """Registration section 4 for every evaluated checkpoint where the files exist."""
    A, B = tree.A, tree.B
    rows = []
    keys = [BASE] + [tree.stage1[l] for l in LINEAGES] + [tree.pilot_stage2[l] for l in LINEAGES]
    keys += [k for k in tree.roles if k.startswith("p4-r")] + [k for k in tree.roles if k.startswith("p4-br-")]
    for key in keys:
        if key == BASE:
            incoming = None
        elif key in tree.stage1.values():
            incoming = BASE
        else:
            incoming = tree.stage1[tree.lineage_of[key]]
        role = tree.roles.get(key)
        matched = None
        if role and role["kind"] == "gate":
            matched = "p4-%s-ctl%d" % (role["lineage"], 1 if role["seed"] == 101 else 2)
        elif role and role["kind"] == "bridge":
            matched = tree.bridge_keys[(tree.reused, role["seed"])]
        row = {"key": key, "incoming": incoming, "matched_control": matched, "tasks": {}}
        for task in (A, B):
            rec = tree.bed(key, task)
            cell = {"found": rec["found"], "qualified": rec["qualified"], "path": rec["path"], "attempt": rec["attempt"]}
            if rec["readable"]:
                b, h = rec["per_budget"][str(tree.serving)], rec["per_budget"][str(tree.diagnostic)]
                tpc = _tokens_per_correct(rec, tree.serving)
                cell.update({"n": rec["n"],
                             "scores": {"strict_serving": b.get("correct_strict"), "canonical_serving": b.get("correct_canonical"),
                                        "strict_diagnostic": h.get("correct_strict"), "canonical_diagnostic": h.get("correct_canonical")},
                             "unfinished_share_diagnostic": _fr(Fraction(h["cut"], rec["n"])) if _int(h.get("cut")) else None,
                             "tokens_per_correct_serving": _show(tpc)})
                for label, other in (("incoming", incoming), ("matched_control", matched)):
                    if other is None:
                        continue
                    orec = tree.bed(other, task)
                    ratio = _ratio(tpc, _tokens_per_correct(orec, tree.serving)) if orec["readable"] else None
                    cell["ratio_to_" + label] = _show(ratio)
                    cell["at_cost_vs_" + label] = (None if ratio is None else (ratio == "inf" or ratio > AT_COST))
                if incoming is not None:
                    ref = tree.bed(incoming, task)
                    if rec["qualified"] and ref["qualified"] and rec["machine_id"] == ref["machine_id"]:
                        try:
                            record = budget_report.account(budget_report.read_sweep(ref["path"]), budget_report.read_sweep(rec["path"]),
                                                           tree.serving, tree.diagnostic, key)
                            cell["signed_terms_vs_incoming"] = {t: record[t] for t in ("F", "residual", "budget", "extraction")}
                        except (budget_report.BudgetReportError, AssertionError, KeyError, TypeError) as exc:
                            cell["signed_terms_vs_incoming"] = "not computed: %s" % exc
                    else:
                        cell["signed_terms_vs_incoming"] = "not computed: the two scorings are not both qualified on one machine"
            row["tasks"][task] = cell
        panel = tree.panel(key)
        row["panel"] = {k: panel.get(k) for k in ("found", "path", "attempt", "serving", "diagnostic", "note")}
        summary = None
        if key in tree.roles:
            record = tree.raw_run(key)["record"]
            summary = record["summary"] if record else None
        elif key in tree.pilot:
            summary, note = tree.pilot_summary(key)
            if note:
                row["training_note"] = note
        if key in tree.roles and tree.raw_run(key)["record"] is not None:
            name = "%s-a%d" % (key, tree.raw_run(key)["record"]["attempt"])
            stats = _json(tree.area / "report-rollouts" / name / "rollout-stats.json")
            if isinstance(stats, dict):                     # amendment 3 B8: before/after rewards, no-reward groups (no bar)
                row["rollouts"] = {"run": name, "gate": (stats.get("totals") or {}).get("gate"), "gate_protocol": stats.get("gate_protocol")}
        if isinstance(summary, dict):
            seconds, gpus = summary.get("seconds"), summary.get("n_gpus")
            ok = isinstance(seconds, (int, float)) and isinstance(gpus, (int, float)) and not isinstance(seconds, bool)
            row["training"] = {"seconds": seconds, "n_gpus": gpus,
                               "gpu_hours": round(seconds * gpus / 3600, 3) if ok and math.isfinite(seconds) else None}
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------------------ build
def read_selection(path: Path, serving: int, diagnostic: int) -> dict:
    selection = _json(path)
    if not isinstance(selection, dict) or selection.get("schema") != SELECTION_SCHEMA:
        raise ReportRefused("%s is not a package-4 selection (schema %s)" % (path, SELECTION_SCHEMA))
    if selection.get("entered") is not True:
        raise ReportRefused("the selection says package 4 was not entered (%s): there is nothing to report" % selection.get("reason"))
    content = {k: v for k, v in selection.items() if k not in ("content_sha256", "generated_at", "located_at")}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != selection.get("content_sha256"):
        raise ReportRefused("%s was altered after selection: its content_sha256 does not match its content" % path)
    constants = selection.get("constants") or {}
    if constants.get("serving") != SERVING or constants.get("diagnostic") != DIAGNOSTIC:
        raise ReportRefused("the selection's constants are not the registered 2048 and 8192")
    if (serving, diagnostic) != (SERVING, DIAGNOSTIC):
        raise ReportRefused("--serving %d and --diagnostic %d are not the registered %d and %d" % (serving, diagnostic, SERVING, DIAGNOSTIC))
    group, tasks, lineages = selection.get("group") or {}, selection.get("tasks") or {}, selection.get("lineages") or {}
    if group.get("recipe") not in RECIPE_CONFIG or tasks.get("A") not in BEDS or tasks.get("B") not in BEDS or tasks["A"] == tasks["B"]:
        raise ReportRefused("the selection names no registered group and tasks (%r, %r)" % (group, tasks))
    first = {"chemistry": "chem", "toolalpaca": "tool"}[tasks["A"]]
    for lin in LINEAGES:
        entry = lineages.get(lin) or {}
        for part, key in (("stage1", "%s-%s-%s" % (group["recipe"], first, lin)),
                          ("pilot_stage2", "%s-%s-%s" % (group["recipe"], group.get("chain"), lin))):
            if not isinstance(entry.get(part), dict) or entry[part].get("key") != key:
                raise ReportRefused("the selection's lineage %s has no %s record for %s" % (lin, part, key))
    return selection


def build(args) -> tuple:
    serving, diagnostic = int(args.serving), int(args.diagnostic)
    work = Path(args.work)
    if not work.is_dir():
        raise ReportRefused("no such work folder: %s" % work)
    blocks = [b.strip() for b in str(args.blocks).split(",") if b.strip()]
    if not blocks or any(b not in ("prevention", "bridges") for b in blocks) or len(set(blocks)) != len(blocks):
        raise ReportRefused("--blocks must be prevention, bridges or both, comma-separated (got %r)" % args.blocks)
    selection_path = Path(args.selection)
    # round-3 ruling F2: the frozen decision inputs first (on the partner's node and on an extracted archive alike)
    stale = frozen_inputs.verify(work, need=("selection", "reservation"), containment=False)
    if stale:
        raise ReportRefused(frozen_inputs.refusal(stale))
    selection = read_selection(selection_path, serving, diagnostic)
    # round-4 ruling G3: the recipe check's baseline seal and its link to the frozen document; corrupted or missing ->
    # descriptive reporting only (no registered label for either block)
    baseline_problems = frozen_inputs.verify_baseline(work)
    campaign = None
    if args.campaign:
        try:
            campaign = runner.load_campaign(Path(args.campaign))
        except (OSError, ValueError, ImportError) as exc:
            raise ReportRefused("the campaign file %s cannot be loaded: %s" % (args.campaign, exc)) from None
        wanted = "k8b-p4-%s-%s" % (selection["group"]["recipe"], selection["group"]["chain"])
        if campaign.get("name") != wanted:
            raise ReportRefused("the campaign %s is %r; the selection's group needs %s" % (args.campaign, campaign.get("name"), wanted))
    tree = Tree(work, selection, serving, diagnostic, campaign, baseline_problems=baseline_problems)
    budget = tree.budget(args.budget_status)
    prevention_block = prevention(tree, budget) if "prevention" in blocks else {"block": "prevention", "screen": NOT_RUN,
                                                                                "screen_why": ["not requested (--blocks)"]}
    bridge_block = bridges(tree, budget) if "bridges" in blocks else {"block": "bridges", "status": NOT_RUN, "problems": ["not requested (--blocks)"]}
    for block, result in (('prevention', prevention_block), ('bridges', bridge_block)):
        result['containment'] = budget[block]['containment']
    reported = no_bar(tree)
    tool = tree.bed(BASE, "toolalpaca")
    tool_n = tool["n"] if _int(tool.get("n")) else None
    note = ("On ToolAlpaca one question is %.2f points (100/%d)." % (100 / tool_n, tool_n)) if tool_n else \
        "On ToolAlpaca the size of one question in points is not known (no untrained-model sweep of ToolAlpaca)."
    scorings = [tree._bed[k] for k in sorted(tree._bed)]
    validation = {"scorings": [{k: r.get(k) for k in ("key", "bed", "found", "qualified", "problems", "path", "attempt",
                                                      "machine_id", "supplied_machine_id", "n")} for r in scorings],
                  "runs": [{k: v for k, v in tree.raw_run(key).items() if k != "record"} | _record_of(tree, key)
                           for key in sorted(tree.roles)]}
    report = {"schema": SCHEMA, "work": str(work.resolve()), "serving": serving, "diagnostic": diagnostic, "alpha": ALPHA,
              "margin": str(MARGIN), "blocks": blocks,
              "selection": {"path": str(selection_path.resolve()), "sha256": sha256_file(selection_path),
                            "content_sha256": selection["content_sha256"], "group": selection["group"], "tasks": selection["tasks"]},
              "campaign": {"name": (campaign or {}).get("name"), "sha256": (campaign or {}).get("_sha256")},
              "recipe_baseline": {"verified": not baseline_problems, "problems": baseline_problems},
              "budget": budget,
              "statements": {"fixed": FIXED_STATEMENT, "a3_caveat": A3_CAVEAT, "toolalpaca_note": note,
                             "amendment_3": dict(AMENDMENT_3)},
              "validation": validation, "prevention": prevention_block, "bridges": bridge_block, "reported_no_bar": reported}
    manifest = {"schema": MANIFEST_SCHEMA, "work": str(work.resolve()), "selection": report["selection"],
                "lineages": {lin: {"stage1": selection["lineages"][lin]["stage1"], "pilot_stage2": selection["lineages"][lin]["pilot_stage2"],
                                   "runs": sorted(k for k, r in tree.roles.items() if r["lineage"] == lin and r["kind"] != "bridge")}
                             for lin in LINEAGES},
                "bridge_runs": {"%s-s%d" % (cfg, seed): key for (cfg, seed), key in sorted(tree.bridge_keys.items())},
                "runs": validation["runs"],
                "scorings": [{k: r.get(k) for k in ("key", "bed", "path", "per_item_path", "attempt", "sweep_sha256",
                                                    "per_item_sha256", "machine_id", "supplied_machine_id", "qualified")} for r in scorings],
                "prefix_checks": [tree._prefix[k] for k in sorted(tree._prefix)],
                "contrasts": sorted(tree.contrasts, key=lambda c: c["id"])}
    return report, manifest


def _record_of(tree: Tree, key: str) -> dict:
    record = tree.raw_run(key)["record"]
    if record is None:
        return {"run_of_record": None}
    return {"run_of_record": {"attempt": record["attempt"], "run": str(record["folder"].resolve()),
                              "run_summary_sha256": sha256_file(record["folder"] / "run-summary.json"),
                              "argv_sha256": sha256_file(record["folder"] / "env" / "argv.txt")}}


# ----------------------------------------------------------------------------------------------------------- render
def _b(value) -> str:
    return "-" if value is None else ("%.4f" % value if isinstance(value, float) else str(value))


def render(report: dict) -> str:
    tasks = report["selection"]["tasks"]
    note = report["statements"]["toolalpaca_note"]
    lines = ["# Package 4 report", "",
             FIXED_STATEMENT, "",
             A3_CAVEAT, "",
             report["statements"]["amendment_3"]["E"] + ".", "", report["statements"]["amendment_3"]["rescoring"] + ".", "",
             "Selected group %s %s: task A %s, task B %s. Serving budget %d, diagnostic budget %d. Every score below is a "
             "strict-correct count at the serving budget unless named otherwise. Interval bounds use the one method of "
             "registration 3.1 at alpha %s per contrast (kit/p4_intervals.py)."
             % (report["selection"]["group"]["recipe"], report["selection"]["group"]["chain"], tasks["A"], tasks["B"],
                report["serving"], report["diagnostic"], report["alpha"]), ""]
    for block in report['blocks']:
        lines += ['%s containment: %s; budget compliance: %s.' % (block, report['budget'][block]['containment'], report['budget'][block]['budget_compliance']), '']
        if report['budget'][block]['containment'] == PROCESS_GROUP_ONLY:
            lines += ['Process groups, environment markers and GPU-process observations do not establish that every descendant has terminated; this run is reported descriptively.', '']
    bad = [s for s in report["validation"]["scorings"] if not s["qualified"]]
    lines += ["## Validation", ""]
    if bad:
        lines += ["Scorings not qualified (each problem names its file):", ""]
        lines += ["- %s on %s: %s" % (s["key"], s["bed"], "; ".join(s["problems"])) for s in bad]
    else:
        lines.append("Every scoring read is qualified.")
    runs = [r for r in report["validation"]["runs"] if not r.get("valid")]
    if runs:
        lines += ["", "Package-4 runs not evaluated:", ""]
        lines += ["- %s: %s" % (r["key"], "; ".join(r.get("problems") or [r.get("why") or "no run of record"])) for r in runs]
    lines += ["", "## Budget (amendment 3 B3)", ""]
    for block, entry in sorted((report.get("budget") or {}).items()):
        lines.append("- %s: %s%s" % (block, entry["label"] or "within its ledger (no stop, accounting available)",
                                     "" if not entry["why"] else " -- " + "; ".join(entry["why"])))
    lines.append("")
    p = report["prevention"]
    lines += ["## Prevention screen (registration 3, amendment A1)", ""]
    lines.append("Nomination screen: **%s**. %s" % (p["screen"], note))
    for why in p.get("screen_why") or []:
        lines.append("- %s" % why)
    if "pairs" in p:
        lines += ["", "| pair | status | loss reproduced | control learned B | Q >= 0 (2 F_i <= F_c) | intervention acquired B "
                  "| B difference > -0.025 | F_c | F_i | Q | B difference | reasons |",
                  "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|"]
        for q in p["pairs"]:
            e = q["exact"]
            lines.append("| %s vs %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                q["control"], q["intervention"], q["status"], _tri(q["prerequisites"]["loss_reproduced"]),
                _tri(q["prerequisites"]["control_learned_B"]), _tri(q["conditions"]["Q_at_least_zero"]),
                _tri(q["conditions"]["intervention_acquired_B"]), _tri(q["conditions"]["B_difference_above_minus_0.025"]),
                _b(e.get("F_control")), _b(e.get("F_intervention")), _b(e.get("Q")), _b(e.get("B_difference")),
                "; ".join(q["reasons"]) or "-"))
        lines += ["", "### " + INTERVAL_HEADING, "",
                  "Registration 3.3 conjunction: %s." % p["interval_gated"]["conjunction"], "",
                  "| pair | label | Q lower | Q upper | Q decision | Delta_B lower | Delta_B upper | Delta_B decision | acquisition |",
                  "|---|---|---:|---:|---|---:|---:|---|---|"]
        for q in p["pairs"]:
            i = q["interval"]
            lines.append("| %s vs %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                q["control"], q["intervention"], i["label"], _b(i["Q"]["lower"]), _b(i["Q"]["upper"]), _b(i["Q"]["decision"]),
                _b(i["delta_B"]["lower"]), _b(i["delta_B"]["upper"]), _b(i["delta_B"]["decision"]), _b(i["acquisition"])))
    lines.append("")
    br = report["bridges"]
    lines += ["## Bridges (supplement S2 to S5, amendment A2)", ""]
    lines.append("Block status: **%s**." % br["status"])
    for why in br.get("problems") or []:
        lines.append("- %s" % why)
    if "per_seed" in br:
        lines += ["", "Reused configuration %s (recipe %s): %s." % (br["reused_configuration"], br["recipe"],
                  ", ".join("seed %s %s" % kv for kv in sorted(br["reused_runs"].items()))), "",
                  "| seed | contrast | questions | exact | lower | upper |", "|---|---|---:|---:|---:|---:|"]
        for seed, e in sorted(br["per_seed"].items()):
            for name, c in e["contrasts"].items():
                lines.append("| %s | %s | %s | %s | %s | %s |" % (seed, STEP_NAMES[name], _b(c["questions"]), _b(c["exact"]), _b(c["lower"]), _b(c["upper"])))
            for name, c in e["half_gaps"].items():
                lines.append("| %s | half-gap, %s step (2 n_A times it: %s) | - | %s | %s | %s |" % (
                    seed, name.replace("_", "-"), _b(c["twice_n_times_value"]), _b(c["exact"]), _b(c["lower"]), _b(c["upper"])))
            for name, c in e["preservation"].items():
                lines.append("| %s | B-preservation, %s step | %s | %s | %s | %s |" % (seed, name.replace("_", "-"), _b(c["questions"]),
                                                                                       _b(c["exact"]), _b(c["lower"]), _b(c["upper"])))
        lines += ["", "Acquisition of B per configuration (both runs, both gains): %s." % ", ".join(
            "%s %s" % (cfg, _tri(v["acquired"])) for cfg, v in br["acquisition"].items())]
    if br.get("registered"):
        reg = br["registered"]
        lines += ["", "Nominations (both seeds). %s" % note, ""]
        for name, v in reg["nominations"].items():
            lines.append("- Nomination, %s: %s." % (v["contrast"], v["label"]))
        for step, v in reg["attribution"].items():
            lines.append("- Nomination, attribution of the %s step: %s%s." % (step.replace("_", "-"), v["label"],
                                                                                ("" if not v["reasons"] else " (%s)" % "; ".join(v["reasons"]))))
        lines.append("- Nomination note: %s" % reg["attribution_note"])
        if reg["both_zero_sentence"]:
            lines.append("- Nomination note: %s" % reg["both_zero_sentence"])
        lines.append("- Nomination, combined algorithm reading: %s" % (reg["combined_sentence"] or "not written; conditions: %s" % ", ".join(
            "%s %s" % (k, _tri(v)) for k, v in reg["combined_conditions"].items())))
        ig = br["interval_gated"]
        lines += ["", "### " + INTERVAL_HEADING, ""]
        for name, label in ig["contrasts"].items():
            lines.append("- %s: %s." % (STEP_NAMES[name], label))
        lines.append("- Second task across the algorithm step (contrast 5): %s." % ig["contrast_5"])
        for step, v in ig["attribution"].items():
            lines.append("- Attribution of the %s step: %s." % (step.replace("_", "-"), v["label"]))
        lines.append("- Combined reading: %s" % (ig["combined_sentence"] or "not written"))
    elif "per_seed" in br:
        lines += ["", "No registered label: the estimates and bounds above are descriptive."]
    lines += ["", "## Reported with no bar (registration section 4)", "",
              "Tokens per strict-correct answer at the serving budget, its ratio to the incoming checkpoint and to the matched "
              "control, and the at-cost flag (ratio above 1.5; infinite when nothing is correct). Signed terms against the "
              "incoming checkpoint from kit/budget_report.py.", "",
              "| checkpoint | task | S(B) | E(B) | S(H) | E(H) | unfinished at H | tokens per correct | ratio to incoming | at cost "
              "| ratio to control | at cost | F, residual, budget, extraction |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|"]
    for row in report["reported_no_bar"]:
        for task, c in row["tasks"].items():
            if "scores" not in c:
                lines.append("| %s | %s | - | - | - | - | - | - | - | - | - | - | - |" % (row["key"], task))
                continue
            s = c["scores"]
            terms = c.get("signed_terms_vs_incoming")
            terms = ", ".join("%+d" % terms[t] for t in ("F", "residual", "budget", "extraction")) if isinstance(terms, dict) else "-"
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row["key"], task, _b(s["strict_serving"]), _b(s["canonical_serving"]), _b(s["strict_diagnostic"]),
                _b(s["canonical_diagnostic"]), _b(c["unfinished_share_diagnostic"]), _b(c["tokens_per_correct_serving"]),
                _b(c.get("ratio_to_incoming")), _tri(c.get("at_cost_vs_incoming")) if "at_cost_vs_incoming" in c else "-",
                _b(c.get("ratio_to_matched_control")), _tri(c.get("at_cost_vs_matched_control")) if "at_cost_vs_matched_control" in c else "-",
                terms))
    lines += ["", "| checkpoint | panel S(B) | panel S(H) | seconds | GPUs | GPU-hours |", "|---|---:|---:|---:|---:|---:|"]
    for row in report["reported_no_bar"]:
        pan, tr = row["panel"], row.get("training") or {}
        lines.append("| %s | %s | %s | %s | %s | %s |" % (row["key"], _b((pan.get("serving") or {}).get("correct")),
                                                         _b((pan.get("diagnostic") or {}).get("correct")), _b(tr.get("seconds")),
                                                         _b(tr.get("n_gpus")), _b(tr.get("gpu_hours"))))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Package 4: the registered labels, applied mechanically.")
    parser.add_argument("--work", required=True, help="the pilot's WORK folder (runs/ and k8b4/ under it)")
    parser.add_argument("--selection", required=True, help="WORK/k8b4/selection/selection.json from kit/p4_select.py")
    parser.add_argument("--out", required=True, help="a new folder, e.g. WORK/k8b4/report-p4-a1")
    parser.add_argument("--serving", type=int, default=SERVING)
    parser.add_argument("--diagnostic", type=int, default=DIAGNOSTIC)
    parser.add_argument("--blocks", default="prevention,bridges")
    parser.add_argument("--campaign", help="the campaign file run (its rows' runner statuses decide block completeness)")
    parser.add_argument("--budget-status", help="kit/p4_budget.py status --json, written by the campaign's budget-status row")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        if out.exists():
            raise ReportRefused("refusing to overwrite %s: choose a new --out" % out)
        report, manifest = build(args)
    except ReportRefused as exc:
        print("p4_report refused: %s" % exc, file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.mkdir(parents=True)
    (out / "p4-report.json").write_text(json.dumps({**report, "generated_at": stamp}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "p4-manifest.json").write_text(json.dumps({**manifest, "generated_at": stamp}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "p4-report.md").write_text(render(report), encoding="utf-8")
    print("prevention screen: %s; bridges: %s" % (report["prevention"]["screen"], report["bridges"]["status"]))
    print("wrote", out / "p4-report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
