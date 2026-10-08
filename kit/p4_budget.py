#!/usr/bin/env python3
"""Package 4's GPU-hour reservation and running ledger, computed mechanically from the runner's own records.

Implements docs/phase2/plan-v3-package4-preregistration-20261002.md section 6 (ceiling, cost reservation,
qualification allowance, retries, the running ledger) and docs/phase2/plan-v3-package4-supplement1-20261003.md S5
(package ceiling 200; prevention 75 unchanged; bridges 125 with 7*T_max + 20E). Nobody estimates or chooses a number
here: every figure is read from start.json / verdict.json / run-summary.json and the campaign files. Where the text
is silent the tool FAILS CLOSED (exit 2, "unavailable", with the reason). It never reads any score.

    python p4_budget.py reserve --work WORK --pilot-campaign k8b-pilot.yaml --selection selection.json --out R.json
    python p4_budget.py gate    --work WORK --campaign p4.yaml --reservation R.json --block prevention --row ROW [--print-limit]
    python p4_budget.py status  --work WORK --campaign p4.yaml --reservation R.json [--json]

GPU-HOURS of an attempt = seconds * gpus / 3600, exact (fractions.Fraction; floats appear only in printed copies).
    Pilot training row: gpus = `n_gpus` of WORK/runs/<NAME>/run-summary.json (NAME from the attempt's start.json env)
    when that file holds it, else the attempt env `NGPU`, else the calculation is unavailable (no default is assumed); the source
    used is recorded with every input record. Pilot scoring row: gpus = 1. Package-4 row: gpus = its `P4_GPUS`.

T (prevention). The arithmetic mean GPU-hours, over EVERY scheduled pilot stage-2 row of the selected recipe
    (`<recipe>-chemtool-r<i>` and `<recipe>-toolchem-r<i>` in the pilot campaign file), of that row's FIRST valid
    completed 40-step attempt = the lowest attempt number whose verdict is PASS (never a later or faster one), whose
    start.json env has STEPS "40". A scheduled row with no PASS attempt, no seconds, or a first PASS that is not a
    40-step attempt makes the calculation UNAVAILABLE; no substitute.
T_max (bridges). The largest, over recipes g8, g32 and sema, of that recipe's mean computed the same way.
E. 1.5 times the arithmetic mean GPU-hours of the first PASS attempt of every scheduled pilot (model, evaluation)
    pair, evaluations chemistry, toolalpaca and panel.

THE READING OF THE EXCLUSION RULE ("smoke, direct-prefix and repeatability scorings are excluded"). A scheduled pair
    is a pilot campaign row named `K-chemistry`, `K-toolalpaca` or `K-panel`, K being `base8b` or a training run key
    `<recipe>-<chain>-r<i>`. The untrained model's two task scorings were made INSIDE the rows
    `pilot-prefix-chemistry` and `pilot-prefix-toolalpaca`, together with a direct 2,048 scoring and a check: they are
    direct-prefix rows and are excluded, so base8b contributes only `base8b-panel` to E. Also excluded: every
    `pilot-*` row (smoke), every `direct-*` and `prefix-*` row (direct-prefix) and `base8b-repeat-chemistry`
    (repeatability). The included and excluded rows are printed and archived in the reservation.

THE RUNNING LEDGER (`gate`). Reads the package-4 campaign file: rows carry env `P4_BLOCK` (prevention | bridges),
    `P4_KIND` (qualification | training | scoring | direct | cpu) and `P4_GPUS` (an integer as a string); rows without
    P4_BLOCK are ignored. spent(B) = sum over ALL attempts of B's rows (any verdict, FAIL included) of
    seconds * P4_GPUS / 3600; an attempt still without a verdict.json is counted from its start.json `started_at` to
    now. Outstanding scorings = rows of B of kind scoring or direct with no PASS attempt; outstanding training = rows
    of B of kind training with no PASS attempt (the gated row included); the block-wide retry reserve of one training
    slot is held until a training row of B has more than one attempt. Every call appends one JSON line to
    WORK/k8b4/budget/ledger.jsonl. Exit 0 allowed, 1 not allowed ("budget-incomplete: ..."), 2 unavailable/refused.
    Bridges never read prevention's records and the reverse.

AMENDMENT 3 (docs/phase2/plan-v3-package4-amendment3-20261004.md), which governs where it differs from the text above:
  B1  E is computed over the 73 separable pilot (model, evaluation) pairs that have their own runner row; the untrained
      model's two task scorings, bundled inside the prefix-pilot rows, are excluded. The reservation says so.
  B2  no pilot scoring is reused: every scoring of record is made again in the package-4 campaign. The reservation says so.
  B3  EVERY GPU row (training, qualification, qualification scoring, scoring, direct) is admitted by its block's ledger
      and runs (kit/p4_run.py) under a run-time LIMIT: the ledger's limit reduced by the 60-second shutdown allowance, so
      that LIMIT + allowance fits. A failed inequality, a zero limit, or a row ended by its limit writes the DURABLE stop record
      WORK/k8b4/budget/stop-<block>.json (never deleted, never overwritten); while it exists `gate` refuses every later
      GPU row of that block (exit 1, "budget-incomplete: block stopped at ..."), retries included. Missing accounting
      evidence (an attempt with no seconds) is "budget compliance unavailable" (exit 2): never a launch either.
  B5  retries. Never retrained: a run with a complete export from any attempt (kit/p4_run.py reconciles before the
      gate). Retry-eligible: a terminated 40-step attempt whose registered settings are verified from its
      run-summary.json, whose complete export is objectively absent or invalid, and which was neither refused by the
      gate nor ended by the spending limit. A nonzero launcher exit alone is "launcher failure, cause unclassified".
      Missing settings evidence makes eligibility unavailable. Qualification training is never retried. One retry per
      block, for the first eligible failed training slot in the registered order. Launches are counted from the
      wrapper's attempt records WORK/k8b4/budget/attempts/<row>-a<N>.json (kit/p4_run.py); an attempt without one counts
      as launched exactly when its run folder WORK/runs/<NAME> exists.
  Accounting of an attempt: superseded by round 3, F1 below (until the wrapper's verified idle time).

ROUND 3 OF THE SEND-5 REVIEW (the manager's rulings F1, F2, F4 and the should-fix on GPU counts), which governs where it
differs from the text above:
  F1  an attempt is charged until its wrapper VERIFIED the GPUs idle (`charged_until` of WORK/k8b4/budget/attempts/
      <row>-a<N>.json), never only until the launcher's exit; the runner's verdict seconds are a lower bound. An
      attempt still `launching`/`running` with no live wrapper, or `unverified` (its workers not verified terminated),
      is charged from its start to now until a later kit/p4_run.py recovery verifies it. An attempt `interrupted` past
      its deadline is a spending-limit termination: not retry-eligible.
  F2  `reserve`, `gate` and `status` re-verify the frozen decision inputs first (kit/p4_frozen.py): the pilot's
      finalization manifest and every input in it, the selection, the reservation and, once it exists, the recipe
      check's frozen record. Any difference: "refused: inputs changed after they were frozen: <which>", exit 2.
  F4  a retry is authorized only after the failed attempt's ACTUAL env/argv.txt and env/resolved-config.yaml are shown
      to be the frozen baseline's under the exact table (kit/p4_recipe.py `verify_run`); missing evidence: eligibility
      unavailable, no retry.
  GPUs  a training row carries P4_GPUS "baseline": its GPU count is the verified baseline's n_gpus (`knobs.NGPU` of
      WORK/k8b4/recipe-check/baseline.json); absent, the calculation is unavailable. No number is declared by the campaign.

ROUND 4 (the manager's ruling G1): an attempt is never charged to an inferred termination time (kit/p4_run.py's
recovery charges through the watchdog's or its own verified time); a block whose stop says its accounting is
unavailable, or that holds an attempt whose workers are not verified terminated, has `accounting` "budget compliance
unavailable" in `status` (no overrun is claimed, nothing more of it launches).

ROUND 5 (the manager's ruling H2): SPENDING-LIMIT EVIDENCE IS READ INDEPENDENTLY OF THE ATTEMPT'S `class`.
`spending_limit_evidence` says whether a launched attempt was ended by its spending limit: its record says `ended_by:
limit` (or past its limit), its watchdog record (WORK/k8b4/budget/watchdog/<row>-a<N>.json) says the job was alive at
`limit_until`, or its end time is at or after `limit_until`. On that evidence `gate` refuses every later GPU row of the
block (exit 1, "budget-incomplete", and writes the stop if it is missing), `eligibility` refuses a retry and `status`
labels the block "budget-incomplete" -- whatever the attempt's `class` says and whether or not a stop file exists. The
stop file itself is written once, durably and atomically (fsync of a temporary file, then an exclusive link).

KIT_P4_LIMIT_CAP ("row=seconds,row=seconds") can only LOWER a row's limit; the dry run (scripts/simulate_p4.py) uses it
to end one training row by its limit. It is recorded in the ledger line whenever it applies.

Standard library only (plus the kit's runner for loading campaign files).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

SCHEMA = "kit-p4-budget.v1"
EXIT_OK, EXIT_NOT_ALLOWED, EXIT_UNAVAILABLE = 0, 1, 2
PREVENTION, BRIDGES = "prevention", "bridges"
BLOCKS = (PREVENTION, BRIDGES)
KINDS = ("qualification", "training", "scoring", "direct", "selftest", "cpu")
CEILING = {PREVENTION: Fraction(75), BRIDGES: Fraction(125)}
PACKAGE_CEILING = Fraction(200)
RECIPES_T_MAX = ("g8", "g32", "sema")
SELECTABLE_RECIPES = ("g8", "sema")                         # supplement S2: "The selected group's recipe is g8 or sema."
EVALUATIONS = ("chemistry", "toolalpaca", "panel")
STAGE2_CHAINS = ("chemtool", "toolchem")
TRAINING_ROW = re.compile(r"^(g8|g32|sema|sfrz)-(chem|tool|chemtool|toolchem)-r(\d+)$")
DEFAULT_TRAINING_GPUS = 8
STAGE2_STEPS = "40"
E_FACTOR = Fraction(3, 2)
LEDGER = Path("k8b4") / "budget" / "ledger.jsonl"
BUDGET_DIR = Path("k8b4") / "budget"
ATTEMPTS_DIR = BUDGET_DIR / "attempts"
STARTED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
GRACE_SECONDS = 60                          # the shutdown allowance the limit reserves after limit_until (amendment 3 B3, H1)
GPU_KINDS = ("qualification", "training", "scoring", "direct", "selftest")
RETRY_STEPS = "40"                          # B5: only a terminated 40-step attempt can be retry-eligible
LIMIT_CAP_ENV = "KIT_P4_LIMIT_CAP"
FROM_BASELINE = "baseline"                  # P4_GPUS of a training row: the verified baseline's n_gpus (round-3 should-fix)
BASELINE = BUDGET_DIR.parent / "recipe-check" / "baseline.json"
BUDGET_INCOMPLETE = "budget-incomplete"
COMPLIANCE_UNAVAILABLE = "budget compliance unavailable"
UNCLASSIFIED = "launcher failure, cause unclassified"
AMENDMENT_3 = {
    "source": "docs/phase2/plan-v3-package4-amendment3-20261004.md (amendment 3)",
    "E": "amendment 3 B1: E is 1.5 times the mean GPU-hours of the first valid full scoring of each separable pilot "
         "(model, evaluation) pair with its own runner row (%d pairs here); the untrained model's two task scorings, "
         "made inside the prefix-pilot rows, are excluded",
    "rescoring": "amendment 3 B2: no pilot scoring is reused; every checkpoint entering a contrast is scored again in "
                 "the package-4 campaign"}


class Unavailable(Exception):
    """The calculation cannot be made from the records; nothing is estimated in its place."""


# ------------------------------------------------------------------------------------ small helpers
def exact(value: Fraction | None) -> dict | None:
    """Both copies of a number: the exact fraction as a string, and a float for reading."""
    if value is None:
        return None
    return {"exact": str(value), "float": float(value)}


def parse_exact(value, what: str) -> Fraction:
    try:
        text = value["exact"] if isinstance(value, dict) else value
        if not isinstance(text, str):
            raise TypeError
        return Fraction(text)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise Unavailable("%s is not an exact value in the reservation: %r" % (what, value)) from None


def as_seconds(value, where: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise Unavailable("%s: seconds missing or not a non-negative number (%r)" % (where, value))
    return Fraction(str(value))                               # the decimal the runner wrote, exactly


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen():
    spec = importlib.util.spec_from_file_location("kit_p4_frozen_for_p4_budget", Path(__file__).resolve().parent / "p4_frozen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner():
    spec = importlib.util.spec_from_file_location("kit_runner_for_p4_budget", Path(__file__).resolve().parent / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_campaign(path: Path) -> dict:
    """A campaign file, validated by the kit's own runner; it must carry a name (its records live under it)."""
    runner = _runner()
    try:
        campaign = runner.load_campaign(Path(path))
    except (OSError, ValueError) as error:
        raise Unavailable("campaign %s cannot be loaded: %s" % (path, error)) from None
    if not campaign.get("name"):
        raise Unavailable("campaign %s has no name" % path)
    return campaign


def attempt_records(work: Path, campaign_name: str, row_id: str) -> list:
    """Every attempt of a row, lowest first: {attempt, dir, start, verdict} (start/verdict None when absent)."""
    base = Path(work) / "campaign" / campaign_name / row_id
    if not base.is_dir():
        return []
    found = []
    for path in base.glob("attempt-*"):
        suffix = path.name.split("-", 1)[1]
        if path.is_dir() and suffix.isdigit():
            found.append(int(suffix))
    records = []
    for number in sorted(found):
        directory = base / ("attempt-%d" % number)
        record = {"attempt": number, "dir": str(directory), "start": None, "verdict": None}
        for key, name in (("start", "start.json"), ("verdict", "verdict.json")):
            if (directory / name).is_file():
                try:
                    record[key] = json.loads((directory / name).read_text())
                except (OSError, ValueError) as error:
                    raise Unavailable("%s/%s cannot be read: %s" % (directory, name, error)) from None
        records.append(record)
    return records


def first_pass(work: Path, campaign_name: str, row_id: str) -> dict:
    """The lowest-numbered attempt whose verdict is PASS; Unavailable when there is none."""
    for record in attempt_records(work, campaign_name, row_id):
        if (record["verdict"] or {}).get("verdict") == "PASS":
            return record
    raise Unavailable("scheduled row %s has no PASS attempt" % row_id)


# ------------------------------------------------------------------------------------ reserve
def training_gpus(work: Path, start: dict | None, row_id: str) -> tuple:
    """(gpus, source) of a pilot training attempt: run-summary n_gpus, else env NGPU, else UNAVAILABLE (no default)."""
    env = (start or {}).get("env") or {}
    name = env.get("NAME")
    if not name:
        raise Unavailable("row %s: its start.json env has no NAME, so its run-summary.json cannot be found" % row_id)
    summary_path = Path(work) / "runs" / name / "run-summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, ValueError) as error:
            raise Unavailable("%s cannot be read: %s" % (summary_path, error)) from None
        if "steps" in summary and summary["steps"] != int(STAGE2_STEPS):
            raise Unavailable("%s records steps %r, not %s" % (summary_path, summary["steps"], STAGE2_STEPS))
        if "n_gpus" in summary:
            gpus = summary["n_gpus"]
            if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 1:
                raise Unavailable("%s: n_gpus is not a positive integer (%r)" % (summary_path, gpus))
            return gpus, "run-summary n_gpus (runs/%s/run-summary.json)" % name
    if "NGPU" in env:
        text = str(env["NGPU"])
        if not text.isdigit() or int(text) < 1:
            raise Unavailable("row %s: env NGPU is not a positive integer (%r)" % (row_id, env["NGPU"]))
        return int(text), "start.json env NGPU"
    raise Unavailable("row %s: no GPU-allocation record (no run-summary n_gpus, no env NGPU); the registration allows no substitute" % row_id)


def stage2_rows(campaign: dict, recipe: str) -> list:
    """Every scheduled pilot stage-2 row of a recipe, in campaign order."""
    out = []
    for row in campaign["rows"]:
        match = TRAINING_ROW.match(row["id"])
        if match and match.group(1) == recipe and match.group(2) in STAGE2_CHAINS:
            out.append(row["id"])
    return out


def recipe_mean(work: Path, campaign: dict, recipe: str) -> tuple:
    """(mean GPU-hours, input records) over every scheduled stage-2 row of the recipe; Unavailable if any is missing."""
    rows = stage2_rows(campaign, recipe)
    if not rows:
        raise Unavailable("the pilot campaign schedules no stage-2 row of recipe %s" % recipe)
    records = []
    for row_id in rows:
        record = first_pass(work, campaign["name"], row_id)
        steps = ((record["start"] or {}).get("env") or {}).get("STEPS")
        if steps != STAGE2_STEPS:
            raise Unavailable("row %s attempt %d (its first PASS) is not a 40-step attempt: start.json env STEPS %r"
                              % (row_id, record["attempt"], steps))
        seconds = as_seconds(record["verdict"].get("seconds"), "row %s attempt %d" % (row_id, record["attempt"]))
        gpus, source = training_gpus(work, record["start"], row_id)
        hours = seconds * gpus / 3600
        records.append({"row": row_id, "recipe": recipe, "attempt": record["attempt"], "seconds": str(seconds),
                        "gpus": gpus, "gpus_source": source, "gpu_hours": exact(hours)})
    mean = sum((parse_exact(r["gpu_hours"], r["row"]) for r in records), Fraction(0)) / len(records)
    return mean, records


def classify_pilot_rows(campaign: dict) -> tuple:
    """(included scoring rows, excluded rows with reasons) under the reading in the module docstring."""
    ids = [row["id"] for row in campaign["rows"]]
    keys = ["base8b"] + [rid for rid in ids if TRAINING_ROW.match(rid)]
    pairs = {"%s-%s" % (key, evaluation) for key in keys for evaluation in EVALUATIONS}
    included, excluded = [], []
    for rid in ids:
        if rid.startswith("pilot-prefix-"):
            excluded.append({"row": rid, "reason": "direct-prefix (holds the untrained model's task scoring, a direct "
                                                   "2,048 scoring and a check)"})
        elif rid.startswith("pilot-"):
            excluded.append({"row": rid, "reason": "smoke (pilot row)"})
        elif rid.startswith("direct-") or rid.startswith("prefix-"):
            excluded.append({"row": rid, "reason": "direct-prefix"})
        elif rid == "base8b-repeat-chemistry":
            excluded.append({"row": rid, "reason": "repeatability"})
        elif rid in pairs:
            included.append(rid)
    return included, excluded


def compute_e(work: Path, campaign: dict) -> tuple:
    """(E, mean, included rows, excluded rows, input records)."""
    included, excluded = classify_pilot_rows(campaign)
    if not included:
        raise Unavailable("the pilot campaign schedules no (model, evaluation) scoring row")
    records = []
    for row_id in included:
        record = first_pass(work, campaign["name"], row_id)
        seconds = as_seconds(record["verdict"].get("seconds"), "row %s attempt %d" % (row_id, record["attempt"]))
        hours = seconds * 1 / 3600
        records.append({"row": row_id, "attempt": record["attempt"], "seconds": str(seconds), "gpus": 1,
                        "gpus_source": "scoring row", "gpu_hours": exact(hours)})
    mean = sum((parse_exact(r["gpu_hours"], r["row"]) for r in records), Fraction(0)) / len(records)
    return E_FACTOR * mean, mean, included, excluded, records


def content_sha256(content: dict) -> str:
    """The sha256 of a document's canonical content (sorted keys, no whitespace): what a later invocation must equal."""
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_selection_document(path: Path) -> dict:
    try:
        selection = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise Unavailable("selection %s cannot be read: %s" % (path, error)) from None
    if selection.get("entered") is not True:
        raise Unavailable("selection %s: entered is %r, not true; package 4 is not built" % (path, selection.get("entered")))
    recipe = (selection.get("group") or {}).get("recipe")
    if recipe not in SELECTABLE_RECIPES:
        raise Unavailable("selection %s: group.recipe is %r; the selected recipe must be one of %s"
                          % (path, recipe, ", ".join(SELECTABLE_RECIPES)))
    return selection


def read_selection(path: Path) -> str:
    return read_selection_document(path)["group"]["recipe"]


def compute_reservation(work: Path, pilot_campaign: Path, selection: Path) -> dict:
    """The reservation document. Unavailable when T or E cannot be computed. When only T_max cannot be (a g32 record
    missing), the bridges block is written as unavailable and not launchable (S5: "the block is 'not run'")."""
    selected = read_selection_document(selection)
    recipe = selected["group"]["recipe"]
    campaign = load_campaign(pilot_campaign)
    work = Path(work)
    t_selected, t_records = recipe_mean(work, campaign, recipe)
    e_value, e_mean, included, excluded, e_records = compute_e(work, campaign)
    by_recipe, stage2_records, t_max_reason = {recipe: t_selected}, list(t_records), None
    for other in RECIPES_T_MAX:
        if other in by_recipe:
            continue
        try:
            by_recipe[other], records = recipe_mean(work, campaign, other)
            stage2_records += records
        except Unavailable as error:
            t_max_reason = "T_max unavailable: recipe %s: %s" % (other, error)
    t_max = None if t_max_reason else max(by_recipe[r] for r in RECIPES_T_MAX)
    t_max_recipe = None if t_max is None else [r for r in RECIPES_T_MAX if by_recipe[r] == t_max][0]

    prevention_reserve = 9 * t_selected + 45 * e_value
    allowance = CEILING[PREVENTION] - prevention_reserve
    prevention = {"ceiling": str(CEILING[PREVENTION]), "rule": "reserve 9T + 45E; qualification_allowance = 75 - 9T - 45E",
                  "reserve": exact(prevention_reserve), "qualification_allowance": exact(allowance),
                  "qualification_may_start": allowance > 0}
    if t_max is None:
        bridges = {"ceiling": str(CEILING[BRIDGES]), "rule": "launched only if 7*T_max + 20E <= 125",
                   "available": False, "reason": t_max_reason, "reserve": None, "may_launch": False}
    else:
        bridges_reserve = 7 * t_max + 20 * e_value
        bridges = {"ceiling": str(CEILING[BRIDGES]), "rule": "launched only if 7*T_max + 20E <= 125", "available": True,
                   "reserve": exact(bridges_reserve), "may_launch": bridges_reserve <= CEILING[BRIDGES]}
    return {
        "schema": SCHEMA,
        "pilot_campaign": {"name": campaign["name"], "sha256": sha256(Path(pilot_campaign))},
        # bound to the frozen selection and, through it, to the pilot's finalization manifest (amendment 3 B7)
        "selection": {"content_sha256": selected.get("content_sha256"), "recipe": recipe, "entered": True,
                      "finalization_sha256": (selected.get("finalization") or {}).get("sha256")},
        "amendment_3": {"source": AMENDMENT_3["source"], "E": AMENDMENT_3["E"] % len(included),
                        "rescoring": AMENDMENT_3["rescoring"]},
        "recipe": recipe,
        "T": exact(t_selected),
        "T_by_recipe": {r: exact(v) for r, v in by_recipe.items()},
        "T_max": exact(t_max),
        "T_max_recipe": t_max_recipe,
        "E": exact(e_value),
        "E_mean_scoring_gpu_hours": exact(e_mean),
        "exclusion_reading": ("smoke, direct-prefix and repeatability scorings are excluded: pilot-*, direct-*, "
                              "prefix-*, base8b-repeat-chemistry; the untrained model's task scorings were made inside "
                              "pilot-prefix-chemistry and pilot-prefix-toolalpaca, so base8b contributes only base8b-panel"),
        "included_scoring_rows": included,
        "excluded_rows": excluded,
        "inputs": {"stage2": stage2_records, "scorings": e_records},
        "blocks": {PREVENTION: prevention, BRIDGES: bridges},
        "package_ceiling": str(PACKAGE_CEILING),
    }


def reserve(work: Path, pilot_campaign: Path, selection: Path, out: Path) -> int:
    """The FIRST reservation is the reservation (amendment 3 B7). A later call recomputes it: identical canonical
    content (content_sha256, which leaves out the time and every absolute path) is "reservation unchanged", exit 0;
    anything else is refused, exit 2, and the frozen file is never replaced."""
    out = Path(out)
    try:
        document = compute_reservation(Path(work), Path(pilot_campaign), Path(selection))
    except Unavailable as error:
        print("unavailable: %s" % error)
        return EXIT_UNAVAILABLE
    digest = content_sha256(document)
    if out.exists():
        try:
            frozen = json.loads(out.read_text())
        except (OSError, ValueError) as error:
            print("refused: the frozen reservation %s cannot be read (%s); it is never replaced" % (out, error))
            return EXIT_UNAVAILABLE
        if frozen.get("content_sha256") == digest and content_sha256(
                {k: v for k, v in frozen.items() if k not in ("content_sha256", "created_at", "located_at")}) == digest:
            print("reservation unchanged: %s (content_sha256 %s)" % (out, digest[:16]))
            return EXIT_OK
        print("refused: the pilot's inputs changed after the reservation: %s holds content_sha256 %s, the recomputation "
              "gives %s; nothing is overwritten" % (out, str(frozen.get("content_sha256"))[:16], digest[:16]))
        return EXIT_UNAVAILABLE
    record = {**document, "content_sha256": digest, "created_at": datetime.now(timezone.utc).strftime(STARTED_AT_FORMAT),
              "located_at": {"work": str(Path(work).resolve()), "pilot_campaign": str(Path(pilot_campaign).resolve()),
                             "selection": str(Path(selection).resolve())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:                              # exclusive create: a race cannot overwrite either
        json.dump(record, handle, indent=1, sort_keys=True)
        handle.write("\n")
    prevention, bridges = document["blocks"][PREVENTION], document["blocks"][BRIDGES]
    print(document["amendment_3"]["E"])
    print(document["amendment_3"]["rescoring"])
    print("recipe %s  T %s  T_max %s (%s)  E %s" % (document["recipe"], document["T"]["float"],
                                                  (document["T_max"] or {}).get("float"), document["T_max_recipe"],
                                                  document["E"]["float"]))
    print("included scoring rows (%d): %s" % (len(document["included_scoring_rows"]), " ".join(document["included_scoring_rows"])))
    print("excluded rows (%d): %s" % (len(document["excluded_rows"]), " ".join(r["row"] for r in document["excluded_rows"])))
    print("prevention: reserve 9T + 45E = %s; qualification_allowance %s; qualification_may_start %s"
          % (prevention["reserve"]["float"], prevention["qualification_allowance"]["float"], prevention["qualification_may_start"]))
    if bridges["available"]:
        print("bridges: reserve 7*T_max + 20E = %s <= 125: may_launch %s" % (bridges["reserve"]["float"], bridges["may_launch"]))
    else:
        print("bridges: unavailable (%s); may_launch False" % bridges["reason"])
    print("written %s" % out)
    return EXIT_OK


# ------------------------------------------------------------------------------------ the running ledger
def p4_rows(campaign: dict) -> dict:
    """{row id: {block, kind, gpus, order}} for rows carrying P4_BLOCK, read from the campaign file; others are ignored.
    A qualification row with P4_GPUS above 1 is a qualification TRAINING run; with 1, the scoring of one."""
    rows = {}
    for index, row in enumerate(campaign["rows"]):
        env = row.get("env") or {}
        if "P4_BLOCK" not in env:
            continue
        block, kind, gpus = env.get("P4_BLOCK"), env.get("P4_KIND"), str(env.get("P4_GPUS", ""))
        if block not in BLOCKS:
            raise Unavailable("row %s: P4_BLOCK %r is not one of %s" % (row["id"], block, BLOCKS))
        if kind not in KINDS:
            raise Unavailable("row %s: P4_KIND %r is not one of %s" % (row["id"], kind, KINDS))
        if gpus == FROM_BASELINE and kind in ("training", "qualification"):
            # a training launch: its GPU count is the VERIFIED baseline's n_gpus (round-3 should-fix), never a number the
            # campaign declares
            rows[row["id"]] = {"block": block, "kind": kind, "gpus": FROM_BASELINE, "order": index, "trains": True}
            continue
        if not gpus.isdigit() or (int(gpus) < 1 and kind != "cpu"):
            raise Unavailable("row %s: P4_GPUS %r is not an integer (at least 1 for a GPU row) or %r for a training row"
                              % (row["id"], env.get("P4_GPUS"), FROM_BASELINE))
        if kind == "training":
            raise Unavailable("row %s: a training row's P4_GPUS must be %r (the verified baseline's n_gpus), not %r"
                              % (row["id"], FROM_BASELINE, gpus))
        rows[row["id"]] = {"block": block, "kind": kind, "gpus": int(gpus), "order": index,
                           "trains": kind == "qualification" and int(gpus) > 1}
    return rows


def ignored_rows(campaign: dict) -> list:
    return [row["id"] for row in campaign["rows"] if "P4_BLOCK" not in (row.get("env") or {})]


def _started(start: dict | None, where: str) -> datetime:
    text = (start or {}).get("started_at")
    try:
        return datetime.strptime(text, STARTED_AT_FORMAT).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        raise Unavailable("%s has no verdict and no readable start.json started_at (%r)" % (where, text)) from None


# ---- the durable stop (amendment 3 B3)
def stop_path(work: Path, block: str) -> Path:
    return Path(work) / BUDGET_DIR / ("stop-%s.json" % block)


def read_stop(work: Path, block: str):
    """The stop record of a block, or None. A stop file that cannot be read is still a stop."""
    path = stop_path(work, block)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        return record if isinstance(record, dict) else {"unreadable": True, "path": str(path)}
    except (OSError, ValueError):
        return {"unreadable": True, "path": str(path)}


def record_stop(work: Path, block: str, entry: dict) -> bool:
    """Write WORK/k8b4/budget/stop-<block>.json once, durably and atomically (the whole content fsynced under a temporary
    name, then LINKED to its name: exclusive, so an existing stop is never replaced, and never seen half-written); True
    when this call wrote it. Nothing ever deletes or replaces it: the first stop is the stop."""
    path = stop_path(work, block)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.tmp-%d-%d" % (path.name, os.getpid(), time.monotonic_ns()))
    with tmp.open("w") as handle:
        json.dump({"schema": SCHEMA, "block": block, **entry}, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
        created = True
    except FileExistsError:
        created = False
    finally:
        tmp.unlink(missing_ok=True)
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
    return created


WATCHDOG_DIR = BUDGET_DIR / "watchdog"


def _time(text):
    for fmt in (STARTED_AT_FORMAT, "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def limit_until_of(record: dict):
    """The end of an attempt's admitted run time: its record's `limit_until`, else launched_at + limit_seconds."""
    if not isinstance(record, dict):
        return None
    at = _time(record.get("limit_until"))
    if at is None:
        launched, limit = _time(record.get("launched_at")), record.get("limit_seconds")
        if launched is not None and isinstance(limit, (int, float)) and not isinstance(limit, bool):
            from datetime import timedelta                                   # noqa: PLC0415
            at = launched + timedelta(seconds=limit)
    return at


def spending_limit_evidence(work: Path, record: dict, watch: dict | None = None):
    """Round-5 ruling H2: why a LAUNCHED attempt counts as ended by its spending limit, or None -- read from the evidence,
    independently of its `class`: `ended_by: limit`; `past_limit` (or round 4's `past_deadline`) true; its watchdog
    record says the job was alive at limit_until; its end time (`ended_precise`, else `ended_at`) is at or after
    limit_until."""
    if not isinstance(record, dict) or not record.get("launched"):
        return None
    why = []
    scheduler = record.get('slurm') or {}
    termination = scheduler.get('termination') or {}
    expected_probe_timeout = record.get('kind') == 'selftest' and record.get('selftest_expected_timeout') is True
    if (termination.get('state') == 'TIMEOUT' and not expected_probe_timeout) or scheduler.get('cancel_at_limit'):
        why.append('scheduler TIMEOUT or owning-step cancellation at the spending limit')
    if record.get("ended_by") == "limit":
        why.append("its record says ended_by limit")
    if record.get("past_limit") is True or record.get("past_deadline") is True:
        why.append("its record says it ended at or after the end of its admitted run time")
    if watch is None and record.get("row") and isinstance(record.get("attempt"), int):
        watch = _json_file(Path(work) / WATCHDOG_DIR / ("%s-a%d.json" % (record["row"], record["attempt"])))
    if isinstance(watch, dict) and (watch.get("job_alive_at_limit") is True or watch.get("group_alive_at_deadline") is True):
        why.append("its watchdog found the job alive when its admitted run time ended")
    limit_until = limit_until_of(record)
    end = _time(record.get("ended_precise")) or _time(record.get("ended_at"))
    if limit_until is not None and end is not None and end >= limit_until:
        why.append("it ended at %s, at or after the end of its admitted run time %s" % (
            record.get("ended_precise") or record.get("ended_at"), limit_until.strftime(STARTED_AT_FORMAT)))
    return "; ".join(why) or None


def block_limit_evidence(work: Path, block: str) -> list:
    """[(row, attempt, why)] of every attempt record of `block` holding spending-limit evidence."""
    out = []
    for path in sorted((Path(work) / ATTEMPTS_DIR).glob("*.json")):
        record = _json_file(path)
        if isinstance(record, dict) and record.get("block") == block:
            why = spending_limit_evidence(work, record)
            if why:
                out.append((record.get("row"), record.get("attempt"), why))
    return out


# ---- the wrapper's attempt records (kit/p4_run.py) and what they prove
def attempt_record_path(work: Path, row_id: str, attempt: int) -> Path:
    return Path(work) / ATTEMPTS_DIR / ("%s-a%d.json" % (row_id, attempt))


def read_attempt_record(work: Path, row_id: str, attempt: int):
    path = attempt_record_path(work, row_id, attempt)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise Unavailable("%s cannot be read: %s" % (path, error)) from None
    return record if isinstance(record, dict) else None


def complete_export(folder: Path) -> bool:
    """The launchers' own test of a merged model (kit/run_grpo_toolalpaca.sh `complete_export`): config.json,
    tokenizer_config.json, and every shard model.safetensors.index.json names (or one model.safetensors), none empty."""
    folder = Path(folder)
    if not ((folder / "config.json").is_file() and (folder / "tokenizer_config.json").is_file()):
        return False
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError, AttributeError):
            return False
        return bool(shards) and all((folder / s).is_file() and (folder / s).stat().st_size > 0 for s in shards)
    single = folder / "model.safetensors"
    return single.is_file() and single.stat().st_size > 0


def _same_fraction(a, b) -> bool:
    try:
        return Fraction(str(a).strip()) == Fraction(str(b).strip())
    except (ValueError, ZeroDivisionError):
        return False


def verify_settings(summary, env: dict) -> tuple:
    """(True, []) when the launcher's run-summary.json records exactly the registered settings the row's env asked for;
    (False, [differences]); (None, [why]) when there is no summary to verify against (settings not recorded)."""
    if not isinstance(summary, dict):
        return None, ["run-summary.json missing or unreadable: the settings are not recorded"]
    wrong = []

    def need(key, ok, wanted):
        if key not in summary:
            wrong.append("%s not recorded (registered %r)" % (key, wanted))
        elif not ok(summary[key]):
            wrong.append("%s %r, registered %r" % (key, summary[key], wanted))

    if env.get("NAME"):
        need("name", lambda v: v == env["NAME"], env["NAME"])
    if env.get("STEPS"):
        need("steps", lambda v: str(v) == str(env["STEPS"]), env["STEPS"])
    if env.get("LR"):
        need("learning_rate", lambda v: _same_fraction(v, env["LR"]), env["LR"])
    if env.get("DATASET"):
        need("dataset", lambda v: v == env["DATASET"], env["DATASET"])
    if "SEED" in env:
        need("seed", lambda v: str(v) == str(env["SEED"]), env["SEED"])
    if env.get("MODEL_DIR"):
        need("model_dir", lambda v: v == env["MODEL_DIR"], env["MODEL_DIR"])
    need("max_response_length", lambda v: str(v) == str(env.get("MAX_RESPONSE") or 8192), env.get("MAX_RESPONSE") or 8192)
    need("finish_gate", lambda v: str(v) == str(env.get("FINISH_GATE") or 0), env.get("FINISH_GATE") or 0)
    if env.get("NGPU"):
        need("n_gpus", lambda v: str(v) == str(env["NGPU"]), env["NGPU"])
    if summary.get("schema") == "kit-grpo-toolalpaca-run.v1":
        if env.get("MINI_BATCH"):
            need("mini_batch", lambda v: str(v) == str(env["MINI_BATCH"]), env["MINI_BATCH"])
        if "LORA" in env:
            need("lora", lambda v: str(v) == str(env["LORA"]), env["LORA"])
    elif summary.get("schema") == "kit-sdpo-run.v1":
        if env.get("TEACHER_RATE"):
            need("teacher_update_rate", lambda v: _same_fraction(v, env["TEACHER_RATE"]), env["TEACHER_RATE"])
        for key, name in (("FEEDBACK", "feedback"), ("SOFT", "soft")):
            if key in env:
                need(name, lambda v, key=key: str(v) == str(env[key]), env[key])
        if "TEMP" in env:
            need("temperature", lambda v: str(v) == (env["TEMP"] or "default"), env["TEMP"] or "default")
    else:
        wrong.append("schema %r is not a package-4 launcher's" % summary.get("schema"))
    return (not wrong), wrong


def _json_file(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def attempt_history(work: Path, campaign_name: str, row_id: str) -> list:
    """Every runner attempt of a row with the wrapper's record and whether it LAUNCHED the launcher: the record says
    so when there is one; without one, an attempt launched exactly when its run folder WORK/runs/<NAME> exists."""
    out = []
    records = attempt_records(work, campaign_name, row_id)
    known = {r['attempt'] for r in records}
    # B3 also covers admitted wrapper calls made outside runner.py. A missing
    # runner start/verdict never makes their GPU exposure disappear from spent.
    for path in (Path(work) / ATTEMPTS_DIR).glob(row_id + '-a*.json'):
        wrapper = _json_file(path)
        k = (wrapper or {}).get('attempt')
        if isinstance(wrapper, dict) and wrapper.get('row') == row_id and isinstance(k, int) and k >= 1 and k not in known:
            records.append({'attempt': k, 'dir': None, 'start': None, 'verdict': None})
            known.add(k)
    for record in sorted(records, key=lambda r: r['attempt']):
        wrapper = read_attempt_record(work, row_id, record["attempt"])
        name = ((record["start"] or {}).get("env") or {}).get("NAME")
        if wrapper is not None:
            launched = bool(wrapper.get("launched"))
        else:
            launched = bool(name) and (Path(work) / "runs" / name).exists()
        out.append({**record, "record": wrapper, "launched": launched, "name": name})
    return out


def _recipe():
    spec = importlib.util.spec_from_file_location("kit_p4_recipe_for_p4_budget", Path(__file__).resolve().parent / "p4_recipe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eligibility(work: Path, attempt: dict, workers_verified: bool = False, baseline=None) -> dict:
    """Amendment 3 B5 with the round-3 rulings F1 and F4: whether one launched training attempt is retry-eligible.
    {eligible: True | False | None (unavailable), class, why, evidence}. Before a retry is allowed the attempt's ACTUAL
    env/argv.txt and env/resolved-config.yaml are compared with the frozen baseline (kit/p4_recipe.py `verify_run`, the
    exact table); missing evidence makes eligibility unavailable, a difference makes the attempt not eligible."""
    env = dict(((attempt.get("start") or {}).get("env") or {}))
    wrapper = attempt.get("record") or {}
    env.update(wrapper.get("launcher_env") or {})
    name, steps = attempt.get("name") or env.get("NAME"), str(env.get("STEPS", ""))
    run = Path(work) / "runs" / name if name else None
    evidence = {"attempt": attempt["attempt"], "run": "runs/%s" % name if name else None,
                "launcher_exit": wrapper.get("status"), "wrapper_class": wrapper.get("class"), "wrapper_state": wrapper.get("state")}
    limit_evidence = spending_limit_evidence(work, wrapper) if wrapper else None
    if limit_evidence:                                         # H2: read from the evidence, whatever the class says
        evidence["spending_limit_evidence"] = limit_evidence
        return {"eligible": False, "class": "budget", "why": "attempt %d was ended by its spending limit (%s): not retry-eligible (B5)"
                % (attempt["attempt"], limit_evidence), "evidence": evidence}
    if wrapper.get("class") in ("budget", "refused") or (wrapper.get("class") == "interrupted" and wrapper.get("past_deadline")):
        return {"eligible": False, "class": "budget" if wrapper.get("class") != "refused" else "refused",
                "why": "attempt %d was %s: not retry-eligible (B5)" % (attempt["attempt"], "refused by the gate" if wrapper.get("class") == "refused"
                                                                         else "ended by the spending limit or interrupted past its deadline"),
                "evidence": evidence}
    if steps != RETRY_STEPS:
        return {"eligible": False, "class": "not a 40-step training run", "why": "attempt %d has STEPS %r: only a terminated "
                "40-step training attempt can be retried; qualification training is never retried (B5)" % (attempt["attempt"], steps),
                "evidence": evidence}
    if wrapper.get("state") in ("launching", "running", "unverified") or (wrapper and not wrapper.get("verified_idle_at")
                                                                          and wrapper.get("launched")):
        return {"eligible": None, "class": "termination not verified", "why": "attempt %d: its workers have not been verified "
                "terminated (state %s): eligibility unavailable" % (attempt["attempt"], wrapper.get("state")), "evidence": evidence}
    if not wrapper:
        if not workers_verified:
            return {"eligible": None, "class": "termination not verified", "why": "attempt %d has no wrapper record and its workers "
                    "have not been verified terminated: eligibility unavailable" % attempt["attempt"], "evidence": evidence}
        evidence["terminated"] = "no wrapper record; the GPUs were verified idle before this gate"
    else:
        evidence["terminated"] = "the wrapper verified the GPUs idle at %s" % wrapper.get("verified_idle_at")
    export = complete_export(run / ("hf-step%s" % steps)) if run is not None else False
    evidence["export_complete"] = export
    if export:
        return {"eligible": False, "class": "complete export", "why": "attempt %d left a complete export: it is never "
                "retrained (kit/p4_run.py keeps it)" % attempt["attempt"], "evidence": evidence}
    summary = _json_file(run / "run-summary.json") if run is not None else None
    verified, wrong = verify_settings(summary, env)
    evidence.update({"settings_verified": verified, "settings": wrong})
    if verified is None:
        return {"eligible": None, "class": "settings not recorded", "why": "attempt %d: %s; eligibility unavailable (B5)"
                % (attempt["attempt"], "; ".join(wrong)), "evidence": evidence}
    if verified is False:
        return {"eligible": False, "class": "settings", "why": "attempt %d ran with settings other than the registered "
                "ones (%s): not retry-eligible" % (attempt["attempt"], "; ".join(wrong)), "evidence": evidence}
    # F4: the attempt's ACTUAL command and resolved configuration against the frozen baseline, under the exact table
    baseline_path = Path(baseline) if baseline else Path(work) / BASELINE
    frozen = _json_file(baseline_path)
    row_id = ((attempt.get("start") or {}).get("row")) or wrapper.get("row") or (name.rsplit("-a", 1)[0] if name else None)
    actual, problems = _recipe().verify_run(Path(work), frozen, row_id, run, baseline_path.parent) if run is not None else (None, ["no run folder"])
    evidence.update({"actual_settings_verified": actual, "actual_settings": problems})
    if actual is None:
        return {"eligible": None, "class": "settings evidence missing", "why": "attempt %d: %s; eligibility unavailable (F4)"
                % (attempt["attempt"], "; ".join(problems[:3])), "evidence": evidence}
    if actual is False:
        return {"eligible": False, "class": "settings", "why": "attempt %d: its actual command or resolved configuration is not the "
                "frozen baseline's under the permitted substitutions (%s): not retry-eligible" % (attempt["attempt"], "; ".join(problems[:3])),
                "evidence": evidence}
    status = wrapper.get("status")
    failure = ("invalid export (launcher exit 4: trained, no complete merged model)" if status == 4 else
               "no complete export after launcher exit 0" if status == 0 else
               "interrupted (wrapper gone, recovered within its deadline)" if wrapper.get("class") == "interrupted" else
               "%s (exit %s)" % (UNCLASSIFIED, status))
    return {"eligible": True, "class": failure, "why": "attempt %d: terminated 40-step attempt, settings verified (summary, actual "
            "command and resolved configuration), complete export objectively absent; %s" % (attempt["attempt"], failure), "evidence": evidence}


def attempt_seconds(work: Path, campaign_name: str, row_id: str, history: list, item: dict, now: datetime) -> tuple:
    """(seconds, state) of one attempt for the ledger (round-3 ruling F1). The wrapper's record decides when the attempt
    ended: until its VERIFIED idle time (`charged_until`), never only until the launcher's exit. An attempt whose workers
    are not verified terminated (state `unverified`, or a wrapper record still launching/running) is charged from its
    start to now. The runner's verdict seconds are a lower bound: the larger of the two is charged."""
    where = "row %s attempt %d" % (row_id, item["attempt"])
    wrapper = item.get("record") or {}
    verdict = as_seconds(item["verdict"].get("seconds"), where) if item["verdict"] is not None else None
    started = item["start"] if (item["start"] or {}).get("started_at") else {"started_at": wrapper.get("started_at")}
    state = wrapper.get("state")
    if wrapper and (state in ("launching", "running") or (state == "unverified" and not wrapper.get("verified_idle_at"))):
        elapsed = (now - _started(started, where)).total_seconds()
        seconds = max(Fraction(str(max(0.0, elapsed))), verdict or Fraction(0))
        return seconds, "running" if state != "unverified" else "workers not verified terminated"
    if wrapper.get("charged_until"):
        until = datetime.strptime(wrapper["charged_until"], STARTED_AT_FORMAT).replace(tzinfo=timezone.utc)
        elapsed = Fraction(str(max(0.0, (until - _started(started, where)).total_seconds())))
        return max(elapsed, verdict or Fraction(0)), "until verified idle (%s)" % (state or "ended")
    if verdict is not None:
        return verdict, "verdict"
    if wrapper.get("ended_at") and "elapsed_seconds" in wrapper:
        return as_seconds(wrapper["elapsed_seconds"], where + " (wrapper record)"), "ended without a runner verdict"
    elapsed = (now - _started(started, where)).total_seconds()
    if elapsed < 0:
        raise Unavailable("%s started_at is after now; the clocks disagree" % where)
    return Fraction(str(elapsed)), "running"


def baseline_gpus(work: Path, row_id: str, baseline=None) -> int:
    """The GPU count of a training row: the verified baseline's n_gpus (`knobs.NGPU` of kit/p4_recipe.py's frozen
    baseline.json, cross-checked there against the pilot's run-summary n_gpus). Unavailable when absent (round-3
    should-fix: the campaign does not declare a number)."""
    path = Path(baseline) if baseline else Path(work) / BASELINE
    doc = _json_file(path)
    entry = ((doc or {}).get("rows") or {}).get(row_id) if isinstance(doc, dict) else None
    ngpu = str(((entry or {}).get("knobs") or {}).get("NGPU", ""))
    if not isinstance(doc, dict) or doc.get("ok") is not True or not isinstance(entry, dict) or not ngpu.isdigit() or int(ngpu) < 1:
        raise Unavailable("row %s: its GPU count is the verified baseline's n_gpus, and %s records none" % (row_id, path))
    return int(ngpu)


def block_state(work: Path, campaign: dict, block: str, now: datetime, baseline=None) -> dict:
    """What a block has spent and still owes, from its own rows' attempt records only."""
    rows = {rid: dict(meta) for rid, meta in p4_rows(campaign).items() if meta["block"] == block}
    spent, qualification_spent = Fraction(0), Fraction(0)
    attempts, passed, counts, histories, launches = [], {}, {}, {}, {}
    for rid, meta in rows.items():
        history = attempt_history(work, campaign["name"], rid)
        histories[rid] = history
        counts[rid] = len(history)
        launches[rid] = [h for h in history if h["launched"]]
        passed[rid] = any((h["verdict"] or {}).get("verdict") == "PASS" for h in history)
        if meta["gpus"] == FROM_BASELINE and history:
            meta["gpus"] = baseline_gpus(work, rid, baseline)
        for item in history:
            seconds, state = attempt_seconds(work, campaign["name"], rid, history, item, now)
            hours = seconds * meta["gpus"] / 3600
            spent += hours
            if meta["kind"] == "qualification":
                qualification_spent += hours
            attempts.append({"row": rid, "attempt": item["attempt"], "kind": meta["kind"], "gpus": meta["gpus"],
                             "verdict": (item["verdict"] or {}).get("verdict", "RUNNING" if state == "running" else "NO_VERDICT"),
                             "running": state == "running", "accounted_by": state, "launched": item["launched"],
                             "seconds": str(seconds), "gpu_hours": exact(hours)})
    training = sorted((rid for rid, m in rows.items() if m["kind"] == "training"), key=lambda r: rows[r]["order"])
    retries_used = sum(max(0, len(launches[rid]) - 1) for rid in training)
    return {
        "block": block, "rows": rows, "attempts": attempts, "attempt_counts": counts, "passed": passed,
        "histories": histories, "launches": {rid: len(v) for rid, v in launches.items()}, "training_order": training,
        "spent": spent, "qualification_spent": qualification_spent,
        "outstanding_training": [rid for rid in training if not passed[rid]],
        "outstanding_scorings": [rid for rid, m in rows.items() if m["kind"] in ("scoring", "direct") and not passed[rid]],
        "retries_used": retries_used, "retry_reserve": 1 if retries_used == 0 else 0,
    }


def load_reservation(path: Path) -> dict:
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise Unavailable("reservation %s cannot be read: %s" % (path, error)) from None
    if document.get("schema") != SCHEMA:
        raise Unavailable("reservation %s: schema %r, not %r" % (path, document.get("schema"), SCHEMA))
    t_value, e_value = parse_exact(document.get("T"), "T"), parse_exact(document.get("E"), "E")
    allowance = parse_exact((document.get("blocks") or {}).get(PREVENTION, {}).get("qualification_allowance"),
                            "qualification_allowance")
    if allowance != CEILING[PREVENTION] - 9 * t_value - 45 * e_value:
        raise Unavailable("reservation %s: qualification_allowance is not 75 - 9T - 45E" % path)
    bridges = (document.get("blocks") or {}).get(BRIDGES) or {}
    t_max = parse_exact(document["T_max"], "T_max") if document.get("T_max") is not None else None
    if t_max is not None and bridges.get("may_launch") is not (7 * t_max + 20 * e_value <= CEILING[BRIDGES]):
        raise Unavailable("reservation %s: bridges may_launch does not equal 7*T_max + 20E <= 125" % path)
    return {"T": t_value, "E": e_value, "T_max": t_max, "allowance": allowance,
            "bridges_may_launch": bridges.get("may_launch") is True, "bridges_reason": bridges.get("reason")}


def seconds_limit(hours: Fraction, gpus: int) -> int:
    """floor(hours * 3600 / gpus), never below 0."""
    return max(0, math.floor(hours * 3600 / gpus))


def run_limit(hours: Fraction, gpus: int) -> int:
    """The run-time limit of a row: seconds_limit reduced by the shutdown allowance, so that limit + allowance fits in the
    hours the ledger leaves (amendment 3 B3); never below 0."""
    return max(0, seconds_limit(hours, gpus) - GRACE_SECONDS)


def limit_cap(row_id: str):
    """The cap KIT_P4_LIMIT_CAP sets on this row ("row=seconds,..."), or None. It can only lower a limit."""
    for part in (os.environ.get(LIMIT_CAP_ENV) or "").split(","):
        name, _sep, value = part.strip().partition("=")
        if name == row_id and value.strip().isdigit():
            return int(value.strip())
    return None


def _inequality(text: str, lhs: Fraction, op: str, rhs: Fraction) -> dict:
    holds = {"<=": lhs <= rhs, "<": lhs < rhs}[op]
    return {"rule": text, "lhs": exact(lhs), "op": op, "rhs": exact(rhs), "holds": holds,
            "printed": "%s: %s %s %s -> %s" % (text, float(lhs), op, float(rhs), "holds" if holds else "FAILS")}


def evaluate_gate(work: Path, campaign_path: Path, reservation_path: Path, block: str, row_id: str,
                  now: datetime | None = None, workers_verified: bool = False, attempt=None, baseline=None) -> dict:
    """The gate decision as data: {exit, allowed, reason, limit_seconds, inequalities, state...}. Never raises
    Unavailable: it becomes exit 2 with its reason. Exit 1 ("budget-incomplete") is a budget stop."""
    now = now or datetime.now(timezone.utc)
    result = {"time": now.strftime(STARTED_AT_FORMAT), "block": block, "row": row_id, "attempt": attempt, "allowed": False,
              "exit": EXIT_UNAVAILABLE, "reason": None, "limit_seconds": None, "grace_seconds": GRACE_SECONDS, "inequalities": []}
    try:
        if block not in BLOCKS:
            raise Unavailable("block %r is not one of %s" % (block, BLOCKS))
        campaign = load_campaign(campaign_path)
        rows = p4_rows(campaign)
        if row_id not in rows:
            raise Unavailable("row %s carries no P4_BLOCK in %s" % (row_id, campaign_path))
        meta = rows[row_id]
        if meta["block"] != block:
            raise Unavailable("row %s belongs to block %s, not %s" % (row_id, meta["block"], block))
        if meta["gpus"] == FROM_BASELINE:
            meta = {**meta, "gpus": baseline_gpus(Path(work), row_id, baseline)}
        result.update({"kind": meta["kind"], "gpus": meta["gpus"], "campaign": campaign["name"]})
        if meta["kind"] not in GPU_KINDS:
            raise Unavailable("row %s is of kind %s; the gate admits GPU rows only (%s)" % (row_id, meta["kind"], ", ".join(GPU_KINDS)))
        stop = read_stop(work, block)
        if stop is not None:                                   # B3: durable, retries included, whatever the arithmetic
            result.update({"allowed": False, "exit": EXIT_NOT_ALLOWED, "stop": stop,
                           "reason": "%s: block %s stopped at %s by %s (%s); no later GPU row of the block runs"
                           % (BUDGET_INCOMPLETE, block, stop.get("time"), stop.get("row"), stop.get("reason"))})
            return result
        limited = block_limit_evidence(Path(work), block)
        if limited:                                            # H2: independently of the attempts' class and of the stop file
            result.update({"allowed": False, "exit": EXIT_NOT_ALLOWED, "limit_evidence": [list(x) for x in limited],
                           "reason": "%s: block %s holds a spending-limit termination (%s); no later GPU row of the block runs"
                           % (BUDGET_INCOMPLETE, block, "; ".join("row %s attempt %s: %s" % x for x in limited[:3]))})
            return result
        try:
            state = block_state(Path(work), campaign, block, now, baseline)
        except Unavailable as error:
            raise Unavailable("%s: %s" % (COMPLIANCE_UNAVAILABLE, error)) from None
        reservation = load_reservation(reservation_path)
        result.update({"spent": exact(state["spent"]), "qualification_spent": exact(state["qualification_spent"]),
                       "outstanding_training": len(state["outstanding_training"]),
                       "outstanding_scorings": len(state["outstanding_scorings"]),
                       "retry_reserve": state["retry_reserve"], "retries_used": state["retries_used"],
                       "running_attempts": [a for a in state["attempts"] if a["running"]],
                       "running_note": "attempts without a verdict are counted from start.json started_at to now, unless "
                                       "the wrapper recorded their end or a later attempt verified the GPUs idle"})
        ceiling, e_value = CEILING[block], reservation["E"]
        if state["passed"][row_id] and meta["trains"]:
            raise Unavailable("row %s already has a PASS attempt; a completed valid checkpoint is never retrained" % row_id)
        if meta['kind'] == 'selftest':
            # The adversarial one-minute timeout is inside a fixed 300-second
            # admission, charged to prevention, and never treated as training/scoring.
            if block != PREVENTION:
                raise Unavailable('containment selftest belongs to prevention')
            check = _inequality('selftest 300 s plus shutdown fits prevention ceiling', state['spent'] + Fraction(360, 3600), '<=', ceiling)
            result['inequalities'].append(check)
            result = _decide(result, [check], ceiling-state['spent'], 1, row_id)
            result['limit_seconds'] = min(300, result['limit_seconds'])
            return result
        if meta["kind"] == "qualification":
            if block != PREVENTION:
                raise Unavailable("the registered text has qualification runs only in the prevention screen")
            if meta["trains"] and state["launches"][row_id] >= 1:
                raise Unavailable("row %s: qualification training is never retried (amendment 3 B5); its attempt %d "
                                  "launched already" % (row_id, [h for h in state["histories"][row_id] if h["launched"]][0]["attempt"]))
            check = _inequality("qualification_spent < qualification_allowance", state["qualification_spent"], "<",
                                reservation["allowance"])
            result["inequalities"].append(check)
            return _decide(result, [check], reservation["allowance"] - state["qualification_spent"], meta["gpus"], row_id)
        if meta["kind"] in ("scoring", "direct"):
            others = [r for r in state["outstanding_scorings"] if r != row_id]
            check = _inequality("this scoring and the other outstanding scorings (%d) fit: (others + 1) * E <= %s - spent"
                                % (len(others), ceiling), (len(others) + 1) * e_value, "<=", ceiling - state["spent"])
            result["inequalities"].append(check)
            return _decide(result, [check], ceiling - state["spent"] - len(others) * e_value, meta["gpus"], row_id)
        # ---- training
        launched = [h for h in state["histories"][row_id] if h["launched"]]
        if len(launched) >= 2:
            raise Unavailable("row %s has launched %d attempts; at most one training retry per block is permitted, so a "
                              "second failure is final and the block is technically incomplete" % (row_id, len(launched)))
        if state["retries_used"] > 1:
            raise Unavailable("block %s has used %d training retries; at most one is permitted, so the block is "
                              "technically incomplete" % (block, state["retries_used"]))
        if launched:
            judged = eligibility(Path(work), launched[0], workers_verified, baseline)
            result["retry"] = judged
            if judged["eligible"] is not True:
                raise Unavailable("row %s may not be retried: %s" % (row_id, judged["why"]))
            if state["retries_used"] >= 1:
                raise Unavailable("block %s has used its one training retry; row %s may not be retried" % (block, row_id))
            first_slot = None
            for rid in state["training_order"]:
                tries = [h for h in state["histories"][rid] if h["launched"]]
                if not state["passed"][rid] and len(tries) == 1:
                    if eligibility(Path(work), tries[0], workers_verified if rid == row_id else bool((tries[0].get("record") or {}).get("verified_idle_at")),
                                   baseline)["eligible"] is True:
                        first_slot = rid
                        break
            if first_slot != row_id:
                raise Unavailable("the block's one retry goes to the first eligible failed slot in the registered order "
                                  "(%s), not %s (amendment 3 B5)" % (first_slot, row_id))
        checks = []
        if block == BRIDGES:
            if reservation["T_max"] is None:
                raise Unavailable("T_max is unavailable in the reservation (%s); the bridges block is not run"
                                  % reservation["bridges_reason"])
            t_block = reservation["T_max"]
            checks.append(_inequality("bridges launch: 7*T_max + 20E <= 125", 7 * t_block + 20 * e_value, "<=", ceiling))
        else:
            t_block = reservation["T"]
            training_rows = [rid for rid, m in state["rows"].items() if m["kind"] == "training"]
            first = all(state["attempt_counts"][rid] == 0 for rid in training_rows if rid != row_id) \
                and state["launches"][row_id] == 0
            result["first_training_row"] = first
            if first:
                qualification = [rid for rid, m in state["rows"].items() if m["kind"] == "qualification"]
                if not qualification:
                    raise Unavailable("the prevention block has no qualification row; training follows a successful "
                                      "qualification")
                failed = [rid for rid in qualification if not state["passed"][rid]]
                if failed:
                    raise Unavailable("qualification is not successful: no PASS attempt for %s" % ", ".join(failed))
                checks.append(_inequality("first prevention training: G + 9T + 45E <= 75",
                                          state["qualification_spent"] + 9 * t_block + 45 * e_value, "<=", ceiling))
        outstanding_training, outstanding_scorings = len(state["outstanding_training"]), len(state["outstanding_scorings"])
        checks.append(_inequality(
            "outstanding_training(%d)*T_block + retry_reserve(%d)*T_block + outstanding_scorings(%d)*E <= %s - spent"
            % (outstanding_training, state["retry_reserve"], outstanding_scorings, ceiling),
            (outstanding_training + state["retry_reserve"]) * t_block + outstanding_scorings * e_value, "<=",
            ceiling - state["spent"]))
        result["inequalities"].extend(checks)
        result["T_block"] = exact(t_block)
        return _decide(result, checks, ceiling - state["spent"] - outstanding_scorings * e_value, meta["gpus"], row_id)
    except Unavailable as error:
        result.update({"exit": EXIT_UNAVAILABLE, "allowed": False, "reason": "unavailable: %s" % error})
        return result


def _decide(result: dict, checks: list, hours: Fraction, gpus: int, row_id: str) -> dict:
    limit = run_limit(hours, gpus)
    result["ledger_seconds"] = seconds_limit(hours, gpus)
    cap = limit_cap(row_id)
    if cap is not None and cap < limit:
        result["limit_capped"] = {"env": LIMIT_CAP_ENV, "cap_seconds": cap, "uncapped_limit_seconds": limit}
        limit = cap
    result["limit_seconds"] = limit
    failed = [c for c in checks if not c["holds"]]
    if failed:
        result.update({"allowed": False, "exit": EXIT_NOT_ALLOWED,
                       "reason": "%s: %s" % (BUDGET_INCOMPLETE, "; ".join(c["printed"] for c in failed))})
    elif limit < 1:                                            # no admitted run time is left
        result.update({"allowed": False, "exit": EXIT_NOT_ALLOWED,
                       "reason": "%s: the runtime limit is %d seconds after the %d-second shutdown allowance: no run time is "
                                 "left to admit" % (BUDGET_INCOMPLETE, limit, GRACE_SECONDS)})
    else:
        result.update({"allowed": True, "exit": EXIT_OK, "reason": None})
    return result


def append_ledger(work: Path, entry: dict) -> Path:
    path = Path(work) / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"schema": SCHEMA, **entry}, sort_keys=True, default=str) + "\n")
    return path


def decide(work: Path, campaign_path: Path, reservation_path: Path, block: str, row_id: str, now: datetime | None = None,
           workers_verified: bool = False, attempt=None, baseline=None) -> dict:
    """evaluate_gate, then one ledger line, then -- on a budget refusal (exit 1) -- the durable stop record."""
    result = evaluate_gate(Path(work), Path(campaign_path), Path(reservation_path), block, row_id, now,
                           workers_verified=workers_verified, attempt=attempt, baseline=baseline)
    append_ledger(work, {k: v for k, v in result.items() if k != "stop"} | ({"stop_seen": True} if "stop" in result else {}))
    if result["exit"] == EXIT_NOT_ALLOWED:
        result["stop_written"] = record_stop(work, block, {"time": result["time"], "row": row_id, "attempt": attempt,
                                                           "reason": result["reason"], "cause": "gate refusal",
                                                           "spent": result.get("spent"), "inequalities": result["inequalities"],
                                                           "limit_seconds": result.get("limit_seconds")})
    return result


def gate(work: Path, campaign_path: Path, reservation_path: Path, block: str, row_id: str, print_limit: bool = False,
         now: datetime | None = None, out=None, err=None, workers_verified: bool = False, attempt=None) -> int:
    """The CLI gate: decide, append one ledger line, write the stop on a budget refusal, print. With --print-limit,
    stdout carries ONE integer when the row is allowed and nothing otherwise; every message then goes to stderr."""
    out, err = out or sys.stdout, err or sys.stderr
    if not Path(work).is_dir():
        print("unavailable: WORK %s is not a directory" % work, file=err if print_limit else out)
        return EXIT_UNAVAILABLE
    result = decide(Path(work), Path(campaign_path), Path(reservation_path), block, row_id, now, workers_verified, attempt)
    talk = err if print_limit else out
    for check in result["inequalities"]:
        print(check["printed"], file=talk)
    if result["exit"] == EXIT_OK:
        if print_limit:
            print(result["limit_seconds"], file=out)
        else:
            print("allowed %s (%s): runtime limit %d seconds (after the %d-second grace)"
                  % (row_id, block, result["limit_seconds"], GRACE_SECONDS), file=out)
    else:
        print(result["reason"], file=talk)
    return result["exit"]


# ------------------------------------------------------------------------------------ status
def compute_status(work: Path, campaign_path: Path, reservation_path: Path, now: datetime | None = None, baseline=None) -> dict:
    now = now or datetime.now(timezone.utc)
    reservation = load_reservation(reservation_path)
    campaign = load_campaign(campaign_path)
    blocks, total_spent, total_reserved, unavailable = {}, Fraction(0), Fraction(0), []
    for block in BLOCKS:
        stop = read_stop(work, block)
        try:
            state = block_state(Path(work), campaign, block, now, baseline)
        except Unavailable as error:
            blocks[block] = {"ceiling": str(CEILING[block]), "accounting": COMPLIANCE_UNAVAILABLE, "why": str(error),
                             "spent": None, "reserved": None, "remaining": None, "stop": stop,
                             "label": COMPLIANCE_UNAVAILABLE}
            unavailable.append(block)
            continue
        t_block = reservation["T"] if block == PREVENTION else reservation["T_max"]
        # round-4 ruling G1: an attempt whose termination could not be verified leaves the block's accounting
        # unavailable (its stop says so, or an attempt is still unverified); that is not a demonstrated overrun
        unverified = [a for a in state["attempts"] if a["accounted_by"] == "workers not verified terminated"]
        if (stop or {}).get("accounting") == "unavailable" or unverified:
            blocks[block] = {"ceiling": str(CEILING[block]), "accounting": COMPLIANCE_UNAVAILABLE, "spent": exact(state["spent"]),
                             "why": "workers not verified terminated: %s" % ((stop or {}).get("reason") or ", ".join(
                                 "%s attempt %s" % (a["row"], a["attempt"]) for a in unverified)),
                             "reserved": None, "remaining": None, "stop": stop, "over_ceiling": None, "label": COMPLIANCE_UNAVAILABLE}
            unavailable.append(block)
            total_spent += state["spent"]
            continue
        limited = block_limit_evidence(Path(work), block)
        entry = {"ceiling": str(CEILING[block]), "spent": exact(state["spent"]), "accounting": "available",
                 "outstanding_training": len(state["outstanding_training"]),
                 "outstanding_scorings": len(state["outstanding_scorings"]), "retry_reserve": state["retry_reserve"],
                 "retries_used": state["retries_used"], "running_attempts": sum(a["running"] for a in state["attempts"]),
                 "stop": stop, "over_ceiling": state["spent"] > CEILING[block],
                 "limit_evidence": [list(x) for x in limited],
                 "label": BUDGET_INCOMPLETE if (stop is not None or limited or state["spent"] > CEILING[block]) else None}
        if block == PREVENTION:
            entry["qualification_spent"] = exact(state["qualification_spent"])
            entry["qualification_allowance"] = exact(reservation["allowance"])
        total_spent += state["spent"]
        if t_block is None:
            entry.update({"reserved": None, "remaining": None, "reason": "T_max unavailable"})
            unavailable.append(block)
        else:
            reserved = ((len(state["outstanding_training"]) + state["retry_reserve"]) * t_block
                        + len(state["outstanding_scorings"]) * reservation["E"])
            entry.update({"reserved": exact(reserved), "remaining": exact(CEILING[block] - state["spent"] - reserved)})
            total_reserved += reserved
        blocks[block] = entry
    package = {"ceiling": str(PACKAGE_CEILING), "spent": exact(total_spent),
               "reserved": None if unavailable else exact(total_reserved),
               "remaining": None if unavailable else exact(PACKAGE_CEILING - total_spent - total_reserved)}
    return {"time": now.strftime(STARTED_AT_FORMAT), "blocks": blocks, "package": package,
            "ignored_rows": ignored_rows(campaign), "unavailable": unavailable, "grace_seconds": GRACE_SECONDS,
            "running_note": "attempts without a verdict are counted from start.json started_at to now, unless the wrapper "
                            "recorded their end or a later attempt verified the GPUs idle"}


def status(work: Path, campaign_path: Path, reservation_path: Path, as_json: bool = False,
           now: datetime | None = None, out=None) -> int:
    out = out or sys.stdout
    try:
        document = compute_status(Path(work), Path(campaign_path), Path(reservation_path), now)
    except Unavailable as error:
        print("unavailable: %s" % error, file=out)
        return EXIT_UNAVAILABLE
    if as_json:
        print(json.dumps(document, indent=1, sort_keys=True), file=out)
    else:
        for block, entry in document["blocks"].items():
            print("%-10s spent %s  reserved %s  remaining %s  (ceiling %s)%s" % (
                block, (entry["spent"] or {}).get("float", COMPLIANCE_UNAVAILABLE), (entry["reserved"] or {}).get("float", "unavailable"),
                (entry["remaining"] or {}).get("float", "unavailable"), entry["ceiling"],
                "  STOPPED: %s" % entry["stop"].get("reason") if entry.get("stop") else ""), file=out)
        package = document["package"]
        print("%-10s spent %s  reserved %s  remaining %s  (ceiling %s)" % (
            "package", package["spent"]["float"], (package["reserved"] or {}).get("float", "unavailable"),
            (package["remaining"] or {}).get("float", "unavailable"), package["ceiling"]), file=out)
    return EXIT_UNAVAILABLE if document["unavailable"] else EXIT_OK


# ------------------------------------------------------------------------------------ CLI
def main(argv=None, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_reserve = sub.add_parser("reserve")
    p_reserve.add_argument("--work", required=True)
    p_reserve.add_argument("--pilot-campaign", required=True)
    p_reserve.add_argument("--selection", required=True)
    p_reserve.add_argument("--out", required=True)
    p_gate = sub.add_parser("gate")
    p_gate.add_argument("--work", required=True)
    p_gate.add_argument("--campaign", required=True)
    p_gate.add_argument("--reservation", required=True)
    p_gate.add_argument("--block", required=True, choices=BLOCKS)
    p_gate.add_argument("--row", required=True)
    p_gate.add_argument("--attempt", type=int)
    p_gate.add_argument("--print-limit", action="store_true")
    p_status = sub.add_parser("status")
    p_status.add_argument("--work", required=True)
    p_status.add_argument("--campaign", required=True)
    p_status.add_argument("--reservation", required=True)
    p_status.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "reserve":
        stale = _frozen().verify(Path(args.work), need=("selection",), containment=False)
        if stale:
            print("refused: %s" % _frozen().refusal(stale), file=sys.stderr)
            return EXIT_UNAVAILABLE
        return reserve(Path(args.work), Path(args.pilot_campaign), Path(args.selection), Path(args.out))
    # round-3 ruling F2: the frozen decision inputs first, whatever the runner skipped
    stale = _frozen().verify(Path(args.work), reservation=args.reservation, need=("selection", "reservation"))
    if stale:
        print("refused: %s" % _frozen().refusal(stale), file=sys.stderr)
        return EXIT_UNAVAILABLE
    if args.command == "gate":
        return gate(Path(args.work), Path(args.campaign), Path(args.reservation), args.block, args.row,
                    args.print_limit, now, attempt=args.attempt)
    return status(Path(args.work), Path(args.campaign), Path(args.reservation), args.json, now)


if __name__ == "__main__":
    sys.exit(main())
