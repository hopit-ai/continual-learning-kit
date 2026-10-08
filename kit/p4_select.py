#!/usr/bin/env python3
"""The selection of package 4, run once, by rule: which two lineages the prevention screen and the bridges start from.

    python p4_select.py --work $WORK --pilot-report $WORK/k8b/report-pilot-a<N>/pilot-report.json --out $WORK/k8b4/selection

Implements sections 1 (entry) and 2 (selection) of docs/phase2/plan-v3-package4-preregistration-20261002.md and
nothing else, and the selector's verification duties of its section 5. It implements the text; it does not improve it.

REFUSED (exit 2, a message on stderr, nothing written):
  - an existing <out>/selection.json; a pilot report that is unreadable or not of schema kit-pilot-report.v1;
  - unless the report's constants are the registered ones: `serving` 2048, `diagnostic` 8192, `acquired_per_100` 5
    (the acquisition threshold of 5 per 100) and `different_machines_allowed` false (0);
  - a report with no `decision` block or no `decision.on_to_package_4` key;
  - an eligible group whose two retention rows (`retention`, keyed by recipe, chain, replicate) lack an integer F or a
    positive integer n, are not `defined`, or do not lose 5 per 100 (100 F >= 5 n) although the decision says they do
    (the report contradicts itself: fail closed), or a group outside g8/sema x chemtool/toolchem (no tie rule exists).

ENTRY (section 1). `decision.on_to_package_4` must be exactly true. False or null writes selection.json with
`entered` false, the reason and no lineages, and exits 1.

SELECTION (section 2). Eligible groups are the entries of `decision.loss_repeats` with `both_defined` true and
`both_lose_5_per_100` true. Each eligible group gets two exact values 100 F / n (Fractions), one per run r1 and r2,
from the report's `retention` rows. Groups are ranked by the SMALLER of their two values, largest first; ties take g8
before sema, then chemtool before toolchem. BOTH runs r1 and r2 of the first group are selected. The larger single
replicate is never selected instead.

WRITTEN: <out>/selection.json with schema, entered, group {recipe, chain}, tasks {A, B} (chemtool: A chemistry, B
toolalpaca; toolchem: the reverse), lineages {r1: {stage1, pilot_stage2}, r2: {...}}, the full ranking (each group's
two values as strings), every group with eligible true/false and why, the pilot report's path and sha256, the verified
constants, and for each selected stage-1 and stage-2 run its run folder of record and run-summary.json sha256 when
present under WORK/runs (the pilot's rule, kit/pilot_report.py `Work.merged_run`: the highest attempt whose
run-summary.json says merged 1). An absent run is recorded as absent, not refused. `content_sha256` is the sha256 of
the content with sorted keys; `generated_at` is outside it, so two runs on one input differ only there.

BOUND AND FROZEN (send-5 review round 2, finding 8; docs/phase2/plan-v3-package4-amendment3-20261004.md B7). Before
selecting, the selector REFUSES (exit 2, nothing written) unless:
  - --pilot-campaign is the pilot campaign of public tag kit-batch3-v1 (commit PILOT_TAG_COMMIT): its sha256, computed
    as kit/runner.py computes `campaign_sha256` (sha256 of the file's text, UTF-8), is PILOT_CAMPAIGN_SHA256, and EVERY
    start.json of the pilot's attempts (WORK/campaign/<name>/<row>/attempt-*/start.json) records that campaign_sha256;
  - no attempt of a scheduled pilot row is unresolved: each row's latest attempt has a verdict.json (a row the runner
    never launched is recorded as "not attempted");
  - --pilot-report is the pilot-report.json of the HIGHEST report attempt folder WORK/k8b/report-pilot-a<N>;
  - every pilot scoring USED TO ESTABLISH entry or selection -- the sweeps the report cites in `retention`,
    `acquisition` and `base` -- is on ONE physical GPU by the recomputed registered fingerprint (kit/p4_report.py
    `recompute_machine_id`: fields present, at least one GPU listed), from the sweep and, when present, its bed-score.json.
It RECOMPUTES entry and selection from the frozen inputs (round-3 ruling F2): kit/pilot_report.py is run on the tree
into a scratch folder outside WORK and its `decision` block and `retention` rows must equal the supplied report's
(WORK paths written as <WORK>), or it refuses: the supplied report is never trusted on its own.
It then writes the FINALIZATION MANIFEST <out>/pilot-finalization.json: every scheduled row's terminal disposition and
attempt; every decision and recipe input with its sha256 (`inputs`) and the patterns that found them (`patterns`): the
pilot reports, every pilot sweep, per-item file, bed-score.json, forgetting.json, prefix check and rollout statistics,
every pilot run's run-summary.json and env/argv.txt, sdpo-commit.txt, sdpo-dirty.txt, data-sha256.txt and
model-files.txt, and every runner record (start.json, verdict.json) of the pilot campaign; the pilot campaign file's
sha256; the report's sha256; the recomputation; the scorings used and their machine id. Then the selection, bound to the
manifest's content_sha256. kit/p4_frozen.py re-verifies all of it before every dependent action.
The FIRST selection is THE selection: <out>/selection.json (one per WORK, not per attempt). A later invocation
recomputes manifest and selection; identical canonical content (content_sha256, which leaves out `generated_at` and
`located_at`, the only places absolute paths appear) prints "selection unchanged" and exits as the first did; anything
else exits 2, "the pilot's inputs changed after selection", and nothing is overwritten. Paths inside the content are
relative to WORK, so the same check passes on an extracted archive at another path.

Standard library only (PyYAML through kit/runner.py to read the pilot campaign); kit/pilot_report.py, kit/runner.py
and kit/p4_report.py are loaded by file path, so `kit/` works when exported alone.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "kit-p4-selection.v1"
PILOT_SCHEMA = "kit-pilot-report.v1"
FILE_NAME = "selection.json"
CONSTANTS = {"serving": 2048, "diagnostic": 8192, "acquired_per_100": 5, "different_machines_allowed": False}
LOSS_PER_100 = 5
RECIPE_ORDER = ("g8", "sema")                 # the tie rule of section 2, first criterion
CHAIN_ORDER = ("chemtool", "toolchem")        # second criterion
REPLICATES = ("r1", "r2")
TASKS_OF = {"chemtool": {"A": "chemistry", "B": "toolalpaca"}, "toolchem": {"A": "toolalpaca", "B": "chemistry"}}
FIRST_CHAIN = {"chemtool": "chem", "toolchem": "tool"}
#: the pilot campaign of public tag kit-batch3-v1 (amendment 3 B7; docs/phase2/plan-v3-evidence/kit-batch3-v1-pilot-campaign-hash.txt)
PILOT_TAG = "kit-batch3-v1"
PILOT_TAG_COMMIT = "671b2596b89fee0f040a72112290adaeabd4ede2"
PILOT_CAMPAIGN_FILE = "kit/campaigns/k8b-pilot.yaml"
PILOT_CAMPAIGN_SHA256 = "93ee04bd3d4970f9965264075a857a19893219a033be459aff3d21438befbe42"
FINALIZATION_SCHEMA = "kit-p4-pilot-finalization.v1"
FINALIZATION_NAME = "pilot-finalization.json"
USED_SECTIONS = ("retention", "acquisition", "base")


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_p4_select_%s" % name, HERE / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot_report = _load("pilot_report")
runner = _load("runner")
frozen_inputs = _load("p4_frozen")
#: the decision inputs of the pilot, by pattern relative to WORK (round-3 ruling F2): everything kit/pilot_report.py
#: reads, the raw scorings behind the sweeps, and every runner record of the pilot campaign; the pilot runs' own files
#: are added per scheduled run key (RUN_FILES)
DECISION_PATTERNS = ("k8b/report-pilot-a*/pilot-report.json", "k8b/report-sweep/*/sweep.json", "k8b/report-sweep/*/per_item.jsonl",
                     "k8b/report-prefix/*.json", "k8b/report-rollouts/*/rollout-stats.json", "k8b/eval/*/bed-score.json",
                     "k8b/forgetting/*/forgetting.json")
RUN_FILES = ("run-summary.json", "env/argv.txt", "env/sdpo-commit.txt", "env/sdpo-dirty.txt", "env/data-sha256.txt", "env/model-files.txt")


class SelectionRefused(ValueError):
    """The pilot report cannot support a selection; nothing is written."""


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha256(content: dict) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_constants(report: dict) -> dict:
    """The registered constants as the report states them; SelectionRefused on any difference (section 5, selector)."""
    found = {}
    for name in ("serving", "diagnostic", "acquired_per_100"):
        value = report.get(name)
        if not _number(value) or value != CONSTANTS[name]:
            raise SelectionRefused("the pilot report's %s is %r, not the registered %r" % (name, value, CONSTANTS[name]))
        found[name] = CONSTANTS[name]
    allowed = report.get("different_machines_allowed")
    if allowed is None or allowed not in (0, False) or isinstance(allowed, float):
        raise SelectionRefused("the pilot report's different_machines_allowed is %r: cross-machine comparison must not have "
                               "been enabled (registered: false)" % (allowed,))
    found["different_machines_allowed"] = False
    return found


def retention_row(report: dict, recipe: str, chain: str, rep: str) -> dict:
    rows = [r for r in report.get("retention") or [] if isinstance(r, dict)
            and r.get("recipe") == recipe and r.get("chain") == chain and r.get("replicate") == rep]
    if len(rows) != 1:
        raise SelectionRefused("the pilot report has %d retention rows for %s %s %s (expected one)" % (len(rows), recipe, chain, rep))
    return rows[0]


def loss_value(report: dict, recipe: str, chain: str, rep: str) -> dict:
    """{F, n, value} with value = 100 F / n exactly, from the report's retention row; refuses a missing or inconsistent row."""
    row = retention_row(report, recipe, chain, rep)
    F, n = row.get("F"), row.get("n")
    if not _int(F) or not _int(n) or n <= 0:
        raise SelectionRefused("the retention row %s-%s-%s lacks an integer F or a positive integer n (F %r, n %r)"
                               % (recipe, chain, rep, F, n))
    if row.get("defined") not in (1, True):
        raise SelectionRefused("the retention row %s-%s-%s is not defined, but decision.loss_repeats says both runs of the "
                               "group are: the report contradicts itself" % (recipe, chain, rep))
    if 100 * F < LOSS_PER_100 * n:
        raise SelectionRefused("the retention row %s-%s-%s loses %d of %d (under 5 per 100), but decision.loss_repeats says "
                               "both runs lose 5 per 100: the report contradicts itself" % (recipe, chain, rep, F, n))
    return {"F": F, "n": n, "value": Fraction(100 * F, n)}


def classify(report: dict) -> tuple:
    """(eligible [(recipe, chain)], groups [{recipe, chain, eligible, why}]) from decision.loss_repeats."""
    entries = (report.get("decision") or {}).get("loss_repeats")
    if not isinstance(entries, list):
        raise SelectionRefused("the pilot report's decision.loss_repeats is not a list")
    eligible, groups, seen = [], [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise SelectionRefused("decision.loss_repeats holds a non-object entry %r" % (entry,))
        recipe, chain = entry.get("recipe"), entry.get("chain")
        if recipe not in RECIPE_ORDER or chain not in CHAIN_ORDER:
            raise SelectionRefused("decision.loss_repeats names the group %r %r; the registered tie rule orders only %s x %s"
                                   % (recipe, chain, "/".join(RECIPE_ORDER), "/".join(CHAIN_ORDER)))
        if (recipe, chain) in seen:
            raise SelectionRefused("decision.loss_repeats names %s %s twice" % (recipe, chain))
        seen.add((recipe, chain))
        why = []
        if entry.get("both_defined") is not True:
            why.append("both_defined is %s, not true" % json.dumps(entry.get("both_defined")))
        if entry.get("both_lose_5_per_100") is not True:
            why.append("both_lose_5_per_100 is %s, not true" % json.dumps(entry.get("both_lose_5_per_100")))
        groups.append({"recipe": recipe, "chain": chain, "eligible": not why,
                       "why": "eligible: both runs defined and both lose 5 per 100" if not why else "; ".join(why)})
        if not why:
            eligible.append((recipe, chain))
    groups.sort(key=lambda g: (RECIPE_ORDER.index(g["recipe"]), CHAIN_ORDER.index(g["chain"])))
    return eligible, groups


def rank(report: dict, eligible: list) -> list:
    """The eligible groups ordered by the smaller of their two exact values, largest first; ties g8, then chemtool."""
    ranked = []
    for recipe, chain in eligible:
        values = {rep: loss_value(report, recipe, chain, rep) for rep in REPLICATES}
        smaller = min(v["value"] for v in values.values())
        ranked.append({"recipe": recipe, "chain": chain, "smaller": smaller, "values": values})
    ranked.sort(key=lambda g: (-g["smaller"], RECIPE_ORDER.index(g["recipe"]), CHAIN_ORDER.index(g["chain"])))
    return [{"rank": i + 1, "recipe": g["recipe"], "chain": g["chain"], "smaller_value": str(g["smaller"]),
             "smaller_value_decimal": "%.6f" % float(g["smaller"]),
             "values": {rep: {"value": str(v["value"]), "value_decimal": "%.6f" % float(v["value"]), "F": v["F"], "n": v["n"]}
                        for rep, v in g["values"].items()}}
            for i, g in enumerate(ranked)]


def provenance(work: Path, key: str) -> dict:
    """The run folder of record of a pilot run (relative to WORK) and its run-summary.json sha256, or `absent`."""
    runs = pilot_report.attempts(work / "runs")
    found = sorted(a for a, _ in runs.get(key, []))
    merged = pilot_report.Work(work).merged_run(key)
    if merged is None:
        return {"key": key, "status": "absent", "attempts_found": found, "attempt": None, "run": None,
                "run_summary_sha256": None, "model_dir": None,
                "why": "no attempt of %s under WORK/runs has a run-summary.json with merged 1" % key}
    attempt, folder, summary = merged
    return {"key": key, "status": "present", "attempts_found": found, "attempt": attempt, "run": "runs/%s" % folder.name,
            "run_summary_sha256": sha256_file(folder / "run-summary.json"), "model_dir": summary.get("model_dir"),
            "rule": "the highest attempt whose run-summary.json says merged 1 (kit/pilot_report.py)"}


# ------------------------------------------------------------------------------------------- finalization (B7)
def campaign_text_sha256(path: Path) -> str:
    """The sha256 kit/runner.py records as `campaign_sha256`: of the file's text, encoded as UTF-8."""
    return hashlib.sha256(Path(path).read_text().encode()).hexdigest()


def _rel(work: Path, path: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(work).resolve()))


def sweep_folders(value) -> list:
    """Every report-sweep folder name a report section cites (the path of a sweep, as kit/pilot_report.py writes it)."""
    found = []
    if isinstance(value, dict):
        for item in value.values():
            found += sweep_folders(item)
    elif isinstance(value, list):
        for item in value:
            found += sweep_folders(item)
    elif isinstance(value, str) and "/report-sweep/" in value:
        tail = value.split("/report-sweep/", 1)[1].strip("/").split("/")
        if tail and tail[0]:
            found.append(tail[0])
    return found


def finalize(work: Path, pilot_campaign: Path, report_path: Path, report: dict) -> dict:
    """The finalization manifest's content; SelectionRefused when the pilot is not the released one or not final."""
    pilot_campaign = Path(pilot_campaign)
    if not pilot_campaign.is_file():
        raise SelectionRefused("no pilot campaign file at %s" % pilot_campaign)
    digest = campaign_text_sha256(pilot_campaign)
    if digest != PILOT_CAMPAIGN_SHA256 or sha256_file(pilot_campaign) != PILOT_CAMPAIGN_SHA256:
        raise SelectionRefused("%s has sha256 %s, not %s (%s at tag %s, commit %s): it is not the released pilot campaign"
                               % (pilot_campaign, digest, PILOT_CAMPAIGN_SHA256, PILOT_CAMPAIGN_FILE, PILOT_TAG, PILOT_TAG_COMMIT))
    campaign = runner.load_campaign(pilot_campaign)
    base = work / "campaign" / campaign["name"]
    rows, unresolved, other_hash = [], [], []
    for row in campaign["rows"]:
        folder = base / row["id"]
        attempts = sorted(int(p.name.split("-")[1]) for p in folder.glob("attempt-*") if p.name.split("-")[1].isdigit()) if folder.is_dir() else []
        for number in attempts:
            start = folder / ("attempt-%d" % number) / "start.json"
            try:
                recorded = json.loads(start.read_text()).get("campaign_sha256")
            except (OSError, ValueError):
                recorded = None
            if recorded != PILOT_CAMPAIGN_SHA256:
                other_hash.append("%s attempt %d (%s)" % (row["id"], number, recorded))
        if not attempts:
            rows.append({"row": row["id"], "attempt": None, "disposition": "not attempted"})
            continue
        verdict = folder / ("attempt-%d" % attempts[-1]) / "verdict.json"
        try:
            outcome = json.loads(verdict.read_text()).get("verdict")
        except (OSError, ValueError):
            outcome = None
        if outcome not in ("PASS", "FAIL"):
            unresolved.append("%s attempt %d" % (row["id"], attempts[-1]))
        rows.append({"row": row["id"], "attempt": attempts[-1], "disposition": outcome})
    if other_hash:
        raise SelectionRefused("pilot attempts not recorded under the released campaign (campaign_sha256 differs or is "
                               "missing): %s" % "; ".join(other_hash[:5]))
    if unresolved:
        raise SelectionRefused("the pilot is not final: the latest attempt of %s has no verdict (running or unresolved)"
                               % "; ".join(unresolved[:5]))
    reports = pilot_report.attempts(work / "k8b").get("report-pilot", [])
    if not reports:
        raise SelectionRefused("no WORK/k8b/report-pilot-a<N> folder")
    highest = max(reports)[1] / "pilot-report.json"
    if Path(report_path).resolve() != highest.resolve():
        raise SelectionRefused("%s is not the pilot report of the highest report attempt (%s)" % (report_path, _rel(work, highest)))
    # every decision and recipe input, with the patterns that find them: a later file matching one is a change too
    keys = [row["id"] for row in campaign["rows"] if (row.get("env") or {}).get("NAME")]
    patterns = list(DECISION_PATTERNS) + ["campaign/%s/*/attempt-*/%s" % (campaign["name"], name) for name in ("start.json", "verdict.json")]
    patterns += ["runs/%s-a*/%s" % (key, name) for key in keys for name in RUN_FILES]
    inputs = frozen_inputs.glob_inputs(work, patterns)
    recomputed = recompute_decision(work, report)
    # the scorings that establish entry and selection: one physical GPU by the recomputed registered fingerprint
    used = sorted({name for section in USED_SECTIONS for name in sweep_folders(report.get(section))})
    if not used:
        raise SelectionRefused("the pilot report cites no sweep in %s: the scorings establishing entry and selection "
                               "cannot be identified" % ", ".join(USED_SECTIONS))
    ids, problems = {}, []
    reporter = _load("p4_report")
    for name in used:
        sweep_path = work / "k8b" / "report-sweep" / name / "sweep.json"
        sweep = _json_file(sweep_path)
        if not isinstance(sweep, dict):
            problems.append("missing: k8b/report-sweep/%s/sweep.json" % name)
            continue
        found, why = reporter.recompute_machine_id(sweep.get("machine"))
        scoring = sweep.get("scoring")
        scored = _json_file(work / "k8b" / "eval" / Path(scoring).name / "bed-score.json") if isinstance(scoring, str) and scoring else None
        if isinstance(scored, dict):
            other, why2 = reporter.recompute_machine_id(scored.get("machine"))
            if other != found:
                problems.append("%s: the sweep and its bed-score.json record different machines (%s, %s)" % (name, found, other))
        if found is None:
            problems.append("%s: %s" % (name, why))
        else:
            ids.setdefault(found, []).append(name)
    if problems or len(ids) != 1:
        raise SelectionRefused("the pilot scorings that establish entry and selection are not on ONE physical GPU by the "
                               "recomputed registered fingerprint: %s" % ("; ".join(problems[:5]) if problems else
                                                                           ", ".join("%s (%d scorings)" % (k, len(v)) for k, v in ids.items())))
    return {"schema": FINALIZATION_SCHEMA,
            "pilot_campaign": {"name": campaign["name"], "file": PILOT_CAMPAIGN_FILE, "tag": PILOT_TAG, "commit": PILOT_TAG_COMMIT,
                               "sha256": PILOT_CAMPAIGN_SHA256, "start_json_records_checked": True},
            "rows": rows, "rows_attempted": sum(r["attempt"] is not None for r in rows),
            "report": {"path": _rel(work, highest), "sha256": sha256_file(highest)},
            "patterns": patterns, "inputs": inputs, "recomputed": recomputed,
            "scorings_used_for_entry_and_selection": used, "machine_id": next(iter(ids))}


def _normal(value, roots: list):
    """A report value with the WORK root written as <WORK> (the recomputation may run on a copy at another path)."""
    if isinstance(value, dict):
        return {k: _normal(v, roots) for k, v in value.items()}
    if isinstance(value, list):
        return [_normal(v, roots) for v in value]
    if isinstance(value, str):
        for root in roots:
            value = value.replace(root, "<WORK>")
    return value


def run_pilot_report(work: Path, out: Path) -> tuple:
    """(report, None) from kit/pilot_report.py run on the pilot tree with the registered constants, or (None, why)."""
    done = subprocess.run([sys.executable, str(HERE / "pilot_report.py"), "--work", str(work), "--serving", str(CONSTANTS["serving"]),
                           "--diagnostic", str(CONSTANTS["diagnostic"]), "--out", str(out)], capture_output=True, text=True)
    report = _json_file(Path(out) / "pilot-report.json")
    if done.returncode != 0 or not isinstance(report, dict):
        return None, "kit/pilot_report.py exited %d: %s" % (done.returncode, (done.stderr or done.stdout).strip()[-300:])
    return report, None


def recompute_decision(work: Path, report: dict) -> dict:
    """Entry and selection RECOMPUTED from the frozen inputs (round-3 ruling F2, as scripts/read_send4.py does): the pilot
    report is made again from the tree by kit/pilot_report.py into a scratch folder outside WORK, and its `decision`
    block and `retention` rows must equal the supplied report's (WORK paths written as <WORK>). SelectionRefused
    otherwise: the supplied report is never trusted on its own."""
    scratch = Path(tempfile.mkdtemp(prefix="p4-select-recompute-"))
    try:
        ours, why = run_pilot_report(Path(work), scratch / "pilot")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if ours is None:
        raise SelectionRefused("the pilot report cannot be recomputed from its inputs: %s" % why)
    # each report written as <WORK> by its own recorded `work` root (the partner's report and this recomputation may run
    # on one tree at two paths: a copy, an extracted archive)
    def roots(doc):
        found = {str(Path(work)), str(Path(work).resolve())} | ({doc["work"]} if isinstance(doc.get("work"), str) and doc["work"] else set())
        return sorted(found, key=len, reverse=True)
    differing = [name for name in ("decision", "retention")
                 if _normal(ours.get(name), roots(ours)) != _normal(report.get(name), roots(report))]
    if differing:
        raise SelectionRefused("the pilot report's %s differs from kit/pilot_report.py's recomputation from the tree: the supplied "
                               "report is not the one its inputs give" % " and ".join(differing))
    return {"tool": "kit/pilot_report.py", "compared": ["decision", "retention"], "equal": True,
            "decision_sha256": frozen_inputs.content_sha256({"d": _normal(ours.get("decision"), roots(ours))}),
            "retention_sha256": frozen_inputs.content_sha256({"r": _normal(ours.get("retention"), roots(ours))})}


def _json_file(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def frozen(path: Path, content: dict, what: str):
    """None when `path` does not exist yet; True when it holds the same canonical content; SelectionRefused otherwise."""
    if not path.exists():
        return None
    old = _json_file(path)
    if not isinstance(old, dict) or old.get("content_sha256") != content_sha256(content):
        raise SelectionRefused("the pilot's inputs changed after selection: %s holds content_sha256 %s, the recomputed %s "
                               "gives %s; nothing is overwritten" % (path, str((old or {}).get("content_sha256"))[:16], what,
                                                                    content_sha256(content)[:16]))
    return True


def select(report: dict, report_path: Path, work: Path, finalization: dict | None = None) -> tuple:
    """(content, exit code): 0 for a selection, 1 when package 4 is not entered."""
    if not isinstance(report, dict) or report.get("schema") != PILOT_SCHEMA:
        raise SelectionRefused("%s is not a pilot report of schema %s" % (report_path, PILOT_SCHEMA))
    constants = verify_constants(report)
    decision = report.get("decision")
    if not isinstance(decision, dict) or "on_to_package_4" not in decision:
        raise SelectionRefused("the pilot report has no decision.on_to_package_4")
    try:
        report_rel = _rel(work, report_path)
    except ValueError:
        raise SelectionRefused("the pilot report %s is not inside WORK %s" % (report_path, work)) from None
    base = {"schema": SCHEMA, "pilot_report": {"path": report_rel, "sha256": sha256_file(report_path)}, "constants": constants,
            "finalization": {"path": "k8b4/selection/%s" % FINALIZATION_NAME, "sha256": content_sha256(finalization)} if finalization else None}
    entry = decision["on_to_package_4"]
    if entry is not True:
        reason = ("decision.on_to_package_4 is %s: package 4 is built only if it is true; the decision returns to the owner"
                  % json.dumps(entry))
        return ({**base, "entered": False, "reason": reason, "pilot_why": decision.get("why"), "group": None, "tasks": None,
                 "lineages": None, "ranking": [], "groups": []}, 1)
    eligible, groups = classify(report)
    ranking = rank(report, eligible)
    if not ranking:
        raise SelectionRefused("decision.on_to_package_4 is true but no group of decision.loss_repeats is eligible: the report "
                               "contradicts itself")
    first = ranking[0]
    recipe, chain = first["recipe"], first["chain"]
    lineages = {}
    for rep in REPLICATES:
        stage1 = "%s-%s-%s" % (recipe, FIRST_CHAIN[chain], rep)
        stage2 = "%s-%s-%s" % (recipe, chain, rep)
        lineages[rep] = {"stage1": provenance(work, stage1), "pilot_stage2": provenance(work, stage2)}
    content = {**base, "entered": True, "reason": "the first group of the ranking: %s %s (smaller value %s per 100)"
               % (recipe, chain, first["smaller_value"]),
               "pilot_why": decision.get("why"), "group": {"recipe": recipe, "chain": chain}, "tasks": dict(TASKS_OF[chain]),
               "lineages": lineages, "ranking": ranking, "groups": groups,
               "rule": "rank eligible groups by the smaller of their two exact values of 100 F / n, largest first; ties g8 "
                       "before sema, then chemtool before toolchem; select both runs r1 and r2 of the first group"}
    return content, 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Package 4 selection (registration sections 1 and 2).")
    parser.add_argument("--work", required=True, help="the pilot's WORK folder (runs/ under it)")
    parser.add_argument("--pilot-campaign", required=True, help="kit/campaigns/k8b-pilot.yaml (the released one)")
    parser.add_argument("--pilot-report", required=True, help="the finalized pilot-report.json (the highest report attempt)")
    parser.add_argument("--out", required=True, help="the selection folder, WORK/k8b4/selection")
    args = parser.parse_args(argv)
    out = Path(args.out)
    target, manifest_path = out / FILE_NAME, out / FINALIZATION_NAME
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        work = Path(args.work)
        if not work.is_dir():
            raise SelectionRefused("no such work folder: %s" % work)
        path = Path(args.pilot_report)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SelectionRefused("cannot read the pilot report %s: %s" % (path, exc)) from None
        finalization = finalize(work, Path(args.pilot_campaign), path, report)
        content, code = select(report, path, work, finalization)
        states = (frozen(manifest_path, finalization, "finalization manifest"), frozen(target, content, "selection"))
        if states[0] != states[1]:
            raise SelectionRefused("only one of %s and %s exists: the frozen records are incomplete; nothing is written"
                                   % (manifest_path, target))
        unchanged = states[0] is True
    except SelectionRefused as exc:
        print("p4_select refused: %s" % exc, file=sys.stderr)
        return 2
    located = {"work": str(work.resolve()), "pilot_report": str(path.resolve()), "pilot_campaign": str(Path(args.pilot_campaign).resolve())}
    if unchanged:
        print("selection unchanged: %s (content_sha256 %s)" % (target, content_sha256(content)[:16]))
        return code
    out.mkdir(parents=True, exist_ok=True)
    for where, doc in ((manifest_path, finalization), (target, content)):
        with where.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({**doc, "content_sha256": content_sha256(doc), "generated_at": stamp, "located_at": located},
                                    indent=1, sort_keys=True) + "\n")
    if code == 0:
        print("selected %s %s, lineages r1 and r2; wrote %s (bound to %s)" % (content["group"]["recipe"], content["group"]["chain"],
                                                                         target, manifest_path.name))
    else:
        print("not entered: %s; wrote %s" % (content["reason"], target))
    return code


if __name__ == "__main__":
    sys.exit(main())
