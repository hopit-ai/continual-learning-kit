#!/usr/bin/env python3
"""Package 4's frozen decision inputs, verified FIRST by every dependent action (send-5 review round 2, finding 2;
round-3 ruling F2; docs/phase2/plan-v3-package4-amendment3-20261004.md B7).

kit/runner.py skips a row whose latest attempt passed, so a batch line run again does not run the selection, the
reservation or the recipe check again. Their frozen records are therefore re-verified by every action that depends on
them, before it does anything: kit/p4_run.py (every GPU row), kit/p4_recipe.py, kit/p4_budget.py `gate` and `status`,
and kit/p4_report.py. On any difference they refuse, naming it: "inputs changed after they were frozen: <which>".

    python p4_frozen.py --work WORK [--reservation R.json] [--baseline B.json]     # prints the verification; exit 0 / 2

WHAT IS FROZEN (paths relative to WORK; every record holds `content_sha256`, the sha256 of its canonical content: sorted
keys, no whitespace, without content_sha256, generated_at, created_at and located_at):
  k8b4/selection/pilot-finalization.json   kit/p4_select.py: every decision input of the pilot with its sha256 (`inputs`)
                                           and the glob patterns they were found by (`patterns`): the pilot report, every
                                           pilot sweep, per-item file, bed-score.json, forgetting.json, prefix check and
                                           rollout statistics, every pilot run's run-summary.json and env/argv.txt,
                                           sdpo-commit.txt, sdpo-dirty.txt, data-sha256.txt, model-files.txt, every runner
                                           record (start.json, verdict.json) of the pilot campaign; the pilot campaign
                                           file's sha256 (kit/campaigns/k8b-pilot.yaml beside this file)
  k8b4/selection/selection.json            bound to the manifest (`finalization.sha256`) and the report (`pilot_report`)
  k8b4/budget/reservation.json             bound to the selection (`selection.content_sha256`)
  k8b4/recipe-check/frozen.json            kit/p4_recipe.py's comparison inputs and verdict, written at the FIRST check
                                           whatever its verdict, bound to the selection; `inputs` as above
  k8b4/recipe-check/baseline.json          the passed check, bound to frozen.json (`frozen_sha256`); `verify_baseline`
                                           checks its seal and link alone (kit/p4_report.py, round-4 ruling G3)
A file of `inputs` whose sha256 differs, that is missing, or a NEW file matching a frozen pattern is a change.
Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINALIZATION = Path("k8b4") / "selection" / "pilot-finalization.json"
SELECTION = Path("k8b4") / "selection" / "selection.json"
RESERVATION = Path("k8b4") / "budget" / "reservation.json"
RECIPE_FROZEN = Path("k8b4") / "recipe-check" / "frozen.json"
RECIPE_BASELINE = Path("k8b4") / "recipe-check" / "baseline.json"
PILOT_CAMPAIGN = HERE / "campaigns" / "k8b-pilot.yaml"
OUTSIDE = ("content_sha256", "generated_at", "created_at", "located_at")
CHANGED = "inputs changed after they were frozen"


def content_sha256(content: dict) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in OUTSIDE}


def seal(doc: dict) -> dict:
    """The document with its content_sha256 (what every frozen record carries)."""
    return {**canonical(doc), "content_sha256": content_sha256(canonical(doc))}


def sha256_file(path: Path):
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def file_records(work: Path, paths) -> list:
    """[{path (relative to WORK), sha256}] of files, sorted."""
    work = Path(work)
    out = []
    for path in sorted(set(Path(p) for p in paths)):
        rel = path.relative_to(work) if path.is_absolute() else path
        out.append({"path": str(rel), "sha256": sha256_file(work / rel)})
    return out


def glob_inputs(work: Path, patterns: list) -> list:
    found = set()
    for pattern in patterns:
        found.update(p for p in Path(work).glob(pattern) if p.is_file())
    return file_records(work, found)


def _read(path: Path):
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _self(doc, where: str, problems: list) -> bool:
    if doc is None:
        problems.append("%s is missing or unreadable" % where)
        return False
    if doc.get("content_sha256") != content_sha256(canonical(doc)):
        problems.append("%s was altered after it was frozen (its content no longer has its content_sha256)" % where)
        return False
    return True


def check_inputs(work: Path, inputs, patterns, where: str, problems: list) -> None:
    """Every recorded input re-hashed; with `patterns`, every file they match now must be a recorded input."""
    work = Path(work)
    recorded = {}
    for entry in inputs or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            problems.append("%s: an input entry is malformed (%r)" % (where, entry))
            continue
        recorded[entry["path"]] = entry.get("sha256")
        now = sha256_file(work / entry["path"])
        if now != entry.get("sha256"):
            problems.append("%s (sha256 %s now, %s when frozen in %s)" % (entry["path"], (now or "missing")[:16],
                                                                         str(entry.get("sha256") or "absent")[:16], where))
    if patterns:
        for entry in glob_inputs(work, patterns):
            if entry["path"] not in recorded:
                problems.append("%s (a new file matching a frozen pattern of %s)" % (entry["path"], where))


def verify(work: Path, *, reservation=None, baseline=None, need=("selection", "reservation", "recipe"),
           recipe_if_present: bool = True, pilot_campaign: Path = PILOT_CAMPAIGN, containment=None) -> list:
    """The differences between the frozen decision inputs and the tree now; empty when everything still holds.
    need: which layers must exist ("selection" always; "reservation"; "recipe" = a passed baseline.json). With
    recipe_if_present, a recipe-check record that exists is verified even when not needed."""
    work, problems = Path(work), []
    manifest = _read(work / FINALIZATION)
    if _self(manifest, str(FINALIZATION), problems):
        check_inputs(work, manifest.get("inputs"), manifest.get("patterns"), str(FINALIZATION), problems)
        recorded = ((manifest.get("pilot_campaign") or {}).get("sha256"))
        now = sha256_file(pilot_campaign)
        if not recorded or now != recorded:
            problems.append("the pilot campaign file %s (sha256 %s, finalized with %s)" % (pilot_campaign.name, (now or "missing")[:16],
                                                                                      str(recorded)[:16]))
    selection = _read(work / SELECTION)
    if _self(selection, str(SELECTION), problems) and manifest is not None:
        bound = (selection.get("finalization") or {}).get("sha256")
        if bound != content_sha256(canonical(manifest)):
            problems.append("%s is not bound to %s (finalization sha256 %s)" % (SELECTION, FINALIZATION, str(bound)[:16]))
        report = selection.get("pilot_report") or {}
        if not isinstance(report.get("path"), str) or sha256_file(work / report["path"]) != report.get("sha256"):
            problems.append("the pilot report %s (sha256 now %s, selected with %s)" % (report.get("path"), str(
                sha256_file(work / report["path"]) if isinstance(report.get("path"), str) else None)[:16], str(report.get("sha256"))[:16]))
    if "reservation" in need:
        path = Path(reservation) if reservation else work / RESERVATION
        doc = _read(path)
        if _self(doc, str(path), problems) and selection is not None:
            if (doc.get("selection") or {}).get("content_sha256") != selection.get("content_sha256"):
                problems.append("%s is not bound to the frozen selection" % path)
    frozen = _read(work / RECIPE_FROZEN)
    if "recipe" in need or (recipe_if_present and (work / RECIPE_FROZEN).exists()):
        if _self(frozen, str(RECIPE_FROZEN), problems):
            if selection is not None and frozen.get("selection_content_sha256") != selection.get("content_sha256"):
                problems.append("%s is not bound to the frozen selection" % RECIPE_FROZEN)
            check_inputs(work, frozen.get("inputs"), None, str(RECIPE_FROZEN), problems)
    if "recipe" in need:
        problems += verify_baseline(work, baseline)
    # A campaign requests this explicitly; check/selftest rows themselves precede the pair.
    if containment is True or (containment is None and os.environ.get('KIT_P4_CONTAINMENT') == 'slurm-step'):
        problems += verify_containment(work)
    return problems


def verify_containment(work):
    problems, docs = [], {}
    for name in ('receipt', 'selftest'):
        rel = Path('k8b4/containment') / ('containment-%s.json' % name)
        doc = _read(Path(work) / rel)
        if _self(doc, str(rel), problems):
            docs[name] = doc
            if doc.get('ok') is not True:
                problems.append('%s is not passing' % rel)
    if len(docs) == 2 and docs['selftest'].get('receipt_sha256') != docs['receipt']['content_sha256']:
        problems.append('containment selftest differs from the frozen cluster receipt')
    return problems


def verify_baseline(work: Path, baseline=None) -> list:
    """Round-4 ruling G3: the recipe check's passed baseline (k8b4/recipe-check/baseline.json) -- present, readable,
    still holding its seal, recording a passed check, and bound to the recipe-check frozen document, which must itself
    hold its seal. Empty when all of that holds. kit/p4_report.py uses the baseline only then; otherwise its report of
    both blocks is descriptive (no registered label)."""
    work, problems = Path(work), []
    path = Path(baseline) if baseline else work / RECIPE_BASELINE
    doc = _read(path)
    frozen = _read(work / RECIPE_FROZEN)
    if _self(doc, str(path), problems):
        if doc.get("ok") is not True:
            problems.append("%s does not record a passed recipe check" % path)
        if frozen is None:
            problems.append("%s is missing: the baseline's link cannot be verified" % RECIPE_FROZEN)
        elif doc.get("frozen_sha256") != content_sha256(canonical(frozen)):
            problems.append("%s is not bound to %s" % (path, RECIPE_FROZEN))
        elif frozen.get("content_sha256") != content_sha256(canonical(frozen)):
            problems.append("%s was altered after it was frozen" % RECIPE_FROZEN)
    return problems


def refusal(problems: list) -> str:
    return "%s: %s" % (CHANGED, "; ".join(problems[:12]) + ("; and %d more" % (len(problems) - 12) if len(problems) > 12 else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", required=True)
    parser.add_argument("--reservation")
    parser.add_argument("--baseline")
    parser.add_argument("--need", default="selection,reservation,recipe")
    args = parser.parse_args(argv)
    problems = verify(Path(args.work), reservation=args.reservation, baseline=args.baseline,
                      need=tuple(x for x in args.need.split(",") if x))
    if problems:
        print(refusal(problems), file=sys.stderr)
        return 2
    print("frozen inputs verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
