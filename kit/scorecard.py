#!/usr/bin/env python3
"""The K5 scorecard: four numbers for a sequence of jobs learned one after another.

    python scorecard.py from-k3 --root $WORK/k3 --arm none --out $WORK/k5/seq-none.json
    python scorecard.py build   --manifest $WORK/k5/seq-none.json --out $WORK/k5/scorecard-none

Nothing here trains, generates or scores anything. It reads the `bed-score.json` files
(`kit/eval_bed.py`) and `forgetting.json` files (`kit/score_forgetting.py`) a campaign has already
written and arranges them into the standard continual-learning scorecard.

THE FOUR NUMBERS. A sequence has T jobs, learned in order; stage i is the model after the i-th job
has been learned, and stage 0 is the untrained model. s[i][j] is job j's held-out score after stage
i, so the input is a (T+1) x T grid of scorings:

    average accuracy   = the mean over all jobs j of s[T][j]
                         -- what the model can still do when the sequence is over.
    backward transfer  = the mean over j < T of s[T][j] - s[j][j]
                         -- what the later jobs cost each earlier one, measured from the moment that
                            job was learned. Negative is forgetting.
    forward transfer   = the mean over j > 1 of s[j-1][j] - s[0][j]
                         -- what the model had already picked up about a job before it was trained
                            on it, measured against the untrained model.
    general delta      = each forgetting panel after stage T minus the untrained model's own score
                         on that panel -- the 300 questions that belong to no job in the sequence.

Every number above is a DIFFERENCE of two counts, and two counts are only comparable when they were
made on the same machine in the same decoding mode: across machines a panel moves by up to 3 points
before any training (receipts 204 and 209). So every scoring read here is checked for its
machine-and-mode fingerprint, and a scorecard mixing two of them is REFUSED --
`--allow-different-machines` overrides that and is recorded in the output as the departure it is.

A MISSING CELL IS A REFUSAL. Each of the (T+1) x T cells must be named by the manifest and must hold
the key being read. Nothing is carried forward from a neighbouring stage, and nothing is imputed: a
grid with a hole in it is not a scorecard.

THE MANIFEST is a JSON file naming those scorings, one entry per seed:

    {"schema": "kit-sequence.v1", "name": "k3-none",
     "runs": [{"seed": 0,
               "untrained": {"scores": {"spider": "eval/base-spider-a1", "gsm8k": "eval/base-gsm8k-a1"},
                             "forgetting": "forgetting/base-a1"},
               "stages": [{"job": "spider",
                           "scores": {"spider": "...", "gsm8k": "..."}, "forgetting": "..."},
                          {"job": "gsm8k",
                           "scores": {"spider": "...", "gsm8k": "..."}, "forgetting": "..."}]}]}

The stages are the sequence, in order: the j-th stage names the job learned there and the scorings of
EVERY job made after it. Each path is a JSON file or the directory holding one, and a relative path
is read against the manifest's own directory. The `forgetting` of the untrained model and of the last
stage are required (the general delta is their difference); the others are optional and reported as
they are found. Every seed must learn the same jobs in the same order: one scorecard is one sequence.

`from-k3` writes such a manifest from a K3 campaign tree (`kit/campaigns/k3-replay.yaml`), where
stage A plus one stage-B arm is a two-job sequence, so this tool is exercised on real shapes as soon
as K3 reports.

Standard library only; imports nothing of ours. Nothing is overwritten: an existing --out is refused.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "kit-scorecard.v1"
MANIFEST_SCHEMA = "kit-sequence.v1"
RESULT_FILES = ("bed-score.json", "forgetting.json")


def _density_module():
    """kit/density.py, loaded from beside this file so the kit works copied anywhere."""
    import importlib.util                                                    # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("density", Path(__file__).resolve().parent / "density.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# The K3 campaign's own tree, for the adapter: a point is `base`, `a-seed<S>` or `b-<arm>-seed<S>`,
# and a scored directory is `<stem>-a<attempt>`. Both patterns are k3_report.py's, on purpose.
POINT = re.compile(r"^(?:base|a-seed(?P<aseed>\d+)|b-(?P<arm>[a-z0-9]+)-seed(?P<bseed>\d+))$")
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
K3_JOBS = ("spider", "gsm8k")            # stage A learns Spider text-to-SQL, stage B learns GSM8K maths


class ScorecardError(ValueError):
    """The manifest or the evidence it names cannot support a scorecard; nothing is written."""


def result_path(where, base: Path) -> Path:
    """The JSON file a cell names: the file itself, or the one result file in the directory."""
    path = Path(where)
    if not path.is_absolute():
        path = base / path
    if path.is_file():
        return path
    if path.is_dir():
        for name in RESULT_FILES:
            if (path / name).is_file():
                return path / name
        raise ScorecardError("%s holds none of %s" % (path, ", ".join(RESULT_FILES)))
    raise ScorecardError("no such result file or directory: %s" % path)


def read_result(where, base: Path) -> tuple:
    """(result, path): one scoring, read from the file or directory a cell names."""
    path = result_path(where, base)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ScorecardError("%s is not JSON: %s" % (path, exc)) from exc
    if not isinstance(result, dict):
        raise ScorecardError("%s is not a JSON object, so it holds no scoring to read" % path)
    return result, path


def number(result: dict, key: str, path: Path) -> float:
    """One key of a scoring as a number; a missing or non-numeric key is a refusal, never a zero."""
    if key not in result:
        raise ScorecardError("%s has no top-level key %r (it has %s)"
                             % (path, key, ", ".join(sorted(result)[:12])))
    value = result[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScorecardError("%s[%r] is %r, which is not a number" % (path, key, value))
    return float(value)


def panel_scores(result: dict, path: Path) -> dict:
    """{panel: correct} from a forgetting.json."""
    panels = result.get("panels")
    if not isinstance(panels, dict) or not panels:
        raise ScorecardError("%s carries no `panels` object: it is not a forgetting.json" % path)
    scores = {}
    for name, block in panels.items():
        if not isinstance(block, dict) or not isinstance(block.get("correct"), (int, float)) \
                or isinstance(block.get("correct"), bool):
            raise ScorecardError("%s: panel %s has no numeric `correct`" % (path, name))
        scores[name] = float(block["correct"])
    return scores


def spread(values: list) -> dict:
    """Mean and SAMPLE standard deviation; the spread of one seed is not a number, so it is null."""
    clean = [v for v in values if v is not None]
    return {"n": len(clean), "mean": round(statistics.fmean(clean), 4) if clean else None,
            "sd": round(statistics.stdev(clean), 4) if len(clean) > 1 else None}


def mean_or_none(values: list):
    return round(statistics.fmean(values), 4) if values else None


def read_manifest(path: Path) -> dict:
    """The manifest, checked for shape: one sequence, the same jobs in the same order for every seed."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScorecardError("cannot read the manifest %s: %s" % (path, exc)) from exc
    try:
        manifest = json.loads(text)
    except ValueError as exc:
        raise ScorecardError("%s is not JSON: %s" % (path, exc)) from exc
    if not isinstance(manifest, dict):
        raise ScorecardError("%s is not a JSON object: a manifest is {schema, name, runs}" % path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ScorecardError("%s has schema %r, not %r: this is not a sequence manifest"
                             % (path, manifest.get("schema"), MANIFEST_SCHEMA))
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ScorecardError("%s lists no runs: a scorecard needs at least one seed's sequence" % path)
    jobs, first_seed, seeds = None, None, []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ScorecardError("%s: run %d is not an object" % (path, index))
        seed = run.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ScorecardError("%s: run %d has no integer `seed`" % (path, index))
        if seed in seeds:
            raise ScorecardError("%s: seed %d appears twice; one run per seed" % (path, seed))
        seeds.append(seed)
        stages = run.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ScorecardError("%s: seed %d lists no stages" % (path, seed))
        order = []
        for position, stage in enumerate(stages, start=1):
            if not isinstance(stage, dict):
                raise ScorecardError("%s: seed %d stage %d is not an object" % (path, seed, position))
            job = stage.get("job")
            if not isinstance(job, str) or not job:
                raise ScorecardError("%s: seed %d stage %d names no `job`" % (path, seed, position))
            if job in order:
                raise ScorecardError("%s: seed %d learns %r twice; a job holds one position in a sequence"
                                     % (path, seed, job))
            order.append(job)
        if not isinstance(run.get("untrained"), dict):
            raise ScorecardError("%s: seed %d has no `untrained` scorings, so stage 0 is unknown" % (path, seed))
        if jobs is None:
            jobs, first_seed = order, seed
        elif order != jobs:
            raise ScorecardError("%s: seed %d learns %s but seed %d learns %s; one scorecard is one sequence"
                                 % (path, seed, " then ".join(order), first_seed, " then ".join(jobs)))
    return {"name": manifest.get("name") or path.stem, "jobs": jobs, "runs": runs,
            "source": manifest.get("source"), "path": path}


def stage_name(index: int, jobs: list) -> str:
    return "stage 0 (untrained)" if index == 0 else "stage %d (%s)" % (index, jobs[index - 1])


def read_run(run: dict, jobs: list, base: Path, key: str) -> dict:
    """One seed's whole grid: {job: [s[0][j], ..., s[T][j]]}, its panels, its fingerprints, its sizes."""
    seed = run["seed"]
    points = [run["untrained"]] + list(run["stages"])
    matrix = {job: [] for job in jobs}
    costs = {job: [] for job in jobs}                       # tokens per correct answer, by stage (or None)
    panel_costs: dict = {}
    density = _density_module()
    machines, sizes, panels, cells = set(), {}, {}, 0
    for index, point in enumerate(points):
        where = "seed %d %s" % (seed, stage_name(index, jobs))
        scores = point.get("scores")
        if not isinstance(scores, dict):
            raise ScorecardError("%s names no `scores`" % where)
        for job in jobs:
            if job not in scores:
                raise ScorecardError("%s has no scoring of job %r. A scorecard needs every one of the "
                                     "%d x %d cells: a missing cell is a refusal, never a number carried "
                                     "over from another stage." % (where, job, len(points), len(jobs)))
            result, path = read_result(scores[job], base)
            matrix[job].append(number(result, key, path))
            costs[job].append(density.tokens_per_correct(result, path))
            machines.add((result.get("machine") or {}).get("id"))
            size = result.get("n")
            if isinstance(size, (int, float)) and not isinstance(size, bool):
                if sizes.setdefault(job, size) != size:
                    raise ScorecardError("%s scored job %r on %g questions, but an earlier stage scored it "
                                         "on %g: those are not the same held-out set, and their difference "
                                         "is not forgetting" % (where, job, size, sizes[job]))
            cells += 1
        if point.get("forgetting") is not None:
            result, path = read_result(point["forgetting"], base)
            panels[index] = panel_scores(result, path)
            panel_costs[index] = {name: density.tokens_per_correct(result, path, panel=name) for name in panels[index]}
            machines.add((result.get("machine") or {}).get("id"))
    last = len(points) - 1
    for index in (0, last):
        if index not in panels:
            raise ScorecardError("seed %d names no forgetting.json for %s; the general delta is the "
                                 "difference between the panels before and after the sequence"
                                 % (seed, stage_name(index, jobs)))
    if set(panels[0]) != set(panels[last]):
        raise ScorecardError("seed %d: the untrained model was scored on panels %s and the end of the "
                             "sequence on %s" % (seed, sorted(panels[0]), sorted(panels[last])))
    return {"seed": seed, "matrix": matrix, "panels": panels, "machines": machines,
            "job_questions": sizes, "cells_read": cells,
            "density": density_block(costs, panel_costs, jobs, last, density)}


def density_block(costs: dict, panel_costs: dict, jobs: list, last: int, density) -> dict:
    """What a correct answer costs, untrained against the end of the sequence (plan section 4c).
    Never a refusal: a scoring without token counts gives `-`, and `known` says how many cells had them."""
    block = {"bar": density.BAR, "jobs": {}, "panels": {}, "known": 0, "cells": 0}
    for job in jobs:
        cell = density.compare(costs[job][last], costs[job][0])
        cell["by_stage"] = [(c or {}).get("tokens_per_correct") for c in costs[job]]
        block["jobs"][job] = cell
        block["cells"] += len(costs[job]); block["known"] += sum(c is not None for c in costs[job])
    for name in sorted(panel_costs.get(0) or {}):
        cell = density.compare((panel_costs.get(last) or {}).get(name), panel_costs[0].get(name))
        block["panels"][name] = cell
        block["cells"] += 2; block["known"] += sum((panel_costs.get(i) or {}).get(name) is not None for i in (0, last))
    block["over_bar"] = sorted([job for job, c in block["jobs"].items() if c["verdict"] == "over bar"]
                               + ["panel " + n for n, c in block["panels"].items() if c["verdict"] == "over bar"])
    return block


def metrics(matrix: dict, jobs: list, panels: dict) -> dict:
    """The four numbers for one seed, from that seed's grid."""
    total = len(jobs)
    final = [matrix[job][total] for job in jobs]
    backward = [matrix[jobs[j - 1]][total] - matrix[jobs[j - 1]][j] for j in range(1, total)]
    forward = [matrix[jobs[j - 1]][j - 1] - matrix[jobs[j - 1]][0] for j in range(2, total + 1)]
    before, after = panels[0], panels[total]
    return {"average_accuracy": mean_or_none(final),
            "backward_transfer": mean_or_none(backward),
            "forward_transfer": mean_or_none(forward),
            "general_delta": {name: after[name] - before[name] for name in sorted(before)},
            "final_scores": {job: matrix[job][total] for job in jobs},
            "learned": {jobs[j - 1]: matrix[jobs[j - 1]][j] - matrix[jobs[j - 1]][j - 1]
                        for j in range(1, total + 1)},
            "per_job_backward": {jobs[j - 1]: matrix[jobs[j - 1]][total] - matrix[jobs[j - 1]][j]
                                 for j in range(1, total)},
            "per_job_forward": {jobs[j - 1]: matrix[jobs[j - 1]][j - 1] - matrix[jobs[j - 1]][0]
                                for j in range(2, total + 1)}}


def build(manifest_path: Path, *, key: str = "correct", allow_different_machines: bool = False) -> dict:
    manifest = read_manifest(manifest_path)
    jobs, base = manifest["jobs"], manifest_path.resolve().parent
    per_seed, machines, sizes, cells = {}, set(), {}, 0
    for run in manifest["runs"]:
        read = read_run(run, jobs, base, key)
        machines |= read["machines"]
        cells += read["cells_read"]
        for job, size in read["job_questions"].items():
            if sizes.setdefault(job, size) != size:
                raise ScorecardError("job %r was scored on %g questions for one seed and %g for another: "
                                     "those seeds cannot be averaged" % (job, size, sizes[job]))
        per_seed[read["seed"]] = dict(metrics(read["matrix"], jobs, read["panels"]),
                                      matrix={job: read["matrix"][job] for job in jobs},
                                      density=read["density"],
                                      panels={str(index): read["panels"][index] for index in sorted(read["panels"])})
    identifiers = sorted(machines, key=lambda item: (item is None, item))
    comparable = len(identifiers) == 1 and identifiers[0] is not None
    if not comparable and not allow_different_machines:
        raise ScorecardError(
            "the scorings named by %s carry %d different machine-and-mode fingerprints (%s). Every number "
            "on a scorecard is a difference of two counts, and a count from one machine cannot be "
            "subtracted from a count on another: a panel moves by up to 3 points across machines before "
            "any training. Re-score on one machine, or pass --allow-different-machines and say so in the "
            "readout." % (manifest_path, len(identifiers), ", ".join(str(m) for m in identifiers)))
    seeds = sorted(per_seed)
    panel_names = sorted(per_seed[seeds[0]]["general_delta"])
    for seed in seeds:
        if sorted(per_seed[seed]["general_delta"]) != panel_names:
            raise ScorecardError("seed %d was scored on panels %s and seed %d on %s: those seeds cannot be "
                                 "averaged" % (seed, sorted(per_seed[seed]["general_delta"]), seeds[0], panel_names))
    average = {name: spread([per_seed[seed][name] for seed in seeds])
               for name in ("average_accuracy", "backward_transfer", "forward_transfer")}
    average["general_delta"] = {panel: spread([per_seed[seed]["general_delta"].get(panel) for seed in seeds])
                                for panel in panel_names}
    return {"schema": SCHEMA, "name": manifest["name"], "manifest": str(manifest_path.resolve()),
            "source": manifest["source"], "key": key, "jobs": jobs, "stages": len(jobs),
            "job_questions": {job: sizes[job] for job in jobs if job in sizes},
            "panels": panel_names, "seeds": seeds,
            "per_seed": {str(seed): per_seed[seed] for seed in seeds}, "average": average,
            "machine_ids": identifiers, "comparable": int(comparable),
            "different_machines_allowed": int(bool(allow_different_machines)),
            "seeds_reported": len(seeds), "jobs_reported": len(jobs), "cells_read": cells}


def show(value, digits: int = 2, *, signed: bool = True) -> str:
    """A number for the readout: a difference carries its sign, a level does not, a null is a dash."""
    if value is None:
        return "-"
    if digits == 0:
        return "%+d" % value if signed else "%d" % value
    return "%+.*f" % (digits, value) if signed else "%.*f" % (digits, value)


def render(card: dict) -> str:
    jobs, seeds, total = card["jobs"], card["seeds"], card["stages"]
    sizes = card["job_questions"]
    lines = ["# Scorecard: %s" % card["name"], "",
             "A sequence of %d jobs, learned in this order: %s. Every number is `%s` from a held-out "
             "scoring the campaign already wrote. Stage 0 is the untrained model and stage %d is the end "
             "of the sequence; s[i][j] is job j's score after stage i."
             % (total, ", ".join(jobs), card["key"], total), "",
             "- **average accuracy**: the mean over jobs of s[T][j].",
             "- **backward transfer**: the mean over j<T of s[T][j] - s[j][j]. Negative is forgetting.",
             "- **forward transfer**: the mean over j>1 of s[j-1][j] - s[0][j].",
             "- **general delta**: each panel after stage %d minus the untrained model's own score." % total,
             ""]
    if not card["comparable"]:
        lines += ["**The scorings do not share one machine-and-mode fingerprint (%s), so none of these "
                  "differences is comparable.**" % ", ".join(str(m) for m in card["machine_ids"]), ""]
    if sizes and len(set(sizes.values())) > 1 and card["key"] == "correct":
        lines += ["**The jobs have held-out sets of different sizes (%s), so `correct` counts are averaged "
                  "across different scales. Read the per-job columns, or build the scorecard again with "
                  "`--key accuracy`.**" % ", ".join("%s %g" % (job, sizes[job]) for job in jobs if job in sizes),
                  ""]
    for seed in seeds:
        row = card["per_seed"][str(seed)]
        lines += ["## Seed %d" % seed, "",
                  "| after stage | " + " | ".join(jobs) + " |", "|---|" + "---|" * len(jobs)]
        for index in range(total + 1):
            lines.append("| %s | %s |" % (stage_name(index, jobs),
                                          " | ".join("%g" % row["matrix"][job][index] for job in jobs)))
        lines += ["", "average accuracy %s, backward transfer %s, forward transfer %s; general delta %s."
                  % (show(row["average_accuracy"], signed=False), show(row["backward_transfer"]),
                     show(row["forward_transfer"]),
                     ", ".join("%s %s" % (panel, show(row["general_delta"][panel], 0))
                               for panel in card["panels"])), ""]
    lines += density_lines(card)
    lines += ["## Over seeds", "",
              "| number | " + " | ".join("seed %d" % seed for seed in seeds) + " | mean | sd |",
              "|---|" + "---|" * (len(seeds) + 2)]
    for name, label, signed in (("average_accuracy", "average accuracy", False),
                                ("backward_transfer", "backward transfer", True),
                                ("forward_transfer", "forward transfer", True)):
        summary = card["average"][name]
        lines.append("| %s | %s | %s | %s |"
                     % (label, " | ".join(show(card["per_seed"][str(seed)][name], signed=signed)
                                          for seed in seeds),
                        show(summary["mean"], signed=signed),
                        "-" if summary["sd"] is None else "%.2f" % summary["sd"]))
    for panel in card["panels"]:
        summary = card["average"]["general_delta"][panel]
        lines.append("| general delta: %s | %s | %s | %s |"
                     % (panel, " | ".join(show(card["per_seed"][str(seed)]["general_delta"][panel], 0)
                                          for seed in seeds),
                        show(summary["mean"]), "-" if summary["sd"] is None else "%.2f" % summary["sd"]))
    if total < 2:
        lines += ["", "Backward and forward transfer are not defined for a sequence of one job, and are "
                  "reported as `-`."]
    lines += ["", "%d seeds, %d jobs, %d cells read from %s."
              % (card["seeds_reported"], card["jobs_reported"], card["cells_read"], card["manifest"]), ""]
    return "\n".join(lines)


def density_lines(card: dict) -> list:
    """The 'what a correct answer costs' section: tokens per correct answer, untrained against the end of
    the sequence, per seed, with the plan's bar. Dashes where a scoring carried no token counts."""
    jobs, seeds = card["jobs"], card["seeds"]
    first = card["per_seed"][str(seeds[0])].get("density")
    if not first:
        return []
    bar = first["bar"]
    lines = ["## What a correct answer costs", "",
             "Tokens per correct answer = the held-out set's output tokens divided by its correct answers: a cost "
             "per task, not a length. The bar (plan 4c): after the sequence a model spends at most %.1f times the "
             "untrained model's on each job and each panel; over it, the gain is reported at cost." % bar, "",
             "| seed | " + " | ".join(jobs) + " | " + " | ".join("panel " + p for p in card["panels"]) + " | over the bar |",
             "|---|" + "---|" * (len(jobs) + len(card["panels"]) + 1)]
    known = cells = 0
    for seed in seeds:
        d = card["per_seed"][str(seed)]["density"]
        known += d["known"]; cells += d["cells"]
        def cell(c):
            if c["ratio"] is None:
                return "-"
            return "%s to %s (x%.2f%s)" % (show(c["untrained"], 0, signed=False), show(c["trained"], 0, signed=False),
                                            c["ratio"], ", over" if c["verdict"] == "over bar" else "")
        lines.append("| %d | %s | %s | %s |" % (seed, " | ".join(cell(d["jobs"][job]) for job in jobs),
                                               " | ".join(cell(d["panels"].get(p, {"ratio": None})) for p in card["panels"]),
                                               ", ".join(d["over_bar"]) or "none"))
    lines.append("")
    if known < cells:
        lines += ["%d of %d scorings carried no token counts (`output_tokens_total` in the result, or a "
                  "`responses.jsonl` beside it), shown as `-`. The four numbers above do not depend on them." % (cells - known, cells), ""]
    return lines


# ------------------------------------------------------------------ the K3 adapter
def latest_dir(directory: Path, filename: str) -> dict:
    """{stem: directory} for the highest attempt of every <stem>-aN folder holding `filename`."""
    found: dict = {}
    if not directory.is_dir():
        return found
    for child in sorted(directory.iterdir()):
        match = ATTEMPT.match(child.name)
        if not match or not (child / filename).is_file():
            continue
        stem, attempt = match["stem"], int(match["attempt"])
        if stem not in found or attempt > found[stem][0]:
            found[stem] = (attempt, child)
    return {stem: child for stem, (_, child) in found.items()}


def k3_points(root: Path) -> dict:
    """{point: {'spider': dir, 'gsm8k': dir, 'panels': dir}} for the K3 campaign tree under `root`."""
    points: dict = {}
    for stem, child in latest_dir(root / "eval", "bed-score.json").items():
        for job in K3_JOBS:
            if stem.endswith("-" + job) and POINT.match(stem[: -len(job) - 1]):
                points.setdefault(stem[: -len(job) - 1], {})[job] = child
    for stem, child in latest_dir(root / "forgetting", "forgetting.json").items():
        if POINT.match(stem):
            points.setdefault(stem, {})["panels"] = child
    return points


def from_k3(root: Path, arm: str, *, name: str | None = None) -> dict:
    """A sequence manifest for one K3 arm: stage A (Spider) then that arm's stage B (GSM8K), per seed.

    Only the cells actually present in the tree are named. A hole is left as a hole on purpose: it is
    `build` that refuses a grid with a missing cell, and it says which cell is missing.
    """
    points = k3_points(root)
    if not points:
        raise ScorecardError("no scorings under %s: expected eval/<point>-<bed>-aN/bed-score.json and "
                             "forgetting/<point>-aN/forgetting.json" % root)
    arms, seeds = {}, []
    for point in points:
        match = POINT.match(point)
        if match["arm"]:
            arms.setdefault(match["arm"], []).append(int(match["bseed"]))
    if arm not in arms:
        raise ScorecardError("no stage-B arm %r under %s (found %s)"
                             % (arm, root, ", ".join(sorted(arms)) or "none"))
    if "base" not in points:
        raise ScorecardError("no `base` point under %s: the untrained model is stage 0 of every sequence" % root)

    def cells(point: str) -> dict:
        found = points.get(point, {})
        entry = {"scores": {job: str(found[job].resolve()) for job in K3_JOBS if job in found}}
        if "panels" in found:
            entry["forgetting"] = str(found["panels"].resolve())
        return entry

    runs = []
    for seed in sorted(set(arms[arm])):
        runs.append({"seed": seed, "untrained": cells("base"),
                     "stages": [dict(cells("a-seed%d" % seed), job=K3_JOBS[0]),
                                dict(cells("b-%s-seed%d" % (arm, seed)), job=K3_JOBS[1])]})
        seeds.append(seed)
    return {"schema": MANIFEST_SCHEMA, "name": name or ("k3-%s" % arm),
            "source": {"campaign": "k3-replay", "root": str(root.resolve()), "arm": arm},
            "jobs": list(K3_JOBS), "runs": runs, "seeds_found": seeds,
            "arms_found": sorted(arms)}


def missing_cells(manifest: dict) -> list:
    """Which of the grid's cells the manifest does not name, in the words `build` would use."""
    gaps = []
    for run in manifest["runs"]:
        for index, point in enumerate([run["untrained"]] + run["stages"]):
            for job in manifest["jobs"]:
                if job not in point.get("scores", {}):
                    gaps.append("seed %d %s: %s" % (run["seed"], stage_name(index, manifest["jobs"]), job))
            if index in (0, len(run["stages"])) and "forgetting" not in point:
                gaps.append("seed %d %s: forgetting.json" % (run["seed"], stage_name(index, manifest["jobs"])))
    return gaps


def cmd_from_k3(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % out)
    try:
        manifest = from_k3(Path(args.root), args.arm, name=args.name)
    except ScorecardError as exc:
        raise SystemExit(str(exc))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    gaps = missing_cells(manifest)
    print("arm %s: %d seeds (%s), %d jobs" % (args.arm, len(manifest["runs"]),
                                              ", ".join(str(s) for s in manifest["seeds_found"]),
                                              len(manifest["jobs"])))
    if gaps:
        print("%d cells are not in the tree yet, and `build` will refuse until they are:" % len(gaps))
        for gap in gaps[:12]:
            print("   ", gap)
    print("wrote", out)
    return 0


def cmd_build(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % out)
    try:
        card = build(Path(args.manifest), key=args.key,
                     allow_different_machines=args.allow_different_machines)
    except ScorecardError as exc:
        raise SystemExit(str(exc))
    out.mkdir(parents=True)
    (out / "scorecard.json").write_text(json.dumps(card, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    text = render(card)
    (out / "scorecard.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The scorecard for a sequence of jobs learned in order.")
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build", help="read a sequence manifest and write scorecard.json and scorecard.md")
    b.add_argument("--manifest", required=True, help="the sequence manifest")
    b.add_argument("--out", required=True, help="a new directory; never overwritten")
    b.add_argument("--key", default="correct", help="the key to read from each scoring (default: correct)")
    b.add_argument("--allow-different-machines", action="store_true",
                   help="report anyway; recorded in the scorecard as a departure")
    k = sub.add_parser("from-k3", help="write a sequence manifest for one arm of a K3 campaign tree")
    k.add_argument("--root", required=True, help="the campaign's k3 directory")
    k.add_argument("--arm", required=True, help="the stage-B arm: none, rehearse10, rehearse30, kl")
    k.add_argument("--name", default=None, help="the sequence's name (default: k3-<arm>)")
    k.add_argument("--out", required=True, help="the manifest file to write; never overwritten")
    args = parser.parse_args(argv)
    return {"build": cmd_build, "from-k3": cmd_from_k3}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
