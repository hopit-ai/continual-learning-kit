#!/usr/bin/env python3
"""The three things a SEQUENCE of jobs needs that a single job does not.

    python sequence.py pool     --from $WORK/data/spider/train.parquet --rows 1280 --seed 0 \\
                                --out $WORK/data/stage/spider-1280
    python sequence.py manifest --campaign $KIT/campaigns/k5-sequence.yaml --root $WORK/k5 \\
                                --order sqlfirst --arm rehearse10 --out $WORK/k5/sequences/x.json
    python sequence.py estimate --campaign $KIT/campaigns/k5-sequence.yaml

`pool` BUILDS A QUESTION FILE of an exact size out of one or more prepared bed files. Two things in
K5 need it. A stage is 40 steps x 32 questions = 1,280 rows and the trainer reads the file once
(`total_epochs=1`), so a bed with fewer questions than that -- Spider has 640 -- cannot fill a stage
unless its questions are written more than once; and the rehearsal arm mixes EVERY earlier job's
questions into the current stage, while `kit/mix.py` takes one earlier job. `pool` answers both: it
draws from its inputs in turn, one row from each, so a pool of three jobs holds equal numbers of all
three, and it goes round again only when an input runs out. Nothing about the answers is stored or
changed: these are the beds' own trainer rows, questions and checkers, and the model answers them
afresh at every step. Deterministic in `--seed`: same inputs, same size, same seed, same bytes.

`manifest` WRITES THE SCORECARD'S MANIFEST from a campaign and the tree that campaign wrote. The
scorecard (`kit/scorecard.py`) reads a (T+1) x T grid of scorings -- every job scored after every
stage, plus the untrained model -- and its shape is documented at the top of that file. This tool
builds that manifest for one order and one arm by reading the campaign file for what the grid IS
(which stages, which jobs, which seeds) and the tree for where each scoring LANDED (the highest
`-aN` attempt), and it REFUSES a missing cell, naming every one that is missing. A scorecard with a
hole in it is not a scorecard, and a manifest that quietly names 19 of 20 cells is how a hole
becomes a number.

`estimate` ADDS UP THE GPU-HOURS the campaign will cost, from the campaign file alone, before
anything runs. One of its three numbers is measured (K3: about 15 seconds a step for the 1.7B on 8 x
H100); the other two are assumptions and are printed as assumptions. Rows tagged `code` are counted
separately, because the coding bed is being written separately.

Standard library, plus pyarrow for `pool` (the trainer reads parquet) and PyYAML for the two
commands that read a campaign file -- the same two the runner already needs. Nothing here trains,
scores or downloads anything, and nothing is ever overwritten: an existing output is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_SCHEMA = "kit-sequence.v1"          # kit/scorecard.py reads this
POOL_SCHEMA = "kit-pool.v1"
ESTIMATE_SCHEMA = "kit-estimate.v1"
PANEL_FILE = HERE / "panels" / "general-v1.jsonl"

#: A measured model of this campaign: `<order>-<arm>-seed<SEED>-stage<STAGE>`, plus `base` for the
#: untrained model, which is stage 0 of every sequence. The same grammar kit/k5_report.py parses.
POINT = re.compile(r"^(?P<order>[a-z0-9]+)-(?P<arm>[a-z0-9]+)-seed(?P<seed>\d+)-stage(?P<stage>\d+)$")
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
#: The trainer's run directory, which kit/plasticity.py's learning half parses for the same fields.
RUN_NAME = re.compile(r"^(?P<job>[a-z0-9]+)-pos(?P<position>\d+)-seed(?P<seed>\d+)")

#: What `estimate` is made of. Only the first is measured.
TRAIN_SECONDS_PER_STEP = 15.0        # K3, Qwen3-1.7B, 8 x H100, 40-step runs
SCORE_SECONDS_PER_ANSWER = 1.5       # ASSUMPTION: one answer of up to 2,048 tokens on one H100
PROBE_SECONDS = 300.0                # ASSUMPTION: 200 prompts, forward passes only, one H100
ORDER_SEED_OFFSET = 1_000_003        # kit/mix.py's, so the two tools' streams never coincide


class SequenceError(ValueError):
    """The inputs, the campaign or the tree cannot support what was asked; nothing is written."""


# ------------------------------------------------------------------------------------ reading
# read_rows and columns_of are DUPLICATED from kit/mix.py on purpose: that file is the kit's
# reference for what a trainer row file is, and a shared helper would let a change there move a
# number here silently. tests/test_kit_k5.py compares the two and fails if either side moves.
def read_rows(path) -> list:
    """Trainer rows from a .jsonl (one object a line) or a .parquet file."""
    path = Path(path)
    if not path.is_file():
        raise SequenceError("no such input file: %s" % path)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq                                      # noqa: PLC0415
        except ImportError:
            raise SequenceError("%s is parquet and pyarrow is not installed; pass the jsonl instead" % path) from None
        rows = pq.read_table(path).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if not rows:
        raise SequenceError("%s has no rows" % path)
    if any(not isinstance(row, dict) for row in rows):
        raise SequenceError("%s has a row that is not a JSON object" % path)
    return rows


def columns_of(rows: list, path) -> tuple:
    """The one column set every row in a file must have. A file that disagrees with itself is refused."""
    first = tuple(sorted(rows[0]))
    for index, row in enumerate(rows):
        if tuple(sorted(row)) != first:
            raise SequenceError("row %d of %s has columns %s, row 0 has %s: one file must be one table"
                                % (index, path, sorted(row), list(first)))
    return first


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_campaign(path: Path) -> dict:
    """The campaign file, as the runner loads it (PyYAML for .yaml, the standard library for .json)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SequenceError("cannot read the campaign %s: %s" % (path, exc)) from exc
    if Path(path).suffix == ".json":
        campaign = json.loads(text)
    else:
        try:
            import yaml                                                       # noqa: PLC0415
        except ImportError:
            raise SequenceError("PyYAML is not installed, so %s cannot be read; it is what "
                                "kit/runner.py needs to run this campaign at all" % path) from None
        campaign = yaml.safe_load(text)
    if not isinstance(campaign, dict) or not isinstance(campaign.get("rows"), list):
        raise SequenceError("%s is not a campaign: it has no `rows` list" % path)
    return campaign


# ------------------------------------------------------------------------------------ pool
def rounds_for(sources: list, rows: int | None) -> int:
    """How many rows each input is asked for; None means 'every input contributes all it can, once'."""
    if rows is None:
        return min(len(source) for source in sources)
    if rows <= 0:
        raise SequenceError("refusing to write %d rows: a stage's training file has to hold rows" % rows)
    return -(-rows // len(sources))                                            # ceiling


def draw_order(count: int, wanted: int, seed: int) -> list:
    """`wanted` row indices of one input: a fresh permutation each time round, deterministic in seed."""
    rng = random.Random(seed)
    order: list = []
    while len(order) < wanted:
        pool = list(range(count))
        rng.shuffle(pool)
        order += pool
    return order[:wanted]


def pool(sources: list, rows: int | None, seed: int) -> tuple:
    """(rows, facts): one row from each input in turn, going round again only when an input runs out."""
    per_input = rounds_for(sources, rows)
    total = rows if rows is not None else per_input * len(sources)
    orders = [draw_order(len(source), per_input, seed + index * ORDER_SEED_OFFSET)
              for index, source in enumerate(sources)]
    out, used = [], [[] for _ in sources]
    for position in range(total):
        which, step = position % len(sources), position // len(sources)
        out.append(sources[which][orders[which][step]])
        used[which].append(orders[which][step])
    facts = {"rows_total": len(out), "inputs_pooled": len(sources), "seed": int(seed),
             "rows_per_input": [len(chosen) for chosen in used],
             "distinct_per_input": [len(set(chosen)) for chosen in used],
             "repeats_per_input": [len(chosen) - len(set(chosen)) for chosen in used],
             "available_per_input": [len(source) for source in sources]}
    return out, facts


def as_jsonl(rows: list) -> str:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)


def data_source_counts(rows: list) -> dict:
    counts: dict = {}
    for row in rows:
        key = str(row.get("data_source", "-"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_table(rows: list):
    """The rows as one arrow table, or None when pyarrow is missing. Built BEFORE anything is written."""
    try:
        import pyarrow as pa                                                  # noqa: PLC0415
    except ImportError:
        return None
    return pa.Table.from_pylist(rows)


def write_new(path: Path, text: str) -> str:
    if path.exists():
        raise SequenceError("refusing to overwrite %s: an output is never replaced, write to a new directory" % path)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_pool(inputs: list, rows: int | None, seed: int, out, *, name: str = "train",
               allow_no_parquet: bool = False) -> dict:
    if not inputs:
        raise SequenceError("a pool needs at least one --from")
    sources, columns = [], None
    for path in inputs:
        read = read_rows(path)
        here = columns_of(read, path)
        if columns is None:
            columns = here
        elif here != columns:
            raise SequenceError("%s has columns %s and %s has %s. The trainer reads ONE table, and a row "
                                "missing a column is a silent failure inside it: prepare every bed with "
                                "its own bed writer and pool those files."
                                % (path, list(here), inputs[0], list(columns)))
        sources.append(read)
    written, facts = pool(sources, rows, seed)
    table = build_table(written)
    if table is None and not allow_no_parquet:
        raise SequenceError("pyarrow is not installed, so the parquet the trainer reads cannot be written "
                            "and nothing has been. Install pyarrow, or pass --allow-no-parquet on purpose.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "pool.manifest.json"
    if manifest_path.exists():
        raise SequenceError("refusing to overwrite %s: an output is never replaced, write to a new directory"
                            % manifest_path)
    jsonl_sha = write_new(out / ("%s.jsonl" % name), as_jsonl(written))
    wrote_parquet = False
    if table is not None:
        parquet = out / ("%s.parquet" % name)
        if parquet.exists():
            raise SequenceError("refusing to overwrite %s: an output is never replaced" % parquet)
        import pyarrow.parquet as pq                                          # noqa: PLC0415
        pq.write_table(table, parquet)
        wrote_parquet = True
    manifest = {"schema": POOL_SCHEMA,
                "what": "questions drawn from the inputs one at a time in turn; the model answers them "
                        "afresh at every step and each bed's own checker rewards its own rows",
                **facts, "columns": list(columns), "data_source_counts": data_source_counts(written),
                "inputs": [{"path": str(Path(path).resolve()), "sha256": sha256_file(path),
                            "rows": len(source)} for path, source in zip(inputs, sources)],
                "outputs": {"jsonl": "%s.jsonl" % name, "jsonl_sha256": jsonl_sha,
                            "parquet": "%s.parquet" % name if wrote_parquet else None}}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


# ------------------------------------------------------------------- the campaign's grid
def without_attempt(name: str) -> str:
    """`<stem>-a3` and the campaign's own unresolved `<stem>-a{attempt}` both name `<stem>`."""
    if name.endswith("-a{attempt}"):
        return name[: -len("-a{attempt}")]
    match = ATTEMPT.match(name)
    return match["stem"] if match else name


def campaign_grid(campaign: dict) -> dict:
    """What the campaign says the grid IS: its points, their jobs, and the beds scored at each.

    Read from the rows themselves, never from a list kept somewhere else: a training row's id is the
    point and its NAME is `<job>-pos<POSITION>-seed<SEED>`, and a scoring row's OUT is
    `.../eval/<point>-<bed>-a{attempt}` or `.../forgetting/<point>-a{attempt}`.
    """
    stages: dict = {}
    beds: dict = {}
    for row in campaign["rows"]:
        env = row.get("env") or {}
        point = POINT.match(str(row.get("id", "")))
        name = RUN_NAME.match(str(env.get("NAME", "")))
        if point and name and "run_grpo" in " ".join(row.get("command") or []):
            key = (point["order"], point["arm"], int(point["seed"]), int(point["stage"]))
            if int(name["position"]) != int(point["stage"]) or int(name["seed"]) != int(point["seed"]):
                raise SequenceError("row %s trains a run called %s: the position and seed in a run's name "
                                    "are what kit/plasticity.py reads, and they must be the row's own"
                                    % (row["id"], env["NAME"]))
            stages[key] = name["job"]
        out = str(env.get("OUT", ""))
        if "/eval/" in out:
            beds.setdefault(without_attempt(out.rsplit("/eval/", 1)[1]), None)
    if not stages:
        raise SequenceError("no training rows in this campaign: a sequence is read from rows whose id is "
                            "<order>-<arm>-seed<SEED>-stage<STAGE> and whose NAME is <job>-pos<N>-seed<S>")
    return {"stages": stages, "eval_stems": sorted(beds)}


def beds_of(grid: dict) -> list:
    """Every bed scored at a point, in the order the campaign scores them at the untrained model."""
    found: list = []
    for stem in grid["eval_stems"]:
        if stem.startswith("base-"):
            bed = stem[len("base-"):]
            if bed not in found:
                found.append(bed)
    if not found:
        raise SequenceError("this campaign scores no bed at `base`: the untrained model is stage 0 of "
                            "every sequence, and without it no scorecard can be built")
    return found


def sequence_of(grid: dict, order: str, arm: str, shared_arm: str = "shared") -> dict:
    """{'jobs': [...], 'seeds': [...], 'points': {(seed, stage): point}} for one order and one arm."""
    seeds, jobs, points = set(), {}, {}
    for (row_order, row_arm, seed, stage), job in grid["stages"].items():
        if row_order != order or row_arm not in (arm, shared_arm):
            continue
        if row_arm == shared_arm and stage != 1:
            continue
        seeds.add(seed)
        points[(seed, stage)] = "%s-%s-seed%d-stage%d" % (order, row_arm, seed, stage)
        if jobs.setdefault(stage, job) != job:
            raise SequenceError("order %s learns %s at stage %d for one seed and %s for another"
                                % (order, jobs[stage], stage, job))
    if not points:
        raise SequenceError("no rows for order %r and arm %r in this campaign" % (order, arm))
    ordered = [jobs[stage] for stage in sorted(jobs)]
    if sorted(jobs) != list(range(1, len(jobs) + 1)):
        raise SequenceError("order %s arm %s has stages %s: a sequence runs from stage 1 with no gaps"
                            % (order, arm, sorted(jobs)))
    return {"jobs": ordered, "seeds": sorted(seeds), "points": points}


# ------------------------------------------------------------------- the tree the campaign wrote
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
    return {stem: child for stem, (_attempt, child) in found.items()}


def read_tree(root: Path, beds: list) -> dict:
    """{point: {'scores': {bed: dir}, 'forgetting': dir, 'plasticity': dir}} under the campaign's k5 tree."""
    points: dict = {}
    for stem, child in latest_dir(root / "eval", "bed-score.json").items():
        for bed in beds:
            if stem.endswith("-" + bed):
                points.setdefault(stem[: -len(bed) - 1], {}).setdefault("scores", {})[bed] = child
    for stem, child in latest_dir(root / "forgetting", "forgetting.json").items():
        points.setdefault(stem, {})["forgetting"] = child
    for stem, child in latest_dir(root / "plasticity", "plasticity.json").items():
        points.setdefault(stem, {})["plasticity"] = child
    return points


def build_manifest(campaign_path: Path, root: Path, order: str, arm: str, *,
                   name: str | None = None) -> dict:
    """The scorecard's manifest for one order and one arm. A missing cell is a refusal."""
    grid = campaign_grid(load_campaign(campaign_path))
    beds = beds_of(grid)
    sequence = sequence_of(grid, order, arm)
    found = read_tree(Path(root), beds)
    jobs, missing, runs = sequence["jobs"], [], []
    for job in jobs:
        if job not in beds:
            raise SequenceError("stage %d of order %s learns %r, and this campaign never scores a %r "
                                "held-out set: a job that is learned and not scored cannot be on a "
                                "scorecard" % (jobs.index(job) + 1, order, job, job))
    for seed in sequence["seeds"]:
        stages = []
        for stage in range(0, len(jobs) + 1):
            point = "base" if stage == 0 else sequence["points"].get((seed, stage))
            where = "stage 0 (untrained)" if stage == 0 else "seed %d stage %d (%s)" % (seed, stage, jobs[stage - 1])
            if point is None:
                missing.append("%s: the campaign has no training row for it" % where)
                continue
            entry = found.get(point, {})
            scores = entry.get("scores", {})
            cell = {"scores": {}}
            for job in jobs:
                if job in scores:
                    cell["scores"][job] = str(scores[job].resolve())
                else:
                    missing.append("%s: no %s scoring (expected %s/eval/%s-%s-aN/bed-score.json)"
                                   % (where, job, root, point, job))
            if "forgetting" in entry:
                cell["forgetting"] = str(entry["forgetting"].resolve())
            elif stage in (0, len(jobs)):
                missing.append("%s: no forgetting.json (expected %s/forgetting/%s-aN/forgetting.json)"
                               % (where, root, point))
            if "plasticity" in entry:
                cell["plasticity"] = str(entry["plasticity"].resolve())
            cell["point"] = point
            if stage:
                cell["job"] = jobs[stage - 1]
            stages.append(cell)
        runs.append({"seed": seed, "untrained": stages[0], "stages": stages[1:]})
    if missing:
        raise SequenceError(
            "%d cells of the %d x %d grid are missing, so no manifest was written. A scorecard needs "
            "every cell: nothing is carried over from a neighbouring stage and nothing is imputed.\n  %s"
            % (len(missing), len(jobs) + 1, len(jobs), "\n  ".join(missing[:20])
               + ("\n  ..." if len(missing) > 20 else "")))
    cells = sum(len(stage["scores"]) for run in runs for stage in [run["untrained"]] + run["stages"])
    return {"schema": MANIFEST_SCHEMA, "name": name or ("k5-%s-%s" % (order, arm)),
            "source": {"campaign": str(Path(campaign_path).resolve()), "root": str(Path(root).resolve()),
                       "order": order, "arm": arm},
            "jobs": jobs, "runs": runs,
            "seeds_named": len(runs), "stages_named": len(jobs), "cells_named": cells}


# ------------------------------------------------------------------------------------ estimate
def panel_questions() -> int:
    if PANEL_FILE.is_file():
        return len([line for line in PANEL_FILE.read_text(encoding="utf-8").split("\n") if line.strip()])
    return 300


def row_cost(row: dict, *, seconds_per_step: float, seconds_per_answer: float,
             probe_seconds: float, panel: int) -> dict:
    """What one row costs in GPU-hours, and why. A row that holds no GPU costs nothing."""
    command = " ".join(row.get("command") or [])
    env = row.get("env") or {}

    def hours(seconds: float, gpus: int) -> float:
        return round(seconds * gpus / 3600.0, 4)

    if "run_grpo" in command:
        steps, gpus = int(env.get("STEPS", 0) or 0), int(env.get("NGPU", 1) or 1)
        return {"kind": "train", "gpu_hours": hours(steps * seconds_per_step, gpus),
                "detail": "%d steps x %g s x %d GPUs" % (steps, seconds_per_step, gpus)}
    if "eval_bed.py" in command:
        questions = next((bar.get("min") for bar in row.get("bars") or [] if bar.get("key") == "n"), 0) or 0
        return {"kind": "score", "gpu_hours": hours(questions * seconds_per_answer, 1),
                "detail": "%d questions x %g s" % (questions, seconds_per_answer)}
    if "score_forgetting.py" in command and "generate" in command:
        return {"kind": "panels", "gpu_hours": hours(panel * seconds_per_answer, 1),
                "detail": "%d panel questions x %g s" % (panel, seconds_per_answer)}
    if "plasticity.py" in command and "probe" in command:
        return {"kind": "probe", "gpu_hours": hours(probe_seconds, 1),
                "detail": "%g s of forward passes" % probe_seconds}
    return {"kind": "cpu", "gpu_hours": 0.0, "detail": "no GPU is held"}


def build_estimate(campaign_path: Path, *, seconds_per_step: float = TRAIN_SECONDS_PER_STEP,
                   seconds_per_answer: float = SCORE_SECONDS_PER_ANSWER,
                   probe_seconds: float = PROBE_SECONDS) -> dict:
    campaign = load_campaign(campaign_path)
    panel = panel_questions()
    kinds: dict = {}
    total = coding = 0.0
    rows = []
    for row in campaign["rows"]:
        cost = row_cost(row, seconds_per_step=seconds_per_step, seconds_per_answer=seconds_per_answer,
                        probe_seconds=probe_seconds, panel=panel)
        tagged = "code" in (row.get("tags") or [])
        entry = kinds.setdefault(cost["kind"], {"rows": 0, "gpu_hours": 0.0, "coding_gpu_hours": 0.0})
        entry["rows"] += 1
        entry["gpu_hours"] = round(entry["gpu_hours"] + cost["gpu_hours"], 4)
        total += cost["gpu_hours"]
        if tagged:
            entry["coding_gpu_hours"] = round(entry["coding_gpu_hours"] + cost["gpu_hours"], 4)
            coding += cost["gpu_hours"]
        rows.append({"row": row.get("id"), "tagged_code": int(tagged), **cost})
    return {"schema": ESTIMATE_SCHEMA, "campaign": str(Path(campaign_path).resolve()),
            "name": campaign.get("name"), "rows": len(campaign["rows"]),
            "assumptions": {
                "train_seconds_per_step": seconds_per_step,
                "train_seconds_per_step_source": "MEASURED: K3, Qwen3-1.7B, 8 x H100, 40-step runs",
                "score_seconds_per_answer": seconds_per_answer,
                "score_seconds_per_answer_source": "ASSUMPTION: not measured on the partner's machine",
                "probe_seconds": probe_seconds,
                "probe_seconds_source": "ASSUMPTION: not measured on the partner's machine",
                "panel_questions": panel},
            "by_kind": {kind: kinds[kind] for kind in sorted(kinds)},
            "gpu_hours": round(total, 2), "gpu_hours_tagged_code": round(coding, 2),
            "gpu_hours_without_code": round(total - coding, 2), "per_row": rows}


def render_estimate(estimate: dict) -> str:
    lines = ["# GPU-hours: %s" % estimate["name"], "",
             "%d rows. One number below is measured and two are assumptions, and an estimate made of "
             "assumptions is an estimate: read it as a size, not as a price." % estimate["rows"], "",
             "| what | rows | GPU-hours | of which tagged `code` |", "|---|---|---|---|"]
    for kind, entry in estimate["by_kind"].items():
        lines.append("| %s | %d | %.1f | %.1f |" % (kind, entry["rows"], entry["gpu_hours"],
                                                    entry["coding_gpu_hours"]))
    lines += ["| **total** | %d | **%.1f** | %.1f |" % (estimate["rows"], estimate["gpu_hours"],
                                                        estimate["gpu_hours_tagged_code"]), "",
              "Without the rows tagged `code`: %.1f GPU-hours." % estimate["gpu_hours_without_code"], "",
              "What it is made of:", ""]
    for key in ("train_seconds_per_step", "score_seconds_per_answer", "probe_seconds"):
        lines.append("- `%s` = %g -- %s" % (key, estimate["assumptions"][key],
                                            estimate["assumptions"]["%s_source" % key]))
    lines += ["- the forgetting panel is %d questions." % estimate["assumptions"]["panel_questions"], ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ subcommands
def cmd_pool(args) -> int:
    try:
        manifest = build_pool(args.source, args.rows, args.seed, args.out, name=args.name,
                              allow_no_parquet=args.allow_no_parquet)
    except SequenceError as exc:
        raise SystemExit(str(exc))
    print("wrote %s: %d rows from %d input(s) %s (%s distinct, %s rows repeated), seed %d%s"
          % (args.out, manifest["rows_total"], manifest["inputs_pooled"],
             manifest["rows_per_input"], manifest["distinct_per_input"], manifest["repeats_per_input"],
             manifest["seed"], "" if manifest["outputs"]["parquet"] else " (jsonl only: pyarrow not installed)"))
    return 0


def cmd_manifest(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % out)
    try:
        manifest = build_manifest(Path(args.campaign), Path(args.root), args.order, args.arm, name=args.name)
    except SequenceError as exc:
        raise SystemExit(str(exc))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("%s %s: %d seeds, %d jobs (%s), %d cells named"
          % (args.order, args.arm, manifest["seeds_named"], manifest["stages_named"],
             " then ".join(manifest["jobs"]), manifest["cells_named"]))
    print("wrote", out)
    return 0


def cmd_estimate(args) -> int:
    try:
        estimate = build_estimate(Path(args.campaign), seconds_per_step=args.seconds_per_step,
                                  seconds_per_answer=args.seconds_per_answer,
                                  probe_seconds=args.probe_seconds)
    except SequenceError as exc:
        raise SystemExit(str(exc))
    text = render_estimate(estimate)
    if args.out:
        out = Path(args.out)
        if out.exists():
            raise SystemExit("refusing to overwrite %s: an output is never replaced" % out)
        out.mkdir(parents=True)
        (out / "estimate.json").write_text(json.dumps(estimate, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8")
        (out / "estimate.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pools, scorecard manifests and GPU-hour estimates "
                                                 "for a sequence of jobs learned one after another.")
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("pool", help="one question file of an exact size, out of one or more bed files")
    p.add_argument("--from", dest="source", action="append", default=[], required=True,
                   help="a prepared bed's trainer file; repeat for every job to draw from")
    p.add_argument("--rows", type=int, default=None,
                   help="exactly this many rows (repeating an input only when it runs out); "
                        "default: every input contributes all it can, once")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True, help="a directory; nothing in it is overwritten")
    p.add_argument("--name", default="train", help="the stem of the written files (default: train)")
    p.add_argument("--allow-no-parquet", action="store_true",
                   help="write the jsonl alone when pyarrow is missing; the trainer reads parquet")

    m = sub.add_parser("manifest", help="the scorecard's manifest for one order and one arm")
    m.add_argument("--campaign", required=True, help="the campaign file that says what the grid is")
    m.add_argument("--root", required=True, help="the tree that campaign wrote (its k5 directory)")
    m.add_argument("--order", required=True, help="the sequence: sqlfirst or mathsfirst")
    m.add_argument("--arm", required=True, help="the stage-2-onward arm: none, rehearse10, ...")
    m.add_argument("--name", default=None, help="the sequence's name (default: k5-<order>-<arm>)")
    m.add_argument("--out", required=True, help="the manifest file to write; never overwritten")

    e = sub.add_parser("estimate", help="the GPU-hours a campaign will cost, before anything runs")
    e.add_argument("--campaign", required=True)
    e.add_argument("--out", default=None, help="a new directory for estimate.json and estimate.md")
    e.add_argument("--seconds-per-step", type=float, default=TRAIN_SECONDS_PER_STEP)
    e.add_argument("--seconds-per-answer", type=float, default=SCORE_SECONDS_PER_ANSWER)
    e.add_argument("--probe-seconds", type=float, default=PROBE_SECONDS)

    args = parser.parse_args(argv)
    return {"pool": cmd_pool, "manifest": cmd_manifest, "estimate": cmd_estimate}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
