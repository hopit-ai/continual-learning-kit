#!/usr/bin/env python3
"""Run a campaign of experiment rows on any machine, with a pilot gate.

    python runner.py plan    campaigns/toy.yaml                # print what would run; executes nothing
    python runner.py prepare campaigns/toy.yaml --all          # every row's IO, on a CPU machine, before any GPU is held
    python runner.py run     campaigns/toy.yaml --row pilot    # run one row
    python runner.py run    campaigns/toy.yaml --all           # run every row in order; stops at a failure a later row needs
    python runner.py status campaigns/toy.yaml                 # verdict of every row
    python runner.py batch  campaigns/a.yaml campaigns/b.yaml  # prepare --all then run --all for each, in order;
                                                               # stops at the first campaign that does not pass
    python runner.py batch  --plan campaigns/a.yaml campaigns/b.yaml   # every campaign's plan; executes nothing
    python runner.py batch  --seeds 0-4 campaigns/a.yaml       # plan, prepare, run and batch: only seeds 0..4

The rule this file enforces: **no row runs big unless the same experiment ran small first and
passed a bar written in advance.** A row marked `pilot: true` is judged against its `bars` when it
finishes. Every later row needs every earlier pilot's latest verdict to be PASS, plus any rows it names
in `needs`, and is REFUSED otherwise: nothing is launched, exit code 2. The campaign file STATES the pilot
gate too: each pilot needs the previous pilot, every other row needs the most recent pilot before it, and
a row that does not reach every earlier pilot through `needs` is an implicit order (below).

`wants` is ordering only: a row runs after the rows it wants when they are scheduled, but a wanted row
that failed, was refused or is not scheduled does not block it at the gate (a report row wants its
per-seed scorings and needs only the pilots and the base scorings; its bars judge what it found).
After a row fails or is refused, `run --all` (and so `batch`) goes on ("continuing: ..."): the later rows that
need it, directly or through rows they need, are refused when reached, and every other row still runs. Only a
failed PILOT stops the run there, naming the row. It exits non-zero either way. A row that only wants the failed row runs, and its start.json and verdict.json
list it under `wants_not_passed`.

`--seeds 0,1,2` or `--seeds 0-4` (plan, prepare, run, batch) skips every row whose `seed:` is outside the
set. A row that belongs to one repeat states its seed (`seed: 3`); a row without the field falls back to
`seed<N>` in its id, or else in its NAME env (campaigns written before the field), and a row with neither
is kept. A skipped row is neither run nor recorded and blocks no row that only wants it; a row that NEEDS
one (and it has not passed already) is refused, naming it. Every start.json and verdict.json records the
seed filter in use.

**All IO happens before a GPU is held.** A row may list `prepare` commands (downloads, format
conversions, tokenising, anything that needs no GPU) and `requires` paths (the model directory, the
data files). `prepare` runs them on any CPU machine, for all rows at once and without waiting for a
pilot, since fetching data early wastes nothing. `run` REFUSES a row whose preparation has not
passed or whose required paths are missing, before its command starts, so an allocated GPU never
waits on a download. Phase 1 paid for exactly that: 700 seconds of cold dataset download inside a
GPU container, and GPUs idle for a quarter to 85 percent of a step while scoring ran serially.
Every such order must be STATED: a row that requires a path another row's prepare (or command) writes must
name that row in `needs`, directly or through a row it needs. `plan` prints each one that is only implied
("IMPLICIT ORDER: ...") and exits 2, and `prepare`/`run`/`batch` refuse the campaign before doing anything:
file order is free here, but a parallel dispatcher sees two independent rows and starts both.
One exception: a row that requires a path under {work} which an EARLIER row's command writes (a model
that row merges, a file it generates) cannot be prepared before that row has run. `prepare --all`
defers it ("deferred: ..."), records nothing, and `run --all` prepares it right before running it.
A missing path nothing in the campaign writes (a dataset root, a model outside {work}) still fails.

What is recorded, and when. `start.json` is written and fsynced BEFORE the command is launched, so a
crash still leaves the identity of what ran. `verdict.json` is written after. Nothing is ever
overwritten: a second try of a row is a new attempt (`--attempt 2`) in its own directory, and the
gate reads the latest attempt.

A PASS is about the inputs it read. Every start.json and verdict.json records, under `inputs`, which attempt of each
row it reads stood at when it started: its attempt and verdict (the rows it needs, pilots left out; the rows it wants,
pilots included). When `run --all` meets a row that has passed but a row it read has since been run again, whatever
the outcome (a training run retried after its scoring passed, a run that had failed when the report was written, a
check retried and failed), it prints "AGAIN ..." and runs the row as a new attempt instead of skipping it, so a retry
can never leave an old scoring or an old report standing. A row may list under `order` the wanted rows it does not
read and wants only so as to run after them (one scoring after another on one GPU): those make nothing out of date.
`order` is a subset of `wants`.

A missing file, a missing key or a crashed command is a FAIL with its reason, never a silent PASS.
Standard library plus PyYAML (JSON campaigns need only the standard library). No network, no GPU
code, no Modal: the rows' own commands decide what hardware they need.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import signal
import tempfile
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA = "kit-campaign.v1"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
AGGREGATES = ("first", "last", "min", "max", "mean")


class CampaignError(ValueError):
    """The campaign file is wrong; nothing may run."""


# ------------------------------------------------------------------------------------ loading
def load_campaign(path: Path) -> dict:
    text = path.read_text()
    if path.suffix == ".json":
        campaign = json.loads(text)
    else:
        import yaml                                                        # noqa: PLC0415
        campaign = yaml.safe_load(text)
    if campaign.get("schema") != SCHEMA:
        raise CampaignError("schema must be %r, found %r" % (SCHEMA, campaign.get("schema")))
    rows = campaign.get("rows") or []
    ids = [row.get("id") for row in rows]
    if not rows or any(not i for i in ids) or len(set(ids)) != len(ids):
        raise CampaignError("rows need unique, non-empty ids: %r" % ids)
    seen_pilots: list = []
    for row in rows:
        if not isinstance(row.get("command"), list) or not row["command"]:
            raise CampaignError("row %s: command must be a non-empty list" % row["id"])
        if row.get("pilot") and not row.get("bars"):
            raise CampaignError("row %s: a pilot without bars cannot gate anything" % row["id"])
        for bar in row.get("bars") or []:
            if "min" not in bar and "max" not in bar:
                raise CampaignError("row %s bar %s: needs min and/or max" % (row["id"], bar.get("name")))
            if bar.get("agg", "last") not in AGGREGATES:
                raise CampaignError("row %s bar %s: agg must be one of %s" % (row["id"], bar.get("name"), AGGREGATES))
        for command in row.get("prepare") or []:
            if not isinstance(command, list) or not command:
                raise CampaignError("row %s: each prepare entry must be a non-empty command list" % row["id"])
        # every earlier pilot is ALWAYS required; a row's own `needs` adds to that, it never replaces it.
        # What the file states is kept apart, so `implicit_edges` can tell a stated pilot edge from the gate's.
        row["_stated_needs"] = list(row.get("needs") or [])
        row["needs"] = list(dict.fromkeys(list(seen_pilots) + row["_stated_needs"]))
        unknown = [n for n in row["needs"] if n not in ids[:ids.index(row["id"])]]
        if unknown:
            raise CampaignError("row %s needs rows that do not come before it: %s" % (row["id"], unknown))
        if not isinstance(row.get("wants") or [], list):
            raise CampaignError("row %s: wants must be a list of row ids" % row["id"])
        row["wants"] = list(dict.fromkeys(row.get("wants") or []))
        unknown = [w for w in row["wants"] if w not in ids[:ids.index(row["id"])]]
        if unknown:
            raise CampaignError("row %s wants rows that do not come before it: %s" % (row["id"], unknown))
        # `order`: the wanted rows this row does NOT read, wanted only so that it runs after them (one scoring after
        # another on one GPU). They stay in `wants`, so a dispatcher that reads only `needs` and `wants` still puts
        # the rows in order; here they are left out of what makes a PASS out of date.
        if not isinstance(row.get("order") or [], list):
            raise CampaignError("row %s: order must be a list of row ids" % row["id"])
        row["order"] = list(dict.fromkeys(row.get("order") or []))
        stray = [o for o in row["order"] if o not in row["wants"]]
        if stray:
            raise CampaignError("row %s: order names rows it does not want: %s (order is a subset of wants)" % (row["id"], stray))
        if "seed" in row and (not isinstance(row["seed"], int) or isinstance(row["seed"], bool) or row["seed"] < 0):
            raise CampaignError("row %s: seed must be a whole number, found %r" % (row["id"], row["seed"]))
        if row.get("pilot"):
            seen_pilots.append(row["id"])
    campaign["_path"] = str(path.resolve())
    campaign["_seeds"], campaign["_skipped"] = None, []
    campaign["_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return campaign


def work_root(campaign: dict) -> Path:
    name = campaign.get("workdir_env", "WORK")
    value = os.environ.get(name)
    if not value:
        raise CampaignError("set %s to the directory that holds outputs" % name)
    return Path(value).resolve()


_SEED = re.compile(r"seed(\d+)")


def seed_of(row: dict) -> int | None:
    """The seed a row carries: its `seed:` field, or (a campaign written before the field) `seed<N>` in its id, or
    else in its NAME env; None for a row with no seed."""
    if "seed" in row:
        return row["seed"]
    for text in (row["id"], str((row.get("env") or {}).get("NAME", ""))):
        match = _SEED.search(text)
        if match:
            return int(match.group(1))
    return None


def parse_seeds(text: str) -> list:
    """`0,1,2,3,4` or `0-4` (or a mix, `0-2,7`) as a sorted list of seeds."""
    seeds = set()
    for part in (p.strip() for p in text.split(",") if p.strip()):
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if not match or int(match.group(2) or match.group(1)) < int(match.group(1)):
            raise CampaignError("--seeds takes numbers and rising ranges, e.g. 0,1,2 or 0-4; found %r" % part)
        seeds.update(range(int(match.group(1)), int(match.group(2) or match.group(1)) + 1))
    if not seeds:
        raise CampaignError("--seeds names no seed: %r" % text)
    return sorted(seeds)


def filter_seeds(campaign: dict, seeds: list | None) -> list:
    """Record the seed filter on the campaign and print each row it skips, once; returns the skipped ids."""
    campaign["_seeds"] = seeds
    campaign["_skipped"] = [] if seeds is None else [row["id"] for row in campaign["rows"]
                                                     if seed_of(row) is not None and seed_of(row) not in seeds]
    for rid in campaign["_skipped"]:
        print("skipped (seed filter): %s" % rid)
    return campaign["_skipped"]


def fill(template: str, row: dict, campaign: dict, attempt: int) -> str:
    values = {"kit": str(Path(__file__).resolve().parent), "work": str(work_root(campaign)), "row": row["id"],
              "attempt": str(attempt), "campaign": campaign["name"]}
    out = str(template)
    for key, value in values.items():
        out = out.replace("{%s}" % key, value)
    return out


def resolve(row: dict, campaign: dict, attempt: int) -> dict:
    env = {k: fill(v, row, campaign, attempt) for k, v in (row.get("env") or {}).items()}
    for key, value in list(env.items()):                       # env values may refer to each other once
        for other, other_value in env.items():
            value = value.replace("{env.%s}" % other, other_value)
        env[key] = value

    def full(text):
        text = fill(text, row, campaign, attempt)
        for key, value in env.items():
            text = text.replace("{env.%s}" % key, value)
        return text
    return {"prepare": [[full(part) for part in command] for command in row.get("prepare") or []],
            "requires": [full(path) for path in row.get("requires") or []],
            "command": [full(part) for part in row["command"]], "env": env,
            "cwd": full(row["cwd"]) if row.get("cwd") else None,
            "bars": [{**bar, "source": full(bar["source"])} for bar in row.get("bars") or []]}


# ------------------------------------------------------------------------------------ state
def row_dir(campaign: dict, row_id: str) -> Path:
    return work_root(campaign) / "campaign" / campaign["name"] / row_id


def attempts(campaign: dict, row_id: str) -> list:
    base = row_dir(campaign, row_id)
    found = [int(p.name.split("-")[1]) for p in base.glob("attempt-*") if p.name.split("-")[1].isdigit()] if base.is_dir() else []
    return sorted(found)


def latest_verdict(campaign: dict, row_id: str) -> dict | None:
    for number in reversed(attempts(campaign, row_id)):
        path = row_dir(campaign, row_id) / ("attempt-%d" % number) / "verdict.json"
        if path.is_file():
            return json.loads(path.read_text())
        return {"verdict": "NO_VERDICT", "attempt": number, "reason": "attempt %d started and wrote no verdict" % number}
    return None


def latest_preparation(campaign: dict, row_id: str) -> dict | None:
    base = row_dir(campaign, row_id)
    found = sorted(base.glob("prepare-*.json"), key=lambda p: int(p.stem.split("-")[1])) if base.is_dir() else []
    return json.loads(found[-1].read_text()) if found else None


def not_ready(campaign: dict, row: dict) -> list:
    """Why this row's inputs are not in place; empty means a GPU would not wait on IO."""
    spec = resolve(row, campaign, 1)
    reasons = []
    if spec["prepare"]:
        done = latest_preparation(campaign, row["id"])
        if done is None:
            reasons.append("its prepare steps have not run (python runner.py prepare ... --row %s; no GPU needed)" % row["id"])
        elif done["verdict"] != "PASS":
            reasons.append("its latest preparation is %s (%s)" % (done["verdict"], done.get("reason")))
    missing = [path for path in spec["requires"] if not Path(path).exists()]
    if missing:
        reasons.append("required paths are missing: %s" % ", ".join(missing))
    return reasons


_OUTPUT = re.compile(r"(?:--out|--output|--out[-_]dir|--output[-_]dir|--local[-_]dir|--save[-_]dir|-o)(?:=|\s+)[\"']?"
                     r"([^\"'\s;&|)]+)|>\s*[\"']?([^\"'\s;&|)]+)")


def outputs(spec: dict) -> set:
    """The paths a row's command says it writes: the value after --out, --output(-dir), --local-dir, --save-dir, -o
    or a `>` redirect, and every env value whose name contains OUT (the kit's convention, `OUT: {work}/runs/...`)."""
    found = {a or b for a, b in _OUTPUT.findall("\n".join(spec["command"]))}
    found.update(value for key, value in spec["env"].items() if "OUT" in key.upper())
    return {path.rstrip("/") for path in found}


def prepared(spec: dict) -> set:
    """The paths a row's prepare steps say they write, read the way `outputs` reads a command."""
    return outputs({"command": [part for command in spec["prepare"] for part in command], "env": {}})


def writers(campaign: dict, path: str, written: list) -> list:
    """The ids in `written` ([(row id, paths it writes)]) that write `path`, or a folder it lies in below {work}
    itself, most specific path first. A path outside {work} (a dataset root, a model elsewhere) has no writer."""
    work = str(work_root(campaign))
    if not path.startswith(work + os.sep):
        return []
    candidates, parent = [path], os.path.dirname(path)
    while parent.startswith(work + os.sep):
        candidates.append(parent)
        parent = os.path.dirname(parent)
    return list(dict.fromkeys(rid for candidate in candidates for rid, paths in written if candidate in paths))


def producers(campaign: dict, row: dict) -> dict:
    """{required path: id of the first EARLIER row whose command writes it}, for this row's requires under {work}.

    Such a path is an input that an earlier GPU row writes (K2's `small-before` reads the model `small-make-damaged`
    merges with `--out`), so it cannot exist when `prepare --all` runs before anything has run. A row writes the path
    when it names the path itself as an output, or a folder the path lies in (below {work} itself)."""
    if not row.get('requires'):return {}
    ids = [r["id"] for r in campaign["rows"]]
    written = [(other["id"], outputs(resolve(other, campaign, 1))) for other in campaign["rows"][:ids.index(row["id"])]]
    found = {}
    for path in resolve(row, campaign, 1)["requires"]:
        owners = writers(campaign, path, written)
        if owners:
            found[path] = owners[0]
    return found


def implicit_edges(campaign: dict) -> list:
    """[(row, required path, writer row, "prepare" or "command")] for every order a campaign implies without stating,
    and (row, None, pilot, "pilot") for every earlier pilot a row does not reach through its stated `needs`.

    A row that requires a path another row's prepare steps (or its command, see `outputs`) write depends on that row,
    and says so only when its `needs`, followed transitively, reach one of the path's writers. `runner.py` gets the
    order free by running rows in file order; a parallel dispatcher sees two independent rows and starts both (K1c,
    30 September: `base17b-gsm8k` required the GSM8K files `base17b-finqa`'s prepare builds). A row whose own prepare
    writes the path depends on nobody for it. The writer named is the earliest one, so adding it to `needs` fixes it.
    A `wants` edge states a file order too (a report reading what a seed row wrote may want it).

    The pilot gate is an order as well (30 September: the partner's dispatcher launched `base-forget-2` beside the
    pilot `base-forget-1`, and the runner refused it), so a row must reach every earlier pilot through `needs`,
    followed transitively; the pilots the gate adds by itself, and `wants`, do not count."""
    rows, order = campaign["rows"], {row["id"]: n for n, row in enumerate(campaign["rows"])}
    specs = {row["id"]: resolve(row, campaign, 1) for row in rows}
    by_prepare = [(rid, prepared(spec)) for rid, spec in specs.items()]
    by_command = [(rid, outputs(spec)) for rid, spec in specs.items()]
    needs = {row["id"]: row.get("_stated_needs", row["needs"]) for row in rows}
    stated = {row["id"]: needs[row["id"]] + list(row.get("wants") or []) for row in rows}

    def upstream(rid: str, edges_of: dict) -> set:
        seen, todo = set(), list(edges_of[rid])
        while todo:
            other = todo.pop()
            if other not in seen:
                seen.add(other)
                todo.extend(edges_of.get(other, []))
        return seen

    edges, pilots = [], []
    for row in rows:
        rid, reached = row["id"], upstream(row["id"], stated)
        for path in specs[rid]["requires"]:
            if writers(campaign, path, [(rid, prepared(specs[rid]))]):
                continue
            found = {}
            for how, written in (("command", by_command), ("prepare", by_prepare)):   # prepare wins when both write it
                found.update({other: how for other in writers(campaign, path, [w for w in written if w[0] != rid])})
            if found and not reached & set(found):
                writer = min(found, key=order.get)
                edges.append((rid, path, writer, found[writer]))
        gated = upstream(rid, needs)
        edges.extend((rid, None, pilot, "pilot") for pilot in pilots if pilot not in gated)
        if row.get("pilot"):
            pilots.append(rid)
    return edges


def refuse_implicit(campaign: dict) -> bool:
    """Print every implicit edge; True when there is any, so the campaign must not be planned, prepared or run."""
    edges = implicit_edges(campaign)
    for rid, path, writer, how in edges:
        if how == "pilot":
            print("IMPLICIT ORDER: %s comes after pilot %s but does not need it" % (rid, writer))
        else:
            print("IMPLICIT ORDER: %s requires %s, written by %s's %s, but does not need %s" % (rid, path, writer, how, writer))
    return bool(edges)


def deferred(campaign: dict, row: dict) -> dict:
    """The missing required paths that an earlier row will produce, when those are ALL that is missing; else {}.
    A row with any other missing path is prepared as usual, so an external input still fails loudly."""
    spec = resolve(row, campaign, 1)
    missing = [path for path in spec["requires"] if not Path(path).exists()]
    produced = producers(campaign, row)
    if not missing or any(path not in produced for path in missing):
        return {}
    return {path: produced[path] for path in missing}


def prepare_row(campaign: dict, row: dict) -> int:
    spec = resolve(row, campaign, 1)
    if not spec["prepare"] and not spec["requires"]:
        return EXIT_OK
    base = row_dir(campaign, row["id"])
    number = len(list(base.glob("prepare-*.json"))) + 1 if base.is_dir() else 1
    steps, reason, clock = [], None, time.monotonic()
    for command in spec["prepare"]:
        print("PREPARE %s: %s" % (row["id"], " ".join(command)))
        started = time.monotonic()
        done = bounded_command(command, timeout=row.get("prepare_timeout_seconds", 3600), env={**os.environ, **spec["env"]}, cwd=spec["cwd"])
        steps.append({"command": command, **done})
        if done["returncode"] != 0:
            reason = done["failure_type"] + ": prepare command exited %d: %s" % (done["returncode"], " ".join(command))
            break
    missing = [path for path in spec["requires"] if not Path(path).exists()]
    if reason is None and missing:
        reason = "required paths are missing after preparation: %s" % ", ".join(missing)
    verdict = "PASS" if reason is None else "FAIL"
    write_durably(base / ("prepare-%d.json" % number), {"schema": SCHEMA, "row": row["id"], "attempt": number, "verdict": verdict,
                                                       "reason": reason, "steps": steps, "requires": spec["requires"],
                                                       "seconds": round(time.monotonic() - clock, 1)})
    print("%s prepare %s%s" % (verdict, row["id"], ": " + reason if reason else ""))
    return EXIT_OK if verdict == "PASS" else EXIT_FAILED


def write_durably(path: Path, payload: dict) -> None:
    """Publish complete fsynced bytes atomically and exclusively, including races.

    POSIX link is the portable atomic rename-if-absent equivalent: the target
    acquires the fully written inode only if absent. Never replace evidence.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.publish-', delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if temporary is not None: os.unlink(temporary)


def bounded_command(command, *, timeout, log=None, env=None, cwd=None):
    """A CPU command cannot hold an allocation indefinitely, including children."""
    if isinstance(timeout, bool) or not 0 < float(timeout) < float('inf'):
        raise CampaignError('command needs a finite positive deadline')
    started = time.monotonic()
    handle = Path(log).open('w') if log else None
    failure = None
    try:
        process = subprocess.Popen(command, env=env, cwd=cwd, start_new_session=True,
                                   stdout=handle, stderr=subprocess.STDOUT if handle else None)
        try: code = process.wait(timeout=float(timeout))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            code, failure = 124, 'cpu_deadline'
        if code and failure is None: failure = 'cpu_exit'
    except OSError as error:
        code, failure = 127, 'cpu_start'
        if handle: handle.write(str(error)+'\n')
    finally:
        if handle: handle.close()
    return {'returncode': code, 'failure_type': failure, 'seconds': time.monotonic()-started,
            'timeout_seconds': float(timeout)}


# ------------------------------------------------------------------------------------ bars
def _values(source: Path, key: str, where: dict | None) -> list:
    text = source.read_text()
    records = [json.loads(line) for line in text.split("\n") if line.strip()] if source.suffix == ".jsonl" else [json.loads(text)]
    out = []
    for record in records:
        if where and any(record.get(field) != wanted for field, wanted in where.items()):
            continue
        scope = record["data"] if isinstance(record.get("data"), dict) and key not in record else record
        if key in scope and isinstance(scope[key], (int, float)) and not isinstance(scope[key], bool):
            out.append(float(scope[key]))
    return out


def judge_bar(bar: dict) -> dict:
    result = {"name": bar.get("name", bar["key"]), "source": bar["source"], "key": bar["key"], "agg": bar.get("agg", "last"),
              "min": bar.get("min"), "max": bar.get("max"), "value": None, "ok": False, "reason": None}
    source = Path(bar["source"])
    if not source.is_file():
        result["reason"] = "file not found"
        return result
    try:
        values = _values(source, bar["key"], bar.get("where"))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        result["reason"] = "unreadable: %s" % exc
        return result
    if not values:
        result["reason"] = "key not found%s" % (" where %s" % bar["where"] if bar.get("where") else "")
        return result
    pick = {"first": lambda v: v[0], "last": lambda v: v[-1], "min": min, "max": max, "mean": statistics.fmean}[result["agg"]]
    value = pick(values)
    result["value"], result["n"] = value, len(values)
    low, high = bar.get("min"), bar.get("max")
    result["ok"] = (low is None or value >= low) and (high is None or value <= high)
    if not result["ok"]:
        result["reason"] = "%.6g is outside [%s, %s]" % (value, low, high)
    return result


# ------------------------------------------------------------------------------------ commands
def gate(campaign: dict, row: dict) -> list:
    """The reasons this row may not run; empty means it may."""
    reasons = []
    for needed in row["needs"]:
        verdict = latest_verdict(campaign, needed)
        if needed in campaign.get("_skipped", ()) and (verdict or {}).get("verdict") != "PASS":
            reasons.append("needs %s, which the seed filter skips (--seeds %s) and which has not passed"
                           % (needed, ",".join(map(str, campaign["_seeds"]))))
        elif verdict is None:
            reasons.append("needs %s, which has not run" % needed)
        elif verdict["verdict"] != "PASS":
            reasons.append("needs %s, whose latest verdict is %s (%s)" % (needed, verdict["verdict"], verdict.get("reason") or "see its verdict.json"))
    return reasons


def inputs_of(campaign: dict, row: dict) -> dict:
    """{row id: [attempt, verdict] of that row's latest attempt, or None if it never ran} for every row this one reads:
    the rows it needs, pilots left out (a pilot it merely has to wait for is a gate, not a producer: re-running one
    must not make a whole campaign stale), and the rows it wants, pilots included (a report that wants a pilot reads
    what that pilot wrote), except those it lists under `order`."""
    pilots = {r["id"] for r in campaign["rows"] if r.get("pilot")}
    order = set(row.get("order") or [])
    read = [n for n in row["needs"] if n not in pilots] + [w for w in row.get("wants") or [] if w not in order]
    found = {}
    for other in dict.fromkeys(read):
        verdict = latest_verdict(campaign, other)
        found[other] = [verdict.get("attempt"), verdict.get("verdict")] if verdict else None
    return found


def stale_inputs(campaign: dict, row: dict, verdict: dict) -> list:
    """The rows this PASS read whose latest attempt is no longer the one it read: `run --all` then runs the row again
    as a new attempt instead of skipping it. That covers a training run retried after its scoring passed (the scoring
    is of a model that is no longer the run's), a report written while a run had failed or before a model was
    re-scored, and a row that had passed and has since been run again and FAILED (a check retried and failed leaves no
    earlier report standing; a failed run retried and failed again has other partial rollouts to summarise).
    A wanted row listed under `order` is not read (the row wants it only so as to run after it: one scoring after
    another on one GPU) and makes nothing stale; without that, re-scoring one model would re-score every later one,
    which the dry run of the pilot did, ninety models after one retry. A verdict written before `inputs` was recorded
    is never stale."""
    then = verdict.get("inputs")
    if not isinstance(then, dict):
        return []
    return sorted(other for other, state in inputs_of(campaign, row).items() if state is not None and then.get(other) != state)


def run_row(campaign: dict, row: dict, attempt: int | None) -> int:
    dispatch_clock=time.monotonic()
    if campaign.get('v4',{}).get('phase')=='phase0' and row['id']=='verify-prepare':
        from kit.v4_phase0 import verification_row_cap
        cap=verification_row_cap(work_root(campaign))
        row={**row,'timeout_seconds':cap,'allocation_cpu_cap_seconds':cap}
    reasons = gate(campaign, row) + not_ready(campaign, row)
    if reasons:
        print("REFUSED %s: %s" % (row["id"], "; ".join(reasons)))
        return EXIT_REFUSED
    if campaign.get('v4') and os.environ.get('SLURM_JOB_ID'):
        ledger_path=work_root(campaign)/'k8b4/containment/allocation-ledger.json'
        ledger=json.loads(ledger_path.read_text())
        entry=ledger['allocations'][os.environ['SLURM_JOB_ID']]
        if campaign['v4'].get('phase')=='phase0':
            from kit.v4_phase0 import allocation_cap
            try: allocation_cap(ledger)
            except ValueError as exc:
                from kit.p4_contain import hard_stop
                hard_stop(work_root(campaign),'block reached: '+str(exc));return EXIT_REFUSED
        from kit.v4_budget import stamp
        command=resolve(row,campaign,1)['command']
        cap=row.get('timeout_seconds',3600)
        if '--time-cap' in command:
            cap=int(command[command.index('--time-cap')+1])
            if campaign['v4'].get('phase')=='main':
                from kit.v4_campaign_ops import registered_caps
                cap=registered_caps(work_root(campaign))[row['id'].split('-')[0]]
        elif campaign['v4'].get('phase')=='main' and row['id'] not in ('containment-selftest','rewrite-allocation-selftest','scientific-allocation-selftest','environment','prepare','owner-registration-gate'):
            from kit.v4_campaign_ops import registered_caps
            cap=registered_caps(work_root(campaign))['merge' if row['id'].startswith('merge-') else 'cpu']
        if float(stamp(entry['planned_end'])) < time.time()+cap:
            print('allocation exhausted; resume in a new allocation: '+row['id'])
            return 75  # No start/attempt exists; not an infrastructure retry.
    done = attempts(campaign, row["id"])
    number = attempt if attempt is not None else (done[-1] + 1 if done else 1)
    if number in done:
        print("REFUSED %s: attempt %d already exists; pass --attempt %d" % (row["id"], number, done[-1] + 1))
        return EXIT_REFUSED
    spec = resolve(row, campaign, number)
    if campaign.get('v4',{}).get('phase')=='main' and '--time-cap' in spec['command']:
        from kit.v4_campaign_ops import registered_caps
        caps=registered_caps(work_root(campaign))
        operation=row['id'].split('-')[0]
        if operation=='baseline':operation='score'
        spec['command'][spec['command'].index('--time-cap')+1]=str(caps[operation])
    if campaign.get('v4'):
        cpu_timeout=row.get('timeout_seconds',3600)
        if campaign['v4'].get('phase')=='main' and '--time-cap' not in spec['command'] and row['id'] not in ('containment-selftest','rewrite-allocation-selftest','scientific-allocation-selftest','environment','prepare','owner-registration-gate'):
            from kit.v4_campaign_ops import registered_caps
            cpu_timeout=registered_caps(work_root(campaign))['merge' if row['id'].startswith('merge-') else 'cpu']
            row={**row,'timeout_seconds':cpu_timeout}
        spec['env']['V4_COMMAND_TIMEOUT']=str(cpu_timeout)
        if '--time-cap' in spec['command']:spec['env']['V4_COMMAND_TIMEOUT']=spec['command'][spec['command'].index('--time-cap')+1]
    out = row_dir(campaign, row["id"]) / ("attempt-%d" % number)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    unmet = [w for w in row.get("wants") or [] if (latest_verdict(campaign, w) or {}).get("verdict") != "PASS"]
    inputs = inputs_of(campaign, row)                                  # which attempt of each producer this attempt reads
    if campaign.get('v4') and any(str(arg).endswith('/p4_contain.py') for arg in spec['command']) and 'row' in spec['command']:
        # Review occurs before a containment row or its optimizer can launch.
        spec['env']['OFFLOAD'] = '0'
        spec['env']['V4_RUNNER_ATTEMPT'] = str(number)
        if done:
            previous_dir=row_dir(campaign,row['id'])/('attempt-%d'%done[-1])
            previous=json.loads((previous_dir/'start.json').read_text())
            verdict_path=previous_dir/'verdict.json'
            if verdict_path.exists() and json.loads(verdict_path.read_text()).get('verdict')=='PASS':
                print('REFUSED %s: first valid result cannot be replaced'%row['id']);return EXIT_REFUSED
            try:
                from kit.v4_retry import admit, automatic
                review_path=work_root(campaign)/'v4/retries'/campaign['name']/row['id']/('attempt-%d.json'%number)
                if not review_path.exists():automatic(work_root(campaign),campaign['name'],row['id'],number,previous)
                review=json.loads(review_path.read_text())
                spec['env']['OFFLOAD']='1' if review['reason_code']=='out_of_memory' else previous['env'].get('OFFLOAD','0')
                if review.get('state_policy')=='resume_valid_state':spec['env']['V4_RESUME_PATH']=review['saved_state_path']
                admit(work_root(campaign),campaign['name'],row['id'],number,
                      {'attempt':number,'command':spec['command'],'env':spec['env'],'inputs':inputs},previous)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                print('REFUSED %s: infrastructure review: %s'%(row['id'],exc));return EXIT_REFUSED
    if unmet:
        print("note %s: runs without %d wanted row(s) that have not passed: %s" % (row["id"], len(unmet), ", ".join(unmet)))
    write_durably(out / "start.json", {"schema": SCHEMA, "campaign": campaign["name"], "campaign_sha256": campaign["_sha256"],
                                       "row": row["id"], "attempt": number, "pilot": bool(row.get("pilot")), "needs": row["needs"],
                                       "wants": row.get("wants") or [], "wants_not_passed": unmet, "seeds": campaign.get("_seeds"),
                                       "inputs": inputs, "started_at": started, **{k: spec[k] for k in ("command", "env", "cwd")}})
    print("RUN %s attempt %d: %s" % (row["id"], number, " ".join(spec["command"])))
    clock = time.monotonic()
    # GPU commands retain their independent Slurm cap; this outer deadline also
    # bounds wrappers. CPU rows explicitly register their maximum wall seconds.
    deadline = row.get('timeout_seconds', 3600)
    if '--time-cap' in spec['command']:
        deadline = int(spec['command'][spec['command'].index('--time-cap')+1]) + (0 if campaign.get('v4',{}).get('phase')=='phase0' else 120)
    command_result = bounded_command(spec['command'], timeout=deadline, log=out/'output.log',
        env={**os.environ, **spec['env']}, cwd=spec['cwd'])
    command_seconds=time.monotonic()-clock
    returncode = command_result['returncode']
    with (out/'output.log').open() as log:
        for line in log: sys.stdout.write(line)
    bars = [judge_bar(bar) for bar in spec["bars"]]
    failed = [b for b in bars if not b["ok"]]
    if returncode != 0:
        verdict, reason = "FAIL", "command exited %d" % returncode
    elif failed:
        verdict, reason = "FAIL", "; ".join("%s: %s" % (b["name"], b["reason"]) for b in failed)
    else:
        verdict, reason = "PASS", None
    write_durably(out / "verdict.json", {"schema": SCHEMA, "row": row["id"], "attempt": number, "verdict": verdict, "reason": reason,
                                         "returncode": returncode, "failure_type": command_result["failure_type"], "timeout_seconds": deadline, "bars": bars, "started_at": started,
                                         "seeds": campaign.get("_seeds"), "wants_not_passed": unmet, "inputs": inputs,
                                         "seconds": round(time.monotonic() - clock, 1)})
    if campaign.get('v4'):
        write_durably(out/'runner-overhead.json',{'ok':True,'allocation_id':os.environ.get('SLURM_JOB_ID'),
            'row':row['id'],'wall_seconds':max(0,time.monotonic()-dispatch_clock-command_seconds),
            'command_seconds':command_seconds,'campaign_sha256':campaign['_sha256']})
    print("%s %s%s" % (verdict, row["id"], ": " + reason if reason else ""))
    if verdict=='FAIL' and campaign.get('v4') and 'row' in spec['command'] and any(str(arg).endswith('/p4_contain.py') for arg in spec['command']):
        try:
            if (work_root(campaign)/'k8b4/containment/v4-stop.json').exists():return 75
            from kit.v4_retry import failure_record
            _,contained=failure_record(work_root(campaign),spec['command'])
            if contained.get('stop_class') in ('allocation_preempted','hard'):return 75
            if number<3:
                from kit.v4_retry import automatic
                automatic(work_root(campaign),campaign['name'],row['id'],number+1,
                    json.loads((out/'start.json').read_text()))
                return run_row(campaign,row,None)
        except (OSError,ValueError,KeyError,TypeError):pass
    return EXIT_OK if verdict == "PASS" else EXIT_FAILED


def cmd_plan(campaign: dict) -> int:
    skipped = campaign.get("_skipped") or []
    print("campaign %s (%s rows), outputs under %s" % (campaign["name"], len(campaign["rows"]), work_root(campaign)))
    if campaign.get("_seeds") is not None:
        print("seed filter: %s (%d rows skipped)" % (",".join(map(str, campaign["_seeds"])), len(skipped)))
    for row in campaign["rows"]:
        if row["id"] in skipped:
            continue
        spec = resolve(row, campaign, 1)
        print("\n[%s]%s needs: %s" % (row["id"], " PILOT" if row.get("pilot") else "", ", ".join(row["needs"]) or "nothing"))
        if seed_of(row) is not None:
            print("  seed: %d%s" % (seed_of(row), "" if "seed" in row else " (from its name)"))
        if row.get("wants"):
            print("  wants: %s" % ", ".join(row["wants"]))
        for needed in (n for n in row["needs"] if n in skipped):
            print("  refused unless it already passed: needs %s, which the seed filter skips" % needed)
        for command in spec["prepare"]:
            print("  prepare (no GPU): %s" % " ".join(command))
        for path in spec["requires"]:
            print("  requires: %s" % path)
        print("  command: %s" % " ".join(spec["command"]))
        for key, value in spec["env"].items():
            print("  env %s=%s" % (key, value))
        for bar in spec["bars"]:
            print("  bar %s: %s of %s in [%s, %s] from %s" % (bar.get("name", bar["key"]), bar.get("agg", "last"), bar["key"],
                                                             bar.get("min"), bar.get("max"), bar["source"]))
    if not implicit_edges(campaign):
        return EXIT_OK
    print()
    refuse_implicit(campaign)
    return EXIT_REFUSED


def cmd_status(campaign: dict) -> int:
    for row in campaign["rows"]:
        verdict = latest_verdict(campaign, row["id"])
        blocked = gate(campaign, row)
        state = "%s (attempt %s)" % (verdict["verdict"], verdict["attempt"]) if verdict else ("BLOCKED" if blocked else "READY")
        print("%-24s %-22s %s" % (row["id"], state, (verdict or {}).get("reason") or "; ".join(blocked) or ""))
    return EXIT_OK


def cmd_prepare(campaign: dict, rows: list, every: bool) -> int:
    codes = []
    for row in rows:                                                  # every row, even after a failure: IO is independent
        if row["id"] in campaign.get("_skipped", ()):
            continue                                                  # printed once when the filter was applied
        waiting = deferred(campaign, row) if every else {}
        if waiting:                                                   # records nothing, so nothing is gated by it
            for path, owner in waiting.items():
                print("deferred: %s waits for %s (produced by %s or later)" % (row["id"], path, owner))
            continue
        codes.append(prepare_row(campaign, row))
    return EXIT_OK if all(code == EXIT_OK for code in codes) else EXIT_FAILED


def needed_later(campaign: dict, rows: list, failed: str) -> list:
    """The rows after `failed` in `rows` that `run --all` will still run (the seed filter keeps them and they have not
    passed already) and that need it, directly or through the rows they need (the gate's pilots included)."""
    by_id = {row["id"]: row for row in campaign["rows"]}
    memo: dict = {}

    def reaches(rid: str) -> bool:
        if rid not in memo:
            memo[rid] = False                                          # needs point only backwards: no cycle
            memo[rid] = any(n == failed or reaches(n) for n in by_id[rid]["needs"])
        return memo[rid]
    later = rows[[row["id"] for row in rows].index(failed) + 1:]
    return [row["id"] for row in later if row["id"] not in campaign.get("_skipped", ()) and reaches(row["id"])
            and (latest_verdict(campaign, row["id"]) or {}).get("verdict") != "PASS"]


def cmd_run(campaign: dict, rows: list, every: bool, attempt: int | None, *, parallel=True) -> int:
    """Rows in file order, so a row's wants (always earlier rows) run first when scheduled. With --all, a PILOT that
    fails or is refused stops the run when a later scheduled row needs it (every later row does). Any other row that
    fails or is refused costs only the rows that need it, directly or transitively: each of those is refused at its
    own gate, and the run goes on with every independent row and exits with the first failure's code. (Until
    1 October 2026 any needed failure stopped the whole run, so one failed training run of twenty-four would have
    stopped the other twenty-three and the report.) A row the seed filter skips is passed over."""
    jobs=(campaign.get('v4') or {}).get('parallel_jobs',1)
    if type(jobs) is not int or not 1<=jobs<=8:
        raise CampaignError('parallel_jobs must be an integer in 1..8')
    if every and parallel and jobs>1:
        return cmd_run_parallel(campaign,rows,attempt,jobs)
    first = EXIT_OK
    for row in rows:
        if row["id"] in campaign.get("_skipped", ()):
            continue                                                  # printed once when the filter was applied
        passed = latest_verdict(campaign, row["id"]) or {}
        if every and passed.get("verdict") == "PASS":
            stale = stale_inputs(campaign, row, passed)
            if not stale:
                print("SKIP %s: already PASS" % row["id"])
                continue
            print("AGAIN %s: its PASS (attempt %s) read %s, which has been run again since"
                  % (row["id"], passed.get("attempt"), ", ".join(stale)))
        code = None
        if every and producers(campaign, row) and not gate(campaign, row) and not deferred(campaign, row):
            done = latest_preparation(campaign, row["id"])
            if (done or {}).get("verdict") != "PASS":                # deferred by `prepare --all`: its inputs exist now
                print("PREPARE %s now: the rows it waited for have run" % row["id"])
                if prepare_row(campaign, row) != EXIT_OK:
                    print("NOT RUN %s: its preparation failed (see the line above); nothing was launched" % row["id"])
                    code = EXIT_FAILED
        if code is None:
            code = run_row(campaign, row, attempt)
        if code == EXIT_OK:
            continue
        if code==75:return code
        if not every:
            return code
        first = first or code
        needing = needed_later(campaign, rows, row["id"])
        if needing and row.get("pilot"):
            print("STOP: %s did not pass and %s needs it; nothing after it runs" % (row["id"], needing[0]))
            return first
        if needing:
            # Not a pilot: only the rows that need it are lost. Each is refused at its own gate when it is reached
            # (nothing is launched), and every row that does not need it still runs, the reports included.
            print("continuing: %s did not pass; the %d later row(s) that need it will be refused (first: %s), every other row still runs"
                  % (row["id"], len(needing), needing[0]))
            continue
        wanting = [r["id"] for r in rows[rows.index(row) + 1:] if row["id"] in (r.get("wants") or [])
                   and r["id"] not in campaign.get("_skipped", ())]
        print("continuing: %s failed; no later row needs it (wanted by: %s)" % (row["id"], ", ".join(wanting) or "none"))
    return first


def cmd_run_parallel(campaign: dict, rows: list, attempt: int | None, jobs: int) -> int:
    """Execute the stated DAG with existing prepare/run/skip/verdict operations.

    Only campaigns explicitly registering parallel_jobs opt in. A failed pilot
    prevents new dispatch; already admitted independent work finishes and keeps
    its chronology. Resource admission remains the containment row's decision.
    """
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    if type(jobs) is not int or not 1<=jobs<=8:raise CampaignError('parallel_jobs must be an integer in 1..8')
    pending={row['id']:row for row in rows if row['id'] not in campaign.get('_skipped',())}
    scheduled=set(pending);done=set();running={};first=EXIT_OK;stopped=False
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        while pending or running:
            if not stopped:
                for ident,row in list(pending.items()):
                    dependencies=set(row['needs'])|set(row.get('wants') or [])
                    if (dependencies & scheduled) <= done and len(running)<jobs:
                        future=executor.submit(cmd_run,campaign,[row],True,attempt,parallel=False)
                        running[future]=row;del pending[ident]
            if not running:
                if stopped:return first
                if pending:raise CampaignError('parallel dependency graph could not advance')
                break
            finished,_=wait(running,return_when=FIRST_COMPLETED)
            for future in finished:
                row=running.pop(future);code=future.result();done.add(row['id'])
                if code!=EXIT_OK:
                    first=first or code
                    if row.get('pilot') or code==75:
                        stopped=True
                        print('STOP: %s did not pass; no further parallel rows are dispatched' % row['id'])
            if stopped and not running:return first
    return first


def allocation_rows(campaign: dict, stage: str | None = None) -> tuple[list, dict | None]:
    """Select a registered single-block allocation without changing the full DAG."""
    stages = (campaign.get('v4') or {}).get('allocation_stages')
    if not stages:
        if stage: raise CampaignError('--stage requires registered allocation_stages')
        return campaign['rows'], None
    if stage is None:
        if len(stages) != 1: raise CampaignError('select --stage teacher, rewrite or scientific; budget block cannot change inside an allocation')
        stage = next(iter(stages))
    if stage not in stages: raise CampaignError('unknown allocation stage: '+stage)
    registration = stages[stage]
    ids = [r['id'] for r in campaign['rows']]
    rows = campaign['rows'][ids.index(registration['first']):ids.index(registration['last'])+1]
    expected = {'qualification':100, 'scientific':560}
    block = registration['block']
    if expected.get(block) != registration['block_limit']: raise CampaignError('allocation block/limit differs from containment contract')
    for row in rows:
        command = row['command']
        if '{kit}/p4_contain.py' not in command: continue
        if command[2] not in ('selftest','row'): raise CampaignError('allocation may contain only selftest and row containment commands')
        for option,value in (('--block',block),('--block-limit',str(expected[block])),('--ceiling','560')):
            if option not in command or command[command.index(option)+1] != value:
                raise CampaignError('mixed or implicit containment block in allocation stage')
    if rows[0]['command'][1:3] != ['{kit}/p4_contain.py','selftest']: raise CampaignError('first allocation containment command must be selftest')
    return rows, registration


def allocation_selftest(campaign: dict, rows: list, registration: dict | None) -> None:
    """FIRST live containment command on every runner entry, including resume.

    No reconcile/check/tracking precedes this. A successful frozen current-job
    verdict is reused by containment; a refused verdict remains immutable.
    """
    if registration is None or not os.environ.get('SLURM_JOB_ID'): return
    command = resolve(rows[0],campaign,1)['command']
    if bounded_command(command,timeout=600 if (campaign.get('v4') or {}).get('phase')=='phase0' else 900,env=os.environ.copy())['returncode']:
        raise CampaignError('current allocation selftest refused; end the job and return its frozen receipt')
    if (campaign.get('v4') or {}).get('phase') in ('main','phase0'):
        from kit.v4_allocation import verify_plan
        try:verify_plan(work_root(campaign),campaign['v4']['phase'],rows[0]['id'])
        except (OSError,ValueError,KeyError,TypeError) as exc:raise CampaignError(str(exc)) from exc


def cmd_batch(paths: list, plan_only: bool, seeds: list | None = None, stage: str | None = None) -> int:
    """prepare --all then run --all for each campaign in order (inside one, a failure stops the run only when a later
    row needs it, as in `run --all`); stop at the first campaign whose run does not pass."""
    results, code = [], EXIT_OK
    for path in paths:
        try:
            campaign = load_campaign(path)
            filter_seeds(campaign, seeds)
            if plan_only:
                print("\n=== %s" % path)
                code = cmd_plan(campaign) or code
                continue
            if refuse_implicit(campaign):
                raise CampaignError("its row order is implied, not stated (the IMPLICIT ORDER lines above)")
            rows, registration = allocation_rows(campaign,stage)
            allocation_selftest(campaign,rows,registration)
            print("\n=== BATCH %s: prepare --all" % campaign["name"])
            prepared = cmd_prepare(campaign, rows, True)
            print("\n=== BATCH %s: run --all" % campaign["name"])
            code = cmd_run(campaign, rows, True, None)
        except CampaignError as exc:
            print("CAMPAIGN ERROR in %s: %s" % (path, exc), file=sys.stderr)
            results.append((str(path), "REFUSED (campaign file: %s)" % exc))
            code = EXIT_REFUSED
            break
        word = {EXIT_OK: "PASS", EXIT_FAILED: "FAIL", EXIT_REFUSED: "REFUSED"}.get(code, "FAIL")
        note = "" if prepared == EXIT_OK else ", prepare --all reported a failure"
        results.append((campaign["name"], "%s (run --all exited %d%s)" % (word, code, note)))
        if code != EXIT_OK:
            break
    if plan_only:
        return code
    print("\nBATCH SUMMARY")
    for name, line in results:
        print("%-28s %s" % (name, line))
    for path in paths[len(results):]:
        print("%-28s NOT RUN (an earlier campaign did not pass)" % path)
    return code


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["batch"]:
        parser = argparse.ArgumentParser(prog="runner.py batch",
                                         description="prepare --all then run --all for each campaign, in order.")
        parser.add_argument("campaigns", type=Path, nargs="+")
        parser.add_argument("--plan", action="store_true", help="print every campaign's plan; executes nothing")
        parser.add_argument("--seeds", help="only these seeds, e.g. 0,1,2 or 0-4; rows with no seed always run")
        parser.add_argument('--stage', help='one registered allocation stage; each stage uses a new sbatch')
        args = parser.parse_args(argv[1:])
        try:
            seeds = parse_seeds(args.seeds) if args.seeds else None
        except CampaignError as exc:
            print("CAMPAIGN ERROR: %s" % exc, file=sys.stderr)
            return EXIT_REFUSED
        return cmd_batch(args.campaigns, args.plan, seeds,args.stage)
    parser = argparse.ArgumentParser(description="Run a campaign of experiment rows with a pilot gate.",
                                     epilog="Several campaigns in order, one command: runner.py batch a.yaml b.yaml ... "
                                            "(prepare --all then run --all for each; --plan prints and executes nothing).")
    parser.add_argument("action", choices=("plan", "prepare", "run", "status"))
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--row")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--attempt", type=int)
    parser.add_argument('--stage', help='one registered allocation stage; each stage uses a new sbatch')
    parser.add_argument("--seeds", help="plan/prepare/run only these seeds, e.g. 0,1,2 or 0-4; rows with no seed are kept")
    args = parser.parse_args(argv)
    try:
        campaign = load_campaign(args.campaign)
        if args.action != "status":
            filter_seeds(campaign, parse_seeds(args.seeds) if args.seeds else None)
        if args.action == "plan":
            return cmd_plan(campaign)
        if args.action == "status":
            return cmd_status(campaign)
        if bool(args.row) == bool(args.all):
            raise CampaignError("%s needs exactly one of --row ID or --all" % args.action)
        available, registration = allocation_rows(campaign,args.stage)
        rows = available if args.all else [r for r in available if r["id"] == args.row]
        if not rows:
            raise CampaignError("no row named %r" % args.row)
        if refuse_implicit(campaign):
            raise CampaignError("its row order is implied, not stated: add each row named above to the row's needs")
        allocation_selftest(campaign,available,registration)
        if args.action == "prepare":
            return cmd_prepare(campaign, rows, args.all)
        return cmd_run(campaign, rows, args.all, args.attempt)
    except CampaignError as exc:
        print("CAMPAIGN ERROR: %s" % exc, file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
