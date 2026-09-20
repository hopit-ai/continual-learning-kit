#!/usr/bin/env python3
"""Run a campaign of experiment rows on any machine, with a pilot gate.

    python runner.py plan    campaigns/toy.yaml                # print what would run; executes nothing
    python runner.py prepare campaigns/toy.yaml --all          # every row's IO, on a CPU machine, before any GPU is held
    python runner.py run     campaigns/toy.yaml --row pilot    # run one row
    python runner.py run    campaigns/toy.yaml --all           # run every row in order; stops at a refusal
    python runner.py status campaigns/toy.yaml                 # verdict of every row

The rule this file enforces: **no row runs big unless the same experiment ran small first and
passed a bar written in advance.** A row marked `pilot: true` is judged against its `bars` when it
finishes. Every later row needs every earlier pilot's latest verdict to be PASS, plus any rows it names
in `needs`, and is REFUSED otherwise: nothing is launched, exit code 2.

**All IO happens before a GPU is held.** A row may list `prepare` commands (downloads, format
conversions, tokenising, anything that needs no GPU) and `requires` paths (the model directory, the
data files). `prepare` runs them on any CPU machine, for all rows at once and without waiting for a
pilot, since fetching data early wastes nothing. `run` REFUSES a row whose preparation has not
passed or whose required paths are missing, before its command starts, so an allocated GPU never
waits on a download. Phase 1 paid for exactly that: 700 seconds of cold dataset download inside a
GPU container, and GPUs idle for a quarter to 85 percent of a step while scoring ran serially.

What is recorded, and when. `start.json` is written and fsynced BEFORE the command is launched, so a
crash still leaves the identity of what ran. `verdict.json` is written after. Nothing is ever
overwritten: a second try of a row is a new attempt (`--attempt 2`) in its own directory, and the
gate reads the latest attempt.

A missing file, a missing key or a crashed command is a FAIL with its reason, never a silent PASS.
Standard library plus PyYAML (JSON campaigns need only the standard library). No network, no GPU
code, no Modal: the rows' own commands decide what hardware they need.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
        # every earlier pilot is ALWAYS required; a row's own `needs` adds to that, it never replaces it
        row["needs"] = list(dict.fromkeys(list(seen_pilots) + list(row.get("needs") or [])))
        unknown = [n for n in row["needs"] if n not in ids[:ids.index(row["id"])]]
        if unknown:
            raise CampaignError("row %s needs rows that do not come before it: %s" % (row["id"], unknown))
        if row.get("pilot"):
            seen_pilots.append(row["id"])
    campaign["_path"] = str(path.resolve())
    campaign["_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return campaign


def work_root(campaign: dict) -> Path:
    name = campaign.get("workdir_env", "WORK")
    value = os.environ.get(name)
    if not value:
        raise CampaignError("set %s to the directory that holds outputs" % name)
    return Path(value).resolve()


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
        done = subprocess.run(command, env={**os.environ, **spec["env"]}, cwd=spec["cwd"])
        steps.append({"command": command, "returncode": done.returncode, "seconds": round(time.monotonic() - started, 1)})
        if done.returncode != 0:
            reason = "prepare command exited %d: %s" % (done.returncode, " ".join(command))
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
    if path.exists():
        raise FileExistsError("refusing to overwrite %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(json.dumps(payload, indent=1, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())


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
        if verdict is None:
            reasons.append("needs %s, which has not run" % needed)
        elif verdict["verdict"] != "PASS":
            reasons.append("needs %s, whose latest verdict is %s (%s)" % (needed, verdict["verdict"], verdict.get("reason") or "see its verdict.json"))
    return reasons


def run_row(campaign: dict, row: dict, attempt: int | None) -> int:
    reasons = gate(campaign, row) + not_ready(campaign, row)
    if reasons:
        print("REFUSED %s: %s" % (row["id"], "; ".join(reasons)))
        return EXIT_REFUSED
    done = attempts(campaign, row["id"])
    number = attempt if attempt is not None else (done[-1] + 1 if done else 1)
    if number in done:
        print("REFUSED %s: attempt %d already exists; pass --attempt %d" % (row["id"], number, done[-1] + 1))
        return EXIT_REFUSED
    spec = resolve(row, campaign, number)
    out = row_dir(campaign, row["id"]) / ("attempt-%d" % number)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_durably(out / "start.json", {"schema": SCHEMA, "campaign": campaign["name"], "campaign_sha256": campaign["_sha256"],
                                       "row": row["id"], "attempt": number, "pilot": bool(row.get("pilot")), "needs": row["needs"],
                                       "started_at": started, **{k: spec[k] for k in ("command", "env", "cwd")}})
    print("RUN %s attempt %d: %s" % (row["id"], number, " ".join(spec["command"])))
    clock = time.monotonic()
    with (out / "output.log").open("w") as log:
        process = subprocess.Popen(spec["command"], env={**os.environ, **spec["env"]}, cwd=spec["cwd"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = process.wait()
    bars = [judge_bar(bar) for bar in spec["bars"]]
    failed = [b for b in bars if not b["ok"]]
    if returncode != 0:
        verdict, reason = "FAIL", "command exited %d" % returncode
    elif failed:
        verdict, reason = "FAIL", "; ".join("%s: %s" % (b["name"], b["reason"]) for b in failed)
    else:
        verdict, reason = "PASS", None
    write_durably(out / "verdict.json", {"schema": SCHEMA, "row": row["id"], "attempt": number, "verdict": verdict, "reason": reason,
                                         "returncode": returncode, "bars": bars, "started_at": started,
                                         "seconds": round(time.monotonic() - clock, 1)})
    print("%s %s%s" % (verdict, row["id"], ": " + reason if reason else ""))
    return EXIT_OK if verdict == "PASS" else EXIT_FAILED


def cmd_plan(campaign: dict) -> int:
    print("campaign %s (%s rows), outputs under %s" % (campaign["name"], len(campaign["rows"]), work_root(campaign)))
    for row in campaign["rows"]:
        spec = resolve(row, campaign, 1)
        print("\n[%s]%s needs: %s" % (row["id"], " PILOT" if row.get("pilot") else "", ", ".join(row["needs"]) or "nothing"))
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
    return EXIT_OK


def cmd_status(campaign: dict) -> int:
    for row in campaign["rows"]:
        verdict = latest_verdict(campaign, row["id"])
        blocked = gate(campaign, row)
        state = "%s (attempt %s)" % (verdict["verdict"], verdict["attempt"]) if verdict else ("BLOCKED" if blocked else "READY")
        print("%-24s %-22s %s" % (row["id"], state, (verdict or {}).get("reason") or "; ".join(blocked) or ""))
    return EXIT_OK


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a campaign of experiment rows with a pilot gate.")
    parser.add_argument("action", choices=("plan", "prepare", "run", "status"))
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--row")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--attempt", type=int)
    args = parser.parse_args(argv)
    try:
        campaign = load_campaign(args.campaign)
        if args.action == "plan":
            return cmd_plan(campaign)
        if args.action == "status":
            return cmd_status(campaign)
        if bool(args.row) == bool(args.all):
            raise CampaignError("%s needs exactly one of --row ID or --all" % args.action)
        rows = campaign["rows"] if args.all else [r for r in campaign["rows"] if r["id"] == args.row]
        if not rows:
            raise CampaignError("no row named %r" % args.row)
        if args.action == "prepare":
            codes = [prepare_row(campaign, row) for row in rows]      # every row, even after a failure: IO is independent
            return EXIT_OK if all(code == EXIT_OK for code in codes) else EXIT_FAILED
        for row in rows:
            if args.all and (latest_verdict(campaign, row["id"]) or {}).get("verdict") == "PASS":
                print("SKIP %s: already PASS" % row["id"])
                continue
            code = run_row(campaign, row, args.attempt)
            if code != EXIT_OK:
                return code
        return EXIT_OK
    except CampaignError as exc:
        print("CAMPAIGN ERROR: %s" % exc, file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
