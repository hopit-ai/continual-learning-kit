#!/usr/bin/env python3
"""Pack everything a partner sends back into one archive, and nothing large.

    python collect.py --work $WORK --out k3-return.tar.gz                                  # the whole $WORK
    python collect.py --work $WORK --out k3-return.tar.gz --campaign campaigns/k3-replay.yaml   # that campaign's trees

What goes in (text only):
  - every *.json, *.md, *.jsonl, *.txt inside a folder named report*, eval, forgetting, deltas, stuck,
    plasticity, scorecard, selection or budget (package 4's selection, reservation and GPU-hour ledger, which its
    report and its archive check read: kit/p4_select.py, kit/p4_budget.py);
  - every metrics.jsonl inside runs/<run>/, and every train-summary.json, run-summary.json, fold.json, merge.log;
  - the runner's own records under campaign/<name>/<row>/: prepare-*.json, attempt-*/start.json and
    attempt-*/verdict.json, and the output.log of any attempt that did not PASS (the log we ask for when a row fails).
What never goes in: model weights (*.safetensors, *.bin, *.pt, *.pth, *.ckpt, *.gguf), checkpoint folders
(checkpoint*, global_step*, hf-step*), *.parquet over 5 MB, and any other file over 50 MB, except the exact registered, hash-verified FinQA raw training source required for archive-only reproduction. A wanted text
file over 50 MB is not dropped silently: its last 5 MB are packed as `<name>.tail`, the manifest lists it under
`truncated`, and the last lines printed name it (INCOMPLETE: ...). Symbolic links are not
followed; they are listed in the manifest with their targets.

Package 4 (send-5 review round 2, finding 9; amendment 3 B8) adds: `rollout-rows.jsonl.gz` inside a report folder (the
reduced per-rollout records of kit/rollout_stats.py: binary, small, the one compressed name allowed); `.yaml` inside
the report folders and `recipe-check` (the resolved configurations kit/p4_recipe.py archives); each run's
`env/resolved-config.yaml`, `env/resolved-config.err`, `env/trainer-started-at.txt` (round-5 ruling H3: whether the
launcher invoked the trainer) and `env/stage.json` (round-6 ruling J3: the launcher's own record of every stage it
reached and of its exit, the only evidence that a failed attempt rolled nothing out); and `kit-snapshot/`, the exact analysis source this
collector runs beside (KIT_SNAPSHOT, plus every campaign file under kit/campaigns/ whose name has runner records under
WORK/campaign/, as campaigns/<name>.yaml), each file listed in the manifest with its sha256, and for each campaign
whether every start.json under it recorded that file's sha256. A report recomputed from the extracted archive runs
from kit-snapshot/ and reads nothing outside the archive.

The archive holds one folder (named after the archive) with `collect.manifest.json` inside it: every file's path,
size and sha256, what was left out and why, and the campaign files named. An existing archive is never overwritten.
Standard library only (PyYAML is used only to read a --campaign YAML's name).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

SCHEMA = "kit-collect.v1"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
MB = 1024 * 1024
REPORT_DIRS = {"eval", "forgetting", "deltas", "stuck", "plasticity", "scorecard", "selection", "budget", "recipe-check", "containment"}
TEXT = {".json", ".md", ".jsonl", ".txt", ".yaml", ".log"}
NAMED = {"train-summary.json", "run-summary.json", "fold.json", "merge.log"}
ROLLOUT_ROWS = "rollout-rows.jsonl.gz"          # kit/rollout_stats.py's reduced records: the one binary name packed
RUN_IDENTITY = {"argv.txt", "sdpo-commit.txt", "sdpo-dirty.txt", "started-at.txt", "finished-at.txt", "data-sha256.txt", "model-files.txt",
                "resolved-config.yaml", "resolved-config.err",
                "trainer-started-at.txt",       # round-5 ruling H3: the launcher invoked the trainer (kit/rollout_stats.py)
                "stage.json", "optimizer-updates.json", "v4-settings.json", "settings.json", "data-manifest.json", "data-labels.json",
                "gpu-memory.jsonl", "training-tokens.jsonl", "export-timings.jsonl", "merge-timing.json", "incoming-checkpoint.json",
                "export-identity-step2.json", "export-identity-step20.json", "export-identity-step40.json",
                "recovered-merge-step2.json", "recovered-merge-step20.json", "recovered-merge-step40.json"}                   # round-6 ruling J3: the launcher's own progress, with its EXIT-trap entry
HERE = Path(__file__).resolve().parent
#: the analysis source packed as kit-snapshot/ (amendment 3 B8: "the exact analysis source")
KIT_SNAPSHOT = ("p4_contain.py", "p4_budget.py", "p4_frozen.py", "p4_intervals.py", "p4_recipe.py", "p4_report.py", "p4_run.py", "p4_select.py", "p4_watchdog.py",
                "pilot_report.py",
                "budget_report.py", "cap_sweep.py", "canonical.py", "eval_bed.py", "rollout_stats.py", "collect.py", "runner.py",
                "tokens_io.py")
WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}
PARQUET_LIMIT, FILE_LIMIT = 5 * MB, 50 * MB
TAIL_BYTES = 5 * MB                # of a wanted file over FILE_LIMIT, this much of its END is packed as `<name>.tail`


def pruned(name: str) -> bool:
    """A folder never descended into: caches, checkpoints and merged model copies."""
    if name == "phase0-cache": return True
    return name.startswith(("checkpoint", "global_step", "hf-step"))


def wanted(rel: Path, root_rel: Path) -> bool:
    """Whether a file (path relative to WORK) is something a partner sends back."""
    parts = rel.parts
    if parts==('reconcile','containment-reconcile.json'):return True
    if parts in (('route-probe.sbatch',),('v4','report-phase0','route-probe.err')):return True
    if parts in (('cpu-format-capture.txt',),('cpu-format-capture.err',),('cpu-format-reading.json',)):return True
    if parts[:4]==('v4','report-phase0','presend','paste') and rel.name in {'version.txt','config.txt','partition.txt','qos.txt','cgroup.conf'}:
        return True
    if parts[:3]==('v4','report-phase0','download') and rel.name in ('download.sbatch','job.err'):return True
    if parts[:3]==('v4','report-phase0','allocation') and rel.name.startswith('phase0-') and rel.suffix in ('.out','.err'):return True
    if parts[:3]==('v4','report-phase0','allocation') and rel.name in ('phase0.sh','phase0.header.sh','phase0.sbatch'):
        return True
    if parts[:2]==('v4','report-inputs') and rel.name in {'v4-data-audit-'+name+'.ids' for name in ('TOKEN80','CLEAN','REACTION','NONREACTION')}:
        return True
    if parts[:2]==('v4','retries') and rel.suffix=='.json':
        return True
    if rel.name.startswith('saved-state-step') and rel.suffix=='.json' and 'env' in parts and 'runs' in parts:
        return True
    if rel.name in NAMED:
        return True
    if rel.name == "metrics.jsonl" and len(parts) >= 3 and parts[-3] == "runs":
        return True
    # What a run WAS: the launcher's `env/` folder (the exact command, the trainer's commit and whether it was clean,
    # start and finish times, the data hashes). A few kilobytes a run, and without the command a returned number
    # cannot be tied to a recipe. `pip-freeze.txt` and `nvidia-smi.txt` are left out: large and the same for every run.
    if len(parts) >= 4 and parts[-4] == "runs" and parts[-2] == "env" and rel.name in RUN_IDENTITY:
        return True
    if (rel.suffix in TEXT or rel.name == ROLLOUT_ROWS) and any(p in REPORT_DIRS or p.startswith("report") for p in parts[:-1]):
        return True
    if rel.suffix=='.parquet' and parts[:1]==('v4',) and any(p in ('report-qualification','report-main','report-phase0','report-inputs') for p in parts):
        return True
    if parts[:2]==('v4','report-inputs') and rel.name in ('mcq.py','preprocess.py'):
        return True
    if len(parts) >= 4 and parts[0] == "campaign":                      # campaign/<name>/<row>/...
        if len(parts) == 4 and re.fullmatch(r"prepare-\d+\.json", rel.name):
            return True
        if len(parts) == 5 and re.fullmatch(r"attempt-\d+", parts[3]):
            if rel.name in ("start.json", "verdict.json", "runner-overhead.json"):
                return True
            if rel.name == "output.log":
                verdict = root_rel / rel.parent / "verdict.json"
                try:
                    return json.loads(verdict.read_text()).get("verdict") != "PASS"
                except (OSError, ValueError):
                    return True                                          # no verdict: the log is the only record
    return False


def too_large(path: Path, size: int) -> str | None:
    if path.suffix in WEIGHTS:
        return "model weights (%s)" % path.suffix
    v4_report=any(part in ('report-qualification','report-main','report-phase0','report-inputs') for part in path.parts)
    if path.suffix == ".parquet" and size > PARQUET_LIMIT and not v4_report:
        return "parquet over 5 MB"
    if size > FILE_LIMIT:
        if path.parts[-5:]==('v4','report-inputs','sources','FinQA','train.json'):
            # The pinned raw FinQA source is 78 MB. It is a required replay
            # input, not a log to tail; admit only those exact public bytes.
            if __package__ in (None,''):
                sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
            from kit.v4_pool import FINQA_HASHES
            if sha256(path)==FINQA_HASHES['train']:return None
        return "over 50 MB"
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(MB), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign_roots(work: Path, campaigns: list) -> tuple:
    """The folders under WORK that the named campaigns write to, and a record of each campaign file."""
    roots, records = set(), []
    for path in campaigns:
        text = path.read_text()
        if path.suffix == ".json":
            name = json.loads(text).get("name")
        else:
            import yaml                                                  # noqa: PLC0415
            name = (yaml.safe_load(text) or {}).get("name")
        if not name:
            raise ValueError("%s has no campaign name" % path)
        roots.add(work / "campaign" / name)
        roots.update(work / top for top in re.findall(r"\{work\}/([^/\"'\s{}]+)", text) if top != "campaign")
        records.append({"path": str(path), "name": name, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    return sorted(roots), records


def gather(work: Path, roots: list) -> tuple:
    files, excluded, links, seen = [], [], [], set()
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            continue
        for folder, dirs, names in os.walk(root):
            base = Path(folder)
            keep = []
            for d in sorted(dirs):
                if base==work and d in ('phase0-envs','phase0-source'):
                    excluded.append({'path':d+'/', 'reason':'CPU environment/source package; hashed receipts are archived'})
                elif (base / d).is_symlink():
                    links.append({"path": str((base / d).relative_to(work)), "target": os.readlink(base / d)})
                elif pruned(d):
                    excluded.append({"path": str((base / d).relative_to(work)) + "/", "reason": "checkpoint or model folder"})
                else:
                    keep.append(d)
            dirs[:] = keep
            for name in sorted(names):
                path = base / name
                rel = path.relative_to(work)
                if rel in seen:
                    continue
                seen.add(rel)
                if path.is_symlink():
                    if wanted(rel, work):
                        links.append({"path": str(rel), "target": os.readlink(path)})
                    continue
                size = path.stat().st_size
                reason = too_large(path, size)
                if reason:
                    if wanted(rel, work) or path.suffix in WEIGHTS or path.suffix == ".parquet":
                        excluded.append({"path": str(rel), "bytes": size, "reason": reason})
                    continue
                if wanted(rel, work):
                    files.append((path, rel, size))
    return files, excluded, links


def kit_snapshot(work: Path) -> tuple:
    """([(source path, archive path relative to the top folder)], record) of the analysis source and the campaign
    files run in WORK, with each campaign's runner records checked against the file's sha256."""
    files, missing, campaigns = [], [], []
    for name in KIT_SNAPSHOT:
        if (HERE / name).is_file():
            files.append((HERE / name, "kit-snapshot/%s" % name))
        else:
            missing.append(name)
    base = work / "campaign"
    for folder in sorted(base.iterdir()) if base.is_dir() else []:
        path = HERE / "campaigns" / ("%s.yaml" % folder.name)
        if not folder.is_dir():
            continue
        if not path.is_file():
            campaigns.append({"name": folder.name, "file": None, "why": "no kit/campaigns/%s.yaml beside this collector" % folder.name})
            continue
        digest = hashlib.sha256(path.read_text().encode()).hexdigest()
        recorded = set()
        for start in folder.glob("*/attempt-*/start.json"):
            try:
                recorded.add(json.loads(start.read_text()).get("campaign_sha256"))
            except (OSError, ValueError):
                recorded.add(None)
        files.append((path, "kit-snapshot/campaigns/%s.yaml" % folder.name))
        campaigns.append({"name": folder.name, "file": "kit-snapshot/campaigns/%s.yaml" % folder.name, "sha256": digest,
                          "runs_recorded_with": sorted(str(r) for r in recorded), "matches_every_run": recorded == {digest}})
    # The v4 readers use qualified kit imports. Preserve their package layout as
    # well as the historical flat snapshot, and include only source (no weights).
    if any(folder.name.startswith('v4-') for folder in base.iterdir()) if base.is_dir() else False:
        for src in sorted(HERE.rglob('*')):
            if src.is_file() and not src.is_symlink() and (src.suffix in ('.py', '.sh') or src.name=='v4_departures.json'):
                files.append((src, 'kit-snapshot/kit/'+str(src.relative_to(HERE))))
    return files, {"files": [arc for _src, arc in files], "missing": missing, "campaigns": campaigns}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pack the results a partner sends back into one small archive.")
    parser.add_argument("--work", type=Path, default=os.environ.get("WORK"), help="the campaigns' WORK folder (default $WORK)")
    parser.add_argument("--out", type=Path, required=True, help="the archive to write, e.g. k3-return.tar.gz")
    parser.add_argument("--campaign", type=Path, action="append", default=[],
                        help="collect only the folders this campaign writes (repeatable); default: all of WORK")
    args = parser.parse_args(argv)
    if not args.work:
        print("REFUSED: pass --work, or set WORK to the folder the campaigns wrote into", file=sys.stderr)
        return EXIT_REFUSED
    work, out = Path(args.work).resolve(), args.out.resolve()
    if not work.is_dir():
        print("REFUSED: %s is not a folder; pass the WORK you ran the campaigns with" % work, file=sys.stderr)
        return EXIT_REFUSED
    if (work/'v4/report-phase0').exists():
        from kit.v4_phase0_site import storage
        with storage(work):return _collect_owned(args,work,out)
    return _collect_owned(args,work,out)


def _collect_owned(args,work,out):
    if out.exists():
        print("REFUSED: %s already exists; nothing is overwritten. Pick a new --out name" % out, file=sys.stderr)
        return EXIT_REFUSED
    if not out.name.endswith((".tar.gz", ".tgz")):
        print("REFUSED: --out must end in .tar.gz, found %s" % out.name, file=sys.stderr)
        return EXIT_REFUSED
    try:
        roots, campaigns = campaign_roots(work, args.campaign) if args.campaign else ([work], [])
    except (OSError, ValueError) as exc:
        print("REFUSED: cannot read a --campaign file: %s" % exc, file=sys.stderr)
        return EXIT_REFUSED
    files, excluded, links = gather(work, roots)
    # A wanted file over the limit (a failed row's log, a very long answers file) is left out whole, but never
    # silently: its last TAIL_BYTES go in as `<path>.tail`, the manifest lists it under `truncated` with the full size
    # and hash, and the summary line below names it.
    tails = [e for e in excluded if e["reason"] == "over 50 MB" and wanted(Path(e["path"]), work)]
    if not files:
        print("NOTHING TO COLLECT under %s: no reports, scorings, run summaries or runner records found" % work,
              file=sys.stderr)
        return EXIT_FAILED
    top = out.name[:-len(".tar.gz")] if out.name.endswith(".tar.gz") else out.name[:-len(".tgz")]
    snapshot, snapshot_record = kit_snapshot(work)
    entries = [{"path": str(rel), "bytes": size, "sha256": sha256(path)} for path, rel, size in files]
    entries += [{"path": arc, "bytes": src.stat().st_size, "sha256": sha256(src)} for src, arc in snapshot]
    manifest = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "work": str(work), "campaigns": campaigns, "roots": [str(r) for r in roots],
                "kit_snapshot": snapshot_record,
                "files": entries, "file_count": len(entries), "total_bytes": sum(e["bytes"] for e in entries),
                "excluded": excluded, "symlinks": links,
                "truncated": [{"path": e["path"], "bytes": e["bytes"], "sha256": sha256(work / e["path"]),
                               "packed_as": e["path"] + ".tail", "packed_bytes": min(TAIL_BYTES, e["bytes"])} for e in tails]}
    try:
        with out.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
            for path, rel, _ in files:
                archive.add(str(path), arcname="%s/%s" % (top, rel), recursive=False)
            for src, arc in snapshot:
                archive.add(str(src), arcname="%s/%s" % (top, arc), recursive=False)
            for entry in manifest["truncated"]:
                with (work / entry["path"]).open("rb") as source:
                    source.seek(max(0, entry["bytes"] - TAIL_BYTES))
                    tail = source.read()
                info = tarfile.TarInfo("%s/%s" % (top, entry["packed_as"]))
                info.size, info.mtime = len(tail), int(datetime.now(timezone.utc).timestamp())
                archive.addfile(info, io.BytesIO(tail))
            blob = json.dumps(manifest, indent=1, sort_keys=True).encode()
            info = tarfile.TarInfo("%s/collect.manifest.json" % top)
            info.size, info.mtime = len(blob), int(datetime.now(timezone.utc).timestamp())
            archive.addfile(info, io.BytesIO(blob))
    except FileExistsError:
        print("REFUSED: %s already exists; nothing is overwritten. Pick a new --out name" % out, file=sys.stderr)
        return EXIT_REFUSED
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    print("%d files, %.1f MB of results, %d left out (see collect.manifest.json)"
          % (len(entries), manifest["total_bytes"] / MB, len(excluded)))
    for entry in manifest["truncated"]:
        print("INCOMPLETE: %s is %.0f MB, over the 50 MB limit; only its last %d MB are in the archive (as %s). "
              "Please send that file separately." % (entry["path"], entry["bytes"] / MB, TAIL_BYTES // MB, entry["packed_as"]))
    print("ARCHIVE %s (%.1f MB)" % (out, out.stat().st_size / MB))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
