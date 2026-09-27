#!/usr/bin/env python3
"""Pack everything a partner sends back into one archive, and nothing large.

    python collect.py --work $WORK --out k3-return.tar.gz                                  # the whole $WORK
    python collect.py --work $WORK --out k3-return.tar.gz --campaign campaigns/k3-replay.yaml   # that campaign's trees

What goes in (text only):
  - every *.json, *.md, *.jsonl, *.txt inside a folder named report*, eval, forgetting, deltas, stuck,
    plasticity or scorecard;
  - every metrics.jsonl inside runs/<run>/, and every train-summary.json, run-summary.json, fold.json, merge.log;
  - the runner's own records under campaign/<name>/<row>/: prepare-*.json, attempt-*/start.json and
    attempt-*/verdict.json, and the output.log of any attempt that did not PASS (the log we ask for when a row fails).
What never goes in: model weights (*.safetensors, *.bin, *.pt, *.pth, *.ckpt, *.gguf), checkpoint folders
(checkpoint*, global_step*, hf-step*), *.parquet over 5 MB, and any other file over 50 MB. Symbolic links are not
followed; they are listed in the manifest with their targets.

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
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-collect.v1"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
MB = 1024 * 1024
REPORT_DIRS = {"eval", "forgetting", "deltas", "stuck", "plasticity", "scorecard"}
TEXT = {".json", ".md", ".jsonl", ".txt"}
NAMED = {"train-summary.json", "run-summary.json", "fold.json", "merge.log"}
WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}
PARQUET_LIMIT, FILE_LIMIT = 5 * MB, 50 * MB


def pruned(name: str) -> bool:
    """A folder never descended into: checkpoints and merged model copies."""
    return name.startswith(("checkpoint", "global_step", "hf-step"))


def wanted(rel: Path, root_rel: Path) -> bool:
    """Whether a file (path relative to WORK) is something a partner sends back."""
    parts = rel.parts
    if rel.name in NAMED:
        return True
    if rel.name == "metrics.jsonl" and len(parts) >= 3 and parts[-3] == "runs":
        return True
    if rel.suffix in TEXT and any(p in REPORT_DIRS or p.startswith("report") for p in parts[:-1]):
        return True
    if len(parts) >= 4 and parts[0] == "campaign":                      # campaign/<name>/<row>/...
        if len(parts) == 4 and re.fullmatch(r"prepare-\d+\.json", rel.name):
            return True
        if len(parts) == 5 and re.fullmatch(r"attempt-\d+", parts[3]):
            if rel.name in ("start.json", "verdict.json"):
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
    if path.suffix == ".parquet" and size > PARQUET_LIMIT:
        return "parquet over 5 MB"
    if size > FILE_LIMIT:
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
                if (base / d).is_symlink():
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
    if not files:
        print("NOTHING TO COLLECT under %s: no reports, scorings, run summaries or runner records found" % work,
              file=sys.stderr)
        return EXIT_FAILED
    top = out.name[:-len(".tar.gz")] if out.name.endswith(".tar.gz") else out.name[:-len(".tgz")]
    entries = [{"path": str(rel), "bytes": size, "sha256": sha256(path)} for path, rel, size in files]
    manifest = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "work": str(work), "campaigns": campaigns, "roots": [str(r) for r in roots],
                "files": entries, "file_count": len(entries), "total_bytes": sum(e["bytes"] for e in entries),
                "excluded": excluded, "symlinks": links}
    try:
        with out.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
            for path, rel, _ in files:
                archive.add(str(path), arcname="%s/%s" % (top, rel), recursive=False)
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
    print("ARCHIVE %s (%.1f MB)" % (out, out.stat().st_size / MB))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
