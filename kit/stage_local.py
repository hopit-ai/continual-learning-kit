#!/usr/bin/env python3
"""Put training inputs on the GPU machine's LOCAL disk before the GPU does any work.

Reading many small files from a mounted network volume during training starves the GPU. In another
of the owner's projects (image training on Modal, 64-step smoke, identical otherwise) moving 20,335
images from the mounted volume to local disk took training-step time from 554 s to 41 s (13.5x) and
the whole training function from 574 s to 57 s (10x); the one-time CPU archive build took 197 s and
the GPU-side copy, extract and verify took 8 s. Those are 64-step measurements from that project, not
ours, and the gain comes from avoiding repeated remote small-file reads: it will not transfer to a
workload that already reads a few large files from fast local storage.

Three commands, all standard library:

    pack    CPU, once.  Many small files -> ONE deterministic tar + a manifest of per-file hashes.
            Reads with 16 concurrent readers (remote volumes reward concurrency), writes in sorted
            order with fixed metadata so the same inputs always give the same bytes.
    unpack  GPU start.  Copy the archive to local disk, extract, verify every hash, leave a marker.
            A later run finds the marker, checks the archive hash, and skips: the archive is reused.
    mirror  GPU start.  For a FEW LARGE files (model weights): parallel copy to local disk with
            hash verification, same marker-and-skip behaviour. No tar: nothing to gain from one.

List EVERY file the loader touches, not just the rows that survive filtering. The project above
needed 20,335 image ids for 8,845 selected rows because its loader checked each candidate before
capping; an archive built from the final rows would have sent the loader back to the volume.

Publishing is write-to-temporary then rename, never hardlink: Modal volumes do not support hardlinks.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = "kit-stage-local.v1"
MARKER = ".staged.json"
CHUNK = 8 * 1024 * 1024
SMALL = 64 * 1024 * 1024            # files above this are streamed, never held whole in memory


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(temporary: Path, final: Path) -> None:
    """Atomic on one filesystem, and needs no hardlink."""
    os.replace(temporary, final)


def read_list(root: Path, list_file: Path | None, pattern: str | None) -> list:
    if (list_file is None) == (pattern is None):
        raise SystemExit("give exactly one of --list FILE or --glob PATTERN")
    names = ([line.strip() for line in list_file.read_text().splitlines() if line.strip()] if list_file
             else [str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file()])
    names = sorted(set(names))
    if not names:
        raise SystemExit("no files to pack")
    missing = [n for n in names if not (root / n).is_file()]
    if missing:
        raise SystemExit("%d listed files do not exist under %s, e.g. %s" % (len(missing), root, missing[:3]))
    if any(n.startswith(("/", "..")) or "/../" in n for n in names):
        raise SystemExit("file names must be relative paths inside --root")
    return names


def cmd_pack(args) -> int:
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    names = read_list(root, Path(args.list) if args.list else None, args.glob)
    clock = time.monotonic()
    manifest = {}

    def load(name):
        path = root / name
        if path.stat().st_size > SMALL:
            return name, None
        return name, path.read_bytes()

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".partial")
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive, ThreadPoolExecutor(max_workers=args.readers) as pool:
        for start in range(0, len(names), args.readers * 8):          # bounded memory: a few batches of small files at a time
            for name, data in pool.map(load, names[start:start + args.readers * 8]):
                info = tarfile.TarInfo(name)
                info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
                if data is None:
                    path = root / name
                    info.size = path.stat().st_size
                    manifest[name] = {"sha256": sha256_file(path), "bytes": info.size}
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
                else:
                    info.size = len(data)
                    manifest[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                    archive.addfile(info, io.BytesIO(data))
    record = {"schema": SCHEMA, "files": manifest, "file_count": len(manifest), "bytes": sum(f["bytes"] for f in manifest.values()),
              "archive_sha256": sha256_file(temporary), "readers": args.readers, "pack_seconds": round(time.monotonic() - clock, 2)}
    manifest_temporary = out.with_name(out.name + ".manifest.json.partial")
    manifest_temporary.write_text(json.dumps(record, indent=1, sort_keys=True))
    publish(temporary, out)
    publish(manifest_temporary, out.with_name(out.name + ".manifest.json"))
    print("packed %d files, %.1f MB, in %.2f s -> %s (%s)" % (record["file_count"], record["bytes"] / 1e6, record["pack_seconds"], out, record["archive_sha256"][:16]))
    return 0


def _already_staged(target: Path, identity: str) -> bool:
    marker = target / MARKER
    if not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text()).get("identity") == identity
    except (json.JSONDecodeError, OSError):
        return False


def _verify(target: Path, files: dict, workers: int) -> list:
    def check(item):
        name, expected = item
        path = target / name
        if not path.is_file():
            return "%s: missing" % name
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            return "%s: bytes differ from the manifest" % name
        return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [problem for problem in pool.map(check, files.items()) if problem]


def cmd_unpack(args) -> int:
    archive, target = Path(args.archive).resolve(), Path(args.to).resolve()
    record = json.loads(archive.with_name(archive.name + ".manifest.json").read_text())
    if _already_staged(target, record["archive_sha256"]):
        print("already staged at %s (%d files); reusing" % (target, record["file_count"]))
        return 0
    if target.exists() and any(target.iterdir()):
        raise SystemExit("refusing to unpack into a non-empty directory that was not staged by this archive: %s" % target)
    clock = time.monotonic()
    target.mkdir(parents=True, exist_ok=True)
    local_archive = target / (archive.name + ".local")
    shutil.copyfile(archive, local_archive)
    copied = time.monotonic()
    if sha256_file(local_archive) != record["archive_sha256"]:
        raise SystemExit("the copied archive does not match its manifest: %s" % archive)
    with tarfile.open(local_archive) as handle:
        handle.extractall(target, filter="data")
    local_archive.unlink()
    extracted = time.monotonic()
    problems = _verify(target, record["files"], args.workers)
    if problems:
        raise SystemExit("%d files failed verification, e.g. %s" % (len(problems), problems[:3]))
    timings = {"copy_seconds": round(copied - clock, 2), "extract_seconds": round(extracted - copied, 2),
               "verify_seconds": round(time.monotonic() - extracted, 2), "total_seconds": round(time.monotonic() - clock, 2)}
    (target / MARKER).write_text(json.dumps({"schema": SCHEMA, "identity": record["archive_sha256"], "file_count": record["file_count"],
                                             "bytes": record["bytes"], **timings}, indent=1, sort_keys=True))
    print("staged %d files to %s in %.2f s (copy %.2f, extract %.2f, verify %.2f)" % (record["file_count"], target, timings["total_seconds"],
          timings["copy_seconds"], timings["extract_seconds"], timings["verify_seconds"]))
    return 0


def cmd_mirror(args) -> int:
    source, target = Path(args.source).resolve(), Path(args.to).resolve()
    names = sorted(str(p.relative_to(source)) for p in source.rglob("*") if p.is_file() and p.name != MARKER)
    if not names:
        raise SystemExit("nothing to mirror under %s" % source)
    sizes = {n: (source / n).stat().st_size for n in names}
    identity = hashlib.sha256(json.dumps(sizes, sort_keys=True).encode()).hexdigest()      # cheap: names and sizes; bytes are hashed once copied
    if _already_staged(target, identity):
        print("already mirrored at %s (%d files); reusing" % (target, len(names)))
        return 0
    if target.exists() and any(target.iterdir()):
        raise SystemExit("refusing to mirror into a non-empty directory that was not staged from this source: %s" % target)
    clock = time.monotonic()

    def copy(name):
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source / name, temporary)
        publish(temporary, destination)
        return name, {"sha256": sha256_file(destination), "bytes": destination.stat().st_size}
    target.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = dict(pool.map(copy, names))
    wrong = [n for n in names if files[n]["bytes"] != sizes[n]]
    if wrong:
        raise SystemExit("size changed while copying: %s" % wrong[:3])
    seconds = round(time.monotonic() - clock, 2)
    (target / MARKER).write_text(json.dumps({"schema": SCHEMA, "identity": identity, "file_count": len(names), "bytes": sum(sizes.values()),
                                             "files": files, "total_seconds": seconds, "source": str(source)}, indent=1, sort_keys=True))
    print("mirrored %d files, %.1f MB, to %s in %.2f s" % (len(names), sum(sizes.values()) / 1e6, target, seconds))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage training inputs on local disk before the GPU does any work.")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("pack"); p.add_argument("--root", required=True); p.add_argument("--list"); p.add_argument("--glob")
    p.add_argument("--out", required=True); p.add_argument("--readers", type=int, default=16)
    u = sub.add_parser("unpack"); u.add_argument("--archive", required=True); u.add_argument("--to", required=True); u.add_argument("--workers", type=int, default=16)
    m = sub.add_parser("mirror"); m.add_argument("--source", required=True); m.add_argument("--to", required=True); m.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    return {"pack": cmd_pack, "unpack": cmd_unpack, "mirror": cmd_mirror}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
