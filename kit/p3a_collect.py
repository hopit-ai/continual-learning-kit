#!/usr/bin/env python3
"""CPU job: tar OUT evidence, excluding weights, caches and private test payloads."""
import io
import tarfile
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3a_common import *


def job(a):
    """Return evidence without model weights, download caches or build scratch."""
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file() and not p.is_symlink()
        and p.resolve() != a.archive.resolve() and p.name != 'collect.manifest.json'
        and p.relative_to(a.out).parts[0] not in ('sources', 'inputs-build')
        and not {'weights', 'models', 'cache', 'caches', '.cache', '__pycache__'} & set(p.relative_to(a.out).parts)
        and p.suffix not in ('.bin', '.safetensors', '.pt', '.pth', '.parquet', '.pyc') and p.name != 'tests.jsonl']
    manifest = dict(schema='kit-collect.v1', files=[dict(path=str(p.relative_to(a.out)), bytes=p.stat().st_size, sha256=s.file_hash(p)) for p in files], truncated=[], symlinks=[])
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    # Store the exact bytes/hash read, including logs that may still be flushing.
    with tarfile.open(a.archive, 'w:gz') as tar:
        for p, record in zip(files, manifest['files']):
            raw = p.read_bytes(); record.update(bytes=len(raw), sha256=t.sha(raw))
            member = tarfile.TarInfo('return/' + record['path']); member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
        raw = (json.dumps(manifest, indent=2) + '\n').encode()
        member = tarfile.TarInfo('return/collect.manifest.json'); member.size = len(raw)
        tar.addfile(member, io.BytesIO(raw))
    print(a.archive)

if __name__ == '__main__':
    raise SystemExit(entry(job, lambda p: p.add_argument('--archive', type=Path, required=True), 'collect'))
