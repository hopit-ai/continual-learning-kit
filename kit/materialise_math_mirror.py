#!/usr/bin/env python3
"""Materialise the MATH tree that kit/second_task_screen.py freezes from the EleutherAI mirror.

Owner, 12 Oct 2026: use the EleutherAI/hendrycks_math mirror ("A. since it is a research norm"); the authors'
Berkeley tarball and Hugging Face hendrycks/competition_math are no longer distributed. The mirror keeps the
original fields (problem, level, type, solution) and split sizes (train 7,500 / test 5,000) but not the original
file numbers, so an item's ID is <split>/<subject>/<row index within that subject's parquet> at the pinned revision.

    python3 scripts/materialise_math_mirror.py --mirror DIR --out DIR/MATH
DIR holds <subject>/<split>-00000-of-00001.parquet downloaded at revision REVISION (see DIR/REVISION, DIR/SHA256SUMS).
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path

REVISION = '21a5633873b6a120296cce3e2df9d5550074f4a3'


def main():
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--mirror', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); mirror = Path(args.mirror); out = Path(args.out)
    if (mirror / 'REVISION').read_text().strip() != REVISION: raise SystemExit('mirror revision differs from the pinned one')
    if out.exists(): raise SystemExit('output exists; materialise into a fresh directory')
    files = sorted(glob.glob(str(mirror / '*/*.parquet'))); n = 0
    for f in files:
        subject = Path(f).parent.name; split = Path(f).name.split('-')[0]
        for i, row in enumerate(pq.read_table(f).to_pylist()):
            target = out / split / subject / f'{i}.json'; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({k: row[k] for k in ('problem', 'level', 'type', 'solution')}, ensure_ascii=False)); n += 1
    print(json.dumps({'items': n, 'revision': REVISION,
                      'parquet_sha256': {str(Path(f).relative_to(mirror)): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}, indent=2))


if __name__ == '__main__':
    main()
