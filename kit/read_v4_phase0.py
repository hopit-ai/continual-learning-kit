#!/usr/bin/env python3
"""Read a hash-checked collector archive; recompute technical PHASE-0 only."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from kit.v4_phase0 import read_archive


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args(argv)
    import os
    from kit.v4_phase0_site import storage
    work=Path(os.environ.get('WORK',str(a.archive.parent/'phase0-reader-work')))
    with storage(work):result=read_archive(a.archive)
    text=json.dumps(result,indent=2)+'\n'
    if a.out:
        with a.out.open('x') as handle:handle.write(text)
    print(text,end='');return 0 if result['status']=='technical pass' else 1

if __name__=='__main__':raise SystemExit(main())
