#!/usr/bin/env python3
"""Under-28-MiB return with hash-verified durable bulk evidence, never weights."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
from kit.next_collect import job as collect, configure


def job(a):
    binding(a.out,a.standin)
    from kit.a4_read import load
    load(a.out)
    require((a.out/'reading/summary.json').exists(),'run registered reader before return')
    require(not any(p.is_symlink() for p in a.out.rglob('*')),'4A evidence must be materialised before durable collection')
    # Handoff and idle-allocation costs can arrive after the earlier reader.
    # Refresh without an active tee log, then pack the exact same journal set.
    from kit import a4_read
    from types import SimpleNamespace as NS
    a4_read.job(NS(out=a.out,standin=a.standin,bootstrap=2000,source=None))
    collect(a)


def main():
    # Collector writes only its external archive; no active tee log inside the frozen source tree.
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--standin',action='store_true');configure(p);a=p.parse_args()
    try:job(a)
    except CorrectnessStop as exc:print('STOP: correctness:',exc);return 2
    except NotReady as exc:print('NOT READY:',exc);return 3
    return 0

if __name__=='__main__':raise SystemExit(main())
