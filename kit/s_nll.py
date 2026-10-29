#!/usr/bin/env python3
"""One isolated GPU NLL measurement using a4_nll's unchanged reductions."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *


def main():
    import argparse
    from kit.a4_nll import measure
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    registration()
    if os.environ.get('M2_ASSIGNMENT'):
        from kit.m2_assignment import active
        active(Path(os.environ['M2_ASSIGNMENT']).parent,os.environ.get('CUDA_VISIBLE_DEVICES','0'),False)
    ref = read(a.reference)
    model(ref, False)
    targets = rows(a.schedule)
    require(all(r.get('verified') is True for r in targets), 'NLL requires incoming-verified targets')
    from kit.m2_paths import local
    atomic(a.output, measure(str(local(ref['model'])), targets, False))

if __name__ == '__main__':
    main()
