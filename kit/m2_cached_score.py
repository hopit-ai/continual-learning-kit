#!/usr/bin/env python3
"""Core entry for inherited panel scorers, using CPU-staged model hashes."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_model_cache import install
install()


def main():
    name=sys.argv.pop(1)
    if name=='next_hellaswag':
        from kit.next_hellaswag import entry,job,configure
        return entry(job,configure,'next-hellaswag-{action}')
    if name=='p3b_score':
        from kit.p3b_score import entry,job,configure
        return entry(job,configure,'p3b-score-{arm}-{seed}-{step}-{panel_set}')
    raise ValueError('unregistered core scorer')


if __name__=='__main__':raise SystemExit(main())
