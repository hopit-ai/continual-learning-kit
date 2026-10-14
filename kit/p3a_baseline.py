#!/usr/bin/env python3
"""Plain 3C baseline job. See README-p3a.md for ordering and environment."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3a_jobs import *
if __name__ == '__main__':
    raise SystemExit(entry(lambda a: baseline(a), lambda p: configure(p, 'baseline'), 'baseline'))
