#!/usr/bin/env python3
"""One full mixed or ordinary stage, using the v4 D/F entry points directly."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_train import job, configure
from kit.p3b_common import entry
if __name__ == '__main__': raise SystemExit(entry(job, configure, 'p3b-{route}-{recipe}-{seed}'))
