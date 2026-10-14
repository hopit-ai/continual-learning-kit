#!/usr/bin/env python3
"""One incoming checkpoint/size: four full-weight F updates, batch 16, no warmup."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit import p3b_train as train
from kit.p3b_common import entry
if __name__ == '__main__': raise SystemExit(entry(lambda a: train.job(a, True), lambda p: train.configure(p, True), 'p3b-recover-{arm}-{size}-{seed}'))
