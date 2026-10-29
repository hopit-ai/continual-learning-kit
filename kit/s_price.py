#!/usr/bin/env python3
"""Arithmetic check of the approved approximate Screen S envelope; no launch authority."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *


def estimate():
    probes = PROBES
    low = probes*.75+3+5+5
    high = probes*1.4+5+5+5
    require(probes == 27 and 33 <= low <= 34 and 52 <= high <= 53, 'amendment pricing changed')
    return dict(probes=probes, incoming_checkpoints=INCOMING_CHECKPOINTS, probe_gpu_hours=[20.25, 37.8],
        drift_gpu_hours=[3, 5], generation_gpu_hours=5, other_scoring_startup_gpu_hours=5,
        total_gpu_hours=[round(low, 2), round(high, 2)], approved_approximate_gpu_hours=[30, 50],
        note='Rounded planning envelope only. Technical dry runs cannot price real generation. Reprice from real staged timing before send. The 15 v4 step-20 exports/inventory are additional, separately repriced. C replication excluded.')


def job(a):
    context(a.out, a.standin)
    atomic(a.out/'pricing.json', estimate())
    print(json.dumps(estimate(), sort_keys=True))

if __name__ == '__main__':
    raise SystemExit(entry(job, lambda p: None))
