#!/usr/bin/env python3
"""Allocation child; shared scientific pipeline also executes stand-ins."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
from kit.a2_common import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--attempt', type=Path, required=True)
    a = p.parse_args()
    require(os.environ.get('UTIL_WATCH_ACTIVE') == '1' and
            os.environ.get('A2_ALLOCATION_NONCE') == a.attempt.name, 'allocation wrapper required')
    cfg = load(a.config)
    from kit.a2_jobs import configured_inventory
    row = next(r for r in configured_inventory(cfg)['rows'] if r['id'] == a.task)
    try:
        from kit.a2_pipeline import operation
        result = operation(cfg, row, a.attempt)
        atomic(a.attempt/'output.json', result)
        return 0
    except (NotReady, FileNotFoundError) as error:
        atomic(a.attempt/'output.json', dict(status='PENDING', reason=str(error)))
        print('PENDING:', error)
        return PENDING
    except Exception as error:
        atomic(a.attempt/'output.json', dict(status='FAILED', reason=str(error)))
        print('STOP:', type(error).__name__, error)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
