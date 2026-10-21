#!/usr/bin/env python3
"""Persist section entry before 4A work, or record an actual allocation end."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
from kit.a4_common import _run_key, _restart_count


def job(a):
    if a.start:
        require(a.parent_out is not None and a.durable is not None,
                '4A start requires parent OUT and durable tree')
        require(a.out.resolve() == (a.parent_out/'A4').resolve(), '4A OUT must be parent OUT/A4')
        with lock(a.out/'node-writer'):
            job_id = os.environ.get('SLURM_JOB_ID', 'standin')
            start_path = a.durable/'job-starts'/(_run_key(job_id, _restart_count())+'.json')
            recorded = json.loads(start_path.read_text()) if start_path.exists() else {}
            value = start_4a(a.out, a.standin, recorded.get('allocation_started_at'))
        from kit.message_handoff import save, locked, committed, sync_dir
        save(a.parent_out, a.durable)
        # rsync creates a new inode: flush the durable copy, not only local NVMe.
        with locked(a.durable):
            generation = committed(a.durable, verify=False)
            marker = generation/'tree/A4/4a-starts'/(_run_key(value['slurm_job_id'], value['slurm_restart_count'])+'.json')
            with marker.open('rb') as handle: os.fsync(handle.fileno())
            for directory in (marker.parent, marker.parent.parent, generation/'tree', generation):
                sync_dir(directory)
        return
    doc = binding(a.out, a.standin)
    with lock(a.out/'node-writer'):
        end_allocation(a.out, doc, a.job_id, a.ended_at, restart=a.restart_count, durable=a.durable)


def configure(p):
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--start', action='store_true', help='mark entry into 4A in the current job')
    action.add_argument('--job-id')
    p.add_argument('--restart-count', type=int, default=0, help='ended Slurm run restart count')
    p.add_argument('--ended-at', type=float, help='actual allocation end, Unix seconds')
    p.add_argument('--parent-out', type=Path); p.add_argument('--durable', type=Path)


def main():
    # The start snapshot must have no active tee log inside its source tree.
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--standin', action='store_true'); configure(p); a=p.parse_args()
    if not a.start and a.ended_at is None: p.error('--job-id requires --ended-at')
    try: job(a)
    except CorrectnessStop as exc: print('STOP: correctness:', exc); return 2
    except NotReady as exc: print('NOT READY:', exc); return 3
    return 0


if __name__ == '__main__': raise SystemExit(main())
