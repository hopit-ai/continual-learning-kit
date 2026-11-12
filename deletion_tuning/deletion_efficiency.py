#!/usr/bin/env python3
"""Optional allocation wrapper for deletion native train/qualify/export/prep rows.

The operator runs the original registered command after --, unchanged. Use the
same durable --out as deletion_jobs to include these clocks in ESSENTIAL.
"""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os
import subprocess
import time
import uuid
from kit.nm_common import atomic
from kit.trace_jobs import timing_phase, timing_finish


def run(out,phase,gpus,command):
    attempt=Path(out)/'attempts'/('native-'+phase)/uuid.uuid4().hex
    attempt.mkdir(parents=True)
    started=time.time()
    timing=attempt/'phase-times.jsonl'
    env=dict(os.environ,V7H_PHASE_FILE=str(timing),V7H_PHASE_GPUS=str(gpus))
    # The subprocess is the entire phase, including load, failure and teardown.
    code=subprocess.call(command,env=env)
    ended=time.time()
    totals=dict(train=0.,score=0.,judge=0.,export=0.,prep=0.,**{'idle-in-allocation':0.})
    totals[phase]=(ended-started)*gpus
    atomic(attempt/'receipt.json',dict(row='native-'+phase,row_type=phase,gpus=gpus,
        started=started,ended=ended,status='COMPLETE' if code==0 else 'ERROR',
        phase_gpu_seconds=totals,process_gpu_hours=(ended-started)*gpus/3600))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--phase',choices=('train','score','judge','export','prep'),required=True)
    parser.add_argument('--gpus',type=int,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.command or args.gpus<0:
        parser.error('nonnegative GPU count and command required')
    command=args.command[1:] if args.command[0]=='--' else args.command
    raise SystemExit(run(args.out,args.phase,args.gpus,command))
