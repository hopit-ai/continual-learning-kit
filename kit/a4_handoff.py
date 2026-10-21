#!/usr/bin/env python3
"""Bound and journal reserved allocation time while saving the chained-job evidence."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *


def job(a):
    doc=binding(a.out,a.standin)
    with lock(a.out/'node-writer'):
        attempt=start_attempt(a.out,doc,'handoff',a.standin);complete=False
        try:
            with job_deadline(doc['handoff_seconds']):
                gpu_run([a.trainer_python,str(KIT/'message_handoff.py'),'--out',str(a.parent_out),
                    '--durable',str(a.durable),'--action','save'],check=True)
            complete=True
        finally:finish_attempt(a.out,attempt,completed=complete)


def configure(p):
    p.add_argument('--parent-out',type=Path,required=True);p.add_argument('--durable',type=Path,required=True)
    p.add_argument('--trainer-python',default=sys.executable)
def main():
    # Saving must not append to a tee log inside the source after it has been copied.
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--standin',action='store_true');configure(p);a=p.parse_args()
    try:job(a)
    except CorrectnessStop as exc:print('STOP: correctness:',exc);return 2
    except NotReady as exc:print('NOT READY:',exc);return 3
    return 0

if __name__=='__main__':raise SystemExit(main())
