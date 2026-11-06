"""Allocation reservations and CPU reconciliation; failed attempts are charged once."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import math
import os
import time
from kit.nm_common import *

ALLOWANCES={'K_CR':1362,'K_core':1307,'K_archived':1028,'CR':1565,'I0':1270}


@contextmanager
def lock(out):
    path=Path(out)/'budget.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);yield


def reserve(out,cfg,row,record,attempt):
    budget=cfg.get('budget',{});branch=budget.get('branch')
    allowances={'CR_confirmation':830} if row.get('family')=='cr' else ALLOWANCES
    ready(branch in allowances,'selected allocation branch absent')
    need(budget.get('approved_gpu_hours')==allowances[branch],'approved branch allowance differs')
    ready(budget.get('approval_reference'),'allocation approval reference absent')
    seconds=budget.get('row_max_seconds',{}).get(row['id'])
    ready(type(seconds) in (int,float) and math.isfinite(seconds) and seconds>0,'prospective row duration bound absent')
    from kit.a4_common import _slurm_times
    started=os.environ.get('SLURM_JOB_START_TIME')
    if started is None:started=_slurm_times(record['slurm_job_id']).get('allocation_started_at')
    ready(started is not None,'actual allocation start absent')
    started=float(started);need(started<=time.time(),'allocation start is in future')
    reservation=row['gpus']*(seconds+time.time()-started+30)/3600
    with lock(out):
        held=0
        for p in (Path(out)/'attempts').glob('*/*/receipt.json'):
            if p.parent==attempt:continue
            other=load(p)
            need(not(other.get('gpus') and other.get('slurm_job_id')==record['slurm_job_id'] and other.get('allocation_started_at')==started),
                'one row per allocation; submit a new scheduler job')
            if other.get('gpus'):
                held+=other.get('allocation_gpu_hours') if other.get('allocation_gpu_hours') is not None else other.get('reserved_gpu_hours',0)
        ready(held+reservation<=allowances[branch],'approved allocation cannot cover prospective row')
        record.update(allocation_started_at=started,reserved_gpu_hours=reservation,max_seconds=seconds,budget_branch=branch)
        atomic(attempt/'receipt.json',record)
    return seconds


def reconcile(out,*,family=None,query=None):
    if query is None:
        from kit.a4_common import _slurm_times
        query=_slurm_times
    rows=[];pending=[];total=0
    with lock(out):
        for p in (Path(out)/'attempts').glob('*/*/receipt.json'):
            value=load(p)
            if family and value['family']!=family:continue
            if not value.get('gpus') or value.get('technical_synthetic'):continue
            if value.get('allocation_gpu_hours') is None:
                start=value.get('allocation_started_at')
                if start is None:
                    pending.append(value['task']+':start');continue
                ended=query(value['slurm_job_id'],start).get('allocation_ended_at')
                if ended is None:pending.append(value['slurm_job_id']);continue
                need(ended>=start,'scheduler end precedes start')
                value.update(allocation_ended_at=ended,allocation_gpu_hours=value['gpus']*(ended-start)/3600,cost_status='COMPLETE')
                atomic(p,value)
                pointer=Path(out)/'state'/(value['task']+'.json')
                if pointer.is_file() and load(pointer)['receipt']==str(p.relative_to(out)):
                    atomic(pointer,dict(receipt=str(p.relative_to(out)),sha256=sha(p)))
            total+=value['allocation_gpu_hours'];rows.append(dict(task=value['task'],phase=value['phase'],nonce=value['nonce'],gpu_hours=value['allocation_gpu_hours'],
                wall_seconds=value['allocation_ended_at']-value['allocation_started_at'],wall_time_risk=value['allocation_ended_at']-value['allocation_started_at']>86400,status=value['status']))
    return dict(status='PENDING' if pending else 'COMPLETE',gpu_hours=total,rows=rows,pending=pending,
        queue_gaps_gpu_hours=0,basis='allocated GPUs × actual allocation wall time, every failed/rejected attempt included')
