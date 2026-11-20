"""Allocation reservations and CPU reconciliation; failed attempts are charged once."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import math
import os
import time
from kit.nm_common import *

CR_WARN = 'WARN CR allocation accounting exceeds 830 GPU-hours (warn-only)'

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
    cr = row.get('family') == 'cr'
    row_start = record.get('started') if cr else started
    if cr:
        need(type(row_start) in (int, float) and math.isfinite(row_start)
             and started <= row_start <= time.time(), 'actual row receipt start absent/invalid')
    reservation=row['gpus']*(seconds+time.time()-row_start+30)/3600
    if cr:record['row_reservation_gpu_hours']=reservation
    with lock(out):
        held=0
        paths=(Path(out)/'attempts').glob('*/*/receipt.json')
        for p in sorted(paths) if cr else paths:
            if p.parent==attempt:continue
            other=load(p)
            need(cr or not(other.get('gpus') and other.get('slurm_job_id')==record['slurm_job_id'] and other.get('allocation_started_at')==started),
                'one row per allocation; submit a new scheduler job')
            if other.get('gpus'):
                if cr and other.get('family')=='cr':
                    _,_,evidence=receipt_bindings(out,p,other)
                    exact=dict(other)
                    if evidence and other.get('cost_status')=='COMPLETE-UPPER-BOUND':
                        held+=cr_upper_bound_charge(other)
                    elif evidence and finish_cr_row(exact):held+=exact['allocation_gpu_hours']
                    else:held+=retained_cr_reservation(other)
                else:
                    held+=other.get('allocation_gpu_hours') if other.get('allocation_gpu_hours') is not None else other.get('reserved_gpu_hours',0)
        if cr:
            if held+reservation > allowances[branch]:
                record.setdefault('budget_warnings', []).append(CR_WARN)
        else:
            ready(held+reservation<=allowances[branch],'approved allocation cannot cover prospective row')
        record.update(allocation_started_at=started,reserved_gpu_hours=reservation,max_seconds=seconds,budget_branch=branch)
        atomic(attempt/'receipt.json',record)
    return seconds


def finish_cr_row(value):
    """Only wrapper completion through confirmed teardown supplies exact cost.

    UTC is audit information. MONOTONIC includes launcher setup and cleanup;
    it remains authoritative across wall-clock adjustments.
    """
    elapsed = value.get('monotonic_elapsed_seconds')
    if (value.get('status') not in ('COMPLETE','FAILED','PENDING','ARCHIVED UNREAD') or value.get('ended') is None
            or value.get('wrapper_completed') is not True
            or value.get('teardown_confirmed') is not True):
        return False
    need(all(type(v) in (int, float) and math.isfinite(v)
             for v in (value.get('started'), value['ended'], elapsed)) and elapsed >= 0,
         'CR wrapper timing invalid')
    value.update(row_started_at=value['started'], row_ended_at=value['ended'],
        allocation_gpu_hours=value['gpus']*elapsed/3600,
        reserved_gpu_hours=0, cost_status='COMPLETE')
    return True


def cr_holder_times(job_id, start):
    """Match the requested incarnation, including history after a requeue."""
    import subprocess
    from datetime import datetime
    from kit.a4_common import _slurm_times
    current = _slurm_times(job_id, start)
    matched = current.get('allocation_started_at') == start and start is not None
    result = dict(current) if matched else {}
    terminal = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
                'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED',
                'REQUEUED', 'REQUEUE_HOLD', 'REQUEUE_FED'}
    try:
        reply = subprocess.run(['sacct', '-D', '-j', str(job_id), '-X', '-n', '-P',
            '-o', 'JobIDRaw,Start,End,State,AllocTRES'],
            capture_output=True, text=True, timeout=10, check=True)
        records = []
        for line in reply.stdout.splitlines():
            fields = line.split('|')
            if len(fields) < 5 or fields[0] != str(job_id): continue
            try:
                if datetime.fromisoformat(fields[1]).timestamp() != start: continue
                ended = (datetime.fromisoformat(fields[2]).timestamp()
                         if fields[3].split()[0].split('+')[0] in terminal else None)
                record = dict(allocation_started_at=start, allocation_ended_at=ended)
                for token in fields[4].split(','):
                    if token.startswith('gres/gpu='):
                        record['allocated_gpus'] = int(token.split('=')[1])
                records.append(record)
            except (ValueError, IndexError):
                continue
        # Ambiguous history cannot certify this incarnation. Identical duplicate
        # accounting records are harmless; conflicting records remain PENDING.
        if records and all(record == records[0] for record in records):
            result.update(records[0])
            matched = True
        elif records:
            matched = False
    except (OSError, subprocess.SubprocessError):
        pass
    if not matched:
        return dict(history_unmatched=True,
                    pending_reason='scheduler holder history cannot match allocation start')
    return result


def receipt_bindings(out, path, value):
    """Validate all existing bindings before any write; never heal a bad hash."""
    need(value['nonce'] == path.parent.name and value['task'] == path.parent.parent.name,
         'CR receipt attempt identity differs')
    pointer = Path(out)/'state'/(value['task']+'.json')
    current = load(pointer) if pointer.is_file() else None
    if current and current.get('receipt') == str(path.relative_to(out)):
        need(current.get('sha256') == sha(path), 'CR existing state receipt hash mismatch')
    completion = path.with_name('completion.json')
    evidence = load(completion) if completion.is_file() else None
    if evidence:
        need(evidence == dict(task=value['task'], nonce=value['nonce'], sha256=sha(path)),
             'CR completion receipt hash/attempt mismatch')
    return pointer, current, evidence


def complete_binding(path, value):
    atomic(path.with_name('completion.json'),
           dict(task=value['task'], nonce=value['nonce'], sha256=sha(path)))


def retained_cr_reservation(value):
    # Preserve the reservation across a crash between receipt and completion
    # writes. Readers must never treat an unbound exact cost as a release.
    amount=value.get('row_reservation_gpu_hours',value.get('reserved_gpu_hours',0))
    if not amount:
        seconds=value.get('max_seconds')
        if seconds is None:
            # Another wrapper can have created its RUNNING receipt before it
            # obtains the budget lock. Use its frozen prospective ceiling,
            # without mistaking setup-in-progress for a released reservation.
            seconds=load(KIT/'resources/cr-confirmation/row-max-seconds.json').get(value['task'])
        need(type(seconds) in (int,float) and math.isfinite(seconds) and seconds>0,
             'CR original reservation/duration evidence absent')
        amount=value['gpus']*(seconds+30)/3600
    need(type(amount) in (int,float) and math.isfinite(amount) and amount>0,
         'CR original reservation invalid')
    return amount


def write_cr_accounting(out,path,value,pointer,current,evidence):
    atomic(path,value)
    if evidence:complete_binding(path,value)
    if current and current.get('receipt')==str(path.relative_to(out)):
        atomic(pointer,dict(receipt=current['receipt'],sha256=sha(path)))


def cr_upper_bound_charge(value):
    start,end=value.get('started'),value.get('allocation_ended_at')
    need(value.get('status') in ('COMPLETE','FAILED','PENDING','ARCHIVED UNREAD')
         and all(type(v) in (int,float) and math.isfinite(v) for v in (start,end))
         and end>=start, 'CR upper-bound receipt is not terminal')
    expected=value['gpus']*(end-start)/3600
    need(expected==value.get('allocation_gpu_hours') and value.get('reserved_gpu_hours')==0,
         'CR upper-bound charge differs')
    return expected


def reconcile_cr(out, path, value, query, holders):
    pointer, current, evidence = receipt_bindings(out, path, value)
    key = (value['slurm_job_id'], value.get('allocation_started_at'))
    if key not in holders:
        holders[key] = dict(query(*key))
    holder = holders[key]
    if key[1] is not None and holder.get('allocation_started_at') is not None:
        if holder['allocation_started_at'] != key[1]:
            holder.clear()
            holder.update(history_unmatched=True,
                          pending_reason='scheduler holder history cannot match allocation start')
    # Exact charges must come from an attempt-bound terminal receipt. An ended
    # timestamp in a RUNNING or unbound receipt is not completion evidence.
    need(not (value.get('status') == 'RUNNING' and value.get('ended') is not None),
         'CR RUNNING receipt has an end timestamp')
    if holder.get('history_unmatched'):
        # Even a completed row stays visibly pending until its holder is matched;
        # retain exact evidence/charges unchanged rather than refusing silently.
        return False
    if evidence and value.get('cost_status') == 'COMPLETE-UPPER-BOUND':
        cr_upper_bound_charge(value)
        return True
    before=dict(value)
    if evidence and finish_cr_row(value):
        if before!=value:write_cr_accounting(out,path,value,pointer,current,evidence)
        return True
    ended = holder.get('allocation_ended_at')
    if ended is None:
        reserved=retained_cr_reservation(value)
        if value.get('reserved_gpu_hours')!=reserved or value.get('allocation_gpu_hours') is not None:
            value.update(reserved_gpu_hours=reserved,allocation_gpu_hours=None,cost_status='PENDING scheduler end')
            write_cr_accounting(out,path,value,pointer,current,evidence)
        return False
    need(type(ended) in (int, float) and math.isfinite(ended) and ended >= value['started'],
         'scheduler end precedes CR row start')
    value.update(row_started_at=value['started'], row_ended_at=ended,
        allocation_ended_at=ended, allocation_gpu_hours=value['gpus']*(ended-value['started'])/3600,
        reserved_gpu_hours=0, cost_status='COMPLETE-UPPER-BOUND',
        cost_reason='wrapper-killed, teardown unconfirmed or completion evidence missing')
    if value['status'] == 'RUNNING': value['status'] = 'PENDING'
    atomic(path, value)
    complete_binding(path, value)
    if current and current.get('receipt') == str(path.relative_to(out)):
        atomic(pointer, dict(receipt=current['receipt'], sha256=sha(path)))
    return True


def reconcile(out,*,family=None,query=None):
    cr_query = query or cr_holder_times
    if query is None:
        from kit.a4_common import _slurm_times
        query=_slurm_times
    rows=[];pending=[];total=0;warnings=[];cr_total=0;has_cr=False;holders={};holder_rows={}
    with lock(out):
        paths=(Path(out)/'attempts').glob('*/*/receipt.json')
        for p in sorted(paths) if family=='cr' else paths:
            value=load(p)
            if family and value['family']!=family:continue
            if not value.get('gpus') or value.get('technical_synthetic'):continue
            if value.get('family') == 'cr':
                has_cr=True
                warnings.extend(value.get('budget_warnings', []))
                if not reconcile_cr(out, p, value, cr_query, holders):
                    key=(value['slurm_job_id'],value.get('allocation_started_at'))
                    pending.append(value['task']+':'+holders[key].get('pending_reason','holder-running'));continue
                cr_total += value['allocation_gpu_hours']
                key = (value['slurm_job_id'], value.get('allocation_started_at'))
                holder_rows[key] = holder_rows.get(key, 0) + value['allocation_gpu_hours']
                if cr_total > 830:
                    warnings.append(CR_WARN)
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
            start = value['row_started_at'] if value['family']=='cr' else value['allocation_started_at']
            end = value['row_ended_at'] if value['family']=='cr' else value['allocation_ended_at']
            total+=value['allocation_gpu_hours'];rows.append(dict(task=value['task'],phase=value['phase'],nonce=value['nonce'],gpu_hours=value['allocation_gpu_hours'],
                wall_seconds=(value['monotonic_elapsed_seconds'] if value['family']=='cr' and value['cost_status']=='COMPLETE' else end-start),
                wall_time_risk=((value['monotonic_elapsed_seconds'] if value['family']=='cr' and value['cost_status']=='COMPLETE' else end-start)>86400),status=value['status'],
                **(dict(cost_status=value['cost_status']) if value['family']=='cr' else {})))
    result = dict(status='PENDING' if pending else 'COMPLETE',gpu_hours=total,rows=rows,pending=pending,
        queue_gaps_gpu_hours=0,basis='allocated GPUs × actual allocation wall time, every failed/rejected attempt included')
    if family == 'cr' or has_cr:
        reports=[]
        for key, holder in sorted(holders.items(), key=lambda item: (str(item[0][0]), item[0][1] or 0)):
            job, start = key
            end, gpus = holder.get('allocation_ended_at'), holder.get('allocated_gpus')
            start = holder.get('allocation_started_at', start)
            life = end-start if end is not None and start is not None else None
            allocated = gpus*life/3600 if gpus is not None and life is not None else None
            charged = holder_rows.get(key, 0)
            unattributed = allocated-charged if allocated is not None else None
            if unattributed is not None and unattributed < -1e-9:
                warnings.append('WARN CR rows exceed scheduler holder GPU-hours')
            reports.append(dict(slurm_job_id=job, allocation_started_at=start,
                allocation_ended_at=end, allocated_gpus=gpus, holder_gpu_hours=allocated,
                row_gpu_hours=charged, holder_unattributed_gpu_hours=unattributed,
                status='COMPLETE' if allocated is not None else 'PENDING scheduler life/GPU allocation'))
        unknown = any(h['holder_unattributed_gpu_hours'] is None for h in reports)
        result.update(warnings=sorted(set(warnings)), holders=reports,
            holder_gpu_hours=None if unknown else sum(h['holder_gpu_hours'] for h in reports),
            holder_row_gpu_hours=sum(holder_rows.values()),
            holder_unattributed_gpu_hours=None if unknown else sum(h['holder_unattributed_gpu_hours'] for h in reports),
            basis='allocated row GPUs × wrapper MONOTONIC elapsed through confirmed teardown; scheduler upper bound for missing completion; no idle holding')
    return result
