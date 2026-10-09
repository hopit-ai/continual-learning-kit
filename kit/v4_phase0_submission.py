#!/usr/bin/env python3
"""Login-node submission and scheduler accounting binding, standard library only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
BASE=Path('v4/report-phase0')


def receipt_path(work):return Path(work)/BASE/'submission.json'


def exclusive_json(path,doc):
    """Retain a submission exactly once, including directory durability."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(doc,indent=2)+'\n');handle.flush();os.fsync(handle.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def validate_plan(work,path):
    path=Path(path).resolve();work=Path(work).resolve()
    if not path.is_relative_to(work):raise ValueError('submission plan must be inside WORK')
    raw=path.read_bytes();doc=json.loads(raw)
    if any(doc.get(k)!=v for k,v in {'phase':'phase0','block':'qualification','block_limit':100,'ceiling':560,'gpus':8,'sbatch_time':'08:00:00','phase0_allocation_cap_gpu_hours':64}.items()):
        raise ValueError('submission requires the frozen phase0 eight-hour, eight-GPU, 64/100/560 plan')
    return path,raw,doc


def record_submission(work,job_id,plan,wrapper):
    """Bind a scheduler job to the exact WORK, retained plan and partner wrapper."""
    work=Path(work).resolve();plan,raw,doc=validate_plan(work,plan);wrapper=Path(wrapper).resolve()
    if not wrapper.is_relative_to(work):raise ValueError('submission wrapper must be retained inside WORK')
    match=re.fullmatch(r'([0-9]+)(?:;([A-Za-z0-9_.-]+))?',str(job_id).strip())
    if not match:raise ValueError('sbatch did not return a parsable job id; retain submission.log and return it to the owner')
    result={'schema':'v4-phase0-submission.v1','job_id':match[1],'cluster':match[2], 'work':str(work),
        'plan_path':str(plan.relative_to(work)),'plan_sha256':hashlib.sha256(raw).hexdigest(),
        'wrapper_path':str(wrapper.relative_to(work)),'wrapper_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        'submitted_at':datetime.now(timezone.utc).isoformat()}
    exclusive_json(receipt_path(work),result)
    return result


def submit(work,plan,wrapper):
    """Call the partner's sbatch form once, durably record output before returning."""
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID'):raise ValueError('submit from a plain login shell outside any Slurm job or step')
    work=Path(work).resolve();plan,raw,doc=validate_plan(work,plan)
    wrapper=Path(wrapper).resolve()
    if not wrapper.is_relative_to(work) or not wrapper.is_file():raise ValueError('retain the partner wrapper inside WORK before submission')
    intent=work/BASE/'submission-intent.json'
    if intent.exists() or receipt_path(work).exists():raise ValueError('already attempted phase0 submission; never submit a replacement job; return this WORK')
    exclusive_json(intent,{'work':str(work),'plan_sha256':hashlib.sha256(raw).hexdigest(),'wrapper':str(wrapper)})
    # Keep sbatch stdout even if the batch/container dies before any payload file.
    log=work/BASE/'submission.log'
    with log.open('xb',buffering=0) as handle:
        done=subprocess.run(['sbatch','--parsable',str(wrapper)],stdout=handle,stderr=subprocess.PIPE,timeout=600)
        os.fsync(handle.fileno())
    error=work/BASE/'submission-error.txt'
    with error.open('xb') as handle:handle.write(done.stderr);handle.flush();os.fsync(handle.fileno())
    if done.returncode:raise ValueError('sbatch submission failed; retain submission.log/submission-error.txt and return this WORK')
    return record_submission(work,log.read_text().strip(),plan,wrapper)


def reconcile_submission(work):
    """Recover only the scheduler-verified submitted job, including pre-payload failures."""
    from kit import p4_contain as pc
    work=Path(work).resolve()
    try:
        raw=receipt_path(work).read_bytes();receipt=json.loads(raw)
        if receipt['schema']!='v4-phase0-submission.v1' or receipt['work']!=str(work):raise ValueError('submission WORK identity differs')
        plan,plan_raw,doc=validate_plan(work,work/receipt['plan_path'])
        if hashlib.sha256(plan_raw).hexdigest()!=receipt['plan_sha256']:raise ValueError('submitted plan hash changed')
        if not re.fullmatch(r'[0-9]+',receipt['job_id']):raise ValueError('invalid submitted job identity')
    except (ValueError,KeyError,OSError) as exc:raise pc.Refused('submission receipt cannot be reconciled: '+str(exc)) from exc
    observation=(pc.allocation_observation(receipt['job_id'],cluster=receipt['cluster']) if receipt.get('cluster') else pc.allocation_observation(receipt['job_id']))
    width=observation.get('width')
    if (observation.get('job_id')!=receipt['job_id'] or observation.get('state') not in pc.TERMINAL or
        isinstance(width,bool) or not isinstance(width,int) or width<=0 or not observation.get('start') or not observation.get('end')):
        raise pc.Refused('submitted job termination/start/GPU width not scheduler-verified; accounting remains unknown')
    if pc.epoch(observation['end'])<pc.epoch(observation['start']):raise pc.Refused('submitted job has a reversed scheduler interval')
    ledger=pc.record_allocation(work,observation,'qualification',100,560)
    if ledger.get('accounting_unknown'):raise pc.Refused('submitted allocation start/width differs from existing ledger; accounting remains unknown')
    entry=ledger['allocations'][receipt['job_id']]
    entry.update(phase0=True,cluster=receipt.get('cluster'),submission_sha256=hashlib.sha256(raw).hexdigest(),terminal_observation=observation)
    ledger.update(block='qualification',block_limit=100)
    pc.wd.write_durably(work/pc.DIRECTORY/'allocation-ledger.json',ledger)
    ledger=pc.refresh_allocation_ledger(work)
    if width!=8:raise pc.Refused('changed safety settings: submitted GPU width differs from the eight-GPU plan; actual allocation charge retained')
    return ledger


def live_submission_accounting(evidence,report):
    """Check a running submission's identity and exposure without post-job fields."""
    name=str(BASE/'submission.json')
    if name not in evidence.files:return None
    import copy
    import time
    from kit import p4_contain as pc
    from kit.v4_budget import allocation_spend
    from kit.v4_phase0 import allocation_cap
    receipt=evidence.json(name);raw=evidence.files[receipt['plan_path']];plan=json.loads(raw)
    def require(ok,cause):
        if not ok:raise ValueError('live submitted allocation '+cause)
    work=getattr(evidence,'work',None)
    require(work is not None and receipt.get('work')==str(Path(work).resolve()),'WORK identity differs')
    job=receipt['job_id']
    require(receipt.get('schema')=='v4-phase0-submission.v1' and re.fullmatch(r'[0-9]+',job) and os.environ.get('SLURM_JOB_ID')==job,'job identity differs')
    require(hashlib.sha256(raw).hexdigest()==receipt['plan_sha256'] and
        all(plan.get(k)==v for k,v in {'phase':'phase0','block':'qualification','block_limit':100,'ceiling':560,'gpus':8,'sbatch_time':'08:00:00','phase0_allocation_cap_gpu_hours':64}.items()),'plan binding differs')
    for key,path in {'phase0_prepare_sha256':'v4/report-inputs/prepare-receipt-phase0.json','phase0_presend_sha256':str(BASE/'presend/containment-presend.json')}.items():
        require(plan.get(key)==hashlib.sha256(evidence.files[path]).hexdigest(),'CPU evidence binding differs')
    ledger=evidence.json('k8b4/containment/allocation-ledger.json')
    require(not ledger.get('accounting_unknown') and not any(e.get('accounting_unknown') for e in ledger['allocations'].values()),'exposure is unknown')
    require(set(ledger['allocations'])-set(plan['prior_allocations'])=={job},'must use exactly one new allocation')
    entry=ledger['allocations'][job]
    require(entry.get('phase0') is True and entry.get('width')==8 and entry.get('start') and not entry.get('end') and entry.get('state')=='RUNNING','running job/start/width differs')
    require('k8b4/containment/v4-stop.json' not in evidence.files,'has an active hard stop')
    require(ledger.get('ceiling')==560 and ledger.get('block_limits',{}).get('qualification')==100,'budget settings differ')
    allocation_cap(ledger)
    # Observe full-width idle/CPU time through this gate, not the last GPU row.
    live=copy.deepcopy(ledger)
    for allocation in live['allocations'].values():
        if not allocation.get('end'):allocation['observed_until']=pc.wd.precise_text(pc.wd.from_epoch(time.time()))
    total,blocks=allocation_spend(live)
    charge,phase_blocks=allocation_spend({'allocations':{job:live['allocations'][job]}})
    require(total<=560 and blocks.get('qualification',0)<=100 and charge<=64 and set(phase_blocks)=={'qualification'},'64/100/560 budget reached or wrong block')
    report['allocation_gpu_hours']=float(charge)
    report['tables']['allocation_accounting']={'job_id':job,'scheduler_state':'RUNNING','allocation_gpu_hours':float(charge),
        'qualification_gpu_hours':float(blocks.get('qualification',0)),'total_gpu_hours':float(total),'terminal':False}
    return job


def archived_submission_accounting(evidence,report):
    """Report full allocation hours before any technical/setup refusal, from raw scheduler evidence."""
    name=str(BASE/'submission.json')
    if name not in evidence.files:return
    from kit import p4_contain as pc
    from kit.v4_budget import allocation_spend
    receipt=evidence.json(name)
    raw=evidence.files[receipt['plan_path']]
    plan=json.loads(raw)
    if (receipt.get('schema')!='v4-phase0-submission.v1' or hashlib.sha256(raw).hexdigest()!=receipt['plan_sha256'] or
        any(plan.get(k)!=v for k,v in {'phase':'phase0','block':'qualification','block_limit':100,'ceiling':560,'gpus':8,'sbatch_time':'08:00:00','phase0_allocation_cap_gpu_hours':64}.items())):raise ValueError('submitted phase0 plan binding differs')
    ledger=evidence.json('k8b4/containment/allocation-ledger.json')
    job=receipt['job_id'];entry=ledger['allocations'][job]
    if not entry.get('phase0') or entry.get('submission_sha256')!=hashlib.sha256(evidence.files[name]).hexdigest():
        raise ValueError('submitted allocation ledger binding differs')
    observation=entry['terminal_observation'];scheduler=observation['observation']
    if scheduler['returncode']!=0:raise ValueError('scheduler accounting command failed')
    def stamp(value):
        return pc.wd.precise_text(pc.wd.from_epoch(pc.archived_epoch(value,observation.get('scheduler_utc_offset'))))
    values=pc.values(scheduler['stdout'])
    if values.get('JobId')==job:
        state=values.get('JobState');start=stamp(values.get('StartTime'));end=stamp(values.get('EndTime'));width=pc.allocation_fields(values)['width']
    else:
        values=pc.accounting_values(scheduler['stdout'],job,fields='JobIDRaw,State,Start,End,AllocTRES')
        rows=[line.split('|') for line in scheduler['stdout'].splitlines() if line.split('|')[0]==job]
        if len(rows)!=1 or len(rows[0])<5:raise ValueError('submitted scheduler job identity differs')
        values['AllocTRES']=rows[0][4]
        state=values.get('State');start=stamp(values.get('StartTime'));end=stamp(values.get('EndTime'));width=pc.allocation_fields(values)['width']
    if state not in pc.TERMINAL or not start or not end or not width or entry['start']!=start or entry['end']!=end or entry['width']!=width:
        raise ValueError('submitted allocation interval/termination/width differs from raw scheduler evidence')
    total,blocks=allocation_spend(ledger)
    charge,phase_blocks=allocation_spend({'allocations':{job:entry}})
    report['allocation_gpu_hours']=float(charge)
    report['tables']['allocation_accounting']={'job_id':job,'scheduler_state':state,'allocation_gpu_hours':float(charge),
        'qualification_gpu_hours':float(blocks.get('qualification',0)),'total_gpu_hours':float(total)}
    if width!=8:raise ValueError('submitted allocation GPU width differs from the eight-GPU plan; actual allocation hours reported')
    if total>560 or blocks.get('qualification',0)>100 or charge>64 or set(phase_blocks)!={'qualification'}:
        raise ValueError('submitted allocation exceeds phase0 64, qualification 100 or ceiling 560 GPU-hour cap')
    if len([j for j in ledger['allocations'] if j not in plan.get('prior_allocations',{})])!=1:
        raise ValueError('phase0 must use exactly one submitted allocation')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--script',type=Path,required=True)
    a=p.parse_args(argv)
    try:doc=submit(a.work,a.plan,a.script)
    except (ValueError,OSError,subprocess.SubprocessError) as exc:print('STOP: '+str(exc));return 2
    print(doc['job_id']);return 0
if __name__=='__main__':raise SystemExit(main())
