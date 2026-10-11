#!/usr/bin/env python3
"""Native main submission: keep attempts; scheduler and projection findings warn."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.v4_phase0_submission import exclusive_json
from kit.v4_phase0_timing import scale
BASE=Path('k8b4/containment/submissions')


def validate_plan(work,plan):
    from kit.v4_phase0_carry import verify_work,public_binding
    from kit.v4_campaign import build
    work=Path(work).resolve();plan=Path(plan).resolve()
    if not plan.is_relative_to(work):raise ValueError('retain the allocation plan inside WORK')
    raw=plan.read_bytes();doc=json.loads(raw)
    if doc.get('phase') not in ('qualification','main') or doc.get('gpus')!=8 or doc.get('ceiling')!=1000:
        from kit.v4_finqa_policy import warn
        warn('submission plan','expected one-node eight-GPU 1000-hour plan',work)
    spec=build(doc['phase'])['v4']['allocation_stages'][doc['stage']]
    if any(doc.get(k)!=spec[k] for k in ('block','block_limit')):
        from kit.v4_finqa_policy import warn
        warn('submission block','plan differs from current registration',work)
    if doc.get('phase0_import_binding')!=public_binding(verify_work(work)['phase0_import']):raise ValueError('submitted phase0 archive binding differs')
    return plan,raw,doc



def require_closed_submissions(work,ledger):
    """Keep unresolved submission exposure visible without refusing a replacement."""
    from kit.v4_finqa_policy import warn
    for folder in (Path(work)/BASE).rglob('intent.json'):
        receipt=folder.with_name('receipt.json')
        if not receipt.exists():warn('submission accounting','job ID/exposure unknown; inspect retained sbatch logs',work);continue
        doc=json.loads(receipt.read_text());entry=ledger.get('allocations',{}).get(doc['job_id'])
        if not entry or not entry.get('end'):warn('submission accounting','reconcile submitted allocation '+doc['job_id'],work)

def submit(work,plan,wrapper):
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID'):
        from kit.v4_finqa_policy import warn
        warn('submission host','plain login shell recommended',work)
    work=Path(work).resolve();plan,raw,doc=validate_plan(work,plan);wrapper=Path(wrapper).resolve()
    if not wrapper.is_relative_to(work) or not wrapper.is_file():raise ValueError('retain your copied pilot wrapper inside WORK')
    from kit.v4_phase0_carry import verify_work,LEDGER_PATH
    from kit.v4_finqa_policy import warn
    try:require_closed_submissions(work,verify_work(work))
    except ValueError as exc:warn('submission accounting',str(exc),work)
    if hashlib.sha256((work/LEDGER_PATH).read_bytes()).hexdigest()!=doc.get('ledger_sha256'):warn('submission accounting','ledger changed since CPU planning; runtime uses current actual spend',work)
    key=hashlib.sha256(raw).hexdigest();folder=work/BASE/key
    if (folder/'receipt.json').exists():return json.loads((folder/'receipt.json').read_text())
    if (folder/'intent.json').exists():
        warn('submission retry','previous attempt unresolved; preserve logs and record a new attempt',work)
        number=2
        while (folder/('attempt-'+str(number))).exists():number+=1
        folder=folder/('attempt-'+str(number))
    from kit.v4_finqa_policy import confine,ceiling
    confine(work,folder);ceiling(work)
    folder.mkdir(parents=True,exist_ok=True);intent=folder/'intent.json'
    exclusive_json(intent,dict(plan_path=str(plan.relative_to(work)),plan_sha256=key,wrapper_path=str(wrapper.relative_to(work)),wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest()))
    with (folder/'submission.log').open('xb',buffering=0) as log:
        done=subprocess.run(['sbatch','--parsable',str(wrapper)],stdout=log,stderr=subprocess.PIPE,timeout=scale(60));os.fsync(log.fileno())
    (folder/'submission-error.txt').write_bytes(done.stderr)
    if done.returncode:raise ValueError('sbatch failed; preserve logs, fix infrastructure and rerun')
    match=re.fullmatch(r'([0-9]+)(?:;([A-Za-z0-9_.-]+))?',(folder/'submission.log').read_text().strip())
    if not match:raise ValueError('unknown submitted job ID; retain failed submission logs for local fix')
    receipt={**json.loads(intent.read_text()),'schema':'v4-finqa-submission.v1','work':str(work),'job_id':match[1],'cluster':match[2]}
    exclusive_json(folder/'receipt.json',receipt)
    return receipt


def reconcile_submissions(work):
    """Recover scheduler spend even when the host payload never started."""
    from kit import p4_contain as pc
    work=Path(work).resolve()
    from kit.v4_finqa_policy import warn
    for intent in sorted((work/BASE).rglob('intent.json')):
        folder=intent.parent
        if not (folder/'receipt.json').exists():warn('submission reconciliation','job ID unknown; retain logs for scheduler reconciliation',work);continue
        receipt=json.loads((folder/'receipt.json').read_text());plan,raw,doc=validate_plan(work,work/receipt['plan_path'])
        if receipt['work']!=str(work):raise pc.Refused('main submission path outside WORK')
        if hashlib.sha256(raw).hexdigest()!=receipt['plan_sha256']:warn('submission reconciliation','retained plan changed; record current scheduler exposure',work)
        observation=pc.allocation_observation(receipt['job_id'],**({'cluster':receipt['cluster']} if receipt.get('cluster') else {}))
        if observation.get('job_id')!=receipt['job_id'] or observation.get('state') not in pc.TERMINAL or not observation.get('start') or not observation.get('end') or not observation.get('width'):
            warn('submission reconciliation','termination/start/width not scheduler-verified for '+receipt['job_id'],work);continue
        ledger=pc.record_allocation(work,observation,doc['block'],doc['block_limit'],1000)
        if ledger.get('accounting_unknown'):warn('submission accounting','allocation telemetry incomplete',work)
        entry=ledger['allocations'][receipt['job_id']]
        entry.update(finqa_submission=receipt,terminal_observation=observation)
        pc.wd.write_durably(work/pc.DIRECTORY/'allocation-ledger.json',ledger)
        pc.refresh_allocation_ledger(work)
        if entry['width']!=8:warn('submission accounting','GPU width differs; actual exposure retained',work)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--script',type=Path,required=True);a=p.parse_args(argv)
    try:doc=submit(a.work,a.plan,a.script)
    except (ValueError,OSError,subprocess.SubprocessError) as exc:print('STOP: '+str(exc));return 2
    print(doc['job_id']);return 0
if __name__=='__main__':raise SystemExit(main())
