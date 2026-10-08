#!/usr/bin/env python3
"""CPU-only next-allocation plan from closed scheduler accounting; no submission."""
from __future__ import annotations
import argparse
import json
import math
import os
import hashlib
from pathlib import Path
import shlex
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.v4_budget import allocation_spend, number
from kit.v4_campaign import build


def completed_units(work,components):
    """Count only current PASS units; failed/started attempts are not deductions."""
    from kit import runner
    from kit.v4_campaign import slots
    old=os.environ.get('WORK');os.environ['WORK']=str(work)
    try:
        campaign=runner.load_campaign(Path(__file__).with_name('campaigns')/'v4-main.yaml')
        rows={row['id']:row for row in campaign['rows']}
        def reusable(name,row):
            verdict=runner.latest_verdict(campaign,name) or {}
            if verdict.get('verdict')!='PASS' or not isinstance(verdict.get('inputs'),dict):return False
            try:
                start=json.loads((runner.row_dir(campaign,name)/('attempt-%d'%verdict['attempt'])/'start.json').read_text())
            except (KeyError,OSError,ValueError):return False
            return (start.get('campaign_sha256')==campaign['_sha256'] and start.get('row')==name
                and start.get('inputs')==verdict['inputs'] and not runner.gate(campaign,row)
                and not runner.stale_inputs(campaign,row,verdict))
        passed={name for name,row in rows.items() if reusable(name,row)}
        done={};units={row['key']:row['units'] for row in components}
        def add(key,n):done[key]=done.get(key,0)+n
        for role in ('teacher','rewrite'):
            for task in ('chemistry','finqa'):
                if role+'-'+task in passed:
                    for op in ('generate','verify','corpus_reload','serial_merge'):
                        key=f'{op}:{role}:{task}';done[key]=units[key]
        for job in slots():
            ident=job['id'];arm=job['arm'];task=job['task']
            if 'schedule-'+ident in passed:add('cpu:schedule:'+task,1)
            if 'train-'+ident in passed:
                add(f'train:{arm}:{task}',40);add(f'startup:{arm}:{task}',1);add(f'export:{arm}:{task}',2)
            if 'merge-'+ident in passed:add(f'merge:{arm}:{task}',2)
            for panel,n in (('chemistry',159),('finqa',1147),('chemistry_probe',51)):
                for cap in (2048,8192):
                    if f'score-{ident}-{panel}-{cap}' in passed:
                        add(f'eval:{panel}:{cap}',n);add(f'reload:{panel}:{cap}',1)
            for panel in ('chemistry','finqa'):
                for step in (0,20,40):
                    if f'probe-{ident}-{panel}-{step}' in passed:
                        add(f'probe:{panel}:2048',160);add('probe_reload:'+panel,1)
        for task in ('chemistry','finqa'):
            if 'conditioned-probe-'+task in passed:
                done['conditioned_probe:'+task]=160;done['conditioned_reload:'+task]=1
        for operation in ('coverage','training-set','gate','report','archive-pack'):
            ident='owner-registration-gate' if operation=='gate' else operation
            if ident in passed:done['cpu:'+operation]=1
        ids=list(rows)
        for stage,spec in campaign['v4']['allocation_stages'].items():
            names=ids[ids.index(spec['first']):ids.index(spec['last'])+1]
            # Resume still checks each PASS before skipping it. Do not deduct
            # dispatch work: this conservatively prices the full scan again.
            if all(name in passed for name in names):
                done['runner:dispatch:'+stage]=units['runner:dispatch:'+stage]
        if passed:done['_passed_rows']=sorted(passed)
        return done
    finally:
        if old is None:os.environ.pop('WORK',None)
        else:os.environ['WORK']=old


def admission(work,ledger,stage,minutes,site_minutes):
    """Re-read qualification evidence and bind owner caps before any sbatch."""
    from kit.v4_readers import read_qualification
    from kit.v4_budget import reproject, stage_caps
    archive=Path(os.environ['V4_QUALIFICATION_ARCHIVE']);owner=json.loads(Path(os.environ['V4_OWNER_PROCEED']).read_text())
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if owner.get('owner_proceed') is not True or owner.get('qualification_sha256')!=digest:
        raise ValueError('owner proceed archive binding differs')
    qualified=read_qualification(archive)
    if qualified['status']!='proceed':raise ValueError('qualification admission refused: '+str(qualified['reasons']))
    if owner.get('row_caps')!=qualified['row_caps']:raise ValueError('owner row caps differ from qualification')
    components=qualified['budget']['components']
    windows={key:min(value,site_minutes) for key,value in qualified['budget']['allocation_windows'].items()}
    windows[stage]=minutes
    result=reproject(ledger,components,windows,completed_units(work,components),caps=stage_caps(components))
    if result['status']!='proceed':raise ValueError('allocation admission refused: '+str(result['reasons']))
    return {**result,'qualification_status':qualified['status'],'qualification_row_caps':qualified['row_caps'],'qualification_sha256':digest,'owner_sha256':hashlib.sha256(Path(os.environ['V4_OWNER_PROCEED']).read_bytes()).hexdigest()}


def plan(work, phase, stage, requested_minutes, site_minutes):
    """Choose whole minutes conservatively, retaining the full eight-GPU reservation."""
    work=Path(work).resolve()
    registration=build(phase)['v4']['allocation_stages'][stage]
    ledger_path=work/'k8b4/containment/allocation-ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    if ledger:
        if any(not entry.get('end') for entry in ledger['allocations'].values()):
            raise ValueError('close and reconcile every allocation before planning its replacement')
        spent,blocks=allocation_spend(ledger)
        limits={'qualification':100,'scientific':560}
        if any(block not in limits or charge>limits[block] for block,charge in blocks.items()):
            raise ValueError('earlier budget block overrun or unknown block')
    else:spent,blocks=number(0,'initial spend'),{}
    for value in (requested_minutes,site_minutes):
        if type(value) is not int or value<=0:raise ValueError('duration must be positive whole minutes')
    remaining=min(number(registration['block_limit'],'limit')-blocks.get(registration['block'],0),number(560,'ceiling')-spent)
    minutes=min(requested_minutes,site_minutes,int(max(0,remaining)*60/8))
    # Ten-minute selftest plus at least the registered minimum row cap.
    if minutes<15:raise ValueError('remaining budget cannot fit selftest and one minimum-cap row; stop: scope')
    duration=f'{minutes//60:02d}:{minutes%60:02d}:00'
    receipt_path=work/'k8b4/containment/containment-receipt.json'
    node=None
    if receipt_path.exists():
        from kit.p4_frozen import seal
        receipt=json.loads(receipt_path.read_text())
        if receipt!=seal(receipt) or not receipt.get('ok'):raise ValueError('reviewed allocation receipt altered or failed')
        node=receipt['node_list']
        if not __import__('re').fullmatch(r'[A-Za-z0-9_.\[\],-]+',node):raise ValueError('unknown allocation node')
    phase0_binding={}
    if phase=='phase0':
        from kit.v4_phase0 import verify_presend, verify_prepare, required_seconds, allocation_cap, ALLOCATION_CAP_GPU_HOURS, frozen_launch_environment
        verify_presend(work);prepared=verify_prepare(work)
        environment_seconds=prepared.get('environment_check',{}).get('deadline_seconds',300)
        if requested_minutes*8/60 > ALLOCATION_CAP_GPU_HOURS:
            raise ValueError('phase0 allocation cap of 22 GPU-hours refuses this request')
        if ledger:allocation_cap(ledger)
        if ledger and any(e.get('phase0') for e in ledger['allocations'].values()):
            raise ValueError('phase0 admits exactly one allocation; return partial evidence for review')
        minutes=min(minutes,110)
        if required_seconds(environment_seconds)>minutes*60:
            raise ValueError('prepared environment-check deadline cannot fit the fixed 110-minute allocation with the 600-second self-test; stop before submission')
        if minutes*60<required_seconds():raise ValueError('phase0 allocation cannot fit all registered row caps, selftest, CPU rows and dispatch; stop: scope')
        verification=prepared.get('verification_seconds')
        if isinstance(verification,bool) or not isinstance(verification,(int,float)) or not math.isfinite(verification) or not 0<=verification<=60:
            raise ValueError('prepared integrity verification measurement is missing or exceeds its 60-second row allowance; stop before submission')
        duration=f'{minutes//60:02d}:{minutes%60:02d}:00'
        from kit.v4_phase0_site import planning_storage
        launch=frozen_launch_environment(work)
        storage=planning_storage(work,launch)
        phase0_binding={'storage':storage,'verification_seconds':verification,'environment_check_deadline_seconds':environment_seconds,'launch_environment':launch,'phase0_allocation_cap_gpu_hours':ALLOCATION_CAP_GPU_HOURS, **{key:hashlib.sha256((work/path).read_bytes()).hexdigest() for key,path in {
            'phase0_prepare_sha256':'v4/report-inputs/prepare-receipt-phase0.json',
            'phase0_presend_sha256':'v4/report-phase0/presend/containment-presend.json'}.items()}}
    projection=admission(work,ledger,stage,minutes,site_minutes) if phase=='main' else None
    prior={job:{key:entry[key] for key in ('start','end','width','segments')} for job,entry in ledger['allocations'].items()} if ledger else {}
    return {**phase0_binding,'prior_allocations':prior,'admission':projection,'schema':'v4-allocation-plan.v1','phase':phase,'stage':stage,'block':registration['block'],
            'block_limit':registration['block_limit'],'ceiling':560,'gpus':8,'sbatch_time':duration,
            'remaining_gpu_hours':float(remaining),'reserved_gpu_hours':8*minutes/60,
            'ledger_sha256':__import__('hashlib').sha256(ledger_path.read_bytes()).hexdigest() if ledger else None,
            'requested_minutes':requested_minutes,'site_minutes':site_minutes,'node_list':node}


def verify_plan(work,phase,first_row):
    """Called AFTER the fresh explicit-block selftest, before runner tracking."""
    if phase=='phase0':
        from kit.v4_phase0 import verify_allocation_plan
        return verify_allocation_plan(work)
    if phase!='main':return
    from kit.v4_budget import stamp
    path=os.environ.get('V4_ALLOCATION_PLAN')
    if not path:raise ValueError('main allocation requires its freshly recomputed admission plan')
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=os.environ.get('V4_ALLOCATION_PLAN_SHA256'):
        raise ValueError('allocation admission plan hash differs from generated script')
    doc=json.loads(raw)
    stage=doc['stage'];registration=build('main')['v4']['allocation_stages'][stage]
    if doc.get('phase')!='main' or registration['first']!=first_row or doc['admission']['status']!='proceed':
        raise ValueError('allocation admission stage/status differs')
    for env,key in (('V4_QUALIFICATION_ARCHIVE','qualification_sha256'),('V4_OWNER_PROCEED','owner_sha256')):
        if hashlib.sha256(Path(os.environ[env]).read_bytes()).hexdigest()!=doc['admission'][key]:
            raise ValueError('allocation admission evidence binding differs')
    ledger=json.loads((Path(work)/'k8b4/containment/allocation-ledger.json').read_text())
    job=os.environ['SLURM_JOB_ID'];entry=ledger['allocations'][job]
    expected=sum(int(part)*scale for part,scale in zip(doc['sbatch_time'].split(':'),(3600,60,1)))
    if (entry['width']!=8 or stamp(entry['planned_end'])-stamp(entry['start'])!=expected
        or entry['segments'][0]['block']!=registration['block']):raise ValueError('actual allocation window/block differs from admission')
    prior={key:{field:value[field] for field in ('start','end','width','segments')}
           for key,value in ledger['allocations'].items() if key!=job}
    if prior!=doc['prior_allocations']:raise ValueError('allocation accounting changed since admission')


def header(doc,work):
    """Site directives to add to the partner's existing container wrapper."""
    site=doc.get('site',{})
    lines=['#!/bin/bash','#SBATCH --no-requeue','#SBATCH --nodes=1','#SBATCH --exclusive',
           '#SBATCH --gpus=8','#SBATCH --time='+doc['sbatch_time']]
    if doc.get('node_list'):lines.append('#SBATCH --nodelist='+doc['node_list'])
    lines.extend('#SBATCH --'+key+'='+site[key] for key in ('partition','qos','account') if key in site)
    if doc['phase']=='phase0':
        logs=Path(work).resolve()/'v4/report-phase0/allocation'
        if any(c.isspace() for c in str(logs)):raise ValueError('Slurm scheduler log path must not contain whitespace')
        lines.extend(('#SBATCH --output='+str(logs/'phase0-%j.out'), '#SBATCH --error='+str(logs/'phase0-%j.err')))
    return '\n'.join(lines)+'\n'


def script(doc,work,kit):
    """Payload executed directly in the confirmed native pilot batch process; check before selftest."""
    work,kit=shlex.quote(str(Path(work).resolve())),shlex.quote(str(Path(kit).resolve()))
    site=doc.get('site',{});launch=doc.get('launch_environment',{})
    if doc['phase']=='phase0' and not launch:raise ValueError('phase0 requires frozen launch environment')
    # The existing host wrapper owns these search paths. Only the prepared trainer's
    # activation may prepend its own bin; never freeze Docker's loader state.
    exports=''.join('export '+key+'='+shlex.quote(value)+'\n' for key,value in sorted(launch.items())
                    if key not in ('PATH','LD_LIBRARY_PATH','LD_PRELOAD','CUDA_HOME'))
    advice=('export V4_ALLOCATION_PLAN='+shlex.quote(doc['plan_path'])+'\nexport V4_ALLOCATION_PLAN_SHA256='+hashlib.sha256((json.dumps(doc,indent=2)+'\n').encode()).hexdigest()+'\n') if doc.get('plan_path') else ''
    activation='source '+shlex.quote(site['activate'])+'\n' if site.get('activate') else ''
    socket_check='python "$KIT/v4_phase0_site.py" socket-check --work "$WORK"\n' if doc['phase']=='phase0' else ''
    environment='python "$KIT/v4_phase0_environment.py" --work "$WORK"\n' if doc['phase']=='phase0' else ''
    cache='CACHE_EXPORTS="$(python "$KIT/v4_phase0_site.py" storage --work "$WORK" --min-gib 30 --allocation)"\neval "$CACHE_EXPORTS"\n' if doc['phase']=='phase0' else ''
    job_tmp='mkdir -p "$TMPDIR" "$RAY_TMPDIR"\n' if doc['phase']=='phase0' else ''
    payload=f'''#!/bin/bash
set -euo pipefail
export WORK={work}
export KIT={kit}
export PYTHONPATH="$KIT/..:${{PYTHONPATH:-}}"
{cache}{exports}export SLURM_EXPORT_ENV=ALL
{advice}{job_tmp}{socket_check}{environment}{activation}python "$KIT/p4_contain.py" selftest --work "$WORK" --out "$WORK/k8b4/containment" --block {doc['block']} --block-limit {doc['block_limit']} --ceiling 560
python "$KIT/runner.py" batch "$KIT/campaigns/v4-{doc['phase']}.yaml" --stage {doc['stage']}
# Returning ends this allocation. Owner review happens only after Slurm ends it.
'''
    return payload if doc['phase']=='phase0' else header(doc,shlex.split(work)[0])+payload.split('\n',1)[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--phase',choices=('qualification','main','phase0'),required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--minutes',type=int,required=True)
    parser.add_argument('--site-minutes',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    for option in ('partition','qos','account','activate'):parser.add_argument('--'+option,required=True)
    args=parser.parse_args(argv)
    outputs=[args.out,args.out.with_suffix('.json')]
    if args.phase=='phase0':outputs.append(args.out.with_suffix('.header.sh'))
    if any(path.exists() for path in outputs):
        parser.error('allocation plan/payload already exists; retain and use the frozen header and payload. Do not rerun planning or overwrite evidence; a setup retry needs a fresh WORK and capture/prepare.')
    try:
        if args.phase=='phase0':
            from kit.v4_phase0_site import storage
            with storage(args.work):doc=plan(args.work,args.phase,args.stage,args.minutes,args.site_minutes)
        else:doc=plan(args.work,args.phase,args.stage,args.minutes,args.site_minutes)
    except (ValueError,OSError) as exc:parser.error(str(exc))
    import re
    for option in ('partition','qos','account'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',getattr(args,option)):parser.error('invalid site directive: '+option)
    if args.phase=='phase0':
        from kit.v4_slurm_capture import verify_capture
        captured=verify_capture(args.work)
        if captured['site']!={name:getattr(args,name) for name in ('partition','qos','account')}:parser.error('site assignment differs from first Slurm capture')
        if captured['site_minutes'] is not None and args.site_minutes!=captured['site_minutes']:parser.error('--site-minutes differs from captured MaxTime')
    if not Path(args.activate).is_absolute() or not Path(args.activate).is_file():parser.error('--activate must name the staged trainer environment activation file')
    if not args.out.resolve().is_relative_to(args.work.resolve()):parser.error('allocation scripts must be retained inside WORK')
    doc['plan_path']=str(args.out.with_suffix('.json').resolve())
    doc['site']={name:getattr(args,name) for name in ('partition','qos','account','activate')}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:handle.write(script(doc,args.work,Path(__file__).resolve().parent))
    with args.out.with_suffix('.json').open('x') as handle:json.dump(doc,handle,indent=2);handle.write('\n')
    if args.phase=='phase0':
        with args.out.with_suffix('.header.sh').open('x') as handle:handle.write(header(doc,args.work))
        print('Add '+str(args.out.with_suffix('.header.sh'))+' to your pilot wrapper; run bash '+shlex.quote(str(args.out))+' directly after its unchanged host-venv environment lines. Submit exactly as k8b_pilot_run.sbatch: sbatch --parsable '+shlex.quote(str(args.out.with_suffix('.sbatch'))))
        return 0
    print('sbatch --no-requeue --account='+args.account+' --qos='+args.qos+' --time='+doc['sbatch_time']+' '+shlex.quote(str(args.out)))
    return 0
if __name__=='__main__':raise SystemExit(main())
