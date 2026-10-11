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
from kit.v4_phase0_timing import scale
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
            for task in ('finqa',):
                if role+'-'+task in passed:
                    for op in ('generate','verify','corpus_reload','serial_merge'):
                        key=f'{op}:{role}:{task}';done[key]=units[key]
        for job in slots():
            ident=job['id'];arm=job['arm'];task=job['task']
            if 'schedule-'+ident in passed:add('cpu:schedule:'+task,1)
            if 'train-'+ident in passed:
                add(f'train:{arm}:{task}',40);add(f'startup:{arm}:{task}',1);add(f'export:{arm}:{task}',2);add(f'merge:{arm}:{task}',1)
            if 'merge-'+ident in passed:add(f'merge:{arm}:{task}',1)
            for panel,n in (('finqa',1147),('general-v1',300)):
                for cap in (2048,8192):
                    if f'score-{ident}-{panel}-{cap}' in passed:
                        add(f'eval:{panel}:{cap}',n);add(f'reload:{panel}:{cap}',1)
            for panel in ('finqa',):
                for step in (0,20,40):
                    if f'probe-{ident}-{panel}-{step}' in passed:
                        add(f'probe:{panel}:2048',160);add('probe_reload:'+panel,1)
        for task in ('finqa',):
            if 'conditioned-probe-'+task in passed:
                done['conditioned_probe:'+task]=160;done['conditioned_reload:'+task]=1
        for operation in ('coverage','training-set','gate','report','archive-pack'):
            ident='model-identity-check' if operation=='gate' else operation
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
    from kit.v4_finqa_policy import warn
    warn('budget projection','measurement-based projection is advisory; main starts automatically',work)
    return {'status':'proceed','owner_proceed_required':False,'warnings':['measurement-based projections are advisory']}


def plan(work, phase, stage, requested_minutes, site_minutes):
    """Choose whole minutes conservatively, retaining the full eight-GPU reservation."""
    work=Path(work).resolve()
    registration=build(phase)['v4']['allocation_stages'][stage]
    ledger_path=work/'k8b4/containment/allocation-ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    if phase in ('qualification','main'):
        from kit.v4_phase0_carry import verify_work
        ledger=verify_work(work)
        from kit.v4_finqa_submission import require_closed_submissions
        from kit.v4_finqa_policy import observe
        observe(require_closed_submissions,work,ledger)
    main_binding={}
    if phase in ('qualification','main'):
        from kit.v4_finqa_site import frozen_environment,planning_storage
        launch=frozen_environment(work)
        main_binding={'storage':planning_storage(work,launch),'launch_environment':launch}
    if ledger:
        if any(not entry.get('end') for entry in ledger['allocations'].values()):
            if phase=='phase0':raise ValueError('close and reconcile every allocation before planning its replacement')
            from kit.v4_finqa_policy import warn
            warn('allocation accounting','prior allocation not yet closed; recording current exposure',work)
        if phase=='phase0':spent,blocks=allocation_spend(ledger)
        else:
            from kit.v4_finqa_policy import known_spend
            spent,blocks=known_spend(ledger,work)
        limits={'qualification':1000,'scientific':560 if phase=='phase0' else 1000}
        if phase!='phase0' and any(block not in limits or charge>limits[block] for block,charge in blocks.items()):
            from kit.v4_finqa_policy import warn
            warn('allocation accounting','planned block overrun or unknown block',work)
    else:spent,blocks=number(0,'initial spend'),{}
    for value in (requested_minutes,site_minutes):
        if type(value) is not int or value<=0:raise ValueError('duration must be positive whole minutes')
    remaining=(min(number(registration['block_limit'],'limit')-blocks.get(registration['block'],0),number(560,'ceiling')-spent) if phase=='phase0' else number(1000,'ceiling')-spent)
    minutes=1440 if phase=='phase0' else min(requested_minutes,site_minutes)
    if phase!='phase0' and remaining<=0:raise ValueError('actual 1000 GPU-hour ceiling reached')
    # Ten-minute selftest plus at least the registered minimum row cap.
    if phase!='phase0' and minutes<15:
        from kit.v4_finqa_policy import warn
        warn('allocation window','short site window; may need continuation',work)
    duration=f'{minutes//60:02d}:{minutes%60:02d}:00'
    receipt_path=work/'k8b4/containment/containment-receipt.json'
    node=None
    if receipt_path.exists():
        from kit.p4_frozen import seal
        receipt=json.loads(receipt_path.read_text())
        if receipt!=seal(receipt) or (phase!='phase0' and not receipt.get('ok')):
            from kit.v4_finqa_policy import warn
            warn('containment receipt','receipt changed or failed',work)
        node=receipt['node_list']
        if not __import__('re').fullmatch(r'[A-Za-z0-9_.\[\],-]+',node):raise ValueError('unknown allocation node')
    phase0_binding={}
    if phase=='phase0':
        from kit.v4_phase0 import verify_presend, verify_prepare, required_seconds, allocation_cap, ALLOCATION_CAP_GPU_HOURS, frozen_launch_environment, verification_allowance
        prepared=verify_prepare(work,rehash=False)
        environment_seconds=prepared.get('environment_check',{}).get('deadline_seconds',scale(600))
        if requested_minutes*8/60 > ALLOCATION_CAP_GPU_HOURS:
            raise ValueError('phase0 allocation cap of 192 GPU-hours refuses this request')
        if ledger:allocation_cap(ledger)
        if ledger and any(e.get('phase0') for e in ledger['allocations'].values()):
            raise ValueError('phase0 admits exactly one allocation; return partial evidence for review')
        verification=prepared.get('verification_seconds')
        allowance=verification_allowance(verification)
        if prepared.get('verification_allowance_seconds',allowance)!=allowance:
            raise ValueError('prepared verification allowance differs from CPU measurement')
        required=required_seconds(environment_seconds,allowance)
        duration=f'{minutes//60:02d}:{minutes%60:02d}:00'
        from kit.v4_phase0_site import planning_storage
        launch=frozen_launch_environment(work)
        storage=planning_storage(work,launch)
        phase0_binding={'storage':storage,'verification_seconds':verification,'verification_allowance_seconds':allowance,'required_seconds':required,'worst_case_seconds':required,'expected_gpu_hours':15,'reservation_slack_seconds':minutes*60-required,'environment_check_deadline_seconds':environment_seconds,'launch_environment':launch,'phase0_allocation_cap_gpu_hours':ALLOCATION_CAP_GPU_HOURS, **{key:hashlib.sha256((work/path).read_bytes()).hexdigest() for key,path in {
            'phase0_prepare_sha256':'v4/report-inputs/prepare-receipt-phase0.json',
            'phase0_presend_sha256':'v4/report-phase0/presend/containment-presend.json'}.items()}}
    elif phase in ('qualification','main'):
        from kit.v4_prepare import measured_verification,main_verification_allowance as verification_allowance
        receipt=work/'v4/report-inputs'/('prepare-receipt-main.json' if phase=='main' else 'prepare-receipt.json')
        if receipt.exists():
            prepared=measured_verification(work,phase)
            allowance=verification_allowance(prepared['verification_seconds'])
            row=next(r for r in build(phase)['rows'] if r['id']=='prepare')
            cap=max(row['allocation_cpu_cap_seconds'],allowance)
            if allowance>minutes*60:
                from kit.v4_finqa_policy import warn
                warn('verification projection',str(allowance)+' seconds exceeds allocation window; continuing',work)
            main_binding.update(verification_seconds=prepared['verification_seconds'],verification_allowance_seconds=allowance,
                verification_row_cap_seconds=cap,verification_row_slack_seconds=cap-allowance,
                prepare_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    projection=admission(work,ledger,stage,minutes,site_minutes) if phase=='main' else None
    from kit.v4_phase0_carry import accounting_allocations
    prior={job:{key:entry[key] for key in ('start','end','width','segments')} for job,entry in accounting_allocations(ledger).items()} if ledger else {}
    if phase in ('qualification','main'):
        main_binding.update(phase0_import_binding={k:v for k,v in ledger.get('phase0_import',{}).items() if k!='allocations'},
                            remaining_qualification_gpu_hours=float(1000-spent),
                            remaining_ceiling_gpu_hours=float(1000-spent))
    return {**phase0_binding,**main_binding,'prior_allocations':prior,'admission':projection,'schema':'v4-allocation-plan.v1','phase':phase,'stage':stage,'block':registration['block'],
            'block_limit':registration['block_limit'],'ceiling':560 if phase=='phase0' else 1000,'gpus':8,'sbatch_time':duration,
            'remaining_gpu_hours':float(remaining),'reserved_gpu_hours':8*minutes/60,
            'ledger_sha256':__import__('hashlib').sha256(ledger_path.read_bytes()).hexdigest() if ledger else None,
            'requested_minutes':requested_minutes,'site_minutes':site_minutes,'node_list':node}


def _verify_plan(work,phase,first_row):
    """Called AFTER the fresh explicit-block selftest, before runner tracking."""
    if phase=='phase0':
        from kit.v4_phase0 import verify_allocation_plan
        return verify_allocation_plan(work)
    if phase not in ('qualification','main'):return
    from kit.v4_phase0_carry import verify_work, accounting_allocations, public_binding
    ledger=verify_work(work)
    from kit.v4_budget import stamp
    path=os.environ.get('V4_ALLOCATION_PLAN')
    if not path:raise ValueError('main-package allocation requires its freshly recomputed admission plan')
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=os.environ.get('V4_ALLOCATION_PLAN_SHA256'):
        raise ValueError('allocation admission plan hash differs from generated script')
    doc=json.loads(raw)
    stage=doc['stage'];registration=build(phase)['v4']['allocation_stages'][stage]
    if doc.get('phase')!=phase or registration['first']!=first_row or (phase=='main' and doc['admission']['status']!='proceed'):
        raise ValueError('allocation admission stage/status differs')
    if 'verification_allowance_seconds' not in doc:
        raise ValueError('allocation admission requires prepared verification allowance')
    from kit.v4_prepare import main_verification_allowance as verification_allowance
    receipt_path=Path(work)/'v4/report-inputs'/('prepare-receipt-main.json' if phase=='main' else 'prepare-receipt.json')
    receipt=json.loads(receipt_path.read_text())
    allowance=verification_allowance(receipt.get('verification_seconds'))
    cap=max(next(r for r in build(phase)['rows'] if r['id']=='prepare')['allocation_cpu_cap_seconds'],allowance)
    if (hashlib.sha256(receipt_path.read_bytes()).hexdigest()!=doc.get('prepare_sha256')
        or receipt.get('verification_allowance_seconds')!=allowance
        or doc.get('verification_seconds')!=receipt['verification_seconds']
        or doc.get('verification_allowance_seconds')!=allowance
        or doc.get('verification_row_cap_seconds')!=cap or doc.get('verification_row_slack_seconds')!=cap-allowance):
        raise ValueError('prepared verification admission binding/allowance differs')
    if doc.get('phase0_import_binding')!=public_binding(ledger['phase0_import']):raise ValueError('phase0 allocation plan binding differs')
    ledger=json.loads((Path(work)/'k8b4/containment/allocation-ledger.json').read_text())
    job=os.environ['SLURM_JOB_ID'];entry=ledger['allocations'][job]
    expected=sum(int(part)*scale for part,scale in zip(doc['sbatch_time'].split(':'),(3600,60,1)))
    if (entry['width']!=8 or stamp(entry['planned_end'])-stamp(entry['start'])!=expected
        or entry['segments'][0]['block']!=registration['block']):raise ValueError('actual allocation window/block differs from admission')
    prior={key:{field:value[field] for field in ('start','end','width','segments')}
           for key,value in accounting_allocations(ledger).items() if key!=job}
    if prior!=doc['prior_allocations']:raise ValueError('allocation accounting changed since admission')


def verify_plan(work,phase,first_row):
    if phase=='phase0':return _verify_plan(work,phase,first_row)
    from kit.v4_finqa_policy import observations,observe
    with observations(work):return observe(_verify_plan,work,phase,first_row,site='allocation plan')


def header(doc,work):
    """Site directives to add to the partner's existing native host wrapper."""
    site=doc.get('site',{})
    lines=['#!/bin/bash','#SBATCH --no-requeue','#SBATCH --nodes=1','#SBATCH --exclusive',
           '#SBATCH --gpus=8','#SBATCH --time='+doc['sbatch_time']]
    if doc.get('node_list'):lines.append('#SBATCH --nodelist='+doc['node_list'])
    lines.extend('#SBATCH --'+key+'='+site[key] for key in ('partition','qos','account') if key in site)
    if doc['phase']=='phase0':
        logs=Path(work).resolve()/'v4/report-phase0/allocation'
        if any(c.isspace() for c in str(logs)):raise ValueError('Slurm scheduler log path must not contain whitespace')
        lines.extend(('#SBATCH --output='+str(logs/'phase0-%j.out'), '#SBATCH --error='+str(logs/'phase0-%j.err')))
    if doc['phase'] in ('qualification','main'):
        logs=Path(work).resolve()/('v4/report-'+doc['phase'])/'allocation'
        if any(c.isspace() for c in str(logs)):raise ValueError('Slurm scheduler log path must not contain whitespace')
        logs.mkdir(parents=True,exist_ok=True)
        lines.extend(('#SBATCH --output='+str(logs/(doc['stage']+'-%j.out')),
                      '#SBATCH --error='+str(logs/(doc['stage']+'-%j.err'))))
    return '\n'.join(lines)+'\n'


def script(doc,work,kit):
    """Payload executed directly in the confirmed native pilot batch process; check before selftest."""
    work_path=Path(work).resolve();kit_path=Path(kit).resolve()
    work,kit=shlex.quote(str(Path(work).resolve())),shlex.quote(str(Path(kit).resolve()))
    site=doc.get('site',{});launch=dict(doc.get('launch_environment',{}))
    if doc['phase'] in ('qualification','main'):
        launch.update(KIT=str(kit_path),PYTHONPATH=str(kit_path.parent)+os.pathsep+launch.get('SDPO_DIR',''),V4_PHASE0_MODE='0',V4_FINQA_CAMPAIGN='1')
    if doc['phase']=='phase0' and not launch:raise ValueError('phase0 requires frozen launch environment')
    # The existing host wrapper owns these search paths. Only the prepared trainer's
    # activation may prepend its own bin; never freeze Docker's loader state.
    exports=''.join('export '+key+'='+shlex.quote(value)+'\n' for key,value in sorted(launch.items())
                    if key not in ('PATH','LD_LIBRARY_PATH','LD_PRELOAD','CUDA_HOME'))
    advice=('export V4_ALLOCATION_PLAN='+shlex.quote(doc['plan_path'])+'\nexport V4_ALLOCATION_PLAN_SHA256='+hashlib.sha256((json.dumps(doc,indent=2)+'\n').encode()).hexdigest()+'\n') if doc.get('plan_path') else ''
    activation='source '+shlex.quote(site['activate'])+'\n' if site.get('activate') else ''
    socket_check='python "$KIT/v4_phase0_site.py" socket-check --work "$WORK"\n' if doc['phase']=='phase0' else ''
    environment='python "$KIT/v4_phase0_environment.py" --work "$WORK"\n' if doc['phase']=='phase0' else ''
    carry='python "$KIT/v4_phase0_carry.py" --work "$WORK" --verify\n' if doc['phase'] in ('qualification','main') else ''
    cache='CACHE_EXPORTS="$(python "$KIT/v4_phase0_site.py" storage --work "$WORK" --min-gib 30 --allocation)"\neval "$CACHE_EXPORTS"\n' if doc['phase']=='phase0' else ''
    job_tmp='mkdir -p "$TMPDIR" "$RAY_TMPDIR"\n' if doc['phase']=='phase0' else ''
    if doc['phase'] in ('qualification','main'):
        cache='CACHE_EXPORTS="$(python "$KIT/v4_finqa_site.py" storage --work "$WORK" --allocation)"\neval "$CACHE_EXPORTS"\n'
        job_tmp='mkdir -p "$TMPDIR" "$RAY_TMPDIR"\n'
        socket_check='python "$KIT/v4_finqa_site.py" check --work "$WORK"\n'
    dispatch=(f'python "$KIT/v4_finqa_flow.py" --work "$WORK"' if doc['phase'] in ('qualification','main') else f'python "$KIT/runner.py" batch "$KIT/campaigns/v4-phase0.yaml" --stage {doc["stage"]}')
    if doc['phase'] in ('qualification','main'):
        advice+='export V4_FLOW_WRAPPER='+shlex.quote(str(Path(doc.get('plan_path',work_path/'flow.json')).with_suffix('.sbatch')))+'\n'
    ending=('# Returning ends this allocation. Owner review happens only after Slurm ends it.' if doc['phase']=='phase0' else '# Flow follows qualification automatically and chains a continuation if this site window ends.')
    payload=f'''#!/bin/bash
set -euo pipefail
export V4_PHASE0_MODE={'1' if doc['phase']=='phase0' else '0'}
export WORK={work}
export KIT={kit}
export PYTHONPATH="$KIT/..:${{PYTHONPATH:-}}"
{cache}{exports}export SLURM_EXPORT_ENV=ALL
{advice}{job_tmp}{socket_check}{environment}{activation}{carry}python "$KIT/p4_contain.py" selftest --work "$WORK" --out "$WORK/k8b4/containment" --block {doc['block']} --block-limit {doc['block_limit']} --ceiling {doc.get('ceiling',560 if doc['phase']=='phase0' else 1000)}
{dispatch}
{ending}
'''
    return payload


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
    outputs.append(args.out.with_suffix('.header.sh'))
    if any(path.exists() for path in outputs):
        if args.phase=='phase0':parser.error('allocation plan/payload already exists; retain and use the frozen header and payload. Do not rerun planning or overwrite evidence; a setup retry needs a fresh WORK and capture/prepare.')
        from kit.v4_finqa_policy import warn
        original=args.out;number=2
        while any(original.with_name(original.stem+'-attempt-'+str(number)+original.suffix).with_suffix(suffix).exists() for suffix in (original.suffix,'.json','.header.sh','.sbatch')):number+=1
        args.out=original.with_name(original.stem+'-attempt-'+str(number)+original.suffix)
        warn('allocation planning retry','prior plan retained; new plan in SAME WORK: '+str(args.out),args.work)
    try:
        if args.phase=='phase0':
            from kit.v4_phase0_site import storage
            with storage(args.work):doc=plan(args.work,args.phase,args.stage,args.minutes,args.site_minutes)
        else:
            from kit.v4_finqa_site import storage
            with storage(args.work):doc=plan(args.work,args.phase,args.stage,args.minutes,args.site_minutes)
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
    if args.phase in ('qualification','main'):
        frozen=json.loads((args.work/'v4/report-inputs/launch-inputs-finqa.json').read_text())
        if args.activate!=frozen['trainer_activate']:
            from kit.v4_finqa_policy import warn
            warn('trainer activation','activation differs from retained phase0 trainer; local infrastructure change recorded',args.work)
        if args.phase=='main':
            for key in ('V4_KIT_TAG',):
                doc['launch_environment'][key]=os.environ.get(key,doc['launch_environment'].get(key,'local-untagged'))
    doc['plan_path']=str(args.out.with_suffix('.json').resolve())
    doc['site']={name:getattr(args,name) for name in ('partition','qos','account','activate')}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:handle.write(script(doc,args.work,Path(__file__).resolve().parent))
    with args.out.with_suffix('.json').open('x') as handle:json.dump(doc,handle,indent=2);handle.write('\n')
    if args.phase in ('phase0','qualification','main'):
        with args.out.with_suffix('.header.sh').open('x') as handle:handle.write(header(doc,args.work))
        print('Add '+str(args.out.with_suffix('.header.sh'))+' to your pilot wrapper; run bash '+shlex.quote(str(args.out))+' directly after its unchanged host-venv environment lines. Submit exactly as k8b_pilot_run.sbatch: sbatch --parsable '+shlex.quote(str(args.out.with_suffix('.sbatch'))))
        return 0
    print('sbatch --no-requeue --account='+args.account+' --qos='+args.qos+' --time='+doc['sbatch_time']+' '+shlex.quote(str(args.out)))
    return 0
if __name__=='__main__':raise SystemExit(main())
