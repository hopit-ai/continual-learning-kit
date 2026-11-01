#!/usr/bin/env python3
"""CPU-only X rehearsal. All checkpoints and P2 calibration are labelled stand-ins."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *
from types import SimpleNamespace as NS
from unittest.mock import patch
import contextlib
import shutil
import time


def synthetic_screen_return(path):
    """A labelled synthetic return analyzed by the current frozen S script."""
    import tarfile,io
    from kit.s_common import SEEDS as SCREEN_SEEDS
    def cells(forward=False):
        return [dict(tier=tier,recipe=recipe,seed=seed,step=step,
            stage='base-to-finqa' if tier==1 else 'finqa-to-second-task',
            domains={d:dict(audit_passed=True,monitor_n=64,train_n=128,
                drift=i,forward=i,plasticity=i,start_nll=-i,loss=i)
                for i,d in enumerate(DOMAINS[:3] if tier==1 else DOMAINS)})
            for tier in (1,2) for recipe in ('D','F') for seed in SCREEN_SEEDS for step in ((40,) if forward else (20,40))]
    evidence={'screen-s/measurements.json':json.dumps(dict(technical_synthetic=True,
        association=cells(),forward=cells(True))).encode(),'screen-s/registration.json':RESOURCE.read_bytes()}
    evidence['screen-s/manifest.json']=json.dumps(dict(technical_synthetic=True,
        files={k:hashlib.sha256(v).hexdigest() for k,v in evidence.items()})).encode()
    with tarfile.open(path,'w:gz') as archive:
        for name,raw in evidence.items():
            member=tarfile.TarInfo(name);member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
    return path


def size_forecast(tier2=False):
    """Registered cohorts, twice text plus metadata reserve scaled by run count.

    Actual registered question IDs and variable output fields avoid forecasting
    from an artificially repeated single CSV row. No generated responses are returned.
    """
    import csv,gzip,io,random
    from kit.m2_score import items_for
    from kit.c_score import COLUMNS
    from collections import Counter
    panels=dict(test=len(items_for('test',True)),math_hard=len(items_for('math_hard',True)),
        finqa=1147,hellaswag=100,**Counter(r['panel'] for r in items_for('general-v1',True)))
    require(panels == dict(test=100,math_hard=100,finqa=1147,hellaswag=100,ifeval=100,math=100,knowledge=100),
        'registered panel sizes changed; reforecast before SEND')
    registered={p:items_for(p,True) for p in ('test','math_hard','finqa')}
    registered.update({p:[r for r in items_for('general-v1',True) if r['panel']==p] for p in ('ifeval','math','knowledge')})
    registered['hellaswag']=[dict(id=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(100)]
    rng=random.Random(20261025)
    from kit.x_tier2 import ARMS2
    arms=(*ARMS,*ARMS2) if tier2 else ARMS
    def compressed(inherited=False):
        output=io.StringIO();writer=csv.DictWriter(output,fieldnames=COLUMNS);writer.writeheader();count=0
        for seed in SEEDS:
            for arm in (('base','finqa-sibling','standalone-legal-plain') if inherited else ('pre',*arms)):
                for step in ((0,) if arm in ('base','pre') else STEPS):
                    for panel,n in panels.items():
                        if panel!='test' and step not in (0,20,40):continue
                        for i in range(n):
                            item=registered[panel][i];unique=item['id']
                            row=dict(arm=arm,seed=seed,step=step,task=panel,panel=panel,cap=2048,question_id=unique,
                                correct=rng.randrange(2),output_tokens=rng.randrange(2049),group=item.get('group',''),gold=item.get('gold',''))
                            writer.writerow({k:row.get(k,'') for k in COLUMNS});count+=1
        raw=output.getvalue().encode()
        # Inherited JSON is intentionally forecast in its actual representation.
        if inherited: raw=json.dumps(dict(rows=list(csv.DictReader(io.StringIO(raw.decode()))))).encode()
        return len(gzip.compress(raw,mtime=0)),count
    scores,n=compressed();inherited,old_n=compressed(True)
    run_count=80 if tier2 else 20
    reserve=8*1024**2+max(0,run_count-20)*64*1024
    forecast=2*(scores+inherited)+reserve
    from kit.x_collect import LIMIT
    require(forecast < .8*LIMIT,'return forecast lacks 20% margin; stop before training')
    return dict(panel_sizes=panels,score_rows=n,inherited_rows=old_n,compressed_scores_bytes=scores,tier2=tier2,
        compressed_inherited_bytes=inherited,metadata_reserve_bytes=reserve,maximum_runs=run_count,
        additional_metadata_per_run_bytes=64*1024,text_multiplier=2,
        forecast_bytes=forecast,limit_bytes=LIMIT,margin_fraction=1-forecast/LIMIT)


def fixtures(root, configuration='both', admission='COMPLETE', hard='INCOMPLETE', short_domain=None, short_part=None):
    from kit import screen_s_dryrun as sd, s_bind, s_audit, s_freeze
    from kit.message_handoff import save
    sources, spec_path, unused = sd.fixtures(root/'s-inputs')
    if short_domain:
        source_doc = read(sources)
        ref = source_doc[short_domain]
        values = sorted(rows(ref['path']), key=lambda r:(hashlib.sha256(('screen-s-v1|'+short_domain+'|'+r['id']).encode()).hexdigest(),r['id']))
        for i,row in enumerate(values,1):
            if short_part is None or (short_part == 'monitor') == (i%3 == 0):
                row['fake_response'] = ',' if short_domain == 'ifeval' else 'incorrect'
        Path(ref['path']).unlink(); write_rows(ref['path'],values)
        ref['sha256'] = sha(ref['path']); atomic(sources,source_doc)
    spec = read(spec_path)
    # Screen fixture's deliberate dev exposure is removed in THIS synthetic
    # cohort so all four domains can be exercised; no S code/registration edits.
    for node in spec['nodes'].values():
        proof = read(checked(node['exposure']))
        proof['pools'] = [p for p in proof['pools'] if p['role'] != 'v4-finqa-dev']
        atomic(node['exposure']['path'], proof)
        node['exposure']['sha256'] = sha(node['exposure']['path'])
    atomic(spec_path, spec)
    core = root/'m-out'; durable = root/'m-durable'; screen = root/'s-out'
    starts = {f'finqa-sibling:{s}': dict(model=f'technical-b-D+replay-s{s}',
              identity=dict(technical_synthetic=True, model=f'technical-b-D+replay-s{s}')) for s in SEEDS}
    starts['base'] = dict(model='technical-base',identity=dict(technical_synthetic=True,model='technical-base'))
    message = dict(starts=starts, protocol=m.protocol(), technical_synthetic=True, hard_math=hard)
    atomic(core/'binding.json', message)
    atomic(core/'admission.json', dict(status=admission, configuration=configuration,
        binding_sha256=sha(core/'binding.json'), technical_synthetic=True))
    m.save_status(core, message)
    from kit.m2_handoff import scoring_identity
    scoring_identity(core,'0',True)
    if hard != 'QUALIFIES':
        (core/'logs').mkdir()
        (core/'logs/m2-run-math-sibling.log').write_text('SKIP: stage 3 '+hard+'\n')
    assets = {}
    for name in ('c_targets','replay','pools'):
        path = root/name; path.mkdir(parents=True)
        assets[name] = dict(path=str(path.resolve()))
    target = root/'c_targets'
    task = [dict(id=f'contract-training-{i}', prompt=hashlib.sha256(f'contract-training-{i}'.encode()).hexdigest(),
        response='Reasoning. Answer: entailment', gold='entailment') for i in range(8)]
    write_rows(target/'pool.jsonl', task)
    for seed in SEEDS: write_rows(target/f'schedule-{seed}.jsonl', [task[i%8] for i in range(1280)])
    names = {'c_targets':['pool.jsonl']+[f'schedule-{s}.jsonl' for s in SEEDS], 'replay':['replay.jsonl'],
        'pools':['earlier.jsonl','gsm8k.jsonl','arc.jsonl']}
    for folder in ('replay','pools'):
        for name in names[folder]:
            write_rows(root/folder/name, [dict(id=f'x-{name}-{i}', prompt=hashlib.sha256(f'x-{name}-{i}'.encode()).hexdigest(),
                response='Answer: 7') for i in range(8)])
    for name in assets:
        path = root/name
        atomic(path/'manifest.json', m.files_manifest(path, names[name], technical_synthetic=True))
        assets[name]['manifest'] = sha(path/'manifest.json')
    atomic(core/'assets.json', dict(inputs=assets, technical_synthetic=True))
    # Frozen transfer panel uses the existing preparation path.
    from kit.next_hellaswag import prepare
    source = root/'hs-source.jsonl'
    write_rows(source, [dict(ind=i, source_id=f'activitynet~{i}',ctx=f'Synthetic action {i}',endings=['one','two','three','four'],label='0') for i in range(120)])
    prepare(NS(out=core, standin=True, rows=source))
    # Core's actual arm names, complete registered cohorts, no invented
    # standalone-legal-<admission> comparator.
    from kit.m2_score import items_for
    from kit.c_score import legal_scores,COLUMNS
    from kit.m2_sim import responses
    from kit.p3b_score import write_csv
    for arm in ('base','standalone-legal-plain'):
        for seed in ((101,) if arm=='base' else SEEDS):
            for step in ((0,) if arm=='base' else STEPS):
                for panel in (m.PANEL_SETS if step in (0,20,40) else ('test',)):
                    items=items_for(panel,True);raw=responses(items,arm,step,panel)
                    if panel=='test': values=legal_scores(items,raw,arm,seed,step,panel)
                    else: values=[dict(arm=arm,seed=seed,step=step,task=panel,panel=r.get('panel',panel),
                        cap=2048,question_id=r['id'],correct=v['standin_correct'],incorrect_format=0) for r,v in zip(items,raw)]
                    path=core/'scores'/f'{arm}-s{seed}-k{step}-{panel}'
                    write_csv(path/'scores.csv',[{k:r.get(k,'') for k in COLUMNS} for r in values],COLUMNS)
                    atomic(path/'manifest.json',m.files_manifest(path,['scores.csv']))
    save(core, durable)
    s_bind.job(NS(out=screen, standin=True, message2=core, sources=sources, spec=spec_path))
    s_audit.job(NS(out=screen, standin=True)); s_freeze.job(NS(out=screen, standin=True))
    utility = root/'finqa-training.jsonl'
    write_rows(utility, [dict(id=f'utility-finqa-{i}', prompt=hashlib.sha256(f'utility-finqa-{i}'.encode()).hexdigest(), response='Answer: 7') for i in range(256)])
    proof_pools = []
    for role in ('b-ifeval-candidates','broad-gsm8k-candidates'):
        path = root/(role+'.jsonl'); write_rows(path,[dict(id=role,prompt=hashlib.sha256(role.encode()).hexdigest())])
        proof_pools.append(dict(role=role,path=str(path),sha256=sha(path)))
    proof_pools.append(dict(role='b-finqa-training',path=str(utility),sha256=sha(utility)))
    incomings = {}
    for seed in SEEDS:
        ref = dict(starts[f'finqa-sibling:{seed}'], technical_synthetic=True, identity=str(seed))
        path = root/f'b-s{seed}-exposure.json'
        atomic(path, dict(complete=True,checkpoint_identity=digest(ref),parent_identity=m.checked_model(starts['base'],True),pools=proof_pools))
        incomings[str(seed)] = dict(path=str(path),sha256=sha(path))
    addon_spec = root/'x-spec.json'
    atomic(addon_spec, dict(p_variant='p2-l2-init',incomings=incomings))
    smoke_path = root/'smoke.json'
    from kit.x_p import SMOKE_CHECKS
    atomic(smoke_path, dict(schema='rung-p2-distributed-smoke.v2',placement='real-verl-fsdp2',world_size=2,model_revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',precision='fp32-masters/fp64-accumulation',variant='p2-l2-init',technical_synthetic=True,passed=True,
        settings_sha256=sha(ROOT/'kit/resources/p/smoke-settings.json'),
        code_sha256=__import__('kit.p2_smoke',fromlist=['code_hashes']).code_hashes(),
        assertions={k:dict(status='PASS') for k in SMOKE_CHECKS}))
    return NS(out=root/'x-out', durable=root/'x-durable', message2=core, message2_durable=durable,
        screen_s=screen, screen_s_durable=root/'s-durable', spec=addon_spec, standin=True), smoke_path


def branch(root, configuration, requeue=False, tier2=False):
    from kit import x_bind, x_support, x_run, x_train, x_read, x_price, x_collect
    from kit import message_handoff as handoff
    from kit.next_dryrun import invoke
    import argparse
    bound, smoke_path = fixtures(root/'fixtures', configuration)
    out, durable = bound.out, bound.durable
    counts = {}; core_hash = sha(bound.message2/'admission.json')
    common = dict(out=out,standin=True)
    def resumed():
        with patch.dict(os.environ, SLURM_JOB_ID='addon-standin', SLURM_RESTART_COUNT=str(sum(counts.values())), SLURM_JOB_START_TIME=str(time.time()-1)):
            handoff.record_job_start(durable, True)
        handoff.save(out,durable)
        before = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
        shutil.rmtree(out); handoff.restore(durable,out)
        require(all(sha(out/k) == v for k,v in before.items()), 'X requeue changed bytes')
    def row(name, func, args):
        require(shutil.disk_usage(out if out.exists() else out.parent).free >= 12*1024**3, 'disk below 12 GiB; stop')
        func(args)
        if requeue:
            # Interruption at the row's durable boundary, then identical requeue.
            resumed(); func(args)
        counts[name] = counts.get(name,0)+1
    row('X1-bind',x_bind.job,bound)
    row('X2-audit',x_support.job,NS(**common,action='audit'))
    row('X3-own-support',x_support.job,NS(**common,action='generate',gpu='0'))
    def dispatch(cmd, **kw):
        name = Path(cmd[1]).stem
        module = __import__('kit.'+name,fromlist=['job'])
        parser = argparse.ArgumentParser()
        parser.add_argument('--out',type=Path,required=True); parser.add_argument('--standin',action='store_true')
        module.configure(parser)
        module.job(parser.parse_args(cmd[2:]))
        return NS(returncode=0)
    args = dict(**common,gpus='0,1,2,3,4,5,6,7',score_gpu='0',sdpo=None,tokenizer=None,panel_root=None,
        smoke=smoke_path,trainer_python=sys.executable,inference_python=sys.executable)
    # x_run's subprocesses remain isolated in production. Route kit rows only
    # inside this labelled CPU fixture; real X4 uses the eight-GPU D path.
    real_run = __import__('subprocess').run
    def route(cmd, **kw):
        if len(cmd)>1 and Path(cmd[1]).stem.startswith('x_'): return dispatch(cmd,**kw)
        return real_run(cmd,**kw)
    from kit.m2_sim import responses as original_responses
    def passing_p(items,arm,step,panel):
        values=original_responses(items,arm,step,panel)
        if panel=='test' and arm.startswith('x-'):
            fraction=.5+(.4 if arm=='x-p' else .25)*step/40
            for i,(item,value) in enumerate(zip(items,values)):
                value['response']='Answer: '+(item['gold'] if i<round(len(items)*fraction) else 'invalid')
        return values
    with patch('subprocess.run',side_effect=route), patch('kit.m2_sim.responses',side_effect=passing_p):
        for name, action in [('X4-P2-measure','measure'), ('X5-controls','controls'), ('X6-P','p'),
                             ('X7-drift-prefix','drift-prefix'), ('X8-drift-continue','drift-continue'), ('X9-plasticity','plasticity')]:
            row(name,x_run.job,NS(**args,action=action))
    row('X10-tables',x_read.job,NS(**common,action='tables',bootstrap=20))
    row('X11-price',x_price.job,NS(**common,timings=None))
    archive = root/'addon-return.tar.gz'
    row('X12-collect',x_collect.job,NS(**common,archive=archive))
    returned = x_collect.load(archive,True)
    require(json.loads(returned['comparison-tables.json']) == read(out/'comparison-tables.json'), 'archive table readback changed')
    from kit.x_qualify import qualify
    screen_return=synthetic_screen_return(root/'screen-s-return.tar.gz')
    qpath = root/'qualification.json'; atomic(qpath,qualify(screen_return,True))
    row('X13-manager-read',x_read.job,NS(out=root/'reading',standin=True,action='read',source=archive,
        screen_qualification=qpath,screen_return=screen_return,bootstrap=20))
    # Owner ruling 1 Nov: this native-score fixture has no reference guard units.
    # Mechanical P2 qualification remains checked; scientific admission is pending.
    require(read(root/'reading/reading.json')['admitted']['p'] is None and
        read(out/'comparison-tables.json')['comparisons']['p'].get('reason') ==
        'REFERENCE-SCORED required for decision checkpoint guards', 'kit-only P guards must remain PENDING at X13')
    tier2_gate_cases=[]
    if tier2:
        from kit import x_tier2
        row('X14-tier2-plan',x_tier2.job,NS(**args,action='plan'))
        with patch('subprocess.run',side_effect=route), patch('kit.m2_sim.responses',side_effect=passing_p):
            for number,name in enumerate(x_tier2.COMPARISONS,15):
                row(f'X{number}-{name}',x_tier2.job,NS(**args,action=name))
        row('X21-tables',x_read.job,NS(**common,action='tables',bootstrap=20))
        row('X22-price',x_price.job,NS(**common,timings=None))
        row('X23-collect',x_collect.job,NS(**common,archive=archive))
        row('X24-manager-read',x_read.job,NS(out=root/'reading',standin=True,action='read',source=archive,
            screen_qualification=qpath,screen_return=screen_return,bootstrap=20))
        require(read(root/'reading/reading.json')['admitted']['p'] is None,'tier 2 changed pending tier-1 reference guards')
    import gzip
    final=x_collect.load(archive,True)
    metadata_bytes=len(gzip.compress(b''.join(raw for name,raw in sorted(final.items())
        if name not in ('scores.csv.gz','inherited.json.gz')),mtime=0))
    if tier2:
        import csv,io
        from kit.x_tier2_read import validate as validate_tier2
        doc=read(out/'binding.json')
        values=list(csv.DictReader(io.StringIO(gzip.decompress(final['scores.csv.gz']).decode())))
        require(not validate_tier2(final,doc,True,values,x_read.validate_compact(final,doc,True)),
            'nominal tier-2 archive failed independent validation')
        if configuration!='generic':
            path=out/'p2/feasibility/s101/receipt.json';original=path.read_bytes()
            for defect in ('missing','failed','stale'):
                try:
                    if defect=='missing':path.unlink()
                    else:
                        value=json.loads(original)
                        if defect=='failed':value['status']='FAIL'
                        else:value['identity']['settings_sha256']='stale'
                        atomic(path,value)
                    states={name:x_tier2.readiness(out,doc,name) for name in x_tier2.COMPARISONS}
                    require(all(state==('PENDING' if defect=='missing' else 'DEFERRED_P2')
                        if x_tier2.COMPARISONS[name]['control_p2'] or x_tier2.COMPARISONS[name]['challenger_p2']
                        else state=='READY' for name,state in states.items()),'node defect escaped tier-2 P2 comparisons')
                    tier2_gate_cases.append(defect+'-receipt-P2-only')
                finally:path.write_bytes(original)
    # Returned feasibility is mandatory only for P2; corrupt copies leave A intact.
    original_tables=json.loads(returned['comparison-tables.json'])
    baseline=x_read.reading(original_tables,read(qpath),True)
    node_cases=[]
    for defect in ('missing','failed','stale'):
        altered=dict(returned);key='p2/feasibility/s101/receipt.json'
        if defect=='missing':altered.pop(key)
        else:
            value=json.loads(altered[key])
            if defect=='failed':value['status']='FAIL'
            else:value['identity']['settings_sha256']='stale'
            altered[key]=json.dumps(value).encode()
        validation=x_read.validate_compact(altered,read(out/'binding.json'),True)
        updated=x_read.isolate_p2(__import__('copy').deepcopy(original_tables),validation)
        reading=x_read.reading(updated,read(qpath),True)
        require(reading['admitted']['p'] is None and all(reading['admitted'][arm]==baseline['admitted'][arm] for arm in ('drift','plasticity')),'node failure escaped P2 comparison')
        node_cases.append(defect+'-receipt-P2-only-INCONCLUSIVE')
    require(sha(bound.message2/'admission.json') == core_hash, 'add-on changed core admission')
    require(read(out/'support.json')['drift']['status'] == ('DEGENERATE' if configuration=='generic' else 'READY'), 'synthetic support unexpectedly insufficient')
    return dict(configuration=configuration,rows=counts,requeued_every_row=requeue,tier2_node_gate_cases=tier2_gate_cases,node_feasibility_cases=['pass',*node_cases],
        stages=len(list((out/'runs').glob('*/complete.json'))),
        probe_receipts=len(list((out/'probes').glob('*/*/receipt.json'))),
        archive_bytes=archive.stat().st_size,metadata_compressed_bytes=metadata_bytes,
        core_admission_unchanged=True,passing_p_admitted_at_X13=False,kit_guards_pending_reference_at_X13=True,
        generic_generation_skipped=not (out/'own').exists() if configuration=='generic' else None)


def run(root, tier2=False):
    root.mkdir(parents=True,exist_ok=True)
    results = []
    for configuration in ('generic','broad','earlier','both'):
        folder = root/configuration
        results.append(branch(folder,configuration,requeue=tier2 or configuration=='both',tier2=tier2))
        shutil.rmtree(folder)  # retain compact summary only; disk is shared/tight
    failures = failure_branches(root/'failures')
    shutil.rmtree(root/'failures')
    forecast=size_forecast(tier2)
    require(max(b['metadata_compressed_bytes'] for b in results)<forecast['metadata_reserve_bytes'],'measured metadata exceeds run-scaled reserve')
    summary = dict(technical_synthetic=True,branches=results,failure_branches=failures,
        size_forecast=forecast,gpu_hours=0)
    atomic(root/'summary.json',summary)
    return summary


def failure_branches(root):
    """Real failure paths, including durable update-10 continuation and core isolation."""
    from kit import x_bind, x_support, x_p, x_train, x_collect
    from kit import message_handoff as handoff
    results = {}
    for status in ('PENDING',):
        a,_ = fixtures(root/status, admission=status)
        try: x_bind.job(a)
        except m.NotReady: results['pending-admission-refused'] = True
        else: raise ValueError('pending admission accepted')
        shutil.rmtree(root/status)
    a,_ = fixtures(root/'m10',hard='QUALIFIES')
    try: x_bind.job(a)
    except ValueError: results['M10-unrecorded-refused'] = True
    else: raise ValueError('unrecorded M10 accepted')
    shutil.rmtree(root/'m10')
    for domain in DOMAINS:
        a,_ = fixtures(root/domain, short_domain=domain, short_part='monitor')
        x_bind.job(a); x_support.job(NS(out=a.out,standin=True,action='audit'))
        x_support.job(NS(out=a.out,standin=True,action='generate',gpu='0'))
        support = read(a.out/'support.json')
        require(support['drift']['status'] == 'UNINFORMATIVE' and len(support['drift']['failures']) == 3, 'shortfall dropped/imputed a domain/seed')
        results['insufficient-'+domain] = True
        shutil.rmtree(root/domain)
    a,smoke_path = fixtures(root/'resume',short_domain='finqa',short_part='train')
    x_bind.job(a); x_support.job(NS(out=a.out,standin=True,action='audit'))
    x_support.job(NS(out=a.out,standin=True,action='generate',gpu='0'))
    support = read(a.out/'support.json')
    require(support['drift']['status'] == 'READY' and support['plasticity']['status'] == 'UNINFORMATIVE', 'probe-only shortage not independent')
    results['probe-only-shortage'] = True
    before = sha(a.message2/'admission.json')
    original = read(smoke_path)
    for key in ('finite','analytic','coefficient','norms','timing','contraction'):
        import copy
        failed=copy.deepcopy(original);failed['assertions'][key]['status']='FAIL';failed['passed']=False
        atomic(smoke_path,failed)
        try: x_p.job(NS(**vars(a),seed=101,smoke=smoke_path,action='measure',gpus='0,1,2,3,4,5,6,7'))
        except ValueError: results['P-smoke-'+key] = True
        else: raise ValueError('P smoke failure accepted')
    atomic(smoke_path,dict(original,schema='rung-p-smoke.v1'))
    try:x_p.job(NS(**vars(a),smoke=smoke_path,action='measure',gpus='0,1,2,3,4,5,6,7'))
    except ValueError:results['P2-old-P-receipt-refused']=True
    else:raise ValueError('old P receipt accepted')
    smoke_path.unlink()
    try:x_p.job(NS(**vars(a),smoke=smoke_path,action='measure',gpus='0,1,2,3,4,5,6,7'))
    except ValueError:results['P2-smoke-missing-refused']=True
    else:raise ValueError('missing P2 smoke accepted')
    atomic(smoke_path,original)
    # The minimal tiny trainer preserves ALL continuation state, with a new suffix.
    scheduled,_ = x_train.targets(read(a.out/'binding.json'),101,None)
    run = a.out/'tiny-resume'
    try: x_train.standin_updates(run,scheduled,40,10)
    except InterruptedError: pass
    saved = read(run/'tiny-state.json')
    handoff.save(a.out,a.durable); shutil.rmtree(a.out); handoff.restore(a.durable,a.out)
    require(read(run/'tiny-state.json') == saved and saved['position']==320, 'update10 full state lost')
    adapted,_ = x_train.targets(read(a.out/'binding.json'),101,None,{d:float(i+1) for i,d in enumerate(DOMAINS)},10)
    continued = x_train.standin_updates(run,adapted,40)
    reference = x_train.standin_updates(a.out/'uninterrupted',adapted,40)
    require(continued == reference, 'resumed optimizer/scheduler/EMA/RNG/data position differs')
    results['mid-update10-resume-new-slots'] = True
    import importlib.util
    spec=importlib.util.spec_from_file_location('addon_cpu_stack',ROOT/'tests/fixtures/addon_cpu_stack.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(module.exercise(root/'real-cpu-resume')['status']=='PASS','real X CPU boundary failed')
    results['X-entry-runner-live-hash-resume-new-slots'] = True
    require(sha(a.message2/'admission.json') == before, 'add-on failure changed core')
    # Core handoff/return remains independently usable after P failure.
    handoff.save(a.message2,a.message2_durable)
    archive = root/'failed-addon.tar.gz'
    x_collect.job(NS(out=a.out,standin=True,archive=archive))
    require('binding.json' in x_collect.load(archive,True), 'partial failure not collectable')
    results['add-on-failure-core-return-independent'] = True
    return results


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--standin',action='store_true',required=True)
    p.add_argument('--tier2',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(a.out).free >= 12*1024**3, 'disk below 12 GiB; stop')
    with (a.out/'dryrun.log').open('w') as log, contextlib.redirect_stdout(log): summary=run(a.out,a.tier2)
    print(json.dumps(summary,sort_keys=True))
    return 0
if __name__=='__main__':raise SystemExit(main())
