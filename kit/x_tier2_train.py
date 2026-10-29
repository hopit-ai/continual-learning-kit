"""X-only tier-2 stages; each arm owns its checkpoints, schedules and optimiser.

The reviewed trainer remains unchanged. Adaptive windows stop at native saved
states, qualify their restoration, then change only future replay rows.
"""
from kit.x_common import *
from kit.x_tier2 import identity
from kit import x_train as train
from kit.x_tier2_slots import compose
from types import SimpleNamespace as NS
from contextlib import contextmanager
import subprocess
import time
import shutil


@contextmanager
def environment(values):
    previous = {key:os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def stage(a):
    doc = context(a.out, a.standin)
    if (a.out/'scoring-assignment.json').exists():
        from kit.x_tier2 import tier1_return
        if not (a.out/'tier2/plan.json').exists():raise m.NotReady('X14 plan pending')
        tier1_return(a.out)
        if not os.environ.get('M2_DURABLE'):raise m.NotReady('X12 durable barrier required')
        from kit.m2_node import barrier
        barrier(Path(os.environ['M2_DURABLE']),'X12',sha(a.out/'scoring-assignment.json'))
    spec = identity(a.arm)
    if terminal_path(a.out,a.arm,a.seed).exists():
        print('SKIP terminal tier-2 arm:',a.arm,a.seed);return
    state = disposition(a.out, a.arm)
    if state != 'READY':
        atomic(a.out/'tier2'/spec['comparison']/'disposition.json', dict(status=state))
        return
    run = a.out/'runs'/f'x-{a.arm}-s{a.seed}'
    allocation = spec['allocation']
    segmented = allocation != 'standalone'
    start = 0 if a.phase in ('full', 'prefix') else (a.end-10 if allocation == 'adaptive' else 10)
    end = a.end
    require((allocation=='standalone' and (start,end)==(0,40)) or
            (allocation=='adaptive' and (start,end) in ((0,10),(10,20),(20,30),(30,40))) or
            (allocation in ('drift','incremental') and (start,end) in ((0,10),(10,40))), 'invalid tier-2 phase')
    require(all((a.out/'scores'/f'x-{a.arm}-s{a.seed}-k0-{p}'/'manifest.json').exists()
                for p in (*m.PANEL_SETS,'hellaswag')), 'tier-2 incoming scores pending')
    for panel in (*m.PANEL_SETS,'hellaswag'):
        m.frozen_files(a.out/'scores'/f'x-{a.arm}-s{a.seed}-k0-{panel}')
    ref = incoming(doc, a.seed)
    p2 = None
    if spec['p2']:
        from kit.x_p import calibration
        p2 = calibration(a.out, doc)  # Includes the scoped P2 PASS smoke gate.
        from kit.x_p2_feasibility import ensure
        if not ensure(a,doc,p2):
            state=disposition(a.out,a.arm)
            atomic(a.out/'tier2'/spec['comparison']/'disposition.json',dict(status='PENDING' if state=='READY' else state))
            return
    frozen = dict(arm='x-'+a.arm,seed=a.seed,**spec,incoming=m.checked_model(ref,a.standin),
        configuration=doc['configuration'],binding_sha256=sha(a.out/'binding.json'),steps=40,batch=32,
        lr=1e-5,warmup=10,fresh_optimizer=True,fresh_ema=True,recipe='D+replay',
        variant='p2-l2-init' if spec['p2'] else 'incumbent',
        p2_calibration_sha256=sha(a.out/'p2/calibration.json') if p2 else None,
        p2_preflight_receipt_sha256=sha(a.out/'p2/feasibility'/f's{a.seed}'/'receipt.json') if p2 else None,
        p2_cache_policy='NATIVE_TIER2_NO_X4_SUBSTITUTION',technical_synthetic=a.standin)
    bind(run/'identity.json', frozen)
    marker = run/'segments'/f'{start}-{end}.json'
    if marker.exists():
        require(sha(run/f'schedule-{end}.jsonl') == read(marker)['schedule_sha256'], 'tier-2 segment schedule changed')
        if (run/'complete.json').exists():m.frozen_files(run)
        elif end == 40:
            finish(run,frozen,rows(run/'schedule-40.jsonl'),read(run/'slots-40.json'),a.standin,spec['p2'])
        return
    tokenizer = None
    if not a.standin:
        require(a.sdpo and a.tokenizer and len(set(a.gpu.split(','))) == 8, 'tier-2 requires pinned SDPO/tokenizer and eight GPUs')
        require(subprocess.check_output(['git','-C',str(a.sdpo),'rev-parse','HEAD'],text=True).strip() ==
            '7c457fc1b1f636ae794eb0362ba37d4743b06fbc', 'SDPO pin changed')
        from transformers import AutoTokenizer
        m.identity(a.tokenizer, 'student')
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    current = rows(run/f'schedule-{start}.jsonl') if start else train.targets(doc,a.seed,tokenizer)[0]
    audit = dict(start=start,end=end,allocation='incumbent')
    scheduled = current
    if start or allocation == 'standalone':
        if allocation in ('standalone','incremental'):
            scores = train.signal(a.out,doc,a.seed,'plasticity')
        else:
            scores = read(a.out/'tier2/signals'/a.arm/f's{a.seed}'/f'k{start}.json')['allocation_signal']
        scheduled,audit = allocated_targets(doc,a.seed,tokenizer,current,scores,start,end,allocation)
    write_rows(run/f'schedule-{end}.jsonl', scheduled)
    bind(run/f'slots-{end}.json',audit)
    if a.standin:
        train.standin_updates(run,scheduled,end,getattr(a,'interrupt_update',None))
        if spec['p2']:
            bind(run/'ema-origin/rank0.json',dict(status='PASS',variant='p2-l2-init',before_first_update=True,
                student_checkpoint=ref['model'],teacher_checkpoint=ref['model']))
            atomic(run/'p2-rank0.json',dict(schema='rung-p2-training.v1',variant='p2-l2-init',teacher_regularised=False,
                completed_updates=end,updates=[dict(lambda_value=p2['coefficient']['value']) for _ in range(end)]))
        for step in STEPS[1:]:
            if start < step <= end:
                train.export(run,step,a,{})
        if end < 40:
            bind(run/'boundaries'/f'k{end}'/'prefix.json',dict(update=end,total_training_steps=40,full_state=True,technical_synthetic=True))
    else:
        _production(a,doc,run,ref,p2,scheduled,start,end)
    if p2:
        width=1 if a.standin else 8
        check_ema_origin(run,ref,width)
        for rank in range(width):
            origin=read(run/'ema-origin'/f'rank{rank}.json')
            first=run/'first-segment/ema-origin'/f'rank{rank}.json'
            if start==0:bind(first,origin)
            else:require(read(first)==origin,'P2 first-segment EMA origin changed')
    bind(marker,dict(start=start,end=end,schedule_sha256=sha(run/f'schedule-{end}.jsonl'),technical_synthetic=a.standin))
    if end == 40:
        finish(run,frozen,scheduled,audit,a.standin,spec['p2'])


def allocated_targets(doc,seed,tokenizer,current,scores,start,end,allocation):
    if allocation=='adaptive':
        return compose(doc,seed,tokenizer,current,scores,start,end)
    scheduled,audit=train.targets(doc,seed,tokenizer,scores,start)
    require(scheduled[:start*32]==current[:start*32],'fixed allocation changed consumed incumbent prefix')
    return scheduled,audit


def check_ema_origin(run,ref,width):
    for rank in range(width):
        value=read(run/'ema-origin'/f'rank{rank}.json')
        require(value['status']=='PASS' and value['variant']=='p2-l2-init' and value['before_first_update'] and
            value['student_checkpoint']==value['teacher_checkpoint']==ref['model'],'tier-2 P2 unchanged-incoming EMA assertion missing')


def finish(run,frozen,scheduled,audit,standin,p2):
    """Idempotent finalisation, including a crash after the last segment marker."""
    bind(run/'summary.json',dict(frozen,schedule_sha256=digest(scheduled),slot_audit=audit))
    names=['identity.json','summary.json']+[f'export-{s}.json' for s in STEPS[1:]]
    names += [str(p.relative_to(run)) for p in sorted((run/'segments').glob('*.json'))]
    if p2:
        if not standin:names += [f'resource-r{r}.json' for r in range(8)]
        names += [f'p2-rank{r}.json' for r in range(1 if standin else 8)]
        names += [f'ema-origin/rank{r}.json' for r in range(1 if standin else 8)]
        names += [f'first-segment/ema-origin/rank{r}.json' for r in range(1 if standin else 8)]
    bind(run/'manifest.json',m.files_manifest(run,names,technical_synthetic=standin))
    bind(run/'complete.json',dict(complete=True))


def retain_attempt(run,start):
    """Invalidate abandoned future exports and retain their bytes before rerun."""
    if not (run/'started.json').exists():return
    kept=run/'partials'/str(time.time_ns());kept.mkdir(parents=True)
    for name in ('metrics.jsonl','started.json','trainer-exit.json','resolved-config.yaml','resolved-config.err','train'):
        if (run/name).exists():(run/name).rename(kept/name)
    for path in list((run/'sdft').iterdir()) if (run/'sdft').exists() else ():
        if path.name.startswith('global_step_') and int(path.name.removeprefix('global_step_'))<=start:continue
        path.rename(kept/path.name)
    for path in list(run.iterdir()):
        if path.name.startswith(('hf-step','export-')):
            step=int(path.name.removeprefix('hf-step').removeprefix('export-').removesuffix('.json'))
            if step>start:path.rename(kept/path.name)
    for path in run.glob('p2-rank*.json'):
        shutil.copy2(path,kept/path.name)  # resume consumes only its <=start history
    return kept


def terminal_path(out,arm,seed):
    return out/'tier2'/identity(arm)['comparison']/f'terminal-{arm}-s{seed}.json'


def segment_resources(a,doc,run,start,end):
    """A completed but unqualified segment is terminal for this arm/seed only."""
    from kit.x_p2_feasibility import check_rank,identity as node_identity,SETTINGS
    expected=dict(node_identity(a.out,doc,a.seed,sha(a.out/'p2/calibration.json')),
        preflight_receipt_sha256=sha(a.out/'p2/feasibility'/f's{a.seed}'/'receipt.json'))
    try:
        for rank in range(8):
            value=read(run/f'resource-r{rank}.json')
            require(value['rank']==rank,'tier-2 resource rank differs')
            check_rank(value,expected,read(SETTINGS),completed_updates=end)
    except (ValueError,KeyError,OSError,TypeError) as error:
        atomic(terminal_path(a.out,a.arm,a.seed),dict(status='DEFERRED',arm=a.arm,seed=a.seed,
            start=start,end=end,returncode=0,reason=str(error)))
        raise
    (run/'segments').mkdir(parents=True,exist_ok=True)
    for rank in range(8):
        shutil.copy2(run/f'resource-r{rank}.json',run/'segments'/f'resource-{start}-{end}-r{rank}.json')


def _production(a,doc,run,ref,p2,scheduled,start,end):
    from kit import m2_train
    from kit.x_resume import resume_slots_command
    from transformers import AutoTokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq
    data=a.out/'data'/f'x-{a.arm}-s{a.seed}'/f'{start}-{end}'
    if not (data/'manifest.json').exists():
        tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
        converted,_=m2_train.convert(scheduled,tokenizer,'legal')
        data.mkdir(parents=True,exist_ok=True)
        for part,values in [('train',converted),('val',converted[:32]),('test',converted[:1])]:
            pq.write_table(pa.Table.from_pylist(values),data/(part+'.parquet'))
        atomic(data/'manifest.json',m.files_manifest(data,['train.parquet','val.parquet','test.parquet'],rows=1280))
    m.frozen_files(data)
    cmd,_,env=m2_train.command(NS(**dict(vars(a),stage='legal')),'D+replay',ref['model'],data,run)
    for key in list(env):
        if key.startswith(('X_P2_','X_TIER2_','M2_ADDON_')) or key in ('X_P_POST_MODEL','X_P_EMA_RECEIPTS','X_DETERMINISTIC_SEED'):
            env.pop(key)
    cmd=[v.replace('c_sdft_entry.py','x_sdft_entry.py') for v in cmd]
    values={}
    if start or end < 40:
        values.update(M2_ADDON_BOUNDARY=str(run/'boundaries'/f'k{end if end<40 else start}'/'prefix.json'),
            X_TIER2_BOUNDARY_STEP=str(end if end<40 else start))
        if end < 40:
            values['M2_ADDON_PREFIX']='1'
        if start:
            values.update(X_TIER2_RESUME_STEP=str(start),X_TIER2_RESUME_BOUNDARY=str(run/'boundaries'/f'k{start}'/'prefix.json'))
        env.update(values)
        for key,value in values.items():
            cmd.append('+ray_kwargs.ray_init.runtime_env.env_vars.'+key+'='+json.dumps(value))
        if start:
            with environment(values):
                cmd=resume_slots_command(cmd,run/'sdft'/f'global_step_{start}')
    if p2:
        from kit.x_p2_measure import runtime
        cmd=runtime(env,cmd,mode='train',batch=a.out/'p2'/f's{a.seed}',run=run,anchor=doc['p2_anchor']['model'],
            seed=a.seed,value=p2['coefficient']['value'])
        from kit.x_p2_feasibility import scored_runtime
        cmd=scored_runtime(env,cmd,a,doc,run)
        env['X_TIER2_NATIVE']='1'
        cmd.append('+ray_kwargs.ray_init.runtime_env.env_vars.X_TIER2_NATIVE="1"')
    bind(run/f'command-{start}-{end}.json',cmd)
    # On requeue retain the exact saved starting boundary; partial outputs from
    # this segment are overwritten, never used to choose a shorter dose.
    retain_attempt(run,start)
    atomic(run/'started.json',dict(start=start,end=end,started_ns=time.time_ns()))
    began=time.monotonic()
    try:
        subprocess.run(cmd,cwd=a.sdpo,env=env,check=True)
    finally:
        atomic(run/'training-costs'/f'{time.time_ns()}.json',dict(gpu_hours=8*(time.monotonic()-began)/3600,
            seconds=time.monotonic()-began,start=start,end=end,technical_synthetic=False))
    if p2:segment_resources(a,doc,run,start,end)
    require((run/'sdft'/f'global_step_{end}'/'actor').is_dir(),'tier-2 segment dose incomplete')
    for step in STEPS[1:]:
        if start < step <= end:
            train.export(run,step,a,env)
    if p2:
        check_ema_origin(run,ref,8)
        for rank in range(8):
            value=read(run/f'p2-rank{rank}.json')
            require(value['completed_updates']==end and len(value['updates'])==end and
                all(r['lambda_value']==p2['coefficient']['value'] for r in value['updates']), 'tier-2 P2 dose or lambda changed')
