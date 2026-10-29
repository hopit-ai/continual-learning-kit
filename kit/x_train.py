#!/usr/bin/env python3
"""Fresh X arms through the unchanged D recipe; drift continues full update-10 state."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *
from kit import m2_train
from kit.x_slots import compose
from kit.x_resume import resume_slots_command
import subprocess
import time


def targets(doc, seed, tokenizer, scores=None, start=0):
    roots = doc['assets']['inputs']
    task = rows(Path(roots['c_targets']['path'])/f'schedule-{seed}.jsonl')
    pools = dict(generic=[dict(r, p3b_replay=True) for r in rows(Path(roots['replay']['path'])/'replay.jsonl')],
                 root=Path(roots['pools']['path']))
    return compose(task, pools, doc['configuration'], seed, tokenizer, scores, start)


def signal(out, doc, seed, arm):
    if arm == 'plasticity':
        from kit.s_probe import receipt_path
        return {d: read(receipt_path(out,f'b-s{seed}',d))['drop'] for d in doc['domains']}
    return {d: read(out/'signals'/f's{seed}'/(d+'.json'))['drift'] for d in doc['domains']}


def export(run, step, a, env):
    target = run/f'hf-step{step}'
    if a.standin:
        atomic(target/'config.json', dict(technical_synthetic=True, arm=a.arm, seed=a.seed, step=step))
        atomic(run/f'export-{step}.json', dict(technical_synthetic=True, model=str(target)))
    else:
        from kit.v4_sft import complete_export
        if not complete_export(target):
            subprocess.run([a.trainer_python, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                '--local_dir', str(run/'sdft'/f'global_step_{step}'/'actor'), '--target_dir', str(target)],
                cwd=a.sdpo, env=env, check=True)
        require(complete_export(target), 'incomplete X export')
        bind(run/f'export-{step}.json', m.identity(target))


def standin_updates(run, scheduled, end, interrupt=None):
    """Tiny CPU optimizer/EMA/RNG/data-position fixture, persisted at EACH update."""
    state = read(run/'tiny-state.json') if (run/'tiny-state.json').exists() else dict(
        update=0, weight=1., moment=0., ema=1., scheduler=0, position=0, rng=0)
    for update in range(state['update']+1, end+1):
        batch = scheduled[(update-1)*32:update*32]
        value = sum(int(digest(r['id'])[:8],16)/2**32 for r in batch)/32
        state['moment'] = .9*state['moment']+.1*value
        lr = min((update-1)/10, 1)*1e-5
        state['weight'] -= lr*state['moment']
        state['ema'] = .99*state['ema']+.01*state['weight']
        state.update(update=update, scheduler=update, position=update*32, rng=state['rng']+32)
        atomic(run/'tiny-state.json', state)
        if update == 10: bind(run/'prefix-state.json', state)
        if interrupt == update: raise InterruptedError('technical interruption at update '+str(update))
    return state


def retain_interrupted(run, phase):
    """Retain append-only attempts; drift continuation keeps the frozen prefix."""
    if not (run/'started.json').exists(): return
    kept=run/'partials'/str(time.time_ns());kept.mkdir(parents=True)
    for name in ('metrics.jsonl','started.json','trainer-exit.json','resolved-config.yaml','resolved-config.err','train'):
        if (run/name).exists(): (run/name).rename(kept/name)
    sdft=run/'sdft'
    if sdft.exists():
        if phase == 'continue':
            for item in list(sdft.iterdir()):
                if item.name != 'global_step_10': item.rename(kept/item.name)
        else: sdft.rename(kept/'sdft')
    # A failure during HF export must not leave an apparently complete export
    # from the abandoned attempt for the restarted stage to reuse.
    for item in list(run.iterdir()):
        if item.name.startswith(('hf-step','export-')):
            if phase == 'continue' and item.name in ('hf-step10','export-10.json'): continue
            item.rename(kept/item.name)
    for name in ('summary.json','manifest.json'):
        if (run/name).exists(): (run/name).rename(kept/name)
    if phase != 'continue':
        for name in ('boundary-state','ema-origin','prefix.json','restore-qualified.json'):
            if (run/name).exists(): (run/name).rename(kept/name)
    else:
        states=run/'boundary-state'
        for item in states.glob('loaded-*.json'):
            item.rename(kept/item.name)
        if (run/'restore-qualified.json').exists(): (run/'restore-qualified.json').rename(kept/'restore-qualified.json')


def stage(a):
    doc = context(a.out, a.standin)
    from kit.x_p2_feasibility import contains_p2
    regularised=contains_p2(a,doc)
    p2=None
    if regularised and doc.get('p_variant')!='p2-l2-init':
        atomic(a.out/'dispositions'/(a.arm+'.json'),dict(status='DEFERRED',reason='P2-containing arm lacks its registered binding'));return
    if doc.get('p_variant')=='p2-l2-init' and (a.arm=='control' or regularised):
        from kit.x_p import calibration
        try:p2=calibration(a.out,doc)
        except (ValueError,KeyError,OSError,TypeError) as error:
            atomic(a.out/'p2'/f'calibration-unavailable-{a.arm}-s{a.seed}.json',dict(
                status='DEFERRED' if regularised else 'CONTROL_WITHOUT_CACHE',reason=str(error),cache_used=False))
            if regularised:
                atomic(a.out/'dispositions'/(a.arm+'.json'),dict(status='DEFERRED',reason='P2 calibration unavailable'))
                print('SKIP P2: calibration unavailable');return
    if regularised and p2 is not None:
        from kit.x_p2_feasibility import ensure,retry_pending
        if not ensure(a,doc,p2):
            if a.arm=='p' and retry_pending(a.out,a.seed):
                print('P2 preflight interrupted; one fresh requeue remains');return
            atomic(a.out/'dispositions'/(a.arm+'.json'),dict(status='DEFERRED',reason='node feasibility missing/failed'));return
    state = disposition(a.out, a.arm)
    if state != 'READY':
        bind(a.out/'dispositions'/(a.arm+'.json'), dict(status=state))
        print('SKIP:', a.arm, state); return
    require(a.arm != 'drift' or a.phase in ('prefix', 'continue'), 'drift requires separate prefix/continue rows')
    require(a.arm == 'drift' or a.phase == 'full', 'only drift has a common prefix')
    require(a.standin or len(set(a.gpu.split(','))) == 8, 'eight training GPUs required')
    for panel in (*m.PANEL_SETS, 'hellaswag'):
        baseline = a.out/'scores'/f'x-{a.arm}-s{a.seed}-k0-{panel}'
        require((baseline/'manifest.json').exists(), 'score stage-start panels before EMA/training')
        m.frozen_files(baseline)
    ref = incoming(doc, a.seed) if not regularised else checkpoint(a.out, doc, 'p', a.seed, 0)
    observed = m.checked_model(ref, a.standin)
    run = a.out/'runs'/f'x-{a.arm}-s{a.seed}'
    stage_identity = dict(arm='x-'+a.arm, seed=a.seed, incoming=observed,
        configuration=doc['configuration'], binding_sha256=sha(a.out/'binding.json'),
        steps=40, batch=32, lr=1e-5, warmup=10, fresh_optimizer=True, fresh_ema=True,
        recipe='D+replay',variant='p2-l2-init' if regularised else 'incumbent',
        p2_calibration_sha256=sha(a.out/'p2/calibration.json') if p2 else None,
        p2_cache_policy='BOUND_X4' if p2 else 'NATIVE_NO_CACHE',technical_synthetic=a.standin)
    bind(run/'identity.json', stage_identity)
    if (run/'complete.json').exists():
        m.frozen_files(run); print('SKIP:', run.name); return
    tokenizer = None
    if not a.standin:
        require(a.sdpo and a.tokenizer, 'pinned SDPO/tokenizer required')
        require(subprocess.check_output(['git','-C',str(a.sdpo),'rev-parse','HEAD'], text=True).strip() ==
                '7c457fc1b1f636ae794eb0362ba37d4743b06fbc', 'SDPO pin changed')
        m.identity(a.tokenizer, 'student')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    adaptive = a.arm == 'plasticity' or a.phase == 'continue'
    scores = signal(a.out, doc, a.seed, a.arm) if adaptive else None
    scheduled, audit = targets(doc, a.seed, tokenizer, scores, 10 if a.phase == 'continue' else 0)
    if p2:require(digest(scheduled[:32])==p2['seeds'][str(a.seed)]['first_batch_sha256'],'P2 first scheduled batch differs from X4')
    phase = 'allocated' if adaptive else 'incumbent'
    write_rows(run/(phase+'-schedule.jsonl'), scheduled)
    bind(run/(phase+'-slots.json'), audit)
    if a.phase == 'continue':
        original = rows(run/'incumbent-schedule.jsonl')
        require(original[:320] == scheduled[:320], 'update-10 resume changed consumed prefix')
        # Every task position and the complete slot mask must remain identical.
        require(all(bool(x.get('p3b_replay')) == bool(y.get('p3b_replay')) and
                    (bool(x.get('p3b_replay')) or x == y) for x,y in zip(original,scheduled)), 'task position/total replay drift')
    if a.standin:
        if regularised:
            import copy
            import torch
            from kit.x_ema import assert_teacher_origin
            student = torch.nn.Linear(1,1,bias=False)
            with torch.no_grad(): student.weight.zero_()
            value = assert_teacher_origin(student,copy.deepcopy(student),ref['model'],ref['model'],ref['model'])
            bind(run/'ema-origin/rank0.json',dict(value,variant='p2-l2-init',technical_synthetic=True))
            bind(run/'p2-rank0.json',dict(schema='rung-p2-training.v1',variant='p2-l2-init',
                updates=[dict(update=i,lambda_value=p2['coefficient']['value']) for i in range(1,41)],completed_updates=40,teacher_regularised=False))
        if a.phase == 'prefix' and (run/'prefix.json').exists(): print('SKIP prefix:', a.seed); return
        standin_updates(run, scheduled, 10 if a.phase == 'prefix' else 40, getattr(a,'interrupt_update',None))
        for step in ((10,) if a.phase == 'prefix' else (10,20,30,40)): export(run, step, a, {})
        if a.phase == 'prefix':
            bind(run/'prefix.json', dict(update=10, total_training_steps=40, full_state=True)); return
    else:
        data = a.out/'data'/f'x-{a.arm}-s{a.seed}'/phase
        converted, token_counts = m2_train.convert(scheduled, tokenizer, 'legal')
        import pyarrow as pa
        import pyarrow.parquet as pq
        if not (data/'manifest.json').exists():
            data.mkdir(parents=True, exist_ok=True)
            for part, values in [('train',converted), ('val',converted[:32]), ('test',converted[:1])]:
                pq.write_table(pa.Table.from_pylist(values), data/(part+'.parquet'))
            atomic(data/'manifest.json', m.files_manifest(data, ['train.parquet','val.parquet','test.parquet'], rows=1280))
        m.frozen_files(data)
        from types import SimpleNamespace as NS
        trainer_args=NS(**dict(vars(a),stage='legal'))
        cmd, resolve, env = m2_train.command(trainer_args, 'D+replay', ref['model'], data, run)
        env.pop('M2_ADDON_PREFIX', None); env.pop('M2_ADDON_BOUNDARY', None)
        env.pop('X_P_POST_MODEL', None); env.pop('X_P_EMA_RECEIPTS', None)
        for key in list(env):
            if key.startswith(('X_P2_','X_TIER2_')) or key=='X_DETERMINISTIC_SEED':env.pop(key)
        # X's entry imports C's unchanged runner and all its validators.
        cmd = [v.replace('c_sdft_entry.py','x_sdft_entry.py') for v in cmd]
        if a.arm == 'drift':
            env.update(M2_ADDON_BOUNDARY=str(run/'prefix.json'))
            if a.phase == 'prefix': env['M2_ADDON_PREFIX']='1'
            for key in (('M2_ADDON_PREFIX','M2_ADDON_BOUNDARY') if a.phase == 'prefix' else ('M2_ADDON_BOUNDARY',)):
                cmd.append('+ray_kwargs.ray_init.runtime_env.env_vars.'+key+'='+json.dumps(env[key]))
        if a.phase == 'continue': cmd = resume_slots_command(cmd, run/'sdft/global_step_10')
        from kit.x_p2_measure import deterministic_runtime
        cmd=deterministic_runtime(env,cmd,a.seed)
        if p2:
            from kit.x_p2_measure import runtime
            cmd=runtime(env,cmd,mode='train' if regularised else 'control',batch=a.out/'p2'/f's{a.seed}',run=run,
                anchor=doc['p2_anchor']['model'],seed=a.seed,value=p2['coefficient']['value'] if regularised else 0.)
        if regularised and p2:
            from kit.x_p2_feasibility import scored_runtime
            cmd=scored_runtime(env,cmd,a,doc,run)
        # Never restart a half-finished drift continuation from incoming weights.
        # Full controls/P/plasticity repeat identical stages after an interruption,
        # matching M's registered restart behavior; drift always reloads its prefix.
        bind(run/(a.phase+'-command.json'), cmd)
        if a.phase != 'prefix' or not (run/'prefix.json').exists():
            retain_interrupted(run,a.phase)
            atomic(run/'started.json',dict(phase=a.phase,started_ns=time.time_ns()))
            start = time.monotonic()
            try:
                subprocess.run(cmd, cwd=a.sdpo, env=env, check=True)
            finally:
                atomic(run/'training-costs'/f'{time.time_ns()}.json', dict(
                    gpu_hours=8*(time.monotonic()-start)/3600, seconds=time.monotonic()-start,
                    phase=a.phase, technical_synthetic=False))
        if a.phase == 'prefix':
            require((run/'prefix.json').exists(), 'prefix did not reach complete update-10 state')
            export(run, 10, a, env); return
        require((run/'sdft/global_step_40/actor').is_dir(), 'final optimizer dose missing')
        if regularised:
            for rank in range(8):
                progress=read(run/f'p2-rank{rank}.json')
                require(progress['variant']=='p2-l2-init' and progress['completed_updates']==40,'P2 incomplete regularised dose')
                value = read(run/'ema-origin'/f'rank{rank}.json')
                require(value['status']=='PASS' and value['student_checkpoint']==value['teacher_checkpoint']==ref['model'],
                        'P2 missing unchanged-incoming EMA assertion')
        for step in (10,20,30,40): export(run, step, a, env)
    bind(run/'summary.json', dict(stage_identity, schedule_sha256=digest(scheduled), slot_audit=audit))
    names = ['identity.json','summary.json']+[f'export-{s}.json' for s in (10,20,30,40)]
    if regularised: names += [f'p2-rank{r}.json' for r in range(1 if a.standin else 8)]+ [f'ema-origin/rank{r}.json' for r in range(1 if a.standin else 8)]
    bind(run/'manifest.json', m.files_manifest(run, names, technical_synthetic=a.standin))
    bind(run/'complete.json', dict(complete=True))


def job(a):
    from kit.x_tier2 import ARMS2
    if a.arm in ARMS2:
        from kit.x_tier2_train import stage as tier2_stage
        with m.lock(a.out/'runs'/f'x-{a.arm}-s{a.seed}'):
            return tier2_stage(a)
    with m.lock(a.out/'runs'/f'x-{a.arm}-s{a.seed}'): return stage(a)


def configure(p):
    from kit.x_tier2 import ARMS2
    p.add_argument('--arm', choices=(*ARMS, *ARMS2), required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--phase', choices=('full','prefix','continue'), default='full')
    p.add_argument('--gpu', default='0,1,2,3,4,5,6,7')
    p.add_argument('--tokenizer'); p.add_argument('--sdpo', type=Path)
    p.add_argument('--trainer-python', default=sys.executable)
    p.add_argument('--end', type=int, choices=(10,20,30,40), default=40)
if __name__ == '__main__': raise SystemExit(entry(job, configure))
