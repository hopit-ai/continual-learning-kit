#!/usr/bin/env python3
"""One exact connected stage; inherited full-weight objectives, exports and restart handling."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit.p3b_train import completed_training
from kit.c_data import encode
import subprocess
import time


def command(a, recipe, model, data, run):
    from kit.m2_paths import local
    model=local(model)
    from kit.c_train import command as c_command
    cmd, resolve, env = c_command(a, recipe, model, data, run)
    if a.stage == 'math':
        def adapt(values):
            result=[]
            for value in values:
                value=value.replace('c_sdft_entry.py','sdft_entry.py').replace('c_reward.py','p3b_reward.py')
                value=value.replace('data.max_prompt_length=10240','data.max_prompt_length=2048')
                value=value.replace('self_distillation.max_reprompt_len=14336','self_distillation.max_reprompt_len=6144')
                result.append(value)
            return result
        cmd,resolve=adapt(cmd),adapt(resolve)
    return cmd,resolve,env


def job(a):
    started = time.monotonic(); a._training_cost_root = None
    try: return stage(a)
    finally:
        if a._training_cost_root is not None:
            atomic(a._training_cost_root / f'{time.time_ns()}.json', dict(kind='training_job',
                gpu_hours=8*(time.monotonic()-started)/3600,
                completed=(a._training_cost_root.parent / 'complete.json').exists(),technical_synthetic=False,
                includes='entire invocation including input checks, data preparation, config resolution, optimizer process and merges'))


def stage(a):
    doc = binding(a.out, a.standin)
    if a.stage=='math' and a.trajectory=='sibling' and doc['hard_math']=='QUALIFIES':
        durable_admission(a.out,doc,a.durable)
    from kit.m2_paths import local
    assets=json.loads((a.out/'assets.json').read_text())
    require(assets['technical_synthetic'] is a.standin,'synthetic assets mismatch')
    if a.stage=='legal':
        require(permitted_legal(a.out,doc,a.configuration),'configuration not nominated or not the surviving incumbent')
        arm=legal_arm(a.configuration); config=a.configuration; recipe='D+replay'
        targets=local(assets['inputs']['c_targets']['path'])
    else:
        if doc['hard_math']!='QUALIFIES':
            print('SKIP: stage 3',doc['hard_math']); return
        arm=math_arm(a.trajectory)
        config=None if a.trajectory=='plain' else admission(a.out,doc)
        recipe='D' if config is None else 'D+replay'
        targets=a.out/'targets-math'
    for key,value in assets['inputs'].items():
        if key in ('pools','replay') and config is None:continue
        check(Path(value['path'])/'manifest.json',value['manifest']);frozen_files(Path(value['path']))
    incoming=record(a.out,doc,arm,a.seed,0)
    for panel in PANEL_SETS:
        baseline=a.out/'scores'/f'{arm}-s{a.seed}-k0-{panel}'
        if not (baseline/'manifest.json').exists(): raise NotReady('score incoming panels before training: '+str(baseline))
        frozen_files(baseline)
    require(a.standin or len(set(a.gpu.split(',')))==8,'training uses eight distinct GPUs')
    observed_model=checked_model(incoming,a.standin)
    manifest=frozen_files(targets)
    require(manifest['technical_synthetic'] is a.standin,'synthetic target mismatch')
    task=t.read_jsonl(targets/f'schedule-{a.seed}.jsonl')
    require(len(task)==1280,'schedule must contain 1280 exposures')
    replay_identity=None; slot_audit=None
    if config is not None:
        from kit.next_slots import compose
        replay_root=local(assets['inputs']['replay']['path'])
        pool_root=local(assets['inputs']['pools']['path'])
        replay=[dict(r,p3b_replay=True) for r in t.read_jsonl(replay_root/'replay.jsonl')]
        tok=None
        if not a.standin:
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
        task,slot_audit=compose(task,replay,pool_root,config,a.seed,tok,
                               task_name='contract_nli' if a.stage=='legal' else 'math')
        replay_identity=dict(manifest=s.file_hash(replay_root/'manifest.json'),
                             pools=s.file_hash(pool_root/'manifest.json'),slot_audit=slot_audit)
    name = f'{arm}-s{a.seed}'; run = a.out / 'runs' / name; data = a.out / 'data' / name
    observed = dict(incoming=observed_model, incoming_record=incoming, rows=s.digest(task), seed=a.seed, recipe=recipe,
        stage=a.stage, configuration=config, steps=40, batch=32, lr=1e-5, warmup=10,
        fresh_optimizer=True,fresh_ema=True,response_tokens=2048,replay=replay_identity,
        binding_sha256=s.file_hash(a.out/'binding.json'),target_manifest=s.file_hash(targets/'manifest.json'),
        protocol=protocol(),settings_sha256=s.file_hash(R/'v4-recipes.json'),technical_synthetic=a.standin)
    with lock(run):
        bind(a.out / 'identities' / (name + '.json'), observed)
        if (run / 'complete.json').exists():
            frozen_files(run)
            if not a.standin:
                for step in STEPS[1:]: identity(run / f'hf-step{step}', receipt=run / f'export-{step}.json')
            print('SKIP:', name); return
        if a.standin:
            from kit.c_sim import Tokenizer
            converted, token_counts = convert(task, Tokenizer(), a.stage)
            for step in STEPS[1:]:
                atomic(run / f'export-{step}.json',dict(technical_synthetic=True,step=step,
                    model=f'message2-standin:{observed["binding_sha256"]}:{arm}:s{a.seed}:k{step}'))
            atomic(run / 'summary.json', dict(**observed, seconds=0, gpu_hours=0, scheduled_tokens=token_counts))
        else:
            a._training_cost_root = run / 'training-costs'
            require(a.sdpo is not None and a.tokenizer, 'pinned SDPO and tokenizer required')
            revision = subprocess.check_output(['git','-C',str(a.sdpo),'rev-parse','HEAD'], text=True).strip()
            require(revision == '7c457fc1b1f636ae794eb0362ba37d4743b06fbc', 'wrong SDPO revision')
            identity(a.tokenizer, 'student')
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
            converted, token_counts = convert(task, tokenizer, a.stage)
            import pyarrow as pa
            import pyarrow.parquet as pq
            data.mkdir(parents=True, exist_ok=True)
            if not (data / 'manifest.json').exists():
                for part, rows in [('train',converted),('val',converted[:32]),('test',[converted[0]])]:
                    pq.write_table(pa.Table.from_pylist(rows), data / (part + '.parquet'))
                atomic(data / 'manifest.json', files_manifest(data, ['train.parquet','val.parquet','test.parquet'], rows=1280))
            frozen_files(data)
            cmd, resolve, env = command(a, recipe, incoming['model'], data, run)
            bind(run / 'command.json', cmd)
            family = recipe[0] if recipe != 'answer-only' else 'F'
            if not completed_training(run, family, 40):
                if (run / 'started.json').exists():
                    kept = run / 'partials' / str(time.time_ns()); kept.mkdir(parents=True)
                    for name in ('sdft','train','metrics.jsonl','started.json','trainer-exit.json','resolved-config.yaml','resolved-config.err'):
                        if (run / name).exists(): (run / name).rename(kept / name)
                    print('Retained interrupted stage at', kept, '; same seed restarts at step 0')
                with (run / 'resolved-config.yaml').open('w') as output, (run / 'resolved-config.err').open('w') as error:
                    done = subprocess.run(resolve, cwd=a.sdpo, env=env, stdout=output, stderr=error)
                require(done.returncode == 0 and (run / 'resolved-config.yaml').stat().st_size > 0, 'trainer config did not resolve')
                atomic(run / 'started.json', observed); clock = time.monotonic()
                done = subprocess.run(cmd, cwd=a.sdpo, env=env)
                atomic(run / 'trainer-exit.json', dict(returncode=done.returncode, seconds=time.monotonic()-clock))
                if done.returncode: raise subprocess.CalledProcessError(done.returncode, cmd)
                if not completed_training(run, family, 40): raise NotReady('final checkpoint missing; interruption evidence retained')
            from kit.v4_sft import complete_export
            for step in STEPS[1:]:
                export = run / f'hf-step{step}'
                if not complete_export(export):
                    checkpoint_path = run / ('sdft' if family == 'D' else 'train') / f'global_step_{step}'
                    if family == 'D': checkpoint_path /= 'actor'
                    subprocess.run([a.trainer_python,'-m','verl.model_merger','merge','--backend','fsdp','--local_dir',str(checkpoint_path),'--target_dir',str(export)], cwd=a.sdpo, env=env, check=True)
                require(complete_export(export), 'incomplete merged export')
                atomic(run / f'export-{step}.json', identity(export))
            seconds = json.loads((run / 'trained.json').read_text())['seconds']
            atomic(run / 'summary.json', dict(**observed, seconds=seconds, gpu_hours=8*seconds/3600 if seconds is not None else None, scheduled_tokens=token_counts))
        atomic(run / 'manifest.json', files_manifest(run, ['summary.json']+[f'export-{step}.json' for step in STEPS[1:]], technical_synthetic=a.standin))
        atomic(run / 'complete.json', dict(complete=True))


def convert(rows, tokenizer, stage):
    if stage=='legal':
        from kit.c_train import training_rows
        return training_rows(rows,tokenizer,'D+replay')
    from kit.p3b_train import training_rows
    converted=training_rows(rows,tokenizer,'D')
    return converted,dict(input=sum(len(t.tokenize_student(tokenizer,r['prompt'] if isinstance(r['prompt'],list) else [dict(role='user',content=r['prompt'])],r['response'])[0]) for r in rows),
                          response_including_eos=sum(len(r['extra_info']['p3b_target_ids']) for r in converted))


def configure(p):
    p.add_argument('--durable',type=Path)
    device(p,'0,1,2,3,4,5,6,7');p.add_argument('--stage',choices=('legal','math'),required=True)
    p.add_argument('--configuration',choices=CONFIGS,default='generic')
    p.add_argument('--trajectory',choices=('plain','sibling'),default='plain')
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--tokenizer');p.add_argument('--sdpo',type=Path);p.add_argument('--trainer-python',default=sys.executable)
if __name__=='__main__': raise SystemExit(entry(job,configure,'m2-train-{stage}-{seed}'))
