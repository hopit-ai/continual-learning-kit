#!/usr/bin/env python3
"""One bound C arm/start/seed, full weights, 40 updates; partner owns sbatch."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.c_common import *
from kit.p3b_train import completed_training
from kit.c_data import encode
import subprocess
import time


def training_rows(rows, tokenizer, recipe):
    from kit.sdft_objective import teacher_messages
    converted = []; tokens = dict(input=0, response_including_eos=0, task_response_including_eos=0, replay_response_including_eos=0)
    for row in rows:
        prompt = row['prompt']; response = row['response']; replay = bool(row.get('p3b_replay'))
        if isinstance(prompt, str): prompt = [dict(role='user', content=prompt)]
        encode(tokenizer, prompt, response)
        pids, tids = t.tokenize_student(tokenizer, prompt, response)
        if recipe.startswith('D') and not replay:
            privileged = teacher_messages(prompt, response)
            rendered = tokenizer.apply_chat_template(privileged, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            require(len(rendered) <= 14336, 'C privileged context over cap; never truncate')
        tokens['input'] += len(pids); tokens['response_including_eos'] += len(tids)
        tokens['replay_response_including_eos' if replay else 'task_response_including_eos'] += len(tids)
        converted.append(dict(id=str(row['id']), prompt=prompt, data_source='p3b_replay' if replay else 'contract_nli',
            reward_model=dict(style='rule', ground_truth='' if replay else row['gold']),
            extra_info=dict(index=str(row['id']), split='train', demonstration=response, p3b_replay=replay,
                            p3b_task_source='contract_nli', p3b_target_ids=tids)))
    return converted, tokens


def command(a, recipe, model, data, run):
    family = 'D' if recipe.startswith('D') else 'F'
    overrides = json.loads((R / 'v4-recipes.json').read_text())['overrides'][family]
    overrides = [x.format(kit=KIT, sdpo=a.sdpo.resolve(), model=Path(model).resolve(), data=data, run=run, seed=a.seed) for x in overrides]
    changes = {'data.max_prompt_length':'10240', 'data.max_length':'12288',
               'data.custom_cls.path':str(KIT / 'c_data.py'), 'data.custom_cls.name':'CSFTDataset',
               'custom_reward_function.path':str(KIT / 'c_reward.py'),
               'actor_rollout_ref.actor.self_distillation.max_reprompt_len':'14336'}
    overrides = [x.split('=',1)[0]+'='+changes.get(x.split('=',1)[0].lstrip('+'),x.split('=',1)[1]) for x in overrides]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONPATH=str(KIT.parent)+':'+str(a.sdpo.resolve()),
        P3B_DATA=str(data), P3B_REPLAY='1' if '+replay' in recipe else '0', P3B_SFT='1', P3B_RECOVERY='0',
        KIT_SFT_ARM_F='0', KIT_SFT_SEED=str(a.seed), KIT_FINISH_GATE='1', ARM='D', KIT_V4_ARM=family,
        V4_CHECKPOINT_STATE='1', V4_PROFILE='scientific', V4_TELEMETRY='1', SDPO_DIR=str(a.sdpo.resolve()),
        QWEN3_8B_TOKENIZER=a.tokenizer, WANDB_MODE='disabled', VERL_FILE_LOGGER_PATH=str(run / 'metrics.jsonl'),
        USER=os.environ.get('USER','partner'), EXPERIMENT=run.name, TASK='c')
    entry_path = str(KIT / ('c_sdft_entry.py' if family == 'D' else 'sft_entry.py'))
    if family == 'D':
        # All variables consumed in Ray's runner/actors must be forwarded explicitly.
        keys = ('P3B_DATA','P3B_REPLAY','SDPO_DIR','QWEN3_8B_TOKENIZER','KIT_FINISH_GATE','KIT_V4_ARM',
                'V4_CHECKPOINT_STATE','V4_TELEMETRY','ARM','V4_PROFILE','PYTHONPATH','VERL_FILE_LOGGER_PATH')
        overrides += [f'+ray_kwargs.ray_init.runtime_env.env_vars.{key}={json.dumps(env[key])}' for key in keys]
        cmd = [a.trainer_python, entry_path]
    else:
        cmd = [a.trainer_python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=8', entry_path]
    args = ['--config-name', 'sdpo' if family == 'D' else 'sft_trainer', *overrides]
    resolve = [a.trainer_python, entry_path, *args, '--cfg', 'job', '--resolve']
    return cmd+args, resolve, env


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
    probe = getattr(a, 'probe_arm', None)
    require(not probe or a.route == 'sequence', 'probe is after-FinQA only')
    recipe = probe_recipe(doc) if probe else doc['W'] if a.recipe == 'W' else a.recipe
    arm = probe_arm(doc, probe) if probe else f'{a.route}-{recipe}'
    incoming_recipe = 'probe-' + probe if probe else recipe
    if probe == 'generic' and doc['W'] == recipe: probe = None  # exact shared W run, including identity
    print('W =', doc['W'], '(provisional)' if doc['provisional'] else '', '; requested arm:', recipe, '\n' + doc['why'])
    incoming = start(doc, incoming_recipe, a.route, a.seed)
    for panel in ('panel','test','finqa','general-v1'):
        baseline = a.out / 'scores' / f'{arm}-s{a.seed}-k0-{panel}'
        if not (baseline / 'manifest.json').exists(): raise NotReady('score incoming panels before training: ' + str(baseline))
        frozen_files(baseline)
    require(a.standin or len(set(a.gpu.split(','))) == 8, 'training uses eight distinct GPUs')
    observed_model = checked_model(incoming, a.standin, base=a.route == 'standalone')
    targets = a.out / 'targets'; manifest = frozen_files(targets)
    frozen_files(targets / 'prepared')
    require(manifest['technical_synthetic'] == a.standin, 'synthetic target mismatch')
    require(manifest['protocol'] == protocol(), 'target protocol changed')
    task = t.read_jsonl(targets / f'schedule-{a.seed}.jsonl')
    require(len(task) == 1280, 'C schedule must contain 1280 exposures')
    if recipe == 'answer-only': task = [dict(r, response='Answer: ' + r['gold']) for r in task]
    replay_identity = None
    if '+replay' in recipe:
        require(a.replay is not None, 'B frozen replay directory required')
        replay_manifest = frozen_files(a.replay)
        require(replay_manifest['technical_synthetic'] == a.standin, 'synthetic replay mismatch')
        replay = [dict(r, p3b_replay=True) for r in t.read_jsonl(a.replay / 'replay.jsonl')]
        evaluation = [r for values in panels().values() for r in values]
        _, excluded = overlap(replay, evaluation)
        require(not excluded, 'B replay overlaps C evaluation; frozen pool must not be silently replaced')
        # B D retains its 2048-token generic prompt cap and full-pool skip order.
        skipped = []
        if not a.standin and recipe == 'D+replay':
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
            for r in replay:
                msgs = r['prompt'] if isinstance(r['prompt'], list) else [dict(role='user',content=r['prompt'])]
                if len(t.tokenize_student(tok, msgs, r['response'])[0]) > 2048: skipped.append(str(r['id']))
        skipped_draws = []
        if probe:
            from kit.next_slots import compose
            require(a.pool_root is not None, 'frozen probe pools required')
            tokenizer = tok if not a.standin and recipe.startswith('D') else None
            task, slot_audit = compose(task, replay, a.pool_root, probe, a.seed, tokenizer, task_name='contract_nli')
        else:
            task = mixed(task, replay, a.seed, skip_ids=skipped, skipped_draws=skipped_draws)
        replay_identity = dict(manifest=s.file_hash(a.replay / 'manifest.json'), skipped_ids=skipped, skipped_draws=skipped_draws)
    name = f'{arm}-s{a.seed}'; run = a.out / 'runs' / name; data = a.out / 'data' / name
    if probe and not (probe == 'generic' and doc['W'] == recipe):
        replay_identity['slot_audit'] = slot_audit
        replay_identity['pools'] = s.file_hash(a.pool_root / 'manifest.json')
    observed = dict(incoming=observed_model, rows=s.digest(task), seed=a.seed, recipe=recipe, route=a.route,
        steps=40, batch=32, task_exposures=slot_audit['exposures']['contract_nli'] if probe else 1120 if '+replay' in recipe else 1280,
        replay_exposures=1280-slot_audit['exposures']['contract_nli'] if probe else 160 if '+replay' in recipe else 0, replay=replay_identity,
        binding_sha256=s.file_hash(a.out / 'binding.json'), target_manifest=s.file_hash(targets / 'manifest.json'),
        protocol=protocol(), settings_sha256=s.file_hash(R / 'v4-recipes.json'), technical_synthetic=a.standin)
    with lock(run):
        bind(a.out / 'identities' / (name + '.json'), observed)
        if (run / 'complete.json').exists():
            frozen_files(run)
            if not a.standin:
                for step in STEPS[1:]: identity(run / f'hf-step{step}', receipt=run / f'export-{step}.json')
            print('SKIP:', name); return
        if a.standin:
            from kit.c_sim import Tokenizer
            converted, token_counts = training_rows(task, Tokenizer(), recipe)
            for step in STEPS[1:]: atomic(run / f'export-{step}.json', dict(technical_synthetic=True, step=step))
            atomic(run / 'summary.json', dict(**observed, seconds=0, gpu_hours=0, scheduled_tokens=token_counts))
        else:
            a._training_cost_root = run / 'training-costs'
            require(a.sdpo is not None and a.tokenizer, 'pinned SDPO and tokenizer required')
            revision = subprocess.check_output(['git','-C',str(a.sdpo),'rev-parse','HEAD'], text=True).strip()
            require(revision == '7c457fc1b1f636ae794eb0362ba37d4743b06fbc', 'wrong SDPO revision')
            identity(a.tokenizer, 'student')
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
            converted, token_counts = training_rows(task, tokenizer, recipe)
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


def configure(p):
    device(p, '0,1,2,3,4,5,6,7'); p.add_argument('--recipe', choices=('W',*RECIPES,'answer-only'), required=True)
    p.add_argument('--route', choices=('standalone','sequence'), required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True); p.add_argument('--replay', type=Path)
    p.add_argument('--probe-arm', choices=('generic','broad','earlier','both')); p.add_argument('--pool-root', type=Path)
    p.add_argument('--tokenizer'); p.add_argument('--sdpo', type=Path); p.add_argument('--trainer-python', default=sys.executable)
if __name__ == '__main__': raise SystemExit(entry(job, configure, 'c-train-{route}-{recipe}-{seed}'))
