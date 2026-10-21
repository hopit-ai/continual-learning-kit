"""Direct v4 trainer invocation and model merging for one registered B job."""
import subprocess
import time
from kit.p3b_common import *


def check_data(root):
    doc = frozen_files(root)
    require(doc['rows'] in (64, 1280), 'wrong package-B dose')
    return doc


def completed_training(run, recipe, steps):
    """Recover a durable final dose; a successful trainer's bad dose stays a stop."""
    if (run / 'trained.json').exists(): return True
    exit_path=run / 'trainer-exit.json'
    exited=json.loads(exit_path.read_text()) if exit_path.exists() else {}
    metrics=t.read_jsonl(run / 'metrics.jsonl') if (run / 'metrics.jsonl').exists() else []
    counters=[int(value) for row in metrics for key,value in row.get('data',row).items()
              if key in ('v4/completed_optimizer_updates','p3b/completed_optimizer_updates')]
    completed=max(counters,default=None)
    require(completed is None or completed<=steps,'trainer exceeded the registered optimizer dose')
    root=run / ('sdft' if recipe=='D' else 'train')
    checkpoint=root / f'global_step_{steps}'
    if recipe=='D': checkpoint /= 'actor'
    tracker=root / 'latest_checkpointed_iteration.txt'
    saved=None
    if tracker.exists() and completed!=steps:
        text=tracker.read_text().strip()   # an interrupted write can leave it empty or truncated
        saved=int(text) if text.isdigit() else None
    dose=completed==steps or saved==steps
    if exited.get('returncode')==0:
        require(completed==steps,'trainer did not complete the registered optimizer dose')
    # A run that finished its dose but lost its final checkpoint produced no model and no scores, so the caller
    # treats it as interrupted and repeats the identical input (manager decision, Astra round 2): nothing was
    # observed that a repeat could select on.
    if checkpoint.is_dir() and dose:
        atomic(run / 'trained.json',dict(seconds=exited.get('seconds'),recovered_completion=True))
        return True
    return False


def configure(p, recovery=False):
    device(p, '0,1,2,3,4,5,6,7')
    p.add_argument('--trainer-python', default=__import__('sys').executable)
    p.add_argument('--sdpo', type=Path); p.add_argument('--tokenizer')
    p.add_argument('--seed', type=int, choices=range(101,106), required=True)
    p.add_argument('--incoming-receipt', type=Path)
    if recovery:
        p.add_argument('--arm', choices=('D','F','base'), required=True)
        p.add_argument('--size', type=int, choices=(16,64), required=True)
    else:
        p.add_argument('--recipe', choices=('D','F','D+replay','F+replay'), required=True)
        p.add_argument('--route', choices=('finqa','standalone','sequence','reset'), required=True)
        p.add_argument('--task-schedule', type=Path); p.add_argument('--schedule-manifest', type=Path)


def training_rows(rows, tokenizer, recipe, recovery=False):
    from kit.v4_sft import encode_example
    converted = []
    for row in rows:
        extra = row.get('extra_info') or {}
        prompt = row['prompt']; prompt = json.loads(prompt) if not recovery and not row.get('p3b_replay') and isinstance(prompt,str) and prompt.startswith('[') else prompt
        if isinstance(prompt,str): prompt = [dict(role='user', content=prompt)]
        response = row.get('response', extra.get('demonstration'))
        require(isinstance(response,str), 'missing verified target')
        if recipe=='D' and not row.get('p3b_replay'):
            from kit.sdft_objective import teacher_messages, SdftDataError
            try: teacher_messages(prompt,response)
            except SdftDataError as exc: raise CorrectnessStop(str(exc)) from exc
        encode_example(tokenizer, prompt, response)
        pids, tids = t.tokenize_student(tokenizer, prompt, response)
        require(recipe != 'D' or len(pids) <= 2048, 'D prompt exceeds unchanged cap; flag unavoidable science change')
        converted.append(dict(id=str(row.get('id',extra.get('index'))), prompt=prompt,
            data_source=row.get('data_source', {'code':'livecodebench','math_screen':'math_screen'}.get(row.get('task'),'p3b_replay')),
            reward_model=row.get('reward_model',dict(style='rule', ground_truth=row.get('gold',''))),
            extra_info=dict(extra, index=str(row.get('id',extra.get('index'))), split='train', demonstration=response,
                p3b_replay=bool(row.get('p3b_replay')), p3b_target_ids=tids)))
    return converted


def d_replay_rows(task, replay, seed, tokenizer):
    """D alone skips over-cap draws in the original seed-shuffled full pool."""
    skipped = []; eligible = []; excluded = []
    for row in replay:
        prompt = row['prompt']
        if isinstance(prompt, str): prompt = [dict(role='user', content=prompt)]
        pids, _ = t.tokenize_student(tokenizer, prompt, row['response'])
        prompt_tokens = len(pids)
        if prompt_tokens > 2048:
            skipped.append(str(row['id']))
            excluded.append(dict(id=str(row['id']), prompt_tokens=prompt_tokens))
        else: eligible.append(str(row['id']))
    require(bool(eligible), 'no eligible D replay rows')
    sources = {r.get('data_source', {'code':'livecodebench','math_screen':'math_screen'}.get(r.get('task'))) for r in task}
    require(len(sources) == 1 and next(iter(sources)) in ('finqa','math_screen','livecodebench'),
            'D replay needs one supported task reward schema')
    source = next(iter(sources))
    replay = [dict(r, extra_info=dict(r.get('extra_info') or {}, p3b_task_source=source)) for r in replay]
    draws = []
    rows = mixed(task, replay, seed, skip_ids=skipped, skipped_draws=draws)
    return rows, dict(seed=seed, prompt_cap=2048, pool_rows=len(replay), eligible_replay_rows=len(eligible),
                      eligible_replay_ids=eligible, skipped_ids=skipped, skipped=excluded, skipped_draws=draws,
                      skipped_draw_count=len(draws))


def job(a, recovery=False):
    frozen_hashes(); tests_path=None; tokenizer=None; replay_selection=None
    if recovery:
        require(a.seed <= 103 and (a.arm != 'base' or a.seed == 101), 'unregistered recovery seed')
        name = f'recover-{a.arm}-n{a.size}-s{a.seed}'; recipe = 'F'; steps, batch = 4, 16
    else:
        require(a.route != 'finqa' or '+replay' in a.recipe, 'tier-1 new stages use replay')
        name = f'{a.route}-{a.recipe}-s{a.seed}'; recipe = a.recipe[0]; steps, batch = 40, 32
        if getattr(a, 'slot_map', None): name = f'v4-{a.slot_map}-s{a.seed}'
    require(a.standin or len(set(a.gpu.split(',')))==8, 'v4 training uses eight distinct GPUs')
    run = a.out / 'runs' / name; data = a.out / 'data' / name
    if a.standin: incoming = dict(technical_synthetic=True)
    else:
        require(a.sdpo is not None and a.tokenizer, 'SDPO and pinned 8B tokenizer required')
        revision=subprocess.check_output(['git','-C',str(a.sdpo),'rev-parse','HEAD'],text=True).strip()
        require(revision=='7c457fc1b1f636ae794eb0362ba37d4743b06fbc','wrong pinned SDPO checkout')
        require(a.incoming_receipt is not None or (not recovery and a.route in ('finqa','standalone')) or (recovery and a.arm == 'base'), 'incoming export hash receipt required')
        base_start=(not recovery and a.route in ('finqa','standalone')) or (recovery and a.arm=='base')
        incoming = identity(a.model, 'student' if base_start else None, a.incoming_receipt)
        identity(a.tokenizer, 'student')
        if a.incoming_receipt and not base_start:
            origin=json.loads(a.incoming_receipt.read_text()).get('model_dir','')
            if recovery or a.route=='sequence':
                arm=a.arm if recovery else getattr(a, 'start_recipe', a.recipe)
                slot=f'{arm}-finqa-alone-s{a.seed}' if arm in ('D','F') else f'finqa-{arm}-s{a.seed}'
                require(slot in Path(origin).parts and Path(origin).name=='hf-step40','receipt is not the matched original FinQA final')
            elif a.route=='reset':
                require(Path(origin).name=='hf-reset' and Path(origin).parent.name==f'{a.recipe}-s{a.seed}','reset route requires matched reset export receipt')
    if recovery:
        freeze = frozen_files(a.out / 'replay')
        require(freeze['technical_synthetic'] == a.standin, 'stand-in replay cannot train real models')
        rows = cycle(t.read_jsonl(a.out / 'replay/refresher-64.jsonl')[:a.size], a.seed, 64)
    else:
        require(a.task_schedule and a.schedule_manifest, 'frozen schedule and manifest required')
        manifest = json.loads(a.schedule_manifest.read_text())
        expected = manifest.get('files', {}).get(a.task_schedule.name, manifest.get('train_sha256'))
        require(expected is not None, 'schedule absent from manifest'); check(a.task_schedule, expected)
        if 'seed' in manifest: require(manifest['seed'] == a.seed, 'wrong task schedule seed')
        if not a.standin and a.route=='finqa':
            require(manifest.get('schema')=='kit-v4-sft-data.v1' and manifest.get('arm')=='COMMON' and manifest.get('scientific_phase_allowed') is True and not manifest.get('stand_in_teacher') and not manifest.get('technical_synthetic'),'FinQA requires original qualified v4 COMMON schedule')
        if a.task_schedule.suffix == '.parquet':
            import pyarrow.parquet as pq
            rows = pq.read_table(a.task_schedule).to_pylist()
        else: rows = t.read_jsonl(a.task_schedule)
        if a.route=='finqa' and not a.standin:
            require([r['extra_info']['index'] for r in rows]==manifest['ids'],'v4 task schedule order changed')
        if a.route != 'finqa':
            require((a.out / 'targets/manifest.json').exists(), 'tier 2 requires frozen targets')
            require(frozen_files(a.out / 'targets')['technical_synthetic']==a.standin,'stand-in targets cannot train real models')
            check(a.task_schedule, frozen_files(a.out / 'targets')['files'][f'schedule-{a.seed}.jsonl'])
            target_info=json.loads((a.out/'targets/inputs.json').read_text())
            require(a.recipe != 'F+replay' or target_info['task']=='math_screen',
                    'F+replay tier 2 is registered for MATH only')
            if target_info['task']=='code':
                tests_path=Path(target_info['tests_path']);check(tests_path,target_info['tests_sha256'])
        if '+replay' in a.recipe:
            freeze = frozen_files(a.out / 'replay'); require(freeze['technical_synthetic'] == a.standin, 'synthetic replay mismatch')
            replay = [dict(r, p3b_replay=True) for r in t.read_jsonl(a.out / 'replay/replay.jsonl')]
            if getattr(a, 'slot_map', None):
                from kit.next_slots import compose
                if recipe == 'D' and not a.standin:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
                rows, replay_selection = compose(rows, replay, a.pool_root, a.slot_map, a.seed, tokenizer)
            elif a.recipe == 'D+replay' and not a.standin:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
                rows, replay_selection = d_replay_rows(rows, replay, a.seed, tokenizer)
                print('D+replay selection:', json.dumps({k:v for k,v in replay_selection.items()
                      if k != 'eligible_replay_ids'}, sort_keys=True), flush=True)
            else: rows = mixed(rows, replay, a.seed)
    require(len(rows) == steps*batch, 'wrong training exposures')
    observed = dict(incoming=incoming, rows=s.digest(rows), seed=a.seed, name=name, route='recovery' if recovery else a.route, recipe=getattr(a,'recipe','F'), steps=steps, batch=batch, settings_sha256=s.file_hash(R/'v4-recipes.json'), technical_synthetic=a.standin)
    if replay_selection is not None: observed['replay_selection'] = replay_selection
    bind(a.out / 'identities' / (name + '.json'), observed)
    if (run / 'complete.json').exists():
        doc = frozen_files(run)
        require(doc['technical_synthetic'] == a.standin, 'completed output type mismatch')
        if not a.standin:
            for step in ([4] if recovery else [10,20,30,40]):
                identity(run / f'hf-step{step}', receipt=run / f'export-{step}.json')
        print('SKIP:', name); return
    data.mkdir(parents=True, exist_ok=True); run.mkdir(parents=True, exist_ok=True)
    if replay_selection is not None: bind(run / 'replay-selection.json', replay_selection)
    if a.standin:
        for step in ([4] if recovery else [10,20,30,40]): atomic(run / f'export-{step}.json', dict(technical_synthetic=True, step=step))
        atomic(run / 'summary.json', dict(**observed, seconds=0, gpu_hours=0))
    else:
        from transformers import AutoTokenizer
        if tokenizer is None: tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        import pyarrow as pa
        import pyarrow.parquet as pq
        converted = training_rows(rows,tokenizer,recipe,recovery)
        if not (data / 'manifest.json').exists():
            for file, values in [('train',converted),('val',converted[:batch]),('test',[dict(converted[0],extra_info={'index':'diagnostic','split':'test'})])]: pq.write_table(pa.Table.from_pylist(values),data / (file+'.parquet'))
            atomic(data / 'manifest.json', files_manifest(data,['train.parquet','val.parquet','test.parquet'],rows=len(rows)))
        check_data(data)
        overrides = json.loads((R / 'v4-recipes.json').read_text())['overrides'][recipe]
        overrides = [x.format(kit=KIT, sdpo=a.sdpo.resolve(), model=Path(a.model).resolve(), data=data, run=run, seed=a.seed) for x in overrides]
        changes = {'trainer.total_training_steps':str(steps),'data.train_batch_size':str(batch),'trainer.save_freq':str(4 if recovery else 10)}
        if recipe == 'D': changes['custom_reward_function.path'] = str(KIT / 'p3b_reward.py')
        if recovery: changes.update({'+optim.v4_warmup_steps':'0'})
        overrides = [x.split('=')[0]+'='+changes.get(x.split('=')[0],x.split('=',1)[1]) for x in overrides]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONPATH=str(KIT.parent)+':'+str(a.sdpo.resolve()),
            P3B_DATA=str(data), P3B_REPLAY='1' if '+replay' in getattr(a,'recipe','') else '0', P3B_SFT='1',
            P3B_RECOVERY='1' if recovery else '0', KIT_SFT_ARM_F='0', KIT_SFT_SEED=str(a.seed), ARM='D',
            KIT_FINISH_GATE='1', KIT_V4_ARM=recipe, V4_CHECKPOINT_STATE='1', V4_PROFILE='scientific', V4_TELEMETRY='1', SDPO_DIR=str(a.sdpo.resolve()), QWEN3_8B_TOKENIZER=a.tokenizer,
            WANDB_MODE='disabled', VERL_FILE_LOGGER_PATH=str(run / 'metrics.jsonl'), USER=os.environ.get('USER','partner'), EXPERIMENT=name, TASK='p3b')
        if tests_path is not None: env['CODE_TESTS']=str(tests_path)
        command = [a.trainer_python, str(KIT / 'sdft_entry.py')] if recipe == 'D' else [a.trainer_python,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',str(KIT / 'sft_entry.py')]
        if recipe == 'D':
            overrides += [f'+ray_kwargs.ray_init.runtime_env.env_vars.{key}={json.dumps(env[key])}' for key in ('P3B_DATA','P3B_REPLAY','SDPO_DIR','QWEN3_8B_TOKENIZER','KIT_FINISH_GATE','KIT_V4_ARM','V4_CHECKPOINT_STATE','V4_TELEMETRY','ARM','V4_PROFILE','PYTHONPATH','VERL_FILE_LOGGER_PATH')+(('CODE_TESTS',) if tests_path is not None else ())]
        command += ['--config-name','sdpo' if recipe == 'D' else 'sft_trainer',*overrides]
        bind(run / 'command.json',command)
        if not completed_training(run,recipe,steps):
            if (run / 'started.json').exists():
                # Only interrupted training reaches here; preserve evidence and
                # repeat the original input from step 0 with the same seed.
                kept = run / 'partials' / ('attempt-' + str(time.time_ns())); kept.mkdir(parents=True)
                for name in ('sdft', 'train', 'metrics.jsonl', 'started.json', 'trainer-exit.json', 'resolved-config.yaml', 'resolved-config.err'):
                    if (run / name).exists(): (run / name).rename(kept / name)
                print('WARNING: unfinished trainer moved to', kept, '- restarting this stage from step 0 with the same seed')
            with (run / 'resolved-config.yaml').open('w') as output, (run / 'resolved-config.err').open('w') as error:
                resolved=subprocess.run([a.trainer_python,str(KIT / ('sdft_entry.py' if recipe=='D' else 'sft_entry.py')),'--config-name','sdpo' if recipe=='D' else 'sft_trainer','--cfg','job','--resolve',*overrides],cwd=a.sdpo,env=env,stdout=output,stderr=error)
            require(resolved.returncode==0 and (run/'resolved-config.yaml').stat().st_size>0,'trainer config did not resolve; see resolved-config.err')
            atomic(run / 'started.json',observed); started=time.monotonic()
            trained=subprocess.run(command,cwd=a.sdpo,env=env)
            atomic(run / 'trainer-exit.json',dict(returncode=trained.returncode,seconds=time.monotonic()-started))
            if trained.returncode: raise subprocess.CalledProcessError(trained.returncode,command)
            if not completed_training(run,recipe,steps): raise NotReady('trainer has no final checkpoint; retain interruption evidence')
        from kit.v4_sft import complete_export
        for step in ([4] if recovery else [10,20,30,40]):
            export = run / f'hf-step{step}'
            if not complete_export(export):
                checkpoint = run / ('sdft' if recipe == 'D' else 'train') / f'global_step_{step}'
                if recipe == 'D': checkpoint /= 'actor'
                subprocess.run([a.trainer_python,'-m','verl.model_merger','merge','--backend','fsdp','--local_dir',str(checkpoint),'--target_dir',str(export)],cwd=a.sdpo,env=env,check=True)
            require(complete_export(export),'incomplete merged export')
            atomic(run / f'export-{step}.json',identity(export))
        seconds=json.loads((run / 'trained.json').read_text())['seconds']
        atomic(run / 'summary.json',dict(**observed,seconds=seconds,gpu_hours=seconds*8/3600 if seconds is not None else None))
    names = ['summary.json'] + (['replay-selection.json'] if replay_selection is not None else [])
    details = dict(replay_selection=replay_selection) if replay_selection is not None else {}
    atomic(run / 'manifest.json',files_manifest(run,names,technical_synthetic=a.standin,**details))
    atomic(run / 'complete.json',dict(complete=True))
