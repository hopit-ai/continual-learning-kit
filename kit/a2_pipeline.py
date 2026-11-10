"""The same chronology, schedules and verifiers serve native runs and CPU fixtures."""
import os
import subprocess
import sys
import time
from collections import Counter
from types import SimpleNamespace as NS
from kit.a2_common import *
from kit.a2_bind import binding, bind_inputs, batch, ledger, contexts
from kit import a4_memory, a4_score, a4_train
from kit.beds import finqa


def tok_for(doc):
    if doc['technical_synthetic']:
        from kit.c_sim import Tokenizer
        return Tokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(doc['inputs']['base'], local_files_only=True)


def stage(out, row):
    return Path(out)/'runs'/f'{row["model"]}-s{row["seed"]}-m{row["month"]}'


def state(out, doc, model, seed, through):
    if model == 'P-replay' and doc.get('P_applicable') is False and doc['arm'] == 'RP':
        model = 'raw-plain'
    for n in reversed(BOUNDARIES):
        if n > through:
            continue
        root = stage(out, dict(model=model, seed=seed, month=n))
        if not (root/'summary.json').exists():
            raise NotReady('preceding step-40 student export absent')
        frozen_files(root)
        if not load(root/'summary.json')['skipped']:
            return dict(model=str(root/'hf-step40'), receipt=str(root/'export-40.json'))
    if doc['arm'] == 'J':
        lineage = doc['connected']['lineages'][str(seed)]
        checked(lineage['stages'][-1]['export'])
        return dict(model=lineage['terminal'], receipt=lineage['stages'][-1]['export']['path'])
    return dict(model=doc['inputs']['base'])


def model_identity(record, doc):
    return dict(record, technical_synthetic=True) if doc['technical_synthetic'] else identity(record['model'], receipt=record.get('receipt'))


def local_contexts(out, month):
    root = Path(out)/'contexts'/month
    frozen_files(root)
    return {r['id']: r for r in t.read_jsonl(root/'prompts.jsonl')}


def members(out, doc, panel, mode, holdouts=False):
    if panel in MONTHS:
        ctx = local_contexts(out, panel)
        return [dict(id=i['id'], panel=panel, item=i, prompt=ctx[i['id']][mode]['prompt'], context=ctx[i['id']])
                for i in batch(doc, panel) if not holdouts or i['sec']['issuer_holdout']]
    return [r for r in t.read_jsonl(checked(doc['inputs']['general'])) if r['panel'] == panel]


def prepare_scores(cfg, doc, row, attempt):
    from kit.a2_read import score_plan
    out = Path(cfg['out'])
    if row['kind'] == 'arrival':
        n = row['month']
        specs = [k for k in score_plan(doc['arm']) if k[0] == 'arrival' and k[3] == n and k[1] != 'frozen-reference']
        if doc['arm'] == 'J':
            specs += [('collector', 'J-current', seed, n, MONTHS[n-1], 'off') for seed in SEEDS]
    else:
        specs = score_plan(doc['arm'])
    prepared = []
    for key in specs:
        kind, model, seed, n, panel, mode = key
        record = dict(model=doc['inputs']['base']) if model == 'frozen-reference' else state(out, doc, model, seed, n-1 if kind in ('arrival', 'collector') else n)
        data = members(out, doc, panel, mode, kind not in ('arrival', 'collector'))
        prepared.append(dict(key=key, record=record, model=model_identity(record, doc), data=data))
    atomic(attempt/'prepared.json', prepared)
    return [attempt/'prepared.json']


def score(cfg, doc, row, attempt):
    from kit.a2_jobs import completed
    from kit.a4_lane import physical
    out = Path(cfg['out'])
    prepared = load(completed(out, row['depends_on'][0]['id'])/'prepared.json')
    synthetic = doc['technical_synthetic']
    # a4_lane measures the actual visible physical GPU and runtime environment.
    os.environ['A4_ALLOCATION_NONCE'] = os.environ['A2_ALLOCATION_NONCE']
    measured = physical(synthetic)
    cells = []
    for index, spec in enumerate(prepared):
        kind, model, seed, n, panel, mode = spec['key']
        raw_path = attempt/'raw'/f'{index}.jsonl'
        if raw_path.exists():
            raw = t.read_jsonl(raw_path)
        else:
            try:
                raw = a4_score.generate(spec['data'], spec['record'], os.environ['CUDA_VISIBLE_DEVICES'], synthetic, model, kind)
            finally:
                from kit.inference_lifecycle import shutdown_since
                shutdown_since()
            write_jsonl(raw_path, raw)
        require(physical(synthetic) == measured, 'scoring hardware changed')
        graded = a4_score.graded(spec['data'], raw, dict(kind=kind, arm=model, seed=seed, month=n,
                                 mode=mode, generation_id=digest(spec['model'])), synthetic)
        cells.append(dict(key=spec['key'], physical=measured, rows=graded,
                          expected_members={i['id']: dict(holdout=bool(i.get('item', {}).get('sec', {}).get('issuer_holdout', False)),
                                                         family=i.get('item', {}).get('sec', {}).get('template', '')) for i in spec['data']},
                          model=spec['model'], prompts_sha256=digest(spec['data']), raw_sha256=s.file_hash(raw_path)))
    atomic(attempt/'cells.json', cells)
    return [attempt/'cells.json', *sorted((attempt/'raw').glob('*.jsonl'))]


def feedback(cfg, doc, row, attempt):
    from kit.a2_jobs import completed
    out = Path(cfg['out'])
    source = completed(out, row['depends_on'][0]['id'])
    cells = load(source/'cells.json')
    month = MONTHS[row['month']-1]
    # Seal every whole-month deployed prediction and collector before admission.
    atomic(out/'seals'/(month+'.json'), dict(predictions_sha256=s.file_hash(source/'cells.json'),
                                           all_month_predictions_sealed=True, month=month))
    paths = [out/'seals'/(month+'.json')]
    if doc['arm'] != 'J':
        return paths
    tok = tok_for(doc)
    lookup = {i['id']: i for i in batch(doc, month)}
    for seed in SEEDS:
        index = next(i for i, c in enumerate(cells) if c['key'] == ['collector', 'J-current', seed, row['month'], month, 'off'])
        raw = t.read_jsonl(source/'raw'/f'{index}.jsonl')
        require(len(raw) == len(lookup) and {r['id'] for r in raw} == set(lookup), 'collector coverage')
        traces, decisions = [], []
        for response in raw:
            item = lookup[response['id']]
            reasons = __import__('kit.a4_ledger', fromlist=['admission']).admission(item, response, tok)
            decisions.append(dict(id=item['id'], reasons=reasons, admitted=not reasons, family=item['sec']['template']))
            if reasons:
                continue
            prompt = local_contexts(out, month)[item['id']]['off']['prompt']
            traces.append(dict(id=item['id'], month=month, item=item, response=response['response'],
                source=item['filename'], date=item['sec']['filing_date'],
                collector_checkpoint=cells[index]['model'], prompt_sha256=digest(prompt.encode()),
                response_sha256=digest(response['response'].encode()), raw_sha256=digest(response),
                collector_predictions_sha256=s.file_hash(source/'cells.json'), verifier=doc['protocol']))
        root = out/'current-ledger'/str(seed)/month
        write_jsonl(root/'traces.jsonl', traces)
        write_jsonl(root/'admission.jsonl', decisions)
        freeze(root, ['traces.jsonl', 'admission.jsonl'])
        paths += [root/'manifest.json', root/'traces.jsonl', root/'admission.jsonl']
    return paths


def prepare_training(cfg, doc, row, attempt):
    out = Path(cfg['out'])
    run = stage(out, row)
    n, seed, model = row['month'], row['seed'], row['model']
    pool = ledger(doc, n)
    if model == 'J-current':
        pool = []
        for month in MONTHS[:n]:
            root = out/'current-ledger'/str(seed)/month
            frozen_files(root)
            pool += t.read_jsonl(root/'traces.jsonl')
    task = [dict(id=r['id'], prompt=[dict(role='user', content=finqa.render_prompt(r['item']))],
                 response=r['response'], gold=finqa.gold_of(r['item'])) for r in pool]
    rows = cycle(task, seed, 1280) if task else []
    audit = dict(unique_traces=len(task), repetitions=dict(Counter(r['id'] for r in rows)),
                 scheduled_task=1280 if rows else 0, scheduled_generic=0)
    if model == 'P-replay' and rows:
        from kit.a2_replay import transfer
        evaluation = [dict(id=i['id'], prompt=finqa.render_prompt(i)) for m in MONTHS for i in batch(doc, m) if i['sec']['issuer_holdout']]
        evaluation += t.read_jsonl(checked(doc['inputs']['general']))
        rows, replay_audit = transfer(rows, doc['i0'], seed, evaluation)
        audit.update(replay=replay_audit, scheduled_task=replay_audit['task_positions'], scheduled_generic=replay_audit['generic_positions'])
    tok = tok_for(doc)
    converted, counts = a4_train.training_rows(rows, tok, 'D+replay' if model == 'P-replay' else 'D') if rows else ([], {})
    # First validate every raw position; a short rewrite can never activate one.
    if model == 'R-rewrite':
        audit.update(rewrite_accepted_positions=0, rewrite_raw_fallback_positions=0)
        for raw, item in zip(rows, converted):
            cached = rewrite_cache(out, doc, raw['id'])
            if cached is None:
                raise NotReady('rewrite cache entry absent: '+raw['id'])
            if cached['accepted']:
                candidate = dict(raw, response=cached['response'])
                try:
                    replacement, _ = a4_train.training_rows([candidate], tok, 'D')
                    item['extra_info']['demonstration'] = replacement[0]['extra_info']['demonstration']
                    audit['rewrite_accepted_positions'] += 1
                except CorrectnessStop:
                    audit.setdefault('stage_length_fallbacks', []).append(raw['id'])
                    audit['rewrite_raw_fallback_positions'] += 1
            else:
                audit['rewrite_raw_fallback_positions'] += 1
    data = out/'data'/run.name
    write_jsonl(data/'schedule.jsonl', rows)
    teaching = {}
    for item in converted:
        key = (item['id'], item['extra_info']['demonstration'], item['extra_info']['p3b_replay'])
        teaching[key] = dict(id=item['id'], demonstration=item['extra_info']['demonstration'],
                            replay=item['extra_info']['p3b_replay'])
    write_jsonl(data/'teaching-contexts.jsonl', list(teaching.values()))
    incoming = state(out, doc, model, seed, n-1)
    prepared = dict(empty=not rows, incoming=incoming, audit=audit, counts=counts,
                    incoming_identity=model_identity(incoming, doc),
                    rows_sha256=digest(rows), recipe='D+replay' if model == 'P-replay' else 'D')
    if rows and not doc['technical_synthetic']:
        identity(cfg['tokenizer'], 'student')
        import pyarrow as pa
        import pyarrow.parquet as pq
        for part, rr in [('train', converted), ('val', converted[:32]), ('test', [dict(converted[0], extra_info=dict(converted[0]['extra_info'], demonstration=''))])]:
            pq.write_table(pa.Table.from_pylist(rr), data/(part+'.parquet'))
        a = NS(sdpo=Path(cfg['sdpo']), tokenizer=cfg['tokenizer'], seed=seed,
               trainer_python=cfg['trainer_python'], gpu=','.join(map(str, range(8))))
        require(subprocess.check_output(['git', '-C', cfg['sdpo'], 'rev-parse', 'HEAD'], text=True).strip() == '7c457fc1b1f636ae794eb0362ba37d4743b06fbc', 'wrong SDPO revision')
        command, resolve, env = a4_train.command(a, prepared['recipe'], incoming['model'], data, run)
        run.mkdir(parents=True, exist_ok=True)
        with (run/'resolved-config.yaml').open('w') as stdout, (run/'resolved-config.err').open('w') as stderr:
            subprocess.run(resolve, cwd=a.sdpo, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=stdout, stderr=stderr, check=True)
        prepared.update(command=command, env={k: v for k, v in env.items() if os.environ.get(k) != v}, cwd=str(a.sdpo))
    atomic(data/'prepared.json', prepared)
    names = ['schedule.jsonl', 'teaching-contexts.jsonl', 'prepared.json'] + (['train.parquet', 'val.parquet', 'test.parquet'] if rows and not doc['technical_synthetic'] else [])
    freeze(data, names)
    return [data/name for name in [*names, 'manifest.json']]


def train(cfg, doc, row, attempt):
    out = Path(cfg['out'])
    run = stage(out, row)
    data = out/'data'/run.name
    frozen_files(data)
    prepared = load(data/'prepared.json')
    if prepared['empty']:
        atomic(run/'trained.json', dict(skipped=True))
    elif doc['technical_synthetic']:
        marker = run/'injected.json'
        if cfg.get('fail_task') == row['id'] and not marker.exists():
            atomic(marker, dict(injected=True))
            raise RuntimeError('CPU fixture interrupted training')
        atomic(run/'trained.json', dict(updates=40, fresh_optimizer=True, incoming_ema=True))
    else:
        from kit.p3b_train import completed_training
        if not completed_training(run, 'D', 40):
            # Retain failed native states, restart unchanged at update zero.
            if (run/'started.json').exists():
                partial = run/'partials'/attempt.name
                partial.mkdir(parents=True)
                for name in ('sdft', 'train', 'metrics.jsonl', 'token-journals', 'started.json'):
                    if (run/name).exists():
                        (run/name).rename(partial/name)
            atomic(run/'started.json', dict(seed=row['seed'], incoming=prepared['incoming']))
            env = dict(os.environ, **prepared['env'])
            env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
            from kit.a4_common import gpu_run
            gpu_run(prepared['command'], cwd=prepared['cwd'], env=env, check=True)
        if not completed_training(run, 'D', 40):
            raise NotReady('native 40-update completion pending')
        atomic(run/'trained.json', dict(updates=40))
    from kit.a2_budget import timing_setup
    atomic(attempt/'timing-setup.json', timing_setup(cfg, doc, row, prepared))
    return [run/'trained.json', attempt/'timing-setup.json']


def export(cfg, doc, row):
    out = Path(cfg['out'])
    run = stage(out, row)
    prepared = load(out/'data'/run.name/'prepared.json')
    load(run/'trained.json')
    names = ['summary.json']
    if not prepared['empty']:
        if doc['technical_synthetic']:
            exported = dict(model=str(run/'hf-step40'), step=40, student=True, technical_synthetic=True)
        else:
            from kit.v4_sft import complete_export
            target = run/'hf-step40'
            if not complete_export(target):
                subprocess.run([cfg['trainer_python'], '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                    '--local_dir', str(run/'sdft/global_step_40/actor'), '--target_dir', str(target)],
                    cwd=cfg['sdpo'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
            require(complete_export(target), 'complete student export absent')
            exported = identity(target)
        atomic(run/'export-40.json', exported)
        names.append('export-40.json')
    runtime = {}
    if not doc['technical_synthetic'] and not prepared['empty']:
        journals = [load(p) for p in (run/'token-journals').glob('*.json')]
        require(journals, 'native D runtime token journals absent')
        runtime['tokens'] = {key: sum(j.get(key, 0) for j in journals) for key in
                            ('student_tokens', 'student_output_tokens', 'teacher_tokens', 'replay_target_tokens')}
        runtime['loss_metrics'] = [r.get('data', r) for p in run.glob('metrics.jsonl') for r in t.read_jsonl(p)]
    atomic(run/'summary.json', dict(skipped=prepared['empty'], incoming=prepared['incoming'],
                                    audit=prepared['audit'], counts=prepared['counts'], runtime=runtime,
                                    updates=0 if prepared['empty'] else 40))
    freeze(run, names)
    return [run/name for name in [*names, 'manifest.json']]


def rewrite_cache(out, doc, item_id):
    path = Path(out)/'rewrites'/('index-'+digest(item_id)+'.json')
    return load(path) if path.exists() else None


def rewrite_prepare(cfg, doc, row, attempt):
    # This is the only adapter that handles ledger items. Its output is a strict whitelist.
    month = MONTHS[row['month']-1]
    ctx = contexts(doc, month)
    prepared = []
    source_binding = doc['arrival_context_bindings'][month]
    for trace in ledger(doc, row['month']):
        if trace['month'] != month:
            continue
        prompt = ctx.get(trace['id'], {}).get('off', {}).get('prompt')
        prepared.append(dict(id=trace['id'], prompt=prompt, response=trace['response'], admitted=True,
                             prompt_sha256=source_binding['off_prompt_sha256'].get(trace['id']),
                             context_binding=source_binding['manifest'],
                             contexts_sha256=source_binding['prompts_sha256']))
    atomic(attempt/'rewrite-inputs.json', prepared)
    return [attempt/'rewrite-inputs.json']


def rewrite_generate(cfg, doc, row, attempt):
    from kit.a2_jobs import completed
    from kit.a2_rewrite import rewrite, unique_binding
    out = Path(cfg['out'])
    prepared = load(completed(out, row['depends_on'][0]['id'])/'rewrite-inputs.json')
    prepared = prepared[row.get('trace_start', 0):row.get('trace_stop', len(prepared))]
    tok = tok_for(doc)
    engine = None
    def fits(prompt, response):
        from kit.eval_bed import render_messages
        from kit.sdft_objective import teacher_messages
        if response is None:
            return len(tok.encode(render_messages(tok, prompt), add_special_tokens=False)) <= 10240
        p = [dict(role='user', content=prompt)]
        _, tids = t.tokenize_student(tok, p, response)
        return len(tids) <= 2048 and len(tok.encode(render_messages(tok, teacher_messages(p, response)), add_special_tokens=False)) <= 14336
    def generate(messages):
        nonlocal engine
        if doc['technical_synthetic']:
            from kit.a2_rewrite import strict_json
            trace = strict_json(messages[1]['content'].split('\n', 1)[1])
            binding = unique_binding(trace)
            atomic(marker, dict(one_attempt=True, key=key))
            return dict(response=canonical(dict(family=trace['family'], cells={k: c['id'] for k, c in binding.items()})).decode(), finish_reason='stop')
        if engine is None:
            from kit.inference_lifecycle import create_llm
            from kit import score_forgetting as general
            engine = create_llm(model=doc['inputs']['base'], enable_lora=False,
                                **dict(general.ENGINE, max_model_len=12288), enforce_eager=True, seed=0)
        from kit.eval_bed import render_messages
        from kit import score_forgetting as general
        from vllm import SamplingParams
        inputs = [render_messages(tok, messages)]
        params = SamplingParams(n=1, **general.DECODING)
        atomic(marker, dict(one_attempt=True, key=key))
        result = engine.generate(inputs, params)[0].outputs[0]
        return dict(response=result.text, finish_reason=result.finish_reason)
    paths = []
    try:
        for trace in prepared:
            key = digest(dict(prompt=trace['prompt_sha256'],
                              supplied_prompt=digest(trace['prompt'].encode()) if trace['prompt'] is not None else None,
                              response=digest(trace['response'].encode()), spec=REGISTRATION))
            cache = out/'rewrites'/(key+'.json')
            if cache.exists():
                value = load(cache)
            else:
                marker = cache.with_suffix('.started.json')
                if marker.exists():
                    value = dict(response=trace['response'], accepted=False, fallback_reason='generation_interrupted_no_retry',
                                 generation_attempted=True, raw_sha256=digest(trace['response'].encode()),
                                 rewrite_sha256=digest(trace['response'].encode()))
                else:
                    value = rewrite(trace['prompt'], trace['response'], trace['admitted'], trace['prompt_sha256'], generate, fits)
                atomic(cache, value)
            index = out/'rewrites'/('index-'+digest(trace['id'])+'.json')
            value = dict(value, id=trace['id'], month=MONTHS[row['month']-1],
                         context_binding=trace['context_binding'], contexts_sha256=trace['contexts_sha256'],
                         bound_prompt_sha256=trace['prompt_sha256'])
            atomic(index, value)
            paths += [cache, index]
    finally:
        from kit.inference_lifecycle import shutdown_since
        shutdown_since()
    atomic(attempt/'rewrite-report.json', dict(attempts=len(prepared), results=[load(p) for p in paths[1::2]]))
    return [*paths, attempt/'rewrite-report.json']


def operation(cfg, row, attempt):
    out = Path(cfg['out'])
    phase = row['phase']
    if phase == 'price':
        from kit.a2_budget import price
        return dict(status='COMPLETE', artifacts=[str(p) for p in price(cfg, out)])
    if phase == 'collect':
        from kit.a2_budget import gate
        try:
            gate(cfg, out)
        except NotReady:
            from kit.a2_return import qualification_return
            return dict(status='COMPLETE', artifacts=[str(p) for p in qualification_return(cfg, out)])
    if phase == 'bind':
        require((out/'binding.json').exists() or not (out/'runs').exists() or
                not any((out/'runs').iterdir()), 'fresh A2 trajectories required; imported run directories forbidden')
        doc = bind_inputs(cfg)
        from kit.a2_replay import applicable
        if doc['arm'] in ('P', 'RP'):
            doc['P_applicable'] = applicable(doc['i0'])
        bind(out/'binding.json', doc)
        return dict(status='COMPLETE', artifacts=[str(out/'binding.json')])
    doc = binding(out)
    environment = 'inference' if phase in ('score', 'rewrite', 'confirmation-score') else 'trainer'
    require(t.runtime_versions() == doc['software'][environment], 'bound runtime environment changed')
    if doc.get('P_applicable') is False and doc['arm'] == 'RP' and row.get('model') == 'P-replay':
        atomic(attempt/'not-applicable.json', dict(comparisons=0, reason='all I0 maps zero'))
        return dict(status='COMPLETE', artifacts=[str(attempt/'not-applicable.json')])
    if phase == 'context':
        month = MONTHS[row['month']-1]
        archived = contexts(doc, month)
        values = []
        for item in batch(doc, month):
            saved = archived.get(item['id'])
            if not saved or any(not isinstance(saved.get(mode, {}).get('prompt'), str)
                                for mode in ('off', 'on')):
                raise NotReady('archived arrival memory-off/on context absent')
            if any(key not in saved for key in ('recurrence', 'stratum', 'recurring_issuer')):
                raise NotReady('archived arrival recurrence metadata absent')
            values.append(saved)
        root = out/'contexts'/month
        write_jsonl(root/'prompts.jsonl', values)
        freeze(root, ['prompts.jsonl'])
        paths = [root/'prompts.jsonl', root/'manifest.json']
    elif phase == 'score-prepare':
        paths = prepare_scores(cfg, doc, row, attempt)
    elif phase == 'score':
        paths = score(cfg, doc, row, attempt)
    elif phase == 'confirmation-bind':
        from kit.a2_confirmation import bind_confirmation
        bind_confirmation(cfg, doc, out)
        paths = [out/'confirmation-binding.json']
    elif phase == 'confirmation-prepare':
        from kit.a2_confirmation import prepare
        paths = prepare(cfg, doc, out, attempt)
    elif phase == 'confirmation-score':
        paths = score(cfg, doc, row, attempt)
    elif phase == 'confirmation-read':
        from kit.a2_jobs import completed
        from kit.a2_read import analyse
        from kit.a2_budget import reconcile
        root = completed(out, row['depends_on'][0]['id'])
        value = analyse(doc['arm'], load(root/'cells.json'), synthetic=doc['technical_synthetic'],
                        confirmation=True, receipt=load(out/'confirmation-binding.json'),
                        assignment=load(root/'receipt.json'), p_applicable=doc.get('P_applicable', True))
        value['accounting'] = reconcile(out, doc['technical_synthetic'])
        atomic(out/'confirmation-reading.json', value)
        paths = [out/'confirmation-reading.json']
    elif phase == 'confirmation-collect':
        from kit.a2_return import collect
        paths = collect(cfg, out, confirmation=True)
    elif phase == 'feedback':
        paths = feedback(cfg, doc, row, attempt)
    elif phase == 'prepare':
        paths = prepare_training(cfg, doc, row, attempt)
    elif phase == 'train':
        paths = train(cfg, doc, row, attempt)
    elif phase == 'export':
        paths = export(cfg, doc, row)
    elif phase == 'rewrite-prepare':
        paths = rewrite_prepare(cfg, doc, row, attempt)
    elif phase == 'rewrite':
        paths = rewrite_generate(cfg, doc, row, attempt)
    elif phase == 'freeze-june':
        value = dict(weights={m: {str(seed): model_identity(state(out, doc, m, seed, 6), doc)
                                  for seed in SEEDS} for m in arm_names(doc['arm'])},
                     memory_sha256=digest(ledger(doc, 6)), frozen_before_confirmation=True)
        bind(out/'frozen-june.json', value)
        paths = [out/'frozen-june.json']
    elif phase == 'read':
        from kit.a2_read import read
        from kit.a2_budget import reconcile
        value = read(out)
        value['accounting'] = reconcile(out, doc['technical_synthetic'])
        value['diagnostics'] = diagnostics(out, doc)
        atomic(out/'reading.json', value)
        paths = [out/'reading.json']
    elif phase == 'collect':
        from kit.a2_return import collect
        paths = collect(cfg, out)
    elif phase in ('month', 'rewrite-finish'):
        atomic(attempt/'complete.json', dict(complete=True))
        paths = [attempt/'complete.json']
    else:
        raise CorrectnessStop('unimplemented row: '+phase)
    return dict(status='COMPLETE', artifacts=[str(p) for p in paths])


def diagnostics(out, doc):
    result = {}
    stage_fallbacks = Counter()
    for month in MONTHS:
        from kit.a2_bind import receipt
        source = receipt(doc['inputs']['ledger'][month])
        traces = t.read_jsonl(source/'traces.jsonl')
        decisions = t.read_jsonl(source/'admission.jsonl')
        by_id = {i['id']: i for i in batch(doc, month)}
        result['raw-ledger/'+month] = dict(attempts=len(decisions), admitted=len(traces),
            rejection_reasons=dict(Counter(reason for d in decisions for reason in d.get('reasons', []))),
            families={f: dict(attempts=sum(by_id[d['id']]['sec']['template'] == f for d in decisions),
                admitted=sum(r['item']['sec']['template'] == f for r in traces)) for f in FAMILIES},
            response_characters=[len(r['response']) for r in traces],
            ledger_manifest_sha256=doc['inputs']['ledger'][month]['sha256'],
            frozen_collector=True, initial_collector_contrast=month in MONTHS[:2])
    for path in sorted((Path(out)/'runs').glob('*/summary.json')):
        summary = load(path)
        audit = summary['audit']
        counts = audit.get('repetitions', {})
        stage_fallbacks.update(audit.get('stage_length_fallbacks', []))
        # Per-ID schedules remain durable; report the full repetition distribution.
        result['stage/'+path.parent.name] = dict(skipped=summary['skipped'], updates=summary['updates'],
            scheduled_task=audit['scheduled_task'], scheduled_generic=audit['scheduled_generic'],
            unique_traces=audit['unique_traces'], repeated_exposures=sum(counts.values())-len(counts),
            repetition_histogram=dict(Counter(map(str, counts.values()))), tokens=summary['counts'],
            runtime=summary.get('runtime'), incoming=summary['incoming'],
            stage_length_fallbacks=dict(Counter(audit.get('stage_length_fallbacks', []))),
            rewrite_accepted_positions=audit.get('rewrite_accepted_positions'),
            rewrite_raw_fallback_positions=audit.get('rewrite_raw_fallback_positions'),
            schedule_sha256=s.file_hash(Path(out)/'data'/path.parent.name/'schedule.jsonl'))
    for path in sorted((Path(out)/'current-ledger').glob('*/*/admission.jsonl')):
        rows = t.read_jsonl(path)
        result['collector/'+path.parent.parent.name+'/'+path.parent.name] = dict(
            attempts=len(rows), admitted=sum(r['admitted'] for r in rows),
            rejected_reasons=dict(Counter(reason for r in rows for reason in r['reasons'])),
            families={family: dict(attempts=sum(r['family'] == family for r in rows),
                admitted=sum(r['family'] == family and r['admitted'] for r in rows)) for family in FAMILIES},
            initial_collector_contrast=path.parent.name in MONTHS[:2])
    rewrites = [load(p) for p in (Path(out)/'rewrites').glob('index-*.json')]
    if rewrites:
        source_families = {r['id']: r['item']['sec']['template'] for r in ledger(doc, 6)}
        report_family = lambda r: r.get('family') or source_families[r['id']]
        effective = lambda r: r['accepted'] and r['id'] not in stage_fallbacks
        failures = Counter(r['fallback_reason'] for r in rewrites if not r['accepted'])
        failures['stage_output_or_teacher_cap'] += len(stage_fallbacks)
        result['rewrite/totals'] = dict(attempts=len(rewrites), accepted=sum(effective(r) for r in rewrites),
            generation_accepted=sum(r['accepted'] for r in rewrites),
            stage_length_fallbacks=dict(stage_fallbacks),
            stage_length_fallback_positions=sum(stage_fallbacks.values()),
            accepted_basis='unique traces with no stage length fallback; partial use is reported per stage',
            pre_generation_failures=sum(not r.get('generation_attempted', False) and not r['accepted'] for r in rewrites),
            model_failures=sum(r.get('generation_attempted', False) and not r['accepted'] for r in rewrites),
            failures=dict(failures),
            by_month_family={m+'/'+f: dict(attempts=sum(r['month'] == m and report_family(r) == f for r in rewrites),
                accepted=sum(r['month'] == m and report_family(r) == f and effective(r) for r in rewrites),
                stage_length_fallbacks=sum(r['month'] == m and report_family(r) == f and
                                          r['id'] in stage_fallbacks for r in rewrites))
                for m in MONTHS for f in FAMILIES})
        raw = {r['id']: r['response'] for r in ledger(doc, 6)}
        for r in sorted(rewrites, key=lambda r: r['id'])[:6]:
            result['rewrite/example/'+r['id']] = dict(raw=raw[r['id']], rewritten=r['response'],
                                                     accepted=r['accepted'])
    result['scope'] = dict(confirmation='required regardless of exploratory verdict',
                           four_a_package_bars='context only; +5 improvement / 3-point retention unchanged',
                           historical_scoring='descriptive', utilisation='warn-only')
    return result
