"""Offline node-input audits and outcome-independent subset binding."""
from collections import Counter
from kit.trace_common import *
from kit.trace_overlap import audit_rows


def cohort(rows, lengths):
    need(len({r['id'] for r in rows}) == len(rows), 'duplicate candidate ID')
    selected, excluded = [], []
    for row in rows:
        n = lengths[row['id']]
        eligible = (n['student'] <= 2048 and n['gold_teacher'] <= 6144
                    and n['gold_response'] <= 2048)
        if eligible:
            selected.append(row)
        else:
            excluded.append(dict(id=row['id'], reason='registered_length', lengths=n))
    ranked = sorted(selected, key=lambda r: (digest_string(
        f'trace-subset-20261111|{r["task"]}|{r["id"]}'), r['id']))
    need(len(ranked) >= 500, 'fewer than 500 eligible training rows')
    from kit.p3b_replay_data import normal
    distinct, seen = [], set()
    for row in ranked:
        content = (normal(row['prompt']), normal(row['answer']))
        if content in seen:
            excluded.append(dict(id=row['id'], reason='duplicate_content'))
        else:
            distinct.append(row)
            seen.add(content)
    need(len(distinct) >= 500, 'fewer than 500 distinct eligible training contents')
    return distinct[:500], excluded


def census(rows, tokenizer):
    """No model or data acquisition. Counts use the staged tokenizer."""
    from kit.trace_prompt import install
    install()
    from kit.sdft_objective import teacher_messages
    def encoded(messages):
        return tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
    result = {}
    for row in rows:
        messages = [dict(role='user', content=row['prompt'])]
        result[row['id']] = dict(student=len(encoded(messages)),
            gold_teacher=len(encoded(teacher_messages(messages, row['answer']))),
            gold_response=len(tokenizer.encode(row['answer']+tokenizer.eos_token,
                                               add_special_tokens=False)))
    return result


def licence_blocks(rights, studies):
    need(studies and set(studies) <= {'s8', *SOURCES}, 'unknown/empty study scope')
    common = {'original_tokenizer', 'base', 'judge', 'panels', 'calibration'}
    result = {}
    for study in studies:
        required = common | set(TASKS if study == 's8' else SUBSET)
        if study in SOURCES:
            required.add('teacher')
        missing = sorted(k for k in required if not (
            rights.get(k, {}).get('permitted') is True and
            rights.get(k, {}).get('evidence_sha256')))
        if missing:
            result[study] = missing
    return result


def prepare(cfg):
    offline()
    out = Path(cfg['out'])
    need(cfg['base']['revision'] == BASE and cfg['judge']['revision'] == JUDGE['revision'],
         'registered model revision differs')
    studies = cfg.get('studies', ['s8', *SOURCES])
    # Shipped operator plans validate available CPU authorities before targets.
    # The generated-demo fidelity audit is completed after targets, at bind.
    if cfg.get('admission'):
        admission_plan = bound_json(cfg['admission'])
        if admission_plan.get('schema') == 'trace-admission-plan.v1':
            from kit.trace_operator import admission
            admission(admission_plan['evidence'], studies, defer_27b=True, cfg=cfg)
    roles = ['base', 'judge', 'original_tokenizer']
    if set(studies) & set(SOURCES):
        roles.append('teacher')
    for role in roles:
        model_record(cfg[role])
    if 'teacher' in roles:
        need(cfg['teacher']['revision'] == '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9',
             '27B teacher revision differs')
    rights = bound_json(cfg['licences'])
    blocked = licence_blocks(rights, studies)
    licence_pending = set(blocked)
    active = [study for study in studies if study not in blocked]
    required_tasks = set(TASKS if 's8' in active else SUBSET if active else ())
    exclusion = bound_json(cfg['sec_clearance'])
    need(exclusion['reserved_outcomes_opened'] is False and exclusion['intersections'] == 0
         and exclusion['near_match_clearance'] is True, 'SEC hash clearance incomplete')
    need(required_tasks <= set(cfg['datasets']) <= set(TASKS), 'TRACE task inventory differs')
    datasets = {task: {split: read_rows(ref) for split, ref in splits.items()}
                for task, splits in cfg['datasets'].items() if task in required_tasks}
    need(all(set(s) == {'train', 'eval', 'test'} for s in datasets.values()),
         'released splits incomplete')
    reviews = bound_json(cfg['audit_review']) if cfg.get('audit_review') else []
    audit = audit_rows(datasets, read_rows(cfg['exclusions']), reviews)
    seal(out/'audit'/'manual-review.json', dict(pairs=audit['flagged']))
    if audit['unresolved'] or not active:
        return dict(status='PENDING', audit=audit, blocked_studies=blocked,
                    manual_review='audit/manual-review.json')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['base']['model'], local_files_only=True)
    original_tok = (AutoTokenizer.from_pretrained(cfg['original_tokenizer']['model'],
                    local_files_only=True) if 's8' in active else None)
    warnings = []
    selected, tests, counts, eos_census, sft_rows = {}, {}, {}, {}, {}
    for task, splits in datasets.items():
        if 's8' in active:
            need(len(splits['train']) == 5000, 'released full-dose training count differs')
        counts[task] = {split: len(rows) for split, rows in splits.items()}
        if 's8' in active:
            from kit.trace_sft import sft_census, deepspeed_config
            need(deepspeed_config()['gradient_clipping'] == GRADIENT_CLIPPING == 1.,
                 'upstream gradient clipping configuration differs')
            eos_census[task] = {}
            for split, rows in splits.items():
                encoded, summary = sft_census(rows, tok, original_tok)
                eos_census[task][split] = summary
                warnings.extend(f'{task}/{split}: {message}' for message in summary['warnings'])
                if split == 'train' and summary['valid']:
                    path = out/'sft'/f'{task}.json'
                    seal(path, encoded)
                    sft_rows[task] = dict(path=str(path), sha256=sha(path))
        if task in SUBSET and set(active) & set(SOURCES):
            lengths = census(splits['train'], tok)
            chosen, excluded = cohort(splits['train'], lengths)
            selected[task] = chosen
            test_lengths = census(splits['test'], tok)
            tests[task] = [r for r in splits['test'] if test_lengths[r['id']]['student'] <= 2048]
            need(tests[task], 'empty length-eligible test split')
            seal(out/'census'/f'{task}.json', dict(train=lengths, test=test_lengths,
                 exclusions=excluded, selected_ids=[r['id'] for r in chosen],
                 excluded_tests=[r['id'] for r in splits['test'] if r not in tests[task]]))
    panels = bound_json(cfg['panels'])
    need(Counter({'ifeval':'instruction','math':'maths'}.get(r['panel'],r['panel']) for r in panels['general-v1']) ==
         Counter(instruction=100, knowledge=100, maths=100), 'general-v1 count differs')
    from kit.score_forgetting import load_panel
    general, _ = load_panel()
    need(panels['general-v1'] == general, 'frozen general-v1 membership changed')
    if 's8' in active:
        need(set(panels['trace_general']) == {'MMLU', 'BBH', 'GSM8K', 'TyDiQA', 'PIQA', 'BoolQ'},
             'TRACE general panel inventory differs')
    need(Counter(r['panel'] for r in panels['open']) ==
         Counter({'Self-Instruct':175, 'LIMA':300, 'CoNa':178}), 'open panel IDs/count differ')
    if isinstance(cfg['calibration'],dict) and cfg['calibration'].get('schema') == 'trace-calibration-assembly.v1':
        from kit.trace_calibration_build import node_assemble
        calibration = node_assemble(cfg['calibration'], panels['open'],
                                    out/'sealed-calibration'/'calibration.jsonl')
    else:
        calibration = read_rows(cfg['calibration'])
    need(Counter(r['panel'] for r in calibration) ==
         Counter({'Self-Instruct':30, 'LIMA':30, 'CoNa':30}), 'calibration count differs')
    need(all(r.get('checkpoint_independent') is True for r in calibration),
         'calibration depends on checkpoints')
    s8_ready = ('s8' in active and all(
        splits['train']['valid'] for splits in eos_census.values()))
    if 's8' in active and not s8_ready:
        blocked['s8'] = ['L-EOS incorrect supervised suffix or EOS-position difference']
    protocol = dict(s8_ready=s8_ready, sft_rows=sft_rows, l_eos=eos_census,
                    blocked_studies=blocked, active_studies=[s for s in active if s not in blocked],
                    warnings=warnings,
                    licences={s:('PENDING' if s in licence_pending else 'CLEARED') for s in studies},
                    gradient_clipping=GRADIENT_CLIPPING, config_sha256=digest(cfg), registration_sha256=sha(REG),
                    datasets=datasets, selected=selected, tests=tests, panels=panels,
                    counts=counts, audit=audit, calibration=calibration,
                    source_sha256={p.name:sha(p) for p in KIT.glob('trace_*.py')})
    return seal(out/'prepared.json', protocol)


def bind(cfg):
    prepared = load(Path(cfg['out'])/'prepared.json')
    need(prepared['config_sha256'] == digest(cfg), 'prepared config differs')
    need(prepared['audit']['unresolved'] == 0, 'overlap review PENDING')
    need(prepared['registration_sha256'] == sha(REG), 'TRACE registration changed')
    demos = load(Path(cfg['out'])/'targets.json') if prepared['selected'] else {}
    need(set(demos) == {r['id'] for rows in prepared['selected'].values() for r in rows},
         '27B fixed cohort incomplete')
    from transformers import AutoTokenizer
    from kit.trace_prompt import install
    install()
    from kit.sdft_objective import teacher_messages
    tok = AutoTokenizer.from_pretrained(cfg['base']['model'], local_files_only=True)
    for task, rows in prepared['selected'].items():
        for row in rows:
            target = demos[row['id']]['text']
            need(demos[row['id']]['verified'] is True, '27B demo verification incomplete')
            teacher = teacher_messages([dict(role='user', content=row['prompt'])], target)
            need(len(tok.apply_chat_template(teacher, tokenize=True, add_generation_prompt=True,
                       enable_thinking=False)) <= 6144, '27B teacher prompt overlength')
            need(len(tok.encode(target+tok.eos_token, add_special_tokens=False)) <= 2048,
                 '27B demonstration response overlength')
        for seed in SEEDS:
            for source in SOURCES:
                scheduled = [dict(r, response=r['answer'] if source == 'gold' else demos[r['id']]['text'],
                                  gold=r['answer']) for r in schedule(rows, seed, task)]
                seal(Path(cfg['out'])/'schedules'/f'{source}-{seed}-{task}.json', scheduled)
    probes = bound_json(cfg['admission'])
    if probes.get('schema') == 'trace-admission-plan.v1':
        from kit.trace_operator import admission
        probes = admission(probes['evidence'], cfg.get('studies', ['s8', *SOURCES]), cfg=cfg)
        seal(Path(cfg['out'])/'admission-receipt.json', probes)
    required = {'original_tokenizer_census', 'supervised_masks', 'prompt_fixtures',
                'scorer_fixtures'}
    if prepared['selected']:
        required.update(('demo_fidelity_gold','demo_fidelity_27b'))
    need(set(probes) >= required, 'admission probes incomplete')
    need(all(probes[k]['valid'] is True and probes[k]['evidence_sha256'] for k in required),
         'admission identity/scorer/leakage probe unresolved')
    return seal(Path(cfg['out'])/'binding.json', dict(prepared_sha256=sha(Path(cfg['out'])/'prepared.json'),
                targets_sha256=sha(Path(cfg['out'])/'targets.json') if prepared['selected'] else None, probes=probes,
                config_sha256=digest(cfg), registration_sha256=sha(REG)))
