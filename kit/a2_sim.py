"""Synthetic source-bound fixtures. No model, data download or private outcome."""
from copy import deepcopy
from kit.a2_common import *
from kit import a4_sim, a4_sec, a4_memory
from kit.a2_rewrite import FCF, CASH
from kit.beds import finqa


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=s.file_hash(path))


def financial(item):
    item = deepcopy(item)
    sc = item['sec']
    header = ['', '12 Months Ended Dec. 31, 2025', '12 Months Ended Dec. 31, 2024']
    cf = [header, ['Operating cash flow', '200', '100'], ['Capital expenditures', '(20)', '(10)']]
    inc = [header, ['Revenue', '1000', '900'], ['Operating income', '100', '100']]
    first = sc['excerpts'][0]
    first['table'] = cf
    second = dict(first, kind='income', r_page=first['r_page'].replace('R1', 'R2'), table=inc)
    sc['excerpts'] = [first, second]
    item['pre_text'] = [a4_sec.v4_excerpt_label(first)]
    item['table'] = cf
    item['post_text'] = [a4_sec.v4_excerpt_label(second), finqa.render_table(inc)]
    fcf = sc['template'] == 'fcf_margin_change'
    spec = [(first, 1, 1), (first, 2, 1), (second, 1, 1), (first, 1, 2), (first, 2, 2), (second, 1, 2)] if fcf else [(first, 1, 1), (second, 2, 1), (first, 1, 2), (second, 2, 2)]
    ops = []
    for excerpt, row, col in spec:
        table = excerpt['table']
        text = table[row][col]
        value = a4_sec.parse_value(text)
        ops.append(dict(label=table[row][0], shown_label=table[row][0], period=table[0][col],
                        cell=text, value=value, unit='US dollars', normalized_usd=str(int(value)),
                        r_page=excerpt['r_page'], table_row=row, table_column=col,
                        accession=sc['accession'], concept='fixture_concept', context_id='fixture_context',
                        context=dict(start='2025-01-01', end='2025-12-31')))
    sc['operands'] = ops
    formula = '100*((a+b)/c-(d+e)/f)' if fcf else '100*(a/b-c/d)'
    value, answer, program = a4_sec.v4_compute(formula, ops)
    sc.update(formula=formula, unrounded_gold=str(value))
    query = (FCF.replace('<CAPEX>', 'Capital expenditures').replace('<OCF>', 'Operating cash flow').replace('<REVENUE>', 'Revenue') if fcf else
             CASH.replace('<OCF>', 'Operating cash flow').replace('<OPERATING_INCOME>', 'Operating income'))
    item['qa'] = dict(question=query, exe_ans=float(value), answer=answer, program=program)
    return item


def fixture(root, arm):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sl, _, _ = a4_sim.fixture(root/'source', 'D')
    slice_refs, ledger_refs, context_refs = {}, {}, {}
    tok = __import__('kit.c_sim', fromlist=['Tokenizer']).Tokenizer()
    for month in MONTHS:
        items = [financial(i) for i in load(sl/f'batch-{month}.json')]
        atomic(sl/f'batch-{month}.json', items)
        slice_refs[month] = ref(sl/f'batch-{month}.json')
        traces = []
        for item in items:
            if item['sec']['issuer_holdout']:
                continue
            response = 'Verified arithmetic.\nAnswer: '+item['qa']['answer']
            traces.append(dict(id=item['id'], month=month, item=item, response=response,
                               source=item['filename'], date=item['sec']['filing_date'],
                               response_sha256=digest(response.encode()),
                               collector_checkpoint='synthetic-base', provenance=dict(unedited=True)))
        dest = root/'raw-ledger'/month
        write_jsonl(dest/'traces.jsonl', traces)
        write_jsonl(dest/'admission.jsonl', [dict(id=r['id'], admitted=True, reasons=[]) for r in traces])
        atomic(dest/'sealed.json', dict(whole_month=True, technical_synthetic=True))
        freeze(dest, ['traces.jsonl', 'admission.jsonl', 'sealed.json'])
        ledger_refs[month] = ref(dest/'manifest.json')
        dest = root/'archive-contexts'/month
        previous = [r for m in MONTHS[:MONTHS.index(month)]
                    for r in t.read_jsonl(checked(ledger_refs[m]).parent/'traces.jsonl')]
        archived = [dict(id=i['id'], month=month, off=a4_memory.build(i, [], tok),
                         on=a4_memory.build(i, previous, tok),
                         **a4_memory.recurrence(i, previous), recurring_issuer=False) for i in items]
        if arm in ('R', 'RP') and month == MONTHS[0]:
            # An ordinary registered parsing failure, through the real wrapper.
            # Inference still uses the exact archived text, without repair.
            archived[0]['off']['prompt'] += ' Unsupported fixture suffix.'
        write_jsonl(dest/'prompts.jsonl', archived)
        freeze(dest, ['prompts.jsonl'])
        context_refs[month] = ref(dest/'manifest.json')
    general = root/'general.jsonl'
    write_jsonl(general, [dict(id=f'{p}-{i}', prompt=f'CPU panel {p} row {i}', panel=p) for p in GENERAL for i in range(10)])
    audit = root/'audit.json'
    atomic(audit, dict(audited=True, months=list(MONTHS), families=list(FAMILIES),
                       slice_hashes={m: r['sha256'] for m, r in slice_refs.items()}))
    inputs = root/'inputs.json'
    atomic(inputs, dict(schema='a2-node-input.v1', technical_synthetic=True, slice=slice_refs,
                        ledger=ledger_refs, contexts=context_refs, general=ref(general), audit=ref(audit), base='synthetic-base'))
    cfg = dict(arm=arm, out=str(root/'out'), inputs=ref(inputs), technical_synthetic=True,
               disposition='active', archive=str(root/f'{arm}-return.tar.gz'), text_out=str(root/'text'),
               trainer_python=__import__('sys').executable, inference_python=__import__('sys').executable,
               job_python=__import__('sys').executable)
    cfg['budget'] = dict(approved_gpu_hours=601,
                         approval_reference='TECHNICAL-ONLY owner stand-in')
    placeholder = root/'receipt.json'
    atomic(placeholder, dict(technical_synthetic=True))
    exclusions = root/'exclusions.json'
    atomic(exclusions, dict(jan_jun_overlap=[], oct_nov_overlap=[],
                            sealed_confirmation_manifest_sha256='0'*64, jul_sep_exposure=[]))
    lineages = {}
    for seed in SEEDS:
        final = 'synthetic-connected-'+str(seed)
        stage = dict(parent='synthetic-base', final=final, seed=seed,
                     **{key: ref(placeholder) for key in ('acquisition', 'effective_config', 'data_manifest', 'schedule_manifest', 'export')})
        lineages[str(seed)] = dict(base='synthetic-base', stages=[stage], terminal=final, complete_registered_sequence=True)
    connected = root/'connected.json'
    atomic(connected, dict(status='VERIFIED', seeds=list(SEEDS), continuous=True,
                           recipe_dispositions=ref(placeholder), exclusions=ref(exclusions), lineages=lineages))
    cfg['connected'] = ref(connected)
    pool = root/'generic.jsonl'
    generic = [dict(id=f'g-{i}', prompt=f'Describe botanical taxonomy genus {i}.', response='A stored unrelated response.') for i in range(12)]
    write_jsonl(pool, generic)
    targets = root/'targets.json'
    atomic(targets, {r['id']: r['response'] for r in generic})
    overlap = root/'overlap.json'
    atomic(overlap, dict(excluded_ids=[]))
    maps = {}
    for seed in SEEDS:
        path = root/f'map-{seed}.json'
        order = cycle(generic, seed, 160)
        atomic(path, dict(seed=seed, slots=[[8, 16, 24, 32] for _ in range(40)],
                          accepted_order=[r['id'] for r in order], skipped_draws=[]))
        maps[str(seed)] = ref(path)
    replay = root/'i0.json'
    atomic(replay, dict(verified=True, source='I0', stage='terminal-contractnli', seeds=list(SEEDS),
                        pool=ref(pool), stored_targets=ref(targets), draw_rules=ref(placeholder),
                        overlap_receipt=ref(overlap), maps=maps))
    cfg['i0'] = ref(replay)
    atomic(root/'config.json', cfg)
    return cfg


def confirmation_fixture(root, cfg):
    """Fake future inputs: never opens any actual October-November filing."""
    root = Path(root)
    refs = {}
    original = load(checked(load(checked(cfg['inputs']))['slice']['2026-06']))
    for month in ('2026-10', '2026-11'):
        rows = []
        for index in range(40):
            item = deepcopy(original[index % len(original)])
            item['id'] = f'confirmation-{month}-{index}'
            item['sec']['accepted'] = month+'-15 00:00:00'
            item['sec']['issuer_holdout'] = index < 12
            item['sec']['issuer_group'] = f'confirmation-component-{index}'
            rows.append(item)
        path = root/f'fake-{month}.json'
        atomic(path, rows)
        refs[month] = ref(path)
    audit = root/'fake-confirmation-audit.json'
    atomic(audit, dict(source_disjoint=True, fixed_exclusions=True, audit_verified=True, release_controls=True))
    path = root/'fake-confirmation.json'
    atomic(path, dict(schema='a2-confirmation.v1', cutoff='2026-12-14', build_date='2026-12-15',
                      frozen_june_sha256=s.file_hash(Path(cfg['out'])/'frozen-june.json'), audit=ref(audit), slice=refs))
    return dict(cfg, allow_confirmation=True, confirmation=ref(path))
