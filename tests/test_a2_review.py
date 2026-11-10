"""Round-2 review regressions: operational bindings, frozen contexts and fallbacks."""
import ast
import copy
from pathlib import Path

import pytest
from kit.a2_common import *
from kit import a2_bind, a2_jobs, a2_pipeline as pipeline, a2_read, a2_replay
from kit.a2_sim import fixture, ref


def test_every_binder_key_and_config_read_is_documented():
    readme = (KIT/'README-a2.md').read_text()
    keys = set()
    for path in (KIT/'a2_bind.py', *KIT.glob('a2_*.py')):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # All binder records, plus every owner configuration read elsewhere.
            binder = path.name == 'a2_bind.py'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
                if binder or isinstance(node.func.value, ast.Name) and node.func.value.id in ('cfg', 'budget'):
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        keys.add(node.args[0].value)
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                if binder or isinstance(node.value, ast.Name) and node.value.id in ('cfg', 'budget'):
                    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                        keys.add(node.slice.value)
            if binder and isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'key':
                keys.update(x.value for x in node.iter.elts if isinstance(x, ast.Constant))
    assert not sorted(k for k in keys if '`'+k+'`' not in readme)


@pytest.mark.parametrize('arm', ['J', 'P', 'R', 'RP'])
def test_copy_4a_contexts_exactly_and_never_rebuild(arm, tmp_path, monkeypatch):
    cfg = fixture(tmp_path/arm, arm)
    data = load(checked(cfg['inputs']))
    path = checked(data['contexts'][MONTHS[0]])
    saved = t.read_jsonl(path.parent/'prompts.jsonl')
    saved[0].update(recurrence=3, stratum='high', recurring_issuer=True)
    saved[0]['on']['prompt'] += ' Archived bytes to preserve.'
    write_jsonl(path.parent/'prompts.jsonl', saved)
    freeze(path.parent, ['prompts.jsonl'])
    data['contexts'][MONTHS[0]] = ref(path)
    atomic(checked(cfg['inputs']), data)
    cfg['inputs'] = ref(Path(cfg['inputs']['path']))
    doc = a2_bind.bind_inputs(cfg)
    atomic(Path(cfg['out'])/'binding.json', doc)
    monkeypatch.setattr(pipeline.a4_memory, 'build', lambda *a: pytest.fail('must use archived 4A contexts'))
    pipeline.operation(cfg, dict(phase='context', month=1), tmp_path/'attempt')
    assert (Path(cfg['out'])/'contexts'/MONTHS[0]/'prompts.jsonl').read_bytes() == (path.parent/'prompts.jsonl').read_bytes()
    # A missing non-holdout off context has the same disposition in every arm.
    saved[0].pop('off')
    write_jsonl(path.parent/'prompts.jsonl', saved)
    freeze(path.parent, ['prompts.jsonl'])
    doc['inputs']['contexts'][MONTHS[0]] = ref(path)
    atomic(Path(cfg['out'])/'binding.json', doc)
    with pytest.raises(NotReady, match='archived arrival'):
        pipeline.operation(cfg, dict(phase='context', month=1), tmp_path/'attempt')


def test_rewrite_inventory_counts_bound_monthly_traces_and_splits_batches(tmp_path):
    cfg = fixture(tmp_path/'R', 'R')
    cfg['score_max_seconds'] = 7200
    inv = a2_jobs.configured_inventory(cfg)
    rewrites = [r for r in inv['rows'] if r['phase'] == 'rewrite']
    assert all(r['time_limit_seconds'] <= 86400 for r in rewrites)
    for n in range(1, 7):
        chunks = [r for r in rewrites if r['month'] == n]
        assert sum(r['commands'] for r in chunks) == 12
        assert [p for r in chunks for p in range(r['trace_start'], r['trace_stop'])] == list(range(12))
        assert all(r['count_basis'] == 'bound admitted ledger' for r in chunks)
    for arm in ('J', 'P', 'R', 'RP'):
        inv = a2_jobs.inventory(arm)
        assert 'A2-'+arm+'-comparison' in inv['long_lanes']
        assert inv['scheduling_conflicts'] == []  # all rows fit the captured 7-day MaxTime
        assert all(r['time_limit_seconds'] <= a2_jobs.SITE_MAX_SECONDS for r in inv['rows'])
        if arm == 'RP':
            assert 'A2-RP-confirmation-score' in inv['long_lanes']


def test_P_overlap_passes_only_excluded_draws_and_checks_receipt(tmp_path):
    cfg = fixture(tmp_path/'P', 'P')
    doc = a2_bind.bind_inputs(cfg)
    source = copy.deepcopy(doc['i0'])
    pool = t.read_jsonl(checked(source['pool']))
    for i, entry in enumerate(pool):
        entry['prompt'] = chr(65+i)*80
    pool_path = tmp_path/'distinct-pool.jsonl'; write_jsonl(pool_path, pool)
    source['pool'] = ref(pool_path)
    evaluation = [pool[0]]
    spec = load(checked(source['maps']['101']))
    spec['accepted_order'] = [pool[0]['id']]+[r['id'] for r in pool[1:]]*20
    path = tmp_path/'spare-order.json'; atomic(path, spec)
    source['maps']['101'] = ref(path)
    overlap = tmp_path/'overlap.json'; atomic(overlap, dict(excluded_ids=[pool[0]['id']]))
    source['overlap_receipt'] = ref(overlap)
    task = [dict(id='sec-task', prompt='SEC question', response='Answer: 1')]*1280
    rows, audit = a2_replay.transfer(task, source, 101, evaluation)
    assert audit['passed_over'] == [dict(id=pool[0]['id'], source_index=0)]
    assert audit['generic_positions'] == 160 and audit['task_positions'] == 1120
    assert [r['id'] for r in rows if r.get('p3b_replay')] == spec['accepted_order'][1:161]
    assert all(a == b for a, b in zip(task, rows) if not b.get('p3b_replay'))
    atomic(overlap, dict(excluded_ids=[])); source['overlap_receipt'] = ref(overlap)
    with pytest.raises(NotReady, match='overlap exclusions'):
        a2_replay.transfer(task, source, 101, evaluation)


def test_unfinished_comparison_is_incomplete(tmp_path):
    atomic(tmp_path/'config.json', dict(arm='R', disposition='active'))
    assert a2_read.read(tmp_path)['status'] == 'INCOMPLETE'


def test_missing_cache_stops_and_stage_length_failures_reach_totals(tmp_path, monkeypatch):
    cfg = fixture(tmp_path/'R', 'R'); doc = a2_bind.bind_inputs(cfg)
    row = dict(model='R-rewrite', seed=101, month=2, id='test-stage')
    with pytest.raises(NotReady, match='rewrite cache entry absent'):
        pipeline.prepare_training(cfg, doc, row, tmp_path/'attempt')
    raw = a2_bind.ledger(doc, 2)
    for trace in raw:
        atomic(Path(cfg['out'])/'rewrites'/('index-'+digest(trace['id'])+'.json'),
               dict(id=trace['id'], month=trace['month'], family=trace['item']['sec']['template'],
                    accepted=True, response='Procedure: forced stage overflow', generation_attempted=True))
    original = pipeline.a4_train.training_rows
    def stage_overflow(rows, tok, recipe):
        if rows[0]['response'].startswith('Procedure:'):
            raise CorrectnessStop('stage teaching context cap')
        return original(rows, tok, recipe)
    monkeypatch.setattr(pipeline.a4_train, 'training_rows', stage_overflow)
    pipeline.prepare_training(cfg, doc, row, tmp_path/'attempt')
    out = Path(cfg['out']); run = pipeline.stage(out, row)
    prepared = load(out/'data'/run.name/'prepared.json')
    assert len(prepared['audit']['stage_length_fallbacks']) == 1280
    assert prepared['audit']['rewrite_accepted_positions'] == 0
    assert prepared['audit']['rewrite_raw_fallback_positions'] == 1280
    pipeline.train(cfg, doc, row, tmp_path/'attempt'); pipeline.export(cfg, doc, row)
    report = pipeline.diagnostics(out, doc)
    assert report['rewrite/totals']['accepted'] == 0
    assert report['rewrite/totals']['generation_accepted'] == 24
    assert report['rewrite/totals']['stage_length_fallback_positions'] == 1280
    assert report['rewrite/totals']['failures']['stage_output_or_teacher_cap'] == 24
    teaching = t.read_jsonl(out/'data'/run.name/'teaching-contexts.jsonl')
    assert all(not r['demonstration'].startswith('Procedure:') for r in teaching)


def test_engine_initialisation_failure_can_resume_but_generation_cannot_retry(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from kit.c_sim import Tokenizer
    from kit import inference_lifecycle
    cfg = fixture(tmp_path/'R', 'R'); doc = a2_bind.bind_inputs(cfg)
    attempt = tmp_path/'prepare'; attempt.mkdir()
    row = dict(month=2, depends_on=[dict(id='prepared')])
    pipeline.rewrite_prepare(cfg, doc, row, attempt)
    # A single valid trace is enough to exercise the actual native generation branch.
    prepared = load(attempt/'rewrite-inputs.json')[:1]
    atomic(attempt/'rewrite-inputs.json', prepared)
    monkeypatch.setattr(a2_jobs, 'completed', lambda *a: attempt)
    monkeypatch.setattr(pipeline, 'tok_for', lambda *a: Tokenizer())
    monkeypatch.setattr(inference_lifecycle, 'shutdown_since', lambda: None)
    doc['technical_synthetic'] = False
    def unavailable(**kw):
        raise RuntimeError('engine initialisation failed')
    monkeypatch.setattr(inference_lifecycle, 'create_llm', unavailable)
    with pytest.raises(RuntimeError, match='initialisation'):
        pipeline.rewrite_generate(cfg, doc, row, tmp_path/'initialisation')
    assert not list((Path(cfg['out'])/'rewrites').glob('*.started.json'))
    calls = []
    def interrupted(*args):
        calls.append(args)
        assert len(list((Path(cfg['out'])/'rewrites').glob('*.started.json'))) == 1
        raise RuntimeError('generation interrupted')
    monkeypatch.setattr(inference_lifecycle, 'create_llm', lambda **kw: SimpleNamespace(generate=interrupted))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: kw))
    with pytest.raises(RuntimeError, match='generation interrupted'):
        pipeline.rewrite_generate(cfg, doc, row, tmp_path/'interrupted')
    pipeline.rewrite_generate(cfg, doc, row, tmp_path/'resumed')
    assert len(calls) == 1
    cached = pipeline.rewrite_cache(cfg['out'], doc, prepared[0]['id'])
    assert cached['fallback_reason'] == 'generation_interrupted_no_retry'
    assert cached['generation_attempted'] and cached['response'] == prepared[0]['response']


def test_rewrite_prompt_check_uses_independent_manifest_binding(tmp_path, monkeypatch):
    cfg = fixture(tmp_path/'R', 'R'); doc = a2_bind.bind_inputs(cfg)
    original = copy.deepcopy(doc['arrival_context_bindings'][MONTHS[1]])
    path = checked(doc['inputs']['contexts'][MONTHS[1]])
    saved = t.read_jsonl(path.parent/'prompts.jsonl')
    saved[0]['off']['prompt'] += ' Changed after binding.'
    write_jsonl(path.parent/'prompts.jsonl', saved)
    with pytest.raises(CorrectnessStop, match='input hash differs'):
        a2_bind.contexts(doc, MONTHS[1])
    # Even if an adapter supplies a newly manifested archive, the independent
    # original per-prompt binding still makes the rewrite a counted raw fallback.
    freeze(path.parent, ['prompts.jsonl']); doc['inputs']['contexts'][MONTHS[1]] = ref(path)
    attempt = tmp_path/'prepared'; attempt.mkdir()
    row = dict(month=2, depends_on=[dict(id='prepared')])
    pipeline.rewrite_prepare(cfg, doc, row, attempt)
    prepared = load(attempt/'rewrite-inputs.json')[:1]
    assert prepared[0]['prompt_sha256'] == original['off_prompt_sha256'][saved[0]['id']]
    assert prepared[0]['prompt_sha256'] != digest(prepared[0]['prompt'].encode())
    atomic(attempt/'rewrite-inputs.json', prepared)
    monkeypatch.setattr(a2_jobs, 'completed', lambda *a: attempt)
    pipeline.rewrite_generate(cfg, doc, row, tmp_path/'rewritten')
    cached = pipeline.rewrite_cache(cfg['out'], doc, prepared[0]['id'])
    assert not cached['accepted'] and cached['fallback_reason'] == 'prompt_missing_or_mismatch'
    assert cached['response'] == prepared[0]['response']
    assert cached['context_binding'] == original['manifest']
    assert not list((Path(cfg['out'])/'rewrites').glob('*.started.json'))


def test_rewrite_batches_execute_exact_disjoint_monthly_ranges(tmp_path, monkeypatch):
    cfg = fixture(tmp_path/'R', 'R'); cfg['score_max_seconds'] = 7200
    doc = a2_bind.bind_inputs(cfg)
    prepared = tmp_path/'prepared'; prepared.mkdir()
    pipeline.rewrite_prepare(cfg, doc, dict(month=2), prepared)
    monkeypatch.setattr(a2_jobs, 'completed', lambda *a: prepared)
    chunks = [r for r in a2_jobs.configured_inventory(cfg)['rows'] if r['phase'] == 'rewrite' and r['month'] == 2]
    attempts = []
    for i, row in enumerate(chunks):
        attempt = tmp_path/str(i)
        pipeline.rewrite_generate(cfg, doc, row, attempt)
        attempts.append(load(attempt/'rewrite-report.json')['attempts'])
    assert attempts == [10, 2]
    entries = [load(p) for p in (Path(cfg['out'])/'rewrites').glob('index-*.json')]
    assert len(entries) == 12 and all(r['accepted'] for r in entries)
    assert {r['id'] for r in entries} == {r['id'] for r in a2_bind.ledger(doc, 2) if r['month'] == MONTHS[1]}
