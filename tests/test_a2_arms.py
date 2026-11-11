"""A2 standalone execution, paired bars, frozen schedules, returns and failures."""
import copy
import json
from pathlib import Path

import pytest
from kit.a2_common import *
from kit import a2_jobs as jobs, a2_replay, a2_read, a2_return, a2_dryrun
from kit.a2_bind import bind_inputs, ledger
from kit.a2_sim import fixture, ref
from kit.a2_dryrun import rehearse


# Deliberate stand-in, never a source-history identity or a release receipt.
SYNTHETIC_KIT_COMMIT = 'deadbeef' * 5


@pytest.fixture
def synthetic_rehearsal_identity(monkeypatch):
    original = a2_dryrun.fixture

    def configured_fixture(root, arm):
        cfg = original(root, arm)
        cfg['kit_commit'] = SYNTHETIC_KIT_COMMIT
        atomic(Path(root)/'config.json', cfg)
        return cfg

    # rehearse writes this config before launching any child; its archived-arm
    # config and confirmation inputs inherit the same explicit synthetic ID.
    monkeypatch.setattr(a2_dryrun, 'fixture', configured_fixture)


def test_frozen_branch_map_and_bars():
    from kit.a2_common import route
    assert route('established', 'met') == 'J'
    assert route('established', 'failed') == 'P'
    assert route('established', 'not established') == 'P'
    assert route('not established', 'failed') == 'RP'
    assert route('not established', 'met') == 'R'
    assert route('unresolved', 'met') is None
    assert route('established', 'unresolved') is None
    assert benefit([2, 2, 2])['pass_bar']
    assert not benefit([1.99]*3)['pass_bar']
    assert not benefit([20, 20, 0])['pass_bar']
    assert guard([-1]*3)['pass_bar']
    assert not guard([-1.001]*3)['pass_bar']
    assert not guard([-1, 0, 1])['pass_bar']
    with pytest.raises(CorrectnessStop): benefit([3, 3])


@pytest.mark.parametrize('arm', ['J', 'P', 'R', 'RP'])
def test_inventory_is_explicit_independent_and_shared_control_fixed(arm):
    inv = jobs.inventory(arm)
    assert inv == load(RES/(arm+'-rows.json'))
    seen = set()
    for row in inv['rows']:
        assert all(d['id'] in seen for d in row['depends_on'])
        assert row['nodes'] == 1 and row['cpus'] in (8, 32)
        assert row['gpus'] == (8 if row['phase'] == 'train' else
                              1 if row['phase'] in ('score', 'rewrite', 'confirmation-score') else 0)
        assert row['command'][-1] == row['id']
        assert row['dependency_options'] == ['--kill-on-invalid-dep=yes']
        assert row['time_limit_seconds'] <= 86400 or 'long_lane_warning' in row
        assert row['time_limit_seconds'] <= jobs.SITE_MAX_SECONDS or 'scheduling_conflict' in row
        seen.add(row['id'])
    trains = [r for r in inv['rows'] if r['phase'] == 'train']
    assert len(trains) == (27 if arm == 'RP' else 18)
    if arm == 'RP':
        assert {r['model'] for r in trains} == {'raw-plain', 'P-replay', 'R-rewrite'}


def test_I0_positions_displace_only_and_repeat_verbatim(tmp_path):
    cfg = fixture(tmp_path/'P', 'P')
    doc = bind_inputs(cfg)
    task = cycle([dict(id='task-'+str(i), prompt='task', response='Answer: 1') for i in range(5)], 101, 1280)
    rows, audit = a2_replay.transfer(task, doc['i0'], 101, [])
    assert audit['generic_positions'] == 160 and audit['task_positions'] == 1120
    assert [i+1 for i, r in enumerate(rows[:32]) if r.get('p3b_replay')] == [8, 16, 24, 32]
    assert all(a == b for a, b in zip(task, rows) if not b.get('p3b_replay'))
    assert a2_replay.transfer(task, doc['i0'], 101, [])[0] == rows
    spec = load(checked(doc['i0']['maps']['101']))
    spec['slots'][0] = []
    spec['slots'][1] = [1]
    path = tmp_path/'changed.json'; atomic(path, spec)
    source = copy.deepcopy(doc['i0']); source['maps']['101'] = ref(path)
    rows, audit = a2_replay.transfer(task, source, 101, [])
    assert not any(r.get('p3b_replay') for r in rows[:32])
    assert rows[32].get('p3b_replay')
    assert audit['generic_positions'] == 153


def test_missing_inputs_pending_and_invalid_ancestry_unverified(tmp_path):
    cfg = fixture(tmp_path/'J', 'J')
    cfg['connected'] = None
    with pytest.raises(NotReady): bind_inputs(cfg)
    cfg = load(tmp_path/'J/config.json')
    conn = load(checked(cfg['connected']))
    conn['lineages']['101']['stages'][0]['reset'] = True
    path = tmp_path/'invalid-connected.json'; atomic(path, conn)
    cfg['connected'] = ref(path)
    with pytest.raises(CorrectnessStop, match='stitched'): bind_inputs(cfg)


@pytest.mark.parametrize('arm', ['J', 'P', 'R', 'RP'])
def test_all_rows_cpu_rehearsal(arm, tmp_path_factory, synthetic_rehearsal_identity):
    root = tmp_path_factory.mktemp('a2-'+arm)/'run'
    result = rehearse(root, arm)
    assert result['status'] == 'TECHNICAL-ONLY'
    assert result['training_stages'] == (27 if arm == 'RP' else 18) and result['all_rows_through_real_wrapper']
    assert result['archived_arm'] == 'ARCHIVED UNREAD' and result['missing_input'] == 'PENDING'
    assert result['archive_readback_equal'] and result['text_readback_equal']
    cfg = load(root/'config.json'); out = Path(cfg['out'])
    from kit.a2_budget import gate
    price = gate(cfg, out)
    assert price['decision'] == 'PROCEED' and len(price['timing_refs']) == len(arm_names(arm))
    assert 'D_timing' not in cfg['budget'] and 'challenger_timing' not in cfg['budget']
    assert any(c['task'] == f'A2-{arm}-price' and c['code'] == 0 for c in result['calls'])
    assert cfg['kit_commit'] == SYNTHETIC_KIT_COMMIT
    assert load(root/'archived-config.json')['kit_commit'] == SYNTHETIC_KIT_COMMIT
    reading = load(out/'reading.json')
    assert reading['status'] == 'TECHNICAL-ONLY' and reading['scientific_verdict'] is None
    returned = a2_return.read_archive(cfg['archive'])
    assert returned['contrasts'] == reading['contrasts']
    texts = sorted((root/'text/exploratory').glob('RETURN-TEXT-*.md'))
    for directory in (root/'text', root/'archived-text'):
        parts = sorted(directory.rglob('RETURN-TEXT-*.md'))
        assert parts
        assert all(SYNTHETIC_KIT_COMMIT in path.read_text() for path in parts)
    for path in texts:
        assert all(len(line) < 90 for line in path.read_text().splitlines())
    imported = a2_return.import_parts([p.read_text() for p in texts])
    assert imported['contrasts'] == reading['contrasts']
    changed = [p.read_text().replace('SHA256 ', 'SHA256 0') for p in texts]
    assert a2_return.import_parts(changed)['contrasts'] == imported['contrasts']
    with pytest.raises(CorrectnessStop): a2_return.import_parts([p.read_text() for p in texts[1:]])
    cells = load(jobs.completed(out, 'A2-'+arm+'-comparison')/'cells.json')
    cells[0]['physical']['uuid'] = 'different'
    assert a2_read.analyse(arm, cells)['status'] == 'UNVERIFIED-HARDWARE'
    assert a2_read.analyse(arm, cells[:-1])['status'] == 'INCOMPLETE'
    confirm = load(out/'confirmation-reading.json')
    assert confirm['status'] == 'TECHNICAL-ONLY' and confirm['admission'] is None
    if arm in ('R', 'RP'):
        cached = [load(p) for p in (out/'rewrites').glob('index-*.json')]
        assert len(cached) == 72 and sum(r['accepted'] for r in cached) == 71
        failed = [r for r in cached if not r['accepted']]
        assert failed[0]['fallback_reason'] == 'unsupported_question'
        assert failed[0]['response'] == 'Verified arithmetic.\nAnswer: 8.0'
        teaching = t.read_jsonl(out/'data/R-rewrite-s101-m2/teaching-contexts.jsonl')
        assert any(r['demonstration'].startswith('Procedure:') for r in teaching)
        assert any(r['id'] == 'fixture-1-0' and r['demonstration'] == failed[0]['response'] for r in teaching)
    if arm == 'J':
        for seed in SEEDS:
            for month in MONTHS[:2]:
                rows = t.read_jsonl(out/'current-ledger'/str(seed)/month/'traces.jsonl')
                assert all(r['collector_checkpoint']['model'] == 'synthetic-connected-'+str(seed) for r in rows)
        data = load(out/'data/J-current-s101-m4/prepared.json')
        assert data['incoming']['model'].endswith('J-current-s101-m2/hf-step40')
    if arm in ('P', 'RP'):
        data = load(out/'data/P-replay-s101-m2/prepared.json')
        assert data['audit']['scheduled_generic'] == 160 and data['audit']['scheduled_task'] == 1120
    if arm == 'RP':
        control = t.read_jsonl(out/'data/raw-plain-s101-m2/schedule.jsonl')
        rewrite = t.read_jsonl(out/'data/R-rewrite-s101-m2/schedule.jsonl')
        replay = t.read_jsonl(out/'data/P-replay-s101-m2/schedule.jsonl')
        assert control == rewrite
        assert all(a == b for a, b in zip(control, replay) if not b.get('p3b_replay'))
        assert set(reading['contrasts']) == {'R', 'P'} and reading['comparisons'] == 2


def test_cost_formula_and_actual_held_accounting(tmp_path):
    from kit.a2_budget import forecast, reconcile
    for arm in ('J', 'P', 'R'):
        assert forecast(arm, 6.3)['allowance_before_confirmation'] == 350
        assert forecast(arm, 6.7)['training_gpu_hours'] == pytest.approx(120.6)
    assert forecast('RP', 6.3)['training_gpu_hours'] == pytest.approx(170.1)
    assert forecast('RP', 6.7)['allowance_before_confirmation'] == 450
    assert forecast('P', 6.7, score_cell_hours=.3)['total_gpu_hours'] == pytest.approx(435.6)
    assert forecast('J', 6.7, collection=25, score_cell_hours=.3)['total_gpu_hours'] == pytest.approx(485.6)
    path = tmp_path/'attempts/a/b/receipt.json'
    atomic(path, dict(task='failed', phase='train', nonce='b', gpus=8, status='FAILED',
        technical_synthetic=False, slurm_job_id='123', allocation_started_at=100, allocation_gpu_hours=None))
    value = reconcile(tmp_path, False, query=lambda job, start: dict(allocation_ended_at=3700))
    assert value['gpu_hours'] == 8 and value['rows'][0]['status'] == 'FAILED'
    assert value['queue_gpu_hours'] == 0


def test_startless_attempt_without_allocation_reconciles_as_zero(tmp_path):
    from kit.a2_budget import reconcile
    receipt = tmp_path/'attempts/failed/no-start/receipt.json'
    atomic(receipt, dict(task='failed', phase='train', nonce='no-start', gpus=8,
                         status='FAILED', slurm_job_id=None, allocation_gpu_hours=None))
    pointer = tmp_path/'state/failed.json'
    atomic(pointer, dict(receipt=str(receipt.relative_to(tmp_path)), sha256=s.file_hash(receipt)))
    result = reconcile(tmp_path, False, query=lambda *a: pytest.fail('no allocation to query'))
    assert result['status'] == 'COMPLETE' and result['pending'] == []
    assert result['gpu_hours'] == result['rows'][0]['gpu_hours'] == 0
    assert result['rows'][0]['reason'] == 'no allocation started: no scheduler job ID or start time'
    assert load(receipt)['allocation_gpu_hours'] == 0
    assert load(pointer)['sha256'] == s.file_hash(receipt)
    assert reconcile(tmp_path, False) == result


def test_startless_attempt_with_live_allocation_stays_pending(tmp_path):
    from kit.a2_budget import reconcile
    receipt = tmp_path/'attempts/failed/live/receipt.json'
    atomic(receipt, dict(task='failed', phase='train', nonce='live', gpus=8,
                         status='FAILED', slurm_job_id='CPU-STANDIN', allocation_gpu_hours=None))
    result = reconcile(tmp_path, False, query=lambda *a: dict(allocation_started_at=100))
    assert result['status'] == 'PENDING' and result['pending'] == ['live']
    assert load(receipt)['allocation_gpu_hours'] is None


def test_zero_maps_and_single_zero_seed_are_not_replaced(tmp_path):
    cfg = fixture(tmp_path/'zero', 'P')
    source = load(checked(cfg['i0']))
    original = copy.deepcopy(source)
    for seed in SEEDS:
        spec = load(checked(source['maps'][str(seed)]))
        spec.update(slots=[[] for _ in range(40)], accepted_order=[])
        path = tmp_path/f'zero-{seed}.json'; atomic(path, spec)
        source['maps'][str(seed)] = ref(path)
    assert not a2_replay.applicable(source)
    source['maps']['102'] = original['maps']['102']
    assert a2_replay.applicable(source)
    task = [dict(id='task', prompt='q', response='Answer: 1')]*1280
    rows, audit = a2_replay.transfer(task, source, 101, [])
    assert rows == task and audit['generic_positions'] == 0


def test_reader_confirmation_supply_and_terminal_guards():
    physical = dict(allocation_nonce='one', uuid='GPU-one', environment='fixed')
    cells = []
    for key in a2_read.confirmation_plan('R'):
        challenger = key[1] == 'R-rewrite'
        primary = key[0] == 'confirmation'
        rows = [dict(question_id=str(i), correct=int(i < (9 if challenger and primary else 8)),
                     holdout=int(key[4].startswith('2026-')), family='cash_conversion_change') for i in range(10)]
        cells.append(dict(key=key, physical=physical, rows=rows,
                          expected_members={r['question_id']: dict(holdout=bool(r['holdout']), family=r['family']) for r in rows}))
    receipt = dict(terminal_june_frozen=True, audit_verified=True, supply={
        m: dict(questions=40, holdouts=10, components=5) for m in ('2026-10', '2026-11')})
    assignment = dict(status='COMPLETE', nonce='one', allocated_uuids=['GPU-one'])
    result = a2_read.analyse('R', cells, confirmation=True, receipt=receipt, assignment=assignment)
    assert result['contrasts']['R']['primary']['mean'] == 10
    assert result['contrasts']['R']['verdict'] == 'CONFIRMED PASS'
    invalid = a2_read.analyse('R', cells, confirmation=True, receipt=receipt)
    assert invalid['status'] == 'UNVERIFIED-HARDWARE'
    assert invalid['contrasts']['R']['verdict'] is None
    assert len(result['contrasts']['R']['guards']) == 15
    assert result['scores']['confirmation/R-rewrite/101/6/2026-10/on']['equal_month_granularity_points'] == 5
    receipt['supply']['2026-10']['components'] = 4
    result = a2_read.analyse('R', cells, confirmation=True, receipt=receipt)
    assert result['status'] == 'PENDING' and result['reason'] == 'insufficient confirmation evidence'


def test_large_text_records_still_fit_parts(tmp_path):
    payload = dict(arm='R', registration_sha256=REGISTRATION, status='ARCHIVED UNREAD',
                   diagnostics={'large': ''.join(digest(str(i).encode())+'\u03b1' for i in range(5000))})
    paths = a2_return.write_parts(tmp_path/'parts', payload, '1'*40)
    assert len(paths) > 1
    assert all(len(p.read_text()) <= 11400 and all(len(x) < 90 for x in p.read_text().splitlines()) for p in paths)
    assert a2_return.import_parts([p.read_text() for p in reversed(paths)])['status'] == 'ARCHIVED UNREAD'


def test_empty_cumulative_pool_skips_both_arms_without_replacement(tmp_path):
    from kit.a2_pipeline import prepare_training, train, export, state
    cfg = fixture(tmp_path/'empty', 'P')
    doc = bind_inputs(cfg)
    for month in MONTHS[:2]:
        path = checked(doc['inputs']['ledger'][month])
        t.write_jsonl(path.parent/'traces.jsonl', [])
        freeze(path.parent, ['traces.jsonl', 'admission.jsonl', 'sealed.json'])
        doc['inputs']['ledger'][month] = ref(path)
    out = Path(cfg['out'])
    for model in arm_names('P'):
        row = dict(model=model, seed=101, month=2, id='synthetic-empty')
        attempt = tmp_path/('attempt-'+model); attempt.mkdir()
        prepare_training(cfg, doc, row, attempt)
        train(cfg, doc, row, attempt)
        export(cfg, doc, row)
        assert state(out, doc, model, 101, 2) == {'model': 'synthetic-base'}
        summary = load(out/'runs'/(model+'-s101-m2')/'summary.json')
        assert summary['skipped'] and summary['updates'] == 0
        assert summary['audit']['scheduled_task'] == summary['audit']['scheduled_generic'] == 0


def qualification_standin(root, arm, cap=601):
    """Measured CPU evidence for budget adversarial tests; never a GPU claim."""
    from kit import a2_budget as budget
    cfg = fixture(root, arm)
    cfg['budget']['approved_gpu_hours'] = cap
    # Real configs carry kit_commit (the node check requires it); exports have no Git.
    cfg['kit_commit'] = SYNTHETIC_KIT_COMMIT
    atomic(Path(root)/'config.json', cfg)
    out = Path(cfg['out'])
    bind(out/'config.json', budget.sealed_config(cfg))
    rows = jobs.configured_inventory(cfg)['rows']
    for row in rows:
        if not row.get('qualification'):
            continue
        attempt = out/'attempts'/row['id']/'standin'
        artifacts = {}
        seconds = 2880 if row['phase'] == 'train' else 360*row.get('commands', 1) if row['phase'] == 'score' else max(1, row.get('commands', 1))*3.6
        if row['phase'] == 'train':
            name = f"{row['model']}-s101-m2"
            data = out/'data'/name
            atomic(data/'prepared.json', dict(empty=False, recipe='D+replay' if row['model'] == 'P-replay' else 'D'))
            freeze(data, ['prepared.json'])
            setup = dict(recipe='D+replay' if row['model'] == 'P-replay' else 'D',
                         effective_command=['unchanged registered stand-in'], technical_synthetic=True)
            atomic(attempt/'timing-setup.json', setup)
            artifacts[str((attempt/'timing-setup.json').relative_to(out))] = s.file_hash(attempt/'timing-setup.json')
        receipt = attempt/'receipt.json'
        atomic(receipt, dict(task=row['id'], phase=row['phase'], nonce='standin', status='COMPLETE',
                             gpus=row['gpus'], technical_synthetic=True, started_at=100,
                             ended_at=100+seconds, allocation_gpu_hours=row['gpus']*seconds/3600,
                             artifacts=artifacts))
        atomic(out/'state'/(row['id']+'.json'), dict(receipt=str(receipt.relative_to(out)), sha256=s.file_hash(receipt)))
    atomic(root/'config.json', cfg)
    return cfg, rows


@pytest.mark.parametrize('arm', ['J', 'P', 'R', 'RP'])
def test_zero_receipts_price_and_proceed_each_arm(tmp_path, arm):
    from kit import a2_budget as budget
    cfg, rows = qualification_standin(tmp_path/arm, arm)
    out = Path(cfg['out'])
    assert 'D_timing' not in cfg['budget'] and 'challenger_timing' not in cfg['budget']
    assert not (out/'qualification').exists()
    assert jobs.run(tmp_path/arm/'config.json', f'A2-{arm}-price') == 0
    price = budget.gate(cfg, out)
    assert price['decision'] == 'PROCEED' and price['total_gpu_hours'] <= 601
    assert set(price['timing_refs']) == set(arm_names(arm))
    assert price['qualification_gpu_hours'] > 0
    assert price['components']['preparation'] == 0
    assert price['components']['confirmation_cells'] > 0
    assert price['components']['loading'] and price['components']['failures_retries']
    if arm in ('R', 'RP'):
        assert price['components']['rewrite'] > 0
    if arm == 'J':
        assert price['components']['collection_cells'] == 18
    assert price['training_gpu_hours'] == pytest.approx(9*len(arm_names(arm))*6.4)
    price_index = next(i for i, r in enumerate(rows) if r['phase'] == 'price')
    assert rows[price_index-1]['phase'] == 'train'
    assert all(r.get('qualification') for r in rows[:price_index])
    assert all(any(d['id'] == f'A2-{arm}-price' for d in r['depends_on']) for r in rows[price_index+1:])
    # A real budget reservation after pricing uses the same accounting path.
    row = next(r for r in rows if not r.get('qualification') and r['phase'] == 'train')
    record = dict(nonce='next', allocation_started_at=__import__('time').time())
    budget.reserve(cfg, out, record, row)
    assert record['reserved_gpu_hours'] > 32
    assert load(out/'config.json') == budget.sealed_config(cfg)


def test_over_cap_stops_every_later_row_and_returns_spend(tmp_path):
    from kit import a2_budget as budget
    cfg, rows = qualification_standin(tmp_path/'P', 'P', cap=100)
    # Test-owned identity: an exported, Git-isolated kit has no HEAD to read.
    cfg['kit_commit'] = SYNTHETIC_KIT_COMMIT
    out = Path(cfg['out'])
    (out/'config.json').unlink()
    atomic(tmp_path/'P/config.json', cfg)
    assert jobs.run(tmp_path/'P/config.json', 'A2-P-price') == PENDING
    with pytest.raises(NotReady, match='owner escalation.*qualification spend'):
        budget.gate(cfg, out)
    for row in rows:
        if row.get('qualification') or row['phase'] in ('price', 'collect'):
            continue
        # Even rows whose scientific predecessors finished cannot bypass price.
        with pytest.raises(NotReady):
            budget.reserve(cfg, out, dict(nonce='later', allocation_started_at=0), row)
    assert jobs.run(tmp_path/'P/config.json', 'A2-P-collect') == 0
    sealed = jobs.completed(out, 'A2-P-collect')
    assert load(sealed/'receipt.json')['status'] == 'COMPLETE'
    original = load(out/'state/A2-P-collect.json')
    assert jobs.run(tmp_path/'P/config.json', 'A2-P-collect') == 0
    assert load(out/'state/A2-P-collect.json') != original
    assert list((Path(cfg['text_out'])/'exploratory/superseded').rglob('RETURN-TEXT-*.md'))
    returned = a2_return.read_archive(cfg['archive'])
    assert returned['status'] == 'PENDING'
    assert returned['qualification']['total_gpu_hours'] > 100
    assert returned['qualification']['qualification_gpu_hours'] == returned['accounting']['gpu_hours'] > 0
    parts = sorted((Path(cfg['text_out'])/'exploratory').glob('RETURN-TEXT-*.md'))
    imported = a2_return.import_parts([p.read_text() for p in parts])
    assert imported['qualification'] == returned['qualification']
    assert imported['accounting'] == returned['accounting']


def test_completed_legacy_escalation_price_retry_still_exits_pending(tmp_path):
    cfg, _ = qualification_standin(tmp_path/'P', 'P', cap=100)
    config = tmp_path/'P/config.json'
    out = Path(cfg['out'])
    assert jobs.run(config, 'A2-P-price') == PENDING
    pointer_path = out/'state/A2-P-price.json'
    pointer = load(pointer_path)
    receipt = out/pointer['receipt']
    value = load(receipt)
    value['status'] = 'COMPLETE'
    value.pop('price_decision')
    atomic(receipt, value)
    atomic(pointer_path, dict(receipt=pointer['receipt'], sha256=s.file_hash(receipt)))
    # A prior successful execution of the price computation must still obey its
    # escalation decision when used as a new scheduler dependency.
    assert jobs.run(config, 'A2-P-price') == PENDING
    assert load(out/'qualification/price.json')['decision'] == 'OWNER ESCALATION'


@pytest.mark.parametrize('failure', ['missing-stage', 'incomplete-price', 'killed-price'])
def test_unpriced_collect_writes_return_without_sealing(tmp_path, failure):
    cfg, rows = qualification_standin(tmp_path/'P', 'P')
    cfg['kit_commit'] = SYNTHETIC_KIT_COMMIT
    out = Path(cfg['out'])
    (out/'config.json').unlink()
    config = tmp_path/'P/config.json'
    atomic(config, cfg)
    if failure == 'killed-price':
        # A failed qualification kills its afterok price job before it ever runs.
        task = next(r['id'] for r in rows if r['phase'] == 'train')
        state = out/'state'/(task+'.json')
        receipt = out/load(state)['receipt']
        value = load(receipt)
        value['status'] = 'FAILED'
        atomic(receipt, value)
        atomic(state, dict(receipt=str(receipt.relative_to(out)), sha256=s.file_hash(receipt)))
        with pytest.raises(NotReady):
            jobs.completed(out, task)
        assert not (out/'state/A2-P-price.json').exists()
        assert not (out/'attempts/A2-P-price').exists()
    elif failure == 'missing-stage':
        task = next(r['id'] for r in rows if r['phase'] == 'train')
        (out/'state'/(task+'.json')).unlink()
        assert jobs.run(config, 'A2-P-price') == PENDING
    else:
        assert jobs.run(config, 'A2-P-price') == 0
        receipt = jobs.completed(out, 'A2-P-price')/'receipt.json'
        value = load(receipt)
        value['status'] = 'PENDING'
        atomic(receipt, value)
        atomic(out/'state/A2-P-price.json', dict(receipt=str(receipt.relative_to(out)),
                                               sha256=s.file_hash(receipt)))
    assert jobs.run(config, 'A2-P-collect') == PENDING
    with pytest.raises(NotReady):
        jobs.completed(out, 'A2-P-collect')
    returned = a2_return.read_archive(cfg['archive'])
    assert returned['qualification']['status'] == returned['qualification']['decision'] == 'NOT PRICED'
    assert returned['qualification']['qualification_gpu_hours'] == returned['accounting']['gpu_hours']
    assert returned['accounting']['gpu_hours'] > 0
    parts = sorted(Path(cfg['text_out']).rglob('RETURN-TEXT-*.md'))
    imported = a2_return.import_parts([p.read_text() for p in parts])
    assert all(imported[k] == v for k, v in returned.items())


def test_lagging_price_retry_then_real_collect(tmp_path, monkeypatch):
    """Run the real wrapper/pipeline with local CPU telemetry and scheduler stand-ins."""
    import os
    import sys
    from kit.a2_budget import gate
    root = tmp_path/'P'
    cfg = fixture(root, 'P')
    cfg['kit_commit'] = SYNTHETIC_KIT_COMMIT
    config = root/'config.json'
    atomic(config, cfg)
    binary = root/'bin'
    binary.mkdir()
    smi = binary/'nvidia-smi'
    smi.write_text('#!'+sys.executable+'\nimport sys\n'
                   'q=next((x for x in sys.argv if x.startswith("--query-")), "")\n'
                   'if q=="--query-gpu=uuid": print("GPU-CPU-STANDIN-0")\n'
                   'elif "utilization.gpu" in q: print("\\n".join(f"{i}, GPU-CPU-STANDIN-{i}, 10, 100" for i in range(8)))\n'
                   'else: print("\\n".join(f"{i}, GPU-CPU-STANDIN-{i}" for i in range(8)))\n')
    smi.chmod(0o755)
    for name in ('scontrol', 'sacct'):
        command = binary/name
        command.write_text('#!/bin/sh\nexit 0\n')
        command.chmod(0o755)
    for key in list(os.environ):
        if key.startswith(('SLURM_', 'CUDA_', 'M2_', 'M3_', 'UTIL_', 'KIT_LANE_')):
            monkeypatch.delenv(key)
    monkeypatch.setenv('PATH', str(binary)+os.pathsep+os.environ['PATH'])
    rows = jobs.configured_inventory(cfg)['rows']
    out = Path(cfg['out'])
    for row in rows:
        if row.get('qualification'):
            assert jobs.run(config, row['id']) == 0
    lag = out/'attempts/lagging-record/standin/receipt.json'
    value = dict(task='lagging-record', phase='score', nonce='standin', gpus=1,
                 status='FAILED', technical_synthetic=True, started_at=100, ended_at=101,
                 allocation_started_at=100, slurm_job_id='CPU-LAG-STANDIN', allocation_gpu_hours=None)
    atomic(lag, value)
    assert jobs.run(config, 'A2-P-price') == PENDING
    assert jobs.run(config, 'A2-P-collect') == PENDING
    assert a2_return.read_archive(cfg['archive'])['qualification']['status'] == 'NOT PRICED'
    with pytest.raises(NotReady):
        jobs.completed(out, 'A2-P-collect')
    value.update(allocation_ended_at=101, allocation_gpu_hours=1/3600)
    atomic(lag, value)
    assert jobs.run(config, 'A2-P-price') == 0
    assert gate(cfg, out)['decision'] == 'PROCEED'
    for row in rows:
        if not row.get('qualification') and row['phase'] != 'price' and not row['phase'].startswith('confirmation'):
            assert jobs.run(config, row['id']) == 0
    jobs.completed(out, 'A2-P-collect')
    reading = load(out/'reading.json')
    returned = a2_return.read_archive(cfg['archive'])
    assert returned['status'] == reading['status'] == 'TECHNICAL-ONLY'
    assert returned['contrasts'] == reading['contrasts']
    assert returned == load(out/'returned-reading.json')
    assert returned['qualification']['decision'] == 'PROCEED'


@pytest.mark.parametrize('target', ['timing', 'price', 'source', 'setup'])
def test_tampered_price_or_timing_refuses(tmp_path, target):
    from kit import a2_budget as budget
    cfg, _ = qualification_standin(tmp_path/'R', 'R')
    out = Path(cfg['out'])
    assert jobs.run(tmp_path/'R/config.json', 'A2-R-price') == 0
    value = budget.gate(cfg, out)
    timing_ref = value['timing_refs']['R-raw']
    timing = load(checked(timing_ref))
    path = {'timing': Path(timing_ref['path']), 'price': out/'qualification/price.json',
            'source': Path(timing['source']['path']), 'setup': Path(timing['effective_setup']['path'])}[target]
    path.write_text(path.read_text()+' ')
    with pytest.raises(CorrectnessStop, match='changed'):
        budget.gate(cfg, out)


@pytest.mark.parametrize('four_a_seconds', [1440, 2880, 4320])
def test_matching_4a_D_reuse_and_mismatched_setup_refusal(tmp_path, four_a_seconds):
    from kit import a2_budget as budget
    cfg, _ = qualification_standin(tmp_path/'P', 'P')
    out = Path(cfg['out'])
    # Make an existing 4A receipt with its own original evidence, then bind it
    # into the initial config, before any wrapper seals or executes a row.
    source = out/'attempts/A2-P-P-plain-s101-m2-train/standin'
    setup = load(source/'timing-setup.json')
    effective = tmp_path/'4a-effective.json'; atomic(effective, setup)
    schedule = tmp_path/'4a-schedule.jsonl'; write_jsonl(schedule, [dict(id=str(i)) for i in range(1280)])
    original = tmp_path/'4a-allocation.json'
    allocation = load(source/'receipt.json')
    allocation.update(ended_at=allocation['started_at']+four_a_seconds,
                      allocation_gpu_hours=8*four_a_seconds/3600)
    atomic(original, allocation)
    timing = tmp_path/'4a-timing.json'
    measured = dict(qualified_8b=True, gpus=8, seconds=four_a_seconds, technical_synthetic=True,
                    origin='4A', setup=setup, source=ref(original), effective_setup=ref(effective),
                    prepared=ref(out/'data/P-plain-s101-m2/manifest.json'),
                    effective_config=ref(effective), schedule=ref(schedule))
    atomic(timing, measured)
    cfg['budget']['D_timing'] = ref(timing)
    # The helper built source evidence only; reset its config seal before launch.
    (out/'config.json').unlink()
    atomic(tmp_path/'P/config.json', cfg)
    assert jobs.run(tmp_path/'P/config.json', 'A2-P-price') == 0
    price = budget.gate(cfg, out)
    choice = price['D_timing_choice']
    assert choice['A2']['seconds'] == 2880
    assert choice['4A'] == dict(ref=ref(timing), seconds=four_a_seconds)
    assert choice['selected'] == ('4A' if four_a_seconds >= 2880 else 'A2')
    assert choice['seconds'] == max(2880, four_a_seconds)
    assert price['timing_refs']['P-plain'] == choice[choice['selected']]['ref']
    assert load(checked(choice['A2']['ref']))['seconds'] == 2880
    assert price['training_gpu_hours'] == pytest.approx(9*8*(2880+max(2880, four_a_seconds))/3600)
    wrong = dict(measured, setup=dict(setup, recipe='F'))
    atomic(timing, wrong)
    with pytest.raises(CorrectnessStop):
        budget.gate(cfg, out)
    # Even with a matching hash, mismatched recipe/setup is invalid evidence.
    with pytest.raises(CorrectnessStop, match='recipe/setup'):
        budget.validate_timing(ref(timing), setup, True, external=True)


def test_qualification_cannot_escape_cap_and_seal(tmp_path):
    from kit import a2_budget as budget
    cfg, rows = qualification_standin(tmp_path/'J', 'J')
    row = next(r for r in rows if r['gpus'])
    with pytest.raises(CorrectnessStop, match='601'):
        budget.reserve(dict(cfg, budget=dict(cfg['budget'], approved_gpu_hours=602)), Path(cfg['out']),
                       dict(nonce='over', allocation_started_at=0), row)
    cfg['budget']['approved_gpu_hours'] = 1
    with pytest.raises(NotReady, match='remaining approved'):
        budget.reserve(cfg, Path(cfg['out']), dict(nonce='over', allocation_started_at=0), row)
    atomic(tmp_path/'J/config.json', cfg)
    with pytest.raises(CorrectnessStop, match='changed'):
        jobs.run(tmp_path/'J/config.json', 'A2-J-price')


def test_refused_real_allocation_is_still_in_accounting(tmp_path, monkeypatch):
    """No trainer, device probe or scheduler command runs in this stand-in."""
    from kit import a2_budget as budget
    import time
    cfg = fixture(tmp_path/'P', 'P')
    cfg['technical_synthetic'] = False
    config = tmp_path/'P/config.json'; atomic(config, cfg)
    start = time.time()-1
    monkeypatch.setenv('SLURM_JOB_ID', 'CPU-ACCOUNTING-STANDIN')
    monkeypatch.setenv('SLURM_JOB_START_TIME', str(start))
    # Arrival lacks its CPU preparation. The allocation must be recorded anyway.
    assert jobs.run(config, 'A2-P-arrival-m1') == PENDING
    result = budget.reconcile(Path(cfg['out']), False,
                              query=lambda job, begun: dict(allocation_ended_at=start+3600))
    assert result['status'] == 'COMPLETE' and result['gpu_hours'] == 1
    assert result['rows'][0]['status'] == 'PENDING'


def test_post_seal_timing_ref_edits_are_refused(tmp_path):
    cfg, _ = qualification_standin(tmp_path/'R', 'R')
    cfg['budget']['D_timing'] = dict(path='/absent', sha256='0'*64)
    atomic(tmp_path/'R/config.json', cfg)
    with pytest.raises(CorrectnessStop, match='changed'):
        jobs.run(tmp_path/'R/config.json', 'A2-R-price')


def test_export_only_recovery_cannot_underprice_first_training_stage(tmp_path):
    from kit import a2_budget as budget
    cfg, _ = qualification_standin(tmp_path/'P', 'P')
    out = Path(cfg['out'])
    task = 'A2-P-P-plain-s101-m2-train'
    completed = out/'attempts'/task/'standin/receipt.json'
    prior = load(completed)
    prior.update(nonce='interrupted', status='FAILED')
    atomic(out/'attempts'/task/'interrupted/receipt.json', prior)
    recovery = load(completed)
    recovery.update(ended_at=101, allocation_gpu_hours=8/3600)
    atomic(completed, recovery)
    atomic(out/'state'/(task+'.json'), dict(receipt=str(completed.relative_to(out)), sha256=s.file_hash(completed)))
    assert jobs.run(tmp_path/'P/config.json', 'A2-P-price') == 0
    price = budget.gate(cfg, out)
    timing = load(checked(price['timing_refs']['P-plain']))
    assert timing['seconds'] == 2881
    assert len(timing['attempts']) == 2
    assert price['training_gpu_hours'] > 9*6.4


def test_startless_failed_training_attempt_does_not_block_price(tmp_path):
    from kit import a2_budget as budget
    cfg, _ = qualification_standin(tmp_path/'P', 'P')
    out = Path(cfg['out'])
    task = 'A2-P-P-plain-s101-m2-train'
    failed = out/'attempts'/task/'no-start/receipt.json'
    atomic(failed, dict(task=task, phase='train', nonce='no-start', status='FAILED',
                        gpus=8, technical_synthetic=True, started_at=1, ended_at=2,
                        slurm_job_id=None, allocation_gpu_hours=None))
    assert jobs.run(tmp_path/'P/config.json', 'A2-P-price') == 0
    price = budget.gate(cfg, out)
    timing = load(checked(price['timing_refs']['P-plain']))
    assert timing['seconds'] == 2880 and len(timing['attempts']) == 2
    assert load(failed)['allocation_gpu_hours'] == 0
    assert next(r for r in budget.reconcile(out, True)['rows'] if r['nonce'] == 'no-start')['reason']


def test_RP_zero_generic_map_preserves_R_only_qualification(tmp_path):
    from kit import a2_budget as budget
    cfg, _ = qualification_standin(tmp_path/'RP', 'RP')
    replay = load(checked(cfg['i0']))
    for seed in SEEDS:
        path = checked(replay['maps'][str(seed)])
        spec = load(path)
        spec.update(slots=[[] for _ in range(40)], accepted_order=[])
        atomic(path, spec)
        replay['maps'][str(seed)] = ref(path)
    path = checked(cfg['i0']); atomic(path, replay)
    cfg['i0'] = ref(path)
    out = Path(cfg['out'])
    (out/'config.json').unlink()  # Source-only stand-ins precede the initial seal.
    atomic(tmp_path/'RP/config.json', cfg)
    rows = jobs.configured_inventory(cfg)['rows']
    assert all(r['gpus'] == 0 for r in rows if r.get('model') == 'P-replay')
    assert jobs.run(tmp_path/'RP/config.json', 'A2-RP-price') == 0
    price = budget.gate(cfg, out)
    assert set(price['timing_refs']) == {'raw-plain', 'R-rewrite'}
    assert price['training_gpu_hours'] == pytest.approx(18*6.4)
    assert 'P-replay' not in price['standalone_training_gpu_hours']


def test_completed_stage_replay_charges_without_changing_price_snapshot(tmp_path):
    from kit import a2_budget as budget
    cfg, _ = qualification_standin(tmp_path/'J', 'J')
    config = tmp_path/'J/config.json'
    out = Path(cfg['out'])
    assert jobs.run(config, 'A2-J-price') == 0
    price = budget.gate(cfg, out)
    task = 'A2-J-J-frozen-s101-m2-train'
    original = load(out/'state'/(task+'.json'))
    assert jobs.run(config, task) == 0
    assert len(list((out/'attempts'/task).glob('*/receipt.json'))) == 2
    assert load(out/'state'/(task+'.json')) == original
    assert budget.gate(cfg, out) == price


def test_return_config_identity_uses_sealed_dictionary(tmp_path):
    from kit.a2_budget import sealed_config
    cfg = fixture(tmp_path/'J', 'J')
    cfg['kit_commit'] = SYNTHETIC_KIT_COMMIT
    out = Path(cfg['out'])
    atomic(out/'config.json', sealed_config(cfg))
    identity = a2_return.return_identity(cfg)
    assert identity['config_sha256'] == digest(sealed_config(cfg))
    assert identity['config_file_sha256'] == digest((out/'config.json').read_bytes())
    assert identity['config_sha256'] != identity['config_file_sha256']
    # Confirmation permission changes are excluded from the sealed dictionary.
    cfg.update(allow_confirmation=True, confirmation=dict(supply='later'))
    assert a2_return.return_identity(cfg) == identity
    # Pretty-printing changes only the separately named file digest.
    (out/'config.json').write_bytes(canonical(sealed_config(cfg)))
    reformatted = a2_return.return_identity(cfg)
    assert reformatted['config_sha256'] == identity['config_sha256']
    assert reformatted['config_file_sha256'] != identity['config_file_sha256']


@pytest.mark.parametrize('priced', [False, True])
def test_price_return_and_identity_share_sealed_digest(tmp_path, priced):
    from kit.a2_budget import sealed_config
    cfg, _ = qualification_standin(tmp_path/'J', 'J')
    out = Path(cfg['out'])
    if priced:
        assert jobs.run(tmp_path/'J/config.json', 'A2-J-price') == 0
    with pytest.raises(NotReady, match='qualification return written'):
        a2_return.qualification_return(cfg, out)
    paths = sorted((Path(cfg['text_out'])/'exploratory').glob('RETURN-TEXT-ESSENTIAL-*.md'))
    result = a2_return.import_parts([p.read_text() for p in paths])
    assert result['qualification']['decision'] == ('PROCEED' if priced else 'NOT PRICED')
    assert result['qualification']['config_sha256'] == result['identity']['config_sha256'] == digest(sealed_config(cfg))


def count_fixture(arm, confirmation=False, real_scale=False):
    """Binary units with stable pair IDs and real population sizes when requested."""
    plan = a2_read.confirmation_plan(arm) if confirmation else a2_read.score_plan(arm)
    counts = dict(zip(MONTHS, (37, 80, 80, 75, 80, 43)))
    held = dict(zip(MONTHS, (5, 24, 19, 14, 20, 10)))
    physical = dict(allocation_nonce='one', uuid='GPU-one', environment='fixed')
    cells = []
    for key in plan:
        kind, model, seed, stage, panel, mode = key
        sec = panel.startswith('2026-')
        n = (counts.get(panel, 40) if sec else 100) if real_scale else 12
        holdouts = held.get(panel, 10) if real_scale else 6
        ids = range(n) if kind in ('arrival', 'base') or not sec else range(holdouts)
        rows = [dict(question_id=f'{panel}-{i}', correct=int(digest([*key, i]), 16) % 2,
                     historical_correct=int(i % 3 == 0), holdout=int(sec and i < holdouts),
                     family=FAMILIES[i % 2] if sec else '', input_tokens=100+i, output_tokens=20+i)
                for i in ids]
        cells.append(dict(key=list(key), physical=physical, rows=rows,
                          expected_members={r['question_id']: dict(holdout=bool(r['holdout']), family=r['family'])
                                            for r in rows}))
    return dict(arm=arm, registration_sha256=REGISTRATION, cells=cells, technical_synthetic=False,
                p_applicable=True, confirmation=confirmation, accounting=dict(gpu_hours=12),
                identity=dict(kit_commit='1'*40, config_sha256='2'*64, protocol_sha256='3'*64),
                assignment=dict(status='COMPLETE', nonce='one', allocated_uuids=['GPU-one']),
                confirmation_receipt=dict(terminal_june_frozen=True, audit_verified=True, supply={
                    m: dict(questions=40, holdouts=10, components=5) for m in ('2026-10', '2026-11')}))


def split_texts(paths, mode):
    return [p.read_text() for p in paths if f'-{mode}-' in p.name]


def without_transcription(reading):
    return {k: v for k, v in reading.items() if k != 'transcription'}


def item_bar_oracle(payload):
    """Independent item joins and interval arithmetic, with no count-reader calls."""
    import statistics
    cells = {tuple(c['key']): c['rows'] for c in payload['cells']}
    confirmation = payload['confirmation']

    def interval(values, primary):
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        radius = 4.302653 * sd / math.sqrt(3)
        positive = sum(v > 0 for v in values)
        passed = mean >= 2 and mean-radius > 0 and positive == 3 if primary else mean-radius >= -1
        return dict(values=values, mean=mean, sd=sd, lower=mean-radius, upper=mean+radius,
                    positive=positive, n=3, df=2, t=4.302653, pass_bar=passed)

    result = {}
    transitions = []
    # Explicit contrasts keep this oracle independent of the runtime contrast lookup.
    systems = {'J': [('J', 'J-frozen', 'J-current')],
               'P': [('P', 'P-plain', 'P-replay')],
               'R': [('R', 'R-raw', 'R-rewrite')],
               'RP': [('P', 'raw-plain', 'P-replay'), ('R', 'raw-plain', 'R-rewrite')]}
    for label, control, challenger in systems[payload['arm']]:
        def paired(kind, seed, stage, panel, mode):
            a, b = ({r['question_id']: r['correct'] for r in cells[(kind, model, seed, stage, panel, mode)]
                     if panel in GENERAL or r['holdout']} for model in (control, challenger))
            assert a and a.keys() == b.keys()
            n = len(a)
            delta = 100*sum(b[q]-a[q] for q in a)/n
            transitions.append(dict(contrast=label, kind=kind, seed=seed, month=stage, panel=panel,
                mode=mode, n=n, delta=delta, granularity_points=100/n,
                correct_to_wrong=sum(a[q] == 1 and b[q] == 0 for q in a),
                wrong_to_correct=sum(a[q] == 0 and b[q] == 1 for q in a)))
            return delta

        months = ('2026-10', '2026-11') if confirmation else MONTHS[1:]
        monthly = {str(seed): [paired('confirmation' if confirmation else 'arrival', seed,
                    6 if confirmation else MONTHS.index(month)+1, month, 'on') for month in months]
                   for seed in SEEDS}
        primary = interval([sum(monthly[str(seed)])/len(months) for seed in SEEDS], True)
        guards = {}
        for stage in ((6,) if confirmation else (2, 4, 6)):
            for panel in (*MONTHS[:stage], 'ifeval', 'knowledge', 'math'):
                for mode in (('off',) if panel in GENERAL else ('on', 'off')):
                    bar = interval([paired('terminal' if confirmation else 'post', seed,
                                           stage, panel, mode) for seed in SEEDS], False)
                    bar['item_denominators'] = [t['n'] for t in transitions[-3:]]
                    bar['item_granularity_points'] = [t['granularity_points'] for t in transitions[-3:]]
                    guards[f'm{stage}/{panel}/{mode}'] = bar
        passed = primary['pass_bar'] and all(g['pass_bar'] for g in guards.values())
        verdict = ('CONFIRMED PASS' if confirmation else 'EXPLORATORY PASS') if passed else 'INCONCLUSIVE'
        result[label] = dict(primary=primary, monthly=monthly, guards=guards, verdict=verdict)
    return result, transitions


@pytest.mark.parametrize('arm', ['J', 'P', 'R', 'RP'])
@pytest.mark.parametrize('confirmation', [False, True])
@pytest.mark.parametrize('real_scale', [False, True])
def test_count_transport_exact_reading_and_size(tmp_path, arm, confirmation, real_scale):
    payload = count_fixture(arm, confirmation, real_scale)
    paths = a2_return.write_parts(tmp_path, payload, '1'*40)
    expected = a2_return.reproduce(payload)
    oracle, transitions = item_bar_oracle(payload)
    assert expected['contrasts'] == oracle
    assert expected['transitions'] == transitions
    assert all(c['primary']['sd'] > 0 for c in oracle.values())
    essential, full = (split_texts(paths, mode) for mode in ('ESSENTIAL', 'FULL'))
    assert without_transcription(a2_return.import_parts(essential)) == expected
    assert without_transcription(a2_return.import_parts(full)) == expected
    assert without_transcription(a2_return.import_parts(essential+full)) == expected
    assert len(essential) <= 4 and sum(len(t.splitlines()) for t in essential) <= 300
    compact = a2_return._import_parts(essential, payload_only=True)
    assert 'cells' not in compact and 'expected_members' not in json.dumps(compact)
    assert compact['identity'] == payload['identity']
    assert compact['keyed_counts']['pairs']
    for i, j, holdout, n, cc, cw, wc, ww in compact['keyed_counts']['pairs']:
        assert cc+cw+wc+ww == n and type(holdout) is bool
    assert bool(compact['keyed_counts']['paths']) != confirmation
    assert all(t.isascii() and max(map(len, t.splitlines())) < 90 for t in essential+full)


@pytest.mark.parametrize('change', ['swapped_seed', 'off_by_one_stage', 'swapped_memory_mode'])
@pytest.mark.parametrize('confirmation', [False, True])
def test_item_oracle_detects_count_key_mutations(change, confirmation):
    payload = count_fixture('RP', confirmation=confirmation, real_scale=True)
    oracle, transitions = item_bar_oracle(payload)
    compact = a2_return.compact_counts(payload)
    assert a2_return.reproduce(compact)['contrasts'] == oracle
    if change == 'off_by_one_stage':
        next(c[0] for c in compact['keyed_counts']['cells'] if c[0][1] == 'P-replay')[3] += 1
        with pytest.raises(CorrectnessStop, match='missing keyed score cells'):
            a2_return.reproduce(compact)
        return
    # Relabel the item cells and recapture consistent counts. A valid 2x2 table
    # can still represent the wrong seed/mode join, which the oracle must catch.
    keys = [c['key'] for c in payload['cells']]
    for key in keys:
        if change == 'swapped_seed':
            key[2] = {101: 102, 102: 101}.get(key[2], key[2])
        elif key[4] in MONTHS and key[0] != 'confirmation':
            key[5] = {'on': 'off', 'off': 'on'}[key[5]]
    mutated = a2_return.reproduce(a2_return.compact_counts(payload))
    assert mutated['contrasts'] != oracle
    assert mutated['transitions'] != transitions


def test_mutated_count_changes_bar_and_refuses_full(tmp_path):
    payload = count_fixture('R', confirmation=True)
    paths = a2_return.write_parts(tmp_path/'original', payload, '1'*40)
    essential = split_texts(paths, 'ESSENTIAL')
    compact = a2_return._import_parts(essential, payload_only=True)
    before = a2_return.import_parts(essential)
    counts = compact['keyed_counts']
    pair = counts['pairs'][0]
    # One wrong/wrong answer becomes wrong/correct. Update its absolute marginal.
    pair[6] += 1
    pair[7] -= 1
    groups = counts['cells'][pair[1]][2]
    next(g for g in groups if g[0] and g[2] > g[3])[3] += 1
    mutated = a2_return.reproduce(compact)
    assert mutated['contrasts']['R']['primary']['mean'] != before['contrasts']['R']['primary']['mean']
    # Repack the edited count payload with the same collect identity. Do not
    # recompute a checksum: it remains an advisory transcription aid.
    import base64, gzip, re
    encoded = base64.b64encode(gzip.compress(canonical(compact), mtime=0)).decode()
    flat = re.sub(r'\s+', '', essential[0])
    assert len(essential) == 1
    edited = re.sub(r'T\|A2\|packed\|.+?(?=SHA256)',
                    lambda _: 'T|A2|packed|'+canonical([0, 1, encoded]).decode(), flat)
    assert a2_return.import_parts([edited])['contrasts'] == mutated['contrasts']
    with pytest.raises(CorrectnessStop, match='ESSENTIAL/FULL disagreement'):
        a2_return.import_parts([edited]+split_texts(paths, 'FULL'))
    import io, tarfile
    archive = tmp_path/'full.tar.gz'
    raw = canonical(payload)
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo('a2-evidence.json'); info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    assert without_transcription(a2_return.import_parts(essential, archive)) == a2_return.read_archive(archive)
    with pytest.raises(CorrectnessStop, match='ESSENTIAL/FULL disagreement'):
        a2_return.import_parts([edited], archive)


def test_acquired_then_lost_is_temporal_and_counted(tmp_path):
    payload = count_fixture('J')
    # A real 0 -> 1 -> 0 -> 1 path, not derivable from arrival/final differences.
    for c in payload['cells']:
        if c['key'][1:3] == ['J-current', 101] and c['key'][4:] == [MONTHS[0], 'on']:
            c['rows'][0]['correct'] = int(c['key'][0] == 'post' and c['key'][3] in (2, 6))
    compact = a2_return.compact_counts(payload)
    reading = a2_return.reproduce(compact)
    expected = a2_return.reproduce(payload)
    assert reading == expected
    record = next(r for r in reading['retention_transitions'] if r.get('acquired_then_lost') and
                  r['model'] == 'J-current' and r['seed'] == 101 and r['month'] == MONTHS[0])
    assert record['acquired_then_lost'] >= 1
    path = next(p for p in compact['keyed_counts']['paths'] if p[1] > 0)
    path[1] -= 1
    assert a2_return.reproduce(compact)['retention_transitions'] != expected['retention_transitions']


def test_count_parts_missing_mixed_and_superseded(tmp_path):
    payload = count_fixture('RP', real_scale=True)
    first = a2_return.write_parts(tmp_path, payload, '1'*40)
    old = {p.name: p.read_text() for p in first}
    second = a2_return.write_parts(tmp_path, payload, '1'*40)
    assert {p.name for p in (tmp_path/'superseded').rglob('*.md')} == set(old)
    essential = split_texts(second, 'ESSENTIAL')
    with pytest.raises(CorrectnessStop, match='INCOMPLETE'):
        a2_return.import_parts(essential[:-1])
    # The same arm, commit and scientific values are still separate collects.
    with pytest.raises(CorrectnessStop, match='mixed A2 collect'):
        a2_return.import_parts(essential+[v for k, v in old.items() if '-FULL-' in k][0:1])
    with pytest.raises(CorrectnessStop):
        a2_return.import_parts([p.read_text() for p in tmp_path.rglob('RETURN-TEXT-*.md')])
    assert a2_return.import_parts(essential)['status'] == 'COMPLETE'
    # Both complete sets still cross-check after a refresh.
    assert a2_return.import_parts([p.read_text() for p in second])['status'] == 'COMPLETE'


def test_identical_pasted_parts_deduplicate_and_changed_duplicates_refuse(tmp_path):
    paths = a2_return.write_parts(tmp_path, count_fixture('RP', real_scale=True), '1'*40)
    texts = [p.read_text() for p in paths]
    expected = a2_return.import_parts(texts)
    assert a2_return.import_parts(texts+list(reversed(texts))) == expected
    # Even an advisory checksum edit makes this a non-identical duplicate.
    edited = texts[0].replace('SHA256 ', 'SHA256 0', 1)
    assert edited != texts[0]
    with pytest.raises(CorrectnessStop, match='duplicate/mixed part numbering'):
        a2_return.import_parts(texts+[edited])


@pytest.mark.parametrize('large', [False, True])
def test_collect_reports_essential_part_count_and_warns(tmp_path, capsys, large):
    payload = count_fixture('J')
    if large:
        payload['diagnostics'] = dict(opaque=''.join(digest(str(i).encode()) for i in range(2000)))
    paths = a2_return.write_parts(tmp_path, payload, '1'*40)
    n = sum('-ESSENTIAL-' in p.name for p in paths)
    printed = capsys.readouterr().out
    assert f'A2 J exploratory ESSENTIAL parts: {n}' in printed
    assert ('WARNING: ESSENTIAL exceeds 4 parts' in printed) == large
    assert (n > 4) == large


@pytest.mark.parametrize('panel', [*MONTHS[:1], *GENERAL])
def test_every_registered_unit_is_binary(panel):
    from kit import a4_score, scorers
    if panel == 'ifeval':
        member = dict(panel=panel, id='two-instructions', meta=dict(
            instruction_id_list=['length_constraints:number_words', 'detectable_format:number_bullet_lists'],
            kwargs=[dict(num_words=2, relation='at least'), dict(num_bullets=2)]))
        # Prompt success is all instructions, never the fraction satisfied.
        assert scorers.score(panel, 'two words', member)['correct'] is False
    elif panel in GENERAL:
        member = dict(panel=panel, id='unit', answer='A' if panel == 'knowledge' else '1', prompt='q')
    else:
        # SEC scoring is int(explicit-unit correctness); the whole pipeline is
        # covered by the wrapper tests above. Fractional records must refuse.
        member = None
    if member is not None:
        rows = a4_score.graded([member], [dict(id=member['id'], panel=panel, response='',
            input_tokens=0, output_tokens=0, finish_reason='stop')], {}, False)
        assert type(rows[0]['correct']) is int and rows[0]['correct'] in (0, 1)
    payload = count_fixture('R')
    cell = next(c for c in payload['cells'] if c['key'][4] == panel)
    cell['rows'][0]['correct'] = .5
    with pytest.raises(CorrectnessStop, match='non-binary'):
        a2_return.compact_counts(payload)


def test_legacy_v7e_packed_and_record_imports(tmp_path):
    import base64, gzip
    from kit.text_return import frame_text
    payload = count_fixture('P')
    packed = base64.b64encode(gzip.compress(canonical(payload), mtime=0)).decode()
    pieces = [packed[i:i+3000] for i in range(0, len(packed), 3000)]
    texts = [frame_text('a2-p', '1'*40, i+1, len(pieces), 'FULL', 'T|FULL-SET|1|'+'0'*64,
              ['T|A2|packed|'+canonical([i, len(pieces), v]).decode()], width=89)
             for i, v in enumerate(pieces)]
    assert without_transcription(a2_return.import_parts(texts)) == a2_return.reproduce(payload)
    # The older uncompressed record importer remains supported too.
    archived = dict(arm='P', registration_sha256=REGISTRATION, status='ARCHIVED UNREAD')
    text = frame_text('a2-p', '1'*40, 1, 1, 'FULL', 'T|FULL-SET|1|'+'0'*64,
                      ['T|A2|meta|'+canonical(archived).decode().replace(' ', chr(92)+'u0020')], width=89)
    assert a2_return.import_parts([text])['status'] == 'ARCHIVED UNREAD'


def test_compact_diagnostics_keep_statistics_and_full_keeps_bulk(tmp_path):
    payload = count_fixture('RP')
    payload['diagnostics'] = {'raw-ledger/m1': dict(response_characters=[3, 4, 3]),
        'stage/s1': dict(runtime=dict(loss_metrics=[dict(loss=2., tokens=3), dict(loss=4., tokens=5)])),
        'rewrite/example/first': dict(raw='raw text', rewritten='new text', accepted=True)}
    paths = a2_return.write_parts(tmp_path, payload, '1'*40)
    essential, full = (a2_return._import_parts(split_texts(paths, mode), payload_only=True)
                       for mode in ('ESSENTIAL', 'FULL'))
    summary = essential['diagnostics']
    assert summary['raw-ledger/m1']['response_character_histogram'] == {'3': 2, '4': 1}
    assert summary['stage/s1']['runtime']['loss_metric_statistics']['loss'] == dict(n=2, total=6., mean=3.)
    assert summary['rewrite/example/first']['detail'] == 'FULL'
    assert full['diagnostics'] == payload['diagnostics']
    assert a2_return.reproduce(essential) == a2_return.reproduce(full)
    assert a2_return.import_parts([p.read_text() for p in paths])['status'] == 'COMPLETE'


@pytest.mark.parametrize('change', ['missing_pair', 'missing_path', 'unused_pair', 'bad_table', 'bad_group'])
def test_invalid_or_missing_count_evidence_refuses(change):
    compact = a2_return.compact_counts(count_fixture('R'))
    counts = compact['keyed_counts']
    if change == 'missing_pair':
        counts['pairs'].pop()
    elif change == 'missing_path':
        counts['paths'].pop()
    elif change == 'unused_pair':
        counts['pairs'].append([0, 0, False, 12, 3, 0, 0, 9])
    elif change == 'bad_table':
        counts['pairs'][0][4] += 1
    else:
        counts['cells'][0][2][0][3] = 1000
    with pytest.raises(CorrectnessStop) as error:
        a2_return.reproduce(compact)
    assert not str(error.value).startswith('INCOMPLETE:')


def test_missing_essential_cli_reports_incomplete(tmp_path):
    import subprocess, sys
    payload = count_fixture('RP', real_scale=True)
    payload['diagnostics'] = dict(opaque=''.join(digest(str(i).encode()) for i in range(400)))
    paths = a2_return.write_parts(tmp_path/'parts', payload, '1'*40)
    essential = [p for p in paths if '-ESSENTIAL-' in p.name]
    assert len(essential) >= 2
    out = tmp_path/'reading.json'
    result = subprocess.run([sys.executable, '-I', str(KIT/'a2_r_read.py'), '--text',
                             *map(str, essential[:-1]), '--out', str(out)], capture_output=True, text=True)
    assert result.returncode == 75 and load(out)['status'] == 'INCOMPLETE'


def test_both_text_sets_publish_atomically(tmp_path, monkeypatch):
    payload = count_fixture('J')
    paths = a2_return.write_parts(tmp_path, payload, '1'*40)
    original = {p.name: p.read_bytes() for p in paths}
    replace = Path.replace
    failed = False
    def fail_one_publish(path, target):
        nonlocal failed
        if path.parent.name.startswith('.return-parts-') and '-FULL-' in path.name and not failed:
            failed = True
            raise OSError('synthetic publish failure')
        return replace(path, target)
    monkeypatch.setattr(Path, 'replace', fail_one_publish)
    with pytest.raises(OSError, match='synthetic publish'):
        a2_return.write_parts(tmp_path, payload, '1'*40)
    assert failed and {p.name: p.read_bytes() for p in tmp_path.glob('RETURN-TEXT-*.md')} == original
