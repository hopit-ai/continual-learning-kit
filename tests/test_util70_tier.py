"""CPU-only allocation, readiness and separation contracts for tier jobs."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from kit import combined_send as combined, m3_manifest as m, sp_manifest as sp, m3_train as train


@pytest.fixture(autouse=True)
def external_test_launch_seal(monkeypatch):
    from kit import v6_infra
    monkeypatch.setenv(v6_infra.ENV,v6_infra.authority_sha256())


def test_disjoint_gpu_jobs_share_host_and_training_conflicts(monkeypatch):
    monkeypatch.setattr(m.socket, 'gethostname', lambda: 'lease-fixture')
    with m.allocation_reservation(['GPU-A'], check_idle=False):
        with m.allocation_reservation(['GPU-B'], check_idle=False): pass
        with pytest.raises(m.Unbound, match='busy'):
            with m.allocation_reservation(['GPU-A'], check_idle=False): pass
        with pytest.raises(m.Unbound, match='busy'):
            with m.allocation_reservation(['GPU-' + c for c in 'ABCDEFGH'], check_idle=False): pass
    with m.allocation_reservation(['GPU-A'], check_idle=False): pass


def test_admission_checks_only_allocated_gpu_processes(monkeypatch):
    monkeypatch.setattr(m.socket, 'gethostname', lambda: 'idle-fixture')
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **kw: 'GPU-B, 123\n')
    with m.allocation_reservation(['GPU-A']): pass
    with pytest.raises(m.Unbound, match='active GPU'):
        with m.allocation_reservation(['GPU-B']): pass


def test_allocation_must_exactly_match_request(monkeypatch):
    from kit import m2_gpu
    monkeypatch.setattr(m2_gpu, 'devices', lambda: {'0':'GPU-A', '1':'GPU-B'})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    assert m.allocated_gpus(1) == ['GPU-B']
    with pytest.raises(m.IsolationError, match='exactly 2'): m.allocated_gpus(2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-B,GPU-A')
    assert m.allocated_gpus(2) == ['GPU-A', 'GPU-B']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    with pytest.raises(m.IsolationError, match='unavailable'): m.allocated_gpus(1)


def test_combined_resource_inventory_and_reference_lanes(tmp_path):
    from kit.combined_dryrun import fixtures
    doc = fixtures(tmp_path)
    for row in doc['tasks']:
        assert row['gpus'] in (0,1,2,8) and row['nodes'] == 1 and row['cpus'] > 0
        assert row['share_node'] == (row['gpus'] != 8)
    rows = {r['id']: r for r in doc['tasks']}
    assert rows['M3-broad']['gpus'] == 1 and rows['M3-math-targets']['gpus'] == 2
    for name in ('M3-math-resolve','M3-K-101-finqa-prepare','M3-K-101-finqa-export','REPORT-text','PF-tier-0'):
        assert rows[name]['gpus'] == 0
    assert rows['M3-K-101-finqa-train']['gpus'] == 8
    for seed in (101,102,103):
        ident=f'REF-m3-{seed}'
        assert rows[ident]['gpus']==1 and rows[ident]['requires']==[ident+'-prepare']
        assert rows[ident+'-prepare']['gpus']==0 and len(rows[ident+'-prepare']['commands'])==2
        assert rows[f'PD-{seed}-pre']['gpus']==1 and rows[f'PD-{seed}-train']['gpus']==8
        assert rows[f'PD-{seed}-post']['gpus']==1 and rows[f'PD-{seed}-finalize']['gpus']==0
    assert rows['PD-read']['gpus']==0 and len(rows['PD-read']['requires'])==3
    assert '| gpus | nodes | cpus | share node |' in combined.inventory(doc)


def test_sp_freshness_checks_detect_mutation_and_expiry(tmp_path):
    model = tmp_path / 'model'; model.mkdir(); file = model / 'weights'; file.write_bytes(b'original')
    receipt = tmp_path / 'receipt.json'; receipt.write_text('{}')
    local = dict(model=str(model), receipt=str(receipt))
    entry = dict(model_file_hashes={'weights':'hash'}, durable_provenance=dict(expires_at=None))
    entry['prepared_metadata'] = sp.prepared_metadata(local, entry)
    assert sp.fresh_binding(local, entry) == entry['durable_provenance']
    file.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='prepared export changed'): sp.fresh_binding(local, entry)
    entry['durable_provenance']['expires_at'] = 0
    with pytest.raises(sp.MissingInput, match='expired'): sp.fresh_binding(local, entry)


def test_training_ends_before_cpu_exports(tmp_path, monkeypatch):
    run = tmp_path / 'run'; run.mkdir()
    prepared = dict(observed=dict(geometry={}, arm='K'), base={}, incoming={}, scheduled_tokens=17)
    (run / 'prepared.json').write_text(json.dumps(prepared))
    monkeypatch.setattr(train, 'prepared_stage', lambda a: (run, tmp_path / 'data', prepared))
    monkeypatch.setattr(train, 'run_root', lambda *a: run)
    a = NS(out=tmp_path, arm='K', seed=101, stage='finqa', standin=True)
    train.train_job(a)
    assert (run / 'native-complete.json').exists()
    assert not list(run.glob('export-*.json')) and not (run / 'complete.json').exists()
    train.export_job(a)
    assert (run / 'complete.json').exists() and len(list(run.glob('export-*.json'))) == 4
    assert json.loads((run / 'summary.json').read_text())['scheduled_tokens'] == 17


def test_native_cpu_proof_replaces_only_identity_hashing(tmp_path, monkeypatch):
    from kit import m3_score as score, p3b_common, p3b_score
    from kit.p3b_common import atomic, files_manifest
    observed = dict(model_file_hashes={'weights':'checked-hash'})
    base = tmp_path / 'base'; base.mkdir(); (base / 'weights').write_text('base')
    atomic(tmp_path / 'inputs/base.json', dict(model=str(base), identity=observed))
    run = train.run_root(tmp_path, 'K', 101, 'finqa')
    model = run / 'hf-step40'; model.mkdir(parents=True)
    weights = model / 'weights'; weights.write_text('original')
    receipt = run / 'export-40.json'; atomic(receipt, observed)
    atomic(run / 'complete.json', {})
    atomic(run / 'manifest.json', files_manifest(run, ['export-40.json', 'complete.json']))
    source = dict(model=str(model), receipt=str(receipt), identity=observed, base=False)
    monkeypatch.setattr(p3b_common, 'identity', lambda *a: observed)
    ready = tmp_path / 'ready.local.json'
    checkpoints = [['base',101,'finqa',0], ['K',101,'finqa',0], ['K',101,'finqa',40]]
    score.prepare_scoring(NS(out=tmp_path, checkpoints_json=json.dumps(checkpoints), ready=ready, standin=False))
    assert all(v['status']=='PRESENT' and v['required'] for v in json.loads(ready.read_text())['exports'].values())
    monkeypatch.setenv('M3_SCORE_READY', str(ready))
    monkeypatch.setattr(p3b_common, 'identity', lambda *a: pytest.fail('GPU lane rehashed model'))
    assert score.checked(tmp_path, 'K',101,'finqa',40,False) == (source, observed)
    original = p3b_score.identity
    with score.prepared_identity(source, observed, False):
        assert p3b_score.identity(model, receipt=receipt) == observed
        with pytest.raises(ValueError, match='model differs'): p3b_score.identity(tmp_path / 'other', receipt=receipt)
    assert p3b_score.identity is original
    weights.write_text('mutated!')
    with pytest.raises(ValueError, match='checkpoint changed'): score.checked(tmp_path, 'K',101,'finqa',40,False)


def test_sp_missing_last_entry_still_scores_ready_entry(tmp_path, monkeypatch):
    state = tmp_path / 'state'; state.mkdir()
    proof = dict(verified=True)
    (tmp_path / 'pool-verification.json').write_text(json.dumps(proof))
    (state / 'SP-prepare-lane.json').write_text(json.dumps(dict(status='DONE', pool_verification_sha256=sp.digest(proof))))
    task = dict(id='SP-panel-lane', uses_gpu=True, after=['SP-prepare-lane'], commands=[['score','ready'],['score','missing']])
    def substitute(value, values, maps):
        if value == 'missing': raise sp.MissingInput('last checkpoint missing')
        return value
    monkeypatch.setattr(sp, 'substitute', substitute)
    calls = []
    assert sp.run(dict(tasks=[task]), task['id'], {'{sp}':tmp_path}, {}, runner=lambda *a,**kw:calls.append(a) or NS(returncode=0)) == 0
    assert calls == [(['score','ready'],)]
    assert json.loads((state / (task['id'] + '.json')).read_text())['status'] == 'MISSING-INPUT'
