"""v5 reference guard job, reader isolation, and hardware checks; CPU fixtures only."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from kit import ref_guard as guard, ref_score_job as job
from kit.p3b_common import atomic, files_manifest


@pytest.fixture
def root(tmp_path):
    from kit.m2_assignment import freeze
    from kit.m3_train import bind_base
    out = tmp_path / 'm3'; out.mkdir()
    bind_base(NS(out=out, model=tmp_path / 'model', receipt=None, standin=True))
    freeze(out / 'm3-scoring-assignment.json', dict(schema='message2-scoring-assignment.v2', assignments={
        f'm3|{seed}': dict(node='CPU-STANDIN', uuid='CPU-STANDIN', environment='CPU-STANDIN') for seed in (101, 102, 103)}))
    job.assign(NS(out=out, family='m3', from_assignment=None))
    atomic(out / 'hellaswag/manifest.json', {})
    (out / 'hellaswag/panel.jsonl').write_text(''.join(json.dumps(dict(id=f'hs-{i}', gold='A')) + '\n' for i in range(100)))
    atomic(out / 'checkpoints.json', [dict(arm='base', seed=101, stage='finqa', step=0),
                                    dict(arm='K', seed=101, stage='finqa', step=0)])
    # Failed validations must be retained byte-for-byte.
    atomic(out / 'format-validation/m3-s101.json', dict(status='PENDING', failed=True))
    atomic(out / 'format-validation/knowledge-m3-s101.json', dict(status='FAIL'))
    return out


def score(out, panel='knowledge-ll'):
    job.run(NS(out=out, family='m3', checkpoints=out / 'checkpoints.json', seed=101, panel=panel, gpu='0', standin=True))


def test_job_both_panels_reader_layout_resume_and_failed_validation(root):
    from kit.m3_read import load_cells
    before = {p.name: p.read_bytes() for p in (root / 'format-validation').glob('*')}
    for panel in guard.PANELS: score(root, panel)
    files = guard.local_files(root)
    assert len(guard.records(files, True)) == 4
    cells, _ = load_cells(root, standin=True)
    for panel in guard.PANELS:
        for arm in ('base', 'K'):
            cell = cells[arm, 101, 'finqa', 0, panel]
            assert cell['method'] == guard.METHOD and cell['pending'] is None
            assert cell['uuid'] == cell['environment'] == 'CPU-STANDIN'
    score(root); score(root, 'hellaswag')
    assert guard.local_files(root) == files
    assert before == {p.name: p.read_bytes() for p in (root / 'format-validation').glob('*')}


def test_plain_script_cli_scores_checkpoint_list(root):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(Path(job.__file__)), '--action', 'score', '--out', str(root),
                             '--family', 'm3', '--checkpoints', str(root / 'checkpoints.json'), '--seed', '101',
                             '--panel', 'hellaswag', '--gpu', '0', '--standin'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(guard.records(guard.local_files(root), True)) == 2


def test_dispatch_calls_the_validation_references_without_production_math(root, monkeypatch):
    from kit import m3_score, m2_format, hellaswag_reference, m2_assignment
    calls = []
    def reference(model, rows, tokenizer):
        calls.append((model, tokenizer, len(rows)))
        return m3_score.standin_likelihoods(rows, 'reference')
    monkeypatch.setattr(m3_score, 'knowledge_reference', reference)
    monkeypatch.setattr(hellaswag_reference, 'score', reference)
    monkeypatch.setattr(m2_format, 'tokenizer_identity', lambda p: None)
    monkeypatch.setattr(m2_format, 'items', lambda out, standin: [dict(id=f'hs-{i}', label=0) for i in range(100)])
    monkeypatch.setattr(m3_score, 'knowledge_production', lambda *a: pytest.fail('production called'))
    monkeypatch.setattr(m2_format, 'likelihoods', lambda *a: pytest.fail('production called'))
    monkeypatch.setattr(job, 'source_for', lambda *a: (dict(model='native-export'), dict(model_file_hashes={'x': 'y'})))
    original = m2_assignment.active
    monkeypatch.setattr(m2_assignment, 'active', lambda out, gpu, standin: original(out, gpu, True))
    for panel in guard.PANELS:
        job.run(NS(out=root, family='m3', checkpoints=root / 'checkpoints.json', seed=101, panel=panel, gpu='0', standin=False))
    assert len(calls) == 4 and all(c[0] == 'native-export' and c[2] == 100 for c in calls)


def test_job_refuses_wrong_assignment_and_intermediate_steps(root):
    from kit.m2_assignment import freeze
    assignment = json.loads((root / guard.ASSIGNMENT).read_text())
    assignment['assignments']['m3.knowledge-ll|101']['uuid'] = 'GPU-wrong'
    (root / guard.ASSIGNMENT).unlink(); (root / (guard.ASSIGNMENT + '.sha256')).unlink()
    freeze(root / guard.ASSIGNMENT, assignment)
    with pytest.raises(Exception, match='off-assignment'): score(root)
    assignment['assignments']['m3.knowledge-ll|101']['uuid'] = 'CPU-STANDIN'
    (root / guard.ASSIGNMENT).unlink(); (root / (guard.ASSIGNMENT + '.sha256')).unlink()
    freeze(root / guard.ASSIGNMENT, assignment)
    atomic(root / 'checkpoints.json', [dict(arm='K', seed=101, stage='finqa', step=20)])
    with pytest.raises(Exception, match='decision checkpoints'): score(root)


def test_overlay_retains_generated_diagnostics_and_refuses_missing_peer(root):
    score(root)
    from kit.m3_score import knowledge_items
    rows = [dict(arm=arm, seed=101, step=0, panel='knowledge', question_id=r['id'], correct='0',
                 output_tokens='99', incorrect_format='1') for arm in ('base', 'K.finqa', 'C0.finqa') for r in knowledge_items()]
    over = guard.overlay(rows, guard.local_files(root), True)
    refs = [r for r in over if r.get('_guard_method') == guard.METHOD]
    assert len(refs) == 200 and all(r['output_tokens'] == '99' and r['incorrect_format'] == '1' for r in refs)
    assert guard.check(over)['reason'] == 'MIXED-SCORERS'
    assert guard.check(refs) == dict(status='COMPLETE', method=guard.METHOD)
    refs[0]['_guard_uuid'] = 'GPU-other'
    assert guard.check(refs)['status'] == 'UNVERIFIED-HARDWARE'


def test_hashes_coverage_and_synthetic_scope_are_checked(root):
    score(root)
    files = guard.local_files(root)
    with pytest.raises(ValueError, match='synthetic'): guard.records(files, False)
    key = next(n for n in files if n.endswith('/scores.csv'))
    files[key] += b'changed'
    with pytest.raises(ValueError, match='hash changed'): guard.records(files, True)
    score(root, 'hellaswag')
    files = guard.local_files(root)
    files['hellaswag/panel.jsonl'] = files['hellaswag/panel.jsonl'].replace(b'"gold": "A"', b'"gold": "B"')
    with pytest.raises(ValueError, match='foreign HellaSwag freeze'): guard.records(files, True)


def test_recorded_uuid_is_enforced_even_with_rehashed_identity(root):
    from kit.m3_read import load_cells
    score(root)
    target = next((root / 'reference-scores').glob('*/*/ref-identity.json'))
    meta = json.loads(target.read_text()); meta['assignment']['uuid'] = 'GPU-other'; atomic(target, meta)
    atomic(target.parent / 'manifest.json', files_manifest(target.parent, ['ref-identity.json', 'scores.csv', 'likelihoods.json', 'cost.json']))
    cells, _ = load_cells(root, standin=True)
    assert any(c.get('hardware_error') for c in cells.values())
    assert guard.check(guard.overlay([], guard.local_files(root), True))['status'] == 'UNVERIFIED-HARDWARE'


def test_method_and_gpu_checks_are_per_panel_seed_and_decision_constituents():
    def row(panel, seed, uuid, step=40, method=guard.METHOD):
        return dict(panel=panel, seed=seed, step=step, _guard_method=method, _guard_uuid=uuid, _guard_environment='env')
    rows = [row('knowledge', 101, 'GPU-A'), row('hellaswag', 101, 'GPU-B'), row('knowledge', 102, 'GPU-C')]
    assert guard.check(rows)['status'] == 'COMPLETE'
    assert guard.check(rows + [row('knowledge', 101, 'GPU-D')])['status'] == 'UNVERIFIED-HARDWARE'
    assert guard.check(rows + [row('knowledge', 101, 'GPU-D', 20, 'KIT-SCORED')])['status'] == 'COMPLETE'
    assert guard.check(rows + [row('knowledge', 101, 'GPU-A', method='KIT-SCORED')])['reason'] == 'MIXED-SCORERS'
    assert guard.check([row('knowledge', 101, 'GPU-A'), row('hellaswag', 101, 'GPU-B', method='KIT-SCORED')])['reason'] == 'MIXED-SCORERS'
    assert guard.check([dict(panel='knowledge', seed=101, step=40, _guard_validation='FAIL')])['status'] == 'PENDING'


def test_large_panels_decode_each_identity_once(monkeypatch):
    raw = json.dumps(dict(assignment=dict(uuid='GPU-A', environment='env', family='core'))).encode()
    files = {'scores/base-s101-k0-general-v1/m2-identity.json': raw}
    rows = [dict(arm='base', seed=101, step=0, panel='knowledge', question_id=str(i), correct='1') for i in range(100)]
    original = guard.json.loads; calls = []
    def decode(value, *args, **kwargs):
        if value == raw: calls.append(value)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(guard.json, 'loads', decode)
    result = guard.overlay(rows, files, True)
    assert len(calls) == 1 and all(r['_guard_uuid'] == 'GPU-A' for r in result)


def test_m3_contrasts_label_reference_mixed_and_hardware():
    from kit import m3_read as reader
    from kit.m3_schedule import ARMS, SEEDS, STAGES
    assignment = {f'm3|{s}': dict(uuid=f'GPU-{s}', environment='env') for s in SEEDS}
    cells = {}
    for arm in ('base', *ARMS):
        for seed in SEEDS:
            for stage in STAGES:
                for step in (0, 40):
                    for metric, panel in reader.PANEL_OF.items():
                        key = (arm, seed, STAGES[0] if arm == 'base' else stage, 0 if arm == 'base' else step, panel)
                        c = cells.setdefault(key, dict(metrics={}, items={}, uuid=f'GPU-{seed}', environment='env'))
                        c['metrics'][metric] = 80.; c['items'][metric] = {'q': 1}
                        if panel in guard.PANELS: c['method'] = guard.METHOD
    table = reader.Table(cells)
    value = reader.contrast(table, 'K-C0', [], [])
    assert value['method'] == guard.METHOD and value['label'] == 'INCONCLUSIVE'
    key = ('K', 101, STAGES[-1], 40, 'knowledge-ll')
    cells[key]['method'] = 'KIT-SCORED'
    assert reader.contrast(reader.Table(cells), 'K-C0', [], [])['reason'] == 'MIXED-SCORERS'
    cells[key]['method'] = guard.METHOD; cells[key]['uuid'] = 'GPU-other'
    assert reader.contrast(reader.Table(cells), 'K-C0', [], [])['label'] == 'UNVERIFIED-HARDWARE'
    for cell in cells.values():
        cell.update(method='KIT-SCORED',validation_sha256='passing-kit-validation',uuid='GPU-fixed')
    value=reader.contrast(reader.Table(cells), 'K-C0', [], [])
    assert value['label']=='PENDING' and value['reason']=='REFERENCE-SCORED required for decision checkpoint guards'


def test_m_reader_uses_reference_knowledge_without_letter_gate():
    from kit.m2_read import analyse
    from kit.m3_score import knowledge_items
    rows = [dict(arm=arm, seed=seed, step=step, cap=2048, panel='knowledge', question_id=r['id'], correct=1,
                 finished=1, output_tokens=30, incorrect_format=1, _guard_method=guard.METHOD,
                 _guard_uuid=f'GPU-{seed}', _guard_environment='env')
            for arm, step in (('base', 0), ('finqa-plain', 40)) for seed in (101, 102, 103) for r in knowledge_items()]
    doc = dict(technical_synthetic=True, hard_math='DOES_NOT_QUALIFY')
    result = analyse(rows, doc, repeats=8)
    bar = result['trajectories'][0]['retention']['knowledge_stage1']
    assert bar['status'] == 'PASS' and bar['method'] == guard.METHOD
    rows[0]['_guard_method'] = 'KIT-SCORED'
    assert analyse(rows, doc, repeats=8)['trajectories'][0]['retention']['knowledge_stage1']['reason'] == 'MIXED-SCORERS'
    for row in rows:row.update(_guard_method='KIT-SCORED',_guard_validation='PASS',incorrect_format=0)
    bar=analyse(rows,doc,repeats=8)['trajectories'][0]['retention']['knowledge_stage1']
    assert bar['status']=='PENDING' and bar['reason']=='REFERENCE-SCORED required for decision checkpoint guards'


def test_archive_reader_applies_reference_units(root, tmp_path):
    import tarfile
    from kit.p3b_read import load
    score(root)
    # A compact native return may have no original HellaSwag score when its validation failed.
    archive = tmp_path / 'return.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in guard.local_files(root): tar.add(root / name, arcname='return/' + name)
    rows, _ = load(archive)
    assert len(rows) == 200 and all(r['_guard_method'] == guard.METHOD for r in rows)


def test_x_collector_keeps_reference_assignment_and_units(root, tmp_path):
    from kit.x_collect import job as collect, load
    score(root)
    atomic(root / 'binding.json', dict(technical_synthetic=True))
    archive = tmp_path / 'x.tar.gz'
    collect(NS(out=root, archive=archive, standin=True))
    files = load(archive, True)
    assert guard.ASSIGNMENT in files and len(guard.records(files, True)) == 2


def test_legacy_validator_view_keeps_failed_validation_and_excludes_partial_guard(root):
    from kit.m2_assignment import validate_files
    score(root, 'hellaswag')
    files = guard.local_files(root)
    partial = 'scores/base-s101-k0-hellaswag/scores.csv'
    files[partial] = b'partial native scoring tail'
    failed = 'format-validation/m3-s101.json'
    files[failed] = (root / failed).read_bytes()
    view = guard.native_files(files)
    assert partial not in view and guard.ASSIGNMENT not in view
    assert not any(n.startswith('reference-scores/') for n in view)
    assert view[failed] == files[failed] and partial in files
    validate_files(view)


def test_native_model_authority_is_required():
    meta = dict(family='m3', arm='base', seed=101, step=0, stage='finqa', technical_synthetic=False,
                model=dict(model_file_hashes={'weights': 'wrong'}))
    files = {'inputs/base.json': json.dumps(dict(identity=dict(model_file_hashes={'weights': 'right'}))).encode()}
    with pytest.raises(ValueError, match='native export'): guard.validate_model(meta, files)


def test_x_generated_denominators_do_not_require_failed_likelihood_validation(root, monkeypatch):
    from kit import x_references, m2_assignment, m2_score, m2_format
    doc = dict(base=dict(model='base'), message2=str(root / 'core'), configuration='generic')
    monkeypatch.setattr(x_references, 'context', lambda *a: doc)
    monkeypatch.setattr(x_references.m, 'binding', lambda *a: dict(comparators={}))
    monkeypatch.setattr(m2_assignment, 'active', lambda *a: dict(family='x-tier1', seed=101, uuid='GPU-A', environment='env'))
    monkeypatch.setattr(m2_format, 'require_validation', lambda *a: pytest.fail('failed likelihood validation must not gate generated panels'))
    (root / 'scoring-assignment.json').write_bytes((root / 'm3-scoring-assignment.json').read_bytes())
    namespace = {}
    exec('def score(a):\n    _reference_test_calls.append(a.panel_set)\n', namespace)
    calls = []
    monkeypatch.setattr(m2_score, '_reference_test_calls', calls, raising=False)
    monkeypatch.setattr(m2_score, 'job', namespace['score'])
    x_references.job(NS(out=root, standin=False, gpu='0', seed=101, panel_root=None, generated_only=True))
    assert calls == list(x_references.m.PANEL_SETS) and 'hellaswag' not in calls


def test_text_return_preserves_public_reference_and_hardware_labels():
    from kit.text_return import essential_endpoints, label
    result = dict(contrasts={'K-C0': dict(method=guard.METHOD, guards={'finqa/final/knowledge': dict(values=[0, 0, 0], lower=-1)})})
    endpoints = essential_endpoints('KCR', result)
    assert next(r for r in endpoints if r['name'].endswith('knowledge'))['method'].endswith(guard.METHOD)
    assert label('UNVERIFIED-HARDWARE') == 'UNVERIFIED-HARDWARE'


def test_x_and_tier2_guard_values_gate_on_method_and_recorded_gpu(monkeypatch):
    from kit import x_read, m2_score, x_tier2_read, x_tier2
    def items(panel, *args):
        if panel == 'general-v1':
            return [dict(id=f'{p}-{i}', panel=p) for p in ('knowledge', 'math', 'ifeval') for i in range(100)]
        return [dict(id=f'{panel}-{i}', gold='entailment', group=str(i)) for i in range(3)]
    monkeypatch.setattr(m2_score, 'items_for', items)
    values = []
    for arm in ('x-pre', 'x-control', 'x-drift'):
        for seed in (101, 102, 103):
            for panel in ('test', 'finqa', 'knowledge', 'math', 'ifeval', 'math_hard', 'hellaswag'):
                population = ([dict(id=f'finqa-{i}') for i in range(1147)] if panel == 'finqa' else
                              [dict(id=f'{panel}-{i}') for i in range(100)] if panel in ('knowledge', 'math', 'ifeval', 'hellaswag') else items(panel))
                for step in ((0, 10, 20, 30, 40) if panel == 'test' else (0, 40)):
                    for item in population:
                        r = dict(arm=arm, seed=seed, step=step, panel=panel, question_id=item['id'], cap=2048,
                                 correct='1', tolerant_correct='1', gold=item.get('gold', ''), group=item.get('group', ''),
                                 output_tokens='10', finished='1', incorrect_format='1')
                        if panel in ('knowledge', 'hellaswag'):
                            r.update(_guard_method=guard.METHOD, _guard_uuid=f'GPU-{seed}-{panel}', _guard_environment='env')
                        values.append(r)
    inherited = [dict(r, arm='base') for r in values if r['arm'] == 'x-pre' and r['step'] == 0]
    doc = dict(technical_synthetic=True, domains=['ifeval', 'arc'], protected=['knowledge', 'math', 'ifeval'], configuration='generic')
    support = {k: dict(status='READY') for k in ('drift', 'plasticity')}
    result = x_read._tables(values, doc, support, inherited, repeats=8)
    assert result['comparisons']['drift']['method'] == guard.METHOD
    assert result['comparisons']['drift']['status'] == 'COMPLETE'
    changed = next(r for r in values if r['arm'] == 'x-drift' and r['panel'] == 'knowledge')
    changed['_guard_method'] = 'KIT-SCORED'
    assert x_read._tables(values, doc, support, inherited, repeats=8)['comparisons']['drift']['reason'] == 'MIXED-SCORERS'
    changed['_guard_method'] = guard.METHOD; changed['_guard_uuid'] = 'GPU-other'
    result = x_read._tables(values, doc, support, inherited, repeats=8)['comparisons']['drift']
    assert result['status'] == 'UNVERIFIED-HARDWARE' and x_read.admitted(result) is None
    changed['_guard_uuid'] = 'GPU-101-knowledge'
    name = next(name for name, spec in x_tier2.COMPARISONS.items() if spec['primary'] != 'gain_auc')
    mapped = [dict(r, arm='x-t2-'+name+'-'+{'x-pre': 'pre', 'x-control': 'control', 'x-drift': 'challenger'}[r['arm']]) for r in values]
    result = x_tier2_read.tables(mapped, doc, support, inherited, 8, True)['comparisons'][name]
    assert result['method'] == guard.METHOD
    assert result['status'] == 'COMPLETE'
    for row in [*values,*mapped,*inherited]:
        if row['panel'] in ('knowledge','hellaswag'):
            row.update(_guard_method='KIT-SCORED',_guard_validation='PASS',incorrect_format='0')
    first=x_read._tables(values,doc,support,inherited,repeats=8)['comparisons']['drift']
    second=x_tier2_read.tables(mapped,doc,support,inherited,8,True)['comparisons'][name]
    assert all(c['status']=='PENDING' and c['reason']=='REFERENCE-SCORED required for decision checkpoint guards' for c in (first,second))


def test_m_collector_archive_round_trip_through_native_reader(tmp_path):
    """Reference review 1: exercise M collection, manifests, overlay and native readback."""
    from kit import m2_collect, m2_read, m2_common, m2_assignment
    from kit.p3b_read import load
    out=tmp_path/'core';out.mkdir()
    atomic(out/'binding.json',dict(protocol=m2_common.protocol(),technical_synthetic=True,
        hard_math='DOES_NOT_QUALIFY',starts={'base':dict(model='synthetic-base'),
        'finqa-plain:101':dict(model='synthetic-finqa-101')}))
    m2_assignment.freeze(out/'scoring-assignment.json',dict(schema='message2-scoring-assignment.v2',assignments={
        'core|101':dict(node='CPU-STANDIN',uuid='CPU-STANDIN',environment='CPU-STANDIN')}))
    atomic(out/'checkpoints.local.json',[dict(arm='base',seed=101,step=0),dict(arm='finqa-plain',seed=101,step=40)])
    atomic(out/'hellaswag/manifest.json',{})
    (out/'hellaswag/panel.jsonl').write_text(''.join(json.dumps(dict(id=f'hs-{i}',gold='A'))+'\n' for i in range(100)))
    job.assign(NS(out=out,family='core',from_assignment=None))
    for panel in guard.PANELS:
        job.run(NS(out=out,family='core',checkpoints=out/'checkpoints.local.json',seed=101,panel=panel,gpu='0',standin=True))
    archive=tmp_path/'message2-return.tar.gz'
    m2_collect.job(NS(out=out,archive=archive,standin=True))
    rows,files=load(archive)
    assert len(guard.records(files,True))==4
    assert len(rows)==400 and all(r['_guard_method']==guard.METHOD for r in rows)
    units={name.rsplit('/',1)[0] for name in files if name.startswith('reference-scores/') and name.endswith('/manifest.json')}
    assert len(units)==4 and all(unit+'/scores.csv' in files for unit in units)
    m2_read.job(NS(out=tmp_path/'readback',source=archive,standin=True,bootstrap=8))
    reading=json.loads((tmp_path/'readback/reading.json').read_text())
    assert reading['trajectories'][0]['retention']['knowledge_stage1']['method']==guard.METHOD


@pytest.mark.parametrize('panel',['knowledge','hellaswag'])
@pytest.mark.parametrize('validation',['PASS','FAIL','PENDING'])
def test_kit_only_decision_guards_require_reference_even_after_validation(panel,validation):
    """Owner ruling 1 Nov, reference review 3: validation cannot authorize kit-only guards."""
    rows=[dict(panel=panel,seed=seed,step=step,_guard_method='KIT-SCORED',
               _guard_validation=validation,_guard_uuid=f'GPU-{seed}',_guard_environment='env')
          for seed in (101,102,103) for step in (0,40)]
    assert guard.check(rows)==dict(status='PENDING',reason='REFERENCE-SCORED required for decision checkpoint guards')
    assert guard.apply(dict(status='PASS'),rows)['status']=='PENDING'


def test_reference_method_labels_round_trip_without_checksum_changes(tmp_path):
    """Reference-labelled endpoint methods must survive whitespace-tolerant paste parsing."""
    from kit.text_return import write_parts,parse_parts
    stat=dict(mean=3.,sd=0.,lower=3.,upper=3.,values=[3.,3.,3.],n=3)
    reader=dict(decisions={'K':'PASS','CR':'PASS'},contrasts={name:dict(label='PASS',method=guard.METHOD,
        primary=stat,guards={'finqa/final/knowledge':stat}) for name in ('K-C0','CR-G')})
    paths=write_parts(tmp_path/'text',dict(readers={'KCR':[reader]},jobs=[]),'kit-message-v5','a'*40)
    parsed=parse_parts([p.read_text() for p in paths])
    assert set(parsed['part_checksums'].values())=={'MATCH'}
    methods=[e['method'] for e in parsed['endpoints'].values() if guard.METHOD in e['method']]
    assert methods and all(' ' not in method and method.endswith(guard.METHOD) for method in methods)
