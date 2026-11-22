"""Owner rule fixtures; synthetic contents only, with real indexed audits."""
from copy import deepcopy
import hashlib
import json

import pytest

from kit import trace_repair as repair
from kit.trace_common import SUBSET, sha
from kit.nm_common import Invalid


def row(task, identity, prompt=None, answer='answer'):
    return dict(task=task, id=identity, group=task+'/'+identity,
                prompt=prompt or hashlib.sha256((task+identity).encode()).hexdigest(), answer=answer)


def config(tmp_path, size=501):
    def ref(name, rows):
        path = tmp_path/name
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows))
        return dict(path=str(path), sha256=sha(path))
    datasets = {}
    for task in SUBSET:
        datasets[task] = {split: ref(task+'-'+split+'.jsonl',
            [row(task, split+str(i)) for i in range(size if split == 'train' else 1)])
            for split in ('train', 'eval', 'test')}
    return dict(studies=['gold', '27b'], datasets=datasets, exclusions=ref('exclusions.jsonl', []))


def replace(cfg, task, split, index, change):
    ref = cfg['datasets'][task][split]
    rows = [json.loads(line) for line in open(ref['path'])]
    rows[index].update(change)
    from pathlib import Path
    Path(ref['path']).write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows))
    ref['sha256'] = sha(ref['path'])
    return rows[index]


@pytest.mark.parametrize('task,reasons,is_trace,left,right,expected', [
    ('NumGLUE-ds', ['source_group', 'prompt'], True, '3 apples', '7 apples', 'R1'),
    ('NumGLUE-ds', ['prompt'], False, '3 apples', '7 apples', 'R2'),
    ('FOMC', ['prompt'], True, '3 apples', '7 apples', 'R3'),
    ('MeetingBank', ['answer'], True, '3 apples', '7 apples', 'R3'),
    ('NumGLUE-ds', ['prompt'], True, '  Ａ １２.５ B 2', 'a 12.5 b 2', 'R4'),
    ('NumGLUE-cm', ['prompt'], True, 'Buy 12.5 red apples then 2', 'Buy 12.5 green apples then 2', 'R4'),
    ('NumGLUE-ds', ['prompt'], True, 'Buy 2 then 3 apples', 'Buy 3 then 2 apples', None),
    ('NumGLUE-ds', ['prompt'], True, 'Buy some red apples', 'Buy some green apples', 'R4'),
])
def test_rules_and_audit_normalisation(task, reasons, is_trace, left, right, expected):
    assert repair.pair_rule(row(task, 't', left), row(task, 'h', right), reasons, is_trace) == expected


def test_r5_exact_instruction_only():
    empty = 'Write a summary of the following meeting transcripts.\nMeeting transcripts:\n\nSummary:'
    assert repair.empty_transcript(row('MeetingBank', 'train', empty))
    assert not repair.empty_transcript(row('MeetingBank', 'train', empty.replace('\n\n', '\nwords\n')))
    assert not repair.empty_transcript(row('MeetingBank', 'train', 'Other instruction. Meeting transcripts: Summary:'))


def test_repair_preserves_ids_order_eval_and_hash_bound_review(tmp_path):
    cfg = config(tmp_path)
    # One source-group removal, and independently the empty transcript.
    replace(cfg, 'FOMC', 'train', 3, dict(group='FOMC/eval0'))
    replace(cfg, 'MeetingBank', 'train', 6, dict(prompt='Write a summary of the following meeting transcripts.\nMeeting transcripts:\n\nSummary:'))
    # A genuine near prompt with different quantities must survive with review.
    prompt = 'A person has 123 red apples in a large basket and buys some more apples today.'
    replace(cfg, 'NumGLUE-ds', 'train', 7, dict(prompt=prompt))
    replace(cfg, 'NumGLUE-ds', 'test', 0, dict(prompt=prompt.replace('123', '789')))
    # An exclusion can carry a TRACE task name; origin, not that name, is R2.
    external = 'A shop stocks 123 bananas in boxes and sells a few bananas each morning to visitors.'
    replace(cfg, 'NumGLUE-ds', 'train', 8, dict(prompt=external))
    from pathlib import Path
    exclusion_path = Path(cfg['exclusions']['path'])
    exclusion_path.write_text(json.dumps(row('FOMC', 'external', external.replace('123', '789')))+'\n')
    cfg['exclusions']['sha256'] = sha(exclusion_path)
    before = deepcopy(cfg)
    out = tmp_path/'repaired'
    datasets = {task: {split: repair.read_rows(ref) for split, ref in splits.items()}
                for task, splits in cfg['datasets'].items()}
    manual = tmp_path/'manual-review.json'
    manual.write_text(json.dumps(dict(pairs=repair.audit_rows(datasets, repair.read_rows(cfg['exclusions']))['flagged'])))
    result = repair.repair(cfg, out, manual)
    assert result['status'] == 'COMPLETE' and cfg == before
    manifest = result['manifest']
    assert manifest['drops_per_rule']['FOMC'] == {'R1': 1}
    assert manifest['drops_per_rule']['MeetingBank'] == {'R5': 1}
    assert manifest['drops_per_rule']['NumGLUE-ds'] == {'R2': 1}
    for task, missing in [('FOMC', 3), ('MeetingBank', 6), ('NumGLUE-ds', 8)]:
        rows = [json.loads(line) for line in (out/(task+'-train.jsonl')).read_text().splitlines()]
        assert [r['id'] for r in rows] == ['train'+str(i) for i in range(501) if i != missing]
    reviews = json.loads((out/'audit_review.json').read_text())
    assert len(reviews) == manifest['remaining_cleared_pair_count'] > 0
    assert all(r['disposition'] == 'cleared' and r['evidence_sha256'] == sha(repair.RULE) for r in reviews)
    assert manifest['unresolved'] == 0
    for task in SUBSET:
        for split in ('eval', 'test'):
            assert sha(cfg['datasets'][task][split]['path']) == cfg['datasets'][task][split]['sha256']


def test_manual_review_must_equal_recomputed_pairs_and_input_hashes_checked(tmp_path):
    cfg = config(tmp_path, 2)
    path = tmp_path/'manual-review.json'
    path.write_text(json.dumps(dict(pairs=[dict(id='invented')])))
    with pytest.raises(Invalid, match='manual-review pairs differ'):
        repair.repair(cfg, tmp_path/'out', path)
    cfg['exclusions']['sha256'] = '0'*64
    with pytest.raises(Invalid, match='input identity differs'):
        repair.repair(cfg, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_distinct_refusal_writes_no_final_files(tmp_path):
    cfg = config(tmp_path, 500)
    for task in SUBSET:
        first = row(task, 'train0')
        replace(cfg, task, 'train', 499, dict(prompt=first['prompt']))
    result = repair.repair(cfg, tmp_path/'out')
    assert result['status'] == 'PENDING'
    assert {p['task'] for p in result['problems']} == set(SUBSET)
    assert all(p['rule'] == '500 distinct' for p in result['problems'])
    assert all(c['after'] == dict(rows=500, distinct=499) for c in result['manifest']['counts'].values())
    assert not (tmp_path/'out').exists()


def test_self_check_refuses_uncleared_source_group(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    replace(cfg, 'FOMC', 'train', 0, dict(group='FOMC/test0'))
    # Simulate an adjudicator defect: self-check must independently catch it.
    monkeypatch.setattr(repair, 'pair_rule', lambda *a: None)
    result = repair.repair(cfg, tmp_path/'out')
    assert result['status'] == 'PENDING'
    assert result['manifest']['unresolved'] > 0
    assert result['problems'][0]['rule'] == 'R1-R4'
    assert not (tmp_path/'out').exists()


def test_identical_contents_cannot_get_inconsistent_verdicts(tmp_path):
    cfg = config(tmp_path, 2)
    first = replace(cfg, 'FOMC', 'train', 0, dict(group='FOMC/test0'))
    replace(cfg, 'FOMC', 'train', 1, dict(prompt=first['prompt']))
    with pytest.raises(Invalid, match='inconsistent verdicts'):
        repair.repair(cfg, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_s8_refused(tmp_path):
    cfg = config(tmp_path, 1)
    cfg['studies'] = ['s8', 'gold', '27b']
    with pytest.raises(Invalid, match='S8 is out of scope'):
        repair.repair(cfg, tmp_path/'out')


def test_cli_pending_and_no_final_artifacts(tmp_path):
    import subprocess
    import sys
    cfg = config(tmp_path, 1)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    result = subprocess.run([sys.executable, '-m', 'kit.trace_repair', '--config', str(path),
                             '--out', str(tmp_path/'out')], capture_output=True, text=True)
    assert result.returncode == 3, result.stderr
    assert json.loads(result.stdout)['status'] == 'PENDING'
    assert not (tmp_path/'out').exists()


def test_rule_resource_checksum():
    sums = (repair.RULE.parent/'SHA256SUMS').read_text().split()
    assert sums == [sha(repair.RULE), repair.RULE.name]


def test_cli_complete_emits_exact_hash_bindings(tmp_path):
    import subprocess
    import sys
    cfg = config(tmp_path, 500)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    out = tmp_path/'out'
    result = subprocess.run([sys.executable, '-m', 'kit.trace_repair', '--config', str(path),
                             '--out', str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    edits = json.loads(result.stdout)['config_edits']
    for task in SUBSET:
        ref = edits['datasets'][task]['train']
        assert ref['sha256'] == sha(ref['path'])
    ref = edits['audit_review']
    assert ref['sha256'] == sha(ref['path'])
    ref = edits['repair_manifest']
    assert ref['sha256'] == sha(ref['path'])
    assert 'repair_manifest' not in json.loads(open(ref['path']).read())['outputs']
    assert edits['out'] == 'CHOOSE_A_FRESH_PREPARATION_STAGE_DIRECTORY'


@pytest.fixture
def repaired_preparation(tmp_path, monkeypatch):
    """Real bound files/audit/prepare; only model loading and token lengths stubbed."""
    import sys
    from types import SimpleNamespace
    from kit import trace_prepare as prep
    from kit.score_forgetting import load_panel
    cfg = config(tmp_path)
    prompt = 'A person has 123 red apples in a large basket and buys some more apples today.'
    replace(cfg, 'NumGLUE-ds', 'train', 7, dict(prompt=prompt))
    replace(cfg, 'NumGLUE-ds', 'test', 0, dict(prompt=prompt.replace('123', '789')))
    # Ensure binding the unrepaired file really reintroduces an R3 removal.
    replace(cfg, 'FOMC', 'train', 3, dict(group='FOMC/eval0'))
    old_train = deepcopy(cfg['datasets']['FOMC']['train'])
    result = repair.repair(cfg, tmp_path/'repaired')
    assert result['status'] == 'COMPLETE'
    edits = result['config_edits']
    for task in SUBSET:
        cfg['datasets'][task]['train'] = edits['datasets'][task]['train']
    cfg.update({key: edits[key] for key in ('audit_review', 'repair_manifest')})
    def bound(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=sha(path))
    rights = {k: dict(permitted=True, evidence_sha256='a'*64)
              for k in (*SUBSET, 'original_tokenizer', 'base', 'teacher', 'judge', 'panels', 'calibration')}
    general, _ = load_panel()
    cfg.update(out=str(tmp_path/'prepared'), base=dict(revision=prep.BASE, model='synthetic'),
        judge=prep.JUDGE, teacher=dict(revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'),
        original_tokenizer={}, licences=bound('rights.json', rights),
        sec_clearance=bound('clearance.json', dict(reserved_outcomes_opened=False,
            intersections=0, near_match_clearance=True)),
        panels=bound('panels.json', {'general-v1': general, 'open': [dict(panel=panel, id=str(i))
            for panel, count in [('Self-Instruct', 175), ('LIMA', 300), ('CoNa', 178)] for i in range(count)]}),
        calibration=bound('calibration.jsonl', []))
    # Calibration is JSONL, unlike the other bound JSON artifacts.
    from pathlib import Path
    path = Path(cfg['calibration']['path'])
    path.write_text(''.join(json.dumps(dict(panel=p, id=p+str(i), checkpoint_independent=True))+'\n'
                           for p in ('Self-Instruct', 'LIMA', 'CoNa') for i in range(30)))
    cfg['calibration']['sha256'] = sha(path)
    monkeypatch.setattr(prep, 'model_record', lambda ref: ref)
    monkeypatch.setattr(prep, 'census', lambda rows, tok: {
        r['id']: dict(student=1, gold_teacher=1, gold_response=1) for r in rows})
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **kw: object())))
    return cfg, old_train


def test_fresh_repair_prepares_with_manifest(repaired_preparation):
    from kit.trace_prepare import prepare
    cfg, _ = repaired_preparation
    result = prepare(cfg)
    assert result['audit']['unresolved'] == 0
    assert result['active_studies'] == ['gold', '27b']
    assert all(len(rows) == 500 for rows in result['selected'].values())


@pytest.mark.parametrize('change', ['equal_prompt', 'equal_quantities'])
def test_stale_rule_review_rechecks_current_content(repaired_preparation, change):
    from kit.trace_prepare import prepare
    cfg, _ = repaired_preparation
    prompt = repair.read_rows(cfg['datasets']['NumGLUE-ds']['train'])[7]['prompt']
    if change == 'equal_quantities':
        prompt = prompt.replace('red', 'tan')
    replace(cfg, 'NumGLUE-ds', 'test', 0, dict(prompt=prompt))
    # Even a rebound manifest cannot make a stale R4 verdict valid.
    from pathlib import Path
    path = Path(cfg['repair_manifest']['path'])
    manifest = json.loads(path.read_text())
    manifest['input_sha256']['datasets']['NumGLUE-ds']['test'] = cfg['datasets']['NumGLUE-ds']['test']['sha256']
    path.write_text(json.dumps(manifest))
    cfg['repair_manifest']['sha256'] = sha(path)
    result = prepare(cfg)
    assert result['status'] == 'PENDING'
    assert any('current pair is not R4 cleared' in p for p in result['problems'])
    assert not (Path(cfg['out'])/'prepared.json').exists()


@pytest.mark.parametrize('change,message', [
    ('missing_manifest', 'repair_manifest required'),
    ('manifest_hash', 'repair_manifest sha256'),
    ('old_train', 'FOMC/train output sha256'),
    ('eval', 'FOMC/eval input sha256'),
    ('test', 'FOMC/test input sha256'),
    ('exclusions', 'exclusions input sha256'),
    ('rule', 'rule_sha256'),
    ('review_manifest', 'audit_review manifest sha256'),
    ('review_hash', 'audit_review sha256'),
    ('unbound_train_change', 'FOMC/train sha256'),
    ('different_quantities', 'NumGLUE-ds/test input sha256'),
])
def test_rule_review_requires_current_manifest_bindings(repaired_preparation, change, message):
    from pathlib import Path
    from kit.trace_prepare import prepare
    cfg, old_train = repaired_preparation
    if change == 'missing_manifest':
        del cfg['repair_manifest']
    elif change == 'manifest_hash':
        cfg['repair_manifest']['sha256'] = '0'*64
    elif change == 'old_train':
        cfg['datasets']['FOMC']['train'] = old_train
    elif change == 'unbound_train_change':
        path = Path(cfg['datasets']['FOMC']['train']['path'])
        path.write_text(path.read_text()+'\n')
    elif change == 'different_quantities':
        prompt = repair.read_rows(cfg['datasets']['NumGLUE-ds']['test'])[0]['prompt']
        replace(cfg, 'NumGLUE-ds', 'test', 0, dict(prompt=prompt.replace('789', '456')))
    elif change in ('eval', 'test'):
        replace(cfg, 'FOMC', change, 0, dict(answer='changed heldout answer'))
    elif change == 'exclusions':
        path = Path(cfg['exclusions']['path'])
        path.write_text('\n')
        cfg['exclusions']['sha256'] = sha(path)
    elif change == 'review_hash':
        cfg['audit_review']['sha256'] = '0'*64
    else:
        path = Path(cfg['repair_manifest']['path'])
        manifest = json.loads(path.read_text())
        if change == 'rule':
            manifest['rule_sha256'] = '0'*64
        else:
            manifest['outputs']['audit_review']['sha256'] = '0'*64
        path.write_text(json.dumps(manifest))
        cfg['repair_manifest']['sha256'] = sha(path)
    result = prepare(cfg)
    assert result['status'] == 'PENDING'
    assert any(message in p for p in result['problems']), result
    assert not (Path(cfg['out'])/'prepared.json').exists()


def test_manual_review_still_prepares_without_manifest(repaired_preparation):
    from pathlib import Path
    from kit.trace_prepare import prepare
    cfg, _ = repaired_preparation
    path = Path(cfg['audit_review']['path'])
    reviews = json.loads(path.read_text())
    for review in reviews:
        review['evidence_sha256'] = 'a'*64
    path.write_text(json.dumps(reviews))
    cfg['audit_review']['sha256'] = sha(path)
    del cfg['repair_manifest']
    # Manual adjudication retains its prior semantics even for equal prompts.
    prompt = repair.read_rows(cfg['datasets']['NumGLUE-ds']['train'])[7]['prompt']
    replace(cfg, 'NumGLUE-ds', 'test', 0, dict(prompt=prompt))
    result = prepare(cfg)
    assert result['audit']['unresolved'] == 0
    assert result['active_studies'] == ['gold', '27b']
    # Preserve the manual-review hash mismatch exception as well.
    cfg['audit_review']['sha256'] = '0'*64
    with pytest.raises(Invalid, match='input identity differs'):
        prepare(cfg)


@pytest.mark.parametrize('change,message', [
    ('swapped_manifest', 'audit_review is not an array of review entries'),
    ('missing_path', 'audit_review is missing or unreadable'),
    ('unreadable_json', 'audit_review is missing or unreadable'),
    ('stale_hash_object', 'audit_review is not an array of review entries'),
])
def test_repair_backed_review_problems_are_named_pending(repaired_preparation, change, message):
    """Astra r2: a repair-backed config must never crash on a bad audit_review binding."""
    from pathlib import Path
    from kit.trace_prepare import prepare
    cfg, _ = repaired_preparation
    if change == 'swapped_manifest':
        cfg['audit_review'] = deepcopy(cfg['repair_manifest'])
    elif change == 'missing_path':
        cfg['audit_review'] = dict(path=str(Path(cfg['out']).parent/'absent-review.json'), sha256='0'*64)
    elif change == 'unreadable_json':
        path = Path(cfg['audit_review']['path'])
        path.write_text('{not json')
        cfg['audit_review']['sha256'] = sha(path)
    else:
        # Wrong config hash and a JSON object rather than the review array.
        path = Path(cfg['audit_review']['path'])
        path.write_text(json.dumps({'not': 'a review array'}))
    result = prepare(cfg)
    assert result['status'] == 'PENDING'
    assert any(message in p for p in result['problems']), result
    assert not (Path(cfg['out'])/'prepared.json').exists()


@pytest.mark.parametrize('change,message', [
    ('non_rule_rebound', 'is not a cleared rule review'),
    ('empty_entry', 'entry 0 lacks a valid id, disposition or evidence_sha256'),
    ('list_id', 'entry 0 lacks a valid id, disposition or evidence_sha256'),
    ('rule_entry_without_id', 'entry 0 lacks a valid id, disposition or evidence_sha256'),
    ('absent_reference', 'audit_review reference is missing or malformed'),
    ('null_reference', 'audit_review reference is missing or malformed'),
    ('empty_reference', 'audit_review reference is missing or malformed'),
    ('duplicate_ids', 'audit_review contains duplicate review ids'),
    ('non_finite_extra', 'audit_review is not canonically serialisable JSON'),
    ('deeply_nested_extra', 'audit_review is'),
])
def test_repair_backed_reviews_are_fully_validated(repaired_preparation, change, message):
    """Astra r3: with repair_manifest present, no review shape bypasses the binding checks or crashes."""
    from pathlib import Path
    from kit.trace_prepare import prepare
    cfg, _ = repaired_preparation
    path = Path(cfg['audit_review']['path'])
    reviews = json.loads(path.read_text())
    def rebind(value):
        path.write_text(json.dumps(value))
        cfg['audit_review']['sha256'] = sha(path)
    if change == 'non_rule_rebound':
        rebind([dict(r, evidence_sha256='a'*64) for r in reviews])
    elif change == 'empty_entry':
        rebind([{}])
    elif change == 'list_id':
        rebind([dict(reviews[0], id=[])])
    elif change == 'rule_entry_without_id':
        rebind([{k: v for k, v in reviews[0].items() if k != 'id'}])
    elif change == 'duplicate_ids':
        rebind(reviews + reviews[:1])
    elif change == 'non_finite_extra':
        # json.loads accepts 1e999 as inf; canonical hashing (allow_nan=False) cannot.
        path.write_text(json.dumps([dict(reviews[0], extra=0)] + reviews[1:]).replace('"extra": 0', '"extra": 1e999', 1))
        cfg['audit_review']['sha256'] = sha(path)
    elif change == 'deeply_nested_extra':
        # Astra r5: 10,000 nested arrays must give a named PENDING, never RecursionError.
        path.write_text(json.dumps([dict(reviews[0], extra=0)] + reviews[1:]).replace(
            '"extra": 0', '"extra": ' + '['*10000 + ']'*10000, 1))
        cfg['audit_review']['sha256'] = sha(path)
    elif change == 'absent_reference':
        del cfg['audit_review']
    elif change == 'null_reference':
        cfg['audit_review'] = None
    else:
        cfg['audit_review'] = {}
    if change in ('non_rule_rebound', 'empty_entry', 'list_id', 'rule_entry_without_id',
                  'duplicate_ids', 'non_finite_extra', 'deeply_nested_extra'):
        # Even with the manifest rebound to the swapped review, the review is refused.
        manifest_path = Path(cfg['repair_manifest']['path'])
        manifest = json.loads(manifest_path.read_text())
        manifest['outputs']['audit_review']['sha256'] = cfg['audit_review']['sha256']
        manifest_path.write_text(json.dumps(manifest))
        cfg['repair_manifest']['sha256'] = sha(manifest_path)
    result = prepare(cfg)
    assert result['status'] == 'PENDING'
    assert any(message in p for p in result['problems']), result
    assert not (Path(cfg['out'])/'prepared.json').exists()
    if change == 'deeply_nested_extra':
        # The shared conftest teardown parses every JSON file under tmp_path.
        path.write_text('[]')
