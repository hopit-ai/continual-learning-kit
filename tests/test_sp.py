"""CPU tests for SP scoring, reading and sidecar manifests (m3b2). Fixtures only; no model or GPU.

The frozen panel, checkers and fixtures (kit/sp_prepare.py, kit/resources/sp/) belong to
m3sp-author and are tested there. These tests use the real pinned checker module read-only; only
the frozen-panel loader is replaced (no scientific freeze exists yet).
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kit import pd_contract  # noqa: E402
from kit import sp_manifest as man  # noqa: E402
from kit import sp_prepare as prep  # noqa: E402  (m3sp-author's; read-only use)
from kit import sp_read as rd  # noqa: E402
from kit import sp_score as sc  # noqa: E402

H = 'a' * 64
GPU = dict(uuid='GPU-1', environment='env')
CK = prep.load_checkers()


def panel_rows(n=4):
    rows = []
    for i in range(n):
        for wording in ('canonical', 'paraphrase'):
            rows.append(dict(id='r%d:%s' % (i, wording), request_id='r%d' % i, wording=wording, primary_family='required_word',
                             prompt='p%d %s' % (i, wording), constraints=[dict(family='required_word', args=dict(word='w%d' % i))]))
    return rows


def scored_rows(pattern):
    """pattern: {request_id: (canonical_strict, paraphrase_strict)}."""
    out = []
    for rid, values in pattern.items():
        for wording, v in zip(('canonical', 'paraphrase'), values):
            out.append(dict(id='%s:%s' % (rid, wording), request_id=rid, wording=wording, primary_family='required_word',
                            strict=v, strict_all_constraints_pass=v, instructions=[dict(family='required_word', passed=v)]))
    return out


# ---------------------------------------------------------------------------------------------
# Strict / instruction scoring with the pinned checker, template-reliance gap, intervals.

def test_strict_and_instruction_scoring_on_raw_text():
    panel = panel_rows()
    raw = [dict(id=r['id'], response=('W%s here' % r['request_id'][1:]) if r['wording'] == 'canonical' else 'nothing')
           for r in panel]
    scored = sc.score_panel(panel, raw, CK)
    assert all(r['strict'] == r['strict_all_constraints_pass'] for r in scored)
    summary = rd.summarise(scored)
    assert summary['wording']['canonical']['strict'] == 100.0 and summary['wording']['paraphrase']['strict'] == 0.0
    assert summary['strict'] == 50.0 and summary['instruction'] == 50.0
    assert summary['template_reliance_gap'] == summary['wording_gap'] == 100.0
    assert any('paraphrase strict pass rate' in w for w in summary['warnings'])     # monitored only
    assert summary['label'] == 'DESCRIPTIVE' and summary['json_fence_failures'] == dict(prompts=0, failures=0)
    with pytest.raises(ValueError):
        sc.score_panel(panel, raw[:-1], CK)
    with pytest.raises(ValueError):
        sc.score_panel(panel, raw + [dict(id='foreign', response='')], CK)


def test_json_fence_failures_are_separate_and_strict_uses_raw_json():
    item = dict(id='j:canonical', request_id='j', wording='canonical', primary_family='json_only',
                prompt='p', constraints=[dict(family='json_only', args={})])
    fenced = sc.score_prompt(item, '```json\n{"a": 1}\n```', CK)
    plain = sc.score_prompt(item, '{"a": 1}', CK)
    assert fenced['strict'] == 0 and fenced['json_fence_failure'] == 1
    assert plain['strict'] == 1 and plain['json_fence_failure'] == 0
    other = sc.score_prompt(panel_rows(1)[0], 'w0', CK)
    assert other['json_fence_failure'] is None


def test_template_reliance_change_and_bootstrap():
    incoming = scored_rows({'a': (1, 1), 'b': (1, 0), 'c': (0, 0), 'd': (1, 1)})
    descendant = scored_rows({'a': (1, 0), 'b': (1, 0), 'c': (1, 0), 'd': (1, 1)})
    assert rd.template_reliance_gap(incoming) == 25.0 and rd.wording_gap(descendant) == 75.0
    assert rd.gap_change(descendant, incoming) == 50.0
    first = rd.bootstrap_pairs(descendant, draws=2000)
    assert first == rd.bootstrap_pairs(descendant, draws=2000)          # fixed RNG
    assert first['lower'] <= first['estimate'] <= first['upper'] and first['unit'] == 'request pair'
    assert rd.bootstrap_pairs(descendant, 'canonical_strict', draws=500)['estimate'] == 100.0
    with pytest.raises(ValueError):
        rd.pairs([r for r in incoming if r['wording'] == 'canonical'])
    with pytest.raises(ValueError):
        rd.gap_change(descendant, scored_rows({'a': (1, 1)}))


def test_three_seed_interval_never_falls_back_to_two():
    full = rd.interval3({101: 1.0, 102: 2.0, 103: 3.0})
    assert full['status'] == 'COMPLETE' and full['mean'] == 2.0
    assert abs(full['upper'] - (2.0 + 4.302653 / 3 ** 0.5)) < 1e-9
    assert rd.interval3({101: 1.0, 102: 2.0})['status'] == 'INCOMPLETE'
    assert rd.interval3({101: 1.0, 102: 2.0, 103: None})['status'] == 'INCOMPLETE'
    assert rd.interval3({101: 1.0, 102: 2.0, 104: 3.0})['status'] == 'INCOMPLETE'


def test_paired_seeds_require_same_physical_gpu():
    a = {s: dict(summary=dict(strict=60.0), gpu_identity=GPU) for s in rd.SEEDS}
    b = {s: dict(summary=dict(strict=50.0), gpu_identity=GPU) for s in rd.SEEDS}
    metric = lambda x: x['summary']['strict']
    assert rd.paired_seeds(a, b, metric)['mean'] == 10.0
    b[102] = dict(summary=dict(strict=50.0), gpu_identity=dict(uuid='GPU-2', environment='env'))
    assert rd.paired_seeds(a, b, metric)['status'] == 'UNPAIRED-GPU'


# ---------------------------------------------------------------------------------------------
# Frozen correct cohorts and length metrics.

INCOMING = dict(checkpoint_id='incoming-101', weights_sha256='b' * 64)


def cohort(tmp_path, name, outcomes, **kw):
    rows = [dict(question_id=q, correct=v) for q, v in outcomes.items()]
    return sc.freeze_correct_cohort(tmp_path / name, family=kw.pop('family', 'f'), seed=101, panel='test',
                                    incoming=kw.pop('incoming', INCOMING), score_rows=rows, scores_sha256=H, **kw)


def test_freeze_correct_cohort_is_incoming_only_and_write_once(tmp_path):
    rows = [dict(question_id='q1', correct='1', tolerant_correct='1'), dict(question_id='q2', correct='0', tolerant_correct='1'),
            dict(question_id='q3', correct='0', tolerant_correct='0')]
    path = tmp_path / 'cohort.json'
    doc = sc.freeze_correct_cohort(path, family='f', seed=101, panel='test', incoming=INCOMING, score_rows=rows,
                                   scores_sha256=H, column='tolerant_correct')
    assert doc['question_ids'] == ['q1', 'q2'] and doc['population'] == 3 and doc['correct_column'] == 'tolerant_correct'
    again = sc.freeze_correct_cohort(path, family='f', seed=101, panel='test', incoming=INCOMING, score_rows=rows,
                                     scores_sha256=H, column='tolerant_correct')
    assert again == doc
    with pytest.raises(ValueError, match='already frozen'):
        sc.freeze_correct_cohort(path, family='f', seed=101, panel='test', incoming=INCOMING, score_rows=rows, scores_sha256=H)
    with pytest.raises(ValueError, match='incoming checkpoint'):
        sc.freeze_correct_cohort(tmp_path / 'x.json', family='f', seed=101, panel='test', incoming=INCOMING,
                                 score_rows=[dict(r, checkpoint_id='descendant-k40') for r in rows], scores_sha256=H)
    with pytest.raises(ValueError):
        sc.freeze_correct_cohort(tmp_path / 'y.json', family='f', seed=104, panel='test', incoming=INCOMING,
                                 score_rows=rows, scores_sha256=H)
    with pytest.raises(ValueError, match='no strict_correct grade'):
        sc.freeze_correct_cohort(tmp_path / 'z.json', family='f', seed=101, panel='test', incoming=INCOMING,
                                 score_rows=rows, scores_sha256=H, column='strict_correct')
    loaded = sc.load_cohort(path)
    loaded['question_ids'].append('q3')
    with pytest.raises(ValueError):
        pd_contract.verify_seal(loaded)


def raw_row(qid, n, finish='stop', eos=True):
    ids = (list(range(n - 1)) + ([151645] if eos and finish == 'stop' else [7])) if n else []
    return dict(id=qid, response='x', token_ids=ids, output_tokens=n, finish_reason=finish)


def test_length_metrics_count_all_tokens_and_truncation(tmp_path):
    c = cohort(tmp_path, 'c.json', {'q1': 1, 'q2': 1, 'q3': 0})
    raw = [raw_row('q1', 4), raw_row('q2', 6), raw_row('q3', 8, 'length')]
    # Descendant: q1 still right, q2 lost (still in C), q3 now right but outside C.
    m = sc.length_metrics(raw, {'q1': 1, 'q2': 0, 'q3': 1}, c, cap=8, eos_token_ids=[151645])
    full = m['full']
    assert (full['n'], full['n_correct'], full['total_tokens'], full['mean_tokens'], full['tokens_per_correct']) == (3, 2, 18, 6.0, 9.0)
    assert full['truncation_rate'] == pytest.approx(1 / 3) and full['status'] == 'COMPLETE' and full['missing'] == 0
    assert m['cohort']['n'] == 2 and m['cohort']['cohort_size'] == 2 and m['cohort']['n_correct'] == 1
    assert m['cohort']['tokens_per_correct'] == 10.0 and m['cohort']['accuracy'] == 50.0
    assert m['eos_terminated'] == 2 and m['convention']['source'] == 'token_ids' and m['status'] == 'COMPLETE'
    zero = sc.length_metrics(raw, {'q1': 0, 'q2': 0, 'q3': 0}, c, cap=8)
    assert zero['full']['tokens_per_correct'] is None and zero['full']['undefined'] == 'zero correct'
    assert zero['convention_sha256'] == m['convention_sha256']


def test_missing_raw_or_placeholder_is_missing_never_short(tmp_path):
    c = cohort(tmp_path, 'c.json', {'q1': 1, 'q2': 1, 'q3': 0})
    placeholder = dict(id='q2', response='', output_tokens=0, finish_reason='missing')    # c_score placeholder
    for raw in ([raw_row('q1', 4), raw_row('q3', 5)], [raw_row('q1', 4), placeholder, raw_row('q3', 5)]):
        m = sc.length_metrics(raw, {'q1': 1, 'q2': 0, 'q3': 1}, c)
        assert m['status'] == 'INCOMPLETE' and m['full']['missing'] == 1
        assert m['full']['mean_tokens'] is None and m['full']['tokens_per_correct'] is None
        assert m['full']['n'] == 3                                            # denominator never shrinks
        assert m['cohort']['status'] == 'INCOMPLETE' and m['cohort']['tokens_per_correct'] is None
    complete = sc.length_metrics([raw_row('q1', 4), raw_row('q2', 4), raw_row('q3', 5)], {'q1': 1, 'q2': 0, 'q3': 1}, c)
    assert complete['status'] == 'COMPLETE'


def test_length_metrics_refuse_inconsistent_inputs(tmp_path):
    good = [raw_row('q1', 4)]
    with pytest.raises(ValueError, match='disagrees'):
        sc.length_metrics([dict(raw_row('q1', 4), output_tokens=5)], {'q1': 1})
    with pytest.raises(ValueError, match='mixed'):
        sc.length_metrics(good + [dict(id='q2', response='x', output_tokens=3, finish_reason='stop')], {'q1': 1, 'q2': 1})
    with pytest.raises(ValueError, match='finish_reason'):
        sc.length_metrics([dict(raw_row('q1', 4), finish_reason='abort')], {'q1': 1})
    with pytest.raises(ValueError, match='exceed'):
        sc.length_metrics([raw_row('q1', 9)], {'q1': 1}, cap=8)
    with pytest.raises(ValueError, match='foreign'):
        sc.length_metrics(good, {'q2': 1})
    other = cohort(tmp_path, 'o.json', {'q9': 1})
    with pytest.raises(ValueError, match='foreign grades'):
        sc.length_metrics(good, {'q1': 1}, other)
    empty = cohort(tmp_path, 'e.json', {'q1': 0})
    assert sc.length_metrics(good, {'q1': 1}, empty)['cohort']['undefined'] == 'empty cohort'
    a = sc.length_metrics(good, {'q1': 1})
    b = sc.length_metrics([dict(id='q1', response='x', output_tokens=4, finish_reason='eos')], {'q1': 1})
    with pytest.raises(ValueError, match='conventions differ'):
        sc.same_convention(a, b)


def test_registered_correctness_columns():
    assert sc.correct_column('test') == sc.correct_column('panel') == 'tolerant_correct'
    assert sc.correct_column('finqa') == sc.correct_column('math_hard') == sc.correct_column('general-v1') == 'correct'
    assert sc.CORRECT_COLUMNS['sp'] == 'strict_all_constraints_pass'
    with pytest.raises(ValueError, match='N/A'):
        sc.correct_column('hellaswag')


def test_trajectory_refuses_pooled_stage_cohorts(tmp_path):
    c1 = cohort(tmp_path, '1.json', {'q1': 1}, family='stage1')
    c2 = cohort(tmp_path, '2.json', {'q1': 1}, family='stage2', incoming=dict(INCOMING, checkpoint_id='stage1-final'))
    raw = [raw_row('q1', 4)]
    p = lambda co, step: dict(checkpoint_id='c%d' % step, step=step, metrics=sc.length_metrics(raw, {'q1': 1}, co))
    assert len(rd.trajectory([p(c1, 0), p(c1, 40)])['points']) == 2
    with pytest.raises(ValueError, match='cannot be pooled'):
        rd.trajectory([p(c1, 0), p(c2, 40)])


# ---------------------------------------------------------------------------------------------
# Launch manifests and scorer layouts (read-only mirrors of the core kit).

def cmd(script, **kw):
    out = ['py', '{kit}/%s.py' % script, '--out', '{out}']
    for k, v in kw.items():
        out += ['--' + k.replace('_', '-'), str(v)]
    return out


def launch():
    rep = lambda **kw: cmd('m2_reprobe', action='score', **kw)
    task = lambda ident, family, seed, commands, **kw: dict(id=ident, kind='score-family', row=kw.pop('row', ident.split('-')[0]),
                                                           requires=['R0'], family=family, seed=seed, commands=commands, **kw)
    tasks = [
        dict(id='R0', kind='prepare', row='RE-PROBE', requires=[], commands=[cmd('m2_reprobe', action='prepare')]),
        dict(id='M-pools', kind='prepare', row='M3-pools', requires=['R0'], commands=[cmd('m2_prepare')]),
        task('R-incoming-101', 'reprobe', 101, [rep(arm='generic', seed=101, step=0, panel_set='test'),
                                                rep(arm='base', seed=101, step=0, panel_set='test')], row='RE-PROBE'),
        dict(id='R-generic-101', kind='train-stage', row='RE-PROBE', requires=['R-incoming-101'], commands=[cmd('m2_reprobe', action='train')]),
        task('R-generic-101-score', 'reprobe', 101, [rep(arm='generic', seed=101, step=s, panel_set='test') for s in (10, 40)], row='RE-PROBE'),
        task('M4-101', 'core', 101, [cmd('m2_run', action='boundaries', seed=101, mode='score')]),
        task('M7-101-incoming', 'core', 101, [cmd('m2_run', action='legal-generic', seed=101, mode='score', score_phase='incoming')]),
        task('M8-101-incoming', 'core', 101, [cmd('m2_run', action='legal-broad', seed=101, mode='score', score_phase='incoming')]),
        task('RA-A', 'return1.A', 101, [cmd('m2_rescore', action='score', package='A', gpu='{score_gpu}')], row='RETURN1-A'),
        task('X-incoming-101', 'x-tier1', 101, [cmd('x_score', arm='pre', seed=101, step=0, panel_set='test'),
                                                cmd('x_score', arm='control', seed=101, step=0, panel_set='test'),
                                                cmd('x_references', seed=101)]),
        task('X15-101-incoming', 'x-tier2.cap', 101, [cmd('x_score', arm='pre', seed=101, step=0, panel_set='test')]),
        task('S8-S10-101', 'screen-s', 101, [cmd('m2_score', arm='s', seed=101, step=0, panel_set='test')], row='S8-S10'),
        task('S8-S10-104', 'screen-s', 104, [cmd('s_score', action='boundaries', seed=104)], row='S8-S10'),
        task('A4-score', 'a4.chem', 101, [cmd('m2_score', arm='x', seed=101, step=0, panel_set='test')], row='A4'),
        task('FV-core-101', 'core', 101, [cmd('m2_format', seed=101)], row='FORMAT-VALIDATION'),
    ]
    return dict(schema='message2-launch.v2', nodes=[], tasks=tasks)


def m3_launch():
    m3 = lambda **kw: cmd('m3_score', seed=101, **kw)
    return dict(schema='message3-launch.v1', nodes=[], tasks=[
        dict(id='M3-base-101', kind='score-family', requires=[], family='m3', seed=101,
             commands=[m3(action='score', arm='base', stage='finqa', step=0, panel_set='finqa'),
                       m3(action='validate-hellaswag')]),
        dict(id='M3-K-101-finqa-train', kind='train-stage', requires=['M3-base-101'], arm='K', stage='finqa', seed=101,
             commands=[m3(action='alias', arm='K', stage='finqa', step=0, panel_set='finqa')]),
        dict(id='M3-K-101-finqa-score', kind='score-family', requires=['M3-K-101-finqa-train'], family='m3', seed=101,
             commands=[m3(action='score', arm='K', stage='finqa', step=40, panel_set='finqa')])])


def test_core_scorer_constants_still_match():
    c_common = pytest.importorskip('kit.c_common')
    m2_common = pytest.importorskip('kit.m2_common')
    assert tuple(c_common.STEPS) == man.M2_STEPS and tuple(c_common.RETENTION_STEPS) == man.M2_RETENTION_STEPS
    assert tuple(m2_common.PANEL_SETS) == man.M2_PANEL_SETS


def test_m2_run_expansion_mirrors_runner_rules():
    o = lambda **kw: man.options(cmd('m2_run', **kw))
    boundaries = man.expand_m2_run(o(action='boundaries', mode='score'))
    assert ('base', 0, 'hellaswag') in boundaries and ('finqa-plain', 40, 'general-v1') in boundaries
    assert ('legal-plain', 10, 'test') in boundaries and ('legal-plain', 10, 'finqa') not in boundaries
    assert ('standalone-legal-plain', 40, 'test') in boundaries
    assert man.expand_m2_run(o(action='boundaries', mode='train')) == []
    incoming = man.expand_m2_run(o(action='legal-generic', mode='score', score_phase='incoming'))
    assert {(a, s) for a, s, _ in incoming} == {('legal-generic', 0)}
    final = man.expand_m2_run(o(action='legal-generic', mode='score', score_phase='final'))
    assert {s for _, s, _ in final} == {10, 20, 30, 40} and ('legal-generic', 30, 'test') in final
    nxt = man.expand_m2_run(o(action='legal-next', mode='score', score_phase='incoming'))
    assert {a for a, _, _ in nxt} == {'legal-earlier', 'legal-both'}
    math = man.expand_m2_run(o(action='math-sibling', mode='score'))
    assert {a for a, _, _ in math} == {'math-sibling'} and len(math) == 20
    assert man.expand_x_p(man.options(cmd('x_run', action='p', mode='score')))[0][0] == 'x-p'
    assert man.expand_x_p(man.options(cmd('x_run', action='p', mode='train'))) == []


def test_planned_inventory_uses_written_arms_and_screens():
    planned, skipped = man.planned_from_launch(launch())
    arms = {(e['task'], e['arm']) for e in planned}
    assert ('R-incoming-101', 'legal-generic') in arms and ('R-incoming-101', 'base') in arms
    assert ('X-incoming-101', 'x-pre') in arms and ('X-incoming-101', 'x-control') in arms
    assert ('X15-101-incoming', 'x-t2-cap-pre') in arms
    assert {s['script'] for s in skipped} >= {'x_references', 's_score', 'm2_format'}
    assert not any(e['task'] == 'FV-core-101' for e in planned)
    kept, dropped = man.screen(planned)
    reasons = {d['planned_id']: d['reason'] for d in dropped}
    assert reasons['m2:S8-S10-101:0'] == 'Screen S is out of this tier' and reasons['m2:A4-score:0'] == '4A is separate'
    roles = {e['planned_id']: e['role'] for e in kept}
    assert roles['m2:R-incoming-101:0'] == 'incoming' and roles['m2:R-incoming-101:1'] == 'base'
    assert roles['m2:R-generic-101-score:0'] == 'intermediate' and roles['m2:R-generic-101-score:1'] == 'final'
    rescore = [e for e in kept if e['script'] == 'm2_rescore']
    assert len(rescore) == 1 and rescore[0]['resolution'] == 'runtime'
    assert all(e['conditional'] for e in kept if e['arm'] == 'legal-broad')


def identity(arm, step, model, panel='test', family='reprobe', gpu='GPU-1', source='message2'):
    return dict(source=source, family=family, arm=arm, seed=101, step=step, panel=panel, status='OK',
                model=dict(technical_synthetic=True, model=model), panel_sha256=H, prompt_sha256=None,
                decoding=dict(temperature=0), gpu_identity=dict(uuid=gpu, environment='env'), cell='%s-k%d' % (arm, step))


def pd_manifest():
    checkpoints = []
    for seed in (101, 102, 103):
        for start in ('base', 'stage1', 'stage2'):
            tensor = {'m.safetensors': hashlib.sha256((start if start == 'base' else '%s%d' % (start, seed)).encode()).hexdigest()}
            checkpoints.append(pd_contract.seal(dict(
                checkpoint_id='pd-%s-%d' % (start, seed), start=start, seed=seed, step=0 if start == 'base' else 40,
                recipe='base' if start == 'base' else 'plain-D', parent_id=None if start == 'base' else 'parent-%s-%d' % (start, seed),
                base_revision='rev', tokenizer_sha256=H, send_binding_sha256=H, export_receipt_sha256=H,
                tensor_files=tensor, weights_sha256=pd_contract.digest(tensor))))
    return pd_contract.make_manifest(checkpoints, [], preparation_sha256=H, code_sha256=H)


IDENTITIES = [identity('legal-generic', 0, 'W0'), identity('base', 0, 'W0'), identity('legal-generic', 10, 'W1'),
              identity('legal-generic', 10, 'W1')]


def test_inventory_binds_dedupes_and_scores_base_incoming_final_only():
    ids = IDENTITIES + [identity('sequence-D', 0, 'R0', family='return1.A'), identity('sequence-D', 40, 'R1', family='return1.A'),
                        identity('sequence-D', 20, 'R2', family='return1.A')]
    inv = man.inventory(launch(), pd_manifest(), (), ids)
    pd_contract.verify_seal(inv)
    missing = {m['planned_id'] for m in inv['missing']}
    assert 'm2:R-generic-101-score:1' in missing                               # final k40 never scored
    assert any(p.startswith('m2:M4-101:0:') for p in missing)                  # core cells not yet scored
    assert all(n['status'] == 'NOT-RUN' and ':legal-broad:' in n['planned_id'] for n in inv['not_run'])
    assert not any(':legal-broad:' in p for p in missing)
    w0 = man.weights_key(dict(technical_synthetic=True, model='W0'))[0]
    w1 = man.weights_key(dict(technical_synthetic=True, model='W1'))[0]
    entries = {(e['weights_sha256'], e['assignment']): e for e in inv['sp_entries']}
    assert len(entries[(w0, 'sp:reprobe-s101')]['uses']) == 2
    assert not any(k[0] == w1 for k in entries)                                # intermediate: no SP(b) scoring
    rescore = sorted(e['role'] for e in inv['sp_entries'] if e['assignment'] == 'sp:return1.A-s101')
    assert rescore == ['final', 'incoming']                                    # k20 rescoring is intermediate
    assert len([c for c in inv['cells'] if c['panel'] == 'test' and c['weights_sha256'] in (w0, w1)]) == 2
    assert len(inv['pd_starts']) == 9 and not any(s['in_chain_inventory'] for s in inv['pd_starts'])
    price = inv['price']
    n = len(inv['sp_entries'])
    assert price['entries'] == n and price['sp_a_gpu_hours'] == 0.0 and price['gate'] is False
    assert price['sp_b_gpu_hours'] == (pytest.approx(1 + 0.25 * n), pytest.approx(1 + 1.0 * n))


def test_familyless_standin_cells_match_only_when_unambiguous():
    loose = dict(identity('legal-generic', 10, 'W1'), family=None)          # planned only under reprobe
    inv = man.inventory(launch(), identities=[loose])
    assert 'm2:R-generic-101-score:0' not in {m['planned_id'] for m in inv['missing']}
    # 'base' k0 test is planned under both reprobe and core: an unassigned cell cannot choose.
    inv = man.inventory(launch(), identities=[dict(identity('base', 0, 'W0'), family=None)])
    missing = {m['planned_id'] for m in inv['missing']}
    assert 'm2:R-incoming-101:1' in missing and 'm2:M4-101:0:base:k0:test' in missing


def test_inventory_freeze_is_write_once(tmp_path):
    inv = man.inventory(launch(), identities=[identity('legal-generic', 0, 'W0')])
    man.freeze_inventory(tmp_path / 'inv.json', inv)
    assert man.freeze_inventory(tmp_path / 'inv.json', inv)['identity_sha256'] == inv['identity_sha256']
    other = man.inventory(launch(), identities=[identity('legal-generic', 0, 'W9')])
    with pytest.raises(ValueError, match='already frozen'):
        man.freeze_inventory(tmp_path / 'inv.json', other)


def test_price_flags_wall_time_risk_without_gating():
    entries = [dict(assignment='sp:core-s101') for _ in range(25)] + [dict(assignment='sp:k-s101')]
    price = man.price_entries(entries)
    assert price['wall_time_risk_at_upper_allowance'] == ['sp:core-s101'] and price['gate'] is False


def test_returned_pools_only_verify_against_frozen_supersets():
    superset = dict(schema=man.SUPERSET_SCHEMA, pools=dict(broad=[man.prompt_hash(dict(prompt='Add 2 and 3.'))]))
    ok = man.verify_pools(superset, dict(broad=[dict(prompt='  add 2 AND 3. ')]))
    assert ok['status'] == 'UNVERIFIED' and ok['containment_status'] == 'VERIFIED' and ok['deletes'] == 0
    bad = man.verify_pools(superset, dict(broad=[dict(prompt='Add 2 and 3.'), dict(prompt='new row')], k=[dict(prompt='x')]))
    assert bad['status'] == 'UNVERIFIED' and bad['pools']['broad']['outside'] == 1
    assert bad['pools']['k']['status'] == 'UNVERIFIED' and bad['deletes'] == 0
    assert man.verify_pools(superset, {})['status'] == 'UNVERIFIED'          # nothing returned is not verified


def test_superset_read_from_m3a2_published_path(tmp_path):
    assert man.SUPERSET_PATH.as_posix() == 'resources/m3/sp-candidate-superset-v1.json'
    absent = man.verify_pools_file(tmp_path / 'none.json', dict(broad=[dict(prompt='x')]))
    assert absent['status'] == 'UNVERIFIED' and 'not published' in absent['reason'] and absent['deletes'] == 0
    path = tmp_path / 'sp-candidate-superset-v1.json'
    path.write_text(json.dumps(dict(schema=man.SUPERSET_SCHEMA, pools=dict(broad=[man.prompt_hash(dict(prompt='x'))]))))
    assert man.verify_pools_file(path, dict(broad=[dict(prompt='x')]))['status'] == 'UNVERIFIED'
    (tmp_path / 'returned').mkdir()
    (tmp_path / 'returned' / 'broad.jsonl').write_text(json.dumps(dict(prompt='x')) + '\n')
    assert man.main(['verify-pools', '--superset', str(path), '--pools-dir', str(tmp_path / 'returned'),
                     '--out', str(tmp_path / 'v.json')]) == 0
    assert json.loads((tmp_path / 'v.json').read_text())['status'] == 'UNVERIFIED'
    _, _, side = sidecar_pair()
    command = next(t for t in side['tasks'] if t['id'] == 'SP-verify-pools')['commands'][0]
    assert '--pools-spec' in command and '{sp}/returned-pools.json' in command


# ---------------------------------------------------------------------------------------------
# On-disk layouts: m2_score, x_score aliases, m3_score aliases.

def write_cell(root, arm, step, tolerant, model, family='reprobe', gpu='GPU-1', truncated=(), drop=(), panel='test',
               identity_name='m2-identity.json', extra=None):
    cell = root / 'scores' / ('%s-s101-k%d-%s' % (arm, step, panel))
    (cell / 'raw').mkdir(parents=True)
    rows = [raw_row(q, 2048 if q in truncated else 3 + i, 'length' if q in truncated else 'stop')
            for i, q in enumerate(tolerant) if q not in drop]
    with (cell / 'raw' / 'responses.jsonl').open('w') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in rows)
    with (cell / 'scores.csv').open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=['arm', 'seed', 'step', 'panel', 'question_id', 'correct', 'tolerant_correct'])
        w.writeheader()
        for q, v in tolerant.items():
            w.writerow(dict(arm=arm, seed=101, step=step, panel=panel, question_id=q, correct=v if panel != 'test' else 0,
                            tolerant_correct=v))
    (cell / identity_name).write_text(json.dumps(dict(
        arm=arm, seed=101, step=step, panel=panel, model=dict(technical_synthetic=True, model=model), panel_sha256=H,
        decoding=dict(temperature=0), gpu_identity=dict(uuid=gpu, environment='env'),
        assignment=dict(family=family, seed=101), **(extra or {}))))
    return cell


def write_alias(root, name, source, scores_sha256, copy_identity=None):
    cell = root / 'scores' / name
    cell.mkdir(parents=True)
    target = root / source if '/' in source else root / 'scores' / source
    identity_file = next(p for p in target.iterdir() if p.name in man.IDENTITY_FILES)
    (cell / 'alias.json').write_text(json.dumps(dict(source=source, scores_sha256=scores_sha256,
                                                  identity_sha256=pd_contract.sha256_file(identity_file))))
    if copy_identity is not None:
        (cell / 'm2-identity.json').write_text(json.dumps(copy_identity))
        (cell / 'scores.csv').write_text('arm,seed,step,panel,question_id,correct,tolerant_correct\n')
    return cell


def test_discover_follows_x_and_m3_aliases_with_hash_checks(tmp_path):
    x = tmp_path / 'x'
    pre = write_cell(x, 'x-pre', 0, {'q1': 1}, 'XP', family='x-tier1')
    meta = dict(json.loads((pre / 'm2-identity.json').read_text()), arm='x-control')
    write_alias(x, 'x-control-s101-k0-test', 'scores/x-pre-s101-k0-test', pd_contract.sha256_file(pre / 'scores.csv'), meta)
    cells = {c['arm']: c for c in man.discover(x)}
    assert cells['x-control']['raw'] == str(pre / 'raw' / 'responses.jsonl') and cells['x-control']['status'] == 'OK'
    m3 = tmp_path / 'm3'
    base = write_cell(m3, 'base', 0, {'q1': 1}, 'B', family='m3', panel='finqa', identity_name='m3-identity.json')
    write_alias(m3, 'K.finqa-s101-k0-finqa', 'base-s101-k0-finqa', pd_contract.sha256_file(base / 'scores.csv'))
    write_alias(m3, 'CR.finqa-s101-k0-finqa', 'base-s101-k0-finqa', H)                        # stale alias
    found = {c['arm']: c for c in man.discover(m3)}
    assert found['K.finqa']['source'] == 'message3' and found['K.finqa']['scores'] == str(base / 'scores.csv')
    assert found['K.finqa']['model'] == found['base']['model'] and found['CR.finqa']['status'] == 'UNVERIFIED'


def test_m3_launch_binds_alias_incoming_and_final(tmp_path):
    m3 = tmp_path / 'm3'
    base = write_cell(m3, 'base', 0, {'q1': 1}, 'B', family='m3', panel='finqa', identity_name='m3-identity.json')
    write_alias(m3, 'K.finqa-s101-k0-finqa', 'base-s101-k0-finqa', pd_contract.sha256_file(base / 'scores.csv'))
    write_cell(m3, 'K.finqa', 40, {'q1': 0}, 'K1', family='m3', panel='finqa', identity_name='m3-identity.json')
    planned, skipped = man.planned_from_launch(m3_launch())
    assert {e['arm'] for e in planned} == {'base', 'K.finqa'} and skipped[0]['action'] == 'validate-hellaswag'
    inv = man.inventory([launch(), m3_launch()], identities=man.discover(m3))
    m3_entries = [e for e in inv['sp_entries'] if e['assignment'] == 'sp:m3-s101']
    assert sorted(len(e['uses']) for e in m3_entries) == [1, 2]                 # base+incoming share weights
    assert not any(m['planned_id'].startswith('m3:') for m in inv['missing'])
    assert len(inv['launch_sha256']) == 2


# ---------------------------------------------------------------------------------------------
# Sidecar isolation.

def sidecar_pair():
    launches = [launch(), m3_launch()]
    ids = IDENTITIES + [identity('base', 0, 'MB', panel='finqa', family='m3', source='message3'),
                        identity('K.finqa', 40, 'MK', panel='finqa', family='m3', source='message3')]
    inv = man.inventory(launches, identities=ids)
    return launches, inv, man.build_sidecar(launches, inv)


def test_sidecar_is_isolated_and_one_gpu_task_per_assignment():
    launches, inv, side = sidecar_pair()
    before = copy.deepcopy(launches)
    order = man.validate_sidecar(side, launches)
    assert launches == before                              # parent manifests never modified
    assert order[-1] == 'SP-read' and all(t.startswith('SP-') for t in order)
    panel = {t['assignment']: t for t in side['tasks'] if t['id'].startswith('SP-panel-')}
    assert set(panel) == {'sp:reprobe-s101', 'sp:m3-s101', 'sp:core-s101', 'sp:x-tier1-s101',
                          'sp:x-tier2.cap-s101', 'sp:return1.A-s101'}
    assert all(t['uses_gpu'] for t in panel.values())
    assert 'R-incoming-101' in panel['sp:reprobe-s101']['after'] and 'M3-K-101-finqa-score' in panel['sp:m3-s101']['after']
    assert all(t['dependency'] == 'afterany' and t['blocks'] == [] for t in side['tasks'])
    assert not any('{out}' in a for t in side['tasks'] for c in t['commands'] for a in c)
    with pytest.raises(ValueError, match='different launch'):
        man.validate_sidecar(side, launches[:1])


@pytest.mark.parametrize('tamper', ['parent-requires', 'parent-imports', 'writes-out', 'afterok', 'barrier',
                                    'launch-changed', 'parent-runs-sp', 'shared-gpu', 'm3-requires'])
def test_sidecar_validation_rejects_blocking_or_writing(tamper):
    launches, inv, side = sidecar_pair()
    l = launches[0]
    if tamper in ('parent-requires', 'parent-imports', 'parent-runs-sp', 'm3-requires'):
        if tamper == 'parent-requires':
            l['tasks'][-1]['requires'].append('SP-read')
        elif tamper == 'parent-imports':
            l['tasks'][-1]['imports'] = [dict(task='SP-length', source='x', target='y')]
        elif tamper == 'm3-requires':
            launches[1]['tasks'][-1]['requires'].append('SP-panel-m3-s101')
        else:
            l['tasks'][-1]['commands'].append(['py', '{kit}/sp_score.py', 'panel'])
        side['parent_launch_sha256'] = [pd_contract.digest(x) for x in launches]
    elif tamper == 'writes-out':
        side['tasks'][0]['commands'][0][-1] = '{out}/sp.json'
    elif tamper == 'afterok':
        side['tasks'][0]['dependency'] = 'afterok'
    elif tamper == 'barrier':
        side['tasks'][0]['commit_barrier'] = 'M9'
    elif tamper == 'launch-changed':
        l['tasks'].append(dict(id='NEW', kind='prepare', row='X', requires=[], commands=[]))
    elif tamper == 'shared-gpu':
        gpu_tasks = [t for t in side['tasks'] if t.get('uses_gpu')]
        gpu_tasks[1]['assignment'] = gpu_tasks[0]['assignment']
    with pytest.raises(ValueError):
        man.validate_sidecar(side, launches)


def fake_runner(code):
    calls = []
    def run(command, check=False):
        calls.append(command)
        if code == 'raise':
            raise OSError('node lost')
        return subprocess.CompletedProcess(command, code)
    return run, calls


def state(tmp_path, task):
    return json.loads((tmp_path / 'state' / (task + '.json')).read_text())


def test_run_never_fails_and_records_status(tmp_path):
    _, inv, side = sidecar_pair()
    values = {'{kit}': ROOT / 'kit', '{sp}': tmp_path, '{score_gpu}': '0'}
    for code, status in ((0, 'DONE'), (3, 'FAILED'), ('raise', 'FAILED')):
        runner, calls = fake_runner(code)
        assert man.run(side, 'SP-verify-pools', values, {}, runner=runner) == 0
        assert state(tmp_path, 'SP-verify-pools')['status'] == status
    task = next(t for t in side['tasks'] if t.get('assignment') == 'sp:reprobe-s101')
    runner, calls = fake_runner(0)
    man.run(side, task['id'], values, {}, runner=runner)
    assert state(tmp_path, task['id'])['status'] == 'MISSING-INPUT' and not calls
    # A missing/failed preparation never consumes a stale entry from a previous attempt.
    values['{models_file}'] = tmp_path / 'models.json'
    runner, calls = fake_runner(3)
    man.run(side, task['id'], values, {}, inputs=[tmp_path], runner=runner)
    assert state(tmp_path, task['id'])['status'] == 'FAILED'
    assert all(not c[1].endswith('sp_score.py') for c in calls)
    runner, calls = fake_runner(0)
    man.run(side, 'SP-length', values, {}, runner=runner)
    assert state(tmp_path, 'SP-length')['status'] == 'MISSING-INPUT' and not calls
    man.run(side, 'SP-length', values, {}, inputs=[tmp_path / 'a', tmp_path / 'b'], runner=runner)
    assert calls[-1].count('--input') == 2
    man.run(side, 'SP-nope', values, {}, runner=runner)
    assert state(tmp_path, 'SP-nope')['status'] == 'FAILED'


def test_job_templates_are_isolated_and_exit_zero(tmp_path):
    jobs = ROOT / 'kit' / 'jobs' / 'message3-sp'
    for script in ('run-task.sh', 'plan.sh'):
        text = (jobs / script).read_text()
        assert subprocess.run(['bash', '-n', str(jobs / script)]).returncode == 0
        assert text.rstrip().endswith('exit 0') and 'set -e' not in text and 'afterany' in (jobs / 'README.md').read_text()
    side = dict(schema=man.SIDECAR_SCHEMA, parent_launch_sha256=[H], inventory_sha256=H, tasks=[
        dict(id='SP-fail', kind='sp-sidecar', after=[], dependency='afterany', writes='{sp}', uses_gpu=False, blocks=[],
             commands=[[sys.executable, '-c', 'raise SystemExit(3)']])])
    (tmp_path / 'side.json').write_text(json.dumps(side))
    env = dict(KIT=str(ROOT / 'kit'), SP_ROOT=str(tmp_path / 'sp'), SIDECAR=str(tmp_path / 'side.json'), TASK='SP-fail',
               JOB_PY=sys.executable, PATH='/usr/bin:/bin')
    assert subprocess.run(['bash', str(jobs / 'run-task.sh')], env=env).returncode == 0
    assert state(tmp_path / 'sp', 'SP-fail')['status'] == 'FAILED'


# ---------------------------------------------------------------------------------------------
# SP(a) length over existing scorer cells.

def test_run_length_freezes_incoming_cohort_and_never_conditions_on_descendants(tmp_path):
    parent = tmp_path / 'parent'
    write_cell(parent, 'legal-generic', 0, {'q1': 1, 'q2': 1, 'q3': 0}, 'W0')
    write_cell(parent, 'legal-generic', 10, {'q1': 0, 'q2': 1, 'q3': 1}, 'W1', drop=('q2',))
    write_cell(parent, 'legal-generic', 40, {'q1': 1, 'q2': 0, 'q3': 1}, 'W2', truncated=('q3',))
    write_cell(parent, 'base', 0, {'q1': 1, 'q2': 0, 'q3': 0}, 'B', gpu='GPU-9')
    inv = man.inventory(launch(), identities=man.discover(parent))
    out = man.run_length(inv, tmp_path / 'sp', [parent])
    assert out['gpu_hours'] == 0.0
    fam = next(f for f in out['families'] if f['family'] == 'reprobe.legal-generic')
    assert [c['status'] for c in fam['cells']] == ['SCORED', 'SCORED', 'SCORED'] and fam['status'] == 'INCOMPLETE'
    k10, k40 = fam['cells'][1]['metrics'], fam['cells'][2]['metrics']
    assert k10['status'] == 'INCOMPLETE' and k10['full']['missing'] == 1 and k10['full']['n'] == 3
    assert k40['cohort']['cohort_size'] == 2 and k40['cohort']['n_correct'] == 1      # q2 lost, still counted
    assert k40['full']['truncation_rate'] == pytest.approx(1 / 3)
    cohort_file = tmp_path / 'sp' / 'cohorts' / 'message2.reprobe.legal-generic.s101.test.json'
    assert json.loads(cohort_file.read_text())['correct_column'] == 'tolerant_correct'
    frozen = cohort_file.read_bytes()
    (parent / 'scores' / 'legal-generic-s101-k40-test' / 'scores.csv').write_text(
        'arm,seed,step,panel,question_id,correct,tolerant_correct\n' +
        ''.join('legal-generic,101,40,test,%s,1,1\n' % q for q in ('q1', 'q2', 'q3')))
    man.run_length(inv, tmp_path / 'sp', [parent])
    assert cohort_file.read_bytes() == frozen                                          # C never moves
    base = out['base_cohorts'][0]
    assert base['kind'] == 'base' and len(base['cells']) == 4
    assert all(c['same_gpu_as_base'] is False for c in base['cells'] if c['arm'] == 'legal-generic')
    assert out['pending_plans'] == ['m2:RA-A:0']
    core = [f for f in out['families'] if f['family'].startswith('core.')]
    assert core and all(f['status'] == 'MISSING-INCOMING' for f in core)              # nothing scored yet


# ---------------------------------------------------------------------------------------------
# Panel job and reader on the frozen-format v2 panel through kit/sp_prepare.py (no stub).

def technical_freeze(root):
    """A technical frozen-format v2 directory assembled only from sp_prepare's own functions and the
    author's sealed fixtures, so sp_score consumes sp_prepare.verify_frozen/load_checkers unchanged.
    It is labelled technical_synthetic and lives under pytest's tmp; it is not the scientific
    freeze, which only sp_prepare.freeze() writes after two genuine independent reviews."""
    root.mkdir(parents=True)
    requests, source_files = prep.load_source(prep.SOURCE, reviews=None)
    panel = prep.panel_rows(requests)
    prep.write_jsonl(root / 'panel.jsonl', panel)
    prep.write_jsonl(root / 'requests.jsonl', requests)
    (root / 'fixtures.jsonl').write_bytes((prep.RESOURCES / 'authored-fixtures.jsonl').read_bytes())
    doc = dict(schema=prep.SCHEMA, technical_synthetic=True, checker_version=CK.VERSION,
               checker_sha256=pd_contract.sha256_file(prep.CHECKERS), source_files=source_files,
               code=prep.code_hashes(), resources=prep.resource_hashes(), n_requests=len(requests), n_prompts=len(panel))
    doc['files'] = {n: pd_contract.sha256_file(root / n) for n in ('panel.jsonl', 'requests.jsonl', 'fixtures.jsonl')}
    prep.write_json(root / 'FROZEN.json', doc)
    return panel


def test_load_panel_consumes_sp_prepare_freeze_format(tmp_path):
    panel = technical_freeze(tmp_path / 'frozen')
    doc, loaded, ck = sc.load_panel(tmp_path / 'frozen')
    assert loaded == panel and len(loaded) == doc['n_prompts'] == 400 and ck.VERSION == CK.VERSION
    sc.verify_fixtures(tmp_path / 'frozen', ck)                                 # author's sealed fixtures pass
    with (tmp_path / 'frozen' / 'panel.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='frozen SP file changed'):
        sc.load_panel(tmp_path / 'frozen')
    with pytest.raises(FileNotFoundError):
        sc.load_panel(tmp_path / 'absent')


def test_panel_job_standin_reader_and_wall_time(tmp_path):
    technical_freeze(tmp_path / 'frozen')
    incoming = dict(entry_id='sp-incoming', checkpoint_id='w0', weights_sha256=H, assignment='sp:core-s101',
                    uses=[dict(family='core', seed=101, role='incoming', step=0)])
    final = dict(entry_id='sp-final', checkpoint_id='w1', weights_sha256='d' * 64, assignment='sp:core-s101',
                 uses=[dict(family='core', seed=101, role='final', step=40)])
    for entry in (incoming, final):
        manifest = sc.panel_job(entry, None, '0', tmp_path, tmp_path / 'frozen', standin=True, gpu_identity=GPU)
        assert set(manifest['files']) == {'raw.jsonl', 'scores.jsonl', 'length.json', 'identity.json'}
    assert sc.panel_job(final, None, '0', tmp_path, tmp_path / 'frozen', standin=True, gpu_identity=GPU) == manifest
    scored = [json.loads(line) for line in (tmp_path / 'panel' / 'sp-final' / 'scores.jsonl').read_text().splitlines()]
    assert len(scored) == 400 and {r['primary_family'] for r in scored} == set(CK.FAMILIES)
    assert any(r['json_fence_failure'] is not None for r in scored)            # json_only prompts reported
    timing = json.loads((tmp_path / 'panel' / 'sp-final' / 'identity.json').read_text())['timing']
    assert set(timing) == {'load', 'score', 'receipt', 'serial_wall_seconds'}
    cells = rd.read_sidecar(tmp_path)
    tables = rd.family_tables(cells, dict(sp_entries=[incoming, final, dict(entry_id='sp-missing')]))
    rows = tables['families']['core']['by_seed']['101']
    assert {r['role'] for r in rows} == {'incoming', 'final'}
    row = next(r for r in rows if r['role'] == 'final')
    assert row['gap_change'] == row['template_reliance_change'] == 0.0
    assert tables['missing_entries'] == ['sp-missing'] and tables['label'] == 'DESCRIPTIVE'
    wall = rd.wall_time(cells)
    assert wall['assignments']['sp:core-s101']['entries'] == 2 and wall['assignments']['sp:core-s101']['flag'] is None
    assert wall['gate'] is False
    with pytest.raises(ValueError, match='MISSING'):
        sc.verify_model(dict(model_file_hashes={'m.safetensors': H}, weights_sha256=H), tmp_path)

# Review regression coverage (m3b2s). All data are local technical fixtures.

def durable_export(tmp_path, weights=b'weights', expiry=None):
    """Owner-produced durable snapshot + existing export; SP only reads these artifacts."""
    import io
    import tarfile
    root = tmp_path / 'durable'
    model = root / 'export'
    model.mkdir(parents=True)
    (model / 'm.safetensors').write_bytes(weights)
    files = {'m.safetensors': pd_contract.sha256_file(model / 'm.safetensors')}
    owner = json.dumps(dict(model_file_hashes=files), sort_keys=True).encode()
    policy = json.dumps(dict(policy='owner-existing-retention', expires_at=expiry), sort_keys=True).encode()
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    payload = {'export/m.safetensors': weights, 'receipt.json': owner, 'retention.json': policy}
    snapshot = dict(schema='message2-stage-snapshot.v2', key='FINAL', complete=True,
                    files={k: dict(sha256=sha(v), bytes=len(v)) for k, v in payload.items()})
    archive = root / 'snapshot.tar'
    with tarfile.open(archive, 'w') as tar:
        for name, raw in {'snapshot.json': json.dumps(snapshot, sort_keys=True).encode(),
                          **{'tree/' + k: v for k, v in payload.items()}}.items():
            info = tarfile.TarInfo(name); info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    receipt = root / 'durable-export.json'
    receipt.write_text(json.dumps(dict(schema='sp-durable-export-v1', durable=True, export='export',
        model_file_hashes=files, snapshot_sha256=pd_contract.sha256_file(archive),
        snapshot_manifest_sha256=pd_contract.digest(snapshot), snapshot_export_prefix='export',
        owner_receipt_member='receipt.json', owner_receipt_sha256=sha(owner),
        retention_policy_member='retention.json', retention_policy_sha256=sha(policy), expires_at=expiry)))
    return dict(model=str(model), receipt=str(receipt), receipt_sha256=pd_contract.sha256_file(receipt),
                archive=str(archive), durable_root=str(root))


def simple_launch(panels=('finqa', 'test', 'math_hard')):
    return dict(schema='message2-launch.v2', tasks=[
        dict(id='scores', kind='score-family', family='reprobe', seed=101, requires=[],
             commands=[cmd('m2_reprobe', action='score', arm='generic', seed=101, step=step, panel_set=panel)
                       for step in (0, 40) for panel in panels])])


def test_review_1_multi_panel_integration(tmp_path):
    parent = tmp_path / 'parent'
    panels = ('finqa', 'test', 'math_hard', 'general-v1')
    for panel in panels:
        for step in (0, 40):
            outcomes = {sub + '-q' + str(i): int(i == 1 or step == 40)
                        for sub in man.GENERAL_SUBPANELS for i in (1, 2)} if panel == 'general-v1' else {'q1': 1, 'q2': int(step == 40)}
            cell = write_cell(parent, 'legal-generic', step, outcomes, 'W%d' % step, panel=panel)
            if panel == 'general-v1':
                rows = sc.read_scores_csv(cell / 'scores.csv')
                for row in rows: row['panel'] = row['question_id'].split('-')[0]
                with (cell / 'scores.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    inv = man.inventory(simple_launch(panels), identities=man.discover(parent))
    out = man.run_length(inv, tmp_path / 'sp', [parent])
    assert len(out['families']) == 6
    files = sorted((tmp_path / 'sp' / 'cohorts').glob('*.json'))
    assert len(files) == 6 and len({f.name for f in files}) == 6
    assert all('message2.reprobe.legal-generic.s101.' in f.name for f in files)
    before = {f.name: f.read_bytes() for f in files}
    assert man.run_length(inv, tmp_path / 'sp', [parent]) == out
    assert {f.name: f.read_bytes() for f in files} == before


def test_review_2_4_real_producer_manifest_shapes(tmp_path, monkeypatch):
    import importlib
    for name in ('m3_manifest', 'm2_manifest'):
        if not (ROOT / 'kit' / (name + '.py')).is_file():
            pytest.skip('optional producer module absent from this kit: ' + name)
    m3 = importlib.import_module('kit.m3_manifest')
    core = importlib.import_module('kit.m2_manifest')
    # Builders read settings only; CPU construction neither trains nor submits anything.
    screen = tmp_path / 'screen.json'
    screen.write_text(json.dumps(dict(v4_step20=[], stages=[])))
    settings = dict(trainer_python='train', inference_python='score', nodes=[],
                    inputs=dict(inventory='inventory', base='base', base_receipt='receipt',
                                contract_targets='targets', replay='replay', panels={}),
                    runtime={}, addon=dict(spec='spec', smoke=False), bind={}, next_source='source',
                    pools={}, prepare={}, screen=dict(spec=str(screen), sources='sources'))
    m3doc = m3.build(settings)
    coredoc = core.build(settings)
    planned, _ = man.planned_from_launch(m3doc)
    aliases = [p for p in planned if p['action'] == 'alias']
    assert len(aliases) == 216 and all(p['family'] == 'm3' for p in aliases)
    assert {p['seed'] for p in aliases} == {101, 102, 103}
    assert all(p['task'].endswith('-train') for p in aliases)
    from kit.m2_return1 import decisions
    inv = man.inventory([coredoc, m3doc], return1=decisions()['rescoring'])
    assert inv['sp_entries'] and all(not e['weights_bound'] for e in inv['sp_entries'])
    assert not inv['price']['coverage_unknown']
    assert {p['role'] for p in inv['planned']} >= {'base', 'incoming', 'final', 'intermediate'}
    families, pending = man.length_plan(inv['planned'])
    assert not pending and any(f['source'] == 'message3' for f in families)
    assert any(f['family'].startswith('return1.') for f in families)
    assert all(f['panel'] not in man.NOT_APPLICABLE for f in families)
    side = man.build_sidecar([coredoc, m3doc], inv)
    man.validate_sidecar(side, [coredoc, m3doc])
    assert len([t for t in side['tasks'] if t['uses_gpu']]) >= 6


def test_review_4_6_runtime_binding_survives_frozen_plan_only_with_owner_receipts(tmp_path):
    local = durable_export(tmp_path)
    inv = man.inventory(simple_launch(('finqa',)))
    entry = next(e for e in inv['sp_entries'] if e['role'] == 'final')
    path = tmp_path / 'inventory.json'
    man.freeze_inventory(path, inv); before = path.read_bytes()
    parent = tmp_path / 'parent'
    cell = write_cell(parent, 'legal-generic', 40, {'q': 1}, 'W40', panel='finqa')
    meta = json.loads((cell / 'm2-identity.json').read_text())
    proof = man.durable_binding(local)
    meta['model'] = dict(model_file_hashes=proof['model_file_hashes'])
    (cell / 'm2-identity.json').write_text(json.dumps(meta))
    bound = man.bind_runtime_entry(entry, inv, [parent], local)
    assert bound['entry_id'] == entry['entry_id'] and bound['weights_bound']
    assert bound['durable_provenance']['snapshot_sha256'] == proof['snapshot_sha256']
    assert path.read_bytes() == before
    sc.verify_model(bound, local['model'])
    with pytest.raises(ValueError, match='owner durable'):
        man.durable_binding(local['model'])
    moved = dict(local, model=str(tmp_path / 'transient'))
    Path(moved['model']).mkdir()
    with pytest.raises(ValueError, match='transient'):
        man.durable_binding(moved)
    archive = Path(local['archive']); archive.unlink()
    with pytest.raises(man.MissingInput, match='no longer survives'):
        man.bind_runtime_entry(entry, inv, [parent], local)
    assert path.read_bytes() == before and Path(local['model']).exists()


def test_review_5_freeze_hardware_rejects_parent_shared_and_missing_environment(tmp_path, monkeypatch):
    parent = [dict(complete_inventory=True, gpus=[dict(uuid='GPU-parent')])]
    assignments = {'sp:m3-s101': GPU}
    hardware = man.freeze_hardware(tmp_path / 'hardware.json', assignments, parent)
    assert hardware == man.freeze_hardware(tmp_path / 'hardware.json', assignments, parent)
    for bad in ({'sp:m3-s101': dict(uuid='GPU-parent', environment='env')},
                {'sp:a-s101': GPU, 'sp:b-s101': GPU}, {'sp:a-s101': dict(uuid='GPU-1')}):
        with pytest.raises(ValueError): man.freeze_hardware(tmp_path / 'bad.json', bad, parent)
    with pytest.raises(ValueError): man.freeze_hardware(tmp_path / 'bad.json', assignments, [])
    entry = dict(assignment='sp:m3-s101')
    expected = pd_contract.seal(dict(GPU, assignment=entry['assignment'], hardware_sha256=hardware['identity_sha256']))
    from kit import m2_assignment, m2_gpu
    called = []
    monkeypatch.setattr(m2_gpu, 'prepare', lambda gpu, expected_uuid: called.append((gpu, expected_uuid)) or dict(index='7'))
    monkeypatch.setattr(m2_gpu, 'assert_current', lambda: called.append('cuda-verified'))
    monkeypatch.setattr(m2_assignment, 'inventory', lambda gpu: dict(GPU, node='technical'))
    index, observed = sc.observed_hardware(entry, expected)
    assert index == '7' and observed == GPU and called[0] == ('GPU-1', 'GPU-1') and called[-1] == 'cuda-verified'
    monkeypatch.setattr(m2_assignment, 'inventory', lambda gpu: dict(uuid='GPU-wrong', environment='env'))
    with pytest.raises(ValueError, match='observed GPU'): sc.observed_hardware(entry, expected)
    monkeypatch.setattr(m2_assignment, 'inventory', lambda gpu: dict(uuid='GPU-1', environment='other'))
    with pytest.raises(ValueError, match='environment'): sc.observed_hardware(entry, expected)


def test_review_6_expired_owner_export_is_missing(tmp_path):
    local = durable_export(tmp_path)
    path = Path(local['receipt']); receipt = json.loads(path.read_text()); receipt['expires_at'] = 1
    path.write_text(json.dumps(receipt)); local['receipt_sha256'] = pd_contract.sha256_file(path)
    before = {p: p.read_bytes() for p in Path(local['durable_root']).rglob('*') if p.is_file()}
    with pytest.raises(man.MissingInput, match='expired'): man.durable_binding(local)
    assert before == {p: p.read_bytes() for p in before}


def test_review_7_likelihood_is_na_even_for_base_cohorts(tmp_path):
    for panel in man.NOT_APPLICABLE:
        with pytest.raises(ValueError, match='N/A'): sc.correct_column(panel)
    parent = tmp_path / 'parent'
    write_cell(parent, 'base', 0, {'q1': 1}, 'B', family='m3', panel='knowledge-ll', identity_name='m3-identity.json')
    assert man.base_cohorts(man.discover(parent), tmp_path / 'sp') == []


def test_review_8_return1_bound_calls_length_and_missing(tmp_path):
    launch_doc = dict(schema='message2-launch.v2', tasks=[dict(id='RA', kind='score-family', family='return1.A', seed=101,
        commands=[cmd('m2_rescore', action='score', package='A')])])
    calls = [dict(score_set='scores/sequence-D-s101-k%d-finqa' % step, step=step, panel_set='finqa',
                  checkpoint='portable-checkpoint-%s' % step) for step in (0, 20, 40)]
    binding = dict(packages=[dict(id='A', seed=101, calls=calls)])
    parent = tmp_path / 'parent'
    write_cell(parent, 'sequence-D', 0, {'a': 1, 'b': 0}, 'W0', family='return1.A', panel='finqa')
    write_cell(parent, 'sequence-D', 40, {'a': 0, 'b': 1}, 'W40', family='return1.A', panel='finqa')
    inv = man.inventory(launch_doc, identities=man.discover(parent), return1=binding)
    assert not man.length_plan(inv['planned'])[1]
    out = man.run_length(inv, tmp_path / 'sp', [parent])
    assert out['families'][0]['status'] == 'INCOMPLETE'
    assert out['families'][0]['cells'][1]['status'] == 'MISSING'
    assert out['families'][0]['cells'][2]['metrics']['cohort']['n_correct'] == 0


def test_review_9_full_population_and_response_text_required(tmp_path):
    co = cohort(tmp_path, 'all.json', {'a': 1, 'b': 0, 'c': 0})
    metrics = sc.length_metrics([raw_row('a', 3), raw_row('b', 4)], {'a': 1, 'b': 0}, co)
    assert metrics['status'] == 'INCOMPLETE' and metrics['full']['n'] == 3
    assert metrics['full']['n_correct'] is None and metrics['full']['missing_ids'] == ['c']
    assert metrics['full']['accuracy'] is None and metrics['full']['tokens_per_correct'] is None
    no_text = dict(id='a', token_ids=[1], output_tokens=1, finish_reason='stop')
    assert sc.length_metrics([no_text], {'a': 1})['status'] == 'INCOMPLETE'
    bad = pd_contract.seal({k: v for k, v in dict(co, population_sha256=H).items() if k != 'identity_sha256'})
    with pytest.raises(ValueError, match='population identity'): sc.length_metrics([], {}, bad)


def test_review_10_alias_identity_hash_is_enforced(tmp_path):
    base = write_cell(tmp_path, 'base', 0, {'q': 1}, 'B', panel='finqa', family='m3', identity_name='m3-identity.json')
    write_alias(tmp_path, 'K.finqa-s101-k0-finqa', base.name, pd_contract.sha256_file(base / 'scores.csv'))
    assert next(c for c in man.discover(tmp_path) if c['arm'] == 'K.finqa')['status'] == 'OK'
    meta = json.loads((base / 'm3-identity.json').read_text()); meta['model']['model'] = 'wrong'
    (base / 'm3-identity.json').write_text(json.dumps(meta))
    assert next(c for c in man.discover(tmp_path) if c['arm'] == 'K.finqa')['status'] == 'UNVERIFIED'


def test_review_10_cache_and_reader_reject_stale_evidence(tmp_path):
    technical_freeze(tmp_path / 'frozen')
    entry = dict(entry_id='sp-test', weights_sha256=H, assignment='sp:m3-s101', uses=[])
    sc.panel_job(entry, None, '0', tmp_path, tmp_path / 'frozen', standin=True, gpu_identity=GPU)
    for other, gpu in ((dict(entry, weights_sha256='b' * 64), GPU), (entry, dict(GPU, environment='other'))):
        with pytest.raises(ValueError, match='cached SP scoring identity'):
            sc.panel_job(other, None, '0', tmp_path, tmp_path / 'frozen', standin=True, gpu_identity=gpu)
    path = tmp_path / 'panel' / entry['entry_id'] / 'scores.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='output changed'): rd.read_sidecar(tmp_path)


def write_panel_result(root, entry, pattern, gpu=GPU, timing=None):
    cell = root / 'panel' / entry['entry_id']
    scored = scored_rows(pattern)
    raw = [raw_row(r['id'], 5) for r in scored]
    sc.write_jsonl(cell / 'raw.jsonl', raw)
    sc.write_jsonl(cell / 'scores.jsonl', scored)
    sc.write_json(cell / 'length.json', sc.length_metrics(raw, {r['id']: r['strict'] for r in scored}))
    sc.write_json(cell / 'identity.json', dict(entry=entry, gpu_identity=gpu, frozen_sha256=H,
        decoding_sha256=pd_contract.digest(sc.DECODING), timing=timing, technical_synthetic=True))
    sc.write_json(cell / 'manifest.json', dict(files={n: pd_contract.sha256_file(cell / n)
        for n in ('identity.json', 'raw.jsonl', 'scores.jsonl', 'length.json')}))


def test_review_3_8_11_cli_reports_stage_edges_cohorts_summaries_and_intervals(tmp_path):
    entries = []
    for seed in (101, 102, 103):
        for stage in ('finqa', 'contract_nli'):
            for arm in ('C0', 'K'):
                for step in (0, 40):
                    use = dict(source='message3', family='m3', seed=seed, role='incoming' if step == 0 else 'final',
                               arm=arm + '.' + stage, comparison=arm + '.' + stage, step=step,
                               incoming_key=['message3', 'm3', arm + '.' + stage, seed, 0, 'finqa'])
                    entry = dict(entry_id='sp-%s-%s-%s-%s' % (arm, stage, seed, step), assignment='sp:m3-s%s' % seed,
                        weights_sha256=pd_contract.digest([arm, stage, seed, step]), uses=[use])
                    entries.append(entry)
                    pattern = {'a': (1, 1), 'b': (0, 0)} if step == 0 else {'a': (1, 0), 'b': (1, 0)}
                    write_panel_result(tmp_path, entry, pattern)
    inv = pd_contract.seal(dict(sp_entries=entries, planned=[]))
    sc.write_json(tmp_path / 'inventory.json', inv)
    assert rd.main(['--sidecar', str(tmp_path), '--inventory', str(tmp_path / 'inventory.json'),
                    '--out', str(tmp_path / 'tables')]) == 0
    report = json.loads((tmp_path / 'tables' / 'sp-tables.json').read_text())
    family = report['families']['m3']
    final = next(r for r in family['by_seed']['101'] if r['role'] == 'final')
    assert final['gap_change'] == 100.0 and final['length']['cohort']['n'] == 2
    assert final['length']['cohort']['n_correct'] == 1
    assert final['length']['full']['n'] == 4
    assert final['wording']['canonical']['strict'] == 100.0
    assert final['wording']['paraphrase']['strict'] == 0.0
    assert final['instruction_by_family']['required_word'] == 50.0
    assert final['strict_by_primary_family']['required_word'] == 50.0
    change = family['intervals']['K.contract_nli']['change_from_incoming']['template_reliance_gap']
    assert change['status'] == 'COMPLETE' and change['mean'] == change['lower'] == change['upper'] == 100.0
    contrast = family['contrasts']['K.finqa-minus-C0.finqa']['template_reliance_gap']
    assert contrast['status'] == 'COMPLETE' and contrast['mean'] == 0.0
    assert len(list((tmp_path / 'cohorts').glob('*.json'))) == 12
    # Missing exclusions propagate to the complete descriptive scores without altering outcomes.
    assert report['status'] == 'UNVERIFIED' and report['pool_verification']['status'] == 'UNVERIFIED'
    cells = rd.read_sidecar(tmp_path)
    one = next(c for c in cells if c['entry']['entry_id'] == 'sp-K-finqa-102-40')
    one['gpu_identity'] = dict(uuid='GPU-other', environment='env')
    report = rd.family_tables(cells, inv)
    assert report['families']['m3']['intervals']['K.finqa']['change_from_incoming']['strict']['status'] == 'UNPAIRED-GPU'
    cells.remove(one)
    report = rd.family_tables(cells, inv)
    assert report['status'] == 'INCOMPLETE'
    assert report['families']['m3']['intervals']['K.finqa']['change_from_incoming']['strict']['status'] == 'INCOMPLETE'


def test_review_3_identical_x_incoming_aliases_deduplicate(tmp_path):
    use = lambda arm, step: dict(source='message2', family='x-tier1', seed=101, arm=arm, comparison=arm,
        role='incoming' if step == 0 else 'final', step=step,
        incoming_key=['message2', 'x-tier1', arm, 101, 0, 'test'])
    incoming = dict(entry_id='incoming', weights_sha256=H, assignment='sp:x-tier1-s101',
                    uses=[use('x-control', 0), use('x-drift', 0), use('x-control', 0)])
    final = dict(entry_id='final', weights_sha256='b' * 64, assignment='sp:x-tier1-s101', uses=[use('x-drift', 40)])
    write_panel_result(tmp_path, incoming, {'a': (1, 1)})
    write_panel_result(tmp_path, final, {'a': (1, 0)})
    report = rd.family_tables(rd.read_sidecar(tmp_path), sidecar=tmp_path)
    final_row = next(r for r in report['families']['x-tier1']['by_seed']['101'] if r['role'] == 'final')
    assert final_row['gap_change'] == 100.0


def test_review_12_author_recheck_invoked_and_response_overlap_detected(tmp_path, monkeypatch):
    frozen = tmp_path / 'frozen'
    technical_freeze(frozen)
    requests = sc.read_jsonl(frozen / 'requests.jsonl')
    entries, bound = [], []
    for role in ('broad', 'k', 'cr'):
        rows = [dict(id=role, prompt='A distinct synthetic request about zeppelins and quartz.',
                     response=requests[0]['witness'] if role == 'k' else 'unrelated fixture response')]
        path = tmp_path / (role + '.jsonl'); sc.write_jsonl(path, rows)
        parent_hash = pd_contract.digest(role)
        entries.append(dict(name=role + '-return', role=role, path=path.name, sha256=pd_contract.sha256_file(path),
                            scope='accepted', candidate_superset_name=role, candidate_superset_sha256=parent_hash))
        bound.append(dict(name=role, role=role, sha256=parent_hash, candidate_members=prep.candidate_members(dict(rows=rows))))
    doc = json.loads((frozen / 'FROZEN.json').read_text()); doc['pools'] = bound
    sc.write_json(frozen / 'FROZEN.json', doc)
    spec = tmp_path / 'returned.json'
    sc.write_json(spec, dict(schema='sp-pools-v2', complete_inventory=True, pools=entries))
    called = []
    real = prep.recheck
    def spy(*args, **kw):
        called.append(args)
        return real(*args, **kw)
    monkeypatch.setattr(prep, 'recheck', spy)
    result_path = tmp_path / 'verification.json'
    assert man.main(['verify-pools', '--frozen', str(frozen), '--pools-spec', str(spec), '--out', str(result_path)]) == 0
    result = json.loads(result_path.read_text())
    assert called and result['status'] == 'UNVERIFIED'
    assert any(h['kind'] == 'witness-vs-pool' and h.get('field') == 'response' for h in result['hits'])
    assert result['panel_changed'] is False and result['rescoring_allowed'] is False
    entries.pop()
    sc.write_json(spec, dict(schema='sp-pools-v2', complete_inventory=True, pools=entries))
    result = real(frozen, spec)
    assert any(h['kind'] == 'missing-return-role' for h in result['hits'])


def test_review_13_wall_time_includes_missing_and_parent_costs():
    use = dict(source='message3', family='m3', arm='K.finqa', seed=101)
    e = dict(entry_id='ready', assignment='sp:m3-s101', uses=[use])
    missing = dict(e, entry_id='missing')
    p = dict(use, source='message3', step=40, panel='finqa')
    cell = dict(entry=e, timing=dict(load=100, score=90000, receipt=100), gpu_hours=25.0)
    observed = dict(p, status='OK', timing=dict(load=100, score=300, receipt=20))
    report = rd.wall_time([cell], dict(sp_entries=[e, missing], planned=[p]), [observed])
    assert report['status'] == 'UNMEASURED' and report['gate'] is False
    assert report['assignments']['sp:m3-s101']['entries'] == 2
    arm = report['arms']['message3.m3.K.finqa-s101']
    assert arm['entries'] == 3 and arm['untimed'] == 1 and arm['flag'] == 'WALL-TIME RISK'
    assert arm['serial_wall_hours'] == pytest.approx((90200 + 420) / 3600)


def test_review_4_unknown_package_coverage_has_no_finite_upper_price():
    inv = man.inventory(launch())
    assert inv['sp_entries'] and inv['price']['coverage_unknown']
    assert inv['price']['sp_b_gpu_hours'][1] is None
    assert any(e['coverage_unknown'] for e in inv['sp_entries'] if e['assignment'] == 'sp:return1.A-s101')


def test_review_5_6_local_substitution_requires_frozen_hardware_and_durable_owner(tmp_path):
    local = durable_export(tmp_path)
    assignment = 'sp:m3-s101'
    hardware = man.freeze_hardware(tmp_path / 'hardware.json', {assignment: GPU},
                                   [dict(complete_inventory=True, gpus=[dict(uuid='GPU-parent')])])
    values = {'{sp}': tmp_path / 'sp'}
    maps = dict(models={'entry': local}, hardware=hardware)
    man.freeze_inventory(tmp_path / 'sp' / 'entries' / 'entry.json',
                         pd_contract.seal(dict(durable_provenance=man.durable_binding(local))))
    assert man.substitute('{model:entry}', values, maps) == local['model']
    path = Path(man.substitute('{gpu_identity:sp:m3-s101}', values, maps))
    proof = json.loads(path.read_text()); pd_contract.verify_seal(proof)
    assert proof['hardware_sha256'] == hardware['identity_sha256'] and proof['assignment'] == assignment
    with pytest.raises(man.MissingInput): man.substitute('{gpu_identity:sp:m3-s101}', values, {})
    Path(local['model']).rename(Path(local['model']).with_name('deleted'))
    with pytest.raises(man.MissingInput, match='no longer survives'):
        man.substitute('{model:entry}', values, maps)


def test_review_6_retention_deadline_cannot_be_extended_by_adapter(tmp_path):
    local = durable_export(tmp_path, expiry=1)
    path = Path(local['receipt']); doc = json.loads(path.read_text()); doc['expires_at'] = None
    path.write_text(json.dumps(doc)); local['receipt_sha256'] = pd_contract.sha256_file(path)
    with pytest.raises(ValueError, match='changes owner retention deadline'): man.durable_binding(local)


def test_review_8_retained_return1_real_shape_without_raw_is_incomplete(tmp_path):
    parent = tmp_path / 'parent'
    cell = write_cell(parent / 'retained' / 'A', 'sequence-D', 0, {'q': 1}, 'W0', panel='finqa')
    original = cell / 'identity.json'; original.write_text('{}')
    companion = cell / 'retained-machine.json'; companion.write_text(json.dumps(dict(gpu_identity=GPU)))
    files = {'m.safetensors': H}
    call = dict(score_set='scores/' + cell.name, step=0, panel_set='finqa', model_hashes=files,
        identity_path='scores/' + cell.name + '/identity.json', identity_sha256=pd_contract.sha256_file(original),
        companion_path='scores/' + cell.name + '/retained-machine.json')
    binding = dict(packages=[dict(id='A', seed=101, calls=[], retained_calls=[call],
                                  assigned_gpu_uuid=GPU['uuid'], compatible_environment=GPU['environment'])])
    sc.write_json(parent / 'binding.json', binding)
    launch_doc = dict(schema='message2-launch.v2', tasks=[dict(id='RA', kind='score-family', family='return1.A', seed=101,
        commands=[cmd('m2_rescore', action='score', package='A')])])
    planned = man.comparison_edges(man.expand_return1(man.planned_from_launch(launch_doc)[0], binding))
    retained = man.retained_return1(parent, planned)
    assert len(retained) == 1 and retained[0]['status'] == 'OK'
    # Core retained products can omit raw responses; never infer their length from scores.csv.
    Path(retained[0]['raw']).unlink()
    retained = man.retained_return1(parent, planned)
    raw, _, correct = man._cell(retained[0], 'finqa', None)
    assert sc.length_metrics(raw, correct)['status'] == 'INCOMPLETE'


def test_review_13_runner_wall_includes_binding_without_double_counting():
    entry = dict(entry_id='entry', assignment='sp:m3-s101', uses=[dict(source='message3', family='m3', arm='K', seed=101)])
    cell = dict(entry=entry, timing=dict(serial_wall_seconds=20), gpu_hours=1)
    execution = [dict(commands=[dict(entry_id='entry', status='DONE', wall_seconds=11),
                                dict(entry_id='entry', status='DONE', wall_seconds=29)])]
    report = rd.wall_time([cell], dict(sp_entries=[entry], planned=[]), execution=execution)
    assert report['assignments']['sp:m3-s101']['serial_wall_hours'] == pytest.approx(40 / 3600)
    assert report['arms']['message3.m3.K-s101']['serial_wall_hours'] == pytest.approx(40 / 3600)

# Final verification review: portable producer imports and native-owner metadata adapters.

def native_m2_source(tmp_path):
    legacy = durable_export(tmp_path)
    doc = json.loads(Path(legacy['receipt']).read_text())
    archive = Path(legacy['archive'])
    archive.with_suffix('.tar.sha256').write_text(doc['snapshot_sha256'] + '\n')
    archive.with_suffix('.tar.manifest.sha256').write_text(doc['snapshot_manifest_sha256'] + '\n')
    return dict(kind='m2-snapshot', durable=True, durable_root=legacy['durable_root'],
                archive=legacy['archive'], model=legacy['model'], export_prefix='export', owner_receipt_member='receipt.json')


def native_m3_source(tmp_path):
    root = tmp_path / 'durable'
    run = root / 'runs' / 'K-s101-finqa'
    model = run / 'hf-step40'; model.mkdir(parents=True)
    (model / 'm.safetensors').write_bytes(b'native scientific export fixture')
    owner = run / 'export-40.json'
    sc.write_json(owner, dict(model_file_hashes={'m.safetensors': pd_contract.sha256_file(model / 'm.safetensors')}))
    sc.write_json(run / 'manifest.json', dict(files={owner.name: pd_contract.sha256_file(owner)}))
    sc.write_json(run / 'complete.json', dict(complete=True, receipt_status='PASS'))
    return dict(kind='m3-export', durable=True, durable_root=str(root), model=str(model), owner_receipt=str(owner))


@pytest.mark.parametrize('kind', ['m2', 'm3'])
def test_final_blocking4_native_adapter_cli_leaves_owner_bytes_untouched(tmp_path, kind):
    source = native_m2_source(tmp_path) if kind == 'm2' else native_m3_source(tmp_path)
    before = {p.relative_to(source['durable_root']): p.read_bytes() for p in Path(source['durable_root']).rglob('*') if p.is_file()}
    spec = tmp_path / 'owner-sources.json'; sc.write_json(spec, {'entry': source})
    out = tmp_path / 'sp'
    script = ROOT / 'kit' / 'jobs' / 'message3-sp' / 'durable_adapter.py'
    result = subprocess.run([sys.executable, str(script), '--sources', str(spec), '--out', str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0 and 'READY' in result.stdout
    models = json.loads((out / 'models.local.json').read_text())
    receipt = json.loads(Path(models['entry']['receipt']).read_text())
    assert receipt['schema'] == 'sp-durable-export-v1' and receipt['adapter'] == 'sp-native-owner-v1'
    assert receipt['retention_policy']['extends_retention'] is False
    assert str(tmp_path) not in Path(models['entry']['receipt']).read_text()
    proof = man.durable_binding(models['entry'])
    assert man.durable_binding(models['entry'], verified=proof) == proof
    sc.verify_model(dict(model_file_hashes=proof['model_file_hashes'], weights_sha256=pd_contract.digest(proof['model_file_hashes'])), source['model'])
    after = {p.relative_to(source['durable_root']): p.read_bytes() for p in Path(source['durable_root']).rglob('*') if p.is_file()}
    assert before == after
    man.adapt_owner_exports(spec, out)
    assert json.loads((out / 'models.local.json').read_text()) == models


def test_final_blocking4_plan_and_runtime_refresh_native_adapters():
    jobs = ROOT / 'kit' / 'jobs' / 'message3-sp'
    assert 'durable_adapter.py' in (jobs / 'plan.sh').read_text()
    assert 'SP_OWNER_EXPORTS' in (jobs / 'run-task.sh').read_text()
    _, _, side = sidecar_pair()
    for task in side['tasks']:
        if task['uses_gpu']:
            scripts = [Path(c[1]).name for c in task['commands']]
            assert scripts == ['durable_adapter.py', 'sp_manifest.py', 'sp_score.py'] * len(task['entries'])


def test_final_blocking4_missing_native_export_never_creates_checkpoint_or_stale_mapping(tmp_path):
    source = native_m3_source(tmp_path)
    spec = tmp_path / 'sources.json'; sc.write_json(spec, {'entry': source})
    out = tmp_path / 'sp'
    assert man.adapt_owner_exports(spec, out)['status'] == 'READY'
    Path(source['owner_receipt']).unlink()
    assert man.adapt_owner_exports(spec, out)['status'] == 'MISSING'
    assert 'entry' not in json.loads((out / 'models.local.json').read_text())
    assert not Path(source['owner_receipt']).exists() and Path(source['model']).is_dir()
    sc.write_json(spec, {})
    assert man.adapt_owner_exports(spec, out, 'entry')['status'] == 'MISSING'
    assert 'entry' not in json.loads((out / 'models.local.json').read_text())


def test_final_minor2_released_parent_gpu_needs_current_no_hold_receipt(tmp_path):
    released = [dict(complete_inventory=True, gpus=[dict(GPU, held=False, released=True, release_receipt_sha256=H)])]
    hardware = man.freeze_hardware(tmp_path / 'hardware.json', {'sp:m3-s101': GPU}, released)
    assert hardware['parent_gpu_uuids'] == [] and hardware['historical_parent_gpu_uuids'] == [GPU['uuid']]
    values = {'{sp}': tmp_path / 'sp'}
    maps = dict(hardware=hardware, parent_inventories=released, observed_parent_gpu_uuids=[GPU['uuid']])
    path = Path(man.substitute('{gpu_identity:sp:m3-s101}', values, maps))
    assert json.loads(path.read_text())['uuid'] == GPU['uuid']
    with pytest.raises(man.MissingInput, match='fresh parent occupancy'):
        man.substitute('{gpu_identity:sp:m3-s101}', values, dict(hardware=hardware))
    held = [dict(complete_inventory=True, gpus=[dict(GPU, held=True)])]
    with pytest.raises(ValueError, match='currently held'):
        man.substitute('{gpu_identity:sp:m3-s101}', values, dict(maps, parent_inventories=held))
    with pytest.raises(ValueError, match='release receipt'):
        man.freeze_hardware(tmp_path / 'bad.json', {'sp:m3-s101': GPU},
                            [dict(complete_inventory=True, gpus=[dict(GPU, held=False)])])


def test_final_minor3_snapshot_hashed_once_then_frozen_proof_checked(tmp_path, monkeypatch):
    source = native_m2_source(tmp_path)
    spec = tmp_path / 'sources.json'; sc.write_json(spec, {'entry': source})
    out = tmp_path / 'sp'; man.adapt_owner_exports(spec, out)
    local = json.loads((out / 'models.local.json').read_text())['entry']
    real = pd_contract.sha256_file
    hashes = []
    def sha(path):
        if Path(path) == Path(source['archive']): hashes.append(str(path))
        return real(path)
    monkeypatch.setattr(pd_contract, 'sha256_file', sha)
    inv = man.inventory(simple_launch(('finqa',)))
    e = next(e for e in inv['sp_entries'] if e['role'] == 'final')
    e = dict(e, entry_id='entry', weights_bound=True, weights_sha256=pd_contract.digest(man.native_owner_document(source)['model_file_hashes']))
    bound = man.bind_runtime_entry(e, inv, [], local)
    man.freeze_inventory(out / 'entries' / 'entry.json', pd_contract.seal(bound))
    assert man.substitute('{model:entry}', {'{sp}': out, '{models_file}': out / 'models.local.json'}, {}) == source['model']
    assert man.durable_binding(local, verified=bound['durable_provenance']) == bound['durable_provenance']
    technical_freeze(tmp_path / 'frozen')
    monkeypatch.setattr(sc, 'observed_hardware', lambda entry, expected: ('0', GPU))
    monkeypatch.setattr(sc, 'generate', lambda model, panel, gpu, standin: (rd.standin_responses(panel), dict(load=0., score=0.)))
    sc.panel_job(bound, source['model'], '0', out, tmp_path / 'frozen', gpu_identity=GPU, local_export=local)
    assert len(hashes) == 1
    archive = Path(source['archive']); archive.touch()
    with pytest.raises(ValueError, match='changed after entry binding'):
        man.durable_binding(local, verified=bound['durable_provenance'])
    assert len(hashes) == 1


def test_final_minor5_alias_pricing_preserves_all_planned_uses():
    launch_doc = m3_launch()
    train = copy.deepcopy(launch_doc['tasks'][1])
    train['id'] = 'M3-K-101-contract-train'
    train['commands'] = [cmd('m3_score', action='alias', arm='K', stage='contract_nli', seed=101, step=0, panel_set='finqa')]
    launch_doc['tasks'].append(train)
    inv = man.inventory(launch_doc)
    assert len(inv['planned']) == 4 and len(inv['sp_entries']) == inv['price']['entries'] == 2
    assert sorted(len(e['uses']) for e in inv['sp_entries']) == [2, 2]
    assert {p for e in inv['sp_entries'] for p in e['planned_ids']} == {p['planned_id'] for p in inv['planned']}
    assert inv['price']['sp_b_gpu_hours'] == (1.5, 3.0)


def test_final_minor6_partial_identity_does_not_wedge_same_weight_retry(tmp_path, monkeypatch):
    technical_freeze(tmp_path / 'frozen')
    entry = dict(entry_id='retry', weights_sha256=H, assignment='sp:m3-s101', uses=[])
    root = tmp_path / 'panel' / 'retry'
    # Reproduce the exact crash window: identity exists with no manifest, even invalid bytes.
    root.mkdir(parents=True); (root / 'identity.json').write_text('{partial')
    other = tmp_path / 'panel' / 'older'; other.mkdir()
    (other / 'identity.json').write_text('{also partial')
    manifest = sc.panel_job(entry, None, '0', tmp_path, tmp_path / 'frozen', standin=True, gpu_identity=GPU)
    assert 'identity.json' in manifest['files'] and sc.verify_outputs(root) == manifest


def test_final_blocking4_parallel_assignment_adapters_preserve_local_map(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    source = native_m3_source(tmp_path)
    spec = tmp_path / 'sources.json'
    entries = ['entry-a', 'entry-b', 'entry-c']
    sc.write_json(spec, {entry: source for entry in entries})
    out = tmp_path / 'sp'
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda entry: man.adapt_owner_exports(spec, out, entry), entries))
    assert all(result['status'] == 'READY' for result in results)
    assert set(json.loads((out / 'models.local.json').read_text())) == set(entries)


def test_json_line_readers_preserve_unicode_separators_and_full_panel(tmp_path, monkeypatch):
    """Screen S and SP retain U+2028/U+2029/U+0085 inside one JSON record."""
    from kit import s_common
    response = 'one' + chr(0x2028) + 'two' + chr(0x2029) + 'three' + chr(0x85) + 'four'
    path = tmp_path / 'screen.jsonl'
    path.write_text(json.dumps({'response': response}, ensure_ascii=False) + '\n')
    assert s_common.rows(path) == [{'response': response}]
    technical_freeze(tmp_path / 'frozen')
    entry = dict(entry_id='unicode', checkpoint_id='w0', weights_sha256=H, assignment='sp:core-s101',
                 uses=[dict(family='core', seed=101, role='incoming', step=0)])
    def generated(model, panel, gpu, standin):
        raw = rd.standin_responses(panel)
        raw[0]['response'] = response
        return raw, dict(load=0., score=0.)
    monkeypatch.setattr(sc, 'generate', generated)
    sc.panel_job(entry, None, '0', tmp_path, tmp_path / 'frozen', standin=True, gpu_identity=GPU)
    cells = rd.read_sidecar(tmp_path)
    assert len(cells) == 1 and len(cells[0]['scored']) == len(cells[0]['raw']) == 400
    assert cells[0]['raw'][0]['response'] == response
