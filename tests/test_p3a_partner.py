"""Plain jobs on CPU fixtures: ordering, recovery, input identity and readout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from kit import p3a_common as c, p3a_inputs as inputs, p3a_read
from kit.beds import code


def fixture(root):
    prepared = root / 'prepared'; prepared.mkdir(parents=True)
    candidates = {}; tests = [{'input': '1\n', 'output': '2\n'}]
    for candidate in c.s.ORDER:
        sample, payloads = [], []
        for family, count in c.s.QUOTAS[candidate].items():
            for index in range(count):
                ident = candidate + '-' + family + '-' + str(index)
                gold = '1/2'
                if candidate == 'code':
                    gold = json.dumps(dict(problem_id=ident, tests_sha256=code.tests_digest(tests)), sort_keys=True)
                    payloads.append(dict(problem_id=ident, tests_sha256=code.tests_digest(tests), tests=tests, testtype='stdin', fn_name=''))
                sample.append(dict(id=ident, task=candidate, split='train', family=family, gold=gold,
                    prompt='CPU fixture ' + ident + ': compute the successor / half.', technical_synthetic=True))
        folder = prepared / candidate; folder.mkdir()
        for name, rows in [('sample', sample), ('train', sample), ('heldout', [dict(r, id=r['id']+'-heldout') for r in sample])]:
            c.t.write_jsonl(folder / (name + '.jsonl'), rows)
        if payloads: c.t.write_jsonl(folder / 'tests.jsonl', payloads)
        candidates[candidate] = dict(files={p.name: c.s.file_hash(p) for p in folder.iterdir()})
    manifest = dict(order=c.s.ORDER, models=c.s.MODELS, candidates=candidates, implementation={},
        preregistration_sha256=c.s.file_hash(c.RESOURCES / 'preregistration.md'), technical_synthetic=True)
    c.t.atomic_json(prepared / 'frozen.json', manifest)
    c.t.atomic_json(root / 'frozen-annex.json', dict(manifest=manifest, manifest_sha256=c.s.digest(manifest)))
    c.t.atomic_json(root / 'code-step1.json', dict(candidate='code', step=1, status='PASS', complete=True,
        freeze_sha256=c.s.digest(manifest), technical_synthetic=True))
    return root


def run(script, out, *args, expected=0, warnings=False):
    result = subprocess.run([sys.executable, str(c.KIT / script), '--out', str(out), '--standin', *map(str,args)], capture_output=True, text=True,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1'})
    assert result.returncode == expected, result.stdout + result.stderr
    assert warnings or 'WARNING: ' not in result.stdout, result.stdout
    return result.stdout


@pytest.mark.parametrize('scenario,selected,headroom', [('lcb-pass','code',None), ('math-runs','math_screen','PASS'), ('math-stop',None,'FAIL')])
def test_jobs_and_reruns(tmp_path, scenario, selected, headroom):
    out = tmp_path / 'out'; source = fixture(tmp_path / 'fixture')
    run('p3a_inputs.py', out, '--sources', source)
    assert 'SKIP' in run('p3a_inputs.py', out, '--sources', source)
    for lane in ('base','r1','r2'):
        run('p3a_3a.py', out, '--lane', lane)
        assert run('p3a_3a.py', out, '--lane', lane).count('SKIP:') == 4
    for name in ('teacher','rewrite'):
        args = ['--scenario', scenario]
        run('p3a_' + name + '.py', out, *args)
        raw = next((out / c.OUTPUTS / 'code' / name).glob('*raw*.jsonl'))
        before = raw.read_bytes()
        # Crash after journal persistence, before a completion marker: replay, no new requests.
        (out / c.OUTPUTS / 'code' / (name + '-complete.json')).unlink()
        run('p3a_' + name + '.py', out, *args)
        assert raw.read_bytes() == before
        assert 'SKIP' in run('p3a_' + name + '.py', out, *args)
    run('p3a_decide.py', out)
    run('p3a_baseline.py', out, '--scenario', scenario)
    if headroom:
        assert 'SKIP' in run('p3a_baseline.py', out, '--scenario', scenario)
        assert json.loads((out / c.INPUTS / 'math_screen-step1.json').read_text())['status'] == headroom
        for name in ('teacher','rewrite'):
            run('p3a_' + name + '.py', out, '--candidate', 'math_screen', '--scenario', scenario)
    else: assert not (out / c.INPUTS / 'math_screen-step1.json').exists()
    run('p3a_decide.py', out)
    (out / 'weights').mkdir(); (out / 'weights/config.json').write_text('{}')
    (out / 'cache').mkdir(); (out / 'cache/private.json').write_text('{}')
    archive = tmp_path / 'return.tar.gz'
    run('p3a_collect.py', out, '--archive', archive)
    run('p3a_collect.py', out, '--archive', archive)
    direct = p3a_read.read(out, tmp_path / 'direct', repeats=20)
    returned = p3a_read.read(archive, tmp_path / 'returned', repeats=20)
    assert direct == returned and direct['three_a_complete']
    assert direct['screen_pass_candidate'] == selected
    assert direct['selected_for_partner_qualification'] is None
    assert (tmp_path / 'returned/3a/table.md').is_file()
    assert all((tmp_path / 'returned/3c' / name / 'admission-table.md').is_file() for name in c.s.ORDER)
    with p3a_read.Archive(archive) as packed:
        assert not any(n.startswith(('cache/','weights/')) or n.endswith('/tests.jsonl') for n in packed.files)
    result = subprocess.run([sys.executable, str(c.KIT / 'p3a_read.py'), '--archive', str(archive), '--out', str(tmp_path / 'cli')], capture_output=True)
    assert result.returncode == 0 and (tmp_path / 'cli/logs/read.log').exists()
    assert len(list((out / 'logs').glob('*.log'))) >= 8


def test_incomplete_never_advances_and_wrong_hash_stops(tmp_path):
    out = tmp_path / 'out'; source = fixture(tmp_path / 'fixture')
    run('p3a_inputs.py', out, '--sources', source)
    assert 'NOT READY' in run('p3a_rewrite.py', out, expected=3)
    run('p3a_decide.py', out)
    assert 'NOT READY' in run('p3a_baseline.py', out, expected=3)
    assert not (out / c.INPUTS / 'math_screen-step1.json').exists()
    (out / c.INPUTS / 'prepared/code/sample.jsonl').write_text('wrong input')
    result = subprocess.run([sys.executable, str(c.KIT / 'p3a_teacher.py'), '--out', str(out), '--standin'], capture_output=True, text=True)
    assert result.returncode == 2 and 'input hash differs' in result.stdout


def test_source_hash_stop_and_missing_model(tmp_path):
    source = tmp_path / 'test5.jsonl'; source.write_text('wrong')
    with pytest.raises(c.CorrectnessStop): inputs.source_identity(tmp_path, {'test5.jsonl': dict(bytes=5,sha256='0'*64)})
    result = subprocess.run([sys.executable, str(c.KIT / 'p3a_3a.py'), '--out', str(tmp_path / 'out'), '--lane','base','--model',str(tmp_path / 'missing')], capture_output=True,text=True)
    assert result.returncode == 2 and 'missing model path' in result.stdout
    assert c.registration()['manifest_sha256'] == c.EXPECTED


def test_wrong_annex_stops(tmp_path):
    out = tmp_path / 'out'
    run('p3a_inputs.py', out, '--sources', fixture(tmp_path / 'fixture'))
    (out / c.INPUTS / 'frozen-annex.json').write_text('{malformed')
    result = run('p3a_decide.py', out, expected=2)
    assert 'STOP: correctness: wrong annex manifest' in result
    (out / '3a/report-inputs/manifest.json').write_text('{}')
    assert 'input hash differs' in run('p3a_3a.py', out, '--lane', 'base', expected=2)



def test_a_crash_propagates_instead_of_reporting_success(tmp_path, monkeypatch):
    # A plain job must never exit 0 after an unexpected error.
    def crash(a): raise OSError('disk gone')
    monkeypatch.setattr(sys, 'argv', ['job', '--out', str(tmp_path / 'out')])
    with pytest.raises(OSError, match='disk gone'):
        c.entry(crash, lambda p: None, 'crash')
    assert (tmp_path / 'out/logs/crash.log').exists()


def test_interrupted_promotion_and_truncated_math_score_recover_on_rerun(tmp_path):
    out = tmp_path / 'out'; source = fixture(tmp_path / 'fixture')
    run('p3a_inputs.py', out, '--sources', source)
    # Crash between promotion and the last copies: the rerun finishes them.
    for name in ('code-step1.json', 'preregistration.md'): (out / c.INPUTS / name).unlink()
    assert 'SKIP' in run('p3a_inputs.py', out, '--sources', source)
    assert all((out / c.INPUTS / name).is_file() for name in ('code-step1.json', 'preregistration.md'))
    for name in ('teacher', 'rewrite'): run('p3a_' + name + '.py', out, '--scenario', 'math-runs')
    run('p3a_decide.py', out)
    run('p3a_baseline.py', out, '--scenario', 'math-runs')
    score = out / c.OUTPUTS / 'math_screen/step1/baseline/bed-score.json'
    score.write_text(score.read_text()[:20])  # evaluator killed mid-write
    log = run('p3a_baseline.py', out, '--scenario', 'math-runs', warnings=True)
    assert 'SKIP' not in log and 'retained unfinished output' in log
    assert json.loads(score.read_text())['per_item']
