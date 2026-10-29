"""Hard-MATH teacher coverage: frozen resources, the frozen decision rule, and the plain job in stand-in mode (no model)."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS
import pytest
from kit import math_hard_teacher as h, math_hard_teacher_read as reader
from kit.beds import math_screen as maths

ROOT = Path(__file__).resolve().parents[1]


def args(out, **kw):
    return NS(out=Path(out), standin=True, model=None, tokenizer=None, gpu=None, scenario=kw.pop('scenario', 'qualifies'), **kw)


def run(out, **kw):
    h.job(args(out, **kw))
    return json.loads((Path(out) / h.DIR / 'decision.json').read_text())


# ------------------------------------------------------------------ frozen resources
def test_frozen_resources_bind_the_panel_and_the_frozen_8b_result():
    f = h.frozen()
    assert len(f['panel']) == 200 and {r['family'] for r in f['panel']} == {'5'}
    assert sum(f['student'].values()) == 138 and sum(not v for v in f['student'].values()) == 62
    assert f['hashes']['data_manifest'].startswith('ee2417d5') and f['hashes']['panel_rows'] == h.PANEL_ROWS_SHA256
    names = [line.split(maxsplit=1)[1] for line in (h.M / 'SHA256SUMS').read_text().splitlines()]
    assert {'beds/math_screen.py', 'second_task_screen.py', 'v4_teacher.py'} <= set(names)
    assert all(n.startswith('resources/math-hard-teacher/') for n in names if n.endswith(('.jsonl', '.json', '.md')))


@pytest.mark.skipif(not (ROOT / 'data/math_hard').is_dir(), reason='research repository copy only')
def test_kit_copies_equal_the_repository_freeze_and_evidence():
    pairs = {'screen_panel.jsonl': 'data/math_hard/screen_panel.jsonl', 'data-manifest.json': 'data/math_hard/manifest.json',
             'student-bed-score.json': 'docs/phase3/evidence/math-hard-screen/student-8b/bed-score.json',
             'student-decision.json': 'docs/phase3/evidence/math-hard-screen/student-8b/decision.json',
             'student-responses.jsonl': 'docs/phase3/evidence/math-hard-screen/student-8b/responses.jsonl'}
    for kit_name, repo_name in pairs.items():
        assert (h.M / kit_name).read_bytes() == (ROOT / repo_name).read_bytes(), kit_name


def test_frozen_registration_and_amendment_are_bound_separately():
    f = h.frozen()
    data = json.loads((h.M / 'data-manifest.json').read_text())
    assert h.s.file_hash(h.M / 'registration.md') == data['preregistration_sha256'] == h.REGISTRATION_SHA256 == f['hashes']['registration']
    assert f['hashes']['amendment_1'] == h.s.file_hash(h.M / 'amendment-1.md')
    reg = (h.M / 'registration.md').read_text()
    assert 'Amendment 1' not in reg and '## Result' not in reg
    rule = (h.M / 'amendment-1.md').read_text()
    assert 'before any teacher output' in rule
    for phrase in ('at least 160 of 200', 'at least 30 of the 62', 'DOES_NOT_QUALIFY', 'INCOMPLETE',
                   '27101/27102/27103/27104', '2,560', '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9', '1,548 s'):
        assert phrase in rule, phrase


@pytest.mark.skipif(not (ROOT / 'docs/phase3').is_dir(), reason='research repository copy only')
def test_kit_copies_are_the_repository_registration_and_amendment():
    docs = ROOT / 'docs/phase3'
    assert (docs / 'hard-math-screen-preregistration-20261020.md').read_bytes() == (h.M / 'registration.md').read_bytes()
    assert (docs / 'hard-math-screen-amendment-1-20261020.md').read_bytes() == (h.M / 'amendment-1.md').read_bytes()
    result = (docs / 'hard-math-screen-result-20261020.md').read_text()
    assert 'hard-math-screen-amendment-1-20261020.md' in result and '138/200' in result
    assert 'hard-math-screen-result-20261020.md' in (h.M / 'amendment-1.md').read_text()


def test_tampered_resources_stop_before_generation(tmp_path, monkeypatch):
    """Relocation retains exact runtime-update pins; data/code tampering still fails."""
    from kit import m2_runtime
    kit = tmp_path / 'kit'
    shutil.copytree(h.M, kit / 'resources/math-hard-teacher')
    for name in ('beds/math_screen.py', 'second_task_screen.py', 'v4_teacher.py'):
        (kit / name).parent.mkdir(parents=True, exist_ok=True); shutil.copy(h.KIT / name, kit / name)
    # check() permits reviewed infrastructure updates only under its registered root.
    # A partial copied kit previously omitted both that registry and its lookup anchor.
    (kit / 'resources/m2').mkdir(parents=True)
    shutil.copy(h.KIT / 'resources/m2/runtime-updates.json', kit / 'resources/m2/runtime-updates.json')
    shutil.copy(Path(m2_runtime.__file__), kit / 'm2_runtime.py')
    monkeypatch.setattr(m2_runtime, '__file__', str(kit / 'm2_runtime.py'))
    monkeypatch.setattr(h, 'KIT', kit); monkeypatch.setattr(h, 'M', kit / 'resources/math-hard-teacher')
    assert len(h.frozen()['panel']) == 200
    implementation = kit / 'second_task_screen.py'
    original = implementation.read_bytes()
    implementation.write_bytes(original + b'\n# unregistered fixture mutation\n')
    with pytest.raises(h.CorrectnessStop, match='input hash differs'):
        h.frozen()
    implementation.write_bytes(original)
    score = kit / 'resources/math-hard-teacher/student-bed-score.json'
    doc = json.loads(score.read_text()); first = next(iter(doc['per_item']))
    doc['per_item'][first] = 1 - doc['per_item'][first]
    score.write_text(json.dumps(doc))
    with pytest.raises(h.CorrectnessStop, match='input hash differs'):
        h.frozen()
    # Even with SHA256SUMS regenerated, the frozen verifier refuses a flipped student verdict.
    sums = kit / 'resources/math-hard-teacher/SHA256SUMS'
    lines = [f"{h.s.file_hash(kit / n)}  {n}" for n in (l.split(maxsplit=1)[1] for l in sums.read_text().splitlines())]
    sums.write_text('\n'.join(lines) + '\n')
    with pytest.raises(h.CorrectnessStop):
        h.frozen()


# ------------------------------------------------------------------ frozen decision rule
def records(panel, student, solved_wrong, solved_right, *, attempt=1):
    wrong = [r for r in panel if not student[r['id']]][:solved_wrong]
    right = [r for r in panel if student[r['id']]][:solved_right]
    chosen = {r['id'] for r in wrong + right}
    demos, rejects = [], []
    for r in panel:
        if r['id'] in chosen:
            demos.append({'id': r['id'], 'attempt': attempt, 'verified': True, 'text': 'Working.\n\\boxed{' + r['gold'] + '}',
                          'verifier_detail': {'score': 1.0}})
            rejects += [{'id': r['id'], 'attempt': k, 'verified': False, 'reasons': ['incorrect_answer']} for k in range(1, attempt)]
        else:
            rejects += [{'id': r['id'], 'attempt': k, 'verified': False, 'reasons': ['incorrect_answer'], 'verifier_detail': {'score': 0.0}}
                        for k in range(1, 5)]
    return demos, rejects


@pytest.mark.parametrize('wrong,right,status', [
    (30, 130, 'QUALIFIES'), (62, 138, 'QUALIFIES'), (40, 120, 'QUALIFIES'),
    (30, 129, 'DOES_NOT_QUALIFY'), (29, 138, 'DOES_NOT_QUALIFY'), (62, 97, 'DOES_NOT_QUALIFY'), (0, 0, 'DOES_NOT_QUALIFY')])
def test_rule_is_80_percent_and_a_gap_of_30_on_integer_counts(wrong, right, status):
    f = h.frozen()
    demos, rejects = records(f['panel'], f['student'], wrong, right, attempt=2)
    result = h.decide(f['panel'], f['student'], demos, rejects)
    assert result['complete'] and result['status'] == status
    assert result['teacher_solved'] == wrong + right and result['teachable_gap'] == wrong and result['student_failures'] == 62
    assert result['third_task_qualifies'] == (status == 'QUALIFIES')
    assert h.decide(f['panel'], f['student'], demos, rejects, technical_synthetic=True)['third_task_qualifies'] is False


def test_incomplete_evidence_is_neither_outcome():
    f = h.frozen()
    demos, rejects = records(f['panel'], f['student'], 62, 138, attempt=2)
    assert h.decide(f['panel'], f['student'], demos, rejects)['status'] == 'QUALIFIES'
    cases = {'missing attempt': (demos, rejects[:-1]),
             'duplicate attempt': (demos, rejects + rejects[:1]),
             'missing item': (demos[1:], rejects),
             'duplicate': (demos + demos[:1], rejects),
             'foreign': (demos + [dict(demos[0], id='test/foreign/0.json')], rejects),
             'box rejected on re-reading': ([dict(demos[0], text='\\boxed{' + demos[0]['id'] + '}')] + demos[1:], rejects)}
    demos2, rejects2 = records(f['panel'], f['student'], 30, 130)
    cases['four attempts short'] = (demos2, rejects2[:-1])
    for name, (d, r) in cases.items():
        result = h.decide(f['panel'], f['student'], d, r)
        assert result['status'] == 'INCOMPLETE' and not result['complete'] and not result['third_task_qualifies'], name


# ------------------------------------------------------------------ plain job, stand-in
@pytest.mark.parametrize('scenario,status,solved,gap', [
    ('qualifies', 'QUALIFIES', 170, 40), ('boundary', 'QUALIFIES', 160, 30),
    ('thin-gap', 'DOES_NOT_QUALIFY', 163, 25), ('low-accuracy', 'DOES_NOT_QUALIFY', 150, 40)])
def test_standin_job_runs_the_3c_teacher_path_and_reader_agrees(tmp_path, scenario, status, solved, gap, capsys):
    result = run(tmp_path, scenario=scenario)
    assert (result['status'], result['teacher_solved'], result['teachable_gap']) == (status, solved, gap)
    assert result['technical_synthetic'] is True and result['third_task_qualifies'] is False
    root = tmp_path / h.DIR
    manifest = h.frozen_files(root)
    raws = [n for n in manifest['files'] if n.startswith('teacher/raw/')]
    first = result['descriptive']['solved_by_attempt']
    # Every unsolved item used all four attempts; solved items stopped at their first verified one.
    assert len(raws) == sum(int(k) * v for k, v in first.items()) + 4 * (200 - solved)
    reasons = result['descriptive']['rejected_attempt_reasons']
    assert {'incorrect_answer', 'missing_working', 'unfinished_generation', 'thinking_content_present'} <= set(reasons)
    rows = h.t.read_jsonl(root / 'per-item.jsonl')
    assert len(rows) == 200 and sum(r['teachable'] for r in rows) == gap
    pool = json.loads((root / 'teacher/manifest.json').read_text())
    assert pool['attempts'] == 4 and pool['stand_in_teacher'] is True
    demo = h.t.read_jsonl(root / 'teacher/demonstrations.jsonl')[0]
    assert maths.compute_score('', demo['text'], next(r['gold'] for r in h.frozen()['panel'] if r['id'] == demo['id']))['score'] == 1
    capsys.readouterr()
    assert reader.main(['--source', str(root)]) == 0
    assert status in capsys.readouterr().out
    # Identical rerun is a SKIP; a changed input stops.
    h.job(args(tmp_path, scenario=scenario))
    assert 'SKIP' in capsys.readouterr().out
    other = 'thin-gap' if scenario != 'thin-gap' else 'qualifies'
    with pytest.raises(h.CorrectnessStop, match='input identity changed'):
        h.job(args(tmp_path, scenario=other))


class Crash(h.StandIn):
    def generate_many(self, requests):
        if requests[0]['attempt'] == 2: raise RuntimeError('simulated node loss in round 2')
        return super().generate_many(requests)


def test_interrupted_job_resumes_without_regenerating_journalled_attempts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(h, 'standin_generator', lambda p, s, sc: Crash(p, s, sc))
    with pytest.raises(RuntimeError, match='node loss'):
        h.job(args(tmp_path))
    root = tmp_path / h.DIR
    journal = sorted((root / 'teacher/raw').glob('*-1.json'))
    assert len(journal) == 200 and not list((root / 'teacher/raw').glob('*-2.json'))
    stamps = {p.name: p.read_bytes() for p in journal}
    assert reader.main(['--source', str(root)]) == 0 and 'INCOMPLETE' in capsys.readouterr().out
    seen = []
    class Counting(h.StandIn):
        def generate_many(self, requests):
            seen.extend(r['attempt'] for r in requests); return super().generate_many(requests)
    monkeypatch.setattr(h, 'standin_generator', lambda p, s, sc: Counting(p, s, sc))
    resumed = run(tmp_path)
    assert 1 not in seen and {p.name: p.read_bytes() for p in sorted((root / 'teacher/raw').glob('*-1.json'))} == stamps
    fresh = run(tmp_path / 'fresh')
    keys = ('status', 'teacher_solved', 'teachable_gap', 'descriptive')
    assert {k: resumed[k] for k in keys} == {k: fresh[k] for k in keys}
    assert resumed['measured']['invocations'] == 2
    costs = [json.loads(p.read_text()) for p in sorted((root / 'costs').glob('*.json'))]
    assert costs[0]['error'].startswith('RuntimeError') and costs[1]['error'] is None


def test_reader_refuses_a_written_decision_that_differs_from_its_evidence(tmp_path):
    run(tmp_path)
    root = tmp_path / h.DIR
    decision = json.loads((root / 'decision.json').read_text())
    decision['teachable_gap'] = 62
    (root / 'decision.json').write_text(json.dumps(decision))
    assert reader.main(['--source', str(root)]) == 2


def test_one_command_entry_points_and_exit_codes(tmp_path):
    out = tmp_path / 'out'
    done = subprocess.run([sys.executable, 'kit/math_hard_teacher.py', '--out', str(out), '--standin'],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and 'QUALIFIES' in done.stdout, done.stdout + done.stderr
    assert (out / 'logs/math-hard-teacher.log').is_file()
    done = subprocess.run([sys.executable, 'kit/math_hard_teacher_read.py', '--source', str(out / h.DIR), '--json'],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and json.loads(done.stdout.split('\nmeasured:')[0])['status'] == 'QUALIFIES'
    # A real run without the pinned models or with one GPU is a correctness stop, before any generation.
    for extra in ([], ['--model', str(tmp_path), '--tokenizer', str(tmp_path), '--gpu', '0']):
        done = subprocess.run([sys.executable, 'kit/math_hard_teacher.py', '--out', str(tmp_path / 'real')] + extra,
                              cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 2 and 'STOP' in done.stdout, done.stdout + done.stderr
        assert not (tmp_path / 'real' / h.DIR / 'teacher').exists()


def test_runbook_places_the_job_before_c_on_two_gpus():
    text = (h.KIT / 'README-message.md').read_text()
    assert 'math_hard_teacher.py' in text and 'math_hard_teacher_read.py' in text
    assert text.index('math_hard_teacher.py') < text.index('Return 1 after C')
    assert '--gpu' in text and 'two GPUs' in text


@pytest.mark.skipif(not (ROOT / 'scripts/publish_kit.py').is_file(), reason='research repository copy only')
def test_published_with_the_kit():
    sys.path.insert(0, str(ROOT / 'scripts'))
    import publish_kit
    assert 'test_math_hard_teacher.py' in publish_kit.TESTS
