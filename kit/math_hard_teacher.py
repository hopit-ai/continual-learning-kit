#!/usr/bin/env python3
"""Hard-MATH teacher coverage: the pinned 27B on the frozen 200-item Level-5 screen panel, then the frozen decision.

Registration: the frozen docs/phase3/hard-math-screen-preregistration-20261020.md (sha256 91bf4f3d..., the hash the data
manifest records) and its teacher-coverage amendment docs/phase3/hard-math-screen-amendment-1-20261020.md, copied byte for
byte to kit/resources/math-hard-teacher/registration.md and amendment-1.md and bound separately. The protocol is the 3C step-2 teacher step, reused by import:
kit.v4_teacher._run_pool (durable raw journal, four registered attempts, first verified), kit.v4_teacher.VLLMGenerator
(vLLM 0.18.0, TP2, eager, batch-invariant) and kit.second_task_screen.verify (the 3C rational-answer verifier and the
frozen-8B target limits). Nothing here changes the panel, the student result or the scorer.

    "$INFERENCE_PY" "$KIT/math_hard_teacher.py" --out "$OUT" --model "$TEACHER" --tokenizer "$BASE" --gpu 6,7
    python "$KIT/math_hard_teacher.py" --out /tmp/mht-standin --standin      # CPU plumbing, never scientific
    python "$KIT/math_hard_teacher_read.py" --source "$OUT/math-hard-teacher"

Resumable: journalled attempts are never regenerated; a finished run prints SKIP; changed inputs stop (exit 2).
"""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_common import *
from collections import Counter
from contextlib import contextmanager
import fcntl
from fractions import Fraction
from functools import partial
import platform
import time
from kit.beds import math_screen as maths

M = KIT / 'resources/math-hard-teacher'
DIR = 'math-hard-teacher'
SCHEMA = 'math-hard-teacher-coverage.v1'
DATA_MANIFEST_SHA256 = 'ee2417d5b1b772e994b80b70ea8866b5f49e79e70d241559e96145c55e1a0b8f'
PANEL_ROWS_SHA256 = '7235a1bd0fd1c24c7d8ec4d65ddd10a9e43bf82bb46e5e3878ce5e4188c239b6'
REGISTRATION_SHA256 = '91bf4f3d1632a55efcb9234e0144242dc80771e956a7fb820e0a98910e3dd7d4'   # frozen; data manifest binds it
N = 200
STUDENT_CORRECT = 138
ACCURACY_BAR = Fraction(8, 10)       # at least 160 of 200, decided on the integer count
GAP_BAR = 30                         # of the 62 frozen-8B failures
ATTEMPTS = 4
IMPLEMENTATION = ('math_hard_teacher.py', 'second_task_screen.py', 'v4_teacher.py', 'beds/math_screen.py', 'p3b_common.py', 'p3a_common.py')
MODELS = {'teacher': list(s.MODELS['teacher']), 'student_tokenizer': list(s.MODELS['student'])}
PROTOCOL = {'source': '3C step-2 teacher step (kit/p3a_jobs.py phase "teacher")', 'schedule': [list(x) for x in t.SCHEDULE],
            'attempts': ATTEMPTS, 'top_p': 1, 'first_verified': True, 'teacher_new_tokens': t.TEACHER_NEW_TOKENS,
            'teacher_input_cap': t.INPUT_CAP, 'student_response_cap_including_eos': t.RESPONSE_CAP,
            'student_input_cap': t.INPUT_CAP, 'thinking': False, 'tensor_parallel_size': 2, 'vllm': '0.18.0',
            'batch_fallback': False, 'prompt': 'panel row prompt (kit.beds.math_screen.PROMPT) as the only user message',
            'verifier': 'kit.second_task_screen.verify, math_screen'}
RULE = {'teacher_solved': 'verified demonstration within the four registered attempts',
        'qualifies_if': 'teacher solved >= 160 of 200 (80%) AND teachable gap >= 30 of the 62 frozen-8B failures',
        'teachable_gap': 'items wrong for the frozen 8B and solved by the teacher',
        'otherwise': 'DOES_NOT_QUALIFY; a hardest-types variant only as a separate declared decision on this screen panel; '
                     'the 100-item evaluation set and the 500-item training pool stay untouched',
        'incomplete': 'missing attempts, duplicate or foreign records, or a demonstration the verifier rejects on re-reading'}
SCENARIOS = {  # stand-in only: (student failures solved, student successes solved)
    'qualifies': (40, 130), 'boundary': (30, 130), 'thin-gap': (25, 138), 'low-accuracy': (40, 110)}


def plain(value):
    return json.loads(json.dumps(value))


@contextmanager
def lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise CorrectnessStop('another writer holds ' + str(root))
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def frozen():
    """Panel, student verdicts and reused protocol code, all by hash, before anything is generated."""
    sums = {}
    for line in (M / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1); check(KIT / name, digest); sums[name] = digest
    for name in ('screen_panel.jsonl', 'data-manifest.json', 'student-bed-score.json', 'student-decision.json',
                 'student-responses.jsonl', 'registration.md', 'amendment-1.md', 'beds/math_screen.py'):
        require(any(n.endswith(name) for n in sums), 'SHA256SUMS does not list ' + name)
    check(M / 'data-manifest.json', DATA_MANIFEST_SHA256)
    manifest = json.loads((M / 'data-manifest.json').read_text())
    check(M / 'registration.md', REGISTRATION_SHA256)
    require(manifest['preregistration_sha256'] == REGISTRATION_SHA256, 'data manifest binds a different registration')
    require('## Amendment 1 (20 October 2026)' in (M / 'amendment-1.md').read_text(), 'amendment 1 missing')
    check(M / 'screen_panel.jsonl', manifest['files']['screen_panel.jsonl'])
    panel = t.read_jsonl(M / 'screen_panel.jsonl')
    ids = [r['id'] for r in panel]
    require(len(panel) == N == len(set(ids)) and s.digest(panel) == manifest['sets']['screen']['sha256'] == PANEL_ROWS_SHA256,
            'screen panel differs from the frozen hard-MATH manifest')
    require(all(r['task'] == 'math_screen' and r['family'] == '5' for r in panel), 'panel is not the Level-5 math_screen panel')
    score = json.loads((M / 'student-bed-score.json').read_text())
    decision = json.loads((M / 'student-decision.json').read_text())
    require(decision.get('manifest_sha256') == DATA_MANIFEST_SHA256 and decision.get('panel_sha256') == PANEL_ROWS_SHA256
            and decision.get('status') == 'QUALIFIES' and decision.get('correct') == STUDENT_CORRECT
            and decision.get('technical_synthetic') is False, 'student decision is not the frozen 8B result')
    require(score['scorer_files']['kit/beds/math_screen.py'] == s.file_hash(KIT / 'beds/math_screen.py'),
            'verifier differs from the one that scored the frozen 8B')
    require(score['responses_sha256'] == s.file_hash(M / 'student-responses.jsonl'), 'student responses differ from bed-score')
    require(all(v in (0, 1) and type(v) in (int, bool) for v in score['per_item'].values()), 'student verdicts are not 0/1')
    student = {k: v == 1 for k, v in score['per_item'].items()}
    require(set(student) == set(ids) and sum(student.values()) == STUDENT_CORRECT == score['correct'], 'student verdicts incomplete')
    gold = {r['id']: r['gold'] for r in panel}
    responses = t.read_jsonl(M / 'student-responses.jsonl')
    require(sorted(r['id'] for r in responses) == sorted(ids), 'student responses do not cover the panel')
    for row in responses:  # the frozen verifier reproduces every student verdict
        require((maths.compute_score(maths.DATA_SOURCE, row['response'], gold[row['id']])['score'] == 1) == student[row['id']],
                'frozen verifier does not reproduce the student verdict: ' + row['id'])
    hashes = {'sha256sums': s.file_hash(M / 'SHA256SUMS'), 'data_manifest': DATA_MANIFEST_SHA256, 'panel_rows': PANEL_ROWS_SHA256,
              'panel_file': manifest['files']['screen_panel.jsonl'], 'student_bed_score': s.file_hash(M / 'student-bed-score.json'),
              'student_responses': score['responses_sha256'], 'registration': REGISTRATION_SHA256,
              'amendment_1': s.file_hash(M / 'amendment-1.md')}
    return {'panel': panel, 'student': student, 'hashes': hashes}


def decide(panel, student, demonstrations, rejections, *, technical_synthetic=False, raws=()):
    """The frozen amendment-1 rule on the complete panel; everything after `status` except the counts is descriptive."""
    ids = [r['id'] for r in panel]
    pool = {r['id']: r for r in panel}
    problems = []
    solved = {}
    for d in demonstrations:
        key = d.get('id')
        if key not in pool or key in solved or d.get('verified') is not True:
            problems.append('duplicate, foreign or unverified demonstration: ' + str(key)); continue
        if maths.compute_score(maths.DATA_SOURCE, d.get('text', ''), pool[key]['gold'])['score'] != 1:
            problems.append('demonstration fails the frozen verifier on re-reading: ' + key); continue
        solved[key] = d
    tried = {}
    for r in rejections:
        key = r.get('id')
        if key not in pool or r.get('verified') is not False:
            problems.append('foreign or verified rejection: ' + str(key)); continue
        tried.setdefault(key, []).append(r.get('attempt'))
    for key in ids:
        expected = list(range(1, solved[key]['attempt'])) if key in solved else list(range(1, ATTEMPTS + 1))
        if sorted(tried.get(key, [])) != expected:
            problems.append('attempts missing or duplicated: ' + key)
    if set(student) != set(ids) or not all(type(v) is bool for v in student.values()):
        problems.append('student verdicts incomplete')
    complete = not problems and len(ids) == N
    failures = [i for i in ids if student.get(i) is False]
    gap = [i for i in failures if i in solved]
    k = len(solved)
    if not complete: status = 'INCOMPLETE'
    elif Fraction(k, N) >= ACCURACY_BAR and len(gap) >= GAP_BAR: status = 'QUALIFIES'
    else: status = 'DOES_NOT_QUALIFY'
    from kit.math_hard_screen import wilson
    answer_right = {r['id'] for r in list(rejections) + list(demonstrations)
                    if (r.get('verifier_detail') or {}).get('score') == 1}
    per_type = {}
    for subject in sorted({r['subject'] for r in panel}):
        members = [r['id'] for r in panel if r['subject'] == subject]
        per_type[subject] = {'n': len(members), 'student_correct': sum(student.get(i) is True for i in members),
                             'teacher_solved': sum(i in solved for i in members), 'teachable_gap': sum(i in gap for i in members)}
    return {'schema': SCHEMA, 'status': status, 'complete': complete, 'problems': problems[:20], 'n': len(ids),
            'teacher_solved': k, 'teacher_accuracy': round(k / len(ids), 6) if ids else None,
            'accuracy_bar': '>= 80% (160 of 200)', 'accuracy_pass': Fraction(k, N) >= ACCURACY_BAR,
            'student_failures': len(failures), 'teachable_gap': len(gap), 'gap_bar': '>= 30 of 62', 'gap_pass': len(gap) >= GAP_BAR,
            'technical_synthetic': technical_synthetic,
            'third_task_qualifies': status == 'QUALIFIES' and not technical_synthetic,
            'if_not_qualified': RULE['otherwise'],
            'descriptive': {
                'wilson95_teacher_accuracy': wilson(k, len(ids)),
                'greedy_first_attempt_solved': sum(d['attempt'] == 1 for d in solved.values()),
                'solved_by_attempt': dict(sorted(Counter(str(d['attempt']) for d in solved.values()).items())),
                'answer_correct_any_attempt': len(answer_right & set(ids)),
                'answer_correct_but_never_verified': len((answer_right & set(ids)) - set(solved)),
                'rejected_attempt_reasons': dict(sorted(Counter(x for r in rejections for x in r.get('reasons', [])).items())),
                'student_solved_teacher_not': sum(student.get(i) is True and i not in solved for i in ids),
                'finish_reasons': dict(sorted(Counter(str(r.get('finish_reason')) for r in raws).items())),
                'teacher_output_tokens': sum(int(r.get('teacher_tokens') or 0) for r in raws),
                'per_type': per_type},
            'registration_allowed': False}


def per_item(panel, student, demonstrations, rejections):
    solved = {d['id']: d for d in demonstrations}
    reasons = {}
    for r in rejections: reasons.setdefault(r['id'], []).append([r['attempt'], r.get('primary_reason')])
    return [{'id': r['id'], 'subject': r['subject'], 'student_correct': student[r['id']], 'teacher_solved': r['id'] in solved,
             'first_verified_attempt': solved[r['id']]['attempt'] if r['id'] in solved else None,
             'teachable': r['id'] in solved and not student[r['id']],
             'rejected_attempts': sorted(reasons.get(r['id'], []))} for r in panel]


class StandIn:
    """Gold-aware CPU generator for plumbing rehearsal only; outputs are labelled technical_synthetic."""
    def __init__(self, panel, student, scenario):
        wrong_n, right_n = SCENARIOS[scenario]
        order = sorted(panel, key=lambda r: t.sha('mht-standin|' + r['id']))
        wrong = [r['id'] for r in order if not student[r['id']]]
        right = [r['id'] for r in order if student[r['id']]]
        chosen = wrong[:wrong_n] + right[:right_n]
        self.first = {i: 1 + n % ATTEMPTS for n, i in enumerate(chosen)}
        self.gold = {r['id']: r['gold'] for r in panel}
        self.calls = 0
    def generate_many(self, requests):
        self.calls += 1
        out = []
        for r in requests:
            gold, first = self.gold[r['id']], self.first.get(r['id'])
            if first is not None and r['attempt'] >= first:
                text, finish = 'Work through the problem step by step.\nSo the answer is \\boxed{' + gold + '}', 'stop'
            else:
                text, finish = [('Work it out. \\boxed{' + gold + '+1}', 'stop'), ('\\boxed{' + gold + '}', 'stop'),
                                ('Working that never ends', 'length'), ('<think>x</think> so \\boxed{' + gold + '}', 'stop')][r['attempt'] - 1]
            out.append({'text': text, 'finish_reason': finish, 'teacher_tokens': len(text.split()), 'technical_synthetic': True})
        return out


class Teacher:
    """The 3C teacher engine, started only when an attempt is actually pending (a finished journal never loads it)."""
    def __init__(self, model): self.model, self.engine, self.started = model, None, None
    def generate_many(self, requests):
        if self.engine is None:
            self.started = time.monotonic()
            self.engine = t.VLLMGenerator(self.model, t.TEACHER_NEW_TOKENS, input_cap=t.INPUT_CAP,
                                          tensor_parallel_size=2, allow_batch_fallback=False)
        return self.engine.generate_many(requests)


def standin_generator(panel, student, scenario):
    return StandIn(panel, student, scenario)


def read_outputs(root):
    teacher = root / 'teacher'
    demonstrations = t.read_jsonl(teacher / 'demonstrations.jsonl') if (teacher / 'demonstrations.jsonl').exists() else []
    rejections = t.read_jsonl(teacher / 'rejections.jsonl') if (teacher / 'rejections.jsonl').exists() else []
    raws = [json.loads(p.read_text()) for p in sorted((teacher / 'raw').glob('*.json'))] if (teacher / 'raw').is_dir() else []
    return demonstrations, rejections, raws


def job(a):
    """Hard-MATH teacher coverage (pinned Qwen3.6-27B, 3C teacher protocol, two GPUs); resumable."""
    root = a.out / DIR
    with lock(root):
        f = frozen()
        if not a.standin:
            require(a.model and a.tokenizer, 'pinned teacher --model and frozen-8B --tokenizer directories required')
            require(a.gpu and len(set(a.gpu.split(','))) == 2 == len(a.gpu.split(',')), 'teacher uses exactly two distinct GPUs (TP2)')
        inputs = plain({'schema': SCHEMA, 'resources': f['hashes'], 'protocol': PROTOCOL, 'models': MODELS, 'rule': RULE,
                        'implementation': {n: s.file_hash(KIT / n) for n in IMPLEMENTATION},
                        'technical_synthetic': a.standin, 'scenario': a.scenario if a.standin else None})
        bind(root / 'inputs.json', inputs)
        if (root / 'manifest.json').exists():
            frozen_files(root)
            print('SKIP: hard-MATH teacher coverage complete')
            print(summary(json.loads((root / 'decision.json').read_text()))); return
        if a.standin:
            from kit.sim.p3a_row import Tokenizer
            tokenizer, identities = Tokenizer(), {'technical_synthetic': True}
            generator = standin_generator(f['panel'], f['student'], a.scenario)
        else:
            # Check the supplied models against the v4 registration before reusing any journalled attempt.
            identities = {'teacher': identity(a.model, 'teacher'), 'student_tokenizer': identity(a.tokenizer, 'student')}
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
            os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, KIT_DETERMINISTIC='1', V4_PHASE0_INFERENCE='1')
            generator = Teacher(model_path(a.model))
        bind(root / 'identity.json', plain(identities))
        manifest = plain({'screen': 'hard-math-teacher-coverage', 'models': MODELS, 'teacher_model': MODELS['teacher'][0],
                          'resources': f['hashes']['sha256sums'], 'stand_in_teacher': a.standin,
                          'technical_synthetic': a.standin, 'scientific_phase_allowed': False})
        journal = root / 'teacher/raw'
        before = len(list(journal.glob('*.json'))) if journal.is_dir() else 0
        clock = time.monotonic(); error = None
        try:
            t._run_pool(f['panel'], root / 'teacher', generator, tokenizer, manifest, ATTEMPTS,
                        verify_fn=partial(s.verify, tests=None), teacher_messages_fn=t.student_messages)
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc); raise
        finally:
            wall = time.monotonic() - clock
            engine = getattr(generator, 'started', None)
            atomic(root / 'costs' / f'{time.time_ns()}.json', {
                'kind': 'math-hard-teacher', 'wall_seconds': wall, 'gpus': 0 if a.standin else 2,
                'gpu_hours': 0 if a.standin else 2 * wall / 3600, 'engine_started': engine is not None,
                'new_attempts': len(list(journal.glob('*.json'))) - before if journal.is_dir() else 0,
                'host': platform.node(), 'versions': t.runtime_versions(), 'error': error, 'technical_synthetic': a.standin})
        demonstrations, rejections, raws = read_outputs(root)
        result = decide(f['panel'], f['student'], demonstrations, rejections, technical_synthetic=a.standin, raws=raws)
        costs = [json.loads(p.read_text()) for p in sorted((root / 'costs').glob('*.json'))]
        result['measured'] = {'wall_seconds': sum(c['wall_seconds'] for c in costs), 'gpu_hours': sum(c['gpu_hours'] for c in costs),
                              'invocations': len(costs), 'engine_starts': sum(c['engine_started'] for c in costs)}
        result['inputs_sha256'] = s.file_hash(root / 'inputs.json')
        atomic(root / 'decision.json', result)
        if not result['complete']:
            raise NotReady('teacher journal incomplete; rerun the same command to resume: ' + '; '.join(result['problems'][:3]))
        t.write_jsonl(root / 'per-item.jsonl', per_item(f['panel'], f['student'], demonstrations, rejections))
        names = ['inputs.json', 'identity.json', 'decision.json', 'per-item.jsonl'] + [
            str(p.relative_to(root)) for p in sorted((root / 'teacher').rglob('*')) if p.is_file() and p.name != '.lock'] + [
            str(p.relative_to(root)) for p in sorted((root / 'costs').glob('*.json'))]
        atomic(root / 'manifest.json', files_manifest(root, names, technical_synthetic=a.standin, schema=SCHEMA))
        print(summary(result))


def summary(result):
    lines = [f"hard-MATH teacher coverage: {result['status']}" + (' (technical stand-in, never scientific)' if result['technical_synthetic'] else ''),
             f"teacher solved {result['teacher_solved']}/{result['n']} ({100 * (result['teacher_accuracy'] or 0):.1f}%; bar {result['accuracy_bar']}: "
             f"{'pass' if result['accuracy_pass'] else 'fail'})",
             f"teachable gap {result['teachable_gap']} of {result['student_failures']} frozen-8B failures (bar {result['gap_bar']}: "
             f"{'pass' if result['gap_pass'] else 'fail'})",
             f"third task qualifies: {result['third_task_qualifies']}"]
    if result['status'] == 'DOES_NOT_QUALIFY': lines.append('next: ' + result['if_not_qualified'])
    if result['status'] == 'INCOMPLETE': lines.append('incomplete: ' + '; '.join(result['problems'][:3]))
    d = result['descriptive']
    lines.append(f"descriptive: greedy first attempt {d['greedy_first_attempt_solved']}, answer right but rejected {d['answer_correct_but_never_verified']}, "
                 f"student right and teacher not {d['student_solved_teacher_not']}")
    lines.append('| type | n | 8B correct | teacher solved | teachable gap |')
    lines.append('|---|---:|---:|---:|---:|')
    for k, v in d['per_type'].items():
        lines.append(f"| {k} | {v['n']} | {v['student_correct']} | {v['teacher_solved']} | {v['teachable_gap']} |")
    return '\n'.join(lines)


def configure(p):
    p.add_argument('--model', help='local pinned Qwen3.6-27B snapshot directory')
    p.add_argument('--tokenizer', help='local pinned frozen Qwen3-8B snapshot (its tokenizer sets the 3C target limits)')
    p.add_argument('--gpu', help='two distinct GPU ordinals, e.g. 6,7')
    p.add_argument('--scenario', choices=sorted(SCENARIOS), default='qualifies', help='stand-in only')


if __name__ == '__main__': raise SystemExit(entry(job, configure, 'math-hard-teacher'))
