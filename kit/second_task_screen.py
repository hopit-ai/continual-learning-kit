"""3C data freeze, scorer audit and decisions. Imports no GPU/Modal runtime.

Reuses the v4 durable teacher journal and Spider-v2's complete-evidence decision
pattern, without its historical campaign-specific approval/quality machinery.
"""
from collections import Counter
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

from kit import v4_teacher as t
from kit.beds import code, math_screen as maths

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / 'docs/phase3/3c-second-task-screen-preregistration.md'
ORDER = ['code', 'math_screen']
LCB_HASHES = {
    'test5.jsonl': '7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214',
    'test6.jsonl': 'bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5',
}
MODELS = {
    'teacher': ('Qwen/Qwen3.6-27B', '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'),
    'student': ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'),
}
QUOTAS = {'code': {'easy': 34, 'medium': 33, 'hard': 33},
          'math_screen': {str(i): 20 for i in range(1, 6)}}


def digest(value):
    return t.sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select(rows, quotas, salt):
    result = []
    for level, n in quotas.items():
        group = sorted((r for r in rows if r['family'] == level),
                       key=lambda r: (t.sha(salt + '|' + r['id']), r['id']))
        if len(group) < n:
            raise ValueError(f'incomplete data: {level} has {len(group)}, needs {n}')
        result.extend(group[:n])
    return result


def code_inputs(root):
    root = Path(root)
    actual = {p.name: file_hash(p) for p in code.source_files(root)}
    if actual != LCB_HASHES:
        raise ValueError('LCB screen requires exactly the two registered source files')
    pool, panel, tests = [], [], []
    for split, output in [('train', pool), ('heldout', panel)]:
        for m in code.iter_members(root, split):
            indices = list(range(len(m['tests'])))
            output.append({'id': m['id'], 'task': 'code', 'split': split,
                           'family': m['difficulty'], 'platform': m['platform'],
                           'prompt': m['prompt'], 'gold': code.ground_truth_for(m, indices, m['tests']),
                           'source_tests_sha256': m['tests_sha256']})
            tests.append(code.tests_row(m, indices, m['tests']))
    if len(pool) != 223 or len(panel) != 51:
        raise ValueError('LCB registered population changed')
    return pool, panel, tests, {'files': actual, 'split_sha256': file_hash(code.SPLIT_FILE)}


def math_inputs(root):
    """Original MATH/{train,test}/subject/id.json, preserving original IDs/splits."""
    root = Path(root)
    sources, exclusions, candidates = {}, [], {'train': [], 'test': []}
    seen = {'train': set(), 'test': set()}
    paths = sorted(root.glob('*/*/*.json'))
    counts = Counter(p.relative_to(root).parts[0] for p in paths)
    if counts != {'train': 7500, 'test': 5000}:
        raise ValueError('incomplete MATH: expected original train7500/test5000 JSON tree')
    test_text = {code.normalise(json.loads(p.read_text())['problem']) for p in paths
                 if p.relative_to(root).parts[0] == 'test'}
    for path in paths:
        ident = path.relative_to(root).as_posix()
        split = ident.split('/')[0]
        sources[ident] = file_hash(path)
        row = json.loads(path.read_text())
        problem, solution = row['problem'], row['solution']
        key = code.normalise(problem)
        reason = None
        if key in seen[split]: reason = 'duplicate_question'
        elif split == 'train' and key in test_text: reason = 'test_question_overlap'
        elif '[asy]' in problem.lower(): reason = 'diagram'
        elif row['level'] not in [f'Level {i}' for i in range(1, 6)]: reason = 'unknown_level'
        seen[split].add(key)
        try:
            box = maths.last_box(solution)
            if box is None: raise ValueError('no gold box')
            gold = str(maths.rational(box[0]))
        except ValueError:
            reason = reason or 'non_rational_gold'
        if reason:
            exclusions.append({'id': ident, 'reason': reason})
            continue
        candidates[split].append({'id': ident, 'task': 'math_screen',
            'split': 'heldout' if split == 'test' else 'train',
            'family': row['level'][-1], 'subject': row['type'], 'gold': gold,
            'prompt': maths.PROMPT.format(problem=problem)})
    pool = select(candidates['train'], {str(i): 100 for i in range(1, 6)}, '3c-v1-math-pool')
    panel = select(candidates['test'], QUOTAS['math_screen'], '3c-v1-math-panel')
    return pool, panel, [], {'files': sources, 'exclusions': exclusions,
        'eligible_counts': {s: dict(Counter(r['family'] for r in rows)) for s, rows in candidates.items()}}


def prepare_candidate(candidate, out, inputs):
    pool, panel, tests, source = inputs
    salt = '3c-v1-code-screen' if candidate == 'code' else '3c-v1-math-screen'
    sample = select(pool, QUOTAS[candidate], salt)
    out = Path(out) / candidate
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [('train', pool), ('heldout', panel), ('sample', sample)]:
        t.write_jsonl(out / (name + '.jsonl'), rows)
    if tests:
        t.write_jsonl(out / 'tests.jsonl', tests)
    t.atomic_json(out / 'source.json', source)
    metadata = {}
    for name, rows in [('pool', pool), ('panel', panel), ('sample', sample)]:
        metadata[name] = {'n': len(rows), 'ids': [r['id'] for r in rows],
            'sha256': digest(rows), 'ids_sha256': digest([r['id'] for r in rows]),
            'strata': dict(Counter(r['family'] for r in rows)),
            'members': [{'id': r['id'], 'prompt_sha256': t.sha(r['prompt']),
                         'gold_sha256': t.sha(r['gold'])} for r in rows]}
    metadata['source_sha256'] = digest(source)
    metadata['files'] = {p.name: file_hash(p) for p in sorted(out.iterdir())}
    return metadata


def math_equivalence_audit(rows):
    checks = []
    for row in rows:
        gold = maths.rational(row['gold'])
        equivalent = rf'\frac{{{2 * gold.numerator}}}{{{2 * gold.denominator}}}'
        good = maths.compute_score('', r'Working. \boxed{' + equivalent + '}', row['gold'])['acc'] == 1
        bad = maths.compute_score('', r'Working. \boxed{' + str(gold + 1) + '}', row['gold'])['acc'] == 0
        checks.append({'id': row['id'], 'scaled_fraction_equivalent': good, 'shifted_value_rejected': bad})
    return {'n': len(rows), 'all_correct': all(r['scaled_fraction_equivalent'] and r['shifted_value_rejected'] for r in checks),
            'rows': checks, 'scope': 'selected scalar rational gold answers; not general symbolic equivalence'}


def freeze(lcb_root, math_root, out, annex):
    """All real datasets are bound together before either inference step."""
    out = Path(out)
    if out.exists(): raise ValueError('freeze output must be a fresh directory')
    if Path(annex).exists(): raise ValueError('annex already exists; do not overwrite a freeze')
    # Read both before writing anything: missing MATH cannot masquerade as a freeze.
    inputs = {'code': code_inputs(lcb_root), 'math_screen': math_inputs(math_root)}
    manifest = {'schema': '3c-second-task-screen.v1', 'order': ORDER,
                'preregistration_sha256': file_hash(PREREG), 'models': MODELS,
                'candidates': {c: prepare_candidate(c, out, inputs[c]) for c in ORDER}}
    for candidate in ORDER:
        base = out / candidate
        rows = t.read_jsonl(base / 'sample.jsonl') + t.read_jsonl(base / 'heldout.jsonl')
        audit = code_adequacy(rows, base / 'tests.jsonl') if candidate == 'code' else math_equivalence_audit(
            t.read_jsonl(base / 'train.jsonl') + t.read_jsonl(base / 'heldout.jsonl'))
        if candidate == 'math_screen' and not audit['all_correct']:
            raise ValueError('MATH scoring validation incomplete; fix before inference')
        t.atomic_json(base / 'scoring-audit.json', audit)
        manifest['candidates'][candidate]['files']['scoring-audit.json'] = file_hash(base / 'scoring-audit.json')
    manifest['implementation'] = {p: file_hash(ROOT / p) for p in (
        'kit/second_task_screen.py', 'kit/beds/math_screen.py', 'kit/beds/code.py',
        'kit/sandbox.py', 'kit/eval_bed.py', 'kit/v4_teacher.py',
        'scripts/second_task_screen_modal.py')}
    t.atomic_json(out / 'frozen.json', manifest)
    t.atomic_json(annex, {'manifest': manifest, 'manifest_sha256': digest(manifest)})
    return manifest


def load_frozen(root, annex):
    root = Path(root)
    seal = json.loads(Path(annex).read_text())
    manifest = json.loads((root / 'frozen.json').read_text())
    if manifest != seal['manifest'] or digest(manifest) != seal['manifest_sha256']:
        raise ValueError('frozen manifest mismatch')
    if manifest['order'] != ORDER or manifest['preregistration_sha256'] != file_hash(PREREG):
        raise ValueError('registered design changed')
    for name, expected in manifest['implementation'].items():
        if file_hash(ROOT / name) != expected: raise ValueError('implementation changed: ' + name)
    for candidate, meta in manifest['candidates'].items():
        for name, expected in meta['files'].items():
            if file_hash(root / candidate / name) != expected:
                raise ValueError('frozen data changed: ' + candidate + '/' + name)
    return manifest


def headroom(panel, records):
    ids = [r['id'] for r in panel]
    keys = [r.get('id') for r in records]
    complete = (len(ids) == len(set(ids)) and len(keys) == len(set(keys))
                and set(keys) == set(ids) and all(type(r.get('correct')) is bool for r in records))
    correct = sum(r.get('correct') is True for r in records)
    return {'status': ('PASS' if 10 * correct <= 9 * len(ids) else 'FAIL') if complete else 'INCOMPLETE',
            'correct': correct, 'n': len(ids), 'complete': complete}


def admission(sample, demonstrations, rewrites, *, complete, synthetic=False, pool=None):
    from kit.v4_contract import technical_synthetic
    synthetic = synthetic or technical_synthetic(sample) or technical_synthetic(demonstrations) or technical_synthetic(rewrites)
    teacher = t.verified_lookup(sample, demonstrations)
    student = t.verified_lookup(sample, rewrites)
    covered = set(teacher) & set(student)
    counts = Counter(r['family'] for r in sample)
    strata = {s: {'n': n, 'successes': sum(r['id'] in covered for r in sample if r['family'] == s),
                  'required': math.ceil(.7 * n) if n >= 10 else None} for s, n in counts.items()}
    passed = len(covered) >= 85 and all(v['required'] is None or v['successes'] >= v['required'] for v in strata.values())
    complete = complete and len(sample) == len({r['id'] for r in sample}) == 100
    descriptive = {}
    for field in ('platform', 'subject'):
        groups = Counter(r[field] for r in sample if field in r)
        if groups:
            descriptive[field] = {key: {'n': n, 'successes': sum(
                r['id'] in covered for r in sample if r.get(field) == key)} for key, n in groups.items()}
    if pool:
        weights = Counter(r['family'] for r in pool)
        descriptive['pool_weighted_coverage'] = sum(
            weights[level] * row['successes'] / row['n'] for level, row in strata.items()) / len(pool)
    return {'status': ('PASS' if passed else 'FAIL') if complete else 'INCOMPLETE',
            'complete': complete, 'intersection': len(covered), 'n': len(sample), 'strata': strata,
            'teacher_successes': len(teacher), 'rewrite_attrition': len(teacher) - len(covered),
            'descriptive': descriptive,
            'technical_synthetic': synthetic, 'selected_for_partner_qualification': complete and passed and not synthetic,
            'full_pool_qualified': False, 'acquisition_demonstrated': False}


def verify(item, raw, tokenizer, *, mapped_target=False, tests=None):
    text, reasons = raw['text'], []
    if raw.get('finish_reason') not in ('stop', 'eos'): reasons.append('unfinished_generation')
    if raw.get('input_over_cap'): reasons.append('generation_input_over_cap')
    if re.search(r'</?think\b', text, re.I): reasons.append('thinking_content_present')
    if item['task'] == 'code':
        result = code.compute_score(code.DATA_SOURCE, text, item['gold'], tests_path=tests)
    else:
        result = maths.compute_score(maths.DATA_SOURCE, text, item['gold'])
        box = maths.last_box(text)
        if box is None or not text[:box[1]].strip(): reasons.append('missing_working')
    if result['score'] != 1: reasons.append('incorrect_answer')
    prompt_ids, target_ids = t.tokenize_student(tokenizer, t.student_messages(item), text)
    if len(target_ids) > t.RESPONSE_CAP: reasons.append('student_response_over_2048')
    if len(prompt_ids) > t.INPUT_CAP: reasons.append('student_input_over_4096')
    return {'id': item['id'], 'task': item['task'], 'family': item['family'],
            'attempt': raw['attempt'], 'text': text, 'verified': not reasons,
            'reasons': reasons, 'primary_reason': reasons[0] if reasons else None,
            'tokens_8b': len(target_ids), 'verifier_detail': result}


def run_phase(sample, out, generator, tokenizer, *, demonstrations=None, synthetic=False, tests=None):
    from functools import partial
    manifest = {'screen': '3c', 'models': MODELS, 'stand_in_teacher': synthetic,
                'technical_synthetic': synthetic, 'scientific_phase_allowed': False}
    # Durable raw attempt journal, first-acceptable selection, no adaptive retries.
    return t._run_pool(sample, out, generator, tokenizer, manifest, 4,
        demonstrations=demonstrations, verify_fn=partial(verify, tests=tests),
        teacher_messages_fn=t.student_messages)


def code_adequacy(rows, tests_path):
    """Descriptive constant-mutant audit on every selected problem; no new gate."""
    tests = code.TestsFile(tests_path)
    report = []
    for row in rows:
        test = tests.get(row['id'])
        first = test['tests'][0]['output']
        if test['testtype'] == 'stdin':
            program = 'import sys\nsys.stdout.write(' + repr(str(first)) + ')'
        else:
            value = json.loads(first)
            program = f"def {test['fn_name']}(*args):\n    return {value!r}"
        verdict = code.score_response('```python\n' + program + '\n```', test)
        report.append({'id': row['id'], 'n_tests': len(test['tests']),
            'unique_inputs': len({str(v['input']) for v in test['tests']}),
            'unique_outputs': len({str(v['output']) for v in test['tests']}),
            'testtype': test['testtype'], 'constant_mutant_survives': verdict['correct']})
    return {'descriptive_only': True, 'rows': report,
            'warnings': [r['id'] for r in report if r['constant_mutant_survives'] or r['unique_inputs'] < 2]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('freeze')
    p.add_argument('--lcb-root', required=True); p.add_argument('--math-root', required=True)
    p.add_argument('--out', required=True); p.add_argument('--annex', required=True)
    p = subs.add_parser('code-audit')
    p.add_argument('--lcb-root', required=True); p.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        print(json.dumps(freeze(args.lcb_root, args.math_root, args.out, args.annex), indent=2))
    else:
        meta = prepare_candidate('code', args.out, code_inputs(args.lcb_root))
        base = Path(args.out) / 'code'
        rows = t.read_jsonl(base / 'sample.jsonl') + t.read_jsonl(base / 'heldout.jsonl')
        t.atomic_json(base / 'adequacy.json', code_adequacy(rows, base / 'tests.jsonl'))
        t.atomic_json(Path(args.out) / 'code-manifest.json', meta)


if __name__ == '__main__':
    main()
