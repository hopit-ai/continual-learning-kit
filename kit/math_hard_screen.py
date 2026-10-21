"""Hard-MATH headroom screen: frozen selection and frozen decision rule. CPU only; no model, network or Modal.

Package C amendment of 20 October (owner: "Go ahead with 1,2,3"): the connected continuation's third task is
chosen by a NEW screen of a hard MATH subset, disjoint from the 3C/tier-2 500/100 rational-answer subset, that
must show frozen-Qwen3-8B accuracy of 30-70% (and, separately, teacher coverage) before it is registered.
Registration: docs/phase3/hard-math-screen-preregistration-20261020.md.

Reused unchanged, never copied:
- eligibility (text-only, last-box exact-rational gold, exact normalised duplicates, train/test overlap) is
  `kit.second_task_screen.math_inputs`, which also re-derives the old 500/100 subset; the build refuses unless
  that re-derivation equals the 3C frozen annex (same pinned source tree, same old subset);
- the verifier and prompt are `kit.beds.math_screen` (scored through `kit/eval_bed.py`, as the 3C screen did);
- the near-duplicate rule is the registered `kit.p3b_replay_data.near` via its greedy `clean`.

    python -m kit.math_hard_screen build --math-root DIR/MATH --mirror DIR --out data/math_hard
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path

from kit import second_task_screen as s, v4_teacher as t
from kit.beds import math_screen as maths

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / 'docs/phase3/hard-math-screen-preregistration-20261020.md'
OLD_ANNEX = ROOT / 'docs/phase3/evidence/3c-second-task-screen/frozen-annex.json'
GENERAL_PANEL = ROOT / 'kit/panels/general-v1.jsonl'
DATA = ROOT / 'data/math_hard'
SCHEMA = 'math-hard-screen.v1'
SOURCE = {'dataset': 'EleutherAI/hendrycks_math', 'revision': '21a5633873b6a120296cce3e2df9d5550074f4a3',
          'ids': '<split>/<subject>/<row index> at that revision (scripts/materialise_math_mirror.py)',
          'licence': 'MIT (dataset card); rights-holder takedown of the original release disclosed in 3C'}
STUDENT = s.MODELS['student']
# Level 5 first; Levels 4-5 only if Level 5 alone cannot fill every set after all exclusions.
LEVEL_SETS = (('5',), ('4', '5'))
SIZES = {'screen': 200, 'eval': 100, 'pool': 500}
SALTS = {'dedup': 'hmath-v1-dedup', 'screen': 'hmath-v1-screen', 'eval': 'hmath-v1-eval', 'pool': 'hmath-v1-pool'}
STRATUM = 'subject'                  # MATH "type"; proportional quotas, largest remainder
SET_FILES = {'screen': 'screen_panel.jsonl', 'eval': 'eval.jsonl', 'pool': 'train_pool.jsonl'}
SCORER_FILES = ('kit/beds/math_screen.py', 'kit/eval_bed.py', 'kit/tokens_io.py')
SELECTION_FILES = ('kit/math_hard_screen.py', 'kit/second_task_screen.py', 'kit/p3b_replay_data.py')
BAND = (Fraction(3, 10), Fraction(7, 10))   # inclusive; decided on the integer count, never a rounded float
Z95 = 1.959963984540054
GENERATION = {'model': STUDENT[0], 'revision': STUDENT[1], 'adapters': None, 'thinking': False,
              'temperature': 0.0, 'top_p': 1.0, 'seed': 0, 'max_new_tokens': 2048, 'max_model_len': 12288,
              'deterministic': True, 'scorer': 'kit/eval_bed.py generate --bed math_screen'}


class Shortfall(ValueError):
    """Too few eligible items for a registered quota: an incomplete preparation, never a substitution."""


def order_key(row, salt):
    return t.sha(salt + '|' + row['id']), row['id']


def quotas(counts, n):
    """Proportional allocation by largest remainder (exact fractions; ties by stratum name)."""
    total = sum(counts.values())
    if total < n:
        raise Shortfall(f'{total} eligible items for {n}')
    shares = {k: Fraction(n * c, total) for k, c in counts.items()}
    result = {k: math.floor(v) for k, v in shares.items()}
    for k in sorted(shares, key=lambda k: (-(shares[k] - result[k]), k))[:n - sum(result.values())]:
        result[k] += 1
    return dict(sorted(result.items()))


def select(rows, wanted, salt, field=STRATUM):
    """kit.second_task_screen.select's rule (hash order within stratum, first n), on a named stratum field."""
    chosen = []
    for key, n in wanted.items():
        group = sorted((r for r in rows if r[field] == key), key=lambda r: order_key(r, salt))
        if len(group) < n:
            raise Shortfall(f'{key} has {len(group)}, needs {n}')
        chosen.extend(group[:n])
    return chosen


PREFIX = maths.PROMPT.format(problem='')


def problem_of(row):
    if not row['prompt'].startswith(PREFIX):
        raise ValueError('prompt is not the registered math_screen prompt: ' + row['id'])
    return row['prompt'][len(PREFIX):]


def counts_by(rows, *fields):
    out = Counter('|'.join(str(r[f]) for f in fields) for r in rows)
    return dict(sorted(out.items()))


def read_source(root):
    """Every source row (with its 3C eligibility verdict) plus the re-derived 3C subset."""
    root = Path(root)
    old_pool, old_panel, _, source = s.math_inputs(root)
    excluded = {e['id']: e['reason'] for e in source['exclusions']}
    rows = []
    for path in sorted(root.glob('*/*/*.json')):
        ident = path.relative_to(root).as_posix()
        raw = json.loads(path.read_text())
        row = {'id': ident, 'split': 'heldout' if ident.startswith('test/') else 'train',
               'level': raw['level'], 'type': raw['type'], 'excluded': excluded.get(ident)}
        if row['excluded'] is None:
            # Identical to math_inputs' gold: the last balanced box, as an exact rational.
            row.update(task='math_screen', family=raw['level'][-1], subject=raw['type'],
                       gold=str(maths.rational(maths.last_box(raw['solution'])[0])),
                       prompt=maths.PROMPT.format(problem=raw['problem']))
        rows.append(row)
    return rows, old_pool, old_panel, source


def check_old_subset(old_pool, old_panel, source, annex):
    """The 3C annex binds the old subset; a different tree or old subset refuses the build."""
    frozen = json.loads(Path(annex).read_text())['manifest']['candidates']['math_screen']
    if (s.digest(source) != frozen['source_sha256'] or [r['id'] for r in old_pool] != frozen['pool']['ids']
            or [r['id'] for r in old_panel] != frozen['panel']['ids']
            or not set(frozen['sample']['ids']) <= set(frozen['pool']['ids'])):
        raise ValueError('source tree or old 500/100 subset differs from the 3C frozen annex')
    return {'annex_sha256': s.file_hash(annex), 'source_sha256': frozen['source_sha256'],
            'pool_ids_sha256': frozen['pool']['ids_sha256'], 'panel_ids_sha256': frozen['panel']['ids_sha256'],
            'n': len(old_pool) + len(old_panel)}


def prepare(rows, old_pool, old_panel, general):
    """Frozen selection for the first level set that fills every quota."""
    from kit.p3b_replay_data import clean       # the registered near-duplicate rule
    old_ids = {r['id'] for r in old_pool + old_panel}
    reference = ([{'id': r['id'], 'prompt': problem_of(r)} for r in old_pool + old_panel]
                 + [{'id': 'general-v1/' + str(g['id']), 'prompt': g['prompt']} for g in general])
    eligible = [r for r in rows if r['excluded'] is None]
    attempts = []
    for levels in LEVEL_SETS:
        names = [f'Level {x}' for x in levels]
        exclusions = [{'id': r['id'], 'reason': r['excluded']} for r in rows if r['excluded'] and r['level'] in names]
        hard = [r for r in eligible if r['family'] in levels]
        exclusions += [{'id': r['id'], 'reason': 'old_3c_subset'} for r in hard if r['id'] in old_ids]
        fresh = [r for r in hard if r['id'] not in old_ids]
        kept, stages = {}, {}
        # Held-out first (screen panel and evaluation set), then train against everything held out.
        for split, against in (('heldout', reference), ('train', None)):
            ordered = sorted((r for r in fresh if r['split'] == split), key=lambda r: order_key(r, SALTS['dedup']))
            candidates = [{'id': r['id'], 'prompt': problem_of(r)} for r in ordered]
            if against is None:
                against = reference + [{'id': r['id'], 'prompt': problem_of(r)} for r in kept['heldout']]
            survivors, near = clean(candidates, against)
            exclusions += [{'id': x['id'], 'reason': 'near_duplicate', 'overlaps': str(x['overlaps'])} for x in near]
            survived = {x['id'] for x in survivors}
            kept[split] = [r for r in ordered if r['id'] in survived]
            stages[split] = {'after_old_subset': len(ordered), 'after_near_duplicates': len(kept[split])}
        try:
            test_counts = Counter(r[STRATUM] for r in kept['heldout'])
            wanted = {'screen': quotas(test_counts, SIZES['screen']), 'eval': quotas(test_counts, SIZES['eval']),
                      'pool': quotas(Counter(r[STRATUM] for r in kept['train']), SIZES['pool'])}
            screen = select(kept['heldout'], wanted['screen'], SALTS['screen'])
            taken = {r['id'] for r in screen}
            evaluation = select([r for r in kept['heldout'] if r['id'] not in taken], wanted['eval'], SALTS['eval'])
            pool = select(kept['train'], wanted['pool'], SALTS['pool'])
        except Shortfall as exc:
            attempts.append({'levels': list(levels), 'shortfall': str(exc), 'stages': stages})
            continue
        attempts.append({'levels': list(levels), 'shortfall': None, 'stages': stages})
        return {'levels': list(levels), 'attempts': attempts, 'quotas': wanted, 'exclusions': exclusions,
                'kept_counts': {k: counts_by(v, 'family', STRATUM) for k, v in kept.items()},
                'sets': {'screen': screen, 'eval': evaluation, 'pool': pool}}
    raise Shortfall('incomplete preparation: no registered level set fills every quota: ' + json.dumps(attempts))


ROW_FIELDS = ('id', 'task', 'split', 'family', 'subject', 'gold', 'prompt')   # the 3C row schema, unchanged


def build(math_root, out, *, annex=OLD_ANNEX, general_panel=GENERAL_PANEL, mirror=None, prereg=PREREG):
    out = Path(out)
    if out.exists():
        raise ValueError('build output must be a fresh directory')
    rows, old_pool, old_panel, source = read_source(math_root)
    old = check_old_subset(old_pool, old_panel, source, annex)
    general = t.read_jsonl(general_panel)
    prepared = prepare(rows, old_pool, old_panel, general)
    sets = {k: [{f: r[f] for f in ROW_FIELDS} for r in v] for k, v in prepared['sets'].items()}
    every = [r['id'] for v in sets.values() for r in v]
    if len(every) != len(set(every)) or set(every) & {r['id'] for r in old_pool + old_panel}:
        raise ValueError('selected sets overlap each other or the old subset')
    audit = s.math_equivalence_audit([r for v in sets.values() for r in v])
    if not audit['all_correct']:
        raise ValueError('gold validation failed; fix before any inference')
    out.mkdir(parents=True)
    for name, file in SET_FILES.items():
        t.write_jsonl(out / file, sets[name])
    t.write_jsonl(out / 'exclusions.jsonl', sorted(prepared['exclusions'], key=lambda r: r['id']))
    t.atomic_json(out / 'scoring-audit.json', audit)
    parquet = None
    if mirror is not None:
        mirror = Path(mirror)
        if (mirror / 'REVISION').read_text().strip() != SOURCE['revision']:
            raise ValueError('mirror revision differs from the pinned one')
        parquet = dict(sorted(line.split()[::-1] for line in (mirror / 'SHA256SUMS').read_text().splitlines() if line.strip()))
    hard = lambda r: r['level'] in [f'Level {x}' for x in prepared['levels']]
    manifest = {
        'schema': SCHEMA, 'preregistration_sha256': s.file_hash(prereg),
        'source': {**SOURCE, 'tree_files': len(rows), 'tree_sha256_3c': old['source_sha256'],
                   'parquet_sha256': parquet},
        'old_subset_excluded': old,
        'general_v1_sha256': s.file_hash(general_panel), 'near_duplicate_rule': 'kit.p3b_replay_data.near via clean',
        'levels': prepared['levels'], 'level_fallback_used': prepared['levels'] != list(LEVEL_SETS[0]),
        'level_attempts': prepared['attempts'], 'stratum': STRATUM, 'salts': SALTS, 'sizes': SIZES,
        'quotas': prepared['quotas'],
        'counts': {
            'source_by_split_level': counts_by(rows, 'split', 'level'),
            'source_by_split_level_type': counts_by(rows, 'split', 'level', 'type'),
            'eligible_3c_by_split_level_type': counts_by([r for r in rows if not r['excluded']], 'split', 'level', 'type'),
            'hard_level_3c_exclusions_by_reason': dict(sorted(Counter(r['excluded'] for r in rows if hard(r) and r['excluded']).items())),
            'hard_exclusions_by_reason': dict(sorted(Counter(e['reason'] for e in prepared['exclusions']).items())),
            'kept_by_split_level_type': prepared['kept_counts']},
        'sets': {name: {'file': SET_FILES[name], 'n': len(v), 'ids_sha256': s.digest([r['id'] for r in v]),
                        'sha256': s.digest(v), 'by_type': counts_by(v, STRATUM), 'by_level': counts_by(v, 'family'),
                        'split': sorted({r['split'] for r in v})} for name, v in sets.items()},
        'generation': GENERATION,
        'decision_rule': {'set': 'screen', 'qualifies_if': '30% <= correct/200 <= 70% (60-140 of 200, inclusive)',
                          'interval': 'Wilson 95%, reported, not part of the decision', 'per_type': 'descriptive'},
        'scorer': {p: s.file_hash(ROOT / p) for p in SCORER_FILES},
        'selection_implementation': {p: s.file_hash(ROOT / p) for p in SELECTION_FILES},
    }
    manifest['files'] = {p.name: s.file_hash(p) for p in sorted(out.iterdir())}
    t.atomic_json(out / 'manifest.json', manifest)
    return manifest


def load_frozen(data=DATA, *, check_scorer=True):
    """Refuse changed data, prompts, scorer or registration before any inference."""
    data = Path(data)
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA:
        raise ValueError('not a hard-MATH screen manifest')
    for name, expected in manifest['files'].items():
        if s.file_hash(data / name) != expected:
            raise ValueError('frozen data changed: ' + name)
    sets = {k: t.read_jsonl(data / SET_FILES[k]) for k in SET_FILES}
    for k, rows in sets.items():
        if s.digest(rows) != manifest['sets'][k]['sha256'] or len(rows) != SIZES[k]:
            raise ValueError('frozen set changed: ' + k)
    if check_scorer:
        if s.file_hash(PREREG) != manifest['preregistration_sha256']:
            raise ValueError('registration changed after the freeze')
        for p, expected in manifest['scorer'].items():
            if s.file_hash(ROOT / p) != expected:
                raise ValueError('scorer changed after the freeze: ' + p)
    return manifest, sets


def wilson(k, n, z=Z95):
    if n <= 0:
        return None
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((centre - half) / (1 + z * z / n), 6), round((centre + half) / (1 + z * z / n), 6)]


def decide(panel, records, *, technical_synthetic=False):
    """Frozen rule on the complete 200-item panel; per-type rows are descriptive."""
    ids = [r['id'] for r in panel]
    keys = [r.get('id') for r in records]
    complete = (len(ids) == SIZES['screen'] == len(set(ids)) and len(keys) == len(set(keys))
                and set(keys) == set(ids) and all(type(r.get('correct')) is bool for r in records))
    hit = {r['id']: r.get('correct') is True for r in records}
    correct, n = sum(hit.get(i, False) for i in ids), len(ids)
    low, high = BAND
    if not complete:
        status = 'INCOMPLETE'
    elif Fraction(correct, n) > high:
        status = 'FAIL_TOO_EASY'
    elif Fraction(correct, n) < low:
        status = 'FAIL_TOO_HARD'
    else:
        status = 'QUALIFIES'
    finish = Counter(r.get('finish_reason') for r in records if r.get('finish_reason') is not None)
    per_type = {}
    for subject in sorted({r[STRATUM] for r in panel}):
        members = [r['id'] for r in panel if r[STRATUM] == subject]
        k = sum(hit.get(i, False) for i in members)
        per_type[subject] = {'n': len(members), 'correct': k, 'accuracy': round(k / len(members), 6),
                             'wilson95': wilson(k, len(members))}
    return {'status': status, 'complete': complete, 'correct': correct, 'n': n,
            'accuracy': round(correct / n, 6) if n else None, 'wilson95': wilson(correct, n),
            'band': ['30%', '70%'], 'band_counts': [math.ceil(low * n), math.floor(high * n)],
            'per_type': per_type, 'finish_reasons': dict(sorted(finish.items())),
            'technical_synthetic': technical_synthetic,
            'student_headroom_qualifies': status == 'QUALIFIES' and not technical_synthetic,
            'teacher_coverage': 'PENDING: pinned Qwen3.6-27B on the partner node, separately',
            'registration_allowed': False}


def records_from(bed_score, responses=None):
    """eval_bed's per-item verdicts (its math_screen scorer) joined with finish reasons."""
    finish = {r['id']: r.get('finish_reason') for r in responses or []}
    return [{'id': k, 'correct': bool(v), 'finish_reason': finish.get(k)} for k, v in bed_score['per_item'].items()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--math-root', required=True)
    b.add_argument('--mirror', help='the parquet download directory holding REVISION and SHA256SUMS')
    b.add_argument('--out', default=str(DATA))
    sub.add_parser('check').add_argument('--data', default=str(DATA))
    a = p.parse_args(argv)
    if a.command == 'build':
        m = build(a.math_root, a.out, mirror=a.mirror)
        print(json.dumps({k: m[k] for k in ('levels', 'level_fallback_used', 'quotas', 'files')}, indent=2))
    else:
        m, sets = load_frozen(a.data)
        print(json.dumps({k: len(v) for k, v in sets.items()}))


if __name__ == '__main__':
    main()
