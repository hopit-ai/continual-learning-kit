"""Deletion endpoint adapters over the unchanged v7b freeze reader/scorer."""
import math
from pathlib import Path

from kit.nm_common import (
    GENERAL, PANELS, SEEDS, STEPS, Invalid, Pending, checked, digest, need, ready,
)
from kit.nm_read import Table, Hardware, read as freeze_read
from kit.nm_text import count_endpoints, recompute

PANELS_ALL = (*PANELS, 'legal-diagnostic')


def frontier(panel):
    steps = STEPS if panel in ('legal', 'legal-diagnostic') else (0, 20, 40)
    return [('base', 0), ('incoming', 0),
            *((arm, step) for arm in ('C', 'T') for step in steps)]


def finite(value):
    if isinstance(value, float):
        need(math.isfinite(value), 'correctness stop: non-finite scientific evidence')
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite(child)


def evidence_check(evidence, binding, lr):
    """Missing qualification is pending; evidence identities may never be rebased."""
    finite(evidence)
    recipe = binding.get('recipe', {})
    ready(all(str(s) in recipe.get('starts', {}) and
              str(s) in recipe.get('schedule_hashes', {}) for s in SEEDS),
          'bound parents or schedules absent')
    ready(evidence.get('qualification') == 'COMPLETE', 'qualification incomplete')
    need(evidence['recipe_sha256'] == digest(binding['recipe']), 'recipe identity differs')
    need(evidence['parents_sha256'] == digest(binding['recipe']['starts']),
         'incoming parent identity differs')
    need(evidence['schedules_sha256'] == digest(binding['recipe']['schedule_hashes']),
         'paired schedule identity differs')
    need(evidence['lr_sha256'] == lr['sha256'], 'scientific LR receipt differs')
    ready(evidence.get('lr_frozen_before_science') is True, 'prospective LR seal absent')
    for seed in map(str, SEEDS):
        q = evidence.get('seeds', {}).get(seed, {})
        ready(all(q.get(k) is True for k in
                  ('paired_schedule', 'dense_immutable', 'step0_hard', 'resume_equal',
                   'component_qualification', 'data_isolation')), 'seed qualification absent')
        ready(q.get('rate') == lr['rate'], 'scientific rate differs')
        canary = q.get('canary', {})
        ready(canary.get('ids_sha256') and canary.get('n', 0) > 0,
              'frozen canary identity absent')
        for role in ('incoming', 'initial_soft', 'initial_hard', 'teacher'):
            ready(role in canary, 'initial damage canary incomplete')
            need(type(canary[role]) in (int, float) and 0 <= canary[role] <= 100,
                 'invalid canary accuracy')
        masks = q.get('masks', {})
        ready(set(masks) == set(map(str, STEPS)), 'per-export masks absent')
        names = set(masks['0'])
        ready(names, 'layer mask inventory absent')
        for step, layers in masks.items():
            need(set(layers) == names, 'mask layer inventory changed')
            for name, counts in layers.items():
                n, zeros, kd, dk = [counts[k] for k in
                                   ('weights', 'hard_zeros', 'keep_to_delete', 'delete_to_keep')]
                need(all(type(v) is int for v in (n, zeros, kd, dk)) and n > 0 and
                     all(0 <= v <= n for v in (zeros, kd, dk)), 'invalid layer counts')
                old = masks['0'][name]
                need(n == old['weights'] and zeros == old['hard_zeros'] + kd - dk,
                     'mask zero/flip arithmetic differs')
                need(kd <= n-old['hard_zeros'] and dk <= old['hard_zeros'],
                     'mask flips exceed initial population')
                if step == '0':
                    need(kd == dk == 0, 'initial mask flips must be zero')


def lr_evidence(ref):
    # Reuse the production validator, including the registered selection rule.
    receipt = checked(ref)
    finite(receipt)
    from deletion_tuning.deletion_lr import frozen_rate
    rate = frozen_rate(Path(ref['path']), ref['sha256'])
    return dict(sha256=ref['sha256'], rate=rate, fallback=receipt['fallback'],
                receipt=receipt)


def read(scores, spec, binding, evidence, lr):
    try:
        finite(scores)
        evidence_check(evidence, binding, lr)
        table = Table(scores, spec)
        for seed in SEEDS:
            for panel in PANELS_ALL:
                for arm, step in frontier(panel):
                    table.cell(seed, panel, arm, step)
                # C's unmodified step zero must be the untouched incoming bank.
                need(spec['checkpoints'][str(seed)]['C']['0'] ==
                     spec['checkpoints'][str(seed)]['incoming']['0'], 'C step zero rebased')
        result = freeze_read('freeze', scores, spec, binding)
        result['warnings'] = [f'INITIAL_CANARY_DAMAGE_s{s}' for s in SEEDS
            if evidence['seeds'][str(s)]['canary']['incoming'] -
               evidence['seeds'][str(s)]['canary']['initial_hard'] > 5]
        result['historical_matched_guard_SD'] = 'UNMEASURED'
        if result.get('primary'):
            result['lanes'] = {f'{s}.{p}': digest(identity)
                               for (s, p), identity in table.used.items()}
            result['initial_scores'] = {
                f'{s}.{p}': [table.acc(s, p, 'incoming'), table.acc(s, p, 'C', 0),
                             table.acc(s, p, 'T', 0)]
                for s in SEEDS for p in PANELS_ALL}
            result['step20_scores'] = {
                f'{s}.{p}': [table.acc(s, p, a, 20) for a in ('C', 'T')]
                for s in SEEDS for p in PANELS_ALL}
            result['legal_diagnostic'] = {
                str(s): {a: [table.acc(s, 'legal-diagnostic', a, k) for k in STEPS]
                         for a in ('C', 'T')} for s in SEEDS}
        result['family'] = 'deletion'
        return result
    except Hardware as error:
        return dict(status='UNVERIFIED-HARDWARE', reason=str(error))
    except (Pending, FileNotFoundError) as error:
        return dict(status='PENDING/INCOMPLETE', reason=str(error))
    except (Invalid, KeyError, TypeError, ValueError) as error:
        stop = 'non-finite' in str(error)
        return dict(status='PENDING/INCOMPLETE' if stop else 'UNVERIFIED',
                    reason=str(error), correctness_stop=stop)


def from_counts(counts, lanes, *, technical=False):
    """Inherited return arithmetic independently reconstructs every df=2 bar."""
    expected = {f'{s}.{p}' for s in SEEDS for p in PANELS_ALL}
    ready(set(lanes) == expected, 'complete lane identity hashes absent')
    ready(set(counts) == {f'{s}.{p}' for s in SEEDS for p in PANELS},
          'complete registered counts absent')
    for key, row in counts.items():
        need(row and row[0] == (1147 if key.endswith('.finqa') else 100),
             'registered denominator differs')
    meta = dict(meta=dict(reported_status='PASS', benchmark=False, technical=technical),
                lane={k: v for k, v in lanes.items() if not k.endswith('.legal-diagnostic')})
    return recompute('freeze', count_endpoints('freeze', counts), meta)
