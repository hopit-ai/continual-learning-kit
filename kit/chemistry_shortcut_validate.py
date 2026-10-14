"""Independent mechanical audit of WRITTEN 3A panel prompts, without model output."""
from __future__ import annotations
import argparse
import collections
import json
import re
import statistics
from decimal import Decimal
from pathlib import Path

OPT = re.compile(r'^([A-D]): (.*)$', re.M)
AFFECTED = ('distribution coefficient logD', 'aqueous solubility')
CUE_CONVENTION = 'Exact inclusive training-gold band; frequency-weighted training-gold median; literal training-gold strings. Uniform ties; uniform four-option fallback when no match.'


def training_gold_profiles(training, taxonomy):
    """Training-only, exact decimal bands and frequency-weighted medians."""
    families = {(q['split'], q['idx']): q['family'] for q in taxonomy['questions']}
    golds = {f: [] for f in AFFECTED}
    for row in training:
        family = families['train', row['idx']]
        if family in golds:
            golds[family].append(dict(OPT.findall(row['prompt']))[row['answer']])
    return {f: {'training_n': len(texts), 'gold_strings': sorted(set(texts)),
                'band_min': str(min(Decimal(t) for t in texts)),
                'band_max': str(max(Decimal(t) for t in texts)),
                'median': str(statistics.median([Decimal(t) for t in texts]))}
            for f, texts in golds.items()}


def gold_cue_diagnostics(records, profiles):
    """Expected accuracy: uniform ties, uniform four-option fallback on no match.

    Band membership includes exact observed endpoints; familiarity is literal
    string membership. These are descriptive rules, never validation gates.
    """
    groups = collections.defaultdict(list)
    for row in records:
        family = row['extra_info']['family']
        if family not in profiles:
            continue
        profile = profiles[family]
        options = dict(OPT.findall(row['prompt'][1]['content']))
        values = {letter: Decimal(t) for letter, t in options.items()}
        gold = row['reward_model']['ground_truth']
        low, high, median = (Decimal(profile[k]) for k in ('band_min', 'band_max', 'median'))
        known = set(profile['gold_strings'])
        distances = {k: abs(v - median) for k, v in values.items()}
        choices = {'training_gold_band': [k for k, v in values.items() if low <= v <= high],
                   'nearest_training_gold_median': [k for k, d in distances.items() if d == min(distances.values())],
                   'training_gold_string': [k for k, text in options.items() if text in known]}
        scores = {name: int(gold in (candidates or list(options))) / len(candidates or options)
                  for name, candidates in choices.items()}
        observation = {'scores': scores, 'gold_seen': int(options[gold] in known),
                       'distractors_seen': sum(text in known for k, text in options.items() if k != gold),
                       'gold_in_band': int(low <= values[gold] <= high)}
        for stratum in ('affected', family):
            groups[stratum].append(observation)
    return {stratum: {'n': len(obs),
                       'heuristic_accuracy': {name: sum(o['scores'][name] for o in obs) / len(obs)
                                              for name in obs[0]['scores']},
                       'gold_training_string_matches': sum(o['gold_seen'] for o in obs),
                       'distractor_training_string_matches': sum(o['distractors_seen'] for o in obs),
                       'distractors_n': 3 * len(obs),
                       'gold_inside_training_band': sum(o['gold_in_band'] for o in obs)}
            for stratum, obs in groups.items()}


def validate(root):
    root = Path(root)
    errors = []
    def check(ok, text):
        if not ok:
            errors.append(text)
    def load(path):
        text = path.read_text()
        return json.loads(text) if text.lstrip().startswith('[') else [json.loads(l) for l in text.split('\n') if l.strip()]
    sources = {s: load(root / 'sources' / (s + '.json')) for s in ('train', 'test')}
    check(not {r['idx'] for r in sources['train']} & {r['idx'] for r in sources['test']}, 'train/test ID overlap')
    tax = json.loads((root / 'sources/taxonomy.json').read_text())
    families = {(q['split'], q['idx']): q['family'] for q in tax['questions']}
    profiles = training_gold_profiles(sources['train'], tax)
    originals = {str(r['idx']): r for r in sources['test']}
    pools = {f: set() for f in AFFECTED}
    for row in sources['train']:
        f = families['train', row['idx']]
        if f in pools:
            pools[f].update(t for _, t in OPT.findall(row['prompt']))
    report = {'schema': 'chemistry-shortcut-validation.v1', 'errors': errors, 'panels': {},
              'diagnostic_convention': CUE_CONVENTION,
              'training_gold_profiles': {f: {k: v for k, v in profile.items() if k != 'gold_strings'} for f, profile in profiles.items()},
              'plausibility': 'All new distractors drawn from same-property training option values; minimum pairwise distance 0.10 log units. Not chemical adjudication.'}
    for panel, expected in (('original', 210), ('letters', 630), ('balanced', 204), ('rank_preserved', 204)):
        records = load(root / panel / 'test.jsonl')
        check(len(records) == expected, panel + ' count')
        seen = set()
        groups = collections.defaultdict(list)
        ranks = collections.defaultdict(collections.Counter)
        heuristic = collections.defaultdict(collections.Counter)
        rank_letter = collections.defaultdict(collections.Counter)
        for row in records:
            info = row['extra_info']; index = str(info['index']); qid = info['question_id']
            check(index not in seen, panel + ' duplicate index'); seen.add(index)
            idx = qid.removeprefix('sciknoweval-test-')
            original = originals[idx]
            f = families['test', original['idx']]
            check(info['family'] == f and info['affected'] == (f in AFFECTED), qid + ' family metadata')
            check(row['prompt'][0] == {'role': 'system', 'content': original['system']}, qid + ' system changed')
            text = row['prompt'][1]['content']; pairs = OPT.findall(text)
            check(len(pairs) == 4 and [k for k, _ in pairs] == list('ABCD'), qid + ' option format')
            opts = dict(pairs); old = dict(OPT.findall(original['prompt']))
            gold = row['reward_model']['ground_truth']; gold_text = old[original['answer']]
            check(OPT.sub('', text) == OPT.sub('', original['prompt']), qid + ' stem/suffix changed')
            check(opts[gold] == gold_text, qid + ' gold spelling/value changed')
            if panel == 'original':
                check(text == original['prompt'] and gold == original['answer'], qid + ' original changed')
            if panel == 'letters':
                rotation = info['variant']
                check(list(opts.values()) == list(old.values())[rotation:] + list(old.values())[:rotation], qid + ' fixed rotation changed')
            rank = None
            if f in AFFECTED:
                values = {k: Decimal(v) for k, v in opts.items()}
                ordered = sorted(values, key=values.get); rank = ordered.index(gold) + 1
                check(len(set(values.values())) == 4, qid + ' numeric ties')
                for stratum in ('affected', f):
                    ranks[stratum][rank] += 1
                    rank_letter[stratum][f'{rank}:{gold}'] += 1
                    for k in range(1, 5):
                        heuristic[stratum][f'rank_{k}'] += int(ordered[k - 1] == gold)
                    mean = sum(values.values()) / 4
                    for name, reverse in (('nearest_mean', False), ('farthest_mean', True)):
                        chosen = sorted(values, key=lambda k: ((-1 if reverse else 1) * abs(values[k] - mean), k))[0]
                        heuristic[stratum][name] += int(chosen == gold)
                    for letter in 'ABCD':
                        heuristic[stratum]['letter_' + letter] += int(letter == gold)
            if panel in ('balanced', 'rank_preserved'):
                check(f in AFFECTED, qid + ' unaffected numeric intervention')
                check(rank == info['gold_rank'], qid + ' rank metadata incorrect')
                vals = list(values.values())
                check(all(abs(a - b) >= Decimal('0.10') for i, a in enumerate(vals) for b in vals[i + 1:]), qid + ' minimum separation')
                check(all(t in pools[f] for k, t in opts.items() if k != gold), qid + ' distractor outside training distribution')
                if panel == 'rank_preserved':
                    check(rank == 3, qid + ' redraw changed rank cue')
            groups[qid].append((gold, rank))
        expected_ids = {f"sciknoweval-test-{r['idx']}" for r in sources['test']
                        if panel in ('original', 'letters') or families['test', r['idx']] in AFFECTED}
        check(set(groups) == expected_ids, panel + ' question coverage')
        for qid, observations in groups.items():
            check(len(observations) == {'original': 1, 'letters': 3, 'balanced': 4, 'rank_preserved': 4}[panel], qid + ' repeated-question count')
            if panel in ('balanced', 'rank_preserved'):
                check(sorted(g for g, _ in observations) == list('ABCD'), qid + ' gold letters not balanced')
            if panel == 'balanced':
                check(sorted(r for _, r in observations) == [1, 2, 3, 4], qid + ' gold ranks not balanced')
        audits = {}
        for stratum, counts in ranks.items():
            n = sum(counts.values())
            rates = {name: count / n for name, count in heuristic[stratum].items()}
            if panel == 'balanced':
                check(all(rates.get('rank_' + str(k), 0) == 0.25 for k in range(1, 5)), stratum + ' simple rank rule beats chance')
            if panel in ('original', 'letters', 'rank_preserved'):
                check(counts[3] == n, panel + ' original rank invariant changed')
            audits[stratum] = {'n': n, 'gold_rank_counts': dict(counts), 'heuristic_accuracy': rates,
                               'rank_letter_counts': dict(rank_letter[stratum])}
        report['panels'][panel] = {'n': len(records), 'questions': len(groups), 'strata': audits,
                                    'gold_cue_diagnostics': gold_cue_diagnostics(records, profiles)}
    report['ok'] = not errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.panels)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(int(not result['ok']))


if __name__ == '__main__':
    main()
