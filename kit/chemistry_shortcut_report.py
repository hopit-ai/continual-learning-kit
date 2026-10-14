"""3A archive-only reader: paired question gains, contrasts, CIs and serving cost."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import random
import sys
import tarfile
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kit.chemistry_shortcut_panels import PANELS, item_id
from kit.chemistry_shortcut_validate import training_gold_profiles, gold_cue_diagnostics, CUE_CONVENTION

MODELS = ('base', 'r1', 'r2')
SEED = 31011
EXPECTED_MANIFEST_SHA256 = "77b57c38bd487e3edd656a2a60ec4e193641ae9ac37c2be3e7b5a9bfc12946fb"


class Evidence:
    """Read files directly from the collector archive; no extraction or local data."""
    def __init__(self, path):
        self.path = Path(path)
        self.tar = tarfile.open(path) if self.path.is_file() else None
        if self.tar:
            self.names = {m.name.split('/', 1)[1]: m for m in self.tar.getmembers()
                          if m.isfile() and '/' in m.name}
        else:
            self.names = {str(p.relative_to(path)): p for p in self.path.rglob('*') if p.is_file()}
    def read(self, name):
        return self.tar.extractfile(self.names[name]).read() if self.tar else self.names[name].read_bytes()
    def json(self, name):
        return json.loads(self.read(name))
    def rows(self, name):
        return [json.loads(l) for l in self.read(name).decode().split('\n') if l.strip()]
    def close(self):
        if self.tar:
            self.tar.close()


def mean(values):
    return sum(values) / len(values)


def percentile(values, p):
    ordered = sorted(values); at = (len(ordered) - 1) * p
    lo = int(at); hi = math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def interval(samples):
    return [percentile(samples, .025), percentile(samples, .975)]


def bootstrap(vectors, strata, repeats, rng):
    """Same sampled underlying IDs for ALL columns/models/versions in a stratum."""
    groups = collections.defaultdict(list)
    for q, vector in vectors.items():
        groups[strata[q]].append(vector)
    groups = [group for _, group in sorted(groups.items())]
    samples = [[] for _ in next(iter(vectors.values()))]
    for _ in range(repeats):
        totals = [0.0] * len(samples)
        for group in groups:
            for _ in group:
                vector = group[rng.randrange(len(group))]
                for column, value in enumerate(vector):
                    totals[column] += value
        for column, total in enumerate(totals):
            samples[column].append(total / len(vectors))
    return samples


def report(evidence, repeats=10000):
    warnings = []
    inputs = '3a/report-inputs/'
    manifest = evidence.json(inputs + 'manifest.json')
    for name, expected in manifest['files'].items():
        if inputs + name not in evidence.names or hashlib.sha256(evidence.read(inputs + name)).hexdigest() != expected:
            warnings.append('Frozen input missing/changed: ' + name)
    manifest_hash = hashlib.sha256(evidence.read(inputs + 'manifest.json')).hexdigest()
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        warnings.append('Returned manifest hash differs from expected freeze: ' + manifest_hash)
    validation = evidence.json(inputs + 'validation.json')
    if not validation['ok']:
        warnings.append('Frozen validation has errors')
    panel_rows = {p: evidence.rows(inputs + p + '/test.jsonl') for p in PANELS}
    profiles = training_gold_profiles(evidence.rows(inputs + 'sources/train.json'), evidence.json(inputs + 'sources/taxonomy.json'))
    diagnostics = {p: gold_cue_diagnostics(records, profiles) for p, records in panel_rows.items()}
    metadata = {r['extra_info']['question_id']: r['extra_info'] for r in panel_rows['original']}
    observations = {}
    tables = []
    provenance = {}
    synthetic = False
    for model in MODELS:
        for panel in PANELS:
            plain = f'3a/eval/{model}-{panel}/bed-score.json'
            prefix = f'3a/eval/{model}-{panel}-a'
            candidates = [(0, plain)] if plain in evidence.names else []
            for name in evidence.names:
                if name.startswith(prefix) and name.endswith('/bed-score.json'):
                    try:
                        candidates.append((int(name[len(prefix):].split('/')[0]), name))
                    except ValueError:
                        pass
            if not candidates:
                warnings.append(f'Missing scoring: {model}/{panel}'); continue
            # Latest complete pair; retain incomplete/retried attempts as warnings.
            complete = [(a, n) for a, n in candidates if n.rsplit('/', 1)[0] + '/responses.jsonl' in evidence.names]
            if not complete:
                warnings.append(f'Missing responses: {model}/{panel}'); continue
            attempt, name = max(complete)
            if max(a for a, _ in candidates) > attempt:
                warnings.append(f'Newer incomplete attempt retained: {model}/{panel}')
            score = evidence.json(name); folder = name.rsplit('/', 1)[0]
            raw = evidence.read(folder + '/responses.jsonl')
            response_list = evidence.rows(folder + '/responses.jsonl')
            responses = {str(r['id']): r for r in response_list}
            ids = {item_id(r) for r in panel_rows[panel]}
            if len(responses) != len(response_list) or set(responses) != ids or set(score.get('per_item', {})) != ids:
                warnings.append(f'Incomplete/mismatched IDs: {model}/{panel}'); continue
            if score.get('responses_sha256') != hashlib.sha256(raw).hexdigest():
                warnings.append(f'Response hash differs: {model}/{panel}')
            if score.get('panel_manifest_sha256') != manifest_hash:
                warnings.append(f'Panel manifest provenance differs: {model}/{panel}')
            synthetic |= bool(score.get('technical_standin'))
            if score.get('technical_standin'):
                warnings.append(f'TECHNICAL STAND-IN: {model}/{panel}')
            else:
                engine = score.get('engine', {})
                if not (score.get('decoding', {}).get('max_tokens') == 2048 and
                        engine.get('tensor_parallel_size') == 1 and engine.get('deterministic') and
                        engine.get('eager') and engine.get('batch_invariant') and engine.get('max_model_len') == 12288 and
                        engine.get('dtype') == 'bfloat16' and
                        all(score.get('decoding', {}).get(k) == v for k, v in
                            {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'repetition_penalty': 1.0}.items())):
                    warnings.append(f'Scoring mode departure: {model}/{panel}')
            for warning in score.get('warnings', []):
                warnings.append(f'{model}/{panel}: {warning}')
            provenance[model + '/' + panel] = {'attempt': attempt, 'model': score.get('model'),
                'model_identity': score.get('model_identity'), 'machine': score.get('machine'),
                'scorer_sha256': score.get('scorer_sha256'), 'timing': score.get('timing')}
            grouped = collections.defaultdict(list)
            for row in panel_rows[panel]:
                rid = item_id(row); q = row['extra_info']['question_id']; response = responses[rid]
                # Recompute the unchanged MCQ rule from returned raw responses.
                extracted = response.get('response', '').split('<answer>')[-1].split('</answer>')[0].strip()
                correct = int(extracted == row['reward_model']['ground_truth'])
                if correct != score['per_item'][rid]:
                    warnings.append('Returned verdict differs from raw MCQ answer: ' + rid)
                grouped[q].append({'correct': correct, 'tokens': response.get('output_tokens'),
                                   'format': int('<answer>' not in response.get('response', '')),
                                   'truncated': int(response.get('finish_reason') == 'length')})
            observations[model, panel] = {q: mean([r['correct'] for r in instances]) for q, instances in grouped.items()}
            for stratum in ('all', 'affected', 'unaffected', 'distribution coefficient logD', 'aqueous solubility'):
                if stratum == 'all' and panel in ('balanced', 'rank_preserved'):
                    continue  # These panels contain only affected questions, not the whole 210.
                chosen = [q for q in grouped if stratum == 'all' or
                          (stratum == 'affected' and metadata[q]['affected']) or
                          (stratum == 'unaffected' and not metadata[q]['affected']) or metadata[q]['family'] == stratum]
                if not chosen:
                    continue
                instances = [r for q in chosen for r in grouped[q]]
                correct = sum(r['correct'] for r in instances)
                tokens = None if any(r['tokens'] is None for r in instances) else sum(r['tokens'] for r in instances)
                tables.append({'model': model, 'panel': panel, 'stratum': stratum, 'questions': len(chosen),
                    'instances': len(instances), 'correct_instances': correct,
                    'accuracy': mean([observations[model, panel][q] for q in chosen]),
                    'output_tokens': tokens, 'tokens_per_correct': tokens / correct if tokens is not None and correct else None,
                    'format_failures': sum(r['format'] for r in instances), 'truncated': sum(r['truncated'] for r in instances)})
    scorer_hashes = {p['scorer_sha256'] for p in provenance.values()}
    if len(scorer_hashes) > 1:
        warnings.append('Scorer identity differs across scoring rows')
    software = {json.dumps(p.get('machine', {}).get('versions'), sort_keys=True) for p in provenance.values() if p.get('machine')}
    if len(software) > 1:
        warnings.append('Scoring software versions differ across rows')
    for row in tables:
        if row['model'] == 'base':
            continue
        peers = [r for r in tables if r['model'] == 'base' and r['panel'] == row['panel'] and r['stratum'] == row['stratum']]
        row['gain'] = row['accuracy'] - peers[0]['accuracy'] if peers else None
    effects = []; gain_intervals = []; decisions = {}; fractions = []
    rng = random.Random(SEED)
    # Two independent family-stratified bootstrap streams for affected/unaffected;
    # each stream preserves all models and versions. Successive draws from the
    # registered RNG make disjoint strata independent for selective contrasts.
    for stratum in ('affected', 'unaffected', 'distribution coefficient logD', 'aqueous solubility', 'all'):
        qs = [q for q, info in metadata.items() if stratum == 'all' or
              (stratum == 'affected' and info['affected']) or (stratum == 'unaffected' and not info['affected']) or info['family'] == stratum]
        available = [p for p in PANELS if all((m, p) in observations and all(q in observations[m, p] for q in qs) for m in MODELS)]
        if not qs or 'original' not in available or 'letters' not in available:
            continue
        names = [('gain_' + m + '_' + p, m, p) for m in ('r1', 'r2') for p in available]
        vectors = {}
        for q in qs:
            vectors[q] = [observations[m, p][q] - observations['base', p][q] for _, m, p in names]
        boot = bootstrap(vectors, {q: metadata[q]['family'] for q in qs}, repeats, rng)
        averages = [mean([v[i] for v in vectors.values()]) for i in range(len(names))]
        lookup = {(m, p): i for i, (_, m, p) in enumerate(names)}
        for i, (_, model, panel) in enumerate(names):
            gain_intervals.append({'model': model, 'panel': panel, 'stratum': stratum,
                                   'gain': averages[i], 'ci95': interval(boot[i])})
        for model in ('r1', 'r2'):
            def values(panel):
                i = lookup[model, panel]; return averages[i], boot[i]
            original, orig_boot = values('original'); perm, perm_boot = values('letters')
            matched = (original + 3 * perm) / 4
            matched_boot = [(a + 3 * b) / 4 for a, b in zip(orig_boot, perm_boot)]
            comparisons = [('letter_minus_original', perm - original, [b - a for a, b in zip(orig_boot, perm_boot)])]
            if 'balanced' in available and 'rank_preserved' in available:
                balanced, bal_boot = values('balanced'); redraw, red_boot = values('rank_preserved')
                comparisons += [('balanced_minus_original', balanced - original, [b - a for a, b in zip(orig_boot, bal_boot)]),
                    ('balanced_minus_matched_letters', balanced - matched, [b - a for a, b in zip(matched_boot, bal_boot)]),
                    ('rank_preserved_minus_matched_letters', redraw - matched, [b - a for a, b in zip(matched_boot, red_boot)]),
                    ('balanced_minus_rank_preserved', balanced - redraw, [b - a for a, b in zip(red_boot, bal_boot)])]
                if stratum == 'affected':
                    fractions.append({'model': model, 'original_affected_gain': original, 'balanced_affected_gain': balanced,
                                      'rank_preserved_affected_gain': redraw,
                                      'rank_referenced_lost_affected_gain': redraw - balanced,
                                      'fraction_rank_preserved_gain_lost': (redraw - balanced) / redraw if redraw > 0 else None,
                                      'rank_referenced_whole_panel_lost_gain_contribution': (redraw - balanced) * 51 / 210,
                                      'headline_fraction': 'fraction_rank_preserved_gain_lost',
                                      'lost_affected_gain': original - balanced,
                                      'fraction_affected_gain_lost': (original - balanced) / original if original > 0 else None,
                                      'whole_panel_lost_gain_contribution': (original - balanced) * 51 / 210})
            for label, estimate, distribution in comparisons:
                effects.append({'model': model, 'stratum': stratum, 'contrast': label, 'change_in_gain': estimate,
                                'ci95': interval(distribution), '_bootstrap': distribution})
    for model in ('r1', 'r2'):
        def effect(stratum, contrast):
            return next((e for e in effects if e['model'] == model and e['stratum'] == stratum and e['contrast'] == contrast), None)
        a = effect('affected', 'balanced_minus_original'); u = effect('unaffected', 'letter_minus_original')
        r = effect('affected', 'balanced_minus_rank_preserved')
        if a and u and r:
            distribution = [x - y for x, y in zip(a['_bootstrap'], u['_bootstrap'])]
            selective = {'model': model, 'stratum': 'affected_minus_unaffected', 'contrast': 'selective_gain_change',
                         'change_in_gain': a['change_in_gain'] - u['change_in_gain'], 'ci95': interval(distribution)}
            effects.append(selective)
            rank_distribution = [x - y for x, y in zip(r['_bootstrap'], u['_bootstrap'])]
            rank_selective = {'model': model, 'stratum': 'affected_minus_unaffected', 'contrast': 'selective_rank_gain_change',
                              'change_in_gain': r['change_in_gain'] - u['change_in_gain'], 'ci95': interval(rank_distribution)}
            effects.append(rank_selective)
            redraw = next(g['gain'] for g in gain_intervals if g['model'] == model and g['panel'] == 'rank_preserved' and g['stratum'] == 'affected')
            decisions[model] = {'strong_support': redraw > 0 and rank_selective['ci95'][1] < 0 and r['ci95'][1] < 0 and u['ci95'][0] > -.05,
                                'original_referenced_loss_below_zero': a['ci95'][1] < 0,
                                'unaffected_within_5pp': u['ci95'][0] > -.05}
        for fraction in fractions:
            if fraction['model'] == model:
                whole = next((g['gain'] for g in gain_intervals if g['model'] == model and g['panel'] == 'original' and g['stratum'] == 'all'), None)
                fraction['original_whole_panel_gain'] = whole
                fraction['fraction_whole_gain_lost_contribution_rank_reference'] = fraction['rank_referenced_whole_panel_lost_gain_contribution'] / whole if whole and whole > 0 else None
                fraction['fraction_whole_gain_lost_contribution'] = fraction['whole_panel_lost_gain_contribution'] / whole if whole and whole > 0 else None
    base_changes = []
    for row in tables:
        if row['model'] == 'base' and row['panel'] != 'original':
            reference = next((r for r in tables if r['model'] == 'base' and r['panel'] == 'original' and r['stratum'] == row['stratum']), None)
            if reference:
                base_changes.append({'panel': row['panel'], 'stratum': row['stratum'], 'accuracy_change_from_original': row['accuracy'] - reference['accuracy']})
    for e in effects:
        e.pop('_bootstrap', None)
    return {'schema': 'chemistry-shortcut-report.v1', 'technical_standin': synthetic,
            'headline_contrast': 'balanced_minus_rank_preserved',
            'surviving_gain_interpretation': 'Bounds rank-cue dependence only; a value-band cue remains untested.',
            'gold_cue_diagnostics': diagnostics,
            'gold_cue_convention': CUE_CONVENTION,
            'training_gold_profiles': {f: {k: v for k, v in profile.items() if k != 'gold_strings'} for f, profile in profiles.items()},
            'complete': len(observations) == 12, 'warnings': warnings, 'panel_manifest_sha256': manifest_hash,
            'bootstrap': {'replicates': repeats, 'seed': SEED, 'unit': 'underlying question, stratified by family'},
            'accuracy_and_cost': tables, 'base_perturbation_effects': base_changes, 'gains': gain_intervals, 'effects': effects, 'descriptive_fractions': fractions,
            'interpretation': decisions, 'seed_disagreement': len(decisions) == 2 and decisions['r1']['strong_support'] != decisions['r2']['strong_support'], 'replicated_strong_support': len(decisions) == 2 and all(d['strong_support'] for d in decisions.values()),
            'provenance': provenance}


def markdown(result):
    lines = ['# 3A Chemistry shortcut readout', '',
             '**TECHNICAL STAND-IN — no scientific model result.**' if result['technical_standin'] else 'Paired question estimates; intervals condition on two trained checkpoints.', '',
             '| Model | Panel | Stratum | Questions / instances | Accuracy % | Gain pp [95% CI] | Tokens/correct |',
             '|---|---|---|---:|---:|---:|---:|']
    def number(value, scale=1):
        return 'undefined' if value is None else f'{value * scale:.2f}'
    for r in result['accuracy_and_cost']:
        gain = next((g for g in result['gains'] if (g['model'], g['panel'], g['stratum']) == (r['model'], r['panel'], r['stratum'])), None)
        gain_text = number(r.get('gain'),100) + (f" [{number(gain['ci95'][0],100)}, {number(gain['ci95'][1],100)}]" if gain else '')
        lines.append(f"| {r['model']} | {r['panel']} | {r['stratum']} | {r['questions']} / {r['instances']} | {number(r['accuracy'],100)} | {gain_text} | {number(r['tokens_per_correct'])} |")
    lines += ['', 'Headline rank contrast: **Balanced minus Rank-preserved gain**.',
              'Balanced minus Original also changes familiarity and band dilution.', '', '| Model | Stratum | Paired change in gain | Estimate pp | 95% interval pp |', '|---|---|---|---:|---:|']
    for e in sorted(result['effects'], key=lambda e: e['contrast'] != result['headline_contrast']):
        lines.append(f"| {e['model']} | {e['stratum']} | {e['contrast']} | {number(e['change_in_gain'],100)} | [{number(e['ci95'][0],100)}, {number(e['ci95'][1],100)}] |")
    lines += ['', '| Model | Rank-preserved gain pp | Balanced gain pp | Headline lost gain % (Rank-preserved reference) | Secondary lost gain % (Original reference) |',
              '|---|---:|---:|---:|---:|']
    for f in result['descriptive_fractions']:
        lines.append(f"| {f['model']} | {number(f['rank_preserved_affected_gain'],100)} | {number(f['balanced_affected_gain'],100)} | {number(f['fraction_rank_preserved_gain_lost'],100)} | {number(f['fraction_affected_gain_lost'],100)} |")
    lines += ['', 'Descriptive training-gold cues (expected accuracy; no model output):',
              result['gold_cue_convention'], '',
              '| Panel | Family | Band % | Nearest training-gold median % | Training-gold string % | Seen distractors / total |',
              '|---|---|---:|---:|---:|---:|']
    for panel, strata in result['gold_cue_diagnostics'].items():
        for family, diagnostic in strata.items():
            accuracy = diagnostic['heuristic_accuracy']
            lines.append(f"| {panel} | {family} | {number(accuracy['training_gold_band'],100)} | {number(accuracy['nearest_training_gold_median'],100)} | {number(accuracy['training_gold_string'],100)} | {diagnostic['distractor_training_string_matches']} / {diagnostic['distractors_n']} |")
    lines += ['', 'These fractions are descriptive, conditional on positive denominators, and unbounded.',
              'Seed disagreement on the registered strong-support criteria: ' + str(result['seed_disagreement'])]
    lines += ['', 'Replicated strong-support criterion: ' + str(result['replicated_strong_support']),
              'Surviving gains bound rank-cue dependence only; a value-band cue remains untested. Ratios are descriptive; see reading.json.', '', 'Warnings:']
    lines += ['- ' + w for w in result['warnings']] or ['- none']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='collector tar.gz or extracted work directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000, help='rehearsal may use fewer; registered scientific table uses 10000')
    args = parser.parse_args()
    evidence = Evidence(args.archive)
    try:
        result = report(evidence, args.bootstrap)
    finally:
        evidence.close()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'reading.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (args.out / 'table.md').write_text(markdown(result))
    print(json.dumps({'complete': result['complete'], 'technical_standin': result['technical_standin'], 'warnings': len(result['warnings'])}))


if __name__ == '__main__':
    main()
