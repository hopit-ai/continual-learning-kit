"""Inference-only, pre-stated Rung P fraction sweep on frozen smoke inputs."""
import math

from kit import p_diag as diag
from kit import p_replace as p
from kit import p_smoke as smoke

FRACTIONS = {'0.05%': .0005, '0.1%': .001, '0.25%': .0025, '0.5%': .005}
VARIANTS = {'V0': 0., **FRACTIONS, 'V2': .01}
REGISTRATION = 'docs/programme-plan-v6-amendment-20261024.md, Rung P, Fraction sweep (pre-stated 25 October)'
RULE_TEXT = (
    'The largest fraction with p99.9 |Δlogit| ≤ 1 and |response NLL change| ≤ 0.1 '
    'nats/token is carried into a full smoke attempt 2 (all mechanical checks); '
    'if no fraction qualifies, rung P as unit replacement is not feasible here '
    'and the named fallback (regenerative L2 towards base weights) is registered instead. '
    'Either outcome is a re-registration of rung P (utility definition and fraction) '
    'that Astra reviews and the owner approves before any partner run; these choices '
    'use smoke development data only, never a scored panel.'
)


def count(fraction, units):
    if not math.isfinite(fraction) or not 0 <= fraction <= 1 or units < 0:
        raise ValueError('invalid fraction or unit count')
    return math.floor(fraction*units)


def selections(stats):
    """Rank p_diag's V2 utility exactly; floor each layer's count, ties by index."""
    result = {name: [] for name in VARIANTS}
    for stat in stats:
        values = stat['all_utility']
        if not values or not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('invalid utility array')
        order = sorted(range(len(values)), key=lambda j: (values[j], j))
        for name, fraction in VARIANTS.items():
            result[name].append(order[:count(fraction, len(values))])
    return result


def apply_rule(variants):
    """Only the four registered candidates participate; no monotonicity assumed."""
    qualifies = {}
    for name in FRACTIONS:
        damage = variants[name]['damage']
        delta, nll = damage['p99_9_logit_delta'], damage['nll_rise_nats_per_token']
        qualifies[name] = (math.isfinite(delta) and 0 <= delta <= 1 and
                           math.isfinite(nll) and abs(nll) <= .1)
    return max((FRACTIONS[k] for k, passed in qualifies.items() if passed), default=None), qualifies


def run_variants(model, tokenizer, prompts, settings, selection):
    """One unchanged baseline; restore and hash-audit every independent variant."""
    initial = diag.projection_hashes(model)
    baseline, _ = diag.snapshot(model, tokenizer, prompts, settings)
    results, audits = {}, {}
    for name, fraction in VARIANTS.items():
        print(f'[progress] fraction sweep {name}: fraction={fraction}', flush=True)
        audit = {}
        with diag.edited(model, selection[name], settings['seed'], audit=audit):
            current, signature = diag.snapshot(model, tokenizer, prompts, settings)
            damage = diag.compare(baseline, current)
            del current
        if any(initial[k] != value for k, value in audit['restored'].items()):
            raise ValueError('restored tensors differ from initial model')
        audits[name] = audit
        results[name] = dict(fraction=fraction, damage=damage, output_sha256=signature,
            selected_per_layer=[len(v) for v in selection[name]],
            restoration=dict(passed=audit['passed'], tensors=len(audit['before']),
                before_sha256=p.digest(audit['before']), restored_sha256=p.digest(audit['restored'])))
    final = diag.projection_hashes(model)
    zero = results['V0']['damage']
    # Greedy decoding is nondeterministic even for a no-op on the H100 setup.
    checks = dict(V0_exact_zero=all(zero[k] == 0 for k in
        ('p99_logit_delta', 'p99_9_logit_delta', 'max_logit_delta', 'nll_rise_nats_per_token')),
        restoration=initial == final and all(a['passed'] for a in audits.values()))
    return results, checks, dict(initial=initial, final=final, variants=audits)


def run(checkpoint, reference, output, arrays_path):
    """Persist a failure record too; scientific qualification is separate from PASS."""
    import json
    from pathlib import Path
    record = dict(schema='rung-p-fraction-sweep.v1', diagnostic_only=True, training=False,
        passed=False, selected_fraction=None, selection_valid=False,
        fraction_units='proportion (0.0005 = 0.05%)',
        candidate_fractions=FRACTIONS, selection_rule=RULE_TEXT, registration=REGISTRATION,
        greedy_reporting='reported only; excluded from V0, reproduction and selection gates')
    try:
        from transformers import AutoTokenizer
        expected = json.loads(Path(reference).read_text())
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        settings, rows, prompts = smoke.frozen_inputs(tokenizer)
        if expected['settings_sha256'] != p.sha(smoke.RESOURCE/'smoke-settings.json'):
            raise ValueError('diagnostic settings differ from frozen inputs')
        for key in ('model', 'revision', 'pool_sha256', 'prompts_sha256', 'greedy'):
            if expected[key] != settings[key]:
                raise ValueError('diagnostic input differs: '+key)
        record.update(reference_sha256=p.sha(reference), reference_variant='V2',
            settings_sha256=expected['settings_sha256'], model=settings['model'],
            revision=settings['revision'], pool_sha256=settings['pool_sha256'],
            prompts_sha256=settings['prompts_sha256'], greedy=settings['greedy'],
            seed=settings['seed'], prompt_count=len(prompts), damage_nll=settings['nll'],
            damage_logits=settings['logits'],
            all_position_definition='every input position, prompt plus response including final EOS',
            utility_definition='kit.p_diag.activation_stats all_utility (V2), unchanged',
            selection='floor(fraction*layer units), lowest utility, ties by unit index',
            replacement='kit.p_replace.redraw: per-unit seeded CPU fp32 normal(0, initializer_range), '
                        'gate then up cast to stored dtype; zero down column')
        model = smoke.load_model(checkpoint).eval().requires_grad_(False)
        if model.config.model_type != 'qwen3' or model.config.hidden_act != 'silu':
            raise ValueError('requires Qwen3 SiLU MLP')
        print('[progress] measuring diagnostic V2 all-position utility', flush=True)
        stats, metadata = diag.activation_stats(model, tokenizer, rows)
        selection = selections(stats)
        if settings['model'] == 'Qwen/Qwen3-8B':
            assert model.config.intermediate_size == 12288
            assert [count(f, 12288) for f in FRACTIONS.values()] == [6, 12, 30, 61]
            assert all([len(selection[k][i]) for k in FRACTIONS] == [6, 12, 30, 61]
                       for i in range(len(model.model.layers)))
        variants, checks, hashes = run_variants(model, tokenizer, prompts, settings, selection)
        reproduction = diag.reproduce(variants['V2']['damage'], expected['variants']['V2']['damage'],
                                      include_greedy=False)
        reproduction['greedy_excluded'] = True
        checks.update(V2_reproduces_diagnostic=reproduction['passed'],
            V2_counts_match=variants['V2']['selected_per_layer'] ==
                            expected['variants']['V2']['selected_per_layer'],
            all_position_tokens_match=metadata['all_tokens'] == expected['all_position_tokens'],
            prompt_tokens_match=metadata['prompt_tokens'] == expected['prompt_tokens'],
            prompt_count_match=len(prompts) == expected['prompt_count'])
        selected, qualifies = apply_rule(variants)
        arrays = p.write_receipt(arrays_path, dict(schema='rung-p-fraction-sweep-arrays.v1',
            activation_stats=stats, selection=selection, measurement=metadata, tensor_hashes=hashes))
        record.update(variants=variants, checks=checks, reproduction=reproduction,
            selected_fraction=selected, fraction_qualifies=qualifies,
            selection_valid=all(checks.values()),
            all_position_tokens=metadata['all_tokens'], prompt_tokens=metadata['prompt_tokens'],
            arrays=dict(file=Path(arrays_path).name, sha256=p.sha(arrays_path),
                        receipt_sha256=arrays['receipt_sha256']), passed=all(checks.values()))
    except Exception as exc:
        record['error'] = type(exc).__name__+': '+str(exc)
    Path(output).write_text(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
    return record


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference', 'output', 'arrays'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = run(args.checkpoint, args.reference, args.output, args.arrays)
    raise SystemExit(0 if result['passed'] else 1)
