"""Inference-only Rung P selection diagnostics; one model, reversible weight edits."""
from contextlib import contextmanager
import hashlib
import math

from kit import p_replace as p
from kit import p_smoke as smoke

PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
VARIANTS = {
    'V0': 'no-op', 'V1': 'registered response utility',
    'V2': 'all-position utility', 'V3': 'seeded random',
    'V4': 'registered selection, down columns only',
    'V5': 'response utility excluding top 0.5% peak activations',
}


def tensor_hash(value):
    import torch
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def projection_hashes(model, names=PROJECTIONS):
    return {f'{i}.{name}': tensor_hash(getattr(layer.mlp, name).weight)
            for i, layer in enumerate(model.model.layers) for name in names}


@contextmanager
def edited(model, selection, seed, *, down_only=False, audit):
    """Back up just edited slices; hash entire edited tensors before/after restore.

    Restore even on an exception. No checkpoint copy, optimizer, or second model.
    """
    import torch
    names = ('down_proj',) if down_only else PROJECTIONS
    before, saved = {}, []
    for i, units in enumerate(selection):
        if len(units) != len(set(units)):
            raise ValueError('duplicate selected unit')
        for name in names:
            weight = getattr(model.model.layers[i].mlp, name).weight
            if any(j < 0 or j >= model.config.intermediate_size for j in units):
                raise ValueError('selected unit out of bounds')
            if units:
                before[f'{i}.{name}'] = tensor_hash(weight)
                part = weight[:, units] if name == 'down_proj' else weight[units]
                saved.append((i, name, units, part.detach().cpu().clone()))
    try:
        with torch.no_grad():
            for i, units in enumerate(selection):
                mlp = model.model.layers[i].mlp
                if down_only:
                    mlp.down_proj.weight[:, units] = 0
                else:
                    p.redraw(mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight,
                             units, seed, i, model.config.initializer_range)
        yield
    finally:
        with torch.no_grad():
            for i, name, units, original in saved:
                weight = getattr(model.model.layers[i].mlp, name).weight
                if name == 'down_proj':
                    weight[:, units] = original.to(weight.device)
                else:
                    weight[units] = original.to(weight.device)
        after = {f'{i}.{name}': tensor_hash(getattr(model.model.layers[i].mlp, name).weight)
                 for i, name, _, _ in saved}
        audit.update(before=before, restored=after, passed=before == after)
        if not audit['passed']:
            raise ValueError('edited tensor restoration hash mismatch')


def activation_stats(model, tokenizer, rows):
    """All sequence positions, including final EOS; raw abs activation, not utility.

    Peak ties take the first row in frozen order, then the first position. Prompt
    means use actual input prompt positions (0..prompt_length-1), not label masks.
    """
    import torch
    from kit.v4_teacher import tokenize_student
    layers = model.model.layers
    stats = [dict(total=torch.zeros(layer.mlp.intermediate_size, device=model.device),
                  prompt=torch.zeros(layer.mlp.intermediate_size, device=model.device),
                  peak=torch.full((layer.mlp.intermediate_size,), -1., device=model.device),
                  position=torch.zeros(layer.mlp.intermediate_size, dtype=torch.long, device=model.device),
                  row=torch.zeros(layer.mlp.intermediate_size, dtype=torch.long, device=model.device))
             for layer in layers]
    row_index, prompt_length = 0, 0
    handles = []
    for i, layer in enumerate(layers):
        def capture(module, args, index=i):
            activation = args[0][0].float().abs()
            s = stats[index]
            s['total'].add_(activation.sum(0))
            s['prompt'].add_(activation[:prompt_length].sum(0))
            peak, position = activation.max(0)
            higher = peak > s['peak']
            s['peak'].copy_(torch.maximum(peak, s['peak']))
            s['position'][higher] = position[higher]
            s['row'][higher] = row_index
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
    tokens, prompt_tokens, metadata = 0, 0, []
    model.eval()
    try:
        with torch.inference_mode():
            for row_index, row in enumerate(rows):
                pids, tids = tokenize_student(tokenizer, row['prompt'], row['response'])
                if not pids or not tids or len(pids)+len(tids) > model.config.max_position_embeddings:
                    raise ValueError('empty or over-context row; no truncation')
                prompt_length = len(pids)
                ids = torch.tensor([pids+tids], device=model.device)
                with torch.autocast(model.device.type, dtype=torch.bfloat16):
                    model.model(input_ids=ids, use_cache=False)
                tokens += len(pids)+len(tids)
                prompt_tokens += len(pids)
                metadata.append(dict(id=row.get('id', str(row_index)), prompt_length=len(pids),
                                     first_token_id=pids[0], response_length=len(tids)))
    finally:
        for handle in handles:
            handle.remove()
    result = []
    for s, layer in zip(stats, layers):
        utility = s['total']/tokens * layer.mlp.down_proj.weight.detach().float().abs().sum(0)
        result.append(dict(all_utility=utility.cpu().tolist(), max_abs=s['peak'].cpu().tolist(),
                           max_position=s['position'].cpu().tolist(), max_row=s['row'].cpu().tolist(),
                           prompt_mean_abs=(s['prompt']/prompt_tokens).cpu().tolist()))
    return result, dict(all_tokens=tokens, prompt_tokens=prompt_tokens, rows=metadata)


def selections(response, stats):
    """Counts floor(width/100); exclude ceil(width*.005) peaks, ties by unit index."""
    if len(response) != len(stats):
        raise ValueError('utility layer count mismatch')
    result = {v: [] for v in VARIANTS}
    excluded = []
    for layer, (values, stat) in enumerate(zip(response, stats)):
        width = len(values)
        if len(stat['all_utility']) != width or len(stat['max_abs']) != width:
            raise ValueError('utility shape mismatch')
        registered = p.lowest(values)
        count = len(registered)
        peak_order = sorted(range(width), key=lambda j: (-stat['max_abs'][j], j))
        banned = peak_order[:math.ceil(width*.005)]
        allowed = set(range(width))-set(banned)
        random_seed = hashlib.sha256(f'rung-p-diag|{layer}'.encode()).hexdigest()
        import torch
        generator = torch.Generator(device='cpu').manual_seed(int(random_seed[:16], 16))
        random = torch.randperm(width, generator=generator)[:count].tolist()
        choices = ([], registered, p.lowest(stat['all_utility']), random, registered,
                   sorted(allowed, key=lambda j: (values[j], j))[:count])
        for variant, units in zip(VARIANTS, choices):
            result[variant].append(list(units))
        excluded.append(banned)
    return result, excluded


def snapshot(model, tokenizer, prompts, settings):
    """Same forwards/generation as p_smoke.damage, cached on CPU for one model."""
    import torch
    from kit.v4_teacher import tokenize_student
    result = []
    signature = hashlib.sha256()
    model.eval()
    with torch.inference_mode():
        for row in prompts:
            pids, tids = tokenize_student(tokenizer, row['prompt'], row['response'])
            ids = torch.tensor([pids+tids], device=model.device)
            logits = smoke.response_logits(model, ids, len(pids))
            nll = torch.nn.functional.cross_entropy(logits[0], torch.tensor(tids, device=model.device),
                                                    reduction='sum').item()
            logits = logits.cpu()
            signature.update(bytes.fromhex(tensor_hash(logits)))
            ids = torch.tensor([pids], device=model.device)
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                **settings['greedy'], eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id)[0, len(pids):].cpu().tolist()
            signature.update(bytes.fromhex(p.digest(output)))
            result.append(dict(id=row['id'], logits=logits, nll=nll, tokens=len(tids), output=output,
                truncated=len(output) == settings['greedy']['max_new_tokens'] and
                          output[-1] != tokenizer.eos_token_id))
    return result, signature.hexdigest()


def compare(before, after):
    if len(before) != len(after):
        raise ValueError('prompt count mismatch')
    deltas, nll_sums, tokens, changed, truncated = [], [0., 0.], 0, 0, [0, 0]
    for a, b in zip(before, after):
        if (a['id'], a['tokens']) != (b['id'], b['tokens']):
            raise ValueError('prompt mismatch')
        deltas.append((a['logits']-b['logits']).abs().flatten())
        nll_sums = [x+y for x, y in zip(nll_sums, [a['nll'], b['nll']])]
        tokens += a['tokens']
        changed += a['output'] != b['output']
        truncated = [x+int(y) for x, y in zip(truncated, [a['truncated'], b['truncated']])]
    return dict(greedy_changed=changed, **smoke.delta_summary(deltas), response_tokens=tokens,
                response_nll_nats_per_token=[v/tokens for v in nll_sums],
                nll_rise_nats_per_token=(nll_sums[1]-nll_sums[0])/tokens,
                greedy_truncated=truncated)


def run_variants(model, tokenizer, prompts, settings, selection):
    baseline_hashes = projection_hashes(model)
    baseline, baseline_signature = snapshot(model, tokenizer, prompts, settings)
    results, audits = {}, {}
    for variant in VARIANTS:
        audit = {}
        print(f'[progress] diagnostic {variant}: {VARIANTS[variant]}', flush=True)
        with edited(model, selection[variant], settings['seed'], down_only=variant == 'V4', audit=audit):
            current, signature = snapshot(model, tokenizer, prompts, settings)
            metrics = compare(baseline, current)
            del current
        if any(baseline_hashes[k] != v for k, v in audit['restored'].items()):
            raise ValueError('restored tensors differ from initial model')
        audits[variant] = audit
        results[variant] = dict(description=VARIANTS[variant], damage=metrics,
            output_sha256=signature, selected_per_layer=[len(v) for v in selection[variant]],
            restoration=dict(passed=audit['passed'], tensors=len(audit['before']),
                             before_sha256=p.digest(audit['before']), restored_sha256=p.digest(audit['restored'])))
    final_hashes = projection_hashes(model)
    zero = results['V0']['damage']
    checks = dict(V0_exact_zero=all(zero[k] == 0 for k in
        ('greedy_changed', 'p99_logit_delta', 'p99_9_logit_delta', 'max_logit_delta', 'nll_rise_nats_per_token'))
        and results['V0']['output_sha256'] == baseline_signature,
        V4_equals_V1=results['V4']['damage'] == results['V1']['damage'] and
                     results['V4']['output_sha256'] == results['V1']['output_sha256'],
        restoration=baseline_hashes == final_hashes and all(a['passed'] for a in audits.values()))
    return results, checks, dict(initial=baseline_hashes, final=final_hashes, variants=audits)


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0)
    return dict(count=len(values), minimum=values[0], maximum=values[-1],
                mean=sum(values)/len(values), median=values[len(values)//2],
                p99=values[math.floor((len(values)-1)*.99)])


def layer_summaries(stats, selection, excluded, metadata):
    result = []
    for i, stat in enumerate(stats):
        groups = {}
        for name, units in [('all', range(len(stat['max_abs']))), ('V1', selection['V1'][i])]:
            positions = [stat['max_position'][j] for j in units]
            groups[name] = dict(max_abs=distribution([stat['max_abs'][j] for j in units]),
                prompt_mean_abs=distribution([stat['prompt_mean_abs'][j] for j in units]),
                peak_position=distribution(positions), peak_at_position_0=sum(v == 0 for v in positions),
                peak_in_first_4=sum(v < 4 for v in positions),
                peak_in_first_16=sum(v < 16 for v in positions),
                peak_in_prompt=sum(stat['max_position'][j] <
                    metadata['rows'][stat['max_row'][j]]['prompt_length'] for j in units))
        registered = set(selection['V1'][i])
        result.append(dict(layer=i, groups=groups, overlap_V1_V2=len(registered & set(selection['V2'][i])),
                           overlap_V1_V5=len(registered & set(selection['V5'][i])),
                           V1_in_excluded_peaks=len(registered & set(excluded[i]))))
    return result


def reproduce(metrics, expected, *, include_greedy=True):
    # Declared before a launch: exact token/count checks; 3% relative or .05 logit
    # absolute tolerance; .01 nats/token absolute tolerance for NLL and its rise.
    tolerances = {k: dict(relative=.03, absolute=.05) for k in
                  ('p99_logit_delta', 'p99_9_logit_delta', 'max_logit_delta')}
    tolerances['nll_rise_nats_per_token'] = dict(relative=0., absolute=.01)
    checks = {k: math.isclose(metrics[k], expected[k], rel_tol=t['relative'], abs_tol=t['absolute'])
              for k, t in tolerances.items()}
    for k in (('greedy_changed', 'response_tokens') if include_greedy else ('response_tokens',)):
        checks[k] = metrics[k] == expected[k]
    checks['response_nll_nats_per_token'] = all(abs(a-b) <= .01 for a, b in
        zip(metrics['response_nll_nats_per_token'], expected['response_nll_nats_per_token']))
    return dict(passed=all(checks.values()), checks=checks, tolerances=tolerances,
                response_nll_absolute_tolerance=.01, expected=expected)


def run(checkpoint, reference, output, arrays_path):
    """Write a compact result even if a prerequisite/check fails."""
    import json
    from pathlib import Path
    from transformers import AutoTokenizer
    record = dict(schema='rung-p-selection-diag.v1', diagnostic_only=True, training=False, passed=False)
    try:
        expected = json.loads(Path(reference).read_text())
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        settings, rows, prompts = smoke.frozen_inputs(tokenizer)
        if expected['settings_sha256'] != p.sha(smoke.RESOURCE/'smoke-settings.json'):
            raise ValueError('attempt 1 settings differ from frozen inputs')
        record.update(reference_sha256=p.sha(reference), settings_sha256=expected['settings_sha256'],
                      model=settings['model'], revision=settings['revision'],
                      pool_sha256=settings['pool_sha256'], prompts_sha256=settings['prompts_sha256'],
                      greedy=settings['greedy'], prompt_count=len(prompts),
                      selection_mask=settings['utility_mask'], damage_nll=settings['nll'],
                      damage_logits=settings['logits'],
                      all_position_definition='every input position, prompt plus response including final EOS',
                      peak_definition='abs input to down_proj; 0-based sequence position; first occurrence wins ties',
                      random_seed='SHA256(rung-p-diag|<decimal layer>), first 8 bytes unsigned big-endian; CPU torch.randperm',
                      exclusion='ceil(width*0.005) largest max abs activation; ties by unit index; select floor(width/100) survivors')
        model = smoke.load_model(checkpoint).eval().requires_grad_(False)
        print('[progress] measuring registered response utility', flush=True)
        response, tokens = p.measure(model, tokenizer, rows)
        print('[progress] measuring all-position utility and activation peaks', flush=True)
        stats, metadata = activation_stats(model, tokenizer, rows)
        selection, excluded = selections(response, stats)
        variants, checks, hashes = run_variants(model, tokenizer, prompts, settings, selection)
        reproduction = reproduce(variants['V1']['damage'], expected['damage'])
        checks['V1_reproduces_attempt1'] = reproduction['passed']
        arrays = p.write_receipt(arrays_path, dict(schema='rung-p-selection-diag-arrays.v1',
            response_utility=response, activation_stats=stats, selection=selection,
            excluded_peak_units=excluded, measurement=metadata, tensor_hashes=hashes,
            response_utility_tokens=tokens))
        record.update(variants=variants, checks=checks, reproduction=reproduction,
                      layers=layer_summaries(stats, selection, excluded, metadata),
                      response_utility_tokens=tokens, all_position_tokens=metadata['all_tokens'],
                      prompt_tokens=metadata['prompt_tokens'],
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
