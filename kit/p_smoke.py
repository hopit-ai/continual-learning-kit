#!/usr/bin/env python3
"""One-GPU technical checks for Rung P. Called only by the manager's Modal smoke."""
from pathlib import Path
import gc
import json
import math
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit import p_replace as p
from kit.p_determinism import configure

RESOURCE = Path(__file__).resolve().parent/'resources/p'
ASSERTIONS = ('unselected_bit_identical', 'selected_down_zero', 'nll_damage', 'logit_damage',
              'zero_lr_no_movement', 'down_moved_update_2', 'gate_up_zero_before_down_moves',
              'gate_up_gradients_update_3', 'finite_losses', 'finite_preclip_norms',
              'preclip_norms_within_2x', 'finite_ema', 'step_time_within_1_1x')


def frozen_inputs(tokenizer=None, settings_path=p.DEFAULT_SETTINGS):
    settings = json.loads(Path(settings_path).read_text())
    for name, key in [('smoke-finqa-train.jsonl', 'pool_sha256'), ('smoke-prompts.jsonl', 'prompts_sha256')]:
        if p.sha(RESOURCE/name) != settings[key]:
            raise ValueError('smoke input freeze changed: '+name)
    rows, sources = p.select_batch([('finqa', RESOURCE/'smoke-finqa-train.jsonl')], tokenizer=tokenizer)
    prompts = [json.loads(line) for line in (RESOURCE/'smoke-prompts.jsonl').read_text().splitlines()]
    if len(prompts) != 32 or len({r['id'] for r in prompts}) != 32:
        raise ValueError('exactly 32 frozen prompts required')
    if any(r['split'] != 'train' for r in prompts):
        raise ValueError('smoke prompts must be training-only')
    return settings, rows, prompts


def bit_equal(a, b):
    import torch
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def unchanged(checkpoint, replaced, receipt):
    """Check every stored tensor byte, allowing only the registered rows/columns."""
    import torch
    from safetensors.torch import load_file
    allowed = {}
    for r in receipt['replaced']:
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            allowed[f'model.layers.{r["layer"]}.mlp.{projection}.weight'] = r['indices']
    same, zero, checked = True, True, 0
    for name in receipt['input']['files']:
        if not name.endswith('.safetensors'):
            continue
        before, after = load_file(str(Path(checkpoint)/name)), load_file(str(Path(replaced)/name))
        same &= before.keys() == after.keys()
        for key, a in before.items():
            b = after[key]
            indices = allowed.get(key)
            if indices is None:
                same &= bit_equal(a, b)
                checked += a.numel()
            else:
                axis = 1 if '.down_proj.' in key else 0
                mask = torch.ones(a.shape[axis], dtype=torch.bool)
                mask[indices] = False
                aa, bb = (a[:, mask], b[:, mask]) if axis == 1 else (a[mask], b[mask])
                same &= bit_equal(aa, bb)
                checked += aa.numel()
                if axis == 1:
                    zero &= bool((b[:, indices] == 0).all())
        del before, after
    if not checked:
        raise ValueError('smoke requires safetensors weights')
    return dict(unselected_bit_identical=dict(value=bool(same), checked_elements=checked, status='PASS' if same else 'FAIL'),
                selected_down_zero=dict(value=bool(zero), status='PASS' if zero else 'FAIL'))


def load_model(checkpoint, device='cuda', attention_implementation='sdpa'):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation=attention_implementation).to(device)


def damage_assertions(metrics, settings):
    nll, delta = metrics['nll_rise_nats_per_token'], metrics['p99_9_logit_delta']
    return {key: dict(value=value, status='PASS' if math.isfinite(value) and value <= limit else 'FAIL')
            for key, value, limit in (
                ('nll_damage', nll, settings['limits']['max_nll_rise_nats_per_token']),
                ('logit_damage', delta, settings['limits']['max_p99_9_logit_delta']))}


def delta_summary(deltas):
    """Exact pooled entry quantiles, linear interpolation; each logit weighted equally."""
    import torch
    values = torch.cat(deltas).sort().values
    if not torch.isfinite(values).all():
        return dict(p99_logit_delta=math.inf, p99_9_logit_delta=math.inf, max_logit_delta=math.inf)
    # torch.quantile has a 2**24-element limit; select interpolation ranks directly.
    def quantile(q):
        rank = (values.numel()-1)*q
        lo, hi = math.floor(rank), math.ceil(rank)
        return values[lo].item()*(hi-rank) + values[hi].item()*(rank-lo) if hi != lo else values[lo].item()
    return dict(p99_logit_delta=quantile(.99), p99_9_logit_delta=quantile(.999),
                max_logit_delta=values[-1].item())


def response_logits(model, ids, prompt_length):
    # Qwen's backbone returns the final, normalised hidden state. Bypass the bf16
    # LM head and project response prediction positions only, with autocast off.
    import torch
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, prompt_length-1:-1]
    with torch.autocast(ids.device.type, enabled=False):
        return torch.nn.functional.linear(hidden.float(), model.lm_head.weight.float(),
            None if model.lm_head.bias is None else model.lm_head.bias.float())


def damage(checkpoint, replaced, tokenizer, prompts, settings, device='cuda'):
    configure(settings)
    import torch
    from kit.v4_teacher import tokenize_student
    attention = settings.get('attention_implementation', 'sdpa')
    before, after = (load_model(path, device, attention).eval() for path in (checkpoint, replaced))
    changed, details, all_deltas, nll_sums, tokens = 0, [], [], [0., 0.], 0
    with torch.inference_mode():
        for row in prompts:
            pids, tids = tokenize_student(tokenizer, row['prompt'], row['response'])
            ids = torch.tensor([pids+tids], device=device)
            a, b = (response_logits(model, ids, len(pids)) for model in (before, after))
            delta = (a-b).abs().flatten().cpu()
            all_deltas.append(delta)
            targets = torch.tensor(tids, device=device)
            nll = [torch.nn.functional.cross_entropy(logits[0], targets, reduction='sum').item()
                   for logits in (a, b)]
            nll_sums = [x+y for x,y in zip(nll_sums,nll)]
            tokens += len(tids)
            del a, b
            ids = torch.tensor([pids], device=device)
            outputs = []
            for model in (before, after):
                outputs.append(model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    **settings['greedy'], eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id)[0, len(pids):].cpu().tolist())
            different = outputs[0] != outputs[1]
            changed += different
            details.append(dict(id=row['id'], changed=different, **delta_summary([delta]),
                                response_tokens=len(tids), response_nll_nats_per_token=[v/len(tids) for v in nll],
                                token_ids=outputs, truncated=[len(o) == settings['greedy']['max_new_tokens'] and
                                    o[-1] != tokenizer.eos_token_id for o in outputs]))
    metrics = dict(greedy_changed=changed, **delta_summary(all_deltas), response_tokens=tokens,
                   response_nll_nats_per_token=[v/tokens for v in nll_sums],
                   nll_rise_nats_per_token=(nll_sums[1]-nll_sums[0])/tokens)
    del before, after
    gc.collect(); torch.cuda.empty_cache()
    return metrics, details


def selected_values(model, records, gradients=False):
    """Per-unit norms: every selected unit must pass, not just a layer aggregate."""
    result = {}
    for record in records:
        layer = record['layer']; mlp = model.model.layers[layer].mlp
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            weight = getattr(mlp, name).weight
            value = weight.grad if gradients else weight.detach()
            if value is None:
                result[f'{layer}.{name}'] = [0.] * len(record['indices'])
                continue
            value = value[:, record['indices']].float() if name == 'down_proj' else value[record['indices']].float().T
            result[f'{layer}.{name}'] = value.norm(dim=0).cpu().tolist()
    return result


def train(checkpoint, tokenizer, rows, records, settings, device='cuda'):
    """Full-weight CE fallback: bf16 actor, CPU fp32 AdamW master weights and EMA.

    CPU masters avoid the loss of LR=1e-6 updates in bf16 weights. The fresh EMA
    starts AFTER replacement. This is a mechanical smoke, not scientific D.
    """
    configure(settings)  # reset torch/Python/NumPy to the same seed for EACH arm
    import torch
    from kit.v4_teacher import tokenize_student, RESPONSE_CAP
    from kit.c_d_prompt import d_prompt_eligible
    torch.manual_seed(settings['seed'])
    net = load_model(checkpoint, device, settings.get('attention_implementation', 'sdpa'))
    net.config.use_cache = False
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    net.train()
    weights = list(net.parameters())
    masters = [torch.nn.Parameter(w.detach().cpu().float()) for w in weights]
    ema = [w.detach().clone() for w in masters]
    optimizer = torch.optim.AdamW(masters, lr=1e-5, betas=(.9, .999), weight_decay=.01, foreach=False)
    metrics = []
    initial_selected = {}
    for r in records:
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            key = f'{r["layer"]}.{name}'
            w = getattr(net.model.layers[r['layer']].mlp, name).weight.detach()
            initial_selected[key] = (w[:, r['indices']] if name == 'down_proj' else w[r['indices']]).cpu().clone()
    for step, lr in enumerate(settings['learning_rates']):
        if device == 'cuda': torch.cuda.synchronize()
        clock = time.monotonic()
        net.zero_grad(set_to_none=True); optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        for row in rows[step*32:(step+1)*32]:
            pids, tids = tokenize_student(tokenizer, row['prompt'], row['response'])
            if not (pids and d_prompt_eligible(pids) and 0 < len(tids) <= RESPONSE_CAP):
                raise ValueError('smoke training caps exceeded; no truncation')
            ids = torch.tensor([pids+tids], device=device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = net(input_ids=ids, use_cache=False).logits[:, len(pids)-1:-1]
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                    torch.tensor(tids, device=device))/32
            loss.backward()
            loss_value += loss.detach().item()
            del logits, loss
        selected_grad = selected_values(net, records, gradients=True)
        # Accumulate squared norm in fp32, before clipping, as in the real trainer.
        norm = torch.stack([w.grad.detach().float().square().sum() for w in weights if w.grad is not None]).sum().sqrt().item()
        scale = min(1., settings['gradient_clip']/(norm+1e-6))
        for w, master in zip(weights, masters):
            if w.grad is not None:
                master.grad = w.grad.detach().to(device='cpu', dtype=torch.float32).mul_(scale)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        finite_ema = True
        with torch.no_grad():
            for w, master, teacher in zip(weights, masters, ema):
                w.copy_(master)
                teacher.lerp_(master, settings['ema_rate'])
                finite_ema &= bool(torch.isfinite(teacher).all())
        distances = {}
        with torch.no_grad():
            for r in records:
                for name in ('gate_proj', 'up_proj', 'down_proj'):
                    key = f'{r["layer"]}.{name}'
                    w = getattr(net.model.layers[r['layer']].mlp, name).weight.detach()
                    part = w[:, r['indices']] if name == 'down_proj' else w[r['indices']]
                    delta = part.cpu().float()-initial_selected[key].float()
                    distances[key] = delta.norm(dim=0 if name == 'down_proj' else 1).tolist()
        if device == 'cuda': torch.cuda.synchronize()
        metrics.append(dict(update=step+1, lr=lr, loss=loss_value, preclip_grad_norm=norm,
                            selected_task_grad_norms=selected_grad, selected_distances=distances,
                            ema_finite=finite_ema, seconds=time.monotonic()-clock))
        print('[progress] '+json.dumps(dict(arm=str(Path(checkpoint).name), **{k: v for k, v in metrics[-1].items()
                              if k not in ('selected_task_grad_norms', 'selected_distances')})), flush=True)
    del optimizer, masters, ema, weights, net
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    return metrics


def learning_assertions(control, replaced):
    def values(row, field, down):
        return [v for key, vv in row[field].items() if ('.down_proj' in key) == down for v in vv]
    ratios = [b['preclip_grad_norm']/a['preclip_grad_norm'] if a['preclip_grad_norm'] > 0 else math.inf
              for a, b in zip(control, replaced)]
    timing = statistics.median(r['seconds'] for r in replaced[1:])/statistics.median(r['seconds'] for r in control[1:])
    checks = {}
    def add(name, value, passed):
        checks[name] = dict(value=value, status='PASS' if passed else 'FAIL')
    zero = [v for vv in replaced[0]['selected_distances'].values() for v in vv]
    add('zero_lr_no_movement', zero, bool(zero) and all(v == 0 for v in zero))
    down = values(replaced[1], 'selected_distances', True)
    add('down_moved_update_2', down, bool(down) and all(v > 0 for v in down))
    initial = values(replaced[0], 'selected_task_grad_norms', False)+values(replaced[1], 'selected_task_grad_norms', False)
    add('gate_up_zero_before_down_moves', initial, bool(initial) and all(v == 0 for v in initial))
    gate = values(replaced[2], 'selected_task_grad_norms', False)
    add('gate_up_gradients_update_3', gate, bool(gate) and all(v > 0 and math.isfinite(v) for v in gate))
    losses = [r['loss'] for r in control+replaced]
    add('finite_losses', losses, all(math.isfinite(v) for v in losses))
    norms = [r['preclip_grad_norm'] for r in control+replaced]
    add('finite_preclip_norms', norms, all(v > 0 and math.isfinite(v) for v in norms))
    add('preclip_norms_within_2x', ratios, all(.5 <= r <= 2 for r in ratios))
    ema = [r['ema_finite'] for r in control+replaced]
    add('finite_ema', ema, all(ema))
    add('step_time_within_1_1x', timing, timing <= 1.1)
    return checks


def validate_utility(utility, settings, rows):
    if (len(utility['pools']) != 1 or utility['pools'][0]['sha256'] != settings['pool_sha256'] or
            [s['id'] for s in utility['selection']] != [r['id'] for r in rows]):
        raise ValueError('utility receipt differs from frozen smoke pool/selection')
    if utility.get('positions', 'response') != settings.get('utility_positions', 'response'):
        raise ValueError('utility positions differ from frozen smoke settings')


def validate_replacement(utility, replacement, settings):
    fraction = settings.get('fraction', .01)
    if (replacement.get('fraction', .01) != fraction or
            replacement.get('positions', 'response') != settings.get('utility_positions', 'response') or
            [(r['layer'], r['indices']) for r in replacement['replaced']] !=
            [(i, p.lowest(v, fraction)) for i, v in enumerate(utility['utilities'])]):
        raise ValueError('replacement selection differs from frozen smoke fraction/positions')


def receipt_summaries(utility_path, replacement_path, utility, replacement):
    return dict(
        utility=dict(file='utility.json', sha256=p.sha(utility_path), receipt_sha256=utility['receipt_sha256'],
            selected_rows=len(utility['selection']), positions=utility.get('positions', 'response'),
            **{k: utility[k] for k in ('utility_tokens', 'response_tokens', 'all_tokens') if k in utility},
            layers=[dict(units=len(v), minimum=min(v), maximum=max(v), mean=statistics.mean(v))
                    for v in utility['utilities']]),
        replacement=dict(file='replacement.json', sha256=p.sha(replacement_path),
            receipt_sha256=replacement['receipt_sha256'],
            fraction=replacement.get('fraction', .01),
            selected_units_per_layer=[len(r['indices']) for r in replacement['replaced']]))


def run(checkpoint, replaced, utility, receipt, output, settings_path=p.DEFAULT_SETTINGS):
    """Always produce one assertion record, including failures before training."""
    record = dict(schema='rung-p-smoke.v1', assertions={k: dict(status='FAIL', value=None,
        reason='not reached') for k in ASSERTIONS}, scientific_d_run=False)
    clock = time.monotonic()
    try:
        settings = json.loads(Path(settings_path).read_text())
        record['settings'] = settings
        record.update(configure(settings))
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        settings, rows, prompts = frozen_inputs(tok, settings_path)
        record['settings'] = settings
        record['code_sha256'] = {name:p.sha(Path(__file__).with_name(name))
            for name in ('p_replace.py', 'p_smoke.py', 'p_determinism.py')}
        record['settings_sha256'] = p.sha(settings_path)
        u, r = p.read_receipt(utility), p.read_receipt(receipt)
        validate_utility(u, settings, rows)
        validate_replacement(u, r, settings)
        if r.get('settings_sha256', record['settings_sha256']) != record['settings_sha256']:
            raise ValueError('replacement used different smoke settings')
        record['receipts'] = receipt_summaries(utility, receipt, u, r)
        if p.checkpoint_hash(checkpoint) != r['input'] or p.checkpoint_hash(replaced) != r['output']:
            raise ValueError('smoke checkpoints differ from replacement receipt')
        if r['utility_receipt_sha256'] != u['receipt_sha256']:
            raise ValueError('replacement used different utility')
        record['assertions'].update(unchanged(checkpoint, replaced, r))
        metrics, details = damage(checkpoint, replaced, tok, prompts, settings)
        record['damage'] = metrics
        record['comparisons'] = details
        record['assertions'].update(damage_assertions(metrics, settings))
        record['control'] = train(checkpoint, tok, rows[:128], r['replaced'], settings)
        record['replaced'] = train(replaced, tok, rows[:128], r['replaced'], settings)
        record['assertions'].update(learning_assertions(record['control'], record['replaced']))
    except Exception as exc:
        record['error'] = type(exc).__name__+': '+str(exc)
    record['passed'] = all(v['status'] == 'PASS' for v in record['assertions'].values())
    record['seconds'] = time.monotonic()-clock
    # Nonfinite measurements are explicit strings in strict JSON, and remain FAIL.
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        return value
    record = finite_json(record)
    Path(output).write_text(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
    return record


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    for arg in ('checkpoint', 'replaced', 'utility', 'receipt', 'output'):
        parser.add_argument('--'+arg, required=True, type=Path)
    parser.add_argument('--settings', type=Path, default=p.DEFAULT_SETTINGS)
    a = parser.parse_args()
    result = run(a.checkpoint, a.replaced, a.utility, a.receipt, a.output, a.settings)
    raise SystemExit(0 if result['passed'] else 1)
