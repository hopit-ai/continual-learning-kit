#!/usr/bin/env python3
"""Rung P's once-at-boundary contribution utility and deterministic unit replacement.

Offline only. Receipts are JSON, including all layer arrays. Checkpoint hashes bind
HF inference files by relative filename and SHA256, excluding caches and receipts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SALT = 'rung-p-v1'
DEFAULT_SETTINGS = Path(__file__).resolve().parent/'resources/p/smoke-settings.json'
HF_FILES = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
            'tokenizer.model', 'chat_template.jinja'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def weight_files(root):
    """Use one HF weight format, never trainer state such as training_args.bin."""
    root = Path(root)
    for name, pattern in (('model.safetensors.index.json', 'model*.safetensors'),
                          ('pytorch_model.bin.index.json', 'pytorch_model*.bin')):
        if (root/name).exists():
            weights = sorted(set(json.loads((root/name).read_text())['weight_map'].values()))
            if not weights or any(Path(n).name != n or not (root/n).is_file() for n in weights):
                raise ValueError('invalid or missing indexed weight shard')
            return weights, [name]
        weights = sorted(p.name for p in root.glob(pattern) if p.is_file())
        if weights:
            return weights, []
    raise ValueError('incomplete HF checkpoint')


def checkpoint_hash(root):
    root = Path(root)
    weights, indexes = weight_files(root)
    files = {p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file() and
             (p.name in HF_FILES or p.name in weights or p.name in indexes)}
    if 'config.json' not in files:
        raise ValueError('incomplete HF checkpoint')
    return dict(sha256=digest(files), files=files)


def write_receipt(path, doc):
    doc = dict(doc, receipt_sha256=digest(doc))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(doc, f, indent=1, sort_keys=True, allow_nan=False)
        f.write('\n')
    return doc


def read_receipt(path):
    doc = json.loads(Path(path).read_text())
    expected = doc.pop('receipt_sha256')
    if digest(doc) != expected:
        raise ValueError('utility receipt hash mismatch')
    return dict(doc, receipt_sha256=expected)


def messages(prompt):
    if isinstance(prompt, str):
        try:
            parsed = json.loads(prompt)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list) and parsed and all(isinstance(m, dict) and
                isinstance(m.get('role'), str) and isinstance(m.get('content'), str) for m in parsed):
            return parsed
        return [dict(role='user', content=prompt)]
    return prompt


def select_batch(pools, replay_pools=(), tokenizer=None, eligibility=None):
    """Hash-order eligible targets. Without a tokenizer, inspect provenance only.

    Production utility always supplies the stage tokenizer before selecting.
    """
    from kit.c_d_prompt import d_prompt_eligible
    from kit.v4_teacher import tokenize_student
    rows, sources, seen_pools, seen_ids = [], [], set(), set()
    replay_pools = set(replay_pools)
    for name, path in pools:
        if not name or '|' in name or name in seen_pools:
            raise ValueError('pool names must be unique, nonempty and contain no |')
        seen_pools.add(name)
        sources.append(dict(pool=name, sha256=sha(path)))
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            extra = r.get('extra_info') or {}
            ident = str(r.get('id', extra.get('index', '')))
            key = name + '|' + ident
            if not ident or key in seen_ids:
                raise ValueError('missing or duplicate pool ID')
            seen_ids.add(key)
            if r.get('split', extra.get('split')) != 'train' or r.get('scored_panel') or r.get('held_out'):
                raise ValueError('utility pools must contain training rows only')
            response = r.get('response', extra.get('demonstration'))
            if not isinstance(response, str) or not response or 'prompt' not in r:
                raise ValueError('utility rows need prompt and frozen response')
            prompt = messages(r['prompt'])
            replay = name in replay_pools or bool(r.get('p3b_replay', extra.get('p3b_replay', False)))
            if tokenizer is not None:
                prompt_ids = tokenize_student(tokenizer, prompt, response)[0]
                n = len(prompt_ids)
                capped = replay and not d_prompt_eligible(prompt_ids)
                if eligibility is not None:
                    eligibility.append(dict(id=key,prompt_tokens=n,replay=replay,
                        cap=2048 if replay else None,capped=capped))
                if capped: continue
            order = hashlib.sha256((SALT + '|' + name + '|' + ident).encode()).hexdigest()
            rows.append(dict(id=key, order_sha256=order, row_sha256=digest(r),
                             prompt=prompt, response=response,
                             p3b_replay=replay))
    if replay_pools - seen_pools:
        raise ValueError('unknown replay pool name')
    if len(rows) < 256:
        raise ValueError('utility batch requires at least 256 eligible rows; no substitution')
    return sorted(rows, key=lambda r: (r['order_sha256'], r['id']))[:256], sorted(sources, key=lambda s: s['pool'])


def replacement_spec(settings=DEFAULT_SETTINGS):
    """Attempt 1 predates explicit fields; its frozen defaults remain registered."""
    doc = json.loads(Path(settings).read_text())
    positions, fraction = doc.get('utility_positions', 'response'), doc.get('fraction', .01)
    if positions not in ('response', 'all') or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('invalid replacement settings')
    return positions, fraction


def lowest(values, fraction=.01):
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('invalid utility array')
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('invalid replacement fraction')
    return sorted(range(len(values)), key=lambda j: (values[j], j))[:math.floor(len(values)*fraction)]


def unit_seed(seed, layer, unit):
    material = f'{SALT}|{seed}|{layer}|{unit}'
    hexdigest = hashlib.sha256(material.encode()).hexdigest()
    return dict(material=material, sha256=hexdigest, torch_seed=int(hexdigest[:16], 16))


def redraw(gate, up, down, units, seed, layer, std):
    """CPU fp32 normal draws, gate then up from one independent stream per unit."""
    import torch
    seeds = []
    with torch.no_grad():
        for unit in units:
            s = unit_seed(seed, layer, unit)
            generator = torch.Generator(device='cpu').manual_seed(s['torch_seed'])
            for weight in (gate, up):
                row = torch.empty(weight.shape[1], dtype=torch.float32).normal_(0., std, generator=generator)
                weight[unit].copy_(row.to(device=weight.device, dtype=weight.dtype))
            down[:, unit] = 0
            seeds.append(s)
    return seeds


def measure(model, tokenizer, rows, positions='response'):
    """Observe the actual bf16 MLP activation entering down_proj; accumulate fp32.

    The activation is SiLU(gate(x))*up(x). Mask the causal prediction positions
    corresponding to valid response labels: D skips the first three task tokens;
    replay CE includes all response labels including EOS. In all mode, use every
    actual input position (prompt plus response, including final EOS), exactly
    p_diag.activation_stats' V2 all_utility accumulation and normalization.
    """
    import torch
    from kit.v4_teacher import tokenize_student
    from kit.sdft_objective import loss_mask
    if positions not in ('response', 'all'):
        raise ValueError('invalid utility positions')
    if model.config.model_type != 'qwen3' or model.config.hidden_act != 'silu':
        raise ValueError('requires Qwen3 SiLU MLP')
    layers = model.model.layers
    sums = [torch.zeros(m.mlp.intermediate_size, device=m.mlp.down_proj.weight.device,
                        dtype=torch.float32) for m in layers]
    mask = None
    handles = []
    for i, layer in enumerate(layers):
        def capture(module, args, index=i):
            # The down-projection input is exactly the multiplicative MLP activation.
            activation = args[0][0] if positions == 'all' else args[0][mask]
            sums[index].add_(activation.float().abs().sum(dim=0))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
    tokens = 0
    model.eval()
    try:
        with torch.inference_mode():
            for row in rows:
                pids, tids = tokenize_student(tokenizer, row['prompt'], row['response'])
                if not pids or not tids or len(pids)+len(tids) > model.config.max_position_embeddings:
                    raise ValueError('empty or over-context utility row; no truncation')
                ids = torch.tensor([pids+tids], device=model.device)
                if positions == 'response':
                    mask = torch.zeros_like(ids, dtype=torch.bool)
                    response_mask = torch.ones((1, len(tids)), device=model.device, dtype=torch.bool)
                    if not row.get('p3b_replay'):
                        response_mask = loss_mask(response_mask).bool()
                    mask[:, len(pids)-1:-1] = response_mask
                # Avoid the full vocabulary head; all MLP layers still execute.
                with torch.autocast(model.device.type, dtype=torch.bfloat16):
                    model.model(input_ids=ids, use_cache=False)
                tokens += len(pids)+len(tids) if positions == 'all' else int(response_mask.sum())
    finally:
        for handle in handles:
            handle.remove()
    if not tokens:
        raise ValueError('utility batch has no tokens under the stage loss mask')
    utilities = [(total / tokens * layer.mlp.down_proj.weight.detach().float().abs().sum(dim=0)).cpu().tolist()
                 for total, layer in zip(sums, layers)]
    return utilities, tokens


def utility(checkpoint, pools, output, device, replay_pools=(), positions='response', settings=None):
    from kit.p_determinism import configure
    policy = json.loads(Path(settings).read_text()) if settings is not None else {}
    runtime = configure(policy)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    select_batch(pools, replay_pools)  # provenance/count failures before loading tokenizer
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    eligibility = []
    rows, sources = select_batch(pools, replay_pools, tokenizer=tok, eligibility=eligibility)
    before = checkpoint_hash(checkpoint)
    attention = {'attn_implementation': policy['attention_implementation']} if runtime else {}
    net = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                                             torch_dtype=torch.bfloat16, **attention).to(device)
    utilities, tokens = measure(net, tok, rows, positions=positions)
    if checkpoint_hash(checkpoint) != before:
        raise ValueError('checkpoint changed during utility measurement')
    return write_receipt(output, dict(schema='rung-p-utility.v1', checkpoint=before, pools=sources,
        selection=[{k: r[k] for k in ('id', 'order_sha256', 'row_sha256', 'p3b_replay')} for r in rows],
        utilities=utilities, positions=positions, utility_tokens=tokens,
        **({'all_tokens': tokens} if positions == 'all' else {'response_tokens': tokens}),
        forward_dtype='bfloat16', accumulation_dtype='float32', **runtime,
        mask=('every input position, prompt plus response including final EOS; equal token weight'
              if positions == 'all' else
              'causal response prediction positions; kit.sdft_objective.loss_mask for task; all EOS-inclusive tokens for replay; equal token weight'),
        replay_pools=sorted(replay_pools),
        prompt_eligibility=eligibility,
        prompt_cap_rule='D+replay: only replay rows skip student prompts >2048 tokens; task targets are not replay-capped',
        template='kit.v4_teacher.tokenize_student; enable_thinking=False', measured_before_replacement=True))


def replace(checkpoint, utility_path, seed, output, receipt, fraction=.01, settings=DEFAULT_SETTINGS):
    import torch
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open
    checkpoint, output = Path(checkpoint), Path(output)
    if output.resolve() == checkpoint.resolve() or checkpoint.resolve() in output.resolve().parents:
        raise ValueError('output must be separate from input checkpoint')
    if Path(receipt).resolve() == Path(utility_path).resolve():
        raise ValueError('replacement receipt must be separate from utility receipt')
    if output.exists() or Path(receipt).exists():
        raise ValueError('use fresh output and receipt paths')
    doc = read_receipt(utility_path)
    expected_positions, expected_fraction = replacement_spec(settings)
    if fraction != expected_fraction or doc.get('positions', 'response') != expected_positions:
        raise ValueError('replacement fraction/utility positions differ from settings')
    before = checkpoint_hash(checkpoint)
    if doc['schema'] != 'rung-p-utility.v1' or doc['checkpoint'] != before or not doc['measured_before_replacement']:
        raise ValueError('utility was not measured on this untouched checkpoint')
    config = json.loads((checkpoint/'config.json').read_text())
    values = doc['utilities']
    if config['model_type'] != 'qwen3' or len(values) != config['num_hidden_layers'] or any(len(v) != config['intermediate_size'] for v in values):
        raise ValueError('utility shape/model mismatch')
    selected = [lowest(v, fraction) for v in values]
    records = [dict(layer=i, indices=units, utilities=[values[i][j] for j in units],
                    seeds=[unit_seed(seed, i, j) for j in units]) for i, units in enumerate(selected)]
    output.mkdir(parents=True, exist_ok=False)
    found = set()
    # Preserve original shard names, tensor dtypes, index and tokenizer bytes.
    for name in before['files']:
        source, target = checkpoint/name, output/name
        if not name.endswith(('.safetensors', '.bin')):
            shutil.copyfile(source, target)
            continue
        state = load_file(str(source)) if name.endswith('.safetensors') else torch.load(source, map_location='cpu', weights_only=True)
        for layer, units in enumerate(selected):
            for projection in ('gate_proj', 'up_proj', 'down_proj'):
                key = f'model.layers.{layer}.mlp.{projection}.weight'
                if key not in state:
                    continue
                found.add(key)
                weight = state[key]
                with torch.no_grad():
                    for j in units:
                        s = unit_seed(seed, layer, j)
                        g = torch.Generator(device='cpu').manual_seed(s['torch_seed'])
                        if projection == 'down_proj':
                            weight[:, j] = 0
                        else:
                            row = torch.empty(config['hidden_size'], dtype=torch.float32).normal_(0., config['initializer_range'], generator=g)
                            if projection == 'up_proj':
                                row.normal_(0., config['initializer_range'], generator=g)
                            weight[j].copy_(row.to(weight.dtype))
        if name.endswith('.safetensors'):
            with safe_open(str(source), framework='pt') as f:
                metadata = f.metadata()
            save_file(state, str(target), metadata=metadata)
        else:
            torch.save(state, target)
        del state
    if len(found) != len(values)*3:
        raise ValueError('checkpoint missing MLP projection weights')
    after = checkpoint_hash(output)
    if checkpoint_hash(checkpoint) != before:
        raise ValueError('input checkpoint changed')
    return write_receipt(receipt, dict(schema='rung-p-replace.v1', seed=seed, input=before, output=after,
        fraction=fraction, positions=expected_positions, settings_sha256=sha(settings),
        utility_receipt_sha256=doc['receipt_sha256'], replaced=records,
        seed_mapping='SHA256 first 8 bytes, unsigned big-endian; CPU fp32 normal, gate then up; cast to original dtype'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    u = sub.add_parser('utility')
    u.add_argument('--checkpoint', type=Path, required=True)
    u.add_argument('--pool', action='append', required=True, metavar='NAME=PATH')
    u.add_argument('--replay-pool', action='append', default=[], metavar='NAME',
                   help='mark a named pool as replay CE (all tokens, including EOS); repeat for each replay pool')
    u.add_argument('--output', type=Path, required=True)
    u.add_argument('--device', default='cuda')
    u.add_argument('--positions', choices=('response', 'all'), default='response')
    u.add_argument('--settings', type=Path, help='attempt 3 process execution policy')
    r = sub.add_parser('replace')
    r.add_argument('--checkpoint', type=Path, required=True)
    r.add_argument('--utility', type=Path, required=True)
    r.add_argument('--seed', type=int, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--receipt', type=Path, required=True)
    r.add_argument('--fraction', type=float, default=.01)
    r.add_argument('--settings', type=Path, default=DEFAULT_SETTINGS,
                   help='frozen settings binding the replacement fraction and utility positions')
    a = p.parse_args(argv)
    if a.action == 'utility':
        policy = {'settings': a.settings} if a.settings is not None else {}
        return utility(a.checkpoint, [v.split('=', 1) for v in a.pool], a.output, a.device,
                       a.replay_pool, positions=a.positions, **policy)
    return replace(a.checkpoint, a.utility, a.seed, a.output, a.receipt,
                   fraction=a.fraction, settings=a.settings)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
