"""Container-only 0.6B mechanics fixture using D's actual update_policy.

Execute with torchrun, two ranks. This module has no Modal dependency.
Synthetic arithmetic prompts, short widths and greedy sharded HF sampling are
engineering fixtures. The dense export is also exercised by vLLM at TP=2.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'e6de91484c29aa9480d55605af694f39b081c455'


def fingerprint(value):
    """Canonical local tensor/state digest, independent of torch.save encoding."""
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if hasattr(item, 'to_local'):
            item = item.to_local()
        if isinstance(item, torch.Tensor):
            x = item.detach().cpu().contiguous()
            digest.update(str((x.dtype, tuple(x.shape))).encode())
            digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif hasattr(item, 'tobytes') and hasattr(item, 'dtype'):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def frozen(model):
    return fingerprint({n: p for n, p in model.named_parameters() if not n.endswith('.diff_mask.gate')})


def distributed_mask_oracle(mesh):
    """Compare the actual two-rank FSDP mask gradient with the dense chain rule."""
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from deletion_tuning.deletion_mask import MaskedLinear
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(101)
        layer = MaskedLinear(torch.nn.Linear(3, 4, bias=False, device='cuda'))
    with torch.no_grad():
        layer.diff_mask.gate.copy_(torch.linspace(-.01, .02, 12, device='cuda').reshape(4, 3))
    weight = layer.weight.detach().clone()
    soft = torch.sigmoid(100 * layer.diff_mask.gate.detach())
    kwargs = dict(mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32),
                  reshard_after_forward=True)
    fully_shard(layer.diff_mask, **kwargs)
    fully_shard(layer, **kwargs)
    x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3) / 10 + dist.get_rank()
    layer(x).sum().backward()
    coefficient = x.sum(0)
    dist.all_reduce(coefficient)
    coefficient /= dist.get_world_size()
    expected = weight * soft * (1-soft) * 100 * coefficient[None]
    actual = layer.diff_mask.gate.grad.full_tensor()
    error = float((actual-expected).norm() / expected.norm().clamp(min=1e-12))
    if not torch.isfinite(actual).all() or error > 1e-4:
        raise ValueError('distributed mask gradient oracle failed')
    return error


class Batch:
    """The small DataProto interface D consumes, keeping its loss code intact."""
    def __init__(self, batch, extra, temperature=1.):
        self.batch = batch
        self.non_tensor_batch = {'extra_info': extra}
        self.meta_info = {'temperature': temperature}

    def split(self, size):
        return [Batch({k: v[start:start + size] for k, v in self.batch.items()},
                      self.non_tensor_batch['extra_info'][start:start + size])
                for start in range(0, len(self.non_tensor_batch['extra_info']), size)]

    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.batch.items()}, self.non_tensor_batch['extra_info'])


def batch_for(actor, tokenizer, rank):
    import torch
    from kit.sdft_actor import forward_micro_batch
    from kit.sdft_objective import build_teacher_inputs
    prompts = [[dict(role='user', content=f'What is {i + rank * 16} plus one?')]
               for i in range(16)]
    texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for p in prompts]
    encoded = tokenizer(texts, padding=True, return_tensors='pt').to('cuda')
    ids, attention = encoded.input_ids, encoded.attention_mask
    actor.actor_module.eval()
    # Both ranks take identical collective counts. Every generated token uses
    # the current hard mask and the real FSDP2 all-gather/reshard path.
    with torch.no_grad():
        for _ in range(8):
            positions = (attention.cumsum(-1) - 1).clamp(min=0)
            token = actor.actor_module(input_ids=ids, attention_mask=attention,
                                       position_ids=positions, use_cache=False).logits[:, -1].argmax(-1)
            ids = torch.cat((ids, token[:, None]), -1)
            attention = torch.cat((attention, torch.ones_like(token[:, None])), -1)
    responses = ids[:, -8:]
    response_mask = torch.ones_like(responses)
    values = dict(input_ids=ids, attention_mask=attention, position_ids=(attention.cumsum(-1)-1).clamp(min=0),
                  responses=responses, response_mask=response_mask)
    old = forward_micro_batch(actor, values, 1.)['log_probs']
    values.update(old_log_probs=old, rollout_log_probs=old.clone())
    demos = [f'The answer is {i + rank * 16 + 1}.' for i in range(16)]
    teacher = build_teacher_inputs(tokenizer, prompts, demos, responses, response_mask,
                                   chat_template_kwargs={'enable_thinking': False}, max_reprompt_len=256)
    for key in ('teacher_input_ids', 'teacher_attention_mask', 'teacher_position_ids', 'self_distillation_mask'):
        values[key] = teacher[key].to('cuda')
    # The global batch is 28 task +4 replay, interleaved across the two ranks.
    extra = [dict(p3b_replay=i in (7, 15), p3b_target_ids=tokenizer.encode(demos[i], add_special_tokens=False)
                  + [tokenizer.eos_token_id]) for i in range(16)]
    return Batch(values, extra)


def train(leg, work):
    import random
    from types import SimpleNamespace
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions, get_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deletion_tuning.deletion_mask import (hard_state, layers, assert_matching_masks, mask_bits, mask_counts)
    from deletion_tuning.deletion_actor import update_policy
    from deletion_tuning.deletion_prepare import prepare
    from deletion_tuning.deletion_worker import install_worker_patch

    if int(os.environ['WORLD_SIZE']) != 2:
        raise ValueError('smoke requires exactly two GPU ranks')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    torch.manual_seed(101)
    # warn_only: the bit-exact resume gate still catches real nondeterminism.
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(101)
    np.random.seed(101)
    os.environ['P3B_REPLAY'] = '1'
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
                                                attn_implementation='eager').cuda()
    if model.config.hidden_size != 1024 or model.config.num_hidden_layers != 28:
        raise ValueError('0.6B geometry changed')
    rows = [dict(id=str(i), prompt=f'What is {i} plus one?') for i in range(32)]
    calibration_path = work / 'calibration.json'
    if rank == 0:
        calibration_path.write_text(json.dumps(prepare(rows, tokenizer, fingerprint(rows), 101)))
    dist.barrier()
    os.environ['DELETION_CALIBRATION'] = str(calibration_path)
    os.environ['DELETION_SEED'] = '101'
    mesh = init_device_mesh('cuda', (2,))
    gradient_oracle_error = distributed_mask_oracle(mesh)
    install_worker_patch()
    import verl.workers.fsdp_workers as worker
    from omegaconf import OmegaConf
    from verl.utils.fsdp_utils import get_shard_placement_fn
    kwargs = dict(mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
                  reshard_after_forward=True, offload_policy=None, shard_placement_fn=get_shard_placement_fn(2))

    def shard(module):
        # Match the pinned worker's full-state capture, hook and broadcast order.
        state = module.state_dict()
        worker.apply_fsdp2(module, kwargs, OmegaConf.create({'wrap_policy': {}}))
        worker.fsdp2_load_full_state_dict(module, state, mesh, None)
        return module

    model = shard(model)
    teacher = shard(AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16, attn_implementation='eager').cuda())
    assert_matching_masks(model, teacher)
    teacher.eval().requires_grad_(False)
    initial_bits = mask_bits(model)
    initial_masks = mask_counts(model, initial_bits)
    # Independent tiny fp32 forward/chain-rule oracle also runs on the GPU.
    sample = next(iter(layers(model).values()))
    g = sample.diff_mask.gate.full_tensor()[:2, :3].detach().float().clone().requires_grad_()
    w = sample.weight.full_tensor()[:2, :3].detach().float()
    soft = torch.sigmoid(100 * g)
    (w * soft).sum().backward()
    expected = w * soft.detach() * (1 - soft.detach()) * 100
    if not torch.allclose(g.grad, expected, atol=1e-5, rtol=1e-5):
        raise ValueError('mask chain-rule oracle failed')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    from verl.workers.config.optimizer import FSDPOptimizerConfig
    optimizer = worker.build_optimizer(model.parameters(), FSDPOptimizerConfig(
        lr=5e-3, betas=(.9, .999), weight_decay=.01))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1.))
    config = SimpleNamespace(self_distillation=SimpleNamespace(), loss_agg_mode='seq-mean-token-mean',
                             ppo_epochs=1, ppo_mini_batch_size=16)
    # Use the same registered D validation values as the real actor config.
    config.self_distillation = dict(full_logit_distillation=True, alpha=0., distillation_topk=None,
                                   distillation_add_tail=False, teacher_regularization='ema', teacher_update_rate=.01)
    config.self_distillation = OmegaConf.create(config.self_distillation)
    owner = SimpleNamespace(actor_module=model, teacher_module=teacher, actor_optimizer=optimizer,
                            config=config, ulysses_sequence_parallel_size=1, scaler=None,
                            _v4_completed_updates=0)
    gradients = []
    def optimizer_step():
        from torch.distributed.tensor import DTensor
        params = [p for p in model.parameters() if p.requires_grad]
        local = [p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad for p in params]
        if any(g is None or not bool(torch.isfinite(g).all()) or not bool(g.abs().sum() > 0) for g in local):
            raise ValueError('missing/nonfinite/zero gate gradients')
        if any(p.grad is not None for p in teacher.parameters()):
            raise ValueError('teacher received gradients')
        if (not getattr(owner, '_deletion_finding_lr', False) and
                owner._v4_completed_updates == 0 and optimizer.param_groups[0]['lr'] != 0):
            raise ValueError('initial zero-LR warmup step changed')
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        scalar = norm.full_tensor() if isinstance(norm, DTensor) else norm
        if not torch.isfinite(scalar) or scalar <= 0:
            raise ValueError('nonfinite/zero gate gradient norm')
        gradients.append(float(scalar))
        optimizer.step()
        return scalar
    owner._optimizer_step = optimizer_step
    before = dict(student=frozen(model), teacher=frozen(teacher))
    if leg == 'finder':
        from deletion_tuning.deletion_lr import run_finder
        record = run_finder(owner, batch_for(owner, tokenizer, rank), code_check=True)
        if before != dict(student=frozen(model), teacher=frozen(teacher)):
            raise ValueError('LR finder changed frozen dense bank')
        (work / f'finder-r{rank}.json').write_text(json.dumps(record))
        dist.barrier()
        dist.destroy_process_group()
        return
    initial_logits = fingerprint({n: p for n, p in model.named_parameters() if p.requires_grad})
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    if leg == 'resumed':
        saved = torch.load(work / f'prefix-r{rank}.pt', weights_only=False, map_location='cpu')
        set_state_dict(model, optimizer, model_state_dict=saved['model'], optim_state_dict=saved['optimizer'], options=options)
        from torch.distributed.checkpoint.state_dict import set_model_state_dict
        set_model_state_dict(teacher, saved['teacher'], options=options)
        scheduler.load_state_dict(saved['scheduler'])
        owner._v4_completed_updates = saved['step']
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
    records, mask_records = [], []
    end = 2 if leg == 'prefix' else 4
    while owner._v4_completed_updates < end:
        batch = batch_for(owner, tokenizer, rank)
        metrics = update_policy(owner, batch)
        scheduler.step()
        if before != dict(student=frozen(model), teacher=frozen(teacher)):
            raise ValueError('frozen dense bank changed')
        if owner._v4_completed_updates != len(records) + (2 if leg == 'resumed' else 0) + 1:
            raise ValueError('optimizer/ramp counter lost')
        records.append(metrics)
        mask_records.append(mask_counts(model, initial_bits))
    state, optim = get_state_dict(model, optimizer, options=options)
    if fingerprint({n: p for n, p in model.named_parameters() if p.requires_grad}) == initial_logits:
        raise ValueError('gate logits did not move after nonzero LR updates')
    teacher_state = get_model_state_dict(teacher, options=options)
    saved = dict(model=state, optimizer=optim, teacher=teacher_state, scheduler=scheduler.state_dict(),
                 step=end, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                 python_rng=random.getstate(), numpy_rng=np.random.get_state())
    torch.save(saved, work / f'{leg}-r{rank}.pt')
    hashes = {k: fingerprint(v) for k, v in saved.items()}
    # Prove hard masked FSDP forward and canonical dense HF export agree.
    model.eval()
    probe = tokenizer('One plus one equals', return_tensors='pt').to('cuda')
    with torch.no_grad():
        masked_logits = model(**probe, use_cache=False).logits.cpu()
    full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    error = None
    if rank == 0:
        dense = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
                                                    attn_implementation='eager')
        dense.load_state_dict(hard_state(full), strict=True)
        dense.cuda().eval()
        with torch.no_grad():
            logits = dense(**probe, use_cache=False).logits.cpu()
        error = float((logits.float() - masked_logits.float()).abs().max())
        if error > .01:
            raise ValueError('hard-mask dense export differs')
        if leg == 'uninterrupted':
            dense.cpu().save_pretrained(work / 'dense')
            tokenizer.save_pretrained(work / 'dense')
    rollout_state = worker.convert_weight_keys(state, model)
    if any('.diff_mask.' in name for name in rollout_state):
        raise ValueError('rollout stream contains mask logits')
    (work / f'{leg}-r{rank}.json').write_text(json.dumps(dict(state=hashes, metrics=records, gradients=gradients,
        frozen=before, export_error=error, gradient_oracle_relative_error=gradient_oracle_error,
        world_size=2, rollout='FSDP2 HF hard-mask greedy', initial_masks=initial_masks,
        masks_after_updates=mask_records, teacher_initial_logits_equal=True,
        tie_word_embeddings=model.config.tie_word_embeddings, sharding='deletion_worker + verl apply_fsdp2/full-state load')))
    dist.barrier()
    dist.destroy_process_group()


def vllm_probe(work):
    from vllm import LLM, SamplingParams
    config = json.loads((work / 'dense/config.json').read_text())
    if config.get('hidden_size') != 1024 or config.get('num_hidden_layers') != 28:
        raise ValueError('vLLM smoke accepts only the 0.6B export geometry')
    model = LLM(model=str(work / 'dense'), tensor_parallel_size=2, dtype='bfloat16',
                max_model_len=256, gpu_memory_utilization=.5, enforce_eager=True, seed=101)
    result = model.generate(['What is one plus one?'], SamplingParams(temperature=0, max_tokens=8))
    tokens = result[0].outputs[0].token_ids
    if not tokens:
        raise ValueError('empty TP2 vLLM rollout')
    (work / 'vllm.json').write_text(json.dumps(dict(tensor_parallel_size=2, tokens=tokens)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('leg', choices=['uninterrupted', 'prefix', 'resumed', 'finder', 'vllm'])
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.leg == 'vllm':
        vllm_probe(args.work)
    else:
        train(args.leg, args.work)


if __name__ == '__main__':
    main()
