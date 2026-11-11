"""House D adapter of the MIT deletion-tuning parameterisation.

The sole vendored module is DifferentiableMask. Ordinary autograd is used,
with canonical weight names, fp32 logits, frozen dense weights and no cache.
"""
# Portions adapted from deleting-weights, copyright (c) 2026 agentic learning
# ai lab. MIT licence in deletion_tuning/vendor/deleting_weights/LICENSE.
import re

import torch
from torch import nn

from deletion_tuning.vendor.deleting_weights.differentiable_mask import DifferentiableMask

GATE = re.compile(r'model\.layers\.\d+\.mlp\.gate_proj')
LOGIT_SUFFIX = '.diff_mask.gate'


class MaskedLinear(nn.Module):
    def __init__(self, linear, total_steps=40):
        super().__init__()
        if linear.bias is not None:
            raise ValueError('deletion requires bias-free gate projections')
        self.weight = linear.weight.requires_grad_(False)
        self.diff_mask = DifferentiableMask(self.weight, [100., 10000.], total_steps, self.weight.device)
        # A matrix gives W and G identical sharding geometry, including export.
        self.diff_mask.gate = nn.Parameter(self.diff_mask.gate.reshape(self.weight.shape))

    def forward(self, inputs):
        if self.diff_mask.gate.dtype != torch.float32:
            raise ValueError('mask logits must remain fp32 under FSDP')
        mask = self.diff_mask(self.weight.shape, self.weight.dtype)
        return nn.functional.linear(inputs, self.weight * mask)


def layers(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, MaskedLinear)}


def install(model, total_steps=40):
    targets = [(name, module) for name, module in model.named_modules() if GATE.fullmatch(name)]
    if not targets or len(targets) != model.config.num_hidden_layers:
        raise ValueError('incomplete Qwen gate inventory')
    if any(not isinstance(module, nn.Linear) or module.bias is not None for _, module in targets):
        raise ValueError('unexpected gate geometry')
    all_weights = list(model.named_parameters(remove_duplicate=False))
    if any(sum(parameter is module.weight for _, parameter in all_weights) != 1 for _, module in targets):
        raise ValueError('gate weights have tied parameter roles')
    model.requires_grad_(False)
    for name, module in targets:
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, MaskedLinear(module, total_steps))
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    return model


def calibrate(model, batches, eta=1 / 2048, strength=3.):
    """Dense Wanda norms, strict per-layer kth threshold and empirical sample SD.

    Run before install/sharding on 32 frozen task prompts, four batches of 8.
    Padding participates exactly as in the authors' implementation.
    """
    targets = {n: m for n, m in model.named_modules() if GATE.fullmatch(n)}
    norms, hooks = {}, []
    was_training = model.training
    def capture(name, module, inputs):
        x = inputs[0].detach().float()
        norm = x.square().sum(tuple(range(x.ndim - 1))).sqrt()
        norms[name] = norms.get(name, 0) + norm
    try:
        for name, module in targets.items():
            hooks.append(module.register_forward_pre_hook(lambda m, x, n=name: capture(n, m, x)))
        model.eval()
        with torch.no_grad():
            for batch in batches:
                model(**batch, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    if set(norms) != set(targets):
        raise ValueError('incomplete Wanda calibration')
    scores = {n: m.weight.detach().float().abs() * norms[n][None] for n, m in targets.items()}
    install(model)
    with torch.no_grad():
        for name, module in layers(model).items():
            flat = scores[name].flatten()
            # CPU kthvalue: CUDA kthvalue has no deterministic implementation (same result).
            threshold = flat.cpu().kthvalue(max(1, int(flat.numel() * eta))).values.to(flat.device)
            gate = module.diff_mask.gate
            sign = torch.where(scores[name] < threshold, -1., 1.)
            gate.add_(sign * gate.std() * strength)
    return model


def set_step(model, step):
    from deletion_tuning.deletion_logic import sharpness
    sharpness(step)  # validate before mutating
    for module in layers(model).values():
        module.diff_mask.current_step = step


@torch.no_grad()
def update_teacher(student, teacher, rate=.01):
    source, target = dict(student.named_parameters()), dict(teacher.named_parameters())
    if source.keys() != target.keys():
        raise ValueError('teacher parameter inventory differs')
    for name, parameter in target.items():
        if name.endswith(LOGIT_SUFFIX):
            parameter.mul_(1 - rate).add_(source[name].to(parameter.device), alpha=rate)
    teacher.eval()


def hard_state(state):
    """Dense rollout/export state. Input bank and gate tensors stay untouched.

    DTensor operations preserve the mesh and sharding until the caller gathers.
    """
    result = {name: value for name, value in state.items() if not name.endswith(LOGIT_SUFFIX)}
    gates = {name.removesuffix(LOGIT_SUFFIX): value for name, value in state.items()
             if name.endswith(LOGIT_SUFFIX)}
    if not gates:
        raise ValueError('missing deletion logits')
    for name, gate in gates.items():
        weight_name = name + '.weight'
        weight = result[weight_name]
        if weight.shape != gate.shape:
            raise ValueError('gate/weight sharding geometry differs')
        result[weight_name] = weight * (gate > 0).to(weight.dtype)
    return result


def shard_masks(model, fsdp_kwargs):
    """Keep G in fp32 while the inherited decoder policy keeps dense BF16."""
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    kwargs = dict(fsdp_kwargs, mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32))
    for module in layers(model).values():
        fully_shard(module.diff_mask, **kwargs)


def mask_bits(model):
    """CPU-local bits, preserving sharding so receipts need no full gather."""
    return {name: (module.diff_mask.gate.to_local() if hasattr(module.diff_mask.gate, 'to_local')
                   else module.diff_mask.gate).detach().cpu() > 0
            for name, module in layers(model).items()}


def mask_counts(model, initial):
    import torch.distributed as dist
    current = mask_bits(model)
    result = {}
    for name, bits in current.items():
        old = initial[name]
        counts = torch.tensor([bits.numel(), (~bits).sum(), (old & ~bits).sum(), (~old & bits).sum()],
                              dtype=torch.int64, device=next(model.parameters()).device)
        if dist.is_initialized():
            dist.all_reduce(counts)
        size, zeros, keep_delete, delete_keep = counts.tolist()
        result[name] = dict(weights=size, hard_zeros=zeros, keep_to_delete=keep_delete,
                            delete_to_keep=delete_keep)
    return result


def assert_matching_masks(student, teacher):
    import torch.distributed as dist
    source, target = layers(student), layers(teacher)
    equal = source.keys() == target.keys()
    for name in source.keys() & target.keys():
        left, right = source[name].diff_mask.gate, target[name].diff_mask.gate
        left = left.to_local() if hasattr(left, 'to_local') else left
        right = right.to_local() if hasattr(right, 'to_local') else right
        equal = equal and torch.equal(left.detach().cpu(), right.detach().cpu())
    flag = torch.tensor(int(equal), device=next(student.parameters()).device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise ValueError('teacher mask logits differ bit for bit from student')
