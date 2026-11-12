"""Pre-registered training-only mask LR qualification, with full trial reset.

The input is one already-bound first TRAINING batch (32 global rows), including
its fixed hard-rollout responses, D teacher prompts and inherited replay slots.
No scorer, retention panel or evaluation callback enters this interface.
"""
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from deletion_tuning.deletion_actor import update_policy
from deletion_tuning.deletion_mask import (
    LOGIT_SUFFIX, assert_matching_masks, layers, mask_bits, mask_counts, set_step,
)

GRID = (1e-3, 3e-3, 5e-3, 1e-2)
FALLBACK = 5e-3
# Qwen3-8B (hidden_size, num_hidden_layers); must match kit/nm_prepare.py and kit/lane_pool.py.
QWEN3_8B_GEOMETRY = (4096, 36)
UPDATES = 3
MAX_DAMAGE = .10
MAX_FLIPS = .01


def local(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = local(tensor).detach().cpu().contiguous()
        digest.update(str((name, value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def dense_digest(model):
    return tensor_digest({n: p for n, p in model.named_parameters() if not n.endswith(LOGIT_SUFFIX)})


def gate_state(model):
    return {name: local(parameter).detach().cpu().clone()
            for name, parameter in model.named_parameters() if name.endswith(LOGIT_SUFFIX)}


@torch.no_grad()
def restore_gates(model, state):
    for name, parameter in model.named_parameters():
        if name.endswith(LOGIT_SUFFIX):
            local(parameter).copy_(state[name].to(parameter.device))


@contextmanager
def dense_masks(*models):
    """Temporarily use the untouched dense bank for the training damage probe."""
    overrides = []
    try:
        for model in models:
            for module in layers(model).values():
                overrides.append((module, module.__dict__.get('forward')))
                module.forward = lambda x, m=module: torch.nn.functional.linear(x, m.weight)
        yield
    finally:
        for module, previous in overrides:
            if previous is None:
                del module.forward
            else:
                module.forward = previous


@torch.no_grad()
def training_loss(owner, data, hard=False):
    """The unchanged D forward-KL + inherited replay CE on fixed training rows."""
    from kit.sdft_actor import teacher_projection, _model_forward, projected_distillation
    from kit import sdft_objective as objective
    from kit.p3b_loss import replay_loss
    student, teacher = owner.actor_module, owner.teacher_module
    was_training = student.training
    student.train(not hard)
    teacher.eval()
    loss = 0.
    micro_batches = data.split(1)
    device = next(student.parameters()).device
    try:
        for batch in micro_batches:
            batch = batch.to(device)
            inputs = batch.batch
            teacher_inputs = dict(responses=inputs['responses'], input_ids=inputs['teacher_input_ids'],
                                  attention_mask=inputs['teacher_attention_mask'],
                                  position_ids=inputs['teacher_position_ids'])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                target = teacher_projection(teacher, teacher_inputs)
                extra = batch.non_tensor_batch['extra_info'][0]
                if extra.get('p3b_replay'):
                    value = replay_loss(student, inputs, extra['p3b_target_ids'])
                else:
                    width = inputs['responses'].shape[1]
                    def head(hidden, weight):
                        return projected_distillation(hidden[:, -width-1:-1], weight, target['hidden'],
                            target['weight'], inputs['responses'], float(data.meta_info['temperature']))[0]
                    kl = _model_forward(student, inputs, head)
                    mask = objective.loss_mask(inputs['response_mask'])
                    counts = mask.sum(-1).clamp(min=1)
                    weights = objective.reference_importance_weights(inputs['old_log_probs'], inputs['rollout_log_probs'])
                    value = ((kl * mask).sum(-1) / counts * (weights * mask).sum(-1) / counts).mean()
                loss += float(value) / len(micro_batches)
        value = torch.tensor(loss, dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(value)
            value /= dist.get_world_size()
        return float(value)
    finally:
        student.train(was_training)


def choose_rate(trials):
    if tuple(row['rate'] for row in trials) != GRID:
        raise ValueError('LR finder requires the complete ordered fixed grid')
    eligible = [row for row in trials if all(math.isfinite(row[key]) for key in
                ('loss_drop', 'initial_damage', 'max_flip_fraction'))
                and row['loss_drop'] > 0 and row['initial_damage'] <= MAX_DAMAGE
                and row['max_flip_fraction'] <= MAX_FLIPS]
    # Fixed grid order breaks exact ties in favour of the smaller LR.
    winner = max(eligible, key=lambda row: row['loss_drop']) if eligible else None
    return (winner['rate'] if winner else FALLBACK), winner is None


def run_finder(owner, data, *, code_check=False):
    student, teacher, optimizer = owner.actor_module, owner.teacher_module, owner.actor_optimizer
    if getattr(owner, '_v4_completed_updates', 0) != 0:
        raise ValueError('LR finder must precede scientific updates')
    if not code_check and (student.config.hidden_size, student.config.num_hidden_layers) != QWEN3_8B_GEOMETRY:
        raise ValueError('production LR selection requires the Qwen3-8B qualification path')
    if not code_check:
        count = torch.tensor(data.batch['input_ids'].shape[0], device=next(student.parameters()).device)
        if dist.is_initialized():
            dist.all_reduce(count)
        if count.item() != 32:
            raise ValueError('production LR finder requires the bound global batch of 32')
    assert_matching_masks(student, teacher)
    trainable = {id(p) for n, p in student.named_parameters() if n.endswith(LOGIT_SUFFIX)}
    if {id(p) for group in optimizer.param_groups for p in group['params']} != trainable:
        raise ValueError('finder optimizer must contain only every mask logit')
    if any(p.requires_grad for n, p in student.named_parameters() if not n.endswith(LOGIT_SUFFIX)):
        raise ValueError('finder dense bank must stay frozen')
    if optimizer.state:
        raise ValueError('LR finder requires a fresh optimizer')
    source, target = gate_state(student), gate_state(teacher)
    dense_before = dict(student=dense_digest(student), teacher=dense_digest(teacher))
    if dense_before['student'] != dense_before['teacher']:
        raise ValueError('LR finder teacher dense bank differs from incoming student')
    optim = copy.deepcopy(optimizer.state_dict())
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
           random.getstate(), np.random.get_state())
    modes = student.training, teacher.training
    steps = {name: module.diff_mask.current_step for name, module in layers(student).items()}
    bits = mask_bits(student)
    initial_counts = mask_counts(student, bits)
    updates = 2 if code_check else UPDATES
    initial_teacher_training = {name: p.requires_grad for name, p in teacher.named_parameters()}
    checked = owner.__dict__.get('_deletion_initial_masks_checked')
    teacher.requires_grad_(False)

    def reset():
        restore_gates(student, source)
        restore_gates(teacher, target)
        optimizer.load_state_dict(copy.deepcopy(optim))
        optimizer.zero_grad()
        owner._v4_completed_updates = 0
        torch.set_rng_state(rng[0])
        if rng[1]:
            torch.cuda.set_rng_state_all(rng[1])
        random.setstate(rng[2])
        np.random.set_state(rng[3])
        set_step(student, 0)

    def measure(hard=False):
        # A common initial hard teacher and alpha=100 isolate student movement.
        current = gate_state(teacher)
        restore_gates(teacher, target)
        set_step(student, 0)
        try:
            return training_loss(owner, data, hard=hard)
        finally:
            restore_gates(teacher, current)

    trials = []
    owner._deletion_finding_lr = True
    try:
        set_step(student, 0)
        with dense_masks(student, teacher):
            incoming = training_loss(owner, data, hard=True)
        initial_soft, initial_hard = measure(), measure(hard=True)
        if not all(math.isfinite(x) for x in (incoming, initial_soft, initial_hard)):
            raise ValueError('nonfinite initial LR-finder training loss')
        denominator = max(abs(incoming), 1e-8)
        damage = max(0., initial_hard - incoming) / denominator
        for rate in GRID:
            reset()
            for group in optimizer.param_groups:
                group['lr'] = rate
            records, peak_flips = [], 0.
            for step in range(updates):
                metrics = update_policy(owner, data)
                if owner._v4_completed_updates != step + 1:
                    raise ValueError('LR finder update did not complete')
                loss = metrics.get('p3b/combined_loss', metrics['actor/pg_loss'])
                if not math.isfinite(loss):
                    raise ValueError('nonfinite LR-finder training loss')
                counts = mask_counts(student, bits)
                peak_flips = max(peak_flips, max((c['keep_to_delete'] + c['delete_to_keep']) / c['weights']
                                                for c in counts.values()))
                records.append(dict(loss=loss, masks=counts))
            final_soft = measure()
            if not math.isfinite(final_soft):
                raise ValueError('nonfinite final LR-finder training loss')
            trials.append(dict(rate=rate, loss_before=initial_soft, loss_after=final_soft,
                loss_drop=initial_soft-final_soft, initial_damage=damage,
                max_flip_fraction=peak_flips,
                updates=records))
        selected, fallback = choose_rate(trials)
        if dense_before != dict(student=dense_digest(student), teacher=dense_digest(teacher)):
            raise ValueError('LR finder changed a frozen dense bank')
        batch_hash = tensor_digest(data.batch)
        extra_hash = hashlib.sha256(json.dumps(data.non_tensor_batch['extra_info'], sort_keys=True,
            default=lambda value: value.tolist()).encode()).hexdigest()
        inputs = [dict(tensors=batch_hash, extra_info=extra_hash)]
        if dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, inputs[0])
            inputs = gathered
        calibration = os.environ.get('DELETION_CALIBRATION')
        return dict(schema='deletion-lr.v1', code_check=code_check, training_only=True,
                    selected_rate=None if code_check else selected, fallback=None if code_check else fallback,
                    grid=list(GRID), updates_per_rate=updates, max_damage=MAX_DAMAGE, max_flips=MAX_FLIPS,
                    incoming_training_loss=incoming, initial_hard_training_loss=initial_hard,
                    initial_masks=initial_counts, trials=trials, training_input_hashes=inputs,
                    dense_bank_hashes_local=dense_before, seed=101,
                    model_geometry=[getattr(student.config, 'hidden_size', None), student.config.num_hidden_layers],
                    calibration_sha256=hashlib.sha256(Path(calibration).read_bytes()).hexdigest() if calibration else None)
    finally:
        reset()
        for name, module in layers(student).items():
            module.diff_mask.current_step = steps[name]
        student.train(modes[0])
        teacher.train(modes[1])
        for name, parameter in teacher.named_parameters():
            parameter.requires_grad_(initial_teacher_training[name])
        if checked is None:
            owner.__dict__.pop('_deletion_initial_masks_checked', None)
        else:
            owner._deletion_initial_masks_checked = checked
        del owner._deletion_finding_lr


def frozen_rate(path, expected_sha256):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('frozen LR receipt hash differs')
    record = json.loads(payload)
    if (record['schema'] != 'deletion-lr.v1' or record['code_check'] is not False or
            record['training_only'] is not True or record['updates_per_rate'] != UPDATES or
            record['max_damage'] != MAX_DAMAGE or record['max_flips'] != MAX_FLIPS or
            record['grid'] != list(GRID) or record['model_geometry'] != list(QWEN3_8B_GEOMETRY) or record['seed'] != 101):
        raise ValueError('receipt is not the registered production LR finder')
    selected, fallback = choose_rate(record['trials'])
    if record['selected_rate'] != selected or record['fallback'] != fallback:
        raise ValueError('frozen LR receipt selection differs')
    return selected
