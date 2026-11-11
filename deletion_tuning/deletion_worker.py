"""Scoped hooks for the pinned D FSDP2 worker, before optimizer construction.

Selected only by deletion_tuning.deletion_entry, with DELETION_CALIBRATION set.
The calibration artifact contains four tokenized batches of eight task prompts
from the already bound schedule, padded/truncated to 1024 at preparation time.
No recipe/config override is added by this adapter.
"""
import hashlib
import json
import os
from pathlib import Path

import torch

from deletion_tuning.deletion_mask import calibrate, hard_state, shard_masks


def validate_calibration(path):
    from deletion_tuning.deletion_logic import calibration_batches
    document = json.loads(Path(path).read_text())
    if document.get('seed') != int(os.environ['DELETION_SEED']):
        raise ValueError('calibration seed differs from paired training seed')
    return calibration_batches(document)


def install_worker_patch():
    from kit.sdft_verl import check_pinned
    check_pinned()
    import verl.workers.fsdp_workers as worker
    if getattr(worker, '_deletion_installed', False):
        return
    original_shard = worker.apply_fsdp2
    original_load = worker.fsdp2_load_full_state_dict
    original_keys = worker.convert_weight_keys
    original_optimizer = worker.build_optimizer
    pending = {}
    initial_gates = None

    def shard(model, kwargs, config):
        nonlocal initial_gates
        rank = torch.distributed.get_rank()
        identical = True
        if rank == 0:
            batches = validate_calibration(os.environ['DELETION_CALIBRATION'])
            device = torch.device('cuda', torch.cuda.current_device())
            model.to(device)
            values = [{k: torch.tensor(v, device=device) for k, v in b.items()} for b in batches]
            # Both actor and teacher receive the same deterministic initial G.
            with torch.random.fork_rng(devices=[device.index]):
                torch.manual_seed(int(os.environ['DELETION_SEED']))
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    calibrate(model, values)
            pending[id(model)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                                  if k.endswith('.diff_mask.gate')}
            fingerprints = {name: (tuple(value.shape), str(value.dtype),
                                   hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest())
                            for name, value in pending[id(model)].items()}
            if initial_gates is None:
                initial_gates = fingerprints
            else:
                identical = initial_gates == fingerprints
        else:
            from deletion_tuning.deletion_mask import install
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                install(model)
            pending[id(model)] = {}
        verdict = [identical]
        torch.distributed.broadcast_object_list(verdict, src=0)
        if not verdict[0]:
            raise ValueError('separately calibrated teacher mask logits differ from student')
        shard_masks(model, kwargs)
        return original_shard(model, kwargs, config)

    def load(model, state, *args, **kwargs):
        state = dict(state)
        state.update(pending.pop(id(model)))
        return original_load(model, state, *args, **kwargs)

    def keys(state, model):
        # Only the rollout stream passes here. Native checkpointing retains G/W.
        return original_keys(hard_state(state), model)

    def optimizer(parameters, config):
        return original_optimizer((p for p in parameters if p.requires_grad), config)

    worker.apply_fsdp2 = shard
    worker.fsdp2_load_full_state_dict = load
    worker.convert_weight_keys = keys
    worker.build_optimizer = optimizer
    worker._deletion_installed = True
