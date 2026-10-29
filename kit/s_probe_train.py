#!/usr/bin/env python3
"""Screen S full-weight FSDP CE: four updates, batch 32, per-response token mean."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def response_ce(logits, targets):
    """One response's mean CE, including its EOS label. No prompt labels."""
    import torch
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), reduction='mean')


def load_probe_model(model, model_class):
    """FSDP retains these fp32 parameters as the optimizer's master weights."""
    import torch
    return model_class.from_pretrained(model, local_files_only=True, torch_dtype=torch.float32)


def probe_precision():
    import torch
    from torch.distributed.fsdp import MixedPrecision
    return MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)


def export_state(state):
    """Keep the full-weight export at bf16 size without casting master weights."""
    import torch
    return {name: value.to(torch.bfloat16) if value.is_floating_point() else value for name, value in state.items()}


def main():
    import argparse
    import os
    import functools
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kit.s_common import require, rows, atomic, sha, registration
    from kit.v4_teacher import tokenize_student
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=101)
    a = p.parse_args()
    registration()
    require(int(os.environ['WORLD_SIZE']) == 8, 'probe requires eight GPUs')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.manual_seed(a.seed)
    target = rows(a.schedule)
    require(len(target) == len({r['id'] for r in target}) == 128 and all(r['verified'] is True for r in target), 'exactly 128 verified unique targets required')
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    net = load_probe_model(a.model, AutoModelForCausalLM)
    net.config.use_cache = False
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    classes = {type(m) for m in net.modules() if type(m).__name__ in net._no_split_modules}
    require(bool(classes), 'missing transformer layer wrap policy')
    net = FSDP(net, device_id=local, use_orig_params=True,
               auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=classes),
               mixed_precision=probe_precision())
    require(all(p.dtype == torch.float32 for p in net.parameters()), 'probe master weights must be fp32')
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5, betas=(.9, .999), weight_decay=.01)
    net.train()
    metrics = []
    for step, lr in enumerate((0., 1e-6, 2e-6, 3e-6)):
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        # Four items per rank = global batch 32; average sequence CE, including EOS.
        for micro in range(4):
            row = target[step*32+micro*8+rank]
            messages = row['prompt'] if isinstance(row['prompt'], list) else [dict(role='user', content=row['prompt'])]
            pids, tids = tokenize_student(tok, messages, row['response'])
            require(0 < len(pids) <= 10240 and 0 < len(tids) <= 2048, 'probe cap drift')
            ids = torch.tensor([pids+tids], device=local)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = net(input_ids=ids, use_cache=False).logits[:, len(pids)-1:-1].float()
                ce = response_ce(logits, torch.tensor(tids, device=local))
            (ce/4).backward()
            total += ce.detach().item()/4
        net.clip_grad_norm_(1.)
        optimizer.step()
        metrics.append(dict(update=step+1, lr=lr, local_sequence_mean=total))
    with FSDP.state_dict_type(net, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = net.state_dict()
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=True)
        net.module.save_pretrained(a.output, state_dict=export_state(state), safe_serialization=True)
        tok.save_pretrained(a.output)
        atomic(a.output.parent/'training-receipt.json', dict(updates=4, batch=32, lrs=[0., 1e-6, 2e-6, 3e-6],
               schedule_sha256=sha(a.schedule), reduction='mean of response token means; EOS inclusive', seed=a.seed, metrics=metrics))
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
