#!/usr/bin/env python3
"""TRACE paper-dose SeqFT port. One persistent DeepSpeed engine per seed."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
from kit.trace_common import *


def suffix_streams(tokenizer, row):
    messages = [dict(role='user', content=row['prompt'])]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
    return (tokenizer.encode(prompt+row['answer'], add_special_tokens=False),
            tokenizer.encode(row['answer'], add_special_tokens=False))


def suffix_tokens(combined, answer, eos):
    target = (answer+[eos])[-1536:]
    ids = (combined+[eos])[-1536:]
    need(len(target) <= len(ids) and ids[-len(target):] == target,
         'published supervised suffix differs from answer tokens')
    return dict(input_ids=ids, attention_mask=[1]*len(ids),
                labels=[-100]*(len(ids)-len(target))+target,
                combined_cap_hit=len(combined)+1 > 1536,
                target_cap_hit=len(answer)+1 > 1536)


def suffix_example(tokenizer, row):
    """Published left clipping and independently measured answer suffix mask."""
    combined, answer = suffix_streams(tokenizer, row)
    return suffix_tokens(combined, answer, tokenizer.eos_token_id)


def eos_comparison(combined, answer, eos, corrected_streams=None):
    """Only one left shift followed by EOS is permitted in each capped stream."""
    changed, bounded = False, True
    streams = (combined, answer)
    if corrected_streams is None:
        corrected_streams = [(ids+[eos])[-1536:] for ids in streams]
    for ids, corrected in zip(streams, corrected_streams):
        original = (ids+[eos] if len(ids) < 1536 else ids)[-1536:]
        if original != corrected:
            changed = True
            bounded = bounded and len(ids) >= 1536 and corrected == original[1:]+[eos]
    return changed, bounded


def sft_census(rows, tokenizer, original_tokenizer=None):
    """CPU-only census of every row, including BPE boundary and conditional EOS."""
    encoded, errors = [], []
    affected, original_affected, position_errors = 0, 0, []
    for row in rows:
        combined, answer = suffix_streams(tokenizer, row)
        eos = tokenizer.eos_token_id
        corrected = None
        try:
            token_row = suffix_tokens(combined, answer, eos)
            encoded.append(token_row)
            corrected = (token_row['input_ids'],[t for t in token_row['labels'] if t != -100])
        except ValueError:
            errors.append(row['id'])
        changed, bounded = eos_comparison(combined, answer, eos, corrected)
        affected += changed
        if not bounded:
            position_errors.append(row['id'])
        if original_tokenizer is not None:
            original_streams = suffix_streams(original_tokenizer, row)
            original_affected += eos_comparison(*original_streams,
                                               original_tokenizer.eos_token_id)[0]
    n = len(rows)
    share = affected/n if n else 0.
    original_share = original_affected/n if n and original_tokenizer is not None else None
    excess = share-original_share if original_share is not None else None
    warnings = []
    if original_share is None:
        warnings.append('L-EOS original-collator rate UNMEASURED')
    elif excess > L_EOS_WARN_EXCESS:
        warnings.append('L-EOS Qwen minus original affected-row share exceeds 1 pp, continuing')
    summary = dict(rows=n, affected=affected, share=share,
                   suffix_errors=len(errors), error_ids=errors,
                   eos_position_errors=len(position_errors), position_error_ids=position_errors,
                   max_eos_position_difference=None if position_errors else int(bool(affected)),
                   original_affected=original_affected if original_tokenizer is not None else None,
                   original_share=original_share, excess_share=excess,
                   warning_excess=L_EOS_WARN_EXCESS, warnings=warnings,
                   valid=bool(n) and not errors and not position_errors)
    return encoded, summary


def deepspeed_config():
    return dict(train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=16,
                train_batch_size=128, bf16=dict(enabled=True),
                gradient_clipping=GRADIENT_CLIPPING,
                zero_optimization=dict(stage=3, stage3_gather_16bit_weights_on_model_save=True))


def export_plan(root, index):
    """Reuse a verified seal or reserve fresh bytes for an interrupted export."""
    import uuid
    root = Path(root)
    sealed = root/f'export-{index}.json'
    if sealed.exists():
        record = load(sealed)
        model_record(record)
        return dict(reuse=True, record=record)
    target = root/f'endpoint-{index}'/uuid.uuid4().hex
    target.mkdir(parents=True)
    return dict(reuse=False, target=str(target))


def finish_export(root, index, target):
    target = Path(target)
    files = {p.name:sha(p) for p in target.glob('*') if p.is_file()}
    need('pytorch_model.bin' in files and 'config.json' in files, 'export incomplete')
    record = dict(model=str(target), files=files,
                  sha256=digest({k:v for k,v in files.items()
                      if k.endswith(('.safetensors','.bin')) or k=='config.json'}))
    return seal(Path(root)/f'export-{index}.json', record)


def main(cfg, seed):
    offline()
    from kit.trace_jobs import timing_phase as phase
    from contextlib import nullcontext
    rank = int(os.environ['LOCAL_RANK'])
    with phase('prep') if rank == 0 else nullcontext():
        prepared = load(Path(cfg['out'])/'prepared.json')
        need(prepared.get('s8_ready') is True, 'S8 CPU census/licences PENDING')
        from kit.trace_common import local_file
        token_rows = {task:local_file(bound_file(ref)) for task,ref in prepared['sft_rows'].items()}
        import torch
        import random
        import numpy as np
        import deepspeed
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch.utils.data import DataLoader, DistributedSampler
        rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(rank)
        deepspeed.init_distributed()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(cfg['base']['model'],
                     local_files_only=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(cfg['base']['model'], local_files_only=True)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(.9,.95),
                                     eps=1e-8, weight_decay=0)
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model, optimizer=optimizer, config=deepspeed_config())
        root = Path(cfg['out'])/'training'/f's8-{seed}'
        root.mkdir(parents=True, exist_ok=True)
        cursor = dict(task=0, epoch=0, micro=0, config_sha256=digest(cfg))
        pointer = root/'resume.json'
        restore_rng = None
        if pointer.exists():
            saved = load(pointer)
            need(saved['config_sha256'] == digest(cfg), 'SeqFT native identity changed')
            _, cursor = engine.load_checkpoint(str(root/'native'), tag=saved['tag'])
            need(cursor['config_sha256'] == digest(cfg), 'SeqFT resume config changed')
            restore_rng = cursor['rng'][rank]
        token_counts = list(restore_rng.get('token_counts',[0]*5)) if restore_rng else [0]*5
        def collate(rows):
            width = ((max(len(r['input_ids']) for r in rows)+7)//8)*8
            result = {}
            for key, pad in [('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100)]:
                result[key] = torch.tensor([[pad]*(width-len(r[key]))+r[key] for r in rows])
            counts = [len(rows),sum(sum(r['attention_mask']) for r in rows),
                      sum(sum(x != -100 for x in r['labels']) for r in rows),
                      sum(r['combined_cap_hit'] for r in rows),
                      sum(r['target_cap_hit'] for r in rows)]
            return result, counts
    from concurrent.futures import ThreadPoolExecutor
    hash_pool = ThreadPoolExecutor(max_workers=1) if rank == 0 else None
    hash_jobs = []
    from kit.trace_jobs import timing_phase as phase
    from contextlib import nullcontext
    for index, (task, epochs) in enumerate(zip(TASKS, EPOCHS)):
        if index < cursor['task']:
            continue
        with phase('train') if rank == 0 else nullcontext():
            rows = load(token_rows[task])
            sampler = DistributedSampler(rows, num_replicas=8, rank=rank, seed=seed,
                                         shuffle=True, drop_last=False)
            loader = DataLoader(rows, sampler=sampler, batch_size=1, collate_fn=collate)
            for epoch in range(epochs):
                if index == cursor['task'] and epoch < cursor['epoch']:
                    continue
                for micro, (batch, batch_counts) in enumerate(loader):
                    if index == cursor['task'] and epoch == cursor['epoch'] and micro < cursor['micro']:
                        continue
                    if restore_rng is not None:
                        random.setstate(restore_rng['python'])
                        np.random.set_state(restore_rng['numpy'])
                        torch.set_rng_state(restore_rng['torch'])
                        torch.cuda.set_rng_state(restore_rng['cuda'], rank)
                        restore_rng = None
                    engine.train()
                    batch = {k:v.to(rank) for k,v in batch.items()}
                    loss = engine(**batch, use_cache=False).loss
                    need(bool(torch.isfinite(loss)), 'nonfinite supervised loss')
                    engine.backward(loss)
                    engine.step()
                    token_counts = [a+b for a,b in zip(token_counts,batch_counts)]
                    if engine.micro_steps % 16 == 0 and engine.global_steps % 10 == 0:
                        local_rng = dict(python=random.getstate(),numpy=np.random.get_state(),
                                         torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(rank),
                                         token_counts=token_counts)
                        rngs = [None]*8
                        torch.distributed.all_gather_object(rngs,local_rng)
                        state = dict(task=index,epoch=epoch,micro=micro+1,rng=rngs,
                                     config_sha256=digest(cfg))
                        tag = f'update-{engine.global_steps}'
                        engine.save_checkpoint(str(root/'native'),tag=tag,client_state=state)
                        if rank == 0:
                            atomic(pointer,dict(tag=tag,config_sha256=digest(cfg)))
                # Save only at accumulation boundaries. Task-final exports preserve
                # residual gradients in the live engine, matching upstream carry.
        plans = [export_plan(root, index+1) if rank == 0 else None]
        torch.distributed.broadcast_object_list(plans, src=0)
        plan = plans[0]
        if not plan['reuse']:
            target = Path(plan['target'])
            with phase('export') if rank == 0 else nullcontext():
                engine.save_16bit_model(str(target), 'pytorch_model.bin')
            if rank == 0:
                engine.module.config.save_pretrained(target)
                tokenizer.save_pretrained(target)
                hash_jobs.append(hash_pool.submit(finish_export, root, index+1, target))
        torch.distributed.barrier()
    counts_by_rank = [None]*8
    torch.distributed.all_gather_object(counts_by_rank, token_counts)
    if rank == 0:
        # Publication failures propagate before the complete export inventory is sealed.
        with phase('export'):
            for job in hash_jobs:
                job.result()
        hash_pool.shutdown(wait=True)
        seal(root/'token-counts.json',[sum(values) for values in zip(*counts_by_rank)])
        exports = {'base':cfg['base']}
        exports.update({f's8-s{seed}-{i}':load(root/f'export-{i}.json') for i in range(1,9)})
        seal(root/'exports.json', exports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    main(dict(load(args.config), _config_path=str(Path(args.config).resolve())), args.seed)
