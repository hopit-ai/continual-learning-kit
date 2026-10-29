#!/usr/bin/env python3
"""500 frozen Level-5 targets through B's unchanged 3D bounded journal."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
import time


def job(a):
    doc=binding(a.out,a.standin)
    if doc['hard_math']!='QUALIFIES':
        print('SKIP: hard-MATH',doc['hard_math']); return
    items=hard_data()['train_pool.jsonl']; root=a.out/'targets-math'
    observed=dict(pool=s.file_hash(M/'train_pool.jsonl'),protocol=protocol(),technical_synthetic=a.standin)
    if not a.standin:
        observed.update(teacher=identity(a.model,'teacher'),tokenizer=identity(a.tokenizer,'student'))
        require(len(set(a.gpu.split(',')))==2,'27B targets use two GPUs')
    bind(root/'inputs.json',observed)
    if (root/'manifest.json').exists(): frozen_files(root); print('SKIP: hard-MATH targets'); return
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,V4_PHASE0_INFERENCE='1')
    started=time.monotonic()
    if a.standin:
        from kit.c_sim import Tokenizer
        class Generator:
            def generate_many(self,requests):
                lookup={r['id']:r for r in items}
                return [dict(text='Working through the rational calculation gives \\boxed{'+lookup[r['id']]['gold']+'}.',
                             finish_reason='stop',teacher_tokens=32) for r in requests]
        tokenizer,generator=Tokenizer(),Generator()
    else:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
        generator=t.VLLMGenerator(a.model,2560,tensor_parallel_size=2,allow_batch_fallback=False)
    s.run_phase(items,root/'teacher',generator,tokenizer,synthetic=a.standin)
    demos=t.read_jsonl(root/'teacher/demonstrations.jsonl')
    if not demos: raise NotReady('zero accepted targets; training input undefined')
    lookup={r['id']:r for r in items}
    rows=[dict(id=d['id'],prompt=t.student_messages(lookup[d['id']]),response=d['text'],
               gold=lookup[d['id']]['gold'],task='math_screen') for d in demos]
    t.write_jsonl(root/'pool.jsonl',rows); names=['inputs.json','pool.jsonl']
    for seed in SEEDS:
        name=f'schedule-{seed}.jsonl'; t.write_jsonl(root/name,cycle(rows,seed,1280)); names.append(name)
    atomic(root/'cost.json',dict(kind='targets',gpu_hours=0 if a.standin else 2*(time.monotonic()-started)/3600,technical_synthetic=a.standin))
    atomic(root/'manifest.json',files_manifest(root,names,valid=len(rows),population=500,coverage=len(rows)/500,
                                              protocol=protocol(),technical_synthetic=a.standin))


def configure(p):
    device(p,'0,1'); p.add_argument('--tokenizer')
if __name__=='__main__': raise SystemExit(entry(job,configure,'m2-targets'))
