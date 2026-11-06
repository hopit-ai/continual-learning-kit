#!/usr/bin/env python3
"""One batched OPR generation lane for one seed's untouched FinQA final."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os
import time
from kit.nm_common import *
from kit.nm_data import generation_seed,finqa_accept


def generate(plan,out,qualification=False):
    from vllm import SamplingParams
    from kit.inference_lifecycle import create_llm,shutdown_since
    p=load(plan);seed=p['seed'];need(seed in SEEDS,'one registered seed per lane')
    prompts=p['prompts'];need(prompts and len({r['id'] for r in prompts})==len(prompts),'generation IDs differ')
    from kit.nm_model import metadata
    need(metadata(p['model'],p['model_metadata'])==p['model_metadata'],'prepared generation checkpoint changed')
    rendered=p['rendered_prompts'];need({r['id'] for r in prompts}<=set(rendered),'CPU generation prompts absent')
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    engine=create_llm(model=p['model'],dtype='bfloat16',tensor_parallel_size=1,
        max_model_len=p['prompt_cap']+2048,gpu_memory_utilization=p['memory_fraction'],max_num_seqs=p['batch_size'],enforce_eager=True,seed=seed)
    started=time.monotonic();attempts={};active=prompts
    try:
        for attempt in range(1,5):
            if not active:break
            parameters=[SamplingParams(n=1,temperature=.7,top_p=1.,max_tokens=2048,seed=generation_seed(seed,r['id'],attempt)) for r in active]
            outputs=engine.generate([rendered[r['id']] for r in active],parameters)
            need(len(outputs)==len(active),'generation batch incomplete');remaining=[]
            for r,o in zip(active,outputs):
                x=o.outputs[0];value=dict(response=x.text,output_tokens=len(x.token_ids),finish_reason=x.finish_reason,rng=generation_seed(seed,r['id'],attempt))
                attempts.setdefault(r['id'],{})[str(attempt)]=value
                if not finqa_accept(value,r['gold']):remaining.append(r)
            active=remaining
    finally:shutdown_since()
    atomic(out,dict(seed=seed,attempts=attempts,wall_seconds=time.monotonic()-started,qualification=qualification,
        tokens=sum(v['output_tokens'] for a in attempts.values() for v in a.values()),
        generator_sha256=p['generator_sha256'],batch_size=p['batch_size']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);p.add_argument('--qualification',action='store_true');a=p.parse_args();generate(a.plan,a.out,a.qualification)
if __name__=='__main__':main()
