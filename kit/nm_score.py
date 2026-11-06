#!/usr/bin/env python3
"""One decision cell within a complete NM lane; inherited scorers/numerics."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os
from kit.nm_common import *


def scorer_authority(panel):
    names=['nm_score.py']
    if panel=='knowledge':names+=['m3_score.py','ref_score_job.py']
    elif panel=='hellaswag':names+=['hellaswag_reference.py','ref_score_job.py']
    elif panel=='legal':names+=['c_score.py','second_task_screen.py']
    elif panel in ('instruction','maths'):names+=['score_forgetting.py','scorers.py']
    elif panel=='finqa':names+=['beds/finqa.py','eval_bed.py']
    else:names+=['a4_score.py','a4_common.py']
    return dict(method='REFERENCE-SCORED' if panel in ('knowledge','hellaswag') else 'FROZEN-GREEDY',
        sha256=digest({name:sha(KIT/name) for name in names}))


def file_stats(source):
    return {name:[(Path(source['model'])/name).stat().st_size,(Path(source['model'])/name).stat().st_mtime_ns] for name in source['files']}


def score(plan,out):
    p=load(plan);panel=p['panel'];source=p['source'];items=p['items']
    ready(file_stats(source)==p['metadata'],'prepared checkpoint metadata changed')
    need(scorer_authority(panel)==p['authority'],'scorer bytes changed')
    from kit.m2_assignment import inventory
    identity=inventory(os.environ.get('CUDA_VISIBLE_DEVICES','0'))
    need(identity['uuid'],'measured UUID unavailable')
    os.environ['M2_ASSIGNED_GPU_UUID']=identity['uuid']
    raw=[];likelihood=None
    if panel in ('knowledge','hellaswag'):
        if panel=='knowledge':
            from kit.m3_score import knowledge_reference
            likelihood=knowledge_reference(source['model'],items,p['tokenizer'])
        else:
            from kit.hellaswag_reference import score as reference
            likelihood=reference(source['model'],items,p['tokenizer'])
        need(len(likelihood)==len(items) and all(r['id']==i['id'] for i,r in zip(items,likelihood)),'reference item identity differs')
        correct={r['id']:int(r['selected']==i['label']) for i,r in zip(items,likelihood)}
    else:
        from vllm import SamplingParams
        from kit.inference_lifecycle import create_llm,shutdown_since
        from kit.score_forgetting import ENGINE,DECODING
        os.environ['VLLM_BATCH_INVARIANT']='1';os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
        rendered=p['rendered'];need(len(rendered)==len(items),'CPU rendered scoring prompts absent')
        model_length=4096 if panel in ('instruction','maths') else 12288
        engine=create_llm(model=source['model'],enable_lora=False,**dict(ENGINE,max_model_len=model_length),enforce_eager=True,seed=0)
        try:outputs=engine.generate(rendered,SamplingParams(n=1,**DECODING))
        finally:shutdown_since()
        need(len(outputs)==len(items),'missing decision response')
        raw=[dict(id=i['id'],response=o.outputs[0].text,output_tokens=len(o.outputs[0].token_ids),
            finish_reason=o.outputs[0].finish_reason,prompt_tokens=len(o.prompt_token_ids),token_ids=list(o.outputs[0].token_ids)) for i,o in zip(items,outputs)]
        correct={}
        for i,r in zip(items,raw):
            if panel in ('instruction','maths'):
                from kit.scorers import score as grade
                value=grade('ifeval' if panel=='instruction' else 'math',r['response'],i)['correct']
            elif panel=='legal':
                from kit.c_common import legal
                value=legal.parse_answer_tolerant('contract_nli',r['response'])==i['gold']
            elif panel=='finqa':
                from kit.beds import finqa
                value=finqa.is_correct(finqa.extract_answer(r['response']),i['gold'])
            else:
                from kit.a4_common import sec
                value=sec.v4_explicit_correct(i['item'],r['response'])
            correct[i['id']]=int(value)
    need(set(correct)=={i['id'] for i in items},'decision membership differs')
    cell=dict(seed=p['seed'],panel=panel,model=p['model'],step=p['step'],items=correct,checkpoint_sha256=source['sha256'],
        nonce=os.environ['NM_ALLOCATION_NONCE'],uuid=identity['uuid'],environment=identity['environment'],
        scorer_sha256=p['authority']['sha256'],method=p['authority']['method'],precision='full',status='COMPLETE',technical_synthetic=False)
    atomic(out,dict(cell=cell,raw=raw,likelihood=likelihood))


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();score(a.plan,a.out)
if __name__=='__main__':main()
