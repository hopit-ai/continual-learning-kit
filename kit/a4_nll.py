#!/usr/bin/env python3
"""EOS-inclusive student NLL on the exact stage schedule, weighted by repetitions."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
from collections import Counter


def measure(model,rows,standin):
    if not standin:model=str(__import__('kit.m2_paths',fromlist=['local']).local(model))
    counts=Counter(s.digest(r) for r in rows);unique={s.digest(r):r for r in rows}
    if standin:
        from kit.c_sim import Tokenizer
        tok=Tokenizer()
    else:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    encoded={}
    for key,row in unique.items():
        prompt=row['prompt']
        if isinstance(prompt,str): prompt=[dict(role='user',content=prompt)]
        pids,tids=t.tokenize_student(tok,prompt,row['response'])
        require(len(pids)<=10240 and 0<len(tids)<=2048,'NLL cap drift')
        encoded[key]=(pids,tids)
    if standin: return dict(sequence_mean=1.,token_mean=1.,exposures=len(rows),unique=len(unique),
        response_tokens=sum(counts[key]*len(ids[1]) for key,ids in encoded.items()))
    from kit.m2_gpu import assert_current
    assert_current()
    import torch
    from transformers import AutoModelForCausalLM
    net=AutoModelForCausalLM.from_pretrained(model,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda').eval()
    total=tokens=sequence=0
    with torch.inference_mode():
        for key,row in unique.items():
            pids,tids=encoded[key]
            require(len(pids)<=10240 and len(tids)<=2048,'NLL cap drift')
            ids=torch.tensor([pids+tids],device='cuda')
            logits=net(input_ids=ids,use_cache=False).logits[:,len(pids)-1:-1].float()
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),torch.tensor(tids,device='cuda'),reduction='sum').item()
            count=counts[key];total+=count*loss;tokens+=count*len(tids);sequence+=count*loss/len(tids)
    return dict(sequence_mean=sequence/len(rows),token_mean=total/tokens,exposures=len(rows),unique=len(unique),response_tokens=tokens)


def job(a):
    rows=t.read_jsonl(a.schedule);observed=dict(model=model_identity(dict(model=a.model,receipt=str(a.receipt) if a.receipt else None),a.standin),schedule=s.file_hash(a.schedule))
    bind(a.output.with_suffix('.identity.json'),observed)
    if a.output.exists(): print('SKIP: NLL',a.output);return
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    atomic(a.output,dict(measure(a.model,rows,a.standin),identity=observed,technical_synthetic=a.standin))


def configure(p):
    p.add_argument('--model',required=True);p.add_argument('--receipt',type=Path);p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',default='0')
if __name__=='__main__':raise SystemExit(entry(job,configure,'a4-nll'))
