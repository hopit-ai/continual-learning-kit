#!/usr/bin/env python3
"""Frozen likelihood scorer validation and strict-letter validity; no scheduler."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,math,re,os,time
from kit.m2_common import *
from kit import m2_assignment as assignment

POLICY=KIT/'resources/m2/format-validity.json'


def policy():
    value=json.loads(POLICY.read_text())
    for name,checksum in value['implementations'].items():check(KIT/name,checksum)
    check(KIT/'resources/m2/hellaswag-validation-items.json',value['hellaswag']['validation_items_sha256'])
    return value


def preprocessing(row):
    def clean(value):
        value=value.strip().replace(' [title] ','. ')
        value=re.sub(r'\[.*?\]','',value)
        return value.replace('  ',' ')
    context=clean(row['activity_label']+': '+row['ctx_a']+' '+row['ctx_b'].capitalize())
    return context,[clean(ending) for ending in row['endings']]


class TechnicalInvalid(ValueError):
    """Observed coverage/finiteness failure; infrastructure exceptions stay retryable."""


def finite_number(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def json_finite(value):
    """Archive undefined likelihoods as JSON null with finite_likelihoods=False."""
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {key:json_finite(child) for key,child in value.items()}
    if isinstance(value,list):return [json_finite(child) for child in value]
    return value


def selected(scores):
    if len(scores)!=4 or not all(finite_number(x) for x in scores):raise TechnicalInvalid('finite likelihoods required for every ending')
    return sorted(range(4),key=lambda i:(-scores[i],i))[0]


def items(out,standin):
    panel=t.read_jsonl(out/'hellaswag/panel.jsonl');ids={r['id'] for r in panel}
    require(len(ids)==100,'same frozen HellaSwag-100 required')
    if standin:
        return [dict(id=r['id'],activity_label='Technical',ctx_a='A fixture',ctx_b='continues',endings=['one','two','three','four'],label='ABCD'.index(r['gold'])) for r in panel]
    from kit.next_hellaswag import download,REVISION,VALIDATION_FILES
    raw,_=download('Rowan/hellaswag',REVISION,'validation',VALIDATION_FILES)
    chosen=[dict(r,id='hellaswag-val-'+str(r['ind'])) for r in raw if r['source_id'].startswith('activitynet~') and 'hellaswag-val-'+str(r['ind']) in ids]
    require(len(chosen)==100 and {r['id'] for r in chosen}==ids,'likelihood items differ from frozen panel')
    return sorted(chosen,key=lambda r:r['id'])


def tokenizer_identity(model):
    expected=policy()['tokenizer_files']
    from kit.m2_paths import local
    model=local(model)
    for name,checksum in expected.items():check(model/name,checksum)
    return expected


def _likelihoods(model,rows,tokenizer_path):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from kit.inference_lifecycle import marker,create_llm,shutdown_since
    tokenizer_identity(tokenizer_path)
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    requests=[];lengths=[];starts=[]
    for row in rows:
        context,endings=preprocessing(row);context_ids=tok.encode(context,add_special_tokens=False)
        if len(endings)!=4:raise TechnicalInvalid('four frozen endings required')
        for ending in endings:
            ids=tok.encode(context+' '+ending,add_special_tokens=False)
            if not (bool(context_ids) and ids[:len(context_ids)]==context_ids and len(ids)>len(context_ids) and bool(ending)):raise TechnicalInvalid('conditional token boundary changed')
            requests.append(dict(prompt_token_ids=ids));lengths.append(len(ending));starts.append(len(context_ids))
    mark=marker()
    try:
        engine=create_llm(model=str(model),tokenizer=str(tokenizer_path),tensor_parallel_size=1,distributed_executor_backend='uni',
            dtype='bfloat16',max_model_len=12288,max_num_seqs=1,gpu_memory_utilization=.9,enforce_eager=True,seed=0)
        outputs=engine.generate(requests,SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1),use_tqdm=False)
        if len(outputs)!=len(requests):raise TechnicalInvalid('complete item/choice likelihood coverage required')
        values=[]
        for request,output,start,denominator in zip(requests,outputs,starts,lengths):
            ids=request['prompt_token_ids'];logs=output.prompt_logprobs
            if len(logs)!=len(ids) or any(logs[j] is None or ids[j] not in logs[j] for j in range(start,len(ids))):raise TechnicalInvalid('complete prompt likelihood coverage required')
            values.append(sum(logs[j][ids[j]].logprob for j in range(start,len(ids)))/denominator)
        return [dict(id=row['id'],scores=values[i*4:i*4+4],selected=selected(values[i*4:i*4+4])) for i,row in enumerate(rows)]
    finally:shutdown_since(mark)


def likelihoods(model,rows,tokenizer_path):
    # Engine shutdown does not guarantee that vLLM/torch release their contexts.
    # Every caller (including on-node multi-checkpoint validation) gets a fresh
    # process which exits before the next GPU phase is allowed to begin.
    from kit.inference_subprocess import compute
    return compute('likelihood',model,rows=rows,tokenizer=tokenizer_path)


def validation(production,reference,expected_ids):
    ids=[r['id'] for r in production];refids=[r['id'] for r in reference]
    complete=len(ids)==len(set(ids))==100 and set(ids)==set(refids)==set(expected_ids) and len(refids)==100
    finite=complete and all(len(r['scores'])==4 and all(finite_number(x) for x in r['scores']) for r in production+reference)
    agreement=0
    if finite:
        ref={r['id']:r for r in reference}
        agreement=sum(selected(r['scores'])==selected(ref[r['id']]['scores']) for r in production)
    return dict(status='PASS' if complete and finite and agreement==100 else 'PENDING',
                complete_item_choice_coverage=complete,finite_likelihoods=finite,selected_choice_agreement=agreement,
                required_agreement=100,items=100,choices_per_item=4)


def validation_name(out,proof):return out/'format-validation'/f"{proof['family']}-s{proof['seed']}.json"


def require_validation(out,proof):
    path=validation_name(out,proof)
    if not path.exists():raise NotReady('format-free HellaSwag technical validation pending')
    value=json.loads(path.read_text())
    require(value['policy_sha256']==s.file_hash(POLICY) and value['assignment']==proof,'likelihood validation identity changed')
    if value['status']!='PASS':raise NotReady('HellaSwag likelihood implementation validation failed; affected decisions PENDING')
    return s.file_hash(path)


def job(a):
    doc=binding(a.out,a.standin);proof=assignment.active(a.out,a.gpu,a.standin)
    require(proof is not None,'assigned validation required')
    declared=policy();rows=items(a.out,a.standin);path=validation_name(a.out,proof)
    refs={'base':doc['starts']['base']}
    for key,value in doc['starts'].items():
        if key.endswith(':'+str(a.seed)):refs[key]=value
    identity=dict(policy_sha256=s.file_hash(POLICY),assignment=proof,checkpoints=refs,items_sha256=s.digest(rows))
    if path.exists():
        require(all(json.loads(path.read_text())[k]==v for k,v in identity.items()),'technical validation declaration changed');return
    checks=[]
    for name,ref in sorted(refs.items()):
        checked_model(ref,a.standin,base=name=='base')
        try:
            if a.standin:
                production=[dict(id=r['id'],scores=[0.,-1.,-2.,-3.],selected=0) for r in rows];reference=production
            else:
                from kit.m2_paths import local
                from kit import hellaswag_reference
                production=likelihoods(local(ref['model']),rows,local(doc['starts']['base']['model']))
                reference=hellaswag_reference.score(str(local(ref['model'])),rows,str(local(doc['starts']['base']['model'])))
            checks.append(dict(checkpoint=name,**validation(production,reference,[r['id'] for r in rows]),production=production,reference=reference))
        except TechnicalInvalid as error:checks.append(dict(checkpoint=name,status='PENDING',error=type(error).__name__+': '+str(error)))
    bind(path,dict(identity,status='PASS' if all(c['status']=='PASS' for c in checks) else 'PENDING',checks=json_finite(checks),
                   declaration=declared,technical_synthetic=a.standin))


def letter_cells(rows,arms=None):
    """Per-cell format diagnostics, including diagnostic-only HellaSwag."""
    cells={}
    for row in rows:
        if row['panel'] not in ('knowledge','hellaswag') or arms is not None and row['arm'] not in arms:continue
        key=f"{row['arm']}|{row['seed']}|{row['step']}|{row['panel']}"
        cell=cells.setdefault(key,dict(n=0,incorrect_format=0,known=True,diagnostic_only=row['panel']=='hellaswag'))
        cell['n']+=1
        value=row.get('incorrect_format')
        if value in (None,''):cell['known']=False
        else:cell['incorrect_format']+=int(value)
    for cell in cells.values():
        cell['incorrect_format_rate']=cell['incorrect_format']/cell['n'] if cell['known'] else None
    return cells


def letter_valid(rows,arms=None):
    cells=[v for v in letter_cells(rows,arms).values() if not v['diagnostic_only']]
    return bool(cells) and all(v['known'] and v['incorrect_format']==policy()['strict_letter']['max_incorrect_format'] for v in cells)


def configure(p):
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--gpu',default='0')
if __name__=='__main__':raise SystemExit(entry(job,configure,'format-validation-{seed}'))
