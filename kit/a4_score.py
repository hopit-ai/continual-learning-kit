#!/usr/bin/env python3
"""One physical GPU, one attempt, pinned scorers and archived arrival contexts."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
from kit.a4_memory import contexts
from kit.p3b_score import write_csv

COLUMNS = 'kind arm seed month panel mode generation_id question_id correct historical_correct input_tokens output_tokens finished holdout family issuer recurrence recurring_issuer stratum response_sha256'.split()


def members(out,doc,panel,mode,standin):
    from kit import score_forgetting as general
    if panel in MONTHS:
        ctx=contexts(out,panel,mode)
        return [dict(id=i['id'], prompt=ctx[i['id']]['prompt'], panel=panel, item=i, context=ctx[i['id']]) for i in items(doc,panel)]
    require(panel=='general-v1','unregistered panel')
    if standin:
        return [dict(id=f'{p}-{i}',prompt=f'CPU fixture {p} {i}',panel=p) for p in
            ('ifeval','knowledge','math') for i in range(10)]
    if panel=='general-v1':
        rows,digest=general.load_panel(); require(digest==doc['retention']['general_v1'],'general panel changed');return rows


def render_prompts(tok,rows):
    from kit.eval_bed import render_messages
    from kit.score_forgetting import render
    return [render(tok,r['prompt']) if isinstance(r['prompt'],str) else render_messages(tok,r['prompt']) for r in rows]


def generate(rows,record,gpu,standin,arm,kind):
    if standin:
        raw=[]
        for n,r in enumerate(rows):
            right=(n%6!=0) if record['model']!='synthetic-base' else (n%5!=0) if 'Example ' in render_prompts(tokenizer({},True),[r])[0] else (n%4!=0)
            text=('Verified arithmetic.\nAnswer: '+(r['item']['qa']['answer'] if right else '987654321.0')) if 'item' in r else 'fixture answer'
            raw.append(dict(id=r['id'],panel=r['panel'],response=text,input_tokens=100,output_tokens=20,
                            token_ids=list(range(20)),finish_reason='stop',fixture_correct=int(right)))
        return raw
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,KIT_DETERMINISTIC='1',VLLM_BATCH_INVARIANT='1',VLLM_ATTENTION_BACKEND='FLASH_ATTN')
    from kit import score_forgetting as general
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from kit.inference_lifecycle import create_llm as LLM
    tok=AutoTokenizer.from_pretrained(record['model'],local_files_only=True)
    prompts=render_prompts(tok,rows)
    require(all(len(tok.encode(p,add_special_tokens=False))<=10240 for p in prompts),'scoring prompt over cap')
    engine=LLM(model=record['model'],enable_lora=False,**dict(general.ENGINE,max_model_len=12288),enforce_eager=True,seed=0)
    outputs=engine.generate(prompts,SamplingParams(n=1,**general.DECODING))
    require(len(outputs)==len(rows),'incomplete scoring batch')
    return [dict(id=r['id'],panel=r['panel'],response=o.outputs[0].text, input_tokens=len(o.prompt_token_ids),
        output_tokens=len(o.outputs[0].token_ids),token_ids=list(o.outputs[0].token_ids),finish_reason=o.outputs[0].finish_reason)
        for r,o in zip(rows,outputs)]


def graded(rows,raw,meta,standin):
    from kit import scorers
    lookup={(r['panel'],r['id']):r for r in raw}
    require(len(lookup)==len(raw)==len(rows) and set(lookup)=={(r['panel'],r['id']) for r in rows},'duplicate/missing/unknown answers')
    result=[]
    for i in rows:
        r=lookup[i['panel'],i['id']]; text=r['response']; item=i.get('item'); sc=(item or {}).get('sec',{})
        if item:
            historical=int(finqa.is_correct(finqa.extract_answer(text),finqa.gold_of(item)))
            correct=int(sec.v4_explicit_correct(item,text)) if sc else historical
        else:
            correct=r['fixture_correct'] if standin else int(scorers.score(i['panel'],text,i)['correct']);historical=correct
        result.append(dict(meta, panel=i['panel'], question_id=i['id'], correct=correct,historical_correct=historical,
            input_tokens=r['input_tokens'],output_tokens=r['output_tokens'],finished=int(r['finish_reason'] in ('stop','eos')),
            holdout=int(sc.get('issuer_holdout',False)),family=sc.get('template',''),issuer=sc.get('issuer_group',''),
            recurrence=i.get('context',{}).get('recurrence',0),stratum=i.get('context',{}).get('stratum',''),response_sha256=t.sha(text.encode())))
        result[-1]['recurring_issuer']=int(i.get('context',{}).get('recurring_issuer',False))
    return result


def _job(a):
    doc=binding(a.out,a.standin)
    require(t.runtime_versions()==doc['software']['inference'],'bound inference environment changed')
    physical=machine(a.out,a.gpu,a.standin); n=a.month
    require(0<=n<=6 and a.panel in (*MONTHS,'general-v1'),'unregistered score row')
    if a.kind=='arrival':
        require(n>=1 and a.panel==MONTHS[n-1] and (a.arm,a.seed,a.mode) in set(arrival_specs()),'unregistered arrival')
        require(not (a.out/'ledger'/MONTHS[n-1]/'manifest.json').exists() or
                (score_root(a.out,a.kind,a.arm,a.seed,n,a.panel,a.mode)/'manifest.json').exists(),'cannot create arrival after feedback')
    else:
        require(a.mode=='off' or (a.arm=='consolidation' and a.panel in MONTHS),'unregistered memory mode')
        require(a.panel not in MONTHS or MONTHS.index(a.panel)<(6 if a.kind=='final' else n),'future holdout')
    if a.arm in ('frozen','memory'):
        record=doc['base'];require(a.kind in ('arrival','base'),'invalid deterministic arm row')
    else:
        require(a.arm in ARMS and a.seed in SEEDS,'unregistered trainable arm')
        record=state(a.out,doc,a.arm,a.seed,n-1 if a.kind in ('arrival','incoming') else n)
    data=members(a.out,doc,a.panel,a.mode,a.standin)
    if a.kind!='arrival' and a.panel in MONTHS: data=[r for r in data if r['item']['sec']['issuer_holdout']]
    observed=model_identity(record,a.standin)
    root=score_root(a.out,a.kind,a.arm,a.seed,n,a.panel,a.mode)
    ident=dict(model=observed, data=s.digest(data), physical=physical,binding=s.file_hash(a.out/'binding.json'),
               technical_synthetic=a.standin)
    bind(a.out/'score-identities'/(root.name+'.json'),ident)
    if (root/'manifest.json').exists(): frozen_files(root);print('SKIP:',root.name);return
    attempt=start_attempt(a.out,doc,'scoring',a.standin,arm=a.arm,seed=a.seed,kind_row=a.kind)
    complete=False;generated=False;generation_key=None
    try:
        with job_deadline(doc['job_limits']['scoring']):
            root.mkdir(parents=True,exist_ok=True)
            # Preserve complete generations across a scorer/readout crash; never resample them.
            rawfile=root/'raw/responses.jsonl'
            generation_key=s.digest(dict(model=observed.get('model_file_hashes',observed),physical=physical,
                prompts=[dict(id=r['id'],panel=r['panel'],prompt=r['prompt']) for r in data],protocol=doc['protocol']))
            if rawfile.exists() and (root/'generated.json').exists():
                check(rawfile,json.loads((root/'generated.json').read_text())['sha256']);raw=t.read_jsonl(rawfile)
            else:
                # Same weights, exact prompts, physical GPU and scorers are one deterministic generation.
                # Aliases retain scientific arm/mode labels and each comparison's own arrival reference.
                cached=a.out/'generation-cache'/generation_key
                if (cached/'manifest.json').exists():
                    frozen_files(cached);raw=t.read_jsonl(cached/'responses.jsonl')
                else:
                    if cached.exists():retain_partial(cached)
                    generated=True
                    raw=generate(data,record,a.gpu,a.standin,a.arm,a.kind)
                    cached.mkdir(parents=True,exist_ok=True);t.write_jsonl(cached/'responses.jsonl',raw)
                    freeze(cached,['responses.jsonl'])
                rawfile.parent.mkdir(parents=True,exist_ok=True)
                if rawfile.exists():rawfile.rename(rawfile.with_name(f'responses.partial-{time.time_ns()}.jsonl'))
                t.write_jsonl(rawfile,raw)
                atomic(root/'generated.json',dict(sha256=s.file_hash(rawfile),cache_key=generation_key,reused=not generated))
            generation=json.loads((root/'generated.json').read_text())
            meta=dict(kind=a.kind,arm=a.arm,seed=a.seed,month=n,mode=a.mode,generation_id=generation['cache_key'])
            result=graded(data,raw,meta,a.standin)
            write_csv(root/'scores.csv',result,COLUMNS)
            atomic(root/'identity.json',ident)
            freeze(root,['scores.csv','identity.json','generated.json','raw/responses.jsonl'])
            complete=True
    finally: finish_attempt(a.out,attempt,completed=complete,generated=generated,generation_id=generation_key)


def job(a):
    with lock(a.out/'node-writer'):return _job(a)


def configure(p):
    p.add_argument('--kind',choices=('arrival','base','incoming','post','final'),required=True)
    p.add_argument('--arm',choices=('frozen','memory',*ARMS),required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--month',type=int,required=True);p.add_argument('--panel',required=True)
    p.add_argument('--mode',choices=('on','off'),required=True);p.add_argument('--gpu',default='0')
if __name__=='__main__': raise SystemExit(entry(job,configure,'a4-score-{kind}-{arm}-{seed}-{month}-{panel}-{mode}'))
