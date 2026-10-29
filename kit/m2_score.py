#!/usr/bin/env python3
"""All connected boundaries and MATH curves; original frozen task graders."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit.m2_gpu import prepared_index
from kit.c_score import legal_scores, COLUMNS
from kit.p3b_score import write_csv
from types import SimpleNamespace as NS
import time
import csv


def canonical_scores(root):
    """Pad inherited scorer columns before freezing the connected evidence tail."""
    manifest=frozen_files(root)
    path=root/'scores.csv'
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);columns=reader.fieldnames;rows=list(reader)
    require(columns and len(columns)==len(set(columns)) and set(columns)<=set(COLUMNS),
            'UNVERIFIED: unexpected scoring columns')
    require(all(None not in r for r in rows),'UNVERIFIED: malformed scoring row')
    if columns!=list(COLUMNS):
        write_csv(path,[{k:r.get(k,'') for k in COLUMNS} for r in rows],COLUMNS)
        manifest['files']['scores.csv']=s.file_hash(path)
        atomic(root/'manifest.json',manifest)


def items_for(panel, standin, panel_root=None):
    if panel in ('test','panel'): return legal_panels()[panel]
    if panel=='math_hard': return hard_data()['eval.jsonl']
    if panel=='general-v1':
        from kit.score_forgetting import load_panel
        return load_panel()[0]
    if panel=='hellaswag': return []
    if standin: return [dict(id=str(i),prompt='Technical finance question '+str(i),ground_truth='0') for i in range(1147)]
    from kit import eval_bed
    from kit.v4_contract import check_eval_items
    require(panel_root is not None,'FinQA panel root required')
    args=NS(bed='finqa',root=str(panel_root),split='test',limit=None,heldout_n=None,allow_subset=False)
    items=eval_bed.items_of(eval_bed.load_bed('finqa'),args)
    check_eval_items(items,'finqa','test'); require(len(items)==1147,'FinQA requires all 1147 questions')
    return items


def job(a):
    doc=binding(a.out,a.standin)
    from kit.m2_handoff import scoring_identity
    gpu=scoring_identity(a.out,a.gpu,a.standin)
    if gpu.get('assignment') and 'c_exports' in doc and not doc.get('purpose'):
        require(gpu['assignment']['family']=='core' and gpu['assignment']['seed']==a.seed,'UNVERIFIED: core family/seed mismatch')
    source=record(a.out,doc,a.arm,a.seed,a.step)
    from kit.m2_paths import local
    source=dict(source,model=str(local(source['model'])),receipt=str(local(source['receipt'])) if source.get('receipt') else None)
    observed=checked_model(source,a.standin,base=a.arm=='base')
    root=a.out/'scores'/f'{a.arm}-s{a.seed}-k{a.step}-{a.panel_set}'
    meta=dict(model=observed,gpu_identity=gpu,assignment=gpu.get('assignment'),protocol=protocol(),binding_sha256=s.file_hash(a.out/'binding.json'),
              arm=a.arm,seed=a.seed,step=a.step,panel=a.panel_set,technical_synthetic=a.standin)
    meta['decoding']=dict(seed=0,temperature=0,thinking=False,max_new_tokens=2048,eager=True,batch_invariant=True)
    meta['panel_sha256']=s.digest(items_for(a.panel_set,a.standin,a.panel_root)) if a.panel_set!='hellaswag' else s.file_hash(a.out/'hellaswag/manifest.json')
    bind(a.out/'m2-score-identities'/(root.name+'.json'),meta)
    complete=(root/'manifest.json').exists()
    if complete:
        manifest=frozen_files(root)
        if 'm2-identity.json' in manifest['files']:
            print('SKIP:',root.name); return
        print('RESUME connected identity tail:',root.name)
    if not complete:
        if a.panel_set=='hellaswag':
            from kit.next_hellaswag import score
            args=NS(**vars(a)); args.model=source['model']; args.checkpoint_receipt=Path(source['receipt']) if source.get('receipt') else None
            if a.standin:
                model=a.out/'standin-models'/root.name; atomic(model/'config.json',dict(fixture=True))
                receipt=model/'receipt.json'; atomic(receipt,dict(technical_synthetic=True,model_file_hashes={'config.json':s.file_hash(model/'config.json')}))
                args.model=str(model); args.checkpoint_receipt=receipt
            if a.standin:score(args)
            else:
                # The strict diagnostic engine must exit before the likelihood
                # worker starts; vLLM shutdown alone left live GPU allocations.
                import subprocess
                command=[sys.executable,str(KIT/'next_hellaswag.py'),'--out',str(a.out),'--action','score',
                    '--model',str(args.model),'--arm',a.arm,'--seed',str(a.seed),'--step',str(a.step),'--gpu',str(a.gpu)]
                if args.checkpoint_receipt:command+=['--checkpoint-receipt',str(args.checkpoint_receipt)]
                subprocess.run(command,check=True)
        elif not a.standin and a.panel_set in ('finqa','general-v1'):
            from kit.p3b_score import job as score
            args=NS(**vars(a),model=source['model'],checkpoint_receipt=Path(source['receipt']) if source.get('receipt') else None,package_a=None)
            score(args)
        else:
            items=items_for(a.panel_set,a.standin,a.panel_root)
            retain_partial(root); root.mkdir(parents=True,exist_ok=True); started=time.monotonic()
            os.environ.update(CUDA_VISIBLE_DEVICES=prepared_index(a.gpu,a.standin),KIT_DETERMINISTIC='1',VLLM_BATCH_INVARIANT='1',VLLM_ATTENTION_BACKEND='FLASH_ATTN')
            if a.standin:
                from kit.m2_sim import responses
                raw=responses(items,a.arm,a.step,'test' if a.panel_set=='panel' else a.panel_set)
            elif a.panel_set in ('test','panel'):
                from kit import score_forgetting as general
                from transformers import AutoTokenizer
                from vllm import SamplingParams
                from kit.inference_lifecycle import create_llm as LLM
                tokenizer=AutoTokenizer.from_pretrained(source['model'],local_files_only=True)
                prompts=[legal.render(tokenizer,r['prompt']) for r in items]
                require(all(len(tokenizer.encode(p,add_special_tokens=False))<=10240 for p in prompts),'legal input exceeds frozen cap')
                engine=LLM(model=source['model'],enable_lora=False,**dict(general.ENGINE,max_model_len=12288),enforce_eager=True,seed=0)
                outputs=engine.generate(prompts,SamplingParams(n=1,**general.DECODING))
                require(len(outputs)==len(items),'incomplete legal scoring batch')
                raw=[dict(id=r['id'],response=o.outputs[0].text,output_tokens=len(o.outputs[0].token_ids),
                          token_ids=list(o.outputs[0].token_ids),prompt_tokens=len(o.prompt_token_ids),finish_reason=o.outputs[0].finish_reason) for r,o in zip(items,outputs)]
            else:
                from kit import eval_bed
                args=NS(bed='math_screen',root=None,split='test',model=source['model'],max_new_tokens=2048,max_model_len=12288,
                        out=str(root/'raw'),limit=None,heldout_n=None,allow_subset=False)
                eval_bed.cmd_generate(args,frozen_items=[dict(r,ground_truth=r['gold']) for r in items])
                raw=t.read_jsonl(root/'raw/responses.jsonl')
            require(len(raw)==len(items) and {r['id'] for r in raw}=={r['id'] for r in items},'incomplete/foreign score responses')
            if a.panel_set in ('test','panel'): rows=legal_scores(items,raw,a.arm,a.seed,a.step,a.panel_set)
            else:
                from kit.beds import math_screen
                lookup={r['id']:r for r in items}; rows=[]
                for r in raw:
                    item=lookup[r['id']]; panel=item.get('panel',a.panel_set)
                    correct=r['standin_correct'] if a.standin else math_screen.compute_score(math_screen.DATA_SOURCE,r['response'],item['gold'])['score']
                    rows.append(dict(arm=a.arm,seed=a.seed,step=a.step,task=a.panel_set,panel=panel,cap=2048,question_id=r['id'],
                                     correct=int(correct),output_tokens=r['output_tokens'],finished=int(r['finish_reason'] in ('stop','eos'))))
            (root/'raw').mkdir(exist_ok=True)
            t.write_jsonl(root/'raw/responses.jsonl',raw)
            write_csv(root/'scores.csv',[{k:r.get(k,'') for k in COLUMNS} for r in rows],COLUMNS)
            atomic(root/'cost.json',dict(kind='scoring',gpu_hours=0 if a.standin else (time.monotonic()-started)/3600,technical_synthetic=a.standin))
            atomic(root/'manifest.json',files_manifest(root,['scores.csv','cost.json','raw/responses.jsonl']))
    canonical_scores(root)
    if a.panel_set=='hellaswag':
        from kit import m2_format as formatting
        proof=gpu.get('assignment')
        validation_sha=formatting.require_validation(a.out,proof) if not a.standin else None
        population=formatting.items(a.out,a.standin)
        if a.standin:
            likelihood=[dict(id=r['id'],scores=[0.,-1.,-2.,-3.],selected=0) for r in population]
        else:likelihood=formatting.likelihoods(source['model'],population,__import__('kit.m2_paths',fromlist=['local']).local(doc.get('base',doc.get('starts',{}).get('base'))['model']))
        require(len(likelihood)==100 and {r['id'] for r in likelihood}=={r['id'] for r in population},'complete HellaSwag likelihood coverage required')
        lookup={r['id']:r for r in likelihood};golds={r['id']:int(r['label']) for r in population}
        raw_lookup={r['id']:r for r in t.read_jsonl(root/'responses.jsonl')}
        from kit.next_hellaswag import strict_label
        manifest=frozen_files(root)
        with (root/'scores.csv').open(newline='') as stream:scored=list(csv.DictReader(stream))
        for r in scored:
            ident=r['question_id'];r['strict_diagnostic_correct']=r['correct']
            r['incorrect_format']=int(strict_label(raw_lookup[ident]['response']) is None)
            r['format_free_correct']=int(formatting.selected(lookup[ident]['scores'])==golds[ident])
            r['correct']=r['format_free_correct']
        write_csv(root/'scores.csv',scored,COLUMNS)
        atomic(root/'likelihoods.json',dict(policy_sha256=s.file_hash(formatting.POLICY),validation_sha256=validation_sha,
            items_sha256=s.digest(population),rows=likelihood))
        meta['format_free']=dict(policy_sha256=s.file_hash(formatting.POLICY),validation_sha256=validation_sha,
            scorer='character-normalised conditional log likelihood',tokenizer=formatting.policy()['tokenizer_files'])
        manifest['files'].update({'scores.csv':s.file_hash(root/'scores.csv'),'likelihoods.json':s.file_hash(root/'likelihoods.json')})
        atomic(root/'manifest.json',manifest)
    if a.panel_set=='general-v1':
        from kit.scorers import extract_letter_strict
        raw_rows=t.read_jsonl(root/'raw/responses.jsonl');lookup={r['id']:r for r in raw_rows}
        manifest=frozen_files(root)
        with (root/'scores.csv').open(newline='') as stream:scored=list(csv.DictReader(stream))
        for r in scored:
            if r['panel']=='knowledge':r['incorrect_format']=0 if a.standin else int(extract_letter_strict(lookup[r['question_id']]['response']) is None)
        write_csv(root/'scores.csv',scored,COLUMNS)
        manifest['files']['scores.csv']=s.file_hash(root/'scores.csv');atomic(root/'manifest.json',manifest)
    raw_path=next((p for p in (root/'raw/responses.jsonl',root/'responses.jsonl') if p.exists()),None)
    if raw_path is not None:
        from collections import Counter
        raw_rows=t.read_jsonl(raw_path)
        finishes=Counter(str(r.get('finish_reason','missing')) for r in raw_rows)
        atomic(root/'generation-summary.json',dict(n=len(raw_rows),finish_reasons=dict(finishes),
            truncation_rate=finishes.get('length',0)/len(raw_rows) if raw_rows else None,
            output_tokens=sum(r['output_tokens'] for r in raw_rows)))
    meta['decoding']=dict(seed=0,temperature=0,thinking=False,max_new_tokens=2048,eager=True,batch_invariant=True)
    atomic(root/'m2-identity.json',meta)
    manifest=frozen_files(root); manifest['files']['m2-identity.json']=s.file_hash(root/'m2-identity.json')
    if (root/'generation-summary.json').exists(): manifest['files']['generation-summary.json']=s.file_hash(root/'generation-summary.json')
    atomic(root/'manifest.json',manifest)


def configure(p):
    p.add_argument('--arm',required=True); p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--step',type=int,choices=STEPS,required=True); p.add_argument('--panel-set',choices=(*PANEL_SETS,'panel','hellaswag'),required=True)
    p.add_argument('--gpu',default='0'); p.add_argument('--panel-root',type=Path)
if __name__=='__main__': raise SystemExit(entry(job,configure,'m2-score-{arm}-{seed}-{step}-{panel_set}'))
