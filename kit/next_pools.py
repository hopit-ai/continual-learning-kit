#!/usr/bin/env python3
"""Prepare broad base-success replay and unique B 27B FinQA targets; one GPU job."""
from pathlib import Path
import sys
if __package__ in (None, ''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.next_common import *
from kit.next_hellaswag import download, strict_label
from kit.p3b_replay_data import clean
from kit.beds.gsm8k import PANEL_PROMPT_SUFFIX
from kit.scorers import extract_number
from fractions import Fraction
import time

GSM='740312add88f781978c0658806c59bc2815b9866'
ARC='210d026faf9955653af8916fad021475a3f00453'
ARC_PROMPT = '\nDo not use Markdown (no bold, italics or headings) anywhere in your reply. End with a final line that is exactly: Answer: X (where X is the correct choice label).'
POOL_PROTOCOL = 'message2-v2-strict-arc-prompt'
NOTICES=dict(gsm8k='GSM8K, Cobbe et al. 2021, openai/gsm8k; MIT License; https://github.com/openai/grade-school-math',
    arc='ARC-Challenge, Clark et al. 2018, allenai/ai2_arc; CC-BY-SA-4.0; https://creativecommons.org/licenses/by-sa/4.0/; prompts reformatted; generated responses checked for final labels only',
    earlier='FinQA, Chen et al. 2021; MIT License; https://github.com/czyssrs/FinQA; verified Qwen3.6-27B targets from Package B')


def exclude(rows,panels,limit=None):
    # B's equality/shingle/edit rules operate on message content, preserving training chats.
    def text(row):
        prompt=row['prompt']
        return dict(row,prompt='\n'.join(m['content'] for m in prompt) if isinstance(prompt,list) else prompt)
    kept,excluded=clean([text(r) for r in rows],[text(r) for r in panels],limit=limit)
    original={r['id']:r for r in rows}
    return [original[r['id']] for r in kept],excluded


def candidates(gsm,arc):
    rows=[]
    for i,r in enumerate(gsm):
        rows.append(dict(id=f'gsm8k-train-{i}',order_id=str(i),source='gsm8k',prompt=r['question']+PANEL_PROMPT_SUFFIX,
            gold=r['answer'].split('####')[-1].strip(),licence_notice=NOTICES['gsm8k'],**({'fake_response':r['fake_response']} if 'fake_response' in r else {})))
    for r in arc:
        labels=r['choices']['label'];texts=r['choices']['text']
        require(r['answerKey'] in labels and len(labels)==len(set(labels)),'malformed ARC choices')
        rows.append(dict(id='arc-c-train-'+r['id'],order_id=r['id'],source='arc',labels=''.join(labels),
            prompt=r['question']+'\n'+'\n'.join(f'{k}. {v}' for k,v in zip(labels,texts))+ARC_PROMPT,
            gold=r['answerKey'],licence_notice=NOTICES['arc'],**({'fake_response':r['fake_response']} if 'fake_response' in r else {})))
    return rows


def correct(row,text):
    if row['source']=='arc':return strict_label(text,row['labels'])==row['gold']
    pred=extract_number(text)
    try:return pred is not None and Fraction(pred)==Fraction(row['gold'].replace(',',''))
    except (ValueError,ZeroDivisionError):return False


def earlier_rows(paths,standin,tokenizer=None):
    unique={}; receipts={}
    for folder in paths:
        manifest=json.loads((folder/'manifest.json').read_text())
        check(folder/'train.parquet',manifest['train_sha256'])
        require(manifest.get('technical_synthetic',False)==standin,'earlier target type mismatch')
        if not standin:
            require(manifest.get('schema')=='kit-v4-sft-data.v1' and manifest.get('arm')=='COMMON' and
                manifest.get('scientific_phase_allowed') is True and not manifest.get('stand_in_teacher'),'unqualified B targets')
        import pyarrow.parquet as pq
        raw=pq.read_table(folder/'train.parquet').to_pylist()
        receipts[str(folder.resolve())]=dict(manifest=s.file_hash(folder/'manifest.json'),train=manifest['train_sha256'])
        for r in raw:
            x=r['extra_info'];require(x['split']=='train' and r['data_source']=='finqa','earlier pool must contain FinQA training only')
            response=x['demonstration'];ident=str(x['index']);prompt=r['prompt']
            if isinstance(prompt,str) and prompt.startswith('['):prompt=json.loads(prompt)
            if not standin:
                require('Qwen3.6-27B' in x.get('teacher_model','') and not x.get('stand_in_teacher'),'not a 27B teacher target')
                item=dict(id=ident,task='finqa',messages=prompt,gold=r['reward_model']['ground_truth'])
                verdict=t.verify(item,dict(text=response,attempt=1,finish_reason='stop'),tokenizer,mapped_target=True)
                require(verdict['verified'],'B target fails original verifier')
            value=dict(id=ident,prompt=prompt,response=response,source='earlier',licence_notice=NOTICES['earlier'])
            require(ident not in unique or unique[ident]==value,'inconsistent B target across seeds')
            unique[ident]=value
    return sorted(unique.values(),key=lambda r:r['id']),receipts


def finqa_exclusions(rows, standin=False):
    from kit.v4_contract import check_eval_items,check_heldout_rows
    from kit.v4_finqa_policy import CorrectnessStop as ValidationStop
    require(bool(rows),'empty FinQA panel')
    native=all('extra_info' in r for r in rows)
    require(native or all('extra_info' not in r for r in rows),'mixed FinQA schemas')
    try:
        if native:
            if not standin:check_heldout_rows(rows)
            rows=[dict(id=str(r['extra_info']['index']),prompt=r['prompt']) for r in rows]
        elif not standin:check_eval_items(rows,'finqa','test')
    except ValidationStop as exc:
        raise CorrectnessStop(str(exc)) from exc
    require(len({r['id'] for r in rows})==len(rows),'duplicate FinQA exclusion IDs')
    if not standin:require(len(rows)==1147,'all FinQA exclusions required')
    return rows


def audit_generic(root, generic, panels, standin):
    # Never silently change B's inherited replay pool; any new overlap stops the probe.
    _,excluded=exclude(generic,panels)
    receipt=dict(generic=s.digest(generic),panels=s.digest(panels),exclusions=excluded,
                 passed=not excluded,technical_synthetic=standin)
    bind(root/'generic-exclusion-audit.json',receipt)
    require(not excluded,'inherited generic replay overlaps evaluation panels; see generic-exclusion-audit.json')


def job(a):
    frozen_next();require(',' not in a.gpu,'one generation GPU')
    panel_doc=frozen_files(a.out/'hellaswag');require(panel_doc['technical_synthetic']==a.standin,'holdout type mismatch')
    inputs=read_inputs(a.out,a.standin);root=a.out/'pools'
    from kit.p3a_common import frozen,INPUTS
    frozen(a.package_a,a.standin)
    finqa=finqa_exclusions(t.read_jsonl(a.finqa_panel),a.standin)
    panels=finqa+t.read_jsonl(KIT/'panels/general-v1.jsonl')+t.read_jsonl(a.out/'hellaswag/panel.jsonl')
    math=t.read_jsonl(a.package_a/INPUTS/'prepared/math_screen/heldout.jsonl')
    panels += [dict(id=r['id'],prompt=r['prompt']) for r in math]
    from kit.c_common import panels as legal_panels
    panels += [r for values in legal_panels().values() for r in values]
    if not a.standin:require(len(math)==100,'all frozen MATH evaluations required')
    base=checkpoint(inputs['base']['path'],inputs['base']['receipt'],a.standin,base=True)
    if a.standin:
        require(a.gsm_rows and a.arc_rows,'fixtures required')
        gsm,arc=t.read_jsonl(a.gsm_rows),t.read_jsonl(a.arc_rows)
        sources=dict(gsm8k=s.file_hash(a.gsm_rows),arc=s.file_hash(a.arc_rows))
        tokenizer=None
    else:
        require(not a.gsm_rows and not a.arc_rows,'real pools require pinned sources')
        gsm,gh=download('openai/gsm8k',GSM,'main/train-')
        arc,ah=download('allenai/ai2_arc',ARC,'ARC-Challenge/train-')
        sources=dict(gsm8k=gh,arc=ah)
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(__import__('kit.m2_paths',fromlist=['local']).local(inputs['base']['path']),local_files_only=True)
    earlier,proof=earlier_rows(a.finqa_targets,a.standin,tokenizer)
    generic=t.read_jsonl(a.out/'replay/replay.jsonl')
    audit_generic(root,generic,panels,a.standin)
    earlier,excluded=exclude(earlier,panels+generic)
    require(bool(earlier),'empty earlier-task pool; inputs incomplete')
    prepared=[]
    raw=candidates(gsm,arc)
    for source in ('gsm8k','arc'):
        ordered=sorted((r for r in raw if r['source']==source),key=lambda r:(t.sha('next-v4-broad-v1|'+source+'|'+r['order_id']),r['order_id']))
        kept,ex=exclude(ordered,panels+generic+earlier+prepared,limit=1024)
        prepared+=kept;excluded+=ex
    provenance=dict(pool_protocol=POOL_PROTOCOL, arc_prompt=ARC_PROMPT, base=base,sources=sources,revisions=dict(gsm8k=GSM,arc=ARC),panels=s.digest(panels),
        generic=s.file_hash(a.out/'replay/replay.jsonl'),earlier_receipts=proof,technical_synthetic=a.standin,
        candidate_preparation=dict(fresh_exclusions_rerun=True,exclusions_sha256=s.digest(excluded),candidate_rows=len(prepared)))
    bind(root/'inputs.json',provenance)
    if (root/'manifest.json').exists():frozen_files(root);print('SKIP pools');return
    write_rows(root/'candidates.jsonl',prepared);write_rows(root/'exclusions.jsonl',excluded)
    write_rows(root/'earlier.jsonl',earlier)
    if not a.standin:
        from kit.arc_prompt_acceptance import validate_candidate_membership
        try:acceptance=validate_candidate_membership(root/'candidates.jsonl')
        except NotReady as error:
            atomic(root/'arc-candidate-status.json',dict(status='PENDING',reason=str(error),no_replenishment=True))
            raise
        bind(root/'arc-candidate-acceptance.json',acceptance)
        atomic(root/'arc-candidate-status.json',dict(status='PASS',**acceptance))
    if getattr(a,'prepare_candidates_only',False):
        atomic(root/'candidate-manifest.json',files_manifest(root,['inputs.json','candidates.jsonl','exclusions.jsonl','earlier.jsonl'],
            pool_protocol=POOL_PROTOCOL,arc_prompt=ARC_PROMPT,technical_synthetic=a.standin,fresh_exclusions_rerun=True))
        print('CPU candidates frozen; no model generation started');return
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,V4_PHASE0_INFERENCE='0')
    started=time.monotonic()
    generator=None;accepted=[];verdicts=[]
    for row in prepared:
        path=root/'raw'/(t.sha(row['id'])+'.json')
        if path.exists():output=json.loads(path.read_text())
        elif a.standin:output=dict(text=row['fake_response'],finish_reason='stop');atomic(path,output)
        else:
            if generator is None:generator=t.VLLMGenerator(inputs['base']['path'],2048,tensor_parallel_size=1,allow_batch_fallback=False)
            output=generator([dict(role='user',content=row['prompt'])],1,seed=0,temperature=0,top_p=1);atomic(path,output)
        passed=output['finish_reason'] in ('stop','eos') and correct(row,output['text'])
        if passed and tokenizer is not None:
            from kit.v4_sft import encode_example
            try:encode_example(tokenizer,[dict(role='user',content=row['prompt'])],output['text'])
            except ValueError:passed=False
        verdicts.append(dict(id=row['id'],accepted=passed,response_sha256=s.file_hash(path)))
        if passed:accepted.append(dict(id=row['id'],prompt=row['prompt'],response=output['text'],source=row['source'],licence_notice=row['licence_notice']))
    for source in ('gsm8k','arc'):
        pool=[r for r in accepted if r['source']==source]
        write_rows(root/(source+'.jsonl'),pool)
        require(bool(pool),'empty '+source+' pool; inputs incomplete; no replenishment')
    write_rows(root/'checks.jsonl',verdicts)
    atomic(root/'licences.json',NOTICES)
    atomic(root/'cost.json',dict(kind='broad_preparation',gpu_hours=0 if a.standin else (time.monotonic()-started)/3600,technical_synthetic=a.standin))
    freeze(root,['generic-exclusion-audit.json','inputs.json','candidates.jsonl','exclusions.jsonl','earlier.jsonl','gsm8k.jsonl','arc.jsonl','checks.jsonl','licences.json','cost.json']+
        [str(p.relative_to(root)) for p in sorted((root/'raw').glob('*.json'))],a.standin,
        pool_protocol=POOL_PROTOCOL, arc_prompt=ARC_PROMPT,
        disclosure='Broad is in-distribution for general-v1 maths and near-distribution for knowledge. HellaSwag is the only out-of-pool panel; ARC shares its letter-answer format. Checks verify final answers, not rationales.')
    print('Pool rows:',len(accepted),'broad;',len(earlier),'earlier;',len(excluded),'excluded')


def configure(p):
    p.add_argument('--prepare-candidates-only',action='store_true')
    p.add_argument('--gpu',default='0');p.add_argument('--package-a',type=Path,required=True)
    p.add_argument('--finqa-panel',type=Path,required=True);p.add_argument('--finqa-targets',type=Path,nargs='+',required=True)
    p.add_argument('--gsm-rows',type=Path);p.add_argument('--arc-rows',type=Path)
if __name__ == '__main__':raise SystemExit(entry(job,configure,'next-pools'))
