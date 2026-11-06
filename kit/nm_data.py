"""Prospective schedules, OPR refresh/fallback and development-only contexts."""
from collections import Counter
from difflib import SequenceMatcher
import random
import re
from kit.nm_common import *


def overlap(a,b):
    def norm(v):return ' '.join(re.findall(r'\w+',v.casefold()))
    a,b=norm(a),norm(b)
    def shingles(v):
        w=v.split();return {tuple(w[i:i+5]) for i in range(max(0,len(w)-4))}
    x,y=shingles(a),shingles(b)
    return a==b or bool(x and y and len(x&y)/len(x|y)>=.8) or SequenceMatcher(None,a,b,autojunk=False).ratio()>=.9


def audit_candidates(candidates,excluded):
    """Audit whole candidate supersets before selecting schedules or Fisher rows."""
    bad=[(a['id'],b['id']) for a in candidates for b in excluded if overlap(a['text'],b['text'])]
    need(not bad,'UNVERIFIED: candidate/evaluation overlap '+str(bad[:3]))
    return dict(candidates=digest(candidates),excluded=digest(excluded),thresholds=dict(jaccard=.8,edit=.9))


def hash_first(rows,prefix,n):
    need(len({r['id'] for r in rows})==len(rows),'duplicate candidate ID')
    ready(len(rows)>=n,'candidate shortfall')
    return sorted(rows,key=lambda r:raw_hash((prefix+'|'+str(r['id'])).encode()))[:n]


def fisher_rows(domains):
    result=[]
    for domain,n in [('instruction',128),('maths',64),('knowledge',64)]:
        pool=domains.get(domain,[])
        need(all(r.get('split') in ('train','proxy') and r.get('verified_base_response') and r.get('base_revision')==BASE for r in pool),'Fisher requires frozen training/proxy base targets')
        result.extend(dict(r,domain=domain) for r in hash_first(pool,'fisher-20261102|'+domain,n))
    return result


def grpo_schedule(rows,seed):
    need(seed in SEEDS,'unregistered seed')
    need(all(r.get('split')=='train' and r.get('demonstration') for r in rows),'GRPO needs training prompts and existing verified D targets')
    selected=hash_first(rows,'grpo-qual-20261102|'+str(seed),320)
    return [dict(row,update=i//8+1,position=(i%8)*4+j+1,sample=j,
                 rng=int(raw_hash(f'grpo-20261102|{seed}|{i//8+1}|{row["id"]}|{j}'.encode())[:16],16)%(2**31))
            for i,row in enumerate(selected) for j in range(4)]


def generation_seed(seed,prompt_id,attempt):
    need(seed in SEEDS and attempt in range(1,5),'OPR attempt/seed differs')
    return int(raw_hash(f'opr-20261102|{seed}|{prompt_id}|{attempt}'.encode())[:16],16)%(2**31)


def finqa_accept(output,gold):
    from kit.beds import finqa
    text=output['response']
    try:text.encode('utf-8',errors='strict')
    except UnicodeError:return False
    return (output.get('finish_reason') in ('stop','eos') and 0<output.get('output_tokens',0)<=2048
            and bool(text.strip()) and finqa.is_correct(finqa.extract_answer(text),gold))


def refresh(schedule,outputs,seed):
    """Outputs are indexed by ID/attempt. Missing attempts are missing evidence."""
    ids=list(dict.fromkeys(str(r['id']) for r in schedule));ready(ids,'empty replay schedule')
    by_id={str(r['id']):r for r in schedule};accepted={};attempts=[]
    need(set(outputs)==set(ids),'OPR generated prompt membership differs')
    for r in schedule:
        original=by_id[str(r['id'])]
        need(all(r.get(k)==original.get(k) for k in ('prompt','gold','response','target_sha256')),'repeated OPR ID has different prompt/target')
    for ident in ids:
        row=by_id[ident]
        need(row.get('stored_verified') and raw_hash(row['response'].encode())==row.get('target_sha256'),'stored target provenance differs')
        for attempt in range(1,5):
            ready(ident in outputs and str(attempt) in outputs[ident],'OPR attempts incomplete')
            out=outputs[ident][str(attempt)]
            need(out.get('rng')==generation_seed(seed,ident,attempt),'OPR sampling seed differs')
            passed=finqa_accept(out,row['gold'])
            attempts.append(dict(id=ident,attempt=attempt,accepted=passed,response_sha256=raw_hash(out['response'].encode()),
                output_tokens=out['output_tokens'],finish_reason=out['finish_reason']))
            if passed:accepted[ident]=out['response'];break
    fraction=len(accepted)/len(ids)
    ready(fraction>=.8,'OPR coverage below 80% for seed '+str(seed))
    exposures=Counter(str(r['id']) for r in schedule)
    rows=[dict(r,response=accepted.get(str(r['id']),r['response']),opr_generated=str(r['id']) in accepted) for r in schedule]
    return rows,dict(seed=seed,coverage=fraction,fallback_fraction=1-fraction,unique=len(ids),accepted=len(accepted),
        exposure_substitution=sum(exposures[i] for i in accepted)/len(schedule),
        exposure_fallback=sum(exposures[i] for i in ids if i not in accepted)/len(schedule),
        generated_identical_to_stored=[i for i in accepted if accepted[i]==by_id[i]['response']],attempts=attempts,
        schedule_ids=digest([str(r['id']) for r in rows]))


def cohort(base,incoming,expected=1147):
    ready(len(base)==len(incoming)==expected and set(base)==set(incoming),'FinQA reference vector incomplete')
    need(all(type(v) is int and v in (0,1) for vv in (base,incoming) for v in vv.values()),'nonbinary correctness')
    members=sorted(i for i in base if not base[i] and incoming[i])
    ready(len(members)>=100,'acquired-answer cohort below 100')
    return members


def final_cohort(pre_base,pre_incoming,base,incoming):
    before=cohort(pre_base,pre_incoming);current=cohort(base,incoming);diagnostics={}
    for name,a,b in [('base',pre_base,base),('incoming',pre_incoming,incoming)]:
        need(set(a)==set(b),'FinQA reference item identity differs')
        disagreements=[dict(id=i,pre=a[i],final=b[i]) for i in sorted(a) if a[i]!=b[i]]
        need(len(disagreements)<=11,'FinQA reference agreement below 99%: '+name)
        diagnostics[name]=dict(agreement=1-len(disagreements)/1147,disagreements=disagreements)
    diagnostics.update(before=len(before),final=len(current),intersection=len(set(before)&set(current)),
        symmetric_difference=sorted(set(before)^set(current)),membership_sha256=digest(current))
    return current,diagnostics


def eligible_case(item,case):
    from kit.a4_memory import eligible
    a,b=item['sec'],case['sec']
    need(a.get('issuer_group') is not None and b.get('issuer_group') is not None,'issuer-component binding absent')
    return item['id']!=case['id'] and a['issuer_group']!=b['issuer_group'] and eligible(item,case)


def v2b_context(item,ledger,tokenizer,*,boundary=False):
    """Reuse the frozen 4A ranker; only teacher context changes, no target answer."""
    from kit.a4_common import sec,finqa
    from kit.sdft_objective import teacher_messages
    need(item['month'] in ('2026-07','2026-08','2026-09'),'reserved/non-development month')
    allowed=[r for r in ledger if r['month'] in ('2026-07','2026-08') and
             (boundary or r['month']<item['month']) and r['id']!=item['id'] and eligible_case(item,r['item'])]
    cases=sorted(allowed,key=lambda r:(-sec.similarity(item,r['item']),r['id']))[:3];retrieved=len(cases)
    prompt=finqa.render_prompt(item)
    def width(messages):return len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False))
    need(width([dict(role='user',content=prompt)])<=10240,'current filing exceeds cap; no truncation')
    # Context replaces the demonstration; query target is never read here.
    while True:
        context='\n\n'.join(sec.v4_case_text(r['item'],r['response'],finqa) for r in cases)
        messages=teacher_messages([dict(role='user',content=prompt)],context)
        if width(messages)<=14336:break
        ready(cases,'teacher context cannot fit');cases.pop()
    return dict(case_ids=[r['id'] for r in cases],context=context,teacher_messages=messages,
                teacher_prompt_tokens=width(messages),boundary=boundary,effective_k=len(cases),cases_dropped=retrieved-len(cases))


def v2b_schedule(ledger,seed):
    need(all(r['month'] in ('2026-07','2026-08') and not r['item']['sec']['issuer_holdout'] for r in ledger),'v2b ledger chronology/holdout violation')
    ready(ledger,'zero development ledger support')
    ordered=sorted(ledger,key=lambda r:str(r['id']));rng=random.Random(seed);rows=[]
    while len(rows)<1280:
        cycle=list(ordered);rng.shuffle(cycle);rows.extend(cycle)
    rows=rows[:1280]
    return rows,dict(unique=len(ledger),exposures=1280,repetitions=dict(Counter(str(r['id']) for r in rows)),
                    projected_unique=41,projected_passes=31.2)
