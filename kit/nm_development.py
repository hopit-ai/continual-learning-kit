"""v2b July–September chronology, common collector and one August boundary."""
from pathlib import Path
from kit.nm_common import *
from kit import nm_data

MONTHS={'july':'2026-07','august':'2026-08','september':'2026-09'}
COUNTS={'july':61,'august':80,'september':52}
HOLDOUTS={'july':21,'august':24,'september':13}


def operation(row,attempt,out,bundle):
    from kit.nm_pipeline import run_lane,row_result
    from kit.nm_score import scorer_authority,file_stats
    from kit.a4_memory import build
    from kit.a4_ledger import admission
    from transformers import AutoTokenizer
    phase=row['phase'];month=row.get('month');published=[];root=out/'v2b'
    tok=AutoTokenizer.from_pretrained(bundle['tokenizer'],local_files_only=True)
    if phase=='contexts':
        items=bundle['development_batches'][month]
        need(len(items)==COUNTS[month] and len({i['id'] for i in items})==COUNTS[month],'v2b development batch denominator differs')
        need(sum(i['sec']['issuer_holdout'] for i in items)==HOLDOUTS[month],'v2b issuer holdout denominator differs')
        need(all(i['month']==MONTHS[month] for i in items),'development month identity differs')
        prior=[]
        if month!='july':prior=load(root/('ledger-july.json' if month=='august' else 'ledger-august.json'))['traces']
        contexts={}
        for item in items:
            off=build(item,[],tok);on=build(item,[r for r in prior if nm_data.eligible_case(item,r['item'])],tok)
            contexts[item['id']]=dict(off=off['prompt'],on=on['prompt'],case_ids=on['case_ids'],effective_k=len(on['case_ids']),cases_dropped=on['cases_dropped'],month=MONTHS[month])
        path=root/f'contexts-{month}.json';seal(path,contexts);published.append(str(path.relative_to(out)))
        if month!='september':
            plan_dir=root/('arrival-'+month);plan_dir.mkdir(parents=True,exist_ok=True);commands=[];cells=[]
            for index,mode in enumerate(('off','on')):
                panel=month+'-'+mode;model='base' if mode=='off' else 'memory';src=dict(bundle['base'])
                if 'files' not in src:
                    from kit.nm_prepare import verify_model
                    src=verify_model(src)
                plan=dict(seed=101,panel=panel,model=model,step=0,source=src,tokenizer=bundle['tokenizer'],metadata=file_stats(src),authority=scorer_authority(panel),
                    items=[dict(id=i['id'],prompt=contexts[i['id']][mode],item=i) for i in items])
                from kit.nm_prepare import render_scoring
                plan['rendered']=render_scoring(plan['items'],bundle['tokenizer'],panel)
                target=plan_dir/(str(index)+'.json');seal(target,plan);output=str(index)+'.score.json';cells.append(dict(plan=str(target),plan_sha256=sha(target),output=output))
                commands.append(dict(argv=[bundle.get('inference_python','python3'),'-I',str(KIT/'nm_score.py'),'--plan',str(target),'--out','{attempt}/'+output],
                    requires=[str(Path(src['model'])/'config.json')],engine=dict(backend='vllm',model=src['model'],max_model_len=12288)))
            seal(plan_dir/'lane.json',dict(commands=commands,cells=cells))
    elif phase=='arrival':
        run_lane(root/('arrival-'+month),attempt)
        published.extend(str(p.relative_to(out)) for p in attempt.glob('*.score.json'))
    elif phase=='ledger':
        arrival=row_result(out,'NM-v2b-'+month+'-arrival');off=load(arrival/'0.score.json');on=load(arrival/'1.score.json')
        items=bundle['development_batches'][month];raw={r['id']:r for r in off['raw']}
        need(set(raw)=={i['id'] for i in items} and set(on['cell']['items'])==set(raw),'whole-month predictions incomplete')
        # Seal all no-update predictions before feedback release. C/T before the
        # sole boundary are exactly the same untrained base on the same prompts.
        seal(root/f'seal-{month}.json',dict(off=sha(arrival/'0.score.json'),on=sha(arrival/'1.score.json'),
            aliases=['C-off','T-off','C-on','T-on'],reason='identical original-base weights before sole August boundary',
            contexts=sha(root/f'contexts-{month}.json')))
        previous=[] if month=='july' else load(root/'ledger-july.json')['traces'];traces=[];decisions=[]
        for item in items:
            r=raw[item['id']];reasons=admission(item,r,tok)
            decisions.append(dict(id=item['id'],admitted=not reasons,reasons=reasons,response_sha256=raw_hash(r['response'].encode())))
            if not reasons:traces.append(dict(id=item['id'],month=MONTHS[month],item=item,response=r['response'],
                collector=bundle['base']['sha256'],raw_sha256=digest(r),source_sha256=digest(item),verifier=scorer_authority(month+'-off'),unedited=True))
        value=dict(traces=[*previous,*traces],decisions=decisions,sealed=sha(root/f'seal-{month}.json'))
        target=root/f'ledger-{month}.json';seal(target,value);published.extend(str(p.relative_to(out)) for p in (target,root/f'seal-{month}.json'))
        if month=='august':seal(root/'ledger.json',value['traces']);published.append(str((root/'ledger.json').relative_to(out)))
    elif phase=='schedule':
        ledger=load(root/'ledger.json');audits={}
        for seed in SEEDS:
            rows,audit=nm_data.v2b_schedule(ledger,seed);audits[str(seed)]=audit
        ready(ledger,'zero common ledger support')
        seal(root/'supply.json',dict(seeds=audits,actual_unique=len(ledger),projected_unique=41,projected_passes=31.2))
        published.append(str((root/'supply.json').relative_to(out)))
    else:raise Invalid('unregistered development phase')
    return dict(status='COMPLETE',published=published)
