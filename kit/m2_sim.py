"""CPU fixtures for message 2. Every artifact is explicitly technical_synthetic."""
from kit.m2_common import *
from types import SimpleNamespace as NS


def responses(items, arm, step, panel):
    """Known acquisition and replay effects; no model inference."""
    raw=[]
    for i,item in enumerate(items):
        p=item.get('panel',panel)
        if arm=='base': fraction={'finqa':.5,'test':.42,'math_hard':.5}.get(p,.6)
        elif arm.startswith('finqa-'): fraction={'finqa':.8,'test':.42,'math_hard':.5}.get(p,.6)
        elif arm.startswith('standalone-'): fraction=.82
        elif arm.startswith('legal-'):
            fraction={'finqa':.8 if step==0 else .7,'test':.42+.4*step/40,'math_hard':.5}.get(p,.6)
            if step>0 and p=='finqa' and arm in ('legal-earlier','legal-both'): fraction=.8
            if step>0 and p=='math' and arm in ('legal-broad','legal-both'): fraction=.7
        else: fraction={'finqa':.8,'test':.82,'math_hard':.5+.2*step/40}.get(p,.6)
        ok=i < round(len(items)*fraction)
        # General items are interleaved across panels; index within panel is needed.
        if panel=='general-v1':
            peers=[r for r in items if r['panel']==p]; j=next(j for j,r in enumerate(peers) if r['id']==item['id'])
            ok=j<round(len(peers)*fraction)
        gold=item.get('gold','0')
        text='A complete worked result.\nAnswer: '+(gold if ok else 'invalid') if panel=='test' else 'Working yields \\boxed{'+gold+'}.'
        raw.append(dict(id=item['id'],panel=p,response=text,output_tokens=10,finish_reason='stop',token_ids=list(range(10)),standin_correct=int(ok)))
    return raw


def fixtures(root, nomination='both', hard='QUALIFIES'):
    from kit import c_bind, c_common, c_score, math_hard_teacher, next_hellaswag
    from kit.c_sim import b_rows
    from kit.p3b_read import analyse as b_analyse
    from kit.p3b_score import write_csv
    from kit.m2_score import items_for
    root.mkdir(parents=True,exist_ok=True)
    rows=b_rows('D'); write_csv(root/'b.csv',rows)
    atomic(root/'b-reading.json',dict(b_analyse(rows,20),technical_synthetic=True))
    starts={'base':dict(model='synthetic-base')}
    for family in ('D','F','D+replay'):
        for seed in c_common.SEEDS: starts[f'{family}:{seed}']=dict(model=f'synthetic-{family}-{seed}')
    atomic(root/'starts.json',starts)
    croot=root/'C'
    c_bind.job(NS(out=croot,standin=True,b_reading=root/'b-reading.json',b_source=root/'b.csv',checkpoints=root/'starts.json'))
    cdoc=c_common.binding(croot,True)
    scores=[]
    def append(arm,seed,k,p,config=None):
        items=items_for(p,True) if p!='hellaswag' else [dict(id=str(i),prompt='External transfer fixture',gold='A') for i in range(100)]
        name='base' if arm=='base' else 'finqa-plain' if k==0 else 'legal-'+(config or 'plain')
        raw=responses(items,name,k,p)
        if p=='test': added=c_score.legal_scores(items,raw,arm,seed,k,p)
        else:
            added=[]
            for item,r in zip(items,raw):
                panel=item.get('panel',p);correct=r['standin_correct']
                if config:
                    # Nomination benefits are fixed before all continuation results.
                    enabled_b=nomination in ('broad','both','pending') and config in ('broad','both')
                    enabled_e=nomination in ('earlier','both','pending') and config in ('earlier','both')
                    if k>0 and panel=='math' and config in ('broad','both') and not enabled_b:
                        peers=[x for x in items if x.get('panel',p)==panel];idx=peers.index(item);correct=int(idx<round(.6*len(peers)))
                    if k>0 and panel=='finqa' and config in ('earlier','both') and not enabled_e: correct=int(int(item['id'])<round(.7*len(items)))
                added.append(dict(arm=arm,seed=seed,step=k,task=p,panel=panel,cap=2048,question_id=item['id'],correct=correct,output_tokens=10,finished=1,incorrect_format=0))
        scores.extend(added)
    for p in ('test','panel','finqa','general-v1'):
        if p=='panel':
            items=legal_panels()['panel'];raw=responses(items,'base',0,'test');scores+=c_score.legal_scores(items,raw,'base',101,0,'panel')
        else:append('base',101,0,p)
    for recipe in c_common.arms(cdoc):
        for route in ('standalone','sequence'):
            arm=f'{route}-{recipe}'
            for seed in SEEDS:
                for k in STEPS:
                    for p in ('test','panel'):
                        if p=='panel':
                            items=legal_panels()['panel'];scores+=c_score.legal_scores(items,responses(items,'legal-plain',k,'test'),arm,seed,k,p)
                        else:append(arm,seed,k,p)
                    if k in RETENTION_STEPS:
                        for p in ('finqa','general-v1'):append(arm,seed,k,p)
                if recipe=='D':
                    run=croot/'runs'/f'{arm}-s{seed}'
                    incoming=c_common.checked_model(c_common.start(cdoc,recipe,route,seed),True,base=route=='standalone')
                    atomic(run/'summary.json',dict(incoming=incoming,technical_synthetic=True))
                    for k in STEPS[1:]: atomic(run/f'export-{k}.json',dict(technical_synthetic=True,step=k))
                    atomic(run/'manifest.json',files_manifest(run,['summary.json']+[f'export-{k}.json' for k in STEPS[1:]],technical_synthetic=True))
                    atomic(run/'complete.json',dict(complete=True))
    for config in CONFIGS:
        arm=c_common.probe_arm(cdoc,config)
        for seed in SEEDS:
            for k in STEPS:
                append(arm,seed,k,'test',config)
                if k in RETENTION_STEPS:
                    for p in ('finqa','general-v1','hellaswag'):append(arm,seed,k,p,config)
    from kit.c_score import COLUMNS
    write_csv(croot/'scores.csv',[{k:r.get(k,'') for k in COLUMNS} for r in scores],COLUMNS)
    targets=root/'legal-targets';targets.mkdir();pool=[dict(id=f'legal-train-{i}',prompt=[dict(role='user',content=f'A training agreement invented solely as fixture {i}.')],response='Full legal reasoning supports the conclusion.\nAnswer: entailment',gold='entailment') for i in range(8)]
    t.write_jsonl(targets/'pool.jsonl',pool)
    for seed in SEEDS:t.write_jsonl(targets/f'schedule-{seed}.jsonl',cycle(pool,seed,1280))
    atomic(targets/'manifest.json',files_manifest(targets,['pool.jsonl']+[f'schedule-{seed}.jsonl' for seed in SEEDS],technical_synthetic=True))
    replay=root/'replay';replay.mkdir();t.write_jsonl(replay/'replay.jsonl',[dict(id=f'generic-{i}',prompt=f'Recite an unrelated generic phrase {i}.',response='Checked frozen base answer.') for i in range(8)])
    atomic(replay/'manifest.json',files_manifest(replay,['replay.jsonl'],technical_synthetic=True))
    pools=root/'pools';pools.mkdir()
    for name in ('earlier','gsm8k','arc'):
        t.write_jsonl(pools/(name+'.jsonl'),[dict(id=f'{name}-{i}',prompt=f'Unrelated fixture {name} task, arbitrary identifier {i}.',response='Previously verified training response.') for i in range(8)])
    atomic(pools/'manifest.json',files_manifest(pools,[n+'.jsonl' for n in ('earlier','gsm8k','arc')],technical_synthetic=True,pool_protocol=__import__('kit.next_pools',fromlist=['POOL_PROTOCOL']).POOL_PROTOCOL))
    hs=root/'hs-source.jsonl'
    t.write_jsonl(hs,[dict(ind=i,source_id='activitynet~'+str(i),ctx=f'Unseen transfer situation {i}.',endings=['one','two','three','four'],label='0') for i in range(120)])
    next_hellaswag.prepare(NS(out=root,standin=True,rows=hs))
    scenario='qualifies' if hard!='DOES_NOT_QUALIFY' else 'thin-gap'
    math_hard_teacher.job(NS(out=root,standin=True,scenario=scenario,model=None,tokenizer=None,gpu='0,1'))
    if hard=='INCOMPLETE': (root/'math-hard-teacher/manifest.json').unlink()
    return NS(b_source=root/'b.csv',b_reading=root/'b-reading.json',c_source=croot,checkpoints=root/'starts.json',teacher_source=root/'math-hard-teacher'), NS(c_targets=targets,replay=replay,pool_root=pools,panel_root=None,hellaswag=root/'hellaswag')
