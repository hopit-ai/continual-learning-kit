#!/usr/bin/env python3
"""Next-message tables: SEC coexistence, interpolation curves, factorial replay nominations."""
from pathlib import Path
import sys
if __package__ in (None, ''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.next_common import *
from kit.m2_seed_policy import SEEDS
from kit.p3b_read import load,paired,curve,accuracy,question_bootstrap
from collections import defaultdict
import csv
import math
import statistics


def interval(values,n=3):
    if len(values)!=n or any(v is None for v in values):return dict(n=len(values),mean=None,lower=None,upper=None)
    avg=statistics.mean(values)
    critical={5:2.7764451052,3:4.3026527297}.get(n)
    radius=critical*statistics.stdev(values)/math.sqrt(n) if critical else 0
    return dict(n=n,mean=avg,lower=avg-radius,upper=avg+radius)


def protects(row,margin=-1):return row['lower'] is not None and row['lower']>=margin


def benefit(row):return row['mean'] is not None and row['mean']>=2 and row['lower']>0


def issuer_interval(pairs,repeats):
    # Average seeds per question first; resample connected issuer components, never individual questions.
    import numpy as np
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for issuer,_,_ in pairs:
        ids=issuer.split('|')
        for x in ids[1:]:parent[find(x)]=find(ids[0])
    groups=defaultdict(list)
    for issuer,q,v in pairs:groups[(find(issuer.split('|')[0]),q)].append(v)
    clusters=defaultdict(list)
    for (issuer,q),values in groups.items():clusters[issuer].append(statistics.mean(values))
    if not clusters:return None
    sums=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()])
    draws=np.random.default_rng(0).multinomial(len(sums),[1/len(sums)]*len(sums),size=repeats)
    values=np.sort((draws@sums)/(draws@counts))
    return dict(n_clusters=len(sums),lower=float(np.quantile(values,.025)),upper=float(np.quantile(values,.975)))


def analyse(rows,repeats=10000,acquisition='math_screen'):
    groups=defaultdict(dict);summaries=[]
    for r in rows:
        require(int(r['cap'])==2048,'mixed answer budgets')
        key=(r['arm'],int(r['seed']),int(r['step']),r['panel'])
        require(r['question_id'] not in groups[key],'duplicate score row')
        require(int(r['correct']) in (0,1),'malformed correctness')
        groups[key][r['question_id']]=r
    # Pooled unseen-issuer tables, with the exact months made explicit in their panel labels.
    pooled=defaultdict(dict)
    for (arm,seed,step,panel),g in list(groups.items()):
        if panel.startswith('sec-2026-'):
            pooled[(arm,seed,step,'sec-pooled-observed-months')].update(g)
            if panel in ('sec-2026-08','sec-2026-09'):pooled[(arm,seed,step,'sec-pooled-Aug-Sep')].update(g)
    groups.update(pooled)
    for (arm,seed,step,panel),g in sorted(groups.items()):
        c=sum(int(r['correct']) for r in g.values());tokens=sum(int(r['output_tokens']) for r in g.values())
        summaries.append(dict(arm=arm,seed=seed,step=step,panel=panel,n=len(g),correct=c,accuracy=accuracy(g),
            finished=sum(int(r['finished']) for r in g.values()),finish_rate=sum(int(r['finished']) for r in g.values())/len(g),output_tokens=tokens,tokens_per_correct=tokens/c if c else None,
            historical_accuracy=100*sum(int(r['historical']) for r in g.values())/len(g) if all('historical' in r for r in g.values()) else None,
            all_tokens_per_correct=(tokens+sum(int(r['prompt_tokens']) for r in g.values()))/c if c and all(r.get('prompt_tokens') not in (None,'') for r in g.values()) else None))
        if panel=='contract_nli':
            from kit.c_read import question_interval
            summaries[-1]['question_interval']=question_interval({q:100*int(r['correct']) for q,r in g.items()},g,repeats)
    sec=[];effects=[]
    for months,mode in [(('2026-07',),'off'),(('2026-08',),'memory'),(('2026-09',),'memory'),(('2026-08','2026-09'),'memory')]:
        changes=[];details=[];clusters=[]
        for seed in SEEDS:
            left={};right={}
            for m in months:
                left.update(groups.get(('v0-base-'+mode,0,0,'sec-'+m),{}));right.update(groups.get(('v0-trained-'+mode,seed,0,'sec-'+m),{}))
            if not left or not right:continue
            p=paired(left,right,repeats);changes.append(p['change']);details.append(dict(seed=seed,**p))
            clusters += [(right[q]['issuer'],q,100*(int(right[q]['correct'])-int(left[q]['correct']))) for q in left]
        stat=interval(changes);ci=issuer_interval(clusters,repeats)
        met=stat['mean'] is not None and stat['mean']>=0
        sec.append(dict(months=months,mode=mode,seeds=details,seed_interval=stat,issuer_clustered_95=ci,
            met=met,confirmed=bool(met and protects(stat,-2) and ci and ci['lower']>=-2),
            policy='Mean >= reference; confirmation NI lower >= -2 points. Development demonstration only.'))
    for who,seeds in [('base',(0,)),('trained',SEEDS)]:
        for seed in seeds:
            for month in ('2026-08','2026-09'):
                off=groups.get(('v0-'+who+'-off',seed,0,'sec-'+month));mem=groups.get(('v0-'+who+'-memory',seed,0,'sec-'+month))
                if off and mem:effects.append(dict(arm=who,seed=seed,month=month,**paired(off,mem,repeats)))
    screen=[]
    for route in ('finqa','standalone','sequence'):
        for recipe in RECIPES:
            for lam in (0,.25,.5,.75,1):
                arm=f'v3-{route}-{recipe}-l{lam:g}'
                for panel in ('finqa','ifeval','knowledge','math')+(('math_screen',) if route!='finqa' else ()):
                    pairs=[]
                    for seed in SEEDS:
                        final=groups.get((f'v3-{route}-{recipe}-l1',seed,40,panel));blend=groups.get((arm,seed,40,panel))
                        if final and blend:pairs.append(dict(seed=seed,score=accuracy(blend),**paired(final,blend,repeats)))
                    screen.append(dict(route=route,recipe=recipe,lam=lam,panel=panel,seeds=pairs,
                        score_interval=interval([p['score'] for p in pairs]),vs_full=interval([p['change'] for p in pairs])))
    screen_rules=[]
    for route in ('finqa','standalone','sequence'):
        for recipe in RECIPES:
            for lam in (.25,.5,.75):
                points={r['panel']:r['vs_full'] for r in screen if r['route']==route and r['recipe']==recipe and r['lam']==lam}
                screen_acquisition=['finqa'] if route=='finqa' else ['math_screen']
                retention=['ifeval','knowledge','math']+(['finqa'] if route!='finqa' else [])
                screen_rules.append(dict(route=route,recipe=recipe,lam=lam,acquisition=screen_acquisition,retention=retention,
                    exploratory_signal=any(benefit(points[p]) and all(protects(points[q]) for q in points if q!=p) for p in retention) and all(protects(points[p]) for p in screen_acquisition),
                    proper_lambda=.5,proper_pending_v2=True,incumbent_changed=False))
    def probe_pair(x,y,panel):
        if panel == 'contract_nli':
            from kit.c_read import paired as legal_paired
            return legal_paired(x,y,'correct',repeats,cluster=True)
        return paired(x,y,repeats)
    curves=[];survivals=[];transitions=[];transfer=[];comparisons=[]
    for arm in ('generic','broad','earlier','both'):
        for seed in SEEDS:
            c=curve(groups,'v4-'+arm,seed,acquisition)
            if c:
                if acquisition=='contract_nli':
                    from kit.c_read import curve_question_intervals
                    c['question_intervals']=curve_question_intervals([groups[('v4-'+arm,seed,k,acquisition)] for k in (0,10,20,30,40)],'correct',repeats)
                curves.append(dict(arm=arm,seed=seed,**c))
            base=groups.get(('base',seed,0,'finqa')) or groups.get(('base',0,0,'finqa'));start=groups.get(('v4-'+arm,seed,0,'finqa'));end=groups.get(('v4-'+arm,seed,40,'finqa'))
            if base and start and end:
                require(set(base)==set(start)==set(end),'FinQA coverage differs')
                ids=[q for q in start if not int(base[q]['correct']) and int(start[q]['correct'])]
                n=len(ids);kept=sum(int(end[q]['correct']) for q in ids)
                survivals.append(dict(arm=arm,seed=seed,n=n,acquired_ids=ids,retained=kept,survival=100*kept/n if n else None,
                    lost_ids=[q for q in ids if not int(end[q]['correct'])]))
            for panel in ('finqa','ifeval','knowledge','math','hellaswag',acquisition):
                start=groups.get(('v4-'+arm,seed,0,panel));end=groups.get(('v4-'+arm,seed,40,panel))
                if start and end:
                    value=dict(arm=arm,seed=seed,panel=panel,**probe_pair(start,end,panel));transitions.append(value)
                    if panel=='hellaswag':transfer.append(value)
    def contrast(candidate,control):
        endpoints={};pairs=[]
        for panel in ('finqa','ifeval','knowledge','math','hellaswag',acquisition):
            ps=[]
            for seed in SEEDS:
                x=groups.get(('v4-'+control,seed,40,panel));y=groups.get(('v4-'+candidate,seed,40,panel))
                if x and y:ps.append(dict(seed=seed,panel=panel,**probe_pair(x,y,panel)))
            pairs+=ps;endpoints[panel]=interval([p['change'] for p in ps])
        auc=[];gains=[];survival=[];survival_pairs=[]
        for seed in SEEDS:
            x=next((r for r in curves if r['arm']==control and r['seed']==seed),None)
            y=next((r for r in curves if r['arm']==candidate and r['seed']==seed),None)
            if x and y:
                auc.append(y['gain_auc']-x['gain_auc'])
                gains.append(y['gain']-x['gain'])
            sx=next((r for r in survivals if r['arm']==control and r['seed']==seed),None)
            sy=next((r for r in survivals if r['arm']==candidate and r['seed']==seed),None)
            if sx and sy:
                require(sx['acquired_ids']==sy['acquired_ids'],'unmatched acquired FinQA cohorts')
                survival.append(sy['survival']-sx['survival'] if sx['n'] else None)
                ids=sx['acquired_ids']
                if ids:
                    x=groups[('v4-'+control,seed,40,'finqa')];y=groups[('v4-'+candidate,seed,40,'finqa')]
                    survival_pairs.append(dict(seed=seed,**paired({q:x[q] for q in ids},{q:y[q] for q in ids},repeats)))
                else:survival_pairs.append(dict(seed=seed,n=0,change=None,gained=[],lost=[],question_bootstrap_95=None))
        endpoints['gain']=interval(gains);endpoints['gain_auc']=interval(auc);endpoints['survival']=interval(survival)
        auc_questions=[];gain_questions=[]
        for seed in SEEDS:
            left=groups.get(('v4-'+control,seed,0,acquisition));right=groups.get(('v4-'+candidate,seed,0,acquisition))
            if not left or not right or not all(('v4-'+arm,seed,step,acquisition) in groups for arm in (control,candidate) for step in (0,10,20,30,40)):continue
            require(set(left)==set(right),'unmatched acquisition questions')
            def gain(arm,q):
                points=[int(groups[('v4-'+arm,seed,k,acquisition)][q]['correct']) for k in (0,10,20,30,40)]
                return sum((x+y)*12.5 for x,y in zip(points,points[1:]))-100*points[0]
            values={q:gain(candidate,q)-gain(control,q) for q in sorted(left)}
            if acquisition=='contract_nli':
                from kit.c_read import question_interval
                require(all(left[q].get('group')==right[q].get('group') for q in left),'legal document binding changed')
                auc_questions.append(dict(seed=seed,**question_interval(values,left,repeats)))
                def endpoint_gain(arm,q):
                    return 100*(int(groups[('v4-'+arm,seed,40,acquisition)][q]['correct'])-int(groups[('v4-'+arm,seed,0,acquisition)][q]['correct']))
                gain_questions.append(dict(seed=seed,**question_interval({q:endpoint_gain(candidate,q)-endpoint_gain(control,q) for q in left},left,repeats)))
            else:
                auc_questions.append(dict(seed=seed,question_bootstrap_95=question_bootstrap(list(values.values()),repeats)))
        result = dict(candidate=candidate,control=control,endpoints=endpoints,question_pairs=pairs,auc_question_pairs=auc_questions,gain_question_pairs=gain_questions,survival_question_pairs=survival_pairs)
        from kit.ref_guard import apply
        return apply(result, [r for r in rows if r['arm'] in ('v4-'+candidate, 'v4-'+control)])
    for candidate,control in [('broad','generic'),('earlier','generic'),('both','broad'),('both','generic')]:comparisons.append(contrast(candidate,control))
    broad=comparisons[0];earlier=comparisons[2] if benefit(broad['endpoints']['math']) and all(protects(broad['endpoints'][p]) for p in ('survival','hellaswag',('gain' if acquisition == 'contract_nli' else 'gain_auc'),acquisition)) else comparisons[1]
    def nominated(c,primary,other):
        return c.get('status', 'COMPLETE') == 'COMPLETE' and benefit(c['endpoints'][primary]) and all(protects(c['endpoints'][p]) for p in (other,'hellaswag',('gain' if acquisition == 'contract_nli' else 'gain_auc'),acquisition))
    broad_pass=nominated(broad,'math','survival')
    earlier_pass=nominated(earlier,'survival','math')
    if earlier['candidate']=='both':earlier_pass=earlier_pass and all(protects(comparisons[3]['endpoints'][p]) for p in (('gain' if acquisition == 'contract_nli' else 'gain_auc'),acquisition))
    return dict(technical_synthetic=False,summaries=summaries,v0=sec,v0_own_memory_effect=effects,
        v0_coexistence_confirmed=all(r['confirmed'] for r in sec if len(r['months'])==1),
        v3_curves=screen,v3_rules=screen_rules,v3_proper_lambda=.5,v3_admission='Pending v2; no lambda selection or incumbent change',
        v4_curve_intervals=[dict(arm=arm,points=[interval([r['points'][i] for r in curves if r['arm']==arm]) for i in range(5)],gain_auc=interval([r['gain_auc'] for r in curves if r['arm']==arm])) for arm in ('generic','broad','earlier','both')],
        v4_survival_intervals=[dict(arm=arm,seed_interval=interval([r['survival'] for r in survivals if r['arm']==arm])) for arm in ('generic','broad','earlier','both')],
        hellaswag_transfer_intervals=[dict(arm=arm,seed_interval=interval([r['change'] for r in transfer if r['arm']==arm])) for arm in ('generic','broad','earlier','both')],
        v4_curves=curves,v4_survival=survivals,v4_transitions=transitions,hellaswag_transfer=transfer,v4_comparisons=comparisons,
        v4_nomination=dict(broad=broad_pass,earlier=earlier_pass,earlier_comparison=[earlier['candidate'],earlier['control']],admission='Pending connected finals'),
        disclosure='Broad matches general-v1 maths and is near knowledge; HellaSwag alone is out-of-pool. ARC shares its answer format. Both exposes 800 acquisition-task prompts versus 960 for single additions and 1120 for generic.')


def job(a):
    frozen_next();rows,files=load(a.source)
    import gzip
    for name,raw in list(files.items()):
        if name.endswith('/scores.csv.gz'):files[name[:-3]]=gzip.decompress(raw)
    for name,raw in files.items():
        if not name.endswith('/manifest.json'):continue
        doc=json.loads(raw);prefix=str(Path(name).parent)+'/'
        for file,digest in doc.get('files',{}).items():
            if prefix+file in files and isinstance(digest,str):
                require(t.sha(files[prefix+file])==digest,'returned frozen evidence differs: '+prefix+file)
    # B load() covers scores/; SEC per-seed journals are separate.
    loaded_sec=any(r.get('task')=='sec' for r in rows)
    for name,raw in files.items():
        if not loaded_sec and name.startswith('sec/') and name.endswith('/scores.csv'):
            import io
            rows+=list(csv.DictReader(io.StringIO(raw.decode())))
    # Shared endpoint generations are evidence once, relabelled for every registered comparison.
    by_source=defaultdict(list)
    for r in rows:
        if r['arm'].startswith('next-shared-'):
            # general-v1 expands into three panel labels in its single score file.
            panel_set='general-v1' if r['panel'] in ('ifeval','knowledge','math') else r['panel']
            by_source[f"scores/{r['arm']}-s0-k0-{panel_set}"].append(r)
    rows=[r for r in rows if not r['arm'].startswith('next-shared-')]
    for name,raw in sorted(files.items()):
        if name.startswith('score-aliases/') and name.endswith('.json'):
            alias=json.loads(raw)
            require(bool(by_source[alias['source']]),'missing shared score source: '+alias['source'])
            rows += [dict(r,arm=alias['arm'],seed=alias['seed'],step=alias['step']) for r in by_source[alias['source']]]
    if not rows:raise NotReady('no next-message scores')
    rows.sort(key=lambda r:(r['arm'],int(r['seed']),int(r['step']),r['panel'],r['question_id']))
    result=analyse(rows,a.bootstrap)
    result['technical_synthetic']=a.standin or any(json.loads(b).get('technical_synthetic',False) for n,b in files.items() if n.endswith(('identity.json','summary.json')))
    result['trace_yield']={n:json.loads(b)['trace_supply'] for n,b in files.items() if n.endswith('explicit-metrics.json')}
    result['source_sha256']=s.digest(rows)
    inputs=json.loads(files['next-inputs/inputs.json']) if 'next-inputs/inputs.json' in files else {}
    result['B_general_margins_reported_not_rejudged']=inputs.get('w_decision',{}).get('b_general_margins',{})
    result['costs']={n:json.loads(b) for n,b in files.items() if n.endswith('cost.json') or n.startswith('runs/') and n.endswith('summary.json')}
    result['costs_note']='Measured process GPU-hours; allocation idle, omitted/failed attempts and site timings require the partner return note.'
    atomic(a.out/'reading-next.json',result)
    for name in ('summaries','v0','v3_curves','v3_rules','v4_curves','v4_curve_intervals','v4_survival','v4_survival_intervals','v4_comparisons','hellaswag_transfer'):
        values=result[name]
        text='# '+name+'\n\n| Record | Registered readout |\n|---|---|\n'+''.join('| '+str(i+1)+' | '+json.dumps(v,sort_keys=True).replace('|',' / ')+' |\n' for i,v in enumerate(values))
        (a.out/(name+'.md')).write_text(text)
    print('Read',len(rows),'question rows;',len(result['v4_curves']),'v4 curves')


def configure(p):
    p.add_argument('--source',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=10000)
if __name__ == '__main__':raise SystemExit(entry(job,configure,'next-read'))
