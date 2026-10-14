#!/usr/bin/env python3
"""Our-side reader: registered bars, curves, reset trigger, family churn and costs."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_common import *
import csv
import gzip
import io
import math
import statistics
import tarfile
from collections import defaultdict


def bound(values, expected=None):
    if not values or (expected is not None and len(values)!=expected): return dict(n=len(values),mean=None,lower=None,upper=None)
    avg=statistics.mean(values)
    # Registered analyses use three or five independent seeds.
    critical={3:2.9199855804,5:2.1318467863}.get(len(values))
    radius=critical*statistics.stdev(values)/math.sqrt(len(values)) if critical else None
    return dict(n=len(values),mean=avg,lower=avg-radius if radius is not None else None,upper=avg+radius if radius is not None else None)


def lower(row, margin): return row['lower'] is not None and row['lower']>=margin


def accuracy(rows): return 100*sum(int(r['correct']) for r in rows.values())/len(rows) if rows else None


def question_bootstrap(values,repeats):
    import numpy as np
    unique,counts=np.unique(values,return_counts=True)
    draws=np.random.default_rng(0).multinomial(len(values),counts/len(values),size=repeats)
    samples=np.sort(draws @ unique / len(values))
    return [float(samples[int(.025*repeats)]),float(samples[min(repeats-1,int(.975*repeats))])]


def paired(a,b,repeats=10000):
    require(set(a)==set(b),'paired scoring question sets differ')
    ids=sorted(a); require(bool(ids),'empty scoring comparison')
    values=[100*(int(b[q]['correct'])-int(a[q]['correct'])) for q in ids]
    samples=question_bootstrap(values,repeats)
    return dict(n=len(ids),change=statistics.mean(values),gained=[q for q,v in zip(ids,values) if v>0],lost=[q for q,v in zip(ids,values) if v<0],
        question_bootstrap_95=samples)


def curve(groups,arm,seed,panel,subset=None):
    points=[]
    for step in (0,10,20,30,40):
        rows=groups.get((arm,seed,step,panel))
        if rows is None: return None
        if subset is not None: rows={q:r for q,r in rows.items() if q in subset}
        if not rows: return None
        points.append(accuracy(rows))
    raw=sum((a+b)*5 for a,b in zip(points,points[1:]))/40
    return dict(points=points,raw_auc=raw*40,normalised_raw_auc=raw,gain_auc=raw-points[0],initial=points[0],final=points[-1],gain=points[-1]-points[0])


def analyse(rows, repeats=10000):
    groups=defaultdict(dict)
    for r in rows:
        if int(r['cap'])!=2048: continue
        seed=int(str(r['seed']).removeprefix('s').replace('-','0') or 0); key=(r['arm'],seed,int(r['step']),r['panel'])
        require(r['question_id'] not in groups[key],'duplicate scoring question')
        require(int(r['correct']) in (0,1) and int(r['finished']) in (0,1) and int(r['output_tokens'])>=0,'malformed score')
        groups[key][r['question_id']]=r
    base={p:g for (arm,seed,step,p),g in groups.items() if arm=='base'}
    for (arm,seed,step,p),g in groups.items():
        if arm=='base': require({q:(r['correct'],r['output_tokens']) for q,r in g.items()}=={q:(r['correct'],r['output_tokens']) for q,r in base[p].items()},'inconsistent repeated base scores')
    summaries=[]; comparisons=[]; families=[]
    family_map={r['id']:{x.split(':')[0] for x in r.get('meta',{}).get('instruction_id_list',[])} for r in t.read_jsonl(KIT / 'panels/general-v1.jsonl') if r['panel']=='ifeval'}
    for (arm,seed,step,panel),g in sorted(groups.items()):
        correct=sum(int(r['correct']) for r in g.values()); tokens=sum(int(r['output_tokens']) for r in g.values())
        reference=base.get(panel)
        ratio=(tokens/correct)/(sum(int(r['output_tokens']) for r in reference.values())/sum(int(r['correct']) for r in reference.values())) if reference and correct and sum(int(r['correct']) for r in reference.values()) and sum(int(r['output_tokens']) for r in reference.values()) else None
        summaries.append(dict(arm=arm,seed=seed,step=step,panel=panel,n=len(g),correct=correct,accuracy=accuracy(g),finish_rate=sum(int(r['finished']) for r in g.values())/len(g),output_tokens=tokens,tokens_per_correct=tokens/correct if correct else None,serving_ratio=ratio,at_cost=ratio is not None and ratio>1.5,cost_defined=ratio is not None))
        if reference:
            comparisons.append(dict(arm=arm,seed=seed,step=step,panel=panel,versus='base',**paired(reference,g,repeats)))
        if panel=='ifeval' and reference:
            pre=groups.get((arm.split('-')[1],seed,40,panel)) if arm.startswith('recover-') else None
            for family in sorted(set().union(*family_map.values())):
                ids={q for q in g if family in family_map.get(q,set())}
                losses=[q for q in ids if int(reference[q]['correct']) and not int(g[q]['correct'])]
                recovered=[q for q in ids if pre and int(reference[q]['correct']) and not int(pre[q]['correct']) and int(g[q]['correct'])]
                families.append(dict(arm=arm,seed=seed,step=step,family=family,n=len(ids),base_correct=sum(int(reference[q]['correct']) for q in ids),correct=sum(int(g[q]['correct']) for q in ids),lost_ids=losses,recovered_ids=recovered))
    recoveries=[]
    for arm in ('D','F'):
        for size in (64,16):
            seeds=[]
            for seed in range(101,104):
                pre={p:groups.get((arm,seed,40,p)) for p in ('ifeval','finqa','knowledge','math')}
                post={p:groups.get((f'recover-{arm}-n{size}',seed,4,p)) for p in pre}
                if not all(pre.values()) or not all(post.values()) or 'ifeval' not in base: continue
                delta={p:paired(pre[p],post[p],repeats) for p in pre}
                i0=accuracy(base['ifeval']); initial_loss=i0-accuracy(pre['ifeval']); gain=delta['ifeval']['change']
                seeds.append(dict(seed=seed,changes=delta,initial_loss=initial_loss,recovery_fraction=gain/initial_loss if initial_loss>0 else None,half_contrast=gain-.5*initial_loss,residual=accuracy(post['ifeval'])-i0))
            bars={p:bound([r['changes'][p]['change'] for r in seeds],3) for p in ('ifeval','finqa','knowledge','math')}
            half=bound([r['half_contrast'] for r in seeds],3); residual=bound([r['residual'] for r in seeds],3)
            initial=bound([r['initial_loss'] for r in seeds],3)
            protection=all(lower(bars[p],-1) for p in ('finqa','knowledge','math'))
            defined=initial['mean'] is not None and initial['mean']>0
            useful=(lower(bars['ifeval'],5) and lower(half,0) and protection) if defined else None
            recoveries.append(dict(arm=arm,size=size,primary=size==64,seeds=seeds,bars=bars,initial_loss=initial,protection=protection,half_recovery=half,residual=residual,useful=useful,near_base=(useful and lower(residual,-3) and residual['upper']<=3) if defined else None,status='UNDEFINED_RECOVERY' if not defined else 'PASS' if useful else 'INCONCLUSIVE_OR_FAIL'))
    curves=[]; acquisitions=[]; deficits=[]; sequence_bars=[]
    triggers=[]
    for recipe in ('D+replay','D','F'):
        for panel in ('code','math_screen'):
            alone=[curve(groups,'standalone-'+recipe,seed,panel) for seed in range(101,106)]
            seq=[curve(groups,'sequence-'+recipe,seed,panel) for seed in range(101,106)]
            for seed,x,y in zip(range(101,106),alone,seq):
                for route,c in [('standalone',x),('sequence',y),('reset',curve(groups,'reset-'+recipe,seed,panel))]:
                    if c:
                        start=groups[(route+'-'+recipe,seed,0,panel)]
                        strata={label:curve(groups,route+'-'+recipe,seed,panel,{q for q,r in start.items() if bool(int(r['correct']))==success}) for label,success in [('initial_success',True),('initial_failure',False)]}
                        curves.append(dict(recipe=recipe,route=route,seed=seed,panel=panel,**c,strata=strata))
            gains=[c['gain'] for c in alone if c]; stats=bound(gains,5)
            bar=100*(3/51 if panel=='code' else 6/100)
            acquired=stats['mean'] is not None and stats['mean']>=bar and sum(v>0 for v in gains)>=4 and lower(stats,0) and stats['lower']>0
            acquisitions.append(dict(recipe=recipe,panel=panel,gains=gains,seed_interval=stats,bar=bar,passes=acquired))
            differences=[]; subsets=[]; ends=[]
            for seed,x,y in zip(range(101,106),alone,seq):
                if not x or not y: continue
                differences.append(y['gain_auc']-x['gain_auc'])
                gx=groups[('standalone-'+recipe,seed,0,panel)]; gy=groups[('sequence-'+recipe,seed,0,panel)]
                require(set(gx)==set(gy),'sequence initial question sets differ')
                both={q for q in gx if not int(gx[q]['correct']) and not int(gy[q]['correct'])}
                cx=curve(groups,'standalone-'+recipe,seed,panel,both); cy=curve(groups,'sequence-'+recipe,seed,panel,both)
                subsets.append(dict(seed=seed,n=len(both),deficit=cy['gain_auc']-cx['gain_auc'] if cx and cy else None))
                ends.append(paired(groups[('standalone-'+recipe,seed,40,panel)],groups[('sequence-'+recipe,seed,40,panel)],repeats))
            ds=bound(differences,5); valid=[r['deficit'] for r in subsets if r['deficit'] is not None]; ss=bound(valid,5)
            trigger=acquired and ds['mean'] is not None and ds['mean']<=-3 and sum(v<0 for v in differences)>=4 and ds['upper']<0 and ss['mean'] is not None and ss['mean']<=-3
            if trigger: triggers.append(recipe)
            deficits.append(dict(recipe=recipe,panel=panel,seed_differences=differences,seed_interval=ds,both_wrong=subsets,subset_interval=ss,complete=len(differences)==5,trigger=trigger))
            protections={}
            for p in ('finqa','ifeval','knowledge','math'):
                changes=[]
                for seed in range(101,106):
                    end=groups.get(('sequence-'+recipe,seed,40,p)); start=groups.get(('sequence-'+recipe,seed,0,p)) if p=='finqa' else base.get(p)
                    if end and start: changes.append(paired(start,end,repeats)['change'])
                protections[p]=bound(changes,5)
            final=bound([e['change'] for e in ends],5)
            stage1={p:bound([accuracy(groups[('sequence-'+recipe,seed,0,p)])-accuracy(base[p]) for seed in range(101,106) if ('sequence-'+recipe,seed,0,p) in groups and p in base],5) for p in ('finqa','ifeval','knowledge','math')}
            passed=acquired and lower(final,-2) and lower(protections['finqa'],-1) and lower(stage1['finqa'],0) and stage1['finqa']['lower']>0 and all(lower(protections[p],-3) and lower(stage1[p],-3) for p in ('ifeval','knowledge','math'))
            serving=[r for r in summaries if r['arm']=='sequence-'+recipe and r['step']==40 and r['panel'] in ('finqa',panel)]
            deployable=len(serving)==10 and all(r['serving_ratio'] is not None and r['serving_ratio']<=1.5 for r in serving)
            sequence_bars.append(dict(deployable=passed and deployable,at_cost=passed and len(serving)==10 and any(r['at_cost'] for r in serving),recipe=recipe,panel=panel,final_vs_standalone=final,final_question_pairs=ends,stage1_vs_base=stage1,stage2_retention=protections,passes=passed))
    reset_comparisons=[]
    for recipe in ('D+replay','D','F'):
        for panel in ('finqa','ifeval','knowledge','math','code','math_screen'):
            pairs=[dict(seed=seed,**paired(groups[('sequence-'+recipe,seed,40,panel)],groups[('reset-'+recipe,seed,40,panel)],repeats)) for seed in range(101,106) if ('sequence-'+recipe,seed,40,panel) in groups and ('reset-'+recipe,seed,40,panel) in groups]
            reset_comparisons.append(dict(recipe=recipe,panel=panel,seeds=pairs,seed_interval=bound([r['change'] for r in pairs],5)))
    finqa_acquisition=[]
    for arm,n in [('D',5),('F',5),('finqa-D+replay',5),('finqa-F+replay',3)]:
        values=[accuracy(groups[(arm,seed,40,'finqa')])-accuracy(base['finqa']) for seed in range(101,101+n) if (arm,seed,40,'finqa') in groups and 'finqa' in base]
        stats=bound(values,n)
        finqa_acquisition.append(dict(arm=arm,seeds=values,seed_interval=stats,passes=lower(stats,0) and stats['lower']>0))
    prevention=[]; reset_damage=[]; auc_pairs=[]
    for candidate,control,nseeds in [('finqa-D+replay','D',5),('finqa-F+replay','F',3),('finqa-F+replay','finqa-D+replay',3)]:
        for panel in ('finqa','ifeval','knowledge','math'):
            pairs=[dict(seed=seed,**paired(groups[(control,seed,40,panel)],groups[(candidate,seed,40,panel)],repeats)) for seed in range(101,101+nseeds) if (control,seed,40,panel) in groups and (candidate,seed,40,panel) in groups]
            prevention.append(dict(candidate=candidate,control=control,panel=panel,seeds=pairs,seed_interval=bound([r['change'] for r in pairs],nseeds)))
    for recipe in ('D+replay','D','F'):
        for panel in ('finqa','ifeval','knowledge','math','code','math_screen'):
            for seed in range(101,106):
                pre=groups.get(('sequence-'+recipe,seed,0,panel)); post=groups.get(('reset-'+recipe,seed,0,panel))
                if pre and post: reset_damage.append(dict(recipe=recipe,seed=seed,panel=panel,**paired(pre,post,repeats)))
        for panel in ('code','math_screen'):
            for seed in range(101,106):
                left=curve(groups,'standalone-'+recipe,seed,panel);right=curve(groups,'sequence-'+recipe,seed,panel)
                if not left or not right: continue
                ids=sorted(groups[('standalone-'+recipe,seed,0,panel)])
                def per_question(arm,q):
                    points=[int(groups[(arm,seed,k,panel)][q]['correct']) for k in (0,10,20,30,40)]
                    return sum((a+b)*12.5 for a,b in zip(points,points[1:]))-100*points[0]
                values=[per_question('sequence-'+recipe,q)-per_question('standalone-'+recipe,q) for q in ids]
                auc_pairs.append(dict(recipe=recipe,panel=panel,seed=seed,question_bootstrap_95=question_bootstrap(values,repeats)))
    acquired_retention=[]
    for (arm,seed,step,p),g in groups.items():
        if p!='finqa' or step!=40 or not arm.startswith(('sequence-','reset-')) or p not in base: continue
        start=groups.get(('sequence-'+arm.split('-',1)[1],seed,0,p))
        if start:
            ids=[q for q in start if int(start[q]['correct']) and not int(base[p][q]['correct'])]
            acquired_retention.append(dict(arm=arm,seed=seed,n=len(ids),retained=sum(int(g[q]['correct']) for q in ids),lost_ids=[q for q in ids if not int(g[q]['correct'])]))
    reset_recipe=None; reset_pending=False
    # Package A selects code or the MATH fallback; an unselected task's absent
    # curves must not block a complete comparison on the selected task.
    reset_panels={p for (arm,seed,step,p) in groups if p in ('code','math_screen')
                  and arm.startswith(('standalone-','sequence-'))}
    for recipe in ('D+replay','D','F'):
        if recipe in triggers:
            reset_recipe=recipe; break
        if not reset_panels or not all(d['complete'] for d in deficits if d['recipe']==recipe and d['panel'] in reset_panels):
            reset_pending=True; break
    return dict(reset_vs_unreset=reset_comparisons,finqa_acquisition=finqa_acquisition,prevention=prevention,reset_immediate_damage=reset_damage,auc_question_bootstrap=auc_pairs,summaries=summaries,question_pairs=comparisons,recovery=recoveries,families=families,curves=curves,acquisition=acquisitions,sequence=sequence_bars,reset_deficits=deficits,reset_recipe=reset_recipe,reset_pending=reset_pending,acquired_finqa_retention=acquired_retention)


def load(source):
    if source.is_dir():
        files={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file() and p.suffix in ('.csv','.gz','.json') and not {'hf-step40','weights','models','sources','data','partials'} & set(p.relative_to(source).parts)}
    elif source.name.endswith(('.csv','.csv.gz')): files={'scores.csv.gz' if source.name.endswith('.gz') else 'scores.csv':source.read_bytes()}
    else:
        with tarfile.open(source,'r:gz') as tar: files={m.name.removeprefix('return/'):tar.extractfile(m).read() for m in tar if m.isfile()}
    if 'collect-manifest.json' in files:
        for record in json.loads(files['collect-manifest.json'])['files']:
            raw=files.get(record['path'])
            require(raw is not None and len(raw)==record['bytes'] and t.sha(raw)==record['sha256'],'returned evidence hash differs: '+record['path'])
    selected=[(n,b) for n,b in files.items() if n.startswith('scores/') and n.endswith('/scores.csv')] if source.is_dir() else []
    if not selected: selected=[(n,b) for n,b in files.items() if n.endswith('scores.csv.gz')]
    if not selected: selected=[(n,b) for n,b in files.items() if n.endswith('scores.csv')]
    rows=[]
    for n,b in selected: rows.extend(csv.DictReader(io.StringIO((gzip.decompress(b) if n.endswith('.gz') else b).decode())))
    return rows,files


def job(a):
    if not a.source.exists(): raise NotReady('no returned score evidence')
    rows,files=load(a.source)
    if not rows: raise NotReady('no scores returned')
    fingerprint=s.digest(dict(rows=rows,costs={n:t.sha(b) for n,b in files.items() if n.endswith(('cost.json','summary.json'))}))
    if (a.out / 'read-complete.json').exists() and json.loads((a.out / 'read-complete.json').read_text())==dict(source=fingerprint,bootstrap=a.bootstrap,standin=a.standin):
        print('SKIP: reading complete'); return
    result=analyse(rows,a.bootstrap)
    synthetic=a.standin or any(json.loads(b).get('technical_synthetic',False) for n,b in files.items() if n.endswith(('identity.json','summary.json')))
    result['technical_synthetic']=synthetic
    summaries=[dict(path=n,**json.loads(b)) for n,b in sorted(files.items()) if n.endswith('/summary.json') and n.startswith('runs/')]
    result['cost']=dict(training_gpu_hours=sum(r.get('gpu_hours',0) for r in summaries if r.get('gpu_hours') is not None),training=summaries,
        preparation=[dict(path=n,**json.loads(b)) for n,b in sorted(files.items()) if n.endswith('cost.json')],costs_complete=False,missing_costs='Absent preparation/scoring/failed-run timings remain unknown; no proxy is a measurement',slot_proxy_gpu_hours=17.7,registered_maximum=956.)
    atomic(a.out / 'reading.json',result); atomic(a.out / 'reset-trigger.json',dict(reset_recipe=result['reset_recipe'],source_sha256=s.digest(rows),technical_synthetic=synthetic))
    atomic(a.out / 'read-complete.json',dict(source=fingerprint,bootstrap=a.bootstrap,standin=a.standin))
    print('Read',len(rows),'question rows; reset recipe:',result['reset_recipe'])


def configure(p):
    p.add_argument('--source',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=10000)
if __name__ == '__main__': raise SystemExit(entry(job,configure,'p3b-read'))
