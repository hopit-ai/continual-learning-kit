#!/usr/bin/env python3
"""Connected acquisition, cumulative retention, paired uncertainty, survival and serving costs."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit.p3b_read import load, bound
from kit.c_common import protocol as c_protocol
from kit.c_read import accuracy, status, paired, question_interval
from collections import defaultdict
import statistics


def analyse(rows, doc, admitted=None, repeats=10000):
    groups=defaultdict(dict)
    expected_ids={p:{r['id'] for r in legal_panels()['test']} if p=='test' else
                  {r['id'] for r in hard_data()['eval.jsonl']} if p=='math_hard' else
                  {r['id'] for r in t.read_jsonl(KIT/'panels/general-v1.jsonl') if r['panel']==p}
                  for p in ('test','math_hard','ifeval','knowledge','math')}
    legal_lookup={r['id']:r for r in legal_panels()['test']}
    for original in rows:
        r=dict(original); key=(r['arm'],int(r['seed']),int(r['step']),r['panel'])
        require(key[1] in SEEDS and key[2] in STEPS and int(r['cap'])==2048,'unregistered seed/step/cap')
        require(r['panel'] in (*PANELS,'hellaswag'),'unknown score panel')
        require(r['arm'] in ('base','finqa-plain','finqa-sibling','legal-plain','math-plain','math-sibling',*[legal_arm(c) for c in CONFIGS],
                             'standalone-legal-plain',*[f'standalone-legal-{c}' for c in CONFIGS],'standalone-math-plain','standalone-math-sibling'),'unregistered score arm')
        require(r['question_id'] not in groups[key],'duplicate score row')
        require(int(r['correct']) in (0,1) and int(r['finished']) in (0,1) and int(r['output_tokens'])>=0,'malformed score row')
        if r['panel']=='test':
            item=legal_lookup.get(r['question_id'])
            require(item is not None and r['gold']==item['gold'] and r['group']==item['group'],'legal gold/document binding changed')
            require(int(r['tolerant_correct'])==int(r['correct']),'tolerant parser mismatch')
        groups[key][r['question_id']]=r
    for (*_,p),g in groups.items():
        if p in expected_ids: require(set(g)==expected_ids[p],'partial/foreign frozen panel: '+p)
        if p=='finqa': require(len(g)==1147,'FinQA requires 1147 questions')
    def get(arm,seed,step,p): return groups.get((arm,seed,step,p))
    def contrast(arm,step,ref,refstep,p):
        pairs=[]
        for seed in SEEDS:
            a=get(ref,seed,refstep,p) or (get(ref,101,refstep,p) if doc['technical_synthetic'] else None); b=get(arm,seed,step,p)
            if a and b: pairs.append(dict(seed=seed,**paired(a,b,'correct',repeats,p=='test')))
        return dict(seeds=pairs,seed_bound=bound([r['change'] for r in pairs],3))
    def bar(arm,step,ref,refstep,p,margin,strict=False):
        c=contrast(arm,step,ref,refstep,p)
        result=dict(**c,margin=margin,status=status(c['seed_bound'],margin,strict))
        if p in ('knowledge','hellaswag'):
            from kit.ref_guard import apply, METHOD
            cells=[r for (a,seed,k,panel),group in groups.items() if panel==p and
                   ((a,k)==(arm,step) or (a,k)==(ref,refstep)) for r in group.values()]
            apply(result, cells)
        if p=='knowledge' and result['status'] not in ('UNVERIFIED-HARDWARE',) and result.get('reason') != 'MIXED-SCORERS' and not all(r.get('_guard_method') == METHOD for r in cells):
            from kit.m2_format import letter_valid,letter_cells
            cells=[r for (a,seed,k,panel),group in groups.items() if panel==p and
                   ((a,k)==(arm,step) or (a,k)==(ref,refstep)) for r in group.values()]
            if not letter_valid(cells):result.update(status='PENDING',reason='FORMAT-CONFOUNDED',format_cells=letter_cells(cells))
        return result
    summaries=[]; families=[]
    family_map={r['id']:{x.split(':')[0] for x in r.get('meta',{}).get('instruction_id_list',[])}
                for r in t.read_jsonl(KIT/'panels/general-v1.jsonl') if r['panel']=='ifeval'}
    for (arm,seed,step,p),g in sorted(groups.items()):
        base=get('base',seed,0,p) or (get('base',101,0,p) if doc['technical_synthetic'] else None); ncorrect=sum(int(r['correct']) for r in g.values()); tokens=sum(int(r['output_tokens']) for r in g.values())
        bc=sum(int(r['correct']) for r in base.values()) if base else 0; bt=sum(int(r['output_tokens']) for r in base.values()) if base else 0
        ratio=(tokens/ncorrect)/(bt/bc) if ncorrect and bc and bt else None
        row=dict(arm=arm,seed=seed,step=step,panel=p,n=len(g),accuracy=accuracy(g),correct=ncorrect,output_tokens=tokens,
                 tokens_per_correct=tokens/ncorrect if ncorrect else None,serving_ratio=ratio,
                 serving_status='UNDEFINED' if ratio is None else 'AT_COST' if ratio>1.5 else 'PASS',
                 finish_rate=sum(int(r['finished']) for r in g.values())/len(g),
                 uncertainty=question_interval({q:100*int(r['correct']) for q,r in g.items()},g,repeats,p=='test'))
        if p in ('knowledge','hellaswag'):
            from kit.m2_format import letter_cells
            row['format_cells']=letter_cells(list(g.values()))
        if p=='test':
            row.update(strict_accuracy=accuracy(g,'strict_correct'),strict_format=accuracy(g,'strict_format'),
                       tolerant_parseability=accuracy(g,'tolerant_parseable'),tolerant_minus_strict=accuracy(g)-accuracy(g,'strict_correct'),
                       per_label={label:dict(n=sum(r['gold']==label for r in g.values()),correct=sum(int(r['correct']) for r in g.values() if r['gold']==label)) for label in legal.LABELS['contract_nli']})
        summaries.append(row)
        if p=='ifeval' and base:
            previous=get(arm,seed,0,p) or base
            for family in sorted(set().union(*family_map.values())):
                ids={q for q in g if family in family_map.get(q,set())}
                families.append(dict(arm=arm,seed=seed,step=step,family=family,n=len(ids),
                    membership_ids=sorted(ids),base_correct=sum(int(base[q]['correct']) for q in ids),correct=sum(int(g[q]['correct']) for q in ids),
                    lost_ids=sorted(q for q in ids if int(base[q]['correct']) and not int(g[q]['correct'])),
                    recovered_ids=sorted(q for q in ids if int(base[q]['correct']) and not int(previous[q]['correct']) and int(g[q]['correct']))))
    trajectories=[];missing=[]
    legal_configs=['plain']+([admitted] if admitted else [])
    for trajectory,config in [('plain','plain'),('sibling',admitted)]:
        if config is None:
            trajectories.append(dict(trajectory=trajectory,status='PENDING',reason='Pending connected finals; sibling MATH never runs'));continue
        finqa='finqa-'+trajectory; legal_stage='legal-'+config; math_stage='math-'+trajectory
        enabled=doc['hard_math']=='QUALIFIES'
        boundaries=[(finqa,40),(legal_stage,40)]+([(math_stage,40)] if enabled else [])
        expected=[('base',seed,0,p) for seed in SEEDS for p in PANELS]
        expected += [(a,seed,k,p) for a,k in boundaries for seed in SEEDS for p in PANELS]
        expected += [(legal_stage,seed,k,p) for seed in SEEDS for k in STEPS for p in ('test',)]
        expected += [(legal_stage,seed,k,p) for seed in SEEDS for k in RETENTION_STEPS for p in PANELS]
        if enabled: expected += [(math_stage,seed,k,p) for seed in SEEDS for k in STEPS for p in PANELS]
        missing += [dict(arm=a,seed=b,step=c,panel=d) for a,b,c,d in sorted(set(expected)) if (a,b,c,d) not in groups]
        acquisition={}
        acquisition['finqa']=bar(finqa,40,'base',0,'finqa',0,True)
        for task,arm,p,threshold in [('contract_nli',legal_stage,'test',None)]+([('math_hard',math_stage,'math_hard',6)] if enabled else []):
            incoming=contrast(arm,40,arm,0,p); gain=incoming['seed_bound']
            base_gain=contrast(arm,40,'base',0,p)
            final=bound([accuracy(get(arm,seed,40,p)) for seed in SEEDS if get(arm,seed,40,p)],3)
            positives=sum(r['change']>0 for r in incoming['seeds'])
            checks=[status(gain,0,True)]; mean_ok=gain['mean'] is not None and (threshold is None or gain['mean']>=threshold)
            if p=='test': checks.append(status(final,c_protocol()['threshold']))
            outcome='PENDING' if 'PENDING' in checks else 'INCONCLUSIVE' if 'FAIL' in checks or positives<3 or not mean_ok else 'PASS' if all(c=='PASS' for c in checks) else 'INCONCLUSIVE'
            base_values=[accuracy(get('base',seed,0,p)) for seed in SEEDS if get('base',seed,0,p)]
            if doc['technical_synthetic'] and len(base_values)==1:base_values*=3
            base_mean=statistics.mean(base_values) if len(base_values)==3 else None
            label='ACQUISITION' if outcome=='PASS' else outcome
            if p=='math_hard' and base_mean is None:
                outcome='PENDING';label='PENDING_BASE'
            elif p=='math_hard' and outcome=='PASS' and final['mean']<=base_mean: label='RECOVERY'
            acquisition[task]=dict(status=outcome,label=label,positive_seeds=positives,from_incoming=incoming,from_base=base_gain,
                                   final_seed_bound=final,measured_base=base_mean,threshold=69.7 if p=='test' else 6)
        retention={}
        for stage,arm in [(2,legal_stage)]+([(3,math_stage)] if enabled else []):
            retention[f'finqa_stage{stage}']=bar(arm,40,finqa,40,'finqa',-1)
        if enabled: retention['contract_nli_stage3']=bar(math_stage,40,legal_stage,40,'test',-1)
        for stage,(arm,k) in enumerate(boundaries,1):
            for p in ('ifeval','knowledge','math'): retention[f'{p}_stage{stage}']=bar(arm,k,'base',0,p,-3)
        sequence={}
        for task,arm,p in [('contract_nli',legal_stage,'test')]+([('math_hard',math_stage,'math_hard')] if enabled else []):
            comparator='standalone-'+arm
            sequence[task]=bar(arm,40,comparator,40,p,-2)
            sequence[task]['policy']='Same recipe, matched seeds, identical panel and scoring. Missing/incompatible comparator stays pending; no easy-MATH substitution.'
            if p=='math_hard':
                standalone=contrast(comparator,40,'base',0,p)
                g=standalone['seed_bound']; positives=sum(r['change']>0 for r in standalone['seeds'])
                checks=status(g,0,True)
                standalone['status']='PENDING' if checks=='PENDING' else 'INCONCLUSIVE' if positives<3 or g['mean']<6 or checks=='FAIL' else checks
                standalone['positive_seeds']=positives
                standalone['mean_gain_bar']=6
                sequence[task]['standalone_acquisition']=standalone
        survival=[]
        for p,own,later in [('finqa',finqa,[(legal_stage,40)]+([(math_stage,40)] if enabled else [])),
                            ('test',legal_stage,[(math_stage,40)] if enabled else [])]:
            for seed in SEEDS:
                base=get('base',seed,0,p) or (get('base',101,0,p) if doc['technical_synthetic'] else None)
                acquired=get(own,seed,40,p)
                if not base or not acquired: continue
                require(set(base)==set(acquired),'acquisition cohort IDs changed')
                ids=sorted(q for q in base if not int(base[q]['correct']) and int(acquired[q]['correct']))
                for arm,k in later:
                    end=get(arm,seed,k,p)
                    if end:
                        kept=sum(int(end[q]['correct']) for q in ids)
                        survival.append(dict(panel=p,seed=seed,boundary=arm,n=len(ids),acquired_ids=ids,retained=kept,
                                             survival=100*kept/len(ids) if ids else None,lost_ids=[q for q in ids if not int(end[q]['correct'])]))
        curves=[]
        for arm,p in [(legal_stage,'test')]+([(math_stage,'math_hard')] if enabled else []):
            for seed in SEEDS:
                points=[get(arm,seed,k,p) for k in STEPS]
                if all(points):
                    scores=[accuracy(g) for g in points]
                    curves.append(dict(arm=arm,seed=seed,panel=p,steps=STEPS,points=scores,gain_auc=sum((a+b)/8 for a,b in zip(scores,scores[1:]))-scores[0]))
        trajectories.append(dict(trajectory=trajectory,configuration=config,acquisition=acquisition,retention=retention,
                                 sequence_acquisition=sequence,acquired_answer_survival=survival,curves=curves))
    return dict(technical_synthetic=doc['technical_synthetic'],complete=not missing and len(legal_configs)==2,
                hard_math=doc['hard_math'],stage3='DROPPED' if doc['hard_math']=='DOES_NOT_QUALIFY' else 'PENDING' if doc['hard_math']=='INCOMPLETE' else 'ENABLED',
                missing=missing,trajectories=trajectories,summaries=summaries,ifeval_families=families,
                interpretation='Analysis bars never stop registered stages. Seed-t bounds are one-sided 95%; question bootstrap is separate, legal clustered by document. Undefined serving denominator stays undefined; >1.5x base is at cost.')


def job(a):
    rows,files=load(a.source)
    require('binding.json' in files,'missing message-2 binding')
    doc=json.loads(files['binding.json'])
    require(doc.get('technical_synthetic') is a.standin,'technical_synthetic archive requires --standin')
    require(doc['protocol']==protocol(),'returned message-2 protocol changed')
    validate_evidence(doc,files,a.standin)
    admission_doc=json.loads(files.get('admission.json',b'{}'))
    admitted=admission_doc.get('configuration') if admission_doc.get('status')=='COMPLETE' else None
    if admitted is not None:
        from kit.m2_admit import decision
        recomputed,_=decision(rows,nomination_from_files(doc,files),a.bootstrap)
        require(recomputed['status']=='COMPLETE' and recomputed['configuration']==admitted,'written admission differs from connected score evidence')
    result=analyse(rows,doc,admitted,a.bootstrap)
    costs=[dict(path=n,**json.loads(raw)) for n,raw in sorted(files.items()) if n.endswith('cost.json') or '/training-costs/' in n or n.startswith('runs/') and n.endswith('/summary.json')]
    charged={r['path'].split('/training-costs/')[0] for r in costs if '/training-costs/' in r['path']}
    result['generation']={n:json.loads(raw) for n,raw in sorted(files.items()) if n.endswith('/generation-summary.json')}
    result['cost']=dict(records=costs,recorded_gpu_hours=sum(r['gpu_hours'] for r in costs if r.get('gpu_hours') is not None and not (r['path'].endswith('/summary.json') and r['path'].rsplit('/',1)[0] in charged)),complete=False,
                        note='Process GPU-hours include recorded failed attempts. Allocation, idle time and absent timings require the return note; estimates are not measurements.')
    atomic(a.out/'reading.json',result)
    print('Read',len(rows),'question rows; missing cells:',len(result['missing']),'; stage 3:',result['stage3'])


def configure(p):
    p.add_argument('--source',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=10000)
if __name__=='__main__': raise SystemExit(entry(job,configure,'m2-read'))
