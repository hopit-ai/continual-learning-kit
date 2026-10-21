#!/usr/bin/env python3
"""Registered seed-paired primary, retention, transitions and cost readout; missing is pending."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
from kit.a4_run import score_plan
import csv
import statistics as stats
import math
from collections import defaultdict


def interval(values,one_sided=False):
    require(len(values)==5,'paired seed-t requires all five seeds')
    mean=stats.mean(values);se=stats.stdev(values)/math.sqrt(5)
    tcrit=2.131846786 if one_sided else 2.776445105
    return dict(mean=mean,lower=mean-tcrit*se,upper=mean+tcrit*se,n=5,df=4,method='paired seed-t '+('one-sided 95%' if one_sided else 'two-sided 95%'))


def primary(matrix):
    require(set(matrix)==set(SEEDS) and all(len(v)==5 for v in matrix.values()),'primary requires five seeds x months 2-6')
    per_seed=[stats.mean(matrix[s]) for s in SEEDS]
    monthly=[stats.mean(matrix[s][m] for s in SEEDS) for m in range(5)]
    ci=interval(per_seed)
    established=ci['mean']>=5 and sum(x>0 for x in per_seed)>=4 and sum(x>0 for x in monthly)>=3 and ci['lower']>0
    return dict(established=established,per_seed=dict(zip(SEEDS,per_seed)),monthly=monthly,interval=ci,
        positive_seeds=sum(x>0 for x in per_seed),positive_months=sum(x>0 for x in monthly),bar_points=5)


def accuracy(rows,field='correct'):
    require(bool(rows),'empty registered accuracy denominator')
    return 100*stats.mean(int(r[field]) for r in rows)


def pair(before,after,field='correct'):
    def index(rows):return {(r['panel'],r['question_id']):int(r[field]) for r in rows}
    x,y=index(before),index(after)
    require(len(x)==len(before) and len(y)==len(after) and x.keys()==y.keys(),'unpaired answer coverage')
    return dict(n=len(x),correct_to_wrong=sum(x[k] and not y[k] for k in x),wrong_to_correct=sum(not x[k] and y[k] for k in x),
        delta_points=100*stats.mean(y[k]-x[k] for k in x))


def load(out,require_complete=True):
    records={};missing=[]
    if not (out/'binding.json').exists():raise NotReady('4A binding pending')
    bound=json.loads((out/'binding.json').read_text());binding_hash=s.file_hash(out/'binding.json')
    for row in score_plan():
        root=score_root(out,*row)
        if not (root/'manifest.json').exists():missing.append(root.name);continue
        frozen_files(root)
        observed=json.loads((root/'identity.json').read_text())
        require(observed['physical']==bound['physical'] and observed['binding']==binding_hash and
            observed['technical_synthetic']==bound['technical_synthetic'],'mixed scoring GPU, environment or binding in reader')
        with (root/'scores.csv').open() as f: records[row]=list(csv.DictReader(f))
    if missing and require_complete:raise NotReady(f'{len(missing)} registered score cells pending: '+missing[0])
    return records,missing


def analyse(records,stage_summaries,costs,bootstrap=2000):
    def get(kind,arm,seed,n,p,mode='off',holdout=False):
        rows=records[(kind,arm,seed,n,p,mode)]
        return [r for r in rows if int(r['holdout'])] if holdout else rows
    matrix={seed:[] for seed in SEEDS};transitions=[];retention=[];descriptive=[];ratios=[]
    for n,p in enumerate(MONTHS,1):
        frozen=get('arrival','frozen',0,n,p,holdout=True);memory=get('arrival','memory',0,n,p,'on',True)
        for seed in SEEDS:
            deployed=get('arrival','consolidation',seed,n,p,'on',True)
            if n>=2:matrix[seed].append(accuracy(deployed)-accuracy(memory))
            direct=get('arrival','direct',seed,n,p,holdout=True);off=get('arrival','consolidation',seed,n,p,'off',True)
            denom=accuracy(memory)-accuracy(frozen)
            ratios.append(dict(month=p,seed=seed,recovery_ratio=(accuracy(direct)-accuracy(frozen))/denom if denom else None,
                memory_gain_kept=(accuracy(off)-accuracy(frozen))/denom if denom else None))
            groups={'all_arrival':lambda r:True,'holdout':lambda r:int(r['holdout']),
                'recurring_issuer':lambda r:int(r['recurring_issuer'])==1}
            for fam in ('cash_conversion_change','fcf_margin_change'):groups['family:'+fam]=lambda r,f=fam:r['family']==f
            for st in ('high','low'):groups['stratum:'+st]=lambda r,s=st:int(r['holdout']) and r['stratum']==s
            a=get('arrival','consolidation',seed,n,p,'on');b=get('arrival','memory',0,n,p,'on')
            for group,test in groups.items():
                aa=[r for r in a if test(r)];bb=[r for r in b if test(r)]
                descriptive.append(dict(month=p,seed=seed,group=group,n=len(aa),descriptive_only=len(aa)<10,
                    explicit_gain=accuracy(aa)-accuracy(bb) if aa else None,
                    historical_gain=accuracy(aa,'historical_correct')-accuracy(bb,'historical_correct') if aa else None))
            for aa,bb,label in [(memory,deployed,'consolidation_vs_memory'),(frozen,direct,'direct_vs_frozen'),
                                (memory,direct,'direct_vs_memory'),(frozen,memory,'memory_vs_frozen'),
                                (frozen,deployed,'consolidation_vs_frozen'),(direct,deployed,'consolidation_vs_direct'),
                                (frozen,off,'consolidation_off_vs_frozen'),(memory,off,'consolidation_off_vs_memory'),
                                (direct,off,'consolidation_off_vs_direct'),
                                (off,deployed,'consolidation_memory_on_vs_off')]:
                for field in ('correct','historical_correct'):
                    transitions.append(dict(comparison=label,month=p,seed=seed,scorer=field,**pair(aa,bb,field)))
    # General retention against BOTH incoming and original base; holdouts against own arrival mode.
    for arm in ARMS:
        for n in (range(1,7) if arm=='direct' else BOUNDARIES):
            for panel in ('general-v1',*MONTHS[:n]):
                for mode in (('on','off') if arm=='consolidation' and panel in MONTHS else ('off',)):
                    for reference in (('arrival',) if panel in MONTHS else ('base','incoming')):
                        panels=('ifeval','knowledge','math') if panel=='general-v1' else (panel,)
                        for sub in panels:
                            losses=[]
                            for seed in SEEDS:
                                after=get('post',arm,seed,n,panel,mode)
                                if reference=='arrival':before=get('arrival',arm,seed,MONTHS.index(panel)+1,panel,mode,True)
                                elif reference=='base':before=get('base','frozen',0,0,panel)
                                else:before=get('incoming',arm,seed,n,panel)
                                before=[r for r in before if r['panel']==sub];after=[r for r in after if r['panel']==sub]
                                paired=pair(before,after);losses.append(-paired['delta_points'])
                                transitions.append(dict(comparison='retention',arm=arm,month=n,panel=sub,mode=mode,reference=reference,seed=seed,**paired))
                            ci=interval(losses,True)
                            retention.append(dict(arm=arm,month=n,panel=sub,mode=mode,reference=reference,interval=ci,
                                established=ci['upper']<=3,margin=3,is_registered_bar=True))
    # Acquired then lost, relative to the first arrival answer, across the full checkpoint sequence.
    acquired_lost=[]
    for arm in ARMS:
        for seed in SEEDS:
            for p in MONTHS:
                arrival_n=MONTHS.index(p)+1
                for mode in (('on','off') if arm=='consolidation' else ('off',)):
                    initial={r['question_id']:int(r['correct']) for r in get('arrival',arm,seed,arrival_n,p,mode,True)}
                    acquired=set();events=[]
                    for n in (range(arrival_n,7) if arm=='direct' else [m for m in BOUNDARIES if m>=arrival_n]):
                        current={r['question_id']:int(r['correct']) for r in get('post',arm,seed,n,p,mode)}
                        lost=sorted(q for q in acquired if not current[q])
                        acquired|={q for q in initial if not initial[q] and current[q]}
                        events.append(dict(stage=n,acquired_then_lost_ids=lost))
                    acquired_lost.append(dict(arm=arm,seed=seed,panel=p,mode=mode,events=events))
    # Unique-question recurrence denominators, equal month weights among present strata; missing stays undefined.
    strata={};question_sets={s:set() for s in ('high','low')}
    for st in ('high','low'):
        month_gains=[]
        for n,p in enumerate(MONTHS[1:],2):
            rr=[r for r in get('arrival','memory',0,n,p,'on',True) if r['stratum']==st]
            question_sets[st]|={(p,r['question_id']) for r in rr}
            vals=[r['explicit_gain'] for r in descriptive if r['month']==p and r['group']=='stratum:'+st and r['explicit_gain'] is not None]
            if vals:month_gains.append(stats.mean(vals))
        strata[st]=dict(unique_questions=len(question_sets[st]),months_present=len(month_gains),gain=stats.mean(month_gains) if month_gains else None)
    # Issuer-component bootstrap resamples whole components across dates, keeping monthly weights.
    import random
    rng=random.Random(20261020);issuer_records=defaultdict(list)
    for n,p in enumerate(MONTHS[1:],2):
        mem={r['question_id']:int(r['correct']) for r in get('arrival','memory',0,n,p,'on',True)}
        for seed in SEEDS:
            for r in get('arrival','consolidation',seed,n,p,'on',True):issuer_records[r['issuer']].append((p,int(r['correct'])-mem[r['question_id']]))
    issuers=sorted(issuer_records);draws=[]
    for _ in range(bootstrap):
        groups=defaultdict(list)
        for issuer in rng.choices(issuers,k=len(issuers)):
            for p,v in issuer_records[issuer]:groups[p].append(v)
        if len(groups)==5:draws.append(100*stats.mean(stats.mean(v) for v in groups.values()))
    draws.sort();boot=[draws[int(.025*(len(draws)-1))],draws[int(.975*(len(draws)-1))]] if draws else None
    # Serving and all-in costs use six-month arrival-correct denominators; collector charged once per system.
    serving=[];allin=[];generation_cost=defaultdict(float)
    for c in costs:
        if c.get('kind')=='scoring' and c.get('generation_id') and c.get('generated'):
            generation_cost[c['generation_id']]+=c.get('gpu_hours',0) or 0
    for arm in ('frozen','memory',*ARMS):
        for seed in ((0,) if arm in ('frozen','memory') else SEEDS):
            mode='on' if arm in ('memory','consolidation') else 'off'
            rr=[r for n,p in enumerate(MONTHS,1) for r in get('arrival',arm,seed,n,p,mode)]
            correct=sum(int(r['correct']) for r in rr);it=sum(int(r['input_tokens']) for r in rr);ot=sum(int(r['output_tokens']) for r in rr)
            serving.append(dict(arm=arm,seed=seed,arrival_correct=correct,output_tokens_per_correct=ot/correct if correct else None,
                all_tokens_per_correct=(it+ot)/correct if correct else None))
            collector=[r for n,p in enumerate(MONTHS,1) for r in get('arrival','frozen',0,n,p)] if arm!='frozen' else []
            train_tokens=sum(x.get('runtime_training_tokens') if x.get('runtime_training_tokens') is not None else
                sum(v for k,v in x.get('scheduled_tokens',{}).items() if k in ('input','response_including_eos'))
                for key,x in stage_summaries.items() if key.startswith(f'{arm}-s{seed}-'))
            charged=[c for c in costs if (c.get('arm')==arm and c.get('seed')==seed) or c.get('arm')=='frozen']
            unique={(r.get('generation_id',r['response_sha256']),r['panel'],r['question_id']):r for r in rr+collector}
            total_tokens=train_tokens+sum(int(r['input_tokens'])+int(r['output_tokens']) for r in unique.values())
            # Canonical generations are charged once to EACH deployable system using them.
            # Their shared-study spend is still counted only once in the global journal.
            generation_ids={r.get('generation_id') for r in unique.values()}
            serving_ids={r.get('generation_id') for r in rr}
            fixed=sum(c.get('gpu_hours',0) or 0 for c in charged if c.get('kind')!='scoring')
            deployment_hours=fixed+sum(generation_cost[g] for g in generation_ids if g)
            allin.append(dict(arm=arm,seed=seed,total_tokens=total_tokens,arrival_correct=correct,
                tokens_per_correct=total_tokens/correct if correct else None,
                gpu_hours=deployment_hours,gpu_hours_per_correct=deployment_hours/correct if correct else None,
                serving_gpu_hours=sum(generation_cost[g] for g in serving_ids if g),
                retention_gpu_hours=sum(c.get('gpu_hours',0) or 0 for c in charged if c.get('kind_row','arrival')!='arrival')))
    established=primary(matrix);kept=all(r['established'] for r in retention if r['arm']=='consolidation' and r['is_registered_bar'])
    # Equal updates and equal measured training-hours: all pairs reported, never choose a best checkpoint.
    equal=[]
    for seed in SEEDS:
        candidates={arm:[] for arm in ARMS}
        for arm in ARMS:
            updates=hours=0
            for n in (range(1,7) if arm=='direct' else BOUNDARIES):
                summary=stage_summaries[stage_name(arm,seed,n)]
                if summary['skipped']:continue
                updates+=40;hours+=sum(c.get('gpu_hours',0) or 0 for c in costs if c.get('kind')=='training' and c.get('arm')==arm and c.get('seed')==seed and c.get('month')==n)
                rows=[r for p in MONTHS for r in get('final',arm,seed,n,p)]
                candidates[arm].append(dict(month=n,updates=updates,gpu_hours=hours,accuracy=accuracy(rows)))
        for d in candidates['direct']:
            for c in candidates['consolidation']:
                if d['updates']==c['updates'] and d['updates'] in (40,80,120):equal.append(dict(seed=seed,basis='updates',direct=d,consolidation=c))
        for c in candidates['consolidation']:
            ds=candidates['direct']; below=[d for d in ds if d['gpu_hours']<=c['gpu_hours']];above=[d for d in ds if d['gpu_hours']>=c['gpu_hours']]
            equal.append(dict(seed=seed,basis='training_gpu_hours',consolidation=c,direct_lower=below[-1] if below else None,
                direct_upper=above[0] if above else None,method='report bracketing saved checkpoints; no interpolation or selection'))
    base_tpc=next(r['output_tokens_per_correct'] for r in serving if r['arm']=='frozen')
    for r in serving:r['at_cost']=r['output_tokens_per_correct'] is not None and base_tpc is not None and r['output_tokens_per_correct']>1.5*base_tpc
    memory_cost=next(r for r in allin if r['arm']=='memory')
    cost_comparison=[]
    memory_serving_hours=memory_cost['serving_gpu_hours']
    volume=sum(len(get('arrival','memory',0,n,p,'on')) for n,p in enumerate(MONTHS,1))
    for r in [x for x in allin if x['arm']=='consolidation']:
        gain=established['established'] and kept
        extra=r['gpu_hours']-memory_cost['gpu_hours']
        served=r['serving_gpu_hours']
        fixed=(r['gpu_hours']-served)-(memory_cost['gpu_hours']-memory_serving_hours)
        saving=(memory_serving_hours-served)/volume
        break_even=0 if fixed<=0 else math.ceil(fixed/saving) if saving>0 else None
        cost_comparison.append(dict(seed=r['seed'],label='gain at extra cost' if gain and extra>0 else 'cost accounting pending' if gain else 'not established',
            gpu_hours_above_memory=extra,cost_advantage=False,operation_cost_no_higher=extra<=0,
            allocation_cost_status='pending partner idle-reservation/storage reconciliation',
            projected_break_even_volume=break_even,
            break_even_basis='GPU-hour projection from measured arrivals; fixed deficit / positive per-query saving; undefined if no saving',
            break_even_reason='Undefined without a measured positive per-query serving-cost saving' if break_even is None else 'Measured operation-cost projection, excluding future volume shifts'))
    endpoints=[]
    for key,rr in records.items():
        for scope in ('all','holdout') if key[4] in MONTHS and key[0]=='arrival' else ('all',):
            rows=[r for r in rr if scope=='all' or int(r['holdout'])]
            if rows:endpoints.append(dict(row=key,scope=scope,n=len(rows),explicit_accuracy=accuracy(rows),historical_accuracy=accuracy(rows,'historical_correct')))
    return dict(verdict='ESTABLISHED' if established['established'] and kept else 'NOT ESTABLISHED',primary=established,
        retention=retention,retention_established=kept,descriptive=descriptive,ratios=ratios,transitions=transitions,
        acquired_then_lost=acquired_lost,recurrence=dict(strata=strata,v6_trigger=None,status='EXPLORATORY; 21 Oct deviation: aggregation was not frozen before v0',missing_stratum='undefined; no v6 trigger',
            aggregation='equal months 2-6; seed mean within month; unique question denominators; require both strata in all five months'),
        issuer_bootstrap=dict(interval=boot,draws=len(draws),unit='issuer component across dates',descriptive=True),
        serving=serving,all_in=allin,equal_compute=equal,cost_comparison=cost_comparison,endpoints=endpoints,
        shared_study_gpu_hours=sum(c.get('gpu_hours',0) or 0 for c in costs),
        note='Inconclusive means not established. Static k=3 remains the incumbent unless joint success is established.')


def job(a):
    source=getattr(a,'source',None)
    if source:
        import tarfile,gzip,io
        with tarfile.open(source,'r:gz') as tar:
            names=tar.getnames();require(len(set(names))==len(names),'duplicate compact archive members')
            raw={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
        manifest=json.loads(raw['collect-manifest.json'])
        require(manifest['technical_synthetic']==a.standin,'archive synthetic mismatch')
        require(set(raw)=={r['path'] for r in manifest['files']}|{'collect-manifest.json'},'unlisted compact archive members')
        for r in manifest['files']:require(t.sha(raw[r['path']])==r['sha256'],'compact member hash mismatch')
        bound=json.loads(raw['binding.json']);binding_hash=t.sha(raw['binding.json'])
        records={}
        for row in score_plan():
            name='scores/'+score_name(*row)+'/scores.csv.gz'
            if name not in raw:raise NotReady('compact score cell pending: '+name)
            observed=json.loads(raw['scores/'+score_name(*row)+'/identity.json'])
            require(observed['physical']==bound['physical'] and observed['binding']==binding_hash and
                observed['technical_synthetic']==a.standin,'mixed scoring GPU, environment or binding in compact reader')
            records[row]=list(csv.DictReader(io.StringIO(gzip.decompress(raw[name]).decode())))
        summaries={Path(n).parent.name:json.loads(b) for n,b in raw.items() if n.startswith('runs/') and n.endswith('/summary.json') and '/partials/' not in n}
        costs=[json.loads(b) for n,b in raw.items() if n.startswith('costs/') and n.endswith('.json')]
        root=a.out/'returned-reading'
    else:
        bound=binding(a.out,a.standin)
        with lock(a.out/'node-writer'):budget(a.out,bound)
        records,missing=load(a.out)
        summaries={p.parent.name:json.loads(p.read_text()) for p in (a.out/'runs').glob('*/summary.json')}
        costs=[json.loads(p.read_text()) for p in sorted((a.out/'costs').glob('*.json'))]
        binding_hash=s.file_hash(a.out/'binding.json');root=a.out/'reading'
    result=analyse(records,summaries,costs,a.bootstrap)
    result.update(technical_synthetic=a.standin,binding_sha256=binding_hash,score_cells=len(records),
        approved_ceiling_gpu_hours=approved_ceiling(bound), ceiling_gpu_hours=bound['ceiling_gpu_hours'])
    if a.standin:result['scientific_verdict']=None
    atomic(root/'summary.json',result)
    (root/'summary.md').write_text(('TECHNICAL STAND-IN ONLY\n\n' if a.standin else '')+result['verdict']+'\n\n'+result['note']+'\n')
    print(result['verdict'],';',len(records),'score cells')


def configure(p):
    p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--source',type=Path)
if __name__=='__main__':raise SystemExit(entry(job,configure,'a4-read'))
