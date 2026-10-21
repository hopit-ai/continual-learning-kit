#!/usr/bin/env python3
"""Registered C bars, seed tables, document bootstrap, retention/churn and serving cost."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.c_common import *
from kit.p3b_read import bound, load, lower
from kit.p3b_score import write_csv
from collections import defaultdict
import statistics
import random
import math


def accuracy(rows, field='correct'):
    return 100*sum(int(r[field]) for r in rows.values())/len(rows) if rows else None


def status(stats, margin, strict=False):
    if stats['mean'] is None or stats['lower'] is None: return 'PENDING'
    if stats['lower'] > margin if strict else stats['lower'] >= margin: return 'PASS'
    if stats['upper'] <= margin if strict else stats['upper'] < margin: return 'FAIL'
    return 'INCONCLUSIVE'


def question_interval(values, rows, repeats, cluster=True):
    """Bootstrap arbitrary question statistics using the same document draws."""
    require(repeats > 0 and bool(values) and set(values)==set(rows), 'empty/mismatched question statistics')
    groups = defaultdict(list)
    for q, value in values.items():
        if cluster: require(bool(rows[q].get('group')), 'missing legal document group')
        groups[rows[q]['group'] if cluster else q].append(value)
    population = [(sum(v),len(v)) for _,v in sorted(groups.items())]
    rng = random.Random(0); samples = []
    for _ in range(repeats):
        draw = rng.choices(population, k=len(population))
        samples.append(sum(v for v,n in draw)/sum(n for v,n in draw))
    samples.sort()
    return dict(n=len(values), clusters=len(population),
        question_bootstrap_95=[samples[int(.025*repeats)], samples[min(repeats-1,int(.975*repeats))]],
        resampling='document' if cluster else 'question')


def paired(left, right, field, repeats, cluster=False):
    require(set(left) == set(right) and bool(left), 'paired C question sets differ/empty')
    if cluster:
        require(all(left[q].get('group')==right[q].get('group') for q in left), 'legal document binding changed')
    values = {q:100*(int(right[q][field])-int(left[q][field])) for q in left}
    return dict(change=statistics.mean(values.values()),
        gained=sorted(q for q,v in values.items() if v>0), lost=sorted(q for q,v in values.items() if v<0),
        **question_interval(values,left,repeats,cluster))


def curve_question_intervals(points, field, repeats):
    require(all(set(g)==set(points[0]) for g in points), 'legal curve question sets differ')
    require(all(g[q].get('group')==points[0][q].get('group') for g in points for q in g), 'legal curve document binding changed')
    def ci(values): return question_interval(values,points[0],repeats)
    values=[{q:100*int(r[field]) for q,r in g.items()} for g in points]
    return dict(points=[ci(v) for v in values],
        gain=ci({q:values[-1][q]-values[0][q] for q in points[0]}),
        gain_auc=ci({q:sum((a[q]+b[q])/8 for a,b in zip(values,values[1:]))-values[0][q] for q in points[0]}))


def analyse(rows, doc, repeats=10000):
    require(repeats > 0, 'bootstrap repetitions must be positive')
    groups = defaultdict(dict)
    frozen = {p:{r['id']:r for r in v} for p,v in panels().items()}
    from kit.score_forgetting import load_panel
    general_items, _ = load_panel()
    general_ids = {p:{r['id'] for r in general_items if r['panel']==p} for p in ('ifeval','knowledge','math')}
    for original in rows:
        if original['arm'].startswith('sequence-probe-') or original['panel'] == 'hellaswag': continue
        r = dict(original); arm = r['arm']; seed = int(r['seed']); step = int(r['step']); panel = r['panel']
        require(seed in SEEDS and step in STEPS and int(r['cap']) == 2048, 'unregistered C seed/step/cap')
        if arm != 'base':
            route, recipe = arm.split('-',1)
            require(route in ('standalone','sequence') and recipe in arms(doc), 'unregistered C score arm')
        else: require(step == 0, 'base score must be step 0')
        require(panel in ('panel','test','finqa','ifeval','knowledge','math'), 'unknown C panel')
        require(panel in ('panel','test') or step in RETENTION_STEPS, 'unregistered retention step')
        for key in ('correct','finished'):
            require(int(r[key]) in (0,1), 'nonbinary C score')
        require(int(r['output_tokens']) >= 0, 'negative output tokens')
        if panel in frozen:
            item = frozen[panel].get(r['question_id']); require(item is not None, 'unknown legal test/panel ID')
            require(r.get('gold') == item['gold'] and r.get('group') == item['group'], 'wrong legal gold/document binding')
            for field in ('strict_correct','tolerant_correct','strict_format','tolerant_parseable'):
                require(int(r[field]) in (0,1), 'malformed legal parser score')
            require(r['correct'] == r['tolerant_correct'] and int(r['strict_correct']) <= int(r['strict_format']) and int(r['tolerant_correct']) <= int(r['tolerant_parseable']), 'inconsistent legal parser scores')
        key = (arm,seed,step,panel)
        require(r['question_id'] not in groups[key], 'duplicate C score question')
        groups[key][r['question_id']] = r
    for (arm,seed,step,panel), g in groups.items():
        expected = set(frozen[panel]) if panel in frozen else general_ids.get(panel)
        if expected is not None: require(set(g) == expected, 'partial C scoring panel: ' + panel)
        elif not doc['technical_synthetic']: require(len(g) == 1147, 'FinQA requires 1147 questions')
    base = {}
    for (arm,seed,step,p),g in groups.items():
        if arm != 'base': continue
        if p in base:
            fields = ('correct','output_tokens','finished') + (('strict_correct',) if p in frozen else ())
            require({q:tuple(r[k] for k in fields) for q,r in g.items()} == {q:tuple(r[k] for k in fields) for q,r in base[p].items()}, 'inconsistent repeated base outputs')
        base[p] = g
    expected = [('base',101,0,p) for p in ('panel','test','finqa','ifeval','knowledge','math')]
    expected += [(f'{route}-{recipe}',seed,step,p) for recipe in arms(doc) for route in ('standalone','sequence')
                 for seed in SEEDS for step in STEPS for p in ('panel','test')]
    expected += [(f'{route}-{recipe}',seed,step,p) for recipe in arms(doc) for route in ('standalone','sequence')
                 for seed in SEEDS for step in RETENTION_STEPS for p in ('finqa','ifeval','knowledge','math')]
    missing = [dict(arm=a,seed=b,step=c,panel=d) for a,b,c,d in expected if (a,b,c,d) not in groups]
    summaries = []; seed_tables = []; comparisons = []; transitions = []; curves = []
    family_map = {r['id']:{x.split(':')[0] for x in r.get('meta',{}).get('instruction_id_list',[])} for r in general_items if r['panel']=='ifeval'}
    families = []
    for (arm,seed,step,p),g in sorted(groups.items()):
        total = sum(int(r['output_tokens']) for r in g.values()); correct = sum(int(r['correct']) for r in g.values())
        reference = base.get(p)
        base_correct = sum(int(r['correct']) for r in reference.values()) if reference else 0
        base_tokens = sum(int(r['output_tokens']) for r in reference.values()) if reference else 0
        ratio = (total/correct)/(base_tokens/base_correct) if correct and base_tokens and base_correct else None
        summary = dict(arm=arm,seed=seed,step=step,panel=p,n=len(g),accuracy=accuracy(g),correct=correct,
            output_tokens=total,tokens_per_correct=total/correct if correct else None,serving_ratio=ratio,
            serving_status='UNDEFINED' if ratio is None else 'AT_COST' if ratio>1.5 else 'PASS',
            finish_rate=sum(int(r['finished']) for r in g.values())/len(g))
        if p in frozen:
            summary.update(strict_accuracy=accuracy(g,'strict_correct'),tolerant_accuracy=accuracy(g,'tolerant_correct'),
                question_intervals={field:question_interval({q:100*int(r[field]) for q,r in g.items()},g,repeats)
                    for field in ('strict_correct','tolerant_correct')},
                strict_format=accuracy(g,'strict_format'),tolerant_parseability=accuracy(g,'tolerant_parseable'),
                tolerant_minus_strict=accuracy(g,'tolerant_correct')-accuracy(g,'strict_correct'),
                per_label={label:dict(n=sum(r['gold']==label for r in g.values()),
                    tolerant_recall=100*sum(int(r['tolerant_correct']) for r in g.values() if r['gold']==label)/sum(r['gold']==label for r in g.values()),
                    strict_recall=100*sum(int(r['strict_correct']) for r in g.values() if r['gold']==label)/sum(r['gold']==label for r in g.values()),
                    tolerant_correct=sum(int(r['tolerant_correct']) for r in g.values() if r['gold']==label),
                    strict_correct=sum(int(r['strict_correct']) for r in g.values() if r['gold']==label)) for label in legal.LABELS['contract_nli']})
        summaries.append(summary)
        if p == 'ifeval' and reference:
            require(set(reference) == set(g), 'general panel IDs changed')
            for family in sorted(set().union(*family_map.values())):
                ids = {q for q in g if family in family_map.get(q,set())}
                incoming = groups.get((arm,seed,0,p),{})
                families.append(dict(arm=arm,seed=seed,step=step,family=family,n=len(ids),
                    base_correct=sum(int(reference[q]['correct']) for q in ids), correct=sum(int(g[q]['correct']) for q in ids),
                    lost=sorted(q for q in ids if int(reference[q]['correct']) and not int(g[q]['correct'])),
                    recovered=sorted(q for q in ids if q in incoming and int(reference[q]['correct']) and not int(incoming[q]['correct']) and int(g[q]['correct']))))
    for recipe in arms(doc):
        for route in ('standalone','sequence'):
            arm = route+'-'+recipe
            for seed in SEEDS:
                for p in ('panel','test'):
                    start_rows = groups.get((arm,seed,0,p)); end = groups.get((arm,seed,40,p))
                    if not start_rows or not end: continue
                    for parser,field in [('strict','strict_correct'),('tolerant','tolerant_correct')]:
                        transition = paired(start_rows,end,field,repeats,True)
                        transitions.append(dict(arm=arm,seed=seed,panel=p,parser=parser,**transition))
                        points = [groups.get((arm,seed,k,p)) for k in STEPS]
                        auc = None
                        if all(points):
                            scores = [accuracy(g,field) for g in points]
                            auc = sum((a+b)*5 for a,b in zip(scores,scores[1:]))/40-scores[0]
                            curves.append(dict(arm=arm,seed=seed,panel=p,parser=parser,steps=STEPS,points=scores,gain_auc=auc,
                                question_intervals=curve_question_intervals(points,field,repeats)))
                        seed_tables.append(dict(arm=arm,seed=seed,panel=p,parser=parser,initial=accuracy(start_rows,field),final=accuracy(end,field),
                            gain=accuracy(end,field)-accuracy(start_rows,field),gain_auc=auc))
    primary = [r for r in seed_tables if r['arm'] == 'standalone-'+doc['W'] and r['panel']=='test' and r['parser']=='tolerant']
    final = bound([r['final'] for r in primary],5); gains = bound([r['gain'] for r in primary],5)
    positives = sum(r['gain']>0 for r in primary)
    checks = [status(final,69.7),status(gains,0,True)]
    decision = 'PENDING' if 'PENDING' in checks or len(primary)!=5 else 'PASS' if checks==['PASS','PASS'] and positives>=4 else 'FAIL' if 'FAIL' in checks or positives<4 else 'INCONCLUSIVE'
    goal1 = dict(primary='W standalone / test / tolerant',W=doc['W'],F_test=82,threshold=69.7,final_seed_bound=final,
        gain_seed_bound=gains,positive_seeds=positives,status=decision,score_frontier_ratio=final['mean']/82 if final['mean'] is not None else None,
        frontier_sampling=dict(n=100,correct=82,wilson_95=[.733,.883],note='Rounded binomial interval; seed bound conditions on observed 82. Different thinking/compute budgets.'))
    route_comparison = []
    for recipe in arms(doc):
        for panel in ('panel','test'):
            for parser in ('strict','tolerant'):
                a = {r['seed']:r for r in seed_tables if r['arm']=='standalone-'+recipe and r['panel']==panel and r['parser']==parser}
                b = {r['seed']:r for r in seed_tables if r['arm']=='sequence-'+recipe and r['panel']==panel and r['parser']==parser}
                route_comparison.append(dict(recipe=recipe,panel=panel,parser=parser,
                    contrasts={key:bound([b[k][key]-a[k][key] for k in SEEDS if k in a and k in b and a[k][key] is not None and b[k][key] is not None],5) for key in ('final','gain','gain_auc')}))
    retention = []
    for recipe in arms(doc):
        contrasts = defaultdict(list); survival = []
        for seed in SEEDS:
            seq = 'sequence-'+recipe; alone = 'standalone-'+recipe
            for p in ('test','finqa','ifeval','knowledge','math'):
                end = groups.get((seq,seed,40,p)); incoming = groups.get((seq,seed,0,p)); reference = groups.get((alone,seed,40,p)) if p=='test' else incoming if p=='finqa' else base.get(p)
                if end and reference: contrasts['legal_vs_standalone' if p=='test' else 'stage2_'+p].append(accuracy(end)-accuracy(reference))
                if p!='test' and incoming and p in base: contrasts['stage1_'+p].append(accuracy(incoming)-accuracy(base[p]))
                if p=='finqa' and end and incoming and p in base:
                    require(set(end)==set(incoming)==set(base[p]), 'FinQA survival IDs differ')
                    acquired = [q for q in end if not int(base[p][q]['correct']) and int(incoming[q]['correct'])]
                    survival.append(dict(seed=seed,acquired_ids=sorted(acquired),n=len(acquired),survived=sum(int(end[q]['correct']) for q in acquired),
                        lost=sorted(q for q in acquired if not int(end[q]['correct'])),net_change=accuracy(end)-accuracy(incoming)))
            for p in ('finqa','ifeval','knowledge','math'):
                for step in (20,40):
                    pre = groups.get((seq,seed,0,p)); post = groups.get((seq,seed,step,p))
                    if pre and post: transitions.append(dict(arm=seq,seed=seed,panel=p,step=step,parser='registered',**paired(pre,post,'correct',repeats)))
        margins = {'legal_vs_standalone':-2,'stage2_finqa':-1,'stage1_finqa':0}
        margins.update({f'{stage}_{p}':-3 for stage in ('stage1','stage2') for p in ('ifeval','knowledge','math')})
        bars = {key:dict(seed_bound=bound(contrasts[key],5),margin=margin,status=status(bound(contrasts[key],5),margin,key=='stage1_finqa')) for key,margin in margins.items()}
        states = [r['status'] for r in bars.values()]
        serving = {}
        for route in ('standalone','sequence'):
            serving[route] = {}
            for task in ('test','finqa'):
                values = [r for r in summaries if r['arm']==route+'-'+recipe and r['panel']==task and r['step']==40]
                ratios = [r['serving_ratio'] for r in values]
                serving[route][task] = dict(per_seed=[dict(seed=r['seed'],ratio=r['serving_ratio'],tokens_per_correct=r['tokens_per_correct'],output_tokens=r['output_tokens']) for r in values],
                    status='PENDING_OR_UNDEFINED' if len(values)!=5 or any(v is None for v in ratios) else 'AT_COST' if any(v>1.5 for v in ratios) else 'PASS')
        at_cost = all(x=='PASS' for x in states) and any(r['status']=='AT_COST' for route in serving.values() for r in route.values())
        retention.append(dict(recipe=recipe,bars=bars,acquired_finqa_survival=survival,serving=serving,at_cost=at_cost,
                              status='PENDING' if 'PENDING' in states else 'PASS' if all(x=='PASS' for x in states) else 'FAIL' if 'FAIL' in states else 'INCONCLUSIVE'))
        for route in ('standalone','sequence'):
            for parser in ('strict','tolerant'):
                for p in ('panel','test'):
                    left = {r['seed']:r for r in seed_tables if r['arm']==route+'-answer-only' and r['parser']==parser and r['panel']==p}
                    right = {r['seed']:r for r in seed_tables if r['arm']==route+'-'+recipe and r['parser']==parser and r['panel']==p}
                    label_comparisons = {}
                    field = 'strict_correct' if parser=='strict' else 'tolerant_correct'
                    for label in legal.LABELS['contract_nli']:
                        ids = {q for q,r in frozen[p].items() if r['gold']==label}
                        by_seed = []
                        for seed in SEEDS:
                            candidate = [groups.get((route+'-'+recipe,seed,k,p)) for k in (0,40)]
                            control = [groups.get((route+'-answer-only',seed,k,p)) for k in (0,40)]
                            if not all(candidate+control): continue
                            points = [100*sum(int(g[q][field]) for q in ids)/len(ids) for g in candidate+control]
                            by_seed.append(dict(seed=seed,n=len(ids),final_recall_difference=points[1]-points[3],gain_recall_difference=(points[1]-points[0])-(points[3]-points[2])))
                        label_comparisons[label] = dict(seeds=by_seed,final_recall_bound=bound([r['final_recall_difference'] for r in by_seed],5),gain_recall_bound=bound([r['gain_recall_difference'] for r in by_seed],5))
                    comparisons.append(dict(recipe=recipe,route=route,panel=p,parser=parser,per_label=label_comparisons,
                        final_vs_answer_only=bound([right[k]['final']-left[k]['final'] for k in SEEDS if k in left and k in right],5),
                        gain_vs_answer_only=bound([right[k]['gain']-left[k]['gain'] for k in SEEDS if k in left and k in right],5)))
    goal1['at_cost'] = goal1['status']=='PASS' and any(r['serving_status']=='AT_COST' for r in summaries if r['arm']=='standalone-'+doc['W'] and r['panel']=='test' and r['step']==40)
    return dict(technical_synthetic=doc['technical_synthetic'],complete=not missing,missing=missing,goal1=goal1,
        seed_tables=seed_tables,summaries=summaries,curves=curves,question_pairs=transitions,answer_only_comparison=comparisons,
        retention=retention,route_comparison=route_comparison,original_v4_stage1=doc.get('original_v4_stage1'),ifeval_families=families,limits='Five seeds; 100 test pairs clustered by document; label checks do not verify rationale truth; cross-arm sequence starts include stage-1 history; answer-only also teaches labels.',
        start_policy=doc['start_policy'])


def job(a):
    if not a.source.exists(): raise NotReady('no C score return')
    rows, files = load(a.source)
    if not rows: raise NotReady('no C scores')
    candidates = [json.loads(b) for n,b in files.items() if n.endswith('binding.json')]
    require(len(candidates)==1, 'return must include exactly one C binding.json')
    doc = candidates[0]; require(doc['protocol']==protocol(), 'returned C protocol changed')
    require(doc['technical_synthetic']==a.standin, 'returned synthetic type mismatch')
    result = analyse(rows,doc,a.bootstrap)
    training = [dict(path=n,**json.loads(b)) for n,b in sorted(files.items()) if n.startswith('runs/') and n.endswith('/summary.json')]
    costs = [dict(path=n,**json.loads(b)) for n,b in sorted(files.items()) if n.endswith('cost.json') or '/teacher/costs/' in n or '/training-costs/' in n]
    charged_runs = {r['path'].split('/training-costs/')[0] for r in costs if '/training-costs/' in r['path']}
    charged = costs + [r for r in training if r['path'].rsplit('/',1)[0] not in charged_runs]
    target_coverage = [json.loads(b) for n,b in files.items() if n=='targets/coverage.json']
    result['cost'] = dict(training=training,preparation_and_scoring=costs,target_coverage=target_coverage,
        recorded_gpu_hours=sum(r['gpu_hours'] for r in charged if r.get('gpu_hours') is not None),
        complete=False,note='Retained partial/failed attempts and missing timings are not zero; see bulk evidence index.',registered_ceiling_gpu_hours=600)
    atomic(a.out / 'reading.json',result)
    write_csv(a.out / 'per-seed.csv',result['seed_tables'],['arm','seed','panel','parser','initial','final','gain','gain_auc'])
    lines = ['# Package C reading','',f"Goal 1: {result['goal1']['status']}; W={doc['W']}; matched frontier 82; exact bar 69.7.",
        f"Complete registered score grid: {result['complete']}; missing cells: {len(result['missing'])}.",'',
        '| Arm | Seed | Panel | Parser | Initial | Final | Gain | Gain AUC |','|---|---:|---|---|---:|---:|---:|---:|']
    lines += ['| '+' | '.join(str(r[k]) for k in ('arm','seed','panel','parser','initial','final','gain','gain_auc'))+' |' for r in result['seed_tables']]
    (a.out / 'reading.md').write_text('\n'.join(lines)+'\n')
    print('Goal 1:',result['goal1']['status'],'; missing score cells:',len(result['missing']))


def configure(p):
    p.add_argument('--source',type=Path,required=True); p.add_argument('--bootstrap',type=int,default=10000)
if __name__ == '__main__': raise SystemExit(entry(job,configure,'c-read'))
