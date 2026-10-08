"""Outcome-blind plan-v4 arithmetic on per-question records; no summary scores."""
from __future__ import annotations
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import math

from kit.p4_intervals import beta_quantile

PANELS={'chemistry':159,'finqa':1147,'chemistry_probe':51}
GAIN={'chemistry':8,'finqa':58}
LOSS={'chemistry':7,'finqa':57}
SUBSETS={'TOKEN80':(79,'851651880b32c2b666e1e3c74713fdca9bb63130b7cc014416ea9c38382d23d0'),
         'REACTION':(78,'e858cbd1a945c84c9a3f080950695c5dd406f10bb466f8ba5fc7182d3c6814ca'),
         'NONREACTION':(81,'4cfd71d54b945eba5bf4b84566a399447d79b75e6335a49b5fb1e1e08b775ab7'),
         'CLEAN':(42,'c33a937fde3db60fc758de9f21493957e2ca81b5e09e5c290b4e44adc21e2698')}
SELECTION_HISTORY=('Selected after inspecting pilot held-out outcomes on these same question identities; '
                   'frozen before any v4 outcomes; descriptive, with no causal or independent-confirmation interpretation.')


def wilson(k,n):
    """Two-sided 95% Wilson interval beside a proportion, never used for bars."""
    if not 0<=k<=n or n<=0: raise ValueError('invalid binomial counts')
    z=1.959963984540054; p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,center-radius),min(1,center+radius)]


def paired(before,after,ids=None):
    """Exact transitions with a conditional-direction Clopper-Pearson interval."""
    if set(before)!=set(after): raise ValueError('paired question identities differ')
    ids=set(before) if ids is None else set(ids)
    if not ids<=set(before): raise ValueError('missing diagnostic question identities')
    if any(before[i] not in (0,1) or after[i] not in (0,1) for i in ids): raise ValueError('invalid verdict')
    b=sum(before[i]==1 and after[i]==0 for i in ids)
    c=sum(before[i]==0 and after[i]==1 for i in ids)
    n=b+c
    interval=[0.,1.] if not n else [0. if c==0 else beta_quantile(.025,c,b+1),
                                     1. if b==0 else beta_quantile(.975,c+1,b)]
    return {'n':len(ids),'before':sum(before[i] for i in ids),'after':sum(after[i] for i in ids),
            'right_to_wrong':b,'wrong_to_right':c,'net':c-b,
            'conditional_direction':c/n if n else None,'interval':interval,
            'interval_label':'conditional-direction interval',
            'note':None if n else 'no discordant pairs; uninformative interval [0, 1]'}


def panel(records,expected_ids,cap=2048):
    """Count strict/canonical answers and ALL within-cap generated tokens."""
    keys=[str(r['id']) for r in records]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected_ids): raise ValueError('missing, duplicate or foreign scoring identities')
    strict={}; canonical={}; tokens=0
    for row in records:
        key=str(row['id'])
        for field in ('strict','canonical'):
            if type(row.get(field)) is not int or row[field] not in (0,1): raise ValueError('missing per-question '+field)
        strict[key]=row['strict']; canonical[key]=row['canonical']
        count=row.get('tokens')
        if type(count) is not int or count<0 or count>cap: raise ValueError('missing/invalid token records')
        tokens+=count
    correct=sum(strict.values()); n=len(keys)
    return {'n':n,'strict':correct,'canonical':sum(canonical.values()),'tokens':tokens,
            'deployment_cost':tokens/correct if correct else None,
            'wilson':wilson(correct,n),'verdicts':strict,'canonical_verdicts':canonical}


def cost_pass(after,reference):
    """Undefined costs, including a zero baseline denominator, cannot establish a pass."""
    if not after['strict'] or not reference['strict']: return False
    return Fraction(after['tokens'],after['strict'])<=Fraction(3,2)*Fraction(reference['tokens'],reference['strict'])


def cell(task_a,task_b,baseline,stage1,final,*,standalone):
    """One seed/order's fixed conjunction, with comparator-specific transitions."""
    comparisons={
        'first_acquisition':paired(baseline[task_a]['verdicts'],stage1[task_a]['verdicts']),
        'second_gain':paired(stage1[task_b]['verdicts'],final[task_b]['verdicts']),
        'retention':paired(stage1[task_a]['verdicts'],final[task_a]['verdicts']),
        'final_first_acquisition':paired(baseline[task_a]['verdicts'],final[task_a]['verdicts']),
        'final_second_acquisition':paired(baseline[task_b]['verdicts'],final[task_b]['verdicts'])}
    passes={
        'first_acquisition':comparisons['first_acquisition']['net']>=GAIN[task_a],
        'second_gain':comparisons['second_gain']['net']>=GAIN[task_b],
        'retention':-comparisons['retention']['net']<=LOSS[task_a],
        'final_first_acquisition':comparisons['final_first_acquisition']['net']>=GAIN[task_a],
        'final_second_acquisition':comparisons['final_second_acquisition']['net']>=GAIN[task_b],
        'deployment_cost':all(cost_pass(standalone[t],baseline[t]) and cost_pass(final[t],baseline[t]) for t in (task_a,task_b)) and cost_pass(final[task_a],stage1[task_a])}
    return {'comparisons':comparisons,'passes':passes,'met':all(passes.values()),
            'failures':[k for k,v in passes.items() if not v]}


def advantage(d,s):
    """Exact percentage-point denominators and a separate two-order mean per seed."""
    differences={}
    for seed in (101,102,103):
        values=[]
        for order in ('chemistry-finqa','finqa-chemistry'):
            task=order.split('-')[1]; key=f'{seed}:{order}'
            diff=Fraction(100*(d[key]-s[key]),PANELS[task]); values.append(diff)
            differences[key]=float(diff)
        differences[str(seed)+':mean']=float(sum(values)/2)
    # Preserve exact arithmetic for decisions, including the -2.5 boundary.
    met=all(sum(Fraction(100*(d[f'{seed}:{o}']-s[f'{seed}:{o}']),PANELS[o.split('-')[1]]) for o in ('chemistry-finqa','finqa-chemistry'))>=10 for seed in (101,102,103)) and all(Fraction(100*(d[k]-s[k]),PANELS[k.split(':')[1].split('-')[1]])>=Fraction(-5,2) for k in d)
    return {'met':met,'differences_pp':differences}


def matches(d,x,cells):
    """No order cancellation, no vacuous criteria, and the strict 2.5 margin."""
    keys={f'{seed}:{order}' for seed in (101,102,103) for order in ('chemistry-finqa','finqa-chemistry')}
    if set(d)!=keys or set(x)!=keys or set(cells)!=keys: return False
    return all(cells[k]['met'] and Fraction(100*(d[k]-x[k]),PANELS[k.split(':')[1].split('-')[1]])<Fraction(5,2) for k in keys)


def lexical(before,after,subsets,full_pass):
    """Diagnostic arithmetic concentration labels cannot change scientific bars."""
    diagnostics={name:paired(before,after,ids) for name,ids in subsets.items()}
    triggered=[name for name in ('TOKEN80','CLEAN') if diagnostics[name]['net']<=0] if full_pass else []
    return {'subsets':diagnostics,'triggers':triggered,
            'label':('Chemistry gain arithmetically concentrated in lexically solvable questions ('+' / '.join(triggered)+')') if triggered else None,
            'selection_history':SELECTION_HISTORY}


def frozen_subsets(files,primary_ids):
    """Verify the registered bytes and identities, including the 79-member TOKEN80."""
    result={}
    for name,(size,digest) in SUBSETS.items():
        raw=files['v4/report-inputs/v4-data-audit-'+name+'.ids']
        if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('frozen subset hash differs: '+name)
        ids=raw.decode().splitlines()
        if len(ids)!=size or len(set(ids))!=size or ids!=sorted(ids) or not set(ids)<=set(primary_ids): raise ValueError('frozen subset identities differ: '+name)
        result[name]=ids
    return result


def rejection_table(pool,attempts,accepted_ids):
    """Teacher acceptance and FORMAT rejection share separately from wrong answers."""
    groups=defaultdict(list); by_id=defaultdict(list)
    lookup={str(i['id']):i for i in pool}
    if len(lookup)!=len(pool): raise ValueError('duplicate corpus identity')
    for r in attempts:
        key=str(r['id'])
        if key not in lookup: raise ValueError('foreign corpus attempt')
        by_id[key].append(r)
    format_reasons={'missing_or_invalid_option_letter','incomplete_worked_response'}
    def is_format(row):
        detail=row.get('verifier_detail',{})
        # A parsed numeric prediction which cannot be mapped to an option is a
        # wrong answer. The compound verifier reason alone cannot distinguish it
        # from a missing prediction; keep the verifier and its reasons unchanged.
        numeric_missing=('missing_numeric_prediction_or_options' in row['reasons'] and detail.get('prediction') is None)
        return bool(format_reasons & set(row['reasons'])) or numeric_missing or (row.get('task')=='finqa' and detail.get('prediction') is None)
    for key,item in lookup.items():
        rows=by_id[key]
        if not 1<=len(rows)<=4 or [r['attempt'] for r in rows]!=list(range(1,len(rows)+1)): raise ValueError('missing/extra corpus attempt')
        winners=[r['attempt'] for r in rows if r.get('verified') is True]
        if winners and (len(winners)!=1 or winners[0]!=len(rows)): raise ValueError('first acceptable attempt not kept')
        if bool(winners)!=(key in set(accepted_ids)): raise ValueError('accepted corpus/journal mismatch')
        for r in rows:
            if type(r.get('verified')) is not bool or not r['verified'] and not r.get('reasons'): raise ValueError('missing verifier verdict/reason')
        for group in ((item['task'],'ALL'),(item['task'],item.get('family','finqa'))):
            groups[group].append((key,rows))
    result=[]
    for (task,family),questions in sorted(groups.items()):
        rejected=[r for _,rows in questions for r in rows if not r['verified']]
        formats=sum(is_format(r) for r in rejected)
        result.append({'task':task,'family':family,'questions':len(questions),'accepted':sum(key in set(accepted_ids) for key,_ in questions),
                       'rejected_attempts':len(rejected),'format_rejections':formats,
                       'format_share_of_rejections':formats/len(rejected) if rejected else None,
                       'wrong_answer_rejections':sum('incorrect_final_answer' in r['reasons'] and not is_format(r) for r in rejected),
                       'reasons':dict(Counter(reason for r in rejected for reason in r['reasons']))})
    return result
