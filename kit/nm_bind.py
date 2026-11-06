#!/usr/bin/env python3
"""Resolve frozen dispatch from bound *existing-reader* verdicts, without outcomes.

Each evidence reference has path, sha256, and a selector (list of JSON keys).
The selector names an existing verdict; no replacement K/CR/P2 reader is run here.
Missing files/keys and unresolved evidence are PENDING, never non-admission.
"""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.nm_common import *

RESOLVED={'PASS':True,'ADMITTED':True,'INCONCLUSIVE':False,'NOT-ADMITTED':False,'NOT_ADMITTED':False}

def selected(doc,keys,name):
    value=doc
    for key in keys:
        ready(isinstance(value,dict) and not value.get('evidence_error'),name+' existing reader reports invalid evidence')
        ready(key in value,name+' verdict unresolved');value=value[key]
    return value


def verdict(ref,name):
    ready(ref is not None,name+' verdict absent')
    doc=checked(ref);value=selected(doc,ref.get('selector',[]),name)
    if ref.get('complete_selector') is not None:
        state=selected(doc,ref['complete_selector'],name+' completeness')
        ready(state==ref.get('complete_value','COMPLETE'),name+' existing reader incomplete/invalid')
    if type(value) is bool:
        ready(ref.get('complete_selector') is not None,name+' Boolean requires existing-reader completeness')
        return value
    if isinstance(value,dict):
        ready(not value.get('evidence_error') and not value.get('identity_failures'),name+' invalid evidence')
        if ref.get('adapter')=='x-admitted':
            # Apply the existing pure reader unchanged, including its own bars.
            from kit.x_read import admitted
            ready(value.get('status')=='COMPLETE',name+' existing X comparison incomplete')
            result=admitted(value);ready(result is not None,name+' existing X verdict unresolved');return result
        if value.get('status')=='COMPLETE' and type(value.get('admitted')) is bool:return value['admitted']
        value=value.get('status')
    if value=='NOT_NOMINATED':
        ready(ref.get('complete_selector') is not None,name+' nomination disposition needs complete evidence')
        return False
    ready(value in RESOLVED,name+' evidence unresolved/invalid: '+str(value))
    return RESOLVED[value]


def recipe(value,name):
    ready(value is not None,name+' recipe absent')
    doc=checked(value)
    need(doc.get('base_revision')==BASE,name+' wrong original base')
    need(doc.get('name')==name,'recipe name differs')
    need(set(doc.get('starts',{}))==set(map(str,SEEDS)),'three exact FinQA parents required')
    need(all(v.get('seed')==int(s) and v.get('stage')=='finqa' and v.get('step')==40 and v.get('sha256')
             for s,v in doc['starts'].items()),'wrong parent ancestry')
    need(doc.get('updates')==40 and doc.get('denominator')==32,'registered update/denominator mismatch')
    ready(set(doc.get('schedule_hashes',{}))==set(map(str,SEEDS)) and doc.get('caps') and doc.get('pool_ids') is not None,'recipe execution binding incomplete')
    ready(all(doc['caps'].get(k)==v for k,v in dict(prompt=10240,teacher=14336,response=2048).items()),'returned legal caps differ; prospective repricing required')
    need(type(doc.get('task_rows')) is int and type(doc.get('replay_rows')) is int and doc['task_rows']>0 and doc['replay_rows']>=0,'recipe row counts absent')
    need(doc['task_rows']+doc['replay_rows']==(48 if name=='CR' else 32),'admitted row geometry differs')
    need(type(doc.get('finqa_slots',0)) is int and 0<=doc.get('finqa_slots',0)<=doc['replay_rows'],'earlier-task slots exceed replay dose')
    ready(not doc['replay_rows'] or doc.get('pool_ids'),'actual replay pool IDs absent')
    ready(doc.get('draw_policy') in ('inherited-core','unfiltered'),'admitted replay draw policy absent')
    if name in ('K','CR'):need(doc['draw_policy']=='unfiltered','K/CR must retain unfiltered draws')
    if name=='K':need(doc.get('task_rows')==28 and doc.get('replay_rows')==4 and not doc.get('finqa_slots'),'K geometry differs; no K+FinQA')
    if name=='CR':need(doc.get('task_rows')==32 and doc.get('replay_rows')==16 and doc.get('finqa_slots')==8,'CR geometry differs')
    return doc

def resolve(spec):
    """Order is K → CR disposition → applicable P2/stacking → freeze I."""
    evidence=spec.get('evidence',{});trace=[]
    k=verdict(evidence.get('K'),'v6a K');trace.append(['K',k])
    cr=verdict(evidence.get('CR'),'v6a CR');trace.append(['CR',cr])
    selected='K' if k else 'CR' if cr else 'I0'
    p2=False
    if selected=='I0':
        for name in ('M8','M9','X'):
            result=verdict(evidence.get(name),'v6b '+name);trace.append([name,result])
        p2=verdict(evidence.get('P2'),'v6b P2');trace.append(['P2',p2])
        # Required stacking is declared in the sealed launch, never inferred from new scores.
        for name in spec.get('required_stacking',[]):trace.append([name,verdict(evidence.get(name),'v6b '+name)])
        admitted=[name for name,passed in trace if passed]
        rec=recipe(spec.get('recipes',{}).get('I0'),'I0')
        need(set(rec.get('admitted_components',[]))<=set(admitted)-{'M8'},'I0 contains rejected or merely nominated component')
        need(not rec.get('finqa_slots') or dict(trace).get('M9') is True,'I0 contains unconfirmed FinQA pool')
        if p2 and rec.get('p2_compatible') and all(dict(trace).get(name) for name in spec.get('required_stacking',[])):
            ready(rec.get('p2_lambda') is not None,'P2 coefficient absent')
        else:need(rec.get('p2_lambda') is None,'incompatible/unadmitted P2 carried')
    else:
        trace.extend([['P2','INAPPLICABLE'],['core-stacking','INAPPLICABLE']])
        rec=recipe(spec.get('recipes',{}).get(selected),selected)
        need(rec.get('p2_lambda') is None,'K/CR cannot carry core P2')
    return dict(schema=SCHEMA,selected=selected,recipe=rec,trace=trace,CR_admitted=cr,
        P2_carried=rec.get('p2_lambda') is not None,spec_sha256=digest(spec),
        precedence=['214dbe6f','7efbb6cb'],evidence=evidence)

def freeze_incumbent(path,bound,spec,family,*,later_names=()):
    """One dispatch-wide selection, sealed before any family-specific gates.

    K/CR dispositions and the selected recipe cannot change across staggered
    binds. Only the later P2/nomination prerequisites of Fisher/OPR may be added
    off I0; I0 already requires its complete v6b evidence before freezing.
    """
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    selected=bound['selected'];evidence=spec.get('evidence',{})
    names=['K','CR']+(['M8','M9','X','P2',*spec.get('required_stacking',[])] if selected=='I0' else [])
    identity=dict(selected=selected,recipe_sha256=digest(bound['recipe']),recipe=bound['recipe'],
        recipe_ref=spec.get('recipes',{}).get(selected),trace=bound['trace'],
        evidence={n:evidence[n] for n in names},required_stacking=spec.get('required_stacking',[]) if selected=='I0' else [])
    allowed={'fisher':{'P2'},'opr':{'nomination'}}.get(family,set())
    need(set(later_names)<=allowed,'unneeded later evidence cannot change dispatch')
    extra={n:evidence[n] for n in later_names if n in evidence and n not in names}
    with path.with_suffix('.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        if path.exists():
            previous=load(path);need(previous['identity']==identity,'dispatch incumbent/evidence changed')
            known=previous.get('later_evidence',{})
            need(all(n not in evidence or evidence[n]==v for n,v in known.items()),'dispatch later evidence changed')
            need(set(extra)-set(known)<=allowed,'unneeded later evidence cannot change dispatch')
            if set(extra)-set(known):atomic(path,dict(previous,later_evidence={**known,**extra}))
        else:seal(path,dict(schema=SCHEMA,identity=identity,later_evidence=extra))


def row_binding(family,spec,*,incumbent_path=None):
    need(family in FAMILIES,'unknown family')
    if family in ('v2b','grpo'):
        return dict(schema=SCHEMA,family=family,status='READY',incumbent='INAPPLICABLE',benchmark=True,spec_sha256=digest(spec))
    bound=resolve(spec);selected=bound['selected'];rec=bound['recipe'];benchmark=False
    if incumbent_path is not None:freeze_incumbent(incumbent_path,bound,spec,family)
    if family=='fisher':
        p2=verdict(spec.get('evidence',{}).get('P2'),'v6b P2')
        if incumbent_path is not None:freeze_incumbent(incumbent_path,bound,spec,family,later_names=['P2'])
        benchmark=selected!='I0' or not (p2 and bound['P2_carried'])
        bound['fisher_calibration']='reuse-compatible-P2' if not benchmark else 'new recipe calibration'
    if family=='opr':
        if rec.get('finqa_slots',0):pass
        elif bound['CR_admitted']:
            rec=recipe(spec.get('recipes',{}).get('CR'),'CR');benchmark=True
        else:
            # Nomination-only precedence is fixed: CR before nominated core.
            nominated=verdict(spec.get('evidence',{}).get('nomination'),'v6b earlier-task nomination')
            if incumbent_path is not None:freeze_incumbent(incumbent_path,bound,spec,family,later_names=['nomination'])
            if not nominated:return dict(schema=SCHEMA,family=family,status='ARCHIVED UNREAD',reason='no CR or earlier-task nomination',binding=bound)
            rec=recipe(spec.get('recipes',{}).get('nominated'),'NOMINATED');benchmark=True
        ready(rec.get('finqa_slots',0)>0,'OPR requires bound earlier-task slots')
    if family=='isdft':
        ready(spec.get('persistence'),'persistence evidence absent')
        persistence=checked(spec['persistence'])
        ready(persistence.get('recipe_sha256')==digest(bound['recipe']),'persistence wrong recipe')
        need(persistence.get('panel')=='instruction' and persistence.get('step')==40,'persistence wrong boundary/panel')
        values=persistence.get('retention');ready(values is not None,'persistence retention absent');summary=stats(values,ONE)
        if summary['mean']>=-3:return dict(schema=SCHEMA,family=family,status='ARCHIVED UNREAD',reason='instruction deficit repaired',persistence=summary,binding=bound)
        bound['persistence']=summary
    return dict(schema=SCHEMA,family=family,status='READY',incumbent=selected,recipe=rec,benchmark=benchmark,binding=bound)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);p.add_argument('--out',required=True);p.add_argument('--family',choices=FAMILIES);p.add_argument('--incumbent',help='dispatch-wide out/incumbent.json (defaults beside binding)')
    a=p.parse_args()
    try:
        result=row_binding(a.family,load(a.spec),incumbent_path=a.incumbent or Path(a.out).parent/'incumbent.json') if a.family else dict(status='READY',**resolve(load(a.spec)))
        seal(a.out,result);print(result['status']);return 0
    except Pending as error:
        print('PENDING:',error);return PENDING
    except Invalid as error:
        print('UNVERIFIED:',error);return 2
if __name__=='__main__':raise SystemExit(main())
