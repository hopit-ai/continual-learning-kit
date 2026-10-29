#!/usr/bin/env python3
"""Frozen broad-first then earlier-task confirmation; durable node decision before sibling MATH."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit.p3b_read import load
from kit.next_read import analyse, benefit, protects


def decision(rows, nomination, repeats=10000):
    mapped=[]
    for row in rows:
        arm=row['arm']; panel='contract_nli' if row['panel']=='test' else row['panel']
        if arm=='base': mapped.append(dict(row,panel=panel))
        elif arm in [legal_arm(c) for c in CONFIGS]: mapped.append(dict(row,arm='v4-'+arm.removeprefix('legal-'),panel=panel))
    from collections import defaultdict
    cells=defaultdict(set)
    for r in mapped:
        key=(r['arm'],int(r['seed']),int(r['step']),r['panel'])
        require(r['question_id'] not in cells[key],'duplicate confirmation question')
        cells[key].add(r['question_id'])
    expected_ids={'contract_nli':{r['id'] for r in legal_panels()['test']}}
    general=t.read_jsonl(KIT/'panels/general-v1.jsonl')
    expected_ids.update({p:{r['id'] for r in general if r['panel']==p} for p in ('ifeval','knowledge','math')})
    def covered(key):
        ids=cells.get(key,set());panel=key[-1]
        return ids==expected_ids[panel] if panel in expected_ids else len(ids)==(1147 if panel=='finqa' else 100)
    observed=set(cells)
    def complete(config):
        return all(covered((f'v4-{config}',seed,k,p)) for seed in SEEDS
                   for p in ('contract_nli','finqa','ifeval','knowledge','math','hellaswag')
                   for k in (STEPS if p=='contract_nli' else RETENTION_STEPS)) and all(covered(('base',seed,0,'finqa')) for seed in SEEDS)
    pending=dict(status='PENDING',configuration=None,reason='Pending connected finals')
    from kit.m2_format import letter_cells
    pending['format_cells']=letter_cells(rows)
    if not complete('generic'): return pending,dict(status='PENDING',admitted=False)
    result=analyse(mapped,repeats,acquisition='contract_nli')
    comparisons={(c['candidate'],c['control']):c for c in result['v4_comparisons']}
    def passes(candidate,control,primary,other):
        e=comparisons[(candidate,control)]['endpoints']
        return benefit(e[primary]) and all(protects(e[p]) for p in (other,'hellaswag','gain','contract_nli'))
    def defined(candidate,control):
        e=comparisons[candidate,control]['endpoints']
        return all(e[p]['n']==3 and e[p]['mean'] is not None and e[p]['lower'] is not None for p in ('math','survival','hellaswag','gain','contract_nli'))
    if nomination['broad'] is None: return pending,dict(status='PENDING',admitted=False)
    if nomination['broad'] and not complete('broad'): return pending,dict(status='PENDING',admitted=False)
    if nomination['broad'] and not defined('broad','generic'):return pending,dict(status='PENDING',admitted=False)
    broad=bool(nomination['broad'] and passes('broad','generic','math','survival'))
    bdecision=dict(status='COMPLETE',admitted=broad,decision='ADMITTED' if broad else 'INCONCLUSIVE' if nomination['broad'] else 'NOT_NOMINATED')
    if nomination['earlier'] is None: return pending,bdecision
    incumbent='broad' if broad else 'generic'
    earlier=False
    if nomination['earlier']:
        candidate='both' if broad else 'earlier'
        if not complete(candidate): return pending,bdecision
        if not defined(candidate,incumbent):return pending,bdecision
        earlier=passes(candidate,incumbent,'survival','math')
        if candidate=='both': earlier=earlier and all(protects(comparisons[('both','generic')]['endpoints'][p]) for p in ('gain','contract_nli'))
        if earlier: incumbent=candidate
    return dict(status='COMPLETE',configuration=incumbent,broad=broad,earlier=earlier,
                broad_decision=bdecision['decision'],earlier_decision='ADMITTED' if earlier else 'INCONCLUSIVE' if nomination['earlier'] else 'NOT_NOMINATED',
                format_cells=letter_cells(rows),comparisons=result['v4_comparisons'],policy='Return-1 nomination AND connected-final confirmation; broad first, then earlier-task against surviving incumbent.'),bdecision


def job(a):
    doc=binding(a.out,a.standin); rows,files=load(a.out)
    validate_evidence(doc,files,a.standin)
    try:current=nomination(a.out,doc)
    except NotReady:current=dict(broad=None,earlier=None)
    result,broad=decision(rows,current,a.bootstrap)
    common=dict(binding_sha256=s.file_hash(a.out/'binding.json'),technical_synthetic=a.standin)
    for name,value in [('broad-decision.json',broad),('admission.json',result)]:
        value=dict(value,**common)
        path=a.out/name
        if path.exists() and json.loads(path.read_text())['status']=='COMPLETE': bind(path,value)
        else: atomic(path,value)
    print('Replay admission:',result['status'],result.get('configuration'))
    return result


def configure(p): p.add_argument('--bootstrap',type=int,default=10000)
if __name__=='__main__': raise SystemExit(entry(job,configure,'m2-admit'))
