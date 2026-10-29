#!/usr/bin/env python3
"""Local-only N=1/N=3 plain-job transport rehearsal; stand-ins never authorize execution."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace as NS
import json
import os
import shutil
import tempfile
from unittest.mock import patch
from kit import m2_assignment as assignment, m2_node, m2_snapshot as snapshots


def branch(root,n,configuration):
    nodes=[f'node{i}' for i in range(n)];durable=root/'durable'
    authority=dict(schema='message2-scoring-assignment.v2',assignments={f'core|{s}':dict(
        node=nodes[(s-101)%n],uuid='CPU-STANDIN',environment='CPU-STANDIN') for s in (101,102,103)})
    assignment.freeze(durable/'scoring-assignment.json',authority)
    def task(ident,kind,action,requires=(),**kw):
        command=['{python}','{kit}/m2_v2_standin.py','--out','{out}','--action',action,'--standin']
        if requires:command+=['--parent','{input_'+requires[0]+'}']
        if kw.get('seed'):command+=['--seed',str(kw['seed'])]
        if action=='reduce':command+=['--configuration',configuration]
        return dict(id=ident,kind=kind,requires=list(requires),command=command,**kw)
    tasks=[task('prepare','prepare','prepare')]
    for seed in (101,102,103):
        tasks += [task(f'train-{seed}','train-stage','train',['prepare'],seed=seed),
            task(f'score-{seed}','score-family','score',[f'train-{seed}'],family='core',seed=seed)]
    tasks += [task('M9','reduce-admission','reduce',['score-101','score-102','score-103'],row='M9',commit_barrier='M9'),
        task('collect','collect','collect',['score-101'])]
    manifest=root/'launch.json';snapshots.atomic(manifest,dict(schema='message2-launch.v2',nodes=nodes,tasks=tasks))
    m2_node.freeze_manifest(manifest,durable)
    outcomes=[];requeues=0
    for row in tasks:
        completed=[]
        for i in range(n):
            args=NS(manifest=manifest,node_slot=f'{i}/{n}',task=row['id'],action=row['kind'],
                local_root=root/f'local-node{i}',durable=durable,standin=True)
            try:result=m2_node.run_task(args)
            except snapshots.Pending:
                assert configuration=='pending' and row['id']=='M9'
                result=dict(status='PENDING')
            if result['status']!='NOT_THIS_SLOT':
                outcomes.append(result['status']);completed.append(args)
        assert len(completed)==1
        for args in completed:
            try:again=m2_node.run_task(args)
            except snapshots.Pending:assert configuration=='pending' and row['id']=='M9'
            else:assert again['status']=='SKIP'
            requeues+=1
    if configuration=='pending':
        assert not (durable/'barriers/M9.json').exists()
    else:m2_node.barrier(durable,'M9',assignment.sha(durable/'scoring-assignment.json'))
    # An off-assignment scorer must fail before generating any score or cache.
    path=durable/'scoring-assignment.json'
    with patch.dict(os.environ,M2_ASSIGNMENT=str(path),M2_FAMILY='core',M2_FAMILY_SEED='101',M2_STANDIN_NODE='foreign'):
        try:assignment.active(root,'0',True)
        except assignment.Unverified:pass
        else:raise AssertionError('off-assignment score was accepted')
    return dict(nodes=n,configuration=configuration,jobs=len(tasks),requeues=requeues,
        pending=configuration=='pending',off_assignment_refused=True,local_roots=[str(root/f'local-node{i}') for i in range(n)])


def _run(root):
    if shutil.disk_usage('/').free<6*1024**3:raise RuntimeError('STOP: under 6 GiB free')
    root.mkdir(parents=True,exist_ok=True);results=[]
    for n in (1,3):
        for configuration in ('none','broad','earlier','both','pending'):
            folder=root/f'n{n}-{configuration}';results.append(branch(folder,n,configuration));shutil.rmtree(folder)
    summary=dict(technical_synthetic=True,results=results,gpu_hours=0)
    snapshots.atomic(root/'summary.json',summary);return summary


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--standin',action='store_true',required=True)
    a=p.parse_args();print(json.dumps(run(a.out),indent=2))


def run(root):
    with __import__('kit.m2_snapshot',fromlist=['technical_space_budget']).technical_space_budget(True):return _run(root)
if __name__=='__main__':main()
