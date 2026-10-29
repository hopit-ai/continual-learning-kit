#!/usr/bin/env python3
"""Prospectively bind reused C targets, replay pools, and all evaluation exclusions."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit.m2_score import items_for


def job(a):
    doc=binding(a.out,a.standin)
    pool_free=getattr(a,'pool_free',False)
    roots=dict(c_targets=a.c_targets) if pool_free else dict(c_targets=a.c_targets,replay=a.replay,pools=a.pool_root)
    manifests={}
    for key,root in roots.items():
        freeze=frozen_files(root)
        require(freeze.get('technical_synthetic') is a.standin, 'synthetic reused input mismatch: '+key)
        if key == 'pools':
            from kit.next_pools import POOL_PROTOCOL
            require(freeze.get('pool_protocol') == POOL_PROTOCOL,
                    'M3b: regenerate broad pool under fixed ARC prompt before training')
            if not a.standin:
                from kit.arc_prompt_acceptance import validate_candidate_membership
                validate_candidate_membership(root/'candidates.jsonl',doc['arc_acceptance'])
        manifests[key]=dict(path=str(root.resolve()),manifest=s.file_hash(root/'manifest.json'))
    evaluation=[]
    for panel in PANEL_SETS:
        for row in items_for(panel,a.standin,a.panel_root):
            evaluation.append(dict(row,prompt=row['prompt']))
    freeze=frozen_files(a.hellaswag)
    require(freeze.get('technical_synthetic') is a.standin,'synthetic HellaSwag mismatch')
    import shutil
    if not (a.out/'hellaswag').exists(): shutil.copytree(a.hellaswag,a.out/'hellaswag')
    check(a.out/'hellaswag/manifest.json',s.file_hash(a.hellaswag/'manifest.json'))
    evaluation += t.read_jsonl(a.hellaswag/'panel.jsonl')
    report={}
    pools={'legal':t.read_jsonl(a.c_targets/'pool.jsonl')}
    if not pool_free:
        pools['generic']=t.read_jsonl(a.replay/'replay.jsonl')
        pools.update({name:t.read_jsonl(a.pool_root/(name+'.jsonl')) for name in ('earlier','gsm8k','arc')})
    for name,rows in pools.items():
        _,excluded=overlap(rows,evaluation)
        report[name]=dict(rows=len(rows),exclusions=excluded,digest=s.digest(rows))
        require(not excluded,'frozen '+name+' pool overlaps continuation evaluation; report, never replace it')
    value=dict(inputs=manifests,overlap=report,evaluation=s.digest(evaluation),technical_synthetic=a.standin)
    previous=a.out/'assets.json'
    if previous.exists() and not pool_free and 'pools' not in json.loads(previous.read_text())['inputs']:
        require(json.loads(previous.read_text())['inputs']['c_targets']==manifests['c_targets'],'pool-free target identity changed')
        atomic(previous,value)
    else:bind(previous,value)
    print('Prepared reused targets/pools; all three task and general-v1 exclusions checked')


def configure(p):
    p.add_argument('--pool-free',action='store_true')
    p.add_argument('--c-targets',type=Path,required=True);p.add_argument('--replay',type=Path)
    p.add_argument('--pool-root',type=Path);p.add_argument('--panel-root',type=Path);p.add_argument('--hellaswag',type=Path,required=True)
if __name__=='__main__': raise SystemExit(entry(job,configure,'m2-prepare'))
