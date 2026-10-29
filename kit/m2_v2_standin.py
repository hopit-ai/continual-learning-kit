#!/usr/bin/env python3
"""CPU job stand-ins for the transport rehearsal only. No scientific result."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os
from kit import m2_assignment as assignment, m2_snapshot as snapshots
from kit.m2_paths import local


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--action',choices=('prepare','train','score','reduce','collect'),required=True)
    p.add_argument('--seed',type=int,default=101);p.add_argument('--configuration',default='none')
    p.add_argument('--parent',type=Path);p.add_argument('--standin',action='store_true',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.action=='prepare':snapshots.atomic(a.out/'frozen.json',dict(technical_synthetic=True,seeds=[101,102,103]))
    elif a.action=='train':
        assert (a.parent/'frozen.json').exists()
        snapshots.atomic(a.out/f'run-s{a.seed}.json',dict(technical_synthetic=True,seed=a.seed,updates=40,
            model=[1,2,3],optimizer=[4,5],scheduler=40,rng=101,EMA=[6,7],loader=1280,origin=str(a.out)))
    elif a.action=='score':
        proof=assignment.active(a.out,'0',True)
        assert proof['seed']==a.seed
        parent=json.loads((a.parent/f'run-s{a.seed}.json').read_text())
        assert local(parent['origin'])==a.parent
        folder=a.out/'scores'/f'legal-generic-s{a.seed}-k40-test';folder.mkdir(parents=True,exist_ok=True)
        (folder/'scores.csv').write_text('arm,seed,step,panel,question_id,correct\nlegal-generic,'+str(a.seed)+',40,test,q,1\n')
        snapshots.atomic(folder/'m2-identity.json',dict(assignment=proof,model=parent['model']))
        snapshots.atomic(folder/'manifest.json',dict(files={n:assignment.sha(folder/n) for n in ('scores.csv','m2-identity.json')}))
    elif a.action=='reduce':
        status='PENDING' if a.configuration=='pending' else 'COMPLETE'
        snapshots.atomic(a.out/'admission.json',dict(technical_synthetic=True,status=status,
            configuration='generic' if a.configuration=='none' else a.configuration))
    else:
        files={str(p.relative_to(a.parent)):p.read_bytes() for p in a.parent.rglob('*.json')}
        assignment.validate_files(files)
        snapshots.atomic(a.out/'return.json',dict(technical_synthetic=True,verified_files=len(files)))
    return 0


if __name__=='__main__':raise SystemExit(main())
