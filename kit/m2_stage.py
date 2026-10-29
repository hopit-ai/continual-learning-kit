#!/usr/bin/env python3
"""Small explicit row operations for the static manifest. Never submits or waits."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
import shutil


def job(a):
    if a.action=='pool-source':
        from kit.m2_paths import local
        for name in ('next-inputs','replay','hellaswag'):
            source=local(a.source)/name
            require(source.is_dir(),'frozen NEXT input missing '+name)
            if not (a.out/name).exists():shutil.copytree(source,a.out/name)
            frozen_files(a.out/name)
    elif a.action=='nomination':
        source=a.durable/('nomination-broad' if getattr(a,'scope','full')=='broad' else 'nomination')
        if not (source/'COMMIT.json').exists():raise NotReady('fresh nomination pending')
        for origin,target in (('record.json','nomination.json'),('COMMIT.json','nomination-commit.json')):
            from kit.m2_snapshot import immutable
            immutable(a.out/target,(source/origin).read_bytes())
        nomination(a.out,binding(a.out,a.standin))
    elif a.action=='m10':
        from kit.x_bind import m10
        doc=binding(a.out,a.standin)
        if doc['hard_math']!='QUALIFIES' and not (a.out/'logs/m2-run-math-sibling.log').exists():
            logs=list((a.out/'shard-logs').rglob('m2-run-math-sibling.log'))
            if not logs:raise NotReady('M10 registered disposition log pending')
            text='\n'.join(p.read_text() for p in logs)
            require('SKIP: stage 3 '+doc['hard_math'] in text,'M10 disposition log mismatch')
            target=a.out/'logs/m2-run-math-sibling.log';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
        atomic(a.out/'m10-disposition.json',m10(a.out,doc))
    elif a.action=='x-probes':
        from kit.x_common import context,disposition
        from kit.s_probe import job as probe
        from types import SimpleNamespace as NS
        doc=context(a.out,a.standin)
        state=disposition(a.out,'plasticity')
        if state=='PENDING':raise NotReady('X support pending')
        if state!='READY':return
        for domain in doc['domains']:
            probe(NS(**dict(vars(a),incoming=f'b-s{a.seed}',domain=domain)))
    elif a.action=='tier2-measure':
        from kit.x_common import context
        from kit.x_tier2 import identity,readiness
        from kit.x_tier2_run import measure
        from types import SimpleNamespace as NS
        doc=context(a.out,a.standin);spec=identity(a.arm)
        state=readiness(a.out,doc,spec['comparison'])
        if state=='PENDING':raise NotReady('tier-2 technical prerequisites pending')
        if state!='READY':return
        measure(NS(**dict(vars(a),score_gpu=a.gpu)),doc,a.arm,a.seed,a.step,spec['allocation']=='adaptive')
    elif a.action=='x-stage-check':
        from kit.x_common import disposition
        state=disposition(a.out,a.arm)
        if state=='PENDING':raise NotReady('X arm prerequisites pending')
        if state!='READY':return
        if a.arm=='p' and (a.out/'failures'/f'p2-s{a.seed}.json').exists():return
        if a.arm.startswith('t2-'):
            from kit.x_tier2_train import terminal_path
            if terminal_path(a.out,a.arm,a.seed).exists():return
        root=a.out/'runs'/f'x-{a.arm}-s{a.seed}'
        if a.phase=='score':
            expected=[a.out/'scores'/f'x-{a.arm}-s{a.seed}-k{k}-{p}'/'manifest.json' for k in STEPS[1:]
                for p in (PANEL_SETS+('hellaswag',) if k in (20,40) else ('test',))]
        elif a.arm.startswith('t2-') and a.step<40:expected=[root/'segments'/f'{a.step-10}-{a.step}.json']
        else:expected=[root/('prefix.json' if a.step==10 else 'complete.json')]
        if not all(p.exists() for p in expected):raise NotReady('X row output still incomplete')


def configure(p):
    p.add_argument('--scope',choices=('broad','full'),default='full')
    p.add_argument('--action',choices=('pool-source','nomination','m10','x-probes','tier2-measure','x-stage-check'),required=True)
    p.add_argument('--source',type=Path);p.add_argument('--durable',type=Path)
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--arm');p.add_argument('--step',type=int)
    p.add_argument('--phase',choices=('train','score'),default='score')
    p.add_argument('--gpu',default='0');p.add_argument('--score-gpu',default='0');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--trainer-python',default=sys.executable);p.add_argument('--inference-python',default=sys.executable)
if __name__=='__main__':raise SystemExit(entry(job,configure,'stage-{action}'))
