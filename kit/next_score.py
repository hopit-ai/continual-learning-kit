#!/usr/bin/env python3
"""One v4 checkpoint × panel, or the on-node shared base control."""
from pathlib import Path
import sys
if __package__ in (None, ''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.next_common import *


def job(a):
    frozen_next();machine(a.out,a.gpu,a.standin);inputs=read_inputs(a.out,a.standin)
    require(a.arm=='base' or a.seed in SEEDS,'unregistered seed')
    require(a.step in ((0,) if a.arm=='base' else (0,10,20,30,40)),'unregistered step')
    require(a.panel_set=='math_screen' or a.step in (0,20,40),'unregistered scoring schedule')
    arm='base' if a.arm=='base' else 'v4-'+a.arm
    if a.arm=='base':parent=inputs['base']
    elif a.step==0:parent=inputs['probe_starts'][SEEDS.index(a.seed)]
    else:
        root=a.out/'runs'/f'{arm}-s{a.seed}'
        frozen_files(root)
        parent=dict(path=str(root/f'hf-step{a.step}'),receipt=str(root/f'export-{a.step}.json'))
    if a.step==0:
        shared_score(a,parent,arm,a.seed,0,a.panel_set,run);return
    if a.standin and a.step>0:
        # B stand-ins emit receipts, not weights; use the fixture start as the stand-in model.
        export=json.loads(Path(parent['receipt']).read_text())
        require(export.get('technical_synthetic') is True and export['step']==a.step,'wrong stand-in export')
        parent=inputs['probe_starts'][SEEDS.index(a.seed)]
    if a.panel_set=='hellaswag':
        command=[sys.executable,KIT/'next_hellaswag.py','--action','score','--out',a.out,'--arm',arm,
                 '--seed',a.seed,'--step',a.step,'--model',parent['path'],'--gpu',a.gpu]
    else:
        command=[sys.executable,KIT/'p3b_score.py','--out',a.out,'--arm',arm,'--seed',a.seed,'--step',a.step,
                 '--model',parent['path'],'--panel-set',a.panel_set,'--gpu',a.gpu]
        if a.package_a:command+=['--package-a',a.package_a]
        if a.finqa:command+=['--panel-root',a.finqa]
    if parent.get('receipt'):command+=['--checkpoint-receipt',parent['receipt']]
    if a.standin:command+=['--standin']
    run(command)


def configure(p):
    p.add_argument('--arm',choices=('base','generic','broad','earlier','both'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--panel-set',choices=('finqa','general-v1','math_screen','hellaswag'),required=True)
    p.add_argument('--gpu',default='0');p.add_argument('--package-a',type=Path);p.add_argument('--finqa',type=Path)
if __name__ == '__main__':raise SystemExit(entry(job,configure,'next-score-{arm}-{seed}-{step}-{panel_set}'))
