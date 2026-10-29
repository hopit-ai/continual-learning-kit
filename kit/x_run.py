#!/usr/bin/env python3
"""Serial plain-script row helpers; sbatch/process groups remain owned by the shell."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *
import subprocess


def job(a):
    doc = context(a.out, a.standin)
    from kit.m2_select import selected
    arm={'controls':'control','p':'p','drift-prefix':'drift','drift-continue':'drift','plasticity':'plasticity'}.get(a.action,a.action)
    seeds=[s for s in SEEDS if selected(a,stage=a.action,arm=arm,seed=s)]
    mode=getattr(a,'mode','all')
    def run(script, *args, python=None):
        cmd = [python or a.inference_python, str(ROOT/'kit'/script), '--out', str(a.out), *map(str,args)]
        if a.standin: cmd.append('--standin')
        subprocess.run(cmd, check=True)
    def scores(arm, seed, steps):
        if mode=='train':return
        for step in steps:
            panels = (*m.PANEL_SETS, 'hellaswag') if step in (0,20,40) else ('test',)
            for panel in panels:
                args = ['--arm',arm,'--seed',seed,'--step',step,'--panel-set',panel,'--gpu',a.score_gpu]
                if a.panel_root: args += ['--panel-root', a.panel_root]
                run('x_score.py', *args)
    def train(arm, seed, phase='full'):
        if mode=='score':return
        args = ['--arm',arm,'--seed',seed,'--phase',phase,'--gpu',a.gpus,'--trainer-python',a.trainer_python]
        if a.sdpo: args += ['--sdpo',a.sdpo]
        if a.tokenizer: args += ['--tokenizer',a.tokenizer]
        run('x_train.py', *args, python=a.trainer_python)
    if a.action == 'measure':
        if doc.get('p_variant')=='disabled':print('SKIP P2: disabled');return
        try:
            require(a.smoke is not None,'P2 smoke receipt required')
            args=['--action','measure','--smoke',a.smoke,'--gpus',a.gpus,'--trainer-python',a.trainer_python]
            if getattr(a,'seed',None) is not None:args+=['--seed',a.seed]
            if a.sdpo:args+=['--sdpo',a.sdpo]
            if a.tokenizer:args+=['--tokenizer',a.tokenizer]
            run('x_p.py',*args,python=a.trainer_python)
        except (ValueError,OSError,subprocess.CalledProcessError) as error:
            if isinstance(error,subprocess.CalledProcessError) and error.returncode in (2,3):raise
            name=f'p-s{a.seed}.json' if getattr(a,'seed',None) is not None else 'p.json'
            atomic(a.out/'dispositions'/name,dict(status='DEFERRED',reason='X4 failed: '+str(error)))
            print('P2 deferred; controls/A continue:',error)
    elif a.action == 'controls':
        for seed in seeds:
            scores('pre',seed,(0,)); scores('control',seed,(0,))
            train('control',seed); scores('control',seed,STEPS[1:])
    elif a.action == 'p':
        if doc.get('p_variant')!='p2-l2-init':print('SKIP P2: disabled');return
        from kit.x_p import calibration
        try:calibration(a.out,doc)
        except (ValueError,KeyError,OSError,TypeError) as error:
            atomic(a.out/'dispositions/p.json',dict(status='DEFERRED',reason='P2 calibration unavailable: '+str(error)))
            print('SKIP P2: calibration unavailable');return
        for seed in seeds:
            try:
                scores('p',seed,(0,))
                train('p',seed); scores('p',seed,STEPS[1:])
            except (ValueError,OSError,subprocess.CalledProcessError) as error:
                if isinstance(error,subprocess.CalledProcessError) and error.returncode in (2,3):raise
                atomic(a.out/'failures'/f'p2-s{seed}.json',dict(status='INCONCLUSIVE',seed=seed,reason=str(error)))
                print('P2 seed failure retained; other seeds/A continue:',seed,error)
    elif a.action == 'drift-prefix':
        for seed in seeds:
            if disposition(a.out,'drift') == 'READY': scores('drift',seed,(0,))
            train('drift',seed,'prefix')
    elif a.action == 'drift-continue':
        if mode!='train':run('x_signal.py','--action','drift','--score-gpu',a.score_gpu)
        for seed in seeds:
            train('drift',seed,'continue'); scores('drift',seed,STEPS[1:])
    elif a.action == 'plasticity':
        if mode!='train':run('x_signal.py','--action','plasticity','--score-gpu',a.score_gpu,'--gpus',a.gpus,
            '--trainer-python',a.trainer_python,'--inference-python',a.inference_python)
        for seed in seeds:
            if disposition(a.out,'plasticity') == 'READY': scores('plasticity',seed,(0,))
            train('plasticity',seed); scores('plasticity',seed,STEPS[1:])


def configure(p):
    from kit.m2_select import configure as selectors
    selectors(p)
    p.add_argument('--action', choices=('measure','controls','p','drift-prefix','drift-continue','plasticity'), required=True)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7'); p.add_argument('--score-gpu', default='0')
    p.add_argument('--sdpo', type=Path); p.add_argument('--tokenizer'); p.add_argument('--panel-root', type=Path)
    p.add_argument('--smoke', type=Path)
    p.add_argument('--trainer-python', default=sys.executable); p.add_argument('--inference-python', default=sys.executable)
if __name__ == '__main__': raise SystemExit(entry(job, configure))
