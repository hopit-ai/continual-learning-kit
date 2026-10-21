#!/usr/bin/env python3
"""Ordered 4A rows, serial subprocesses; rerun unchanged across chained Slurm jobs."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
from kit.a4_memory import archive
import subprocess


def score_plan():
    """Every registered score cell, including each saved final checkpoint after month six."""
    for p in ('general-v1',):yield ('base','frozen',0,0,p,'off')
    for n,month in enumerate(MONTHS,1):
        for arm,seed,mode in arrival_specs():yield ('arrival',arm,seed,n,month,mode)
        for arm in ARMS:
            if arm=='consolidation' and n not in BOUNDARIES:continue
            for seed in SEEDS:
                for p in ('general-v1',):yield ('incoming',arm,seed,n,p,'off')
                for p in (*MONTHS[:n],'general-v1'):
                    for mode in (('on','off') if arm=='consolidation' and p in MONTHS else ('off',)):
                        yield ('post',arm,seed,n,p,mode)
    for arm in ARMS:
        for seed in SEEDS:
            for n in (range(1,7) if arm=='direct' else BOUNDARIES):
                for p in MONTHS:yield ('final',arm,seed,n,p,'off')


def job(a):
    doc=binding(a.out,a.standin)
    require(not a.fail_once or a.standin,'failure injection is stand-in only')
    require(1<=a.through_month<=6,'invalid month limit')
    def call(script,args,inference=False):
        cmd=[a.inference_python if inference else a.trainer_python,str(KIT/(script+'.py')),'--out',str(a.out)]
        if a.standin:cmd+=['--standin']
        done=subprocess.run(cmd+[str(x) for x in args],check=False)
        if done.returncode==2:raise CorrectnessStop(script+' reported a correctness stop')
        if done.returncode==3:raise NotReady(script+' prerequisites pending')
        if done.returncode:raise subprocess.CalledProcessError(done.returncode,cmd)
    def score(kind,arm,seed,n,p,mode):
        call('a4_score',['--kind',kind,'--arm',arm,'--seed',seed,'--month',n,'--panel',p,'--mode',mode,'--gpu',a.score_gpu],True)
    with lock(a.out/'execution'):
        for p in ('general-v1',):score('base','frozen',0,0,p,'off')
        for n,month in enumerate(MONTHS[:a.through_month],1):
            archive(a.out,doc,month,a.standin)
            for arm,seed,mode in arrival_specs():score('arrival',arm,seed,n,month,mode)
            call('a4_ledger',['--month',month])
            for arm in ARMS:
                if arm=='consolidation' and n not in BOUNDARIES:continue
                for seed in SEEDS:
                    for p in ('general-v1',):score('incoming',arm,seed,n,p,'off')
                    args=['--arm',arm,'--seed',seed,'--month',n,'--gpu',a.gpu,'--score-gpu',a.score_gpu,
                        '--trainer-python',a.trainer_python,'--inference-python',a.inference_python]
                    if a.sdpo:args+=['--sdpo',a.sdpo]
                    if a.tokenizer:args+=['--tokenizer',a.tokenizer]
                    if a.fail_once and arm=='direct' and seed==101 and n==1:args+=['--fail-once']
                    call('a4_train',args)
                    for p in (*MONTHS[:n],'general-v1'):
                        for mode in (('on','off') if arm=='consolidation' and p in MONTHS else ('off',)):
                            score('post',arm,seed,n,p,mode)
            bind(a.out/'months'/(month+'.json'),dict(complete=True,seal=s.file_hash(a.out/'seals'/(month+'.json'))))
        if a.through_month==6:
            for row in score_plan():
                if row[0]=='final':score(*row)
            call('a4_read',[])


def configure(p):
    p.add_argument('--through-month',type=int,default=6);p.add_argument('--gpu',default='0,1,2,3,4,5,6,7')
    p.add_argument('--score-gpu',default='0');p.add_argument('--tokenizer');p.add_argument('--sdpo',type=Path)
    p.add_argument('--trainer-python',default=sys.executable);p.add_argument('--inference-python',default=sys.executable)
    p.add_argument('--fail-once',action='store_true')
if __name__=='__main__':raise SystemExit(entry(job,configure,'a4-run'))
