"""Failure-isolated serial tier-2 orchestration; no admission decisions on node."""
from kit.x_common import *
from kit.m2_gpu import prepared_index
from kit.x_tier2 import COMPARISONS, readiness, tier1_return
from kit.x_tier2_slots import ema
from types import SimpleNamespace as NS
import subprocess
import time
import shutil


def measure(a,doc,arm,seed,step,adaptive):
    from kit.s_generate import selected
    from kit.s_drift import nll
    from kit.m2_handoff import scoring_identity
    scoring_machine=scoring_identity(a.out,a.score_gpu,a.standin)
    os.environ['CUDA_VISIBLE_DEVICES']=prepared_index(a.score_gpu,a.standin)
    target=a.out/'tier2/signals'/arm/f's{seed}'/f'k{step}.json'
    refs=[incoming(doc,seed),checkpoint(a.out,doc,arm,seed,step)]
    monitors={d:selected(a.out,f'b-s{seed}',d)['monitor'] for d in doc['domains']}
    frozen=dict(assignment=(scoring_machine or {}).get('assignment'),checkpoints=[m.checked_model(r,a.standin) for r in refs],targets={d:digest(v) for d,v in monitors.items()},
        baseline='stage-start',step=step,ema_coefficient=.5 if adaptive else None)
    bind(target.with_suffix('.inputs.json'),frozen)
    if target.exists():
        require(read(target)['inputs']==frozen,'tier-2 drift inputs changed')
        return
    began=time.monotonic()
    if step==10:
        baseline={d:nll(refs[0]['model'],values,a.standin,refs[0].get('receipt')) for d,values in monitors.items()}
    else:
        first=read(target.parent/'k10.json')
        require(first['inputs']['checkpoints'][0]==frozen['checkpoints'][0] and
            first['inputs']['targets']==frozen['targets'],'bound k10 baseline identity changed')
        baseline={d:first['measurements'][d][0] for d in monitors}
    measurements={d:[baseline[d],nll(refs[1]['model'],values,a.standin,refs[1].get('receipt'))] for d,values in monitors.items()}
    raw={d:v[1]['token_mean']-v[0]['token_mean'] for d,v in measurements.items()}
    previous=read(target.parent/f'k{step-10}.json')['allocation_signal'] if adaptive and step>10 else None
    signal=ema(previous,raw) if adaptive else raw
    bind(target,dict(inputs=frozen,measurements=measurements,raw_drift=raw,allocation_signal=signal,
        seconds=time.monotonic()-began,gpu_hours=0 if a.standin else (time.monotonic()-began)/3600))


def comparison(a,doc,name):
    require((a.out/'tier2/plan.json').exists(),'X14 plan pending')
    tier1_return(a.out)
    status=readiness(a.out,doc,name)
    if status != 'READY':
        atomic(a.out/'tier2'/name/'disposition.json',dict(status=status))
        print('SKIP tier 2:',name,status)
        return
    require(shutil.disk_usage(a.out).free >= 12*1024**3,'disk below 12 GiB; stop')
    def run(script,*args,python=None):
        cmd=[python or a.inference_python,str(ROOT/'kit'/script),'--out',str(a.out),*map(str,args)]
        if a.standin:
            cmd.append('--standin')
        subprocess.run(cmd,check=True)
    def score(arm,seed,steps):
        if getattr(a,'mode','all')=='train':return
        for step in steps:
            for panel in ((*m.PANEL_SETS,'hellaswag') if step in (0,20,40) else ('test',)):
                args=['--arm',arm,'--seed',seed,'--step',step,'--panel-set',panel,'--gpu',a.score_gpu]
                if a.panel_root:
                    args+=['--panel-root',a.panel_root]
                run('x_score.py',*args)
    def train(arm,seed,phase,end):
        if getattr(a,'mode','all')=='score':return
        args=['--arm',arm,'--seed',seed,'--phase',phase,'--end',end,'--gpu',a.gpus,'--trainer-python',a.trainer_python]
        if a.sdpo:
            args+=['--sdpo',a.sdpo]
        if a.tokenizer:
            args+=['--tokenizer',a.tokenizer]
        run('x_train.py',*args,python=a.trainer_python)
    spec=COMPARISONS[name]
    if spec['signal'] in ('plasticity','both'):
        # These are independent incoming-checkpoint measurements, never training
        # controls. S's probe function verifies and reuses its immutable receipts.
        try:
            run('x_signal.py','--action','plasticity','--score-gpu',a.score_gpu,'--gpus',a.gpus,
                '--trainer-python',a.trainer_python,'--inference-python',a.inference_python)
        except Exception as exc:
            atomic(a.out/'tier2'/name/'failure.json',dict(status='PENDING',reason=str(exc)))
            return
    for side in ('control','challenger'):
        arm='t2-'+name+'-'+side
        for seed in SEEDS:
            from kit.m2_select import selected
            if not selected(a,stage=name,arm=arm,seed=seed):continue
            from kit.x_tier2_train import terminal_path
            if terminal_path(a.out,arm,seed).exists():
                print('SKIP terminal tier-2 arm:',arm,seed);continue
            if shutil.disk_usage(a.out).free < 12*1024**3:
                raise SystemExit('STOP: disk below 12 GiB')
            try:
                score(arm,seed,(0,))
                allocation=spec[side]
                if allocation=='standalone':
                    train(arm,seed,'full',40)
                else:
                    train(arm,seed,'prefix',10)
                    if allocation=='adaptive':
                        for start in (10,20,30):
                            measure(a,doc,arm,seed,start,True)
                            train(arm,seed,'continue',start+10)
                        # Final refresh is retained for the registered ten-step
                        # measurement cadence; no updates remain to allocate.
                        measure(a,doc,arm,seed,40,True)
                    else:
                        if allocation=='drift':
                            measure(a,doc,arm,seed,10,False)
                        train(arm,seed,'continue',40)
                score(arm,seed,STEPS[1:])
            except Exception as exc:
                atomic(a.out/'tier2'/name/f'failure-{side}-s{seed}.json',dict(
                    status='DEFERRED' if terminal_path(a.out,arm,seed).exists() else 'PENDING',arm=arm,seed=seed,reason=str(exc)))
                print('Tier-2 failure retained; other arms continue:',arm,seed,exc)
    print('Tier-2 execution finished:',name,'; manager binding pending')
