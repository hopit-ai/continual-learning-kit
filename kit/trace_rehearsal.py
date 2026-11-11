#!/usr/bin/env python3
"""Offline CPU stand-ins through every normal allocation row. Zero model imports."""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from kit.trace_common import *


def tiny_native(seed, frozen=False, resume=None, steps=40):
    """Small CPU state oracle for mask, EMA, optimizer and RNG persistence."""
    import copy
    import math
    import random
    rng = random.Random(seed)
    state = dict(student=[.1,.2,.3,.4,.5,.6],teacher=[.1,.2,.3,.4,.5,.6],
                 momentum=[0.]*6,variance=[0.]*6,update=0,exposures=0)
    if resume is not None:
        state = copy.deepcopy(resume)
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value,list) else value
        rng.setstate(tuples(state.pop('rng')))
    original = state['student'][:3]
    for step in range(state['update'],steps):
        features = [rng.uniform(-1,1) for _ in range(6)]
        student = math.fsum(x*y for x,y in zip(state['student'],features))
        teacher = math.fsum(x*y for x,y in zip(state['teacher'],features))+.2
        gradient = 1/(1+math.exp(-student))-1/(1+math.exp(-teacher))
        lr = 1e-5*min(step/10,1)
        for i in range(6):
            if frozen and i<3:
                continue
            g = gradient*features[i]
            state['momentum'][i] = .9*state['momentum'][i]+.1*g
            state['variance'][i] = .999*state['variance'][i]+.001*g*g
            first = state['momentum'][i]/(1-.9**(step+1))
            second = state['variance'][i]/(1-.999**(step+1))
            state['student'][i] *= 1-lr*.01
            state['student'][i] -= lr*first/(math.sqrt(second)+1e-8)
        state['teacher'] = [.99*t+.01*v for t,v in zip(state['teacher'],state['student'])]
        state['update'] = step+1
        state['exposures'] += 32
    need(not frozen or original==state['student'][:3],'synthetic frozen block moved')
    state['rng'] = rng.getstate()
    return state


def standin_row(cfg,row,attempt):
    root = Path(cfg['out'])
    phase = row['phase']
    if phase in ('prepare','targets','bind'):
        value = dict(technical_synthetic=True, config_sha256=digest(cfg),
                     registration_sha256=sha(REG))
        if phase=='prepare':
            scope = cfg.get('studies',['s8',*SOURCES])
            tasks = TASKS if 's8' in scope else SUBSET
            value.update(active_studies=scope,blocked_studies={},
                         licences={study:'CLEARED' for study in scope},
                         counts={task:dict(train=5000,eval=1000,test=2000) for task in tasks},
                         l_eos={task:{split:dict(rows=n,affected=0,suffix_errors=0,
                             eos_position_errors=0,share=0.,original_share=0.,excess_share=0.)
                             for split,n in [('train',5000),('eval',1000),('test',2000)]}
                             for task in tasks} if 's8' in scope else {})
        name = {'prepare':'prepared','targets':'targets','bind':'binding'}[phase]
        seal(root/(name+'.json'),value)
        return value
    if phase=='qualify':
        return dict(technical_synthetic=True,identity=True,leakage=True,
                    freeze_bit_equality=True,scorer_fixtures=True,licences=True,
                    utilisation=31.,estimated_gpu_hours=99999.,warnings=['timing estimate exceeded'])
    if phase=='train':
        study,seed = row['study'],row['seed']
        stages_used = (list(range(1,9)) if study=='s8' else stages(study,seed))
        states_used = [tiny_native(seed,frozen=isinstance(stage,dict) and stage['freeze'])
                       for stage in stages_used]
        seal(root/'training'/f'{study}-{seed}'/'standin.json',
             dict(stages=stages_used,native_states=states_used,technical_synthetic=True))
        return dict(stages=len(stages_used),technical_synthetic=True)
    if phase=='score':
        study,seed = row['study'],row['seed']
        tasks = TASKS if study=='s8' else SUBSET
        panels = ('instruction','maths','knowledge')
        if study=='s8':
            panels += ('MMLU','BBH','GSM8K','TyDiQA','PIQA','BoolQ')
        for state in states(study,seed):
            values = {}
            for task in (*tasks,*panels):
                # Deterministic noninteger scores exercise float transport.
                score = 35+int(digest([study,seed,state['checkpoint'],task])[:8],16)%500000/10000
                values[task] = dict(sum=score*2000,n=2000,mean=score)
            atomic(root/'scores'/f'{study}-{seed}'/state['id']/'result.json',values)
        atomic(root/'scores'/f'{study}-{seed}'/'lane.json',
               dict(states=states(study,seed),complete=True,uuid_sha256=digest('fake-GPU'),
                    environment_sha256=digest('CPU-standin'),attempt=attempt.name,
                    config_sha256=digest(cfg)))
        return standin_row(cfg,dict(row,phase='judge'),attempt)
    if phase=='judge':
        study,seed = row['study'],row['seed']
        values = {s['id']:{panel:[[n//5, n//3, n-n//5-n//3,0],
                                      [n//5+1,n//3-1,n-n//5-n//3,0]]
                  for panel,n in [('Self-Instruct',175),('LIMA',300),('CoNa',178)]}
                  for s in states(study,seed) if s['stage']}
        atomic(root/'judge'/f'{study}-{seed}'/'summary.json',values)
        if study==cfg.get('studies',['s8',*SOURCES])[0] and seed==101:
            atomic(root/'judge'/'calibration-labels.json',[
                dict(id=f'pair-{i:02}',panel=('Self-Instruct','LIMA','CoNa')[i//30],
                     labels=[0,0]) for i in range(90)])
        return dict(technical_synthetic=True)
    from kit.trace_read import collect
    return collect(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    need(not args.out.exists(),'fresh rehearsal directory required')
    args.out.mkdir(parents=True)
    config = args.out/'config.json'
    atomic(config,dict(out=str(args.out.resolve()),technical_synthetic=True,
                       estimated_ceiling_gpu_h=-1))
    from kit.trace_jobs import run
    receipts = [run(config,row['id'],True) for row in inventory()['rows']]
    need(all(r['gpu_hours']==0 for r in receipts),'CPU rehearsal charged GPU hours')
    atomic(args.out/'rehearsal.json',dict(status='TECHNICAL-ONLY',rows=len(receipts),
                gpu_hours=0,downloads=0,comparisons=3,
                trained_cells=84,distinct_checkpoints=78,evaluation_cells=99))
    print(f'TECHNICAL-ONLY {len(receipts)} rows, 0 GPU-h, 0 downloads')


if __name__=='__main__':
    main()
