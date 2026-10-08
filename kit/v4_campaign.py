#!/usr/bin/env python3
"""Write the v4 qualification/main manifests for kit/runner.py; no dispatch here.

The file generator is outcome-blind. Each scientific standalone is declared once;
chain MODEL_DIRs refer to that export. All GPU argv are contained Slurm rows.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

TASKS=('chemistry','finqa')
ARMS=('S','F','R','D')
SEEDS=(101,102,103)
CAPS=(2048,8192)


def slots():
    """Fixed 48-slot order, with D seed-101 standalone checkpoints first."""
    result=[]
    for task in TASKS:
        result.append({'id':f'D-{task}-alone-s101','arm':'D','seed':101,'task':task,'incoming':None,'order':task+'-alone'})
    for seed in SEEDS:
        for arm in ARMS:
            for task in TASKS:
                ident=f'{arm}-{task}-alone-s{seed}'
                if ident not in {r['id'] for r in result}:
                    result.append({'id':ident,'arm':arm,'seed':seed,'task':task,'incoming':None,'order':task+'-alone'})
            for first,second in (TASKS,TASKS[::-1]):
                result.append({'id':f'{arm}-{first}-{second}-s{seed}','arm':arm,'seed':seed,'task':second,
                               'incoming':f'{arm}-{first}-alone-s{seed}','order':first+'-'+second})
    return result


def containment(argv,*,phase,gpus=1,cap=900,concurrent=False):
    """Wrap GPU work with the final reservation and Slurm-assignment interface."""
    from kit.p4_contain import MIN_ROW_CAP
    if cap < MIN_ROW_CAP: raise ValueError('row cap below containment minimum')
    if concurrent and gpus != 1: raise ValueError('only one-GPU rows may be concurrent')
    block='qualification' if phase in ('qualification','rewrite') else 'scientific'
    command=['python','{kit}/p4_contain.py','row','--work','{work}',
             '--out','{work}/v4/containment/{campaign}/{row}/attempt-{attempt}','--gpus',str(gpus),'--time-cap',str(cap),
             '--block',block,'--block-limit','100' if block=='qualification' else '560','--ceiling','560']
    if concurrent: command.append('--concurrent')
    return command+['--']+argv


def build(phase):
    """Build a strict dependency graph, preserving pilot gates and one scoring GPU."""
    if phase=='phase0':
        from kit.v4_phase0 import campaign
        return campaign()
    if phase not in ('qualification','main'): raise ValueError('unknown campaign phase')
    rows=[];training_order=False
    def add(ident,argv,*,gpu=0,cap=900,pilot=False,needs=None,env=None,concurrent=False):
        # Preparation keeps its technical gates. Training slots use true lineage
        # dependencies; predecessor ordering cannot block independent slots.
        required=([] if not rows or training_order else [rows[-1]['id']]) if needs is None else list(needs)
        required += [r['id'] for r in rows if r.get('pilot')][-1:]
        row={'id':ident,'needs':list(dict.fromkeys(required)),
             'command':containment(argv,phase='rewrite' if ident.startswith('rewrite-') else phase,gpus=gpu,cap=cap,concurrent=concurrent) if gpu else argv}
        if training_order and rows:
            row.update(wants=[rows[-1]['id']],order=[rows[-1]['id']])
        row['env']={'V4_TELEMETRY':'1', **(env or {})}
        row['timeout_seconds']=cap if not gpu else cap+120
        if not gpu:row['allocation_cpu_cap_seconds']=cap
        if pilot:
            row['pilot']=True
            row['bars']=[{'source':'{work}/v4/report-status/'+ident+'.json','key':'ok','min':1}]
        rows.append(row)
        return row
    def action(ident,operation,*arguments,gpu=0,cap=900,pilot=False,env=None,needs=None,concurrent=False):
        return add(ident,['python','{kit}/v4_campaign_ops.py',operation,'--work','{work}',
                          '--phase',phase,'--row',ident,*arguments],gpu=gpu,cap=cap,pilot=pilot,env=env,needs=needs,concurrent=concurrent)
    add('containment-selftest',['python','{kit}/p4_contain.py','selftest','--work','{work}',
                               '--out','{work}/k8b4/containment','--block','qualification' if phase=='qualification' else 'scientific',
                               '--block-limit','100' if phase=='qualification' else '560','--ceiling','560'])
    action('environment','environment',pilot=True)
    action('prepare','verify-prepare',pilot=True)
    if phase=='main': action('owner-registration-gate','gate',pilot=True)
    if phase=='qualification':
        action('runner-profile','runner-profile',cap=900,pilot=True)
        for task in (*TASKS,'chemistry_probe'):
            for cap in CAPS:
                action(f'baseline-{task}-{cap}','score','--checkpoint','baseline','--task',task,'--cap',str(cap),gpu=1,cap=3600)
        for task in TASKS:
            for repeat in range(1,5):
                action(f'score-agreement{repeat}-{task}-2048','score','--checkpoint','agreement'+str(repeat),
                       '--task',task,'--cap','2048',gpu=1,cap=3600)
        action('scoring-agreement','agreement',pilot=True)
        action('teacher-determinism','determinism',gpu=8,cap=1800,pilot=True)
        action('full-wave-profile','profile-waves',gpu=8,cap=3600,pilot=True)
    corpus_gate=rows[-1]['id']
    for task in TASKS:
        action('teacher-'+task,'teacher','--task',task,gpu=8,cap=1800 if phase=='qualification' else 14400,needs=[corpus_gate]+(['teacher-chemistry'] if task=='finqa' else []))
    if phase=='main':
        add('rewrite-allocation-selftest',['python','{kit}/p4_contain.py','selftest','--work','{work}',
            '--out','{work}/k8b4/containment','--block','qualification','--block-limit','100','--ceiling','560'],needs=['teacher-'+t for t in TASKS])
    for task in TASKS:
        action('rewrite-'+task,'rewrite','--task',task,gpu=8,cap=1800 if phase=='qualification' else 14400,needs=(['rewrite-allocation-selftest'] if phase=='main' else ['teacher-'+t for t in TASKS])+(['rewrite-chemistry'] if task=='finqa' else []))
    action('coverage','coverage',pilot=True,needs=['rewrite-'+t for t in TASKS])
    action('training-set','training-set',pilot=True)
    if phase=='qualification':
        action('resolved-configs','configs',pilot=True)
        action('restoration-check','restoration',gpu=8,cap=3600,pilot=True)
    if phase=='main':
        add('scientific-allocation-selftest',['python','{kit}/p4_contain.py','selftest','--work','{work}',
            '--out','{work}/k8b4/containment','--block','scientific','--block-limit','560','--ceiling','560'])
    if phase=='qualification':
        for task in TASKS:
            action('profile-probe-'+task,'probe','--slot','q-S-chemistry','--task',task,'--step','0',gpu=1,cap=1800)
            action('profile-conditioned-'+task,'probe','--slot','q-S-'+task,'--task',task,'--step','0','--conditioned',gpu=1,cap=1800)
        jobs=[{'id':f'q-{arm}-{task}','arm':arm,'seed':101,'task':task,'incoming':None,'order':task+'-alone'} for arm in ARMS for task in TASKS]
    else: jobs=slots()
    if phase=='main':
        for task in TASKS:
            action('conditioned-probe-'+task,'probe','--slot','D-'+task+'-alone-s101',
                   '--task',task,'--step','0','--conditioned',gpu=1,cap=1800)
    training_order=True
    for index,job in enumerate(jobs):
        ident=job['id']
        action('schedule-'+ident,'schedule','--slot',ident,needs=['training-set']+(['merge-'+job['incoming']] if job['incoming'] else []))
        action('train-'+ident,'train','--slot',ident,gpu=8,cap=1800 if phase=='qualification' else 10800,needs=['schedule-'+ident])
        action('merge-'+ident,'merge','--slot',ident,cap=1200,needs=['train-'+ident])
        for task in (*TASKS,'chemistry_probe') if phase=='main' else TASKS:
            for cap in CAPS if phase=='main' else (2048,):
                action(f'score-{ident}-{task}-{cap}','score','--checkpoint',ident,'--task',task,'--cap',str(cap),gpu=1,cap=3600,needs=['merge-'+ident])
        if phase=='main':
            for step in (0,20,40):
                for task in TASKS:
                    action(f'probe-{ident}-{task}-{step}','probe','--slot',ident,'--task',task,'--step',str(step),gpu=1,cap=1800,needs=['merge-'+ident])
            if index==1: action('D-standalone-gate','standalone',pilot=True,needs=[f'score-D-{task}-alone-s101-{panel}-{cap}' for task in TASKS for panel in (*TASKS,'chemistry_probe') for cap in CAPS])
    action('report','report',pilot=True)
    action('archive-pack','archive-pack',cap=600)
    action('PAUSE' if phase=='qualification' else 'FINISHED','pause' if phase=='qualification' else 'finished')
    return {'schema':'kit-campaign.v1','name':'v4-'+phase,'workdir_env':'WORK','rows':rows,
            'v4':{'phase':phase,'parallel_jobs':1,'cpu_prepare':{'first':'import-smoke','command':['python','{kit}/v4_prepare.py','--work','{work}','--phase',phase]},'generation_shards':{'teacher':4,'rewrite':8},'seeds':list(SEEDS),'steps':2 if phase=='qualification' else 40,
                  'stop_policy':{'row_failure':'automatic reviewed infrastructure admission; otherwise FAIL',
                    'allocation_preempted':'end allocation; reconcile on login node; next allocation selftest first',
                    'hard':'manager-written stop-sha256 acknowledgement, reconcile, fresh allocation selftest; accounting retained'},
                  'qualification_limit':100,'ceiling':560,'probe_questions_per_task':20,
                  'scoring_gpu':'one Slurm-assigned physical GPU, stable fingerprint across scoring',
                  'allocation_stages':({'qualification':{'first':'containment-selftest','last':'PAUSE','block':'qualification','block_limit':100}} if phase=='qualification' else {
                      'teacher':{'first':'containment-selftest','last':'teacher-finqa','block':'scientific','block_limit':560},
                      'rewrite':{'first':'rewrite-allocation-selftest','last':'training-set','block':'qualification','block_limit':100},
                      'scientific':{'first':'scientific-allocation-selftest','last':'FINISHED','block':'scientific','block_limit':560}}),
                  'slot_order':jobs}}


def write_campaigns(out):
    """Write reviewable YAML (JSON is a YAML subset), refusing overwrites."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    paths=[]
    for phase in ('qualification','main'):
        path=out/('v4-'+phase+'.yaml')
        with path.open('x') as handle: json.dump(build(phase),handle,indent=2); handle.write('\n')
        paths.append(path)
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'campaigns')
    args=parser.parse_args(argv)
    for path in write_campaigns(args.out): print(path)
    return 0


if __name__=='__main__': raise SystemExit(main())
