#!/usr/bin/env python3
"""Ordered runbook rows, serial child jobs; the partner's existing wrapper owns TERM/save."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
import subprocess


def job(a):
    doc=binding(a.out,a.standin)
    from kit.m2_select import selected
    seeds=[seed for seed in SEEDS if selected(a,stage=a.action,arm=a.action,seed=seed)]
    mode=getattr(a,'mode','all')
    score_phase=getattr(a,'score_phase','all')
    def run(name,*args,python=None):
        cmd=[python or a.trainer_python,str(KIT/(name+'.py')),'--out',str(a.out),*map(str,args)]
        if a.standin: cmd.append('--standin')
        subprocess.run(cmd,check=True)
    def scores(arm,seed,step,panels):
        if mode=='train' or score_phase=='incoming' and step!=0 or score_phase=='final' and step==0:return
        for panel in panels:
            args=['--arm',arm,'--seed',seed,'--step',step,'--panel-set',panel,'--gpu',a.score_gpu]
            if a.panel_root: args+=['--panel-root',a.panel_root]
            run('m2_score',*args,python=getattr(a,'inference_python',a.trainer_python))
    def train(stage,seed,**kw):
        if mode=='score':return
        args=['--stage',stage,'--seed',seed,'--gpu',a.gpu,'--trainer-python',a.trainer_python]
        if a.durable is not None: args+=['--durable',a.durable]
        for key,value in dict(tokenizer=a.tokenizer,sdpo=a.sdpo,**kw).items():
            if value is not None: args+=['--'+key.replace('_','-'),value]
        run('m2_train',*args)
    if a.action=='boundaries':
        for seed in seeds:
            scores('base',seed,0,(*PANEL_SETS,'hellaswag'))
            for trajectory in ('plain','sibling'): scores('finqa-'+trajectory,seed,40,PANEL_SETS)
            for step in STEPS: scores('legal-plain',seed,step,PANEL_SETS if step in RETENTION_STEPS else ('test',))
            scores('standalone-legal-plain',seed,40,('test',))
    elif a.action=='targets':
        args=['--gpu',a.teacher_gpu]
        if a.model: args+=['--model',a.model]
        if a.tokenizer: args+=['--tokenizer',a.tokenizer]
        run('m2_targets',*args,python=a.inference_python)
    elif a.action.startswith('legal-'):
        config=a.action.removeprefix('legal-')
        if a.standin_pending:
            require(a.standin,'--standin-pending is a technical fixture only')
            print('PENDING: connected confirmation explicitly unfinished; return evidence');return
        if config=='next':
            nomination_value=nomination(a.out,doc)
            if nomination_value['earlier'] is not True: print('SKIP: earlier-task not nominated');return
            path=a.out/'broad-decision.json'
            if not path.exists(): raise NotReady('broad confirmation must be read before earlier-task confirmation')
            broad=json.loads(path.read_text())
            if broad['status']!='COMPLETE': print('PENDING: broad confirmation incomplete');return
            config='both' if broad['admitted'] else 'earlier'
        if not permitted_legal(a.out,doc,config): print('SKIP: replay configuration not nominated');return
        for seed in seeds:
            scores(legal_arm(config),seed,0,(*PANEL_SETS,'hellaswag'))
            train('legal',seed,configuration=config)
            for step in STEPS[1:]:
                scores(legal_arm(config),seed,step,(*PANEL_SETS,'hellaswag') if step in RETENTION_STEPS else ('test',))
    elif a.action.startswith('math-'):
        if doc['hard_math']!='QUALIFIES': print('SKIP: stage 3',doc['hard_math']);return
        trajectory=a.action.removeprefix('math-')
        if trajectory=='sibling':
            try: admission(a.out,doc)
            except NotReady: print('PENDING: sibling MATH waits; return evidence');return
            durable_admission(a.out,doc,a.durable)
        for seed in seeds:
            scores(math_arm(trajectory),seed,0,PANEL_SETS)
            train('math',seed,trajectory=trajectory)
            for step in STEPS[1:]: scores(math_arm(trajectory),seed,step,PANEL_SETS)
    elif a.action=='admit': run('m2_admit','--bootstrap',a.bootstrap)


def configure(p):
    from kit.m2_select import configure as selectors
    selectors(p)
    p.add_argument('--score-phase',choices=('all','incoming','final'),default='all')
    p.add_argument('--durable',type=Path)
    p.add_argument('--action',choices=('boundaries','targets','legal-generic','legal-broad','legal-next','admit','math-plain','math-sibling'),required=True)
    p.add_argument('--gpu',default='0,1,2,3,4,5,6,7');p.add_argument('--score-gpu',default='0');p.add_argument('--teacher-gpu',default='0,1')
    p.add_argument('--tokenizer');p.add_argument('--model');p.add_argument('--sdpo',type=Path);p.add_argument('--panel-root',type=Path)
    p.add_argument('--trainer-python',default=sys.executable);p.add_argument('--inference-python',default=sys.executable)
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--standin-pending',action='store_true',help='CPU fixture with unfinished confirmation; requires --standin')
if __name__=='__main__': raise SystemExit(entry(job,configure,'m2-run-{action}'))
