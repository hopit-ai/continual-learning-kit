#!/usr/bin/env python3
"""Materialise static M/R/S/X jobs. The partner alone submits selected jobs/dependencies."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
from kit.m2_seed_policy import SEEDS
from kit.m2_assignment import require
from kit import m2_snapshot as snapshots


def training_uses_p2_cache(command):
    script=Path(command[1]).stem if len(command)>1 else ''
    if script=='x_run':return '--action' in command and command[command.index('--action')+1]=='p'
    if script!='x_train' or '--arm' not in command:return False
    arm=command[command.index('--arm')+1]
    if arm in ('control','p'):return True
    if arm.startswith('t2-'):
        from kit.x_tier2 import identity
        return identity(arm)['p2']
    return False


def build(settings):
    """Settings contain paths and CLI arguments, never scientific row definitions."""
    tasks=[];train_py=settings['trainer_python'];score_py=settings['inference_python']
    runtime=settings.get('runtime',{})
    def cmd(script,**kw):
        python=train_py if script in ('m2_train','x_train') else score_py
        result=[python,'{kit}/'+script+'.py','--out','{out}']
        for key,value in kw.items():
            if value is None or value is False:continue
            result+=['--'+key.replace('_','-')]
            if value is not True:result+=list(map(str,value)) if isinstance(value,list) else [str(value)]
        if settings.get('technical_synthetic') and script in ('m2_admit','m2_read','x_read') and 'bootstrap' not in kw:result+=['--bootstrap','20']
        return result
    def options(script):
        keys={'m2_train':('sdpo','tokenizer'),'m2_run':('sdpo','tokenizer','panel_root'),
              'm2_score':('panel_root',),'m2_reprobe':('sdpo','tokenizer','panel_root'),
              'x_train':('sdpo','tokenizer'),'x_score':('panel_root',),'x_references':('panel_root',),
              'x_run':('sdpo','tokenizer','panel_root'),'s_score':('finqa',)}.get(script,())
        return {k:runtime[k] for k in keys if k in runtime}
    def add(ident,kind,row,parents,commands,*,family=None,seed=None,barriers=(),overlay=None,**kw):
        task=dict(id=ident,kind=kind,row=row,requires=list(dict.fromkeys(parents)),
            overlay=list(parents if overlay is None else overlay),commands=commands,compact_overlay=True,**kw)
        if kind in ('reduce-admission','collect'):
            task['overlay']=[];task['evidence_from']=list(parents)
        elif kind=='score-family':
            task['overlay']=[];task['evidence_from']=list(parents);task['view_mode']='score'
        if kind=='train-stage' and any('--phase' in c and c[c.index('--phase')+1]=='continue' for c in commands):task['view_mode']='full'
        if kind=='train-stage' and any(training_uses_p2_cache(c) for c in commands):
            task['training_cache_prefixes']=[f'p2/s{seed}/']
        if family:task.update(family=family,seed=seed)
        if barriers:task['barriers']=list(barriers)
        tasks.append(task);return ident
    def invoke(script,**kw):return cmd(script,**dict(options(script),**kw))
    def mrun(action,seed,mode,phase='all'):
        return invoke('m2_run',action=action,seed=seed,mode=mode,score_phase=phase,
            trainer_python=train_py,inference_python=score_py,gpu=runtime.get('gpus','0,1,2,3,4,5,6,7'),score_gpu='{score_gpu}',durable='{durable}',model=runtime.get('model'))
    def scored(script,arm,seed,steps,extra=()):
        return [invoke(script,arm=arm,seed=seed,step=k,panel_set=p,gpu='{score_gpu}') for k in steps
            for p in (('test',*extra) if k in (10,30) else (*(('finqa','test','general-v1','hellaswag') if script=='m2_reprobe' else ('finqa','test','math_hard','general-v1','hellaswag')),*extra))]
    # INPUTS is populated by m2_bundle before jobs. It is an immutable tar snapshot.
    add('INPUTS','prepare','M1',[],[[score_py,'-c','raise SystemExit("PENDING: run m2_bundle before jobs")']],overlay=[])
    # First job on every node. Runtime enforces this gate even if a caller
    # omits sbatch dependencies. Failed nodes do not veto independent nodes.
    for index,node in enumerate(settings['nodes']):
        add(f'PF-node-{index}','prepare','PREFLIGHT',[],
            [cmd('m2_node_preflight',trainer_python=train_py,inference_python=score_py,durable='{durable}',local_root='{local}',node=node)],
            overlay=[],preflight_slot=index,uses_gpu=True,required_outputs=['preflight/node.json'])
    add('PF-ARC','prepare','ARC-PREFLIGHT',['INPUTS'],
        [cmd('m2_stage',action='pool-source',source=settings['next_source']),
         cmd('next_pools',**dict(settings['pools'],prepare_candidates_only=True))],overlay=[],
        required_outputs=['pools/candidate-manifest.json'])
    add('M2','prepare','M2',['INPUTS'],[cmd('m2_bind',**settings['bind'])],overlay=[])
    add('M3b','prepare','M3b',['INPUTS','PF-ARC'],[cmd('m2_stage',action='pool-source',source=settings['next_source']),cmd('next_pools',**settings['pools'])],overlay=[])
    add('M3','prepare','M3',['M2'],[cmd('m2_prepare',**dict(settings['prepare'],pool_free=True))],overlay=['M2'])
    add('M-pools','prepare','M3-pools',['M3','M3b'],[cmd('m2_prepare',**dict(settings['prepare'],pool_root='{input_M3b}/pools'))],overlay=['M3'])
    add('R0','prepare','RE-PROBE',['M-pools'],[cmd('m2_reprobe',action='prepare',message2='{input_M-pools}')],overlay=[])
    # All four incoming controls and all base guards precede training.
    rfinal=[]
    for seed in SEEDS:
        incoming='R-incoming-'+str(seed)
        commands=scored('m2_reprobe','base',seed,(0,),('panel',))
        for arm in ('generic','broad','earlier','both'):commands+=scored('m2_reprobe',arm,seed,(0,),('panel',))
        add(incoming,'score-family','RE-PROBE',['R0'],[[*c[:4],'--action','score',*c[4:]] for c in commands],family='reprobe',seed=seed)
        finals=[]
        for arm in ('generic','broad','earlier','both'):
            train=f'R-{arm}-{seed}'
            add(train,'train-stage','RE-PROBE',[incoming],[invoke('m2_reprobe',action='train',arm=arm,seed=seed,gpu=runtime.get('gpus','0,1,2,3,4,5,6,7'),trainer_python=train_py)],family='reprobe',seed=seed)
            final=train+'-score'
            commands=scored('m2_reprobe',arm,seed,(10,20,30,40),('panel',))
            add(final,'score-family','RE-PROBE',[train],[[*c[:4],'--action','score',*c[4:]] for c in commands],family='reprobe',seed=seed);finals.append(final)
        rfinal+=finals
    add('R-broad','reduce-admission','RE-PROBE',['R0',*[n for n in rfinal if n.startswith(('R-generic-','R-broad-'))]],[cmd('m2_reprobe',action='nominate',scope='broad',durable='{durable}')],partial_inputs=True,required_outputs=['nomination-commit.json'])
    add('R-nominate','reduce-admission','RE-PROBE',['R0',*rfinal],[cmd('m2_reprobe',action='nominate',durable='{durable}')],partial_inputs=True,required_outputs=['nomination-commit.json'])
    add('R-collect','collect','RE-PROBE',['R0','R-broad','R-nominate',*rfinal],[cmd('m2_reprobe',action='collect',archive='{out}/reprobe-return.tar.gz')],partial_inputs=True)
    # Independent Tier A scoring; optional B must be enabled at freeze time.
    add('RA0','prepare','RETURN1-A',['M2'],[cmd('m2_rescore',action='prepare',message2='{input_M2}',enable_tier_b=settings.get('enable_tier_b',False))],overlay=[])
    from kit.m2_return1 import decisions
    rescoring=[]
    for package in decisions()['rescoring']['packages']:
        if package['tier']=='B' and not settings.get('enable_tier_b'):continue
        ident='R'+package['id']
        add(ident,'score-family','RETURN1-'+package['tier'],['RA0'],[cmd('m2_rescore',action='score',package=package['id'],gpu='{score_gpu}',panel_root=runtime.get('panel_root'))],family='return1.'+package['id'],seed=package['seed']);rescoring.append(ident)
    add('RA-collect','collect','RETURN1-A',rescoring,[cmd('m2_rescore',action='read'),cmd('m2_rescore',action='collect',archive='{out}/return1-rescoring.tar.gz')],partial_inputs=True)
    boundaries=[];plain=[];generic=[];broad=[]
    add('M5','prepare','M5',['M3'],[mrun('targets',None,'all')])
    for seed in SEEDS:
        ident=f'M4-{seed}'
        add(ident,'score-family','M4',['M3'],[mrun('boundaries',seed,'score')],family='core',seed=seed);boundaries.append(ident)
        inc=f'M6-{seed}-incoming'
        add(inc,'score-family','M6',[ident,'M5'],[mrun('math-plain',seed,'score','incoming')],family='core',seed=seed)
        tr=f'M6-{seed}-train';add(tr,'train-stage','M6',[inc],[mrun('math-plain',seed,'train')],family='core',seed=seed)
        final=f'M6-{seed}-score';add(final,'score-family','M6',[tr],[mrun('math-plain',seed,'score','final')],family='core',seed=seed);plain.append(final)
    # Nomination files are imported into the core binding; C-route runs never overlay M.
    add('M-nomination','prepare','M7',['M-pools','R-broad'],[cmd('m2_stage',action='nomination',scope='broad',durable='{durable}')],overlay=['M-pools'])
    add('M-final-nomination','prepare','M9',['M-pools','R-nominate'],[cmd('m2_stage',action='nomination',durable='{durable}')],overlay=['M-pools'])
    def legal_wave(action,prefix,parents):
        final=[]
        for seed in SEEDS:
            inc=f'{prefix}-{seed}-incoming';add(inc,'score-family',prefix,parents+[f'M4-{seed}'],[mrun(action,seed,'score','incoming')],family='core',seed=seed)
            tr=f'{prefix}-{seed}-train';add(tr,'train-stage',prefix,[inc],[mrun(action,seed,'train')],family='core',seed=seed)
            sc=f'{prefix}-{seed}-score';add(sc,'score-family',prefix,[tr],[mrun(action,seed,'score','final')],family='core',seed=seed);final.append(sc)
        return final
    generic=legal_wave('legal-generic','M7',['M-pools'])
    broad=legal_wave('legal-broad','M8',['M-nomination'])
    add('M8-read','reduce-admission','M8',generic+broad,[cmd('m2_admit')],required_values=[dict(path='broad-decision.json',equals=dict(status='COMPLETE'))])
    earlier=legal_wave('legal-next','M9',['M8-read','M-final-nomination'])
    add('M9','reduce-admission','M9',['M8-read',*earlier],[cmd('m2_admit')],commit_barrier='M9')
    sibling=[]
    for seed in SEEDS:
        inc=f'M10-{seed}-incoming';add(inc,'score-family','M10',['M9','M5'],[mrun('math-sibling',seed,'score','incoming')],family='core',seed=seed,barriers=['M9'])
        tr=f'M10-{seed}-train';add(tr,'train-stage','M10',[inc],[mrun('math-sibling',seed,'train')],family='core',seed=seed,barriers=['M9'])
        sc=f'M10-{seed}-score';add(sc,'score-family','M10',[tr],[mrun('math-sibling',seed,'score','final')],family='core',seed=seed,barriers=['M9']);sibling.append(sc)
    add('M10','prepare','M10',sibling,[cmd('m2_stage',action='m10')],barriers=['M9'],commit_barrier='M10')
    add('M-core-collect','collect','M12',['M3',*boundaries,*plain,'M9','M10'],[cmd('m2_read',source='{out}'),cmd('m2_collect',archive='{out}/message2-return.tar.gz')],partial_inputs=True)
    # S starts from M2 and has its own binding, state and archive.
    screen=settings['screen'];spec=json.loads(Path(screen['spec']).read_text())
    add('S1','prepare','S1',['INPUTS'],[cmd('s_inputs',sources=screen['sources'])],overlay=[])
    add('S2','prepare','S2',['S1','M2'],[cmd('s_bind',message2='{input_M2}',**screen)],overlay=['S1'])
    add('S3','prepare','S3',['S2'],[cmd('s_inventory')])
    add('S5','prepare','S5',['S2'],[cmd('s_audit')])
    add('S6','prepare','S6',['S5'],[cmd('s_freeze')],commit_barrier='S6')
    sfinal=[]
    for slot in spec['v4_step20']:
        ident='S4-'+slot['slot'];add(ident,'score-family','S4',['S2'],[invoke('s_score',action='v4-step20',slot=slot['slot'],gpu='{score_gpu}')],family='s-inventory.'+slot['slot'],seed=0);sfinal.append(ident)
    incoming_ids=sorted({s['incoming'] for s in spec['stages']})
    from kit.s_generate import domains_for
    probes={}
    for incoming in incoming_ids:
        gen='S7-'+incoming;add(gen,'score-family','S7',['S6'],[cmd('s_generate',incoming=incoming,gpu='{score_gpu}')],family='s-generation.'+incoming,seed=0)
        for domain in domains_for(spec,incoming):
            ident=f'S10-{incoming}-{domain}-train';add(ident,'train-stage','S10',[gen],[cmd('s_probe',incoming=incoming,domain=domain,phase='train',trainer_python=train_py,inference_python=score_py)])
            probes[incoming,domain]=ident
    from kit.s_common import SEEDS as SCREEN_SEEDS
    for seed in SCREEN_SEEDS:
        stages=[s for s in spec['stages'] if s['seed']==seed];parents=['S6'];commands=[]
        for stage in stages:
            parents.append('S7-'+stage['incoming']);commands.append(cmd('s_drift',stage=stage['id'],gpu='{score_gpu}'))
        for incoming in sorted({s['incoming'] for s in stages}):
            for domain in domains_for(spec,incoming):
                parents.append(probes[incoming,domain]);commands.append(cmd('s_probe',incoming=incoming,domain=domain,phase='score',score_gpu='{score_gpu}',inference_python=score_py))
        commands.append(invoke('s_score',action='boundaries',seed=seed,gpu='{score_gpu}'))
        ident=f'S8-S10-{seed}';add(ident,'score-family','S8-S10',parents,commands,family='screen-s',seed=seed);sfinal.append(ident)
    add('S12','collect','S12',['S3',*sfinal],[cmd('s_price'),cmd('s_collect',archive='{out}/screen-s-return.tar.gz')],partial_inputs=True)
    # X depends on M's committed admission/disposition and S6, never S's analysis.
    add('X1','prepare','X1',['M10','S6'],[cmd('x_bind',durable='{durable}',message2='{input_M10}',message2_durable='{durable}',screen_s='{input_S6}',screen_s_durable='{durable}',spec=settings['addon']['spec'])],overlay=[],barriers=['M9','M10','S6'])
    add('X2','prepare','X2',['X1'],[cmd('x_support',action='audit')])
    xgen=[]
    for seed in SEEDS:
        ident=f'X3-{seed}';add(ident,'score-family','X3',['X2'],[cmd('x_support',action='generate',seed=seed,gpu='{score_gpu}')],family='x-generation',seed=seed);xgen.append(ident)
    add('X3','prepare','X3',xgen,[cmd('x_support',action='reduce')])
    measures=[]
    for seed in SEEDS:
        ident=f'X4-{seed}';add(ident,'train-stage','X4',['X3'],[invoke('x_run',action='measure',seed=seed,mode='train',smoke=settings['addon']['smoke'],trainer_python=train_py,inference_python=score_py)],family='x-tier1',seed=seed);measures.append(ident)
    add('X4','reduce-admission','X4',measures,[cmd('x_p',action='reduce',smoke=settings['addon']['smoke'])])
    xfinal=[]
    for seed in SEEDS:
        pre=f'X-incoming-{seed}';commands=scored('x_score','pre',seed,(0,))+[invoke('x_references',seed=seed,gpu='{score_gpu}')]
        for arm in ('control','p','drift','plasticity'):commands+=scored('x_score',arm,seed,(0,))
        add(pre,'score-family','X5',['X4'],commands,family='x-tier1',seed=seed)
        # Independent incoming/domain probes are trained once, then assigned NLL.
        pt=f'X9-probes-{seed}-train';add(pt,'train-stage','X9',['X3'],[cmd('m2_stage',action='x-probes',seed=seed,phase='train',trainer_python=train_py,inference_python=score_py)],family='x-tier1',seed=seed)
        ps=f'X9-probes-{seed}-score';add(ps,'score-family','X9',[pt],[cmd('m2_stage',action='x-probes',seed=seed,phase='score',score_gpu='{score_gpu}',inference_python=score_py)],family='x-tier1',seed=seed)
        for arm,row in (('control','X5'),('p','X6'),('drift','X7'),('plasticity','X9')):
            parent=[pre]+([ps] if arm=='plasticity' else [])
            tr=f'{row}-{arm}-{seed}-train'
            command=invoke('x_train',arm=arm,seed=seed,phase='prefix' if arm=='drift' else 'full',trainer_python=train_py)
            if arm=='p':command=invoke('x_run',action='p',seed=seed,mode='train',trainer_python=train_py,inference_python=score_py)
            add(tr,'train-stage',row,parent,[command],family='x-tier1',seed=seed)
            if arm=='drift':
                sig=f'X8-{seed}-signal';add(sig,'score-family','X8',[tr],[cmd('x_signal',action='drift',seed=seed,score_gpu='{score_gpu}')],family='x-tier1',seed=seed)
                tr=f'X8-{seed}-continue';add(tr,'train-stage','X8',[sig],[invoke('x_train',arm=arm,seed=seed,phase='continue',trainer_python=train_py)],family='x-tier1',seed=seed)
            final=tr+'-score'
            commands=scored('x_score',arm,seed,(10,20,30,40))
            if arm=='p':commands=[invoke('x_run',action='p',seed=seed,mode='score',trainer_python=train_py,inference_python=score_py,score_gpu='{score_gpu}')]
            add(final,'score-family',row,[tr],commands,family='x-tier1',seed=seed);xfinal.append(final)
    add('X12','collect','X12',xfinal,[cmd('x_read',action='tables'),cmd('x_price'),cmd('x_collect',archive='{out}/addon-return.tar.gz')],commit_barrier='X12',partial_inputs=True)
    add('X14','prepare','X14',['X12'],[cmd('x_tier2',action='plan')],barriers=['X12'])
    from kit.x_tier2 import COMPARISONS
    second=[]
    for number,(comparison,specification) in enumerate(COMPARISONS.items(),15):
        row='X'+str(number);family='x-tier2.'+comparison
        for seed in SEEDS:
            pre=f'{row}-{seed}-incoming';commands=scored('x_score','pre',seed,(0,))+[invoke('x_references',seed=seed,gpu='{score_gpu}')]
            for side in ('control','challenger'):commands+=scored('x_score',f't2-{comparison}-{side}',seed,(0,))
            add(pre,'score-family',row,['X14'],commands,family=family,seed=seed,barriers=['X12'])
            # Rescore the same independent trained probes for this comparison's assignment.
            ps=f'{row}-{seed}-probe-score';add(ps,'score-family',row,['X14',f'X9-probes-{seed}-train'],[cmd('m2_stage',action='x-probes',seed=seed,phase='score',score_gpu='{score_gpu}',inference_python=score_py)],family=family,seed=seed,barriers=['X12'])
            for side in ('control','challenger'):
                arm=f't2-{comparison}-{side}';allocation=specification[side];parent=[pre,ps]
                tr=f'{row}-{side}-{seed}-train'
                add(tr,'train-stage',row,parent,[invoke('x_train',arm=arm,seed=seed,phase='full' if allocation=='standalone' else 'prefix',end=40 if allocation=='standalone' else 10,trainer_python=train_py)],family=family,seed=seed,barriers=['X12'])
                if allocation!='standalone':
                    for step in ((10,20,30) if allocation=='adaptive' else (10,)):
                        parent=[tr]
                        if allocation in ('adaptive','drift'):
                            sig=tr+f'-measure{step}';add(sig,'score-family',row,[tr],[cmd('m2_stage',action='tier2-measure',arm=arm,seed=seed,step=step,gpu='{score_gpu}')],family=family,seed=seed,barriers=['X12']);parent=[sig]
                        tr=f'{row}-{side}-{seed}-continue{step}'
                        add(tr,'train-stage',row,parent,[invoke('x_train',arm=arm,seed=seed,phase='continue',end=step+10 if allocation=='adaptive' else 40,trainer_python=train_py)],family=family,seed=seed,barriers=['X12'])
                    if allocation=='adaptive':
                        final_measure=tr+'-measure40';add(final_measure,'score-family',row,[tr],[cmd('m2_stage',action='tier2-measure',arm=arm,seed=seed,step=40,gpu='{score_gpu}')],family=family,seed=seed,barriers=['X12']);tr=final_measure
                final=tr+'-score';add(final,'score-family',row,[tr],scored('x_score',arm,seed,(10,20,30,40)),family=family,seed=seed,barriers=['X12']);second.append(final)
    add('X23','collect','X23',['X12',*second],[cmd('x_read',action='tables'),cmd('x_price',tier2=True),cmd('x_collect',archive='{out}/addon-extended-return.tar.gz')],partial_inputs=True)
    validation_tasks={}
    for family,seed in sorted({(t['family'],t['seed']) for t in tasks if t['kind']=='score-family' and
            t['family'] in ('reprobe','core','x-tier1') or t['kind']=='score-family' and t.get('family','').startswith('x-tier2.')}):
        ident=f'FV-{family}-{seed}'
        add(ident,'score-family','FORMAT-VALIDATION',['M3'],[cmd('m2_format',seed=seed,gpu='{score_gpu}')],family=family,seed=seed)
        validation_tasks[family,seed]=ident
    for task in tasks:
        ident=validation_tasks.get((task.get('family'),task.get('seed')))
        if task['kind']=='score-family' and ident and task['id']!=ident:
            task['requires'].append(ident);task.setdefault('imports',[]).append(dict(task=ident,source='format-validation',target='format-validation'))
    for task in tasks:
        if task['kind']=='train-stage' and task['row'].startswith('X') and task['row']!='X4':
            for command in task['commands'][:]:
                if command[1].endswith('/x_train.py'):
                    arm=command[command.index('--arm')+1];seed=command[command.index('--seed')+1]
                    phase=command[command.index('--phase')+1]
                    end=command[command.index('--end')+1] if '--end' in command else ('10' if phase=='prefix' else '40')
                    task['commands'].append(cmd('m2_stage',action='x-stage-check',arm=arm,seed=seed,phase='train',step=end))
        if task['id'] in ('M3b','M5') or task['row'] in ('S7','X3') and task['kind']=='prepare':task['uses_gpu']=True
    ordered=[];waiting=tasks[:];done=set()
    while waiting:
        ready=[task for task in waiting if set(task['requires'])<=done]
        require(bool(ready),'static dependency cycle')
        ordered.extend(ready);done.update(task['id'] for task in ready)
        waiting=[task for task in waiting if task['id'] not in done]
    return dict(schema='message2-launch.v2',nodes=settings['nodes'],node_preflight_tasks={node:f'PF-node-{index}' for index,node in enumerate(settings['nodes'])},tasks=ordered,technical_synthetic=settings.get('technical_synthetic',False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--settings',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--enable-tier-b',action='store_true')
    a=p.parse_args();settings=json.loads(a.settings.read_text())
    if a.enable_tier_b:settings['enable_tier_b']=True
    doc=build(settings);snapshots.atomic(a.output,doc)
    from kit.m2_node import load_manifest
    load_manifest(a.output);print(len(doc['tasks']),'explicit jobs; no jobs submitted')
if __name__=='__main__':main()
