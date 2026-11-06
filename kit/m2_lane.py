"""CPU expansion of core lanes into ordered checkpoint/panel argv arrays.

The lane pool consumes commands; reducers run in the CPU publication job.
Engine descriptions mirror the unchanged leaf constructors; no scorer numerics change.
"""
import argparse
import importlib
from pathlib import Path
from kit import m2_common as m


def option(command,key,default=None):return command[command.index(key)+1] if key in command else default


def arguments(module,command):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--standin',action='store_true')
    module.configure(p)
    return p.parse_args(command[2:])


def plan(commands,out,task,standin):
    result=[];descriptions=[];reducers=[];refs=[];score_outputs=set()
    def ref(value,base=False):refs.append((value,base))
    def add(command,record=None,length=12288,workload=None):
        argv=[str(x) for x in command];result.append(argv);entry=dict(argv=argv)
        if record is not None:
            from kit.m2_paths import local
            model=local(record['model'])
            entry.update(requires=[str(model/'config.json')]+([str(local(record['receipt']))] if record.get('receipt') else []),
                engine=dict(backend='vllm',model=str(model),max_model_len=length,
                            **({'workload':workload} if workload else {})))
        descriptions.append(entry)
    for command in commands:
        name=Path(command[1]).stem
        seed=int(option(command,'--seed',task.get('seed') or 101))
        step=int(option(command,'--step',0));arm=option(command,'--arm')
        if name in ('m2_run','x_run') and option(command,'--mode')=='score':
            module=importlib.import_module('kit.'+name);a=arguments(module,command);a.command_sink=[]
            module.job(a)
            nested=plan(a.command_sink,out,task,standin)
            result+=nested['commands'];descriptions+=nested['pool_commands'];reducers+=nested['reducers'];refs+=nested['refs']
        elif name=='m2_rescore':
            doc=m.binding(out,standin);package=next(p for p in doc['packages'] if p['id']==option(command,'--package'))
            from kit.m2_rescore import effective_calls,ref_for
            for index,call in enumerate(effective_calls(package)):
                value=ref_for(doc,call);ref(value,call['checkpoint']=='base' or Path(call['score_set']).name.startswith('base-'))
                add([*command,'--call',index],value,workload='likelihood' if call.get('panel_set') in ('hellaswag','knowledge-ll') else None)
        elif name in ('m2_score','m2_reprobe','x_score'):
            if name=='m2_reprobe':
                from kit.m2_reprobe import checkpoint
                value=checkpoint(out,m.binding(out,standin),'base' if arm=='base' else m.legal_arm(arm),seed,step);ref(value,arm=='base')
            elif name=='x_score':
                from kit.x_common import context,checkpoint
                doc=context(out,standin)
                if arm=='pre':
                    from kit.x_common import incoming
                    value=incoming(doc,seed);ref(value)
                else:
                    from kit.x_common import disposition
                    value=None
                    if disposition(out,arm)=='READY':
                        value=checkpoint(out,doc,arm,seed,step);ref(value)
            else:value=m.record(out,m.binding(out,standin),arm,seed,step);ref(value,arm=='base')
            if name=='x_score' and step==0 and __import__('kit.x_score',fromlist=['aliasable']).aliasable(arm,doc):reducers.append(command)
            else:add(command,value)
        elif name=='s_drift':
            from kit.s_common import context
            from kit.s_generate import domains_for
            doc=context(out,standin);stage=next(s for s in doc['spec']['stages'] if s['id']==option(command,'--stage'))
            for domain in domains_for(doc['spec'],stage['incoming']):
                for k,ident in {'0':stage['incoming'],**stage['checkpoints']}.items():
                    ref(doc['spec']['nodes'][ident]['checkpoint'])
                    add([*command,'--domain',domain,'--step',k])
        elif name=='s_score':
            from kit.s_common import context
            doc=context(out,standin);spec=doc['spec']
            if option(command,'--action')=='v4-step20':
                values=[(None,r['checkpoint']) for r in spec['v4_step20'] if r['slot']==option(command,'--slot')]
            else:
                stages=[s for s in spec['stages'] if s['seed']==seed]
                values=[(ident,spec['nodes'][ident]['checkpoint']) for ident in sorted({s['incoming'] for s in stages}|{s['checkpoints'][k] for s in stages for k in ('20','40')})]
            for ident,value in values:
                ref(value)
                for panel in ('general-v1','finqa'):
                    key=(m.s.digest(doc.get('identities',{}).get(ident,value)),panel)
                    if key not in score_outputs:
                        score_outputs.add(key);add([*command,*(['--node',ident] if ident else []),'--score-panel',panel],value)
            reducers.append([*command,'--read-only'])
        elif name=='s_probe' and option(command,'--phase')=='score':
            from kit.s_common import context
            doc=context(out,standin);incoming=option(command,'--incoming');domain=option(command,'--domain')
            ref(doc['spec']['nodes'][incoming]['checkpoint'])
            from kit.s_generate import selected
            own=selected(out,incoming,domain)
            if len(own['train'])!=128 or len(own['monitor'])!=64:
                add(command);continue
            work=out/'probes'/incoming/domain/'work'
            ref(dict(model=str(work/'hf-probe'),receipt=str(work/'export.json')))
            for part in ('before','after'):add([*command,'--measurement',part])
            reducers.append([*command,'--measurement','reduce'])
        elif name=='x_references':
            from kit.x_common import context
            doc=context(out,standin);core=m.binding(Path(doc['message2']),standin)
            values={'base':doc['base']};key='standalone-legal-'+doc['configuration']
            if f'{key}:{seed}' in core.get('comparators',{}):values[key]=core['comparators'][f'{key}:{seed}']
            for label,value in values.items():
                ref(value,label=='base')
                for panel in (*m.PANEL_SETS,'hellaswag'):add([*command,'--reference-arm',label,'--panel-set',panel],value)
            reducers.append([*command,'--read-only'])
        elif name=='x_signal' and option(command,'--action')=='drift' or name=='m2_stage' and option(command,'--action')=='tier2-measure':
            from kit.x_common import context,incoming,checkpoint,disposition
            doc=context(out,standin)
            if name=='x_signal':arm='drift';step=10
            if disposition(out,arm)=='READY':
                ref(incoming(doc,seed));ref(checkpoint(out,doc,arm,seed,step))
                for domain in doc['domains']:
                    for part in ('before','after'):add([*command,'--domain',domain,'--measurement',part])
                reducers.append([*command,'--measurement','reduce'])
        elif name=='m2_stage' and option(command,'--action')=='x-probes' and option(command,'--phase')=='score':
            from kit.x_common import context,disposition
            doc=context(out,standin)
            if disposition(out,'plasticity')=='READY':
                for domain in doc['domains']:
                    child=[command[0],str(m.KIT/'s_probe.py'),'--out',str(out),'--incoming',f'b-s{seed}','--domain',domain,'--phase','score',
                        '--score-gpu',option(command,'--score-gpu','0'),'--inference-python',option(command,'--inference-python',command[0])]
                    if standin:child+=['--standin']
                    nested=plan([child],out,task,standin)
                    result+=nested['commands'];descriptions+=nested['pool_commands'];reducers+=nested['reducers'];refs+=nested['refs']
        elif name in ('s_generate','x_support'):
            if name=='s_generate':from kit.s_common import context
            else:from kit.x_common import context
            doc=context(out,standin);ident=option(command,'--incoming',f'b-s{seed}')
            from kit.s_generate import domains_for
            value=doc['spec']['nodes'][ident]['checkpoint'];ref(value)
            for domain in domains_for(doc['spec'],ident):
                for part in ('train','monitor'):add([*command,'--domain',domain,'--part',part],value,8192)
            reducers.append([*command,'--finalize'])
        elif name=='next_pools':
            from kit.next_common import read_inputs
            base=read_inputs(out,standin)['base'];value=dict(base,model=base['path']);ref(value,True)
            for i in range(3):add([*command,'--generation-shard',i],value,8192)
            reducers.append([*command,'--reduce-only'])
        elif name=='m2_format':
            doc=m.binding(out,standin)
            for label,value in sorted(doc['starts'].items()):
                if label=='base' or label.endswith(':'+str(seed)):
                    ref(value,label=='base');add([*command,'--checkpoint',label])
            reducers.append([*command,'--read-only'])
        else:add(command)
    from kit.lane_pool import capacity
    return dict(schema='message2-lane-commands.v1',commands=result,pool_commands=descriptions,command_count=len(result),
        lane_concurrency=max((capacity(c) for c in descriptions),default=1),reducers=reducers,refs=refs)


def stage(plan,standin):
    """Materialise and hash every checkpoint before any GPU command may start."""
    if standin:return
    from kit.m2_paths import local
    from kit.p3a_common import NotReady
    seen=set()
    for ref,base in plan['refs']:
        key=str(ref['model'])
        if key in seen:continue
        seen.add(key)
        try:
            model=local(ref['model'])
            if not model.exists():raise NotReady('lane checkpoint pending: '+key)
            m.checked_model(ref,False,base=base)
        except FileNotFoundError as error:raise NotReady('lane checkpoint/receipt pending: '+key) from error


def pool_invocations(commands,directory):
    """Generation segments use K=3, likelihood K<=2; serial commands are barriers.

    HF NLL and format-validation commands have no qualified engine declaration.
    Never combine them with vLLM commands: lane_pool uses the minimum capacity.
    Reducers run only after all segments exit, on CPU, under the same lane proof.
    """
    from kit.lane_pool import capacity,invocation
    groups=[]
    for command in commands:
        k=capacity(command)
        if not groups or groups[-1][0]!=k:groups.append((k,[]))
        groups[-1][1].append(command)
    return [invocation(group,Path(directory)/str(i),concurrency=k) for i,(k,group) in enumerate(groups)]
