#!/usr/bin/env python3
"""CPU wrapper rehearsal. Real manifests/CLIs/shell wrappers; scientific children are stand-ins."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
from kit import combined_send as c, m2_manifest, m2_node, m2_assignment, m2_snapshot as snap, m3_manifest, sp_manifest


def fixtures(root):
    from kit.s_common import inventory_slots
    spec=root/'screen.json'
    nodes={'base':dict(root=True)}; stages=[]
    for tier in (1,2):
        for recipe in ('D','F'):
            for seed in (101,102,103):
                incoming='base' if tier==1 else f't1-{recipe}-s{seed}-40'
                if tier==2:nodes[incoming]=dict(root=False)
                stages.append(dict(id=f't{tier}-{recipe}-s{seed}',seed=seed,incoming=incoming))
    c.write(spec,dict(stages=stages,nodes=nodes,v4_step20=[dict(slot=r['id']) for r in inventory_slots()]))
    core=m2_manifest.build(dict(nodes=['core-fixture'],trainer_python=sys.executable,inference_python=sys.executable,
        bind={},next_source='/staged/NEXT',pools={},prepare={},screen=dict(spec=str(spec),sources='/staged/sources'),
        addon=dict(spec='/staged/x',smoke='/staged/smoke')))
    authority=dict(schema='message2-scoring-assignment.v2',assignments={f"{t['family']}|{t['seed']}":dict(
        node='core-fixture',uuid='CPU-STANDIN',environment='CPU-STANDIN') for t in core['tasks'] if t.get('family')})
    core=m2_node.bind_slots(core,authority)
    c.write(root/'core.json',core)
    tier=m3_manifest.build(dict(nodes=['tier-fixture'],message2_nodes=['core-fixture'],technical_synthetic=True,
        trainer_python=sys.executable,inference_python=sys.executable,
        inputs=dict(inventory=str(root/'hardware.json'),contract_targets='/staged/C',replay='/staged/B',finqa_targets=['/staged/finqa'],
                    panels={'general-v1':'/staged/general'},base='/staged/base')))
    c.write(root/'kcr.json',tier)
    from kit.m2_return1 import decisions
    inv=sp_manifest.inventory([core,tier],return1=decisions()['rescoring'])
    side=sp_manifest.build_sidecar([core,tier],inv)
    c.write(root/'sp.json',side);c.write(root/'inventory.json',inv)
    s=dict(core_manifest=str(root/'core.json'),kcr_manifest=str(root/'kcr.json'),sp_manifest=str(root/'sp.json'),
        sp_inventory=str(root/'inventory.json'),core_local_root=str(root/'core-local'),core_durable=str(root/'core-durable'),
        kcr_root=str(root/'kcr-state'),pd_root=str(root/'pd-state'),sp_root=str(root/'sp-state'),control_root=str(root/'control'),
        pd_nodes=['tier-fixture'],sp_nodes=['tier-fixture'],tier_hardware=str(root/'hardware.json'),
        trainer_python=sys.executable,inference_python=sys.executable,sp_node_pools=str(root/'node-pools.json'),
        pd_plans={str(seed):str(root/f'pd-plan-{seed}.json') for seed in (101,102,103)},technical_synthetic=True)
    c.write(s['tier_hardware'],dict(technical_synthetic=True))
    for seed in (101,102,103): c.write(s['pd_plans'][str(seed)],dict(seed=seed,exports={},smoke_receipt={}))
    doc=c.build(s);c.write(root/'combined.json',doc)
    durable=Path(s['core_durable']);m2_assignment.freeze(durable/'scoring-assignment.json',authority)
    m2_node.freeze_manifest(s['core_manifest'],durable)
    return doc


def bridge(args):
    """Only the explicit CPU harness installs these child-process stand-ins."""
    from contextlib import ExitStack
    context=c.load(os.environ['COMBINED_CPU_CONTEXT']);doc=c.load(context['manifest']);s=doc['settings']
    ident=context['id'];scenario=context.get('scenario','pass');count=Counter()
    target=Path(args[0]).name
    def leaf(command, **kwargs):
        count['scientific_children_stood_in']+=1
        if scenario=='broad-pending' and ident=='M3-broad': return NS(returncode=3)
        if scenario=='tier-failure': return NS(returncode=7)
        if ident=='M3-init':m3_manifest.init_root(s['kcr_root'],core_submitted=True)
        return NS(returncode=0)
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys,'argv',[args[0],*args[1:]]))
        if target=='m2_preflight.py': return 0
        if target=='combined_preflight.py':
            from kit import combined_preflight as pf
            stack.enter_context(patch.object(pf,'check',return_value=dict(status='PASS',technical_synthetic=True,
                trainer_imports='SYNTHETIC',inference_imports='SYNTHETIC',cgroup='SYNTHETIC',gpus=[])))
            result=pf.main()
        elif target=='m2_node.py':
            task=next(t for t in c.load(s['core_manifest'])['tasks'] if t['id']==ident)
            out=Path(s['core_local_root'])/ident/'out'
            class Child:
                def __init__(self, command, **kwargs):
                    count['scientific_children_stood_in']+=1
                    out.mkdir(parents=True,exist_ok=True)
                    for consumer in c.load(s['core_manifest'])['tasks']:
                        for item in consumer.get('imports',[]):
                            if item['task']==ident:
                                c.write(out/item['source']/'cpu-fixture.json',dict(technical_synthetic=True))
                    for name in task.get('required_outputs',[]):
                        c.write(out/name,dict(status='PASS',node='core-fixture',technical_synthetic=True,
                            assignment_sha256=m2_assignment.sha(Path(s['core_durable'])/'scoring-assignment.json')))
                    for cond in task.get('required_values',[]):c.write(out/cond['path'],cond['equals'])
                    if task.get('commit_barrier')=='M9':c.write(out/'admission.json',dict(status='COMPLETE'))
                    if task.get('commit_barrier')=='X12':
                        archive=out/'addon-return.tar.gz';archive.write_bytes(b'CPU STANDIN archive')
                        c.write(out/'collect-complete.json',dict(archive=str(archive),archive_sha256=c.sha(archive)))
                def wait(self,timeout=None):return 0
            stack.enter_context(patch.object(m2_node.subprocess,'Popen',Child))
            sys.argv.append('--standin');result=m2_node.main()
        elif target=='m3_manifest.py':
            stack.enter_context(patch.object(m3_manifest.socket,'gethostname',return_value='tier-fixture'))
            stack.enter_context(patch.object(m3_manifest.subprocess,'run',side_effect=leaf))
            result=m3_manifest.main(args[1:])
        elif target=='sp_manifest.py':
            original=sp_manifest.run
            def run(*a,**kw):return original(*a,**kw,runner=leaf)
            original_sub=sp_manifest.substitute
            def substitute(value,values,maps):
                if str(value).startswith(('{model:', '{gpu_identity:')):return 'CPU-STANDIN'
                return original_sub(value,values,maps)
            stack.enter_context(patch.object(sp_manifest,'run',side_effect=run))
            stack.enter_context(patch.object(sp_manifest,'substitute',side_effect=substitute))
            result=sp_manifest.main(args[1:])
        elif target=='pd_node.py':
            from kit import pd_node as pd
            stack.enter_context(patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'))
            stack.enter_context(patch.object(pd.smoke,'smoke_gates',return_value={'status':'QUALIFIED'}))
            # Synthetic qualification is local to this process, never a real receipt.
            plans=Path(s['pd_plans'][ident.split('-')[1]])
            value=c.load(plans);value['smoke_receipt']['code_sha256']=pd.probe.code_identity();c.write(plans,value)
            uuid='00000000-0000-0000-0000-000000000001'
            stack.enter_context(patch.object(pd.smoke,'discover_scoring_gpu',return_value=dict(uuid=uuid,devices=[uuid],environment='a'*64)))
            def launch(*a,**kw):
                count['scientific_children_stood_in']+=1
                if scenario=='tier-failure':raise RuntimeError('CPU injected distributed failure')
            stack.enter_context(patch.object(pd.probe,'launch_ranks',side_effect=launch))
            try:result=pd.main()
            except RuntimeError:result=7
        else:raise AssertionError('unexpected wrapper child '+target)
    c.write(Path(context['record']),dict(id=ident,result=result,counts=dict(count),technical_synthetic=True))
    return result


def tree_hash(root):
    h=hashlib.sha256()
    for p in sorted(Path(root).rglob('*')):
        if p.is_file():h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    doc=fixtures(root);s=doc['settings'];results=[]
    shim=root/'cpu-python'
    shim.write_text('#!/bin/sh\nexec '+__import__('shlex').quote(sys.executable)+' '+__import__('shlex').quote(str(Path(__file__).resolve()))+' --bridge-and-standin "$@"\n');shim.chmod(0o755)
    env=c.environment(s);env.update(JOB_PY=str(shim),PYTHONDONTWRITEBYTECODE='1',SP_INPUTS=s['core_local_root']+' '+s['kcr_root'])
    context=root/'context.json';env['COMBINED_CPU_CONTEXT']=str(context)
    def invoke(row,scenario='pass'):
        ident=row['id'];record=root/'records'/(scenario+'-'+ident+'.json')
        c.write(context,dict(manifest=str(root/'combined.json'),id=ident,scenario=scenario,record=str(record)))
        group=row['group']
        if group=='core':
            task=next(t for t in c.load(s['core_manifest'])['tasks'] if t['id']==ident)
            command=['bash',str(c.KIT/'jobs/message2-v2'/f'{c.WRAPPERS[task["kind"]]}.sh'),'--task',ident,'--node-slot',task['node_slot']]
        elif group=='preflight':command=[str(shim),str(c.KIT/'combined_preflight.py'),'--settings-json',json.dumps(s),'--node',row['node']]
        elif group=='kcr':command=['bash',str(c.KIT/'jobs/message3/run-task.sh'),ident]
        elif group=='sp':command=['bash',str(c.KIT/'jobs/message3-sp/run-task.sh')];env['TASK']=ident
        else:
            seed=str(row['seed']);pdroot=Path(s['pd_root']);pdroot.mkdir(parents=True,exist_ok=True)
            suffix='-failed' if scenario=='tier-failure' else ''
            command=[str(shim),str(c.KIT/'pd_node.py'),'--plan',s['pd_plans'][seed],
                '--bound-plan',str(pdroot/(seed+suffix+'.json')),'--out',str(pdroot/(seed+suffix))]
        done=subprocess.run(command,env=env,capture_output=True,text=True)
        with (root/'wrappers.log').open('a') as log:log.write(ident+' '+scenario+' exit='+str(done.returncode)+'\n'+done.stdout+done.stderr)
        result=c.load(record) if record.exists() else dict(id=ident,result=done.returncode,counts={})
        result.update(group=group,wrapper_exit=done.returncode,scenario=scenario)
        return result
    # Core manifest is topological; preflight first. INPUTS is a stand-in bundle row here.
    rows=doc['tasks']
    by_id={r['id']:r for r in rows}
    coretasks=c.load(s['core_manifest'])['tasks']
    coreorder=[t for t in coretasks if t['id'].startswith('PF-node-')]+[t for t in coretasks if not t['id'].startswith('PF-node-')]
    ordered=[by_id[t['id']] for t in coreorder]+[r for r in rows if r['group']=='preflight']+[r for r in rows if r['group'] in ('kcr','sp','pd')]
    for row in ordered:
        result=invoke(row);results.append(result)
        if result['wrapper_exit']!=0 or result['result']!=0:raise AssertionError(result)
        if row['group']=='sp':assert c.load(Path(s['sp_root'])/'state'/(row['id']+'.json'))['status']=='DONE',row['id']
    assert len(results)==len(rows) and {r['id'] for r in results}=={r['id'] for r in rows}
    before=tree_hash(s['core_durable']);before_local=tree_hash(s['core_local_root'])
    # Fresh M3 tree; real runner computes the entire dependency closure.
    shutil.rmtree(s['kcr_root'])
    pending=[]
    for row in [r for r in rows if r['group']=='kcr']:
        result=invoke(row,'broad-pending')
        if result['wrapper_exit']==75:pending.append(row['id'])
        else:assert result['wrapper_exit']==0,result
    expected=[t['id'] for t in c.load(s['kcr_manifest'])['tasks'] if t.get('arm') in ('G','CR')]
    assert all(m3_manifest.status(s['kcr_root'],i)['status']=='PENDING' for i in expected)
    assert all(m3_manifest.status(s['kcr_root'],t['id'])['status']=='DONE' for t in c.load(s['kcr_manifest'])['tasks'] if t.get('arm') in ('C0','K'))
    # Failure in each tier family has no path to the already completed core trees.
    failure_rows=[next(r for r in rows if r['id']==i) for i in ('M3-K-101-finqa-train','PD-101','SP-length')]
    failures=[invoke(r,'tier-failure') for r in failure_rows]
    assert [f['wrapper_exit'] for f in failures]==[1,7,0],failures
    assert c.load(Path(s['sp_root'])/'state/SP-length.json')['status']=='FAILED'
    assert tree_hash(s['core_durable'])==before and tree_hash(s['core_local_root'])==before_local
    # Real first-node wrapper gate: completed core outputs remain immutable on requeue.
    requeue=invoke(next(r for r in rows if r['id']=='M-core-collect'));assert requeue['wrapper_exit']==0
    summary=dict(technical_synthetic=True,gpu_hours=0,rows=len(rows),groups=dict(Counter(r['group'] for r in rows)),
        successful_wrapper_rows=len(results),children_stood_in=sum(r['counts'].get('scientific_children_stood_in',0) for r in results),
        broad_pending=pending,only_coverage_arms_pending=True,tier_failures=failures,core_trees_unchanged=True,
        core_collector_requeue_exit=requeue['wrapper_exit'],
        limits='Transport/CLI/dependency rehearsal. All scientific children, imports, cgroups and GPUs are synthetic; not scientific qualification.')
    (root/'wrapper-walk.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in results))
    c.write(root/'summary.json',summary)
    (root/'inventory.md').write_text(c.inventory(doc))
    return summary


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--bridge-and-standin':return bridge(sys.argv[2:])
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--standin',action='store_true',required=True);a=p.parse_args()
    print(json.dumps(run(a.out),indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
