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
    from kit import v6_infra
    with patch.dict(os.environ,{v6_infra.ENV:v6_infra.authority_sha256()}):return _fixtures(root)


def _fixtures(root):
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
    slice_dir,b,replay=root/'a4-fixtures/slice',root/'a4-fixtures/B',root/'a4-fixtures/replay'
    s['a4_config']=str(root/'a4-config.json');s['a4_nodes']=['tier-fixture']
    c.write(s['a4_config'],dict(out=str(root/'A4'),slice=str(slice_dir),b_source=str(b),replay=str(replay),
        base='synthetic-base',stage_gpu_hours=6.666666666666667,technical_synthetic=True,
        trainer_python=sys.executable,inference_python=sys.executable,job_python=sys.executable,
        archive=str(root/'a4-return.tar.gz'),fail_task='A4-direct-s101-m1-train'))
    doc=c.build(s);c.write(root/'combined.json',doc)
    durable=Path(s['core_durable']);m2_assignment.freeze(durable/'scoring-assignment.json',authority)
    m2_node.freeze_manifest(s['core_manifest'],durable)
    return doc


def bridge(args):
    """Only the explicit CPU harness installs these child-process stand-ins."""
    from contextlib import ExitStack
    context=c.load(os.environ['COMBINED_CPU_CONTEXT']);doc=c.load(context['manifest']);s=doc['settings']
    ident=context['id'];scenario=context.get('scenario','pass');count=Counter()
    args=[a for a in args if a != '-I']
    if args[0]=='-c':
        return subprocess.run([sys.executable,*args]).returncode
    target=Path(args[0]).name
    def leaf(command, **kwargs):
        count['scientific_children_stood_in']+=1
        if scenario=='broad-pending' and ident=='M3-broad': return NS(returncode=3)
        if scenario=='tier-failure': return NS(returncode=7)
        if ident=='M3-init':m3_manifest.init_root(s['kcr_root'],core_submitted=True)
        if Path(command[2] if len(command)>2 and command[1]=='-I' else command[0]).name=='m3_score.py':
            def option(flag):return command[command.index(flag)+1]
            if '--ready' in command and '--checkpoints-json' in command:
                checkpoints=json.loads(option('--checkpoints-json'))
                keys=[json.dumps(r) for r in checkpoints]
                c.write(option('--ready'),dict(status='READY',technical_synthetic=True,
                    checkpoints={k:dict(source=dict(model='CPU-STANDIN'),observed={}) for k in keys},
                    exports={k:dict(status='PRESENT',required=json.loads(k)[3] not in (10,20,30)) for k in keys}))
            elif '--arm' in command and '--panel-set' in command:
                from kit.m3_score import unit_root
                from kit.p3b_common import files_manifest,atomic
                cell=unit_root(Path(option('--out')),option('--arm'),int(option('--seed')),
                    option('--stage'),int(option('--step')),option('--panel-set'))
                atomic(cell/'m3-identity.json',dict(technical_synthetic=True,assignment=dict(
                    allocation_job=kwargs.get('env',{}).get('KIT_LANE_JOB'),uuid='CPU-STANDIN',environment='CPU-STANDIN')))
                (cell/'scores.csv').write_text('arm,seed,step,panel,question_id,correct\n')
                atomic(cell/'manifest.json',files_manifest(cell,['m3-identity.json','scores.csv'],technical_synthetic=True))

        if ident=='SP-verify-pools' or ident.startswith('SP-prepare-'):
            c.write(Path(s['sp_root'])/'pool-verification.json',dict(technical_synthetic=True))
        return NS(returncode=0)
    if target=='util_watch.py':
        assert int(args[args.index('--expect-gpus')+1])==next(r['gpus'] for r in doc['tasks'] if r['id']==ident)
        marker=Path(context['record']).with_suffix('.watch.json')
        assert not marker.exists(),'nested watcher'
        c.write(marker,dict(technical_synthetic=True,expect_gpus=int(args[args.index('--expect-gpus')+1])))
        child_env=dict(os.environ,UTIL_WATCH_ACTIVE='1')
        return subprocess.run(args[args.index('--')+1:],env=child_env).returncode
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys,'argv',[args[0],*args[1:]]))
        if target=='combined_send.py':
            original_need=c.need
            def need(ok,message):
                if message in ('synthetic inventory is CPU rehearsal only','tier preflight missing/stale'):return
                return original_need(ok,message)
            def dispatch(command,**kwargs):
                if len(command)>1 and command[1]=='-I':command=[os.environ['JOB_PY'],*command[1:]]
                return subprocess.run(command,**kwargs)
            row=next(r for r in doc['tasks'] if r['id']==ident)
            node=row.get('node') or ('core-fixture' if row['group']=='core' or row['group']=='reference' and row.get('family')!='m3' else 'tier-fixture')
            stack.enter_context(patch.object(c.socket,'gethostname',return_value=node))
            stack.enter_context(patch.object(c,'need',side_effect=need))
            return c.execute(doc,ident,runner=dispatch)
        if target=='node_verification.py':
            if 'run' in args:
                return subprocess.run(args[args.index('--')+1:],env=os.environ).returncode
            result=leaf(args).returncode
        elif target=='m2_preflight.py': return 0
        elif target=='combined_preflight.py':
            from kit import combined_preflight as pf
            stack.enter_context(patch.object(pf,'check',return_value=dict(status='PASS',technical_synthetic=True,
                trainer_imports='SYNTHETIC',inference_imports='SYNTHETIC',cgroup='SYNTHETIC',gpus=[])))
            result=pf.main()
        elif target=='m2_node.py':
            task=next(t for t in c.load(s['core_manifest'])['tasks'] if t['id']==ident)
            out=Path(s['core_local_root'])/task.get('workspace_task',ident)/'out'
            class Child:
                def __init__(self, command, **kwargs):
                    count['scientific_children_stood_in']+=1
                    out.mkdir(parents=True,exist_ok=True)
                    for consumer in c.load(s['core_manifest'])['tasks']:
                        for item in consumer.get('imports',[]):
                            if item['task'] in (ident,task.get('workspace_task')):
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
            from kit import m2_lane
            stack.enter_context(patch.object(m2_lane,'plan',side_effect=lambda commands,*a:dict(commands=commands,reducers=[],refs=[],command_count=len(commands),technical_synthetic=True)))
            stack.enter_context(patch.object(m2_node,'gpu_leases',return_value=[]))
            stack.enter_context(patch.object(m2_node.subprocess,'Popen',Child))
            sys.argv.append('--standin');result=m2_node.main()
        elif target=='m3_manifest.py':
            from contextlib import nullcontext
            stack.enter_context(patch.object(m3_manifest,'node_reservation',side_effect=lambda *a,**kw:nullcontext(())))
            stack.enter_context(patch.object(m3_manifest.socket,'gethostname',return_value='tier-fixture'))
            stack.enter_context(patch.object(m3_manifest.subprocess,'run',side_effect=leaf))
            result=m3_manifest.main(args[1:])
        elif target=='a4_jobs.py':
            from kit import a4_jobs
            count['a4_real_dispatches']+=1
            result=a4_jobs.main()
        elif target=='report_text.py':
            from kit import report_text
            result=report_text.main([*args[1:],'--standin','--kit-tag','kit-message-v6b','--kit-commit','b'*40])
        elif target in ('ref_score_job.py','pd_jobs_read.py'):
            result=leaf(args).returncode
        elif target=='sp_manifest.py':
            from contextlib import nullcontext
            stack.enter_context(patch.dict(os.environ,UTIL_WATCH_ACTIVE='1'))
            stack.enter_context(patch.object(m3_manifest,'allocated_gpus',return_value=['CPU-STANDIN']))
            stack.enter_context(patch.object(m3_manifest,'allocation_reservation',side_effect=lambda *a,**kw:nullcontext()))
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
            row=next(r for r in doc['tasks'] if r['id']==ident)
            from kit import pd_node as pd
            def phase(*a,**kw):
                count['scientific_children_stood_in']+=1
                if ident.endswith('-collect'):c.write(Path(s['pd_root'])/('seed-'+str(row['seed']))/'seed-evidence.verified.json',dict(technical_synthetic=True))
                return 7 if scenario=='tier-failure' else 0
            stack.enter_context(patch.object(pd,'run_phase',side_effect=phase))
            result=pd.main()
        else:raise AssertionError('unexpected wrapper child '+target)
    c.write(Path(context['record']),dict(id=ident,result=result,counts=dict(count),technical_synthetic=True))
    return result


def tree_hash(root):
    h=hashlib.sha256()
    for p in sorted(Path(root).rglob('*')):
        if p.is_file():h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def run(root, *, resume_core=False):
    root=Path(root).resolve()
    if not resume_core:root.mkdir(parents=True,exist_ok=False)
    if resume_core:
        assert not (root/'summary.json').exists(),'completed rehearsal; use a fresh output tree'
        doc=c.load(root/'combined.json');s=doc['settings']
        assert doc['technical_synthetic'] and not (Path(c.load(s['a4_config'])['out'])/'binding.json').exists()
        assert all(c.sha(s[k])==v for k,v in doc['sources'].items())
        from kit.v6_infra import verify_runtime
        verify_runtime(c.load(s['core_manifest'])['lane_runtime'])
        shim=root/'cpu-python';binary=root/'bin'
        shutil.rmtree(s['kcr_root'],ignore_errors=True)
    else:
        doc=fixtures(root);s=doc['settings'];results=[]
        from kit import a4_sim
        slice_dir,b,replay=a4_sim.fixture(root/'a4-fixtures','D')
        config=c.load(s['a4_config']);config.update(slice=str(slice_dir),b_source=str(b),replay=str(replay));c.write(s['a4_config'],config)
        doc['sources']['a4_config']=c.sha(s['a4_config'])
        shim=root/'cpu-python'
        shim.write_text('#!/bin/sh\nif [ "${1:-}" = -I ] && [ "${2:-}" = -c ]; then exec '+__import__('shlex').quote(sys.executable)+' "$@"; fi\nexec '+__import__('shlex').quote(sys.executable)+' '+__import__('shlex').quote(str(Path(__file__).resolve()))+' --bridge-and-standin "$@"\n');shim.chmod(0o755)
        binary=root/'bin';binary.mkdir()
        smi=binary/'nvidia-smi'
        smi.write_text('#!'+sys.executable+'\nimport os,sys\n'
            'q=" ".join(sys.argv)\n'
            'if "query-compute-apps" in q:pass\n'
            'elif "index,uuid,utilization" in q:print("\\n".join(f"{i}, GPU-CPU-STANDIN-{i}, 90, 1024" for i in range(8)))\n'
            'elif "index,uuid" in q:print("\\n".join(f"{i}, GPU-CPU-STANDIN-{i}" for i in range(8)))\n'
            'else:print("GPU-CPU-STANDIN-"+os.environ.get("CUDA_VISIBLE_DEVICES","0").split(",")[0])\n')
        smi.chmod(0o755)
    results=[]
    env=c.environment(s);env['PATH']=str(binary)+os.pathsep+env['PATH'];env['COMBINED_MANIFEST']=str(root/'combined.json');env.update(JOB_PY=str(shim),PYTHONDONTWRITEBYTECODE='1',SP_INPUTS=s['core_local_root']+' '+s['kcr_root'])
    s['tier_send']=dict(status='SEND',date='2026-11-02',review_sha256=c.sha(c.REVIEW))
    c.write(root/'combined.json',doc)
    c.write(Path(s['control_root'])/'core-submitted.json',dict(core_manifest_sha256=doc['sources']['core_manifest']))
    context=root/'context.json';env['COMBINED_CPU_CONTEXT']=str(context)
    def invoke(row,scenario='pass'):
        ident=row['id'];record=root/'records'/(scenario+'-'+ident+'-'+str(__import__('time').time_ns())+'.json')
        c.write(context,dict(manifest=str(root/'combined.json'),id=ident,scenario=scenario,record=str(record)))
        group=row['group']
        command=['bash',str(c.KIT/'jobs/combined/run-task.sh'),ident]
        done=subprocess.run(command,env=env,capture_output=True,text=True)
        with (root/'wrappers.log').open('a') as log:log.write(ident+' '+scenario+' exit='+str(done.returncode)+'\n'+done.stdout+done.stderr)
        result=c.load(record) if record.exists() else dict(id=ident,result=done.returncode,counts={})
        result.update(group=group,wrapper_exit=done.returncode,scenario=scenario,combined_wrapper=True,watchers=int(record.with_suffix('.watch.json').exists()))
        return result
    # Core manifest is topological; preflight first. INPUTS is a stand-in bundle row here.
    rows=doc['tasks']
    by_id={r['id']:r for r in rows}
    coretasks=c.load(s['core_manifest'])['tasks']
    coreorder=[t for t in coretasks if t['id'].startswith('PF-node-')]+[t for t in coretasks if not t['id'].startswith('PF-node-')]
    ordered=[by_id[t['id']] for t in coreorder]+[r for r in rows if r['group']=='preflight']+[r for r in rows if r['group'] in ('kcr','sp','pd','reference','4a','report')]
    pending_4a=invoke(by_id['A4-arrival-m1'],'a4-pending')
    assert pending_4a['wrapper_exit']==75
    failed_4a=[]
    for row in ordered:
        if row['id']=='A4-direct-s101-m1-train':
            failed_4a=[invoke(row,'a4-failure'),invoke(by_id['A4-direct-s101-m1-export'],'a4-pending-export')]
            assert [r['wrapper_exit'] for r in failed_4a]==[1,75]
        if resume_core and row['group'] in ('core','preflight'):
            candidates=[]
            for path in (root/'records').glob('pass-'+row['id']+'-*.json'):
                if path.name.endswith('.watch.json'):continue
                value=c.load(path)
                if value.get('id')==row['id'] and value.get('result')==0 and path.with_suffix('.watch.json').exists():
                    candidates.append((path,value))
            assert candidates,'no successful wrapper record for '+row['id']
            _,result=max(candidates,key=lambda v:v[0].stat().st_mtime_ns)
            result.update(group=row['group'],wrapper_exit=0,scenario='pass',combined_wrapper=True,watchers=1)
        else:result=invoke(row)
        results.append(result)
        (root/'wrapper-walk.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in results))
        if result['wrapper_exit']!=0 or result['result']!=0 or result['watchers']!=1:raise AssertionError(result)
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
    failure_rows=[next(r for r in rows if r['id']==i) for i in ('M3-K-101-finqa-train','PD-101-train','SP-length')]
    failures=[invoke(r,'tier-failure') for r in failure_rows]
    assert [f['wrapper_exit'] for f in failures]==[1,7,0],failures
    assert c.load(Path(s['sp_root'])/'state/SP-length.json')['status']=='FAILED'
    assert tree_hash(s['core_durable'])==before and tree_hash(s['core_local_root'])==before_local
    # Real first-node wrapper gate: completed core outputs remain immutable on requeue.
    requeue=invoke(next(r for r in rows if r['id']=='M-core-collect'));assert requeue['wrapper_exit']==0
    summary=dict(technical_synthetic=True,gpu_hours=0,rows=len(rows),groups=dict(Counter(r['group'] for r in rows)),
        successful_wrapper_rows=len(results),every_group_through_combined_wrapper=True,watcher_attempts=len(list((root/'records').glob('*.watch.json'))),children_stood_in=sum(r['counts'].get('scientific_children_stood_in',0) for r in results),
        a4_pending_resubmitted=True,a4_failed_training_recovered=True,a4_failure_attempts=failed_4a,
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
