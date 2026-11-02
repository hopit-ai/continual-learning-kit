#!/usr/bin/env python3
"""Run one explicitly selected plain job. sbatch dependencies belong to the partner."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from kit import m2_assignment as assignment, m2_snapshot as snapshots
from kit.m2_select import slot

class ChildCrash(RuntimeError):
    pass


ACTIONS=('prepare','train-stage','score-family','reduce-admission','collect')


def resources(task):
    if 'resources' in task:return task['resources']
    return task.get('resources',dict(nodes=1,gpus=8 if task['kind']=='train-stage' else 1 if task['kind']=='score-family' or task.get('uses_gpu') else 0,cpus=4,share_node=task['kind']!='train-stage'))


def seal_environment(env,doc):
    # Only the durably bound launch supplies trust, including for old sends.
    seal=doc.get('infrastructure_update')
    for name in ('KIT_V6_AUTHORITY_SHA','M2_INFRA_AUTHORITY_SHA256'):
        env.pop(name,None)
        if seal:env[name]=seal
    return env


def begin_preparations(root):
    for path in (root/'.gpu-work').glob('*.json'):
        if not path.name.endswith('.done.json') and not path.with_suffix('.done.json').exists():path.unlink()
    active=root/'prepared-current.json';snapshots.atomic(active,[])
    return active


def allocated_gpus(task,standin=False,expected=None):
    count=resources(task)['gpus']
    if not count:return {}
    if standin:return {str(i):expected['uuid'] if expected and count==1 else 'CPU-STANDIN-'+str(i) for i in range(count)}
    from kit.m2_gpu import devices
    inventory=devices();visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    assignment.require(visible is not None,'GPU jobs require scheduler CUDA_VISIBLE_DEVICES')
    tokens=visible.split(',');selected={i:u for i,u in inventory.items() if i in tokens or u in tokens}
    assignment.require(len(tokens)==len(selected)==count,'allocation GPU count differs from resource request')
    return selected


def allocated_idle(devices):
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=5)
    if any(line.split(',')[0].strip() in devices.values() for line in processes.splitlines()):
        raise snapshots.Pending('allocated GPU still has active GPU work')


def gpu_leases(local_root,devices,*,check_idle=False):
    """All-or-nothing nonblocking UUID leases; CPU jobs have no GPU lease."""
    # Synthetic CPU fixtures must not contend with another worktree's fixtures.
    directory=Path(local_root)/'.synthetic-gpu-leases' if devices and all(u.startswith('CPU-STANDIN') for u in devices.values()) else Path('/tmp')/'kit-gpu-leases'
    directory.mkdir(parents=True,exist_ok=True)
    leases=[]
    try:
        for uuid in sorted(set(devices.values())):
            assignment.require('/' not in uuid and uuid not in ('.','..'),'unsafe GPU UUID')
            lease=(directory/(uuid+'.lock')).open('a');leases.append(lease)
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if devices and check_idle:allocated_idle(devices)
        return leases
    except BlockingIOError:
        for lease in leases:lease.close()
        raise snapshots.Pending('allocated GPU busy; supply sbatch dependencies')
    except BaseException:
        for lease in leases:lease.close()
        raise


def load_manifest(path):
    doc=json.loads(Path(path).read_text())
    assignment.require(doc.get('schema')=='message2-launch.v2','launch manifest schema')
    tasks=doc['tasks']
    assignment.require(len({t['id'] for t in tasks})==len(tasks),'duplicate task IDs')
    ids={t['id'] for t in tasks}
    for task in tasks:
        snapshots.safe_key(task['id'])
        commands=task.get('commands',[task.get('command')])
        assignment.require(task['kind'] in ACTIONS and bool(commands) and all(isinstance(cmd,list) and cmd and all(isinstance(v,str) for v in cmd) for cmd in commands),'invalid plain job')
        if task.get('commit_barrier')=='M9':assignment.require(task['kind']=='reduce-admission' and task.get('row')=='M9','only M9 reducer may commit admission')
        assignment.require(set(task.get('requires',[]))<=ids,'unknown prerequisite')
        assignment.require(task['id'] not in task.get('requires',[]),'self prerequisite')
        if doc.get('resource_policy'):
            request=resources(task)
            assignment.require('resources' in task and request['nodes']==1 and request['gpus'] in (0,1,2,8) and
                isinstance(request['cpus'],int) and request['cpus']>0 and request['share_node'] is (request['gpus']!=8),'invalid explicit resources')
        if task.get('placement_kind',task['kind'])=='score-family':
            assignment.require(task.get('family') and task.get('seed') is not None,'scorer requires family/seed')
        if task.get('partial_inputs'):assignment.require(task['kind'] in ('collect','reduce-admission'),'only reducers/collectors may refresh partial inputs')
        if task['id']=='M10' or task.get('row') in ('M10','X1'):
            assignment.require('M9' in task.get('barriers',[]),'M9 durable barrier required')
        if task.get('row')=='X1':
            assignment.require({'M10','S6'}<=set(task.get('barriers',[])),'X1 M10/S6 prerequisites required')
        if task.get('row') in ('X14','X15','X16','X17','X18','X19','X20'):
            assignment.require('X12' in task.get('barriers',[]),'X12 durable tier-1 return required')
    # Validate the static DAG without executing it or deciding which task to run.
    remaining={t['id']:set(t.get('requires',[])) for t in tasks}
    while remaining:
        ready={k for k,v in remaining.items() if not v}
        assignment.require(bool(ready),'cyclic launch manifest')
        remaining={k:v-ready for k,v in remaining.items() if k not in ready}
    return doc


def bind_slots(doc,authority):
    import copy
    doc=copy.deepcopy(doc);n=len(doc['nodes'])
    assignment.require(n>0 and len(set(doc['nodes']))==n,'unique launch nodes required')
    assignment.require({v['node'] for v in authority['assignments'].values()}<=set(doc['nodes']),'assignment node absent from launch nodes')
    for task in doc['tasks']:
        if task.get('placement_kind',task['kind'])=='score-family':
            key=f"{task['family']}|{task['seed']}"
            assignment.require(key in authority['assignments'],'launch contains unassigned scoring families')
            index=doc['nodes'].index(authority['assignments'][key]['node'])
        else:index=next(i for i in range(n) if assigned_task(task,i,n))
        value=f'{index}/{n}'
        assignment.require(task.get('node_slot',value)==value,'task slot differs from frozen assignment')
        task['node_slot']=value
    return doc


def freeze_manifest(source,durable):
    doc=bind_slots(load_manifest(source),assignment.load(Path(durable)/'scoring-assignment.json'));path=Path(durable)/'launch.json'
    raw=(json.dumps(doc,indent=2,sort_keys=True)+'\n').encode()
    if path.exists():assignment.require(path.read_bytes()==raw,'launch manifest is frozen')
    else:snapshots.immutable(path,raw)
    return path


def assigned_task(task,i,n):
    # Training waves use a stable hash. Scoring tasks belong to their frozen node.
    import hashlib
    if 'preflight_slot' in task:return task['preflight_slot']==i
    return int(hashlib.sha256(task.get('placement_id',task['id']).encode()).hexdigest(),16)%n==i


def barrier(durable,name,expected_assignment):
    path=Path(durable)/'barriers'/(name+'.json')
    if not path.exists():raise snapshots.Pending('durable '+name+' barrier pending')
    doc=json.loads(path.read_text())
    assignment.require(doc['assignment']==expected_assignment,'barrier assignment changed')
    pointer=snapshots.read_pointer(durable,doc['task'],completed=True)
    assignment.require(pointer['sha256']==doc['snapshot'],'durable barrier snapshot changed')
    return doc


class TermState:
    def __init__(self):
        self.child=None;self.interrupted=False;self.publishing=False;self.quiescent=False
    def __enter__(self):
        self.previous=signal.signal(signal.SIGTERM,self.handle)
        return self
    def handle(self,signum,frame):
        self.interrupted=True
        if self.child is not None:
            try:os.killpg(self.child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        elif not self.publishing:
            raise snapshots.Pending('TERM during staging; no child started; requeue task')
    def spawn(self,command,**kwargs):
        self.child=None;self.quiescent=True
        previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM})
        try:
            # The controller is single-threaded. Unblock in the forked child
            # before exec; the parent registers it before unmasking TERM.
            self.child=subprocess.Popen(command,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM}),**kwargs)
            self.quiescent=False
            return self.child
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)
    def wait(self):
        deadline=None
        while True:
            if self.interrupted and deadline is None:deadline=time.monotonic()+float(os.environ.get('M2_TERM_GRACE_SECONDS','10'))
            if deadline is not None and time.monotonic()>=deadline:
                try:os.killpg(self.child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:result=self.child.wait(timeout=10)
                except subprocess.TimeoutExpired as error:raise ChildCrash('child did not quiesce after SIGKILL; no snapshot may be published') from error
                self.quiescent=True
                return result
            try:
                result=self.child.wait(timeout=.2)
                if self.interrupted:
                    try:os.killpg(self.child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                self.quiescent=True
                return result
            except subprocess.TimeoutExpired:pass
    def __exit__(self,*args):signal.signal(signal.SIGTERM,self.previous)


def run_task(a):
    key=snapshots.safe_key(a.task)
    task=next((t for t in load_manifest(a.manifest)['tasks'] if t['id']==key),{})
    key=snapshots.safe_key(task.get('workspace_task',key))
    root=Path(a.local_root)/key;root.mkdir(parents=True,exist_ok=True)
    with (root/'.stage.lock').open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise snapshots.Pending('stage already has a writer: '+key)
        with snapshots.technical_space_budget(a.standin),TermState() as termination:return _run_task(a,termination)


def _run_task(a,termination=None):
    wall_started=time.monotonic()
    doc=load_manifest(a.manifest);i,n=slot(a.node_slot)
    if doc.get('resource_policy'):
        from kit.v6_infra import bound
        bound(doc.get('infrastructure_update'))
    if 'lane_runtime' in doc:
        from kit.v6_infra import verify_runtime
        verify_runtime(doc['lane_runtime'])
    frozen=Path(a.durable)/'launch.json'
    if not frozen.exists():raise snapshots.Pending('launch manifest not durably frozen')
    doc=bind_slots(doc,assignment.load(Path(a.durable)/'scoring-assignment.json'))
    assignment.require(load_manifest(frozen)==doc,'launch manifest differs from frozen prepare')
    assignment.require(n==len(doc['nodes']),'node-slot count differs from frozen launch manifest')
    task=next((t for t in doc['tasks'] if t['id']==a.task),None)
    assignment.require(task is not None,'unknown task selector')
    assignment.require(task['kind']==a.action,'task/action mismatch')
    assignment_path=Path(a.durable)/'scoring-assignment.json'
    assignments=assignment.load(assignment_path);assignment_sha=assignment.sha(assignment_path)
    if task.get('placement_kind',task['kind'])=='score-family':
        expected=assignments['assignments'][f"{task['family']}|{task['seed']}"]
        if expected['node'] not in doc['nodes']:raise snapshots.Pending('assigned scoring node unavailable')
        if doc['nodes'][i]!=expected['node']:return dict(status='NOT_THIS_SLOT',task=task['id'])
    elif not assigned_task(task,i,n):return dict(status='NOT_THIS_SLOT',task=task['id'])
    if not a.standin:
        assignment.require(assignment.platform.node()==doc['nodes'][i],'node-slot hostname differs before staging')
        if not doc.get('resource_policy') and task['kind']=='score-family' and not task.get('execution_phase'):
            __import__('kit.m2_gpu',fromlist=['gpu_index']).gpu_index(expected['uuid'])
    if task.get('execution_phase') in ('gpu','publish'):
        return run_staged(a,task,doc,assignments,assignment_sha,termination)
    stage_task=task if task.get('execution_phase')=='stage' else None
    if stage_task:task=task['payload']
    failed=Path(a.local_root)/task['id']/'gpu-result.json'
    recovery=stage_task and failed.exists() and (json.loads(failed.read_text())['result']!=0 or json.loads(failed.read_text())['interrupted'])
    if stage_task and not recovery and (Path(a.local_root)/task['id']/'staged.json').exists():
        try:
            pointer=snapshots.read_pointer(a.durable,stage_task['id'],completed=True)
            return dict(status='SKIP',task=stage_task['id'],snapshot=pointer['sha256'])
        except snapshots.Pending:pass
    preflight=doc.get('node_preflight_tasks',{}).get(doc['nodes'][i])
    if preflight and task['id']!=preflight:
        try:
            proof=snapshots.read_pointer(a.durable,preflight,completed=True)
            assignment.require(proof['assignment']==assignment_sha,'node preflight assignment changed')
            readiness=Path(a.local_root)/task['id']/'preflight-readiness'
            snapshots.restore_view(a.durable,preflight,proof['sha256'],readiness,prefix='preflight/node.json',expected_manifest=proof['manifest_sha256'])
            receipt=json.loads((readiness/'preflight/node.json').read_text())
            assignment.require(receipt.get('status')=='PASS' and receipt.get('node')==doc['nodes'][i] and
                receipt.get('assignment_sha256')==assignment_sha and receipt.get('technical_synthetic') is a.standin,
                'node preflight real/synthetic type, identity or status differs')
        except snapshots.Pending as error:
            raise snapshots.Pending('first node preflight job pending/failed: '+preflight) from error
    if not a.standin and task.get('row')!='PREFLIGHT':
        from kit.next_blend import local_nvme
        local_nvme(Path(a.local_root),False)
        snapshots.flock_preflight(a.durable,nodes=len(doc['nodes']))
    root=Path(a.local_root)/task['id'];out=root/'out';root.mkdir(parents=True,exist_ok=True)
    parents={}
    for name in task.get('barriers',[]):barrier(a.durable,name,assignment_sha)
    inputs={};parent_complete={};refresh_from=None
    path_map={};origins={};dependencies={};canonical={str(out):'stage://'+task['id']}
    for key in task.get('requires',[]):
        try:pointer=snapshots.read_pointer(a.durable,key,completed=not task.get('partial_inputs'))
        except snapshots.Pending:
            if not task.get('partial_inputs'):raise
            parents[key]=None;parent_complete[key]=False;continue
        assignment.require(pointer['assignment']==assignment_sha,'parent assignment changed')
        parents[key]=pointer['sha256'];parent_complete[key]=pointer['complete'];target=Path(a.local_root)/'.replicas'/pointer['sha256']
        stamp=root/'input-receipts'/(key+'.json')
        mode=task.get('view_mode','evidence')
        with snapshots.lock(target.parent/(pointer['sha256']+'.lock')):
            parent_doc=snapshots.restore_view(a.durable,key,pointer['sha256'],target,mode=mode,expected_manifest=pointer['manifest_sha256'])
        snapshots.atomic(stamp,dict(sha256=pointer['sha256']))
        base=out if key in task.get('overlay',[])+task.get('evidence_from',[]) else target
        aliases={parent_doc['origin']:'',**parent_doc.get('origins',{})}
        aliases[str(target)]=''
        for original,relative in aliases.items():
            path_map[original]=str(base/relative)
            canonical[original]='stage://'+key+'/'+relative
            canonical[str(base/relative)]='stage://'+key+'/'+relative
            if base==out:origins[original]=relative
            else:dependencies[original]=dict(task=key,sha256=pointer['sha256'],path=relative,manifest_sha256=pointer['manifest_sha256'])
        for original,ref in parent_doc.get('dependencies',{}).items():
            dependencies.setdefault(original,ref)
            if base==out:
                original_path=Path(original);origin_path=Path(parent_doc['origin'])
                if origin_path in original_path.parents:
                    dependencies[str(out/original_path.relative_to(origin_path))]=ref
        for relative in parent_doc['files']:
            if not snapshots.selected_file(relative,'evidence'):
                parts=Path(relative).parts
                heavy=next((j for j,v in enumerate(parts) if v.startswith(('hf-','global_step_','checkpoint-'))),len(parts)-1)
                relative=str(Path(*parts[:heavy+1]))
                for base_path in (parent_doc['origin'],str(out)):
                    original=str(Path(base_path)/relative)
                    dependencies[original]=dict(task=key,sha256=pointer['sha256'],path=relative,manifest_sha256=pointer['manifest_sha256'])
        inputs[key]=str(target)
    for original,ref in dependencies.items():
        replica=Path(a.local_root)/'.replicas'/ref['sha256']
        path_map.setdefault(original,str(replica/ref['path']))
        canonical[original]='stage://'+ref['task']+'/'+ref['path']
        canonical[str(replica/ref['path'])]='stage://'+ref['task']+'/'+ref['path']
    identity=dict(task=task,parents=parents,parent_complete=parent_complete,launch=assignment.sha(frozen),assignment=assignment_sha)
    state=root/'inputs.json'
    if state.exists():
        previous_identity=json.loads(state.read_text())
        if previous_identity!=identity:
            assignment.require(task.get('partial_inputs') and all(previous_identity[k]==identity[k] for k in ('task','launch','assignment')) and
                all(not done or previous_identity['parents'][k]==parents[k] for k,done in previous_identity['parent_complete'].items()),'identical-input resume required')
            refresh_from=previous_identity['parents']
            if out.exists():__import__('shutil').rmtree(out)
        snapshots.atomic(state,identity)
    else:snapshots.atomic(state,identity)
    try:
        old=snapshots.read_pointer(a.durable,task['id'])
        if snapshots.compatible_parents(old,parents,parent_complete,task.get('partial_inputs')) and old['parents']!=parents:
            refresh_from=old['parents']
            if out.exists():__import__('shutil').rmtree(out)
        assignment.require((old['parents']==parents or refresh_from==old['parents']) and old['assignment']==assignment_sha,'requeue parent identity changed')
        if old['complete']:
            if task.get('commit_barrier') and not (Path(a.durable)/'barriers'/(task['commit_barrier']+'.json')).exists():
                if out.exists():__import__('shutil').rmtree(out)
                selected='admission.json' if task['commit_barrier']=='M9' else 'collect-complete.json' if task['commit_barrier']=='X12' else None
                if selected:
                    snapshots.restore_view(a.durable,task['id'],old['sha256'],out,prefix=selected,expected_manifest=old['manifest_sha256'])
                else:out.mkdir(parents=True)
                publish_barrier(a.durable,task,out,old,assignment_sha,recovery=True)
            return dict(status='SKIP',task=task['id'],snapshot=old['sha256'])
        if refresh_from is None:
            old_doc=snapshots.header(Path(a.durable)/'snapshots'/task['id']/old['tar'],old['manifest_sha256'])
            for original,ref in old_doc.get('dependencies',{}).items():
                dependencies[original]=ref
                if Path(old_doc['origin']) in Path(original).parents:
                    dependencies[str(out/Path(original).relative_to(old_doc['origin']))]=ref
        if not out.exists() and refresh_from is None:
            restored,_=snapshots.restore(a.durable,task['id'],out,completed=False)
            for original,relative in {restored['origin']:'',**restored.get('origins',{})}.items():
                path_map[original]=str(out/relative);origins[original]=relative
                canonical[original]='stage://'+task['id']+'/'+relative
    except snapshots.Pending:pass
    if not out.exists():
        out.mkdir()
        # Only this stage owns the writable view. No shared CURRENT/OUT writer.
        for key in dict.fromkeys(task.get('overlay',[])+task.get('evidence_from',[])):
            if key not in inputs and task.get('partial_inputs'):continue
            assignment.require(key in inputs,'overlay must be a verified dependency')
            skipped=snapshots.overlay(inputs[key],out,namespace=key,compact=task.get('compact_overlay',False),mode='full' if task.get('view_mode')=='full' else 'evidence')
            for relative in skipped:
                original=str(out/relative)
                dependencies[original]=dict(task=key,sha256=parents[key],path=relative,manifest_sha256=snapshots.read_pointer(a.durable,key)['manifest_sha256'])
    if task.get('view_mode')=='full':
        # Only the selected continuation's native state, not other arms/ancestors.
        selected_commands=task.get('commands',[task.get('command')])
        for command in selected_commands:
            if '--arm' not in command or '--seed' not in command:continue
            wanted='runs/x-'+command[command.index('--arm')+1]+'-s'+command[command.index('--seed')+1]+'/'
            seen=set()
            for ref in dependencies.values():
                relative=ref['path']
                if not (relative==wanted.rstrip('/') or relative.startswith(wanted)) or (ref['sha256'],relative) in seen:continue
                seen.add((ref['sha256'],relative));replica=Path(a.local_root)/'.replicas'/ref['sha256']
                with snapshots.lock(replica.parent/(ref['sha256']+'.lock')):
                    snapshots.restore_view(a.durable,ref['task'],ref['sha256'],replica,prefix=relative,expected_manifest=ref.get('manifest_sha256'))
                source=replica/relative;target=out/relative
                snapshots.space_guard(out,sum(p.stat().st_size for p in source.rglob('*') if p.is_file()) if source.is_dir() else source.stat().st_size)
                if source.is_dir():snapshots.overlay(source,target,mode='full')
                elif source.is_file():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():assignment.require(assignment.sha(source)==assignment.sha(target),'continuation state differs')
                    else:__import__('shutil').copyfile(source,target)
    # P2's registered first-batch transport uses torch.load directly. Its tensor
    # and RNG caches are training inputs, not optimizer state or score evidence.
    # Copy only explicitly declared cache prefixes into this stage's writable
    # tree; never redirect those writes into an immutable parent replica.
    cached=set()
    for ref in dependencies.values():
        relative=ref['path']
        if not any(relative.startswith(prefix) for prefix in task.get('training_cache_prefixes',[])):continue
        if Path(relative).suffix not in ('.pt','.pth') or (ref['sha256'],relative) in cached:continue
        cached.add((ref['sha256'],relative));replica=Path(a.local_root)/'.replicas'/ref['sha256']
        with snapshots.lock(replica.parent/(ref['sha256']+'.lock')):
            snapshots.restore_view(a.durable,ref['task'],ref['sha256'],replica,prefix=relative,expected_manifest=ref.get('manifest_sha256'))
        source=replica/relative;destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assignment.require(assignment.sha(source)==assignment.sha(destination),'training cache differs')
        else:__import__('shutil').copyfile(source,destination)
    # Newly omitted journals and resumed partial shards resolve through exact immutable objects.
    for original,ref in dependencies.items():
        replica=Path(a.local_root)/'.replicas'/ref['sha256']
        path_map[original]=str(replica/ref['path'])
        canonical[original]='stage://'+ref['task']+'/'+ref['path']
        canonical[str(replica/ref['path'])]='stage://'+ref['task']+'/'+ref['path']
    for item in task.get('imports',[]):
        source=Path(inputs[item['task']])/item['source'];destination=out/item['target']
        assignment.require(source.exists(),'missing verified imported input '+str(source))
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source.is_dir():snapshots.overlay(source,destination)
        elif destination.exists():assignment.require(assignment.sha(source)==assignment.sha(destination),'import differs')
        else:__import__('shutil').copyfile(source,destination)
    target=out/'scoring-assignment.json'
    if target.exists():assignment.require(assignment.sha(target)==assignment_sha,'local assignment changed')
    else:target.write_bytes(assignment_path.read_bytes())
    replacements={'out':str(out),'local':str(root),'durable':str(a.durable),'kit':str(Path(__file__).resolve().parent),
        'python':sys.executable,**{'input_'+k:v for k,v in inputs.items()}}
    canonical[str(out)]='stage://'+task['id']
    if task['kind']=='score-family':replacements['score_gpu']='0' if a.standin else expected['uuid']
    else:replacements['score_gpu']=task.get('score_gpu','0')
    def argument(value):
        value=str(value).format_map(replacements)
        for origin,target in sorted(path_map.items(),key=lambda item:-len(item[0])):
            if value==origin or value.startswith(origin+'/'):
                result=target+value[len(origin):]
                # Legacy CLIs may open direct Path arguments before calling any
                # kit reader. Stage that exact prefix, never the full ancestry.
                from kit.m2_paths import materialize
                materialize(Path(result),dependencies=dependencies,replica_root=Path(a.local_root)/'.replicas',durable=a.durable)
                return result
        return value
    if not a.standin and task.get('family') in ('core','reprobe','x-tier1') or not a.standin and task.get('family','').startswith('x-tier2.'):
        from kit.lane_evidence import clear_view
        if stage_task and task['kind']=='score-family':clear_view(out)
    if not stage_task and (out/'lane-evidence').exists():
        from kit.lane_evidence import materialize
        materialize(out)
    commands=[[argument(arg) for arg in cmd] for cmd in task.get('commands',[task.get('command')])]
    snapshots.atomic(root/'dependencies.json',dependencies)
    snapshots.atomic(root/'path-map.json',path_map)
    snapshots.atomic(root/'canonical-roots.json',canonical)
    env=seal_environment(dict(os.environ),doc)
    if not resources(stage_task or task)['gpus']:env['CUDA_VISIBLE_DEVICES']=''
    env.pop('M2_PATH_MAP',None);env.pop('M2_CANONICAL_ROOTS',None)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',M2_ASSIGNMENT=str(target),
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),M2_PATH_MAP_FILE=str(root/'path-map.json'),
        M2_DEPENDENCIES_FILE=str(root/'dependencies.json'),M2_CANONICAL_ROOTS_FILE=str(root/'canonical-roots.json'),M2_DURABLE=str(a.durable),M2_REPLICA_ROOT=str(Path(a.local_root)/'.replicas'))
    if task['kind']=='score-family' and not a.standin:env['M2_ASSIGNED_GPU_UUID']=expected['uuid']
    if task.get('family'):
        env.update(M2_FAMILY=task['family'],M2_FAMILY_SEED=str(task['seed']))
    if a.standin:
        env['M2_STANDIN_NODE']=doc['nodes'][i]
        env['M2_TECHNICAL_STANDIN']='1'
        for command in commands:
            if len(command)>1 and command[1].startswith(str(Path(__file__).resolve().parent)+'/') and '--standin' not in command:command.append('--standin')
    if stage_task:
        # Only CPU staging materialises the complete lane before GPU admission.
        active=begin_preparations(root)
        env.update(M2_PREPARED_WORK_FILE=str(active),M2_EXECUTION_PHASE='stage',M2_MODEL_HASH_RECEIPTS_FILE=str(root/'model-hashes.json'))
        env.pop('M2_MODEL_HASH_RECEIPTS_SHA256',None)
        lane=None
        if resources(task)['gpus']==1:
            from kit.m2_lane import plan,stage
            previous=dict(os.environ)
            try:
                os.environ.update(env)
                if recovery and not a.standin:
                    failed_proof=json.loads(failed.read_text()).get('scoring_assignment') or {}
                    old_job=failed_proof.get('allocation_job')
                    if old_job:
                        import shutil
                        scratch=out/'.lane-parts'
                        if scratch.exists():shutil.rmtree(scratch)
                        for path in out.rglob('*.json'):
                            if 'lane-evidence' in path.relative_to(out).parts:continue
                            try:value=json.loads(path.read_text())
                            except (OSError,ValueError):continue
                            if not isinstance(value,dict):continue
                            proof=value.get('assignment') or (value.get('inputs') or {}).get('assignment') if isinstance(value.get('inputs',{}),dict) else value.get('assignment')
                            if isinstance(proof,dict) and proof.get('allocation_job')==old_job:path.unlink()
                lane=plan(commands,out,task,a.standin);stage(lane,a.standin)
            except __import__('kit.p3a_common',fromlist=['NotReady']).NotReady as error:raise snapshots.Pending(str(error)) from error
            finally:
                os.environ.clear();os.environ.update(previous)
            snapshots.atomic(root/'lane-commands.json',lane)
        if task['kind']!='score-family' and not a.standin:
            for command in commands:
                if Path(command[1]).stem=='m2_stage' and '--action' in command and command[command.index('--action')+1]=='x-stage-check':continue
                termination.spawn(command,env=dict(env,M2_EXECUTION_PHASE='stage',CUDA_VISIBLE_DEVICES=''),start_new_session=True)
                result=termination.wait()
                if result==3:raise snapshots.Pending('CPU training preparation pending')
                if result:raise ChildCrash('CPU training preparation failed')
        prepared=[Path(p) for p in json.loads(active.read_text())]
        env.pop('M2_PREPARED_WORK_FILE',None)
        if (root/'model-hashes.json').exists():env['M2_MODEL_HASH_RECEIPTS_SHA256']=assignment.sha(root/'model-hashes.json')
        context=dict(task=task,commands=commands,lane=lane,prepared_work={str(p.relative_to(root)):assignment.sha(p) for p in prepared},gpu_skip=(lane is not None and not lane['commands']) or (task['kind']=='train-stage' and not a.standin and not prepared),env={k:v for k,v in env.items() if k.startswith(('M2_','HF_','TRANSFORMERS_','KIT_V6_')) or k in ('CUDA_DEVICE_ORDER','PYTHONPATH')},parents=parents,parent_complete=parent_complete,
            origins=origins,refresh_from=refresh_from,dependencies=dependencies,assignment=assignment_sha,
            launch=assignment.sha(frozen),node=doc['nodes'][i],stage_task=stage_task['id'])
        snapshots.atomic(root/'staged.json',context)
        if context['gpu_skip']:
            snapshots.atomic(root/'gpu-result.json',dict(result=0,interrupted=False,commands=[],command_times=[],seconds=0,gpu_hours=0,
                uuids=[],quiescent=True,skipped=True,scoring_assignment=None,staged_sha256=assignment.sha(root/'staged.json')))
            print('SKIP: no GPU submission needed for',task['id'])
        if recovery:snapshots.atomic(root/'retry-ready.json',dict(previous_result=assignment.sha(failed)))
        ready=root/'ready';snapshots.atomic(ready/'ready.json',dict(task=task['id'],assignment=assignment_sha,context=assignment.sha(root/'staged.json')))
        pointer=snapshots.publish(ready,a.durable,stage_task['id'],parents=parents,assignment=assignment_sha,complete=True)
        return dict(status='COMPLETE',task=stage_task['id'],snapshot=pointer['sha256'])
    leases=[]
    devices=allocated_gpus(task,a.standin,expected if task['kind']=='score-family' else None)
    leases=gpu_leases(a.local_root,devices)
    if task['kind']=='collect':capture_preflights(out,a.durable,doc,a.standin)
    started=time.monotonic();child=None;finished=False
    assignment.require(termination is not None,'early TERM controller required')
    interrupted=termination.interrupted
    try:
        result=3
        for command in commands:
            if termination.interrupted:break
            result=3
            try:child=termination.spawn(command,env=env,start_new_session=True)
            except OSError as exc:
                result=4
                raise ChildCrash('failed to start child: '+str(exc)) from exc
            result=termination.wait()
            if result:break
        if result==0 and not completed_outputs(out,task):result=3
        finished=True
    finally:
        interrupted=termination.interrupted
        if termination.child is not None and not termination.quiescent:
            for lease in leases:lease.close()
            raise ChildCrash('child termination unconfirmed; local state retained without publishing')
        termination.publishing=True
        # The child process group has finished writing before the full-state tar is made.
        snapshots.atomic(root/'attempt.json',dict(commands=commands,interrupted=interrupted,wall_seconds=time.monotonic()-started))
        snapshots.atomic(out/'execution'/(task['id']+'.json'),dict(task=task['id'],assignment=assignment_sha,
            input_staging_seconds=started-wall_started,child_wall_seconds=time.monotonic()-started,result=result,
            technical_synthetic=a.standin,gpu_hours=None,note='Wall time is not GPU-hours; reconcile allocation billing separately.'))
        unresolved=(task.get('commit_barrier')=='M9' and (not (out/'admission.json').exists() or json.loads((out/'admission.json').read_text()).get('status')!='COMPLETE'))
        binding_sha=assignment.sha(out/'binding.json') if (out/'binding.json').exists() else task.get('binding_sha256')
        if task.get('binding_sha256'):assignment.require(binding_sha==task['binding_sha256'],'task binding changed')
        try:
            snapshot_started=time.monotonic()
            if (out/'lane-evidence').exists():
                from kit.lane_evidence import clear_view
                clear_view(out)
            pointer=snapshots.publish(out,a.durable,task['id'],parents=parents,
                binding=binding_sha,assignment=assignment_sha,complete=finished and not interrupted and not unresolved and all(parent_complete.values()) and result==0,origins=origins,refresh_from=refresh_from,dependencies=dependencies,parent_complete=parent_complete,partial_inputs=task.get('partial_inputs',False))
            snapshots.atomic(Path(a.durable)/'timings'/task['id']/(pointer['sha256']+'.json'),dict(task=task['id'],snapshot=pointer['sha256'],
                assignment=assignment_sha,wall_seconds=time.monotonic()-wall_started,input_staging_seconds=started-wall_started,
                snapshot_seconds=time.monotonic()-snapshot_started,
                technical_synthetic=a.standin,gpu_hours=None))
        finally:
            for lease in leases:lease.close()
    if interrupted:raise snapshots.Pending('TERM state saved; requeue identical task '+task['id'])
    if result==2:raise assignment.Unverified('UNVERIFIED: scoring/input identity refused by child '+task['id'])
    if result==3:raise snapshots.Pending(f'child pending; partial snapshot retained for {task["id"]}')
    if result:raise ChildCrash(f'child crashed with exit {result}; partial snapshot retained for {task["id"]}')
    if not pointer['complete']:raise snapshots.Pending('partial evidence returned; prerequisites unresolved for '+task['id'])
    if task.get('commit_barrier'):publish_barrier(a.durable,task,out,pointer,assignment_sha)
    __import__('shutil').rmtree(root/'inputs',ignore_errors=True)
    __import__('shutil').rmtree(out,ignore_errors=True)
    return dict(status='COMPLETE',task=task['id'],snapshot=pointer['sha256'])


def run_staged(a,task,doc,authority,assignment_sha,termination):
    """Consume a CPU-staged local tree; CPU follow-up alone publishes full bytes."""
    key=task['workspace_task'];root=Path(a.local_root)/key;out=root/'out'
    try:
        old=snapshots.read_pointer(a.durable,key,completed=True)
        return dict(status='SKIP',task=task['id'],snapshot=old['sha256'])
    except snapshots.Pending:pass
    if not (root/'staged.json').exists():raise snapshots.Pending('CPU staging missing on this host; requeue staging first')
    context=json.loads((root/'staged.json').read_text());payload=context['task']
    assignment.require(context['assignment']==assignment_sha and context['launch']==assignment.sha(Path(a.durable)/'launch.json') and
        context['node']==doc['nodes'][slot(a.node_slot)[0]] and payload==task['payload'],'staged job identity changed')
    staged=snapshots.read_pointer(a.durable,context['stage_task'],completed=True)
    assignment.require(staged['assignment']==assignment_sha,'staging assignment changed')
    ready=json.loads((root/'ready/ready.json').read_text())
    assignment.require(ready['context']==assignment.sha(root/'staged.json'),'staged context changed')
    for name,checksum in context.get('prepared_work',{}).items():assignment.require(assignment.sha(root/name)==checksum,'prepared training plan changed')
    env=seal_environment(dict(os.environ,**context['env']),doc);result_path=root/'gpu-result.json'
    if task['execution_phase']=='gpu':
        if result_path.exists():
            previous=json.loads(result_path.read_text())
            if previous.get('skipped'):
                assignment.require(previous.get('staged_sha256')==assignment.sha(root/'staged.json'),'skipped GPU result changed')
                print('SKIP: no GPU submission needed for',key)
                return dict(status='SKIP',task=task['id'])
            if previous['result']==0 and not previous['interrupted']:return dict(status='SKIP',task=task['id'])
            if not a.standin:
                retry=root/'retry-ready.json'
                if not retry.is_file() or json.loads(retry.read_text())['previous_result']!=assignment.sha(result_path):raise snapshots.Pending('requeue CPU staging before a fresh complete GPU lane retry')
        expected=authority['assignments'][f"{payload['family']}|{payload['seed']}"] if payload['kind']=='score-family' else None
        devices=allocated_gpus(task,a.standin,expected);leases=gpu_leases(a.local_root,devices,check_idle=not a.standin)
        env.update(CUDA_VISIBLE_DEVICES=','.join(devices),M2_EXECUTION_PHASE='gpu')
        env['M3_GPU_LOCK_FDS']=json.dumps([lease.fileno() for lease in leases])
        if len(devices)==1 and not a.standin:
            env.update(KIT_LANE_JOB=__import__('uuid').uuid4().hex,M2_ASSIGNED_GPU_UUID=next(iter(devices.values())),
                M2_ALLOCATION_PROOF=str(root/('allocation-'+str(time.time_ns())+'.json')))
        commands=__import__('copy').deepcopy([] if context.get('gpu_skip') else context['lane']['commands'] if context.get('lane') else context['commands'])
        checks=[c for c in commands if len(c)>1 and Path(c[1]).stem=='m2_stage' and '--action' in c and c[c.index('--action')+1]=='x-stage-check']
        commands=[c for c in commands if c not in checks]
        described={tuple(c['argv']):c for c in (context.get('lane') or {}).get('pool_commands',[])}
        pool_commands=[]
        for command in commands:
            entry=__import__('copy').deepcopy(described.get(tuple(command),{}))
            for flag in ('--gpu','--gpus','--teacher-gpu','--score-gpu'):
                if flag in command:command[command.index(flag)+1]=','.join(devices) if flag not in ('--score-gpu',) else next(iter(devices))
            if len(command)>1 and Path(command[1]).stem=='m2_run' and '--action' in command and command[command.index('--action')+1]=='targets':command+=['--teacher-gpu',','.join(devices)]
            entry['argv']=command;pool_commands.append(entry)
        start=time.monotonic();result=0 if not commands else 3;finished=False;command_times=[]
        try:
            from kit.m2_node_preflight import probe_allocated
            proof=probe_allocated(devices,authority,context['node'],a.standin)
            snapshots.atomic(out/'preflight'/('gpu-'+key+'.json'),dict(status='PASS',gpus=proof,assignment=assignment_sha,technical_synthetic=a.standin))
            scoring_proof=None
            if expected:
                observed={k:proof[0][k] for k in ('node','uuid','environment')} if not a.standin else expected
                scoring_proof=dict(assignment_sha256=assignment_sha,family=payload['family'],seed=payload['seed'],**observed)
                if not a.standin:scoring_proof.update(allocation_job=env['KIT_LANE_JOB'],historical_assignment=expected)
            if len(devices)==1 and commands and not a.standin:
                from kit.m2_lane import pool_invocations
                commands=pool_invocations(pool_commands,root/'lane-attempts'/str(time.time_ns()))
            for command in commands:
                if termination.interrupted:break
                command_start=time.monotonic()
                try:termination.spawn(command,env=env,start_new_session=True,pass_fds=tuple(lease.fileno() for lease in leases))
                except OSError as error:raise ChildCrash('failed to start GPU child') from error
                result=termination.wait()
                if '--plan' in command and any(Path(c).name=='lane_pool.py' for c in command):
                    pool=Path(command[command.index('--out')+1])/'lane.json'
                    if pool.exists():
                        for row in json.loads(pool.read_text())['commands']:
                            command_times.append(dict(command=row['argv'],seconds=None if row['end'] is None or row['start'] is None else row['end']-row['start'],result=row['exit_code']))
                else:command_times.append(dict(command=command,seconds=time.monotonic()-command_start,result=result))
                if result:break
            finished=True
        finally:
            for lease in leases:lease.close()
            if termination.child is not None and not termination.quiescent:raise ChildCrash('GPU child termination unconfirmed; CPU publication refused')
            if expected and not a.standin and Path(env['M2_ALLOCATION_PROOF']).exists():
                scoring_proof=json.loads(Path(env['M2_ALLOCATION_PROOF']).read_text())
            snapshots.atomic(result_path,dict(result=result if finished else 4,interrupted=termination.interrupted,
                commands=commands,command_times=command_times,seconds=time.monotonic()-start,gpu_hours=0 if a.standin else len(devices)*(time.monotonic()-start)/3600,uuids=list(devices.values()),quiescent=True,staged_sha256=assignment.sha(root/'staged.json'),
                scoring_assignment=scoring_proof if 'scoring_proof' in locals() else None))
        if termination.interrupted or result in (3,75):raise snapshots.Pending('GPU task pending/interrupted; CPU follow-up may preserve partial state')
        if result==2:raise assignment.Unverified('UNVERIFIED: GPU child refused identity')
        if result:raise ChildCrash('GPU child failed; CPU follow-up must preserve partial state')
        return dict(status='COMPLETE',task=task['id'],native_local=True)
    env.update(CUDA_VISIBLE_DEVICES='',M2_EXECUTION_PHASE='publish')
    if not result_path.exists():raise snapshots.Pending('GPU task has not exited; no CPU publication yet')
    env.update(M2_GPU_RESULT_FILE=str(result_path),M2_GPU_RESULT_SHA256=assignment.sha(result_path))
    saved_proof=json.loads(result_path.read_text()).get('scoring_assignment') or {}
    if saved_proof.get('allocation_job'):env['KIT_LANE_JOB']=saved_proof['allocation_job']
    result=json.loads(result_path.read_text());assignment.require(result['quiescent'],'GPU child still writing')
    assignment.require(result.get('staged_sha256')==assignment.sha(root/'staged.json'),'GPU result belongs to another CPU staging pass')
    if result['result']==0 and not result['interrupted']:
        for command in (context.get('lane') or {}).get('reducers',[]):
            termination.spawn(command,env=env,start_new_session=True)
            if termination.wait():raise snapshots.Pending('CPU lane reduction pending')
        if payload['kind']=='train-stage' and not a.standin and not context.get('gpu_skip'):
            for command in context['commands']:
                if Path(command[1]).stem=='m2_stage' and '--action' in command and command[command.index('--action')+1]=='x-stage-check':continue
                termination.spawn(command,env=env,start_new_session=True)
                if termination.wait():raise snapshots.Pending('CPU training finalization pending')
        from kit.m2_train import finish_exports
        previous=dict(os.environ)
        try:
            os.environ.update(env)
            finish_exports(out)
        finally:
            os.environ.clear();os.environ.update(previous)
        for command in context['commands']:
            if len(command)>1 and Path(command[1]).stem=='m2_stage' and '--action' in command and command[command.index('--action')+1]=='x-stage-check':
                termination.spawn(command,env=env,start_new_session=True)
                if termination.wait():raise snapshots.Pending('CPU stage completion check pending')
    if result['result']==0 and not result['interrupted'] and not a.standin and payload['kind']=='score-family' and (payload.get('family') in ('core','reprobe','x-tier1') or payload.get('family','').startswith('x-tier2.')):
        from kit.lane_evidence import archive
        archive(out,key,payload['family'],payload['seed'])
    complete=result['result']==0 and not result['interrupted'] and completed_outputs(out,payload)
    snapshots.atomic(out/'execution'/(key+'.json'),dict(task=key,assignment=assignment_sha,result=result['result'],
        child_wall_seconds=result['seconds'],command_times=result.get('command_times',[]),gpu_uuids=result['uuids'],technical_synthetic=a.standin,gpu_hours=result['gpu_hours']))
    termination.publishing=True
    binding_sha=assignment.sha(out/'binding.json') if (out/'binding.json').exists() else payload.get('binding_sha256')
    pointer=snapshots.publish(out,a.durable,key,parents=context['parents'],binding=binding_sha,assignment=assignment_sha,
        complete=complete,origins=context['origins'],refresh_from=context['refresh_from'],dependencies=context['dependencies'],
        parent_complete=context['parent_complete'],partial_inputs=False)
    # This small CPU-published receipt makes the GPU row's completion durable.
    snapshots.atomic(root/'gpu-receipt/result.json',result)
    snapshots.publish(root/'gpu-receipt',a.durable,key+'-gpu',parents={context['stage_task']:staged['sha256']},assignment=assignment_sha,complete=complete)
    if not complete:raise snapshots.Pending('partial GPU state published; requeue identical GPU task')
    if payload.get('commit_barrier'):publish_barrier(a.durable,payload,out,pointer,assignment_sha)
    __import__('shutil').rmtree(out)
    return dict(status='COMPLETE',task=key,snapshot=pointer['sha256'])


def watched_task(a):
    task=next(t for t in load_manifest(a.manifest)['tasks'] if t['id']==a.task)
    if resources(task)['gpus'] and not a.standin and not os.environ.get('UTIL_WATCH_ACTIVE'):
        root=Path(a.local_root)/task.get('workspace_task',task['id'])
        command=[sys.executable,'-I',str(Path(__file__).resolve()),*sys.argv[1:]]
        code=subprocess.run([sys.executable,'-I',str(Path(__file__).with_name('util_watch.py')),
            '--out',str(root/'utilisation'/(str(time.time_ns())+'.jsonl')),'--expect-gpus',str(resources(task)['gpus']),'--',*command]).returncode
        if code==3:raise snapshots.Pending('GPU task pending')
        if code:raise ChildCrash('watched GPU task failed')
        return dict(status='COMPLETE',task=task['id'])
    return run_task(a)


def capture_preflights(out,durable,manifest,standin=False):
    """Return verified successes/failures/missing gates with every independent return."""
    import tempfile,shutil
    for key in [*manifest.get('node_preflight_tasks',{}).values(),'PF-ARC']:
        status=dict(task=key,status='PENDING',reason='not yet run')
        try:
            pointer=snapshots.read_pointer(durable,key)
            status=dict(task=key,status='PASS' if pointer['complete'] else 'PENDING',snapshot=pointer['sha256'])
            with tempfile.TemporaryDirectory(prefix='m2-preflight-',dir=out.parent) as temp:
                target=Path(temp)/'receipt'
                for prefix in ('preflight','pools/arc-candidate-status.json','pools/arc-candidate-acceptance.json','pools/candidate-manifest.json','execution/'+key+'.json'):
                    snapshots.restore_view(durable,key,pointer['sha256'],target,prefix=prefix,expected_manifest=pointer['manifest_sha256'])
                kind=target/'preflight/node.json' if key.startswith('PF-node-') else target/'pools/candidate-manifest.json'
                if pointer['complete'] and kind.exists() and json.loads(kind.read_text()).get('technical_synthetic') is not standin:
                    status.update(status='UNVERIFIED',reason='real/synthetic preflight evidence differs')
                snapshots.overlay(target,out/'node-preflight'/key)
        except snapshots.Pending:pass
        snapshots.atomic(out/'node-preflight'/key/'status.json',status)


def completed_outputs(out,task):
    for condition in task.get('terminal_conditions',[]):
        path=out/condition['path']
        if path.exists():
            value=json.loads(path.read_text())
            if all(value.get(k)==v for k,v in condition.get('equals',{}).items()):return True
    for condition in task.get('required_values',[]):
        path=out/condition['path']
        if not path.exists():return False
        value=json.loads(path.read_text())
        if not all(value.get(k)==v for k,v in condition['equals'].items()):return False
    return all((out/name).is_file() for name in task.get('required_outputs',[]))


def publish_barrier(durable,task,out,pointer,assignment_sha,*,recovery=False):
    name=task['commit_barrier']
    if name=='M9':
        admission=json.loads((out/'admission.json').read_text())
        if admission.get('status')!='COMPLETE':raise snapshots.Pending('M9 admission unresolved')
    if name=='X12':
        marker=json.loads((out/'collect-complete.json').read_text())
        if recovery:
            # The completed snapshot was fully verified before CURRENT installed.
            # Its immutable manifest binds the archive SHA; restore/hash only the
            # marker now, then consumers verify their selected archive bytes.
            source=Path(durable)/'snapshots'/task['id']/pointer['tar']
            doc=snapshots.header(source,pointer['manifest_sha256'])
            archive_name=Path(marker.get('archive','addon-return.tar.gz')).name
            archives=[r for name,r in doc['files'].items() if Path(name).name==archive_name]
            assignment.require(doc['complete'] and doc['assignment']==assignment_sha and len(archives)==1 and
                archives[0]['sha256']==marker['archive_sha256'],'X12 immutable snapshot archive proof differs')
        else:
            from kit.m2_paths import local
            archive=local(marker.get('archive',out/'addon-return.tar.gz'))
            if not archive.exists():archive=out/archive.name
            assignment.require(archive.exists() and assignment.sha(archive)==marker['archive_sha256'],'X12 archive verification')
    commit=dict(task=task['id'],snapshot=pointer['sha256'],assignment=assignment_sha)
    if name=='M9':commit['admission_sha256']=assignment.sha(out/'admission.json')
    path=Path(durable)/'barriers'/(name+'.json')
    with snapshots.lock(path.parent):
        if path.exists():assignment.require(json.loads(path.read_text())==commit,'immutable admission/return barrier changed')
        else:snapshots.atomic(path,commit)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=ACTIONS,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--task');p.add_argument('--node-slot',default='0/1')
    p.add_argument('--local-root',type=Path,required=True);p.add_argument('--durable',type=Path,required=True)
    p.add_argument('--assignment',type=Path);p.add_argument('--standin',action='store_true')
    a=p.parse_args()
    try:
        if a.action=='prepare' and a.task is None:
            assignment.require(a.assignment is not None,'prepare requires hardware assignment JSON')
            authority=assignment.load(a.assignment);manifest=bind_slots(load_manifest(a.manifest),authority)
            assignment.require(all(f"{t['family']}|{t['seed']}" in authority['assignments'] for t in manifest['tasks'] if t['kind']=='score-family'),'launch contains unassigned scoring families')
            assignment.require({v['node'] for v in authority['assignments'].values()}<=set(manifest['nodes']),'assignment node absent from launch nodes')
            if not a.standin:
                snapshots.flock_preflight(a.durable,nodes=len(manifest['nodes']))
                from kit.next_blend import local_nvme
                local_nvme(a.local_root,False)
            assignment.freeze(a.durable/'scoring-assignment.json',authority)
            freeze_manifest(a.manifest,a.durable)
            print('FROZEN launch and scoring assignment');return 0
        assignment.require(a.task is not None,'one explicit task selector required')
        result=watched_task(a);print(json.dumps(result,sort_keys=True));return 5 if result['status']=='NOT_THIS_SLOT' else 0
    except ChildCrash as exc:print('CRASH:',exc);return 4
    except snapshots.Pending as exc:print('PENDING:',exc);return 3
    except (assignment.Unverified,ValueError,FileNotFoundError) as exc:print(str(exc));return 2


if __name__=='__main__':raise SystemExit(main())
