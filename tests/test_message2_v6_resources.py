"""CPU-only tests for right-sized handoffs, admission and checkpoint commands."""
import copy,json,os,sys,threading
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from kit import m2_manifest,m2_node as node,m2_snapshot as snap,m2_assignment as assignment,m2_lane as lane,v6_infra


@pytest.fixture(autouse=True)
def external_test_launch_seal(monkeypatch):
    from kit import v6_infra
    monkeypatch.setenv(v6_infra.ENV,v6_infra.authority_sha256())


def settings(tmp_path):
    path=tmp_path/'spec.json';path.write_text(json.dumps(dict(stages=[dict(id=f'D-{s}',incoming='base',seed=s) for s in (101,102,103)],nodes=dict(base=dict(root=True)),v4_step20=[])))
    return dict(nodes=['node0'],trainer_python=sys.executable,inference_python=sys.executable,bind={},next_source='/next',pools={},prepare={},screen=dict(spec=str(path),sources='/sources'),addon=dict(spec='/addon',smoke='/smoke'))


def test_every_core_row_requests_resources_and_consumers_wait_for_publication(tmp_path):
    doc=m2_manifest.build(settings(tmp_path));path=tmp_path/'launch.json';snap.atomic(path,doc);node.load_manifest(path)
    tasks={t['id']:t for t in doc['tasks']}
    for task in doc['tasks']:
        r=task['resources'];assert r['nodes']==1 and r['gpus'] in (0,1,2,8) and r['cpus']>0
        assert r['share_node']==(r['gpus']!=8)
        if task.get('execution_phase')=='gpu':
            key=task['workspace_task'];assert task['requires']==[key+'-stage']
            assert tasks[key]['resources']['gpus']==0 and tasks[key]['requires']==[key+'-gpu']
            assert tasks[key+'-stage']['resources']['gpus']==0
            assert node.assigned_task(tasks[key],0,1)==node.assigned_task(task,0,1)
    assert tasks['PF-node-0']['resources']['gpus']==tasks['PF-ARC']['resources']['gpus']==0
    assert tasks['M3b-gpu']['resources']['gpus']==1 and tasks['M5-gpu']['resources']['gpus']==2
    assert tasks['M6-101-train-gpu']['resources']['gpus']==8
    assert 'M6-101-train' in tasks['M6-101-score-stage']['requires']
    assert tasks['P2-qualify-101-gpu']['resources']['gpus']==8
    assert 'P2-qualify-101' in tasks['X6-p-101-train-stage']['requires']
    for i in range(3):assert node.assigned_task(tasks['P2-qualify-101-gpu'],i,3)==node.assigned_task(tasks['X6-p-101-train-gpu'],i,3)


def close(leases):
    for lease in leases:lease.close()


def test_leases_allow_disjoint_gpu_jobs_cpu_takes_none_and_training_cannot_share(tmp_path):
    assert node.gpu_leases(tmp_path,{})==[]
    first=node.gpu_leases(tmp_path,{'0':'GPU-0'})
    second=node.gpu_leases(tmp_path,{'1':'GPU-1'})
    with pytest.raises(snap.Pending):node.gpu_leases(tmp_path,{'1':'GPU-1'})
    with pytest.raises(snap.Pending):node.gpu_leases(tmp_path,{str(i):'GPU-'+str(i) for i in range(8)})
    close(first);close(second)
    all_gpus=node.gpu_leases(tmp_path,{str(i):'GPU-'+str(i) for i in range(8)})
    with pytest.raises(snap.Pending):node.gpu_leases(tmp_path,{'7':'GPU-7'})
    close(all_gpus)
    assert not (tmp_path/'.allocation.lock').exists()


def test_allocation_requires_exact_count_and_records_scheduler_uuid(monkeypatch):
    from kit import m2_gpu
    monkeypatch.setattr(m2_gpu,'devices',lambda:{str(i):'GPU-'+str(i) for i in range(8)})
    task=dict(resources=dict(gpus=1))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    assert node.allocated_gpus(task,expected=dict(uuid='GPU-3'))=={'3':'GPU-3'}
    assert node.allocated_gpus(task,expected=dict(uuid='GPU-4'))=={'3':'GPU-3'}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3,4')
    with pytest.raises(assignment.Unverified):node.allocated_gpus(task)
    assert node.allocated_gpus(dict(resources=dict(gpus=0)))=={}


def test_allocated_identity_checks_run_in_parallel_before_science(monkeypatch):
    from kit import m2_node_preflight as pf
    gate=threading.Barrier(2)
    expected={str(i):dict(node='node0',uuid='GPU-'+str(i),environment='env') for i in range(2)}
    def run(command,**kwargs):
        gate.wait(timeout=3)
        return NS(stdout=json.dumps(expected[command[-1]]))
    monkeypatch.setattr(pf.subprocess,'run',run)
    proof=pf.probe_allocated({str(i):'GPU-'+str(i) for i in range(2)},dict(assignments={str(i):v for i,v in expected.items()}),'node0')
    assert len(proof)==2 and all(p['uuid_asserted'] for p in proof)


def test_missing_last_checkpoint_prevents_entire_lane_without_model_load(tmp_path,monkeypatch):
    from kit import m2_paths,m2_common
    first=tmp_path/'first';first.mkdir();missing=tmp_path/'last'
    checked=[];monkeypatch.setattr(m2_common,'checked_model',lambda ref,*a,**kw:checked.append(ref['model']))
    monkeypatch.setattr(m2_paths,'local',lambda value:Path(value))
    plan=dict(refs=[(dict(model=str(first)),False),(dict(model=str(missing)),False)])
    with pytest.raises(m2_common.NotReady,match='lane checkpoint pending'):lane.stage(plan,False)
    assert checked==[str(first)]


def miniature(tmp_path,fail_once=False):
    script=tmp_path/'checkpoint_command.py'
    script.write_text("import json,os,sys\nfrom pathlib import Path\nout=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)\nassert os.environ['M2_EXECUTION_PHASE']=='gpu'\n"+("attempt=out/'attempt.json'\nif not attempt.exists():\n attempt.write_text('{}');raise SystemExit(3)\n" if fail_once else '')+"(out/'output.json').write_text(json.dumps({'visible':os.environ['CUDA_VISIBLE_DEVICES']}))\n")
    payload=dict(id='lane',kind='score-family',row='TEST',family='core',seed=101,requires=[],overlay=[],commands=[[sys.executable,str(script),'--out','{out}']],required_outputs=['output.json'])
    tasks=[]
    for phase,suffix,kind,parents,gpus in [('stage','-stage','prepare',[],0),('gpu','-gpu','score-family',['lane-stage'],1),('publish','','prepare',['lane-gpu'],0)]:
        task=copy.deepcopy(payload);task.update(id='lane'+suffix,kind=kind,requires=parents,execution_phase=phase,workspace_task='lane',payload=payload,placement_kind='score-family',placement_id='lane',resources=dict(nodes=1,gpus=gpus,cpus=8,share_node=True));tasks.append(task)
    manifest=tmp_path/'launch.json';snap.atomic(manifest,dict(schema='message2-launch.v2',resource_policy='allocated-gpus.v1',infrastructure_update=v6_infra.identity(),nodes=['node0'],tasks=tasks))
    durable=tmp_path/'durable';assignment.freeze(durable/'scoring-assignment.json',dict(schema='message2-scoring-assignment.v2',assignments={'core|101':dict(node='node0',uuid='CPU-STANDIN',environment='CPU-STANDIN')}));node.freeze_manifest(manifest,durable)
    def run(index):
        task=tasks[index]
        return node.run_task(NS(manifest=manifest,durable=durable,local_root=tmp_path/'local',task=task['id'],action=task['kind'],node_slot='0/1',standin=True))
    return run,durable,tmp_path/'local/lane'


def test_gpu_exit_never_merges_or_publishes_and_cpu_handoff_requeues(tmp_path):
    run,durable,root=miniature(tmp_path)
    assert run(0)['status']=='COMPLETE'
    plan=json.loads((root/'lane-commands.json').read_text());assert len(plan['commands'])==1
    assert run(1)['status']=='COMPLETE'
    assert (root/'out/output.json').exists()
    with pytest.raises(snap.Pending):snap.read_pointer(durable,'lane',completed=True)
    assert run(2)['status']=='COMPLETE'
    assert snap.read_pointer(durable,'lane',completed=True)['complete']
    assert not (root/'out').exists()
    for index in range(3):assert run(index)['status']=='SKIP'


def test_gpu_failure_preserves_partial_state_only_from_cpu_and_requeues(tmp_path):
    run,durable,root=miniature(tmp_path,True)
    run(0)
    with pytest.raises(snap.Pending):run(1)
    with pytest.raises(snap.Pending):snap.read_pointer(durable,'lane')
    with pytest.raises(snap.Pending):run(2)
    assert not snap.read_pointer(durable,'lane')['complete']
    assert run(1)['status']=='COMPLETE' and run(2)['status']=='COMPLETE'


def test_lane_planner_splits_s_checkpoint_domains_and_panel_groups(tmp_path,monkeypatch):
    from kit import s_common
    refs={key:dict(model=key,receipt=key+'.json') for key in ('initial','middle','final')}
    spec=dict(nodes={key:dict(checkpoint=ref,root=True) for key,ref in refs.items()},stages=[dict(id='stage',seed=101,incoming='initial',checkpoints={'20':'middle','40':'final'})])
    monkeypatch.setattr(s_common,'context',lambda *a:dict(spec=spec))
    commands=[[sys.executable,'kit/s_drift.py','--out',str(tmp_path),'--stage','stage'],[sys.executable,'kit/s_score.py','--out',str(tmp_path),'--action','boundaries','--seed','101']]
    plan=lane.plan(commands,tmp_path,dict(seed=101),True)
    assert len(plan['commands'])==9+6 and len(plan['reducers'])==1
    assert all('--step' in c and '--domain' in c for c in plan['commands'][:9])
    assert all('--node' in c and '--score-panel' in c for c in plan['commands'][9:])


def test_native_training_consumes_cpu_plan_once_and_never_serializes_inherited_secrets(tmp_path,monkeypatch):
    from kit import m2_train as train
    out=tmp_path/'task/out';out.mkdir(parents=True);run=out/'runs/stage';run.mkdir(parents=True)
    native=run/'sdft/global_step_40/actor'
    a=NS(out=out,standin=False,sdpo=None,stage='legal',seed=101)
    monkeypatch.setenv('M2_EXECUTION_PHASE','stage');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setenv('PRIVATE_TEST_TOKEN','never serialize')
    train.prepare_training(a,['native-trainer'],dict(os.environ,RUNTIME_OVERRIDE='frozen'),run,[native])
    value=json.loads(train.prepared_path(a).read_text())
    assert value['env']=={'RUNTIME_OVERRIDE':'frozen'}
    calls=[]
    def execute(command,**kwargs):
        calls.append(command);assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='GPU-3'
        native.mkdir(parents=True);(native/'rank0.pt').write_bytes(b'complete native state')
        return NS(returncode=0)
    monkeypatch.setattr(train.subprocess,'run',execute)
    monkeypatch.setenv('M2_EXECUTION_PHASE','gpu');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-3')
    assert train.run_prepared(a) and train.run_prepared(a)
    assert calls==[['native-trainer']] and not list(run.glob('hf-*'))
    monkeypatch.setenv('M2_EXECUTION_PHASE','publish');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    assert train.published_training(a)
    (native/'rank0.pt').unlink();native.rmdir()
    monkeypatch.setenv('M2_EXECUTION_PHASE','gpu')
    with pytest.raises(train.CorrectnessStop,match='native checkpoint missing'):train.run_prepared(a)


def test_cpu_reducer_uses_completed_lane_proof_without_gpu_and_refuses_tampering(tmp_path,monkeypatch):
    path=tmp_path/'scoring-assignment.json'
    expected=dict(node='node0',uuid='GPU-3',environment='frozen')
    assignment.freeze(path,dict(schema='message2-scoring-assignment.v2',assignments={'core|101':expected}))
    proof=dict(assignment_sha256=assignment.sha(path),family='core',seed=101,**expected)
    result=tmp_path/'gpu-result.json';snap.atomic(result,dict(result=0,interrupted=False,quiescent=True,uuids=['GPU-3'],scoring_assignment=proof))
    for k,v in dict(M2_ASSIGNMENT=str(path),M2_FAMILY='core',M2_FAMILY_SEED='101',M2_EXECUTION_PHASE='publish',M2_GPU_RESULT_FILE=str(result),M2_GPU_RESULT_SHA256=assignment.sha(result)).items():monkeypatch.setenv(k,v)
    monkeypatch.setattr(assignment,'inventory',lambda *a:pytest.fail('CPU reducer queried a GPU'))
    assert assignment.active(tmp_path,'3',False)==proof
    doc=json.loads(result.read_text());doc['scoring_assignment']['uuid']='GPU-4';snap.atomic(result,doc)
    with pytest.raises(assignment.Unverified,match='sealed GPU completion'):assignment.active(tmp_path,'3',False)
    monkeypatch.setenv('M2_GPU_RESULT_SHA256',assignment.sha(result))
    with pytest.raises(assignment.Unverified,match='off-assignment constituent'):assignment.active(tmp_path,'3',False)


def test_concurrent_checkpoint_processes_can_bind_common_lane_metadata(tmp_path):
    import subprocess
    path=tmp_path/'scoring-assignment.json'
    assignment.freeze(path,dict(schema='message2-scoring-assignment.v2',assignments={'core|101':dict(node='node0',uuid='CPU-STANDIN',environment='CPU-STANDIN')}))
    program="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from kit.m2_handoff import scoring_identity;scoring_identity(Path(sys.argv[2]),'0',True)"
    env=dict(os.environ,M2_ASSIGNMENT=str(path),M2_FAMILY='core',M2_FAMILY_SEED='101',M2_STANDIN_NODE='node0',M2_EXECUTION_PHASE='gpu')
    children=[subprocess.Popen([sys.executable,'-I','-c',program,str(Path(__file__).resolve().parents[1]),str(tmp_path)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(3)]
    for child in children:
        stdout,stderr=child.communicate(timeout=30)
        assert child.returncode==0,(stdout,stderr)
    value=json.loads((tmp_path/'scoring-families/core-s101.json').read_text())
    assert value['uuid']=='CPU-STANDIN' and not list(tmp_path.rglob('*.tmp'))


@pytest.mark.parametrize('tier2',[False,True])
def test_split_signal_parts_stay_private_and_canonical_return_keeps_assignment(tmp_path,monkeypatch,tier2):
    from kit import x_signal,x_tier2_run,x_common,s_generate,s_drift,m2_handoff,m2_common,x_collect
    family='x-tier2.incremental' if tier2 else 'x-tier1'
    authority=dict(schema='message2-scoring-assignment.v2',assignments={family+'|101':dict(node='node0',uuid='CPU-STANDIN',environment='CPU-STANDIN')})
    assignment.freeze(tmp_path/'scoring-assignment.json',authority)
    proof=dict(assignment_sha256=assignment.sha(tmp_path/'scoring-assignment.json'),family=family,seed=101,**authority['assignments'][family+'|101'])
    monitor=[dict(id=str(i),verified=True,prompt='monitor',response='answer') for i in range(64)]
    doc=dict(domains=['arc'],technical_synthetic=True)
    snap.atomic(tmp_path/'binding.json',doc)
    monkeypatch.setattr(m2_handoff,'scoring_identity',lambda *a:dict(assignment=proof))
    monkeypatch.setattr(m2_common,'checked_model',lambda ref,*a,**kw:dict(model=ref['model']))
    monkeypatch.setattr(s_generate,'selected',lambda *a:dict(monitor=monitor))
    monkeypatch.setattr(x_common,'incoming',lambda *a:dict(model='incoming'))
    monkeypatch.setattr(x_common,'checkpoint',lambda *a:dict(model='updated'))
    def measure(model,*args):return dict(token_mean=1. if model=='incoming' else 1.25,sequence_mean=1.,response_tokens=64)
    monkeypatch.setattr(s_drift,'nll',measure)
    a=NS(out=tmp_path,standin=True,seed=101,score_gpu='0',domain='arc',action='drift')
    if tier2:
        monkeypatch.setattr(x_tier2_run,'incoming',x_common.incoming);monkeypatch.setattr(x_tier2_run,'checkpoint',x_common.checkpoint)
        canonical=tmp_path/'tier2/signals/t2-incremental-control/s101/k10.json'
        def run():return x_tier2_run.measure(a,doc,'t2-incremental-control',101,10,False)
    else:
        monkeypatch.setattr(x_signal,'context',lambda *a:doc);monkeypatch.setattr(x_signal,'disposition',lambda *a:'READY')
        monkeypatch.setattr(x_signal,'selected',s_generate.selected);monkeypatch.setattr(x_signal,'nll',measure)
        monkeypatch.setattr(x_signal,'incoming',x_common.incoming);monkeypatch.setattr(x_signal,'checkpoint',x_common.checkpoint)
        canonical=tmp_path/'signals/s101/arc.json'
        def run():return x_signal.job(a)
    for part in ('before','after'):
        a.measurement=part;run()
    assert not canonical.exists() and len(list((tmp_path/'.lane-parts').rglob('*.json')))==4
    a.measurement='reduce';run()
    value=json.loads(canonical.read_text())
    assert value['inputs']['assignment']==proof
    assert (value['raw_drift']['arc'] if tier2 else value['drift'])==.25
    archive=tmp_path/'return.tar.gz';x_collect.job(NS(out=tmp_path,standin=True,archive=archive))
    files=x_collect.load(archive,True)
    assert not any('.lane-parts' in name or name.endswith(('.before.json','.after.json')) for name in files)
    assignment.validate_files(files,require_assignment=True)
