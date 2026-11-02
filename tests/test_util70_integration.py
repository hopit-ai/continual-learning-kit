"""Allocation/job boundaries using CPU fixtures; no GPU or remote execution."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest
from kit import lane_evidence as evidence, lane_pool, m2_assignment as assignment
from kit import m2_manifest, m2_lane, m3_manifest, m3_score, pd_jobs, ref_guard, ref_score_job


def freeze(tmp_path):
    path=tmp_path/'scoring-assignment.json'
    assignment.freeze(path,dict(schema='message2-scoring-assignment.v2',assignments={
        'core|101':dict(node='old-host',uuid='GPU-old',environment='old-env')}))
    return path


def test_allocated_identity_records_actual_gpu_keeps_historical_bytes_and_rejects_tampering(tmp_path,monkeypatch):
    path=freeze(tmp_path);original=path.read_bytes()
    monkeypatch.setenv('M2_ASSIGNMENT',str(path));monkeypatch.setenv('KIT_LANE_JOB','a'*32)
    monkeypatch.setenv('M2_ALLOCATION_PROOF',str(tmp_path/'actual.json'))
    monkeypatch.setattr(assignment,'inventory',lambda *a:dict(node='new-host',uuid='GPU-new',environment='new-env'))
    proof=assignment.active(tmp_path,'0',True,family='core',seed=101)
    assert proof['uuid']=='GPU-new' and proof['allocation_job']=='a'*32
    assert path.read_bytes()==original and assignment.verify(dict(assignment=proof),original)=='core|101'
    changed=copy.deepcopy(proof);changed['historical_assignment']['uuid']='GPU-forged'
    with pytest.raises(assignment.Unverified):assignment.verify(dict(assignment=changed),original)
    monkeypatch.setattr(assignment,'inventory',lambda *a:dict(node='new-host',uuid='GPU-other',environment='new-env'))
    with pytest.raises(assignment.Unverified,match='environment changed'):assignment.active(tmp_path,'0',True,family='core',seed=101)
    monkeypatch.delenv('KIT_LANE_JOB')
    with pytest.raises(assignment.Unverified,match='off-assignment'):assignment.active(tmp_path,'0',True,family='core',seed=101)


def test_same_uuid_in_different_jobs_is_unverified_for_generated_and_reference_contrasts():
    rows=[dict(seed=101,_allocation_job=j,_allocation_uuid='GPU-one',_allocation_environment='env') for j in ('a'*32,'b'*32)]
    assert evidence.hardware_status(rows)=='UNVERIFIED-HARDWARE'
    rows[1]['_allocation_job']='a'*32
    assert evidence.hardware_status(rows) is None
    reference=[dict(panel='knowledge',seed=101,step=40,_guard_method=ref_guard.METHOD,
        _guard_uuid='GPU-one',_guard_environment='env',_guard_job=j) for j in ('a'*32,'b'*32)]
    assert ref_guard.check(reference)['status']=='UNVERIFIED-HARDWARE'


def score_set(out,key,n):
    root=out/'scores'
    for i in range(n):
        p=root/('checkpoint-'+str(i));p.mkdir(parents=True)
        (p/'m2-identity.json').write_text(json.dumps(dict(assignment=dict(allocation_job=key))))
        (p/'scores.csv').write_text(key+'\n')
    evidence.archive(out,key,'core',101)


def test_complete_later_lane_selects_its_own_baselines_preserves_previous_jobs_and_hashes(tmp_path):
    score_set(tmp_path,'early',1);score_set(tmp_path,'complete',3)
    files={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    selected=evidence.select(files)
    assert selected['scores/checkpoint-0/scores.csv']==b'complete\n'
    assert len([p for p in selected if p.endswith('/scores.csv')])==3
    evidence.materialize(tmp_path)
    assert (tmp_path/'scores/checkpoint-0/scores.csv').read_text()=='complete\n'
    evidence.clear_view(tmp_path)
    assert not (tmp_path/'scores').exists() and (tmp_path/'lane-evidence/early/lane.json').exists()
    files['lane-evidence/complete/scores/checkpoint-0/scores.csv']=b'tampered'
    with pytest.raises(ValueError,match='changed'):evidence.select(files)


def test_competing_equal_coverage_jobs_never_select_by_timestamp(tmp_path):
    score_set(tmp_path,'first',2);score_set(tmp_path,'second',2)
    files={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='UNVERIFIED-HARDWARE'):evidence.select(files)


def test_reference_both_panels_require_complete_cpu_preparation_before_pool(tmp_path,monkeypatch):
    a=NS(out=tmp_path,family='core',seed=101,panel='both',checkpoints=tmp_path/'checkpoints.json',gpu='0')
    a.checkpoints.write_text(json.dumps([dict(arm='base',seed=101,step=0)]))
    monkeypatch.setattr(lane_pool,'execute',lambda *a,**kw:pytest.fail('missing panel admitted'))
    with pytest.raises(Exception,match='CPU preparation missing'):ref_score_job.run_lane(a)


def test_reference_both_panels_expand_each_checkpoint_once(tmp_path,monkeypatch):
    a=NS(out=tmp_path,family='core',seed=101,panel='both',checkpoints=tmp_path/'checkpoints.json',gpu='0',standin=False)
    checkpoints=[dict(arm='base',seed=101,step=0),dict(arm='legal-generic',seed=101,step=40)]
    a.checkpoints.write_text(json.dumps(checkpoints));(tmp_path/ref_guard.ASSIGNMENT).write_text('{}')
    for panel in ref_guard.PANELS:
        child=copy.copy(a);child.panel=panel;path=ref_score_job.prepared_path(child);path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(checkpoints=checkpoints,assignment_sha256=ref_score_job.s.file_hash(tmp_path/ref_guard.ASSIGNMENT),
            sources=[dict(source={},observed={},metadata={})]*2)))
    monkeypatch.setattr(ref_score_job,'model_metadata',lambda *a:{})
    calls=[];monkeypatch.setattr(lane_pool,'execute',lambda commands,*a,**kw:calls.extend(commands) or 0)
    seals=[];monkeypatch.setattr(evidence,'complete_attempt',lambda *a:seals.append(a))
    ref_score_job.run_lane(a)
    assert len(calls)==2
    assert len(seals)==1 and len(seals[0][2])==4
    assert [(c[c.index('--panel')+1],c[c.index('--cell-index')+1]) for c in calls]==[('both',str(i)) for i in range(2)]


def test_pd_pooled_cells_keep_one_job_proof_and_full_order(tmp_path,monkeypatch):
    monkeypatch.setattr(pd_jobs,'inputs',lambda *a:{})
    job=dict(phase='post',identity_sha256='job',gpu_identity=dict(uuid='GPU-one'))
    (tmp_path/'post').mkdir()
    requests=[(dict(checkpoint_id=str(i)),'model','receipt','start',i) for i in range(18)]
    def execute(commands,*a,**kw):
        assert len(commands)==18
        for command in commands:
            task=Path(command[command.index('--score-cell')+1]);doc=json.loads(task.read_text())
            assert doc['job']==job
            pd_jobs.write_frozen(task.with_suffix('.result.json'),pd_jobs.seal(dict(index=doc['update'])))
        return 0
    monkeypatch.setattr(lane_pool,'execute',execute)
    cells=pd_jobs.scoring_cells({},tmp_path,{},job,requests,True)
    assert [c['index'] for c in cells]==list(range(18))


def test_new_core_graph_has_complete_comparisons_and_local_adaptive_baselines(tmp_path):
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(stages=[dict(id='stage',seed=101,incoming='base')],nodes=dict(base=dict(root=True)),v4_step20=[])))
    settings=dict(nodes=['node'],trainer_python=sys.executable,inference_python=sys.executable,bind={},pools={},prepare={},next_source='next',
        screen=dict(spec=str(spec),sources='sources'),addon=dict(spec='spec',smoke='smoke'))
    tasks={t['id']:t for t in m2_manifest.build(settings)['tasks']}
    assert tasks['M8-read']['requires']==[f'M8-compare-{s}' for s in (101,102,103)]
    assert tasks['R-nominate']['requires']==['R0',*[f'R-compare-full-{s}' for s in (101,102,103)]]
    assert tasks['X12']['requires']==[f'X-complete-{s}' for s in (101,102,103)]
    assert len(tasks['X15-complete-101-gpu']['payload']['commands'])>20
    assert 'for part in (\'before\',\'after\')' in Path(m2_lane.__file__).read_text()


def test_kcr_final_jobs_cover_full_registered_curves_and_keep_seed_jobs_separate():
    settings=dict(nodes=['node'],trainer_python='train',inference_python='score',inputs=dict(inventory='hardware',base='base',
        contract_targets='targets',replay='replay',panels={}))
    tasks={t['id']:t for t in m3_manifest.build(settings)['tasks']}
    for seed in (101,102,103):
        task=tasks[f'M3-rescore-{seed}']
        assert task['gpus']==1 and task['command_count']==198 and task['lane_concurrency']==3
        assert all(c[c.index('--seed')+1]==str(seed) for c in task['commands'])
        for arm in ('C0','K','G','CR'):
            assert any(c[c.index('--arm')+1]==arm and c[c.index('--step')+1]=='0' for c in task['commands'])
    assert all('--generation-shard' in c for c in tasks['M3-broad']['commands'])
    assert len(tasks['M3-broad-finish']['commands'])==1 and '--generation-shard' not in tasks['M3-broad-finish']['commands'][0]


def test_five_kcr_reader_fixture_cases_use_independent_reference_guard_cells():
    # The merged builder already corrected the shared helper; keep the method explicit.
    from test_m3_protocol import make_cells
    cells=make_cells()
    guards=[c for key,c in cells.items() if key[-1] in ('hellaswag','knowledge-ll')]
    assert guards and all(c['method']==ref_guard.METHOD for c in guards)


def test_pool_preserves_inherited_lease_descriptors_and_job_identity(tmp_path,monkeypatch):
    tool=tmp_path/'nvidia-smi';tool.write_text('#!'+sys.executable+'\nprint("GPU-fake")\n');tool.chmod(0o755)
    monkeypatch.setenv('PATH',str(tmp_path)+os.pathsep+os.environ['PATH']);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.delenv('M2_ASSIGNED_GPU_UUID',raising=False)
    lease=(tmp_path/'lease').open('w')
    try:
        monkeypatch.setenv('M3_GPU_LOCK_FDS',json.dumps([lease.fileno()]))
        child="import json,os;os.fstat(json.loads(os.environ['M3_GPU_LOCK_FDS'])[0]);print(os.environ['KIT_LANE_JOB'])"
        plan=tmp_path/'plan.json';plan.write_text(json.dumps([dict(argv=[sys.executable,'-I','-c',child])]*3))
        record=lane_pool.run(plan,3,tmp_path/'pool',0)
        assert record['status']=='COMPLETE'
        assert {(tmp_path/'pool'/str(i)/'command.log').read_text().strip() for i in range(3)}=={record['job_id']}
    finally:lease.close()


def test_s_generation_partition_lanes_preserve_selection_and_output_bytes(tmp_path,monkeypatch):
    import shutil
    from kit import s_bind,s_audit,s_freeze,s_generate,s_common,v6_infra
    from kit.screen_s_dryrun import fixtures
    if (Path(__file__).resolve().parents[1]/'.git').exists():
        from scripts.v6_infra_authority import generate
        authority=tmp_path/'cpu-test-authority.json';authority.write_text(json.dumps(generate('main'),sort_keys=True))
        monkeypatch.setattr(v6_infra,'AUTHORITY',str(authority))
    monkeypatch.setenv('M2_INFRA_AUTHORITY_SHA256',v6_infra.identity())
    monkeypatch.delenv('M2_ASSIGNMENT',raising=False)
    sources,spec,message2=fixtures(tmp_path/'fixtures')
    ready=tmp_path/'ready'
    s_bind.job(NS(out=ready,standin=True,message2=message2,sources=sources,spec=spec))
    s_audit.job(NS(out=ready,standin=True));s_freeze.job(NS(out=ready,standin=True))
    solo=tmp_path/'solo';parallel=tmp_path/'parallel';shutil.copytree(ready,solo);shutil.copytree(ready,parallel)
    s_generate.job(NS(out=solo,standin=True,incoming='base',gpu='0'))
    for domain in s_generate.domains_for(s_common.read(spec),'base'):
        for part in ('train','monitor'):
            s_generate.job(NS(out=parallel,standin=True,incoming='base',gpu='0',domain=domain,part=part))
    s_generate.job(NS(out=parallel,standin=True,incoming='base',gpu='0',finalize=True))
    for domain in s_generate.domains_for(s_common.read(spec),'base'):
        assert s_generate.selected(solo,'base',domain)==s_generate.selected(parallel,'base',domain)
        for part in ('train','monitor'):
            assert (solo/'own/base'/domain/(part+'.jsonl')).read_bytes()==(parallel/'own/base'/domain/(part+'.jsonl')).read_bytes()


def test_broad_generation_shards_cover_each_candidate_once_and_keep_raw_bytes(tmp_path):
    import shutil
    from kit import m3_pools
    root=tmp_path/'solo/pools/broad';root.mkdir(parents=True)
    candidates=[dict(id=str(i),source='gsm8k' if i%2 else 'arc',prompt='fixture '+str(i),gold='A',fake_response='Answer: A') for i in range(15)]
    m3_pools.t.write_jsonl(root/'candidates.jsonl',candidates)
    (root/'prepared.json').write_text(json.dumps(dict(files={'candidates.jsonl':m3_pools.s.file_hash(root/'candidates.jsonl')})))
    parallel=tmp_path/'parallel';shutil.copytree(tmp_path/'solo',parallel)
    solo_calls=[];parallel_calls=[]
    m3_pools.broad_job(NS(out=tmp_path/'solo',standin=True),generate=lambda r:solo_calls.append(r['id']) or dict(text=r['fake_response'],finish_reason='stop'))
    for i in range(3):
        m3_pools.broad_job(NS(out=parallel,standin=True,generation_shard=i),generate=lambda r:parallel_calls.append(r['id']) or dict(text=r['fake_response'],finish_reason='stop'))
    assert sorted(solo_calls)==sorted(parallel_calls) and len(set(parallel_calls))==15
    for path in (root/'raw').glob('*.json'):
        assert path.read_bytes()==(parallel/'pools/broad/raw'/path.name).read_bytes()


def test_sp_frozen_code_amendment_requires_an_external_authority_seal(monkeypatch):
    from kit import sp_prepare,v6_infra
    frozen=sp_prepare.RESOURCES/'FROZEN.json';before=frozen.read_bytes()
    monkeypatch.delenv('M2_INFRA_AUTHORITY_SHA256',raising=False)
    with pytest.raises(ValueError,match='code differs'):sp_prepare.verify_frozen()
    monkeypatch.setenv('M2_INFRA_AUTHORITY_SHA256',v6_infra.identity())
    assert sp_prepare.verify_frozen()['code']!=sp_prepare.code_hashes()
    assert frozen.read_bytes()==before


def test_bound_runtime_rejects_helper_drift_and_replaced_ambient_authority(tmp_path,monkeypatch):
    import hashlib
    from kit import v6_infra
    source=v6_infra.ROOT
    for name in (*v6_infra.LANE_CODE,v6_infra.AUTHORITY):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((source/name).read_bytes())
    monkeypatch.setattr(v6_infra,'ROOT',tmp_path)
    bound=v6_infra.runtime_identity();assert v6_infra.verify_runtime(bound)==bound['authority']
    pool=tmp_path/'kit/lane_pool.py';pool.write_bytes(pool.read_bytes()+b'\n# changed runtime\n')
    with pytest.raises(ValueError,match='runtime changed'):v6_infra.verify_runtime(bound)
    authority=tmp_path/v6_infra.AUTHORITY;doc=json.loads(authority.read_text())
    doc['files'].append(dict(path='kit/lane_pool.py',pinned_sha256=bound['files']['kit/lane_pool.py'],
        new_sha256=hashlib.sha256(pool.read_bytes()).hexdigest(),reason='unbound replacement'))
    authority.write_text(json.dumps(doc));monkeypatch.setenv('M2_INFRA_AUTHORITY_SHA256',v6_infra.identity())
    with pytest.raises(ValueError,match='changed after binding'):v6_infra.verify_runtime(bound)


def test_tier_cpu_child_gets_the_manifest_bound_seal(tmp_path,monkeypatch):
    from kit import v6_infra
    monkeypatch.delenv('M2_INFRA_AUTHORITY_SHA256',raising=False)
    runtime=v6_infra.runtime_identity()
    doc=dict(lane_runtime=runtime,technical_synthetic=True,tasks=[dict(id='M3-init',kind='prepare',requires=[],gpus=0,commands=[['python','fake.py']])])
    monkeypatch.setattr(m3_manifest,'load',lambda *a:doc)
    calls=[]
    def run(command,**kw):calls.append(kw['env']);return NS(returncode=0)
    monkeypatch.setattr(m3_manifest.subprocess,'run',run)
    assert m3_manifest.run_task('ignored','M3-init',tmp_path,{})==0
    assert calls[0]['M2_INFRA_AUTHORITY_SHA256']==runtime['authority']


def test_pd_split_identity_includes_shared_lane_runtime(monkeypatch):
    from kit import v6_infra
    before=pd_jobs.implementation_identity();runtime=v6_infra.runtime_identity()
    runtime['files']['kit/lane_pool.py']='0'*64
    monkeypatch.setattr(v6_infra,'runtime_identity',lambda:runtime)
    assert pd_jobs.implementation_identity()!=before


def test_reference_cpu_readiness_preserves_old_jobs_and_rescores_complete_frontier(tmp_path,monkeypatch):
    a=NS(out=tmp_path,family='core',seed=101,panel='hellaswag',standin=True,checkpoints=tmp_path/'list.json')
    a.checkpoints.write_text(json.dumps([dict(arm='base',seed=101,step=0)]))
    (tmp_path/ref_guard.ASSIGNMENT).write_text('{}')
    old=tmp_path/'reference-scores/core/base-s101-k0-hellaswag';old.mkdir(parents=True)
    (old/'ref-identity.json').write_bytes(b'historical bytes')
    other=old.parent/'base-s102-k0-hellaswag';other.mkdir();(other/'ref-identity.json').write_bytes(b'other seed')
    monkeypatch.setattr(ref_score_job,'source_for',lambda *a:({},{}))
    ref_score_job.prepare(a)
    assert not old.exists() and (other/'ref-identity.json').read_bytes()==b'other seed'
    assert [p.read_bytes() for p in (tmp_path/'reference-history').rglob('ref-identity.json')]==[b'historical bytes']
    assert ref_score_job.prepared_path(a).is_file()
    with ref_score_job.lane_writer(a):
        with pytest.raises(Exception,match='already has a writer'):ref_score_job.prepare(a)


def test_reprobe_reference_resolves_from_native_reprobe_checkpoint_binding(tmp_path,monkeypatch):
    from kit import m2_common,m2_reprobe,m2_paths
    reference=dict(model='checkpoint',receipt=None)
    monkeypatch.setattr(m2_common,'binding',lambda *a:dict(purpose=m2_reprobe.REPAIR))
    monkeypatch.setattr(m2_reprobe,'checkpoint',lambda *a:reference)
    monkeypatch.setattr(m2_common,'checked_model',lambda *a,**kw:dict(model_file_hashes={'weights':'hash'}))
    monkeypatch.setattr(m2_paths,'local',lambda p:Path(p) if p else None)
    source,observed=ref_score_job.source_for(tmp_path,'reprobe',dict(arm='legal-generic',seed=101,step=40),False)
    assert source['model']=='checkpoint' and observed['model_file_hashes']=={'weights':'hash'}


def test_core_reference_and_report_dispatch_do_not_need_tier_send_or_tier_files(tmp_path,monkeypatch):
    from kit import combined_dryrun,combined_send
    doc=combined_dryrun.fixtures(tmp_path);doc['technical_synthetic']=False
    for name in ('kcr_manifest','sp_manifest','sp_inventory'):Path(doc['settings'][name]).unlink()
    monkeypatch.setattr(combined_send.socket,'gethostname',lambda:'core-fixture')
    calls=[]
    def run(command,**kw):calls.append(command);return NS(returncode=0)
    assert combined_send.execute(doc,'REF-reprobe-101-prepare',runner=run)==0
    assert combined_send.execute(doc,'REPORT-text',runner=run)==0
    assert len(calls)==3
