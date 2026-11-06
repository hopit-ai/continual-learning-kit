"""Staged v6a execution and evidence regressions; labelled CPU inputs only."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
from kit import combined_send as combined,lane_evidence,m3_score,m3_read,m2_gpu,lane_pool,ref_score_job,ref_guard,v6_infra,v6a
from kit.p3b_common import atomic,files_manifest
from scripts import v6a_dryrun as dry


@pytest.fixture
def staged(tmp_path,monkeypatch):
    doc,manifest,shim=dry.fixtures(tmp_path/'fixture')
    monkeypatch.setenv('COMBINED_MANIFEST',str(manifest))
    monkeypatch.setenv('KIT_V6_AUTHORITY_SHA',v6_infra.authority_sha256())
    return doc,manifest,shim


def test_manifest_only_kcr_reference_and_report_no_other_sources(staged):
    doc,_,_=staged;rows=doc['tasks'];ids={r['id'] for r in rows}
    assert len(rows)==14 and set(doc['sources'])=={'kcr_manifest'}
    assert all(r['id'].startswith(('M3-rescore-','REF-m3-','REPORT-text')) for r in rows)
    assert all(set(r['requires'])<=ids for r in rows)
    assert all(r.get('family') in (None,'m3') for r in rows)
    report=next(r for r in rows if r['id']=='REPORT-text')
    assert report['dependency']=='afterany' and set(report['requires'])=={r['id'] for r in rows if r['gpus']}
    assert all(r['lane_concurrency']==2 for r in rows if r['gpus'])
    assert all('--workspace' not in c for r in rows if r['id'].endswith('-ready') for c in r['commands'])
    assert len([c for r in rows if r['group']=='kcr' and r['gpus'] for c in r['commands']])==594
    assert 'commands (K=3' not in combined.inventory(doc)


def test_v6a_dispatch_ignores_core_send_date_preflight_and_unrelated_settings(staged):
    doc,_,_=staged;calls=[]
    def runner(command,**kwargs):calls.append((command,kwargs));return NS(returncode=0)
    assert combined.execute(doc,'REF-m3-assign',runner=runner)==0
    assert len(calls)==1 and '--schedule-checkpoints' in calls[0][0]
    env=calls[0][1]['env']
    assert env['KIT_V6A']=='1' and env['M2_INFRA_AUTHORITY_SHA256']==os.environ['KIT_V6_AUTHORITY_SHA']
    assert not set(env)&{'M2_MANIFEST','MANIFEST','LOCAL_ROOT','DURABLE'}
    assert combined.execute(doc,'REF-m3-101-prepare',runner=runner)==0
    Path(doc['settings']['kcr_manifest']).write_text('{}')
    assert combined.execute(doc,'REF-m3-assign',runner=runner)==1
    state=combined.load(Path(doc['settings']['control_root'])/'state/REF-m3-assign.json')
    assert 'source manifest changed' in state['error']
    assert combined.load(state['attempt_state'])['nonce']==state['nonce']


def test_schedule_checkpoint_list_is_complete_base_and_all_arm_stage_finals(staged):
    doc,_,_=staged;root=Path(doc['settings']['kcr_root']);path=root/'checkpoints.json'
    ref_score_job.assign(NS(out=root,family='m3',from_assignment=None,schedule_checkpoints=True,checkpoints=path))
    cells=json.loads(path.read_text())
    from kit.m3_schedule import SEEDS,ARMS,STAGES
    assert len(cells)==39
    assert {(c['arm'],c['seed'],c['stage'],c['step']) for c in cells}=={
        *[('base',s,STAGES[0],0) for s in SEEDS],*[ (a,s,st,40) for a in ARMS for s in SEEDS for st in STAGES]}
    ref_score_job.assign(NS(out=root,family='m3',from_assignment=None,schedule_checkpoints=True,checkpoints=path))
    assert len(json.loads(path.read_text()))==39


def attempt(parent,nonce,identities=None):
    root=parent/nonce;root.mkdir(parents=True)
    names=[]
    for i,(job,gpu) in enumerate(identities or [(nonce,'GPU-measured')]*2):
        unit=root/'scores'/str(i);name=str((unit/'m3-identity.json').relative_to(root));names.append(name)
        atomic(unit/'m3-identity.json',dict(assignment=dict(allocation_job=job,uuid=gpu,environment='measured-env')))
        atomic(unit/'scores.csv',dict(technical_synthetic=True))
        atomic(unit/'manifest.json',files_manifest(unit,['m3-identity.json','scores.csv']))
    lane_evidence.complete_attempt(root,nonce,names)
    return root


def test_newest_complete_single_nonce_selected_partial_and_mixed_kept_for_audit(tmp_path):
    first=attempt(tmp_path,'a'*32);latest=attempt(tmp_path,'b'*32)
    (tmp_path/('c'*32)).mkdir()  # interrupted after scoring some units
    assert lane_evidence.latest_attempt(tmp_path)==latest and first.exists()
    attempt(tmp_path,'d'*32,[('d'*32,'GPU-A'),('d'*32,'GPU-B')])
    attempt(tmp_path,'e'*32,[('e'*32,'GPU-A'),('f'*32,'GPU-A')])
    assert lane_evidence.latest_attempt(tmp_path)==latest
    (first/'scores/0/scores.csv').write_text('changed')
    assert lane_evidence.latest_attempt(tmp_path)==latest  # old corruption never blocks a fresh job
    (latest/'scores/0/scores.csv').write_text('changed')
    # Only malformed hardware attempts remain; the reader must label these,
    # rather than combining them with other attempts' surviving score files.
    assert lane_evidence.latest_attempt(tmp_path).name=='e'*32


def test_reader_falls_back_to_older_complete_attempt_after_newer_corruption(tmp_path):
    first=attempt(tmp_path,'a'*32);latest=attempt(tmp_path,'b'*32)
    (latest/'scores/0/scores.csv').write_text('changed')
    assert lane_evidence.latest_attempt(tmp_path)==first
    (first/'scores/0/scores.csv').write_text('changed')
    with pytest.raises(ValueError,match='constituent changed'):lane_evidence.latest_attempt(tmp_path)


def test_reader_accepts_sequential_seed_gpu_reuse_but_refuses_mixed_within_seed():
    cells={}
    for seed in (101,102,103):
        for panel in ('finqa','knowledge-ll','hellaswag'):
            cells['K',seed,'finqa',40,panel]=dict(uuid='GPU-reused',environment='measured',allocation_job=f'{seed:032x}',
                reference_assignment=dict(uuid='GPU-reused',environment='measured'),
                method=ref_guard.METHOD if panel in ref_guard.PANELS else 'KIT-SCORED',attempt_scored=True)
    assert not m3_read.identity_failures(m3_read.Table(cells),{})
    cells['K',101,'finqa',40,'hellaswag']['allocation_job']='f'*32
    assert any('mixed reference allocation' in f['reason'] for f in m3_read.identity_failures(m3_read.Table(cells),{}))
    cells['K',102,'finqa',40,'finqa'].pop('allocation_job')
    assert any('missing allocation identity' in f['reason'] for f in m3_read.identity_failures(m3_read.Table(cells),{}))


def test_cpu_ready_is_repeatable_without_creating_score_cells(staged):
    doc,_,_=staged;root=Path(doc['settings']['kcr_root']);workspace=root/'scoring-jobs/M3-rescore-101'
    a=NS(out=root,standin=True,ready=root/'ready.json',workspace=workspace,
         checkpoints_json=json.dumps([['base',101,'finqa',0]]))
    m3_score.prepare_scoring(a);m3_score.prepare_scoring(a)
    assert workspace.is_dir() and not list(workspace.iterdir())


@pytest.mark.parametrize('missing',['none','receipt','model','model-file','final','base'])
def test_cpu_presence_inventory_fixes_lane_and_requires_endpoints(staged,missing):
    import shutil
    from kit.p3b_common import NotReady
    doc,_,_=staged;root=Path(doc['settings']['kcr_root'])
    run=root/'runs/K-s101-finqa'
    if missing in ('receipt','final'):(run/f'export-{40 if missing=="final" else 10}.json').unlink()
    elif missing=='model':shutil.rmtree(run/'hf-step10')
    elif missing=='model-file':(run/'hf-step10/config.json').unlink()
    elif missing=='base':shutil.rmtree(Path(json.loads((root/'inputs/base.json').read_text())['model']))
    row=next(r for r in doc['tasks'] if r['id']=='M3-rescore-101-ready')
    a=NS(out=root,standin=True,ready=root/'ready.json',workspace=None,
         checkpoints_json=row['commands'][0][row['commands'][0].index('--checkpoints-json')+1])
    if missing in ('final','base'):
        with pytest.raises(NotReady,match='required exports missing'):m3_score.prepare_scoring(a)
        assert json.loads(a.ready.read_text())['status']=='PENDING'
        with pytest.raises(NotReady):m3_score.prepared_commands([],a.ready)
        control=Path(doc['settings']['control_root'])
        atomic(control/'state/M3-rescore-101-ready.json',dict(status='DONE'))
        atomic(root/'scoring-ready/M3-rescore-101.local.json',json.loads(a.ready.read_text()))
        assert combined.execute(doc,'M3-rescore-101',runner=lambda *a,**kw:pytest.fail('missing endpoint launched'))==75
        return
    m3_score.prepare_scoring(a)
    ready=json.loads(a.ready.read_text());key=json.dumps(['K',101,'finqa',10])
    assert ready['exports'][key]==dict(status='PRESENT' if missing=='none' else 'MISSING-EXPORT',required=False)
    commands=next(r for r in doc['tasks'] if r['id']=='M3-rescore-101')['commands']
    selected=m3_score.prepared_commands(commands,a.ready)
    assert len(selected)==198-(missing!='none')
    assert all(int(c[c.index('--step')+1]) in (0,10,20,30,40) for c in selected)
    # Adding a pruned export after readiness never changes this attempt's subset.
    if missing=='receipt':
        atomic(run/'export-10.json',dict(model_file_hashes={'config.json':'new'}))
        assert m3_score.prepared_commands(commands,a.ready)==selected


def test_pruned_curves_real_wrappers_keep_primary_verdicts_and_mark_auc_incomplete(tmp_path):
    result=dry.run(tmp_path/'pruned',missing_intermediates=True)
    assert result['native_commands']==450 and result['reference_commands']==39
    assert all(d['verdict']=='INCONCLUSIVE' for d in result['KCR'])


@pytest.mark.parametrize('final_missing',[False,True])
def test_completed_export_validation_records_pruned_curves_and_requires_final(staged,final_missing):
    from kit import m3_train
    from kit.p3b_common import NotReady
    root=Path(staged[0]['settings']['kcr_root']);run=root/'runs/K-s101-finqa'
    (run/'export-10.json').unlink()
    (run/'hf-step20/config.json').unlink()
    if final_missing:(run/'hf-step40/config.json').unlink()
    a=NS(out=root,arm='K',seed=101,stage='finqa',standin=True)
    if final_missing:
        with pytest.raises(NotReady,match='required final export missing'):m3_train.export_job(a)
    else:m3_train.export_job(a)
    assert json.loads((run/'export-availability.local.json').read_text())['exports']=={
        '10':'MISSING-EXPORT','20':'MISSING-EXPORT','30':'PRESENT',
        '40':'MISSING-EXPORT' if final_missing else 'PRESENT'}
    if not final_missing:
        # Pruning is tolerated; changing a surviving receipt remains an identity error.
        atomic(run/'export-30.json',dict(model_file_hashes={'config.json':'tampered'}))
        with pytest.raises(Exception,match='hash differs'):m3_train.export_job(a)


def test_busy_allocation_is_pending_and_does_not_launch_any_child(staged,monkeypatch):
    from contextlib import contextmanager
    from kit import m3_manifest
    doc,_,_=staged;root=Path(doc['settings']['kcr_root']);control=Path(doc['settings']['control_root'])
    atomic(control/'state/M3-rescore-101-ready.json',dict(status='DONE'))
    atomic(root/'scoring-ready/M3-rescore-101.local.json',dict(checkpoints={}))
    monkeypatch.setattr(m3_manifest,'allocated_gpus',lambda n:['GPU-busy'])
    @contextmanager
    def busy(*a,**kw):
        raise m3_manifest.Unbound('allocated GPU reservation is busy')
        yield
    monkeypatch.setattr(m3_manifest,'allocation_reservation',busy)
    assert combined.execute(doc,'M3-rescore-101',runner=lambda *a,**kw:pytest.fail('busy lane launched'))==75
    assert json.loads((control/'state/M3-rescore-101.json').read_text())['status']=='PENDING'


@pytest.mark.parametrize('changes',[{},dict(hidden_size=2048),dict(model_type='other')])
@pytest.mark.parametrize('panel',['finqa','test','math_hard','general-v1','hellaswag','knowledge-ll'])
def test_rescore_concurrency_uses_measured_architecture_only(tmp_path,changes,panel):
    config=dry.pinned_base_config()
    config.update(changes);atomic(tmp_path/'model/config.json',config)
    atomic(tmp_path/'ready.json',dict(checkpoints={'one':dict(source=dict(model=str(tmp_path/'model')))}))
    command=['python','-I','m3_score.py','--panel-set',panel]
    expected=2 if panel in ('hellaswag','knowledge-ll') else 3
    assert v6a.rescore_concurrency(tmp_path/'ready.json',[command])==(1 if changes else expected)
    assert v6a.rescore_concurrency(tmp_path/'ready.json',[[*command,'--max-model-len','16384']])==1


@pytest.mark.parametrize('slurm,rows',[('5',{'0':'GPU-other','5':'GPU-measured'}),('5',{'0':'GPU-measured'}),('GPU-measured',{'5':'GPU-measured'})])
def test_cgroup_mapping_prefers_slurm_allocation(monkeypatch,slurm,rows):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setenv('SLURM_JOB_GPUS',slurm)
    monkeypatch.delenv('SLURM_STEP_GPUS',raising=False)
    assert m2_gpu.allocated_devices(rows)=={'0':'GPU-measured'}


def test_pool_gpu_uuid_prefers_scheduler_physical_id(monkeypatch):
    calls=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setenv('SLURM_JOB_GPUS','5')
    monkeypatch.delenv('SLURM_STEP_GPUS',raising=False)
    def run(command,**kw):calls.append(command);return NS(stdout='GPU-measured\n')
    monkeypatch.setattr(lane_pool.subprocess,'run',run)
    assert lane_pool.gpu_uuid()=='GPU-measured' and '--id=5' in calls[0]


def test_pool_falls_back_to_single_cgroup_uuid_when_physical_selector_unavailable(monkeypatch):
    calls=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setenv('SLURM_JOB_GPUS','5')
    monkeypatch.delenv('SLURM_STEP_GPUS',raising=False)
    def run(command,**kw):
        calls.append(command)
        if '--id=5' in command:raise subprocess.CalledProcessError(1,command)
        return NS(stdout='GPU-measured\n')
    monkeypatch.setattr(lane_pool.subprocess,'run',run)
    assert lane_pool.gpu_uuid()=='GPU-measured' and len(calls)==2


def test_knowledge_production_uses_effective_shared_engine_fraction(monkeypatch):
    """Exercise the real knowledge path and create_llm, substituting only SDKs."""
    from types import ModuleType
    from kit import m2_format,inference_lifecycle,m2_paths
    monkeypatch.setenv('KIT_GPU_MEMORY_FRACTION','.31')
    monkeypatch.setattr(m2_gpu,'assert_current',lambda:None)
    monkeypatch.setattr(m2_paths,'local',lambda p:p)
    monkeypatch.setattr(m2_format,'tokenizer_identity',lambda p:None)
    fake=ModuleType('vllm');fake.SamplingParams=lambda **kw:kw;calls=[]
    class Engine:
        def __init__(self,**kw):calls.append(kw)
        def generate(self,requests,*a,**kw):
            return [NS(prompt_logprobs=[{v:NS(logprob=-1.)} for v in r['prompt_token_ids']]) for r in requests]
        def shutdown(self):pass
    fake.LLM=Engine;monkeypatch.setitem(sys.modules,'vllm',fake)
    transformers=ModuleType('transformers')
    transformers.AutoTokenizer=NS(from_pretrained=lambda *a,**kw:NS(encode=lambda text,**kw:list(text.encode())))
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    result=m3_score.knowledge_production('model',[dict(id='one',context='Question?',endings=['a','b','c','d'])],'tokenizer')
    assert len(result)==1 and len(result[0]['scores'])==4
    assert calls[0]['gpu_memory_utilization']==.42 and calls[0]['max_model_len']==12288


def test_reference_default_requires_cpu_prepare_before_gpu_work(staged,monkeypatch):
    doc,_,_=staged;root=Path(doc['settings']['kcr_root']);path=root/'list.json'
    atomic(path,[dict(arm='base',seed=101,stage='finqa',step=0)])
    monkeypatch.setattr(ref_score_job,'source_for',lambda *a:pytest.fail('must not resolve models before CPU preparation'))
    with pytest.raises(Exception,match='prepare every reference checkpoint'):
        ref_score_job.run(NS(out=root,family='m3',seed=101,panel='knowledge-ll',checkpoints=path,gpu='0',standin=False))


def test_operator_authority_required_no_self_hash_fallback(monkeypatch):
    monkeypatch.delenv('KIT_V6_AUTHORITY_SHA',raising=False)
    monkeypatch.setenv('M2_INFRA_AUTHORITY_SHA256',v6_infra.authority_sha256())
    with pytest.raises(ValueError,match='KIT_V6_AUTHORITY_SHA'):v6_infra.activate_v6a()
    monkeypatch.setenv('KIT_V6A','1')
    with pytest.raises(ValueError,match='KIT_V6_AUTHORITY_SHA'):v6_infra.activate()
    monkeypatch.setenv('KIT_V6_AUTHORITY_SHA','0'*64)
    with pytest.raises(ValueError,match='authority changed'):v6_infra.activate_v6a()


def test_verified_base_can_be_rescored_under_each_arm_without_an_export_receipt(tmp_path,monkeypatch):
    """The real inherited scorer admission must recognise stage-1's frozen base."""
    from kit import p3b_score,next_hellaswag,score_forgetting,m2_assignment
    monkeypatch.setenv('M3_SCORE_READY','CPU-PREPARED-FIXTURE')
    source=dict(base=True,model=str(tmp_path/'base'),receipt=None)
    observed=dict(model_file_hashes={'config.json':'cpu-verified'})
    monkeypatch.setattr(p3b_score,'frozen_hashes',lambda:None)
    monkeypatch.setattr(score_forgetting,'load_panel',lambda:([dict(id='q',panel='ifeval')],'panel'))
    monkeypatch.setattr(m2_assignment,'active',lambda *a:dict(uuid='GPU-fixture'))
    saved=[]
    class BoundIdentity(Exception):pass
    def bind(path,value):saved.append(value);raise BoundIdentity
    monkeypatch.setattr(p3b_score,'bind',bind)
    args=NS(out=tmp_path,arm='K.finqa',seed=101,step=0,panel_set='general-v1',model=source['model'],
            gpu='0',standin=False,checkpoint_receipt=None,panel_root=None,package_a=None,base_checkpoint=True)
    with m3_score.prepared_identity(source,observed,False):
        assert next_hellaswag.checkpoint(source['model'],None,base=False)==observed
        with pytest.raises(BoundIdentity):p3b_score.job(args)
    assert saved[0]['model']==observed
    with m3_score.prepared_identity(dict(source,base=False),observed,False):
        with pytest.raises(Exception,match='checkpoint receipt required'):
            next_hellaswag.checkpoint(source['model'],None,base=False)


@pytest.mark.parametrize('panel',['finqa','general-v1'])
def test_m3_incoming_base_dispatch_carries_verified_base_identity(tmp_path,monkeypatch,panel):
    from kit import p3b_score
    source=dict(base=True,model=str(tmp_path/'base'),receipt=None)
    monkeypatch.setattr(m3_score,'checked',lambda *a:(source,dict(model_file_hashes={'config.json':'verified'})))
    monkeypatch.setattr(m3_score,'proof_for',lambda *a:dict(uuid='GPU-fixture',environment='CPU-STANDIN'))
    monkeypatch.setattr(m3_score,'resolved_gpu',lambda a:a.gpu)
    calls=[]
    def inherited(a):
        calls.append(a)
        assert a.base_checkpoint is True and a.checkpoint_receipt is None
        unit=Path(a.out)/'scores'/f'{a.arm}-s{a.seed}-k{a.step}-{a.panel_set}'
        atomic(unit/'identity.json',dict(panel_sha256='synthetic-fixture'))
        (unit/'scores.csv').write_text('correct\n1\n')
        atomic(unit/'manifest.json',files_manifest(unit,['identity.json','scores.csv']))
    monkeypatch.setattr(p3b_score,'job',inherited)
    m3_score.score_job(NS(out=tmp_path,arm='K',seed=101,stage='finqa',step=0,panel_set=panel,
                         gpu='0',standin=False,panel_root=None))
    assert len(calls)==1


def test_wrapper_reference_and_report_roots_do_not_read_other_group_settings(tmp_path):
    import shutil
    kit=tmp_path/'kit';kit.mkdir()
    shutil.copy(combined.KIT/'util_watch.py',kit/'util_watch.py')
    (kit/'combined_send.py').write_text('raise SystemExit(0)\n')
    manifest=tmp_path/'manifest.json'
    combined.write(manifest,dict(settings=dict(control_root=str(tmp_path/'control with spaces')),
        tasks=[dict(id='REF-m3-assign',group='reference',gpus=0),dict(id='REPORT-text',group='report',gpus=0)]))
    env=dict(os.environ,KIT=str(kit),COMBINED_MANIFEST=str(manifest),JOB_PY=sys.executable)
    env.pop('UTIL_WATCH_ACTIVE',None)
    for ident in ('REF-m3-assign','REPORT-text'):
        done=subprocess.run(['bash',str(combined.KIT/'jobs/combined/run-task.sh'),ident],env=env,capture_output=True,text=True)
        assert done.returncode==0,done.stderr
    summaries=list((tmp_path/'control with spaces').rglob('*.summary.json'))
    assert len(summaries)==2 and all(json.loads(p.read_text())['gpus']==0 for p in summaries)


def test_all_v6a_rows_real_wrappers_pending_failure_retry_and_reference_only_guards(tmp_path):
    result=dry.run(tmp_path/'rehearsal')
    assert result['rows']==14 and result['wrapper_calls']==18
    assert result['pending']==2 and result['failed']==1 and result['parts']==3
    assert result['native_commands']==594 and result['reference_commands']==39
    assert all(d['verdict']=='INCONCLUSIVE' for d in result['KCR'])
    text=Path(result['text_out']);full=json.loads((text/'full-results.local.json').read_text())
    kcr=full['readers']['KCR'][0]
    assert not kcr['identity_failures'] and all(c.get('method')==ref_guard.METHOD for c in kcr['contrasts'].values())
    assert all(r[0]['_reader_status']=='DEFERRED' for f,r in full['readers'].items() if f!='KCR')
    from kit.text_return import parse_parts
    parsed=parse_parts([p.read_text() for p in sorted(text.glob('RETURN-TEXT-*.md'))])
    assert parsed['infrastructure_authority']==v6_infra.authority_sha256()


@pytest.mark.parametrize('refusal',['host','manifest','authority'])
def test_refusals_record_each_attempt(staged,monkeypatch,refusal):
    doc,_,_=staged
    if refusal=='host':doc['nodes']['kcr']=['foreign-node']
    elif refusal=='manifest':doc['sources']['kcr_manifest']='0'*64
    else:monkeypatch.delenv('KIT_V6_AUTHORITY_SHA')
    for _ in range(2):
        assert combined.execute(doc,'REF-m3-assign',runner=lambda *a,**kw:pytest.fail('refused job launched'))==1
    state=Path(doc['settings']['control_root'])/'state'
    records=list((state/'attempts/REF-m3-assign').glob('*.json'))
    assert len(records)==2 and len({combined.load(p)['nonce'] for p in records})==2
    assert all(combined.load(p)['status']=='FAILED' and combined.load(p)['error'] for p in records)


def test_v6a_inventory_and_dependent_submission_scope(staged):
    inventory=combined.inventory(staged[0])
    assert 'Approved tier allowance' not in inventory and 'PD <=' not in inventory
    runbook=(combined.KIT/'README-v6a.md').read_text()
    assert runbook.count('--kill-on-invalid-dep=yes --dependency=')==4
    assert '--dependency=afterany:' in runbook and 'REF utilisation is unmeasured' in runbook
    assert 'sbatch --parsable --nodes=1 --gpus=0' not in runbook


@pytest.mark.parametrize('family',['native','reference'])
def test_corrupt_attempt_is_per_seed_unverified_and_no_seal_is_pending(staged,monkeypatch,family):
    root=Path(staged[0]['settings']['kcr_root']);monkeypatch.setenv('KIT_V6A','1')
    # A stale unsealed native unit must not be resolved, even with no nonce dirs.
    monkeypatch.setattr(m3_score,'resolve',lambda *a:pytest.fail('unsealed legacy cells read'))
    cells,assignment=m3_read.load_cells(root,True)
    assert not cells
    assert {v['label'] for v in m3_read.analyse(cells,assignment)['contrasts'].values()}=={'PENDING'}
    parent=root/('scoring-jobs/M3-rescore-101' if family=='native' else 'reference-attempts/m3/101')
    atomic(parent/('a'*32)/'attempt.json',{'schema':'corrupt'})
    cells,assignment=m3_read.load_cells(root,True)
    result=m3_read.analyse(cells,assignment)
    assert {v['label'] for v in result['contrasts'].values()}=={'UNVERIFIED'}
    assert {f['cell'][1] for f in result['identity_failures']}=={101}


def test_wrapper_path_setup_failure_runs_child_once_unwatched(tmp_path):
    kit=tmp_path/'kit';kit.mkdir();marker=tmp_path/'calls'
    (kit/'combined_send.py').write_text('import os;assert os.environ["UTIL_WATCH_ACTIVE"]=="1";open('+repr(str(marker))+',"a").write("once\\n");raise SystemExit(37)\n')
    shim=tmp_path/'python';shim.write_text('#!'+sys.executable+'\nimport os,sys\nif "import json,sys,uuid" in " ".join(sys.argv):raise SystemExit(9)\nos.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n');shim.chmod(0o755)
    manifest=tmp_path/'manifest.json';atomic(manifest,dict(mode='v6a',settings={},tasks=[dict(id='REPORT-text',group='report',gpus=0)]))
    env=dict(os.environ,KIT=str(kit),JOB_PY=str(shim),COMBINED_MANIFEST=str(manifest));env.pop('UTIL_WATCH_ACTIVE',None)
    done=subprocess.run(['bash',str(combined.KIT/'jobs/v6a/run.sh'),'REPORT-text'],env=env,capture_output=True,text=True)
    assert done.returncode==37 and marker.read_text()=='once\n' and 'path unavailable' in done.stderr


def test_simultaneous_row_attempts_keep_distinct_state_records(staged):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    doc,_,_=staged;barrier=threading.Barrier(2)
    def runner(*args,**kwargs):barrier.wait(timeout=5);return NS(returncode=0)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures=[workers.submit(combined.execute,doc,'REF-m3-assign',runner=runner) for _ in range(2)]
        assert [f.result() for f in futures]==[0,0]
    state=Path(doc['settings']['control_root'])/'state'
    records=[combined.load(p) for p in (state/'attempts/REF-m3-assign').glob('*.json')]
    assert len(records)==2 and len({r['nonce'] for r in records})==2
    assert combined.load(state/'REF-m3-assign.json')['nonce'] in {r['nonce'] for r in records}


@pytest.mark.parametrize('corrupt',[[],{'schema':'lane-attempt.v1','completed_ns':'bad'},
    {'schema':'lane-attempt.v1','completed_ns':1,'files':{},'identities':[],'nonce':'a'*32}])
def test_malformed_attempt_receipts_are_unverified_per_seed(staged,monkeypatch,corrupt):
    root=Path(staged[0]['settings']['kcr_root']);monkeypatch.setenv('KIT_V6A','1')
    atomic(root/'scoring-jobs/M3-rescore-102'/('a'*32)/'attempt.json',corrupt)
    cells,assignment=m3_read.load_cells(root,True)
    failures=m3_read.analyse(cells,assignment)['identity_failures']
    assert len(failures)==1 and failures[0]['cell'][1]==102
    assert 'corrupt allocation attempt' in failures[0]['reason']
