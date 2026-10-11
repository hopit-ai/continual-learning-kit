"""Plan 15.21/15.22 through real runner, planner, retry and warning paths; CPU only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from kit import runner
from kit import v4_finqa_policy as policy
from kit.v4_campaign import build


def graph(tmp_path,rows,phase='main'):
    path=tmp_path/f'{phase}.json'
    path.write_text(json.dumps(dict(schema=runner.SCHEMA,name='v4-'+phase,workdir_env='WORK',
        rows=rows,v4=dict(phase=phase,scope='finqa-alone',self_monitored=True,ceiling=1000))))
    return path


def ledger(work,seconds,width=8):
    from kit import p4_watchdog as wd
    start=1700000000;end=start+seconds
    entry=dict(start=wd.precise_text(wd.from_epoch(start)),end=wd.precise_text(wd.from_epoch(end)),
        width=width,segments=[dict(block='qualification',start=wd.precise_text(wd.from_epoch(start)),end=wd.precise_text(wd.from_epoch(end)))])
    doc=dict(allocations={'1':entry},block_limits={'qualification':100},ceiling=955)
    file=work/'k8b4/containment/allocation-ledger.json';file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(doc))
    return doc


def test_graph_has_no_pause_owner_or_performance_pilot():
    for phase in ('qualification','main'):
        doc=build(phase);assert doc['v4']['ceiling']==1000
        assert doc['v4']['qualification_limit']==1000
        assert doc['v4']['planned_qualification_gpu_hours']==100
        assert not any(r.get('pilot') or r['id']=='PAUSE' for r in doc['rows'])
        assert 'V4_OWNER_PROCEED' not in json.dumps(doc)
        assert all(s['block_limit']==1000 for s in doc['v4']['allocation_stages'].values())
        assert all(r['command'][r['command'].index('--ceiling')+1]=='1000' for r in doc['rows'] if '--ceiling' in r['command'])
    q=next(r for r in build('qualification')['rows'] if r['id']=='teacher-finqa')
    assert int(q['command'][q['command'].index('--time-cap')+1])==144000


@pytest.mark.parametrize('message',['coverage gate failed','scoring GPU or environment differs from qualification',
 'batch-invariant determinism failed','missing optimizer metrics','restoration phase0 measurement registration differs',
 'allocation projection exceeds planned qualification block','failed training needs reviewed infrastructure retry',
 'unsafe scheduler socket path','capture WORK must be owned, not a symlink','linked retirement evidence refused'])
def test_operational_require_records_warning_not_refusal(tmp_path,monkeypatch,message):
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('WORK',str(tmp_path))
    assert policy.require(False,message,'regression') is False
    assert policy.warnings(tmp_path)[0]['message']==message


@pytest.mark.parametrize('message',['registered model identity differs: teacher','prepared input changed: data.parquet',
 'initial 8B revision differs','export identity changed','scored checkpoint shards differ','writes outside WORK','disk exhaustion before export','actual 1000 GPU-hour ceiling reached'])
def test_four_correctness_classes_still_stop(tmp_path,monkeypatch,message):
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('WORK',str(tmp_path))
    with pytest.raises(policy.CorrectnessStop):policy.require(False,message)


def test_phase0_keeps_its_v7_policy(tmp_path,monkeypatch):
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('V4_PHASE0_MODE','1')
    with pytest.raises(ValueError):policy.require(False,'coverage gate failed')


def test_ceiling_uses_actual_spend_not_100_block_or_reservation(tmp_path,monkeypatch):
    monkeypatch.setenv('WORK',str(tmp_path));monkeypatch.setenv('V4_FINQA_CAMPAIGN','1')
    ledger(tmp_path,999*3600/8)
    assert policy.ceiling(tmp_path)==999
    ledger(tmp_path,1000*3600/8)
    with pytest.raises(policy.CorrectnessStop,match='actual 1000'):policy.ceiling(tmp_path)


def test_failed_row_fixed_in_same_work_gets_new_attempt_and_pass_is_skipped(tmp_path,monkeypatch):
    monkeypatch.setenv('WORK',str(tmp_path));monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    script=tmp_path/'p4_contain.py'
    script.write_text('import pathlib,sys\nflag=pathlib.Path(sys.argv[-1]);sys.exit(0 if flag.exists() else 1)\n')
    flag=tmp_path/'fixed'
    row=dict(id='score-local',command=[sys.executable,str(script),'row','--out',str(tmp_path/'out/attempt-{attempt}'),str(flag)])
    following=dict(id='after',needs=['score-local'],command=[sys.executable,'-c','print("later row ran")'])
    path=graph(tmp_path,[row,following])
    assert runner.main(['run',str(path),'--all'])==1
    campaign=runner.load_campaign(path)
    assert runner.latest_verdict(campaign,'score-local')['verdict']=='FAIL'
    assert runner.latest_verdict(campaign,'after')['verdict']=='PASS'
    flag.touch()
    assert runner.main(['run',str(path),'--row','score-local'])==0
    assert runner.attempts(campaign,'score-local')==[1,2]
    assert runner.latest_verdict(campaign,'score-local')['verdict']=='PASS'
    receipt=json.loads((tmp_path/'v4/retries/v4-main/score-local/attempt-2.json').read_text())
    assert receipt['review_policy']=='owner self-monitoring; no manual review'
    assert runner.main(['run',str(path),'--all'])==0
    assert runner.main(['run',str(path),'--row','score-local'])==0
    assert runner.attempts(campaign,'after')==[1]
    assert runner.attempts(campaign,'score-local')==[1,2]
    assert (tmp_path/'campaign/v4-main/score-local/attempt-1/output.log').exists()


def test_flow_runs_main_after_qualification_failure_and_no_pause(tmp_path,monkeypatch):
    from kit import v4_finqa_flow as flow
    monkeypatch.setenv('WORK',str(tmp_path));monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    campaigns=tmp_path/'campaigns';campaigns.mkdir()
    for phase,command in [('qualification',[sys.executable,'-c','raise RuntimeError("diagnostic failed")']),
                          ('main',[sys.executable,'-c','print("main followed automatically")'])]:
        p=graph(campaigns,[dict(id='row-'+phase,command=command)],phase)
        p.rename(campaigns/f'v4-{phase}.yaml')
    assert flow.run(tmp_path,campaigns=campaigns)==1
    main=runner.load_campaign(campaigns/'v4-main.yaml')
    assert runner.latest_verdict(main,'row-main')['verdict']=='PASS'
    assert policy.warnings(tmp_path)


def test_low_coverage_common_data_admission_warns_but_hash_mismatch_stops(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kit import v4_run as run, v4_teacher as t
    monkeypatch.setenv('WORK',str(tmp_path));monkeypatch.setenv('V4_FINQA_CAMPAIGN','1')
    rows=[dict(extra_info=dict(index=str(i),split='train',demonstration='verified demo',rewrite='verified rewrite'),
               prompt=[dict(role='user',content='question')]) for i in range(1280)]
    train=tmp_path/'train.parquet';pq.write_table(pa.Table.from_pylist(rows),train)
    manifest=tmp_path/'manifest.json'
    doc=dict(schema='kit-v4-sft-data.v1',arm='COMMON',seed=101,train_sha256=t.sha(train.read_bytes()),
             technical_synthetic=False,scientific_phase_allowed=False,stand_in_teacher=False,rows=1280,
             ids=[r['extra_info']['index'] for r in rows])
    manifest.write_text(json.dumps(doc))
    monkeypatch.setattr(run,'check_schedule',lambda *a:None)
    env=dict(DATA_MANIFEST=str(manifest),TRAIN_FILE=str(train),SEED='101',ARM='F',STEPS='40')
    assert run.check_data(env)==doc
    assert any('unqualified' in r['message'] for r in policy.warnings(tmp_path))
    doc['train_sha256']='0'*64;manifest.write_text(json.dumps(doc))
    with pytest.raises(policy.CorrectnessStop,match='hash mismatch'):run.check_data(env)


@pytest.mark.parametrize('socket_warning',[False,True])
def test_disk_advice_is_not_admission_and_next_write_is_checked(tmp_path,monkeypatch,socket_warning):
    from kit import v4_finqa_site as site
    monkeypatch.setattr(site,'disk_requirement',lambda *a:dict(runs=[],required_bytes=1879*site.GIB))
    def socket_check(*args):
        if socket_warning:raise ValueError('unsafe scheduler socket path')
        return {}
    monkeypatch.setattr(site.site,'socket_path_check',socket_check)
    monkeypatch.setattr(site.site,'free_space',lambda work,minimum: (100*site.GIB if minimum==0 else (_ for _ in ()).throw(ValueError('too large'))))
    assert site.planning_storage(tmp_path,{'WORK':str(tmp_path),'TMPDIR':str(tmp_path)})['advisory_only']
    if socket_warning:assert any(r['site']=='socket paths' for r in policy.warnings(tmp_path))
    from types import SimpleNamespace
    monkeypatch.setattr(policy.shutil,'disk_usage',lambda *a:SimpleNamespace(free=100))
    assert policy.disk(tmp_path,99,'write')==100
    with pytest.raises(policy.CorrectnessStop):policy.disk(tmp_path,101,'write')


def test_local_retry_selects_complete_latest_state_without_review(tmp_path,monkeypatch):
    from kit import v4_retry as retry
    from kit.v4_retention import state_names
    ident='F-finqa-alone-s101';root=tmp_path/'runs'/ident/'checkpoints/global_step_20';root.mkdir(parents=True)
    for name in state_names('F'):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'complete state fixture')
    (root/'data.pt').write_bytes(b'data loader')
    value=retry.local_retry(tmp_path,'v4-main','train-'+ident,7,{'attempt':6})
    assert value['state_policy']=='resume_valid_state'
    assert value['completed_updates']==20
    assert value['saved_state_sha256']==retry.saved_state(root,'F')
    assert 'termination_verified' not in value


def test_continuation_uses_native_wrapper_and_afterany(tmp_path,monkeypatch):
    from kit import v4_finqa_flow as flow
    script=tmp_path/'flow.sbatch';script.write_text('#!/bin/bash\n')
    monkeypatch.setenv('V4_FLOW_WRAPPER',str(script));monkeypatch.setenv('SLURM_JOB_ID','123')
    observed=[]
    def submit(command,**kwargs):
        observed.append(command);return subprocess.CompletedProcess(command,0,'456;cluster\n','')
    monkeypatch.setattr(flow.subprocess,'run',submit)
    assert flow.continuation(tmp_path)['job_id']=='456'
    assert '--dependency=afterany:123' in observed[0]
    assert observed[0][-1]==str(script)
    assert list((tmp_path/'v4/report-flow/submissions').glob('*.json'))


def test_missing_continuation_wrapper_is_retryable_infrastructure(tmp_path,monkeypatch):
    from kit import v4_finqa_flow as flow
    monkeypatch.setenv('V4_FLOW_WRAPPER',str(tmp_path/'missing.sbatch'))
    with pytest.raises(FileNotFoundError,match='native continuation wrapper missing'):
        flow.continuation(tmp_path)
    monkeypatch.setenv('V4_FLOW_WRAPPER',str(tmp_path.parent/'outside.sbatch'))
    with pytest.raises(policy.CorrectnessStop,match='writes outside WORK'):
        flow.continuation(tmp_path)


def test_staged_payload_runs_flow_without_owner_receipt(tmp_path):
    from kit.v4_allocation import script
    doc=dict(phase='qualification',stage='qualification',block='scientific',block_limit=1000,
             ceiling=1000,plan_path=str(tmp_path/'flow.json'),launch_environment={})
    payload=script(doc,tmp_path,Path(__file__).resolve().parents[1]/'kit')
    assert 'v4_finqa_flow.py' in payload
    assert 'V4_FLOW_WRAPPER=' in payload
    assert 'V4_OWNER_PROCEED' not in payload


def test_warnings_cannot_write_through_symlink_outside_work(tmp_path,monkeypatch):
    work=tmp_path/'work';work.mkdir();outside=tmp_path/'outside';outside.mkdir()
    (work/'v4').symlink_to(outside,target_is_directory=True)
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('WORK',str(work))
    with pytest.raises(policy.CorrectnessStop,match='outside WORK'):policy.warn('regression','observation')
    assert not list(outside.iterdir())


def test_local_software_fix_rechecks_data_without_review_receipt(tmp_path,monkeypatch):
    from kit import v4_prepare as prep
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('WORK',str(tmp_path))
    code=tmp_path/'helper.py';code.write_text('print(1)')
    data=tmp_path/'train.parquet';data.write_bytes(b'pinned input')
    receipt=prep.file_receipt([code,data]);code.write_text('print(2)')
    assert prep.verify_files(receipt)
    assert policy.warnings(tmp_path)
    data.write_bytes(b'changed data')
    with pytest.raises(ValueError,match='prepared input changed'):prep.verify_files(receipt)


def test_missing_telemetry_cannot_hide_known_ceiling(tmp_path,monkeypatch):
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('WORK',str(tmp_path))
    doc=ledger(tmp_path,1000*3600/8);doc['allocations']['unknown']={'start':None,'width':None}
    (tmp_path/'k8b4/containment/allocation-ledger.json').write_text(json.dumps(doc))
    with pytest.raises(policy.CorrectnessStop,match='actual 1000'):policy.ceiling(tmp_path)
    assert policy.warnings(tmp_path)


def test_prior_unknown_job_is_not_charged_as_forever_running(tmp_path,monkeypatch):
    from kit import p4_contain as pc
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('SLURM_JOB_ID','current')
    doc=ledger(tmp_path,999*3600/8)
    doc['allocations']['old']=dict(start=1,end=None,width=8,observed_until=2,segments=[dict(block='scientific',start=1)])
    path=tmp_path/'k8b4/containment/allocation-ledger.json';path.write_text(json.dumps(doc))
    refreshed=pc.refresh_allocation_ledger(tmp_path)
    assert refreshed['allocations']['old']['observed_until']==2
    assert float(policy.known_spend(refreshed,tmp_path)[0])<1000


def test_launcher_configuration_findings_warn_and_preserve_local_value(tmp_path):
    kit=Path(__file__).resolve().parents[1]/'kit'
    env={**os.environ,'KIT':str(kit),'WORK':str(tmp_path),'V4_FINQA_CAMPAIGN':'1','V4_PHASE0_MODE':'0','LR':'2e-5'}
    command='source "$KIT/v4_profile.sh"; printf "LOCAL_LR=%s\n" "$LR"'
    done=subprocess.run(['bash','-c',command],env=env,capture_output=True,text=True)
    assert done.returncode==0 and 'LOCAL_LR=2e-5' in done.stdout and 'WARNING:' in done.stdout
    assert policy.warnings(tmp_path)
    env['V4_PHASE0_MODE']='1'
    strict=subprocess.run(['bash','-c',command],env=env,capture_output=True,text=True)
    assert strict.returncode==2


def test_changed_selected_model_path_does_not_escape_identity_check(tmp_path,monkeypatch):
    from kit import v4_prepare as prep,v4_finqa_site as site,v4_teacher as teacher
    folder=tmp_path/'v4/report-inputs';folder.mkdir(parents=True)
    original=tmp_path/'initial';original.mkdir();replacement=tmp_path/'replacement';replacement.mkdir()
    doc={'environment':dict(MODEL_DIR=str(original),TEACHER_MODEL_DIR=str(original),QWEN3_8B_TOKENIZER=str(original))}
    (tmp_path/site.LAUNCH).write_text(json.dumps(doc))
    (folder/'model-identities.json').write_text(json.dumps({'initial':dict(model_revision='pinned',model_file_hashes={'w':'old'})}))
    monkeypatch.setenv('MODEL_DIR',str(replacement));monkeypatch.setenv('TEACHER_MODEL_DIR',str(original));monkeypatch.setenv('QWEN3_8B_TOKENIZER',str(original))
    monkeypatch.setattr(teacher,'local_identity',lambda *a:dict(model_revision='pinned',model_file_hashes={'w':'different'}))
    with pytest.raises(ValueError,match='selected model identity differs'):prep.verify_selected_models(tmp_path)


def test_child_correctness_stops_but_actual_engine_failure_is_retryable(tmp_path,monkeypatch):
    monkeypatch.setenv('WORK',str(tmp_path))
    for message,expected in [('OSError: [Errno 28] No space left on device',2),('ValueError: selected model identity differs: teacher',2),('RuntimeError: CUDA out of memory',1)]:
        path=graph(tmp_path,[dict(id='operation',needs=[],command=[sys.executable,'-c','print('+repr(message)+');raise SystemExit(1)'])])
        doc=runner.load_campaign(path)
        assert runner.run_row(doc,doc['rows'][0],None)==expected
        assert runner.latest_verdict(doc,'operation')['verdict']=='FAIL'


def test_archived_software_fix_warns_but_data_hash_still_stops(tmp_path,monkeypatch):
    from kit import v4_prepare as prep
    import hashlib
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('WORK',str(tmp_path))
    monkeypatch.setattr(prep,'smoke_inventory',lambda:dict(modules=[],dependencies=[]))
    names=['first-sft-cpu-batch','entry-help-sft_entry.py','entry-help-sdft_entry.py','entry-help-S',*('launcher-'+a for a in 'SFRD')]
    smoke=dict(ok=True,inventory=dict(modules=[],dependencies=[]),environment=dict(trainer_commit='same',runtime_versions={}),observations=[dict(name=n,returncode=0,log_sha256=hashlib.sha256(b'log').hexdigest()) for n in names])
    raw=json.dumps(smoke).encode();data=b'pinned data'
    files={'v4/report-inputs/import-smoke/receipt.json':raw,'kit-snapshot/helper.py':b'local infrastructure fix','v4/datasets/train.parquet':data}
    files.update({f'v4/report-inputs/import-smoke/{i}.log':b'log' for i in range(len(names))})
    receipt=dict(before_allocation=True,import_smoke_sha256=hashlib.sha256(raw).hexdigest(),archived_files={'kit-snapshot/helper.py':hashlib.sha256(b'original').hexdigest(),'v4/datasets/train.parquet':hashlib.sha256(data).hexdigest()})
    class Evidence:
        def __init__(self):self.files=files
        def json(self,name):
            if name.endswith('prepare-receipt.json'):return receipt
            if name.endswith('environment.json'):return dict(trainer_commit='same',runtime_versions={})
            return json.loads(self.files[name])
    archive=Evidence();assert prep.verify_archived(archive)==receipt
    assert any('software/observation changed' in r['message'] for r in policy.warnings(tmp_path))
    files['v4/datasets/train.parquet']=b'wrong data'
    with pytest.raises(ValueError,match='prepared archive input changed'):prep.verify_archived(archive)


def test_phase0_charge_conflict_warns_and_keeps_verified_import(tmp_path,monkeypatch):
    import copy
    from kit import v4_phase0_carry as carry
    from test_kit_finqa_phase0_carry import ledger as imported_ledger
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('WORK',str(tmp_path))
    doc=imported_ledger();doc['allocations']=copy.deepcopy(doc['phase0_import']['allocations'])
    doc['allocations']['123']['width']=7
    merged=carry.accounting_allocations(doc)
    assert merged['123']['width']==8 and policy.warnings(tmp_path)
    assert float(policy.known_spend(doc,tmp_path)[0])==pytest.approx(44/3)


def test_missing_verification_timing_is_warning_not_main_gate(tmp_path,monkeypatch):
    from kit import v4_prepare as prep
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('WORK',str(tmp_path))
    assert prep.main_verification_allowance(None)==600
    assert policy.warnings(tmp_path)


def test_retirement_deadline_warning_cannot_block_real_train_consumer(tmp_path,monkeypatch):
    from kit import v4_campaign_ops as ops,v4_retention as retention,v4_prepare as prep
    monkeypatch.setenv('MODEL_DIR',str(tmp_path/'model'));monkeypatch.setenv('WORK',str(tmp_path));monkeypatch.setenv('V4_PHASE0_MODE','0')
    monkeypatch.setattr(prep,'verify',lambda *a:True)
    monkeypatch.setattr(retention,'sweep',lambda *a,**kw:(_ for _ in ()).throw(ValueError('retirement IO deadline exceeded')))
    monkeypatch.setattr(ops.t,'local_identity',lambda *a:dict(model_file_hashes={'CPU adapter':'pinned'}))
    calls=[]
    def external_training(command,env):
        calls.append(command);(tmp_path/'runs/q-F-finqa/env').mkdir(parents=True)
    monkeypatch.setattr(ops,'run',external_training)
    assert ops.main(['train','--work',str(tmp_path),'--phase','qualification','--row','train-q-F-finqa','--slot','q-F-finqa'])==0
    assert calls and calls[0][1].endswith('run_v4.sh')
    assert any('retention before training'==r['site'] for r in policy.warnings(tmp_path))


@pytest.mark.parametrize('filename',['flow.sh','flow.main.sh'])
def test_main_replanning_retains_original_and_uses_same_work(tmp_path,monkeypatch,filename):
    from kit import v4_allocation as allocation,v4_finqa_site as site
    from contextlib import nullcontext
    inputs=tmp_path/'v4/report-inputs';inputs.mkdir(parents=True)
    activation=tmp_path/'activate';activation.write_text('# external CPU activation adapter')
    (inputs/'launch-inputs-finqa.json').write_text(json.dumps(dict(trainer_activate=str(activation))))
    target=tmp_path/filename;target.write_text('original evidence')
    monkeypatch.setattr(site,'storage',lambda work:nullcontext())
    monkeypatch.setattr(allocation,'plan',lambda *a:dict(phase='qualification',stage='qualification',block='scientific',block_limit=1000,ceiling=1000,gpus=8,sbatch_time='24:00:00',launch_environment={}))
    argv=['--work',str(tmp_path),'--phase','qualification','--stage','qualification','--minutes','1440','--site-minutes','1440','--out',str(target),'--partition','p','--qos','q','--account','a','--activate',str(activation)]
    assert allocation.main(argv)==0
    assert target.read_text()=='original evidence'
    assert target.with_name(target.stem+'-attempt-2'+target.suffix).exists()
    assert allocation.main(argv)==0 and target.with_name(target.stem+'-attempt-3'+target.suffix).exists()
    assert list((tmp_path/'v4/warnings').glob('*.json'))


def test_main_payload_uses_current_kit_after_infrastructure_move(tmp_path):
    from kit.v4_allocation import script
    current=tmp_path/'current-kit';frozen=tmp_path/'old-kit'
    doc=dict(phase='qualification',stage='qualification',block='scientific',block_limit=1000,ceiling=1000,launch_environment=dict(KIT=str(frozen),PYTHONPATH=str(frozen.parent),V4_PHASE0_MODE='1',SDPO_DIR='/trainer'))
    payload=script(doc,tmp_path,current)
    assert 'export KIT='+str(frozen) not in payload
    assert 'export KIT='+str(current) in payload
    assert 'export V4_PHASE0_MODE=1' not in payload


def test_flow_records_actual_boundaries_and_runs_main_without_pause(tmp_path,monkeypatch):
    from kit import v4_finqa_flow as flow
    campaigns=tmp_path/'campaigns';campaigns.mkdir()
    for phase in ('qualification','main'):
        path=graph(tmp_path,[dict(id='finished-'+phase,needs=[],command=[sys.executable,'-c','print("CPU successful operation")'])],phase=phase)
        (campaigns/f'v4-{phase}.yaml').write_bytes(path.read_bytes())
    monkeypatch.setenv('WORK',str(tmp_path));monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    observations=iter((12.,13.,13.,15.));monkeypatch.setattr(flow,'ceiling',lambda work:next(observations))
    assert flow.run(tmp_path,campaigns=campaigns)==0
    progress=[json.loads(p.read_text()) for p in (tmp_path/'v4/report-flow/progress').glob('*.json')]
    actual={p['phase']:p for p in progress}
    assert actual['qualification']['actual_spend_at_end']==13
    assert actual['main']['actual_spend_at_end']==15 and actual['main']['observed_allocation_gpu_hours_delta']==2
    assert all(not p['owner_proceed_required'] for p in progress)
    assert runner.latest_verdict(runner.load_campaign(campaigns/'v4-main.yaml'),'finished-main')['verdict']=='PASS'


@pytest.mark.parametrize('message',['27B registered revision/shards differ','incoming checkpoint shard lineage differs','general checkpoint differs','not the registered 27B architecture','common rows/targets differ from reverified intersection'])
def test_reader_identity_checks_are_explicit_not_message_guesses(tmp_path,monkeypatch,message):
    from kit import v4_evidence as evidence
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.setenv('WORK',str(tmp_path))
    with pytest.raises(policy.CorrectnessStop,match='model/data identity'):
        evidence.require(False,message,identity=True)
    assert policy.correctness('model/data identity: '+message)


def test_main_reader_propagates_wrong_checkpoint_identity_from_slot(tmp_path,monkeypatch):
    from kit import v4_finqa_read as reading
    class Evidence:
        files={'kit-snapshot/campaigns/v4-qualification.yaml':b'present'}
        def json(self,name):return {'v4':{'self_monitored':True}}
    monkeypatch.setattr(reading,'_campaign_evidence',lambda archive:dict(status='incomplete',reasons=[],disqualified={'S-finqa-alone-s101':'model/data identity: general checkpoint differs'},tables={}))
    result=reading.campaign_evidence(Evidence())
    assert result['status']=='stop: correctness'
    assert any('model/data identity' in r for r in result['reasons'])
