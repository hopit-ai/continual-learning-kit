"""Prospective three-seed 4A jobs: real wrappers/processes with labelled CPU stand-ins."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
from kit import a4_common as c,a4_jobs as jobs,a4_read,a4_bind,a4_sim


def test_amended_intervals_bars_and_old_df_rejection():
    assert c.SEEDS==(101,102,103) and c.AMENDMENT['date']=='2026-11-02'
    two=a4_read.interval([4,6,8]);one=a4_read.interval([4,6,8],True)
    assert two['df']==one['df']==2 and two['n']==3
    assert two['tcrit']==4.302653 and one['tcrit']==2.919986
    assert two['lower']==pytest.approx(6-4.302653*2/math.sqrt(3))
    assert one['upper']==pytest.approx(6+2.919986*2/math.sqrt(3))
    assert a4_read.primary({s:[6]*5 for s in c.SEEDS})['established']
    assert not a4_read.primary({s:[4.99]*5 for s in c.SEEDS})['established']
    assert a4_read.primary({s:[30,30,30,-1,-1] for s in c.SEEDS})['established']
    assert not a4_read.primary({s:[30,30,-1,-1,-1] for s in c.SEEDS})['established']
    result=a4_read.primary({s:[30,30,30,-1,-1] for s in c.SEEDS})
    assert result['required_positive_months']==3
    assert result['required_positive_month_fraction']==c.AMENDMENT['positive_month_fraction']==[3,5]
    assert a4_read.primary({s:[30,30,30,30,-1] for s in c.SEEDS})['established']
    result=a4_read.primary({101:[20]*5,102:[20]*5,103:[-1]*5})
    assert result['positive_seeds']==2 and not result['established']
    from kit.text_return import essential_endpoints
    bars={r['name']:r['bar'] for r in essential_endpoints('4A',dict(amendment=c.AMENDMENT,primary=result))}
    assert bars['consolidation-memory']=='mean>=5;lo>0;pos=3;months>=3'
    assert bars['positive-months']=='positive-months>=3'
    assert a4_read.interval([3]*3,True)['upper']==3
    with pytest.raises(c.CorrectnessStop):a4_read.interval([3]*5)


def test_rows_are_explicit_frozen_acyclic_and_right_sized():
    doc=jobs.inventory()
    assert json.loads((c.RES/'v6b-rows.json').read_text())==doc
    rows=doc['rows'];seen=set()
    assert len(rows)==197 and doc['scientific_score_cells']==418 and doc['training_stages']==27
    assert sum(r['gpus']==0 for r in rows)==136
    assert sum(r['gpus']==1 for r in rows)==34
    assert sum(r['gpus']==8 for r in rows)==27
    for row in rows:
        assert row['group']=='4a' and row['nodes']==1 and row['cpus']>0
        assert row['dependency_options']==['--kill-on-invalid-dep=yes']
        assert all(d['id'] in seen and d['type'] in ('afterok','afterany') for d in row['depends_on'])
        assert row['gpus']==(8 if row['phase']=='train' else 1 if row['phase']=='score' else 0)
        assert row['may_share_node']==(row['gpus']!=8)
        assert row['command'][-1]==row['id'] and row['gpu_hour_basis']
        assert row['time_limit_seconds']>0 and row['time'].count(':')==2
        if row['gpus']:
            hard=jobs.duration_limit(dict(job_limits=doc['scheduling_job_limits']),row)
            assert row['job_deadline_seconds']==hard
            assert row['time_limit_seconds']>=hard+max(900,.1*hard)
        if row['phase']=='score':assert row['lane_concurrency']==3
        seen.add(row['id'])
    assert jobs.row_by_id('A4-read')['depends_on']==[dict(id='A4-comparison',type='afterany')]
    assert sum(r['expected_command_counts']['score'] for r in rows if r['gpus'])==673
    assert sum(r['expected_command_counts']['nll'] for r in rows if r['gpus'])==54
    # The original B selector still binds all ten exports, independent of 4A's seed amendment.
    source={}
    for route in ('standalone','sequence'):
        for seed in range(101,106):
            source[f'scores/{route}-D-s{seed}-k40-math_screen/identity.json']=json.dumps(dict(model=dict(
                model_dir=f'/B/runs/{route}-D-s{seed}/hf-step40',model_file_hashes={'x':'a'}))).encode()
    assert len(a4_bind.selected_exports(source,'D',False))==10


@pytest.fixture(scope='module')
def rehearsal(tmp_path_factory):
    root=tmp_path_factory.mktemp('a4-v6b-rehearsal');fixture=root/'fixtures'
    sl,b,replay=a4_sim.fixture(fixture,'D')
    out=root/'A4';cfg=dict(out=str(out),slice=str(sl),b_source=str(b),replay=str(replay),base='synthetic-base',
        stage_gpu_hours=6.666666666666667,technical_synthetic=True,archive=str(root/'return.tar.gz'),
        trainer_python=sys.executable,inference_python=sys.executable,job_python=sys.executable,
        fail_task='A4-direct-s101-m1-train')
    config=root/'config.json';c.atomic(config,cfg)
    binary=root/'bin';binary.mkdir()
    smi=binary/'nvidia-smi'
    smi.write_text('#!'+sys.executable+'\nimport sys,os\n'
        'query=next((x for x in sys.argv if x.startswith("--query-")),"")\n'
        'ids=os.environ.get("CUDA_VISIBLE_DEVICES","0").split(",")\n'
        'if query=="--query-gpu=uuid": print("GPU-CPU-STANDIN-"+ids[0])\n'
        'elif "utilization.gpu" in query: print("\\n".join(f"{i}, GPU-CPU-STANDIN-{i}, 90, 1024" for i in range(8)))\n'
        'elif query=="--query-gpu=index,uuid": print("\\n".join(f"{i}, GPU-CPU-STANDIN-{i}" for i in range(8)))\n')
    smi.chmod(0o755)
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','CUDA_','M2_','M3_','UTIL_','KIT_LANE_'))}
    env.update(KIT=str(c.KIT),A4_CONFIG=str(config),JOB_PY=sys.executable,PATH=str(binary)+os.pathsep+os.environ['PATH'])
    log=root/'wrapper.log';calls=[]
    def call(task,expected=0):
        result=subprocess.run(['bash',str(c.KIT/'jobs/a4/run-task.sh'),task],env=env,capture_output=True,text=True,timeout=180)
        with log.open('a') as f:f.write(task+'\n'+result.stdout+result.stderr+'\n')
        assert result.returncode==expected,f'{task}: {result.returncode}\n'+result.stdout[-3000:]+result.stderr[-3000:]
        calls.append((task,result.returncode))
    # Missing prerequisite exits immediately before any GPU query/lease/watcher.
    call('A4-arrival-m1',75)
    call('A4-read',75)
    for row in jobs.inventory()['rows']:
        if row['id']==cfg['fail_task']:
            call(row['id'],1)
            call('A4-direct-s101-m1-export',75)
        call(row['id'])
    return NS(root=root,out=out,config=config,env=env,calls=calls,call=call)


def test_every_row_through_wrapper_pending_retry_failure_and_archive(rehearsal):
    r=rehearsal
    assert {task for task,code in r.calls if code==0}=={x['id'] for x in jobs.inventory()['rows']}
    assert ('A4-arrival-m1',75) in r.calls and ('A4-direct-s101-m1-train',1) in r.calls
    records,missing=a4_read.load(r.out)
    assert not missing and len(records)==418 and jobs.hardware(records)=='VERIFIED'
    assert len(list((r.out/'runs').glob('*/summary.json')))==27
    assert len(list((r.out/'runs').glob('*/nll-before.json')))==27
    for month in c.MONTHS:
        assert len(c.ledger(r.out,c.MONTHS.index(month)+1))==9*(c.MONTHS.index(month)+1)
    summary=json.loads((r.out/'reading/summary.json').read_text())
    returned=json.loads((r.out/'returned-reading/summary.json').read_text())
    assert summary==returned and summary['amendment']==c.AMENDMENT
    assert summary['primary']['interval']['df']==2 and summary['hardware_status']=='VERIFIED'
    assert summary['scientific_verdict'] is None
    assert '2026-11-02' in (r.out/'reading/summary.md').read_text()
    for task in ('A4-arrival-m1','A4-direct-s101-m1-train'):
        paths=list((r.out/'v6b/attempts'/task).glob('*/receipt.json'))
        assert len(paths)==2 and len({p.parent.name for p in paths})==2
    final=jobs.status(r.out,'A4-comparison')
    lane=json.loads((final/'lane/run/lane.json').read_text())
    assert lane['K']==3 and lane['gpu_memory_fraction']==.31 and lane['status']=='COMPLETE'
    assert len(lane['commands'])<418 # exact prompt/weight deduplication within allocation
    assert not (final/'nll-lane').exists() and not (final/'nll').exists()
    prepared=jobs.status(r.out,'A4-comparison-prepare')
    assert json.loads((prepared/'prepared.json').read_text())['nll']==[]
    stage=jobs.status(r.out,'A4-direct-s101-m1-score')
    assert len(list((stage/'nll').glob('*/result.json')))==2
    assert json.loads((stage/'nll-lane/run/lane.json').read_text())['K']==1
    identities={json.loads(p.read_text())['physical']['allocation_nonce'] for p in (final/'scores').glob('*/identity.json')}
    assert identities=={final.name}
    # Every allocation has exactly one complete watcher with the manifest GPU count.
    watchers=list((r.out/'v6b/attempts').glob('*/*/util.summary.json'))
    assert len(watchers)==199 # 197 rows, interrupted training, early afterany reader
    for path in watchers:
        data=json.loads(path.read_text());row=jobs.row_by_id(path.parent.parent.name)
        assert data['gpus']==row['gpus'] and data['telemetry_complete']
    # The deterministic controls are not replicated over seeds.
    assert len([k for k in records if k[0]=='arrival' and k[1]=='frozen'])==6
    assert len([k for k in records if k[0]=='arrival' and k[1]=='memory'])==6


@pytest.mark.parametrize('field,value',[('_allocation_job','another-attempt'),('_allocation_uuid','GPU-other'),('_allocation_environment','other-env')])
def test_mixed_evidence_is_labelled_without_admission_refusal(rehearsal,field,value):
    r=rehearsal;records,_=a4_read.load(r.out);records=copy.deepcopy(records)
    next(iter(records.values()))[0][field]=value
    summaries={p.parent.name:json.loads(p.read_text()) for p in (r.out/'runs').glob('*/summary.json')}
    bound=c.binding(r.out,True)
    result=jobs.write_reading(NS(standin=True,bootstrap=2),records,summaries,[],bound,c.s.file_hash(r.out/'binding.json'),r.root/(field+'-reading'))
    assert result['verdict']==result['hardware_status']=='UNVERIFIED-HARDWARE'


def test_missing_complete_comparison_never_joins_partial_attempts(rehearsal):
    r=rehearsal;pointer=r.out/'v6b/status/A4-comparison.json';saved=pointer.read_bytes()
    try:
        c.atomic(pointer,dict(status='PENDING'))
        with pytest.raises(c.NotReady):a4_read.load(r.out)
        assert jobs.read_comparison(r.out,False)==({},['A4-comparison'])
    finally:pointer.write_bytes(saved)


def test_watcher_expected_zero_bypasses_nvml_and_records_count(tmp_path):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    done=subprocess.run([sys.executable,'-I',str(c.KIT/'util_watch.py'),'--out',str(tmp_path/'util.jsonl'),'--expect-gpus','0','--',sys.executable,'-I','-c','raise SystemExit(75)'],env=env,capture_output=True,text=True)
    assert done.returncode==75
    summary=json.loads((tmp_path/'util.summary.json').read_text())
    assert summary['gpus']==0 and summary['telemetry_complete']


def test_real_launch_requires_external_authority_even_with_ambient_self_seal(tmp_path,monkeypatch):
    path=tmp_path/'config.json';c.atomic(path,dict(out=str(tmp_path/'A4'),technical_synthetic=False))
    monkeypatch.delenv('KIT_V6_AUTHORITY_SHA',raising=False)
    monkeypatch.setenv('M2_INFRA_AUTHORITY_SHA256','ambient-authority-is-not-launch-permission')
    with pytest.raises(c.CorrectnessStop,match='bound launch'):
        jobs.run(NS(config=path,task='A4-bind'))
    with pytest.raises(c.CorrectnessStop,match='bound v6b launch'):
        a4_bind.job(NS(v6b=True,standin=False))
    assert not (tmp_path/'A4').exists()


def test_budget_keeps_live_reservations_and_charges_actual_failed_allocations(tmp_path,monkeypatch):
    from kit import a4_common
    started=1000.;nonce='reserved-attempt';path=tmp_path/'v6b/attempts/A4-direct-s101-m1-train'/nonce/'receipt.json'
    c.atomic(path,dict(task='A4-direct-s101-m1-train',gpus=8,slurm_job_id='fake-101',
        allocation_nonce=nonce,allocation_started_at=started,allocation_ended_at=None,status='RUNNING',
        technical_synthetic=False,reserved_gpu_hours=12))
    monkeypatch.setattr(a4_common,'_slurm_times',lambda *a:{})
    monkeypatch.setattr(jobs.time,'time',lambda:2000.)
    assert jobs.reconcile(tmp_path,False,admission=True)==12
    with pytest.raises(c.NotReady,match='actual allocation end'):
        jobs.reconcile(tmp_path,False)
    # A failed job is charged by actual start/end, including all eight held GPUs.
    value=json.loads(path.read_text());value['status']='FAILED';c.atomic(path,value)
    monkeypatch.setattr(a4_common,'_slurm_times',lambda *a:dict(allocation_ended_at=4600.))
    assert jobs.reconcile(tmp_path,False)==8
    assert jobs.reconcile(tmp_path,False)==8
    costs=[json.loads(p.read_text()) for p in (tmp_path/'costs').glob('*.json')]
    assert sum(v['gpu_hours'] for v in costs)==8 and not costs[0]['completed']
    # The next job's queue gap cannot inflate the earlier allocation.
    monkeypatch.setattr(jobs.time,'time',lambda:100000.)
    assert jobs.reconcile(tmp_path,False)==8


def test_concurrent_score_seconds_are_normalised_to_one_allocation(tmp_path):
    attempt=tmp_path/'v6b/attempts/A4-arrival-m1'/'nonce';attempt.mkdir(parents=True)
    for index,seconds in enumerate((3600,3600,3600)):
        root=attempt/'scores'/str(index);root.mkdir(parents=True)
        c.atomic(root/'generated.json',dict(reused=False,seconds=seconds,cache_key='generation-'+str(index)))
        (root/'scores.csv').write_text('arm,seed,month,kind\nfrozen,0,1,arrival\n')
    value=dict(task='A4-arrival-m1',gpus=1,allocation_started_at=0,allocation_ended_at=3600,status='COMPLETE',technical_synthetic=False)
    jobs.cost_allocation(tmp_path,attempt/'receipt.json',value,1)
    costs=[json.loads(p.read_text()) for p in (tmp_path/'costs').glob('*.json')]
    assert sum(c['gpu_hours'] for c in costs)==pytest.approx(1)
    assert len([c for c in costs if c['kind']=='scoring' and c['gpu_hours']==pytest.approx(1/3)])==3


def test_native_training_stops_before_cpu_export_and_completed_dose_is_recovered(tmp_path,monkeypatch):
    row=jobs.row_by_id('A4-direct-s101-m1-train');run=jobs.stage_root(tmp_path,row)
    run.mkdir(parents=True);attempt=tmp_path/'attempt';attempt.mkdir()
    code='from pathlib import Path; import json; r=Path('+repr(str(run))+'); (r/"sdft/global_step_40/actor").mkdir(parents=True); (r/"metrics.jsonl").write_text(json.dumps({"v4/completed_optimizer_updates":40})+"\\n")'
    c.atomic(tmp_path/'data'/run.name/'prepared.json',dict(empty=False,observed=dict(technical_synthetic=True),command=[sys.executable,'-I','-c',code],env={},cwd=str(tmp_path)))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7')
    doc=dict(W='D',software=dict(trainer=c.t.runtime_versions()),technical_synthetic=False)
    c.freeze(tmp_path/'data'/run.name,['prepared.json'],technical_synthetic=True)
    jobs.native_train(tmp_path,doc,row,{},attempt)
    assert (run/'trained.json').exists() and not (run/'hf-step40').exists() and not (run/'export-40.json').exists()
    monkeypatch.setattr(jobs,'gpu_run',lambda *a,**kw:pytest.fail('complete native dose must not retrain'))
    jobs.native_train(tmp_path,doc,row,{},attempt)


def test_cpu_export_merges_without_visible_gpu(tmp_path,monkeypatch):
    from kit import v4_sft
    row=jobs.row_by_id('A4-direct-s101-m1-export');run=jobs.stage_root(tmp_path,row)
    c.atomic(run/'trained.json',dict(technical_synthetic=True))
    (run/'sdft/global_step_40/actor').mkdir(parents=True)
    c.atomic(run/'token-journals/one.json',dict(student_tokens=1,teacher_tokens=2,replay_target_tokens=0))
    c.atomic(tmp_path/'data'/run.name/'prepared.json',dict(empty=False,observed=dict(technical_synthetic=True),scheduled_tokens=dict(input=1,response_including_eos=1)))
    calls=[]
    def merge(command,**kw):
        calls.append(command);assert kw['env']['CUDA_VISIBLE_DEVICES']==''
        assert command[command.index('--local_dir')+1]==str(run/'sdft/global_step_40/actor')
        (run/'hf-step40').mkdir();return NS(returncode=0)
    monkeypatch.setattr(jobs.subprocess,'run',merge)
    monkeypatch.setattr(v4_sft,'complete_export',lambda p:p.exists())
    monkeypatch.setattr(jobs,'identity',lambda *a:dict(model_file_hashes={'fixture':'fixture'}))
    c.freeze(tmp_path/'data'/run.name,['prepared.json'],technical_synthetic=True)
    jobs.export_stage(tmp_path,dict(technical_synthetic=False,W='D'),row,dict(sdpo=str(tmp_path)))
    assert len(calls)==1 and (run/'export-40.json').exists()
    assert json.loads((run/'summary.json').read_text())['runtime_training_tokens']==3


def test_consistent_but_wrong_proof_is_checked_against_measured_lane(rehearsal):
    r=rehearsal;source=jobs.status(r.out,'A4-comparison')
    names=[r.out/'v6b/status/A4-comparison.json',source/'receipt.json',source/'coverage.json',source/'lane/run/lane.json']
    names += [p for p in (source/'scores').rglob('*') if p.is_file() and 'raw' not in p.parts]
    files={str(p.relative_to(r.out)):p.read_bytes() for p in names}
    for name in list(files):
        if not name.endswith('/identity.json'):continue
        ident=json.loads(files[name]);ident['physical']['uuid']='GPU-consistent-but-not-measured'
        files[name]=json.dumps(ident).encode()
        manifest_name=name.rsplit('/',1)[0]+'/manifest.json';manifest=json.loads(files[manifest_name])
        manifest['files']['identity.json']=c.t.sha(files[name]);files[manifest_name]=json.dumps(manifest).encode()
    records=jobs.comparison_files(files,c.binding(r.out,True),c.s.file_hash(r.out/'binding.json'))
    assert jobs.hardware(records)=='UNVERIFIED-HARDWARE'


def test_current_allocation_uuid_is_converted_without_historical_pinning(monkeypatch):
    from kit import m2_gpu
    monkeypatch.setattr(m2_gpu,'devices',lambda:{'5':'GPU-current','6':'GPU-other'})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-current')
    monkeypatch.setenv('M2_ASSIGNED_GPU_UUID','GPU-historical')
    assert jobs.visible_devices(['GPU-current'],False)=='5'
    assert os.environ['M2_ASSIGNED_GPU_UUID']=='GPU-current'
    assert jobs.visible_devices([],False)==''


def test_prepared_training_bytes_are_verified_before_any_gpu_work(tmp_path,monkeypatch):
    row=jobs.row_by_id('A4-direct-s101-m1-train');root=tmp_path/'data/direct-s101-m1'
    c.atomic(root/'prepared.json',dict(empty=True))
    c.freeze(root,['prepared.json'])
    c.atomic(root/'prepared.json',dict(empty=False))
    monkeypatch.setattr(jobs,'gpu_run',lambda *a,**kw:pytest.fail('tampered inputs must stop before GPU work'))
    with pytest.raises(c.CorrectnessStop,match='input hash differs'):
        jobs.native_train(tmp_path,{},row,{},tmp_path/'attempt')


def test_frozen_scientific_and_extra_comparison_prediction_counts():
    months=dict(zip(c.MONTHS,[37,80,80,75,80,43]));holdouts=dict(zip(c.MONTHS,[5,24,19,14,20,10]))
    def count(row):
        kind,arm,seed,n,p,mode=row
        return 300 if p=='general-v1' else months[p] if kind=='arrival' else holdouts[p]
    scientific=list(__import__('kit.a4_run',fromlist=['score_plan']).score_plan())
    assert sum(map(count,scientific))==25381
    chronological=[r for r in scientific if r[0] in ('arrival','incoming','post')]
    assert len(chronological)==255 and sum(map(count,chronological))==22597
    assert sum(map(count,chronological+scientific))==47978


def test_gpu_admission_uses_cached_reservations_without_scheduler_waits(tmp_path,monkeypatch):
    from kit import a4_common
    path=tmp_path/'v6b/attempts/A4-direct-s101-m1-train'/'active'/'receipt.json'
    c.atomic(path,dict(task='A4-direct-s101-m1-train',gpus=8,slurm_job_id='fake-101',
        allocation_nonce='active',allocation_started_at=1000,allocation_ended_at=None,status='RUNNING',
        ended_at=None,technical_synthetic=False,reserved_gpu_hours=12))
    monkeypatch.setattr(a4_common,'_slurm_times',lambda *a:pytest.fail('GPU admission must not query/wait on accounting'))
    monkeypatch.setattr(jobs.time,'time',lambda:2000.)
    assert jobs.reconcile(tmp_path,False,admission=True,query_scheduler=False)==12
    assert not (tmp_path/'costs').exists()


@pytest.mark.parametrize('module',('a4_run','a4_train','a4_score','a4_handoff'))
def test_real_legacy_mixed_allocation_entrypoints_are_retired(module):
    import importlib
    with pytest.raises(c.CorrectnessStop,match='retired'):
        importlib.import_module('kit.'+module).job(NS(standin=False))


def test_real_binding_cannot_silently_choose_legacy_gpu_pinning():
    with pytest.raises(c.CorrectnessStop,match='--v6b'):
        a4_bind.job(NS(standin=False))


def test_duration_reserve_rounds_up_complete_concurrency_waves():
    row=dict(gpus=1,expected_command_counts=dict(score=4,nll=2))
    doc=dict(job_limits=dict(training=4500,scoring=600,nll=600))
    assert jobs.duration_limit(doc,row)==2460 # two 600-second waves, paired NLL, startup
    assert jobs.duration_limit(doc,dict(gpus=8))==4500


def test_rows_supply_sbatch_time_per_type_and_scale_to_actual_limits(tmp_path):
    rows={r['id']:r for r in jobs.inventory()['rows']}
    for task,expected in [('A4-bind','04:00:00'),('A4-direct-s101-m1-train','01:30:00'),
        ('A4-arrival-m1','01:00:00'),('A4-direct-s101-m6-score','01:30:00'),('A4-comparison','26:00:00')]:
        assert rows[task]['time']==expected
    path=tmp_path/'config.json';output=tmp_path/'rows.json'
    c.atomic(path,dict(stage_gpu_hours=10,score_max_seconds=1200,nll_max_seconds=900))
    done=subprocess.run([sys.executable,'-I',str(c.KIT/'a4_jobs.py'),'rows','--config',str(path),'--out',str(output)],
        capture_output=True,text=True)
    assert done.returncode==0,done.stdout+done.stderr
    doc=json.loads(output.read_text())
    assert doc['scheduling_job_limits']==dict(training=6750,scoring=1200,nll=900)
    for row in doc['rows']:
        if not row['gpus']:continue
        hard=jobs.duration_limit(dict(job_limits=doc['scheduling_job_limits']),row)
        assert row['time_limit_seconds']>=hard+max(900,.1*hard)
        hours,minutes,seconds=map(int,row['time'].split(':'))
        assert hours*3600+minutes*60+seconds==row['time_limit_seconds']


def test_real_cpu_wrapper_never_queries_nvml_or_acquires_gpu_leases(tmp_path,monkeypatch):
    from kit import m3_manifest,v6_infra
    def forbidden(*a,**kw):pytest.fail('CPU row must not query or lease GPUs')
    monkeypatch.setattr(m3_manifest,'allocation_reservation',forbidden)
    monkeypatch.setattr(m3_manifest,'allocated_gpus',forbidden)
    binary=tmp_path/'bin';binary.mkdir();smi=binary/'nvidia-smi';marker=tmp_path/'nvml-called'
    smi.write_text('#!'+sys.executable+'\nfrom pathlib import Path\nPath('+repr(str(marker))+').touch()\nraise SystemExit(1)\n');smi.chmod(0o755)
    monkeypatch.setenv('PATH',str(binary)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('KIT_V6_AUTHORITY_SHA',v6_infra.authority_sha256()) # explicit mock launch seal
    monkeypatch.setenv('SLURM_JOB_ID','cpu-test-allocation')
    monkeypatch.setenv('SLURM_JOB_START_TIME',str(jobs.time.time()-1))
    monkeypatch.delenv('UTIL_WATCH_ACTIVE',raising=False)
    path=tmp_path/'config.json';c.atomic(path,dict(out=str(tmp_path/'A4'),technical_synthetic=False))
    assert jobs.run(NS(config=path,task='A4-read'))==75 # afterany, unbound reader
    assert not marker.exists()
    summary=next((tmp_path/'A4/v6b/attempts/A4-read').glob('*/util.summary.json'))
    record=json.loads(summary.read_text())
    assert record['gpus']==record['expected_gpus']==0 and record['telemetry_complete']


def test_cpu_bind_uses_inference_environment_and_probes_trainer_separately(tmp_path,monkeypatch):
    path=tmp_path/'config.json'
    c.atomic(path,dict(out=str(tmp_path/'A4'),technical_synthetic=True,
        trainer_python='TRAINER-PY',inference_python='INFERENCE-PY'))
    monkeypatch.delenv('UTIL_WATCH_ACTIVE',raising=False)
    calls=[]
    def record(command,**kw):calls.append(command);return NS(returncode=0)
    monkeypatch.setattr(jobs,'gpu_run',record)
    assert jobs.run(NS(config=path,task='A4-bind'))==0
    command=calls[0]
    assert command[command.index('--')+1]=='INFERENCE-PY'


def complete_pointer(out,task):
    root=out/'retained'/task
    c.atomic(root/'artifact.json',dict(verified=True))
    c.atomic(root/'receipt.json',dict(task=task,status='COMPLETE',artifacts={
        str((root/'artifact.json').relative_to(out)):c.s.file_hash(root/'artifact.json')}))
    c.atomic(out/'v6b/status'/(task+'.json'),dict(status='COMPLETE',
        receipt=str((root/'receipt.json').relative_to(out)),sha256=c.s.file_hash(root/'receipt.json')))
    return root


@pytest.mark.parametrize('task',('A4-bind','A4-arrival-m3','A4-direct-s101-m1-train','A4-read'))
def test_verified_complete_resubmission_is_noop_before_attempt_or_gpu_admission(tmp_path,monkeypatch,task):
    out=tmp_path/'A4';path=tmp_path/'config.json'
    c.atomic(path,dict(out=str(out),technical_synthetic=True))
    root=complete_pointer(out,task);pointer=out/'v6b/status'/(task+'.json');saved=pointer.read_bytes()
    monkeypatch.setattr(jobs,'gpu_run',lambda *a,**kw:pytest.fail('completed row must not run again'))
    monkeypatch.setattr(jobs,'reconcile',lambda *a,**kw:pytest.fail('completed row must not create a reservation'))
    assert jobs.run(NS(config=path,task=task))==0
    assert pointer.read_bytes()==saved and jobs.status(out,task)==root
    assert not (out/'v6b/attempts').exists()


@pytest.mark.parametrize('name',('receipt.json','artifact.json'))
def test_complete_pointer_must_verify_before_noop_and_is_never_overwritten(tmp_path,monkeypatch,name):
    out=tmp_path/'A4';path=tmp_path/'config.json';task='A4-arrival-m3'
    c.atomic(path,dict(out=str(out),technical_synthetic=True))
    root=complete_pointer(out,task);pointer=out/'v6b/status'/(task+'.json');saved=pointer.read_bytes()
    (root/name).write_text('{}')
    monkeypatch.setattr(jobs,'gpu_run',lambda *a,**kw:pytest.fail('invalid completion must fail closed'))
    with pytest.raises(c.CorrectnessStop,match='hash differs'):jobs.run(NS(config=path,task=task))
    assert pointer.read_bytes()==saved and not (out/'v6b/attempts').exists()


@pytest.mark.parametrize('exit_code',(0,1,75))
def test_inflight_duplicate_cannot_downgrade_successful_peer(tmp_path,monkeypatch,exit_code):
    out=tmp_path/'A4';path=tmp_path/'config.json';task='A4-read';retained=[]
    c.atomic(path,dict(out=str(out),technical_synthetic=True))
    def finish_peer(*a,**kw):
        retained.append(complete_pointer(out,task))
        return NS(returncode=exit_code)
    monkeypatch.setattr(jobs,'gpu_run',finish_peer)
    assert jobs.run(NS(config=path,task=task))==0
    assert jobs.status(out,task)==retained[0]
    attempts=list((out/'v6b/attempts'/task).glob('*/receipt.json'))
    assert len(attempts)==1
    assert json.loads(attempts[0].read_text())['status']=={0:'COMPLETE',1:'FAILED',75:'PENDING'}[exit_code]
