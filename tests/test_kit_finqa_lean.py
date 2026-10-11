"""Owner's main timing policy, real row entry points and reusable caches; CPU only."""
import json
from pathlib import Path
import subprocess
import pytest
from kit import v4_phase0_timing as timing,v4_finqa_campaign as campaign

@pytest.mark.parametrize('mode',['V4_FINQA_CAMPAIGN','V4_PHASE0_MODE'])
def test_shared_limits_scaled_once_without_changing_observations(monkeypatch,mode):
    monkeypatch.setenv(mode,'1')
    assert timing.TIMING_MULTIPLIER==10
    assert timing.shared_limit(120)==1200
    for phase in ('qualification','main'):
        graph=campaign.build(phase)
        assert graph['v4']['timing_multiplier']==10
        assert (graph['v4']['qualification_limit'],graph['v4']['ceiling'])==(1000,1000)
        assert graph['v4']['containment_selftest']=='informational'
        row=next(r for r in graph['rows'] if r['id']=='teacher-finqa')
        cap=int(row['command'][row['command'].index('--time-cap')+1])
        assert cap==144000
        assert row['timeout_seconds']==cap+1200
    source=campaign.phase0_measured_launch_allowances()
    assert source['qualification_launch_seconds']==300 and source['restoration_leg_seconds']==450

@pytest.mark.parametrize('error',[ValueError('failed'),RuntimeError('failed'),subprocess.CalledProcessError(1,['trainer'])])
def test_failed_main_operation_has_failure_receipt_and_no_pass(tmp_path,monkeypatch,error):
    from kit import v4_campaign_ops as ops,v4_prepare as prep
    monkeypatch.setattr(prep,'verify',lambda *a:None)
    monkeypatch.setattr(ops,'train',lambda *a:(_ for _ in ()).throw(error))
    assert ops.main(['train','--work',str(tmp_path),'--phase','qualification','--row','train-q-S-finqa','--slot','q-S-finqa'])==1
    assert not (tmp_path/'v4/report-status/train-q-S-finqa.json').exists()
    assert 'failed' in (tmp_path/'v4/report-qualification/failures/train-q-S-finqa.json').read_text() or isinstance(error,subprocess.CalledProcessError)

@pytest.mark.parametrize('phase',['qualification','main'])
def test_incomplete_real_report_consumer_fails_row(tmp_path,monkeypatch,phase):
    from kit import v4_campaign_ops as ops,v4_readers as readers
    def incomplete(*a):return dict(status='incomplete',reasons=['missing checkpoint receipt'],tables={})
    monkeypatch.setattr(readers,'qualification_evidence',incomplete)
    monkeypatch.setattr(readers,'campaign_evidence',incomplete)
    assert ops.main(['report','--work',str(tmp_path),'--phase',phase,'--row','report'])==0
    assert (tmp_path/'v4/report-status/report.json').exists()
    assert list((tmp_path/'v4/warnings').glob('*.json'))
    assert json.loads((tmp_path/('v4/report-'+phase)/'reading.json').read_text())['status']=='progress report'


def test_fresh_work_reuses_caches_and_collector_excludes_them(tmp_path,monkeypatch):
    from kit.v4_phase0_site import storage_environment,CACHE_KEYS
    from kit import collect
    import tarfile
    monkeypatch.setenv('TASK_ROOT',str(tmp_path/'task'))
    first=storage_environment(tmp_path/'one');second=storage_environment(tmp_path/'two')
    for key in CACHE_KEYS:
        if key=='TMPDIR':assert first[key]!=second[key]
        else:
            assert first[key]==second[key]
            assert Path(first[key]).is_relative_to(tmp_path/'task/cache')
    Path(first['PIP_CACHE_DIR'],'wheel.json').write_text('cached wheel')
    work=tmp_path/'two';(work/'report').mkdir();(work/'report/measurement.json').write_text('{}')
    output=tmp_path/'return.tar.gz'
    assert collect.main(['--work',str(work),'--out',str(output)])==0
    with tarfile.open(output) as archive:
        assert not any('wheel.json' in n or 'phase0-cache' in n for n in archive.getnames())


def test_informational_selftest_keeps_hash_binding_hard(tmp_path,monkeypatch):
    from kit import p4_contain as pc
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1')
    monkeypatch.setenv('SLURM_JOB_ID','123')
    receipt=pc.frozen_write(tmp_path/pc.RECEIPT,{'ok':False,'job_id':'123','simulation':False})
    pc.frozen_write(tmp_path/pc.SELFTEST,{'ok':False,'receipt_sha256':receipt['content_sha256']})
    problems,docs=pc.verify_receipts(tmp_path)
    assert not problems and docs['selftest']['informational']
    assert pc.stop_class('termination outside time cap',True,False)=='row_failure'
    (tmp_path/pc.RECEIPT).write_text('{}')
    assert pc.verify_receipts(tmp_path)[0]


def test_huge_timeout_envelope_is_not_a_tag_budget_failure():
    from kit.v4_finqa_ingest import report_categories,qualification_fit
    graph=campaign.build('qualification')
    fit=qualification_fit(graph,12.06,{'qualification':12.06})
    assert fit['qualification_gpu_hours']>100 and fit['clipped_minutes']==0
    report=dict(reader={'status':'technical pass'},refusals=[],comparisons=[],campaigns={'qualification':{'c1_envelope_accepted':True}},
      budget=dict(actual_gpu_hours=12.06,qualification_fit=fit,phase0_qualification_gpu_hours=12.06,qualification_envelope_gpu_hours=fit['qualification_gpu_hours']-12.06,
                  qualification_total_gpu_hours=fit['qualification_gpu_hours'],missing_timings=[]))
    report_categories(report)
    assert not any(b['status']=='FAIL' for b in report['tag_blockers'])


def test_main_row_can_start_with_generous_cap_and_keeps_paid_deadline(tmp_path,monkeypatch):
    from kit import p4_contain as pc,p4_watchdog as wd
    import time
    monkeypatch.setenv('V4_FINQA_CAMPAIGN','1')
    monkeypatch.setenv('SLURM_JOB_ID','123')
    now=time.time();end=wd.precise_text(wd.from_epoch(now+1800))
    monkeypatch.setattr(pc,'track_allocation',lambda *a,**k:None)
    monkeypatch.setattr(pc,'stop_gate',lambda *a:None)
    monkeypatch.setattr(pc,'verify_receipts',lambda *a,**k:([],{'receipt':{}}))
    monkeypatch.setattr(pc,'_live_cluster',lambda *a:dict(job_id='123',gpu_count=8))
    monkeypatch.setattr(pc,'allocation_budget',lambda *a:dict(admitted_at=wd.precise_text(wd.from_epoch(now)),allocations={'123':{'planned_end':end}}))
    monkeypatch.setattr(pc,'allocation_observation',lambda *a:dict(job_id='123',width=8,start=wd.precise_text(wd.from_epoch(now)),time_limit_seconds=1800))
    monkeypatch.setattr(pc,'allocation_budget',lambda *a:dict(spent_gpu_hours=0,allocations={'123':{'planned_end':end}}))
    from kit import runner
    observed=[]
    def execute(argv,**kwargs):
        observed.append((argv,kwargs['timeout']));kwargs['log'].write_text('CPU external step adapter\n');return dict(returncode=0)
    monkeypatch.setattr(runner,'bounded_command',execute)
    monkeypatch.setattr(pc,'refresh_allocation_ledger',lambda *a:None)
    result=pc.run_row(tmp_path,tmp_path/'row',['trainer'],8,108000,block='scientific',block_limit=1000,ceiling=1000)
    assert result['ok']==1 and result['requested_time_cap_seconds']==108000
    assert result['time_cap_seconds']<=1800
    assert observed[0][1]<=1800
    clock=next(arg.split('=',1)[1] for arg in observed[0][0] if arg.startswith('--time='))
    h,m,s=map(int,clock.split(':'));assert h*3600+m*60+s<=1800


def test_large_verification_allowance_reaches_real_runner_cap(tmp_path,monkeypatch):
    import hashlib
    from kit import runner,v4_prepare as prep
    phase='qualification';receipt=tmp_path/'v4/report-inputs/prepare-receipt.json';receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'verification_seconds':300,'verification_allowance_seconds':4500}))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(phase=phase,verification_allowance_seconds=4500,verification_row_cap_seconds=4500,prepare_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())))
    monkeypatch.setenv('V4_ALLOCATION_PLAN',str(plan));monkeypatch.setenv('V4_ALLOCATION_PLAN_SHA256',hashlib.sha256(plan.read_bytes()).hexdigest())
    monkeypatch.setenv('WORK',str(tmp_path));monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    graph=campaign.build(phase);row=next(r for r in graph['rows'] if r['id']=='prepare');row.update(needs=[],pilot=False,bars=[])
    graph['rows']=[row];path=tmp_path/'graph.json';path.write_text(json.dumps(graph));loaded=runner.load_campaign(path)
    seen=[]
    def execute(argv,**kw):
        seen.append(kw['timeout']);kw['log'].parent.mkdir(parents=True,exist_ok=True);kw['log'].write_text('verified\n')
        return dict(returncode=0,failure_type=None)
    monkeypatch.setattr(runner,'bounded_command',execute)
    assert runner.run_row(loaded,loaded['rows'][0],None)==0 and seen==[4500]

@pytest.mark.parametrize('rewriting',[False,True])
def test_main_generation_selects_pinned_v7_engine_route(monkeypatch,rewriting,tmp_path):
    from kit.v4_generate_shards import shard_commands
    for key,value in dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',V4_TEACHER_PYTHON='/inference/python',MODEL_DIR='/8b',TEACHER_MODEL_DIR='/27b',QWEN3_8B_TOKENIZER='/tokenizer').items():monkeypatch.setenv(key,value)
    commands=shard_commands(tmp_path,'main','finqa',rewriting)
    assert len(commands)==(8 if rewriting else 4)
    assert all(env['V4_PHASE0_INFERENCE']==('0' if rewriting else '1') for _,env in commands)
    assert all(argv[0]==('/inference/python' if not rewriting else __import__('sys').executable) for argv,_ in commands)


def test_fr_uses_v7_distributed_defaults_without_cuda(monkeypatch):
    import datetime
    import torch.distributed.distributed_c10d as c10d
    monkeypatch.setattr(c10d,'default_pg_timeout',datetime.timedelta(seconds=1800))
    monkeypatch.setattr(c10d,'default_pg_nccl_timeout',datetime.timedelta(seconds=600))
    timing.distributed_defaults()
    assert c10d.default_pg_timeout.total_seconds()==18000
    assert c10d.default_pg_nccl_timeout.total_seconds()==6000


def test_main_allocation_planning_windows_are_generous_not_new_budgets():
    from kit import v4_budget as budget
    assert budget.DEFAULT_WINDOWS=={'teacher':3000,'rewrite':3000,'scientific':3600}
    assert budget.CEILING==1000 and budget.QUALIFICATION_LIMIT==100
