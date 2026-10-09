"""Owner-approved v4 timing and informational containment policy."""
import json
from pathlib import Path
import pytest


def test_v4_limits_and_registration(monkeypatch):
    """One multiplier scales v3 time limits without scaling non-time thresholds."""
    from kit import v4_phase0 as p,p4_contain as pc
    from kit.v4_phase0_timing import TIMING_MULTIPLIER,shared_limit
    from kit.v4_phase0_environment import environment_timing
    assert TIMING_MULTIPLIER==10 and p.ALLOCATION_CAP_GPU_HOURS==192
    assert p.verification_allowance(84)==1260
    assert environment_timing({'trainer':1,'inference':400})['deadline_seconds']==23050
    doc=p.campaign();rows=doc['rows']
    assert doc['v4']['maximum_gpu_hours']==192
    assert rows[0]['pilot'] is False and rows[0]['bars']==[]
    for row in rows:
        if '--time-cap' in row['command']:
            cap=float(row['command'][row['command'].index('--time-cap')+1])
            assert cap in (6000,15000)
    monkeypatch.setenv('V4_PHASE0_MODE','1')
    assert shared_limit(pc.SELFTEST_SECONDS)==6000
    assert shared_limit(pc.GPU_PROBE_SECONDS)==1600
    assert p.required_seconds(6000,1260)==278160
    from kit import v4_campaign_ops as ops
    calls=[]
    monkeypatch.delenv('V4_COMMAND_TIMEOUT',raising=False)
    monkeypatch.setattr(ops.subprocess,'run',lambda *a,**kw:calls.append(kw['timeout']))
    ops.run(['owned-stand-in'])
    assert calls==[18000]


def test_failed_real_selftest_is_full_informational_receipt(tmp_path,monkeypatch):
    """A real self-test refusal preserves evidence without setting a downstream stop."""
    from kit import p4_contain as pc,p4_frozen
    monkeypatch.setenv('V4_PHASE0_MODE','1');monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setattr(pc,'stop_gate',lambda *a:None)
    def fail(*args):raise pc.Refused('device restriction failed')
    monkeypatch.setattr(pc,'_selftest',fail)
    doc=pc._recorded_selftest(tmp_path,tmp_path/'out',50,None,'qualification',100,560)
    assert not doc['ok'] and doc['informational'] and 'FAILED' in doc['warning'] and 'device restriction' in doc['problems'][0]
    assert doc==p4_frozen.seal(doc)
    assert not (tmp_path/pc.DIRECTORY/'p4-v4-stop.json').exists()


def test_failed_selftest_reaches_training_report_and_pause(tmp_path):
    """The real runner must continue after a failed self-test and prominently report it."""
    from kit.simulate_v4_phase0 import rehearse
    doc=rehearse(tmp_path/'rehearsal','failed_selftest')
    assert doc['training_started'] and doc['pause']['state']=='PAUSE'
    assert 'selftest' in json.dumps(doc['reader']).lower()
    assert doc['graph_fits_registered_reservation'] is False
    assert doc['expected_gpu_hours']==15


def test_real_phase0_step_runs_without_passing_selftest(tmp_path,monkeypatch):
    """The real Slurm-row launch path must not read a passing containment receipt."""
    import os,sys,time,hashlib
    from kit import p4_contain as pc
    work=tmp_path/'work';work.mkdir();bindir=tmp_path/'bin';bindir.mkdir()
    monkeypatch.setenv('V4_PHASE0_MODE','1');monkeypatch.setenv('SLURM_JOB_ID','123')
    plan=work/'allocation.json';plan.write_text(json.dumps({'phase':'phase0','phase0_allocation_cap_gpu_hours':192}))
    monkeypatch.setenv('V4_ALLOCATION_PLAN',str(plan));monkeypatch.setenv('V4_ALLOCATION_PLAN_SHA256',hashlib.sha256(plan.read_bytes()).hexdigest())
    monkeypatch.setenv('PATH',str(bindir)+os.pathsep+os.environ['PATH'])
    executable=bindir/'srun'
    executable.write_text('#!'+sys.executable+'\nimport os,sys\na=sys.argv[1:]\nwhile a and a[0].startswith("--"):a.pop(0)\nos.environ.update(SLURM_STEP_ID="1",CUDA_VISIBLE_DEVICES="0")\nos.execvpe(a[0],a,os.environ)\n');executable.chmod(0o755)
    begin=pc.wd.precise_text(pc.wd.from_epoch(time.time()-1))
    monkeypatch.setattr(pc,'allocation_observation',lambda job:{'job_id':'123','width':8,'start':begin,'end':None,'time_limit_seconds':86400,'cpus':32,'memory_mb':64000})
    pc.frozen_write(work/pc.DIRECTORY/'containment-selftest.json',{'ok':False,'v4_qualified':False,'problems':['stand-in failure']})
    result=pc.run_row(work,work/'row',[sys.executable,'-c','print("payload executed")'],1,15000)
    assert result['ok']==1 and result['informational'] and result['time_cap']==15000
    assert result['slurm']['SLURM_STEP_ID']=='1' and '--gres=gpu:1' in result['actual_command']
    assert 'payload executed' in (work/'row/phase0-step.log').read_text()
    assert json.loads((work/pc.DIRECTORY/'allocation-ledger.json').read_text())['allocations']['123']['phase0']


def test_agreement_records_scheduler_failure_without_a_pass_gate(monkeypatch,capfd):
    """An informational scheduler readout must not crash the actual agreement operation."""
    from contextlib import nullcontext
    from kit import v4_phase0 as p,v4_evidence as e,v4_readers as r,v4_qualification as q
    monkeypatch.setattr(r,'scorer_context',lambda *a:(None,None))
    monkeypatch.setattr(p,'finqa_context',lambda *a:nullcontext())
    def failure(*a):raise ValueError('self-test did not pass')
    monkeypatch.setattr(e,'scheduler_rows',failure)
    monkeypatch.setattr(q,'archive_agreement',lambda *a:{'pairs':'recorded'})
    assert p.agreement(object())=={'pairs':'recorded'}
    assert 'WARNING' in capfd.readouterr().err
