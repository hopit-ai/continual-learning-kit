"""Allocation telemetry uses a fake NVML CLI only; never touches a GPU."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kit import util_watch as watch, util_phase


@pytest.fixture
def env(tmp_path):
    binary=tmp_path/'nvidia-smi'
    binary.write_text('#!'+sys.executable+'\nimport os\nfrom pathlib import Path\n'
        'Path(os.environ["FAKE_CALLED"]).touch()\n'
        'print(os.environ.get("FAKE_GPU_ROWS","0, GPU-a, 20, 120\\n1, GPU-b, 80, 340"))\n')
    binary.chmod(0o755)
    result={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','CUDA_','NVIDIA_','UTIL_'))}
    result.update(PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],CUDA_VISIBLE_DEVICES='0,1',FAKE_CALLED=str(tmp_path/'queried'))
    return result


def argv(out,code,*,window='10'):
    return [sys.executable,'-I',str(ROOT/'kit/util_watch.py'),'--out',str(out),'--interval','.05',
            '--window',window,'--',sys.executable,'-I','-c',code]


def run(tmp_path,env,code='import time;time.sleep(.25)',**kw):
    out=tmp_path/'util.jsonl'
    done=subprocess.run(argv(out,code,**kw),env=env,capture_output=True,text=True,timeout=10)
    return done,[json.loads(s) for s in out.read_text().splitlines()],json.loads(out.with_suffix('.summary.json').read_text())


def test_exit_passthrough_summary_scoped_uuid_and_phases(tmp_path,env):
    env.update(CUDA_VISIBLE_DEVICES='GPU-b',SLURM_GPUS_ON_NODE='1',UTIL_JOB='local-attempt')
    code='import os,json,time;open(os.environ["UTIL_PHASE_FILE"],"a").write(json.dumps({"phase":"score","t":time.time()})+"\\n");time.sleep(.2);raise SystemExit(37)'
    done,samples,summary=run(tmp_path,env,code)
    assert done.returncode==37 and {s['gpu'] for s in samples}=={'GPU-b'}
    assert all(set(s)=={'t','gpu','util','mem'} and s['mem']==340 for s in samples)
    assert summary['job']=='local-attempt' and summary['gpus']==1
    assert summary['returncode']==37 and summary['telemetry_complete'] is True
    assert summary['wall_s']>0 and summary['mean_util']==pytest.approx(80)
    assert summary['min_window_mean'] is None and summary['seconds_below_70']==0
    assert summary['phases'][0]['phase']=='score'


@pytest.mark.parametrize('visibility',['','-1','NoDevFiles'])
def test_zero_gpu_never_queries_host(tmp_path,env,visibility):
    env['CUDA_VISIBLE_DEVICES']=visibility
    done,samples,summary=run(tmp_path,env,'raise SystemExit(75)')
    assert done.returncode==75 and samples==[] and not Path(env['FAKE_CALLED']).exists()
    assert summary['gpus']==0 and summary['gpu_uuids']==[] and summary['telemetry_complete']
    assert summary['mean_util'] is None and summary['min_window_mean'] is None


def test_zero_gpus_reported_by_nvml_without_visibility_env(tmp_path,env):
    env.pop('CUDA_VISIBLE_DEVICES');env['FAKE_GPU_ROWS']=''
    done,samples,summary=run(tmp_path,env,'raise SystemExit(75)')
    assert done.returncode==75 and not samples and summary['gpus']==0
    assert summary['telemetry_complete'] and summary['mean_util'] is None


def test_rolling_warn_is_rate_limited_and_below_time_recorded(tmp_path,env):
    done,samples,summary=run(tmp_path,env,window='.1')
    assert done.returncode==0 and len(samples)>=4
    assert 1<=done.stderr.count('WARN utilisation rolling mean')<=int(summary['wall_s']/.1)+1
    assert summary['mean_util']==pytest.approx(50) and summary['min_window_mean']==pytest.approx(50)
    assert 0<summary['seconds_below_70']<=summary['wall_s']


def test_time_weighted_rolling_window():
    stats=watch.Rolling(10)
    assert stats.add(5,100)==100
    assert stats.add(15,0)==0
    assert stats.total/stats.elapsed==pytest.approx(100/3)
    assert stats.add(20,100)==50
    assert stats.below==pytest.approx(10) and stats.minimum==0


def test_phase_jsonl_preserves_unicode_line_separators(tmp_path):
    phase='score\u2028checkpoint\u2029panel\u0085done'
    path=tmp_path/'phases.jsonl'
    path.write_text(json.dumps(dict(phase=phase,t=1),ensure_ascii=False)+'\n')
    assert watch.phases(path)==[dict(phase=phase,t=1)]


@pytest.mark.parametrize('scope',['1','GPU-b','GPU-'])
def test_allocated_set_is_frozen_or_ambiguous(tmp_path,env,scope):
    env['CUDA_VISIBLE_DEVICES']=scope
    done,samples,summary=run(tmp_path,env)
    assert done.returncode==0
    if scope=='GPU-':
        assert not samples and not summary['telemetry_complete'] and summary['mean_util'] is None
    else:assert {s['gpu'] for s in samples}=={'GPU-b'} and summary['gpus']==1


def test_slurm_scope_and_count_mismatch_are_not_host_samples(tmp_path,env):
    env.pop('CUDA_VISIBLE_DEVICES');env.update(SLURM_JOB_GPUS='1',SLURM_GPUS_ON_NODE='2')
    done,samples,summary=run(tmp_path,env)
    assert done.returncode==0 and not samples and summary['telemetry_complete'] is False
    assert summary['mean_util'] is None and summary['gpus']==1


def test_remapped_cuda_uses_slurm_physical_mapping(tmp_path,env):
    env.update(CUDA_VISIBLE_DEVICES='0',SLURM_JOB_GPUS='1',SLURM_GPUS_ON_NODE='1')
    done,samples,summary=run(tmp_path,env)
    assert done.returncode==0 and samples and summary['telemetry_complete']
    assert summary['gpu_uuids']==['GPU-b'] and summary['mean_util']==80


@pytest.mark.parametrize('slurm',[dict(SLURM_GPUS_ON_NODE='8'),dict(SLURM_JOB_GPUS='0')])
def test_empty_cuda_cannot_claim_zero_when_slurm_allocated_gpus(tmp_path,env,slurm):
    env.update(CUDA_VISIBLE_DEVICES='',**slurm)
    done,samples,summary=run(tmp_path,env)
    assert done.returncode==0 and not samples and not summary['telemetry_complete']
    assert summary['mean_util'] is None and not Path(env['FAKE_CALLED']).exists()


def test_broken_nvml_does_not_change_child_exit_code(tmp_path,env):
    env['FAKE_GPU_ROWS']='0, GPU-a, N/A, 120'
    done,samples,summary=run(tmp_path,env,'raise SystemExit(9)')
    assert done.returncode==9 and not samples
    assert not summary['telemetry_complete'] and summary['mean_util'] is None


def test_signals_forwarded_to_child_process_group(tmp_path,env):
    env['CUDA_VISIBLE_DEVICES']=''
    ready=tmp_path/'ready';grand_signal=tmp_path/'grand-signal'
    grand='import signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,lambda s,f:(Path('+repr(str(grand_signal))+').write_text(str(s)),exit(0)));Path('+repr(str(ready))+').touch();time.sleep(10)'
    child='import signal,time,subprocess,sys;signal.signal(signal.SIGTERM,lambda s,f:(time.sleep(.2),sys.exit(42)));subprocess.Popen([sys.executable,"-I","-c",'+repr(grand)+']);time.sleep(10)'
    out=tmp_path/'util.jsonl';proc=subprocess.Popen(argv(out,child),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        until=time.monotonic()+5
        while not ready.exists() and time.monotonic()<until:time.sleep(.01)
        assert ready.exists()
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=5)
        assert proc.returncode==42 and grand_signal.read_text()==str(signal.SIGTERM)
        assert json.loads(out.with_suffix('.summary.json').read_text())['returncode']==42
    finally:
        if proc.poll() is None:proc.kill();proc.wait()


def test_signal_exit_code_uses_shell_convention(tmp_path,env):
    env['CUDA_VISIBLE_DEVICES']=''
    done,_,summary=run(tmp_path,env,'import os,signal;os.kill(os.getpid(),signal.SIGTERM)')
    assert done.returncode==128+signal.SIGTERM and summary['returncode']==-signal.SIGTERM


def test_inherited_lease_descriptor_reaches_child(tmp_path,env):
    env['CUDA_VISIBLE_DEVICES']='';reader,writer=os.pipe()
    os.write(writer,b'lease');os.close(writer)
    try:
        code=f'import os;assert os.read({reader},5)==b"lease"'
        done=subprocess.run(argv(tmp_path/'util.jsonl',code),env=env,pass_fds=(reader,),capture_output=True,text=True,timeout=10)
        assert done.returncode==0,done.stderr
    finally:os.close(reader)


def test_phase_helper_is_optional_and_best_effort(tmp_path,monkeypatch):
    monkeypatch.delenv('UTIL_PHASE_FILE',raising=False);util_phase.mark('unused')
    path=tmp_path/'phases.jsonl';monkeypatch.setenv('UTIL_PHASE_FILE',str(path));util_phase.mark('train')
    assert json.loads(path.read_text())['phase']=='train'
    monkeypatch.setenv('UTIL_PHASE_FILE',str(tmp_path/'missing/phases'));util_phase.mark('ignored')


@pytest.mark.parametrize('failure',['summary-exists','output-exists','parent-is-file'])
def test_watcher_output_errors_run_child_once_unwatched(tmp_path,env,failure):
    out=tmp_path/'util.jsonl';marker=tmp_path/'child-starts'
    if failure=='summary-exists':out.with_suffix('.summary.json').write_text('original')
    elif failure=='output-exists':out.write_text('original')
    else:
        (tmp_path/'blocked').write_text('file');out=tmp_path/'blocked/util.jsonl'
    code='import os;assert os.environ["UTIL_WATCH_ACTIVE"]=="1";open('+repr(str(marker))+',"a").write("start\\n");raise SystemExit(37)'
    done=subprocess.run(argv(out,code),env=env,capture_output=True,text=True,timeout=10)
    assert done.returncode==37 and marker.read_text()=='start\n'
    assert 'running child unwatched' in done.stderr
    if failure=='summary-exists':assert out.with_suffix('.summary.json').read_text()=='original'


def test_cgroup_scoped_nvml_physical_index_with_cuda_zero(tmp_path,env):
    env.update(CUDA_VISIBLE_DEVICES='0',SLURM_JOB_GPUS='5',SLURM_GPUS_ON_NODE='1',FAKE_GPU_ROWS='5, GPU-allocated, 82, 123')
    done,samples,summary=run(tmp_path,env)
    assert done.returncode==0 and samples and summary['telemetry_complete']
    assert summary['gpu_uuids']==['GPU-allocated'] and 'index mismatch' in done.stderr


def test_combined_and_kcr_wrappers_share_one_watcher_and_retain_attempts(tmp_path,env):
    import shutil
    kit=tmp_path/'kit';(kit/'jobs/message3').mkdir(parents=True)
    shutil.copy(ROOT/'kit/util_watch.py',kit/'util_watch.py')
    shutil.copy(ROOT/'kit/jobs/message3/run-task.sh',kit/'jobs/message3/run-task.sh')
    (kit/'m3_manifest.py').write_text('import os;assert os.environ["UTIL_WATCH_ACTIVE"]=="1";raise SystemExit(75)\n')
    (kit/'combined_send.py').write_text('import os,subprocess;raise SystemExit(subprocess.call(["bash",os.environ["KIT"]+"/jobs/message3/run-task.sh","M3-example"]))\n')
    manifest=tmp_path/'combined.json';roots={k:str(tmp_path/k) for k in ('core_local_root','kcr_root','pd_root','sp_root','control_root')}
    manifest.write_text(json.dumps(dict(settings=roots,tasks=[dict(id='M3-example',group='kcr')])))
    env.update(KIT=str(kit),JOB_PY=sys.executable,COMBINED_MANIFEST=str(manifest),M3_OUT=roots['kcr_root'],M3_MANIFEST='unused',CUDA_VISIBLE_DEVICES='')
    for i in range(2):
        done=subprocess.run(['bash',str(ROOT/'kit/jobs/combined/run-task.sh'),'M3-example'],env=env,timeout=10)
        assert done.returncode==75
    summaries=list((Path(roots['kcr_root'])/'state/utilisation').glob('*.summary.json'))
    assert len(summaries)==2
    assert all(json.loads(p.read_text())['job']=='M3-example' for p in summaries)


def test_stuck_sampler_is_daemon_and_never_blocks_child(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    import threading
    entered=threading.Event();release=threading.Event()
    def stuck(timeout):
        entered.set();release.wait(10)
        return []
    monkeypatch.setattr(watch,'query',stuck)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    args=NS(out=tmp_path/'stuck.jsonl',window=1,interval=.01,warn=70,expect_gpus=1,
        command=[sys.executable,'-I','-c','import time;time.sleep(.1);raise SystemExit(37)'])
    start=time.monotonic()
    try:
        assert watch.run(args)==37 and entered.is_set()
        assert time.monotonic()-start<1
        assert any(t.name=='util-sampler' and t.daemon for t in threading.enumerate())
        summary=json.loads(args.out.with_suffix('.summary.json').read_text())
        assert not summary['telemetry_complete'] and summary['mean_util'] is None
    finally:release.set()
