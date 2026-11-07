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
    assert summary['startup_mean_util']==pytest.approx(80)
    assert summary['phases'][0]['phase']=='score'
    assert summary['startup_grace_seconds']==summary['window_s']==10
    assert summary['holder_slices']==[] and summary['holder_slice_telemetry_complete'] is False


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
    assert stats.startup_total/10==50 and stats.startup_below==5


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
    env.update(CUDA_VISIBLE_DEVICES='0',SLURM_JOB_GPUS='5',SLURM_GPUS_ON_NODE='2')
    done,samples,summary=run(tmp_path,env)
    assert done.returncode==0 and not samples and summary['telemetry_complete'] is False
    assert summary['mean_util'] is None and summary['gpus']==1


def test_remapped_cuda_and_slurm_agree_on_count(tmp_path,env):
    env.update(CUDA_VISIBLE_DEVICES='0',SLURM_JOB_GPUS='1',SLURM_GPUS_ON_NODE='1')
    done,samples,summary=run(tmp_path,env)
    assert done.returncode==0 and {s['gpu'] for s in samples}=={'GPU-a'}
    assert summary['telemetry_complete'] and summary['gpus']==1


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
    manifest.write_text(json.dumps(dict(settings=roots,tasks=[dict(id='M3-example',group='kcr',gpus=0)])))
    env.update(KIT=str(kit),JOB_PY=sys.executable,COMBINED_MANIFEST=str(manifest),M3_OUT=roots['kcr_root'],M3_MANIFEST='unused',CUDA_VISIBLE_DEVICES='')
    for i in range(2):
        done=subprocess.run(['bash',str(ROOT/'kit/jobs/combined/run-task.sh'),'M3-example'],env=env,timeout=10)
        assert done.returncode==75
    # Fix B: every group's watcher output lives under the shared control root.
    summaries=list((Path(roots['control_root'])/'utilisation/kcr').glob('*.summary.json'))
    assert len(summaries)==2
    assert all(json.loads(p.read_text())['job']=='M3-example' for p in summaries)


def test_manifest_zero_gpu_overrides_absent_or_host_visibility_without_nvml(tmp_path,env):
    env.pop('CUDA_VISIBLE_DEVICES');env['SLURM_GPUS_ON_NODE']='8'
    out=tmp_path/'zero.jsonl';command=argv(out,'raise SystemExit(75)');command[3:3]=['--expect-gpus','0']
    done=subprocess.run(command,env=env,timeout=10)
    summary=json.loads(out.with_suffix('.summary.json').read_text())
    assert done.returncode==75 and summary['gpus']==0 and summary['telemetry_complete']
    assert not Path(env['FAKE_CALLED']).exists()


@pytest.mark.parametrize('cuda',[None,'0','GPU-a'])
def test_cgroup_visible_nvml_list_uses_slurm_counts_not_physical_ids(tmp_path,env,cuda):
    env.update(FAKE_GPU_ROWS='5, GPU-a, 90, 120',SLURM_JOB_GPUS='5',SLURM_STEP_GPUS='5',SLURM_GPUS_ON_NODE='1')
    if cuda is None:env.pop('CUDA_VISIBLE_DEVICES')
    else:env['CUDA_VISIBLE_DEVICES']=cuda
    done,samples,summary=run(tmp_path,env)
    assert done.returncode==0 and {r['gpu'] for r in samples}=={'GPU-a'}
    assert summary['telemetry_complete']


def test_partial_startup_is_separate_from_complete_rolling_windows():
    stats=watch.Rolling(300)
    stats.add(30,0);assert stats.minimum is None and stats.below==0
    stats.add(300,100);assert stats.minimum==90 and stats.below==0
    stats.add(330,100);assert stats.minimum==90 and stats.below==0
    assert stats.startup_total/300==90 and stats.startup_below==30


@pytest.mark.parametrize('group',['core','kcr','pd','sp','preflight','reference','report'])
def test_every_manifest_group_runs_through_combined_wrapper_with_exact_gpu_count(tmp_path,env,group):
    import shutil
    kit=tmp_path/'kit';kit.mkdir();shutil.copy(ROOT/'kit/util_watch.py',kit/'util_watch.py')
    (kit/'combined_send.py').write_text('import os;assert os.environ["UTIL_WATCH_ACTIVE"]=="1";raise SystemExit(75)')
    roots={key:str(tmp_path/key) for key in ('core_local_root','kcr_root','pd_root','sp_root','control_root')}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(settings=roots,tasks=[dict(id='attempt',group=group,gpus=0)])))
    env.update(KIT=str(kit),COMBINED_MANIFEST=str(manifest),JOB_PY=sys.executable)
    done=subprocess.run(['bash',str(ROOT/'kit/jobs/combined/run-task.sh'),'attempt'],env=env,timeout=10)
    assert done.returncode==75 and not Path(env['FAKE_CALLED']).exists()
    summaries=list(tmp_path.rglob('*.summary.json'));assert len(summaries)==1
    assert json.loads(summaries[0].read_text())['gpus']==0


def test_standalone_message3_wrapper_uses_manifest_zero_gpu_count(tmp_path,env):
    import shutil
    kit=tmp_path/'kit';kit.mkdir();shutil.copy(ROOT/'kit/util_watch.py',kit/'util_watch.py')
    (kit/'m3_manifest.py').write_text('raise SystemExit(75)')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(tasks=[dict(id='cpu',gpus=0)])))
    env.update(KIT=str(kit),M3_MANIFEST=str(manifest),M3_OUT=str(tmp_path/'out'),JOB_PY=sys.executable)
    done=subprocess.run(['bash',str(ROOT/'kit/jobs/message3/run-task.sh'),'cpu'],env=env,timeout=10)
    assert done.returncode==75 and not Path(env['FAKE_CALLED']).exists()
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
        assert time.monotonic()-start<3
        assert any(t.name=='util-sampler' and t.daemon for t in threading.enumerate())
        summary=json.loads(args.out.with_suffix('.summary.json').read_text())
        assert not summary['telemetry_complete'] and summary['mean_util'] is None
    finally:release.set()


def test_slow_query_finishes_after_child_without_losing_telemetry(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    def slow(timeout):
        assert timeout==1
        time.sleep(.8)
        return [dict(uuid='GPU-a',index='0',util=80,mem=1)]
    monkeypatch.setattr(watch,'query',slow)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    args=NS(out=tmp_path/'slow.jsonl',window=300,interval=.01,warn=70,expect_gpus=1,
        command=[sys.executable,'-I','-c','import time;time.sleep(.1);raise SystemExit(37)'])
    assert watch.run(args)==37
    summary=json.loads(args.out.with_suffix('.summary.json').read_text())
    assert summary['telemetry_complete'] and summary['telemetry_errors']==0
    assert summary['mean_util']==80 and summary['gpu_uuids']==['GPU-a']


@pytest.mark.parametrize('selector,index,complete', [('0','0',True),('5','5',True),
    ('GPU-holder','5',True),('0','5',False)])
def test_holder_slice_empty_inventory_queries_cuda_selector(tmp_path, env, selector, index, complete):
    binary=tmp_path/'nvidia-smi'
    binary.write_text('#!'+sys.executable+'\nimport sys,os\nfrom pathlib import Path\n'
        'with Path(os.environ["FAKE_CALLED"]).open("a") as f:f.write(" ".join(sys.argv[1:])+"\\n")\n'
        f'if any(a.startswith("--id=") for a in sys.argv):print("{index}, GPU-holder, 87, 420")\n')
    binary.chmod(0o755)
    env.update(CUDA_VISIBLE_DEVICES=selector,SLURM_GPUS_ON_NODE='8',SLURM_JOB_GPUS='0,1,2,3,4,5,6,7')
    out=tmp_path/'holder.jsonl'
    command=argv(out,'import time;time.sleep(.2);raise SystemExit(37)')
    command[3:3]=['--holder-slice','--expect-gpus','1']
    done=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
    summary=json.loads(out.with_suffix('.summary.json').read_text())
    assert done.returncode==37 and summary['telemetry_complete'] is complete
    if complete:
        assert summary['gpu_uuids']==['GPU-holder'] and summary['mean_util']==pytest.approx(87)
    else:
        assert summary['gpu_uuids'] is None and summary['mean_util'] is None
        assert out.read_text()==''
    assert summary['holder_selector_queries']>=1 and summary['holder_slice']
    assert summary['holder_slices']==(['GPU-holder'] if complete else [])
    assert summary['holder_slice_telemetry_complete'] is complete
    assert '--id='+selector in Path(env['FAKE_CALLED']).read_text()


@pytest.mark.parametrize('selector,rows', [('0,1','0, GPU-a, 85, 123'),('GPU-holder','5, GPU-other, 85, 123'),('0','')])
def test_holder_slice_missing_or_ambiguous_measurements_remain_unknown(tmp_path,env,selector,rows):
    env.update(CUDA_VISIBLE_DEVICES=selector,FAKE_GPU_ROWS=rows)
    out=tmp_path/'holder-invalid.jsonl';command=argv(out,'import time;time.sleep(.1)')
    command[3:3]=['--holder-slice','--expect-gpus','1']
    done=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
    summary=json.loads(out.with_suffix('.summary.json').read_text())
    assert done.returncode==0 and not summary['telemetry_complete'] and summary['mean_util'] is None
    assert out.read_text()==''
    assert summary['holder_slices']==[] and summary['holder_slice_telemetry_complete'] is False


def test_invalid_holder_scope_preserves_single_child_and_failure_exit(tmp_path,env):
    env.update(CUDA_VISIBLE_DEVICES='0,1',SLURM_GPUS_ON_NODE='8')
    marker=tmp_path/'child-starts'
    code='import os;assert os.environ["UTIL_WATCH_ACTIVE"]=="1";open('+repr(str(marker))+',"a").write("start\\n");raise SystemExit(37)'
    out=tmp_path/'invalid-scope.jsonl';command=argv(out,code)
    command[3:3]=['--holder-slice','--expect-gpus','1']
    done=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
    assert done.returncode==37 and marker.read_text()=='start\n'
    assert json.loads(out.with_suffix('.summary.json').read_text())['telemetry_complete'] is False


@pytest.mark.parametrize('selector,indexes,expected', [
    ('2',[2,3,4,5],None),  # holder CUDA 2 can be physical 4, never claim sibling 2
    ('0',[2,3,4,5],None),
    ('2',[0,1,2,3],'GPU-2'),
    ('2',[3,2,1,0],'GPU-2'),
    ('5',[5],'GPU-5'),
    ('0',[5],None),  # even a singleton must match the numeric token exactly
    ('2',[0,2,2],None),
    ('GPU-4',[2,3,4,5],'GPU-4'),
    ('GPU-',[2,3,4,5],None),
    ('MIG-4',[4],None),
])
def test_holder_numeric_namespace_requires_exact_index_and_dense_inventory(selector,indexes,expected):
    rows=[dict(index=str(i),uuid='GPU-'+str(i),util=87,mem=420) for i in indexes]
    if expected is None:
        with pytest.raises(ValueError):watch.allocated(rows,[selector],{},1,holder_slice=True)
    else:
        assert [r['uuid'] for r in watch.allocated(rows,[selector],{},1,holder_slice=True)]==[expected]


@pytest.mark.parametrize('selector,expected',[('2',None),('GPU-4','GPU-4')])
def test_holder_renumbered_inventory_never_records_sibling_gpu(tmp_path,env,selector,expected):
    env.update(CUDA_VISIBLE_DEVICES=selector,SLURM_GPUS_ON_NODE='4',SLURM_JOB_GPUS='2,3,4,5',
        UTIL_HOLDER_SLICE='1',FAKE_GPU_ROWS='\n'.join(f'{i}, GPU-{i}, 87, 420' for i in range(2,6)))
    done,samples,summary=run(tmp_path,env)
    assert done.returncode==0
    if expected is None:
        assert not samples and summary['gpu_uuids'] is None and summary['mean_util'] is None
        assert not summary['telemetry_complete']
    else:
        assert {r['gpu'] for r in samples}=={expected} and summary['telemetry_complete']


@pytest.mark.parametrize('grace',[None,'0','.2','30'])
def test_startup_grace_is_recorded_with_window_default(tmp_path,env,grace):
    out=tmp_path/'grace.jsonl';command=argv(out,'import time;time.sleep(.15);raise SystemExit(37)',window='.1')
    if grace is not None:command[3:3]=['--startup-grace',grace]
    done=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
    summary=json.loads(out.with_suffix('.summary.json').read_text())
    assert done.returncode==37 and summary['returncode']==37
    assert summary['startup_grace_seconds']==(.1 if grace is None else float(grace))
    assert summary['window_s']==.1
    assert summary['holder_slices']==[] and summary['holder_slice_telemetry_complete'] is False


@pytest.mark.parametrize('grace',['-1','nan','inf'])
def test_invalid_startup_grace_rejected_before_launch(tmp_path,env,grace):
    marker=tmp_path/'launched';out=tmp_path/'invalid.jsonl'
    command=argv(out,'from pathlib import Path;Path('+repr(str(marker))+').touch()')
    command[3:3]=['--startup-grace',grace]
    done=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
    assert done.returncode==2 and not marker.exists() and not out.exists()


def test_holder_summary_retains_measured_uuid_when_later_telemetry_fails(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    calls=[]
    def query(timeout):
        calls.append(timeout)
        if len(calls)>1:raise ValueError('telemetry lost')
        return [dict(uuid='GPU-holder',index='0',util=87,mem=420)]
    monkeypatch.setattr(watch,'query',query)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-holder')
    args=NS(out=tmp_path/'partial.jsonl',window=300,interval=.01,warn=70,expect_gpus=1,holder_slice=True,
        command=[sys.executable,'-I','-c','import time;time.sleep(.15);raise SystemExit(37)'])
    assert watch.run(args)==37
    summary=json.loads(args.out.with_suffix('.summary.json').read_text())
    assert summary['holder_slices']==['GPU-holder'] and len(calls)>1
    assert summary['telemetry_complete'] is False and summary['holder_slice_telemetry_complete'] is False
    assert summary['startup_grace_seconds']==300


@pytest.mark.parametrize('allocation,holder',[
    ({},False),({'UTIL_HOLDER_SLICE':'0'},False),({'UTIL_HOLDER_SLICE':'1'},True),
    ({'SLURM_GPUS_ON_NODE':'1'},False),({'SLURM_GPUS_ON_NODE':'8'},True),
    ({'SLURM_JOB_GPUS':'5'},False),({'SLURM_JOB_GPUS':'4-4'},False),
    ({'SLURM_STEP_GPUS':'0-1'},True),({'SLURM_JOB_GPUS':'2,5'},True),
    ({'SLURM_STEP_GPUS':'GPU-a,GPU-b'},True),
])
@pytest.mark.parametrize('wrapper',['combined/run-task.sh','message3/run-task.sh','v6a/run.sh'])
@pytest.mark.parametrize('gpus',[0,1,8])
def test_wrappers_share_holder_scope_rule(tmp_path,env,allocation,holder,wrapper,gpus):
    capture=tmp_path/'capture.json';shim=tmp_path/'python'
    shim.write_text('#!'+sys.executable+'\nimport os,json,sys\n'
        'if sys.argv[1:3]==["-I","-c"]:os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n'
        'open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    shim.chmod(0o755)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(mode='v6a',tasks=[dict(id='fixture',group='kcr',gpus=gpus)],
        settings=dict(control_root=str(tmp_path/'control')))))
    env.update(allocation)
    env.update(KIT=str(ROOT/'kit'),COMBINED_MANIFEST=str(manifest),M3_MANIFEST=str(manifest),
        M3_OUT=str(tmp_path/'m3'),JOB_PY=str(shim),CAPTURE=str(capture))
    done=subprocess.run(['bash',str(ROOT/'kit/jobs'/wrapper),'fixture'],env=env,capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    args=json.loads(capture.read_text())
    scope=args[args.index('--expect-gpus'):args.index('--')]
    assert scope==['--expect-gpus',str(gpus),*(['--holder-slice'] if holder and gpus==1 else [])]


@pytest.mark.parametrize('selector',[None,'0'])
def test_plain_single_gpu_allocation_retains_cgroup_renumbering(tmp_path,env,selector):
    env.update(SLURM_GPUS_ON_NODE='1',SLURM_JOB_GPUS='5',FAKE_GPU_ROWS='5, GPU-physical, 87, 420')
    if selector is None:env.pop('CUDA_VISIBLE_DEVICES',None)
    else:env['CUDA_VISIBLE_DEVICES']=selector
    assert not watch.use_holder_slice(env,1)
    out=tmp_path/'plain.jsonl';command=argv(out,'import time;time.sleep(.2)')
    command[3:3]=['--expect-gpus','1']
    done=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
    summary=json.loads(out.with_suffix('.summary.json').read_text())
    assert done.returncode==0 and summary['telemetry_complete']
    assert summary['gpu_uuids']==['GPU-physical'] and not summary['holder_slice']
