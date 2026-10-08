"""Public, offline regressions for the confirmed host-venv site."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
VERSIONS={'torch':'2.9.0+cu128','vllm':'0.12.0','verl':'0.7.0.dev0','transformers':'4.57.1','flash_attn':'2.8.3+cu12torch2.9','ray':'2.53.0','numpy':'1.26.4'}


@pytest.mark.parametrize('package',list(VERSIONS))
def test_host_base_mismatch_names_dependency(package):
    """Each actual trainer dependency must reject a drifted base before any install."""
    from kit.v4_phase0_site import validate_base_versions
    bad={**VERSIONS,package:'wrong'}
    with pytest.raises(ValueError,match=package):validate_base_versions(bad)
    validate_base_versions(VERSIONS)


def test_storage_sets_all_caches_and_never_uses_login_tmp(tmp_path,monkeypatch):
    """Every subprocess and in-process temporary file must use owned storage, not the 1 GB /tmp."""
    from kit.v4_phase0_site import storage,storage_environment,CACHE_KEYS
    work=tmp_path/'work';task=tmp_path/'task'
    monkeypatch.setenv('TASK_ROOT',str(task));monkeypatch.setenv('TMPDIR','/tmp')
    original=os.environ['TMPDIR']
    with storage(work):
        import tempfile
        env=storage_environment(work)
        assert all(Path(env[k]).is_relative_to(work/'phase0-cache') for k in CACHE_KEYS)
        assert Path(env['HF_HOME']).is_relative_to(task) and not Path(env['HF_HOME']).is_relative_to(work)
        with tempfile.TemporaryDirectory() as temp:assert Path(temp).is_relative_to(work/'phase0-cache/tmp')
        assert all(os.environ[k]==env[k] for k in CACHE_KEYS)
    assert os.environ['TMPDIR']==original
    from kit.collect import main
    import tarfile
    buried=work/'phase0-cache/tmp/report-cache';buried.mkdir(parents=True)
    (buried/'should-not-ship.json').write_text('{}')
    (work/'report').mkdir();(work/'report/results.json').write_text('{}')
    archive=tmp_path/'return.tar.gz'
    assert main(['--work',str(work),'--out',str(archive)])==0
    with tarfile.open(archive) as tar:
        assert any(name.endswith('report/results.json') for name in tar.getnames())
        assert not any('phase0-cache' in name for name in tar.getnames())


def test_large_install_refuses_low_space_before_pip(tmp_path,monkeypatch):
    """A large wheel install must refuse before invoking pip if owned storage is too small."""
    from kit import v4_phase0_site as site,v4_phase0 as p
    monkeypatch.setattr(site.shutil,'disk_usage',lambda path:shutil._ntuple_diskusage(1,1,0))
    monkeypatch.setattr(p,'bounded_command',lambda *a,**kw:pytest.fail('must refuse before commands'))
    with pytest.raises(ValueError,match='free space.*30'):p.environment_build(tmp_path/'work')


def models(tmp_path):
    from kit.v4_phase0_download import MODELS
    hf=tmp_path/'task/hf-cache';paths={}
    for role,(repo,rev) in MODELS.items():
        path=hf/'hub'/('models--'+repo.replace('/','--'))/'snapshots'/rev;path.mkdir(parents=True)
        (path/'config.json').write_text(json.dumps({'_commit_hash':rev,'vocab_size':248320 if role=='teacher' else 151936}))
        (path/'model.safetensors').write_bytes(b'owned CPU stand-in weights')
        paths[role]=path
    return hf,paths


@pytest.mark.parametrize('mutation',['none','revision','size','outside','inside_work'])
def test_download_receipt_verifies_pinned_paths_and_sizes(tmp_path,mutation):
    """Prepare must require the compute download receipt and detect changed snapshots."""
    from kit.v4_phase0_download import make_receipt,verify_download_receipt
    hf,paths=models(tmp_path);work=tmp_path/'work';work.mkdir()
    doc=make_receipt(work,hf,paths,{'job_id':'17','gpus':0})
    if mutation=='revision':doc['models']['initial']['revision']='wrong'
    if mutation=='size':(paths['initial']/'model.safetensors').write_bytes(b'changed')
    if mutation=='outside':doc['models']['initial']['path']=str(tmp_path/'elsewhere')
    if mutation=='inside_work':doc['hf_home']=str(work/'hf-cache')
    if mutation=='none':
        assert verify_download_receipt(work,doc,paths)['allocation_gpu_hours']==0
        alias=tmp_path/'task-alias';alias.symlink_to(hf.parent,target_is_directory=True)
        aliases={role:alias/'hf-cache'/path.relative_to(hf) for role,path in paths.items()}
        assert verify_download_receipt(work,doc,aliases)['allocation_gpu_hours']==0
    else:
        with pytest.raises(ValueError,match='download'):verify_download_receipt(work,doc,paths)


def test_zero_gpu_download_job_uses_actual_host_and_bounded_directive(tmp_path):
    """Models must download on a compute-node CPU job, with explicit site, memory and time."""
    from kit.v4_phase0_download import download_script
    env=tmp_path/'download-env'
    subprocess.run([sys.executable,'-m','venv','--without-pip',str(env)],check=True,capture_output=True)
    python=env/'bin/python'
    text=download_script(tmp_path/'work',tmp_path/'task',ROOT/'kit','a','p','q',python)
    import shlex
    actual=shlex.split(text.splitlines()[-1])[0]
    assert actual==str(python.absolute()),'resolving a venv Python symlink discards its installed downloader packages'
    checked=subprocess.run([actual,'-I','-c','import sys; assert sys.prefix=='+repr(str(env))],capture_output=True,text=True)
    assert checked.returncode==0,checked.stderr
    for field in ('--gpus=0','--no-requeue','--time=02:00:00','--cpus-per-task=8','--mem=32G','--account=a','--qos=q','--partition=p'):assert '#SBATCH '+field in text
    assert 'srun ' not in text and 'v4_phase0_download.py' in text
    assert text.index('v4_phase0_site.py')<text.index('v4_phase0_download.py')
    assert subprocess.run(['bash','-n'],input=text,text=True,capture_output=True).returncode==0
    folder=tmp_path/'work/v4/report-phase0/download';folder.mkdir(parents=True)
    (folder/'download.sbatch').write_text(text);(folder/'job.err').write_text('owned scheduler stderr')
    from kit.collect import main
    import tarfile
    archive=tmp_path/'download-return.tar.gz'
    assert main(['--work',str(tmp_path/'work'),'--out',str(archive)])==0
    with tarfile.open(archive) as tar:
        assert any(name.endswith('download/download.sbatch') for name in tar.getnames())
        assert any(name.endswith('download/job.err') for name in tar.getnames())


def test_download_job_executes_both_snapshots_and_retains_zero_charge(tmp_path,monkeypatch):
    """The actual downloader must record both pinned models after scheduler-verified zero-GPU execution."""
    from kit import v4_phase0_download as d
    hf,paths=models(tmp_path);work=tmp_path/'work';work.mkdir()
    monkeypatch.setenv('SLURM_JOB_ID','17');monkeypatch.setenv('TASK_ROOT',str(tmp_path/'task'))
    monkeypatch.setattr(d,'scheduler_job',lambda:{'job_id':'17','gpus':0})
    monkeypatch.setattr(d,'DOWNLOAD_GIB',0)  # owned tiny stand-in; production remains 100 GiB
    calls=[]
    def download(repo,revision,cache,seconds,log):
        calls.append((repo,revision));log.write_text('CPU stand-in snapshot\n');return paths['initial' if repo.endswith('8B') else 'teacher']
    monkeypatch.setattr(d,'download_snapshot',download)
    receipt=d.download(work,hf)
    assert len(calls)==2 and receipt['allocation_gpu_hours']==0 and receipt['slurm']['gpus']==0
    saved=json.loads((work/'v4/report-phase0/download/receipt.json').read_text())
    assert saved==receipt and all(m['bytes']>0 for m in saved['models'].values())


def test_host_rehearsal_has_download_receipt_and_real_report_pause(tmp_path):
    """The confirmed host path must exercise receipt verification and keep all earlier CPU cases."""
    from kit.simulate_v4_phase0 import rehearse,CASES
    assert {'success','missing_paste','failed_selftest','determinism_mismatch','out_of_memory','wall_time','pyxis_route','pre_selftest_refusal','host_venv_success'}<=set(CASES)
    result=rehearse(tmp_path/'host','host_venv_success')
    assert result['runner_exit']==0 and result['pause']['state']=='PAUSE'
    assert result['download']['allocation_gpu_hours']==0
    assert result['reader']['status']=='incomplete' and result['reader']['allocation_gpu_hours']>0


def test_runbook_primary_route_and_compute_only_hf_downloads():
    """The confirmed native env must be runnable without a container or login-node HF calls."""
    text=(ROOT/'kit/README-phase0.md').read_text()
    assert 'your route: host venv (confirmed 8 Oct)' in text
    assert 'export PATH="$HOME/envs/train/bin:$PATH"' in text
    assert 'Do not submit `phase0.sh` directly on the bare host' not in text
    assert 'huggingface-cli" download' not in text
    assert 'v4_phase0_download.py' in text and '--gpus=0' in text
    assert 'prepare and the planner in the same container route interactively' not in text
    assert '1 GB' in text and 'PIP_CACHE_DIR' in text and 'HF_HOME' in text
    from kit.v4_allocation import script
    payload=script({'phase':'phase0','stage':'phase0','block':'qualification','block_limit':100,
        'launch_environment':{'WORK':'/owned/work'}},Path('/owned/work'),ROOT/'kit')
    assert '--min-gib 30' in payload and payload.index('v4_phase0_site.py')<payload.index('v4_phase0_environment.py')
    assert 'CACHE_EXPORTS=' in payload and 'eval "$CACHE_EXPORTS"' in payload
    publisher=(ROOT/'scripts/publish_kit.py')
    if publisher.exists():assert 'confirmed host-venv payload, zero-GPU download job' in publisher.read_text()


def test_prepare_missing_compute_receipt_refuses_before_build(tmp_path,monkeypatch):
    """Even admitted local inputs must not bypass the durable compute download receipt."""
    from kit import v4_phase0 as p
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(p,'validate_prepare_inputs',lambda *a:None)
    monkeypatch.setattr(p,'_prepare_owned',lambda *a:pytest.fail('receipt must precede build'))
    with pytest.raises(ValueError,match='compute-node download receipt'):p.prepare(tmp_path/'work')


@pytest.mark.parametrize('tres,step,accepted',[
    ('cpu=8,mem=32G', 'batch',True),('cpu=8,gres/gpu=0','batch',True),
    ('cpu=8,gres/gpu:h100=1','batch',False),('cpu=8,gres/gpu=8','batch',False),
    ('cpu=8','0',False)])
def test_downloader_checks_scheduler_zero_gpu_and_batch_route(monkeypatch,tres,step,accepted):
    """Staging must reject GPU allocations and enclosing numeric steps, even with a job ID."""
    from kit import v4_phase0_download as d
    monkeypatch.setenv('SLURM_JOB_ID','17');monkeypatch.setenv('SLURM_STEP_ID',step)
    monkeypatch.setattr(d.subprocess,'check_output',lambda *a,**kw:'JobId=17 AllocTRES='+tres+'\n')
    if accepted:assert d.scheduler_job()['gpus']==0
    else:
        with pytest.raises(ValueError,match='download'):d.scheduler_job()


def test_cpu_endpoint_failure_is_bounded_readable_and_precedes_commands(tmp_path,monkeypatch):
    """An inaccessible allowed login endpoint must stop with the compute-job fallback, without HF."""
    import urllib.request
    from kit import v4_phase0_site as s,runner
    calls=[]
    def blocked(request,timeout):
        calls.append((request.full_url,timeout));raise OSError('403 denied')
    monkeypatch.setattr(urllib.request,'urlopen',blocked)
    monkeypatch.setattr(runner,'bounded_command',lambda *a,**kw:pytest.fail('must not dispatch after blocked endpoint'))
    with pytest.raises(ValueError,match='403 denied.*zero-GPU'):
        s.network_command(tmp_path/'work',['git','clone','owned'],urls=['https://github.com'])
    assert calls==[('https://github.com',10)]


def test_real_build_recipe_preserves_base_and_uses_owned_offline_caches(tmp_path,monkeypatch):
    """Actual command construction must constrain trainer pins, isolate inference, and keep HF offline."""
    from kit import v4_phase0 as p,v4_phase0_site as s
    monkeypatch.delenv('SLURM_JOB_ID',raising=False);monkeypatch.setenv('SDPO_DIR',str(tmp_path/'source'))
    monkeypatch.setenv('TASK_ROOT',str(tmp_path/'task'));monkeypatch.setattr(s,'installed_versions',lambda:VERSIONS.copy())
    monkeypatch.setattr(s,'distribution_inventory',lambda paths=None,**kw:[{'name':s.distribution_name(k),'version':v,'location':str(tmp_path/'base-site')} for k,v in VERSIONS.items()])
    endpoints=[];monkeypatch.setattr(s,'check_network',lambda urls:endpoints.extend(urls))
    commands=[]
    def command(argv,**kw):
        commands.append((argv,kw['env']));kw['log'].write_text('No broken requirements found.\n' if argv[1:]==['-m','pip','check'] else 'owned CPU stand-in\n')
        return {'returncode':0,'failure_type':None,'seconds':1}
    monkeypatch.setattr(p,'bounded_command',command)
    work=tmp_path/'work';trainer,inference,owned=p.environment_build(work)
    assert trainer!=inference and len(commands)==8
    assert 'huggingface.co' not in ' '.join(endpoints)
    assert commands[2][0][1:4]==['-m','venv','--system-site-packages']
    for argv,env in commands:
        assert env['HF_HUB_OFFLINE']==env['TRANSFORMERS_OFFLINE']=='1'
        assert all(Path(env[k]).is_relative_to(work/'phase0-cache') for k in s.CACHE_KEYS)
        if 'pip' in argv:assert Path(argv[0]).is_relative_to(work/'phase0-envs') and '--upgrade' not in argv
    assert '-c' in commands[3][0] and '-c' not in commands[7][0]
    base=json.loads((work/'v4/report-phase0/environment-build/base-environment.json').read_text())
    assert base['versions']==VERSIONS and base['route']=='host-venv'
    constraints=(work/'v4/report-phase0/environment-build/base-constraints.txt').read_text()
    assert all(s.distribution_name(name)+'=='+version in constraints for name,version in VERSIONS.items() if name!='verl') and 'verl==' not in constraints


def test_storage_cli_exports_and_refuses_low_space_without_partial_exports(tmp_path):
    """Runbook command substitution must receive valid exports or a clear nonzero refusal."""
    script=ROOT/'kit/v4_phase0_site.py';work=tmp_path/'work'
    env={**os.environ,'TASK_ROOT':str(tmp_path/'task')}
    done=subprocess.run([sys.executable,str(script),'storage','--work',str(work)],env=env,capture_output=True,text=True)
    assert done.returncode==0 and 'export TMPDIR=' in done.stdout
    denied=subprocess.run([sys.executable,str(script),'storage','--work',str(work),'--min-gib','999999999'],env=env,capture_output=True,text=True)
    assert denied.returncode==2 and not denied.stdout and 'STOP: owned storage free space' in denied.stderr


def test_mismatched_base_records_versions_and_a_readable_setup_blocker(tmp_path,monkeypatch):
    """A refused native base must still leave its observed versions and cause in the return archive."""
    from kit import v4_phase0_site as s
    monkeypatch.setattr(s,'installed_versions',lambda:{**VERSIONS,'torch':'wrong'})
    work=tmp_path/'work'
    with pytest.raises(ValueError,match='torch'):s.base_environment(work)
    base=work/'v4/report-phase0'
    assert json.loads((base/'environment-build/base-environment.json').read_text())['versions']['torch']=='wrong'
    assert 'expected 2.9.0+cu128, found wrong' in (base/'setup-blocker.txt').read_text()


def test_real_nested_trainer_inherits_native_base_without_modifying_it(tmp_path):
    """System-site-packages alone misses a parent venv; actual native modules and .pth entries must survive."""
    from kit.v4_phase0_site import bind_trainer_base
    base=tmp_path/'base';trainer=tmp_path/'trainer';work=tmp_path/'work'
    subprocess.run([sys.executable,'-m','venv','--without-pip',str(base)],check=True,capture_output=True)
    python=base/'bin/python'
    site=Path(subprocess.check_output([str(python),'-I','-c','import sysconfig; print(sysconfig.get_path("purelib"))'],text=True).strip())
    original=site/'phase0_native_dependency.py';original.write_text("VALUE='unchanged native base'\n")
    extra=base/'editable';extra.mkdir();(extra/'phase0_native_editable.py').write_text('VALUE=17\n')
    (site/'existing-native.pth').write_text(str(extra)+'\n')
    subprocess.run([str(python),'-m','venv','--without-pip','--system-site-packages',str(trainer)],check=True,capture_output=True)
    probe=[str(trainer/'bin/python'),'-I','-c',"import phase0_native_dependency as n,phase0_native_editable as e; assert n.VALUE=='unchanged native base' and e.VALUE==17"]
    assert subprocess.run(probe,capture_output=True).returncode!=0  # demonstrate the real parent-venv failure
    code='import sys,json; sys.path.insert(0,'+repr(str(ROOT))+'); from kit.v4_phase0_site import bind_trainer_base; print(json.dumps(bind_trainer_base('+repr(str(work))+','+repr(str(trainer))+','+repr({'site_packages':[str(site)]})+'))) '
    bridge=json.loads(subprocess.check_output([str(python),'-I','-c',code],text=True))
    success=subprocess.run(probe,capture_output=True,text=True)
    assert success.returncode==0,success.stderr
    assert original.read_text()=="VALUE='unchanged native base'\n"
    import hashlib
    assert hashlib.sha256(Path(bridge['path']).read_bytes()).hexdigest()==bridge['sha256']
    assert json.loads((work/'v4/report-phase0/environment-build/base-inheritance.json').read_text())==bridge
