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
_HOST_DISK_USAGE = shutil.disk_usage


def require_host_build_space(path):
    """The real venv-builder/subprocess needs its production 30 GiB host reserve."""
    available = _HOST_DISK_USAGE(path).free
    if available < 30 * 1024**3:
        pytest.skip(f'host resource absent: phase-0 builder needs 30 GiB free; available {available/1024**3:.1f} GiB')


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
        assert all(Path(env[k]).is_relative_to(work/'phase0-cache' if k=='TMPDIR' else task/'cache') for k in CACHE_KEYS)
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
    for field in ('--gpus=0','--no-requeue','--time=20:00:00','--cpus-per-task=8','--mem=32G','--account=a','--qos=q','--partition=p'):assert '#SBATCH '+field in text
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
    assert calls==[('https://github.com',6000)]


def test_real_build_recipe_preserves_base_and_uses_owned_offline_caches(tmp_path,monkeypatch):
    """Actual command construction must constrain trainer pins, isolate inference, and keep HF offline."""
    require_host_build_space(tmp_path)
    from kit import v4_phase0 as p,v4_phase0_site as s
    monkeypatch.delenv('SLURM_JOB_ID',raising=False);monkeypatch.setenv('SDPO_DIR',str(tmp_path/'source'))
    monkeypatch.setenv('TASK_ROOT',str(tmp_path/'task'));monkeypatch.setattr(s,'installed_versions',lambda:VERSIONS.copy())
    monkeypatch.setattr(s,'torch_runtime_identity',lambda:{'version':'2.9.0+cu128','cuda':'12.8'})
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
        assert all(Path(env[k]).is_relative_to(work/'phase0-cache' if k=='TMPDIR' else tmp_path/'task/cache') for k in s.CACHE_KEYS)
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


def test_partner_torch_metadata_without_suffix_records_cpu_runtime(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from kit import v4_phase0_site as site,runner
    # Exactly the micromamba metadata/module identities reported by the partner.
    versions={**VERSIONS,'torch':'2.9.0'}
    monkeypatch.setattr(site,'installed_versions',lambda:versions.copy())
    monkeypatch.setattr(site,'distribution_inventory',lambda **kw:[])
    class NoCUDA:
        def __getattr__(self,key):pytest.fail('CPU identity probe must not call CUDA: '+key)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(__version__='2.9.0+cu128',version=SimpleNamespace(cuda='12.8'),cuda=NoCUDA()))
    def command(argv,**kw):
        kw['log'].write_text('No broken requirements found.\n')
        return {'returncode':0,'failure_type':None,'seconds':.1}
    monkeypatch.setattr(runner,'bounded_command',command)
    doc=site.base_environment(tmp_path/'work')
    assert doc['versions']['torch']=='2.9.0'
    assert doc['torch_runtime']=={'version':'2.9.0+cu128','cuda':'12.8'}
    assert json.loads((tmp_path/'work/v4/report-phase0/environment-build/base-environment.json').read_text())==doc


def test_partner_84_second_rehash_measure_plan_and_real_runner(tmp_path,monkeypatch):
    from kit import v4_phase0 as p,v4_allocation as a,runner,v4_prepare as prep
    work=tmp_path/'work';inputs=work/'v4/report-inputs';inputs.mkdir(parents=True)
    source=work/'model-bytes';source.write_bytes(b'actual prepared bytes')
    receipt=inputs/'prepare-receipt-phase0.json'
    receipt.write_text(json.dumps({'phase':'phase0','before_allocation':True,'environment_check':{
        **__import__('kit.v4_phase0_environment',fromlist=['environment_timing']).environment_timing({'trainer':1,'inference':1})},**prep.file_receipt([source])}))
    presend=work/'v4/report-phase0/presend/containment-presend.json';presend.parent.mkdir(parents=True);presend.write_text('{}')
    monkeypatch.setattr(p,'verify_presend',lambda work:{})
    with monkeypatch.context() as timer:
        ticks=iter((100.,184.));timer.setattr(p.time,'monotonic',lambda:next(ticks))
        assert p.measure_prepare_verification(work)==84
    monkeypatch.setattr(p,'frozen_launch_environment',lambda work:{'WORK':str(work)})
    monkeypatch.setattr(p.site,'planning_storage',lambda *args:{})
    doc=a.plan(work,'phase0','phase0',110,10080)
    assert doc['verification_seconds']==84 and doc['verification_allowance_seconds']==1260
    assert doc['required_seconds']>86400 and doc['reservation_slack_seconds']<0
    assert doc['sbatch_time']=='24:00:00' and doc['reserved_gpu_hours']==192
    plan=work/'allocation.json';plan.write_text(json.dumps(doc,indent=2)+'\n')
    monkeypatch.setenv('V4_ALLOCATION_PLAN',str(plan));monkeypatch.setenv('V4_ALLOCATION_PLAN_SHA256',__import__('hashlib').sha256(plan.read_bytes()).hexdigest())
    monkeypatch.setenv('WORK',str(work));monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    graph=p.campaign();path=work/'campaign.json';path.write_text(json.dumps(graph));campaign=runner.load_campaign(path)
    row=next(row for row in campaign['rows'] if row['id']=='verify-prepare').copy();row.update(needs=[],pilot=False,bars=[])
    observed=[]
    def execute(argv,**kw):
        observed.append((kw['timeout'],kw['env']['V4_COMMAND_TIMEOUT']))
        p.verify_prepare(work);kw['log'].parent.mkdir(parents=True,exist_ok=True);kw['log'].write_text('real immutable verifier passed\n')
        return {'returncode':0,'failure_type':None,'seconds':84}
    monkeypatch.setattr(runner,'bounded_command',execute)
    assert runner.run_row(campaign,row,None)==0 and observed==[(1260,'1260')]
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='prepared input changed'):p.verify_prepare(work)


@pytest.mark.parametrize('metadata,runtime,cuda,accepted',[
    ('2.9.0','2.9.0+cu128','12.8',True),('2.9.0','2.9.0','12.8',True),
    ('2.9.0+cu128','2.9.0+cu128','12.8',True),('2.9.1','2.9.0+cu128','12.8',False),
    ('2.8.0','2.8.0+cu128','12.8',False),('2.9.0','2.9.0','12.7',False),
    ('2.9.0','2.10.0+cu128','12.8',False),('2.9.0rc1','2.9.0+cu128','12.8',False)])
def test_partner_torch_public_and_cpu_runtime_admission(metadata,runtime,cuda,accepted):
    from kit import v4_phase0_site as s
    call=lambda:s.validate_base_versions({**VERSIONS,'torch':metadata},{'version':runtime,'cuda':cuda})
    if accepted:assert call()['torch']==metadata
    else:
        with pytest.raises(ValueError,match='torch'):call()


@pytest.mark.parametrize('package',['vllm','ray','numpy','transformers'])
def test_partner_other_public_pins_stay_exact(package):
    from kit import v4_phase0_site as s
    assert s.validate_base_versions(VERSIONS)==VERSIONS
    with pytest.raises(ValueError,match=package):s.validate_base_versions({**VERSIONS,package:VERSIONS[package]+'+unreviewed'})
    assert s.validate_base_versions({**VERSIONS,'flash_attn':'2.8.3+cu12torch2.9'})


@pytest.mark.parametrize('metadata,runtime,cuda',[
    ('2.9.0+cpu','2.9.0+cu128','12.8'),
    ('2.9.0+cu126','2.9.0+cu128','12.8'),
    ('2.9.0','2.9.0+cu128','12.6'),
    ('2.9.0+cu128','2.9.0+cu128','12.6')])
def test_torch_local_suffix_and_cuda_identity_must_be_consistent(metadata,runtime,cuda):
    from kit.v4_phase0_site import validate_base_versions
    with pytest.raises(ValueError,match='torch'):
        validate_base_versions({**VERSIONS,'torch':metadata},{'version':runtime,'cuda':cuda})


@pytest.mark.parametrize('phase',['qualification','main'])
def test_plan_missing_verification_allowance_refuses_after_hash_rebinding(tmp_path,monkeypatch,phase):
    import hashlib
    from kit import v4_allocation as allocation,v4_phase0_carry as carry,v4_finqa_campaign as campaign
    records={'p0':dict(start=1,end=2,width=8,segments=[dict(block='qualification',start=1,end=2)])}
    binding=dict(schema='v4-phase0-binding.v1',archive_sha256='a'*64,ledger_sha256='b'*64,
                 records_sha256=carry.digest(carry.canonical(records)),allocations=records)
    block='qualification' if phase=='qualification' else 'scientific'
    ledger=dict(phase0_import=binding,allocations={'123':dict(start=100,end=None,width=8,planned_end=3700,
                segments=[dict(block=block,start=100,end=None)])})
    path=tmp_path/carry.LEDGER_PATH;path.parent.mkdir(parents=True);path.write_text(json.dumps(ledger))
    # External archive authentication is orthogonal; real plan, prepare hash,
    # immutable prior accounting, actual window and admission consumer execute.
    monkeypatch.setattr(carry,'verify_work',lambda work:ledger)
    receipt=tmp_path/'v4/report-inputs'/('prepare-receipt-main.json' if phase=='main' else 'prepare-receipt.json')
    receipt.parent.mkdir(parents=True);receipt.write_text(json.dumps(dict(verification_seconds=84,verification_allowance_seconds=1260)))
    stage='qualification' if phase=='qualification' else 'scientific'
    cap=next(r for r in campaign.build(phase)['rows'] if r['id']=='prepare')['allocation_cpu_cap_seconds']
    doc=dict(phase=phase,stage=stage,sbatch_time='01:00:00',prior_allocations=records,
             phase0_import_binding=carry.public_binding(binding),verification_seconds=84,
             verification_allowance_seconds=1260,verification_row_cap_seconds=cap,verification_row_slack_seconds=cap-1260,
             prepare_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),admission={'status':'proceed'})
    for env,key in (('V4_QUALIFICATION_ARCHIVE','qualification_sha256'),('V4_OWNER_PROCEED','owner_sha256')):
        evidence=tmp_path/(key+'.json');evidence.write_text('{}')
        monkeypatch.setenv(env,str(evidence));doc['admission'][key]=hashlib.sha256(evidence.read_bytes()).hexdigest()
    plan=tmp_path/'plan.json'
    def freeze():
        plan.write_text(json.dumps(doc));monkeypatch.setenv('V4_ALLOCATION_PLAN',str(plan))
        monkeypatch.setenv('V4_ALLOCATION_PLAN_SHA256',hashlib.sha256(plan.read_bytes()).hexdigest())
    freeze();monkeypatch.setenv('SLURM_JOB_ID','123')
    first=campaign.build(phase)['v4']['allocation_stages'][stage]['first']
    allocation.verify_plan(tmp_path,phase,first)
    del doc['verification_allowance_seconds'];freeze()
    allocation.verify_plan(tmp_path,phase,first)
    assert list((tmp_path/'v4/warnings').glob('*.json'))


@pytest.mark.parametrize('environment,measured,accepted',[(180,84,True),(234,84,True),(235,84,False),(300,84,False),(180,120,True),(180,120.001,False)])
def test_partner_phase0_allowance_fits_fixed_reservation_only(tmp_path,monkeypatch,environment,measured,accepted):
    from kit import v4_phase0 as p,v4_allocation as a
    work=tmp_path/'work'
    for name in ('v4/report-inputs/prepare-receipt-phase0.json','v4/report-phase0/presend/containment-presend.json'):
        path=work/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    monkeypatch.setattr(p,'verify_presend',lambda work:{})
    monkeypatch.setattr(p,'verify_prepare',lambda work,**kw:{'verification_seconds':measured,'environment_check':{'deadline_seconds':environment}})
    monkeypatch.setattr(p,'frozen_launch_environment',lambda work:{})
    monkeypatch.setattr(p.site,'planning_storage',lambda *a:{})
    doc=a.plan(work,'phase0','phase0',110,10080)
    assert doc['sbatch_time']=='24:00:00' and doc['verification_allowance_seconds']==p.verification_allowance(measured)



@pytest.mark.parametrize('measured',[True,-1,None,float('nan'),float('inf')])
def test_partner_verification_allowance_refuses_invalid_measurements(measured):
    from kit.v4_prepare import verification_allowance
    with pytest.raises(ValueError,match='measurement'):verification_allowance(measured)


@pytest.mark.parametrize('phase,measurement,accepted',[('qualification',84,True),('main',84,True),('qualification',201,True),('main',601,True),('qualification',1201,True),('main',1201,True)])
def test_partner_main_real_cpu_measurement_fits_existing_caps(tmp_path,monkeypatch,phase,measurement,accepted):
    from kit import v4_prepare as prep,v4_phase0_carry as carry,v4_finqa_site as site,v4_allocation as a
    work=tmp_path/'work';folder=work/'v4/report-inputs';folder.mkdir(parents=True)
    source=work/'prepared-model';source.write_bytes(b'CPU model bytes')
    receipt=folder/('prepare-receipt-main.json' if phase=='main' else 'prepare-receipt.json')
    receipt.write_text(json.dumps({'before_allocation':True,**prep.file_receipt([source])}))
    ledger={'allocations':{'prior':dict(start=1,end=2,width=8,segments=[dict(block='qualification',start=1,end=2)])}}
    ledger_path=work/carry.LEDGER_PATH;ledger_path.parent.mkdir(parents=True);ledger_path.write_text(json.dumps(ledger))
    monkeypatch.setattr(carry,'verify_work',lambda work:ledger)
    monkeypatch.setattr(site,'frozen_environment',lambda work:{})
    monkeypatch.setattr(site,'planning_storage',lambda *a:{})
    monkeypatch.setattr(a,'admission',lambda *a:{'status':'proceed'})
    ticks=iter((100.,100.+measurement));monkeypatch.setattr(prep.time,'monotonic',lambda:next(ticks))
    stage='teacher' if phase=='main' else 'qualification'
    if accepted:
        doc=a.plan(work,phase,stage,300,300)
        assert doc['verification_seconds']==measurement and doc['verification_allowance_seconds']==prep.main_verification_allowance(measurement)
        cap=max(9000 if phase=='main' else 3000,prep.main_verification_allowance(measurement))
        assert doc['verification_row_cap_seconds']==cap and doc['verification_row_slack_seconds']==cap-prep.main_verification_allowance(measurement)
        assert doc['requested_minutes']==300 and doc['sbatch_time']=='05:00:00'
        frozen=json.loads(receipt.read_text());assert frozen['verification_seconds']==measurement and frozen['verification_allowance_seconds']==prep.main_verification_allowance(measurement)
    else:
        with pytest.raises(ValueError,match='cannot fit.*fixed prepare row/reservation'):a.plan(work,phase,stage,300,300)
        assert not (work/f'v4/report-{phase}/allocation').exists()

@pytest.fixture(autouse=True)
def native_cpu_runtime(monkeypatch):
    """Old ML stand-ins also provide the independently observed native torch identity."""
    from kit import v4_phase0_site as site
    monkeypatch.setattr(site,'torch_runtime_identity',lambda:{'version':'2.9.0+cu128','cuda':'12.8'})
