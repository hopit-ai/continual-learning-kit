"""Offline host-venv review regressions; each guards a pre-admission failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]


def distribution(name,version,location,**kw):return dict(name=name,version=version,location=str(location),**kw)


@pytest.mark.parametrize('name',['torch','numpy','datasets','peft','Some_Pkg'])
def test_every_own_distribution_shadow_is_refused(tmp_path,name):
    """Checking only seven pins misses newly installed distributions shadowing the native base."""
    from kit import v4_phase0_site as s
    base=[distribution(name.replace('_','-'),'1',tmp_path/'base')]
    own=[distribution(name,'2',tmp_path/'trainer')]
    with pytest.raises(ValueError,match='shadow'):s.validate_shadowing(base,own,tmp_path/'SDPO')


@pytest.mark.parametrize('allowed',[True,False])
def test_verl_exception_requires_the_owned_editable_checkout(tmp_path,allowed):
    """The sole shadow exception is verl installed from the prepared owned SDPO tree."""
    from kit import v4_phase0_site as s
    owned=tmp_path/'SDPO'
    base=[distribution('verl','0.7.0.dev1',tmp_path/'base')]
    own=[distribution('verl','0.7.0.dev0',tmp_path/'trainer',editable_source=str(owned if allowed else tmp_path/'other'))]
    if allowed:s.validate_shadowing(base,own,owned)
    else:
        with pytest.raises(ValueError,match='owned SDPO'):s.validate_shadowing(base,own,owned)


def test_real_metadata_inventory_has_versions_locations_and_editable_source(tmp_path):
    """Distribution admission must inspect real dist-info, including editable provenance."""
    from kit import v4_phase0_site as s
    site=tmp_path/'site';site.mkdir()
    info=site/'example_pkg-1.2.dist-info';info.mkdir()
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: Example_Pkg\nVersion: 1.2\n')
    (info/'direct_url.json').write_text(json.dumps({'url':(tmp_path/'source').as_uri(),'dir_info':{'editable':True}}))
    rows=s.distribution_inventory([site])
    assert rows==[distribution('example-pkg','1.2',site,editable_source=str(tmp_path/'source'))]


@pytest.mark.parametrize('pip_check_failure',[False,True])
def test_build_checks_shadowing_and_pip_check_before_inference(tmp_path,monkeypatch,pip_check_failure):
    """The actual builder must audit the trainer and stop on dependency conflicts before inference installs."""
    from kit import v4_phase0 as p,v4_phase0_site as s
    from test_kit_phase0_host import VERSIONS
    monkeypatch.delenv('SLURM_JOB_ID',raising=False);monkeypatch.setenv('SDPO_DIR',str(tmp_path/'SDPO'))
    monkeypatch.setattr(s,'installed_versions',lambda:VERSIONS.copy())
    base=[distribution(k.replace('_','-'),v,tmp_path/'base') for k,v in VERSIONS.items()]+[distribution('peft','0.18.0',tmp_path/'base')]
    # Do not replace base_environment: exercise full constraint generation.
    monkeypatch.setattr(s,'distribution_inventory',lambda paths=None,**kw:base)
    monkeypatch.setattr(s,'check_network',lambda urls:None)
    commands=[]
    def command(argv,**kw):
        commands.append(argv);kw['log'].write_text(('new test dependency conflict\n' if pip_check_failure else 'No broken requirements found.\n') if argv[1:]==['-m','pip','check'] else 'owned offline command stand-in\n')
        if 'audit-trainer' in argv:
            s.write_trainer_inventory(tmp_path/'work',base,[],str(tmp_path/'work/phase0-envs/trainer'))
        failed=pip_check_failure and argv[1:]==['-m','pip','check']
        return {'returncode':1 if failed else 0,'failure_type':'cpu_exit' if failed else None,'seconds':1}
    monkeypatch.setattr(p,'bounded_command',command)
    if pip_check_failure:
        with pytest.raises(ValueError,match='pip check'):p.environment_build(tmp_path/'work')
        assert not any('vllm==0.18.0' in row for row in commands)
    else:
        p.environment_build(tmp_path/'work')
        audit=next(i for i,row in enumerate(commands) if 'audit-trainer' in row)
        check=next(i for i,row in enumerate(commands) if row[1:]==['-m','pip','check'])
        infer=next(i for i,row in enumerate(commands) if 'vllm==0.18.0' in row)
        assert audit<check<infer
        assert '--without-pip' in commands[2], 'venv bootstrap pip must not shadow the base installer'
        constraints=(tmp_path/'work/v4/report-phase0/environment-build/base-constraints.txt').read_text()
        assert 'peft==0.18.0' in constraints and 'verl==' not in constraints
        receipt=json.loads((tmp_path/'work/v4/report-phase0/environment-build/trainer-inventory.json').read_text())
        assert receipt['visible']==base


@pytest.mark.parametrize('nested',[False,True])
def test_micromamba_base_does_not_need_a_parent_venv_bridge(tmp_path,monkeypatch,nested):
    """Global/micromamba sites are inherited already; reprocessing their .pth files is unnecessary."""
    from kit import v4_phase0_site as s
    monkeypatch.setattr(s.sys,'prefix',str(tmp_path/'base'))
    monkeypatch.setattr(s.sys,'base_prefix',str(tmp_path/'global' if nested else tmp_path/'base'))
    doc=s.bind_trainer_base(tmp_path/'work',tmp_path/'trainer',{'site_packages':[str(tmp_path/'site')]})
    assert doc['enabled']==nested
    assert bool(list((tmp_path/'trainer').rglob('phase0-host-base.pth')))==nested


@pytest.fixture
def gpu_world(tmp_path,monkeypatch):
    from kit import v4_phase0_environment as e,v4_phase0_site as s,v4_teacher as t
    from test_kit_phase0_host import VERSIONS
    work=tmp_path/'work';inputs=work/'v4/report-inputs';inputs.mkdir(parents=True)
    base=work/'v4/report-phase0';base.mkdir()
    for role in ('trainer','inference'):
        python=work/'phase0-envs'/role/'bin/python';python.parent.mkdir(parents=True);python.write_text('stand-in')
    paths={key:str(work/'phase0-envs/inference/bin/python') for key in ('MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER','SDPO_DIR','FINQA_ROOT','V4_TEACHER_PYTHON')}
    paths['TMPDIR']='/tmp';monkeypatch.setenv('TMPDIR','/tmp')
    (inputs/'launch-inputs-phase0.json').write_text(json.dumps({'environment':paths}))
    (inputs/'prepare-receipt-phase0.json').write_text(json.dumps({'environment_check':__import__('kit.v4_phase0_environment',fromlist=['admitted_environment_timing']).admitted_environment_timing({'trainer':1,'inference':1})}))
    (base/'environment.json').write_text(json.dumps({'trainer_dependency_versions':VERSIONS,'runtime_versions':{'torch':'stub'}}))
    monkeypatch.setattr(s,'installed_versions',lambda:VERSIONS.copy());monkeypatch.setattr(t,'runtime_versions',lambda:{'torch':'stub'})
    monkeypatch.setattr(e.shutil,'which',lambda name:'/owned/bin/'+name)
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURM_STEP_ID','batch')
    for name in ('verl','qwen_vl_utils'):monkeypatch.setitem(sys.modules,name,SimpleNamespace())
    monkeypatch.setitem(sys.modules,'vllm.engine.arg_utils',SimpleNamespace(EngineArgs=object))
    calls=[];fault={'role':None,'kind':None}
    def run(argv,**kw):
        kw['log'].write_text('owned trial step\n')
        if argv[0]=='srun':return {'returncode':0,'failure_type':None,'seconds':0}
        role=argv[-1];kind=fault['kind'] if fault['role']==role else None
        monkeypatch.setattr(e.sys,'prefix',str(work/'phase0-envs'/role))
        def ones(n,device):
            calls.append((role,device))
            if kind=='kernel' and device=='cuda:3':raise RuntimeError('CUDA kernel failed on device 3')
            return SimpleNamespace(add_=lambda value:None,item=lambda:2)
        torch=SimpleNamespace(__version__='2.9.0+cu128' if role=='trainer' else '2.10.0+cu129',
            version=SimpleNamespace(cuda='12.8' if role=='trainer' else '12.9'),ones=ones,
            cuda=SimpleNamespace(is_available=lambda:kind!='unavailable',device_count=lambda:7 if kind=='count' else 8,synchronize=lambda device:None),
            _C=SimpleNamespace(_cuda_getDriverVersion=lambda:12090))
        monkeypatch.setitem(sys.modules,'torch',torch)
        monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(__version__='0.12.0' if role=='trainer' else '0.18.0'))
        try:e.probe(work,role)
        except (ValueError,RuntimeError) as exc:
            kw['log'].write_text(str(exc));return {'returncode':1,'failure_type':'cpu_exit','seconds':.1}
        return {'returncode':0,'failure_type':None,'seconds':.1}
    monkeypatch.setattr(e,'bounded_command',run)
    return work,e,calls,fault


@pytest.mark.parametrize('role',['trainer','inference'])
@pytest.mark.parametrize('kind,cause',[('unavailable','CUDA unavailable'),('count','GPU count'),('kernel','device 3')])
def test_both_probes_refuse_bad_cuda_before_selftest(gpu_world,role,kind,cause):
    """Import success must not spend the job on a CUDA runtime that cannot use every allocated device."""
    work,e,calls,fault=gpu_world;fault.update(role=role,kind=kind)
    result=e.check_environment(work)
    assert not result['ok'] and 'STOP before the self-test' in result['message'] and cause in result['message']
    assert not (work/'k8b4/containment').exists()


def test_both_probes_record_driver_runtime_and_eight_real_ops(gpu_world):
    """Both role receipts must show the driver, CUDA runtime and a successful element op on every GPU."""
    work,e,calls,fault=gpu_world
    assert e.check_environment(work)['ok']
    for role,cuda in [('trainer','12.8'),('inference','12.9')]:
        receipt=json.loads((work/f'v4/report-phase0/environment-check-{role}-cuda.json').read_text())
        assert receipt['ok'] and receipt['cuda_runtime']==cuda and receipt['driver']['version']==12090
        assert [row['index'] for row in receipt['devices']]==list(range(8))
        assert [dev for r,dev in calls if r==role]==['cuda:'+str(i) for i in range(8)]


@pytest.mark.parametrize('output',['','/irrelevant/path\nwarning on stderr\n'])
def test_completed_snapshot_download_ignores_trailing_log_noise(tmp_path,monkeypatch,output):
    """Merged stdout/stderr is not a reliable snapshot path; completion must use the pinned cache layout."""
    from kit import v4_phase0_download as d
    repo,revision=d.MODELS['initial'];hf=tmp_path/'hf';expected=d.snapshot_path(hf,repo,revision)
    expected.mkdir(parents=True)
    def command(argv,**kw):kw['log'].write_text(output);return {'returncode':0,'failure_type':None}
    monkeypatch.setattr(d,'bounded_command',command)
    assert d.download_snapshot(repo,revision,hf,60,tmp_path/'download.log')==expected


def test_blocked_prepare_endpoints_have_no_impossible_compute_fallback():
    """Prepare cannot run in Slurm; a blocked wheel endpoint must return its setup blocker."""
    text=(ROOT/'kit/README-phase0.md').read_text()
    assert 'If PyPI or PyTorch wheels are blocked, retain the setup blocker and return it' in text
    assert 'If any is blocked, run that same step using the bounded zero-GPU' not in text


def test_public_models_need_no_interactive_hf_auth():
    """Public snapshot staging must not require an extra HF login or leak a token into logs."""
    text=(ROOT/'kit/README-phase0.md').read_text()
    assert 'interactive HF login' not in text and 'HF_TOKEN' in text and 'never logged' in text


@pytest.fixture
def planner_work(monkeypatch):
    from kit import v4_phase0 as p,v4_phase0_site as s
    with tempfile.TemporaryDirectory(prefix='hr-',dir='/tmp') as folder:
        work=Path(folder).resolve()
        for name in ('v4/report-inputs/prepare-receipt-phase0.json','v4/report-phase0/presend/containment-presend.json'):
            path=work/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
        monkeypatch.setattr(p,'verify_presend',lambda work:{})
        # Isolate filesystem admission; the real relaxed graph is refused in v3 integration.
        monkeypatch.setattr(p,'required_seconds',lambda *args:28800)
        monkeypatch.setattr(p,'verify_prepare',lambda work,**kw:{'verification_seconds':.5,'environment_check':{'deadline_seconds':180}})
        from test_kit_phase0_storage import CONFIG
        model=work/'model';model.mkdir();(model/'config.json').write_text(json.dumps(CONFIG))
        monkeypatch.setattr(p,'frozen_launch_environment',lambda work:{'WORK':str(work),'TMPDIR':str(work/'phase0-cache/tmp'),'MODEL_DIR':str(model),'TEACHER_MODEL_DIR':str(model)})
        yield work,s


def test_planner_refuses_long_vllm_socket_base_before_gpu_submission(planner_work,monkeypatch):
    """An overlong vLLM IPC base must stop on CPU independently of the shared TMPDIR length."""
    from kit.v4_allocation import plan
    from kit import v4_phase0 as p
    work,s=planner_work
    monkeypatch.setattr(p,'frozen_launch_environment',lambda work:{'WORK':str(work),'TMPDIR':str(work/'phase0-cache/tmp'),'VLLM_RPC_BASE_PATH':'/'+('x'*70)})
    with pytest.raises(ValueError,match='vLLM.*107'):plan(work,'phase0','phase0',480,10080)


def test_planner_prices_four_merged_exports_and_caches_on_cpu(planner_work,monkeypatch):
    """Forty free GiB cannot hold retained trainer states, merged exports and caches; refuse before sbatch."""
    from kit.v4_allocation import plan
    work,s=planner_work
    monkeypatch.setattr(s.shutil,'disk_usage',lambda path:SimpleNamespace(free=40*1024**3))
    with pytest.raises(ValueError,match='free space'):plan(work,'phase0','phase0',480,10080)


def test_real_trainer_audit_rejects_own_dist_info_shadow(tmp_path):
    """The actual trainer CLI must scan OWN metadata and retain the complete refused inventory."""
    from kit import v4_phase0_site as s
    work=tmp_path/'work';trainer=work/'phase0-envs/trainer'
    subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(trainer)],check=True,capture_output=True)
    python=trainer/'bin/python'
    own=Path(subprocess.check_output([str(python),'-I','-c','import sysconfig; print(sysconfig.get_path("purelib"))'],text=True).strip())
    info=own/'native_test_pkg-2.dist-info';info.mkdir()
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: Native_Test_Pkg\nVersion: 2\n')
    folder=work/'v4/report-phase0/environment-build';folder.mkdir(parents=True)
    (folder/'base-environment.json').write_text(json.dumps({'inventory':[distribution('native-test-pkg','1',tmp_path/'base')]}))
    done=subprocess.run([str(python),str(ROOT/'kit/v4_phase0_site.py'),'audit-trainer','--work',str(work)],capture_output=True,text=True)
    assert done.returncode==2 and 'shadows native base distribution native-test-pkg' in done.stderr
    receipt=json.loads((folder/'trainer-inventory.json').read_text())
    assert receipt['own']==[distribution('native-test-pkg','2',own)]
    assert distribution('native-test-pkg','2',own) in receipt['visible']


def test_blocked_wheels_leave_returnable_setup_blocker(tmp_path,monkeypatch):
    """A blocked login wheel endpoint must retain its cause instead of suggesting prepare inside Slurm."""
    from kit import v4_phase0 as p,v4_phase0_site as s
    from test_kit_phase0_host import VERSIONS
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(s,'installed_versions',lambda:VERSIONS.copy())
    def blocked(urls):raise ValueError('PyPI: 403 denied; retain the setup blocker and return it')
    monkeypatch.setattr(s,'check_network',blocked)
    with pytest.raises(ValueError,match='403 denied'):p.environment_build(tmp_path/'work')
    assert '403 denied' in (tmp_path/'work/v4/report-phase0/setup-blocker.txt').read_text()


@pytest.fixture(autouse=True)
def native_cpu_runtime(monkeypatch):
    """Old ML stand-ins also provide the independently observed native torch identity."""
    from kit import v4_phase0_site as site
    monkeypatch.setattr(site,'torch_runtime_identity',lambda:{'version':'2.9.0+cu128','cuda':'12.8'})
