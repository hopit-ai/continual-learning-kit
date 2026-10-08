"""Real delegated preparation and multiprocessing integration for scoped items 6/7."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import pytest
from test_kit_phase0_storage import CONFIG,GIB

ROOT=Path(__file__).resolve().parents[1]

# Only ML imports, model/data contents and remote builds are stand-ins. Both
# prepare stages, subprocess delegation, exclusive writes, freezing and rehash
# are the real shipping code; no private inputs or external packages are needed.
TRAINER_BOOTSTRAP=r'''
import sys,os,json,hashlib,types
from pathlib import Path
from types import SimpleNamespace
from kit import v4_phase0 as p,v4_prepare,v4_contract,v4_pool,v4_phase0_download
from kit.v4_datasets import SDPO_COMMIT
from kit.runner import write_durably
work=Path(os.environ['WORK']);base=work/p.BASE
p.site.JOB_TMP=os.environ['TEST_JOB_TMP']
p.site.RAY_JOB_TMP=os.environ['TEST_JOB_TMP']+'/ray'
p.validate_prepare_inputs=lambda *a:None
p.verify_presend=lambda *a:{}
v4_phase0_download.verify_download_receipt=lambda *a:{}
p.site.shutil.disk_usage=lambda path:SimpleNamespace(free=1000*1024**3)
pins={k:v.rstrip('*')+'1' if v.endswith('*') else v for k,v in p.site.BASE_VERSIONS.items()}
p.site.installed_versions=lambda:pins.copy()
def smoke(work,phase):
    path=Path(work)/'v4/report-inputs/import-smoke-phase0.json'
    write_durably(path,{'observations':[{'name':'phase0-trainer-imports','seconds':1,'returncode':0}]})
    return path
v4_prepare.import_smoke=smoke
p.inference_check=lambda work:{'import_seconds':1,'package_sha256':'a'*64}
pa=types.ModuleType('pyarrow');pa.__path__=[]
pa.Table=SimpleNamespace(from_pylist=lambda rows:rows)
pq=types.ModuleType('pyarrow.parquet');pq.write_table=lambda rows,path:Path(path).write_text(json.dumps(rows))
pa.parquet=pq;sys.modules.update({'pyarrow':pa,'pyarrow.parquet':pq})
for split in ('train','test'):
    v4_pool.FINQA_HASHES[split]=p.digest(Path(os.environ['FINQA_ROOT'])/(split+'.json'))
def rows():
    return [{'prompt':[{'role':'user','content':'CPU integration stand-in'}],
             'extra_info':{'index':'stand-in','problem':'CPU integration stand-in','description':'finqa'},
             'reward_model':{'ground_truth':'2'}}]
p.t.authors_rows=lambda *a,**k:rows()
p.t.finqa.load=lambda *a:rows()
p.t.finqa.rows_for_trainer=lambda *a:rows()
v4_contract.TOKENIZER_FILES={'tokenizer.json':p.digest(Path(os.environ['QWEN3_8B_TOKENIZER'])/'tokenizer.json')}
v4_contract.pinned_tokenizer=lambda:SimpleNamespace(apply_chat_template=lambda *a,**k:[1,2])
def identity(path,model='Qwen/Qwen3.6-27B'):
    return {'model':model,'model_revision':v4_contract.INITIAL_8B_REVISION if model=='Qwen/Qwen3-8B' else 'b'*40,
            'model_config':{'vocab_size':248320},'model_file_hashes':{'model.safetensors':p.digest(Path(path)/'model.safetensors')}}
p.t.local_identity=identity
p.t.runtime_versions=lambda:{'CPU-integration':'stand-in'}
p.subprocess.check_output=lambda *a,**k:SDPO_COMMIT+'\n'
for name,doc in [('environment-build/base-environment.json',{'versions':pins}),
                 ('environment-build/trainer-inventory.json',{'visible':[],'own':[]}),
                 ('environment-build/trainer-pip-check.json',{}),('environment-build/base-inheritance.json',{}),
                 ('download/receipt.json',{'scope':'CPU integration stand-in'})]:
    path=base/name
    if not path.exists():write_durably(path,doc)
for name in ('base-pip-check.log','5.log','base-constraints.txt'):
    path=base/'environment-build'/name
    if not path.exists():path.write_text('CPU integration stand-in\n')
assert Path(sys.argv[1]).resolve()==p.KIT/'v4_phase0.py'
raise SystemExit(p.main(sys.argv[2:]))
'''


@pytest.fixture
def prepared_site(tmp_path,monkeypatch):
    from kit import v4_phase0 as p,v4_phase0_download as download
    task=tmp_path/'lustre-storage/fsx_efa/user/realistic_partner_username/v4-phase0'
    work=task/'work.ABCDEF';work.mkdir(parents=True)
    paths={}
    for key,folder in [('MODEL_DIR','initial'),('TEACHER_MODEL_DIR','teacher'),('QWEN3_8B_TOKENIZER','tokenizer'),('FINQA_ROOT','finqa'),('SDPO_DIR','SDPO')]:
        path=task/folder;path.mkdir();paths[key]=str(path);monkeypatch.setenv(key,str(path))
    for key in ('MODEL_DIR','TEACHER_MODEL_DIR'):
        (Path(paths[key])/'config.json').write_text(json.dumps(CONFIG))
        (Path(paths[key])/'model.safetensors').write_bytes(b'CPU stand-in, never loaded')
    (Path(paths['QWEN3_8B_TOKENIZER'])/'tokenizer.json').write_text('{}')
    for split in ('train','test'):(Path(paths['FINQA_ROOT'])/(split+'.json')).write_text('[]')
    scorer=Path(paths['SDPO_DIR'])/'verl/utils/reward_score/feedback/mcq.py';scorer.parent.mkdir(parents=True);scorer.write_text('# CPU stand-in\n')
    trainer=work/'phase0-envs/trainer/bin/python';trainer.parent.mkdir(parents=True)
    trainer.write_text('#!'+sys.executable+'\n'+TRAINER_BOOTSTRAP);trainer.chmod(0o755)
    for key,value in {'WORK':str(work),'TASK_ROOT':str(task),'V4_TEACHER_PYTHON':sys.executable,'V4_KIT_TAG':'CPU-integration-stand-in'}.items():monkeypatch.setenv(key,value)
    monkeypatch.setenv('PYTHONPATH',str(ROOT));monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(p,'validate_prepare_inputs',lambda *a:None)
    monkeypatch.setattr(download,'verify_download_receipt',lambda *a:{})
    monkeypatch.setattr(p,'verify_presend',lambda *a:{})
    monkeypatch.setattr(p.site.shutil,'disk_usage',lambda path:SimpleNamespace(free=1000*GIB))
    with tempfile.TemporaryDirectory(prefix='p0tmp-',dir='/tmp') as job_tmp:
        monkeypatch.setenv('TEST_JOB_TMP',job_tmp)
        monkeypatch.setattr(p.site,'JOB_TMP',job_tmp,raising=False)
        monkeypatch.setattr(p.site,'RAY_JOB_TMP',job_tmp+'/ray')
        yield work,trainer,paths,p


def test_parent_and_delegated_prepare_preserve_all_receipts(prepared_site,monkeypatch):
    """A real parent-to-trainer subprocess must not exclusively publish the same storage receipt twice."""
    work,trainer,paths,p=prepared_site;original=[]
    def built(work):
        (work/p.BASE/'environment-build').mkdir(parents=True,exist_ok=True)
        original.append((work/p.BASE/'preparation-storage.json').read_bytes())
        return trainer,Path(sys.executable),Path(paths['SDPO_DIR'])
    monkeypatch.setattr(p,'environment_build',built)
    try:receipt=p.prepare(work)
    except ValueError:
        pytest.fail((work/p.BASE/'environment-build/trainer-prepare.log').read_text())
    assert receipt.is_file()
    assert (work/p.BASE/'preparation-storage.json').read_bytes()==original[0]
    delegated=work/p.BASE/'preparation-storage-trainer.json';assert delegated.is_file()
    doc=p.verify_prepare(work) # Real full-file integrity verification, not a stub.
    for file in (work/p.BASE/'preparation-storage.json',delegated):
        assert doc['files'][str(file)]['sha256']==hashlib.sha256(file.read_bytes()).hexdigest()
        assert doc['archived_files'][str(file.relative_to(work))]==hashlib.sha256(file.read_bytes()).hexdigest()
    assert (work/'v4/report-inputs/import-smoke-phase0.json').is_file()
    assert (work/'v4/report-inputs/launch-inputs-phase0.json').is_file()
    archive=work.parent/'return.tar.gz'
    done=subprocess.run([sys.executable,str(ROOT/'kit/collect.py'),'--work',str(work),'--out',str(archive)],env=os.environ.copy(),capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stdout+done.stderr
    from kit.v4_archive import Archive
    with Archive(archive) as returned:
        for file in (work/p.BASE/'preparation-storage.json',delegated):assert returned.files[str(file.relative_to(work))]==file.read_bytes()


@pytest.fixture
def frozen_site(prepared_site):
    work,trainer,paths,p=prepared_site
    done=subprocess.run([str(trainer),str(p.KIT/'v4_phase0.py'),'prepare','--work',str(work),'--verified-environments'],env=os.environ.copy(),capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stdout+done.stderr
    return work,p.frozen_launch_environment(work)


def test_real_multiprocessing_listener_under_frozen_lustre_environment(frozen_site):
    """Actual multiprocessing listener-name construction must fit after real prepare freezes a long WORK."""
    work,launch=frozen_site
    assert len(os.fsencode(work/'phase0-cache/tmp'))>107
    assert launch['TMPDIR']==os.environ['TEST_JOB_TMP']
    code='''import multiprocessing.util as u,multiprocessing.connection as c,os,json
p=c.arbitrary_address('AF_UNIX')
print(json.dumps({'path':p,'bytes':len(os.fsencode(p))}))
u._run_finalizers()
'''
    done=subprocess.run([sys.executable,'-c',code],env={**os.environ,**launch},capture_output=True,text=True,timeout=10)
    assert done.returncode==0,done.stderr
    observed=json.loads(done.stdout)
    assert observed['path'].startswith(launch['TMPDIR']+'/pymp-') and '/listener-' in observed['path']
    assert observed['bytes']<=107
    from kit.v4_phase0_site import CACHE_KEYS
    assert all(Path(launch[key]).is_relative_to(work/'phase0-cache') for key in CACHE_KEYS if key!='TMPDIR')
    assert Path(launch['HF_HOME']).is_relative_to(work.parent) and not Path(launch['HF_HOME']).is_relative_to(work)


def test_real_generated_payload_keeps_short_tmpdir_and_shared_caches(frozen_site,tmp_path):
    """Real bash/cache exports must preserve the frozen short TMPDIR before admission and stop before self-test."""
    from kit.v4_allocation import script
    from kit.v4_phase0_site import CACHE_KEYS
    work,launch=frozen_site;bindir=tmp_path/'bin';bindir.mkdir()
    observed=work/'payload-environment.json'
    python=bindir/'python'
    python.write_text('#!'+sys.executable+'\n'+
        'import os,sys,subprocess,json,tempfile\nfrom pathlib import Path\n'+
        'if sys.argv[1].endswith("v4_phase0_site.py"):raise SystemExit(subprocess.run([sys.executable,*sys.argv[1:]]).returncode)\n'+
        'assert sys.argv[1].endswith("v4_phase0_environment.py"), "self-test must not start"\n'+
        'Path('+repr(str(observed))+').write_text(json.dumps({"env":dict(os.environ),"tempdir":tempfile.gettempdir()}))\nraise SystemExit(2)\n')
    python.chmod(0o755)
    payload=tmp_path/'job.sh'
    payload.write_text(script({'phase':'phase0','stage':'phase0','launch_environment':launch,'block':'qualification','block_limit':100},work,ROOT/'kit'))
    done=subprocess.run(['bash',str(payload)],env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],'SLURM_JOB_ID':'123'},capture_output=True,text=True,timeout=10)
    assert done.returncode==2,done.stdout+done.stderr
    doc=json.loads(observed.read_text())
    assert doc['tempdir']==launch['TMPDIR']==os.environ['TEST_JOB_TMP']
    assert Path(launch['RAY_TMPDIR']).is_dir()
    assert all(Path(doc['env'][key]).is_relative_to(work/'phase0-cache') for key in CACHE_KEYS if key!='TMPDIR')
    assert (work/'v4/report-phase0/socket-path-check.json').is_file()


def test_old_long_tmpdir_fails_real_fd_reduction(frozen_site):
    """The negative control must reproduce AF_UNIX path too long through the actual fd-sharing listener."""
    work,launch=frozen_site;old=work/'phase0-cache/tmp'
    code='''import os,tempfile,multiprocessing.reduction as r,multiprocessing.util as u
fd=os.open('/dev/null',os.O_RDONLY)
try:r.DupFd(fd)
finally:os.close(fd);u._run_finalizers()
'''
    done=subprocess.run([sys.executable,'-c',code],env={**os.environ,**launch,'TMPDIR':str(old)},capture_output=True,text=True,timeout=10)
    assert done.returncode!=0 and 'AF_UNIX path too long' in done.stderr,done.stderr


def test_cpu_tensor_transfer_through_real_fd_sharing(frozen_site):
    """A CPU tensor must cross a real multiprocessing fd-sharing socket with the frozen short TMPDIR."""
    work,launch=frozen_site
    assert launch['TMPDIR']==os.environ['TEST_JOB_TMP'] # Fails before even on platforms without Linux tensor sharing.
    if sys.platform!='linux':pytest.skip('Linux file_descriptor tensor sharing unavailable on this host; real listener-path and long-path negative control run separately')
    pytest.importorskip('torch')
    code='''import torch,torch.multiprocessing as mp,multiprocessing.reduction as r,multiprocessing.util as u
mp.set_sharing_strategy('file_descriptor')
x=torch.arange(16)
wire=r.ForkingPickler.dumps(x)
def receive(wire):
    y=r.ForkingPickler.loads(wire)
    assert y.device.type=='cpu' and torch.equal(y,torch.arange(16))
proc=mp.get_context('fork').Process(target=receive,args=(wire,));proc.start();proc.join(15)
if proc.is_alive():proc.terminate();proc.join();raise RuntimeError('tensor transfer timed out')
assert proc.exitcode==0
u._run_finalizers()
'''
    done=subprocess.run([sys.executable,'-c',code],env={**os.environ,**launch},capture_output=True,text=True,timeout=30)
    if done.returncode and 'Operation not permitted' in done.stderr:pytest.skip('sandbox denies real Unix socket/shared-memory creation')
    assert done.returncode==0,done.stdout+done.stderr

@pytest.mark.parametrize('free_gib',[0,1])
def test_environment_admits_job_tmp_space_before_any_step(tmp_path,monkeypatch,free_gib):
    """Insufficient job-private temporary storage must refuse before any Slurm step or self-test starts."""
    from kit import v4_phase0_environment as e,v4_phase0_site as site
    work=tmp_path/'work';inputs=work/'v4/report-inputs';inputs.mkdir(parents=True)
    python=work/'phase0-envs/trainer/bin/python';python.parent.mkdir(parents=True);python.touch()
    launch={'TMPDIR':'/tmp/phase0-tmp','RAY_TMPDIR':'/tmp/phase0-ray','VLLM_RPC_BASE_PATH':'/tmp','V4_TEACHER_PYTHON':str(python)}
    (inputs/'launch-inputs-phase0.json').write_text(json.dumps({'environment':launch}))
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURM_STEP_ID','batch')
    for key,value in launch.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(site.shutil,'disk_usage',lambda path:SimpleNamespace(free=free_gib*GIB))
    monkeypatch.setattr(Path,'is_dir',lambda path:True)
    calls=[]
    def command(argv,**kw):calls.append(argv);return {'returncode':0,'seconds':.01}
    monkeypatch.setattr(e,'bounded_command',command)
    result=e.check_environment(work,seconds=300)
    if free_gib==0:
        assert not result['ok'] and not calls
        assert 'STOP before the self-test' in result['message'] and 'TMPDIR' in result['message'] and '1 GiB' in result['message']
    else:
        assert result['ok'] and len(calls)==3
        assert result['job_tmp']['minimum_gib']==1 and result['job_tmp']['free_bytes']==GIB


def test_every_actual_socket_base_is_checked():
    """The same socket-base checker must refuse multiprocessing, Ray and vLLM overlong bases."""
    from kit.v4_phase0_site import socket_path_check
    valid={'TMPDIR':'/tmp/phase0-tmp','RAY_TMPDIR':'/tmp/phase0-ray','VLLM_RPC_BASE_PATH':'/tmp'}
    for key in valid:
        with pytest.raises(ValueError,match='107'):socket_path_check({**valid,key:'/'+('x'*107)})


def test_login_first_batch_uses_no_multiprocessing_workers(tmp_path,monkeypatch):
    """CPU prepare must read its real first-batch probe without opening sockets under login-node shared TMPDIR."""
    from kit import v4_prepare as p,runner
    monkeypatch.setenv('SDPO_DIR',str(tmp_path/'SDPO'))
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:'c'*40)
    monkeypatch.setattr(p,'smoke_inventory',lambda:{'dependencies':[],'modules':[],'shells':[]})
    def command(argv,**kw):
        kw['log'].write_text('CPU integration stand-in\n')
        return {'returncode':0,'seconds':1}
    monkeypatch.setattr(runner,'bounded_command',command)
    path=p.import_smoke(tmp_path/'work','phase0')
    code=next(row['command'][-1] for row in json.loads(path.read_text())['observations'] if row['name']=='first-sft-cpu-batch')
    assert 'num_workers=0' in code and 'num_workers=1' not in code


def test_containment_entry_keeps_short_job_tmpdir_for_contained_steps(tmp_path,monkeypatch):
    """Astra closed check, item 7: the containment CLI must not restore the long shared TMPDIR inside the allocation."""
    import tempfile as _tempfile
    from kit import p4_contain, v4_phase0_site as site
    task=tmp_path/('lustre-storage-fsx_efa-user-'+'x'*40);work=task/'v4-phase0'/'work.ABCDEF'
    (work/'v4/report-phase0').mkdir(parents=True)
    monkeypatch.setenv('TASK_ROOT',str(task))
    job_tmp=Path(_tempfile.mkdtemp(prefix='j',dir='/tmp'))
    monkeypatch.setattr(site,'JOB_TMP',str(job_tmp))
    seen={}
    def fake_main(argv):
        # What start_step would receive: dict(os.environ), and the listener path multiprocessing would build.
        seen['env']=dict(os.environ)
        seen['listener']=os.path.join(_tempfile.gettempdir(),'pymp-'+'k'*8,'listener-'+'k'*8)
        return 0
    monkeypatch.setattr(p4_contain,'_main',fake_main)
    try:
        monkeypatch.setenv('SLURM_JOB_ID','404801')
        assert p4_contain.main(['run','--work',str(work)])==0
        assert seen['env']['TMPDIR']==str(job_tmp)
        assert len(os.fsencode(seen['listener']))<=107
        assert seen['env']['TRITON_CACHE_DIR'].startswith(str(work))  # bulk caches stay on shared storage
        # The real allocation constant leaves ample room for the listener path.
        assert len(os.fsencode(os.path.join('/tmp/phase0-tmp','pymp-'+'k'*8,'listener-'+'k'*8)))<=107
        monkeypatch.delenv('SLURM_JOB_ID')
        assert p4_contain.main(['reconcile','--work',str(work)])==0
        assert seen['env']['TMPDIR']==str(work/'phase0-cache'/'tmp')  # login node unchanged
    finally:
        import shutil;shutil.rmtree(job_tmp,ignore_errors=True)
