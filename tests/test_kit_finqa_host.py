"""Main native-host route: CPU capacity, cache/sockets and payload regressions."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_kit_phase0_storage import CONFIG,PARAMETERS,GIB

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def host(tmp_path,monkeypatch):
    task=tmp_path/'lustre-storage/fsx_efa/user/partner_with_long_name/v4-finqa'
    work=task/'work.ABCDEF';work.mkdir(parents=True)
    model=task/'hf-cache/model';model.mkdir(parents=True)
    (model/'config.json').write_text(json.dumps(CONFIG))
    monkeypatch.setenv('TASK_ROOT',str(task))
    launch={'MODEL_DIR':str(model),'TEACHER_MODEL_DIR':str(model),'TMPDIR':'/tmp/finqa-tmp','RAY_TMPDIR':'/tmp/finqa-ray','VLLM_RPC_BASE_PATH':'/tmp'}
    return work,launch


def test_main_disk_bound_includes_all_slots_qualification_and_restoration(host):
    from kit.v4_finqa_site import disk_requirement
    work,launch=host;disk=disk_requirement(launch)
    assert disk['initial_parameters']==PARAMETERS
    assert len(disk['runs'])==43 # 23 main, five smoke, three restoration runs per arm.
    assert sum(r['retained_checkpoints'] for r in disk['runs'])==54
    assert sum(r['exports'] for r in disk['runs'])==61
    assert disk['required_bytes']==154*PARAMETERS+704*GIB
    assert disk['required_gib']==1879
    assert {r['arm'] for r in disk['runs']}=={'S','F','R','D','D-R'}


def test_main_cpu_planning_records_small_disk_advice(host,monkeypatch):
    from kit import v4_finqa_site as s,v4_phase0_site as p
    work,launch=host
    monkeypatch.setattr(p.shutil,'disk_usage',lambda path:SimpleNamespace(free=869*GIB))
    disk=s.planning_storage(work,launch)
    assert disk['required_gib']==1879 and disk['free_bytes']==869*GIB and disk['advisory_only']


def test_main_payload_precedes_containment_with_short_job_tmp(host):
    from kit.v4_allocation import script
    work,launch=host
    text=script(dict(phase='qualification',stage='qualification',block='qualification',block_limit=100,
                     sbatch_time='01:00:00',ceiling=1000,launch_environment=launch),work,ROOT/'kit')
    assert 'v4_finqa_site.py" storage --work "$WORK" --allocation' in text
    assert text.index('mkdir -p "$TMPDIR" "$RAY_TMPDIR"')<text.index('v4_finqa_site.py" check')<text.index(' selftest ')
    assert '--ceiling 1000' in text
    assert '#SBATCH' not in text # Header belongs in the copied pilot wrapper.


def test_main_containment_entry_keeps_short_job_tmp(host,monkeypatch):
    from kit import p4_contain as c,v4_phase0_site as p,v4_finqa_site as s
    work,launch=host
    path=work/'v4/report-inputs/phase0-binding.json';path.parent.mkdir(parents=True);path.write_text('{}')
    with __import__('tempfile').TemporaryDirectory(prefix='fq-',dir='/tmp') as folder:
        monkeypatch.setattr(s,'JOB_TMP',folder)
        monkeypatch.setenv('SLURM_JOB_ID','123')
        def inside(argv):
            assert os.environ['TMPDIR']==folder
            assert __import__('tempfile').gettempdir()==folder
            assert Path(os.environ['PIP_CACHE_DIR']).is_relative_to(work.parent/'cache')
            assert len(os.fsencode(os.environ['RAY_TMPDIR']))+68<=107
            assert os.environ['VLLM_RPC_BASE_PATH']=='/tmp'
            return 0
        monkeypatch.setattr(c,'_main',inside)
        assert c.main(['selftest','--work',str(work)])==0


def test_main_runbook_has_native_wrapper_receipt_and_quota_commands():
    text=(ROOT/'kit/README-finqa.md').read_text()
    for value in ('$HOME/envs/train/bin:$PATH','TASK_ROOT','quota -s','lfs quota','1,879',
                  'v4_finqa_site.py','v4_finqa_submission.py','k8b_pilot_run.sbatch','no second download'):
        assert value in text
    assert 'existing\npartner container wrapper' not in text


def test_native_submission_calls_own_wrapper_once_and_retains_job(host,monkeypatch):
    from kit import v4_finqa_submission as sub,v4_phase0_carry as carry
    import subprocess
    work,launch=host;plan=work/'allocation/q1.json';plan.parent.mkdir()
    binding=dict(schema='v4-phase0-binding.v1',archive_sha256='a'*64,ledger_sha256='b'*64,records_sha256='c'*64)
    doc=dict(phase='qualification',stage='qualification',block='scientific',block_limit=1000,ceiling=1000,gpus=8,
             phase0_import_binding=binding)
    ledger=work/carry.LEDGER_PATH;ledger.parent.mkdir(parents=True);ledger.write_text('{}')
    doc['ledger_sha256']=__import__('hashlib').sha256(ledger.read_bytes()).hexdigest()
    plan.write_text(json.dumps(doc));wrapper=plan.with_suffix('.sbatch');wrapper.write_text('# own native wrapper\n')
    monkeypatch.setattr(carry,'verify_work',lambda work:{'phase0_import':binding,'allocations':{}})
    monkeypatch.delenv('SLURM_JOB_ID',raising=False);monkeypatch.delenv('SLURM_STEP_ID',raising=False)
    calls=[]
    def sbatch(argv,**kwargs):
        calls.append(argv);kwargs['stdout'].write(b'456;partner\n');return SimpleNamespace(returncode=0,stderr=b'')
    monkeypatch.setattr(sub.subprocess,'run',sbatch)
    result=sub.submit(work,plan,wrapper)
    assert result['job_id']=='456' and result['cluster']=='partner'
    assert calls==[['sbatch','--parsable',str(wrapper)]]
    assert (work/sub.BASE/result['plan_sha256']/'receipt.json').exists()
    assert sub.submit(work,plan,wrapper)['job_id']=='456'
    assert len(calls)==1


def test_bound_zero_gpu_receipt_is_reused_without_download(host,monkeypatch,tmp_path):
    from kit import v4_finqa_site as site,v4_phase0_site as p,v4_phase0_carry as carry,v4_phase0_download as download
    import hashlib,io,tarfile
    work,launch=host;hf=tmp_path/'original/hf-cache';paths={}
    for role,(repo,rev) in download.MODELS.items():
        path=download.snapshot_path(hf,repo,rev);path.mkdir(parents=True)
        (path/'config.json').write_text(json.dumps({**CONFIG,'_commit_hash':rev}))
        (path/'model.safetensors').write_bytes(b'CPU stand-in')
        paths[role]=path
    receipt=download.make_receipt(work,hf,paths,{'job_id':'122','gpus':0})
    env=dict(MODEL_DIR=str(paths['initial']),TEACHER_MODEL_DIR=str(paths['teacher']),QWEN3_8B_TOKENIZER=str(paths['initial']),
             SDPO_DIR=str(tmp_path/'sdpo'),FINQA_ROOT=str(tmp_path/'finqa'),V4_TEACHER_PYTHON=str(tmp_path/'infer/bin/python'))
    for key in ('SDPO_DIR','FINQA_ROOT'):Path(env[key]).mkdir()
    Path(env['V4_TEACHER_PYTHON']).parent.mkdir(parents=True);Path(env['V4_TEACHER_PYTHON']).write_text('CPU stand-in')
    trainer=tmp_path/'trainer';(trainer/'bin').mkdir(parents=True);(trainer/'bin/activate').write_text('# stand-in')
    files={'v4/report-inputs/launch-inputs-phase0.json':json.dumps({'environment':env,'trainer_prefix':str(trainer)}).encode(),
           'v4/report-phase0/download/receipt.json':json.dumps(receipt).encode(),
           'v4/report-phase0/environment.json':json.dumps({'inference':{'package_sha256':'d'*64}}).encode()}
    manifest={'schema':'kit-collect.v1','files':[dict(path=n,bytes=len(r),sha256=hashlib.sha256(r).hexdigest()) for n,r in files.items()],'truncated':[],'symlinks':[]}
    files['collect.manifest.json']=json.dumps(manifest).encode()
    archive=work/carry.ARCHIVE_PATH;archive.parent.mkdir(parents=True)
    with tarfile.open(archive,'w:gz') as tar:
        for name,raw in files.items():
            info=tarfile.TarInfo('return/'+name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    monkeypatch.setattr(carry,'verify_work',lambda work:{'phase0_import':{'archive_sha256':'a'*64}})
    monkeypatch.setattr(p.shutil,'disk_usage',lambda path:SimpleNamespace(free=14000*GIB))
    monkeypatch.setattr(download,'download_snapshot',lambda *args:pytest.fail('no second download'))
    monkeypatch.setenv('V4_KIT_TAG','frozen-test-tag')
    doc,disk=site.inherit(work)
    assert doc['environment']['HF_HOME']==str(hf)
    assert doc['environment']['MODEL_DIR']==env['MODEL_DIR']
    assert doc['trainer_activate']==str(trainer/'bin/activate') and disk['required_gib']==1879
    assert site.frozen_environment(work)==doc['environment']
    (paths['initial']/'model.safetensors').write_bytes(b'changed snapshot size')
    with pytest.raises(ValueError,match='receipt.*changed'):site.inherit(work)


def test_pre_payload_main_submission_reaches_real_reconcile_entry(host,monkeypatch):
    from kit import p4_contain as contain
    work,launch=host;folder=work/'k8b4/containment/submissions/plan';folder.mkdir(parents=True)
    ledger=work/'k8b4/containment/allocation-ledger.json';ledger.write_text(json.dumps({'allocations':{}}))
    monkeypatch.setattr(contain,'track_allocation',lambda *args,**kwargs:None)
    calls=[]
    def reconciled(*args,**kwargs):calls.append(args);return {'ok':True}
    monkeypatch.setattr(contain,'reconcile',reconciled)
    assert contain.main(['reconcile','--work',str(work),'--out',str(work/'reconcile'),'--seconds','1'])==0
    assert len(calls)==1


def test_pending_native_submission_prevents_zero_spend_replan(host):
    from kit.v4_finqa_submission import require_closed_submissions,BASE
    work,launch=host;folder=work/BASE/'plan';folder.mkdir(parents=True)
    (folder/'intent.json').write_text('{}')
    require_closed_submissions(work,{'allocations':{}})
    assert list((work/'v4/warnings').glob('*.json'))
    (folder/'receipt.json').write_text(json.dumps({'job_id':'456'}))
    require_closed_submissions(work,{'allocations':{}})
    require_closed_submissions(work,{'allocations':{'456':{'end':'closed'}}})


@pytest.mark.parametrize('free_gib',[1878,1879])
def test_main_initial_free_capacity_boundary(host,monkeypatch,free_gib):
    from kit import v4_allocation as a,v4_phase0_carry as c,v4_finqa_site as s,v4_phase0_site as p
    work,launch=host
    ledger={'allocations':{'phase0-standin':dict(start=1,end=2,width=8,segments=[dict(start=1,end=2,block='qualification')])}}
    path=work/c.LEDGER_PATH;path.parent.mkdir(parents=True);path.write_text(json.dumps(ledger))
    monkeypatch.setattr(c,'verify_work',lambda work:ledger)
    monkeypatch.setattr(s,'frozen_environment',lambda work:launch)
    monkeypatch.setattr(p.shutil,'disk_usage',lambda path:SimpleNamespace(free=free_gib*GIB))
    doc=a.plan(work,'qualification','qualification',720,10080)
    assert doc['storage']['required_gib']==doc['storage']['additional_required_gib']==1879
    assert doc['storage']['advisory_only']
    assert (doc['block_limit'],doc['ceiling'])==(1000,1000)
