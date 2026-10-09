"""Long shared-storage paths must not become Unix socket paths or bypass CPU disk admission."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4
import pytest
from test_kit_phase0_storage import CONFIG,GIB

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def shared_work(tmp_path,monkeypatch):
    from kit import v4_phase0 as p,v4_phase0_site as s
    task=tmp_path/'lustre-storage/fsx_efa/user/a_real_partner_username/v4-phase0'
    work=task/'work.ABCDEF';work.mkdir(parents=True)
    model=task/'hf-cache/initial';model.mkdir(parents=True)
    (model/'config.json').write_text(json.dumps(CONFIG))
    monkeypatch.setenv('TASK_ROOT',str(task))
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    for key in ('MODEL_DIR','TEACHER_MODEL_DIR'):monkeypatch.setenv(key,str(model))
    launch={**s.storage_environment(work,task_root=task,allocation=True),'MODEL_DIR':str(model),'TEACHER_MODEL_DIR':str(model),'WORK':str(work)}
    assert len(os.fsencode(work/'phase0-cache/tmp'))>80
    for name in ('v4/report-inputs/prepare-receipt-phase0.json','v4/report-phase0/presend/containment-presend.json'):
        path=work/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    monkeypatch.setattr(p,'verify_presend',lambda work:{})
    # Isolate filesystem admission; the real relaxed graph is refused in v3 integration.
    monkeypatch.setattr(p,'required_seconds',lambda *args:28800)
    monkeypatch.setattr(p,'verify_prepare',lambda work,**kw:{'verification_seconds':.5,'environment_check':{'deadline_seconds':180}})
    monkeypatch.setattr(p,'frozen_launch_environment',lambda work:launch)
    monkeypatch.setattr(s.shutil,'disk_usage',lambda path:SimpleNamespace(free=1000*GIB))
    return task,work,launch,p,s


def test_realistic_lustre_task_root_plans_successfully(shared_work):
    """Shared WORK paths over eighty bytes must pass when the actual socket bases are short."""
    from kit.v4_allocation import plan
    task,work,launch,p,s=shared_work
    doc=plan(work,'phase0','phase0',480,10080)
    assert doc['storage']['tmpdir']==launch['TMPDIR']
    assert doc['storage']['vllm']['socket_suffix_bytes']==37
    assert doc['storage']['ray']['socket_suffix_bytes']==68
    assert (doc['storage']['required_gib'],doc['sbatch_time'],doc['block_limit'],doc['ceiling'])==(869,'08:00:00',100,560)


def test_frozen_socket_paths_fit_while_large_caches_stay_shared(shared_work):
    """Only tiny IPC paths use job-private /tmp; long temporary files and caches stay on WORK."""
    task,work,launch,p,s=shared_work
    assert launch['VLLM_RPC_BASE_PATH']=='/tmp' and 'VLLM_RPC_BASE_PATH' not in s.CACHE_KEYS
    vllm=launch['VLLM_RPC_BASE_PATH']+'/'+str(uuid4())
    ray=launch['RAY_TMPDIR']+'/ray/session_2026-10-08_12-34-56_123456_1234567/sockets/plasma_store'
    assert len(os.fsencode(vllm))<=107 and len(os.fsencode(ray))<=107
    assert len(os.fsencode(launch['RAY_TMPDIR']))+68<=107
    with s.storage(work):
        import tempfile
        assert tempfile.gettempdir()==str(work/'phase0-cache/tmp')
        assert all(Path(os.environ[k]).is_relative_to(work/'phase0-cache') for k in s.CACHE_KEYS)
        assert Path(os.environ['HF_HOME']).is_relative_to(task) and not Path(os.environ['HF_HOME']).is_relative_to(work)


@pytest.mark.parametrize('where',['planner','payload'])
def test_overlong_vllm_socket_base_refuses(shared_work,monkeypatch,capsys,where):
    """A short TMPDIR must never hide a vLLM socket base whose UUID pathname exceeds 107 bytes."""
    from kit.v4_allocation import plan
    task,work,launch,p,s=shared_work
    launch.update(TMPDIR='/tmp/short',VLLM_RPC_BASE_PATH='/'+('x'*70))
    if where=='planner':
        with pytest.raises(ValueError,match='vLLM.*107'):plan(work,'phase0','phase0',480,10080)
    else:
        monkeypatch.setenv('VLLM_RPC_BASE_PATH',launch['VLLM_RPC_BASE_PATH'])
        assert s.main(['socket-check','--work',str(work)])==2
        assert 'vLLM' in capsys.readouterr().err


def test_payload_checks_both_socket_paths_before_environment_check(shared_work):
    """The real generated payload must check both frozen socket bases before CUDA probes and self-test."""
    from kit.v4_allocation import script
    task,work,launch,p,s=shared_work
    text=script({'phase':'phase0','stage':'phase0','launch_environment':launch,'block':'qualification','block_limit':100},work,ROOT/'kit')
    assert text.index(' socket-check ')<text.index('v4_phase0_environment.py')<text.index(' selftest ')
    assert 'export VLLM_RPC_BASE_PATH=/tmp\n' in text
    assert 'export TMPDIR=/tmp/phase0-tmp\n' in text and 'mkdir -p "$TMPDIR" "$RAY_TMPDIR"' in text


@pytest.mark.parametrize('free_gib',[868,869])
def test_prepare_checks_full_storage_before_builds(shared_work,monkeypatch,free_gib):
    """Prepare must retain a readable CPU blocker for insufficient shared space before any environment build."""
    from kit import v4_phase0_download as download
    task,work,launch,p,s=shared_work
    (work/'v4/report-inputs/prepare-receipt-phase0.json').unlink()
    monkeypatch.setattr(p,'validate_prepare_inputs',lambda *a:None)
    monkeypatch.setattr(download,'verify_download_receipt',lambda *a:{})
    monkeypatch.setattr(s.shutil,'disk_usage',lambda path:SimpleNamespace(free=free_gib*GIB))
    calls=[]
    def prepared(*args):calls.append(args);return work/'prepared.json'
    monkeypatch.setattr(p,'_prepare_owned',prepared)
    if free_gib==868:
        with pytest.raises(ValueError,match='869 GiB required'):p.prepare(work)
        assert not calls
        assert '869 GiB required' in (work/'v4/report-phase0/setup-blocker.txt').read_text()
    else:
        assert p.prepare(work)==work/'prepared.json' and len(calls)==1
        disk=json.loads((work/'v4/report-phase0/preparation-storage.json').read_text())
        assert disk['required_gib']==869 and disk['free_bytes']==869*GIB


def test_runbook_requires_large_shared_lustre_storage_up_front():
    """The runnable setup must choose large shared storage instead of the quota-limited home default."""
    text=(ROOT/'kit/README-phase0.md').read_text();intro=text[:text.index('Send this package')]
    assert 'large shared storage' in intro and 'login and compute' in intro and '869 GiB' in intro
    assert '/lustre-storage/fsx_efa/user/$USER/v4-phase0' in text
    assert '$HOME/v4-phase0' not in text and 'prepare refuses' in text
