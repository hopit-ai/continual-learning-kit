"""CPU admission must retain measured requirements instead of clipping away failures."""
import builtins
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_kit_phase0_storage import CONFIG,GIB

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('slowest,expected',[(80,300),(80.001,300.003),(400,1260)])
def test_requirement_is_untruncated(slowest,expected):
    """Clipping the requirement would admit imports that cannot fit the CUDA allowance."""
    from kit.v4_phase0_environment import environment_timing
    assert environment_timing({'trainer':1,'inference':slowest})['deadline_seconds']==pytest.approx(expected)


@pytest.mark.parametrize('role',['trainer','inference'])
@pytest.mark.parametrize('slowest',[80,80.001,400])
def test_real_planner_rederives_requirement_before_submission(tmp_path,monkeypatch,role,slowest):
    """The real prepare verifier must reject even an old clipped receipt before creating a plan."""
    from kit import v4_phase0 as p,v4_phase0_site as site
    from kit.v4_allocation import plan
    work=tmp_path/'work';model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps(CONFIG))
    imports={'trainer':1,'inference':1};imports[role]=slowest
    receipt={'phase':'phase0','before_allocation':True,'verification_seconds':.5,
             'environment_check':{'import_seconds':imports,'deadline_seconds':300}}
    for name,doc in [('v4/report-inputs/prepare-receipt-phase0.json',receipt),
                     ('v4/report-phase0/presend/containment-presend.json',{})]:
        path=work/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc))
    launch={**site.storage_environment(work,task_root=tmp_path,allocation=True),'MODEL_DIR':str(model),'TEACHER_MODEL_DIR':str(model)}
    monkeypatch.setattr(p,'verify_presend',lambda work:{})
    monkeypatch.setattr(p,'verify_files',lambda doc:None)
    monkeypatch.setattr(p,'frozen_launch_environment',lambda work:launch)
    monkeypatch.setattr(site.shutil,'disk_usage',lambda path:SimpleNamespace(free=1000*GIB))
    if slowest==80:
        doc=plan(work,'phase0','phase0',110,10080)
        assert doc['environment_check_deadline_seconds']==300
        assert (doc['sbatch_time'],doc['phase0_allocation_cap_gpu_hours'],doc['block_limit'],doc['ceiling'])==('01:50:00',22,100,560)
    else:
        with pytest.raises(ValueError,match='requirement.*exceeds.*300.*before submission'):
            plan(work,'phase0','phase0',110,10080)
        assert not (work/'v4/report-phase0/allocation').exists()


@pytest.mark.parametrize('role',['trainer','inference'])
@pytest.mark.parametrize('slowest',[80,80.001,400])
def test_prepare_refuses_before_dataset_work(tmp_path,monkeypatch,role,slowest):
    """CPU prepare must stop on oversized measured imports before freezing inputs for a GPU allocation."""
    from kit import v4_phase0 as p,v4_prepare
    imports={'trainer':1,'inference':1};imports[role]=slowest
    smoke=tmp_path/'smoke.json'
    smoke.write_text(json.dumps({'observations':[{'name':'phase0-trainer-imports','seconds':imports['trainer'],'returncode':0}]}))
    monkeypatch.setattr(p,'verify_presend',lambda work:{})
    monkeypatch.setattr(p.site,'validate_base_versions',lambda versions:None)
    monkeypatch.setattr(p.site,'installed_versions',lambda:{})
    monkeypatch.setattr(v4_prepare,'import_smoke',lambda work,phase:smoke)
    monkeypatch.setattr(p,'inference_check',lambda work:{'import_seconds':imports['inference']})
    original=builtins.__import__
    class DatasetReached(Exception):pass
    def stop_at_dataset(name,*args,**kwargs):
        if name=='pyarrow':raise DatasetReached()
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',stop_at_dataset)
    work=tmp_path/'work'
    if slowest==80:
        with pytest.raises(DatasetReached):p._prepare_owned(work,True)
    else:
        with pytest.raises(ValueError,match='requirement.*exceeds.*300.*before submission'):p._prepare_owned(work,True)
        assert not (work/'v4/report-inputs/prepare-receipt-phase0.json').exists()


@pytest.mark.parametrize('slowest',[80.001,400])
def test_runtime_refuses_unclipped_oversize_receipt(tmp_path,slowest):
    """A forged oversized receipt must never extend the runtime's registered 300-second cap."""
    from kit.v4_phase0_environment import prepared_deadline
    path=tmp_path/'v4/report-inputs/prepare-receipt-phase0.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'environment_check':{'import_seconds':{'trainer':1,'inference':slowest},'deadline_seconds':max(120,3*slowest)+60}}))
    with pytest.raises(ValueError,match='requirement.*exceeds.*300'):prepared_deadline(tmp_path)


def test_runbook_states_unclipped_admission_and_quota():
    """The partner must know CPU admission rejects oversized imports and filesystem free space is not quota."""
    text=(ROOT/'kit/README-phase0.md').read_text()
    assert 'min(300 seconds' not in text
    assert '80 seconds' in text and 'prepare and planning refuse on CPU' in text
    assert 'Confirm your quota' in text and '869 GiB' in text
    assert 'quota -s' in text and 'lfs quota -u $USER' in text and 'mmlsquota' in text
    assert 'filesystem free space, not your quota' in text
