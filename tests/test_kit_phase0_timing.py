"""Import timing is recorded; the complete graph has the only performance admission."""
import builtins,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('slowest,raw,deadline',[(80,300,705),(80.001,300.003,706),(400,1260,2305)])
def test_requirement_is_untruncated(slowest,raw,deadline):
    """Preserve the old observed requirement while removing its 300-second refusing expectation."""
    from kit.v4_phase0_environment import environment_timing,admitted_environment_timing
    doc=environment_timing({'trainer':1,'inference':slowest})
    assert doc['requirement_seconds']==pytest.approx(raw)
    assert doc['deadline_seconds']==deadline and admitted_environment_timing(doc['import_seconds'])==doc
    assert doc['timing_expectation']=='record-only'

@pytest.mark.parametrize('role',['trainer','inference'])
@pytest.mark.parametrize('slowest',[80,80.001,400])
def test_prepare_records_imports_without_refusing_dataset_work(tmp_path,monkeypatch,role,slowest):
    """CPU preparation must reach data preparation even with slow but successful imports."""
    from kit import v4_phase0 as p,v4_prepare
    imports={'trainer':1,'inference':1};imports[role]=slowest
    smoke=tmp_path/'smoke.json';smoke.write_text(json.dumps({'observations':[{'name':'phase0-trainer-imports','seconds':imports['trainer'],'returncode':0}]}))
    monkeypatch.setattr(p,'verify_presend',lambda work:{})
    monkeypatch.setattr(p.site,'torch_runtime_identity',lambda:{'version':'2.9.0+cu128','cuda':'12.8'})
    monkeypatch.setattr(p.site,'validate_base_versions',lambda *a:None)
    monkeypatch.setattr(p.site,'installed_versions',lambda:{})
    monkeypatch.setattr(v4_prepare,'import_smoke',lambda work,phase:smoke)
    monkeypatch.setattr(p,'inference_check',lambda work:{'import_seconds':imports['inference']})
    original=builtins.__import__
    class DatasetReached(Exception):pass
    def stop_at_dataset(name,*args,**kwargs):
        if name=='pyarrow':raise DatasetReached()
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',stop_at_dataset)
    with pytest.raises(DatasetReached):p._prepare_owned(tmp_path/'work',True)

@pytest.mark.parametrize('slowest',[80.001,400])
def test_runtime_binds_relaxed_receipt(tmp_path,slowest):
    """Runtime must use the recorded relaxed derivation and reject a forged clipped deadline."""
    from kit.v4_phase0_environment import admitted_environment_timing,prepared_deadline
    path=tmp_path/'v4/report-inputs/prepare-receipt-phase0.json';path.parent.mkdir(parents=True)
    doc=admitted_environment_timing({'trainer':1,'inference':slowest});path.write_text(json.dumps({'environment_check':doc}))
    assert prepared_deadline(tmp_path)==doc['deadline_seconds']
    doc['deadline_seconds']=300;path.write_text(json.dumps({'environment_check':doc}))
    with pytest.raises(ValueError,match='timing.*differs'):prepared_deadline(tmp_path)

def test_runbook_states_approved_reservation_and_actual_accounting():
    """The partner must see the approved reservation, actual billing and unchanged quota distinction."""
    text=(ROOT/'kit/README-phase0.md').read_text()
    assert 'record-only' in text and '27,816 seconds' in text and '984 seconds' in text
    assert '--minutes 480' in text and '--time=08:00:00' in text and '64 GPU-hour allocation cap' in text
    assert 'actual elapsed time' in text and '15 GPU-hours' in text and 'no replacement job' in text
    assert 'Confirm your quota' in text and '869 GiB' in text
    assert 'quota -s' in text and 'lfs quota -u $USER' in text and 'mmlsquota' in text
    assert 'filesystem free space, not your quota' in text
