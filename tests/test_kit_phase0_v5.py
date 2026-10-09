"""Partner's phase 0 v4 run, 9 October: a failed training row was recorded as a pass, and the S launcher joined
SDPO_DIR with an absolute scheduled dataset path."""
import json
import subprocess
from pathlib import Path
import pytest
from kit import v4_phase0 as phase0

KIT=Path(phase0.__file__).resolve().parent


@pytest.mark.parametrize('error',[ValueError('run_v4.sh returned non-zero exit status 2'),RuntimeError('x'),KeyError('k'),
                                  subprocess.CalledProcessError(2,['run_v4.sh'])])
def test_failed_operation_fails_its_row(tmp_path,monkeypatch,error):
    def boom(*a,**k):raise error
    monkeypatch.setattr(phase0,'operation',boom)
    code=phase0.main(['train','--work',str(tmp_path),'--row','train-q-S-finqa','--slot','q-S-finqa'])
    assert code!=0
    assert not (tmp_path/'v4/report-status/train-q-S-finqa.json').exists()
    assert json.loads((tmp_path/phase0.BASE/'failures'/'train-q-S-finqa.json').read_text())['operation']=='train'


def test_successful_operation_still_passes(tmp_path,monkeypatch):
    (tmp_path/phase0.BASE).mkdir(parents=True)
    monkeypatch.setattr(phase0,'operation',lambda *a,**k:None)
    assert phase0.main(['train','--work',str(tmp_path),'--row','train-q-S-finqa','--slot','q-S-finqa'])==0
    assert json.loads((tmp_path/'v4/report-status/train-q-S-finqa.json').read_text())=={'ok':1}


def test_s_launcher_checks_the_resolved_data_dir_not_a_joined_path():
    text=(KIT/'run_sdpo_toolalpaca.sh').read_text()
    assert '[[ -f "$SDPO_DIR/$DATASET/$f" ]]' not in text
    assert '[[ -f "$DATA_DIR/$f" ]]' in text


def sft_metrics():
    rows=[]
    for step in (1,2):
        rows.append({'step':step,'data':{'train/loss':1.0/step,'train/lr':1e-5*min((step-1)/10,1),'train/time(s)':3.0,
                                         'v4/completed_optimizer_updates':step}})
    rows.append({'step':2,'data':{'val/loss':0.9}})  # the pinned SFT trainer's final validation record
    return rows


def test_sft_final_validation_record_is_not_a_duplicate_step():
    from kit.v4_qualification import check_metrics
    try:check_metrics(sft_metrics(),'F',2,movement_required=False)
    except ValueError as exc:assert 'duplicate' not in str(exc),exc


def test_two_training_records_for_one_step_still_refuse():
    from kit.v4_qualification import check_metrics
    rows=sft_metrics()[:2]+[sft_metrics()[1]]
    with pytest.raises(ValueError,match='duplicate'):check_metrics(rows,'F',2,movement_required=False)


def test_reader_compares_gpu_uuids_with_and_without_the_gpu_prefix():
    text=(KIT/'v4_phase0.py').read_text()
    assert "removeprefix('gpu-')" in text and "'training memory physical GPU assignment differs'" in text
