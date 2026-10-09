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
