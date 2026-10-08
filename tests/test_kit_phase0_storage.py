"""Public CPU admission for retained phase-0 training state and peak merge storage."""
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
GIB=1024**3
# Architecture of the pinned initial Qwen3-8B, not an 8-billion rounded label.
CONFIG={'model_type':'qwen3','hidden_size':4096,'intermediate_size':12288,
        'num_hidden_layers':36,'num_attention_heads':32,'num_key_value_heads':8,
        'head_dim':128,'vocab_size':151936,'tie_word_embeddings':False,'attention_bias':False}
PARAMETERS=8190735360


@pytest.fixture
def planner(monkeypatch):
    from kit import v4_phase0 as p,v4_phase0_site as s
    from kit.v4_allocation import plan
    with tempfile.TemporaryDirectory(prefix='ds-',dir='/tmp') as folder:
        work=Path(folder).resolve();models=work/'models';models.mkdir()
        (models/'config.json').write_text(json.dumps(CONFIG))
        for name in ('v4/report-inputs/prepare-receipt-phase0.json','v4/report-phase0/presend/containment-presend.json'):
            path=work/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
        launch={'WORK':str(work),'TMPDIR':s.JOB_TMP,
                'MODEL_DIR':str(models),'TEACHER_MODEL_DIR':str(models)}
        monkeypatch.setattr(p,'verify_presend',lambda work:{})
        monkeypatch.setattr(p,'verify_prepare',lambda work:{'verification_seconds':.5,'environment_check':{'deadline_seconds':180}})
        monkeypatch.setattr(p,'frozen_launch_environment',lambda work:launch)
        yield work,launch,s,lambda:plan(work,'phase0','phase0',110,10080)


def test_old_96_gib_refuses_through_cpu_planner(planner,monkeypatch):
    """An allocation must never be admitted on the old merged-exports-only disk estimate."""
    work,launch,s,plan=planner
    monkeypatch.setattr(s.shutil,'disk_usage',lambda path:SimpleNamespace(free=96*GIB))
    with pytest.raises(ValueError,match='869 GiB required'):plan()
    assert not (work/'v4/report-phase0/allocation').exists()


@pytest.mark.parametrize('free_gib',[868,869])
def test_computed_capacity_boundary_keeps_gpu_budgets(planner,monkeypatch,free_gib):
    """Admission must use the calculated capacity boundary while preserving the registered GPU caps."""
    work,launch,s,plan=planner
    monkeypatch.setattr(s.shutil,'disk_usage',lambda path:SimpleNamespace(free=free_gib*GIB))
    if free_gib==868:
        with pytest.raises(ValueError,match='869 GiB required'):plan()
    else:
        doc=plan();disk=doc['storage']
        assert disk['required_gib']==869 and disk['free_bytes']==869*GIB
        assert (doc['sbatch_time'],doc['block_limit'],doc['ceiling'],doc['phase0_allocation_cap_gpu_hours'])==('01:50:00',100,560,22)
        assert disk['initial_parameters']==PARAMETERS
        arms=disk['arms']
        assert set(arms)=={'S','F','R','D'}
        for arm,row in arms.items():
            count=2 if arm in 'SD' else 1
            assert row['retained_checkpoints']==count and row['fsdp_ranks']==8
            assert row['model_bytes']==4*PARAMETERS*count
            assert row['adam_moments_bytes']==8*PARAMETERS*count
            assert row['merged_export_bytes']==2*PARAMETERS
            assert row['ema_bytes']==(4*PARAMETERS*count if arm in 'SD' else 0)
        assert disk['peak_merge_bytes']==96*PARAMETERS
        assert disk['required_bytes']==96*PARAMETERS+136*GIB
        assert disk['checkpoint_policy']=='retain model, optimizer, extra and S/D EMA after checks'


def test_snapshot_cache_reserve_grows_with_full_download(planner,monkeypatch):
    """A full model snapshot larger than the nominal cache reserve must increase CPU admission."""
    from kit import v4_phase0_storage as d
    work,launch,s,plan=planner
    monkeypatch.setattr(d,'snapshot_bytes',lambda path:110*GIB)
    monkeypatch.setattr(s.shutil,'disk_usage',lambda path:SimpleNamespace(free=1000*GIB))
    disk=plan()['storage']
    assert disk['model_cache_bytes']==220*GIB and disk['required_gib']==989


@pytest.mark.parametrize('arm',['S','D'])
def test_real_worker_ema_save_is_in_disk_policy(planner,monkeypatch,arm):
    """Both on-policy arms save an EMA through the real wrapper and must have disk reserved for it."""
    from kit import v4_state,v4_restore_check
    work,launch,s,plan=planner;events=[]
    class Manager:
        def __init__(self,**kwargs):assert kwargs['checkpoint_config']['save_contents']==['model']
        def save_checkpoint(self,path,**kwargs):events.append(('EMA',Path(path).name))
    class Worker:
        def __init__(self):
            self.ref_module_fsdp=object();self.actor=SimpleNamespace(teacher_module=self.ref_module_fsdp);self.tokenizer=object()
        def save_checkpoint(self,path,**kwargs):events.append(('model',Path(path).name))
        def load_checkpoint(self,*args,**kwargs):pass
    monkeypatch.setenv('V4_CHECKPOINT_STATE','1')
    monkeypatch.setattr(v4_restore_check,'install_checkpoint_probe',lambda:None)
    monkeypatch.setitem(sys.modules,'verl.utils.checkpoint.fsdp_checkpoint_manager',SimpleNamespace(FSDPCheckpointManager=Manager))
    v4_state.install_worker(Worker)
    Worker().save_checkpoint(work/'global_step_2/actor',global_step=2)
    assert events==[('model','actor'),('EMA','v4-ema')]
    monkeypatch.setattr(s.shutil,'disk_usage',lambda path:SimpleNamespace(free=1000*GIB))
    assert plan()['storage']['arms'][arm]['ema_bytes']==8*PARAMETERS


def test_missing_model_configuration_refuses_on_cpu(planner,monkeypatch):
    """Missing model dimensions cannot silently fall back to an underpriced 8B label."""
    work,launch,s,plan=planner
    (Path(launch['MODEL_DIR'])/'config.json').unlink()
    monkeypatch.setattr(s.shutil,'disk_usage',lambda path:SimpleNamespace(free=1000*GIB))
    with pytest.raises(ValueError,match='phase0 disk.*config'):plan()


def test_runbook_explains_conservative_phase0_ceiling():
    """The partner must see why phase0 retains 560 when the campaign ceiling becomes 955."""
    text=(ROOT/'kit/README-phase0.md').read_text()
    assert '955' in text and '15.11' in text and 'deliberately' in text
