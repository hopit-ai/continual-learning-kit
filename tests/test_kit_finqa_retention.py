"""Main-only deletion guards, durable evidence, interruption and peak admission."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from test_kit_phase0_storage import CONFIG, PARAMETERS, GIB


def state(run,arm,step):
    from kit.v4_arms import checkpoint_path
    from kit.v4_retention import state_names
    root=checkpoint_path(run,arm,step)
    for name in state_names(arm):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((name+str(step)).encode())
    return root


def export(run,step):
    from kit.sim.v4_merge_standin import export as fake
    from kit import v4_retention as r
    fake(run/f'hf-step{step}')
    with patch.object(r,'reload_weights',return_value={'tensor_count':1,'dtype':'BF16'}):r.verify_export(run,step)


@pytest.mark.parametrize('arm',['S','F','R','D','D-R'])
def test_new_durable_state_supersedes_only_old_state_and_keeps_latest(tmp_path,arm):
    from kit import v4_retention as r
    from kit.v4_retry import saved_state
    run=tmp_path/'runs/slot';old=state(run,arm,1);new=state(run,arm,2)
    log=run/'metrics.jsonl';log.write_text('keep')
    r.after_save(run,arm,2,2)
    assert not (old/'data.pt').exists() and saved_state(new,arm)
    assert log.read_text()=='keep'
    journal=json.loads(next((run/'env/retention').glob('*.json')).read_text())
    assert journal['status']=='deleted' and all(f['sha256'] and f['bytes']>0 for f in journal['files'])
    assert journal['condition']['kind']=='newer_durable_state'
    r.after_save(run,arm,2,2) # idempotent replay
    assert saved_state(new,arm) # interruption can resume the retained latest state


def test_missing_new_state_or_unverified_step20_export_never_deletes(tmp_path):
    from kit import v4_retention as r
    run=tmp_path/'runs/slot';old=state(run,'D',20);new=state(run,'D',40)
    (new/'data.pt').unlink()
    with pytest.raises(ValueError):r.after_save(run,'D',40,40)
    assert (old/'data.pt').exists()
    (new/'data.pt').write_bytes(b'new')
    with pytest.raises(ValueError):r.after_save(run,'D',40,40)
    assert (old/'data.pt').exists()
    export(run,20);r.after_save(run,'D',40,40)
    assert not (old/'data.pt').exists() and (new/'data.pt').exists()
    assert (run/'hf-step20/model.safetensors').exists()


def evaluation(work,out,task,*,probe=False,qualification=False):
    from kit import v4_teacher as t
    panel=Path(work)/'v4/report-inputs'/(task+('-probe.jsonl' if probe else '-panel.jsonl'))
    panel.parent.mkdir(parents=True,exist_ok=True)
    size=20 if probe else 300 if task=='general-v1' else 1147
    if not panel.exists():t.write_jsonl(panel,[{'id':str(i),'CPU_standin':True} for i in range(size)])
    count=20 if qualification else size*(8 if probe else 1)
    t.write_jsonl(out/('raw_outputs.jsonl' if probe else 'responses.jsonl'),[{'id':str(i),'CPU_standin':True} for i in range(count)])
    for name in (('timing.json','probe-identity.json') if probe else ('forgetting.json' if task=='general-v1' else 'bed-score.json',)):
        t.atomic_json(out/name,{'CPU_standin':True})
    if not probe:t.write_jsonl(out/'tokens.jsonl',[{'id':str(i),'CPU_standin':True} for i in range(count)])


def finish_fixture(work):
    from kit import v4_retention as r,v4_teacher as t
    ident='q-D-finqa';run=work/'runs'/ident;state(run,'D',2);export(run,2)
    t.atomic_json(run/'run-summary.json',{'merged':1,'merge_returncode':0,'dose_complete':True,'completed_optimizer_updates':2})
    out=work/'v4/report-qualification/eval'/f'{ident}-finqa-2048';out.mkdir(parents=True)
    t.atomic_json(out/'checkpoint-identity.json',t.local_identity(run/'hf-step2','Qwen/Qwen3-8B'))
    evaluation(work,out,'finqa',qualification=True)
    for row in ('train-'+ident,'merge-'+ident,'score-'+ident+'-finqa-2048'):
        path=work/'campaign/v4-qualification'/row/'attempt-1';path.mkdir(parents=True)
        t.atomic_json(path/'verdict.json',{'row':row,'attempt':1,'verdict':'PASS'})
        t.atomic_json(path/'start.json',{'row':row,'attempt':1})
    return run,ident


@pytest.mark.parametrize('missing',['merge','export','reload','evaluation','evaluation_report','tokens','verdict','identity'])
def test_final_state_deleted_only_after_every_condition(tmp_path,missing):
    from kit import v4_retention as r
    run,ident=finish_fixture(tmp_path)
    paths={'merge':run/'run-summary.json','export':run/'hf-step2/model.safetensors',
           'reload':run/'env/export-reload-step2.json',
           'evaluation':tmp_path/f'v4/report-qualification/eval/{ident}-finqa-2048/responses.jsonl',
           'evaluation_report':tmp_path/f'v4/report-qualification/eval/{ident}-finqa-2048/bed-score.json',
           'tokens':tmp_path/f'v4/report-qualification/eval/{ident}-finqa-2048/tokens.jsonl',
           'verdict':tmp_path/f'campaign/v4-qualification/score-{ident}-finqa-2048/attempt-1/verdict.json',
           'identity':tmp_path/f'v4/report-qualification/eval/{ident}-finqa-2048/checkpoint-identity.json'}
    path=paths[missing];raw=path.read_bytes();path.unlink()
    with pytest.raises((ValueError,OSError)):r.finish_slot(tmp_path,'qualification',ident)
    assert (run/'sdft/global_step_2/data.pt').exists()
    path.write_bytes(raw)
    r.finish_slot(tmp_path,'qualification',ident)
    assert not (run/'sdft/global_step_2/data.pt').exists()
    r.finish_slot(tmp_path,'qualification',ident)
    assert (run/'hf-step2/model.safetensors').exists()


def test_deletion_intent_precedes_unlink_and_can_resume(tmp_path,monkeypatch):
    from kit import v4_retention as r
    run=tmp_path/'runs/slot';old=state(run,'S',1);new=state(run,'S',2)
    unlink=Path.unlink;calls=[]
    def interrupted(path,*args,**kwargs):
        if path.suffix=='.pt':
            journal=json.loads(next((run/'env/retention').glob('*.json')).read_text())
            assert journal['status']=='intent'
            calls.append(path)
            if len(calls)==2:raise OSError('interrupted deletion')
        return unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',interrupted)
    with pytest.raises(OSError):r.after_save(run,'S',2,2)
    monkeypatch.setattr(Path,'unlink',unlink)
    r.after_save(run,'S',2,2)
    assert not (old/'data.pt').exists() and (new/'data.pt').exists()


def test_unknown_state_or_symlink_refuses_whole_deletion(tmp_path):
    from kit import v4_retention as r
    run=tmp_path/'runs/slot';old=state(run,'F',1);state(run,'F',2)
    unknown=old/'unreviewed.pt';unknown.write_bytes(b'unknown')
    with pytest.raises(ValueError):r.after_save(run,'F',2,2)
    assert (old/'data.pt').exists()
    unknown.unlink();(old/'linked').symlink_to(run/'metrics.jsonl')
    with pytest.raises(ValueError):r.after_save(run,'F',2,2)
    assert (old/'data.pt').exists()


def test_main_disk_bound_uses_actual_serial_and_restoration_peaks(tmp_path):
    from kit.v4_finqa_site import disk_requirement
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text(json.dumps(CONFIG))
    disk=disk_requirement({'MODEL_DIR':str(model),'TEACHER_MODEL_DIR':str(model)})
    assert disk['required_bytes']==154*PARAMETERS+704*GIB
    assert disk['required_gib']==1879
    assert disk['peaks']['restoration']['state_count']==6
    assert disk['peaks']['scientific']['state_count']==2
    assert disk['retained_export_count']==61
    assert disk['historical_checkpoint_path_count']==574
    assert disk['peaks']['scientific']['extra_and_serialization_bytes']==574*GIB
    assert disk['peaks']['qualification']['extra_and_serialization_bytes']==34*GIB
    assert disk['peaks']['scientific']['total_bytes']+130*GIB==154*PARAMETERS+704*GIB


def test_incomplete_replacement_cannot_create_a_third_state(tmp_path):
    from kit import v4_retention as r
    run=tmp_path/'runs/slot';state(run,'D',20);state(run,'D',40)
    (run/'sdft/global_step_40/data.pt').unlink()
    r.before_save(run,'D',40) # resume can replace interrupted final state, latest20 is intact
    with pytest.raises(ValueError,match='coexistence'):r.before_save(run,'D',30)
    assert not (run/'sdft/global_step_30').exists()


def test_successive_saves_do_not_rebind_previously_deleted_evidence(tmp_path):
    from kit import v4_retention as r
    run=tmp_path/'runs/slot';state(run,'F',1);state(run,'F',2)
    r.after_save(run,'F',2,40);first=(run/'env/retention/state-step1.json').read_bytes()
    state(run,'F',3);r.after_save(run,'F',3,40)
    assert (run/'env/retention/state-step1.json').read_bytes()==first
    assert not (run/'train/global_step_2/data.pt').exists()


def test_changed_remaining_file_refuses_resumed_deletion(tmp_path,monkeypatch):
    from kit import v4_retention as r
    run=tmp_path/'runs/slot';old=state(run,'F',1);state(run,'F',2)
    original=Path.unlink
    def interrupt(path,*a,**kw):
        if path.suffix=='.pt':raise OSError('interrupt')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'unlink',interrupt)
    with pytest.raises(OSError):r.after_save(run,'F',2,2)
    monkeypatch.setattr(Path,'unlink',original)
    (old/'data.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed after intent'):r.after_save(run,'F',2,2)
    assert len(list(old.rglob('*.pt')))==25


def test_new_slot_refuses_old_unevaluated_state(tmp_path):
    from kit import v4_retention as r
    run=tmp_path/'runs/S-finqa-alone-s101';state(run,'S',40)
    with pytest.raises(ValueError,match='previous slot'):r.sweep(tmp_path,starting_slot='F-finqa-alone-s101')
    assert (run/'tool-sdpo/global_step_40/data.pt').exists()


def test_restoration_states_stay_until_verified_recorded_verdict(tmp_path,monkeypatch):
    from kit import v4_retention as r,v4_restore_check as restore
    for leg in ('uninterrupted','killed','resumed'):state(tmp_path/'runs'/('restoration-S-'+leg),'S',2)
    def refuse(archive):raise ValueError('missing verdict')
    monkeypatch.setattr(restore,'verify_archive',refuse)
    with pytest.raises(ValueError,match='missing verdict'):r.finish_restoration(tmp_path)
    assert len(list((tmp_path/'runs').rglob('data.pt')))==3
    root=tmp_path/'v4/report-qualification/restoration';root.mkdir(parents=True)
    (root/'qualification.json').write_text('{"passed":true}')
    monkeypatch.setattr(restore,'verify_archive',lambda archive:{'S':{'passed':True}})
    r.finish_restoration(tmp_path);r.finish_restoration(tmp_path)
    assert len(list((tmp_path/'runs').rglob('data.pt')))==0
    assert (root/'qualification.json').exists()


def test_cpu_reload_verifies_real_bf16_shards_and_hash_identity(tmp_path):
    torch=pytest.importorskip('torch');save=pytest.importorskip('safetensors.torch').save_file
    from kit import v4_retention as r
    run=tmp_path/'runs/slot';target=run/'hf-step20';target.mkdir(parents=True)
    for name in ('config.json','tokenizer.json','tokenizer_config.json'):(target/name).write_text('{}')
    save({'weight':torch.ones(2,dtype=torch.bfloat16)},str(target/'model.safetensors'))
    r.verify_export(run,20)
    assert json.loads((run/'env/export-reload-step20.json').read_text())['tensor_count']==1
    save({'weight':torch.zeros(2,dtype=torch.bfloat16)},str(target/'model.safetensors'))
    with pytest.raises(ValueError,match='identity changed'):r.verify_export(run,20)


def test_both_reader_evidence_checks_preserve_retirement_proofs(tmp_path):
    from kit import v4_retention as r
    from kit.v4_archive import DirectoryEvidence
    run,ident=finish_fixture(tmp_path);r.finish_slot(tmp_path,'qualification',ident)
    marker=tmp_path/'v4/report-inputs/retention-policy.json';marker.parent.mkdir(parents=True,exist_ok=True);marker.write_text(json.dumps(r.POLICY))
    r.verify_evidence(DirectoryEvidence(tmp_path))
    (run/'run-summary.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='proof changed'):r.verify_evidence(DirectoryEvidence(tmp_path))


def test_save_hook_runs_retention_for_slots_and_skips_sent_phase0(tmp_path,monkeypatch):
    from kit import v4_timing as timing,v4_retention as r
    from types import SimpleNamespace
    torch=pytest.importorskip('torch')
    run=tmp_path/'runs/slot';(run/'env').mkdir(parents=True)
    monkeypatch.setenv('V4_TELEMETRY','1');monkeypatch.setenv('VERL_FILE_LOGGER_PATH',str(run/'metrics.jsonl'))
    monkeypatch.setenv('KIT_V4_ARM','F');monkeypatch.setenv('STEPS','2');monkeypatch.setenv('RANK','0')
    events=[]
    monkeypatch.setattr(r,'before_save',lambda *a:events.append('before'))
    monkeypatch.setattr(r,'after_save',lambda *a:events.append('after'))
    class Trainer:
        def save(self,step):events.append('original');return 'unchanged'
    timing.install_telemetry(Trainer,exports='save')
    monkeypatch.setenv('V4_MAIN_RETENTION','slot')
    assert Trainer().save(2)=='unchanged' and events==['before','original','after']
    events.clear();monkeypatch.delenv('V4_MAIN_RETENTION')
    assert Trainer().save(2)=='unchanged' and events==['original']


@pytest.mark.parametrize('missing',['score-finqa-2048','score-finqa-8192','score-general-v1-2048','score-general-v1-8192','probe-finqa-0','probe-finqa-20','probe-finqa-40'])
def test_all_seven_main_evaluations_required_before_final_retirement(tmp_path,missing):
    from kit import v4_retention as r,v4_teacher as t
    ident='D-finqa-alone-s101';run=tmp_path/'runs'/ident;root=state(run,'D',40)
    export(run,20);export(run,40)
    t.atomic_json(run/'run-summary.json',{'merged':1,'merge_returncode':0,'dose_complete':True,'completed_optimizer_updates':40})
    initial=tmp_path/'v4/report-inputs/model-identities.json';initial.parent.mkdir(parents=True,exist_ok=True)
    t.atomic_json(initial,{'initial':t.local_identity(run/'hf-step20','Qwen/Qwen3-8B')})
    rows=['train-'+ident,'merge-'+ident]
    for spec in ['score-finqa-2048','score-finqa-8192','score-general-v1-2048','score-general-v1-8192','probe-finqa-0','probe-finqa-20','probe-finqa-40']:
        op,taskcap=spec.split('-',1);row=op+'-'+ident+'-'+taskcap;rows.append(row)
        directory=f'eval/{ident}-{taskcap}' if op=='score' else f'probes/{ident}-{taskcap}'
        out=tmp_path/'v4/report-main'/directory;out.mkdir(parents=True)
        step=40 if op=='score' else int(taskcap.split('-')[-1])
        t.atomic_json(out/'checkpoint-identity.json',t.local_identity(run/f'hf-step{step or 20}','Qwen/Qwen3-8B'))
        evaluation(tmp_path,out,'general-v1' if taskcap.startswith('general-v1') else 'finqa',probe=op=='probe')
    for row in rows:
        out=tmp_path/'campaign/v4-main'/row/'attempt-1';out.mkdir(parents=True)
        t.atomic_json(out/'verdict.json',{'row':row,'attempt':1,'verdict':'PASS'})
        t.atomic_json(out/'start.json',{'row':row,'attempt':1})
    op,taskcap=missing.split('-',1)
    path=tmp_path/f'campaign/v4-main/{op}-{ident}-{taskcap}/attempt-1/verdict.json';raw=path.read_bytes();path.unlink()
    with pytest.raises(OSError):r.finish_slot(tmp_path,'main',ident)
    assert (root/'data.pt').exists()
    path.write_bytes(raw);r.finish_slot(tmp_path,'main',ident)
    assert not (root/'data.pt').exists()
    assert (run/'hf-step20/model.safetensors').exists() and (run/'hf-step40/model.safetensors').exists()


def test_interrupted_cpu_trainer_reloads_retained_model_optimizer_and_ema(tmp_path):
    torch=pytest.importorskip('torch')
    from kit import v4_retention as r
    from kit.v4_arms import checkpoint_path
    from kit.v4_retry import saved_state
    def trainer():
        torch.manual_seed(101)
        model=torch.nn.Linear(1,1);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        ema={k:v.detach().clone() for k,v in model.state_dict().items()}
        return model,optimizer,ema
    def update(model,optimizer,ema):
        optimizer.zero_grad();loss=(model(torch.tensor([[2.]]))-3).square().sum()
        loss.backward();optimizer.step()
        for k,v in model.state_dict().items():ema[k].mul_(.99).add_(v,alpha=.01)
        return float(loss.detach())
    expected,opt,expected_ema=trainer();trace=[update(expected,opt,expected_ema) for _ in range(3)]
    model,optimizer,ema=trainer();run=tmp_path/'runs/resume';observed=[]
    for step in (1,2):
        observed.append(update(model,optimizer,ema));r.before_save(run,'D',step)
        root=checkpoint_path(run,'D',step)
        for name in r.state_names('D'):
            file=root/name;file.parent.mkdir(parents=True,exist_ok=True)
            payload=(optimizer.state_dict() if name.startswith('actor/optim_') else
                     ema if '/v4-ema/' in name else
                     model.state_dict() if name.startswith('actor/model_') else
                     {'completed':step,'rng':torch.get_rng_state()})
            torch.save(payload,file)
        r.after_save(run,'D',step,40)
    assert not (checkpoint_path(run,'D',1)/'data.pt').exists()
    latest=checkpoint_path(run,'D',2);digest=saved_state(latest,'D')
    model,optimizer,ema=trainer() # process interruption loses live objects
    model.load_state_dict(torch.load(latest/'actor/model_world_size_8_rank_0.pt',weights_only=True))
    optimizer.load_state_dict(torch.load(latest/'actor/optim_world_size_8_rank_0.pt',weights_only=True))
    ema=torch.load(latest/'actor/v4-ema/model_world_size_8_rank_0.pt',weights_only=True)
    loader=torch.load(latest/'data.pt',weights_only=True);torch.set_rng_state(loader['rng'])
    assert loader['completed']==2 and saved_state(latest,'D')==digest
    observed.append(update(model,optimizer,ema))
    assert observed==trace
    assert all(torch.equal(v,expected.state_dict()[k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,expected_ema[k]) for k,v in ema.items())


def test_linked_deletion_evidence_refuses_before_any_unlink(tmp_path):
    from kit import v4_retention as r
    run=tmp_path/'runs/slot';old=state(run,'F',1);state(run,'F',2)
    env=run/'env';env.mkdir();elsewhere=tmp_path/'uncollected';elsewhere.mkdir()
    (env/'retention').symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError,match='linked retirement evidence'):r.after_save(run,'F',2,2)
    assert (old/'data.pt').exists() and not list(elsewhere.iterdir())


def test_partial_evaluation_cannot_retire_state_despite_pass_row(tmp_path):
    from kit import v4_retention as r,v4_teacher as t
    run,ident=finish_fixture(tmp_path)
    path=tmp_path/f'v4/report-qualification/eval/{ident}-finqa-2048/responses.jsonl'
    t.write_jsonl(path,t.read_jsonl(path)[:-1])
    with pytest.raises(ValueError,match='evaluation incomplete'):r.finish_slot(tmp_path,'qualification',ident)
    assert (run/'sdft/global_step_2/data.pt').exists()


def test_cpu_adapter_writes_complete_evaluation_in_fresh_work(tmp_path):
    from kit.sim.v4_merge_standin import prepare_retained_run,recover,evaluation_and_sweep
    from kit import v4_teacher as t
    ident='q-D-finqa'
    prepare_retained_run(tmp_path,'qualification',ident);recover(tmp_path,'qualification',ident)
    assert not (tmp_path/'v4/report-inputs').exists()
    evaluation_and_sweep(tmp_path,'qualification','score-'+ident+'-finqa-2048')
    assert len(t.read_jsonl(tmp_path/'v4/report-inputs/finqa-panel.jsonl'))==1147
    output=tmp_path/f'v4/report-qualification/eval/{ident}-finqa-2048'
    assert len(t.read_jsonl(output/'responses.jsonl'))==len(t.read_jsonl(output/'tokens.jsonl'))==20
    assert (output/'bed-score.json').exists()


def restoration_arm_fixture(work,arm):
    """CPU state bytes and real comparator input, never a real-stack qualification."""
    from kit import v4_teacher as t
    root=work/'v4/report-qualification/restoration'/arm;root.mkdir(parents=True,exist_ok=True)
    base=root.parent.parent
    if not (base/'environment.json').exists():t.atomic_json(base/'environment.json',{'CPU_fixture':True})
    for leg,event in [('uninterrupted','saved'),('killed','saved'),('resumed','loaded')]:
        run=work/'runs'/f'restoration-{arm}-{leg}'
        state(run,arm,2)
        for rank in range(8):
            t.atomic_json(root/f'{leg}-{event}-model-s2-r{rank}.json',dict(rank=rank,step=2,components={c:hashlib.sha256(c.encode()).hexdigest() for c in ('model','optimizer','rng','scheduler')}))
            if arm in ('S','D','D-R'):t.atomic_json(root/f'{leg}-{event}-EMA-s2-r{rank}.json',{'components':{'EMA':'e'*64}})
        t.atomic_json(root/f'{leg}-{event}-dataloader-s2.json',{'components':{'dataloader':'d'*64}})
        if leg!='killed':t.write_jsonl(root/(leg+'-metrics.jsonl'),[{'step':3,'data':{'train/loss' if arm in ('F','R') else 'actor/pg_loss':1.0}}])
    t.atomic_json(root/'kill.json',dict(gpu_idle_verified=True,gpu_query=dict(returncode=0,stdout=''),launcher_returncode=-9,ray_stop_returncode=0,signal=9,terminated=True,saved_step=2))
    t.atomic_json(root/'next-loss.json',dict(step=3,uninterrupted=1.,resumed=1.))
    return root


@pytest.mark.parametrize('damage',['missing','failed','raw_changed','result_changed','environment_changed'])
def test_per_arm_retirement_requires_own_durable_reverified_verdict(tmp_path,damage):
    from kit import v4_restore_check as restore,v4_retention as r,v4_teacher as t
    root=restoration_arm_fixture(tmp_path,'S');restoration_arm_fixture(tmp_path,'F')
    verdict=restore.record_arm_verdict(tmp_path,'S')
    path=root/'verdict.json';original=path.read_bytes()
    if damage=='missing':path.unlink()
    elif damage in ('failed','result_changed'):
        doc=json.loads(original)
        if damage=='failed':doc['passed']=False
        else:doc['result']['next_step_loss']=2.
        t.atomic_json(path,doc)
    elif damage=='raw_changed':t.atomic_json(root/'next-loss.json',dict(step=3,uninterrupted=2.,resumed=2.))
    else:t.atomic_json(root.parent.parent/'environment.json',{'changed':True})
    with pytest.raises((ValueError,OSError,KeyError)):r.finish_restoration_arm(tmp_path,'S')
    assert len(list((tmp_path/'runs').rglob('data.pt')))==6
    assert not list((tmp_path/'runs').rglob('retention/*.json'))
    assert verdict['passed']


def test_per_arm_retirement_preserves_other_arms_and_resumes_interrupted_cleanup(tmp_path,monkeypatch):
    from kit import v4_restore_check as restore,v4_retention as r
    from kit.v4_archive import DirectoryEvidence
    root=restoration_arm_fixture(tmp_path,'D-R');restoration_arm_fixture(tmp_path,'D')
    restore.record_arm_verdict(tmp_path,'D-R')
    assert not (root.parent/'qualification.json').exists() # full verdict is not a cleanup prerequisite
    unlink=Path.unlink;calls=[]
    def interrupted(path,*args,**kwargs):
        if path.suffix=='.pt':
            calls.append(path)
            if len(calls)==2:raise OSError('interrupted arm cleanup')
        return unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',interrupted)
    with pytest.raises(OSError):r.finish_restoration_arm(tmp_path,'D-R')
    monkeypatch.setattr(Path,'unlink',unlink)
    r.finish_restoration_arm(tmp_path,'D-R');r.finish_restoration_arm(tmp_path,'D-R')
    assert not list((tmp_path/'runs').glob('restoration-D-R-*/**/*.pt'))
    assert len(list((tmp_path/'runs').glob('restoration-D-*/**/data.pt')))==3
    for path in (tmp_path/'runs').rglob('env/retention/*.json'):
        doc=json.loads(path.read_text());assert doc['condition']['arm']=='D-R'
        assert doc['reason']=='recorded_verified_restoration_arm_verdict'
    marker=tmp_path/'v4/report-inputs/retention-policy.json';marker.parent.mkdir(parents=True);marker.write_text(json.dumps(r.POLICY))
    r.verify_evidence(DirectoryEvidence(tmp_path))


def test_full_restoration_is_five_bound_per_arm_verdicts_and_conjunction(tmp_path):
    from kit import v4_restore_check as restore,v4_retention as r,v4_teacher as t
    from kit.v4_archive import DirectoryEvidence
    from kit.v4_finqa_campaign import ALL_ARMS,RESTORATION_CAP_SECONDS,RESTORATION_LEG_SECONDS
    for arm in ALL_ARMS:
        restoration_arm_fixture(tmp_path,arm);restore.record_arm_verdict(tmp_path,arm);r.finish_restoration_arm(tmp_path,arm)
    receipt=restore.record_full_verdict(tmp_path)
    assert receipt['passed']==all(row['passed'] for row in receipt['results'].values())
    assert set(receipt['arm_verdicts'])==set(ALL_ARMS)
    assert restore.verify_archive(DirectoryEvidence(tmp_path))==receipt['results']
    assert RESTORATION_CAP_SECONDS==7550 and RESTORATION_LEG_SECONDS==450
    root=tmp_path/'v4/report-qualification/restoration';path=root/'D/verdict.json';doc=json.loads(path.read_text());doc['passed']=False;t.atomic_json(path,doc)
    with pytest.raises(ValueError):restore.verify_archive(DirectoryEvidence(tmp_path))
    with pytest.raises(ValueError):restore.record_full_verdict(tmp_path)


def test_failed_arm_comparison_records_failure_and_retains_all_its_states(tmp_path):
    from kit import v4_restore_check as restore,v4_retention as r,v4_teacher as t
    root=restoration_arm_fixture(tmp_path,'D')
    t.atomic_json(root/'next-loss.json',dict(step=3,uninterrupted=1.,resumed=2.))
    with pytest.raises(ValueError):restore.record_arm_verdict(tmp_path,'D')
    assert json.loads((root/'verdict.json').read_text())['passed'] is False
    with pytest.raises(ValueError):r.finish_restoration_arm(tmp_path,'D')
    assert len(list((tmp_path/'runs').rglob('data.pt')))==3


@pytest.mark.parametrize('failed_arm',[None,'F'])
def test_real_restoration_driver_retires_each_arm_before_next_launch(tmp_path,monkeypatch,failed_arm):
    from types import SimpleNamespace
    from kit import v4_restore_check as restore,v4_retention as r,v4_teacher as t,v4_campaign_ops as ops,runner
    from kit.v4_finqa_campaign import ALL_ARMS
    base=tmp_path/'v4/report-qualification';(base/'scheduled/restoration-finqa').mkdir(parents=True)
    inputs=tmp_path/'v4/report-inputs';inputs.mkdir();t.atomic_json(inputs/'phase0-binding.json',{})
    events=[];timeouts=[]
    monkeypatch.setattr(ops,'launcher_env',lambda work,arm,name,*args,**kwargs:{'TEST_ARM':arm,'TEST_NAME':name})
    def launch(command,**kwargs):
        if command[0]=='ray':return {'returncode':0}
        env=kwargs['env'];arm=env['TEST_ARM'];leg=env['V4_RESTORE_LEG']
        timeouts.append(kwargs['timeout']);events.append((arm,leg))
        if leg=='uninterrupted':
            for previous in ALL_ARMS[:ALL_ARMS.index(arm)]:
                assert not list((tmp_path/'runs').glob(f'restoration-{previous}-*/**/*.pt'))
                assert (base/'restoration'/previous/'verdict.json').exists()
            assert not (base/'restoration/qualification.json').exists()
            root=restoration_arm_fixture(tmp_path,arm)
            t.atomic_json(root/'ready-to-kill.json',{'step':2,'checkpoint_durable':True})
            for part in ('uninterrupted','killed','resumed'):
                run=tmp_path/'runs'/f'restoration-{arm}-{part}';r.durable_state(run,arm,2)
                if part!='killed':(run/'metrics.jsonl').write_bytes((root/(part+'-metrics.jsonl')).read_bytes())
        if arm==failed_arm and leg=='resumed':
            path=base/'restoration'/arm/'resumed-loaded-model-s2-r7.json'
            doc=json.loads(path.read_text());doc['components']['model']='0'*64;t.atomic_json(path,doc)
        return {'returncode':0}
    monkeypatch.setattr(runner,'bounded_command',launch)
    class Killed:
        pid=999999;returncode=None
        def __init__(self,*args,**kwargs):events.append((kwargs['env']['TEST_ARM'],'killed'))
        def poll(self):return None
        def wait(self,timeout):self.returncode=-9;return -9
    monkeypatch.setattr(restore.subprocess,'Popen',Killed)
    monkeypatch.setattr(restore.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=''))
    monkeypatch.setattr(restore.os,'killpg',lambda *args:None)
    if failed_arm:
        with pytest.raises(ValueError,match='state hashes differ'):restore.run_check(tmp_path)
        assert events==[(a,leg) for a in ('S','F') for leg in ('uninterrupted','killed','resumed')]
        assert len(list((tmp_path/'runs').glob('restoration-F-*/**/data.pt')))==3
        assert not list((tmp_path/'runs').glob('restoration-S-*/**/*.pt'))
        assert not (base/'restoration/qualification.json').exists()
    else:
        results=restore.run_check(tmp_path)
        assert events==[(a,leg) for a in ALL_ARMS for leg in ('uninterrupted','killed','resumed')]
        assert all(v['passed'] for v in results.values()) and len(results)==5
        assert not list((tmp_path/'runs').rglob('*.pt'))
        recorded=json.loads((base/'restoration/qualification.json').read_text())
        assert recorded['results']==results and len(recorded['arm_verdicts'])==5
        assert restore.run_check(tmp_path)==results and len(events)==15 # replay needs no deleted trainer bytes
    assert set(timeouts)=={450}
