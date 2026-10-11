"""Round-2 regressions: real orchestration/consumers, explicit CPU model adapters."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import pytest
from test_kit_finqa_retention import state, export, finish_fixture
from test_kit_finqa_host import host, test_bound_zero_gpu_receipt_is_reused_without_download as host_setup

ROOT=Path(__file__).resolve().parents[1]


def timing_archive(tmp_path,monkeypatch,seconds=150):
    from kit import v4_phase0 as phase0
    monkeypatch.setattr(phase0,'read_archive',lambda path:{'status':'technical pass','tables':{'training':{a:{'steps':2} for a in 'SFRD'}}})
    files={'k8b4/containment/allocation-ledger.json':json.dumps({'allocations':{'q':dict(width=8,start=1,end=6601,segments=[dict(block='qualification',start=1,end=6601)])}}).encode()}
    for arm in 'SFRD':
        files[f'v4/report-phase0/timings/train-q-{arm}-finqa.json']=json.dumps(dict(ok=True,row=f'train-q-{arm}-finqa',phase='phase0',operation='train',wall_seconds=seconds,allocation_id='q')).encode()
    files['collect.manifest.json']=json.dumps({'schema':'kit-collect.v1','files':[dict(path=n,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for n,raw in files.items()],'truncated':[],'symlinks':[]}).encode()
    archive=tmp_path/'SYNTHETIC-phase0.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,raw in files.items():
            info=tarfile.TarInfo('return/'+name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    return archive


@pytest.mark.parametrize('consumer',['restoration','row_validator'])
def test_A3_below_floor_generator_to_real_consumers(tmp_path,monkeypatch,consumer):
    from kit import v4_finqa_campaign as fc,v4_restore_check as restore,p4_contain as contain,runner,v4_campaign_ops as ops
    graph=fc.build('qualification',phase0_launch_archive=timing_archive(tmp_path,monkeypatch))
    rows={r['id']:r for r in graph['rows']};caps=graph['v4']['launch_allowances']
    assert caps['qualification_launch_seconds']==225 and caps['restoration_leg_seconds']==450
    assert caps['source']['slowest_two_step_launch_seconds']==150
    if consumer=='restoration':
        monkeypatch.setenv('V4_RESTORATION_LAUNCH_ALLOWANCES',rows['restoration-check']['env']['V4_RESTORATION_LAUNCH_ALLOWANCES'])
        (ops.paths(tmp_path,'qualification')/'scheduled/restoration-finqa').mkdir(parents=True)
        monkeypatch.setattr(ops,'launcher_env',lambda *a,**kw:{})
        observed=[]
        def launch(*a,**kw):observed.append(kw['timeout']);raise RuntimeError('CPU external-launch boundary')
        monkeypatch.setattr(runner,'bounded_command',launch)
        with pytest.raises(RuntimeError,match='CPU external-launch'):restore.run_check(tmp_path)
        assert observed==[450]
    else:
        row=rows['train-q-D-R-finqa'];command=row['command'];cap=int(command[command.index('--time-cap')+1])
        monkeypatch.setattr(contain,'stop_gate',lambda *a:(_ for _ in ()).throw(RuntimeError('CPU after real row validation')))
        # Unwrap only allocation tracking (external scheduler); run_row's actual validator runs.
        with pytest.raises(RuntimeError,match='after real row validation'):
            contain.run_row.__wrapped__(tmp_path,tmp_path/'out',['true'],8,cap)
        assert all(r['timeout_seconds']>=contain.MIN_ROW_CAP for r in graph['rows'])


def test_A1_real_runner_failure_probe_later_rows_and_retry(tmp_path,monkeypatch):
    from kit import runner,v4_campaign_ops as ops,v4_prepare as prep,v4_retention as r,v4_teacher as t
    from kit.sim.v4_merge_standin import prepare_retained_run,recover,evaluation_and_sweep
    ident='D-finqa-alone-s101';run=tmp_path/'runs'/ident
    prepare_retained_run(tmp_path,'main',ident);recover(tmp_path,'main',ident)
    specs=['score-finqa-2048','score-finqa-8192','score-general-v1-2048','score-general-v1-8192','probe-finqa-0','probe-finqa-20','probe-finqa-40']
    for name in ['train-'+ident,'merge-'+ident]+[s.split('-',1)[0]+'-'+ident+'-'+s.split('-',1)[1] for s in specs]:
        out=tmp_path/'campaign/v4-main'/name/'attempt-1';out.mkdir(parents=True)
        t.atomic_json(out/'start.json',dict(row=name,attempt=1,inputs={}))
        t.atomic_json(out/'verdict.json',dict(row=name,attempt=1,verdict='PASS',inputs={}))
        if name.startswith(('score-','probe-')):evaluation_and_sweep(tmp_path,'main',name)
    failed=f'score-{ident}-general-v1-2048';probe=f'probe-{ident}-finqa-40'
    for name in (failed,probe):
        import shutil
        shutil.rmtree(tmp_path/'campaign/v4-main'/name)
    monkeypatch.setenv('WORK',str(tmp_path));monkeypatch.setenv('V4_FINQA_CAMPAIGN','1');monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(prep,'verify',lambda *a:None)
    monkeypatch.setattr(ops,'scoring_environment_gate',lambda *a:None)
    monkeypatch.setattr(ops,'registered_caps',lambda *a:dict.fromkeys(('teacher','rewrite','train','merge','score','probe','cpu'),420))
    monkeypatch.setattr(ops,'score',lambda *a:None);monkeypatch.setattr(ops,'probe',lambda *a:None)
    monkeypatch.setattr(ops,'schedule',lambda *a:None);monkeypatch.setattr(ops,'report',lambda *a:None)
    monkeypatch.setattr(ops,'train',lambda *a:None)
    monkeypatch.setattr(ops,'run',lambda *a:None) # external collector only
    operations=[(failed,'score',['--checkpoint',ident,'--task','general-v1','--cap','2048']),
                (probe,'probe',['--slot',ident,'--task','finqa','--step','40']),
                ('schedule-next','schedule',['--slot','F-finqa-alone-s101']),('train-'+ident,'train',['--slot',ident]),('report','report',[]),
                ('archive-pack','archive-pack',[]),('FINISHED','finished',[])]
    rows=[dict(id=name,needs=[],wants=[failed] if name!=failed else [],order=[failed] if name!=failed else [],command=[sys.executable,str(ROOT/'kit/v4_campaign_ops.py'),op,'--work',str(tmp_path),'--phase','main','--row',name,*args]) for name,op,args in operations]
    rows[0]['command']=[sys.executable,str(ROOT/'kit/p4_contain.py'),'row','--out',str(tmp_path/'containment/attempt-{attempt}'),'--time-cap','420','--',*rows[0]['command']]
    path=tmp_path/'campaign.json';path.write_text(json.dumps(dict(schema=runner.SCHEMA,name='v4-main',rows=rows,v4={'phase':'main','scope':'finqa-alone'})))
    campaign=runner.load_campaign(path);fail_once=True
    def external(command,*,log,**kw):
        nonlocal fail_once
        code=0
        try:
            body=command[command.index('--')+1:] if '--' in command else command
            ops.main(body[2:])
            if command[command.index('--row')+1]==failed and fail_once:code=1;fail_once=False
        except (ValueError,OSError,KeyError) as exc:code=1;log.parent.mkdir(parents=True,exist_ok=True);log.write_text(str(exc))
        if not log.exists():log.write_text('explicit CPU model adapter\n')
        return dict(returncode=code,failure_type='CPU injected IO failure' if code else None)
    monkeypatch.setattr(runner,'bounded_command',external)
    from kit import v4_retry
    # Synthetic external scheduler has no automatic termination proof: review is explicit.
    monkeypatch.setattr(v4_retry,'failure_record',lambda *a:(_ for _ in ()).throw(ValueError('CPU scheduler adapter needs review')))
    assert runner.run_row(campaign,campaign['rows'][0],None)==runner.EXIT_FAILED
    for row in campaign['rows'][1:]:
        assert runner.run_row(campaign,row,None)==0
        assert list(run.rglob('*.pt')) # failed score forbids deletion
    r.sweep(tmp_path,starting_slot='F-finqa-alone-s101')
    assert list(run.rglob('*.pt')) and list((tmp_path/'v4/warnings').glob('*.json'))
    # No review file: a local fix reruns the failed row as a new attempt.
    # New start exists, but verdict does not, when the real operation entry sweeps.
    assert runner.run_row(campaign,campaign['rows'][0],None)==0
    assert list(run.rglob('*.pt'))
    r.sweep(tmp_path,starting_slot='F-finqa-alone-s101')
    assert not list(run.rglob('*.pt'))


def test_3_finish_slot_replays_interrupted_midpoint_intent(tmp_path,monkeypatch):
    from kit import v4_retention as r
    run,ident=finish_fixture(tmp_path);old=state(run,'D',1)
    unlink=Path.unlink
    monkeypatch.setattr(Path,'unlink',lambda path,*a,**kw:(_ for _ in ()).throw(OSError('interrupt')) if path.suffix=='.pt' else unlink(path,*a,**kw))
    with pytest.raises(OSError):r.after_save(run,'D',2,2)
    monkeypatch.setattr(Path,'unlink',unlink)
    r.finish_slot(tmp_path,'qualification',ident)
    assert not list(old.rglob('*.pt'))
    assert json.loads((run/'env/retention/state-step1.json').read_text())['status']=='deleted'


@pytest.mark.parametrize('changed',[False,True])
def test_4_resume_recovers_mid_export_only_from_verified_state(tmp_path,monkeypatch,changed):
    from kit import v4_retention as r,v4_campaign_ops as ops
    from kit.sim.v4_merge_standin import export as cpu_export
    run=tmp_path/'runs/slot';old=state(run,'D',20);r.durable_state(run,'D',20);new=state(run,'D',40)
    calls=[]
    def merge(argv,env):calls.append(argv);cpu_export(Path(argv[argv.index('--target_dir')+1]))
    monkeypatch.setattr(ops,'run',merge);monkeypatch.setattr(r,'reload_weights',lambda *a:{'tensor_count':1,'dtype':'BF16'})
    if changed:(old/'data.pt').write_bytes(b'changed')
    if changed:
        with pytest.raises(ValueError):r.after_save(run,'D',40,40)
        assert not calls and (old/'data.pt').exists()
    else:
        r.after_save(run,'D',40,40)
        assert calls and (run/'env/export-reload-step20.json').exists()
        assert not (old/'data.pt').exists() and (new/'data.pt').exists()


def test_5_durable_hashes_reused_and_retirement_cap_registered(tmp_path,monkeypatch):
    from kit import v4_retention as r,v4_finqa_campaign as fc
    run=tmp_path/'runs/slot';old=state(run,'F',1);state(run,'F',2)
    r.durable_state(run,'F',1);r.durable_state(run,'F',2)
    original=r.sha_file;reads=[]
    def count(path):
        if Path(path).suffix=='.pt':reads.append(path)
        return original(path)
    monkeypatch.setattr(r,'sha_file',count)
    r.after_save(run,'F',2,2)
    assert not reads and not (old/'data.pt').exists()
    graph=fc.build('qualification');rows={row['id']:row for row in graph['rows']}
    assert graph['v4']['retirement_io_seconds']==r.RETIREMENT_IO_SECONDS
    assert rows['schedule-q-F-finqa']['timeout_seconds']==10*(300+r.RETIREMENT_IO_SECONDS)
    assert rows['report']['timeout_seconds']==10*(300+r.RETIREMENT_IO_SECONDS)


def test_5_telemetry_uses_the_single_durable_state_hash(tmp_path,monkeypatch):
    from kit import v4_retention as r,v4_timing as timing,v4_retry
    run=tmp_path/'runs/slot';(run/'env').mkdir(parents=True)
    for key,value in dict(V4_TELEMETRY='1',V4_CHECKPOINT_STATE='1',V4_MAIN_RETENTION='slot',
                          VERL_FILE_LOGGER_PATH=str(run/'metrics.jsonl'),KIT_V4_ARM='F',RANK='0',STEPS='2').items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(v4_retry,'saved_state',lambda *a:pytest.fail('second complete state read before durable_state'))
    class Trainer:
        config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(run/'train')))
        def save(self,step):state(run,'F',step)
    timing.install_telemetry(Trainer,exports='save')
    Trainer().save(2)
    assert (run/'env/saved-state-step2.json').exists()


def test_5_retirement_io_deadline_retains_replayable_intent(tmp_path,monkeypatch):
    from kit import v4_retention as r
    run=tmp_path/'runs/slot';old=state(run,'F',1);state(run,'F',2)
    # Only retirement is bounded; save/hash/merge use the trainer row cap.
    r.durable_state(run,'F',1);r.durable_state(run,'F',2)
    clock=[0]
    def now():clock[0]+=r.RETIREMENT_IO_SECONDS+1;return clock[0]
    monkeypatch.setattr(r.time,'monotonic',now)
    with pytest.raises(OSError,match='IO cap'):
        r.retire(tmp_path,run,'F',1,{'kind':'newer_durable_state','new_step':2,'proofs':{}})
    assert (old/'data.pt').exists()


def test_6_replay_rejects_forged_unlink_target_even_if_manifest_rehashed(tmp_path,monkeypatch):
    from kit import v4_retention as r,v4_teacher as t
    run=tmp_path/'runs/slot';old=state(run,'F',1);state(run,'F',2)
    protected=old/'keep.json';protected.write_text('{}');unlink=Path.unlink
    monkeypatch.setattr(Path,'unlink',lambda path,*a,**kw:(_ for _ in ()).throw(OSError('interrupt')) if path.suffix=='.pt' else unlink(path,*a,**kw))
    with pytest.raises(OSError):r.after_save(run,'F',2,2)
    monkeypatch.setattr(Path,'unlink',unlink)
    journal=run/'env/retention/state-step1.json';doc=json.loads(journal.read_text())
    next(row for row in doc['files'] if row['path']=='keep.json')['delete']=True
    doc['manifest_sha256']=t.sha(json.dumps(doc['files'],sort_keys=True));t.atomic_json(journal,doc)
    with pytest.raises(ValueError,match='unsafe|unreviewed'):r.retire(tmp_path,run,'F',1,doc['condition'])
    assert protected.exists() and (old/'data.pt').exists()


def test_7_bound_archive_requires_retention_policy(tmp_path):
    from kit import v4_retention as r,v4_teacher as t
    from kit.v4_archive import DirectoryEvidence
    (tmp_path/'v4/report-inputs').mkdir(parents=True)
    t.atomic_json(tmp_path/'v4/report-inputs/phase0-binding.json',{})
    with pytest.raises(ValueError,match='retention policy'):r.verify_evidence(DirectoryEvidence(tmp_path))


def test_8_header_creates_scheduler_log_directory_on_cpu(tmp_path):
    from kit.v4_allocation import header
    for phase,stage in [('qualification','qualification'),('main','teacher')]:
        text=header(dict(phase=phase,stage=stage,sbatch_time='01:00:00'),tmp_path)
        folder=tmp_path/f'v4/report-{phase}/allocation'
        assert folder.is_dir() and str(folder/(stage+'-%j.out')) in text


def test_9_runbook_failure_returns_without_logging_out(tmp_path):
    text=(ROOT/'kit/README-finqa.md').read_text()
    # Execute the actual documented helper in a live shell; its caller must survive failure.
    start=text.index('finqa_host_exports()');end=text.index('\n}',start)+2
    function=text[start:end]
    stub=tmp_path/'v4_finqa_site.py';stub.write_text('raise SystemExit(2)\n')
    result=subprocess.run(['/bin/bash','-c',function+'\nfinqa_host_exports inherit\nstatus=$?\nprintf "SURVIVED:%s\\n" "$status"'],env={**os.environ,'KIT':str(tmp_path),'WORK':str(tmp_path)},capture_output=True,text=True)
    assert result.returncode==0 and 'SURVIVED:2' in result.stdout and 'STOP' in result.stderr


def test_A2_new_shell_tag_exports_and_real_main_prepare(host,tmp_path,monkeypatch,capsys):
    from kit import v4_finqa_site as site,v4_prepare as prep,v4_campaign_ops as ops,v4_teacher as t
    host_setup(host,monkeypatch,tmp_path)
    work,_=host;doc=json.loads((work/site.LAUNCH).read_text());Path(doc['environment']['MODEL_DIR'],'model.safetensors').write_bytes(b'CPU stand-in')
    monkeypatch.setenv('V4_KIT_TAG',doc['environment']['V4_KIT_TAG']);site.inherit(work)
    original=(work/site.LAUNCH).read_bytes()
    monkeypatch.setattr(prep,'import_smoke',lambda work,phase:smoke)
    smoke=work/'v4/report-inputs/import-smoke/receipt.json';smoke.parent.mkdir(parents=True);t.atomic_json(smoke,{'CPU_adapter':True})
    monkeypatch.setattr(t,'load_student_tokenizer',lambda *a:SimpleNamespace(encode=lambda *a,**kw:[1],apply_chat_template=lambda *a,**kw:'CPU prompt'))
    def data(work,phase):
        base=work/f'v4/report-{phase}';base.mkdir(parents=True,exist_ok=True)
        t.write_jsonl(base/'finqa-pool.jsonl',[{'id':'CPU fixture','task':'finqa','question':'CPU fixture prompt'}])
    monkeypatch.setattr(ops,'prepare',data)
    package=tmp_path/'inference.py';package.write_text('# CPU external inference adapter')
    def inference(argv,**kw):
        content={'vllm':'0.17.0'} if 'vllm.__version__' in argv[-1] else {str(package):{'sha256':prep.digest(package),'bytes':package.stat().st_size}}
        return SimpleNamespace(returncode=0,stdout=json.dumps(content),stderr='')
    # Real qualification prepare first, in the qualification login environment.
    with monkeypatch.context() as context:
        for key,value in site.exports(work).items():context.setenv(key,value)
        context.delenv('SLURM_JOB_ID',raising=False)
        context.setattr(prep.subprocess,'run',inference)
        assert prep.prepare(work,'qualification').is_file()
    monkeypatch.setenv('V4_KIT_TAG','new-main-tag')
    assert site.inherit(work)[0]['environment']['V4_KIT_TAG']=='new-main-tag'
    assert (work/site.LAUNCH).read_bytes()==original
    assert site.main(['exports','--work',str(work)])==0
    exports=capsys.readouterr().out
    # A fresh login shell receives only frozen exports, plus the NEW tag supplied by the owner.
    shell=subprocess.run(['/bin/bash','-c',exports+'\n'+sys.executable+' -c '+__import__('shlex').quote('import os,json;print(json.dumps(dict(os.environ)))')],env={'PATH':os.environ['PATH'],'V4_KIT_TAG':'new-main-tag'},capture_output=True,text=True,check=True)
    fresh=json.loads(shell.stdout);assert fresh['V4_KIT_TAG']=='new-main-tag'
    assert fresh['MODEL_DIR']==doc['environment']['MODEL_DIR'] and fresh['V4_TRAINER_ACTIVATE']==doc['trainer_activate']
    with monkeypatch.context() as context:
        for key in list(os.environ):context.delenv(key,raising=False)
        for key,value in fresh.items():context.setenv(key,value)
        context.setattr(prep.subprocess,'run',inference)
        assert prep.prepare(work,'main').is_file()
        assert prep.verify(work,'main')
    # Plan 15.21 permits moving infrastructure while preserving initial evidence.
    monkeypatch.setattr(site,'__file__',str(tmp_path/'another-kit/v4_finqa_site.py'))
    moved=site.inherit(work)[0]
    assert moved['environment']['KIT']==str(tmp_path/'another-kit')
    assert (work/site.LAUNCH).read_bytes()==original
    assert list((work/'v4/warnings').glob('*.json'))
    assert site.exports(work)['KIT']==str(tmp_path/'another-kit')


def test_A_liveness_step20_real_save_hook_outlives_retirement_cap(tmp_path,monkeypatch):
    from kit import v4_retention as r,v4_timing as timing,v4_campaign_ops as ops
    from kit.sim.v4_merge_standin import export as cpu_export
    run=tmp_path/'runs/D-finqa-alone-s101';(run/'env').mkdir(parents=True)
    for key,value in dict(V4_TELEMETRY='1',V4_MAIN_RETENTION='slot',VERL_FILE_LOGGER_PATH=str(run/'metrics.jsonl'),
                          KIT_V4_ARM='D',RANK='0',STEPS='40').items():monkeypatch.setenv(key,value)
    clock=[0.0];events=[];original=r.sha_file
    monkeypatch.setattr(r,'RETIREMENT_IO_SECONDS',.05)
    monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    def slow_hash(path):
        clock[0]+=1;events.append('hash');return original(path)
    def slow_merge(argv,env):
        clock[0]+=1;events.append('merge');cpu_export(Path(argv[argv.index('--target_dir')+1]))
    def slow_reload(path):
        clock[0]+=1;events.append('reload');return {'tensor_count':1,'dtype':'BF16'}
    monkeypatch.setattr(r,'sha_file',slow_hash);monkeypatch.setattr(ops,'run',slow_merge)
    monkeypatch.setattr(r,'reload_weights',slow_reload)
    class Trainer:
        config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(run/'sdft')))
        def save(self,step):r.before_save(run,'D',step);state(run,'D',step)
    timing.install_telemetry(Trainer,exports='save')
    Trainer().save(20)
    assert {'hash','merge','reload'}<=set(events) and clock[0]>r.RETIREMENT_IO_SECONDS
    assert (run/'env/saved-state-step20.json').exists() and (run/'env/export-reload-step20.json').exists()
    assert list(run.rglob('*.pt')) and (run/'hf-step20/model.safetensors').exists()


def liveness_main_fixture(work):
    from kit import v4_teacher as t
    from kit.sim.v4_merge_standin import prepare_retained_run,recover,evaluation_and_sweep
    ident='D-finqa-alone-s101'
    prepare_retained_run(work,'main',ident);recover(work,'main',ident)
    rows=['train-'+ident,'merge-'+ident]+[f'score-{ident}-{task}-{cap}' for task in ('finqa','general-v1') for cap in (2048,8192)]
    rows += [f'probe-{ident}-finqa-{step}' for step in (0,20,40)]
    for name in rows:
        out=work/'campaign/v4-main'/name/'attempt-1';out.mkdir(parents=True)
        t.atomic_json(out/'start.json',dict(row=name,attempt=1,inputs={}))
        t.atomic_json(out/'verdict.json',dict(row=name,attempt=1,verdict='PASS',inputs={}))
        if name.startswith(('score-','probe-')):evaluation_and_sweep(work,'main',name)
    return work/'runs'/ident


def test_A_liveness_next_slot_real_entry_retires_without_export_reread(tmp_path,monkeypatch):
    from kit import v4_retention as r,v4_campaign_ops as ops,v4_prepare as prep,v4_teacher as t
    run=liveness_main_fixture(tmp_path);clock=[0.0];original=t.local_identity;hashes=[]
    def slow_identity(*args,**kwargs):
        clock[0]+=1;hashes.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(r,'RETIREMENT_IO_SECONDS',.05);monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(t,'local_identity',slow_identity);monkeypatch.setenv('V4_FINQA_CAMPAIGN','1')
    monkeypatch.setattr(prep,'verify',lambda *a:None);monkeypatch.setattr(ops,'scoring_environment_gate',lambda *a:None)
    monkeypatch.setattr(ops,'registered_caps',lambda *a:{})
    launched=[];monkeypatch.setattr(ops,'train',lambda *a:launched.append(a))
    ops.main(['train','--work',str(tmp_path),'--phase','main','--row','train-F-finqa-alone-s101','--slot','F-finqa-alone-s101'])
    assert launched and not hashes and not list(run.rglob('*.pt'))
    assert all((run/f'hf-step{s}/model.safetensors').exists() for s in (20,40))


def test_A_liveness_stuck_retirement_refuses_next_slot_without_deletion(tmp_path,monkeypatch):
    from kit import v4_retention as r,v4_campaign_ops as ops
    run=liveness_main_fixture(tmp_path);before={p:p.read_bytes() for p in run.rglob('*.pt')}
    clock=[0.0];original=r.sha_file
    monkeypatch.setattr(r,'RETIREMENT_IO_SECONDS',.05);monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    def stuck(path):clock[0]+=1;return original(path)
    monkeypatch.setattr(r,'sha_file',stuck);monkeypatch.setenv('V4_FINQA_CAMPAIGN','1')
    from kit import v4_prepare as prep
    monkeypatch.setattr(prep,'verify',lambda *a:None)
    monkeypatch.setattr(ops,'scoring_environment_gate',lambda *a:None)
    monkeypatch.setattr(ops,'registered_caps',lambda *a:{})
    launched=[];monkeypatch.setattr(ops,'train',lambda *a:launched.append(a))
    assert ops.main(['train','--work',str(tmp_path),'--phase','main','--row','train-F-finqa-alone-s101','--slot','F-finqa-alone-s101'])==0
    assert launched and list((tmp_path/'v4/warnings').glob('*.json'))
    assert before and all(p.read_bytes()==raw for p,raw in before.items())
    assert not list(run.glob('env/retention/state-step40.json'))


def test_A_liveness_no_post_return_deadline_failure(monkeypatch):
    from kit import v4_retention as r
    clock=[0.0];monkeypatch.setattr(r,'RETIREMENT_IO_SECONDS',.05)
    monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    @r.bounded_retirement
    def completed():
        r.io_check();clock[0]=1;return 'durably completed'
    assert completed()=='durably completed' and r._deadline.get() is None


def test_A_liveness_export_cache_avoids_hash_and_reload_but_detects_changes(tmp_path,monkeypatch):
    from kit import v4_retention as r,v4_teacher as t
    run=tmp_path/'runs/slot';run.mkdir(parents=True);export(run,20)
    original=t.local_identity;reads=[]
    def count(*args,**kwargs):reads.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(t,'local_identity',count)
    monkeypatch.setattr(r,'reload_weights',lambda *a:pytest.fail('unchanged export was already fully reloaded'))
    r.verify_export(run,20);r.verify_export(run,20)
    assert not reads
    weights=run/'hf-step20/model.safetensors';raw=weights.read_bytes();stamp=weights.stat()
    weights.write_bytes(b'X'*len(raw));os.utime(weights,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    with pytest.raises(ValueError,match='export identity changed'):r.verify_export(run,20)
    assert reads # same-size, restored-mtime mutation still invalidates inode/ctime fingerprint


def test_A_liveness_read_allowance_enters_manifest_and_pricing():
    from kit import v4_retention as r,v4_finqa_campaign as fc
    assert r.RETIREMENT_IO_SECONDS==160
    assert r.RETIREMENT_IO_SECONDS==__import__('math').ceil(r.RETIREMENT_EVIDENCE_READ_BYTES/r.RETIREMENT_READ_BYTES_PER_SECOND)+r.RETIREMENT_METADATA_SECONDS
    for phase in ('qualification','main'):
        graph=fc.build(phase)
        for row in graph['rows']:
            if row['id'].startswith('schedule-') or row['id']=='report':
                assert row['timeout_seconds']==10*((300 if phase=='qualification' else 900)+r.RETIREMENT_IO_SECONDS)
        if phase=='qualification':assert sum(row['timeout_seconds'] for row in graph['rows'])*8/3600>1000 # advisory worst-case sum


@pytest.mark.parametrize('change',['membership','config','cache_symlink'])
def test_A_liveness_export_cache_rechecks_membership_and_receipt_safety(tmp_path,change):
    from kit import v4_retention as r
    run=tmp_path/'runs/slot';run.mkdir(parents=True);export(run,20)
    if change=='membership':(run/'hf-step20/new.json').write_text('{}')
    elif change=='config':(run/'hf-step20/config.json').write_text('{"changed":true}')
    else:
        cache=run/'env/export-files-step20.json';copy=run/'env/cache-copy.json'
        cache.rename(copy);cache.symlink_to(copy)
    with pytest.raises(ValueError,match='export identity changed|linked export hash receipt'):r.verify_export(run,20)


@pytest.mark.parametrize('boundary',['hash','reload'])
def test_A_liveness_export_cache_does_not_bless_mutation_during_initial_verification(tmp_path,monkeypatch,boundary):
    from kit import v4_retention as r,v4_teacher as t
    from kit.sim.v4_merge_standin import export as cpu_export
    run=tmp_path/'runs/slot';run.mkdir(parents=True);cpu_export(run/'hf-step20')
    weights=run/'hf-step20/model.safetensors'
    def mutate():weights.write_bytes(b'CHANGED')
    if boundary=='hash':
        original=t.local_identity
        def hashing(*args,**kwargs):result=original(*args,**kwargs);mutate();return result
        monkeypatch.setattr(t,'local_identity',hashing)
    else:
        def reload(path):mutate();return {'tensor_count':1,'dtype':'BF16'}
        monkeypatch.setattr(r,'reload_weights',reload)
    with pytest.raises(ValueError,match='export changed during hashing|export changed during reload'):r.verify_export(run,20)
    assert not (run/'env/export-reload-step20.json').exists()
