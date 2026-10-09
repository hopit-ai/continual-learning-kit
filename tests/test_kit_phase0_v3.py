"""Phase-0 v3 regressions, including verbatim ports from 13cff0f and 63bc3c6."""
import json,sys
from pathlib import Path
import pytest
from kit.v4_phase0_site import BASE_VERSIONS
VERSIONS={**BASE_VERSIONS,"verl":"0.7.0.dev0","flash_attn":"2.8.3"}

def test_eight_hour_planner_admits_real_relaxed_graph_with_actual_block_spend(tmp_path,monkeypatch):
    """The approved eight-hour graph must fit without reserving 64 hours against prior actual block spend."""
    from kit import v4_phase0 as p,v4_allocation as a,p4_contain as pc
    work=tmp_path/'work'
    for name in ('v4/report-inputs/prepare-receipt-phase0.json','v4/report-phase0/presend/containment-presend.json'):
        path=work/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    begin='2026-10-08T00:00:00Z';end='2026-10-08T06:15:00Z'
    ledger={'allocations':{'122':{'start':begin,'end':end,'width':8,'segments':[{'start':begin,'end':end,'block':'qualification'}]}}}
    pc.wd.write_durably(work/pc.DIRECTORY/'allocation-ledger.json',ledger)
    monkeypatch.setattr(p,'verify_prepare',lambda work,**kw:{'verification_seconds':84,'verification_allowance_seconds':126,'environment_check':{'deadline_seconds':600}})
    monkeypatch.setattr(p,'frozen_launch_environment',lambda work:{'WORK':str(work)})
    monkeypatch.setattr(p.site,'planning_storage',lambda *args:{})
    doc=a.plan(work,'phase0','phase0',480,10080)
    assert doc['sbatch_time']=='08:00:00' and doc['reserved_gpu_hours']==64
    assert doc['phase0_allocation_cap_gpu_hours']==64 and doc['remaining_gpu_hours']==50
    assert doc['required_seconds']==27816 and doc['reservation_slack_seconds']==984
    assert doc['block_limit']==100 and '#SBATCH --time=08:00:00' in a.header(doc,work)
    assert '#SBATCH --no-requeue' in a.header(doc,work)
    doc['prior_allocations']={}
    # A consumed phase-0 allocation still cannot be replaced, even after a cheap failure.
    ledger['allocations']['122']['phase0']=True
    pc.wd.write_durably(work/pc.DIRECTORY/'allocation-ledger.json',ledger)
    with pytest.raises(ValueError,match='exactly one allocation'):a.plan(work,'phase0','phase0',480,10080)


def test_phase0_live_block_gate_uses_actual_elapsed_charge(tmp_path,monkeypatch):
    """A phase-0 job must charge elapsed time, not refuse its unused eight-hour reservation against the block."""
    import hashlib
    from kit import p4_contain as pc
    work=tmp_path/'work';now=[pc.epoch('2026-10-09T12:00:00Z')]
    monkeypatch.setattr(pc.time,'time',lambda:now[0])
    prior_start='2026-10-08T00:00:00Z';prior_end='2026-10-08T06:15:00Z'
    ledger={'allocations':{'122':{'job_id':'122','state':'COMPLETED','start':prior_start,'end':prior_end,'width':8,'segments':[{'start':prior_start,'end':prior_end,'block':'qualification'}]}},'block_limits':{'qualification':100},'ceiling':560}
    pc.wd.write_durably(work/pc.DIRECTORY/'allocation-ledger.json',ledger)
    plan=work/'phase0.json';plan.write_text(json.dumps({'phase':'phase0','phase0_allocation_cap_gpu_hours':64}))
    monkeypatch.setenv('V4_ALLOCATION_PLAN',str(plan));monkeypatch.setenv('V4_ALLOCATION_PLAN_SHA256',hashlib.sha256(plan.read_bytes()).hexdigest())
    start=pc.wd.precise_text(pc.wd.from_epoch(now[0]-60))
    live={'job_id':'123','allocation_start':start,'allocation_time_limit':'08:00:00','gpu_count':8,'allocation_state':'RUNNING'}
    accepted=pc.allocation_budget(work,live,600,'qualification',100,560)
    assert accepted['blocks']['qualification']==pytest.approx(50+8/60)
    assert accepted['allocations']['123']['phase0'] is True
    now[0]+=7*3600
    with pytest.raises(pc.Refused,match='budget refused'):pc.allocation_budget(work,live,600,'qualification',100,560)

def test_partner_torch_metadata_without_suffix_records_cpu_runtime(tmp_path,monkeypatch):
    """The reviewed partner admission/allowance regression must hold in the public kit."""
    from types import SimpleNamespace
    from kit import v4_phase0_site as site,runner
    # Exactly the micromamba metadata/module identities reported by the partner.
    versions={**VERSIONS,'torch':'2.9.0'}
    monkeypatch.setattr(site,'installed_versions',lambda:versions.copy())
    monkeypatch.setattr(site,'distribution_inventory',lambda **kw:[])
    class NoCUDA:
        def __getattr__(self,key):pytest.fail('CPU identity probe must not call CUDA: '+key)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(__version__='2.9.0+cu128',version=SimpleNamespace(cuda='12.8'),cuda=NoCUDA()))
    def command(argv,**kw):
        kw['log'].write_text('No broken requirements found.\n')
        return {'returncode':0,'failure_type':None,'seconds':.1}
    monkeypatch.setattr(runner,'bounded_command',command)
    doc=site.base_environment(tmp_path/'work')
    assert doc['versions']['torch']=='2.9.0'
    assert doc['torch_runtime']=={'version':'2.9.0+cu128','cuda':'12.8'}
    assert json.loads((tmp_path/'work/v4/report-phase0/environment-build/base-environment.json').read_text())==doc

@pytest.mark.parametrize('metadata,runtime,cuda,accepted',[
    ('2.9.0','2.9.0+cu128','12.8',True),('2.9.0','2.9.0','12.8',True),
    ('2.9.0+cu128','2.9.0+cu128','12.8',True),('2.9.1','2.9.0+cu128','12.8',False),
    ('2.8.0','2.8.0+cu128','12.8',False),('2.9.0','2.9.0','12.7',False),
    ('2.9.0','2.10.0+cu128','12.8',False),('2.9.0rc1','2.9.0+cu128','12.8',False)])
def test_partner_torch_public_and_cpu_runtime_admission(metadata,runtime,cuda,accepted):
    """The reviewed partner admission/allowance regression must hold in the public kit."""
    from kit import v4_phase0_site as s
    call=lambda:s.validate_base_versions({**VERSIONS,'torch':metadata},{'version':runtime,'cuda':cuda})
    if accepted:assert call()['torch']==metadata
    else:
        with pytest.raises(ValueError,match='torch'):call()

@pytest.mark.parametrize('package',['vllm','ray','numpy','transformers'])
def test_partner_other_public_pins_stay_exact(package):
    """The reviewed partner admission/allowance regression must hold in the public kit."""
    from kit import v4_phase0_site as s
    assert s.validate_base_versions(VERSIONS)==VERSIONS
    with pytest.raises(ValueError,match=package):s.validate_base_versions({**VERSIONS,package:VERSIONS[package]+'+unreviewed'})
    assert s.validate_base_versions({**VERSIONS,'flash_attn':'2.8.3+cu12torch2.9'})

@pytest.mark.parametrize('metadata,runtime,cuda',[
    ('2.9.0+cpu','2.9.0+cu128','12.8'),
    ('2.9.0+cu126','2.9.0+cu128','12.8'),
    ('2.9.0','2.9.0+cu128','12.6'),
    ('2.9.0+cu128','2.9.0+cu128','12.6')])
def test_torch_local_suffix_and_cuda_identity_must_be_consistent(metadata,runtime,cuda):
    """The reviewed partner admission/allowance regression must hold in the public kit."""
    from kit.v4_phase0_site import validate_base_versions
    with pytest.raises(ValueError,match='torch'):
        validate_base_versions({**VERSIONS,'torch':metadata},{'version':runtime,'cuda':cuda})

@pytest.mark.parametrize('measured',[True,-1,None,float('nan'),float('inf')])
def test_partner_verification_allowance_refuses_invalid_measurements(measured):
    """The reviewed partner admission/allowance regression must hold in the public kit."""
    from kit.v4_prepare import verification_allowance
    with pytest.raises(ValueError,match='measurement'):verification_allowance(measured)

@pytest.mark.parametrize('devices,leak,expected',[(1,False,True),(1,True,False),(8,False,False)])
def test_real_selftest_client_loss_cancellation(tmp_path,monkeypatch,devices,leak,expected):
    """Real self-test ordering must exercise CUDA before a six-second client-loss cancellation."""
    from kit import p4_contain as pc
    from types import SimpleNamespace
    work=tmp_path/'work';out=work/pc.DIRECTORY;out.mkdir(parents=True)
    folder=out/'selftest-processes';folder.mkdir()
    clock=[1000.];lost=[None];receipt_written=[False]
    kinds=['double-fork-setsid','double-fork-setsid-parent','cleared-exec','ignores-term','orphan-grandchild','orphan-grandchild-parent','later-gpu','launcher']
    for i,k in enumerate(kinds):(folder/(k+'.pid')).write_text(str(9000+i))
    real_read=pc.wd.read_json
    def advance(seconds):
        clock[0]+=seconds
        # Cold import + context startup takes 7.2 s. The old ordering cancels
        # at t=6 before it can publish a receipt. New ordering waits, then kills.
        if not receipt_written[0] and clock[0]>=1007.2 and (lost[0] is None or clock[0]<lost[0]+6):
            (folder/'gpu.json').write_text(json.dumps({'ok':devices==1,'accessible_cuda_devices':devices,'why':'one-GPU step exposes '+str(devices)+' CUDA devices'}));receipt_written[0]=True
    monkeypatch.setattr(pc.time,'time',lambda:clock[0]);monkeypatch.setattr(pc.time,'sleep',advance)
    monkeypatch.setenv('SLURM_JOB_ID','123')
    receipt={'expected':pc.EXPECTED,'job_id':'123','content_sha256':'a'*64}
    monkeypatch.setattr(pc,'allocation_observation',lambda job:{'job_id':job,'width':8,'start':pc.wd.precise_text(pc.wd.from_epoch(999)),'time_limit_seconds':28800})
    monkeypatch.setattr(pc,'allocation_budget',lambda *a:{'allocations':{'123':{}}})
    monkeypatch.setattr(pc,'cpu_preflight',lambda:{'ok':True})
    monkeypatch.setattr(pc,'check',lambda *a:None)
    monkeypatch.setattr(pc,'verify_receipts',lambda *a,**kw:([],{'receipt':receipt}))
    monkeypatch.setattr(pc,'reviewed_receipt',lambda work:receipt)
    monkeypatch.setattr(pc,'_live_cluster',lambda doc:{'job_id':'123'})
    monkeypatch.setattr(pc,'cpu_probe',lambda *a:{'ok':True})
    slurm={'job_id':'123','step_id':'1','scheduler_start':pc.wd.precise_text(pc.wd.from_epoch(1000)), 'deadline':pc.wd.precise_text(pc.wd.from_epoch(1600)), 'expected_gpu_uuids':'0','J':60,'W':40,'kill_by':pc.wd.precise_text(pc.wd.from_epoch(1160))}
    def popen(*a,**kw):
        pc.wd.write_durably(folder/'wrapper.json',{'ready':True,'srun_pid':8001,'watchdog_pid':8002,'slurm':slurm})
        return SimpleNamespace(pid=8000,poll=lambda:None,wait=lambda **kw:0,kill=lambda:None)
    monkeypatch.setattr(pc.subprocess,'Popen',popen)
    def kill(pid,sig):
        if pid==8001 and lost[0] is None:
            assert not (folder/'clients-lost.json').exists(),'late touch must not be released while enforcing clients still live'
            lost[0]=clock[0]
    monkeypatch.setattr(pc.os,'kill',kill)
    monkeypatch.setattr(pc,'pid_alive',lambda pid: (leak and pid==9000) or lost[0] is None or clock[0]<lost[0]+6)
    monkeypatch.setattr(pc,'steps',lambda *a:([{'StepId':'123.1','State':'CANCELLED'}],{}))
    def evidence(*a):
        return {'verified':not leak,'verified_at':pc.wd.precise_text(pc.wd.from_epoch(clock[0])),'scheduler_end':pc.wd.precise_text(pc.wd.from_epoch(lost[0]+6)), 'state':'CANCELLED','failure_type':None,'gpu_idle':not leak}
    monkeypatch.setattr(pc,'evidence',evidence);monkeypatch.setattr(pc,'cancel',lambda *a:None)
    result=pc._selftest(work,out,5,None,'qualification',100,560)
    assert result['ok'] is expected,result['problems']
    if expected:
        assert result['gpu_acquisition']['accessible_cuda_devices']==1
        assert lost[0]>=1007.2
        assert result['termination_mode']=='client_loss_cancellation'
        assert result['late_gpu_acquisition'] is None # early cancellation may prevent a second receipt
        assert len(result['deaths'])==8
    elif devices==8:assert any('8 CUDA devices' in p for p in result['problems'])
    else:assert any('not every adversarial pid' in p for p in result['problems'])


def test_relaxed_graph_cpu_refusal_and_timing_measurements(tmp_path,monkeypatch):
    """Unmeasured short gates must relax and fit the complete graph inside the approved eight hours."""
    from kit import v4_phase0 as p,v4_allocation as a
    from kit.v4_phase0_environment import admitted_environment_timing
    from kit.v4_prepare import verification_allowance
    assert admitted_environment_timing({'trainer':1,'inference':400})['deadline_seconds']>=2000
    assert verification_allowance(84)==126
    graph=p.campaign()
    for row in graph['rows']:
        if row['id'] not in ('verify-prepare','containment-selftest'):
            cap=int(row['command'][row['command'].index('--time-cap')+1]) if '--time-cap' in row['command'] else row['allocation_cpu_cap_seconds']
            assert cap>=600
    work=tmp_path/'work';monkeypatch.setattr(p,'verify_presend',lambda work:{})
    monkeypatch.setattr(p,'verify_prepare',lambda work,**kw:{'verification_seconds':84,'verification_allowance_seconds':126,'environment_check':{'deadline_seconds':600}})
    with pytest.raises(ValueError,match='relaxed graph.*eight-hour.*64 GPU-hour'):
        a.plan(work,'phase0','phase0',480,400)
    assert p.required_seconds(600,126)<28800


def test_partner_rehash_measurement_plan_binding_and_real_runner(tmp_path,monkeypatch):
    """An 84-second real rehash must register 126 seconds and reach the real row runner with that cap."""
    # Adapted from 13cff0f, now exercising the actual approved relaxed graph.
    from kit import v4_phase0 as p,v4_allocation as a,runner,v4_prepare as prep
    import hashlib
    work=tmp_path/'work';inputs=work/'v4/report-inputs';inputs.mkdir(parents=True)
    source=work/'model-bytes';source.write_bytes(b'actual prepared bytes')
    receipt=inputs/'prepare-receipt-phase0.json'
    from kit.v4_phase0_environment import admitted_environment_timing
    receipt.write_text(json.dumps({'phase':'phase0','before_allocation':True,'environment_check':admitted_environment_timing({'trainer':1,'inference':1}),**prep.file_receipt([source])}))
    presend=work/'v4/report-phase0/presend/containment-presend.json';presend.parent.mkdir(parents=True);presend.write_text('{}')
    monkeypatch.setattr(p,'verify_presend',lambda work:{})
    with monkeypatch.context() as timer:
        ticks=iter((100.,184.));timer.setattr(p.time,'monotonic',lambda:next(ticks))
        assert p.measure_prepare_verification(work)==84
    frozen=json.loads(receipt.read_text());assert frozen['verification_allowance_seconds']==126
    monkeypatch.setattr(p,'frozen_launch_environment',lambda work:{'WORK':str(work)})
    monkeypatch.setattr(p.site,'planning_storage',lambda *args:{})
    doc=a.plan(work,'phase0','phase0',480,10080)
    assert doc['verification_allowance_seconds']==126 and doc['required_seconds']==27816
    plan=work/'allocation.json';plan.write_text(json.dumps(doc,indent=2)+'\n')
    monkeypatch.setenv('V4_ALLOCATION_PLAN',str(plan));monkeypatch.setenv('V4_ALLOCATION_PLAN_SHA256',hashlib.sha256(plan.read_bytes()).hexdigest())
    monkeypatch.setenv('WORK',str(work));monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    path=work/'campaign.json';path.write_text(json.dumps(p.campaign()));campaign=runner.load_campaign(path)
    row=next(row for row in campaign['rows'] if row['id']=='verify-prepare').copy();row.update(needs=[],pilot=False,bars=[])
    observed=[]
    def execute(argv,**kw):
        observed.append((kw['timeout'],kw['env']['V4_COMMAND_TIMEOUT']))
        p.verify_prepare(work);kw['log'].parent.mkdir(parents=True,exist_ok=True);kw['log'].write_text('real immutable verifier passed\n')
        return {'returncode':0,'failure_type':None,'seconds':84}
    monkeypatch.setattr(runner,'bounded_command',execute)
    assert runner.run_row(campaign,row,None)==0 and observed==[(126,'126')]
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='prepared input changed'):p.verify_prepare(work)
    doc['verification_allowance_seconds']=60;plan.write_text(json.dumps(doc))
    monkeypatch.setenv('V4_ALLOCATION_PLAN_SHA256',hashlib.sha256(plan.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='allowance differs'):p.verification_row_cap(work)


def test_real_probe_program_publishes_before_sleep_with_warm_import(tmp_path):
    """The executed detached probe must publish its early receipt before the five-second late delay."""
    import ast,os,subprocess,time
    from kit.p4_contain import PAYLOAD
    tree=ast.parse(PAYLOAD)
    code=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='late_code' for t in n.targets)))
    state=tmp_path/'slurm';state.mkdir();(state/'config.json').write_text(json.dumps({'GPUImportDelay':2.2}))
    folder=tmp_path/'probe';folder.mkdir();owners=tmp_path/'owners';owners.mkdir()
    command=[sys.executable,'-c',code,str(folder),'5','5',str(owners),'yes',str(state),'0']
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={})
    try:
        limit=time.monotonic()+4.5
        while not (folder/'gpu.json').exists() and proc.poll() is None and time.monotonic()<limit:time.sleep(.02)
        assert (folder/'gpu.json').exists(),'old ordering delays receipt until >7.2 s; client-loss cancellation at 6 s precludes it'
        early=json.loads((folder/'gpu.json').read_text());assert early['ok'] and early['accessible_cuda_devices']==1
        assert not (folder/'gpu-late.json').exists()
        (folder/'clients-lost.json').write_text('{}')
        # Simulate rapid scheduler cancellation: it is valid without a late receipt.
        proc.kill();proc.wait(timeout=5)
        assert not (folder/'gpu-late.json').exists()
    finally:
        if proc.poll() is None:proc.kill();proc.wait(timeout=5)
        proc.communicate()


@pytest.mark.parametrize('estimate,expected',[(None,600),(0,600),(30,600),(120,600),(300,1500),(400,2000)])
def test_unmeasured_patience_policy(estimate,expected):
    """Unmeasured limits must never retain a sub-ten-minute timeout or ignore an available estimate."""
    from kit.v4_phase0_timing import relaxed_timeout
    assert relaxed_timeout(estimate)==expected


def test_reconciliation_default_has_no_short_expectation_gate():
    """Login-node reconciliation must not retain a 60-second timing-only maximum."""
    import inspect
    from kit.p4_contain import reconcile
    assert inspect.signature(reconcile).parameters['seconds'].default==600


def test_reader_admits_real_graph_and_rejects_shortened_reservation(tmp_path):
    """The new graph must fit the approved archive plan and still refuse an undersized reservation."""
    from kit.v4_phase0 import verification_plan
    from kit.v4_phase0_environment import admitted_environment_timing
    receipt={'verification_seconds':84,'verification_allowance_seconds':126,'environment_check':admitted_environment_timing({'trainer':1,'inference':1})}
    doc={'sbatch_time':'08:00:00','verification_seconds':84,'verification_allowance_seconds':126,'environment_check_deadline_seconds':600}
    assert verification_plan(doc,receipt)==126
    doc['sbatch_time']='07:00:00'
    with pytest.raises(ValueError,match='cannot fit fixed eight-hour'):verification_plan(doc,receipt)


def test_standin_slurm_cancels_six_seconds_after_client_loss(tmp_path,monkeypatch):
    """The shipped Slurm stand-in must cancel and clear detached members after client loss, not wait for TIMEOUT."""
    import importlib.util,signal
    from types import SimpleNamespace
    monkeypatch.setenv('SIM_SLURM_STATE',str(tmp_path));monkeypatch.setenv('SIM_SLURM_SECONDS_PER_MINUTE','60')
    path=Path(__file__).resolve().parents[1]/'kit/sim/bin/slurm_sim.py'
    spec=importlib.util.spec_from_file_location('phase0_v3_scheduler',path);sim=importlib.util.module_from_spec(spec);spec.loader.exec_module(sim)
    (tmp_path/'config.json').write_text(json.dumps({'CancelOnClientLossSeconds':6,'GPUImportDelay':7.2}))
    sid='123.cancellation';sim.write({'StepId':sid,'State':'PENDING','TimeLimit':1,'GPUCount':1,'ClientPid':42000})
    clock=[1000.];killed=set();sent=[]
    monkeypatch.setattr(sim.time,'monotonic',lambda:clock[0]);monkeypatch.setattr(sim.time,'time',lambda:clock[0]);monkeypatch.setattr(sim.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(sim.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=41001,returncode=None,poll=lambda:None))
    def tree():
        found={p:(41001 if p==41002 and clock[0]==1000 else 1,str(p)) for p in (41001,41002) if p not in killed}
        if clock[0]<1007.2:found[42000]=(1,'client')
        return found
    monkeypatch.setattr(sim,'tree',tree)
    def kill(pid,sig):
        sent.append((clock[0],pid,sig))
        if sig==signal.SIGKILL:killed.add(pid)
    monkeypatch.setattr(sim.os,'kill',kill)
    sim.timer(sid,['ignored']);doc=sim.read(sid)
    assert doc['State']=='CANCELLED' and doc['ClientLossCancellation']
    assert 1013.2<=doc['EndTime']<=1013.3
    assert killed=={41001,41002}
