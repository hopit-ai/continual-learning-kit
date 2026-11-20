"""Offline CR review regressions: real wrapper flow, CPU children and receipts."""
from contextlib import nullcontext
from pathlib import Path
import os
import signal
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from kit import nm_budget as budget, nm_jobs, cr_jobs, cr_confirmation as cr, cr_process
from kit.nm_common import KIT, Invalid, atomic, digest, launch_config, load, sha, source_seal


def attempt(root, **updates):
    path = root/'attempts/row/nonce/receipt.json'
    value = dict(task='row', nonce='nonce', family='cr', phase='train', gpus=8,
        slurm_job_id='CPU-HOLDER', allocation_started_at=1, started=100, ended=3700,
        status='COMPLETE', wrapper_completed=True, teardown_confirmed=True,
        monotonic_elapsed_seconds=3600, reserved_gpu_hours=9)
    value.update(updates);atomic(path, value)
    budget.complete_binding(path, value)
    atomic(root/'state/row.json', dict(receipt=str(path.relative_to(root)), sha256=sha(path)))
    return path, value


def holder(*args):
    return dict(allocation_started_at=1, allocation_ended_at=7300, allocated_gpus=8)


@pytest.mark.parametrize('damage', ['pointer', 'receipt', 'completion', 'identity'])
def test_tampering_rejected_without_rewriting_any_binding(tmp_path, damage):
    path, value = attempt(tmp_path)
    if damage == 'pointer': atomic(tmp_path/'state/row.json', dict(receipt=str(path.relative_to(tmp_path)), sha256='0'*64))
    if damage == 'receipt': value['ended'] = 9999;atomic(path,value)
    if damage == 'completion': atomic(path.with_name('completion.json'), dict(task='row', nonce='another', sha256=sha(path)))
    if damage == 'identity':
        value['nonce']='another';atomic(path,value)
        budget.complete_binding(path,value)
        atomic(tmp_path/'state/row.json',dict(receipt=str(path.relative_to(tmp_path)),sha256=sha(path)))
    before = {p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    with pytest.raises(Invalid): budget.reconcile(tmp_path, family='cr', query=holder)
    assert all(p.read_bytes() == data for p,data in before.items())


@pytest.mark.parametrize('missing', ['no-end', 'no-completion', 'unconfirmed'])
def test_incomplete_keeps_reservation_until_scheduler_end_then_upper_bound(tmp_path, missing):
    changes = {'running-end':dict(status='RUNNING'), 'no-end':dict(ended=None,wrapper_completed=False),
               'no-completion':{}, 'unconfirmed':dict(teardown_confirmed=False)}[missing]
    path, value = attempt(tmp_path, **changes)
    if missing == 'no-completion': path.with_name('completion.json').unlink()
    assert budget.reconcile(tmp_path,family='cr',query=lambda *a:{})['status']=='PENDING'
    assert load(path)['reserved_gpu_hours']==9
    result=budget.reconcile(tmp_path,family='cr',query=holder)
    final=load(path)
    assert final['cost_status']=='COMPLETE-UPPER-BOUND' and final['status']!='RUNNING'
    assert final['allocation_gpu_hours']==16 and final['reserved_gpu_hours']==0
    assert result['status']=='COMPLETE' and result['gpu_hours']==16
    assert load(tmp_path/'state/row.json')['sha256']==sha(path)
    assert budget.reconcile(tmp_path,family='cr',query=holder)==result


def test_running_end_inconsistency_is_rejected_even_with_matching_hashes(tmp_path):
    attempt(tmp_path,status='RUNNING')
    with pytest.raises(Invalid,match='RUNNING receipt has an end'):
        budget.reconcile(tmp_path,family='cr',query=holder)


def test_killed_wrapper_with_only_running_receipt_is_recovered(tmp_path):
    path, value=attempt(tmp_path,status='RUNNING',ended=None,wrapper_completed=False)
    path.with_name('completion.json').unlink()
    result=budget.reconcile(tmp_path,family='cr',query=holder)
    assert result['gpu_hours']==16 and result['rows'][0]['cost_status']=='COMPLETE-UPPER-BOUND'


def test_holder_life_rows_and_idle_difference_are_reported_warn_only(tmp_path):
    path, value=attempt(tmp_path,monotonic_elapsed_seconds=10800)
    result=budget.reconcile(tmp_path,family='cr',query=holder)
    assert result['gpu_hours']==24 and result['status']=='COMPLETE'
    assert result['holder_gpu_hours']==pytest.approx(8*7299/3600)
    assert result['holder_row_gpu_hours']==24
    assert result['holder_unattributed_gpu_hours']==pytest.approx(8*7299/3600-24)
    assert 'WARN CR rows exceed scheduler holder GPU-hours' in result['warnings']
    assert result['warnings']==sorted(result['warnings'])


@pytest.mark.parametrize('jump', [-10000,10000])
@pytest.mark.parametrize('interrupted', [False,True])
def test_real_eight_gpu_cr_wrapper_with_stubbed_gpu_run(tmp_path,monkeypatch,jump,interrupted):
    from kit import cr_process, m3_manifest, v6_infra
    row=next(r for r in cr_jobs.inventory() if r['gpus']==8)
    task=row['id'];out=tmp_path/'out'
    cfg=dict(out=str(out),job_python=sys.executable,bind_python='fixture-training-only',
        infrastructure_update='a'*64,send_approval='CPU fixture approval',
        budget=dict(branch='CR_confirmation',approved_gpu_hours=830,approval_reference='CPU fixture',
                    row_max_seconds={task:7200}))
    config=tmp_path/'launch.json';atomic(config,cfg)
    atomic(out/'launch/cr.json',dict(config_sha256=digest(launch_config(cfg,'cr')),code=source_seal(),cr_code=cr.code_seal()))
    monkeypatch.setattr(cr_jobs,'settings',lambda v:v)
    monkeypatch.setattr(v6_infra,'verify',lambda v:None)
    monkeypatch.setattr(nm_jobs,'current',lambda *a,**k:None)
    monkeypatch.setattr(m3_manifest,'allocated_gpus',lambda n:list(range(n)))
    monkeypatch.setattr(m3_manifest,'allocation_reservation',lambda ids:nullcontext(()))
    monkeypatch.setenv('SLURM_JOB_ID','CPU-HOLDER');monkeypatch.setenv('SLURM_JOB_START_TIME','1')
    monkeypatch.delenv('KIT_V6_AUTHORITY_SHA',raising=False)
    now={'utc':100.,'mono':0.}
    monkeypatch.setattr(nm_jobs.time,'time',lambda:now['utc'])
    monkeypatch.setattr(nm_jobs.time,'monotonic',lambda:now['mono'])
    def gpu_run(command,*,env,pass_fds,evidence):
        assert command[0]==sys.executable and env['INFERENCE_PY']==sys.executable
        now.update(utc=100+jump,mono=3600.)
        evidence['teardown_confirmed']=True
        if interrupted:
            previous=signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(SystemExit(143)))
            try: os.kill(os.getpid(),signal.SIGTERM)
            finally: signal.signal(signal.SIGTERM,previous)
        return NS(returncode=0)
    monkeypatch.setattr(cr_process,'gpu_run',gpu_run)
    assert nm_jobs.run(config,task)==(75 if interrupted else 0)
    pointer=load(out/'state'/(task+'.json'));path=out/pointer['receipt'];value=load(path)
    assert pointer['sha256']==sha(path)
    assert value['ended']==100+jump and value['monotonic_elapsed_seconds']==3600
    assert value['allocation_gpu_hours']==8 and value['reserved_gpu_hours']==0
    assert value['teardown_confirmed'] and value['wrapper_completed']
    assert budget.reconcile(out,family='cr',query=holder)['gpu_hours']==8


def test_term_resistant_descendant_in_own_session_is_killed_before_completion(tmp_path,monkeypatch):
    from kit.a4_common import job_deadline
    from kit.cr_process import gpu_run
    from kit.cr_process import processes
    marker=tmp_path/'child.pid';evidence={}
    readfd,writefd=os.pipe()
    try: processes()
    except PermissionError:
        # This macOS sandbox forbids ps. Use only identities explicitly written
        # by this fixture, still exercising real sessions, TERM, KILL and EOF.
        def fixture_processes():
            if not marker.is_file(): return {}
            parent,child=map(int,marker.read_text().split(','));table={}
            for pid,ppid in ((parent,os.getpid()),(child,parent)):
                try:os.kill(pid,0)
                except ProcessLookupError:continue
                table[pid]=(ppid,'S')
            return table
        monkeypatch.setattr('kit.cr_process.processes',fixture_processes)
    worker="import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
    launcher=("import subprocess,sys,time;from pathlib import Path;"
              f"p=subprocess.Popen([sys.executable,'-c',{worker!r}],start_new_session=True,close_fds=False);"
              f"Path({str(marker)!r}).write_text(str(__import__('os').getpid())+','+str(p.pid));time.sleep(60)")
    try:
        with pytest.raises(Exception,match='hard duration limit'):
            with job_deadline(1): gpu_run([sys.executable,'-c',launcher],evidence=evidence,pass_fds=(writefd,))
        assert marker.is_file()
        os.close(writefd);writefd=None
        import select
        assert select.select([readfd],[],[],2)[0] and os.read(readfd,1)==b'', evidence
        assert evidence['teardown_confirmed']==sys.platform.startswith('linux')
    finally:
        os.close(readfd)
        if writefd is not None:os.close(writefd)
        if marker.is_file():
            for pid in map(int,marker.read_text().split(',')):
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass


def test_public_baseline_and_receipt_cannot_restore_historical_private_digest_sets():
    import re
    resource=KIT/'resources/cr-confirmation'
    assert not (resource/'v7c-baseline.json').exists()
    for name in ('v7i-baseline.json','diff-receipt-v7i1.json'):
        # Inspect decoded nested strings as well as the JSON byte representation.
        text=(resource/name).read_text().replace('\\n','\n')
        assert 'PRIVATE_TOKEN_SHA256' not in text
        assert not re.search(r'frozenset\s*\([^)]*[0-9a-f]{64}',text)
    baseline=load(resource/'v7i-baseline.json');doc=load(resource/'diff-receipt-v7i1.json')
    assert doc['inherited']==baseline['inherited']
    assert all('patch' not in v and 'text' not in v for v in doc['inherited'])
    assert {'inherited-public-tag','test-redaction-cherry-pick'}=={v['category'] for v in doc['inherited']}
    assert sha(resource/'diff-receipt-v7c.json')==cr.V7C_RECEIPT_SHA256


def test_accounting_warn_and_holder_idle_reach_essential_and_full_pasted_returns(tmp_path):
    from kit.text_return import write_parts,parse_parts
    from kit.cr_text import payload
    path,value=attempt(tmp_path/'account',monotonic_elapsed_seconds=896*3600/8)
    account=budget.reconcile(tmp_path/'account',family='cr',query=lambda *a:dict(
        allocation_started_at=1,allocation_ended_at=896*3600/8+7300,allocated_gpus=8))
    full=dict(nm_family='cr',infrastructure_authority='a'*64,
              reading=dict(status='PENDING/INCOMPLETE'),accounting=account)
    assert budget.CR_WARN in account['warnings']
    rows=payload(full)
    assert 'W|NM-cr|WARN_CR_830_GPU_HOURS_WARN_ONLY|1' in rows
    assert any(r.startswith('T|NM|account|holder_unattributed_gpu_hours|') for r in rows)
    paths=write_parts(tmp_path/'return',full,'kit-message-v7i1','b'*40)
    for parts in (paths,sorted((tmp_path/'return/full').glob('RETURN-TEXT-*.md'))):
        texts=[p.read_text() for p in parts]
        assert texts and all(len(line)<90 for text in texts for line in text.splitlines())
        parsed=parse_parts(texts)
        assert 'WARN_CR_830_GPU_HOURS_WARN_ONLY' in parsed['utilisation_warnings']
        assert parsed['metadata']['account']['holder_unattributed_gpu_hours']==account['holder_unattributed_gpu_hours']


def test_holder_gpu_count_is_from_scheduler_allocation_tres(monkeypatch):
    from datetime import datetime
    from kit import a4_common
    monkeypatch.setattr(a4_common,'_slurm_times',lambda *a:dict(allocation_started_at=100,allocation_ended_at=3700))
    def scheduler(command,**kwargs):
        assert command[-1]=='JobIDRaw,Start,End,State,AllocTRES' and '-X' in command
        # A step and a differently-started requeue must not supply holder GPUs.
        text='CPU-HOLDER.step|'+datetime.fromtimestamp(100).isoformat()+'|Unknown|RUNNING|gres/gpu=99\n'
        text+='CPU-HOLDER|'+datetime.fromtimestamp(50).isoformat()+'|Unknown|RUNNING|gres/gpu=99\n'
        text+='CPU-HOLDER|'+datetime.fromtimestamp(100).isoformat()+'|'+datetime.fromtimestamp(3700).isoformat()+'|COMPLETED|cpu=32,gres/gpu=8\n'
        return NS(stdout=text)
    monkeypatch.setattr(subprocess,'run',scheduler)
    assert budget.cr_holder_times('CPU-HOLDER',100)==dict(
        allocation_started_at=100,allocation_ended_at=3700,allocated_gpus=8)


def test_crash_between_terminal_receipt_and_completion_keeps_reservation(tmp_path):
    path,value=attempt(tmp_path,allocation_gpu_hours=8,reserved_gpu_hours=0,
                       row_reservation_gpu_hours=20,cost_status='COMPLETE')
    path.with_name('completion.json').unlink()
    result=budget.reconcile(tmp_path,family='cr',query=lambda *a:{})
    assert result['status']=='PENDING' and result['gpu_hours']==0
    assert load(path)['reserved_gpu_hours']==20 and load(path)['allocation_gpu_hours'] is None
    assert not path.with_name('completion.json').exists()
    assert load(tmp_path/'state/row.json')['sha256']==sha(path)
    result=budget.reconcile(tmp_path,family='cr',query=holder)
    assert result['gpu_hours']==16
    assert load(path)['cost_status']=='COMPLETE-UPPER-BOUND'


def test_reserve_does_not_trust_unbound_zero_reservation(tmp_path,monkeypatch):
    path,value=attempt(tmp_path,allocation_gpu_hours=0,reserved_gpu_hours=0,
                       row_reservation_gpu_hours=831,cost_status='COMPLETE')
    path.with_name('completion.json').unlink()
    new=tmp_path/'attempts/next/nonce';new.mkdir(parents=True)
    record=dict(task='next',nonce='nonce',family='cr',slurm_job_id='CPU-HOLDER',started=100)
    cfg=dict(budget=dict(branch='CR_confirmation',approved_gpu_hours=830,
        approval_reference='CPU fixture',row_max_seconds={'next':60}))
    monkeypatch.setenv('SLURM_JOB_START_TIME','1');monkeypatch.setattr(budget.time,'time',lambda:100)
    budget.reserve(tmp_path,cfg,dict(id='next',family='cr',gpus=8),record,new)
    assert budget.CR_WARN in record['budget_warnings']


def test_reservation_sums_existing_upper_bound_without_downgrading_to_monotonic(tmp_path,monkeypatch):
    path,value=attempt(tmp_path,allocation_gpu_hours=831,reserved_gpu_hours=0,
        allocation_ended_at=100+831*3600/8,cost_status='COMPLETE-UPPER-BOUND')
    new=tmp_path/'attempts/next/nonce';new.mkdir(parents=True)
    record=dict(task='next',nonce='nonce',family='cr',slurm_job_id='CPU-HOLDER',started=100)
    cfg=dict(budget=dict(branch='CR_confirmation',approved_gpu_hours=830,
        approval_reference='CPU fixture',row_max_seconds={'next':60}))
    monkeypatch.setenv('SLURM_JOB_START_TIME','1');monkeypatch.setattr(budget.time,'time',lambda:100)
    budget.reserve(tmp_path,cfg,dict(id='next',family='cr',gpus=8),record,new)
    assert budget.CR_WARN in record['budget_warnings']
    assert load(path)['allocation_gpu_hours']==831


def test_concurrent_row_setup_does_not_block_another_cr_reservation(tmp_path,monkeypatch):
    row=next(r for r in cr_jobs.inventory() if r['gpus']==8)
    path=tmp_path/'attempts'/row['id']/'nonce'/'receipt.json'
    # Model the window between initial wrapper receipt and budget-lock entry.
    atomic(path,dict(task=row['id'],nonce='nonce',family='cr',gpus=8,
        slurm_job_id='CPU-HOLDER',allocation_started_at=1,started=100,ended=None,status='RUNNING'))
    new=tmp_path/'attempts/next/nonce';new.mkdir(parents=True)
    record=dict(task='next',nonce='nonce',family='cr',slurm_job_id='CPU-HOLDER',started=100)
    cfg=dict(budget=dict(branch='CR_confirmation',approved_gpu_hours=830,
        approval_reference='CPU fixture',row_max_seconds={'next':60}))
    monkeypatch.setenv('SLURM_JOB_START_TIME','1');monkeypatch.setattr(budget.time,'time',lambda:100)
    assert budget.reserve(tmp_path,cfg,dict(id='next',family='cr',gpus=8),record,new)==60
    result=budget.reconcile(tmp_path,family='cr',query=lambda *a:{})
    assert result['status']=='PENDING' and load(path)['reserved_gpu_hours']>0


def test_linux_without_proc_children_fails_closed(monkeypatch):
    monkeypatch.setattr(cr_process.sys, 'platform', 'linux')
    original_iterdir, original_read = Path.iterdir, Path.read_text
    own_task = Path('/proc', str(cr_process.os.getpid()), 'task')
    monkeypatch.setattr(Path, 'iterdir', lambda p: iter([own_task/'fixture']) if p==own_task else original_iterdir(p))
    def missing(path, *args, **kwargs):
        if path.parent==own_task/'fixture' and path.name=='children': raise FileNotFoundError()
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',missing)
    with pytest.raises(RuntimeError,match='CR proc children inventory unavailable'):
        cr_process.processes()



def test_linux_cpu_teardown_preflight_without_gpu():
    if not sys.platform.startswith('linux'):
        pytest.skip('Linux execution UNVERIFIED on this platform')
    assert cr_process.cpu_preflight()['teardown_confirmed'] is True


def test_cpu_preflight_runs_supervisor_with_cuda_hidden(monkeypatch):
    monkeypatch.setattr(cr_process.sys,'platform','linux')
    monkeypatch.setattr(cr_process,'processes',lambda:{})
    calls=[]
    def supervised(command,evidence,**kwargs):
        calls.append((command,kwargs))
        evidence['teardown_confirmed']=True
    monkeypatch.setattr(cr_process,'gpu_run',supervised)
    assert cr_process.cpu_preflight()['teardown_confirmed']
    command,kwargs=calls[0]
    assert command[:3]==[sys.executable,'-I','-c']
    assert 'start_new_session=True' in command[3] and 'SIG_IGN' in command[3]
    assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and kwargs['check']
    def unconfirmed(command,evidence,**kwargs): pass
    monkeypatch.setattr(cr_process,'gpu_run',unconfirmed)
    with pytest.raises(RuntimeError,match='preflight unconfirmed'):
        cr_process.cpu_preflight()
