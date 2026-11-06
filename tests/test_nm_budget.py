from pathlib import Path
import pytest
from kit.nm_common import *
from kit.nm_budget import reconcile,reserve
from kit.nm_fisher import price


def test_fisher_price_uses_measured_domain_examples_and_all_calibrations():
    p=price(dict(per_domain_seconds={d:[1,1] for d in ('instruction','maths','knowledge')}),{str(s):2 for s in SEEDS},.2,.25)
    assert p['compute_gpu_hours']=='0.58' and p['total_gpu_hours']=='0.93'
    with pytest.raises(Pending):price(dict(per_domain_seconds={}),{},0,.25)


def test_reconcile_failed_attempt_full_allocation_and_current_hash(tmp_path):
    root=tmp_path/'attempts'/'task'/'nonce';root.mkdir(parents=True)
    p=root/'receipt.json';atomic(p,dict(task='task',family='fisher',phase='train',gpus=8,nonce='nonce',status='FAILED',
        slurm_job_id='123',allocation_started_at=100,allocation_gpu_hours=None,started=110,ended=120))
    atomic(tmp_path/'state/task.json',dict(receipt=str(p.relative_to(tmp_path)),sha256=sha(p)))
    pending=reconcile(tmp_path,query=lambda *args:{})
    assert pending['status']=='PENDING' and pending['gpu_hours']==0
    done=reconcile(tmp_path,query=lambda job,start:dict(allocation_ended_at=3700))
    assert done['gpu_hours']==8 and done['status']=='COMPLETE'
    assert load(tmp_path/'state/task.json')['sha256']==sha(p)
    again=reconcile(tmp_path,query=lambda *a:(_ for _ in ()).throw(AssertionError('must not requery finalised')))
    assert again['gpu_hours']==8


def test_reserve_uses_fixed_owner_branch_and_hard_bound(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_START_TIME','100')
    monkeypatch.setattr('kit.nm_budget.time.time',lambda:100)
    attempt=tmp_path/'attempts/task/n';attempt.mkdir(parents=True)
    cfg=dict(budget=dict(branch='K_CR',approved_gpu_hours=1362,approval_reference='owner',row_max_seconds={'task':3600}))
    row=dict(id='task',gpus=8);record=dict(slurm_job_id='123')
    assert reserve(tmp_path,cfg,row,record,attempt)==3600
    assert record['reserved_gpu_hours']==pytest.approx(8*(3630)/3600)
    cfg['budget']['approved_gpu_hours']=100
    with pytest.raises(Invalid):reserve(tmp_path,cfg,row,record,attempt)


def test_fisher_streaming_exact_quantiles_and_zero_support(tmp_path):
    import numpy as np
    import torch
    from kit.nm_fisher_stats import summarize
    values=torch.tensor([0.,0.,.3,1.,2.,5.,10.],dtype=torch.float32)
    paths=[]
    for i,part in enumerate((values[:3],values[3:])):
        p=tmp_path/f'rank-{i}.pt';torch.save(dict(weight=part),p);paths.append(p)
    s=summarize(paths,chunk=2)
    assert s['parameters']==7 and s['nonzero']==5 and s['maximum']==10
    for q,value in s['quantiles'].items():assert value==pytest.approx(np.quantile(values.numpy(),float(q)))
    assert s['effective_support']==pytest.approx(float(values.double().sum()**2/values.double().square().sum()))


def test_fisher_total_sums_published_rounded_components():
    p=price(dict(per_domain_seconds={d:[.02] for d in ('instruction','maths','knowledge')}),{str(s):.453333333333333 for s in SEEDS},0,.25)
    assert p['compute_gpu_hours']=='0.01' and p['reserve_gpu_hours']=='0.00' and p['total_gpu_hours']=='0.01'


def test_calibration_resubmit_retains_original_measured_time(tmp_path):
    from kit.nm_pipeline import run_native
    atomic(tmp_path/'prepared.json',dict(env={}))
    atomic(tmp_path/'native-receipt.json',dict(status='CALIBRATED'))
    atomic(tmp_path/'native-timing.json',dict(native_receipt=str(tmp_path/'native-receipt.json'),wall_seconds=42.5))
    result=run_native({},tmp_path,True)
    assert result['reused_complete'] and result['wall_seconds']==42.5
