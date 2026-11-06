import copy
import math
import pytest
import torch
from kit import nm_loss as n
from kit.nm_common import Pending,Invalid,money


def test_fisher_individual_squares_and_global_trace():
    model=torch.nn.Linear(1,1,bias=False).double();model.weight.data.fill_(0)
    rows=[torch.tensor([[1.]],dtype=torch.double),torch.tensor([[-1.]],dtype=torch.double)]
    raw=n.fisher_estimate(model,rows,lambda m,x:m(x).mean())
    assert raw['weight'].item()==1  # square(mean gradient) would incorrectly give zero
    weights,meta=n.normalize_fisher(raw)
    assert weights['weight'].item()==1 and meta['parameters']==1


def test_fisher_shared_median_gradient_calibration():
    records={str(s):dict(task_grad_norm=g,incoming_distance=d,weighted_distance=w,source='first-training-batch',no_mutation=True) for s,g,d,w in [(101,2,1,3),(102,8,2,4),(103,18,3,3)]}
    v=n.calibrate(records)
    assert v['lambda_iso']==pytest.approx(.4)
    assert v['lambda_fisher']==pytest.approx(.2)
    assert sorted(v['ratios'].values())[1]==pytest.approx(1)
    records['102']['weighted_distance']=0
    with pytest.raises(Pending):n.calibrate(records)


def test_weighted_penalty_gradient_matches_autograd():
    p=torch.nn.Parameter(torch.tensor([1.,-2.,3.],dtype=torch.double));b=torch.tensor([.2,1.,-1.]);w=torch.tensor([0.,2.,1.])
    p.grad=torch.tensor([.3,.4,.5],dtype=torch.double);original=p.grad.clone()
    a=n.WeightedAnchor([(p,0,b,w)])
    sums=a.sums();a.add_gradient(.4)
    expected=torch.autograd.grad(.4*.5*(w*(p-b).square()).sum(),p)[0]
    assert torch.allclose(p.grad-original,expected,atol=1e-7)
    assert float(sums[2])==pytest.approx(float((.5*(w*(p-b).square()).sum()).detach()))
    with pytest.raises(Invalid):n.WeightedAnchor([(p,0,b,w),(p,0,b,w)])


@pytest.mark.parametrize('world',[1,2,8])
def test_global_divisor_never_changes_with_cr_or_ranks(world):
    theta=torch.tensor(.3,requires_grad=True);x=torch.arange(48,dtype=torch.float32)/48
    local=[]
    for rank in range(world):
        term=(theta*x[rank::world]).square().sum()*world/32
        local.append(torch.autograd.grad(term,theta,retain_graph=True)[0])
    got=sum(local)/world
    expected=torch.autograd.grad((theta*x).square().sum()/32,theta)[0]
    assert got==pytest.approx(float(expected),rel=1e-6)


def test_grouped_advantage_zero_groups_and_detachment():
    reward=torch.tensor([[0.,0.,0.,0.],[1.,1.,1.,1.],[0.,1.,0.,1.]],requires_grad=True)
    a=n.advantages(reward)
    assert torch.equal(a[:2],torch.zeros_like(a[:2])) and not a.requires_grad
    assert a[-1].tolist()==pytest.approx([-1,1,-1,1],abs=3e-6)


@pytest.mark.parametrize('adv,rho,gradient',[(1.,1.,-1.),(1.,1.5,0.),(-1.,.5,0.),(-1.,1.,1.)])
def test_grpo_clipping_gradient_signs(adv,rho,gradient):
    x=torch.tensor([[math.log(rho)]],requires_grad=True);old=torch.zeros_like(x,requires_grad=True)
    value=n.grpo_tokens(x,old,torch.tensor([adv]),torch.ones_like(x));value.backward()
    assert x.grad.item()==pytest.approx(gradient) and old.grad is None


def test_freeze_only_decoder_blocks_ties_and_optimizer():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(4)])
            self.model.embed_tokens=torch.nn.Embedding(3,2);self.lm_head=torch.nn.Linear(2,3)
    model=Toy();optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    audit=n.freeze_mask(model);n.filter_optimizer(optimizer)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    for p in model.parameters():
        if p.requires_grad:p.grad=torch.ones_like(p)
    optimizer.step()
    for name,p in model.named_parameters():
        frozen=any(name.startswith('model.layers.'+str(i)+'.') for i in range(3))
        assert p.requires_grad!=frozen
        assert torch.equal(before[name],p)==frozen
    assert sum(v['count'] for v in audit if v['frozen'])==18
    model.lm_head.weight=model.model.layers[0].weight
    with pytest.raises(Invalid):n.freeze_mask(model)


def test_task_kl_reference_gradient_and_no_base_gradient():
    from kit.m3_loss import projected_forward_kl
    torch.manual_seed(1)
    h=torch.randn(1,4,3,requires_grad=True);w=torch.randn(7,3,requires_grad=True)
    bh=torch.randn(1,4,3,requires_grad=True);bw=torch.randn(7,3,requires_grad=True)
    result=projected_forward_kl(h,w,bh,bw,2).mean()
    b=(bh.detach()@bw.detach().T).log_softmax(-1);s=(h@w.T).log_softmax(-1)
    ref=(b.exp()*(b-s)).sum(-1).mean()
    a=torch.autograd.grad(result,(h,w),retain_graph=True);r=torch.autograd.grad(ref,(h,w))
    assert float(abs(result-ref).detach())<1e-5
    assert all(torch.allclose(x,y,atol=1e-5) for x,y in zip(a,r))
    assert bh.grad is None and bw.grad is None


def test_half_up_only_component_boundary():
    assert money('1.825')=='1.83' and money('12.324')=='12.32'


def test_fisher_accumulates_on_parameter_device_without_host_copies(monkeypatch):
    model=torch.nn.Linear(2,1,bias=False).double();model.weight.data.fill_(0)
    rows=[torch.tensor([[2.,1.]],dtype=torch.double),torch.tensor([[-2.,3.]],dtype=torch.double)]
    def forbidden(*args,**kw):raise AssertionError('per-example host copy')
    monkeypatch.setattr(torch.Tensor,'cpu',forbidden)
    raw=n.fisher_estimate(model,rows,lambda m,x:m(x).mean())
    assert raw['weight'].device==model.weight.device and raw['weight'].dtype==torch.float32
    assert torch.equal(raw['weight'],torch.tensor([[4.,5.]]))
    normalized,meta=n.distributed_trace(raw,model.weight.device)
    assert torch.allclose(normalized['weight'],torch.tensor([[4/4.5,5/4.5]]))
    assert meta['trace']==9 and meta['parameters']==2


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('updates',[0,4,8,20,40])
def test_frozen_ema_accepts_exact_known_rounding_and_refuses_movement(dtype,updates):
    torch.manual_seed(9);student=torch.randn(100,dtype=dtype);teacher=student.clone()
    for _ in range(updates):teacher.mul_(.99).add_(student,alpha=.01)
    assert n.frozen_ema_equal(teacher,student,updates)
    teacher[0]+=1
    assert not n.frozen_ema_equal(teacher,student,updates)
