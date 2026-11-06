"""Isolated NM mathematical components; no changes to historical D implementations."""
import math
import statistics
import torch
from kit.nm_common import need,ready,SEEDS
from kit.m3_loss import projected_forward_kl,projected_ce,sequence_mean
from kit.sdft_actor import _model_forward,teacher_projection


def task_kl(model,base,inputs):
    """Full-vocabulary base→student KL, response/EOS mean, plain task prefix."""
    snapshot=teacher_projection(base,inputs);width=inputs['responses'].shape[-1]
    def head(hidden,weight):
        return sequence_mean(projected_forward_kl(hidden[:,-width-1:-1],weight,
            snapshot['hidden'],snapshot['weight'],256),inputs['response_mask'])
    return _model_forward(model,inputs,head)


def advantages(rewards):
    need(rewards.shape[-1]==4 and bool(((rewards==0)|(rewards==1)).all()),'GRPO groups require four binary rewards')
    r=rewards.detach().float();center=r-r.mean(-1,keepdim=True)
    return (center/(center.square().mean(-1,keepdim=True).sqrt()+1e-6)).detach()


def grpo_tokens(logps,old,advantage,mask):
    need(logps.shape==old.shape==mask.shape,'GRPO token geometry differs')
    need(advantage.shape==logps.shape[:-1],'GRPO advantage geometry differs')
    rho=(logps-old.detach()).exp();a=advantage.detach()[...,None]
    return -sequence_mean(torch.minimum(rho*a,rho.clamp(.8,1.2)*a),mask)


def sampled_logps(model,inputs):
    width=inputs['responses'].shape[-1]
    return _model_forward(model,inputs,lambda h,w:-projected_ce(h[:,-width-1:-1],w,inputs['responses'],256))


def calibrate(records,compatible_lambda=None):
    need(set(records)==set(map(str,SEEDS)),'Fisher calibration needs seeds 101–103')
    task=[];weighted=[]
    for seed in map(str,SEEDS):
        r=records[seed];g,d,w=(r[k] for k in ('task_grad_norm','incoming_distance','weighted_distance'))
        ready(all(type(x) in (float,int) and math.isfinite(x) and x>0 for x in (g,d,w)),'Fisher norms must be finite positive')
        need(r.get('source')=='first-training-batch' and r.get('no_mutation') is True,'outcome-free untouched calibration required')
        task.append(g/d);weighted.append(d/w)
    lam=.1*statistics.median(task) if compatible_lambda is None else compatible_lambda
    ready(type(lam) in (float,int) and math.isfinite(lam) and lam>0,'invalid isotropic coefficient')
    lam_f=lam*statistics.median(weighted)
    ratios={s:lam_f*records[s]['weighted_distance']/(lam*records[s]['incoming_distance']) for s in records}
    error=abs(statistics.median(ratios.values())-1)
    ready(math.isfinite(lam_f) and lam_f>0 and error<=1e-4,'Fisher median gradient match failed')
    return dict(lambda_iso=lam,lambda_fisher=lam_f,ratios=ratios,median_ratio_relative_error=error,
        inputs=records,convention='raw initial regulariser-gradient norm; not AdamW update matching')


def fisher_estimate(model,rows,loss_fn):
    """One sequence backward at a time. FSDP ranks see the same individual row.

    The caller freezes all base/proxy identities. Accumulation is fp32, gradients
    are never batch-averaged before squaring. Tied parameters count once.
    Local fp32 accumulators stay on the parameter device until publication.
    """
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    ready(parameters and rows,'Fisher requires nonempty parameters/data')
    out={n:torch.zeros_like(local(p),dtype=torch.float32) for n,p in parameters.items()}
    for row in rows:
        model.zero_grad(set_to_none=True);loss=loss_fn(model,row)
        need(loss.numel()==1 and bool(torch.isfinite(loss)),'nonfinite individual Fisher sequence score')
        loss.backward()
        for n,p in parameters.items():
            need(p.grad is not None,'Fisher trainable parameter has no gradient')
            g=local(p.grad).detach().float()
            out[n].addcmul_(g,g)
    model.zero_grad(set_to_none=True)
    finite=torch.ones((),device=next(iter(out.values())).device,dtype=torch.bool)
    for f in out.values():
        finite.logical_and_(torch.isfinite(f).all());f.div_(len(rows))
    need(bool(finite),'nonfinite Fisher gradient')
    return out


def local(value):return value.to_local() if hasattr(value,'to_local') else value


def normalize_fisher(shards):
    """CPU tiny-reference helper; production uses distributed_trace on local shards."""
    count=sum(v.numel() for v in shards.values());total=sum(float(v.double().sum()) for v in shards.values())
    ready(count>0 and math.isfinite(total) and total>0,'Fisher trace must be positive finite')
    need(all(bool(torch.isfinite(v).all() and (v>=0).all()) for v in shards.values()),'Fisher must be finite nonnegative')
    mean=total/count
    return {n:v/mean for n,v in shards.items()},dict(mean=mean,parameters=count,trace=total)


def distributed_trace(shards,device):
    values=torch.zeros(2,device=device,dtype=torch.float64)
    for v in shards.values():values[0]+=v.double().sum()
    values[1]=sum(v.numel() for v in shards.values())
    if torch.distributed.is_initialized():torch.distributed.all_reduce(values)
    total,count=values.tolist();ready(count>0 and math.isfinite(total) and total>0,'Fisher trace must be finite positive')
    mean=total/count
    return {n:v/mean for n,v in shards.items()},dict(mean=mean,parameters=int(count),trace=total)


class WeightedAnchor:
    """Analytic penalty once after reduced/unscaled D gradients, before clipping.

    Each segment is (parameter, local offset, fp32 base, fp32 diagonal weight).
    Frozen coordinates remain in scalar logging but get no gradient.
    """
    def __init__(self,segments):
        self.segments=[];spans={}
        for p,start,b,w in segments:
            b,w=b.detach().float().reshape(-1),w.detach().float().reshape(-1)
            need(b.shape==w.shape and bool(torch.isfinite(w).all() and (w>=0).all()),'invalid Fisher shard weights')
            need(local(p).is_contiguous() and start>=0 and start+b.numel()<=local(p).numel(),'invalid local shard geometry')
            used=spans.setdefault(id(p),[]);end=start+b.numel()
            need(all(end<=lo or start>=hi for lo,hi in used),'duplicate tied penalty coordinates');used.append((start,end))
            self.segments.append((p,start,b.to(p.device),w.to(p.device)))
        ready(self.segments,'empty penalty')
    def sums(self):
        out=torch.zeros(4,dtype=torch.float64,device=self.segments[0][0].device)
        with torch.no_grad():
            for p,start,b,w in self.segments:
                delta=local(p).detach().reshape(-1)[start:start+b.numel()].float()-b
                out[0]+=delta.double().square().sum();out[1]+=(w*delta).double().square().sum()
                out[2]+=(w.double()*delta.double().square()).sum()/2
                if p.requires_grad:
                    need(p.grad is not None,'missing trainable gradient')
                    out[3]+=local(p.grad).reshape(-1)[start:start+b.numel()].double().square().sum()
        if torch.distributed.is_initialized():torch.distributed.all_reduce(out)
        need(bool(torch.isfinite(out).all()),'nonfinite penalty calibration')
        return out
    def add_gradient(self,value):
        need(math.isfinite(value) and value>=0,'invalid penalty coefficient')
        if value==0:return
        with torch.no_grad():
            for p,start,b,w in self.segments:
                if not p.requires_grad:continue
                need(p.grad is not None,'missing trainable gradient')
                g=local(p.grad).view(-1)[start:start+b.numel()]
                delta=local(p).detach().view(-1)[start:start+b.numel()].float()-b
                g.add_((w*delta).to(g.dtype),alpha=value)


def freeze_mask(model):
    """All aliases are checked; a tensor spanning frozen/unfrozen roles refuses."""
    import re
    grouped={}
    for name,p in model.named_parameters(remove_duplicate=False):grouped.setdefault(id(p),[]).append((name,p))
    audit=[]
    for aliases in grouped.values():
        flags=[bool(re.search(r'(?:^|\.)model\.layers\.[012]\.',n)) for n,p in aliases]
        # Root QwenForCausalLM names begin model.layers; FSDP prefixes are permitted.
        need(all(flags) or not any(flags),'tied parameter spans frozen and trainable roles')
        p=aliases[0][1];frozen=all(flags)
        p.requires_grad_(not frozen)
        if frozen:p.grad=None
        audit.append(dict(names=[n for n,p in aliases],shape=list(p.shape),count=p.numel(),frozen=frozen))
    need(any(r['frozen'] for r in audit),'decoder blocks 0–2 not found')
    return audit


def filter_optimizer(optimizer):
    for group in optimizer.param_groups:group['params']=[p for p in group['params'] if p.requires_grad]
    for p in list(optimizer.state):
        if not p.requires_grad:del optimizer.state[p]


def frozen_ema_equal(teacher,student,updates):
    """Exact frozen EMA value, allowing the pinned .99/.01 mul/add rounding.

    The student is checked bit-identical separately. Lerp leaves equal inputs
    unchanged; mul_/add_ can round even when both inputs are the same tensor.
    Simulate that recurrence in the native dtype, including resumed updates.
    """
    teacher,student=local(teacher).detach(),local(student).detach()
    if torch.equal(teacher,student):return True
    expected=student.clone()
    for _ in range(updates):expected.mul_(.99).add_(student,alpha=.01)
    return torch.equal(teacher,expected)
