#!/usr/bin/env python3
"""Registered regenerative-L2 mechanical smoke. No execution on import."""
from pathlib import Path
import gc,hashlib,json,math,statistics,sys,time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.s_common import ROOT,sha,atomic,require
from kit.p2_l2 import Anchor,VARIANT,REVISION,CHUNK

DISTANCE_CHUNK=4*1024*1024
ADD_CHUNK=1024*1024
from kit.p_determinism import configure
from kit.p_smoke import frozen_inputs,load_model

ASSERTIONS=('finite','analytic','coefficient','norms','timing','reservation','contraction','identical_start')
SETTINGS=ROOT/'kit/resources/p/smoke3-settings.json'


SCHEMA='rung-p2-distributed-smoke.v2'


def code_hashes():
    names=('kit/p2_smoke.py','kit/p2_l2.py','kit/x_p2_actor.py','kit/x_p2_runtime.py',
        'kit/x_memory.py','kit/x_smoke_memory.py','kit/x_smoke_batches.py','kit/x_smoke_inputs.py','kit/p3b_common.py','kit/p3b_train.py','kit/p3b_loss.py','kit/next_slots.py','kit/c_d_prompt.py','kit/c_data.py','kit/sdft_objective.py','kit/v4_contract.py','kit/x_smoke_settings.py','kit/resources/p2/split-forecast.json','kit/x_p2_resources.py','kit/x_p2_measure.py','kit/x_ema.py','kit/x_resume.py',
        'kit/x_p2_state.py','kit/x_p2_smoke_runtime.py','kit/x_p2_smoke_read.py','kit/x_p2_export.py','kit/x_p2_verdict.py','kit/x_smoke_timing.py','kit/resources/p2/attempt4-forecast.json','kit/x_integration_smoke.py',
        'kit/x_sdft_entry.py','kit/x_sdft_runner.py','kit/x_worker.py','kit/x_smoke_checkpoint.py','kit/smoke_steps.py',
        'kit/sdft_actor.py','kit/m2_train.py','kit/c_train.py','kit/c_sdft_runner.py',
        'kit/p_determinism.py','kit/p_smoke.py','kit/resources/p/smoke3-settings.json',
        'kit/resources/p/smoke-finqa-train.jsonl','kit/resources/p2/smoke.json','scripts/smoke_x_integration_modal.py')
    return {n:sha(ROOT/n) for n in names}


def tensor_hash(values):
    digest=hashlib.sha256()
    for value in values:
        digest.update(str((str(value.dtype),tuple(value.shape))).encode())
        for start in range(0,value.numel(),CHUNK):
            digest.update(value.detach().reshape(-1)[start:start+CHUNK].cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def finite(values):
    import torch
    return all(bool(torch.isfinite(v).all()) for v in values)


def distance(left,right):
    # fp32 master differences with fp64 norm accumulation, bounded chunks.
    import torch
    total=0.
    for a,b in zip(left,right):
        require(a.numel()==b.numel(),'smoke distance geometry differs')
        for i in range(0,a.numel(),DISTANCE_CHUNK):
            # Exported weights may be CUDA bf16; the anchor is CPU fp32.
            aa=a.detach().reshape(-1)[i:i+DISTANCE_CHUNK].to(device=b.device,dtype=b.dtype)
            bb=b.detach().reshape(-1)[i:i+DISTANCE_CHUNK]
            total+=float(torch.linalg.vector_norm(aa-bb,dtype=torch.float64).square())
    return total


def analytic_checks():
    """Value/autograd, zero-at-base, tied parameters and simulated local shards."""
    import torch,copy
    torch.manual_seed(101)
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__();self.a=torch.nn.Parameter(torch.randn(11));self.tied=self.a
    base=Tied();net=copy.deepcopy(base)
    with torch.no_grad():net.a.add_(.25)
    objective=sum((p-dict(base.named_parameters())[n]).square().sum()/2 for n,p in net.named_parameters())
    objective.backward();expected=net.a.grad.clone();net.a.grad.zero_()
    anchor=Anchor.tiny(net,base);value=float(anchor.sums()[0])/2;anchor.add_gradient(1.)
    full=net.a.grad.clone();net.a.grad.zero_()
    shards=[Anchor([(net.a,lo,base.a.detach()[lo:hi])]) for lo,hi in ((0,3),(3,7),(7,11))]
    for shard in shards:shard.add_gradient(1.)
    tied=len(list(net.parameters()))==1
    zero=copy.deepcopy(base);zero.a.grad=torch.zeros_like(zero.a)
    origin=Anchor.tiny(zero,base);origin.add_gradient(1.)
    return dict(value_error=abs(value-float(objective.detach())),gradient_error=float((full-expected).abs().max()),
        shard_error=float((net.a.grad-full).abs().max()),zero_at_base=bool((zero.a.grad==0).all()),tied_once=tied,
        passed=math.isclose(value,float(objective.detach()),abs_tol=1e-7) and torch.allclose(full,expected) and
            torch.equal(net.a.grad,full) and bool((zero.a.grad==0).all()) and tied)


def host_peak(cgroup=Path('/sys/fs/cgroup')):
    """Probe cgroup first; labelled whole-container RSS sampler fallback."""
    for name in ('memory.peak','memory/memory.max_usage_in_bytes'):
        path=Path(cgroup)/name
        try:return dict(bytes=int(path.read_text().strip()),source=str(path),container_scope=True)
        except (OSError,ValueError):pass
    # Linux procfs is shared by all processes in this PID namespace. This is a
    # labelled sampled RSS sum, not a kernel high-water mark (shared pages count
    # conservatively more than once). The supervisor samples throughout startup.
    try:
        total=0
        for path in Path('/proc').glob('[0-9]*/status'):
            try:
                for line in path.read_text().splitlines():
                    if line.startswith('VmRSS:'):total+=int(line.split()[1])*1024
            except OSError:pass
        if total:return dict(bytes=total,source='sampled /proc/*/status RSS sum (shared pages counted per process)',container_scope=True,sampled=True)
    except OSError:pass
    import resource
    return dict(bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source='fallback process ru_maxrss; whole-container accounting unavailable',container_scope=False,sampled=True)


class HostMonitor:
    def __init__(self):
        import threading
        self.initial=host_peak();self.peak=dict(self.initial);self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.25):self.sample()
        self.thread=threading.Thread(target=sample,daemon=True)
    def sample(self):
        value=host_peak()
        if value['bytes']>self.peak['bytes']:self.peak=value
        return self.peak
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.sample();self.stop.set();self.thread.join()


def norm_squared(values):
    import torch
    return sum(float(torch.linalg.vector_norm(v.detach().reshape(-1)[i:i+DISTANCE_CHUNK],dtype=torch.float64).square())
        for v in values for i in range(0,v.numel(),DISTANCE_CHUNK))


def gradient_dot(masters,base):
    import torch
    return sum(float(torch.dot(p.grad.reshape(-1)[i:i+DISTANCE_CHUNK].double(),
        (p.detach().reshape(-1)[i:i+DISTANCE_CHUNK]-b.reshape(-1)[i:i+DISTANCE_CHUNK]).double()))
        for p,b in zip(masters,base) for i in range(0,p.numel(),DISTANCE_CHUNK))


def add_in_place(masters,base,value):
    import torch
    if value==0:return
    with torch.no_grad():
        for p,b in zip(masters,base):
            for start in range(0,p.numel(),ADD_CHUNK):
                g=p.grad.view(-1)[start:start+ADD_CHUNK]
                g.add_(p.view(-1)[start:start+ADD_CHUNK],alpha=value).add_(b.view(-1)[start:start+ADD_CHUNK],alpha=-value)


def arm(*args,**kwargs):
    raise ValueError('P2 registration revised 25 October: use the real-verl 1.7B/two-H100 integration smoke')


def main():
    raise SystemExit('P2 registration revised 25 October: use scripts/smoke_x_integration_modal.py; CPU-offload smoke retired')

if __name__=='__main__':main()
