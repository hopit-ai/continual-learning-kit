"""Registered P2 coefficient and chunked analytic L2 on unique local shards."""
from __future__ import annotations
import math
import statistics
from pathlib import Path
from kit.s_common import require

VARIANT='p2-l2-init'
REVISION='b968826d9c46dd6066d109eabc6255188de91218'
CHUNK=65536


def local(value):return value.to_local() if hasattr(value,'to_local') else value


def coefficient(records):
    require(set(records)=={str(s) for s in range(101,104)},'P2 needs all three seed ratios')
    ratios=[]
    for seed in range(101,104):
        row=records[str(seed)];g,d=row['task_grad_norm'],row['incoming_distance']
        require(all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in (g,d)),
            'P2 coefficient requires finite positive norms for every seed')
        ratios.append(g/d)
    require(all(math.isfinite(v) and v>0 for v in ratios),'P2 nonfinite ratio')
    value=.1*statistics.median(ratios)
    require(math.isfinite(value) and value>0,'P2 nonfinite coefficient')
    realised={str(s):value/ratio for s,ratio in zip(range(101,104),ratios)}
    require(math.isclose(statistics.median(realised.values()),.1,rel_tol=1e-12),'P2 median ratio drift')
    return dict(value=value,ratios=dict(zip(map(str,range(101,104)),ratios)),realised_ratios=realised)


def deviation(observed,expected):
    """Calibration drift is diagnostic; never changes the frozen coefficient."""
    import warnings
    finite=all(isinstance(v,(int,float)) and math.isfinite(v) for v in (observed,expected))
    relative=abs(observed-expected)/max(abs(expected),1e-30) if finite else None
    warning=relative is None or relative>1e-3
    if warning:warnings.warn('P2 calibration relative deviation above 1e-3: '+str(relative),RuntimeWarning)
    return dict(observed=observed,expected=expected,relative_deviation=relative,warning=warning)


class Anchor:
    """(parameter, offset, frozen fp32 local base) segments, excluding padding.

    No microbatch/world-size multiplier: D has already accumulated, reduced and
    unscaled its gradient when add_gradient is called.
    """
    def __init__(self,segments,*,copy=True):
        import torch
        self.segments=[];occupied={}
        for parameter,offset,base in segments:
            require(local(parameter).is_contiguous(),'P2 noncontiguous local parameter')
            if parameter.grad is not None:require(local(parameter.grad).is_contiguous(),'P2 noncontiguous local gradient')
            require(parameter.requires_grad,'P2 anchor includes nontrainable parameter')
            base=base.detach().to(device=parameter.device,dtype=torch.float32)
            if copy:base=base.clone()
            base=base.reshape(-1)
            require(not base.requires_grad and offset>=0 and offset+base.numel()<=local(parameter).numel(),'P2 shard geometry')
            spans=occupied.setdefault(id(parameter),[])
            require(all(offset+base.numel()<=lo or offset>=hi for lo,hi in spans),'P2 duplicate/tied shard')
            spans.append((offset,offset+base.numel()));self.segments.append((parameter,offset,base))
        require(self.segments,'P2 empty anchor')
    @classmethod
    def tiny(cls,student,base):
        values=dict(base.named_parameters())
        return cls([(p,0,values[n]) for n,p in student.named_parameters() if p.requires_grad])
    def chunks(self):
        for index,(parameter,offset,base) in enumerate(self.segments):
            if base.device != parameter.device:
                base=base.to(parameter.device);self.segments[index]=(parameter,offset,base)
            for start in range(0,base.numel(),CHUNK):
                count=min(CHUNK,base.numel()-start)
                yield parameter,offset+start,base[start:start+count]
    def sums(self):
        import torch
        sums=torch.zeros(2,dtype=torch.float64,device=self.segments[0][0].device)
        with torch.no_grad():
            for parameter,start,base in self.chunks():
                delta=local(parameter).detach().reshape(-1)[start:start+base.numel()].float()-base
                sums[0]+=delta.double().square().sum()
                require(parameter.grad is not None,'P2 missing trainable gradient')
                grad=local(parameter.grad).detach().reshape(-1)[start:start+base.numel()]
                sums[1]+=grad.double().square().sum()
        if torch.distributed.is_initialized():torch.distributed.all_reduce(sums)
        require(bool(torch.isfinite(sums[0])),'P2 nonfinite distance')
        return sums
    def add_gradient(self,value):
        import torch
        require(math.isfinite(value) and value>=0,'P2 invalid lambda')
        if value==0:return  # preserve D bit-for-bit, including absent gradients
        with torch.no_grad():
            for parameter,start,base in self.chunks():
                require(parameter.grad is not None,'P2 missing trainable gradient')
                require(local(parameter.grad).is_contiguous(),'P2 noncontiguous local gradient')
                grad=local(parameter.grad).view(-1)[start:start+base.numel()]
                # Keep the analytic delta in fp32 before casting to D's local
                # gradient dtype. The CPU smoke's two in-place adds are separate.
                delta=local(parameter).detach().view(-1)[start:start+base.numel()].float()-base
                grad.add_(delta.to(grad.dtype),alpha=value)


def load_anchor(student,checkpoint):
    """Read only matching FSDP2 DTensor or FSDP1 shards from mmap'ed safetensors, in chunks.

    Pinned D uses FSDP2 Shard(0); the legacy FSDP1 adapter also supports
    use_orig_params=False. FlatParameter's unique FQNs exclude tied
    aliases; _shard_param_infos excludes alignment and end padding. Unsupported
    geometry fails closed instead of loading a full base model on every GPU.
    """
    import json,torch
    from safetensors import safe_open
    from contextlib import ExitStack
    checkpoint=Path(checkpoint)
    index=checkpoint/'model.safetensors.index.json'
    mapping=json.loads(index.read_text())['weight_map'] if index.exists() else None
    segments=[]
    with ExitStack() as stack:
        files={p.name:stack.enter_context(safe_open(str(p),framework='pt',device='cpu')) for p in sorted(checkpoint.glob('*.safetensors'))}
        if mapping is None:mapping={key:name for name,handle in files.items() for key in handle.keys()}
        def read(name,start,count):
            require(name in mapping,'P2 base missing parameter '+name)
            view=files[mapping[name]].get_slice(name);shape=view.get_shape()
            if len(shape)==1:return view[start:start+count].float()
            require(len(shape)==2,'P2 unsupported base tensor rank')
            width=shape[1];lo,hi=start//width,(start+count-1)//width+1
            return view[lo:hi,:].reshape(-1)[start-lo*width:start-lo*width+count].float()
        seen=set()
        for name,p in student.named_parameters():
            if not p.requires_grad or id(p) in seen:continue
            seen.add(id(p))
            if hasattr(p,'to_local'):
                # Pinned D is FSDP2, one fully-sharded mesh, Shard(0). Read only
                # its contiguous logical row range; DTensor has no pad elements.
                require(len(p.placements)==1 and type(p.placements[0]).__name__=='Shard' and p.placements[0].dim==0,
                    'P2 requires the pinned one-dimensional fully-sharded DTensor layout')
                mesh=p.device_mesh;coordinate=mesh.get_coordinate()
                require(coordinate is not None and len(coordinate)==1,'P2 unsupported FSDP2 mesh')
                rows=p.shape[0];width=mesh.size();block=(rows+width-1)//width
                first=min(rows,coordinate[0]*block);last=min(rows,first+block)
                stride=math.prod(p.shape[1:]);count=(last-first)*stride
                require(local(p).numel()==count,'P2 DTensor logical shard/padding differs')
                key=name.replace('_fsdp_wrapped_module.','')
                for start in range(0,count,CHUNK):
                    size=min(CHUNK,count-start);segments.append((p,start,read(key,first*stride+start,size)))
                continue
            require(hasattr(p,'_shard_param_infos'),'P2 unsupported sharded parameter geometry')
            prefix=name.removesuffix('_flat_param').replace('_fsdp_wrapped_module.','')
            for fqn,info in zip(p._fqns,p._shard_param_infos):
                if not info.in_shard:continue
                for start in range(0,info.numel_in_shard,CHUNK):
                    count=min(CHUNK,info.numel_in_shard-start)
                    segments.append((p,info.offset_in_shard+start,read(prefix+fqn,info.intra_param_start_idx+start,count)))
        # Anchor construction transfers these local fragments only. No full model.
        return Anchor(segments)
