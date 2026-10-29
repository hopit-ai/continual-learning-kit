"""Live local-shard P2 accounting; fp32 masters, fp64 reductions, no full clone."""
import hashlib,json,math
from kit.p2_l2 import local,CHUNK
from kit.s_common import require
from kit.v4_restore_check import state_hash


def reduce(value):
    import torch
    if torch.distributed.is_initialized():torch.distributed.all_reduce(value)
    return value


def distance(anchor):
    import torch
    total=torch.zeros((),dtype=torch.float64,device=local(anchor.segments[0][0]).device)
    for p,start,b in anchor.chunks():
        v=local(p).detach().view(-1)[start:start+b.numel()]
        require(v.dtype==torch.float32,'registered smoke requires actual fp32 master parameters')
        total+=(v.double()-b.double()).square().sum()
    return float(reduce(total))


def geometry(student,anchor):
    """Bind unique parameters, ties, logical ranges, padding and anchor coverage."""
    import torch
    aliases={}
    for name,p in student.named_parameters(remove_duplicate=False):
        if p.requires_grad:aliases.setdefault(id(p),[]).append(name.replace('_fsdp_wrapped_module.',''))
    spans={}
    for p,offset,b in anchor.segments:spans.setdefault(id(p),[]).append((offset,offset+b.numel()))
    inventory=[]
    for name,p in student.named_parameters():
        if not p.requires_grad:continue
        v=local(p);pieces=sorted(spans.get(id(p),[]));cursor=0
        for lo,hi in pieces:require(lo==cursor and hi>lo,'anchor shard hole/overlap');cursor=hi
        require(cursor==v.numel(),'anchor must cover every logical local parameter exactly once')
        require(v.dtype==torch.float32,'P2 state requires fp32 master parameters')
        if hasattr(p,'to_local'):
            require(len(p.placements)==1 and type(p.placements[0]).__name__=='Shard' and p.placements[0].dim==0,'unsupported P2 smoke/node shard')
            mesh=p.device_mesh;coord=mesh.get_coordinate();width=mesh.size()
            block=(p.shape[0]+width-1)//width;first=min(p.shape[0],coord[0]*block);last=min(p.shape[0],first+block)
            stride=math.prod(p.shape[1:]);lo,hi=first*stride,last*stride
        else:lo,hi=0,v.numel()  # explicitly CPU-only fixture; node reader requires width8
        require(hi-lo==v.numel(),'P2 logical shard padding mismatch')
        inventory.append(dict(name=name.replace('_fsdp_wrapped_module.',''),shape=list(p.shape),
            aliases=aliases[id(p)],lo=lo,hi=hi,local_numel=v.numel(),padding_numel=0,dtype=str(v.dtype)))
    return dict(schema='rung-p2-shards.v1',inventory=inventory,anchor_elements=sum(b.numel() for _,_,b in anchor.segments),
        parameter_elements=sum(r['local_numel'] for r in inventory),tied_once=True,full_local_coverage=True)


def validate_geometry(records,width):
    require(len(records)==width,'all P2 shard ranks required')
    require(all(len({v['name'] for v in r['inventory']})==len(r['inventory']) for r in records),'duplicate P2 parameter inventory')
    inventories=[{v['name']:v for v in r['inventory']} for r in records]
    require(inventories and all(set(v)==set(inventories[0]) for v in inventories),'P2 rank inventories differ')
    for r in records:require(r['anchor_elements']==r['parameter_elements'] and r['tied_once'] and r['full_local_coverage'],'P2 anchor/tie accounting failed')
    for name in inventories[0]:
        rows=[v[name] for v in inventories];shape=rows[0]['shape'];cursor=0
        require(all(r['shape']==shape and r['aliases']==rows[0]['aliases'] and r['dtype']=='torch.float32' and r['padding_numel']==0 for r in rows),'P2 shapes/ties/dtypes differ')
        for r in sorted(rows,key=lambda r:r['lo']):
            require(r['lo']==cursor and r['hi']-r['lo']==r['local_numel'],'P2 global shard hole/overlap');cursor=r['hi']
        require(cursor==math.prod(shape),'P2 global parameter coverage incomplete')
    return True


def finite_state(actor):
    import torch
    states=list(actor.actor_module.parameters())+list(actor.teacher_module.parameters())
    states += [v for s in actor.actor_optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
    return all(bool(torch.isfinite(local(v)).all()) for v in states) and all(p.grad is None for p in actor.teacher_module.parameters())


def optimizer_materialised(actor):
    import torch
    for p in actor.actor_module.parameters():
        if not p.requires_grad:continue
        state=actor.actor_optimizer.state.get(p,{})
        if not all(k in state and local(state[k]).numel()==local(p).numel() and local(state[k]).dtype==torch.float32 for k in ('exp_avg','exp_avg_sq')):return False
    return True


def audit_add(anchor,value):
    """Audit outside the timer; time the same chunked analytic add as node D.

    Keep only one parameter's pre-add gradient for elementwise verification,
    never another whole model. Differences stay bounded by Anchor.CHUNK.
    """
    import torch,time
    from kit.p2_l2 import Anchor
    sums=torch.zeros(5,dtype=torch.float64,device=local(anchor.segments[0][0]).device)
    maximum=torch.zeros_like(sums[0]);started=time.monotonic();addition=0.
    groups={}
    for segment in anchor.segments:groups.setdefault(id(segment[0]),[]).append(segment)
    with torch.no_grad():
        for segments in groups.values():
            part=Anchor(segments,copy=False);p=segments[0][0]
            require(p.grad is not None,'smoke trainable gradient missing')
            before=local(p.grad).detach().clone().view(-1)
            for parameter,start,b in part.chunks():
                v=local(parameter).detach().view(-1)[start:start+b.numel()]
                g=before[start:start+b.numel()]
                require(v.dtype==g.dtype==b.dtype==torch.float32,'smoke gradient/master/anchor precision is not fp32')
                delta=v-b;sums[0]+=delta.double().square().sum();sums[1]+=g.double().square().sum();sums[2]+=(g.double()*delta.double()).sum()
            if local(p).is_cuda:torch.cuda.synchronize()
            tick=time.monotonic()
            part.add_gradient(value)
            if local(p).is_cuda:torch.cuda.synchronize()
            addition+=time.monotonic()-tick
            for parameter,start,b in part.chunks():
                v=local(parameter).detach().view(-1)[start:start+b.numel()]
                g=local(parameter.grad).view(-1)[start:start+b.numel()]
                expected=before[start:start+b.numel()]+value*(v-b)
                maximum=torch.maximum(maximum,(g-expected).double().abs().max())
                sums[3]+=g.double().square().sum();sums[4]+=expected.double().square().sum()
            del before
    reduce(sums)
    if torch.distributed.is_initialized():torch.distributed.all_reduce(maximum,op=torch.distributed.ReduceOp.MAX)
    d,g,dot,combined,expected=map(float,sums)
    error=float(maximum);elapsed=time.monotonic()-started
    require(math.isclose(combined,g+2*value*dot+value*value*d,rel_tol=2e-5,abs_tol=1e-10) and error<=1e-6,'real distributed analytic gradient identity failed')
    return dict(distance2=d,regulariser=d/2,task_grad_norm=math.sqrt(g),combined_squared=combined,
        audit_excluded_seconds=max(0.,elapsed-addition),addition_seconds=addition,
        task_delta_dot=dot,analytic_combined_squared=g+2*value*dot+value*value*d,analytic_gradient_error=error)
