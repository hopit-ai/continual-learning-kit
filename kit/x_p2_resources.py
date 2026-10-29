"""Identity-bound resource telemetry shared by X4 and disposable X6 preflight."""
from pathlib import Path
import json,os,threading
from kit.s_common import atomic,read,sha,require
from kit.x_p2_state import geometry,optimizer_materialised,finite_state


def host_sample():
    if not os.environ.get('X_P2_NODE_IDENTITY') and os.environ.get('X_P2_SMOKE_ROLE'):
        from kit.x_smoke_memory import sample
        return sample()
    from kit.x_memory import sample
    return sample(node=bool(os.environ.get('X_P2_NODE_IDENTITY')))


def merge_host(previous,current):
    from kit.x_smoke_memory import SCOPE
    if current.get('scope')==SCOPE:
        from kit.x_smoke_memory import merge_peak
    else:
        from kit.x_memory import merge_peak
    return merge_peak(previous,current)


class Peaks:
    """Begin before native model/teacher/optimizer construction, not after it."""
    def __init__(self):
        import torch
        self.device=torch.cuda.current_device();self.device_used=0
        self.host=host_sample();self.stop=threading.Event();self.lock=threading.Lock();self.allocated=0;self.reserved=0
        self.sample_device()
        def sample():
            while not self.stop.wait(.1):
                self.sample_host()
                self.sample_device()
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
    def sample_host(self):
        with self.lock:
            self.host=merge_host(self.host,host_sample());return dict(self.host)
    def sample_device(self):
        import torch
        free,total=torch.cuda.mem_get_info(self.device)
        self.device_used=max(self.device_used,total-free)


def start(worker):
    if os.environ.get('X_P2_NODE_IDENTITY') or os.environ.get('X_P2_SMOKE_ROLE'):
        worker._x_p2_peaks=Peaks()


class Observer:
    def __init__(self,worker,anchor,identity):
        self.worker=worker;self.anchor=anchor;self.identity=identity;self.geometry=geometry(worker.actor.actor_module,anchor)
        self.peaks=worker._x_p2_peaks
    def snapshot(self,event):
        import torch
        p=self.peaks;p.sample_device()
        w=self.worker;props=torch.cuda.get_device_properties(torch.cuda.current_device())
        p.allocated=max(p.allocated,torch.cuda.max_memory_allocated());p.reserved=max(p.reserved,torch.cuda.max_memory_reserved())
        p.sample_host()
        value=dict(schema='rung-p2-node-rank.v1',rank=w.rank,world_size=torch.distributed.get_world_size(),
            identity=self.identity,geometry=self.geometry,event=event,completed_updates=getattr(w.actor,'_v4_completed_updates',0),
            gpu=dict(total_bytes=props.total_memory,allocated_bytes=p.allocated,reserved_bytes=p.reserved,device_used_bytes=p.device_used,
                device_used_scope='CUDA device used sampled at 100ms and boundaries; allocator lifetime peaks also retained',uuid=str(getattr(props,'uuid',''))),
            host=p.host,optimizer_materialised=optimizer_materialised(w.actor),finite_state=finite_state(w.actor),
            fp32_anchor=all(b.dtype==torch.float32 and not b.requires_grad for _,_,b in self.anchor.segments),
            teacher_included=sum(v.numel() for v in w.actor.teacher_module.parameters())==sum(v.numel() for v in w.actor.actor_module.parameters())>0,
            teacher_parameter_elements=sum(v.numel() for v in w.actor.teacher_module.parameters()),
            transients_included_asserted=True)
        atomic(Path(os.environ['X_P2_RUN'])/f'resource-r{w.rank}.json',value)
        return value


def install(worker,anchor):
    if not os.environ.get('X_P2_NODE_IDENTITY'):return
    path=Path(os.environ['X_P2_NODE_IDENTITY'])
    require(sha(path)==os.environ['X_P2_NODE_IDENTITY_SHA'],'node feasibility identity file changed')
    worker._x_p2_observer=Observer(worker,anchor,read(path))
    value=worker._x_p2_observer.snapshot('initialized')
    if os.environ['X_P2_MODE']=='measure':
        atomic(Path(os.environ['X_P2_BATCH'])/f'geometry-r{worker.rank}.json',value)


def snapshot(worker,event):
    if hasattr(worker,'_x_p2_observer'):return worker._x_p2_observer.snapshot(event)


def install_save(cls):
    if not (os.environ.get('X_P2_NODE_IDENTITY') or os.environ.get('X_P2_SMOKE_ROLE')) or getattr(cls,'_x_p2_resource_save',False):return
    import functools
    original=cls.save_checkpoint
    @functools.wraps(original)
    def saved(self,*args,**kwargs):
        if not os.environ.get('X_P2_NODE_IDENTITY') and os.environ.get('X_P2_SMOKE_ROLE'):
            peaks=self._x_p2_peaks
            peaks.sample_host()
            try:return original(self,*args,**kwargs)
            finally:
                host=peaks.sample_host()
                if hasattr(self,'_x_smoke'):
                    self._x_smoke['host']=host
                    atomic(Path(os.environ['X_P2_RUN'])/f'smoke-rank{self.rank}.json',self._x_smoke)
        result=original(self,*args,**kwargs);snapshot(self,'checkpoint-complete');return result
    cls.save_checkpoint=saved;cls._x_p2_resource_save=True
