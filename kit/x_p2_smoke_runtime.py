"""Real FSDP2 smoke measurements on the production X worker/actor boundary."""
from pathlib import Path
import json,math,os,time
from kit.s_common import ROOT,read,atomic,require,sha
from kit.p2_l2 import load_anchor,local
from kit.x_p2_state import distance,geometry,finite_state,optimizer_materialised
from kit.v4_restore_check import state_hash
from kit.x_p2_resources import host_sample

_CONFIGURED=None
SETTINGS=ROOT/'kit/resources/p2/smoke.json'


def configure():
    if os.environ.get('X_P2_DETERMINISTIC_SMOKE')!='1':return
    require(os.environ.get('X_INTEGRATION_SMOKE')=='development-only' and os.environ.get('V4_PROFILE')=='technical-smoke','strict smoke settings cannot enter node arms')
    from kit.p_determinism import configure as strict
    global _CONFIGURED
    _CONFIGURED=strict(read(SETTINGS));return _CONFIGURED


def initialize(worker):
    import torch
    actor=worker.actor
    if not getattr(actor,'_p2_extended',False):
        from kit.x_p2_actor import extend
        extend(actor,load_anchor(actor.actor_module,os.environ['X_P2_ANCHOR']),0.,audit=True)
    # Keep an audit anchor, never a second whole-model GPU copy.
    anchor=getattr(actor,'_p2_anchor',None)
    require(anchor is not None,'smoke lacks actual regularizer anchor')
    require(torch.distributed.get_world_size()==2,'registered smoke requires exactly two ranks')
    require(all(local(p).dtype==torch.float32 for p in actor.actor_module.parameters() if p.requires_grad),'production path cannot supply fp32 master precision')
    initial=dict(student=state_hash({n:local(p) for n,p in actor.actor_module.named_parameters()}),
        teacher=state_hash({n:local(p) for n,p in actor.teacher_module.named_parameters()}))
    worker._x_smoke=dict(schema='rung-p2-live-smoke-rank.v1',rank=worker.rank,world_size=2,
        attention_implementation=getattr(getattr(actor.actor_module,'config',None),'_attn_implementation',None),
        deterministic_configuration=_CONFIGURED,role=os.environ['X_P2_SMOKE_ROLE'],initial=initial,geometry=geometry(actor.actor_module,anchor),
        distance_trajectory=[distance(anchor)],updates=[],base_versions=[b._version for _,_,b in anchor.segments],
        settings_sha256=sha(SETTINGS),precision='fp32-masters/fp64-accumulation')
    worker._x_smoke_anchor=anchor


def updated(worker,result,started):
    import torch
    # Native RPC includes reduction, optimizer, scheduler and EMA, and preserves
    # D's timing boundary. Synchronize before recording the completed update.
    torch.cuda.synchronize();seconds=time.monotonic()-started;record=worker._x_smoke;actor=worker.actor;anchor=worker._x_smoke_anchor
    metrics=result.meta_info['metrics']
    def number(value):
        if isinstance(value,(tuple,list)):return sum(map(float,value))/len(value)
        return float(value)
    task=number(metrics['actor/pg_loss'])+number(metrics.get('p3b/replay_sft_loss',0.))
    require(math.isfinite(task) and finite_state(actor) and optimizer_materialised(actor),'nonfinite or non-fp32 materialised smoke state')
    require(all(p.grad is None for p in actor.teacher_module.parameters()),'teacher gradients present')
    require(record['base_versions']==[b._version for _,_,b in anchor.segments],'frozen smoke base mutated')
    update=dict(actor._p2_records[-1],seconds=seconds-actor._p2_records[-1]['audit_excluded_seconds'],
        full_seconds=seconds,task_loss=task,
        total_loss=task+actor._p2_records[-1]['lambda_value']*actor._p2_records[-1]['regulariser'],
        finite=True,lr=number(metrics['actor/lr']),optimizer_materialised=True)
    record['updates'].append(update);record['distance_trajectory'].append(distance(anchor))
    props=torch.cuda.get_device_properties(torch.cuda.current_device())
    free,total=torch.cuda.mem_get_info()
    used=max(record.get('gpu',{}).get('device_used_bytes',0),total-free)
    peaks=getattr(worker,'_x_p2_peaks',None)
    if peaks is not None:
        peaks.sample_device();used=max(used,peaks.device_used)
    record.update(gpu=dict(device_used_bytes=used,reserved_bytes=torch.cuda.max_memory_reserved(),allocated_bytes=torch.cuda.max_memory_allocated(),total_bytes=props.total_memory),
        host=peaks.sample_host() if peaks is not None else host_sample(),base_unchanged=True,teacher_gradients_absent=True,teacher_regularised=False)
    atomic(Path(os.environ['X_P2_RUN'])/f'smoke-rank{worker.rank}.json',record)

    if os.environ.get('X_P2_EARLY_VERDICT')=='1' and len(record['updates'])==4:
        from kit.x_p2_export import live_student
        live_student(worker)
        # Preserve live-export transients in the same registered resource gate.
        if peaks is not None:
            peaks.sample_device();record['host']=peaks.sample_host()
            record['gpu'].update(device_used_bytes=peaks.device_used,
                reserved_bytes=torch.cuda.max_memory_reserved(),allocated_bytes=torch.cuda.max_memory_allocated())
            atomic(Path(os.environ['X_P2_RUN'])/f'smoke-rank{worker.rank}.json',record)
