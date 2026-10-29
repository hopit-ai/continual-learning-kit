"""X-only hook at D's existing post-unscale/pre-clip optimizer boundary."""
from contextlib import contextmanager
import functools
import types
import time
import math
from kit.p2_l2 import Anchor
from kit.s_common import require


class MeasurementReady(Exception):pass


@contextmanager
def before_clip(actor,callback):
    """Keep D's scaler/clipping/optimizer implementation and ordering unchanged."""
    import torch
    module=actor.actor_module
    namespace=None
    if hasattr(module,'clip_grad_norm_'):
        owner=module;original=owner.clip_grad_norm_
    elif any(hasattr(p,'to_local') for p in module.parameters()):
        # D's FSDP2 branch calls its imported helper, not torch.nn.utils.
        namespace=actor._p2_native_step.__func__.__globals__
        require('fsdp2_clip_grad_norm_' in namespace,'P2 missing native FSDP2 clip boundary')
        original=namespace['fsdp2_clip_grad_norm_']
    else:owner=torch.nn.utils;original=owner.clip_grad_norm_
    called=[]
    @functools.wraps(original)
    def clip(*args,**kwargs):
        require(not called,'P2 more than one clip per optimizer update');called.append(True)
        callback()
        return original(*args,**kwargs)
    if namespace is None:owner.clip_grad_norm_=clip
    else:namespace['fsdp2_clip_grad_norm_']=clip
    try:yield
    finally:
        if namespace is None:owner.clip_grad_norm_=original
        else:namespace['fsdp2_clip_grad_norm_']=original


def extend(actor,anchor,value=0.,measurement=False,expected=None,audit=False):
    """Only the X P2 worker installs this; teacher is never an anchor member."""
    require(not getattr(actor,'_p2_extended',False),'P2 actor already extended')
    teacher_ids={id(p) for p in actor.teacher_module.parameters()}
    require(all(id(p) not in teacher_ids for p,_,_ in anchor.segments),'P2 teacher regularisation forbidden')
    original=actor._optimizer_step;actor._p2_native_step=original;actor._p2_records=[];actor._p2_anchor=anchor
    def step(self):
        def boundary():
            if audit:
                import torch
                if any(p.is_cuda for p,_,_ in anchor.segments):torch.cuda.synchronize()
            started=time.monotonic()
            sums=anchor.sums();distance,task=sums.sqrt().tolist()
            sums_seconds=time.monotonic()-started
            record=dict(distance2=float(sums[0]),regulariser=float(sums[0])/2,
                incoming_distance=distance,task_grad_norm=task,lambda_value=value,
                regulariser_grad_norm=value*distance,after_reduction_unscaling=True,before_clipping=True)
            if expected is not None and not self._p2_records:
                from kit.p2_l2 import deviation
                record['calibration_deviations']=dict(task_gradient=deviation(task,expected['task_grad_norm']),
                    incoming_distance=deviation(distance,expected['incoming_distance']))
            record['regulariser_to_task_ratio']=value*distance/task if task>0 else None
            self._p2_records.append(record)
            if measurement:
                record['boundary_seconds']=time.monotonic()-started
                raise MeasurementReady()
            record['addition_skipped_nonfinite_task_gradient']=not math.isfinite(task)
            if math.isfinite(task):
                if audit:
                    from kit.x_p2_state import audit_add
                    record.update(audit_add(anchor,value))
                    record['audit_excluded_seconds']+=sums_seconds
                else:anchor.add_gradient(value)
            record['boundary_seconds']=time.monotonic()-started
        with before_clip(self,boundary):norm=original()
        import torch
        # D owns nonfinite-step skipping; retain the same behavior for P2.
        require(self._p2_records,'P2 native clipping hook was not reached')
        self._p2_records[-1].update(combined_preclip_grad_norm=float(norm),
            combined_to_task_ratio=(float(norm)/self._p2_records[-1]['task_grad_norm'] if self._p2_records[-1]['task_grad_norm']>0 else None),
            native_nonfinite_skip=not bool(torch.isfinite(norm)))
        if audit:
            require(math.isclose(float(norm)**2,self._p2_records[-1]['combined_squared'],rel_tol=2e-5,abs_tol=1e-10),'native clipping and fp64 distributed preclip norm differ')
        return norm
    actor._optimizer_step=types.MethodType(step,actor);actor._p2_extended=True
    return actor
