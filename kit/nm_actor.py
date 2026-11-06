"""NM-only native actor: unchanged D task loss plus the registered single change."""
import types
import torch
from kit.nm_common import need,atomic
from kit import nm_loss as loss
from kit import sdft_objective as objective
from kit.sdft_actor import _model_forward,teacher_projection,projected_distillation,forward_micro_batch
from kit.m3_loss import replay_inputs,replay_loss


def reduce(values,device):
    out=torch.tensor(values,dtype=torch.float64,device=device)
    if torch.distributed.is_initialized():torch.distributed.all_reduce(out)
    return out.tolist()


def install_penalty(actor,anchor,value,measurement=False):
    """Reuse the pinned native clipping boundary, never a microbatch loss add-on."""
    from kit.x_p2_actor import before_clip
    original=actor._optimizer_step;actor._p2_native_step=original
    def step(self):
        def boundary():
            v=anchor.sums();self._nm_calibration=dict(incoming_distance=float(v[0].sqrt()),
                weighted_distance=float(v[1].sqrt()),task_grad_norm=float(v[3].sqrt()),
                source='first-training-batch',no_mutation=measurement)
            self._nm_penalty=dict(value=float(v[2]),coefficient=value)
            if measurement:
                from kit.x_p2_actor import MeasurementReady
                if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
                    atomic(self.nm_config['receipt'],dict(status='CALIBRATED',seed=self.nm_config['seed'],**self._nm_calibration))
                raise MeasurementReady()
            anchor.add_gradient(value)
        with before_clip(self,boundary):return original()
    actor._optimizer_step=types.MethodType(step,actor)


def update_policy(self,data):
    c=self.nm_config;world=self.nm_world_size;device=next(self.actor_module.parameters()).device
    rows=list(data.split(1));need(len(rows)==c['rows_per_update']//world,'incomplete registered accumulation')
    need(self.config.ppo_epochs==1 and self.ulysses_sequence_parallel_size==1,'one pass and no sequence parallelism required')
    objective.check_self_distillation(self.config.self_distillation,self.config.loss_agg_mode)
    grpo=c['family']=='grpo' and c['arm']=='T'
    telemetry=c['family']=='grpo'
    task_anchor=c['family']=='isdft' and c['arm']=='T'
    anchored=c.get('recipe')=='K'
    base=getattr(self,'frozen_base_module',None)
    if telemetry or task_anchor or anchored:
        need(base is not None and base is not self.actor_module and (grpo or base is not self.teacher_module),'separate original frozen base required')
        need(not base.training and all(not p.requires_grad and p.grad is None for p in base.parameters()),'base must be frozen/no-grad')
    self.actor_module.train();self.teacher_module.eval();self.actor_optimizer.zero_grad();totals=[0.,0.,0.,0.,0.,0.]
    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    update=getattr(self,'_v4_completed_updates',0)+1
    for index,row in enumerate(rows):
        row=row.to(device);x=row.batch;e=row.non_tensor_batch['extra_info'][0];replay=bool(e.get('m3_replay'))
        need(e['nm_update']==update and (c['balance_batch'] or e['nm_position']==rank+index*world+1),'schedule order/update changed')
        values=replay_inputs(x,e['m3_target_ids'],int(e['m3_eos_token_id'])) if replay else x
        width=x['responses'].shape[-1]
        if not replay and not grpo:
            need('self_distillation_mask' in x and bool((x['self_distillation_mask']==1).all()),'task demonstration target missing')
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            teacher=None
            if not grpo:
                teacher=teacher_projection(self.teacher_module,dict(responses=x['responses'],input_ids=x['teacher_input_ids'],
                    attention_mask=x['teacher_attention_mask'],position_ids=x['teacher_position_ids']))
            snapshot=teacher_projection(base,values) if anchored else None
            task_snapshot=teacher_projection(base,x) if task_anchor or telemetry else None
            if replay:
                value=replay_loss(self.actor_module,values,projection=snapshot if anchored else None)
            elif grpo:
                temperature=float(data.meta_info['temperature'])
                logps=_model_forward(self.actor_module,x,lambda h,w:-loss.projected_ce(h[:,-width-1:-1]/temperature,w,x['responses'],256))
                value=loss.grpo_tokens(logps,x['old_log_probs'],x['nm_advantage'],x['response_mask'])
            else:
                def head(hidden,weight,_teacher=teacher,_snapshot=task_snapshot):
                    hidden=hidden[:,-width-1:-1]
                    kl,_,_=projected_distillation(hidden,weight,_teacher['hidden'],_teacher['weight'],x['responses'],1.,256)
                    if task_anchor:
                        anchor=loss.projected_forward_kl(hidden,weight,_snapshot['hidden'],_snapshot['weight'],256)
                        return torch.stack((kl,anchor),-1)
                    return kl
                kl=_model_forward(self.actor_module,x,head)
                anchor=None
                if task_anchor:kl,anchor=kl[...,0],kl[...,1]
                mask=objective.loss_mask(x['response_mask']);count=mask.sum(-1).clamp(min=1)
                weights=objective.reference_importance_weights(x['old_log_probs'],x['rollout_log_probs'])
                value=((kl*mask).sum(-1)/count*(weights*mask).sum(-1)/count).mean()
                if task_anchor:
                    added=loss.sequence_mean(anchor,x['response_mask']);value=value+.10*added;totals[5]+=float(added.detach())
            # Task KL shares D's student forward. Replay ranks need only the
            # aligned read-only base dummy; no extra actor backward/collective.
            if telemetry:
                def base_head(hidden,weight,_snapshot=task_snapshot):
                    return loss.sequence_mean(loss.projected_forward_kl(hidden[:,-width-1:-1],weight,_snapshot['hidden'],_snapshot['weight'],256),x['response_mask'])
                with torch.no_grad():base_kl=_model_forward(self.actor_module,x,base_head)
                totals[2]+=float(base_kl)
            finite=bool(torch.isfinite(value))
            need(not reduce([int(not finite)],device)[0],'nonfinite loss on a rank')
            scaled=value*(world/32)
            (self.scaler.scale(scaled) if self.scaler is not None else scaled).backward()
            totals[int(replay)]+=float(value.detach());totals[3]+=int(x['response_mask'].sum());totals[4]+=x['input_ids'].numel()
        del teacher,snapshot,task_snapshot,value,scaled
    norm=self._optimizer_step();need(bool(torch.isfinite(torch.as_tensor(norm))),'nonfinite native gradient norm')
    if not grpo:self._update_teacher()
    self._v4_completed_updates=update
    if base is not None:need(all(p.grad is None for p in base.parameters()),'reference received gradient')
    totals=reduce(totals,device)
    metrics={'actor/pg_loss':totals[0]/32,'nm/replay_loss':totals[1]/32,'actor/grad_norm':float(norm),
             'v4/completed_optimizer_updates':update,'nm/global_divisor':32,'nm/response_tokens':totals[3],
             'nm/actor_padded_tokens':totals[4]}
    if task_anchor:metrics['nm/task_base_KL']=totals[5]/32
    if telemetry:
        # Global reward/group telemetry was computed on the complete batch before rank dispatch.
        metrics.update({k:float(v) for k,v in data.meta_info['nm_telemetry'].items()})
        metrics['nm/base_forward_KL']=totals[2]/32
    if hasattr(self,'_nm_penalty'):metrics['nm/penalty']=self._nm_penalty['value']
    self._nm_history=getattr(self,'_nm_history',[])+[dict(metrics)]
    self.actor_optimizer.zero_grad();return metrics


def actor_class(parent):
    class NMActor(parent):
        _forward_micro_batch=forward_micro_batch
        update_policy=update_policy
    return NMActor
