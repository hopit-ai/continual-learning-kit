"""Isolated M3 actor hooks; the D actor and its moving teacher remain untouched."""
from __future__ import annotations
import hashlib
import time
import os
import functools
from contextlib import contextmanager, nullcontext
from pathlib import Path
import torch
from kit import sdft_objective as objective
from kit.sdft_actor import (_model_forward, teacher_projection, projected_distillation,
                            forward_micro_batch, ema_movement)
from kit.m3_contract import geometry, CR_LAYOUT, STAGES, stage_number, code_digest, verify_base_files
from kit.m3_loss import replay_inputs, replay_loss


def freeze_reference(module):
    module.eval()
    module.requires_grad_(False)
    for value in module.parameters():
        value.grad = None
    return module


def reference_hash(module):
    """Hash actual local shard bytes; rank-ordered digests cover the full base."""
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        local = value.to_local() if hasattr(value, 'to_local') else value
        digest.update(name.encode())
        digest.update(str((tuple(local.shape), local.dtype)).encode())
        digest.update(local.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    result = digest.hexdigest()
    if torch.distributed.is_initialized():
        hashes = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(hashes, result)
        result = hashlib.sha256(''.join(hashes).encode()).hexdigest()
    return result


def load_frozen_base(directory, revision, mesh, *, binding, attention_implementation='flash_attention_2',
                     model_loader=None, shard=None, config_loader=None, model_factory=None, state_loader=None):
    """Rank-zero CPU load, meta construction elsewhere, then sharded broadcast.

    Config/safetensors are verified against inputs/base.json before loading.
    No rank materializes a full base on its GPU, and only rank zero reads weights.
    Injected factories permit fixture-only CPU coverage of the same loading plan.
    """
    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    payload=[None]
    if rank==0:
        try: payload[0]=dict(digest=verify_base_files(directory,binding,revision))
        except (OSError,ValueError,KeyError): payload[0]=dict(error=True)
    if torch.distributed.is_initialized(): torch.distributed.broadcast_object_list(payload,src=0)
    if payload[0].get('error'): raise ValueError('frozen base identity failed before load')
    if attention_implementation not in ('flash_attention_2','sdpa'):
        raise ValueError('reference requires the actor resolved memory-efficient attention implementation')
    if model_loader is None:
        from transformers import AutoModelForCausalLM
        model_loader=AutoModelForCausalLM.from_pretrained
    kwargs=dict(torch_dtype=torch.bfloat16,trust_remote_code=False,attn_implementation=attention_implementation)
    if rank==0:
        module=model_loader(str(directory),revision=revision,local_files_only=True,
                            low_cpu_mem_usage=True,device_map='cpu',**kwargs)
        full_state=module.state_dict()
    else:
        if config_loader is None:
            from transformers import AutoConfig
            config_loader=AutoConfig.from_pretrained
        if model_factory is None:
            from transformers import AutoModelForCausalLM
            model_factory=AutoModelForCausalLM.from_config
        config=config_loader(str(directory),local_files_only=True,trust_remote_code=False)
        with torch.device('meta'): module=model_factory(config,**kwargs)
        full_state={}
    freeze_reference(module)
    if any(value.dtype!=torch.bfloat16 for value in module.parameters()):
        raise ValueError('reference loader did not honor explicit bf16 dtype')
    if shard is None:
        if not torch.distributed.is_initialized() or not torch.cuda.is_available():
            raise RuntimeError('production frozen reference requires distributed CUDA FSDP2')
        from verl.utils.fsdp_utils import apply_fsdp2,MixedPrecisionPolicy,fsdp2_load_full_state_dict
        apply_fsdp2(module,dict(mesh=mesh,mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=True),reshard_after_forward=True),{})
        (state_loader or fsdp2_load_full_state_dict)(module,full_state,mesh,None)
    else:
        module=shard(module,mesh)
        if state_loader is not None: state_loader(module,full_state,mesh,None)
    module._m3_files_sha256=payload[0]['digest']
    module._m3_attention_implementation=attention_implementation
    del full_state
    return freeze_reference(module)


def count_scheduler_steps(scheduler, on_step):
    """Instrument the instance through a subclass, preserving serializable state.

    Assigning an instance-level `step` closure would put that callable inside
    PyTorch's scheduler state_dict and break native checkpoint serialization.
    """
    class M3Scheduler(type(scheduler)):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            on_step()
            return result
    scheduler.__class__ = M3Scheduler
    return scheduler


def checkpoint_metadata(actor, *, optimizer_steps, ema_steps, scheduler_steps, frozen_base):
    """M3 metadata saved beside native actor/optimizer/scheduler and EMA shards."""
    return dict(completed_updates=getattr(actor,'_v4_completed_updates',0),optimizer_steps=optimizer_steps,
        ema_steps=ema_steps,scheduler_steps=scheduler_steps,frozen_base=frozen_base,code_sha256=code_digest(),
        oracle_completed=getattr(actor,'_m3_oracle_completed',False),oracle_error=getattr(actor,'_m3_oracle_error',None),
        oracle_update=getattr(actor,'_m3_oracle_update',None),
        **{k:actor.m3_config[k] for k in ('arm','stage','seed','base_revision','profile')})


def validate_resume_metadata(actor, state, frozen_base):
    for key in ('arm','stage','seed','base_revision','profile'):
        if state.get(key)!=actor.m3_config[key]: raise ValueError('resume identity mismatch')
    if state.get('frozen_base')!=frozen_base or state.get('code_sha256')!=code_digest():
        raise ValueError('resume frozen-base/code identity mismatch')
    steps=[state.get(k) for k in ('completed_updates','optimizer_steps','ema_steps','scheduler_steps')]
    if any(type(value) is not int or value<0 for value in steps) or len(set(steps))!=1:
        raise ValueError('resume step counters are inconsistent')
    return state


def _all_reduce(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor)
    return tensor.cpu().tolist()


def independent_gradient_oracle(module, raw_backward):
    """Bypass native reductions, SUM fp32 full gradients / literal 32, then shard.

    The reference never reads geometry/local_loss_scale or uses reduce-scatter.
    FSDP2 retains unsharded gradients under set_requires_gradient_sync(False).
    DDP's no_sync supports the real CPU gloo test of the same SUM/32 oracle.
    """
    from contextlib import nullcontext
    distributed=torch.distributed.is_initialized()
    parameters=list(module.parameters())
    def local(t):return t.to_local() if hasattr(t,'to_local') else t
    observed=[None if p.grad is None else local(p.grad).detach().cpu().clone() for p in parameters]
    module.zero_grad(set_to_none=True)
    fsdp=hasattr(module,'set_requires_gradient_sync')
    groups=[]
    if fsdp:
        module.set_requires_gradient_sync(False)
        seen=set()
        for child in module.modules():
            if not hasattr(child,'_get_fsdp_state'):continue
            group=child._get_fsdp_state()._fsdp_param_group
            if group is not None and id(group) not in seen:groups.append(group);seen.add(id(group))
    context=module.no_sync() if not fsdp and hasattr(module,'no_sync') else nullcontext()
    distances=[0.,0.]
    try:
        with context: raw_backward()
        entries={id(fp.sharded_param):fp for group in groups for fp in group.fsdp_params}
        for p,old in zip(parameters,observed):
            fp=entries.get(id(p))
            raw=(fp.unsharded_accumulated_grad if fp is not None else p.grad)
            if fp is not None and raw is None:
                unsharded=getattr(fp,'_unsharded_param',None)
                raw=unsharded.grad if unsharded is not None else None
            present=torch.tensor([int(raw is not None),int(old is not None)],device=p.device,dtype=torch.int64)
            if distributed:torch.distributed.all_reduce(present,op=torch.distributed.ReduceOp.SUM)
            world=torch.distributed.get_world_size() if distributed else 1
            if tuple(present.tolist())==(0,0):continue
            if tuple(present.tolist())!=(world,world):
                raise RuntimeError('independent gradient parameter participation differs')
            reference=local(raw).detach().float()
            if tuple(reference.shape)!=tuple(p.shape):raise RuntimeError('reference gradient is not unsharded')
            if distributed:torch.distributed.all_reduce(reference,op=torch.distributed.ReduceOp.SUM)
            reference.div_(32.)  # Literal registered global denominator; independent of production geometry.
            if hasattr(p,'placements'):
                if len(p.placements)!=1 or not hasattr(p.placements[0],'dim'):
                    raise RuntimeError('oracle requires the registered one-dimensional FSDP shard mesh')
                dim=p.placements[0].dim
                rank=p.device_mesh.get_local_rank()
                count=p.device_mesh.size()
                chunk=(p.shape[dim]+count-1)//count
                offset=min(rank*chunk,p.shape[dim]);size=min(chunk,p.shape[dim]-offset)
                reference=reference.narrow(dim,offset,size)
            if reference.shape!=old.shape:raise RuntimeError('oracle DTensor shard shape differs')
            flat=reference.reshape(-1)
            for start in range(0,flat.numel(),65536):
                expected=flat[start:start+65536].double()
                actual=old.reshape(-1)[start:start+65536].to(expected.device).double()
                distances[0]+=float((actual-expected).square().sum())
                distances[1]+=float(expected.square().sum())
    finally:
        for group in groups:
            for fp in group.fsdp_params:
                fp.unsharded_accumulated_grad=None
                if getattr(fp,'_unsharded_param',None) is not None:fp._unsharded_param.grad=None
        if fsdp:module.set_requires_gradient_sync(True)
        module.zero_grad(set_to_none=True)
        for p,old in zip(parameters,observed):
            if old is not None:
                p.grad=torch.zeros_like(p)
                local(p.grad).copy_(old.to(p.device))
    totals=_all_reduce(distances,next(module.parameters()).device)
    error=(totals[0]/max(totals[1],1e-30))**.5
    if not error<=1e-4:raise RuntimeError('independent SUM/32 gradient relative error exceeds 1e-4')
    return error



@contextmanager
def training_determinism(enabled):
    """Preflight training only; restore flags before in-process vLLM can run."""
    if not enabled:
        yield
        return
    state=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.backends.cuda.matmul.allow_tf32)
    flash=os.environ.get('FLASH_ATTENTION_DETERMINISTIC')
    try:
        torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;os.environ['FLASH_ATTENTION_DETERMINISTIC']='1'
        yield
    finally:
        torch.use_deterministic_algorithms(state[0],warn_only=state[1])
        torch.backends.cudnn.deterministic=state[2];torch.backends.cudnn.benchmark=state[3]
        torch.backends.cuda.matmul.allow_tf32=state[4]
        if flash is None:os.environ.pop('FLASH_ATTENTION_DETERMINISTIC',None)
        else:os.environ['FLASH_ATTENTION_DETERMINISTIC']=flash


def preflight_training_only(function):
    @functools.wraps(function)
    def wrapped(self,*args,**kwargs):
        with training_determinism(self.m3_config['profile']=='preflight'):
            return function(self,*args,**kwargs)
    return wrapped


@preflight_training_only
def update_policy(self, data):
    """One complete update: aligned reference collectives, global sum/32, one EMA.

    Workers attach m3_config and (K only) frozen_base_module after native init.
    Native worker owns the one scheduler step following this hook.
    """
    if torch.cuda.is_available(): torch.cuda.synchronize()
    update_started = time.monotonic()
    objective.check_self_distillation(self.config.self_distillation, self.config.loss_agg_mode)
    if self.ulysses_sequence_parallel_size != 1 or self.config.ppo_epochs != 1:
        raise ValueError('M3 requires one epoch and no sequence parallelism')
    if self.teacher_module is None or self.teacher_module is self.actor_module:
        raise ValueError('a separate incoming-initialized EMA teacher is required')
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else self.m3_world_size
    shape = geometry(self.m3_config['arm'], self.m3_config['stage'], world)
    rows = list(data.split(1))
    device = next(self.actor_module.parameters()).device
    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    violations=int(len(rows)!=shape['accumulation_rounds'])
    flags=[bool(row.non_tensor_batch['extra_info'][0].get('m3_replay')) for row in rows]
    domains=('instruction','gsm8k','arc','finqa','contract_nli')
    expected_domains=({'instruction':4} if self.m3_config['arm'] in ('C0','K') else
                      {'instruction':shape['replay_rows']} if self.m3_config['arm']=='G' else
                      CR_LAYOUT[STAGES[stage_number(self.m3_config['stage'])-1]])
    local_domains=[]
    for index,(row,replay) in enumerate(zip(rows,flags)):
        extra=row.non_tensor_batch['extra_info'][0]
        required=self.m3_config['profile'] in ('scientific','preflight')
        violations+=int(required and any(k not in extra for k in ('m3_update','m3_position')))
        violations+=int('m3_update' in extra and extra['m3_update']!=getattr(self,'_v4_completed_updates',0)+1)
        violations+=int('m3_position' in extra and extra['m3_position']!=rank+index*world+1)
        if replay:
            local_domains.append(extra.get('m3_domain'))
            try:replay_inputs(row.batch,extra['m3_target_ids'],int(extra['m3_eos_token_id']))
            except (KeyError,ValueError,TypeError):violations+=1
        else:violations+=int(bool((row.batch['self_distillation_mask']!=1).any()))
    violations+=sum(d not in domains for d in local_domains)
    counts=_all_reduce([len(rows)-sum(flags),sum(flags)]+[local_domains.count(d) for d in domains]+[violations],device)
    if counts[-1] or counts[:-1]!=[shape['task_rows'],shape['replay_rows']]+[expected_domains.get(d,0) for d in domains]:
        raise ValueError('a rank has invalid mandatory schedule metadata, targets or global row/domain counts')
    anchored = self.m3_config['arm'] == 'K'
    base = getattr(self, 'frozen_base_module', None)
    if anchored and (base is None or base is self.teacher_module or base is self.actor_module):
        raise ValueError('K needs a third, independently frozen base module')
    if base is not None:
        if base.training or any(p.requires_grad or p.grad is not None for p in base.parameters()):
            raise ValueError('frozen base is trainable or has gradients')
    self.actor_module.train()
    self.teacher_module.eval()
    self.actor_optimizer.zero_grad()
    losses = [0., 0.]
    tokens = [0, 0, 0, 0, 0, 0]
    reference_seconds=[0.,0.]
    padded=[0,0,0,0]
    initial_kl=[0.,0.]
    temperature = float(data.meta_info['temperature'])
    if temperature != 1.:
        raise ValueError('registered temperature is one')
    def row_loss(row, replay):
        row = row.to(device)
        inputs = row.batch
        extra = row.non_tensor_batch['extra_info'][0]
        values = replay_inputs(inputs, extra['m3_target_ids'], int(extra['m3_eos_token_id'])) if replay else inputs
        # Same module order/call count on every rank, including discarded dummies.
        teacher_inputs = {'responses': inputs['responses'], 'input_ids': inputs['teacher_input_ids'],
                          'attention_mask': inputs['teacher_attention_mask'], 'position_ids': inputs['teacher_position_ids']}
        def timed_projection(model, forwarded, slot):
            if device.type=='cuda': torch.cuda.synchronize(device)
            started=time.monotonic()
            result=teacher_projection(model,forwarded)
            if device.type=='cuda': torch.cuda.synchronize(device)
            reference_seconds[slot]+=time.monotonic()-started
            return result
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            teacher = timed_projection(self.teacher_module,teacher_inputs,0)
            projection = timed_projection(base,values,1) if anchored else None
        if replay:
            del teacher
            loss = replay_loss(self.actor_module, values, projection=projection)
        else:
            # K's base dummy uses the plain task prompt, never the demonstration.
            del projection
            if bool((inputs['self_distillation_mask'] != 1).any()):
                raise ValueError('task demonstration target missing')
            width = inputs['responses'].shape[-1]
            def head(hidden, weight, snapshot=teacher):
                kl, _, _ = projected_distillation(hidden[:, -width-1:-1], weight,
                    snapshot['hidden'], snapshot['weight'], inputs['responses'], temperature, 256)
                return kl
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                kl = _model_forward(self.actor_module, inputs, head)
            mask = objective.loss_mask(inputs['response_mask'])
            count = mask.sum(-1).clamp(min=1)
            weights = objective.reference_importance_weights(inputs['old_log_probs'], inputs['rollout_log_probs'])
            sequence_weight = (weights * mask).sum(-1) / count
            loss = ((kl * mask).sum(-1) / count * sequence_weight).mean()
            del teacher, kl
        return (loss,int(values['attention_mask'].sum()),
                int(values['response_mask'].sum()) if replay else int(inputs['response_mask'].sum()),
                int(teacher_inputs['attention_mask'].sum()),int(values['attention_mask'].sum()) if anchored else 0,
                int(inputs['attention_mask'].sum()),values['input_ids'].numel(),teacher_inputs['input_ids'].numel(),
                values['input_ids'].numel() if anchored else 0,inputs['input_ids'].numel())

    for row, replay in zip(rows, flags):
        loss,scheduled,response,ema_tokens,base_tokens,rollout_tokens,actor_padded,ema_padded,base_padded,rollout_padded=row_loss(row,replay)
        if _all_reduce([int(not bool(torch.isfinite(loss)))],device)[0]:
            raise FloatingPointError('nonfinite M3 loss on a rank; all ranks stop before backward')
        tokens[2] += scheduled
        tokens[int(replay)] += response
        tokens[3]+=ema_tokens
        tokens[4]+=base_tokens
        tokens[5]+=rollout_tokens
        for i,value in enumerate((actor_padded,ema_padded,base_padded,rollout_padded)):padded[i]+=value
        if anchored and replay and getattr(self,'_v4_completed_updates',0)==0:
            initial_kl[0]+=float(loss.detach())*response;initial_kl[1]+=response
        # DDP mean contributes 1/world. Never divide by 5/6/7 rounds.
        scaled = loss * shape['local_loss_scale']
        assert shape['local_loss_scale'] / world == 1 / 32
        (self.scaler.scale(scaled) if self.scaler is not None else scaled).backward()
        losses[int(replay)] += float(loss.detach())
        del loss, scaled
    initial_global=_all_reduce(initial_kl,device)
    initial_kl_mean=initial_global[0]/max(initial_global[1],1)
    if anchored and getattr(self,'_v4_completed_updates',0)==0 and self.m3_config['profile'] in ('technical-smoke','preflight'):
        if not -1e-5<=initial_kl_mean<=1e-3:raise RuntimeError('equal-base replay KL exceeds 1e-3 nats/token')
    oracle_error = None
    run_oracle=(self.m3_config['profile'] in ('technical-smoke','preflight')
                and self.actor_optimizer.param_groups[0]['lr']>0
                and not getattr(self,'_m3_oracle_completed',False))
    if run_oracle:
        if self.scaler is not None:raise ValueError('independent oracle requires bf16 without gradient scaling')
        def raw_backward():
            for row,replay in zip(rows,flags):
                raw,*_=row_loss(row,replay)
                finite=bool(torch.isfinite(raw).item())
                if _all_reduce([int(not finite)],device)[0]:raise FloatingPointError('nonfinite oracle loss on a rank')
                raw.backward()
        tracker=getattr(self,'_m3_memory_tracker',None)
        with tracker.oracle_phase() if tracker is not None else nullcontext():
            oracle_error=independent_gradient_oracle(self.actor_module,raw_backward)
        self._m3_oracle_completed=True
        self._m3_oracle_error=oracle_error
        self._m3_oracle_update=getattr(self,'_v4_completed_updates',0)+1
    norm = self._optimizer_step()
    if not bool(torch.isfinite(torch.as_tensor(norm))):
        raise FloatingPointError('nonfinite M3 gradient norm')
    _, movement = ema_movement(self, self._update_teacher)
    self._v4_completed_updates = getattr(self, '_v4_completed_updates', 0) + 1
    oracle_passes=2 if run_oracle else 1
    tokens[3]*=oracle_passes; tokens[4]*=oracle_passes
    for i in (0,1,2):padded[i]*=oracle_passes
    sums = _all_reduce(losses+tokens+reference_seconds+padded,device)
    self.actor_optimizer.zero_grad()
    if base is not None and any(p.grad is not None for p in base.parameters()):
        raise RuntimeError('frozen base received gradients')
    metrics = {'actor/pg_loss': sums[0]/32, 'm3/replay_loss': sums[1]/32,
            'm3/combined_loss': (sums[0]+sums[1])/32, 'actor/grad_norm': float(norm),
            'm3/task_tokens': sums[2], 'm3/replay_tokens': sums[3], 'm3/scheduled_tokens': sums[4],
            'm3/ema_forward_tokens':sums[5],'m3/base_forward_tokens':sums[6],
            'm3/rollout_input_tokens':sums[7],
            'm3/ema_forward_seconds_rank_sum':sums[8],'m3/base_forward_seconds_rank_sum':sums[9],
            'm3/ema_forward_microsteps':len(rows)*world*oracle_passes,
            'm3/base_forward_microsteps':len(rows)*world*oracle_passes if anchored else 0,
            'm3/actor_padded_tokens':sums[10],'m3/ema_padded_tokens':sums[11],
            'm3/base_padded_tokens':sums[12],'m3/rollout_padded_tokens':sums[13],
            'm3/global_rows': shape['global_rows'], 'm3/accumulation_rounds': len(rows),
            'm3/global_divisor': 32, 'm3/completed_updates': self._v4_completed_updates,
            'v4/completed_optimizer_updates': self._v4_completed_updates, **movement}
    if torch.cuda.is_available(): torch.cuda.synchronize()
    metrics['m3/update_seconds'] = time.monotonic()-update_started
    if anchored and initial_global[1]>0:metrics['m3/initial_replay_kl_token_mean']=initial_kl_mean
    if oracle_error is not None:
        metrics['m3/gradient_relative_error'] = oracle_error
    self._m3_history = getattr(self, '_m3_history', []) + [dict(metrics,
        learning_rate=self.actor_optimizer.param_groups[0]['lr'])]
    return metrics


def actor_class(parent):
    """Create an M3 subclass without replacing methods on the shared D class."""
    class M3Actor(parent):
        _forward_micro_batch = forward_micro_batch
        update_policy = update_policy
    return M3Actor



def state_digest(value):
    """Canonical tensor/state hash without pickle metadata, paths or identities."""
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            local=item.to_local() if hasattr(item,'to_local') else item
            digest.update(str((tuple(local.shape),local.dtype)).encode())
            digest.update(local.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=lambda k:(type(k).__name__,str(k))):visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:visit(child)
        elif item is None or isinstance(item,(int,float,str,bool)):
            digest.update(repr(item).encode())
        else:raise ValueError('unsupported native state value for equality fingerprint')
    visit(value)
    result=digest.hexdigest()
    if torch.distributed.is_initialized():
        values=[None]*torch.distributed.get_world_size()
        torch.distributed.all_gather_object(values,result)
        result=hashlib.sha256(''.join(values).encode()).hexdigest()
    return result
