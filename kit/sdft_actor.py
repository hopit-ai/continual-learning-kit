"""D's bounded vocabulary path, executed inside model-head/FSDP forward hooks.

Only hidden states, parameter-sized weights/gradients and [B,T] statistics survive
between chunks. No full response/prompt vocabulary tensor is returned or saved.
The teacher head captures unsharded weights while its FSDP forward is active;
the student's custom autograd returns gradients through its normal head hooks.
"""
from contextlib import contextmanager
import torch
from kit import sdft_objective as objective


class _ProjectedDistillation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, teacher_hidden, teacher_weight, targets, temperature, chunk):
        teacher_hidden, teacher_weight = teacher_hidden.detach(), teacher_weight.detach()
        shape = hidden.shape[:-1]
        kl = torch.empty(shape, dtype=torch.float32, device=hidden.device)
        logps, entropy = torch.empty_like(kl), torch.empty_like(kl)
        for start in range(0, shape[1], chunk):
            end = start + chunk
            s = torch.log_softmax((hidden[:, start:end] @ weight.T).float() / temperature, -1)
            t = torch.log_softmax((teacher_hidden[:, start:end] @ teacher_weight.T).float() / temperature, -1)
            kl[:, start:end] = (t.exp() * (t - s)).sum(-1)
            logps[:, start:end] = s.gather(-1, targets[:, start:end, None]).squeeze(-1)
            entropy[:, start:end] = -(s.exp() * s).sum(-1)
        ctx.save_for_backward(hidden, weight, teacher_hidden, teacher_weight)
        ctx.temperature, ctx.chunk = temperature, chunk
        ctx.mark_non_differentiable(logps, entropy)
        return kl, logps, entropy

    @staticmethod
    def backward(ctx, grad_kl, grad_logps, grad_entropy):
        hidden, weight, teacher_hidden, teacher_weight = ctx.saved_tensors
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        for start in range(0, hidden.shape[1], ctx.chunk):
            end = start + ctx.chunk
            h = hidden[:, start:end]
            # Float32 vocabulary intermediates exist ONLY for this token chunk.
            s = torch.softmax((h @ weight.T).float() / ctx.temperature, -1)
            t = torch.softmax((teacher_hidden[:, start:end] @ teacher_weight.T).float() / ctx.temperature, -1)
            g = (s - t) * (grad_kl[:, start:end, None] / ctx.temperature)
            # Avoid autocast rounding gradients until they return to the head's dtype.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                grad_hidden[:, start:end] = (g @ weight.float()).to(hidden.dtype)
                grad_weight.add_((g.reshape(-1, g.shape[-1]).T @ h.float().reshape(-1, h.shape[-1])))
        return grad_hidden, grad_weight.to(weight.dtype), None, None, None, None, None


def projected_distillation(hidden, weight, teacher_hidden, teacher_weight, targets, temperature=1., chunk_tokens=256):
    if chunk_tokens < 1 or temperature <= 0:
        raise ValueError('positive chunk size and temperature required')
    if hidden.shape != teacher_hidden.shape or weight.shape != teacher_weight.shape or targets.shape != hidden.shape[:-1]:
        raise ValueError('student/teacher projection shapes differ')
    return _ProjectedDistillation.apply(hidden, weight, teacher_hidden, teacher_weight, targets, float(temperature), int(chunk_tokens))


@contextmanager
def _head_forward(model, function):
    head = model.get_output_embeddings()
    if head is None or getattr(head, 'bias', None) is not None:
        raise ValueError('D requires a bias-free Qwen output head')
    # Preserve the instance/class binding exactly; do not alter pinned model source.
    had_override = 'forward' in head.__dict__
    previous = head.forward
    head.forward = lambda hidden: function(hidden, head.weight)
    try:
        yield
    finally:
        if had_override: head.forward = previous
        else: del head.forward


def _model_forward(model, inputs, function):
    if inputs['position_ids'].ndim != 2:
        raise ValueError('D bounded path supports registered text-only Qwen positions')
    with _head_forward(model, function):
        return model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                     position_ids=inputs['position_ids'], use_cache=False).logits


def teacher_projection(model, inputs):
    snapshot = {}
    width = inputs['responses'].shape[1]
    def capture(hidden, weight):
        snapshot['hidden'] = hidden[:, -width-1:-1].detach()
        snapshot['weight'] = weight.detach().clone()
        return hidden.new_zeros((*hidden.shape[:-1], 1))
    with torch.no_grad(): _model_forward(model, inputs, capture)
    return snapshot


def forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, return_all_logps=False,
                        distill_topk=None, topk_indices=None, module=None, **kwargs):
    """Old-policy sampled logps/entropy with bounded head projection; never [B,T,V]."""
    if return_all_logps or distill_topk or topk_indices is not None or kwargs.get('calculate_sum_pi_squared'):
        raise ValueError('D never returns full vocabulary or top-k distributions')
    model = module if module is not None else self.actor_module
    width = micro_batch['responses'].shape[1]
    def sample(hidden, weight):
        hidden = hidden[:, -width-1:-1]
        result = torch.empty((*hidden.shape[:-1], 2), dtype=torch.float32, device=hidden.device)
        with torch.no_grad():
            for start in range(0, width, objective.DEFAULT_CHUNK_TOKENS):
                end = start + objective.DEFAULT_CHUNK_TOKENS
                logps = torch.log_softmax((hidden[:,start:end] @ weight.T).float() / temperature, -1)
                result[:,start:end,0] = logps.gather(-1, micro_batch['responses'][:,start:end,None]).squeeze(-1)
                result[:,start:end,1] = -(logps.exp()*logps).sum(-1) if calculate_entropy else 0
        return result
    with torch.autocast(device_type=micro_batch['input_ids'].device.type, dtype=torch.bfloat16,
                        enabled=micro_batch['input_ids'].device.type == 'cuda'):
        values = _model_forward(model, micro_batch, sample)
    out = {'log_probs':values[...,0]}
    if calculate_entropy: out['entropys'] = values[...,1]
    return out


def ema_movement(owner, update):
    """Measure actual sharded EMA movement, including bf16 rounding, every step.

    Previous teacher shards are retained on CPU only across this update. Sum
    squared shard norms across ranks; avoid a second full model on the GPU.
    """
    student=list(owner.actor_module.parameters());teacher=list(owner.teacher_module.parameters())
    if len(student)!=len(teacher):raise ValueError('EMA parameter geometry differs')
    def local(value):return value.to_local() if hasattr(value,'to_local') else value
    before=[local(value).detach().cpu().clone() for value in teacher]
    result=update()
    distances=[0.,0.]
    for old, learner, target in zip(before,student,teacher):
        learner,target=local(learner).detach().cpu(),local(target).detach().cpu()
        if old.shape!=learner.shape or old.shape!=target.shape:raise ValueError('EMA parameter geometry differs')
        for start in range(0,old.numel(),65536):
            previous=old.reshape(-1)[start:start+65536].double()
            current=target.reshape(-1)[start:start+65536].double()
            actor=learner.reshape(-1)[start:start+65536].double()
            distances[0]+=float(((current-actor)**2).sum())
            distances[1]+=float(((current-previous)**2).sum())
    sums=torch.tensor(distances,dtype=torch.float64,device=local(student[0]).device)
    if torch.distributed.is_initialized():torch.distributed.all_reduce(sums)
    values=sums.sqrt().cpu().tolist()
    return result, {'sdft/teacher_student_l2':values[0],'sdft/teacher_change_l2':values[1]}


def update_policy(self, data):
    """One batch update, chunked head/KL/backward; optimizer first, then EMA once."""
    objective.check_self_distillation(self.config.self_distillation, self.config.loss_agg_mode)
    if self.ulysses_sequence_parallel_size != 1 or self.config.ppo_epochs != 1:
        raise ValueError('registered D requires one PPO epoch and no sequence parallelism')
    if self.teacher_module is None: raise ValueError('D requires a separate EMA teacher')
    self.actor_module.train()
    temperature = float(data.meta_info['temperature'])
    mini_batches = data.split(self.config.ppo_mini_batch_size)
    if len(mini_batches) != 1: raise ValueError('D requires one optimizer update per prompt batch')
    metrics = {'actor/pg_loss':0., 'actor/kl_loss':0., 'actor/lr':self.actor_optimizer.param_groups[0]['lr']}
    self.actor_optimizer.zero_grad()
    device = next(self.actor_module.parameters()).device
    for mini_batch in mini_batches:
        micro_batches = mini_batch.split(1)
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(device)
            inputs = micro_batch.batch
            if bool((inputs['self_distillation_mask'] != 1).any()): raise ValueError('D needs every target')
            teacher_inputs = {'responses':inputs['responses'], 'input_ids':inputs['teacher_input_ids'],
                              'attention_mask':inputs['teacher_attention_mask'], 'position_ids':inputs['teacher_position_ids']}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                teacher = teacher_projection(self.teacher_module, teacher_inputs)
                width = inputs['responses'].shape[1]
                def loss_head(hidden, weight, _teacher=teacher):
                    kl, logps, entropy = projected_distillation(hidden[:, -width-1:-1], weight, _teacher['hidden'], _teacher['weight'],
                                                               inputs['responses'], temperature, objective.DEFAULT_CHUNK_TOKENS)
                    return torch.stack((kl, logps, entropy), -1)
                statistics = _model_forward(self.actor_module, inputs, loss_head)
            mask = objective.loss_mask(inputs['response_mask'])
            counts = mask.sum(-1).clamp(min=1)
            # Recompute the reference ratio here: do not use verl's lower-clamped tensor.
            weights = objective.reference_importance_weights(inputs['old_log_probs'], inputs['rollout_log_probs'])
            sequence_weight = (weights * mask).sum(-1) / counts
            loss = ((statistics[...,0] * mask).sum(-1) / counts * sequence_weight).mean()
            scaled = loss / len(micro_batches)
            if self.scaler is not None: self.scaler.scale(scaled).backward()
            else: scaled.backward()
            metrics['actor/pg_loss'] += loss.detach().item() / len(micro_batches)
            metrics['actor/entropy'] = (statistics[...,2].detach()*inputs['response_mask']).sum().item() / inputs['response_mask'].sum().clamp(min=1).item()
            metrics['sdft/is_sequence_weight_mean'] = sequence_weight.mean().item()
            del teacher, statistics, loss, scaled
        norm = self._optimizer_step()
        metrics['actor/grad_norm'] = norm.detach().item()
        if bool(torch.isfinite(norm).item()):
            _, movement = ema_movement(self,self._update_teacher)
            metrics.update(movement)
            self._v4_completed_updates = getattr(self, '_v4_completed_updates', 0) + 1
        metrics['v4/completed_optimizer_updates'] = getattr(self, '_v4_completed_updates', 0)
    self.actor_optimizer.zero_grad()
    return metrics
