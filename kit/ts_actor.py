"""TS-only native masked actor. D still enters the incumbent NM actor unchanged."""
import torch

from kit import nm_actor, sdft_objective as objective, ts_arms as ts
from kit.nm_common import need
from kit.sdft_actor import _model_forward, teacher_projection, projected_distillation, forward_micro_batch
from kit.m3_loss import replay_inputs, replay_loss


def update_policy(self, data):
    if self.nm_config['ts']['arm'] == 'D':
        metrics = nm_actor.update_policy(self, data)
        metrics.update(data.meta_info.get('ts_telemetry', {}))
        metrics['ts/actual_task_loss_mass'] = metrics['actor/pg_loss']
        self._nm_history[-1] = dict(metrics)
        return metrics
    config = self.nm_config
    world = self.nm_world_size
    device = next(self.actor_module.parameters()).device
    rows = list(data.split(1))
    need(len(rows) == config['rows_per_update'] // world, 'incomplete TS accumulation')
    need(self.config.ppo_epochs == 1 and self.ulysses_sequence_parallel_size == 1,
         'one pass and no sequence parallelism required')
    objective.check_self_distillation(self.config.self_distillation, self.config.loss_agg_mode)
    update = getattr(self, '_v4_completed_updates', 0) + 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    for index, row in enumerate(rows):
        extra = row.non_tensor_batch['extra_info'][0]
        need(extra['nm_update'] == update and
             (config['balance_batch'] or extra['nm_position'] == rank + index * world + 1),
             'scheduled order/update changed')
        if not extra.get('m3_replay'):
            mask = row.batch['self_distillation_mask']
            need(mask.numel() == 1 and bool(((mask == 0) | (mask == 1)).all()), 'invalid TS mask')
    self.actor_module.train()
    self.teacher_module.eval()
    self.actor_optimizer.zero_grad()
    totals = [0., 0., 0.]
    for row in rows:
        row = row.to(device)
        inputs = row.batch
        extra = row.non_tensor_batch['extra_info'][0]
        replay = bool(extra.get('m3_replay'))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            # Replay ranks execute the same frozen-teacher projection boundary.
            # The mixed task/replay schedule must preserve FSDP collectives.
            teacher = teacher_projection(self.teacher_module, dict(
                responses=inputs['responses'], input_ids=inputs['teacher_input_ids'],
                attention_mask=inputs['teacher_attention_mask'], position_ids=inputs['teacher_position_ids']))
            if replay:
                values = replay_inputs(inputs, extra['m3_target_ids'], int(extra['m3_eos_token_id']))
                value = replay_loss(self.actor_module, values)
            else:
                width = inputs['responses'].shape[-1]
                def head(hidden, weight, _teacher=teacher):
                    kl, _, _ = projected_distillation(hidden[:, -width-1:-1], weight,
                        _teacher['hidden'], _teacher['weight'], inputs['responses'], 1., 256)
                    return kl
                kl = _model_forward(self.actor_module, inputs, head)
                per_row = ts.per_row_loss(kl, inputs['response_mask'], inputs['old_log_probs'], inputs['rollout_log_probs'])
                value = (per_row * inputs['self_distillation_mask'].detach()).sum()
                totals[2] += float((objective.loss_mask(inputs['response_mask']) *
                                    inputs['self_distillation_mask'][:, None]).sum())
            need(not nm_actor.reduce([int(not bool(torch.isfinite(value)))], device)[0], 'nonfinite TS loss on a rank')
            scaled = value * (world / 32)
            (self.scaler.scale(scaled) if self.scaler is not None else scaled).backward()
            totals[int(replay)] += float(value.detach())
            del teacher
    # Even an empty S/M task batch reaches all three scheduled native steps.
    norm = self._optimizer_step()
    need(bool(torch.isfinite(torch.as_tensor(norm))), 'nonfinite TS gradient norm')
    self._update_teacher()
    self._v4_completed_updates = update
    totals = nm_actor.reduce(totals, device)
    metrics = dict(data.meta_info.get('ts_telemetry', {}))
    metrics.update({'actor/pg_loss': totals[0] / 32, 'ts/actual_task_loss_mass': totals[0] / 32,
                    'ts/replay_loss': totals[1] / 32, 'ts/loss_bearing_tokens': totals[2],
                    'actor/grad_norm': float(norm), 'nm/global_divisor': 32,
                    'v4/completed_optimizer_updates': update})
    self._nm_history = getattr(self, '_nm_history', []) + [dict(metrics)]
    self.actor_optimizer.zero_grad()
    return metrics


def actor_class(parent):
    class TSActor(parent):
        _forward_micro_batch = forward_micro_batch
        update_policy = update_policy
    return TSActor
