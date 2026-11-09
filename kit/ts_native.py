"""Isolated TS native entry. Import only inside the pinned trainer environment."""
from copy import deepcopy
from pathlib import Path

import ray
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.single_controller.base.decorator import register, Dispatch

from kit import nm_native, ts_actor, ts_arms as ts, ts_generate, ts_masks
from kit.nm_common import need, atomic


def check_config(config):
    native = OmegaConf.to_container(config.nm, resolve=True)
    identity = native['ts']
    shape = ts.settings(identity['binding'], identity['stage'])
    need(identity['arm'] in ts.ARMS and native['seed'] in ts.SEEDS, 'unknown TS arm/seed')
    need(native['rows_per_update'] == shape['rows_per_update'] and native['caps'] == shape['caps'], 'TS geometry differs')
    need(native['updates'] == 40 and native['family'] == 'freeze' and native['arm'] == 'C', 'TS native bridge differs')
    need(native.get('penalty') is None and native.get('measurement') is False, 'unexpected TS penalty/measurement')
    need(config.data.train_batch_size == shape['rows_per_update'] and
         config.actor_rollout_ref.actor.ppo_mini_batch_size == shape['rows_per_update'], 'TS accumulation differs')
    # NM validates every inherited D setting, on a detached copy. CR stage 1/3
    # have 40/56 rows. Only those declared accumulation sizes are normalised.
    legacy = deepcopy(config)
    with open_dict(legacy):
        legacy.nm.rows_per_update = 32
        legacy.data.train_batch_size = 32
        legacy.actor_rollout_ref.actor.ppo_mini_batch_size = 32
    nm_native.check_config(legacy)
    if identity['arm'] != 'D':
        need(config.actor_rollout_ref.rollout.mode == 'async', 'TS auxiliary adapter requires the pinned async manager')
    need(config.actor_rollout_ref.rollout.n == 1, 'attempt zero must keep incumbent n=1')
    return native


def teacher_batch(self, batch, reward_tensor, reward_extra_infos_dict=None):
    config = OmegaConf.to_container(self.config.nm, resolve=True)
    identity = config['ts']
    if identity['arm'] == 'D':
        tensors, metrics = nm_native.teacher_batch(self, batch, reward_tensor, reward_extra_infos_dict)
        tasks = [i for i, e in enumerate(batch.non_tensor_batch['extra_info']) if not e.get('m3_replay')]
        count = len(tasks)
        loss_tokens = nm_native.objective.loss_mask(batch.batch['response_mask'][tasks]).sum()
        counters = dict(attempted_rows=count, attempted_trajectories=count, eligible_rows=count,
                        loss_bearing_tokens=int(loss_tokens), empty_batches=0, sibling_generation=0)
        reasons = batch.non_tensor_batch['ts_finish_reason']
        counters.update({'finish_reason_' + reason: sum(reasons[i] == reason for i in tasks)
                         for reason in ('stop', 'length', 'abort')})
        batch.meta_info['ts_telemetry'] = {'ts/' + key: value for key, value in counters.items()}
        metrics.update(batch.meta_info['ts_telemetry'])
        return tensors, metrics
    extras = batch.non_tensor_batch['extra_info']
    tasks = [i for i, extra in enumerate(extras) if not extra.get('m3_replay')]
    need(tasks and 'ts_attempts' in batch.non_tensor_batch, 'auxiliary attempts missing')
    groups, signals = [], []
    for index in tasks:
        extra = extras[index]
        mask = batch.batch['response_mask'][index].bool()
        ids = batch.batch['responses'][index][mask].tolist()
        records = batch.non_tensor_batch['ts_attempts'][index]
        plan = ts.attempt_plan(identity['arm'], config['seed'], identity['stage'],
            extra['nm_update'], extra['nm_position'], extra['index'])[1:]
        need(records is not None and len(records) == 7, 'missing checker input/auxiliary output')
        need([(r['index'], r['seed']) for r in records] == [(p['index'], p['seed']) for p in plan],
             'auxiliary request identity differs')
        gold = batch.non_tensor_batch['reward_model'][index]['ground_truth']
        group, signal = ts_generate.checked_signal(self.tokenizer, identity['stage'], identity['arm'], ids,
            batch.non_tensor_batch['ts_finish_reason'][index], records, gold, extra['demonstration'],
            reward_tensor[index].sum())
        signals.append(signal)
        groups.append(group)
    built = ts.teacher_inputs(self.tokenizer, [batch.non_tensor_batch['raw_prompt'][i] for i in tasks],
        signals, batch.batch['responses'][tasks], batch.batch['response_mask'][tasks], config['caps'],
        student_prompts=[batch.non_tensor_batch['ts_student_prompt'][i] for i in tasks])
    ts_masks.record(Path(config['receipt']).parent, extras[tasks[0]]['nm_update'],
                    [extras[i] for i in tasks], signals, groups)
    reasons = [None] * len(extras)
    details = [None] * len(extras)
    for index, signal in zip(tasks, signals):
        reasons[index] = signal.reason
        details[index] = signal.detail
    batch.non_tensor_batch['ts_mask_reason'] = np.array(reasons, dtype=object)
    batch.non_tensor_batch['ts_mask_detail'] = np.array(details, dtype=object)
    tensors = {}
    for key in ('teacher_input_ids', 'teacher_attention_mask', 'teacher_position_ids', 'self_distillation_mask'):
        value = built[key]
        padded = value[:1].expand(len(extras), *value.shape[1:]).clone()
        padded[tasks] = value
        tensors[key] = padded.to(batch.batch['input_ids'].device)
    counters = ts.telemetry(signals, groups, batch.batch['response_mask'][tasks].cpu(), torch.zeros(len(tasks)))
    counters.pop('actual_task_loss_mass')  # The actor records measured KL mass.
    batch.meta_info['ts_telemetry'] = {'ts/' + key: value for key, value in counters.items()}
    return DataProto.from_dict(tensors=tensors), dict(batch.meta_info['ts_telemetry'])


class TSTrainer(nm_native.NMTrainer):
    _maybe_build_self_distillation_batch = teacher_batch

    def _update_actor(self, batch):
        if self.config.nm.ts.get('attempt0_probe', False):
            from kit.ts_qualification import write_probe
            write_probe(batch, OmegaConf.to_container(self.config.nm, resolve=True))
        return super()._update_actor(batch)

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        if self.config.nm.ts.arm != 'D':
            ts_masks.supersede(Path(self.config.nm.receipt).parent, self.global_steps)
        return result

    def init_workers(self):
        # Identical passive observers on D and M preserve their sampling path.
        with open_dict(self.config.actor_rollout_ref.rollout.agent):
            self.config.actor_rollout_ref.rollout.agent.agent_loop_manager_class = 'kit.ts_rollout.TSAgentLoopManager'
            self.config.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(ts.KIT / 'resources/ts/ts-agents.yaml')
        result = super().init_workers()
        identity = OmegaConf.to_container(self.config.nm.ts, resolve=True)
        if identity['arm'] != 'D':
            ts_generate.install(self.async_rollout_manager, dict(identity, seed=self.config.nm.seed))
        return result


class TSWorker(nm_native.NMWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result = super().init_model()
        if self._is_actor:
            self.actor.__class__ = ts_actor.actor_class(type(self.actor))
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def nm_receipt(self):
        result = super().nm_receipt()
        identity = self.actor.nm_config['ts']
        return dict(result, arm=identity['arm'], stage=identity['stage'], binding=identity['binding'])


class TSTaskRunner(nm_native.main_ppo.TaskRunner):
    def add_actor_rollout_worker(self, config):
        _, group = super().add_actor_rollout_worker(config)
        self.role_worker_mapping[nm_native.Role.ActorRolloutRef] = ray.remote(TSWorker)
        return TSWorker, group

    def run(self, config):
        nm_native.check_pinned()
        native = check_config(config)
        with open_dict(config.actor_rollout_ref):
            config.actor_rollout_ref.nm = config.nm
        previous = nm_native.main_ppo.RayPPOTrainer
        class ReceiptTrainer(TSTrainer):
            def fit(self):
                from kit.ts_qualification import ProbeComplete
                try:
                    result = super().fit()
                except ProbeComplete:
                    need(native['ts'].get('attempt0_probe') is True, 'unexpected identity probe stop')
                    return None
                receipts = self.actor_rollout_wg.nm_receipt()
                need(all(r['config_sha256'] == receipts[0]['config_sha256'] for r in receipts), 'rank receipt mismatch')
                atomic(native['receipt'], dict(receipts[0], rank_receipts=receipts))
                return result
        nm_native.main_ppo.RayPPOTrainer = ReceiptTrainer
        try:
            return super().run(config)
        finally:
            nm_native.main_ppo.RayPPOTrainer = previous
