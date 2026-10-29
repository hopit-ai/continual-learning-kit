"""Pinned production worker/runner subclasses for M3 only.

Import this module in the staged verl environment; CPU math lives in m3_loss.
"""
from __future__ import annotations
from copy import deepcopy
import os
import json
import math
import functools
import threading
import socket
from pathlib import Path
import time
import ray
import torch
from omegaconf import OmegaConf, open_dict
from kit import sdft_objective as objective
from kit.sdft_verl import check_pinned
from kit.m3_contract import (validate_config, geometry, SCHEMA_VERSION, LOSS_IDS,
                             write_receipt, prompt_cap, rank_major_indices, COMMON_OVERRIDES,
                             CONTEXT_CAP, REPROMPT_CAP, load_preflight, receipt_status, PROFILES, validate_output_layout, verify_base_files, validate_preflight_inputs)
from kit.m3_loss import audit_long_context_rows, cuda_full_vocab_check
from kit.m3_smoke import cgroup_sample, read_jsonl, MemoryPhaseTracker
from kit.m3_actor import actor_class, load_frozen_base, reference_hash, count_scheduler_steps, checkpoint_metadata, validate_resume_metadata, state_digest
from verl import DataProto
import verl.trainer.main_ppo as main_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.single_controller.base.decorator import register, Dispatch
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils.dataset.rl_dataset import RLHFDataset
from kit.m3_entry import (SMOKE_INIT_TIMEOUT, smoke_loopback_settings, smoke_initialization_deadline,
                          smoke_runner_ready_path)


def check_config(config):
    c = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else deepcopy(config)
    m3 = validate_config(c['m3'])
    world = c['trainer']['n_gpus_per_node'] * c['trainer']['nnodes']
    if world != PROFILES[m3['profile']][0] or c['trainer']['nnodes'] != 1:
        raise ValueError('M3 requires an isolated single node with registered rank count')
    shape = geometry(m3['arm'], m3['stage'], world)
    actor = c['actor_rollout_ref']['actor']
    attention=c['actor_rollout_ref']['model'].get('override_config',{}).get('attn_implementation','flash_attention_2')
    if attention not in ('flash_attention_2','sdpa'):raise ValueError('M3 requires matching memory-efficient actor/base attention')
    allowance = prompt_cap(m3['arm'], m3['stage'])
    checks = [(c['data']['train_batch_size'], shape['global_rows']),
              (actor['ppo_mini_batch_size'], shape['global_rows']),
              (actor['ppo_micro_batch_size_per_gpu'], 1),
              (actor['strategy'], 'fsdp2'), (actor['use_dynamic_bsz'], False),
              (actor['grad_clip'], 1.), (actor['optim']['betas'], [.9, .999]),
              (c['data']['shuffle'], False), (c['trainer']['balance_batch'], False), (actor.get('shuffle', False), False),
              (c['data']['max_prompt_length'], allowance),
              (actor['self_distillation']['max_reprompt_len'], REPROMPT_CAP),
              (c['trainer']['total_training_steps'], 2 if m3['profile']=='preflight' and m3['preflight_phase']=='prefix' else PROFILES[m3['profile']][1]),
              (c['data']['apply_chat_template_kwargs']['enable_thinking'], False),
              (c['actor_rollout_ref']['rollout']['temperature'], 1.)]
    rollout=c['actor_rollout_ref']['rollout']
    checks.extend([(rollout['gpu_memory_utilization'],.55),(rollout['free_cache_engine'],True)])
    if any(actual != expected for actual, expected in checks):
        raise ValueError('M3 resolved training geometry/recipe differs')
    custom = c['data'].get('custom_cls', {}) or {}
    smoke_jsonl=(m3['profile']=='technical-smoke' and custom.get('name')=='M3SmokeJSONLDataset'
                 and Path(custom.get('path') or '').resolve()==Path(__file__).resolve())
    if (custom.get('name') or custom.get('path')) and not smoke_jsonl:
        raise ValueError('M3 PPO requires native RL dataset or the isolated smoke JSONL subclass')
    for key in ('max_model_len','actor_rollout_ref.rollout.max_model_len',
                'actor_rollout_ref.rollout.max_num_batched_tokens','actor_rollout_ref.actor.ppo_max_token_len_per_gpu'):
        value=c
        for part in key.split('.'): value=value[part]
        if value<CONTEXT_CAP: raise ValueError('common rollout/teacher context cannot fit longest admitted rows')
    reward=c.get('custom_reward_function',{})
    if Path(reward.get('path','')).name!='m3_entry.py' or reward.get('name')!='compute_score':
        raise ValueError('M3 diagnostic reward router required for every stage')
    if c['data'].get('gen_batch_size', shape['global_rows']) != shape['global_rows']:
        raise ValueError('generation batch must preserve complete scheduled update')
    for value in (m3['base_model'], c['actor_rollout_ref']['model']['path']):
        if not Path(value).is_dir():
            raise ValueError('M3 requires locally staged models; downloads forbidden')
    if c['actor_rollout_ref'].get('ref', {}).get('model', {}).get('path', c['actor_rollout_ref']['model']['path']) != c['actor_rollout_ref']['model']['path']:
        raise ValueError('EMA must initialize from incoming student')
    # Inherit every D objective/recipe check after normalizing ONLY M3's declared
    # batch and long-input allowances on a detached validation copy.
    legacy = deepcopy(c)
    legacy['data']['train_batch_size'] = 32
    legacy['data']['max_prompt_length'] = 2048
    legacy['actor_rollout_ref']['actor']['ppo_mini_batch_size'] = 32
    legacy['actor_rollout_ref']['actor']['self_distillation']['max_reprompt_len'] = 6144
    root=validate_output_layout(m3,c['trainer']['default_local_dir'])
    scope=Path(m3['receipt']).resolve().parent
    for key in ('rollout_data_dir','validation_data_dir'):
        if c['trainer'].get(key) and not Path(c['trainer'][key]).resolve().is_relative_to(scope):
            raise ValueError('M3 auxiliary outputs escape the isolated receipt scope')
    if c['trainer'].get('default_hdfs_dir') is not None:raise ValueError('M3 requires local isolated outputs')
    if m3['profile']=='preflight':
        if not m3['input_manifest'] or not Path(m3['input_manifest']).is_file() or not m3['host_reservation_bytes'] or m3['host_reservation_bytes']<=0:
            raise ValueError('preflight requires frozen input manifest and host reservation')
        frozen=validate_preflight_inputs(m3['input_manifest'],arm=m3['arm'],stage=m3['stage'],root=root)
        actual=c['data']['train_files'];actual=[actual] if isinstance(actual,str) else actual
        expected=[str((Path(m3['input_manifest']).parent/name).resolve()) for name in frozen['train_files']]
        if [str(Path(name).resolve()) for name in actual]!=expected:raise ValueError('preflight trainer files differ from frozen inputs')
    previous = os.environ.get('V4_PROFILE')
    os.environ['V4_PROFILE'] = 'technical-smoke' if m3['profile']=='preflight' else m3['profile']
    try:
        objective.check_trainer_config(legacy)
        from kit.v4_contract import check_resolved
        check_resolved(legacy,'D',profile_name=os.environ['V4_PROFILE'],
                       resume_path=c['trainer'].get('resume_from_path') if c['trainer']['resume_mode']=='resume_path' else None)
    finally:
        if previous is None: os.environ.pop('V4_PROFILE', None)
        else: os.environ['V4_PROFILE'] = previous
    return m3


def teacher_batch(owner, batch, reward_tensor, reward_extra_infos_dict=None):
    """Task demonstration context; replay ranks receive a masked dummy task copy."""
    extras = batch.non_tensor_batch['extra_info']
    tasks = [i for i, extra in enumerate(extras) if not extra.get('m3_replay')]
    if not tasks:
        raise ValueError('global M3 batch must include task rows')
    prompts = batch.non_tensor_batch['raw_prompt']
    built = objective.build_teacher_inputs(owner.tokenizer, [prompts[i] for i in tasks],
        [objective.demonstration_of(extras[i]) for i in tasks], batch.batch['responses'][tasks],
        batch.batch['response_mask'][tasks], chat_template_kwargs={'enable_thinking': False},
        max_reprompt_len=int(owner.config.actor_rollout_ref.actor.self_distillation.max_reprompt_len))
    tensors = {}
    for key in ('teacher_input_ids', 'teacher_attention_mask', 'teacher_position_ids'):
        value = built[key]
        padded = value[:1].expand(len(extras), *value.shape[1:]).clone()
        padded[tasks] = value
        tensors[key] = padded.to(batch.batch['input_ids'].device)
    tensors['self_distillation_mask'] = torch.ones(len(extras), device=batch.batch['input_ids'].device)
    return DataProto.from_dict(tensors=tensors), {'m3/teacher_prompt_tokens_max': max(built['teacher_prompt_lengths'])}


class M3SmokeJSONLRows(list):
    def __getitem__(self, index):
        # Native __getitem__ mutates the row while assembling raw_prompt.
        return deepcopy(super().__getitem__(index))


class M3SmokeJSONLDataset(RLHFDataset):
    """Native text-row behavior with JSONL IO, confined to engineering smoke.

    Pinned verl still imports Hugging Face datasets at module import; this
    subclass removes Parquet/Arrow data reads, not that upstream dependency.
    """
    def _read_files_and_tokenize(self):
        if self.config.get('shuffle') or self.max_samples>0:
            raise ValueError('smoke JSONL schedules cannot shuffle or subsample')
        if any(Path(filename).suffix!='.jsonl' for filename in self.data_files):
            raise ValueError('smoke dataset requires locally frozen JSONL')
        self.dataframe=M3SmokeJSONLRows(row for filename in self.data_files for row in read_jsonl(filename))
        if not self.dataframe:raise ValueError('empty frozen smoke schedule')


class M3SmokeResourcePool:
    def __init__(self, manager):
        self.manager=manager

    def __getattr__(self, key):
        return getattr(self.manager,key)

    def create_resource_pool(self):
        with smoke_initialization_deadline('ray-resource-pool'):
            return self.manager.create_resource_pool()


class M3PPOTrainer(RayPPOTrainer):
    _maybe_build_self_distillation_batch = teacher_batch

    def __init__(self, *args, **kwargs):
        config=kwargs['config']; tokenizer=kwargs['tokenizer']
        files=config.data.train_files
        if isinstance(files,str): files=[files]
        if config.m3.profile=='technical-smoke':
            rows=(row for filename in files for row in read_jsonl(filename))
        else:
            import pyarrow.parquet as pq
            rows=(row for filename in files for row in pq.read_table(filename).to_pylist())
        self._m3_length_audit=audit_long_context_rows(tokenizer,rows,
            prompt_limit=int(config.data.max_prompt_length),response_limit=int(config.data.max_response_length),
            reprompt_limit=int(config.actor_rollout_ref.actor.self_distillation.max_reprompt_len),
            context_limit=int(config.actor_rollout_ref.rollout.max_model_len))
        super().__init__(*args,**kwargs)
        if config.m3.profile=='technical-smoke':
            self.resource_pool_manager=M3SmokeResourcePool(self.resource_pool_manager)

    def _update_actor(self, batch):
        # No balancing: transpose only for native contiguous rank dispatch, then
        # restore the round-major order for subsequent logs/metrics/checkpoints.
        if self.config.m3.profile=='preflight':
            root=Path(self.config.m3.receipt).resolve().parents[3]
            cached=root/'preflight'/self.config.m3.arm/'uninterrupted'/'batches'/f'update-{self.global_steps}.pkl'
            if self.config.m3.preflight_phase=='uninterrupted':
                cached.parent.mkdir(parents=True,exist_ok=True);batch.save_to_disk(cached)
            else:
                if not cached.is_file():raise ValueError('preflight paired rollout cache missing')
                batch=DataProto.load_from_disk(cached)
        world=self.config.trainer.n_gpus_per_node
        expected=geometry(self.config.m3.arm,self.config.m3.stage,world)['global_rows']
        if len(batch)!=expected: raise ValueError('one complete M3 scheduled update required')
        order=torch.tensor(rank_major_indices(expected,world),device=batch.batch['input_ids'].device)
        batch.reorder(order)
        try:
            return super()._update_actor(batch)
        finally:
            batch.reorder(torch.argsort(order))



class M3SmokeRayWorkerGroup(RayWorkerGroup):
    """Single-container rendezvous; never ask Ray to discover a node address."""
    def __init__(self, *args, **kwargs):
        kwargs['ray_wait_register_center_timeout']=SMOKE_INIT_TIMEOUT
        with smoke_initialization_deadline('ray-worker-group'):
            super().__init__(*args, **kwargs)

    def _get_master_addr_port(self, pg):
        if self._master_port is None:
            with socket.socket() as listener:
                listener.bind(('127.0.0.1',0))
                self._master_port=str(listener.getsockname()[1])
        self._master_addr='127.0.0.1'


class M3ActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    def __init__(self, config, role, **kwargs):
        if config.m3.profile=='technical-smoke':
            # RayWorkerGroup supplies one shared free loopback port. Override a
            # stale inherited init method before native Worker/init_process_group.
            os.environ.update(smoke_loopback_settings()['env'], MASTER_PORT=os.environ['MASTER_PORT'])
            os.environ['DIST_INIT_METHOD']='tcp://127.0.0.1:'+os.environ['MASTER_PORT']
            config.nccl_timeout=SMOKE_INIT_TIMEOUT
            with smoke_initialization_deadline('distributed-rendezvous'):
                super().__init__(config,role,**kwargs)
        else:
            super().__init__(config,role,**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        check_pinned()
        self._m3_node_preflight=(load_preflight(self.config.m3.get('preflight'),self.config.m3.arm)
                                 if self.config.m3.profile=='scientific' else None)
        torch.cuda.reset_peak_memory_stats()
        reservation=(self._m3_node_preflight['telemetry']['host_reservation_bytes']
                     if self._m3_node_preflight else self.config.m3.get('host_reservation_bytes') or 64*1024**3)
        self._m3_gpu_stop = threading.Event()
        self._m3_gpu_device=torch.cuda.current_device()
        def device_sample():
            free,total=torch.cuda.mem_get_info(self._m3_gpu_device)
            return total-free,total,torch.cuda.max_memory_reserved(self._m3_gpu_device)
        self._m3_memory_tracker=MemoryPhaseTracker(device_sample,
            lambda:torch.cuda.reset_peak_memory_stats(self._m3_gpu_device),torch.cuda.empty_cache,
            lambda:torch.cuda.synchronize(self._m3_gpu_device),lambda:cgroup_sample(reservation))
        sample_gpu=self._m3_memory_tracker.sample
        sample_gpu()
        self._m3_cuda_full_kl=None
        if self.config.m3.profile in ('technical-smoke','preflight'):
            result=[None]
            if self.rank==0:
                try:result[0]=cuda_full_vocab_check()
                except (RuntimeError,ValueError):result[0]=dict(passed=False)
            torch.distributed.broadcast_object_list(result,src=0)
            self._m3_cuda_full_kl=result[0]
            if not result[0]['passed']:raise RuntimeError('CUDA full-vocabulary qualification failed')
        def monitor_gpu():
            # Driver-level device usage includes rollout subprocesses and NCCL,
            # unlike the actor allocator's process-local reserved-byte counter.
            torch.cuda.set_device(self._m3_gpu_device)
            while not self._m3_gpu_stop.wait(.05):
                sample_gpu()
        self._m3_gpu_device = torch.cuda.current_device()
        self._m3_gpu_thread = threading.Thread(target=monitor_gpu,daemon=True)
        self._m3_gpu_thread.start()
        result = super().init_model()
        if self._is_actor:
            self.actor.__class__ = actor_class(type(self.actor))
            self.actor.m3_config = OmegaConf.to_container(self.config.m3, resolve=True)
            self.actor.m3_world_size = self.world_size
            self.actor._m3_memory_tracker=self._m3_memory_tracker
            self.actor.frozen_base_module = None
            if self.actor.m3_config['arm'] == 'K':
                self.actor.frozen_base_module = load_frozen_base(self.config.m3.base_model,
                    self.config.m3.base_revision,self.device_mesh,binding=Path(self.config.m3_root)/'inputs/base.json',
                    attention_implementation=self.config.model.get('override_config',{}).get('attn_implementation','flash_attention_2'))
            self._m3_frozen_before = (reference_hash(self.actor.frozen_base_module)
                                     if self.actor.frozen_base_module is not None else None)
            self._m3_started = time.monotonic()
            self._m3_scheduler_steps = 0
            self._m3_optimizer_steps = 0
            self._m3_ema_steps = 0
            optimizer_step = self.actor_optimizer.step
            @functools.wraps(optimizer_step)
            def counted_optimizer(*args, **kwargs):
                result = optimizer_step(*args, **kwargs)
                self._m3_optimizer_steps += 1
                return result
            counted_optimizer._wrapped_by_lr_sched = True
            self.actor_optimizer.step = counted_optimizer
            ema_step = self.actor._update_teacher
            def counted_ema():
                result = ema_step()
                self._m3_ema_steps += 1
                return result
            self.actor._update_teacher = counted_ema
            # Instrument this instance only; native worker calls scheduler.step once.
            def scheduler_stepped():
                self._m3_scheduler_steps += 1
            count_scheduler_steps(self.actor_lr_scheduler, scheduler_stepped)
        return result

    def _ema_checkpoint_manager(self):
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        return FSDPCheckpointManager(model=self.actor.teacher_module, processing_class=self.tokenizer,
            checkpoint_config={'save_contents':['model'], 'load_contents':['model']})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if hdfs_path is not None:
            raise ValueError('M3 checkpoints require isolated local native state')
        result = super().save_checkpoint(local_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        self._ema_checkpoint_manager().save_checkpoint(str(Path(local_path)/'m3-ema'), global_step=global_step)
        if self.rank == 0:
            state=checkpoint_metadata(self.actor,optimizer_steps=self._m3_optimizer_steps,
                ema_steps=self._m3_ema_steps,scheduler_steps=self._m3_scheduler_steps,frozen_base=self._m3_frozen_before)
            (Path(local_path)/'m3-state.json').write_text(json.dumps(state, sort_keys=True)+'\n')
        torch.distributed.barrier()
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if hdfs_path is not None or del_local_after_load:
            raise ValueError('M3 native state must remain immutable and local')
        state=None
        if local_path is not None:
            state=json.loads((Path(local_path)/'m3-state.json').read_text())
            validate_resume_metadata(self.actor,state,self._m3_frozen_before)
        result = super().load_checkpoint(local_path)
        if state is not None:
            self._ema_checkpoint_manager().load_checkpoint(str(Path(local_path)/'m3-ema'))
            self.actor._v4_completed_updates = state['completed_updates']
            self.actor._m3_oracle_completed=state.get('oracle_completed',False)
            self.actor._m3_oracle_error=state.get('oracle_error')
            self.actor._m3_oracle_update=state.get('oracle_update')
            self._m3_optimizer_steps = state['optimizer_steps']
            self._m3_ema_steps = state['ema_steps']
            self._m3_scheduler_steps = state['scheduler_steps']
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def m3_receipt(self):
        base = self.actor.frozen_base_module
        after = reference_hash(base) if base is not None else None
        fingerprints=None
        optimizer_materialized=False
        if self.config.m3.profile=='preflight':
            fingerprints=dict(actor=reference_hash(self.actor.actor_module),ema=reference_hash(self.actor.teacher_module),
                optimizer=state_digest(self.actor_optimizer.state_dict()),scheduler=state_digest(self.actor_lr_scheduler.state_dict()),frozen_base=after)
            states=list(self.actor_optimizer.state.values())
            ready=bool(states) and all('exp_avg' in v and 'exp_avg_sq' in v for v in states)
            flags=torch.tensor(int(ready),device='cuda')
            torch.distributed.all_reduce(flags,op=torch.distributed.ReduceOp.MIN)
            optimizer_materialized=bool(flags.item())
        steps = getattr(self.actor, '_v4_completed_updates', 0)
        m3 = self.actor.m3_config
        shape = geometry(m3['arm'], m3['stage'], self.world_size)
        self._m3_gpu_stop.set()
        self._m3_gpu_thread.join(timeout=1)
        memory=self._m3_memory_tracker.snapshot();training=memory['training'];oracle=memory['oracle']
        peaks = torch.tensor([training['gpu_peak_fraction'],
            training['cgroup_peak_fraction'] if training['cgroup_peak_fraction'] is not None else -1.,
            training['allocator_peak_reserved_bytes'],oracle['gpu_peak_fraction'],oracle['allocator_peak_reserved_bytes'],
            oracle['cgroup_peak_fraction'] if oracle['cgroup_peak_fraction'] is not None else -1.],device='cuda')
        torch.distributed.all_reduce(peaks, op=torch.distributed.ReduceOp.MAX)
        receipt = dict(schema_version=SCHEMA_VERSION, **{k: m3[k] for k in ('arm','stage','seed','profile')},
            frozen_base_files_sha256=self.config.m3_base_files_sha256,
            attention_implementation=self.config.model.get('override_config',{}).get('attn_implementation','flash_attention_2'),
            status='INCOMPLETE', node_preflight=self._m3_node_preflight, loss_ids=LOSS_IDS, world_size=self.world_size,
            **{k: shape[k] for k in ('global_rows','task_rows','replay_rows','accumulation_rounds','global_divisor')},
            completed_updates=steps, optimizer_steps=self._m3_optimizer_steps, ema_steps=self._m3_ema_steps,
            scheduler_steps=self._m3_scheduler_steps, frozen_base_before=self._m3_frozen_before,
            frozen_base_after=after, frozen_base_has_grad=base is not None and any(p.grad is not None for p in base.parameters()),
            gates={'frozen_reference': after == self._m3_frozen_before,
                   'step_counters': steps == self._m3_scheduler_steps == self._m3_optimizer_steps == self._m3_ema_steps,
                   'finite_losses_gradients_ema': bool(getattr(self.actor,'_m3_history',[])) and all(math.isfinite(v) for row in getattr(self.actor, '_m3_history', []) for v in row.values()),
                   'gpu_memory': float(peaks[0]) <= .90,
                   'cgroup_ram': float(peaks[1])<=.95 if float(peaks[1])>=0 else None},
            telemetry={'worker_seconds': time.monotonic()-self._m3_started, 'gpu_peak_fraction': float(peaks[0]),
                       'gpu_peak_measure': 'sampled_driver_device_used_bytes_including_rollout',
                       'cgroup_peak_fraction': float(peaks[1]) if float(peaks[1])>=0 else None,
                       'allocator_peak_reserved_bytes':int(peaks[2]),
                       'memory_gate_scope':'startup/rollout/training/optimizer/EMA; oracle excluded',
                       'oracle_gpu_peak_fraction':float(peaks[3]),'oracle_allocator_peak_reserved_bytes':int(peaks[4]),
                       'oracle_cgroup_peak_fraction':float(peaks[5]) if float(peaks[5])>=0 else None,
                       'deterministic_algorithms_scope':'preflight actor/EMA/base/oracle updates only; restored before vLLM',
                       'vllm_deterministic_algorithms':'not enabled by M3; prior process settings retained'},
            updates=getattr(self.actor, '_m3_history', []),state_hashes=fingerprints,
            gradient_oracle=dict(completed=getattr(self.actor,'_m3_oracle_completed',False),
                update=getattr(self.actor,'_m3_oracle_update',None),relative_error=getattr(self.actor,'_m3_oracle_error',None)),
            optimizer_materialized=optimizer_materialized,cuda_full_kl=self._m3_cuda_full_kl,scientific_evidence=False)
        receipt['status']=receipt_status(receipt)
        if self.rank == 0:
            write_receipt(receipt, m3['receipt'])
        return receipt


class M3TaskRunner(main_ppo.TaskRunner):
    def add_actor_rollout_worker(self, config):
        actor, group = super().add_actor_rollout_worker(config)
        if actor is not AsyncActorRolloutRefWorker or Role.ActorRolloutRef not in self.role_worker_mapping:
            raise ValueError('M3 requires native legacy FSDP2 actor/EMA workers')
        self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(M3ActorRolloutRefWorker)
        return M3ActorRolloutRefWorker, M3SmokeRayWorkerGroup if config.m3.profile=='technical-smoke' else group

    def run(self, config):
        if config.m3.profile=='technical-smoke':
            smoke_runner_ready_path(config.m3.receipt).touch()
        check_pinned()
        m3=check_config(config)
        root=validate_output_layout(m3,config.trainer.default_local_dir)
        files_digest=verify_base_files(m3['base_model'],root/'inputs/base.json',m3['base_revision'])
        if m3['profile']=='preflight':verify_base_files(config.actor_rollout_ref.model.path,root/'inputs/base.json',m3['base_revision'])
        if m3['profile']=='scientific':
            qualified=load_preflight(m3['preflight'],m3['arm'])
            if qualified is None:raise ValueError('scientific training is INCOMPLETE without passing 8B/eight-H100 preflight')
            attention=config.actor_rollout_ref.model.get('override_config',{}).get('attn_implementation','flash_attention_2')
            if qualified['frozen_base_files_sha256']!=files_digest or qualified['attention_implementation']!=attention:
                raise ValueError('scientific base files/attention differ from node qualification')
        with open_dict(config.actor_rollout_ref):
            config.actor_rollout_ref.m3 = config.m3
            config.actor_rollout_ref.m3_root=str(root)
            config.actor_rollout_ref.m3_base_files_sha256=files_digest
        # The pinned driver resolves this global when constructing its trainer.
        # Restore it after this isolated runner finishes; no shared class patches.
        previous = main_ppo.RayPPOTrainer
        class ReceiptTrainer(M3PPOTrainer):
            def fit(self):
                result = super().fit()
                receipts=self.actor_rollout_wg.m3_receipt()
                receipt=dict(receipts[0])
                receipt['length_audit']=self._m3_length_audit
                receipt['gates']['long_context']=True
                receipt['status']=receipt_status(receipt)
                write_receipt(receipt,config.m3.receipt)
                return result
        main_ppo.RayPPOTrainer = ReceiptTrainer
        try:
            return super().run(config)
        finally:
            main_ppo.RayPPOTrainer = previous
