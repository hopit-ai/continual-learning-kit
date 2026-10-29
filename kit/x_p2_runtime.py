"""P2 first-batch transport and measurement stop, outside the reviewed trainer."""
from pathlib import Path
import functools
import os
from kit.s_common import atomic,read,sha,require
from kit.v4_restore_check import state_hash
from kit.p2_l2 import VARIANT


def deterministic():
    """No X-specific kernel flags or reseeding; native D owns initialization.

    Actor RNG for the cached first batch is restored separately. The two-H100
    smoke uses p_determinism.configure, independently of this production path.
    """
    return None


def install_trainer(cls):
    if not os.environ.get('X_P2_MODE') or getattr(cls,'_x_p2_transport',False):return
    original,fit=cls._update_actor,cls.fit
    class Measured(Exception):pass
    @functools.wraps(original)
    def update(self,batch):
        import torch
        mode=os.environ['X_P2_MODE'];root=Path(os.environ['X_P2_BATCH'])
        if mode=='prepare':return original(self,batch)
        if self.global_steps==1 and not os.environ.get('X_TIER2_NATIVE'):
            if mode=='measure':
                root.mkdir(parents=True,exist_ok=True)
                target=root/'actor-input.pt';temporary=target.with_suffix('.tmp')
                torch.save(batch.to('cpu'),temporary);temporary.replace(target)
                atomic(root/'batch.json',dict(schema='rung-p2-first-batch.v1',variant=VARIANT,
                    sha256=sha(target),tensor_hash=state_hash(dict(batch.batch.items())),
                    tensor_keys=sorted(batch.batch.keys()),
                    includes='Full DataProto: rollout tensors, response/task masks, teacher contexts, old/rollout/ref log probabilities, importance inputs, replay targets and metadata'))
            else:
                receipt=read(root/'batch.json')
                require(receipt['variant']==VARIANT and sha(root/'actor-input.pt')==receipt['sha256'],'P2 first-batch cache changed')
                cached=torch.load(root/'actor-input.pt',weights_only=False)
                require(state_hash(dict(cached.batch.items()))==receipt['tensor_hash'],'P2 cached tensors changed')
                atomic(Path(os.environ['X_P2_RUN'])/'cached-first-batch.json',dict(
                    schema='rung-p2-rollout-substitution.v1',trained_batch_sha256=receipt['sha256'],
                    rollout_log='step-1 fresh generation is discarded; actor trains on X4 actor-input.pt',
                    cache=str(root/'actor-input.pt')))
                # The normal generation call consumes the same first sampling
                # invocation; the actor uses exactly X4's bound inputs instead.
                batch=cached
        if os.environ.get('X_INTEGRATION_SMOKE')=='development-only':
            from kit.x_smoke_batches import paired_input
            batch=paired_input(batch,mode,self.global_steps-1,root,Path(os.environ['X_P2_RUN']))
        result=original(self,batch)
        if mode=='measure':raise Measured()
        if os.environ.get('X_P2_EARLY_VERDICT')=='1' and self.global_steps==4:
            from kit.x_p2_verdict import publish_before_checkpoint
            publish_before_checkpoint(Path(os.environ['X_P2_RUN']))
        return result
    @functools.wraps(fit)
    def fitted(self,*args,**kwargs):
        try:return fit(self,*args,**kwargs)
        except Measured:return None
    cls._update_actor,cls.fit=update,fitted;cls._x_p2_transport=True


def expected_first():
    if os.environ.get('X_TIER2_NATIVE'):
        return None  # Tier 2 freezes X4 lambda, never substitutes X4's rollout.
    path=os.environ.get('X_P2_CALIBRATION')
    if path is None:return None  # CPU adapter; real X5/X6 always bind it.
    require(sha(path)==os.environ['X_P2_CALIBRATION_SHA'],'P2 calibration file changed')
    value=read(path);require(value['variant']==VARIANT,'P2 expected-gradient variant differs')
    return value['seeds'][os.environ['X_P2_SEED']]


def install_worker(cls):
    if not os.environ.get('X_P2_MODE') or getattr(cls,'_x_p2_worker',False):return
    original_init,original_update=cls.init_model,cls.update_actor
    @functools.wraps(original_init)
    def init(self,*args,**kwargs):
        from kit.x_p2_resources import start as resource_start
        resource_start(self)
        result=original_init(self,*args,**kwargs)
        if self._is_actor:
            from kit.p2_l2 import load_anchor
            from kit.x_p2_actor import extend
            from kit.x_ema import assert_teacher_origin
            root=Path(os.environ['X_P2_RUN'])
            receipt=assert_teacher_origin(self.actor_module_fsdp,self.actor.teacher_module,
                self.config.model.path,self.config.model.path,self.config.model.path)
            atomic(root/'ema-origin'/f'rank{self.rank}.json',dict(receipt,variant=VARIANT))
            if os.environ['X_P2_MODE']!='control':
                anchor=load_anchor(self.actor.actor_module,os.environ['X_P2_ANCHOR'])
                extend(self.actor,anchor,float(os.environ.get('X_P2_LAMBDA','0')),
                    measurement=os.environ['X_P2_MODE']=='measure',expected=expected_first(),audit=bool(os.environ.get('X_P2_SMOKE_ROLE')))
                from kit.x_p2_resources import install as resource_install
                resource_install(self,anchor)
            if os.environ.get('X_P2_SMOKE_ROLE'):
                from kit.x_p2_smoke_runtime import initialize
                initialize(self)
        return result
    @functools.wraps(original_update)
    def update(self,data,*args,**kwargs):
        import torch
        from kit.x_p2_actor import MeasurementReady
        import time
        smoke_clock=time.monotonic()
        mode=os.environ['X_P2_MODE'];batch=Path(os.environ['X_P2_BATCH']);run=Path(os.environ['X_P2_RUN'])
        manager=self.checkpoint_manager
        first=getattr(self.actor,'_v4_completed_updates',0)==0
        # Snapshot semantic state without a second model; checkpoint manager
        # provides the same per-rank RNG state that native D saves/restores.
        def hashes():
            if hasattr(self,'_x_cpu_state_hashes'):return self._x_cpu_state_hashes()
            from kit.x_resume import live_worker
            from verl.utils.checkpoint import fsdp_checkpoint_manager as m
            def model_hash(model):
                with m.get_fsdp_state_ctx(model,m.StateDictType.SHARDED_STATE_DICT,
                    m.ShardedStateDictConfig(offload_to_cpu=True),m.ShardedOptimStateDictConfig(offload_to_cpu=True)):
                    return state_hash(model.state_dict())
            return dict(model=model_hash(manager.model),optimizer=state_hash(manager.optimizer.state_dict()),
                scheduler=state_hash(manager.lr_scheduler.state_dict()),teacher=model_hash(self.actor.teacher_module),rng=state_hash(manager.get_rng_state()))
        if mode=='train' and os.environ.get('X_TIER2_NATIVE') and not self.actor._p2_records:
            prior=run/f'p2-rank{self.rank}.json'
            completed=getattr(self.actor,'_v4_completed_updates',0)
            if completed:
                value=read(prior)
                require(len(value['updates'])>=completed,'tier-2 P2 history missing at resume')
                self.actor._p2_records=value['updates'][:completed]
        if first and mode!='prepare' and not os.environ.get('X_TIER2_NATIVE'):
            rng_path=batch/f'rng-r{self.rank}.pt'
            if mode=='measure':torch.save(manager.get_rng_state(),rng_path)
            else:
                expected=read(batch/f'measure-r{self.rank}.json')
                require(sha(rng_path)==expected['rng_file_sha256'],'P2 RNG receipt changed')
                manager.load_rng_state(torch.load(rng_path,weights_only=False))
        if mode!='measure':
            result=original_update(self,data,*args,**kwargs)
            from kit.x_p2_resources import snapshot
            snapshot(self,'update-complete')
            if os.environ.get('X_P2_SMOKE_ROLE'):
                from kit.x_p2_smoke_runtime import updated
                updated(self,result,smoke_clock)
            if first and mode=='control' and expected_first() is not None:
                from kit.p2_l2 import deviation
                value=result.meta_info['metrics']['actor/grad_norm']
                if isinstance(value,(list,tuple)):value=sum(value)/len(value)
                atomic(run/f'control-first-r{self.rank}.json',dict(
                    task_gradient=deviation(float(value),expected_first()['task_grad_norm']),
                    lambda_frozen=True,numerical_policy='unchanged core D'))
            if mode=='train':
                atomic(run/f'p2-rank{self.rank}.json',dict(schema='rung-p2-training.v1',variant=VARIANT,
                    updates=self.actor._p2_records,completed_updates=self.actor._v4_completed_updates,teacher_regularised=False,
                    max_memory_reserved=memory_reserved()))
            return result
        before=hashes();rng=manager.get_rng_state()
        try:
            original_update(self,data,*args,**kwargs)
            raise ValueError('P2 measurement failed to stop before clipping')
        except MeasurementReady:
            manager.load_rng_state(rng)
            after=hashes();require(before==after,'P2 X4 mutated model/optimizer/scheduler/EMA/RNG')
            record=self.actor._p2_records[-1]
            from kit.x_p2_resources import snapshot
            value=snapshot(self,'measurement-complete')
            if value is not None:atomic(batch/f'geometry-r{self.rank}.json',value)
            atomic(batch/f'measure-r{self.rank}.json',dict(schema='rung-p2-measure.v1',variant=VARIANT,
                **record,rank=self.rank,state_before=before,state_after=after,max_memory_reserved=memory_reserved(),
                rng_file_sha256=sha(batch/f'rng-r{self.rank}.pt'),optimizer_steps=0,scheduler_steps=0,ema_steps=0))
            from verl import DataProto
            return DataProto(meta_info={'metrics':{'p2/measurement_only':1.}})
        finally:
            manager.load_rng_state(rng)
            self.actor.actor_optimizer.zero_grad(set_to_none=True)
    cls.init_model,cls.update_actor=init,update;cls._x_p2_worker=True
    from kit.x_p2_resources import install_save
    install_save(cls)


def memory_reserved():
    import torch
    return int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
