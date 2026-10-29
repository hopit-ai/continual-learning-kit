"""X-only save/restore qualification at the live update-10 boundary."""
from pathlib import Path
import functools
import os
from kit import m2_train
from kit.s_common import atomic, read, require
from kit.v4_restore_check import state_hash


def boundary_step(event='saved'):
    if os.environ.get('X_INTEGRATION_SMOKE'):
        require(os.environ['X_INTEGRATION_SMOKE']=='development-only' and
            os.environ.get('V4_PROFILE')=='technical-smoke','short X boundary is development-only')
        return 2
    key='X_TIER2_RESUME_STEP' if event == 'loaded' else 'X_TIER2_BOUNDARY_STEP'
    if os.environ.get(key):
        step=int(os.environ[key])
        require(step in (10,20,30), 'unregistered tier-2 boundary')
        return step
    return 10


def total_steps():return 3 if boundary_step()==2 else 40


def resume_slots_command(cmd, checkpoint=None):
    if checkpoint is None: return list(cmd)
    checkpoint = Path(checkpoint)
    require(checkpoint.name == f'global_step_{boundary_step("loaded")}', 'allocation continuation must resume its registered boundary')
    require((checkpoint/'data.pt').is_file() and (checkpoint/'actor/v4-ema').is_dir(),
            'full update-10 state required; weights-only restart forbidden')
    from kit.v4_retry import saved_state
    saved_state(checkpoint, 'D')
    return [v for v in cmd if not v.startswith(('trainer.resume_mode=', 'trainer.resume_from_path='))]+[
        'trainer.resume_mode=resume_path', 'trainer.resume_from_path='+str(checkpoint)]


def live_worker(worker, event='saved'):
    """Hash live local FSDP shards, including optimizer, RNG, scheduler and EMA.

    The optional context is only a CPU fixture adapter; production uses exactly
    the sharded state context used by the kit's real-stack restoration probe.
    """
    manager=worker.checkpoint_manager
    context=getattr(manager,'_x_state_context',None)
    if context is None:
        from verl.utils.checkpoint import fsdp_checkpoint_manager as module
        def context(model):
            return module.get_fsdp_state_ctx(model,module.StateDictType.SHARDED_STATE_DICT,
                module.ShardedStateDictConfig(offload_to_cpu=True),module.ShardedOptimStateDictConfig(offload_to_cpu=True))
    with context(manager.model):
        components=dict(model=state_hash(manager.model.state_dict()),optimizer=state_hash(manager.optimizer.state_dict()))
    with context(worker.actor.teacher_module): components['EMA']=state_hash(worker.actor.teacher_module.state_dict())
    scheduler=manager.lr_scheduler.state_dict()
    components.update(rng=state_hash(manager.get_rng_state()),scheduler=state_hash(scheduler))
    step=boundary_step(event)
    require(worker.actor._v4_completed_updates == step, 'actor boundary update mismatch')
    require(scheduler['last_epoch'] == step, 'scheduler boundary update mismatch')
    return dict(rank=worker.rank,step=step,completed_updates=worker.actor._v4_completed_updates,
                scheduler_step=scheduler['last_epoch'],components=components)


def install_worker_boundary(cls):
    """Wrap outside v4_state: its EMA and actor counter load must finish first."""
    if not os.environ.get('M2_ADDON_BOUNDARY') or getattr(cls,'_x_boundary_worker',False): return
    for method,event in (('save_checkpoint','saved'),('load_checkpoint','loaded')):
        original=getattr(cls,method)
        @functools.wraps(original)
        def wrapped(self,local_path,*args,_original=original,_event=event,**kwargs):
            result=_original(self,local_path,*args,**kwargs)
            if local_path is not None and Path(local_path).parent.name == f'global_step_{boundary_step(_event)}':
                value=live_worker(self,_event)
                boundary=Path(os.environ.get('X_TIER2_RESUME_BOUNDARY',os.environ['M2_ADDON_BOUNDARY']) if _event == 'loaded' else os.environ['M2_ADDON_BOUNDARY'])
                if _event == 'loaded':
                    expected=read(boundary)['ranks'][str(self.rank)]
                    require(value == expected, 'restored live worker state differs at rank '+str(self.rank))
                atomic(boundary.parent/'boundary-state'/f'{_event}-r{self.rank}.json',value)
            return result
        setattr(cls,method,wrapped)
    cls._x_boundary_worker=True


def loader_batches(state):
    # torchdata's single-worker and multi-worker StatefulDataLoader schemas.
    if '_num_yielded' in state: return state['_num_yielded']
    snapshot=state.get('_snapshot',{})
    if '_snapshot_step' in snapshot: return snapshot['_snapshot_step']+state.get('_steps_since_snapshot',0)
    raise ValueError('unrecognised dataloader yielded-batch counter')


def trainer_state(trainer,event):
    step=boundary_step(event)
    require(trainer.global_steps == step and trainer.total_training_steps == total_steps(), 'trainer boundary/total dose mismatch')
    loader=trainer.train_dataloader.state_dict()
    require(loader_batches(loader) == step, 'dataloader boundary position mismatch')
    boundary=Path(os.environ.get('X_TIER2_RESUME_BOUNDARY',os.environ['M2_ADDON_BOUNDARY']) if event == 'loaded' else os.environ['M2_ADDON_BOUNDARY'])
    width=int(trainer.config.trainer.n_gpus_per_node)*int(trainer.config.trainer.nnodes)
    ranks={str(rank):read(boundary.parent/'boundary-state'/f'{event}-r{rank}.json') for rank in range(width)}
    require(all(set(r['components']) == {'model','optimizer','rng','scheduler','EMA'} and r['step']==step and
        r['completed_updates']==r['scheduler_step']==step for r in ranks.values()), 'incomplete live-state qualification')
    return dict(ranks=ranks,dataloader=state_hash(loader),dataloader_batches=step)


def install_update10_boundary(cls):
    if not os.environ.get('M2_ADDON_BOUNDARY') or getattr(cls, '_m2_addon_prefix', False): return
    save, load, fit = cls._save_checkpoint, cls._load_checkpoint, cls.fit
    class PrefixSaved(Exception): pass
    @functools.wraps(save)
    def saved(self, *args, **kwargs):
        result = save(self, *args, **kwargs)
        if os.environ.get('M2_ADDON_PREFIX') == '1' and self.global_steps == boundary_step():
            from kit.v4_retry import saved_state
            checkpoint=Path(self.config.trainer.default_local_dir)/f'global_step_{boundary_step()}'
            live=trainer_state(self,'saved')
            atomic(Path(os.environ['M2_ADDON_BOUNDARY']), dict(update=boundary_step(),total_training_steps=total_steps(),full_state=True,
                checkpoint_sha256=saved_state(checkpoint,'D',width=len(live['ranks'])),**live))
            # Write before throwing: the outer telemetry hook cannot run its tail.
            atomic(Path(self.config.trainer.default_local_dir).parent/f'env/saved-state-step{boundary_step()}.json',
                dict(step=boundary_step(),arm='D',path=str(checkpoint),state_sha256=read(Path(os.environ['M2_ADDON_BOUNDARY']))['checkpoint_sha256'],
                     contents=['model','optimizer','extra','dataloader','EMA']))
            raise PrefixSaved()
        return result
    @functools.wraps(load)
    def loaded(self,*args,**kwargs):
        result=load(self,*args,**kwargs)
        if self.config.trainer.resume_mode == 'resume_path':
            boundary=Path(os.environ.get('X_TIER2_RESUME_BOUNDARY',os.environ['M2_ADDON_BOUNDARY']))
            expected=read(boundary)
            actual=trainer_state(self,'loaded')
            require(actual == {k:expected[k] for k in actual}, 'restored trainer/dataloader state differs before update eleven')
            atomic(boundary.parent/'restore-qualified.json',dict(
                status='PASS',before_update=boundary_step('loaded')+1,checkpoint_sha256=expected['checkpoint_sha256'],**actual))
        return result
    @functools.wraps(fit)
    def fitted(self, *args, **kwargs):
        try: return fit(self, *args, **kwargs)
        except PrefixSaved:
            if os.environ.get('X_INTEGRATION_SMOKE')=='development-only':
                # The planned exit skips iterator exhaustion. StatefulDataLoader
                # retains its live iterator, whose workers must be joined before
                # the Ray trainer process exits. Saved loader state is untouched.
                for name in ('train_dataloader','val_dataloader'):
                    loader=getattr(self,name,None)
                    iterator=getattr(loader,'_iterator',None)
                    shutdown=getattr(iterator,'_shutdown_workers',None)
                    if shutdown is not None:shutdown()
            return None
    cls._save_checkpoint, cls._load_checkpoint, cls.fit = saved, loaded, fitted
    cls._m2_addon_prefix = True


def checked_resume(config):
    """Permit only the bound prefix; live restoration must qualify before update 11."""
    trainer = config['trainer']
    if trainer['resume_mode'] == 'disable': return None
    require(trainer['resume_mode'] == 'resume_path', 'X refuses automatic checkpoint selection')
    path = Path(trainer['resume_from_path'])
    step=boundary_step('loaded')
    expected = Path(trainer['default_local_dir'])/f'global_step_{step}'
    require(path.resolve() == expected.resolve(), 'X continuation must use its own update-10 state')
    boundary=Path(os.environ.get('X_TIER2_RESUME_BOUNDARY',os.environ['M2_ADDON_BOUNDARY']))
    receipt=read(boundary)
    from kit.v4_retry import saved_state
    require(receipt['update']==step and receipt['total_training_steps']==total_steps() and receipt['full_state'] and
        receipt['checkpoint_sha256']==saved_state(path,'D',width=len(receipt['ranks'])), 'prefix file-tree hash differs')
    return str(path)
