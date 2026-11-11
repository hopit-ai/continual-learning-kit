"""Importable Ray classes selected only by the deletion entry point."""
from kit import sdft_verl as v
from kit.c_sdft_runner import CTaskRunner
from deletion_tuning.deletion_actor import update_policy
from deletion_tuning.deletion_worker import install_worker_patch
import os
from copy import deepcopy
from pathlib import Path


def registered_lr():
    from deletion_tuning.deletion_lr import FALLBACK, frozen_rate
    mode = os.environ.get('DELETION_LR_MODE', 'scientific')
    if mode == 'qualify':
        if os.environ['DELETION_SEED'] != '101':
            raise ValueError('LR qualification uses the first paired seed 101')
        return FALLBACK
    if mode != 'scientific':
        raise ValueError('unknown deletion LR mode')
    return frozen_rate(Path(os.environ['DELETION_LR_RECEIPT']), os.environ['DELETION_LR_SHA256'])


def validation_config(config):
    """Normalize only the declared mask-LR departure for inherited I0 checks."""
    result = deepcopy(config)
    actual = result['actor_rollout_ref']['actor']['optim']['lr']
    if actual != registered_lr():
        raise ValueError('mask LR differs from the frozen qualification receipt')
    result['actor_rollout_ref']['actor']['optim']['lr'] = 1e-5
    return result


class DeletionPPOActor(v.dp_actor.DataParallelPPOActor):
    def update_policy(self, data):
        if os.environ.get('DELETION_LR_MODE') == 'qualify':
            from deletion_tuning.deletion_actor import qualification_update
            return qualification_update(self, data)
        return update_policy(self, data)


class DeletionActorRolloutRefWorker(v.SdftActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_worker_patch()
        # The pinned worker imports this symbol inside init_model, after __init__.
        # This Ray process belongs exclusively to the deletion arm.
        import verl.workers.actor as actor
        actor.DataParallelPPOActor = DeletionPPOActor
        from kit.v4_timing import install_telemetry
        install_telemetry(DeletionPPOActor, updates='update_policy')


class DeletionTaskRunner(CTaskRunner):
    def run(self, config):
        if config.actor_rollout_ref.actor.strategy != 'fsdp2':
            raise ValueError('deletion requires inherited FSDP2')
        if config.actor_rollout_ref.actor.optim.lr != registered_lr():
            raise ValueError('mask LR differs from the registered qualification mode')
        from kit import sdft_objective as objective, v4_contract as contract
        original_objective, original_contract = objective.check_trainer_config, contract.check_resolved
        objective.check_trainer_config = lambda c: original_objective(validation_config(c))
        contract.check_resolved = lambda c, *args, **kwargs: original_contract(validation_config(c), *args, **kwargs)
        try:
            return super().run(config)
        finally:
            objective.check_trainer_config, contract.check_resolved = original_objective, original_contract

    def add_actor_rollout_worker(self, config):
        _, group_cls = super().add_actor_rollout_worker(config)
        self.role_worker_mapping[v.ray_trainer.Role.ActorRolloutRef] = v.ray.remote(
            DeletionActorRolloutRefWorker)
        return DeletionActorRolloutRefWorker, group_cls
