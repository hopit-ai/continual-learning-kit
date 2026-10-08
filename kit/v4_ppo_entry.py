"""S's pinned Hydra entry with v4 admission before workers and actual update receipts."""
import os
import torch
from omegaconf import OmegaConf
import ray
import verl.trainer.main_ppo as main_ppo
import verl.workers.actor.dp_actor as dp_actor
import verl.trainer.ppo.ray_trainer as ray_trainer
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from kit.v4_contract import check_resolved
from kit.v4_run import check_data


def install_counted_actor():
    cls = dp_actor.DataParallelPPOActor
    if getattr(cls, '_v4_counted', False): return
    original = cls.update_policy
    def update(self, data):
        step = self._optimizer_step
        def counted_step():
            norm = step()
            if bool(torch.isfinite(norm).item()):
                self._v4_completed_updates = getattr(self, '_v4_completed_updates', 0) + 1
            return norm
        self._optimizer_step = counted_step
        try: metrics = original(self, data)
        finally: self._optimizer_step = step
        metrics['v4/completed_optimizer_updates'] = getattr(self, '_v4_completed_updates', 0)
        return metrics
    cls.update_policy = update
    cls._v4_counted = True


class V4SWorker(AsyncActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        install_counted_actor()
        from kit.v4_timing import install_telemetry
        install_telemetry(dp_actor.DataParallelPPOActor,updates='update_policy')
        install_telemetry(AsyncActorRolloutRefWorker,memory_methods=('init_model','save_checkpoint'))
        from kit.v4_state import install_worker
        install_worker(AsyncActorRolloutRefWorker)
        super().__init__(*args, **kwargs)


class V4SRunner(main_ppo.TaskRunner):
    def run(self, config):
        check_resolved(OmegaConf.to_container(config, resolve=True), 'S', profile_name=os.environ.get('V4_PROFILE','scientific'))
        check_data({**os.environ, 'ARM':'S'})
        from kit.v4_timing import install_telemetry
        from kit.v4_restore_check import install_trainer_probe
        install_trainer_probe(ray_trainer.RayPPOTrainer,save='_save_checkpoint',load='_load_checkpoint')
        install_telemetry(ray_trainer.RayPPOTrainer,exports='_save_checkpoint')
        return super().run(config)

    def add_actor_rollout_worker(self, config):
        worker, group = super().add_actor_rollout_worker(config)
        if worker is not AsyncActorRolloutRefWorker: raise ValueError('v4 S requires pinned legacy FSDP worker')
        self.role_worker_mapping[ray_trainer.Role.ActorRolloutRef] = ray.remote(V4SWorker)
        return V4SWorker, group


if __name__ == '__main__':
    main_ppo.TaskRunner = V4SRunner
    main_ppo.main()
