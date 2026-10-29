"""Opt-in add-on prefix boundary; C's runner/protocol bytes remain unchanged."""
from kit import sdft_verl as v
from kit.c_sdft_runner import CTaskRunner


class AddonTaskRunner(CTaskRunner):
    def run(self, config):
        from kit.x_resume import install_update10_boundary, checked_resume
        from kit import v4_contract
        from omegaconf import OmegaConf
        from kit.x_p2_runtime import install_trainer
        install_trainer(v.ray_trainer.RayPPOTrainer)
        install_update10_boundary(v.ray_trainer.RayPPOTrainer)
        from kit.x_smoke_checkpoint import install as install_checkpoint_accounting
        install_checkpoint_accounting(v.ray_trainer.RayPPOTrainer)
        resume = checked_resume(OmegaConf.to_container(config,resolve=True))
        original = v4_contract.check_resolved
        def validate(c,*args,**kwargs):
            return original(c,*args,**dict(kwargs,resume_path=resume))
        v4_contract.check_resolved = validate
        try: return super().run(config)
        finally: v4_contract.check_resolved = original

    def add_actor_rollout_worker(self, config):
        actor, group = super().add_actor_rollout_worker(config)
        import os
        if os.environ.get('X_P_POST_MODEL') or os.environ.get('M2_ADDON_BOUNDARY') or os.environ.get('X_P2_MODE') or os.environ.get('X_DETERMINISTIC_SEED') or os.environ.get('X_INTEGRATION_SMOKE'):
            from kit.x_worker import AddonWorker
            self.role_worker_mapping[v.ray_trainer.Role.ActorRolloutRef] = v.ray.remote(AddonWorker)
            return AddonWorker, group
        return actor, group
