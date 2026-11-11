"""TRACE-scoped Ray bridge, reusing NM's D loss, EMA and frozen native state."""
import ray
from verl.single_controller.base.decorator import register, Dispatch
from kit import nm_native
from kit.trace_prompt import install


class TraceWorker(nm_native.NMWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        install()
        return super().init_model()


class TraceTaskRunner(nm_native.NMTaskRunner):
    def add_actor_rollout_worker(self,config):
        _,group = super().add_actor_rollout_worker(config)
        self.role_worker_mapping[nm_native.Role.ActorRolloutRef] = ray.remote(TraceWorker)
        return TraceWorker,group

    def run(self,config):
        install()
        return super().run(config)
