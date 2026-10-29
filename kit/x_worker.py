"""P-only worker; the inherited SDFT worker still owns every training operation."""
from kit import sdft_verl as v


class AddonWorker(v.SdftActorRolloutRefWorker):
    def __init__(self,*args,**kwargs):
        from kit.x_p2_runtime import deterministic
        deterministic()
        from kit.x_p2_smoke_runtime import configure
        configure()
        super().__init__(*args,**kwargs)
        import os
        from kit.x_resume import install_worker_boundary
        install_worker_boundary(v.AsyncActorRolloutRefWorker)
        from kit.x_p2_runtime import install_worker
        install_worker(v.AsyncActorRolloutRefWorker)
        if os.environ.get('X_P_POST_MODEL'):
            from kit.x_ema import install_init_guard
            install_init_guard(v.AsyncActorRolloutRefWorker)
