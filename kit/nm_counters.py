"""Checkpoint-safe counters shared by NM and its TRACE/TS worker subclasses."""
from kit.m3_actor import count_scheduler_steps


def install_counters(worker):
    worker._nm_steps = dict(optimizer=0, scheduler=0, ema=0)
    for owner, method, key in ((worker.actor_optimizer, 'step', 'optimizer'),
                               (worker.actor, '_update_teacher', 'ema')):
        original = getattr(owner, method)

        def counted(*args, _original=original, _key=key, **kwargs):
            value = _original(*args, **kwargs)
            worker._nm_steps[_key] += 1
            return value

        counted._wrapped_by_lr_sched = True
        setattr(owner, method, counted)

    def scheduler_step():
        # Read the current dictionary: native resume replaces it after loading.
        worker._nm_steps['scheduler'] += 1

    count_scheduler_steps(worker.actor_lr_scheduler, scheduler_step)
