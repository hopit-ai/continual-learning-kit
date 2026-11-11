"""Deletion-only ramp and EMA adapter around the unchanged D update loop."""
from kit.sdft_actor import update_policy as sdft_update_policy
from deletion_tuning.deletion_mask import set_step, update_teacher


def teacher_rate(config):
    rate = float(config.self_distillation['teacher_update_rate'])
    if rate != .01:
        raise ValueError('deletion requires registered teacher_update_rate=0.01')
    return rate


def update_policy(self, data):
    """Inherit D's loss, optimizer and finite-step gate; replace only its EMA.

    The override belongs to this actor instance and is restored even on errors.
    No sealed module, class or function is changed by importing this adapter.
    """
    rate = teacher_rate(self.config)
    set_step(self.actor_module, getattr(self, '_v4_completed_updates', 0))
    if self.teacher_module is None:
        raise ValueError('deletion requires a separate EMA teacher')
    if not getattr(self, '_deletion_initial_masks_checked', False):
        from deletion_tuning.deletion_mask import assert_matching_masks
        if getattr(self, '_v4_completed_updates', 0) == 0:
            assert_matching_masks(self.actor_module, self.teacher_module)
        self.teacher_module.requires_grad_(False)
        self._deletion_initial_masks_checked = True
    self.teacher_module.eval()
    had_override = '_update_teacher' in self.__dict__
    previous = self.__dict__.get('_update_teacher')
    self._update_teacher = lambda: update_teacher(self.actor_module, self.teacher_module, rate=rate)
    try:
        return sdft_update_policy(self, data)
    finally:
        if had_override:
            self._update_teacher = previous
        else:
            del self._update_teacher


def qualification_update(self, data):
    """Run on the partner's first TRAINING batch, then terminate qualification."""
    import json
    import os
    from pathlib import Path
    import torch.distributed as dist
    from deletion_tuning.deletion_lr import run_finder
    record = run_finder(self, data)
    if dist.get_rank() == 0:
        with Path(os.environ['DELETION_LR_RECEIPT']).open('x') as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write('\n')
    dist.barrier()
    raise RuntimeError('LR qualification complete. Freeze the receipt hash and start a fresh scientific stage.')
