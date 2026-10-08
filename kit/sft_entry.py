"""Run the pinned SDPO fork's supervised trainer (verl.trainer.fsdp_sft_trainer) with scoped runtime patches.

The trainer builds its data loaders as `StatefulDataLoader(..., pin_memory=True, pin_memory_device=...)`. With the
pinned stack (torchdata 0.11.0 under torch 2.9) the pin-memory thread crashes on its first batch:
`_pin_memory_loop() takes 4 positional arguments but 5 were given` (found by the K2b smoke, 29 September 2026). The
GRPO trainer never takes that path. This entry turns pinned memory off in StatefulDataLoader before the trainer is
imported, so batches are copied to the GPU from ordinary rather than page-locked host memory.
For v4 F/R only, it also fixes the ordered sampler and absolute warmup/constant scheduler.
The authors' files are not edited.

Launched by kit/run_sft.sh as `python -m torch.distributed.run ... kit/sft_entry.py --config-name sft_trainer ...`;
every argument after the script is the trainer's own Hydra command line.
"""
from __future__ import annotations

import torchdata.stateful_dataloader as _stateful
import os

_original_init = _stateful.StatefulDataLoader.__init__


def _unpinned_init(self, *args, **kwargs):
    kwargs["pin_memory"] = False
    kwargs.pop("pin_memory_device", None)
    _original_init(self, *args, **kwargs)


_stateful.StatefulDataLoader.__init__ = _unpinned_init

from verl.trainer import fsdp_sft_trainer  # noqa: E402  (imported after the patch, on purpose)

if os.environ.get("KIT_SFT_ARM_F") == "1":
    # Arms F/R's parquet is already the common 1,280-exposure schedule. The pinned
    # trainer unconditionally asks DistributedSampler to shuffle; override only
    # its local symbol, leaving authors' source and legacy K2b behavior intact.
    import torch
    from torch.utils.data.distributed import DistributedSampler as _Sampler

    def _ordered_sampler(*args, **kwargs):
        kwargs["shuffle"] = False
        return _Sampler(*args, **kwargs)

    # The pinned WSD helper decays and derives warmup from epoch length.
    # Keep its call site, but use the registered absolute warmup + constant rate.
    from kit.v4_sft import registered_scheduler
    fsdp_sft_trainer.get_wsd_schedule_with_warmup = registered_scheduler
    fsdp_sft_trainer.DistributedSampler = _ordered_sampler
    if os.environ.get('OFFLOAD')=='1':
        # The pinned FSDP2 branch passes CPUOffload to fully_shard, which accepts
        # CPUOffloadPolicy. Placement is the only reviewed OOM retry change.
        from torch.distributed.fsdp import CPUOffloadPolicy
        fsdp_sft_trainer.CPUOffload = lambda **kwargs: CPUOffloadPolicy()

    torch.manual_seed(int(os.environ["KIT_SFT_SEED"]))
    from kit.v4_contract import check_resolved, check_schedule
    from kit.v4_run import check_data
    from kit.v4_teacher import atomic_json
    from pathlib import Path
    import pyarrow.parquet as pq
    _original_run_sft = fsdp_sft_trainer.run_sft
    def _checked_run_sft(config):
        from omegaconf import OmegaConf
        check_resolved(OmegaConf.to_container(config, resolve=True), os.environ['KIT_SFT_ARM'],
                       profile_name=os.environ.get('V4_PROFILE','scientific'))
        # Direct legacy F data also has explicit non-optimized shell admission.
        check_schedule([r for path in config.data.train_files for r in pq.read_table(path).to_pylist()], config.trainer.total_training_steps)
        return _original_run_sft(config)
    fsdp_sft_trainer.run_sft = _checked_run_sft
    _original_training_step = fsdp_sft_trainer.FSDPSFTTrainer.training_step
    def _counted_training_step(self, batch):
        used_lr = self.optimizer.param_groups[0]["lr"]
        step = self.optimizer.step
        def counted_step(*args, **kwargs):
            result = step(*args, **kwargs)
            self._v4_updates = getattr(self, '_v4_updates', 0) + 1
            return result
        self.optimizer.step = counted_step
        try: result = _original_training_step(self, batch)
        finally: self.optimizer.step = step
        result['train/lr'] = used_lr
        result['v4/completed_optimizer_updates'] = getattr(self, '_v4_updates', 0)
        if int(os.environ.get('RANK','0')) == 0:
            path = Path(os.environ['VERL_FILE_LOGGER_PATH']).parent / 'env/optimizer-updates.json'
            atomic_json(path, {'completed_optimizer_updates':result['v4/completed_optimizer_updates']})
        return result
    fsdp_sft_trainer.FSDPSFTTrainer.training_step = _counted_training_step
    from kit.v4_state import install_sft
    install_sft(fsdp_sft_trainer.FSDPSFTTrainer)
    from kit.v4_timing import install_telemetry
    install_telemetry(fsdp_sft_trainer.FSDPSFTTrainer,updates='training_step',exports='save_checkpoint',
                      memory_methods=('__init__','save_checkpoint'))

if __name__ == "__main__":
    fsdp_sft_trainer.main()
