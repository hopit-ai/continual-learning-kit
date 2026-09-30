"""Run the pinned SDPO fork's supervised trainer (verl.trainer.fsdp_sft_trainer) with ONE runtime patch.

The trainer builds its data loaders as `StatefulDataLoader(..., pin_memory=True, pin_memory_device=...)`. With the
pinned stack (torchdata 0.11.0 under torch 2.9) the pin-memory thread crashes on its first batch:
`_pin_memory_loop() takes 4 positional arguments but 5 were given` (found by the K2b smoke, 29 September 2026). The
GRPO trainer never takes that path. This entry turns pinned memory off in StatefulDataLoader before the trainer is
imported, so batches are copied to the GPU from ordinary rather than page-locked host memory. Nothing else changes:
not the data, the order, the loss, the optimiser, or any arithmetic. The authors' files are not edited.

Launched by kit/run_sft.sh as `python -m torch.distributed.run ... kit/sft_entry.py --config-name sft_trainer ...`;
every argument after the script is the trainer's own Hydra command line.
"""
from __future__ import annotations

import torchdata.stateful_dataloader as _stateful

_original_init = _stateful.StatefulDataLoader.__init__


def _unpinned_init(self, *args, **kwargs):
    kwargs["pin_memory"] = False
    kwargs.pop("pin_memory_device", None)
    _original_init(self, *args, **kwargs)


_stateful.StatefulDataLoader.__init__ = _unpinned_init

from verl.trainer import fsdp_sft_trainer  # noqa: E402  (imported after the patch, on purpose)

if __name__ == "__main__":
    fsdp_sft_trainer.main()
