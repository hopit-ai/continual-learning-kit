"""Plan v4 arm D's trainer entry: verl's own Hydra entry with kit/sdft_verl.py's runner (see that module).

Launched by kit/run_sdft.sh as `python kit/sdft_entry.py --config-name sdpo <overrides>`, with kit/ and the pinned
checkout on PYTHONPATH; every argument is the trainer's own Hydra command line, so `--cfg job --resolve` prints the
resolved configuration and trains nothing, as it does for `python -m verl.trainer.main_ppo`.
"""
from __future__ import annotations

from kit import sdft_verl

if __name__ == "__main__":
    sdft_verl.main()
