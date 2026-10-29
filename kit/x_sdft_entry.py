#!/usr/bin/env python3
"""X-only C D entry; all inherited config/pin/loss validation stays enabled."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import os
    if os.environ.get('X_P2_MODE') or os.environ.get('X_DETERMINISTIC_SEED'):
        from kit.x_p2_runtime import deterministic
        deterministic()
    from kit.x_p2_smoke_runtime import configure
    configure()
    from kit import sdft_verl as v
    from kit.x_sdft_runner import AddonTaskRunner
    v.check_pinned()
    v.main_ppo.TaskRunner = AddonTaskRunner
    v.main_ppo.main()

if __name__ == '__main__': main()
