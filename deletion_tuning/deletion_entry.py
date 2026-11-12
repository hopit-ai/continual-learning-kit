"""Mask arm on the existing long-context D runner and native checkpoints.

Use identical I0 Hydra overrides, plus DELETION_CALIBRATION (tokenized task
prompts) and DELETION_SEED. The inherited C runner validates every D setting.
"""
import os
import sys
import json
from pathlib import Path


def main():
    from kit.trace_common import local_file, stage_command
    # Immutable input bytes only; all native checkpoints remain durable.
    files=[]
    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        key,value=arg.split('=',1)
        if key.lstrip('+') in ('data.train_files','data.val_files'):
            try:
                paths=json.loads(value)
            except ValueError:
                paths=value
            for path in paths if isinstance(paths,list) else [paths]:
                if Path(path).is_file():
                    files.append(path)
    sys.argv=stage_command(sys.argv,files)
    if os.environ.get('DELETION_CALIBRATION'):
        os.environ['DELETION_CALIBRATION']=str(local_file(os.environ['DELETION_CALIBRATION']))
    from omegaconf import OmegaConf
    from kit import sdft_verl as v
    from deletion_tuning.deletion_runner import DeletionTaskRunner, registered_lr
    from deletion_tuning.deletion_worker import validate_calibration
    validate_calibration(os.environ['DELETION_CALIBRATION'])
    registered_lr()
    os.environ['V4_CHECKPOINT_STATE'] = '1'
    v.check_pinned()

    original = v.main_ppo.run_ppo
    def run(config, *args, **kwargs):
        path = 'ray_kwargs.ray_init.runtime_env.env_vars'
        env = dict(OmegaConf.select(config, path, default={}))
        for key in ('DELETION_CALIBRATION', 'DELETION_SEED', 'V4_CHECKPOINT_STATE',
                    'DELETION_LR_MODE', 'DELETION_LR_RECEIPT', 'DELETION_LR_SHA256'):
            if key not in os.environ:
                continue
            env[key] = os.environ[key]
        OmegaConf.update(config, path, env, merge=False, force_add=True)
        return original(config, *args, **kwargs)
    v.main_ppo.run_ppo = run
    v.main_ppo.TaskRunner = DeletionTaskRunner
    v.main_ppo.main()


if __name__ == '__main__':
    main()
