"""Strict, process-local execution policy for the final Rung P smoke attempt.

Keep torch out of module imports: the container environment must precede it.
Older attempts do not opt into this policy.
"""
import os
import random

DETERMINISTIC_ENV = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '101'}


def configure(settings):
    if not settings.get('deterministic', False):
        return {}
    if settings['attention_implementation'] != 'eager' or settings['deterministic_env'] != DETERMINISTIC_ENV:
        raise ValueError('attempt 3 requires registered eager attention and deterministic environment')
    # Fail rather than pretending a late workspace/hash-seed change was effective.
    if any(os.environ.get(k) != v for k, v in DETERMINISTIC_ENV.items()):
        raise ValueError('set deterministic environment before starting the Python process')
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])  # also seeds all CUDA devices
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                attention_implementation='eager',
                env_vars={k: os.environ[k] for k in DETERMINISTIC_ENV},
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark, seed=settings['seed'])
