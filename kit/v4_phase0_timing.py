"""Owner-approved v4 timing policy; observations are never multiplied."""
from contextlib import contextmanager
import math
import os
import json
from pathlib import Path

TIMING_MULTIPLIER = 10

def scale(seconds):
    """Apply the multiplier once to a v3 limit, not to elapsed measurements."""
    return seconds * TIMING_MULTIPLIER

def is_phase0():
    """Shared containment code changes policy only for the phase-0 route."""
    if os.environ.get('V4_PHASE0_MODE')=='1':return True
    path=os.environ.get('V4_ALLOCATION_PLAN')
    if path:
        try:return json.loads(Path(path).read_text()).get('phase')=='phase0'
        except (OSError,ValueError):pass
    return False

def is_finqa():
    if is_phase0():return False
    if os.environ.get('V4_FINQA_CAMPAIGN')=='1':return True
    path=os.environ.get('V4_ALLOCATION_PLAN')
    if path:
        try:return json.loads(Path(path).read_text()).get('phase') in ('qualification','main')
        except (OSError,ValueError):pass
    return False


def informational_containment():
    return is_phase0() or is_finqa() or os.environ.get('V4_ARCHIVE_TIMING')=='1'


def shared_limit(seconds):
    return scale(seconds) if informational_containment() else seconds

def relaxed_timeout(estimate=None):
    """Unmeasured short limits get ten minutes or five times an available estimate."""
    if estimate is None:return 600
    if isinstance(estimate,bool) or not isinstance(estimate,(int,float)) or not math.isfinite(estimate) or estimate<0:
        raise ValueError('timing estimate must be finite nonnegative seconds')
    return max(600,math.ceil(5*estimate))


@contextmanager
def archive_timing(evidence):
    enabled=any(evidence.json(n).get('informational') is True for n in evidence.files if n.endswith('containment-selftest.json'))
    old=os.environ.get('V4_ARCHIVE_TIMING')
    if enabled:os.environ['V4_ARCHIVE_TIMING']='1'
    try:yield
    finally:
        if old is None:os.environ.pop('V4_ARCHIVE_TIMING',None)
        else:os.environ['V4_ARCHIVE_TIMING']=old


def distributed_defaults():
    """v4 F/R subprocesses: pinned SFT initializes groups with PyTorch defaults."""
    import datetime
    import torch.distributed.distributed_c10d as c10d
    c10d.default_pg_timeout=datetime.timedelta(seconds=scale(1800))
    c10d.default_pg_nccl_timeout=datetime.timedelta(seconds=scale(600))
