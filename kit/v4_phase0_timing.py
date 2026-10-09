"""Owner-approved phase-0 timing policy; observations are never multiplied."""
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

def shared_limit(seconds):
    return scale(seconds) if is_phase0() else seconds

def relaxed_timeout(estimate=None):
    """Unmeasured short limits get ten minutes or five times an available estimate."""
    if estimate is None:return 600
    if isinstance(estimate,bool) or not isinstance(estimate,(int,float)) or not math.isfinite(estimate) or estimate<0:
        raise ValueError('timing estimate must be finite nonnegative seconds')
    return max(600,math.ceil(5*estimate))
