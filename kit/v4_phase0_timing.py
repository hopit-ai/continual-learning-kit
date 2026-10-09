"""Phase-0 patience policy; budget admission remains independent and fail closed."""
import math

def relaxed_timeout(estimate=None):
    """Unmeasured short limits get ten minutes or five times an available estimate."""
    if estimate is None:return 600
    if isinstance(estimate,bool) or not isinstance(estimate,(int,float)) or not math.isfinite(estimate) or estimate<0:
        raise ValueError('timing estimate must be finite nonnegative seconds')
    return max(600,math.ceil(5*estimate))
