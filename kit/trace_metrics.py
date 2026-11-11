"""TRACE metric adapter for locally staged, hash-bound upstream metric code.

Node rows supply the exact task metric callable and scale in metric_authority.
No upstream source or dataset is downloaded by this module.
"""
import importlib.util
import math
import sys
from kit.trace_common import bound_file, need


def score(row, prediction):
    authority = row['metric_authority']
    path = bound_file(authority)
    # Upstream evaluations import their sibling metrics.py by absolute name.
    for ref in authority.get('dependencies',[]):
        bound_file(ref)
    spec = importlib.util.spec_from_file_location('trace_upstream_metric',path)
    module = importlib.util.module_from_spec(spec)
    original = list(sys.path)
    try:
        sys.path.insert(0,str(path.parent))
        spec.loader.exec_module(module)
        fn = getattr(module,authority['function'])
        args = [[prediction],[row['answer']]]
        if authority.get('uses_source'):
            args.insert(0,[row['prompt']])
        result = fn(*args)
    finally:
        sys.path[:] = original
    value = result[authority['key']]*authority['scale_to_100']
    need(type(value) in (float,int) and math.isfinite(value) and 0<=value<=100,
         'upstream metric range differs')
    return value


def general(row,prediction):
    from kit.scorers import score as inherited
    panel = {'instruction':'ifeval','maths':'math'}.get(row['panel'],row['panel'])
    return 100*int(inherited(panel,prediction,row)['correct'])
