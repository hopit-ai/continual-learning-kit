"""Private per-measurement wall times; reductions charge their GPU work."""
import time
from kit.p3b_common import atomic,require


def record(path,started,standin,assignment):
    seconds=time.monotonic()-started
    atomic(path.with_suffix('.cost.json'),dict(seconds=seconds,gpu_hours=0 if standin else seconds/3600,assignment=assignment))


def hours(paths,standin,assignment):
    if standin:return 0
    import json
    total=0
    for path in paths:
        value=json.loads(path.with_suffix('.cost.json').read_text())
        require(value['assignment']==assignment,'measurement cost belongs to another allocation')
        total+=value['gpu_hours']
    return total
