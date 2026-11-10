"""A2 frozen identities and small CPU utilities. No outcome discovery or network."""
import hashlib
import json
import math
from pathlib import Path
from kit.a4_common import (KIT, SEEDS, MONTHS, BOUNDARIES, REVISION, require,
                           NotReady, CorrectnessStop, atomic, bind, frozen_files,
                           freeze, t, s, cycle, identity, lock)

REGISTRATION = '276f4f6e04c3d0c228a47e04f80fa1d8de22335a5af5aee1ee68dabab98bf789'
RES = KIT / 'resources/a2'
FAMILIES = ('fcf_margin_change', 'cash_conversion_change')
GENERAL = ('ifeval', 'knowledge', 'math')
TWO = 4.302653
PENDING = 75


def canonical(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':'), sort_keys=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    t.write_jsonl(Path(path), rows)


def checked(record):
    """Every node input is explicitly named by absolute path and full SHA256."""
    if not record or not record.get('path') or not record.get('sha256'):
        raise NotReady('input path/hash binding absent')
    path = Path(record['path'])
    require(path.is_absolute(), 'node input must be an absolute path')
    if not path.is_file():
        raise NotReady('input absent: ' + str(path))
    require(s.file_hash(path) == record['sha256'], 'bound input changed: ' + str(path))
    return path


def protocol():
    from kit.a4_common import protocol as a4_protocol
    paths = [*sorted(KIT.glob('a2_*.py')), KIT/'README-a2.md',
             *sorted((KIT/'jobs/a2').glob('*')), *sorted(RES.glob('*'))]
    return dict(a4=a4_protocol(), a2={str(p.relative_to(KIT)): s.file_hash(p)
                                    for p in paths if p.is_file()})


def arm_names(arm):
    require(arm in ('J', 'P', 'R', 'RP'), 'unknown A2 arm')
    return {'J': ('J-frozen', 'J-current'), 'P': ('P-plain', 'P-replay'),
            'R': ('R-raw', 'R-rewrite'),
            'RP': ('raw-plain', 'P-replay', 'R-rewrite')}[arm]


def contrasts(arm):
    names = arm_names(arm)
    return [(name.split('-')[0], name, names[0]) for name in names[1:]]


def interval(values):
    import statistics
    require(len(values) == 3 and all(type(x) in (int, float) and math.isfinite(x)
                                    for x in values), 'exactly three finite seed values required')
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    radius = TWO * sd / math.sqrt(3)
    return dict(values=values, mean=mean, sd=sd, lower=mean-radius, upper=mean+radius,
                positive=sum(x > 0 for x in values), n=3, df=2, t=TWO)


def benefit(values):
    result = interval(values)
    return dict(result, pass_bar=result['mean'] >= 2 and result['positive'] == 3
                and result['lower'] > 0)


def guard(values):
    result = interval(values)
    return dict(result, pass_bar=result['lower'] >= -1)


def route(improvement, retention):
    """Inputs are supplied later by the registered 4A reading; never read it here."""
    if improvement == 'unresolved' or retention == 'unresolved':
        return None
    require(improvement in ('established', 'not established') and
            retention in ('met', 'failed', 'not established'), 'invalid A2 branch classes')
    if improvement == 'established':
        return 'J' if retention == 'met' else 'P'
    return 'RP' if retention == 'failed' else 'R'
