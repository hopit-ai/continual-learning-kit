"""Exact prospective infrastructure pairs; historical scientific pins stay immutable."""
import hashlib
from functools import lru_cache
import json
import os
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
AUTHORITY='kit/resources/v6/infrastructure-update.json'
SCHEMA='v6-infrastructure-update.v1'


def authority_sha256():
    path=ROOT/AUTHORITY
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def validate(doc):
    if set(doc)!={'schema','statement','files'} or doc['schema']!=SCHEMA or doc['statement']!='scientific definitions unchanged':raise ValueError('invalid infrastructure authority')
    seen=set()
    for row in doc['files']:
        if set(row)!={'path','pinned_sha256','new_sha256','reason'}:raise ValueError('invalid infrastructure pair')
        path=Path(row['path'])
        if path.is_absolute() or '..' in path.parts or path.suffix not in ('.py','.sh') or path.parts[0] not in ('kit','scripts'):raise ValueError('authority may amend implementation code only')
        if not row['reason'] or any(not re.fullmatch('[0-9a-f]{64}',row[k]) for k in ('pinned_sha256','new_sha256')):raise ValueError('invalid infrastructure hash/reason')
        key=(row['path'],row['pinned_sha256'])
        if key in seen:raise ValueError('duplicate infrastructure pair')
        seen.add(key)
    return doc


@lru_cache(maxsize=8)
def validated_bytes(raw):
    # Cache validation by exact immutable bytes, never by mtime or an exposed dict.
    validate(json.loads(raw))


def verify(expected):
    raw=(ROOT/AUTHORITY).read_bytes()
    if not expected or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('infrastructure authority changed after binding')
    validated_bytes(raw)
    return json.loads(raw)


def identity():
    value=authority_sha256()
    if value:verify(value)
    return value


def bound(expected):
    """Pin before checking amended bytes; None is valid only for historical sends."""
    if authority_sha256()!=expected:raise ValueError('infrastructure authority changed after binding')
    if expected:verify(expected)
    return expected


def allowed(path,expected,observed,authority_sha256=None):
    # Callers supply the hash in their protocol/binding, or the node's frozen
    # launch environment. Never treat the authority's own digest as its seal.
    seal=(os.environ.get('KIT_V6_AUTHORITY_SHA') if os.environ.get('KIT_V6A') else
          authority_sha256 or os.environ.get('M2_INFRA_AUTHORITY_SHA256'))
    if not seal:return False
    try:
        name=str(Path(path).resolve().relative_to(ROOT))
        doc=verify(seal)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=observed:return False
        return any(r['path']==name and r['pinned_sha256']==expected and r['new_sha256']==observed for r in doc['files'])
    except (OSError,ValueError,KeyError,TypeError):return False


def activate(out=None):
    """Select the prospective seal, or verify the seal of an existing send first."""
    if os.environ.get('KIT_V6A'):return activate_v6a()
    path=Path(out)/'binding.json' if out is not None else None
    if path is not None and path.is_file():
        doc=json.loads(path.read_text())
        expected=doc.get('infrastructure_update',doc.get('protocol',doc.get('scoring_protocol',{})).get('infrastructure_update'))
        if 'infrastructure_update' in doc or 'runtime_updates' in doc.get('protocol',{}) or 'registration_sha256' in doc or 'scoring_protocol' in doc:bound(expected)
        else:expected=os.environ.get('M2_INFRA_AUTHORITY_SHA256') or identity()
    else:
        expected=os.environ.get('M2_INFRA_AUTHORITY_SHA256') or identity()
        if expected:verify(expected)
    if expected:os.environ['M2_INFRA_AUTHORITY_SHA256']=expected
    else:os.environ.pop('M2_INFRA_AUTHORITY_SHA256',None)
    return expected


def activate_v6a():
    """The staged send's operator message is the sole authority trust anchor."""
    expected=os.environ.get('KIT_V6_AUTHORITY_SHA')
    if not expected:
        raise ValueError('v6a requires KIT_V6_AUTHORITY_SHA from the operator message; never derive it from the file')
    verify(expected)
    os.environ['M2_INFRA_AUTHORITY_SHA256']=expected
    return expected


def compatible_files(expected,observed,seal=None):
    if not isinstance(expected,dict) or set(expected)!=set(observed):return False
    return all(value==observed[name] or allowed(ROOT/name,value,observed[name],seal) for name,value in expected.items())


LANE_CODE=tuple('kit/'+name for name in ('lane_pool.py','lane_evidence.py','util_watch.py','v6_infra.py',
    'm2_assignment.py','m2_gpu.py','ref_guard.py','ref_score_job.py','inference_lifecycle.py',
    'm3_manifest.py','m3_score.py','sp_manifest.py','sp_score.py'))


def runtime_identity():
    return dict(authority=identity(),files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in LANE_CODE})


def verify_runtime(value):
    if not isinstance(value,dict) or set(value)!={'authority','files'} or set(value['files'])!=set(LANE_CODE):
        raise ValueError('incomplete allocation runtime binding')
    bound(value['authority'])
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected for name,expected in value['files'].items()):
        raise ValueError('allocation runtime changed after binding')
    return value['authority']
