"""Owner plan 15.21: observations warn; identity, write safety and actual spend stop."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import shutil
import time

CEILING = 1000
PLANNED_QUALIFICATION = 100
_context = ContextVar('finqa_observations', default=None)
_context_work = ContextVar('finqa_observation_work', default=None)

class CorrectnessStop(ValueError):
    pass

def enabled():
    from kit.v4_phase0_timing import is_finqa, is_phase0
    return not is_phase0() and (is_finqa() or _context.get() is not None)

def correctness(message):
    text=str(message).lower()
    if any(word in text for word in ('context source','inference installed-package','software input','package version','runtime version')):return False
    # Identity refers to model/data bytes, not software or scheduler observations.
    # An unsafe socket or linked observation is not itself an outside-WORK
    # write. Writers use confine(); retirement retains bytes when its proofs
    # cannot be verified, rather than turning that finding into a run ban.
    return ('outside work' in text
            or 'disk quota exceeded' in text or '[errno 122]' in text or 'disk exhaustion' in text or 'no space left on device' in text or '[errno 28]' in text or 'actual 1000' in text
            or ('phase0 return' in text and any(w in text for w in ('sha','changed','differs')))
            or any(word in text for word in ('model', 'weight', 'tokenizer', 'pool', 'source', 'dataset', 'data hash', '8b revision', 'teacher revision',
                                             'input changed', 'prepared input', 'schedule seed/hash', 'checkpoint changed',
                                             'checkpoint shard', 'checkpoint identity', 'checkpoint differs', 'export identity',
                                             'export reload identity', 'export changed during', 'evaluation reload differs',
                                             'frozen source', 'prompt changed', 'panel hash', 'corpus manifest'))
               and any(word in text for word in ('hash', 'changed', 'differ', 'mismatch', 'revision', 'identity', 'missing', 'unknown', 'not the registered')))

def warn(site, message, work=None):
    entry={'site':site,'message':str(message),'row':os.environ.get('V4_CURRENT_ROW'), 'time_ns':time.time_ns()}
    context=_context.get()
    if context is not None:context.append(entry)
    destination=work or _context_work.get() or os.environ.get('WORK')
    if destination:
        folder=confine(destination,Path(destination)/'v4/warnings');folder.mkdir(parents=True,exist_ok=True)
        with (folder/(str(entry['time_ns'])+'-'+str(os.getpid())+'.json')).open('x') as handle:
            json.dump(entry,handle,sort_keys=True);handle.write('\n')
    print('WARNING: '+site+': '+str(message))
    return entry

def require(condition, message, site='operational check', *, identity=False):
    if condition:return True
    if not enabled() or identity or correctness(message):raise CorrectnessStop(message)
    warn(site,message);return False

@contextmanager
def observations(work=None):
    entries=[];token=_context.set(entries);place=_context_work.set(work or _context_work.get())
    try:yield entries
    finally:_context.reset(token);_context_work.reset(place)

def observe(function, *args, site=None, **kwargs):
    """Use only for an observation/check, never an operation which must execute."""
    try:return function(*args,**kwargs)
    except Exception as exc:
        if not enabled() or isinstance(exc,CorrectnessStop) or correctness(exc):raise
        warn(site or function.__name__,str(exc));return None

def warnings(work):
    return [json.loads(p.read_text()) for p in sorted((Path(work)/'v4/warnings').glob('*.json'))]

def disk(work, required_bytes, site):
    work=Path(work).resolve();free=shutil.disk_usage(work).free
    if free < required_bytes:raise CorrectnessStop(f'disk exhaustion before {site}: next write needs {required_bytes} bytes, free {free}')
    return free

def ceiling(work):
    from kit.v4_budget import allocation_spend
    path=Path(work)/'k8b4/containment/allocation-ledger.json'
    if not path.exists():warn('allocation accounting','ledger not yet available',work);return 0
    try:refresh_accounting(work)
    except Exception as exc:
        if correctness(exc):raise CorrectnessStop(str(exc)) from exc
        warn('allocation reconciliation',str(exc),work)
    try:spent,_=known_spend(json.loads(path.read_text()),work)
    except (ValueError,KeyError,TypeError) as exc:warn('allocation accounting',str(exc),work);return 0
    if spent >= CEILING:raise CorrectnessStop('actual 1000 GPU-hour ceiling reached')
    return float(spent)


def known_spend(ledger,work):
    """Use complete observed intervals; keep unknown exposure visible as a warning."""
    from kit.v4_budget import allocation_spend
    try:return allocation_spend(ledger)
    except (ValueError,KeyError,TypeError) as exc:
        warn('allocation telemetry',str(exc),work)
        from fractions import Fraction
        entries={**ledger.get('allocations',{}),**ledger.get('phase0_import',{}).get('allocations',{})}
        spent=Fraction(0);blocks={}
        for job,entry in entries.items():
            try:cost,parts=allocation_spend({'allocations':{job:entry}})
            except (ValueError,KeyError,TypeError):continue
            spent+=cost
            for key,value in parts.items():blocks[key]=blocks.get(key,Fraction(0))+value
        return spent,blocks


def runtime_seconds(work, default):
    from kit.v4_budget import stamp
    from kit import p4_contain as pc
    if not os.environ.get('SLURM_JOB_ID'):return default
    ledger=observe(pc.refresh_allocation_ledger,work,site='runtime allocation telemetry')
    if not ledger:
        file=Path(work)/'k8b4/containment/allocation-ledger.json'
        if not file.exists():return default
        ledger=json.loads(file.read_text())
    spent,_=known_spend(ledger,work)
    if spent>=CEILING:raise CorrectnessStop('actual 1000 GPU-hour ceiling reached')
    entry=ledger.get('allocations',{}).get(os.environ['SLURM_JOB_ID'],{})
    remaining=(CEILING-float(spent))*3600/(entry.get('width') or 8)
    if entry.get('planned_end'):remaining=min(remaining,float(stamp(entry['planned_end']))-time.time())
    return max(.01,min(default,remaining))


def confine(work,path):
    root=Path(work).resolve();destination=Path(path).resolve()
    if not destination.is_relative_to(root):raise CorrectnessStop('writes outside WORK: '+str(path))
    return destination


def refresh_accounting(work):
    """Close previous jobs from scheduler observations before charging a continuation."""
    from kit import p4_contain as pc
    job=os.environ.get('SLURM_JOB_ID')
    if not job:return
    path=Path(work)/'k8b4/containment/allocation-ledger.json'
    if not path.exists():return
    ledger=json.loads(path.read_text())
    for prior,entry in ledger.get('allocations',{}).items():
        if prior==job or entry.get('end'):continue
        try:obs=pc.allocation_observation(prior)
        except (ValueError,OSError,RuntimeError) as exc:
            warn('prior allocation accounting',str(exc),work);continue
        if obs.get('end'):
            entry.update(end=obs['end'],state=obs.get('state'),terminal_observation=obs)
        else:warn('prior allocation accounting','termination telemetry not yet available for '+prior,work)
    pc.wd.write_durably(path,ledger)
    pc.track_allocation(work,'scientific',CEILING,CEILING)


def writer_bytes(work, *, arm=None):
    """Known FP32 checkpoint/BF16 export footprint; capacity is per next write."""
    from kit.v4_phase0_storage import qwen3_parameters,GIB
    path=Path(os.environ.get('MODEL_DIR',''))/'config.json'
    try:parameters=qwen3_parameters(json.loads(path.read_text()))
    except (OSError,ValueError,KeyError,TypeError):
        warn('next-write capacity','model config not available for footprint estimate; retaining actual ENOSPC handling',work)
        return 0
    if arm is None:return 2*parameters+GIB
    return parameters*(16 if arm in ('S','D','D-R') else 12)+GIB


def protect_write(path):
    if enabled() and os.environ.get('WORK'):confine(os.environ['WORK'],path)
