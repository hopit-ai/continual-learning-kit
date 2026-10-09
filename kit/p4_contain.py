#!/usr/bin/env python3
"""Slurm containment for v4 rows and the parked package-4 callers.

Standard library only. `check` archives exact scheduler output, `selftest` is an
outside-step adversarial observer, `park` arms identity evidence before exec.
The registered clock/enforcement allowance J=60 is conservative, not inferred
from a successful smoke test. A real cluster receipt remains necessary.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import functools
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:sys.path.insert(0,str(HERE.parent))
from kit.v4_phase0_timing import shared_limit, is_phase0

DIRECTORY = Path('k8b4/containment')
RECEIPT = DIRECTORY / 'containment-receipt.json'
SELFTEST = DIRECTORY / 'containment-selftest.json'
ALLOWANCE = 60
START_WAIT = 600
EXPECTED = {'ProctrackType': 'proctrack/cgroup', 'TaskPlugin': 'task/cgroup,task/affinity',
            'JobAcctGatherType': 'jobacct_gather/cgroup', 'KillWait': 40, 'GraceTime': 600}
SELFTEST_SECONDS = 600
GPU_PROBE_SECONDS = 160
MIN_ROW_CAP = 300
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED'}
DESCRIPTIVE = ('Process groups, environment markers and GPU-process observations do not establish that every descendant '
               'has terminated; this run is reported descriptively.')


def _load(name):
    spec = importlib.util.spec_from_file_location('contain_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frozen = _load('p4_frozen')
wd = _load('p4_watchdog')


class Refused(Exception):
    pass


def epoch(value):
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        # Slurm's timestamps without an offset are in the scheduler/client's local zone.
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    return float(value)


def minutes(now, limit_until, kill_by, W, J) -> int:
    """Largest whole m satisfying BOTH S+60m+J<=limit_until and
    S+60m+J+W<=kill_by; `now` is the planned/actual scheduler start S.
    Round down, equality passes; <=0 REFUSES (Slurm --time=0 is unlimited).
    With kill_by=limit_until+45, W<=45 does not tighten the first bound;
    W>45 can force a smaller minute when its rounding slack is <W-45.
    This costs another whole minute or more of run time,
    in addition to J and the ordinary sub-minute rounding loss.
    W and J are nonnegative seconds; registration enforces J>=60 separately.
    """
    if W < 0 or J < 0:
        raise ValueError('W and J must be nonnegative')
    return math.floor(min(epoch(limit_until) - epoch(now) - J,
                          epoch(kill_by) - epoch(now) - J - W) / 60)


def allowance():
    try:
        value = int(os.environ.get('KIT_P4_SLURM_ALLOWANCE', ALLOWANCE))
    except ValueError:
        raise Refused('KIT_P4_SLURM_ALLOWANCE must be an integer') from None
    if value < ALLOWANCE:
        raise Refused('KIT_P4_SLURM_ALLOWANCE may only raise the registered 60 seconds')
    return shared_limit(value)


def command(argv, timeout=600):
    """Every external observation retains stdout, stderr, status and their exact sha256."""
    if timeout==600:timeout=shared_limit(timeout)
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=max(.01, timeout))
        raw, error, rc = done.stdout, done.stderr, done.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        raw, error, rc = '', str(e), -1
    return {'argv': list(argv), 'stdout': raw, 'stderr': error, 'returncode': rc,
            'stdout_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'stderr_sha256': hashlib.sha256(error.encode()).hexdigest()}


def values(text):
    return dict(re.findall(r'(\w+)\s*=\s*([^\s]+)', text))


def config_values(text):
    return {m[1]: m[2].strip() for line in text.splitlines()
            if (m := re.match(r'^\s*(\w+)\s*=\s*(.*?)\s*$', line))}


def frozen_write(path, doc):
    if is_phase0() and Path(path).name=='containment-selftest.json':
        doc={**doc,'informational':True,'warning':None if doc.get('ok') else 'WARNING: containment self-test FAILED; informational only; owner monitors the run.'}
    sealed = frozen.seal(doc)
    old = wd.read_json(path)
    if Path(path).exists():
        if old != sealed:
            raise Refused('%s differs from its first frozen receipt; it is not replaced' % path)
        return old
    if not wd.create_once(Path(path), sealed):
        if wd.read_json(path) != sealed:
            raise Refused('concurrent differing receipt: %s' % path)
    return sealed


def duration_seconds(value):
    if str(value).isdigit():
        return int(value)  # sacctmgr parsable GraceTime is seconds
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', str(value))
    if not match or int(match[3]) >= 60 or int(match[4]) >= 60:
        raise ValueError('unknown Slurm duration')
    return int(match[1] or 0)*86400 + int(match[2])*3600 + int(match[3])*60 + int(match[4])


def _cgroup_config(cfg):
    """Read cgroup.conf beside the effective slurm.conf; missing paths are unknown."""
    if cfg.get('ConstrainDevices') is not None:
        raw = '\n'.join('%s = %s' % (k, cfg[k]) for k in ('ConstrainDevices', 'SignalChildrenProcesses') if k in cfg)+'\n'
        observation = raw_observation(['scontrol', 'show', 'config', 'Cgroup Support Configuration'], raw)
        return {k.lower(): v for k, v in config_values(raw).items()}, observation
    configured = os.environ.get('SLURM_CONF') or cfg.get('SLURM_CONF')
    candidates = ([Path(configured).parent / 'cgroup.conf'] if configured else
                  [Path(p) for p in ('/etc/slurm/cgroup.conf', '/etc/slurm-llnl/cgroup.conf', '/usr/local/etc/cgroup.conf')])
    present = []
    for candidate in candidates:
        try:
            if candidate.is_file(): present.append(candidate)
        except OSError:
            pass  # denied discovery is unknown, never an inferred default
    raw, error, rc = '', '', 0
    if len(present) != 1:
        error, rc = 'need readable effective cgroup.conf; set SLURM_CONF to the node configuration path', 1
        path = candidates[0]
    else:
        path = present[0]
        try:
            raw = path.read_text()
        except OSError as e:
            error, rc = str(e), 1
    observation = {'argv': ['read-cgroup-conf', str(path)], 'stdout': raw, 'stderr': error, 'returncode': rc,
                   'stdout_sha256': hashlib.sha256(raw.encode()).hexdigest(), 'stderr_sha256': hashlib.sha256(error.encode()).hexdigest()}
    parsed = config_values('\n'.join(line.split('#', 1)[0] for line in raw.splitlines()))
    return {k.lower(): v for k, v in parsed.items()}, observation


def raw_observation(argv, raw, error='', rc=0):
    return {'argv': argv, 'stdout': raw, 'stderr': error, 'returncode': rc,
            'stdout_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'stderr_sha256': hashlib.sha256(error.encode()).hexdigest()}


def slurm_number(value, unit):
    match = re.fullmatch(r'(\d+)(?:\s+'+unit+r')?', str(value))
    return int(match[1]) if match else None


def _cluster_settings(from_file=None, qos_choice=None):
    problems, raw = [], []
    job = os.environ.get('SLURM_JOB_ID')
    if not from_file and (not job or not re.fullmatch(r'[0-9]+', job)):
        problems.append('SLURM_JOB_ID is required (a Slurm allocation)')
    def run(*args):
        if from_file:
            name = ('version.txt' if args[1:] == ('--version',) else
                    'config.txt' if args[1:] == ('show', 'config') else
                    'partition.txt' if args[1:3] == ('show', 'partition') else 'qos.txt')
            try: r = raw_observation(['read-paste', str(Path(from_file)/name)], (Path(from_file)/name).read_text())
            except OSError as error: r = raw_observation(['read-paste', name], '', str(error), 1)
        else:
            r = command(list(args))
        raw.append(r)
        if r['returncode']:
            problems.append('%s failed: %s' % (' '.join(args), r['stderr']))
        return r['stdout']
    version = run('scontrol', '--version').strip()
    cfg = config_values(run('scontrol', 'show', 'config'))
    if from_file:
        path = Path(from_file)/'cgroup.conf'
        try: cgroup_raw = raw_observation(['read-paste', str(path)], path.read_text())
        except OSError as error: cgroup_raw = raw_observation(['read-paste', str(path)], '', str(error), 1)
        cgroup = {k.lower(): v for k, v in config_values('\n'.join(line.split('#',1)[0] for line in cgroup_raw['stdout'].splitlines())).items()}
    else:
        cgroup, cgroup_raw = _cgroup_config(cfg)
    raw.append(cgroup_raw)
    if cgroup_raw['returncode']:
        problems.append(cgroup_raw['stderr'])
    allocation = values(run('scontrol', 'show', 'job', job)) if job and not from_file else {}
    partition = values(run('scontrol', 'show', 'partition', allocation.get('Partition', 'PASTED'))) if allocation.get('Partition') or from_file else {}
    qos_name = qos_choice or allocation.get('QOS')
    qos_raw = run('sacctmgr', '-n', '-P', 'show', 'qos',
                  'format=Name,PreemptMode,GraceTime,Preempt')
    if from_file and not qos_name:
        names = {r.split('|')[0] for r in qos_raw.splitlines() if r.strip()}
        qos_name = next(iter(names)) if len(names) == 1 else None
    qos_rows = [r.split('|') for r in qos_raw.splitlines() if r.split('|')[0] == qos_name]
    qos = dict(zip(('Name', 'PreemptMode', 'GraceTime', 'Preempt'), qos_rows[0][:4])) if len(qos_rows) == 1 else {}
    if not re.fullmatch(r'slurm(?:-wlm)? \d+\.\d+\.\d+(?:[^\n]*)', version):
        problems.append('Slurm version is unknown; need successful scontrol --version with a numeric installed version')
    for setting in ('ProctrackType', 'JobAcctGatherType'):
        if cfg.get(setting) != EXPECTED[setting]:
            problems.append('%s must be %s; found %s' % (setting, EXPECTED[setting], cfg.get(setting, 'unknown')))
    if set(cfg.get('TaskPlugin', '').split(',')) != set(EXPECTED['TaskPlugin'].split(',')):
        problems.append('TaskPlugin must be task/cgroup,task/affinity; need the partner settings')
    try:
        W = slurm_number(cfg['KillWait'], 'sec')
    except (KeyError, ValueError):
        W = None
    if W != EXPECTED['KillWait']:
        problems.append('KillWait must be the answered 40 seconds; found %s; need a new reviewed bound if changed' % W)
    overtimes = {'global': str(slurm_number(cfg.get('OverTimeLimit'), 'min')) if slurm_number(cfg.get('OverTimeLimit'), 'min') is not None else cfg.get('OverTimeLimit'), 'partition': partition.get('OverTimeLimit'),
                 'job': allocation.get('OverTimeLimit')}
    for scope in ('global', 'partition'):
        overtime = overtimes[scope]
        if overtime != '0' and not (scope == 'partition' and overtime == 'NONE' and overtimes['global'] == '0'):
            problems.append('%s OverTimeLimit is %s; need explicit 0 (partition NONE may inherit known global 0)' % (scope, overtime or 'unknown'))
    if overtimes['job'] not in (None, '0'):
        problems.append('job OverTimeLimit must be 0; need zero effective overtime')
    if cgroup.get('constraindevices', '').lower() in ('true', '1'):
        cgroup['constraindevices'] = 'yes'
    if cgroup.get('constraindevices', '').lower() != 'yes':
        problems.append('ConstrainDevices is %s; need explicit yes in effective cgroup.conf and a one-device CUDA probe' % cgroup.get('constraindevices', 'unknown'))
    # Record grace; CANCEL/preempt-none are bounded by the independent step timeout.
    # Suspend, gang and requeue modes can extend wall exposure and are refused.
    preemption = {'type': cfg.get('PreemptType'), 'global_mode': cfg.get('PreemptMode'),
                  'partition_mode': partition.get('PreemptMode'), 'qos_name': qos_name,
                  'qos_mode': qos.get('PreemptMode'), 'grace_time_raw': qos.get('GraceTime'),
                  'exempt_time_raw': cfg.get('PreemptExemptTime'),
                  'qos_preemptors': sorted(r.split('|')[0] for r in qos_raw.splitlines()
                      if len(r.split('|')) >= 4 and qos_name in r.split('|')[3].split(','))}
    try:
        grace = duration_seconds(qos['GraceTime'])
    except (KeyError, ValueError):
        grace = None
    preemption['grace_seconds'] = grace
    modes = [str(preemption[k]).upper() for k in ('global_mode', 'partition_mode', 'qos_mode')]
    global_mode, partition_mode, qos_mode = modes
    effective = qos_mode if qos_mode != 'CLUSTER' else (global_mode if partition_mode == 'UNSET' else partition_mode)
    preemption['effective_mode'] = effective
    allowed = {'OFF', 'CANCEL'}
    requeue_policy = 'REQUEUE' in (global_mode, partition_mode, qos_mode, effective)
    if requeue_policy and preemption['type'] == 'preempt/qos':
        allowed.add('REQUEUE')
        # CPU paste is conditional only. The first selftest and every live settings
        # recheck independently demand the actual job's immutable non-requeue flag.
        if not from_file and allocation.get('Requeue') != '0':
            problems.append('REQUEUE/preempt/qos requires live Requeue=0; submit with --no-requeue')
        if any(len(r.split('|')) < 4 for r in qos_raw.splitlines() if r.strip()):
            problems.append('REQUEUE needs all QOS tiers in format=Name,PreemptMode,GraceTime,Preempt')
        if slurm_number(cfg.get('UnkillableStepTimeout'), 'sec') != 500:
            problems.append('REQUEUE needs the reviewed UnkillableStepTimeout=500 sec')
    if (preemption['type'] not in ('preempt/none', 'preempt/qos', 'preempt/partition_prio') or
            global_mode not in allowed or partition_mode not in allowed | {'UNSET'} or
            qos_mode not in allowed | {'CLUSTER'} or effective not in allowed or not qos):
        problems.append('Preemption mode is unknown or extends exposure: %s; require OFF/CANCEL or REQUEUE/preempt/qos with --no-requeue; refuse SUSPEND and GANG' % preemption)
    if preemption['qos_preemptors']:
        problems.append('QOS %s is a preemptible tier listed by %s; use the partner own protected QOS' % (qos_name, ','.join(preemption['qos_preemptors'])))
    if not from_file and (allocation.get('NumNodes') != '1' or allocation.get('Shared') != '0' and allocation.get('OverSubscribe') != 'NO'):
        problems.append('allocation must be one node and exclusive; need NumNodes=1 and Shared=0 or OverSubscribe=NO')
    gpu = re.search(r'(?:gres/)?gpu(?:[:=])([0-9]+)', allocation.get('AllocTRES', '') or allocation.get('TresPerNode', ''))
    doc = {'schema': 'kit-p4-containment-receipt.v1', 'ok': not problems, 'problems': problems, 'job_id': job,
           'simulation': version.endswith('(stand-in)'), 'expected': EXPECTED, 'preemption': preemption,
           'JobAcctGatherType': cfg.get('JobAcctGatherType'),
           'slurm_version': version, 'ProctrackType': cfg.get('ProctrackType'), 'TaskPlugin': cfg.get('TaskPlugin'),
           'SignalChildrenProcesses': cgroup.get('signalchildrenprocesses', 'not reported'),
           'ConstrainDevices': cgroup.get('constraindevices'), 'cgroup_config_path': cgroup_raw['argv'][-1], 'KillWait': W,
           'UnkillableStepTimeout': slurm_number(cfg.get('UnkillableStepTimeout'), 'sec'),
           'Requeue': allocation.get('Requeue'), 'live_qos': allocation.get('QOS'),
           'OverTimeLimit': overtimes, 'partition': allocation.get('Partition'), 'node_list': allocation.get('NodeList'),
           'gpu_count': int(gpu[1]) if gpu else None, 'commands': raw,
           'allocation_start': scheduler_stamp(allocation.get('StartTime')),
           'allocation_end': scheduler_stamp(allocation.get('EndTime')) if allocation.get('JobState') in TERMINAL else None, 'allocation_state': allocation.get('JobState'),
           'allocation_time_limit': allocation.get('TimeLimit'),
           'allocation_cpus': allocation_fields(allocation)['cpus'],
           'allocation_memory_mb': allocation_fields(allocation)['memory_mb']}
    if not from_file and doc['gpu_count'] != 8:
        doc['problems'].append('allocation GPU count must be the answered 8; need --gres=gpu:8')
        doc['ok'] = False
    return doc


def allocation_path(work, rel, job=None):
    first = wd.read_json(Path(work)/RECEIPT)
    if not first:
        ledger=wd.read_json(Path(work)/DIRECTORY/'allocation-ledger.json') or {}
        first={'job_id':next(iter(ledger.get('allocations',{})),None)} if ledger else None
    job = str(job or os.environ.get('SLURM_JOB_ID', ''))
    if not first or str(first.get('job_id') or '') == job:
        return Path(work)/rel
    if not re.fullmatch(r'[0-9]+', job):
        raise Refused('unknown allocation identity')
    return Path(work)/DIRECTORY/'allocations'/job/Path(rel).name


def reviewed_receipt(work):
    first=wd.read_json(Path(work)/RECEIPT)
    if first: return first
    ledger=wd.read_json(Path(work)/DIRECTORY/'allocation-ledger.json') or {}
    for job in ledger.get('allocations',{}):
        receipt=wd.read_json(allocation_path(work,RECEIPT,job))
        verdict=wd.read_json(allocation_path(work,SELFTEST,job))
        if (receipt and verdict and receipt==frozen.seal(receipt) and verdict==frozen.seal(verdict) and
                receipt.get('ok') is True and verdict.get('v4_qualified') is True and
                verdict.get('receipt_sha256')==receipt.get('content_sha256')):
            return receipt
    return None


SAFETY_FIELDS = ('expected', 'simulation', 'slurm_version', 'ProctrackType', 'TaskPlugin',
                 'JobAcctGatherType', 'SignalChildrenProcesses', 'ConstrainDevices', 'KillWait',
                 'OverTimeLimit', 'partition', 'gpu_count', 'preemption', 'UnkillableStepTimeout', 'Requeue', 'live_qos')


def tracked_allocation(fn):
    """Account every live command, including refusals before safety/admission checks."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        offline = fn.__name__ == 'check' and (kwargs.get('from_file') or len(args)>2 and args[2])
        if fn.__name__ == 'start_step':
            record = args[2]
            work = record.work if hasattr(record, 'work') else record.path.parents[3]
        else:
            work = args[0] if args else kwargs['work']
        if fn.__name__=='selftest': kwargs['_overall_started']=time.time()
        if not offline:
            track_allocation(work, kwargs.get('block'), kwargs.get('block_limit'), kwargs.get('ceiling'))
        try:
            return fn(*args, **kwargs)
        except Refused as error:
            if hard_reason(str(error)):
                hard_stop(work, str(error))
            raise
        finally:
            if not offline:
                track_allocation(work, kwargs.get('block'), kwargs.get('block_limit'), kwargs.get('ceiling'))
    return wrapped


def hard_reason(reason):
    """Safety refusals, distinct from a failed but bounded computation."""
    return any(word in reason.lower() for word in (
        'unknown', 'unverified', 'not verified', 'outside time cap', 'time cap',
        'time limit exceeded', 'overlaps', 'settings changed', 'settings changed or unsafe',
        'budget block limit changed', 'budget ceiling changed', 'budget refused',
        'budget history', 'existing selftest changed', 'changed safety', 'safety settings differ',
        'budget segment cannot change'))


def hard_stop(work, reason, row_record=None):
    path=Path(work)/DIRECTORY/'v4-stop.json'
    doc=frozen.seal({'schema':'kit-v4-containment-stop.v1','stop_class':'hard','ok':False,
         'stop_id':os.urandom(16).hex(),'allocation_id':os.environ.get('SLURM_JOB_ID'),
         'stopped_at':wd.precise_text(),'failure_type':reason,'row_record':str(row_record) if row_record else None})
    wd.create_once(path,doc)
    old=wd.read_json(path)
    if old!=frozen.seal(old):raise Refused('hard stop record changed')
    return old


def stop_gate(work):
    from kit.v4_ack import gate
    try: gate(work)
    except (OSError,ValueError,ImportError) as error:
        raise Refused('containment hard stop: '+str(error)) from error
    for path in (Path(work)/DIRECTORY/'rows').glob('*.json'):
        row=wd.read_json(path) or {}
        if row.get('stop_class')=='allocation_preempted' and str(row.get('allocation_id'))==os.environ.get('SLURM_JOB_ID'):
            raise Refused('allocation preempted; end this allocation, reconcile and self-test the next allocation')


def stop_class(failure, verified, within_cap=True):
    if not verified or not within_cap or (failure and hard_reason(failure)):
        return 'hard'
    return 'allocation_preempted' if failure=='preempted' else 'row_failure' if failure else None


@tracked_allocation
def check(work, out, from_file=None, qos=None):
    """Archive settings, or a CPU-only pre-send check of exact pasted outputs."""
    doc = _cluster_settings(from_file, qos) if from_file else _cluster_settings()
    if from_file:
        doc.update(schema='kit-v4-containment-presend.v1', v4_qualified=False, from_file=str(from_file))
        return frozen_write(Path(out)/'containment-presend.json', doc)
    target = allocation_path(work, RECEIPT)
    if Path(out).resolve() == (Path(work)/DIRECTORY).resolve() and target != Path(work)/RECEIPT:
        out = target.parent
    first = frozen_write(target, doc)
    if (Path(out)/RECEIPT.name).resolve() != allocation_path(work, RECEIPT).resolve():
        frozen_write(Path(out)/RECEIPT.name, first)
    return first


def _live_cluster(receipt):
    """Compare safety settings while archiving changing Slurm runtime fields."""
    observed = _cluster_settings()
    if not observed['ok']:
        raise Refused('cluster settings changed or unsafe: '+ '; '.join(observed['problems']))
    stable = SAFETY_FIELDS + ('job_id', 'node_list')
    changed = [key for key in stable if observed.get(key) != receipt.get(key)]
    if changed:
        raise Refused('cluster settings changed since qualification: %s; need review and a new qualification allocation' % ', '.join(changed))
    return observed


def verify_receipts(work, *, selftest=True, job=None):
    problems, docs = [], {}
    for key, rel in [('receipt', RECEIPT)] + ([('selftest', SELFTEST)] if selftest else []):
        doc = wd.read_json(allocation_path(work, rel, job))
        if not frozen._self(doc, str(rel), problems):
            continue
        docs[key] = doc
        if doc.get('ok') is not True:
            problems.append('%s is not a passing receipt' % rel)
    if selftest and len(docs) == 2 and docs['selftest'].get('receipt_sha256') != docs['receipt']['content_sha256']:
        problems.append('selftest is not bound to the frozen cluster receipt')
    return problems, docs


def gpu_identity(devices=None, count=None, timeout=600):
    r = command(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'] + (['-i', devices] if devices else []), timeout)
    uuids = [line.strip() for line in r['stdout'].splitlines() if line.strip()]
    if r['returncode'] or not uuids or any(not u.startswith('GPU-') for u in uuids):
        raise Refused('physical GPU identity unavailable: %s' % r)
    if count is not None and len(uuids) != count:
        raise Refused('step must see the same %d physical GPUs; found %s' % (count, uuids))
    return uuids, r


def step_devices(env, record, count, allocation_width):
    """Select exactly the requested step width from the allocation, for every launcher."""
    if not isinstance(count, int) or not 1 <= count <= allocation_width:
        raise Refused('step GPU count exceeds its allocation or is not positive')
    allocation = env.get('SLURM_JOB_GPUS') or env.get('CUDA_VISIBLE_DEVICES')
    if not allocation:
        raise Refused('allocation GPU selectors unavailable; cannot derive the step devices')
    available = [d.strip() for d in allocation.split(',')]
    assigned = record.get('assigned_devices')
    selected = available[:count] if assigned is None else assigned
    if (not isinstance(selected, list) or len(selected) != count or
            any(not isinstance(d, str) or not d for d in selected) or len(set(selected)) != count or
            not set(selected).issubset(available)):
        raise Refused('step assignment must contain exactly the requested distinct GPUs inside the allocation')
    return ','.join(selected), available


def scheduler_devices(record):
    """Owning-step observations use archived physical UUIDs, including during recovery."""
    uuids = (record.get('slurm') or {}).get('expected_gpu_uuids')
    return ','.join(uuids) if uuids else record.get('cuda_visible_devices')


def step_values(raw):
    # Real scontrol wraps each record: NodeList, StartTime and TimeLimit may be
    # on continuation lines. Keep those lines with their owning StepId.
    return [values(part) for part in re.split(r'(?=\bStepId=)', raw) if part.startswith('StepId=')]


def steps(job, timeout=600):
    r = command(['scontrol', 'show', 'step', str(job)], timeout)
    return step_values(r['stdout']), r


def step_ref(job, step):
    if not re.fullmatch(r'[0-9]+', str(job)) or not re.fullmatch(r'[0-9a-zA-Z_]+', str(step)):
        raise Refused('invalid owning step identity')
    return '%s.%s' % (job, step)


def cancel(job, step):
    return command(['scancel', step_ref(job, step)])


def accounting_values(raw, ref, fields='JobIDRaw,State,Start,End'):
    """Parse only an explicit known field mapping; running Unknown End stays unknown."""
    names = fields.split(',') if isinstance(fields, str) else list(fields)
    identity = 'JobIDRaw' if 'JobIDRaw' in names else 'JobID'
    if not {identity, 'State', 'Start', 'End'} <= set(names) or len(set(names)) != len(names):
        return {}
    rows = [line.split('|') for line in raw.splitlines()
            if len(line.split('|')) > names.index(identity) and line.split('|')[names.index(identity)] == ref]
    if len(rows) != 1 or len(rows[0]) != len(names):
        return {}
    row = dict(zip(names, rows[0]))
    result = {'StepId': row[identity], 'State': row['State'].split()[0].rstrip('+'),
              'StartTime': row['Start'], 'EndTime': row['End']}
    for source, target in (('Timelimit', 'TimeLimit'), ('AllocTRES', 'AllocTRES'), ('ReqTRES', 'ReqTRES'), ('ExitCode', 'ExitCode')):
        if source in row: result[target] = row[source]
    if 'Elapsed' in row:
        try: result['elapsed_seconds'] = duration_seconds(row['Elapsed'])
        except ValueError: return {}
    return result


def check_live_format(from_file):
    """CPU parser rehearsal of a pasted capture; never a runtime qualification."""
    raw = Path(from_file).read_text()
    problems = []
    def section(label):
        marker = '--- '+label+' ---\n'
        if marker not in raw:
            problems.append('missing capture: '+label); return ''
        return raw.split(marker, 1)[1].split('--- ', 1)[0]
    job = values(section('scontrol show job'))
    step = values(section('scontrol show step'))
    queue = section('squeue -s')
    match = re.search(r'--- sacct[^\n]*--format=([^\s]+) ---\n', raw)
    fields = match[1] if match else ''
    accounting = {}
    if match:
        for line in raw[match.end():].splitlines():
            ref = line.split('|')[0]
            if re.fullmatch(r'\d+(?:\.[A-Za-z0-9_]+)?', ref):
                record = accounting_values(line, ref, fields)
                if record: accounting[ref] = record
                else: problems.append('sacct field shape differs: '+ref)
    else: problems.append('need sacct explicit --format mapping, never ALL')
    job_id = job.get('JobId', '')
    refs = {job_id, job_id+'.batch', job_id+'.extern', step.get('StepId')}
    if not refs <= set(accounting): problems.append('missing exact allocation/batch/extern/step accounting')
    if job.get('Requeue') != '0': problems.append('live job needs Requeue=0 (--no-requeue)')
    try:
        duration_seconds(job['TimeLimit']); duration_seconds(step['TimeLimit'])
    except (KeyError, ValueError): problems.append('unknown job or step TimeLimit')
    for ref in refs - {job_id}:
        if ref not in queue.split(): problems.append('missing squeue step: '+str(ref))
    return {'schema':'kit-slurm-live-format.v1', 'ok':not problems, 'problems':problems,
            'v4_qualified':False, 'job':job, 'step':step, 'accounting':accounting,
            'sacct_fields':fields, 'capture':raw_observation(['read-paste',str(from_file)],raw)}


def scheduler_stamp(value):
    try:
        return wd.precise_text(wd.from_epoch(epoch(value)))
    except (TypeError, ValueError):
        return None


def archived_epoch(value, utc_offset=None):
    """Replay scheduler-local text independently of the archive reader's timezone."""
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        if not utc_offset:
            raise ValueError('scheduler-local timestamp has no archived UTC offset')
        stamp = datetime.fromisoformat(value + utc_offset)
    return stamp.timestamp()


def evidence(job, step, devices, until):
    """Observe from outside the step. No missing scheduler response establishes emptiness."""
    ref, observations = step_ref(job, step), []
    last = {}
    while time.time() <= epoch(until):
        left = max(.01, epoch(until) - time.time())
        control = command(['scontrol', 'show', 'step', ref], min(5, left))
        queue = command(['squeue', '-s', '-h', '--steps=' + ref, '-o', '%i'], min(5, max(.01, epoch(until)-time.time())))
        gpu = command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'] + (['-i', devices] if devices else []),
                      min(5, max(.01, epoch(until)-time.time())))
        v = values(control['stdout'])
        source, accounting = 'scontrol', None
        if control['returncode'] != 0 or v.get('State') not in TERMINAL or not scheduler_stamp(v.get('EndTime')):
            # Finished steps can disappear from scontrol. Missing live output alone
            # is never success: require an exact, terminal accounting record.
            accounting = command(['sacct', '-n', '-P', '-j', ref, '--format=JobIDRaw,State,Start,End'],
                                 min(5, max(.01, epoch(until)-time.time())))
            archived = accounting_values(accounting['stdout'], ref)
            if accounting['returncode'] == 0 and archived.get('State') in TERMINAL:
                v, source = archived, 'sacct'
        proof = accounting if source == 'sacct' else control
        gone = proof['returncode'] == 0 and v.get('StepId') == ref and v.get('State') in TERMINAL and not v.get('TrackingError') and bool(scheduler_stamp(v.get('EndTime')))
        empty_queue = queue['returncode'] == 0 and not queue['stdout'].strip()
        idle = gpu['returncode'] == 0 and not gpu['stdout'].strip()
        last = {'step_gone': gone and empty_queue, 'gpus_idle': idle, 'state': v.get('State'),
                'scheduler_start': scheduler_stamp(v.get('StartTime')), 'scheduler_end': scheduler_stamp(v.get('EndTime')),
                'scheduler_start_raw': v.get('StartTime'), 'scheduler_end_raw': v.get('EndTime'),
                'scheduler_source': source, 'scheduler_utc_offset': time.strftime('%z'), 'sacct': accounting, 'termination_reason': v.get('State'),
                'failure_type': 'preempted' if v.get('State') == 'PREEMPTED' else None,
                'grace_seconds': int(v['GraceTimeSeconds']) if v.get('GraceTimeSeconds', '').isdigit() else None,
                'scontrol': control, 'squeue': queue, 'nvidia_smi': gpu, 'verified_at': wd.precise_text()}
        observations.append(last)
        if gone and empty_queue and idle:
            # A preempted allocation can leave its step CANCELLED or even
            # COMPLETED during grace. Inspect the owning JOB as well as the step.
            job_control = command(['scontrol', 'show', 'job', str(job)], min(5, max(.01, epoch(until)-time.time())))
            job_values = values(job_control['stdout'])
            job_state = job_values.get('JobState') if job_control['returncode'] == 0 and job_values.get('JobId') == str(job) else None
            job_accounting = None
            if not job_state:
                job_accounting = command(['sacct', '-n', '-P', '-j', str(job), '--format=JobIDRaw,State,Start,End'],
                                         min(5, max(.01, epoch(until)-time.time())))
                if job_accounting['returncode'] == 0:
                    job_state = accounting_values(job_accounting['stdout'], str(job)).get('State')
            failure = 'preempted' if 'PREEMPTED' in (job_state, v.get('State')) else None
            if not job_state and failure is None:
                failure = 'allocation state unknown'
            elif job_state not in ('RUNNING', 'COMPLETING', 'COMPLETED') and failure is None:
                failure = 'allocation '+job_state.lower()
            last.update(job_state=job_state, job_scontrol=job_control, job_sacct=job_accounting, failure_type=failure,
                        verified_at=wd.precise_text())
            observations[-1] = last
            return {**last, 'verified': True, 'observations': observations}
        time.sleep(min(shared_limit(.1), max(0, epoch(until)-time.time())))
    return {**last, 'verified': False, 'observations': observations}


@tracked_allocation
def start_step(argv, env, record, count, *, require_selftest=True, requested_cap=None):
    """One parked step. Deadlines are already durable and never moved after admission."""
    work = record.work if hasattr(record, 'work') else record.path.parents[3]
    errors, docs = verify_receipts(work, selftest=require_selftest)
    if errors:
        raise Refused('; '.join(errors))
    receipt = docs['receipt']
    job_id = env.get('SLURM_JOB_ID')
    if job_id != receipt.get('job_id'):
        raise Refused('allocation differs from the frozen cluster receipt')
    W, J = shared_limit(receipt['KillWait']), allowance()
    d = record.data
    planned_start = time.time()
    requested = minutes(planned_start, d['limit_until'], d['kill_by'], W, J)
    if requested_cap is not None:
        requested = min(requested, requested_cap)
    if requested < 1:
        raise Refused('no whole Slurm minute fits both admitted deadlines (never --time=0)')
    live_obs=allocation_observation(job_id)
    with containment_lock(work):
        ledger=wd.read_json(Path(work)/DIRECTORY/'allocation-ledger.json') or {}
        block=d.get('budget_block',ledger.get('block','qualification'))
        live={'job_id':job_id,'gpu_count':live_obs.get('width'),'allocation_start':live_obs.get('start'),
              'allocation_end':live_obs.get('end'),'allocation_time_limit':sbatch_time(live_obs.get('time_limit_seconds') or 0),
              'allocation_cpus':live_obs.get('cpus'),'allocation_memory_mb':live_obs.get('memory_mb')}
        allocation_budget(work,live,max(1,epoch(d['deadline'])-time.time()),block,
                          ledger.get('block_limits',{}).get(block,100),ledger.get('ceiling',560))
    allocation = (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES')
                  or env.get('SLURM_JOB_GPUS') or env.get('CUDA_VISIBLE_DEVICES'))
    if not allocation: raise Refused('allocation GPU selectors unavailable')
    allocation_devices=[x.strip() for x in allocation.split(',')]
    allocation_uuids,allocation_query=gpu_identity(allocation,len(allocation_devices))
    gpu_map=dict(zip(allocation_devices,allocation_uuids))
    if count>len(gpu_map): raise Refused('allocation exposes fewer GPUs than requested')
    cpus=live_obs.get('cpus'); memory=live_obs.get('memory_mb'); width=live_obs.get('width')
    if not cpus or not memory or not width or cpus<width or memory<width:
        raise Refused('allocation CPU/memory resources unknown or insufficient for shared steps')
    step_cpus=cpus*count//width; step_memory=memory*count//width
    devices=','.join(allocation_uuids)  # conservative until Slurm's assignment is archived
    expected=[]
    record.update(allocation_gpu_devices=allocation_devices)
    preceding, preceding_raw = steps(job_id)
    if preceding_raw['returncode']:
        raise Refused('cannot inventory existing allocation steps before submission')
    preceding_ids = {s.get('StepId') for s in preceding}
    folder = record.path.parent / (record.path.stem + '.slurm')
    folder.mkdir()
    ready, release = folder / 'park.json', folder / 'release'
    name = '%s-a%s' % (d['row'], d['attempt'])
    actual = ['srun', '--ntasks=1', '--exclusive', '--gres=gpu:%d' % count, '--time=%d' % requested,
              '--mem=%dM'%step_memory, '--cpus-per-task=%d'%step_cpus,
              '--kill-on-bad-exit=1', '--job-name=' + name, sys.executable, str(HERE / 'p4_contain.py'),
              'park', '--ready', str(ready), '--release', str(release), '--count', str(count),
              '--allocation-gpu-map='+json.dumps(gpu_map,separators=(',',':')), '--'] + list(argv)
    record.update(slurm={'job_id': job_id, 'step_id': None, 'receipt_sha256': receipt['content_sha256'],
                         'selftest_sha256': docs.get('selftest', {}).get('content_sha256'), 'actual_command': actual,
                         'planned_start': wd.precise_text(wd.from_epoch(planned_start)), 'requested_minutes': requested, 'effective_minutes': None, 'W': W, 'J': J,
                         'start_wait_seconds': shared_limit(START_WAIT), 'expected_gpu_uuids': expected, 'physical_gpu_query': None,
                         'allocation_gpu_query':allocation_query, 'allocation_gpu_map':gpu_map,
                         'step_cpus':step_cpus,'step_memory_mb':step_memory,
                         'preceding_steps': preceding_raw, 'assignment_required': True,
                         'limit_until': d['limit_until'], 'kill_by': d['kill_by'], 'deadline': d['deadline']})
    proc = subprocess.Popen(actual, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    record.update(pid=proc.pid, pgid=proc.pid, srun_pid=proc.pid)
    bound = min(time.monotonic() + shared_limit(START_WAIT), time.monotonic() + epoch(d['kill_by'])-time.time())
    sid, step, raw = None, None, None
    try:
        while time.monotonic() < bound:
            found, raw = steps(job_id, min(shared_limit(5), max(.01, bound-time.monotonic())))
            parked = wd.read_json(ready)
            parked_ref = (step_ref(job_id, parked['step_id']) if parked and parked.get('job_id') == job_id and parked.get('step_id') else None)
            own = [s for s in found if s.get('StepId') not in preceding_ids and
                   (s.get('Name') == name or s.get('StepId') == parked_ref)]
            if len(own) > 1:
                raise Refused('ambiguous owning step name: no command released')
            if own:
                step = own[0]; sid = step['StepId'].split('.', 1)[1]
                record.update(slurm={**record.data['slurm'], 'step_id': sid, 'start_observation': raw})
                if step.get('State') in TERMINAL:
                    raise Refused('step ended while parked')
                if step.get('State') == 'RUNNING' and parked:
                    if time.monotonic() >= bound:
                        raise Refused('step did not start within registered wait/deadlines')
                    S = epoch(step['StartTime'])
                    effective = int(step['TimeLimit']) if step.get('TimeLimit', '').isdigit() else parse_minutes(step.get('TimeLimit', ''))
                    allowed = minutes(S, d['limit_until'], d['kill_by'], W, J)
                    physical_ids = parked.get('step_gpu_ids')
                    if not physical_ids:
                        raise Refused('scheduler physical GPU assignment unavailable')
                    assigned, assignment_query = gpu_identity(physical_ids, count, min(shared_limit(5), max(.01, bound-time.monotonic())))
                    if effective < 1 or effective > requested or effective > allowed:
                        raise Refused('actual scheduler start/time limit fails the timing recheck; command remains parked')
                    if (parked.get('job_id')!=job_id or parked.get('step_id')!=sid or parked.get('problems') or
                            parked.get('gpu_uuids')!=assigned or not set(assigned).issubset(allocation_uuids) or
                            step.get('NodeList')!=parked.get('node_list')):
                        raise Refused('parked step physical GPUs/nodes differ from scheduler assignment')
                    # RPC arrival order belongs to Slurm. Archive each actual assignment
                    # under the same lock used by admission before any argv is released.
                    with containment_lock(work):
                        for path in (Path(work)/DIRECTORY/'rows').glob('*.json'):
                            if path.resolve()==record.path.resolve(): continue
                            other=wd.read_json(path) or {}
                            if (other.get('state')!='ended' and
                                    str(other.get('allocation_id',(other.get('slurm') or {}).get('job_id')))==job_id):
                                other_uuids=(other.get('slurm') or {}).get('gpu_uuids',[])
                                if set(assigned)&set(other_uuids):
                                    raise Refused('scheduler GPU assignment overlaps another live row')
                        expected=assigned; devices=','.join(assigned)
                        record.update(cuda_visible_devices=devices,assigned_devices=physical_ids.split(','),
                            pgid=parked['pgid'],job_window_start=wd.process_start_time(parked['pid']),
                            slurm={**record.data['slurm'],'scheduler_start':scheduler_stamp(step['StartTime']),
                                'scheduler_start_raw':step['StartTime'],'scheduler_utc_offset':time.strftime('%z'),
                                'effective_minutes':effective,'rechecked_minutes':allowed,'node_list':parked['node_list'],
                                'gpu_uuids':assigned,'expected_gpu_uuids':assigned,'gpu_count':len(assigned),'park':parked,
                                'physical_gpu_query':assignment_query,'assignment_gpu_query':assignment_query})
                    return proc, release
            if proc.poll() is not None:
                raise Refused('srun returned before the parked step was ready')
            time.sleep(shared_limit(.05))
        raise Refused('step did not start within registered wait/deadlines')
    except BaseException:
        # A submitted but not yet identified step must also be found and cancelled. Never cancel the allocation.
        if sid is None:
            found, raw = steps(job_id)
            own = [s for s in found if s.get('Name') == name and s.get('StepId') not in preceding_ids]
            if len(own) == 1:
                sid = own[0]['StepId'].split('.', 1)[1]
        if sid is not None:
            c = cancel(job_id, sid)
            e = evidence(job_id, sid, devices, max(time.time()+shared_limit(5), epoch(d['deadline'])))
            record.update(slurm={**record.data['slurm'], 'step_id': sid, 'cancel': c, 'termination': e})
            if e.get('verified'):
                record.update(charged_until=wd._ceil_text(wd.parse_time(e['verified_at'])), verified_idle_at=wd._ceil_text(wd.parse_time(e['verified_at'])))
            else:
                record.update(accounting='budget compliance unavailable')
                wd.write_stop(record.path, d, 'cancelled parked/pending step not verified gone', 'step cancellation unverified', unavailable=True)
        else:
            # No proof that a pending RPC vanished: stop subsequent admission rather than pretend it is gone.
            record.update(accounting='budget compliance unavailable', pending_step_unresolved=True)
            wd.write_stop(record.path, d, 'submitted step could not be identified/cancelled', 'pending step unresolved', unavailable=True)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=shared_limit(10))
        raise


def parse_minutes(value):
    # Slurm emits TimeLimit=HH:MM:SS or D-HH:MM:SS (not only integer minutes).
    days, _, clock = value.rpartition('-')
    try:
        h, m, s = map(int, (clock or value).split(':'))
        seconds = (int(days or 0)*24 + h)*3600 + m*60 + s
        if seconds % 60:
            raise ValueError()
        return seconds // 60
    except ValueError:
        raise Refused('scheduler effective TimeLimit is not a positive whole minute: %r' % value) from None


def park(ready, release, count, argv, expected_uuids=None, allocation_gpu_map=None):
    problems = []
    # Slurm may renumber CUDA_VISIBLE_DEVICES inside its device cgroup.
    # UUIDs have no local/global index ambiguity. The observer independently
    # resolves SLURM_STEP_GPUS outside the step before releasing the argv.
    if allocation_gpu_map is not None:
        try:
            ids=os.environ['SLURM_STEP_GPUS'].split(',')
            if len(ids)!=count or len(set(ids))!=count: raise KeyError('step GPU count')
            expected_uuids=','.join(allocation_gpu_map[i] for i in ids)
        except KeyError:
            problems.append('scheduler assigned GPUs outside the allocation map')
            expected_uuids='GPU-invalid-assignment'
    devices = expected_uuids or os.environ.get('CUDA_VISIBLE_DEVICES') or ','.join(str(i) for i in range(count))
    try:
        uuids, raw = gpu_identity(devices, count)
    except Refused as e:
        uuids, raw = [], None; problems.append(str(e))
    wd.write_durably(Path(ready), {'pid': os.getpid(), 'pgid': os.getpgrp(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                                 'step_id': os.environ.get('SLURM_STEP_ID'), 'node_list': os.environ.get('SLURM_STEP_NODELIST') or os.environ.get('SLURM_NODELIST'),
                                 'gpu_uuids': uuids, 'gpu_query': raw, 'step_gpu_ids': os.environ.get('SLURM_STEP_GPUS'),
                                 'command': argv, 'problems': problems})
    while not Path(release).exists():
        time.sleep(shared_limit(.02))
    if problems:
        return 125
    os.execvpe(argv[0], argv, os.environ)


def scheduler_problems(work, record):
    """Complete archived per-attempt evidence, rechecked by the reporter; never query a live cluster on archive replay."""
    errors, docs = verify_receipts(work, job=(record.get('slurm') or {}).get('job_id'))
    if record.get('containment') != 'slurm-step':
        return ['process-group-only execution']
    s = record.get('slurm') or {}
    if not isinstance(s, dict):
        return errors + ['unreadable scheduler evidence']
    required = ('job_id', 'step_id', 'receipt_sha256', 'selftest_sha256', 'actual_command', 'node_list', 'gpu_uuids', 'gpu_count',
                'requested_minutes', 'effective_minutes', 'W', 'J', 'scheduler_start', 'start_wait_seconds', 'start_observation', 'park',
                'physical_gpu_query', 'limit_until', 'kill_by', 'deadline')
    errors += ['scheduler evidence missing ' + k for k in required if s.get(k) is None]
    for key in ('receipt', 'selftest'):
        if docs.get(key) and s.get(key + '_sha256') != docs[key]['content_sha256']:
            errors.append('attempt is not bound to the frozen ' + key)
    t = s.get('termination') or {}
    if not isinstance(t, dict):
        t = {}; errors.append('unreadable scheduler termination')
    if not t.get('verified') or not t.get('step_gone') or not t.get('gpus_idle') or not isinstance(t.get('state'), str) or t.get('state') not in TERMINAL or not t.get('scheduler_end'):
        errors.append('complete scheduler termination and idle evidence absent')
    source = t.get('scheduler_source', 'scontrol')
    if source not in ('scontrol', 'sacct'):
        errors.append('unknown scheduler termination source')
        source = 'scontrol'
    for key in (source, 'squeue', 'nvidia_smi'):
        r = t.get(key) or {}
        if not valid_command(r):
            errors.append('termination command evidence missing/altered: ' + key)
    try:
        ref = step_ref(s['job_id'], s['step_id'])
        control = (accounting_values((t.get('sacct') or {}).get('stdout', ''), ref) if source == 'sacct' else
                   values((t.get('scontrol') or {}).get('stdout', '')))
        if control.get('StepId') != ref or control.get('State') != t.get('state') or control.get('TrackingError'):
            errors.append('scheduler end state does not match archived scheduler output')
        if archived_epoch(control['EndTime'], t.get('scheduler_utc_offset')) != epoch(t['scheduler_end']):
            errors.append('scheduler end timestamp does not match archived output')
        if archived_epoch(control['StartTime'], t.get('scheduler_utc_offset')) != epoch(s['scheduler_start']):
            errors.append('scheduler start timestamp does not match archived output')
        start_raw = s['start_observation']
        if not valid_command(start_raw):
            errors.append('scheduler startup observation missing/altered')
        started = [v for v in step_values(start_raw['stdout']) if v.get('StepId') == ref]
        if len(started) != 1:
            errors.append('no unique archived owning-step startup')
        else:
            start = started[0]
            m = int(start['TimeLimit']) if start['TimeLimit'].isdigit() else parse_minutes(start['TimeLimit'])
            if m != s['effective_minutes'] or archived_epoch(start['StartTime'], s.get('scheduler_utc_offset')) != epoch(s['scheduler_start']) or start['NodeList'] != s['node_list']:
                errors.append('archived scheduler start/grant/nodes differ from attempt')
        park = s['park']
        query = park.get('gpu_query') or {}
        outside = s['physical_gpu_query']
        if not valid_command(query) or not valid_command(outside):
            errors.append('inside/outside physical GPU query missing/altered')
        if ([u.strip() for u in query.get('stdout', '').splitlines() if u.strip()] != s['gpu_uuids'] or
                [u.strip() for u in outside.get('stdout', '').splitlines() if u.strip()] != s['expected_gpu_uuids'] or
                park.get('gpu_uuids') != s['gpu_uuids'] or park.get('job_id') != s['job_id'] or park.get('step_id') != s['step_id'] or
                park.get('node_list') != s['node_list'] or park.get('problems')):
            errors.append('parked step GPU/node identity differs from archived queries')
        if s.get('assignment_required') or record.get('allocation_width') is not None:
            assignment = s.get('assignment_gpu_query') or {}
            expected_flag = '--expected-uuids='+','.join(s['expected_gpu_uuids'])
            if s.get('allocation_gpu_map'):
                expected_flag='--allocation-gpu-map='+json.dumps(s['allocation_gpu_map'],separators=(',',':'))
                allocation_query=s.get('allocation_gpu_query') or {}
                if (not valid_command(allocation_query) or
                        allocation_query.get('stdout','').splitlines()!=list(s['allocation_gpu_map'].values()) or
                        not set(s['gpu_uuids']).issubset(s['allocation_gpu_map'].values())):
                    errors.append('allocation GPU map missing/altered')
                if '--mem=%dM'%s.get('step_memory_mb',0) not in s['actual_command'] or '--cpus-per-task=%d'%s.get('step_cpus',0) not in s['actual_command']:
                    errors.append('shared step CPU/memory flags missing or altered')
            if (not valid_command(assignment) or not park.get('step_gpu_ids') or
                    assignment.get('argv', [])[-2:] != ['-i', park['step_gpu_ids']] or
                    [u.strip() for u in assignment.get('stdout', '').splitlines() if u.strip()] != s['expected_gpu_uuids'] or
                    expected_flag not in s['actual_command']):
                errors.append('physical step GPU assignment missing or differs from expected UUIDs')
        actual = s['actual_command']
        flags = ('--ntasks=1', '--exclusive', '--gres=gpu:%d' % s['gpu_count'], '--time=%d' % s['requested_minutes'],
                 '--kill-on-bad-exit=1', '--job-name=%s-a%d' % (record['row'], record['attempt']))
        if not isinstance(actual, list) or not actual or actual[0] != 'srun' or any(flag not in actual for flag in flags):
            errors.append('actual srun command differs from registered owning-step flags')
        if not park.get('command') or actual[-len(park['command']):] != park['command']:
            errors.append('actual step command differs from parked command')
        if record.get('kind') != 'selftest' and park.get('command') != record.get('command'):
            errors.append('parked command differs from row command')
        if (t.get('squeue') or {}).get('stdout', '').strip() or (t.get('nvidia_smi') or {}).get('stdout', '').strip():
            errors.append('archived step queue/GPU observation is not empty')
        if s['J'] < ALLOWANCE or s['W'] != docs['receipt']['KillWait'] or s['effective_minutes'] < 1 or s['effective_minutes'] > s['requested_minutes']:
            errors.append('invalid scheduler timing evidence')
        if s['effective_minutes'] > minutes(s['scheduler_start'], s['limit_until'], s['kill_by'], s['W'], s['J']):
            errors.append('scheduler timing inequalities do not hold')
        if any(s[k] != record[k] for k in ('limit_until', 'kill_by', 'deadline')):
            errors.append('attempt deadlines differ from scheduler evidence')
        if len(s['gpu_uuids']) != s['gpu_count'] or s['gpu_uuids'] != s.get('expected_gpu_uuids'):
            errors.append('physical GPU identity/count changed')
        if not s['node_list'] or not s['actual_command'] or (isinstance(s['start_wait_seconds'],bool) or not isinstance(s['start_wait_seconds'],(int,float)) or not math.isfinite(s['start_wait_seconds']) or s['start_wait_seconds']<=0) or s['gpu_count'] < 1:
            errors.append('invalid scheduler command/node/start-wait/count evidence')
        if s['job_id'] != docs['receipt']['job_id'] or ((record.get('gate') or {}).get('gpus') not in (None, s['gpu_count'])):
            errors.append('scheduler allocation/GPU count differs from admission')
        baseline = wd.read_json(Path(work) / frozen.RECIPE_BASELINE) or {}
        registered_count = (((baseline.get('rows') or {}).get(record.get('row')) or {}).get('knobs') or {}).get('NGPU')
        if registered_count is not None and int(registered_count) != s['gpu_count']:
            errors.append('scheduler GPU count differs from the frozen trainer recipe')
        if not epoch(s['scheduler_start']) <= epoch(t['scheduler_end']) <= epoch(s['kill_by']):
            errors.append('scheduler termination outside admitted start/kill_by window')
        if epoch(t['scheduler_end']) > epoch(t['verified_at']):
            errors.append('scheduler end occurs after termination verification')
        if epoch(record['charged_until']) < epoch(t['verified_at']):
            errors.append('attempt is not charged through verified termination')
        if epoch(t['verified_at']) > epoch(s['deadline']):
            errors.append('termination verified past deadline')
    except (KeyError, TypeError, ValueError, AttributeError, Refused):
        errors.append('unreadable scheduler arithmetic/identity')
    return errors


def valid_command(raw):
    return (isinstance(raw, dict) and raw.get('returncode') == 0 and isinstance(raw.get('argv'), list) and
            bool(raw['argv']) and all(isinstance(a, str) for a in raw['argv']) and
            all(isinstance(raw.get(k), str) and hashlib.sha256(raw[k].encode()).hexdigest() == raw.get(k+'_sha256')
                for k in ('stdout', 'stderr')))


# The payload deliberately retains no environment dependency after exec. It writes
# each pid before detaching. Device restriction is exercised while clients live;
# the same detached holder re-touches CUDA after client loss when the step survives.
PAYLOAD = r'''
import json, os, signal, subprocess, sys, time
from pathlib import Path
folder, delay, gpu_seconds = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
folder.mkdir(parents=True, exist_ok=True)
exe = sys.executable
from kit.v4_phase0_timing import shared_limit
# Freeze timings into the detached program before its environment is cleared.
late_code_multiplier = shared_limit(1)
owners = os.environ.get('SIM_GPU_OWNERS')
sim = os.environ.get('SIM_SLURM_STATE')
stand_in = len(sys.argv) > 4 and sys.argv[4] == '--stand-in'
late_code = r"""
import json, os, sys, time
from pathlib import Path
folder, delay, duration, owners, simulated = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5] == 'yes'
sim_state = sys.argv[6]
try:
    if simulated:
        if not owners: raise RuntimeError('stand-in GPU acquisition disabled')
        cfg_path = Path(sim_state) / 'config.json'
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        time.sleep(float(cfg.get('GPUImportDelay', 0)))
        accessible = 1 if cfg.get('ConstrainDevices', True) in (True, 'yes') else 8
        if accessible != 1: raise RuntimeError('one-GPU step exposes %d CUDA devices; need effective ConstrainDevices=yes' % accessible)
        Path(owners).mkdir(parents=True, exist_ok=True)
        Path(owners, 'late-gpu%s-%d' % (sys.argv[7] if len(sys.argv)>7 else '0', os.getpid())).write_text(str(os.getpid()))
        result = {'ok': True, 'exercised': 'stand-in GPU (no CUDA here)', 'accessible_cuda_devices': accessible}
    else:
        import torch
        # Query CUDA runtime, not NVML enumeration, with the environment cleared.
        # This proves device restriction as well as process tracking: CVD alone
        # cannot prevent a one-GPU row using all eight allocated GPUs.
        accessible = torch._C._cuda_getDeviceCount()
        if accessible != 1: raise RuntimeError('one-GPU step exposes %d CUDA devices; need effective ConstrainDevices=yes' % accessible)
        x = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()
        result = {'ok': True, 'exercised': 'torch CUDA allocation', 'accessible_cuda_devices': accessible}
except Exception as e:
    result = {'ok': False, 'why': 'gpu acquisition not exercised: %s' % e}
# This receipt proves device restriction independently of cancellation latency.
path=folder/'gpu.json'
tmp=folder/'gpu.json.tmp';tmp.write_text(json.dumps(result));os.replace(tmp,path)
while not (folder/'clients-lost.json').exists(): time.sleep(.02 * float(sys.argv[8]) if len(sys.argv)>8 else .02)
time.sleep(delay)
try:
    if not result['ok']:raise RuntimeError(result['why'])
    if not simulated:
        x.add_(1);torch.cuda.synchronize()
    else:
        Path(owners, 'late-gpu%s-%d' % (sys.argv[7], os.getpid())).write_text(str(os.getpid()))
    late={'ok':True,'exercised':'CUDA re-touch after client loss' if not simulated else 'stand-in re-touch'}
except Exception as e:late={'ok':False,'why':str(e)}
(folder/'gpu-late.json').write_text(json.dumps(late))
time.sleep(duration)
while True: time.sleep(.1 * (float(sys.argv[8]) if len(sys.argv)>8 else 1))
"""
def note(kind):
    (folder / (kind + '.pid')).write_text(str(os.getpid()))
def sleep():
    while True: time.sleep(shared_limit(.1))
for kind in ['double-fork-setsid', 'cleared-exec', 'ignores-term', 'orphan-grandchild', 'later-gpu']:
    pid = os.fork()
    if pid: continue
    note(kind)
    # Let the stand-in sample the tree. This is not required by a real cgroup.
    time.sleep(shared_limit(.15))
    if kind in ('double-fork-setsid', 'orphan-grandchild'):
        if kind == 'double-fork-setsid': os.setsid()
        note(kind + '-parent')
        if os.fork():
            time.sleep(shared_limit(.15))
            os._exit(0)
        note(kind)
    if kind == 'ignores-term': signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if kind == 'cleared-exec':
        os.environ.clear()
        os.execve(exe, [exe, '-c', 'import time; time.sleep(%s)' % shared_limit(600)], {})
    if kind == 'later-gpu':
        # This is the reviewer's exact escape: detach, exec with NO marker (or
        # any environment at all), then acquire a GPU only after client death.
        os.setsid()
        os.execve(exe, [exe, '-c', late_code, str(folder), str(delay), str(gpu_seconds), owners or '', 'yes' if stand_in and sim else 'no', sim or '', os.environ.get('SIM_VISIBLE_GPU_IDS', '0').split(',')[0], str(late_code_multiplier)], {k:v for k,v in os.environ.items() if k in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'DYLD_LIBRARY_PATH', 'PATH')})
    sleep()
note('launcher')
sleep()
'''


CPU_PAYLOAD = r'''
import os, signal, sys, time
from pathlib import Path
folder = Path(sys.argv[1])
factor = float(sys.argv[2]) if len(sys.argv)>2 else 1
if os.fork() == 0:
    os.setsid()
    time.sleep(.15*factor)
    if os.fork():
        time.sleep(.15*factor)
        os._exit(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (folder / 'grandchild.pid').write_text(str(os.getpid()))
    while True: time.sleep(.1*factor)
time.sleep(5*factor)
'''


def cpu_probe(work, out, until):
    """CPU-only normal-exit probe: a detached grandchild must die with its step."""
    folder = Path(out) / 'cpu-probe'
    folder.mkdir(parents=True, exist_ok=True)
    job = os.environ['SLURM_JOB_ID']
    name = 'contain-cpu-%d' % os.getpid()
    argv = ['srun', '--ntasks=1', '--exclusive', '--gres=gpu:0', '--time='+str(shared_limit(1)),
            '--kill-on-bad-exit=1', '--job-name='+name, sys.executable, '-c', CPU_PAYLOAD, str(folder), str(shared_limit(1))]
    proc, sid = None, None
    doc = {'ok': False, 'problems': [], 'argv': argv, 'time_limit_minutes': shared_limit(1), 'gpus': 0}
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        while time.time() < until:
            found, raw = steps(job, min(shared_limit(5), max(.01, until-time.time())))
            own = [s for s in found if s.get('Name') == name]
            if len(own) == 1:
                sid = own[0]['StepId'].split('.', 1)[1]
                doc.update(step_id=sid, observation=raw)
            if proc.poll() is not None:
                break
            time.sleep(shared_limit(.05))
        if sid is None:
            raise Refused('CPU detached-grandchild probe: owning step identity unavailable; need scheduler visibility')
        pidfile = folder / 'grandchild.pid'
        pid = int(pidfile.read_text()) if pidfile.exists() else None
        doc['grandchild_pid'] = pid
        termination = evidence(job, sid, '0', until)
        doc['termination'] = termination
        if not termination.get('verified') or termination.get('state') != 'COMPLETED' or termination.get('failure_type'):
            doc['problems'].append('CPU step did not complete with verified empty containment; need normal-exit cgroup cleanup')
        if pid is None or pid_alive(pid):
            doc['problems'].append('detached grandchild survived the CPU step or was unobserved; need proctrack/cgroup to kill detached descendants at step end')
        doc['ok'] = not doc['problems']
    except (OSError, ValueError, Refused, subprocess.SubprocessError) as error:
        doc['problems'].append('CPU detached-grandchild probe failed: %s' % error)
    finally:
        if sid:
            doc['cleanup'] = cancel(job, sid)
        if proc and proc.poll() is None:
            proc.kill(); proc.wait(timeout=shared_limit(5))
        pidfile = folder / 'grandchild.pid'
        if pidfile.exists():
            try: os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except (OSError, ValueError): pass
    return doc


class RowRecord:
    """Minimal durable record adapter shared with the existing parked-step launcher."""
    def __init__(self, work, path, data):
        self.work, self.path, self.data = Path(work), Path(path), data
        if not wd.create_once(self.path, data):
            raise Refused('row output already exists; choose a new attempt output, never overwrite evidence')

    def update(self, **fields):
        self.data.update(fields)
        wd.write_durably(self.path, self.data)


@contextlib.contextmanager
def containment_lock(work):
    folder = Path(work)/DIRECTORY
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'admission.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def probe_bounds(start, J, W):
    """One minute plus periodic-controller allowance and KillWait, bounded at 160 s."""
    if shared_limit(60)+J+W > shared_limit(GPU_PROBE_SECONDS):
        raise Refused('GPU probe bound exceeds the registered 160 seconds')
    return start+shared_limit(60)+J, start+shared_limit(60)+J+W, start+shared_limit(60)+J+W


def cpu_preflight():
    """Free parse dry run and cleared-environment torch import before freezing evidence."""
    settings = _cluster_settings()
    if not settings['ok']:
        return {'ok':True, 'problems':[], 'settings_dry_run':settings, 'torch_import':{'not_attempted':'settings check failed; freeze configuration refusal before probes'}}
    env = {k:v for k,v in os.environ.items() if k in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'DYLD_LIBRARY_PATH', 'PATH')}
    argv = [sys.executable, '-c', 'import torch; print(torch.__version__)']
    if settings.get('simulation'):
        argv = [sys.executable, '-c', 'print("stand-in: no real torch/CUDA qualification")']
    try:
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=shared_limit(600))
        imported = raw_observation(argv, result.stdout, result.stderr, result.returncode)
    except (OSError, subprocess.SubprocessError) as error:
        imported = raw_observation(argv, '', str(error), -1)
    # Configuration failures are frozen by selftest; import failures are fixable
    # preflight failures and consume no experiment or GPU allowance.
    return {'ok': imported['returncode'] == 0, 'problems': [] if imported['returncode'] == 0 else
            ['cleared-environment torch import failed: '+imported['stderr']], 'settings_dry_run': settings,
            'torch_import': imported}


def memory_mb(value):
    m = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)', str(value))
    if not m: return None
    return int(float(m[1])*{'':1,'K':1/1024,'M':1,'G':1024,'T':1024**2}[m[2]])


def allocation_fields(v):
    tres = dict(part.split('=',1) for part in v.get('AllocTRES','').split(',') if '=' in part)
    gpu = re.search(r'(?:gres/)?gpu(?:[:=])(\d+)', v.get('AllocTRES','') or v.get('TresPerNode',''))
    try: duration = duration_seconds(v['TimeLimit'])
    except (KeyError,ValueError): duration = None
    return {'width':int(gpu[1]) if gpu else None, 'time_limit_seconds':duration,
            'cpus':int(v.get('NumCPUs',tres.get('cpu','0'))) or None,
            'memory_mb':memory_mb(tres.get('mem',v.get('MinMemoryNode')))}


def allocation_observation(job, cluster=None):
    cluster_args=['-M',cluster] if cluster else []
    control = command(['scontrol',*cluster_args,'show','job',str(job)])
    v = values(control['stdout']) if control['returncode']==0 else {}
    if v.get('JobId') == str(job):
        fields = allocation_fields(v)
        return {**fields, 'job_id':str(job), 'start':scheduler_stamp(v.get('StartTime')),
                'end':scheduler_stamp(v.get('EndTime')) if v.get('JobState') in TERMINAL else None,
                'state':v.get('JobState'), 'scheduler_utc_offset':time.strftime('%z'), 'observation':control}
    accounting = command(['sacct',*cluster_args,'-n','-P','-j',str(job),'--format=JobIDRaw,State,Start,End,AllocTRES'])
    v = accounting_values(accounting['stdout'],str(job),fields='JobIDRaw,State,Start,End,AllocTRES') if accounting['returncode']==0 else {}
    rows = [line.split('|') for line in accounting['stdout'].splitlines() if line.split('|')[0]==str(job)]
    if len(rows)==1 and len(rows[0])>=5: v['AllocTRES']=rows[0][4]
    return {**allocation_fields(v),'job_id':str(job), 'start':scheduler_stamp(v.get('StartTime')),
            'end':scheduler_stamp(v.get('EndTime')) if v.get('State') in TERMINAL else None, 'state':v.get('State'), 'scheduler_utc_offset':time.strftime('%z'), 'observation':accounting}


def track_allocation(work, block=None, block_limit=None, ceiling=None):
    job = os.environ.get('SLURM_JOB_ID')
    if not job: return None
    observation = allocation_observation(job)
    with containment_lock(work):
        return record_allocation(work, observation, block, block_limit, ceiling)


def record_allocation(work, observation, block=None, block_limit=None, ceiling=None):
    path = Path(work)/DIRECTORY/'allocation-ledger.json'
    ledger = wd.read_json(path) or {'schema':'kit-v4-allocation-ledger.v2','allocations':{}}
    job = str(observation['job_id']); block = block or ledger.get('block','qualification')
    ledger.setdefault('block_limits',{}).setdefault(block,block_limit if block_limit is not None else 100)
    ledger.setdefault('ceiling',ceiling if ceiling is not None else 560)
    entry = ledger['allocations'].setdefault(job,{'start':None,'end':None,'width':None,'rows':{},'segments':[]})
    start,width = observation.get('start'),observation.get('width')
    if start and width:
        if entry['start'] and (epoch(entry['start'])!=epoch(start) or entry['width']!=width):
            entry['accounting_unknown']='allocation start/width changed'
        else:
            entry.update(start=start,width=width)
            if entry.get('accounting_unknown')=='allocation start/width unknown': entry.pop('accounting_unknown')
    if not entry['segments'] and entry.get('start'):
        entry['segments']=[{'block':block,'start':entry['start'],'end':None}]
    entry.update(end=observation.get('end') or entry.get('end'), state=observation.get('state'),
                 observation=observation, cpus=observation.get('cpus'), memory_mb=observation.get('memory_mb'))
    duration=observation.get('time_limit_seconds')
    if duration and entry.get('start'):
        entry['planned_end']=wd.precise_text(wd.from_epoch(epoch(entry['start'])+duration))
        entry['time_limit_seconds']=duration
    elif not observation.get('end'):
        entry.pop('planned_end',None); entry.pop('time_limit_seconds',None)
    if not entry.get('start') or not entry.get('width'): entry['accounting_unknown']='allocation start/width unknown'
    wd.write_durably(path,ledger)
    return refresh_allocation_ledger(work)


def sbatch_time(seconds):
    minutes=max(0,math.floor(seconds/60)); days,minutes=divmod(minutes,1440)
    h,m=divmod(minutes,60)
    return ('%d-'%days if days else '')+'%02d:%02d:00'%(h,m)


def allocation_budget(work, live, cap, block, block_limit, ceiling):
    try:return _allocation_budget(work,live,cap,block,block_limit,ceiling)
    except Refused as error:
        if hard_reason(str(error)):hard_stop(work,str(error))
        raise


def _allocation_budget(work, live, cap, block, block_limit, ceiling):
    """Charge phase 0 by elapsed time; other campaigns reserve through planned end."""
    if any(not math.isfinite(x) or x<=0 for x in (cap,block_limit,ceiling)):
        raise Refused('budget inputs must be positive finite values')
    path=Path(work)/DIRECTORY/'allocation-ledger.json'
    old=wd.read_json(path) or {}
    limits=old.get('block_limits',{})
    if not is_phase0() and block in limits and limits[block]!=block_limit: raise Refused('budget block limit changed from its campaign input')
    if not is_phase0() and old.get('ceiling') is not None and old['ceiling']!=ceiling: raise Refused('budget ceiling changed from its campaign input')
    try: duration=duration_seconds(live['allocation_time_limit'])
    except (KeyError,ValueError): duration=None
    observation={'job_id':str(live['job_id']),'start':live.get('allocation_start'),'end':live.get('allocation_end'),
        'width':live.get('gpu_count'),'state':live.get('allocation_state','RUNNING'),
        'time_limit_seconds':duration,'cpus':live.get('allocation_cpus'),'memory_mb':live.get('allocation_memory_mb')}
    ledger=record_allocation(work,observation,block,block_limit,ceiling)
    job=str(live['job_id']); entry=ledger['allocations'][job]
    if not live.get('allocation_start') or not live.get('gpu_count'):
        raise Refused('allocation start/width unknown; unknown exposure is never zero')
    # Every refused allocation is already present and charged before these checks.
    for other,prior in ledger['allocations'].items():
        if other!=job and not prior.get('end'):
            obs=allocation_observation(other)
            prior.update(end=obs.get('end'),terminal_observation=obs)
            wd.write_durably(path,ledger)
            if not prior.get('end') or obs.get('state') not in TERMINAL:
                raise Refused('previous allocation end unknown; reconcile allocation before budget admission')
    ledger=refresh_allocation_ledger(work); entry=ledger['allocations'][job]
    phase0 = False
    plan_path = os.environ.get('V4_ALLOCATION_PLAN')
    if plan_path:
        plan_bytes = Path(plan_path).read_bytes()
        if hashlib.sha256(plan_bytes).hexdigest() != os.environ.get('V4_ALLOCATION_PLAN_SHA256'):
            raise Refused('changed safety settings: allocation plan hash differs')
        plan_doc = json.loads(plan_bytes)
        if plan_doc.get('phase') == 'phase0':
            phase0 = True
            from kit.v4_phase0 import allocation_cap, ALLOCATION_CAP_GPU_HOURS
            entry['phase0'] = True
            wd.write_durably(path, ledger)  # refused allocations and their tails stay charged
            if plan_doc.get('phase0_allocation_cap_gpu_hours') != ALLOCATION_CAP_GPU_HOURS:
                raise Refused('changed safety settings: phase0 allocation cap differs')
            try: allocation_cap(ledger)
            except ValueError as error:
                reason = 'allocation budget refused: '+str(error)
                frozen_write(allocation_path(work,DIRECTORY/'allocation-admission.json',job), {
                    'schema':'kit-v4-allocation-admission.v1','job_id':job,'ok':False,
                    'reason':reason,'phase0_allocation_cap_gpu_hours':ALLOCATION_CAP_GPU_HOURS,
                    'requested_sbatch_time':sbatch_time(duration or 0),
                    'computed_at':wd.precise_text(),'allowed_sbatch_time':None})
                raise Refused(reason) from error
    if ledger.get('accounting_unknown'): raise Refused('allocation start/width unknown; unknown exposure is never zero')
    if not duration: raise Refused('allocation TimeLimit is unknown or unlimited; cannot reserve its planned end')
    now=time.time(); start=epoch(entry['start']); planned=start+duration
    if start>now or planned<=now or entry.get('end'): raise Refused('allocation is not running within its planned TimeLimit')
    if entry['segments'][-1]['block']!=block:
        raise Refused('budget segment cannot change inside an allocation; end qualification before the pause')
    refusal_path=allocation_path(work,DIRECTORY/'allocation-admission.json',job)
    prior_verdict=wd.read_json(refusal_path)
    if prior_verdict and not prior_verdict.get('ok'):
        raise Refused(prior_verdict['reason'])
    blocks=ledger['blocks']; spent=ledger['spent_gpu_hours']; reserved=(planned-now)*entry['width']/3600
    # Exclude this allocation's remaining reservation from the other-allocations sum.
    other_reserved=sum(a.get('reserved_gpu_hours',0) for j,a in ledger['allocations'].items() if j!=job)
    remaining=min(block_limit-blocks.get(block,0),ceiling-spent-other_reserved)
    permitted=max(0,remaining)*3600/entry['width']
    allowed=sbatch_time(permitted) if permitted>=60 else None
    if not phase0 and any(name not in ledger['block_limits'] for name in blocks): raise Refused('budget history has unknown block limits')
    historical=any(charge>ledger['block_limits'][name] for name,charge in blocks.items())
    # Phase 0 has one owner-approved 192-GPU-hour reservation; the block charges
    # actual elapsed allocation time, including idle time and failed setup.
    if not phase0 and (historical or reserved>remaining+1e-9):
        reason=('allocation budget refused: request sbatch --time=%s or shorter in a new allocation (block %s, ceiling %s)'%(allowed,block_limit,ceiling) if allowed else 'allocation budget refused: no positive sbatch --time is available; block or ceiling exhausted')
        if phase0:reason='allocation budget refused: actual phase0 spend exceeds block or ceiling; stop and return evidence; no replacement job'
        doc=frozen_write(refusal_path,{'schema':'kit-v4-allocation-admission.v1','job_id':job,'ok':False,
            'reason':reason,'allowed_sbatch_time':allowed,'allowed_seconds':math.floor(permitted/60)*60,
            'requested_sbatch_time':sbatch_time(duration),'planned_end':entry['planned_end'],
            'spent_gpu_hours':spent,'remaining_gpu_hours':max(0,remaining),'computed_at':wd.precise_text(wd.from_epoch(now)),
            'reserved_gpu_hours':reserved,'block':block,'block_limit':block_limit,'ceiling':ceiling})
        ledger.update(admission_refusal=doc,block=block,block_limit=block_limit)
        wd.write_durably(path,ledger)
        raise Refused(reason)
    if not phase0 and now+cap>planned: raise Refused('row/self-test cap does not fit remaining allocation TimeLimit; request a longer safe sbatch --time')
    entry['reservation_until']=entry['planned_end']
    ledger.update(block=block,block_limit=block_limit,admitted_at=wd.precise_text(wd.from_epoch(now)),
                  reserved_gpu_hours=reserved+other_reserved,block_spent_gpu_hours=blocks.get(block,0),admission_refusal=None)
    wd.write_durably(path,ledger)
    return ledger


def refresh_allocation_ledger(work,row=None):
    path=Path(work)/DIRECTORY/'allocation-ledger.json'; ledger=wd.read_json(path)
    if not ledger: return None
    now=time.time(); spent=0.; reserved=0.; blocks={}; unknown=False; reservation_unknown=False
    for job,entry in ledger['allocations'].items():
        entry['rows']={}
        for row_path in (Path(work)/DIRECTORY/'rows').glob('*.json'):
            doc=wd.read_json(row_path) or {}
            if str(doc.get('allocation_id',(doc.get('slurm') or {}).get('job_id')))==job:
                entry['rows'][str(row_path)]={k:doc.get(k) for k in ('started_at','deadline','charged_until','state','gpus','assigned_devices','ledger_charge_gpu_hours')}
        if entry.get('accounting_unknown') or not entry.get('start') or not entry.get('width'):
            entry.update(charged_gpu_hours=None,reserved_gpu_hours=None); unknown=True; continue
        end=epoch(entry['end']) if entry.get('end') else now
        if end<epoch(entry['start']): entry['accounting_unknown']='invalid allocation end'; unknown=True; continue
        entry['charged_gpu_hours']=(end-epoch(entry['start']))*entry['width']/3600
        entry['reservation_until']=entry.get('end') or entry.get('planned_end')
        entry['reserved_gpu_hours']=(0. if entry.get('end') else max(0,epoch(entry['planned_end'])-now)*entry['width']/3600 if entry.get('planned_end') else None)
        spent+=entry['charged_gpu_hours']
        if entry['reserved_gpu_hours'] is None: reservation_unknown=True
        else: reserved+=entry['reserved_gpu_hours']
        for seg in entry['segments']:
            finish=min(end,epoch(seg['end'])) if seg.get('end') else end
            blocks[seg['block']]=blocks.get(seg['block'],0)+max(0,finish-epoch(seg['start']))*entry['width']/3600
        entry['observed_until']=wd.precise_text(wd.from_epoch(now))
    ledger.update(spent_gpu_hours=None if unknown else spent,reserved_gpu_hours=None if unknown or reservation_unknown else reserved,
                  blocks=blocks,accounting_unknown=unknown,reservation_unknown=reservation_unknown,block_spent_gpu_hours=blocks.get(ledger.get('block'),0))
    wd.write_durably(path,ledger)
    return ledger


@tracked_allocation
def reconcile(work, out, seconds=600, *, force=False):
    """Cancel and verify each abandoned owning step; preserve failures and never restart argv."""
    if seconds==600:seconds=shared_limit(seconds)
    if not math.isfinite(seconds) or not seconds>0:
        raise Refused('reconcile observation window must be finite positive seconds')
    work = Path(work); rows = []; until = time.time()+seconds
    with containment_lock(work):
        ledger_path=work/DIRECTORY/'allocation-ledger.json'
        ledger=wd.read_json(ledger_path)
        from kit.v4_phase0_submission import receipt_path, reconcile_submission
        if receipt_path(work).is_file():ledger=reconcile_submission(work)
        if not ledger or not ledger.get('allocations'):
            raise Refused('missing allocation ledger; verify WORK before reconciliation')
        for path in (work/DIRECTORY/'rows').glob('*.json'):
            row = wd.read_json(path) or {}
            if row.get('state') == 'ended': continue
            if not force and row.get('client_pid') and pid_alive(row['client_pid']):
                raise Refused('row client is alive; use --force to cancel it explicitly')
            s = row.get('slurm') or {}
            row.update(ok=0, status=2, charged_until=None)
            if not s.get('step_id'):
                row.update(state='unverified', failure_type='pending step unresolved')
            else:
                cancellation = cancel(s['job_id'], s['step_id'])
                terminated = evidence(s['job_id'], s['step_id'], ','.join(s.get('expected_gpu_uuids', [])), until)
                verified = bool(terminated.get('verified'))
                row.update(state='ended' if verified else 'unverified', ok=0, status=2,
                    failure_type=terminated.get('failure_type') or 'client interrupted',
                    charged_until=terminated.get('verified_at') if verified else None,
                    slurm={**s, 'cancel':cancellation, 'termination':terminated})
            if row.get('state')=='ended' and s.get('step_id'):
                # Recovery preserves a discovered cap breach as the typed cause,
                # rather than replacing it with a generic client interruption.
                if terminated.get('state')=='TIMEOUT':row['failure_type']='time cap'
                if (terminated.get('scheduler_end') and row.get('deadline') and
                        epoch(terminated['scheduler_end'])>epoch(row['deadline'])):
                    row['failure_type']='termination outside time cap'
            width = row.get('allocation_width', 8)
            row['ledger_charge_gpu_hours'] = (max(0, epoch(row['charged_until'])-epoch(row['started_at']))*width/3600
                if row.get('charged_until') else width*row.get('time_cap_seconds', 300)/3600)
            wd.write_durably(path, row); rows.append(row)
            cls=stop_class(row['failure_type'],row.get('state')=='ended',
                           bool(row.get('charged_until') and row.get('deadline') and epoch(row['charged_until'])<=epoch(row['deadline'])))
            row['stop_class']=cls; wd.write_durably(path,row)
            if cls=='hard':hard_stop(work,row['failure_type'],path)
        ledger_path = work/DIRECTORY/'allocation-ledger.json'
        ledger = wd.read_json(ledger_path)
        if ledger:
            for job, allocation in ledger['allocations'].items():
                observation = (allocation_observation(job,cluster=allocation['cluster']) if allocation.get('cluster') else allocation_observation(job))
                allocation.update(end=observation['end'] or allocation.get('end'), terminal_observation=observation)
            wd.write_durably(ledger_path, ledger)
            ledger = refresh_allocation_ledger(work)
    doc = {'ok':all(r.get('state') == 'ended' for r in rows) and all(a.get('end') or (a.get('terminal_observation') or {}).get('state') in ('RUNNING','COMPLETING') for a in (ledger or {}).get('allocations', {}).values()), 'rows':rows, 'allocation_ledger':ledger}
    if (work/DIRECTORY/'v4-stop.json').exists():
        from kit.v4_ack import gate
        try:
            if not doc['ok']:raise ValueError('termination/exposure still unresolved')
            with containment_lock(work):gate(work,release=True)
        except (OSError,ValueError) as error:
            doc.update(ok=False,hard_stop_reason=str(error))
    wd.write_durably(Path(out)/'containment-reconcile.json', doc)
    return doc


@tracked_allocation
def run_row(work, out, argv, gpus, time_cap, *, block="qualification", block_limit=100, ceiling=560, concurrent=False):
    """Launch any v4 argv under a TOTAL GPU exposure cap, including W and J.

    Campaign supplies block and ceiling. Allocation width times wall exposure
    includes idle time; overlapping reservations share the allocation charge.
    """
    if is_phase0():return phase0_row(work,out,argv,gpus,time_cap,block,block_limit,ceiling)
    if not argv or not isinstance(gpus, int) or not 1 <= gpus <= 8 or not math.isfinite(time_cap) or time_cap < MIN_ROW_CAP:
        raise Refused('row needs argv, 1..8 GPUs and total time-cap >=300 seconds (startup + one minute + J=60 + KillWait=40 + cleanup)')
    if concurrent and gpus != 1: raise Refused('concurrency is only for 1-GPU rows; multi-GPU rows run alone')
    work, out = Path(work), Path(out)
    if (out / 'containment-row.json').exists():
        raise Refused('row output already exists; need a new attempt output to preserve evidence')
    stop_gate(work)
    errors, docs = verify_receipts(work)
    if docs.get('selftest', {}).get('v4_qualified') is not True:
        errors.append('need a v4 selftest verdict including CPU detached-grandchild and delayed one-GPU probes')
    if errors:
        raise Refused('; '.join(errors))
    live = _live_cluster(docs['receipt'])
    with containment_lock(work):
        for path in (work / DIRECTORY / 'rows').glob('*.json'):
            prior = wd.read_json(path) or {}
            if prior.get('state') != 'ended':
                if (not concurrent or gpus != 1 or prior.get('gpus') != 1 or prior.get('state') not in ('launching', 'running') or
                        prior.get('allocation_id') != live['job_id'] or not prior.get('client_pid') or
                        not pid_alive(prior['client_pid']) or
                        (prior.get('state') == 'running' and not prior.get('assigned_devices')) or
                        time.time() >= epoch(prior['deadline'])):
                    raise Refused('prior row unresolved at %s; reconcile owning-step termination before admission' % path)
        if len([1 for p in (work/DIRECTORY/'rows').glob('*.json') if (wd.read_json(p) or {}).get('state')!='ended']) >= live['gpu_count']:
            raise Refused('allocation has no free one-GPU row slots')
        assigned = []  # Slurm chooses the actual bitmap after RPC arrival.
        budget = allocation_budget(work, live, time_cap, block, block_limit, ceiling)
        stamp = '%d-%d' % (time.time_ns(), os.getpid())
        now = epoch(budget['admitted_at']); until = now+time_cap
        data = {'schema': 'kit-v4-contained-row.v1', 'row': 'row-'+stamp, 'attempt': 1, 'command': list(argv),
                'containment': 'slurm-step', 'kind': 'row', 'state': 'launching', 'gpus': gpus,
                'output_path':str(out.resolve()),
                'allocation_id':live['job_id'], 'allocation_width':live['gpu_count'], 'assigned_devices':assigned, 'concurrent':concurrent,
                'client_pid':os.getpid(), 'budget_block':block, 'budget':budget,
                'ledger_charge_gpu_hours':live['gpu_count']*time_cap/3600,
                'time_cap_seconds': time_cap, 'cluster_recheck': live, 'started_at': wd.precise_text(wd.from_epoch(now)),
                **{k: wd.precise_text(wd.from_epoch(until)) for k in ('limit_until', 'kill_by', 'deadline')}}
        record = RowRecord(work, work / DIRECTORY / 'rows' / (stamp+'.json'), data)
    proc = None
    try:
        env = dict(os.environ)
        proc, release = start_step(argv, env, record, gpus)
        release.touch()
        record.update(state='running', launched=True, slurm={**record.data['slurm'], 'released_at': wd.precise_text()})
        while proc.poll() is None and time.time() < until:
            time.sleep(shared_limit(.05))
        s = record.data['slurm']
        cancellation = cancel(s['job_id'], s['step_id'])
        terminated = evidence(s['job_id'], s['step_id'], ','.join(s.get('expected_gpu_uuids', assigned)), until)
        failure = termination_failure(terminated, proc.poll())
        if not terminated.get('scheduler_end') or epoch(terminated['scheduler_end']) > until:
            failure = 'termination outside time cap'
        verified = bool(terminated.get('verified') and terminated.get('verified_at'))
        within_cap = bool(terminated.get('scheduler_end') and epoch(terminated['scheduler_end'])<=until
                          and verified and epoch(terminated['verified_at'])<=until)
        if verified and not within_cap:
            failure = 'termination outside time cap'
        if not verified and failure != 'preempted':
            failure = 'workers not verified terminated within time cap'
        record.update(state='ended' if verified else 'unverified', ok=int(failure is None),
                      status=0 if failure is None else 2, failure_type=failure,
                      slurm={**s, 'cancel': cancellation, 'termination': terminated},
                      charged_until=terminated.get('verified_at') if verified else None,
                      charged_gpu_hours_upper_bound=(epoch(terminated['verified_at'])-now)*gpus/3600 if verified else None,
                      ledger_charge_gpu_hours=(epoch(terminated['verified_at'])-now)*live['gpu_count']/3600 if verified else live['gpu_count']*time_cap/3600)
        cls=stop_class(failure,verified,within_cap)
        record.update(stop_class=cls)
        if cls=='hard':hard_stop(work,failure,record.path)
        if cls=='allocation_preempted':record.update(allocation_ended=True)
    except (Refused, OSError, subprocess.SubprocessError) as error:
        record.update(ok=0, status=2, state='unverified', failure_type='containment refused', reason=str(error))
        record.update(stop_class='hard')
        hard_stop(work,str(error),record.path)
    finally:
        if proc and proc.poll() is None:
            proc.kill(); proc.wait(timeout=shared_limit(5))
        with containment_lock(work):
            refresh_allocation_ledger(work, record.data)
        wd.write_durably(out / 'containment-row.json', record.data)
    return record.data


def phase0_row(work,out,argv,gpus,time_cap,block,block_limit,ceiling):
    """Run the existing payload in a Slurm step; containment observations are informational."""
    from kit.runner import bounded_command
    work,out=Path(work).resolve(),Path(out).resolve()
    if not out.is_relative_to(work):raise Refused('row output must be inside WORK')
    out.mkdir(parents=True,exist_ok=True)
    obs=allocation_observation(os.environ.get('SLURM_JOB_ID'))
    live={'job_id':obs['job_id'],'gpu_count':obs.get('width'),'allocation_start':obs.get('start'),
          'allocation_end':obs.get('end'),'allocation_time_limit':sbatch_time(obs.get('time_limit_seconds') or 0)}
    ledger=allocation_budget(work,live,time_cap,block,block_limit,ceiling)
    start=time.time()
    minutes_requested=max(1,math.ceil(time_cap/60))
    step=out/'slurm-step.json'
    entry="import json,os,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith('SLURM_') or k=='CUDA_VISIBLE_DEVICES'})); os.execvpe(sys.argv[2],sys.argv[2:],os.environ)"
    command=['srun','--ntasks=1','--exclusive','--gres=gpu:'+str(gpus),'--time='+str(minutes_requested),
             '--kill-on-bad-exit=1']
    if obs.get('cpus') and obs.get('memory_mb') and obs.get('width'):
        command+=['--cpus-per-task='+str(obs['cpus']*gpus//obs['width']),'--mem='+str(obs['memory_mb']*gpus//obs['width'])+'M']
    command += [sys.executable,'-c',entry,str(step),*argv]
    result=bounded_command(command,timeout=time_cap,env=os.environ.copy(),log=out/'phase0-step.log')
    sys.stdout.write((out/'phase0-step.log').read_text(errors='replace'))
    doc={'schema':'kit-v4-contained-row.v1','informational':True,'allocation_id':obs['job_id'],
         'allocation_width':obs['width'],'gpus':gpus,'command':list(argv),'actual_command':command,
         'started_at':wd.precise_text(wd.from_epoch(start)),'ended_at':wd.precise_text(),
         'time_cap':time_cap,'state':'ended','status':result['returncode'],'ok':int(result['returncode']==0),
         'slurm':wd.read_json(step) or {},'execution':result,'containment':'Slurm step; owner monitored'}
    refresh_allocation_ledger(work)
    wd.write_durably(out/'containment-row.json',doc)
    return doc


def termination_failure(termination, status):
    """Scheduler failures stay typed even if a launcher exits zero during grace."""
    state = termination.get('state')
    if state == 'PREEMPTED' or termination.get('failure_type') == 'preempted':
        return 'preempted'
    if termination.get('failure_type'):
        return termination['failure_type']
    if state == 'TIMEOUT':
        return 'time cap'
    if not termination.get('verified'):
        return 'workers not verified terminated'
    if state != 'COMPLETED' or status != 0:
        return 'launcher failure'
    return None


@tracked_allocation
def selftest(work, out, gpu_seconds=5, attempt_record=None, *, block='qualification', block_limit=100, ceiling=560, _overall_started=None):
    """One frozen CPU/GPU experiment per allocation, with shared allocation accounting."""
    folder = allocation_path(work, SELFTEST).parent
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'selftest.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refused('selftest already running for this allocation; no second experiment') from None
        previous_handler=signal.getsignal(signal.SIGALRM)
        previous_timer=signal.getitimer(signal.ITIMER_REAL)
        def expired(signum,frame):raise Refused('containment selftest exceeded its registered 600-second safety bound')
        signal.signal(signal.SIGALRM,expired)
        seconds=max(.001,shared_limit(SELFTEST_SECONDS)-(time.time()-_overall_started)) if _overall_started is not None else shared_limit(SELFTEST_SECONDS)
        signal.setitimer(signal.ITIMER_REAL,seconds)
        try:return _recorded_selftest(work, out, gpu_seconds, attempt_record, block, block_limit, ceiling, _overall_started)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous_handler)
            signal.setitimer(signal.ITIMER_REAL,*previous_timer)


def _recorded_selftest(work, out, gpu_seconds, attempt_record, block, block_limit, ceiling, overall_started=None):
    result = None
    try:
        old=wd.read_json(allocation_path(work,SELFTEST))
        if old and old!=frozen.seal(old):raise Refused('existing selftest changed')
        if not old or old.get('ok'):stop_gate(work)
        try:result = _selftest(work, out, gpu_seconds, attempt_record, block, block_limit, ceiling, overall_started)
        except (Refused,OSError,subprocess.SubprocessError) as exc:
            if not is_phase0():raise
            result=frozen_write(allocation_path(work,SELFTEST),{'schema':'kit-p4-containment-selftest.v1','ok':False,'v4_qualified':False,'problems':[str(exc)]})
        if not result.get('ok') and not is_phase0():
            hard_stop(work,'containment selftest failed: '+ '; '.join(result.get('problems',[])))
        return result
    finally:
        with containment_lock(work):
            ledger = wd.read_json(Path(work)/DIRECTORY/'allocation-ledger.json')
            job = os.environ.get('SLURM_JOB_ID')
            entry = (ledger or {}).get('allocations', {}).get(job)
            if entry and entry.get('qualification_interval', {}).get('state') == 'running':
                entry['qualification_interval'].update(end=wd.precise_text(), state='ended',
                    ok=bool(result and result.get('ok')))
                wd.write_durably(Path(work)/DIRECTORY/'allocation-ledger.json', ledger)
                refresh_allocation_ledger(work)


def _selftest(work, out, gpu_seconds, attempt_record, block, block_limit, ceiling, overall_started=None):
    work, out = Path(work), Path(out)
    if is_phase0() and gpu_seconds==5:gpu_seconds=shared_limit(gpu_seconds)
    if not math.isfinite(gpu_seconds) or not 0 <= gpu_seconds <= shared_limit(GPU_PROBE_SECONDS):
        raise Refused('--gpu-seconds must be finite and between 0 and 160')
    if out.resolve() == (work/DIRECTORY).resolve():
        out = allocation_path(work, SELFTEST).parent
    started = overall_started if overall_started is not None else time.time()
    target=allocation_path(work,SELFTEST)
    old=wd.read_json(target)
    if old:
        if old!=frozen.seal(old): raise Refused('existing selftest changed')
        if old.get('failure_type')!='allocation budget':
            errors,docs=verify_receipts(work,selftest=False)
            if old.get('receipt_sha256')!=docs.get('receipt',{}).get('content_sha256'):
                raise Refused('existing selftest changed; first receipt is never replaced')
        if old.get('ok') is True and not is_phase0():
            obs=allocation_observation(os.environ.get('SLURM_JOB_ID'))
            # An archive-only replay cannot launch; a live cached pass still
            # must obey the current allocation reservation, without replacing it.
            if obs.get('start') and obs.get('width'):
                live={'job_id':obs['job_id'],'gpu_count':obs['width'],'allocation_start':obs['start'],
                      'allocation_end':obs.get('end'),'allocation_time_limit':sbatch_time(obs.get('time_limit_seconds') or 0)}
                _live_cluster(docs['receipt'])
                with containment_lock(work): allocation_budget(work,live,1,block,block_limit,ceiling)
        if (out/SELFTEST.name).resolve()!=target.resolve(): frozen_write(out/SELFTEST.name,old)
        return old
    observation=allocation_observation(os.environ.get('SLURM_JOB_ID'))
    live_budget={'job_id':observation['job_id'],'gpu_count':observation.get('width'),
        'allocation_start':observation.get('start'),'allocation_end':observation.get('end'),
        'allocation_time_limit':sbatch_time(observation.get('time_limit_seconds') or 0)}
    try:
        with containment_lock(work): allocation_budget(work,live_budget,shared_limit(SELFTEST_SECONDS),block,block_limit,ceiling)
    except Refused as error:
        failed=frozen_write(target,{'schema':'kit-p4-containment-selftest.v1','ok':False,'v4_qualified':False,
            'failure_type':'allocation budget','problems':[str(error)],
            'allocation_admission':wd.read_json(allocation_path(work,DIRECTORY/'allocation-admission.json'))})
        if (out/SELFTEST.name).resolve()!=target.resolve(): frozen_write(out/SELFTEST.name,failed)
        return failed
    if not allocation_path(work, SELFTEST).exists():
        preflight = cpu_preflight()
        wd.write_durably(allocation_path(work, SELFTEST).parent/'cpu-preflight.json', preflight)
        if not preflight['ok']:
            raise Refused('; '.join(preflight['problems']))
    if not allocation_path(work, RECEIPT).exists():
        check(work, out)
    errors, docs = verify_receipts(work, selftest=False)
    target = allocation_path(work, SELFTEST)
    requested_target = out / SELFTEST.name
    if target.exists():
        old = wd.read_json(target)
        if not old or old != frozen.seal(old) or old.get('receipt_sha256') != docs.get('receipt', {}).get('content_sha256'):
            raise Refused('existing selftest changed; first receipt is never replaced')
        if requested_target.resolve() != target.resolve():
            frozen_write(requested_target, old)
        return old  # Frozen experimental evidence is not rerun to obtain a passing outcome.
    if docs.get('receipt', {}).get('expected') != EXPECTED:
        errors.append('need a fresh v4 cluster receipt with the answered settings and known preemption applicability')
    if errors:
        failed = frozen_write(target, {'schema': 'kit-p4-containment-selftest.v1', 'ok': False,
                              'problems': errors + docs.get('receipt', {}).get('problems', []),
                              'receipt_sha256': docs.get('receipt', {}).get('content_sha256'),
                              'settings': docs.get('receipt'), 'elapsed_seconds': time.time()-started})
        if requested_target.resolve() != target.resolve():
            frozen_write(requested_target, failed)
        return failed
    try:
        baseline = reviewed_receipt(work) or docs['receipt']
        first_verdict = wd.read_json(allocation_path(work,SELFTEST,baseline['job_id']))
        if docs['receipt']['job_id'] != baseline['job_id'] and (not first_verdict or first_verdict.get('v4_qualified') is not True or first_verdict != frozen.seal(first_verdict)) and (first_verdict or {}).get('failure_type')!='allocation budget':
            raise Refused('reviewed first allocation has no passing frozen verdict')
        changed = [k for k in SAFETY_FIELDS if baseline.get(k) != docs['receipt'].get(k)]
        if changed:
            raise Refused('allocation safety settings differ from reviewed first receipt: '+', '.join(changed))
        live = _live_cluster(docs['receipt'])
    except Refused as error:
        failed = frozen_write(target, {'schema': 'kit-p4-containment-selftest.v1', 'ok': False,
                              'problems': [str(error)], 'settings': docs['receipt'],
                              'receipt_sha256': docs['receipt']['content_sha256']})
        if requested_target.resolve() != target.resolve():
            frozen_write(requested_target, failed)
        return failed
    with containment_lock(work):
        for path in (work/DIRECTORY/'rows').glob('*.json'):
            if (wd.read_json(path) or {}).get('state') != 'ended':
                raise Refused('prior row unresolved; cannot qualify alongside an active/unverified row')
        budget = allocation_budget(work, live, shared_limit(SELFTEST_SECONDS), block, block_limit, ceiling)
        budget['allocations'][live['job_id']]['qualification_interval'] = {
            'start':wd.precise_text(wd.from_epoch(started)), 'end':None, 'state':'running',
            'cpu_gpus':0, 'gpu_probe_gpus':1, 'gpu_probe_seconds_cap':160,
            'deadline':wd.precise_text(wd.from_epoch(started+shared_limit(SELFTEST_SECONDS)))}
        wd.write_durably(work/DIRECTORY/'allocation-ledger.json', budget)
    cpu = cpu_probe(work, out, min(started+shared_limit(SELFTEST_SECONDS), time.time()+shared_limit(120)))
    # Remaining time is observed, not a prediction gate. Actual receipt/death
    # deadlines and the independent scheduler limits still determine the verdict.
    cpu['remaining_selftest_seconds']=max(0,started+shared_limit(SELFTEST_SECONDS)-time.time())
    if time.time()>=started+shared_limit(SELFTEST_SECONDS)-shared_limit(40):
        cpu={**cpu,'ok':False,'problems':['containment self-test hard deadline reached before GPU launch']}
    if not cpu['ok']:
        failed = frozen_write(target, {'schema': 'kit-p4-containment-selftest.v1', 'ok': False,
                              'problems': cpu['problems'], 'cpu_probe': cpu, 'settings': docs['receipt'],
                              'receipt_sha256': docs['receipt']['content_sha256']})
        if requested_target.resolve() != target.resolve():
            frozen_write(requested_target, failed)
        return failed
    folder = out / 'selftest-processes'
    folder.mkdir(parents=True, exist_ok=True)
    delay = float(os.environ.get('SIM_SLURM_GPU_DELAY', '5')) if os.environ.get('SIM_SLURM_STATE') else shared_limit(5)
    # Independent wrapper process: the CLI observer stays OUTSIDE the step.
    request = {'work': str(work), 'folder': str(folder), 'gpu_seconds': gpu_seconds, 'delay': delay,
               'simulation': docs['receipt'].get('simulation') is True,
               'attempt_record': str(attempt_record.path) if attempt_record else None}
    request_path = folder / 'request.json'
    wd.write_durably(request_path, request)
    wrapper = None
    info, problems, deaths = None, [], {}
    until = min(started + shared_limit(SELFTEST_SECONDS)-shared_limit(40), time.time() + shared_limit(START_WAIT))
    try:
        wrapper = subprocess.Popen([sys.executable, str(HERE / 'p4_contain.py'), '_selftest-wrapper', str(request_path)], start_new_session=True)
        while time.time() < until:
            info = wd.read_json(folder / 'wrapper.json')
            if info and info.get('ready'):
                break
            if wrapper.poll() is not None:
                raise Refused('selftest wrapper failed before release: %s' % info)
            time.sleep(shared_limit(.05))
        if not info or not info.get('ready'):
            raise Refused('selftest did not start')
        # Observe early CUDA restriction before removing clients; scheduler alone then enforces cleanup.
        # Invariant: prove one-device CUDA restriction with living clients FIRST.
        # Only then remove all client enforcement and require every adversarial PID
        # dead plus idle GPUs by kill_by. Slurm may cancel promptly on client loss;
        # a late receipt is observational, never required after that valid cleanup.
        early_until=min(time.time()+shared_limit(180), started+shared_limit(SELFTEST_SECONDS)-shared_limit(40), epoch(info['slurm']['deadline']))
        gpu=None
        while time.time()<early_until:
            gpu=wd.read_json(folder/'gpu.json')
            if gpu is not None:break
            if wrapper.poll() is not None:break
            time.sleep(shared_limit(.05))
        if gpu is None:raise Refused('early GPU acquisition not exercised before client removal (bounded wait up to 180 seconds)')
        if not gpu.get('ok') or gpu.get('accessible_cuda_devices')!=1:
            problems.append(gpu.get('why','one-GPU device restriction failed'))
        for pid in [wrapper.pid, info['srun_pid'], info['watchdog_pid']]:
            try: os.kill(pid, signal.SIGKILL)
            except ProcessLookupError: pass
        wrapper.wait(timeout=shared_limit(5))
        clients=[wrapper.pid,info['srun_pid'],info['watchdog_pid']]
        client_deadline=min(epoch(info['slurm']['kill_by']),started+shared_limit(SELFTEST_SECONDS)-shared_limit(40))
        while time.time()<client_deadline and any(pid_alive(pid) for pid in clients):time.sleep(shared_limit(.02))
        if any(pid_alive(pid) for pid in clients):raise Refused('enforcing self-test clients did not die by the containment deadline')
        # Release the late touch only AFTER observed client death. This prevents
        # a slow client teardown from making a supposedly adversarial touch early.
        client_loss_at=wd.precise_text()
        wd.write_durably(folder/'clients-lost.json',{'client_loss_at':client_loss_at,'observed_dead_clients':clients})
        slurm = info['slurm']; S = epoch(slurm['scheduler_start'])
        limit, kill_by, deadline = probe_bounds(S, slurm['J'], slurm['W'])
        kill_by = min(kill_by, epoch(slurm['kill_by']))
        deadline = min(deadline, started+shared_limit(SELFTEST_SECONDS)-shared_limit(40), epoch(slurm['deadline']))
        pids = {p.stem: int(p.read_text()) for p in folder.glob('*.pid')}
        # Lose outside process discovery across the timeout as well as losing
        # every enforcing client. The observer resumes within J to verify the
        # deadline; its later death observations are upper bounds, not exact
        # kernel exit times. Fast clocks belong solely to the stand-in probe.
        probe_minute = float(os.environ.get('SIM_SLURM_SECONDS_PER_MINUTE', '60')) if os.environ.get('SIM_SLURM_STATE') else 60
        stalled_from = time.time()
        stall_until = min(kill_by-1, S+probe_minute+slurm['W']+.5)
        while time.time() < stall_until:
            time.sleep(min(shared_limit(.1), stall_until-time.time()))
        stalled = {'started_at': wd.precise_text(wd.from_epoch(stalled_from)), 'resumed_at': wd.precise_text(),
                   'seconds': time.time()-stalled_from, 'step_clock_seconds_per_minute': probe_minute}
        terminal_seen = None
        while time.time() < kill_by:
            for name, pid in pids.items():
                if name not in deaths and not pid_alive(pid):
                    deaths[name] = {'pid': pid, 'observed_dead_at': wd.precise_text(), 'seconds_from_limit': time.time()-limit, 'seconds_from_step_timeout': time.time()-(S+shared_limit(60))}
            gpu = wd.read_json(folder / 'gpu.json')
            if len(deaths) == len(pids):
                break
            state, _ = steps(slurm['job_id'])
            if any(s.get('StepId') == step_ref(slurm['job_id'], slurm['step_id']) and s.get('State') in TERMINAL for s in state):
                terminal_seen = terminal_seen or time.time()
                if time.time() - terminal_seen > 1:
                    break  # A terminal scheduler that left adversarial pids alive already fails.
            time.sleep(shared_limit(.05))
        gpu = wd.read_json(folder / 'gpu.json') or {'ok': False, 'why': 'gpu acquisition not exercised: no acquisition receipt'}
        if not gpu.get('ok'):
            problems.append(gpu.get('why', 'gpu acquisition not exercised'))
        late_gpu=wd.read_json(folder/'gpu-late.json')
        if late_gpu is not None and not late_gpu.get('ok'):problems.append('late GPU re-touch failed: '+str(late_gpu.get('why')))
        expected = {'double-fork-setsid', 'double-fork-setsid-parent', 'cleared-exec', 'ignores-term',
                    'orphan-grandchild', 'orphan-grandchild-parent', 'later-gpu', 'launcher'}
        if set(pids) != expected or set(deaths) != expected:
            problems.append('not every adversarial pid was recorded and dead by kill_by')
        terminated = evidence(slurm['job_id'], slurm['step_id'], ','.join(slurm['expected_gpu_uuids']), deadline)
        if not terminated['verified'] or terminated.get('state') not in ('TIMEOUT', 'CANCELLED') or terminated.get('failure_type'):
            problems.append('step was not verified timeout/cancel with idle GPUs')
        if not terminated.get('scheduler_end') or epoch(terminated.get('verified_at')) > kill_by:
            problems.append('whole-step termination/GPU release was not verified by kill_by')
        doc = {'schema': 'kit-p4-containment-selftest.v1', 'ok': not problems, 'problems': problems, 'v4_qualified': not problems,
               'simulation': docs['receipt'].get('simulation') is True, 'settings': docs['receipt'],
               'cpu_probe': cpu, 'cluster_recheck': live, 'failure_type': terminated.get('failure_type') if problems else None,
               'elapsed_seconds': time.time()-started, 'cost_bounds': {'wall_seconds': shared_limit(600), 'gpus': 1, 'gpu_seconds': shared_limit(160)},
               'receipt_sha256': docs['receipt']['content_sha256'], 'slurm': slurm,
               'killed_clients': {'wrapper': wrapper.pid, 'srun': info['srun_pid'], 'watchdog': info['watchdog_pid']},
               'pids': pids, 'deaths': deaths, 'gpu_acquisition': gpu, 'late_gpu_acquisition':late_gpu,
            'client_loss_at':client_loss_at, 'termination_mode':'client_loss_cancellation' if terminated.get('state')=='CANCELLED' else 'time_limit', 'termination': terminated,
               'stalled_discovery': stalled, 'death_time_kind': 'outside observation upper bound (including a deliberate discovery stall)',
               'limit_until': wd.precise_text(wd.from_epoch(limit)), 'kill_by': wd.precise_text(wd.from_epoch(kill_by)),
               'deadline': wd.precise_text(wd.from_epoch(deadline)), 'step_timeout_at': wd.precise_text(wd.from_epoch(S+shared_limit(60))), 'gpu_seconds': gpu_seconds,
               'proc_observation': '/proc stat (zombies have exited)' if Path('/proc/self/stat').exists() else 'ps stat fallback (stand-in on macOS)'}
        sealed = frozen_write(target, doc)
        if requested_target.resolve() != target.resolve():
            frozen_write(requested_target, sealed)
        if attempt_record:
            attempt_record.data = wd.read_json(attempt_record.path) or attempt_record.data
            # The probe's expected timeout is not its admitted 300-second spending limit.
            at = wd.parse_time(terminated.get('verified_at'))
            attempt_record.update(slurm={**slurm, 'selftest_sha256': sealed['content_sha256'], 'termination': terminated}, launched=True, state='ended' if doc['ok'] else 'unverified',
                                  **{'class': 'complete' if doc['ok'] else 'workers not verified terminated'}, ok=int(doc['ok']), status=0 if doc['ok'] else 2,
                                  charged_until=wd._ceil_text(at), ended_at=wd._ceil_text(at), ended_precise=wd.precise_text(at),
                                  verified_idle_at=wd._ceil_text(at) if doc['ok'] else None, elapsed_seconds=time.time()-epoch(attempt_record.data['launched_at']),
                                  selftest_expected_timeout=True)
            if not doc['ok'] and not is_phase0():
                wd.write_stop(attempt_record.path, attempt_record.data, '; '.join(problems), 'containment selftest failed', unavailable=True)
        return sealed
    except (Refused, OSError, subprocess.SubprocessError) as error:
        # Startup/observation failures are also a first, frozen failed experiment;
        # a later invocation cannot silently rerun the probe to obtain a pass.
        failed = frozen_write(target, {'schema': 'kit-p4-containment-selftest.v1', 'ok': False,
                                      'problems': ['selftest could not complete: %s' % error],
                                      'receipt_sha256': docs['receipt']['content_sha256'], 'partial_wrapper': info,
                                      'pids': {p.stem: int(p.read_text()) for p in folder.glob('*.pid')}, 'deaths': deaths})
        if requested_target.resolve() != target.resolve():
            frozen_write(requested_target, failed)
        if attempt_record:
            attempt_record.data = wd.read_json(attempt_record.path) or attempt_record.data
            attempt_record.update(state='unverified', accounting='budget compliance unavailable',
                                  **{'class': 'containment selftest failed'}, ok=0, status=2, reason=str(error))
            if not is_phase0():wd.write_stop(attempt_record.path, attempt_record.data, str(error), 'containment selftest failed', unavailable=True)
        return failed
    finally:
        if wrapper is not None and wrapper.poll() is None:
            wrapper.kill(); wrapper.wait(timeout=shared_limit(5))
        # Failure evidence is preserved first. Then only this step and its recorded test pids are cleaned up.
        if info:
            for key in ('srun_pid', 'watchdog_pid'):
                try: os.kill(info[key], signal.SIGKILL)
                except (KeyError, ProcessLookupError): pass
        if info and info.get('slurm', {}).get('step_id'):
            cancel(info['slurm']['job_id'], info['slurm']['step_id'])
        for path in folder.glob('*.pid'):
            try: os.kill(int(path.read_text()), signal.SIGKILL)
            except (OSError, ValueError): pass


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        p = Path('/proc/%d/stat' % pid)
        if Path('/proc/self/stat').exists():
            state = p.read_text().rsplit(')', 1)[1].split()[0]
        else:
            observed = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True)
            if observed.returncode:
                return True  # A denied/failed observation never proves death.
            state = observed.stdout.strip()
        return bool(state) and not state.startswith('Z')
    except (OSError, ValueError):
        return True  # kill(0) established existence; a failed scan cannot undo it.


def selftest_wrapper(request_path):
    # Uses the same park/start/recheck code as a row; the selftest itself cannot yet require a selftest receipt.
    req = wd.read_json(Path(request_path)); work, folder = Path(req['work']), Path(req['folder'])
    class Record:
        def __init__(self):
            self.work = work
            self.path = Path(req['attempt_record']) if req.get('attempt_record') else folder / 'probe-record.json'
            self.data = wd.read_json(self.path) or {'row': 'containment-probe', 'attempt': 1, 'containment': 'slurm-step'}
        def update(self, **fields):
            self.data.update(fields)
            wd.write_durably(self.path, self.data)
    r = Record()
    if not r.data.get('limit_until'):
        now = time.time()
        r.update(limit_until=wd.precise_text(wd.from_epoch(now+shared_limit(300))), kill_by=wd.precise_text(wd.from_epoch(now+shared_limit(345))),
                 deadline=wd.precise_text(wd.from_epoch(now+shared_limit(360))))
    r.update(job_marker='containment-selftest-%d' % os.getpid(), cuda_visible_devices='0', kill_grace_seconds=30)
    os.environ[wd.MARKER_VAR] = r.data['job_marker']
    argv = [sys.executable, '-c', PAYLOAD, str(folder), str(req['delay']), str(req['gpu_seconds'])]
    if req.get('simulation'):
        argv.append('--stand-in')  # real cluster receipts always require torch/CUDA
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    proc, release = start_step(argv, env, r, 1, require_selftest=False, requested_cap=shared_limit(1))
    # Actual kit watchdog, started with the parked process group, then deliberately killed.
    run = _load('p4_run')
    watchdog_pid, why = run.start_watchdog(r.path, folder / 'watchdog.json', r.data['pgid'])
    if watchdog_pid is None:
        cancel(r.data['slurm']['job_id'], r.data['slurm']['step_id'])
        raise Refused(why)
    wd.write_durably(folder / 'wrapper.json', {'ready': False, 'srun_pid': proc.pid, 'watchdog_pid': watchdog_pid, 'slurm': r.data['slurm']})
    r.update(state='running', launched=True, launched_at=r.data.get('launched_at') or wd.precise_text(),
             watchdog={'pid': watchdog_pid, 'record': str(folder/'watchdog.json')})
    release.touch()
    end = time.time()+shared_limit(600)
    while time.time()<end:
        if len(list(folder.glob('*.pid'))) == 8:
            # The grandchildren overwrite their pid after the second fork.
            time.sleep(shared_limit(.4))
            wd.write_durably(folder / 'wrapper.json', {'ready': True, 'srun_pid': proc.pid, 'watchdog_pid': watchdog_pid, 'slurm': r.data['slurm']})
            proc.wait()
            return 0
        time.sleep(shared_limit(.02))
    cancel(r.data['slurm']['job_id'], r.data['slurm']['step_id'])
    try: os.kill(watchdog_pid, signal.SIGKILL)
    except ProcessLookupError: pass
    raise Refused('adversarial workers did not record their pids')


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_selftest-wrapper':
        return selftest_wrapper(argv[1])
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('check', 'selftest'):
        p = sub.add_parser(name); p.add_argument('--work', required=True); p.add_argument('--out', required=True)
        if name == 'check':
            p.add_argument('--from-file'); p.add_argument('--qos')
        if name == 'selftest':
            p.add_argument('--gpu-seconds', type=float, default=shared_limit(5))
            p.add_argument('--block', default='qualification'); p.add_argument('--block-limit', type=float, default=100)
            p.add_argument('--ceiling', type=float, default=560)
            p.add_argument('--campaign'); p.add_argument('--reservation'); p.add_argument('--attempt', type=int, default=1)
    p = sub.add_parser('check-live-format'); p.add_argument('--from-file', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('row'); p.add_argument('--work', required=True); p.add_argument('--out', required=True)
    p.add_argument('--gpus', type=int, required=True); p.add_argument('--time-cap', type=float, required=True)
    p.add_argument('--block', default='qualification'); p.add_argument('--block-limit', type=float, required=True)
    p.add_argument('--ceiling', type=float, required=True); p.add_argument('--concurrent', action='store_true')
    p.add_argument('command', nargs=argparse.REMAINDER)
    p = sub.add_parser('reconcile'); p.add_argument('--work', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=float, default=shared_limit(600)); p.add_argument('--force',action='store_true')
    p = sub.add_parser('park'); p.add_argument('--ready', required=True); p.add_argument('--release', required=True)
    p.add_argument('--count', type=int, required=True); p.add_argument('command', nargs=argparse.REMAINDER)
    p.add_argument('--expected-uuids'); p.add_argument('--allocation-gpu-map',type=json.loads)
    args = parser.parse_args(argv)
    try:
        if args.mode == 'check-live-format':
            doc = frozen_write(args.out, check_live_format(args.from_file))
            print(json.dumps(doc)); return 0 if doc['ok'] else 2
        # CLI tracking must charge refusals, but cannot manufacture the prior
        # ledger needed to reconcile a mistaken/never-qualified WORK.
        missing_reconcile_ledger=False
        if args.mode=='reconcile':
            prior=wd.read_json(Path(args.work)/DIRECTORY/'allocation-ledger.json')
            from kit.v4_phase0_submission import receipt_path
            missing_reconcile_ledger=(not prior or not prior.get('allocations')) and not receipt_path(args.work).is_file()
        if args.mode not in ('park',) and getattr(args,'work',None) and not getattr(args,'from_file',None):
            track_allocation(args.work,getattr(args,'block',None),getattr(args,'block_limit',None),getattr(args,'ceiling',None))
        if missing_reconcile_ledger:raise Refused('missing allocation ledger at reconcile invocation; verify WORK')
        if args.mode == 'park':
            return park(args.ready, args.release, args.count, args.command[1:] if args.command[:1] == ['--'] else args.command,
                        args.expected_uuids,args.allocation_gpu_map)
        if args.mode == 'reconcile':
            doc = reconcile(args.work, args.out, args.seconds, force=args.force)
            print(json.dumps(doc)); return 0 if doc['ok'] else 2
        if args.mode == 'row':
            doc = run_row(args.work, args.out, args.command[1:] if args.command[:1] == ['--'] else args.command, args.gpus, args.time_cap, block=args.block, block_limit=args.block_limit, ceiling=args.ceiling, concurrent=args.concurrent)
            print(json.dumps(doc)); return doc['status']
        record = None
        if args.mode == 'selftest' and allocation_path(args.work, SELFTEST).exists() and args.campaign:
            # A runner may reattempt this pilot after losing its verdict. Preserve
            # provenance for that invocation while reusing the first frozen probe.
            run = _load('p4_run'); work = Path(args.work)
            if not run.pb.attempt_record_path(work, 'containment-selftest', args.attempt).exists():
                replay = run.Record(work, 'containment-selftest', args.attempt,
                                    {'block': 'prevention', 'kind': 'selftest', 'containment': 'slurm-step',
                                     'command': [sys.executable, str(HERE/'p4_contain.py')] + argv})
                replay.update(state='reconciled', **{'class': 'reconciled'},
                              reason='first frozen selftest reused; no GPU work launched',
                              ended_at=wd.now_text(), elapsed_seconds=0)
        if args.mode == 'selftest' and args.campaign and not allocation_path(args.work, SELFTEST).exists():
            run = _load('p4_run'); work = Path(args.work)
            stale = frozen.verify(work, containment=False)  # this row creates the selftest receipt
            if stale: raise Refused(frozen.refusal(stale))
            record = run.Record(work, 'containment-selftest', args.attempt, {'block': 'prevention', 'kind': 'selftest', 'containment': 'slurm-step',
                                                                          'cuda_visible_devices': '0',
                                                                          'command': [sys.executable, str(HERE/'p4_contain.py')] + argv})
            record.update(recovered=run.recover(work, record.path, shared_limit(600)))
            acknowledged, note = run.acknowledgement_gate(work)
            if not acknowledged:
                record.update(state='busy', reason=note, elapsed_seconds=0, ended_at=wd.now_text())
                raise Refused(note)
            idle, note = run.wait_idle('0', shared_limit(600))
            if not idle:
                record.update(state='busy', reason=note, elapsed_seconds=0, ended_at=wd.now_text())
                raise Refused(note)
            gate = run.pb.decide(work, Path(args.campaign), Path(args.reservation or work/frozen.RESERVATION), 'prevention', 'containment-selftest',
                                 workers_verified=True, attempt=args.attempt)
            if gate['exit'] != 0:
                record.update(state='refused', **{'class': 'budget' if gate['exit']==1 else 'refused'}, reason=gate['reason'], ended_at=wd.now_text(), elapsed_seconds=0)
                raise Refused(gate['reason'])
            now = time.time(); limit = gate['limit_seconds']
            record.update(gate=gate, state='launching', limit_seconds=limit, launched_at=wd.precise_text(wd.from_epoch(now)),
                          limit_until=wd.precise_text(wd.from_epoch(now+limit)), kill_by=wd.precise_text(wd.from_epoch(now+limit+shared_limit(45))),
                          deadline=wd.precise_text(wd.from_epoch(now+limit+shared_limit(60))))
        doc = check(args.work, args.out, args.from_file, args.qos) if args.mode == 'check' else selftest(args.work, args.out, args.gpu_seconds, record, block=args.block, block_limit=args.block_limit, ceiling=args.ceiling)
        print(json.dumps({'ok': doc['ok'], 'problems': doc['problems'], 'content_sha256': doc['content_sha256']}))
        return 0 if (doc['ok'] or (args.mode=='selftest' and is_phase0())) else 2
    except (Refused, OSError) as e:
        print('containment refused: %s' % e, file=sys.stderr); return 2



def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if '--work' in argv:
        at=argv.index('--work')
        if at+1<len(argv) and (Path(argv[at+1])/'v4/report-phase0').exists():
            from kit.v4_phase0_site import storage, JOB_TMP
            # Inside the allocation keep the payload's short job-private TMPDIR, so contained
            # steps (and their multiprocessing listeners) never inherit the long shared path.
            in_job=bool(os.environ.get('SLURM_JOB_ID'))
            if in_job:Path(JOB_TMP).mkdir(parents=True,exist_ok=True)
            with storage(argv[at+1],allocation=in_job):return _main(argv)
    return _main(argv)

if __name__ == '__main__':
    sys.exit(main())
