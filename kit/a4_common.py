"""Registered 4A identities and chronology; no network or scheduler."""
from kit.p3b_common import *
from kit.c_common import lock, overlap
from kit import a4_sec as sec
from kit.beds import finqa
import socket
import time
import math
from datetime import datetime, timezone

MONTHS = tuple(f'2026-{m:02}' for m in range(1, 7))
SEEDS = tuple(range(101, 106))
ARMS = ('direct', 'consolidation')
BOUNDARIES = (2, 4, 6)
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
RES = KIT / 'resources/a4'


def protocol():
    paths = [*sorted(KIT.glob('a4_*.py')), KIT/'a4_payload.sh', KIT/'README-4a.md', RES/'slice.json', RES/'registration.md', RES/'SHA256SUMS',
             KIT/'resources/p3b/v4-recipes.json', KIT/'beds/finqa.py', KIT/'eval_bed.py',
             KIT/'score_forgetting.py', KIT/'scorers.py', KIT/'canonical.py',
             KIT/'c_train.py', KIT/'c_data.py', KIT/'c_sdft_entry.py', KIT/'c_sdft_runner.py',
             KIT/'p3b_loss.py', KIT/'sdft_objective.py', KIT/'sdft_actor.py', KIT/'sdft_verl.py',
             KIT/'sft_entry.py', KIT/'p3b_train.py', KIT/'p3b_common.py', KIT/'p3a_common.py',
             KIT/'v4_teacher.py', KIT/'v4_sft.py', KIT/'v4_contract.py', KIT/'v4_state.py',
             KIT/'v4_timing.py', KIT/'v4_phase0_timing.py', KIT/'p3b_reward.py',
             KIT/'p3b_read.py', KIT/'next_w.py', KIT/'c_bind.py',
             KIT/'resources/p3b/model-hashes.json', KIT/'panels/general-v1.jsonl']
    return {str(p.relative_to(KIT)): s.file_hash(p) for p in paths}


def binding(out, standin):
    p = out/'binding.json'
    if not p.exists(): raise NotReady('4A binding pending')
    d = json.loads(p.read_text())
    require(d['technical_synthetic'] == standin, '4A synthetic mismatch')
    require(d['protocol'] == protocol(), '4A implementation changed after binding')
    approved_ceiling(d)
    return d


def items(doc, month):
    require(month in MONTHS, 'unregistered month')
    p = Path(doc['slice'])/f'batch-{month}.json'
    check(p, doc['slice_files'][p.name])
    rows = json.loads(p.read_text())
    return sorted(rows, key=lambda i: (i['sec']['accepted'], i['sec']['accession'], i['id']))


def ledger(out, through):
    rows = []
    for month in MONTHS[:through]:
        root = out/'ledger'/month
        frozen_files(root)
        rows += t.read_jsonl(root/'traces.jsonl')
    return rows


def stage_name(arm, seed, month): return f'{arm}-s{seed}-m{month}'


def state(out, doc, arm, seed, through):
    require(arm in ARMS and seed in SEEDS, 'unregistered arm/seed')
    for n in reversed(range(1, through+1)):
        if arm == 'consolidation' and n not in BOUNDARIES: continue
        root = out/'runs'/stage_name(arm, seed, n)
        frozen_files(root)
        summary = json.loads((root/'summary.json').read_text())
        if not summary['skipped']:
            return dict(model=str(root/'hf-step40'), receipt=str(root/'export-40.json'))
    return doc['base']


def model_identity(record, standin):
    if standin: return dict(model=record['model'], technical_synthetic=True)
    return identity(record['model'], 'student' if not record.get('receipt') else None, record.get('receipt'))


def tokenizer(doc, standin):
    if standin:
        from kit.c_sim import Tokenizer
        return Tokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(doc['base']['model'], local_files_only=True)


def score_name(kind, arm, seed, month, panel, mode):
    return f'{kind}-{arm}-s{seed}-m{month}-{panel}-{mode}'


def score_root(out, kind, arm, seed, month, panel, mode):
    return out/'scores'/score_name(kind, arm, seed, month, panel, mode)


def arrival_specs():
    yield 'frozen', 0, 'off'
    yield 'memory', 0, 'on'
    for seed in SEEDS:
        yield 'direct', seed, 'off'
        yield 'consolidation', seed, 'on'
        yield 'consolidation', seed, 'off'


def freeze(root, names, **kw):
    atomic(root/'manifest.json', files_manifest(root, names, **kw))


# Owner, 21 October; registration section "Approved budget".
APPROVAL_REFERENCE = 'Approved with 600 partner GPU-hours. If it is tight, make it more'
APPROVAL_REGISTRATION = 'docs/phase4/4a-registration-20261020.md'
APPROVED_CEILINGS = {'F': 600, 'F+replay': 600, 'D': 800, 'D+replay': 800}


def approved_ceiling(doc):
    pinned = APPROVED_CEILINGS.get(doc.get('W'))
    value = doc.get('approved_ceiling_gpu_hours')
    require(pinned is not None and isinstance(value, (int, float)) and
            not isinstance(value, bool) and value == pinned,
            'pinned approved ceiling required for bound W')
    require(doc.get('approval_reference') == APPROVAL_REFERENCE,
            'pinned owner approval reference required for 4A ceiling')
    return value


def _slurm_times(job_id, allocation_start=None):
    """Read the allocation itself, never the next job's start or scheduled end."""
    terminal_states = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
        'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
    def stamp(value):
        if value in (None, '', 'Unknown', 'N/A', 'None'): return None
        return datetime.fromisoformat(value).timestamp()
    try:
        result = subprocess.run(['scontrol', 'show', 'job', '-o', str(job_id)],
                                capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        fields = {}
    else:
        fields = dict(part.split('=', 1) for part in result.stdout.split() if '=' in part)
    if fields.get('JobId'):
        require(fields['JobId'] == str(job_id), 'Slurm allocation job ID differs')
        # EndTime for a RUNNING job is a forecast, not an observed allocation end.
        terminal = fields.get('JobState', '').split('+')[0] in terminal_states
        return dict(**({'slurm_restart_count': int(fields['Restarts'])} if 'Restarts' in fields else {}),
                    allocation_started_at=stamp(fields.get('StartTime')),
                    allocation_ended_at=stamp(fields.get('EndTime')) if terminal else None)
    # scontrol drops finished jobs after MinJobAge; accounting retains history.
    try:
        result = subprocess.run(['sacct', '-D', '-j', str(job_id), '-X', '-n', '-P', '-o', 'JobIDRaw,Start,End,State'],
                                capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return {}
    records = []
    for line in result.stdout.splitlines():
        parts = line.strip().split('|')
        if len(parts) < 4: continue
        recorded_job, start, end, state = parts[:4]
        if recorded_job != str(job_id): continue
        started = stamp(start)
        if started is None or started != allocation_start: continue
        terminal = state.split()[0].split('+')[0] in terminal_states if state else False
        records.append(dict(allocation_started_at=started,
            allocation_ended_at=stamp(end) if terminal else None, source='sacct'))
    return max(records, key=lambda v: v['allocation_started_at']) if records else {}


def _restart_count():
    value = os.environ.get('SLURM_RESTART_COUNT', '0')
    require(value.isdecimal(), 'invalid Slurm restart count')
    return int(value)


def _check_restart_count(observed, restart):
    require(observed.get('slurm_restart_count', restart) == restart,
            'SLURM_RESTART_COUNT differs from scontrol Restarts; '
            'export SLURM_RESTART_COUNT from scontrol show job $SLURM_JOB_ID')


def _run_key(job_id, restart=0):
    require(bool(job_id) and all(ch.isalnum() or ch in '-_.' for ch in job_id),
            'invalid Slurm job ID')
    require(isinstance(restart, int) and restart >= 0, 'invalid Slurm restart count')
    return f'{job_id}-r{restart}'


def _allocation_path(out, job_id, restart=0):
    return out/'allocations'/(_run_key(job_id, restart)+'.json')


def _a4_start_path(out, job_id, restart=0):
    return out/'4a-starts'/_allocation_path(out, job_id, restart).name


def _a4_marker(out, job_id, standin, restart=0):
    path = _a4_start_path(out, job_id, restart)
    if not path.exists(): return None
    value = json.loads(path.read_text())
    require(value['slurm_job_id'] == job_id and value['slurm_restart_count'] == restart and
            value['technical_synthetic'] == standin, '4A start marker identity differs')
    require(math.isfinite(value['a4_started_at']) and value['a4_started_at'] <= time.time(),
            'invalid 4A start marker')
    return value


def _check_allocation_start(value):
    start = os.environ.get('SLURM_JOB_START_TIME')
    require(start is None or float(start) == value['allocation_started_at'],
            'SLURM_JOB_START_TIME differs from existing allocation start; '
            'check SLURM_RESTART_COUNT for a missing or reused requeue count')


def start_4a(out, standin, job_start=None):
    """Under node-writer, mark section entry once per Slurm run attempt."""
    from kit.message_handoff import atomic as durable_atomic, sync_dir
    job_id = os.environ.get('SLURM_JOB_ID')
    require(standin or job_id, 'SLURM_JOB_ID required for 4A start marker')
    job_id = job_id or 'standin'; restart = _restart_count()
    observed = {} if standin else _slurm_times(job_id)
    _check_restart_count(observed, restart)
    value = _a4_marker(out, job_id, standin, restart)
    if value is not None:
        _check_allocation_start(value)
        return value
    now = time.time()
    start = os.environ.get('SLURM_JOB_START_TIME')
    if start is not None: job_start = float(start)
    if job_start is None and not standin:
        job_start = observed.get('allocation_started_at')
    if job_start is None: job_start = now
    require(math.isfinite(job_start) and job_start <= now, 'invalid allocation start')
    # Any earlier run (another job, or an earlier run of this requeued job) started 4A.
    continuation = any((v['slurm_job_id'], v['slurm_restart_count']) != (job_id, restart)
                       for p in (out/'4a-starts').glob('*.json') for v in [json.loads(p.read_text())])
    value = dict(slurm_job_id=job_id, slurm_restart_count=restart, a4_started_at=now,
        a4_started_at_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        allocation_started_at=job_start, continuation=continuation,
        technical_synthetic=standin)
    durable_atomic(_a4_start_path(out, job_id, restart), value)
    sync_dir(out)  # Persist creation of the marker directory too.
    return value


def _a4_started_at(out, job_id, standin, restart=0):
    marker = _a4_marker(out, job_id, standin, restart)
    return None if marker is None else marker['a4_started_at']


def _set_a4_window(value, marker):
    value['a4_started_at'] = None if marker is None else marker['a4_started_at']
    value['a4_window_started_at'] = (None if marker is None else
        value['allocation_started_at'] if marker['continuation'] else marker['a4_started_at'])


def _allocation(out, doc, now):
    standin = doc['technical_synthetic']
    job_id = os.environ.get('SLURM_JOB_ID')
    require(standin or job_id, 'SLURM_JOB_ID required for held-allocation accounting')
    job_id = job_id or 'standin'; restart = _restart_count()
    marker = _a4_marker(out, job_id, standin, restart)
    path = _allocation_path(out, job_id, restart)
    if path.exists():
        value = json.loads(path.read_text())
        _check_allocation_start(value)
        require(value['technical_synthetic'] == standin, 'allocation synthetic mismatch')
        require(value['allocation_ended_at'] is None, 'current allocation already ended')
        if marker and value['start_source'] == 'first_4a_record':
            value.update(allocation_started_at=marker['allocation_started_at'], start_source='4a_start_marker')
    else:
        start = os.environ.get('SLURM_JOB_START_TIME')
        observed = {} if start or marker or standin else _slurm_times(job_id)
        start = float(start) if start else (marker['allocation_started_at'] if marker else
                                           observed.get('allocation_started_at'))
        value = dict(slurm_job_id=job_id, slurm_restart_count=restart,
            allocation_started_at=now if start is None else start,
            allocation_ended_at=None, start_source='SLURM_JOB_START_TIME' if os.environ.get('SLURM_JOB_START_TIME') else
                '4a_start_marker' if marker else observed.get('source', 'scontrol') if start is not None else 'first_4a_record',
            gpus=doc['allocation_gpus'], technical_synthetic=standin,
            approved_ceiling_gpu_hours=approved_ceiling(doc))
        require(value['allocation_started_at'] <= now, 'allocation start is in the future')
    _set_a4_window(value, marker)
    if marker is not None: value['last_4a_record_at'] = now
    atomic(path, value)
    return value


def _allocation_from_marker(out, doc, job_id, restart=0, persist=True):
    """Recover a run even when a hard kill left only its durable section marker."""
    path = _allocation_path(out, job_id, restart)
    if path.exists(): return json.loads(path.read_text())
    marker = _a4_marker(out, job_id, doc['technical_synthetic'], restart)
    require(marker is not None, 'allocation start record missing')
    value = dict(slurm_job_id=job_id, slurm_restart_count=restart,
        allocation_started_at=marker['allocation_started_at'],
        allocation_ended_at=None, start_source='4a_start_marker',
        last_4a_record_at=marker['a4_started_at'], gpus=doc['allocation_gpus'],
        technical_synthetic=doc['technical_synthetic'],
        approved_ceiling_gpu_hours=approved_ceiling(doc))
    _set_a4_window(value, marker)
    if persist: atomic(path, value)
    return value


def end_allocation(out, doc, job_id, ended, source='job_end_marker', restart=0, durable=None):
    """Persist an actual run-end marker (or recovered scheduler end), under node-writer."""
    path = _allocation_path(out, job_id, restart)
    value = _allocation_from_marker(out, doc, job_id, restart, persist=False)
    require(math.isfinite(ended) and value['allocation_started_at'] <= ended <= time.time(), 'invalid allocation end')
    require(value.get('last_4a_record_at') is None or ended >= value['last_4a_record_at'],
            'allocation end is before this run\'s last 4A record')
    folders = [out/'allocations', out/'4a-starts']
    durable = durable or os.environ.get('DURABLE_ROOT') or os.environ.get('DURABLE')
    if durable is not None: folders.append(Path(durable)/'job-starts')
    next_starts = [v['allocation_started_at'] for folder in folders
                   for p in folder.glob('*.json') for v in [json.loads(p.read_text())]
                   if v['slurm_job_id'] == job_id and v['slurm_restart_count'] > restart]
    require(not next_starts or ended <= min(next_starts),
            'allocation end exceeds a later run allocation start for Slurm job '+_run_key(job_id, restart))
    require(value['allocation_ended_at'] in (None, ended) or
            value.get('end_source') == 'last_4a_record_before_requeue', 'allocation end changed')
    value.update(allocation_ended_at=ended, end_source=source)
    atomic(path, value)


def _journal(out, attempt, ended, allocation, **kw):
    a4_start = allocation.get('a4_window_started_at')
    started = max(attempt['started_at'], allocation['allocation_started_at'],
                  a4_start if a4_start is not None else ended)
    ended = min(ended, allocation['allocation_ended_at'] or ended)
    seconds = max(0., ended-started)
    value = dict(attempt, seconds=seconds, ended_at=ended,
        allocation_started_at=allocation['allocation_started_at'],
        allocation_ended_at=allocation['allocation_ended_at'],
        a4_started_at=allocation.get('a4_started_at'), a4_window_started_at=a4_start,
        gpu_hours=0. if attempt['technical_synthetic'] else seconds*attempt['gpus']/3600, **kw)
    atomic(out/'costs'/(attempt['id']+'.json'), value)
    return value


def _account_allocations(out, doc, now, current_job):
    """Rebuild each job's held-time remainder; retries cannot double-charge intervals."""
    identities = {(v['slurm_job_id'], v['slurm_restart_count'])
                  for folder in ('allocations', '4a-starts')
                  for p in (out/folder).glob('*.json') for v in [json.loads(p.read_text())]}
    identities.add(current_job)
    for job_id, restart in sorted(identities):
        path = _allocation_path(out, job_id, restart)
        if not path.exists() and not _a4_start_path(out, job_id, restart).exists(): continue
        allocation = _allocation_from_marker(out, doc, job_id, restart)
        marker = _a4_marker(out, job_id, allocation['technical_synthetic'], restart)
        _set_a4_window(allocation, marker)
        a4_start = allocation['a4_window_started_at']
        current = (job_id, restart) == current_job
        attempts = [json.loads(p.read_text()) for p in sorted((out/'attempts').glob('*.json'))]
        attempts = [v for v in attempts if (v['slurm_job_id'], v['slurm_restart_count']) == (job_id, restart)]
        provisional = allocation.get('end_source') == 'last_4a_record_before_requeue'
        if a4_start is not None and not current and (allocation['allocation_ended_at'] is None or provisional):
            # Slurm's job record now describes the later run, not this ended run.
            requeued = any(j == job_id and r > restart for j, r in identities)
            observed = {} if requeued and not provisional else _slurm_times(job_id, allocation['allocation_started_at'])
            requeued = requeued or observed.get('slurm_restart_count', restart) > restart
            actual = observed.get('allocation_ended_at')
            if requeued and (observed.get('allocation_started_at') != allocation['allocation_started_at'] or
                             observed.get('slurm_restart_count', restart) != restart):
                actual = None
            if actual is not None:
                ended = actual
                source = observed.get('source', 'scontrol')
            elif requeued or provisional:
                ended = allocation['last_4a_record_at']
                for attempt in attempts:
                    journal = out/'costs'/(attempt['id']+'.json')
                    if not journal.exists() or json.loads(journal.read_text()).get('reconciled'):
                        ended = max(ended, attempt['started_at']+attempt['max_seconds'])
                next_starts = [_allocation_from_marker(out, doc, j, r)['allocation_started_at']
                               for j, r in identities if j == job_id and r > restart]
                if (observed.get('slurm_restart_count', restart) > restart and
                        observed.get('allocation_started_at') is not None):
                    next_starts.append(observed['allocation_started_at'])
                if not next_starts: raise NotReady('next allocation start required for Slurm job '+_run_key(job_id, restart))
                ended = min(ended, min(next_starts))
                source = 'last_4a_record_before_requeue'
            else:
                ended = None
            if ended is None: raise NotReady('actual allocation end required for Slurm job '+_run_key(job_id, restart))
            require(ended >= allocation['allocation_started_at'] and ended <= now,
                    'invalid recovered allocation end')
            allocation.update(allocation_ended_at=ended,
                              end_source=source)
        atomic(path, allocation)
        ended = allocation['allocation_ended_at']
        if ended is None:
            ended = now if current else allocation.get('last_4a_record_at', a4_start or now)
        charged = 0.
        for attempt in attempts:
            journal = out/'costs'/(attempt['id']+'.json')
            if not journal.exists():
                value = _journal(out, attempt, ended, allocation, completed=False, reconciled=True)
            else:
                value = json.loads(journal.read_text())
                # Refresh allocation-end provenance in completed attempt receipts too.
                if (value['allocation_ended_at'] != allocation['allocation_ended_at'] or
                        value.get('a4_window_started_at') != a4_start or a4_start is None):
                    value = _journal(out, attempt, ended if value.get('reconciled') else value['ended_at'], allocation,
                        **{k:v for k,v in value.items() if k not in attempt and k not in {
                            'seconds','ended_at','allocation_started_at','allocation_ended_at','a4_started_at','a4_window_started_at','gpu_hours'}})
            charged += value['seconds']
        held = 0. if a4_start is None else max(0., ended-max(allocation['allocation_started_at'], a4_start))
        require(charged <= held+1e-6, 'overlapping attempts exceed held allocation time')
        idle = max(0., held-charged)
        atomic(out/'costs'/('allocation-idle-'+_run_key(job_id, restart)+'.json'), dict(allocation,
            id='allocation-idle-'+_run_key(job_id, restart), kind='allocation_idle', seconds=idle,
            accounted_until=ended, held_seconds=held,
            gpu_hours=0. if allocation['technical_synthetic'] else idle*allocation['gpus']/3600))


def budget(out, doc):
    """Under node-writer: reconcile attempts against their own allocation, excluding queues."""
    approved_ceiling(doc)
    require(not (out/'allocation.json').exists(), 'legacy wall-clock accounting requires explicit reconciliation')
    now = time.time(); allocation = _allocation(out, doc, now)
    _account_allocations(out, doc, now, (allocation['slurm_job_id'], allocation['slurm_restart_count']))
    used = sum(json.loads(p.read_text()).get('gpu_hours', 0) or 0 for p in (out/'costs').glob('*.json'))
    require(used < approved_ceiling(doc), 'approved 4A GPU-hour ceiling exhausted')
    return used


def start_attempt(out, doc, kind, standin, **kw):
    job_id = os.environ.get('SLURM_JOB_ID', 'standin')
    require(_a4_started_at(out, job_id, standin, _restart_count()) is not None, '4A start marker required before attempt')
    used = budget(out, doc)
    limit = doc['handoff_seconds'] if kind == 'handoff' else doc['job_limits'][kind]
    gpus = doc['allocation_gpus']
    worst = gpus*(limit+(0 if kind == 'handoff' else doc['handoff_seconds']))/3600
    require(standin or used+worst <= approved_ceiling(doc),
        'remaining approved ceiling cannot cover worst-case '+kind+' and reserved handoff')
    value = dict(id=str(time.time_ns()), kind=kind, started_at=time.time(), gpus=gpus,
        slurm_job_id=job_id, slurm_restart_count=_restart_count(), approved_ceiling_gpu_hours=approved_ceiling(doc),
        max_seconds=limit, technical_synthetic=standin, **kw)
    atomic(out/'attempts'/(value['id']+'.json'), value)
    return value


def finish_attempt(out, attempt, **kw):
    path = _allocation_path(out, attempt['slurm_job_id'], attempt['slurm_restart_count'])
    allocation = json.loads(path.read_text())
    now = time.time()
    allocation['last_4a_record_at'] = now
    atomic(path, allocation)
    value = _journal(out, attempt, now, allocation, **kw)
    # Recompute idle on next budget; a crash between writes is repaired from job records.
    return value


from contextlib import contextmanager
import signal
import subprocess


@contextmanager
def job_deadline(seconds):
    """A job's worst case is an enforced limit, including model load and export."""
    def expired(signum,frame):raise CorrectnessStop('4A GPU job reached its hard duration limit')
    previous=signal.signal(signal.SIGALRM,expired)
    old_timer=signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,*old_timer)
        signal.signal(signal.SIGALRM,previous)


def gpu_run(command,check=False,**kwargs):
    """Terminate the entire trainer/launcher tree on timeout, error or parent TERM."""
    def term(signum,frame):raise SystemExit(128+signum)
    previous=signal.signal(signal.SIGTERM,term)
    process=None
    try:
        process=subprocess.Popen(command,start_new_session=True,**kwargs)
        code=process.wait()
        if check and code:raise subprocess.CalledProcessError(code,command)
        return subprocess.CompletedProcess(command,code)
    finally:
        if process is not None:
            # Launchers can exit while grandchildren still own GPUs.
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        signal.signal(signal.SIGTERM,previous)


def machine(out, gpu, standin):
    from kit.next_common import machine as pinned
    value = dict(pinned(out, gpu, standin), host=socket.gethostname())
    bind(out/'scoring-physical.json', value)
    return value
