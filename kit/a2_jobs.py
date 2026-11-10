#!/usr/bin/env python3
"""A2 inventories and allocation wrapper. The partner supplies sbatch; no submission."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
import subprocess
import time
import uuid
from contextlib import nullcontext
from kit.a2_common import *

# Captured partner site MaxTime=7-00:00:00 -> SITE_MINUTES 10080 (kit/README-phase0.md).
SITE_MAX_SECONDS = int(os.environ.get('SITE_MINUTES', '10080')) * 60


def inventory(arm, stage_seconds=14400, score_seconds=600, rewrite_counts=None):
    require(stage_seconds > 0 and score_seconds > 0, 'positive scheduling limits required')
    rows = []
    def add(name, phase, deps=(), gpus=0, **extra):
        hard = stage_seconds if phase == 'train' else score_seconds*extra.get('commands', 1)+60
        seconds = math.ceil((hard+max(900, .1*hard))/60)*60 if gpus else 14400
        row = dict(id='A2-'+arm+'-'+name, arm=arm, phase=phase, nodes=1, gpus=gpus,
                   cpus=32 if gpus == 8 else 8, may_share_node=gpus != 8,
                   time=f'{seconds//3600:02}:{seconds//60%60:02}:00',
                   time_limit_seconds=seconds, job_deadline_seconds=hard if gpus else None,
                   depends_on=[dict(id=d, type='afterany' if phase in ('read', 'confirmation-read') else 'afterok') for d in deps],
                   dependency_options=['--kill-on-invalid-dep=yes'],
                   command=['{job_python}', '-I', '{kit}/a2_jobs.py', 'run', '--config',
                            '{a2_config}', '--task', 'A2-'+arm+'-'+name],
                   gpu_hour_basis='allocated GPUs x actual allocation wall hours, including idle/failure',
                   **extra)
        rows.append(row)
        if seconds > 86400:
            # Informational: long serial lane (utilisation watch); submit if it fits the site MaxTime.
            row['long_lane_warning'] = ('Frozen complete-comparison single-allocation rule keeps this lane whole'
                if phase in ('score', 'confirmation-score') else
                'One registered training stage cannot be split without a qualified unchanged resume')
        if seconds > SITE_MAX_SECONDS:
            row['scheduling_conflict'] = 'Row ceiling exceeds the captured site MaxTime (SITE_MINUTES)'
        return row['id']
    previous = add('bind', 'bind')
    for n in range(1, 7):
        context = add(f'context-m{n}', 'context', [previous], month=n)
        prep = add(f'arrival-prepare-m{n}', 'score-prepare', [context], month=n, kind='arrival')
        arrival = add(f'arrival-m{n}', 'score', [prep], 1, month=n, kind='arrival',
                      commands=len(arm_names(arm))*len(SEEDS)*2+(3 if arm == 'J' else 0))
        feedback = add(f'feedback-m{n}', 'feedback', [arrival], month=n)
        if arm in ('R', 'RP'):
            rp = add(f'rewrite-prepare-m{n}', 'rewrite-prepare', [feedback], month=n)
            # Default inventories use the frozen 40/family/month supply ceiling.
            # Configured inventories count the actual, hash-bound admitted ledger.
            count = 80 if rewrite_counts is None else rewrite_counts[MONTHS[n-1]]
            batch_size = max(1, int((86400/1.1-60)//score_seconds))
            require(math.ceil((score_seconds+60+max(900, .1*(score_seconds+60)))/60)*60 <= 86400,
                    'one rewrite command cannot fit 24 h; qualified per-command timing needed')
            chunks = []
            for start in range(0, max(count, 1), batch_size):
                stop = min(start+batch_size, count)
                name = f'rewrite-m{n}' if count <= batch_size else f'rewrite-m{n}-b{len(chunks)+1}'
                chunks.append(add(name, 'rewrite', [rp], 1, month=n, commands=stop-start,
                                  trace_start=start, trace_stop=stop,
                                  count_basis='supply ceiling' if rewrite_counts is None else 'bound admitted ledger'))
            feedback = add(f'rewrite-finish-m{n}', 'rewrite-finish', chunks, month=n)
        finishes = []
        if n in BOUNDARIES:
            for model in arm_names(arm):
                for seed in SEEDS:
                    name = f'{model}-s{seed}-m{n}'
                    args = dict(model=model, seed=seed, month=n)
                    p = add(name+'-prepare', 'prepare', [feedback], **args)
                    train = add(name+'-train', 'train', [p], 8, **args)
                    finishes.append(add(name+'-export', 'export', [train], **args))
        previous = add(f'month-m{n}', 'month', [feedback, *finishes], month=n)
    from kit.a2_read import score_plan
    count = len(score_plan(arm))
    p = add('comparison-prepare', 'score-prepare', [previous], kind='comparison')
    comp = add('comparison', 'score', [p], 1, kind='comparison', commands=count)
    freeze_row = add('freeze-june', 'freeze-june', [comp])
    read = add('read', 'read', [freeze_row])
    add('collect', 'collect', [read])
    cb = add('confirmation-bind', 'confirmation-bind', [freeze_row])
    cp = add('confirmation-prepare', 'confirmation-prepare', [cb])
    from kit.a2_read import confirmation_plan
    cs = add('confirmation-score', 'confirmation-score', [cp], 1,
             commands=len(confirmation_plan(arm)))
    cr = add('confirmation-read', 'confirmation-read', [cs])
    add('confirmation-collect', 'confirmation-collect', [cr])
    # Reuse the first registered seed-101 stages, with their exact ancestors.
    # No export or later seed is needed to measure a complete native stage.
    by_id = {r['id']: r for r in rows}
    qualification = set()
    def qualify(task):
        if task in qualification:
            return
        qualification.add(task)
        for dep in by_id[task]['depends_on']:
            qualify(dep['id'])
    for model in arm_names(arm):
        qualify(f'A2-{arm}-{model}-s101-m2-train')
    price = add('price', 'price', [r['id'] for r in rows if r['id'] in qualification])
    price_row = rows.pop()
    for row in rows:
        row['qualification'] = row['id'] in qualification
        if not row['qualification']:
            row['depends_on'].append(dict(id=price, type='afterok'))
        if row['phase'] == 'collect':
            # A failed price still needs a CPU-only return of spend and price.
            row['depends_on'] = [dict(id=price, type='afterany'),
                                 *[dict(d, type='afterany') for d in row['depends_on'] if d['id'] != price]]
    rows = [r for r in rows if r['qualification']] + [price_row] + [r for r in rows if not r['qualification']]
    return dict(schema='a2-rows.v1', arm=arm, registration_sha256=REGISTRATION,
                training_stages=9*len(arm_names(arm)), rows=rows,
                confirmation='PENDING until separately governed December build/release; dormant rows',
                scheduling_conflicts=[r['id'] for r in rows if 'scheduling_conflict' in r],
                long_lanes=[r['id'] for r in rows if 'long_lane_warning' in r])


def configured_inventory(cfg):
    counts = None
    if cfg['arm'] in ('R', 'RP') and cfg.get('disposition') == 'active':
        from kit.a2_bind import receipt
        data = load(checked(cfg.get('inputs')))
        counts = {m: len(t.read_jsonl(receipt(data['ledger'][m])/'traces.jsonl')) for m in MONTHS}
    value = inventory(cfg['arm'], cfg.get('stage_max_seconds', 14400),
                      cfg.get('score_max_seconds', 600), counts)
    if cfg['arm'] in ('P', 'RP') and cfg.get('disposition') == 'active':
        from kit.a2_replay import applicable
        if not applicable(load(checked(cfg.get('i0')))):
            for row in value['rows']:
                if cfg['arm'] == 'P' or row.get('model') == 'P-replay':
                    row.update(gpus=0, cpus=8, may_share_node=True,
                               time='04:00:00', time_limit_seconds=14400, job_deadline_seconds=None)
    return value


def completed(out, task):
    pointer = Path(out)/'state'/(task+'.json')
    if not pointer.exists():
        raise NotReady('row absent: '+task)
    p = load(pointer)
    path = Path(out)/p['receipt']
    require(s.file_hash(path) == p['sha256'], 'row receipt changed')
    value = load(path)
    if value['status'] != 'COMPLETE':
        raise NotReady('row unfinished: '+task)
    for name, sha in value.get('artifacts', {}).items():
        artifact = Path(out)/name
        require(artifact.is_file() and s.file_hash(artifact) == sha, 'row artifact changed: '+name)
    return path.parent


def run(config, task):
    cfg = load(config)
    out = Path(cfg['out'])
    require(out.is_absolute(), 'absolute durable A2 out required')
    rows = configured_inventory(cfg)['rows']
    matches = [r for r in rows if r['id'] == task]
    require(len(matches) == 1, 'unknown A2 row')
    row = matches[0]
    synthetic = cfg.get('technical_synthetic', False)
    # This branch touches no node inputs, models, or scientific scores.
    if cfg.get('disposition') == 'archived':
        atomic(out/'reading.json', dict(arm=cfg['arm'], status='ARCHIVED UNREAD',
                                       registration_sha256=REGISTRATION))
        if row['phase'] in ('collect', 'confirmation-collect'):
            from kit.a2_return import disposition_return
            disposition_return(cfg, 'ARCHIVED UNREAD', row['phase'] == 'confirmation-collect')
        print('ARCHIVED UNREAD')
        return 0
    if cfg.get('disposition') != 'active':
        raise NotReady('branch dispatch unresolved')
    if (out/'binding.json').exists():
        bound = load(out/'binding.json')
        if cfg['arm'] == 'P' and bound.get('P_applicable') is False:
            atomic(out/'reading.json', dict(arm='P', status='NOT APPLICABLE', comparisons=0))
            if row['phase'] in ('collect', 'confirmation-collect'):
                from kit.a2_return import disposition_return
                disposition_return(cfg, 'NOT APPLICABLE', row['phase'] == 'confirmation-collect')
            print('NOT APPLICABLE: all three I0 maps have zero generic positions')
            return 0
        if cfg['arm'] == 'RP' and bound.get('P_applicable') is False and row.get('model') == 'P-replay':
            row = dict(row, gpus=0, job_deadline_seconds=None)
    with lock(out/'row-locks'/task):
        prior = None
        try:
            prior = completed(out, task)
        except NotReady:
            pass
        from kit.a2_budget import sealed_config
        bind(out/'config.json', sealed_config(cfg))
        attempt = out/'attempts'/task/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        started = time.time()
        value = dict(task=task, phase=row['phase'], gpus=row['gpus'], nonce=attempt.name,
                     status='RUNNING', technical_synthetic=synthetic, started_at=started,
                     slurm_job_id=os.environ.get('SLURM_JOB_ID'), allocation_gpu_hours=0 if synthetic else None)
        atomic(attempt/'receipt.json', value)
        code = 2
        try:
            from kit.a2_budget import approval, gate, reserve
            if synthetic and row['gpus']:
                value['allocation_started_at'] = started
                reserve(cfg, out, value, row)
            from kit.a4_jobs import reservation, visible_devices
            from kit.a4_common import job_deadline, _slurm_times, gpu_run
            if not synthetic:
                job = os.environ.get('SLURM_JOB_ID')
                require(job, 'one scheduler allocation per real row required')
                start = os.environ.get('SLURM_JOB_START_TIME') or _slurm_times(job).get('allocation_started_at')
                require(start is not None, 'actual scheduler start absent')
                value['allocation_started_at'] = float(start)
                if not row['gpus']:
                    from kit.a2_budget import reconcile
                    reconcile(out, False)
                for receipt in (out/'attempts').glob('*/*/receipt.json'):
                    other = load(receipt)
                    require(receipt.parent == attempt or (other.get('slurm_job_id'), other.get('allocation_started_at')) !=
                            (job, float(start)), 'one row per allocation; use a new scheduler job')
                if row['gpus']:
                    from kit.a2_budget import reserve
                    reserve(cfg, out, value, row)
                atomic(attempt/'receipt.json', value)
            approval(cfg)
            from kit.m3_manifest import allocated_gpus
            if not row.get('qualification') and row['phase'] not in ('price', 'collect'):
                gate(cfg, out)
            for dep in row['depends_on']:
                if dep['type'] == 'afterok':
                    completed(out, dep['id'])
            if prior is not None:
                # An idempotent invocation still consumed its own allocation.
                code = 0
                value['status'] = 'COMPLETE'
                return code
            ids = (['GPU-CPU-STANDIN-'+str(i) for i in range(row['gpus'])] if synthetic
                   else allocated_gpus(row['gpus'])) if row['gpus'] else []
            value['allocated_uuids'] = ids
            atomic(attempt/'receipt.json', value)
            with reservation(ids, synthetic, attempt) as fds:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible_devices(ids, synthetic),
                           M3_GPU_LOCK_FDS=json.dumps(fds), A2_ALLOCATION_NONCE=attempt.name,
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                require(not os.environ.get('UTIL_WATCH_ACTIVE'), 'run A2 wrapper outside a watcher')
                python = cfg.get('inference_python' if row['phase'] in ('score', 'rewrite', 'confirmation-score') else 'trainer_python', sys.executable)
                command = [cfg.get('job_python', sys.executable), '-I', str(KIT/'util_watch.py'),
                           '--out', str(attempt/'util.jsonl'), '--expect-gpus', str(row['gpus']), '--',
                           python, '-I', str(KIT/'a2_execute.py'), '--config', str(config),
                           '--task', task, '--attempt', str(attempt)]
                with job_deadline(row['job_deadline_seconds']) if row['gpus'] else nullcontext():
                    code = gpu_run(command, env=env, pass_fds=fds).returncode
            value['status'] = 'COMPLETE' if code == 0 else 'PENDING' if code == PENDING else 'FAILED'
        except (NotReady, FileNotFoundError) as error:
            value.update(status='PENDING', error=str(error))
            code = PENDING
            print('PENDING:', error)
        except Exception as error:
            value.update(status='FAILED', error=str(error))
            code = 2
            print('STOP:', type(error).__name__, error)
        finally:
            value['ended_at'] = time.time()
            if synthetic or not row['gpus']:
                value['allocation_ended_at'] = value['ended_at']
            output = attempt/'output.json'
            names = load(output).get('artifacts', []) if output.exists() and code == 0 else []
            value['artifacts'] = {str(Path(name).relative_to(out)): s.file_hash(name) for name in names}
            atomic(attempt/'receipt.json', value)
            if prior is None:
                atomic(out/'state'/(task+'.json'), dict(receipt=str((attempt/'receipt.json').relative_to(out)),
                                                     sha256=s.file_hash(attempt/'receipt.json')))
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('rows', 'run'))
    p.add_argument('--arm', choices=('J', 'P', 'R', 'RP'))
    p.add_argument('--config', type=Path)
    p.add_argument('--task')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    try:
        if a.action == 'rows':
            cfg = load(a.config) if a.config else {}
            atomic(a.out, configured_inventory(cfg) if cfg else inventory(a.arm))
            return 0
        return run(a.config, a.task)
    except (NotReady, FileNotFoundError) as error:
        print('PENDING:', error)
        return PENDING
    except CorrectnessStop as error:
        print('STOP:', error)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
