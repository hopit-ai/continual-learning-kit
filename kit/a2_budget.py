"""A2 cost forecasts, reservations and actual allocated-time reconciliation."""
from kit.a2_common import *
import time

NO_ALLOCATION = 'no allocation started: no scheduler job ID or start time'


def forecast(arm, d, challenger=None, *, rewrite_stage=None, collection=0, preparation=0,
             rewrite=0, confirmation=None, score_cell_hours=.1, scoring_load_failure=5):
    require(arm in ('J', 'P', 'R', 'RP'), 'unknown A2 forecast')
    challenger = d if challenger is None else challenger
    training = (9*(d+challenger+(d if rewrite_stage is None else rewrite_stage)) if arm == 'RP'
                else 9*d+9*challenger)
    additional = collection+preparation+rewrite
    from kit.a2_read import score_plan, confirmation_plan
    exploratory_cells = len(score_plan(arm))+6*len(arm_names(arm))*len(SEEDS)*2+(18 if arm == 'J' else 0)
    confirmation_cells = len(confirmation_plan(arm))
    exploratory_scoring = max(60, exploratory_cells*score_cell_hours+scoring_load_failure)
    if confirmation is None:
        confirmation = confirmation_cells*score_cell_hours+scoring_load_failure
    allowance = 50*math.ceil((1.5*training+exploratory_scoring+80+additional)/50)
    return dict(arm=arm, training_gpu_hours=training, allowance_before_confirmation=allowance,
                confirmation_gpu_hours=confirmation, total_gpu_hours=allowance+confirmation,
                components=dict(collection=collection, preparation=preparation, rewrite=rewrite,
                                scoring_reserve=exploratory_scoring, inherited_scoring_floor=60,
                                exploratory_cells=exploratory_cells, confirmation_cells=confirmation_cells,
                                score_cell_gpu_hours=score_cell_hours, scoring_load_failure=scoring_load_failure,
                                overhead_failures=80),
                status='PROVISIONAL; reprice from qualification receipts before continuation')


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=s.file_hash(path))


def approval(cfg):
    budget = cfg.get('budget')
    if not budget or not budget.get('approval_reference'):
        raise NotReady('A2 execution/spending approval absent; build approval is separate')
    ceiling = budget.get('approved_gpu_hours')
    require(type(ceiling) in (int, float) and math.isfinite(ceiling) and 0 < ceiling <= 601,
            'invalid approved A2 budget; single selected-arm cap must be <= 601 GPU-h')
    return ceiling


def sealed_config(cfg):
    return {k: v for k, v in cfg.items() if k not in ('allow_confirmation', 'confirmation')}


def timing_setup(cfg, doc, row, prepared):
    """Compare effective settings, stack and hardware, excluding only stage paths.

    Reused 4A evidence must carry this same descriptor plus its original
    effective configuration and allocation records, all hash-bound.
    """
    if doc['technical_synthetic']:
        hardware = ['TECHNICAL-ONLY']*8
    else:
        import torch
        hardware = [dict(name=torch.cuda.get_device_properties(i).name,
                         memory=torch.cuda.get_device_properties(i).total_memory,
                         capability=list(torch.cuda.get_device_capability(i))) for i in range(8)]
    from kit.a2_pipeline import stage
    run = stage(cfg['out'], row)
    substitutions = [(str(Path(cfg['out'])/'data'/run.name), '{data}'),
                     (str(run), '{run}'), (prepared['incoming']['model'], '{model}'),
                     (str(KIT), '{kit}')]
    for key in ('sdpo', 'tokenizer', 'trainer_python'):
        if cfg.get(key):
            substitutions.append((cfg[key], '{'+key+'}'))
    command = []
    for arg in prepared.get('command', []):
        for old, new in sorted(substitutions, key=lambda pair: -len(pair[0])):
            arg = arg.replace(old, new)
        command.append(arg)
    return dict(recipe=prepared['recipe'], updates=40, batch=32, lr=1e-5,
                warmup=10, fresh_optimizer=True, incoming_student_ema=True,
                model_revision=REVISION, software=doc['software']['trainer'],
                hardware=hardware, effective_command=command,
                inherited_recipe=protocol()['a4'], technical_synthetic=doc['technical_synthetic'])


def validate_timing(ref, setup, synthetic, *, external=False):
    measured = load(checked(ref))
    require(measured.get('qualified_8b') is True and measured.get('gpus') == 8,
            'qualified measured 8-GPU stage timing required')
    require(measured.get('technical_synthetic', False) is synthetic,
            'timing synthetic identity differs')
    require(measured.get('setup') == setup, 'timing recipe/setup differs from A2 D stage')
    seconds = measured.get('seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
            'positive measured timing seconds required')
    source = load(checked(measured.get('source')))
    require(source.get('status') == 'COMPLETE' and source.get('gpus') == 8,
            'timing source is not a complete 8-GPU allocation')
    require(source.get('technical_synthetic', False) is synthetic, 'timing source synthetic identity differs')
    attempts = [load(checked(r)) for r in measured.get('attempts', [measured['source']])]
    require(attempts and source in attempts, 'complete timing source absent from measured attempts')
    elapsed = 0
    for attempt in attempts:
        require(attempt.get('gpus') == 8 and attempt.get('technical_synthetic', False) is synthetic and
                attempt.get('task') == source.get('task'), 'timing attempt setup/task differs')
        if attempt.get('allocation_reconciliation_reason') == NO_ALLOCATION:
            require(attempt.get('allocation_started_at') is None and not attempt.get('slurm_job_id') and
                    attempt.get('allocation_gpu_hours') == 0, 'invalid zero-allocation timing attempt')
            continue
        if synthetic:
            duration = attempt['ended_at']-attempt['started_at']
        else:
            require(attempt.get('allocation_ended_at') is not None and
                    attempt.get('allocation_started_at') is not None,
                    'timing actual allocation ends absent')
            duration = attempt['allocation_ended_at']-attempt['allocation_started_at']
            require(abs(8*duration/3600-attempt.get('allocation_gpu_hours', -1)) < 1e-6,
                    'timing differs from actual allocation ends')
        require(duration > 0, 'timing allocation duration must be positive')
        elapsed += duration
    require(abs(seconds-elapsed) < 1e-6, 'timing differs from bound allocation attempts')
    evidence = load(checked(measured.get('effective_setup')))
    require(evidence == setup, 'timing effective setup changed')
    checked(measured.get('prepared'))
    if external:
        require(measured.get('origin') == '4A' and setup['recipe'] == 'D',
                'only matching existing 4A D receipts may be reused')
        # Retain original native effective config and schedule, never just flags.
        require(load(checked(measured.get('effective_config'))) == setup,
                '4A effective configuration differs from A2 setup')
        schedule = t.read_jsonl(checked(measured.get('schedule')))
        require(len(schedule) == 1280 and not any(r.get('p3b_replay') for r in schedule),
                '4A D schedule/dose differs from A2')
    return measured


def timing_attempts(out, row):
    from kit.a2_jobs import completed
    completion = load(completed(out, row['id'])/'receipt.json')
    # Snapshot through this completion. Later idempotent allocations are still
    # charged, but cannot change an already sealed qualification measurement.
    return sorted(p for p in (Path(out)/'attempts'/row['id']).glob('*/receipt.json')
                  if load(p)['started_at'] <= completion['started_at'])


def measured_row(out, row):
    from kit.a2_jobs import completed
    root = completed(out, row['id'])
    value = load(root/'receipt.json')
    synthetic = value['technical_synthetic']
    # A recovery may only export an already completed native stage. Include
    # all first-stage attempts so that recovery cannot qualify a near-zero rate.
    attempts = [load(p) for p in timing_attempts(out, row)] if row['phase'] == 'train' else [value]
    seconds = sum(0 if a.get('allocation_reconciliation_reason') == NO_ALLOCATION else
                  a['ended_at']-a['started_at'] if synthetic else
                  a['allocation_gpu_hours']*3600/row['gpus'] for a in attempts)
    require(math.isfinite(seconds) and seconds > 0, 'positive completed allocation timing required')
    return root, value, seconds


def timing_models(cfg):
    names = arm_names(cfg['arm'])
    if cfg['arm'] == 'RP':
        from kit.a2_replay import applicable
        if not applicable(load(checked(cfg['i0']))):
            return tuple(m for m in names if m != 'P-replay')
    return names


def conservative_d_timing(a2_ref, four_a_ref, setup, synthetic):
    a2 = validate_timing(a2_ref, setup, synthetic)
    four_a = validate_timing(four_a_ref, setup, synthetic, external=True)
    selected = '4A' if four_a['seconds'] >= a2['seconds'] else 'A2'
    choice = dict(A2=dict(ref=a2_ref, seconds=a2['seconds']),
                  **{'4A': dict(ref=four_a_ref, seconds=four_a['seconds'])},
                  selected=selected, seconds=max(a2['seconds'], four_a['seconds']))
    return (four_a if selected == '4A' else a2), choice


def compute_price(cfg, out, rows, timings, observations, spend):
    """Full study expenditure, including dormant confirmation, with no RP duplication."""
    names = arm_names(cfg['arm'])
    rates = {m: timings[m]['seconds']*8/3600 if m in timings else 0 for m in names}
    score_rate = max(o['seconds']/o['commands']/3600 for o in observations if o['phase'] == 'score')
    rewrites = [o['seconds']/o['commands']/3600 for o in observations
                if o['phase'] == 'rewrite' and o['commands']]
    rewrite = sum(r.get('commands', 0) for r in rows if r['phase'] == 'rewrite')*max(rewrites, default=0)
    value = forecast(cfg['arm'], rates[names[0]], rates[names[1]],
                     rewrite_stage=rates[names[-1]], rewrite=rewrite,
                     score_cell_hours=score_rate)
    # Qualification is registered study work, not an extra stage. Never price
    # below spend already incurred plus the measured baseline of remaining work.
    baseline = value['training_gpu_hours'] + value['components']['exploratory_cells']*score_rate + rewrite + value['confirmation_gpu_hours']
    successful = sum(o['gpus']*o['seconds']/3600 for o in observations)
    value['total_gpu_hours'] = max(value['total_gpu_hours'], spend+max(0, baseline-successful))
    value['components'].update(preparation=0, exports_transfers=0,
                              loading='included in full allocation per-command measurements plus 5 per scoring phase',
                              failures_retries='1.5 x training and 80 GPU-h overhead; all attempts charged',
                              collection_cells=18 if cfg['arm'] == 'J' else 0,
                              collection_gpu_hours_in_scoring=18*score_rate if cfg['arm'] == 'J' else 0,
                              rp_shared_control_stages=9 if cfg['arm'] == 'RP' else 0)
    value['standalone_training_gpu_hours'] = {m: 9*(rates[names[0]]+rates[m]) for m in names[1:] if m in timings}
    value['status'] = 'MEASURED TECHNICAL-ONLY' if cfg.get('technical_synthetic', False) else 'MEASURED'
    return value


def price(cfg, out):
    from kit.a2_jobs import configured_inventory, completed
    ceiling = approval(cfg)
    rows = configured_inventory(cfg)['rows']
    accounting = reconcile(out, cfg.get('technical_synthetic', False))
    if accounting['status'] != 'COMPLETE':
        raise NotReady('qualification actual allocation ends absent; retry CPU price row')
    directory = Path(out)/'qualification'
    timings, refs, observations, artifacts = {}, {}, [], []
    synthetic = cfg.get('technical_synthetic', False)
    for row in rows:
        if not row.get('qualification') or not row['gpus']:
            continue
        root, receipt, seconds = measured_row(out, row)
        observations.append(dict(task=row['id'], phase=row['phase'], gpus=row['gpus'],
                                 commands=row.get('commands', 1), seconds=seconds,
                                 source=reference(root/'receipt.json')))
        if row['phase'] != 'train':
            continue
        from kit.a2_pipeline import stage
        data = Path(out)/'data'/stage(out, row).name
        prepared = load(data/'prepared.json')
        require(not prepared['empty'], 'empty/skipped stage cannot qualify timing')
        setup = load(root/'timing-setup.json')
        measured = dict(schema='a2-stage-timing.v1', qualified_8b=True, gpus=8,
                        technical_synthetic=synthetic, model=row['model'], seconds=seconds,
                        setup=setup, source=reference(root/'receipt.json'),
                        attempts=[reference(p) for p in timing_attempts(out, row)],
                        effective_setup=reference(root/'timing-setup.json'), prepared=reference(data/'manifest.json'))
        path = directory/(row['model']+'-timing.json')
        bind(path, measured)
        refs[row['model']] = reference(path)
        timings[row['model']] = validate_timing(refs[row['model']], setup, synthetic)
        artifacts.append(path)
    control = arm_names(cfg['arm'])[0]
    d_choice = None
    if cfg['budget'].get('D_timing'):
        timings[control], d_choice = conservative_d_timing(
            refs[control], cfg['budget']['D_timing'], timings[control]['setup'], synthetic)
        refs[control] = d_choice[d_choice['selected']]['ref']
    # Challenger timing is always the first real registered stage of THIS arm.
    require(set(timings) == set(timing_models(cfg)), 'selected-arm timing coverage incomplete')
    value = compute_price(cfg, out, rows, timings, observations, accounting['gpu_hours'])
    if d_choice is not None:
        value['D_timing_choice'] = d_choice
    value.update(schema='a2-price.v1', config_sha256=digest(sealed_config(cfg)),
                 protocol_sha256=digest(protocol()), registration_sha256=REGISTRATION,
                 timing_refs=refs, observations=observations, qualification_gpu_hours=accounting['gpu_hours'],
                 approved_gpu_hours=ceiling, approval_reference=cfg['budget']['approval_reference'],
                 decision='PROCEED' if value['total_gpu_hours'] <= ceiling else 'OWNER ESCALATION')
    # Reconcile may be retried before a successful price row; completed rows are immutable.
    atomic(directory/'price.json', value)
    atomic(directory/'price-ref.json', reference(directory/'price.json'))
    return [*artifacts, directory/'price.json', directory/'price-ref.json']


def gate(cfg, out):
    from kit.a2_jobs import completed, configured_inventory
    ceiling = approval(cfg)
    root = completed(out, 'A2-'+cfg['arm']+'-price')
    value = load(checked(load(Path(out)/'qualification/price-ref.json')))
    require(value.get('config_sha256') == digest(sealed_config(cfg)) and
            value.get('protocol_sha256') == digest(protocol()) and
            value.get('registration_sha256') == REGISTRATION and
            value.get('approved_gpu_hours') == ceiling and
            value.get('approval_reference') == cfg['budget']['approval_reference'],
            'price config/protocol/approval binding differs')
    rows = configured_inventory(cfg)['rows']
    timings = {}
    for model in timing_models(cfg):
        row = next(r for r in rows if r.get('model') == model and r.get('qualification') and r['phase'] == 'train')
        stage_root = completed(out, row['id'])
        setup = load(stage_root/'timing-setup.json')
        if model == arm_names(cfg['arm'])[0] and cfg['budget'].get('D_timing'):
            timings[model], choice = conservative_d_timing(
                reference(Path(out)/'qualification'/(model+'-timing.json')),
                cfg['budget']['D_timing'], setup, cfg.get('technical_synthetic', False))
            require(value.get('D_timing_choice') == choice and
                    value['timing_refs'][model] == choice[choice['selected']]['ref'],
                    'price D timing choice differs from conservative measured timings')
        else:
            timings[model] = validate_timing(value['timing_refs'][model], setup, cfg.get('technical_synthetic', False))
    expected_observations = []
    for row in rows:
        if not row.get('qualification') or not row['gpus']:
            continue
        source, _, seconds = measured_row(out, row)
        expected_observations.append(dict(task=row['id'], phase=row['phase'], gpus=row['gpus'],
                                          commands=row.get('commands', 1), seconds=seconds,
                                          source=reference(source/'receipt.json')))
    require(value['observations'] == expected_observations, 'price observations differ from measured allocations')
    recomputed = compute_price(cfg, out, rows, timings, value['observations'], value['qualification_gpu_hours'])
    require(all(value.get(k) == v for k, v in recomputed.items()),
            'price arithmetic differs from measured full-arm price')
    if value['total_gpu_hours'] > ceiling:
        raise NotReady(f"A2 owner escalation: measured full selected-arm price {value['total_gpu_hours']:.6g} GPU-h "
                       f"exceeds approved {ceiling:g}; qualification spend {value['qualification_gpu_hours']:.6g} GPU-h; all later work stopped")
    return value


def reserve(cfg, out, record, row):
    ceiling = approval(cfg)
    if not row.get('qualification'):
        gate(cfg, out)
    with lock(Path(out)/'budget'):
        used = 0
        for p in (Path(out)/'attempts').glob('*/*/receipt.json'):
            other = load(p)
            if other.get('nonce') != record['nonce'] and other.get('gpus'):
                if other.get('allocation_gpu_hours') is not None:
                    used += other['allocation_gpu_hours']
                else:
                    elapsed = max(0, time.time()-other.get('allocation_started_at', time.time()))
                    used += max(other.get('reserved_gpu_hours', 0), other['gpus']*elapsed/3600)
        reserved = row['gpus']*(row['job_deadline_seconds']+time.time()-record['allocation_started_at']+30)/3600
        if used+reserved > ceiling:
            raise NotReady('remaining approved A2 allowance cannot cover allocation; owner escalation')
        record['reserved_gpu_hours'] = reserved
        # Publish while holding the same lock, so concurrent rows see the reservation.
        path = Path(out)/'attempts'/row['id']/record['nonce']/'receipt.json'
        if path.exists():
            atomic(path, record)


def reconcile(out, synthetic, query=None):
    from kit.a4_common import _slurm_times
    query = query or _slurm_times
    rows, pending = [], []
    for path in sorted((Path(out)/'attempts').glob('*/*/receipt.json')):
        value = load(path)
        if not value['gpus']:
            continue
        hours = value.get('allocation_gpu_hours')
        if hours is None:
            start = value.get('allocation_started_at')
            if start is None and not value.get('slurm_job_id'):
                hours = 0
                value.update(allocation_gpu_hours=hours,
                             allocation_reconciliation_reason=NO_ALLOCATION)
            else:
                observed = query(value['slurm_job_id'], start)
                start = start if start is not None else observed.get('allocation_started_at')
                end = observed.get('allocation_ended_at')
                if start is None or end is None:
                    pending.append(value['nonce'])
                    continue
                require(end >= start, 'actual scheduler end precedes start')
                hours = value['gpus']*(end-start)/3600
                value.update(allocation_started_at=start, allocation_ended_at=end, allocation_gpu_hours=hours)
            atomic(path, value)
            pointer = Path(out)/'state'/(value['task']+'.json')
            if pointer.exists() and load(pointer)['receipt'] == str(path.relative_to(out)):
                atomic(pointer, dict(receipt=str(path.relative_to(out)), sha256=s.file_hash(path)))
        rows.append(dict(task=value['task'], phase=value['phase'], nonce=value['nonce'],
                         gpu_hours=hours, status=value['status'],
                         **({'reason': value['allocation_reconciliation_reason']}
                            if 'allocation_reconciliation_reason' in value else {})))
    return dict(status='PENDING' if pending else 'COMPLETE', gpu_hours=sum(r['gpu_hours'] for r in rows),
                rows=rows, pending=pending, queue_gpu_hours=0, utilisation='warn-only')
