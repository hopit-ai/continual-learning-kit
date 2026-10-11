#!/usr/bin/env python3
"""Manager-side CPU ingest of a phase-0 return for the automatic main flow.

No archived kit code or supplied diff is executed. The real reader imports its
hash-pinned scorer. Reports survive refusals. Output
must be a new directory, distinct from the read-only inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GIB = 1024**3
DISK_GIB = 1879
DEFERRED = 'DEFERRED TO QUALIFICATION'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(quantity, measured, extrapolated, cap, unit, source, *, note=''):
    """Margin is cap / projected requirement; future-stage evidence is deferred.

    Mandatory phase-0 evidence is enforced separately by its real reader/C1
    hook. Absence of later-stage measurements cannot become a tag refusal.
    """
    from kit.v4_budget import number
    cap = float(number(cap, quantity+' cap', positive=True))
    missing = measured is None or extrapolated is None
    if missing:
        margin = None
        status = DEFERRED
    else:
        measured = float(number(measured, quantity+' measured'))
        extrapolated = float(number(extrapolated, quantity+' extrapolated'))
        margin = cap/extrapolated if extrapolated else None
        status = 'FAIL' if extrapolated > cap else 'TIGHT' if extrapolated > 0 and margin < 1.25 else 'PASS'
    return dict(quantity=quantity, measured=measured, extrapolated=extrapolated,
                cap=cap, margin=margin, unit=unit, status=status, missing=missing,
                source=source, note=note)


def qualification_fit(graph, spent, blocks, *, requested_minutes=720, site_minutes=None):
    """The allocator's whole-minute clip, plus a serial cap-consumption trace.

    SITE_MINUTES comes from the archive when captured; otherwise show budget
    clipping alone. Rows-at-risk assume all preceding outer caps are consumed,
    then apply the real runner's inner GPU / outer CPU start-admission cap.
    Faster rows can fit. Unmeasured dispatch/startup can worsen this bound.
    """
    from kit.v4_budget import number, gpu_hours
    spent = number(spent, 'actual carry')
    qualification = number(blocks.get('qualification', 0), 'qualification carry')
    remaining = max(number(0, 'zero'), 1000-spent)
    budget_minutes = int(remaining*60/8)
    if type(requested_minutes) is not int or requested_minutes <= 0:
        raise ValueError('qualification request must be positive whole minutes')
    if site_minutes is not None and (type(site_minutes) is not int or site_minutes <= 0):
        raise ValueError('captured site limit must be positive whole minutes')
    limit = site_minutes if site_minutes is not None else requested_minutes
    allowed = min(requested_minutes, limit)
    seconds = sum(number(r['timeout_seconds'], 'row timeout') for r in graph['rows'])
    envelope = gpu_hours(8, seconds)
    required_minutes = math.ceil(seconds/60)
    full_allowed = min(required_minutes,
                       site_minutes if site_minutes is not None else required_minutes)
    trace = []
    elapsed = 0
    for row in graph['rows']:
        cap = row_cap(row)
        available = allowed*60-elapsed
        if available < cap:
            trace.append(dict(row=row['id'], starts_at_worst_case_seconds=elapsed,
                              available_seconds=available, admission_cap_seconds=cap))
        elapsed += row['timeout_seconds']
    return dict(status='WARNING' if qualification+envelope > 100 or spent+envelope > 1000 else 'PASS',
                requested_minutes=requested_minutes, site_minutes=site_minutes,
                site_limit_known=site_minutes is not None, budget_max_minutes=budget_minutes,
                allowed_minutes=allowed, allocator_refuses=spent>=1000,
                clipped_minutes=requested_minutes-allowed,
                clipped_gpu_hours=float(gpu_hours(8, (requested_minutes-allowed)*60)),
                required_seconds=float(seconds), required_minutes=required_minutes,
                full_envelope_allowed_minutes=full_allowed,
                full_envelope_clipped_minutes=required_minutes-full_allowed,
                envelope_shortfall_seconds=float(max(0, seconds-allowed*60)),
                qualification_gpu_hours=float(qualification+envelope),
                remaining_qualification_gpu_hours=float(1000-spent),
                rows_at_risk=trace,
                trace_assumption='Each preceding outer row cap fully consumed; row deadline is a generous wall-clock backstop. '
                                 'Extra startup/dispatch excluded; rows at risk are warnings, never projection refusals.')


def row_cap(row):
    argv = row['command']
    return float(argv[argv.index('--time-cap')+1]) if '--time-cap' in argv else float(row['timeout_seconds'])


def diff_record(path, out):
    if path is None:
        return None
    raw = Path(path).read_bytes()
    touched = set()
    for line in raw.decode('utf-8', errors='replace').splitlines():
        if line.startswith('diff --git '):
            names = shlex.split(line[len('diff --git '):])
            if len(names) != 2:
                raise ValueError('malformed git diff header')
            for name in names:
                touched.add(name[2:] if name.startswith(('a/', 'b/')) else name)
        elif line.startswith(('rename from ', 'rename to ', 'copy from ', 'copy to ')):
            touched.add(line.split(' ', 2)[2])
    (out/'local.diff').write_bytes(raw)
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                files_touched=sorted(touched), archived_as='local.diff',
                contents_executed=False)


def measurements(archive, reader, main, qualification):
    """Keep serial, four-attempt bounds; a 20-question sample proves no full wave.

    Include every archived wall-time receipt, even those without a corresponding
    main operation. Such observations remain visible as missing-cap evidence.
    """
    from kit.v4_budget import number
    from kit.v4_prepare import main_verification_allowance as verification_allowance
    from kit.v4_timing import sample
    base = 'v4/report-phase0/'
    rows = {r['id']:r for r in main['rows']}
    qrows = {r['id']:r for r in qualification['rows']}
    ledger = archive.json('k8b4/containment/allocation-ledger.json')
    comparisons, timings, inventory = [], {}, []

    def add(key, measured, projected, cap, source, unit='seconds', note=''):
        comparisons.append(comparison(key, measured, projected, cap, unit, source, note=note))

    def observe(key, record, units=1):
        timings.setdefault(key, []).append(sample(ledger, record, units))

    def walk(value, source):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('wall_seconds', 'seconds', 'import_seconds', 'peak_allocated_bytes',
                           'peak_reserved_bytes', 'free_bytes', 'used_bytes') and isinstance(item, (int, float)):
                    inventory.append(dict(source=source+'/'+key, value=item, operation_name=value.get('name')))
                else:
                    walk(item, source+'/'+key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, source+'/'+str(i))

    for name in sorted(archive.files):
        observation_source = name.startswith((base, 'runs/q-', 'campaign/v4-phase0/', 'k8b4/containment/')) or \
            name.startswith('v4/report-inputs/import-smoke-phase0/') or name=='v4/report-inputs/prepare-receipt-phase0.json'
        if observation_source and name.endswith(('.json', '.jsonl')):
            walk(archive.jsonl(name) if name.endswith('.jsonl') else archive.json(name), name)

    receipt_path = 'v4/report-inputs/prepare-receipt-phase0.json'
    receipt = archive.json(receipt_path)
    environment = receipt['environment_check']
    add('environment-check-deadline', max(environment['import_seconds'].values()),
        environment['deadline_seconds'], environment['deadline_seconds'], receipt_path,
        note='v7 CPU-derived generous deadline; elapsed imports are recorded measurements.')
    for role, measured in environment['import_seconds'].items():
        add('environment-import:'+role, measured, measured, environment['deadline_seconds'], receipt_path)
    measured = receipt['verification_seconds']
    allowance = verification_allowance(measured)
    for phase, cap in (('qualification', row_cap(qrows['prepare'])), ('main', row_cap(rows['prepare']))):
        add('prepared-byte-verification:'+phase, measured, allowance, cap, receipt_path,
            note='10 x max(60, ceil(1.5 x measured)); allocation reservation and actual budget stay fixed.')

    for role in ('teacher', 'rewrite'):
        prefix = base+role+'/finqa/'
        raw = archive.jsonl(prefix+'raw_outputs.jsonl')
        records = archive.jsonl(prefix+'timings.jsonl')
        pool = archive.jsonl(prefix+'pool.jsonl')
        manifest = archive.json(prefix+'manifest.json')
        slowest_generate = max(float(number(r['generate_timing']['wall_seconds'], 'generate')) for r in raw)
        slowest_verify = max(float(number(r['verify']['wall_seconds'], 'verify')) for r in records)
        reload = max(float(number(r['reload_timing']['wall_seconds'], 'reload')) for r in raw)
        serial = archive.json(prefix+'serial-merge-timing.json')
        serial_rate = float(number(serial['wall_seconds'], 'serial'))/len(raw)
        for r in raw:
            observe('generate:'+role+':finqa', r['generate_timing'])
            observe('corpus_reload:'+role+':finqa', r['reload_timing'])
        for r in records:
            observe('verify:'+role+':finqa', r['verify'])
        observe('serial_merge:'+role+':finqa', serial, len(raw))
        # Worst serial attempts rather than optimistic throughput discount from
        # the registered 4/8 shards: phase 0 has not measured those full waves.
        projected = 1.5*(reload+4*1441*(slowest_generate+slowest_verify+serial_rate))
        wall = archive.json(base+'timings/'+role+'-finqa.json')['wall_seconds']
        add('generation:'+role+':1441-questions', wall, projected, row_cap(rows[role+'-finqa']), prefix,
            note=f'{len(pool)} questions, {len(raw)} attempts; model {manifest.get("teacher_model")}; '
                 '4 attempts/question, slowest serial generation + verification + merge, 1.5 allowance; no shard discount.')
        add('generation:'+role+':reload', reload, reload, row_cap(rows[role+'-finqa']), prefix)
        add('generation:'+role+':serial-merge', float(serial['wall_seconds']), 1.5*4*1441*serial_rate,
            row_cap(rows[role+'-finqa']), prefix+'serial-merge-timing.json')
        # Throughput is an observation, not an assertion of full-pool acceptance.
        comparisons[-3]['questions_per_second'] = len(pool)/float(wall) if wall else None
        comparisons[-3]['attempts_per_second'] = len(raw)/float(wall) if wall else None
        add('generation:'+role+':1441-questions:qualification', wall,
            projected, row_cap(qrows[role+'-finqa']), prefix,
            note='Qualification now builds the full 1,441-question corpus: same conservative four-attempt bound as main.')
        add('generation:'+role+':1441-observed-throughput', wall,
            1.5*float(wall)*1441/len(pool), min(row_cap(qrows[role+'-finqa']),row_cap(rows[role+'-finqa'])), prefix,
            note='Observed whole-row questions/second extrapolated to 1,441 questions x 1.5, including measured reload overhead. '
                 'Phase-0 acceptance/attempt mix only; not the conservative four-attempt bound and no unmeasured shard speedup.')

    for arm, training in reader['tables']['training'].items():
        prefix = 'runs/q-'+arm+'-finqa/'
        launch_path = base+'timings/train-q-'+arm+'-finqa.json'
        launch = archive.json(launch_path)
        launch_seconds = float(number(launch['wall_seconds'], 'launch', positive=True))
        slowest_step = max(training['seconds_per_step'])
        startup = max(0, launch_seconds-sum(training['seconds_per_step'])-
                      training['export_seconds']-training['merge_seconds'])
        exports = archive.jsonl(prefix+'env/export-timings.jsonl')
        merge = archive.json(prefix+'env/merge-timing.json')
        for step in training['seconds_per_step']:
            observe('train:'+arm+':finqa', dict(launch, wall_seconds=step))
        observe('startup:'+arm+':finqa', dict(launch, wall_seconds=startup))
        for export in exports:
            observe('export:'+arm+':finqa', export)
        observe('merge:'+arm+':finqa', merge)
        jobs = [j for j in main['v4']['slot_order'] if j['arm']==arm]
        cap = min(row_cap(rows['train-'+j['id']]) for j in jobs)
        projected = 1.5*max(20*launch_seconds, startup+40*slowest_step+
                            2*training['export_seconds']+2*training['merge_seconds'])
        add('training:'+arm+':40-updates', launch_seconds, projected, cap, launch_path,
            note='1.5 x max(20 x whole two-step launch, startup + 40 x slowest step + two exports/merges). '
                 'Large step-20 state IO/reload remains unmeasured.')
        add('training:'+arm+':export', training['export_seconds'], 1.5*2*training['export_seconds'], cap,
            prefix+'env/export-timings.jsonl', note='Two saves inside train cap; shared cap, not separate headroom.')
        add('training:'+arm+':merge', training['merge_seconds'], 1.5*training['merge_seconds'],
            min(row_cap(rows['merge-'+j['id']]) for j in jobs), prefix+'env/merge-timing.json')
        add('training:'+arm+':2-updates:qualification', launch_seconds, 1.5*launch_seconds,
            row_cap(qrows['train-q-'+arm+'-finqa']), launch_path, note='C1 whole two-step launch x 1.5 vs substituted/floored smoke cap.')
        add('training:'+arm+':merge:qualification', training['merge_seconds'], 1.5*training['merge_seconds'],
            row_cap(qrows['merge-q-'+arm+'-finqa']), prefix+'env/merge-timing.json')
        add('training:'+arm+':export:qualification', training['export_seconds'], 1.5*training['export_seconds'],
            row_cap(qrows['train-q-'+arm+'-finqa']), prefix+'env/export-timings.jsonl', note='One export shares the qualification train cap.')
        for rank, memory in training['memory'].items():
            for kind in ('allocated', 'reserved'):
                measured = memory['peak_'+kind+'_bytes']/GIB
                add('gpu-memory:'+arm+':'+str(rank)+':'+kind, measured, measured, 80_000_000_000/GIB,
                    prefix+'env/gpu-memory.jsonl', 'GiB',
                    'Conservative nominal 80 GB H100 capacity converted to GiB; confirm actual device capacity. '
                    'Two steps do not prove 40-update peak fits.')

    for name in sorted(archive.files):
        if name.startswith(base+'eval/') and name.endswith('/timing.json'):
            record = archive.json(name)
            ident = name[len(base+'eval/'):].split('/')[0]
            checkpoint, _, cap_text = ident.rsplit('-', 2)
            n = len(archive.jsonl(name.removesuffix('timing.json')+'responses.jsonl'))
            generated = float(number(record['generate']['wall_seconds'], 'score generation'))
            reload = float(number(record['reload']['wall_seconds'], 'score reload'))
            projected = 1.5*(reload+generated*1147/n)
            cap = min(row_cap(r) for r in main['rows'] if r['id'].startswith('score-') and r['id'].endswith('-finqa-'+cap_text))
            add('scoring:'+checkpoint+':finqa:'+cap_text, generated+reload, projected, cap, name,
                note=f'{n} questions -> 1147, reload once + generation x panel ratio, 1.5 allowance.')
            add('scoring:'+checkpoint+':reload', reload, 1.5*reload, cap, name)
            observe('eval:finqa:'+cap_text, record['generate'], n)
            observe('reload:finqa:'+cap_text, record['reload'])
            qrow = 'score-'+checkpoint+'-finqa-'+cap_text
            if qrow in qrows:
                qn = 50 if checkpoint.startswith('agreement') else 20
                add('scoring:'+checkpoint+':finqa:'+cap_text+':qualification', generated+reload,
                    1.5*(reload+generated*qn/n), row_cap(qrows[qrow]), name,
                    note=f'{n} phase-0 questions -> {qn} qualification questions; reload once, 1.5 allowance.')

    # The enclosing command, dispatch and all preparation/refusal observations
    # are listed separately from component estimates; caps never double count.
    for name in sorted(archive.files):
        if name.startswith(base+'timings/') and name.endswith('.json'):
            record = archive.json(name)
            ident = name[len(base+'timings/'):-5]
            if ident.startswith('train-q-'):
                cap = min(row_cap(r) for r in main['rows'] if r['id'].startswith('train-'))
            elif ident.startswith('merge-q-'):
                cap = min(row_cap(r) for r in main['rows'] if r['id'].startswith('merge-'))
            elif ident.startswith('score-'):
                cap = min(row_cap(r) for r in main['rows'] if r['id'].startswith('score-'))
            else:
                cap = row_cap(rows[ident]) if ident in rows else None
            if ident in qrows:
                add('row:'+ident+':qualification', record['wall_seconds'], 1.5*float(record['wall_seconds']),
                    row_cap(qrows[ident]), name, note='Observed whole row x 1.5 vs corresponding qualification cap.')
            if cap is None and ident not in qrows:
                comparisons.append(dict(quantity='row:'+ident, measured=record.get('wall_seconds'),
                    extrapolated=None, cap=None, margin=None, unit='seconds', status=DEFERRED, missing=True,
                    source=name, note='No equivalent main row; qualification-only evidence requires manager review.'))
            elif cap is not None:
                add('row:'+ident, record['wall_seconds'], 1.5*float(record['wall_seconds']), cap, name)

    disk_paths = [name for name in archive.files if name.startswith(base) and
                  ('storage' in name or 'disk' in name) and name.endswith('.json')]
    used = [(name, archive.json(name)['used_bytes']) for name in disk_paths if 'used_bytes' in archive.json(name)]
    free = [(name, archive.json(name)['free_bytes']) for name in disk_paths if 'free_bytes' in archive.json(name)]
    add('disk-used', max((float(number(v, 'disk used'))/GIB for _, v in used), default=None),
        max((float(number(v, 'disk used'))/GIB for _, v in used), default=None), DISK_GIB,
        ', '.join(n for n, _ in used) or 'not recorded', 'GiB',
        '1879 GiB serial-main capacity advice. Archive compressed size is not disk usage; phase-0 usage is not main peak.')
    if free:
        available = min(float(number(v, 'disk free'))/GIB for _, v in free)
        row = comparison('disk-free-admission', available, DISK_GIB, max(available, 1e-9), 'GiB',
                         ', '.join(n for n, _ in free))
        row['note'] = '1879 GiB is capacity advice, never prepare admission; refresh free space before each next write.'
        comparisons.append(row)
    else:
        add('disk-free-admission', None, None, DISK_GIB, 'not recorded', 'GiB',
            '1879 GiB is capacity advice, never prepare admission; only insufficient free space for the next write stops work. Archive size cannot substitute.')
    for quantity, cap, note in (
        ('training:D-R:40-updates', min(row_cap(r) for r in main['rows'] if r['id'].startswith('train-')), 'D-R is absent from phase 0.'),
        ('scoring:finqa:8192', min(row_cap(r) for r in main['rows'] if r['id'].startswith('score-') and r['id'].endswith('-finqa-8192')), 'The phase-0 B=2048 observation does not measure H=8192.'),
        ('scoring:general-v1:2048', min(row_cap(r) for r in main['rows'] if r['id'].startswith('score-') and r['id'].endswith('-general-v1-2048')), 'General-v1 panel was not scored in phase 0.'),
        ('scoring:general-v1:8192', min(row_cap(r) for r in main['rows'] if r['id'].startswith('score-') and r['id'].endswith('-general-v1-8192')), 'General-v1 panel was not scored in phase 0.'),
        ('probe:finqa', min(row_cap(r) for r in main['rows'] if r['id'].startswith('probe-')), 'Eight-trajectory probes were not measured in phase 0.'),
        ('full-state-save-merge-reload:step20', min(row_cap(r) for r in main['rows'] if r['id'].startswith('train-')), 'Two-step phase-0 export does not measure the main step-20 full state and reload.'),
        ('generation:full-wave-profile', row_cap(rows['teacher-finqa']), 'Four teacher and eight rewrite shard waves require qualification profiling.')):
        add(quantity, None, None, cap, 'not recorded', note=note)
    add('disk-main-peak', None, None, DISK_GIB, 'not recorded', 'GiB',
        'Qualify restoration/state IO and retention-based main peak estimate during qualification; phase-0 usage is not main peak.')
    # No timed receipt disappears simply because it is not a whole campaign
    # row. Compare all remaining individual observations to their enclosing
    # operation; retain exact path/field in both reports for manager audit.
    for observation in inventory:
        source = observation['source']
        if not source.endswith(('/seconds', '/wall_seconds', '/import_seconds')):
            continue
        cap = None
        if source.startswith(base+'teacher/'):
            cap = row_cap(rows['teacher-finqa'])
        elif source.startswith(base+'rewrite/'):
            cap = row_cap(rows['rewrite-finqa'])
        elif source.startswith(base+'eval/'):
            cap = min(row_cap(r) for r in main['rows'] if r['id'].startswith('score-'))
        elif source.startswith('runs/q-'):
            operation = 'merge-' if '/merge-timing.json' in source else 'train-'
            cap = min(row_cap(r) for r in main['rows'] if r['id'].startswith(operation))
        elif source.startswith(base+'environment-check'):
            cap = row_cap(rows['environment'])
        elif source.startswith(base+'presend/capture.json'):
            from kit.v4_phase0_timing import scale
            cap = scale(120)
        elif source.startswith('v4/report-inputs/import-smoke-phase0/'):
            from kit.v4_phase0_timing import scale
            cap = scale(120)
        elif source.startswith((base+'environment-build/', base+'environment.json')) or source.startswith(receipt_path):
            cap = row_cap(rows['prepare'])
        elif source.startswith('k8b4/containment/'):
            cap = row_cap(rows['containment-selftest'])
        elif source.startswith('campaign/v4-phase0/'):
            ident = source.split('/')[2]
            if ident in rows:
                cap = row_cap(rows[ident])
            elif ident.startswith('train-'):
                cap = min(row_cap(r) for r in main['rows'] if r['id'].startswith('train-'))
            elif ident.startswith('score-'):
                cap = min(row_cap(r) for r in main['rows'] if r['id'].startswith('score-'))
        if cap is not None:
            add('observation:'+source, observation['value'], observation['value'], cap, source,
                note='Individual observed duration vs enclosing main operation cap; aggregate estimates listed separately.')
        elif not source.startswith(base+'timings/'):
            comparisons.append(dict(quantity='observation:'+source, measured=observation['value'],
                extrapolated=None, cap=None, margin=None, unit='seconds', status=DEFERRED, missing=True,
                source=source, note='No registered corresponding main cap; retain for manager review.'))
    return comparisons, timings, inventory


def report_categories(report):
    """Separate technical input blockers from self-monitoring observations."""
    passed = report['reader']['status']=='technical pass'
    blockers = [dict(item='Phase-0 reader technical pass', status='PASS' if passed else 'FAIL',
                     reason='Technical outputs only.' if passed else '; '.join(report['refusals']))]
    if passed:
        blockers.append(dict(item='Local-patch/source identity recorded', status='WARNING',
                             reason=report.get('identity', {}).get('reason', 'Source identity correspondence is an observation to flag in the note.')))
        graph = report.get('campaigns', {}).get('qualification', {})
        valid = graph.get('c1_envelope_accepted', False)
        blockers.append(dict(item='C1-substituted qualification graph valid', status='PASS' if valid else 'FAIL',
            reason='Real runner validated the substituted graph and carry binding.' if valid else
                   'C1 graph absent or refused; diagnostic candidates are not launch eligible.'))
        budget = report.get('budget', {})
        if budget:
            fit = budget['qualification_fit']
            reason = (f'{budget["phase0_qualification_gpu_hours"]:.6f} phase-0 + '
                      f'{budget["qualification_envelope_gpu_hours"]:.6f} substituted envelope = '
                      f'{budget["qualification_total_gpu_hours"]:.6f} / 100 GPU-hours. '
                      f'Allocator request {fit["requested_minutes"]} min -> {fit["allowed_minutes"]} min; '
                      f'clip {fit["clipped_minutes"]} min ({fit["clipped_gpu_hours"]:.6f} GPU-hours). '
                      f'Full envelope needs {fit["required_minutes"]} whole minutes; '
                      f'shortfall in planned window {fit["envelope_shortfall_seconds"]:.0f} seconds. '
                      'Rows at risk under cap consumption: '+(', '.join(r['row'] for r in fit['rows_at_risk']) or 'none')+'.')
            blockers.append(dict(item='Qualification actual-spend accounting including phase-0', status='FAIL' if budget['actual_gpu_hours']>=1000 else 'WARNING' if budget['qualification_total_gpu_hours']>100 else 'PASS', reason=reason+' Projection/block fit is advisory; only actual 1,000 GPU-hour spend stops work.'))
        else:
            blockers.append(dict(item='Qualification fit including phase-0 spend', status='FAIL', reason='No verified qualification fit available.'))
        if report['refusals']:
            blockers.append(dict(item='Source/campaign verification', status='FAIL', reason='; '.join(report['refusals'])))
    blockers.append(dict(item='Partner disk-free requirement restated', status='ADVICE',
        reason='1,879 GiB is capacity advice printed at prepare; refuse only when free space is below the next step write requirement. '
               'No fresh host free-space or main peak measurement is required from phase 0 for tagging.'))
    measured, deferred = [], []
    for row in report['comparisons']:
        row['category'] = DEFERRED if row['status']==DEFERRED else 'MEASURED NOW'
        (deferred if row['status']==DEFERRED else measured).append(row)
    for key in report.get('budget', {}).get('missing_timings', []):
        deferred.append(dict(quantity='main-budget:'+key, status=DEFERRED,
            measured=None, extrapolated=None, cap=None, margin=None, unit='seconds',
            source='qualification timing collection', note='Collect during qualification; price during qualification.', category=DEFERRED))
    if passed:
        deferred.append(dict(quantity='main budget projection', status=DEFERRED,
            measured=None, extrapolated=None, cap=1000, margin=None, unit='GPU-hours',
            source='live qualification progress report', category=DEFERRED,
            note='Qualification measurements refine projections; main follows automatically without a proceed receipt.'))
    report.update(tag_blockers=blockers, measured_now=measured, deferred_to_qualification=deferred,
        status='NO-GO' if any(r['status']=='FAIL' for r in blockers) else 'GO WITH WARNINGS',
        decisions=[dict(status=r['status'], category='TAG BLOCKERS', reason=r['item']+': '+r['reason']) for r in blockers])


def write_report(out, report):
    report_categories(report)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# FinQA manager ingest', '', 'Status: **'+report['status']+'**. Main flow: **automatic; owner self-monitors**.',
             '', 'Phase-0 verdict: **'+report['reader']['status']+'**.', '']
    lines += ['- STOP: '+r for r in report['refusals']]
    if report.get('identity'):
        lines += ['', 'Kit identity: '+report['identity']['reason']]
    if report.get('budget'):
        b = report['budget']
        lines += ['', f'Qualification: {b["phase0_qualification_gpu_hours"]:.6f} phase-0 + '
                  f'{b["qualification_envelope_gpu_hours"]:.6f} qualification envelope = '
                  f'{b["qualification_total_gpu_hours"]:.6f} / 100 GPU-hours (**{b["qualification_status"]}**).',
                  f'Campaign ceiling: 1000 GPU-hours; actual carry {b["actual_gpu_hours"]:.6f}; '
                  f'remaining {b["remaining_ceiling_gpu_hours"]:.6f}.',
                  f'Main static envelope (all rows at full eight-GPU allocation width): '
                  f'{b["main_static_envelope_gpu_hours"]:.6f} GPU-hours. This is a cap envelope, not a measured admission.',
                  'Main budget admission: **'+b['main_admission']['status']+'**: '+
                  '; '.join(b['main_admission']['reasons']),
                  f'Available measured component projection + carry + 45-hour contingency: '
                  f'{b["partial_projected_gpu_hours"]:.6f} GPU-hours. '+b['partial_projection_scope'],
                  'Measurements collected during qualification: '+', '.join(b['missing_timings'])]
    lines += ['', '## Manager go/no-go', '', '### TAG BLOCKERS', '']+[
        f'- {r["status"]}: {r["item"]}. {r["reason"]}' for r in report['tag_blockers']]
    if report.get('budget'):
        fit = report['budget']['qualification_fit']
        lines += ['', fit['trace_assumption'],
                  'Captured site limit: '+(str(fit['site_minutes'])+' minutes.' if fit['site_limit_known'] else
                    'not recorded; no budget clipping; a smaller site window can need continuation.'),
                  f'Requesting the {fit["required_minutes"]}-minute whole-envelope window would be limited to '
                  f'{fit["full_envelope_allowed_minutes"]} minutes (clip {fit["full_envelope_clipped_minutes"]} minutes).', '',
                  '| Row needing a continuation if preceding caps are fully consumed | Worst-case start (s) | Available (s) | Wall-clock backstop (s) |',
                  '|---|---:|---:|---:|']
        for row in fit['rows_at_risk']:
            lines.append(f'| {row["row"]} | {row["starts_at_worst_case_seconds"]} | {row["available_seconds"]} | {row["admission_cap_seconds"]:g} |')
    lines += ['', '### MEASURED NOW', '',
              '| Quantity | Measured | Extrapolated | Cap | Cap / requirement | Verdict | Source / method |',
              '|---|---:|---:|---:|---:|---|---|']
    def fmt(value):
        return 'missing' if value is None else f'{value:.6g}'
    for r in report['measured_now']:
        lines.append(f'| {r["quantity"]} ({r["unit"]}) | {fmt(r["measured"])} | {fmt(r["extrapolated"])} | '
                     f'{fmt(r["cap"])} | {fmt(r["margin"])} | {r["status"]} | '+
                     (r['source']+'; '+r.get('note', '')).replace('|', '\\|').replace('\n', ' ')+' |')
    lines += ['', '### DEFERRED TO QUALIFICATION', '',
              '| Quantity | Status | Collection / pricing point |', '|---|---|---|']
    for row in report['deferred_to_qualification']:
        lines.append('| '+row['quantity']+' | '+DEFERRED+' | '+row.get('note', '').replace('|', '\\|')+' |')
    lines += ['', 'Zero requirements pass with no finite ratio. Expected later-stage measurements are deferred, never labelled FAIL.',
              'All numeric time/memory/storage observations and available budget samples are retained in report.json.',
              'Full-wave throughput, D-R, H/general scoring, probes, runner setup/dispatch and full-state IO '
              'still require qualification. No two-step or CPU rehearsal proves main admission.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def ingest(archive_path, out, local_diff=None):
    from kit.v4_phase0 import read_archive
    from kit.v4_archive import Archive
    from kit.v4_finqa_campaign import build
    from kit import v4_budget as budget, v4_phase0_carry as carry, runner
    archive_path, out = Path(archive_path).resolve(), Path(out).resolve()
    if out == archive_path or out in archive_path.parents or (local_diff and out in Path(local_diff).resolve().parents):
        raise ValueError('output directory must be distinct from read-only inputs')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='v4-finqa-ingest.v1', status='NO-GO', phase0_exports_are_scientific_initialisations=False,
                  owner_review_required=False, archive={'name':archive_path.name, 'sha256':sha(archive_path)},
                  reader=read_archive(archive_path), local_diff=None,
                  refusals=[], comparisons=[], decisions=[])
    try:
        if report['reader']['status'] != 'technical pass':
            report['refusals'] = report['reader']['reasons'] or [report['reader']['status']]
            report['decisions'].append(dict(status='NO-GO', reason='Phase-0 reader refused; no campaign generated.'))
            try:
                report['local_diff'] = diff_record(local_diff, out)
            except (OSError, ValueError) as exc:
                report['refusals'].append(str(exc))
            return report
        report['local_diff'] = diff_record(local_diff, out)
        with Archive(archive_path) as archive:
            tag = archive.json('v4/report-phase0/environment.json')['kit_tag']
            touched = (report['local_diff'] or {}).get('files_touched', [])
            expected = bool(touched) and set(touched) <= {'kit/v4_phase0_site.py', 'kit/v4_phase0.py', 'kit/v4_allocation.py'}
            report['identity'] = dict(recorded_kit_tag=tag, trusted_clean_tag=False,
                expected_partner_patch_files=expected,
                snapshot_sha256={n:hashlib.sha256(raw).hexdigest() for n, raw in archive.files.items() if n.startswith('kit-snapshot/')},
                reason='Recorded tag alone cannot authenticate patched executable bytes. '+
                       ('Local diff preserved; file scope '+('matches' if expected else 'differs from')+
                        ' the reported infrastructure patch files (torch/verification or v7 scheduler UUID evidence); flag semantic review and source correspondence in the return note.'
                        if touched else 'Reported local patches have no supplied diff; flag the missing source correspondence in the return note.'))
        # Production C1 hook, real runner graph validator, real carry-forward and
        # exact allocation arithmetic; never modify a partner manifest in place.
        qualification = build('qualification', phase0_launch_archive=archive_path)
        graphs = dict(qualification=qualification, main=build('main', phase0_launch_archive=archive_path))
        campaigns = out/'campaigns'
        campaigns.mkdir()
        report['campaigns'] = {}
        for phase, graph in graphs.items():
            path = campaigns/f'v4-{phase}.yaml'
            path.write_text(json.dumps(graph, indent=2)+'\n')
            runner.load_campaign(path)
            report['campaigns'][phase] = dict(path=str(path.relative_to(out)), sha256=sha(path), rows=len(graph['rows']),
                c1_envelope_accepted=True,
                envelope_gpu_hours=float(budget.gpu_hours(8, sum(r['timeout_seconds'] for r in graph['rows']))))
        report['campaigns']['qualification']['launch_allowances'] = graphs['qualification']['v4']['launch_allowances']
        carry.bind(out/'work', archive_path, campaigns=campaigns)
        ledger = carry.verify_work(out/'work', campaigns=campaigns)
        spent, blocks = budget.allocation_spend(ledger)
        with Archive(archive_path) as archive:
            comparisons, timings, inventory = measurements(archive, report['reader'], graphs['main'], graphs['qualification'])
            site_minutes = archive.json('v4/report-phase0/presend/capture.json').get('site_minutes') \
                if 'v4/report-phase0/presend/capture.json' in archive.files else None
        report.update(comparisons=comparisons, measurement_inventory=inventory, budget_timings=timings)
        keys = {r['key'] for r in budget.work_items()}
        keys.update(f'{op}:{role}:finqa' for role in ('teacher', 'rewrite') for op in ('generate', 'verify'))
        missing = sorted(keys-set(timings))
        admission = dict(status=DEFERRED, reasons=['Collect qualification timings and report projections as warnings; continue automatically.']) \
            if missing else budget.admit(ledger, timings)
        priced_components = []
        price_rows = budget.work_items()+[dict(key=f'{op}:{role}:finqa', units=4*1441,
                       block='qualification' if role=='rewrite' else 'main')
                       for role in ('teacher', 'rewrite') for op in ('generate', 'verify')]
        for row in price_rows:
            samples = timings.get(row['key'])
            if not samples:
                continue
            seconds, width = max((budget.number(s['wall_seconds'], 'wall')/budget.number(s['units'], 'units', positive=True),
                                  budget.number(s['allocation_gpus'], 'width', positive=True)) for s in samples)
            priced_components.append(dict(**row, slowest_seconds_per_unit=float(seconds), allocation_gpus=int(width),
                projected_gpu_hours=float(budget.ALLOWANCE*budget.gpu_hours(width, seconds*row['units']))))
        envelope = report['campaigns']['qualification']['envelope_gpu_hours']
        total = float(blocks.get('qualification', 0))+envelope
        report['budget'] = dict(actual_gpu_hours=float(spent), phase0_qualification_gpu_hours=float(blocks.get('qualification', 0)),
            qualification_envelope_gpu_hours=envelope, qualification_total_gpu_hours=total,
            qualification_limit_gpu_hours=1000,planned_qualification_gpu_hours=100, qualification_status='WARNING' if total > 100 else 'TIGHT' if total > 80 else 'PASS',
            qualification_fit=qualification_fit(graphs['qualification'], spent, blocks, site_minutes=site_minutes),
            ceiling_gpu_hours=1000, remaining_ceiling_gpu_hours=float(1000-spent), contingency_gpu_hours=45,
            main_static_envelope_gpu_hours=report['campaigns']['main']['envelope_gpu_hours'],
            available_measured_components=priced_components,
            partial_projected_gpu_hours=float(spent)+45+sum(r['projected_gpu_hours'] for r in priced_components),
            partial_projection_scope='Unpacked available components only, actual carry + 1.5 margin + 45 contingency; '
                                     'not an admission or an upper bound. Missing measurements remain unpriced.',
            missing_timings=missing, main_admission=admission)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        report['refusals'].append(str(exc))
        report['decisions'].append(dict(status='NO-GO', reason=str(exc)))
    finally:
        if sha(archive_path) != report['archive']['sha256']:
            report['refusals'].append('phase-0 archive changed during ingest')
            report['decisions'].append(dict(status='NO-GO', reason=report['refusals'][-1]))
        write_report(out, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--local-diff', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = ingest(args.archive, args.out, args.local_diff)
    except (ValueError, OSError) as exc:
        print(str(exc))
        return 2
    print('Phase-0 verdict: '+report['reader']['status'])
    for reason in report['refusals']:
        print(reason)
    print('Manager status: '+report['status'])
    print('Manager report: '+str(args.out/'report.md'))
    # Measurement findings are advisory; source/reader failures still return nonzero.
    return 2 if report['refusals'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
