"""CPU-only A2 reader. Exact paired counts; exploratory reuse never admits an arm."""
from kit.a2_common import *


def score_plan(arm):
    cells = []
    for n, month in enumerate(MONTHS, 1):
        for model in arm_names(arm):
            for seed in SEEDS:
                for mode in ('on', 'off'):
                    cells.append(('arrival', model, seed, n, month, mode))
        for mode in ('on', 'off'):
            cells.append(('arrival', 'frozen-reference', 0, n, month, mode))
    for n in BOUNDARIES:
        for model in arm_names(arm):
            for seed in SEEDS:
                for month in MONTHS[:n]:
                    for mode in ('on', 'off'):
                        cells.append(('post', model, seed, n, month, mode))
                for panel in GENERAL:
                    cells.append(('post', model, seed, n, panel, 'off'))
    for panel in GENERAL:
        cells.append(('base', 'frozen-reference', 0, 0, panel, 'off'))
    return cells


def confirmation_plan(arm):
    cells = []
    for model in arm_names(arm):
        for seed in SEEDS:
            for month in ('2026-10', '2026-11'):
                cells.append(('confirmation', model, seed, 6, month, 'on'))
            for month in MONTHS:
                for mode in ('on', 'off'):
                    cells.append(('terminal', model, seed, 6, month, mode))
            for panel in GENERAL:
                cells.append(('terminal', model, seed, 6, panel, 'off'))
    return cells


def name(key):
    return '/'.join(map(str, key))


def pair(first, second):
    def index(rows):
        values = {r['question_id']: r['correct'] for r in rows}
        require(len(values) == len(rows) and all(type(v) is int and v in (0, 1)
                                               for v in values.values()), 'invalid per-item correct counts')
        return values
    a, b = index(first), index(second)
    require(a and a.keys() == b.keys(), 'unpaired or empty score denominator')
    return dict(n=len(a), delta=100*sum(b[k]-a[k] for k in a)/len(a),
                correct_to_wrong=sum(a[k] == 1 and b[k] == 0 for k in a),
                wrong_to_correct=sum(a[k] == 0 and b[k] == 1 for k in a))


def analyse(arm, cells, *, synthetic=False, confirmation=False, receipt=None, p_applicable=True,
            assignment=None):
    plan = confirmation_plan(arm) if confirmation else score_plan(arm)
    by_key = {tuple(c['key']): c for c in cells}
    if len(by_key) != len(cells):
        return dict(status='UNVERIFIED', reason='duplicate scientific cells')
    missing = [name(k) for k in plan if k not in by_key]
    if missing:
        return dict(status='INCOMPLETE', missing=missing)
    require(set(by_key) == set(plan), 'unregistered score cells')
    for cell in cells:
        expected = cell.get('expected_members')
        require(isinstance(expected, dict) and set(expected) == {r['question_id'] for r in cell['rows']},
                'scoring population differs from prepared membership')
        for row in cell['rows']:
            member = expected[row['question_id']]
            require(bool(row['holdout']) == member['holdout'] and row.get('family', '') == member['family'],
                    'holdout/family assignment differs from prepared membership')
    hardware = {tuple(by_key[k].get('physical', {}).get(x) for x in ('allocation_nonce', 'uuid', 'environment'))
                for k in plan}
    hardware_ok = len(hardware) == 1 and all(next(iter(hardware)))
    if hardware_ok:
        nonce, gpu, environment = next(iter(hardware))
        hardware_ok = bool(assignment and assignment.get('status') == 'COMPLETE' and
                           assignment.get('nonce') == nonce and assignment.get('allocated_uuids') == [gpu])
    result = dict(arm=arm, registration_sha256=REGISTRATION, seeds=list(SEEDS),
                  hardware_status='VERIFIED' if hardware_ok else 'UNVERIFIED-HARDWARE',
                  contrasts={}, provenance='PARTNER-COMPUTED', independent_reverification='UNVERIFIED',
                  checksums='advisory', utilisation='warn-only',
                  phase='confirmation' if confirmation else 'exploratory',
                  admission='PENDING: confirmation evidence', technical_synthetic=synthetic)
    if confirmation:
        if not receipt or not receipt.get('terminal_june_frozen') or not receipt.get('audit_verified'):
            return dict(result, status='PENDING', reason='confirmation freeze/audit missing')
        for month in ('2026-10', '2026-11'):
            supply = receipt.get('supply', {}).get(month, {})
            if supply.get('questions', 0) < 40 or supply.get('holdouts', 0) < 10 or supply.get('components', 0) < 5:
                return dict(result, status='PENDING', reason='insufficient confirmation evidence')
    transitions = []
    for label, challenger, control in contrasts(arm):
        if label == 'P' and not p_applicable:
            continue
        def delta(kind, seed, n, panel, mode):
            a = by_key[(kind, control, seed, n, panel, mode)]['rows']
            b = by_key[(kind, challenger, seed, n, panel, mode)]['rows']
            if panel.startswith('2026-'):
                a, b = ([r for r in rows if r['holdout']] for rows in (a, b))
            paired = pair(a, b)
            transitions.append(dict(contrast=label, kind=kind, seed=seed, month=n, panel=panel,
                                    mode=mode, granularity_points=100/paired['n'], **paired))
            return paired['delta']
        months = ('2026-10', '2026-11') if confirmation else MONTHS[1:]
        values = []
        monthly = {}
        for seed in SEEDS:
            diffs = [delta('confirmation' if confirmation else 'arrival', seed,
                           6 if confirmation else MONTHS.index(m)+1, m, 'on') for m in months]
            monthly[str(seed)] = diffs
            values.append(sum(diffs)/len(diffs))
        primary = benefit(values)
        guards = {}
        for n in ((6,) if confirmation else BOUNDARIES):
            for panel in (*MONTHS[:n], *GENERAL):
                for mode in (('on', 'off') if panel in MONTHS else ('off',)):
                    key = f'm{n}/{panel}/{mode}'
                    guards[key] = guard([delta('terminal' if confirmation else 'post', seed,
                                               n, panel, mode) for seed in SEEDS])
                    guards[key]['item_denominators'] = [r['n'] for r in transitions[-3:]]
                    guards[key]['item_granularity_points'] = [r['granularity_points'] for r in transitions[-3:]]
        passed = primary['pass_bar'] and all(g['pass_bar'] for g in guards.values())
        result['contrasts'][label] = dict(primary=primary, monthly=monthly, guards=guards,
                                          verdict=('CONFIRMED PASS' if confirmation else 'EXPLORATORY PASS')
                                          if passed else 'INCONCLUSIVE')
    result['transitions'] = transitions
    result['comparisons'] = len(result['contrasts'])
    result['denominators'] = {name(k): dict(n=len(by_key[k]['rows']),
        holdouts=sum(bool(r['holdout']) for r in by_key[k]['rows']),
        families={f: sum(r.get('family') == f for r in by_key[k]['rows']) for f in FAMILIES}) for k in plan}
    def summary(rows):
        n = len(rows)
        return dict(n=n, correct=sum(r['correct'] for r in rows),
                    accuracy=100*sum(r['correct'] for r in rows)/n if n else None,
                    historical_correct=sum(r.get('historical_correct', r['correct']) for r in rows),
                    input_tokens=sum(r.get('input_tokens', 0) for r in rows),
                    output_tokens=sum(r.get('output_tokens', 0) for r in rows),
                    granularity_points=100/n if n else None)
    result['scores'] = {name(k): dict(all=summary(by_key[k]['rows']),
        holdout=summary([r for r in by_key[k]['rows'] if r['holdout']]),
        family={f: summary([r for r in by_key[k]['rows'] if r.get('family') == f]) for f in FAMILIES}) for k in plan}
    if confirmation:
        for key in plan:
            if key[0] == 'confirmation':
                result['scores'][name(key)]['equal_month_granularity_points'] = 50/len(by_key[key]['rows'])
    if not confirmation:
        result['retention_transitions'] = []
        for model in arm_names(arm):
            for seed in SEEDS:
                for index, month in enumerate(MONTHS, 1):
                    for mode in ('on', 'off'):
                        first = [r for r in by_key[('arrival', model, seed, index, month, mode)]['rows'] if r['holdout']]
                        snapshots = [first]
                        for n in BOUNDARIES:
                            if n >= index:
                                post = by_key[('post', model, seed, n, month, mode)]['rows']
                                result['retention_transitions'].append(dict(model=model, seed=seed,
                                    month=month, boundary=n, mode=mode, **pair(first, post)))
                                snapshots.append(post)
                        lookup = [{r['question_id']: r['correct'] for r in rr} for rr in snapshots]
                        acquired_then_lost = 0
                        for item_id in lookup[0]:
                            acquired = False
                            lost = False
                            for before, after in zip(lookup, lookup[1:]):
                                acquired |= before[item_id] == 0 and after[item_id] == 1
                                lost |= acquired and before[item_id] == 1 and after[item_id] == 0
                            acquired_then_lost += lost
                        result['retention_transitions'].append(dict(model=model, seed=seed, month=month,
                            mode=mode, acquired_then_lost=acquired_then_lost, n=len(first)))
    result['status'] = 'COMPLETE' if hardware_ok else 'UNVERIFIED-HARDWARE'
    if not hardware_ok:
        for contrast in result['contrasts'].values():
            contrast['verdict'] = None
    if confirmation and hardware_ok:
        result['admission'] = {k: v['verdict'] for k, v in result['contrasts'].items()}
    if synthetic:
        result['simulated_status'] = result['status']
        result['status'] = 'TECHNICAL-ONLY'
        result['admission'] = None
        result['scientific_verdict'] = None
    return result


def read(out):
    from kit.a2_jobs import completed
    out = Path(out)
    cfg = load(out/'config.json')
    if cfg.get('disposition') == 'archived':
        return dict(arm=cfg['arm'], status='ARCHIVED UNREAD')
    try:
        root = completed(out, 'A2-'+cfg['arm']+'-comparison')
    except NotReady:
        return dict(arm=cfg['arm'], status='INCOMPLETE', reason='complete comparison unfinished')
    cells = load(root/'cells.json')
    doc = load(out/'binding.json')
    try:
        return analyse(cfg['arm'], cells, synthetic=cfg.get('technical_synthetic', False),
                       p_applicable=doc.get('P_applicable', True), assignment=load(root/'receipt.json'))
    except CorrectnessStop as error:
        return dict(arm=cfg['arm'], status='UNVERIFIED', reason=str(error))
