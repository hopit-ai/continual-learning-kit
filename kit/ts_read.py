"""TS contrasts from validated per-item vectors, with the inherited paired bars."""
from kit import nm_read, ts_arms as ts
from kit.nm_common import SEEDS, benefit, guard, Pending


PANELS = ('instruction', 'maths', 'knowledge', 'hellaswag', 'finqa', 'legal', 'math_hard')
TASK_PANEL = dict(finqa='finqa', contract_nli='legal', math_hard='math_hard')
CONTRASTS = {'S-D': ('S', 'D', 'primary'), 'H-D': ('H', 'D', 'primary'),
             'S-M': ('S', 'M', 'secondary'), 'M-D': ('M', 'D', 'secondary')}


def frontier(panel, arms=ts.ARMS, *, include_base=True):
    cells = [('base', None, 0)] if include_base else []
    for arm in arms:
        for number, stage in enumerate(ts.STAGES):
            steps = {40}
            if panel == TASK_PANEL[stage]:
                steps.update(ts.STEPS)
            if panel in [TASK_PANEL[s] for s in ts.STAGES[:number]]:
                steps.add(20)
            cells.extend((arm, stage, step) for step in sorted(steps))
    return cells


def model_name(arm, stage):
    return 'base' if arm == 'base' else arm + '.' + stage


def evaluate(doc, spec, *, arms=ts.ARMS, allow_missing=False):
    table = nm_read.Table(doc, spec)
    status = {arm: 'COMPLETE' if arm == 'base' or arm in arms else 'PENDING/DEFERRED'
              for arm in ('base', *ts.ARMS)}
    for seed in SEEDS:
        for panel in PANELS:
            for arm, stage, step in frontier(panel, arms, include_base=True):
                try:
                    table.cell(seed, panel, model_name(arm, stage), step)
                except Pending:
                    if not allow_missing:
                        raise
                    status[arm] = 'PENDING'
    complete = [a for a in arms if status[a] == status['base'] == 'COMPLETE']
    def acc(arm, seed, stage, panel, step=40):
        return table.acc(seed, panel, model_name(arm, stage), step)
    def retained(arm, seed):
        parts = [acc(arm, seed, 'math_hard', p) - table.acc(seed, p, 'base') for p in PANELS[:3]]
        parts += [acc(arm, seed, 'math_hard', 'finqa') - acc(arm, seed, 'finqa', 'finqa'),
                  acc(arm, seed, 'math_hard', 'legal') - acc(arm, seed, 'contract_nli', 'legal')]
        return sum(parts) / 5
    guards = {}
    for number, stage in enumerate(ts.STAGES):
        for panel in PANELS[:4]:
            guards[stage + '/general/' + panel] = lambda a, s, st=stage, p=panel: acc(a, s, st, p)
        panel = TASK_PANEL[stage]
        guards[stage + '/task/final'] = lambda a, s, st=stage, p=panel: acc(a, s, st, p)
        guards[stage + '/task/gain'] = lambda a, s, st=stage, p=panel: acc(a, s, st, p) - acc(a, s, st, p, 0)
        for earlier in ts.STAGES[:number]:
            panel = TASK_PANEL[earlier]
            guards[stage + '/absolute/' + panel] = lambda a, s, st=stage, p=panel: acc(a, s, st, p)
            guards[stage + '/retention/' + panel] = lambda a, s, st=stage, e=earlier, p=panel: acc(a, s, st, p) - acc(a, s, e, p)
    contrasts = {}
    for name, (challenger, control, priority) in CONTRASTS.items():
        if challenger not in complete or control not in complete:
            contrasts[name] = dict(priority=priority, status='PENDING', reason='missing or deferred arm evidence')
            continue
        endpoint = benefit([retained(challenger, s) - retained(control, s) for s in SEEDS])
        paired = {name: guard([fn(challenger, s) - fn(control, s) for s in SEEDS]) for name, fn in guards.items()}
        contrasts[name] = dict(priority=priority, endpoint='R', primary=endpoint, guards=paired,
            status='PASS' if endpoint['pass'] and all(g['pass'] for g in paired.values()) else 'INCONCLUSIVE')
    curves, survival = {}, {}
    for arm in complete:
        for seed in SEEDS:
            for stage in ts.STAGES:
                panel = TASK_PANEL[stage]
                values = [acc(arm, seed, stage, panel, step) for step in ts.STEPS]
                curves[f'{arm}.{seed}.{stage}'] = dict(gain=values[-1] - values[0],
                    gain_auc=sum((a + b) * 5 for a, b in zip(values, values[1:])) / 40 - values[0])
                incoming = table.items(seed, panel, model_name(arm, stage), 0)
                acquisition = table.items(seed, panel, model_name(arm, stage), 40)
                final = table.items(seed, panel, model_name(arm, 'math_hard'), 40)
                acquired = [i for i in incoming if not incoming[i] and acquisition[i]]
                survival[f'{arm}.{seed}.{stage}'] = dict(denominator=len(acquired), survived=sum(final[i] for i in acquired),
                    fraction=sum(final[i] for i in acquired) / len(acquired) if acquired else None,
                    transitions={f'{a}->{b}': sum(acquisition[i] == a and final[i] == b for i in incoming) for a in (0, 1) for b in (0, 1)})
    # Common cohorts use the intersection of each compared arm's acquired IDs.
    common = {}
    for name, (challenger, control, _) in CONTRASTS.items():
        if challenger not in complete or control not in complete:
            continue
        for seed in SEEDS:
            for stage in ts.STAGES[:2]:
                panel = TASK_PANEL[stage]
                cohorts = []
                for arm in (challenger, control):
                    incoming = table.items(seed, panel, model_name(arm, stage), 0)
                    acquisition = table.items(seed, panel, model_name(arm, stage), 40)
                    cohorts.append({i for i in incoming if not incoming[i] and acquisition[i]})
                ids = cohorts[0] & cohorts[1]
                common[f'{name}.{seed}.{stage}'] = dict(denominator=len(ids), survived={a: sum(
                    table.items(seed, panel, model_name(a, 'math_hard'), 40)[i] for i in ids) for a in (challenger, control)})
    pending = any(value != 'COMPLETE' for value in status.values())
    return dict(status='PENDING/INCOMPLETE' if pending else 'TECHNICAL-ONLY' if table.synthetic else 'COMPLETE',
                arms=status, contrasts=contrasts,
                acquisition=curves, survival=survival, common_cohorts=common)
