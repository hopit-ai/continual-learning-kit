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



# Count schema v1. Cell keys are the unchanged six-field scientific keys.
# Disjoint groups: [holdout, family, n, correct, historical, input, output].
# Pairs: [first cell index, second cell index, holdout-only, n, cc, cw, wc, ww].
# Paths: [ordered cell indices, acquired-then-lost, n]. No item IDs are pasted.
class CountEvidence:
    def __init__(self, cells, counts=None, capture=None):
        self.counts = counts
        self.capture = capture
        if counts is not None:
            require(counts.get('schema') == 'a2-keyed-counts.v1', 'unknown A2 count schema')
            cells = [dict(key=k, physical=p, groups=g) for k, p, g in counts['cells']]
        self.cells = cells
        self.indices = {tuple(c['key']): i for i, c in enumerate(cells)}
        self.pairs, self.paths = {}, {}
        self.used_pairs, self.used_paths = set(), set()
        if counts is not None:
            for i, j, h, *v in counts['pairs']:
                key = (i, j, h)
                require(key not in self.pairs and type(h) is bool, 'duplicate/invalid count pair')
                self.pairs[key] = v
            for indices, lost, n in counts['paths']:
                key = tuple(indices)
                require(key not in self.paths, 'duplicate count path')
                self.paths[key] = (lost, n)
        self.groups = []
        for cell in cells:
            if counts is None:
                rows = cell['rows']
                require(len({r['question_id'] for r in rows}) == len(rows), 'duplicate scored unit')
                expected = cell.get('expected_members')
                require(isinstance(expected, dict) and set(expected) == {r['question_id'] for r in rows},
                        'scoring population differs from prepared membership')
                groups = {}
                for r in rows:
                    member = expected[r['question_id']]
                    require(bool(r['holdout']) == member['holdout'] and r.get('family', '') == member['family'],
                            'holdout/family assignment differs from prepared membership')
                    # All registered units, including prompt-level IFEval, are binary.
                    require(type(r['correct']) is int and r['correct'] in (0, 1) and
                            type(r.get('historical_correct', r['correct'])) is int and
                            r.get('historical_correct', r['correct']) in (0, 1), 'non-binary scored unit')
                    key = (bool(r['holdout']), r.get('family', ''))
                    v = groups.setdefault(key, [0]*5)
                    values = [1, r['correct'], r.get('historical_correct', r['correct']),
                              r.get('input_tokens', 0), r.get('output_tokens', 0)]
                    require(all(type(x) is int and x >= 0 for x in values), 'invalid score totals')
                    for i, x in enumerate(values):
                        v[i] += x
                groups = [[*key, *v] for key, v in sorted(groups.items())]
            else:
                groups = cell['groups']
            seen = set()
            for h, f, n, correct, historical, it, ot in groups:
                require(type(h) is bool and isinstance(f, str) and (h, f) not in seen,
                        'duplicate/invalid population group')
                seen.add((h, f))
                require(all(type(x) is int and x >= 0 for x in (n, correct, historical, it, ot)) and
                        correct <= n and historical <= n, 'invalid binary population counts')
            self.groups.append(groups)
        if capture is not None:
            capture.update(schema='a2-keyed-counts.v1',
                           cells=[[c['key'], c.get('physical', {}), g] for c, g in zip(cells, self.groups)],
                           pairs=[], paths=[])

    def totals(self, key, holdout=False, family=None):
        rows = [g[2:] for g in self.groups[self.indices[tuple(key)]]
                if (not holdout or g[0]) and (family is None or g[1] == family)]
        return [sum(r[i] for r in rows) for i in range(5)]

    def summary(self, key, holdout=False, family=None):
        n, correct, historical, it, ot = self.totals(key, holdout, family)
        return dict(n=n, correct=correct, accuracy=100*correct/n if n else None,
                    historical_correct=historical, input_tokens=it, output_tokens=ot,
                    granularity_points=100/n if n else None)

    def paired(self, first, second, holdout=False):
        i, j = (self.indices[tuple(k)] for k in (first, second))
        key = (i, j, holdout)
        if self.counts is None:
            def index(cell):
                return {r['question_id']: r['correct'] for r in cell['rows'] if not holdout or r['holdout']}
            a, b = index(self.cells[i]), index(self.cells[j])
            require(a and a.keys() == b.keys(), 'unpaired or empty score denominator')
            v = [len(a), *[sum(a[k] == x and b[k] == y for k in a)
                          for x, y in ((1, 1), (1, 0), (0, 1), (0, 0))]]
        else:
            require(key in self.pairs, 'missing keyed pair counts')
            v = self.pairs[key]
        require(len(v) == 5 and all(type(x) is int and x >= 0 for x in v), 'invalid 2x2 counts')
        n, cc, cw, wc, ww = v
        require(n > 0 and n == cc+cw+wc+ww, '2x2 denominator differs')
        a, b = self.totals(first, holdout), self.totals(second, holdout)
        require(a[:2] == [n, cc+cw] and b[:2] == [n, cc+wc], '2x2 marginals differ from scores')
        if self.capture is not None and key not in self.used_pairs:
            self.capture['pairs'].append([*key, *v])
        self.used_pairs.add(key)
        return dict(n=n, delta=100*(wc-cw)/n, correct_to_wrong=cw, wrong_to_correct=wc)

    def acquired(self, keys):
        indices = tuple(self.indices[tuple(k)] for k in keys)
        n = self.totals(keys[0], True)[0]
        if self.counts is None:
            lookup = [{r['question_id']: r['correct'] for r in self.cells[i]['rows'] if r['holdout']}
                      for i in indices]
            require(all(r.keys() == lookup[0].keys() for r in lookup), 'unpaired retention path')
            lost = 0
            for item in lookup[0]:
                acquired, was_lost = False, False
                for before, after in zip(lookup, lookup[1:]):
                    acquired |= before[item] == 0 and after[item] == 1
                    was_lost |= acquired and before[item] == 1 and after[item] == 0
                lost += was_lost
        else:
            require(indices in self.paths, 'missing acquisition path counts')
            lost, denominator = self.paths[indices]
            require(denominator == n, 'acquisition denominator differs')
        require(type(lost) is int and 0 <= lost <= n, 'invalid acquired-then-lost count')
        if self.capture is not None:
            self.capture['paths'].append([list(indices), lost, n])
        self.used_paths.add(indices)
        return lost

    def finish(self):
        if self.counts is not None:
            require(self.used_pairs == self.pairs.keys() and self.used_paths == self.paths.keys(),
                    'unregistered/unused keyed counts')

def analyse(arm, cells, *, synthetic=False, confirmation=False, receipt=None, p_applicable=True,
            assignment=None, counts=None, capture=None):
    evidence = CountEvidence(cells, counts, capture)
    cells = evidence.cells
    plan = confirmation_plan(arm) if confirmation else score_plan(arm)
    by_key = {tuple(c['key']): c for c in cells}
    if len(by_key) != len(cells):
        return dict(status='UNVERIFIED', reason='duplicate scientific cells')
    missing = [name(k) for k in plan if k not in by_key]
    if counts is not None:
        require(not missing, 'missing keyed score cells')
    if missing:
        return dict(status='INCOMPLETE', missing=missing)
    require(set(by_key) == set(plan), 'unregistered score cells')
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
            paired = evidence.paired((kind, control, seed, n, panel, mode),
                                     (kind, challenger, seed, n, panel, mode),
                                     panel.startswith('2026-'))
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
    result['denominators'] = {name(k): dict(n=evidence.totals(k)[0],
        holdouts=evidence.totals(k, True)[0],
        families={f: evidence.totals(k, family=f)[0] for f in FAMILIES}) for k in plan}
    result['scores'] = {name(k): dict(all=evidence.summary(k), holdout=evidence.summary(k, True),
        family={f: evidence.summary(k, family=f) for f in FAMILIES}) for k in plan}
    if confirmation:
        for key in plan:
            if key[0] == 'confirmation':
                result['scores'][name(key)]['equal_month_granularity_points'] = 50/evidence.totals(key)[0]
    if not confirmation:
        result['retention_transitions'] = []
        for model in arm_names(arm):
            for seed in SEEDS:
                for index, month in enumerate(MONTHS, 1):
                    for mode in ('on', 'off'):
                        first = ('arrival', model, seed, index, month, mode)
                        snapshots = [first]
                        for n in BOUNDARIES:
                            if n >= index:
                                post = ('post', model, seed, n, month, mode)
                                result['retention_transitions'].append(dict(model=model, seed=seed,
                                    month=month, boundary=n, mode=mode, **evidence.paired(first, post, True)))
                                snapshots.append(post)
                        result['retention_transitions'].append(dict(model=model, seed=seed, month=month,
                            mode=mode, acquired_then_lost=evidence.acquired(snapshots),
                            n=evidence.totals(first, True)[0]))
    evidence.finish()
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
