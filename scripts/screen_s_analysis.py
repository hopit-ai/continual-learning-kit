#!/usr/bin/env python3
"""Screen S preregistered manager analysis. Reads a compact archive only; stdlib."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import tarfile

SEEDS = (101, 102, 103)
DOMAINS = ('ifeval', 'gsm8k', 'arc', 'finqa')


def tau_b(x, y):
    """Exact ties, no jitter; a constant marginal has undefined tau-b."""
    concordant = discordant = tx = ty = 0
    for i in range(len(x)):
        for j in range(i):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if not dx and not dy:
                continue
            if not dx:
                tx += 1
            elif not dy:
                ty += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant+discordant+tx)*(concordant+discordant+ty))
    return (concordant-discordant)/denom if denom else None


def empirical_lower(draws):
    """Frozen inverse empirical CDF: draw rank 250 (one-based), not 251."""
    if len(draws)!=10000:raise ValueError('exactly 10000 bootstrap draws required')
    return sorted(draws)[249]


def aggregate(cells):
    """Equal cells within seed; bootstrap whole seed chains, never domains/steps."""
    if any(seed not in SEEDS for seed, _ in cells):
        raise ValueError('unregistered seed chain')
    means = {}
    for seed in SEEDS:
        values = [v for s, v in cells if s == seed]
        means[seed] = sum(values)/len(values) if values else None
    if any(v is None for v in means.values()):
        return dict(mean=None, per_seed=means, lower=None, positive=None, status='INCONCLUSIVE')
    values = list(means.values())
    rng = random.Random(20261025)
    draws = sorted(sum(rng.choices(values, k=len(SEEDS)))/len(SEEDS) for _ in range(10000))
    # Inverse empirical CDF at .025: nearest-rank, one-sided 97.5% lower bound.
    lower = empirical_lower(draws)
    mean = sum(values)/len(SEEDS)
    positive = sum(v > 0 for v in values)
    return dict(mean=mean, per_seed=means, lower=lower, positive=positive,
                status='PASS' if mean >= .5 and positive == len(SEEDS) and lower > 0 else 'FAIL')


def test_cells(cells, signal, tier1_fallback=True):
    descriptive = None
    if signal == 'forward' and tier1_fallback:
        required = {(1, recipe, 40, seed) for recipe in ('D', 'F') for seed in SEEDS}
        present = {(c['tier'], c['recipe'], c['step'], c['seed']) for c in cells}
        if not required <= present:
            tier1 = [c for c in cells if c['tier'] == 1]
            descriptive = test_cells(tier1, signal, tier1_fallback=False)
            cells = [c for c in cells if c['tier'] != 1]
    valid, baseline, omissions, cell_taus = [], [], [], []
    seen = set()
    for cell in cells:
        key = (cell['tier'], cell['recipe'], cell['stage'], cell['step'], cell['seed'])
        if key in seen:
            raise ValueError('duplicate cell')
        seen.add(key)
        if cell['tier'] not in (1, 2) or cell['stage'] != ('base-to-finqa' if cell['tier'] == 1 else 'finqa-to-second-task') or cell['seed'] not in SEEDS or cell['recipe'] not in ('D', 'F') or cell['step'] not in (20, 40):
            raise ValueError('unregistered primary cell')
        expected = DOMAINS[:3] if cell['tier'] == 1 else DOMAINS
        data = [d for d in expected if cell['domains'].get(d, {}).get('audit_passed') is True
                and cell['domains'][d].get('monitor_n', 0) >= 64
                and cell['domains'][d].get('loss') is not None
                and cell['domains'][d].get(signal) is not None
                and (signal != 'plasticity' or cell['domains'][d].get('train_n', 0) == 128)]
        if len(data) < 3:
            omissions.append(dict(cell=key, reason='fewer than three informative domains'))
            continue
        vals = [cell['domains'][d] for d in data]
        if any(not math.isfinite(float(v[k])) for v in vals for k in (signal, 'loss')):
            raise ValueError('nonfinite measurement')
        result = tau_b([v[signal] for v in vals], [v['loss'] for v in vals])
        level = tau_b([v['start_nll'] for v in vals], [v['loss'] for v in vals]) if all(v['start_nll'] is not None and math.isfinite(float(v['start_nll'])) for v in vals) else None
        if result is None or (signal != 'plasticity' and level is None):
            omissions.append(dict(cell=key, reason='undefined tau/baseline'))
            continue
        valid.append((cell['seed'], result))
        cell_taus.append(dict(cell=key, tau=result, difficulty=level, informative_domains=data))
        if level is not None:
            baseline.append((cell['seed'], level))
    result = aggregate(valid)
    result.update(cells=len(valid), cell_taus=cell_taus, omitted=omissions, difficulty=aggregate(baseline))
    # Partial tier-1 exports are descriptive only. Tier 2 must always supply
    # both recipes and all three seeds, even if seed aggregation itself is defined.
    expected = {(tier, recipe, step, seed) for tier in (1, 2) for recipe in ('D', 'F')
                for step in ((40,) if signal == 'forward' else (20, 40)) for seed in SEEDS}
    present = {(c['tier'], c['recipe'], c['step'], c['seed']) for c in cells}
    if signal == 'forward':
        if not tier1_fallback:
            expected = {k for k in expected if k[0] == 1}
        elif not any(t == 1 for t, _, _, _ in present):
            expected = {k for k in expected if k[0] == 2}
    missing = expected-present
    result['missing_cells'] = sorted(missing)
    if descriptive is not None:
        result['tier1_descriptive'] = descriptive
    if omissions or missing:
        result['status'] = 'INCONCLUSIVE'
    if signal in ('drift', 'forward') and result['status'] == 'PASS':
        if result['difficulty']['mean'] is None:
            result['status'] = 'INCONCLUSIVE'
        elif result['mean'] <= result['difficulty']['mean']:
            result['status'] = 'FAIL'
    return result


def analyze(doc, standin=False):
    if doc.get('technical_synthetic') and not standin:
        raise ValueError('technical_synthetic archive requires --standin')
    association = test_cells(doc['association'], 'drift')
    forward = test_cells(doc['forward'], 'forward')
    plasticity = test_cells(doc['association'], 'plasticity')
    drift_ok = association['status'] == forward['status'] == 'PASS'
    plasticity_ok = plasticity['status'] == 'PASS'
    return dict(drift_association=association, drift_forward=forward, plasticity=plasticity,
                qualifies=dict(drift=drift_ok, plasticity=plasticity_ok),
                decision='DRIFT_FIRST' if drift_ok else 'PLASTICITY' if plasticity_ok else 'NO_SIGNAL',
                technical_synthetic=doc.get('technical_synthetic', False))


def load_archive(path, standin=False):
    with tarfile.open(path, 'r:gz') as tar:
        members = tar.getmembers()
        if sum(m.size for m in members) > 28*1024*1024:
            raise ValueError('compact archive expands beyond 28 MiB')
        if any(not m.isfile() or m.size > 28*1024*1024 for m in members):
            raise ValueError('invalid compact archive member')
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError('duplicate archive member')
        manifest = json.load(tar.extractfile('screen-s/manifest.json'))
        if set(names) != {'screen-s/manifest.json'} | set(manifest['files']):
            raise ValueError('archive membership differs')
        raw = {}
        for name, expected in manifest['files'].items():
            value = tar.extractfile(name).read()
            if hashlib.sha256(value).hexdigest() != expected:
                raise ValueError('archive hash mismatch')
            raw[name] = value
        doc = json.loads(raw['screen-s/measurements.json'])
        registration = json.loads(raw['screen-s/registration.json'])
        code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if registration['code']['scripts/screen_s_analysis.py'] != code:
            raise ValueError('analysis differs from frozen registration')
        if manifest['technical_synthetic'] != doc['technical_synthetic']:
            raise ValueError('synthetic archive mismatch')
        if doc['technical_synthetic'] and not standin:
            raise ValueError('technical_synthetic archive requires --standin')
        from kit.m2_assignment import validate_files
        validate_files({k.removeprefix('screen-s/'):v for k,v in raw.items()},require_assignment=not standin)
        from kit.p3a_common import NotReady
        binding=json.loads(raw.get('screen-s/binding.json',b'{}'))
        assigned='screen-s/scoring-assignment.json' in raw
        for stage in binding.get('spec',{}).get('stages',[]):
            for node in (stage['incoming'],stage['checkpoints']['20'],stage['checkpoints']['40']):
                namespace=f"screen-s-s{stage['seed']}/" if assigned else ''
                expected=f"screen-s/accuracy/{namespace}{node}.json"
                if expected not in raw:raise NotReady('knowledge format-validity coverage PENDING')
        for name,value in raw.items():
            if name.startswith('screen-s/accuracy/') and name.endswith('.json'):
                failures=json.loads(value).get('strict_format_failures')
                if not failures or failures.get('knowledge')!=0:raise NotReady('FORMAT-CONFOUNDED: Screen S knowledge endpoint; affected decisions PENDING')
        return doc


def summary(result):
    lines = ['Screen S — replay-signal validity',
             'TECHNICAL SYNTHETIC; no scientific decision' if result['technical_synthetic'] else 'Registered primary cohort: three seed chains']
    for key in ('drift_association', 'drift_forward', 'plasticity'):
        r = result[key]
        lines.append(f"{key}: {r['status']}; tau={r['mean']}; positive seeds={r['positive']}/3; lower97.5={r['lower']}; cells={r['cells']}; difficulty={r['difficulty']['mean']}; missing={len(r['missing_cells'])}; undefined={len(r['omitted'])}")
        lines.append('  seed means: '+', '.join(f'{s}={v}' for s, v in r['per_seed'].items()))
        if 'tier1_descriptive' in r:
            d = r['tier1_descriptive']
            lines.append(f"  tier-1 forward descriptive only: cells={d['cells']}; tau={d['mean']}; missing={len(d['missing_cells'])}; undefined={len(d['omitted'])}")
    lines.append('Qualifies: '+str(result['qualifies'])+'; decision: '+result['decision'])
    lines.append('No combined score. ARC proxies knowledge; wrong-to-right is learning, never extra credit. C is descriptive only.')
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--standin', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args(argv)
    from kit.p3a_common import NotReady
    try:result = analyze(load_archive(a.archive, a.standin), a.standin)
    except NotReady as error:
        if a.output:a.output.write_text(json.dumps(dict(status='PENDING',reason=str(error)),indent=2)+'\n')
        print('PENDING:',error);return 3
    if a.output:
        a.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(summary(result))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
