#!/usr/bin/env python3
"""Frozen speculative identities and technical prerequisites for X tier 2.

Execution never decides efficacy. Both possible A/P2 stacks are materialised;
the manager reader binds exactly one after the tier-1 evidence is resolved.
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *


COMPARISONS = {
    'stack-drift': dict(signal='drift', control='drift', challenger='drift', control_p2=False, challenger_p2=True, primary='gain_auc'),
    'stack-plasticity': dict(signal='plasticity', control='standalone', challenger='standalone', control_p2=False, challenger_p2=True, primary='gain_auc'),
    'b': dict(signal='drift', control='drift', challenger='adaptive', control_p2=False, challenger_p2=False, primary='protected_mean'),
    'b-p2': dict(signal='drift', control='drift', challenger='adaptive', control_p2=True, challenger_p2=True, primary='protected_mean'),
    'incremental': dict(signal='both', control='drift', challenger='incremental', control_p2=False, challenger_p2=False, primary='protected_mean'),
    'incremental-p2': dict(signal='both', control='drift', challenger='incremental', control_p2=True, challenger_p2=True, primary='protected_mean'),
}
ARMS2 = tuple('t2-'+name+'-'+side for name in COMPARISONS for side in ('control', 'challenger'))


def identity(arm):
    require(arm in ARMS2, 'unregistered tier-2 arm')
    name, side = arm[3:].rsplit('-', 1)
    spec = COMPARISONS[name]
    return dict(comparison=name, side=side, allocation=spec[side], p2=spec[side+'_p2'])


def readiness(out, doc, name):
    """Resolved technical failure skips; missing/unfinished evidence stays pending."""
    spec = COMPARISONS[name]
    if len(doc['domains']) == 1:
        return 'DEGENERATE'
    if not (out/'support.json').exists():
        return 'PENDING'
    support = read(out/'support.json')
    signals = ('drift', 'plasticity') if spec['signal'] == 'both' else (spec['signal'],)
    for signal in signals:
        state = support[signal]['status']
        if state != 'READY':
            return 'PENDING' if state == 'PENDING' else 'UNINFORMATIVE'
    if spec['control_p2'] or spec['challenger_p2']:
        if doc.get('p_variant') != 'p2-l2-init':
            return 'DEFERRED_P2'
        if (out/'dispositions/p.json').exists() and read(out/'dispositions/p.json')['status'] == 'DEFERRED':
            return 'DEFERRED_P2'
        if not (out/'p2/calibration.json').exists():
            return 'PENDING'
        from kit.x_p import calibration
        try:
            calibration(out, doc)
            from kit.x_p2_feasibility import validate_local
            missing=False
            for seed in SEEDS:
                root=out/'p2/feasibility'/f's{seed}'
                if (root/'failure.json').exists() or (root/'terminal-failure.json').exists():
                    return 'DEFERRED_P2'
                if not (root/'receipt.json').exists():
                    missing=True
                else:validate_local(out,doc,seed)
            if missing:return 'PENDING'
        except (ValueError, KeyError, OSError, TypeError):
            return 'DEFERRED_P2'
    return 'READY'


def tier1_return(out):
    require((out/'comparison-tables.json').exists(),'tier-1 node rows pending')
    require((out/'collect-complete.json').exists(),'X12 tier-1 collection pending')
    marker=read(out/'collect-complete.json')
    archive=__import__('kit.m2_paths',fromlist=['local']).local(marker.get('archive',out/'addon-return.tar.gz'))
    require(archive.is_file(),'X12 tier-1 archive missing')
    require(sha(archive)==marker['archive_sha256'],'X12 tier-1 archive changed')


def plan(out, doc):
    tier1_return(out)
    result = dict(schema='message2-tier2.v1', binding_sha256=sha(out/'binding.json'),
        comparisons={name: dict(spec, status=readiness(out, doc, name),
            controls=[f'x-t2-{name}-control-s{s}' for s in SEEDS],
            challengers=[f'x-t2-{name}-challenger-s{s}' for s in SEEDS],
            seeds=list(SEEDS), stages=6) for name, spec in COMPARISONS.items()},
        execution_max_stages=36, bound_max_stages=30,
        note='Both stacking candidates execute before manager binding; no control stage is shared.',
        technical_synthetic=doc['technical_synthetic'])
    bind(out/'tier2/plan.json', result)
    return result


def job(a):
    doc = context(a.out, a.standin)
    if a.action == 'plan':
        plan(a.out, doc)
    elif a.action=='measure':
        from kit.x_tier2_run import measure
        spec=identity(a.arm)
        from kit.m2_assignment import active
        proof=active(a.out,a.score_gpu,a.standin)
        if proof:require(proof['family']=='x-tier2.'+spec['comparison'] and proof['seed']==a.seed,'UNVERIFIED: tier-2 signal assignment')
        measure(a,doc,a.arm,a.seed,a.step,spec['allocation']=='adaptive')
    else:
        # Shared smoke/reader code imports these identities. Load the node-only
        # runner (and its Screen S generation dependencies) only for execution.
        from importlib import import_module
        import_module('kit.x_tier2_run').comparison(a, doc, a.action)


def configure(p):
    from kit.m2_select import configure as selectors
    selectors(p)
    p.add_argument('--action', choices=('plan','measure', *COMPARISONS), required=True)
    p.add_argument('--arm',choices=ARMS2);p.add_argument('--step',type=int,choices=(10,20,30,40))
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--score-gpu', default='0')
    p.add_argument('--sdpo', type=Path)
    p.add_argument('--tokenizer')
    p.add_argument('--panel-root', type=Path)
    p.add_argument('--trainer-python', default=sys.executable)
    p.add_argument('--inference-python', default=sys.executable)


if __name__ == '__main__':
    raise SystemExit(entry(job, configure))
