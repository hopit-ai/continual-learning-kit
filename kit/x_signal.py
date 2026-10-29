#!/usr/bin/env python3
"""Incoming/step10 token-mean drift and independent untouched-incoming CE probes."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *
from kit.m2_gpu import prepared_index
from kit.s_generate import selected
from kit.s_drift import nll


def job(a):
    doc = context(a.out, a.standin)
    from kit.m2_select import selected as select_task
    seeds=[s for s in SEEDS if select_task(a,stage=a.action,arm=a.action,seed=s)]
    state = disposition(a.out, a.action)
    if state != 'READY':
        bind(a.out/'dispositions'/(a.action+'.json'), dict(status=state))
        print('SKIP signal:', a.action, state); return
    if a.action == 'plasticity':
        from kit.s_probe import job as probe
        for seed in seeds:
            for domain in doc['domains']:
                probe(type(a)(**dict(vars(a), incoming=f'b-s{seed}', domain=domain)))
        return
    from kit.m2_handoff import scoring_identity
    scoring_machine=scoring_identity(a.out, a.score_gpu, a.standin)
    os.environ['CUDA_VISIBLE_DEVICES'] = prepared_index(a.score_gpu,a.standin)
    for seed in seeds:
        for domain in doc['domains']:
            path = a.out/'signals'/f's{seed}'/(domain+'.json')
            targets = selected(a.out, f'b-s{seed}', domain)['monitor']
            refs = [incoming(doc, seed), checkpoint(a.out, doc, 'drift', seed, 10)]
            identities = [m.checked_model(r, a.standin) for r in refs]
            inputs = dict(assignment=scoring_machine.get('assignment'),targets=digest(targets), checkpoints=identities)
            bind(path.with_suffix('.inputs.json'), inputs)
            if path.exists():
                require(read(path)['inputs'] == inputs, 'frozen drift changed'); continue
            import time
            start = time.monotonic()
            measurements = [nll(r['model'], targets, a.standin, r.get('receipt')) for r in refs]
            bind(path, dict(inputs=inputs, before=measurements[0], after=measurements[1],
                drift=measurements[1]['token_mean']-measurements[0]['token_mean'],
                seconds=time.monotonic()-start,gpu_hours=0 if a.standin else (time.monotonic()-start)/3600))


def configure(p):
    from kit.m2_select import configure as selectors
    selectors(p)
    p.add_argument('--phase',choices=('all','train','score'),default='all')
    p.add_argument('--action', choices=('drift','plasticity'), required=True)
    p.add_argument('--score-gpu', default='0')
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--trainer-python', default=sys.executable)
    p.add_argument('--inference-python', default=sys.executable)
if __name__ == '__main__': raise SystemExit(entry(job, configure))
