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
                if getattr(a,'domain',None) and a.domain!=domain:continue
                probe(type(a)(**dict(vars(a), incoming=f'b-s{seed}', domain=domain)))
        return
    from kit.m2_handoff import scoring_identity
    scoring_machine=scoring_identity(a.out, a.score_gpu, a.standin)
    if getattr(a,'measurement',None)!='reduce':os.environ['CUDA_VISIBLE_DEVICES']=prepared_index(a.score_gpu,a.standin)
    for seed in seeds:
        for domain in doc['domains']:
            if getattr(a,'domain',None) and a.domain!=domain:continue
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
            measurement=getattr(a,'measurement',None)
            part=a.out/'.lane-parts'/path.relative_to(a.out)
            if measurement in ('before','after'):
                index=('before','after').index(measurement);ref=refs[index]
                measured=part.with_suffix('.'+measurement+'.json')
                bind(measured,nll(ref['model'],targets,a.standin,ref.get('receipt')))
                __import__('kit.m2_cost',fromlist=['record']).record(measured,start,a.standin,inputs['assignment'])
                continue
            measurements=([read(part.with_suffix('.'+name+'.json')) for name in ('before','after')] if measurement=='reduce' else
                [nll(r['model'],targets,a.standin,r.get('receipt')) for r in refs])
            bind(path, dict(inputs=inputs, before=measurements[0], after=measurements[1],
                drift=measurements[1]['token_mean']-measurements[0]['token_mean'],
                seconds=time.monotonic()-start,gpu_hours=__import__('kit.m2_cost',fromlist=['hours']).hours([part.with_suffix('.'+name+'.json') for name in ('before','after')],a.standin,inputs['assignment']) if measurement=='reduce' else 0 if a.standin else (time.monotonic()-start)/3600))


def configure(p):
    from kit.m2_select import configure as selectors
    selectors(p)
    p.add_argument('--domain',choices=DOMAINS);p.add_argument('--measurement',choices=('before','after','reduce'))
    p.add_argument('--phase',choices=('all','train','score'),default='all')
    p.add_argument('--action', choices=('drift','plasticity'), required=True)
    p.add_argument('--score-gpu', default='0')
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--trainer-python', default=sys.executable)
    p.add_argument('--inference-python', default=sys.executable)
if __name__ == '__main__': raise SystemExit(entry(job, configure))
