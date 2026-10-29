#!/usr/bin/env python3
"""Serial Screen S rows; save/restore uses message_handoff at every job boundary."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *
from types import SimpleNamespace as NS
import subprocess


def job(a):
    doc = context(a.out, a.standin)
    from kit.m2_select import selected
    def keep(incoming=None,stage=None):
        if getattr(a,'incoming',None) and a.incoming!=incoming:return False
        if stage and getattr(a,'seed',None) is not None and stage['seed']!=a.seed:return False
        return selected(a,stage=stage['id'] if stage else incoming,arm=a.action,seed=stage['seed'] if stage else None)
    if a.action == 'generate':
        from kit.s_generate import job as generate
        for incoming in sorted({c['incoming'] for c in doc['spec']['stages']}):
            if not keep(incoming):continue
            if a.standin:
                generate(NS(**vars(a), incoming=incoming))
            else:
                subprocess.run([a.inference_python, str(ROOT/'kit/s_generate.py'), '--out', str(a.out), '--incoming', incoming, '--gpu', a.gpu], check=True)
    elif a.action == 'drift':
        from kit.s_drift import job as drift
        for stage in doc['spec']['stages']:
            if not keep(stage=stage):continue
            if a.standin:
                drift(NS(**vars(a), stage=stage['id']))
            else:
                subprocess.run([a.inference_python, str(ROOT/'kit/s_drift.py'), '--out', str(a.out), '--stage', stage['id'], '--gpu', a.gpu], check=True)
    else:
        from kit.s_probe import job as probe
        from kit.s_generate import domains_for
        for incoming in sorted({c['incoming'] for c in doc['spec']['stages']}):
            if not keep(incoming):continue
            for domain in domains_for(doc['spec'], incoming):
                probe(NS(**vars(a), incoming=incoming, domain=domain))


def configure(p):
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--incoming');p.add_argument('--node-slot',default='0/1')
    p.add_argument('--select-stage');p.add_argument('--select-arm')
    p.add_argument('--phase',choices=('all','train','score'),default='all')
    p.add_argument('--action', choices=('generate', 'drift', 'probes'), required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--score-gpu', default='0')
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--trainer-python', default=sys.executable)
    p.add_argument('--inference-python', default=sys.executable)
if __name__ == '__main__':
    raise SystemExit(entry(job, configure))
