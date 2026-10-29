#!/usr/bin/env python3
"""Message-2 identity guard; reuse durable save/restore and TERM handling unchanged."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit import message_handoff
import platform
import subprocess


def scoring_identity(out,gpu,standin):
    require(bool(gpu) and ',' not in gpu,'one physical scoring GPU required')
    from kit.m2_assignment import active
    proof=active(out,gpu,standin)
    if proof is not None:
        bind(out/'m2-scoring-machine.json',dict(assignment_sha256=proof['assignment_sha256']))
        value=dict(uuid=proof['uuid'],host=proof['node'],environment=proof['environment'],assignment=proof)
        bind(out/'scoring-families'/f"{proof['family']}-s{proof['seed']}.json",value)
        return value
    value=dict(uuid='CPU-STANDIN',host='CPU-STANDIN',environment='CPU-STANDIN')
    if (out/'m2-scoring-machine.json').exists():
        require(json.loads((out/'m2-scoring-machine.json').read_text())==value,'UNVERIFIED: scoring host/environment/physical GPU changed')
    bind(out/'m2-scoring-machine.json',value)
    return value


def job(a):
    if a.action=='identity':
        print(scoring_identity(a.out,a.gpu,a.standin));return
    return message_handoff.job(a)


def configure(p):
    p.add_argument('--durable',type=Path,required=True);p.add_argument('--gpu',default='0')
    p.add_argument('--action',choices=('job-start','save','restore','identity','path'),required=True)
    p.add_argument('--move-aside',action='store_true')
if __name__=='__main__':raise SystemExit(entry(job,configure,'m2-handoff-{action}'))
