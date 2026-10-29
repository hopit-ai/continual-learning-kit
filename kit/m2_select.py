"""Deterministic selectors for plain scripts. No polling, submission or internal scheduling."""
import hashlib
from kit.m2_seed_policy import SEEDS


def slot(value):
    try:
        i,n=map(int,value.split('/'))
        if n < 1 or not 0 <= i < n: raise ValueError()
        return i,n
    except (ValueError,AttributeError):
        raise ValueError('--node-slot must be zero-based i/N, 0 <= i < N') from None


def selected(a, *, stage='', arm='', seed=None):
    if getattr(a,'seed',None) is not None and seed is not None and a.seed != seed: return False
    if getattr(a,'select_arm',None) and arm != a.select_arm: return False
    if getattr(a,'select_stage',None) and stage != a.select_stage: return False
    i,n=slot(getattr(a,'node_slot','0/1'))
    key=f'{stage}|{arm}|{seed}'
    return int(hashlib.sha256(key.encode()).hexdigest(),16)%n == i


def configure(p):
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--select-arm')
    p.add_argument('--select-stage')
    p.add_argument('--node-slot',default='0/1')
    p.add_argument('--mode',choices=('all','train','score'),default='all')
