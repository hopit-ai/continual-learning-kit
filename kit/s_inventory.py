#!/usr/bin/env python3
"""Inventory original v4 step-10 exports without creating replacements."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *


def job(a):
    doc = context(a.out, a.standin)
    slots = doc['spec']['v4_step10']
    require(len(slots) == len(inventory_slots()) and {r['slot'] for r in slots} == {r['id'] for r in inventory_slots()}, 'inventory all 15 original v4 slots')
    result = []
    for slot in slots:
        ref = slot.get('checkpoint')
        exists = ref is not None and (a.standin or (__import__('kit.m2_paths',fromlist=['local']).local(ref['model']).is_dir() and __import__('kit.m2_paths',fromlist=['local']).local(ref['receipt']).is_file()))
        result.append(dict(slot=slot['slot'], status='AVAILABLE' if exists else 'MISSING',
                           identity=model(ref, a.standin) if exists else None))
    bind(a.out/'v4-step10-inventory.json', dict(slots=result, technical_synthetic=a.standin))
    print('v4 step-10 exports:', sum(r['status'] == 'AVAILABLE' for r in result), '/15 available')

if __name__ == '__main__':
    raise SystemExit(entry(job, lambda p: None))
