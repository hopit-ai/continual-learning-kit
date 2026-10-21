#!/usr/bin/env python3
"""Apply C's amended W ranking to complete unreset B evidence, using B's existing bars."""
from pathlib import Path
import sys
if __package__ in (None, ''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.next_common import *
from kit.p3b_read import load,analyse,lower
import statistics


def job(a):
    rows,files=load(a.source); result=analyse(rows,a.bootstrap)
    required={(f'{route}-{r}',seed,step,panel) for r in RECIPES for seed in SEEDS for route in ('standalone','sequence')
        for panel in ('math_screen','finqa','ifeval','knowledge','math') for step in ((0,10,20,30,40) if panel=='math_screen' else (0,20,40))}
    observed={(r['arm'],int(r['seed']),int(r['step']),r['panel']) for r in rows if int(r['cap'])==2048}
    if not required<=observed:raise NotReady('incomplete unreset B evidence; W remains pending')
    synthetic=any(json.loads(b).get('technical_synthetic',False) for name,b in files.items() if name.endswith(('identity.json','summary.json')))
    require(not synthetic or a.standin,'W evidence type mismatch')
    from kit.c_bind import choose
    decision=choose(result)
    ranked=decision['ranking']
    value=dict(recipe=decision['W'],complete_unreset=True,provisional=decision['provisional'],ranking=ranked,
        b_general_margins={r['recipe']:dict(stage1=r['stage1_vs_base'],stage2=r['stage2_retention']) for r in result['sequence'] if r['panel']=='math_screen'},
        tier2_interpretation='Forgetting and recipe comparison only; limited by headroom (base 89/100)',source_sha256=s.digest(rows),technical_synthetic=a.standin)
    bind(a.out/'W.json',value);print('W:',value['recipe'],'provisional:',value['provisional'])


def configure(p):
    p.add_argument('--source',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=10000)
if __name__ == '__main__':raise SystemExit(entry(job,configure,'next-W'))
