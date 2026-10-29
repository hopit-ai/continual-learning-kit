#!/usr/bin/env python3
"""Score 15 v4 step-20 exports and all primary start/20/40 boundaries on one GPU."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *
from types import SimpleNamespace as NS
import csv
import subprocess


def accuracy(out, ref, gpu, finqa, standin):
    from kit.p3b_score import job as score
    identity = model(ref, standin)
    from kit.m2_assignment import cache_identity
    assignment=cache_identity(out)
    key = digest(dict(model=identity,assignment=assignment))
    # Only exact checkpoint identities share scoring work, including base across seeds.
    arm = 'screen-'+key
    folder = out/'panel-scoring'
    for panel in ('general-v1', 'finqa'):
        if standin:
            score(NS(out=folder, arm=arm, seed=101, step=20, panel_set=panel, model='standin',
                     checkpoint_receipt=None, panel_root=finqa, package_a=None, gpu=gpu, standin=True))
        else:
            require(finqa is not None, 'staged FinQA evaluation root required')
            subprocess.run([sys.executable, str(ROOT/'kit/p3b_score.py'), '--out', str(folder), '--arm', arm,
                '--seed', '101', '--step', '20', '--panel-set', panel, '--model', ref['model'],
                '--checkpoint-receipt', ref['receipt'], '--panel-root', str(finqa), '--gpu', gpu], check=True)
    values = {}
    evidence = {};sources={};format_failures={}
    for panel in ('general-v1', 'finqa'):
        path = folder/'scores'/f'{arm}-s101-k20-{panel}'/'scores.csv'
        with path.open() as f:
            rr = list(csv.DictReader(f))
        knowledge=[r for r in rr if r['panel']=='knowledge']
        if knowledge:
            require(all(r.get('incorrect_format') not in (None,'') for r in knowledge),'knowledge strict-format diagnostics missing')
            format_failures['knowledge']=sum(int(r['incorrect_format']) for r in knowledge)
        for name in {r['panel'] for r in rr}:
            members = [r for r in rr if r['panel'] == name]
            values[name] = {r['question_id']: int(r['correct']) for r in members}
        evidence[panel] = sha(path)
        prefix=str(path.parent.relative_to(out))
        sources[panel]=dict(prefix=prefix,files={name:sha(path.parent/name) for name in ('scores.csv','identity.json','manifest.json')})
    from kit.m2_assignment import active
    return dict(identity=identity,strict_format_failures=format_failures, assignment=active(out,gpu,standin), panels=values, score_sha256=evidence,source_files=sources)


def job(a):
    doc = context(a.out, a.standin)
    from kit.m2_handoff import scoring_identity
    scoring_identity(a.out, a.gpu, a.standin)
    spec = doc['spec']
    from kit.m2_assignment import cache_identity
    proof=cache_identity(a.out)
    namespace=f"{proof['family']}-s{proof['seed']}" if proof else None
    if a.action == 'v4-step20':
        slots = spec['v4_step20']
        require(len(slots) == len(inventory_slots()) and {r['slot'] for r in slots} == {r['id'] for r in inventory_slots()}, 'score all 15 step-20 exports')
        for slot in slots:
            if getattr(a,'slot',None) and slot['slot']!=a.slot:continue
            if proof:require(proof['family']=='s-inventory.'+slot['slot'],'UNVERIFIED: inventory scoring family mismatch')
            bind(a.out/'v4-step20'/(slot['slot']+'.json'), accuracy(a.out, slot['checkpoint'], a.gpu, a.finqa, a.standin))
        print('v4 step-20 scoring: 15 exports, FinQA + general-v1')
    else:
        stages=[c for c in spec['stages'] if getattr(a,'seed',None) is None or c['seed']==a.seed]
        if proof:require(proof['family']=='screen-s' and all(c['seed']==proof['seed'] for c in stages),'UNVERIFIED: Screen S family/seed mismatch')
        for ident in sorted({c['incoming'] for c in stages} | {c['checkpoints'][s] for c in stages for s in ('20', '40')}):
            ref = spec['nodes'][ident]['checkpoint']
            require(model(ref, a.standin) == doc['identities'][ident], 'panel checkpoint changed')
            bind((a.out/'accuracy'/namespace if namespace else a.out/'accuracy')/(ident+'.json'), accuracy(a.out, ref, a.gpu, a.finqa, a.standin))
        print('Primary accuracy: all incoming, step-20 and step-40 boundaries')


def configure(p):
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--slot')
    p.add_argument('--action', choices=('v4-step20', 'boundaries'), required=True)
    p.add_argument('--finqa', type=Path)
    p.add_argument('--gpu', default='0')
if __name__ == '__main__':
    raise SystemExit(entry(job, configure))
