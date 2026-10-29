#!/usr/bin/env python3
"""Score X checkpoints with the M graders, panels and physical-GPU guard."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *
from kit.x_tier2 import ARMS2
from types import SimpleNamespace as NS


def job(a):
    doc = context(a.out, a.standin)
    from kit.m2_assignment import active
    proof=active(a.out,a.gpu,a.standin)
    family=proof['family'] if proof else None
    if proof:
        expected='x-tier2.'+a.arm[3:].rsplit('-',1)[0] if a.arm.startswith('t2-') else 'x-tier1'
        require(proof['seed']==a.seed and (a.arm=='pre' and family.startswith('x-') or family==expected),'UNVERIFIED: X contrast family/seed mismatch')
    pre_arm='x-t2-'+family.removeprefix('x-tier2.')+'-pre' if family and family.startswith('x-tier2.') else 'x-pre'
    if a.arm == 'pre':
        require(a.step == 0, 'pre-replacement boundary only')
        ref = incoming(doc, a.seed)
    else:
        state = disposition(a.out, a.arm)
        if state != 'READY': print('SKIP scoring:', a.arm, state); return
        ref = checkpoint(a.out, doc, a.arm, a.seed, a.step)
    from kit.m2_handoff import scoring_identity
    machine = scoring_identity(a.out,a.gpu,a.standin)
    # Mixed inherited bars fail closed in the reader if this differs. X-only
    # paired comparisons remain interpretable on their own frozen machine.
    match=a.out/'scoring-core-match'/f"{family}-s{a.seed}.json" if proof else a.out/'scoring-core-match.json'
    bind(match,dict(status='MATCH' if machine == doc['core_scoring_machine']['identity']
        else 'PENDING_CROSS_MACHINE',core=doc['core_scoring_machine'],observed=machine))
    if a.step == 0 and a.arm in ('control','drift','plasticity','p',*ARMS2) and (a.arm!='p' or doc.get('p_variant')=='p2-l2-init'):
        import csv
        from kit.p3b_score import write_csv
        from kit.c_score import COLUMNS
        source=a.out/'scores'/f'{pre_arm}-s{a.seed}-k0-{a.panel_set}'
        if not (source/'manifest.json').exists():job(NS(**dict(vars(a),arm='pre',step=0)))
        m.frozen_files(source)
        target=a.out/'scores'/f'x-{a.arm}-s{a.seed}-k0-{a.panel_set}'
        alias=dict(source=str(source.relative_to(a.out)),scores_sha256=sha(source/'scores.csv'),
                   identity_sha256=sha(source/'m2-identity.json'))
        bind(target/'alias.json',alias)
        if (target/'manifest.json').exists():
            m.frozen_files(target); print('SKIP incoming alias:',a.arm,a.seed,a.panel_set); return
        with (source/'scores.csv').open() as f: values=[dict(r,arm='x-'+a.arm) for r in csv.DictReader(f)]
        write_csv(target/'scores.csv',values,COLUMNS)
        atomic(target/'m2-identity.json',dict(read(source/'m2-identity.json'),arm='x-'+a.arm))
        atomic(target/'manifest.json',m.files_manifest(target,['scores.csv','m2-identity.json','alias.json']))
        print('Alias incoming by hash:',a.arm,a.seed,a.panel_set); return
    from kit import m2_score
    from types import FunctionType
    args = NS(**dict(vars(a), arm=pre_arm if a.arm=='pre' else 'x-'+a.arm))
    # Run the exact reviewed scoring function in a private namespace. Its M
    # module, globals, files and default callers remain completely unchanged.
    namespace = dict(m2_score.__dict__, binding=lambda out,standin: doc,
                     record=lambda out,binding,arm,seed,step: ref)
    score = FunctionType(m2_score.job.__code__,namespace,'x_bound_score',m2_score.job.__defaults__,m2_score.job.__closure__)
    score(args)


def configure(p):
    p.add_argument('--arm', choices=('pre', *ARMS, *ARMS2), required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--step', type=int, choices=STEPS, required=True)
    p.add_argument('--panel-set', choices=(*m.PANEL_SETS, 'hellaswag'), required=True)
    p.add_argument('--panel-root', type=Path)
    p.add_argument('--gpu', default='0')
if __name__ == '__main__': raise SystemExit(entry(job, configure))
