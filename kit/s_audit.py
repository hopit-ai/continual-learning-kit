#!/usr/bin/env python3
"""Exposure proof against every ancestry pool, full candidate lists and every panel."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *
from kit.s_inputs import load


def required_roles(spec, ident):
    """Each receipt attests its own closed ancestry, including this stage's pool."""
    if spec['nodes'][ident]['root']:
        return set()
    stages = [c for c in spec['stages'] if ident in c['checkpoints'].values()]
    require(len(stages) == 1, 'exposure node must belong to exactly one stage: '+ident)
    roles = {'v4-finqa-training'}
    if stages[0]['tier'] == 2:
        roles |= {'b-ifeval-candidates', 'broad-gsm8k-candidates', 'b-second-task-training'}
    return roles


def overlaps_ancestry(candidates, ancestry):
    """Only exact identities establish dev exposure; near matches exclude rows."""
    from kit.p3b_replay_data import normal
    ids = {str(r['id']) for r in ancestry}
    exact = {normal(r['prompt']) for r in ancestry}
    for row in candidates:
        if str(row['id']) in ids or normal(row['prompt']) in exact:
            return True
    return False


def exposure_audit(spec, sources, standin):
    from kit.next_pools import exclude
    pools, missing = [], []
    seen_pools = set()
    evidence = {}
    for ident, node in spec['nodes'].items():
        proof = node.get('exposure')
        if not proof:
            missing.append(ident+': missing complete exposure receipt')
            continue
        receipt = read(checked(proof))
        evidence[ident] = proof['sha256']
        if not (receipt.get('complete') is True and receipt.get('parent') == node['parent']
                and receipt.get('checkpoint_identity') == digest(node['checkpoint'])):
            missing.append(ident+': incomplete or mismatched ancestry receipt')
            continue
        node_pools = [(ref, rows(checked(ref))) for ref in receipt['pools']]
        roles = {ref['role'] for ref, rr in node_pools if rr}
        for role in sorted(required_roles(spec, ident)-roles):
            missing.append(ident+': missing '+role)
        for ref, rr in node_pools:
            key = (ref['role'], ref['sha256'])
            if key not in seen_pools:
                pools.append((ref['role'], rr))
                seen_pools.add(key)
    # Candidate, not merely accepted/trained memberships, must be proven complete.
    roles = {role for role, _ in pools}
    for role in ('b-ifeval-candidates', 'broad-gsm8k-candidates', 'v4-finqa-training'):
        if role not in roles:
            missing.append('missing '+role)
    if not standin:
        if len({r['id'] for role, rr in pools if role == 'b-ifeval-candidates' for r in rr}) != 2048:
            missing.append('B candidate inventory must contain 2048 rows')
        if len({r['id'] for role, rr in pools if role == 'broad-gsm8k-candidates' for r in rr}) != 1024:
            missing.append('GSM8K candidate inventory must contain 1024 rows')
    panels = [r for ref in spec['panels'].values() for r in rows(checked(ref))]
    ancestry = [r for _, rr in pools for r in rr]
    excluded_from = panels+ancestry
    result = dict(evidence=evidence, missing=missing, domains={})
    # Any incompleteness invalidates proof for all domains; no best-effort pass.
    for domain in DOMAINS:
        candidate = load(domain, sources[domain], standin)
        ordered = sorted(candidate, key=lambda r: (hashlib.sha256(('screen-s-v1|'+domain+'|'+r['id']).encode()).hexdigest(), r['id']))
        # If any v4 pool used FinQA dev, the entire domain is uninformative.
        used_dev = domain == 'finqa' and any(role == 'v4-finqa-dev' for role, _ in pools)
        if domain == 'finqa' and not used_dev:
            # Check ancestry independently: a panel match may otherwise mask a
            # second match to a mislabeled training pool. No source-to-source
            # de-duplication can count as ancestry exposure here.
            used_dev = overlaps_ancestry(ordered, ancestry)
        kept, excluded = exclude(ordered, excluded_from)
        result['domains'][domain] = dict(passed=not missing and not used_dev,
            reason=missing or (['v4 used FinQA dev'] if used_dev else []),
            eligible=kept if not missing and not used_dev else [], exclusions=excluded,
            source_sha256=sources[domain]['sha256'])
    return result


def job(a):
    doc = context(a.out, a.standin)
    result = exposure_audit(doc['spec'], doc['sources'], a.standin)
    bind(a.out/'audit.json', dict(result, technical_synthetic=a.standin))
    print('Audit:', {d: v['passed'] for d, v in result['domains'].items()})

if __name__ == '__main__':
    raise SystemExit(entry(job, lambda p: None))
