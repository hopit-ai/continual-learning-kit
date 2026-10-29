#!/usr/bin/env python3
"""X2/X3: extend S's exposure audit without changing S partitions or registration."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *
from kit.s_freeze import partitions
from kit.s_generate import selected


def pool_paths(doc):
    assets = doc['assets']['inputs']
    root = Path(assets['pools']['path'])
    result = dict(ifeval=Path(assets['replay']['path'])/'replay.jsonl')
    if 'gsm8k' in doc['domains']: result.update(gsm8k=root/'gsm8k.jsonl', arc=root/'arc.jsonl')
    if 'finqa' in doc['domains']: result['finqa'] = root/'earlier.jsonl'
    return result


def audit(a):
    from kit.next_pools import exclude
    from kit.s_audit import overlaps_ancestry
    doc = context(a.out, a.standin)
    screen = Path(doc['screen_s'])
    require(sha(screen/'binding.json') == doc['screen_binding_sha256'] and
            sha(screen/'audit.json') == doc['screen_audit_sha256'] and
            sha(screen/'frozen/manifest.json') == doc['screen_partition_sha256'], 'Screen S evidence changed')
    old = read(screen/'audit.json')
    ancestry, missing, evidence, declared_dev = [], [], {}, False
    actual_pools = {name: rows(path) for name, path in pool_paths(doc).items()}
    for seed in SEEDS:
        ident = f'b-s{seed}'
        node = doc['spec']['nodes'][ident]
        proof = read(checked(node['exposure']))
        evidence[ident] = node['exposure']['sha256']
        if not (proof.get('complete') is True and proof.get('checkpoint_identity') == digest(node['checkpoint']) and
                proof.get('parent_identity') == m.checked_model(doc['base'],a.standin)):
            missing.append(ident+': closed incoming ancestry not proven')
        pools = [(r['role'], rows(checked(r))) for r in proof.get('pools', [])]
        roles = {role for role, rr in pools if rr}
        declared_dev = declared_dev or 'v4-finqa-dev' in roles
        for role in ('b-ifeval-candidates', 'broad-gsm8k-candidates', 'b-finqa-training'):
            if role not in roles: missing.append(ident+': missing '+role)
        if not a.standin:
            for role, n in (('b-ifeval-candidates', 2048), ('broad-gsm8k-candidates', 1024)):
                if len({r['id'] for k, rr in pools if k == role for r in rr}) != n:
                    missing.append(ident+': incomplete '+role)
        # Full B training membership is already hash-bound in this proof;
        # no P utility-response file or 256-row threshold applies to P2.
        ancestry.extend(r for _, rr in pools for r in rr)
    # Admitted pools are added explicitly, even if a B ancestor receipt omitted them.
    ancestry.extend(r for rr in actual_pools.values() for r in rr)
    panels = [r for ref in doc['spec']['panels'].values() for r in rows(checked(ref))]
    result = dict(evidence=evidence, missing=missing, domains={}, admitted_pools={k: digest(v) for k,v in actual_pools.items()})
    files = {}
    for domain in DOMAINS:
        original = partitions(screen, domain)
        passed = old['domains'][domain]['passed'] and not missing
        if domain == 'finqa':
            from kit.s_inputs import load
            used_dev = declared_dev or overlaps_ancestry(load(domain, doc['sources'][domain], a.standin), ancestry)
            passed = passed and not used_dev
        else: used_dev = False
        counts = {}
        for part, values in original.items():
            kept, dropped = exclude(values, ancestry+panels)
            # Do not repartition: preserve S's original disjoint hash order.
            path = a.out/'frozen'/f'{domain}-{part}.jsonl'
            write_rows(path, kept if passed else [])
            files[path.name] = sha(path)
            counts[part] = dict(n=len(kept) if passed else 0, exclusions=dropped)
        result['domains'][domain] = dict(passed=passed, reason=missing or (['FinQA dev exposed'] if used_dev else old['domains'][domain]['reason']), partitions=counts)
    bind(a.out/'audit.json', result)
    bind(a.out/'frozen/manifest.json', dict(files=files, audit_sha256=sha(a.out/'audit.json'),
        binding_sha256=sha(a.out/'binding.json'), original_partition_sha256=doc['screen_partition_sha256']))


def generate(a):
    doc = context(a.out, a.standin)
    if len(doc['domains']) == 1:
        bind(a.out/'support.json', dict(drift=dict(status='DEGENERATE',failures=[]),
            plasticity=dict(status='DEGENERATE',failures=[]),degenerate=True,policy='Generic-only; X3 skipped'))
        print('SKIP X3: both A arms DEGENERATE'); return
    # Reuse the unchanged greedy decoder, checkers and raw-journal restart path.
    from kit.s_generate import job as own
    for seed in SEEDS:
        if getattr(a,'seed',None) is not None and seed!=a.seed:continue
        own(type(a)(**dict(vars(a), incoming=f'b-s{seed}')))
    if getattr(a,'seed',None) is not None:return
    return reduce_support(a,doc)


def reduce_support(a,doc):
    if len(doc['domains'])==1:return generate(a)
    receipt = {}
    for signal in ('drift', 'plasticity'):
        failures = []
        for seed in SEEDS:
            for domain in doc['domains']:
                values = selected(a.out, f'b-s{seed}', domain)
                if len(values['monitor']) != 64 or (signal == 'plasticity' and len(values['train']) != 128):
                    failures.append(dict(seed=seed, domain=domain, train=len(values['train']), monitor=len(values['monitor'])))
        receipt[signal] = dict(status='UNINFORMATIVE' if failures else 'READY', failures=failures)
    bind(a.out/'support.json', dict(receipt, degenerate=len(doc['domains']) == 1,
        policy='Every allocated domain, all three seeds; no domain/seed drop, replenishment or imputed zero'))


def job(a):
    if a.action=='audit':return audit(a)
    if a.action=='reduce':return reduce_support(a,context(a.out,a.standin))
    return generate(a)


def configure(p):
    p.add_argument('--action', choices=('audit', 'generate','reduce'), required=True)
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--gpu', default='0')
if __name__ == '__main__': raise SystemExit(entry(job, configure))
