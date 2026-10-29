"""Frozen rung A shares and fixed task positions; integer replay apportionment."""
from collections import Counter
import math
from kit.x_common import *


def incumbent(configuration):
    weights = dict(ifeval=4)
    if configuration in ('broad', 'both'): weights.update(gsm8k=2, arc=2)
    if configuration in ('earlier', 'both'): weights['finqa'] = 4
    return {k: v/sum(weights.values()) for k, v in weights.items()}


def shares(scores, configuration):
    fixed = incumbent(configuration)
    require(set(scores) == set(fixed), 'every allocated domain requires a signal')
    require(all(v is not None and math.isfinite(v) for v in scores.values()), 'missing/nonfinite signal cannot be zero-imputed')
    positive = {k: max(0., v) for k, v in scores.items()}
    total = sum(positive.values())
    return ({k: .1/len(fixed)+.9*positive[k]/total for k in fixed} if total else fixed)


def apportion(weights, slots):
    require(all(v > 0 for v in weights.values()) and abs(sum(weights.values())-1) < 1e-9, 'invalid shares')
    counts = {k: math.floor(v*slots) for k, v in weights.items()}
    order = sorted(weights, key=lambda k: (-(weights[k]*slots-counts[k]), k))
    for key in order[:slots-sum(counts.values())]: counts[key] += 1
    return counts


def schedule(weights, configuration, seed, start):
    """Periodic hash-selected reservations enforce ALL rolling eight-update windows.

    The remaining slots are hash ordered, preserving exact largest-remainder totals.
    start is 10 for drift, 0 for standalone plasticity; updates are zero indexed.
    """
    from kit.next_slots import MAPS
    positions = sorted(p-1 for pp in MAPS[configuration].values() for p in pp)
    require(start in (0, 10), 'wrong allocation window')
    total = (40-start)*len(positions)
    counts = apportion(weights, total)
    result = {}
    for domain in sorted(weights):
        phase = int(digest(['addon-slots-v1', seed, domain]), 16) % 8
        for update in range(start+phase, 40, 8):
            available = [p for p in positions if (update, p) not in result]
            position = min(available, key=lambda p: digest(['addon-slots-v1', seed, domain, update, p]))
            result[update, position] = domain
    used = Counter(result.values())
    require(all(used[k] <= counts[k] for k in weights), 'registered rolling coverage infeasible')
    tokens = [(d, i) for d in sorted(weights) for i in range(counts[d]-used[d])]
    tokens.sort(key=lambda x: (digest(['addon-slots-v1', seed, 'draw', *x]), x))
    free = [(u, p) for u in range(start, 40) for p in positions if (u,p) not in result]
    free.sort(key=lambda x: (digest(['addon-slots-v1', seed, 'position', *x]), x))
    for position, (domain, _) in zip(free, tokens): result[position] = domain
    require(Counter(result.values()) == counts, 'allocation totals changed')
    for u in range(start, 33):
        require(set(weights) <= {d for (v,p), d in result.items() if u <= v < u+8}, 'rolling coverage failed')
    return [dict(update=u+1, position=p+1, domain=result[u,p]) for u,p in sorted(result)]


def compose(task, pools, configuration, seed, tokenizer, scores=None, start=0):
    from kit.next_slots import compose as fixed_compose, draw
    # Caller supplies the exact frozen M3 task and admitted pool roots.
    fixed, audit = fixed_compose(task, pools['generic'], pools['root'], configuration, seed, tokenizer, task_name='contract_nli')
    if scores is None: return fixed, dict(incumbent=audit)
    weights = shares(scores, configuration)
    slots = schedule(weights, configuration, seed, start)
    pool_rows = {'ifeval': pools['generic']}
    for domain, filename in (('gsm8k','gsm8k'), ('arc','arc'), ('finqa','earlier')):
        if domain in weights: pool_rows[domain] = rows(pools['root']/(filename+'.jsonl'))
    prefix = Counter()
    if start:
        # Match the exact incumbent's source order, including broad alternation.
        for r in fixed[:start*32]:
            if r.get('p3b_replay'):
                source = r['extra_info']['next_pool']
                prefix[{'generic':'ifeval', 'earlier':'finqa'}.get(source, source)] += 1
    count = Counter(r['domain'] for r in slots)
    skipped, pool_audit, draws = [], {}, {}
    for domain in weights:
        values = draw(pool_rows[domain], seed, prefix[domain]+count[domain], tokenizer, skipped, domain, pool_audit)
        draws[domain] = iter(values[prefix[domain]:])
    result = list(fixed)
    for slot in slots:
        row = next(draws[slot['domain']])
        row['extra_info']['p3b_task_source'] = 'contract_nli'
        result[(slot['update']-1)*32+slot['position']-1] = row
    require(result[:start*32] == fixed[:start*32], 'ten-update prefix changed')
    for i, row in enumerate(fixed):
        if not row.get('p3b_replay'): require(result[i] == row, 'task exposure/position changed')
    return result, dict(shares=weights, counts=dict(count), slots=slots, pool_selection=pool_audit,
        skipped_draws=skipped, prefix_draws=dict(prefix), incumbent=audit,
        salt='addon-slots-v1', method='largest remainder; ties by domain; periodic rolling coverage, hash-ordered remainder')
