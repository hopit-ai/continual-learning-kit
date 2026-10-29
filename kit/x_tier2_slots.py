"""Tier-2 windows, preserving consumed rows, task exposure and rolling coverage."""
from collections import Counter
import math
from kit.x_common import *
from kit.x_slots import shares, apportion


def window(weights, configuration, seed, start, end):
    from kit.next_slots import MAPS
    positions = sorted(p-1 for pp in MAPS[configuration].values() for p in pp)
    require(start in (0, 10, 20, 30) and end in (10, 20, 30, 40) and start < end, 'invalid tier-2 allocation window')
    counts = apportion(weights, (end-start)*len(positions))
    reserved = {}
    for domain in sorted(weights):
        # Absolute phases persist across adaptive windows. A reservation never
        # shifts at an EMA refresh, including every cross-boundary rolling window.
        phase = int(digest(['addon-tier2-slots-v1', seed, domain]), 16) % 8
        for update in range(start+(phase-start)%8, end, 8):
            free = [p for p in positions if (update, p) not in reserved]
            p = min(free, key=lambda p: digest(['addon-tier2-slots-v1', seed, domain, update, p]))
            reserved[update, p] = domain
    used = Counter(reserved.values())
    require(all(used[d] <= counts[d] for d in weights), 'rolling allocation infeasible')
    tokens = [(d, i) for d in sorted(weights) for i in range(counts[d]-used[d])]
    tokens.sort(key=lambda t: (digest(['addon-tier2-slots-v1', seed, 'draw', start, *t]), t))
    free = [(u,p) for u in range(start,end) for p in positions if (u,p) not in reserved]
    free.sort(key=lambda t: (digest(['addon-tier2-slots-v1', seed, 'position', start, *t]), t))
    for position, (domain, _) in zip(free, tokens):
        reserved[position] = domain
    require(Counter(reserved.values()) == counts, 'tier-2 largest remainder changed')
    return [dict(update=u+1, position=p+1, domain=reserved[u,p]) for u,p in sorted(reserved)]


def compose(doc, seed, tokenizer, current, scores, start, end):
    from kit.next_slots import draw
    weights = shares(scores, doc['configuration'])
    slots = window(weights, doc['configuration'], seed, start, end)
    roots = doc['assets']['inputs']
    pool = Path(roots['pools']['path'])
    pools = dict(ifeval=[dict(r, p3b_replay=True) for r in rows(Path(roots['replay']['path'])/'replay.jsonl')])
    for domain, filename in (('gsm8k','gsm8k'), ('arc','arc'), ('finqa','earlier')):
        if domain in weights:
            pools[domain] = rows(pool/(filename+'.jsonl'))
    prefix = Counter()
    for row in current[:start*32]:
        if row.get('p3b_replay'):
            source = row['extra_info']['next_pool']
            prefix[{'generic':'ifeval','earlier':'finqa'}.get(source,source)] += 1
    counts = Counter(r['domain'] for r in slots)
    skipped, audit, draws = [], {}, {}
    for domain in weights:
        values = draw(pools[domain], seed, prefix[domain]+counts[domain], tokenizer, skipped, domain, audit)
        draws[domain] = iter(values[prefix[domain]:])
    result = list(current)
    for slot in slots:
        row = next(draws[slot['domain']])
        row['extra_info']['p3b_task_source'] = 'contract_nli'
        result[(slot['update']-1)*32+slot['position']-1] = row
    require(result[:start*32] == current[:start*32], 'tier-2 consumed prefix changed')
    for before, after in zip(current,result):
        require(bool(before.get('p3b_replay')) == bool(after.get('p3b_replay')) and
                (before.get('p3b_replay') or before == after), 'tier-2 task exposure or replay mask changed')
    # Includes transitions from the common prefix and every previous EMA window.
    for update in range(max(0,start-7), end-7):
        seen = { {'generic':'ifeval','earlier':'finqa'}.get(r['extra_info']['next_pool'],r['extra_info']['next_pool'])
                for r in result[update*32:(update+8)*32] if r.get('p3b_replay') }
        require(set(weights) <= seen, 'tier-2 cross-window rolling coverage failed')
    return result, dict(start=start,end=end,shares=weights,counts=dict(counts),slots=slots,
        prefix_draws=dict(prefix),pool_selection=audit,skipped_draws=skipped,salt='addon-tier2-slots-v1')


def ema(previous, raw):
    require(previous is None or set(previous) == set(raw), 'EMA domains changed')
    require(all(v is not None and math.isfinite(v) for v in raw.values()),'nonfinite drift cannot be zero-imputed')
    require(previous is None or all(math.isfinite(v) and v>=0 for v in previous.values()),'invalid previous positive EMA')
    positive = {d:max(0.,v) for d,v in raw.items()}
    return positive if previous is None else {d:.5*previous[d]+.5*positive[d] for d in raw}
