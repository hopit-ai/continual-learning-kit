"""Message-3 K/CR batch schedules (registration v2): shared instruction stream and per-update row layouts.

Pure CPU code; no model, tokenizer or scheduler IO. Builder A's trainer consumes the rows
(`m3_domain`, `m3_loss`, `m3_round`, `m3_rank`, `m3_weight`); `contract_agreement` compares these
constants with `kit/m3_contract.py`.

v2 rules:
  * one shared instruction stream: next_slots' seed-only generic cycling (random.Random(seed) shuffle of
    the pool, positions 8/16/24/32) WITHOUT its length-filtering draw, so the same 160 IDs recur each stage;
  * G's extra instruction rows: hash order m3-cr-v2|seed|stage|generic-extra|id, excluding the stage's
    shared IDs and previously used extras;
  * other CR domains: hash order m3-cr-v2|seed|stage|<domain>|id, no within-stage repeats;
  * no filtered draws: every arm uses one common long-context cap; overlength rows fail preparation.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import random

SEEDS = (101, 102, 103)
STAGES = ('finqa', 'contract_nli', 'math_hard')
ARMS = ('C0', 'K', 'G', 'CR')
UPDATES = 40
TASK_ROWS = 32
RANKS = 8
GLOBAL_DIVISOR = 32
NAMESPACE = 'm3-cr-v2'
GENERIC_POSITIONS = (8, 16, 24, 32)  # next_slots.py generic map (1-based)
SHARED_GENERIC = len(GENERIC_POSITIONS)
DOMAINS = ('instruction', 'gsm8k', 'arc', 'finqa', 'contract_nli')
CR_LAYOUT = {
    'finqa': {'instruction': 4, 'gsm8k': 2, 'arc': 2},
    'contract_nli': {'instruction': 4, 'gsm8k': 2, 'arc': 2, 'finqa': 8},
    'math_hard': {'instruction': 4, 'gsm8k': 2, 'arc': 2, 'finqa': 8, 'contract_nli': 8},
}
G_GENERIC = {stage: sum(CR_LAYOUT[stage].values()) for stage in STAGES}  # 8/16/24 per update
REPLAY_LOSS = {'C0': 'CE', 'K': 'KL', 'G': 'CE', 'CR': 'CE'}
# Builder A's contract owns the common long-context limits (one cap for every arm and stage, v2 finding 1).
from kit.m3_contract import PROMPT_CAP, RESPONSE_CAP, REPROMPT_CAP, CONTEXT_CAP  # noqa: E402
# Unique accepted rows needed so no row repeats within a stage (v2 finding 5).
SHARED_IDS = UPDATES * SHARED_GENERIC  # 160
MINIMA = dict(instruction=SHARED_IDS, instruction_g=UPDATES * G_GENERIC['math_hard'],  # 960: shared 160 + 800 G extras
              gsm8k=UPDATES * 2, arc=UPDATES * 2, finqa=UPDATES * 8, contract_nli=UPDATES * 8)


class ScheduleError(ValueError):
    pass


def require(ok, message):
    if not ok: raise ScheduleError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def target_sha256(row):
    """Identity of one supervised target, shared across arms."""
    return digest(dict(prompt=row['prompt'], response=row['response']))


def domains_for(arm, stage):
    require(arm in ARMS and stage in STAGES, f'unregistered arm/stage {arm}/{stage}')
    return tuple(CR_LAYOUT[stage]) if arm == 'CR' else ('instruction',)


def rows_per_update(arm, stage):
    return TASK_ROWS if arm in ('C0', 'K') else TASK_ROWS + G_GENERIC[stage]


def prompt_cap(arm, stage):
    """Delegates to builder A's contract (manager round-3 ruling a)."""
    require(arm in ARMS and stage in STAGES, f'unregistered arm/stage {arm}/{stage}')
    from kit import m3_contract
    return m3_contract.prompt_cap(arm, stage)


def check_lengths(pools, lengths):
    """Overlength inputs fail preparation loudly; nothing is filtered out of a schedule."""
    if lengths is None: return
    over = [(d, str(r['id']), lengths[str(r['id'])]) for d, rows in pools.items() for r in rows
            if lengths.get(str(r['id']), 0) > PROMPT_CAP]
    require(not over, f'rows exceed the common {PROMPT_CAP}-token prompt cap: {over[:5]}')


def shared_stream(pool, seed):
    """next_slots generic cycling without the length filter: seed-only, identical for every stage and arm."""
    require(len(pool) >= SHARED_IDS, f'instruction pool has {len(pool)} rows; {SHARED_IDS} unique shared IDs required')
    ids = [str(r['id']) for r in pool]
    require(len(ids) == len(set(ids)), 'duplicate IDs in instruction pool')
    order = list(pool); random.Random(seed).shuffle(order)  # == kit.p3b_common.cycle(pool, seed, len(pool))
    return [dict(row=order[i], draw_index=i, pass_index=0) for i in range(SHARED_IDS)]


def hash_order(pool, seed, stage, domain, exclude=()):
    """Registered order sha256('m3-cr-v2|seed|stage|domain|id'), ID tie-break, excluded IDs removed."""
    ids = [str(r['id']) for r in pool]
    require(len(ids) == len(set(ids)), f'duplicate IDs in {domain} pool')
    skip = {str(i) for i in exclude}
    return sorted((r for r in pool if str(r['id']) not in skip),
                  key=lambda r: (sha(f"{NAMESPACE}|{seed}|{stage}|{domain}|{r['id']}"), str(r['id'])))


def stream(pool, seed, stage, domain, n, exclude=()):
    """n draws without replacement within the stage; an insufficient pool is an error, never a repeat."""
    order = hash_order(pool, seed, stage, domain, exclude)
    require(len(order) >= n, f'{domain} pool has {len(order)} eligible rows; {n} unique rows needed for stage {stage}')
    return [dict(row=order[i], draw_index=i, pass_index=0) for i in range(n)]


def geometry(n_rows):
    """Eight ranks, one row/rank/microbatch: row j goes to round j//8 on rank j%8 (A transposes to rank-major)."""
    require(n_rows % RANKS == 0, f'{n_rows} rows do not fill {RANKS} ranks')
    return [dict(round=j // RANKS, rank=j % RANKS) for j in range(n_rows)]


def local_scale(world_size=RANKS):
    """Per-row local multiplier so the rank-averaged sum is sum(rows)/32: loss/4 on eight ranks, loss/16 on two."""
    return world_size / GLOBAL_DIVISOR


def reduce_like_ddp(losses, world_size=RANKS):
    """Independent reference: rows dealt round-major to ranks, scaled locally, then averaged over ranks."""
    per_rank = [0.0] * world_size
    for j, loss in enumerate(losses): per_rank[j % world_size] += loss * local_scale(world_size)
    return sum(per_rank) / world_size


def check_reduction(losses, world_size=RANKS, tolerance=1e-12):
    require(len(losses) % world_size == 0, 'rows do not fill the ranks')
    expected = sum(losses) / GLOBAL_DIVISOR
    observed = reduce_like_ddp(losses, world_size)
    require(abs(observed - expected) <= tolerance * max(1.0, abs(expected)), f'reduction {observed} != sum/32 {expected}')
    return expected


def _row(arm, seed, stage, update, position, place, role, domain, loss, source, draw=None, stream_name=None):
    row = source if draw is None else draw['row']
    value = dict(arm=arm, seed=seed, stage=stage, update=update, position=position, round=place['round'], rank=place['rank'],
                 role=role, domain=domain, loss=loss, weight=1 / GLOBAL_DIVISOR, id=str(row['id']),
                 target_sha256=target_sha256(row))
    if draw is not None: value.update(draw_index=draw['draw_index'], pass_index=draw['pass_index'])
    if stream_name: value['stream'] = stream_name
    return value


def build(arm, seed, stage, task, pools, *, lengths=None):
    """Full 40-update schedule for one arm/seed/stage.

    task: the stage's existing 1,280-row (40x32) task schedule for this seed, shared by all arms.
    pools: {domain: rows}; lengths: optional {id: prompt_tokens} over every pool row (overlength = error).
    """
    require(arm in ARMS and seed in SEEDS and stage in STAGES, f'unregistered {arm}/{seed}/{stage}')
    require(len(task) == UPDATES * TASK_ROWS, 'task schedule must be 40 x 32')
    check_lengths(pools, lengths)
    shared = shared_stream(pools['instruction'], seed)
    extras = []
    if arm == 'G':
        extras = stream(pools['instruction'], seed, stage, 'generic-extra', UPDATES * (G_GENERIC[stage] - SHARED_GENERIC),
                        exclude=[d['row']['id'] for d in shared])
    other = {}
    if arm == 'CR':
        for domain, count in CR_LAYOUT[stage].items():
            if domain != 'instruction': other[domain] = stream(pools[domain], seed, stage, domain, UPDATES * count)
    # Consume B's native task order and its exact replacement function, without filtered replay draws.
    from kit.p3b_common import mixed
    consumed = mixed(task, pools['instruction'], seed) if arm in ('C0', 'K') else None
    replay_loss = REPLAY_LOSS[arm]
    rows = []
    for u in range(UPDATES):
        update = u + 1
        block = task[u * TASK_ROWS:(u + 1) * TASK_ROWS]
        four = shared[u * SHARED_GENERIC:(u + 1) * SHARED_GENERIC]
        layout = geometry(rows_per_update(arm, stage))
        if arm in ('C0', 'K'):
            generic = iter(four)
            for p in range(1, TASK_ROWS + 1):
                if p in GENERIC_POSITIONS:
                    require(str(consumed[u * TASK_ROWS + p - 1]['id']) == str(four[GENERIC_POSITIONS.index(p)]['row']['id']), 'B/M3 replay replacement identity differs')
                    rows.append(_row(arm, seed, stage, update, p, layout[p - 1], 'replay', 'instruction', replay_loss, None, next(generic), 'shared'))
                else:
                    rows.append(_row(arm, seed, stage, update, p, layout[p - 1], 'task', stage, 'D', consumed[u * TASK_ROWS + p - 1]))
            continue
        for p in range(1, TASK_ROWS + 1):
            rows.append(_row(arm, seed, stage, update, p, layout[p - 1], 'task', stage, 'D', block[p - 1]))
        replay = [('instruction', d, 'shared') for d in four]
        if arm == 'G':
            n = G_GENERIC[stage] - SHARED_GENERIC
            replay += [('instruction', d, 'extra') for d in extras[u * n:(u + 1) * n]]
        else:
            for domain, count in CR_LAYOUT[stage].items():
                if domain != 'instruction': replay += [(domain, d, None) for d in other[domain][u * count:(u + 1) * count]]
        for offset, (domain, draw, stream_name) in enumerate(replay):
            p = TASK_ROWS + offset + 1
            rows.append(_row(arm, seed, stage, update, p, layout[p - 1], 'replay', domain, replay_loss, None, draw, stream_name))
    return rows, summarise(arm, seed, stage, rows, lengths)


def draw_audit(rows):
    """Count gaps in the actual per-domain/per-stream draw indices, including leading skipped draws."""
    groups = {}; skipped = []; by_domain = Counter()
    for row in rows:
        if row['role'] != 'replay': continue
        key = (row['domain'], row.get('stream') or row['domain'])
        expected = groups.get(key, 0); observed = row['draw_index']
        require(type(observed) is int and observed >= expected and row.get('pass_index') == 0, 'replay draw order/replacement differs')
        gap = observed - expected
        if gap: skipped.append(dict(domain=key[0], stream=key[1], first_skipped=expected, draw_index=observed, count=gap))
        by_domain[key[0]] += gap; groups[key] = observed + 1
    return dict(skipped_draws=skipped, skipped_draw_count=sum(by_domain.values()), skipped_draw_count_by_domain=dict(by_domain))


def summarise(arm, seed, stage, rows, lengths=None):
    exposures = Counter(r['domain'] for r in rows)
    replay = [r for r in rows if r['role'] == 'replay']
    repeats = {}
    for domain in sorted({r['domain'] for r in replay}):
        counts = Counter(r['id'] for r in replay if r['domain'] == domain)
        repeats[domain] = dict(unique=len(counts), max_repeat=max(counts.values()), exposures=sum(counts.values()))
    require(all(v['max_repeat'] == 1 for v in repeats.values()), f'within-stage replay repeat in {arm}/{seed}/{stage}')
    tokens = None
    if lengths is not None:
        tokens = {d: sum(lengths.get(r['id'], 0) for r in replay if r['domain'] == d) for d in repeats}
    ids = {d: sorted({r['id'] for r in replay if r['domain'] == d}) for d in repeats}
    return dict(arm=arm, seed=seed, stage=stage, updates=UPDATES, rows_per_update=rows_per_update(arm, stage),
                rounds=rows_per_update(arm, stage) // RANKS, local_scale=local_scale(), global_divisor=GLOBAL_DIVISOR,
                replay_loss=REPLAY_LOSS[arm], prompt_cap=PROMPT_CAP, response_cap=RESPONSE_CAP, exposures=dict(exposures),
                task_rows=sum(r['role'] == 'task' for r in rows), replay_rows=len(replay), repeats=repeats,
                **draw_audit(rows), replay_prompt_tokens=tokens, replay_ids=ids,
                schedule_sha256=digest(rows))


def shared_generic(rows):
    """Per update, the four shared instruction IDs in stream order."""
    out = {}
    for r in rows:
        if r['domain'] == 'instruction' and r['role'] == 'replay' and r.get('stream') == 'shared':
            out.setdefault(r['update'], []).append((r['draw_index'], r['id']))
    return {u: [i for _, i in sorted(v)] for u, v in out.items()}


def audit_arms(schedules, summaries=None):
    """schedules: {arm: rows} for one seed/stage. Every update's four shared IDs match across arms; G extras are
    disjoint from the shared IDs and never repeat; CR/G share 32 task rows; C0/K share the same 28;
    skipped_draw_count == 0 for every arm/domain."""
    require(set(schedules) == set(ARMS), 'all four arms required for the cross-arm audit')
    first = shared_generic(schedules['C0'])
    require(len(first) == UPDATES and all(len(v) == SHARED_GENERIC for v in first.values()), 'C0 lacks four generic rows/update')
    for arm, rows in schedules.items():
        require(draw_audit(rows)['skipped_draw_count'] == 0, f'{arm} has filtered draws')
        require(shared_generic(rows) == first, f'{arm} does not share the four generic IDs at every update')
    for arm, summary in (summaries or {}).items():
        require(summary['skipped_draw_count'] == 0 and not summary['skipped_draws'], f'{arm} has filtered draws')
    shared = {i for v in first.values() for i in v}
    extras = [r['id'] for r in schedules['G'] if r.get('stream') == 'extra']
    require(not shared & set(extras) and len(extras) == len(set(extras)), 'G extras overlap shared IDs or repeat')
    def task(arm): return [(r['update'], r['position'], r['target_sha256']) for r in schedules[arm] if r['role'] == 'task']
    require(task('C0') == task('K'), 'C0/K task rows differ')
    require(task('CR') == task('G'), 'CR/G task rows differ')
    require(set(task('C0')) <= set(task('CR')), 'C0 task rows are not the same positions of the CR/G schedule')
    require(len(task('CR')) == UPDATES * TASK_ROWS and len(task('C0')) == UPDATES * (TASK_ROWS - SHARED_GENERIC), 'task exposure')
    return dict(shared_generic_sha256=digest(first), task_sha256=digest(task('CR')), shared_ids=sorted(shared), g_extra_ids=sorted(extras))


def cross_stage_repeats(summaries):
    """Replay IDs recurring across stages of one arm/seed chain (reported separately; allowed)."""
    seen = {}
    for s in summaries:
        for domain, ids in s['replay_ids'].items():
            for i in ids: seen.setdefault((s['arm'], s['seed'], domain, i), set()).add(s['stage'])
    out = Counter()
    for (arm, seed, domain, _), stages in seen.items():
        if len(stages) > 1: out[f'{arm}|{seed}|{domain}'] += 1
    return dict(out)


def exposure_totals():
    out = {}
    for arm in ARMS:
        for stage in STAGES:
            n = rows_per_update(arm, stage)
            task = UPDATES * (TASK_ROWS - SHARED_GENERIC) if arm in ('C0', 'K') else UPDATES * TASK_ROWS
            out[arm, stage] = (task, UPDATES * n - task)
    return out


def contract_agreement():
    """Compare with builder A's contract; differences are reported, never silently resolved."""
    try:
        from kit import m3_contract as contract
    except ImportError:
        return dict(status='PENDING', reason='kit/m3_contract.py not present')
    checks = {}
    for name in ('SEEDS', 'STAGES', 'ARMS', 'UPDATES', 'TASK_ROWS', 'RANKS', 'GLOBAL_DIVISOR', 'CR_LAYOUT', 'GENERIC_POSITIONS'):
        if hasattr(contract, name):
            checks[name] = json.dumps(getattr(contract, name), sort_keys=True, default=list) == json.dumps(globals()[name], sort_keys=True, default=list)
    if hasattr(contract, 'geometry'):
        checks['geometry'] = all(contract.geometry(a, s)['global_rows'] == rows_per_update(a, s) for a in ARMS for s in STAGES)
    if hasattr(contract, 'prompt_cap'):
        checks['common_prompt_cap'] = all(contract.prompt_cap(a, s) == PROMPT_CAP for a in ARMS for s in STAGES)
    status = 'PASS' if checks and all(checks.values()) else 'MISMATCH' if checks else 'PENDING'
    return dict(status=status, checks=checks, mismatched=sorted(k for k, v in checks.items() if not v))
