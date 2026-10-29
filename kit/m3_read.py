#!/usr/bin/env python3
"""Message-3 K/CR reader (registration v2) under the 29 October three-seed rule, plus budget and wall-time accounting.

Decisions (seeds 101-103 only; df=2), each a separate prospective registration:
  K  : K-C0 on stage-3 general-v1 instruction accuracy minus original base.
  CR : CR-G alone, on the stage-3 equal-weight mean retained accuracy (instruction, general maths, knowledge,
       FinQA, legal test). CR-C0 and G-C0 are descriptive package/exposure contrasts with no bars.
Primary: mean >= 2 points, two-sided paired 95% lower bound (t=4.302653) > 0, positive in 3/3 seeds. Guards: the same
paired lower bound >= -1 for individual general panels (knowledge = format-free likelihood), HellaSwag likelihood,
earlier-task retention and absolute scores, and each stage's task final and gain. Gain-AUC is diagnostic.
Labels: PASS, INCONCLUSIVE (valid, unmet; incumbent unchanged), PENDING (missing evidence), UNVERIFIED (identity).
Inherited absolute package bars (one-sided t=2.919986) are report-only, never gates for these contrasts.
"""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import math
from kit.m3_schedule import ARMS, SEEDS, STAGES

TWO_SIDED_T = 4.302653
ONE_SIDED_T = 2.919986
SENSITIVITY = TWO_SIDED_T / math.sqrt(3)  # 2.48414: guard passes iff mean >= -1 + 2.48414 x SD
MIN_BENEFIT = 2.0
GUARD = -1.0
STEPS = (0, 10, 20, 30, 40)
GENERAL = ('ifeval', 'math', 'knowledge')  # instruction, general maths, knowledge (format-free likelihood)
GUARDED_GENERAL = GENERAL + ('hellaswag',)
TASK_METRIC = {'finqa': 'finqa', 'contract_nli': 'test', 'math_hard': 'math_hard'}
PANEL_OF = {'ifeval': 'general-v1', 'math': 'general-v1', 'knowledge_strict': 'general-v1', 'knowledge': 'knowledge-ll',
            'hellaswag': 'hellaswag', 'finqa': 'finqa', 'test': 'test', 'math_hard': 'math_hard'}
SCORED_PANELS = ('finqa', 'test', 'math_hard', 'general-v1', 'hellaswag', 'knowledge-ll')
LEGAL_THRESHOLD = 69.7
SERVING_CEILING = 1.5
CONTRASTS = {'K-C0': ('K', 'C0', 'k'), 'CR-G': ('CR', 'G', 'cr'), 'CR-C0': ('CR', 'C0', 'cr'), 'G-C0': ('G', 'C0', 'cr')}
DECISIONS = {'K': 'K-C0', 'CR': 'CR-G'}
DESCRIPTIVE = ('CR-C0', 'G-C0')
# Repriced planning proxies (v2): FinQA anchor 7.73 measured; legal 6.572 = mean of return-1 c-native generic legal
# invocations; hard MATH 7.73 provisional (no matching receipt). CR/G scale 1.25/1.5/1.75; CR adds 7.73/seed legal replay.
STAGE_GPU_H = {'finqa': 7.73, 'contract_nli': 6.572, 'math_hard': 7.73}
STAGE_SCALE = {'finqa': 1.25, 'contract_nli': 1.5, 'math_hard': 1.75}
LEGAL_REPLAY_GPU_H = 7.73
K_RESERVE = 1.3
SCORING_PER_CALL = (0.05, 0.10)
RESERVE = (40, 80)
WALL_FLAG_HOURS = 24


class Missing(KeyError):
    pass


class Pending(KeyError):
    pass


def stats(values, t):
    n = len(values); mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else float('nan')
    return dict(n=n, mean=mean, sd=sd, lower=mean - t * sd / math.sqrt(n), values=values)


def sensitivity(sd):
    """Pre-send power note: the mean a guard needs at this paired SD, and whether a zero-mean guard would pass."""
    return dict(required_mean=GUARD + SENSITIVITY * sd, zero_mean_passes=sd <= -GUARD / SENSITIVITY)


class Table:
    """cells[(arm, seed, stage, step, panel)] = dict(metrics={metric: points}, items={metric: {qid: 0/1}}, uuid, environment,
    format_failures={metric: count}, output_tokens, correct, tolerant={metric: points}, pending=reason or None)."""

    def __init__(self, cells):
        self.cells = cells; self.used = set()

    def cell(self, arm, seed, stage, step, metric):
        key = ('base', seed, STAGES[0], 0, PANEL_OF[metric]) if arm == 'base' else (arm, seed, stage, step, PANEL_OF[metric])
        if key not in self.cells or metric not in self.cells[key]['metrics']: raise Missing(key + (metric,))
        if self.cells[key].get('pending'): raise Pending(key + (self.cells[key]['pending'],))
        self.used.add((key, metric)); return self.cells[key]

    def acc(self, arm, seed, stage, step, metric):
        return self.cell(arm, seed, stage, step, metric)['metrics'][metric]

    def base(self, seed, metric):
        return self.acc('base', seed, STAGES[0], 0, metric)

    def retained(self, arm, seed):
        final = STAGES[-1]
        parts = [self.acc(arm, seed, final, 40, m) - self.base(seed, m) for m in GENERAL]
        parts.append(self.acc(arm, seed, final, 40, 'finqa') - self.acc(arm, seed, 'finqa', 40, 'finqa'))
        parts.append(self.acc(arm, seed, final, 40, 'test') - self.acc(arm, seed, 'contract_nli', 40, 'test'))
        return sum(parts) / len(parts)

    def primary(self, kind, arm, seed):
        if kind == 'k': return self.acc(arm, seed, STAGES[-1], 40, 'ifeval') - self.base(seed, 'ifeval')
        return self.retained(arm, seed)

    def gain_auc(self, arm, seed, stage):
        metric = TASK_METRIC[stage]; start = self.acc(arm, seed, stage, 0, metric)
        gains = [self.acc(arm, seed, stage, k, metric) - start for k in STEPS]
        return sum((gains[i] + gains[i + 1]) / 2 * (STEPS[i + 1] - STEPS[i]) for i in range(len(STEPS) - 1)) / STEPS[-1]

    def guards(self):
        """Registered guards: name -> function(arm, seed)."""
        out = {}
        at = lambda st, m: (lambda arm, seed: self.acc(arm, seed, st, 40, m))
        for number, stage in enumerate(STAGES):
            for metric in GUARDED_GENERAL: out[f'{stage}/final/{metric}'] = at(stage, metric)
            for earlier in STAGES[:number]:
                metric = TASK_METRIC[earlier]
                out[f'{stage}/retention_absolute/{metric}'] = at(stage, metric)
                out[f'{stage}/retention/{metric}'] = (lambda st, e, m: lambda arm, seed: self.acc(arm, seed, st, 40, m) - self.acc(arm, seed, e, 40, m))(stage, earlier, metric)
            metric = TASK_METRIC[stage]
            out[f'{stage}/task/final'] = at(stage, metric)
            out[f'{stage}/task/gain'] = (lambda st, m: lambda arm, seed: self.acc(arm, seed, st, 40, m) - self.acc(arm, seed, st, 0, m))(stage, metric)
        return out

    def diagnostics(self):
        """Reported, never decision-bearing: gain-AUC, step-20 retention, strict-letter knowledge."""
        out = {}
        for number, stage in enumerate(STAGES):
            out[f'{stage}/task/gain_auc'] = (lambda st: lambda arm, seed: self.gain_auc(arm, seed, st))(stage)
            for earlier in STAGES[:number]:
                metric = TASK_METRIC[earlier]
                out[f'{stage}/step20/{metric}'] = (lambda st, m: lambda arm, seed: self.acc(arm, seed, st, 20, m))(stage, metric)
            out[f'{stage}/final/knowledge_strict'] = (lambda st: lambda arm, seed: self.acc(arm, seed, st, 40, 'knowledge_strict'))(stage)
        return out


def identity_failures(table, assignment):
    """Every constituent of a seed must sit on that seed's dedicated GPU/environment, shared across arms and base."""
    failures = []
    for key, cell in table.cells.items():
        expected = assignment.get(f'm3|{key[1]}') if assignment else None
        if expected is None: failures.append(dict(cell=list(key), reason='unassigned seed')); continue
        if cell.get('uuid') != expected['uuid'] or cell.get('environment') != expected['environment']:
            failures.append(dict(cell=list(key), reason='off-assignment constituent'))
        if cell.get('identity_error'): failures.append(dict(cell=list(key), reason=cell['identity_error']))
    return failures


def paired(table, fn, challenger, control):
    return stats([fn(challenger, s) - fn(control, s) for s in SEEDS], TWO_SIDED_T)


def optional(table, functions, challenger, control):
    out = {}
    for name, fn in functions.items():
        try: out[name] = paired(table, fn, challenger, control)
        except (Missing, Pending) as exc: out[name] = dict(status='PENDING', missing=[str(x) for x in exc.args[0]])
    return out


def contrast(table, name, failures, pending):
    challenger, control, kind = CONTRASTS[name]
    descriptive = name in DESCRIPTIVE
    result = dict(name=name, challenger=challenger, control=control, role='descriptive; no admission bars' if descriptive else 'decision-bearing')
    blocked = sorted({p for p in pending if p in (challenger, control)})
    if blocked: return dict(result, label='PENDING', reason='arm PENDING: ' + ','.join(blocked))
    table.used = set()
    try:
        primary = paired(table, lambda arm, s: table.primary(kind, arm, s), challenger, control)
        guards = {g: paired(table, f, challenger, control) for g, f in table.guards().items()}
    except Missing as exc:
        return dict(result, label='PENDING', reason='missing constituent ' + json.dumps([str(x) for x in exc.args[0]]))
    except Pending as exc:
        return dict(result, label='PENDING', reason='constituent pending ' + json.dumps([str(x) for x in exc.args[0]]))
    used_seeds = {key[1] for key, _ in table.used}
    bad = [f for f in failures if f['cell'][1] in used_seeds and f['cell'][0] in (challenger, control, 'base')]
    diagnostics = optional(table, table.diagnostics(), challenger, control)
    for value in guards.values(): value['sensitivity'] = sensitivity(value['sd'])
    if bad: return dict(result, label='UNVERIFIED', reason=bad[:5])
    if descriptive: return dict(result, label='DESCRIPTIVE', primary=primary, guards=guards, diagnostics=diagnostics)
    diffs = primary['values']
    bars = dict(mean_at_least_2=primary['mean'] >= MIN_BENEFIT, lower_bound_positive=primary['lower'] > 0, positive_3_of_3=all(d > 0 for d in diffs))
    failed = sorted(g for g, v in guards.items() if not v['lower'] >= GUARD)
    met = all(bars.values()) and not failed
    return dict(result, label='PASS' if met else 'INCONCLUSIVE', primary=primary, bars=bars, guards=guards, failed_guards=failed,
                diagnostics=diagnostics, note=None if met else 'unmet bars: incumbent unchanged')


def decide(contrasts):
    out = {decision: contrasts[name]['label'] for decision, name in DECISIONS.items()}
    out['CR_interpretation'] = 'cumulative coverage confirmation opens' if out['CR'] == 'PASS' else None
    out['stacking'] = 'separately registered one-change stacking comparison needed' if out['K'] == 'PASS' and out['CR'] == 'PASS' else None
    return out


def survival(table, challenger, control):
    """Acquired-answer survival: of items correct at each arm's acquisition final, the fraction still correct at
    later finals; per arm cohort and on the two arms' intersection. Reported per seed, never pooled into extra seeds."""
    out = {}
    for number, earlier in enumerate(STAGES[:-1]):
        metric = TASK_METRIC[earlier]
        for later in STAGES[number + 1:]:
            rows = {}
            for s in SEEDS:
                try:
                    acquired = {arm: {q for q, v in table.cell(arm, s, earlier, 40, metric)['items'][metric].items() if v} for arm in (challenger, control)}
                    final = {arm: table.cell(arm, s, later, 40, metric)['items'][metric] for arm in (challenger, control)}
                except (Missing, Pending, KeyError): rows[s] = dict(status='PENDING'); continue
                common = acquired[challenger] & acquired[control]
                rate = lambda arm, cohort: (sum(final[arm].get(q, 0) for q in cohort) / len(cohort)) if cohort else None
                rows[s] = dict(cohort={arm: dict(n=len(acquired[arm]), survival=rate(arm, acquired[arm])) for arm in (challenger, control)},
                               intersection=dict(n=len(common), **{arm: rate(arm, common) for arm in (challenger, control)}))
            out[f'{earlier}->{later}/{metric}'] = rows
    return out


def package_bars(table, arm):
    """Inherited absolute bars, one-sided t=2.919986, per arm. REPORT-ONLY: never gates for K-C0 or CR-G."""
    bars = dict(role='report-only')
    def guarded(name, fn, ok):
        try:
            value = stats([fn(s) for s in SEEDS], ONE_SIDED_T); bars[name] = dict(value, status='MET' if ok(value) else 'NOT_MET')
        except (Missing, Pending) as exc: bars[name] = dict(status='PENDING', missing=[str(x) for x in exc.args[0]])
    guarded('finqa_acquisition', lambda s: table.acc(arm, s, 'finqa', 40, 'finqa') - table.base(s, 'finqa'), lambda v: v['lower'] > 0)
    def tolerant(s, step):
        cell = table.cell(arm, s, 'contract_nli', step, 'test')
        if not cell.get('tolerant') or 'test' not in cell['tolerant']: raise Missing((arm, s, 'contract_nli', step, 'tolerant'))
        return cell['tolerant']['test']
    guarded('legal_tolerant', lambda s: tolerant(s, 40), lambda v: v['mean'] >= LEGAL_THRESHOLD and v['lower'] >= LEGAL_THRESHOLD)
    guarded('legal_tolerant_gain', lambda s: tolerant(s, 40) - tolerant(s, 0), lambda v: v['mean'] > 0)
    for stage in STAGES:
        for metric in GUARDED_GENERAL:
            guarded(f'{stage}/{metric}>=base-3', lambda s, st=stage, m=metric: table.acc(arm, s, st, 40, m) - table.base(s, m), lambda v: v['lower'] >= -3)
    for later in STAGES[1:]:
        guarded(f'{later}/finqa>=acquisition-1', lambda s, st=later: table.acc(arm, s, st, 40, 'finqa') - table.acc(arm, s, 'finqa', 40, 'finqa'), lambda v: v['lower'] >= -1)
    guarded(f'{STAGES[-1]}/legal>=acquisition-1', lambda s: table.acc(arm, s, STAGES[-1], 40, 'test') - table.acc(arm, s, 'contract_nli', 40, 'test'), lambda v: v['lower'] >= -1)
    try:
        acquisition = stats([table.acc(arm, s, 'math_hard', 40, 'math_hard') - table.acc(arm, s, 'math_hard', 0, 'math_hard') for s in SEEDS], ONE_SIDED_T)
        bars['math_acquisition'] = dict(acquisition, status='DESCRIPTIVE')
    except (Missing, Pending) as exc: bars['math_acquisition'] = dict(status='PENDING', missing=[str(x) for x in exc.args[0]])
    bars['math_standalone_reference'] = dict(status='PENDING', reason='matching standalone references are never borrowed across recipes')
    return bars


def serving(table, arm):
    """Tokens per correct answer on each task final vs the inherited 1.5x base ceiling (exceeding = 'at cost')."""
    out = {}
    for stage in STAGES:
        panel = PANEL_OF[TASK_METRIC[stage]]
        try:
            cells = [table.cells[(arm, s, stage, 40, panel)] for s in SEEDS]; base = [table.cells[('base', s, STAGES[0], 0, panel)] for s in SEEDS]
        except KeyError: out[stage] = dict(status='PENDING'); continue
        ratio = lambda cs: sum(c['output_tokens'] for c in cs) / max(1, sum(c['correct'] for c in cs))
        value, reference = ratio(cells), ratio(base)
        out[stage] = dict(tokens_per_correct=value, base=reference, ceiling=SERVING_CEILING * reference,
                          status='AT_COST' if value > SERVING_CEILING * reference else 'WITHIN_CEILING')
    return out


def strict_letter(table):
    """Generated-letter knowledge format failures: diagnostic only (v2)."""
    return sum(c.get('format_failures', {}).get('knowledge_strict', 0) for c in table.cells.values())


def analyse(cells, assignment, pending=(), receipts=None):
    """pending: arms whose pools are unavailable. receipts: {(arm, seed, stage): status}; FAIL is an identity failure,
    anything but PASS (INCOMPLETE until the 8B eight-H100 preflight passes) leaves the arm PENDING."""
    table = Table(cells)
    pending = set(pending)
    failures = identity_failures(table, assignment)
    for (arm, seed, stage), status in (receipts or {}).items():
        if status == 'FAIL': failures.append(dict(cell=[arm, seed, stage, 40, 'receipt'], reason='trainer receipt FAIL'))
        elif status != 'PASS': pending.add(arm)
    contrasts = {name: contrast(table, name, failures, pending) for name in CONTRASTS}
    return dict(schema='message3-read.v2', seeds=list(SEEDS), t_two_sided=TWO_SIDED_T, t_one_sided=ONE_SIDED_T, sensitivity_factor=SENSITIVITY,
                contrasts=contrasts, decisions=decide(contrasts),
                survival={name: survival(table, *CONTRASTS[name][:2]) for name in DECISIONS.values()},
                package_bars={arm: package_bars(table, arm) for arm in ARMS}, serving={arm: serving(table, arm) for arm in ARMS},
                strict_letter_knowledge_format_failures=strict_letter(table),
                identity_failures=failures, pending_arms=sorted(pending),
                rules='Unmet valid bars = INCONCLUSIVE (incumbent unchanged); missing = PENDING; identity failure = UNVERIFIED. '
                      'K-C0 and CR-G are separate prospective decisions, not a family-wide claim; CR-C0/G-C0 are descriptive.')


# ----------------------------------------------------------------------------- budget and wall time

def registered_costs():
    """v2 repriced incremental K/CR tier (GPU-h, three seeds). Proxies, not measured evidence; telemetry replaces them."""
    seeds = len(SEEDS)
    c0 = sum(STAGE_GPU_H.values())
    g_stage = {s: STAGE_GPU_H[s] * STAGE_SCALE[s] for s in STAGES}
    g = sum(g_stage.values()); cr = g + LEGAL_REPLAY_GPU_H
    training = [c0 * seeds * 2 + (g + cr) * seeds, c0 * seeds * (1 + K_RESERVE) + (g + cr) * seeds]
    return dict(stage_gpu_hours=STAGE_GPU_H, provisional=['math_hard'], C0=dict(per_seed=c0, total=c0 * seeds),
                K=dict(per_seed=[c0, K_RESERVE * c0], total=[c0 * seeds, K_RESERVE * c0 * seeds]),
                G=dict(per_stage=g_stage, per_seed=g, total=g * seeds), CR=dict(per_seed=cr, total=cr * seeds, legal_replay_per_seed=LEGAL_REPLAY_GPU_H),
                training_subtotal=training, scoring=None, reserve=list(RESERVE),
                note='C0 trained/charged once; PD/SP separately priced; missing timings replaced before send')


def scoring_budget(calls):
    """Receipt-based call budget: .05-.10 GPU-h/call; serial wall time on one dedicated GPU per seed."""
    low, high = calls * SCORING_PER_CALL[0], calls * SCORING_PER_CALL[1]
    per_seed = [low / len(SEEDS), high / len(SEEDS)]
    return dict(calls=calls, gpu_hours=[low, high], serial_hours_per_seed=per_seed,
                serial_hours_per_arm_seed=[v / len(ARMS) for v in per_seed],
                flag=any(v > WALL_FLAG_HOURS for v in per_seed))


def total_budget(calls):
    costs = registered_costs(); scoring = scoring_budget(calls)
    costs['scoring'] = scoring['gpu_hours']
    costs['total'] = [costs['training_subtotal'][0] + scoring['gpu_hours'][0] + RESERVE[0], costs['training_subtotal'][1] + scoring['gpu_hours'][1] + RESERVE[1]]
    return costs, scoring


def padded_token_cost(updates, gpu_hours_per_million_tokens):
    """Manager repricing API: independently calibrated phase rates times measured padded-token work.

    Rates are GPU-hours per million padded tokens for actor/ema/base/rollout.
    Actor metrics already include any oracle work; never multiply it again.
    This estimates update phases only, without changing registered budgets or pricing startup/exports.
    """
    import math
    domains = ('actor', 'ema', 'base', 'rollout')
    def number(value): return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    if set(gpu_hours_per_million_tokens) != set(domains) or any(not number(v) for v in gpu_hours_per_million_tokens.values()):
        raise ValueError('four finite nonnegative padded-token phase rates required')
    totals = dict.fromkeys(domains, 0)
    for row in updates:
        for domain in domains:
            value = row.get('m3/' + domain + '_padded_tokens')
            if not number(value): raise ValueError('missing/invalid padded-token telemetry: ' + domain)
            totals[domain] += value
    costs = {domain: totals[domain] * gpu_hours_per_million_tokens[domain] / 1_000_000 for domain in domains}
    return dict(padded_tokens=totals, phase_gpu_hours=costs, update_gpu_hours=sum(costs.values()),
                scope='update phases only; manager must add startup, export, scoring and preparation')


def actual_costs(out):
    out = Path(out); totals = {}; attempts = 0
    ledgers = sorted(out.glob('gpu-costs/*.json')) + sorted(out.glob('preparation-costs/*.json'))
    from kit.m3_score import unit_root
    scored = set()
    for path in ledgers:
        value = json.loads(path.read_text())
        if value.get('kind') == 'm3_scoring_attempt' and all(k in value for k in ('arm', 'seed', 'stage', 'step', 'panel')):
            scored.add(unit_root(out, value['arm'], value['seed'], value['stage'], value['step'], value['panel']).name)
    prepared = {json.loads(p.read_text()).get('kind') for p in ledgers}
    paths = sorted(out.glob('runs/*/training-costs/*.json')) + ledgers
    paths += [p for p in sorted(out.glob('scores/*/cost.json')) if p.parent.name not in scored]
    paths += [p for p in sorted(out.glob('pools/*/cost.json')) if 'm3_broad_preparation' not in prepared or p.parent.name != 'broad']
    if 'm3_math_targets' not in prepared and (out / 'targets-math/cost.json').exists():
        source = out / 'targets-math-source.json'
        if not source.exists() or json.loads(source.read_text()).get('path') != 'core-copy': paths.append(out / 'targets-math/cost.json')
    for path in paths:
        value = json.loads(path.read_text())
        if value.get('included_in'): continue
        kind = value.get('kind', 'unknown')
        key = f"{kind}:{value['arm']}" if 'arm' in value else kind
        totals[key] = totals.get(key, 0) + (value.get('gpu_hours') or 0); attempts += 1
    return dict(gpu_hours=totals, total=sum(totals.values()), attempts=attempts)


def wall_time(states, tasks):
    """Account all attempts, including failed runs and time queued before execution."""
    per_arm = {}; queue = {}; execution = {}; waiting = {}
    for task in tasks:
        state = states.get(task['id'], {})
        attempts = state.get('attempts') or ([state] if 'seconds' in state else [])
        hours = 0; active = 0; delay = 0
        previous_end = None
        for attempt in attempts:
            seconds = max(0, attempt.get('seconds', 0)); active += seconds / 3600
            start, end = attempt.get('started_at'), attempt.get('ended_at')
            submitted = attempt.get('submitted_at', start)
            if start is not None and end is not None:
                # Retries can share the original submission; count each queued interval once.
                queued = max(0, start - max(submitted, previous_end or submitted))
                previous_end = end
            else: queued = 0
            delay += queued / 3600; hours += (seconds + queued) / 3600
        if not attempts and state.get('submitted_at'):
            hours = max(0, __import__('time').time() - state['submitted_at']) / 3600
            delay = hours
        execution[task['id']] = active; waiting[task['id']] = delay
        if task.get('arm') and task.get('seed'):
            key = f"{task['arm']}|{task['seed']}"; per_arm[key] = per_arm.get(key, 0) + hours
        if task['kind'] == 'score-family' and task.get('seed'):
            key = str(task['seed']); queue[key] = queue.get(key, 0) + hours
    flags = [f'ARM {k} measured {v:.1f} h > 24 h' for k, v in sorted(per_arm.items()) if v > WALL_FLAG_HOURS]
    flags += [f'SEED {k} scoring queue {v:.1f} h > 24 h' for k, v in sorted(queue.items()) if v > WALL_FLAG_HOURS]
    combined = sum(queue.values())
    if combined > WALL_FLAG_HOURS: flags.append(f'COMBINED scoring queue {combined:.1f} h > 24 h')
    return dict(arm_seed_hours=per_arm, seed_queue_hours=queue, combined_queue_hours=combined, flags=flags,
                execution_hours=execution, waiting_hours=waiting)


# ----------------------------------------------------------------------------- SP checkpoint lists

TIER_SCHEMA = 'sp-tier-checkpoints-v1'
TIER_ARMS = {'K': ('C0', 'K'), 'CR': ('G', 'CR')}  # C0 is listed once, with K (it is trained and charged once)


def sp_digest(value):
    """m3b2's pd_contract.digest: sha256 of canonical JSON (sorted keys, no NaN, UTF-8)."""
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sp_tier_checkpoints(tier, out=None):
    """K/CR checkpoint lists in m3b2's sp-tier-checkpoints-v1 format. SP scores base, incoming and final checkpoints
    only. Without an M3 root the list is planned (runtime resolution); with one, completed exports bind their
    model_file_hashes and weights_sha256."""
    entries = []
    def files_of(path, nested=False):
        if out is None or not path.exists(): return None
        doc = json.loads(path.read_text())
        return (doc.get('identity') or {}).get('model_file_hashes') if nested else doc.get('model_file_hashes')
    def bound(entry, files):
        if files: entry.update(model_file_hashes=files, weights_sha256=sp_digest(files))
        return entry
    root = Path(out) if out is not None else Path('.')
    for seed in SEEDS:
        base_files = files_of(root / 'inputs/base.json', nested=True)
        entries.append(bound(dict(checkpoint_id=f'm3-base-s{seed}', family='m3', seed=seed, arm='base', step=0, role='base', task='M3-bind-base'), base_files))
        for arm in TIER_ARMS[tier]:
            for number, stage in enumerate(STAGES):
                task = f'M3-{arm}-{seed}-{stage}-train'
                incoming = base_files if number == 0 else files_of(root / 'runs' / f'{arm}-s{seed}-{STAGES[number - 1]}' / 'export-40.json')
                entries.append(bound(dict(checkpoint_id=f'm3-{arm}-s{seed}-{stage}-k0', family='m3', seed=seed, arm=arm, step=0, role='incoming',
                                          task=task, stage=stage), incoming))
                final = files_of(root / 'runs' / f'{arm}-s{seed}-{stage}' / 'export-40.json')
                entries.append(bound(dict(checkpoint_id=f'm3-{arm}-s{seed}-{stage}-k40', family='m3', seed=seed, arm=arm, step=40, role='final',
                                          task=task, stage=stage), final))
    return dict(schema=TIER_SCHEMA, tier=tier, checkpoints=entries)


def sp_tiers_job(a):
    from kit.m3_manifest import require_m3_root
    require_m3_root(a.out)
    for tier in TIER_ARMS:
        path = a.out / 'sp-tiers' / f'{tier}.json'; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sp_tier_checkpoints(tier, a.out), indent=1, sort_keys=True) + '\n')
    print('SP tier checkpoint lists written under', a.out / 'sp-tiers')


# ----------------------------------------------------------------------------- loading

def load_cells(out, standin=False):
    """Read every frozen M3 score unit (following aliases) into reader cells, with identity checks."""
    import csv
    from kit.p3b_common import frozen_files
    from kit.m3_score import resolve, ASSIGNMENT
    from kit import m2_assignment
    out = Path(out); raw = (out / ASSIGNMENT).read_bytes(); doc = json.loads(raw)
    if not standin and len({v['uuid'] for v in doc['assignments'].values()}) < len(SEEDS):
        raise ValueError('scientific assignment lacks three dedicated physical GPUs')
    cells = {}
    units = [('base', s, STAGES[0], 0) for s in SEEDS] + [(a, s, st, k) for a in ARMS for s in SEEDS for st in STAGES for k in STEPS]
    for arm, seed, stage, step in units:
        for panel in SCORED_PANELS:
            root = resolve(out, arm, seed, stage, step, panel)
            if root is None: continue
            frozen_files(root)
            cell = dict(metrics={}, items={}, format_failures={}, tolerant={}, output_tokens=0, correct=0, pending=None)
            try:
                meta = json.loads((root / 'm3-identity.json').read_text())
                if bool(meta.get('technical_synthetic')) != standin: raise ValueError('synthetic score provenance differs')
                m2_assignment.verify(meta, raw)
                proof = meta['assignment']
                if proof['seed'] != seed or proof['family'] != 'm3': cell['identity_error'] = 'foreign family/seed'
                cell.update(uuid=proof['uuid'], environment=proof['environment'])
                if panel in ('hellaswag', 'knowledge-ll'):
                    from kit.m3_score import validation_status
                    cell['validation_sha256'] = validation_status(out, panel, seed, standin=standin)
                    if cell['validation_sha256'] is None: cell['pending'] = 'likelihood validation not PASS (base + C0 s101 stage-1 final)'
            except Exception as exc:
                cell['identity_error'] = 'UNVERIFIED: ' + str(exc)
            with (root / 'scores.csv').open(newline='') as stream: rows = list(csv.DictReader(stream))
            groups = {}
            for r in rows:
                metric = panel if panel != 'general-v1' else ('knowledge_strict' if r['panel'] == 'knowledge' else r['panel'])
                if panel == 'knowledge-ll': metric = 'knowledge'
                groups.setdefault(metric, []).append(r)
            for metric, values in groups.items():
                cell['metrics'][metric] = 100 * sum(int(r['correct']) for r in values) / len(values)
                cell['items'][metric] = {r['question_id']: int(r['correct']) for r in values}
                cell['format_failures'][metric] = sum(int(r.get('incorrect_format') or 0) for r in values)
                if values and values[0].get('tolerant_correct') not in (None, ''):
                    cell['tolerant'][metric] = 100 * sum(int(r['tolerant_correct']) for r in values) / len(values)
                cell['output_tokens'] += sum(int(r.get('output_tokens') or 0) for r in values)
                cell['correct'] += sum(int(r['correct']) for r in values)
            cells[(arm, seed, stage, step, panel)] = cell
    return cells, doc['assignments']


def read_job(a):
    from kit.m3_manifest import require_m3_root, load as load_manifest
    require_m3_root(a.out)
    cells, assignment = load_cells(a.out, standin=a.standin)
    availability = json.loads((a.out / 'pools/availability.json').read_text()) if (a.out / 'pools/availability.json').exists() else {}
    from kit.m3_pools import pool_root
    pending = []
    for arm in ARMS:
        available = pool_root(a.out, arm) / 'availability.json'
        if not available.exists() or json.loads(available.read_text()).get(arm, {}).get('status') != 'READY': pending.append(arm)
    receipts = {}
    for arm in ARMS:
        for seed in SEEDS:
            for stage in STAGES:
                from kit.m3_train import validate_stage_receipt
                try: receipts[(arm, seed, stage)] = validate_stage_receipt(a.out, arm, seed, stage)['status']
                except Exception: receipts[(arm, seed, stage)] = 'INCOMPLETE'
    result = analyse(cells, assignment, pending, None if a.standin else receipts)
    manifest = a.out / 'm3-launch.json'
    tasks = load_manifest(manifest)['tasks'] if manifest.exists() else []
    calls = sum(1 for t in tasks if t['kind'] == 'score-family' for c in t['commands'] if '--panel-set' in c)
    states = {p.stem: json.loads(p.read_text()) for p in sorted((a.out / 'state').glob('*.json'))}
    costs, scoring = total_budget(calls)
    result['costs'] = dict(registered=costs, scoring=scoring, actual=actual_costs(a.out))
    result['wall_time'] = wall_time(states, tasks)
    result['technical_synthetic'] = a.standin
    (a.out / 'm3-read.json').write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + '\n')
    lines = ['| Decision | Label |', '|---|---|'] + [f'| {k} ({DECISIONS[k]}) | {result["decisions"][k]} |' for k in DECISIONS]
    lines += ['', '| Contrast | Role | Label | Mean | Lower bound | Seeds positive |', '|---|---|---|---:|---:|---:|']
    for name, value in result['contrasts'].items():
        p = value.get('primary')
        cells_text = f"{p['mean']:.2f} | {p['lower']:.2f} | {sum(v > 0 for v in p['values'])}/3" if p else ' | | '
        lines.append(f"| {name} | {value['role']} | {value['label']} | {cells_text} |")
    if result['wall_time']['flags']: lines += ['', '**WALL-TIME FLAGS (> 24 h):** ' + '; '.join(result['wall_time']['flags'])]
    if scoring['flag']: lines += ['', '**PROJECTED SCORING QUEUE > 24 h PER SEED**']
    (a.out / 'm3-read.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result['decisions'], sort_keys=True))


def job(a):
    return sp_tiers_job(a) if a.action == 'sp-tiers' else read_job(a)


def configure(p):
    p.add_argument('--action', choices=('read', 'sp-tiers'), default='read')


if __name__ == '__main__':
    from kit.p3b_common import entry
    raise SystemExit(entry(job, configure, 'm3-read-{action}'))
