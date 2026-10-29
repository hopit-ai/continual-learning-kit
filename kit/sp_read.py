#!/usr/bin/env python3
"""SP reader: descriptive tables for the unseen-wording panel and frozen-cohort length.

    python kit/sp_read.py --sidecar DIR --out DIR

Everything here is DESCRIPTIVE in this send. No function admits, vetoes or gates K, CR, X or any
incumbent; the existing registered serving-cost bars stay decision-bearing and are not touched.
Three-seed intervals use exactly seeds 101-103 (mean +/- 4.302653 * SD / sqrt(3)); a missing seed
is INCOMPLETE, never an n=2 interval. Paired comparisons require the same physical GPU and
environment; otherwise the pair is labelled UNPAIRED-GPU and no interval is formed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.pd_contract import require, verify_seal, digest, sha256_file
from kit.v4_teacher import read_jsonl

SEEDS = (101, 102, 103)
T_975_DF2 = 4.302653
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_RNG = 20261029
LABEL = 'DESCRIPTIVE'


# ---------------------------------------------------------------------------------------------
# One checkpoint's panel summary.

def summarise(scored):
    """Strict prompt accuracy, instruction-level accuracy, per-family and per-wording, in points."""
    require(bool(scored), 'no scored prompts')
    def pct(values):
        values = list(values)
        return 100.0 * sum(values) / len(values) if values else None
    instructions = [(i['family'], i['passed']) for r in scored for i in r['instructions']]
    families = sorted({f for f, _ in instructions})
    by_wording = {w: [r for r in scored if r['wording'] == w] for w in ('canonical', 'paraphrase')}
    out = dict(label=LABEL, prompts=len(scored), strict=pct(r['strict'] for r in scored),
               instruction=pct(p for _, p in instructions),
               instruction_by_family={f: pct(p for g, p in instructions if g == f) for f in families},
               strict_by_primary_family={f: pct(r['strict'] for r in scored if r['primary_family'] == f)
                                         for f in sorted({r['primary_family'] for r in scored})},
               wording={w: dict(strict=pct(r['strict'] for r in rows),
                                instruction=pct(i['passed'] for r in rows for i in r['instructions']))
                        for w, rows in by_wording.items()})
    out['wording_gap'] = wording_gap(scored)
    out['template_reliance_gap'] = out['wording_gap']   # registration v2 name: canonical - paraphrase
    fenced = [r['json_fence_failure'] for r in scored if r.get('json_fence_failure') is not None]
    out['json_fence_failures'] = dict(prompts=len(fenced), failures=sum(fenced))    # descriptive only
    out['warnings'] = rate_warnings(out)
    return out


def rate_warnings(summary, low=5.0, high=95.0):
    """Monitored warnings only (never disqualifiers or reasons to rewrite the panel)."""
    out = []
    for name, value in (('strict', summary['strict']),
                        ('canonical', summary['wording']['canonical']['strict']),
                        ('paraphrase', summary['wording']['paraphrase']['strict'])):
        if value is not None and not low <= value <= high:
            out.append('WARNING: %s strict pass rate %.1f outside %g-%g%%' % (name, value, low, high))
    return out


WALL_TIME_RISK_HOURS = 24.0


def wall_time(cells, inventory=None, other=(), execution=()):
    """All registered SP and parent scoring loads/scores/receipts, per assignment and arm/seed."""
    per, arms = {}, {}
    def account(table, key, timing, gpu_hours):
        row = table.setdefault(key, dict(entries=0, serial_wall_hours=0.0, gpu_hours=0.0, untimed=0))
        row['entries'] += 1
        timing = timing or {}
        seconds = timing.get('serial_wall_seconds')
        if seconds is None and all(timing.get(k) is not None for k in ('load', 'score', 'receipt')):
            seconds = sum(timing[k] for k in ('load', 'score', 'receipt'))
        if seconds is None:
            row['untimed'] += 1
        else:
            row['serial_wall_hours'] += seconds / 3600
        row['gpu_hours'] += gpu_hours or 0.0
    scored = {c['entry']['entry_id']: c for c in cells}
    runtime = {}
    for task in execution:
        for item in task.get('commands', []):
            if item.get('entry_id'):
                runtime.setdefault(item['entry_id'], []).append(item)
    entries = (inventory or {}).get('sp_entries', [c['entry'] for c in cells])
    for e in entries:
        c = scored.get(e['entry_id'], {})
        timed = runtime.get(e['entry_id'], [])
        # Runner totals include binding, provenance verification, CUDA resolution and receipts.
        # Use them instead of the nested scorer timing, so these costs are never counted twice.
        complete = (len(timed) >= 2 and all(i.get('status') == 'DONE' and i.get('wall_seconds') is not None for i in timed))
        timing = dict(serial_wall_seconds=sum(i['wall_seconds'] for i in timed)) if complete else c.get('timing')
        account(per, e.get('assignment', 'unknown'), timing, c.get('gpu_hours'))
        uses = e.get('uses') or [e]
        keys = {'%s.%s.%s-s%s' % (u.get('source'), u.get('family'), u.get('arm'), u.get('seed')) for u in uses}
        for key in keys:
            account(arms, key, timing, c.get('gpu_hours'))
    # Parent score cells are registered independently of whether their evidence survived.
    from kit.sp_manifest import _match_key
    observed = {_match_key(c): c for c in other if c.get('status') == 'OK'}
    for p in (inventory or {}).get('planned', []):
        c = observed.get(_match_key(p), {})
        key = '%s.%s.%s-s%s' % (p.get('source'), p.get('family'), p.get('arm'), p.get('seed'))
        account(arms, key, c.get('timing'), c.get('gpu_hours'))
    for p in (inventory or {}).get('other_scoring', []):
        key = '%s.%s.other-s%s' % (p.get('source'), p.get('family'), p.get('seed'))
        account(arms, key, p.get('timing'), p.get('gpu_hours'))
    for row in [*per.values(), *arms.values()]:
        row['status'] = 'UNMEASURED' if row['untimed'] else 'MEASURED'
        row['flag'] = 'WALL-TIME RISK' if row['serial_wall_hours'] > WALL_TIME_RISK_HOURS else None
    return dict(label=LABEL, gate=False, assignments=per, arms=arms,
                status='UNMEASURED' if any(r['untimed'] for r in [*per.values(), *arms.values()]) else 'MEASURED',
                gpu_hours=sum(r['gpu_hours'] for r in per.values()))


def pairs(scored):
    """{request_id: (canonical_strict, paraphrase_strict)}; both wordings required for every request."""
    table = {}
    for r in scored:
        table.setdefault(r['request_id'], {})[r['wording']] = r['strict']
    require(all(set(v) == {'canonical', 'paraphrase'} for v in table.values()), 'both wordings required per request')
    return {k: (v['canonical'], v['paraphrase']) for k, v in sorted(table.items())}


def wording_gap(scored):
    """Paired gap in points: canonical strict accuracy minus paraphrase strict accuracy."""
    p = pairs(scored)
    return 100.0 * sum(c - q for c, q in p.values()) / len(p)


template_reliance_gap = wording_gap   # registration v2 name


def gap_change(descendant, incoming):
    """Change in the paired wording gap from the family's incoming checkpoint (points)."""
    require(set(pairs(descendant)) == set(pairs(incoming)), 'gap change needs the same request pairs')
    return wording_gap(descendant) - wording_gap(incoming)


# ---------------------------------------------------------------------------------------------
# Intervals.

def interval3(by_seed):
    """Three-seed paired interval on per-seed values {seed: value}; INCOMPLETE unless all of 101-103."""
    missing = [s for s in SEEDS if by_seed.get(s) is None]
    if missing or set(by_seed) - set(SEEDS):
        return dict(status='INCOMPLETE', missing=missing, label=LABEL)
    values = [float(by_seed[s]) for s in SEEDS]
    mean = sum(values) / 3
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / 2)
    half = T_975_DF2 * sd / math.sqrt(3)
    return dict(status='COMPLETE', mean=mean, sd=sd, lower=mean - half, upper=mean + half,
                values={s: by_seed[s] for s in SEEDS}, label=LABEL)


def paired_seeds(challenger, control, metric):
    """Per-seed challenger - control on `metric`, same physical GPU/environment only.

    challenger/control: {seed: {"summary": summarise(...), "gpu_identity": {...}}}.
    """
    diffs, unpaired, unverified = {}, [], []
    for seed in SEEDS:
        a, b = challenger.get(seed), control.get(seed)
        if a is None or b is None:
            continue
        if any(a.get(k) != b.get(k) for k in ('frozen_sha256', 'decoding_sha256')):
            unverified.append(seed)
            continue
        ga, gb = a.get('gpu_identity') or {}, b.get('gpu_identity') or {}
        if not ga.get('uuid') or not ga.get('environment') or (ga.get('uuid'), ga.get('environment')) != (gb.get('uuid'), gb.get('environment')):
            unpaired.append(seed)
            continue
        diffs[seed] = metric(a) - metric(b)
    result = (dict(status='UNVERIFIED', seeds=unverified, label=LABEL) if unverified else
              dict(status='UNPAIRED-GPU', seeds=unpaired, label=LABEL) if unpaired else interval3(diffs))
    result['per_seed'] = diffs
    return result


PAIR_STATISTICS = {
    'wording_gap': lambda c, q: c - q,
    'canonical_strict': lambda c, q: c,
    'paraphrase_strict': lambda c, q: q,
    'strict': lambda c, q: (c + q) / 2,
}


def bootstrap_pairs(scored, statistic='wording_gap', draws=BOOTSTRAP_DRAWS, rng_seed=BOOTSTRAP_RNG):
    """Descriptive percentile bootstrap over request pairs; both wordings of a request move together.

    Each statistic is a mean over requests of a per-pair value, so resampling per-pair values is
    exactly resampling whole pairs. Ranks are 1-based: lower = rank ceil(0.025 * draws).
    """
    values = [100.0 * PAIR_STATISTICS[statistic](c, q) for c, q in pairs(scored).values()]
    rng = random.Random(rng_seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(draws))
    lo = means[math.ceil(0.025 * draws) - 1]
    hi = means[math.ceil(0.975 * draws) - 1]
    return dict(statistic=statistic, draws=draws, rng=rng_seed, estimate=sum(values) / n, lower=lo, upper=hi,
                unit='request pair', label=LABEL)


# ---------------------------------------------------------------------------------------------
# Length trajectories.

def trajectory(points):
    """A plotted trajectory may only join cells that share one frozen cohort.

    points: [{"checkpoint_id", "step", "metrics": length_metrics(...)}]. Stage-specific incoming
    cohorts are different populations and are refused rather than pooled.
    """
    cohorts = {p['metrics']['cohort']['cohort_identity_sha256'] for p in points if p['metrics'].get('cohort')}
    require(len(cohorts) == 1 and all(p['metrics'].get('cohort') for p in points),
            'trajectory mixes cohorts; stage-specific cohorts cannot be pooled')
    conventions = {p['metrics']['convention_sha256'] for p in points}
    require(len(conventions) == 1, 'UNVERIFIED: length conventions differ across checkpoints')
    return dict(cohort_identity_sha256=cohorts.pop(), label=LABEL,
                points=[dict(checkpoint_id=p['checkpoint_id'], step=p.get('step'), **{
                    k: p['metrics']['cohort'][k] for k in ('cohort_size', 'accuracy', 'mean_tokens',
                                                           'tokens_per_correct', 'truncation_rate')}) for p in points])


# ---------------------------------------------------------------------------------------------
# CPU stand-in responses and the sidecar reader.

def standin_responses(panel):
    """Deterministic technical-synthetic responses: half the prompts obey nothing in particular."""
    out = []
    for i, r in enumerate(panel):
        text = 'a plain answer.' if i % 2 else 'A Plain Answer, With Capitals.'
        out.append(dict(id=r['id'], response=text, token_ids=list(range(5 + i % 3)), output_tokens=5 + i % 3,
                        prompt_tokens=10, finish_reason='stop', technical_synthetic=True))
    return out


def read_sidecar(sidecar, inventory=None):
    """Collect every completed SP panel cell under the sidecar root; missing cells stay missing."""
    root = Path(sidecar) / 'panel'
    cells = []
    for cell in sorted(root.iterdir()) if root.is_dir() else []:
        if not (cell / 'manifest.json').exists():
            continue
        from kit.sp_score import verify_outputs
        verify_outputs(cell)
        identity = json.loads((cell / 'identity.json').read_text())
        entry = identity['entry']
        if inventory is not None:
            expected = next((e for e in inventory['sp_entries'] if e['entry_id'] == entry['entry_id']), None)
            require(expected is not None, 'UNVERIFIED: foreign returned SP entry')
            for field in ('assignment', 'uses', 'planned_ids'):
                require(entry.get(field) == expected.get(field), 'UNVERIFIED: returned SP plan differs')
            if expected.get('weights_sha256'):
                require(expected['weights_sha256'] == entry['weights_sha256'], 'UNVERIFIED: returned SP weights differ')
        if not identity.get('technical_synthetic'):
            from kit.sp_score import DECODING, load_panel
            verify_seal(entry)
            frozen, _, _ = load_panel()
            require(identity.get('decoding') == DECODING and identity.get('decoding_sha256') == digest(DECODING),
                    'UNVERIFIED: returned decoding differs')
            require(identity.get('frozen_sha256') == (frozen.get('identity_sha256') or digest(frozen)),
                    'UNVERIFIED: returned frozen panel differs')
            hardware = json.loads((Path(sidecar) / 'hardware.json').read_text())
            verify_seal(hardware)
            require(identity.get('hardware_sha256') == hardware['identity_sha256'] and
                    identity.get('gpu_identity') == hardware['assignments'].get(entry['assignment']),
                    'UNVERIFIED: returned hardware authority differs')
        scored = read_jsonl(cell / 'scores.jsonl')
        cells.append(dict(entry=identity['entry'], gpu_identity=identity['gpu_identity'], summary=summarise(scored),
                          length=json.loads((cell / 'length.json').read_text()), scored=scored,
                          raw=read_jsonl(cell / 'raw.jsonl'),
                          frozen_sha256=identity.get('frozen_sha256'), decoding_sha256=identity.get('decoding_sha256'),
                          timing=identity.get('timing'), gpu_hours=identity.get('gpu_hours')))
    return cells


def family_tables(cells, inventory=None, sidecar=None):
    """Preserve stage/arm edges; frozen incoming populations precede descendant length analysis."""
    from kit import sp_score
    from kit.sp_manifest import gpu_key
    tables, indexed = {}, {}
    def family_name(value):
        import re
        return re.sub(r'(return1\.[AB])10[123]$', r'\1', str(value))
    def comp(use):
        return use.get('comparison') or use.get('arm') or 'default'
    def key(use):
        return (use.get('source'), use.get('family'), use.get('seed'), use.get('arm'), use.get('step'))
    for cell in cells:
        for use in cell['entry'].get('uses') or [dict(family=cell['entry'].get('family'), seed=cell['entry'].get('seed'), role=cell['entry'].get('role'))]:
            indexed.setdefault(key(use), {})[cell['entry']['weights_sha256']] = cell
            bucket = tables.setdefault(family_name(use['family']), {}).setdefault(use['seed'], [])
            if not any(c['entry']['entry_id'] == cell['entry']['entry_id'] and u == use for c, u in bucket):
                bucket.append((cell, use))
    out = {}
    for family, seeds in sorted(tables.items()):
        rows, comparisons = {}, {}
        for seed, members in sorted(seeds.items(), key=lambda kv: str(kv[0])):
            for cell, use in members:
                incoming_key = use.get('incoming_key')
                candidates = indexed.get((incoming_key[0], incoming_key[1], incoming_key[3], incoming_key[2], incoming_key[4]), {}) if incoming_key else {
                    c['entry']['weights_sha256']: c for c, u in members
                    if u.get('role') == 'incoming' and comp(u) == comp(use)}
                incoming = next(iter(candidates.values())) if len(candidates) == 1 else None
                row = dict(entry_id=cell['entry']['entry_id'], checkpoint_id=cell['entry'].get('checkpoint_id'),
                           role=use.get('role'), arm=use.get('arm'), comparison=comp(use), step=use.get('step'),
                           **cell['summary'], gap_change=None, gpu_identity=cell['gpu_identity'], length=cell['length'])
                if use.get('role') not in ('base', 'incoming'):
                    if incoming is None:
                        row['gap_change'] = 'MISSING-INCOMING'
                    elif gpu_key(cell['gpu_identity']) != gpu_key(incoming['gpu_identity']) or gpu_key(cell['gpu_identity']) is None:
                        row['gap_change'] = 'UNPAIRED-GPU'
                    elif (cell.get('frozen_sha256'), cell.get('decoding_sha256')) != (incoming.get('frozen_sha256'), incoming.get('decoding_sha256')):
                        row['gap_change'] = 'UNVERIFIED'
                    else:
                        # Freeze solely from the incoming strict verdicts, before descendant length.
                        if sidecar is not None and incoming.get('raw') is not None:
                            name = 'sp.%s.%s.s%s.%s.sp.json' % (use.get('source'), family, seed, comp(use))
                            co = sp_score.freeze_correct_cohort(Path(sidecar) / 'cohorts' / name,
                                family=family + '.' + comp(use), seed=seed, panel='sp',
                                incoming=dict(checkpoint_id=incoming['entry']['entry_id'], weights_sha256=incoming['entry']['weights_sha256']),
                                score_rows=[dict(question_id=r['id'], strict_all_constraints_pass=r['strict_all_constraints_pass']) for r in incoming['scored']],
                                scores_sha256=digest(incoming['scored']), raw_sha256=digest(incoming['raw']),
                                panel_sha256=incoming.get('frozen_sha256'), column='strict_all_constraints_pass')
                            row['length'] = sp_score.length_metrics(cell['raw'], {r['id']: r['strict_all_constraints_pass'] for r in cell['scored']}, co)
                            row['incoming_length'] = sp_score.length_metrics(incoming['raw'], {r['id']: r['strict_all_constraints_pass'] for r in incoming['scored']}, co)
                        row['gap_change'] = gap_change(cell['scored'], incoming['scored'])
                row['template_reliance_change'] = row['gap_change']
                rows.setdefault(str(seed), []).append(row)
                comparisons.setdefault(comp(use), {}).setdefault(use.get('role'), {})[seed] = cell
        intervals = {}
        for comparison, roles in comparisons.items():
            final, incoming = roles.get('final', {}), roles.get('incoming', {})
            # Explicit per-descendant baseline, including shared X pre aliases.
            for seed, members in seeds.items():
                for c, u in members:
                    if comp(u) == comparison and u.get('role') == 'final' and u.get('incoming_key'):
                        candidates = indexed.get((u['incoming_key'][0], u['incoming_key'][1], u['incoming_key'][3], u['incoming_key'][2], u['incoming_key'][4]), {})
                        if len(candidates) == 1:
                            incoming[seed] = next(iter(candidates.values()))
            intervals[comparison] = dict(
                template_reliance_gap=interval3({s: c['summary']['template_reliance_gap'] for s, c in final.items()}),
                change_from_incoming={metric: paired_seeds(final, incoming, lambda c, m=metric: c['summary'][m])
                                      for metric in ('strict', 'instruction', 'template_reliance_gap')})
        contrasts = {}
        for comparison, roles in comparisons.items():
            arm, dot, stage = comparison.partition('.')
            control = ('C0' if arm == 'K' else 'G' if arm == 'CR' else None)
            if comparison.startswith('x-') and comparison != 'x-control' and not comparison.endswith('-control'):
                control = comparison.rsplit('-', 1)[0] + '-control' if comparison.endswith('-challenger') else 'x-control'
                dot, stage = '', ''
            if control:
                control += dot + stage
                contrasts[comparison + '-minus-' + control] = {
                    metric: paired_seeds(roles.get('final', {}), comparisons.get(control, {}).get('final', {}),
                                         lambda c, m=metric: c['summary'][m])
                    for metric in ('strict', 'instruction', 'template_reliance_gap')}
        out[family] = dict(label=LABEL, by_seed=rows, intervals=intervals, contrasts=contrasts)
    scored_ids = {c['entry']['entry_id'] for c in cells}
    missing = [e['entry_id'] for e in (inventory or {}).get('sp_entries', []) if e['entry_id'] not in scored_ids]
    # Missing-only comparisons must still have a report and an INCOMPLETE interval.
    for e in (inventory or {}).get('sp_entries', []):
        for use in e.get('uses', []):
            fam = out.setdefault(family_name(use['family']), dict(label=LABEL, by_seed={}, intervals={}, contrasts={}))
            fam['intervals'].setdefault(comp(use), dict(template_reliance_gap=interval3({}),
                change_from_incoming={m: interval3({}) for m in ('strict', 'instruction', 'template_reliance_gap')}))
    return dict(label=LABEL, status='INCOMPLETE' if missing else 'COMPLETE', families=out, missing_entries=missing)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sidecar', type=Path, required=True)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--input', type=Path, action='append', default=[])
    a = p.parse_args(argv)
    inventory = None
    if a.inventory:
        inventory = json.loads(a.inventory.read_text())
        verify_seal(inventory)
    cells = read_sidecar(a.sidecar, inventory)
    tables = family_tables(cells, inventory, a.sidecar)
    tables['bootstrap'] = {c['entry']['entry_id']: bootstrap_pairs(c['scored']) for c in cells}
    from kit.sp_manifest import discover
    execution = [json.loads(p.read_text()) for p in (a.sidecar / 'state').glob('SP-panel-*.json')]
    tables['wall_time'] = wall_time(cells, inventory, [c for root in a.input for c in discover(root)], execution)
    verification_path = a.sidecar / 'pool-verification.json'
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    from kit import sp_prepare
    frozen_path = sp_prepare.RESOURCES / 'FROZEN.json'
    clean = (verification.get('schema') == 'sp-recheck-v2' and verification.get('status') == 'CLEAN' and
             frozen_path.is_file() and verification.get('frozen_sha256') == sha256_file(frozen_path))
    tables['pool_verification'] = verification or dict(status='UNVERIFIED', reason='returned-pool verification missing')
    if not clean:
        tables['status'] = 'UNVERIFIED'
        for family in tables['families'].values():
            family['verification_status'] = 'UNVERIFIED'
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'sp-tables.json').write_text(json.dumps(tables, sort_keys=True, indent=1) + '\n')
    print('SP tables:', len(cells), 'cells,', len(tables['missing_entries']), 'missing entries;', LABEL)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
