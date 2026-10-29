#!/usr/bin/env python3
"""Bind W at send from completed unreset B reading and hash-bound start receipts."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.c_common import *
from kit.p3b_read import analyse, load, lower


def choose(reading):
    """C §3 lexicographic rule, including F+replay's last-place tie priority."""
    ranked = []
    summaries = reading['summaries']
    for recipe in RECIPES:
        seq = next((r for r in reading['sequence'] if r['recipe'] == recipe and r['panel'] == 'math_screen'), None)
        curves = [r for r in reading['curves'] if r['recipe'] == recipe and r['route'] == 'sequence' and r['panel'] == 'math_screen']
        if seq is None or len(curves) != 5 or {r['seed'] for r in curves} != set(SEEDS):
            raise NotReady('incomplete unreset B tier 2: ' + recipe)
        # FinQA and every general panel must exist at boundaries and midpoint.
        for arm, steps in [(f'sequence-{recipe}', RETENTION_STEPS), (f'standalone-{recipe}', RETENTION_STEPS)]:
            if any(not any(r['arm'] == arm and r['seed'] == seed and r['step'] == step and r['panel'] == panel
                           for r in summaries) for seed in SEEDS for step in steps
                   for panel in ('finqa', 'ifeval', 'knowledge', 'math')):
                raise NotReady('incomplete B retention evidence: ' + arm)
        tier1 = recipe if recipe in ('D', 'F') else 'finqa-' + recipe
        if any(not any(r['arm'] == tier1 and r['seed'] == seed and r['step'] == 40 and r['panel'] == panel
                       for r in summaries) for seed in SEEDS for panel in ('finqa', 'ifeval', 'knowledge', 'math')):
            raise NotReady('incomplete unreset B tier 1: ' + recipe)
        margins = [lower(seq['final_vs_standalone'], -2), lower(seq['stage2_retention']['finqa'], -1)]
        margins += [lower(seq[key][p], -3) for key in ('stage1_vs_base', 'stage2_retention') for p in ('ifeval', 'knowledge', 'math')]
        ratios = []
        for panel in ('finqa', 'math_screen'):
            group = [r for r in summaries if r['arm'] == 'sequence-' + recipe and r['step'] == 40 and r['panel'] == panel]
            if len(group) != 5:
                raise NotReady('undefined/incomplete B serving cost: ' + recipe)
            ratios.append(sum(r['serving_ratio'] if r['serving_ratio'] is not None else 1e300 for r in group) / 5)
        row = dict(recipe=recipe, joint_success=bool(seq['passes']), confirmed_retention_margins=sum(margins),
                   mean_sequence_math_final=sum(r['final'] for r in curves) / 5,
                   max_task_serving_ratio=max(ratios), task_serving_ratios=dict(zip(('finqa', 'math_screen'), ratios)))
        row['rank'] = [int(row['joint_success']), sum(margins), row['mean_sequence_math_final'], -max(ratios), -TIE_ORDER.index(recipe)]
        ranked.append(row)
    ranked.sort(key=lambda r: r['rank'], reverse=True)
    return dict(W=ranked[0]['recipe'], provisional=not ranked[0]['joint_success'], ranking=ranked,
                why='Lexicographic: confirmed joint B acquisition/retention, confirmed retention margins, mean sequence MATH final, lower maximum of mean per-seed task serving ratios; ties F,D,D+replay,F+replay. No reset or recovery enters. B tier 2 is forgetting/recipe evidence only, limited by headroom (base 89/100); no second-task acquisition claim.')


def check_start(record, recipe, seed, standin):
    if standin: return dict(record, identity=dict(technical_synthetic=True, model=record['model']))
    require(record.get('receipt'), 'exact original export receipt required')
    receipt = json.loads(__import__('kit.m2_paths',fromlist=['local']).local(record['receipt']).read_text())
    origin = Path(receipt.get('model_dir', ''))
    slot = f'{recipe[0]}-finqa-alone-s{seed}'
    require(slot in origin.parts and origin.name == 'hf-step40', 'wrong original FinQA final: ' + slot)
    return dict(record, receipt_sha256=s.file_hash(record['receipt']), identity=identity(record['model'], receipt=record['receipt']))


def job(a):
    protocol()
    if not a.b_reading.exists() or not a.b_source.exists(): raise NotReady('completed unreset B return pending')
    rows, _ = load(a.b_source)
    if not rows: raise NotReady('no B scores')
    supplied = json.loads(a.b_reading.read_text())
    require(bool(supplied.get('technical_synthetic')) == a.standin, 'B reading synthetic mismatch')
    computed = analyse(rows, repeats=20)  # bootstrap is descriptive; seed bars unchanged
    for key in ('summaries', 'sequence', 'curves'):
        # Question bootstrap intervals in sequence are irrelevant to the seed ranking.
        if key == 'sequence':
            fields = ('recipe', 'panel', 'passes', 'final_vs_standalone', 'stage1_vs_base', 'stage2_retention')
            project = lambda xs: [{k: r[k] for k in fields} for r in xs]
            require(project(supplied[key]) == project(computed[key]), 'B reading differs from score evidence: ' + key)
        else: require(supplied[key] == computed[key], 'B reading differs from score evidence: ' + key)
    decision = choose(computed)
    starts_in = json.loads(a.checkpoints.read_text()); starts = {}
    require('base' in starts_in, 'missing frozen base path')
    base = starts_in['base']; starts['base'] = dict(base, identity=checked_model(base, a.standin, base=True))
    for recipe in set([decision['W'][0], 'F']):
        for seed in SEEDS:
            key = f'{recipe}:{seed}'
            require(key in starts_in, 'missing start ' + key)
            starts[key] = check_start(starts_in[key], recipe, seed, a.standin)
    original_v4 = [r for r in computed['summaries'] if r['arm'] in ('base','D','F') and r['panel'] in ('finqa','ifeval','knowledge','math')]
    original_v4_pairs = [r for r in computed['question_pairs'] if r['arm'] in ('D','F') and r['panel'] in ('finqa','ifeval','knowledge','math') and r['step']==40]
    doc = dict(**decision, starts=starts, protocol=protocol(), start_policy=START_POLICY,
               original_v4_stage1=dict(summaries=original_v4,base_comparisons=original_v4_pairs),
               b_reading_sha256=s.file_hash(a.b_reading), b_scores_sha256=s.digest(rows),
               technical_synthetic=a.standin)
    bind(a.out / 'binding.json', doc)
    print('W =', doc['W'], '(provisional)' if doc['provisional'] else '(confirmed joint B bars)')
    print(doc['why']); print('Starts:', START_POLICY)


def configure(p):
    p.add_argument('--b-reading', type=Path, required=True); p.add_argument('--b-source', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, required=True)
if __name__ == '__main__': raise SystemExit(entry(job, configure, 'c-bind'))
