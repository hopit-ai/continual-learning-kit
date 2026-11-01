#!/usr/bin/env python3
"""Score decision checkpoint guards with the unchanged validation reference implementations.

Checkpoint list: JSON array of {arm, seed, step, stage (M3 only)}. Paths and
model hashes are resolved from the native binding/export receipts, never the list.
Each invocation scores one (family, seed, panel) sequentially on its frozen GPU.
"""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os
import time
from contextlib import nullcontext
from kit.p3b_common import atomic, bind, files_manifest, frozen_files, require, s
from kit.ref_guard import METHOD, PANELS, ASSIGNMENT


def source_for(out, family, checkpoint, standin):
    arm, seed, step = (checkpoint[k] for k in ('arm', 'seed', 'step'))
    if family == 'm3':
        from kit.m3_schedule import ARMS, STAGES
        require(arm in ('base', *ARMS) and checkpoint.get('stage') in STAGES, 'unregistered M3 checkpoint')
        from kit.m3_score import checked
        return checked(out, arm, seed, checkpoint['stage'], step, standin)
    if family == 'core':
        from kit.m2_common import binding, record, checked_model
        doc = binding(out, standin); source = record(out, doc, arm, seed, step)
        from kit.m2_paths import local
        source = dict(source, model=str(local(source['model'])), receipt=str(local(source['receipt'])) if source.get('receipt') else None)
        return source, checked_model(source, standin, base=arm == 'base')
    from kit.x_common import context, incoming, checkpoint as exported
    from kit.m2_common import checked_model
    doc = context(out, standin)
    if arm == 'base':
        core = json.loads((Path(doc['message2']) / 'binding.json').read_text())
        source = core['starts']['base']
        from kit.m2_paths import local
        source = dict(source, model=str(local(source['model'])), receipt=str(local(source['receipt'])) if source.get('receipt') else None)
        return source, checked_model(source, standin, base=True)
    native_arm = arm.removeprefix('x-')
    pre = native_arm == 'pre' or native_arm.endswith('-pre')
    expected = 'x-tier2.' + native_arm[3:].rsplit('-', 1)[0] if native_arm.startswith('t2-') else 'x-tier1'
    require(family == expected, 'reference checkpoint belongs to another X contrast')
    source = incoming(doc, seed) if pre else exported(out, doc, native_arm, seed, step)
    return source, checked_model(source, standin, base=False)


def assign(a):
    from kit.m2_assignment import load, freeze
    native = a.from_assignment or a.out / ('m3-scoring-assignment.json' if a.family == 'm3' else 'scoring-assignment.json')
    doc = load(native)
    assignments = {f'{family}.{panel}|{seed}': value for key, value in doc['assignments'].items()
                   for family, seed in [key.rsplit('|', 1)] for panel in PANELS}
    require(assignments, 'no native assignments for family')
    freeze(a.out / ASSIGNMENT, dict(schema='message2-scoring-assignment.v2', assignments=assignments))


def run(a):
    keys = ('M2_ASSIGNMENT', 'M2_FAMILY', 'M2_FAMILY_SEED', 'CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'M2_ASSIGNED_GPU_UUID')
    previous = {k: os.environ.get(k) for k in keys}
    try: return _run(a)
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def _run(a):
    from kit import m2_assignment
    from kit.m3_score import normalized_assignment_inventory, knowledge_items, knowledge_reference, standin_likelihoods, selected
    from kit import m2_format
    from kit.hellaswag_reference import score as hellaswag_reference
    from kit.p3b_score import write_csv
    from kit.c_score import COLUMNS
    require(a.panel in PANELS and a.seed in (101, 102, 103), 'registered panel/seed required')
    path = a.out / ASSIGNMENT
    os.environ.update(M2_ASSIGNMENT=str(path.resolve()), M2_FAMILY=a.family + '.' + a.panel, M2_FAMILY_SEED=str(a.seed))
    scope = normalized_assignment_inventory() if a.family == 'm3' else nullcontext()
    with scope:
        proof = m2_assignment.active(a.out, a.gpu, a.standin)
    checkpoints = json.loads(a.checkpoints.read_text())
    if isinstance(checkpoints, dict): checkpoints = checkpoints['checkpoints']
    checkpoints = [c for c in checkpoints if c['seed'] == a.seed]
    require(checkpoints, 'no checkpoints for assigned seed')
    require(all(c['step'] in (0, 40) for c in checkpoints), 'reference job accepts decision checkpoints (0/40) only')
    require(len({(c['arm'], c.get('stage'), c['step']) for c in checkpoints}) == len(checkpoints), 'duplicate checkpoint')
    population = knowledge_items() if a.panel == 'knowledge-ll' else m2_format.items(a.out, a.standin)
    # Use precisely the tokenizer employed by the native validation.
    if a.family == 'm3':
        from kit.m3_train import base_record
        tokenizer = base_record(a.out)['model']
    else:
        doc = json.loads((a.out / 'binding.json').read_text())
        if a.family == 'core':
            from kit.m2_paths import local
            tokenizer = str(local(doc.get('base', doc.get('starts', {}).get('base'))['model']))
        else:
            # X retains the original M binding as its tokenizer authority.
            from kit.m2_paths import local
            core = json.loads((Path(doc['message2']) / 'binding.json').read_text())
            tokenizer = str(local(core.get('base', core.get('starts', {}).get('base'))['model']))
    if not a.standin: m2_format.tokenizer_identity(tokenizer)
    for checkpoint in checkpoints:
        start = time.monotonic()
        source, observed = source_for(a.out, a.family, checkpoint, a.standin)
        arm = checkpoint['arm']; step = checkpoint['step']; stage = checkpoint.get('stage')
        unit = arm if a.family != 'm3' or arm == 'base' else f'{arm}.{stage}'
        root = a.out / 'reference-scores' / a.family / f'{unit}-s{a.seed}-k{step}-{a.panel}'
        identity = dict(schema='reference-guard.v1', method=METHOD, family=a.family, arm=arm, seed=a.seed, step=step,
                        stage=stage, panel=a.panel, panel_sha256=s.digest(population), model=observed,
                        assignment=proof, gpu_identity=proof, technical_synthetic=a.standin,
                        implementation_sha256=s.file_hash(Path(__file__).with_name('hellaswag_reference.py' if a.panel == 'hellaswag' else 'm3_score.py')))
        if (root / 'manifest.json').exists():
            frozen_files(root)
            require(json.loads((root / 'ref-identity.json').read_text()) == identity, 'reference resume identity changed')
            continue
        if a.standin: likelihood = standin_likelihoods(population, root.name)
        elif a.panel == 'hellaswag': likelihood = hellaswag_reference(source['model'], population, tokenizer)
        else: likelihood = knowledge_reference(source['model'], population, tokenizer)
        require(len(likelihood) == len({r['id'] for r in likelihood}) == 100 and
                {r['id'] for r in likelihood} == {r['id'] for r in population}, 'reference likelihood coverage incomplete')
        gold = {r['id']: int(r['label']) for r in population}
        panel = 'knowledge' if a.panel == 'knowledge-ll' else 'hellaswag'
        rows = [dict(arm=unit, seed=a.seed, step=step, task=panel, panel=panel, cap=2048, question_id=r['id'],
                     correct=int(selected(r['scores']) == gold[r['id']]), format_free_correct=int(selected(r['scores']) == gold[r['id']]),
                     output_tokens=0, finished=1) for r in likelihood]
        bind(root / 'ref-identity.json', identity)
        write_csv(root / 'scores.csv', [{k: r.get(k, '') for k in COLUMNS} for r in rows], COLUMNS)
        atomic(root / 'likelihoods.json', dict(items=population, rows=likelihood))
        atomic(root / 'cost.json', dict(kind='reference_guard_scoring', method=METHOD, seed=a.seed, panel=a.panel,
                                       gpu_hours=0 if a.standin else (time.monotonic() - start) / 3600, technical_synthetic=a.standin))
        atomic(root / 'manifest.json', files_manifest(root, ['ref-identity.json', 'scores.csv', 'likelihoods.json', 'cost.json']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('assign', 'score'), required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--family', required=True)
    p.add_argument('--from-assignment', type=Path)
    p.add_argument('--checkpoints', type=Path); p.add_argument('--seed', type=int); p.add_argument('--panel', choices=PANELS)
    p.add_argument('--gpu', default='0'); p.add_argument('--standin', action='store_true')
    a = p.parse_args()
    if a.action == 'assign': assign(a)
    else: run(a)


if __name__ == '__main__': main()
