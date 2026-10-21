#!/usr/bin/env python3
"""Bind a completed W decision and the available receipt-checked finals; no model inference."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.next_common import *
import shutil


def sibling(w):
    require(w in RECIPES, 'unknown W')
    return w[0] + '+replay'


def job(a):
    frozen_next()
    decision = json.loads(a.w_identity.read_text())
    require(decision.get('recipe') in RECIPES, 'unknown W')
    if decision.get('complete_unreset') is not True:raise NotReady('W pending: need completed unreset B decision')
    require(decision.get('technical_synthetic', False) == a.standin, 'W type mismatch')
    w = decision['recipe']; finals = []; missing = []; starts = {}
    base = checkpoint(a.base, a.base_receipt, a.standin, base=True)
    if a.base_receipt is None:
        a.base_receipt=a.out/'next-inputs/base-receipt.json'
        bind(a.base_receipt,base)
    for recipe in RECIPES:
        for seed in SEEDS:
            if recipe in ('D', 'F'):
                paths = sorted(a.v4_exports.glob(f'**/{recipe}-finqa-alone-s{seed}/hf-step40'))
                require(len(paths) <= 1, 'ambiguous v4 final')
                path = paths[0] if paths else None
                receipt = a.v4_work / f'runs/{recipe}-finqa-alone-s{seed}/env/export-identity-step40.json'
            else:
                path = a.package_b / f'runs/finqa-{recipe}-s{seed}/hf-step40'
                receipt = path.parent / 'export-40.json'
            name = f'finqa-{recipe}-s{seed}'
            if path is None or not path.exists() or not receipt.exists(): missing.append(name); continue
            model = checkpoint(path, receipt, a.standin)
            starts[(recipe, seed)] = dict(path=str(path.resolve()), receipt=str(receipt.resolve()), identity=model)
            finals.append(dict(name=name, recipe=recipe, seed=seed, route='finqa',
                start=dict(path=str(a.base.resolve()), receipt=str(a.base_receipt.resolve()) if a.base_receipt else None, identity=base),
                final=starts[(recipe, seed)]))
    if len(starts)!=20:raise NotReady('all 20 tier-1 FinQA finals are required; missing: '+', '.join(missing))
    for recipe in RECIPES:
        for route in ('standalone', 'sequence'):
            for seed in SEEDS:
                name = f'{route}-{recipe}-s{seed}'; path = a.package_b / 'runs' / name / 'hf-step40'
                receipt = path.parent / 'export-40.json'
                if not path.exists() or not receipt.exists(): missing.append(name); continue
                require(route != 'sequence' or (recipe, seed) in starts, 'sequence start missing')
                final = checkpoint(path, receipt, a.standin)
                if not a.standin and route == 'sequence':
                    ident = json.loads((a.package_b / 'identities' / (name + '.json')).read_text())
                    require(ident['incoming']['model_file_hashes'] == starts[(recipe,seed)]['identity']['model_file_hashes'], 'sequence start identity differs')
                finals.append(dict(name=name, recipe=recipe, seed=seed, route=route,
                    start=starts[(recipe,seed)] if route=='sequence' else dict(path=str(a.base.resolve()), receipt=str(a.base_receipt.resolve()) if a.base_receipt else None, identity=base),
                    final=dict(path=str(path.resolve()), receipt=str(receipt.resolve()), identity=final)))
    if any((r, seed) not in starts for r in (w, sibling(w)) for seed in SEEDS):
        raise NotReady('W and replay-sibling tier-1 finals must all be present')
    # These are immutable copies of B inputs; no new sampling or generic refresher.
    for name in ('replay', 'targets'):
        source = a.package_b / name; doc = frozen_files(source)
        require(doc['technical_synthetic'] == a.standin, 'B input type mismatch')
        for file in set(doc['files']) | {'manifest.json', 'inputs.json'}:
            src = source / file
            if not src.exists(): continue
            dst = a.out / name / file
            if dst.exists(): check(dst, s.file_hash(src))
            else: dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
    value = dict(w=w, w_decision=decision, w_decision_sha256=s.file_hash(a.w_identity),
        base=dict(path=str(a.base.resolve()), receipt=str(a.base_receipt.resolve()) if a.base_receipt else None, identity=base),
        sec_recipe=sibling(w), sec=[starts[(sibling(w), seed)] for seed in SEEDS],
        probe_starts=[starts[(w[0],seed)] for seed in SEEDS], finals=finals, missing_slots=missing,
        technical_synthetic=a.standin)
    root=a.out/'next-inputs'; bind(root/'inputs.json', value)
    tables={
        'sec.tsv':[(0,value['base'])]+list(zip(SEEDS,value['sec'])),
        'probe.tsv':list(zip(SEEDS,value['probe_starts'])),
    }
    for name,records in tables.items():
        require(all(not any(char in record[key] for char in ('\t','\n')) for seed,record in records for key in ('path','receipt')), 'TSV paths contain tabs/newlines')
        text=''.join(f"{seed}\t{record['path']}\t{record['receipt']}\n" for seed,record in records)
        path=root/name
        if path.exists():require(path.read_text()==text,'path table changed')
        else:path.write_text(text)
    text=''.join(r['name']+'\n' for r in finals)
    path=root/'screen.txt'
    if path.exists():require(path.read_text()==text,'screen slots changed')
    else:path.write_text(text)
    freeze(root, ['inputs.json','sec.tsv','probe.tsv','screen.txt']+(['base-receipt.json'] if (root/'base-receipt.json').exists() else []), a.standin)
    print('Bound W:', w, '; finals:', len(finals), '; missing slots:', len(missing))


def configure(p):
    for name in ('w-identity','base','package-b','v4-work','v4-exports'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--base-receipt', type=Path)
if __name__ == '__main__': raise SystemExit(entry(job, configure, 'next-bind'))
