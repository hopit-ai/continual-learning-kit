#!/usr/bin/env python3
"""Three-page deletion return and CPU-only count-based manager reader."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import math
import re

from kit.nm_common import (
    PANELS, SEEDS, STEPS, Invalid, Pending, atomic, checked, digest, load, need,
    ready, sha,
)
from kit.trace_read import pack, pages, parse_pages
from kit.nm_read import Hardware
from deletion_tuning.deletion_score import (
    PANELS_ALL, finite, from_counts, lr_evidence, read,
)


def allocated_cost(evidence):
    accounting = evidence.get('accounting', {})
    hours = accounting.get('gpu_hours')
    if (accounting.get('status') == 'COMPLETE' and accounting.get('all_attempts') is True
            and type(hours) in (int, float) and math.isfinite(hours) and hours >= 0):
        return hours
    return 'PENDING'


def framed(records, commit, kind='ESSENTIAL'):
    """Reuse TRACE's ASCII pager and framing, relabeling its descriptive scope."""
    result = []
    for page in pages(dict(records=records), commit, kind):
        lines = page.splitlines()
        lines[0] = lines[0].replace('KIT TRACE', 'KIT DELETION')
        lines[1] = kind+' REGISTERED df=2'
        body = '\n'.join(lines[1:-1])+'\n'
        lines[-1] = 'SHA256 '+hashlib.sha256(body.encode('ascii')).hexdigest()
        result.append('\n'.join(lines)+'\n')
    return result


def part3(cfg, binding, evidence, lr):
    recipe = binding.get('recipe', {})
    changes = evidence.get('changes', ['UNMEASURED'])
    need(isinstance(changes, list) and changes and
         all(isinstance(c, str) and re.fullmatch('[A-Z0-9_-]{1,48}', c) for c in changes),
         'changes must be privacy-safe codes')
    rate = lr.get('rate', 'PENDING')
    fallback = str(lr.get('fallback', 'PENDING')).lower()
    rows = ['DELETION|kit|'+cfg['kit_commit'],
            'DELETION|recipe|'+digest(recipe),
            'DELETION|parents|'+digest(recipe.get('starts', {})),
            'DELETION|schedules|'+digest(recipe.get('schedule_hashes', {})),
            'DELETION|lr|receipt='+lr.get('sha256', 'PENDING'),
            f'DELETION|lr|rate={rate}|fallback={fallback}',
            'DELETION|cost|allocated_gpu_h='+str(allocated_cost(evidence))]
    from kit.trace_jobs import timing_rows as phase_rows
    rows.extend(key+'='+json.dumps(value,separators=(',',':'))
                for key,value in phase_rows(cfg['out']).items())
    rows.extend('DELETION|changes|'+c for c in changes)
    body = '\n'.join(['ESSENTIAL REGISTERED df=2', 'TRANSCRIPTION-UNVERIFIED',
                      'CHECKSUM ADVISORY', *rows])+'\n'
    return (f'KIT DELETION {cfg["kit_commit"]} PART 3/3\n'+body+
            'SHA256 '+hashlib.sha256(body.encode('ascii')).hexdigest()+'\n')


def gather(cfg):
    """Only the current complete attempt is eligible; failed retries hide old data."""
    root = Path(cfg['out'])
    cells, specs, missing = [], dict(panels={}, scorers={}, checkpoints={}), []
    serving = []
    for seed in SEEDS:
        for panel in PANELS_ALL:
            name = f'score-{seed}-{panel}'
            try:
                pointer = load(root/'state'/f'{name}.json')
                attempt = root/pointer['attempt']
                ready(attempt.resolve().is_relative_to(root.resolve()), 'unsafe attempt pointer')
                need(sha(attempt/'receipt.json') == pointer['receipt_sha256'], 'receipt changed')
                receipt = load(attempt/'receipt.json')
                if receipt['status'] == 'UNVERIFIED-HARDWARE':
                    raise Hardware('current lane hardware unverified')
                need(receipt['status'] != 'UNVERIFIED', 'current lane identity unverified')
                ready(receipt['status'] == 'COMPLETE', 'current lane incomplete')
                complete = load(attempt/'complete.json')
                need(complete['config_sha256'] == digest(cfg), 'lane config changed')
                need(sha(attempt/'scores.json') == complete['scores'] and
                     sha(attempt/'spec.json') == complete['spec'], 'lane artifact changed')
                doc, spec = load(attempt/'scores.json'), load(attempt/'spec.json')
                from deletion_tuning.deletion_score import frontier
                expected = {(seed, panel, a, k) for a, k in frontier(panel)}
                observed = [(c['seed'], c['panel'], c['model'], c['step']) for c in doc['cells']]
                need(len(observed) == len(expected) and set(observed) == expected,
                     'lane frontier differs')
                cells.extend(doc['cells'])
                for raw_path in sorted(attempt.glob('*.score.json')):
                    raw_doc = load(raw_path)
                    cell, raw = raw_doc['cell'], raw_doc.get('raw', [])
                    correct = sum(cell['items'].values())
                    tokens = sum(r['output_tokens'] for r in raw) if raw else None
                    serving.append(dict(seed=seed, panel=panel, model=cell['model'],
                        step=cell['step'], correct=correct, output_tokens=tokens,
                        tokens_per_correct=tokens/correct if tokens is not None and correct else None))
                for group in ('panels', 'scorers'):
                    for key, value in spec[group].items():
                        need(key not in specs[group] or specs[group][key] == value,
                             'panel/scorer binding differs between lanes')
                        specs[group][key] = value
                dest = specs['checkpoints'].setdefault(str(seed), {})
                for arm, steps in spec['checkpoints'][str(seed)].items():
                    for step, value in steps.items():
                        prior = dest.setdefault(arm, {})
                        need(step not in prior or prior[step] == value,
                             'checkpoint binding differs between panels')
                        prior[step] = value
            except (FileNotFoundError, Pending):
                missing.append(f'{seed}.{panel}')
    for row in serving:
        base = next((r['tokens_per_correct'] for r in serving if r['seed'] == row['seed']
                     and r['panel'] == row['panel'] and r['model'] == 'base'), None)
        ratio = row['tokens_per_correct']/base if base and row['tokens_per_correct'] is not None else None
        row.update(ratio_to_base=ratio, at_cost=ratio > 1.5 if ratio is not None else 'UNMEASURED')
    return dict(cells=cells, technical_synthetic=False,
                serving=serving or 'UNMEASURED'), specs, missing


def build_records(reading, cfg, evidence, lr):
    records = dict(schema='deletion-return.v1', kit=cfg['kit_commit'],
                   status=reading['status'], technical=reading['status'] == 'TECHNICAL-ONLY',
                   correctness_stop=reading.get('correctness_stop', False),
                   damage_units='accuracy-percent',
                   lr=lr.get('sha256', 'PENDING'),
                   lr_choice=[lr.get('rate'), lr.get('fallback')],
                   evidence=digest(evidence), counts_scope='exact-integer-correct-counts')
    records['allocated_gpu_h'] = allocated_cost(evidence)
    pack(records, 'warnings', reading.get('warnings', []))
    for key, counts in reading.get('correct_counts', {}).items():
        records['C.'+key] = counts
    for lane, identity in reading.get('lanes', {}).items():
        records['L.'+lane] = identity
    if reading.get('primary'):
        pack(records, 'primary', reading['primary']['values'])
        for key, guard in reading['guards'].items():
            pack(records, 'guard.'+key, guard['values'])
    optional = {}
    for key in ('recipe_sha256', 'parents_sha256', 'schedules_sha256'):
        records[key] = evidence.get(key, 'PENDING')
    for seed, q in evidence.get('seeds', {}).items():
        canary = q.get('canary', {})
        if all(k in canary for k in ('incoming', 'initial_soft', 'initial_hard', 'teacher')):
            pack(records, 'damage.'+seed,
                 [canary[k] for k in ('incoming', 'initial_soft', 'initial_hard', 'teacher')])
            records['canary.'+seed] = [canary['n'], canary['ids_sha256']]
        for step, layers in q.get('masks', {}).items():
            records[f'masks.{seed}.{step}'] = [sum(v[k] for v in layers.values()) for k in
                ('weights', 'hard_zeros', 'keep_to_delete', 'delete_to_keep')]
        optional['mask_hash.'+seed] = digest(q.get('masks', {}))
    for index, trial in enumerate(lr.get('receipt', {}).get('trials', [])):
        pack(records, 'trial.'+str(index), [trial[k] for k in
             ('rate', 'loss_before', 'loss_after', 'loss_drop', 'initial_damage', 'max_flip_fraction')])
    for key, values in reading.get('initial_scores', {}).items():
        pack(records, 'initial.'+key, values)
    for key, values in reading.get('step20_scores', {}).items():
        pack(optional, 'step20.'+key, values)
    # FULL always retains all diagnostics, including every layer and LR trial.
    full_records = {**records, **optional}
    records['full'] = digest(full_records)
    # A count-based decision and initial damage are never spilled.
    need(len(framed(records, cfg['kit_commit'])) <= 2, 'mandatory return exceeds two pages')
    for key, value in sorted(optional.items()):
        candidate = {**records, key: value}
        if len(framed(candidate, cfg['kit_commit'])) <= 2:
            records[key] = value
    return records, full_records


def collect(cfg, destination):
    from deletion_tuning.deletion_jobs import inputs
    need(re.fullmatch('[0-9a-f]{40}', cfg['kit_commit']), 'full public commit required')
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    full = out/'full'
    full.mkdir()
    evidence, lr, binding, scores, spec = {}, {}, {}, {}, {}
    missing = []
    try:
        _, binding = inputs(cfg)
        evidence = checked(cfg['evidence'])
        finite(evidence)
        lr = lr_evidence(cfg['lr_receipt'])
        scores, spec, missing = gather(cfg)
        reading = read(scores, spec, binding, evidence, lr)
        if missing and reading['status'] == 'PENDING/INCOMPLETE' and not reading.get('correctness_stop'):
            reading = dict(status='PENDING/INCOMPLETE', missing=missing)
    except Hardware:
        reading = dict(status='UNVERIFIED-HARDWARE')
    except (Pending, FileNotFoundError) as error:
        reading = dict(status='PENDING/INCOMPLETE', reason=type(error).__name__)
    except (Invalid, KeyError, ValueError, TypeError) as error:
        stop = 'non-finite' in str(error)
        reading = dict(status='PENDING/INCOMPLETE' if stop else 'UNVERIFIED',
                       correctness_stop=stop, reason=type(error).__name__)
    # NaNs never become printable or gain a scientific endpoint.
    if reading.get('correctness_stop'):
        evidence = {}
    records, full_records = build_records(reading, cfg, evidence, lr)
    text = framed(records, cfg['kit_commit'])
    while len(text) < 2:
        text.append(framed({}, cfg['kit_commit'])[0])
    text = [re.sub(r'PART \d+/\d+', f'PART {i}/3', page, count=1)
            for i, page in enumerate(text, 1)]
    text.append(part3(cfg, binding, evidence, lr))
    need(all(len(p.splitlines()) <= 74 and all(len(line) < 90 and line.isascii()
             for line in p.splitlines()) for p in text), 'paste capacity exceeded')
    for i, page in enumerate(text, 1):
        (out/f'return-{i}.txt').write_text(page, encoding='ascii')
    docs = dict(scores=scores, spec=spec, binding=binding, evidence=evidence,
                lr_receipt=lr.get('receipt', {}), reading=reading)
    def safe(value):
        # Retain invalid observations locally, while writing standards-compliant JSON.
        if isinstance(value, float) and not math.isfinite(value):
            return 'NONFINITE:'+str(value)
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [safe(v) for v in value]
        return value
    for name, doc in docs.items():
        atomic(full/(name+'.json'), safe(doc))
    # Original receipts/raw outputs remain in the attempt tree; bind every retry.
    atomic(full/'attempts.json', {str(p.relative_to(Path(cfg['out']))): sha(p)
        for p in Path(cfg['out']).glob('attempts/**/*')
        if p.is_file() and not p.resolve().is_relative_to(out.resolve())})
    atomic(full/'inputs.json', cfg)
    for i, page in enumerate(framed(full_records, cfg['kit_commit'], 'FULL'), 1):
        (full/f'return-{i}.txt').write_text(page, encoding='ascii')
    atomic(full/'manifest.json', dict(files={p.name: sha(p) for p in full.iterdir()},
                                     missing=missing))
    local = parse([p for p in text])
    if reading.get('primary'):
        for key in ('primary', 'guards'):
            need(local[key] == reading[key], 'count and item-level readings differ')
    atomic(out/'reading.json', reading)
    return reading


def parse(texts, expect_commit=None):
    """Checksums are advisory. Counts, not pasted verdicts or bounds, decide bars."""
    result = dict(status='PENDING/INCOMPLETE', transcription='TRANSCRIPTION-UNVERIFIED',
                  checksums='ADVISORY', incumbent_changed=False)
    try:
        frames, part3_rows = {}, []
        for text in texts:
            lines = text.splitlines()
            need(lines and all(len(line) < 90 for line in lines), 'invalid paste width')
            match = re.fullmatch(r'KIT DELETION ([0-9a-f]{40}) PART ([123])/3', lines[0])
            need(match, 'invalid deletion frame')
            commit, number = match.groups()
            need(expect_commit in (None, commit), 'unexpected kit commit')
            need(all(c == commit for c, _ in frames.values()), 'mixed kit commits')
            need(number not in frames, 'duplicate return page')
            frames[number] = (commit, text)
            if number == '3':
                part3_rows = [line for line in lines if line.startswith('DELETION|')]
        ready(set(frames) == {'1', '2', '3'}, 'missing ESSENTIAL page')
        payload = parse_pages([frames[k][1] for k in ('1', '2')])
        result.update({k: v for k, v in payload.items() if k in
                       ('checksum_mismatches', 'invalid_records')})
        records = payload['records']
        finite(records)
        # json.loads otherwise permits non-finite literals and overflow numbers.
        for k in ('1', '2'):
            for line in frames[k][1].splitlines():
                if '=' in line:
                    try:
                        value = json.loads(line.split('=', 1)[1])
                    except ValueError:
                        continue
                    finite(value)
        need(not payload.get('invalid_records'), 'damaged return records')
        need(records.get('schema') == 'deletion-return.v1' and
             records.get('kit') == commit, 'return identity differs')
        expected_keys = {'kit', 'recipe', 'parents', 'schedules', 'lr', 'cost', 'changes'}
        need({r.split('|')[1] for r in part3_rows} == expected_keys, 'Part 3 keys incomplete')
        need(part3_rows.count('DELETION|kit|'+commit) == 1, 'Part 3 kit differs')
        ready('DELETION|lr|receipt='+records['lr'] in part3_rows, 'Part 3 LR seal differs')
        for name in ('recipe', 'parents', 'schedules'):
            if records[name+'_sha256'] != 'PENDING':
                need(part3_rows.count('DELETION|'+name+'|'+records[name+'_sha256']) == 1,
                     'Part 3 '+name+' differs')
        need(part3_rows.count('DELETION|cost|allocated_gpu_h='+str(records['allocated_gpu_h'])) == 1,
             'Part 3 allocation cost differs')
        rate, fallback = records['lr_choice']
        if rate is not None:
            need(part3_rows.count(f'DELETION|lr|rate={rate}|fallback={str(fallback).lower()}') == 1,
                 'Part 3 LR choice differs')
        third = frames['3'][1].splitlines()
        if not third[-1].startswith('SHA256 ') or third[-1][7:] != hashlib.sha256(
                ('\n'.join(third[1:-1])+'\n').encode('ascii')).hexdigest():
            result.setdefault('checksum_mismatches', []).append(3)
        result['part3'] = part3_rows
        status = records.get('status')
        if status in ('PENDING/INCOMPLETE', 'UNVERIFIED', 'UNVERIFIED-HARDWARE'):
            result.update(status=status, correctness_stop=records.get('correctness_stop', False))
            return result
        need(status in ('PASS', 'INCONCLUSIVE', 'TECHNICAL-ONLY'), 'unknown reported status')
        counts = {k[2:]: v for k, v in records.items() if k.startswith('C.')}
        lanes = {k[2:]: v for k, v in records.items() if k.startswith('L.')}
        need(all(re.fullmatch('[0-9a-f]{64}', v) for v in lanes.values()),
             'invalid lane identity digest')
        need(type(records['technical']) is bool, 'invalid technical provenance')
        for seed in map(str, SEEDS):
            ready('damage.'+seed+'.0' in records and 'canary.'+seed in records,
                  'initial canary damage absent')
            for step in map(str, STEPS):
                ready(f'masks.{seed}.{step}' in records, 'mask export summary absent')
        result.update(from_counts(counts, lanes, technical=records['technical']))
        result['records'] = records
        return result
    except (Pending, FileNotFoundError) as error:
        result.update(status='PENDING/INCOMPLETE', reason=str(error))
    except (Invalid, KeyError, TypeError, ValueError) as error:
        stop = 'non-finite' in str(error)
        result.update(status='PENDING/INCOMPLETE' if stop else 'UNVERIFIED',
                      correctness_stop=stop, reason=str(error))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--config', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('read')
    r.add_argument('--text', type=Path, nargs='+', required=True)
    r.add_argument('--expect-commit')
    r.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.action == 'collect':
        result = collect(load(a.config), a.out)
    else:
        result = parse([path.read_text() for path in a.text], a.expect_commit)
        if a.out:
            atomic(a.out, result)
    print(result['status'])
    return 75 if result['status'].startswith('PENDING') else 2 if result['status'].startswith('UNVERIFIED') else 0


if __name__ == '__main__':
    raise SystemExit(main())
