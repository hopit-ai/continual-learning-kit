#!/usr/bin/env python3
"""Unregistered TS inventory and plain per-row operations. Operator owns sbatch."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import subprocess
import uuid

from kit import ts_arms as ts, ts_read
from kit.nm_common import KIT, SEEDS, BASE, need, ready, load, atomic, seal, checked, sha, digest, Pending


def implementation():
    from kit.nm_common import source_seal
    resources = ('rows-ts.json', 'ts-token-fixtures.json', 'ts-agents.yaml',
                 'real-prompt-fixtures.json', 'verl-seams.json')
    files = [*KIT.glob('ts_*.py'), KIT / 'jobs/ts/run-task.sh',
             *(KIT / 'resources/ts' / name for name in resources), KIT / 'README-ts.md']
    return dict(source_seal(), **{str(path.relative_to(KIT.parent)): sha(path) for path in files})


def inventory(bindings=ts.BINDINGS, arms=ts.ARMS):
    need(arms and len(set(arms)) == len(arms) and set(arms) <= set(ts.ARMS), 'unknown or duplicate TS arms')
    arms = tuple(a for a in ts.ARMS if a in arms)
    rows = []
    def add(binding, name, phase, dependencies=(), gpus=0, **identity):
        ident = f'TS-{binding}-{name}'
        afterany = phase in ('read', 'collect')
        rows.append(dict(id=ident, binding=binding, phase=phase, nodes=1, gpus=gpus,
            cpus=32 if gpus == 8 else 8, may_share_node=gpus != 8,
            needs=[] if afterany else list(dependencies), wants=list(dependencies) if afterany else [],
            depends_on=[dict(id=d, type='afterany' if afterany else 'afterok') for d in dependencies],
            dependency_options=['--kill-on-invalid-dep=yes'],
            command=['{env.TS_PYTHON}', '-I', '{kit}/ts_jobs.py', 'run',
                     '--config', '{work}/ts-config.json', '--task', ident],
            env=dict(TS_PYTHON='python3'), timeout_seconds=172800 if gpus == 8 else 14400,
            **identity))
        return ident
    for binding in bindings:
        need(binding in ts.BINDINGS, 'unknown replay binding')
        bound = add(binding, 'bind', 'bind')
        probes = []
        for arm in ('D', 'M'):
            prepared = add(binding, f'qualify-{arm}-prepare', 'qualify-prepare', [bound],
                           arm=arm, seed=101, stage='finqa')
            probes.append(add(binding, f'qualify-{arm}', 'qualify', [prepared], 8,
                              arm=arm, seed=101, stage='finqa'))
        qualified = add(binding, 'qualify-attempt0-read', 'qualify-read', probes)
        exports = {(arm, seed): [] for arm in arms for seed in SEEDS}
        for arm in arms:
            for seed in SEEDS:
                prior = qualified
                for stage in ts.STAGES:
                    name = f'{arm}-s{seed}-{stage}'
                    prepared = add(binding, name + '-prepare', 'prepare', [prior], arm=arm, seed=seed, stage=stage)
                    trained = add(binding, name + '-train', 'train', [prepared], 8, arm=arm, seed=seed, stage=stage)
                    prior = add(binding, name + '-export', 'export', [trained], arm=arm, seed=seed, stage=stage)
                    exports[arm, seed].append(prior)
        scores = []
        for arm in ('base', *arms):
            for seed in SEEDS:
                for panel in ts_read.PANELS:
                    name = f'{arm}-s{seed}-score-{panel}'
                    dependencies = [bound] if arm == 'base' else exports[arm, seed]
                    prepared = add(binding, name + '-prepare', 'score-prepare', dependencies, arm=arm, seed=seed, panel=panel)
                    count = len(ts_read.frontier(panel, () if arm == 'base' else (arm,), include_base=arm == 'base'))
                    scores.append(add(binding, name, 'score', [prepared], 1, arm=arm, seed=seed, panel=panel,
                                      expected_score_cells=count, lane_concurrency=1))
        read = add(binding, 'read', 'read', scores, arms=list(arms))
        add(binding, 'collect', 'collect', [read])
    return dict(schema='kit-campaign.v1', name='teacher-signal-DRAFT', workdir_env='TS_WORK',
                status='DRAFT - NOT FOR SEND', registration=None, replay_binding=None,
                seeds=list(SEEDS), arms=list(arms), rows=rows)


def row_for(task):
    rows = [r for r in inventory()['rows'] if r['id'] == task]
    need(len(rows) == 1, 'invalid TS row/dependency')
    return rows[0]


def stage_root(out, row):
    if row['phase'] in ('qualify-prepare', 'qualify'):
        return out / 'qualification' / row['binding'] / row['arm']
    return out / 'runs' / row['binding'] / f"{row['arm']}-s{row['seed']}" / row['stage']


def lane_path(out, kind, row):
    return out / kind / f"{row['arm']}-s{row['seed']}-{row['panel']}.json"


def dependency(out, task):
    path = out / 'state' / (task + '.json')
    ready(path.is_file(), 'dependency absent: ' + task)
    record = load(path)
    ready(record['status'] == 'COMPLETE', 'dependency incomplete: ' + task)
    for name, expected in record.get('artifacts', {}).items():
        ready((out / name).is_file(), 'dependency artifact missing')
        need(sha(out / name) == expected, 'dependency artifact changed')
    return record


def bundle_for(out, synthetic):
    return {} if synthetic else checked(load(out / 'binding.json')['bundle'])


def score_prepare(cfg, row, out):
    from kit import nm_prepare, nm_score, ts_score
    bundle = bundle_for(out, cfg.get('technical_synthetic') is True)
    synthetic = cfg.get('technical_synthetic') is True
    panel, seed = row['panel'], row['seed']
    items = [dict(id=f'{panel}-{i}') for i in range(10)] if synthetic else checked(bundle['panels'][panel])
    if not synthetic:
        need(len(items) == (1147 if panel == 'finqa' else 100), 'frozen panel denominator differs')
    cells = []
    lane_arm = row['arm']
    frontier = ts_read.frontier(panel, () if lane_arm == 'base' else (lane_arm,), include_base=lane_arm == 'base')
    for arm, stage, step in frontier:
        if arm == 'base':
            from kit.ts_sim import base
            source = base(seed) if synthetic else bundle['base']
        else:
            source = load(stage_root(out, dict(row, arm=arm, stage=stage)) / 'exports.json')[str(step)]
        plan = dict(seed=seed, panel=panel, model=ts_read.model_name(arm, stage), step=step,
                    source=source, items=items, authority=ts_score.authority(panel))
        if not synthetic:
            source = nm_prepare.verify_model(source)
            plan.update(source=source, tokenizer=bundle['tokenizer'], metadata=nm_score.file_stats(source))
            if panel not in ('knowledge', 'hellaswag'):
                from transformers import AutoTokenizer
                from kit.eval_bed import render, render_messages
                tokenizer = AutoTokenizer.from_pretrained(bundle['tokenizer'], local_files_only=True, trust_remote_code=False)
                rendered = [render_messages(tokenizer, r['prompt']) if isinstance(r['prompt'], list) else render(tokenizer, r['prompt']) for r in items]
                cap = 10240 if panel == 'legal' else 4096 if panel == 'finqa' else 2048
                need(all(len(tokenizer.encode(text, add_special_tokens=False)) <= cap for text in rendered), 'scoring prompt over cap')
                plan['rendered'] = rendered
        cells.append(plan)
    path = lane_path(out, 'plans', row)
    seal(path, dict(cells=cells))
    return path


def operation(cfg, row, out, attempt):
    synthetic = cfg.get('technical_synthetic') is True
    phase = row['phase']
    published = []
    if phase == 'bind':
        if synthetic:
            binding = dict(status='TECHNICAL-ONLY', binding=row['binding'], technical_synthetic=True,
                           implementation=implementation())
        else:
            ready(cfg.get('replay_binding') == row['binding'], 'send-time replay binding is PENDING')
            document = checked(cfg['binding'])
            need(document['binding'] == row['binding'] and document['base_revision'] == BASE, 'TS binding identity differs')
            ready(document.get('recipe_receipts'), 'realised replay recipe receipts absent')
            for ref in document['recipe_receipts']:
                checked(ref)
            bundle = checked(document['bundle'])
            need(bundle['base']['revision'] == BASE, 'original base revision differs')
            binding = dict(document, technical_synthetic=False, checker_hashes=ts.checker_hashes(),
                           implementation=implementation())
        seal(out / 'binding.json', binding); published.append(out / 'binding.json')
    elif phase in ('prepare', 'qualify-prepare'):
        if phase == 'prepare':
            qualification = load(out / 'qualification' / row['binding'] / 'attempt0-identity.json')
            arm = qualification['arms'][row['arm']]
            ready(arm['status'] in ('PASS', 'TECHNICAL-ONLY'),
                  row['arm'] + ' technically UNVERIFIED: ' + ', '.join(
                      arm['reasons'] + ['exception=' + name for name in arm.get('exception_types', [])]))
        root = stage_root(out, row)
        root.mkdir(parents=True, exist_ok=True)
        number = ts.STAGES.index(row['stage'])
        bundle = bundle_for(out, synthetic)
        from kit.ts_sim import base
        incoming = (base(row['seed']) if synthetic else bundle['base']) if number == 0 else load(
            root.parent / ts.STAGES[number - 1] / 'exports.json')['40']
        if synthetic:
            prepared = dict(incoming=incoming, config=ts.settings(row['binding'], row['stage']), technical_synthetic=True)
            seal(root / 'prepared.json', prepared)
        else:
            from kit.ts_prepare import prepare
            prepared = prepare(bundle, row['binding'], row['arm'], row['seed'], row['stage'], incoming, root,
                               attempt0_probe=phase == 'qualify-prepare')
        published.append(root / 'prepared.json')
    elif phase == 'qualify':
        root = stage_root(out, row)
        prepared = load(root / 'prepared.json')
        if synthetic:
            count = ts.settings(row['binding'], 'finqa')['task_rows']
            probe = dict(schema='ts-attempt0-probe.v2', binding=row['binding'], arm=row['arm'], seed=101,
                         update=1, optimizer_updates=0, incoming_sha256=prepared['incoming']['sha256'],
                         schedule_sha256=digest(['CPU-STANDIN', row['binding']]), technical_synthetic=True,
                         slots=[dict(position=i, question_id=str(i), token_ids=[i, 8],
                                     request=dict(prompt_token_ids=[1, i], sampling_params=dict(temperature=1., top_p=1.)),
                                     **(dict(finish_reasons=['stop'] * 8, masked_reason=None) if row['arm'] == 'M' else {}))
                                for i in range(1, count + 1)])
            seal(root / 'native-receipt.json', probe)
        else:
            ready(cfg.get('send_approval'), 'separate qualification run approval absent')
            need(all(sha(root / 'data' / name) == value for name, value in prepared['files'].items()), 'probe schedule changed')
            done = subprocess.run(prepared['command'], cwd=prepared['cwd'], env=dict(os.environ, **prepared['env']))
            need(done.returncode == 0, 'native attempt-zero probe failed')
        published.append(root / 'native-receipt.json')
    elif phase == 'qualify-read':
        from kit.ts_qualification import compare
        probes = [load(out / 'qualification' / row['binding'] / arm / 'native-receipt.json') for arm in ('D', 'M')]
        control = checked(cfg['attempt0_d_control']) if cfg.get('attempt0_d_control') else None
        result = compare(*probes, control=control)
        path = out / 'qualification' / row['binding'] / 'attempt0-identity.json'
        seal(path, result); published.append(path)
    elif phase == 'train':
        root = stage_root(out, row)
        prepared = load(root / 'prepared.json')
        if synthetic:
            from kit.ts_sim import train
            train(row, prepared['incoming'], root, missing_checker=cfg.get('missing_checker_task') == row['id'])
        else:
            ready(cfg.get('send_approval'), 'separate run approval absent')
            need(all(sha(root / 'data' / name) == value for name, value in prepared['files'].items()), 'prepared schedule changed')
            # Resume an interrupted native stage through the pinned state path.
            command = list(prepared['command'])
            tracker = root / 'native' / 'latest_checkpointed_iteration.txt'
            step = 0
            if tracker.is_file():
                step = int(tracker.read_text().strip())
                command = [v for v in command if v.split('=', 1)[0].lstrip('+') not in ('trainer.resume_mode', 'trainer.resume_from_path')]
                command += ['++trainer.resume_mode=resume_path', '++trainer.resume_from_path=' + str(root / 'native' / f'global_step_{step}')]
            if row['arm'] != 'D':
                from kit.ts_masks import supersede
                supersede(root, step)
            done = subprocess.run(command, cwd=prepared['cwd'], env=dict(os.environ, **prepared['env']))
            need(done.returncode == 0, 'native TS training failed')
        result = load(root / 'native-receipt.json')
        need(result['status'] == 'COMPLETE' and result['updates'] == 40 and result['steps'] == dict(optimizer=40, scheduler=40, ema=40),
             'native TS step counts differ')
        published.append(root / 'native-receipt.json')
        if row['arm'] != 'D':
            masks = sorted((root / 'mask-history').glob('update-*.json'))
            need(len(masks) == 40, 'forty update mask ledgers required')
            published.extend(masks)
    elif phase == 'export':
        root = stage_root(out, row)
        if synthetic:
            seal(root / 'exports.json', load(root / 'native-receipt.json')['exports'])
        else:
            from kit.nm_prepare import export_stage
            export_stage(load(root / 'prepared.json'), root, cfg.get('job_python', sys.executable))
        published.append(root / 'exports.json')
    elif phase == 'score-prepare':
        published.append(score_prepare(cfg, row, out))
    elif phase == 'score':
        from kit import ts_score
        plans = load(lane_path(out, 'plans', row))['cells']
        cells, costs = [], []
        os.environ['NM_ALLOCATION_NONCE'] = attempt.name
        for index, plan in enumerate(plans):
            if synthetic:
                items = {r['id']: int((i + (0 if plan['model'] == 'base' else plan['step'] // 10)) % 4 != 0) for i, r in enumerate(plan['items'])}
                cell = dict(seed=plan['seed'], panel=plan['panel'], model=plan['model'], step=plan['step'], items=items,
                    checkpoint_sha256=plan['source']['sha256'], nonce=attempt.name, uuid='CPU-STANDIN', environment='CPU-STANDIN',
                    scorer_sha256=plan['authority']['sha256'], method=plan['authority']['method'], precision='full',
                    status='COMPLETE', technical_synthetic=True)
                costs.append(dict(model=cell['model'], step=cell['step'], output_tokens=50, correct=sum(items.values()), truncated=0))
            else:
                path = attempt / f'plan-{index}.json'; atomic(path, plan)
                ts_score.score(path, attempt / f'score-{index}.json')
                scored = load(attempt / f'score-{index}.json'); cell = scored['cell']
                raw = scored.get('raw', [])
                costs.append(dict(model=cell['model'], step=cell['step'], output_tokens=sum(r['output_tokens'] for r in raw) if raw else None,
                                  correct=sum(cell['items'].values()), truncated=sum(r['finish_reason'] == 'length' for r in raw) if raw else None))
            cells.append(cell)
        for cost in costs:
            cost['tokens_per_correct'] = cost['output_tokens'] / cost['correct'] if cost['correct'] and cost['output_tokens'] is not None else None
        target = lane_path(out, 'scores', row)
        seal(target, dict(cells=cells, costs=costs)); published.append(target)
    elif phase == 'read':
        from kit.ts_read import evaluate
        cells, costs = [], {}
        if load(out / 'binding.json')['implementation'] != implementation():
            atomic(out / 'reading.json', dict(status='UNVERIFIED', binding=row['binding'],
                   reason='implementation changed after binding', technical_synthetic=synthetic))
            return [out / 'reading.json']
        arms = tuple(row.get('arms', ts.ARMS))
        missing = []
        for seed in SEEDS:
            for panel in ts_read.PANELS:
                lane_costs = []
                for arm in ('base', *arms):
                    lane = dict(row, arm=arm, seed=seed, panel=panel)
                    path = lane_path(out, 'scores', lane)
                    task = f"TS-{row['binding']}-{arm}-s{seed}-score-{panel}"
                    state = out / 'state' / (task + '.json')
                    if not path.is_file() or not state.is_file() or load(state)['status'] != 'COMPLETE':
                        missing.append(f'{arm}.{seed}.{panel}')
                        continue
                    dependency(out, task)
                    result = load(path)
                    cells.extend(result['cells']); lane_costs.extend(result['costs'])
                base_cost = next((c for c in lane_costs if c['model'] == 'base'), {})
                base_tokens = base_cost.get('output_tokens')
                for cost in lane_costs:
                    ratio = cost['output_tokens'] / base_tokens if base_tokens and cost['output_tokens'] is not None else None
                    cost.update(base_serving_ratio=ratio,
                                serving_ceiling='UNDEFINED' if ratio is None else 'AT COST' if ratio > 1.5 else 'WITHIN 1.5x')
                costs[f'{seed}.{panel}'] = lane_costs
        spec = dict(panels={}, checkpoints={}, scorers={})
        for cell in cells:
            spec['panels'].setdefault(cell['panel'], list(cell['items']))
            spec['checkpoints'].setdefault(str(cell['seed']), {}).setdefault(cell['model'], {})[str(cell['step'])] = cell['checkpoint_sha256']
            spec['scorers'][cell['panel']] = dict(sha256=cell['scorer_sha256'], method=cell['method'])
        if not synthetic:
            # Independent plan authorities prevent accepting a self-certified score.
            for seed in SEEDS:
                for panel in ts_read.PANELS:
                    for arm in ('base', *arms):
                        path = lane_path(out, 'plans', dict(row, arm=arm, seed=seed, panel=panel))
                        if not path.is_file():
                            continue
                        for p in load(path)['cells']:
                            spec['panels'][panel] = [r['id'] for r in p['items']]
                            spec['scorers'][panel] = p['authority']
                            spec['checkpoints'][str(seed)].setdefault(p['model'], {})[str(p['step'])] = p['source']['sha256']
        result = evaluate(dict(cells=cells, technical_synthetic=synthetic), spec, arms=arms, allow_missing=True)
        qualification = out / 'qualification' / row['binding'] / 'attempt0-identity.json'
        if qualification.is_file():
            result['qualification'] = load(qualification)
            for arm in arms:
                if result['qualification']['arms'][arm]['status'] == 'UNVERIFIED':
                    result['arms'][arm] = 'UNVERIFIED'
                    result['status'] = 'UNVERIFIED'
        result['missing'] = missing
        from kit.nm_pipeline import generation_utilisation
        utilisation = {}
        util_root = Path(cfg.get('util_root', os.environ.get('TS_WORK', str(out)))) / 'util'
        for lane in inventory((row['binding'],), arms)['rows']:
            if not lane['gpus']:
                continue
            summaries = list((util_root / lane['id']).glob('*.summary.json'))
            latest = max(summaries, key=lambda p: p.stat().st_mtime_ns) if summaries else util_root / 'missing'
            utilisation[lane['id']] = generation_utilisation(latest)
        warnings = [task + '/' + warning for task, record in utilisation.items() for warning in record['warnings']]
        result.update(binding=row['binding'], costs=costs, checker_hashes=ts.checker_hashes(), technical_synthetic=synthetic,
                      accounting_status='TECHNICAL-ONLY zero GPU cost' if synthetic else 'PENDING scheduler allocation-end accounting',
                      utilisation=utilisation, utilisation_warnings=warnings)
        histories = {}
        for arm in arms:
            for seed in SEEDS:
                for stage in ts.STAGES:
                    path = stage_root(out, dict(row, arm=arm, seed=seed, stage=stage)) / 'native-receipt.json'
                    task = f"TS-{row['binding']}-{arm}-s{seed}-{stage}-train"
                    state = out / 'state' / (task + '.json')
                    if not path.is_file() or not state.is_file() or load(state)['status'] != 'COMPLETE':
                        continue
                    dependency(out, task)
                    receipt = load(path)
                    histories[f'{arm}.{seed}.{stage}'] = receipt['history']
        result['training_telemetry'] = histories
        from kit.ts_masks import overlap
        result['mask_overlap_S_M'] = {}
        if {'S', 'M'} <= set(arms):
            for seed in SEEDS:
                for stage in ts.STAGES:
                    left = stage_root(out, dict(row, arm='S', seed=seed, stage=stage))
                    right = stage_root(out, dict(row, arm='M', seed=seed, stage=stage))
                    try:
                        result['mask_overlap_S_M'][f'{seed}.{stage}'] = overlap(left, right)
                    except Pending:
                        result['mask_overlap_S_M'][f'{seed}.{stage}'] = dict(status='PENDING')
        atomic(out / 'reading.json', result); published.append(out / 'reading.json')
    elif phase == 'collect':
        reading_task = f"TS-{row['binding']}-read"
        reading_state = out / 'state' / (reading_task + '.json')
        if reading_state.is_file() and load(reading_state)['status'] == 'COMPLETE':
            dependency(out, reading_task)
            result = load(out / 'reading.json')
        else:
            result = dict(status='PENDING/INCOMPLETE', binding=row['binding'], technical_synthetic=synthetic)
        from kit.ts_text import return_parts
        parts = return_parts(result)
        for index, part in enumerate(parts, 1):
            path = out / 'text' / f'RETURN-TS-{index}.txt'
            path.parent.mkdir(exist_ok=True)
            path.write_text(part); published.append(path)
        path = out / 'return.txt'
        path.write_text('\n'.join(parts)); published.append(path)
    else:
        raise ValueError('unknown TS phase')
    return published


def run(config, task):
    cfg = load(config)
    matches = [r for r in inventory(arms=cfg.get('arms', ts.ARMS))['rows'] if r['id'] == task]
    need(len(matches) == 1, 'invalid TS row/dependency')
    row = matches[0]
    out = Path(cfg['out']) / row['binding']
    need(out.is_absolute(), 'absolute TS output root required')
    nonce = uuid.uuid4().hex
    attempt = out / 'attempts' / task / nonce
    attempt.mkdir(parents=True, exist_ok=False)
    record = dict(task=task, status='RUNNING', technical_synthetic=cfg.get('technical_synthetic') is True, artifacts={})
    code = 0
    import fcntl
    lock = out / 'locks' / (task + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # A competing attempt must never replace the active row's pointer.
            record.update(status='PENDING', reason='same TS row already running', exit_code=75)
            atomic(attempt / 'receipt.json', record)
            return 75
        try:
            record['inputs'] = {ref['id']: digest(load(out / 'state' / (ref['id'] + '.json')))
                                if (out / 'state' / (ref['id'] + '.json')).is_file() else None
                                for ref in row['depends_on']}
            for ref in row['depends_on']:
                if ref['type'] == 'afterok':
                    parent = dependency(out, ref['id'])
                    need(parent['technical_synthetic'] == record['technical_synthetic'], 'TS provenance changed')
            if row['phase'] not in ('bind', 'read', 'collect'):
                binding = load(out / 'binding.json')
                need(binding['technical_synthetic'] == record['technical_synthetic'], 'TS binding provenance changed')
                need(binding['implementation'] == implementation(), 'TS implementation changed after binding')
            state = out / 'state' / (task + '.json')
            previous = load(state) if state.is_file() else {}
            if (previous.get('status') == 'COMPLETE' and previous.get('inputs') == record['inputs']
                    and row['phase'] not in ('read', 'collect')):
                previous = dependency(out, task)
                need(previous['technical_synthetic'] == record['technical_synthetic'], 'TS output provenance changed')
                record.update(status='COMPLETE', artifacts=previous['artifacts'], reused_complete=True)
            else:
                published = operation(cfg, row, out, attempt)
                record.update(status='COMPLETE', artifacts={str(p.relative_to(out)): sha(p) for p in published})
        except (Pending, FileNotFoundError) as error:
            record.update(status='PENDING', reason=str(error)); code = 75
        except (ValueError, KeyError, TypeError) as error:
            record.update(status='FAILED', reason=str(error)); code = 2
        record['exit_code'] = code
        atomic(attempt / 'receipt.json', record)
        atomic(out / 'state' / (task + '.json'), record)
    print(json.dumps(dict(task=task, status=record['status'], exit_code=code)))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inventory', 'run', 'resources'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--binding', choices=ts.BINDINGS)
    parser.add_argument('--arms', nargs='+', choices=ts.ARMS, default=ts.ARMS)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--task')
    args = parser.parse_args()
    if args.action == 'resources':
        need(args.task, 'task required')
        print(row_for(args.task)['gpus'])
        return 0
    if args.action == 'inventory':
        result = inventory((args.binding,) if args.binding else ts.BINDINGS, args.arms)
        if args.out:
            atomic(args.out, result)
        else:
            print(json.dumps(result, indent=2))
        return 0
    need(args.config and args.task, 'config and task required')
    return run(args.config, args.task)


if __name__ == '__main__':
    raise SystemExit(main())
