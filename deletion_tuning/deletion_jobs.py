#!/usr/bin/env python3
"""Independent CPU staging and single-allocation deletion score rows."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import fcntl
import json
import os
import time
import uuid

from kit.nm_common import (
    BASE, KIT, SEEDS, Invalid, Pending, atomic, checked, digest, load, need,
    ready, seal, sha,
)
from kit.nm_prepare import audit_bundle, render_scoring, verify_model
from kit.nm_score import scorer_authority, file_stats
from kit.nm_read import Hardware
from deletion_tuning.deletion_score import PANELS_ALL, finite, frontier


def inventory():
    rows = []
    for seed in SEEDS:
        for panel in PANELS_ALL:
            name = f'{seed}-{panel}'
            cells = [dict(model=a, step=k) for a, k in frontier(panel)]
            prep = f'prepare-{name}'
            rows.append(dict(id=prep, phase='prepare', seed=seed, panel=panel,
                             gpus=0, depends_on=[], cells=cells,
                             readiness='only this seed and panel frontier'))
            rows.append(dict(id=f'score-{name}', phase='score', seed=seed, panel=panel,
                             gpus=1, depends_on=[dict(id=prep, type='afterok')], cells=cells,
                             lane=f'{seed}.{panel}', physical_gpus=1,
                             concurrency_cap=2 if panel in ('knowledge', 'hellaswag') else 3))
    rows.append(dict(id='collect', phase='collect', gpus=0,
                     depends_on=[dict(id=r['id'], type='afterany')
                                 for r in rows if r['phase'] == 'score']))
    for row in rows:
        row['command'] = ['{python}', '-I', '{repo}/deletion_tuning/deletion_jobs.py',
                          'run', '--config', '{config}', '--row', row['id']]
    return dict(schema='deletion-jobs.v1', rows=rows,
                scheduling='submit independent lanes immediately; operator owns sbatch')


def inputs(cfg):
    bundle, binding = checked(cfg['bundle']), checked(cfg['binding'])
    ready(binding.get('status') == 'READY', 'whole incumbent binding pending')
    need(binding.get('family') == 'freeze' and not binding.get('benchmark'),
         'expected whole bound layer-freeze incumbent projection')
    return bundle, binding


def prepare(cfg, seed, panel):
    """Read/hash/render on CPU; reserve no GPU until the entire lane is staged."""
    need(not os.environ.get('CUDA_VISIBLE_DEVICES'), 'prepare requires a CPU allocation')
    bundle, binding = inputs(cfg)
    ready(str(seed) in binding.get('recipe', {}).get('starts', {}),
          'bound incoming parent absent')
    ready(panel in bundle.get('panels', {}) and panel in bundle.get('panel_authorities', {}),
          'bound panel absent')
    export_refs = bundle.get('exports', {}).get(str(seed), {})
    ready(all(a in export_refs for a in ('C', 'T')), 'export manifests absent')
    # The inherited source, exclusion and strict-panel audits remain authoritative.
    audit_bundle(bundle, binding)
    items = bundle['panels'][panel]
    need(checked(bundle['panel_authorities'][panel]) == items, 'panel authority differs')
    exports = {a: checked(export_refs[a]) for a in ('C', 'T')}
    sources = {('base', 0): bundle['base'],
               ('incoming', 0): binding['recipe']['starts'][str(seed)]}
    for arm, step in frontier(panel):
        if arm not in ('C', 'T'):
            continue
        ready(str(step) in exports[arm], 'export curve point absent')
        source = exports[arm][str(step)]
        need(source['seed'] == seed and source['arm'] == arm and source['step'] == step,
             'export provenance differs')
        if arm == 'T':
            need(source.get('hard_masked') is True, 'T export must be hard masked')
        sources[arm, step] = source
    need(sources['C', 0]['sha256'] == sources['incoming', 0]['sha256'],
         'C step zero must equal incoming')
    root = Path(cfg['out'])/'plans'/f'{seed}-{panel}'
    root.mkdir(parents=True, exist_ok=True)
    scorer_panel = 'legal' if panel == 'legal-diagnostic' else panel
    authority = scorer_authority(scorer_panel)
    rendered = (None if panel in ('knowledge', 'hellaswag') else
                render_scoring(items, bundle['tokenizer'], scorer_panel))
    commands, plans, checkpoints, verified = [], [], {}, {}
    for index, ((arm, step), source) in enumerate(sources.items()):
        key = (source['model'], source['sha256'])
        if key not in verified:
            verified[key] = verify_model(source)
        src = verified[key]
        checkpoints.setdefault(arm, {})[str(step)] = src['sha256']
        plan = dict(seed=seed, panel=scorer_panel, model=arm, step=step, source=src,
                    items=items, tokenizer=bundle['tokenizer'], metadata=file_stats(src),
                    authority=authority)
        if rendered is not None:
            plan['rendered'] = rendered
        path = root/f'{index}.json'
        seal(path, plan)
        output = f'{index}.score.json'
        plans.append(dict(path=str(path), sha256=sha(path), output=output))
        engine = (dict(backend='hf', profile='reference-bf16') if rendered is None else
                  dict(backend='vllm', model=src['model'],
                       max_model_len=4096 if panel in ('instruction', 'maths') else 12288))
        commands.append(dict(argv=[bundle.get('inference_python', sys.executable), '-I',
                                   str(KIT/'nm_score.py'), '--plan', str(path),
                                   '--out', '{attempt}/'+output],
                             requires=[str(Path(src['model'])/n) for n in src['files']],
                             engine=engine))
    identity = dict(seed=seed, panel=panel, panel_sha256=digest(items),
                    binding=cfg['binding']['sha256'], scorer=authority,
                    implementation={p.name: sha(p) for p in
                                    Path(__file__).parent.glob('deletion_*.py')})
    concurrency = 1
    qualification = bundle.get('score_qualifications', {}).get(panel)
    if qualification:
        q = checked(qualification)
        need(q['binding_sha256'] == digest(identity), 'lane qualification identity differs')
        cap = 2 if rendered is None else 3
        if q.get('numeric_match') is True and q.get('memory_safe') is True:
            need(all(isinstance(q.get(k), str) and q[k] for k in ('uuid', 'environment')),
                 'qualified worker identity absent')
            need(type(q['concurrency']) is int and 1 <= q['concurrency'] <= cap,
                 'unqualified scorer concurrency')
            concurrency = q['concurrency']
    spec = dict(panels={panel: [i['id'] for i in items]},
                scorers={panel: authority}, checkpoints={str(seed): checkpoints})
    seal(root/'lane.json', dict(plans=plans, commands=commands, spec=spec,
                               config_sha256=digest(cfg), identity=identity,
                               concurrency=concurrency, qualification=qualification))
    return root


def score(cfg, row, attempt):
    from kit.lane_pool import execute
    from kit.m3_manifest import allocated_gpus, allocation_reservation
    root = Path(cfg['out'])/'plans'/f'{row["seed"]}-{row["panel"]}'
    pointer = load(Path(cfg['out'])/'state'/f'prepare-{row["seed"]}-{row["panel"]}.json')
    receipt_path = Path(cfg['out'])/pointer['attempt']/'receipt.json'
    need(sha(receipt_path) == pointer['receipt_sha256'], 'CPU preparation receipt changed')
    prepared = load(receipt_path)
    ready(prepared['status'] == 'COMPLETE', 'current CPU preparation incomplete')
    need(prepared['lane_sha256'] == sha(root/'lane.json'), 'CPU lane manifest changed')
    plan = load(root/'lane.json')
    need(plan['config_sha256'] == digest(cfg), 'CPU plan configuration changed')
    need(plan['identity']['implementation'] == {p.name: sha(p) for p in
         Path(__file__).parent.glob('deletion_*.py')}, 'deletion implementation changed')
    for entry in plan['plans']:
        need(sha(entry['path']) == entry['sha256'], 'CPU score plan changed')
    from kit.trace_common import stage_command
    commands = [dict(c, argv=stage_command(
        [v.replace('{attempt}', str(attempt)) for v in c['argv']],
        [e['path'] for e in plan['plans']]))
                for c in plan['commands']]
    ready(all(Path(p).is_file() for c in commands for p in c['requires']),
          'entire scoring frontier must exist before GPU admission')
    ids = allocated_gpus(1)
    if plan['concurrency'] > 1:
        from kit.m2_assignment import inventory as gpu_identity
        qualified = checked(plan['qualification'])
        observed = gpu_identity(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
        need(qualified['uuid'] == observed['uuid'] == ids[0] and
             qualified['environment'] == observed['environment'],
             'score concurrency worker differs from qualification')
    # Keep the scheduler's device ordinal: inherited vLLM rejects UUID selectors.
    env = dict(os.environ, NM_ALLOCATION_NONCE=attempt.name, KIT_LANE_JOB=attempt.name,
               NM_SYNTHETIC='0', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    for key in ('M2_ASSIGNMENT', 'M2_FAMILY', 'M2_FAMILY_SEED', 'M2_ASSIGNED_GPU_UUID',
                'KIT_GPU_MEMORY_FRACTION', 'M3_GPU_LOCK_FDS', 'KIT_M3_MODE'):
        env.pop(key, None)
    with allocation_reservation(ids) as fds:
        env['M3_GPU_LOCK_FDS'] = json.dumps(fds)
        code = execute(commands, attempt/'pool', env=env, pass_fds=fds,
                       concurrency=plan['concurrency'], stagger_seconds=0)
    ready(code not in (3, 75), 'scoring lane pending')
    need(code == 0, 'correctness stop: scorer failed; rerun complete lane')
    cells = [load(attempt/e['output'])['cell'] for e in plan['plans']]
    finite(cells)
    for cell in cells:
        cell['panel'] = row['panel']
    # Validate every cell here, before publishing the one complete attempt.
    from kit.nm_read import Table
    table = Table(dict(cells=cells, technical_synthetic=False), plan['spec'])
    for arm, step in frontier(row['panel']):
        table.cell(row['seed'], row['panel'], arm, step)
    atomic(attempt/'scores.json', dict(cells=cells, technical_synthetic=False))
    atomic(attempt/'spec.json', plan['spec'])
    atomic(attempt/'complete.json', dict(config_sha256=digest(cfg),
                                        scores=sha(attempt/'scores.json'),
                                        spec=sha(attempt/'spec.json')))


def run(cfg, row_id):
    matches = [r for r in inventory()['rows'] if r['id'] == row_id]
    need(len(matches) == 1, 'unknown deletion row')
    row = matches[0]
    need(Path(cfg['out']).is_absolute(), 'absolute output root required')
    out = Path(cfg['out']).resolve()
    lock = out/'locks'/f'{row_id}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Pending('same row running')
        dependencies = {dep['id']: sha(out/'state'/(dep['id']+'.json'))
                        if (out/'state'/(dep['id']+'.json')).is_file() else None
                        for dep in row['depends_on']}
        pointer_path = out/'state'/f'{row_id}.json'
        if pointer_path.exists():
            try:
                pointer = load(pointer_path)
                previous = out/pointer['attempt']
                need(previous.resolve().is_relative_to(out), 'unsafe attempt pointer')
                need(sha(previous/'receipt.json') == pointer['receipt_sha256'],
                     'completed deletion receipt changed')
                receipt = load(previous/'receipt.json')
                if (receipt['status'] == 'COMPLETE' and receipt.get('config_sha256')
                        and receipt.get('dependencies_sha256') == dependencies):
                    need(receipt['config_sha256'] == digest(cfg),
                         'completed deletion configuration changed')
                    for name, expected in receipt.get('artifacts_sha256', {}).items():
                        need(sha(out/name) == expected, 'completed deletion artifact changed')
                    if row['phase'] == 'prepare':
                        prepare(cfg, row['seed'], row['panel'])
                    return 0
            except Pending:
                pass
            except (Invalid, FileNotFoundError):
                return 2
        attempt = out/'attempts'/row_id/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        start = time.time()
        from kit.trace_jobs import timing_phase as phase, timing_finish as finish
        phase_file=attempt/'phase-times.jsonl'
        os.environ.update(V7H_PHASE_FILE=str(phase_file),V7H_PHASE_GPUS=str(row['gpus']))
        record = dict(row=row_id, row_type=row['phase'], status='RUNNING', gpus=row['gpus'],
                      config_sha256=digest(cfg), dependencies_sha256=dependencies)
        code = 0
        try:
            if row['phase'] == 'prepare':
                lane = prepare(cfg, row['seed'], row['panel'])
                record['lane_sha256'] = sha(lane/'lane.json')
            elif row['phase'] == 'score':
                from kit.trace_common import output_tree
                with output_tree(attempt) as local_attempt:
                    # Allocation nonce stays the durable attempt's unique name.
                    with phase('score'):
                        score(cfg, row, local_attempt)
            else:
                from deletion_tuning.deletion_return import collect
                collect(cfg, attempt/'return')
            record['status'] = 'COMPLETE'
            artifacts = list(attempt.rglob('*'))
            if row['phase'] in ('prepare', 'score'):
                lane_root = out/'plans'/f'{row["seed"]}-{row["panel"]}'
                artifacts += list(lane_root.rglob('*.json'))
            record['artifacts_sha256'] = {str(p.relative_to(out)): sha(p)
                for p in artifacts if p.is_file() and p.name != 'receipt.json'}
        except (Pending, FileNotFoundError) as error:
            record.update(status='PENDING/INCOMPLETE', error_type=type(error).__name__)
            code = 75
        except Hardware:
            record.update(status='UNVERIFIED-HARDWARE', error_type='Hardware')
            code = 2
        except Exception as error:
            # Raw diagnostic paths stay local; none enter the pasted return.
            (attempt/'error.txt').write_text(str(error)+'\n')
            stop = 'correctness stop' in str(error)
            record.update(status='PENDING/INCOMPLETE' if stop else 'UNVERIFIED',
                          correctness_stop=stop, error_type=type(error).__name__)
            code = 75 if stop else 2
        finally:
            record.update(started=start, ended=time.time())
            record['phase_gpu_seconds']=finish(phase_file,record['ended']-start,row['gpus'])
            record['process_gpu_hours'] = row['gpus']*(record['ended']-start)/3600
            record['allocation_gpu_hours'] = None if row['gpus'] else 0
            atomic(attempt/'receipt.json', record)
            atomic(out/'state'/f'{row_id}.json', dict(attempt=str(attempt.relative_to(out)),
                                                    receipt_sha256=sha(attempt/'receipt.json')))
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('inventory', 'run'))
    p.add_argument('--config', type=Path)
    p.add_argument('--row')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.action == 'inventory':
        if a.out:
            atomic(a.out, inventory())
        else:
            print(json.dumps(inventory(), indent=2))
        return 0
    need(a.config and a.row, 'config and row required')
    return run(load(a.config), a.row)


if __name__ == '__main__':
    raise SystemExit(main())
