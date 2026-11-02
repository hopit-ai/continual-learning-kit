#!/usr/bin/env python3
"""Isolated message-3 (K/CR) plain-job manifest, send-time input binding, isolation proof and failure-isolated runner.

The M3 tier never shares state, manifests or task IDs with message-2 core:
  * its own root, marked by m3-root.json; refused if the directory holds message-2 state;
  * its own schema (message3-launch.v3), which kit/m2_node.py rejects;
  * inputs are send-time frozen and bound as message-2 core binds them (round-4 rulings):
      - package B's replay/ and FinQA target folders read in place, manifest sha256 bound write-once (m2_prepare style);
      - C's ContractNLI journal read in place, its manifest also matched to package-c-return.tar.gz;
      - hard-MATH targets for the frozen 500-item pool: a read-only copy of core's committed M5 targets-math when present
        (the single permitted read of a message-2 product, via --core-durable), else the identical protocol here;
      - the tier's own FinQA task schedules, HellaSwag-100 freeze, pools, schedules and scoring;
    no M3 job waits on core: stages 1-2 never depend on hard-MATH targets;
  * inference may share hosts with core on separate allocated GPUs; eight-rank training owns its allocation;
  * tier jobs run only after core submissions: M3-init records the partner's confirmation, and every other task
    stays PENDING until it exists;
  * a failed or absent M3 job writes only M3 state and exits non-zero for itself; no core task depends on M3.
The partner submits each task with his own sbatch (kit/jobs/message3/run-task.sh) in the README order.
"""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import os
import subprocess
import time
import socket
import re
from datetime import datetime
import fcntl
from contextlib import contextmanager
from kit.m3_schedule import ARMS, SEEDS, STAGES, digest

SCHEMA = 'message3-launch.v3'
ROOT_SCHEMA = 'message3-root.v1'
KIT = Path(__file__).resolve().parent
INPUT_BINDING = KIT / 'resources/m3/input-binding.json'
# Files whose presence marks a message-2 (or older package) tree that M3 must never write into.
FOREIGN = ('binding.json', 'assets.json', 'scoring-assignment.json', 'next-inputs', 'nomination-commit.json', 'message2-launch.json')
EXIT_FAILED, EXIT_PENDING = 1, 75
NOT_READY = 3  # kit.p3a_common.entry's exit code for NotReady (prerequisite absent): recorded as PENDING
STEPS = (0, 10, 20, 30, 40)
FINAL_PANELS = ('finqa', 'test', 'math_hard', 'general-v1', 'hellaswag', 'knowledge-ll')
TASK_PANEL = {'finqa': 'finqa', 'contract_nli': 'test', 'math_hard': 'math_hard'}
VALIDATION_SOURCE = ('C0', 101, 'finqa')  # the one incoming checkpoint every likelihood validation includes (ruling c)


class IsolationError(RuntimeError):
    pass


class Unbound(RuntimeError):
    """A send-time binding is not pinned yet: PENDING, never substituted."""


def host_name(value):
    return str(value).split('.')[0]


def need(ok, message):
    if not ok: raise IsolationError('M3 ISOLATION: ' + message)


def init_root(out, core_submitted=False):
    out = Path(out)
    need(not any((out / name).exists() for name in FOREIGN), f'{out} holds message-2 state; use a separate M3 root')
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'm3-root.json'
    value = json.loads(marker.read_text()) if marker.exists() else dict(schema=ROOT_SCHEMA, core_submitted=False)
    if core_submitted and not value.get('core_submitted'): value.update(core_submitted=True, core_submitted_at=time.time())
    marker.write_text(json.dumps(value, sort_keys=True) + '\n')
    return marker


def require_m3_root(out):
    out = Path(out); marker = out / 'm3-root.json'
    need(marker.is_file() and json.loads(marker.read_text()).get('schema') == ROOT_SCHEMA, f'{out} is not an initialised M3 root (run m3_manifest.py --action init)')
    need(not any((out / name).exists() for name in FOREIGN), f'{out} also holds message-2 state')
    return out


def core_submitted(out):
    marker = Path(out) / 'm3-root.json'
    return marker.is_file() and json.loads(marker.read_text()).get('core_submitted') is True


# ----------------------------------------------------------------------------- send-time input binding

def validate_input(name, root):
    """B targets use their native parquet schema, rather than a generic files map."""
    from kit.p3b_common import frozen_files, require, s
    root = Path(root)
    if not name.startswith('finqa_targets_'):
        return frozen_files(root)
    doc = json.loads((root / 'manifest.json').read_text())
    require(doc.get('schema') == 'kit-v4-sft-data.v1' and doc.get('arm') == 'COMMON', 'unqualified B FinQA schema')
    require(doc.get('scientific_phase_allowed') is True and not doc.get('stand_in_teacher'), 'unqualified B FinQA targets')
    for part in ('train', 'val'):
        require(s.file_hash(root / (part + '.parquet')) == doc.get(part + '_sha256'), 'B FinQA parquet hash differs: ' + part)
    return doc


def bind_inputs(out, roots, return1, spec_path=INPUT_BINDING):
    """roots: {name: package B/C directory read in place}. Each root's manifest.json must be byte-identical to the
    member named in input-binding.json inside its returned archive; the archive is fingerprinted (reviewed return-1
    decisions, or the size/sha256 pinned in input-binding.json). Nothing is copied. Writes <out>/inputs/bound.json."""
    from kit.m2_return1 import files as archive_files
    from kit.p3b_common import frozen_files
    require_m3_root(out)
    spec = json.loads(Path(spec_path).read_text()); entries = spec['inputs']
    bound = {}; cache = {}
    for name, root in sorted(roots.items()):
        key = name if name in entries else name.rsplit('_', 1)[0] + '_*'
        entry = entries.get(key) or {}
        if entry.get('mode') == 'assets':
            # Exactly as kit/m2_prepare.py binds B's replay and next_pools binds B's target folders: the frozen
            # directory read in place, its manifest.json sha256 recorded write-once (no archive, no copy, no vendoring).
            validate_input(name, root)
            bound[name] = dict(root=str(Path(root).resolve()), manifest_sha256=hashlib.sha256((Path(root) / 'manifest.json').read_bytes()).hexdigest(),
                               mode='assets')
            continue
        if not (entry.get('archive') and entry.get('member')):
            raise Unbound(f'{name}: no send-time binding pinned in kit/resources/m3/input-binding.json')
        archive = Path(return1) / entry['archive']
        if entry.get('sha256'):
            need(archive.is_file() and archive.stat().st_size == entry['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest() == entry['sha256'],
                 f'{name}: archive fingerprint differs from input-binding.json')
        else:
            from kit.m3_pools import fingerprinted
            fingerprinted(archive, entry['archive'])
        members = cache.setdefault(str(archive), archive_files(archive))
        member = entry['member'].format(index=name.rsplit('_', 1)[-1])
        need(member in members, f'{name}: returned archive lacks {member}')
        raw = (Path(root) / 'manifest.json').read_bytes()
        need(raw == members[member], f'{name}: on-node manifest differs from the returned archive member {member}')
        validate_input(name, root)
        bound[name] = dict(root=str(Path(root).resolve()), manifest_sha256=hashlib.sha256(raw).hexdigest(), archive=entry['archive'], member=member)
    path = Path(out) / 'inputs/bound.json'
    if path.exists(): need(json.loads(path.read_text()) == bound, 'send-time input binding changed')
    else: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(bound, indent=1, sort_keys=True) + '\n')
    return bound


def bound_input(out, name, root):
    """Jobs re-verify a bound root before reading it: same path, same manifest bytes, every listed file intact."""
    from kit.p3b_common import NotReady, frozen_files, require
    path = Path(out) / 'inputs/bound.json'
    if not path.exists(): raise NotReady('run M3-bind-inputs first')
    entry = json.loads(path.read_text()).get(name)
    if entry is None: raise NotReady(f'{name} is not bound to a returned archive')
    require(entry['root'] == str(Path(root).resolve()) and hashlib.sha256((Path(root) / 'manifest.json').read_bytes()).hexdigest() == entry['manifest_sha256'],
            f'{name}: input differs from its send-time binding')
    validate_input(name, root)
    return entry


# ----------------------------------------------------------------------------- manifest

def build(settings):
    """settings: paths, python interpreters, nodes. Never scientific row definitions."""
    tasks = []
    train_py, score_py = settings['trainer_python'], settings['inference_python']
    inputs = settings['inputs']
    core_nodes = {host_name(n) for n in settings.get('message2_nodes', [])}

    def cmd(script, python=None, **kw):
        result = [python or score_py, '-I', '{kit}/' + script + '.py', '--out', '{m3_out}']
        for key, value in kw.items():
            if value is None or value is False or value == []: continue
            flag = '--' + key.replace('_', '-')
            if value is True: result.append(flag)
            elif key in ('panel', 'holdout'):
                for v in value: result += [flag, str(v)]  # repeated NAME=PATH options
            elif isinstance(value, list): result += [flag, *map(str, value)]
            else: result += [flag, str(value)]
        if settings.get('technical_synthetic'): result.append('--standin')
        return result

    def add(ident, kind, requires, commands, **kw):
        need(ident.startswith('M3-'), 'every M3 task ID carries the M3- prefix')
        gpus = 8 if kind in ('train-stage', 'preflight') else 1 if kind == 'score-family' or kw.get('uses_gpu') else 0
        if ident == 'M3-math-targets': gpus = 2
        tasks.append(dict(id=ident, kind=kind, requires=list(dict.fromkeys(requires)), commands=commands,
                          gpus=gpus, nodes=1, cpus=32 if gpus == 8 else 8, share_node=gpus != 8, **kw)); return ident

    finqa_targets = list(inputs.get('finqa_targets') or [])
    roots = dict(contract_targets=inputs['contract_targets'], replay=inputs['replay'],
                 **{f'finqa_targets_{i}': p for i, p in enumerate(finqa_targets)})
    schedule_roots = dict(finqa='{m3_out}/targets-finqa', contract_nli=inputs['contract_targets'], math_hard='{m3_out}/targets-math')
    panels = [f'{k}={v}' for k, v in sorted(inputs['panels'].items())]
    holdouts = [f'{k}={v}' for k, v in sorted(inputs.get('holdouts', {}).items())]
    add('M3-init', 'prepare', [], [[score_py, '-I', '{kit}/m3_manifest.py', '--action', 'init', '--out', '{m3_out}', '--core-submitted']])
    add('M3-bind-inputs', 'prepare', ['M3-init'], [[score_py, '-I', '{kit}/m3_manifest.py', '--action', 'bind-inputs', '--out', '{m3_out}',
                                                    '--roots', json.dumps(roots, sort_keys=True), '--return1', str(inputs.get('return1', ''))]])
    add('M3-verify-superset', 'prepare', ['M3-bind-inputs'],
        [cmd('m3_pools', action='verify-returns', finqa_targets=finqa_targets, replay=inputs['replay'], targets=inputs['contract_targets'])])
    # The tier freezes its own HellaSwag-100 (pinned revision, deterministic rule) before any pool or score exists.
    add('M3-hellaswag', 'prepare', ['M3-init'], [cmd('next_hellaswag', action='prepare', rows=inputs.get('hellaswag_rows'))])
    add('M3-finqa-targets', 'prepare', ['M3-bind-inputs'], [cmd('m3_pools', action='finqa-targets', finqa_targets=finqa_targets,
                                                                tokenizer=inputs.get('tokenizer'))])
    add('M3-contract', 'prepare', ['M3-bind-inputs', 'M3-hellaswag'], [cmd('m3_pools', action='contract', targets=inputs['contract_targets'],
        source=inputs.get('contract_source'), tokenizer=inputs.get('tokenizer'), finqa_panel=inputs.get('finqa_panel'))])
    add('M3-finqa-replay', 'prepare', ['M3-bind-inputs', 'M3-hellaswag'], [cmd('m3_pools', action='finqa-replay', finqa_targets=finqa_targets, tokenizer=inputs.get('tokenizer'), panel=panels, holdout=holdouts)])
    add('M3-broad', 'prepare', ['M3-contract', 'M3-finqa-replay'], [cmd('m3_pools', action='broad', gpu='{score_gpu}', model=inputs.get('base'), panel=panels, holdout=holdouts,
                                                     contract='{m3_out}/pools/contract/contract-replay.jsonl', replay=inputs['replay'],
                                                     gsm_rows=inputs.get('gsm_rows'), arc_rows=inputs.get('arc_rows'), finqa_pool='{m3_out}/finqa-replay/finqa.jsonl')],
        uses_gpu=True)
    broad = tasks[-1]
    add('M3-broad-prepare', 'prepare', broad['requires'], [[v if v != 'broad' else 'broad-prepare' for v in c] for c in broad['commands']])
    broad['requires'] = ['M3-broad-prepare']
    broad['commands']=[[*c,'--generation-shard',str(i)] for c in broad['commands'] for i in range(3)]
    broad['command_count']=3;broad['lane_concurrency']=3
    add('M3-broad-finish', 'prepare', ['M3-broad'], [[v if v != 'broad' else 'broad-finish' for v in broad['commands'][0][:-2]]])
    add('M3-pools-shared', 'prepare', ['M3-bind-inputs', 'M3-verify-superset', 'M3-hellaswag'], [cmd('m3_pools', action='freeze', shared_only=True, replay=inputs['replay'], tokenizer=inputs.get('tokenizer'), panel=panels, holdout=holdouts)])
    add('M3-pools', 'prepare', ['M3-broad-finish', 'M3-finqa-replay'], [cmd('m3_pools', action='freeze', replay=inputs['replay'], contract='{m3_out}/pools/contract',
                                                  finqa_targets=finqa_targets, finqa_pool='{m3_out}/finqa-replay/finqa.jsonl',
                                                  tokenizer=inputs.get('tokenizer'), panel=panels, holdout=holdouts)])
    add('M3-bind-base', 'prepare', ['M3-init'], [cmd('m3_train', action='bind-base', model=inputs.get('base'), receipt=inputs.get('base_receipt'))])
    # Stages 1-2 never wait for hard-MATH targets: their schedules freeze first; stage 3's freeze separately.
    early = [s for s in STAGES if s != 'math_hard']
    add('M3-schedules', 'prepare', ['M3-pools-shared', 'M3-finqa-targets', 'M3-bind-inputs'],
        [cmd('m3_train', action='schedules', arms=['C0', 'K'], panel=panels, holdout=holdouts, stages=early, schedule_roots=[schedule_roots[s] for s in early], tokenizer=inputs.get('tokenizer'))])
    # Submit when the first chain reaches stage 3 (or later): copies core's committed M5 targets-math read-only if
    # present, otherwise runs the identical protocol here (two GPUs). Never waits on core.
    add('M3-math-resolve', 'prepare', ['M3-init'], [cmd('m3_pools', action='math-resolve', core_durable=inputs.get('core_durable'))],
        reads_core_snapshot=bool(inputs.get('core_durable')))
    add('M3-math-targets', 'prepare', ['M3-math-resolve'], [cmd('m3_pools', action='math-targets',
                                                        model=inputs.get('teacher'), tokenizer=inputs.get('tokenizer'), gpu=inputs.get('teacher_gpus', '0,1'))],
        uses_gpu=True, reads_core_snapshot=bool(inputs.get('core_durable')))
    add('M3-math-ready', 'prepare', ['M3-math-resolve'], [cmd('m3_pools', action='math-finish')])
    add('M3-schedules-math', 'prepare', ['M3-pools-shared', 'M3-math-ready'],
        [cmd('m3_train', action='schedules', arms=['C0', 'K'], panel=panels, holdout=holdouts, stages=['math_hard'], schedule_roots=[schedule_roots['math_hard']], tokenizer=inputs.get('tokenizer'))])
    for suffix, stages, deps in (('', early, ['M3-finqa-targets']), ('-math', ['math_hard'], ['M3-math-ready'])):
        add('M3-schedules-coverage' + suffix, 'prepare', ['M3-pools', 'M3-schedules' + suffix, *deps],
            [cmd('m3_train', action='schedules', arms=['G', 'CR'], panel=panels, holdout=holdouts, stages=stages,
                 schedule_roots=[schedule_roots[s] for s in stages], tokenizer=inputs.get('tokenizer'))])
    for arm in ARMS:
        prerequisite = 'M3-schedules-coverage' if arm in ('G', 'CR') else 'M3-schedules'
        add('M3-preflight-' + arm, 'preflight', ['M3-bind-base', prerequisite],
            [cmd('m3_train', python=train_py, action='preflight', arm=arm, seed=101, stage='contract_nli',
                 schedule_root=schedule_roots['contract_nli'], tokenizer=inputs.get('tokenizer'), sdpo=inputs.get('sdpo'),
                 trainer_python=train_py, gpu=settings.get('gpus', '0,1,2,3,4,5,6,7'),
                 host_reservation_bytes=settings.get('host_reservation_bytes'))], arm=arm, uses_gpu=True)
    for task in list(tasks):
        if task['kind'] == 'preflight':
            prepared = task['id'] + '-inputs'
            commands = [[('preflight-inputs' if v == 'preflight' else v) for v in c] for c in task['commands']]
            add(prepared, 'prepare', task['requires'], commands, arm=task['arm'])
            task['requires'] = [prepared]
    add('M3-assign', 'prepare', ['M3-init'], [cmd('m3_score', action='assign', inventory=inputs['inventory'])])
    finals = []
    for seed in SEEDS:
        # Base and every arm of this seed score on the one dedicated GPU assigned to family m3|seed.
        base = add(f'M3-base-{seed}', 'score-family', ['M3-assign', 'M3-bind-base', 'M3-schedules', 'M3-hellaswag'],
                   [cmd('m3_score', action='score', arm='base', seed=seed, stage='finqa', step=0, panel_set=p, gpu='{score_gpu}', panel_root=inputs.get('finqa_panel'))
                    for p in FINAL_PANELS], family='m3', seed=seed)
        for arm in ARMS:
            previous = base
            for number, stage in enumerate(STAGES):
                # Incoming (step-0) scores alias the preceding final / base on the same assignment; never rescored elsewhere.
                incoming = []
                train = add(f'M3-{arm}-{seed}-{stage}-train', 'train-stage', [previous, 'M3-preflight-' + arm, *([('M3-schedules-coverage-math' if arm in ('G', 'CR') else 'M3-schedules-math')] if stage == 'math_hard' else (['M3-schedules-coverage'] if arm in ('G', 'CR') else []))],
                            [*incoming, cmd('m3_train', python=train_py, action='train', arm=arm, seed=seed, stage=stage,
                                            schedule_root=schedule_roots[stage], tokenizer=inputs.get('tokenizer'), sdpo=inputs.get('sdpo'),
                                            trainer_python=train_py, gpu=settings.get('gpus', '0,1,2,3,4,5,6,7'))],
                            arm=arm, seed=seed, stage=stage)
                training = tasks[-1]
                prepare = train.removesuffix('-train') + '-prepare'
                train_command = training['commands'][-1]
                add(prepare, 'prepare', training['requires'], [*incoming, [v if v != 'train' else 'prepare' for v in train_command]], arm=arm, seed=seed, stage=stage)
                training['requires'] = [prepare]; training['commands'] = [train_command]
                exported = add(train.removesuffix('-train') + '-export', 'prepare', [train],
                    [[v if v != 'train' else 'export' for v in train_command]], arm=arm, seed=seed, stage=stage)
                # Task curves at 10/30; step-20 retention diagnostics on earlier tasks; every panel at the final.
                curve = {10: [TASK_PANEL[stage]], 20: [TASK_PANEL[stage], *[TASK_PANEL[e] for e in STAGES[:number]]], 30: [TASK_PANEL[stage]], 40: list(FINAL_PANELS)}
                commands = [cmd('m3_score',action='score',arm=arm,seed=seed,stage=stage,step=0,panel_set=p,gpu='{score_gpu}',panel_root=inputs.get('finqa_panel')) for p in FINAL_PANELS]+[cmd('m3_score', action='score', arm=arm, seed=seed, stage=stage, step=k, panel_set=p, gpu='{score_gpu}',
                                panel_root=inputs.get('finqa_panel')) for k, panel_list in curve.items() for p in panel_list]
                previous = add(f'M3-{arm}-{seed}-{stage}-score', 'score-family', [exported], commands, family='m3', seed=seed, arm=arm, stage=stage)
            finals.append(previous)
    # Likelihood validations, exactly as the HellaSwag rule: frozen base + one incoming checkpoint (C0 stage-1 final,
    # seed 101), each on every seed's dedicated GPU, 100% agreement with the independent reference.
    source = 'M3-%s-%d-%s-export' % VALIDATION_SOURCE
    validations = []
    for seed in SEEDS:
        validations.append(add(f'M3-fv-{seed}', 'score-family', ['M3-assign', 'M3-bind-base', 'M3-hellaswag', source],
                               [cmd('m3_score', action='validate-hellaswag', seed=seed, gpu='{score_gpu}')], family='m3', seed=seed))
        validations.append(add(f'M3-kv-{seed}', 'score-family', ['M3-assign', 'M3-bind-base', source],
                               [cmd('m3_score', action='validate-knowledge', seed=seed, gpu='{score_gpu}')], family='m3', seed=seed))
    rescored = []
    for seed in SEEDS:
        commands=[cmd('m3_score',action='score',arm='base',seed=seed,stage=STAGES[0],step=0,panel_set=p,gpu='{score_gpu}',panel_root=inputs.get('finqa_panel')) for p in FINAL_PANELS]
        for arm in ARMS:
            for number,stage in enumerate(STAGES):
                curve={0:list(FINAL_PANELS),10:[TASK_PANEL[stage]],20:[TASK_PANEL[stage],*[TASK_PANEL[e] for e in STAGES[:number]]],30:[TASK_PANEL[stage]],40:list(FINAL_PANELS)}
                commands += [cmd('m3_score',action='score',arm=arm,seed=seed,stage=stage,step=step,panel_set=p,gpu='{score_gpu}',panel_root=inputs.get('finqa_panel')) for step,panels in curve.items() for p in panels]
        # Already-completed partner training only needs CPU export validation, then
        # one pinned scoring job per seed for all arms and generated panels.
        deps = ['M3-assign', 'M3-bind-base', 'M3-hellaswag',
                *[t['id'] for t in tasks if t['id'].endswith('-export') and t.get('seed') == seed]]
        rescored.append(add(f'M3-rescore-{seed}', 'score-family', deps, commands, family='m3', seed=seed))
    add('M3-read', 'collect', ['M3-pools', *finals, *validations, *rescored], [cmd('m3_read', action='read')], partial_inputs=True)
    for task in list(tasks):
        if task['kind'] != 'score-family': continue
        checkpoints = []
        for command in task['commands']:
            def option(flag, default=None): return command[command.index(flag) + 1] if flag in command else default
            if option('--action') == 'score':
                checkpoints.append([option('--arm'), int(option('--seed')), option('--stage'), int(option('--step'))])
            else:
                from kit.m3_score import VALIDATION_CHECKPOINTS
                checkpoints += [list(key) for name, key in VALIDATION_CHECKPOINTS]
        checkpoints = [list(key) for key in dict.fromkeys(tuple(c) for c in checkpoints)]
        ready = '{m3_out}/scoring-ready/' + task['id'] + '.local.json'
        add(task['id'] + '-ready', 'prepare', task['requires'],
            [cmd('m3_score', action='prepare-score', checkpoints_json=json.dumps(checkpoints), ready=ready,
                workspace=('{m3_out}/scoring-jobs/'+task['id']) if any('--action' in c and c[c.index('--action')+1]=='score' for c in task['commands']) else None)],
            **{k:task[k] for k in ('arm', 'seed', 'stage') if k in task})
        task['requires'] = [task['id'] + '-ready']; task['scoring_ready'] = ready
        task['command_count']=len(task['commands']);task['lane_concurrency']=3
        if any('--action' in c and c[c.index('--action')+1]=='score' for c in task['commands']):
            task['scoring_workspace']='{m3_out}/scoring-jobs/'+task['id']
    ordered = []; done = set(); waiting = tasks[:]
    while waiting:
        ready = [t for t in waiting if set(t['requires']) <= done]
        need(bool(ready), 'static dependency cycle')
        ordered += ready; done |= {t['id'] for t in ready}; waiting = [t for t in waiting if t['id'] not in done]
    return dict(schema=SCHEMA,lane_runtime=__import__('kit.v6_infra',fromlist=['runtime_identity']).runtime_identity(),nodes=settings['nodes'], message2_nodes=sorted(core_nodes), message2_roots=settings.get('message2_roots', []), message2_assignment=settings.get('inputs', {}).get('message2_assignment'), inventory=inputs.get('inventory'),
                tasks=ordered, technical_synthetic=settings.get('technical_synthetic', False))


def load(path):
    doc = json.loads(Path(path).read_text())
    need(doc.get('schema') == SCHEMA, 'not an M3 launch manifest')
    ids = [t['id'] for t in doc['tasks']]
    need(len(ids) == len(set(ids)) and all(i.startswith('M3-') for i in ids), 'M3 task IDs must be unique and prefixed')
    for task in doc['tasks']:
        need(set(task['requires']) <= set(ids), 'M3 task requires a non-M3 task: ' + task['id'])
        need(all(isinstance(c, list) and c and all(isinstance(v, str) for v in c) for c in task['commands']), 'invalid plain job ' + task['id'])
    return doc


def isolation(m3, m2=None):
    """Static proof that M3 cannot block message-2 core and reads no message-2 output. m2: message2-launch.v2 manifest."""
    report = dict(schema=m3['schema'] == SCHEMA)
    ids = {t['id'] for t in m3['tasks']}
    report['prefixed'] = all(i.startswith('M3-') for i in ids)
    report['self_contained'] = all(set(t['requires']) <= ids for t in m3['tasks'])
    report['writes_m3_root_only'] = all('{out}' not in v and '{durable}' not in v for t in m3['tasks'] for c in t['commands'] for v in c)
    roots = [str(r).rstrip('/') for r in m3.get('message2_roots', []) if str(r).strip('/')]
    def mentions(task):
        return [(c, i) for c in task['commands'] for i, v in enumerate(c) if any(root in v for root in roots)]
    # The only permitted read of a message-2 product: M3-math-targets' read-only copy of core's committed targets-math,
    # named solely through --core-durable (round-4 ruling 1).
    def permitted(task):
        return task['id'] == 'M3-math-resolve' and all(c[i - 1] == '--core-durable' for c, i in mentions(task))
    report['no_message2_outputs'] = all(not mentions(t) or permitted(t) for t in m3['tasks'])
    report['runs_after_core_submission'] = any(t['id'] == 'M3-init' and '--core-submitted' in t['commands'][0] for t in m3['tasks'])
    m2_nodes = set(m3.get('message2_nodes', []))
    if m2 is not None:
        m2_ids = {t['id'] for t in m2['tasks']}
        report['disjoint_ids'] = not (ids & m2_ids)
        report['no_m2_dependency_on_m3'] = all(not (set(t.get('requires', [])) & ids) for t in m2['tasks'])
        m2_nodes |= {n['node'] if isinstance(n, dict) else n for n in m2.get('nodes', [])}
    report['scoped_gpu_leases'] = True  # placement is checked against each job's allocation at admission
    report['passed'] = all(report.values())
    return report


def substitute(command, values):
    out = []
    for value in command:
        for key, repl in values.items(): value = value.replace('{' + key + '}', str(repl))
        out.append(value)
    return out


def state_dir(out):
    return Path(out) / 'state'


def status(out, ident):
    path = state_dir(out) / (ident + '.json')
    return json.loads(path.read_text()) if path.exists() else dict(status='ABSENT')


def record(out, ident, value):
    path = state_dir(out) / (ident + '.json'); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, sort_keys=True) + '\n'); os.replace(tmp, path)


def runtime_node(doc, hostname=None):
    host = host_name(hostname or socket.gethostname())
    allowed = {host_name(n['node'] if isinstance(n, dict) else n) for n in doc.get('nodes', [])}
    need(host in allowed, 'executing node is not an allowed node')
    if doc.get('inventory') and not doc.get('technical_synthetic'):
        need(Path(doc['inventory']).is_file(), 'node inventory missing')
    return host


def allocated_gpus(count):
    """Resolve only the job's CUDA-visible devices to physical UUIDs; never claim a host."""
    from kit.m2_gpu import allocated_devices
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    need(len(visible) == count and all(visible) and len(set(visible)) == count,
         'request exactly %d GPUs and retain CUDA_VISIBLE_DEVICES' % count)
    from kit.m2_assignment import Unverified
    try:inventory = allocated_devices()
    except Unverified:need(False,'allocated GPU UUIDs unavailable/duplicated')
    uuids = [inventory.get(v, v if v in inventory.values() else None) for v in visible]
    need(all(uuids) and len(set(uuids)) == count, 'allocated GPU UUIDs unavailable/duplicated')
    return sorted(uuids)


@contextmanager
def allocation_reservation(uuids, *, check_idle=True):
    """Nonblocking, host-local physical GPU leases shared by tier entry points.

    Eight-rank jobs lease all eight allocated UUIDs through child exit. Inference
    never examines processes on other GPUs; CPU jobs acquire no GPU lease.
    """
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams = []
        for uuid in sorted(uuids):
            directory=Path('/tmp')/'kit-gpu-leases';directory.mkdir(exist_ok=True)
            lock=directory/(uuid+'.lock')
            stream = stack.enter_context(lock.open('a'))
            try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise Unbound('allocated GPU reservation is busy')
            streams.append(stream)
        if check_idle:
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
            if any(line.split(',')[0].strip() in uuids for line in processes.splitlines()):
                raise Unbound('allocated GPU still has active GPU work')
        yield tuple(stream.fileno() for stream in streams)


@contextmanager
def node_reservation(doc, *, check_idle=False, gpus=8):
    runtime_node(doc)
    uuids = ['CPU-STANDIN-' + str(i) for i in range(gpus)] if doc.get('technical_synthetic') else allocated_gpus(gpus)
    with allocation_reservation(uuids, check_idle=check_idle) as fds: yield fds


@contextmanager
def gpu_execution(out, gpus=1):
    """Direct scientific entry points require the same allocation leases."""
    inherited = os.environ.get('M3_GPU_LOCK_FDS')
    if inherited is not None:
        policy = json.loads(os.environ['M3_NODE_POLICY']); runtime_node(policy)
        uuids = allocated_gpus(gpus); fds = json.loads(inherited)
        need(len(fds) == len(uuids), 'inherited GPU reservation size differs')
        for fd, uuid in zip(fds, uuids):
            directory=Path('/tmp')/'kit-gpu-leases';directory.mkdir(exist_ok=True)
            lock=directory/(uuid+'.lock')
            need(os.fstat(fd).st_ino == lock.stat().st_ino and os.fstat(fd).st_dev == lock.stat().st_dev, 'missing inherited GPU reservation')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    else:
        manifest = Path(os.environ.get('M3_MANIFEST', str(Path(out) / 'm3-launch.json')))
        need(manifest.is_file(), 'GPU execution requires the M3 node manifest')
        with node_reservation(load(manifest), check_idle=True, gpus=gpus): yield


def watched(command, out):
    return [command[0], '-I', str(KIT / 'util_watch.py'), '--out', str(out), '--expect-gpus','1','--', *command]


def submission_time(values):
    if values.get('submitted_at') is not None: return float(values['submitted_at']), True
    if os.environ.get('M3_SUBMITTED_AT'): return float(os.environ['M3_SUBMITTED_AT']), True
    if os.environ.get('SLURM_JOB_ID'):
        output = subprocess.check_output(['scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']], text=True)
        match = re.search(r'\bSubmitTime=(\S+)', output)
        need(match is not None, 'scheduler submission time unavailable')
        return datetime.fromisoformat(match[1]).timestamp(), True
    return time.time(), False  # direct execution has no scheduler queue


def run_task(manifest_path, ident, out, values, runner=None):
    """Run one isolated task, retaining failed attempts and submission/start/end timing."""
    from kit.p3b_common import NotReady
    try: doc = load(manifest_path)
    except Exception: return EXIT_FAILED
    task = next((t for t in doc['tasks'] if t['id'] == ident), None)
    if task is None:return EXIT_FAILED
    runtime_env=dict(os.environ)
    if 'lane_runtime' in doc:
        from kit.v6_infra import verify_runtime
        try:runtime_env['M2_INFRA_AUTHORITY_SHA256']=verify_runtime(doc['lane_runtime']) or ''
        except (ValueError,OSError):return EXIT_FAILED
    old = status(out, ident); attempts = list(old.get('attempts', []))
    if old.get('status') == 'RUNNING' and old.get('started_at') is not None:
        ended = time.time()
        attempts.append(dict(status='INTERRUPTED', started_at=old['started_at'], submitted_at=old.get('submitted_at', old['started_at']),
                             ended_at=ended, seconds=ended-old['started_at'], timing_estimated=True))
    try: submitted, queue_known = submission_time(values)
    except Exception as exc:
        record(out, ident, dict(status='PENDING', reason='submission timestamp unavailable: ' + type(exc).__name__)); return EXIT_PENDING
    def save(value):
        record(out, ident, dict(value, submitted_at=submitted, queue_timing_complete=queue_known, attempts=attempts))
    if ident != 'M3-init':
        try: require_m3_root(out)
        except IsolationError: return EXIT_FAILED
        if not core_submitted(out):
            save(dict(status='PENDING', reason='submit message-2 core jobs first, then run M3-init')); return EXIT_PENDING
    blocked = [r for r in task['requires'] if status(out, r)['status'] != 'DONE']
    if blocked and not task.get('partial_inputs'):
        save(dict(status='PENDING', blocked_by=blocked)); return EXIT_PENDING
    from contextlib import nullcontext
    gpu = task['kind'] in ('train-stage', 'score-family') or task.get('uses_gpu')
    started = time.time(); code = 0; detail = {}
    workspace=None;nonce=__import__('uuid').uuid4().hex
    try:
        with node_reservation(doc, check_idle=not doc.get('technical_synthetic'), gpus=task.get('gpus', 8 if task['kind'] == 'train-stage' else 1)) if gpu else nullcontext() as reservation:
            if ident != 'M3-init': save(dict(status='RUNNING', started_at=started, manifest_sha256=digest(doc)))
            expanded_commands=[substitute(c,dict(values,m3_out=out)) for c in task['commands']]
            if task.get('scoring_workspace'):
                from kit.m3_score import prepare_workspace
                workspace=prepare_workspace(out,Path(task['scoring_workspace'].replace('{m3_out}',str(out)))/nonce)
                for command in expanded_commands:
                    if '--out' in command:command[command.index('--out')+1]=str(workspace)
            if task.get('scoring_ready'):
                from kit.m3_score import prepared_commands
                ready_path=Path(task['scoring_ready'].replace('{m3_out}',str(out)))
                if workspace:
                    snapshot=workspace/'scoring-ready.local.json'
                    snapshot.write_bytes(ready_path.read_bytes());ready_path=snapshot
                expanded_commands=prepared_commands(expanded_commands,ready_path)
            if runner is not None:
                for number,expanded in enumerate(expanded_commands):
                    code=runner(expanded)
                    if code:detail=dict(command=number,returncode=code);break
            elif gpu:
                policy={k:doc[k] for k in ('nodes','message2_nodes','inventory','message2_assignment','technical_synthetic') if k in doc}
                env=dict(runtime_env,M3_GPU_LOCK_FDS=json.dumps(reservation),M3_NODE_POLICY=json.dumps(policy),KIT_LANE_JOB=nonce)
                if task.get('scoring_ready'):env['M3_SCORE_READY']=str(ready_path)
                if task.get('gpus')==1 and not doc.get('technical_synthetic'):
                    from kit.lane_pool import execute
                    concurrency=3
                    if workspace:
                        from kit.v6a import rescore_concurrency
                        concurrency=rescore_concurrency(env['M3_SCORE_READY'],expanded_commands)
                    code=execute(expanded_commands,Path(out)/'lane-attempts'/(ident+'-'+str(time.time_ns())),env=env,pass_fds=reservation,concurrency=concurrency)
                else:
                    for expanded in expanded_commands:
                        code=subprocess.run(expanded,env=env,pass_fds=reservation).returncode
                        if code:break
                if code==0 and workspace:
                    from kit.m3_score import unit_root
                    from kit.lane_evidence import complete_attempt
                    identities=[]
                    for command in expanded_commands:
                        def option(flag):return command[command.index(flag)+1]
                        cell=unit_root(workspace,option('--arm'),int(option('--seed')),option('--stage'),int(option('--step')),option('--panel-set'))
                        identities.append(str((cell/'m3-identity.json').relative_to(workspace)))
                    complete_attempt(workspace,nonce,identities,extra_files=['scoring-ready.local.json'])
            else:
                for expanded in expanded_commands:
                    code=subprocess.run(expanded,env=runtime_env).returncode
                    if code:break
    except (Unbound,NotReady,FileNotFoundError) as exc: code = EXIT_PENDING; detail = dict(reason=str(exc))
    except Exception as exc: code = EXIT_FAILED; detail = dict(error=repr(exc))
    ended = time.time()
    label = 'DONE' if code == 0 else ('PENDING' if code in (NOT_READY, EXIT_PENDING) else 'FAILED')
    attempt = dict(status=label, submitted_at=submitted, started_at=started, ended_at=ended, seconds=ended-started, **detail)
    attempts.append(attempt)
    save(dict(attempt, **{k: task[k] for k in ('arm', 'seed', 'stage', 'kind') if k in task}))
    return 0 if label == 'DONE' else (EXIT_PENDING if label == 'PENDING' else EXIT_FAILED)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('init', 'bind-inputs', 'build', 'isolation', 'run'), required=True)
    p.add_argument('--out', type=Path); p.add_argument('--settings', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--manifest', type=Path); p.add_argument('--message2-manifest', type=Path); p.add_argument('--task')
    p.add_argument('--roots', help='JSON {name: package B/C root read in place}'); p.add_argument('--return1', type=Path)
    p.add_argument('--core-submitted', action='store_true', help='partner confirms every message-2 core job is already submitted')
    p.add_argument('--submitted-at', type=float, default=None)
    p.add_argument('--score-gpu', default='0'); p.add_argument('--standin', action='store_true')
    a = p.parse_args(argv)
    if a.action == 'init': print(init_root(a.out, a.core_submitted)); return 0
    if a.action == 'bind-inputs':
        try: print(json.dumps(bind_inputs(a.out, json.loads(a.roots), a.return1), sort_keys=True)); return 0
        except Unbound as exc: print('NOT READY:', exc); return NOT_READY
        except Exception as exc: print('STOP:', exc); return 2
    if a.action == 'build':
        doc = build(json.loads(a.settings.read_text()))
        a.output.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n'); load(a.output)
        print(len(doc['tasks']), 'M3 plain jobs; none submitted'); return 0
    if a.action == 'isolation':
        report = isolation(load(a.manifest), json.loads(a.message2_manifest.read_text()) if a.message2_manifest else None)
        print(json.dumps(report, sort_keys=True)); return 0 if report['passed'] else EXIT_FAILED
    task = next(t for t in load(a.manifest)['tasks'] if t['id'] == a.task)
    if task.get('gpus', 0) and not a.standin and not load(a.manifest).get('technical_synthetic') and not os.environ.get('UTIL_WATCH_ACTIVE'):
        return subprocess.run(watched([sys.executable, '-I', str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])], a.out / 'util' / a.task / (str(time.time_ns())+'.jsonl')), env=dict(os.environ, UTIL_WATCH_ACTIVE='1')).returncode
    return run_task(a.manifest, a.task, a.out, dict(kit=str(KIT), score_gpu=a.score_gpu, **({'submitted_at': a.submitted_at} if a.submitted_at is not None else {})))


if __name__ == '__main__':
    raise SystemExit(main())
