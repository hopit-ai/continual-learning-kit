#!/usr/bin/env python3
"""Offline operator bindings and evidence-backed admission/qualification receipts.

No permission, correctness disposition or measurement is invented by this writer.
Node native probes are executable, hash-bound installed-stack assets.
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json
import re
import shlex
import math
import os
import subprocess
from kit.trace_common import *

RESOURCE = KIT/'resources/trace/operator'
GENERAL = ('MMLU', 'BBH', 'GSM8K', 'TyDiQA', 'PIQA', 'BoolQ')
ADMISSION = ('original_tokenizer_census', 'supervised_masks', 'prompt_fixtures',
             'scorer_fixtures', 'demo_fidelity_gold', 'demo_fidelity_27b')
NATIVE = ('noise_floor', 'world_size_reduction', 'ema_recurrence', 'native_resume',
          'export_identity', 'freeze_bit_equality', 'likelihood', 'scorer_concurrency')


def reference(path):
    path = Path(path).resolve()
    need(path.is_file(), 'node file not staged')
    return dict(path=str(path), sha256=sha(path))


def references(value):
    """Complete shorthand file references; verify every supplied hash."""
    if isinstance(value, list):
        return [references(v) for v in value]
    if isinstance(value, dict):
        result = {k: references(v) for k, v in value.items()}
        if 'path' in result:
            ref = reference(result['path'])
            need('sha256' not in result or result['sha256'] == ref['sha256'],
                 'supplied input hash differs')
            result.update(ref)
        return result
    return value


def finite(value):
    need(type(value) in (float, int) and math.isfinite(value), 'finite measurement required')
    return value


def zero(value):
    need(finite(value) == 0, 'zero-error admission bound exceeded')


def validate_admission(name, value):
    need(value.get('technical_synthetic') is False, 'production admission evidence required')
    need(value.get('registration_sha256') == sha(REG), 'admission registration differs')
    need(value.get('rows', 0) > 0, 'empty admission evidence')
    if name == 'original_tokenizer_census':
        need(value.get('same_rows') is True, 'paired same-row census required')
        need(value.get('original_revision'), 'original tokenizer revision required')
        need(value.get('by_task_split'), 'task/split census required')
        for task, splits in value['by_task_split'].items():
            need(task in TASKS and set(splits) == {'train', 'eval', 'test'},
                 'length census task/splits differ')
            for row in splits.values():
                need(row['rows'] > 0, 'empty length census')
                need(finite(row['train_prompt_excess']) <= .05 and
                     finite(row['inference_prompt_excess']) <= .05 and
                     finite(row['target_loss_excess']) <= .01, 'relative length bound exceeded')
                zero(row['wrong_supervised_tokens'])
                need(0 <= finite(row['max_eos_position_shift']) <= 1, 'EOS shift bound exceeded')
                # EOS affected share is diagnostic and deliberately has no bound here.
    elif name == 'supervised_masks':
        zero(value['wrong_supervised_tokens'])
        need(0 <= finite(value['max_eos_position_shift']) <= 1, 'EOS shift bound exceeded')
    elif name == 'prompt_fixtures':
        need(value.get('cases_per_task') and
             all(n >= 20 for n in value['cases_per_task'].values()), 'twenty prompt cases/task required')
        zero(value['semantic_changes'])
        zero(value['leaked_targets'])
        zero(value['missing_roles_or_eos'])
    elif name == 'scorer_fixtures':
        need(value.get('cases_per_metric') and
             all(n >= 20 for n in value['cases_per_metric'].values()), 'twenty scorer cases required')
        zero(value['label_or_unit_errors'])
        need(0 <= finite(value['max_error']) <= 1e-6, 'scorer tolerance exceeded')
    else:
        need(value.get('training_only') is True and value.get('summaries') == 50,
             'fixed fifty training summaries required')
        need(len(set(value.get('summary_ids', []))) == 50, 'summary audit IDs incomplete')
        zero(value['material_mismatches'])
        zero(value['invalid_targets'])
        zero(value['leaked_targets'])
    return True


def admission_context(cfg):
    return digest({key: cfg[key] for key in ('base', 'original_tokenizer', 'datasets', 'panels')})


def fidelity_ids(prepared):
    rows = prepared['selected']['MeetingBank']
    need(len(rows) >= 50, 'fifty training summaries required')
    return [r['id'] for r in sorted(rows, key=lambda r: digest(
        ['trace-fidelity-20261112', r['id']]))[:50]]


def admission(evidence, studies, defer_27b=False, cfg=None):
    required = list(ADMISSION[:4])
    if set(studies) & set(SOURCES):
        required += list(ADMISSION[4:])
    receipt = {}
    for name in required:
        if name == 'demo_fidelity_27b' and defer_27b:
            continue
        ref = evidence[name]
        if set(ref) == {'deferred_path'}:
            ref = reference(ref['deferred_path'])
        value = bound_json(ref)
        validate_admission(name, value)
        if cfg is not None:
            need(value.get('input_sha256') == admission_context(cfg),
                 'admission model/data/panel identity differs')
        tasks = set(TASKS if 's8' in studies else SUBSET)
        if name == 'original_tokenizer_census':
            need(set(value['by_task_split']) == tasks, 'length census task coverage differs')
            if cfg is not None:
                need(value['original_revision'] == cfg['original_tokenizer']['revision'],
                     'original tokenizer census revision differs')
        if name == 'prompt_fixtures':
            need(set(value['cases_per_task']) == tasks, 'prompt fixture task coverage differs')
        if name == 'scorer_fixtures':
            metrics = tasks | {'instruction', 'maths'} | (set(GENERAL) if 's8' in studies else set())
            need(set(value['cases_per_metric']) == metrics, 'scorer fixture metric coverage differs')
        if cfg is not None and name.startswith('demo_fidelity'):
            prepared_path = Path(cfg['out'])/'prepared.json'
            if prepared_path.exists():
                expected = fidelity_ids(load(prepared_path))
                need(value['summary_ids'] == expected, 'fixed training summary inventory differs')
            if name == 'demo_fidelity_27b':
                target_path = Path(cfg['out'])/'targets.json'
                need(target_path.is_file() and value.get('targets_sha256') == sha(target_path),
                     'generated demonstration identity differs')
        receipt[name] = dict(valid=True, evidence_sha256=ref['sha256'])
    return receipt


def validate_native(values):
    need(set(values) == set(NATIVE), 'native numerical probe inventory differs')
    for value in values.values():
        need(value.get('technical_synthetic') is False and value.get('world_size') == 8,
             'real eight-rank installed-stack probes required')
        need(value.get('registration_sha256') == sha(REG), 'native registration differs')
        need(value.get('model_sha256') and value.get('environment_sha256'),
             'native model/environment identity absent')
    need(len({v['model_sha256'] for v in values.values()}) == 1 and
         len({v['environment_sha256'] for v in values.values()}) == 1,
         'native probes used different models/environments')
    noise = values['noise_floor']
    tolerance = noise['fixed_tolerance']
    for field in ('student', 'ema'):
        floor = finite(noise[field+'_noise'])
        need(floor >= 0 and 0 <= finite(tolerance[field]) <= 3*floor,
             'numerical tolerance exceeds three times D/D noise')
    need(noise.get('fixed_before_freeze') is True, 'noise bound must precede freeze probe')
    need(values['world_size_reduction']['world_sizes'] == [1, 2, 8], 'reduction worlds differ')
    need(0 <= finite(values['world_size_reduction']['max_error']) <= 1e-4,
         'reduction tolerance exceeded')
    for name in ('ema_recurrence', 'native_resume'):
        for field in ('student', 'ema'):
            need(0 <= finite(values[name][field+'_error']) <= tolerance[field],
                 name+' exceeds fixed noise tolerance')
    need(values['native_resume'].get('optimizer_equal') is True and
         values['native_resume'].get('rng_equal') is True and
         values['native_resume'].get('mask_equal') is True, 'native resume incomplete')
    need(values['freeze_bit_equality'].get('blocks') == [0, 1, 2], 'freeze boundary differs')
    zero(values['freeze_bit_equality']['changed_coordinates'])
    zero(values['freeze_bit_equality']['optimizer_coordinates'])
    zero(values['export_identity']['mismatches'])
    need(0 <= finite(values['likelihood']['max_error']) <= 1e-6, 'likelihood tolerance exceeded')
    zero(values['scorer_concurrency']['mismatches'])
    return True


def qualify(cfg, study, destination):
    """Execute installed native probes; retain all raw evidence beside the receipt."""
    offline()
    from kit.trace_score import metric_adapter
    for ref in cfg['metrics'].values():
        metric_adapter(ref)
    prepared = load(Path(cfg['out'])/'prepared.json')
    need(prepared['registration_sha256'] == sha(REG) and prepared['audit']['unresolved'] == 0,
         'prepared identity/leakage evidence unresolved')
    need(study in prepared['active_studies'], 'study PENDING')
    need(prepared['config_sha256'] == digest(cfg),
         'prepared configuration differs')
    for role in ('base', 'judge', 'original_tokenizer', *(['teacher'] if study in SOURCES else [])):
        model_record(cfg[role])
    binding = load(Path(cfg['out'])/'binding.json')
    need(binding['config_sha256'] == digest(cfg), 'CPU binding differs')
    probes = cfg['operator']['native_probes'][study]
    need(set(probes) == set(NATIVE), 'native command inventory differs')
    values, evidence = {}, {}
    destination = Path(destination)
    for name in NATIVE:
        manifest = bound_json(probes[name])
        need(isinstance(manifest['argv'], list) and manifest['argv'] and manifest['files'],
             'native command/files missing')
        for ref in manifest['files']:
            bound_file(ref)
        target = destination.parent/(name+'.json')
        need(not target.exists(), 'fresh native evidence destination required')
        subprocess.run(manifest['argv'], check=True, env=dict(os.environ,
            TRACE_CONFIG=cfg['_config_path'], TRACE_STUDY=study,
            TRACE_PROBE_OUT=str(target), TRACE_NOISE_OUT=str(destination.parent/'noise_floor.json')))
        values[name] = load(target)
        evidence[name] = reference(target)
    validate_native(values)
    need(all(v['model_sha256'] == cfg['base']['sha256'] for v in values.values()),
         'native probes used wrong base model')
    detail = dict(probes=evidence, config_sha256=digest(cfg), registration_sha256=sha(REG),
                  binding_sha256=sha(Path(cfg['out'])/'binding.json'))
    seal(destination.parent/'evidence.json', detail)
    result = dict(identity=True, leakage=True, freeze_bit_equality=True,
                  scorer_fixtures=True, licences=True,
                  evidence_sha256=sha(destination.parent/'evidence.json'))
    seal(destination, result)
    return result


def stage_assets(config):
    """Hash staged inputs before collecting CPU observations against their identity."""
    cfg = references(copy.deepcopy(config))
    need(cfg.get('technical_synthetic') is False, 'production binding requires explicit mode')
    studies = cfg.get('studies', ['s8', *SOURCES])
    need(studies and set(studies) <= {'s8', *SOURCES}, 'study scope differs')
    roles = ['base', 'judge', 'original_tokenizer']
    if set(studies) & set(SOURCES):
        roles.append('teacher')
    for role in roles:
        record = cfg[role]
        need(isinstance(record.get('revision'), str) and record['revision'],
             'staged model/tokenizer revision required')
        directory = Path(record['model']).resolve()
        files = {p.relative_to(directory).as_posix(): sha(p)
                 for p in directory.rglob('*') if p.is_file()}
        need(files, 'model/tokenizer assets absent')
        need(not record.get('files') or record['files'] == files, 'model files differ')
        record.update(model=str(directory), files=files)
        model_files = {k:v for k,v in files.items() if k.endswith('.safetensors') or
                       k.endswith('.safetensors.index.json') or k == 'config.json'}
        identity = digest(model_files if role == 'base' else files)
        need(not record.get('sha256') or record['sha256'] == identity, 'model digest differs')
        record['sha256'] = identity
    return cfg


def bind(config, destination):
    """Bind local assets and emit all operator-owned adapter/command manifests."""
    cfg = stage_assets(config)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    studies = cfg.get('studies', ['s8', *SOURCES])
    needed = set(TASKS if 's8' in studies else SUBSET)
    metric_names = sorted(needed | {'instruction', 'maths'} | (set(GENERAL) if 's8' in studies else set()))
    vendor = KIT/'resources/trace/vendor/hf-datasets-2.21.0-sari'
    metric_files = [reference(KIT/'trace_datasets.py'),
                    *(reference(vendor/name) for name in ('sari.py', 'LICENSE', 'PROVENANCE.json'))]
    cfg['metrics'] = {}
    for name in metric_names:
        fixtures = load(RESOURCE/'fixtures'/f'{name}.json')
        if name not in ('instruction', 'maths'):
            authority = cfg['operator']['metric_authorities'][name]
            for fixture in fixtures:
                fixture['row']['metric_authority'] = authority
        path = root/'fixtures'/f'{name}.json'
        seal(path, fixtures)
        cfg['metrics'][name] = dict(**reference(KIT/'trace_metrics.py'),
            function='general' if name in ('instruction', 'maths') else 'score',
            fixtures=reference(path), dependencies=metric_files)
        if name not in ('instruction', 'maths'):
            cfg['metrics'][name]['authority'] = authority
    from kit.trace_score import metric_adapter
    for ref in cfg['metrics'].values():
        metric_adapter(ref)
    # Generated 27B demos cannot be audited until targets completes. Pin the
    # audit plan now; its resulting receipt is sealed by the CPU bind row.
    admission(cfg['operator']['admission_evidence'], studies, defer_27b=True, cfg=cfg)
    path = root/'admission-plan.json'
    seal(path, dict(schema='trace-admission-plan.v1',
                    evidence=cfg['operator']['admission_evidence']))
    cfg['admission'] = reference(path)
    cfg['qualification'] = {}
    for study in studies:
        path = root/'qualification'/f'{study}.json'
        probes = cfg['operator']['native_probes'][study]
        need(set(probes) == set(NATIVE), 'native probe assets incomplete')
        files = [reference(KIT/'trace_operator.py'), reference(KIT/'trace_metrics.py'),
                 *metric_files]
        for probe in probes.values():
            command = bound_json(probe)
            need(command.get('argv') and command.get('files'), 'native command assets incomplete')
            for item in command['files']:
                bound_file(item)
            files += [probe, *command['files']]
        manifest = load(RESOURCE/'qualification'/f'{study}.json')
        manifest['argv'] = [cfg.get('trainer_python', sys.executable), '-I',
                            str(KIT/'trace_operator.py'), 'qualify']
        manifest['files'] = files
        seal(path, manifest)
        cfg['qualification'][study] = reference(path)
    seal(root/'config.json', cfg)
    return cfg


def dependencies(row):
    edges=[(d,'afterok') if isinstance(d,str) else (d['id'],d.get('type','afterok'))
           for d in row.get('depends',[])]
    edges.extend((d['id'],d.get('type','afterok')) for d in row.get('depends_on',[]))
    return list(dict.fromkeys(edges))


def plan(inventories,nodes):
    """Estimate equal-duration capacity waves without adding scheduler edges."""
    if nodes not in (1,2,4,8):
        raise ValueError('nodes must be 1, 2, 4 or 8')
    rows={}
    for scope,value in inventories.items():
        if scope not in ('trace','deletion','a2'):
            raise ValueError('unsupported inventory scope')
        for row in value['rows']:
            if not isinstance(row['id'],str) or not row['id'] or re.search(r'[\x00-\x1f\x7f]',row['id']):
                raise ValueError('invalid row ID')
            key=scope+':'+row['id']
            if key in rows:
                raise ValueError('duplicate inventory ID')
            if type(row['gpus']) is not int or not 0<=row['gpus']<=8:
                raise ValueError('gpus must be an integer from 0 (CPU only) to 8')
            cpus=row.get('cpus',32 if row['gpus']==8 else 8)
            if type(cpus) is not int or cpus<=0:
                raise ValueError('cpus must be a positive integer')
            if 'mem_mb' in row and (type(row['mem_mb']) is not int or row['mem_mb']<=0):
                raise ValueError('mem_mb must be a positive integer')
            if 'time' in row:
                validate_time(row['time'])
            rows[key]=dict(row,scope=scope,cpus=cpus,order=len(rows),planned_nodes=nodes,
                           deps=[(scope+':'+d,t) for d,t in dependencies(row)])
    if any(d not in rows for row in rows.values() for d,_ in row['deps']):
        raise ValueError('missing dependency; complete inventory required')
    waiting=dict(rows)
    done=set()
    waves=[]
    while waiting:
        capacity=[8]*nodes
        wave=[]
        for key,row in list(waiting.items()):
            if not all(d in done for d,_ in row['deps']):
                continue
            node=next((i for i,free in enumerate(capacity) if free>=row['gpus']),None)
            if node is None:
                continue
            capacity[node]-=row['gpus']
            wave.append((key,dict(row,node=node)))
            del waiting[key]
        if not wave:
            raise ValueError('cyclic dependency inventory')
        done.update(key for key,_ in wave)
        waves.append(wave)
    return waves


def ordered_rows(waves):
    """Topologically order inventory rows independently of capacity estimates."""
    rows=dict(sorted((item for wave in waves for item in wave),key=lambda item:item[1]['order']))
    ordered=[]
    done=set()
    while len(ordered)<len(rows):
        ready=[(key,row) for key,row in rows.items() if key not in done and
               all(d in done for d,_ in row['deps'])]
        if not ready:
            raise ValueError('cyclic dependency inventory')
        ordered.extend(ready)
        done.update(key for key,_ in ready)
    return ordered


def validate_time(value):
    if not isinstance(value,str) or not re.fullmatch(
            r'(?:[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}|[0-9]+:[0-9]{2}:[0-9]{2}|[0-9]+:[0-9]{2}|[0-9]+)',value):
        raise ValueError('unsupported time syntax')
    parts=value.replace('-',':').split(':')
    if (len(parts)>1 and int(parts[-1])>=60) or (len(parts)>2 and int(parts[-2])>=60):
        raise ValueError('invalid time minutes or seconds')
    if '-' in value and int(parts[1])>=24:
        raise ValueError('invalid time hours within day')


def dependency_policy(options):
    """Only cancellation policy may supplement the generated sbatch arguments."""
    if not isinstance(options,list) or any(option!='--kill-on-invalid-dep=yes' for option in options):
        raise ValueError('unsupported dependency policy option; require --kill-on-invalid-dep=yes')
    # All inventories need explicit cancellation, independent of site defaults.
    # Identical duplicates collapse; conflicting policies are rejected above.
    return ['--kill-on-invalid-dep=yes']


def commands(waves):
    rows=ordered_rows(waves)
    variables={key:f'j{i:04}' for i,(key,_) in enumerate(rows)}
    lengths=dict.fromkeys(variables,1)
    for key,row in reversed(rows):
        for dep,_ in row['deps']:
            lengths[dep]=max(lengths[dep],lengths[key]+1)
    lines=['# Plan only. Slurm packs queued jobs onto available nodes.',
           '# Rolling-window utilisation is not guaranteed.',
           '# Ordering hint only. Prefer long chains first using the remaining lengths below.']
    if rows:
        lines.append(f'# Estimate for {rows[0][1]["planned_nodes"]} nodes of 8 GPUs, '
                     f'{len(waves)} waves assuming equal row durations.')
    for key,row in rows:
        groups={}
        for dep,kind in row['deps']:
            if not isinstance(kind,str) or not re.fullmatch('[a-z]+',kind):
                raise ValueError('invalid dependency type')
            groups.setdefault(kind,[]).append('$'+variables[dep])
        dependency=','.join(kind+':'+':'.join(values) for kind,values in groups.items())
        args=['sbatch','--parsable','--nodes=1',f'--gpus-per-node={row["gpus"]}']
        if dependency:
            args.append('--dependency='+dependency)
        args.extend(dependency_policy(row.get('dependency_options',[])))
        args.append('--cpus-per-task='+str(row['cpus']))
        if 'mem_mb' in row:
            args.append('--mem='+str(row['mem_mb'])+'M')
        if 'time' in row:
            args.append('--time='+row['time'])
        scope=row['scope']
        # Use portable variable-based wrappers, without disclosing input paths.
        if scope=='trace':
            command='bash "$KIT/jobs/trace/run-task.sh" '+shlex.quote(row['id'])
        elif scope=='deletion':
            command='bash "$DELETION_ROOT/deletion_tuning/jobs/run-task.sh" '+shlex.quote(row['id'])
        else:
            command='bash "$KIT/jobs/a2/run-task.sh" '+shlex.quote(row['id'])
        args.append('--wrap='+command)
        # Dependency references expand inside double quotes. All their names and
        # edge types are generated or validated above. Other arguments are scalars.
        quoted=['"'+arg+'"' if arg.startswith('--dependency=') else shlex.quote(arg) for arg in args]
        lines.append(f'# {scope} {row["id"]}, remaining chain {lengths[key]} rows')
        lines.append(variables[key]+'=$('+' '.join(quoted)+')')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('assets', 'bind', 'admission', 'qualify', 'dag-plan'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--nodes','--node-count',type=int,choices=(1,2,4,8))
    for scope in ('trace','deletion','a2'):
        parser.add_argument('--'+scope,type=Path,help='saved full inventory JSON for print-only DAG plan')
    args = parser.parse_args()
    offline()
    if args.action=='dag-plan':
        inventories={scope:load(getattr(args,scope)) for scope in ('trace','deletion','a2')
                     if getattr(args,scope)}
        need(args.nodes and inventories,'nodes and at least one complete inventory required')
        print(commands(plan(inventories,args.nodes)),end='')
        return
    if args.action == 'qualify':
        config_path = Path(os.environ['TRACE_CONFIG']).resolve()
        cfg = dict(load(config_path), _config_path=str(config_path))
        qualify(cfg, os.environ['TRACE_STUDY'], Path(os.environ['TRACE_QUALIFICATION_OUT']))
    else:
        need(args.config and args.out, 'config and out required')
        config = load(args.config)
        if args.action == 'assets':
            cfg = stage_assets(config)
            seal(args.out/'assets.json', cfg)
            seal(args.out/'admission-input.json', dict(input_sha256=admission_context(cfg),
                                                      registration_sha256=sha(REG)))
        elif args.action == 'bind':
            bind(config, args.out)
        else:
            seal(args.out, admission(references(config['operator']['admission_evidence']),
                                     config.get('studies', ['s8', *SOURCES]), cfg=config))


if __name__ == '__main__':
    main()
