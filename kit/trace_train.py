#!/usr/bin/env python3
"""Plain allocation entry for one TRACE seed/study. Lazy GPU imports only."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import subprocess
from kit.trace_common import *


# Capture the inherited converter once. Only the TRACE call site installs this adapter.
from kit.nm_prepare import converted_rows as nm_converted_rows


def converted_trace_rows(*args, **kwargs):
    rows, audit = nm_converted_rows(*args, **kwargs)
    for row in rows:
        row['data_source'] = 'trace'
    return rows, audit


def d_stage(cfg, stage, seed, root, incoming):
    from kit.trace_prompt import install
    install()
    from kit.nm_prepare import training_command, export_stage, verify_model
    from kit.nm_model import metadata
    model = verify_model(cfg['base'])
    base_binding = root/'base-binding.json'
    seal(base_binding, dict(base=model))
    bundle = dict(base=model, tokenizer=cfg['base']['model'], sdpo=cfg['sdpo'],
                  trainer_python=cfg.get('trainer_python', sys.executable),
                  base_binding=str(base_binding), base_metadata=metadata(model['model'], model['files']))
    binding = dict(recipe=dict(name='D', starts={str(seed):incoming},
                   caps=dict(prompt=2048, teacher=6144, response=2048), balance_batch=False))
    rows = load(Path(cfg['out'])/'schedules'/f'{cfg["source"]}-{seed}-{stage["task"]}.json')
    rows = [dict(row, domain='trace_'+stage['task']) for row in rows]
    prepared_path = root/'prepared.json'
    if prepared_path.exists():
        prepared = load(prepared_path)
        need(prepared['incoming']['sha256'] == incoming['sha256'], 'resume ancestry changed')
    else:
        from unittest.mock import patch
        from kit import nm_prepare
        with patch.object(nm_prepare, 'converted_rows', converted_trace_rows):
            prepared = training_command(bundle, binding, rows, root, 'freeze',
                                        'T' if stage['freeze'] else 'C', seed)
    # Isolate the TRACE reward adapter without altering inherited callers.
    command = [str(KIT/'trace_entry.py') if arg == str(KIT/'nm_entry.py') else arg
               for arg in prepared['command']]
    command = [arg.replace('custom_reward_function.path='+str(KIT/'nm_entry.py'),
                           'custom_reward_function.path='+str(KIT/'trace_entry.py')) for arg in command]
    if (root/'native-receipt.json').exists():
        need(load(root/'native-receipt.json')['status']=='COMPLETE', 'native receipt incomplete')
        return export_stage(prepared, root, cfg.get('trainer_python', sys.executable))['40']
    checkpoints = sorted((root/'native').glob('global_step_*'),
                         key=lambda p:int(p.name.rsplit('_',1)[1]))
    if checkpoints:
        # Restore every native component, including EMA/RNG and mask identity.
        latest = checkpoints[-1]
        need((latest/'actor'/'nm-state.json').is_file(), 'incomplete native snapshot')
        command = [arg for arg in command if arg.split('=',1)[0].lstrip('+')
                   not in ('trainer.resume_mode','trainer.resume_from_path')]
        command += ['++trainer.resume_mode=resume_path',
                    '++trainer.resume_from_path='+str(latest)]
    subprocess.run(command, env=dict(os.environ, **prepared['env']),
                   cwd=prepared['cwd'], check=True)
    return export_stage(prepared, root, cfg.get('trainer_python', sys.executable))['40']


def complete_record(record):
    root = Path(record['model'])
    return dict(record, files={str(p.relative_to(root)):sha(p)
                              for p in root.rglob('*') if p.is_file()})


def subset(cfg, source, seed):
    root = Path(cfg['out'])/'training'/f'{source}-{seed}'
    root.mkdir(parents=True, exist_ok=True)
    exports = {'base':cfg['base']}
    for stage in stages(source, seed):
        destination = root/stage['id']
        completed = destination/'trace-complete.json'
        if completed.exists():
            receipt = load(completed)
            need(receipt['config_sha256'] == digest(cfg), 'completed stage config changed')
            from kit.nm_prepare import verify_model
            exports[stage['id']] = verify_model(receipt['export'])
            continue
        export = d_stage(dict(cfg, source=source), stage, seed, destination,
                         exports[stage['incoming']])
        export = complete_record(export)
        exports[stage['id']] = export
        seal(completed, dict(export=export, stage=stage, config_sha256=digest(cfg)))
    seal(root/'exports.json', exports)


def s8(cfg, seed):
    # The entire sequence uses one persistent optimizer/engine in torchrun.
    subprocess.run([cfg.get('trainer_python', sys.executable), '-m', 'torch.distributed.run',
                    '--standalone', '--nproc_per_node=8', str(KIT/'trace_sft.py'),
                    '--config', cfg['_config_path'], '--seed', str(seed)], check=True)


def run(cfg, study, seed):
    offline()
    need(seed in SEEDS, 'unregistered TRACE seed')
    if study == 's8':
        s8(cfg, seed)
    else:
        subset(cfg, study, seed)
