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


def d_stage(cfg, stage, seed, root, incoming, mode="all"):
    root.mkdir(parents=True, exist_ok=True)
    if mode == 'native':
        need((root/'prepared.json').is_file() and (root/'prep-key.json').is_file(),
             'CPU training preparation required before GPU admission')
    from kit.trace_jobs import timing_phase as phase
    with phase('prep'):
        from kit.trace_prompt import install
        install()
        from kit.nm_prepare import training_command, verify_model
        from kit.trace_train import timed_export as export_stage
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
            for name, checksum in prepared['files'].items():
                need(sha(root/'data'/name)==checksum, 'prepared parquet changed')
        else:
            from unittest.mock import patch
            from kit import nm_prepare
            with patch.object(nm_prepare, 'converted_rows', converted_trace_rows):
                prepared = training_command(bundle, binding, rows, root, 'freeze',
                                            'T' if stage['freeze'] else 'C', seed)
        if mode in ('prepare','native'):
            from kit.trace_score import tokenizer_identity
            key = dict(key=digest([tokenizer_identity(cfg['base']['model']),
                       prepared['native']['caps'],stage,prepared['audit']]))
            if mode == 'prepare':
                seal(root/'prep-key.json',key)
                return prepared
            need(load(root/'prep-key.json')==key,'CPU tokenizer/cap/profile binding changed')
    # Isolate the TRACE reward adapter without altering inherited callers.
    command = [str(KIT/'trace_entry.py') if arg == str(KIT/'nm_entry.py') else arg
               for arg in prepared['command']]
    command = [arg.replace('custom_reward_function.path='+str(KIT/'nm_entry.py'),
                           'custom_reward_function.path='+str(KIT/'trace_entry.py')) for arg in command]
    if (root/'native-receipt.json').exists():
        need(load(root/'native-receipt.json')['status']=='COMPLETE', 'native receipt incomplete')
        return None if mode == 'native' else export_stage(prepared, root, cfg.get('trainer_python', sys.executable))['40']
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
    from kit.trace_common import stage_command
    command = stage_command(command, [root/'data'/name for name in prepared['files']])
    from kit.trace_jobs import timing_phase as phase
    with phase('train'):
        subprocess.run(command, env=dict(os.environ, **prepared['env']),
                       cwd=prepared['cwd'], check=True)
    return None if mode == 'native' else export_stage(prepared, root, cfg.get('trainer_python', sys.executable))['40']


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
            model_record(receipt['export'])
            verify_model(receipt['export'])
            exports[stage['id']] = receipt['export']
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


from pathlib import Path
import os
import subprocess
from kit.trace_common import *


def efficiency_inventory(studies=None):
    from kit.trace_common import inventory as baseline
    value=baseline(studies)
    rows=[]
    for row in value['rows']:
        if row['phase']=='bind':
            rows.append(row)
            rows.append(dict(id='score-prep',phase='score-prep',gpus=0,depends=['bind']))
        elif row['phase']=='train' and row['study'] in SOURCES:
            study,seed=row['study'],row['seed']
            exports=[]
            for stage in stages(study,seed):
                key=stage['id']
                parent=(['export-'+stage['incoming']] if stage['incoming']!='base' else [])
                common=dict(study=study,seed=seed,stage=stage)
                rows.extend([
                    dict(id='prep-'+key,phase='prep-stage',gpus=0,
                         depends=[*row['depends'],*parent],**common),
                    dict(id='native-'+key,phase='train-stage',gpus=8,
                         depends=['prep-'+key],**common),
                    dict(id='export-'+key,phase='export-stage',gpus=0,
                         depends=['native-'+key],**common)])
                exports.append('export-'+key)
            rows.append(dict(row,phase='assemble',gpus=0,depends=exports))
        elif row['phase']=='score':
            rows.append(dict(row,depends=[*row['depends'],'score-prep']))
        else:
            rows.append(row)
    return dict(value,rows=rows,schema='trace-efficient-rows.v1')


def stage_root(cfg,study,seed,stage):
    return Path(cfg['out'])/'training'/f'{study}-{seed}'/stage['id']


def stage_incoming(cfg,study,seed,stage):
    if stage['incoming']=='base':
        return cfg['base']
    return load(Path(cfg['out'])/'training'/f'{study}-{seed}'/stage['incoming']/'exports.json')['40']


def stream_sha(path):
    """Hash checkpoint files with bounded memory, preserving the SHA-256 bytes."""
    import hashlib
    checksum=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def export_stage_cpu(prepared,root,python):
    """Same merger argv and files as NM; a verified seal commits each merge."""
    from kit.nm_prepare import verify_model
    from types import FunctionType
    # Keep the sealed verifier's code and checks, with a private hash binding.
    # A global monkeypatch would also affect concurrent inherited callers.
    verify_model=FunctionType(verify_model.__code__,dict(verify_model.__globals__,sha=stream_sha))
    root=Path(root)
    need(load(root/'native-receipt.json')['status']=='COMPLETE','native stage incomplete')
    exports={'0':prepared['incoming']}
    for step in (10,20,30,40):
        source=root/'native'/f'global_step_{step}'/'actor'
        need(source.is_dir(),'native export checkpoint absent')
        binding=digest(dict(prepared=prepared,step=step,
            source={str(p.relative_to(source)):stream_sha(p) for p in source.rglob('*') if p.is_file()}))
        receipt=root/f'merge-{step}.json'
        if receipt.exists():
            saved=load(receipt)
            need(saved['binding']==binding,'merge source changed')
            record=saved['export']
            verify_model(record)
            for name,checksum in record['files'].items():
                need(stream_sha(Path(record['model'])/name)==checksum,'merge file changed')
        else:
            target=root/f'hf-step{step}'
            subprocess.run([python,'-m','verl.model_merger','merge','--backend','fsdp',
                '--local_dir',str(source),'--target_dir',str(target)],cwd=prepared['cwd'],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
            files={p.name:stream_sha(p) for p in sorted([*target.glob('*.safetensors'),
                                                *target.glob('*.safetensors.index.json')])}
            need(files,'CPU export weights absent')
            files['config.json']=stream_sha(target/'config.json')
            record=dict(model=str(target),sha256=digest(files),files=files)
            verify_model(record)
            seal(receipt,dict(binding=binding,export=record))
        exports[str(step)]=record
    seal(root/'exports.json',exports)
    return exports


def execute_efficiency(cfg,row):
    from kit.trace_train import d_stage, complete_record
    if row['phase']=='score-prep':
        from kit.trace_score import prepare_tokens as prepare
        return prepare(cfg)
    study,seed=row['study'],row['seed']
    if row['phase']=='assemble':
        exports={'base':cfg['base']}
        for stage in stages(study,seed):
            root=stage_root(cfg,study,seed,stage)
            record=complete_record(load(root/'exports.json')['40'])
            model_record(record)
            exports[stage['id']]=record
            seal(root/'trace-complete.json',dict(export=record,stage=stage,config_sha256=digest(cfg)))
        return seal(Path(cfg['out'])/'training'/f'{study}-{seed}'/'exports.json',exports)
    stage=row['stage']
    root=stage_root(cfg,study,seed,stage)
    parent=stage_incoming(cfg,study,seed,stage)
    if row['phase']=='export-stage':
        exports=timed_export(load(root/'prepared.json'),root,cfg.get('trainer_python',__import__('sys').executable))
        record=complete_record(exports['40'])
        seal(root/'trace-complete.json',dict(export=record,stage=stage,config_sha256=digest(cfg)))
        return exports
    return d_stage(dict(cfg,source=study),stage,seed,root,parent,
                   mode='prepare' if row['phase']=='prep-stage' else 'native')


def timed_export(prepared,root,python):
    from kit.trace_jobs import timing_phase as phase
    with phase('export'):
        return export_stage_cpu(prepared,root,python)
