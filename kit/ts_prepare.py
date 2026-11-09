"""CPU preparation for hash-bound TS schedules and fresh stage-native states."""
from pathlib import Path
from collections import Counter
import json
import os
import sys

from kit import nm_prepare, ts_arms as ts
from kit.nm_common import BASE, KIT, ROOT, need, seal, sha, load, digest, checked


def audit_schedule(rows, binding, stage, seed):
    shape = ts.settings(binding, stage)
    width = shape['rows_per_update']
    need(len(rows) == 40 * width, 'TS stage must contain exactly forty complete updates')
    for start in range(0, len(rows), width):
        batch = rows[start:start + width]
        replay = [r for r in batch if r.get('m3_replay')]
        task = [r for r in batch if not r.get('m3_replay')]
        need(len(task) == shape['task_rows'] and len(replay) == shape['replay_rows'], 'TS task/replay dose differs')
        need(all(r.get('split', r.get('extra_info', {}).get('split')) == 'train' for r in batch), 'training-only rows required')
        need(all(isinstance(r.get('response'), str) and r['response'].strip() for r in batch), 'verified demonstration/replay target missing')
        if binding == 'I0':
            need(all(r.get('domain') in ('instruction', 'generic') for r in replay), 'I0 requires four generic replay slots')
        else:
            from kit.m3_schedule import CR_LAYOUT
            need(Counter(r.get('domain') for r in replay) == CR_LAYOUT[stage], 'CR replay domain geometry differs')
    return dict(seed=seed, stage=stage, schedule_sha256=digest(rows), **shape)


def prepare(bundle, binding, arm, seed, stage, incoming, root, *, attempt0_probe=False):
    from transformers import AutoTokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = checked(bundle['schedules'][str(seed)][stage])
    shape = ts.settings(binding, stage)
    audit = audit_schedule(rows, binding, stage, seed)
    tokenizer = AutoTokenizer.from_pretrained(bundle['tokenizer'], local_files_only=True, trust_remote_code=False)
    converted, token_audit = nm_prepare.converted_rows(rows, tokenizer, 'freeze', 'C', shape['caps'], shape['rows_per_update'])
    source = dict(finqa='finqa', contract_nli='contract_nli', math_hard='math_screen')[stage]
    for row in converted:
        if not row['extra_info']['m3_replay']:
            row['data_source'] = source
    # The same input bytes are prepared for all four arms. Only the native TS
    # config and the isolated entry point differ from the bound incumbent.
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data = root / 'data'
    data.mkdir(exist_ok=True)
    for name, values in [('train', converted), ('test', [dict(converted[0], extra_info=dict(converted[0]['extra_info'], demonstration=''))])]:
        pq.write_table(pa.Table.from_pylist(values), data / (name + '.parquet'))
    base = nm_prepare.verify_model(bundle['base'])
    incoming = nm_prepare.verify_model(incoming)
    from kit.nm_model import metadata
    base_binding = root / 'base.json'
    seal(base_binding, dict(model=base['model'], revision=BASE, identity=dict(model_file_hashes=base['files'])))
    identity = dict(binding=binding, arm=arm, stage=stage)
    if attempt0_probe:
        identity.update(attempt0_probe=True, schedule_sha256=audit['schedule_sha256'])
    native = dict(family='freeze', arm='C', ts=identity, seed=seed, updates=40,
                  rows_per_update=shape['rows_per_update'], caps=shape['caps'], recipe=binding,
                  base_model=base['model'], base_revision=BASE, base_binding=str(base_binding),
                  base_binding_sha256=sha(base_binding), base_metadata=metadata(base['model'], base['files']),
                  incoming=incoming, incoming_metadata=metadata(incoming['model'], incoming['files']),
                  balance_batch=bundle['balance_batch'], penalty=None, measurement=False,
                  fisher_dir=None, fisher_manifest_sha256=None,
                  receipt=str(root / 'native-receipt.json'))
    recipe = load(KIT / 'resources/p3b/v4-recipes.json')['overrides']['D']
    overrides = [v.format(kit=KIT, sdpo=bundle['sdpo'], model=incoming['model'], data=data, run=root, seed=seed) for v in recipe]
    caps = shape['caps']
    changes = {'data.train_batch_size': shape['rows_per_update'],
               'actor_rollout_ref.actor.ppo_mini_batch_size': shape['rows_per_update'],
               'actor_rollout_ref.rollout.n': 1,
               'actor_rollout_ref.rollout.temperature': 1.,
               'actor_rollout_ref.rollout.top_p': 1.,
               'actor_rollout_ref.rollout.top_k': -1,
               'actor_rollout_ref.rollout.do_sample': True,
               'actor_rollout_ref.rollout.ignore_eos': False,
               'actor_rollout_ref.rollout.repetition_penalty': 1.,
               'actor_rollout_ref.rollout.val_kwargs.n': 16,
               'actor_rollout_ref.rollout.val_kwargs.temperature': 0.,
               'actor_rollout_ref.rollout.val_kwargs.top_p': 1.,
               'actor_rollout_ref.rollout.val_kwargs.top_k': -1,
               'actor_rollout_ref.rollout.val_kwargs.do_sample': False,
               'data.max_response_length': 2048,
               'trainer.save_freq': 10,
               'trainer.total_training_steps': 40,
               'data.max_prompt_length': caps['prompt'],
               'actor_rollout_ref.actor.self_distillation.max_reprompt_len': caps['teacher'],
               'max_model_len': caps['teacher'] + 2048,
               'actor_rollout_ref.rollout.max_model_len': caps['teacher'] + 2048,
               'actor_rollout_ref.rollout.max_num_batched_tokens': caps['teacher'] + 2048,
               'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': caps['teacher'] + 2048,
               'trainer.balance_batch': bundle['balance_batch'],
               'trainer.default_local_dir': str(root / 'native'),
               'custom_reward_function.path': str(KIT / 'ts_entry.py'),
               'custom_reward_function.name': 'compute_score'}
    if arm != 'D':
        changes['actor_rollout_ref.rollout.mode'] = 'async'
        changes['actor_rollout_ref.rollout.agent.agent_loop_config_path'] = str(KIT / 'resources/ts/ts-agents.yaml')
    overrides = [v for v in overrides if v.split('=', 1)[0].lstrip('+') not in changes]
    overrides += ['++' + key + '=' + json.dumps(value) for key, value in changes.items()]
    overrides += ['+nm.' + key + '=' + json.dumps(value) for key, value in native.items()]
    env = dict(PYTHONPATH=str(ROOT) + os.pathsep + bundle['sdpo'], KIT_FINISH_GATE='1',
               WANDB_MODE='disabled', V4_CHECKPOINT_STATE='1', V4_PROFILE='scientific', ARM='D', KIT_V4_ARM='D',
               SDPO_DIR=bundle['sdpo'], VERL_FILE_LOGGER_PATH=str(root / 'metrics.jsonl'))
    overrides += ['++ray_kwargs.ray_init.runtime_env.env_vars.' + key + '=' + json.dumps(value) for key, value in env.items()]
    command = [bundle.get('trainer_python', sys.executable), '-I', str(KIT / 'ts_entry.py'), '--config-name', 'sdpo', *overrides]
    result = dict(command=command, env=env, cwd=bundle['sdpo'], native=native, incoming=incoming,
                  audit=dict(schedule=audit, tokens=token_audit),
                  files={name + '.parquet': sha(data / (name + '.parquet')) for name in ('train', 'test')})
    seal(root / 'prepared.json', result)
    return result
