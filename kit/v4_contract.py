"""Immutable v4 prompt identities, eligibility and recipe admission (CPU only)."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path(os.environ['V4_REFERENCE_ROOT']) if os.environ.get('V4_REFERENCE_ROOT') else None
INITIAL_8B_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
TOKENIZER_FILES = {'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
                   'tokenizer_config.json':'d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'}
PROMPT_CAP = 2048
DEMONSTRATION_CONTEXT_CAP = 6144
RESPONSE_CAP = 2048
REGISTERED_STEPS = 40
BATCH_PROMPTS = 32


def input_path(name,value=None):
    """Require an explicit input without resolving any personal reference defaults."""
    value=value or os.environ.get(name)
    if not value:raise ValueError(name+' is required; set it in the environment or pass its path argument')
    path=Path(value)
    if not path.is_dir():raise ValueError(name+' must name a readable input directory: '+str(path))
    return path


def prompt_hash(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def pinned_tokenizer():
    path = input_path('QWEN3_8B_TOKENIZER')
    from transformers import AutoTokenizer
    for name, expected in TOKENIZER_FILES.items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError('v4 pinned tokenizer changed: ' + name)
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


@lru_cache(maxsize=16384)
def _cached_prompt_length(serialized):
    return len(pinned_tokenizer().apply_chat_template(json.loads(serialized), add_generation_prompt=True, enable_thinking=False))


def prompt_length(messages, tokenizer=None):
    # Exactly rl_dataset.doc2len's text-only path, with the registered thinking kwarg.
    if tokenizer is None:
        return _cached_prompt_length(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False))


def technical_synthetic(value):
    """Detect the reserved smoke label in rows, metadata and corpus manifests."""
    if isinstance(value, dict):
        return value.get('technical_synthetic') is True or any(technical_synthetic(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(technical_synthetic(v) for v in value)
    return False


def check_synthetic_rows(rows, steps=40, profile_name='scientific'):
    """Gold-derived technical targets may enter only an explicitly short smoke."""
    marked = technical_synthetic(rows)
    if marked and (profile_name != 'technical-smoke' or int(steps) >= REGISTERED_STEPS):
        raise ValueError('technical_synthetic rows are refused by scientific/full-dose readers')
    return marked


def check_schedule(rows, steps, tokenizer=None):
    """Reject insufficient effective dose and every overlong row before workers exist."""
    check_synthetic_rows(rows, steps, os.environ.get('V4_PROFILE', 'scientific'))
    if len(rows) < int(steps) * BATCH_PROMPTS:
        raise ValueError('insufficient effective prompt schedule: need steps x 32 rows')
    for row in rows:
        messages = json.loads(row['prompt']) if isinstance(row['prompt'], str) else row['prompt']
        if prompt_length(messages, tokenizer) > PROMPT_CAP:
            raise ValueError('overlong training prompt: silent prompt filtering is refused')
    return {'requested_optimizer_updates': int(steps), 'effective_prompt_rows': len(rows),
            'prompt_cap': PROMPT_CAP, 'filter_overlong_prompts': False, 'tokenizer_revision': INITIAL_8B_REVISION}


def profile(steps, name='scientific'):
    steps = int(steps)
    if name not in ('scientific', 'technical-smoke'):
        raise ValueError('unknown v4 profile')
    if (name == 'scientific' and steps != REGISTERED_STEPS) or not 1 <= steps <= REGISTERED_STEPS:
        raise ValueError('v4 profile scientific requires 40 steps; use explicit technical-smoke profile for fewer')
    return name


def check_resolved(config, arm, *, profile_name='scientific',reward_path=None,resume_path=None):
    """Enforce shared constants in actual composed Hydra configuration."""
    def at(key):
        node = config
        for part in key.split('.'): node = node[part]
        return node
    profile(at('trainer.total_training_steps'), profile_name)
    resume_path=resume_path or (os.environ.get('V4_RESUME_PATH') if os.environ.get('V4_RETRY_ADMITTED')=='1' else None)
    values = {'data.train_batch_size':32, 'data.max_response_length':2048,
              'trainer.test_freq':-1, 'trainer.resume_mode':'resume_path' if resume_path else 'disable'}
    if resume_path:values['trainer.resume_from_path']=resume_path
    if arm in 'FR':
        values.update({'data.max_prompt_length':4096, 'data.max_length':6144, 'data.truncation':'error',
                       'optim.lr':1e-5, 'optim.weight_decay':.01, 'optim.v4_warmup_steps':10, 'optim.v4_scheduler':'constant'})
    else:
        values.update({'data.max_prompt_length':2048, 'data.shuffle':False, 'data.filter_overlong_prompts':False,
                       'data.truncation':'error', 'trainer.val_before_train':False,
                       'actor_rollout_ref.actor.optim.lr':1e-5, 'actor_rollout_ref.actor.optim.weight_decay':.01,
                       'actor_rollout_ref.actor.optim.lr_warmup_steps':10, 'actor_rollout_ref.actor.optim.lr_scheduler_type':'constant',
                       'actor_rollout_ref.actor.ppo_mini_batch_size':32, 'actor_rollout_ref.rollout.n':8 if arm == 'S' else 1,
                       'actor_rollout_ref.actor.self_distillation.teacher_update_rate':.05 if arm == 'S' else .01,
                       'actor_rollout_ref.actor.self_distillation.include_environment_feedback':False})
        if os.environ.get('KIT_FINISH_GATE') != '1': raise ValueError('v4 KIT_FINISH_GATE must be 1')
    seed_keys = ('trainer.seed',) if arm in 'FR' else ('data.seed', 'actor_rollout_ref.actor.data_loader_seed', 'actor_rollout_ref.actor.fsdp_config.seed')
    seeds = [at(key) for key in seed_keys]
    if len(set(seeds)) != 1 or seeds[0] not in (101, 102, 103): raise ValueError('resolved v4 seed must be shared and registered')
    if arm == 'D':
        values.update({'actor_rollout_ref.model.use_remove_padding':False, 'actor_rollout_ref.model.use_fused_kernels':False,
                       'actor_rollout_ref.actor.ulysses_sequence_parallel_size':1})
    if arm == 'S':
        values['custom_reward_function.path'] = reward_path or str(ROOT / 'kit/beds/v4_reward.py')
    for key, expected in values.items():
        if at(key) != expected: raise ValueError(f'v4 resolved setting {key} must be {expected!r}')


@lru_cache(maxsize=1)
def frozen_heldout():
    """Independent question-only prompt hashes from a checked-in public-source freeze."""
    from kit.v4_resources import resource
    path = resource('v4-heldout-prompt-identity.json')
    return json.loads(path.read_text())['prompts']


def check_heldout_rows(rows):
    check_synthetic_rows(rows)
    frozen = frozen_heldout()
    for row in rows:
        info = row.get('extra_info') or {}
        if info.get('demonstration') or info.get('rewrite'): raise ValueError('held-out targets leaked')
        key = row['data_source'] + '|' + str(info.get('index'))
        if frozen.get(key) != prompt_hash(row['prompt']):
            raise ValueError('held-out prompt differs from frozen question-only identity: ' + key)


def check_eval_items(items, bed, split):
    if bed not in ('chemistry','finqa') or split != 'test': return
    frozen = frozen_heldout()
    source = 'sciknoweval' if bed == 'chemistry' else 'finqa'
    for item in items:
        key = str(item['id']).removeprefix(source + '-test-')
        messages = item['prompt'] if isinstance(item['prompt'], list) else [{'role':'user','content':item['prompt']}]
        if frozen.get(source + '|' + key) != prompt_hash(messages):
            raise ValueError('evaluation prompt differs from frozen question-only identity: ' + key)
