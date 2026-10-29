"""Final v2 interface for isolated K/CR trainers; three seeds, no downloads.

Entry: python -m kit.m3_entry --arm C0|K|G|CR --stage finqa|contract_nli|
math_hard --seed 101|102|103 --base-model LOCAL_DIRECTORY --receipt JSON
[--profile scientific|technical-smoke|preflight] [--base-revision SHA] [--preflight JSON]
--config-name sdpo [Hydra overrides]. Numeric stage aliases 1/2/3 are accepted.

Technical smoke alone uses the M3SmokeJSONLDataset subclass in m3_runner,
with standard-library JSONL IO; its native base still imports datasets/Arrow.
Scientific dataset: native verl.utils.dataset.rl_dataset.RLHFDataset, with custom_cls
path/name null. Never use CSFTDataset on the PPO path. Entry applies the common
long-context/balance-off overrides below to every arm and stage. Dispatch is
transposed only around _update_actor and restored afterwards; no token balancing.

Reward: custom_reward_function.path points to m3_entry.py, name compute_score.
Task data_source tags: finqa, contract_nli, math_screen; replay: p3b_replay.
extra_info contains m3_replay (bool), m3_domain, m3_target_ids (including EOS),
m3_eos_token_id; m3_update/m3_position bind the round-major frozen schedule.

Scientific receipt: runs/<arm>-s<seed>-<stage>/m3-receipt.json under the isolated
M3 root. Node preflight: preflight/<arm>.json under that same root; pass its
location with --preflight. Its schema is m3-node-preflight.v1 (validator below).
A scientific receipt can be PASS only after a passing 8B/eight-H100 preflight
with at least three nonzero-LR updates, longest legal inputs, materialized
optimizer, native checkpoint/resume equality and memory/correctness gates.
Until that evidence passes it is INCOMPLETE. Timing is telemetry, NEVER a gate.
Small-model smoke does not qualify the scientific node. No evidence is invented. The disposable preflight profile runs eight ranks and four updates, then a fresh-process two-update prefix/resume equality check; only its qualification writer can produce preflight/<arm>.json.

Runner/worker: kit.m3_runner.M3TaskRunner / M3ActorRolloutRefWorker.
Actor/loss: kit.m3_actor.update_policy / kit.m3_loss. Receipt schema v3 below.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

SCHEMA_VERSION = 3
ARMS = ('C0', 'K', 'G', 'CR')
SEEDS = (101, 102, 103)
STAGES = ('finqa', 'contract_nli', 'math_hard')
STAGE_NUMBERS = (1, 2, 3)
UPDATES, TASK_ROWS, RANKS = 40, 32, 8
GENERIC_POSITIONS = (8, 16, 24, 32)
CR_LAYOUT = {
    'finqa': {'instruction':4, 'gsm8k':2, 'arc':2},
    'contract_nli': {'instruction':4, 'gsm8k':2, 'arc':2, 'finqa':8},
    'math_hard': {'instruction':4, 'gsm8k':2, 'arc':2, 'finqa':8, 'contract_nli':8},
}
BASE_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
PROJECTION_TOKENS, GLOBAL_DIVISOR = 256, 32
PROMPT_CAP, RESPONSE_CAP, REPROMPT_CAP, CONTEXT_CAP = 10240, 2048, 14336, 16384
DATASET_CLASS = 'verl.utils.dataset.rl_dataset.RLHFDataset'
TASK_DATA_SOURCES = {'finqa':'finqa', 'contract_nli':'contract_nli', 'math_hard':'math_screen'}
REPLAY_DATA_SOURCE = 'p3b_replay'
REWARD_ENTRYPOINT = 'compute_score'
RECEIPT_TEMPLATE = 'runs/{arm}-s{seed}-{stage}/m3-receipt.json'
PREFLIGHT_WORK_TEMPLATE='preflight/{arm}/{phase}/mechanics.json'
SMOKE_RECEIPT_TEMPLATE='smoke/{arm}.json'
PROFILES={'scientific':(8,40),'technical-smoke':(2,2),'preflight':(8,4)}
PREFLIGHT_TEMPLATE = 'preflight/{arm}.json'
# Shared producer/runner interface, final-review BLOCKING 1 (trainer owner).
# m3a2 builds this manifest before invoking m3_smoke preflight. Paths are relative
# to input_manifest.parent and must stay inside the initialized M3 root:
# {schema, arm, stage, seed, code_sha256, files:{relative:sha256},
#  train_files:[relative], source_schedule_manifest:relative,
#  source_schedule_audit:relative}. stage is contract_nli (preferred) or
# math_hard; seed is 101. train_files hold exactly four converted updates with
# legal task rows tagged m3_domain=contract_nli and mandatory update/position.
# Both source schedule files and all train_files must occur in files. The source
# manifest must freeze audit.json. The legal row comes from audit.longest_rows
# ['task:contract_nli'] or ['contract_nli']; prompt_tokens is authoritative.
# Do NOT serialize longest_legal_prompt_tokens: the consumer derives it from
# that hash-frozen schedule audit. A successful qualification remains immutable;
# failed attempts live under preflight/<arm>/attempts/<opaque-counter>/.
PREFLIGHT_INPUT_SCHEMA='m3-preflight-inputs.v1'
PREFLIGHT_INPUT_STAGES=('contract_nli','math_hard')
PREFLIGHT_INPUT_REQUIRED=('schema','arm','stage','seed','code_sha256','files','train_files',
                          'source_schedule_manifest','source_schedule_audit')
CODE_SHA256_SCOPE='Exact bytes of CODE_FILES below (six M3 files and four inherited trainer/contract files); excludes git metadata and all other files.'

COMMON_OVERRIDES = {
    'actor_rollout_ref.rollout.gpu_memory_utilization': .55,
    'actor_rollout_ref.rollout.free_cache_engine': True,
    'trainer.balance_batch': False, 'data.max_prompt_length': PROMPT_CAP,
    'data.max_response_length': RESPONSE_CAP, 'data.filter_overlong_prompts': False,
    'data.truncation': 'error', 'data.custom_cls.path': None, 'data.custom_cls.name': None,
    'actor_rollout_ref.actor.self_distillation.max_reprompt_len': REPROMPT_CAP,
    'max_model_len': CONTEXT_CAP, 'actor_rollout_ref.rollout.max_model_len': CONTEXT_CAP,
    'actor_rollout_ref.rollout.max_num_batched_tokens': CONTEXT_CAP,
    'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': CONTEXT_CAP,
}
LOSS_IDS = {'task':'d_forward_kl_ema_skip3_is2', 'ce':'replay_ce_eos_seqmean',
            'kl':'replay_forward_kl_base_t1_eos_seqmean'}
CONFIG_KEYS = ('arm','stage','seed','profile','base_model','base_revision','receipt','preflight','preflight_phase','input_manifest','host_reservation_bytes')
RECEIPT_REQUIRED = ('schema_version','arm','stage','seed','profile','status','loss_ids','world_size',
    'global_rows','task_rows','replay_rows','accumulation_rounds','global_divisor','completed_updates',
    'optimizer_steps','ema_steps','scheduler_steps','frozen_base_before','frozen_base_after',
    'frozen_base_has_grad','frozen_base_files_sha256','attention_implementation','gates','telemetry')
TRAINING_GATES = ('frozen_reference','step_counters','finite_losses_gradients_ema','gpu_memory','cgroup_ram')
PREFLIGHT_GATES = ('optimizer_materialized','longest_legal_rows','checkpoint_resume_equal',
    'finite_losses_gradients_ema','masks','global_sum_over_32','frozen_reference','gpu_memory','cgroup_ram','initial_replay_kl','cuda_full_kl','independent_gradient')
CODE_FILES = ('m3_contract.py','m3_actor.py','m3_runner.py','m3_entry.py','m3_loss.py','m3_smoke.py',
              'sdft_actor.py','sdft_objective.py','sdft_verl.py','v4_contract.py')


def stage_number(stage):
    if stage in STAGES: return STAGES.index(stage)+1
    if type(stage) is int and stage in STAGE_NUMBERS: return stage
    if stage in ('1','2','3'): return int(stage)
    raise ValueError('unregistered stage')


def prompt_cap(arm, stage):
    if arm not in ARMS: raise ValueError('unregistered arm')
    stage_number(stage)
    return PROMPT_CAP


def geometry(arm, stage, world_size=8):
    stage = stage_number(stage)
    if arm not in ARMS or world_size not in (1,2,8):
        raise ValueError('unregistered arm or reduction rank count')
    task = 28 if arm in ('C0','K') else 32
    replay = 4 if arm in ('C0','K') else 8*stage
    rows = task+replay
    assert rows % world_size == 0
    scale = world_size/GLOBAL_DIVISOR
    assert scale/world_size == 1/GLOBAL_DIVISOR
    return dict(task_rows=task,replay_rows=replay,global_rows=rows,
        accumulation_rounds=rows//world_size,global_divisor=GLOBAL_DIVISOR,local_loss_scale=scale)


def rank_major_indices(global_rows, world_size):
    """Transpose the frozen round-major schedule for contiguous native dispatch."""
    if world_size not in (1,2,8) or global_rows % world_size:
        raise ValueError('incomplete accumulation geometry')
    return [i for rank in range(world_size) for i in range(rank,global_rows,world_size)]


def validate_config(config):
    # preflight is optional for planning/technical smoke; scientific launch needs
    # a valid receipt, and absence can never manufacture scientific PASS.
    if set(config)-set(CONFIG_KEYS) or set(CONFIG_KEYS)-{'preflight','preflight_phase','input_manifest','host_reservation_bytes'}-set(config):
        raise ValueError('m3 config keys differ from contract')
    geometry(config['arm'],config['stage'])
    if config['seed'] not in SEEDS or config['profile'] not in PROFILES:
        raise ValueError('unregistered seed/profile')
    if config['profile'] in ('scientific','preflight') and config['base_revision']!=BASE_REVISION:
        raise ValueError('scientific base revision differs')
    if not all(config[k] for k in ('base_model','base_revision','receipt')):
        raise ValueError('base identity and receipt are mandatory')
    if config.get('preflight_phase','uninterrupted') not in ('uninterrupted','prefix','resume'):
        raise ValueError('unregistered preflight phase')
    return dict(config,preflight=config.get('preflight'),preflight_phase=config.get('preflight_phase','uninterrupted'),
                input_manifest=config.get('input_manifest'),host_reservation_bytes=config.get('host_reservation_bytes'))


def code_file_hashes():
    return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in CODE_FILES}


def code_digest():
    # Commit/tag the tested kit before node qualification. Any named file byte
    # change invalidates qualification; unrelated core commits do not. Never
    # weaken bit-exact resume equality to work around a deterministic failure.
    return hashlib.sha256(json.dumps(code_file_hashes(),sort_keys=True).encode()).hexdigest()



def validate_preflight_inputs(location, *, arm, stage, root):
    """Validate producer-frozen inputs and derive longest legal tokens from audit."""
    location=Path(location).resolve();root=Path(root).resolve()
    value=json.loads(location.read_text())
    if (any(key not in value for key in PREFLIGHT_INPUT_REQUIRED)
            or value.get('schema')!=PREFLIGHT_INPUT_SCHEMA or value.get('arm')!=arm
            or value.get('stage')!=stage or stage not in PREFLIGHT_INPUT_STAGES
            or value.get('seed')!=101 or value.get('code_sha256')!=code_digest()
            or 'longest_legal_prompt_tokens' in value):
        raise ValueError('preflight input schema/identity differs; legal length must come from frozen audit')
    if not location.is_relative_to(root):raise ValueError('preflight inputs escape isolated root')
    files=value['files']
    if not isinstance(files,dict) or not files:raise ValueError('preflight files must be frozen')
    def bound(relative):
        if relative not in files or Path(relative).is_absolute():raise ValueError('preflight input not bound')
        path=(location.parent/relative).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=files[relative]:
            raise ValueError('preflight admitted input changed or escaped root')
        return path
    for name in files:bound(name)
    if not isinstance(value['train_files'],list) or not value['train_files']:
        raise ValueError('preflight requires converted four-update schedule files')
    for name in value['train_files']:bound(name)
    source=bound(value['source_schedule_manifest']);audit_path=bound(value['source_schedule_audit'])
    frozen=json.loads(source.read_text())
    if frozen.get('technical_synthetic') is True:raise ValueError('preflight source schedules must be real')
    relative=str(audit_path.relative_to(source.parent))
    if frozen.get('files',{}).get(relative)!=files[value['source_schedule_audit']]:
        raise ValueError('preflight audit is not frozen by source schedule manifest')
    audit=json.loads(audit_path.read_text());longest=audit.get('longest_rows') or {}
    legal=longest.get('task:contract_nli') or longest.get('contract_nli')
    if not isinstance(legal,dict) or type(legal.get('prompt_tokens')) is not int or not 0<legal['prompt_tokens']<=PROMPT_CAP:
        raise ValueError('frozen schedule audit lacks admitted longest legal row')
    return dict(value,longest_legal_prompt_tokens=legal['prompt_tokens'],
                longest_legal_row_id=str(legal['id']),source_schedule_audit_sha256=files[value['source_schedule_audit']])


def validate_preflight(record, arm, *, expected_code=None):
    """Validate complete qualification evidence; a bare status PASS is insufficient."""
    if (record.get('schema')!='m3-node-preflight.v1' or record.get('arm')!=arm or
            record.get('base_revision')!=BASE_REVISION or record.get('model_size')!='8B' or
            record.get('world_size')!=8 or record.get('status')!='PASS'):
        raise ValueError('8B/eight-H100 preflight identity/status differs')
    if type(record.get('nonzero_lr_updates')) is not int or record['nonzero_lr_updates']<3:
        raise ValueError('preflight needs at least three nonzero-LR updates')
    if record.get('code_sha256') != (expected_code or code_digest()):
        raise ValueError('preflight code identity differs')
    identity = record.get('input_manifest_sha256','')
    if len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('preflight must bind admitted inputs')
    if any(record.get('gates',{}).get(key) is not True for key in PREFLIGHT_GATES):
        raise ValueError('preflight correctness or memory evidence missing/failing')
    telemetry = record.get('telemetry',{})
    for key, maximum in (('gpu_peak_fraction',.90),('cgroup_peak_fraction',.95)):
        value=telemetry.get(key)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=maximum:
            raise ValueError('preflight memory limit exceeded or missing')
    if not isinstance(telemetry.get('host_reservation_bytes'),int) or telemetry['host_reservation_bytes']<=0:
        raise ValueError('preflight must bind host RAM reservation')
    timings=telemetry.get('nonzero_update_seconds',[])
    if len(timings)<3 or any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 for v in timings):
        raise ValueError('preflight timing telemetry missing; no speed threshold applies')
    if arm=='K' and (not record.get('frozen_base_before') or
                    record['frozen_base_before']!=record.get('frozen_base_after') or
                    record.get('frozen_base_has_grad') is not False):
        raise ValueError('preflight frozen reference changed or missing')
    if not record.get('frozen_base_files_sha256') or record.get('attention_implementation') not in ('flash_attention_2','sdpa'):
        raise ValueError('preflight lacks pinned base files or attention identity')
    if arm=='K' and not -1e-5<=record.get('initial_replay_kl_token_mean',float('inf'))<=1e-3:
        raise ValueError('equal-base replay KL exceeds 1e-3')
    return record


def load_preflight(location, arm):
    if not location or not Path(location).is_file(): return None
    return validate_preflight(json.loads(Path(location).read_text()),arm)


def receipt_status(receipt):
    """Missing node qualification stays INCOMPLETE; timing never changes status."""
    if receipt['profile']=='preflight': return 'INCOMPLETE'
    if receipt['profile']=='scientific':
        if not receipt.get('node_preflight') or not receipt.get('length_audit'): return 'INCOMPLETE'
        audit=receipt['length_audit']
        if audit.get('skipped_draw_count')!=0 or audit.get('prompt_cap',0)<PROMPT_CAP or audit.get('response_cap')!=RESPONSE_CAP:
            raise ValueError('scientific long-context audit missing or incompatible')
        validate_preflight(receipt['node_preflight'],receipt['arm'])
    if any(value is False for value in receipt['gates'].values()): return 'FAIL'
    if any(receipt['gates'].get(key) is not True for key in TRAINING_GATES): return 'INCOMPLETE'
    limit=PROFILES[receipt['profile']][1]
    if any(receipt[k]!=limit for k in ('completed_updates','optimizer_steps','ema_steps','scheduler_steps')):
        return 'INCOMPLETE'
    if receipt['frozen_base_has_grad'] or receipt['frozen_base_before']!=receipt['frozen_base_after']:
        return 'FAIL'
    if receipt['arm']=='K' and not receipt['frozen_base_before']: return 'INCOMPLETE'
    return 'PASS'


def validate_receipt(receipt):
    if any(k not in receipt for k in RECEIPT_REQUIRED): raise ValueError('incomplete M3 receipt schema')
    if receipt['schema_version']!=SCHEMA_VERSION or receipt['seed'] not in SEEDS:
        raise ValueError('receipt schema/seed differs')
    expected=geometry(receipt['arm'],receipt['stage'],receipt['world_size'])
    for key in ('task_rows','replay_rows','global_rows','accumulation_rounds','global_divisor'):
        if receipt[key]!=expected[key]: raise ValueError('receipt geometry differs: '+key)
    if receipt['profile'] not in PROFILES:
        raise ValueError('receipt profile differs')
    if receipt['world_size']!=PROFILES[receipt['profile']][0]:
        raise ValueError('receipt rank count differs from profile')
    limit=PROFILES[receipt['profile']][1]
    if any(type(receipt[k]) is not int or not 0<=receipt[k]<=limit for k in
           ('completed_updates','optimizer_steps','ema_steps','scheduler_steps')):
        raise ValueError('receipt step counters invalid')
    if receipt['loss_ids']!=LOSS_IDS: raise ValueError('receipt loss identities differ')
    if not receipt['frozen_base_files_sha256'] or receipt['attention_implementation'] not in ('flash_attention_2','sdpa'):
        raise ValueError('receipt base files/attention identity missing')
    if receipt['profile']=='scientific' and receipt.get('node_preflight'):
        if (receipt['node_preflight'].get('frozen_base_files_sha256')!=receipt['frozen_base_files_sha256'] or
                receipt['node_preflight'].get('attention_implementation')!=receipt['attention_implementation']):
            raise ValueError('receipt base files/attention differ from node qualification')
    if receipt['status'] not in ('PASS','FAIL','INCOMPLETE'): raise ValueError('unknown receipt status')
    if receipt['status']=='PASS' and receipt_status(receipt)!='PASS':
        raise ValueError('PASS requires all mechanics gates and scientific node preflight')
    return receipt


def write_receipt(receipt, destination):
    validate_receipt(receipt)
    target=Path(destination); target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    temporary.replace(target)


def find_m3_root(receipt, *, require=False):
    absolute=Path(receipt).resolve()
    for parent in absolute.parents:
        if (parent/'m3-root.json').is_file():
            from kit.m3_manifest import require_m3_root
            return require_m3_root(parent).resolve()
    if require: raise ValueError('receipt must belong to an isolated initialized M3 root')
    if absolute.parent.parent.name=='runs': return absolute.parent.parent.parent
    return absolute.parent


def validate_output_layout(config, local_dir):
    root=find_m3_root(config['receipt'],require=True)
    from kit.m3_manifest import require_m3_root
    require_m3_root(root)
    template=(RECEIPT_TEMPLATE if config['profile']=='scientific' else
              PREFLIGHT_WORK_TEMPLATE if config['profile']=='preflight' else SMOKE_RECEIPT_TEMPLATE)
    expected=root/template.format(arm=config['arm'],seed=config['seed'],stage=config['stage'],phase=config.get('preflight_phase','uninterrupted'))
    destination=Path(config['receipt']).resolve()
    working=Path(local_dir).resolve()
    if destination!=expected.resolve() or not working.is_relative_to(expected.parent.resolve()) or working==expected.parent.resolve():
        raise ValueError('M3 receipt/checkpoint location differs from isolated registered layout')
    # Inspect each directory's ancestry too, rejecting symlinks into foreign state.
    if find_m3_root(working/'checkpoint.json',require=True)!=root:
        raise ValueError('checkpoint output belongs to a different M3 root')
    return root


def verify_base_files(directory, binding, revision):
    """Hash local config/safetensors against inputs/base.json BEFORE any load."""
    record=json.loads(Path(binding).read_text())
    if record.get('revision')!=revision or record.get('technical_synthetic') is True:
        raise ValueError('base binding revision/identity differs')
    expected=record.get('identity',{}).get('model_file_hashes',{})
    required={k:v for k,v in expected.items() if k=='config.json' or k.endswith('.safetensors') or k.endswith('.safetensors.index.json')}
    directory=Path(directory)
    observed_names={p.name for p in directory.iterdir() if p.name=='config.json' or p.name.endswith('.safetensors') or p.name.endswith('.safetensors.index.json')}
    if 'config.json' not in required or not any(k.endswith('.safetensors') for k in required) or observed_names!=set(required):
        raise ValueError('base binding must enumerate exact config and safetensor files')
    for name,digest in required.items():
        h=hashlib.sha256()
        with (directory/name).open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''): h.update(block)
        if h.hexdigest()!=digest: raise ValueError('frozen base files differ from inputs/base.json')
    return hashlib.sha256(json.dumps(required,sort_keys=True).encode()).hexdigest()
