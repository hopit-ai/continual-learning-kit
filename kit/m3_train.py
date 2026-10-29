#!/usr/bin/env python3
"""Message-3 K/CR stage driver (registration v2): bound base, published schedules, row conversion and one stage.

Actions (isolated M3 root only; package roots are read in place and bound write-once in <m3_out>/inputs/bound.json):
  bind-base  CPU  bind the pinned Qwen3-8B base (inputs/base.json) and verify its config/safetensors with the contract
  schedules  CPU  publish C0/K/G/CR schedules for the given --stages (seeds 101-103), cross-arm and longest-row audits;
                  stages 1-2 freeze without waiting for hard-MATH targets, stage 3 freezes separately
  train      GPU  one arm/seed/stage from that arm's preceding final, through builder A's kit.m3_entry
The contract's common long-context/balance-off/dataset/reward overrides are applied by kit.m3_entry alone; this driver
passes only the frozen D recipe minus those keys, the batch geometry and the explicit receipt/preflight locations.
"""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_common import (KIT, NotReady, R, atomic, bind, files_manifest, frozen_files, identity, require, s, t)
from kit import m3_schedule as plan
import json
import os
import subprocess
import time
import shutil

from kit import m3_contract as contract

STEPS = (0, 10, 20, 30, 40)
# Builder A's contract owns the interface: data-source tags, reward entry point, receipt/preflight locations.
DATA_SOURCE = contract.TASK_DATA_SOURCES
REPLAY_SOURCE = contract.REPLAY_DATA_SOURCE
# Stage task-schedule roots: FinQA and hard MATH are built in the M3 root; ContractNLI is C's journal read in place.
OWN_SCHEDULE = {'finqa': 'targets-finqa', 'math_hard': 'targets-math'}


def schedule_binding(a, stage, root):
    if stage in OWN_SCHEDULE:
        require(Path(root).resolve() == (a.out / OWN_SCHEDULE[stage]).resolve(), f'{stage} schedules come only from the tier-built {OWN_SCHEDULE[stage]}')
        value = dict(root=str(root), manifest_sha256=s.file_hash(Path(root) / 'manifest.json'))
        if stage == 'finqa' and not a.standin:
            from kit.m3_pools import check_finqa_root
            value['source'] = check_finqa_root(Path(root))
        if stage == 'math_hard':
            from kit.m3_pools import TARGETS_SOURCE, check_math_root
            check_math_root(Path(root), a.standin)
            value['source'] = json.loads((a.out / TARGETS_SOURCE).read_text())
        return value
    if a.standin: return None
    from kit.m3_manifest import bound_input
    return bound_input(a.out, 'contract_targets', root)


def base_files_digest(out):
    """Builder A's verify_base_files: hash config/safetensors against inputs/base.json before any use (review item 6)."""
    base = base_record(out)
    return contract.verify_base_files(base['model'], Path(out) / 'inputs/base.json', base['revision'])


def base_record(out):
    path = out / 'inputs/base.json'
    if not path.exists(): raise NotReady('bind the pinned base first: m3_train.py --action bind-base')
    return json.loads(path.read_text())


def run_root(out, arm, seed, stage):
    return out / 'runs' / f'{arm}-s{seed}-{stage}'


def record(out, arm, seed, stage, step):
    """Checkpoint for (arm, seed, stage, step). Step 0 of a stage is the arm's preceding final (base for stage 1)."""
    require(step in STEPS, 'unregistered step')
    if arm == 'base' or (stage == plan.STAGES[0] and step == 0):
        base = base_record(out); return dict(model=base['model'], receipt=base.get('receipt'), identity=base['identity'], base=True)
    if step == 0: return record(out, arm, seed, plan.STAGES[plan.STAGES.index(stage) - 1], 40)
    root = run_root(out, arm, seed, stage)
    if not (root / 'complete.json').exists(): raise NotReady('stage not complete: ' + root.name)
    frozen_files(root)
    receipt = json.loads((root / f'export-{step}.json').read_text())
    return dict(model=str(root / f'hf-step{step}'), receipt=str(root / f'export-{step}.json'), identity=receipt, base=False)


def bind_base(a):
    """inputs/base.json in the shape kit.m3_contract.verify_base_files reads (revision + identity.model_file_hashes covering
    config.json and every safetensors file); the scientific binding is verified before it is written."""
    require(a.model, '--model pinned base directory required')
    if a.standin: observed = dict(technical_synthetic=True, model_file_hashes={'config.json': 'standin'}, model=str(a.model))
    else: observed = identity(a.model, 'student', a.receipt)
    value = dict(model=str(Path(a.model).resolve()), receipt=str(a.receipt) if a.receipt else None, identity=observed,
                 revision=contract.BASE_REVISION, technical_synthetic=a.standin)
    if not a.standin:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='m3-base-') as scratch:
            candidate = Path(scratch) / 'base.json'; candidate.write_text(json.dumps(value))
            value['frozen_base_files_sha256'] = contract.verify_base_files(value['model'], candidate, value['revision'])
    bind(a.out / 'inputs/base.json', value)
    print('Bound M3 base')


# ----------------------------------------------------------------------------- schedules

def task_schedule(root, seed):
    manifest = frozen_files(root)
    rows = t.read_jsonl(root / f'schedule-{seed}.jsonl')
    require(len(rows) == plan.UPDATES * plan.TASK_ROWS, 'task schedule must contain 1,280 exposures')
    return rows, manifest


def tokenizer_for(a):
    if a.standin: return None
    require(a.tokenizer, 'pinned tokenizer required'); identity(a.tokenizer, 'student')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)


def as_chat(prompt):
    if isinstance(prompt, str) and prompt.startswith('['): prompt = json.loads(prompt)
    return [dict(role='user', content=prompt)] if isinstance(prompt, str) else prompt


def longest_rows(rows_by_domain, tokenizer):
    """Tokenize the longest admitted row of every domain through the common rollout and teacher configuration.

    Rollout prompt <= 10,240, stored response <= 2,048 incl. EOS, fully rendered teacher context (prompt +
    demonstration) <= 14,336. Any excess is a preparation failure (never truncated, never dropped)."""
    from kit.sdft_objective import teacher_messages
    report = {}
    for domain, rows in rows_by_domain.items():
        if not rows: continue
        measured = []
        for r in rows:
            from kit.m3_pools import validate_encoding
            prompt = as_chat(r['prompt']); pids, tids = validate_encoding(tokenizer, prompt, r['response'])
            measured.append((len(pids), len(tids), r))
        prompt_tokens, response_tokens, row = max(measured, key=lambda x: (x[0] + x[1], x[0]))
        teacher = tokenizer.apply_chat_template(teacher_messages(as_chat(row['prompt']), row['response']), tokenize=True,
                                                add_generation_prompt=True, enable_thinking=False)
        value = dict(id=str(row['id']), prompt_tokens=prompt_tokens, response_tokens_including_eos=response_tokens, teacher_context_tokens=len(teacher),
                     max_prompt=max(m[0] for m in measured), max_response=max(m[1] for m in measured))
        require(value['max_prompt'] <= plan.PROMPT_CAP and value['max_response'] <= plan.RESPONSE_CAP and value['teacher_context_tokens'] <= plan.REPROMPT_CAP,
                f'{domain} longest row exceeds the common configuration: ' + json.dumps(value))
        report[domain] = value
    return report


def schedule_dir(out, stage, arm=None):
    return Path(out) / ('schedules-coverage' if arm in ('G', 'CR') else 'schedules') / stage


def schedule_path(out, arm, seed, stage):
    return schedule_dir(out, stage, arm) / arm / f's{seed}.jsonl'


def audit_coverage(out, stage, seed, built, summaries):
    """The coverage pass must assert all four arms against the immutable shared pass."""
    root = schedule_dir(out, stage, 'C0'); frozen_files(root)
    shared_summaries = json.loads((root / 'summaries.json').read_text())
    schedules = dict(built); values = dict(summaries)
    for arm in ('C0', 'K'):
        schedules[arm] = t.read_jsonl(schedule_path(out, arm, seed, stage))
        values[arm] = shared_summaries[f'{arm}|{seed}|{stage}']
    return plan.audit_arms(schedules, values)


def schedules_job(a):
    """Schedules for the requested stages (4 arms x 3 seeds each), audited and frozen per stage before training.
    Stages 1-2 freeze first; hard MATH freezes separately once its targets exist, so no early stage waits."""
    from kit.m3_pools import load, pool_root, read_panels, parse_specs, task_evidence, SUPERSET
    arms = list(getattr(a, 'arms', None) or plan.ARMS)
    if set(arms) & {'C0', 'K'} and set(arms) & {'G', 'CR'}:
        from copy import copy
        for group in (['C0', 'K'], ['G', 'CR']):
            part = copy(a); part.arms = [arm for arm in group if arm in arms]; schedules_job(part)
        return
    scope = arms[0]
    stages = list(a.stages or plan.STAGES)
    require(set(stages) <= set(plan.STAGES) and len(a.schedule_roots) == len(stages), 'one task schedule root per requested stage')
    roots = dict(zip(stages, a.schedule_roots))
    tokenizer = tokenizer_for(a)
    pools = {}; status = {}; lengths = None
    for arm in arms:
        try:
            pools[arm], meta = load(a.out, arm); status[arm] = 'READY'; lengths = meta['lengths'] or lengths
        except NotReady as exc: status[arm] = 'PENDING: ' + str(exc)
    if not pools: raise NotReady('requested replay pools unavailable')
    if 'C0' in arms: require(status['C0'] == 'READY', 'C0 instruction pool unavailable: ' + status['C0'])
    panels = []
    if not a.standin:
        require(getattr(a, 'holdout', None), 'frozen holdout evidence required for task schedules')
        panels, panel_ids = read_panels(parse_specs(a.panel) | parse_specs(a.holdout), False)
        for arm in pools:
            frozen_ids = json.loads((pool_root(a.out, arm) / 'inputs.json').read_text())['panels']
            require(panel_ids == frozen_ids, 'task panels/holdouts changed after pool freeze')
    for stage in stages:
        root = schedule_dir(a.out, stage, scope)
        if (root / 'manifest.json').exists():
            frozen_files(root)
            if scope in ('G', 'CR'):
                sums = json.loads((root / 'summaries.json').read_text())
                for seed in plan.SEEDS:
                    built = {arm: t.read_jsonl(schedule_path(a.out, arm, seed, stage)) for arm in ('G', 'CR')}
                    audit_coverage(a.out, stage, seed, built, {arm: sums[f'{arm}|{seed}|{stage}'] for arm in built})
            print('SKIP: frozen M3 schedules for', stage); continue
        binding = schedule_binding(a, stage, roots[stage])
        names = []; audits = {}; summaries = {}; tasks = []; task_verification = {}
        for seed in plan.SEEDS:
            task, _ = task_schedule(roots[stage], seed); tasks += task
            if not a.standin: task_verification[str(seed)] = task_evidence(task, stage, panels, getattr(a, 'superset', None) or SUPERSET)
            built = {}; sums = {}
            for arm in arms:
                if arm not in pools: continue
                rows, summary = plan.build(arm, seed, stage, task, pools[arm], lengths=lengths)
                require(summary['skipped_draw_count'] == 0, 'filtered draw')
                built[arm] = rows; sums[arm] = summary; name = f'{arm}/s{seed}.jsonl'
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                t.write_jsonl(root / name, rows); names.append(name)
                summaries[f'{arm}|{seed}|{stage}'] = dict(summary, task_manifest=s.file_hash(roots[stage] / 'manifest.json'))
            if scope in ('G', 'CR'):
                if set(built) != {'G', 'CR'}: raise NotReady('coverage freeze requires both G and CR plus frozen C0/K schedules')
                audits[str(seed)] = audit_coverage(a.out, stage, seed, built, sums)
            elif set(built) == set(plan.ARMS): audits[str(seed)] = plan.audit_arms(built, sums)
            else:
                # C0/K still share the four IDs and 28 task rows when CR/G are PENDING.
                require(len({plan.digest(plan.shared_generic(v)) for v in built.values()}) == 1, 'shared generic IDs differ')
                audits[str(seed)] = dict(partial=sorted(built))
        longest = None
        if tokenizer is not None:
            union = {}
            for value in pools.values(): union.update(value)
            union['task:' + stage] = tasks
            longest = longest_rows(union, tokenizer)
        earlier = {}
        for prior in plan.STAGES[:plan.STAGES.index(stage)]:
            path = schedule_dir(a.out, prior, scope) / 'summaries.json'
            if path.exists(): earlier.update(json.loads(path.read_text()))
        atomic(root / 'summaries.json', summaries)
        atomic(root / 'audit.json', dict(stage=stage, audits=audits, availability=status, contract=plan.contract_agreement(), longest_rows=longest,
                                         schedule_binding=binding, task_verification=task_verification, cross_stage_repeats=plan.cross_stage_repeats([*earlier.values(), *summaries.values()]),
                                         exposure_totals={f'{k[0]}|{k[1]}': v for k, v in plan.exposure_totals().items() if k[1] == stage}))
        atomic(root / 'manifest.json', files_manifest(root, names + ['summaries.json', 'audit.json'], technical_synthetic=a.standin, stage=stage,
                                                      pools=s.file_hash(pool_root(a.out, arms[0]) / 'manifest.json')))
        print('Frozen', len(names), 'M3 schedules for', stage, json.dumps(status, sort_keys=True))


# ----------------------------------------------------------------------------- conversion

def lookup_rows(task, pools):
    rows = {('task', str(r['id'])): r for r in task}
    for domain, values in pools.items():
        for r in values: rows[(domain, str(r['id']))] = r
    return rows


def convert(schedule, task, pools, tokenizer, arm, stage, replay_source=REPLAY_SOURCE):
    """Builder A's row contract: extra_info m3_replay, m3_domain, m3_target_ids (incl. EOS), m3_eos_token_id."""
    sources = lookup_rows(task, pools)
    converted = []; tokens = dict(input=0, task_response_including_eos=0, replay_response_including_eos=0)
    for item in schedule:
        key = ('task', item['id']) if item['role'] == 'task' else (item['domain'], item['id'])
        row = sources.get(key); require(row is not None, 'scheduled row missing from frozen inputs: ' + repr(key))
        require(plan.target_sha256(row) == item['target_sha256'], 'scheduled target hash changed: ' + repr(key))
        prompt = as_chat(row['prompt']) if item['role'] == 'task' else (row['prompt'] if isinstance(row['prompt'], list) else [dict(role='user', content=row['prompt'])])
        response = row['response']
        if tokenizer is not None:
            from kit.m3_pools import validate_encoding
            pids, tids = validate_encoding(tokenizer, prompt, response)
            eos = tokenizer.eos_token_id
            require(len(pids) <= plan.PROMPT_CAP, f'{key} prompt {len(pids)} exceeds the common {plan.PROMPT_CAP} cap; never truncated')
            require(len(tids) <= plan.RESPONSE_CAP and tids and tids[-1] == eos, 'stored response must fit 2,048 tokens and end in EOS')
            if item['role'] == 'task':
                from kit.sdft_objective import teacher_messages
                teacher_messages(prompt, response)
        else:
            pids, tids, eos = list(range(len(str(prompt)) // 8)), list(range(len(response) // 8)) + [0], 0
        replay = item['role'] == 'replay'
        tokens['input'] += len(pids); tokens['replay_response_including_eos' if replay else 'task_response_including_eos'] += len(tids)
        converted.append(dict(id=item['id'], prompt=prompt, data_source=replay_source if replay else DATA_SOURCE[stage],
            reward_model=dict(style='rule', ground_truth='' if replay else str(row.get('gold', row.get('reward_model', {}).get('ground_truth', '')))),
            extra_info=dict(index=item['id'], split='train', demonstration=response, m3_replay=replay, m3_domain=item['domain'],
                            m3_target_ids=list(tids), m3_eos_token_id=int(eos), m3_update=item['update'], m3_position=item['position'])))
    return converted, tokens


# ----------------------------------------------------------------------------- disposable node qualification

def prepare_preflight(a):
    """Build the contract's four-update native inputs from real, frozen stage-2 legal schedules."""
    from kit.m3_pools import load
    require(a.seed == 101 and a.stage in contract.PREFLIGHT_INPUT_STAGES and not a.standin, 'node preflight needs real legal inputs and seed 101')
    if not getattr(a, 'host_reservation_bytes', None) or a.host_reservation_bytes <= 0:
        raise NotReady('declare the whole-node host RAM reservation in bytes before preflight')
    source = schedule_dir(a.out, a.stage, a.arm); frozen_files(source)
    root = a.out / 'preflight-inputs' / a.arm; manifest = root / 'manifest.json'
    if manifest.exists():
        contract.validate_preflight_inputs(manifest, arm=a.arm, stage=a.stage, root=a.out)
        return root / 'node-plan.json'
    audit = json.loads((source / 'audit.json').read_text())
    require(schedule_binding(a, a.stage, a.schedule_root) == audit.get('schedule_binding'), 'preflight task source differs from frozen schedule audit')
    longest = (audit.get('longest_rows') or {}).get('task:contract_nli') or (audit.get('longest_rows') or {}).get('contract_nli')
    require(longest is not None, 'frozen schedule audit must identify its longest admitted legal row')
    pools, _ = load(a.out, a.arm)
    task, _ = task_schedule(a.schedule_root, 101)
    legal = next((r for r in task if str(r['id']) == str(longest['id'])), None) if a.stage == 'contract_nli' else None
    if legal is None: legal = next((r for r in pools.get('contract_nli', []) if str(r['id']) == str(longest['id'])), None)
    require(legal is not None, 'longest audited legal row missing from frozen task/pool inputs')
    tokenizer = tokenizer_for(a)
    pids, _ = t.tokenize_student(tokenizer, as_chat(legal['prompt']), legal['response'])
    require(len(pids) == longest['prompt_tokens'], 'legal prompt length differs from frozen schedule audit')
    schedule = t.read_jsonl(schedule_path(a.out, a.arm, 101, a.stage))
    selected = [dict(r) for r in schedule if r['update'] <= 4]
    require(len(selected) == 4 * contract.geometry(a.arm, a.stage)['global_rows'], 'preflight requires four complete frozen updates')
    for item in selected:
        if item['role'] == 'task':
            item.update(id=str(legal['id']), domain='contract_nli', target_sha256=plan.target_sha256(legal))
    converted, _ = convert(selected, [*task, legal], pools, tokenizer, a.arm, a.stage)
    for row in converted:
        if not row['extra_info']['m3_replay']: row['data_source'] = DATA_SOURCE['contract_nli']
    import pyarrow as pa
    import pyarrow.parquet as pq
    if root.exists():
        archive = a.out / 'preflight' / a.arm / 'attempts' / ('inputs-' + str(time.time_ns()))
        hashes = {str(p.relative_to(root)): s.file_hash(p) for p in root.rglob('*') if p.is_file()}
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(archive))
        atomic(archive / 'attempt-manifest.json', dict(status='INCOMPLETE', files=hashes))
    data = root / 'data'; data.mkdir(parents=True)
    for part, rows in (('train', converted), ('val', converted[:contract.geometry(a.arm, a.stage)['global_rows']]), ('test', converted[:1])):
        pq.write_table(pa.Table.from_pylist(rows), data / (part + '.parquet'))
    base = base_record(a.out)
    cmd, _ = command(a, a.arm, a.stage, base['model'], data, a.out / 'preflight' / a.arm / 'uninterrupted', base, contract.geometry(a.arm, a.stage))
    cmd[cmd.index('--profile') + 1] = 'preflight'
    cmd[cmd.index('--receipt') + 1] = str((a.out / contract.PREFLIGHT_WORK_TEMPLATE.format(arm=a.arm, phase='uninterrupted')).resolve())
    end = cmd.index('sdpo') + 1
    phase_owned = {'trainer.total_training_steps', 'trainer.save_freq', 'trainer.default_local_dir', 'trainer.resume_mode',
                   'trainer.resume_from_path', 'trainer.del_local_ckpt_after_load', 'trainer.rollout_data_dir', 'trainer.validation_data_dir',
                   'vars.log_dir', 'vars.ckpt_dir', 'ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH'}
    cmd = [*cmd[:end], *[v for v in cmd[end:] if override_key(v) not in phase_owned]]
    cmd[end:end] = ['--input-manifest', str(manifest.resolve()), '--host-reservation-bytes', str(a.host_reservation_bytes)]
    atomic(root / 'node-plan.json', dict(command=cmd))
    def relative(path): return os.path.relpath(Path(path).resolve(), root.resolve())
    sources = [source / 'manifest.json', source / 'audit.json', schedule_path(a.out, a.arm, 101, a.stage)]
    files = {relative(path): s.file_hash(path) for path in [*sources, *data.glob('*.parquet'), root / 'node-plan.json']}
    atomic(manifest, dict(schema=contract.PREFLIGHT_INPUT_SCHEMA, arm=a.arm, stage=a.stage, seed=101,
           code_sha256=contract.code_digest(), files=files, train_files=['data/train.parquet'],
           source_schedule_manifest=relative(source / 'manifest.json'), source_schedule_audit=relative(source / 'audit.json')))
    contract.validate_preflight_inputs(manifest, arm=a.arm, stage=a.stage, root=a.out)
    return root / 'node-plan.json'


def preflight_job(a):
    from kit.m3_smoke import run_preflight
    node = prepare_preflight(a)
    forwarded = dict(CUDA_VISIBLE_DEVICES=a.gpu, PYTHONPATH=str(KIT.parent) + ':' + str(a.sdpo.resolve()),
                     SDPO_DIR=str(a.sdpo.resolve()), QWEN3_8B_TOKENIZER=str(a.tokenizer),
                     HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    before = {key: os.environ.get(key) for key in forwarded}
    os.environ.update(forwarded)
    try: return run_preflight(node)
    finally:
        for key, value in before.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


# ----------------------------------------------------------------------------- training

def names():
    return dict(receipt=contract.RECEIPT_TEMPLATE, preflight=contract.PREFLIGHT_TEMPLATE, replay_source=contract.REPLAY_DATA_SOURCE,
                task_sources=contract.TASK_DATA_SOURCES, reward=f'kit/m3_entry.py:{contract.REWARD_ENTRYPOINT}', dataset=contract.DATASET_CLASS)


# Keys kit.m3_entry sets itself (contract COMMON_OVERRIDES: caps, balance-off, native RLHFDataset; reward entry point; and
# its forwarded runtime environment). This driver never passes them, so the config never depends on override order.
ENTRY_ENV = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'KIT_FINISH_GATE', 'WANDB_MODE', 'V4_PROFILE')
ENTRY_OWNED = frozenset([*contract.COMMON_OVERRIDES, 'custom_reward_function.path', 'custom_reward_function.name',
                         *('ray_kwargs.ray_init.runtime_env.env_vars.' + k for k in ENTRY_ENV)])
GEOMETRY_KEYS = ('data.train_batch_size', 'actor_rollout_ref.actor.ppo_mini_batch_size')
FORWARDED = ('P3B_DATA', 'SDPO_DIR', 'QWEN3_8B_TOKENIZER', 'KIT_V4_ARM', 'V4_CHECKPOINT_STATE', 'V4_TELEMETRY', 'ARM', 'PYTHONPATH', 'VERL_FILE_LOGGER_PATH')


def override_key(value):
    return value.split('=', 1)[0].lstrip('+')


def command(a, arm, stage, model, data, run, base, geometry, resume=None):
    """The frozen v4 D recipe minus every entry-owned key, the M3 batch geometry and explicit receipt/preflight paths
    (review items 2 and 11). Each Hydra key appears exactly once."""
    recipe = json.loads((R / 'v4-recipes.json').read_text())['overrides']['D']
    recipe = [x.format(kit=KIT, sdpo=a.sdpo.resolve(), model=Path(model).resolve(), data=data, run=run, seed=a.seed) for x in recipe]
    rows = str(geometry['global_rows'])
    overrides = []
    for value in recipe:
        key = override_key(value)
        if key in ENTRY_OWNED or key in ('trainer.resume_mode', 'trainer.resume_from_path'): continue
        overrides.append(value.split('=', 1)[0] + '=' + rows if key in GEOMETRY_KEYS else value)
    overrides += ['++trainer.resume_mode=' + ('resume_path' if resume else 'disable')]
    if resume: overrides += ['++trainer.resume_from_path=' + str(Path(resume).resolve())]
    present = {override_key(v) for v in overrides}
    overrides += [f'++{k}={rows}' for k in GEOMETRY_KEYS if k not in present]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONPATH=str(KIT.parent) + ':' + str(a.sdpo.resolve()), P3B_DATA=str(data),
               SDPO_DIR=str(a.sdpo.resolve()), QWEN3_8B_TOKENIZER=a.tokenizer, VERL_FILE_LOGGER_PATH=str(run / 'metrics.jsonl'),
               V4_CHECKPOINT_STATE='1', V4_TELEMETRY='1', ARM='D', KIT_V4_ARM='D', EXPERIMENT=run.name, TASK='m3')
    overrides += [f'+ray_kwargs.ray_init.runtime_env.env_vars.{key}={json.dumps(env[key])}' for key in FORWARDED]
    keys = [override_key(v) for v in overrides]
    require(len(keys) == len(set(keys)) and not set(keys) & ENTRY_OWNED, 'M3 overrides must name each key once and leave entry-owned keys alone')
    preflight = (Path(a.out) / contract.PREFLIGHT_TEMPLATE.format(arm=arm)).resolve()
    flags = ['--arm', arm, '--stage', stage, '--seed', str(a.seed), '--profile', 'scientific', '--base-model', str(base['model']),
             '--base-revision', base['revision'], '--receipt', str(receipt_path(a.out, arm, a.seed, stage).resolve()), '--preflight', str(preflight)]
    return [a.trainer_python, '-m', 'kit.m3_entry', *flags, '--config-name', 'sdpo', *overrides], env


def receipt_path(out, arm, seed, stage):
    return Path(out) / contract.RECEIPT_TEMPLATE.format(arm=arm, seed=seed, stage=stage)


def validate_stage_receipt(out, arm, seed, stage):
    """Open and validate the actual hash-frozen scientific receipt, including its base identity."""
    run = run_root(Path(out), arm, seed, stage)
    manifest = frozen_files(run)
    path = receipt_path(out, arm, seed, stage)
    require(path.name in manifest['files'], 'scientific receipt not frozen in run manifest')
    receipt = contract.validate_receipt(json.loads(path.read_text()))
    require((receipt['arm'], receipt['seed'], contract.stage_number(receipt['stage']), receipt['profile']) ==
            (arm, seed, contract.stage_number(stage), 'scientific'), 'scientific receipt identity differs')
    base = base_record(Path(out))
    require(not receipt.get('technical_synthetic') and not base.get('technical_synthetic'), 'synthetic scientific receipt/base')
    require(receipt['frozen_base_files_sha256'] == base['frozen_base_files_sha256'], 'receipt frozen base differs')
    return receipt


def native_checkpoint(run, arm, seed, stage):
    """Latest complete native state; partial newer directories stay untouched."""
    from kit.v4_retry import tree_hash
    root = Path(run) / 'sdft'
    checkpoints = sorted(root.glob('global_step_*'), key=lambda p: int(p.name.split('_')[-1]), reverse=True)
    for path in checkpoints:
        step = int(path.name.split('_')[-1])
        require(0 < step <= 40, 'unregistered checkpoint dose')
        actor = path / 'actor'
        required = [path / 'data.pt', actor / 'm3-state.json']
        for rank in range(8):
            required += [actor / f'{kind}_world_size_8_rank_{rank}.pt' for kind in ('model', 'optim', 'extra_state')]
            required += [actor / 'm3-ema' / f'model_world_size_8_rank_{rank}.pt']
        if not all(p.is_file() and p.stat().st_size for p in required): continue
        meta = json.loads((actor / 'm3-state.json').read_text())
        require(all(meta.get(k) == v for k, v in dict(arm=arm, seed=seed, stage=stage, profile='scientific',
                base_revision=contract.BASE_REVISION, code_sha256=contract.code_digest()).items()), 'native resume identity differs')
        require(all(meta.get(k) == step for k in ('completed_updates', 'optimizer_steps', 'ema_steps', 'scheduler_steps')),
                'native resume counters differ')
        snapshot = dict(step=step, sha256=tree_hash(path))
        bind(Path(run) / 'native-snapshots' / (path.name + '.json'), snapshot)
        return path
    return None


def archive_failed_attempt(run):
    """Retain interrupted pre-checkpoint state and byte hashes before an identical restart."""
    run = Path(run); archive = run / 'attempts' / str(time.time_ns())
    names = []
    for name in ('sdft', 'rollouts', 'validation', 'started.json', 'trainer-exit.json', 'metrics.jsonl', 'm3-receipt.json'):
        source = run / name
        if not source.exists(): continue
        if source.is_dir(): names += [str(p.relative_to(run)) for p in source.rglob('*') if p.is_file()]
        else: names.append(name)
    hashes = {name: s.file_hash(run / name) for name in names}
    archive.mkdir(parents=True)
    for name in ('sdft', 'rollouts', 'validation', 'started.json', 'trainer-exit.json', 'metrics.jsonl', 'm3-receipt.json'):
        source = run / name
        if source.exists(): shutil.move(str(source), str(archive / name))
    require(all(s.file_hash(archive / name) == value for name, value in hashes.items()), 'attempt archive changed interrupted evidence')
    atomic(archive / 'manifest.json', dict(files=hashes, reason='interruption_before_complete_native_checkpoint'))
    return archive


def training_action(run, arm, seed, stage):
    checkpoint = native_checkpoint(run, arm, seed, stage)
    if checkpoint is not None and checkpoint.name == 'global_step_40':
        if not (Path(run) / 'm3-receipt.json').exists():
            raise NotReady('step-40 state preserved; receipt-only native resume is required before export (trainer interface pending)')
        return 'export', checkpoint
    if checkpoint is not None: return 'resume', checkpoint
    if (Path(run) / 'started.json').exists(): archive_failed_attempt(run)
    return 'start', None


def train_job(a):
    from kit.m3_pools import load, pool_root
    from kit.m3_score import unit_root, PANELS
    arm, seed, stage = a.arm, a.seed, a.stage
    pools, meta = load(a.out, arm)
    frozen_files(schedule_dir(a.out, stage, arm))
    if not a.standin:
        from kit.m3_pools import SUPERSET
        audit = json.loads((schedule_dir(a.out, stage, arm) / 'audit.json').read_text())
        evidence = (audit.get('task_verification') or {}).get(str(seed), {})
        if evidence.get('status') != 'VERIFIED': raise NotReady('task exclusion/candidate evidence missing')
        require(evidence.get('superset_sha256') == s.file_hash(SUPERSET), 'task candidate universe changed')
    schedule = t.read_jsonl(schedule_path(a.out, arm, seed, stage))
    binding = schedule_binding(a, stage, a.schedule_root)
    task, _ = task_schedule(a.schedule_root, seed)
    tokenizer = tokenizer_for(a)
    rebuilt, _ = plan.build(arm, seed, stage, task, pools, lengths=meta['lengths'])
    require(rebuilt == schedule, 'published schedule differs from frozen inputs')
    geometry = contract.geometry(arm, stage)
    require(geometry['global_rows'] == plan.rows_per_update(arm, stage), 'contract geometry differs from the schedule')
    incoming = record(a.out, arm, seed, stage, 0); base = base_record(a.out)
    # Review item 6: the frozen base is the identity bound in inputs/base.json, re-verified before every stage.
    base_files = None if a.standin else base_files_digest(a.out)
    require(a.standin or base.get('frozen_base_files_sha256') == base_files, 'frozen base files differ from their bind-base record')
    for panel in PANELS:
        if not (unit_root(a.out, arm, seed, stage, 0, panel) / 'manifest.json').exists():
            raise NotReady(f'score the incoming checkpoint before training: {arm}/{seed}/{stage}/{panel}')
    run = run_root(a.out, arm, seed, stage); data = a.out / 'data' / run.name
    observed = dict(arm=arm, seed=seed, stage=stage, incoming=incoming['identity'], base=base['identity'], schedule=s.digest(schedule),
                    pools=s.file_hash(pool_root(a.out, arm) / 'manifest.json'), task_manifest=s.file_hash(a.schedule_root / 'manifest.json'),
                    geometry=geometry, limits=contract.COMMON_OVERRIDES, steps=40, lr=1e-5, warmup=10, fresh_optimizer=True, schedule_binding=binding,
                    frozen_base_files_sha256=base_files,
                    ema='incoming-initialised .99/.01', interface=names(), technical_synthetic=a.standin)
    bind(a.out / 'identities' / (run.name + '.json'), observed)
    if (run / 'complete.json').exists():
        frozen_files(run)
        if not a.standin: validate_stage_receipt(a.out, arm, seed, stage)
        print('SKIP:', run.name); return
    started = time.monotonic()
    try:
        converted, tokens = convert(schedule, task, pools, tokenizer, arm, stage)
        require(len(converted) == plan.UPDATES * geometry['global_rows'], 'converted rows differ from 40 x global rows')
        if a.standin:
            for step in STEPS[1:]:
                atomic(run / f'export-{step}.json', dict(technical_synthetic=True, step=step,
                       model=f'm3-standin:{arm}:s{seed}:{stage}:k{step}', model_file_hashes={'config.json': s.digest([arm, seed, stage, step])}))
            receipt = dict(status='INCOMPLETE', technical_synthetic=True)
        else:
            require(len(set(a.gpu.split(','))) == 8, 'training uses eight distinct GPUs on one node')
            try: preflight = contract.load_preflight(a.out / contract.PREFLIGHT_TEMPLATE.format(arm=arm), arm)
            except ValueError as exc: raise NotReady(f'8B eight-H100 node preflight for {arm} not passing: {exc}') from exc
            if preflight is None:
                raise NotReady(f'8B eight-H100 node preflight receipt for {arm} missing; blocks only {arm} comparisons')
            require(a.sdpo is not None, 'pinned SDPO checkout required')
            revision = subprocess.check_output(['git', '-C', str(a.sdpo), 'rev-parse', 'HEAD'], text=True).strip()
            require(revision == '7c457fc1b1f636ae794eb0362ba37d4743b06fbc', 'wrong SDPO revision')
            import pyarrow as pa
            import pyarrow.parquet as pq
            data.mkdir(parents=True, exist_ok=True)
            if not (data / 'manifest.json').exists():
                for part, rows in (('train', converted), ('val', converted[:geometry['global_rows']]), ('test', converted[:1])):
                    pq.write_table(pa.Table.from_pylist(rows), data / (part + '.parquet'))
                atomic(data / 'manifest.json', files_manifest(data, ['train.parquet', 'val.parquet', 'test.parquet'], rows=len(converted)))
            frozen_files(data)
            action, checkpoint = training_action(run, arm, seed, stage)
            cmd, env = command(a, arm, stage, incoming['model'], data, run, base, geometry, checkpoint if action == 'resume' else None)
            attempt = str(time.time_ns())
            bind(run / 'commands' / (attempt + '.json'), dict(action=action, command=cmd))
            if action != 'export':
                if not (run / 'started.json').exists(): atomic(run / 'started.json', observed)
                clock = time.monotonic()
                done = subprocess.run(cmd, cwd=a.sdpo, env=env)
                atomic(run / 'trainer-attempts' / (attempt + '.json'), dict(returncode=done.returncode, seconds=time.monotonic()-clock, action=action))
                if done.returncode: raise subprocess.CalledProcessError(done.returncode, cmd)
                require(native_checkpoint(run, arm, seed, stage).name == 'global_step_40', 'final complete native state missing')
            receipt = contract.validate_receipt(json.loads(receipt_path(a.out, arm, seed, stage).read_text()))
            require(receipt['status'] != 'FAIL', 'M3 trainer receipt FAIL')
            require((receipt['arm'], receipt['seed'], contract.stage_number(receipt['stage']), receipt['profile']) ==
                    (arm, seed, contract.stage_number(stage), 'scientific'), 'trainer receipt arm/seed/stage/profile differs')
            require(receipt['frozen_base_files_sha256'] == base_files, 'trainer used a frozen base other than the bound inputs/base.json')
            from kit.v4_sft import complete_export
            for step in STEPS[1:]:
                export = run / f'hf-step{step}'
                if not complete_export(export):
                    subprocess.run([a.trainer_python, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp', '--local_dir',
                                    str(run / 'sdft' / f'global_step_{step}' / 'actor'), '--target_dir', str(export)], cwd=a.sdpo, env=env, check=True)
                require(complete_export(export), 'incomplete merged export')
                atomic(run / f'export-{step}.json', identity(export))
        atomic(run / 'summary.json', dict(observed, scheduled_tokens=tokens, receipt_status=receipt['status'], seconds=time.monotonic() - started))
        atomic(run / 'manifest.json', files_manifest(run, ['summary.json'] + ([] if a.standin else ['m3-receipt.json']) + [f'export-{k}.json' for k in STEPS[1:]], technical_synthetic=a.standin))
        if not a.standin: validate_stage_receipt(a.out, arm, seed, stage)
        atomic(run / 'complete.json', dict(complete=True, receipt_status=receipt['status']))
    finally:
        atomic(run / 'training-costs' / f'{time.time_ns()}.json', dict(kind='m3_training', arm=arm, seed=seed, stage=stage,
               gpu_hours=0 if a.standin else 8 * (time.monotonic() - started) / 3600, wall_seconds=time.monotonic() - started,
               completed=(run / 'complete.json').exists(), technical_synthetic=a.standin))


def job(a):
    from kit.m3_manifest import require_m3_root
    require_m3_root(a.out)
    from contextlib import nullcontext
    from kit.m3_manifest import gpu_execution
    with gpu_execution(a.out) if not a.standin and a.action in ('train', 'preflight') else nullcontext():
        return {'bind-base': bind_base, 'schedules': schedules_job, 'train': train_job, 'preflight-inputs': prepare_preflight, 'preflight': preflight_job}[a.action](a)

def configure(p):
    p.add_argument('--action', choices=('bind-base', 'schedules', 'train', 'preflight-inputs', 'preflight'), required=True)
    p.add_argument('--arm', choices=plan.ARMS); p.add_argument('--seed', type=int, choices=plan.SEEDS); p.add_argument('--stage', choices=plan.STAGES)
    p.add_argument('--schedule-root', type=Path); p.add_argument('--schedule-roots', type=Path, nargs='+')
    p.add_argument('--arms', nargs='+', choices=plan.ARMS)
    p.add_argument('--panel', action='append'); p.add_argument('--holdout', action='append'); p.add_argument('--superset', type=Path)
    p.add_argument('--stages', nargs='+', choices=plan.STAGES, help='stages to freeze (default all); hard MATH freezes separately')
    p.add_argument('--host-reservation-bytes', type=int)
    p.add_argument('--model', type=Path); p.add_argument('--receipt', type=Path); p.add_argument('--tokenizer')
    p.add_argument('--sdpo', type=Path); p.add_argument('--trainer-python', default=sys.executable); p.add_argument('--gpu', default='0,1,2,3,4,5,6,7')


if __name__ == '__main__':
    from kit.p3b_common import entry
    raise SystemExit(entry(job, configure, 'm3-train-{action}'))
