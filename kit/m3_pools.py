#!/usr/bin/env python3
"""Message-3 replay pools (registration v2): frozen instruction, rebuilt broad, FinQA earlier and ContractNLI replay.

Plain partner job. Inputs are send-time frozen and bound exactly as message-2 core binds them (round-4 ruling 2):
B's replay/ and FinQA target folders read in place (frozen manifest, manifest sha256 bound write-once), C's journal
likewise. The only read of a message-2 runtime product is the read-only copy of core's committed hard-MATH targets.
  finqa-targets CPU  stage-1 task targets + 1,280-row schedules from B's verified 27B FinQA training targets
                     (native seed-specific COMMON parquet order, as p3b_train.job consumes it)
  math-targets  GPU* stage-3 targets for the frozen 500-item Level-5 training pool (round-4 ruling 1): a read-only copy of
                     core's committed M5 targets-math snapshot when present, else the identical bounded 3D protocol in the
                     M3 namespace (*two GPUs only on the fallback path). Records which path was used. Never waits.
  contract      CPU  rebuild contract-replay.jsonl from the verified 27B ContractNLI journal
  broad         GPU  GSM8K/ARC base-success pools in CR's own namespace (frozen strict ARC prompt)
  freeze        CPU  instruction audit, FinQA earlier rows, cross-pool/panel exclusions, length audit, v2 minima, manifest
Our side, before send (never on the node):
  superset           write kit/resources/m3/sp-candidate-superset-v1.json (m3b2's sp-candidate-superset-v1 format)
Nothing here generates ContractNLI targets: a missing journal resumes through kit/c_targets.py.
Exclusions are the three registered prompt rules only (manager ruling: B's bare-ID collision rule is dropped).
"""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_common import (KIT, CorrectnessStop, NotReady, atomic, bind, files_manifest, frozen_files, require, s, t)
from kit import m3_schedule as plan
from collections import Counter
import json
import os
import time
from functools import wraps

M3 = KIT / 'resources/m3'
SUPERSET = M3 / 'sp-candidate-superset-v1.json'
CORE_TARGETS_TASK = 'M5'  # message-2 core task that produces targets-math (kit/m2_manifest.py)
FINQA_SCHEDULE_PROTOCOL = 'b-native-finqa-schedule.v1'
TARGETS_SOURCE = 'targets-math-source.json'  # beside, never inside, the copied/generated targets-math tree
SUPERSET_SCHEMA = 'sp-candidate-superset-v1'
ARC_BINDING = M3 / 'arc-prompt-binding.json'
FROZEN_ARC = KIT / 'resources/m2/arc-prompt-FROZEN.json'
BROAD_KEY = 'next-v4-broad-v1|'  # message-2's frozen candidate order, reused unchanged in CR's namespace
BROAD_LIMIT = 1024
NOTICES = dict(
    instruction='RLVR-IFeval, allenai/RLVR-IFeval @ 47c03c73621c4aab2b824b7818681117d662770e; ODC-BY; checker Apache-2.0 (open-instruct if_functions.py)',
    contract_nli='ContractNLI, Koreeda and Manning 2021; CC-BY-4.0 per TERMS; verified Qwen3.6-27B training demonstrations (labels verified, rationales not certified)')
# Exclusion order: earlier pools are references for later ones. Instruction is audit-only (never edited);
# ContractNLI rows share agreements across hypotheses by design, so it is never de-duplicated within itself.
ORDER = ('instruction', 'contract_nli', 'finqa', 'gsm8k', 'arc')
WITHIN = dict(instruction=False, contract_nli=False, finqa=True, gsm8k=True, arc=True)
CR_DOMAINS = ('instruction', 'gsm8k', 'arc', 'finqa', 'contract_nli')
REAL_PANELS = dict(finqa=1147, math_hard=100, hellaswag=100)
REQUIRED_PANELS = ('finqa', 'general-v1', 'hellaswag', 'math_hard', 'legal-panel', 'legal-test')


# ----------------------------------------------------------------------------- ARC binding

def arc_prompt():
    """The frozen strict ARC prompt, byte-bound to the reviewed FROZEN record merged from main."""
    binding = json.loads(ARC_BINDING.read_text())
    require(s.file_hash(FROZEN_ARC) == binding['frozen_sha256'], 'ARC FROZEN record differs from the M3-bound hash')
    record = json.loads(FROZEN_ARC.read_text())
    require(record['frozen_arc_prompt'] == binding['frozen_arc_prompt'], 'ARC FROZEN prompt differs from M3 binding')
    from kit.next_pools import ARC_PROMPT, POOL_PROTOCOL
    require(ARC_PROMPT == binding['frozen_arc_prompt'] and POOL_PROTOCOL == binding['pool_protocol'], 'next_pools ARC recipe differs from FROZEN')
    require(record['n'] == 128 and record['accepted_count'] >= binding['min_accepted'] and record['parse_count'] >= binding['min_parse'],
            '128-row ARC acceptance receipt not passing')
    return dict(binding, record_sha256=s.file_hash(FROZEN_ARC))


# ----------------------------------------------------------------------------- exclusions

def text_of(prompt):
    return '\n'.join(m['content'] for m in prompt) if isinstance(prompt, list) else prompt


class Matcher:
    """B's three registered rules on prompt text: normalised equality, word 5-shingle Jaccard >=.8 and
    normalised edit similarity >=.9 (kit/p3b_replay_data.near). IDs never trigger an exclusion."""

    def __init__(self, panels):
        from kit.p3b_replay_data import prepared
        self.prepared = prepared
        self.reference = []; self.exact = {}
        for row in panels: self.add(row, 'panel')

    def add(self, row, namespace):
        value = self.prepared(text_of(row['prompt']))
        entry = (f"{namespace}:{row['id']}", value)
        self.reference.append(entry); self.exact.setdefault(value[0], entry[0])
        return value

    def find(self, row, value=None):
        from kit.p3b_replay_data import near
        value = value or self.prepared(text_of(row['prompt']))
        if value[0] in self.exact: return self.exact[value[0]], 'normalised_equality'
        for key, other in self.reference:
            if near(value, other): return key, 'near_duplicate'
        return None, None


def exclusion_pass(pools, panels):
    """pools: {domain: rows}. Returns (kept, exclusions, audit_failures) in the registered ORDER."""
    from kit.p3b_replay_data import near
    matcher = Matcher(panels)
    kept = {}; exclusions = []; failures = []
    for domain in ORDER:
        if domain not in pools: continue
        accepted = []; local = []
        for row in pools[domain]:
            value = matcher.prepared(text_of(row['prompt']))
            hit, rule = matcher.find(row, value)
            if hit is None and WITHIN[domain]:
                for key, other in local:
                    if near(value, other): hit, rule = key, 'within_pool_near_duplicate'; break
            record = dict(domain=domain, id=str(row['id']), overlaps=hit, rule=rule, prompt_sha256=t.sha(text_of(row['prompt'])))
            if hit is not None:
                (failures if domain == 'instruction' else exclusions).append(record)
                continue
            accepted.append(row); local.append((f'{domain}:{row["id"]}', value))
        # Pool rows become references for later domains only after the whole domain is processed.
        for row in accepted: matcher.add(row, domain)
        kept[domain] = accepted
    return kept, exclusions, failures


def read_panels(specs, standin):
    """specs: {name: path}; every row needs id/prompt. FinQA goes through next_pools' frozen check.
    Panels and holdouts are frozen before preparation; later runs verify their identity (bind), never edit them."""
    from kit.next_pools import finqa_exclusions
    rows = []; identities = {}
    if not standin: require(set(REQUIRED_PANELS) <= set(specs), 'missing exclusion panels: ' + ','.join(sorted(set(REQUIRED_PANELS) - set(specs))))
    for name, path in sorted(specs.items()):
        values = t.read_jsonl(path)
        if name == 'finqa': values = finqa_exclusions(values, standin)
        require(all('id' in r and 'prompt' in r for r in values), f'panel {name} rows need id/prompt')
        if not standin and name in REAL_PANELS: require(len(values) == REAL_PANELS[name], f'{name} must contain all {REAL_PANELS[name]} items')
        identities[name] = dict(sha256=s.file_hash(path), rows=len(values))
        rows += [dict(id=f'{name}:{r["id"]}', prompt=r['prompt']) for r in values]
    return rows, identities


def parse_specs(values):
    out = {}
    for value in values or ():
        name, _, path = value.partition('=')
        require(name and path and name not in out, 'panel/holdout must be NAME=PATH, unique')
        out[name] = Path(path)
    return out


# ----------------------------------------------------------------------------- instruction

def instruction_pool(replay_root):
    """B's checked RLVR-IFeval replay.jsonl, exactly; refresher-64 stays reserved."""
    manifest = frozen_files(replay_root)
    for name in ('accepted.jsonl', 'refresher-64.jsonl', 'replay.jsonl'): require(name in manifest['files'], 'B replay freeze lacks ' + name)
    accepted = t.read_jsonl(replay_root / 'accepted.jsonl'); refresher = t.read_jsonl(replay_root / 'refresher-64.jsonl')
    replay = t.read_jsonl(replay_root / 'replay.jsonl')
    require(refresher == accepted[:64] and replay == accepted[64:], 'B replay split differs from accepted[:64]/[64:]')
    require(not {str(r['id']) for r in refresher} & {str(r['id']) for r in replay}, 'refresher-64 overlaps replay')
    rows = [dict(id=str(r['id']), prompt=r['prompt'], response=r['response'], domain='instruction',
                 licence_notice=NOTICES['instruction']) for r in replay]
    return rows, dict(manifest=s.file_hash(replay_root / 'manifest.json'), replay=s.file_hash(replay_root / 'replay.jsonl'),
                      technical_synthetic=manifest.get('technical_synthetic', False))


# ----------------------------------------------------------------------------- send-time frozen archives

def fingerprinted(path, name):
    """Size + sha256 of a return-1 archive against the reviewed return-1 decisions (as kit/m2_return1.archives)."""
    from kit.m2_return1 import decisions
    expected = decisions()['provenance']['input_fingerprints'][name]
    path = Path(path)
    require(path.is_file() and path.stat().st_size == expected['bytes'] and s.file_hash(path) == expected['sha256'],
            'return-1 archive fingerprint differs: ' + name)
    return dict(name=name, bytes=expected['bytes'], sha256=expected['sha256'])


def unpack(files, scratch):
    """Write regular archive members (already safety-checked by kit.m2_return1.files) under a scratch root."""
    for name, raw in files.items():
        path = Path(name); require(not path.is_absolute() and '..' not in path.parts, 'unsafe archive member')
        target = scratch / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
    return scratch


def task_root(root, rows, receipt, standin, **extra):
    """pool.jsonl + one 40x32 seed-shuffled schedule per seed (kit.p3b_common.cycle, as c_targets/m2_targets do)."""
    from kit.p3b_common import cycle
    root.mkdir(parents=True, exist_ok=True)
    require(bool(rows), 'zero task targets; training input undefined')
    t.write_jsonl(root / 'pool.jsonl', rows); names = ['pool.jsonl']
    for seed in plan.SEEDS:
        name = f'schedule-{seed}.jsonl'; t.write_jsonl(root / name, cycle(rows, seed, plan.UPDATES * plan.TASK_ROWS)); names.append(name)
    atomic(root / 'receipt.json', receipt); names.append('receipt.json')
    atomic(root / 'manifest.json', files_manifest(root, names, technical_synthetic=standin, valid=len(rows), **extra))


def finqa_rows(folders, standin, tokenizer):
    """B's verified 27B FinQA training targets exactly as message 2 reads them (next_pools.earlier_rows: manifest and
    train_sha256 checks, kit-v4-sft-data.v1 COMMON schema, 27B teacher, original verifier), with gold for the reward."""
    from kit.next_pools import earlier_rows
    import pyarrow.parquet as pq
    rows, receipts = earlier_rows(folders, standin, tokenizer)
    gold = {}
    for folder in folders:
        for r in pq.read_table(Path(folder) / 'train.parquet').to_pylist():
            gold[str(r['extra_info']['index'])] = r['reward_model']['ground_truth']
    return [dict(id=r['id'], prompt=r['prompt'], response=r['response'], gold=gold[r['id']], task='finqa') for r in rows], receipts


def finqa_replay(a):
    """Materialize reverified, excluded earlier FinQA before broad candidate selection."""
    from kit.m3_train import tokenizer_for
    root = a.out / 'finqa-replay'
    if (root / 'manifest.json').exists(): frozen_files(root); return
    require(a.finqa_targets, 'bound B FinQA target folders required')
    rows, receipts = finqa_rows(a.finqa_targets, a.standin, tokenizer_for(a))
    panels, identities = read_panels(parse_specs(a.panel) | parse_specs(a.holdout), a.standin)
    kept, excluded, _ = exclusion_pass({'finqa': rows}, panels)
    root.mkdir(parents=True, exist_ok=True)
    t.write_jsonl(root / 'finqa.jsonl', kept['finqa'])
    t.write_jsonl(root / 'exclusions.jsonl', excluded)
    atomic(root / 'inputs.json', dict(receipts=receipts, panels=identities, technical_synthetic=a.standin))
    atomic(root / 'manifest.json', files_manifest(root, ['finqa.jsonl', 'exclusions.jsonl', 'inputs.json']))


def finqa_task_schedules(folders, standin, tokenizer):
    """B already scheduled these parquets. Preserve their seed, order and repeated exposures verbatim.

    v4_campaign_ops.schedule shuffled the original common corpus once; v4_sft wrote the ordered
    exposures. p3b_train.job reads that parquet directly and mixed() replaces replay positions.
    next_pools.earlier_rows remains the verifier/unique lookup, never a new task sampler.
    """
    import pyarrow.parquet as pq
    from kit.m3_manifest import validate_input
    schedules = {}; sources = {}; common = {}
    for folder in map(Path, folders):
        manifest = json.loads((folder / 'manifest.json').read_text())
        seed = manifest.get('seed')
        require(type(seed) is int, 'B FinQA task input must declare its native seed')
        if seed not in plan.SEEDS: continue  # other historical folders may still serve earlier replay
        require(seed not in schedules, 'duplicate B FinQA task schedule seed')
        if not standin: validate_input('finqa_targets_0', folder)
        require(manifest.get('schema') == 'kit-v4-sft-data.v1' and manifest.get('arm') == 'COMMON', 'B FinQA task input must be a native COMMON schedule')
        require(manifest.get('technical_synthetic', False) == standin, 'B FinQA task schedule type mismatch')
        for part in ('train', 'val'):
            require(s.file_hash(folder / (part + '.parquet')) == manifest.get(part + '_sha256'), 'B FinQA task parquet hash differs: ' + part)
        raw = pq.read_table(folder / 'train.parquet').to_pylist()
        require(len(raw) == manifest.get('rows') == plan.UPDATES * plan.TASK_ROWS and manifest.get('schedule_steps', 40) == 40,
                'B FinQA task schedule must already contain 40x32 exposures')
        require([r['extra_info']['index'] for r in raw] == manifest.get('ids'), 'B FinQA native task schedule order changed')
        verified, _ = finqa_rows([folder], standin, tokenizer)
        lookup = {r['id']: r for r in verified}
        ordered = [lookup[str(r['extra_info']['index'])] for r in raw]
        for row in ordered:
            require(row['id'] not in common or common[row['id']] == row, 'B FinQA target differs across native seed schedules')
            common[row['id']] = row
        schedules[seed] = ordered
        sources[str(seed)] = dict(manifest_sha256=s.file_hash(folder / 'manifest.json'), train_sha256=manifest['train_sha256'],
                                  val_sha256=manifest['val_sha256'], ids_sha256=s.digest([r['id'] for r in ordered]),
                                  schedule_sha256=manifest.get('schedule_sha256'), exposures=len(ordered),
                                  first_update_task_ids=[r['id'] for i, r in enumerate(ordered[:32], 1) if i not in plan.GENERIC_POSITIONS])
    require(set(schedules) == set(plan.SEEDS), 'B FinQA native schedules required for each of seeds 101-103; no pooled/cycled fallback')
    return schedules, dict(protocol=FINQA_SCHEDULE_PROTOCOL, sources=sources, technical_synthetic=standin)


def check_finqa_root(root):
    """Refuse an obsolete union/shuffle task freeze, including on a resumed scientific stage."""
    manifest = frozen_files(root)
    require(manifest.get('schedule_protocol') == FINQA_SCHEDULE_PROTOCOL, 'obsolete FinQA task schedule; prepare native B schedules in a fresh M3 root')
    receipt = json.loads((root / 'receipt.json').read_text())
    require(receipt.get('protocol') == FINQA_SCHEDULE_PROTOCOL and set(receipt.get('sources', {})) == set(map(str, plan.SEEDS)),
            'FinQA native source evidence missing')
    for seed in plan.SEEDS:
        rows = t.read_jsonl(root / f'schedule-{seed}.jsonl')
        require(len(rows) == plan.UPDATES * plan.TASK_ROWS and s.digest([r['id'] for r in rows]) == receipt['sources'][str(seed)]['ids_sha256'],
                'FinQA task exposures differ from B native order')
    return receipt


def finqa_targets(a):
    """Copy the verified per-seed B task consumption; never shuffle/de-duplicate/cycle the task schedule."""
    root = a.out / 'targets-finqa'
    require(bool(a.finqa_targets), 'B FinQA native scheduled target folders required')
    from kit.m3_train import tokenizer_for
    schedules, receipt = finqa_task_schedules(a.finqa_targets, a.standin, tokenizer_for(a))
    if (root / 'manifest.json').exists():
        require(check_finqa_root(root) == receipt, 'B FinQA native task source binding changed')
        print('SKIP: frozen native B FinQA schedules'); return
    root.mkdir(parents=True, exist_ok=True)
    names = []
    for seed in plan.SEEDS:
        name = f'schedule-{seed}.jsonl'; t.write_jsonl(root / name, schedules[seed]); names.append(name)
    # This file is an exposure inventory, not a source to sample from.
    t.write_jsonl(root / 'pool.jsonl', [r for seed in plan.SEEDS for r in schedules[seed]])
    atomic(root / 'receipt.json', receipt); names += ['pool.jsonl', 'receipt.json']
    atomic(root / 'manifest.json', files_manifest(root, names, technical_synthetic=a.standin, task='finqa',
           schedule_protocol=FINQA_SCHEDULE_PROTOCOL, exposures_per_seed=1280))
    check_finqa_root(root)
    print('Frozen three native B FinQA 40x32 schedules; task order and replacement preserved')


def check_math_root(root, standin):
    """Registered hard-MATH targets: the frozen 500-item Level-5 pool only, never the screen or evaluation panel."""
    from kit.m2_common import M, hard_data
    manifest = frozen_files(root)
    inputs = json.loads((root / 'inputs.json').read_text())
    require(inputs.get('pool') == s.file_hash(M / 'train_pool.jsonl'), 'hard-MATH targets are not bound to the frozen 500-item pool')
    require(manifest.get('population') == 500 and manifest.get('technical_synthetic', False) is standin, 'hard-MATH target population/type differs')
    data = hard_data(); pool = {r['id'] for r in data['train_pool.jsonl']}; held = {r['id'] for r in data['eval.jsonl']}
    rows = t.read_jsonl(root / 'pool.jsonl')
    require(bool(rows) and {r['id'] for r in rows} <= pool and not {r['id'] for r in rows} & held, 'hard-MATH targets outside the training pool')
    for seed in plan.SEEDS:
        require(f'schedule-{seed}.jsonl' in manifest['files'] and len(t.read_jsonl(root / f'schedule-{seed}.jsonl')) == plan.UPDATES * plan.TASK_ROWS,
                f'hard-MATH schedule for seed {seed} missing or not 40x32')
    return manifest


def core_targets(durable, destination, task=CORE_TARGETS_TASK):
    """Read-only verified target copy; no core lease or retention change.
    None when core has not durably committed it yet."""
    import shutil
    import tempfile
    from kit import m2_snapshot as snapshots
    try: pointer = snapshots.read_pointer(durable, task, completed=True)
    except snapshots.Pending: return None
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='m3-core-targets-', dir=destination.parent) as scratch:
        tree = Path(scratch) / 'tree'
        from kit.combined_snapshot import restore_targets
        doc, pointer = restore_targets(durable, task, tree)
        require('targets-math/manifest.json' in doc['files'], 'core snapshot lacks committed targets-math')
        shutil.copytree(tree / 'targets-math', Path(scratch) / 'targets-math')
        frozen_files(Path(scratch) / 'targets-math')
        (Path(scratch) / 'targets-math').replace(destination)
    return dict(path='core-copy', task=task, snapshot_sha256=pointer['sha256'], snapshot_manifest_sha256=pointer['manifest_sha256'],
                targets_manifest_sha256=s.file_hash(destination / 'manifest.json'))


def costed_preparation(kind, ranks):
    def decorate(fn):
        @wraps(fn)
        def wrapped(a, *args, **kwargs):
            start = time.time(); complete = False
            try:
                result = fn(a, *args, **kwargs); complete = True; return result
            finally:
                end = time.time()
                atomic(a.out / 'preparation-costs' / (str(time.time_ns()) + '.json'),
                       dict(kind=kind, started_at=start, ended_at=end, completed=complete,
                            gpu_hours=0 if a.standin else ranks*(end-start)/3600, technical_synthetic=a.standin))
        return wrapped
    return decorate


@costed_preparation('m3_math_targets', 2)
def generate_math_targets(a, root):
    """Fallback: the identical registered protocol in the M3 namespace (kit/m2_targets.py without message 2's binding):
    500 frozen items, package B's 3D bounded journal (<=4 attempts, seeds 27101-27104, temperatures 0/.3/.6/.8, top-p 1,
    thinking off, 2,560 output / 4,096 input tokens, frozen rational verifier, 2,048 EOS-inclusive student limit)."""
    import time
    from kit.m2_common import M, hard_data, protocol
    from kit.p3b_common import identity
    items = hard_data()['train_pool.jsonl']
    observed = dict(pool=s.file_hash(M / 'train_pool.jsonl'), protocol=protocol(), technical_synthetic=a.standin, namespace='m3')
    if not a.standin:
        observed.update(teacher=identity(a.model, 'teacher'), tokenizer=identity(a.tokenizer, 'student'))
        require(len(set(a.gpu.split(','))) == 2, '27B targets use two GPUs')
    bind(root / 'inputs.json', observed)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, V4_PHASE0_INFERENCE='1')
    started = time.monotonic()
    if a.standin:
        from kit.c_sim import Tokenizer
        class Generator:
            def generate_many(self, requests):
                lookup = {r['id']: r for r in items}
                return [dict(text='Working through the rational calculation gives \\boxed{' + lookup[r['id']]['gold'] + '}.',
                             finish_reason='stop', teacher_tokens=32) for r in requests]
        tokenizer, generator = Tokenizer(), Generator()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        generator = t.VLLMGenerator(a.model, 2560, tensor_parallel_size=2, allow_batch_fallback=False)
    s.run_phase(items, root / 'teacher', generator, tokenizer, synthetic=a.standin)
    demos = t.read_jsonl(root / 'teacher/demonstrations.jsonl')
    if not demos: raise NotReady('zero accepted targets; training input undefined')
    lookup = {r['id']: r for r in items}
    rows = [dict(id=d['id'], prompt=t.student_messages(lookup[d['id']]), response=d['text'], gold=lookup[d['id']]['gold'], task='math_screen') for d in demos]
    atomic(root / 'cost.json', dict(kind='m3_math_targets', gpu_hours=0 if a.standin else 2 * (time.monotonic() - started) / 3600, technical_synthetic=a.standin))
    from kit.p3b_common import cycle
    t.write_jsonl(root / 'pool.jsonl', rows); names = ['inputs.json', 'pool.jsonl', 'cost.json']
    for seed in plan.SEEDS:
        name = f'schedule-{seed}.jsonl'; t.write_jsonl(root / name, cycle(rows, seed, plan.UPDATES * plan.TASK_ROWS)); names.append(name)
    atomic(root / 'manifest.json', files_manifest(root, names, valid=len(rows), population=500, coverage=len(rows) / 500,
                                                  protocol=protocol(), technical_synthetic=a.standin))
    return dict(path='m3-fallback', coverage=len(rows) / 500)


def math_targets(a):
    """Stage-3 targets (round-4 ruling 1). Primary: core's committed, hash-recorded targets-math, copied read-only. Fallback,
    when core has not committed when M3 stage 3 is ready: the identical protocol here. Never waits; records the path."""
    root = a.out / 'targets-math'; receipt = a.out / TARGETS_SOURCE
    if (root / 'manifest.json').exists():
        check_math_root(root, a.standin); require(receipt.exists(), 'hard-MATH targets lack their source record'); print('SKIP: frozen M3 hard-MATH targets'); return
    require(not root.exists(), 'partial M3 hard-MATH targets present; move them aside before retrying')
    source = core_targets(a.core_durable, root) if a.core_durable else None
    if source is None:
        root.mkdir(parents=True)
        source = generate_math_targets(a, root)
    check_math_root(root, a.standin)
    atomic(receipt, dict(source, decided_at=__import__('time').time(), core_durable_given=bool(a.core_durable), technical_synthetic=a.standin,
                         targets_manifest_sha256=s.file_hash(root / 'manifest.json')))
    print('Hard-MATH targets frozen via', source['path'])


# ----------------------------------------------------------------------------- SP candidate-prompt superset

def sp_prompt_hash(prompt):
    """m3b2's sp-candidate-superset-v1 rule (kit/sp_manifest.prompt_hash): NFKC, casefold, whitespace-collapsed sha256."""
    import hashlib
    import unicodedata
    if isinstance(prompt, list): prompt = '\n'.join(m.get('content', '') for m in prompt if isinstance(m, dict))
    text = ' '.join(unicodedata.normalize('NFKC', str(prompt)).casefold().split())
    return hashlib.sha256(text.encode()).hexdigest()


def superset(sources):
    """sources: {pool name: iterable of prompts}. Pool names are the M3 returned-pool file stems."""
    return dict(schema=SUPERSET_SCHEMA, pools={name: sorted({sp_prompt_hash(p) for p in prompts}) for name, prompts in sorted(sources.items())})


def offline_broad_sources():
    """Read only pinned local snapshots, verifying the exact parquet files before use."""
    from kit.next_hellaswag import download
    pins = json.loads((M3 / 'candidate-sources.json').read_text())
    loaded = {}
    for name, fragment in (('gsm8k', 'main/train-'), ('arc', 'ARC-Challenge/train-')):
        pin = pins[name]
        loaded[name] = download(pin['repository'], pin['revision'], fragment, pin['files'])
    return loaded


def superset_job(a):
    """Every prompt that could enter an M3 pool or task schedule, frozen before preparation (superset, not the pools)."""
    from kit import c_targets as ct
    from kit.m2_common import hard_data
    from kit.next_pools import candidates, GSM, ARC
    sources = {}
    import pyarrow.parquet as pq
    pins = json.loads((M3 / 'candidate-sources.json').read_text())
    if a.replay is not None:
        replay_root = a.replay; frozen_files(replay_root)
        names = [n for n in ('candidates.jsonl', 'accepted.jsonl') if (replay_root / n).exists()]
        require(bool(names), 'B replay candidates/accepted rows required')
        sources['instruction'] = [r['prompt'] for n in names for r in t.read_jsonl(replay_root / n)]
    else:
        require(a.instruction_source and s.file_hash(a.instruction_source) == pins['instruction']['sha256'], 'pinned offline instruction source required')
        sources['instruction'] = [r.get('prompt', r.get('messages')) for r in pq.read_table(a.instruction_source).to_pylist()]
    if a.standin:
        require(a.gsm_rows and a.arc_rows, 'fixtures required'); gsm, arc = t.read_jsonl(a.gsm_rows), t.read_jsonl(a.arc_rows)
        legal = ct.synthetic_rows()['train']
    else:
        loaded = offline_broad_sources()
        gsm, _ = loaded['gsm8k']; arc, _ = loaded['arc']
        require(a.source is not None, 'ContractNLI source directory required'); legal = ct.source_rows(a.source, False)['train']
    rows = candidates(gsm, arc)
    for source in ('gsm8k', 'arc'): sources[source] = [r['prompt'] for r in rows if r['source'] == source]
    sources['contract_nli'] = sources['task-contract_nli'] = [r['prompt'] for r in legal]
    finqa = []
    import pyarrow.parquet as pq
    for folder in a.finqa_targets or ():
        for r in pq.read_table(folder / 'train.parquet').to_pylist():
            prompt = r['prompt']; finqa.append(json.loads(prompt) if isinstance(prompt, str) and prompt.startswith('[') else prompt)
    if not finqa and not a.standin:
        from kit.beds.finqa import render_prompt
        require(a.finqa_source and s.file_hash(a.finqa_source) == pins['finqa']['sha256'], 'pinned offline FinQA training source required')
        finqa = [render_prompt(r) for r in json.loads(a.finqa_source.read_text())]
    require(a.standin or bool(finqa), 'FinQA candidate training prompts required')
    sources['finqa'] = sources['task-finqa'] = finqa
    sources['task-math_hard'] = [t.student_messages(r) for r in hard_data()['train_pool.jsonl']]
    doc = superset(sources)
    doc['sources'] = dict(instruction=pins['instruction'] if a.replay is None else {'manifest_sha256': s.file_hash(a.replay / 'manifest.json')},
                          finqa=pins['finqa'] if a.finqa_source else {'bound_target_folders': True},
                          broad_sources={k: pins[k] for k in ('gsm8k', 'arc')}, hard_math_pool_sha256=s.file_hash(__import__('kit.m2_common', fromlist=['M']).M / 'train_pool.jsonl'))
    output = a.output or SUPERSET
    Path(output).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    print({k: len(v) for k, v in doc['pools'].items()})


def verify_returned_superset(a):
    """CPU pre-send check of actual bound B scheduled parquets/replay and C task prompts; never modifies inputs."""
    from kit.m3_manifest import validate_input
    report = Path(a.output) if a.output else a.out / 'returned-superset-verification.json'
    inputs = {}; pools = {}; finqa = []; value = None
    try:
        if not a.finqa_targets or not a.replay or not a.targets:
            raise NotReady('provide B seed-specific FinQA target folders, B frozen replay directory and C frozen target journal')
        import pyarrow.parquet as pq
        seen = set()
        for i, folder in enumerate(map(Path, a.finqa_targets)):
            doc = validate_input('finqa_targets_' + str(i), folder)
            seen.add(doc.get('seed'))
            finqa += [dict(prompt=json.loads(r['prompt']) if isinstance(r['prompt'], str) and r['prompt'].startswith('[') else r['prompt'])
                      for r in pq.read_table(folder / 'train.parquet').to_pylist()]
            inputs['finqa_' + str(i)] = dict(manifest_sha256=s.file_hash(folder / 'manifest.json'), train_sha256=doc['train_sha256'])
        require(set(plan.SEEDS) == seen and len(a.finqa_targets) == len(plan.SEEDS), 'returned FinQA schedules must cover exactly seeds 101-103')
        replay = frozen_files(a.replay)
        require('replay.jsonl' in replay['files'], 'B replay manifest lacks replay.jsonl')
        pools['instruction'] = t.read_jsonl(a.replay / 'replay.jsonl')
        pools['finqa'] = pools['task-finqa'] = finqa
        legal = frozen_files(a.targets); rows = []
        for seed in plan.SEEDS:
            name = f'schedule-{seed}.jsonl'; require(name in legal['files'], 'C target journal lacks ' + name)
            rows += t.read_jsonl(a.targets / name)
        pools['contract_nli'] = pools['task-contract_nli'] = rows
        inputs['replay'] = dict(manifest_sha256=s.file_hash(a.replay / 'manifest.json'), replay_sha256=replay['files']['replay.jsonl'])
        inputs['contract'] = dict(manifest_sha256=s.file_hash(a.targets / 'manifest.json'))
        value = verify_superset(pools, a.superset or SUPERSET)
        atomic(report, dict(value, inputs=inputs))
        if value['status'] == 'PENDING': raise NotReady('published superset unavailable')
        if value['status'] != 'VERIFIED': raise CorrectnessStop('returned prompt universe mismatch; block affected training and send')
        return value
    except (FileNotFoundError, NotReady) as error:
        atomic(report, dict(status='PENDING', inputs=inputs, reason='actual returned input files or published superset unavailable'))
        raise NotReady('returned B/C inputs required for CPU superset verification') from error
    except Exception as error:
        if value is None:
            atomic(report, dict(status='UNVERIFIED', inputs=inputs, error_type=type(error).__name__,
                                reason='returned input integrity or schema validation failed'))
        raise


def verify_superset(pools, path=SUPERSET):
    """Each frozen M3 pool must lie inside the published superset (verification only; nothing is removed)."""
    if not Path(path).exists(): return dict(status='PENDING', reason='superset not published')
    doc = json.loads(Path(path).read_text()); require(doc.get('schema') == SUPERSET_SCHEMA, 'unknown superset schema')
    result = {}
    for name, rows in pools.items():
        allowed = set(doc['pools'].get(name, ()))
        outside = sorted({sp_prompt_hash(r['prompt']) for r in rows} - allowed)
        result[name] = dict(status='VERIFIED' if allowed and not outside else 'UNVERIFIED', rows=len(rows), outside=len(outside))
    return dict(status='VERIFIED' if all(v['status'] == 'VERIFIED' for v in result.values()) else 'UNVERIFIED',
                superset_sha256=s.file_hash(path), pools=result)


# ----------------------------------------------------------------------------- ContractNLI replay

def contract_replay(targets_root, splits, evaluation, tokenizer, standin):
    """Recheck every accepted 27B training demonstration through c_targets.select_pool/verify.

    targets_root is a completed kit/c_targets.py root (its frozen <=1,024-candidate, <=4-attempt gold-blind
    journal). Returns rows and receipt; raises NotReady when the journal is incomplete (resume with c_targets).
    """
    from kit import c_targets as ct
    if not (targets_root / 'manifest.json').exists():
        raise NotReady('ContractNLI journal incomplete; resume kit/c_targets.py on ' + str(targets_root) + ' (frozen schedule, no new candidates)')
    manifest = frozen_files(targets_root)
    require(manifest.get('technical_synthetic', False) is standin, 'ContractNLI journal type mismatch')
    require(manifest.get('response_cap_including_eos') == 2048 and [list(x) for x in manifest.get('attempts', [])] == [list(x) for x in t.SCHEDULE],
            'ContractNLI journal was not produced under the frozen 4-attempt schedule')
    frozen_files(targets_root / 'prepared')
    items = t.read_jsonl(targets_root / 'prepared/train.jsonl')
    require(len(items) <= 1024, 'more than 1,024 frozen candidates')
    # Membership must be reproducible from the source under the same rules (cross-split/heldout groups).
    selected, _ = ct.select_pool(splits, evaluation, tokenizer)
    require([r['id'] for r in selected] == [r['id'] for r in items], 'ContractNLI candidate membership differs from select_pool')
    require(all(r['source_split'] == 'train' for r in items), 'non-training ContractNLI candidate')
    held = {r.get('group') for r in evaluation if r.get('group')}
    other = {r['group'] for split, rows in splits.items() if split != 'train' for r in rows}
    lookup = {r['id']: r for r in items}
    accepted = t.read_jsonl(targets_root / 'teacher/accepted.jsonl')
    rows = []; receipts = []
    for record in accepted:
        item = lookup.get(record['id']); require(item is not None, 'accepted demonstration outside frozen candidates')
        require(item['group'] not in held and item['group'] not in other, 'dev/test/panel agreement in replay')
        attempt = int(record['attempt']); require(1 <= attempt <= len(t.SCHEDULE), 'attempt outside the frozen schedule')
        raws = []
        for k in range(1, attempt + 1):
            path = targets_root / 'teacher/raw' / f"{t.sha(item['id'])}-{k}.json"
            require(path.exists(), 'missing journal attempt ' + str(k) + ' for ' + item['id'])
            raw = json.loads(path.read_text()); seed, temperature = t.SCHEDULE[k - 1]
            request = dict(id=item['id'], attempt=k, seed=seed, temperature=temperature, top_p=1, messages=ct.messages(item))
            require(raw['request'] == request, 'journal request is not the frozen gold-blind request')
            checked = ct.verify(item, raw, tokenizer)
            require(checked['verified'] is (k == attempt), 'journal acceptance is not the first verified attempt for ' + item['id'])
            raws.append((path, raw, checked))
        path, raw, checked = raws[-1]
        require(checked['text'] == record['text'], 'accepted text differs from journal')
        pids, _ = t.tokenize_student(tokenizer, [dict(role='user', content=item['prompt'])], checked['text'])
        rows.append(dict(id=item['id'], prompt=[dict(role='user', content=item['prompt'])], response=checked['text'], gold=item['gold'],
                         group=item['group'], document_id=item.get('document_id'), hypothesis_id=item.get('hypothesis_id'),
                         domain='contract_nli', attempt=attempt, raw_sha256=s.file_hash(path), input_tokens=len(pids),
                         response_tokens_including_eos=checked['response_tokens_including_eos'], licence_notice=NOTICES['contract_nli']))
        receipts.append(dict(id=item['id'], attempts=[dict(attempt=k + 1, raw_sha256=s.file_hash(p), verified=c['verified'], reasons=c['reasons'])
                                                     for k, (p, _, c) in enumerate(raws)]))
    rows.sort(key=lambda r: r['id'])
    return rows, dict(journal_manifest=s.file_hash(targets_root / 'manifest.json'), candidates=len(items), accepted=len(rows),
                      receipts=receipts, document_counts=document_counts(rows),
                      disclosure='Labels verified by the strict terminal-line verifier; rationale truth not certified. Stage-2 training rows are replayed intentionally.')


def document_counts(rows):
    """Per-agreement repeat counts (up to 17 hypotheses share one agreement by design; never de-duplicated)."""
    counts = Counter(str(r.get('document_id') or r.get('group')) for r in rows)
    return dict(documents=len(counts), rows=len(rows), max_per_document=max(counts.values()) if counts else 0,
                distribution={str(k): v for k, v in sorted(Counter(counts.values()).items())}, per_document=dict(sorted(counts.items())))


def contract_job(a):
    from kit import c_targets as ct
    root = a.out / 'pools/contract'
    if a.standin:
        from kit.c_sim import Tokenizer
        tokenizer = Tokenizer(); splits = ct.synthetic_rows()
    else:
        require(a.source and a.tokenizer, 'offline ContractNLI source and pinned tokenizer required')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        splits = ct.source_rows(a.source, False)
    evaluation = ct.evaluation_items(a)
    rows, receipt = contract_replay(a.targets, splits, evaluation, tokenizer, a.standin)
    bind(root / 'inputs.json', dict(journal=receipt['journal_manifest'], evaluation=s.digest([dict(id=r['id'], prompt=r['prompt']) for r in evaluation]),
                                    technical_synthetic=a.standin))
    t.write_jsonl(root / 'contract-replay.jsonl', rows); atomic(root / 'receipts.json', receipt)
    atomic(root / 'manifest.json', files_manifest(root, ['inputs.json', 'contract-replay.jsonl', 'receipts.json'], technical_synthetic=a.standin, rows=len(rows)))
    print('ContractNLI replay rows:', len(rows), 'of', receipt['candidates'], 'candidates;', receipt['document_counts']['documents'], 'agreements')


# ----------------------------------------------------------------------------- broad (CR namespace)

def broad_candidates(gsm, arc, references):
    """message-2's frozen recipe: next_pools candidates, frozen ARC prompt, same hash order, limit 1,024/source."""
    from kit.next_pools import candidates
    from kit.p3b_replay_data import near
    raw = candidates(gsm, arc); prepared = []; excluded = []
    pools = {source: sorted((r for r in raw if r['source'] == source), key=lambda r: (t.sha(BROAD_KEY + source + '|' + r['order_id']), r['order_id']))
             for source in ('gsm8k', 'arc')}
    matcher = Matcher(references)
    for source in ('gsm8k', 'arc'):
        kept = []
        for row in pools[source]:
            if len(kept) >= BROAD_LIMIT: break
            value = matcher.prepared(row['prompt'])
            hit, rule = matcher.find(row, value)
            if hit is None:
                hit = next((f"{p['source']}:{p['id']}" for p, pv in prepared if near(value, pv)), None)
                rule = 'within_pool_near_duplicate' if hit else None
            if hit: excluded.append(dict(domain=source, id=row['id'], overlaps=hit, rule=rule, prompt_sha256=t.sha(row['prompt']))); continue
            kept.append(row); prepared.append((row, value))
    return [r for r, _ in prepared], excluded


@costed_preparation('m3_broad_preparation', 1)
def broad_job(a, generate=None):
    from kit.next_pools import correct, NOTICES as BROAD_NOTICES, GSM, ARC
    binding = arc_prompt(); root = a.out / 'pools/broad'
    if not a.standin:
        from kit.arc_prompt_acceptance import validate_frozen
        binding['acceptance'] = validate_frozen(FROZEN_ARC)
    panels, panel_ids = read_panels(parse_specs(a.panel) | parse_specs(a.holdout), a.standin)
    others = []
    instruction = a.instruction or (a.replay / 'replay.jsonl' if a.replay else None)
    require(instruction is not None, "B's frozen replay/ directory required")
    require(a.finqa_pool is not None and a.finqa_pool.is_file(), 'materialized excluded FinQA replay required before broad candidates')
    frozen_files(a.finqa_pool.parent)
    for name, path in (('instruction', instruction), ('contract_nli', a.contract), ('finqa', a.finqa_pool)):
        if path is not None: others += [dict(id=f"{name}:{r['id']}", prompt=r['prompt']) for r in t.read_jsonl(path)]
    if a.standin:
        require(a.gsm_rows and a.arc_rows, 'fixtures required')
        gsm, arc = t.read_jsonl(a.gsm_rows), t.read_jsonl(a.arc_rows); sources = dict(gsm8k=s.file_hash(a.gsm_rows), arc=s.file_hash(a.arc_rows))
    else:
        require(not a.gsm_rows and not a.arc_rows, 'real pools use the pinned sources')
        loaded = offline_broad_sources()
        gsm, gh = loaded['gsm8k']; arc, ah = loaded['arc']
        sources = dict(gsm8k=gh, arc=ah)
    prepared, excluded = broad_candidates(gsm, arc, panels + others)
    bind(root / 'inputs.json', dict(arc=binding, sources=sources, revisions=dict(gsm8k=GSM, arc=ARC), panels=panel_ids, others=s.digest(others),
                                    technical_synthetic=a.standin, candidates=s.digest([r['id'] for r in prepared])))
    if (root / 'manifest.json').exists(): frozen_files(root); print('SKIP broad'); return
    t.write_jsonl(root / 'candidates.jsonl', prepared); t.write_jsonl(root / 'exclusions.jsonl', excluded)
    tokenizer = None; started = time.monotonic(); generator = None
    if not a.standin:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, V4_PHASE0_INFERENCE='0')
    accepted = []; checks = []
    for row in prepared:
        path = root / 'raw' / (t.sha(row['id']) + '.json')
        if path.exists(): output = json.loads(path.read_text())
        else:
            if generate is not None: output = generate(row)
            elif a.standin: output = dict(text=row['fake_response'], finish_reason='stop')
            else:
                if generator is None: generator = t.VLLMGenerator(a.model, 2048, tensor_parallel_size=1, allow_batch_fallback=False)
                output = generator([dict(role='user', content=row['prompt'])], 1, seed=0, temperature=0, top_p=1)
            atomic(path, output)
        passed = output['finish_reason'] in ('stop', 'eos') and correct(row, output['text'])
        if passed and tokenizer is not None:
            validate_encoding(tokenizer, [dict(role='user', content=row['prompt'])], output['text'])
        checks.append(dict(id=row['id'], accepted=passed, response_sha256=s.file_hash(path)))
        if passed: accepted.append(dict(id=row['id'], prompt=row['prompt'], response=output['text'], domain=row['source'], gold=row['gold'],
                                        licence_notice=BROAD_NOTICES['gsm8k' if row['source'] == 'gsm8k' else 'arc']))
    for source in ('gsm8k', 'arc'): t.write_jsonl(root / (source + '.jsonl'), [r for r in accepted if r['domain'] == source])
    t.write_jsonl(root / 'checks.jsonl', checks)
    atomic(root / 'cost.json', dict(kind='m3_broad_preparation', gpu_hours=0 if a.standin else (time.monotonic() - started) / 3600, technical_synthetic=a.standin))
    names = ['inputs.json', 'candidates.jsonl', 'exclusions.jsonl', 'gsm8k.jsonl', 'arc.jsonl', 'checks.jsonl', 'cost.json'] + \
        [str(p.relative_to(root)) for p in sorted((root / 'raw').glob('*.json'))]
    atomic(root / 'manifest.json', files_manifest(root, names, technical_synthetic=a.standin, pool_protocol=binding['pool_protocol'],
                                                  arc_prompt=binding['frozen_arc_prompt'], namespace=plan.NAMESPACE))
    print('Broad accepted', len(accepted), 'of', len(prepared), 'candidates')


# ----------------------------------------------------------------------------- freeze

def availability(kept, failures, missing):
    """v2 no-repeat minima. C0/K: 160 unique shared instruction rows. G/CR (G is CR's control, so they move together):
    960 instruction rows (shared 160 + 800 G extras), GSM8K/ARC 80 each, earlier FinQA/legal 320 each."""
    count = {d: len(kept.get(d, [])) for d in CR_DOMAINS}
    need = dict(instruction=plan.MINIMA['instruction_g'], gsm8k=plan.MINIMA['gsm8k'], arc=plan.MINIMA['arc'],
                finqa=plan.MINIMA['finqa'], contract_nli=plan.MINIMA['contract_nli'])
    short_c0 = {} if count['instruction'] >= plan.MINIMA['instruction'] else {'instruction': [count['instruction'], plan.MINIMA['instruction']]}
    short_cr = {d: [count[d], n] for d, n in need.items() if count[d] < n}
    c0_ok = not failures and not short_c0 and 'instruction' not in missing
    cr_ok = c0_ok and not short_cr and not missing
    out = {}
    for arm, ok, short in (('C0', c0_ok, short_c0), ('K', c0_ok, short_c0), ('G', cr_ok, short_cr), ('CR', cr_ok, short_cr)):
        out[arm] = dict(status='READY' if ok else 'PENDING')
        if not ok: out[arm]['reason'] = dict(audit_failures=len(failures), short_pools=short, missing=sorted(missing))
    return out


def validate_encoding(tokenizer, prompt, response):
    """Use only the final M3 contract limits. Never truncate or silently filter."""
    from kit import m3_contract as contract
    from kit.sdft_objective import teacher_messages
    pids, tids = t.tokenize_student(tokenizer, prompt, response)
    teacher = tokenizer.apply_chat_template(teacher_messages(prompt, response), tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False)
    require(len(pids) <= contract.PROMPT_CAP, 'prompt exceeds common M3 cap; never truncated')
    require(tids and tids[-1] == tokenizer.eos_token_id and len(tids) <= contract.RESPONSE_CAP,
            'response exceeds common M3 cap or lacks EOS')
    require(len(teacher) <= contract.REPROMPT_CAP and len(teacher) + contract.RESPONSE_CAP <= contract.CONTEXT_CAP,
            'teacher context exceeds common M3 cap; never truncated')
    return pids, tids


def length_audit(pools, tokenizer):
    """Every admitted row must fit the common caps (prompt <= 10,240; response <= 2,048 incl. EOS). Fail loudly."""
    lengths = {}; over = []
    for domain, rows in pools.items():
        for r in rows:
            prompt = r['prompt'] if isinstance(r['prompt'], list) else [dict(role='user', content=r['prompt'])]
            pids, tids = validate_encoding(tokenizer, prompt, r['response'])
            lengths[str(r['id'])] = len(pids)
            if len(pids) > plan.PROMPT_CAP or len(tids) > plan.RESPONSE_CAP:
                over.append(dict(domain=domain, id=str(r['id']), prompt_tokens=len(pids), response_tokens_including_eos=len(tids)))
    if over: raise CorrectnessStop(f'{len(over)} admitted rows exceed the common caps (never filtered): ' + json.dumps(over[:5]))
    return lengths


def freeze_job(a):
    shared = getattr(a, 'shared_only', False)
    root = a.out / ('pools-shared' if shared else 'pools')
    if (root / 'manifest.json').exists(): frozen_files(root); print('SKIP: frozen M3 pools'); return
    panels, panel_ids = read_panels(parse_specs(a.panel) | parse_specs(a.holdout), a.standin)
    pools = {}; receipts = {}; missing = set()
    require(a.replay is not None, "B's frozen replay/ directory required (bound in place, as message 2 binds it)")
    pools['instruction'], receipts['instruction'] = instruction_pool(a.replay)
    if not shared and a.contract and (a.contract / 'manifest.json').exists():
        frozen_files(a.contract); pools['contract_nli'] = t.read_jsonl(a.contract / 'contract-replay.jsonl')
        receipts['contract_nli'] = dict(manifest=s.file_hash(a.contract / 'manifest.json'))
    else: missing.add('contract_nli')
    tokenizer = None
    if not a.standin:
        require(a.tokenizer, 'pinned tokenizer required for the length audit')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    if not shared and a.finqa_pool:
        pools['finqa'] = [dict(r, domain='finqa') for r in t.read_jsonl(a.finqa_pool)]
        if not a.standin: frozen_files(a.finqa_pool.parent)
        receipts['finqa'] = dict(rows=s.file_hash(a.finqa_pool), technical_synthetic=a.standin)
    elif not shared and a.finqa_targets:
        from kit.next_pools import earlier_rows
        earlier, proof = earlier_rows(a.finqa_targets, a.standin, tokenizer)
        pools['finqa'] = [dict(r, domain='finqa') for r in earlier]; receipts['finqa'] = proof
    else: missing.add('finqa')
    broad_root = a.out / 'pools/broad'
    if not shared and (broad_root / 'manifest.json').exists():
        frozen_files(broad_root); arc_prompt()
        pools.update({d: t.read_jsonl(broad_root / (d + '.jsonl')) for d in ('gsm8k', 'arc')})
        receipts['broad'] = dict(manifest=s.file_hash(broad_root / 'manifest.json'), source='m3-cr-namespace')
    else: missing.update(('gsm8k', 'arc'))
    kept, exclusions, failures = exclusion_pass(pools, panels)
    # Broad targets were excluded at candidate time; a new conflict cannot be silently repaired.
    if any(e['domain'] in ('gsm8k', 'arc') for e in exclusions): missing.update(('gsm8k', 'arc'))
    lengths = length_audit(kept, tokenizer) if tokenizer is not None else None
    status = availability(kept, failures, missing)
    verification = verify_superset(kept, getattr(a, 'superset', None) or SUPERSET)
    if not a.standin:
        for arm in plan.ARMS:
            domains = {d for stage in plan.STAGES for d in plan.domains_for(arm, stage)}
            if verification['status'] != 'VERIFIED' and any(verification.get('pools', {}).get(d, {}).get('status') != 'VERIFIED' for d in domains):
                status[arm] = dict(status='PENDING', reason='candidate superset missing or mismatched')
        require(bool(a.holdout), 'frozen PD/SP/core holdout evidence required')
    holdout_ids = {k: v for k, v in panel_ids.items() if k in parse_specs(a.holdout)}
    bind(root / 'inputs.json', dict(panels=panel_ids, holdouts=holdout_ids, receipts=receipts, technical_synthetic=a.standin,
                                    rules=dict(normalised_equality=True, shingle_jaccard=.8, shingle_words=5, edit_similarity=.9,
                                               bare_id_collision=False, order=ORDER, within=WITHIN)))
    for domain in CR_DOMAINS:
        if domain in kept: t.write_jsonl(root / (domain + '.jsonl'), kept[domain])
    t.write_jsonl(root / 'exclusions.jsonl', exclusions); t.write_jsonl(root / 'audit-failures.jsonl', failures)
    atomic(root / 'availability.json', status)
    atomic(root / 'licences.json', {d: sorted({r.get('licence_notice') or '' for r in kept.get(d, [])}) for d in CR_DOMAINS})
    atomic(root / 'contract-document-counts.json', document_counts(kept.get('contract_nli', [])))
    # Pools must lie inside the superset published before send (verification only; recorded, never edited).
    atomic(root / 'superset-verification.json', verification)
    names = ['inputs.json', 'exclusions.jsonl', 'audit-failures.jsonl', 'availability.json', 'licences.json', 'contract-document-counts.json',
             'superset-verification.json']
    if lengths is not None: atomic(root / 'lengths.json', lengths); names.append('lengths.json')
    names += [d + '.jsonl' for d in CR_DOMAINS if d in kept]
    atomic(root / 'manifest.json', files_manifest(root, names, technical_synthetic=a.standin, counts={d: len(v) for d, v in kept.items()},
                                                  minima=plan.MINIMA, target_sha256={d: s.digest([[r['prompt'], r['response']] for r in v]) for d, v in kept.items()}))
    print(json.dumps(status, sort_keys=True))
    if failures: raise CorrectnessStop('inherited instruction replay overlaps panels/holdouts; see audit-failures.jsonl (C0/K/G/CR PENDING)')


def pool_root(out, arm):
    shared = Path(out) / 'pools-shared'
    return shared if arm in ('C0', 'K') and (shared / 'manifest.json').exists() else Path(out) / 'pools'


def task_evidence(rows, stage, panels, superset_path=SUPERSET):
    """Task schedules are verified against the same prospectively frozen holdouts and candidate universe."""
    verification = verify_superset({'task-' + stage: rows}, superset_path)
    if verification['status'] != 'VERIFIED': raise NotReady('task candidate superset missing or mismatched: ' + stage)
    matcher = Matcher(panels)
    for row in rows:
        hit, _ = matcher.find(row)
        require(hit is None, 'task overlaps frozen panel/holdout: ' + str(row['id']))
    return verification


def load(out, arm):
    """Frozen pools for an arm; PENDING availability raises NotReady (never a reduced substitution)."""
    root = pool_root(out, arm); manifest = frozen_files(root)
    if not manifest.get('technical_synthetic'):
        evidence = json.loads((root / 'superset-verification.json').read_text())
        require(evidence.get('superset_sha256') == s.file_hash(SUPERSET), 'frozen candidate universe changed')
        for domain in {d for stage in plan.STAGES for d in plan.domains_for(arm, stage)}:
            if evidence.get('pools', {}).get(domain, {}).get('status') != 'VERIFIED': raise NotReady('candidate universe unverified: ' + domain)
    status = json.loads((root / 'availability.json').read_text())[arm]
    if status['status'] != 'READY': raise NotReady(f'{arm} PENDING: ' + json.dumps(status.get('reason'), sort_keys=True))
    domains = sorted({d for stage in plan.STAGES for d in plan.domains_for(arm, stage)})
    lengths = json.loads((root / 'lengths.json').read_text()) if 'lengths.json' in manifest['files'] else None
    return {d: t.read_jsonl(root / (d + '.jsonl')) for d in domains}, dict(manifest, lengths=lengths)


def job(a):
    if a.action == 'superset': return superset_job(a)
    if a.action == 'verify-returns': return verify_returned_superset(a)
    from kit.m3_manifest import require_m3_root, bound_input
    require_m3_root(a.out)
    if not a.standin:
        # Package B/C roots read in place, bound write-once by M3-bind-inputs exactly as m2_prepare binds them.
        if a.action == 'contract': bound_input(a.out, 'contract_targets', a.targets)
        if a.action in ('freeze', 'broad') and a.replay: bound_input(a.out, 'replay', a.replay)
        if a.action in ('freeze', 'finqa-targets', 'finqa-replay'):
            for index, folder in enumerate(a.finqa_targets or ()): bound_input(a.out, f'finqa_targets_{index}', folder)
    from contextlib import nullcontext
    from kit.m3_manifest import gpu_execution
    with gpu_execution(a.out) if not a.standin and a.action in ('broad', 'math-targets') else nullcontext():
        return {'finqa-replay': finqa_replay, 'finqa-targets': finqa_targets, 'math-targets': math_targets, 'contract': contract_job, 'broad': broad_job, 'freeze': freeze_job}[a.action](a)

def configure(p):
    p.add_argument('--action', choices=('finqa-replay', 'finqa-targets', 'math-targets', 'contract', 'broad', 'freeze', 'superset', 'verify-returns'), required=True)
    p.add_argument('--core-durable', type=Path, help="message-2 core durable root: read-only copy of its committed M5 targets-math")
    p.add_argument('--instruction-source', type=Path); p.add_argument('--finqa-source', type=Path)
    p.add_argument('--shared-only', action='store_true')
    p.add_argument('--output', type=Path); p.add_argument('--superset', type=Path)
    p.add_argument('--gpu', default='0'); p.add_argument('--model'); p.add_argument('--tokenizer')
    p.add_argument('--source', type=Path); p.add_argument('--targets', type=Path); p.add_argument('--finqa-panel', type=Path)
    p.add_argument('--panel', action='append', help='NAME=PATH exclusion panel (repeat)')
    p.add_argument('--holdout', action='append', help='NAME=PATH frozen PD/SP/core holdout (repeat)')
    p.add_argument('--replay', type=Path); p.add_argument('--contract', type=Path); p.add_argument('--instruction', type=Path)
    p.add_argument('--finqa-pool', type=Path); p.add_argument('--finqa-targets', type=Path, nargs='+')
    p.add_argument('--gsm-rows', type=Path); p.add_argument('--arc-rows', type=Path)


if __name__ == '__main__':
    from kit.p3b_common import entry
    raise SystemExit(entry(job, configure, 'm3-pools-{action}'))
