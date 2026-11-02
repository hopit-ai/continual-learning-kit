"""CPU tests for the message-3 K/CR (registration v2) data, schedule, manifest, scoring-assignment and reader protocol."""
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from util70_fixtures import prospective_build_authority

from kit import m3_schedule as plan
from kit import m3_manifest as manifest
from kit import m3_read as reader

ROOT = Path(__file__).resolve().parents[1]


def uid(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rows(domain, n, prefix=None):
    # Hex payloads keep prompts far apart under every registered near-duplicate rule.
    return [dict(id=f'{prefix or domain}-{i}', prompt=f'{domain} item {i} ' + uid(f'{domain}{i}'), response=f'answer {i}') for i in range(n)]


def task_rows(stage):
    pool = rows(stage, 100)
    return [pool[i % 100] for i in range(1280)]


def pools():
    """Exactly the v2 no-repeat minima for G/CR (instruction 960; GSM8K/ARC 80; FinQA/legal 320)."""
    return dict(instruction=rows('instruction', 960), gsm8k=rows('gsm8k', 80), arc=rows('arc', 80), finqa=rows('finqa', 320), contract_nli=rows('contract_nli', 320))


POOLS = pools()


# ----------------------------------------------------------------------------- schedules

@pytest.mark.parametrize('stage', plan.STAGES)
@pytest.mark.parametrize('arm', plan.ARMS)
def test_layout_counts_positions_and_no_filtered_draws(arm, stage):
    schedule, summary = plan.build(arm, 101, stage, task_rows(stage), POOLS)
    n = plan.rows_per_update(arm, stage)
    assert len(schedule) == 40 * n and summary['rounds'] == n // 8 and summary['skipped_draw_count'] == 0
    assert summary['prompt_cap'] == 10240 and summary['response_cap'] == 2048
    assert all(v['max_repeat'] == 1 for v in summary['repeats'].values())
    for update in range(1, 41):
        block = [r for r in schedule if r['update'] == update]
        assert [r['position'] for r in block] == list(range(1, n + 1))
        replay = [r for r in block if r['role'] == 'replay']
        if arm in ('C0', 'K'):
            assert [r['position'] for r in replay] == [8, 16, 24, 32] and {r['stream'] for r in replay} == {'shared'}
        elif arm == 'G':
            assert len(replay) == plan.G_GENERIC[stage] and {r['domain'] for r in replay} == {'instruction'}
            assert [r['stream'] for r in replay] == ['shared'] * 4 + ['extra'] * (plan.G_GENERIC[stage] - 4)
        else:
            assert {d: sum(r['domain'] == d for r in replay) for d in plan.CR_LAYOUT[stage]} == plan.CR_LAYOUT[stage]
        assert all(r['loss'] == ('D' if r['role'] == 'task' else plan.REPLAY_LOSS[arm]) for r in block)
        assert [(r['round'], r['rank']) for r in block] == [(j // 8, j % 8) for j in range(n)]


def test_exposure_totals_match_registration():
    totals = plan.exposure_totals()
    assert totals['C0', 'finqa'] == totals['K', 'math_hard'] == (1120, 160)
    assert [totals['CR', s] for s in plan.STAGES] == [totals['G', s] for s in plan.STAGES] == [(1280, 320), (1280, 640), (1280, 960)]


def test_shared_stream_is_message2_seed_only_cycling_and_identical_each_stage():
    from kit.p3b_common import cycle
    pool = POOLS['instruction']
    for seed in plan.SEEDS:
        expected = [r['id'] for r in cycle(pool, seed, len(pool))[:160]]
        assert [d['row']['id'] for d in plan.shared_stream(pool, seed)] == expected
        order = list(pool); random.Random(seed).shuffle(order)
        assert expected == [r['id'] for r in order[:160]]
        per_stage = [plan.shared_generic(plan.build('C0', seed, stage, task_rows(stage), POOLS)[0]) for stage in plan.STAGES]
        assert per_stage[0] == per_stage[1] == per_stage[2]
        assert [i for u in range(1, 41) for i in per_stage[0][u]] == expected
    with pytest.raises(plan.ScheduleError): plan.shared_stream(rows('instruction', 159), 101)


@pytest.mark.parametrize('seed', plan.SEEDS)
@pytest.mark.parametrize('stage', plan.STAGES)
def test_cross_arm_audit_shared_ids_and_disjoint_g_extras(seed, stage):
    task = task_rows(stage)
    built = {arm: plan.build(arm, seed, stage, task, POOLS) for arm in plan.ARMS}
    audit = plan.audit_arms({a: v[0] for a, v in built.items()}, {a: v[1] for a, v in built.items()})
    assert len(audit['shared_ids']) == 160 and len(audit['g_extra_ids']) == 40 * (plan.G_GENERIC[stage] - 4)
    assert not set(audit['shared_ids']) & set(audit['g_extra_ids'])
    eligible = [r for r in POOLS['instruction'] if r['id'] not in set(audit['shared_ids'])]
    order = sorted(eligible, key=lambda r: (uid(f"m3-cr-v2|{seed}|{stage}|generic-extra|{r['id']}"), r['id']))
    extras = [r['id'] for r in built['G'][0] if r.get('stream') == 'extra']
    assert extras == [r['id'] for r in order[:len(extras)]]
    broken = {a: v[0] for a, v in built.items()}; broken['G'] = [dict(r) for r in broken['G']]
    next(r for r in broken['G'] if r.get('stream') == 'shared' and r['update'] == 1)['id'] = 'other'
    with pytest.raises(plan.ScheduleError): plan.audit_arms(broken)
    sums = {a: dict(v[1]) for a, v in built.items()}; sums['K']['skipped_draw_count'] = 1
    with pytest.raises(plan.ScheduleError): plan.audit_arms({a: v[0] for a, v in built.items()}, sums)


def test_cr_domain_hash_order_and_no_within_stage_repeats():
    schedule, _ = plan.build('CR', 102, 'math_hard', task_rows('math_hard'), POOLS)
    legal = [r['id'] for r in schedule if r['domain'] == 'contract_nli' and r['role'] == 'replay']
    expected = sorted(POOLS['contract_nli'], key=lambda r: (uid(f"m3-cr-v2|102|math_hard|contract_nli|{r['id']}"), r['id']))
    assert legal == [r['id'] for r in expected] and len(set(legal)) == 320
    short = dict(POOLS, finqa=rows('finqa', 319))
    with pytest.raises(plan.ScheduleError): plan.build('CR', 102, 'contract_nli', task_rows('contract_nli'), short)
    with pytest.raises(plan.ScheduleError): plan.stream(rows('arc', 8) + rows('arc', 1), 101, 'finqa', 'arc', 5)
    summaries = [plan.build('CR', 101, s, task_rows(s), POOLS)[1] for s in plan.STAGES]
    assert plan.cross_stage_repeats(summaries)['CR|101|instruction'] == 160


def test_overlength_rows_fail_preparation_never_filtered():
    lengths = {r['id']: 100 for v in POOLS.values() for r in v}
    for r in POOLS['contract_nli']: lengths[r['id']] = 10240
    schedule, summary = plan.build('CR', 101, 'math_hard', task_rows('math_hard'), POOLS, lengths=lengths)
    assert summary['skipped_draw_count'] == 0 and sum(r['domain'] == 'contract_nli' for r in schedule) == 320
    lengths[POOLS['instruction'][3]['id']] = 10241
    for arm in plan.ARMS:
        with pytest.raises(plan.ScheduleError, match='common 10240'): plan.build(arm, 101, 'finqa', task_rows('finqa'), POOLS, lengths=lengths)
    assert all(plan.prompt_cap(a, s) == 10240 for a in plan.ARMS for s in plan.STAGES)


@pytest.mark.parametrize('world_size', (8, 2))
@pytest.mark.parametrize('stage', plan.STAGES)
@pytest.mark.parametrize('arm', plan.ARMS)
def test_reduction_equals_global_sum_over_32(arm, stage, world_size):
    n = plan.rows_per_update(arm, stage)
    losses = [random.Random(n + i).uniform(0, 3) for i in range(n)]
    assert plan.check_reduction(losses, world_size) == pytest.approx(sum(losses) / 32, rel=1e-12)
    assert plan.local_scale(world_size) == (0.25 if world_size == 8 else 1 / 16)
    rounds = n // world_size
    naive = sum(sum(l / rounds for j, l in enumerate(losses) if j % world_size == r) for r in range(world_size)) / world_size
    if n != 32: assert naive != pytest.approx(sum(losses) / 32)


def test_agreement_with_builder_a_contract():
    from kit import m3_contract as contract
    report = plan.contract_agreement()
    assert report['status'] == 'PASS' and not report['mismatched'], report
    assert (plan.PROMPT_CAP, plan.RESPONSE_CAP, plan.REPROMPT_CAP) == (contract.PROMPT_CAP, contract.RESPONSE_CAP, contract.REPROMPT_CAP)
    assert all(plan.prompt_cap(a, s) == contract.prompt_cap(a, s) == contract.PROMPT_CAP for a in plan.ARMS for s in plan.STAGES)
    for arm in plan.ARMS:
        for stage in plan.STAGES:
            shape = contract.geometry(arm, stage)
            assert shape['global_rows'] == plan.rows_per_update(arm, stage) and shape['local_loss_scale'] == plan.local_scale()
            n = shape['global_rows']; order = contract.rank_major_indices(n, 8); layout = plan.geometry(n)
            assert [layout[j]['rank'] for j in order] == sorted(layout[j]['rank'] for j in range(n))


def test_converted_rows_carry_builder_a_fields():
    from kit.m3_train import convert
    task = task_rows('contract_nli')
    schedule, _ = plan.build('CR', 101, 'contract_nli', task, POOLS)
    converted, tokens = convert(schedule, task, POOLS, None, 'CR', 'contract_nli')
    assert len(converted) == 40 * 48 and tokens['replay_response_including_eos'] > 0
    for row, item in zip(converted, schedule):
        x = row['extra_info']
        assert x['m3_replay'] is (item['role'] == 'replay') and x['m3_domain'] == item['domain']
        assert isinstance(x['m3_target_ids'], list) and isinstance(x['m3_eos_token_id'], int) and 'p3b_target_ids' not in x
    changed = [dict(r) for r in schedule]; changed[3]['target_sha256'] = '0' * 64
    with pytest.raises(Exception): convert(changed, task, POOLS, None, 'CR', 'contract_nli')
    from kit import m3_contract as contract
    assert {r['data_source'] for r, i in zip(converted, schedule) if i['role'] == 'replay'} == {contract.REPLAY_DATA_SOURCE}
    assert {r['data_source'] for r, i in zip(converted, schedule) if i['role'] == 'task'} == {contract.TASK_DATA_SOURCES['contract_nli']}


@pytest.mark.parametrize('arm,stage', [('C0', 'finqa'), ('CR', 'math_hard'), ('G', 'contract_nli')])
def test_trainer_command_leaves_entry_owned_keys_to_m3_entry_and_passes_preflight(tmp_path, monkeypatch, arm, stage):
    """Review items 2 and 11: explicit --preflight under the M3 root; no CSFTDataset, reward path or common cap here."""
    from kit import m3_contract as contract
    from kit.m3_train import command, ENTRY_OWNED, ENTRY_ENV, override_key
    for key in ENTRY_ENV: monkeypatch.setenv(key, 'test')  # hydra_arguments writes these; restored at teardown
    out = tmp_path / 'm3'; manifest.init_root(out)
    a = NS(out=out, sdpo=tmp_path / 'sdpo', seed=101, gpu='0,1,2,3,4,5,6,7', tokenizer='tok', trainer_python='py')
    geometry = contract.geometry(arm, stage)
    cmd, env = command(a, arm, stage, tmp_path / 'model', tmp_path / 'data', out / f'runs/{arm}-s101-{stage}', dict(model='/base', revision=contract.BASE_REVISION), geometry)
    assert cmd[cmd.index('--preflight') + 1] == str((out / contract.PREFLIGHT_TEMPLATE.format(arm=arm)).resolve())
    assert cmd[cmd.index('--receipt') + 1] == str((out / contract.RECEIPT_TEMPLATE.format(arm=arm, seed=101, stage=stage)).resolve())
    overrides = cmd[cmd.index('sdpo') + 1:]
    keys = [override_key(v) for v in overrides]
    assert len(keys) == len(set(keys)) and not set(keys) & ENTRY_OWNED
    assert not any('CSFTDataset' in v or 'c_reward' in v or 'p3b_reward' in v or 'custom_cls' in v for v in overrides)
    values = {override_key(v): v.split('=', 1)[1] for v in overrides}
    assert values['data.train_batch_size'] == values['actor_rollout_ref.actor.ppo_mini_batch_size'] == str(geometry['global_rows'])
    from kit import m3_entry
    final = m3_entry.hydra_arguments(dict(arm=arm, stage=stage, seed=101, profile='scientific', base_model='/base', base_revision=contract.BASE_REVISION,
                                          receipt='r', preflight='p', preflight_phase='uninterrupted', input_manifest=None, host_reservation_bytes=None), overrides)
    final_keys = [override_key(v) for v in final]
    assert len(final_keys) == len(set(final_keys)), 'every Hydra key is set exactly once after the entry adds its own'


def test_bind_base_shape_is_what_the_contract_verifies(tmp_path):
    """Review item 6: inputs/base.json carries revision + identity.model_file_hashes; the stage re-verifies the files."""
    from kit import m3_contract as contract
    from kit.m3_train import base_files_digest
    model = tmp_path / 'qwen'; model.mkdir()
    (model / 'config.json').write_text('{"a": 1}'); (model / 'model.safetensors').write_bytes(b'weights')
    files = {n: uid_bytes((model / n).read_bytes()) for n in ('config.json', 'model.safetensors')}
    out = tmp_path / 'm3'; manifest.init_root(out); (out / 'inputs').mkdir()
    record = dict(model=str(model), revision=contract.BASE_REVISION, identity=dict(model_file_hashes=files), technical_synthetic=False)
    (out / 'inputs/base.json').write_text(json.dumps(record))
    digest = base_files_digest(out)
    assert digest == contract.verify_base_files(model, out / 'inputs/base.json', contract.BASE_REVISION)
    (model / 'model.safetensors').write_bytes(b'other weights')
    with pytest.raises(ValueError): base_files_digest(out)


# ----------------------------------------------------------------------------- pools and exclusions

def test_arc_prompt_binding_matches_frozen_record_from_main():
    from kit import m3_pools
    binding = m3_pools.arc_prompt()
    assert binding['record_sha256'] == binding['frozen_sha256'] == '6ce5528c1f4cd2d75fca3f98aaadcb6f6f188dd21cfd8a912886d36e65ef8989'
    evidence = ROOT / 'docs/phase3/evidence/arc-prompt/FROZEN.json'
    if evidence.exists(): assert evidence.read_bytes() == m3_pools.FROZEN_ARC.read_bytes()


def test_exclusion_rules_are_the_three_prompt_rules_only():
    pytest.importorskip('rapidfuzz')
    from kit.m3_pools import exclusion_pass
    sentence = 'the quick brown fox jumps over the lazy dog near the quiet river bank today'
    panels = [dict(id='general-v1:a', prompt='What is the capital of France? Answer briefly.'), dict(id='finqa:b', prompt=sentence),
              dict(id='7', prompt='an unrelated panel prompt ' + uid('x'))]  # same bare ID as pool row '7'
    pool = dict(
        instruction=rows('instruction', 3),
        contract_nli=[dict(id=f'c{i}', prompt='Agreement text shared by every hypothesis ' * 20 + f'Hypothesis {i}.', response='r') for i in range(3)],
        finqa=rows('finqa', 3) + [dict(id='finqa-dup', prompt=rows('finqa', 1)[0]['prompt'] + '.', response='r'), dict(id='7', prompt='distinct ' + uid('7'), response='r')],
        gsm8k=[dict(id='g-eq', prompt='  WHAT is the capital of   France? answer briefly.', response='r'),
               dict(id='g-jac', prompt=sentence.replace('today', 'tonight'), response='r')] + rows('gsm8k', 2),
        arc=[dict(id='a-edit', prompt='What is the capitol of France? Answer briefly.', response='r'),
             dict(id='a-pool', prompt=rows('gsm8k', 1)[0]['prompt'], response='r')] + rows('arc', 2))
    kept, excluded, failures = exclusion_pass(pool, panels)
    reasons = {e['id']: (e['rule'], e['overlaps']) for e in excluded}
    assert not failures and len(kept['instruction']) == 3
    assert reasons['g-eq'] == ('normalised_equality', 'panel:general-v1:a')
    assert reasons['g-jac'][0] == 'near_duplicate' and reasons['a-edit'][0] in ('near_duplicate', 'normalised_equality')
    assert reasons['a-pool'] == ('normalised_equality', 'gsm8k:gsm8k-0')
    assert reasons['finqa-dup'][0] == 'within_pool_near_duplicate'
    assert '7' in {r['id'] for r in kept['finqa']}, 'a bare-ID collision with a panel item is not an exclusion (manager ruling)'
    assert len(kept['contract_nli']) == 3, 'agreements shared across hypotheses are never de-duplicated within the legal pool'
    assert exclusion_pass(dict(instruction=[dict(id='i', prompt=sentence, response='r')]), panels)[2][0]['id'] == 'i'


def _freeze_inputs(tmp_path, sizes):
    from kit.p3b_common import files_manifest, atomic, t
    replay = tmp_path / 'b-replay'; replay.mkdir()
    accepted = rows('instruction', 64 + sizes['instruction'])
    t.write_jsonl(replay / 'accepted.jsonl', accepted); t.write_jsonl(replay / 'refresher-64.jsonl', accepted[:64]); t.write_jsonl(replay / 'replay.jsonl', accepted[64:])
    atomic(replay / 'manifest.json', files_manifest(replay, ['accepted.jsonl', 'refresher-64.jsonl', 'replay.jsonl'], technical_synthetic=True))
    contract = tmp_path / 'contract'; contract.mkdir()
    legal = [dict(r, document_id=f'doc-{i // 17}') for i, r in enumerate(rows('contract_nli', sizes['contract_nli']))]
    t.write_jsonl(contract / 'contract-replay.jsonl', legal)
    atomic(contract / 'manifest.json', files_manifest(contract, ['contract-replay.jsonl'], technical_synthetic=True))
    out = tmp_path / 'm3'; manifest.init_root(out)
    broad = out / 'pools/broad'; broad.mkdir(parents=True)
    t.write_jsonl(broad / 'gsm8k.jsonl', rows('gsm8k', sizes['gsm8k'])); t.write_jsonl(broad / 'arc.jsonl', rows('arc', sizes['arc']))
    atomic(broad / 'manifest.json', files_manifest(broad, ['gsm8k.jsonl', 'arc.jsonl'], technical_synthetic=True))
    finqa = tmp_path / 'finqa.jsonl'; t.write_jsonl(finqa, rows('finqa', sizes['finqa']))
    panel = tmp_path / 'panel.jsonl'; t.write_jsonl(panel, [dict(id='p1', prompt='a panel prompt ' + uid('p1'))])
    return NS(out=out, standin=True, replay=replay, contract=contract, finqa_pool=finqa, finqa_targets=None,
              tokenizer=None, panel=[f'general-v1={panel}'], holdout=[])


FULL = dict(instruction=960, gsm8k=80, arc=80, finqa=320, contract_nli=320)


def test_freeze_pools_ready_at_v2_minima_with_document_counts(tmp_path):
    pytest.importorskip('rapidfuzz')
    from kit import m3_pools
    a = _freeze_inputs(tmp_path, FULL)
    m3_pools.freeze_job(a)
    status = json.loads((a.out / 'pools/availability.json').read_text())
    assert {k: v['status'] for k, v in status.items()} == dict(C0='READY', K='READY', G='READY', CR='READY')
    loaded, meta = m3_pools.load(a.out, 'CR')
    assert set(loaded) == set(plan.DOMAINS) and len(loaded['instruction']) == 960 and meta['lengths'] is None
    assert set(m3_pools.load(a.out, 'C0')[0]) == {'instruction'}
    counts = json.loads((a.out / 'pools/contract-document-counts.json').read_text())
    assert counts['rows'] == 320 and counts['max_per_document'] == 17 and counts['documents'] == 19
    inputs = json.loads((a.out / 'pools/inputs.json').read_text())
    assert inputs['rules']['bare_id_collision'] is False


def test_short_pools_make_cr_and_g_pending_only(tmp_path):
    pytest.importorskip('rapidfuzz')
    from kit import m3_pools
    from kit.p3b_common import NotReady
    a = _freeze_inputs(tmp_path, dict(instruction=200, gsm8k=79, arc=10, finqa=20, contract_nli=20))
    m3_pools.freeze_job(a)
    status = json.loads((a.out / 'pools/availability.json').read_text())
    assert status['C0']['status'] == status['K']['status'] == 'READY'
    assert status['CR']['status'] == status['G']['status'] == 'PENDING'
    assert status['G']['reason']['short_pools']['gsm8k'] == [79, 80] and status['G']['reason']['short_pools']['instruction'] == [200, 960]
    with pytest.raises(NotReady): m3_pools.load(a.out, 'G')


def test_short_shared_instruction_pool_blocks_c0(tmp_path):
    pytest.importorskip('rapidfuzz')
    from kit import m3_pools
    a = _freeze_inputs(tmp_path, dict(instruction=150, gsm8k=8, arc=8, finqa=8, contract_nli=8))
    m3_pools.freeze_job(a)
    status = json.loads((a.out / 'pools/availability.json').read_text())
    assert all(v['status'] == 'PENDING' for v in status.values()) and status['C0']['reason']['short_pools'] == {'instruction': [150, 160]}


def test_contract_replay_rechecks_frozen_27b_journal(tmp_path):
    pytest.importorskip('rapidfuzz')
    from kit import c_targets as ct
    from kit import m3_pools
    from kit.c_sim import Tokenizer
    from kit.p3b_common import NotReady
    a = NS(out=tmp_path, standin=True, source=None, tokenizer=None, finqa_panel=None, prepare_only=False, gpu='0,1', model=None, standin_zero=False)
    with pytest.raises(NotReady): m3_pools.contract_replay(tmp_path / 'targets', ct.synthetic_rows(), [], Tokenizer(), True)
    ct.job(a)
    evaluation = ct.evaluation_items(a)
    replay, receipt = m3_pools.contract_replay(tmp_path / 'targets', ct.synthetic_rows(), evaluation, Tokenizer(), True)
    accepted = ct.t.read_jsonl(tmp_path / 'targets/teacher/accepted.jsonl')
    assert len(replay) == len(accepted) == receipt['accepted'] > 0
    assert all(r['domain'] == 'contract_nli' and r['response'].rstrip().endswith('Answer: ' + r['gold']) for r in replay)
    late = [r for r in replay if r['attempt'] == 2]
    assert late and all(int(r['id'].rsplit(':', 1)[1]) % 4 == 0 for r in late)
    assert all(len(x['attempts']) == next(r['attempt'] for r in replay if r['id'] == x['id']) for x in receipt['receipts'])
    assert receipt['document_counts']['rows'] == len(replay)


# ----------------------------------------------------------------------------- manifest, staging and isolation

def settings(tmp_path, **extra):
    value = dict(trainer_python='TPY', inference_python='IPY', nodes=['m3-node-a'], message2_nodes=['core-node'], technical_synthetic=True,
                 message2_roots=['/m2-out'],
                 inputs=dict(inventory='inv.json', contract_targets='/c/targets', replay='/b/replay', finqa_targets=['/b/targets/finqa'], return1='/returns',
                             core_durable='/m2-out/durable', panels={'general-v1': 'g.jsonl'}, base='base', tokenizer=None, finqa_pool='finqa.jsonl'))
    value.update(extra); return value


def closure(tasks, ident):
    seen = set(); stack = [ident]
    while stack:
        for parent in tasks[stack.pop()]['requires']:
            if parent not in seen: seen.add(parent); stack.append(parent)
    return seen


def test_manifest_is_isolated_complete_and_ordered_after_core(tmp_path):
    doc = manifest.build(settings(tmp_path))
    path = tmp_path / 'm3.json'; path.write_text(json.dumps(doc)); manifest.load(path)
    ids = [t['id'] for t in doc['tasks']]
    assert all(i.startswith('M3-') for i in ids) and len(ids) == len(set(ids)) and ids[:2] == ['M3-init', 'M3-bind-inputs']
    assert 'M3-stage' not in ids and {'M3-math-targets', 'M3-hellaswag'} <= set(ids)
    assert sum(t['kind'] == 'train-stage' for t in doc['tasks']) == 4 * 3 * 3
    tasks = {t['id']: t for t in doc['tasks']}
    for seed in plan.SEEDS:
        for name in (f'M3-kv-{seed}', f'M3-fv-{seed}'):
            assert 'M3-C0-101-finqa-export' in closure(tasks, name) and name in tasks['M3-read']['requires']
        assert not any(r.startswith(('M3-kv', 'M3-fv')) for r in tasks[f'M3-base-{seed}']['requires'])
    hellaswag = tasks['M3-hellaswag']['commands'][0]
    assert hellaswag[2].endswith('next_hellaswag.py') and hellaswag[hellaswag.index('--action') + 1] == 'prepare'
    schedules = tasks['M3-schedules']['commands'][0]
    assert schedules[schedules.index('--stages') + 1:schedules.index('--stages') + 3] == ['finqa', 'contract_nli']
    assert schedules[schedules.index('--schedule-roots') + 1:schedules.index('--schedule-roots') + 3] == ['{m3_out}/targets-finqa', '/c/targets']
    late = tasks['M3-schedules-math']['commands'][0]
    assert late[late.index('--stages') + 1] == 'math_hard' and late[late.index('--schedule-roots') + 1] == '{m3_out}/targets-math'
    # Round-4 ruling 1: stages 1-2 never wait for hard-MATH targets; every stage-3 train does.
    for arm in plan.ARMS:
        for seed in plan.SEEDS:
            for stage in ('finqa', 'contract_nli'):
                assert 'M3-math-targets' not in closure(tasks, f'M3-{arm}-{seed}-{stage}-train')
            assert {'M3-math-ready', 'M3-schedules-coverage-math' if arm in ('G', 'CR') else 'M3-schedules-math'} <= closure(tasks, f'M3-{arm}-{seed}-math_hard-train')
    math = tasks['M3-math-resolve']['commands'][0]
    assert math[math.index('--core-durable') + 1] == '/m2-out/durable' and tasks['M3-math-resolve']['requires'] == ['M3-init']
    finqa = tasks['M3-finqa-targets']['commands'][0]
    assert finqa[finqa.index('--finqa-targets') + 1] == '/b/targets/finqa'
    assert json.loads(tasks['M3-bind-inputs']['commands'][0][tasks['M3-bind-inputs']['commands'][0].index('--roots') + 1]) == dict(
        contract_targets='/c/targets', replay='/b/replay', finqa_targets_0='/b/targets/finqa')
    scoring = [t for t in doc['tasks'] if t['kind'] == 'score-family']
    assert all(t.get('family') == 'm3' for t in scoring)
    final = next(t for t in doc['tasks'] if t['id'] == 'M3-K-101-math_hard-score')
    panels = [(c[c.index('--step') + 1], c[c.index('--panel-set') + 1]) for c in final['commands']]
    assert ('20', 'finqa') in panels and ('20', 'test') in panels and ('40', 'knowledge-ll') in panels and ('10', 'finqa') not in panels
    m2 = dict(schema='message2-launch.v2', nodes=[dict(node='core-node')], tasks=[dict(id='M2', requires=[]), dict(id='M9', requires=['M2'])])
    report = manifest.isolation(doc, m2)
    assert report['passed'] and report['no_message2_outputs'] and report['runs_after_core_submission']
    def leak(ident, value):
        changed = json.loads(json.dumps(doc)); next(t for t in changed['tasks'] if t['id'] == ident)['commands'][0].append(value); return changed
    assert not manifest.isolation(leak('M3-assign', '/m2-out/targets-math'), m2)['no_message2_outputs']
    assert not manifest.isolation(leak('M3-math-targets', '/m2-out/pools'), m2)['no_message2_outputs'], 'core is read only via --core-durable'
    assert not manifest.isolation(doc, dict(m2, tasks=m2['tasks'] + [dict(id='M10', requires=['M3-read'])]))['no_m2_dependency_on_m3']
    assert manifest.isolation(manifest.build(settings(tmp_path, nodes=['core-node'])), m2)['passed']
    from kit import m2_node
    with pytest.raises(Exception): m2_node.load_manifest(path)


def _archive(path, members):
    import io
    import tarfile
    with tarfile.open(path, 'w:gz') as archive:
        for name, raw in members.items():
            info = tarfile.TarInfo(name); info.size = len(raw); archive.addfile(info, io.BytesIO(raw))
    return path


def _frozen_root(path, name='schedule-101.jsonl'):
    from kit.p3b_common import files_manifest, atomic, t
    path.mkdir(parents=True, exist_ok=True); t.write_jsonl(path / name, rows('finqa', 3))
    atomic(path / 'manifest.json', files_manifest(path, [name]))
    return path


def test_bind_inputs_hash_binds_package_roots_to_returned_archives(tmp_path):
    from kit.p3b_common import NotReady
    root = _frozen_root(tmp_path / 'c-out/targets')
    returns = tmp_path / 'returns'; returns.mkdir()
    archive = _archive(returns / 'package-x-return.tar.gz', {'return/targets/manifest.json': (root / 'manifest.json').read_bytes()})
    spec = tmp_path / 'binding.json'
    entry = dict(archive='package-x-return.tar.gz', member='targets/manifest.json', bytes=archive.stat().st_size, sha256=uid_bytes(archive.read_bytes()))
    spec.write_text(json.dumps(dict(schema='m3-input-binding.v1', inputs=dict(contract_targets=entry, unpinned=dict(archive=None, member=None)))))
    out = tmp_path / 'm3'; manifest.init_root(out)
    bound = manifest.bind_inputs(out, {'contract_targets': root}, returns, spec)
    assert bound['contract_targets']['member'] == 'targets/manifest.json' and manifest.bound_input(out, 'contract_targets', root)
    with pytest.raises(manifest.Unbound): manifest.bind_inputs(out, {'unpinned': root}, returns, spec)
    other = _frozen_root(tmp_path / 'other', 'schedule-102.jsonl')
    with pytest.raises(manifest.IsolationError): manifest.bind_inputs(tmp_path / 'm3', {'contract_targets': other}, returns, spec)
    with pytest.raises(NotReady): manifest.bound_input(out, 'unpinned', root)
    (root / 'schedule-101.jsonl').write_text('changed\n')
    with pytest.raises(Exception): manifest.bound_input(out, 'contract_targets', root)
    tampered = dict(entry, sha256='0' * 64); spec.write_text(json.dumps(dict(schema='m3-input-binding.v1', inputs=dict(contract_targets=tampered))))
    with pytest.raises(manifest.IsolationError): manifest.bind_inputs(tmp_path / 'm3', {'contract_targets': root}, returns, spec)


def uid_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def test_b_inputs_bind_in_place_like_message2_assets(tmp_path):
    """Round-4 ruling 2: B's replay/ and FinQA folders: frozen directory read in place, manifest sha256 recorded once."""
    replay = _frozen_root(tmp_path / 'B/replay', 'replay.jsonl'); finqa = tmp_path / 'B/targets/finqa'; finqa.mkdir(parents=True)
    from kit.p3b_common import atomic, s
    for part in ('train', 'val'): (finqa / (part + '.parquet')).write_bytes(b'parquet-fixture')
    atomic(finqa / 'manifest.json', dict(schema='kit-v4-sft-data.v1', arm='COMMON', scientific_phase_allowed=True, stand_in_teacher=False,
           train_sha256=s.file_hash(finqa / 'train.parquet'), val_sha256=s.file_hash(finqa / 'val.parquet')))
    out = tmp_path / 'm3'; manifest.init_root(out)
    bound = manifest.bind_inputs(out, {'replay': replay, 'finqa_targets_0': finqa}, tmp_path / 'no-returns')
    assert bound['replay'] == dict(root=str(replay.resolve()), manifest_sha256=uid_bytes((replay / 'manifest.json').read_bytes()), mode='assets')
    assert bound['finqa_targets_0']['mode'] == 'assets' and manifest.bound_input(out, 'finqa_targets_0', finqa)
    assert not (out / 'inputs/replay').exists(), 'bound in place, never copied or vendored'
    assert manifest.bind_inputs(out, {'replay': replay, 'finqa_targets_0': finqa}, tmp_path / 'no-returns') == bound
    (replay / 'replay.jsonl').write_text('changed\n')
    with pytest.raises(Exception): manifest.bound_input(out, 'replay', replay)


def test_shipped_input_binding_matches_message2_bindings():
    spec = json.loads((ROOT / 'kit/resources/m3/input-binding.json').read_text())
    assert spec['inputs']['contract_targets'] == dict(spec['inputs']['contract_targets'], archive='package-c-return.tar.gz', member='targets/manifest.json')
    assert spec['inputs']['replay']['mode'] == spec['inputs']['finqa_targets_*']['mode'] == 'assets'
    assert 'finqa_schedule' not in spec['inputs'] and not (ROOT / 'kit/resources/m3/b-replay').exists()
    from kit.m2_return1 import decisions
    assert 'package-c-return.tar.gz' in decisions()['provenance']['input_fingerprints']


def test_m3_root_refuses_message2_tree(tmp_path):
    (tmp_path / 'm2').mkdir(); (tmp_path / 'm2/binding.json').write_text('{}')
    with pytest.raises(manifest.IsolationError): manifest.init_root(tmp_path / 'm2')
    with pytest.raises(manifest.IsolationError): manifest.require_m3_root(tmp_path / 'fresh')
    manifest.init_root(tmp_path / 'fresh'); manifest.require_m3_root(tmp_path / 'fresh')
    assert not manifest.core_submitted(tmp_path / 'fresh')
    manifest.init_root(tmp_path / 'fresh', core_submitted=True); assert manifest.core_submitted(tmp_path / 'fresh')


def _toy_manifest(tmp_path):
    doc = dict(schema=manifest.SCHEMA, nodes=[], tasks=[
        dict(id='M3-a', kind='prepare', requires=[], commands=[['x', '{m3_out}']]),
        dict(id='M3-b', kind='prepare', requires=['M3-a'], commands=[['y']]),
        dict(id='M3-c', kind='prepare', requires=[], commands=[['boom']]),
        dict(id='M3-d', kind='prepare', requires=[], commands=[['notready']]),
        dict(id='M3-read', kind='collect', requires=['M3-a'], commands=[['read']], partial_inputs=True)])
    path = tmp_path / 'm3.json'; path.write_text(json.dumps(doc)); return path


def test_failed_m3_job_is_recorded_and_never_escapes(tmp_path):
    out = tmp_path / 'm3'; manifest.init_root(out, core_submitted=True); path = _toy_manifest(tmp_path)
    seen = []
    def runner(command):
        seen.append(command)
        if command[0] == 'boom': raise OSError('cannot start')
        return dict(x=1, y=0, notready=3, read=0)[command[0]]
    assert manifest.run_task(path, 'M3-a', out, {}, runner) == manifest.EXIT_FAILED
    assert manifest.status(out, 'M3-a')['status'] == 'FAILED' and seen[0] == ['x', str(out)]
    assert manifest.run_task(path, 'M3-b', out, {}, runner) == manifest.EXIT_PENDING
    assert manifest.status(out, 'M3-b')['blocked_by'] == ['M3-a'] and len(seen) == 1
    assert manifest.run_task(path, 'M3-c', out, {}, runner) == manifest.EXIT_FAILED and 'cannot start' in manifest.status(out, 'M3-c')['error']
    assert manifest.run_task(path, 'M3-d', out, {}, runner) == manifest.EXIT_PENDING
    assert manifest.run_task(path, 'M3-read', out, {}, runner) == 0
    assert manifest.run_task(path, 'M3-missing', out, {}, runner) == manifest.EXIT_FAILED
    assert {p.name for p in tmp_path.iterdir()} == {'m3', 'm3.json'}


def test_tier_jobs_wait_for_core_submission(tmp_path):
    out = tmp_path / 'm3'; manifest.init_root(out); path = _toy_manifest(tmp_path)
    assert manifest.run_task(path, 'M3-c', out, {}, lambda c: 0) == manifest.EXIT_PENDING
    assert 'core' in manifest.status(out, 'M3-c')['reason']
    manifest.init_root(out, core_submitted=True)
    assert manifest.run_task(path, 'M3-c', out, {}, lambda c: 0) == 0


# ----------------------------------------------------------------------------- scoring

def _standin_root(tmp_path, monkeypatch):
    from kit import m3_score, m2_assignment
    from kit.m3_train import bind_base
    out = tmp_path / 'm3'; manifest.init_root(out, core_submitted=True)
    for key in ('M2_ASSIGNMENT', 'M2_FAMILY', 'M2_FAMILY_SEED', 'M2_STANDIN_NODE'): monkeypatch.setenv(key, '')
    monkeypatch.delenv('M2_STANDIN_NODE')
    standin = m3_score.build_assignment([dict(node='CPU-STANDIN', gpus=[dict(uuid='CPU-STANDIN', environment='CPU-STANDIN')])], standin=True)
    m2_assignment.freeze(out / m3_score.ASSIGNMENT, standin)
    bind_base(NS(out=out, model=tmp_path / 'base', receipt=None, standin=True))
    return out


def test_scoring_assignment_one_dedicated_gpu_per_seed_refusing_core(tmp_path, monkeypatch):
    from kit import m3_score, m2_assignment
    nodes = [dict(node='n1', gpus=[dict(uuid='GPU-1', environment='e'), dict(uuid='GPU-2', environment='e')]),
             dict(node='n2', gpus=[dict(uuid='GPU-3', environment='e')])]
    doc = m3_score.build_assignment(nodes)
    assert [doc['assignments'][f'm3|{s}']['uuid'] for s in plan.SEEDS] == ['GPU-1', 'GPU-3', 'GPU-2']
    assert m3_score.build_assignment(nodes, message2_nodes=['n2']) == doc
    with pytest.raises(Exception): m3_score.build_assignment(nodes, core_uuids=['GPU-2'])
    out = _standin_root(tmp_path, monkeypatch)
    proofs = {s: m3_score.proof_for(out, '0', s, True) for s in plan.SEEDS}
    assert all(p['family'] == 'm3' and p['seed'] == s and p['uuid'] == 'CPU-STANDIN' for s, p in proofs.items())
    assert m2_assignment.verify(dict(assignment=proofs[101]), (out / m3_score.ASSIGNMENT).read_bytes()) == 'm3|101'


def test_knowledge_parsing_matches_frozen_panel():
    from kit import m3_score
    items = m3_score.knowledge_items()
    lines = (ROOT / 'kit/panels/general-v1.jsonl').read_text().splitlines()
    raw = {r['id']: r for r in (json.loads(l) for l in lines if l.strip()) if r['panel'] == 'knowledge'}
    assert len(items) == 100 and all(len(r['endings']) == 4 for r in items)
    for r in items:
        assert r['context'].startswith('Question: ') and r['context'].endswith('\nAnswer:') and 'Think step by step' not in r['context']
        assert 'ABCD'[r['label']] == raw[r['id']]['answer'] and all(f'({l}) {e}' in raw[r['id']]['prompt'] for l, e in zip('ABCD', r['endings']))
    with pytest.raises(Exception): m3_score.knowledge_document(dict(id='x', answer='A', prompt='no options here'))
    assert m3_score.selected([-1.0, -0.5, -0.5, -2.0]) == 1


def _fake_run(out, arm, seed, stage):
    from kit.p3b_common import files_manifest, atomic
    run = out / 'runs' / f'{arm}-s{seed}-{stage}'
    atomic(run / 'export-40.json', dict(technical_synthetic=True, model=f'm3-standin:{arm}:{seed}:{stage}', model_file_hashes={'model.safetensors': 'f' * 64}))
    atomic(run / 'manifest.json', files_manifest(run, ['export-40.json'])); atomic(run / 'complete.json', dict(complete=True))
    return run


def test_knowledge_likelihood_validated_on_base_and_c0_stage1_final_before_counting(tmp_path, monkeypatch):
    from kit import m3_score
    from kit.p3b_common import NotReady
    out = _standin_root(tmp_path, monkeypatch)
    unit = NS(out=out, arm='base', seed=101, stage='finqa', step=0, panel_set='knowledge-ll', gpu='0', standin=True, panel_root=None)
    m3_score.score_job(unit)  # scoring never waits for validation
    cells, _ = reader.load_cells(out, standin=True)
    cell = cells[('base', 101, 'finqa', 0, 'knowledge-ll')]
    assert set(cell['metrics']) == {'knowledge'} and len(cell['items']['knowledge']) == 100 and 'validation' in cell['pending']
    validate = NS(out=out, gpu='0', seed=101, standin=True)
    with pytest.raises(NotReady): m3_score.validate_knowledge(validate)  # C0 stage-1 final (seed 101) not trained yet
    _fake_run(out, 'C0', 101, 'finqa')
    m3_score.validate_knowledge(validate)
    record = json.loads(m3_score.validation_path(out, 'knowledge-ll', 'm3', 101).read_text())
    assert record['status'] == 'PASS' and set(record['checkpoints']) == {'base', 'C0-s101-finqa-k40'}
    assert [c['checkpoint'] for c in record['checks']] == ['base', 'C0-s101-finqa-k40'] and all(c['selected_choice_agreement'] == 100 for c in record['checks'])
    assert m3_score.validation_status(out, 'knowledge-ll', 101, standin=True) and m3_score.validation_status(out, 'knowledge-ll', 102, standin=True) is None
    cells, _ = reader.load_cells(out, standin=True)
    assert cells[('base', 101, 'finqa', 0, 'knowledge-ll')]['pending'] is None
    failed = dict(record, status='PENDING'); m3_score.validation_path(out, 'knowledge-ll', 'm3', 101).write_text(json.dumps(failed))
    assert m3_score.validation_status(out, 'knowledge-ll', 101, standin=True) is None


def _unit(out, arm, seed, stage, step, panel, proof):
    from kit.p3b_common import files_manifest, atomic
    from kit.m3_score import unit_root
    root = unit_root(out, arm, seed, stage, step, panel); root.mkdir(parents=True)
    (root / 'scores.csv').write_text('panel,question_id,correct,output_tokens,incorrect_format\nfinqa,1,1,10,0\nfinqa,2,0,12,0\n')
    atomic(root / 'm3-identity.json', dict(assignment=proof, seed=seed))
    atomic(root / 'manifest.json', files_manifest(root, ['scores.csv', 'm3-identity.json']))
    return root


def test_incoming_alias_points_to_same_assignment_and_detects_tampering(tmp_path):
    from kit import m3_score
    out = tmp_path / 'm3'; manifest.init_root(out)
    proof = dict(family='m3', seed=101, uuid='GPU-1', environment='e', node='n1')
    base = _unit(out, 'base', 101, 'finqa', 0, 'finqa', proof)
    m3_score.alias_job(NS(out=out, arm='K', seed=101, stage='finqa', step=0, panel_set='finqa'))
    assert m3_score.resolve(out, 'K', 101, 'finqa', 0, 'finqa') == base
    with pytest.raises(Exception): m3_score.alias_job(NS(out=out, arm='K', seed=101, stage='contract_nli', step=0, panel_set='finqa'))
    final = _unit(out, 'K', 101, 'finqa', 40, 'finqa', proof)
    m3_score.alias_job(NS(out=out, arm='K', seed=101, stage='contract_nli', step=0, panel_set='finqa'))
    assert m3_score.resolve(out, 'K', 101, 'contract_nli', 0, 'finqa') == final
    (base / 'scores.csv').write_text('panel,question_id,correct\nfinqa,1,1\n')
    with pytest.raises(Exception): m3_score.resolve(out, 'K', 101, 'finqa', 0, 'finqa')


# ----------------------------------------------------------------------------- reader

ASSIGN = {f'm3|{s}': dict(uuid=f'GPU-{s}', environment='env', node='n') for s in plan.SEEDS}
NOISE = {101: 0.0, 102: 1.0, 103: -0.5}
PANEL_METRICS = {'general-v1': ('ifeval', 'math', 'knowledge_strict'), 'knowledge-ll': ('knowledge',), 'hellaswag': ('hellaswag',),
                 'finqa': ('finqa',), 'test': ('test',), 'math_hard': ('math_hard',)}


def make_cells(bonus=lambda arm, seed, stage, step, metric: 0.0):
    cells = {}
    def put(arm, seed, stage, step):
        for panel, metrics in PANEL_METRICS.items():
            values = {m: 50 + NOISE[seed] + (0 if arm == 'base' else bonus(arm, seed, stage, step, m)) for m in metrics}
            items = {m: {f'q{i}': int(i < round(v)) for i in range(100)} for m, v in values.items()}
            cells[(arm, seed, stage, step, panel)] = dict(method='REFERENCE-SCORED' if panel in ('hellaswag', 'knowledge-ll') else 'KIT-SCORED', metrics=values, items=items, uuid=f'GPU-{seed}', environment='env', pending=None,
                format_failures={m: 0 for m in metrics}, output_tokens=1000, correct=50, tolerant={'test': 75.0} if panel == 'test' else {})
    for s in plan.SEEDS: put('base', s, 'finqa', 0)
    for arm in plan.ARMS:
        for s in plan.SEEDS:
            for stage in plan.STAGES:
                for k in reader.STEPS: put(arm, s, stage, k)
    return cells


K_WIN = {101: 5.0, 102: 6.0, 103: 5.5}
CR_WIN = {101: 4.0, 102: 4.5, 103: 5.0}


def winning(arm, seed, stage, step, metric):
    if arm == 'K' and stage == 'math_hard' and step == 40 and metric == 'ifeval': return K_WIN[seed]
    if arm == 'CR' and stage == 'math_hard' and step == 40 and metric in reader.GENERAL: return CR_WIN[seed]
    return 0.0


def test_reader_two_decisions_and_descriptive_contrasts():
    result = reader.analyse(make_cells(winning), ASSIGN)
    k = result['contrasts']['K-C0']
    assert k['label'] == 'PASS' and k['primary']['values'] == pytest.approx([5.0, 6.0, 5.5])
    assert k['primary']['lower'] == pytest.approx(5.5 - 4.302653 * 0.5 / 3 ** .5)
    assert result['contrasts']['CR-G']['label'] == 'PASS' and result['contrasts']['CR-G']['primary']['mean'] == pytest.approx(3 * 4.5 / 5)
    assert result['contrasts']['CR-C0']['label'] == result['contrasts']['G-C0']['label'] == 'DESCRIPTIVE'
    assert 'bars' not in result['contrasts']['CR-C0'] and result['contrasts']['CR-C0']['primary']['mean'] == pytest.approx(2.7)
    assert result['decisions'] == dict(K='PASS', CR='PASS', CR_interpretation='cumulative coverage confirmation opens',
                                       stacking='separately registered one-change stacking comparison needed')
    assert 'math_hard/task/gain_auc' in k['diagnostics'] and not any('gain_auc' in g for g in k['guards'])
    assert 'math_hard/retention/test' in k['guards'] and 'math_hard/retention_absolute/finqa' in k['guards']


def test_missing_curve_exports_preserve_registered_primaries_and_guards():
    complete = reader.analyse(make_cells(winning), ASSIGN)
    cells = make_cells(winning)
    exports = {}
    for key in list(cells):
        if key[3] in (10, 20, 30):
            exports[json.dumps(list(key[:4]))] = dict(status='MISSING-EXPORT', required=False)
            del cells[key]
    result = reader.analyse(cells, ASSIGN, exports=exports)
    assert result['decisions'] == complete['decisions']
    assert result['missing_intermediate_exports'] == {str(s): 36 for s in plan.SEEDS}
    for name in ('K-C0', 'CR-G'):
        assert result['contrasts'][name]['primary'] == complete['contrasts'][name]['primary']
        assert result['contrasts'][name]['guards'] == complete['contrasts'][name]['guards']
        diagnostics = result['contrasts'][name]['diagnostics']
        for metric, value in diagnostics.items():
            if 'gain_auc' in metric or 'step20' in metric:
                assert value['status'] == 'INCOMPLETE' and value['note'] == 'MISSING-EXPORT'
                assert 'mean' not in value and 'values' not in value
    del cells['K', 101, 'math_hard', 40, 'general-v1']
    assert reader.analyse(cells, ASSIGN, exports=exports)['decisions']['K'] == 'PENDING'


def test_reader_unmet_bar_and_failed_guard_are_inconclusive():
    small = reader.analyse(make_cells(lambda a, s, st, k, m: 1.0 if (a, st, k, m) == ('K', 'math_hard', 40, 'ifeval') else 0.0), ASSIGN)
    assert small['contrasts']['K-C0']['label'] == 'INCONCLUSIVE' and not small['contrasts']['K-C0']['bars']['mean_at_least_2']
    def harm(a, s, st, k, m):
        if a == 'K' and st == 'finqa' and k == 40 and m == 'finqa': return -5.0
        return winning(a, s, st, k, m)
    guarded = reader.analyse(make_cells(harm), ASSIGN)['contrasts']['K-C0']
    assert guarded['label'] == 'INCONCLUSIVE' and 'finqa/task/final' in guarded['failed_guards'] and all(guarded['bars'].values())
    def auc_only(a, s, st, k, m):
        if a == 'K' and st == 'finqa' and k in (10, 20, 30) and m == 'finqa': return -9.0
        return winning(a, s, st, k, m)
    assert reader.analyse(make_cells(auc_only), ASSIGN)['contrasts']['K-C0']['label'] == 'PASS', 'gain-AUC is diagnostic only'
    def both(a, s, st, k, m):
        if a in ('CR', 'G') and st == 'math_hard' and k == 40 and m in reader.GENERAL: return CR_WIN[s]
        return 0.0
    mixed = reader.analyse(make_cells(both), ASSIGN)
    assert mixed['decisions']['CR'] == 'INCONCLUSIVE' and mixed['contrasts']['CR-C0']['label'] == 'DESCRIPTIVE'


def test_reader_pending_and_unverified_are_scoped_to_the_affected_comparison():
    cells = make_cells(winning); del cells[('G', 102, 'contract_nli', 40, 'finqa')]
    result = reader.analyse(cells, ASSIGN)
    assert result['decisions']['CR'] == 'PENDING' and result['decisions']['K'] == 'PASS'
    assert reader.analyse(make_cells(winning), ASSIGN, pending=['G'])['decisions'] == dict(K='PASS', CR='PENDING', CR_interpretation=None, stacking=None)
    cells = make_cells(winning); cells[('CR', 103, 'finqa', 40, 'finqa')]['uuid'] = 'GPU-other'
    off = reader.analyse(cells, ASSIGN)
    assert off['decisions']['CR'] == 'UNVERIFIED-HARDWARE' and off['decisions']['K'] == 'PASS'
    receipts = {(a, s, st): 'PASS' for a in plan.ARMS for s in plan.SEEDS for st in plan.STAGES}
    receipts[('K', 101, 'finqa')] = 'FAIL'; receipts[('G', 101, 'finqa')] = 'INCOMPLETE'
    gated = reader.analyse(make_cells(winning), ASSIGN, receipts=receipts)
    assert gated['decisions']['K'] == 'UNVERIFIED' and gated['decisions']['CR'] == 'PENDING'


def test_reader_knowledge_is_likelihood_and_strict_letter_is_diagnostic():
    cells = make_cells(winning)
    for key, cell in cells.items():
        if key[4] == 'general-v1': cell['format_failures']['knowledge_strict'] = 3
    result = reader.analyse(cells, ASSIGN)
    assert result['decisions']['K'] == result['decisions']['CR'] == 'PASS' and result['strict_letter_knowledge_format_failures'] > 0
    cells = make_cells(winning); cells[('CR', 101, 'math_hard', 40, 'knowledge-ll')]['pending'] = 'likelihood validation missing'
    pending = reader.analyse(cells, ASSIGN)
    assert pending['decisions']['CR'] == 'PENDING' and 'pending' in pending['contrasts']['CR-G']['reason']


def test_reader_power_sensitivity_survival_and_report_only_bars():
    assert reader.SENSITIVITY == pytest.approx(2.48414, abs=1e-5)
    assert reader.sensitivity(0.314)['required_mean'] == pytest.approx(-0.219, abs=2e-3)  # registration power note
    assert reader.sensitivity(0.480)['required_mean'] == pytest.approx(0.193, abs=2e-3)
    assert reader.sensitivity(5.508)['required_mean'] == pytest.approx(12.682, abs=1e-3)
    assert reader.sensitivity(0.40)['zero_mean_passes'] and not reader.sensitivity(0.41)['zero_mean_passes']
    result = reader.analyse(make_cells(winning), ASSIGN)
    assert all('sensitivity' in g for g in result['contrasts']['K-C0']['guards'].values())
    survival = result['survival']['CR-G']['finqa->math_hard/finqa'][101]
    assert survival['cohort']['CR']['n'] == 50 and survival['intersection']['n'] == 50 and survival['intersection']['CR'] == 1.0
    bars = result['package_bars']['K']
    assert bars['role'] == 'report-only' and bars['legal_tolerant']['status'] == 'MET' and bars['math_standalone_reference']['status'] == 'PENDING'
    assert result['serving']['CR']['finqa']['status'] == 'WITHIN_CEILING'


def test_repriced_budget_and_scoring_wall_time():
    cost = reader.registered_costs()
    assert cost['C0']['total'] == pytest.approx(66.09, abs=.01) and cost['K']['total'] == pytest.approx([66.09, 85.92], abs=.01)
    assert cost['G']['total'] == pytest.approx(99.14, abs=.01) and cost['CR']['total'] == pytest.approx(122.33, abs=.01)
    assert cost['training_subtotal'] == pytest.approx([353.66, 373.49], abs=.02)
    total, scoring = reader.total_budget(618)
    assert scoring['gpu_hours'] == pytest.approx([30.9, 61.8]) and scoring['serial_hours_per_seed'] == pytest.approx([10.3, 20.6])
    assert total['total'] == pytest.approx([425, 516], abs=1) and not scoring['flag']
    assert reader.scoring_budget(1500)['flag']


def test_wall_time_flags_over_24_hours():
    tasks = [dict(id='M3-K-101-finqa-train', kind='train-stage', arm='K', seed=101), dict(id='M3-K-101-finqa-score', kind='score-family', arm='K', seed=101),
             dict(id='M3-C0-102-finqa-score', kind='score-family', arm='C0', seed=102)]
    states = {'M3-K-101-finqa-train': dict(status='DONE', seconds=20 * 3600), 'M3-K-101-finqa-score': dict(status='DONE', seconds=5 * 3600),
              'M3-C0-102-finqa-score': dict(status='FAILED', seconds=99 * 3600)}
    report = reader.wall_time(states, tasks)
    assert report['arm_seed_hours'] == {'K|101': pytest.approx(25.0), 'C0|102': pytest.approx(99.0)}
    assert 'ARM K|101 measured 25.0 h > 24 h' in report['flags'] and 'ARM C0|102 measured 99.0 h > 24 h' in report['flags']
    assert report['combined_queue_hours'] == 104.0


def test_sp_tier_checkpoint_lists_are_published_in_m3b2_format(tmp_path):
    required = ('checkpoint_id', 'family', 'seed', 'arm', 'step')
    for tier in ('K', 'CR'):
        published = json.loads((ROOT / f'kit/resources/m3/sp-tier-checkpoints-{tier}.json').read_text())
        assert published == reader.sp_tier_checkpoints(tier)
        assert published['schema'] == 'sp-tier-checkpoints-v1' and len(published['checkpoints']) == 3 + 2 * 3 * 3 * 2
        assert all(all(k in c for k in required) and c['seed'] in plan.SEEDS and c['role'] in ('base', 'incoming', 'final') for c in published['checkpoints'])
        assert {c['step'] for c in published['checkpoints']} == {0, 40} and len({c['checkpoint_id'] for c in published['checkpoints']}) == 39
        assert {c['arm'] for c in published['checkpoints']} == {'base', *reader.TIER_ARMS[tier]}
    out = tmp_path / 'm3'; manifest.init_root(out)
    (out / 'inputs').mkdir(); (out / 'inputs/base.json').write_text(json.dumps(dict(identity=dict(model_file_hashes={'a': 'b' * 64}))))
    _fake_run(out, 'K', 101, 'finqa')
    bound = {c['checkpoint_id']: c for c in reader.sp_tier_checkpoints('K', out)['checkpoints']}
    files = {'model.safetensors': 'f' * 64}
    assert bound['m3-K-s101-finqa-k40']['model_file_hashes'] == bound['m3-K-s101-contract_nli-k0']['model_file_hashes'] == files
    assert bound['m3-K-s101-finqa-k40']['weights_sha256'] == reader.sp_digest(files) and 'model_file_hashes' not in bound['m3-K-s101-math_hard-k0']
    assert bound['m3-base-s101']['model_file_hashes'] == bound['m3-K-s101-finqa-k0']['model_file_hashes'] == {'a': 'b' * 64}


def test_sp_candidate_superset_format_and_verification(tmp_path):
    import unicodedata
    from kit import m3_pools
    prompts = ['  What IS  2+2? ', [dict(role='user', content='Ｆｕｌｌ width')]]
    doc = m3_pools.superset({'gsm8k': prompts})
    def m3b2(prompt):  # m3b2's kit/sp_manifest.prompt_hash, restated
        if isinstance(prompt, list): prompt = '\n'.join(m.get('content', '') for m in prompt if isinstance(m, dict))
        return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', str(prompt)).casefold().split()).encode()).hexdigest()
    assert doc == dict(schema='sp-candidate-superset-v1', pools={'gsm8k': sorted({m3b2(p) for p in prompts})})
    path = tmp_path / 'superset.json'; path.write_text(json.dumps(doc))
    assert m3_pools.verify_superset({'gsm8k': [dict(prompt='what is 2+2?')]}, path)['status'] == 'VERIFIED'
    assert m3_pools.verify_superset({'gsm8k': [dict(prompt='new prompt')]}, path)['status'] == 'UNVERIFIED'
    assert m3_pools.verify_superset({'arc': [dict(prompt='what is 2+2?')]}, path)['status'] == 'UNVERIFIED'
    assert m3_pools.verify_superset({}, tmp_path / 'absent.json')['status'] == 'PENDING'


def _math_args(out, **kw):
    base = dict(out=out, standin=True, core_durable=None, model=None, tokenizer=None, gpu='0,1'); base.update(kw)
    return NS(**base)


@pytest.fixture(scope='module')
def fallback_targets(tmp_path_factory):
    """One stand-in run of the identical registered protocol (500 frozen items) in an M3 namespace."""
    import os
    from kit import m3_pools
    out = tmp_path_factory.mktemp('m3-fallback') / 'm3'; manifest.init_root(out)
    saved = dict(os.environ)
    try:
        m3_pools.math_resolve(_math_args(out)); m3_pools.math_targets(_math_args(out)); m3_pools.math_finish(_math_args(out))
    finally: os.environ.clear(); os.environ.update(saved)  # the protocol sets CUDA_VISIBLE_DEVICES like m2_targets
    return out


def test_hard_math_fallback_runs_the_registered_protocol_on_the_500_pool(fallback_targets,monkeypatch):
    from kit.v6_infra import identity
    monkeypatch.setenv('M2_INFRA_AUTHORITY_SHA256',identity())
    from kit import m3_pools
    from kit.m2_common import hard_data
    from kit import math_hard_teacher as h
    out = fallback_targets; root = out / 'targets-math'
    source = json.loads((out / m3_pools.TARGETS_SOURCE).read_text())
    assert source['path'] == 'm3-fallback' and source['core_durable_given'] is False
    data = hard_data(); pool = {r['id'] for r in data['train_pool.jsonl']}
    ids = {r['id'] for r in t_read(root / 'pool.jsonl')}
    assert ids <= pool and len(pool) == 500 and not ids & {r['id'] for r in data['eval.jsonl']}
    assert not ids & {r['id'] for r in h.frozen()['panel']}, 'never trains on the 200-item screen'
    assert json.loads((root / 'manifest.json').read_text())['population'] == 500
    assert all(len(t_read(root / f'schedule-{s}.jsonl')) == 1280 for s in plan.SEEDS)
    assert (root / 'teacher/demonstrations.jsonl').exists(), 'attempt journal archived beside the targets'
    m3_pools.math_targets(_math_args(out))  # frozen: second call is a no-op


def t_read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def test_hard_math_core_snapshot_is_copied_read_only_with_its_hash(fallback_targets, tmp_path):
    import shutil
    from kit import m3_pools, m2_snapshot
    # Stand in for core's committed M5: publish an out tree holding targets-math through m2_snapshot itself.
    core_out = tmp_path / 'core/local/M5/out'; core_out.parent.mkdir(parents=True)
    shutil.copytree(fallback_targets / 'targets-math', core_out / 'targets-math')
    (core_out / 'execution').mkdir(); (core_out / 'execution/M5.json').write_text('{}')
    durable = tmp_path / 'core/durable'
    with m2_snapshot.technical_space_budget(True):
        pointer = m2_snapshot.publish(core_out, durable, 'M5', complete=True)
    before = {p: p.stat().st_mtime_ns for p in durable.rglob('*')}
    out = tmp_path / 'm3'; manifest.init_root(out)
    with m2_snapshot.technical_space_budget(True):
        m3_pools.math_resolve(_math_args(out, core_durable=durable)); m3_pools.math_finish(_math_args(out, core_durable=durable))
    source = json.loads((out / m3_pools.TARGETS_SOURCE).read_text())
    assert source['path'] == 'core-copy' and source['snapshot_sha256'] == pointer['sha256'] and source['snapshot_manifest_sha256'] == pointer['manifest_sha256']
    assert (out / 'targets-math/manifest.json').read_bytes() == (core_out / 'targets-math/manifest.json').read_bytes()
    assert source['targets_manifest_sha256'] == uid_bytes((core_out / 'targets-math/manifest.json').read_bytes())
    assert {p: p.stat().st_mtime_ns for p in durable.rglob('*')} == before, 'core durable tree is never written'
    assert not (out / 'targets-math/execution').exists()


def test_hard_math_falls_back_when_core_has_not_committed(tmp_path):
    from kit import m3_pools
    assert m3_pools.core_targets(tmp_path / 'empty-durable', tmp_path / 'dest') is None
    pending = tmp_path / 'durable/snapshots/M5'; pending.mkdir(parents=True)
    (pending / 'CURRENT.json').write_text(json.dumps(dict(tar='a' * 64 + '.tar', sha256='a' * 64, complete=False)))
    (pending / ('a' * 64 + '.tar.sha256')).write_text('a' * 64 + '\n')
    assert m3_pools.core_targets(tmp_path / 'durable', tmp_path / 'dest') is None, 'an unfinished snapshot is not a commit'


def test_hard_math_root_must_be_the_frozen_500_pool(fallback_targets, tmp_path):
    import shutil
    from kit import m3_pools
    from kit.p3b_common import atomic, files_manifest
    copy = tmp_path / 'targets-math'; shutil.copytree(fallback_targets / 'targets-math', copy)
    m3_pools.check_math_root(copy, True)
    inputs = json.loads((copy / 'inputs.json').read_text()); inputs['pool'] = '0' * 64; atomic(copy / 'inputs.json', inputs)
    manifest_doc = json.loads((copy / 'manifest.json').read_text()); manifest_doc['files']['inputs.json'] = uid_bytes((copy / 'inputs.json').read_bytes())
    atomic(copy / 'manifest.json', manifest_doc)
    with pytest.raises(Exception): m3_pools.check_math_root(copy, True)


def test_actual_costs_sum_receipts(tmp_path):
    (tmp_path / 'runs/K-s101-finqa/training-costs').mkdir(parents=True)
    (tmp_path / 'runs/K-s101-finqa/training-costs/1.json').write_text(json.dumps(dict(kind='m3_training', arm='K', gpu_hours=2.5)))
    (tmp_path / 'scores/base-s101-k0-finqa').mkdir(parents=True)
    (tmp_path / 'scores/base-s101-k0-finqa/cost.json').write_text(json.dumps(dict(kind='scoring', gpu_hours=.5)))
    assert reader.actual_costs(tmp_path) == dict(gpu_hours={'m3_training:K': 2.5, 'scoring': .5}, total=3.0, attempts=2)

# Cross-family code review regression tests (m3a2s).

class M3LengthTokenizer:
    eos_token, eos_token_id = '<eos>', 2

    def encode(self, text, add_special_tokens=False):
        return [8] * len(text[:-5]) + [2] if text.endswith(self.eos_token) else [8] * len(text)

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = '[user]' + ''.join(m['content'] for m in messages) + '[assistant]'
        return self.encode(text) if tokenize else text


@pytest.mark.parametrize('length', [4097, 5000, 10240])
def test_m3_conversion_accepts_registered_long_inputs_and_refuses_overlength(length):
    from kit.m3_train import convert
    tok = M3LengthTokenizer()
    overhead = len(tok.apply_chat_template([dict(role='user', content='')]))
    row = dict(id='long-legal', prompt=[dict(role='user', content='L' * (length-overhead))], response='reasoning then Answer: entailment')
    schedule = [dict(id=row['id'], role='replay', domain='contract_nli', target_sha256=plan.target_sha256(row), update=1, position=1)]
    converted, tokens = convert(schedule, [], {'contract_nli': [row]}, tok, 'CR', 'math_hard')
    assert tokens['input'] == length and converted[0]['extra_info']['m3_target_ids'][-1] == tok.eos_token_id
    row['prompt'][0]['content'] += 'X' * (10241-length)
    schedule[0]['target_sha256'] = plan.target_sha256(row)
    with pytest.raises(Exception, match='common M3 cap'): convert(schedule, [], {'contract_nli': [row]}, tok, 'CR', 'math_hard')


def test_b_native_val_hash_rechecked_and_schema_not_generic(tmp_path):
    from kit.p3b_common import atomic, s
    root = tmp_path / 'targets'; root.mkdir()
    for part in ('train', 'val'): (root / (part + '.parquet')).write_bytes(part.encode())
    atomic(root / 'manifest.json', dict(schema='kit-v4-sft-data.v1', arm='COMMON', scientific_phase_allowed=True,
           train_sha256=s.file_hash(root / 'train.parquet'), val_sha256=s.file_hash(root / 'val.parquet')))
    assert 'files' not in manifest.validate_input('finqa_targets_0', root)
    (root / 'val.parquet').write_bytes(b'corrupt')
    with pytest.raises(Exception, match='hash differs'): manifest.validate_input('finqa_targets_0', root)


def test_failed_broad_has_no_dependency_path_to_shared_chains(tmp_path, monkeypatch):
    doc = manifest.build(settings(tmp_path)); tasks = {t['id']: t for t in doc['tasks']}
    for arm in ('C0', 'K'):
        for seed in plan.SEEDS:
            for stage in plan.STAGES:
                assert not ({'M3-broad', 'M3-contract', 'M3-finqa-replay', 'M3-pools'} & closure(tasks, f'M3-{arm}-{seed}-{stage}-train'))
    for arm in ('G', 'CR'):
        assert 'M3-broad' in closure(tasks, f'M3-{arm}-101-finqa-train')
    out = tmp_path / 'm3'; manifest.init_root(out, core_submitted=True)
    path = tmp_path / 'launch.json'; path.write_text(json.dumps(doc))
    for ident in closure(tasks, 'M3-C0-101-finqa-train'):
        manifest.record(out, ident, dict(status='DONE'))
    manifest.record(out, 'M3-broad', dict(status='FAILED'))
    monkeypatch.setattr(manifest.socket, 'gethostname', lambda: 'm3-node-a')
    assert manifest.run_task(path, 'M3-C0-101-finqa-train', out, {}, lambda c: 0) == 0
    assert manifest.run_task(path, 'M3-G-101-finqa-train', out, {}, lambda c: 0) == manifest.EXIT_PENDING
    assert manifest.run_task(path, 'M3-CR-101-finqa-train', out, {}, lambda c: 0) == manifest.EXIT_PENDING
    broad = tasks['M3-broad']['commands'][0]
    assert 'M3-finqa-replay' in closure(tasks, 'M3-broad')
    assert broad[broad.index('--finqa-pool') + 1] == '{m3_out}/finqa-replay/finqa.jsonl'


def test_shared_freeze_and_later_coverage_freeze_are_independent(tmp_path):
    pytest.importorskip('rapidfuzz')
    from kit import m3_pools
    import shutil
    a = _freeze_inputs(tmp_path, FULL)
    shutil.rmtree(a.out / 'pools/broad')
    a.shared_only = True
    m3_pools.freeze_job(a)
    assert set(m3_pools.load(a.out, 'C0')[0]) == {'instruction'}
    assert m3_pools.pool_root(a.out, 'C0').name == 'pools-shared'
    with pytest.raises(Exception): m3_pools.load(a.out, 'G')
    shared_bytes = (a.out / 'pools-shared/manifest.json').read_bytes()
    a.shared_only = False
    m3_pools.freeze_job(a)
    assert (a.out / 'pools-shared/manifest.json').read_bytes() == shared_bytes
    assert json.loads((a.out / 'pools/availability.json').read_text())['G']['status'] == 'PENDING'


def test_task_candidate_evidence_requires_membership_and_exclusions(tmp_path):
    pytest.importorskip('rapidfuzz')
    from kit import m3_pools
    candidate = dict(id='train', prompt='A unique task prompt ' + uid('train'))
    path = tmp_path / 'universe.json'
    with pytest.raises(Exception, match='superset'): m3_pools.task_evidence([candidate], 'math_hard', [], path)
    path.write_text(json.dumps(m3_pools.superset({'task-math_hard': [candidate['prompt']]})))
    assert m3_pools.task_evidence([candidate], 'math_hard', [], path)['status'] == 'VERIFIED'
    with pytest.raises(Exception, match='overlaps'): m3_pools.task_evidence([candidate], 'math_hard', [candidate], path)
    with pytest.raises(Exception, match='superset'): m3_pools.task_evidence([dict(candidate, prompt='foreign')], 'math_hard', [], path)


def test_published_math_candidate_universe_is_500_training_items(monkeypatch):
    from kit.v6_infra import identity
    monkeypatch.setenv('M2_INFRA_AUTHORITY_SHA256',identity())
    from kit import m3_pools
    from kit.m2_common import hard_data
    from kit import math_hard_teacher
    doc = json.loads(m3_pools.SUPERSET.read_text())
    expected = {m3_pools.sp_prompt_hash(m3_pools.t.student_messages(r)) for r in hard_data()['train_pool.jsonl']}
    screen = {m3_pools.sp_prompt_hash(m3_pools.t.student_messages(r)) for r in math_hard_teacher.frozen()['panel']}
    assert set(doc['pools']['task-math_hard']) == expected and len(expected) == 500 and not expected & screen
    assert {'instruction', 'gsm8k', 'arc', 'finqa', 'contract_nli', 'task-finqa', 'task-contract_nli', 'task-math_hard'} <= set(doc['pools'])


def test_scientific_assignment_needs_three_gpus():
    from kit.m3_score import build_assignment
    nodes = [dict(node='fixture', gpus=[dict(uuid='GPU-fixture', environment='test')])]
    with pytest.raises(Exception, match='three distinct'): build_assignment(nodes)
    assert len(build_assignment(nodes, standin=True)['assignments']) == 3


def test_resolved_physical_gpu_is_forwarded_to_inherited_scorer(tmp_path, monkeypatch):
    from kit import m3_score, m2_gpu, p3b_score
    out = _standin_root(tmp_path, monkeypatch)
    a = NS(out=out, gpu='0', seed=101, standin=False, arm='base', stage='finqa', step=0, panel_set='finqa', panel_root=None)
    def proof(*args):
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
        return dict(family='m3', seed=101, uuid='GPU-assigned', environment='e')
    monkeypatch.setattr(m3_score, 'proof_for', proof)
    verified = []
    monkeypatch.setattr(m2_gpu, 'assert_current', lambda: verified.append(True))
    monkeypatch.setattr(m3_score, 'checked', lambda *args: (dict(model='local', base=True), dict(model_file_hashes={})))
    def stop(args):
        assert args.gpu == '1' and __import__('os').environ['CUDA_VISIBLE_DEVICES'] == '1'
        raise RuntimeError('inference boundary reached')
    monkeypatch.setattr(p3b_score, 'job', stop)
    with pytest.raises(RuntimeError, match='inference boundary'): m3_score.score_job(a)
    assert verified and a.gpu == '1'


@pytest.mark.parametrize('mutation', ['synthetic', 'assignment', 'weights', 'items', 'coverage', 'nan', 'agreement', 'details'])
def test_likelihood_validation_rejects_stale_or_incomplete_evidence(tmp_path, monkeypatch, mutation):
    from kit import m3_score
    out = _standin_root(tmp_path, monkeypatch); _fake_run(out, 'C0', 101, 'finqa')
    m3_score.validate_knowledge(NS(out=out, gpu='0', seed=101, standin=True))
    path = m3_score.validation_path(out, 'knowledge-ll', 'm3', 101)
    value = json.loads(path.read_text())
    if mutation == 'synthetic':
        assert m3_score.validation_status(out, 'knowledge-ll', 101, standin=False) is None
        return
    if mutation == 'assignment': value['assignment']['uuid'] = 'foreign'
    if mutation == 'weights': value['checkpoints']['base']['model_file_hashes']['config.json'] = 'changed'
    if mutation == 'items': value['items_sha256'] = 'foreign'
    if mutation == 'coverage': value['checks'][0]['reference'].pop()
    if mutation == 'nan': value['checks'][0]['reference'][0]['scores'][0] = float('nan')
    if mutation == 'agreement': value['checks'][0]['reference'][0]['scores'] = [0, 1e9, 0, 0]
    if mutation == 'details': value['checks'][0]['finite_likelihoods'] = False
    path.write_text(json.dumps(value))
    assert m3_score.validation_status(out, 'knowledge-ll', 101, standin=True) is None


def test_runtime_node_refusal_and_exclusive_training_scoring_lock(tmp_path, monkeypatch):
    doc = dict(schema=manifest.SCHEMA, nodes=['allowed'], message2_nodes=['core'], technical_synthetic=True, tasks=[
        dict(id='M3-train', kind='train-stage', requires=[], commands=[['train']]),
        dict(id='M3-score', kind='score-family', requires=[], commands=[['score']])])
    path = tmp_path / 'launch.json'; path.write_text(json.dumps(doc))
    out = tmp_path / 'm3'; manifest.init_root(out, core_submitted=True)
    seen = []
    for host in ('core', 'unlisted'):
        monkeypatch.setattr(manifest.socket, 'gethostname', lambda: host)
        assert manifest.run_task(path, 'M3-train', out, {}, lambda c: seen.append(c)) == manifest.EXIT_FAILED
    assert not seen
    monkeypatch.setattr(manifest.socket, 'gethostname', lambda: 'allowed')
    def train(command):
        assert manifest.run_task(path, 'M3-score', out, {}, lambda c: seen.append(c)) == manifest.EXIT_PENDING
        return 0
    assert manifest.run_task(path, 'M3-train', out, {}, train) == 0
    assert not seen
    assert manifest.run_task(path, 'M3-score', out, {}, lambda c: 0) == 0


def _native_fixture(run, step, arm='K', seed=101, stage='finqa'):
    from kit import m3_contract as contract
    from kit.p3b_common import atomic
    path = run / 'sdft' / f'global_step_{step}'; actor = path / 'actor'; actor.mkdir(parents=True)
    (path / 'data.pt').write_bytes(b'loader-state')
    for rank in range(8):
        for kind in ('model', 'optim', 'extra_state'): (actor / f'{kind}_world_size_8_rank_{rank}.pt').write_bytes(b'native-state')
        ema = actor / 'm3-ema'; ema.mkdir(exist_ok=True); (ema / f'model_world_size_8_rank_{rank}.pt').write_bytes(b'ema-state')
    atomic(actor / 'm3-state.json', dict(arm=arm, seed=seed, stage=stage, profile='scientific', base_revision=contract.BASE_REVISION,
           code_sha256=contract.code_digest(), frozen_base='base', **dict.fromkeys(('completed_updates', 'optimizer_steps', 'ema_steps', 'scheduler_steps'), step)))
    return path


def test_interruption_resumes_latest_complete_native_state_then_exports_only(tmp_path):
    from kit import m3_train, m3_contract as contract
    from kit.p3b_common import atomic
    run = tmp_path / 'runs/K-s101-finqa'
    assert m3_train.training_action(run, 'K', 101, 'finqa') == ('start', None)
    atomic(run / 'started.json', dict(started=True))
    assert m3_train.training_action(run, 'K', 101, 'finqa') == ('start', None)
    assert len(list((run / 'attempts').glob('*/manifest.json'))) == 1
    old = _native_fixture(run, 10)
    partial = run / 'sdft/global_step_20'; partial.mkdir(); (partial / 'partial').write_bytes(b'keep')
    assert m3_train.training_action(run, 'K', 101, 'finqa') == ('resume', old)
    a = NS(out=tmp_path, sdpo=tmp_path / 'sdpo', seed=101, gpu='0,1,2,3,4,5,6,7', tokenizer='tok', trainer_python='py')
    cmd, _ = m3_train.command(a, 'K', 'finqa', tmp_path / 'incoming', tmp_path / 'data', run,
                             dict(model='base', revision=contract.BASE_REVISION), contract.geometry('K', 'finqa'), old)
    assert '++trainer.resume_mode=resume_path' in cmd and '++trainer.resume_from_path=' + str(old.resolve()) in cmd
    assert not any('resume_mode=disable' in v for v in cmd)
    final = _native_fixture(run, 40)
    atomic(run / 'm3-receipt.json', dict(status='INCOMPLETE'))
    assert m3_train.training_action(run, 'K', 101, 'finqa') == ('export', final)
    assert (partial / 'partial').read_bytes() == b'keep'
    (final / 'actor/model_world_size_8_rank_0.pt').write_bytes(b'changed')
    with pytest.raises(Exception, match='identity changed'): m3_train.training_action(run, 'K', 101, 'finqa')


def test_reader_opens_frozen_receipt_and_checks_stage_base_identity(tmp_path):
    from kit import m3_train, m3_contract as contract
    from kit.p3b_common import atomic, files_manifest
    out = tmp_path / 'm3'; manifest.init_root(out)
    atomic(out / 'inputs/base.json', dict(technical_synthetic=False, frozen_base_files_sha256='a'*64))
    run = out / 'runs/K-s101-finqa'
    atomic(run / 'complete.json', dict(receipt_status='PASS'))
    atomic(run / 'manifest.json', files_manifest(run, ['complete.json']))
    with pytest.raises(Exception, match='receipt not frozen'): m3_train.validate_stage_receipt(out, 'K', 101, 'finqa')
    receipt = dict(schema_version=contract.SCHEMA_VERSION, arm='K', seed=101, stage='finqa', profile='scientific', status='INCOMPLETE',
                   world_size=8, loss_ids=contract.LOSS_IDS, **{k:v for k,v in contract.geometry('K', 'finqa').items() if k!='local_loss_scale'},
                   completed_updates=40, optimizer_steps=40, ema_steps=40, scheduler_steps=40, frozen_base_before='base', frozen_base_after='base',
                   frozen_base_has_grad=False, frozen_base_files_sha256='a'*64, attention_implementation='sdpa', gates={}, telemetry={})
    atomic(run / 'm3-receipt.json', receipt)
    atomic(run / 'manifest.json', files_manifest(run, ['m3-receipt.json']))
    assert m3_train.validate_stage_receipt(out, 'K', 101, 'finqa')['status'] == 'INCOMPLETE'
    for key, value in (('arm', 'C0'), ('seed', 102), ('stage', 'contract_nli'), ('frozen_base_files_sha256', 'b'*64)):
        atomic(run / 'm3-receipt.json', dict(receipt, **{key:value}))
        atomic(run / 'manifest.json', files_manifest(run, ['m3-receipt.json']))
        with pytest.raises(Exception): m3_train.validate_stage_receipt(out, 'K', 101, 'finqa')


def test_attempt_costs_include_math_validation_failed_scores_and_queue(tmp_path):
    from kit.p3b_common import atomic
    atomic(tmp_path / 'targets-math/cost.json', dict(kind='m3_math_targets', gpu_hours=2.))
    atomic(tmp_path / 'gpu-costs/1.json', dict(kind='m3_likelihood_validation', gpu_hours=1., completed=True))
    atomic(tmp_path / 'gpu-costs/2.json', dict(kind='m3_scoring_attempt', gpu_hours=3., completed=False))
    atomic(tmp_path / 'gpu-costs/3.json', dict(kind='m3_hellaswag_likelihood', gpu_hours=.5, included_in='m3_scoring_attempt'))
    assert reader.actual_costs(tmp_path)['total'] == 6.
    tasks = [dict(id='M3-score', kind='score-family', seed=101, arm='K')]
    states = {'M3-score': dict(status='DONE', attempts=[dict(status='FAILED', submitted_at=0, started_at=24*3600, ended_at=25*3600, seconds=3600),
              dict(status='DONE', submitted_at=0, started_at=26*3600, ended_at=27*3600, seconds=3600)])}
    report = reader.wall_time(states, tasks)
    assert report['arm_seed_hours']['K|101'] == 27. and report['waiting_hours']['M3-score'] == 25.
    assert report['execution_hours']['M3-score'] == 2. and report['flags']


def test_shared_then_coverage_schedules_publish_without_mutating_shared(tmp_path):
    pytest.importorskip('rapidfuzz')
    from kit import m3_pools, m3_train
    a = _freeze_inputs(tmp_path, FULL)
    a.shared_only = True; m3_pools.freeze_job(a)
    root = a.out / 'targets-finqa'
    m3_pools.task_root(root, task_rows('finqa')[:100], {}, True)
    args = NS(out=a.out, standin=True, stages=['finqa'], schedule_roots=[root], tokenizer=None, arms=['C0', 'K'])
    m3_train.schedules_job(args)
    shared = m3_train.schedule_dir(a.out, 'finqa') / 'manifest.json'; before = shared.read_bytes()
    a.shared_only = False; m3_pools.freeze_job(a)
    args.arms = ['G', 'CR']; m3_train.schedules_job(args)
    assert before == shared.read_bytes()
    assert (m3_train.schedule_dir(a.out, 'finqa', 'G') / 'manifest.json').exists()
    for seed in plan.SEEDS:
        streams = [plan.shared_generic(m3_pools.t.read_jsonl(m3_train.schedule_path(a.out, arm, seed, 'finqa'))) for arm in plan.ARMS]
        assert streams[0] == streams[1] == streams[2] == streams[3]


def test_child_gpu_entry_requires_live_reservation_and_checks_runtime_host(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest.socket, 'gethostname', lambda: 'allowed')
    policy = dict(nodes=['allowed'], message2_nodes=['core'])
    monkeypatch.setattr(manifest, 'allocated_gpus', lambda count: ['GPU-fixture'])
    with manifest.node_reservation(policy, gpus=1) as fds:
        monkeypatch.setenv('M3_GPU_LOCK_FDS', json.dumps(fds)); monkeypatch.setenv('M3_NODE_POLICY', json.dumps(policy))
        with manifest.gpu_execution(tmp_path): pass
        monkeypatch.setattr(manifest.socket, 'gethostname', lambda: 'core')
        with pytest.raises(Exception, match='executing node'):
            with manifest.gpu_execution(tmp_path): pass
    monkeypatch.delenv('M3_GPU_LOCK_FDS'); monkeypatch.delenv('M3_NODE_POLICY')
    monkeypatch.delenv('M3_MANIFEST', raising=False)
    with pytest.raises(Exception, match='requires the M3 node manifest'):
        with manifest.gpu_execution(tmp_path): pass


def test_submission_time_uses_scheduler_submit_time_without_recording_identifier(monkeypatch):
    monkeypatch.delenv('M3_SUBMITTED_AT', raising=False)
    monkeypatch.setenv('SLURM_JOB_ID', 'fixture')
    monkeypatch.setattr(manifest.subprocess, 'check_output', lambda *args, **kw: 'SubmitTime=2026-10-29T10:00:00 StartTime=2026-10-30T11:00:00')
    value, known = manifest.submission_time({})
    assert known and value == manifest.datetime.fromisoformat('2026-10-29T10:00:00').timestamp()
    assert manifest.submission_time({'submitted_at': 123.}) == (123., True)


def test_pinned_broad_sources_verify_local_parquet_hashes(tmp_path, monkeypatch):
    from kit import m3_pools, next_hellaswag
    pins = json.loads((m3_pools.M3 / 'candidate-sources.json').read_text())
    calls = []; original_download = next_hellaswag.download
    def local(repo, revision, fragment, expected_files):
        calls.append((repo, revision, fragment, expected_files))
        return [dict(fixture=True)], dict(expected_files)
    monkeypatch.setattr(next_hellaswag, 'download', local)
    loaded = m3_pools.offline_broad_sources()
    for name, call in zip(('gsm8k', 'arc'), calls):
        assert call[0] == pins[name]['repository'] and call[1] == pins[name]['revision'] and call[3] == pins[name]['files']
        assert loaded[name][1] == pins[name]['files']
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist([dict(question='fixture')]), tmp_path / 'train-000.parquet')
    with pytest.raises(Exception, match='SHA256 mismatch'):
        # Exercise the underlying verifier with a real CPU parquet fixture; no network/cache call.
        original_download('fixture', 'revision', 'train-', {'train-000.parquet': '0'*64}, root=tmp_path)


def test_node_reservation_refuses_orphaned_gpu_work(monkeypatch):
    monkeypatch.setattr(manifest.socket, 'gethostname', lambda: 'allowed')
    monkeypatch.setattr(manifest.subprocess, 'check_output', lambda *args, **kw: 'GPU-fixture, 123\n')
    monkeypatch.setattr(manifest, 'allocated_gpus', lambda count: ['GPU-fixture'])
    with pytest.raises(manifest.Unbound, match='active GPU work'):
        with manifest.node_reservation(dict(nodes=['allowed']), check_idle=True): pass


def test_materialized_finqa_replay_is_required_reference_for_broad_candidates(tmp_path, monkeypatch):
    pytest.importorskip('rapidfuzz')
    from kit import m3_pools, next_pools
    gsm = [dict(question='FinQA reference conflict ' + uid('conflict'), answer='#### 7')]
    prompt = next_pools.candidates(gsm, [])[0]['prompt']
    earlier = [dict(id='earlier', prompt=prompt, response='A verified explanation then Answer: 7')]
    out = tmp_path / 'm3'; manifest.init_root(out)
    monkeypatch.setattr(m3_pools, 'finqa_rows', lambda *args: (earlier, dict(verified=True)))
    a = NS(out=out, standin=True, finqa_targets=[tmp_path / 'bound-targets'], tokenizer=None, panel=[], holdout=[])
    m3_pools.finqa_replay(a)
    reference = m3_pools.t.read_jsonl(out / 'finqa-replay/finqa.jsonl')
    assert reference == earlier
    selected, excluded = m3_pools.broad_candidates(gsm, [], reference)
    assert not selected and excluded[0]['rule'] == 'normalised_equality'


@pytest.fixture
def native_b_finqa(tmp_path, monkeypatch):
    """Build B's real COMMON scheduled parquets through its campaign and v4_sft code, CPU tokenizer only."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kit import v4_campaign_ops, v4_sft
    from kit.p3b_common import atomic, s
    tokenizer = M3LengthTokenizer(); tokenizer.pad_token_id = 0
    demonstration = 'The report provides the quantities and their arithmetic gives the required total.\nAnswer: 7'
    # Non-lexical common order, fewer than 1,280 rows: B's own builder must preserve replacement/repetition.
    common = [dict(prompt=[dict(role='user', content='Report fixture ' + uid(str(i)) + ': compute its total.')],
                   data_source='finqa', reward_model=dict(style='rule', ground_truth='7'),
                   extra_info=dict(index=f'q{(i*37)%67}', split='train', demonstration=demonstration, rewrite=demonstration,
                                   problem='fixture arithmetic', description='fixture report', teacher_model='Qwen3.6-27B', stand_in_teacher=False))
              for i in range(67)]
    work = tmp_path / 'b'; base = v4_campaign_ops.paths(work, 'main'); (base / 'common').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(common), base / 'common/finqa.parquet')
    atomic(base / 'common/manifest.json', dict(schema='kit-v4-common-training.v1', teacher_models=['Qwen3.6-27B'],
           stand_in_teacher=False, coverage=dict(scientific_phase_allowed=True),
           files=dict(finqa=dict(sha256=s.file_hash(base / 'common/finqa.parquet')))))
    test = work / 'v4/datasets/datasets/v4_finqa/test.parquet'; test.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([dict(fixture=True)]), test)
    monkeypatch.setattr(v4_sft, 'load_student_tokenizer', lambda *args: tokenizer)
    monkeypatch.setattr(v4_campaign_ops, 'job_for', lambda ident, phase: dict(task='finqa', seed=int(ident.rsplit('-', 1)[1])))
    monkeypatch.setenv('QWEN3_8B_TOKENIZER', 'cpu-fixture')
    for seed in plan.SEEDS: v4_campaign_ops.schedule(work, 'main', f'fixture-{seed}')
    return [base / 'scheduled' / f'finqa-s{seed}' for seed in plan.SEEDS], tokenizer


def test_finqa_seed101_first_28_task_ids_match_b_d_consumption(native_b_finqa, tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    from kit import m3_pools, m3_train, p3b_train
    from kit.p3b_common import cycle, t
    folders, tokenizer = native_b_finqa
    # B's scientific D+replay path reads this parquet and calls d_replay_rows, without resampling tasks.
    native = pq.read_table(folders[0] / 'train.parquet').to_pylist()
    generic = [dict(r, p3b_replay=True) for r in POOLS['instruction']]
    b_mixed, audit = p3b_train.d_replay_rows(native, generic, 101, tokenizer)
    expected = [str(r['extra_info']['index']) for r in b_mixed[:32] if not r.get('p3b_replay')]
    assert len(expected) == 28 and audit['skipped_draw_count'] == 0
    out = tmp_path / 'm3'; manifest.init_root(out)
    monkeypatch.setattr(m3_train, 'tokenizer_for', lambda a: tokenizer)
    args = NS(out=out, standin=False, finqa_targets=list(reversed(folders)), tokenizer='cpu-fixture')
    m3_pools.finqa_targets(args)
    root = out / 'targets-finqa'; receipt = m3_pools.check_finqa_root(root)
    prepared = t.read_jsonl(root / 'schedule-101.jsonl')
    native_ids = [str(r['extra_info']['index']) for r in native]
    assert [r['id'] for r in prepared] == native_ids and len(set(native_ids)) == 67
    # Reproduce the previous merge/sort/re-shuffle and prove this fixture detects it.
    earlier, _ = m3_pools.finqa_rows(folders, False, tokenizer)
    assert [r['id'] for r in cycle(earlier, 101, 1280)] != native_ids
    for arm in ('C0', 'K'):
        schedule, _ = plan.build(arm, 101, 'finqa', prepared, POOLS)
        observed = [r['id'] for r in schedule[:32] if r['role'] == 'task']
        assert observed == expected == receipt['sources']['101']['first_update_task_ids']
        assert [r['id'] for r in schedule if r['role'] == 'task'] == [str(r['extra_info']['index']) for r in b_mixed if not r.get('p3b_replay')]
    for arm in ('G', 'CR'):
        schedule, _ = plan.build(arm, 101, 'finqa', prepared, POOLS)
        assert [r['id'] for r in schedule[:32]] == native_ids[:32]
    for seed, folder in zip(plan.SEEDS, folders):
        assert [r['id'] for r in t.read_jsonl(root / f'schedule-{seed}.jsonl')] == json.loads((folder / 'manifest.json').read_text())['ids']
    assert len(t.read_jsonl(root / 'pool.jsonl')) == 3 * 1280
    before = (root / 'manifest.json').read_bytes(); m3_pools.finqa_targets(args)
    assert (root / 'manifest.json').read_bytes() == before
    assert m3_train.schedule_binding(NS(out=out, standin=False), 'finqa', root)['source'] == receipt


@pytest.mark.parametrize('failure', ['missing-seed', 'duplicate-seed', 'order', 'hash'])
def test_finqa_native_schedule_has_no_pooled_fallback(native_b_finqa, failure):
    from kit import m3_pools
    folders, tokenizer = native_b_finqa
    if failure == 'missing-seed': folders = folders[:2]
    if failure == 'duplicate-seed': folders = folders + [folders[0]]
    if failure == 'order':
        path = folders[0] / 'manifest.json'; value = json.loads(path.read_text()); value['ids'] = list(reversed(value['ids'])); path.write_text(json.dumps(value))
    if failure == 'hash': (folders[0] / 'train.parquet').write_bytes(b'changed')
    with pytest.raises(Exception): m3_pools.finqa_task_schedules(folders, False, tokenizer)


def test_finqa_rejects_obsolete_merged_and_reshuffled_freeze(tmp_path):
    from kit import m3_pools
    root = tmp_path / 'targets-finqa'; m3_pools.task_root(root, rows('finqa', 67), {}, True)
    with pytest.raises(Exception, match='obsolete FinQA task schedule'): m3_pools.check_finqa_root(root)


@pytest.mark.parametrize('arm', plan.ARMS)
def test_preflight_manifest_builds_stage2_longest_legal_inputs_without_gpu(tmp_path, monkeypatch, arm):
    import pyarrow.parquet as pq
    from kit import m3_train, m3_pools, m3_contract as contract, m3_entry
    from kit.p3b_common import atomic, files_manifest, t
    out = tmp_path / 'm3'; manifest.init_root(out)
    tokenizer = M3LengthTokenizer()
    legal = rows('legal', 100); legal[0]['prompt'] = 'L' * 5000
    task_root = out / 'contract-targets'; m3_pools.task_root(task_root, legal, {}, False)
    task = t.read_jsonl(task_root / 'schedule-101.jsonl')
    schedule, _ = plan.build(arm, 101, 'contract_nli', task, POOLS)
    source = m3_train.schedule_dir(out, 'contract_nli', arm); name = f'{arm}/s101.jsonl'
    (source / arm).mkdir(parents=True); t.write_jsonl(source / name, schedule)
    pids, _ = t.tokenize_student(tokenizer, m3_train.as_chat(legal[0]['prompt']), legal[0]['response'])
    atomic(source / 'audit.json', dict(schedule_binding={'fixture': 'bound'}, longest_rows={'task:contract_nli': dict(id=legal[0]['id'], prompt_tokens=len(pids))}))
    atomic(source / 'manifest.json', files_manifest(source, [name, 'audit.json'], technical_synthetic=False))
    atomic(out / 'inputs/base.json', dict(model=str(tmp_path / 'base'), revision=contract.BASE_REVISION))
    monkeypatch.setattr(m3_pools, 'load', lambda *args: (POOLS, {'lengths': None}))
    monkeypatch.setattr(m3_train, 'tokenizer_for', lambda *args: tokenizer)
    monkeypatch.setattr(m3_train, 'schedule_binding', lambda *args: {'fixture': 'bound'})
    a = NS(out=out, arm=arm, seed=101, stage='contract_nli', standin=False, schedule_root=task_root, sdpo=tmp_path / 'sdpo',
           gpu='0,1,2,3,4,5,6,7', tokenizer='fixture', trainer_python='python', host_reservation_bytes=64*1024**3)
    partial = out / 'preflight-inputs' / arm / 'data/train.parquet'
    partial.parent.mkdir(parents=True); partial.write_bytes(b'interrupted-preparation')
    node = m3_train.prepare_preflight(a)
    archived = next((out / 'preflight' / arm / 'attempts').iterdir())
    assert (archived / 'data/train.parquet').read_bytes() == b'interrupted-preparation'
    assert json.loads((archived / 'attempt-manifest.json').read_text())['files']['data/train.parquet']
    path = node.parent / 'manifest.json'; raw = json.loads(path.read_text())
    assert raw['schema'] == contract.PREFLIGHT_INPUT_SCHEMA and 'longest_legal_prompt_tokens' not in raw
    admitted = contract.validate_preflight_inputs(path, arm=arm, stage='contract_nli', root=out)
    assert admitted['longest_legal_prompt_tokens'] == len(pids)
    converted = pq.read_table(node.parent / raw['train_files'][0]).to_pylist()
    assert len(converted) == 4 * contract.geometry(arm, 'contract_nli')['global_rows']
    assert {r['extra_info']['m3_update'] for r in converted} == {1, 2, 3, 4}
    for r in converted:
        extra = r['extra_info']; assert 1 <= extra['m3_position'] <= contract.geometry(arm, 'contract_nli')['global_rows']
        if not extra['m3_replay']:
            assert extra['m3_domain'] == 'contract_nli' and r['data_source'] == 'contract_nli' and r['id'] == legal[0]['id']
    argv = json.loads(node.read_text())['command']; config, overrides = m3_entry.parse_args(argv[3:])
    assert config['profile'] == 'preflight' and config['stage'] == 'contract_nli'
    keys = [m3_train.override_key(v) for v in m3_entry.hydra_arguments(config, overrides)]
    assert len(keys) == len(set(keys))
    assert m3_train.prepare_preflight(a) == node


def test_manifest_preflight_and_coverage_dependencies_are_executable(tmp_path):
    doc = manifest.build(settings(tmp_path)); tasks = {t['id']: t for t in doc['tasks']}
    for arm in plan.ARMS:
        job = tasks['M3-preflight-' + arm]
        assert job['uses_gpu'] and job['kind'] == 'preflight' and 'M3-math-targets' not in closure(tasks, job['id'])
        assert job['commands'][0][job['commands'][0].index('--stage') + 1] == 'contract_nli'
        for seed in plan.SEEDS:
            assert job['id'] in closure(tasks, f'M3-{arm}-{seed}-finqa-train')
    assert 'M3-schedules' in tasks['M3-schedules-coverage']['requires']
    assert 'M3-schedules-math' in tasks['M3-schedules-coverage-math']['requires']


def test_precheckpoint_restart_archives_native_rollouts_and_hashes(tmp_path):
    from kit import m3_train
    from kit.p3b_common import atomic, frozen_files
    run = tmp_path / 'runs/C0-s101-finqa'
    atomic(run / 'started.json', {'started': True})
    atomic(run / 'training-costs/failed.json', {'kind': 'm3_training', 'gpu_hours': 1})
    for name in ('sdft/partial.pt', 'rollouts/part.json', 'validation/part.json'):
        path = run / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'interrupted-evidence')
    assert m3_train.training_action(run, 'C0', 101, 'finqa') == ('start', None)
    archive = next((run / 'attempts').iterdir()); evidence = frozen_files(archive)
    assert set(evidence['files']) == {'started.json', 'sdft/partial.pt', 'rollouts/part.json', 'validation/part.json'}
    assert (run / 'training-costs/failed.json').exists() and not (run / 'started.json').exists()
    assert m3_train.training_action(run, 'C0', 101, 'finqa') == ('start', None)
    assert len(list((run / 'attempts').iterdir())) == 1


def test_missing_step40_receipt_is_pending_without_repeating_training(tmp_path):
    from kit import m3_train
    run = tmp_path / 'runs/K-s101-finqa'; path = _native_fixture(run, 40)
    with pytest.raises(Exception, match='receipt-only native resume'): m3_train.training_action(run, 'K', 101, 'finqa')
    assert path.exists() and not (run / 'attempts').exists()


def test_transient_likelihood_error_has_retryable_attempt_instead_of_final_record(tmp_path, monkeypatch):
    from kit import m3_score
    out = _standin_root(tmp_path, monkeypatch); _fake_run(out, 'C0', 101, 'finqa')
    original = m3_score.standin_likelihoods; calls = []
    def transient(rows, salt):
        calls.append(salt)
        if len(calls) == 1: raise RuntimeError('temporary inference failure')
        return original(rows, salt)
    monkeypatch.setattr(m3_score, 'standin_likelihoods', transient)
    a = NS(out=out, gpu='0', seed=101, standin=True)
    with pytest.raises(RuntimeError, match='temporary inference'): m3_score.validate_knowledge(a)
    path = m3_score.validation_path(out, 'knowledge-ll', 'm3', 101)
    assert not path.exists()
    attempts = list((path.parent / 'attempts' / path.stem).glob('*.json'))
    assert len(attempts) == 1 and json.loads(attempts[0].read_text())['status'] == 'ERROR'
    m3_score.validate_knowledge(a)
    assert json.loads(path.read_text())['status'] == 'PASS'


def test_genuine_likelihood_disagreement_is_frozen_pending_and_rerun_forbidden(tmp_path, monkeypatch):
    """1 Nov amendment, 'Likelihood-scorer validation: near-tie ruling': large gaps stay PENDING."""
    from kit import m3_score
    out = _standin_root(tmp_path, monkeypatch); _fake_run(out, 'C0', 101, 'finqa')
    proof = m3_score.proof_for(out, '0', 101, True)
    monkeypatch.setattr(m3_score, 'proof_for', lambda *args: proof)
    monkeypatch.setattr(m3_score, 'resolved_gpu', lambda *args: '0')
    monkeypatch.setattr(m3_score, 'checked', lambda *args: None)
    rows = m3_score.knowledge_items()
    production = m3_score.standin_likelihoods(rows, 'fixture')
    reference = json.loads(json.dumps(production)); selected = m3_score.selected(reference[0]['scores'])
    reference[0]['scores'][(selected + 1) % 4] = 1e9
    a = NS(out=out, gpu='0', seed=101, standin=False)
    call = lambda: m3_score.validate(a, 'knowledge-ll', rows, lambda *args: production, lambda *args: reference,
                                    m3_score.s.file_hash(m3_score.KNOWLEDGE_POLICY), {})
    with pytest.raises(Exception, match='disagreement PENDING'): call()
    path = m3_score.validation_path(out, 'knowledge-ll', 'm3', 101)
    before = path.read_bytes(); assert json.loads(before)['status'] == 'PENDING'
    with pytest.raises(Exception, match='rerun forbidden'): call()
    assert path.read_bytes() == before


def test_skipped_draw_count_is_computed_from_actual_stream_indices(monkeypatch):
    original = plan.shared_stream
    def filtered(pool, seed): return [dict(row, draw_index=row['draw_index'] + 1) for row in original(pool, seed)]
    monkeypatch.setattr(plan, 'shared_stream', filtered)
    schedules = {}; summaries = {}
    for arm in plan.ARMS:
        schedules[arm], summaries[arm] = plan.build(arm, 101, 'contract_nli', task_rows('contract_nli'), POOLS)
        assert summaries[arm]['skipped_draw_count'] == 1 and summaries[arm]['skipped_draw_count_by_domain']['instruction'] == 1
    with pytest.raises(plan.ScheduleError, match='filtered draws'): plan.audit_arms(schedules, summaries)


def test_coverage_pass_audits_all_four_frozen_arm_schedules(tmp_path, monkeypatch):
    pytest.importorskip('rapidfuzz')
    from kit import m3_pools, m3_train
    from kit.p3b_common import atomic, s
    a = _freeze_inputs(tmp_path, FULL); a.shared_only = True; m3_pools.freeze_job(a)
    task_root = a.out / 'targets-finqa'; m3_pools.task_root(task_root, task_rows('finqa')[:100], {}, True)
    args = NS(out=a.out, standin=True, stages=['finqa'], schedule_roots=[task_root], tokenizer=None, arms=['C0', 'K'])
    m3_train.schedules_job(args)
    a.shared_only = False; m3_pools.freeze_job(a)
    original = plan.audit_arms; calls = []
    def audit(schedules, summaries=None): calls.append(set(schedules)); return original(schedules, summaries)
    monkeypatch.setattr(plan, 'audit_arms', audit)
    args.arms = ['G', 'CR']; m3_train.schedules_job(args)
    assert calls == [set(plan.ARMS)] * 3
    root = m3_train.schedule_dir(a.out, 'finqa', 'G')
    assert all(len(value['shared_ids']) == 160 for value in json.loads((root / 'audit.json').read_text())['audits'].values())
    # A self-consistent but divergent frozen K schedule must fail the production resume audit.
    shared = m3_train.schedule_dir(a.out, 'finqa', 'K'); path = m3_train.schedule_path(a.out, 'K', 101, 'finqa')
    rows = m3_pools.t.read_jsonl(path); next(r for r in rows if r['role'] == 'replay')['id'] = 'foreign'
    m3_pools.t.write_jsonl(path, rows)
    doc = json.loads((shared / 'manifest.json').read_text()); doc['files']['K/s101.jsonl'] = s.file_hash(path); atomic(shared / 'manifest.json', doc)
    with pytest.raises(plan.ScheduleError, match='generic IDs'): m3_train.schedules_job(args)


def test_hostname_normalization_wraps_socket_and_platform_inventory(tmp_path, monkeypatch):
    from kit import m3_score, m2_assignment
    assert manifest.runtime_node(dict(nodes=['allowed'], message2_nodes=[]), 'allowed.cluster') == 'allowed'
    assert manifest.runtime_node(dict(nodes=['core.cluster'], message2_nodes=['core']), 'core.cluster') == 'core'
    assert manifest.build(settings(tmp_path, nodes=['core-node.cluster']))
    out = _standin_root(tmp_path, monkeypatch)
    original = m2_assignment.inventory
    monkeypatch.setattr(m2_assignment, 'inventory', lambda *args: dict(node='CPU-STANDIN.cluster', uuid='CPU-STANDIN', environment='CPU-STANDIN'))
    before = m2_assignment.inventory
    assert m3_score.proof_for(out, '0', 101, True)['node'] == 'CPU-STANDIN'
    assert m2_assignment.inventory is before


def test_padded_token_repricing_exposes_manager_calibration_without_budget_changes():
    updates = [{'m3/actor_padded_tokens': 2e6, 'm3/ema_padded_tokens': 3e6, 'm3/base_padded_tokens': 4e6, 'm3/rollout_padded_tokens': 5e6}]
    rates = dict(actor=1, ema=2, base=3, rollout=4)
    before = reader.registered_costs()
    result = reader.padded_token_cost(updates, rates)
    assert result['update_gpu_hours'] == 40 and reader.registered_costs() == before
    with pytest.raises(ValueError): reader.padded_token_cost([{}], rates)
    with pytest.raises(ValueError): reader.padded_token_cost(updates, dict(rates, base=float('nan')))


def test_returned_input_superset_verifies_actual_parquets_and_replay(native_b_finqa, tmp_path):
    from kit import m3_pools
    from kit.p3b_common import atomic, files_manifest, t
    import pyarrow.parquet as pq
    folders, _ = native_b_finqa
    replay = tmp_path / 'replay'; replay.mkdir(); instruction = rows('instruction', 160)
    t.write_jsonl(replay / 'replay.jsonl', instruction)
    atomic(replay / 'manifest.json', files_manifest(replay, ['replay.jsonl']))
    targets = tmp_path / 'c-targets'; legal = rows('legal', 100); m3_pools.task_root(targets, legal, {}, False)
    finqa = pq.read_table(folders[0] / 'train.parquet').to_pylist()
    public = tmp_path / 'superset.json'
    public.write_text(json.dumps(m3_pools.superset(dict(instruction=[r['prompt'] for r in instruction], finqa=[r['prompt'] for r in finqa],
        **{'task-finqa': [r['prompt'] for r in finqa], 'contract_nli': [r['prompt'] for r in legal], 'task-contract_nli': [r['prompt'] for r in legal]}))))
    a = NS(out=tmp_path / 'report', output=None, finqa_targets=folders, replay=replay, targets=targets, superset=public)
    assert m3_pools.verify_returned_superset(a)['status'] == 'VERIFIED'
    report = a.out / 'returned-superset-verification.json'
    assert json.loads(report.read_text())['inputs']['finqa_0']['train_sha256']
    original = (folders[0] / 'train.parquet').read_bytes()
    (folders[0] / 'train.parquet').write_bytes(b'corrupted-return')
    with pytest.raises(Exception, match='parquet hash differs'): m3_pools.verify_returned_superset(a)
    assert json.loads(report.read_text())['status'] == 'UNVERIFIED'
    (folders[0] / 'train.parquet').write_bytes(original)
    doc = json.loads(public.read_text()); doc['pools']['task-finqa'] = []; public.write_text(json.dumps(doc))
    with pytest.raises(Exception, match='universe mismatch'): m3_pools.verify_returned_superset(a)
    assert json.loads(report.read_text())['status'] == 'UNVERIFIED'
    a.finqa_targets = None
    with pytest.raises(Exception, match='returned B/C inputs required'): m3_pools.verify_returned_superset(a)
    assert json.loads(report.read_text())['status'] == 'PENDING'
    a.finqa_targets = folders; public.unlink()
    with pytest.raises(Exception, match='returned B/C inputs required'): m3_pools.verify_returned_superset(a)
    assert json.loads(report.read_text())['status'] == 'PENDING'


def test_legacy_pending_likelihood_record_is_archived_and_retryable(tmp_path, monkeypatch):
    from kit import m3_score
    out = _standin_root(tmp_path, monkeypatch); _fake_run(out, 'C0', 101, 'finqa')
    a = NS(out=out, gpu='0', seed=101, standin=True); m3_score.validate_knowledge(a)
    path = m3_score.validation_path(out, 'knowledge-ll', 'm3', 101)
    value = json.loads(path.read_text()); value['status'] = 'PENDING'; path.write_text(json.dumps(value))
    m3_score.validate_knowledge(a)
    assert json.loads(path.read_text())['status'] == 'PASS'
    assert len(list((path.parent / 'attempts' / path.stem).glob('legacy-*.json'))) == 1


def test_preflight_action_uses_gpu_execution_before_running_producer(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from kit import m3_train
    out = tmp_path / 'm3'; manifest.init_root(out)
    events = []
    @contextmanager
    def reservation(root, gpus):
        assert gpus == 8
        events.append('reserved'); yield; events.append('released')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    monkeypatch.setattr(manifest, 'gpu_execution', reservation)
    monkeypatch.setattr(m3_train, 'preflight_job', lambda a: events.append('preflight'))
    m3_train.job(NS(out=out, standin=False, action='preflight'))
    assert events == ['reserved', 'preflight', 'released']


def test_legacy_pending_with_real_disagreement_cannot_be_retried(tmp_path, monkeypatch):
    """1 Nov amendment, 'Likelihood-scorer validation: near-tie ruling': terminal PENDING, no retry.

    Reconstruct the original 100%-agreement receipt schema with internally
    consistent counters, instead of mutating raw rows beneath a newer receipt.
    """
    from kit import m3_score,m2_format,likelihood_validation as validation
    out = _standin_root(tmp_path, monkeypatch); _fake_run(out, 'C0', 101, 'finqa')
    a = NS(out=out, gpu='0', seed=101, standin=True); m3_score.validate_knowledge(a)
    path = m3_score.validation_path(out, 'knowledge-ll', 'm3', 101)
    value = json.loads(path.read_text()); value['status'] = 'PENDING';value.pop('validation_rule')
    reference = value['checks'][0]['reference'][0]['scores']; reference[(m3_score.selected(reference) + 1) % 4] = 1e9
    ids=[r['id'] for r in m3_score.knowledge_items()]
    for c in value['checks']:
        outcome=m2_format.validation(c['production'],c['reference'],ids)
        for k in set(c)-set(validation.LEGACY_FIELDS)-{'production','reference','checkpoint','status'}:c.pop(k)
        for k in validation.LEGACY_FIELDS:c[k]=outcome[k]
        c['status']='PASS' if c['selected_choice_agreement']==100 else 'FAIL'
    path.write_text(json.dumps(value));original=path.read_bytes()
    with pytest.raises(Exception, match='legacy reference disagreement PENDING'): m3_score.validate_knowledge(a)
    assert json.loads(path.read_text())['status'] == 'PENDING'
    assert any(p.read_bytes()==original for p in (path.parent/'attempts'/path.stem).glob('*.json'))


def test_preflight_forwards_offline_runtime_to_children_and_restores_environment(tmp_path, monkeypatch):
    import os
    from kit import m3_train, m3_smoke
    out = tmp_path / 'm3'; node = out / 'preflight-inputs/K/node-plan.json'
    node.parent.mkdir(parents=True); node.write_text('{}')
    monkeypatch.setattr(m3_train.contract, 'validate_preflight_inputs', lambda *a, **kw: {})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'before')
    def execute(path):
        assert path == node and os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'
        assert os.environ['HF_HUB_OFFLINE'] == '1' and str(tmp_path / 'sdpo') in os.environ['PYTHONPATH']
        return {'status': 'CPU-FIXTURE'}
    monkeypatch.setattr(m3_smoke, 'run_preflight', execute)
    a = NS(out=out, arm='K', stage='contract_nli', gpu='0,1,2,3,4,5,6,7', sdpo=tmp_path / 'sdpo', tokenizer='fixture')
    assert m3_train.preflight_job(a)['status'] == 'CPU-FIXTURE'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'before'
