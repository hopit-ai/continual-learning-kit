"""Mixed D regressions, executing pinned verl bodies when its runtime is absent."""
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import json
import random
import sys
import types

import numpy as np
import pytest
import torch

from kit import p3b_common as c, p3b_train as train, p3b_reward as reward


ROOT = Path(__file__).resolve().parents[1]
SDPO = next((p / 'references/SDPO' for p in (ROOT, *ROOT.parents)
             if (p / 'references/SDPO/verl/protocol.py').is_file()), None)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = '!'

    def encode(self, text, **kwargs): return [1] * len(text)
    def __call__(self, text, **kwargs): return {'input_ids': self.encode(text)}
    def apply_chat_template(self, messages, **kwargs): return ' '.join(m['content'] for m in messages)
    def decode(self, ids, **kwargs): return 'Answer: 2' if int(ids[0]) == 3 else r'\boxed{2}'


def _body(path, name, methods=None):
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, 'name', '') == name)
    node.decorator_list = []
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.AnnAssign) or getattr(n, 'name', '') in methods]
        node.decorator_list = [ast.Name(id='dataclass', ctx=ast.Load())]
    return node


@pytest.fixture
def verl_path(monkeypatch):
    """Real NaiveRewardManager + compute_reward + DataProto methods, no copied aggregation.

    Only TensorDict storage and unrelated dependency imports are substituted.
    The constructor, consistency check, indexing, split and chunk bodies are verl's.
    """
    if SDPO is None: pytest.skip('pinned SDPO source checkout unavailable')
    monkeypatch.setenv('SDPO_DIR', str(SDPO))
    monkeypatch.setenv('KIT_FINISH_GATE', '1')
    monkeypatch.setenv('P3B_REPLAY', '1')

    class TensorDict(dict):
        def __init__(self, source, batch_size=None, device=None):
            super().__init__(source)
            self.batch_size = (next(iter(source.values())).shape[0],) if batch_size is None else batch_size
            self.device = next(iter(source.values())).device
        def __getitem__(self, key):
            if isinstance(key, str): return super().__getitem__(key)
            return TensorDict({k: v[key] for k, v in self.items()})
        def chunk(self, chunks, dim):
            parts = {k: v.chunk(chunks, dim) for k, v in self.items()}
            return [TensorDict({k: v[i] for k, v in parts.items()}) for i in range(chunks)]

    ns = dict(np=np, torch=torch, dataclass=dataclass, field=field, TensorDict=TensorDict,
              DataProtoItem=types.SimpleNamespace, defaultdict=defaultdict,
              AbstractRewardManager=type('AbstractRewardManager', (), {'_extract_reward_from_rm_scores': lambda *a: None}))
    proto = _body(SDPO / 'verl/protocol.py', 'DataProto',
                  {'__post_init__', '__len__', '__getitem__', 'check_consistency', 'from_dict', 'slice', 'select_idxs', 'split', 'chunk'})
    # Retain from_dict's classmethod decorator; auto padding is disabled in these tests.
    manager = _body(SDPO / 'verl/workers/reward_manager/naive.py', 'NaiveRewardManager')
    compute = _body(SDPO / 'verl/trainer/ppo/reward.py', 'compute_reward')
    metrics = [_body(SDPO / 'verl/trainer/ppo/metric_utils.py', name)
               for name in ('_compute_response_info', 'compute_data_metrics')]
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                           proto, manager, compute, *metrics], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), '<pinned verl reward and DataProto>', 'exec'), ns)
    ns['DataProto'].is_padding_enabled = lambda self: False
    return types.SimpleNamespace(**ns)


def _batch(v, source):
    # Replay-only final DP shard reproduces acc length 0 vs batch size 4.
    sources = [source] * 28 + ['p3b_replay'] * 4
    responses = torch.tensor([[3 if source == 'finqa' else 4, 2]] * 32)
    responses[1::2, -1] = 1  # finished and truncated tasks/replay
    return v.DataProto.from_dict(
        tensors=dict(prompts=torch.ones(32, 2, dtype=torch.long), responses=responses,
                     attention_mask=torch.ones(32, 4, dtype=torch.long)),
        non_tensors=dict(data_source=sources, reward_model=[{'ground_truth': '2'} for _ in sources],
                         extra_info=[dict(p3b_task_source=source, p3b_replay=s == 'p3b_replay') for s in sources]))


@pytest.mark.parametrize('source', ['finqa', 'math_screen'])
def test_real_reward_aggregation_and_dataproto_split(verl_path, source):
    v = verl_path
    batch = _batch(v, source)
    manager = v.NaiveRewardManager(Tokenizer(), 0, compute_score=reward.compute_score)
    scores, extras = v.compute_reward(batch, manager)
    batch.non_tensor_batch.update({k: np.array(values) for k, values in extras.items()})
    shards = batch.chunk(8)
    assert len(shards) == 8 and all(len(s) == 4 for s in shards)
    assert all(len(m) == 1 for shard in shards for m in shard.split(1))
    assert all(len(values) == 32 for values in extras.values())
    assert extras['p3b_replay'] == [0] * 28 + [1] * 4
    assert scores[28:].eq(0).all()


@pytest.mark.parametrize('seed', range(101, 106))
def test_over_cap_replay_skips_in_original_order(seed):
    lengths = [2049, 2048, 4, 4, 4, 3714]
    pool = [dict(id=str(i), prompt='x' * length,
                 response='ok', p3b_replay=True) for i, length in enumerate(lengths)]
    task = [dict(id=f't{i}', prompt='task', response='worked solution', data_source='finqa') for i in range(1280)]
    original = list(pool); random.Random(seed).shuffle(original)
    f = c.mixed(task, pool, seed)
    assert [r['id'] for r in f if r.get('p3b_replay')] == [original[i % 6]['id'] for i in range(160)]
    d, audit = train.d_replay_rows(task, pool, seed, Tokenizer())
    expected = [r for r in original if r['id'] not in ('0', '5')]
    assert [r['id'] for r in d if r.get('p3b_replay')] == [expected[i % 4]['id'] for i in range(160)]
    assert [r for r in d if not r.get('p3b_replay')] == [r for r in f if not r.get('p3b_replay')]
    assert audit['eligible_replay_rows'] == 4 and audit['skipped_ids'] == ['0', '5']
    assert audit['skipped'] == [dict(id='0', prompt_tokens=2049), dict(id='5', prompt_tokens=3714)]
    assert audit['skipped_draw_count'] == len(audit['skipped_draws']) > 0
    assert all(original[s['draw_index'] % 6]['id'] == s['id'] and s['id'] in ('0', '5')
               for s in audit['skipped_draws'])
    assert audit['skipped_draw_count'] + 160 == audit['skipped_draws'][-1]['draw_index'] + 1 + (
        160 - audit['skipped_draws'][-1]['replay_slot'])
    train.training_rows(d, Tokenizer(), 'D')
    # The unchanged guard continues to refuse task prompts, and unscreened replay.
    with pytest.raises(c.CorrectnessStop, match='D prompt exceeds'):
        train.training_rows([dict(task[0], prompt='x' * 2049)], Tokenizer(), 'D')
    with pytest.raises(c.CorrectnessStop, match='D prompt exceeds'):
        train.training_rows(c.cycle(pool, seed, 160), Tokenizer(), 'D')


@pytest.mark.parametrize('recipe', ['D+replay', 'F+replay'])
@pytest.mark.parametrize('route', ['finqa', 'standalone', 'sequence', 'reset'])
def test_job_writes_replay_audit(tmp_path, monkeypatch, route, recipe):
    from kit import v4_sft
    out = tmp_path / 'out'; replay = out / 'replay'; replay.mkdir(parents=True)
    c.t.write_jsonl(replay / 'replay.jsonl', [dict(id=str(i), prompt='x' * (2049 if i == 0 else 4), response='ok') for i in range(5)])
    c.atomic(replay / 'manifest.json', c.files_manifest(replay, ['replay.jsonl'], technical_synthetic=False))
    task = [dict(id=f't{i}', prompt='task', response='worked solution', data_source='finqa',
                 extra_info={'index':f't{i}'}) for i in range(1280)]
    if route != 'finqa':
        for r in task:
            r.pop('data_source'); r['task'] = 'math_screen'
    targets = out / 'targets'; targets.mkdir()
    schedule = targets / 'schedule-101.jsonl'; c.t.write_jsonl(schedule, task)
    manifest = targets / 'manifest.json'
    c.atomic(manifest, c.files_manifest(targets, [schedule.name], technical_synthetic=False,
        schema='kit-v4-sft-data.v1', arm='COMMON', scientific_phase_allowed=True,
        ids=[r['id'] for r in task], seed=101))
    c.atomic(targets / 'inputs.json', dict(task='math_screen'))
    receipt = tmp_path / 'receipt.json'
    c.atomic(receipt, dict(model_dir=f'/finqa-{recipe}-s101/hf-step40' if route == 'sequence' else f'/{recipe}-s101/hf-reset'))
    run = out / 'runs' / f'{route}-{recipe}-s101'; run.mkdir(parents=True)
    c.atomic(run / 'trained.json', dict(seconds=1))
    args = types.SimpleNamespace(out=out, seed=101, recipe=recipe, route=route,
        gpu='0,1,2,3,4,5,6,7', standin=False, sdpo=tmp_path, tokenizer='fake', model='fake',
        incoming_receipt=receipt if route in ('sequence', 'reset') else None,
        task_schedule=schedule, schedule_manifest=manifest, trainer_python=sys.executable)
    monkeypatch.setattr(train, 'frozen_hashes', lambda: None)
    monkeypatch.setattr(train, 'identity', lambda *a, **k: {})
    monkeypatch.setattr(train.subprocess, 'check_output', lambda *a, **k: '7c457fc1b1f636ae794eb0362ba37d4743b06fbc')
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())))
    pa = types.ModuleType('pyarrow'); pa.Table = types.SimpleNamespace(from_pylist=lambda values: values)
    pq = types.ModuleType('pyarrow.parquet'); pq.write_table = lambda values, path: path.write_text(json.dumps(values))
    monkeypatch.setitem(sys.modules, 'pyarrow', pa); monkeypatch.setitem(sys.modules, 'pyarrow.parquet', pq)
    monkeypatch.setattr(v4_sft, 'complete_export', lambda path: True)
    train.job(args)
    if recipe == 'F+replay':
        # Real F conversion retains the over-cap replay row, full-pool cycle and
        # ordinary F entry point on every route, including MATH sequence/reset.
        assert not (run / 'replay-selection.json').exists()
        rows = json.loads((out / 'data' / run.name / 'train.parquet').read_text())
        replay_rows = [r for r in rows if r['extra_info']['p3b_replay']]
        assert len(replay_rows) == 160
        assert [r['id'] for r in replay_rows] == [r['id'] for r in c.cycle(c.t.read_jsonl(replay / 'replay.jsonl'), 101, 160)]
        assert any(len(r['prompt'][0]['content']) == 2049 for r in replay_rows)
        assert all(r['extra_info']['p3b_replay'] == (i % 32 in (7, 15, 23, 31)) for i,r in enumerate(rows))
        assert str(c.KIT / 'sft_entry.py') in json.loads((run / 'command.json').read_text())
        train.job(args)
        return
    audit = json.loads((run / 'replay-selection.json').read_text())
    saved = c.frozen_files(run)
    assert saved['replay_selection'] == audit
    assert saved['replay_selection']['skipped_ids'] == ['0']
    assert audit['skipped'] == [dict(id='0', prompt_tokens=2049)]
    assert saved['files']['replay-selection.json'] == c.s.file_hash(run / 'replay-selection.json')
    assert json.loads((out / 'identities' / (run.name + '.json')).read_text())['replay_selection'] == audit
    assert json.loads((run / 'summary.json').read_text())['replay_selection'] == audit
    assert audit['seed'] == 101 and audit['skipped_draw_count'] > 0
    rows = json.loads((out / 'data' / run.name / 'train.parquet').read_text())
    assert all(r['extra_info']['p3b_task_source'] == ('finqa' if route == 'finqa' else 'math_screen')
               for r in rows if r['extra_info']['p3b_replay'])
    # A retry with the same inputs binds the same audit and does not select again.
    train.job(args)


@pytest.mark.parametrize('source', ['finqa', 'math_screen', 'livecodebench'])
@pytest.mark.parametrize('gate', ['0', '1'])
@pytest.mark.parametrize('truncated', [False, True])
def test_reward_schemas_and_non_replay_bytes(verl_path, tmp_path, monkeypatch, source, gate, truncated):
    from kit.beds import code, math_screen, v4_reward
    monkeypatch.setenv('KIT_FINISH_GATE', gate)
    gold = '2'
    if source == 'livecodebench':
        tests = [dict(input='', output='', testtype='stdin')]
        digest = code.tests_digest(tests)
        file = tmp_path / 'tests.jsonl'
        c.t.write_jsonl(file, [dict(problem_id='p', tests=tests, tests_sha256=digest, testtype='stdin', fn_name=None)])
        monkeypatch.setenv('CODE_TESTS', str(file))
        gold = json.dumps(dict(problem_id='p', tests_sha256=digest))
    extra = dict(truncated=truncated, p3b_task_source=source)
    text = 'Answer: 2' if source == 'finqa' else r'\boxed{2}'
    task = reward.compute_score(source, text, gold, extra)
    replay = reward.compute_score('p3b_replay', text, '', extra)
    assert task.keys() == replay.keys()
    assert task['p3b_replay'] == 0 and replay['p3b_replay'] == 1
    assert all(replay[k] == 0 for k in ('score', 'acc', 'incorrect_format'))
    assert replay['pred'] == ''
    if source == 'finqa':
        assert replay['truncated'] == int(truncated) and replay['score_before_gate'] == 0
        assert task['score'] == (0. if truncated and gate == '1' else 1.)
    # Unset replay flag calls exactly the old path, without a marker or schema edits.
    monkeypatch.delenv('P3B_REPLAY')
    scorer = {'finqa': v4_reward, 'math_screen': math_screen, 'livecodebench': code}[source]
    assert json.dumps(reward.compute_score(source, text, gold, extra)) == json.dumps(scorer.compute_score(source, text, gold, extra))


def test_training_reward_metrics_exclude_replay(verl_path):
    v = verl_path; batch = _batch(v, 'math_screen')
    batch.batch.update(token_level_scores=torch.ones(32, 2), token_level_rewards=torch.ones(32, 2),
                       advantages=torch.ones(32, 2), returns=torch.ones(32, 2), response_mask=torch.ones(32, 2))
    for key in ('token_level_scores', 'token_level_rewards', 'advantages', 'returns'):
        batch.batch[key][28:] = 100
    original = v.compute_data_metrics
    reward.install_metrics(v)
    assert v.compute_data_metrics is not original
    actual = v.compute_data_metrics(batch, use_critic=False)
    task = original(batch[:28], use_critic=False)
    assert {k:val for k,val in actual.items() if k.startswith('critic/')} == {k:val for k,val in task.items() if k.startswith('critic/')}
    assert actual['p3b/task_rows'] == 28 and actual['p3b/replay_rows'] == 4
    assert actual['response_length/mean'] == 2  # Workload lengths include every generated row.


def test_group_metrics_exclude_replay_and_teacher_padding(verl_path, monkeypatch):
    import os
    from kit import sdft_objective as objective
    v = verl_path; batch = _batch(v, 'finqa')
    batch.batch.update(input_ids=torch.cat((batch.batch['prompts'], batch.batch['responses']), -1),
                       response_mask=torch.ones(32, 2))
    batch.non_tensor_batch['uid'] = np.array([str(i) for i in range(32)])
    batch.non_tensor_batch['raw_prompt'] = np.array([[dict(role='user', content='question')]] * 32, dtype=object)
    for extra in batch.non_tensor_batch['extra_info']: extra['demonstration'] = 'worked solution'
    class Config(dict): __getattr__ = dict.__getitem__
    config = Config(v4_context_source='demonstration', data={}, actor_rollout_ref=Config(actor=Config(
        policy_loss={'loss_mode':'sdpo'}, self_distillation=types.SimpleNamespace(max_reprompt_len=6144, success_reward_threshold=1.))))
    ns = dict(objective=objective, os=os, OmegaConf=None, _SHOWN=True)
    monkeypatch.setitem(sys.modules, 'verl', types.SimpleNamespace(DataProto=v.DataProto))
    node = _body(ROOT / 'kit/sdft_verl.py', 'build_sdft_teacher_batch')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<kit teacher builder>', 'exec'), ns)
    scores = torch.ones(32, 2); scores[28:] = 0
    built, metrics = ns['build_sdft_teacher_batch'](types.SimpleNamespace(config=config, tokenizer=Tokenizer()), batch, scores)
    assert metrics['self_distillation/success_group_fraction'] == 1
    assert metrics['sdft/no_success_group_fraction'] == 0
    for key in ('teacher_input_ids', 'teacher_attention_mask', 'teacher_position_ids'):
        assert torch.equal(built.batch[key][0], built.batch[key][28])


def test_telemetry_counts_frozen_replay_labels(verl_path, monkeypatch):
    from kit.v4_timing import training_tokens
    batch = _batch(verl_path, 'math_screen')
    batch.batch.update(response_mask=torch.ones(32, 6), self_distillation_mask=torch.ones(32))
    for extra in batch.non_tensor_batch['extra_info']: extra['p3b_target_ids'] = [1, 1, 2]
    counted = training_tokens(batch)
    assert counted == dict(generated_trajectories=32, generated_tokens=192, supervised_tokens=96,
                           task_supervised_tokens=84, replay_supervised_tokens=12)
    monkeypatch.delenv('P3B_REPLAY')
    assert training_tokens(batch) == dict(generated_trajectories=32, generated_tokens=192, supervised_tokens=192)
