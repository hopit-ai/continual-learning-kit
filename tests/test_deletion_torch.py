"""CPU torch tests. Entire module skips when torch is unavailable."""
import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch', reason='CPU/GPU torch environment required')
from torch import nn
from deletion_tuning.deletion_mask import install, layers, set_step, update_teacher, hard_state, calibrate


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.gate_proj = nn.Linear(3, 4, bias=False)
        self.head = nn.Linear(4, 2, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return self.head(self.model.layers[0].mlp.gate_proj(input_ids).tanh())


class Causal(Tiny):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 3)
        self.head = nn.Linear(4, 7, bias=False)

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=super().forward(self.embedding(input_ids)))


def test_soft_math_gradient_hard_export_and_immutable_bank():
    torch.manual_seed(101)
    model = Tiny()
    dense = copy.deepcopy(model)
    before = {n: p.clone() for n, p in model.named_parameters()}
    install(model)
    layer = next(iter(layers(model).values()))
    x = torch.randn(2, 3)
    model.train()
    logits = layer.diff_mask.gate
    expected = nn.functional.linear(x, layer.weight * torch.sigmoid(100 * logits))
    assert torch.equal(layer(x), expected)
    model(x).sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in model.named_parameters() if n in before)
    opt = torch.optim.AdamW([logits], lr=1e-5, weight_decay=.01)
    opt.step()
    assert all(torch.equal(dict(model.named_parameters())[n], p) for n, p in before.items())
    dense.load_state_dict(hard_state(model.state_dict()), strict=True)
    model.eval()
    assert torch.equal(model(x), dense(x))
    assert all('diff_mask' not in n for n in hard_state(model.state_dict()))


def test_named_logit_only_ema_exact_and_teacher_hard():
    model = install(Tiny())
    teacher = copy.deepcopy(model)
    before = {n: p.clone() for n, p in teacher.named_parameters()}
    with torch.no_grad():
        for module in layers(model).values():
            module.diff_mask.gate.add_(.01)
    update_teacher(model, teacher)
    for name, parameter in teacher.named_parameters():
        expected = before[name]
        if name.endswith('.diff_mask.gate'):
            expected = expected * .99 + dict(model.named_parameters())[name] * .01
        assert torch.allclose(parameter, expected, atol=1e-8, rtol=0)
    assert not teacher.training


def test_wanda_matches_independent_norm_and_prior():
    torch.manual_seed(101)
    model = Tiny()
    x = torch.randn(8, 3)
    # Reuse seed to make the independent initial G identical.
    state = torch.get_rng_state()
    reference = copy.deepcopy(model)
    install(reference)
    g = next(iter(layers(reference).values())).diff_mask.gate.detach().clone()
    torch.set_rng_state(state)
    weight = model.model.layers[0].mlp.gate_proj.weight.detach().clone()
    scores = weight.abs() * (2 * x.square().sum(0).sqrt())[None]
    threshold = scores.flatten().kthvalue(3).values
    expected = g + torch.where(scores < threshold, -1., 1.) * g.std() * 3
    calibrate(model, [dict(input_ids=x), dict(input_ids=x)], eta=.25)
    actual = next(iter(layers(model).values())).diff_mask.gate
    assert torch.allclose(actual, expected, atol=1e-8, rtol=0)


def test_resume_model_optimizer_teacher_rng_and_ramp_exact():
    torch.manual_seed(101)
    initial = install(Tiny())
    def branch(resume):
        model = copy.deepcopy(initial)
        teacher = copy.deepcopy(initial)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        for step in range(4):
            set_step(model, step)
            opt.zero_grad()
            model(torch.ones(2, 3)).square().sum().backward()
            opt.step()
            update_teacher(model, teacher)
            if resume and step == 1:
                state = copy.deepcopy((model.state_dict(), teacher.state_dict(), opt.state_dict()))
                model = copy.deepcopy(initial)
                teacher = copy.deepcopy(initial)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
                model.load_state_dict(state[0])
                teacher.load_state_dict(state[1])
                opt.load_state_dict(state[2])
        return model, teacher, opt
    full, ft, fo = branch(False)
    resumed, rt, ro = branch(True)
    for a, b in ((full, resumed), (ft, rt)):
        assert all(torch.equal(v, b.state_dict()[n]) for n, v in a.state_dict().items())
    assert fo.param_groups[0]['lr'] == ro.param_groups[0]['lr']
    assert next(iter(layers(resumed).values())).diff_mask.current_step == 3


def test_refuses_bias_and_missing_gate_inventory():
    model = Tiny()
    model.model.layers[0].mlp.gate_proj = nn.Linear(3, 4, bias=True)
    with pytest.raises(ValueError):
        install(model)
    model = Tiny()
    model.tied_alias = model.model.layers[0].mlp.gate_proj.weight
    with pytest.raises(ValueError, match='tied'):
        install(model)
    with pytest.raises(ValueError):
        hard_state({'weight': torch.ones(2, 2)})


@pytest.mark.parametrize('finite', [True, False])
def test_actual_d_mixed_update_keeps_recipe_and_frozen_weights(monkeypatch, finite):
    from deletion_tuning.deletion_actor import update_policy
    from deletion_tuning.deletion_smoke import Batch

    monkeypatch.setenv('P3B_REPLAY', '1')
    torch.manual_seed(101)
    model = install(Causal())
    teacher = copy.deepcopy(model)
    teacher_before = {n: p.clone() for n, p in teacher.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    config = SimpleNamespace(self_distillation=dict(full_logit_distillation=True, alpha=0.,
        teacher_regularization='ema', teacher_update_rate=.01), loss_agg_mode='seq-mean-token-mean', ppo_epochs=1, ppo_mini_batch_size=32)
    owner = SimpleNamespace(actor_module=model, teacher_module=teacher, actor_optimizer=optimizer,
        config=config, ulysses_sequence_parallel_size=1, scaler=None)
    gradients = []
    def step():
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        gradients.append(float(norm))
        if finite:
            optimizer.step()
        return norm if finite else torch.tensor(float('nan'))
    owner._optimizer_step = step
    ids = torch.randint(0, 7, (32, 12))
    values = dict(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=torch.arange(12).expand(32, -1),
        responses=ids[:, -8:], response_mask=torch.ones(32, 8), old_log_probs=torch.zeros(32, 8),
        rollout_log_probs=torch.zeros(32, 8), teacher_input_ids=ids.clone(),
        teacher_attention_mask=torch.ones_like(ids), teacher_position_ids=torch.arange(12).expand(32, -1),
        self_distillation_mask=torch.ones(32))
    values['teacher_input_ids'][:, 1] = 3
    extra = [dict(p3b_replay=i >= 28, p3b_target_ids=[1, 2, 3]) for i in range(32)]
    before = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    from deletion_tuning.deletion_lr import training_loss
    probe_loss = training_loss(owner, Batch(values, extra))
    result = update_policy(owner, Batch(values, extra))
    assert probe_loss == pytest.approx(result['p3b/combined_loss'], abs=1e-7)
    assert result['v4/completed_optimizer_updates'] == int(finite)
    assert result['p3b/replay_tokens'] == 12
    assert result['p3b/task_response_tokens'] == 28 * 8
    assert gradients[0] > 0
    assert all(torch.equal(dict(model.named_parameters())[n], p) for n, p in before.items())
    assert all(torch.equal(dict(teacher.named_parameters())[n], p) for n, p in before.items())
    for name, parameter in teacher.named_parameters():
        if name.endswith('.diff_mask.gate'):
            expected = teacher_before[name].clone()
            if finite:
                expected.mul_(.99).add_(dict(model.named_parameters())[name], alpha=.01)
            assert torch.equal(parameter, expected)
    assert '_update_teacher' not in owner.__dict__
    assert next(iter(layers(model).values())).diff_mask.current_step == 0
    assert not teacher.training


def test_all_ones_mask_matches_dense_outputs():
    model = Tiny()
    dense = copy.deepcopy(model)
    install(model)
    with torch.no_grad():
        for module in layers(model).values():
            module.diff_mask.gate.fill_(1.)
    x = torch.randn(2, 3)
    assert torch.equal(model(x), dense(x))
    model.eval()
    assert torch.equal(model(x), dense(x))


def test_worker_hooks_install_before_sharding_and_filter_optimizer(monkeypatch):
    import sys
    import types
    from deletion_tuning import deletion_worker
    worker = types.ModuleType('verl.workers.fsdp_workers')
    events = []
    def shard(model, kwargs, config):
        assert layers(model)
        assert all(not p.requires_grad for n, p in model.named_parameters() if not n.endswith('.diff_mask.gate'))
        events.append('shard')
    worker.apply_fsdp2 = shard
    worker.fsdp2_load_full_state_dict = lambda model, state: state
    worker.convert_weight_keys = lambda state, model: state
    worker.build_optimizer = lambda params, config: list(params)
    verl = types.ModuleType('verl')
    workers = types.ModuleType('verl.workers')
    verl.workers = workers
    workers.fsdp_workers = worker
    for name, module in [('verl', verl), ('verl.workers', workers), ('verl.workers.fsdp_workers', worker)]:
        monkeypatch.setitem(sys.modules, name, module)
    fake_d = types.ModuleType('kit.sdft_verl')
    fake_d.check_pinned = lambda: None
    monkeypatch.setitem(sys.modules, 'kit.sdft_verl', fake_d)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    import contextlib
    monkeypatch.setattr(torch.random, 'fork_rng', lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(deletion_worker, 'shard_masks', lambda *args: events.append('fp32-mask-shard'))
    deletion_worker.install_worker_patch()
    model = Tiny()
    worker.apply_fsdp2(model, {}, {})
    assert events == ['fp32-mask-shard', 'shard']
    assert worker.fsdp2_load_full_state_dict(model, {'bank': 'unchanged'}) == {'bank': 'unchanged'}
    params = worker.build_optimizer(model.parameters(), None)
    assert params and all(p.requires_grad for p in params)
    dense = worker.convert_weight_keys(model.state_dict(), model)
    assert all('diff_mask' not in key for key in dense)


def test_mask_receipt_counts_both_flip_directions_and_teacher_identity():
    from deletion_tuning.deletion_mask import mask_bits, mask_counts, assert_matching_masks
    model = install(Tiny())
    teacher = copy.deepcopy(model)
    assert_matching_masks(model, teacher)
    gate = next(iter(layers(model).values())).diff_mask.gate
    with torch.no_grad():
        gate.fill_(1.)
        gate[0, 0] = -1.
    initial = mask_bits(model)
    assert next(iter(mask_counts(model, initial).values()))['hard_zeros'] == 1
    with torch.no_grad():
        gate[0, 0] = 1.
        gate[0, 1] = -1.
    row = next(iter(mask_counts(model, initial).values()))
    assert row == dict(weights=12, hard_zeros=1, keep_to_delete=1, delete_to_keep=1)
    with pytest.raises(ValueError, match='bit for bit'):
        assert_matching_masks(model, teacher)


@pytest.mark.parametrize('rate', [0., .02])
def test_teacher_rate_must_match_registered_config(rate):
    from deletion_tuning.deletion_actor import teacher_rate
    with pytest.raises(ValueError, match='teacher_update_rate'):
        teacher_rate(SimpleNamespace(self_distillation={'teacher_update_rate': rate}))


def test_finder_rule_fixed_grid_bounds_fallback_and_ties(tmp_path):
    import hashlib
    import json
    from deletion_tuning.deletion_lr import GRID, choose_rate, frozen_rate
    trials = [dict(rate=rate, loss_drop=.1, initial_damage=.1, max_flip_fraction=.01)
              for rate in GRID]
    assert choose_rate(trials) == (.001, False)
    trials[-1]['loss_drop'] = .2
    assert choose_rate(trials) == (.01, False)
    trials[-1]['max_flip_fraction'] = .010001
    assert choose_rate(trials) == (.001, False)
    for row in trials:
        row['initial_damage'] = .100001
    assert choose_rate(trials) == (.005, True)
    for row in trials:
        row.update(initial_damage=0., loss_drop=float('nan'))
    assert choose_rate(trials) == (.005, True)
    with pytest.raises(ValueError, match='complete ordered'):
        choose_rate(trials[:-1])
    record = dict(schema='deletion-lr.v1', code_check=False, training_only=True,
                  updates_per_rate=3, max_damage=.1, max_flips=.01, grid=list(GRID),
                  model_geometry=[4096, 36], seed=101, trials=trials, selected_rate=.005, fallback=True)
    path = tmp_path / 'rate.json'
    path.write_text(json.dumps(record))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert frozen_rate(path, digest) == .005
    with pytest.raises(ValueError, match='hash differs'):
        frozen_rate(path, '0' * 64)
    record['code_check'] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='production'):
        frozen_rate(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize('fails', [False, True])
def test_actual_d_finder_exercises_grid_and_restores_every_trial(monkeypatch, fails):
    from deletion_tuning.deletion_smoke import Batch, fingerprint
    from deletion_tuning.deletion_lr import GRID, run_finder
    monkeypatch.setenv('P3B_REPLAY', '1')
    torch.manual_seed(101)
    model = install(Causal())
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.005)
    config = SimpleNamespace(self_distillation=dict(full_logit_distillation=True, alpha=0.,
        teacher_regularization='ema', teacher_update_rate=.01), loss_agg_mode='seq-mean-token-mean',
        ppo_epochs=1, ppo_mini_batch_size=4)
    owner = SimpleNamespace(actor_module=model, teacher_module=teacher, actor_optimizer=optimizer,
                            config=config, ulysses_sequence_parallel_size=1, scaler=None, _v4_completed_updates=0)
    observed = []
    def step():
        observed.append((optimizer.param_groups[0]['lr'], owner._v4_completed_updates))
        if fails:
            raise RuntimeError('trial interrupted')
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        return norm
    owner._optimizer_step = step
    ids = torch.randint(0, 7, (4, 12))
    values = dict(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=torch.arange(12).expand(4, -1),
        responses=ids[:, -8:], response_mask=torch.ones(4, 8), old_log_probs=torch.zeros(4, 8),
        rollout_log_probs=torch.zeros(4, 8), teacher_input_ids=ids.clone(),
        teacher_attention_mask=torch.ones_like(ids), teacher_position_ids=torch.arange(12).expand(4, -1),
        self_distillation_mask=torch.ones(4))
    values['teacher_input_ids'][:, 1] = 3
    data = Batch(values, [dict(p3b_replay=i == 3, p3b_target_ids=[1, 2, 3]) for i in range(4)])
    before = fingerprint((model.state_dict(), teacher.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
    if fails:
        with pytest.raises(RuntimeError, match='trial interrupted'):
            run_finder(owner, data, code_check=True)
    else:
        record = run_finder(owner, data, code_check=True)
        assert record['selected_rate'] is None and record['code_check'] is True
        assert observed == [(rate, step) for rate in GRID for step in range(2)]
        assert len(record['initial_masks']) == 1
        assert all(len(row['updates']) == 2 and row['updates'][0]['masks'] for row in record['trials'])
        assert record['training_input_hashes']
    assert fingerprint((model.state_dict(), teacher.state_dict(), optimizer.state_dict(), torch.get_rng_state())) == before
    assert owner._v4_completed_updates == 0
    assert not hasattr(owner, '_deletion_finding_lr')
    assert next(iter(layers(model).values())).diff_mask.current_step == 0
