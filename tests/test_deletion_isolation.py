"""Deletion dispatch and instance hooks must leave inherited actors untouched."""
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip('torch')


@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('instance_override', [False, True])
def test_actor_override_restored_and_other_instances_untouched(monkeypatch, fails, instance_override):
    from deletion_tuning import deletion_actor as adapter
    from kit import sdft_actor
    original_update = sdft_actor.update_policy
    events = []

    class Owner:
        _update_teacher = lambda self: 'inherited'

    owner, control = Owner(), Owner()
    owner.actor_module = object()
    owner.config = SimpleNamespace(self_distillation={'teacher_update_rate': .01})
    owner.teacher_module = SimpleNamespace(eval=lambda: events.append('teacher-eval'), requires_grad_=lambda _: None)
    owner._v4_completed_updates = 17
    if instance_override:
        owner._update_teacher = lambda: 'instance'
    previous = owner._update_teacher
    monkeypatch.setattr(adapter, 'set_step', lambda model, step: events.append(('step', step)))
    monkeypatch.setattr(adapter, 'update_teacher', lambda model, teacher, rate: events.append(('mask-ema', rate)))

    def update(actor, data):
        assert actor is owner and data == 'batch'
        assert control._update_teacher() == 'inherited'
        assert sdft_actor.update_policy is original_update
        actor._update_teacher()
        if fails:
            raise RuntimeError('update failed')
        return {'unchanged': True}

    monkeypatch.setattr(adapter, 'sdft_update_policy', update)
    if fails:
        with pytest.raises(RuntimeError, match='update failed'):
            adapter.update_policy(owner, 'batch')
    else:
        assert adapter.update_policy(owner, 'batch') == {'unchanged': True}
    assert events == [('step', 17), 'teacher-eval', ('mask-ema', .01)]
    assert owner._update_teacher == previous
    assert ('_update_teacher' in owner.__dict__) is instance_override


def test_ray_runner_selects_importable_deletion_worker_without_changing_d(monkeypatch):
    events = []

    class PPOActor:
        update_policy = lambda self, data: 'inherited'

    class Worker:
        def __init__(self):
            events.append('inherited-worker-init')

    class Runner:
        def __init__(self):
            self.role_worker_mapping = {}

        def add_actor_rollout_worker(self, config):
            self.role_worker_mapping['actor'] = Worker
            return Worker, 'group'

        def run(self, config):
            return 'inherited-config-checks'

    v = ModuleType('kit.sdft_verl')
    v.dp_actor = SimpleNamespace(DataParallelPPOActor=PPOActor)
    v.SdftActorRolloutRefWorker = Worker
    v.ray_trainer = SimpleNamespace(Role=SimpleNamespace(ActorRolloutRef='actor'))
    v.ray = SimpleNamespace(remote=lambda worker: ('ray-remote', worker))
    c = ModuleType('kit.c_sdft_runner')
    c.CTaskRunner = Runner
    actor = ModuleType('verl.workers.actor')
    actor.DataParallelPPOActor = PPOActor
    workers = ModuleType('verl.workers')
    workers.actor = actor
    verl = ModuleType('verl')
    verl.workers = workers
    timing = ModuleType('kit.v4_timing')
    timing.install_telemetry = lambda *args, **kwargs: events.append('telemetry')
    import kit
    monkeypatch.setattr(kit, 'sdft_verl', v, raising=False)
    for name, module in [('kit.sdft_verl', v), ('kit.c_sdft_runner', c),
                         ('kit.v4_timing', timing), ('verl', verl),
                         ('verl.workers', workers), ('verl.workers.actor', actor)]:
        monkeypatch.setitem(sys.modules, name, module)
    from deletion_tuning import deletion_worker
    monkeypatch.setattr(deletion_worker, 'install_worker_patch', lambda: events.append('mask-worker-hooks'))
    monkeypatch.delitem(sys.modules, 'deletion_tuning.deletion_runner', raising=False)
    module = importlib.import_module('deletion_tuning.deletion_runner')
    try:
        monkeypatch.setattr(module, 'registered_lr', lambda: .005)
        raw = dict(actor_rollout_ref=dict(actor=dict(optim=dict(lr=.005, weight_decay=.01))))
        normalized = module.validation_config(raw)
        assert normalized['actor_rollout_ref']['actor']['optim'] == dict(lr=1e-5, weight_decay=.01)
        assert raw['actor_rollout_ref']['actor']['optim']['lr'] == .005
        raw['actor_rollout_ref']['actor']['optim']['lr'] = .003
        with pytest.raises(ValueError, match='frozen qualification'):
            module.validation_config(raw)
        runner = module.DeletionTaskRunner()
        selected, group = runner.add_actor_rollout_worker(None)
        assert selected is module.DeletionActorRolloutRefWorker and group == 'group'
        assert selected.__module__ == 'deletion_tuning.deletion_runner'
        assert runner.role_worker_mapping['actor'] == ('ray-remote', selected)
        selected()
        assert events == ['inherited-worker-init', 'mask-worker-hooks', 'telemetry']
        assert actor.DataParallelPPOActor is module.DeletionPPOActor
        assert v.dp_actor.DataParallelPPOActor is PPOActor
        assert PPOActor().update_policy(None) == 'inherited'
        monkeypatch.setenv('DELETION_LR_MODE', 'qualify')
        from deletion_tuning import deletion_actor
        monkeypatch.setattr(deletion_actor, 'qualification_update', lambda self, data: 'qualified-'+data)
        assert module.DeletionPPOActor().update_policy('training') == 'qualified-training'
        monkeypatch.delenv('DELETION_LR_MODE')
        config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(
            strategy='fsdp2', optim=SimpleNamespace(lr=.005))))
        assert runner.run(config) == 'inherited-config-checks'
        config.actor_rollout_ref.actor.strategy = 'fsdp'
        with pytest.raises(ValueError, match='FSDP2'):
            runner.run(config)
    finally:
        sys.modules.pop('deletion_tuning.deletion_runner', None)
