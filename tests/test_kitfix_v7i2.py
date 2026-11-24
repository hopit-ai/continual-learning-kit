"""Offline checkpoint, release identity and subset regressions for v7i2."""
import ast
from io import BytesIO
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch

def worker():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .99**step)
    return SimpleNamespace(actor_optimizer=optimizer, actor_lr_scheduler=scheduler,
                           actor=SimpleNamespace(_update_teacher=lambda: 'ema'))


def install_worker_counters(w):
    """Execute the worker's actual counter block without importing verl or Ray.

    The baseline's inline loop is supported too, so the checkpoint regression
    fails at torch.save on the old implementation, rather than at import time.
    """
    path = Path(__file__).resolve().parents[1]/'kit/nm_native.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NMWorker')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'init_model')
    for i, node in enumerate(init.body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'install_counters':
            from kit.nm_counters import install_counters
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'),
                 dict(self=w, install_counters=install_counters))
            return
        if isinstance(node, ast.Assign) and any(ast.unparse(t) == 'self._nm_steps' for t in node.targets):
            # Preserve the closure's function scope in the original worker.
            fn = ast.parse('def install(self):\n    pass').body[0]
            fn.body = init.body[i:-1]
            module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
            namespace = {}
            exec(compile(module, str(path), 'exec'), namespace)
            namespace['install'](w)
            return
    raise AssertionError('NMWorker counter installation missing')


def advance(w):
    w.actor_optimizer.zero_grad()
    w.actor_optimizer.param_groups[0]['params'][0].grad = torch.ones(1)
    w.actor_optimizer.step()
    w.actor_lr_scheduler.step()
    assert w.actor._update_teacher() == 'ema'


def test_native_counter_checkpoint_and_resume():
    w = worker()
    install_worker_counters(w)
    for _ in range(3):
        advance(w)
    assert w._nm_steps == dict(optimizer=3, scheduler=3, ema=3)
    scheduler_state = w.actor_lr_scheduler.state_dict()
    optimizer_state = w.actor_optimizer.state_dict()
    torch.save(optimizer_state, BytesIO())
    torch.save(scheduler_state, BytesIO())
    assert not any(callable(value) for value in scheduler_state.values())
    assert 'step' not in scheduler_state and 'step' not in optimizer_state
    fresh = worker()
    install_worker_counters(fresh)
    fresh.actor_optimizer.load_state_dict(optimizer_state)
    fresh.actor_lr_scheduler.load_state_dict(scheduler_state)
    # NMWorker restores the separate nm-state.json counters this way.
    fresh._nm_steps = dict(w._nm_steps)
    assert fresh.actor_lr_scheduler.state_dict() == scheduler_state
    advance(fresh)
    assert fresh._nm_steps == dict(optimizer=4, scheduler=4, ema=4)
    torch.save(fresh.actor_lr_scheduler.state_dict(), BytesIO())
    torch.save(fresh.actor_optimizer.state_dict(), BytesIO())


def test_old_instance_patch_reproduces_pickle_failure():
    w = worker()
    for owner in (w.actor_optimizer, w.actor_lr_scheduler):
        original = owner.step
        def counted(*args, _original=original, **kwargs):
            return _original(*args, **kwargs)
        counted._wrapped_by_lr_sched = True
        owner.step = counted
    torch.save(w.actor_optimizer.state_dict(), BytesIO())
    with pytest.raises((AttributeError, TypeError), match='(local object|pickle)'):
        torch.save(w.actor_lr_scheduler.state_dict(), BytesIO())


def test_no_instance_scheduler_step_patch_anywhere_in_kit():
    root = Path(__file__).resolve().parents[1]/'kit'
    for path in root.rglob('*.py'):
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and target.attr == 'step':
                        assert 'sched' not in ast.unparse(target.value).lower(), (path.name, node.lineno)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'setattr':
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == 'step':
                    assert 'sched' not in ast.unparse(node.args[0]).lower(), (path.name, node.lineno)
        # Catch the original indirect owner/method/key loop too.
        assert not re.search(r"\([^\n]*scheduler\s*,\s*['\"]step['\"]", source), path.name


def test_worker_installs_helper_and_subclasses_inherit_init():
    root = Path(__file__).resolve().parents[1]/'kit'
    native = ast.parse((root/'nm_native.py').read_text())
    nm = next(n for n in native.body if isinstance(n, ast.ClassDef) and n.name == 'NMWorker')
    init = next(n for n in nm.body if isinstance(n, ast.FunctionDef) and n.name == 'init_model')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'install_counters'
               for n in ast.walk(init))
    for filename, name in [('trace_native.py', 'TraceWorker'), ('ts_native.py', 'TSWorker')]:
        tree = ast.parse((root/filename).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        assert any(ast.unparse(b).split('.')[-1] == 'NMWorker' for b in cls.bases)
        override = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'init_model'), None)
        if override:
            assert 'super().init_model()' in ast.unparse(override)


def test_v7i2_refuses_cr_with_shipped_receipt_before_input_work(monkeypatch, tmp_path):
    from kit import cr_confirmation as cr
    from kit.nm_common import Pending, sha
    path = Path(cr.__file__).parent/'resources/cr-confirmation/diff-receipt-v7i1.json'
    monkeypatch.setattr(cr, 'bind_inputs', lambda *a, **k: pytest.fail('input work preceded seal refusal'))
    with pytest.raises(Pending, match='kit-message-v7i3'):
        cr.verify_diff(dict(path=str(path), sha256=sha(path)))


def test_publisher_accepts_tag_and_ships_regressions():
    from scripts import publish_kit
    tree = ast.parse(Path(publish_kit.__file__).read_text())
    patterns = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'fullmatch'
                and n.args and isinstance(n.args[0], ast.Constant)]
    assert any(re.fullmatch(pattern, 'kit-message-v7i2') for pattern in patterns)
    assert {'test_kitfix_v7i2.py', 'test_trace_repair.py'} <= set(publish_kit.TESTS)


def test_subset_departure_and_accuracy_authority():
    from kit.trace_common import SUBSET, TASKS
    import json
    root = Path(__file__).resolve().parents[1]/'kit'
    assert SUBSET == ('FOMC', 'MeetingBank', 'NumGLUE-ds')
    assert len(TASKS) == 8 and 'NumGLUE-cm' in TASKS
    metrics = json.loads((root/'resources/trace/operator/task-authorities-template.json').read_text())
    for task in ('NumGLUE-ds', 'NumGLUE-cm'):
        assert metrics[task]['key'] == 'accuracy' and metrics[task]['scale_to_100'] == 100
