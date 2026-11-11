"""Runs with Python/pytest alone. NumPy is used for independent mask oracles."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from deletion_tuning.deletion_logic import calibration_batches, sharpness, smoke_price, wanda_deletions

ROOT = Path(__file__).resolve().parents[1]


def test_ramp_is_completed_update_based_and_clamped():
    assert [sharpness(n) for n in (0, 1, 20, 40, 99)] == [100., 347.5, 5050., 10000., 10000.]
    with pytest.raises(ValueError):
        sharpness(-1)


def test_strict_wanda_ties_and_small_fraction():
    assert wanda_deletions([1, 1, 2, 3], .5) == [False] * 4
    assert wanda_deletions([1, 2, 3, 4], .5) == [True, False, False, False]
    assert wanda_deletions([1, 2, 3, 4]) == [False] * 4
    with pytest.raises(ValueError):
        wanda_deletions([float('nan')])


def test_numpy_sigmoid_gradient_and_hard_dense_equivalence():
    np = pytest.importorskip('numpy')
    gate = np.array([[-.02, 0, .02], [.01, -.01, .03]])
    weight = np.arange(6).reshape(2, 3) / 10
    x = np.array([.2, -.3, .5])
    soft = 1 / (1 + np.exp(-100 * gate))
    analytical = weight * x * soft * (1 - soft) * 100
    eps = 1e-7
    finite = (weight * x * (1 / (1 + np.exp(-100 * (gate + eps))) -
                           1 / (1 + np.exp(-100 * (gate - eps))))) / (2 * eps)
    assert np.max(np.abs(analytical - finite)) < 1e-8
    assert np.isfinite(analytical).all() and np.any(analytical != 0)
    assert np.array_equal((weight * (gate > 0)) @ x, np.sum(weight * (gate > 0) * x, axis=1))


def test_exact_full_reservation_price():
    quote = smoke_price()
    assert quote['count'] == 2 and quote['gpu'] == 'A10G'
    assert quote['seconds'] == 3900
    assert quote['runtime_max_usd'] == pytest.approx(8.607768, abs=1e-9)
    assert quote['total_cap_usd'] == pytest.approx(9.668186, abs=1e-6)
    assert quote['total_cap_usd'] <= 10


def test_calibration_identity_and_shapes():
    document = dict(schema='deletion-calibration.v1', task_ids=list(map(str, range(32))),
                    schedule_sha256='a' * 64,
                    batches=[dict(input_ids=[[1, 2]] * 8, attention_mask=[[1, 1]] * 8)] * 4)
    document['sha256'] = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert len(calibration_batches(document)) == 4
    document['task_ids'][0] = 'changed'
    with pytest.raises(ValueError, match='identity'):
        calibration_batches(document)


def test_container_has_no_modal_or_large_model_entry():
    for relative in ('deletion_tuning/deletion_smoke.py', 'deletion_tuning/deletion_container.py'):
        source = (ROOT / relative).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(x.name != 'modal' for x in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module != 'modal'
        assert 'Qwen3-8B' not in source
    launcher = (ROOT / 'scripts/deletion_smoke_modal.py').read_text()
    assert 'Sandbox.create' in launcher and "gpu='A10G:2'" in launcher
    assert 'app list' not in launcher and '.run_commands(' not in launcher
    assert '--max_restarts=0' in (ROOT / 'deletion_tuning/deletion_container.py').read_text()


def test_vendored_license_and_file_hashes():
    vendor = ROOT / 'deletion_tuning/vendor/deleting_weights'
    provenance = json.loads((vendor / 'PROVENANCE.json').read_text())
    assert provenance['commit'] == '94f0340ebb574f95bb41d844165253a6785e215e'
    for name, expected in provenance['vendored_sha256'].items():
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == expected
    assert 'MIT License' in (vendor / 'LICENSE').read_text()


def test_preparation_preserves_first_task_order_excludes_replay_and_caps():
    from deletion_tuning.deletion_prepare import prepare
    class Tokenizer:
        padding_side = 'left'
        def apply_chat_template(self, prompt, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return prompt[0]['content']
        def encode(self, text, **kwargs):
            return list(range(int(text)))
        def pad(self, data, **kwargs):
            width = max(map(len, data['input_ids']))
            return dict(input_ids=[[0] * (width-len(x)) + x for x in data['input_ids']],
                        attention_mask=[[0] * (width-len(x)) + [1] * len(x) for x in data['input_ids']])
    rows = [dict(id='replay', prompt='1', p3b_replay=True)]
    rows += [dict(id=str(i), prompt=str(1100+i), extra_info=None) for i in range(35)]
    result = prepare(rows, Tokenizer(), 'a'*64, 101)
    assert result['task_ids'] == list(map(str, range(32)))
    assert all(len(x) == 1024 for b in result['batches'] for x in b['input_ids'])
    assert result['seed'] == 101


def test_default_cli_only_prints_price():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/deletion_smoke_modal.py')],
                            capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    assert record['status'] == 'PRICED, UNLAUNCHED'
    assert record['image_build_seconds'] == 0
    assert record['total_cap_usd'] < 10


def test_launcher_one_mock_container_and_durable_receipt(monkeypatch, tmp_path):
    import contextlib
    import sys
    from types import SimpleNamespace
    from scripts.deletion_smoke_modal import launch, IMAGE_ID
    calls = []
    files = {}
    class Files:
        def write_bytes(self, data, path):
            files[path] = data
        def write_text(self, data, path):
            files[path] = data.encode()
        def read_bytes(self, path):
            assert path == '/work/results/receipt.json'
            return b'{"status":"PASS"}'
    class Sandbox:
        object_id = 'mock-only'
        filesystem = Files()
        def exec(self, *args, **kwargs):
            assert args == ('python', '-m', 'deletion_tuning.deletion_container')
            return SimpleNamespace(wait=lambda: 0, stdout=iter(['step 1\n']), stderr=iter([]))
        def terminate(self, **kwargs):
            calls.append(('terminate', kwargs))
    async def create(*args, **kwargs):
        calls.append(('create', kwargs))
        return Sandbox()
    class App:
        def __init__(self, name):
            assert name == 'deletion-06b-smoke'
        def run(self, detach=False):
            assert detach is True
            return contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, 'modal', SimpleNamespace(App=App,
        Image=SimpleNamespace(from_id=lambda image: image if image == IMAGE_ID else None),
        Sandbox=SimpleNamespace(create=SimpleNamespace(aio=create))))
    receipt = tmp_path / 'receipt.json'
    launch(receipt)
    record = json.loads(receipt.read_text())
    assert record['status'] == 'PASS'
    assert (tmp_path / 'receipt.sandbox_id').read_text().strip() == 'mock-only'
    assert 'step 1' in (tmp_path / 'receipt.container.log').read_text()
    created = [v for name, v in calls if name == 'create']
    assert len(created) == 1
    assert created[0]['gpu'] == 'A10G:2' and created[0]['timeout'] == 3900
    assert created[0]['cpu'] == (4., 4.) and created[0]['memory'] == (32768, 32768)
    assert '/work/deletion_tuning/vendor/deleting_weights/LICENSE' in files
    assert any(name == 'terminate' for name, _ in calls)
    with pytest.raises(FileExistsError):
        launch(receipt)
    assert len([v for name, v in calls if name == 'create']) == 1
