"""Frozen confirmation closure and owner-conditioned cache isolation."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kit import cr_confirmation as cr, cr_jobs
from kit.cr_dependencies import FROZEN_DEPENDENCIES
from kit.nm_common import KIT, atomic, load, sha


def uncovered_imports(kit, frozen):
    """Conservative plugin coverage plus every explicit repository import.

    All kit Python/shell files are included because CR uses computed worker,
    reward and entry-point names. AST imports also follow external scripts and
    fixtures, including imports nested in optional/runtime-only branches.
    """
    root = kit.parent
    covered = {(kit/name).resolve() for name in frozen}
    missing = {p.resolve() for p in kit.rglob('*') if p.is_file()
               and p.suffix in ('.py', '.sh') and '__pycache__' not in p.parts} - covered
    for path in covered:
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level:
                    parent = path.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    module = '.'.join((*parent.relative_to(root).parts, module)).strip('.')
                modules = [module, *(module+'.'+a.name for a in node.names)]
            for module in modules:
                candidate = root.joinpath(*module.split('.')).with_suffix('.py')
                if candidate.is_file() and candidate.resolve() not in covered:
                    missing.add(candidate.resolve())
    return sorted(str(p.relative_to(root)) for p in missing)


def test_confirmation_imports_and_runtime_plugins_are_covered():
    assert not uncovered_imports(KIT, FROZEN_DEPENDENCIES)
    assert len(FROZEN_DEPENDENCIES) == 477
    assert set(cr.code_seal()) == set(FROZEN_DEPENDENCIES) | {p for p, _ in cr.OWNER_RESOURCES}


def test_coverage_guard_detects_an_uncovered_confirmation_import(tmp_path):
    kit = tmp_path/'kit'
    kit.mkdir()
    (kit/'cr_entry.py').write_text('def optional():\n    import helper\n')
    (tmp_path/'helper.py').write_text('value = 1\n')
    assert uncovered_imports(kit, ('cr_entry.py',)) == ['helper.py']
    assert not uncovered_imports(kit, ('cr_entry.py', '../helper.py'))


def test_every_cache_environment_read_is_removed():
    tree = ast.parse((KIT/'m2_model_cache.py').read_text())
    reads = {node.args[0].value for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and ast.unparse(node.func) == 'os.environ.get'}
    assert reads <= set(cr_jobs.MODEL_CACHE_ENV)
    assert {'M2_MODEL_HASH_RECEIPTS_FILE', 'M2_MODEL_HASH_RECEIPTS_FILE_SHA256'} <= set(cr_jobs.MODEL_CACHE_ENV)


@pytest.mark.parametrize('entry', ['parent-row', 'allocation-child', 'allocation-wrapper'])
def test_cr_row_never_consults_inherited_model_cache(tmp_path, entry):
    # A valid cache hit would return this sentinel without fresh model hashing.
    # Poison raw bytes separately detect any attempted cache read.
    cache = tmp_path/'cache.json'
    cache.write_text('must never read this invalid cache')
    config = tmp_path/'launch.json'
    atomic(config, dict(out=str(tmp_path/'B'), technical_synthetic=True,
        cr_confirmation=dict(out=str(tmp_path/'C'), six_row_root=str(tmp_path/'A'),
                             v6a_root=str(tmp_path/'v6a'), v6b_root=str(tmp_path/'v6b'), hosts=['CPU-STANDIN'])))
    proof = tmp_path/'proof.json'
    script = tmp_path/'probe.py'
    script.write_text('''import sys, os
from pathlib import Path
sys.path.insert(0, ROOT)
from kit import m2_model_cache as cache, nm_jobs, nm_execute, cr_jobs
from kit.nm_common import atomic, load
# Exercise the real installed cache hook. Fresh hashing is a CPU stub so no
# model/GPU is required; any cache consult still parses the poison receipt.
cache.uncached_identity = lambda *a, **k: {'identity': 'fresh hashes'}
cache.install()
def probe():
    from kit import v4_teacher
    assert v4_teacher.local_identity('.') == {'identity': 'fresh hashes'}
    assert all(name not in os.environ for name in cr_jobs.MODEL_CACHE_ENV)
    child = __import__('subprocess').check_output([sys.executable, '-I', '-c',
        'import os,json;print(json.dumps(dict(os.environ)))'], text=True)
    import json
    assert all(name not in json.loads(child) for name in cr_jobs.MODEL_CACHE_ENV)
    atomic(PROOF, {'cache_consulted': False, 'fresh_hash_path': True, 'child_clean': True})
if ENTRY == 'parent-row':
    from kit import a4_common
    def launch(command, *, env, pass_fds):
        assert all(name not in env for name in cr_jobs.MODEL_CACHE_ENV)
        probe()
        return __import__('types').SimpleNamespace(returncode=0)
    a4_common.gpu_run = launch
    assert nm_jobs.run(CONFIG, 'NM-cr-bind') == 0
else:
    # Call operation directly too, so cleanup cannot rely solely on settings.
    cr_jobs.verify_standin = lambda cfg: probe()
    os.environ['KIT_M3_MODE'] = 'cr-confirmation'
    cfg = {'technical_synthetic': True}
    attempt = Path(PROOF).parent/'C/attempt'; attempt.mkdir(parents=True)
    if ENTRY == 'allocation-wrapper':
        os.environ['UTIL_WATCH_ACTIVE'] = '1'
        sys.argv = ['nm_execute', '--config', CONFIG, '--task', 'NM-cr-qualify-G', '--attempt', str(attempt)]
        assert nm_execute.main() == 0
    else:
        cr_jobs.operation(cfg, {'phase': 'qualify'}, attempt, Path(PROOF).parent/'C')
'''.replace('ROOT', repr(str(KIT.parent))).replace('PROOF', repr(str(proof)))
        .replace('ENTRY', repr(entry)).replace('CONFIG', repr(str(config))))
    env = dict(os.environ)
    env.update({name: str(cache) for name in cr_jobs.MODEL_CACHE_ENV})
    env['M2_EXECUTION_PHASE'] = 'gpu'
    for key in ('KIT_V6A', 'KIT_V6_AUTHORITY_SHA', 'M2_INFRA_AUTHORITY_SHA256', 'UTIL_WATCH_ACTIVE'):
        env.pop(key, None)
    result = subprocess.run([sys.executable, '-I', '-B', str(script)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert load(proof) == dict(cache_consulted=False, fresh_hash_path=True, child_clean=True)
    assert cache.read_text() == 'must never read this invalid cache'
