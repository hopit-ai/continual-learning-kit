"""Frozen confirmation closure and owner-conditioned cache isolation."""
import ast
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from kit import cr_confirmation as cr, cr_jobs
from kit.cr_dependencies import FROZEN_DEPENDENCIES
from kit.nm_common import KIT, atomic, load, sha


# Only these independent A2 additions are exempt from the full kit code scan.
# Pin the literal list independently so additions cannot broaden the exemption.
A2_CODE_PATHS = (
    'a2_bind.py',
    'a2_budget.py',
    'a2_common.py',
    'a2_confirmation.py',
    'a2_dryrun.py',
    'a2_execute.py',
    'a2_j_read.py',
    'a2_jobs.py',
    'a2_p_read.py',
    'a2_pipeline.py',
    'a2_r_read.py',
    'a2_read.py',
    'a2_replay.py',
    'a2_return.py',
    'a2_rewrite.py',
    'a2_sim.py',
    'jobs/a2/run-task.sh',
)
A2_CODE_PATHS_SHA256 = '5b6d917ef8a0875f927695d287199b490d5443e037ea12f39d6c04790705fb91'

# Independent TRACE additions have the same exact-path coverage boundary as A2.
TRACE_CODE_PATHS = (
    'jobs/trace/run-task.sh',
    'resources/trace/vendor/hf-datasets-2.21.0-sari/sari.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/SuperGLUE_BoolQ/SuperGLUE_BoolQ_ppl.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/SuperGLUE_BoolQ/SuperGLUE_BoolQ_ppl_314b96.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/bbh/bbh_gen.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/bbh/bbh_gen_5b92b0.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/gsm8k/gsm8k_gen.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/gsm8k/gsm8k_gen_1d7fe4.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/mmlu/mmlu_ppl.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/mmlu/mmlu_ppl_ac766d.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/piqa/piqa_ppl.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/piqa/piqa_ppl_1cf9f0.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/tydiqa/tydiqa_gen.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/datasets/tydiqa/tydiqa_gen_978d2a.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/summarizers/groups/bbh.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/summarizers/groups/mmlu.py',
    'resources/trace/vendor/opencompass-0.1.5/configs/summarizers/groups/tydiqa.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/datasets/bbh.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/datasets/boolq.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/datasets/gsm8k.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/datasets/mmlu.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/datasets/tydiqa.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/models/huggingface.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/openicl/icl_evaluator/hf_metrics/accuracy.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/openicl/icl_evaluator/icl_base_evaluator.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/openicl/icl_evaluator/icl_hf_evaluator.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/openicl/icl_inferencer/icl_ppl_inferencer.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/openicl/icl_prompt_template.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/openicl/icl_retriever/icl_base_retriever.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/openicl/icl_retriever/icl_fix_k_retriever.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/openicl/icl_retriever/icl_zero_retriever.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/tasks/openicl_infer.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/utils/prompt.py',
    'resources/trace/vendor/opencompass-0.1.5/opencompass/utils/text_postprocessors.py',
    'trace_calibration.py',
    'trace_calibration_build.py',
    'trace_common.py',
    'trace_datasets.py',
    'trace_entry.py',
    'trace_general_build.py',
    'trace_general_metrics.py',
    'trace_general_score.py',
    'trace_jobs.py',
    'trace_judge.py',
    'trace_metrics.py',
    'trace_native.py',
    'trace_operator.py',
    'trace_overlap.py',
    'trace_prepare.py',
    'trace_prompt.py',
    'trace_read.py',
    'trace_rehearsal.py',
    'trace_score.py',
    'trace_sft.py',
    'trace_targets.py',
    'trace_train.py',
)
TRACE_CODE_PATHS_SHA256 = 'e359065054c1141179c5a57543c16c3256e3b92f1fedf9abc71da15033ca12b6'

def uncovered_imports(kit, frozen):
    """Conservative plugin coverage plus every explicit repository import.

    Every kit Python/shell path requires coverage except the exact A2 and TRACE allowlists.
    This catches new plugins even when no explicit import names them.
    AST imports remain checked in EVERY frozen Python file, including external
    scripts/fixtures and imports nested in optional/runtime-only branches.
    """
    root = kit.parent
    covered = {(kit/name).resolve() for name in frozen}
    missing = {p.resolve() for p in kit.rglob('*') if p.is_file()
               and p.suffix in ('.py', '.sh')
               and p.relative_to(kit).as_posix() not in (*A2_CODE_PATHS, *TRACE_CODE_PATHS)} - covered
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


def a2_references(kit, frozen):
    """Reject imports, computed-name fragments and shell paths anywhere sealed."""
    pattern = re.compile(r'\ba2_|\bjobs[/.\\]a2\b')
    return [name for name in frozen
            if pattern.search(name) or pattern.search((kit/name).read_text())]


def directory_scan_loaders(kit, frozen):
    """Conservatively inventory code-directory scans in dynamic-loader files.

    Include importlib in child-command strings. Python globs and directory enumeration over kit are suspect;
    straightforward aliases of the kit directory are included.
    Data-only globs and fixed-prefix plugin maps cannot select the A2 paths.
    This is a regression inventory, not a general Python reachability proof.
    """
    import fnmatch
    found = []
    for name in frozen:
        path = kit/name
        if path.suffix != '.py':
            continue
        source = path.read_text()
        if not re.search(r'importlib|__import__', source):
            continue
        tree = ast.parse(source)
        aliases = {'KIT', 'kit'}
        # Track straightforward aliases, including kit=Path(KIT).
        for _ in range(3):
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value:
                    if (isinstance(node.value, ast.Name) and node.value.id in aliases or
                        isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == 'Path' and len(node.value.args) == 1
                        and isinstance(node.value.args[0], ast.Name) and node.value.args[0].id in aliases):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        aliases.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            method = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')
            if method not in ('glob', 'rglob', 'iglob', 'iterdir', 'walk', 'listdir', 'scandir'):
                continue
            patterns = [n.value for arg in node.args for n in ast.walk(arg)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            broad_python = any(fnmatch.fnmatch(p, pattern) or fnmatch.fnmatch('kit/'+p, pattern) or fnmatch.fnmatch(Path(p).name, pattern)
                               for pattern in patterns for p in A2_CODE_PATHS)
            receiver = ast.unparse(func.value) if isinstance(func, ast.Attribute) else ''
            kit_directory = (any(re.search(r'\b'+re.escape(alias)+r'\b', receiver) for alias in aliases)
                             or '__file__' in receiver or any('kit' in p for p in patterns))
            if (broad_python and kit_directory) or (method in ('iterdir', 'walk', 'listdir', 'scandir') and kit_directory):
                found.append((name, node.lineno))
    return sorted(found)


def test_sealed_directory_scan_loader_inventory_is_frozen():
    # v4_prepare's import_smoke scans all kit code, but CR never calls it.
    # Keep the known off-CR loader visible and tied to the approved receipt;
    # any added loader or changed source fails this inventory/pin check.
    assert directory_scan_loaders(KIT, cr.code_seal()) == [('v4_prepare.py', 117), ('v4_prepare.py', 121), ('v4_prepare.py', 258)]
    receipt = load(KIT/'resources/cr-confirmation/diff-receipt-v7c.json')
    assert sha(KIT/'v4_prepare.py') == receipt['code']['v4_prepare.py']


@pytest.mark.parametrize('scan', ["KIT.glob('*.py')", "KIT.rglob('*.py')", 'KIT.iterdir()',
                                  "glob.glob('kit/*.py')", 'KIT_ALIAS.iterdir()'])
def test_directory_scan_loader_guard_detects_discovery(tmp_path, scan):
    (tmp_path/'entry.py').write_text('import importlib\nKIT_ALIAS = KIT\nfor path in '+scan+':\n    importlib.import_module(path.stem)\n')
    assert directory_scan_loaders(tmp_path, ('entry.py',)) == [('entry.py', 3)]


def test_confirmation_imports_and_runtime_plugins_are_covered():
    assert len(A2_CODE_PATHS) == 17
    assert tuple(sorted(A2_CODE_PATHS)) == A2_CODE_PATHS
    assert hashlib.sha256(('\n'.join(A2_CODE_PATHS)+'\n').encode()).hexdigest() == A2_CODE_PATHS_SHA256
    assert all((KIT/name).is_file() for name in A2_CODE_PATHS)
    assert len(TRACE_CODE_PATHS) == 56
    assert tuple(sorted(TRACE_CODE_PATHS)) == TRACE_CODE_PATHS
    assert hashlib.sha256(('\n'.join(TRACE_CODE_PATHS)+'\n').encode()).hexdigest() == TRACE_CODE_PATHS_SHA256
    assert all((KIT/name).is_file() for name in TRACE_CODE_PATHS)
    assert not uncovered_imports(KIT, FROZEN_DEPENDENCIES)
    assert not a2_references(KIT, cr.code_seal())
    assert len(FROZEN_DEPENDENCIES) == 477
    assert set(cr.code_seal()) == set(FROZEN_DEPENDENCIES) | {p for p, _ in cr.OWNER_RESOURCES}


def test_only_exact_trace_paths_are_exempt(tmp_path):
    kit = tmp_path/'kit'
    kit.mkdir()
    (kit/'trace_common.py').write_text('VALUE = 1\n')
    assert not uncovered_imports(kit, ())
    (kit/'trace_unregistered.py').write_text('VALUE = 2\n')
    assert uncovered_imports(kit, ()) == ['kit/trace_unregistered.py']
    (kit/'trace_unregistered.py').unlink()
    (kit/'cr_entry.py').write_text('from kit import trace_common\n')
    assert uncovered_imports(kit, ('cr_entry.py',)) == ['kit/trace_common.py']


def test_coverage_guard_detects_an_uncovered_confirmation_import(tmp_path):
    kit = tmp_path/'kit'
    kit.mkdir()
    (kit/'cr_entry.py').write_text('def optional():\n    import helper\n')
    (tmp_path/'helper.py').write_text('value = 1\n')
    assert uncovered_imports(kit, ('cr_entry.py',)) == ['helper.py']
    assert not uncovered_imports(kit, ('cr_entry.py', '../helper.py'))


def test_only_exact_a2_paths_are_exempt_and_imports_still_require_coverage(tmp_path):
    kit = tmp_path/'kit'
    kit.mkdir()
    (kit/'a2_bind.py').write_text('value = 1\n')
    assert not uncovered_imports(kit, ())
    (kit/'cr_entry.py').write_text('from kit import a2_bind\n')
    assert uncovered_imports(kit, ('cr_entry.py',)) == ['kit/a2_bind.py']
    assert a2_references(kit, ('cr_entry.py',)) == ['cr_entry.py']
    # Unimported new code still requires coverage, regardless of its prefix.
    (kit/'zz_new.py').write_text('value = 1\n')
    (kit/'a2_extra.py').write_text('value = 1\n')
    assert uncovered_imports(kit, ('cr_entry.py', 'a2_bind.py')) == ['kit/a2_extra.py', 'kit/zz_new.py']


@pytest.mark.parametrize('source', ['import kit.a2_extra', "plugin = 'a2_' + name",
                                     "command = 'kit/jobs/a2/run-task.sh'"])
def test_a2_guard_catches_imports_and_computed_names(tmp_path, source):
    (tmp_path/'entry.py').write_text(source+'\n')
    assert a2_references(tmp_path, ('entry.py',)) == ['entry.py']


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
