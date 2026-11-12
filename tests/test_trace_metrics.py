"""CPU bridge regressions using staged stand-ins and the vendored SARI source."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from kit import trace_datasets, trace_metrics
from kit.trace_common import sha


def staged_row(root, value=73, body=None):
    root.mkdir()
    evaluations = root / 'evaluations'
    evaluations.mkdir()
    dependency = root / 'metrics.py'
    dependency.write_text('from datasets import load_metric\n'
                          f'VALUE = {value}\n')
    evaluation = evaluations / 'eval_fake.py'
    evaluation.write_text(body or
        'from metrics import VALUE\n'
        'def eval(sources, predictions, answers):\n'
        '    return {"sari": ({"sari": VALUE},)}\n')
    authority = dict(path=str(evaluation), sha256=sha(evaluation), function='eval',
                     key=['sari', 0, 'sari'], scale_to_100=1, uses_source=True,
                     dependencies=[dict(path=str(dependency), sha256=sha(dependency))])
    return dict(prompt='source', answer='answer', metric_authority=authority)


def assert_restored(row, expected=None, error=None):
    modules = sys.modules.copy()
    path = list(sys.path)
    if error:
        with pytest.raises(error):
            trace_metrics.score(row, 'answer')
    else:
        assert trace_metrics.score(row, 'answer') == expected
    assert sys.path == path
    assert all(sys.modules.get(name) is module for name, module in modules.items())
    owned = {'datasets', 'metrics', 'eval_fake', 'trace_upstream_metric', 'trace_hf_sari_2_21_0'}
    leaked = [name for name in sys.modules.keys() - modules.keys()
              if name.partition('.')[0] in owned]
    assert leaked == []


def test_layout_list_path_and_two_staged_trees_restore_cached_modules(tmp_path, monkeypatch):
    for name in ('datasets', 'metrics', 'eval_fake', 'trace_upstream_metric',
                 'trace_hf_sari_2_21_0'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    first = staged_row(tmp_path / 'first')
    second = staged_row(tmp_path / 'second', value=29)
    assert_restored(first, 73)
    assert_restored(second, 29)
    assert_restored(first, 73)


def test_string_key_and_scale_unchanged(tmp_path):
    row = staged_row(tmp_path / 'tree', body=
        'from metrics import VALUE\n'
        'def eval(predictions, answers):\n'
        '    return {"accuracy": 0.75}\n')
    row['metric_authority'].update(key='accuracy', scale_to_100=100, uses_source=False)
    assert_restored(row, 75)


def test_all_dependency_parents_follow_evaluation_directory(tmp_path):
    row = staged_row(tmp_path / 'tree', body=
        'from metrics import VALUE\n'
        'from extra_metric import OFFSET\n'
        'import sys\n'
        'def eval(*args):\n'
        '    assert sys.path[0].endswith("evaluations")\n'
        '    assert sys.path[1].endswith("tree")\n'
        '    assert sys.path[2].endswith("extra")\n'
        '    return {"sari": ({"sari": VALUE + OFFSET},)}\n')
    extra = tmp_path / 'extra'
    extra.mkdir()
    source = extra / 'extra_metric.py'
    source.write_text('OFFSET = 2\n')
    row['metric_authority']['dependencies'].append(dict(path=str(source), sha256=sha(source)))
    assert_restored(row, 75)


@pytest.mark.parametrize('key', [None, 0, True, [], ['sari', True], ['sari', {}], ('sari',)])
def test_invalid_key_paths_refused(tmp_path, key):
    row = staged_row(tmp_path / 'tree')
    row['metric_authority']['key'] = key
    assert_restored(row, error=ValueError)


@pytest.mark.parametrize('value', ['float("nan")', 'float("inf")', '-1', '101'])
def test_range_check_preserved(tmp_path, value):
    row = staged_row(tmp_path / 'tree', value=value)
    assert_restored(row, error=ValueError)


@pytest.mark.parametrize('body', [
    'from metrics import VALUE\nraise RuntimeError("import failed")\n',
    'from metrics import VALUE\ndef eval(*args):\n    raise RuntimeError("call failed")\n',
    'from metrics import load_metric\ndef eval(*args):\n    return load_metric("rouge")\n',
    'from metrics import load_metric\ndef eval(*args):\n    return load_metric("sari")\n',
])
def test_failures_restore_import_state(tmp_path, monkeypatch, body):
    row = staged_row(tmp_path / 'tree', body=body)
    damaged = tmp_path / 'damaged.py'
    damaged.write_text('raise AssertionError("must never execute")')
    monkeypatch.setattr(trace_datasets, 'SARI_PATH', damaged)
    assert_restored(row, error=(RuntimeError, ValueError))


@pytest.mark.parametrize('name, message', [
    ('sari', 'vendored SARI hash differs'),
    ('rouge', 'supports only sari'),
])
def test_loader_refusals_precede_execution(tmp_path, monkeypatch, name, message):
    damaged = tmp_path / 'damaged.py'
    damaged.write_text('raise AssertionError("must never execute")')
    monkeypatch.setattr(trace_datasets, 'SARI_PATH', damaged)
    row = staged_row(tmp_path / 'tree', body=
        'from metrics import load_metric\n'
        f'def eval(*args):\n    return load_metric({name!r})\n')
    modules, path = sys.modules.copy(), list(sys.path)
    with pytest.raises(ValueError, match=message):
        trace_metrics.score(row, 'answer')
    assert sys.modules == modules and sys.path == path


def test_upstream_replaced_and_deleted_import_state_is_restored(tmp_path, monkeypatch):
    sentinel = ModuleType('trace_test_sentinel')
    monkeypatch.setitem(sys.modules, sentinel.__name__, sentinel)
    row = staged_row(tmp_path / 'tree', body=
        'from metrics import VALUE\n'
        'import sys\n'
        'def eval(*args):\n'
        '    del sys.modules["trace_test_sentinel"]\n'
        '    sys.modules = {}\n'
        '    sys.path = []\n'
        '    raise RuntimeError("changed state")\n')
    modules, path = sys.modules, sys.path
    assert_restored(row, error=RuntimeError)
    assert sys.modules is modules and sys.path is path


def test_compatibility_is_scoped_and_vendor_computes_offline(tmp_path, monkeypatch):
    # Tokenisation stand-ins keep this test independent of real metric packages.
    bleu = ModuleType('sacrebleu')
    bleu.__version__ = '2.4.3'
    bleu.metrics = SimpleNamespace(bleu=SimpleNamespace(
        _get_tokenizer=lambda name: lambda: lambda sentence: sentence))
    monkeypatch.setitem(sys.modules, 'sacrebleu', bleu)
    monkeypatch.setitem(sys.modules, 'sacremoses', ModuleType('sacremoses'))
    monkeypatch.delitem(sys.modules, 'datasets', raising=False)
    row = staged_row(tmp_path / 'tree', body=
        'from metrics import load_metric\n'
        'import datasets\n'
        'def eval(sources, predictions, answers):\n'
        '    assert datasets.__name__ == "datasets"\n'
        '    metric = load_metric("sari")\n'
        '    assert metric._info().features["sources"]\n'
        '    return {"sari": (metric.compute(sources=sources, predictions=predictions,\n'
        '                                    references=[[a] for a in answers]),)}\n')
    assert_restored(row, 100)
    assert 'datasets' not in sys.modules


def test_vendor_provenance_and_template_are_bound():
    provenance = json.loads((trace_datasets.SARI_PATH.parent / 'PROVENANCE.json').read_text())
    assert sha(trace_datasets.SARI_PATH) == provenance['sha256'] == trace_datasets.SARI_SHA256
    assert provenance['tag'] == '2.21.0' and provenance['owner_approved']
    root = Path(trace_metrics.__file__).parent / 'resources/trace/operator'
    template = json.loads((root / 'task-authorities-template.json').read_text())
    assert len(template) == 8
    assert template['20Minuten']['key'] == ['sari', 0, 'sari']
    assert all(len(authority['dependencies']) == 1 for authority in template.values())


def test_third_party_imports_survive_without_reload(tmp_path, monkeypatch):
    """Release review 12 Nov: evicting numpy/nltk after each call re-imported them per
    prediction (NumPy reload warning, hours of scoring time). Only upstream-owned modules go."""
    library = tmp_path / 'site'
    library.mkdir()
    (library / 'third_party_fake_lib.py').write_text('LOADS = []\nLOADS.append(1)\n')
    monkeypatch.syspath_prepend(str(library))
    monkeypatch.delitem(sys.modules, 'third_party_fake_lib', raising=False)
    row = staged_row(tmp_path / 'tree', body=
        'import third_party_fake_lib\n'
        'from metrics import VALUE\n'
        'def eval(sources, predictions, answers):\n'
        '    return {"sari": ({"sari": VALUE + len(third_party_fake_lib.LOADS)},)}\n')
    assert trace_metrics.score(row, 'answer') == 74
    first = sys.modules['third_party_fake_lib']
    assert trace_metrics.score(row, 'answer') == 74
    assert sys.modules['third_party_fake_lib'] is first and first.LOADS == [1]
    for name in ('metrics', 'eval_fake', 'trace_upstream_metric', 'trace_hf_sari_2_21_0'):
        assert name not in sys.modules
    assert getattr(sys.modules.get('datasets'), 'load_metric', None) is not trace_datasets.load_metric


def test_module_holding_the_stand_in_is_evicted(tmp_path, monkeypatch):
    library = tmp_path / 'site'
    library.mkdir()
    (library / 'grabs_datasets_fake.py').write_text('import datasets as HELD\n')
    monkeypatch.syspath_prepend(str(library))
    monkeypatch.delitem(sys.modules, 'grabs_datasets_fake', raising=False)
    row = staged_row(tmp_path / 'tree', body=
        'import grabs_datasets_fake\n'
        'from metrics import VALUE\n'
        'def eval(sources, predictions, answers):\n'
        '    return {"sari": ({"sari": VALUE},)}\n')
    assert trace_metrics.score(row, 'answer') == 73
    assert 'grabs_datasets_fake' not in sys.modules


@pytest.mark.parametrize('module, ok', [('difflib', True), ('fuzzywuzzy.StringMatcher', False)])
def test_fuzzywuzzy_backend_pinned_to_difflib(tmp_path, monkeypatch, module, ok):
    matcher = type('SequenceMatcher', (), {'__module__': module})
    monkeypatch.setitem(sys.modules, 'fuzzywuzzy.fuzz', SimpleNamespace(SequenceMatcher=matcher))
    row = staged_row(tmp_path / 'tree')
    if ok:
        assert trace_metrics.score(row, 'answer') == 73
    else:
        with pytest.raises(Exception, match='difflib matcher'):
            trace_metrics.score(row, 'answer')
