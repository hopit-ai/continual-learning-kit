"""Pinned OpenCompass general evaluators; stand-ins provide package glue only."""
from contextlib import contextmanager
import ast
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import sys
import types
import warnings

from kit.trace_common import KIT, load, need, sha
from kit.trace_datasets import UPSTREAM_IMPORT_LOCK

VENDOR = KIT/'resources/trace/vendor/opencompass-0.1.5'
PANELS = ('MMLU', 'BBH', 'GSM8K', 'TyDiQA', 'PIQA', 'BoolQ')
PIN = '9b21613d171ccac154530a3c7bbeba6743b469f4'


def verify_vendor():
    provenance = load(VENDOR/'PROVENANCE.json')
    need(provenance['commit'] == PIN, 'OpenCompass pin differs')
    for rel, expected in provenance['sha256'].items():
        need(sha(VENDOR/rel) == expected, 'vendored OpenCompass identity differs: '+rel)
    return provenance


@lru_cache(maxsize=1)
def _verified_vendor():
    return verify_vendor()


def _bbh_subtasks():
    tree = ast.parse((VENDOR/'configs/datasets/bbh/bbh_gen_5b92b0.py').read_text())
    return {node.targets[0].id: frozenset(ast.literal_eval(node.value))
            for node in tree.body if isinstance(node, ast.Assign) and
            isinstance(node.targets[0], ast.Name) and
            node.targets[0].id in ('bbh_multiple_choice_sets', 'bbh_free_form_sets')}


BBH_SUBTASKS = _bbh_subtasks()


class Registry:
    def register_module(self, name=None, *args, **kwargs):
        if callable(name):
            return name
        return lambda obj: obj


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        spec.loader.exec_module(module)
    return module


@contextmanager
def upstream():
    """Restore import objects and upstream-owned entries on success AND failure.

    Third-party modules (including C extensions) remain loaded, as in trace_metrics.
    No local/remote dataset loader is ever invoked.
    """
    with UPSTREAM_IMPORT_LOCK:
        _verified_vendor()
        # Import real numerical dependencies before masking package glue.
        import numpy
        import sklearn.metrics
        path_object, modules_object = sys.path, sys.modules
        original_path, original_modules = list(sys.path), sys.modules.copy()
        roots = ('opencompass', 'datasets', 'evaluate', 'trace_oc_accuracy')
        def owned(name):
            return any(name == root or name.startswith(root+'.') for root in roots)
        def package(name, path=None):
            obj = types.ModuleType(name)
            obj.__path__ = [str(path)] if path else []
            sys.modules[name] = obj
            return obj
        def module(name):
            obj = types.ModuleType(name)
            sys.modules[name] = obj
            return obj
        original_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            for name in list(sys.modules):
                if owned(name):
                    del sys.modules[name]
            oc = VENDOR/'opencompass'
            for name, path in [('opencompass', oc), ('opencompass.openicl', oc/'openicl'),
                               ('opencompass.utils', oc/'utils'),
                               ('opencompass.datasets', oc/'datasets')]:
                package(name, path)
            reg = module('opencompass.registry')
            for name in ('ICL_EVALUATORS', 'LOAD_DATASET', 'TEXT_POSTPROCESSORS'):
                setattr(reg, name, Registry())
            base = module('opencompass.datasets.base')
            base.BaseDataset = type('BaseDataset', (), {})
            ds = module('datasets')
            ds.Dataset = type('Dataset', (), {})
            def forbidden(*args, **kwargs):
                raise RuntimeError('dataset loading is forbidden in the metric bridge')
            ds.load_dataset = forbidden
            for name in ('Features', 'Value', 'Sequence'):
                setattr(ds, name, lambda *args, **kwargs: None)
            ev = module('evaluate')
            class Metric:
                config_name = 'default'
                def compute(self, **kwargs):
                    return self._compute(**kwargs)
            ev.Metric = Metric
            ev.MetricInfo = lambda **kwargs: kwargs
            ev.utils = types.SimpleNamespace(file_utils=types.SimpleNamespace(
                add_start_docstrings=lambda *args: lambda obj: obj))
            accuracy = oc/'openicl/icl_evaluator/hf_metrics/accuracy.py'
            accuracy_class = _load('trace_oc_accuracy', accuracy).Accuracy
            def local_accuracy(path):
                need(Path(path).resolve() == accuracy.resolve(), 'only local accuracy is permitted')
                return accuracy_class()
            ev.load = local_accuracy
            evaluator = package('opencompass.openicl.icl_evaluator', oc/'openicl/icl_evaluator')
            evaluator.BaseEvaluator = _load('opencompass.openicl.icl_evaluator.icl_base_evaluator',
                oc/'openicl/icl_evaluator/icl_base_evaluator.py').BaseEvaluator
            hf = _load('opencompass.openicl.icl_evaluator.icl_hf_evaluator',
                       oc/'openicl/icl_evaluator/icl_hf_evaluator.py')
            _load('opencompass.utils.text_postprocessors', oc/'utils/text_postprocessors.py')
            modules = {name: _load('opencompass.datasets.'+name, oc/'datasets'/f'{name}.py')
                       for name in ('gsm8k', 'bbh', 'tydiqa')}
            yield dict(AccEvaluator=hf.AccEvaluator, **modules)
        finally:
            sys.dont_write_bytecode = original_bytecode
            sys.path, sys.modules = path_object, modules_object
            sys.path[:] = original_path
            for name in sys.modules.keys() - original_modules.keys():
                location = getattr(sys.modules[name], '__file__', None)
                if owned(name) or (location and Path(location).resolve().is_relative_to(VENDOR.resolve())):
                    del sys.modules[name]
            sys.modules.update(original_modules)


def _score(oc, row, prediction):
    panel, gold = row['panel'], row['answer']
    need(panel in PANELS, 'unknown TRACE general metric')
    if panel in ('MMLU', 'PIQA', 'BoolQ') and isinstance(prediction, dict):
        from kit.trace_general_score import select_label
        prediction = select_label(row['labels'], prediction['losses'])
    if panel == 'GSM8K':
        prediction = oc['gsm8k'].gsm8k_postprocess(prediction)
    if panel == 'TyDiQA':
        need(type(gold) is list and gold and all(isinstance(x, str) for x in gold),
             'TyDiQA requires a nonempty list of gold answers')
        value = oc['tydiqa'].TydiQAEvaluator().score([prediction], [gold])['f1']
    elif panel == 'BBH':
        subset = row['subset']
        if subset in BBH_SUBTASKS['bbh_multiple_choice_sets']:
            process = oc['bbh'].bbh_mcq_postprocess
            value = oc['AccEvaluator']().score([process(prediction)], [process(gold)])['accuracy']
        else:
            need(subset in BBH_SUBTASKS['bbh_free_form_sets'], 'unknown BBH subtask')
            value = oc['bbh'].BBHEvaluator().score([prediction], [gold])['score']
    else:
        value = oc['AccEvaluator']().score([prediction], [gold])['accuracy']
    need(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 100,
         'OpenCompass metric range differs')
    return value


def score_many(rows, predictions):
    """One isolated upstream context, with unchanged per-item evaluator arithmetic."""
    need(len(rows) == len(predictions), 'general predictions missing')
    with upstream() as oc:
        return [_score(oc, row, prediction) for row, prediction in zip(rows, predictions)]


def score(row, prediction):
    return score_many([row], [prediction])[0]


def gsm8k_gold(text):
    with upstream() as oc:
        return oc['gsm8k'].gsm8k_dataset_postprocess(text)
