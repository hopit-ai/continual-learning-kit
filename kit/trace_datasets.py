"""Offline datasets API surface used by the pinned Hugging Face SARI script."""
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import threading

from kit.trace_common import need


SARI_PATH = (Path(__file__).resolve().parent / 'resources/trace/vendor' /
             'hf-datasets-2.21.0-sari/sari.py')
SARI_SHA256 = '3816e522bc92f40c0d02313355da13964d045a99db1c07e9d18ccdb78d23eed4'
# All adapters loaded by metric_adapter share this import-state lock.
UPSTREAM_IMPORT_LOCK = threading.RLock()


class MetricInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Metric:
    def compute(self, **kwargs):
        return self._compute(**kwargs)


def _feature(*args, **kwargs):
    return args, kwargs


def _add_start_docstrings(*docs):
    return lambda obj: obj


def load_metric(name):
    need(name == 'sari', 'offline datasets compatibility supports only sari')
    source = SARI_PATH.read_bytes()
    need(hashlib.sha256(source).hexdigest() == SARI_SHA256,
         'vendored SARI hash differs')
    module = ModuleType('trace_hf_sari_2_21_0')
    module.__file__ = str(SARI_PATH)
    # Compile the verified bytes directly, avoiding bytecode and network loaders.
    exec(compile(source, str(SARI_PATH), 'exec'), module.__dict__)
    return module.Sari()


def compatibility_module():
    module = ModuleType('datasets')
    module.load_metric = load_metric
    module.Metric = Metric
    module.MetricInfo = MetricInfo
    module.Features = dict
    module.Value = _feature
    module.Sequence = _feature
    module.utils = SimpleNamespace(file_utils=SimpleNamespace(
        add_start_docstrings=_add_start_docstrings))
    return module
