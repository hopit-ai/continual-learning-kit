"""TRACE metric adapter for locally staged, hash-bound upstream metric code.

Node rows supply the exact task metric callable and scale in metric_authority.
No upstream source or dataset is downloaded by this module.
"""
import importlib.util
import math
import sys
from pathlib import Path
from kit.trace_common import bound_file, need
from kit.trace_datasets import SARI_PATH, UPSTREAM_IMPORT_LOCK, compatibility_module

VENDOR_DIR = str(SARI_PATH.parent.resolve())
UPSTREAM_MODULE_NAMES = frozenset({'datasets', 'trace_upstream_metric', 'trace_hf_sari_2_21_0'})


def score(row, prediction):
    authority = row['metric_authority']
    path = bound_file(authority)
    dependencies = [bound_file(ref) for ref in authority.get('dependencies', [])]
    key = authority['key']
    need(isinstance(key, str) or
         (type(key) is list and bool(key) and
          all(type(step) in (str, int) for step in key)),
         'upstream metric key must be a string or nonempty str/int list path')
    steps = [key] if isinstance(key, str) else key
    with UPSTREAM_IMPORT_LOCK:
        result = _upstream_call(path, dependencies, authority, row, prediction)
    fuzz = sys.modules.get('fuzzywuzzy.fuzz')
    if fuzz is not None:
        # fuzz.ratio differs between python-Levenshtein and difflib on unequal
        # strings; Py150 edit similarity is pinned to the pure difflib matcher.
        need(getattr(getattr(fuzz, 'SequenceMatcher', None), '__module__', None) == 'difflib',
             'fuzzywuzzy must use the difflib matcher; uninstall python-Levenshtein')
    for step in steps:
        result = result[step]
    value = result*authority['scale_to_100']
    need(type(value) in (float,int) and math.isfinite(value) and 0<=value<=100,
         'upstream metric range differs')
    return value


def _upstream_call(path, dependencies, authority, row, prediction):
    # Absolute imports must resolve to this staged tree, even after another call.
    directories = list(dict.fromkeys(str(p.parent.resolve())
                                    for p in [path, *dependencies]))
    names = {p.stem for p in [path, *dependencies]}
    names.update(p.stem for directory in directories
                 for p in Path(directory).glob('*.py'))
    spec = importlib.util.spec_from_file_location('trace_upstream_metric',path)
    module = importlib.util.module_from_spec(spec)
    path_object, modules_object = sys.path, sys.modules
    original_path = list(path_object)
    original_modules = modules_object.copy()
    stand_in = None
    try:
        for name in list(sys.modules):
            if name.partition('.')[0] in names:
                del sys.modules[name]
        sys.path[:0] = directories
        stand_in = compatibility_module()
        sys.modules['datasets'] = stand_in
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        fn = getattr(module,authority['function'])
        args = [[prediction],[row['answer']]]
        if authority.get('uses_source'):
            args.insert(0,[row['prompt']])
        return fn(*args)
    finally:
        sys.path, sys.modules = path_object, modules_object
        sys.path[:] = original_path
        # Evict only upstream-tree, vendored and stand-in modules. Third-party
        # libraries first imported here (numpy, nltk, rouge, ...) stay loaded:
        # re-importing C extensions per prediction is unsafe and slow.
        for name in sys.modules.keys() - original_modules.keys():
            if _upstream_owned(name, sys.modules[name], names, directories, stand_in):
                del sys.modules[name]
        sys.modules.update(original_modules)


def _upstream_owned(name, module, names, directories, stand_in):
    if name.partition('.')[0] in names or name in UPSTREAM_MODULE_NAMES:
        return True
    if module is stand_in:
        return True
    location = getattr(module, '__file__', None)
    if location:
        resolved = str(Path(location).resolve())
        if any(resolved.startswith(d.rstrip('/') + '/') for d in (*directories, VENDOR_DIR)):
            return True
    try:
        return any(value is stand_in for value in vars(module).values())
    except TypeError:
        return False


def general(row,prediction):
    from kit.scorers import score as inherited
    panel = {'instruction':'ifeval','maths':'math'}.get(row['panel'],row['panel'])
    return 100*int(inherited(panel,prediction,row)['correct'])
