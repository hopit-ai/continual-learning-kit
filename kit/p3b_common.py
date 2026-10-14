"""Plain package-B IO and identities; no scheduling or campaign state."""
import json
import os
from pathlib import Path
import random
from kit.p3a_common import (KIT, CorrectnessStop, NotReady, atomic, check, entry as plain_entry, logging,
                           model_path, retain_partial, s, t)
R = KIT / 'resources/p3b'


def require(ok, message):
    if not ok: raise CorrectnessStop(message)


def bind(path, value):
    if path.exists(): require(json.loads(path.read_text()) == value, 'input identity changed: ' + str(path))
    else: atomic(path, value)


def identity(path, role=None, receipt=None):
    observed = t.local_identity(model_path(path), s.MODELS[role][0] if role else 'Qwen/Qwen3-8B')
    if role:
        require(observed['model_revision'] == s.MODELS[role][1], 'wrong/unknown pinned ' + role + ' revision')
        expected=json.loads((R / 'model-hashes.json').read_text())[role]['files']
        require(all(observed['model_file_hashes'].get(k)==v for k,v in expected.items()),'pinned '+role+' files differ from v4 model registration')
    if receipt:
        expected = json.loads(Path(receipt).read_text())
        require(observed['model_file_hashes'] == expected['model_file_hashes'], 'checkpoint differs from export receipt')
    return observed


def files_manifest(root, names, **extra):
    return dict(files={n: s.file_hash(root / n) for n in names}, **extra)


def frozen_files(root):
    if not (root / 'manifest.json').exists(): raise NotReady('missing freeze: ' + str(root))
    doc = json.loads((root / 'manifest.json').read_text())
    for name, expected in doc['files'].items(): check(root / name, expected)
    return doc


def cycle(rows, seed, n):
    require(bool(rows), 'empty training pool')
    order = list(rows); random.Random(seed).shuffle(order)
    return [order[i % len(order)] for i in range(n)]


def mixed(task, replay, seed):
    require(len(task) == 1280, 'task schedule must be 40 x 32')
    generic = iter(cycle(replay, seed, 160))
    return [next(generic) if i % 32 in (7, 15, 23, 31) else row for i, row in enumerate(task)]


def device(p, default='0'):
    p.add_argument('--gpu', default=default)
    p.add_argument('--model')


def frozen_hashes():
    from kit.p3a_common import registration
    registration()
    for line in (R / 'SHA256SUMS').read_text().splitlines():
        expected,name=line.split(); check(R / Path(name).name,expected)
    from kit.chemistry_shortcut_report import EXPECTED_MANIFEST_SHA256
    root = KIT / 'resources/3a-shortcut'
    check(root / 'manifest.json', EXPECTED_MANIFEST_SHA256)
    doc = json.loads((root / 'manifest.json').read_text())
    for name, expected in doc['files'].items(): check(root / name, expected)
    for name, expected in doc['code'].items(): check(KIT.parent / name, expected)
    return {'3c': __import__('kit.p3a_common', fromlist=['EXPECTED']).EXPECTED, '3a': EXPECTED_MANIFEST_SHA256}


def entry(job, configure, name):
    def logged(a):
        for key in list(os.environ):
            if key.startswith('V4_'): os.environ.pop(key)
        try: return job(a)
        except (CorrectnessStop, NotReady): raise
        except Exception:
            import traceback
            traceback.print_exc()  # Capture the crash before the tee context closes.
            raise
    return plain_entry(logged, configure, name)
