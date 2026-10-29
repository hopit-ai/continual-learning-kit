"""Screen S local-only identities, immutable rows and registration checks."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT/'kit/resources/s/registration.json'
DOMAINS = ('ifeval', 'gsm8k', 'arc', 'finqa')
PANELS = dict(ifeval='ifeval', gsm8k='math', arc='knowledge', finqa='finqa')
SEEDS = (101, 102, 103)
PRIMARY_STAGES = 2*2*len(SEEDS)
INCOMING_CHECKPOINTS = 1+2*len(SEEDS)
PROBES = 3+2*len(SEEDS)*4


def inventory_slots():
    """Original v4 identities in the authorised S cohort; never create exports."""
    from kit.v4_campaign import slots
    return [row for row in slots() if row['seed'] in SEEDS]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    from kit.m2_paths import local
    path=local(path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    from kit.m2_paths import local
    path=local(path)
    return json.loads(Path(path).read_text())


def rows(path):
    from kit.m2_paths import local
    path=local(path)
    from kit.v4_teacher import read_jsonl
    return read_jsonl(path)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temp.replace(path)


def bind(path, value):
    if Path(path).exists():
        require(__import__('kit.m2_paths',fromlist=['equivalent']).equivalent(read(path),value), 'frozen input changed: '+str(path))
    else:
        atomic(path, value)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(v, sort_keys=True)+'\n' for v in values)
    if path.exists():
        require(path.read_text() == text, 'frozen rows changed: '+str(path))
    else:
        path.write_text(text)


def checked(ref):
    path = Path(ref['path'])
    require(sha(path) == ref['sha256'], 'evidence SHA256 mismatch: '+str(path))
    from kit.m2_paths import local
    return local(path)


def registration():
    doc = read(RESOURCE)
    for name, expected in doc['code'].items():
        require(sha(ROOT/name) == expected, 'registered code changed: '+name)
    return doc


def context(out, standin):
    registration()
    doc = read(Path(out)/'binding.json')
    require(doc['technical_synthetic'] is standin, 'technical_synthetic requires --standin')
    require(doc['registration_sha256'] == sha(RESOURCE), 'registration changed after binding')
    return doc


def model(ref, standin):
    if standin:
        require(ref.get('technical_synthetic') is True, 'stand-in checkpoint required')
        return dict(technical_synthetic=True, identity=ref['identity'])
    from kit.p3b_common import identity
    require(ref.get('receipt'), 'original export receipt required')
    return identity(ref['model'], receipt=ref['receipt'])


def entry(job, configure):
    import argparse,subprocess
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--standin', action='store_true')
    configure(p)
    a = p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from kit.inference_lifecycle import marker,shutdown_since
    from kit.p3a_common import NotReady
    mark=marker()
    try:
        try:job(a)
        finally:
            active_error=sys.exc_info()[0] is not None
            try:shutdown_since(mark)
            except Exception as shutdown_error:
                if not active_error:raise
                print('inference shutdown also failed:',shutdown_error,file=sys.stderr)
    except NotReady as exc:
        print('PENDING:',exc);return 3
    except subprocess.CalledProcessError as exc:
        if exc.returncode in (2,3):return exc.returncode
        raise
    except (ValueError, FileNotFoundError) as exc:
        print('STOP:', exc)
        return 2
    return 0
