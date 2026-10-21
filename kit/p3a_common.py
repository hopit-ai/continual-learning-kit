"""Small helpers for plain package-A jobs; no scheduler or campaign state."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from kit import second_task_screen as s, v4_teacher as t

KIT = Path(__file__).resolve().parent
RESOURCES = KIT / 'resources/3c-screen'
INPUTS = Path('3c/report-inputs')
OUTPUTS = Path('3c/report-screen')
EXPECTED = 'c75c3d5a2fcd140f434316fb768f6d590a056e51a3e410d56abb433ef9c64d37'

from kit.v4_finqa_policy import CorrectnessStop as PolicyStop


class CorrectnessStop(PolicyStop):
    pass


class NotReady(RuntimeError):
    """A prerequisite step has not finished (or finished incomplete); run that first."""


def atomic(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    t.atomic_json(path, value)


def check(path, expected):
    if not Path(path).is_file() or s.file_hash(path) != expected:
        raise CorrectnessStop('input hash differs or missing: ' + str(path))


def implementation_path(name):
    path = KIT.parent / name
    if name == 'scripts/second_task_screen_modal.py' and not path.is_file():
        return RESOURCES / 'second_task_screen_modal.py'
    return path


def registration():
    doc = json.loads((RESOURCES / 'public-inputs.json').read_text())
    for name, expected in doc['resources'].items(): check(RESOURCES / name, expected)
    check(KIT / 'materialise_math_mirror.py', doc['materialiser_sha256'])
    seal = json.loads((RESOURCES / 'frozen-annex.json').read_text())
    if seal['manifest_sha256'] != EXPECTED or s.digest(seal['manifest']) != EXPECTED:
        raise CorrectnessStop('wrong annex manifest')
    for name, expected in seal['manifest']['implementation'].items(): check(implementation_path(name), expected)
    return doc


def frozen(out, standin=False):
    root = out / INPUTS / 'prepared'
    if not (root / 'frozen.json').is_file() or not (out / INPUTS / 'frozen-annex.json').is_file():
        raise CorrectnessStop('missing frozen input manifest/annex')
    try:
        seal = json.loads((out / INPUTS / 'frozen-annex.json').read_text())
        manifest = json.loads((root / 'frozen.json').read_text())
    except (OSError, ValueError) as exc: raise CorrectnessStop('wrong annex manifest: ' + str(exc)) from exc
    if (manifest != seal.get('manifest') or s.digest(manifest) != seal.get('manifest_sha256')
            or bool(manifest.get('technical_synthetic')) != standin
            or (not standin and seal['manifest_sha256'] != EXPECTED)):
        raise CorrectnessStop('wrong annex manifest')
    check(out / INPUTS / 'preregistration.md', manifest['preregistration_sha256'])
    for name, expected in manifest['implementation'].items(): check(implementation_path(name), expected)
    for candidate, meta in manifest['candidates'].items():
        for name, expected in meta['files'].items(): check(root / candidate / name, expected)
    return seal['manifest_sha256']


def decision(out, candidate, step, freeze, standin):
    path = out / (INPUTS / (candidate + '-step1.json') if step == 1 else OUTPUTS / candidate / 'decision.json')
    if not path.exists(): return {}
    row = json.loads(path.read_text())
    if (row.get('candidate') != candidate or row.get('step') != step
            or row.get('freeze_sha256') != freeze or bool(row.get('technical_synthetic')) != standin):
        raise CorrectnessStop('wrong decision input identity: ' + str(path))
    return row


def allowed(out, candidate, freeze, standin, baseline=False):
    # Registered order: a complete result that rules a step out is a SKIP (exit 0);
    # a missing or incomplete prerequisite is NOT READY (exit 3).
    if candidate == 'math_screen':
        previous = decision(out, 'code', 2, freeze, standin)
        if not previous.get('complete'): raise NotReady('MATH needs a complete LCB step-2 decision')
        if previous.get('status') != 'FAIL':
            print('SKIP: LCB step 2 is a complete', previous.get('status') + '; MATH is not run'); return False
    if not baseline:
        row = decision(out, candidate, 1, freeze, standin)
        if not row.get('complete'): raise NotReady('step 2 needs a complete step-1 decision for ' + candidate)
        if row.get('status') != 'PASS':
            print('SKIP: step 1 for', candidate, 'is a complete', row.get('status') + '; step 2 is not run'); return False
    return True


def model_path(value):
    if not value or not (Path(value) / 'config.json').is_file():
        raise CorrectnessStop('missing model path: ' + str(value))
    return str(Path(value).resolve())


def retain_partial(path):
    if path.exists():
        dest = path.parent / 'partials' / (path.name + '-' + str(time.time_ns()))
        dest.parent.mkdir(exist_ok=True); path.rename(dest)
        print('WARNING: retained unfinished output at', dest)


@contextmanager
def logging(out, name):
    logs = out / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    saved = [os.dup(fd) for fd in (1, 2)]
    tee = subprocess.Popen(['tee', '-a', str(logs / (name + '.log'))], stdin=subprocess.PIPE, stdout=saved[0], stderr=saved[1])
    try:
        sys.stdout.flush(); sys.stderr.flush()
        for fd in (1, 2): os.dup2(tee.stdin.fileno(), fd)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        for fd, old in zip((1, 2), saved): os.dup2(old, fd); os.close(old)
        tee.stdin.close()
        # INFRASTRUCTURE FIX (not a decision rule). vLLM worker subprocesses are spawned
        # while fds 1,2 point at the tee pipe, so they inherit the write end. If the
        # engine is not shut down before this exits, tee never sees EOF and tee.wait()
        # blocks forever -- observed 2026-10-16, p3a_teacher.py hung 3h25m AFTER writing
        # every output. Bound the wait and force tee closed; the task script reaps any
        # surviving GPU workers afterwards.
        try:
            tee.wait(timeout=120)
        except subprocess.TimeoutExpired:
            sys.stderr.write('logging(): tee did not exit in 120s (inherited pipe); killing it\n')
            tee.kill(); tee.wait()


def entry(job, configure, name):
    p = argparse.ArgumentParser(description=job.__doc__)
    p.add_argument('--out', type=Path, default=os.environ.get('OUT'), required=not os.environ.get('OUT'))
    p.add_argument('--standin', action='store_true', help='CPU plumbing only; never scientific output')
    configure(p); a = p.parse_args(); a.out = a.out.resolve()
    with logging(a.out, name.format(**vars(a))):
        # Do not inherit campaign accounting or gates from a caller's environment.
        for key in ('V4_FINQA_CAMPAIGN', 'V4_PHASE0_MODE', 'V4_ALLOCATION_PLAN', 'KIT_V4_ARM', 'V4_BATCH_FALLBACK_CAUSE'):
            os.environ.pop(key, None)
        try: job(a)
        except PolicyStop as exc: print("STOP: correctness:", exc); return 2
        except NotReady as exc: print('NOT READY:', exc); return 3
        # Any other error propagates with its traceback and a non-zero exit.
    return 0
