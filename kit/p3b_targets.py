#!/usr/bin/env python3
"""Partner job: extend A's completed teacher journal over the selected full pool."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_common import *
import shutil


def selected(out, standin):
    from kit.p3a_common import frozen, decision, INPUTS
    if not (out / INPUTS / 'prepared/frozen.json').exists(): raise NotReady('package A inputs/selection pending')
    freeze = frozen(out, standin)
    for task in s.ORDER:
        row = decision(out, task, 2, freeze, standin)
        if not row.get('complete'): raise NotReady('package A selection pending')
        if row['status'] == 'PASS': return task
        require(row['status'] == 'FAIL', 'unexpected package A decision')
    print('SKIP: neither task qualified'); return None


def job(a):
    from kit.p3a_common import INPUTS, OUTPUTS
    task = selected(a.package_a, a.standin)
    if task is None: return
    base = a.package_a / INPUTS / 'prepared' / task
    prior = a.package_a / OUTPUTS / task / 'teacher'
    require((prior.parent / 'teacher-complete.json').exists(), 'A teacher journal not complete')
    root = a.out / 'targets'; root.mkdir(parents=True, exist_ok=True)
    observed = {'technical_synthetic': True} if a.standin else dict(teacher=identity(a.model, 'teacher'), tokenizer=identity(a.tokenizer, 'student'))
    if not a.standin:
        for role,path in [('teacher',prior.parent/'teacher-identity.json'),('student',a.package_a/INPUTS/'student-model.json')]:
            require(path.is_file(),'missing archived A model identity: '+role)
            saved=json.loads(path.read_text()); expected=json.loads((R/'model-hashes.json').read_text())[role]['files']
            require(all(saved.get('model_file_hashes',{}).get(k)==v for k,v in expected.items()),'incompatible archived A '+role+' identity; do not resample')
    bind(root / 'inputs.json', dict(task=task, tests_path=str((base/'tests.jsonl').resolve()) if task=='code' else None, tests_sha256=s.file_hash(base/'tests.jsonl') if task=='code' else None, pool=s.file_hash(base / 'train.jsonl'), raw={p.name:s.file_hash(p) for p in sorted((prior / 'raw').glob('*.json'))}, **observed))
    if (root / 'manifest.json').exists(): frozen_files(root); print('SKIP: common targets'); return
    for name in ('raw', 'timings'):
        for path in (prior / name).glob('*.json'):
            dest = root / 'teacher' / name / path.name; dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists(): check(dest, s.file_hash(path))
            else: shutil.copyfile(path, dest)
    items = t.read_jsonl(base / 'train.jsonl')
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, V4_PHASE0_INFERENCE='1')
    import time
    started=time.monotonic()
    if a.standin:
        from kit.sim.p3a_row import Tokenizer, Generator
        tokenizer, generator = Tokenizer(), Generator(task, 'lcb-pass')
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        generator = t.VLLMGenerator(a.model, 2560, tensor_parallel_size=2, allow_batch_fallback=False)
    try: s.run_phase(items, root / 'teacher', generator, tokenizer, synthetic=a.standin, tests=str(base / 'tests.jsonl') if task == 'code' else None)
    except ValueError as exc: raise CorrectnessStop(str(exc)) from exc
    demos = t.read_jsonl(root / 'teacher/demonstrations.jsonl')
    require(bool(demos), 'zero valid targets; selected task undefined')
    lookup = {r['id']:r for r in items}
    rows = [dict(id=d['id'], prompt=t.student_messages(lookup[d['id']]), response=d['text'], task=task,
                 gold=lookup[d['id']]['gold']) for d in demos]
    t.write_jsonl(root / 'pool.jsonl', rows)
    names = ['pool.jsonl', 'inputs.json']
    for seed in range(101,106):
        order = cycle(rows, seed, 1280); name = f'schedule-{seed}.jsonl'
        t.write_jsonl(root / name, order); names.append(name)
    atomic(root / 'cost.json',dict(kind='new_teacher_preparation',gpu_hours=0 if a.standin else 2*(time.monotonic()-started)/3600,reused_A_charged=False,technical_synthetic=a.standin))
    atomic(root / 'manifest.json', files_manifest(root, names, task=task, valid=len(rows), population=len(items), exposures=1280,
        coverage=len(rows)/len(items), repeated_exposures=1280-min(1280,len(rows)), technical_synthetic=a.standin))


def configure(p):
    device(p, '0,1'); p.add_argument('--tokenizer'); p.add_argument('--package-a', type=Path, required=True)
if __name__ == '__main__': raise SystemExit(entry(job, configure, 'p3b-targets'))
