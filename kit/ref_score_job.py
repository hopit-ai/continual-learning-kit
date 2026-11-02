#!/usr/bin/env python3
"""Score decision checkpoint guards with the unchanged validation reference implementations.

Checkpoint list: JSON array of {arm, seed, step, stage (M3 only)}. Paths and
model hashes are resolved from the native binding/export receipts, never the list.
Each invocation scores one (family, seed, panel) sequentially on its frozen GPU.
"""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import fcntl
import hashlib
import shutil
import json
import os
import time
import subprocess
from contextlib import nullcontext,contextmanager
from kit.p3b_common import atomic, bind, files_manifest, frozen_files, require, s
from kit.ref_guard import METHOD, PANELS, ASSIGNMENT


def source_for(out, family, checkpoint, standin):
    arm, seed, step = (checkpoint[k] for k in ('arm', 'seed', 'step'))
    if family == 'm3':
        from kit.m3_schedule import ARMS, STAGES
        require(arm in ('base', *ARMS) and checkpoint.get('stage') in STAGES, 'unregistered M3 checkpoint')
        from kit.m3_score import checked
        return checked(out, arm, seed, checkpoint['stage'], step, standin)
    if family in ('core','reprobe'):
        from kit.m2_common import binding, record, checked_model
        doc=binding(out,standin)
        if family=='reprobe':
            from kit.m2_reprobe import checkpoint,REPAIR
            require(doc.get('purpose')==REPAIR,'reference re-probe binding differs')
            source=checkpoint(out,doc,arm,seed,step)
        else:source=record(out,doc,arm,seed,step)
        from kit.m2_paths import local
        source = dict(source, model=str(local(source['model'])), receipt=str(local(source['receipt'])) if source.get('receipt') else None)
        return source, checked_model(source, standin, base=arm == 'base')
    from kit.x_common import context, incoming, checkpoint as exported
    from kit.m2_common import checked_model
    doc = context(out, standin)
    if arm == 'base':
        core = json.loads((Path(doc['message2']) / 'binding.json').read_text())
        source = core['starts']['base']
        from kit.m2_paths import local
        source = dict(source, model=str(local(source['model'])), receipt=str(local(source['receipt'])) if source.get('receipt') else None)
        return source, checked_model(source, standin, base=True)
    native_arm = arm.removeprefix('x-')
    pre = native_arm == 'pre' or native_arm.endswith('-pre')
    expected = 'x-tier2.' + native_arm[3:].rsplit('-', 1)[0] if native_arm.startswith('t2-') else 'x-tier1'
    require(family == expected, 'reference checkpoint belongs to another X contrast')
    source = incoming(doc, seed) if pre else exported(out, doc, native_arm, seed, step)
    return source, checked_model(source, standin, base=False)


def load(path): return json.loads(Path(path).read_text())


def assign(a):
    from kit.m2_assignment import load, freeze
    native = a.from_assignment or a.out / ('m3-scoring-assignment.json' if a.family == 'm3' else 'scoring-assignment.json')
    doc = load(native)
    explicit = getattr(a, 'lane_assignment', None)
    if explicit:
        assignments = load(explicit)['assignments']
        expected = {f'{a.family}.{panel}|{seed}' for panel in PANELS for seed in (101, 102, 103)}
        require(set(assignments) == expected, 'lane assignment must name every family/panel/seed exactly once')
        require(all(all(v.get(k) for k in ('node', 'uuid', 'environment')) for v in assignments.values()), 'incomplete lane hardware')
    else:
        assignments = {f'{family}.{panel}|{seed}': value for key, value in doc['assignments'].items()
                       for family, seed in [key.rsplit('|', 1)] if family == a.family for panel in PANELS}
    require(assignments, 'no native assignments for family')
    freeze(a.out / ASSIGNMENT, dict(schema='message2-scoring-assignment.v2', assignments=assignments))
    if getattr(a,'schedule_checkpoints',False):
        require(a.family=='m3' and a.checkpoints is not None,'M3 schedule checkpoint output required')
        from kit.m3_schedule import ARMS,SEEDS,STAGES
        atomic(a.checkpoints,[c for seed in SEEDS for c in
            [dict(arm='base',seed=seed,stage=STAGES[0],step=0),
             *[dict(arm=arm,seed=seed,stage=stage,step=40) for arm in ARMS for stage in STAGES]]])


def lane_checkpoints(a):
    if not a.checkpoints.is_file():
        from kit.p3b_common import NotReady
        raise NotReady('reference checkpoint list absent')
    checkpoints = load(a.checkpoints)
    if isinstance(checkpoints, dict): checkpoints = checkpoints['checkpoints']
    checkpoints = [c for c in checkpoints if c['seed'] == a.seed]
    require(checkpoints, 'no checkpoints for assigned seed')
    require(all(c['step'] in (0, 40) for c in checkpoints), 'reference job accepts decision checkpoints (0/40) only')
    require(len({(c['arm'], c.get('stage'), c['step']) for c in checkpoints}) == len(checkpoints), 'duplicate checkpoint')
    return checkpoints


def file_metadata(path):
    try: st = Path(path).stat()
    except FileNotFoundError as error:
        from kit.p3b_common import NotReady
        raise NotReady('CPU-prepared checkpoint file absent') from error
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def model_metadata(source, observed):
    result = {name: file_metadata(Path(source['model']) / name) for name in observed.get('model_file_hashes', {})}
    if source.get('receipt'):
        result['@receipt'] = file_metadata(source['receipt'])
        for name in ('manifest.json', 'complete.json'):
            path = Path(source['receipt']).parent / name
            if path.exists(): result['@' + name] = file_metadata(path)
    require(result, 'CPU preparation needs model file metadata')
    return result


def prepared_path(a): return a.out / 'reference-ready' / (a.family + '-' + str(a.seed) + '-' + a.panel + '.local.json')


@contextmanager
def lane_writer(a):
    directory=a.out/'reference-ready';directory.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256((a.family+'|'+str(a.seed)).encode()).hexdigest()
    with (directory/(key+'.lock')).open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            from kit.p3b_common import NotReady
            raise NotReady('reference lane already has a writer') from None
        try:yield key
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def prepare(a):
    with lane_writer(a) as key:return prepare_lane(a,key)


def prepare_lane(a,key):

    # Resolve and hash every checkpoint before GPU admission; never start a partial lane.
    checkpoints = lane_checkpoints(a); sources = []
    for checkpoint in checkpoints:
        source, observed = source_for(a.out, a.family, checkpoint, a.standin)
        sources.append(dict(source=source, observed=observed, metadata=None if a.standin else model_metadata(source, observed)))
    # CPU admission for a fresh frontier/retry preserves prior jobs, then every
    # checkpoint is scored anew under the next allocation's actual identity.
    parent=a.out/'reference-scores'/a.family
    previous=[] if os.environ.get('KIT_V6A') else [p for p in parent.glob('*') if p.is_dir() and ('-s'+str(a.seed)+'-k') in p.name]
    if previous:
        history=a.out/'reference-history'/key/str(time.time_ns());history.mkdir(parents=True,exist_ok=False)
        for path in previous:shutil.move(str(path),str(history/path.name))
    atomic(prepared_path(a), dict(checkpoints=checkpoints, sources=sources, assignment_sha256=s.file_hash(a.out / ASSIGNMENT)))


def run_lane(a):
    with lane_writer(a):return run_ready_lane(a)


def run_ready_lane(a):
    """Admit both complete panel lanes, then score independent checkpoint cells."""
    from copy import copy
    from kit.lane_pool import execute
    panels=PANELS if a.panel=='both' else (a.panel,)
    checkpoints=lane_checkpoints(a)
    for panel in panels:
        child=copy(a);child.panel=panel;ready=prepared_path(child)
        if not ready.is_file():
            from kit.p3b_common import NotReady
            raise NotReady('CPU preparation missing for complete reference lane')
        prepared=load(ready)
        require(prepared['checkpoints']==checkpoints and prepared['assignment_sha256']==s.file_hash(a.out/ASSIGNMENT),'reference preparation changed')
        require(len(prepared['sources'])==len(checkpoints),'incomplete reference preparation')
        for item in prepared['sources']:
            if not getattr(a,'standin',False):require(model_metadata(item['source'],item['observed'])==item['metadata'],'prepared reference checkpoint changed')
    commands=[]
    for index in range(len(checkpoints)):
        commands.append([os.environ.get('INFERENCE_PY',sys.executable),'-I',str(Path(__file__).resolve()),'--action','score','--out',str(a.out),
            '--family',a.family,'--seed',str(a.seed),'--checkpoints',str(a.checkpoints),'--panel',a.panel,
            '--gpu',a.gpu,'--cell-index',str(index),*(['--standin'] if getattr(a,'standin',False) else [])])
    from kit.p3b_common import NotReady
    import uuid
    nonce=os.environ.get('KIT_LANE_JOB') or uuid.uuid4().hex
    attempt=a.out/'reference-attempts'/a.family/str(a.seed)/nonce
    attempt.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,REF_POOL_CHILD='1',REF_ATTEMPT_ROOT=str(attempt),KIT_LANE_JOB=nonce)
    fds=tuple(json.loads(env.get('M3_GPU_LOCK_FDS','[]')))
    code=execute(commands,attempt/'pool',env=env,pass_fds=fds,concurrency=2)
    if code in (3,75):raise NotReady('reference lane incomplete')
    require(code==0,'reference lane failed')
    from kit.lane_evidence import complete_attempt
    identities=[]
    for panel in panels:
        for checkpoint in checkpoints:
            arm=checkpoint['arm'];unit=arm if a.family!='m3' or arm=='base' else arm+'.'+checkpoint['stage']
            identities.append(f"reference-scores/{a.family}/{unit}-s{a.seed}-k{checkpoint['step']}-{panel}/ref-identity.json")
    complete_attempt(attempt,nonce,identities)


def run(a):
    keys = ('M2_ASSIGNMENT', 'M2_FAMILY', 'M2_FAMILY_SEED', 'CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'M2_ASSIGNED_GPU_UUID')
    previous = {k: os.environ.get(k) for k in keys}
    try:
        if a.panel!='both':return _run(a)
        from copy import copy
        checkpoints=lane_checkpoints(a)
        for index in range(len(checkpoints)):
            if getattr(a,'cell_index',None) is not None and a.cell_index!=index:continue
            cache={}
            try:
                for panel in PANELS:
                    child=copy(a);child.panel=panel;child.cell_index=index;child.reference_cache=cache
                    _run(child)
            finally:
                if cache:
                    cache.clear()
                    import torch
                    torch.cuda.empty_cache()

    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def _run(a):
    from kit import m2_assignment
    from kit.m3_score import normalized_assignment_inventory, knowledge_items, knowledge_reference, standin_likelihoods, selected
    from kit import m2_format
    from kit.hellaswag_reference import score as hellaswag_reference
    from kit.p3b_score import write_csv
    from kit.c_score import COLUMNS
    require(a.panel in PANELS and a.seed in (101, 102, 103), 'registered panel/seed required')
    checkpoints = lane_checkpoints(a)
    ready = prepared_path(a)
    if a.standin:
        sources = [source_for(a.out, a.family, c, True) for c in checkpoints]
    else:
        if not ready.exists():
            from kit.p3b_common import NotReady
            raise NotReady('prepare every reference checkpoint on CPU before GPU admission')
        prepared = load(ready)
        require(prepared['checkpoints'] == checkpoints and prepared['assignment_sha256'] == s.file_hash(a.out / ASSIGNMENT), 'reference preparation changed')
        require(len(prepared['sources']) == len(checkpoints), 'incomplete reference preparation')
        for item in prepared['sources']:
            require(model_metadata(item['source'], item['observed']) == item['metadata'], 'prepared reference checkpoint changed')
        sources = [(v['source'], v['observed']) for v in prepared['sources']]
    path = a.out / ASSIGNMENT
    os.environ.update(M2_ASSIGNMENT=str(path.resolve()), M2_FAMILY=a.family + '.' + a.panel, M2_FAMILY_SEED=str(a.seed))
    scope = normalized_assignment_inventory() if a.family == 'm3' else nullcontext()
    with scope:
        proof = m2_assignment.active(a.out, a.gpu, a.standin)
    population = knowledge_items() if a.panel == 'knowledge-ll' else m2_format.items(a.out, a.standin)
    # Use precisely the tokenizer employed by the native validation.
    if a.family == 'm3':
        from kit.m3_train import base_record
        tokenizer = base_record(a.out)['model']
    else:
        doc = json.loads((a.out / 'binding.json').read_text())
        if a.family in ('core','reprobe'):
            from kit.m2_paths import local
            tokenizer = str(local(doc.get('base', doc.get('starts', {}).get('base'))['model']))
        else:
            # X retains the original M binding as its tokenizer authority.
            from kit.m2_paths import local
            core = json.loads((Path(doc['message2']) / 'binding.json').read_text())
            tokenizer = str(local(core.get('base', core.get('starts', {}).get('base'))['model']))
    if not a.standin: m2_format.tokenizer_identity(tokenizer)
    for index,(checkpoint, (source, observed)) in enumerate(zip(checkpoints, sources)):
        if getattr(a,'cell_index',None) is not None and a.cell_index!=index:continue
        start = time.monotonic()
        arm = checkpoint['arm']; step = checkpoint['step']; stage = checkpoint.get('stage')
        unit = arm if a.family != 'm3' or arm == 'base' else f'{arm}.{stage}'
        root = Path(os.environ.get('REF_ATTEMPT_ROOT',str(a.out))) / 'reference-scores' / a.family / f'{unit}-s{a.seed}-k{step}-{a.panel}'
        identity = dict(schema='reference-guard.v1', method=METHOD, family=a.family, arm=arm, seed=a.seed, step=step,
                        stage=stage, panel=a.panel, panel_sha256=s.digest(population), model=observed,
                        assignment=proof, gpu_identity=proof, technical_synthetic=a.standin,
                        implementation_sha256=s.file_hash(Path(__file__).with_name('hellaswag_reference.py' if a.panel == 'hellaswag' else 'm3_score.py')))
        if os.environ.get('REF_ATTEMPT_ROOT'):identity['allocation_attempt']=Path(os.environ['REF_ATTEMPT_ROOT']).name
        if (root / 'manifest.json').exists():
            frozen_files(root)
            require(json.loads((root / 'ref-identity.json').read_text()) == identity, 'reference resume identity changed')
            continue
        if a.standin: likelihood = standin_likelihoods(population, root.name)
        elif hasattr(a,'reference_cache'):
            cache=a.reference_cache
            if not cache:
                import torch
                from transformers import AutoModelForCausalLM,AutoTokenizer
                from kit.m2_gpu import assert_current
                assert_current()
                # Preserve the validation's dtype and CPU-first transfer exactly.
                cache['runtime']=(AutoTokenizer.from_pretrained(tokenizer,local_files_only=True),
                    AutoModelForCausalLM.from_pretrained(source['model'],local_files_only=True,
                        torch_dtype=torch.bfloat16).to('cuda').eval())
                cache['model']=source['model']
            require(cache['model']==source['model'],'reference panel checkpoint differs')
            if a.panel=='hellaswag':
                from kit.hellaswag_reference import _score
                likelihood=_score(source['model'],population,tokenizer,runtime=cache['runtime'])
            else:likelihood=knowledge_reference(source['model'],population,tokenizer,runtime=cache['runtime'])
        elif a.panel == 'hellaswag': likelihood = hellaswag_reference(source['model'], population, tokenizer)
        else: likelihood = knowledge_reference(source['model'], population, tokenizer)
        require(len(likelihood) == len({r['id'] for r in likelihood}) == 100 and
                {r['id'] for r in likelihood} == {r['id'] for r in population}, 'reference likelihood coverage incomplete')
        gold = {r['id']: int(r['label']) for r in population}
        panel = 'knowledge' if a.panel == 'knowledge-ll' else 'hellaswag'
        rows = [dict(arm=unit, seed=a.seed, step=step, task=panel, panel=panel, cap=2048, question_id=r['id'],
                     correct=int(selected(r['scores']) == gold[r['id']]), format_free_correct=int(selected(r['scores']) == gold[r['id']]),
                     output_tokens=0, finished=1) for r in likelihood]
        bind(root / 'ref-identity.json', identity)
        write_csv(root / 'scores.csv', [{k: r.get(k, '') for k in COLUMNS} for r in rows], COLUMNS)
        atomic(root / 'likelihoods.json', dict(items=population, rows=likelihood))
        atomic(root / 'cost.json', dict(kind='reference_guard_scoring', method=METHOD, seed=a.seed, panel=a.panel,
                                       gpu_hours=0 if a.standin else (time.monotonic() - start) / 3600, technical_synthetic=a.standin))
        atomic(root / 'manifest.json', files_manifest(root, ['ref-identity.json', 'scores.csv', 'likelihoods.json', 'cost.json']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('assign', 'prepare', 'score'), required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--family', required=True)
    p.add_argument('--from-assignment', type=Path); p.add_argument('--lane-assignment', type=Path,
        help='reviewed local v2 assignments keyed family.panel|seed; panels may use separate GPUs')
    p.add_argument('--checkpoints', type=Path); p.add_argument('--seed', type=int); p.add_argument('--panel', choices=(*PANELS,'both'))
    p.add_argument('--schedule-checkpoints',action='store_true',help='generate all registered M3 decision checkpoints at assignment')
    p.add_argument('--cell-index',type=int,help=argparse.SUPPRESS)
    p.add_argument('--gpu', default='0'); p.add_argument('--standin', action='store_true')
    a = p.parse_args()
    if a.action == 'assign': assign(a)
    elif a.action == 'prepare': prepare(a)
    elif a.standin: run(a)
    else:
        from kit.v6_infra import activate,verify_runtime
        if os.environ.get('KIT_V6A'):
            from kit.v6_infra import activate_v6a
            activate_v6a()
        elif a.family=='m3':
            manifest=Path(os.environ.get('M3_MANIFEST',str(a.out/'m3-launch.json')))
            if not manifest.is_file():
                from kit.p3b_common import NotReady
                raise NotReady('bound M3 launch manifest required for reference scoring')
            os.environ['M2_INFRA_AUTHORITY_SHA256']=verify_runtime(load(manifest)['lane_runtime']) or ''
        else:activate(a.out)
        from kit.m3_manifest import allocated_gpus, allocation_reservation, watched
        if not os.environ.get('UTIL_WATCH_ACTIVE'):
            return subprocess.run(watched([sys.executable, '-I', str(Path(__file__).resolve()), *sys.argv[1:]], a.out / 'util' / (a.family + '-' + str(a.seed) + '-' + a.panel) / (str(time.time_ns())+'.jsonl')), env=dict(os.environ, UTIL_WATCH_ACTIVE='1')).returncode
        if os.environ.get('REF_POOL_CHILD'):run(a)
        elif os.environ.get('M3_GPU_LOCK_FDS'):
            run_lane(a)
        else:
            with allocation_reservation(allocated_gpus(1)) as fds:
                old=os.environ.get('M3_GPU_LOCK_FDS')
                try:
                    os.environ['M3_GPU_LOCK_FDS']=json.dumps(fds)
                    run_lane(a)
                finally:
                    if old is None:os.environ.pop('M3_GPU_LOCK_FDS',None)
                    else:os.environ['M3_GPU_LOCK_FDS']=old
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except __import__('kit.p3b_common', fromlist=['NotReady']).NotReady as error:
        print('PENDING:', error); raise SystemExit(3)
