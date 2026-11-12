"""TRACE's offline protocol, immutable inputs and plain-script inventory."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
import os

from kit.nm_common import BASE, SEEDS, atomic, canonical, digest, load, need, sha, stats

KIT = Path(__file__).resolve().parent
TASKS = ('C-STANCE', 'FOMC', 'MeetingBank', 'Py150', 'ScienceQA',
         'NumGLUE-cm', 'NumGLUE-ds', '20Minuten')
SUBSET = ('FOMC', 'MeetingBank', 'NumGLUE-cm')
SOURCES = ('gold', '27b')
EPOCHS = (1, 1, 5, 5, 1, 5, 5, 5)
JUDGE = dict(model='Qwen/Qwen2.5-7B-Instruct',
             revision='a09a35458c702b33eeacc393d103063234e8bc28')
REG = KIT / 'resources/trace/registration.md'
DEPARTURES = ('M', 'T', 'P', 'L', 'L-EOS', 'S', 'D-obj', 'D-demo',
              'F', 'E', 'J', 'B', 'Q', 'H', 'R')
GRADIENT_CLIPPING = 1.0
L_EOS_WARN_EXCESS = 0.01


def offline():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled')


def bound_file(ref):
    path = Path(ref['path'])
    if not path.is_file():
        raise FileNotFoundError('bound input is not staged')
    need(sha(path) == ref['sha256'], 'input identity differs')
    return path


def bound_json(ref):
    return load(bound_file(ref))


def read_rows(ref):
    path = bound_file(ref)
    import json
    with path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    need(len({r['id'] for r in rows}) == len(rows), 'duplicate row IDs')
    return rows


def model_record(ref):
    root = Path(ref['model'])
    need(root.is_dir() and ref.get('files'), 'staged model files required')
    actual = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}
    need(actual == ref['files'], 'model/tokenizer file inventory changed')
    return ref


def seal(path, value):
    path = Path(path)
    if path.exists():
        need(load(path) == value, 'sealed TRACE artifact changed')
    else:
        atomic(path, value)
    return value


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def schedule(rows, seed, task):
    need(seed in SEEDS and len(rows) == 500, 'subset dose/seed differs')
    ranked = sorted(rows, key=lambda r: (digest_string(
        f'trace-order-20261111|{seed}|{task}|{r["id"]}'), r['id']))
    return ranked * 2 + ranked[:280]


def digest_string(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def stages(source, seed):
    """Nine distinct stages per source/seed, preserving each common parent."""
    prefix = f'{source}-s{seed}'
    rows = [dict(id=prefix+'-parent', task=SUBSET[0], freeze=False,
                 incoming='base', arm='parent')]
    for arm in ('D', 'F'):
        previous = rows[0]['id']
        for index, task in enumerate(SUBSET[1:], 2):
            name = f'{prefix}-{arm}{index}'
            rows.append(dict(id=name, task=task, freeze=arm == 'F',
                             incoming=previous, arm=arm, stage=index))
            previous = name
        for index, task in enumerate(SUBSET[1:], 2):
            rows.append(dict(id=f'{prefix}-U{arm}{index}', task=task,
                             freeze=arm == 'F', incoming='base', arm='U'+arm,
                             stage=index))
    return rows


def states(study, seed):
    if study == 's8':
        return [dict(id='base', arm='S', stage=0, checkpoint='base')] + [
            dict(id=f'S{i}', arm='S', stage=i, checkpoint=f's8-s{seed}-{i}')
            for i in range(1, 9)]
    need(study in SOURCES, 'unknown TRACE study')
    prefix = f'{study}-s{seed}'
    rows = []
    for arm in ('D', 'F'):
        rows.extend([dict(id=arm+'0', arm=arm, stage=0, checkpoint='base'),
                     dict(id=arm+'1', arm=arm, stage=1, checkpoint=prefix+'-parent')])
        for i in (2, 3):
            rows.append(dict(id=arm+str(i), arm=arm, stage=i,
                             checkpoint=f'{prefix}-{arm}{i}'))
            rows.append(dict(id='U'+arm+str(i), arm='U'+arm, stage=i,
                             checkpoint=f'{prefix}-U{arm}{i}'))
    return rows


def inventory(studies=None):
    """Rows are operator-owned allocations. No scheduler submission occurs here."""
    rows = [dict(id='prepare', phase='prepare', gpus=0, depends=[]),
            dict(id='targets', phase='targets', gpus=2, depends=['prepare']),
            dict(id='bind', phase='bind', gpus=0, depends=['targets'])]
    for study in ('s8', *SOURCES):
        qualification = 'qualify-'+study
        rows.append(dict(id=qualification, phase='qualify', study=study,
                         gpus=8, depends=['bind']))
        for seed in SEEDS:
            train = f'train-{study}-{seed}'
            rows.append(dict(id=train, phase='train', study=study, seed=seed,
                             gpus=8, depends=[qualification]))
            rows.append(dict(id=f'score-{study}-{seed}', phase='score', study=study,
                             seed=seed, gpus=1, depends=[train]))
    if studies is not None:
        need(set(studies) <= {'s8', *SOURCES} and studies, 'unknown/empty study scope')
        rows = [r for r in rows if 'study' not in r or r['study'] in studies]
        if not set(studies) & set(SOURCES):
            rows = [r for r in rows if r['id'] != 'targets']
            next(r for r in rows if r['id'] == 'bind')['depends'] = ['prepare']
    rows.append(dict(id='collect', phase='collect', gpus=0,
                     depends=[r['id'] for r in rows if r['phase'] == 'score']))
    planned = studies or ('s8', *SOURCES)
    cells = [state for study in planned for seed in SEEDS for state in states(study,seed)]
    return dict(schema='trace-rows.v1', comparisons=len(planned), reporting='DESCRIPTIVE',
                trained_cells=sum(state['stage']>0 for state in cells),
                distinct_checkpoints=len({state['checkpoint'] for state in cells if state['stage']}),
                evaluation_cells=len(cells),
                estimated_ceiling_gpu_h=3500, rows=rows)


def estimates():
    return dict(s8=[1350, 1500], gold=[400, 500], **{'27b': [400, 500]},
                preparation=[40, 120], judge=[75, 300], total=[2265, 2920],
                estimated_ceiling_gpu_h=3500, judge_calls=109884)


from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import tempfile
from kit.nm_common import need, sha


def scratch():
    return os.environ.get('V7H_LOCAL_ROOT') or os.environ.get('SLURM_TMPDIR')


def local_file(path):
    path=Path(path)
    if not scratch():
        return path
    target=Path(scratch())/'v7h-inputs'/sha(path)/path.name
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as stream:
            temporary=Path(stream.name)
        shutil.copyfile(path,temporary)
        need(sha(temporary)==sha(path),'node-local input changed')
        os.replace(temporary,target)
    need(sha(target)==sha(path),'node-local replica changed')
    return target


def publish_tree(source,destination):
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    # Complete markers follow their data. Row completion is published by caller.
    paths=sorted((p for p in source.rglob('*') if p.is_file()),
                 key=lambda p: (p.name in ('complete.json','lane.json','exports.json'),str(p)))
    for path in paths:
        target=destination/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as stream:
            temporary=Path(stream.name)
        shutil.copyfile(path,temporary)
        need(sha(path)==sha(temporary),'durable publication changed')
        os.replace(temporary,target)


@contextmanager
def output_tree(destination):
    if not scratch():
        yield Path(destination)
        return
    with tempfile.TemporaryDirectory(prefix='v7h-row-',dir=scratch()) as work:
        try:
            yield Path(work)
        finally:
            # Include interrupted work; only the outer complete receipt commits it.
            publish_tree(work,destination)


def stage_command(command,files):
    mapping={str(p):str(local_file(p)) for p in files}
    return [replace_paths(arg,mapping) for arg in command]


def replace_paths(value,mapping):
    for original,replica in sorted(mapping.items(),key=lambda p:-len(p[0])):
        value=value.replace(original,replica)
    return value
