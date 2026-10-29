"""Receipts shared by the next-message plain jobs."""
from kit.p3b_common import *
import subprocess
import sys

N = KIT / 'resources/next'
SEEDS = tuple(range(101, 106))
RECIPES = ('D', 'F', 'D+replay', 'F+replay')


def frozen_next():
    for line in (N / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        check(KIT / name, digest)


def checkpoint(path, receipt, standin=False, base=False):
    from kit.m2_paths import local
    path=local(path);receipt=local(receipt)
    require(path is not None, 'missing checkpoint path')
    if receipt is not None:require(Path(receipt).is_file(),'missing checkpoint receipt')
    if standin:
        require(receipt is not None, 'stand-in receipt required')
        doc = json.loads(Path(receipt).read_text())
        require(doc.get('technical_synthetic') is True, 'stand-in receipt required')
        check(Path(path) / 'config.json', doc['model_file_hashes']['config.json'])
        return doc
    require(base or receipt is not None, 'checkpoint receipt required')
    require('failed-v3' not in Path(path).parts and not {'resets', 'recoveries'} & set(Path(path).parts), 'excluded checkpoint')
    return identity(path, 'student' if base else None, receipt)


def machine(out, gpu, standin):
    require(',' not in gpu, 'one physical scoring GPU required')
    from kit.m2_assignment import active
    proof=active(out,gpu,standin)
    if proof is not None:
        value=dict(uuid=proof['uuid'],environment=proof['environment'],assignment=proof)
        bind(out/'next-scoring-families'/f"{proof['family']}-s{proof['seed']}.json",value)
        return value
    if standin: value = dict(uuid='CPU-STANDIN', environment='CPU-STANDIN')
    else:
        uuid = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        require(bool(uuid) and '\n' not in uuid, 'cannot identify physical scoring GPU')
        value = dict(uuid=uuid, environment=s.digest(dict(python=sys.executable, versions=t.runtime_versions())))
    bind(out / 'next-scoring-machine.json', value)
    return value


def run(command):
    subprocess.run([str(x) for x in command], check=True)


def freeze(root, names, standin, **extra):
    atomic(root / 'manifest.json', files_manifest(root, names, technical_synthetic=standin, **extra))


def read_inputs(out, standin):
    doc = frozen_files(out / 'next-inputs')
    require(doc['technical_synthetic'] == standin, 'input type mismatch')
    return json.loads((out / 'next-inputs/inputs.json').read_text())


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    t.write_jsonl(path, rows)


def shared_score(a, parent, arm, seed, step, panel, runner):
    """Generate an identical checkpoint/panel once; aliases retain scientific arm labels."""
    observed=checkpoint(parent['path'],parent.get('receipt'),a.standin,
                        base=parent['identity']['model_file_hashes']==read_inputs(a.out,a.standin)['base']['identity']['model_file_hashes'])
    require(observed['model_file_hashes']==parent['identity']['model_file_hashes'],'shared endpoint identity changed')
    from kit.m2_assignment import cache_identity
    key=s.digest(dict(model=observed['model_file_hashes'],panel=panel,assignment=cache_identity(a.out)))[:24]
    canonical='next-shared-'+key
    source=f'scores/{canonical}-s0-k0-{panel}'
    alias=f'{arm}-s{seed}-k{step}-{panel}'
    bind(a.out/'score-aliases'/(alias+'.json'),dict(source=source,arm=arm,seed=seed,step=step,panel_set=panel))
    receipt=a.out/'shared-identities'/(key+'.json')
    bind(receipt,dict(model_file_hashes=observed['model_file_hashes'],technical_synthetic=a.standin))
    command=[sys.executable,KIT/('next_hellaswag.py' if panel=='hellaswag' else 'p3b_score.py'),
             '--out',a.out,'--arm',canonical,'--seed',0,'--step',0,'--model',parent['path'],
             '--checkpoint-receipt',receipt,'--gpu',a.gpu]
    if panel=='hellaswag':command+=['--action','score']
    else:
        command+=['--panel-set',panel]
        if getattr(a,'package_a',None):command+=['--package-a',a.package_a]
        if getattr(a,'finqa',None):command+=['--panel-root',a.finqa]
    if a.standin:command+=['--standin']
    if (a.out/source/'manifest.json').exists():
        frozen_files(a.out/source);print('SHARED',alias);return
    runner(command)
