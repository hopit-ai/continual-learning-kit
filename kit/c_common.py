"""Package C identities and frozen legal protocol; no scheduler or network IO."""
from kit.p3b_common import *
import importlib.util
from contextlib import contextmanager
import fcntl

C = KIT / 'resources/c'
SEEDS = tuple(range(101, 106))
RECIPES = ('D', 'F', 'D+replay', 'F+replay')
TIE_ORDER = ('F', 'D', 'D+replay', 'F+replay')
STEPS = (0, 10, 20, 30, 40)
RETENTION_STEPS = (0, 20, 40)
START_POLICY = 'C core and probe: exact v4 D/F FinQA finals; replay during ContractNLI only'
_spec = importlib.util.spec_from_file_location('c_frozen_legal', C / 'legal_headroom.py')
legal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legal)


def protocol():
    doc = json.loads((C / 'freeze.json').read_text())
    for name, sha in doc['files'].items(): check(C / name, sha)
    implementations = [*sorted(KIT.glob('c_*.py'))] + [KIT / name for name in ('p3b_common.py','p3b_train.py','p3b_loss.py','p3b_score.py','p3b_read.py','p3b_replay_data.py','v4_teacher.py','v4_sft.py','sdft_verl.py','sdft_objective.py','sdft_actor.py','sft_entry.py','score_forgetting.py','eval_bed.py','next_common.py','next_slots.py','next_pools.py','next_hellaswag.py','next_read.py')]
    return dict(freeze=s.file_hash(C / 'freeze.json'), files=doc['files'], implementation={p.name:s.file_hash(p) for p in implementations}, frontier_test=82, threshold=69.7)


def panels():
    protocol()
    return {split: legal.load_panel(C, 'contract_nli', split)[0] for split in ('panel', 'test')}


@contextmanager
def lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise CorrectnessStop('another writer holds ' + str(out))
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def binding(out, standin):
    path = out / 'binding.json'
    if not path.exists(): raise NotReady('W is pending; run c_bind.py at send with completed unreset B evidence')
    doc = json.loads(path.read_text())
    require(doc['technical_synthetic'] == standin, 'synthetic binding mismatch')
    require(doc['protocol'] == protocol(), 'C protocol changed after binding')
    require(doc['start_policy'] == START_POLICY, 'sequence start policy changed')
    return doc


def arms(doc):
    return list(dict.fromkeys([doc['W'], 'F', 'answer-only']))


def probe_recipe(doc):
    return doc['W'][0] + '+replay'


def probe_arm(doc, arm):
    require(arm in ('generic','broad','earlier','both'), 'unregistered probe map')
    return 'sequence-' + (doc['W'] if arm == 'generic' and doc['W'] == probe_recipe(doc) else 'probe-' + arm)


def start(doc, recipe, route, seed):
    require(recipe in arms(doc) or (route == 'sequence' and recipe in ('probe-generic','probe-broad','probe-earlier','probe-both')), 'arm not in bound C')
    family = doc['W'][0] if recipe.startswith('probe-') else 'F' if recipe == 'answer-only' else recipe[0]
    key = 'base' if route == 'standalone' else f'{family}:{seed}'
    return doc['starts'][key]


def checkpoint(doc, out, arm, seed, step):
    if arm == 'base':
        require(step == 0, 'base must be step 0')
        return doc['starts']['base']
    route, recipe = arm.split('-', 1)
    require(route in ('standalone', 'sequence') and (recipe in arms(doc) or recipe in ('probe-generic','probe-broad','probe-earlier','probe-both')), 'unregistered arm')
    if step == 0: return start(doc, recipe, route, seed)
    root = out / 'runs' / f'{arm}-s{seed}'
    if not (root / 'complete.json').exists(): raise NotReady('stage not complete: ' + str(root))
    frozen_files(root)
    return dict(model=str(root / f'hf-step{step}'), receipt=str(root / f'export-{step}.json'))


def checked_model(record, standin, base=False):
    if standin:
        logical=record['model']
        if record.get('receipt'):
            from kit.m2_paths import local
            path=local(record['receipt'])
            if path.exists():logical=json.loads(path.read_text()).get('model',logical)
        return dict(technical_synthetic=True, model=logical)
    if record.get('receipt_sha256'): check(record['receipt'], record['receipt_sha256'])
    observed = identity(record['model'], 'student' if base else None, record.get('receipt'))
    if record.get('identity'): require(__import__('kit.m2_paths',fromlist=['equivalent']).equivalent(observed,record['identity']), 'bound checkpoint identity changed')
    return observed


def overlap(rows, evaluation):
    """B's frozen near-prompt checks against evaluations only; no new train quotas."""
    from kit.p3b_replay_data import prepared, near
    reference = [(r, prepared(r['prompt'])) for r in evaluation]
    kept, excluded = [], []
    for row in rows:
        value = prepared(row['prompt'])
        hit = next((r for r, p in reference if row['id'] == r['id'] or near(value, p)), None)
        if hit: excluded.append(dict(id=row['id'], reason='evaluation_prompt_overlap', overlaps=hit['id']))
        else: kept.append(row)
    return kept, excluded
