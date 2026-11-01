"""Public send expectations: content identities only; locations stay on the node.

Reference ordinals follow the frozen decisions' call order, never path sorting.
All non-location fields are compared, including unknown future content fields.
Mismatch diagnostics deliberately contain neither values nor locations.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from kit.p3b_common import CorrectnessStop

KIT = Path(__file__).resolve().parent
SCHEMA = 'm2-public-send-expectation.v1'
EXPECTED = KIT/'resources/m2/send-binding-expected.json'
RECEIPTS = {
    'hellaswag_attempt2': 'hellaswag-format-check.attempt2.json',
    'p2_attempt5': 'p2-smoke.attempt5.json',
    'pd_attempt3': 'pd-smoke.attempt3.json',
}


def receipt_hashes():
    result = {key: hashlib.sha256((KIT/'resources/m2/receipts'/name).read_bytes()).hexdigest()
              for key, name in RECEIPTS.items()}
    pd = json.loads((KIT/'resources/m2/receipts'/RECEIPTS['pd_attempt3']).read_text())
    result['pd_sealed_identity_sha256'] = pd['identity_sha256']
    return result


def _public_document(binding, cdoc, ruling, *, receipts=None):
    """Project a verified binding, preserving all fields except rescore locations.

    C identities are recomputed from the original C binding/verified export list.
    Callers must run m2_return1.binding and decisions before invoking this helper.
    """
    result = copy.deepcopy(binding)
    refs = result.pop('rescore_refs')
    ordered = []
    seen = set()
    for package in ruling['rescoring']['packages']:
        for call in package['calls']:
            key = call['checkpoint']
            if key in seen: continue
            seen.add(key)
            if key in refs:
                content = {k: v for k, v in refs[key].items() if k not in ('model', 'receipt')}
                ordered.append(dict(reference=len(seen), content=content))
    if not set(refs) <= seen:
        raise CorrectnessStop('unregistered rescore reference')
    result['rescore_refs'] = ordered
    identities = {}
    for seed in (101, 102, 103):
        parent = cdoc['starts'][f'D:{seed}']
        exports = [e for e in ruling['continuation']['exports']
                   if e['arm'] == 'sequence-D' and e['seed'] == seed and e['step'] == 40]
        if len(exports) != 1: raise CorrectnessStop('C final identity coverage differs')
        identities[str(seed)] = dict(
            finqa=dict(model_hashes=parent['identity']['model_file_hashes'], receipt_sha256=parent['receipt_sha256']),
            contract_nli=dict(model_hashes=exports[0]['model_hashes'], receipt_sha256=exports[0]['receipt_sha256']))
    result['c_identities'] = identities
    result['handoff_receipts'] = receipt_hashes() if receipts is None else receipts
    amendment=KIT/'resources/m2/likelihood-near-tie.json'
    result['validation_amendment_sha256']=hashlib.sha256(amendment.read_bytes()).hexdigest()
    result['schema'] = SCHEMA
    # Unknown location fields must fail closed, rather than be silently dropped.
    def no_locations(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in {'host','hostname','username','user','job_id','jobid','slurm_job_id','gpu_uuid','uuid','settings','tier_hardware'}:
                    raise CorrectnessStop('unexpected private metadata in public expectation')
                no_locations(key); no_locations(child)
        elif isinstance(value, list):
            for child in value: no_locations(child)
        elif isinstance(value, str) and (value.startswith(('/', '~', 'file:')) or '\\' in value):
            raise CorrectnessStop('unexpected location in public expectation')
    no_locations(result)
    from kit.m2_privacy import scan
    with tempfile.TemporaryDirectory(prefix='public-expect-check-') as directory:
        root=Path(directory);path=root/'expectation.json'
        path.write_text(json.dumps(result))
        if scan(root,[path]):raise CorrectnessStop('private metadata in public expectation')
    return result


def public_document(binding, cdoc, ruling, *, receipts=None):
    try:return _public_document(binding,cdoc,ruling,receipts=receipts)
    except CorrectnessStop:raise
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        raise CorrectnessStop('public expectation content invalid or resources unavailable') from None


def check_public(path, binding, cdoc, ruling):
    try:expected=json.loads(Path(path).read_text())
    except (OSError,ValueError):
        raise CorrectnessStop('public send expectation unreadable or invalid') from None
    if Path(path).resolve()==EXPECTED.resolve():
        try:
            frozen=json.loads((KIT/'resources/m2/freeze.json').read_text())['files'][EXPECTED.name]
            if hashlib.sha256(EXPECTED.read_bytes()).hexdigest()!=frozen:
                raise CorrectnessStop('shipped public expectation pin differs')
        except (OSError,ValueError,KeyError):
            raise CorrectnessStop('shipped public expectation pin missing or invalid') from None
    observed = public_document(binding, cdoc, ruling)
    if expected != observed:
        raise CorrectnessStop('public send expectation mismatch (content fields; locations excluded)')
    return observed
