"""Immutable allocation score sets; CPU readers select one complete job per lane.

Historical jobs remain in their own namespace. A later complete comparison
re-scores its baselines and controls, instead of overwriting or mixing jobs.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

TREES=('scores','m2-score-identities','score-identities','scoring-families','references','scoring-core-match','next-scoring-families')


def complete_attempt(root, nonce, identities, extra_files=()):
    """Seal exactly the expected cells; an interrupted lane has no completion seal."""
    root=Path(root)
    for name in identities:
        path=root/name
        if not path.is_file():raise ValueError('incomplete allocation attempt: '+name)
        from kit.p3b_common import frozen_files
        frozen_files(path.parent)
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for name in identities for p in (root/name).parent.rglob('*') if p.is_file()}
    for name in extra_files:
        files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    from kit.p3b_common import atomic
    atomic(root/'attempt.json',dict(schema='lane-attempt.v1',nonce=nonce,
        completed_ns=time.time_ns(),identities=identities,files=files))


def latest_attempt(parent):
    """Newest complete, unmixed attempt. No partial cells leak between nonces.

    If only malformed hardware attempts exist, return the newest so the native
    reader can label its actual evidence UNVERIFIED-HARDWARE, not missing.
    An unusable older attempt never blocks a newer complete attempt. Invalid
    attempts remain on disk; no scores are merged while trying older receipts.
    """
    candidates=[];errors=[]
    for path in Path(parent).glob('*/attempt.json'):
        try:
            doc=json.loads(path.read_text())
            if not isinstance(doc,dict) or doc.get('schema')!='lane-attempt.v1':raise ValueError('unknown lane attempt')
            if not isinstance(doc.get('completed_ns'),int) or isinstance(doc['completed_ns'],bool):raise ValueError('invalid attempt time')
            if not isinstance(doc.get('files'),dict) or not doc['files']:raise ValueError('empty attempt files')
            if not isinstance(doc.get('identities'),list) or not doc['identities'] or not all(isinstance(n,str) for n in doc['identities']):raise ValueError('invalid attempt identities')
            if not isinstance(doc.get('nonce'),str):raise ValueError('invalid attempt nonce')
            candidates.append((doc['completed_ns'],path.parent,doc))
        except (OSError,ValueError,KeyError,TypeError) as error:errors.append(error)
    invalid_hardware=[]
    for _,root,doc in sorted(candidates,key=lambda v:v[0],reverse=True):
        try:
            for name,digest in doc['files'].items():
                child=root/name
                if Path(name).is_absolute() or '..' in Path(name).parts or not child.is_file() or hashlib.sha256(child.read_bytes()).hexdigest()!=digest:
                    raise ValueError('allocation attempt constituent changed')
            observed=set()
            for name in doc['identities']:
                if name not in doc['files']:raise ValueError('unsealed attempt identity')
                proof=json.loads((root/name).read_text())['assignment']
                observed.add((proof.get('allocation_job'),proof.get('uuid'),proof.get('environment')))
            valid=len(observed)==1 and all(next(iter(observed))) and next(iter(observed))[0]==doc['nonce']==root.name
            if valid:return root
            invalid_hardware.append(root)
        except (OSError,ValueError,KeyError,TypeError) as error:errors.append(error)
    if invalid_hardware:return invalid_hardware[0]
    if errors:raise ValueError('allocation attempt constituent changed') from errors[0]
    return None


def archive(out,key,family,seed):
    out=Path(out)
    if not (out/'scores').exists() and not (out/'references').exists():return
    root=out/'lane-evidence'/key;root.mkdir(parents=True,exist_ok=False)
    for name in TREES:
        if (out/name).exists():shutil.move(str(out/name),str(root/name))
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    (root/'lane.json').write_text(json.dumps(dict(schema='allocation-score-set.v1',family=family,seed=seed,files=files),sort_keys=True)+'\n')


def select(files):
    files=dict(files);lanes={}
    for name,raw in files.items():
        if not name.startswith('lane-evidence/') or not name.endswith('/lane.json'):continue
        doc=json.loads(raw);prefix=name.rsplit('/',1)[0]+'/'
        if doc.get('schema')!='allocation-score-set.v1':raise ValueError('unrecognised allocation score set')
        for child,sha in doc['files'].items():
            if prefix+child not in files or hashlib.sha256(files[prefix+child]).hexdigest()!=sha:
                raise ValueError('allocation score set changed')
        coverage={p for p in doc['files'] if p.endswith(('/m2-identity.json','/assignment-identity.json'))}
        lanes.setdefault((doc['family'],doc['seed']),[]).append((coverage,prefix,doc))
    chosen=[]
    for values in lanes.values():
        maximal=[v for v in values if not any(v[0]<other[0] for other in values)]
        # Incomparable partial frontiers cannot be combined into a contrast.
        if len(maximal)!=1:
            maximal.sort(key=lambda v:(len(v[0]),v[1]),reverse=True)
            if len(maximal)>1 and maximal[0][0]==maximal[1][0]:
                raise ValueError('UNVERIFIED-HARDWARE: competing allocation score sets')
        chosen.append(maximal[0])
    virtual={k:v for k,v in files.items() if not k.startswith('lane-evidence/')}
    for coverage,prefix,doc in chosen:
        for child in doc['files']:
            raw=files[prefix+child]
            if child in virtual and virtual[child]!=raw:
                # Only materialised copies of the selected namespace may coexist.
                raise ValueError('UNVERIFIED-HARDWARE: conflicting allocation score views')
            virtual[child]=raw
    return virtual


def materialize(out):
    out=Path(out)
    if not (out/'lane-evidence').exists():return
    files={str(p.relative_to(out)):p.read_bytes() for p in (out/'lane-evidence').rglob('*') if p.is_file()}
    for name,raw in select(files).items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=raw:raise ValueError('conflicting allocation score view')
        path.write_bytes(raw)


def clear_view(out):
    """Only remove CPU materialised copies; immutable namespaced jobs are retained."""
    out=Path(out)
    for name in TREES:
        path=out/name
        if path.is_dir():shutil.rmtree(path)
        elif path.exists():path.unlink()


def hardware_status(rows):
    """A numeric contrast may use only one allocation's generated scores per seed."""
    observed={}
    for row in rows:
        if row.get('_guard_method')=='REFERENCE-SCORED':continue
        job=row.get('_allocation_job')
        observed.setdefault(int(row['seed']),set()).add((job,row.get('_allocation_uuid'),row.get('_allocation_environment')))
    if any(any(v[0] for v in values) and (len(values)!=1 or any(not all(v) for v in values)) for values in observed.values()):
        return 'UNVERIFIED-HARDWARE'
    return None
