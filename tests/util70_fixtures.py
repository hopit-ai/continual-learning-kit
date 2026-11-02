"""CPU rehearsals: immutable build metadata and explicit synthetic references."""
import copy
import hashlib
from functools import wraps
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from types import ModuleType,SimpleNamespace as NS
from unittest.mock import patch


def compact_replicas(root, nodes, seen):
    """Share identical immutable cache bytes without dropping prepared paths.

    Large static graph fixtures retain hundreds of verified parent views. Only
    .replicas files are shared; writable allocation workspaces and durable
    archives keep their own files and all scientific evidence stays intact.
    """
    inodes=seen.setdefault('inodes',set())
    for slot in range(nodes):
        for path in (Path(root)/f'node{slot}'/'.replicas').rglob('*'):
            if not path.is_file() or path.is_symlink():continue
            stat=path.stat()
            identity=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if stat.st_size<1024 or identity in inodes:continue
            inodes.add(identity)
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').digest()
            key=(stat.st_size,digest);original=seen.setdefault(key,path)
            if original==path or original.stat().st_ino==stat.st_ino:continue
            temporary=Path(root)/'.replica-links'/uuid.uuid4().hex
            temporary.parent.mkdir(exist_ok=True)
            try:
                os.link(original,temporary);temporary.replace(path)
            finally:temporary.unlink(missing_ok=True)


def immutable_protocols(function):
    """Hash this immutable test build once; still assert its exact bytes at exit.

    These long fixtures mutate only their own data/checkpoints. Pin-drift tests
    run separately with live checkers; no source/authority mutation is permitted
    during a rehearsal. Returning copies prevents fixture mutations of metadata.
    """
    @wraps(function)
    def run(*args,**kwargs):
        from kit import c_common,m2_common,m2_format,v6_infra
        originals=(c_common.protocol,m2_common.protocol,m2_format.policy)
        frozen=tuple(f() for f in originals)
        replacements=[(lambda value=value:copy.deepcopy(value)) for value in frozen]
        seal=v6_infra.inherited_seal()
        def policy():
            # Keep launch trust live; immutable code pins are rechecked at exit.
            if v6_infra.inherited_seal()!=seal or v6_infra.authority_sha256()!=seal:return originals[2]()
            return copy.deepcopy(frozen[2])
        replacements[2]=policy
        with kwargs['monkeypatch'].context() as scope:
            for module in list(sys.modules.values()):
                if not isinstance(module,ModuleType) or not module.__name__.startswith('kit.'):continue
                for name,value in list(vars(module).items()):
                    for old,new in zip(originals,replacements):
                        if value is old:scope.setattr(module,name,new)
            result=function(*args,**kwargs)
        assert tuple(f() for f in originals)==frozen,'rehearsal source bytes changed during execution'
        return result
    return run


def reference_guards(root,family,seed):
    """Actual reference writer/reader schemas, synthetic likelihoods, zero GPU.

    Native weights/receipts are checked normally for a labelled CPU fixture.
    Correct reference choices are explicit test data; generated graders remain
    unchanged. These files cannot satisfy a non-synthetic reader.
    """
    from kit import m2_assignment as assignment,m3_score,ref_score_job as ref,ref_guard
    from kit.p3b_common import atomic
    root=Path(root);out=root/'out'
    context=json.loads((root/'staged.json').read_text())
    if family not in ('core','reprobe'):return
    # Training rows may inherit scores but run on a different node. References
    # belong only to the completed score-family lane's measured placement.
    if context['task']['kind']!='score-family':return
    assert json.loads((out/'binding.json').read_text())['technical_synthetic'] is True
    models={}
    for path in (out/'scores').glob('*/m2-identity.json'):
        meta=json.loads(path.read_text())
        if meta['seed']==seed and meta['step'] in (0,40):
            assert meta['technical_synthetic'] is True
            models[meta['arm'],meta['step']]=meta['model']
    if not models:return
    authority=assignment.load(out/'scoring-assignment.json')
    assignment.freeze(out/ref_guard.ASSIGNMENT,dict(schema=authority['schema'],assignments={
        f'{key.rsplit("|",1)[0]}.{panel}|{key.rsplit("|",1)[1]}':value
        for key,value in authority['assignments'].items() for panel in ref_guard.PANELS}))
    checkpoints=[dict(arm=arm,seed=seed,step=step) for arm,step in sorted(models)]
    listing=root/'synthetic-reference-list.json';atomic(listing,checkpoints)
    original=ref.source_for
    def source(out,family,checkpoint,standin):
        source,observed=original(out,family,checkpoint,standin)
        # Use the identical canonical identity already checked by native scoring.
        return source,models[checkpoint['arm'],checkpoint['step']]
    def correct_choices(rows,salt):
        return [dict(id=r['id'],scores=[0. if i==int(r['label']) else -1. for i in range(4)],selected=int(r['label'])) for r in rows]
    with patch.dict(os.environ,dict(context['env'],M2_EXECUTION_PHASE='gpu'),clear=True),patch.object(ref,'source_for',source),patch.object(m3_score,'standin_likelihoods',correct_choices):
        lanes=[NS(out=out,family=family,seed=seed,panel=panel,checkpoints=listing,gpu='0',standin=True) for panel in ref_guard.PANELS]
        # CPU admission archives the previous frontier. Prepare both panels
        # before either scores, so the second preparation cannot archive the first.
        for a in lanes:ref.prepare(a)
        for a in lanes:ref.run(a)
    # Readiness/history are scratch owned by this CPU fixture, not scientific cells.
    for name in ('reference-ready','reference-history'):shutil.rmtree(out/name,ignore_errors=True)


# Fix-unit tests may amend implementation bytes, but must never regenerate the
# shipping authority. This prospective document exists only under pytest tmp/.
from functools import lru_cache
import pytest


@lru_cache(maxsize=1)
def prospective_core_authority_bytes():
    import hashlib
    from kit import v6_infra
    doc=json.loads((v6_infra.ROOT/v6_infra.AUTHORITY).read_text())
    for row in doc['files']:row['new_sha256']=hashlib.sha256((v6_infra.ROOT/row['path']).read_bytes()).hexdigest()
    if (v6_infra.ROOT/'.git').exists():
        from scripts.v6_infra_authority import current_pins
        pairs={(r['path'],r['pinned_sha256']) for r in doc['files']}
        for name,pinned in current_pins():
            path=v6_infra.ROOT/name
            if path.is_file() and (name,pinned) not in pairs:
                actual=hashlib.sha256(path.read_bytes()).hexdigest()
                if actual!=pinned:doc['files'].append(dict(path=name,pinned_sha256=pinned,new_sha256=actual,reason='Explicitly sealed prospective CPU test build'))
    return (json.dumps(doc,sort_keys=True,indent=2)+'\n').encode()


@pytest.fixture
def prospective_core_authority(tmp_path,monkeypatch):
    from kit import v6_infra
    path=tmp_path/'temporary-core-infrastructure-update.json'
    path.write_bytes(prospective_core_authority_bytes())
    monkeypatch.setattr(v6_infra,'AUTHORITY',str(path))
    seal=v6_infra.identity()
    monkeypatch.setenv('M2_INFRA_AUTHORITY_SHA256',seal)
    monkeypatch.setenv('KIT_V6_AUTHORITY_SHA',seal)
    return path


@pytest.fixture(scope='module',autouse=True)
def prospective_build_authority(tmp_path_factory):
    """Externally seal disposable current bytes for a whole CPU fixture module."""
    from kit import v6_infra
    path=tmp_path_factory.mktemp('prospective-build')/'authority.json'
    path.write_bytes(prospective_core_authority_bytes())
    with pytest.MonkeyPatch.context() as scope:
        scope.setattr(v6_infra,'AUTHORITY',str(path))
        seal=v6_infra.authority_sha256()
        scope.setenv(v6_infra.ENV,seal)
        scope.setenv('M2_INFRA_AUTHORITY_SHA256',seal)
        yield path


def launch_subset(doc, **fields):
    """A reduced CPU launch retains its parent's explicit trust/resources."""
    result={key:doc[key] for key in ('schema','nodes','infrastructure_update','lane_runtime','resource_policy','technical_synthetic') if key in doc}
    result.update(fields)
    for task in result.get('tasks',[]):
        if 'resources' not in task:
            assert task['kind'] in ('prepare','collect','reduce-admission')
            task['resources']=dict(nodes=1,gpus=0,cpus=8,share_node=True)
    return result
