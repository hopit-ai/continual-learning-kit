"""Exact seals and committed prospective transitions; no GPU/network calls."""
import hashlib,json,subprocess
from pathlib import Path
import pytest
from kit import v6_infra as infra
from scripts.v6_infra_authority import generate,current_pins,uncovered_pins


def sha(raw):return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def authority(tmp_path,monkeypatch):
    monkeypatch.delenv(infra.ENV,raising=False);monkeypatch.delenv('M2_INFRA_AUTHORITY_SHA256',raising=False)
    monkeypatch.setattr(infra,'ROOT',tmp_path)
    path=tmp_path/'kit/test.py';path.parent.mkdir();path.write_bytes(b'new implementation')
    row=dict(path='kit/test.py',pinned_sha256=sha(b'old implementation'),new_sha256=sha(path.read_bytes()),reason='CPU publication only')
    doc=dict(schema=infra.SCHEMA,statement='scientific definitions unchanged',files=[row])
    seal=tmp_path/infra.AUTHORITY;seal.parent.mkdir(parents=True);seal.write_text(json.dumps(doc))
    return path,seal,doc


def test_exact_pair_requires_external_seal_and_actual_bytes(authority,monkeypatch):
    path,seal,doc=authority;row=doc['files'][0];expected=sha(seal.read_bytes())
    assert not infra.allowed(path,row['pinned_sha256'],row['new_sha256'])
    assert infra.allowed(path,row['pinned_sha256'],row['new_sha256'],expected)
    monkeypatch.setenv(infra.ENV,expected)
    assert infra.allowed(path,row['pinned_sha256'],row['new_sha256'])
    assert not infra.allowed(path,row['pinned_sha256'],row['new_sha256'],None)
    assert not infra.allowed(path,'f'*64,row['new_sha256'],expected)
    path.write_bytes(b'unlisted implementation')
    assert not infra.allowed(path,row['pinned_sha256'],sha(path.read_bytes()),expected)
    assert not infra.allowed(path,row['pinned_sha256'],row['new_sha256'],expected)


def test_activation_never_approves_its_own_digest(authority,monkeypatch):
    with pytest.raises(ValueError,match='seal absent'):infra.activate()
    monkeypatch.setenv(infra.ENV,sha(authority[1].read_bytes()))
    assert infra.activate()==sha(authority[1].read_bytes())


def test_tampered_authority_refused_even_with_replaced_environment(authority,monkeypatch,tmp_path):
    path,seal,doc=authority;expected=sha(seal.read_bytes());row=doc['files'][0]
    out=tmp_path/'send';out.mkdir();(out/'binding.json').write_text(json.dumps(dict(protocol=dict(infrastructure_update=expected))))
    assert infra.activate(out)==expected
    doc['files'][0]['reason']='tampered';seal.write_text(json.dumps(doc))
    assert not infra.allowed(path,row['pinned_sha256'],row['new_sha256'],expected)
    monkeypatch.setenv(infra.ENV,sha(seal.read_bytes()))
    with pytest.raises(ValueError,match='changed after binding'):infra.activate(out)
    assert infra.ENV not in __import__('os').environ


@pytest.mark.parametrize('binding',[{}, {'registration_sha256':'historical'}, {'infrastructure_update':None},
    {'protocol':{}}, {'protocol':None}, {'protocol':[]}, {'protocol':'invalid'}, [],
    {'infrastructure_update':'a'*64,'scoring_protocol':{'infrastructure_update':'b'*64}}])
def test_old_or_malformed_binding_cannot_inherit_a_new_seal(authority,monkeypatch,tmp_path,binding):
    monkeypatch.setenv(infra.ENV,sha(authority[1].read_bytes()))
    monkeypatch.setenv('M2_INFRA_AUTHORITY_SHA256',sha(authority[1].read_bytes()))
    out=tmp_path/'old-send';out.mkdir();(out/'binding.json').write_text(json.dumps(binding))
    with pytest.raises(ValueError):infra.activate(out)
    assert infra.inherited_seal() is None


@pytest.mark.parametrize('mutation',[lambda d:d.update(statement='science changed'),lambda d:d['files'][0].update(path='kit/resources/s/registration.json'),lambda d:d['files'][0].update(path='kit/../outside.py'),lambda d:d['files'].append(d['files'][0].copy()),lambda d:d.update(files=None),lambda d:d['files'][0].update(classification='anything')])
def test_schema_forbids_data_paths_ambiguity_and_changed_science(authority,mutation):
    _,_,doc=authority;mutation(doc)
    with pytest.raises(ValueError):infra.validate(doc)


def test_shell_wrapper_exact_pair_uses_same_external_seal(authority):
    path,seal,doc=authority;wrapper=path.with_suffix('.sh');path.rename(wrapper)
    row=doc['files'][0];row['path']='kit/test.sh';seal.write_text(json.dumps(doc))
    assert infra.allowed(wrapper,row['pinned_sha256'],row['new_sha256'],sha(seal.read_bytes()))
    schema=json.loads((Path(__file__).resolve().parents[1]/'kit/resources/v6/infrastructure-update.schema.json').read_text())
    assert __import__('re').fullmatch(schema['properties']['files']['items']['properties']['path']['pattern'],'kit/test.sh')


def git(root,*args):return subprocess.check_output(['git',*args],cwd=root).decode().strip()


def commit(root):
    git(root,'add','.');git(root,'-c','user.name=CPU Fixture','-c','user.email=fixture@localhost','commit','-qm','CPU fixture')
    return git(root,'rev-parse','HEAD')


@pytest.fixture
def repo(tmp_path):
    git(tmp_path,'init','-q')
    (tmp_path/'kit/resources').mkdir(parents=True)
    for name in ('test.py','job.sh'):(tmp_path/'kit'/name).write_bytes(b'old implementation')
    (tmp_path/'kit/resources/pins.json').write_text(json.dumps(dict(code={
        'kit/test.py':sha(b'old implementation'),'kit/job.sh':sha(b'old implementation')})))
    base=commit(tmp_path)
    # A superseded generation is deliberately not present in the live pin map.
    (tmp_path/'kit/test.py').write_bytes(b'intermediate implementation');commit(tmp_path)
    (tmp_path/'kit/test.py').write_bytes(b'new implementation');(tmp_path/'kit/job.sh').write_bytes(b'new wrapper')
    target=commit(tmp_path)
    return tmp_path,base,target


def test_generator_uses_current_pins_not_every_historical_generation(repo):
    root,base,target=repo;before=(root/'kit/resources/pins.json').read_bytes()
    doc=generate(base,root,target);assert doc==generate(base,root,target)
    assert doc['base_commit']==base and doc['target_commit']==target
    assert len(doc['files'])==2 and {r['pinned_sha256'] for r in doc['files']}=={sha(b'old implementation')}
    assert all(r['classification']=='scheduling' and r['pin_sources'] for r in doc['files'])
    assert before==(root/'kit/resources/pins.json').read_bytes()
    assert uncovered_pins(doc,root,target)==[]
    for row in doc['files']:
        bad=json.loads(json.dumps(doc));bad['files'].remove(row)
        assert uncovered_pins(bad,root,target)==[(row['path'],row['pinned_sha256'],row['new_sha256'])]


@pytest.mark.parametrize('staged',[False,True])
def test_generator_refuses_uncommitted_tracked_bytes(repo,staged):
    root,base,target=repo;(root/'kit/test.py').write_bytes(b'unreviewed')
    if staged:git(root,'add','kit/test.py')
    with pytest.raises(ValueError,match='clean committed'):generate(base,root,target)


def test_generator_records_named_target_instead_of_reading_worktree(repo):
    root,base,target=repo
    (root/'diagnostic.txt').write_text('untracked scratch is not an implementation pin')
    doc=generate(base,root,base)
    assert doc['target_commit']==base and doc['files']==[]


def test_authority_cannot_supply_its_own_historical_pin(repo):
    root,base,target=repo;p=root/infra.AUTHORITY;p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(files=[dict(path='kit/test.py',sha256='f'*64)])));target=commit(root)
    assert ('kit/test.py','f'*64) not in current_pins(root,target)


def test_repository_authority_covers_every_current_pin():
    """Release gate: manager regenerates LAST, after all fix branches commit."""
    root=Path(__file__).resolve().parents[1]
    if not (root/'.git').exists():pytest.skip('source-tree release gate; standalone export has no Git pin inventory')
    doc=json.loads((root/infra.AUTHORITY).read_text())
    assert not uncovered_pins(doc,root,working_tree=True), 'Regenerate the authority LAST from the clean committed integration tree'


def test_generator_refuses_untracked_implementation(repo):
    root,base,target=repo;(root/'kit/uncommitted.py').write_text('unreviewed code')
    with pytest.raises(ValueError,match='clean committed'):generate(base,root,target)


def test_runtime_checker_preserves_explicit_none_instead_of_ambient_fallback(authority,monkeypatch):
    from kit import m2_runtime
    path,seal,doc=authority;row=doc['files'][0]
    monkeypatch.setattr(m2_runtime,'__file__',str(path.parent/'m2_runtime.py'))
    updates=path.parent/'resources/m2/runtime-updates.json';updates.parent.mkdir(parents=True);updates.write_text('{"files":{}}')
    monkeypatch.setenv(infra.ENV,sha(seal.read_bytes()))
    assert m2_runtime.allowed_update(path,row['pinned_sha256'],row['new_sha256'])
    assert not m2_runtime.allowed_update(path,row['pinned_sha256'],row['new_sha256'],None)


def test_new_s_binding_cannot_self_seal_from_an_unbound_historical_parent(authority,monkeypatch,tmp_path):
    from kit import s_bind
    from types import SimpleNamespace as NS
    parent=tmp_path/'historical';parent.mkdir();(parent/'binding.json').write_text('{}')
    monkeypatch.setenv(infra.ENV,sha(authority[1].read_bytes()))
    with pytest.raises(ValueError,match='seal absent'):
        s_bind.job(NS(out=tmp_path/'screen',message2=parent))


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1]/'scripts/publish_kit.py').exists(),
                    reason='the publisher itself is not part of the public export')
def test_export_cpu_gate_supplies_an_explicit_build_seal(authority,monkeypatch,tmp_path):
    from scripts import publish_kit
    from kit import m2_privacy
    from types import SimpleNamespace as NS
    target=tmp_path/'export';path=target/'kit/resources/v6/infrastructure-update.json';path.parent.mkdir(parents=True)
    path.write_bytes(authority[1].read_bytes());calls=[]
    monkeypatch.setattr(m2_privacy,'scan',lambda *a:[])
    monkeypatch.setattr(publish_kit.subprocess,'run',lambda *a,**kw:calls.append(kw) or NS(returncode=0,stdout='1 passed'))
    publish_kit.check(target)
    assert calls[0]['env']['KIT_V6_AUTHORITY_SHA']==sha(path.read_bytes())
    assert not list(target.glob('.publish-tests-*'))


def test_generator_refuses_a_late_base_that_omits_still_live_pins(repo):
    root,base,target=repo
    with pytest.raises(ValueError,match='pre-v6 kit release base'):
        generate(target,root,target)
    assert uncovered_pins(generate(base,root,target),root,target)==[]


def test_generator_cli_uses_the_explicit_release_base(repo,tmp_path,monkeypatch):
    from scripts import v6_infra_authority as generator
    root,base,target=repo;out=tmp_path/'generated-authority.json'
    monkeypatch.setattr(generator,'generate',lambda base,target:generate(base,root,target))
    generator.main(['--base',base,'--target',target,'--out',str(out)])
    doc=json.loads(out.read_text())
    assert doc['base_commit']==base and doc['target_commit']==target
    assert not uncovered_pins(doc,root,target)
    assert 'pre-v6' in generator.__doc__ and '68100bf6' in generator.__doc__
