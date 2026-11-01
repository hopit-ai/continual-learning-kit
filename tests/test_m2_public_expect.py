"""V5: node-local archive binding must match every public content field."""
import copy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kit import m2_expect as e


def fixture():
    hashes={'config.json':'a'*64}
    cdoc={'starts':{f'D:{s}':dict(identity=dict(model_file_hashes=hashes),receipt_sha256=str(s)*20) for s in (101,102,103)}}
    ruling=dict(rescoring=dict(packages=[dict(calls=[dict(checkpoint='/local/node/checkpoint')])]),
        continuation=dict(exports=[dict(arm='sequence-D',seed=s,step=40,model_hashes=hashes,receipt_sha256=str(s)*20) for s in (101,102,103)]))
    binding=dict(W='D',provisional=True,hard_math='QUALIFIES',nomination=dict(status='PENDING'),
        archive_sha256='b'*64,rescore_refs={'/local/node/checkpoint':dict(model='/local/node/model',receipt='/local/node/proof',model_hashes=hashes)})
    return binding,cdoc,ruling


def test_matching_fields_and_relocated_paths_pass_without_printing(tmp_path,capsys):
    b,c,r=fixture();p=tmp_path/'expected.json'
    p.write_text(json.dumps(e.public_document(b,c,r)))
    moved=copy.deepcopy(b);moved['rescore_refs']['/local/node/checkpoint'].update(model='/different/place',receipt='/different/proof')
    assert e.check_public(p,moved,c,r)==json.loads(p.read_text())
    text=p.read_text()+capsys.readouterr().out+capsys.readouterr().err
    assert '/local/node' not in text and '/different' not in text


@pytest.mark.parametrize('field',['archive_sha256','W','hard_math','provisional','nomination','extra_content'])
def test_any_changed_hash_decision_or_extra_field_refuses(tmp_path,field):
    b,c,r=fixture();p=tmp_path/'expected.json';p.write_text(json.dumps(e.public_document(b,c,r)))
    b[field]='changed'
    with pytest.raises(ValueError,match='content fields; locations excluded') as exc:e.check_public(p,b,c,r)
    assert '/local/node' not in str(exc.value) and 'changed' not in str(exc.value)


def test_changed_model_identity_refuses(tmp_path):
    b,c,r=fixture();p=tmp_path/'expected.json';p.write_text(json.dumps(e.public_document(b,c,r)))
    b['rescore_refs']['/local/node/checkpoint']['model_hashes']['config.json']='0'*64
    with pytest.raises(ValueError,match='expectation mismatch'):e.check_public(p,b,c,r)


def test_public_expected_privacy_and_handoff_hashes():
    from kit.m2_privacy import scan
    doc=json.loads(e.EXPECTED.read_text())
    assert not scan(ROOT,[e.EXPECTED])
    assert doc['handoff_receipts']==e.receipt_hashes()
    assert doc['W']=='D' and doc['provisional'] and doc['hard_math']=='QUALIFIES'
    assert doc['nomination']['status']=='PENDING' and set(doc['c_identities'])=={'101','102','103'}
    assert all(set(ref['content'])=={'model_hashes'} for ref in doc['rescore_refs'])


def test_real_shipped_p2_pd_receipts_still_qualify():
    from kit.x_p import smoke
    from kit import pd_probe, pd_smoke_modal
    assert smoke(e.KIT/'resources/m2/receipts'/e.RECEIPTS['p2_attempt5'],False)['passed']
    pd=json.loads((e.KIT/'resources/m2/receipts'/e.RECEIPTS['pd_attempt3']).read_text())
    assert pd_smoke_modal.smoke_gates(pd)['status']=='QUALIFIED'
    assert pd['code_sha256']==pd_probe.code_identity()


def test_node_send_checks_before_freezing_local_binding(tmp_path,monkeypatch):
    from kit import m2_send
    from types import SimpleNamespace as NS
    b,c,r=fixture();archive=tmp_path/'return.tar.gz';archive.write_bytes(b'original archive stand-in')
    b['b_archive_sha256']=e.hashlib.sha256(archive.read_bytes()).hexdigest()
    expected=tmp_path/'expected.json';expected.write_text(json.dumps(e.public_document(b,c,r)))
    monkeypatch.setattr(m2_send,'read_return1',lambda a:(b,c,{},{}))
    import kit.m2_return1
    monkeypatch.setattr(kit.m2_return1,'decisions',lambda p:r)
    a=NS(public_from=None,public_expect=expected,b_source=archive,c_source=archive,teacher_source=archive,b_reading=archive,decisions=archive,out=tmp_path/'send')
    m2_send.job(a)
    assert (a.out/'send-binding.json').is_file()
    a.out=tmp_path/'refused';b['W']='wrong'
    with pytest.raises(ValueError,match='expectation mismatch'):m2_send.job(a)
    assert not (a.out/'send-binding.json').exists()


@pytest.mark.parametrize('key',['hostname','username','job_id','gpu_uuid'])
def test_public_generator_refuses_private_metadata_without_printing(key):
    b,c,r=fixture();b[key]='private-fixture-value'
    with pytest.raises(ValueError,match='private metadata') as exc:e.public_document(b,c,r)
    assert 'private-fixture-value' not in str(exc.value)


def test_real_node_command_defaults_to_public_expectation():
    import argparse
    from kit.m2_send import configure
    p=argparse.ArgumentParser();configure(p)
    assert p.parse_args([]).public_expect==e.EXPECTED


def test_cli_public_refusal_has_no_traceback_or_location(tmp_path):
    import subprocess
    missing=tmp_path/'private-node-path'/'expected.json'
    program="""import sys
from pathlib import Path
sys.path.insert(0,sys.argv.pop(1))
from kit.m2_expect import check_public
from kit.p3b_common import entry
def job(a):check_public(a.expect,{},{},{})
def configure(p):p.add_argument('--expect',type=Path,required=True)
raise SystemExit(entry(job,configure,'public-check'))
"""
    done=subprocess.run([sys.executable,'-I','-c',program,str(ROOT),'--out',str(tmp_path/'state'),'--expect',str(missing)],capture_output=True,text=True)
    assert done.returncode==2
    assert 'public send expectation unreadable or invalid' in done.stdout
    assert 'Traceback' not in done.stdout+done.stderr
    assert str(missing) not in done.stdout+done.stderr and str(ROOT) not in done.stdout+done.stderr


def test_core_binder_enforces_fixed_public_check_before_local_expect(tmp_path,monkeypatch):
    """V5 review finding 2: a plausible local expectation cannot bypass public content."""
    from types import SimpleNamespace as NS
    from kit import m2_bind,m2_return1
    b,c,r=fixture();archive=tmp_path/'original.tar.gz';archive.write_bytes(b'original B bytes')
    local=tmp_path/'send-binding.json';local.write_text(json.dumps(b))
    monkeypatch.setattr(m2_bind,'read_return1',lambda a:(copy.deepcopy(b),c,{},{}))
    monkeypatch.setattr(m2_return1,'decisions',lambda p:r)
    monkeypatch.setattr(m2_bind,'check_expect',lambda *args:pytest.fail('local expectation ran before public refusal'))
    a=NS(standin=False,b_source=archive,decisions=local,expect=local,out=tmp_path/'out')
    with pytest.raises(ValueError,match='public send expectation mismatch'):m2_bind.job(a)
    assert not a.out.exists()


@pytest.mark.parametrize('tree',[False,True])
def test_binder_hashes_original_b_input_not_local_claim(tmp_path,monkeypatch,tree):
    from types import SimpleNamespace as NS
    from kit import m2_bind,m2_return1
    b,c,r=fixture();archive=tmp_path/'original.tar.gz';archive.write_bytes(b'B source bytes')
    source=tmp_path/'live-tree' if tree else archive
    if tree:source.mkdir()
    monkeypatch.setattr(m2_bind,'read_return1',lambda a:(b,c,{},{}))
    monkeypatch.setattr(m2_return1,'decisions',lambda p:r)
    def public(path,observed,cdoc,ruling):
        assert path==e.EXPECTED and observed['b_archive_sha256']==e.hashlib.sha256(archive.read_bytes()).hexdigest()
        assert cdoc==c and ruling==r
    monkeypatch.setattr(e,'check_public',public)
    def stop(*args):raise RuntimeError('public check completed before local check')
    monkeypatch.setattr(m2_bind,'check_expect',stop)
    with pytest.raises(RuntimeError,match='public check completed'):
        m2_bind.job(NS(standin=False,b_source=source,b_archive=archive,decisions=archive,expect=archive))


def test_send_refuses_standin_with_real_decisions_before_reading(monkeypatch):
    from types import SimpleNamespace as NS
    from kit import m2_send
    monkeypatch.setattr(m2_send,'read_return1',lambda a:pytest.fail('real reader must not run'))
    with pytest.raises(ValueError,match='--standin cannot be combined with --decisions'):
        m2_send.job(NS(standin=True,decisions=e.EXPECTED))


def test_public_expectation_is_frozen():
    frozen=json.loads((e.KIT/'resources/m2/freeze.json').read_text())['files']
    assert frozen[e.EXPECTED.name]==e.hashlib.sha256(e.EXPECTED.read_bytes()).hexdigest()


def test_resolver_template_covers_exactly_17_registered_tier_a_keys():
    from kit.m2_return1 import decisions
    from kit.m2_bind import tier_a_base
    ruling=decisions();calls=[c for p in ruling['rescoring']['packages'] if p['tier']=='A' for c in p['calls']]
    template=json.loads((e.KIT/'resources/m2/rescore-paths.template.json').read_text())
    assert len(template)==17 and set(template)=={c['checkpoint'] for c in calls}
    assert all(v.get('model') and (v.get('receipt') or all(tier_a_base(c) for c in calls if c['checkpoint']==key)) for key,v in template.items())
    readme=(e.KIT/'README-combined.md').read_text()
    expected=json.loads(e.EXPECTED.read_text())
    assert expected['b_archive_sha256'] in readme
    assert all(value in readme for value in expected['return1_archives'].values())


def test_bind_refuses_standin_with_real_decisions_before_reading(monkeypatch):
    from types import SimpleNamespace as NS
    from kit import m2_bind
    monkeypatch.setattr(m2_bind,'read_return1',lambda a:pytest.fail('real reader must not run'))
    with pytest.raises(ValueError,match='--standin cannot be combined with --decisions'):
        m2_bind.job(NS(standin=True,decisions=e.EXPECTED))
