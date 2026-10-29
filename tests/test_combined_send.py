"""Integration-only contracts: no scientific implementation is replaced here."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from kit import combined_send as c, combined_dryrun as dry, m2_privacy


@pytest.fixture
def doc(tmp_path): return dry.fixtures(tmp_path)


def test_every_source_task_and_pd_seed_is_in_inventory(doc):
    tasks=doc['tasks']; ids={r['id'] for r in tasks}
    for key in ('core_manifest','kcr_manifest','sp_manifest'):
        assert {t['id'] for t in c.load(doc['settings'][key])['tasks']}<=ids
    assert {'PD-101','PD-102','PD-103'}<=ids
    assert '1,800 GPU-h' in c.inventory(doc)
    assert next(b for b in doc['budgets'] if b['component']=='K/CR')['gpu_h']==[786.297517,1387.157525]
    assert doc['total_gpu_h'] is None
    assert next(r for r in tasks if r['id']=='M-core-collect')['dependency']=='afterany'
    assert next(r for r in tasks if r['id']=='M3-read')['dependency']=='afterany'
    assert next(b for b in doc['budgets'] if b['component']=='PD')['gpu_h']==[221.594915,263.620406]
    assert not any('/staged/' in line for line in c.inventory(doc).splitlines())
    core={r['id'] for r in tasks if r['group']=='core'}
    assert all(set(r['requires'])<=core for r in tasks if r['group']=='core')


def test_build_rejects_core_held_node_alias_and_nested_roots(doc):
    s=copy.deepcopy(doc['settings']);s['pd_nodes']=['core-fixture.cluster']
    with pytest.raises(ValueError,match='core-held'):c.build(s)
    s=copy.deepcopy(doc['settings']);s['sp_root']=str(Path(s['core_local_root'])/'sp')
    with pytest.raises(ValueError,match='non-nested'):c.build(s)


def production(doc):
    doc=copy.deepcopy(doc);doc['technical_synthetic']=False
    return doc


def ready(doc):
    s=doc['settings'];control=Path(s['control_root'])
    c.write(control/'core-submitted.json',dict(core_manifest_sha256=doc['sources']['core_manifest']))
    s['tier_send']=dict(status='SEND',date='2026-11-02',review_sha256=c.sha(c.REVIEW))
    c.write(s['tier_hardware'],dict(synthetic=True))
    proof=dict(status='PASS',technical_synthetic=False,hardware_sha256=c.sha(s['tier_hardware']),
        settings_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest())
    c.write(control/'preflight/tier-fixture.json',proof)


def test_real_combined_core_dispatch_and_tier_submission_gate(doc):
    doc=production(doc);calls=[]
    def runner(command,**kw):calls.append(command);return NS(returncode=0)
    with patch.object(c.socket,'gethostname',return_value='core-fixture'):
        assert c.execute(doc,'M4-101',runner=runner)==0
        assert calls[-1][0]=='bash' and calls[-1][-2:]==['--node-slot','0/1']
        with pytest.raises((ValueError,FileNotFoundError)):c.execute(doc,'PD-101',runner=runner)
    with patch.object(c.socket,'gethostname',return_value='tier-fixture'):
        with pytest.raises(FileNotFoundError):c.execute(doc,'M3-init',runner=runner)
    assert len(calls)==1


def test_combined_tier_gates_pd_node_and_sp_author_hooks(doc):
    doc=production(doc);ready(doc);s=doc['settings'];calls=[]
    def runner(command,**kw):calls.append(command);return NS(returncode=7)
    with patch.object(c.socket,'gethostname',return_value='tier-fixture'):
        assert c.execute(doc,'M3-init',runner=runner)==7
        with pytest.raises(ValueError,match='allocation'):c.execute(doc,'PD-101',runner=runner)
        s['pd_allocation_approved']=True;ready(doc)
        assert c.execute(doc,'PD-101',runner=runner)==7
        assert Path(calls[-1][1]).name=='pd_node.py'
        panel=next(r['id'] for r in doc['tasks'] if r['id'].startswith('SP-panel-'))
        assert c.execute(doc,panel,runner=runner)==7
        assert 'node_verification.py' in ' '.join(calls[-1]) and 'run' in calls[-1] and '--' in calls[-1]
        assert c.execute(doc,'SP-read',runner=runner)==7
        assert 'report' in calls[-1] and calls[-1][-1].endswith('verified-tables')
        s['tier_send']['date']='2026-11-03'
        with pytest.raises(ValueError,match='Opus'):c.execute(doc,'M3-init',runner=runner)
    assert not Path(s['core_durable'],'barriers/M9.json').exists()


def test_changed_sources_and_synthetic_receipts_fail_closed(doc):
    with pytest.raises(ValueError,match='CPU rehearsal'):c.execute(doc,'M3-init')
    doc=production(doc);ready(doc)
    proof=Path(doc['settings']['control_root'])/'preflight/tier-fixture.json'
    value=c.load(proof);value['technical_synthetic']=True;c.write(proof,value)
    with patch.object(c.socket,'gethostname',return_value='tier-fixture'):
        with pytest.raises(ValueError,match='preflight'):c.execute(doc,'M3-init')
    Path(doc['settings']['core_manifest']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):c.execute(doc,'M4-101')


def test_tier_node_preflight_retains_pending_for_dependency_failure(tmp_path):
    from kit import combined_preflight as pf
    s={'control_root':str(tmp_path)}
    with patch.object(pf,'check',side_effect=RuntimeError('missing import')),patch('sys.argv',[
        'preflight','--settings-json',json.dumps(s),'--node','tier-fixture']):
        assert pf.main()==3
    proof=c.load(tmp_path/'preflight/tier-fixture.json')
    assert proof['status']=='PENDING' and proof['technical_synthetic'] is False


def test_all_tier_files_in_publisher_scope_and_private_runtime_maps_not_copied(tmp_path):
    publish_kit=pytest.importorskip('scripts.publish_kit',reason='publisher is private build tooling')
    root=Path(__file__).resolve().parents[1]
    target=publish_kit.build(tmp_path/'public')
    files=c.load(target/'EXPORT-INVENTORY.json')
    for prefix in ('kit/m3_', 'kit/pd_', 'kit/sp_', 'kit/resources/m3/', 'kit/resources/pd/', 'kit/resources/sp/', 'kit/jobs/message3/', 'kit/jobs/message3-sp/', 'kit/jobs/combined/'):
        expected={str(p.relative_to(root)) for p in (root/'kit').rglob('*') if p.is_file() and str(p.relative_to(root)).startswith(prefix) and '__pycache__' not in p.parts}
        assert expected and expected<=set(files)
        assert all(c.sha(target/p)==c.sha(root/p) for p in expected)
    assert c.sha(target/'kit/resources/sp/FROZEN.json')==c.sha(root/'kit/resources/sp/FROZEN.json')
    assert not m2_privacy.scan(target)
    leak=target/'kit/resources/sp/runtime-local.json';leak.write_text(json.dumps({'job_id':123456789}))
    assert m2_privacy.scan(target)==['kit/resources/sp/runtime-local.json:1']


def test_kcr_shell_wrapper_handles_no_submission_timestamp_and_optional_timestamp(tmp_path):
    import os,subprocess,sys
    capture=tmp_path/'argv.json';shim=tmp_path/'python-shim'
    shim.write_text('#!'+sys.executable+'\nimport json,os,sys\nopen(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    shim.chmod(0o755)
    env=dict(os.environ,KIT=str(c.KIT),M3_OUT=str(tmp_path/'m3'),M3_MANIFEST=str(tmp_path/'launch.json'),JOB_PY=str(shim),CAPTURE=str(capture))
    env.pop('M3_SUBMITTED_AT',None)
    wrapper=c.KIT/'jobs/message3/run-task.sh'
    subprocess.run(['bash',str(wrapper),'M3-init'],env=env,check=True)
    assert '--submitted-at' not in c.load(capture)
    env['M3_SUBMITTED_AT']='1000'
    subprocess.run(['bash',str(wrapper),'M3-init'],env=env,check=True)
    assert c.load(capture)[-4:-2]==['--submitted-at','1000']


def test_readonly_target_copy_verifies_hashes_without_core_lease_or_mutation(tmp_path):
    from kit import combined_snapshot, m2_snapshot as snap
    source=tmp_path/'source';c.write(source/'targets-math/manifest.json',dict(technical_synthetic=True))
    (source/'targets-math/pool.jsonl').write_text('{}\n')
    durable=tmp_path/'core'
    with snap.technical_space_budget(True):
        pointer=snap.publish(source,durable,'M5',complete=True)
        def inventory():return {str(p.relative_to(durable)):(p.read_bytes(),p.stat().st_mtime_ns) for p in durable.rglob('*') if p.is_file()}
        before=inventory()
        doc,seen=combined_snapshot.restore_targets(durable,'M5',tmp_path/'copy')
        assert seen==pointer and inventory()==before
        assert (tmp_path/'copy/targets-math/pool.jsonl').read_bytes()==b'{}\n'
        tar=durable/'snapshots/M5'/pointer['tar'];tar.write_bytes(tar.read_bytes()+b'tamper')
        with pytest.raises(Exception):combined_snapshot.restore_targets(durable,'M5',tmp_path/'bad')
        assert not list(durable.rglob('*.lease')) and not (tmp_path/'bad').exists()


def test_core_dispatch_survives_deleted_tier_manifests_and_unavailable_tier_imports(doc):
    """B1: core runs even after the optional K/CR inputs are abandoned."""
    import builtins
    doc=production(doc)
    for key in ('kcr_manifest','sp_manifest','sp_inventory'):
        Path(doc['settings'][key]).unlink()
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert name not in ('kit.m3_manifest','kit.sp_manifest')
        assert not (name=='kit' and set(kwargs.get('fromlist', args[2] if len(args)>2 else ()) or ()) & {'m3_manifest','sp_manifest'})
        return original(name,*args,**kwargs)
    with patch.object(c.socket,'gethostname',return_value='core-fixture'),patch('builtins.__import__',guarded):
        assert c.execute(doc,'M4-101',runner=lambda *a,**k:NS(returncode=0))==0


def test_tier_review_hash_binds_shipped_bytes(doc,tmp_path,monkeypatch):
    doc=production(doc);ready(doc)
    review=tmp_path/'review.md';review.write_bytes(c.REVIEW.read_bytes()+b'changed')
    monkeypatch.setattr(c,'REVIEW',review)
    with patch.object(c.socket,'gethostname',return_value='tier-fixture'):
        with pytest.raises(ValueError,match='Opus'):c.execute(doc,'M3-init')


def test_sp_report_success_does_not_erase_failed_sidecar(doc):
    doc=production(doc);ready(doc);codes=iter((7,0))
    with patch.object(c.socket,'gethostname',return_value='tier-fixture'):
        assert c.execute(doc,'SP-read',runner=lambda *a,**k:NS(returncode=next(codes)))==7


def test_combined_shell_wrapper_invokes_selected_python(tmp_path):
    import os,subprocess,sys
    capture=tmp_path/'argv.json';shim=tmp_path/'job-python'
    shim.write_text('#!'+sys.executable+'\nimport json,os,sys\nopen(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    shim.chmod(0o755)
    env=dict(os.environ,KIT=str(c.KIT),COMBINED_MANIFEST=str(tmp_path/'launch.json'),JOB_PY=str(shim),CAPTURE=str(capture))
    subprocess.run(['bash',str(c.KIT/'jobs/combined/run-task.sh'),'M4-101'],env=env,check=True)
    assert c.load(capture)==[str(c.KIT/'combined_send.py'),'run','--manifest',env['COMBINED_MANIFEST'],'--task','M4-101']


def test_combined_cli_gate_and_unexpected_errors_have_distinct_codes(doc,monkeypatch):
    path=Path(doc['settings']['control_root']).parent/'combined.json'
    monkeypatch.setattr('sys.argv',['combined','run','--manifest',str(path),'--task','M4-101'])
    assert c.main()==3
    with patch.object(c,'execute',side_effect=RuntimeError('implementation bug')):
        assert c.main()==1


def test_privacy_rejects_private_prefixes_and_email(tmp_path):
    probes=['/private/'+'tmp/example','/opt/'+'anaconda3/bin/python','somebody'+'@'+'example.test']
    for value in probes:
        path=tmp_path/'example.txt';path.write_text(value)
        assert m2_privacy.scan(tmp_path)==['example.txt:1']


def test_email_exception_is_bound_to_frozen_scientific_bytes(tmp_path):
    source=c.KIT/'resources/c/contract_nli/train.jsonl'
    path=tmp_path/'kit/resources/c/contract_nli/train.jsonl'
    path.parent.mkdir(parents=True);path.write_bytes(source.read_bytes())
    assert m2_privacy.scan(tmp_path)==[]
    path.write_bytes(path.read_bytes()+b'\ncontact' + b'@' + b'private.test')
    assert m2_privacy.scan(tmp_path)
