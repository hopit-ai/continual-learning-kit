"""V7b public export and native environment regressions, CPU only."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('module_name',[
    'test_kit_bed_authors','test_kit_long_cap','test_kit_phase0_v7',
])
def test_scorer_environment_fixture_preserves_native_mapping_and_restores_writes(module_name):
    module=importlib.import_module(module_name)
    environment=os.environ;before=dict(environment)
    fixture=module.restore_environment.__wrapped__()
    next(fixture)
    try:
        assert os.environ is environment and isinstance(os.environ,os._Environ)
        os.environ['VLLM_BATCH_INVARIANT']='v7b-native-fixture'
        assert environment['VLLM_BATCH_INVARIANT']=='v7b-native-fixture'
        observed=subprocess.check_output([sys.executable,'-I','-c',
            'import os;print(os.environ["VLLM_BATCH_INVARIANT"])'],text=True).strip()
        assert observed=='v7b-native-fixture'
    finally:
        with pytest.raises(StopIteration):next(fixture)
    assert os.environ is environment and dict(os.environ)==before
    observed=json.loads(subprocess.check_output([sys.executable,'-I','-c',
        'import json,os;print(json.dumps(os.environ.get("VLLM_BATCH_INVARIANT")))'],text=True))
    assert observed==before.get('VLLM_BATCH_INVARIANT')


def test_runnable_confirmation_in_standalone_export(tmp_path):
    target=ROOT
    if (ROOT/'scripts/publish_kit.py').is_file():
        from scripts import publish_kit
        target=publish_kit.build(tmp_path/'public',tag='kit-message-v7b')
        assert (target/'scripts/cr_confirmation_fixture.py').read_bytes()==(ROOT/'scripts/cr_confirmation_fixture.py').read_bytes()
        assert (target/'tests/test_cr_finqa.py').read_bytes()==(ROOT/'tests/test_cr_finqa.py').read_bytes()
        assert not (target/'.git').exists()
        publish_kit.check_markdown_links(target)
    done=subprocess.run([sys.executable,'scripts/cr_confirmation_dryrun.py','--runnable',
        '--out',str(tmp_path/'runnable')],cwd=target,capture_output=True,text=True,timeout=600)
    assert done.returncode==0,done.stdout[-4000:]+done.stderr[-4000:]
    result=json.loads((tmp_path/'runnable/result.json').read_text())
    assert result['rows']==result['complete']==78
    assert result['finqa_runnable'] and result['authority_unchanged']
    assert result['finqa_reproduction']['status']=='COMPLETE'
    assert result['parts']<=4


def test_opr_advisories_fit_seven_family_essential_return(tmp_path,monkeypatch):
    from test_cr_confirmation import evidence
    from test_nm_text import stress_dispatch
    from scripts.cr_confirmation_fixture import build
    from kit import cr_confirmation,nm_pipeline,nm_text
    monkeypatch.setenv('KIT_M3_MODE','cr-confirmation')
    cfg,_=build(tmp_path/'fixture')
    proof=cr_confirmation.bind_inputs(cfg,tmp_path/'binding')['finqa_reproduction']
    cells,assignment=evidence();reading=cr_confirmation.analyse(cells,assignment)
    reading['finqa_reproduction']=proof
    full=stress_dispatch()
    opr=next(family['reading'] for family in full['nm_dispatch'] if family['nm_family']=='opr')
    record=nm_pipeline.generation_utilisation(tmp_path/'missing-summary.json')
    opr['generation_qualification']={str(seed):record for seed in (101,102,103)}
    opr['utilisation_warnings']=[f'generation_qualification_s{seed}_'+warning
        for seed in (101,102,103) for warning in record['warnings']]
    full['nm_dispatch'].append(dict(nm_family='cr',reading=reading,infrastructure_authority='a'*64))
    parts=nm_text.write_parts(tmp_path/'text',full,'kit-message-v7b','b'*40)
    parsed=nm_text.parse_parts([part.read_text() for part in parts])
    assert len(parts)<=4
    assert parsed['families']['opr']['utilisation_warnings']==opr['utilisation_warnings']
    assert parsed['families']['cr']['metadata']['finqa']['reproduction']==proof


@pytest.mark.skipif(not (ROOT/'scripts/publish_kit.py').is_file(),reason='private export builder is absent')
def test_export_links_ship_reviewed_bytes_and_pass_privacy(tmp_path):
    from scripts import publish_kit
    from kit.m2_privacy import scan
    target=publish_kit.build(tmp_path/'public',tag='kit-message-v7b')
    copies={
        'kit/resources/nm/registrations/registration-cr-confirmation.md':
            'docs/phase3/next-message-20261102/registration-cr-confirmation.md',
        'docs/phase2/v4-frozen-gate-commands.md':'docs/phase2/v4-frozen-gate-commands.md',
        'docs/phase3/evidence/screen-s/finqa-arc-feasibility-20261025.json':
            'docs/phase3/evidence/screen-s/finqa-arc-feasibility-20261025.json',
    }
    for destination,source in copies.items():
        assert (target/destination).read_bytes()==(ROOT/source).read_bytes()
        assert not publish_kit.DENY.search((target/destination).read_text())
    assert scan(target,paths=[target/name for name in copies])==[]
    publish_kit.check_markdown_links(target)
    (target/'kit/resources/nm/registrations/registration-cr-confirmation.md').unlink()
    with pytest.raises(SystemExit,match='README-next-message.md.*registration-cr-confirmation.md'):
        publish_kit.check_markdown_links(target)


@pytest.mark.skipif(not (ROOT/'scripts/publish_kit.py').is_file(),reason='private export builder is absent')
def test_publisher_rejects_dangling_runbook_links_before_test_gate(tmp_path,monkeypatch):
    from scripts import publish_kit
    kit=tmp_path/'kit';kit.mkdir()
    (kit/'shipped file.json').write_text('{}')
    runbook=kit/'README.md'
    runbook.write_text('[shipped](shipped%20file.json#anchor)\n'
        '[angle](<shipped file.json>)\n[reference][receipt]\n'
        '[receipt]: shipped%20file.json "Title"\n'
        '[web](https://example.com/missing)\n[anchor](#local)\n'
        '`[inline example](missing-inline.json)`\n'
        '```md\n[example](missing-example.json)\n```\n')
    publish_kit.check_markdown_links(tmp_path)
    monkeypatch.setattr(publish_kit.subprocess,'run',lambda *a,**kw:pytest.fail('tests ran before link validation'))
    runbook.write_text(runbook.read_text()+'[missing](missing.json#section)\n')
    with pytest.raises(SystemExit,match='kit/README.md:.*missing.json'):
        publish_kit.check(tmp_path)


@pytest.mark.skipif(not (ROOT/'scripts/publish_kit.py').is_file(),reason='private export builder is absent')
@pytest.mark.parametrize('label',['across\na line break','nested [inner] label','outer [inner [deep]] label',r'escaped \[brackets\]'])
def test_publisher_balances_link_labels_and_ignores_footnote_definitions(tmp_path,label):
    from scripts import publish_kit
    kit=tmp_path/'kit';kit.mkdir();runbook=kit/'README.md'
    (kit/'present.md').write_text('shipped')
    (kit/'README-other.md').write_text('['+label+'](present.md)\n')
    prefix='[^1]: See frozen registration for details.\n[^note]: Discussion only.\n'
    runbook.write_text(prefix+'['+label+'](present.md)\n')
    publish_kit.check_markdown_links(tmp_path)
    runbook.write_text(prefix+'['+label+'](missing.md)\n')
    with pytest.raises(SystemExit,match=r'kit/README.md:3: missing.md'):
        publish_kit.check_markdown_links(tmp_path)
    # Real reference definitions are still checked, including angle destinations.
    runbook.write_text(prefix+'[receipt]: <missing receipt.json>\n')
    with pytest.raises(SystemExit,match='missing receipt.json'):
        publish_kit.check_markdown_links(tmp_path)
