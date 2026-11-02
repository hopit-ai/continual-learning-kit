"""Combined v6b integration contracts; all inputs are CPU fixtures."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from kit import combined_send as c,combined_dryrun as dry,a4_jobs,v6_infra


@pytest.fixture
def doc(tmp_path,monkeypatch):
    monkeypatch.setenv(v6_infra.ENV,v6_infra.authority_sha256())
    value=dry.fixtures(tmp_path)
    config=tmp_path/'a4-config.json';c.write(config,dict(out=str(tmp_path/'A4')))
    value['settings']['a4_config']=str(config)
    return c.build(value['settings'])


def test_all_4a_rows_keep_resources_dependencies_counts_and_report_frontier(doc):
    rows={r['id']:r for r in doc['tasks']}
    for original in a4_jobs.inventory()['rows']:
        row=rows[original['id']]
        assert row['group']=='4a' and row['gpus']==original['gpus']
        assert row['nodes']==original['nodes'] and row['cpus']==original['cpus']
        assert row['requires']==[d['id'] for d in original['depends_on']]
        assert row['depends_on']==original['depends_on']
        assert row['dependency_options']==['--kill-on-invalid-dep=yes']
        assert row['expected_command_counts']==original['expected_command_counts']
    assert rows['A4-read']['dependency']=='afterany'
    assert rows['REPORT-text']['dependency']=='afterany'
    assert 'A4-collect' in rows['REPORT-text']['requires']
    assert doc['sources']['a4_config']==c.sha(doc['settings']['a4_config'])


def test_v6b_reuses_v6a_and_never_resubmits_it(doc):
    value=c.build_v6b(doc['settings']);rows=value['tasks'];ids={r['id'] for r in rows}
    assert all(r['group']!='kcr' and r.get('family')!='m3' for r in rows)
    assert {'core','4a','pd','sp','reference','report'}<={r['group'] for r in rows}
    assert all(set(r['requires'])<=ids for r in rows)
    assert any(r.get('reused_v6a_requires') for r in rows)
    assert all(r['dependency_options']==['--kill-on-invalid-dep=yes'] for r in rows)


def test_4a_dispatch_is_independent_and_config_is_sealed(doc,monkeypatch):
    doc['technical_synthetic']=False;s=doc['settings'];calls=[]
    monkeypatch.setenv('UTIL_WATCH_ACTIVE','1')
    monkeypatch.setattr(c.socket,'gethostname',lambda:'tier-fixture')
    assert c.execute(doc,'A4-bind',runner=lambda cmd,**kw:(calls.append((cmd,kw)) or NS(returncode=75)))==75
    cmd,kw=calls[0]
    assert cmd==['bash',str(c.KIT/'jobs/a4/run-task.sh'),'A4-bind']
    assert kw['env']['A4_CONFIG']==s['a4_config']
    Path(s['a4_config']).write_text('{}')
    with pytest.raises(ValueError,match='4A config changed'):c.execute(doc,'A4-bind')


def test_report_automatically_adds_mandatory_4a_source(doc,monkeypatch):
    doc['technical_synthetic']=False;s=doc['settings'];calls=[]
    inputs=Path(s['control_root'])/'report-inputs.local.json';c.write(inputs,dict(M=[]))
    monkeypatch.setattr(c.socket,'gethostname',lambda:'tier-fixture')
    assert c.execute(doc,'REPORT-text',runner=lambda cmd,**kw:(calls.append(cmd) or NS(returncode=0)))==0
    prepared=c.load(calls[0][calls[0].index('--inputs')+1])
    assert prepared['4A']==[dict(source=c.load(s['a4_config'])['out'],bootstrap=2000)]
    assert c.load(inputs)==dict(M=[])
    assert any(spec['family']=='4A' for spec in prepared['utilisation'])
    jobs=next(spec for spec in prepared['jobs'] if spec['family']=='4A')
    assert len(c.load(jobs['manifest'])['tasks'])==197
    assert jobs['state']==str(Path(c.load(s['a4_config'])['out'])/'v6b/status')


def test_three_seed_4a_text_preserves_amended_df_and_month_bar():
    from kit import text_return,a4_read,a4_common
    result=dict(primary=a4_read.primary({s:[6]*5 for s in (101,102,103)}),
        amendment=a4_common.AMENDMENT,retention=[],transitions=[])
    rows=text_return.essential_endpoints('4A',result)
    primary=next(r for r in rows if r['name']=='consolidation-memory')
    assert primary['values']==[6,6,6] and primary['n']==3
    assert primary['method']=='seed-t-n3-df2'
    assert primary['bar']=='mean>=5;lo>0;pos=3;months>=3'
    monthly=next(r for r in rows if r['name']=='positive-months')
    assert len(monthly['values'])==5 and monthly['bar']=='positive-months>=3'
    names,mapping=set(),text_return.alias_context()
    assert '|mean>=5;lo>0;pos=3;months>=3|' in text_return.erow('4A',primary,names,mapping)
    assert '|positive-months>=3|' in text_return.erow('4A',monthly,names,mapping)


def test_reused_v6a_reader_requires_completed_attempts_and_restores_environment(tmp_path,monkeypatch):
    import os
    from kit import report_text,m3_read
    source=tmp_path/'kcr';source.mkdir()
    def read(args):
        assert os.environ['KIT_V6A']=='1'
        assert args.out==source
        c.write(args.result_out/'m3-read.json',dict(status='PENDING'))
    monkeypatch.delenv('KIT_V6A',raising=False)
    monkeypatch.setattr(m3_read,'read_job',read)
    result=report_text.native('KCR',dict(source=str(source),v6a=True),tmp_path/'scratch',True,2)
    assert result['status']=='PENDING' and 'KIT_V6A' not in os.environ


def test_unstaged_4a_config_requires_rebuilding_the_launch_binding(doc,monkeypatch):
    doc['technical_synthetic']=False;doc['sources'].pop('a4_config')
    monkeypatch.setattr(c.socket,'gethostname',lambda:'tier-fixture')
    with pytest.raises(ValueError,match='rebuild the bound inventory'):
        c.execute(doc,'A4-bind',runner=lambda *a,**kw:pytest.fail('unbound config was launched'))


def test_refused_old_reader_cannot_clear_the_report_or_next_family_authority(tmp_path,monkeypatch):
    import os
    from kit import report_text,text_return
    seal=v6_infra.authority_sha256()
    monkeypatch.setenv(v6_infra.ENV,seal)
    old=tmp_path/'old';old.mkdir();c.write(old/'binding.json',dict(protocol={}))
    seen=[]
    def native(family,*args):
        seen.append(family)
        if family=='M':v6_infra.activate(old)
        assert os.environ[v6_infra.ENV]==os.environ['M2_INFRA_AUTHORITY_SHA256']==seal
        return dict(status='PENDING')
    monkeypatch.setattr(report_text,'native',native)
    parts=report_text.report(dict(M=[dict(source=str(old))],**{'4A':[{}]}),
        tmp_path/'text',tag='kit-test',commit='a'*40)
    parsed=text_return.parse_parts([p.read_text() for p in parts])
    assert seen==['M','4A'] and parsed['infrastructure_authority']==seal
    assert os.environ[v6_infra.ENV]==os.environ['M2_INFRA_AUTHORITY_SHA256']==seal


def test_partial_report_does_not_follow_a_changed_or_unbound_4a_config(doc,monkeypatch):
    doc['technical_synthetic']=False;s=doc['settings'];calls=[]
    c.write(s['a4_config'],dict(out=str(Path(s['control_root'])/'different-4a')))
    monkeypatch.setattr(c.socket,'gethostname',lambda:'tier-fixture')
    assert c.execute(doc,'REPORT-text',runner=lambda cmd,**kw:(calls.append(cmd) or NS(returncode=0)))==0
    prepared=c.load(calls[0][calls[0].index('--inputs')+1])
    assert prepared['4A']==[dict(source=str(Path(s['control_root'])/'4a-pending'),bootstrap=2000)]
