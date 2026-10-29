"""Portable attempt-5 authority tests; CPU fixtures and an inert Modal SDK only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace as NS
import pytest
from kit import x_smoke_settings as settings,x_integration_smoke as smoke
from kit.s_common import ROOT,atomic,read


def ledger():return '| '+' | '.join(settings.ATTEMPT5_LEDGER_ROW)+' |'


@pytest.fixture
def authority(tmp_path,monkeypatch):
    """Local byte-bound forecast/audit fixtures, not paid qualification evidence."""
    forecast=read(ROOT/'kit/resources/p2/attempt4-forecast.json')
    basis=tmp_path/'basis.json';basis.write_text('{"fixture":true}\n')
    forecast['measured_basis']=dict(path='basis.json',sha256=hashlib.sha256(basis.read_bytes()).hexdigest())
    atomic(tmp_path/'kit/resources/p2/attempt4-forecast.json',forecast)
    atomic(tmp_path/'kit/resources/p2/split-forecast.json',{})
    test=tmp_path/'tests/test_x_smoke_stage_audit.py';test.parent.mkdir();test.write_text('# CPU fixture\n')
    monkeypatch.setattr(settings,'ROOT',tmp_path)
    monkeypatch.setattr(settings,'ATTEMPT5_STAGE_AUDIT',tmp_path/'audit.json')
    from kit.p2_smoke import code_hashes
    atomic(settings.ATTEMPT5_STAGE_AUDIT,dict(schema='p2-cpu-stage-audit.v1',passed=True,
        code_sha256=code_hashes(),test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
        stages={s:dict(batches=4,task_per_batch=28,replay_positions=[8,16,24,32])
                for s in ('preparation','measure','control','p2','prefix','resume')}))
    directory=tmp_path/'claims';stem=settings.PART_STEMS['p2']
    for n in range(1,5):atomic(directory/(stem+f'.attempt{n}.json'),dict(attempt=n,immutable=True))
    return directory


@pytest.mark.parametrize('attempt',range(1,6))
def test_resources_preserve_historical_limits_and_extend_only_attempt5(attempt):
    assert settings.AUTHORISED_ATTEMPTS==dict(p2=5,resume=1)
    assert settings.resources('p2',attempt)==dict(timeout_seconds=3000,total_ceiling_usd=14.17) if attempt in (4,5) else settings.resources('p2',attempt)==dict(timeout_seconds=1800,total_ceiling_usd=9.77)
    assert settings.resources('resume',1)==dict(timeout_seconds=1800,total_ceiling_usd=9.77)
    with pytest.raises(ValueError,match='unauthorised'):settings.resources('p2',6)
    with pytest.raises(ValueError,match='unauthorised'):settings.resources('resume',2)


def test_attempt5_reuses_complete_attempt4_forecast_and_keeps_45_minute_gate(authority):
    assert settings.forecast('p2',5)==settings.forecast('p2',4)
    forecast=settings.forecast('p2',5)
    assert forecast['forecast_limit_seconds']==2700
    assert forecast['launches']==4 and forecast['logical_actor_calls']==13 and forecast['fp32_exports']==3
    path=settings.ROOT/'kit/resources/p2/attempt4-forecast.json';value=read(path)
    value['parts']['p2']['seconds']['upper_scenario']=2700.001;atomic(path,value)
    with pytest.raises(ValueError,match='45 minutes'):smoke.reserve_attempt(authority,'p2',ledger=ledger())
    assert not list(authority.glob('*.claim.json'))


@pytest.mark.parametrize('change',[None,'mention','date','ceiling','owner','scope','status'])
def test_exact_owner_row_required_before_attempt5_claim(authority,change):
    row=list(settings.ATTEMPT5_LEDGER_ROW)
    if change=='date':row[0]='2026-10-28'
    if change=='ceiling':row[5]='≤$15.01'
    if change=='owner':row[4]='unrelated approval'
    if change=='scope':row[3]+='; retry allowed'
    if change=='status':row[6]='complete'
    text=None if change is None else 'x-integration-p2-smoke attempt 5' if change=='mention' else '| '+' | '.join(row)+' |'
    before={p.name:p.read_bytes() for p in authority.iterdir()}
    with pytest.raises(ValueError,match='ledger'):smoke.reserve_attempt(authority,'p2',ledger=text)
    assert {p.name:p.read_bytes() for p in authority.iterdir()}==before


def test_attempt5_claim_once_then_attempt6_refused_and_history_unchanged(authority):
    before={p.name:p.read_bytes() for p in authority.iterdir()}
    assert smoke.reserve_attempt(authority,'p2',ledger=ledger())==(5,settings.PART_STEMS['p2'])
    claim=read(authority/'x-integration-p2-smoke.attempt5.claim.json')
    assert claim==dict(attempt=5,part='p2',stem='x-integration-p2-smoke',status='RESERVED',total_attempt_cap=5)
    with pytest.raises(ValueError,match='exhausted'):smoke.reserve_attempt(authority,'p2',ledger=ledger())
    assert all((authority/name).read_bytes()==raw for name,raw in before.items())


@pytest.mark.parametrize('defect',['missing','code','test','stages'])
def test_fresh_complete_current_code_audit_required_before_claim(authority,defect):
    path=settings.ATTEMPT5_STAGE_AUDIT;value=read(path)
    if defect=='missing':path.unlink()
    else:
        if defect=='code':value['code_sha256']={}
        if defect=='test':value['test_sha256']='0'*64
        if defect=='stages':value['stages'].pop('resume')
        atomic(path,value)
    with pytest.raises(ValueError,match='audit|CPU'):smoke.reserve_attempt(authority,'p2',ledger=ledger())
    assert not list(authority.glob('*.claim.json'))


def receipt():
    from kit.p2_smoke import SCHEMA,ASSERTIONS,code_hashes
    child=dict(schema=SCHEMA,variant='p2-l2-init',placement='real-verl-fsdp2',world_size=2,
        model_revision=smoke.MODEL_REVISION,precision='fp32-masters/fp64-accumulation',technical_synthetic=False,
        passed=True,assertions={k:dict(status='PASS') for k in ASSERTIONS},code_sha256=code_hashes(),
        paired_batches=dict(schema='x-smoke-paired-batches.v1',smoke_only=True,control_first=True,
            updates=[dict(update=i,identity=True,sha256='a'*64,input_hash='b'*64) for i in range(4)]))
    return dict(schema='x-integration-smoke.v3',part='p2',attempt=5,maximum_attempts=5,retries=0,gpu='H100:2',
        evidence_stem='x-integration-p2-smoke',passed=True,p2_smoke=child,**settings.resources('p2',5))


def test_attempt5_receipt_keeps_all_assertions_and_current_code_map():
    from kit.x_p import validate_smoke
    value=receipt();assert validate_smoke(value,False)==value['p2_smoke']
    for field,bad in [('attempt',6),('maximum_attempts',4),('retries',1),('gpu','H100:1'),('timeout_seconds',1800),('total_ceiling_usd',15.01)]:
        with pytest.raises(ValueError,match='bounded split'):validate_smoke(dict(value,**{field:bad}),False)
    for field in ('code_sha256','paired_batches'):
        broken=copy.deepcopy(value);broken['p2_smoke'][field]={}
        with pytest.raises(ValueError):validate_smoke(broken,False)
    for assertion in value['p2_smoke']['assertions']:
        broken=copy.deepcopy(value);broken['p2_smoke']['assertions'][assertion]['status']='FAIL'
        with pytest.raises(ValueError,match='failed or incomplete'):validate_smoke(broken,False)


def test_real_launcher_reserves_attempt5_one_bounded_container_without_retry(authority,monkeypatch):
    """The SDK is inert; actual authority, resource, forecast and audit gates run."""
    root=settings.ROOT;stem=settings.PART_STEMS['p2'];evidence=root/'docs/phase2/evidence'
    evidence.parent.mkdir(parents=True);authority.rename(evidence)
    (root/'docs/phase2/ledger.md').write_text(ledger())
    calls=[];written=[]
    fake=ModuleType('modal');fake.App=lambda *args:NS(local_entrypoint=lambda:lambda fn:fn)
    def create(*args,**kwargs):
        calls.append((args,kwargs))
        return NS(stdout=['[record] '+json.dumps(dict(passed=False,steps=[]))+'\n'],wait=lambda:None,terminate=lambda:None)
    fake.Sandbox=NS(create=create)
    helper=ModuleType('smoke_common_modal');helper.ROOT=root
    helper.partner_image=lambda *args:NS(add_local_file=lambda *args,**kwargs:object())
    helper.write_record=lambda record,*args,**kwargs:written.append(record)
    monkeypatch.setitem(sys.modules,'modal',fake);monkeypatch.setitem(sys.modules,'smoke_common_modal',helper)
    monkeypatch.setattr(sys,'path',list(sys.path))
    spec=importlib.util.spec_from_file_location('attempt5_inert_launcher',ROOT/'scripts/smoke_x_integration_modal.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'check_probe',lambda path:dict(decision={}))
    with pytest.raises(SystemExit):module.main('p2')
    assert len(calls)==1
    args,config=calls[0]
    assert args[-2:]==('--part','p2') and config['gpu']=='H100:2' and config['timeout']==3000
    assert config['cpu']==(8,8) and config['memory']==(65536,65536)
    assert written[0]['attempt']==written[0]['maximum_attempts']==5
    assert written[0]['retries']==0 and written[0]['create_calls']==1 and written[0]['total_ceiling_usd']==14.17
    with pytest.raises(ValueError,match='exhausted'):module.main('p2')
    assert len(calls)==1
