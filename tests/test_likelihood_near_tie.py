"""Amendment: 'Likelihood-scorer validation: near-tie ruling', fixed 1 Nov.

These tests change only technical validation acceptance. Scores and choices use
frozen character normalisation; diagnostic gaps use pinned LL/token lengths.
"""
import copy
import json
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kit import m2_format as f,likelihood_validation as v


def population(monkeypatch,a=.0009,b=.0009,ratio=1):
    ids=['fixture-'+str(i) for i in range(100)]
    monkeypatch.setattr(v,'lengths',lambda:{i:dict(characters=[ratio]*4,tokens=[1]*4) for i in ids})
    p=[dict(id=i,scores=[0,-a,-2,-3],selected=0) for i in ids]
    r=[dict(id=i,scores=[-b,0,-2,-3],selected=1) for i in ids]
    return ids,p,r


@pytest.mark.parametrize('a,b,expected',[(.000999,.000999,'PASS'),(.001,.0001,'PENDING'),(.0001,.001,'PENDING'),(.002,.002,'PENDING'),(0,0,'PASS')])
def test_strict_both_scorer_gap_rule_and_per_item_receipt(monkeypatch,a,b,expected):
    ids,p,r=population(monkeypatch,a,b)
    before=copy.deepcopy((p,r));out=f.validation(p,r,ids)
    assert out['status']==expected and len(out['per_item_gaps'])==100
    assert out['gap_unit']=='nats/token' and out['strict_gap_threshold']==.001
    assert out['selected_choice_agreement']==(100 if a==b==0 else 0)
    assert all(g['production_gap']==a and g['reference_gap']==b for g in out['per_item_gaps'])
    assert (p,r)==before


def test_token_gap_conversion_does_not_apply_threshold_in_character_units(monkeypatch):
    ids,p,r=population(monkeypatch,.0004,.0004,ratio=4)
    out=f.validation(p,r,ids)
    assert out['status']=='PENDING'
    assert all(g['production_gap']==.0016 and g['reference_gap']==.0016 for g in out['per_item_gaps'])
    assert p[0]['scores']==[0,-.0004,-2,-3] and p[0]['selected']==0


def test_unknown_length_or_missing_nonfinite_item_cannot_enable_disagreement(monkeypatch):
    ids,p,r=population(monkeypatch)
    monkeypatch.setattr(v,'lengths',lambda:{})
    assert f.validation(p,r,ids)['status']=='PENDING'
    assert f.validation(p,p,ids)['status']=='PASS'  # Unanimous rule unchanged.
    assert f.validation(p[:-1],r,ids)['status']=='PENDING'
    p[0]['scores'][0]=float('nan')
    assert not f.validation(p,r,ids)['finite_likelihoods']


def test_pinned_real_lengths_and_original_handoff_receipt():
    lengths=v.lengths()
    assert len(lengths)==200
    assert all(len(c['characters'])==len(c['tokens'])==4 and all(n>0 for n in c['characters']+c['tokens']) for c in lengths.values())
    assert f.policy()['hellaswag']['validation_items_sha256']
    from kit.hellaswag_prompt_check import validate_receipt
    assert validate_receipt(ROOT/'kit/resources/m2/receipts/hellaswag-format-check.attempt2.json')['record_sha256']==v.amendment()['historical_unanimous_receipt_sha256']


def validate_fixture(tmp_path,monkeypatch,a=.0009,b=.0009):
    from kit import m3_score as m,m3_train
    ids,p,r=population(monkeypatch,a,b)
    proof=dict(family='m3',seed=101)
    checkpoint=dict(model='fixture',identity=dict(model_file_hashes={'config.json':'a'*64}))
    monkeypatch.setattr(m,'proof_for',lambda *args:proof)
    monkeypatch.setattr(m,'resolved_gpu',lambda *args:'0')
    monkeypatch.setattr(m,'validation_checkpoints',lambda *args:{'base':checkpoint,'C0-s101-finqa-k40':checkpoint})
    monkeypatch.setattr(m3_train,'base_record',lambda *args:checkpoint)
    monkeypatch.setattr(m,'checked',lambda *args:None)
    monkeypatch.setattr(m,'gpu_phase',lambda *args,**kw:nullcontext())
    calls=[]
    def production(*args):calls.append('production');return p
    def reference(*args):calls.append('reference');return r
    args=NS(out=tmp_path,gpu='0',seed=101,standin=False)
    def run():m.validate(args,'knowledge-ll',[dict(id=i) for i in ids],production,reference,m.s.file_hash(m.KNOWLEDGE_POLICY),{})
    return m,run,calls


@pytest.mark.parametrize('a,b,expected',[(.0009,.0009,'PASS'),(.0009,.001,'PENDING')])
def test_m3_wrapper_records_gaps_and_terminal_pending_never_retries(tmp_path,monkeypatch,a,b,expected):
    m,run,calls=validate_fixture(tmp_path,monkeypatch,a,b)
    if expected=='PASS':run()
    else:
        with pytest.raises(ValueError,match='PENDING; rerun forbidden'):run()
    path=m.validation_path(tmp_path,'knowledge-ll','m3',101)
    value=json.loads(path.read_text())
    assert value['status']==expected and all(len(c['per_item_gaps'])==100 for c in value['checks'])
    before=path.read_bytes();count=len(calls)
    if expected=='PASS':run()
    else:
        with pytest.raises(ValueError,match='rerun forbidden'):run()
    assert len(calls)==count and path.read_bytes()==before


def test_legacy_near_tie_is_rederived_on_cpu_preserving_original_receipt(tmp_path,monkeypatch):
    m,run,calls=validate_fixture(tmp_path,monkeypatch)
    run();path=m.validation_path(tmp_path,'knowledge-ll','m3',101)
    value=json.loads(path.read_text());value.pop('validation_rule');value['status']='FAIL'
    for c in value['checks']:
        for key in set(c)-set(v.LEGACY_FIELDS)-{'production','reference','checkpoint','status'}:c.pop(key)
        c['status']='FAIL'
    path.write_text(json.dumps(value));original=path.read_bytes();count=len(calls)
    run()
    assert len(calls)==count and json.loads(path.read_text())['status']=='PASS'
    assert any(p.read_bytes()==original for p in (path.parent/'attempts'/path.stem).glob('*.json'))


def test_legacy_larger_gap_stays_pending_without_new_gpu_calls(tmp_path,monkeypatch):
    m,run,calls=validate_fixture(tmp_path,monkeypatch,.0009,.001)
    with pytest.raises(ValueError):run()
    path=m.validation_path(tmp_path,'knowledge-ll','m3',101)
    value=json.loads(path.read_text());value.pop('validation_rule');value['status']='FAIL'
    for c in value['checks']:
        for key in set(c)-set(v.LEGACY_FIELDS)-{'production','reference','checkpoint','status'}:c.pop(key)
        c['status']='FAIL'
    path.write_text(json.dumps(value));original=path.read_bytes();count=len(calls)
    with pytest.raises(ValueError,match='legacy reference disagreement PENDING'):run()
    assert len(calls)==count and json.loads(path.read_text())['status']=='PENDING'
    assert any(p.read_bytes()==original for p in (path.parent/'attempts'/path.stem).glob('*.json'))


def test_disagreeing_pair_is_recorded_even_when_token_top_two_differ(monkeypatch):
    ids,p,r=population(monkeypatch)
    out=f.validation(p,r,ids)
    assert all(g['disagreeing_pair']==[0,1] and g['production_choice']==0 and g['reference_choice']==1 for g in out['per_item_gaps'])
    unanimous=f.validation(p,p,ids)
    assert all(g['disagreeing_pair'] is None and g['production_choice']==g['reference_choice']==0 for g in unanimous['per_item_gaps'])


@pytest.mark.parametrize('pending',[False,True])
def test_fresh_core_validation_second_call_is_byte_identical(tmp_path,monkeypatch,pending):
    """V5 review finding 3: current-rule receipts are immutable on a second call."""
    ids,p,r=population(monkeypatch)
    rows=[dict(id=i) for i in ids];proof=dict(family='core',seed=101)
    monkeypatch.setattr(f,'binding',lambda *args:dict(starts=dict(base=dict(model='fixture'))))
    monkeypatch.setattr(f.assignment,'active',lambda *args:proof)
    monkeypatch.setattr(f,'items',lambda *args:rows)
    monkeypatch.setattr(f,'checked_model',lambda *args,**kw:None)
    from kit import hellaswag_reference
    monkeypatch.setattr(f,'likelihoods',lambda *args:p)
    if pending:r[0]['scores'][0]=None
    else:r=p
    monkeypatch.setattr(hellaswag_reference,'score',lambda *args:r)
    args=NS(out=tmp_path,standin=False,gpu='0',seed=101)
    f.job(args);path=f.validation_name(tmp_path,proof);original=path.read_bytes()
    assert json.loads(original)['validation_rule']==v.RULE
    assert json.loads(original)['status']==('PENDING' if pending else 'PASS')
    monkeypatch.setattr(f,'likelihoods',lambda *args:pytest.fail('must not score again'))
    f.job(args)
    assert path.read_bytes()==original and not (path.parent/'attempts').exists()


def test_core_current_rule_diagnostics_cannot_be_rewritten(tmp_path,monkeypatch):
    ids,p,r=population(monkeypatch)
    proof=dict(family='core',seed=101);rows=[dict(id=i) for i in ids]
    refs=dict(base=dict(model='fixture'))
    monkeypatch.setattr(f,'binding',lambda *args:dict(starts=refs))
    monkeypatch.setattr(f.assignment,'active',lambda *args:proof)
    monkeypatch.setattr(f,'items',lambda *args:rows)
    path=f.validation_name(tmp_path,proof)
    receipt=dict(policy_sha256=f.s.file_hash(f.POLICY),assignment=proof,checkpoints=refs,items_sha256=f.s.digest(rows),validation_rule='unknown-rule',status='PASS',checks=[dict(checkpoint='base',**f.validation(p,p,ids),production=p,reference=p)])
    f.atomic(path,receipt);before=path.read_bytes()
    with pytest.raises(ValueError,match='stored technical validation receipt differs'):
        f.job(NS(out=tmp_path,standin=False,gpu='0',seed=101))
    assert path.read_bytes()==before and not (path.parent/'attempts').exists()


def test_kcr_hellaswag_missing_lengths_is_configuration_error_before_gpu(tmp_path,monkeypatch):
    from kit import m3_score as m
    monkeypatch.setattr(v,'lengths',lambda:{})
    monkeypatch.setattr(m,'proof_for',lambda *args:pytest.fail('configuration error must precede GPU checks'))
    with pytest.raises(ValueError,match='configuration error: HellaSwag panel IDs missing'):
        m.validate(NS(out=tmp_path),'hellaswag',[dict(id='missing')],None,None,'hash',{})
    assert not (tmp_path/'format-validation').exists()


def test_legacy_fail_acceptance_is_explicit_in_reader_notes(tmp_path,monkeypatch):
    """1 Nov near-tie ruling; review 5 requires explicit legacy re-evaluation notes."""
    from kit import m2_assignment
    m,run,calls=validate_fixture(tmp_path,monkeypatch)
    run();path=m.validation_path(tmp_path,'knowledge-ll','m3',101)
    value=json.loads(path.read_text());value.pop('validation_rule');value['status']='FAIL'
    for c in value['checks']:
        for key in set(c)-set(v.LEGACY_FIELDS)-{'production','reference','checkpoint','status'}:c.pop(key)
        c['status']='FAIL'
    path.write_text(json.dumps(value));before=path.read_bytes()
    assignment=tmp_path/'assignment.json';assignment.write_text('{}')
    monkeypatch.setattr(m,'assignment_path',lambda *args:assignment)
    monkeypatch.setattr(m2_assignment,'verify',lambda *args:None)
    monkeypatch.setattr(m,'knowledge_items',lambda:[dict(id='fixture-'+str(i)) for i in range(100)])
    notes=[]
    assert m.validation_status(tmp_path,'knowledge-ll',101,notes=notes)==m.s.file_hash(path)
    assert len(notes)==1 and 'Legacy FAIL' in notes[0] and 'stored receipt unchanged' in notes[0]
    assert path.read_bytes()==before


def test_legacy_notice_is_in_reader_json_and_markdown(tmp_path,monkeypatch):
    from kit import m3_read,m3_manifest,m3_pools
    note='Legacy FAIL receipt re-evaluated PASS under '+v.RULE+'; stored receipt unchanged; gaps recomputed on CPU'
    monkeypatch.setattr(m3_manifest,'require_m3_root',lambda *args:None)
    monkeypatch.setattr(m3_read,'load_cells',lambda *args,**kw:({'fixture':dict(validation_notes=[note])},{}))
    monkeypatch.setattr(m3_read,'analyse',lambda *args:dict(decisions={'K':'PASS','CR':'PASS'},contrasts={}))
    monkeypatch.setattr(m3_pools,'pool_root',lambda out,arm:tmp_path/arm)
    m3_read.read_job(NS(out=tmp_path,standin=False))
    assert json.loads((tmp_path/'m3-read.json').read_text())['validation_notes']==[note]
    assert note in (tmp_path/'m3-read.md').read_text()
