import copy
from pathlib import Path
import pytest
from kit.nm_common import *
from kit.nm_bind import row_binding,resolve


def ref(tmp_path,name,value):
    p=tmp_path/(name+'.json');atomic(p,value);return dict(path=str(p),sha256=sha(p))


def recipe(name,finqa_slots=0,**kw):
    return dict(name=name,base_revision=BASE,starts={str(s):dict(seed=s,stage='finqa',step=40,sha256='a'*64) for s in SEEDS},
        updates=40,denominator=32,task_rows=28 if name=='K' else 32-finqa_slots if name=='NOMINATED' else 32,replay_rows=4 if name=='K' else 16 if name=='CR' else finqa_slots if name=='NOMINATED' else 0,
        finqa_slots=8 if name=='CR' else finqa_slots,schedule_hashes={str(s):'a' for s in SEEDS},caps=dict(prompt=10240,teacher=14336,response=2048),pool_ids=['bound-pool'] if name in ('K','CR','NOMINATED') else [],draw_policy='unfiltered' if name in ('K','CR') else 'inherited-core',p2_lambda=None,**kw)


def spec(tmp_path,k='PASS',cr='INCONCLUSIVE'):
    return dict(evidence={name:ref(tmp_path,name,status) for name,status in [('K',k),('CR',cr)]},
        recipes={name:ref(tmp_path,'recipe-'+name,recipe(name)) for name in ('K','CR','I0')})


def test_k_precedence_no_v6b_wait_but_cr_disposition_required(tmp_path):
    s=spec(tmp_path,cr='PASS');r=resolve(s)
    assert r['selected']=='K' and ['P2','INAPPLICABLE'] in r['trace']
    s['evidence'].pop('CR')
    with pytest.raises(Pending):resolve(s)


def test_cr_branch_omits_p2_and_core_stacking(tmp_path):
    s=spec(tmp_path,k='INCONCLUSIVE',cr='PASS');s['required_stacking']=['missing']
    assert row_binding('freeze',s)['incumbent']=='CR'
    with pytest.raises(Pending):row_binding('fisher',s)
    s['evidence']['P2']=ref(tmp_path,'P2','PASS')
    assert row_binding('fisher',s)['benchmark']


def test_i0_waits_all_applicable_verdicts_and_compatible_stack(tmp_path):
    s=spec(tmp_path,k='INCONCLUSIVE')
    with pytest.raises(Pending):resolve(s)
    for name in ('M8','M9','X','P2'):s['evidence'][name]=ref(tmp_path,name,'INCONCLUSIVE')
    assert resolve(s)['selected']=='I0'
    s['recipes']['I0']=ref(tmp_path,'illegal',recipe('I0',admitted_components=['M8']))
    with pytest.raises(Invalid):resolve(s)


def test_pending_hash_wrong_verdict_cannot_select_after_outcomes(tmp_path):
    s=spec(tmp_path);s['evidence']['K']=ref(tmp_path,'K2','PENDING')
    with pytest.raises(Pending):resolve(s)
    s['evidence']['K']['sha256']='0'*64
    with pytest.raises(Invalid):resolve(s)


def test_opr_cr_benchmark_before_nominated_core(tmp_path):
    s=spec(tmp_path,cr='PASS');b=row_binding('opr',s)
    assert b['recipe']['name']=='CR' and b['benchmark']
    s=spec(tmp_path);s['evidence']['nomination']=ref(tmp_path,'nom','INCONCLUSIVE')
    assert row_binding('opr',s)['status']=='ARCHIVED UNREAD'
    s['evidence']['nomination']=ref(tmp_path,'nom2','PASS');s['recipes']['nominated']=ref(tmp_path,'nominated',recipe('NOMINATED',finqa_slots=8))
    assert row_binding('opr',s)['recipe']['name']=='NOMINATED'
    assert row_binding('opr',s)['benchmark']


def test_persistence_gate_strict_negative_three_and_same_recipe(tmp_path):
    s=spec(tmp_path);rec=resolve(s)['recipe']
    s['persistence']=ref(tmp_path,'persist',dict(recipe_sha256=digest(rec),panel='instruction',step=40,retention=[-3,-3,-3]))
    assert row_binding('isdft',s)['status']=='ARCHIVED UNREAD'
    s['persistence']=ref(tmp_path,'persist2',dict(recipe_sha256=digest(rec),panel='instruction',step=40,retention=[-4,-4,-4]))
    assert row_binding('isdft',s)['status']=='READY'
    assert row_binding('grpo',{})['incumbent']=='INAPPLICABLE'


def test_existing_reader_boolean_requires_complete_evidence(tmp_path):
    from kit.nm_bind import verdict
    r=ref(tmp_path,'x-reader',dict(admitted=dict(p=False),status=dict(p='PENDING')))
    r.update(selector=['admitted','p'],complete_selector=['status','p'])
    with pytest.raises(Pending):verdict(r,'P2')
    r=ref(tmp_path,'x-reader2',dict(admitted=dict(p=False),status=dict(p='COMPLETE')))
    r.update(selector=['admitted','p'],complete_selector=['status','p'])
    assert verdict(r,'P2') is False
    r.pop('complete_selector')
    with pytest.raises(Pending):verdict(r,'P2')


def test_staggered_family_launch_does_not_freeze_future_verdicts():
    cfg=dict(out='/durable/nm',infrastructure_update='a'*64,inputs=dict(grpo={'path':'grpo','sha256':'x'}),budget=dict(branch='CR',row_max_seconds={'NM-grpo-a':10}))
    before=digest(launch_config(cfg,'grpo'))
    cfg['inputs']['fisher']={'path':'later-p2-return','sha256':'y'}
    cfg['budget']['row_max_seconds']['NM-fisher-a']=100
    assert digest(launch_config(cfg,'grpo'))==before
    cfg['inputs']['grpo']['sha256']='changed'
    assert digest(launch_config(cfg,'grpo'))!=before


def test_selector_cannot_hide_invalid_P2_evidence_as_non_admission(tmp_path):
    from kit.nm_bind import verdict
    r=ref(tmp_path,'invalid-p2',dict(comparisons=dict(p=dict(status='INCONCLUSIVE',evidence_error='calibration identity differs'))))
    r['selector']=['comparisons','p','status']
    with pytest.raises(Pending):verdict(r,'v6b P2')


def test_missing_and_wrong_recipe_persistence_stay_pending(tmp_path):
    s=spec(tmp_path);frozen=tmp_path/'out'/'incumbent.json'
    with pytest.raises(Pending,match='persistence evidence absent'):row_binding('isdft',s,incumbent_path=frozen)
    assert frozen.is_file()  # I freezes even when this row's extra prerequisite is pending.
    s['persistence']=ref(tmp_path,'wrong-persistence',dict(recipe_sha256='0'*64,panel='instruction',step=40,retention=[-4]*3))
    with pytest.raises(Pending,match='persistence wrong recipe'):row_binding('isdft',s,incumbent_path=frozen)


def test_dispatch_seal_refuses_changed_evidence_or_recipe_at_second_bind(tmp_path):
    s=spec(tmp_path);frozen=tmp_path/'incumbent.json'
    row_binding('freeze',s,incumbent_path=frozen);before=load(frozen)
    later=copy.deepcopy(s);later['evidence']['P2']=ref(tmp_path,'later-P2','PASS')
    assert row_binding('fisher',later,incumbent_path=frozen)['incumbent']=='K'
    assert load(frozen)['identity']==before['identity']
    changed=copy.deepcopy(later);changed['evidence']['K']=ref(tmp_path,'changed-K',dict(status='PASS',different_evidence=True))
    with pytest.raises(Invalid,match='incumbent/evidence changed'):row_binding('fisher',changed,incumbent_path=frozen)
    changed=copy.deepcopy(later);r=recipe('K');r['schedule_hashes']['101']='changed'
    changed['recipes']['K']=ref(tmp_path,'changed-recipe',r)
    with pytest.raises(Invalid,match='incumbent/evidence changed'):row_binding('fisher',changed,incumbent_path=frozen)
    changed=copy.deepcopy(later);changed['evidence']['P2']=ref(tmp_path,'revised-P2','INCONCLUSIVE')
    with pytest.raises(Invalid,match='later evidence changed'):row_binding('fisher',changed,incumbent_path=frozen)
    # An unrelated row need not carry Fisher's later P2 file under K.
    assert row_binding('freeze',s,incumbent_path=frozen)['incumbent']=='K'


def test_real_bind_operation_uses_dispatch_seal(tmp_path):
    from kit.nm_pipeline import operation
    from kit.nm_jobs import row_for
    out=tmp_path/'out';s=spec(tmp_path)
    b=ref(tmp_path,'bundle',dict(binding_spec=ref(tmp_path,'spec',s)))
    cfg=dict(inputs=dict(freeze=b),infrastructure_update='a'*64)
    assert operation(cfg,row_for('NM-freeze-bind'),tmp_path,out,tmp_path/'config')['status']=='READY'
    s['evidence']['CR']=ref(tmp_path,'different-CR','PASS')
    cfg['inputs']['opr']=ref(tmp_path,'opr-bundle',dict(binding_spec=ref(tmp_path,'opr-spec',s)))
    with pytest.raises(Invalid,match='incumbent/evidence changed'):
        operation(cfg,row_for('NM-opr-bind'),tmp_path,out,tmp_path/'config')


def test_pending_later_verdict_does_not_seal_an_unresolved_hash(tmp_path):
    s=spec(tmp_path);frozen=tmp_path/'incumbent.json'
    row_binding('freeze',s,incumbent_path=frozen)
    s['evidence']['P2']=ref(tmp_path,'p2-pending','PENDING')
    with pytest.raises(Pending):row_binding('fisher',s,incumbent_path=frozen)
    assert not load(frozen)['later_evidence']
    s['evidence']['P2']=ref(tmp_path,'p2-resolved','PASS')
    assert row_binding('fisher',s,incumbent_path=frozen)['status']=='READY'
    assert load(frozen)['later_evidence']['P2']==s['evidence']['P2']
