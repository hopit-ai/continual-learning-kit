import pytest
from kit.nm_common import *
from kit import nm_data as n


def pool(size=5):
    return [dict(id=str(i),prompt='Question '+str(i),response='Answer: 7',gold='7',stored_verified=True,target_sha256=raw_hash(b'Answer: 7')) for i in range(size)]


def outputs(rows,success):
    return {r['id']:{str(a):dict(response='Answer: 7' if int(r['id'])<success else 'Answer: 8',output_tokens=4,finish_reason='stop',rng=n.generation_seed(101,r['id'],a)) for a in range(1,5)} for r in rows}


def test_opr_first_success_fixed_fallback_and_exact_exposures():
    rows=pool();generated=outputs(rows,4);generated['0']['1']['response']='Answer: 8'
    refreshed,audit=n.refresh(rows+rows[:2],generated,101)
    assert audit['coverage']==.8 and audit['exposure_substitution']==pytest.approx(6/7)
    assert refreshed[-3]['response']==rows[-1]['response'] and refreshed[-3]['opr_generated'] is False
    assert [r['id'] for r in refreshed]==[r['id'] for r in rows+rows[:2]]
    assert audit['attempts'][0]['accepted'] is False and audit['attempts'][1]['accepted'] is True


def test_opr_below_80_missing_attempt_and_truncated_correct_all_pending():
    rows=pool()
    with pytest.raises(Pending):n.refresh(rows,outputs(rows,3),101)
    value=outputs(rows,5);value['0']['1']['finish_reason']='length'
    for a in ('2','3','4'):value['0'][a]['finish_reason']='length'
    assert n.refresh(rows,value,101)[1]['coverage']==.8
    value['1'].pop('1')
    with pytest.raises(Pending):n.refresh(rows,value,101)


def test_exact_99_percent_vector_boundary_and_cohort_support():
    base={str(i):int(i<100) for i in range(1147)};incoming={str(i):int(i<300) for i in range(1147)}
    changed=dict(base)
    for i in range(11):changed[str(i)]^=1
    members,diag=n.final_cohort(base,incoming,changed,incoming)
    assert len(members)==211 and len(diag['base']['disagreements'])==11
    changed['11']^=1
    with pytest.raises(Invalid):n.final_cohort(base,incoming,changed,incoming)


def test_grpo_320_unique_and_1280_trajectories_no_rejection():
    rows=[dict(id=str(i),split='train',demonstration='verified target') for i in range(330)]
    schedule=n.grpo_schedule(rows,101)
    assert len(schedule)==1280 and len({r['id'] for r in schedule})==320
    assert all(len({r['id'] for r in schedule[i:i+4]})==1 for i in range(0,1280,4))
    assert {r['update'] for r in schedule}==set(range(1,41))
    with pytest.raises(Pending):n.grpo_schedule(rows[:319],101)


def test_fisher_fixed_training_quotas_and_no_panel_substitution():
    domains={d:[dict(id=str(i),split='train',verified_base_response=True,base_revision=BASE) for i in range(n)] for d,n in [('instruction',128),('maths',64),('knowledge',64)]}
    assert len(n.fisher_rows(domains))==256
    domains['maths'][0]['split']='test'
    with pytest.raises(Invalid):n.fisher_rows(domains)


def test_candidate_superset_exclusion_and_seed_cycling():
    with pytest.raises(Invalid):n.audit_candidates([dict(id='a',text='Same text, punctuation!')],[dict(id='b',text='same text punctuation')])
    ledger=[dict(id=str(i),month='2026-07',item=dict(sec=dict(issuer_holdout=False))) for i in range(3)]
    rows,audit=n.v2b_schedule(ledger,101)
    assert len(rows)==1280 and audit['unique']==3 and sum(audit['repetitions'].values())==1280
    ledger[0]['month']='2026-10'
    with pytest.raises(Invalid):n.v2b_schedule(ledger,101)
