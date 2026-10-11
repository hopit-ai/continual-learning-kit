"""Public boundary guards need no private input, GPU, Slurm or model package."""
import copy
from fractions import Fraction
import pytest
from kit import v4_budget as budget,v4_phase0_carry as carry,v4_allocation as allocation


def ledger():
    records={'123':dict(start=1,end=6601,width=8,segments=[dict(block='qualification',start=1,end=6601)])}
    imported=dict(schema='v4-phase0-binding.v1',archive_sha256='a'*64,ledger_sha256='b'*64,
                  records_sha256=carry.digest(carry.canonical(records)),allocations=records)
    return dict(allocations={},phase0_import=imported)


def test_fresh_work_needs_bound_closed_return(tmp_path):
    with pytest.raises(ValueError,match='phase0'):allocation.plan(tmp_path,'qualification','qualification',750,750)


def test_14_67_hours_count_once_in_both_limits():
    doc=ledger();actual,blocks=budget.allocation_spend(doc)
    assert actual==blocks['qualification']==Fraction(44,3)
    assert 100-blocks['qualification']==Fraction(256,3)
    assert 955-actual==Fraction(2821,3)
    doc['allocations']=copy.deepcopy(doc['phase0_import']['allocations'])
    assert budget.allocation_spend(doc)[0]==actual


@pytest.mark.parametrize('case',['open','changed','conflict','invalid_binding'])
def test_unknown_or_conflicting_import_refuses(case):
    doc=ledger()
    if case=='invalid_binding':doc['phase0_import']=[]
    elif case=='conflict':
        doc['allocations']=copy.deepcopy(doc['phase0_import']['allocations']);doc['allocations']['123']['width']=7
    else:doc['phase0_import']['allocations']['123']['end']=None if case=='open' else 6602
    with pytest.raises(ValueError,match='phase0'):budget.allocation_spend(doc)


def test_main_source_verification_runs_after_activation_before_selftest(tmp_path):
    doc=dict(phase='qualification',block='qualification',block_limit=100,ceiling=955,stage='qualification',sbatch_time='01:00:00',site={'activate':'/staged/trainer/bin/activate'})
    text=allocation.script(doc,tmp_path,tmp_path/'kit')
    assert text.index('source /staged/trainer/bin/activate')<text.index('v4_phase0_carry.py')<text.index('p4_contain.py" selftest')
