import copy
import pytest
from kit.nm_common import *
from kit.nm_sim import fixture
from kit.nm_read import read


@pytest.mark.parametrize('family',FAMILIES)
def test_every_row_endpoint_valid_technical_only(family):
    doc,spec,binding=fixture(family);result=read(family,doc,spec,binding)
    assert result['status']=='TECHNICAL-ONLY',result
    assert result['simulated_status'] in ('PASS','QUALIFIED')
    assert result['primary']['df']==2
    assert all(len(line)<=90 for line in text_lines(family,result,'a'*64))


@pytest.mark.parametrize('identity',['nonce','uuid','environment'])
def test_mixed_hardware_is_labelled_not_admission(identity):
    doc,spec,binding=fixture('fisher')
    target=next(c for c in doc['cells'] if c['seed']==101 and c['panel']=='instruction' and c['model']=='T' and c['step']==40)
    target[identity]='different'
    assert read('fisher',doc,spec,binding)['status']=='UNVERIFIED-HARDWARE'


def test_missing_required_curve_or_reference_is_pending():
    doc,spec,binding=fixture('freeze');doc['cells']=[c for c in doc['cells'] if not(c['seed']==102 and c['panel']=='legal' and c['model']=='T' and c['step']==10)]
    assert read('freeze',doc,spec,binding)['status']=='PENDING/INCOMPLETE'


def test_no_two_seed_fallback_or_reference_method_downgrade():
    doc,spec,binding=fixture('isdft');doc['cells']=[c for c in doc['cells'] if c['seed']!=103]
    assert read('isdft',doc,spec,binding)['status']=='PENDING/INCOMPLETE'
    doc,spec,binding=fixture('isdft')
    next(c for c in doc['cells'] if c['panel']=='knowledge')['precision']='quantised'
    assert read('isdft',doc,spec,binding)['status']=='UNVERIFIED'


def test_complete_valid_missed_bar_is_inconclusive():
    doc,spec,binding=fixture('fisher')
    for c in doc['cells']:
        if c['model']=='T':
            counterpart=next(x for x in doc['cells'] if (x['seed'],x['panel'],x['model'],x['step'])==(c['seed'],c['panel'],'C',c['step']))
            c['items']=counterpart['items'].copy()
    r=read('fisher',doc,spec,binding)
    assert r['status']=='TECHNICAL-ONLY' and r['simulated_status']=='INCONCLUSIVE'


def test_grpo_guard_uses_one_sided_negative_three_not_new_rung_bar():
    doc,spec,binding=fixture('grpo');r=read('grpo',doc,spec,binding)
    assert r['guards']['instruction']['t']==ONE and r['guards']['instruction']['margin']==-3
    assert r['primary']['t']==TWO and r['primary']['margin']==-1
    assert r['guards']['acquisition']['positive']==3 and not r['incumbent_changed']


def test_grpo_insensitive_D_is_pending_not_challenger_failure():
    doc,spec,binding=fixture('grpo')
    for c in doc['cells']:
        if c['panel']=='finqa' and c['model']=='C' and c['step']==40:
            c['items']=next(x['items'].copy() for x in doc['cells'] if x['panel']=='finqa' and x['model']=='base' and x['seed']==c['seed'])
    assert read('grpo',doc,spec,binding)['status']=='PENDING/INCOMPLETE'


def test_v2b_september_guard_all_52_and_prior_months_descriptive():
    doc,spec,binding=fixture('v2b')
    for c in doc['cells']:
        if c['model']=='T' and c['panel'].startswith(('july','august')):c['items']={i:0 for i in c['items']}
    r=read('v2b',doc,spec,binding)
    assert r['simulated_status']=='PASS' and len(r['guards'])==5
    assert r['diagnostics']['memory_on_guard_N']==52 and r['diagnostics']['primary_N']==13
    spec['panels']['september-on']=spec['panels']['september-on'][:13]
    assert read('v2b',doc,spec,binding)['status']=='UNVERIFIED'


def test_opr_final_cohort_arm_independent_and_agreement_per_vector():
    doc,spec,binding=fixture('opr')
    before=read('opr',doc,spec,binding)['diagnostics']['cohorts']['101']['membership_sha256']
    for c in doc['cells']:
        if c['model']=='T' and c['panel']=='finqa':c['items']={i:0 for i in c['items']}
    after=read('opr',doc,spec,binding)
    assert after['diagnostics']['cohorts']['101']['membership_sha256']==before
    for ident in list(spec['pre_cohort']['101']['base'])[:12]:spec['pre_cohort']['101']['base'][ident]^=1
    assert read('opr',doc,spec,binding)['status']=='UNVERIFIED'


def test_opr_final_reference_cohort_insufficient_is_pending():
    doc,spec,binding=fixture('opr')
    for c in doc['cells']:
        if c['model']=='incoming' and c['panel']=='finqa':c['items']={i:0 for i in c['items']}
    assert read('opr',doc,spec,binding)['status']=='PENDING/INCOMPLETE'


def test_wrong_checkpoint_and_panel_identity_never_dropped():
    doc,spec,binding=fixture('fisher');doc['cells'][0]['checkpoint_sha256']='wrong'
    assert read('fisher',doc,spec,binding)['status']=='UNVERIFIED'
    doc,spec,binding=fixture('fisher');doc['cells'][0]['items'].pop(next(iter(doc['cells'][0]['items'])))
    assert read('fisher',doc,spec,binding)['status']=='UNVERIFIED'


def test_fisher_and_opr_benchmark_cannot_admit():
    for family in ('fisher','opr'):
        doc,spec,binding=fixture(family);binding['benchmark']=True
        r=read(family,doc,spec,binding)
        assert r['simulated_status']=='NON-INCUMBENT BENCHMARK' and not r['incumbent_changed']
        assert 'NON-INCUMBENT BENCHMARK' in r['interpretation']


@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('key',['panels','checkpoints','scorers'])
def test_missing_spec_keys_are_pending(family,key):
    doc,spec,binding=fixture(family);spec.pop(key)
    assert read(family,doc,spec,binding)['status']=='PENDING/INCOMPLETE'


def test_grpo_descriptive_general_average_and_insensitive_D_details():
    doc,spec,binding=fixture('grpo');r=read('grpo',doc,spec,binding)
    assert r['diagnostics']['general_retention']==stats([10.,10.,10.])
    for c in doc['cells']:
        if c['panel']=='finqa' and c['model']=='C' and c['step']==40:
            c['items']=next(x['items'].copy() for x in doc['cells'] if x['panel']=='finqa' and x['model']=='base' and x['seed']==c['seed'])
    r=read('grpo',doc,spec,binding)
    assert r['status']=='PENDING/INCOMPLETE' and r['diagnostics']['D_sensitivity']['positive']==0
    assert r['primary'] and r['guards']


def test_synthetic_vectors_cannot_be_relabelled_scientific():
    doc,spec,binding=fixture('freeze');doc['technical_synthetic']=False
    assert read('freeze',doc,spec,binding)['status']=='UNVERIFIED'


def test_archived_cli_never_opens_outcome_or_spec_files(tmp_path):
    import subprocess
    import sys
    binding=tmp_path/'binding.json';atomic(binding,dict(family='opr',status='ARCHIVED UNREAD',reason='no nomination'))
    result=subprocess.run([sys.executable,'-I',str(KIT/'nm_read.py'),'--family','opr','--binding',str(binding),
        '--scores',str(tmp_path/'do-not-open-scores.json'),'--spec',str(tmp_path/'do-not-open-spec.json'),'--out',str(tmp_path/'reading.json')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert load(tmp_path/'reading.json')['status']=='ARCHIVED UNREAD'


@pytest.mark.parametrize('computed',[False,True])
def test_grpo_pending_cli_exit_depends_on_missing_evidence(tmp_path,computed):
    import subprocess
    import sys
    doc,spec,binding=fixture('grpo')
    for c in doc['cells']:
        if c['panel']=='finqa' and c['model']=='C' and c['step']==40:
            c['items']=next(x['items'].copy() for x in doc['cells'] if x['panel']=='finqa' and x['model']=='base' and x['seed']==c['seed'])
    if not computed:doc['cells']=[]
    for name,value in [('scores',doc),('spec',spec),('binding',binding)]:atomic(tmp_path/(name+'.json'),value)
    result=subprocess.run([sys.executable,'-I',str(KIT/'nm_read.py'),'--family','grpo','--scores',str(tmp_path/'scores.json'),
        '--spec',str(tmp_path/'spec.json'),'--binding',str(tmp_path/'binding.json'),'--out',str(tmp_path/'reading.json')],capture_output=True,text=True)
    assert result.returncode==(0 if computed else 75),result.stdout+result.stderr
    reading=load(tmp_path/'reading.json')
    assert reading['status']=='PENDING/INCOMPLETE' and bool(reading.get('primary'))==computed
