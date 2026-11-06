import json
import re
import subprocess
import sys
import pytest
from kit.nm_common import *
from kit.nm_sim import fixture
from kit.nm_read import read
from kit.text_return import write_parts,parse_parts,body_text
from kit.nm_text import payload


def make(tmp_path,family,*,change=None,max_chars=12000,benchmark=False):
    doc,spec,binding=fixture(family);binding['benchmark']=benchmark
    if change:change(doc,spec,binding)
    r=read(family,doc,spec,binding)
    full=dict(nm_family=family,reading=r,infrastructure_authority='a'*64,
        accounting=dict(status='COMPLETE',attempts=9,gpu_hours=12.5,by_phase=dict(train=dict(attempts=9,gpu_hours=12.5,statuses=dict(COMPLETE=6,FAILED=3)))))
    paths=write_parts(tmp_path,full,'kit-test','b'*40,max_chars=max_chars)
    return r,paths,full


@pytest.mark.parametrize('family',FAMILIES)
def test_existing_report_transport_roundtrips_every_verdict_from_seed_values(tmp_path,family):
    r,paths,full=make(tmp_path,family)
    assert 1<=len(paths)<=4
    texts=[p.read_text() for p in paths]
    assert all(t.isascii() and len(t)<=12000 and all(len(line)<90 for line in t.splitlines()) for t in texts)
    imported=parse_parts(texts,expect_commit='b'*40)
    assert imported['status']==r['status'] and imported['simulated_status']==r['simulated_status']
    assert imported['primary']==r['primary'] and imported['guards']==r['guards']
    assert imported['package_summaries']=={}
    detailed=parse_parts([p.read_text() for p in (tmp_path/'full').glob('RETURN-TEXT-*.md')])
    assert detailed['package_summaries']==r['package_summaries']
    assert imported['endpoint_verdict_rederived'] and not imported['incumbent_changed']
    assert set(imported['part_checksums'].values())=={'MATCH'}
    assert imported['metadata']['account']['gpu_hours']==12.5 and imported['metadata']['account']['attempts']==9
    assert 'attempt' not in imported['metadata']
    assert detailed['metadata']['attempt']['train']['statuses']['FAILED']==3
    joined=''.join(texts)
    for identity in r['lanes'].values():
        assert all(v not in joined for v in identity)
    assert imported['metadata']['lane']
    if family=='opr':
        assert set(imported['metadata']['coverage'])==set(map(str,SEEDS))
        assert imported['metadata']['cohort']['101'][1]>=100
    if family=='fisher':assert imported['metadata']['fisher']['coefficients']==[.1,.05,1.,0.]
    if family=='isdft':assert imported['endpoints']['persistence']['values']==[-4.]*3
    if family=='grpo':assert 'general_retention' in imported['endpoints']


def test_cpu_cli_and_existing_reporter_accept_pasted_text_only(tmp_path):
    from kit.report_text import report
    r,_,full=make(tmp_path/'initial','freeze');atomic(tmp_path/'reading.json',r)
    atomic(tmp_path/'account.json',full['accounting'])
    paths=report(dict(_mode='nm',family='freeze',reading=str(tmp_path/'reading.json'),accounting=str(tmp_path/'account.json'),infrastructure_update='a'*64),
        tmp_path/'report',tag='kit-test',commit='b'*40,standin=True)
    out=tmp_path/'local.json'
    proc=subprocess.run([sys.executable,'-I',str(KIT/'nm_return.py'),'--text',*[str(p) for p in paths],'--out',str(out)],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert load(out)['simulated_status']=='PASS'


@pytest.mark.parametrize('checksum',['removed','damaged','wrong'])
def test_advisory_checksum_never_rejects_and_wrapping_is_restored(tmp_path,checksum):
    _,paths,_=make(tmp_path,'grpo');texts=[]
    for path in paths:
        text=path.read_text()
        if checksum=='removed':text=re.sub(r'SHA256 [^\n]+\n$','',text)
        if checksum=='damaged':text=re.sub(r'SHA256 [^\n]+\n$','SHA256 copy-error!\n',text)
        if checksum=='wrong':text=re.sub(r'SHA256 [^\n]+\n$','SHA256 '+'0'*64+'\n',text)
        # Real chat wrapping and quote prefixes, including split record names.
        text='> '+'\n> '.join(text[i:i+47] for i in range(0,len(text),47))
        texts.append(text)
    imported=parse_parts(texts)
    assert imported['simulated_status']=='QUALIFIED'
    assert set(imported['part_checksums'].values())=={'TRANSCRIPTION-UNVERIFIED'}


def test_importer_recomputes_guard_and_ignores_pasted_summary_and_decision(tmp_path):
    _,paths,_=make(tmp_path,'freeze')
    texts=[p.read_text() for p in paths]
    # Change a valid per-seed vector, preserving the framing/counts. The pasted
    # decision still says PASS and its old summary bounds still look positive.
    changed=[]
    for text in texts:
        flat=''.join(text.splitlines())
        flat=re.sub(r'(T\|NM\|count\|10[123]\.instruction\|)\[[^]]+\]',r'\g<1>[100,20,80,60,55]',flat)
        changed.append(flat)
    imported=parse_parts(changed)
    assert imported['simulated_status']=='INCONCLUSIVE' and not imported['reported_verdict_matches']
    assert imported['guards']['instruction']['lower']==-5
    assert 'TRANSCRIPTION-UNVERIFIED' in imported['part_checksums'].values()


def test_opr_reports_eleven_disagreements_and_all_coverage_fractions(tmp_path):
    def change(doc,spec,binding):
        # Change pre-training base only; final cohort is still arm-independent.
        for ident in list(spec['pre_cohort']['101']['base'])[:11]:spec['pre_cohort']['101']['base'][ident]^=1
    r,paths,_=make(tmp_path,'opr',change=change)
    imported=parse_parts([p.read_text() for p in paths])
    assert imported['simulated_status']==r['simulated_status']
    assert imported['metadata']['agreement']['101.base'][1]==11
    assert len([k for k in imported['metadata']['disagree'] if k.startswith('101.base.')])==11
    assert imported['metadata']['coverage']['101']==[5,4,.8,.2,.8,.2]


def test_insensitive_D_retains_inputs_and_rederives_pending(tmp_path):
    def change(doc,spec,binding):
        for c in doc['cells']:
            if c['model']=='C' and c['panel']=='finqa' and c['step']==40:
                c['items']=next(x['items'].copy() for x in doc['cells'] if x['seed']==c['seed'] and x['panel']=='finqa' and x['model']=='base')
    r,paths,_=make(tmp_path,'grpo',change=change)
    local=parse_parts([p.read_text() for p in paths])
    assert r['status']==local['status']=='PENDING/INCOMPLETE'
    assert local['endpoint_verdict_rederived'] and local['endpoints']['D_sensitivity']['values']==[0.]*3


@pytest.mark.parametrize('family',['opr','fisher'])
def test_benchmark_text_has_exact_scope_and_cannot_admit(tmp_path,family):
    _,paths,_=make(tmp_path,family,benchmark=True)
    local=parse_parts([p.read_text() for p in paths])
    assert local['simulated_status']=='NON-INCUMBENT BENCHMARK'
    assert local['metadata']['meta']['scope']=='NON-INCUMBENT BENCHMARK'


def test_parts_missing_duplicated_or_mixed_are_never_complete(tmp_path):
    _,paths,_=make(tmp_path/'one','v2b',max_chars=4000)
    assert len(paths)>1
    texts=[p.read_text() for p in paths]
    with pytest.raises(Pending):parse_parts(texts[:-1])
    with pytest.raises(Invalid,match='duplicate part'):parse_parts(texts+[texts[0]])
    with pytest.raises(Invalid,match='mixed kit identities'):parse_parts([texts[0],texts[1].replace('b'*40,'c'*40)])


def test_seed_values_are_not_rounded_across_benefit_threshold(tmp_path):
    r,_,full=make(tmp_path/'first','isdft');r['primary']=benefit([1.9999999999]*3)
    r['simulated_status']='INCONCLUSIVE'
    for seed in SEEDS:r['correct_counts'][str(seed)+'.instruction']=[10**12,200000000000,800000000000,600000000000,619999999999]
    paths=write_parts(tmp_path/'second',full,'kit-test','b'*40)
    imported=parse_parts([p.read_text() for p in paths])
    assert imported['simulated_status']=='INCONCLUSIVE' and imported['primary']['mean']<2


def test_production_reporter_verifies_external_authority(tmp_path):
    from kit.report_text import report
    doc,spec,binding=fixture('freeze');atomic(tmp_path/'reading.json',read('freeze',doc,spec,binding))
    with pytest.raises(ValueError):
        report(dict(_mode='nm',family='freeze',reading=str(tmp_path/'reading.json'),infrastructure_update='a'*64),
            tmp_path/'report',tag='kit-test',commit='b'*40)


def stress_dispatch():
    from kit.nm_sim import text_stress_fixture
    from kit.nm_jobs import inventory
    families=[]
    for family in FAMILIES:
        doc,spec,binding=text_stress_fixture(family)
        reading=read(family,doc,spec,binding)
        assert reading.get('primary'),reading
        phases={r['phase'] for r in inventory()['rows'] if r['family']==family}
        accounting=dict(status='COMPLETE',attempts=999999,gpu_hours=123456.78901234567,
            by_phase={phase:dict(attempts=999999,gpu_hours=123456.78901234567,statuses=dict(COMPLETE=999997,FAILED=1,PENDING=1)) for phase in phases})
        families.append(dict(nm_family=family,reading=reading,infrastructure_authority='a'*64,accounting=accounting))
    return dict(nm_dispatch=families)


def test_six_family_worst_case_dispatch_is_four_parts_total_with_local_full(tmp_path):
    full=stress_dispatch()
    paths=write_parts(tmp_path,full,'kit-test','b'*40)
    texts=[p.read_text() for p in paths]
    assert len(paths)<=4 and sum(map(len,texts))<=4*11400
    assert all(len(t)<=11400 and max(map(len,t.splitlines()))<90 for t in texts)
    result=parse_parts(texts)
    assert result['status']=='DISPATCH' and list(result['families'])==list(FAMILIES)
    details=parse_parts([p.read_text() for p in (tmp_path/'full').glob('RETURN-TEXT-*.md')])
    for v in full['nm_dispatch']:
        family=v['nm_family'];local=result['families'][family];native=v['reading']
        assert local['primary']==native['primary'] and local['guards']==native['guards']
        assert local['status']==native['status'] and local.get('simulated_status')==native.get('simulated_status')
        assert local['endpoint_verdict_rederived']
        assert 'attempt' not in local['metadata'] and local['package_summaries']=={}
        assert details['families'][family]['package_summaries']==native['package_summaries']
        if family=='v2b':
            assert not any(k.startswith(('prior_months.','pooled_prior_months.')) for k in local['endpoints'])
            assert any(k.startswith('prior_months.') for k in details['families'][family]['endpoints'])
        if family=='opr':assert len(local['metadata']['disagree'])==66
    proc=subprocess.run([sys.executable,'-I',str(KIT/'nm_return.py'),'--text',*map(str,paths),'--out',str(tmp_path/'local.json')],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert load(tmp_path/'local.json')['families']==result['families']


def test_dispatch_overflow_fails_closed_and_keeps_full_locally(tmp_path):
    full=stress_dispatch()
    with pytest.raises(Invalid,match='dispatch ESSENTIAL exceeds four parts'):
        write_parts(tmp_path,full,'kit-test','b'*40,max_chars=4000)
    assert not list(tmp_path.glob('RETURN-TEXT-*.md'))
    assert (tmp_path/'full').is_dir()
    result=parse_parts([p.read_text() for p in (tmp_path/'full').glob('RETURN-TEXT-*.md')])
    assert set(result['families'])==set(FAMILIES)
