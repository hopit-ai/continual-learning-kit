"""Text-only return integration: unchanged native readers, synthetic CPU evidence.

The full fixture reuses message2/addon/a4 dry-run jobs and the existing PD/SP
CPU evidence constructors. Only K/CR's score-cell IO is doubled with the same
make_cells geometry used by test_m3_protocol; its native read_job/analyse,
assignment, contrasts, guards, cost and wall-time functions execute unchanged.
No fixtures qualify scientific execution.
"""
from pathlib import Path
import hashlib
import io
import tarfile
from types import SimpleNamespace as NS
import json
import re
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kit import report_text as report
from kit import text_return as transport
from text_return_fixtures import full_inputs, m3_cells

COMMIT='a'*40


def mangle(text):
    """Hard wraps even hashes/numbers, quotes each physical line, adds CRLF/NBSP."""
    lines=[]
    for line in text.splitlines():
        for i in range(0,len(line),71):
            lines.append('> '+line[i:i+71].replace('"','\u201c').replace(' ', '\u00a0'))
    return '\r\n'.join(lines)+'\r\n'


@pytest.fixture(scope='module')
def full(tmp_path_factory):
    root=tmp_path_factory.mktemp('all-family-text')
    inputs=full_inputs(root)
    # Private strings exist in the native result trees and collector inputs.
    # They must never escape through native stdout, warnings, unknown keys,
    # exception messages or text parts. The local JSON stays on the node.
    secrets=['person-secret','private-node.invalid','GPU-secret-device','99123456789',
             '/srv/private-data/person-secret/results']
    secrets += ['4812345','gpu-node-17','4912345']
    noise=dict(path=secrets[4],hostname=secrets[1],username=secrets[0],job_id=secrets[3],gpu_uuid=secrets[2],array_task=4812345)
    for family,specs in inputs.items():
        if family=='jobs':continue
        for spec in specs:
            source=Path(spec.get('source',spec.get('evidence')))
            if source.is_dir():(source/'private-extra.local.json').write_text(json.dumps(noise))
    # Exercise the actual native cost forwarding, not just projection doubles.
    # M reads arbitrary cost-record fields; X reads its manifest-bound costs.json.
    core=Path(inputs['M'][0]['source'])
    (core/'cost.json').write_text(json.dumps(dict(gpu_hours=0,seconds=0,array_task=4812345,pid=4912345,**{'gpu-node-17':dict(gpu_hours=0)})))
    from kit import x_collect
    archive=root/'canary-addon.tar.gz'
    x_collect.job(NS(out=Path(inputs['X'][0]['source']),archive=archive,standin=True))
    with tarfile.open(archive,'r:gz') as tar:
        files={m.name:tar.extractfile(m).read() for m in tar.getmembers()}
    costs=json.loads(files['addon/costs.json']);costs.update(array_task=4812345,pid=4912345)
    costs['gpu-node-17']=dict(gpu_hours=0)
    files['addon/costs.json']=json.dumps(costs,sort_keys=True).encode()
    manifest=json.loads(files['addon/manifest.json'])
    manifest['files']['addon/costs.json']=hashlib.sha256(files['addon/costs.json']).hexdigest()
    files['addon/manifest.json']=json.dumps(manifest,sort_keys=True).encode()
    with tarfile.open(archive,'w:gz') as tar:
        for name,raw in files.items():
            info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    inputs['X'][0]['source']=str(archive)
    original=report.native
    def with_noise(*args,**kw):
        value=original(*args,**kw)
        value['private_noise']=noise
        value['unexpected_extra']={s:s for s in secrets}
        value['array_task']=4812345
        value['pid']=4912345
        value['unlisted_node_timings']={'gpu-node-17':dict(gpu_hours=.123456)}
        print(' '.join(secrets),file=sys.stderr)
        return value
    with patch('kit.m3_read.load_cells',side_effect=lambda *a,**kw:m3_cells()),patch.object(report,'native',side_effect=with_noise):
        parts=report.report(inputs,root/'text',standin=True,commit=COMMIT,bootstrap=20,full_detail=True)
    return root,inputs,parts,secrets


def test_full_native_readers_round_trip_and_all_verdicts(full):
    root,_,parts,_=full
    source=json.loads((root/'text/full-results.local.json').read_text())
    assert set(source['readers'])==set(transport.FAMILIES)
    assert not [(f,r.get('_local_exception')) for f,rs in source['readers'].items() for r in rs if r.get('_reader_status')]
    assert source['readers']['PD'][0]['status']=='PLASTICITY LOSS SHOWN'
    assert any(c.get('array_task')==4812345 for c in source['readers']['M'][0]['cost']['records'])
    assert source['readers']['X'][0]['cost']['array_task']==4812345
    assert source['readers']['KCR'][0]['decisions']['K']=='PASS'
    assert source['readers']['KCR'][0]['decisions']['CR']=='PASS'
    expected=transport.parse_parts([p.read_text() for p in parts])
    pasted=transport.parse_parts([mangle(p.read_text()) for p in reversed(parts)])
    assert pasted==expected and pasted['complete']
    assert pasted['results_sha256']==hashlib.sha256(transport.canonical(source).encode('ascii')).hexdigest()
    first=transport.parse_parts([mangle(parts[0].read_text())],part1_only=True)
    assert first['decisions']==pasted['decisions']
    assert not first['metrics'] and not first['complete']
    assert {d['family'] for d in first['decisions']}==set(transport.FAMILIES)
    assert all(d['verdict'] in transport.LABELS for d in pasted['decisions'])
    for r in [*pasted['decisions'],*pasted['metrics'].values(),*pasted['jobs']]:
        assert r['provenance']=='PARTNER-COMPUTED' and r['independent_reverification']=='UNVERIFIED'
    # Every projected endpoint/guard leaf survives at four significant figures;
    # the full-precision node JSON remains bound by RESULTS-SHA256.
    names=transport.vocabulary()
    for family,rs in source['readers'].items():
        context=json.loads((root/'text/metric-map.local.json').read_text())
        public=transport.rounded(transport.project(rs[0],names,context,family))
        for pointer,value in transport.leaves(public):
            assert pasted['metrics'][family+pointer]['value']==value
    assert all(any(m['family']==family and m['seed']=='101' for m in pasted['metrics'].values()) for family in ('M','X','S','KCR','PD','SP','4A'))


def test_full_privacy_and_part_sizes(full,capsys):
    _,_,parts,secrets=full
    combined='\n'.join(p.read_text() for p in parts)
    for secret in secrets:assert secret not in combined
    assert all(p.read_text().isascii() and '\t' not in p.read_text() and len(p.read_text())<=12000 for p in parts)
    assert all(p.name==f'RETURN-TEXT-{k}-of-{len(parts)}.md' for k,p in enumerate(parts,1))
    assert all(s not in capsys.readouterr().out for s in secrets)


@pytest.fixture
def small(tmp_path):
    paths=report.report({},tmp_path/'text',standin=True,commit=COMMIT,full_detail=True)
    return [p.read_text() for p in paths]


def test_checksum_is_sha256_of_exact_body(small):
    for text in small:
        lines=text.splitlines(keepends=True)
        assert hashlib.sha256(''.join(lines[1:-1]).encode('ascii')).hexdigest()==lines[-1].split()[1]


@pytest.mark.parametrize('kind',['number','delete','footer'])
def test_checksum_failure_names_part(small,kind):
    text=small[1]
    if kind=='number':text=text.replace('|1800|','|1801|')
    elif kind=='delete':text=text.replace('T|COSTS|','T|COSTX|')
    else:text=text[:-10]+'0'*9+'\n'
    with pytest.raises(ValueError,match='part 2: checksum failed'):transport.parse_parts([small[0],text])


def test_missing_duplicate_mixed_and_truncation_refused(small):
    with pytest.raises(ValueError,match='missing parts'):transport.parse_parts([small[0]])
    with pytest.raises(ValueError,match='duplicate part 1'):transport.parse_parts([small[0],small[0]])
    with pytest.raises(ValueError,match='mixed kit'):transport.parse_parts([small[0],small[1].replace(COMMIT,'b'*40)])
    with pytest.raises(ValueError):transport.parse_parts([small[0],small[1][:-40]])
    with pytest.raises(ValueError,match='unframed|missing or truncated'):transport.parse_parts([small[0]+'extra',small[1]])


def test_unconfigured_partial_set_lists_all_decisions(small):
    parsed=transport.parse_parts([small[0]],True)
    assert len(parsed['decisions'])>=50
    assert all(d['verdict']==('DEFERRED' if d['family']=='4A' else 'PENDING') for d in parsed['decisions'])
    a4=[d for d in parsed['decisions'] if d['family']=='4A']
    assert {d['comparison'] for d in a4}=={'consolidation-memory','retention'}
    assert all(d['guards']==['NOT_CONFIGURED'] for d in a4)


def test_reader_failure_isolation_and_no_traceback_leak(tmp_path,capsys):
    def fail(family,*a,**kw):
        print('private-user private-node')
        raise ValueError('private-user private-node')
    with patch.object(report,'native',side_effect=fail):
        parts=report.report({f:[dict(source='absent')] for f in transport.FAMILIES},tmp_path/'out',commit=COMMIT)
    parsed=transport.parse_parts([p.read_text() for p in parts])
    assert {d['family'] for d in parsed['decisions']}==set(transport.FAMILIES)
    assert all(d['verdict']=='PENDING' and d['guards']==['READER_REFUSED'] for d in parsed['decisions'])
    text=''.join(p.read_text() for p in parts)+capsys.readouterr().out
    assert 'private-user' not in text and 'private-node' not in text


def test_job_ranges_all_statuses_and_private_ids(tmp_path):
    tasks=[dict(id='private-task-'+str(i)) for i in range(1,5)]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(tasks=tasks)))
    state=tmp_path/'state';state.mkdir()
    for i,status in ((1,'DONE'),(2,'FAILED'),(3,'RUNNING')):
        (state/(tasks[i-1]['id']+'.json')).write_text(json.dumps(dict(status=status,hostname='private-node')))
    rows,mapping=report.job_summary([dict(family='KCR',manifest=str(manifest),state=str(state))])
    assert rows==['J|KCR1|u1|DONE','J|KCR1|u2|FAILED','J|KCR1|u3-4|PENDING']
    assert len(mapping)==4 and 'private-task' not in ''.join(rows)


def test_accounting_allowance_warning_is_not_gate():
    readers={'KCR':[dict(costs=dict(actual=dict(total=1400,attempts=2)))],
             'PD':[dict(accounting=dict(recorded_gpu_hours=264))],
             'SP':[dict(wall_time=dict(gpu_hours=216,status='MEASURED'))]}
    cost=report.accounting(readers)
    assert cost['tier_recorded_gpu_hours']==1880 and cost['tier_over_allowance_warning'] and not cost['budget_gate']
    assert cost['tier_complete']
    assert report.accounting({})['tier_complete'] is False


def test_small_capacity_fails_closed_without_dropping_verdicts(tmp_path):
    with pytest.raises(ValueError,match='Part 1 capacity'):
        report.report({},tmp_path/'text',commit=COMMIT,max_chars=1000)


def test_parser_cli_and_part1(small,tmp_path):
    path=tmp_path/'paste.txt';path.write_text(mangle(small[0]))
    out=tmp_path/'read.json'
    done=subprocess.run([sys.executable,str(ROOT/'scripts/read_text_return.py'),str(path),'--part1-only','--out',str(out)],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    assert json.loads(out.read_text())['independent_reverification']=='UNVERIFIED'
    path.write_text(small[0].replace('D|M|','D|X|',1));out.unlink()
    done=subprocess.run([sys.executable,str(ROOT/'scripts/read_text_return.py'),str(path),'--part1-only','--out',str(out)],capture_output=True,text=True)
    assert done.returncode==2 and 'part 1: checksum failed' in done.stderr and not out.exists()


def test_canonical_results_digest_survives_integer_key_json_reload():
    value={'curves':{101:{0:1.,5:2.,10:3.,20:4.}}}
    assert transport.canonical(value)==transport.canonical(json.loads(json.dumps(value)))
    with pytest.raises(ValueError,match='duplicate canonical'):
        transport.canonical({1:0,'1':2})


def test_numeric_looking_private_key_is_never_printed():
    secret='99123456789'
    names=transport.vocabulary()
    projected=transport.project({'unexpected':{secret:secret},'job_id':secret,'uuid':'GPU-secret-device'},names,{})
    assert secret not in transport.canonical(projected)
    assert 'GPU-secret-device' not in transport.canonical(projected)


def test_public_jobs_resolve_from_essential_ops_without_private_manifest(tmp_path):
    labels,digest=transport.job_inventory()
    name='PD-101';assert name in labels
    manifest=tmp_path/'manifest.local.json';manifest.write_text(json.dumps(dict(tasks=[dict(id=name)])))
    state=tmp_path/'state';state.mkdir();(state/(name+'.json')).write_text('{"status":"DONE"}')
    rows,_=report.job_summary([dict(family='PD',manifest=str(manifest),state=str(state))])
    paths=transport.write_parts(tmp_path/'text',dict(readers={'PD':[dict(status='PENDING')]},jobs=rows),'kit-message-v3',COMMIT)
    result=transport.parse_parts([p.read_text() for p in paths])
    assert result['job_inventory_sha256']==digest
    assert result['jobs'][0]['registered_tasks']==[name]
    assert result['jobs'][0]['inventory_matches']


def test_native_pd_process_status_is_distinct_from_scientific_pending(tmp_path):
    manifest=tmp_path/'manifest.local.json'
    manifest.write_text(json.dumps(dict(tasks=[dict(id='PD-101'),dict(id='PD-102'),dict(id='PD-103')])))
    state=tmp_path/'state';state.mkdir()
    (state/'PD-101.json').write_text(json.dumps(dict(schema='pd-node-launch-receipt-v1',status='PROCESS_COMPLETED',scientific_result=False)))
    (state/'PD-102.json').write_text(json.dumps(dict(schema='pd-node-launch-receipt-v1',status='PENDING',error_type='RuntimeError',scientific_result=False)))
    rows,_=report.job_summary([dict(family='PD',manifest=str(manifest),state=str(state))])
    assert {row.split('|')[3] for row in rows}=={'DONE','FAILED','PENDING'}
    assert json.loads((state/'PD-102.json').read_text())['status']=='PENDING'


def test_nonfinite_tier_result_cannot_block_core_text(tmp_path):
    def fake(family,*a,**kw):
        return dict(status='INCONCLUSIVE',primary=dict(mean=float('nan'))) if family=='PD' else dict(_reader_status='PENDING')
    with patch.object(report,'native',side_effect=fake):
        paths=report.report({'M':[dict(source='synthetic')],'PD':[dict(source='synthetic')]},tmp_path/'text',commit=COMMIT)
    result=transport.parse_parts([p.read_text() for p in paths])
    assert any(d['family']=='M' for d in result['decisions'])
    pd=next(d for d in result['decisions'] if d['family']=='PD')
    assert pd['verdict']=='UNVERIFIED'


def test_chat_quote_removal_preserves_a_wrapped_greater_than(small):
    # Exercise the backwards-compatible legacy FULL frame, whose checksum
    # predates the new aggregate FULL-set binding.
    legacy=[]
    for text in small:
        lines=text.splitlines();body='\n'.join(x for x in lines[1:-1] if not x.startswith(('T|MODE|','T|FULL-SET|')))+'\n'
        legacy.append(lines[0]+'\n'+body+'SHA256 '+transport.sha(body)+'\n')
    lines=legacy[1].splitlines()
    rows=lines[1:-1]+['R|M|reader|all|/finqa>base|3']
    body='\n'.join(rows)+'\n'
    text=lines[0]+'\n'+body+'SHA256 '+transport.sha(body)+'\n'
    # A hard wrap immediately before '>' produces '> >base...' when quoted.
    # Only the first '> ' is chat markup; the second '>' is metric content.
    text=text.replace('/finqa>base','/finqa\n>base')
    pasted='\r\n'.join('> '+line for line in text.splitlines())+'\r\n'
    result=transport.parse_parts([mangle(legacy[0]),pasted])
    assert result['metrics']['M/finqa>base']['value']==3


@pytest.fixture(scope='module')
def essential(full,tmp_path_factory):
    root,_,_,_=full
    raw=json.loads((root/'text/full-results.local.json').read_text())
    out=tmp_path_factory.mktemp('essential')
    paths=transport.write_parts(out,raw,'kit-message-v3',COMMIT)
    return raw,paths


def test_essential_default_exhaustive_size_all_verdicts_and_privacy(essential,full):
    raw,parts=essential
    assert 3<=len(parts)<=4
    texts=[p.read_text() for p in parts]
    assert all(len(t)<=12000 and t.isascii() and '\t' not in t for t in texts)
    result=transport.parse_parts(texts)
    assert result==transport.parse_parts([mangle(t) for t in reversed(texts)])
    old=transport.parse_parts([p.read_text() for p in full[2]])
    assert result['decisions']==old['decisions'] and len(result['decisions'])==58
    pd=next(d for d in result['decisions'] if d['family']=='PD')
    assert pd['seeds_in_direction']==3 and pd['seed_direction']=='negative'
    assert result['endpoints']['PD:stage2-base']['seeds_in_direction']==3
    assert any(d['comparison']=='admission-evaluated' for d in result['decisions'])
    assert result['complete'] and not result['detail_complete']
    assert result['results_sha256']==transport.sha(transport.canonical(raw))
    assert result['jobs']==old['jobs'] and not transport.parse_parts([texts[0]],True)['jobs']
    assert all('T|FULL-SET|'+str(result['full_part_count'])+'|'+result['full_set_sha256'] in t for t in texts)
    assert all(secret not in ''.join(texts) for secret in full[3])
    assert {r['family'] for r in result['endpoints'].values()}==set(transport.FAMILIES)
    for endpoint in result['endpoints'].values():
        assert endpoint['provenance']=='PARTNER-COMPUTED' and endpoint['independent_reverification']=='UNVERIFIED'
    k=result['endpoints']['KCR:K-C0.primary']
    assert k['values']==[5,5.5,6] and k['sd']==.5 and k['seeds_in_direction']==3 and k['lower']==4.258
    assert result['endpoints']['X:p.acquisition-positive']['seeds_in_direction']==3
    assert result['endpoints']['4A:consolidation-memory']['seeds']==[101,102,103,104,105]
    assert all('J|' not in t for t in texts[:2]) and 'T|TIER-ALLOWANCE|1800' in texts[2]


def test_supplements_merge_any_full_part_and_later_cli(essential,tmp_path):
    _,parts=essential;texts=[p.read_text() for p in parts]
    full_parts=sorted((parts[0].parent/'full').glob('*.md'),key=lambda p:int(p.name.split('-')[2]))
    base=transport.parse_parts(texts)
    # A requested part need not be FULL Part 1 or adjacent to another page.
    page=full_parts[len(full_parts)//2].read_text()
    combined=transport.parse_parts(texts+[mangle(page)])
    merged=transport.parse_parts([mangle(page)],base=base)
    assert combined==merged and len(merged['metrics'])>len(base['metrics'])
    assert not merged['detail_complete'] and merged['decisions']==base['decisions']
    first=transport.parse_parts([full_parts[0].read_text()],base=merged)
    assert first['decisions']==base['decisions']
    with pytest.raises(ValueError,match='duplicate FULL supplement'):transport.parse_parts([page],base=merged)
    wrong=page.replace(base['full_set_sha256'],'0'*64)
    lines=wrong.splitlines();body='\n'.join(lines[1:-1])+'\n'
    wrong=lines[0]+'\n'+body+'SHA256 '+transport.sha(body)+'\n'
    with pytest.raises(ValueError,match='mixed FULL set'):transport.parse_parts([wrong],base=base)
    local=tmp_path/'base.json';local.write_text(transport.canonical(base))
    paste=tmp_path/'paste.txt';paste.write_text(mangle(page));out=tmp_path/'merged.json'
    done=subprocess.run([sys.executable,str(ROOT/'scripts/read_text_return.py'),str(paste),'--base',str(local),'--out',str(out)],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    assert json.loads(out.read_text())==combined
    # All later pages also verify the aggregate ordered content-hash digest.
    rest=[p.read_text() for p in full_parts if p.read_text()!=page]
    complete=transport.parse_parts(rest,base=merged)
    assert complete['complete'] and complete['detail_complete'] and not complete['missing_full_parts']
    together=transport.parse_parts(texts+[p.read_text() for p in full_parts])
    assert complete==together


def test_essential_overflow_fourth_page_and_fail_closed(essential,tmp_path):
    raw,_=essential
    parts=transport.write_parts(tmp_path/'four',raw,'kit-message-v3',COMMIT,max_chars=7000)
    assert len(parts)==4 and all(len(p.read_text())<=7000 for p in parts)
    parsed=transport.parse_parts([mangle(p.read_text()) for p in parts])
    original=transport.parse_parts([p.read_text() for p in essential[1]])
    assert parsed['endpoints']==original['endpoints'] and parsed['decisions']==original['decisions']
    # More input roots must never cause silent omission or a fifth paste page.
    many=dict(raw,readers={k:v*5 for k,v in raw['readers'].items()})
    with pytest.raises(ValueError,match='capacity'):
        transport.write_parts(tmp_path/'excess',many,'kit-message-v3',COMMIT)


def test_guard_selection_unrounded_half_point_boundary_and_native_stats():
    def stat(lower):return dict(values=[-1,-.5,0],mean=-.5,sd=.5,n=3,lower=lower)
    reader={'contrasts':{'K-C0':dict(primary=stat(0),guards={'far':stat(-.49999),'edge':stat(-.5),'failed':stat(-2),'unknown':{}},failed_guards=['failed'])}}
    rows=transport.essential_endpoints('KCR',reader)
    assert {r['name'] for r in rows}=={'K-C0.primary','K-C0.edge','K-C0.failed','K-C0.unknown'}
    assert next(r for r in rows if r['name']=='K-C0.edge')['lower']==-.5
    assert next(r for r in rows if r['name']=='K-C0.primary')['sd']==.5
    assert transport.rounded({'v':1.23456789})=={'v':1.235}


def test_full_flag_selects_retained_numbered_detail_set(tmp_path):
    paths=report.report({},tmp_path/'text',commit=COMMIT,full_detail=True)
    assert all(p.parent.name=='full' for p in paths)
    assert len(list((tmp_path/'text').glob('RETURN-TEXT-*.md')))==3
    assert transport.parse_parts([p.read_text() for p in paths])['detail_complete']


@pytest.mark.parametrize('field',['commit','k','n','tag','leading_k','leading_n'])
@pytest.mark.parametrize('part1_only',[False,True])
def test_header_one_character_errors_refused(small,field,part1_only):
    text=small[0];header,body=text.split('\n',1)
    if field=='commit':header=header.replace(COMMIT,'b'+COMMIT[1:])
    elif field=='k':header=header.replace('PART 1/','PART 2/')
    elif field=='n':header=header[:-1]+str(int(header[-1])+1)
    elif field=='leading_k':header=header.replace('PART 1/','PART 01/')
    elif field=='leading_n':header=header.replace('/2','/02')
    else:header=header.replace('kit-message-v3','kit-message-v4')
    with pytest.raises(ValueError,match='header/body identity mismatch'):
        transport.parse_parts([header+'\n'+body],part1_only=part1_only)


def test_missing_or_forged_body_identity_and_expected_commit(small,tmp_path):
    text=small[0]
    for prefix in ('T|KIT|','T|PART|'):
        lines=text.splitlines();body='\n'.join(x for x in lines[1:-1] if not x.startswith(prefix))+'\n'
        changed=lines[0]+'\n'+body+'SHA256 '+transport.sha(body)+'\n'
        with pytest.raises(ValueError,match='header/body identity mismatch'):transport.parse_parts([changed],True)
    with pytest.raises(ValueError,match='unexpected kit commit'):transport.parse_parts([text],True,expect_commit='b'*40)
    path=tmp_path/'paste.txt';path.write_text(text);out=tmp_path/'out.json'
    for expected,code in ((COMMIT,0),('b'*40,2)):
        done=subprocess.run([sys.executable,str(ROOT/'scripts/read_text_return.py'),str(path),'--part1-only','--expect-commit',expected,'--out',str(out)],capture_output=True,text=True)
        assert done.returncode==code,done.stderr
        if code:assert 'unexpected kit commit' in done.stderr


def test_full_supplement_misnumbering_is_immediately_refused(essential):
    _,parts=essential;base=transport.parse_parts([p.read_text() for p in parts])
    page=next((parts[0].parent/'full').glob('RETURN-TEXT-2-of-*.md')).read_text()
    header,body=page.split('\n',1)
    for changed in (header.replace('PART 2/','PART 3/'),header[:-1]+str((int(header[-1])+1)%10)):
        with pytest.raises(ValueError,match='header/body identity mismatch'):
            transport.parse_parts([changed+'\n'+body],base=base)


@pytest.mark.parametrize('family,bad',[('S',{}),('KCR',{'contrasts':{}}),('PD',{}),('X',{'eligible_comparisons':{'p':{'inherited':{'broken':{}}}}}),('PD',{'status':'PASS','primary':{'mean':float('nan')}})])
def test_family_formatting_errors_never_block_other_decisions(tmp_path,family,bad):
    raw=dict(readers={family:[bad],'SP':[dict(status='INCOMPLETE')]},jobs=[])
    parts=transport.write_parts(tmp_path/'text',raw,'kit-message-v3',COMMIT)
    result=transport.parse_parts([p.read_text() for p in parts])
    damaged=[d for d in result['decisions'] if d['family']==family]
    assert damaged and all(d['verdict']=='UNVERIFIED' and d['guards']==['READER_ERROR'] for d in damaged)
    assert next(d for d in result['decisions'] if d['family']=='SP')['verdict']=='PENDING'


def test_projection_and_endpoint_errors_are_isolated(tmp_path):
    original=transport.project
    def broken(v,names,mapping,location='',*args):
        if location=='PD' and v.get('status')=='PASS':raise TypeError('private-host path')
        return original(v,names,mapping,location,*args)
    with patch.object(transport,'project',side_effect=broken):
        paths=transport.write_parts(tmp_path/'projection',dict(readers={'PD':[dict(status='PASS')],'SP':[dict(status='INCOMPLETE')]}),'kit-message-v3',COMMIT)
    result=transport.parse_parts([p.read_text() for p in paths])
    assert next(d for d in result['decisions'] if d['family']=='PD')['verdict']=='UNVERIFIED'
    with patch.object(transport,'essential_endpoints',side_effect=lambda f,r:(_ for _ in ()).throw(TypeError()) if f=='PD' and not r.get('_reader_status') else []):
        paths=transport.write_parts(tmp_path/'endpoints',dict(readers={'PD':[dict(status='PASS')],'SP':[dict(status='INCOMPLETE')]}),'kit-message-v3',COMMIT)
    assert next(d for d in transport.parse_parts([p.read_text() for p in paths])['decisions'] if d['family']=='PD')['guards']==['READER_ERROR']


@pytest.mark.parametrize('error,code,status',[(FileNotFoundError,'MISSING_INPUT','PENDING'),(ValueError,'READER_REFUSED','PENDING'),(RuntimeError,'READER_ERROR','UNVERIFIED')])
def test_reader_missing_refused_and_error_codes(tmp_path,error,code,status):
    with patch.object(report,'native',side_effect=error('private-node')):
        parts=report.report({'PD':[dict(source='unused')]},tmp_path/'text',commit=COMMIT)
    d=next(d for d in transport.parse_parts([p.read_text() for p in parts])['decisions'] if d['family']=='PD')
    assert d['verdict']==status and d['guards']==[code]


def test_refusal_traceback_stays_local(tmp_path,capsys):
    inputs=tmp_path/'input.local.json';inputs.write_text('{}');out=tmp_path/'text'
    with patch.object(report,'report',side_effect=RuntimeError('private-node private-user')):
        assert report.main(['--inputs',str(inputs),'--out',str(out)])==2
    assert 'private-node' in (out/'refusal.local.txt').read_text()
    assert 'private-node' not in capsys.readouterr().err


def test_hmac_alias_key_local_unknown_numeric_canaries_dropped(full,essential,tmp_path):
    root,_,parts,secrets=full
    context=json.loads((root/'text/metric-map.local.json').read_text())
    assert len(context['hmac_key'])==64 and context['dropped_unknown_numeric']
    name='gpu-node-17'
    first=transport.safe_key(name,transport.vocabulary(),context)
    second=transport.safe_key(name,transport.vocabulary(),transport.alias_context())
    assert first!=second and first!='field-'+hashlib.sha256(name.encode()).hexdigest()[:16]
    text=''.join(p.read_text() for p in parts+essential[1])
    for secret in secrets+[context['hmac_key']]:assert secret not in text
    assert 'UNKNOWN_NUMERIC_DROPPED' in text
    # The random key is confined to the alias map, not the full-results JSON,
    # set manifest, job map, terminal output or either transport set.
    for p in (root/'text').rglob('*'):
        if p.is_file() and p.name!='metric-map.local.json':assert context['hmac_key'] not in p.read_text()
    value=transport.project({'array_task':4812345,'pid':4912345,'gpu-node-17':{'mean':.5}},transport.vocabulary(),context)
    assert '4812345' not in transport.canonical(value) and '4912345' not in transport.canonical(value)


def test_tier2_dispositions_consistent_and_admission_label():
    for state in ('UNINFORMATIVE','DEGENERATE'):
        rows=transport.decisions('X',dict(rung_b=state,incremental_plasticity=state,stacking=state))
        assert all(d['verdict']=='DEFERRED' for d in rows if d['comparison'] in ('rung_b','incremental_plasticity','stacking'))
    assert 'admission-evaluated' in {d['comparison'] for d in transport.decisions('M',{})}


def test_invisible_whatsapp_formatting_is_removed(small):
    invisible='\u200b\u200c\u200d\u200e\u200f\u2060\ufeff'
    originals=[mangle(t) for t in small]
    pasted=[invisible+invisible.join(t)+invisible for t in originals]
    assert transport.parse_parts(pasted)==transport.parse_parts(small)


def test_emit_self_parses_every_part(tmp_path):
    with patch.object(transport,'parse_parts',wraps=transport.parse_parts) as parser:
        paths=report.report({},tmp_path/'text',commit=COMMIT)
    all_parts=list((tmp_path/'text').rglob('RETURN-TEXT-*.md'))
    assert parser.call_count==len(all_parts)
    assert all(call.kwargs.get('_single_part') and call.kwargs.get('expect_commit')==COMMIT for call in parser.call_args_list)
    assert all(p.exists() for p in paths)


@pytest.mark.parametrize('function',['project','decisions'])
def test_family_fallback_is_independent_of_broken_formatter(tmp_path,function):
    original=getattr(transport,function)
    def broken(*args,**kwargs):
        family=args[3] if function=='project' and len(args)>3 else args[0] if function=='decisions' else kwargs.get('location','')
        if family=='PD':raise RuntimeError('private-node permanently refused')
        return original(*args,**kwargs)
    with patch.object(transport,function,side_effect=broken):
        paths=transport.write_parts(tmp_path/'text',dict(readers={'PD':[dict(status='PASS')],'SP':[dict(status='INCOMPLETE')]}),'kit-message-v3',COMMIT)
    result=transport.parse_parts([p.read_text() for p in paths])
    assert next(d for d in result['decisions'] if d['family']=='PD')['verdict']=='UNVERIFIED'
    assert next(d for d in result['decisions'] if d['family']=='SP')['verdict']=='PENDING'


@pytest.mark.parametrize('bad_field',['mean','per_seed'])
def test_malformed_numeric_fields_never_print_private_strings(tmp_path,bad_field):
    secret='/srv/private-user/private-node'
    signal=dict(mean=1,lower=.1,positive=3,status='PASS',per_seed={'101':1,'102':1,'103':1})
    if bad_field=='mean':signal['mean']=secret
    else:signal['per_seed']['101']=secret
    raw=dict(readers={'S':[{k:dict(signal) for k in ('drift_association','drift_forward','plasticity')}],'SP':[dict(status='INCOMPLETE')]})
    paths=transport.write_parts(tmp_path/'text',raw,'kit-message-v3',COMMIT)
    result=transport.parse_parts([p.read_text() for p in paths])
    assert all(d['verdict']=='UNVERIFIED' and d['guards']==['READER_ERROR'] for d in result['decisions'] if d['family']=='S')
    assert secret not in ''.join(p.read_text() for p in (tmp_path/'text').rglob('RETURN-TEXT-*.md'))
