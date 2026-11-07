"""Owner-approved native B schedule regeneration, confined to confirmation."""
from pathlib import Path
import json
import os
import pytest
from kit import cr_confirmation as cr, cr_finqa, cr_jobs
from kit.nm_common import atomic, load, sha, Pending
from scripts.cr_confirmation_fixture import build


@pytest.fixture
def native(tmp_path,monkeypatch):
    monkeypatch.setenv('KIT_M3_MODE','cr-confirmation')
    return build(tmp_path/'fixture')


def rebind(cfg,path,doc):
    for row in doc['sources']:
        row['sha256']=sha(row['path'])
    atomic(path,doc);cfg['realised_inputs']['sha256']=sha(path)


def test_native_b_reproduction_and_six_distinct_orders(native,tmp_path):
    cfg,_=native;attempt=tmp_path/'attempt'
    result=cr.bind_inputs(cfg,attempt);proof=result['finqa_reproduction']
    assert result['status']==proof['status']=='COMPLETE'
    assert len(proof['checks'])==7 and all(c['byte_equal'] and c['expected']==c['observed'] for c in proof['checks'])
    assert proof==load(attempt/'finqa-reproduction.json')
    old=[Path(p).read_bytes() for p in result['realised']['task_sources']['finqa']['v6a_schedules'].values()]
    fresh=[(attempt/'finqa-generated'/f'schedule-{s}.jsonl').read_bytes() for s in cr_finqa.FRESH]
    assert len(set(old+fresh))==6
    for seed in cr_finqa.FRESH:
        assert sha(attempt/'finqa-generated'/f'schedule-{seed}.jsonl')==proof['regenerated'][str(seed)]['schedule_sha256']
    # B's original common order is not the sorted, deduplicated earlier replay pool.
    from kit.p3b_common import cycle,t
    sorted_rows=sorted({r['id']:r for r in t.read_jsonl(Path(result['realised']['task_sources']['finqa']['v6a_schedules']['101']))}.values(),key=lambda r:r['id'])
    assert cycle(sorted_rows,104,1280)!=t.read_jsonl(attempt/'finqa-generated/schedule-104.jsonl')


@pytest.mark.parametrize('mismatch',['old-order','old-bytes','stored-order','stored-targets','omitted-stored','wrong-algorithm','missing-common'])
def test_mismatch_is_pending_no_released_orders_and_warning_in_text(native,tmp_path,mismatch):
    cfg,path=native;doc=load(path);source=doc['task_sources']['finqa']
    if mismatch.startswith('old'):
        target=Path(source['v6a_schedules']['102'])
        if mismatch=='old-bytes':target.write_bytes(target.read_bytes()+b'\n')
        else:
            lines=target.read_bytes().splitlines(keepends=True);lines[0],lines[1]=lines[1],lines[0];target.write_bytes(b''.join(lines))
    elif mismatch=='stored-order':
        target=Path(source['b_work'])/'v4/report-main/finqa-s104-schedule.json';data=load(target);data['ids'].reverse();atomic(target,data)
    elif mismatch=='stored-targets':
        import pyarrow as pa
        import pyarrow.parquet as pq
        folder=Path(source['b_work'])/'v4/report-main/scheduled/finqa-s105';rows=pq.read_table(folder/'train.parquet').to_pylist()
        for row in rows:row['extra_info']['demonstration']='Another complete reasoning explanation using the report and the stated quantities.\nAnswer: 7'
        pq.write_table(pa.Table.from_pylist(rows),folder/'train.parquet');manifest=load(folder/'manifest.json');manifest['train_sha256']=sha(folder/'train.parquet');atomic(folder/'manifest.json',manifest)
    elif mismatch=='omitted-stored':doc['sources']=[r for r in doc['sources'] if 'finqa-s104' not in r['path']]
    elif mismatch=='wrong-algorithm':source['algorithm']='seed-shuffle-cycle-v1'
    else:source['corpus']='/missing-common.parquet'
    rebind(cfg,path,doc);attempt=tmp_path/'attempt'
    with pytest.raises(Pending,match='owner escalation'):cr.bind_inputs(cfg,attempt)
    result=load(attempt/'input-binding.json')
    assert result['status']=='PENDING' and result['escalation']=='owner'
    assert not (attempt/'finqa-generated').exists()
    if result.get('finqa_reproduction'):
        from kit.cr_text import payload
        full=dict(reading=dict(status='PENDING/INCOMPLETE',finqa_reproduction=result['finqa_reproduction']),infrastructure_authority='a'*64)
        rows=payload(full)
        assert any(r.startswith('W|NM-cr|FinQA_') and 'owner_escalation' in r for r in rows)
        from kit.nm_text import write_parts,parse_parts
        full['nm_family']='cr';parts=write_parts(tmp_path/'text',full,'kit-message-v7','b'*40)
        parsed=parse_parts([p.read_text() for p in parts])
        assert parsed['status']=='PENDING/INCOMPLETE'
        assert parsed['metadata']['finqa']['reproduction']==result['finqa_reproduction']


def test_optional_b_orders_absent_are_not_required(native,tmp_path):
    import shutil
    cfg,path=native;doc=load(path);source=doc['task_sources']['finqa'];base=Path(source['b_work'])/'v4/report-main'
    for seed in (104,105):
        (base/f'finqa-s{seed}-schedule.json').unlink();shutil.rmtree(base/f'scheduled/finqa-s{seed}')
    doc['sources']=[r for r in doc['sources'] if Path(r['path']).is_file()];rebind(cfg,path,doc)
    proof=cr.bind_inputs(cfg,tmp_path/'attempt')['finqa_reproduction']
    assert [r['name'] for r in proof['checks']]==['v6a-101','v6a-102','v6a-103']


def test_preparation_and_train_binding_require_gate_and_exact_generated_bytes(native,tmp_path):
    cfg,_=native;out=Path(cfg['out']);proof=cr.bind_inputs(cfg,tmp_path/'binding')
    atomic(out/'confirmation-binding.json',proof)
    # Staging of v6a hard-MATH source is independent of its fresh cycle schedules.
    (out/'targets-math').symlink_to(Path(cfg['v6a_root']).parent/'targets-math',target_is_directory=True)
    cr_jobs.prepare(cfg,out)
    root=out/'targets-finqa';bound=cr.task_binding(out,'finqa',root)
    assert bound['source']['algorithm']==cr_finqa.PROTOCOL
    for seed in cr.SEEDS:
        assert (out/'schedules-coverage/finqa/G'/f's{seed}.jsonl').is_file()
    (root/'schedule-104.jsonl').write_bytes(b'tampered\n')
    with pytest.raises(Pending,match='owner escalation'):cr.task_binding(out,'finqa',root)
    failed=cr_jobs.read_result(cfg,out)
    assert failed['status']=='PENDING/INCOMPLETE'
    assert failed['finqa_reproduction']['status']=='PENDING'
    assert 'owner escalation' in failed['finqa_reproduction']['errors'][0]


def test_final_read_retains_tampered_fresh_schedule_failure_after_successful_recheck(native,tmp_path,monkeypatch):
    from kit import m3_read
    from kit.cr_text import payload
    from kit.nm_text import write_parts,parse_parts
    from test_cr_confirmation import evidence
    cfg,_=native;out=Path(cfg['out']);binding=cr.bind_inputs(cfg,tmp_path/'binding')
    atomic(out/'confirmation-binding.json',binding)
    (out/'targets-math').symlink_to(Path(cfg['v6a_root']).parent/'targets-math',target_is_directory=True)
    cr_jobs.prepare(cfg,out)
    cells,assignment=evidence()
    atomic(out/'m3-scoring-assignment.json',assignment)
    monkeypatch.setattr(m3_read,'load_cells',lambda *a:(cells,assignment))
    assert cr_jobs.read_result(cfg,out)['status']=='PASS'
    (out/'targets-finqa/schedule-104.jsonl').write_bytes(b'tampered\n')
    failed=cr_jobs.read_result(cfg,out)
    # The real source recheck ran successfully AFTER task_binding wrote its
    # failed proof. No early return at "native assignment absent" is possible.
    check=max((out/'read-input-checks').glob('*/input-binding.json'),key=lambda p:p.stat().st_mtime_ns)
    assert load(check)==binding
    assert check.stat().st_mtime_ns>(out/'finqa-task-check.json').stat().st_mtime_ns
    assert failed['status']=='PENDING/INCOMPLETE'
    assert failed['qualification_pending']==cr.FINQA_PENDING
    assert failed['finqa_reproduction']==load(out/'finqa-task-check.json')
    assert failed['finqa_reproduction']['status']=='PENDING'
    assert 'FinQA regenerated schedule mismatch' in failed['finqa_reproduction']['errors'][0]
    assert failed['decisions']=={'CR':'PENDING'} and failed['contrasts']['CR-G']['label']=='PENDING'
    full=dict(nm_family='cr',reading=failed,infrastructure_authority='a'*64)
    rows=payload(full)
    assert any(r.startswith('W|NM-cr|FinQA_task_binding_') and 'owner_escalation' in r for r in rows)
    assert any(r.startswith('W|NM-cr|qualification_pending_') and 'owner_escalation' in r for r in rows)
    parts=write_parts(tmp_path/'text',full,'kit-message-v7','b'*40)
    parsed=parse_parts([p.read_text() for p in parts])
    assert parsed['metadata']['finqa']['reproduction']==failed['finqa_reproduction']
    assert any(w.startswith('qualification_pending_') for w in parsed['utilisation_warnings'])


@pytest.mark.parametrize('record',['confirmation-binding.json','finqa-task-check.json',
    'attempts/NM-cr-bind/failed/input-binding.json','input-checks/failed/input-binding.json',
    'read-input-checks/failed/input-binding.json'])
def test_any_failed_proof_precedes_newer_success(tmp_path,record):
    good=dict(status='COMPLETE',errors=[])
    failed=dict(status='PENDING',errors=['owner escalation'])
    path=tmp_path/record
    atomic(path,failed if path.name=='finqa-task-check.json' else dict(finqa_reproduction=failed))
    os.utime(path,ns=(1_000_000_000,1_000_000_000))
    atomic(tmp_path/'read-input-checks/newer/input-binding.json',dict(finqa_reproduction=good))
    assert cr_jobs.finqa_proof(tmp_path)==failed


def test_confirmation_text_contains_proof_hashes_and_all_seven_still_fit(native,tmp_path):
    from test_cr_confirmation import evidence
    from test_nm_text import stress_dispatch
    from kit.nm_text import write_parts,parse_parts
    cfg,_=native;proof=cr.bind_inputs(cfg,tmp_path/'attempt')['finqa_reproduction']
    cells,assignment=evidence();reading=cr.analyse(cells,assignment);reading['finqa_reproduction']=proof
    full=stress_dispatch();full['nm_dispatch'].append(dict(nm_family='cr',reading=reading,infrastructure_authority='a'*64))
    parts=write_parts(tmp_path/'seven',full,'kit-message-v7','b'*40);parsed=parse_parts([p.read_text() for p in parts])
    assert len(parts)<=4
    assert parsed['families']['cr']['metadata']['finqa']['reproduction']==proof


@pytest.mark.parametrize('trigger', [
    'b-work-null', 'corpus-null', 'b-work-null-after-hash-failure', 'tokenizer-array',
    'discovery-symlink-loop', 'membership-symlink-loop', 'stored-manifest-array',
    'stored-row-null', 'stored-extra-info-null', 'job-python-null',
])
def test_malformed_finqa_inputs_are_proof_errors_and_outer_pending(native, tmp_path, monkeypatch, trigger):
    cfg, path = native
    doc = load(path)
    source = doc['task_sources']['finqa']
    base = Path(source['b_work'])/'v4/report-main'
    if trigger.startswith('b-work-null'): source['b_work'] = None
    elif trigger == 'corpus-null': source['corpus'] = None
    elif trigger == 'tokenizer-array':
        cfg['technical_synthetic'] = False
        cfg['tokenizer'] = []
    elif trigger.endswith('symlink-loop'):
        loop = tmp_path/'loop'
        loop.symlink_to(loop)
        if trigger == 'discovery-symlink-loop': source['b_work'] = str(loop)
        else: doc['sources'].append(dict(path=str(loop), sha256='a'*64, role='schedule-sources'))
    elif trigger == 'stored-manifest-array': atomic(base/'scheduled/finqa-s104/manifest.json', [])
    elif trigger == 'stored-row-null':
        import pyarrow as pa
        import pyarrow.parquet as pq
        from types import SimpleNamespace
        original = pq.read_table
        def table(item, *args, **kwargs):
            if isinstance(item, pa.BufferReader): return SimpleNamespace(to_pylist=lambda: [None])
            return original(item, *args, **kwargs)
        monkeypatch.setattr(pq, 'read_table', table)
    elif trigger == 'stored-extra-info-null':
        import pyarrow as pa
        import pyarrow.parquet as pq
        target = base/'scheduled/finqa-s104/train.parquet'
        rows = pq.read_table(target).to_pylist()
        rows[0]['extra_info'] = None
        pq.write_table(pa.Table.from_pylist(rows), target)
        manifest = load(target.parent/'manifest.json')
        manifest['train_sha256'] = sha(target)
        atomic(target.parent/'manifest.json', manifest)
    elif trigger == 'job-python-null': cfg['job_python'] = None
    # The symlink loop cannot be rehashed, but the rest of the inventory is valid.
    for row in doc['sources']:
        if trigger == 'membership-symlink-loop' and row['path'] == str(loop): continue
        row['sha256'] = sha(row['path'])
    if trigger == 'b-work-null-after-hash-failure': doc['sources'][0]['sha256'] = '0'*64
    atomic(path, doc)
    cfg['realised_inputs']['sha256'] = sha(path)
    attempt = tmp_path/'attempt'
    with pytest.raises(Pending, match='owner escalation'): cr.bind_inputs(cfg, attempt)
    report = load(attempt/'input-binding.json')
    proof = report['finqa_reproduction']
    assert report['status'] == proof['status'] == 'PENDING'
    assert report['escalation'] == proof['escalation'] == 'owner'
    assert proof['errors'] and not (attempt/'finqa-generated').exists()
    assert not any('input checking failed' in e for e in report['errors']+proof['errors'])
    if trigger.startswith('stored-') or trigger == 'job-python-null':
        assert proof == load(attempt/'finqa-reproduction.json')
    if trigger == 'stored-row-null' or trigger == 'stored-extra-info-null':
        assert any('training rows/extra_info invalid' in e for e in proof['errors'])


@pytest.mark.parametrize('trigger', ['b-work-null', 'corpus-null', 'symlink-loop'])
def test_native_discovery_reports_pending_directly(native, tmp_path, trigger):
    cfg, path = native
    source = load(path)['task_sources']['finqa']
    if trigger == 'b-work-null': source['b_work'] = None
    elif trigger == 'corpus-null': source['corpus'] = None
    else:
        loop = tmp_path/'loop'
        loop.symlink_to(loop)
        source['b_work'] = str(loop)
    with pytest.raises(Pending, match='owner escalation'): cr_finqa.native_paths(source)


@pytest.mark.parametrize('appearing', ['order', 'folder'])
def test_optional_b_files_appearing_during_worker_are_refused(native, tmp_path, monkeypatch, appearing):
    import shutil
    cfg, path = native
    doc = load(path)
    source = doc['task_sources']['finqa']
    base = Path(source['b_work'])/'v4/report-main'
    order = base/'finqa-s104-schedule.json'
    folder = base/'scheduled/finqa-s104'
    order.unlink()
    shutil.rmtree(folder)
    doc['sources'] = [row for row in doc['sources'] if Path(row['path']).is_file()]
    rebind(cfg, path, doc)
    run = cr_finqa.subprocess.run
    def worker(*args, **kwargs):
        done = run(*args, **kwargs)
        if appearing == 'order': atomic(order, [])
        else: atomic(folder/'manifest.json', [])
        return done
    monkeypatch.setattr(cr_finqa.subprocess, 'run', worker)
    attempt = tmp_path/'attempt'
    with pytest.raises(Pending, match='unbound native input'): cr.bind_inputs(cfg, attempt)
    proof = load(attempt/'input-binding.json')['finqa_reproduction']
    assert proof['status'] == 'PENDING' and proof['escalation'] == 'owner'
    assert any('unbound native input' in e for e in proof['errors'])
    assert not any('JSON invalid' in e for e in proof['errors'])
    assert not (attempt/'finqa-generated').exists()


@pytest.mark.parametrize('trigger', ['null-source', 'null-config', 'null-python', 'null-schedules', 'unexpected'])
def test_standalone_reproduction_returns_pending_proof(native, tmp_path, monkeypatch, trigger):
    cfg, path = native
    source = load(path)['task_sources']['finqa']
    if trigger == 'null-source': source = None
    elif trigger == 'null-config': cfg = None
    elif trigger == 'null-python': cfg['job_python'] = None
    elif trigger == 'null-schedules': source['v6a_schedules'] = None
    else:
        def discovery(*args): raise AttributeError('unexpected discovery error')
        monkeypatch.setattr(cr_finqa, 'native_paths', discovery)
    attempt = tmp_path/'attempt'
    proof = cr_finqa.reproduce(source, cfg, attempt)
    assert proof == load(attempt/'finqa-reproduction.json')
    assert proof['status'] == 'PENDING' and proof['escalation'] == 'owner' and proof['errors']
    assert not (attempt/'finqa-generated').exists()


@pytest.mark.parametrize('variant',['roleless','empty-manifest','bad-hash','manifest-type'])
def test_auxiliary_sources_and_legacy_empty_manifests(native,tmp_path,variant):
    cfg,path=native;doc=load(path)
    # Tokenizer/native members need no invented scientific role.
    roles=set()
    for row in doc['sources']:
        role=row.get('role')
        if role in roles:row.pop('role',None)
        else:roles.add(role)
    aux=tmp_path/'auxiliary.json';atomic(aux,{})
    row=dict(path=str(aux),sha256=sha(aux))
    if variant=='empty-manifest':row['manifest']=True
    elif variant=='bad-hash':row['sha256']='0'*64
    elif variant=='manifest-type':row['manifest']='true'
    doc['sources'].append(row);atomic(path,doc);cfg['realised_inputs']['sha256']=sha(path)
    attempt=tmp_path/'attempt'
    if variant in ('bad-hash','manifest-type'):
        with pytest.raises(Pending):cr.bind_inputs(cfg,attempt)
        report=load(attempt/'input-binding.json')
        assert report['status']=='PENDING'
        assert any(('hash mismatch' if variant=='bad-hash' else 'manifest flag must be boolean') in e for e in report['errors'])
    else:
        report=cr.bind_inputs(cfg,attempt)
        assert report['status']==report['finqa_reproduction']['status']=='COMPLETE'
        assert report['sources'][-1]['expected']==report['sources'][-1]['observed']
