"""Fresh-seed confirmation: unchanged math, fail-closed bindings and real wrapper."""
from pathlib import Path
import json
import os
import subprocess
import sys
import pytest
from kit import cr_confirmation as cr, cr_jobs, m3_schedule, m3_contract, m3_read
from kit.nm_common import atomic,load,sha,Pending,Invalid,KIT


@pytest.fixture(autouse=True)
def confirmation(monkeypatch):
    monkeypatch.setenv('KIT_M3_MODE','cr-confirmation')


def evidence():
    cells={};assignments={}
    for arm,seed,stage,step,panel in cr.required_cells():
        reference=panel in ('knowledge-ll','hellaswag')
        proof=dict(uuid=f'GPU-{seed}-'+str(reference),environment='same-env',allocation_job=f'{seed}-'+str(reference))
        metrics=('ifeval','math','knowledge_strict') if panel=='general-v1' else ('knowledge',) if panel=='knowledge-ll' else (panel,)
        counts={metric:(95 if arm=='CR' and step==40 and metric in ('ifeval','math','knowledge') else 80) for metric in metrics}
        cell=dict(metrics={m:float(n) for m,n in counts.items()},items={m:{str(i):int(i<n) for i in range(100)} for m,n in counts.items()},
                  format_failures={},tolerant={},output_tokens=100,correct=80,attempt_scored=True,**proof)
        if reference:cell.update(method='REFERENCE-SCORED',reference_assignment=proof)
        else:assignments[f'm3|{seed}']=proof
        cells[arm,seed,stage,step,panel]=cell
    return cells,assignments


def test_isolated_seeds_and_unchanged_recipes(monkeypatch):
    assert m3_schedule.SEEDS==m3_contract.SEEDS==m3_read.SEEDS==(101,102,103)
    assert m3_schedule.active_seeds()==(104,105,106)
    pools={d:[dict(id=f'{d}-{i}',prompt=f'{d} {i}',response='answer') for i in range(n)] for d,n in dict(instruction=960,gsm8k=80,arc=80,finqa=320,contract_nli=320).items()}
    task=[dict(id=str(i),prompt='task',response='verified') for i in range(1280)]
    for stage,n in zip(m3_schedule.STAGES,(40,48,56)):
        built={};summaries={}
        for arm in ('G','CR'):
            built[arm],summaries[arm]=m3_schedule.build(arm,104,stage,task,pools)
            assert summaries[arm]['rows_per_update']==n
            assert summaries[arm]['task_rows']==1280 and summaries[arm]['skipped_draw_count']==0
            assert summaries[arm]['global_divisor']==32 and summaries[arm]['local_scale']==.25
        m3_schedule.audit_arms(built,summaries)
        # Independent oracle for unchanged seed substitution, with no global
        # mutation of the exploratory constants.
        ordered=sorted(pools['instruction'],key=lambda r:(m3_schedule.sha(f"m3-cr-v2|104|{stage}|generic-extra|{r['id']}"),r['id']))
        assert m3_schedule.hash_order(pools['instruction'],104,stage,'generic-extra')==ordered
    with pytest.raises(ValueError):m3_schedule.build('CR',101,'finqa',task,pools)
    with pytest.raises(ValueError):m3_schedule.build('K',104,'finqa',task,pools)
    monkeypatch.delenv('KIT_M3_MODE')
    assert m3_schedule.active_seeds()==(101,102,103)
    with pytest.raises(ValueError):m3_schedule.build('CR',104,'finqa',task,pools)


def test_three_fresh_seeds_only_crg_frozen_statistics_and_counts():
    cells,assignment=evidence();r=cr.analyse(cells,assignment)
    assert r['status']=='PASS' and set(r['contrasts'])=={'CR-G'}
    assert r['seeds']==[104,105,106] and r['decisions']=={'CR':'PASS'}
    c=r['contrasts']['CR-G'];assert c['primary']['values']==[9,9,9]
    assert c['primary']['sd']==0 and c['primary']['lower']==9
    assert set(r['ifeval'])==set(m3_read.STAGES)
    from kit.cr_text import compact,expand,mode
    counts=expand(compact(r['correct_counts']))
    with mode():again=cr.from_counts(counts)
    assert again['contrasts']['CR-G']['primary']==c['primary']
    assert again['contrasts']['CR-G']['guards']==c['guards']
    assert 'confirmation opens' not in json.dumps(r)
    assert r['future_precedence'].startswith('v8') and not r['incumbent_changed']
    s=m3_read.stats([1.,2.,3.],4.302653)
    assert s['sd']==1 and s['lower']==pytest.approx(2-4.302653/(3**.5))


@pytest.mark.parametrize('kind',['old','foreign-arm','missing-seed','missing-curve','curve-hardware','reference-hardware','mixed-job'])
def test_contamination_incomplete_and_entire_lane_hardware(kind):
    cells,assign=evidence()
    if kind=='old':cells['CR',101,'finqa',40,'finqa']=cells['CR',104,'finqa',40,'finqa'].copy()
    elif kind=='foreign-arm':cells['K',104,'finqa',40,'finqa']=cells['CR',104,'finqa',40,'finqa'].copy()
    elif kind=='missing-seed':cells={k:v for k,v in cells.items() if k[1]!=106}
    elif kind=='missing-curve':del cells['CR',104,'contract_nli',10,'test']
    elif kind=='curve-hardware':cells['CR',104,'contract_nli',10,'test']['uuid']='GPU-foreign'
    elif kind=='reference-hardware':cells['CR',104,'math_hard',40,'knowledge-ll']['uuid']='GPU-foreign'
    else:cells['CR',105,'math_hard',30,'math_hard']['allocation_job']='another-attempt'
    r=cr.analyse(cells,assign)
    expected='UNVERIFIED' if kind in ('old','foreign-arm') else 'PENDING/INCOMPLETE' if kind.startswith('missing') else 'UNVERIFIED-HARDWARE'
    assert r['status']==expected


def test_ifeval_each_boundary_is_decision_bearing_even_with_primary_gain():
    cells,assign=evidence()
    for seed in (104,105,106):
        cells['CR',seed,'contract_nli',40,'general-v1']['metrics']['ifeval']=78.
    r=cr.analyse(cells,assign)
    assert r['status']=='INCONCLUSIVE'
    assert r['contrasts']['CR-G']['primary']['mean']==9
    assert 'contract_nli/final/ifeval' in r['contrasts']['CR-G']['failed_guards']
    assert r['ifeval']['contract_nli']['lower']==-2


def inventory_fixture(tmp_path):
    sources=[]
    for role in cr.ROLES:
        path=tmp_path/(role+'.json');atomic(path,dict(path='m3-fallback') if role=='math-mode' else {})
        sources.append(dict(path=str(path),sha256=sha(path),role=role))
    paths={r['role']:r['path'] for r in sources}
    source=paths['schedule-sources']
    doc=dict(kit_commit='059d6293',hard_math_mode='m3-fallback',sources=sources,
        pools_manifest=paths['pools'],base_binding=paths['references'],hellaswag_manifest=paths['panels'],
        native_inventory=paths['scorers'],math_source=paths['math-mode'],
        task_sources={stage:dict(corpus=source,v6a_schedules={str(seed):source for seed in (101,102,103)},algorithm='seed-shuffle-cycle-v1') for stage in m3_read.STAGES})
    path=tmp_path/'node-inventory.json';atomic(path,doc)
    return dict(out=str(tmp_path/'confirmation'),v6a_root=str(tmp_path/'v6a'),v6b_root=str(tmp_path/'v6b'),six_row_root=str(tmp_path/'root-A'),realised_inputs=dict(path=str(path),sha256=sha(path))),path


@pytest.mark.parametrize('bad',['missing','manifest-hash','source-hash','mode','mode-receipt','incomplete','member'])
def test_missing_or_mismatched_realised_inputs_pending_escalation_no_reconstruction(tmp_path,bad):
    cfg,path=inventory_fixture(tmp_path);doc=load(path)
    if bad=='missing':path.unlink()
    elif bad=='manifest-hash':cfg['realised_inputs']['sha256']='0'*64
    else:
        if bad=='source-hash':Path(doc['sources'][0]['path']).write_text('changed')
        elif bad=='mode':doc['hard_math_mode']=None
        elif bad=='mode-receipt':doc['hard_math_mode']='core-copy'
        elif bad=='incomplete':doc['sources']=doc['sources'][:-1]
        else:
            p=Path(doc['sources'][0]['path']);atomic(p,dict(files={'unbound.json':'a'*64}));doc['sources'][0].update(sha256=sha(p),manifest=True)
        atomic(path,doc);cfg['realised_inputs']['sha256']=sha(path)
    attempt=tmp_path/'attempt'
    with pytest.raises(Pending):cr.bind_inputs(cfg,attempt)
    r=load(attempt/'input-binding.json');assert r['status']=='PENDING' and r['escalation']=='owner'
    assert not (tmp_path/'confirmation').exists()
    assert all('expected' in row and 'observed' in row for row in r['sources'])


@pytest.mark.parametrize('content',['changed','[]'])
def test_corrupt_math_mode_receipt_is_pending_not_a_crash(tmp_path,content):
    # Set order of ROLES varies per process; pin the math-mode receipt explicitly.
    cfg,path=inventory_fixture(tmp_path);doc=load(path)
    mode=next(r for r in doc['sources'] if r['role']=='math-mode')
    Path(mode['path']).write_text(content)
    if content=='[]':mode['sha256']=sha(Path(mode['path']))
    atomic(path,doc);cfg['realised_inputs']['sha256']=sha(path)
    attempt=tmp_path/'attempt'
    with pytest.raises(Pending):cr.bind_inputs(cfg,attempt)
    r=load(attempt/'input-binding.json');assert r['status']=='PENDING' and r['escalation']=='owner'
    assert any('mode' in e for e in r['errors'])


def test_separate_roots_and_predata_diff_fail_closed(tmp_path):
    cfg,_=inventory_fixture(tmp_path)
    assert cr.isolated(cfg,tmp_path/'six-rows')==tmp_path/'confirmation'
    cfg['out']=cfg['v6a_root']
    with pytest.raises(Invalid):cr.isolated(cfg)
    with pytest.raises(Pending):cr.verify_diff(None)


def test_inventory_complete_native_curves_and_reference_decision_steps(tmp_path):
    from kit.nm_jobs import inventory
    baseline=inventory()['rows'];combined=inventory(confirmation=True)['rows']
    assert combined[:len(baseline)]==baseline
    seen={r['id'] for r in baseline};rows=cr_jobs.inventory()
    for row in rows:
        assert all(d['id'] in seen for d in row['depends_on'])
        assert row['gpus']==(8 if row['phase'] in ('qualify','train') else 1 if row['phase']=='score' else 0)
        assert '--kill-on-invalid-dep=yes' in row['dependency_options']
        assert row['lane_concurrency']<=2
        seen.add(row['id'])
    for seed in (104,105,106):
        lanes=[r for r in rows if r['phase']=='score' and r['seed']==seed]
        assert len(lanes)==2 and {r['lane'] for r in lanes}=={'native','reference'}
        commands=cr_jobs.native_commands(dict(job_python=sys.executable),tmp_path,seed)
        assert len(commands)==76
        assert {c[c.index('--arm')+1] for c in commands}=={'base','G','CR'}
        assert {int(c[c.index('--step')+1]) for c in commands}=={0,10,20,30,40}
        from kit.ref_score_job import lane_checkpoints
        path=tmp_path/'checkpoints.json';atomic(path,[c for s in (104,105,106) for c in cr_jobs.checkpoints(s)])
        from types import SimpleNamespace as NS
        a=NS(family='m3',seed=seed,checkpoints=path)
        assert len(lane_checkpoints(a))==13
        assert {c['step'] for c in lane_checkpoints(a)}=={0,40}
        assert {r['lane']:r['expected_score_cells'] for r in lanes}=={'native':76,'reference':26}
        old=load(path);old.append(dict(arm='CR',seed=101,stage='finqa',step=40));atomic(path,old)
        with pytest.raises(ValueError,match='UNVERIFIED'):lane_checkpoints(a)


def test_text_seventh_family_integer_rederivation_and_dispatch_budget(tmp_path):
    from kit.nm_text import write_parts,parse_parts
    cells,assignment=evidence();r=cr.analyse(cells,assignment)
    r['utilisation_warnings']=['OPERATOR_RECEIPT_ABSENT','104-native-TELEMETRY_INCOMPLETE']
    full=dict(nm_family='cr',reading=r,infrastructure_authority='a'*64)
    paths=write_parts(tmp_path/'text',dict(nm_dispatch=[full]),'kit-message-v7','b'*40)
    result=parse_parts([p.read_text() for p in paths])['families']['cr']
    assert result['status']=='PASS' and result['endpoint_verdict_rederived']
    assert result['utilisation_warnings']==r['utilisation_warnings']
    assert any('W|NM-cr|OPERATOR_RECEIPT_ABSENT|1' in p.read_text() for p in paths)
    assert result['contrasts']['CR-G']['primary']['mean']==9 and len(paths)<=4
    counts=result['metadata']['count'];counts['104.ifeval'][0]=1
    from kit.cr_text import recompute
    with pytest.raises(ValueError):recompute(result['metadata'])


def test_every_new_row_real_wrapper_pending_failure_and_nonce_isolation(tmp_path):
    from scripts.cr_confirmation_dryrun import run
    r=run(tmp_path/'dry-run')
    assert r['rows']==78 and r['wrapper_calls']==79 and r['pending']==75 and r['failed']==0 and r['malformed_pending']==1
    assert r['parts']<=4 and r['technical_synthetic']


def test_all_seven_stress_dispatch_within_four_essential_parts(tmp_path):
    from test_nm_text import stress_dispatch
    from kit.nm_text import write_parts,parse_parts
    cells,assignment=evidence();reading=cr.analyse(cells,assignment)
    # Match the full precision denominators used by the six-row stress fixture.
    # Integer counts, rather than rounded endpoint vectors, remain the authority.
    reading['lane_hashes']={str(seed)+'.'+lane:'e'*64 for seed in (104,105,106) for lane in ('native','reference')}
    warnings=cr_jobs.qualify_utilisation({},dict(rolling_window_seconds=300,startup_grace_seconds=60))
    reading['utilisation_warnings']=['OPERATOR_RECEIPT_ABSENT',
        *[f'{seed}-{lane}-{warning}' for seed in cr.SEEDS for lane in ('native','reference') for warning in warnings]]
    full=stress_dispatch();full['nm_dispatch'].append(dict(nm_family='cr',reading=reading,infrastructure_authority='a'*64))
    parts=write_parts(tmp_path/'seven',full,'kit-message-v7','b'*40)
    parsed=parse_parts([p.read_text() for p in parts])
    assert len(parts)<=4 and set(parsed['families'])=={'v2b','opr','fisher','isdft','freeze','grpo','cr'}
    assert parsed['families']['cr']['status']=='PASS'


def test_separate_830_budget_cannot_consume_six_row_branch(tmp_path,monkeypatch):
    from kit.nm_budget import reserve
    monkeypatch.setenv('SLURM_JOB_START_TIME','100')
    monkeypatch.setattr('kit.nm_budget.time.time',lambda:100)
    attempt=tmp_path/'attempts/NM-cr-train/nonce';attempt.mkdir(parents=True)
    row=dict(id='NM-cr-train',family='cr',gpus=8);record=dict(slurm_job_id='123')
    cfg=dict(budget=dict(branch='CR_confirmation',approved_gpu_hours=830,approval_reference='owner 6 November',row_max_seconds={'NM-cr-train':3600}))
    assert reserve(tmp_path,cfg,row,record,attempt)==3600
    cfg['budget'].update(branch='I0',approved_gpu_hours=1270)
    with pytest.raises(Pending):reserve(tmp_path,cfg,row,record,attempt)


def test_real_reference_writer_and_reader_accept_fresh_final_and_bind_identity(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    from kit import m3_manifest,m3_train,ref_score_job,ref_guard,m2_assignment
    from kit.p3b_common import files_manifest
    out=tmp_path/'confirmation';m3_manifest.init_root(out)
    model=tmp_path/'base';atomic(model/'config.json',{})
    m3_train.bind_base(NS(out=out,model=model,receipt=None,standin=True))
    run=out/'runs/G-s104-contract_nli';atomic(run/'complete.json',dict(technical_synthetic=True))
    names=['complete.json']
    for step in (10,20,30,40):
        atomic(run/f'hf-step{step}/config.json',{})
        name=f'export-{step}.json';atomic(run/name,dict(technical_synthetic=True,model_file_hashes={'config.json':sha(run/f'hf-step{step}/config.json')}));names.append(name)
    atomic(run/'manifest.json',files_manifest(run,names))
    m2_assignment.freeze(out/'ref-scoring-assignment.json',dict(schema='message2-scoring-assignment.v2',assignments={
        f'm3.{panel}|104':dict(node='CPU-STANDIN',uuid='CPU-STANDIN',environment='CPU-STANDIN') for panel in ('knowledge-ll','hellaswag')}))
    path=out/'checkpoints.json';atomic(path,[dict(arm='G',seed=104,stage='contract_nli',step=40)])
    monkeypatch.setenv('KIT_LANE_JOB','a'*32)
    a=NS(out=out,family='m3',seed=104,step=40,panel='knowledge-ll',checkpoints=path,gpu='0',standin=True)
    ref_score_job.run(a)
    records=ref_guard.records(ref_guard.local_files(out),True)
    assert len(records)==1
    meta,rows=records[0];assert meta['seed']==104 and meta['step']==40 and len(rows)==100
    assert meta['assignment']['allocation_job']=='a'*32
    # Even re-sealed intermediate reference evidence is outside the contract.
    for step in (10,20,30):
        files={name.replace('-k40-',f'-k{step}-'):raw for name,raw in ref_guard.local_files(out).items()}
        name=next(name for name in files if name.endswith('/ref-identity.json'))
        changed=json.loads(files[name]);changed['step']=step
        files[name]=json.dumps(changed).encode()
        manifest=name.rsplit('/',1)[0]+'/manifest.json'
        changed_manifest=json.loads(files[manifest])
        changed_manifest['files']['ref-identity.json']=ref_guard.sha(files[name])
        files[manifest]=json.dumps(changed_manifest).encode()
        with pytest.raises(ValueError,match='not a registered checkpoint'):
            ref_guard.records(files,True)


def test_return_rederives_changed_valid_counts_without_trusting_pasted_verdict():
    from kit.cr_text import compact,recompute
    cells,assignment=evidence();r=cr.analyse(cells,assignment)
    counts=compact(r['correct_counts'])
    for seed in (104,105,106):
        # CR's ContractNLI final instruction becomes a -2 guard contrast.
        counts[f'{seed}.ifeval'][11]=78
    metadata=dict(meta=dict(reported_status='PASS',technical=False),count=counts)
    changed=recompute(metadata)
    assert changed['status']=='INCONCLUSIVE' and not changed['reported_verdict_matches']
    metadata['count']['101.ifeval']=metadata['count']['104.ifeval']
    assert recompute(metadata)['status']=='UNVERIFIED'


def test_reference_final_assignment_receipt_error_is_unverified_before_missing():
    cells,assign=evidence()
    cells['CR',104,'contract_nli',40,'knowledge-ll']['hardware_error']='assignment digest differs'
    del cells['G',106,'math_hard',40,'finqa']
    result=cr.analyse(cells,assign)
    assert result['status']=='UNVERIFIED-HARDWARE' and result['decisions']=={'CR':'UNVERIFIED-HARDWARE'}


def test_cpu_lane_preparation_refuses_old_seed_before_model_io(tmp_path):
    from types import SimpleNamespace as NS
    from kit.m3_score import prepare_scoring
    with pytest.raises(Exception,match='UNVERIFIED'):
        prepare_scoring(NS(checkpoints_json=json.dumps([['CR',101,'finqa',40]]),out=tmp_path,standin=True))


def test_old_partial_score_is_unverified_even_without_assignment(tmp_path):
    out=tmp_path/'confirmation'
    atomic(out/'scoring-jobs/partial/scores/old/m3-identity.json',dict(seed=101,arm='CR'))
    assert cr_jobs.read_result({},out)['status']=='UNVERIFIED'


def test_old_unsealed_csv_cannot_hide_behind_missing_identity(tmp_path):
    out=tmp_path/'confirmation';scores=out/'scores/partial/scores.csv';scores.parent.mkdir(parents=True)
    scores.write_text('seed,correct\n101,1\n')
    assert cr_jobs.read_result({},out)['status']=='UNVERIFIED'


@pytest.mark.parametrize('step',[10,20,30])
def test_intermediate_reference_checkpoints_are_rejected(tmp_path,step):
    from types import SimpleNamespace as NS
    from kit.ref_score_job import lane_checkpoints
    path=tmp_path/'checkpoints.json'
    atomic(path,[dict(arm='CR',seed=104,stage='finqa',step=step)])
    with pytest.raises(ValueError,match='0/40'):
        lane_checkpoints(NS(checkpoints=path,family='m3',seed=104))
    assert not any(k[3]==step and k[4] in ('knowledge-ll','hellaswag') for k in cr.required_cells())


def test_shipped_inventories_and_complete_duration_bounds():
    import math
    from kit.nm_jobs import inventory
    rows=cr_jobs.inventory()
    assert load(KIT/'resources/cr-confirmation/rows.json')['rows']==rows
    assert load(KIT/'resources/nm/rows-v7.json')==json.loads(json.dumps(inventory(confirmation=True)))
    bounds=load(KIT/'resources/cr-confirmation/row-max-seconds.json')
    assert len(bounds)==78 and set(bounds)=={r['id'] for r in rows}
    for row in rows:
        seconds=bounds[row['id']]
        assert type(seconds) is int and seconds>0
        if row['phase']=='train':upper={'finqa':31.95,'contract_nli':38.34,'math_hard':44.73}[row['stage']]/8
        elif row['phase']=='qualify':upper=12.24/8
        elif row['phase']=='score':upper=row['expected_score_cells']*.10
        else:continue
        assert seconds==math.ceil(upper*3600*1.2/60)*60
    score=load(KIT/'resources/cr-confirmation/scoring-inventory.json')
    keys={(c['arm'],c['seed'],c['stage'],c['step'],c['panel']) for c in score['cells']}
    assert len(score['cells'])==len(keys)==306 and keys==cr.required_cells()
    assert score['native_calls']==228 and score['reference_panel_cells']==78
    assert score['reference_processes_with_both_panels']==39


@pytest.mark.parametrize('family',['cr','v2b','opr','fisher','isdft','freeze','grpo'])
def test_notready_mapping_is_scoped_to_confirmation(tmp_path,monkeypatch,family):
    from kit import nm_execute,nm_jobs,nm_pipeline
    from kit.p3b_common import NotReady
    def unavailable(*a,**kw):raise NotReady('missing frozen input')
    cfg=dict(out=str(tmp_path/'six'),cr_confirmation=dict(out=str(tmp_path/'cr'),v6a_root=str(tmp_path/'v6a'),v6b_root=str(tmp_path/'v6b'),six_row_root=str(tmp_path/'root-A')))
    config=tmp_path/'launch.json';atomic(config,cfg)
    attempt=tmp_path/'attempt'
    monkeypatch.setenv('UTIL_WATCH_ACTIVE','1')
    monkeypatch.setattr(sys,'argv',['nm_execute','--config',str(config),'--task','test','--attempt',str(attempt)])
    monkeypatch.setattr(nm_jobs,'row_for',lambda task:dict(family=family))
    monkeypatch.setattr(cr_jobs,'operation',unavailable)
    monkeypatch.setattr(nm_pipeline,'operation',unavailable)
    assert nm_execute.main()==(75 if family=='cr' else 2)
    result=load(attempt/'output.json')
    assert result['status']==('PENDING' if family=='cr' else 'FAILED')


def test_diff_suggestions_require_file_at_baseline(tmp_path,monkeypatch):
    root=tmp_path/'repo';kit=root/'kit';kit.mkdir(parents=True)
    (kit/'nm_jobs.py').write_text('before\n')
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    git('init')
    git('add','kit/nm_jobs.py')
    git('-c','user.name=Test','-c','user.email=test-fixture','commit','-m','baseline')
    baseline=git('rev-parse','HEAD')
    (kit/'nm_jobs.py').write_text('after\n')
    (kit/'cr_text.py').write_text('new wrapper\n')
    monkeypatch.setattr(cr,'KIT',kit)
    monkeypatch.setattr(cr,'code_seal',lambda:{name:sha(kit/name) for name in ('nm_jobs.py','cr_text.py')})
    result=cr.diff_receipt(tmp_path/'receipt.json',base=baseline)
    assert {h['file']:h['suggested_category'] for h in result['hunks']}=={'kit/nm_jobs.py':'namespace','kit/cr_text.py':None}
    assert all(h['category'] is None for h in result['hunks'])
    diagnostic=cr.diff_receipt(tmp_path/'diagnostic.json',base='0'*40)
    assert diagnostic['diagnostic_hunks']
    assert all(h['suggested_category'] is None for h in diagnostic['diagnostic_hunks'])


@pytest.mark.parametrize('window,grace',[(300,60),(60,60),(300,0),(None,60),(300,None)])
def test_utilisation_reports_receipt_window_and_startup_grace(window,grace):
    receipt=dict(rolling_window_seconds=300,startup_grace_seconds=60)
    telemetry=dict(window_s=window,startup_grace_seconds=grace,mean_util=95,min_window_mean=90,
                   telemetry_complete=True,holder_slice_telemetry_complete=True,holder_slices=[{}])
    warnings=cr_jobs.qualify_utilisation(telemetry,receipt)
    assert ('UTILISATION_WINDOW_GRACE_DIFFERS' in warnings)==((window,grace)!=(300,60))


def test_confirmation_schedule_binding_rechecks_math_pool_and_eval_disjointness(tmp_path):
    from types import SimpleNamespace as NS
    from kit import m3_train,m3_pools
    from kit.m2_common import M,hard_data
    from kit.p3b_common import files_manifest,t,s
    out=tmp_path/'confirmation';root=out/'targets-math';root.mkdir(parents=True)
    pool=hard_data()['train_pool.jsonl'];held=hard_data()['eval.jsonl']
    atomic(root/'inputs.json',dict(pool=s.file_hash(M/'train_pool.jsonl')))
    t.write_jsonl(root/'pool.jsonl',[pool[0]])
    names=['inputs.json','pool.jsonl']
    for seed in (101,102,103):
        name=f'schedule-{seed}.jsonl';t.write_jsonl(root/name,[pool[0]]*1280);names.append(name)
    def freeze():atomic(root/'manifest.json',files_manifest(root,names,population=500,technical_synthetic=True))
    freeze()
    fresh=out/'targets-math_hard';atomic(fresh/'manifest.json',{})
    atomic(out/'confirmation-binding.json',dict(realised=dict(task_sources=dict(math_hard=dict(corpus='recorded')))))
    a=NS(out=out,standin=True)
    assert m3_train.schedule_binding(a,'math_hard',fresh)['source']==dict(corpus='recorded')
    # Re-seal a held-out row so hashing alone cannot hide the scientific error.
    t.write_jsonl(root/'pool.jsonl',[held[0]]);freeze()
    with pytest.raises(ValueError,match='outside the training pool'):
        m3_train.schedule_binding(a,'math_hard',fresh)


@pytest.mark.parametrize('window,grace',[(300,60),(60,60),(300,None)])
def test_reader_reports_utilisation_definition_without_gating(tmp_path,monkeypatch,window,grace):
    from kit import m3_train
    out=tmp_path/'confirmation'
    cells,assignment=evidence()
    atomic(out/'m3-scoring-assignment.json',{})
    bound=dict(code=cr.code_seal(),finqa_reproduction=dict(status='COMPLETE',errors=[]))
    atomic(out/'confirmation-binding.json',bound)
    receipt=dict(rolling_window_seconds=300,startup_grace_seconds=60)
    monkeypatch.setattr(cr_jobs,'verify_operator',lambda cfg,warnings:receipt)
    monkeypatch.setattr(cr_jobs,'bind_inputs',lambda *a:bound)
    monkeypatch.setattr(m3_read,'load_cells',lambda *a:(cells,assignment))
    monkeypatch.setattr(m3_train,'validate_stage_receipt',lambda *a:dict(status='PASS'))
    for seed in cr.SEEDS:
        for lane in ('native','reference'):
            nonce=f'{seed}-'+str(lane=='reference')
            atomic(out/'attempts'/f'NM-cr-{lane}-s{seed}'/nonce/'util.summary.json',
                   dict(window_s=window,startup_grace_seconds=grace,telemetry_complete=True,
                        mean_util=95,min_window_mean=90,holder_slice_telemetry_complete=True,holder_slices=[{}]))
    result=cr_jobs.read_result({},out)
    assert result['status']=='PASS' and result['decisions']=={'CR':'PASS'}
    assert bool(result['utilisation_warnings'])==((window,grace)!=(300,60))
    assert 'qualification_pending' not in result


@pytest.mark.parametrize('window,grace',[(300,60),(300,0),(0,60),(float('nan'),60),(300,-1),(300,'60')])
def test_operator_utilisation_definition_is_finite_and_valid(tmp_path,window,grace):
    doc=dict(status='PASS',qualified_changes=['3a','3b','3d','3g','3h','likelihood-K2','holder-slice'],
             scorer_numerics_unchanged=True,likelihood_concurrency=2,
             rolling_window_seconds=window,startup_grace_seconds=grace,
             code={name:sha(KIT/name) for name in ('inference_lifecycle.py','lane_pool.py','util_watch.py','eval_bed.py')})
    path=tmp_path/'operator.json'
    # Permit an invalid NaN fixture that production's canonical writer refuses.
    path.write_text(json.dumps(doc))
    cfg=dict(operator_receipt=dict(path=str(path),sha256=sha(path)))
    warnings=[]
    assert json.dumps(cr_jobs.verify_operator(cfg,warnings),sort_keys=True)==json.dumps(doc,sort_keys=True)
    assert ('OPERATOR_WINDOW_GRACE_INVALID' in warnings)==((window,grace) not in ((300,60),(300,0)))


@pytest.mark.parametrize('kind',['absent','missing','malformed','hash','status','changes','precision','code'])
def test_optional_operator_evidence_only_warns(tmp_path,kind):
    path=tmp_path/'operator.json'
    doc=dict(status='PASS',qualified_changes=['3a','3b','3d','3g','3h','likelihood-K2','holder-slice'],
             scorer_numerics_unchanged=True,likelihood_concurrency=2,
             rolling_window_seconds=300,startup_grace_seconds=60,
             code={name:sha(KIT/name) for name in ('inference_lifecycle.py','lane_pool.py','util_watch.py','eval_bed.py')})
    if kind=='status':doc['status']='PENDING'
    if kind=='changes':doc['qualified_changes']=[]
    if kind=='precision':doc['scorer_numerics_unchanged']=False
    if kind=='code':doc['code']={}
    atomic(path,doc)
    cfg=dict(operator_receipt=dict(path=str(path),sha256=sha(path)))
    if kind=='absent':cfg={}
    if kind=='missing':path.unlink()
    if kind=='malformed':path.write_text('not-json')
    if kind=='hash':cfg['operator_receipt']['sha256']='0'*64
    warnings=[]
    cr_jobs.verify_operator(cfg,warnings)
    assert warnings and all(w.startswith('OPERATOR_') for w in warnings)


@pytest.mark.parametrize('kind',['missing','malformed','low','incomplete','holder','window','grace','nan','complete'])
def test_reader_monitoring_never_changes_science(tmp_path,monkeypatch,kind):
    from kit import m3_train
    out=tmp_path/'confirmation';cells,assignment=evidence()
    atomic(out/'m3-scoring-assignment.json',{})
    bound=dict(code=cr.code_seal(),finqa_reproduction=dict(status='COMPLETE',errors=[]));atomic(out/'confirmation-binding.json',bound)
    monkeypatch.setattr(cr_jobs,'bind_inputs',lambda *a:bound)
    monkeypatch.setattr(m3_read,'load_cells',lambda *a:(cells,assignment))
    monkeypatch.setattr(m3_train,'validate_stage_receipt',lambda *a:dict(status='PASS'))
    telemetry=dict(window_s=300,startup_grace_seconds=300,telemetry_complete=True,
                   mean_util=95,min_window_mean=90,holder_slice_telemetry_complete=True,holder_slices=['GPU-measured'])
    if kind=='low':telemetry.update(mean_util=10,min_window_mean=0)
    if kind=='incomplete':telemetry.update(telemetry_complete=False,mean_util=None,min_window_mean=None)
    if kind=='holder':telemetry.update(holder_slice_telemetry_complete=False,holder_slices=[])
    if kind=='window':telemetry['window_s']=None
    if kind=='grace':telemetry.pop('startup_grace_seconds')
    if kind=='nan':telemetry['mean_util']='NaN'
    for seed in cr.SEEDS:
        for lane in ('native','reference'):
            path=out/'attempts'/f'NM-cr-{lane}-s{seed}'/f'{seed}-{lane=="reference"}'/'util.summary.json'
            if kind=='missing':continue
            atomic(path,telemetry)
            if kind=='malformed':path.write_text('not-json')
    result=cr_jobs.read_result({},out)
    assert result['status']=='PASS' and result['decisions']=={'CR':'PASS'}
    assert 'qualification_pending' not in result
    assert 'OPERATOR_RECEIPT_ABSENT' in result['utilisation_warnings']
    assert (len(result['utilisation_warnings'])==1)==(kind=='complete')
    atomic(tmp_path/'reading.json',result)
    assert load(tmp_path/'reading.json')['utilisation_warnings']==result['utilisation_warnings']


@pytest.mark.parametrize('kind',['binding-missing','binding-changed','code-changed','missing-cell','mixed-job','uuid','environment','old-seed'])
def test_reader_keeps_strict_integrity_checks_with_missing_monitoring(tmp_path,monkeypatch,kind):
    from kit import m3_train
    out=tmp_path/'confirmation';cells,assignment=evidence()
    atomic(out/'m3-scoring-assignment.json',{})
    bound=dict(code=cr.code_seal(),finqa_reproduction=dict(status='COMPLETE',errors=[]));atomic(out/'confirmation-binding.json',bound)
    observed=bound
    if kind=='binding-missing':(out/'confirmation-binding.json').unlink()
    if kind=='binding-changed':observed=dict(bound,changed=True)
    if kind=='code-changed':
        observed=dict(bound,code={});atomic(out/'confirmation-binding.json',observed)
    if kind=='missing-cell':del cells['CR',104,'contract_nli',10,'test']
    if kind=='mixed-job':cells['CR',104,'contract_nli',10,'test']['allocation_job']='other'
    if kind in ('uuid','environment'):cells['CR',104,'contract_nli',10,'test'][kind]='other'
    if kind=='old-seed':atomic(out/'scores/partial/m3-identity.json',dict(seed=101,arm='CR'))
    monkeypatch.setattr(cr_jobs,'bind_inputs',lambda *a:observed)
    monkeypatch.setattr(m3_read,'load_cells',lambda *a:(cells,assignment))
    monkeypatch.setattr(m3_train,'validate_stage_receipt',lambda *a:dict(status='PASS'))
    result=cr_jobs.read_result({},out)
    expected='UNVERIFIED-HARDWARE' if kind in ('mixed-job','uuid','environment') else 'UNVERIFIED' if kind=='old-seed' else 'PENDING/INCOMPLETE'
    assert result['status']==expected
    assert result['utilisation_warnings']


@pytest.mark.parametrize('proof',[None,{},dict(status='PENDING',errors=[]),
    dict(status='COMPLETE',errors=['owner escalation']),dict(status='COMPLETE',errors=[])])
def test_final_read_requires_complete_finqa_proof(tmp_path,monkeypatch,proof):
    from kit import m3_train
    from kit.cr_text import payload
    out=tmp_path/'confirmation';cells,assignment=evidence()
    atomic(out/'m3-scoring-assignment.json',assignment)
    bound=dict(code=cr.code_seal())
    if proof is not None:bound['finqa_reproduction']=proof
    atomic(out/'confirmation-binding.json',bound)
    rechecks=[]
    def recheck(*args):
        rechecks.append(args)
        return bound
    monkeypatch.setattr(cr_jobs,'bind_inputs',recheck)
    monkeypatch.setattr(m3_read,'load_cells',lambda *a:(cells,assignment))
    monkeypatch.setattr(m3_train,'validate_stage_receipt',lambda *a:dict(status='PASS'))
    result=cr_jobs.read_result({},out)
    assert len(rechecks)==1
    if proof and proof.get('status')=='COMPLETE' and not proof.get('errors'):
        assert result['status']=='PASS'
    else:
        assert result['status']=='PENDING/INCOMPLETE'
        assert result['qualification_pending']==cr.FINQA_PENDING
        assert any(r.startswith('W|NM-cr|qualification_pending_') and 'owner_escalation' in r
            for r in payload(dict(reading=result,infrastructure_authority='a'*64)))


@pytest.mark.parametrize('hosts',[None,[],['gpu-node-1'],['CPU-STANDIN','gpu-node-1'],['CPU-STANDIN.example']])
def test_synthetic_rejects_non_standin_hosts_before_bypasses(tmp_path,hosts):
    cfg=dict(out=str(tmp_path/'confirmation'),v6a_root=str(tmp_path/'v6a'),v6b_root=str(tmp_path/'v6b'),six_row_root=str(tmp_path/'root-A'),
        technical_synthetic=True,hosts=hosts)
    with pytest.raises(Invalid,match='non-stand-in host refused'):
        cr_jobs.settings(dict(out=str(tmp_path/'dispatch'),technical_synthetic=True,cr_confirmation=cfg))
    with pytest.raises(Invalid,match='non-stand-in host refused'):
        cr_jobs.operation(cfg,dict(phase='qualify'),tmp_path/'attempt',Path(cfg['out']))
    with pytest.raises(Invalid,match='non-stand-in host refused'):
        cr_jobs.read_result(cfg,Path(cfg['out']))
    with pytest.raises(Pending,match='non-stand-in host refused'):
        cr.bind_inputs(cfg,tmp_path/'attempt')
    assert load(tmp_path/'attempt/input-binding.json')['status']=='PENDING'


def test_standin_category_covers_bypasses_without_approving_hunks(tmp_path,monkeypatch):
    root=tmp_path/'repo';kit=root/'kit';kit.mkdir(parents=True)
    path=kit/'cr_jobs.py';path.write_text('before\n')
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    git('init');git('add','kit/cr_jobs.py')
    git('-c','user.name=Test','-c','user.email=test-fixture','commit','-m','baseline')
    baseline=git('rev-parse','HEAD')
    path.write_text("if cfg.get('technical_synthetic'): pass\n")
    monkeypatch.setattr(cr,'KIT',kit)
    monkeypatch.setattr(cr,'code_seal',lambda:{'cr_jobs.py':sha(path)})
    receipt=cr.diff_receipt(tmp_path/'diff.json',base=baseline)
    assert 'technical-standin' in receipt['allowed_categories']
    assert receipt['category_definitions']['technical-standin']==cr.TECHNICAL_STANDIN
    assert receipt['hunks'][0]['suggested_category']=='technical-standin'
    assert receipt['hunks'][0]['category'] is None and receipt['status']=='PENDING'


@pytest.mark.parametrize('target', ['missing', 'directory'])
def test_unchecked_manifest_digest_never_opens_manifest(tmp_path, monkeypatch, target):
    from kit.cr_finqa import InputChecks
    cfg, path = inventory_fixture(tmp_path)
    doc = load(path)
    source = tmp_path/target
    if target == 'directory': source.mkdir()
    doc['sources'].append(dict(path=str(source), manifest=True))
    atomic(path, doc)
    cfg['realised_inputs']['sha256'] = sha(path)
    original = InputChecks.document
    def document(self, item, expected, label):
        assert item != source, 'unchecked manifest was parsed'
        return original(self, item, expected, label)
    monkeypatch.setattr(InputChecks, 'document', document)
    with pytest.raises(Pending): cr.bind_inputs(cfg, tmp_path/'attempt')
    report = load(tmp_path/'attempt/input-binding.json')
    assert report['status'] == 'PENDING' and report['escalation'] == 'owner'
    assert report['sources'][-1]['expected'] is None
    assert report['sources'][-1]['observed'] is None


@pytest.mark.parametrize('trigger', [
    'cfg-null', 'ref-null', 'inventory-path-null', 'inventory-not-json', 'inventory-array',
    'sources-null', 'source-null', 'role-array', 'source-path-missing', 'source-path-null',
    'source-digest-null', 'source-digest-array', 'manifest-not-json', 'manifest-array',
    'manifest-files-null', 'tasks-null', 'tasks-array', 'task-null',
    'schedules-null', 'schedules-array', 'mode-deep', 'inventory-deep',
    'nan', 'nan-with-hash-failure', 'infinity',
])
def test_malformed_operator_inputs_publish_pending(tmp_path, trigger):
    cfg, path = inventory_fixture(tmp_path)
    doc = load(path)
    row = next(r for r in doc['sources'] if r['role'] == 'candidate-lists')
    if trigger == 'cfg-null': cfg = None
    elif trigger == 'ref-null': cfg['realised_inputs'] = None
    elif trigger == 'inventory-path-null': cfg['realised_inputs']['path'] = None
    elif trigger == 'sources-null': doc['sources'] = None
    elif trigger == 'source-null': doc['sources'].append(None)
    elif trigger == 'role-array': row['role'] = []
    elif trigger == 'source-path-missing': del row['path']
    elif trigger == 'source-path-null': row['path'] = None
    elif trigger == 'source-digest-null': row['sha256'] = None
    elif trigger == 'source-digest-array': row['sha256'] = []
    elif trigger.startswith('manifest-'):
        content = {'manifest-not-json': 'not-json', 'manifest-array': '[]', 'manifest-files-null': '{"files":null}'}[trigger]
        Path(row['path']).write_text(content)
        row.update(sha256=sha(row['path']), manifest=True)
    elif trigger == 'tasks-null': doc['task_sources'] = None
    elif trigger == 'tasks-array': doc['task_sources'] = []
    elif trigger == 'task-null': doc['task_sources']['finqa'] = None
    elif trigger == 'schedules-null': doc['task_sources']['finqa']['v6a_schedules'] = None
    elif trigger == 'schedules-array': doc['task_sources']['finqa']['v6a_schedules'] = []
    elif trigger == 'mode-deep':
        mode = next(r for r in doc['sources'] if r['role'] == 'math-mode')
        Path(mode['path']).write_text('['*2000+'0'+']'*2000)
        mode['sha256'] = sha(mode['path'])
    elif trigger in ('nan', 'nan-with-hash-failure', 'infinity'):
        doc['extra'] = float('inf') if trigger == 'infinity' else float('nan')
        if trigger == 'nan-with-hash-failure': row['sha256'] = '0'*64
    content = json.dumps(doc)
    if trigger == 'inventory-not-json': content = 'not-json'
    elif trigger == 'inventory-array': content = '[]'
    elif trigger == 'inventory-deep': content = '{"extra":'+'['*2000+'0'+']'*2000+'}'
    path.write_text(content)
    if isinstance(cfg, dict) and isinstance(cfg.get('realised_inputs'), dict):
        cfg['realised_inputs']['sha256'] = sha(path)
    attempt = tmp_path/'attempt'
    with pytest.raises(Pending, match='owner escalation'): cr.bind_inputs(cfg, attempt)
    raw = (attempt/'input-binding.json').read_text()
    report = json.loads(raw, parse_constant=lambda value: pytest.fail('nonfinite receipt '+value))
    assert report['status'] == 'PENDING' and report['escalation'] == 'owner'
    assert report['errors'] and not any('input checking failed' in e for e in report['errors'])
    assert {'expected', 'observed'} <= report['inventory'].keys()
    assert all({'expected', 'observed'} <= row.keys() for row in report['sources'])
    if trigger in ('nan', 'nan-with-hash-failure', 'infinity'):
        assert report['realised']['extra'] is None
        assert any('non-finite' in e for e in report['errors'])


@pytest.mark.parametrize('location', [
    'inventory-hash', 'inventory-read', 'source-hash', 'manifest-read',
    'mode-hash', 'mode-read', 'manifest-resolve', 'source-resolve',
])
@pytest.mark.parametrize('failure', [PermissionError, RuntimeError, ValueError, RecursionError])
def test_checked_input_filesystem_failures_publish_pending(tmp_path, monkeypatch, location, failure):
    cfg, path = inventory_fixture(tmp_path)
    doc = load(path)
    row = next(r for r in doc['sources'] if r['role'] == 'candidate-lists')
    mode = next(r for r in doc['sources'] if r['role'] == 'math-mode')
    member = tmp_path/'member.json'
    if location.startswith('manifest'):
        atomic(Path(row['path']), dict(files={member.name: 'a'*64}))
        row.update(sha256=sha(row['path']), manifest=True)
    atomic(path, doc)
    cfg['realised_inputs']['sha256'] = sha(path)
    target = path if location.startswith('inventory') else Path(mode['path']) if location.startswith('mode') else Path(row['path'])
    original_read = Path.read_bytes
    original_resolve = Path.resolve
    calls = []
    def read(item):
        if item == target and not location.endswith('resolve'):
            calls.append(item)
            if not location.endswith('read') or len(calls) == 2:
                raise failure('operator input unavailable')
        return original_read(item)
    def resolve(item, *args, **kwargs):
        if location.endswith('resolve') and item == (member if location == 'manifest-resolve' else target):
            calls.append(item)
            raise failure('operator input unresolvable')
        return original_resolve(item, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(Path, 'resolve', resolve)
    with pytest.raises(Pending): cr.bind_inputs(cfg, tmp_path/'attempt')
    report = load(tmp_path/'attempt/input-binding.json')
    assert calls and report['status'] == 'PENDING' and report['escalation'] == 'owner'
    assert any(failure.__name__ in error for error in report['errors'])
    assert not any('input checking failed' in error for error in report['errors'])


def test_unexpected_input_exception_still_publishes_pending(tmp_path, monkeypatch):
    def check(cfg, attempt, report, checked): raise AttributeError('unexpected input failure')
    monkeypatch.setattr(cr, '_check_inputs', check)
    with pytest.raises(Pending, match='AttributeError'): cr.bind_inputs({}, tmp_path/'attempt')
    report = load(tmp_path/'attempt/input-binding.json')
    assert report['status'] == 'PENDING' and report['escalation'] == 'owner'
    assert report['finqa_reproduction']['status'] == 'PENDING'


def test_binding_receipt_output_failure_remains_loud(tmp_path, monkeypatch):
    def write(*args): raise PermissionError('receipt output unavailable')
    monkeypatch.setattr(cr, 'atomic', write)
    with pytest.raises(PermissionError, match='receipt output unavailable'):
        cr.bind_inputs({}, tmp_path/'attempt')


@pytest.mark.parametrize('trigger', ['no-git', 'no-head', 'git-unavailable'])
def test_diff_receipt_without_git_diagnostics_is_pending(tmp_path, monkeypatch, trigger):
    kit = tmp_path/'standalone/kit'
    kit.mkdir(parents=True)
    source = kit/'cr_confirmation.py'
    source.write_text('standalone fixture\n')
    monkeypatch.setattr(cr, 'KIT', kit)
    monkeypatch.setattr(cr, 'code_seal', lambda: {'cr_confirmation.py': sha(source)})
    monkeypatch.delenv('GIT_DIR', raising=False)
    monkeypatch.delenv('GIT_WORK_TREE', raising=False)
    # The publisher gate runs this inside an export nested in a checkout; stop
    # git discovering that parent repository above the standalone fixture.
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    if trigger == 'no-head':
        subprocess.run(['git', 'init'], cwd=kit.parent, check=True, capture_output=True)
    elif trigger == 'git-unavailable':
        def missing(*args, **kwargs): raise FileNotFoundError('git unavailable')
        monkeypatch.setattr(subprocess, 'run', missing)
    report = cr.diff_receipt(tmp_path/'diff.json', base='0'*40)
    assert report == load(tmp_path/'diff.json')
    assert report['status'] == 'PENDING' and report['escalation'] == 'owner'
    assert 'unavailable' in report['reason'] and report['hunks'] == []
    assert 'diagnostic_base_commit' not in report


@pytest.mark.parametrize('trigger',[
    'out-missing','out-type','out-relative','roots-null','cr-null','config-null','config-json',
])
@pytest.mark.parametrize('entry',['child','wrapper'])
def test_cr_bind_settings_errors_are_owner_pending(tmp_path,trigger,entry):
    cfg=dict(out=str(tmp_path/'dispatch'),technical_synthetic=True,
        cr_confirmation=dict(out=str(tmp_path/'cr'),v6a_root=str(tmp_path/'v6a'),v6b_root=str(tmp_path/'v6b'),six_row_root=str(tmp_path/'root-A'),hosts=['CPU-STANDIN']))
    if trigger=='out-missing':del cfg['cr_confirmation']['out']
    elif trigger=='out-type':cfg['cr_confirmation']['out']=17
    elif trigger=='out-relative':cfg['cr_confirmation']['out']='relative'
    elif trigger=='roots-null':cfg['cr_confirmation']['other_roots']=None
    elif trigger=='cr-null':cfg['cr_confirmation']=None
    elif trigger=='config-null':cfg=None
    config=tmp_path/'launch.json';atomic(config,cfg)
    if trigger=='config-json':config.write_text('{bad JSON')
    attempt=tmp_path/'attempt'
    env=dict(os.environ,UTIL_WATCH_ACTIVE='1')
    if entry=='child':
        command=[sys.executable,'-I',str(KIT/'nm_execute.py'),'--config',str(config),
            '--task','NM-cr-bind','--attempt',str(attempt)]
    else:
        command=[sys.executable,'-I',str(KIT/'nm_jobs.py'),'run','--config',str(config),'--task','NM-cr-bind']
    done=subprocess.run(command,env=env,capture_output=True,text=True)
    assert done.returncode==75,done.stdout+done.stderr
    assert 'Traceback' not in done.stdout+done.stderr
    reports=list(tmp_path.rglob('input-binding.json'));assert len(reports)==1
    report=load(reports[0]);output=load(reports[0].parent/'output.json')
    assert report['status']==output['status']=='PENDING'
    assert report['escalation']==output['escalation']=='owner' and report['errors']
    assert {'expected','observed'}<=report.keys()
    if entry=='wrapper':
        record=load(reports[0].parent/'receipt.json')
        assert record['status']=='PENDING' and record['exit_code']==75


@pytest.mark.parametrize('trigger',[
    'diff-ref-type','diff-ref-path-type','diff-json','diff-array','diff-edited',
    'native-missing-nodes','native-null-nodes','native-node-type','native-excluded-type',
    'native-json','native-array','native-edited','launch-budget-type','host-policy',
])
def test_cr_bind_later_inputs_publish_final_status_once(tmp_path,monkeypatch,trigger):
    from kit import nm_execute
    native=tmp_path/'native.json';atomic(native,dict(nodes=['allowed'],message2_nodes=[]))
    cfg=dict(out=str(tmp_path/'dispatch'),cr_confirmation=dict(out=str(tmp_path/'cr'),
        v6a_root=str(tmp_path/'v6a'),v6b_root=str(tmp_path/'v6b'),six_row_root=str(tmp_path/'root-A'),hosts=['allowed']))
    crcfg=cfg['cr_confirmation']
    receipt=tmp_path/'diff.json';receipt.write_text('{malformed')
    crcfg['diff_receipt']=dict(path=str(receipt),sha256=sha(receipt))
    if trigger=='diff-ref-type':crcfg['diff_receipt']='receipt.json'
    elif trigger=='diff-ref-path-type':crcfg['diff_receipt']['path']=[]
    elif trigger=='diff-array':atomic(receipt,[]);crcfg['diff_receipt']['sha256']=sha(receipt)
    elif trigger=='diff-edited':receipt.write_text('{}')
    if not trigger.startswith('diff-'):monkeypatch.setattr(cr_jobs,'verify_diff',lambda *a:{})
    if trigger=='native-missing-nodes':atomic(native,{})
    elif trigger=='native-null-nodes':atomic(native,dict(nodes=None))
    elif trigger=='native-node-type':atomic(native,dict(nodes=[dict(node=3)]))
    elif trigger=='native-excluded-type':atomic(native,dict(nodes=['allowed'],message2_nodes='excluded'))
    elif trigger=='native-json':native.write_text('{malformed')
    elif trigger=='native-array':atomic(native,[])
    elif trigger=='launch-budget-type':crcfg['budget']=None
    elif trigger=='host-policy':crcfg['hosts']=['excluded']
    expected=sha(native)
    if trigger=='native-edited':native.write_text('edited after inventory binding')
    def checked_inputs(cfg,attempt,report,checked):
        report['realised']=dict(native_inventory=str(native))
        report['sources']=[dict(path=str(native),expected=expected,observed=expected)]
    monkeypatch.setattr(cr,'_check_inputs',checked_inputs)
    writes=[];write=cr.atomic
    def capture(path,doc):
        if Path(path).name=='input-binding.json':writes.append(doc['status'])
        return write(path,doc)
    monkeypatch.setattr(cr,'atomic',capture)
    config=tmp_path/'launch.json';atomic(config,cfg);attempt=tmp_path/'attempt'
    monkeypatch.setenv('UTIL_WATCH_ACTIVE','1')
    monkeypatch.setattr(sys,'argv',['nm_execute','--config',str(config),'--task','NM-cr-bind','--attempt',str(attempt)])
    assert nm_execute.main()==(2 if trigger=='host-policy' else 75)
    report=load(attempt/'input-binding.json');output=load(attempt/'output.json')
    assert writes==['PENDING'] and report['errors'] and report['escalation']=='owner'
    assert output['status']==('FAILED' if trigger=='host-policy' else 'PENDING')
    if trigger=='host-policy':assert 'HostPolicy' in output['reason']
    if trigger.startswith(('native-','diff-')):
        record=report['native_inventory' if trigger.startswith('native-') else 'diff_receipt']
        assert {'expected','observed'}<=record.keys()
    assert not (tmp_path/'cr/confirmation-binding.json').exists()


@pytest.mark.parametrize('trigger',['authority-type','python-type','other-root-type'])
def test_cr_bind_parent_input_errors_are_pending(tmp_path,monkeypatch,trigger):
    from kit import nm_jobs
    cfg=dict(out=str(tmp_path/'dispatch'),job_python=sys.executable,
        infrastructure_update='a'*64,cr_confirmation=dict(out=str(tmp_path/'cr'),
        v6a_root=str(tmp_path/'v6a'),v6b_root=str(tmp_path/'v6b'),six_row_root=str(tmp_path/'root-A'),hosts=['allowed']))
    if trigger=='authority-type':cfg['infrastructure_update']=[]
    elif trigger=='python-type':cfg['job_python']=[]
    else:cfg['cr_confirmation']['other_roots']=[[]]
    monkeypatch.setenv('SLURM_JOB_ID','operator-error-fixture')
    config=tmp_path/'launch.json';atomic(config,cfg)
    assert nm_jobs.run(config,'NM-cr-bind')==75
    # Settings-stage failures write beside the dispatch root, never into the
    # confirmation tree (isolation); later input errors stay in that tree.
    root=tmp_path/'dispatch/cr-input-errors' if trigger=='python-type' else tmp_path/'cr'
    record=nm_jobs.current(root,'NM-cr-bind',complete=False)
    attempt=root/'attempts/NM-cr-bind'/record['nonce']
    assert record['status']=='PENDING'
    report=load(attempt/'input-binding.json')
    assert report['status']=='PENDING' and report['escalation']=='owner'
    assert report['errors'] and {'expected','observed'}<=report.keys()
    assert load(attempt/'output.json')['status']=='PENDING'


def test_refused_confirmation_tree_receives_no_error_attempt(tmp_path):
    # Opus round 2, finding 2: an isolation refusal must not write into the
    # tree it refused; error attempts go beside the dispatch root.
    dispatch=tmp_path/'dispatch'
    cfg=dict(out=str(dispatch),technical_synthetic=True,
        cr_confirmation=dict(out=str(dispatch),v6a_root=str(tmp_path/'v6a'),v6b_root=str(tmp_path/'v6b'),six_row_root=str(tmp_path/'root-A'),hosts=['CPU-STANDIN']))
    config=tmp_path/'launch.json';atomic(config,cfg)
    done=subprocess.run([sys.executable,'-I',str(KIT/'nm_jobs.py'),'run','--config',str(config),'--task','NM-cr-bind'],
        env=dict(os.environ,UTIL_WATCH_ACTIVE='1'),capture_output=True,text=True)
    assert done.returncode==75,done.stdout+done.stderr
    reports=list(tmp_path.rglob('input-binding.json'));assert len(reports)==1
    assert reports[0].is_relative_to(dispatch/'cr-input-errors')
    assert not (dispatch/'attempts').exists() and not (dispatch/'state').exists()


def test_settings_stage_host_refusal_is_failed_with_receipt(tmp_path):
    # Opus round 2, finding 3: a host-policy refusal stays FAILED but leaves a receipt.
    cfg=dict(out=str(tmp_path/'dispatch'),technical_synthetic=True,
        cr_confirmation=dict(out=str(tmp_path/'cr'),v6a_root=str(tmp_path/'v6a'),v6b_root=str(tmp_path/'v6b'),six_row_root=str(tmp_path/'root-A'),hosts=['real-node']))
    config=tmp_path/'launch.json';atomic(config,cfg)
    done=subprocess.run([sys.executable,'-I',str(KIT/'nm_jobs.py'),'run','--config',str(config),'--task','NM-cr-bind'],
        env=dict(os.environ,UTIL_WATCH_ACTIVE='1'),capture_output=True,text=True)
    assert done.returncode==2,done.stdout+done.stderr
    assert 'Traceback' not in done.stdout+done.stderr
    receipts=list(tmp_path.rglob('receipt.json'));assert len(receipts)==1
    assert load(receipts[0])['status']=='FAILED'
