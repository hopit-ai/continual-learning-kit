import pytest
from kit.nm_common import *
from kit import nm_pipeline as p,lane_pool


def qualification(tmp_path,binding,**kw):
    path=tmp_path/'qualification.json'
    atomic(path,dict(schema='nm-lane-qualification.v1',binding_sha256=digest(binding),concurrency=3,
        telemetry_complete=True,numeric_match=True,memory_safe=True,mean_util=80,min_window_mean=75,**kw))
    return dict(path=str(path),sha256=sha(path))


def test_lane_qualification_is_per_bound_panel_and_missing_is_serial(tmp_path):
    binding=dict(family='v2b',panel='september-off')
    assert p.qualified_concurrency(3,None,binding)==1
    assert p.qualified_concurrency(3,dict(path=str(tmp_path/'missing'),sha256='a'*64),binding)==1
    ref=qualification(tmp_path,binding)
    assert p.qualified_concurrency(3,ref,binding)==3
    assert p.qualified_concurrency(1,ref,binding)==1
    with pytest.raises(Invalid,match='qualification binding differs'):
        p.qualified_concurrency(3,ref,dict(family='v2b',panel='september-on'))
    doc=load(ref['path']);doc['min_window_mean']=69;atomic(ref['path'],doc);ref['sha256']=sha(ref['path'])
    assert p.qualified_concurrency(3,ref,binding)==1


@pytest.mark.parametrize('requested,qualified,actual',[(3,False,1),(3,True,3),(1,True,1)])
def test_real_lane_passes_and_records_inventory_concurrency(tmp_path,monkeypatch,requested,qualified,actual):
    attempt=tmp_path/'attempt';attempt.mkdir();plan=tmp_path/'plan';plan.mkdir()
    cell=plan/'cell.json';atomic(cell,dict(frozen=True));binding=dict(panel='legal')
    ref=qualification(tmp_path,binding) if qualified else None
    atomic(plan/'lane.json',dict(cells=[dict(plan=str(cell),plan_sha256=sha(cell),output='score.json')],
        commands=[dict(argv=['standin'])],qualification=ref,qualification_binding=binding,lane_concurrency=3 if qualified else 1))
    seen=[]
    def execute(commands,directory,**kw):
        seen.append(kw['concurrency']);atomic(attempt/'score.json',dict(cell=dict(complete=True)));return 0
    monkeypatch.setattr(lane_pool,'execute',execute)
    assert p.run_lane(plan,attempt,concurrency=requested)==[dict(complete=True)]
    assert seen==[actual] and load(attempt/'lane-concurrency.json')['actual']==actual


def test_cpu_plan_defaults_new_panel_to_serial(tmp_path,monkeypatch):
    atomic(tmp_path/'config.json',{})
    src=dict(model=str(tmp_path),sha256='a'*64,files={'config.json':'b'*64})
    monkeypatch.setattr(p,'bound',lambda *a:dict(recipe={}))
    monkeypatch.setattr(p,'exports_for',lambda *a:{str(s):src for s in STEPS})
    monkeypatch.setattr(p.nm_prepare,'render_scoring',lambda rows,*a:[r['prompt'] for r in rows])
    bundle=dict(base=src,tokenizer=str(tmp_path),panels={'september-off':[dict(id='x',prompt='p')]})
    monkeypatch.setattr(p,'row_result',lambda *a:tmp_path)
    for root in ('one','two'):atomic(tmp_path/root/'v2b'/'contexts-september.json',{'x':dict(on='context')})
    first=p.prepare_lane(tmp_path/'one','v2b',101,'september-off',bundle)
    lane=load(first/'lane.json');assert lane['lane_concurrency']==1
    bundle['lane_qualifications']={'september-off':qualification(tmp_path,lane['qualification_binding'])}
    second=p.prepare_lane(tmp_path/'two','v2b',101,'september-off',bundle)
    assert load(second/'lane.json')['lane_concurrency']==3


def test_score_operation_passes_the_inventory_concurrency(tmp_path,monkeypatch):
    from kit.nm_jobs import row_for
    seen=[]
    monkeypatch.setattr(p,'source',lambda *a:{})
    monkeypatch.setattr(p,'bound',lambda *a:{})
    monkeypatch.setattr(p,'run_lane',lambda *a,**kw:seen.append(kw['concurrency']))
    for panel in ('legal','knowledge'):
        row=row_for('NM-fisher-s101-'+panel)
        p.operation({},row,tmp_path,tmp_path,tmp_path/'config.json')
        assert seen[-1]==row['lane_concurrency']


def test_qualification_disappearing_after_preparation_falls_back_to_recorded_one(tmp_path,monkeypatch):
    attempt=tmp_path/'attempt';attempt.mkdir();binding=dict(panel='legal');ref=qualification(tmp_path,binding)
    Path(ref['path']).unlink()
    atomic(tmp_path/'lane.json',dict(cells=[],commands=[dict(argv=['standin'])],lane_concurrency=3,qualification=ref,qualification_binding=binding))
    seen=[]
    monkeypatch.setattr(lane_pool,'execute',lambda *a,**kw:seen.append(kw['concurrency']) or 0)
    p.run_lane(tmp_path,attempt,concurrency=3)
    assert seen==[1]
    assert load(attempt/'lane-concurrency.json')['actual']==1


@pytest.mark.parametrize('computed',[False,True])
def test_production_reading_pending_only_blocks_missing_primary(tmp_path,monkeypatch,computed):
    from kit import nm_read,nm_jobs
    from kit.nm_sim import fixture
    doc,spec,binding=fixture('grpo')
    result=nm_read.read('grpo',doc,spec,binding)
    result['status']='PENDING/INCOMPLETE'
    if not computed:result=dict(status='PENDING/INCOMPLETE',reason='missing scores')
    root=tmp_path/'lane';root.mkdir();atomic(root/'scores.json',doc)
    attempt=tmp_path/'attempt';attempt.mkdir();published=[]
    atomic(tmp_path/'launch.json',dict(infrastructure_update='a'*64))
    monkeypatch.setattr(nm_jobs,'inventory',lambda:dict(rows=[dict(id='score',family='grpo',phase='score')]))
    monkeypatch.setattr(p,'row_result',lambda *a:root)
    monkeypatch.setattr(p,'bound',lambda *a:binding)
    monkeypatch.setattr(p,'exports_for',lambda *a:{str(k):dict(sha256='b'*64) for k in STEPS})
    monkeypatch.setattr(nm_read,'read',lambda *a:result)
    bundle=dict(base=dict(sha256='c'*64),panels={name:[dict(id=i) for i in ids] for name,ids in spec['panels'].items()})
    if computed:
        output=p.reading(dict(family='grpo'),attempt,tmp_path,bundle,published)
        assert output['status']=='COMPLETE' and output['reading_status']=='PENDING/INCOMPLETE'
        assert str((attempt/'reading.json').relative_to(tmp_path)) in published
    else:
        with pytest.raises(Pending,match='missing scores'):p.reading(dict(family='grpo'),attempt,tmp_path,bundle,published)
    assert load(attempt/'reading.json')['status']=='PENDING/INCOMPLETE'


def test_dispatch_replaces_family_and_failed_publication_preserves_cycle(tmp_path,monkeypatch):
    from kit.nm_sim import fixture
    from kit.nm_read import read
    from kit import nm_text
    def full(family):
        doc,spec,binding=fixture(family)
        return dict(nm_family=family,reading=read(family,doc,spec,binding),infrastructure_authority='a'*64)
    pointer=tmp_path/'return-cycle.json'
    for index,family in enumerate(('v2b','grpo','v2b')):
        attempt=tmp_path/str(index);attempt.mkdir()
        paths=p.dispatch_text(tmp_path,attempt,full(family))
        imported=nm_text.parse_parts([v.read_text() for v in paths])
        assert set(imported['families'])==({'v2b'} if index==0 else {'v2b','grpo'})
    before=pointer.read_bytes();last=load(pointer)
    assert last['evidence']['v2b']['path']==str(tmp_path/'2'/'return-evidence.json')
    def overflow(*a,**kw):raise Invalid('dispatch ESSENTIAL exceeds four parts')
    monkeypatch.setattr(nm_text,'write_parts',overflow)
    attempt=tmp_path/'failed';attempt.mkdir()
    with pytest.raises(Invalid,match='four parts'):p.dispatch_text(tmp_path,attempt,full('freeze'))
    assert pointer.read_bytes()==before


@pytest.mark.parametrize('summary',[
    None, 'not-json', [], {},
    dict(telemetry_complete=False,mean_util=90,min_window_mean=85),
    dict(telemetry_complete=True,mean_util=69,min_window_mean=68),
    dict(telemetry_complete=True,mean_util=80,min_window_mean=None),
    dict(telemetry_complete=True,mean_util=80,min_window_mean=75),
])
def test_opr_generation_utilisation_is_advisory_and_retained_in_text(tmp_path,monkeypatch,summary):
    from kit import nm_generate,nm_read,nm_text
    from kit.nm_jobs import row_for
    from kit.nm_sim import fixture
    qualification=tmp_path/'qualification';qualification.mkdir()
    path=qualification/'util.summary.json'
    if isinstance(summary,str):path.write_text(summary)
    elif summary is not None:atomic(path,summary)
    attempt=tmp_path/'attempt';attempt.mkdir();calls=[]
    monkeypatch.setattr(p,'source',lambda *a:{})
    monkeypatch.setattr(p,'bound',lambda *a:{})
    monkeypatch.setattr(p,'row_result',lambda *a:qualification)
    def generate(plan,output,qualify):
        calls.append((plan,qualify));atomic(output,dict(attempts=[]))
    monkeypatch.setattr(nm_generate,'generate',generate)
    output=p.operation({},row_for('NM-opr-generate-s101'),attempt,tmp_path,tmp_path/'config.json')
    assert output['status']=='COMPLETE' and calls==[(tmp_path/'opr/generation-s101.json',False)]
    measured=output['generation_qualification']
    good=isinstance(summary,dict) and summary.get('min_window_mean')==75
    assert bool(output['utilisation_warnings']) is not good
    if isinstance(summary,dict):
        assert measured['mean_util']==summary.get('mean_util')
        assert measured['min_window_mean']==summary.get('min_window_mean')
    doc,spec,binding=fixture('opr');baseline=nm_read.read('opr',doc,spec,binding)
    spec['generation_qualification']={'101':measured}
    reading=nm_read.read('opr',doc,spec,binding)
    assert reading['status']==baseline['status'] and reading['primary']==baseline['primary']
    assert reading['utilisation_warnings']==output['utilisation_warnings']
    full=dict(nm_family='opr',reading=reading,infrastructure_authority='a'*64)
    parts=nm_text.write_parts(tmp_path/'text',full,'kit-message-v7b','b'*40)
    parsed=nm_text.parse_parts([part.read_text() for part in parts])
    assert parsed['status']==reading['status']
    assert parsed['metadata']['utilisation']['generation_qualification_s101']==measured
    assert parsed.get('utilisation_warnings',[])==output['utilisation_warnings']
    lines=text_lines('opr',reading,'a'*64)
    assert any(line.startswith('W|NM-opr|') for line in lines) is not good


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,101,10**400,True,'80'])
def test_invalid_generation_utilisation_values_warn_without_raising(tmp_path,value):
    import json
    path=tmp_path/'summary.json'
    path.write_text(json.dumps(dict(telemetry_complete=True,mean_util=value,min_window_mean=value)))
    result=p.generation_utilisation(path)
    assert result['mean_util'] is None and result['min_window_mean'] is None
    assert len(result['warnings'])==2


@pytest.mark.parametrize('historical',[False,True])
def test_production_opr_reading_collects_advisory_and_reproduces_archive_spec(tmp_path,monkeypatch,historical):
    from kit import nm_jobs,nm_read
    from kit.nm_sim import fixture
    doc,spec,binding=fixture('opr')
    binding['recipe']=dict(starts={str(seed):dict(sha256='d'*64) for seed in SEEDS})
    score=tmp_path/'score';atomic(score/'scores.json',doc)
    atomic(tmp_path/'launch.json',dict(infrastructure_update='a'*64))
    monkeypatch.setattr(nm_jobs,'inventory',lambda:dict(rows=[dict(id='score',family='opr',phase='score')]))
    monkeypatch.setattr(nm_jobs,'current',lambda *a:None)
    monkeypatch.setattr(p,'bound',lambda *a:binding)
    monkeypatch.setattr(p,'exports_for',lambda *a:{str(k):dict(sha256='b'*64) for k in STEPS})
    measured=dict(telemetry_complete=False,mean_util=61,min_window_mean=None,warnings=['TELEMETRY_INCOMPLETE'])
    def row_result(out,task):
        if task=='score':return score
        root=tmp_path/task;root.mkdir(exist_ok=True)
        atomic(root/'output.json',{} if historical else dict(generation_qualification=measured))
        return root
    monkeypatch.setattr(p,'row_result',row_result)
    for seed in SEEDS:
        atomic(tmp_path/'opr'/f'refresh-s{seed}.json',dict(cohort=spec['pre_cohort'][str(seed)],audit=spec['opr_refresh'][str(seed)]))
    # Use the frozen synthetic scientific fixture, keeping the production reader
    # and the pipeline's advisory collection and publication paths real.
    original=nm_read.read
    def read(family,scores,reader_spec,bound):
        for key in ('scorers','checkpoints'):reader_spec[key]=spec[key]
        return original(family,dict(scores,technical_synthetic=True),reader_spec,bound)
    monkeypatch.setattr(nm_read,'read',read)
    bundle=dict(base=dict(sha256='c'*64),panels={name:[dict(id=i) for i in ids] for name,ids in spec['panels'].items()})
    attempt=tmp_path/'reading';attempt.mkdir()
    result=p.reading(dict(family='opr'),attempt,tmp_path,bundle,[])
    assert result['status']=='COMPLETE'
    reading=load(attempt/'reading.json');reader_spec=load(attempt/'reader-spec.json')
    assert canonical(reading)==canonical(read('opr',load(attempt/'scores.json'),reader_spec,binding))
    if historical:assert 'generation_qualification' not in reading
    else:
        assert reading['generation_qualification']=={str(seed):measured for seed in SEEDS}
        assert len(reading['utilisation_warnings'])==3
        assert 'W|NM-opr|generation_qualification_s101_TELEMETRY_INCOMPLETE|1' in (attempt/'summary.txt').read_text()
