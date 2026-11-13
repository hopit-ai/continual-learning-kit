"""Standalone CPU probe decisions, receipts, recovery and deadline boundaries."""
import copy
import json
import subprocess
import threading
from pathlib import Path
import sys
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from kit import trace_common as c, trace_judge as judge, trace_jobs as jobs
from kit import trace_operator as dag_plan

ROOT=Path(__file__).resolve().parents[1]


class Tokenizer:
    def encode(self,text,**kwargs):
        return [ord(ch)+10 for ch in text]

    def apply_chat_template(self,messages,tokenize=False,**kwargs):
        text='\n'.join(m['content'] for m in messages)
        return self.encode(text) if tokenize else text


def tokenizer(monkeypatch):
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a,**k:Tokenizer())))


def label_cells(science=True):
    sizes=judge.PANELS if science else dict.fromkeys(judge.PANELS,30)
    return [dict(id=f'{panel}-{i}',panel=panel,labels=[1,-1]) for panel,n in sizes.items() for i in range(n)]


def passes():
    return {name:{cell:dict(labels=label_cells(cell=='science'),seconds=3 if name[0]=='D' else 2)
                  for cell in ('science','calibration')} for name in ('D1','F1','D2','F2')}


def test_probe_exact_threshold_and_repeatability():
    value=passes()
    assert judge.decide(value)['decision']=='FAST'  # exact 2/3, not .67
    for name in ('F1','F2'):
        for panel in judge.PANELS:
            row=next(r for r in value[name]['science']['labels'] if r['panel']==panel)
            row['labels']=[0,0]  # six changes, one in each panel/order
        value[name]['calibration']['labels'][0]['labels'][0]=0
    result=judge.decide(value)
    assert result['decision']=='FAST'
    assert [result['diagnostics'][cell]['mismatches'] for cell in ('science','calibration')]==[6,1]
    value['F2']['science']['labels'][1]['labels'][0]=0
    assert judge.decide(value)['decision']=='VOID'


@pytest.mark.parametrize('mutation',['science_cell','calibration','malformed','speed','denominator'])
def test_probe_rejects_without_tuning(mutation):
    value=passes()
    for name in ('F1','F2'):
        if mutation=='science_cell':
            for r in value[name]['science']['labels'][:2]:r['labels'][0]=0
        if mutation=='calibration':value[name]['calibration']['labels'][0]['labels']=[0,0]
        if mutation=='malformed':value[name]['science']['labels'][0]['labels'][0]=None
        if mutation=='speed':value[name]['science']['seconds']=2.001
        if mutation=='denominator':value[name]['science']['labels'].pop()
    if mutation=='denominator':
        with pytest.raises(ValueError):judge.decide(value)
    else:
        assert judge.decide(value)['decision']=='DEFAULT'


def environment():
    packages={name:dict(version='standin',metadata_sha256=c.digest(name),record_sha256=c.digest(name))
              for name in ('torch','transformers','vllm')}
    return dict(gpu_model='standin',driver=['standin'],packages=packages,software_sha256=c.digest(packages))


def profile_engine(profile):
    return dict(model='judge',tokenizer='judge',dtype='bfloat16',tensor_parallel_size=1,
                max_model_len=12544,enforce_eager=profile=='DEFAULT',
                gpu_memory_utilization=.85 if profile=='DEFAULT' else .90,
                max_num_seqs=256 if profile=='DEFAULT' else 512,attention_backend='FLASH_ATTN')


def probe_binding(cfg,science,calibration,hardware):
    return dict(lane='gold-101',config_sha256=c.digest(cfg),lane_sha256=c.digest('lane'),
                gpu_uuid_sha256=c.digest('same'),hardware=hardware,
                science_sha256=c.digest(science),calibration_sha256=c.digest(calibration))


@pytest.mark.parametrize('rater_keys',[None,('astra','opus')])
def test_probe_chronology_receipt_failure_and_profile_gate(tmp_path,monkeypatch,rater_keys):
    cfg=dict(out=str(tmp_path),judge_throughput_probe=True)
    if rater_keys:
        path=tmp_path/'raters.json';c.atomic(path,dict.fromkeys(rater_keys,{}))
        cfg['raters']=dict(path=str(path),sha256=c.sha(path))
    assert judge.admit(cfg,'gold-101')==dict(first=True)
    order=[]
    def cell(cfg,pairs,profile):
        science=len(pairs)==653
        order.append((profile,science))
        requests=[f'{r["id"]}-{i}' for r in pairs for i in (0,1)]
        audit=dict(token_ids_sha256=c.digest(pairs),request_ids=requests,
                   seeds=[int(c.digest([0,r])[:8],16) for r in requests],
                   grouping='one call, all both-order requests',canonical_orders=[0,1]*len(pairs),retry_count=0,
                   gpu_uuid_sha256=c.digest('same'),hardware=environment(),engine=profile_engine(profile))
        labels=label_cells(science)
        return dict(labels=labels,raw=[dict(text='raw')],seconds=3 if profile=='DEFAULT' else 1,
                    audit=audit,label_sha256=c.digest(labels),retry_count=0,malformed=0)
    monkeypatch.setattr(judge,'probe_cell',cell)
    science,calibration=label_cells(),label_cells(False)
    registered=judge.execute_probe(cfg,science,calibration,environment(),
                                   probe_binding(cfg,science,calibration,environment()))
    assert registered[0]==science
    assert order==[(p,s) for p in ('DEFAULT','FAST','DEFAULT','FAST') for s in (True,False)]
    assert c.load(tmp_path/'judge/calibration-labels.json')==calibration
    receipt=c.load(tmp_path/'judge-probe/receipt.json')
    assert receipt['decision']=='FAST'
    assert receipt['J_status'].startswith('PENDING: rater keys differ' if rater_keys else 'PENDING: raters JSON absent')
    assert judge.probe_profile(cfg,'gold-102',environment())=='FAST'
    assert judge.probe_profile(cfg,'gold-102',dict(model='other'))=='DEFAULT'
    assert judge.probe_profile(cfg,'gold-101',environment())=='DEFAULT'
    assert judge.admit(cfg,'gold-102')==dict(first=False)
    with pytest.raises(ValueError,match='single-shot'):
        judge.execute_probe(cfg,science,calibration,environment(),
                            probe_binding(cfg,science,calibration,environment()))
    from kit.trace_read import pages, parse_pages
    records={'judge-profile':judge.probe_essential(receipt)}
    assert records['judge-profile']=='TRACE|judge-profile|FAST|mism=0,0|t=0.333333'
    assert parse_pages(pages(dict(records=records),'test'))['records']==records
    assert len(records['judge-profile'])<90


def test_probe_crash_default_once(tmp_path,monkeypatch):
    cfg=dict(out=str(tmp_path))
    monkeypatch.setattr(judge,'probe_cell',lambda *a: (_ for _ in ()).throw(RuntimeError('OOM')))
    with pytest.raises(RuntimeError):judge.execute_probe(cfg,label_cells(),label_cells(False),{},{})
    before=(tmp_path/'judge-probe/receipt.json').read_bytes()
    judge.probe_failure(cfg,ValueError('no retry'))
    assert (tmp_path/'judge-probe/receipt.json').read_bytes()==before
    assert c.load(tmp_path/'judge-probe/receipt.json')['decision']=='DEFAULT'


def judge_fixture(tmp_path,monkeypatch,probe=False):
    tokenizer(monkeypatch)
    from kit import inference_lifecycle,m2_gpu
    engine_options=[]
    class Engine:
        def generate(self,inputs,params,**kw):
            return [SimpleNamespace(outputs=[SimpleNamespace(text='{"winner":"A"}',
                        finish_reason='stop',token_ids=[1,2])]) for _ in inputs]
    def create(**kwargs):
        engine_options.append(kwargs)
        engine=Engine()
        engine.llm_engine=SimpleNamespace(scheduler_config=SimpleNamespace(
            max_num_seqs=kwargs.get('max_num_seqs',256)))
        engine._kit_engine_kwargs=dict(kwargs,attention_backend='FLASH_ATTN')
        return engine
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=lambda **kw:kw))
    monkeypatch.setattr(inference_lifecycle,'create_llm',create)
    monkeypatch.setattr(judge,'create_judge_engine',lambda options:create(**options))
    monkeypatch.setattr(inference_lifecycle,'marker',lambda:None)
    monkeypatch.setattr(inference_lifecycle,'shutdown_since',lambda *a:None)
    monkeypatch.setattr(m2_gpu,'current_uuid',lambda:'GPU-STANDIN')
    monkeypatch.setattr(m2_gpu,'assert_current',lambda:None)
    monkeypatch.setattr(judge,'hardware',environment)
    monkeypatch.delenv('TRACE_TOKEN_CACHE',raising=False)
    base_path=tmp_path/'base'
    c.atomic(base_path/'config.json',dict(synthetic=True))
    base=dict(model=str(base_path),files={'config.json':c.sha(base_path/'config.json')})
    cfg=dict(out=str(tmp_path),judge=base)
    if probe:cfg['judge_throughput_probe']=True
    scientific=[dict(id=f'{panel}-{i}',panel=panel,prompt='question')
                for panel,n in judge.PANELS.items() for i in range(n)]
    calibration=[dict(r,prompt='calibration',a='first',b='second',checkpoint_independent=True)
                 for r in label_cells(False)]
    c.atomic(tmp_path/'prepared.json',dict(panels=dict(open=scientific),calibration=calibration,
                                         active_studies=['gold']))
    for state in c.states('gold',101):
        c.atomic(tmp_path/'scores/gold-101'/state['id']/'open.json',
                 [dict(id=r['id'],text='answer '+state['id']) for r in scientific])
    c.atomic(tmp_path/'scores/gold-101/lane.json',dict(complete=True,config_sha256=c.digest(cfg),
             uuid_sha256=c.digest(m2_gpu.normalized_uuid('GPU-STANDIN')),environment_sha256=c.digest(environment()),
             artifacts_sha256={str(p.relative_to(tmp_path)):c.sha(p)
                               for p in (tmp_path/'scores/gold-101').rglob('open.json')}))
    return cfg,engine_options


def test_real_probe_wiring_profiles_requests_and_registered_d1(tmp_path,monkeypatch):
    cfg,options=judge_fixture(tmp_path,monkeypatch,True)
    judge.admit(cfg,'gold-101')
    # Eight cell clocks: D1 science/calibration, F1, D2, F2.
    moments=iter([0,3,3,6,6,7,7,8,8,11,11,14,14,15,15,16])
    original=judge.time.monotonic
    def now():
        try:return next(moments)
        except StopIteration:return original()
    monkeypatch.setattr(judge.time,'monotonic',now)
    judge.run(cfg,'gold',101)
    receipt=c.load(tmp_path/'judge-probe/receipt.json')
    assert receipt['decision']=='FAST'
    assert [o['enforce_eager'] for o in options[:8]]==[True,True,False,False,True,True,False,False]
    assert all(o['enforce_eager'] for o in options[8:])
    assert all(o['gpu_memory_utilization']==.90 and o['max_num_seqs']==512 for o in options[2:4]+options[6:8])
    assert c.load(tmp_path/'judge/gold-101/D1/labels.json')==receipt['passes']['D1']['science']['labels']
    assert c.load(tmp_path/'judge/calibration-labels.json')==receipt['passes']['D1']['calibration']['labels']
    assert c.load(tmp_path/'judge/gold-101/summary.json')['_profile']=='DEFAULT'
    assert receipt['profiles']['DEFAULT']['max_num_seqs']==256
    assert receipt['profiles']['FAST']['max_num_seqs']==512
    assert receipt['J_status']=='PENDING: raters JSON absent'


def test_cpu_admission_waits_for_single_decision(tmp_path):
    cfg=dict(out=str(tmp_path))
    judge.admit(cfg,'first')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(judge.admit,cfg,'second')
        assert not future.done()
        judge.probe_failure(cfg,RuntimeError('failed'))
        assert future.result(timeout=3)==dict(first=False)


def test_probe_interrupted_after_d1_reuses_registered_labels(tmp_path,monkeypatch):
    cfg,options=judge_fixture(tmp_path,monkeypatch,True)
    judge.admit(cfg,'gold-101')
    initial=judge.probe_cell
    count=0
    def interrupted(cfg,pairs,profile):
        nonlocal count
        count+=1
        if count==3:raise KeyboardInterrupt('F1 interrupted')
        return initial(cfg,pairs,profile)
    monkeypatch.setattr(judge,'probe_cell',interrupted)
    judge.run(cfg,'gold',101)
    assert c.load(tmp_path/'judge-probe/receipt.json')['decision']=='DEFAULT'
    saved=(tmp_path/'judge/gold-101/D1/labels.json').read_bytes()
    options.clear()
    judge.run(cfg,'gold',101)
    assert (tmp_path/'judge/gold-101/D1/labels.json').read_bytes()==saved
    # Nine other states; no repeated D1 or calibration, and no re-probe.
    assert len(options)==9
    assert all(o['enforce_eager'] for o in options)
    assert count==3


def test_malformed_probe_receipt_defaults_instead_of_enabling_fast(tmp_path):
    cfg=dict(out=str(tmp_path))
    judge.admit(cfg,'first')
    c.atomic(tmp_path/'judge-probe/receipt.json',dict(decision='FAST'))
    assert judge.probe_profile(cfg,'later',{})=='DEFAULT'
    receipt=judge.decision_receipt(cfg)
    assert receipt['error_type']=='MalformedReceipt'
    assert receipt['decision']=='DEFAULT'


@pytest.mark.parametrize('attack',[
    'new_gpu','new_lane','environment','science_input','science_labels','science_raw',
    'calibration_labels','calibration_raw','calibration_journal','calibration_binding',
])
def test_astra_d1_recovery_mismatch_rejudges_whole_lane(tmp_path,monkeypatch,attack):
    cfg,options=judge_fixture(tmp_path,monkeypatch,True)
    judge.admit(cfg,'gold-101')
    original=judge.probe_cell
    calls=[]
    def interrupted(*args):
        calls.append(args[-1])
        if len(calls)==3:raise KeyboardInterrupt('F1 interrupted')
        return original(*args)
    monkeypatch.setattr(judge,'probe_cell',interrupted)
    judge.run(cfg,'gold',101)
    assert len(calls)==3
    lane_path=tmp_path/'scores/gold-101/lane.json'
    lane=c.load(lane_path)
    if attack=='new_gpu':
        from kit import m2_gpu
        monkeypatch.setattr(m2_gpu,'current_uuid',lambda:'GPU-OTHER')
        lane['uuid_sha256']=c.digest(m2_gpu.normalized_uuid('GPU-OTHER'))
    if attack=='new_lane':lane['attempt']='new scoring allocation'
    if attack=='environment':monkeypatch.setattr(judge,'hardware',lambda:dict(environment(),driver=['other']))
    if attack=='science_input':
        path=tmp_path/'scores/gold-101/D1/open.json'
        rows=c.load(path);rows[0]['text']='new scoring output';c.atomic(path,rows)
        lane['artifacts_sha256'][str(path.relative_to(tmp_path))]=c.sha(path)
    c.atomic(lane_path,lane)
    journal=tmp_path/'judge-probe/d1-science.json'
    if attack in ('science_labels','science_raw'):
        saved=c.load(journal)
        if attack=='science_labels':
            saved['labels'][0]['labels'][0]=0
            saved['label_sha256']=c.digest(saved['labels'])
        else:saved['raw'][0]['text']='{"winner":"B"}'
        c.atomic(journal,saved)
    if attack in ('calibration_labels','calibration_raw'):
        path=tmp_path/'judge'/('calibration-labels.json' if attack=='calibration_labels' else
                                'calibration-local-raw.json')
        c.atomic(path,[])
    if attack=='calibration_journal':(tmp_path/'judge-probe/d1-calibration.json').unlink()
    if attack=='calibration_binding':
        path=tmp_path/'judge-probe/d1-calibration.json'
        saved=c.load(path);saved['binding']['lane']='other-101';c.atomic(path,saved)
    options.clear()
    judge.run(cfg,'gold',101)
    assert len(calls)==3  # A recovery never invokes the probe again.
    assert len(options)==11  # All ten scientific states and calibration are regenerated.
    assert all(o['enforce_eager'] and o['gpu_memory_utilization']==.85 for o in options)
    assert c.load(tmp_path/'judge-probe/receipt.json')['decision']=='DEFAULT'
    assert len(c.load(tmp_path/'judge/calibration-labels.json'))==90


def test_recovery_checks_scoring_content_hash_before_using_d1(tmp_path,monkeypatch):
    cfg,options=judge_fixture(tmp_path,monkeypatch,True)
    judge.admit(cfg,'gold-101')
    (tmp_path/'scores/gold-101/D1/open.json').write_text('[]')
    with pytest.raises(ValueError,match='scoring D1 content differs'):
        judge.run(cfg,'gold',101)
    assert options==[]


@pytest.mark.parametrize('cell',['science','calibration'])
@pytest.mark.parametrize('attack',['unchanged','labels','tokens','settings','timing'])
def test_astra_accepted_d1_rejects_self_consistent_replacement(tmp_path,monkeypatch,cell,attack):
    cfg,options=judge_fixture(tmp_path,monkeypatch,True)
    judge.admit(cfg,'gold-101')
    original=judge.probe_cell
    calls=[]
    def measured(*args):
        calls.append(args[-1])
        value=original(*args)
        value['seconds']=3 if args[-1]=='DEFAULT' else 1
        return value
    monkeypatch.setattr(judge,'probe_cell',measured)
    judge.run(cfg,'gold',101)
    receipt=c.load(tmp_path/'judge-probe/receipt.json')
    assert receipt['decision']=='FAST' and len(calls)==8
    if attack=='unchanged':
        options.clear()
        judge.run(cfg,'gold',101)
        assert len(options)==9 and len(calls)==8
        assert c.load(tmp_path/'judge-probe/receipt.json')==receipt
        return
    journal=tmp_path/'judge-probe'/f'd1-{cell}.json'
    saved=c.load(journal)
    if attack=='labels':
        # Astra's attack: internally valid TIE response, labels and both hashes.
        saved['raw'][0]['text']='{"winner":"TIE"}'
        saved['labels'][0]['labels'][0]=0
        saved['raw_sha256']=c.digest(saved['raw'])
        saved['label_sha256']=c.digest(saved['labels'])
    if attack=='tokens':saved['audit']['token_ids_sha256']=c.digest('replacement tokens')
    if attack=='settings':
        saved['audit']['engine']['max_num_seqs']+=1
        # Preserve the existing journal-to-journal settings consistency check.
        other=tmp_path/'judge-probe'/f'd1-{"calibration" if cell=="science" else "science"}.json'
        peer=c.load(other)
        peer['audit']['engine']=copy.deepcopy(saved['audit']['engine'])
        c.atomic(other,peer)
    if attack=='timing':saved['seconds']+=1
    c.atomic(journal,saved)
    if cell=='calibration':
        c.atomic(tmp_path/'judge/calibration-labels.json',saved['labels'])
        c.atomic(tmp_path/'judge/calibration-local-raw.json',saved['raw'])
    # Both journals still pass their own schema, hash and binding checks.
    judge.validate_cell(saved,cell=='science','DEFAULT',environment())
    assert saved['binding']==receipt['binding']
    options.clear()
    judge.run(cfg,'gold',101)
    assert len(calls)==8  # No second probe.
    assert len(options)==11  # Complete ten-state lane and calibration regenerated.
    assert all(o['enforce_eager'] and o['gpu_memory_utilization']==.85 for o in options)
    rejected=c.load(tmp_path/'judge-probe/receipt.json')
    assert rejected['decision']=='DEFAULT' and rejected['d1_reuse_rejected'] is True
    assert judge.probe_profile(cfg,'later',environment())=='DEFAULT'
    # The rejected journals remain local evidence; another restart must never
    # rehabilitate them merely because the receipt now records a failure.
    options.clear()
    judge.run(cfg,'gold',101)
    assert len(options)==11 and len(calls)==8
    assert all(o['enforce_eager'] and o['gpu_memory_utilization']==.85 for o in options)
    for name,path in [('science',tmp_path/'judge/gold-101/D1/labels.json'),
                      ('calibration',tmp_path/'judge/calibration-labels.json')]:
        assert c.load(path)==receipt['passes']['D1'][name]['labels']


def complete_receipt():
    value=passes()
    hw=environment()
    for name,run in value.items():
        profile='DEFAULT' if name[0]=='D' else 'FAST'
        for cell in run.values():
            requests=[f'{r["id"]}-{order}' for r in cell['labels'] for order in (0,1)]
            cell.update(label_sha256=c.digest(cell['labels']),malformed=0,retry_count=0,
                audit=dict(token_ids_sha256=c.digest(requests),request_ids=requests,
                    seeds=[int(c.digest([0,r])[:8],16) for r in requests],
                    canonical_orders=[0,1]*len(cell['labels']),grouping='one call, all both-order requests',
                    retry_count=0,gpu_uuid_sha256=c.digest('same'),hardware=hw,engine=profile_engine(profile)))
    return dict(judge.decide(value),schema='trace-judge-probe.v1',passes=value,hardware=hw,
                profiles={p:profile_engine(p) for p in ('DEFAULT','FAST')},
                binding=probe_binding({},value['D1']['science'],value['D1']['calibration'],hw))


@pytest.mark.parametrize('field',[
    'schema','binding','profiles','hardware','diagnostics','repeatable','timings','ratio','O',
    'passes.D1.science.label_sha256','passes.D1.science.malformed','passes.D1.science.retry_count',
    'passes.D1.science.seconds','passes.D1.science.audit.token_ids_sha256',
    'passes.D1.science.audit.request_ids','passes.D1.science.audit.seeds',
    'passes.D1.science.audit.canonical_orders','passes.D1.science.audit.grouping',
    'passes.D1.science.audit.retry_count','passes.D1.science.audit.engine',
    'passes.D1.science.audit.gpu_uuid_sha256','passes.D1.science.audit.hardware',
    'hardware.driver','hardware.gpu_model','hardware.packages.vllm.record_sha256',
])
def test_missing_execution_evidence_persists_default(tmp_path,field):
    cfg=dict(out=str(tmp_path))
    judge.admit(cfg,'first')
    receipt=complete_receipt()
    judge.validate_receipt(receipt)
    target=receipt
    parts=field.split('.')
    for part in parts[:-1]:target=target[part]
    del target[parts[-1]]
    c.atomic(tmp_path/'judge-probe/receipt.json',receipt)
    assert judge.probe_profile(cfg,'later',environment())=='DEFAULT'
    saved=c.load(tmp_path/'judge-probe/receipt.json')
    assert saved['error_type']=='MalformedReceipt' and saved['decision']=='DEFAULT'
    assert judge.decision_receipt(cfg)==saved


@pytest.mark.parametrize('attack',[
    'astra_sparse','token_hash','request_order','seeds','canonical_orders','grouping',
    'gpu','software','label_hash','retry','malformed','timing','matrix','profile','cross_cell_profile',
])
def test_malformed_execution_evidence_persists_default(tmp_path,attack):
    cfg=dict(out=str(tmp_path));judge.admit(cfg,'first')
    r=complete_receipt()
    cell=r['passes']['F2']['science']
    audit=cell['audit']
    if attack=='astra_sparse':r=dict(judge.decide(passes()),passes=passes(),profiles={},hardware=environment())
    if attack=='token_hash':audit['token_ids_sha256']=c.digest('different tokens')
    if attack=='request_order':audit['request_ids'].reverse()
    if attack=='seeds':audit['seeds'][0]+=1
    if attack=='canonical_orders':audit['canonical_orders'][0]=True
    if attack=='grouping':audit['grouping']='one request per call'
    if attack=='gpu':audit['gpu_uuid_sha256']=c.digest('GPU-B')
    if attack=='software':audit['hardware']=dict(environment(),driver=['other'])
    if attack=='label_hash':cell['label_sha256']='0'*64
    if attack=='retry':cell['retry_count']=audit['retry_count']=1
    if attack=='malformed':cell['malformed']=1
    if attack=='timing':cell['seconds']=float('inf')
    if attack=='matrix':r['diagnostics']['science']['matrices']['LIMA-0'][0].pop()
    if attack=='profile':r['profiles']['FAST']['gpu_memory_utilization']=.42
    if attack=='cross_cell_profile':r['passes']['D2']['calibration']['audit']['engine']['max_num_seqs']=512
    # JSON can carry nonfinite input from an external writer.
    path=tmp_path/'judge-probe/receipt.json'
    path.write_text(json.dumps(r))
    assert judge.probe_profile(cfg,'later',environment())=='DEFAULT'
    assert c.load(path)['error_type']=='MalformedReceipt'


def test_receipt_retains_malformed_as_fourth_matrix_label():
    receipt=complete_receipt()
    for run in receipt['passes'].values():
        for cell in run.values():
            cell['labels'][0]['labels'][0]=None
            cell['label_sha256']=c.digest(cell['labels'])
            cell['malformed']=1
    receipt.update(judge.decide(receipt['passes']))
    judge.validate_receipt(receipt)
    assert receipt['decision']=='FAST'
    matrix=receipt['diagnostics']['science']['matrices']['Self-Instruct-0']
    assert len(matrix)==4 and all(len(row)==4 for row in matrix) and matrix[3][3]==1


def test_judge_profiles_isolate_memory_override_and_record_resolved_kwargs(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from kit import inference_lifecycle as lifecycle,m2_gpu,m2_paths,lane_pool
    monkeypatch.setenv('KIT_GPU_MEMORY_FRACTION','0.42')
    monkeypatch.setenv('KIT_LANE_CONCURRENCY','99')
    monkeypatch.setenv('KIT_INFERENCE_WORKLOAD','likelihood')
    monkeypatch.setattr(m2_gpu,'assert_current',lambda:None)
    monkeypatch.setattr(m2_paths,'local',lambda value:value)
    monkeypatch.setattr(lane_pool,'engine_startup',lambda kwargs:nullcontext({}))
    monkeypatch.setattr(lifecycle,'track',lambda engine:engine)
    observed=[]
    def llm(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(llm_engine=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=kwargs.get('max_num_seqs',256))))
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(LLM=llm,__version__='0.18.0'))
    monkeypatch.setattr(lifecycle,'attention_kwargs',lambda kw:dict(kw,attention_backend='FLASH_ATTN'))
    for profile in ('DEFAULT','FAST'):
        requested=profile_engine(profile)
        requested.pop('attention_backend')
        if profile=='DEFAULT':requested.pop('max_num_seqs')
        engine=judge.create_judge_engine(requested)
        resolved=judge.effective_settings(engine,requested)
        judge.validate_settings(resolved,profile)
        assert resolved==profile_engine(profile)
    assert [o['gpu_memory_utilization'] for o in observed]==[.85,.90]
    # An actual engine configuration overrides the request in the audit.
    engine.llm_engine.vllm_config=SimpleNamespace(cache_config=SimpleNamespace(gpu_memory_utilization=.42))
    resolved=judge.effective_settings(engine,requested)
    assert resolved['gpu_memory_utilization']==.42
    with pytest.raises(ValueError,match='frozen settings'):judge.validate_settings(resolved,'FAST')


def test_resolved_profile_drift_records_default_decision(tmp_path,monkeypatch):
    cfg,options=judge_fixture(tmp_path,monkeypatch,True)
    judge.admit(cfg,'gold-101')
    original=judge.probe_cell
    def drift(*args):
        cell=original(*args)
        if args[-1]=='FAST':cell['audit']['engine']['gpu_memory_utilization']=.42
        return cell
    monkeypatch.setattr(judge,'probe_cell',drift)
    judge.run(cfg,'gold',101)
    assert c.load(tmp_path/'judge-probe/receipt.json')['decision']=='DEFAULT'


def test_incomplete_d1_calibration_preserves_science_and_never_reprobes(tmp_path,monkeypatch):
    cfg,options=judge_fixture(tmp_path,monkeypatch,True)
    judge.admit(cfg,'gold-101')
    original=judge.probe_cell
    calls=[]
    def interrupted(*args):
        calls.append(args[-1])
        if len(calls)==2:raise KeyboardInterrupt('D1 calibration interrupted')
        return original(*args)
    monkeypatch.setattr(judge,'probe_cell',interrupted)
    judge.run(cfg,'gold',101)
    assert calls==['DEFAULT','DEFAULT']
    assert len(options)==11
    assert c.load(tmp_path/'judge-probe/receipt.json')['decision']=='DEFAULT'
    options.clear()
    judge.run(cfg,'gold',101)
    assert len(options)==11 and calls==['DEFAULT','DEFAULT']


@pytest.mark.parametrize('attack',['config','lane'])
def test_receipt_claim_binding_is_validated_before_fast_adoption(tmp_path,attack):
    cfg=dict(out=str(tmp_path))
    judge.admit(cfg,'gold-101')
    receipt=complete_receipt()
    receipt['binding']['config_sha256']=c.digest(cfg)
    path=tmp_path/'judge-probe/receipt.json'
    c.atomic(path,receipt)
    assert judge.probe_profile(cfg,'later',environment())=='FAST'
    if attack=='config':receipt['binding']['config_sha256']=c.digest('other config')
    else:receipt['binding']['lane']='other-lane'
    c.atomic(path,receipt)
    assert judge.probe_profile(cfg,'later',environment())=='DEFAULT'
    assert c.load(path)['error_type']=='MalformedReceipt'


@pytest.mark.parametrize('failed_call,persistent',[(n,False) for n in range(2,9)]+[(2,True)])
@pytest.mark.parametrize('failure',[TimeoutError,RuntimeError])
def test_probe_failure_keeps_d1_and_completes_row(tmp_path,monkeypatch,failed_call,persistent,failure):
    import signal
    cfg,_=judge_fixture(tmp_path/'baseline',monkeypatch)
    judge.run(cfg,'gold',101)
    baseline=c.load(Path(cfg['out'])/'judge/gold-101/D1/labels.json')
    cfg,_=judge_fixture(tmp_path/'probe',monkeypatch,True)
    config=tmp_path/'config.json'
    cfg.update(technical_synthetic=True)
    c.atomic(config,cfg)
    cfg['_config_path']=str(config.resolve())
    lane_path=Path(cfg['out'])/'scores/gold-101/lane.json'
    lane=c.load(lane_path);lane['config_sha256']=c.digest(cfg);c.atomic(lane_path,lane)
    judge.admit(cfg,'gold-101')
    original=judge.judge_pairs
    calls=[]
    def generate(*args,**kwargs):
        calls.append((len(args[1]),signal.getitimer(signal.ITIMER_REAL)[0]))
        if len(calls)<=2:
            assert calls[-1][1]==0  # Neither registered D1 call has a probe timer.
        if len(calls)==failed_call or (persistent and len(args[1])==90):
            if failure is TimeoutError and failed_call>2:
                assert calls[-1][1]>0
                signal.raise_signal(signal.SIGALRM)  # Exercise the real deadline handler.
            raise failure('injected calibration/probe failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(judge,'judge_pairs',generate)
    # Synthetic CPU row wrapper retains the real row receipt/error handling.
    row=dict(id='score-gold-101',phase='score',gpus=0,depends=[],study='gold',seed=101)
    monkeypatch.setattr(jobs,'configured_inventory',lambda cfg:dict(rows=[row]))
    monkeypatch.setattr(jobs,'execute',lambda cfg,*args:judge.run(cfg,'gold',101))
    if persistent:
        # A persistent D1 calibration failure ends the row ERROR (resubmit recovers); it is
        # never swallowed, so calibration cannot go missing silently.
        with pytest.raises(BaseException):
            jobs.run(config,row['id'],standin=True)
        state=Path(cfg['out'])/'state'/(row['id']+'.json')
        assert not state.exists() or c.load(state).get('status')!='COMPLETE'
        assert not (Path(cfg['out'])/'judge/calibration-labels.json').exists()
        assert judge.decision_receipt(cfg)['decision']=='DEFAULT'
        assert signal.getitimer(signal.ITIMER_REAL)[0]==0
        return
    jobs.run(config,row['id'],standin=True)
    receipt=judge.decision_receipt(cfg)
    assert receipt['decision']=='DEFAULT'
    assert c.load(Path(cfg['out'])/'judge/gold-101/D1/labels.json')==baseline
    # D1 science is generated exactly once, including when D1 calibration fails.
    assert len(calls)==failed_call+9+(failed_call==2)
    assert jobs.row_receipt(cfg['out'],row['id'])['status']=='COMPLETE'
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0


def test_successful_probe_deadlines_exclude_both_registered_d1_calls(tmp_path,monkeypatch):
    cfg,_=judge_fixture(tmp_path,monkeypatch,True)
    judge.admit(cfg,'gold-101')
    original=judge.probe_cell
    active=False
    observed=[]
    @contextmanager
    def deadline():
        nonlocal active
        active=True
        try:yield
        finally:active=False
    def cell(*args):
        observed.append(active)
        return original(*args)
    monkeypatch.setattr(judge,'probe_deadline',deadline)
    monkeypatch.setattr(judge,'probe_cell',cell)
    judge.run(cfg,'gold',101)
    assert observed==[False,False,True,True,True,True,True,True]


def scorer_config(tmp_path):
    cfg=dict(out=str(tmp_path/'run'),studies=['gold'],technical_synthetic=False,judge_throughput_probe=True)
    config=tmp_path/'config.json'
    c.atomic(config,cfg)
    runtime=dict(cfg,_config_path=str(config.resolve()))
    c.atomic(Path(cfg['out'])/'prepared.json',dict(active_studies=['gold'],audit=dict(unresolved=0),
        registration_sha256=c.sha(c.REG),source_sha256={p.name:c.sha(p) for p in c.KIT.glob('trace_*.py')}))
    return config,runtime


def test_probe_finalizers_are_cpu_afterany_inventory_rows():
    for efficiency in (False,True):
        rows=jobs.configured_inventory(dict(studies=['gold'],judge_throughput_probe=True,
                                             efficiency=efficiency))['rows']
        for scorer in (row for row in rows if row['phase']=='score'):
            finalizer=next(row for row in rows if row['id']=='finalize-'+scorer['id'])
            assert finalizer['gpus']==0 and finalizer['cpus']==1
            assert finalizer['depends']==[]
            assert dag_plan.dependencies(finalizer)==[(scorer['id'],'afterany'),
                                                      ('admit-'+scorer['id'],'afterany')]


def cpu_fifo_probe_schedule(value,cpus):
    """FIFO eligible jobs retain CPUs while polling; first scorer is killed."""
    rows=dag_plan.ordered_rows(dag_plan.plan(dict(trace=value),1))
    pending={key:row for key,row in rows if row['phase'] in ('judge-admit','score','judge-finalize')}
    done={key for key,_ in rows if key not in pending}  # All scientific prerequisites ready.
    running={}
    claim=None
    decision=False
    starts=[]
    queue=[]
    while pending or running:
        changed=False
        # Already eligible admissions queue before the scorer newly released by
        # the first claim, even if that scorer was submitted earlier.
        queue.extend(key for key,row in pending.items() if key not in queue and
                     all(dep in done for dep,_ in row['deps']))
        for key in list(queue):
            row=pending[key]
            if sum(r['cpus'] for r in running.values())+row['cpus']>cpus:continue
            if sum(r['gpus'] for r in running.values())+row['gpus']>8:continue
            starts.append(key)
            queue.remove(key)
            del pending[key]
            changed=True
            if row['phase']=='judge-admit' and claim is None:
                claim=row['id'].removeprefix('admit-')
                done.add(key)
            else:
                running[key]=row
        for key,row in list(running.items()):
            if row['phase']=='judge-admit' and not decision:continue
            if row['phase']=='judge-finalize' and row['id']=='finalize-'+claim:
                decision=True
            # Scorer termination releases CPUs but leaves the decision to afterany.
            del running[key]
            done.add(key)
            changed=True
        if not changed:break
    return dict(starts=starts,pending=pending,running=running,decision=decision)


@pytest.mark.parametrize('cpus',[32,64])
@pytest.mark.parametrize('efficiency',[False,True])
def test_astra_cpu_fifo_admission_fanout_leaves_scorer_and_finalizer_capacity(cpus,efficiency):
    import shlex
    value=jobs.configured_inventory(dict(judge_throughput_probe=True,efficiency=efficiency))
    gates=[r for r in value['rows'] if r['phase']=='judge-admit']
    assert len(gates)==9  # Full three-study, three-seed fan-out.
    assert all(r['cpus']==1 and r['gpus']==0 and r['mem_mb']==256 for r in gates)
    planned=dag_plan.ordered_rows(dag_plan.plan(dict(trace=value),1))
    scorers=[r for _,r in planned if r['phase']=='score']
    finalizers=[r for _,r in planned if r['phase']=='judge-finalize']
    assert all(r['cpus']==8 for r in scorers)
    assert all(r['cpus']==1 and r['gpus']==0 and r['mem_mb']==256 for r in finalizers)
    assert sum(r['cpus'] for r in gates)+8+1<=cpus
    lines=[line for line in dag_plan.commands(dag_plan.plan(dict(trace=value),1)).splitlines()
           if '=$(sbatch ' in line]
    for (_,row),line in zip(planned,lines):
        if row['phase'] in ('judge-admit','judge-finalize'):
            args=shlex.split(line.split('=$(',1)[1][:-1])
            assert {'--cpus-per-task=1','--gpus-per-node=0','--mem=256M'}<=set(args)
    # Reproduce Astra's old resource requests: four/eight waiters occupy all CPUs.
    old=copy.deepcopy(value)
    for row in old['rows']:
        if row['phase']=='judge-admit':row.pop('cpus')
    stalled=cpu_fifo_probe_schedule(old,cpus)
    assert len(stalled['running'])==cpus//8
    assert all(r['phase']=='judge-admit' for r in stalled['running'].values())
    assert not stalled['decision'] and not any(key.startswith('trace:score-') for key in stalled['starts'])
    fixed=cpu_fifo_probe_schedule(value,cpus)
    assert fixed['decision'] and not fixed['pending'] and not fixed['running']
    assert len(fixed['starts'])==27


@pytest.mark.parametrize('memory',[0,-1,True,1.5,'256','256 --gpus-per-node=9',None])
def test_polling_memory_resource_rejects_invalid_or_injected_values(memory):
    row=dict(id='gate',phase='judge-admit',gpus=0,cpus=1,mem_mb=memory)
    with pytest.raises(ValueError,match='mem_mb must be a positive integer'):
        dag_plan.plan(dict(trace=dict(rows=[row])),1)


@pytest.mark.parametrize('outcome',['missing','pending','complete','error','preflight_missing'])
def test_claimed_scorer_always_persists_default_on_exit(tmp_path,monkeypatch,outcome):
    config,cfg=scorer_config(tmp_path)
    judge.admit(cfg,'gold-101')
    def dependency(*args):
        if outcome=='preflight_missing':raise FileNotFoundError('missing admission receipt')
        return dict(config_sha256=c.digest(cfg),technical_synthetic=False)
    monkeypatch.setattr(jobs,'row_receipt',dependency)
    def execute(*args):
        if outcome=='missing':raise FileNotFoundError('CPU token preparation missing')
        if outcome=='error':raise RuntimeError('scorer died')
        return dict(status='PENDING' if outcome=='pending' else 'COMPLETE')
    monkeypatch.setattr(jobs,'execute',execute)
    monkeypatch.setattr(judge,'probe_cell',lambda *a:pytest.fail('finalization started a probe'))
    if outcome in ('error','preflight_missing'):
        with pytest.raises((RuntimeError,FileNotFoundError)):jobs.run(config,'score-gold-101')
    else:
        receipt=jobs.run(config,'score-gold-101')
        assert receipt['status']==('PENDING' if outcome in ('missing','pending') else 'COMPLETE')
    assert c.load(Path(cfg['out'])/'judge-probe/receipt.json')['decision']=='DEFAULT'
    assert judge.admit(cfg,'gold-102')==dict(first=False)


def test_killed_claimed_scorer_is_released_by_cpu_finalizer(tmp_path,monkeypatch):
    import time
    config,cfg=scorer_config(tmp_path)
    monkeypatch.setattr(judge,'probe_cell',lambda *a:pytest.fail('finalizer started a probe'))
    # Admission was cancelled before claiming, and its scorer dependency-cancelled.
    initial=jobs.run(config,'finalize-score-gold-101')
    assert initial['status']=='COMPLETE' and initial['result']==dict(first=False)
    original_receipt=jobs.row_receipt
    def dependency(out,row_id):
        if row_id=='train-gold-101':
            return dict(config_sha256=c.digest(cfg),technical_synthetic=False)
        return original_receipt(out,row_id)
    monkeypatch.setattr(jobs,'row_receipt',dependency)
    # Unchanged resubmission now admits the same lane.
    assert jobs.run(config,'admit-score-gold-101')['result']==dict(first=True)
    ready=tmp_path/'ready'
    code='''
import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from kit import trace_jobs as jobs,trace_common as c
config,ready=sys.argv[2:]
cfg=dict(c.load(config),_config_path=str(Path(config).resolve()))
jobs.row_receipt=lambda *a:dict(config_sha256=c.digest(cfg),technical_synthetic=False)
def execute(*args):
    Path(ready).write_text('scorer running')
    time.sleep(60)
jobs.execute=execute
jobs.run(config,'score-gold-101')
'''
    child=subprocess.Popen([sys.executable,'-I','-c',code,str(ROOT),str(config),str(ready)],
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+10
        while not ready.exists() and child.poll() is None and time.monotonic()<deadline:
            time.sleep(.02)
        assert ready.exists()
        with ThreadPoolExecutor(max_workers=2) as pool:
            entered=threading.Event()
            original_execute=jobs.execute
            def execute(*args):
                entered.set()
                return original_execute(*args)
            monkeypatch.setattr(jobs,'execute',execute)
            future=pool.submit(jobs.run,config,'finalize-score-gold-101')
            waiting=None
            try:
                try:
                    assert entered.wait(5)
                    assert not future.done()  # Cached finalizer still waits for scorer lock.
                    waiting=pool.submit(judge.admit,cfg,'gold-102')
                    assert not waiting.done()
                    assert not (Path(cfg['out'])/'judge-probe/receipt.json').exists()
                finally:
                    child.kill()
                    child.communicate(timeout=5)
                assert child.returncode<0
                receipt=future.result(timeout=5)
                assert waiting.result(timeout=5)==dict(first=False)
            finally:
                # A failing regression must also release its polling test thread.
                if waiting is not None and not waiting.done():
                    judge.finalize_probe(cfg,'gold-101')
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
    path=Path(cfg['out'])/'judge-probe/receipt.json'
    assert receipt['status']=='COMPLETE' and receipt['gpus']==0
    assert receipt['result']==dict(first=True)
    assert jobs.row_receipt(cfg['out'],'finalize-score-gold-101')==receipt
    assert initial['result']==dict(first=False)
    assert c.load(path)['decision']=='DEFAULT'
    assert judge.admit(cfg,'gold-102')==dict(first=False)
    assert not (Path(cfg['out'])/'state/score-gold-101.json').exists()
    assert not (Path(cfg['out'])/'judge-probe/started.json').exists()
    saved=path.read_bytes()
    assert jobs.run(config,'finalize-score-gold-101')['status']=='COMPLETE'
    assert path.read_bytes()==saved


@pytest.mark.parametrize('decision',['DEFAULT','FAST','VOID'])
def test_cancelled_scorer_finalizer_preserves_other_claims_and_existing_decisions(tmp_path,monkeypatch,decision):
    config,cfg=scorer_config(tmp_path)
    monkeypatch.setattr(judge,'probe_cell',lambda *a:pytest.fail('finalizer started a probe'))
    assert jobs.run(config,'finalize-score-gold-101')['result']==dict(first=False)
    judge.admit(cfg,'gold-101')
    path=Path(cfg['out'])/'judge-probe/receipt.json'
    jobs.run(config,'finalize-score-gold-102')
    assert not path.exists()
    # Scheduler cancellation can leave no scorer attempt or receipt at all.
    assert not (Path(cfg['out'])/'attempts/score-gold-101').exists()
    receipt=complete_receipt()
    receipt['binding']['config_sha256']=c.digest(cfg)
    if decision=='DEFAULT':
        for name in ('F1','F2'):
            receipt['passes'][name]['science']['seconds']=3
    elif decision=='VOID':
        cell=receipt['passes']['F2']['science']
        cell['labels'][0]['labels'][0]=0
        cell['label_sha256']=c.digest(cell['labels'])
    receipt.update(judge.decide(receipt['passes']))
    judge.validate_receipt(receipt)
    assert receipt['decision']==decision
    c.atomic(path,receipt)
    saved=path.read_bytes()
    for _ in range(2):
        current=jobs.run(config,'finalize-score-gold-101')
        assert current['status']=='COMPLETE' and current['result']==dict(first=True)
        assert path.read_bytes()==saved
    assert judge.probe_profile(cfg,'gold-102',environment())==('FAST' if decision=='FAST' else 'DEFAULT')


def test_cached_scorer_still_finalizes_later_claim(tmp_path,monkeypatch):
    config,cfg=scorer_config(tmp_path)
    original_receipt=jobs.row_receipt
    def dependency(out,row_id):
        if row_id=='admit-score-gold-101':
            return dict(config_sha256=c.digest(cfg),technical_synthetic=False)
        return original_receipt(out,row_id)
    monkeypatch.setattr(jobs,'row_receipt',dependency)
    monkeypatch.setattr(jobs,'execute',lambda *a:dict(status='COMPLETE'))
    jobs.run(config,'score-gold-101')
    judge.admit(cfg,'gold-101')
    monkeypatch.setattr(jobs,'execute',lambda *a:pytest.fail('cached scorer executed again'))
    monkeypatch.setattr(judge,'probe_cell',lambda *a:pytest.fail('cached scorer started a probe'))
    jobs.run(config,'score-gold-101')
    assert c.load(Path(cfg['out'])/'judge-probe/receipt.json')['decision']=='DEFAULT'
    assert judge.admit(cfg,'gold-102')==dict(first=False)


def test_cancelled_claimed_scorer_with_no_attempt_defaults(tmp_path,monkeypatch):
    config,cfg=scorer_config(tmp_path)
    judge.admit(cfg,'gold-101')
    monkeypatch.setattr(judge,'probe_cell',lambda *a:pytest.fail('finalizer started a probe'))
    assert jobs.run(config,'finalize-score-gold-101')['status']=='COMPLETE'
    assert c.load(Path(cfg['out'])/'judge-probe/receipt.json')['decision']=='DEFAULT'
    assert judge.admit(cfg,'gold-102')==dict(first=False)
    assert not (Path(cfg['out'])/'attempts/score-gold-101').exists()
