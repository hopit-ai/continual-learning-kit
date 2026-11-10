"""V7 operator fixes: CPU SDK doubles, real subprocesses and measured telemetry."""
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from kit import inference_lifecycle as life, lane_pool as pool, m2_gpu, m2_paths

ROOT = Path(__file__).resolve().parents[1]
DISTURBED = ('Error in memory profiling. Initial free memory increased: other processes '
             'release GPU memory while vLLM is profiling during initialization.')


@pytest.fixture
def sdk(monkeypatch):
    for key in ('KIT_LANE_JOB', 'KIT_LANE_CONCURRENCY', 'KIT_GPU_MEMORY_FRACTION',
                'KIT_INFERENCE_WORKLOAD', 'VLLM_BATCH_INVARIANT', 'VLLM_ATTENTION_BACKEND'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('VLLM_BATCH_INVARIANT', '0')
    monkeypatch.setenv('VLLM_ATTENTION_BACKEND', 'FLASH_ATTN')
    calls = []
    class Engine:
        def __init__(self, **kwargs): calls.append(kwargs)
        def shutdown(self): pass
    fake = ModuleType('vllm'); fake.LLM = Engine; fake.__version__ = '0.18.0'
    fake.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, 'vllm', fake)
    monkeypatch.setattr(m2_gpu, 'assert_current', lambda: None)
    monkeypatch.setattr(m2_paths, 'local', lambda p: p)
    mark = life.marker()
    yield calls
    life.shutdown_since(mark)


def test_batch_invariant_start_passes_backend_and_receipts_preserve_scientific_kwargs(tmp_path, monkeypatch, sdk):
    monkeypatch.setenv('VLLM_BATCH_INVARIANT', '1')
    monkeypatch.setenv('KIT_ENGINE_RECEIPT_DIR', str(tmp_path/'receipts'))
    kwargs = dict(model='Qwen/Qwen3-8B', max_model_len=12288, dtype='bfloat16',
                  max_num_seqs=1, enforce_eager=True, seed=0, gpu_memory_utilization=.85)
    life.create_llm(**kwargs)
    assert sdk == [dict(kwargs, attention_config={'backend': 'FLASH_ATTN'})]
    receipt = json.loads(next((tmp_path/'receipts').glob('*.json')).read_text())
    assert receipt['attention_backend'] == 'FLASH_ATTN'
    assert os.environ['VLLM_ATTENTION_BACKEND'] == 'FLASH_ATTN'


@pytest.mark.parametrize('field', ['attention_config', 'attention_backend', 'legacy'])
def test_backend_follows_available_engineargs_signature(monkeypatch, sdk, field):
    args = ModuleType('vllm.engine.arg_utils')
    if field == 'attention_config':
        class EngineArgs:
            def __init__(self, attention_config=None): pass
    elif field == 'attention_backend':
        class EngineArgs:
            def __init__(self, attention_backend=None): pass
    else:
        class EngineArgs:
            def __init__(self, model=None): pass
    args.EngineArgs = EngineArgs
    monkeypatch.setitem(sys.modules, 'vllm.engine.arg_utils', args)
    monkeypatch.setenv('VLLM_BATCH_INVARIANT', '1')
    life.create_llm(model='model')
    if field == 'legacy':
        assert sdk[0] == dict(model='model')
    else:
        assert sdk[0][field] == ({'backend': 'FLASH_ATTN'} if field == 'attention_config' else 'FLASH_ATTN')
    assert os.environ['VLLM_ATTENTION_BACKEND'] == 'FLASH_ATTN'


@pytest.mark.parametrize('kwargs', [
    {}, {'attention_backend':'FLASHINFER'}, {'attention_config':{'backend':'FLASHINFER'}},
    {'attention_config':NS(backend='FLASHINFER')},
    {'attention_backend':'FLASH_ATTN','attention_config':{'backend':'FLASHINFER'}},
])
def test_batch_invariant_refuses_other_resolved_backends(monkeypatch,sdk,kwargs):
    monkeypatch.setenv('VLLM_BATCH_INVARIANT','1')
    monkeypatch.setenv('VLLM_ATTENTION_BACKEND','FLASHINFER')
    with pytest.raises(ValueError,match='requires FLASH_ATTN.*FLASHINFER'):
        life.create_llm(model='model',**kwargs)
    assert not sdk


@pytest.mark.parametrize('kwargs', [
    {'attention_backend':'FLASH_ATTN'}, {'attention_config':{'backend':'FLASH_ATTN','other':'kept'}},
    {'attention_config':NS(backend='FLASH_ATTN')},
])
def test_explicit_flash_attention_overrides_legacy_environment(monkeypatch,sdk,kwargs):
    monkeypatch.setenv('VLLM_BATCH_INVARIANT','1')
    monkeypatch.setenv('VLLM_ATTENTION_BACKEND','FLASHINFER')
    life.create_llm(model='model',**kwargs)
    assert sdk==[dict(model='model',**kwargs)]
    assert os.environ['VLLM_ATTENTION_BACKEND']=='FLASH_ATTN'


def test_batch_invariant_accepts_named_backend_enum(monkeypatch,sdk):
    from enum import Enum
    class Backend(Enum): FLASH_ATTN=1
    monkeypatch.setenv('VLLM_BATCH_INVARIANT','1')
    life.create_llm(model='model',attention_backend=Backend.FLASH_ATTN)
    assert sdk[0]['attention_backend'] is Backend.FLASH_ATTN
    assert os.environ['VLLM_ATTENTION_BACKEND']=='FLASH_ATTN'


def test_non_batch_invariant_backend_is_unchanged(monkeypatch,sdk):
    monkeypatch.setenv('VLLM_ATTENTION_BACKEND','FLASHINFER')
    life.create_llm(model='model',attention_config={'backend':'FLASHINFER'})
    assert sdk==[dict(model='model',attention_config={'backend':'FLASHINFER'})]


def test_engine_start_with_installed_vllm_engineargs_signature(monkeypatch):
    """Import the actual SDK contract when installed; construct only a CPU double."""
    vllm = pytest.importorskip('vllm')
    from vllm.engine.arg_utils import EngineArgs
    parameters = inspect.signature(EngineArgs).parameters
    for key in ('KIT_LANE_JOB', 'KIT_LANE_CONCURRENCY', 'KIT_INFERENCE_WORKLOAD', 'KIT_GPU_MEMORY_FRACTION'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('VLLM_BATCH_INVARIANT', '1')
    monkeypatch.setenv('VLLM_ATTENTION_BACKEND', 'FLASH_ATTN')
    monkeypatch.setattr(m2_gpu, 'assert_current', lambda: None)
    monkeypatch.setattr(m2_paths, 'local', lambda p: p)
    calls = []
    monkeypatch.setattr(vllm, 'LLM', lambda **kw: calls.append(kw) or NS(shutdown=lambda: None))
    mark = life.marker()
    try:
        life.create_llm(model='CPU-contract-only', seed=0)
        kw = calls[0]
        field = 'attention_config' if 'attention_config' in kw else 'attention_backend'
        assert field in parameters, 'batch-invariant engines must pass an EngineArgs backend'
        assert kw[field] == ({'backend': 'FLASH_ATTN'} if field == 'attention_config' else 'FLASH_ATTN')
    finally:
        life.shutdown_since(mark)


def test_shared_start_waits_for_measured_share_and_selects_worker(tmp_path, monkeypatch, sdk):
    monkeypatch.setenv('KIT_LANE_JOB', 'independent-lane')
    monkeypatch.setenv('KIT_LANE_CONCURRENCY', '2')
    monkeypatch.setenv('KIT_GPU_MEMORY_FRACTION', '.42')
    monkeypatch.setenv('M2_ASSIGNED_GPU_UUID', 'GPU-v7-wait')
    monkeypatch.setenv('KIT_ENGINE_RECEIPT_DIR', str(tmp_path/'receipts'))
    monkeypatch.setattr(pool, 'ENGINE_LOCK_ROOT', tmp_path/'locks')
    events = []; samples = iter([(419, 1000), None, (420, 1000)])
    def memory(uuid):
        assert not sdk and uuid == 'GPU-v7-wait'
        item = next(samples); events.append(item)
        if item is None: raise pool.GPUQueryUnavailable('neighbour teardown')
        return item
    monkeypatch.setattr(pool, 'gpu_memory', memory)
    monkeypatch.setattr(pool.time, 'sleep', lambda seconds: None)
    life.create_llm(model='Qwen/Qwen3-8B', max_model_len=12288, seed=0)
    assert len(events) == 3 and len(sdk) == 1
    assert sdk[0]['worker_cls'] == 'kit.inference_worker.MemoryProfileWorker'
    receipt = json.loads(next((tmp_path/'receipts').glob('*.json')).read_text())
    assert receipt['startup']['free_mib'] == 420 and receipt['startup']['fraction'] == .42


@pytest.mark.parametrize('measurement', [None, (419, 1000)])
def test_memory_admission_timeout_never_starts_engine(tmp_path, monkeypatch, sdk, measurement):
    monkeypatch.setenv('KIT_LANE_JOB', 'lane')
    monkeypatch.setenv('M2_ASSIGNED_GPU_UUID', 'GPU-v7-timeout')
    monkeypatch.setenv('KIT_ENGINE_START_TIMEOUT_SECONDS', '.02')
    monkeypatch.setattr(pool, 'ENGINE_LOCK_ROOT', tmp_path/'locks')
    def memory(uuid):
        if measurement is None: raise pool.GPUQueryUnavailable('empty')
        return measurement
    monkeypatch.setattr(pool, 'gpu_memory', memory)
    cause='query unavailable' if measurement is None else 'free/total 419/1000 MiB'
    with pytest.raises(TimeoutError, match=cause):
        life.create_llm(model='model', gpu_memory_utilization=.42)
    assert not sdk


@pytest.fixture
def startup_clock(tmp_path,monkeypatch):
    clock=[0.]
    monkeypatch.setenv('KIT_LANE_JOB','lane')
    monkeypatch.setenv('M2_ASSIGNED_GPU_UUID','GPU-budget-test')
    monkeypatch.setenv('KIT_ENGINE_START_TIMEOUT_SECONDS','1')
    monkeypatch.setattr(pool,'ENGINE_LOCK_ROOT',tmp_path/'locks')
    monkeypatch.setattr(pool.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(pool.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    return clock


@pytest.mark.parametrize('k',[1,2,3])
def test_startup_lock_wait_has_own_k_minus_one_budget(monkeypatch,startup_clock,k):
    monkeypatch.setenv('KIT_LANE_CONCURRENCY',str(k))
    def locked(*args):raise BlockingIOError('another constructor')
    monkeypatch.setattr(pool.fcntl,'flock',locked)
    monkeypatch.setattr(pool,'gpu_memory',lambda uuid:pytest.fail('memory queried before lock'))
    with pytest.raises(TimeoutError,match=f'lock wait.*{k-1}s budget, K={k}'):
        with pool.engine_startup({'gpu_memory_utilization':.42}):pytest.fail('admitted without lock')
    assert k-1<=startup_clock[0]<k-1+.11


def test_memory_gets_full_deadline_after_long_lock_queue(monkeypatch,startup_clock):
    monkeypatch.setenv('KIT_LANE_CONCURRENCY','3')
    events=[]
    def lock(fd,operation):
        if operation==pool.fcntl.LOCK_UN:events.append('unlock');return
        if startup_clock[0]<1.5:raise BlockingIOError('cold constructors')
        events.append(('lock',startup_clock[0]))
    monkeypatch.setattr(pool.fcntl,'flock',lock)
    def memory(uuid):
        elapsed=startup_clock[0]-events[0][1]
        return (420 if elapsed>=.8 else 419),1000
    monkeypatch.setattr(pool,'gpu_memory',memory)
    with pool.engine_startup({'gpu_memory_utilization':.42}) as receipt:
        assert receipt['free_mib']==420 and receipt['wait_s']>=2.3
    assert events[-1]=='unlock'


@pytest.mark.parametrize('measurement',[None,(419,1000)])
def test_memory_timeout_after_lock_releases_lock_and_names_cause(monkeypatch,startup_clock,measurement):
    monkeypatch.setenv('KIT_LANE_CONCURRENCY','2')
    events=[]
    def lock(fd,operation):
        if operation==pool.fcntl.LOCK_UN:events.append('unlock');return
        if startup_clock[0]<.8:raise BlockingIOError('cold constructor')
        events.append('lock')
    def memory(uuid):
        if measurement is None:raise pool.GPUQueryUnavailable('empty')
        return measurement
    monkeypatch.setattr(pool.fcntl,'flock',lock)
    monkeypatch.setattr(pool,'gpu_memory',memory)
    cause='query unavailable' if measurement is None else 'free/total 419/1000 MiB'
    with pytest.raises(TimeoutError,match=cause):
        with pool.engine_startup({'gpu_memory_utilization':.42}):pytest.fail('not admitted')
    assert 1.8<=startup_clock[0]<2.1 and events==['lock','unlock']


def test_startup_lock_root_ignores_job_temp_environment(tmp_path,monkeypatch):
    for key in ('TMPDIR','TMP','TEMP'):monkeypatch.setenv(key,str(tmp_path))
    assert pool.ENGINE_LOCK_ROOT.parent==Path('/tmp')


def test_startup_lock_is_physical_and_cross_process(tmp_path):
    events = tmp_path/'events.jsonl'
    code = '''
import json,os,sys,time
from pathlib import Path
from kit import lane_pool as p
p.gpu_memory=lambda uuid:(1000,1000)
def event(kind):
 with Path(sys.argv[2]).open('a') as f:f.write(json.dumps(dict(kind=kind,gpu=os.environ['M2_ASSIGNED_GPU_UUID'],lane=os.environ['KIT_LANE_JOB'],t=time.monotonic()))+'\\n')
with p.engine_startup(dict(gpu_memory_utilization=.42)):
 event('start');time.sleep(.3);event('end')
'''
    children = []
    for lane, gpu, visible in [('first', 'GPU-a', '0'), ('second', 'GPU-a', '5'), ('third', 'GPU-b', '0')]:
        job_tmp=tmp_path/lane;job_tmp.mkdir()
        env = dict(os.environ, KIT_LANE_JOB=lane, KIT_LANE_CONCURRENCY='3',
                   M2_ASSIGNED_GPU_UUID=gpu+'-'+tmp_path.name, CUDA_VISIBLE_DEVICES=visible,
                   TMPDIR=str(job_tmp),TMP=str(job_tmp),TEMP=str(job_tmp))
        children.append(subprocess.Popen([sys.executable, '-c', code, str(tmp_path), str(events)], env=env))
    try:
        assert [p.wait(timeout=10) for p in children] == [0, 0, 0]
    finally:
        for p in children:
            if p.poll() is None: p.kill(); p.wait()
    rows = [json.loads(s) for s in events.read_text().splitlines()]
    intervals = {lane: {r['kind']: r['t'] for r in rows if r['lane']==lane} for lane in ('first','second','third')}
    a, b = sorted([intervals['first'], intervals['second']], key=lambda x:x['start'])
    assert b['start'] >= a['end']
    third = intervals['third']
    assert any(max(third['start'], i['start']) < min(third['end'], i['end']) for i in (a,b))


def test_memory_query_falls_back_to_uuid_scoped_inventory(monkeypatch):
    calls = []
    def query(command, **kw):
        calls.append(command)
        return '' if any(v.startswith('--id=') for v in command) else 'GPU-other, 999, 1000\nGPU-a, 420, 1000\n'
    monkeypatch.setattr(pool.subprocess, 'check_output', query)
    assert pool.gpu_memory('GPU-a') == (420, 1000)
    assert '--id=GPU-a' in calls[0] and not any(v.startswith('--id=') for v in calls[1])
    with pytest.raises(pool.GPUQueryUnavailable): pool.gpu_memory('GPU-missing')


@pytest.mark.parametrize('free,expected_kv', [(500,240),(300,240),(200,180),(80,60)])
def test_disturbed_profile_remeasures_once_and_bounds_reservation(tmp_path, monkeypatch, free, expected_kv):
    monkeypatch.setenv('KIT_ENGINE_RECEIPT_DIR', str(tmp_path))
    worker = NS(init_snapshot=NS(free_memory=100), requested_memory=420, device='cuda:0',
                peak_activation_memory=20, non_torch_memory=0)
    calls = []
    def measure():
        calls.append(worker.init_snapshot.free_memory)
        if len(calls)==1: raise AssertionError(DISTURBED)
        return worker.requested_memory - 160 - 20
    assert life.profile_with_retry(worker, measure, lambda **kw:NS(free_memory=free)) == expected_kv
    assert calls == [100,free] and worker.requested_memory == 420
    assert worker.available_kv_cache_memory_bytes == expected_kv
    assert json.loads(next((tmp_path/'profiles').glob('*.json')).read_text())['event'] == 'memory_profile_remeasure'
    with pytest.raises(AssertionError, match='memory profiling'):
        life.profile_with_retry(worker, lambda: (_ for _ in ()).throw(AssertionError(DISTURBED)), lambda **kw:pytest.fail('third measurement'))


@pytest.mark.parametrize('message', [DISTURBED, 'unrelated assertion'])
def test_second_disturbance_and_unrelated_failures_propagate(message):
    worker = NS(init_snapshot=NS(free_memory=100), requested_memory=420, device='cuda:0',
                peak_activation_memory=20, non_torch_memory=0)
    calls = []
    def fail():
        calls.append(True); raise AssertionError(message)
    with pytest.raises(AssertionError, match='profiling|unrelated'):
        life.profile_with_retry(worker, fail, lambda **kw:NS(free_memory=500))
    assert len(calls) == (2 if message==DISTURBED else 1)


@pytest.mark.parametrize('k', [1,2,3])
def test_likelihood_constructor_uses_42_and_refuses_three(monkeypatch, sdk, k):
    monkeypatch.setenv('KIT_LANE_CONCURRENCY', str(k))
    monkeypatch.setenv('KIT_GPU_MEMORY_FRACTION', '.31')
    if k==3:
        with pytest.raises(ValueError, match='K<=2'):
            life.create_llm(_kit_workload='likelihood', model='Qwen/Qwen3-8B', max_model_len=12288)
        assert not sdk
    else:
        life.create_llm(_kit_workload='likelihood', model='Qwen/Qwen3-8B', max_model_len=12288)
        assert sdk[0]['gpu_memory_utilization']==.42


@pytest.mark.parametrize('argv', [
    ['m3_score.py','--panel-set','knowledge-ll'], ['m2_score.py','--panel-set','hellaswag'],
    ['x_score.py','--panel-set','hellaswag'],
    ['x_references.py','--panel-set','test','--panel-set','hellaswag'], ['m2_format.py','--checkpoint','base'],
    ['nm_score.py','--panel','knowledge'],
])
def test_all_likelihood_command_profiles_cap_two(tmp_path, argv):
    engine = dict(backend='vllm', model='Qwen/Qwen3-8B', max_model_len=12288)
    assert pool.capacity(dict(argv=argv, engine=engine))==2
    command = pool.invocation([dict(argv=argv, engine=engine)], tmp_path/'pool')
    assert command[command.index('--concurrency')+1]=='2'
    assert pool.capacity(dict(argv=['score.py','--panel-set','general-v1'],engine=engine))==3


def test_tee_inherited_pipe_teardown_takes_five_seconds(tmp_path):
    code = '''
import subprocess,sys,time
from pathlib import Path
from kit.p3a_common import logging
start=time.monotonic()
child=None
try:
 with logging(Path(sys.argv[1]),'bounded'):
  child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
 elapsed=time.monotonic()-start
 assert 4.5<=elapsed<9,elapsed
finally:
 if child is not None:child.terminate();child.wait()
'''
    done = subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,timeout=12)
    assert done.returncode==0, done.stderr
    assert 'tee did not exit in 5s' in done.stderr


def test_math_test_label_generates_only_explicit_frozen_items(tmp_path, monkeypatch, sdk):
    from kit import eval_bed as bed
    from kit.beds import math_screen
    monkeypatch.delenv('KIT_V4_ARM', raising=False)
    args = NS(bed='math_screen', root=None, split='test', model=str(tmp_path/'model'),
              max_new_tokens=2048, max_model_len=12288, out=str(tmp_path/'scores'),
              limit=None, heldout_n=None, allow_subset=False)
    assert bed.split_of(args)=='test'
    with pytest.raises(bed.EvalBedError, match='metadata only'):
        bed.items_of(math_screen, args)
    monkeypatch.setattr(bed,'items_of',lambda *a:pytest.fail('must not load a test split'))
    monkeypatch.setattr(bed,'machine_fingerprint',lambda *a:{})
    fake = sys.modules['vllm']
    class Engine(fake.LLM):
        def generate(self, prompts, params):
            assert len(prompts)==1 and params['max_tokens']==2048
            return [NS(prompt_token_ids=[1], outputs=[NS(text=r'Worked. \boxed{2}',token_ids=[2],finish_reason='stop')])]
    fake.LLM=Engine
    trans=ModuleType('transformers')
    trans.AutoTokenizer=NS(from_pretrained=lambda *a,**kw:NS(apply_chat_template=lambda *a,**kw:'prompt'))
    monkeypatch.setitem(sys.modules,'transformers',trans)
    (tmp_path/'model').mkdir();(tmp_path/'model/config.json').write_text('{}')
    items=[dict(id='frozen-hard-math-id',prompt='Compute 1+1',ground_truth='2')]
    assert bed.cmd_generate(args,frozen_items=items)==0
    result=json.loads((tmp_path/'scores/bed-score.json').read_text())
    assert result['split']=='test' and result['per_item']=={'frozen-hard-math-id':1}
    assert result['items_sha256']==bed.items_digest(items)


def test_math_test_call_sites_supply_frozen_items():
    """Both M2 and M3 route this label through the frozen-item generation API."""
    import ast
    found=[]
    for path in (ROOT/'kit').glob('*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call):continue
            kw={k.arg:k.value for k in node.keywords}
            if not all(isinstance(kw.get(k),ast.Constant) and kw[k].value==v for k,v in [('bed','math_screen'),('split','test')]):continue
            found.append(path.name)
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr=='cmd_generate' and any(k.arg=='frozen_items' for k in n.keywords)]
            assert calls,path
    assert set(found)=={'m2_score.py','m3_score.py'}


def test_finished_diagnostic_engine_shutdown_precedes_likelihood_child(tmp_path, monkeypatch):
    from kit import next_hellaswag as score, eval_bed as bed, m2_format, inference_subprocess
    rows=[dict(id=str(i),prompt='Pick an ending',gold='A') for i in range(100)]
    panel=tmp_path/'hellaswag';panel.mkdir()
    (panel/'panel.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(score,'frozen_next',lambda:None)
    monkeypatch.setattr(score,'machine',lambda *a:{})
    monkeypatch.setattr(score,'checkpoint',lambda *a,**kw:{})
    monkeypatch.setattr(score,'frozen_files',lambda *a:dict(technical_synthetic=False,panel_sha256='frozen'))
    monkeypatch.setattr(score,'freeze',lambda *a,**kw:None)
    monkeypatch.setattr(score,'prepared_index',lambda *a:'0')
    events=[];mark=life.marker()
    # An unrelated registered engine is outside this scorer's lifecycle scope.
    sentinel=life.track(NS(shutdown=lambda:events.append('sentinel')))
    def generate(args, **kw):
        life.track(NS(shutdown=lambda:events.append('shutdown')))
        root=Path(args.out);root.mkdir(parents=True)
        (root/'responses.jsonl').write_text(''.join(json.dumps(dict(id=r['id'],response='Answer: A',output_tokens=3,finish_reason='stop'))+'\n' for r in rows))
    monkeypatch.setattr(bed,'cmd_generate',generate)
    def child(command,**kw):
        assert events==['shutdown'] and life._engines[-1] is sentinel
        assert kw['env']['KIT_INFERENCE_WORKLOAD']=='likelihood'
        events.append('likelihood-child')
        result=Path(command[command.index('--result')+1]);result.write_text(json.dumps(dict(value=['unchanged'])))
        return NS(returncode=0,args=command)
    monkeypatch.setattr(inference_subprocess.subprocess,'run',child)
    try:
        score.score(NS(out=tmp_path,gpu='0',standin=False,model='model',checkpoint_receipt=None,arm='C0',seed=101,step=40))
        assert m2_format.likelihoods('model',rows,'tokenizer')==['unchanged']
        assert events==['shutdown','likelihood-child']
    finally:life.shutdown_since(mark)


@pytest.mark.parametrize('base',[False,True])
def test_m3_diagnostic_process_exits_before_likelihood_and_keeps_prepared_identity(tmp_path,monkeypatch,base):
    from kit import m3_score as m3
    source=dict(model='model',receipt='receipt.json',base=base)
    observed=dict(model_file_hashes={'config.json':'CPU-fixture'})
    monkeypatch.setattr(m3,'proof_for',lambda *a:dict(uuid='GPU-m3'))
    monkeypatch.setattr(m3,'resolved_gpu',lambda a:a.gpu)
    monkeypatch.setattr(m3,'checked',lambda *a:(source,observed))
    monkeypatch.setenv('M3_SCORE_READY','CPU-prepared-fixture')
    (tmp_path/'hellaswag').mkdir()
    (tmp_path/'hellaswag/manifest.json').write_text(json.dumps(dict(files={},panel_sha256='frozen-panel')))
    root=m3.unit_root(tmp_path,'K',101,'finqa',40,'hellaswag')
    child_code='''
import argparse,json,os,runpy,sys
from pathlib import Path
from types import SimpleNamespace as NS
from kit import m3_score as m3,next_hellaswag as hs,p3b_common as common,inference_lifecycle as life
source=json.loads(os.environ['DIAGNOSTIC_SOURCE'])
observed=dict(model_file_hashes={'config.json':'CPU-fixture'})
def proof(out,gpu,seed,standin):
 assert gpu=='2' and seed==101 and not standin
 for fd in json.loads(os.environ['M3_GPU_LOCK_FDS']):os.fstat(fd)
 return dict(uuid='GPU-m3')
def checked(out,arm,seed,stage,step,standin):
 assert (arm,seed,stage,step,standin)==('K',101,'finqa',40,False)
 return source,observed
def score(a):
 assert a.arm=='K.finqa' and a.model==source['model']
 assert a.checkpoint_receipt==(None if source['base'] else Path(source['receipt']))
 assert hs.checkpoint(a.model,a.checkpoint_receipt,base=False)==observed
 life.track(NS(shutdown=lambda:None))
 root=m3.unit_root(a.out,'K',a.seed,'finqa',a.step,'hellaswag');root.mkdir(parents=True)
 (root/'diagnostic-pid').write_text(str(os.getpid()))
 (root/'manifest.json').write_text(json.dumps(dict(files={})))
def entry(job,configure,name):
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.set_defaults(standin=False)
 configure(p);job(p.parse_args());return 0
m3.proof_for=proof;m3.resolved_gpu=lambda a:a.gpu;m3.checked=checked
hs.score=score;common.entry=entry
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    monkeypatch.setenv('DIAGNOSTIC_SOURCE',json.dumps(source))
    calls=[];run=subprocess.run;mark=life.marker()
    def child(command,**kw):
        assert Path(command[1]).name=='m2_cached_score.py' and command[2]=='m3_hellaswag'
        assert kw['check'] is True and kw['pass_fds']==(lease.fileno(),)
        assert '--standin' not in command
        calls.append('diagnostic')
        return run([command[0],'-c',child_code,*command[1:]],**kw)
    monkeypatch.setattr(subprocess,'run',child)
    def likelihood(*args):
        pid=int((root/'diagnostic-pid').read_text())
        assert pid!=os.getpid() and life.marker()==mark
        with pytest.raises(ProcessLookupError):os.kill(pid,0)
        calls.append('likelihood')
        return dict(policy_sha256='unchanged-policy',validation_sha256=None)
    monkeypatch.setattr(m3,'format_free',likelihood)
    with (tmp_path/'lease').open('w') as lease:
        monkeypatch.setenv('M3_GPU_LOCK_FDS',json.dumps([lease.fileno()]))
        m3.score_job(NS(out=tmp_path,arm='K',seed=101,stage='finqa',step=40,panel_set='hellaswag',
                        gpu='2',standin=False,panel_root=None))
    assert calls==['diagnostic','likelihood']
    meta=json.loads((root/'m3-identity.json').read_text())
    assert meta['model']==observed and meta['panel_sha256']=='frozen-panel'


def test_m3_diagnostic_failure_never_starts_likelihood(tmp_path,monkeypatch):
    from kit import m3_score as m3
    monkeypatch.setattr(m3,'proof_for',lambda *a:dict(uuid='GPU-m3'))
    monkeypatch.setattr(m3,'resolved_gpu',lambda a:a.gpu)
    monkeypatch.setattr(m3,'checked',lambda *a:(dict(model='model',receipt=None,base=True),{}))
    monkeypatch.setattr(m3,'format_free',lambda *a:pytest.fail('likelihood after failed diagnostic'))
    def failed(command,**kw):raise subprocess.CalledProcessError(2,command)
    monkeypatch.setattr(subprocess,'run',failed)
    with pytest.raises(subprocess.CalledProcessError) as error:
        m3.score_job(NS(out=tmp_path,arm='base',seed=101,stage='finqa',step=0,panel_set='hellaswag',
                        gpu='0',standin=False,panel_root=None))
    assert error.value.returncode==2
    assert not (m3.unit_root(tmp_path,'base',101,'finqa',0,'hellaswag')/'m3-identity.json').exists()


def test_lifecycle_authority_reasons_describe_engine_configuration():
    from kit import v6_infra
    doc=v6_infra.validate(json.loads((ROOT/v6_infra.AUTHORITY).read_text()))
    rows=[r for r in doc['files'] if r['path']=='kit/inference_lifecycle.py']
    assert rows and all(r['reason']=='engine backend/memory configuration; scoring definitions unchanged' for r in rows)


def test_worker_class_remeasure_runs_in_worker_process_api(monkeypatch, sdk):
    import importlib.util
    memory=ModuleType('vllm.utils.mem_utils')
    memory.MemorySnapshot=lambda **kw:NS(free_memory=500)
    worker=ModuleType('vllm.v1.worker.gpu_worker')
    class Worker:
        def __init__(self):
            self.init_snapshot=NS(free_memory=100);self.device='cuda:0';self.requested_memory=420;self.calls=0
            self.peak_activation_memory=20;self.non_torch_memory=0
        def determine_available_memory(self):
            self.calls+=1
            if self.calls==1:raise AssertionError(DISTURBED)
            return self.requested_memory-180
    worker.Worker=Worker
    monkeypatch.setitem(sys.modules,memory.__name__,memory)
    monkeypatch.setitem(sys.modules,worker.__name__,worker)
    spec=importlib.util.spec_from_file_location('cpu_memory_worker',ROOT/'kit/inference_worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    engine_worker=module.MemoryProfileWorker()
    assert engine_worker.determine_available_memory()==240 and engine_worker.calls==2


def test_live_likelihood_pool_overrides_requested_three_with_two_and_42(tmp_path,monkeypatch):
    binary=tmp_path/'nvidia-smi'
    binary.write_text('#!'+sys.executable+'\nprint("GPU-live-likelihood")\n');binary.chmod(0o755)
    monkeypatch.setenv('PATH',str(tmp_path)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    source='import os,time;assert os.environ["KIT_LANE_CONCURRENCY"]=="2";assert os.environ["KIT_GPU_MEMORY_FRACTION"]=="0.42";time.sleep(.1)'
    commands=[dict(argv=[sys.executable,'-c',source,'--panel-set','hellaswag'],
        engine=dict(backend='vllm',model='Qwen/Qwen3-8B',max_model_len=12288)) for _ in range(3)]
    path=tmp_path/'plan.json';path.write_text(json.dumps(commands))
    receipt=pool.run(path,3,tmp_path/'pool',stagger_seconds=0)
    assert receipt['status']=='COMPLETE' and receipt['K']==2 and receipt['requested_K']==3
    assert receipt['gpu_memory_fraction']==.42


@pytest.mark.skipif(not (ROOT/'management/private-names.json').is_file(), reason='export rebuilding requires private manager input')
def test_public_export_ships_worker_and_runs_operator_regressions(tmp_path,monkeypatch):
    from scripts import publish_kit as publisher
    monkeypatch.setattr(publisher,'licence_text',lambda:'CPU export fixture; no network')
    target=publisher.build(tmp_path/'public',tag='kit-message-v7')
    assert (target/'kit/inference_worker.py').read_bytes()==(ROOT/'kit/inference_worker.py').read_bytes()
    assert (target/'tests/test_v7_operator_fixes.py').read_bytes()==Path(__file__).read_bytes()
    (target/'tmp').mkdir()
    done=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_v7_operator_fixes.py',
        '-k','not public_export','--basetemp=tmp/v7-fx-export-tests','-p','no:cacheprovider'],
        cwd=target,capture_output=True,text=True,timeout=60,
        env={k:v for k,v in os.environ.items() if k != 'PYTEST_ADDOPTS'})
    assert done.returncode==0,done.stdout+done.stderr
    assert 'passed' in done.stdout and 'deselected' in done.stdout
