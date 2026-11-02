"""CPU-only process scheduling with a fake nvidia-smi; no CUDA/Modal calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from kit import lane_pool as pool,inference_lifecycle as lifecycle


@pytest.fixture
def gpu(tmp_path,monkeypatch):
    tool=tmp_path/'nvidia-smi'
    tool.write_text('#!'+sys.executable+'\nimport os\nprint(os.environ.get("FAKE_GPU", "GPU-test"))\n')
    tool.chmod(0o755)
    monkeypatch.setenv('PATH',str(tmp_path)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.delenv('M2_ASSIGNED_GPU_UUID',raising=False)
    return tool


def plan(tmp_path,sources,requires=None):
    path=tmp_path/'plan.json'
    path.write_text(json.dumps([dict(argv=[sys.executable,'-I','-c',s],requires=requires or []) for s in sources]))
    return path


def test_concurrent_bound_order_fraction_uuid_and_stagger(tmp_path,gpu):
    source="import os,time,json; print(json.dumps(dict(uuid=os.environ['M2_ASSIGNED_GPU_UUID'],visible=os.environ['CUDA_VISIBLE_DEVICES'],fraction=os.environ['KIT_GPU_MEMORY_FRACTION']))); time.sleep(.35)"
    record=pool.run(plan(tmp_path,[source]*5),2,tmp_path/'out',.12)
    assert record['status']=='COMPLETE' and record['uuid']=='GPU-test' and record['K']==2
    events=[]
    for i,c in enumerate(record['commands']):
        assert c['index']==i and c['exit_code']==0 and c['uuid']=='GPU-test'
        data=json.loads((tmp_path/'out'/str(i)/'command.log').read_text())
        assert data==dict(uuid='GPU-test',visible='0',fraction='0.42')
        events.extend([(c['start'],1),(c['end'],-1)])
    active=peak=0
    for _,change in sorted(events):active+=change;peak=max(peak,active)
    assert peak==2
    assert record['commands'][1]['start']-record['commands'][0]['start']>=.10
    assert record['commands'][2]['start']>=record['commands'][0]['end']
    assert json.loads((tmp_path/'out/lane.json').read_text())==record
    with pytest.raises(FileExistsError):pool.run(tmp_path/'plan.json',2,tmp_path/'out')


def test_pending_preflight_does_not_even_query_gpu(tmp_path,monkeypatch):
    monkeypatch.setattr(pool,'gpu_uuid',lambda:pytest.fail('GPU must not be queried for missing prerequisites'))
    record=pool.run(plan(tmp_path,["raise AssertionError('must not start')"],[str(tmp_path/'missing')]),2,tmp_path/'out')
    assert record['status']=='PENDING' and record['uuid'] is None
    assert all(c['start'] is None for c in record['commands'])


@pytest.mark.parametrize('code,status',[(7,'FAILED'),(75,'PENDING'),(3,'PENDING')])
def test_failure_never_retries_or_starts_more_but_peers_finish(tmp_path,gpu,code,status):
    commands=['import time; time.sleep(.4)',f'import sys; sys.exit({code})',"raise AssertionError('must not start')"]
    record=pool.run(plan(tmp_path,commands),2,tmp_path/'out',0)
    assert record['status']==status
    assert [c['exit_code'] for c in record['commands']]==[0,code,None]
    assert record['commands'][0]['end']>record['commands'][1]['end']


@pytest.mark.parametrize('message',['PENDING: checkpoint absent','NOT READY: missing binding','PENDING'])
def test_existing_zero_exit_pending_commands_cannot_complete_lane(tmp_path,gpu,message):
    record=pool.run(plan(tmp_path,['print('+repr(message)+')','pass']),1,tmp_path/'out',0)
    assert record['status']=='PENDING'
    assert record['commands'][0]['exit_code']==0 and record['commands'][0]['status']=='PENDING'
    assert record['commands'][1]['start'] is None


@pytest.mark.parametrize('visible,response', [('0,1','GPU-test'),('','GPU-test'),('-1','GPU-test'),
    ('0','GPU-a\nGPU-b'),('0','garbage'),('GPU-other','GPU-test')])
def test_reject_bad_visibility(tmp_path,gpu,monkeypatch,visible,response):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visible);monkeypatch.setenv('FAKE_GPU',response)
    record=pool.run(plan(tmp_path,['pass']),2,tmp_path/'out',0)
    assert record['status']=='FAILED' and record['commands'][0]['start'] is None


def test_uuid_change_between_commands_fails_closed(tmp_path,gpu,monkeypatch):
    values=iter(['GPU-test','GPU-test','GPU-other'])
    monkeypatch.setattr(pool,'gpu_uuid',lambda:next(values))
    record=pool.run(plan(tmp_path,['pass','pass']),1,tmp_path/'out',0)
    assert record['status']=='FAILED' and 'UUID changed' in record['error']
    assert record['commands'][0]['exit_code']==0 and record['commands'][1]['start'] is None


def test_timeout_reaps_children_and_marks_failed(tmp_path,gpu):
    record=pool.run(plan(tmp_path,['import time; time.sleep(30)']),2,tmp_path/'out',0,timeout_seconds=2)
    assert record['status']=='FAILED' and 'deadline' in record['error']
    assert record['commands'][0]['exit_code']<0


def test_cli_pending_and_three_way_fraction(tmp_path,gpu):
    path=plan(tmp_path,['import os; assert os.environ["KIT_GPU_MEMORY_FRACTION"]=="0.31"']*3)
    done=subprocess.run([sys.executable,'-I',str(Path(pool.__file__)),'--plan',str(path),
        '--concurrency','3','--stagger-seconds','0','--out',str(tmp_path/'out')],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    path.write_text(json.dumps([dict(argv=['never-executed'],requires=[str(tmp_path/'missing')])]))
    done=subprocess.run([sys.executable,'-I',str(Path(pool.__file__)),'--plan',str(path),
        '--concurrency','2','--out',str(tmp_path/'pending')],capture_output=True,text=True)
    assert done.returncode==75 and done.stdout.strip()=='PENDING'


@pytest.mark.parametrize('fraction',[None,'0.42','0.28'])
def test_override_at_shared_constructor_preserves_every_other_kwarg(tmp_path,monkeypatch,fraction):
    from kit import m2_gpu,m2_paths
    monkeypatch.delenv('KIT_GPU_MEMORY_FRACTION',raising=False)
    if fraction is not None:monkeypatch.setenv('KIT_GPU_MEMORY_FRACTION',fraction)
    monkeypatch.setenv('KIT_ENGINE_RECEIPT_DIR',str(tmp_path/'engines'))
    monkeypatch.setenv('M2_ASSIGNED_GPU_UUID','GPU-test')
    monkeypatch.setattr(m2_gpu,'assert_current',lambda:None)
    monkeypatch.setattr(m2_paths,'local',lambda path:path)
    fake=ModuleType('vllm');calls=[]
    class Engine:
        def __init__(self,**kw):calls.append(kw)
        def shutdown(self):pass
    fake.LLM=Engine;monkeypatch.setitem(sys.modules,'vllm',fake)
    mark=lifecycle.marker()
    try:
        for default in (.85,.9):
            kwargs=dict(model='model',tensor_parallel_size=1,dtype='bfloat16',max_model_len=12288,
                max_num_seqs=1,gpu_memory_utilization=default,enforce_eager=True,seed=0)
            lifecycle.create_llm(**kwargs)
            assert calls[-1]==dict(kwargs,gpu_memory_utilization=float(fraction) if fraction else default)
    finally:lifecycle.shutdown_since(mark)
    receipts=[json.loads(p.read_text()) for p in (tmp_path/'engines').glob('*.json')]
    assert sorted(r['gpu_memory_utilization'] for r in receipts)==sorted(float(fraction) if fraction else v for v in (.85,.9))
    assert all(r['uuid']=='GPU-test' and r['max_model_len']==12288 for r in receipts)


@pytest.mark.parametrize('value',['','0','1','-0.1','nan','inf','garbage'])
def test_override_fail_closed(value,monkeypatch):
    monkeypatch.setenv('KIT_GPU_MEMORY_FRACTION',value)
    with pytest.raises(ValueError):lifecycle.engine_kwargs({'gpu_memory_utilization':.85})


def test_no_override_preserves_unset_library_default(monkeypatch):
    monkeypatch.delenv('KIT_GPU_MEMORY_FRACTION',raising=False)
    original=dict(model='model',seed=0)
    assert lifecycle.engine_kwargs(original)==original and 'gpu_memory_utilization' not in original


@pytest.mark.parametrize('fraction',[None,'0.42'])
def test_generated_panel_records_actual_effective_engine(tmp_path,monkeypatch,fraction):
    from types import SimpleNamespace as NS
    from kit import score_forgetting as general,m2_gpu,m2_paths
    monkeypatch.delenv('KIT_GPU_MEMORY_FRACTION',raising=False)
    if fraction:monkeypatch.setenv('KIT_GPU_MEMORY_FRACTION',fraction)
    monkeypatch.setenv('V4_TELEMETRY','1')
    monkeypatch.setattr(m2_gpu,'assert_current',lambda:None);monkeypatch.setattr(m2_paths,'local',lambda p:p)
    member=dict(panel='ifeval',id='one',prompt='Please answer')
    monkeypatch.setattr(general,'load_panel',lambda:([member],'panel-hash'))
    monkeypatch.setattr(general,'machine_fingerprint',lambda *a:dict(technical_synthetic=True))
    monkeypatch.setattr(general,'model_identity',lambda *a:dict(technical_synthetic=True))
    monkeypatch.setattr(general,'grade',lambda *a:dict(ifeval=dict(n=1,correct=1,parsed=1,median_output_chars=6)))
    fake=ModuleType('vllm');fake.__version__='fixture';fake.SamplingParams=lambda **kw:kw;calls=[]
    class Engine:
        def __init__(self,**kw):calls.append(kw)
        def generate(self,*a,**kw):
            return [NS(prompt_token_ids=[1],outputs=[NS(text='answer',token_ids=[2],finish_reason='stop')])]
        def shutdown(self):pass
    fake.LLM=Engine;monkeypatch.setitem(sys.modules,'vllm',fake)
    tokenizer=NS(apply_chat_template=lambda *a,**kw:'rendered')
    transformers=ModuleType('transformers');transformers.AutoTokenizer=NS(from_pretrained=lambda *a,**kw:tokenizer)
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    out=tmp_path/'score';mark=lifecycle.marker()
    try:general.cmd_generate(NS(model=model,out=out,max_new_tokens=2048,max_model_len=12288))
    finally:lifecycle.shutdown_since(mark)
    expected=float(fraction) if fraction else .85
    assert calls[0]['gpu_memory_utilization']==expected
    assert calls[0]['max_model_len']==12288 and calls[0]['enforce_eager'] and calls[0]['seed']==0
    assert json.loads((out/'forgetting.json').read_text())['engine']['gpu_memory_utilization']==expected
    assert json.loads((out/'engine-status.json').read_text())['engine']['gpu_memory_utilization']==expected


def test_teacher_reload_receipt_uses_effective_fraction(monkeypatch):
    from types import SimpleNamespace as NS
    from kit import v4_teacher,m2_gpu,m2_paths
    monkeypatch.setenv('KIT_GPU_MEMORY_FRACTION','.28')
    monkeypatch.delenv('V4_PHASE0_INFERENCE',raising=False);monkeypatch.delenv('V4_BATCH_FALLBACK_CAUSE',raising=False)
    monkeypatch.setattr(m2_gpu,'assert_current',lambda:None);monkeypatch.setattr(m2_paths,'local',lambda p:p)
    fake=ModuleType('vllm');fake.SamplingParams=lambda **kw:kw;calls=[]
    class Engine:
        def __init__(self,**kw):calls.append(kw)
        def get_tokenizer(self):return NS()
        def shutdown(self):pass
    fake.LLM=Engine;monkeypatch.setitem(sys.modules,'vllm',fake)
    mark=lifecycle.marker()
    try:generator=v4_teacher.VLLMGenerator('model',2048,tensor_parallel_size=1,allow_batch_fallback=False)
    finally:lifecycle.shutdown_since(mark)
    assert calls[0]['gpu_memory_utilization']==.28 and calls[0]['max_model_len']==8192
    assert generator.reload_timing['batch_policy']['gpu_memory_utilization']==.28


def test_legacy_repair_generation_uses_same_override_helper(tmp_path,monkeypatch):
    import runpy
    from types import SimpleNamespace as NS
    monkeypatch.setenv('KIT_GPU_MEMORY_FRACTION','.42')
    fake=ModuleType('vllm');fake.SamplingParams=lambda **kw:kw;calls=[]
    class Engine:
        def __init__(self,**kw):calls.append(kw)
        def generate(self,prompts,*a):return [NS(outputs=[NS(text='unchanged')]) for _ in prompts]
    fake.LLM=Engine;monkeypatch.setitem(sys.modules,'vllm',fake)
    transformers=ModuleType('transformers')
    transformers.AutoTokenizer=NS(from_pretrained=lambda *a,**kw:NS(apply_chat_template=lambda *a,**kw:'rendered'))
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    prompts=tmp_path/'prompts.jsonl';prompts.write_text(''.join(json.dumps(dict(prompt=str(i)))+'\n' for i in range(20)))
    out=tmp_path/'targets.jsonl'
    module=runpy.run_path(str(Path(pool.__file__).parent/'repair/make_targets.py'))
    assert module['main'](['--model','model','--prompts',str(prompts),'--out',str(out)])==0
    assert calls==[dict(model='model',dtype='bfloat16',gpu_memory_utilization=.42,max_model_len=2048)]
    assert json.loads(Path(str(out)+'.stats.json').read_text())['engine']['gpu_memory_utilization']==.42


def test_uuid_timeout_retries_with_thirty_second_budget(monkeypatch):
    calls=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    def run(command,**kw):
        calls.append(kw['timeout'])
        if len(calls)<3:raise subprocess.TimeoutExpired(command,kw['timeout'])
        return type('Result',(),{'stdout':'GPU-measured\n'})()
    monkeypatch.setattr(pool.subprocess,'run',run)
    assert pool.gpu_uuid()=='GPU-measured' and calls==[30,30,30]


def test_transient_uuid_failure_never_terminates_active_peers(tmp_path,gpu,monkeypatch):
    values=iter(['GPU-test','GPU-test',None,None])
    monkeypatch.setattr(pool,'gpu_uuid',lambda:next(values))
    record=pool.run(plan(tmp_path,['import time;time.sleep(.2)','pass']),2,tmp_path/'out',0)
    assert record['status']=='COMPLETE'
    assert [c['exit_code'] for c in record['commands']]==[0,0]
    assert record['uuid_query_warnings']==[1,'completion']


def test_missing_initial_uuid_is_pending_without_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(pool,'gpu_uuid',lambda:None)
    record=pool.run(plan(tmp_path,['raise AssertionError']),2,tmp_path/'out',0)
    assert record['status']=='PENDING' and record['commands'][0]['start'] is None


def test_malformed_later_uuid_telemetry_never_terminates_peers(tmp_path,gpu,monkeypatch):
    calls=[]
    def measured():
        calls.append(None)
        if len(calls)>2:raise pool.GPUQueryUnavailable('malformed NVML response')
        return 'GPU-test'
    monkeypatch.setattr(pool,'gpu_uuid',measured)
    record=pool.run(plan(tmp_path,['import time;time.sleep(.2)','pass']),2,tmp_path/'out',0)
    assert record['status']=='COMPLETE' and [c['exit_code'] for c in record['commands']]==[0,0]
    assert record['uuid_query_warnings']==[1,'completion']
