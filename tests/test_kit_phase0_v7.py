"""Partner's phase 0 v6 run, 10 October: four TP-2 teacher engines capturing CUDA graphs at once took 27 minutes,
then the first sample_tokens RPC timed out. Generation engines are now always eager; Astra's review of v7 added
report-failure propagation, a fresh-process batch-invariance fallback and tenfold distributed deadlines."""
import dataclasses
import json
import os
import sys
import types
from pathlib import Path
import pytest
from kit import v4_teacher as t
from kit import v4_phase0 as phase0

KIT=Path(t.__file__).resolve().parent


@pytest.fixture
def fake_vllm(monkeypatch):
    seen=[]
    class LLM:
        fail=False
        def __init__(self,**kwargs):
            seen.append({**kwargs,'invariant':os.environ['VLLM_BATCH_INVARIANT']})
            if LLM.fail:raise RuntimeError('invariant kernels unavailable')
        def get_tokenizer(self):return object()
    @dataclasses.dataclass
    class EngineArgs:
        distributed_timeout_seconds:int|None=None
    module=types.ModuleType('vllm');module.LLM=LLM;module.SamplingParams=object;module.__version__='0.18.0'
    engine=types.ModuleType('vllm.engine');args=types.ModuleType('vllm.engine.arg_utils');args.EngineArgs=EngineArgs
    for name,mod in (('vllm',module),('vllm.engine',engine),('vllm.engine.arg_utils',args)):monkeypatch.setitem(sys.modules,name,mod)
    for key in ('VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS','V4_BATCH_FALLBACK_CAUSE','VLLM_BATCH_INVARIANT'):monkeypatch.delenv(key,raising=False)
    return LLM,seen


@pytest.mark.parametrize('tp',[1,2])
def test_teacher_and_rewriter_engines_are_eager(fake_vllm,tp):
    _,seen=fake_vllm
    engine=t.VLLMGenerator('m',2048,tensor_parallel_size=tp)
    assert seen[-1]['enforce_eager'] is True and engine.batch_policy['enforce_eager'] is True
    assert os.environ['VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS']=='3000'
    assert seen[-1].get('distributed_timeout_seconds')==(6000 if tp==2 else None)
    if tp==2:
        import torch.distributed.distributed_c10d as c10d
        assert (c10d.default_pg_timeout.total_seconds(),c10d.default_pg_nccl_timeout.total_seconds())==(18000,6000)


def test_batch_fallback_restarts_in_a_fresh_process_with_the_flag_off(fake_vllm,monkeypatch):
    LLM,seen=fake_vllm;LLM.fail=True;calls=[]
    events=[]
    def execve(path,argv,env):events.append('exec');calls.append(env);raise SystemExit(0)
    monkeypatch.setattr(t.os,'execve',execve)
    import psutil
    class Worker:
        def terminate(self):events.append('terminate')
        def kill(self):events.append('kill')
    monkeypatch.setattr(psutil,'Process',lambda:types.SimpleNamespace(children=lambda recursive:[Worker()]))
    monkeypatch.setattr(psutil,'wait_procs',lambda procs,timeout:(events.append('reap') or [],[]))
    with pytest.raises(SystemExit):t.VLLMGenerator('m',2048,tensor_parallel_size=2,allow_batch_fallback=True)
    assert seen[-1]['invariant']=='1'
    assert events[:3]==['terminate','reap','reap'] and events[-1]=='exec'
    assert calls[0]['VLLM_BATCH_INVARIANT']=='0' and 'invariant kernels unavailable' in calls[0]['V4_BATCH_FALLBACK_CAUSE']
    # The restarted process: flag off before vLLM is imported, fallback labelled, a second failure is final.
    LLM.fail=False;monkeypatch.setenv('V4_BATCH_FALLBACK_CAUSE',calls[0]['V4_BATCH_FALLBACK_CAUSE'])
    engine=t.VLLMGenerator('m',2048,tensor_parallel_size=2,allow_batch_fallback=True)
    assert seen[-1]['invariant']=='0'
    assert engine.batch_policy['batch_invariant'] is False and engine.batch_policy['engine_starts']==2
    LLM.fail=True
    with pytest.raises(RuntimeError):t.VLLMGenerator('m',2048,tensor_parallel_size=2,allow_batch_fallback=True)


def test_incomplete_report_fails_its_row(tmp_path,monkeypatch):
    monkeypatch.setattr(phase0,'operation',lambda *a,**k:{'status':'incomplete','reasons':['missing checkpoint receipt']})
    assert phase0.main(['report','--work',str(tmp_path),'--row','report'])!=0
    assert not (tmp_path/'v4/report-status/report.json').exists()
    assert 'missing checkpoint receipt' in json.loads((tmp_path/phase0.BASE/'failures'/'report.json').read_text())['cause']


def test_complete_report_passes(tmp_path,monkeypatch):
    (tmp_path/phase0.BASE).mkdir(parents=True)
    monkeypatch.setattr(phase0,'operation',lambda *a,**k:{'status':'technical pass'})
    assert phase0.main(['report','--work',str(tmp_path),'--row','report'])==0


def test_distributed_deadlines_are_tenfold():
    for launcher in ('run_sdpo_toolalpaca.sh','run_sdft.sh'):
        assert 'ARGV+=("actor_rollout_ref.nccl_timeout=6000")' in (KIT/launcher).read_text()
    for scorer in ('eval_bed.py','score_forgetting.py'):
        assert 'timeout=30)' not in (KIT/scorer).read_text()
