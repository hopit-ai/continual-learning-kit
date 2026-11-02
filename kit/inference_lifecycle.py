"""Close registered inference workers while their logging pipe is still open."""
_engines = []


def engine_kwargs(kwargs):
    """Only override the memory reservation; leave all scientific kwargs intact."""
    import math
    import os
    result=dict(kwargs)
    value=os.environ.get('KIT_GPU_MEMORY_FRACTION')
    if value is not None:
        fraction=float(value)
        if not math.isfinite(fraction) or not 0<fraction<1:
            raise ValueError('KIT_GPU_MEMORY_FRACTION must be finite and between 0 and 1')
        result['gpu_memory_utilization']=fraction
    return result


def track(engine):
    _engines.append(engine)
    return engine


def create_llm(*args, **kwargs):
    kwargs=engine_kwargs(kwargs)
    from kit.m2_gpu import assert_current
    assert_current()
    from vllm import LLM
    from kit.m2_paths import local
    for key in ('model','tokenizer'):
        if key in kwargs:kwargs[key]=str(local(kwargs[key]))
    engine=track(LLM(*args, **kwargs))
    # Lane controllers collect engine configuration without touching scorer output.
    import os
    if os.environ.get('KIT_ENGINE_RECEIPT_DIR'):
        import json
        import time
        from pathlib import Path
        root=Path(os.environ['KIT_ENGINE_RECEIPT_DIR']);root.mkdir(parents=True,exist_ok=True)
        value=dict(gpu_memory_utilization=kwargs.get('gpu_memory_utilization'),
            max_model_len=kwargs.get('max_model_len'),max_num_seqs=kwargs.get('max_num_seqs'),
            tensor_parallel_size=kwargs.get('tensor_parallel_size',1),
            uuid=os.environ.get('M2_ASSIGNED_GPU_UUID'),pid=os.getpid())
        with (root/(str(os.getpid())+'-'+str(time.time_ns())+'.json')).open('x') as stream:
            json.dump(value,stream,sort_keys=True,allow_nan=False)
    return engine


def shutdown_since(mark=0):
    failures = []
    while len(_engines) > mark:
        engine = _engines.pop()
        inner = getattr(engine, 'llm_engine', None)
        targets = (engine, inner, getattr(inner, 'engine_core', None),
                   getattr(inner, 'model_executor', None))
        shutdown = next((getattr(x, 'shutdown') for x in targets
                         if callable(getattr(x, 'shutdown', None))), None)
        try:
            if shutdown is None:
                raise RuntimeError('inference engine has no supported shutdown method')
            shutdown()
        except Exception as exc:
            failures.append(exc)
    if failures:
        raise RuntimeError('inference shutdown failed') from failures[0]


def marker():
    return len(_engines)
