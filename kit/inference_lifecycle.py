"""Close registered inference workers while their logging pipe is still open."""
_engines = []


def engine_kwargs(kwargs):
    """Only override the memory reservation; leave all scientific kwargs intact."""
    import math
    import os
    result=dict(kwargs)
    value=os.environ.get('KIT_GPU_MEMORY_FRACTION')
    concurrency=int(os.environ.get('KIT_LANE_CONCURRENCY','1'))
    if concurrency>1:
        from kit.lane_pool import capacity
        if concurrency>capacity(dict(engine=dict(backend='vllm',model=result.get('model'),max_model_len=result.get('max_model_len')))) or result.get('tensor_parallel_size',1)!=1:
            raise ValueError('concurrent engine model/context is outside the qualified Qwen3-8B profile')
    if value is not None:
        fraction=float(value)
        if not math.isfinite(fraction) or not 0<fraction<1:
            raise ValueError('KIT_GPU_MEMORY_FRACTION must be finite and between 0 and 1')
        result['gpu_memory_utilization']=fraction
    return result


def track(engine):
    _engines.append(engine)
    return engine


def attention_kwargs(kwargs):
    """Name the batch-invariant backend in EngineArgs as well as the legacy env."""
    import inspect
    import os
    result = dict(kwargs)
    if os.environ.get('VLLM_BATCH_INVARIANT') != '1':
        return result
    config = result.get('attention_config')
    configured = config.get('backend') if isinstance(config, dict) else getattr(config, 'backend', None)
    backend = result.get('attention_backend') or configured or os.environ.get('VLLM_ATTENTION_BACKEND') or 'FLASH_ATTN'
    backend_name = getattr(backend, 'name', backend)
    for value in (backend, result.get('attention_backend'), configured):
        if value is not None and getattr(value, 'name', value) != 'FLASH_ATTN':
            raise ValueError(f'batch-invariant mode requires FLASH_ATTN; resolved backend: {getattr(value, "name", value)}')
    os.environ['VLLM_ATTENTION_BACKEND'] = backend_name
    if result.get('attention_backend') or configured:
        return result
    try:
        from vllm.engine.arg_utils import EngineArgs
        parameters = inspect.signature(EngineArgs).parameters
    except ImportError:
        parameters = {'attention_config': None}  # SDK doubles; real 0.18 has this field.
    if 'attention_config' in parameters:
        if config is None or isinstance(config,dict):
            result['attention_config'] = dict(config or {}, backend=backend)
        else:
            from copy import copy
            result['attention_config'] = copy(config)
            result['attention_config'].backend = backend
    elif 'attention_backend' in parameters:
        result['attention_backend'] = backend
    # Older vLLM without either EngineArgs field uses the env set above.
    return result


def create_llm(*args, _kit_workload=None, **kwargs):
    kwargs=engine_kwargs(kwargs)
    import os
    workload=_kit_workload or os.environ.get('KIT_INFERENCE_WORKLOAD')
    concurrency=int(os.environ.get('KIT_LANE_CONCURRENCY','1'))
    if workload=='likelihood':
        if concurrency>2:
            raise ValueError('likelihood engines require K<=2')
        kwargs['gpu_memory_utilization']=.42
    from kit.m2_gpu import assert_current
    import vllm
    from vllm import LLM
    kwargs=attention_kwargs(kwargs)
    modern=getattr(vllm,'__version__','').startswith('0.18.')
    if (concurrency>1 or os.environ.get('KIT_LANE_JOB')) and modern:
        if kwargs.get('worker_cls','auto')!='auto':
            raise ValueError('shared lanes require the kit memory-profile worker')
        kwargs['worker_cls']='kit.inference_worker.MemoryProfileWorker'
    from kit.m2_paths import local
    for key in ('model','tokenizer'):
        if key in kwargs:kwargs[key]=str(local(kwargs[key]))
    from kit.lane_pool import engine_startup
    with engine_startup(kwargs) as startup:
        assert_current()
        engine=track(LLM(*args, **kwargs))
    # Lane controllers collect engine configuration without touching scorer output.
    if os.environ.get('KIT_ENGINE_RECEIPT_DIR'):
        import json
        import time
        from pathlib import Path
        root=Path(os.environ['KIT_ENGINE_RECEIPT_DIR']);root.mkdir(parents=True,exist_ok=True)
        config=kwargs.get('attention_config')
        backend=kwargs.get('attention_backend') or (config.get('backend') if isinstance(config,dict) else getattr(config,'backend',None))
        actual=getattr(getattr(engine,'llm_engine',None),'vllm_config',None)
        backend=getattr(getattr(actual,'attention_config',None),'backend',None) or backend
        if backend is None and not modern:backend=os.environ.get('VLLM_ATTENTION_BACKEND')
        value=dict(gpu_memory_utilization=kwargs.get('gpu_memory_utilization'),
            max_model_len=kwargs.get('max_model_len'),max_num_seqs=kwargs.get('max_num_seqs'),
            tensor_parallel_size=kwargs.get('tensor_parallel_size',1),
            uuid=os.environ.get('M2_ASSIGNED_GPU_UUID'),pid=os.getpid(),
            attention_backend=getattr(backend,'name',backend),
            workload=workload,worker_cls=kwargs.get('worker_cls','auto'),startup=startup)
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


def profile_with_retry(worker, measure, snapshot):
    """Re-measure only vLLM's specific neighbour-release assertion, once.

    The replacement baseline already includes the loaded model. Keep the original
    fractional reservation, and bound the resulting KV budget by current free
    memory minus the re-measured activation/non-torch growth. Model weights must
    not be subtracted a second time from that remaining free-memory bound.
    """
    try:
        return measure()
    except AssertionError as error:
        message=str(error)
        if not ('Error in memory profiling.' in message and
                'release GPU memory while vLLM is profiling' in message):
            raise
        if getattr(worker,'_kit_profile_remeasured',False):raise
    worker._kit_profile_remeasured=True
    original=worker.init_snapshot
    fresh=snapshot(device=worker.device)
    worker.init_snapshot=fresh
    import json
    import os
    import time
    from pathlib import Path
    event=dict(event='memory_profile_remeasure',pid=os.getpid(),t=time.time(),
               initial_free_bytes=original.free_memory,free_bytes=fresh.free_memory,
               requested_bytes=worker.requested_memory)
    print(json.dumps(event,sort_keys=True),flush=True)
    root=os.environ.get('KIT_ENGINE_RECEIPT_DIR')
    if root:
        path=Path(root)/'profiles';path.mkdir(parents=True,exist_ok=True)
        (path/(str(os.getpid())+'-'+str(time.time_ns())+'.json')).write_text(json.dumps(event,sort_keys=True)+'\n')
    measured=measure()  # A second disturbance, OOM or unrelated failure propagates.
    free_bound=fresh.free_memory-max(0,worker.peak_activation_memory)-max(0,worker.non_torch_memory)
    worker.available_kv_cache_memory_bytes=min(measured,int(free_bound))
    return worker.available_kv_cache_memory_bytes
