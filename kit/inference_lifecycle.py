"""Close registered inference workers while their logging pipe is still open."""
_engines = []


def track(engine):
    _engines.append(engine)
    return engine


def create_llm(*args, **kwargs):
    from kit.m2_gpu import assert_current
    assert_current()
    from vllm import LLM
    from kit.m2_paths import local
    for key in ('model','tokenizer'):
        if key in kwargs:kwargs[key]=str(local(kwargs[key]))
    return track(LLM(*args, **kwargs))


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
