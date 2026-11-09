"""TS-owned observation bridge for raw vLLM finish reasons.

An engine proxy captures the raw finish reason in each request's async context
before the pinned server constructs TokenOutput. The inherited server receives
the original sampling parameters and supplies tokens and log probabilities.
"""
from contextvars import ContextVar
from pathlib import Path

import ray
import verl
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from kit.ts_pinned import check

check(Path(verl.__file__).resolve().parent.parent)

_finish_reason = ContextVar('ts_vllm_finish_reason', default=None)


class TSTokenOutput(TokenOutput):
    finish_reason: str | None = None


class FinishReasonEngine:
    def __init__(self, engine):
        self.engine = engine

    def __getattr__(self, name):
        return getattr(self.engine, name)

    async def generate(self, *args, **kwargs):
        async for output in self.engine.generate(*args, **kwargs):
            if output.outputs:
                _finish_reason.set(output.outputs[0].finish_reason)
            yield output


class TSHttpServer(vLLMHttpServer):
    async def generate(self, *args, **kwargs):
        if not isinstance(self.engine, FinishReasonEngine):
            self.engine = FinishReasonEngine(self.engine)
        token = _finish_reason.set(None)
        try:
            output = await super().generate(*args, **kwargs)
            return TSTokenOutput(**output.model_dump(), finish_reason=_finish_reason.get())
        finally:
            _finish_reason.reset(token)


class TSReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TSHttpServer)


class TSAgentLoopManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.rollout_replica_class = TSReplica
        super().__init__(*args, **kwargs)
