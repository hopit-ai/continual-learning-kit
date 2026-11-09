"""Native single-turn observers and concurrent request-seeded siblings."""
import asyncio
from copy import deepcopy
from contextvars import ContextVar

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from kit.nm_common import need


_finish_reason = ContextVar('ts_finish_reason', default=None)
_request = ContextVar('ts_attempt0_request', default=None)


class FinishObserver:
    def __init__(self, manager):
        self.manager = manager

    def __getattr__(self, name):
        return getattr(self.manager, name)

    async def generate(self, *args, **kwargs):
        _request.set(deepcopy(dict(prompt_token_ids=kwargs['prompt_ids'],
                                   sampling_params=kwargs['sampling_params'])))
        output = await self.manager.generate(*args, **kwargs)
        _finish_reason.set(getattr(output, 'finish_reason', None))
        return output


class TSTargetLoop(SingleTurnAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_manager = FinishObserver(self.server_manager)

    async def run(self, sampling_params, **kwargs):
        token = _finish_reason.set(None)
        request_token = _request.set(None)
        try:
            output = await super().run(dict(sampling_params), **kwargs)
            output.extra_fields['ts_finish_reason'] = _finish_reason.get()
            output.extra_fields['ts_attempt0_request'] = _request.get()
            return output
        finally:
            _finish_reason.reset(token)
            _request.reset(request_token)


class TSAuxiliaryLoop(TSTargetLoop):
    async def run(self, sampling_params, **kwargs):
        plan = kwargs['ts_plan']
        need(len(plan) == 7 and [r['index'] for r in plan] == list(range(1, 8)), 'invalid auxiliary plan')
        run = super().run
        outputs = await asyncio.gather(*(run(dict(sampling_params, seed=r['seed']), **kwargs) for r in plan))
        records = [dict(token_ids=o.response_ids, seed=r['seed'], index=r['index'],
                        finish_reason=o.extra_fields['ts_finish_reason']) for r, o in zip(plan, outputs)]
        output = outputs[0]
        output.extra_fields['ts_attempts'] = records
        return output
