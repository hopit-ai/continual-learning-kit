"""TS scoring extension for hard MATH. Other panels use the incumbent scorer."""
import os

from kit import nm_score
from kit.nm_common import KIT, need, ready, load, atomic, digest, sha


def authority(panel):
    if panel != 'math_hard':
        return nm_score.scorer_authority(panel)
    return dict(method='FROZEN-GREEDY', sha256=digest({name: sha(KIT / name) for name in
        ('ts_score.py', 'beds/math_screen.py', 'eval_bed.py')}))


def score(plan, out):
    prepared = load(plan)
    if prepared['panel'] != 'math_hard':
        return nm_score.score(plan, out)
    from kit.beds.math_screen import compute_score
    from kit.inference_lifecycle import create_llm, shutdown_since
    from kit.m2_assignment import inventory
    from kit.score_forgetting import ENGINE, DECODING
    from vllm import SamplingParams
    need(authority('math_hard') == prepared['authority'], 'hard MATH scorer changed')
    ready(nm_score.file_stats(prepared['source']) == prepared['metadata'], 'checkpoint metadata changed')
    identity = inventory(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    need(identity['uuid'], 'scoring UUID unavailable')
    os.environ.update(VLLM_BATCH_INVARIANT='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    engine = create_llm(model=prepared['source']['model'], enable_lora=False,
                        **dict(ENGINE, max_model_len=4096), enforce_eager=True, seed=0)
    try:
        outputs = engine.generate(prepared['rendered'], SamplingParams(n=1, **DECODING))
    finally:
        shutdown_since()
    need(len(outputs) == len(prepared['items']), 'missing hard MATH scoring response')
    raw, items = [], {}
    for item, output in zip(prepared['items'], outputs):
        response = output.outputs[0]
        value = int(compute_score('math_screen', response.text, item['gold'])['score'])
        items[item['id']] = value
        raw.append(dict(id=item['id'], response=response.text, output_tokens=len(response.token_ids),
                        finish_reason=response.finish_reason, correct=value))
    cell = dict(seed=prepared['seed'], panel='math_hard', model=prepared['model'], step=prepared['step'],
                items=items, checkpoint_sha256=prepared['source']['sha256'],
                nonce=os.environ['NM_ALLOCATION_NONCE'], uuid=identity['uuid'], environment=identity['environment'],
                scorer_sha256=prepared['authority']['sha256'], method=prepared['authority']['method'],
                precision='full', status='COMPLETE', technical_synthetic=False)
    atomic(out, dict(cell=cell, raw=raw))
