#!/usr/bin/env python3
"""Complete single-GPU TRACE lanes and offline-bound metric authorities."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import importlib.util
import math
import os
import subprocess
from kit.trace_common import *


def metric_adapter(ref):
    path = bound_file(ref)
    spec = importlib.util.spec_from_file_location('trace_metric_'+sha(path), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, ref['function'])
    fixtures = bound_json(ref['fixtures'])
    need(len(fixtures) >= 20, 'twenty metric fixtures required')
    for fixture in fixtures:
        actual = fn(fixture['row'], fixture['prediction'])
        need(type(actual) in (float, int) and math.isfinite(actual) and
             abs(actual-fixture['score']) <= 1e-6, 'scorer fixture mismatch')
    return fn


def aggregate(values):
    need(values and all(type(x) in (float,int) and math.isfinite(x) and 0 <= x <= 100
                       for x in values), 'metric range/units differ')
    return dict(sum=math.fsum(values), n=len(values), mean=math.fsum(values)/len(values))


def generation(model, tokenizer, rows, prompt_cap, response_cap, temperature, seed):
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_BATCH_INVARIANT'] = '1'
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from kit.inference_lifecycle import create_llm, shutdown_since, marker
    from kit.m2_gpu import assert_current
    assert_current()
    tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    prompts = [tok.apply_chat_template(r.get('messages') or [dict(role='user',content=r['prompt'])],
               tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    token_ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
    # S8 retains the registered left clipping. The subset was length-filtered.
    uncut_lengths = [len(ids) for ids in token_ids]
    token_ids = [ids[-prompt_cap:] for ids in token_ids]
    scope = marker()
    try:
        engine = create_llm(model=str(model), tokenizer=str(tokenizer),
                    dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
                    max_model_len=prompt_cap+response_cap, gpu_memory_utilization=.85)
        params = [SamplingParams(temperature=temperature, top_p=1, top_k=-1,
                   max_tokens=response_cap, seed=int(digest([seed,r['id']])[:8],16)) for r in rows]
        outputs = engine.generate([dict(prompt_token_ids=ids) for ids in token_ids], params,
                                  use_tqdm=False)
        need(len(outputs) == len(rows), 'generation cells missing')
        return [dict(id=r['id'], text=o.outputs[0].text,
                     finish_reason=o.outputs[0].finish_reason,
                     tokens=len(o.outputs[0].token_ids), prompt_tokens=len(ids),
                     prompt_cap_hit=uncut > prompt_cap,
                     response_cap_hit=len(o.outputs[0].token_ids) >= response_cap)
                for r,o,ids,uncut in zip(rows,outputs,token_ids,uncut_lengths)]
    finally:
        shutdown_since(scope)


def score_state(cfg, study, seed, state, model):
    protocol = load(Path(cfg['out'])/'prepared.json')
    panels = protocol['panels']
    tasks = TASKS if study == 's8' else SUBSET
    work = Path(cfg['out'])/'scores'/f'{study}-{seed}'/state['id']
    work.mkdir(parents=True, exist_ok=True)
    results = {}
    for task in tasks:
        rows = protocol['datasets'][task]['test'] if study == 's8' else protocol['tests'][task]
        outputs = generation(model, cfg['base']['model'], rows, 1024 if study=='s8' else 2048,
                             512 if study=='s8' else 2048, .1, seed)
        fn = metric_adapter(cfg['metrics'][task])
        values = [fn(row,output['text']) for row,output in zip(rows,outputs)]
        results[task] = aggregate(values)
        atomic(work/(task+'.json'), dict(outputs=outputs, values=values))
    general = panels['general-v1']
    for panel in ('instruction','maths'):
        aliases = {'instruction':('instruction','ifeval'), 'maths':('maths','math')}
        rows = [r for r in general if r['panel'] in aliases[panel]]
        outputs = generation(model, cfg['base']['model'], rows, 2048, 2048, 0., seed)
        fn = metric_adapter(cfg['metrics'][panel])
        values = [fn(row,output['text']) for row,output in zip(rows,outputs)]
        results[panel] = aggregate(values)
        atomic(work/(panel+'.json'), dict(outputs=outputs, values=values))
    # Use the kit's independent reference implementation and its frozen parsing.
    from kit.m3_score import knowledge_reference
    rows = [r for r in general if r['panel'] == 'knowledge']
    reference = knowledge_reference(model, rows, cfg['base']['model'])
    values = [100*int(r['selected'] == 'ABCD'.index(row['answer'])) for r,row in zip(reference,rows)]
    results['knowledge'] = aggregate(values)
    atomic(work/'knowledge.json', dict(records=reference, values=values))
    if study == 's8':
        for panel, spec in panels['trace_general'].items():
            # Frozen few-shot messages, extraction and decoding are node inputs.
            rows = spec['rows']
            outputs = generation(model, cfg['base']['model'], rows,
                         spec['prompt_cap'], spec['response_cap'], spec['temperature'], seed)
            fn = metric_adapter(cfg['metrics'][panel])
            values = [fn(row,output['text']) for row,output in zip(rows,outputs)]
            results[panel] = aggregate(values)
            atomic(work/(panel+'.json'), dict(outputs=outputs, values=values))
    outputs = generation(model, cfg['base']['model'], panels['open'], 2048, 2048, .1, seed)
    atomic(work/'open.json', outputs)
    atomic(work/'result.json', results)


def lane(cfg, study, seed, attempt):
    from kit.m2_gpu import allocated_devices, prepare, current_uuid, normalized_uuid
    devices = allocated_devices()
    need(len(devices) == 1, 'one physical scoring GPU required')
    gpu, uuid = next(iter(devices.items()))
    prepare(gpu, uuid)
    need(normalized_uuid(current_uuid()) == normalized_uuid(uuid), 'CUDA identity differs')
    import importlib.metadata
    environment = {name:importlib.metadata.version(name) for name in ('torch','transformers','vllm')}
    exports = load(Path(cfg['out'])/'training'/f'{study}-{seed}'/'exports.json')
    rows = states(study,seed)
    for state in rows:
        record = exports[state['checkpoint']]
        model_record(record)
        command = [cfg.get('score_python',sys.executable), str(KIT/'trace_score.py'),
                   '--config', cfg['_config_path'], '--study',study,'--seed',str(seed),
                   '--state',state['id'],'--model',record['model']]
        subprocess.run(command, check=True)
        need(normalized_uuid(current_uuid()) == normalized_uuid(uuid), 'scoring GPU changed')
    atomic(Path(cfg['out'])/'scores'/f'{study}-{seed}'/'lane.json',
           dict(states=rows, uuid_sha256=digest(normalized_uuid(uuid)), environment_sha256=digest(environment),
                attempt=attempt, config_sha256=digest(cfg), complete=True,
                artifacts_sha256={str(p.relative_to(Path(cfg['out']))):sha(p)
                    for p in (Path(cfg['out'])/'scores'/f'{study}-{seed}').rglob('*.json')
                    if p.name!='lane.json'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--study', choices=('s8',*SOURCES), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    offline()
    os.environ['VLLM_BATCH_INVARIANT'] = '1'
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
    cfg = load(args.config)
    state = next(s for s in states(args.study,args.seed) if s['id']==args.state)
    score_state(cfg,args.study,args.seed,state,args.model)
