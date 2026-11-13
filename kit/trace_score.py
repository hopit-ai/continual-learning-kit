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
    for dependency in ref.get('dependencies', []):
        bound_file(dependency)
    if path.resolve() == (KIT/'trace_general_metrics.py').resolve():
        from kit import trace_general_metrics as module
    else:
        spec = importlib.util.spec_from_file_location('trace_metric_'+sha(path), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    fn = getattr(module, ref['function'])
    if 'authority' in ref:
        implementation = fn
        def fn(row, prediction):
            return implementation(dict(row, metric_authority=ref['authority']), prediction)
    fixtures = bound_json(ref['fixtures'])
    need(len(fixtures) >= 20, 'twenty metric fixtures required')
    batch = module.score_many if path.resolve() == (KIT/'trace_general_metrics.py').resolve() else None
    actuals = (batch([f['row'] for f in fixtures], [f['prediction'] for f in fixtures])
               if batch else [fn(f['row'], f['prediction']) for f in fixtures])
    need(len(actuals) == len(fixtures), 'scorer fixture results missing')
    for fixture, actual in zip(fixtures, actuals):
        need(type(actual) in (float, int) and math.isfinite(actual) and
             0 <= actual <= 100 and
             type(fixture['score']) in (float, int) and math.isfinite(fixture['score']) and
             0 <= fixture['score'] <= 100 and
             abs(actual-fixture['score']) <= 1e-6, 'scorer fixture mismatch')
    if batch:
        fn.score_many = batch
    return fn


def aggregate(values):
    need(values and all(type(x) in (float,int) and math.isfinite(x) and 0 <= x <= 100
                       for x in values), 'metric range/units differ')
    return dict(sum=math.fsum(values), n=len(values), mean=math.fsum(values)/len(values))


def generation(model, tokenizer, rows, prompt_cap, response_cap, temperature, seed,
               *, judge_profile=None, audit=None):
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_BATCH_INVARIANT'] = '1'
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from kit.inference_lifecycle import create_llm, shutdown_since, marker
    from kit.m2_gpu import assert_current
    assert_current()
    from kit.trace_score import cached_tokens as cached, encode_tokens as encode
    arrays = (cached(tokenizer, rows, prompt_cap) if judge_profile is None else None)
    if arrays is None:
        tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
        arrays = encode(tok, rows, prompt_cap)
    token_ids, uncut_lengths = arrays['token_ids'], arrays['uncut_lengths']
    extra = {}
    if judge_profile is not None:
        from kit.trace_judge import settings
        extra = settings(judge_profile)
    if audit is not None:
        audit.update(token_ids_sha256=digest(token_ids), request_ids=[r['id'] for r in rows],
                     seeds=[int(digest([seed,r['id']])[:8],16) for r in rows],
                     grouping='one call, all both-order requests', retry_count=0,
                     canonical_orders=[int(r['id'].rsplit('-',1)[1]) for r in rows])
    scope = marker()
    try:
        engine_options = dict(model=str(model), tokenizer=str(tokenizer),
                    dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
                    max_model_len=prompt_cap+response_cap, gpu_memory_utilization=.85)
        engine_options.update(extra)
        if judge_profile is None:
            engine = create_llm(**engine_options)
        else:
            from kit.trace_judge import create_judge_engine
            engine = create_judge_engine(engine_options)
        if judge_profile is not None:
            from kit.trace_judge import effective_settings, validate_settings
            resolved = effective_settings(engine, engine_options)
            validate_settings(resolved,judge_profile)
            if audit is not None:
                audit['engine'] = resolved
        params = [SamplingParams(temperature=temperature, top_p=1, top_k=-1,
                   max_tokens=response_cap, seed=int(digest([seed,r['id']])[:8],16)) for r in rows]
        outputs = engine.generate([dict(prompt_token_ids=ids) for ids in token_ids], params,
                                  use_tqdm=False)
        need(len(outputs) == len(rows), 'generation cells missing')
        return [dict(id=r['id'], text=o.outputs[0].text,
                     finish_reason=o.outputs[0].finish_reason,
                     tokens=len(o.outputs[0].token_ids), prompt_tokens=len(ids),
                     prompt_cap_hit=uncut > prompt_cap,
                     response_cap_hit=len(o.outputs[0].token_ids) >= response_cap,
                     **(dict(prompt_truncated_tokens=max(0,uncut-prompt_cap))
                        if r.get('render_mode') else {}))
                for r,o,ids,uncut in zip(rows,outputs,token_ids,uncut_lengths)]
    finally:
        shutdown_since(scope)


def score_state(cfg, study, seed, state, model):
    from kit.trace_common import local_file
    protocol = load(local_file(Path(cfg['out'])/'prepared.json'))
    panels = protocol['panels']
    tasks = TASKS if study == 's8' else SUBSET
    work = Path(cfg['out'])/'scores'/f'{study}-{seed}'/state['id']
    from kit.trace_common import output_tree
    with output_tree(work) as work:
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
            from kit.trace_general_score import likelihood, likelihood_runtime, aggregate as general_aggregate
            ppl_outputs = {}
            with likelihood_runtime(model, cfg['base']['model']) as runtime:
                for panel, spec in panels['trace_general'].items():
                    if spec['mode'] == 'ppl':
                        ppl_outputs[panel] = likelihood(model, spec['rows'], cfg['base']['model'], runtime=runtime)
            for panel, spec in panels['trace_general'].items():
                rows = spec['rows']
                if spec['mode'] == 'ppl':
                    outputs = ppl_outputs[panel]
                    predictions = [o['selected'] for o in outputs]
                else:
                    outputs = generation(model, cfg['base']['model'], rows,
                                 spec['prompt_cap'], spec['response_cap'], spec['temperature'], seed)
                    predictions = [o['text'] for o in outputs]
                fn = metric_adapter(cfg['metrics'][panel])
                values = fn.score_many(rows, predictions)
                results[panel] = general_aggregate(panel, rows, values)
                if spec['mode'] == 'ppl':
                    audit = dict(prompt_cap_hits=sum(o['prompt_cap_hits'] for o in outputs),
                                 truncated_tokens=sum(o['truncated_tokens'] for o in outputs),
                                 shot_dropped_items=sum(any(len(ids) < 5 for ids in
                                     o.get('shot_ids_by_label', {}).values()) for o in outputs)
                                     if panel == 'MMLU' else 0)
                else:
                    audit = dict(prompt_cap_hits=sum(o['prompt_cap_hit'] for o in outputs),
                                 truncated_tokens=sum(o['prompt_truncated_tokens'] for o in outputs),
                                 response_cap_hits=sum(o['response_cap_hit'] for o in outputs))
                if audit['prompt_cap_hits']:
                    import warnings
                    warnings.warn(f"{panel}: {audit['prompt_cap_hits']} prompt cap hits", stacklevel=2)
                atomic(work/(panel+'.json'), dict(outputs=outputs, values=values, audit=audit))
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




from pathlib import Path
import os
from kit.trace_common import digest, sha, load, need, seal


def tokenizer_identity(tokenizer):
    root=Path(tokenizer)
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()
           and not p.name.endswith(('.bin','.safetensors'))}
    need(files,'tokenizer assets absent')
    return digest(files)


def encode_tokens(tok,rows,prompt_cap):
    prompts=[r['prompt'] if r.get('render_mode') == 'raw' else
             tok.apply_chat_template(r.get('messages') or [dict(role='user',content=r['prompt'])],
             tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
    ids=[tok.encode(p,add_special_tokens=False) for p in prompts]
    return dict(token_ids=[x[-prompt_cap:] for x in ids],uncut_lengths=[len(x) for x in ids])


def cached_tokens(tokenizer,rows,prompt_cap,*,write=False,cache=None):
    cache=cache or os.environ.get('TRACE_TOKEN_CACHE')
    if not cache:
        return None
    identity=tokenizer_identity(tokenizer)
    key=digest([identity,prompt_cap,'chat-no-thinking-v1',rows])
    path=Path(cache)/(key+'.json')
    if path.exists():
        from kit.trace_common import local_file
        value=load(local_file(path))
        need(value['key']==key and value['sha256']==digest(value['arrays']), 'token cache changed')
        return value['arrays']
    if not write:
        raise FileNotFoundError('CPU token preparation missing')
    from transformers import AutoTokenizer
    arrays=encode_tokens(AutoTokenizer.from_pretrained(tokenizer,local_files_only=True),rows,prompt_cap)
    seal(path,dict(key=key,sha256=digest(arrays),arrays=arrays))
    return arrays


def token_groups(protocol,study):
    from kit.trace_common import TASKS,SUBSET
    panels=protocol['panels']
    for task in (TASKS if study=='s8' else SUBSET):
        yield (protocol['datasets'][task]['test'] if study=='s8' else protocol['tests'][task],
               1024 if study=='s8' else 2048)
    general=panels['general-v1']
    for aliases in (('instruction','ifeval'),('maths','math')):
        yield [r for r in general if r['panel'] in aliases],2048
    if study=='s8':
        for spec in panels['trace_general'].values():
            if spec['mode'] == 'generation':
                yield spec['rows'],spec['prompt_cap']
    yield panels['open'],2048


def prepare_tokens(cfg):
    protocol=load(Path(cfg['out'])/'prepared.json')
    cache=Path(cfg['out'])/'token-cache'
    for study in protocol['active_studies']:
        for rows,cap in token_groups(protocol,study):
            cached_tokens(cfg['base']['model'],rows,cap,write=True,cache=cache)
    return dict(files={p.name:sha(p) for p in cache.glob('*.json')})


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
