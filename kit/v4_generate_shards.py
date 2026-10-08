#!/usr/bin/env python3
"""Contained eight-GPU corpus row: four TP2 teacher or eight TP1 rewrite shards."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit import v4_teacher as t


def shard_commands(work,phase,task,rewriting=False):
    from kit.v4_campaign_ops import paths
    base=paths(work,phase);role='rewrite' if rewriting else 'teacher';count=8 if rewriting else 4;tp=1 if rewriting else 2
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=8 or len(set(devices))!=8 or any(not d for d in devices):raise ValueError('eight distinct contained GPUs required for shard row')
    python=sys.executable if rewriting else os.environ['V4_TEACHER_PYTHON']
    commands=[]
    for index in range(count):
        argv=[python,str(Path(__file__).with_name('v4_teacher.py')),'rewrite' if rewriting else 'generate',
            '--task',task,'--pool',str(base/(task+'-pool.jsonl')),'--out',str(base/(role+'-shards')/task),
            '--model',os.environ['MODEL_DIR' if rewriting else 'TEACHER_MODEL_DIR'],
            '--tokenizer-8b',os.environ['QWEN3_8B_TOKENIZER'],'--tp',str(tp),'--shard-index',str(index),'--shard-count',str(count)]
        if rewriting:argv+=['--demonstrations',str(base/'teacher'/task)]
        commands.append((argv,{'CUDA_VISIBLE_DEVICES':','.join(devices[index*tp:(index+1)*tp]),'V4_TELEMETRY':'1'}))
    return commands


def seed_reused_shards(work,phase,task,rewriting=False):
    """Route approved journals to the exact hash shard before any engine starts."""
    from kit.v4_campaign_ops import paths
    role='rewrite' if rewriting else 'teacher';count=8 if rewriting else 4;base=paths(work,phase)
    source=base/role/task
    for path in (source/'raw').glob('*.json'):
        row=json.loads(path.read_text());index=int(t.sha(str(row['id'])),16)%count
        destination=base/(role+'-shards')/task/t.shard_directory(index,count)
        for part in ('raw','timings'):
            original=source/part/path.name
            if not original.exists():continue
            target=destination/part/path.name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=original.read_bytes():raise ValueError('approved shard journal changed')
            if not target.exists():target.write_bytes(original.read_bytes())


def generate(work,phase,task,rewriting=False):
    from kit.v4_campaign_ops import paths
    from kit.runner import bounded_command
    base=paths(work,phase);role='rewrite' if rewriting else 'teacher';count=8 if rewriting else 4
    root=base/role/task
    if (root/'coverage.json').exists() and t.read_jsonl(root/'pool.jsonl')==t.read_jsonl(base/(task+'-pool.jsonl')):
        return  # first complete frozen corpus is immutable.
    if phase=='main':seed_reused_shards(work,phase,task,rewriting)
    commands=shard_commands(work,phase,task,rewriting)
    logs=base/'shard-logs'/role/task;logs.mkdir(parents=True,exist_ok=True)
    def launch(pair):
        index,(command,env)=pair
        return bounded_command(command,timeout=int(os.environ.get('V4_COMMAND_TIMEOUT','14400')),
            env={**os.environ,**env},log=logs/f'{index}.log')
    with ThreadPoolExecutor(max_workers=count) as pool:results=list(pool.map(launch,enumerate(commands)))
    t.atomic_json(logs/'launches.json',{'role':role,'shards':count,'tp':1 if rewriting else 2,'results':results})
    if any(result['returncode'] for result in results):raise ValueError('generation_shard_failure')
    dirs=[base/(role+'-shards')/task/t.shard_directory(i,count) for i in range(count)]
    merged=base/(role+'-merged')/task;merged.parent.mkdir(parents=True,exist_ok=True)
    if not merged.exists():
        from kit.v4_timing import utc_now, receipt
        started=utc_now();clock=time.monotonic()
        t.merge_shards(dirs,merged,t.load_student_tokenizer(os.environ['QWEN3_8B_TOKENIZER']))
        # This serial replay is additional to each worker's verification. Price
        # the complete merge (tokenizer loading, both replay passes and IO) per
        # observed attempt, then project four attempts across the full pool.
        raw=t.read_jsonl(merged/'raw_outputs.jsonl');pool_items=t.read_jsonl(merged/'pool.jsonl')
        t.atomic_json(merged/'serial-merge-timing.json',receipt(started,time.monotonic()-clock,
            verified_attempts=len(raw),raw_sha256=t.sha(json.dumps(raw,sort_keys=True)),
            pool_sha256=t.sha(json.dumps(pool_items,sort_keys=True))))
    if not (merged/'serial-merge-timing.json').is_file():raise ValueError('serial merge timing missing; review interrupted merge')
    # Preserve every prior approved attempt; publish the complete merged tree only
    # after all shards verify. The sampled root contains journals, not a corpus.
    import shutil
    if root.exists():
        history=base/(role+'-approved-sample')/task;history.parent.mkdir(parents=True,exist_ok=True)
        if history.exists():raise ValueError('sample history already exists')
        os.rename(root,history)
    root.parent.mkdir(parents=True,exist_ok=True);os.rename(merged,root)
    t.atomic_json(root/'shard-execution.json',{'shards':count,'tp':1 if rewriting else 2,
        'batch_geometry':t.batched_worst_case(len(t.read_jsonl(root/'pool.jsonl')),shard_count=count,
            shard_sizes=[len(t.select_shard(t.read_jsonl(root/'pool.jsonl'),i,count)) for i in range(count)])})


def determinism(work):
    """Run in the separately pinned inference environment, on a TP2 pair."""
    script=Path(__file__).resolve()
    env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[:2])}
    subprocess.run([os.environ['V4_TEACHER_PYTHON'],str(script),'--determinism',str(work)],env=env,check=True,timeout=1500)


def _determinism(work):
    from kit.v4_campaign_ops import paths
    pool=t.read_jsonl(paths(work,'qualification')/'chemistry-pool.jsonl')[:3]
    engine=t.VLLMGenerator(os.environ['TEACHER_MODEL_DIR'],t.TEACHER_NEW_TOKENS,tensor_parallel_size=2,allow_batch_fallback=True)
    requests=[{'messages':t.teacher_messages(item),'seed':t.SCHEDULE[0][0],'temperature':0.,'top_p':1} for item in pool]
    left=engine.generate_many(requests[:2]);right=engine.generate_many([requests[2],requests[1],requests[0]])
    raw={'schema':'v4-teacher-determinism.v1','batch_policy':engine.batch_policy,'left':left,'right':right,
         'same_prompt_indices':[[0,2],[1,1]],'passed':all(left[i]['text']==right[j]['text'] and left[i].get('token_ids')==right[j].get('token_ids') for i,j in ((0,2),(1,1)))}
    t.atomic_json(paths(work,'qualification')/'teacher-determinism.json',raw)
    # Fallback availability is a reported qualification finding; frozen corpus
    # inputs remain fixed. It never pretends serial equivalence is proven.
    if engine.batch_policy['batch_invariant'] and not raw['passed']:raise ValueError('batch-invariant determinism failed')

def profile_waves(work):
    """Full-cap, full-width waves; actual latency bounds the batched projection."""
    from kit.v4_campaign_ops import paths
    from kit.runner import bounded_command
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(set(devices))!=8:raise ValueError('wave profiling requires eight contained GPUs')
    root=paths(work,'qualification')/'wave-profile';root.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for role in ('teacher','rewrite'):
        tp=2 if role=='teacher' else 1;count=8//tp
        def launch(index):
            command=[os.environ['V4_TEACHER_PYTHON'] if role=='teacher' else sys.executable,str(Path(__file__).resolve()),
                     '--profile-wave',str(root/f'{role}-{index}.json'),role]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(devices[index*tp:(index+1)*tp])}
            return bounded_command(command,timeout=1500,env=env,log=root/f'{role}-{index}.log')
        with ThreadPoolExecutor(max_workers=count) as pool:results=list(pool.map(launch,range(count)))
        if any(result['returncode'] for result in results):raise ValueError('full-wave qualification failed: '+role)
        jobs.append({'role':role,'results':results})
    t.atomic_json(root/'launches.json',{'real_stack':True,'jobs':jobs})


def _profile_wave(out,role):
    from kit.v4_timing import utc_now,receipt
    import time
    tp=2 if role=='teacher' else 1
    cap=t.TEACHER_NEW_TOKENS if role=='teacher' else t.REWRITE_NEW_TOKENS
    input_cap=t.INPUT_CAP if role=='teacher' else t.REWRITE_INPUT_CAP
    engine=t.VLLMGenerator(os.environ['TEACHER_MODEL_DIR' if role=='teacher' else 'MODEL_DIR'],cap,
        input_cap=input_cap,tensor_parallel_size=tp,allow_batch_fallback=role=='teacher')
    # Synthetic token-budget stress inputs, never training/held-out questions.
    prompts=[{'prompt_token_ids':[int(engine.tokenizer.encode(' x',add_special_tokens=False)[0])]*input_cap} for _ in range(t.H100_MAX_NUM_SEQS)]
    params=engine.params(seed=27101,temperature=0.,top_p=1,n=1,max_tokens=cap,min_tokens=cap,ignore_eos=True)
    started=utc_now();clock=time.monotonic();outputs=engine.engine.generate(prompts,params,use_tqdm=False)
    elapsed=time.monotonic()-clock
    counts=[len(row.outputs[0].token_ids) for row in outputs]
    if counts!=[cap]*t.H100_MAX_NUM_SEQS:raise ValueError('full-wave profile output dose incomplete')
    t.atomic_json(Path(out),{**receipt(started,elapsed),'role':role,'tp':tp,'input_cap':input_cap,
        'max_new_tokens':cap,'max_num_seqs':t.H100_MAX_NUM_SEQS,'output_tokens':counts,
        'batch_policy':engine.batch_policy,'reload_timing':engine.reload_timing,'technical_budget_profile':True})

if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--profile-wave':_profile_wave(sys.argv[2],sys.argv[3])
    elif len(sys.argv)==3 and sys.argv[1]=='--determinism':_determinism(Path(sys.argv[2]))
    elif sys.argv[1:] in (['--help'],['-h']):print('v4_generate_shards: campaign corpus dispatch or --determinism WORK')
    else:raise SystemExit('use the registered campaign operation')
