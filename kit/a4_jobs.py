#!/usr/bin/env python3
"""Prospective v6b 4A jobs. CPU preparation, native training, CPU export, complete lanes."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
from kit import a4_train, a4_score, a4_memory, a4_ledger, a4_nll
from kit.a4_run import score_plan
from types import SimpleNamespace as NS
from contextlib import nullcontext,ExitStack
import argparse
import csv
import gzip
import io
import shutil
import uuid

SCHEMA='a4-v6b-rows.v1'
PENDING=75


def inventory(job_limits=None):
    limits=dict(training=4500,scoring=600,nll=600) if job_limits is None else job_limits
    require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0
        for v in limits.values()) and set(limits)=={'training','scoring','nll'},'invalid scheduling job limits')
    rows=[]
    def add(name,phase,deps=(),gpus=0,*,arm=None,seed=None,month=None,scores=0,nll=0,after='afterok'):
        row=dict(id='A4-'+name,group='4a',phase=phase,gpus=gpus,nodes=1,cpus=32 if gpus==8 else 8,
            may_share_node=gpus!=8,depends_on=[dict(id=d,type=after) for d in deps],
            dependency_options=['--kill-on-invalid-dep=yes'],
            command=['{job_python}','-I','{kit}/a4_jobs.py','run','--config','{a4_config}','--task','A4-'+name],
            expected_command_counts=dict(dispatch=1,score=scores if gpus==1 else 0,nll=nll if gpus==1 else 0,trainer=int(phase=='train'),
                export=int(phase=='export')),lane_concurrency=3 if scores else 1,
            count_semantics='productive command upper bounds; exact model/prompt deduplication and empty-ledger skips recorded in prepared.json',
            gpu_hour_basis=('8 × measured training wall hours; limit 1.5 × bound W stage GPU-h' if gpus==8 else
                '1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1' if gpus else '0 GPU-h; CPU work only'))
        hard=duration_limit(dict(job_limits=limits),row) if gpus else None
        floor=5400 if gpus==8 else (93600 if name=='comparison' else 5400 if nll else 3600) if gpus else 14400
        seconds=math.ceil(max(floor,hard+max(900,.1*hard) if hard is not None else floor)/60)*60
        row.update(time=f'{seconds//3600:02}:{seconds//60%60:02}:00',time_limit_seconds=seconds,
            job_deadline_seconds=hard,time_basis='sbatch --time; hard work bound plus at least 15 minutes or 10%; CPU planning allowance 4h')
        if arm is not None:row['arm']=arm
        if seed is not None:row['seed']=seed
        if month is not None:row['month']=month
        if phase=='score-prepare':row['prepared_score_cells']=scores;row['prepared_nll_endpoints']=nll
        rows.append(row);return row['id']
    previous=add('bind','bind')
    for n,month in enumerate(MONTHS,1):
        context=add(f'context-m{n}','context',[previous],month=n)
        prepared=add(f'arrival-prepare-m{n}','score-prepare',[context],month=n,scores=len(list(arrival_specs())))
        arrival=add(f'arrival-m{n}','score',[prepared],1,month=n,scores=len(list(arrival_specs())))
        ledger_row=add(f'ledger-m{n}','ledger',[arrival],month=n)
        finishes=[]
        for arm in ARMS:
            if arm=='consolidation' and n not in BOUNDARIES:continue
            for seed in SEEDS:
                name=stage_name(arm,seed,n)
                kw=dict(arm=arm,seed=seed,month=n)
                prepare=add(name+'-prepare','prepare',[ledger_row],**kw)
                train=add(name+'-train','train',[prepare],8,**kw)
                export=add(name+'-export','export',[train],**kw)
                count=len(stage_scores(arm,seed,n))
                sp=add(name+'-score-prepare','score-prepare',[export],scores=count,nll=2,**kw)
                sc=add(name+'-score','score',[sp],1,scores=count,nll=2,**kw)
                finishes.append(add(name+'-finish','finish',[sc],**kw))
        previous=add(f'month-m{n}','month',[ledger_row,*finishes],month=n)
    count=len(list(score_plan()))
    prep=add('comparison-prepare','score-prepare',[previous],scores=count)
    comparison=add('comparison','score',[prep],1,scores=count)
    reading=add('read','read',[comparison],after='afterany')
    add('collect','collect',[reading])
    return dict(schema=SCHEMA,amendment=AMENDMENT,scheduling_job_limits=limits,ceiling_gpu_hours=800,training_stages=27,
        scientific_score_cells=count,rows=rows,
        gpu_hours=dict(training_D_proxy_range=[170.1,180.9],training_contingency_D_range=[255.15,271.35],
            scoring_reserve=60,overhead_failure_reserve=80,
            formula='ceil_to_50(1.5 × 27 × measured_stage_gpu_hours + 140)',
            D_forecast_range=[400,450],approved_ceiling=800,
            note='Planning only. Repeated complete-lane calls and 54 HF NLL calls consume the same reserves; no SEC throughput qualification. CPU export holds zero GPUs.'))


def stage_scores(arm,seed,n):
    return [r for r in score_plan() if r[0] in ('incoming','post') and r[1:4]==(arm,seed,n)]


def row_by_id(task):
    values=[r for r in inventory()['rows'] if r['id']==task]
    require(len(values)==1,'unconfigured 4A row: '+task);return values[0]


def status(out,task):
    path=out/'v6b/status'/(task+'.json')
    if not path.exists():raise NotReady('PENDING: '+task)
    value=json.loads(path.read_text())
    if value['status']!='COMPLETE':raise NotReady('PENDING: '+task)
    receipt=out/value['receipt'];check(receipt,value['sha256'])
    result=json.loads(receipt.read_text())
    require(result['task']==task and result['status']=='COMPLETE','invalid row receipt')
    for name,digest in result.get('artifacts',{}).items():check(out/name,digest)
    return receipt.parent


def completed(out,task):
    try:status(out,task)
    except NotReady:return False
    return True


def config(path):
    value=json.loads(path.read_text())
    require(isinstance(value.get('technical_synthetic',False),bool),'invalid synthetic flag')
    return value


def model_for(out,doc,row):
    kind,arm,seed,n,p,mode=row
    if arm in ('frozen','memory'):return doc['base']
    return state(out,doc,arm,seed,n-1 if kind in ('arrival','incoming') else n)


def stage_root(out,row):return out/'runs'/stage_name(row['arm'],row['seed'],row['month'])


def prepare_training(out,doc,row,cfg):
    arm,seed,n=row['arm'],row['seed'],row['month'];run=stage_root(out,row);data=out/'data'/run.name
    incoming=state(out,doc,arm,seed,n-1)
    tok=tokenizer(doc,doc['technical_synthetic']);schedule,audit=a4_train.schedule(out,doc,arm,seed,n,tok)
    observed=dict(incoming=model_identity(incoming,doc['technical_synthetic']),rows=s.digest(schedule),seed=seed,recipe=doc['W'],
        binding=s.file_hash(out/'binding.json'),ledger={m:s.file_hash(out/'ledger'/m/'manifest.json') for m in MONTHS[:n]},audit=audit)
    bind(out/'stage-identities'/(run.name+'.json'),observed)
    data.mkdir(parents=True,exist_ok=True);t.write_jsonl(data/'schedule.jsonl',schedule)
    converted,counts=a4_train.training_rows(schedule,tok,doc['W']) if schedule else ([],{})
    if schedule and not doc['technical_synthetic']:
        require(cfg.get('sdpo') and cfg.get('tokenizer'),'pinned SDPO/tokenizer required')
        require(subprocess.check_output(['git','-C',cfg['sdpo'],'rev-parse','HEAD'],text=True).strip()=='7c457fc1b1f636ae794eb0362ba37d4743b06fbc','wrong SDPO revision')
        identity(cfg['tokenizer'],'student')
        import pyarrow as pa
        import pyarrow.parquet as pq
        for part,rr in [('train',converted),('val',converted[:32]),('test',[dict(converted[0],extra_info=dict(converted[0]['extra_info'],demonstration=''))])]:
            pq.write_table(pa.Table.from_pylist(rr),data/(part+'.parquet'))
    names=['schedule.jsonl']+(['train.parquet','val.parquet','test.parquet'] if schedule and not doc['technical_synthetic'] else [])
    prepared=dict(observed=observed,incoming=incoming,empty=not schedule,scheduled_tokens=counts)
    if schedule and not doc['technical_synthetic']:
        a=NS(sdpo=Path(cfg['sdpo']),seed=seed,gpu=','.join(str(i) for i in range(8)),tokenizer=cfg['tokenizer'],trainer_python=cfg.get('trainer_python',sys.executable))
        command,resolve,env=a4_train.command(a,doc['W'],incoming['model'],data,run)
        # Resolve Hydra and all input validation on CPU, before GPU admission.
        run.mkdir(parents=True,exist_ok=True)
        with (run/'resolved-config.yaml').open('w') as output,(run/'resolved-config.err').open('w') as error:
            subprocess.run(resolve,cwd=a.sdpo,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=output,stderr=error,check=True)
        prepared.update(command=command,env={k:v for k,v in env.items() if os.environ.get(k)!=v},cwd=str(a.sdpo))
    bind(data/'prepared.json',prepared)
    freeze(data,[*names,'prepared.json'],rows=len(schedule),technical_synthetic=doc['technical_synthetic'])
    return prepared


def native_train(out,doc,row,cfg,attempt):
    run=stage_root(out,row);data=out/'data'/run.name
    frozen_files(data);prepared=json.loads((data/'prepared.json').read_text())
    if prepared['empty']:return
    require(t.runtime_versions()==doc['software']['trainer'],'trainer environment changed')
    from kit.p3b_train import completed_training
    with lock(run):
        if completed_training(run,doc['W'][0],40):return
        if doc['technical_synthetic']:
            if cfg.get('fail_task')==row['id'] and not (run/'failure-injected.json').exists():
                atomic(run/'failure-injected.json',dict(technical_synthetic=True));raise RuntimeError('stand-in interrupted native training')
            atomic(run/'trained.json',dict(technical_synthetic=True,updates=40));return
        if (run/'started.json').exists():
            partial=run/'partials'/attempt.name;partial.mkdir(parents=True)
            for name in ('sdft','train','metrics.jsonl','started.json','trainer-exit.json','token-journals'):
                if (run/name).exists():(run/name).rename(partial/name)
        atomic(run/'started.json',prepared['observed']);clock=time.monotonic()
        env=dict(os.environ,**prepared['env']);env['CUDA_VISIBLE_DEVICES']=os.environ['CUDA_VISIBLE_DEVICES']
        done=gpu_run(prepared['command'],cwd=prepared['cwd'],env=env)
        atomic(run/'trainer-exit.json',dict(returncode=done.returncode,seconds=time.monotonic()-clock))
        require(done.returncode==0,'trainer failed; retain partials and restart same seed at zero')
        if not completed_training(run,doc['W'][0],40):raise NotReady('native step-40 checkpoint pending')


def export_stage(out,doc,row,cfg):
    run=stage_root(out,row);data=out/'data'/run.name
    frozen_files(data);prepared=json.loads((data/'prepared.json').read_text())
    if prepared['empty']:
        atomic(run/'summary.json',dict(skipped=True,reason='empty_monthly_ledger' if row['arm']=='direct' else 'empty_cumulative_ledger',**prepared['observed']))
        freeze(run,['summary.json']);return
    if not (run/'trained.json').exists():raise NotReady('native step-40 checkpoint pending')
    if doc['technical_synthetic']:atomic(run/'export-40.json',dict(technical_synthetic=True,step=40))
    else:
        from kit.v4_sft import complete_export
        export=run/'hf-step40'
        if not complete_export(export):
            ckpt=run/('sdft' if doc['W'][0]=='D' else 'train')/'global_step_40'
            if doc['W'][0]=='D':ckpt/='actor'
            subprocess.run([cfg.get('trainer_python',sys.executable),'-m','verl.model_merger','merge','--backend','fsdp','--local_dir',str(ckpt),'--target_dir',str(export)],cwd=cfg['sdpo'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        require(complete_export(export),'incomplete CPU merged checkpoint')
        atomic(run/'export-40.json',identity(export))
    runtime_tokens=None
    if not doc['technical_synthetic']:
        if doc['W'].startswith('D'):
            journals=[json.loads(p.read_text()) for p in (run/'token-journals').glob('*.json')]
            require(bool(journals),'D token accounting missing')
            runtime_tokens=sum(r['student_tokens']+r['teacher_tokens']+r['replay_target_tokens'] for r in journals)
        else:
            metrics=[r.get('data',r) for p in run.rglob('metrics.jsonl') if 'partials' not in p.parts for r in t.read_jsonl(p)]
            require(any('p3b/total_tokens' in r for r in metrics),'F token accounting missing')
            runtime_tokens=sum(r.get('p3b/total_tokens',0) for r in metrics)
    atomic(run/'summary.json',dict(skipped=False,**prepared['observed'],scheduled_tokens=prepared['scheduled_tokens'],runtime_training_tokens=runtime_tokens))
    # Export is ready before its scoring job. Stage completion comes only after NLL/post scores.
    freeze(run,['summary.json','export-40.json'])


def prepare_scores(out,doc,row,attempt,cfg):
    if row.get('arm'):specs=stage_scores(row['arm'],row['seed'],row['month'])
    elif row.get('month'):specs=[r for r in score_plan() if r[0]=='arrival' and r[3]==row['month']]
    else:specs=list(score_plan())
    scores=[];nll=[];groups={};models={}
    def checked(record):
        key=s.digest(record)
        if key not in models:models[key]=model_identity(record,doc['technical_synthetic'])
        return models[key]
    directory=attempt/'data';directory.mkdir()
    for index,key in enumerate(specs):
        record=model_for(out,doc,key);kind,arm,seed,n,p,mode=key
        data=a4_score.members(out,doc,p,mode,doc['technical_synthetic'])
        if kind!='arrival' and p in MONTHS:data=[r for r in data if r['item']['sec']['issuer_holdout']]
        observed=checked(record)
        spec=dict(row=key,record=record,model=observed,data=data,binding=s.file_hash(out/'binding.json'),software=doc['software']['inference'],max_seconds=doc['job_limits']['scoring'],technical_synthetic=doc['technical_synthetic'])
        group=s.digest(dict(model=observed.get('model_file_hashes',observed),data=data))
        if group in groups:
            groups[group]['aliases'].append(dict(index=index,row=list(key)))
            continue
        spec['aliases']=[];spec['index']=index;groups[group]=spec
    for spec in groups.values():
        path=directory/(str(spec['index'])+'.json');atomic(path,spec)
        scores.append(dict(path=str(path),sha256=s.file_hash(path),index=spec['index']))
    # Stage-score rows already measure both endpoints together. The complete
    # comparison consumes those diagnostics without repeating unused HF loads.
    stages=[row] if row.get('arm') else []
    for stage in stages:
        run=stage_root(out,stage);summary=json.loads((run/'summary.json').read_text())
        if summary['skipped']:continue
        schedule=out/'data'/run.name/'schedule.jsonl'
        incoming=json.loads((out/'data'/run.name/'prepared.json').read_text())['incoming']
        for which,record in [('before',incoming),('after',dict(model=str(run/'hf-step40'),receipt=str(run/'export-40.json')))]:
            nll.append(dict(stage=run.name,which=which,record=record,model=checked(record),software=doc['software']['inference'],max_seconds=doc['job_limits']['nll'],
                schedule=str(schedule),schedule_sha256=s.file_hash(schedule)))
    atomic(attempt/'prepared.json',dict(scores=scores,nll=nll,actual_command_counts=dict(score=len(scores),nll=len(nll),score_cells=len(specs)),technical_synthetic=doc['technical_synthetic']))
    freeze(attempt,['prepared.json',*[str(Path(v['path']).relative_to(attempt)) for v in scores]],technical_synthetic=doc['technical_synthetic'])


def run_scores(out,doc,row,attempt,cfg):
    source=status(out,row['depends_on'][0]['id'])
    frozen_files(source)
    prepared=json.loads((source/'prepared.json').read_text())
    from kit import lane_pool
    commands=[]
    for i,spec in enumerate(prepared['scores']):
        check(Path(spec['path']),spec['sha256'])
        commands.append(dict(argv=[cfg.get('inference_python',sys.executable),'-I',str(KIT/'a4_lane.py'),'--spec',spec['path'],'--out',str(attempt/'scores'/str(spec['index']))],requires=[spec['path']],engine=dict(backend='vllm',model='Qwen/Qwen3-8B' if doc['technical_synthetic'] else json.loads(Path(spec['path']).read_text())['record']['model'],max_model_len=12288)))
    command=lane_pool.invocation(commands,attempt/'lane',python=cfg.get('inference_python',sys.executable))
    command+=['--stagger-seconds','0']
    result=gpu_run(command,check=False,pass_fds=tuple(json.loads(os.environ.get('M3_GPU_LOCK_FDS','[]'))))
    if result.returncode in (3,75):raise NotReady('lane prerequisites pending')
    require(result.returncode==0,'complete scoring lane failed')
    # HF NLL uses K=1 (full logits); both incoming/final measured in this same allocation.
    nll_commands=[]
    for index,spec in enumerate(prepared['nll']):
        path=attempt/'data'/f'nll-{index}.json';atomic(path,dict(spec,technical_synthetic=doc['technical_synthetic'],binding=s.file_hash(out/'binding.json')))
        nll_commands.append(dict(argv=[cfg.get('inference_python',sys.executable),'-I',str(KIT/'a4_lane.py'),'--nll','--spec',str(path),'--out',str(attempt/'nll'/str(index))],requires=[str(path),spec['schedule']]))
    if nll_commands:
        command=lane_pool.invocation(nll_commands,attempt/'nll-lane',python=cfg.get('inference_python',sys.executable))
        command[command.index('--concurrency')+1]='1';command+=['--stagger-seconds','0']
        result=gpu_run(command,check=False,pass_fds=tuple(json.loads(os.environ.get('M3_GPU_LOCK_FDS','[]'))))
        if result.returncode in (3,75):raise NotReady('NLL lane prerequisites pending')
        require(result.returncode==0,'paired NLL lane failed')
    atomic(attempt/'coverage.json',dict(scores=[list(r) for r in (stage_scores(row['arm'],row['seed'],row['month']) if row.get('arm') else
        [r for r in score_plan() if r[0]=='arrival' and r[3]==row['month']] if row.get('month') else score_plan())],
        nll=prepared['nll']))


def publish_scores(out,source):
    coverage=json.loads((source/'coverage.json').read_text())
    for i,key in enumerate(coverage['scores']):
        root=score_root(out,*key);src=source/'scores'/str(i)
        frozen_files(src)
        if root.exists():
            require((root/'manifest.json').read_bytes()==(src/'manifest.json').read_bytes(),'chronological score already published by another attempt')
        else:shutil.copytree(src,root)
    for i,spec in enumerate(coverage['nll']):
        src=source/'nll'/str(i)/'result.json';value=json.loads(src.read_text())
        atomic(out/'runs'/spec['stage']/('nll-'+spec['which']+'.json'),value)


def execute(a):
    cfg=config(a.config);out=Path(cfg['out']);row=row_by_id(a.task);attempt=a.attempt
    standin=cfg.get('technical_synthetic',False)
    require(os.environ.get('A4_ALLOCATION_NONCE')==attempt.name and os.environ.get('UTIL_WATCH_ACTIVE')=='1',
        'execute requires the allocation wrapper and its single utilisation watcher')
    bind(out/'v6b/config.json',cfg)
    if row['phase']=='bind':
        from kit import a4_bind
        values={k:v for k,v in cfg.items() if k in ('base','stage_gpu_hours','trainer_python','score_max_seconds','nll_max_seconds','handoff_seconds')}
        values.update({k:Path(cfg[k]) if cfg.get(k) else None for k in ('b_source','slice','replay','timing_receipt')})
        values.update(out=out,standin=standin,gpu='0',v6b=True)
        values.setdefault('trainer_python',sys.executable)
        a4_bind.job(NS(**values));return
    doc=binding(out,standin)
    phase=row['phase']
    if phase in ('context','ledger','prepare','train','export'):
        require(t.runtime_versions()==doc['software']['trainer'],'bound trainer environment changed during CPU/training phase')
    if phase=='context':a4_memory.archive(out,doc,MONTHS[row['month']-1],standin)
    elif phase=='score-prepare':prepare_scores(out,doc,row,attempt,cfg)
    elif phase=='score':run_scores(out,doc,row,attempt,cfg)
    elif phase=='ledger':
        publish_scores(out,status(out,row['depends_on'][0]['id']))
        a4_ledger.job(NS(out=out,standin=standin,month=MONTHS[row['month']-1]))
    elif phase=='prepare':prepare_training(out,doc,row,cfg)
    elif phase=='train':native_train(out,doc,row,cfg,attempt)
    elif phase=='export':export_stage(out,doc,row,cfg)
    elif phase=='finish':
        publish_scores(out,status(out,row['depends_on'][0]['id']))
        run=stage_root(out,row);summary=json.loads((run/'summary.json').read_text())
        if not summary['skipped']:freeze(run,['summary.json','export-40.json','nll-before.json','nll-after.json'])
    elif phase=='month':bind(out/'months'/(MONTHS[row['month']-1]+'.json'),dict(complete=True,seal=s.file_hash(out/'seals'/(MONTHS[row['month']-1]+'.json'))))
    elif phase=='read':read_job(NS(out=out,standin=standin,bootstrap=2000))
    elif phase=='collect':
        from kit.next_collect import job as collect
        # All payload/reading jobs finished. The CPU collector verifies every omitted durable byte.
        collect(NS(out=out,standin=standin,archive=Path(cfg['archive']),durable_root=Path(cfg.get('durable_root',str(out)))))
        from kit.a4_read import job as read
        read(NS(out=out,standin=standin,bootstrap=2000,source=Path(cfg['archive'])))


def hardware(records):
    if any(r.get('_hardware_unverified') for rr in records.values() for r in rr):return 'UNVERIFIED-HARDWARE'
    identities={(r.get('_allocation_job'),r.get('_allocation_uuid'),r.get('_allocation_environment')) for rr in records.values() for r in rr}
    return 'VERIFIED' if len(identities)==1 and all(next(iter(identities))) else 'UNVERIFIED-HARDWARE'


def comparison_files(files,bound,binding_hash):
    pointer=json.loads(files['v6b/status/A4-comparison.json']);require(pointer['status']=='COMPLETE','complete comparison pending')
    receipt=files[pointer['receipt']];require(t.sha(receipt)==pointer['sha256'],'comparison receipt changed')
    prefix=str(Path(pointer['receipt']).parent)+'/'
    record=json.loads(receipt);require(record['status']=='COMPLETE' and record['task']=='A4-comparison','incomplete comparison')
    for name,digest in record.get('artifacts',{}).items():require(t.sha(files[name])==digest,'comparison allocation artifact changed')
    coverage=json.loads(files[prefix+'coverage.json'])['scores']
    lane=json.loads(files[prefix+'lane/run/lane.json'])
    require(lane['status']=='COMPLETE','comparison lane incomplete')
    require(coverage==[list(r) for r in score_plan()],'complete comparison coverage differs')
    records={}
    for i,key in enumerate(score_plan()):
        root=prefix+'scores/'+str(i)+'/'
        manifest=json.loads(files[root+'manifest.json'])
        for name,digest in manifest['files'].items():
            # Compact collection compresses CSVs and hash-lists raw responses separately.
            raw=gzip.decompress(files[root+name+'.gz']) if name=='scores.csv' and root+name not in files else files.get(root+name)
            if raw is not None:require(t.sha(raw)==digest,'comparison member changed: '+root+name)
            else:require(name=='raw/responses.jsonl','comparison member missing: '+root+name)
        ident=json.loads(files[root+'identity.json'])
        require(ident['binding']==binding_hash and ident['technical_synthetic']==bound['technical_synthetic'],'comparison binding differs')
        raw=files.get(root+'scores.csv')
        if raw is None:raw=gzip.decompress(files[root+'scores.csv.gz'])
        rows=list(csv.DictReader(io.StringIO(raw.decode())))
        require(all((r['kind'],r['arm'],int(r['seed']),int(r['month']),r['mode'])==(key[0],key[1],key[2],key[3],key[5]) for r in rows),'comparison scientific row labels differ')
        measured=ident['physical']
        unverified=measured.get('allocation_nonce')!=record['allocation_nonce'] or measured.get('uuid')!=lane.get('uuid') or lane.get('job_id')!=record['allocation_nonce']
        for r in rows:r.update(_allocation_job=measured.get('allocation_nonce'),_allocation_uuid=measured.get('uuid'),_allocation_environment=measured.get('environment'),_hardware_unverified=unverified)
        records[key]=rows
    return records


def read_comparison(out,require_complete=True):
    try:source=status(out,'A4-comparison')
    except NotReady:
        if require_complete:raise
        return {},['A4-comparison']
    # Only the selected complete attempt is loaded; earlier failed/PENDING epochs never merge.
    names=[out/'binding.json',out/'v6b/status/A4-comparison.json',source/'receipt.json',source/'coverage.json',source/'lane/run/lane.json']
    names += [p for p in (source/'scores').rglob('*') if p.is_file() and 'raw' not in p.parts]
    files={str(p.relative_to(out)):p.read_bytes() for p in names}
    bound=json.loads((out/'binding.json').read_text())
    return comparison_files(files,bound,s.file_hash(out/'binding.json')),[]


def write_reading(a,records,summaries,costs,bound,binding_hash,root):
    from kit.a4_read import analyse
    result=analyse(records,summaries,costs,a.bootstrap)
    hw=hardware(records)
    result.update(hardware_status=hw,technical_synthetic=a.standin,binding_sha256=binding_hash,score_cells=len(records),
        approved_ceiling_gpu_hours=approved_ceiling(bound),ceiling_gpu_hours=bound['ceiling_gpu_hours'])
    if hw!='VERIFIED':result['verdict']='UNVERIFIED-HARDWARE'
    if a.standin:result['scientific_verdict']=None
    atomic(root/'summary.json',result)
    (root/'summary.md').write_text(('TECHNICAL STAND-IN ONLY\n\n' if a.standin else '')+result['verdict']+'\n\nAmendment 2026-11-02: seeds 101–103; df 2; two-sided t=4.302653; one-sided t=2.919986; 3/3 seeds; ≥3/5 monthly means (3 of the unchanged five months).\n\n'+result['note']+'\n')
    return result


def read_job(a):
    out=a.out;bound=binding(out,a.standin);records,_=read_comparison(out)
    reconcile(out,a.standin)
    summaries={p.parent.name:json.loads(p.read_text()) for p in (out/'runs').glob('*/summary.json')}
    costs=[json.loads(p.read_text()) for p in (out/'costs').glob('*.json')]
    return write_reading(a,records,summaries,costs,bound,s.file_hash(out/'binding.json'),out/'reading')


def read_archive(a,files,bound,binding_hash):
    records=comparison_files(files,bound,binding_hash)
    summaries={Path(n).parent.name:json.loads(raw) for n,raw in files.items() if n.startswith('runs/') and n.endswith('/summary.json') and '/partials/' not in n}
    costs=[json.loads(raw) for n,raw in files.items() if n.startswith('costs/') and n.endswith('.json')]
    return write_reading(a,records,summaries,costs,bound,binding_hash,a.out/'returned-reading')


def cost_allocation(out,path,value,hours):
    """Allocate one GPU's held cost across concurrent generations without triple billing."""
    row=row_by_id(value['task']);attempt=path.parent;parts=[]
    if row['phase']=='score':
        for generated in sorted((attempt/'scores').glob('*/generated.json')):
            record=json.loads(generated.read_text())
            if record['reused']:continue
            with (generated.parent/'scores.csv').open() as stream:meta=next(csv.DictReader(stream))
            parts.append((generated.parent.name,float(record['seconds']),dict(kind='scoring',arm=meta['arm'],seed=int(meta['seed']),
                month=int(meta['month']),kind_row=meta['kind'],generation_id=record['cache_key'],generated=True)))
        coverage=attempt/'coverage.json'
        specs=json.loads(coverage.read_text())['nll'] if coverage.exists() else []
        for index,spec in enumerate(specs):
            result=attempt/'nll'/str(index)/'result.json'
            if not result.exists():continue
            record=json.loads(result.read_text());arm,seed,month=spec['stage'].split('-')
            parts.append(('nll-'+str(index),float(record['seconds']),dict(kind='nll',arm=arm,seed=int(seed[1:]),month=int(month[1:]),which=spec['which'])))
    total=sum(weight for _,weight,_ in parts)
    for name,weight,meta in parts:
        atomic(out/'costs'/('v6b-'+attempt.name+'-'+name+'.json'),dict(meta,gpu_hours=hours*weight/total if total else 0.,
            allocation_nonce=attempt.name,technical_synthetic=value['technical_synthetic'],
            allocation_basis='single-GPU held time apportioned by process seconds; concurrent seconds normalised, not added'))
    atomic(out/'costs'/('v6b-'+attempt.name+'.json'),dict(kind='training' if row['phase']=='train' else 'allocation_idle',
        arm=row.get('arm'),seed=row.get('seed'),month=row.get('month'),gpu_hours=0. if total else hours,
        held_allocation_gpu_hours=hours,gpus=value['gpus'],seconds=value['allocation_ended_at']-value['allocation_started_at'],
        completed=value['status']=='COMPLETE',allocation_nonce=attempt.name,technical_synthetic=value['technical_synthetic']))


def visible_devices(ids,standin):
    """Use this allocation's measured UUIDs, converting UUID visibility for vLLM."""
    if standin:return ','.join(str(i) for i in range(len(ids)))
    if not ids:return ''
    from kit.m2_gpu import devices
    inverse={uuid:index for index,uuid in devices().items()}
    require(all(uuid in inverse for uuid in ids),'allocated GPU no longer visible')
    if len(ids)==1:os.environ['M2_ASSIGNED_GPU_UUID']=ids[0]
    return ','.join(inverse[uuid] for uuid in ids)


def duration_limit(doc,row):
    if row['gpus']==8:return doc['job_limits']['training']
    counts=row['expected_command_counts']
    # Four commands can require two complete waves at K=3. Fractional waves
    # would under-reserve the registered per-command hard duration limit.
    return math.ceil(counts['score']/3)*doc['job_limits']['scoring']+counts['nll']*doc['job_limits']['nll']+60


@contextmanager
def reservation(ids,standin,attempt):
    if not ids:
        yield ();return
    if not standin:
        from kit.m3_manifest import allocation_reservation
        with allocation_reservation(ids,check_idle=True) as fds:yield fds
        return
    # CPU rehearsals keep their fake descriptor files inside their worktree.
    with ExitStack() as stack:
        streams=[stack.enter_context((attempt/('lease-'+str(i))).open('a')) for i in range(len(ids))]
        yield tuple(stream.fileno() for stream in streams)


def reconcile(out,standin,admission=False,query_scheduler=True):
    """Actual allocation cost plus atomic live reservations; never wait for another job."""
    from kit.a4_common import _slurm_times
    held=0.
    for path in (out/'v6b/attempts').glob('*/*/receipt.json'):
        value=json.loads(path.read_text());gpus=value['gpus']
        if not gpus:continue
        end=value.get('allocation_ended_at')
        if end is None:
            if standin:end=value.get('ended_at')
            elif query_scheduler and (not admission or value.get('ended_at') is not None):
                end=_slurm_times(value['slurm_job_id'],value['allocation_started_at']).get('allocation_ended_at')
            if end is None:
                if not admission:raise NotReady('actual allocation end required: '+value['slurm_job_id'])
                # Other allocations can keep working under their already-reserved ceiling.
                # A hard kill keeps its reservation until actual terminal accounting arrives.
                elapsed=max(0,time.time()-value['allocation_started_at'])
                held+=max(value.get('reserved_gpu_hours',0),0. if standin else gpus*(elapsed+30)/3600)
                continue
            require(end>=value['allocation_started_at'],'allocation end precedes start')
            value['allocation_ended_at']=end;atomic(path,value)
            pointer=out/'v6b/status'/(value['task']+'.json')
            with lock(out/'v6b/status-locks'/value['task']):
                if pointer.exists():
                    current=json.loads(pointer.read_text())
                    if current['receipt']==str(path.relative_to(out)):current['sha256']=s.file_hash(path);atomic(pointer,current)
        hours=0. if standin else gpus*max(0,end-value['allocation_started_at'])/3600
        held+=hours
        if query_scheduler:cost_allocation(out,path,value,hours)
    return held


def run(a):
    cfg=config(a.config);out=Path(cfg['out']);row=row_by_id(a.task);standin=cfg.get('technical_synthetic',False)
    require(out.is_absolute(),'absolute durable 4A out required')
    # The launch document is the only authority source. Never trust the file's own hash.
    if not standin:
        from kit.v6_infra import verify
        seal=os.environ.get('KIT_V6_AUTHORITY_SHA')
        require(bool(seal),'4A requires KIT_V6_AUTHORITY_SHA from the bound launch')
        verify(seal);os.environ['M2_INFRA_AUTHORITY_SHA256']=seal
    if completed(out,a.task):
        print('COMPLETE:',a.task);return 0
    # No inherited historical assignment or nonce pins a later allocation's GPU.
    for key in ('M2_ASSIGNED_GPU_UUID','KIT_LANE_JOB','M3_GPU_LOCK_FDS','M3_NODE_POLICY'):
        os.environ.pop(key,None)
    attempt=out/'v6b/attempts'/a.task/uuid.uuid4().hex;attempt.mkdir(parents=True)
    started=time.time();job_id=os.environ.get('SLURM_JOB_ID','standin')
    require(standin or job_id!='standin','SLURM_JOB_ID required')
    start=os.environ.get('SLURM_JOB_START_TIME')
    if start is None and not standin:
        from kit.a4_common import _slurm_times
        start=_slurm_times(job_id).get('allocation_started_at')
        require(start is not None,'actual Slurm allocation start required')
    start=float(start) if start is not None else started
    require(start<=started,'allocation start is in the future')
    value=dict(task=a.task,gpus=row['gpus'],allocation_nonce=attempt.name,slurm_job_id=job_id,
        slurm_restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0')),
        allocation_started_at=start,allocation_ended_at=None,started_at=started,ended_at=None,status='RUNNING',technical_synthetic=standin)
    code=1;atomic(attempt/'receipt.json',value)
    try:
        for dep in row['depends_on']:
            if dep['type']=='afterok':status(out,dep['id'])
        with lock(out/'v6b/budget-lock'):
            # One row per scheduler allocation/run. Requeue runs have distinct starts.
            if not standin:
                others=[json.loads(p.read_text()) for p in (out/'v6b/attempts').glob('*/*/receipt.json') if p.parent!=attempt]
                require(not any(v['slurm_job_id']==job_id and v['allocation_started_at']==start for v in others),'one 4A row per allocation; resubmit with a fresh scheduler job')
            (attempt/'receipt.json').unlink()
            # Scheduler accounting, cost reduction and disk-heavy reconciliation
            # belong to CPU jobs. GPU admission uses cached ends/live reservations.
            used=reconcile(out,standin,admission=True,query_scheduler=row['gpus']==0)
            if row['gpus']:
                doc=binding(out,standin)
                limit=duration_limit(doc,row)
                # The duration is a hard allocation-work limit, never a licence to spend above approval.
                value['max_seconds']=limit
                value['reserved_gpu_hours']=0. if standin else row['gpus']*(limit+time.time()-start+30)/3600
                require(standin or used+value['reserved_gpu_hours']<=approved_ceiling(doc),'remaining approved 4A ceiling cannot cover this row')
            else:limit=None
            atomic(attempt/'receipt.json',value)
        from kit.m3_manifest import allocated_gpus
        ids=(['GPU-CPU-STANDIN-'+str(i) for i in range(row['gpus'])] if standin else allocated_gpus(row['gpus'])) if row['gpus'] else []
        visible=visible_devices(ids,standin)
        with reservation(ids,standin,attempt) as fds:
            env=dict(os.environ,M3_GPU_LOCK_FDS=json.dumps(fds),KIT_LANE_JOB=attempt.name,
                A4_ALLOCATION_NONCE=attempt.name,A4_STANDIN='1' if standin else '0',
                CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=visible)
            command=[cfg.get('trainer_python',sys.executable),'-I',str(KIT/'a4_jobs.py'),'execute','--config',str(a.config),'--task',a.task,'--attempt',str(attempt)]
            if row['phase'] in ('bind','score'):command[0]=cfg.get('inference_python',sys.executable)
            if env.get('UTIL_WATCH_ACTIVE')!='1':
                command=[cfg.get('job_python',sys.executable),'-I',str(KIT/'util_watch.py'),'--out',str(attempt/'util.jsonl'),'--expect-gpus',str(row['gpus']),'--',*command]
            with job_deadline(limit) if limit else nullcontext():
                code=gpu_run(command,env=env,pass_fds=fds).returncode
            value['status']='COMPLETE' if code==0 else 'PENDING' if code in (3,75) else 'FAILED'
    except NotReady as exc:value.update(status='PENDING',error=str(exc));code=PENDING;print('PENDING:',exc)
    except CorrectnessStop as exc:value.update(status='FAILED',error=str(exc));code=2;print('STOP:',exc)
    except BaseException:
        value['status']='FAILED';raise
    finally:
        value['ended_at']=time.time()
        if standin or not row['gpus']:value['allocation_ended_at']=value['ended_at']
        if code==0:
            artifacts=[]
            if (attempt/'manifest.json').exists():artifacts.append(attempt/'manifest.json')
            if row['phase']=='prepare':artifacts.append(out/'data'/stage_root(out,row).name/'manifest.json')
            if row['phase']=='score':artifacts.extend([attempt/'coverage.json',attempt/'lane/run/lane.json'])
            value['artifacts']={str(p.relative_to(out)):s.file_hash(p) for p in artifacts}
        atomic(attempt/'receipt.json',value)
        with lock(out/'v6b/status-locks'/a.task):
            # A duplicate already in flight must not replace a successful peer.
            if completed(out,a.task):code=0;print('COMPLETE:',a.task)
            else:atomic(out/'v6b/status'/(a.task+'.json'),dict(status=value['status'],receipt=str((attempt/'receipt.json').relative_to(out)),sha256=s.file_hash(attempt/'receipt.json')))
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('rows','run','execute'))
    parser.add_argument('--config',type=Path);parser.add_argument('--task');parser.add_argument('--attempt',type=Path);parser.add_argument('--out',type=Path)
    a=parser.parse_args()
    try:
        if a.action=='rows':
            limits=None
            if a.config:
                cfg=config(a.config)
                limits=dict(training=1.5*cfg['stage_gpu_hours']*3600/8,
                    scoring=cfg.get('score_max_seconds',600),nll=cfg.get('nll_max_seconds',600))
            atomic(a.out,inventory(limits));return 0
        if a.action=='run':return run(a)
        execute(a);return 0
    except NotReady as exc:print('PENDING:',exc);return PENDING
    except CorrectnessStop as exc:print('STOP:',exc);return 2

if __name__=='__main__':raise SystemExit(main())
