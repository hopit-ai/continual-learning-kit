#!/usr/bin/env python3
"""Raw operation timing receipts and qualification budget measurements.

Every width comes from the allocation ledger. Timing observations identify the
raw operation and its units; no reported score, acceptance rate or budget summary
supplies a projection input.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit import v4_teacher as t
from kit.v4_budget import number


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def receipt(started,seconds,*,ok=True,**fields):
    return {'started_at':started,'ended_at':utc_now(),'wall_seconds':seconds,
            'allocation_id':os.environ.get('SLURM_JOB_ID'),'ok':ok,**fields}


def sample(ledger,record,units=1):
    """Convert one raw measurement with its observed allocation width and unit count."""
    if record.get('ok') is not True:raise ValueError('failed timing observation')
    job=str(record.get('allocation_id'))
    if job not in ledger['allocations']:raise ValueError('timing allocation absent from ledger')
    seconds=number(record.get('wall_seconds'),'measured operation seconds')
    count=number(units,'measured operation units',positive=True)
    return {'wall_seconds':str(seconds),'allocation_gpus':ledger['allocations'][job]['width'],'units':str(count)}


def qualification_timings(archive,*,completed=None):
    """Derive admission inputs from raw attempts, trainer metrics and operation receipts."""
    from kit.v4_campaign import TASKS, ARMS
    ledger=archive.json('k8b4/containment/allocation-ledger.json')
    base='v4/report-qualification/'
    observations=defaultdict(list)
    def add(key,record,units=1):observations[key].append(sample(ledger,record,units))
    for task in TASKS:
        for model in ('teacher','rewrite'):
            root=base+model+'/'+task+'/'
            raw=archive.jsonl(root+'raw_outputs.jsonl')
            timings=archive.jsonl(root+'timings.jsonl')
            by_attempt={(str(r['id']),r['attempt']):r for r in raw}
            if not raw:raise ValueError('missing corpus reload timing: '+model+':'+task)
            add(f'corpus_reload:{model}:{task}',raw[0]['reload_timing'])
            if len(by_attempt)!=len(raw) or len(timings)!=len(raw):raise ValueError('missing/duplicate attempt timing')
            seen=set()
            for entry in timings:
                key=(str(entry['id']),entry['attempt'])
                if key in seen or key not in by_attempt:raise ValueError('foreign/duplicate attempt timing')
                seen.add(key)
                if entry['raw_sha256']!=t.sha(json.dumps(by_attempt[key],sort_keys=True)):
                    raise ValueError('attempt timing is bound to different raw output')
                for component in ('generate','verify'):
                    add(f'{component}:{model}:{task}',entry[component])
            merge=archive.json(root+'serial-merge-timing.json')
            pool=archive.jsonl(root+'pool.jsonl')
            if (merge.get('verified_attempts')!=len(raw) or merge.get('raw_sha256')!=t.sha(json.dumps(raw,sort_keys=True))
                or merge.get('pool_sha256')!=t.sha(json.dumps(pool,sort_keys=True))):
                raise ValueError('serial merge timing journal/count binding differs')
            add(f'serial_merge:{model}:{task}',merge,len(raw))
            whole=archive.json(base+'timings/'+model+'-'+task+'.json')
            measured=sum(number(entry[part]['wall_seconds'],'attempt seconds') for entry in timings for part in ('generate','verify'))+number(raw[0]['reload_timing']['wall_seconds'],'corpus reload')
            shard_proof=archive.files.get(root+'shard-execution.json')
            if shard_proof is not None:
                execution=json.loads(shard_proof);shards=4 if model=='teacher' else 8
                if execution.get('shards')!=shards or execution.get('tp')!=(2 if model=='teacher' else 1):raise ValueError('corpus shard execution differs')
                lanes=[number(0,'empty lane') for _ in range(shards)];reloads=[number(0,'empty reload') for _ in range(shards)]
                for entry in timings:
                    lane=int(t.sha(str(entry['id'])),16)%shards
                    lanes[lane]+=sum(number(entry[part]['wall_seconds'],'shard timing') for part in ('generate','verify'))
                    reloads[lane]=max(reloads[lane],number(by_attempt[(str(entry['id']),entry['attempt'])]['reload_timing']['wall_seconds'],'shard reload'))
                measured=max(a+b for a,b in zip(lanes,reloads))
            residual=number(whole['wall_seconds'],'corpus row seconds')-measured-number(merge['wall_seconds'],'serial merge seconds')
            if residual<0:raise ValueError('negative corpus bookkeeping residual after serial merge')
            add(f'corpus_reload:{model}:{task}',{**whole,'wall_seconds':str(number(raw[0]['reload_timing']['wall_seconds'],'reload')+residual)})
        for arm in ARMS:
            ident=f'q-{arm}-{task}';root='runs/'+ident+'/'
            row=archive.json(base+'timings/train-'+ident+'.json')
            metrics=archive.jsonl(root+'metrics.jsonl')
            steps=[]
            for metric in metrics:
                data=metric.get('data',metric)
                field='train/time(s)' if arm in 'FR' else 'timing_s/step'
                if field not in data:continue
                step=metric.get('step')
                if step not in (1,2):raise ValueError('unexpected qualification timing step')
                if step in [i for i,_ in steps]:raise ValueError('duplicate training step timing')
                measured={**row,'wall_seconds':data[field]}
                add(f'train:{arm}:{task}',measured)
                steps.append((step,number(data[field],'step seconds')))
            if {step for step,_ in steps}!={1,2}:raise ValueError('missing timing: train:'+arm+':'+task)
            exports=archive.jsonl(root+'env/export-timings.jsonl')
            if len(exports)!=1 or exports[0]['step']!=2:raise ValueError('missing qualification export timing')
            for exported in exports:add(f'export:{arm}:{task}',exported)
            merged=archive.json(root+'env/recovered-merge-step2.json' if root+'env/recovered-merge-step2.json' in archive.files else root+'env/merge-timing.json')
            validation=archive.json(base+'timings/merge-'+ident+'.json')
            add(f'merge:{arm}:{task}',{**merged,'wall_seconds':str(number(merged['wall_seconds'],'merge')+number(validation['wall_seconds'],'merge validation'))})
            residual=number(row['wall_seconds'],'training row seconds')-sum(seconds for _,seconds in steps)-sum(number(e['wall_seconds'],'export seconds') for e in exports)-number(merged['wall_seconds'],'merge seconds')
            if residual<0:raise ValueError('negative startup residual: '+ident)
            add(f'startup:{arm}:{task}',{**row,'wall_seconds':str(residual)})
    for task in (*TASKS,'chemistry_probe'):
        for cap in (2048,8192):
            root=base+f'eval/baseline-{task}-{cap}/'
            responses=archive.jsonl(root+'responses.jsonl')
            timing=archive.json(root+'timing.json')
            add(f'eval:{task}:{cap}',timing['generate'],len(responses))
            whole=archive.json(base+f'timings/baseline-{task}-{cap}.json')
            residual=number(whole['wall_seconds'],'evaluation row')-number(timing['generate']['wall_seconds'],'generation')
            if residual<number(timing['reload']['wall_seconds'],'reload'):raise ValueError('negative evaluation bookkeeping residual')
            add(f'reload:{task}:{cap}',{**whole,'wall_seconds':str(residual)})
    probe_sizes=[]
    for task in TASKS:
        probe=archive.jsonl('v4/report-inputs/'+task+'-probe.jsonl')
        probe_sizes.append(len(probe))
        root=base+'probes/q-S-chemistry-'+task+'-0/'
        timing=archive.json(root+'timing.json')
        rows=archive.jsonl(root+'raw_outputs.jsonl')
        if {(r['id'],r['attempt']) for r in rows}!={(i['id'],attempt) for i in probe for attempt in range(1,9)}:
            raise ValueError('missing eight-attempt timing probe')
        for row in rows:add('probe:'+task+':2048',row['generate_timing'])
        whole=archive.json(base+'timings/profile-probe-'+task+'.json')
        residual=number(whole['wall_seconds'],'probe row')-sum(number(row['generate_timing']['wall_seconds'],'probe attempt') for row in rows)
        if residual<number(timing['reload']['wall_seconds'],'reload'):raise ValueError('negative probe bookkeeping residual')
        add('probe_reload:'+task,{**whole,'wall_seconds':str(residual)})
        root=base+'probes/conditioned-'+task+'/'
        conditioned=archive.jsonl(root+'raw_outputs.jsonl')
        if not conditioned:raise ValueError('missing conditioned probe timing: '+task)
        whole=archive.json(base+'timings/profile-conditioned-'+task+'.json')
        residual=number(whole['wall_seconds'],'conditioned row')-sum(number(row['generate_timing']['wall_seconds'],'conditioned attempt') for row in conditioned)
        if residual<number(archive.json(root+'timing.json')['reload']['wall_seconds'],'reload'):raise ValueError('negative conditioned bookkeeping residual')
        add('conditioned_reload:'+task,{**whole,'wall_seconds':str(residual)})
        for row in conditioned:add('conditioned_probe:'+task,row['generate_timing'])
    if probe_sizes!=[20,20]:raise ValueError('changed frozen probe size')
    # Full-cap waves run on all registered replicas, including idle allocation
    # cost. Recompute geometry from frozen pools rather than trusting a report.
    geometry={}
    for task in TASKS:
        full=archive.jsonl('v4/report-inputs/'+task+'-probe.jsonl')  # data_context supplies complete source pools
        full=getattr(archive,'expected_pools',{}).get(task,full)
        for model,shards in (('teacher',4),('rewrite',8)):
            samples=[]
            for index in range(shards):
                record=archive.json(base+f'wave-profile/{model}-{index}.json')
                cap=t.TEACHER_NEW_TOKENS if model=='teacher' else t.REWRITE_NEW_TOKENS
                input_cap=t.INPUT_CAP if model=='teacher' else t.REWRITE_INPUT_CAP
                if (record.get('technical_budget_profile') is not True or record.get('output_tokens')!=[cap]*t.H100_MAX_NUM_SEQS
                    or record.get('max_new_tokens')!=cap or record.get('input_cap')!=input_cap or record.get('tp')!=(2 if model=='teacher' else 1)):
                    raise ValueError('invalid full-cap wave profile')
                samples.append(sample(ledger,record))
            observations[f'generate:{model}:{task}']=samples
            geometry[model+':'+task]={'pool_size':len(full),'shards':shards,
                'shard_sizes':[len(t.select_shard(full,index,shards)) for index in range(shards)]}
    for task in TASKS:
        add('cpu:schedule:'+task,archive.json(base+'timings/schedule-q-S-'+task+'.json'))
    for operation in ('coverage','training-set','report','archive-pack'):
        add('cpu:'+operation,archive.json(base+'timings/'+operation+'.json'))
    # The main owner gate only validates frozen identities; conservatively bound
    # it by the CPU verify-prepare measurement, which hashes every static input.
    add('cpu:gate',archive.json(base+'timings/prepare.json'))
    # A fresh-process calibration exercises the complete main graph; its whole
    # receipt includes interpreter/import, graph preparation and record IO.
    profile=archive.json(base+'runner-profile/profile.json')
    from kit.v4_campaign import build
    main=build('main');expected=[row['id'] for row in main['rows']]
    dispatch=profile.get('dispatch',[])
    if (profile.get('cpu_profile_only') is not True or profile.get('rows')!=len(expected)
        or [row.get('row') for row in dispatch]!=expected):raise ValueError('runner profile graph differs')
    measured_profile=archive.json(base+'timings/runner-profile.json')
    registered_graph=archive.json('v4/report-inputs/runner-main-campaign.json')
    if registered_graph!=main or profile.get('campaign_sha256')!=t.sha(json.dumps(registered_graph,sort_keys=True,separators=(',',':'))):raise ValueError('runner profile campaign hash differs')
    duration=number(measured_profile['wall_seconds'],'runner profile duration')
    if duration<number(profile.get('prepare_seconds'),'runner prepare seconds')+sum(number(row.get('seconds'),'runner dispatch seconds') for row in dispatch):
        raise ValueError('runner profile shorter than measured preparation/dispatch')
    maximum=max(number(row.get('seconds'),'runner dispatch seconds') for row in dispatch)
    qualification=build('qualification')
    for row in qualification['rows']:
        name='campaign/v4-qualification/'+row['id']+'/attempt-1/runner-overhead.json'
        observation=archive.json(name)
        if observation.get('row')!=row['id']:raise ValueError('runner dispatch receipt row differs')
        observed=sample(ledger,observation)
        wrapper=number(observation.get('command_seconds'),'runner command seconds')
        operation_name=base+'timings/'+row['id']+'.json'
        if operation_name in archive.files:
            body=number(archive.json(operation_name)['wall_seconds'],'operation body seconds')
            if wrapper<body:raise ValueError('runner command shorter than measured body')
            wrapper-=body
        elif row['id'] in ('containment-selftest','PAUSE'):
            # The first selftest is separately billed at its full cap. PAUSE is
            # a tiny CPU command, but preserve its wrapper/import time.
            if row['id']=='containment-selftest':wrapper=number(0,'reused selftest')
        else:raise ValueError('missing runner operation body timing: '+row['id'])
        maximum=max(maximum,number(observed['wall_seconds'],'runner dispatch seconds')+wrapper)
    for stage in ('teacher','rewrite','scientific'):
        add('runner:dispatch:'+stage,{**measured_profile,'wall_seconds':str(maximum)})
        add('runner:prepare:'+stage,measured_profile)
    allocation_profile=archive.json('v4/report-inputs/allocation-profile.json')
    if allocation_profile.get('schema')!='v4-allocation-profile.v1':raise ValueError('allocation window registration missing')
    observations['allocation_windows']=allocation_profile['windows']
    elapsed=archive.json('k8b4/containment/containment-selftest.json')['elapsed_seconds']
    add('selftest',{'ok':True,'allocation_id':next(iter(ledger['allocations'])),'wall_seconds':max(600,elapsed)})
    observations['selftest:rewrite']=list(observations['selftest'])
    # Only the reader's reverified complete journals authorize sample deductions.
    # Smokes never count as completed scientific units.
    return {'timings':{**dict(observations),'batch_geometry':geometry},'completed':completed or {},'probe_n':20}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv);command=args.command
    if command[:1]==['--']:command=command[1:]
    if not command:parser.error('missing measured command')
    started=utc_now();clock=time.monotonic()
    result=subprocess.run(command,timeout=int(os.environ.get("V4_COMMAND_TIMEOUT","1800")))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    t.atomic_json(args.out,receipt(started,time.monotonic()-clock,ok=result.returncode==0,command=command,returncode=result.returncode))
    return result.returncode


if __name__=='__main__':raise SystemExit(main())


def append_receipt(path,record):
    """Serialize distributed telemetry writers without losing any rank's observation."""
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        handle.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())


def gpu_memory():
    """Record CUDA allocator peaks per physical GPU, retaining unknown identities."""
    import torch
    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get('RANK','0'))
    device=torch.cuda.current_device()
    props=torch.cuda.get_device_properties(device)
    return {'rank':rank,'uuid':str(getattr(props,'uuid','unknown')),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)}


def memory_receipt(operation,owner=None):
    """Observe cumulative CUDA peaks after initialization, update and final export."""
    if os.environ.get('V4_TELEMETRY')!='1':return
    out=Path(os.environ['VERL_FILE_LOGGER_PATH']).parent/'env/gpu-memory.jsonl'
    optimizer=getattr(owner,'optimizer',getattr(owner,'actor_optimizer',None))
    groups=[{'weight_decay':group.get('weight_decay'),'lr':group.get('lr')} for group in optimizer.param_groups] if optimizer is not None else []
    append_receipt(out,{**gpu_memory(),'operation':operation,'recorded_at':utc_now(),
                        'optimizer_parameter_groups':groups})


def install_telemetry(cls,*,updates=None,exports=None,memory_methods=()):
    """Wrap existing methods without changing their arguments, return values or recipes."""
    if os.environ.get('V4_TELEMETRY')!='1':return
    import functools
    for method in set(memory_methods)|({updates} if updates else set())|({exports} if exports else set()):
        original=getattr(cls,method)
        if getattr(original,'_v4_timed',False):continue
        def wrapper(self,*args,_original=original,_method=method,**kwargs):
            started=utc_now();clock=time.monotonic()
            tokens=training_tokens(args[0] if args else kwargs.get('batch',kwargs.get('data'))) if _method==updates else None
            optimizer=getattr(self,'optimizer',getattr(self,'actor_optimizer',None))
            used_lr=optimizer.param_groups[0]['lr'] if _method==updates and optimizer is not None else None
            result=_original(self,*args,**kwargs)
            if _method==updates and used_lr is not None and isinstance(result,dict):
                result['train/lr' if os.environ.get('KIT_V4_ARM') in 'FR' else 'actor/lr']=used_lr
            if _method in memory_methods or _method==updates:memory_receipt(_method,self)
            if tokens is not None:
                append_receipt(Path(os.environ['VERL_FILE_LOGGER_PATH']).parent/'env/training-tokens.jsonl',
                    {'rank':gpu_memory()['rank'],'step':getattr(self,'_v4_completed_updates',getattr(self,'_v4_updates',None)),**tokens})
            if _method==exports:
                rank=int(os.environ.get('RANK','0'))
                if rank==0:
                    step=kwargs.get('step',args[0] if args else getattr(self,'global_steps',None))
                    target=Path(os.environ['VERL_FILE_LOGGER_PATH']).parent/'env/export-timings.jsonl'
                    append_receipt(target,receipt(started,time.monotonic()-clock,step=step))
                    if os.environ.get('V4_CHECKPOINT_STATE')=='1':
                        from kit.v4_retry import saved_state
                        arm=os.environ['KIT_V4_ARM']
                        checkpoint=Path(self.config.trainer.default_local_dir)/f'global_step_{step}'
                        digest=saved_state(checkpoint,arm)
                        t.atomic_json(target.parent/f'saved-state-step{step}.json',
                            {'step':step,'arm':arm,'path':str(checkpoint),'state_sha256':digest,
                             'contents':['model','optimizer','extra','dataloader']+(['EMA'] if arm in 'SD' else [])})
            return result
        wrapped=functools.wraps(original)(wrapper);wrapped._v4_timed=True
        setattr(cls,method,wrapped)


def training_tokens(batch):
    """Read actual local-rank masks before the trainer consumes its batch."""
    tensors=getattr(batch,'batch',batch)
    if not hasattr(tensors,'keys'):return None
    if 'response_mask' in tensors:
        mask=tensors['response_mask'];supervised=mask
        if 'self_distillation_mask' in tensors:supervised=mask*tensors['self_distillation_mask'].reshape(-1,1)
        return {'generated_trajectories':int(mask.shape[0]),'generated_tokens':int(mask.sum().item()),
                'supervised_tokens':int(supervised.sum().item())}
    if 'loss_mask' in tensors:
        return {'generated_trajectories':0,'generated_tokens':0,
                'supervised_tokens':int(tensors['loss_mask'][:,1:].sum().item())}
    return None
