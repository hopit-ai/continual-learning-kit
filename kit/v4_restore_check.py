"""Real-stack qualification of saved-state restoration; fail closed without proof."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

COMPONENTS={'model','optimizer','rng','scheduler','dataloader'}


from kit.v4_phase0_timing import shared_limit

def state_hash(value):
    """Canonical semantic tensor/container hash, independent of torch.save bytes."""
    digest=hashlib.sha256()
    def visit(item):
        import torch
        if hasattr(item,'local_shards'):
            visit([(str(s.metadata),s.tensor) for s in item.local_shards()]);return
        if hasattr(item,'to_local'):item=item.to_local()
        if isinstance(item,torch.Tensor):
            tensor=item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype,tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes());return
        if isinstance(item,dict):
            for key in sorted(item,key=lambda k:(type(k).__name__,repr(k))):visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:visit(child)
        elif hasattr(item,'dtype') and hasattr(item,'tobytes'):
            digest.update(str((item.dtype,item.shape)).encode());digest.update(item.tobytes())
        else:digest.update((type(item).__name__+':'+repr(item)+';').encode())
    visit(value);return digest.hexdigest()


def checkpoint_step(path):
    """Step from the nearest global_step_N component: global_step_N, global_step_N/actor or global_step_N/v4-ema."""
    for part in (Path(path),*Path(path).parents):
        if part.name.startswith('global_step_'):return int(part.name.removeprefix('global_step_'))
    raise ValueError(f'no global_step_N component in checkpoint path: {path}')


def install_checkpoint_probe():
    """Observe LIVE restored state, not just the files handed to a loader."""
    if not os.environ.get('V4_RESTORE_PROBE_OUT'):return
    import functools
    import torch
    import verl.utils.checkpoint.fsdp_checkpoint_manager as module
    cls=module.FSDPCheckpointManager
    if getattr(cls,'_v4_restoration_probe',False):return
    def observe(owner,path,event):
        from kit.v4_teacher import atomic_json
        kind='EMA' if Path(path).name=='v4-ema' else 'model'
        components={}
        cfg=module.ShardedStateDictConfig(offload_to_cpu=True)
        optim=module.ShardedOptimStateDictConfig(offload_to_cpu=True)
        with module.get_fsdp_state_ctx(owner.model,module.StateDictType.SHARDED_STATE_DICT,cfg,optim):
            components[kind]=state_hash(owner.model.state_dict())
            if owner.optimizer is not None:components['optimizer']=state_hash(owner.optimizer.state_dict())
        if kind=='model':
            components['rng']=state_hash(owner.get_rng_state())
            components['scheduler']=state_hash(owner.lr_scheduler.state_dict() if owner.lr_scheduler else None)
        out=Path(os.environ['V4_RESTORE_PROBE_OUT']);out.mkdir(parents=True,exist_ok=True)
        leg=os.environ['V4_RESTORE_LEG']
        step=2 if event=='loaded' else checkpoint_step(path)
        atomic_json(out/f'{leg}-{event}-{kind}-s{step}-r{owner.rank}.json',
                    {'leg':leg,'event':event,'step':step,'rank':owner.rank,'components':components})
    for name,event in (('save_checkpoint','saved'),('load_checkpoint','loaded')):
        original=getattr(cls,name)
        def wrapped(self,local_path,*args,_original=original,_event=event,**kwargs):
            result=_original(self,local_path,*args,**kwargs)
            if local_path is not None:observe(self,local_path,_event)
            return result
        setattr(cls,name,functools.wraps(original)(wrapped))
    cls._v4_restoration_probe=True


def install_trainer_probe(cls, *, save, load, sft=False):
    if not os.environ.get('V4_RESTORE_PROBE_OUT') or getattr(cls,'_v4_loader_probe',False):return
    import functools
    from kit.v4_teacher import atomic_json
    for method,event in ((save,'saved'),(load,'loaded')):
        original=getattr(cls,method)
        def wrapped(self,*args,_original=original,_event=event,**kwargs):
            result=_original(self,*args,**kwargs)
            leg=os.environ['V4_RESTORE_LEG'];out=Path(os.environ['V4_RESTORE_PROBE_OUT'])
            step=int(kwargs.get('step',args[0] if args else getattr(self,'global_steps',getattr(self,'resume_global_step',0))))
            if _event=='loaded':step=int(result or getattr(self,'resume_global_step',0) if sft else getattr(self,'global_steps',0))
            import torch
            rank=torch.distributed.get_rank() if sft and torch.distributed.is_initialized() else 0
            if step==2 and rank==0:
                atomic_json(out/f'{leg}-{_event}-dataloader-s2.json',{'leg':leg,'event':_event,'step':2,
                    'components':{'dataloader':state_hash(self.train_dataloader.state_dict())}})
                if _event=='saved' and leg=='killed':
                    atomic_json(out/'ready-to-kill.json',{'step':2,'checkpoint_durable':True})
                    # The external qualification driver sends SIGKILL after this
                    # checkpoint boundary; no graceful finalizer simulates it.
                    while True:time.sleep(1)
            return result
        setattr(cls,method,functools.wraps(original)(wrapped))
    cls._v4_loader_probe=True


def compare_evidence(files,arm):
    """All ranks, all state classes, actual SIGKILL, and next-step loss agree."""
    def doc(name):return json.loads(files[name])
    components=COMPONENTS|({'EMA'} if arm in ('S','D','D-R') else set())
    observed={}
    for leg,event in (('uninterrupted','saved'),('killed','saved'),('resumed','loaded')):
        values={}
        for rank in range(8):
            record=doc(f'{leg}-{event}-model-s2-r{rank}.json')
            if record['rank']!=rank or record['step']!=2:raise ValueError('restoration rank/step differs')
            for component,digest in record['components'].items():values[f'{component}:r{rank}']=digest
            if arm in ('S','D','D-R'):values[f'EMA:r{rank}']=doc(f'{leg}-{event}-EMA-s2-r{rank}.json')['components']['EMA']
        values['dataloader']=doc(f'{leg}-{event}-dataloader-s2.json')['components']['dataloader']
        expected={f'{c}:r{rank}' for c in components-{'dataloader'} for rank in range(8)}|{'dataloader'}
        if set(values)!=expected or any(not isinstance(d,str) or len(d)!=64 for d in values.values()):raise ValueError('restoration state components incomplete')
        observed[leg]=values
    if observed['uninterrupted']!=observed['killed'] or observed['killed']!=observed['resumed']:
        raise ValueError('restoration state hashes differ')
    termination=doc('kill.json')
    if termination.get('gpu_idle_verified') is not True or termination.get('gpu_query',{}).get('returncode')!=0 or termination.get('gpu_query',{}).get('stdout','missing').strip():raise ValueError('restoration workers still own GPUs')
    if termination.get('launcher_returncode')!=-signal.SIGKILL or termination.get('ray_stop_returncode')!=0:raise ValueError('restoration kill termination not verified')
    if termination.get('signal')!=signal.SIGKILL or termination.get('terminated') is not True or termination.get('saved_step')!=2:
        raise ValueError('restoration check requires actual verified kill')
    loss=doc('next-loss.json')
    left,right=loss['uninterrupted'],loss['resumed']
    key='train/loss' if arm in 'FR' else 'actor/pg_loss'
    for leg,value in (('uninterrupted',left),('resumed',right)):
        metrics=[json.loads(line) for line in files[leg+'-metrics.jsonl'].decode().split('\n') if line.strip()]
        next_step=[row.get('data',row)[key] for row in metrics if row.get('step')==3]
        if next_step!=[value]:raise ValueError('restoration next loss differs from raw step metrics')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (left,right)):
        raise ValueError('restoration next-step loss missing/nonfinite')
    if not math.isclose(left,right,rel_tol=1e-6,abs_tol=1e-7):raise ValueError('restoration next-step loss differs')
    if loss.get('step')!=3:raise ValueError('restoration next-step dose differs')
    return {'passed':True,'state_hashes':observed['resumed'],'next_step_loss':right,'steps_before_kill':2,'steps_after_resume':1}


def arm_evidence(archive,arm):
    prefix='v4/report-qualification/restoration/'+arm+'/'
    return {name[len(prefix):]:raw for name,raw in archive.files.items()
            if name.startswith(prefix) and name.endswith(('.json','.jsonl')) and name!=prefix+'verdict.json'}


def verify_arm_archive(archive,arm):
    """Recompute a durable arm verdict from its bound raw observations, without weights."""
    from kit.v4_arms import ARM_TABLE
    from kit.v4_qualification import departures_hash
    if arm not in ARM_TABLE:raise ValueError('unknown restoration arm')
    receipt=archive.json('v4/report-qualification/restoration/'+arm+'/verdict.json')
    if receipt.get('schema')!='v4-restoration-arm.v1' or receipt.get('arm')!=arm or receipt.get('real_stack') is not True or receipt.get('passed') is not True:
        raise ValueError('restoration arm verdict not passed/verified: '+arm)
    if receipt.get('departures_sha256')!=departures_hash() or receipt.get('environment')!=archive.json('v4/report-qualification/environment.json'):
        raise ValueError('restoration arm recipe/environment changed')
    files=arm_evidence(archive,arm)
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
    if not hashes or receipt.get('evidence_sha256')!=hashes:raise ValueError('restoration arm evidence binding differs')
    result=compare_evidence(files,arm)
    if receipt.get('result')!=result:raise ValueError('restoration arm comparison verdict differs')
    return result


def record_arm_verdict(work,arm):
    from kit.v4_archive import DirectoryEvidence
    from kit.v4_qualification import departures_hash
    from kit.runner import write_durably
    path=Path(work)/'v4/report-qualification/restoration'/arm/'verdict.json'
    archive=DirectoryEvidence(work)
    if path.exists():
        verify_arm_archive(archive,arm)
        return archive.json(path.relative_to(work).as_posix())
    files=arm_evidence(archive,arm)
    receipt={'schema':'v4-restoration-arm.v1','arm':arm,'real_stack':True,
             'environment':archive.json('v4/report-qualification/environment.json'),
             'departures_sha256':departures_hash(),
             'evidence_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}}
    try:result=compare_evidence(files,arm)
    except (ValueError,KeyError,TypeError) as exc:
        write_durably(path,{**receipt,'passed':False,'error':str(exc)})
        raise
    write_durably(path,{**receipt,'passed':result['passed'],'result':result})
    verify_arm_archive(DirectoryEvidence(work),arm) # verify the recorded bytes before deletion/admission
    return json.loads(path.read_text())


def record_full_verdict(work):
    from kit.v4_archive import DirectoryEvidence
    from kit.v4_qualification import departures_hash
    from kit.v4_finqa_campaign import ALL_ARMS
    from kit.runner import write_durably
    archive=DirectoryEvidence(work);root='v4/report-qualification/restoration/'
    results={arm:verify_arm_archive(archive,arm) for arm in ALL_ARMS}
    bindings={arm:{'path':root+arm+'/verdict.json','sha256':hashlib.sha256(archive.files[root+arm+'/verdict.json']).hexdigest()} for arm in ALL_ARMS}
    receipt={'schema':'v4-restoration-qualification.v2','real_stack':True,
             'passed':all(result['passed'] for result in results.values()),
             'environment':archive.json('v4/report-qualification/environment.json'),
             'departures_sha256':departures_hash(),'results':results,'arm_verdicts':bindings}
    write_durably(Path(work)/root/'qualification.json',receipt)
    verify_archive(DirectoryEvidence(work))
    return receipt


def verify_archive(archive):
    root='v4/report-qualification/restoration/';receipt=archive.json(root+'qualification.json')
    if receipt.get('real_stack') is not True or receipt.get('passed') is not True:raise ValueError('restoration qualification not passed')
    from kit.v4_qualification import departures_hash
    if receipt.get('departures_sha256')!=departures_hash():raise ValueError('restoration recipe changed')
    env=archive.json('v4/report-qualification/environment.json')
    if receipt.get('environment')!=env:raise ValueError('restoration environment changed')
    from kit.v4_retention import POLICY
    marker='v4/report-inputs/retention-policy.json'
    if marker in archive.files and archive.json(marker)==POLICY and receipt.get('schema')!='v4-restoration-qualification.v2':
        raise ValueError('main restoration needs bound per-arm verdicts')
    if receipt.get('schema')=='v4-restoration-qualification.v2':
        from kit.v4_finqa_campaign import ALL_ARMS
        if set(receipt.get('arm_verdicts',{}))!=set(ALL_ARMS):raise ValueError('restoration arm verdicts incomplete')
        output={}
        for arm in ALL_ARMS:
            path=root+arm+'/verdict.json'
            binding={'path':path,'sha256':hashlib.sha256(archive.files.get(path,b'')).hexdigest()}
            if receipt['arm_verdicts'][arm]!=binding:raise ValueError('restoration full verdict arm binding differs')
            output[arm]=verify_arm_archive(archive,arm)
        if receipt.get('results')!=output or receipt['passed']!=all(r['passed'] for r in output.values()):
            raise ValueError('restoration full verdict differs from per-arm conjunction')
        return output
    output={}
    for arm in (('S','F','R','D','D-R') if 'v4/report-inputs/general-v1-panel.jsonl' in archive.files else 'SFRD'):
        prefix=root+arm+'/'
        files={name[len(prefix):]:raw for name,raw in archive.files.items() if name.startswith(prefix)}
        output[arm]=compare_evidence(files,arm)
    return output


def restoration_launch_allowances():
    """Read C1's frozen row registration, or the unmeasured 450-second default."""
    from kit.v4_finqa_campaign import phase0_measured_launch_allowances
    raw=os.environ.get('V4_RESTORATION_LAUNCH_ALLOWANCES')
    if raw is None:return phase0_measured_launch_allowances()
    allowances=json.loads(raw)
    for key in ('qualification_launch_seconds','restoration_leg_seconds'):
        if type(allowances.get(key)) is not int or allowances[key]<=0:
            from kit.v4_finqa_policy import require
            require(False,'invalid restoration launch allowance','C1 timing')
            allowances[key]=phase0_measured_launch_allowances()[key]
    source=allowances['source']
    if source.get('measured') is True:
        from kit.v4_budget import number
        expected=math.ceil(number(source['slowest_two_step_launch_seconds'],'measured phase0 launch',positive=True)*number('1.5','allowance'))
        from kit.v4_finqa_campaign import RESTORATION_LEG_SECONDS
        if source.get('kind')!='verified_phase0_archive' or source.get('allowance')!=1.5 or len(source.get('archive_sha256',''))!=64 or len(source.get('raw_receipts',[]))!=4 or allowances['qualification_launch_seconds']!=expected or allowances['restoration_leg_seconds']!=max(RESTORATION_LEG_SECONDS,expected):
            from kit.v4_finqa_policy import require
            require(False,'restoration phase0 measurement registration differs','C1 timing')
    elif allowances!=phase0_measured_launch_allowances():
        from kit.v4_finqa_policy import require
        require(False,'unmeasured restoration allowance changed','C1 timing')
    return allowances


def preserve_incomplete_arm(work, root, arm, legs):
    """Move an interrupted attempt intact before reusing the canonical leg paths."""
    from kit.v4_finqa_policy import confine
    out=confine(work,root/arm)
    runs=[confine(work,Path(work)/'runs'/f'restoration-{arm}-{leg}') for leg in legs]
    if not (out.exists() and any(out.iterdir())) and not any(run.exists() for run in runs):return
    attempts=confine(work,root/'failed-attempts'/arm)
    number=1
    while (attempts/f'attempt-{number}').exists():number+=1
    saved=attempts/f'attempt-{number}';saved.mkdir(parents=True)
    if out.exists():out.rename(saved/'observations')
    (saved/'runs').mkdir()
    for run in runs:
        if run.exists():run.rename(saved/'runs'/run.name)


def run_check(work):
    """Separate bounded real-model legs for every launcher/state implementation."""
    from kit.v4_campaign_ops import paths, launcher_env
    from kit.v4_finqa_campaign import ALL_ARMS, RESTORATION_LEGS
    from kit.runner import bounded_command, write_durably
    from kit import v4_teacher as t, v4_sft
    import pyarrow.parquet as pq
    import shutil
    work=Path(work);base=paths(work,'qualification');root=base/'restoration';root.mkdir(parents=True,exist_ok=True)
    allowances=restoration_launch_allowances();leg_seconds=shared_limit(allowances['restoration_leg_seconds'])
    t.atomic_json(root/'launch-allowances.json',allowances)
    if (root/'qualification.json').exists():
        from kit.v4_archive import DirectoryEvidence
        result=verify_archive(DirectoryEvidence(work))
        if (work/'v4/report-inputs/phase0-binding.json').exists():
            from kit.v4_retention import finish_restoration
            finish_restoration(work)
        return result
    scheduled=base/'scheduled/restoration-finqa'
    if not scheduled.exists():
        rows=pq.read_table(base/'common/finqa.parquet').to_pylist()
        if not rows:raise ValueError('restoration check lacks verified training rows')
        ids=[str(row['extra_info']['index']) for row in rows]
        schedule=base/'restoration-schedule.json';t.atomic_json(schedule,{'seed':101,'steps':3,'profile':'technical-smoke','ids':[ids[i%len(ids)] for i in range(96)]})
        v4_sft.main(['--training-set',str(base/'common'),'--task','finqa','--schedule',str(schedule),
            '--out',str(scheduled),'--tokenizer-8b',os.environ['QWEN3_8B_TOKENIZER']])
        (scheduled/'dataset').mkdir();shutil.copyfile(scheduled/'train.parquet',scheduled/'dataset/train.parquet')
        shutil.copyfile(work/'v4/datasets/datasets/v4_finqa/test.parquet',scheduled/'dataset/test.parquet')
    results={}
    for arm in ALL_ARMS:
        out=root/arm;out.mkdir(parents=True,exist_ok=True)
        main=(work/'v4/report-inputs/phase0-binding.json').exists()
        if main and (out/'verdict.json').exists() and json.loads((out/'verdict.json').read_text()).get('passed') is True:
            from kit.v4_retention import finish_restoration_arm
            # Resume an interrupted retirement before starting any new arm.
            # Failed or changed verdicts refuse here and keep the arm's states.
            from kit.v4_finqa_policy import observe
            results[arm]=observe(finish_restoration_arm,work,arm) or {'passed':False,'warning':'retirement/comparison unavailable; states retained'}
            continue
        preserve_incomplete_arm(work,root,arm,RESTORATION_LEGS)
        out.mkdir(parents=True,exist_ok=True)
        for leg in RESTORATION_LEGS:
            name=f'restoration-{arm}-{leg}'
            env={**os.environ,**launcher_env(work,arm,name,scheduled,steps=3),
                'V4_RESTORE_PROBE_OUT':str(out),'V4_RESTORE_LEG':leg}
            if (work/'v4/report-inputs/phase0-binding.json').exists():
                env.update(V4_MAIN_RETENTION='restoration',KEEP_TRAINER_CKPT='1')
            from kit.v4_arms import checkpoint_path
            checkpoint=checkpoint_path(work/'runs'/f'restoration-{arm}-killed',arm,2)
            if leg=='resumed':env.update(V4_RESUME_PATH=str(checkpoint),V4_RETRY_ADMITTED='1')
            command=['bash',str(Path(__file__).with_name('run_v4.sh'))]
            if leg!='killed':
                result=bounded_command(command,timeout=leg_seconds,env=env,log=out/(leg+'.log'))
                if result['returncode']:raise ValueError('restoration real-stack '+leg+': '+arm)
            else:
                with (out/'killed.log').open('w') as log:
                    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    deadline=time.monotonic()+leg_seconds
                    while not (out/'ready-to-kill.json').exists():
                        if process.poll() is not None:raise ValueError('restoration trainer ended before kill boundary')
                        if time.monotonic()>deadline:
                            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=shared_limit(10))
                            raise ValueError('restoration checkpoint deadline')
                        time.sleep(.1)
                    from kit.v4_retry import saved_state
                    saved_state(checkpoint,arm)
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=shared_limit(10))
                    # Ray workers can have independent sessions. Stop this sole
                    # owned runtime before the next leg, within the contained row.
                    stopped=bounded_command(['ray','stop','--force'],timeout=shared_limit(60),env=env,log=out/'ray-stop.log')
                    if stopped['returncode']:raise ValueError('restoration kill termination unverified')
                    idle_deadline=time.monotonic()+shared_limit(60)
                    while True:
                        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=shared_limit(10))
                        if query.returncode==0 and not query.stdout.strip():break
                        if time.monotonic()>=idle_deadline:raise ValueError('restoration GPU workers not terminated')
                        time.sleep(.25)
                    t.atomic_json(out/'kill.json',{'signal':int(signal.SIGKILL),'terminated':True,'saved_step':2,
                        'launcher_returncode':process.returncode,'ray_stop_returncode':stopped['returncode'],'gpu_idle_verified':True,'gpu_query':{'returncode':query.returncode,'stdout':query.stdout,'stdout_sha256':hashlib.sha256(query.stdout.encode()).hexdigest()}})
        losses={}
        for leg in ('uninterrupted','resumed'):
            metric_path=work/'runs'/f'restoration-{arm}-{leg}'/'metrics.jsonl'
            shutil.copyfile(metric_path,out/(leg+'-metrics.jsonl'))
            records=t.read_jsonl(metric_path)
            metric=next(r.get('data',r) for r in records if r.get('step')==3)
            losses[leg]=metric['train/loss' if arm in 'FR' else 'actor/pg_loss']
        t.atomic_json(out/'next-loss.json',{'step':3,**losses})
        files={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
        if main:
            from kit.v4_finqa_policy import observe
            recorded=observe(record_arm_verdict,work,arm)
            from kit.v4_retention import finish_restoration_arm
            results[arm]=(observe(finish_restoration_arm,work,arm) if recorded else None) or {'passed':False,'warning':'comparison unavailable; states retained'}
        else:results[arm]=compare_evidence(files,arm)
    if (work/'v4/report-inputs/phase0-binding.json').exists():
        from kit.v4_finqa_policy import observe
        recorded=observe(record_full_verdict,work)
        if recorded:return recorded['results']
        write_durably(root/'progress.json',{'results':results,'warnings':'incomplete restoration comparison; no state deletion'})
        return results
    from kit.v4_qualification import departures_hash
    write_durably(root/'qualification.json',{'schema':'v4-restoration-qualification.v1','real_stack':True,
        'passed':True,'environment':json.loads((base/'environment.json').read_text()),
        'departures_sha256':departures_hash(),'results':results})
    if (work/'v4/report-inputs/phase0-binding.json').exists():
        from kit.v4_retention import finish_restoration
        finish_restoration(work)
    return results
