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
        step=2 if event=='loaded' else int(Path(path).parent.name.removeprefix('global_step_')) if kind=='EMA' else int(Path(path).name.removeprefix('global_step_'))
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
    components=COMPONENTS|({'EMA'} if arm in 'SD' else set())
    observed={}
    for leg,event in (('uninterrupted','saved'),('killed','saved'),('resumed','loaded')):
        values={}
        for rank in range(8):
            record=doc(f'{leg}-{event}-model-s2-r{rank}.json')
            if record['rank']!=rank or record['step']!=2:raise ValueError('restoration rank/step differs')
            for component,digest in record['components'].items():values[f'{component}:r{rank}']=digest
            if arm in 'SD':values[f'EMA:r{rank}']=doc(f'{leg}-{event}-EMA-s2-r{rank}.json')['components']['EMA']
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


def verify_archive(archive):
    root='v4/report-qualification/restoration/';receipt=archive.json(root+'qualification.json')
    if receipt.get('real_stack') is not True or receipt.get('passed') is not True:raise ValueError('restoration qualification not passed')
    from kit.v4_qualification import departures_hash
    if receipt.get('departures_sha256')!=departures_hash():raise ValueError('restoration recipe changed')
    env=archive.json('v4/report-qualification/environment.json')
    if receipt.get('environment')!=env:raise ValueError('restoration environment changed')
    output={}
    for arm in 'SFRD':
        prefix=root+arm+'/'
        files={name[len(prefix):]:raw for name,raw in archive.files.items() if name.startswith(prefix)}
        output[arm]=compare_evidence(files,arm)
    return output


def run_check(work):
    """Separate bounded real-model legs for every launcher/state implementation."""
    from kit.v4_campaign_ops import paths, launcher_env
    from kit.runner import bounded_command, write_durably
    from kit import v4_teacher as t, v4_sft
    import pyarrow.parquet as pq
    import shutil
    work=Path(work);base=paths(work,'qualification');root=base/'restoration';root.mkdir(parents=True,exist_ok=True)
    if (root/'qualification.json').exists():
        from kit.v4_archive import DirectoryEvidence
        return verify_archive(DirectoryEvidence(work))
    scheduled=base/'scheduled/restoration-chemistry'
    if not scheduled.exists():
        rows=pq.read_table(base/'common/chemistry.parquet').to_pylist()
        if not rows:raise ValueError('restoration check lacks verified training rows')
        ids=[str(row['extra_info']['index']) for row in rows]
        schedule=base/'restoration-schedule.json';t.atomic_json(schedule,{'seed':101,'steps':3,'profile':'technical-smoke','ids':[ids[i%len(ids)] for i in range(96)]})
        v4_sft.main(['--training-set',str(base/'common'),'--task','chemistry','--schedule',str(schedule),
            '--out',str(scheduled),'--tokenizer-8b',os.environ['QWEN3_8B_TOKENIZER']])
        (scheduled/'dataset').mkdir();shutil.copyfile(scheduled/'train.parquet',scheduled/'dataset/train.parquet')
        shutil.copyfile(work/'v4/datasets/datasets/v4_chem/test.parquet',scheduled/'dataset/test.parquet')
    results={}
    for arm in 'SFRD':
        out=root/arm;out.mkdir(parents=True,exist_ok=True)
        for leg in ('uninterrupted','killed','resumed'):
            name=f'restoration-{arm}-{leg}'
            env={**os.environ,**launcher_env(work,arm,name,scheduled,steps=3),
                'V4_RESTORE_PROBE_OUT':str(out),'V4_RESTORE_LEG':leg}
            checkpoint=work/'runs'/f'restoration-{arm}-killed'/({'S':'tool-sdpo','D':'sdft','F':'train','R':'train'}[arm])/'global_step_2'
            if leg=='resumed':env.update(V4_RESUME_PATH=str(checkpoint),V4_RETRY_ADMITTED='1')
            command=['bash',str(Path(__file__).with_name('run_v4.sh'))]
            if leg!='killed':
                result=bounded_command(command,timeout=1200,env=env,log=out/(leg+'.log'))
                if result['returncode']:raise ValueError('restoration real-stack '+leg+': '+arm)
            else:
                with (out/'killed.log').open('w') as log:
                    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    deadline=time.monotonic()+1200
                    while not (out/'ready-to-kill.json').exists():
                        if process.poll() is not None:raise ValueError('restoration trainer ended before kill boundary')
                        if time.monotonic()>deadline:
                            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                            raise ValueError('restoration checkpoint deadline')
                        time.sleep(.1)
                    from kit.v4_retry import saved_state
                    saved_state(checkpoint,arm)
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                    # Ray workers can have independent sessions. Stop this sole
                    # owned runtime before the next leg, within the contained row.
                    stopped=bounded_command(['ray','stop','--force'],timeout=60,env=env,log=out/'ray-stop.log')
                    if stopped['returncode']:raise ValueError('restoration kill termination unverified')
                    idle_deadline=time.monotonic()+60
                    while True:
                        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
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
        results[arm]=compare_evidence(files,arm)
    from kit.v4_qualification import departures_hash
    write_durably(root/'qualification.json',{'schema':'v4-restoration-qualification.v1','real_stack':True,
        'passed':True,'environment':json.loads((base/'environment.json').read_text()),
        'departures_sha256':departures_hash(),'results':results})
    return results
