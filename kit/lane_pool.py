#!/usr/bin/env python3
"""Run ready, independent commands from one lane on exactly one allocated GPU.

Plan: [{"argv":["python","-I","kit/scorer.py",...],"requires":["export/config.json",...]}].
Paths and commands are relative to the submitting working directory. All prerequisites
are checked before querying the GPU. Missing => PENDING (exit 75), never wait/retry.
Use fresh output trees for every command; never merge scores from separate lane jobs.
The scheduler wrapper owns the per-GPU lease and util_watch.py invocation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid as uuid_module

FRACTIONS={1:None,2:.42,3:.31}   # .28 left 1.12 GiB KV < 1.69 GiB needed at max_model_len 12288 (smoke attempt 3)
PENDING=75
PENDING_CODES=(3,PENDING)  # existing kit NotReady convention and this runner's convention


class GPUQueryUnavailable(ValueError):
    """NVML returned no unambiguous UUID, rather than a measured mismatch."""


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=1,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def gpu_uuid():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    command=['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits']
    selector=None
    if visible is not None:
        tokens=visible.split(',')
        if len(tokens)!=1 or not tokens[0] or not (tokens[0].isdigit() or tokens[0].startswith('GPU-')):
            raise ValueError('exactly one visible GPU required')
        selector=tokens[0]
    scheduler=os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS')
    if scheduler and len(scheduler.split(','))==1:selector=scheduler
    result=None
    for attempt in range(3):
        try:
            result=subprocess.run(command+(['--id='+selector] if selector else []),check=True,capture_output=True,text=True,timeout=30)
            break
        except subprocess.CalledProcessError:
            # Device cgroups may reject the physical selector.
            try:
                result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=30)
                break
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired,OSError):pass
        except (subprocess.TimeoutExpired,OSError):pass
    if result is None:return None  # unavailable telemetry is not a UUID mismatch
    rows=result.stdout.strip().splitlines()
    if len(rows)!=1 or not rows[0].strip().startswith('GPU-'):
        raise GPUQueryUnavailable('exactly one physical GPU UUID required')
    uuid=rows[0].strip()
    if visible and visible.startswith('GPU-') and uuid!=visible:
        raise ValueError('GPU UUID differs from visible allocation')
    return uuid


def recheck_uuid():
    # Admission already validated the allocation scope. A malformed later NVML
    # response is unknown telemetry, not evidence that the physical GPU changed.
    try:return gpu_uuid()
    except GPUQueryUnavailable:return None


def load_plan(path):
    raw=path.read_bytes();plan=json.loads(raw)
    if not isinstance(plan,list) or not plan:raise ValueError('nonempty ordered command list required')
    for command in plan:
        if not isinstance(command,dict) or set(command)-{'argv','requires'}:
            raise ValueError('command must contain argv and optional requires only')
        argv=command.get('argv');requires=command.get('requires',[])
        if not isinstance(argv,list) or not argv or any(not isinstance(v,str) or not v for v in argv):
            raise ValueError('argv must be nonempty strings; shell commands are not accepted')
        if not isinstance(requires,list) or any(not isinstance(v,str) or not v for v in requires):
            raise ValueError('requires must be a list of prerequisite paths')
    return plan,hashlib.sha256(raw).hexdigest()


def run(plan_path,concurrency,out,stagger_seconds=20,timeout_seconds=None):
    if concurrency not in FRACTIONS:raise ValueError('concurrency must be 1, 2 or 3')
    if not 0<=stagger_seconds<=60:raise ValueError('stagger must be between 0 and 60 seconds')
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds<=0):
        raise ValueError('timeout must be finite and positive')
    plan,digest=load_plan(Path(plan_path));out=Path(out);out.mkdir(parents=True,exist_ok=False)
    commands=[dict(index=i,argv=c['argv'],requires=c.get('requires',[]),uuid=None,
        exit_code=None,start=None,end=None,status='NOT_RUN') for i,c in enumerate(plan)]
    record=dict(schema='kit-lane-pool.v1',plan_sha256=digest,K=concurrency,uuid=None,
        gpu_memory_fraction=FRACTIONS[concurrency],stagger_seconds=stagger_seconds,
        start=time.time(),end=None,status='RUNNING',commands=commands)
    active={};next_index=0;last_start=None;error=None
    deadline=None if timeout_seconds is None else time.monotonic()+timeout_seconds
    try:
        missing=[p for c in commands for p in c['requires'] if not Path(p).exists()]
        if missing:
            record.update(status='PENDING',missing=missing)
        else:
            uuid=gpu_uuid();record['uuid']=uuid
            if uuid is None:
                record['status']='PENDING'
                record['missing']=['measured GPU UUID unavailable']
                return record
            env=dict(os.environ,M2_ASSIGNED_GPU_UUID=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                KIT_LANE_JOB=os.environ.get('KIT_LANE_JOB') or uuid_module.uuid4().hex)
            record['job_id']=env['KIT_LANE_JOB']   # keep the inherited index: vLLM rejects UUIDs
            if concurrency>1:env['KIT_GPU_MEMORY_FRACTION']=str(FRACTIONS[concurrency])
            else:env.pop('KIT_GPU_MEMORY_FRACTION',None)
            while active or next_index<len(commands):
                if deadline is not None and time.monotonic()>=deadline:
                    raise TimeoutError('lane deadline exceeded')
                for index,(process,log) in list(active.items()):
                    code=process.poll()
                    if code is None:continue
                    log.close();del active[index]
                    with (out/str(index)/'command.log').open(errors='replace') as stream:
                        pending=code in PENDING_CODES or code==0 and any(
                            line.startswith(('PENDING:','NOT READY:')) or line.strip()=='PENDING' for line in stream)
                    failed=code!=0 and code not in PENDING_CODES
                    commands[index].update(exit_code=code,end=time.time(),
                        status='FAILED' if failed else 'PENDING' if pending else 'COMPLETE')
                    if failed or pending:record['status']='FAILED' if failed or record['status']=='FAILED' else 'PENDING'
                if record['status']!='RUNNING':
                    if not active:break
                elif next_index<len(commands) and len(active)<concurrency and (
                        not active or last_start is None or time.monotonic()-last_start>=stagger_seconds):
                    command=commands[next_index]
                    # Recheck immediately before starting, but never wait for a prerequisite.
                    if any(not Path(p).exists() for p in command['requires']):
                        command['status']='PENDING';record['status']='PENDING';continue
                    current=recheck_uuid()
                    if current is not None and current!=uuid:raise ValueError('lane GPU UUID changed')
                    if current is None:record.setdefault('uuid_query_warnings',[]).append(next_index)
                    directory=out/str(next_index);directory.mkdir()
                    child_env=dict(env,KIT_ENGINE_RECEIPT_DIR=str((directory/'engines').resolve()))
                    log=(directory/'command.log').open('x')
                    command.update(uuid=uuid,start=time.time(),status='RUNNING')
                    try:process=subprocess.Popen(command['argv'],env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        pass_fds=tuple(json.loads(env.get('M3_GPU_LOCK_FDS','[]'))))
                    except BaseException:
                        log.close();command.update(status='FAILED',end=time.time());raise
                    active[next_index]=(process,log);next_index+=1;last_start=time.monotonic()
                    write(out/'lane.json',record)
                if active:time.sleep(.05)
            if record['status']=='RUNNING':
                current=recheck_uuid()
                if current is not None and current!=uuid:raise ValueError('lane GPU UUID changed at completion')
                if current is None:record.setdefault('uuid_query_warnings',[]).append('completion')
                record['status']='COMPLETE'
    except BaseException as exc:
        record['status']='FAILED';error=type(exc).__name__+': '+str(exc)
    finally:
        # Normally failed commands let their peers finish. Controller failure/interrupt
        # instead reaps whole process groups so no orphan keeps the allocation alive.
        for index,(process,log) in active.items():
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:code=process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);code=process.wait()
            log.close();commands[index].update(exit_code=code,end=time.time(),status='FAILED')
        record.update(end=time.time(),error=error);write(out/'lane.json',record)
    return record


def invocation(commands,directory,python=None,concurrency=3,stagger_seconds=20):
    """Freeze one fresh ordered plan. CPU callers may prepare it before allocation."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    plan=directory/'plan.json'
    write(plan,[c if isinstance(c,dict) else dict(argv=[str(v) for v in c]) for c in commands])
    return [str(python or sys.executable),'-I',str(Path(__file__).resolve()),'--plan',str(plan),
        '--concurrency',str(concurrency),'--stagger-seconds',str(stagger_seconds),'--out',str(directory/'run')]


def execute(commands,directory,*,env=None,pass_fds=(),runner=subprocess.run,concurrency=3,stagger_seconds=20):
    command=invocation(commands,directory,concurrency=concurrency,stagger_seconds=stagger_seconds)
    return runner(command,env=env,pass_fds=pass_fds).returncode


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,choices=FRACTIONS,required=True)
    parser.add_argument('--stagger-seconds',type=float,default=20)
    parser.add_argument('--timeout-seconds',type=float)
    args=parser.parse_args(argv)
    def stop(signum,frame):raise KeyboardInterrupt('lane controller signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,stop)
    try:record=run(args.plan,args.concurrency,args.out,args.stagger_seconds,args.timeout_seconds)
    finally:signal.signal(signal.SIGTERM,previous)
    print(record['status'])
    return 0 if record['status']=='COMPLETE' else PENDING if record['status']=='PENDING' else 1


if __name__=='__main__':raise SystemExit(main())
