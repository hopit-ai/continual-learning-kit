#!/usr/bin/env python3
"""Run ready, independent commands from one lane on exactly one allocated GPU.

Plan: [{"argv":["python","-I","kit/scorer.py",...],"requires":["export/config.json",...]}].
Paths and commands are relative to the submitting working directory. All prerequisites
are checked at admission of each command. Missing => PENDING (exit 75), never wait for external work.
Use fresh output trees for every command; never merge scores from separate lane jobs.
The scheduler wrapper owns the per-GPU lease and util_watch.py invocation.
"""
import argparse
from contextlib import contextmanager
import fcntl
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
# Never follow TMPDIR: Slurm may give independent jobs different temp roots.
ENGINE_LOCK_ROOT=Path('/tmp')/('kit-engine-start-'+str(os.getuid()))


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


def capacity(command):
    """Only explicitly described, qualified vLLM lanes may overlap models.

    Missing engine descriptions (including HF references/PD) select K=1.
    A local model must identify Qwen3-8B in config.json; a directory name is
    insufficient. The constructor independently checks the actual engine kwargs.
    """
    engine=command.get('engine',{})
    if not isinstance(engine,dict):return 1
    if engine.get('backend')=='hf' and engine.get('profile') in ('reference-bf16','pd-fp32'):return 2
    if engine.get('backend')!='vllm':return 1
    model=engine.get('model');length=engine.get('max_model_len')
    if not isinstance(length,int) or isinstance(length,bool) or not 0<length<=12288:return 1
    qualified=2 if likelihood_command(command) else 3
    if model=='Qwen/Qwen3-8B':return qualified
    try:
        config=json.loads((Path(model)/'config.json').read_text())
        if all(config.get(k)==v for k,v in dict(model_type='qwen3',hidden_size=4096,
                num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,intermediate_size=12288).items()):return qualified
    except (OSError,ValueError,TypeError):pass
    return 1


def likelihood_command(command):
    """A diagnostic generation followed by likelihood scoring is a likelihood lane."""
    if command.get('engine',{}).get('workload')=='likelihood':return True
    argv=command.get('argv',[])
    if any(Path(str(v)).name=='m2_format.py' for v in argv):return True
    for flag in ('--panel-set','--panel','--kind'):
        values=[argv[i+1] for i,v in enumerate(argv[:-1]) if v==flag]
        if values and values[-1] in ('hellaswag','knowledge-ll','knowledge','likelihood'):return True
    return False


def gpu_memory(uuid):
    """Measured free/total MiB for one physical UUID, including cgroup fallback."""
    command=['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits']
    for selectors in (['--id='+uuid],[]):
        try:
            raw=subprocess.check_output(command+selectors,text=True,stderr=subprocess.DEVNULL,timeout=5)
            rows=[r.split(',') for r in raw.splitlines() if r.strip()]
            matches=[r for r in rows if len(r)==3 and r[0].strip()==uuid]
            if len(matches)!=1:continue
            free,total=(float(v.strip()) for v in matches[0][1:])
            if all(math.isfinite(v) for v in (free,total)) and 0<=free<=total and total>0:
                return free,total
        except (OSError,subprocess.SubprocessError,ValueError):pass
    raise GPUQueryUnavailable('free memory for allocated UUID unavailable')


@contextmanager
def engine_startup(kwargs):
    """Hold a host-local physical-GPU lock through admission and full LLM startup.

    Running engines may overlap. Starting engines may not overlap, even across
    independent controllers or likelihood children in the same holder.
    """
    uuid=os.environ.get('M2_ASSIGNED_GPU_UUID')
    if not os.environ.get('KIT_LANE_JOB') or not uuid:
        yield None
        return
    timeout=float(os.environ.get('KIT_ENGINE_START_TIMEOUT_SECONDS','600'))
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('positive engine startup timeout required')
    concurrency=int(os.environ.get('KIT_LANE_CONCURRENCY','1'))
    if concurrency not in FRACTIONS:raise ValueError('engine concurrency must be 1, 2 or 3')
    started=time.monotonic()
    lock_timeout=(concurrency-1)*timeout
    lock_deadline=started+lock_timeout
    root=ENGINE_LOCK_ROOT
    root.mkdir(parents=True,exist_ok=True)
    path=root/(hashlib.sha256(uuid.encode()).hexdigest()+'.lock')
    with path.open('a') as lock:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()>=lock_deadline:
                    raise TimeoutError(f'engine startup lock wait deadline exceeded ({lock_timeout:g}s budget, K={concurrency})')
                time.sleep(.1)
        try:
            deadline=time.monotonic()+timeout
            fraction=float(kwargs.get('gpu_memory_utilization',.9))
            if not math.isfinite(fraction) or not 0<fraction<1:raise ValueError('invalid engine memory fraction')
            while True:
                try:free,total=gpu_memory(uuid)
                except GPUQueryUnavailable:free=total=None
                if free is not None and free>=total*fraction:break
                if time.monotonic()>=deadline:
                    cause='query unavailable' if free is None else f'free/total {free:g}/{total:g} MiB; required fraction {fraction:g}'
                    raise TimeoutError('engine memory share unavailable before startup deadline: '+cause)
                time.sleep(.2)
            yield dict(uuid=uuid,free_mib=free,total_mib=total,fraction=fraction,wait_s=time.monotonic()-started)
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)
def recheck_uuid():
    # Admission already validated the allocation scope. A malformed later NVML
    # response is unknown telemetry, not evidence that the physical GPU changed.
    try:return gpu_uuid()
    except GPUQueryUnavailable:return None


def load_plan(path):
    raw=path.read_bytes();plan=json.loads(raw)
    if not isinstance(plan,list) or not plan:raise ValueError('nonempty ordered command list required')
    for command in plan:
        if not isinstance(command,dict) or set(command)-{'argv','requires','engine'}:
            raise ValueError('command must contain argv and optional requires/engine only')
        argv=command.get('argv');requires=command.get('requires',[])
        if not isinstance(argv,list) or not argv or any(not isinstance(v,str) or not v for v in argv):
            raise ValueError('argv must be nonempty strings; shell commands are not accepted')
        if not isinstance(requires,list) or any(not isinstance(v,str) or not v for v in requires):
            raise ValueError('requires must be a list of prerequisite paths')
    return plan,hashlib.sha256(raw).hexdigest()


def run(plan_path,concurrency,out,stagger_seconds=20,timeout_seconds=None,resume=False):
    if concurrency not in FRACTIONS:raise ValueError('concurrency must be 1, 2 or 3')
    if not 0<=stagger_seconds<=60:raise ValueError('stagger must be between 0 and 60 seconds')
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds<=0):
        raise ValueError('timeout must be finite and positive')
    plan,digest=load_plan(Path(plan_path));out=Path(out)
    requested=concurrency;concurrency=min(concurrency,*(capacity(c) for c in plan))
    previous=None
    if resume:
        previous=json.loads((out/'lane.json').read_text())
        if previous['plan_sha256']!=digest or previous['status']!='PENDING' or not os.environ.get('SLURM_JOB_ID') or previous.get('allocation_id')!=os.environ['SLURM_JOB_ID'] or not os.environ.get('KIT_LANE_JOB') or previous.get('job_id')!=os.environ['KIT_LANE_JOB']:
            raise ValueError('resume requires the same plan, allocation and lane nonce; rescore the complete contrast in a new allocation')
        if previous['uuid']!=gpu_uuid():raise ValueError('resume GPU UUID changed')
    else:out.mkdir(parents=True,exist_ok=False)
    commands=[dict(index=i,argv=c['argv'],requires=c.get('requires',[]),uuid=None,
        exit_code=None,start=None,end=None,status='NOT_RUN') for i,c in enumerate(plan)]
    if previous:
        for i,c in enumerate(previous['commands']):
            if c['status']=='COMPLETE':commands[i]=c
    record=dict(schema='kit-lane-pool.v1',plan_sha256=digest,K=concurrency,requested_K=requested,uuid=None,
        gpu_memory_fraction=FRACTIONS[concurrency],stagger_seconds=stagger_seconds,
        start=previous['start'] if previous else time.time(),end=None,status='RUNNING',commands=commands,
        allocation_id=os.environ.get('SLURM_JOB_ID'))
    active={};next_index=0;last_start=None;error=None
    deadline=None if timeout_seconds is None else time.monotonic()+timeout_seconds
    try:
        missing=[p for p in commands[0]['requires'] if not Path(p).exists()]
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
            env['KIT_LANE_CONCURRENCY']=str(concurrency)
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
                    if command['status']=='COMPLETE':
                        next_index+=1;continue
                    # Recheck immediately before starting, but never wait for a prerequisite.
                    missing=[p for p in command['requires'] if not Path(p).exists()]
                    if missing:
                        if active:
                            time.sleep(.05);continue  # a running producer may supply it
                        command['status']='PENDING';record.update(status='PENDING',missing=missing);continue
                    current=recheck_uuid()
                    if current is not None and current!=uuid:raise ValueError('lane GPU UUID changed')
                    if current is None:record.setdefault('uuid_query_warnings',[]).append(next_index)
                    directory=out/str(next_index)
                    if directory.exists():directory.rename(out/(str(next_index)+'-attempt-'+str(time.time_ns())))
                    directory.mkdir()
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


def invocation(commands,directory,python=None,concurrency=3,stagger_seconds=0):
    """Freeze one fresh ordered plan. CPU callers may prepare it before allocation."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    plan=directory/'plan.json'
    commands=[c if isinstance(c,dict) else dict(argv=[str(v) for v in c]) for c in commands]
    write(plan,commands)
    concurrency=min(concurrency,*(capacity(c) for c in commands))
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
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    def stop(signum,frame):raise KeyboardInterrupt('lane controller signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,stop)
    try:record=run(args.plan,args.concurrency,args.out,args.stagger_seconds,args.timeout_seconds,args.resume)
    finally:signal.signal(signal.SIGTERM,previous)
    print(record['status'])
    return 0 if record['status']=='COMPLETE' else PENDING if record['status']=='PENDING' else 1


if __name__=='__main__':raise SystemExit(main())
