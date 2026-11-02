#!/usr/bin/env python3
"""Observe an allocation without changing its command, dose or exit status.

python -I kit/util_watch.py --out util.jsonl -- <command...>
Samples and UUIDs are local evidence. Missing telemetry is unknown, never zero.
"""
from __future__ import annotations
import argparse
from collections import deque
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading


def visible(env):
    """CUDA identifiers are physical NVML indexes/UUIDs, never visible ordinals."""
    for key in ('CUDA_VISIBLE_DEVICES','SLURM_STEP_GPUS','SLURM_JOB_GPUS','NVIDIA_VISIBLE_DEVICES'):
        if key not in env:continue
        raw=env[key].strip()
        if raw.lower() in ('','-1','none','void','nodevfiles'):return []
        if raw.lower()=='all':continue
        ids=[s.strip() for s in raw.split(',')]
        if len(set(ids))!=len(ids) or any(not s for s in ids):raise ValueError('invalid allocated GPU identifiers')
        return ids
    if env.get('SLURM_GPUS_ON_NODE')=='0':return []
    return None  # nvidia-smi is constrained by the allocation's device cgroup.


def query(timeout):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL,timeout=timeout)
    rows=[]
    for fields in csv.reader(raw.splitlines()):
        if not fields:continue
        if len(fields)!=4:raise ValueError('invalid NVML row')
        index,uuid,util,mem=(v.strip() for v in fields)
        util,mem=float(util),float(mem)
        if not uuid or not math.isfinite(util) or not 0<=util<=100 or not math.isfinite(mem) or mem<0:
            raise ValueError('invalid NVML measurement')
        rows.append(dict(index=index,uuid=uuid,util=util,mem=mem))
    if len({r['uuid'] for r in rows})!=len(rows):raise ValueError('duplicate NVML UUID')
    return rows


def allocated(rows, ids, env):
    def select(tokens):
        result=[]
        for ident in tokens:
            matches=[r for r in rows if r['index']==ident or r['uuid']==ident or
                     ident.startswith(('GPU-','MIG-')) and r['uuid'].startswith(ident)]
            if len(matches)!=1:raise ValueError('allocated GPU missing or ambiguous')
            result.extend(matches)
        return result
    scheduler={k:env[k] for k in ('SLURM_STEP_GPUS','SLURM_JOB_GPUS') if k in env}
    expected=visible(scheduler) if scheduler else None
    scope=expected if expected else ids
    try:result=rows if scope is None else select(scope)
    except ValueError:
        # ConstrainDevices can expose physical NVML index 5 as CUDA ordinal 0.
        # Only use the whole inventory when its size equals the visible scope.
        if scope and all(v.isdigit() for v in scope) and len(rows)==len(scope):result=rows
        else:raise
    if ids and expected and ids!=expected:
        mismatch=(tuple(ids),tuple(expected))
        if mismatch not in _scope_warnings:
            print('WARN CUDA/Slurm GPU index mismatch; using measured allocation UUIDs',file=sys.stderr)
            _scope_warnings.add(mismatch)
    if len({r['uuid'] for r in result})!=len(result):raise ValueError('allocated GPU repeated')
    if scheduler:
        if expected is not None and len(expected)!=len(result):
            raise ValueError('CUDA and Slurm GPU counts differ')
    count=env.get('SLURM_GPUS_ON_NODE','')
    if count.isdigit() and len(result)!=int(count):raise ValueError('allocated GPU count differs')
    return result


_scope_warnings=set()


class Rolling:
    """Time-weighted activity; eviction statistics use complete windows only."""
    def __init__(self, window):
        self.window=window;self.segments=deque();self.total=0.;self.elapsed=0.
        self.minimum=None;self.below=0.

    def add(self, end, value):
        start=self.elapsed;duration=end-start
        if duration>0:
            self.segments.append((start,end,value));self.total+=duration*value
        # The window integral minus 70 * window length is piecewise linear.
        # Integrate threshold crossings between samples rather than charging a
        # whole interval from its endpoint (important around a rolling eviction).
        points=sorted({start,end,*[p+self.window for a,b,v in self.segments for p in (a,b)
            if start<p+self.window<end],*([self.window] if start<self.window<end else [])})
        def measurement(t):
            cut=max(0.,t-self.window);size=t-cut
            area=sum(max(0.,min(b,t)-max(a,cut))*v for a,b,v in self.segments)
            return area-70*size,area/size if size else value
        values=[measurement(t) for t in points]
        complete=[t for t in points if t>=self.window]
        measured=[measurement(t) for t in complete]
        for (a,b),((left,_),(right,_)) in zip(zip(complete,complete[1:]),zip(measured,measured[1:])):
            if left<0 and right<0:self.below+=b-a
            elif left<0:self.below+=(b-a)*(-left)/(right-left)
            elif right<0:self.below+=(b-a)*(-right)/(left-right)
        for _,mean in measured:self.minimum=mean if self.minimum is None else min(self.minimum,mean)
        self.elapsed=end;cut=max(0.,end-self.window)
        while self.segments and self.segments[0][1]<=cut:self.segments.popleft()
        mean=values[-1][1]
        return mean


def phases(path):
    result=[]
    try:
        for line in path.read_text().split('\n'):
            try:
                item=json.loads(line)
                if isinstance(item.get('phase'),str) and isinstance(item.get('t'),(int,float)) and math.isfinite(item['t']):
                    result.append(dict(phase=item['phase'],t=item['t']))
            except (ValueError,TypeError,AttributeError):pass
    except OSError:pass
    return result


def run(a):
    a.out=Path(a.out).resolve();a.out.parent.mkdir(parents=True,exist_ok=True)
    summary_path=a.out.with_suffix('.summary.json');phase_path=a.out.with_suffix('.phases.jsonl')
    # Never overwrite an earlier attempt or merge UUIDs across invocations.
    if summary_path.exists() or phase_path.exists():raise ValueError('use a fresh utilisation path')
    stream=a.out.open('x',buffering=1)
    env=dict(os.environ,UTIL_PHASE_FILE=str(phase_path),UTIL_WATCH_ACTIVE='1')
    child=None;pending=[];old={}
    def forward(signum,frame):
        if child is None:pending.append(signum)
        else:
            try:os.killpg(child.pid,signum)
            except ProcessLookupError:pass
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGQUIT):
        old[signum]=signal.signal(signum,forward)
    start=time.monotonic();started_at=time.time();stats=Rolling(a.window)
    uuids=None;ids=None;errors=0;last=None;last_warn=-math.inf;code=127;scope_valid=True
    def warn(mean,elapsed):
        nonlocal last_warn
        if mean<a.warn and elapsed-last_warn>=max(a.window,a.interval):
            print(f'WARN utilisation rolling mean {mean:.1f}% below {a.warn:g}%',file=sys.stderr,flush=True)
            last_warn=elapsed
    try:
        # Preserve inherited lease descriptors just as the original exec wrapper
        # did. Our telemetry stream is non-inheritable (Python's default).
        child=subprocess.Popen(a.command,env=env,start_new_session=True,close_fds=False)
        a._child=child
        for signum in pending:forward(signum,None)
        try:ids=[] if getattr(a,'expect_gpus',None)==0 else visible(env)
        except ValueError:errors+=1;scope_valid=False
        if ids==[]:
            uuids=[]
            try:
                if getattr(a,'expect_gpus',None)!=0:allocated([],[],env)
            except ValueError:errors+=1
        stopped=threading.Event();finalized=threading.Event()
        def sample():
            nonlocal uuids,errors,last
            while True:
                if ids!=[] and scope_valid:
                    try:
                        measured=query(max(1.,min(5.,a.interval)))
                        if finalized.is_set():return
                        rows=allocated(measured,ids,env)
                        if getattr(a,'expect_gpus',None) is not None and len(rows)!=a.expect_gpus:
                            raise ValueError('manifest GPU count differs')
                        current=[r['uuid'] for r in rows]
                        if uuids is not None and set(current)!=set(uuids):raise ValueError('allocated UUIDs changed')
                        uuids=current
                        if rows:
                            now=time.time()
                            for row in rows:stream.write(json.dumps(dict(t=now,gpu=row['uuid'],util=row['util'],mem=row['mem']))+'\n')
                            value=sum(r['util'] for r in rows)/len(rows);elapsed=time.monotonic()-start
                            mean=stats.add(elapsed,last if last is not None else value);last=value
                            if elapsed>=a.window:warn(mean,elapsed)
                    except Exception:
                        errors+=1
                        print('WARN utilisation telemetry unavailable; summary is incomplete',file=sys.stderr,flush=True) if errors==1 else None
                if stopped.wait(a.interval):break
        sampler=threading.Thread(target=sample,daemon=True,name='util-sampler')
        sampler.start()
        try:code=child.wait()
        finally:
            stopped.set();sampler.join(timeout=.5)
            finalized.set()
            if sampler.is_alive():
                errors+=1
                print('WARN utilisation sampler unavailable; child already finished',file=sys.stderr)
    except Exception as error:
        # A telemetry/logging error after launch must never terminate or rerun
        # scientific work. Wait for the existing child exactly once.
        print('WARN util_watch error; child continues unwatched: '+type(error).__name__,file=sys.stderr)
        errors+=1
        if child is not None:code=child.wait()
    finally:
        wall=time.monotonic()-start
        if last is not None:
            mean=stats.add(wall,last)
            if wall>=a.window:warn(mean,wall)
        complete=errors==0 and uuids is not None
        result=dict(job=os.environ.get('UTIL_JOB',os.environ.get('SLURM_JOB_ID')),gpus=len(uuids) if uuids is not None else len(ids) if ids is not None else None,
            gpu_uuids=uuids,wall_s=wall,mean_util=min(100.,max(0.,stats.total/wall)) if complete and last is not None else None,
            min_window_mean=min(100.,max(0.,stats.minimum)) if complete and stats.minimum is not None else None,
            seconds_below_70=min(wall,max(0.,stats.below)) if complete and last is not None else None,
            phases=phases(phase_path),telemetry_complete=complete,telemetry_errors=errors,
            started_at=started_at,returncode=code,interval_s=a.interval,window_s=a.window)
        try:summary_path.write_text(json.dumps(result,allow_nan=False)+'\n')
        except OSError:print('WARN utilisation summary could not be written',file=sys.stderr)
        finally:
            try:stream.close()
            except OSError:pass
            for signum,handler in old.items():signal.signal(signum,handler)
    return code if code>=0 else 128-code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--interval',type=float,default=5)
    p.add_argument('--window',type=float,default=300);p.add_argument('--warn',type=float,default=75)
    p.add_argument('--expect-gpus',type=int,choices=(0,1,2,8))
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args(argv)
    if a.command[:1]==['--']:a.command=a.command[1:]
    if not a.command or not all(math.isfinite(v) and v>0 for v in (a.interval,a.window)) or not math.isfinite(a.warn) or not 0<=a.warn<=100:
        p.error('command, positive interval/window and warn in 0..100 required')
    try:return run(a)
    except Exception as error:
        print('WARN util_watch unavailable; running child unwatched: '+type(error).__name__,file=sys.stderr)
        if getattr(a,'_child',None) is not None:
            code=a._child.wait()
            return code if code>=0 else 128-code
        # Setup failed before Popen. Keep the one-watcher flag so a nested
        # scientific entry does not try the same broken observer again.
        return subprocess.call(a.command,env=dict(os.environ,UTIL_WATCH_ACTIVE='1'),close_fds=False)


if __name__=='__main__':raise SystemExit(main())
