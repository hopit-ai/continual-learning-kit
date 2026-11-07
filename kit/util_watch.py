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


def use_holder_slice(env, expected):
    """Scope a one-GPU row to a holder only when explicitly requested or shared."""
    if expected != 1:return False
    if env.get('UTIL_HOLDER_SLICE') == '1':return True
    count=env.get('SLURM_GPUS_ON_NODE','')
    if count.isdigit() and int(count)>1:return True
    for key in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS'):
        tokens=env.get(key,'').split(',') if env.get(key) else []
        def size(token):
            bounds=token.strip().split('-')
            return max(0,int(bounds[1])-int(bounds[0])+1) if len(bounds)==2 and all(v.isdigit() for v in bounds) else 1
        if sum(size(token) for token in tokens)>1:return True
    return False


def visible(env):
    """Resolve the CUDA/cgroup namespace; Slurm IDs may be node-wide."""
    for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES'):
        if key not in env:continue
        raw=env[key].strip()
        if raw.lower() in ('','-1','none','void','nodevfiles'):return []
        if raw.lower()=='all':continue
        ids=[s.strip() for s in raw.split(',')]
        if len(set(ids))!=len(ids) or any(not s for s in ids):raise ValueError('invalid allocated GPU identifiers')
        return ids
    if env.get('SLURM_GPUS_ON_NODE')=='0':return []
    return None  # nvidia-smi is constrained by the allocation's device cgroup.


def query(timeout, selector=None):
    command=['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits']
    if selector is not None:command.append('--id='+selector)
    raw=subprocess.check_output(command,text=True,stderr=subprocess.DEVNULL,timeout=timeout)
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


def holder_scope(env, expected):
    """A holder slice owns one CUDA token; Slurm may describe its whole parent."""
    token=env.get('CUDA_VISIBLE_DEVICES','').strip()
    if expected not in (None,1) or not (token.isdigit() or token.startswith('GPU-')) or ',' in token:
        raise ValueError('holder-slice requires exactly one CUDA_VISIBLE_DEVICES GPU')
    return token,{k:v for k,v in env.items() if not k.startswith('SLURM_')}


def allocated(rows, ids, env, expected=None, *, holder_slice=False):
    def select(tokens):
        result=[]
        for ident in tokens:
            if holder_slice:
                # Holder CUDA ordinals can be renumbered relative to NVML's
                # physical indexes. A sparse inventory cannot resolve them.
                if ident.isdigit():
                    indexes=[r['index'] for r in rows]
                    if len(rows)!=1 and sorted(indexes)!=sorted(str(i) for i in range(len(rows))):
                        raise ValueError('holder GPU numeric namespace ambiguous')
                    matches=[r for r in rows if r['index']==ident]
                elif ident.startswith('GPU-'):
                    matches=[r for r in rows if r['uuid'].startswith(ident)]
                else:matches=[]
                if len(matches)!=1:raise ValueError('holder GPU missing or ambiguous')
                result.extend(matches)
                continue
            matches=[r for r in rows if r['index']==ident or r['uuid']==ident or
                     ident.startswith(('GPU-','MIG-')) and r['uuid'].startswith(ident)]
            if len(matches)!=1:
                # ConstrainDevices can renumber CUDA ordinals while NVML keeps
                # physical indexes. Only the complete cgroup list is safe here.
                if tokens==[str(i) for i in range(len(rows))]:return rows
                raise ValueError('allocated GPU missing or ambiguous')
            result.extend(matches)
        return result
    scheduler=[env[k].split(',') for k in ('SLURM_STEP_GPUS','SLURM_JOB_GPUS') if env.get(k)]
    if ids and any(tokens!=ids for tokens in scheduler):
        mismatch=(tuple(ids),tuple(tuple(v) for v in scheduler))
        if mismatch not in _scope_warnings:
            print('WARN CUDA/Slurm GPU index mismatch; using measured allocation UUIDs',file=sys.stderr)
            _scope_warnings.add(mismatch)
    result=rows if ids is None else select(ids)
    if len({r['uuid'] for r in result})!=len(result):raise ValueError('allocated GPU repeated')
    for key in ('SLURM_STEP_GPUS','SLURM_JOB_GPUS'):
        if key not in env:continue
        tokens=env[key].split(',') if env[key] else []
        def size(token):
            bounds=token.split('-')
            return int(bounds[1])-int(bounds[0])+1 if len(bounds)==2 and all(v.isdigit() for v in bounds) else 1
        if sum(size(t) for t in tokens)!=len(result):raise ValueError('Slurm GPU count differs')
    count=env.get('SLURM_GPUS_ON_NODE','')
    if count.isdigit() and len(result)!=int(count):raise ValueError('allocated GPU count differs')
    if expected is not None and len(result)!=expected:raise ValueError('manifest GPU count differs')
    return result


_scope_warnings=set()


class Rolling:
    """Time-weighted sample-and-hold activity; eviction uses full windows only."""
    def __init__(self, window):
        self.window=window;self.segments=deque();self.total=0.;self.elapsed=0.
        self.minimum=None;self.below=0.;self.startup_total=0.;self.startup_below=0.

    def add(self, end, value):
        start=self.elapsed;duration=end-start
        if duration>0:
            self.segments.append((start,end,value));self.total+=duration*value
            startup=max(0.,min(end,self.window)-start)
            self.startup_total+=startup*value
            if value<70:self.startup_below+=startup
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
        for (a,b),((left,_),(right,_)) in zip(zip(points,points[1:]),zip(values,values[1:])):
            if a<self.window:continue
            if left<0 and right<0:self.below+=b-a
            elif left<0:self.below+=(b-a)*(-left)/(right-left)
            elif right<0:self.below+=(b-a)*(-right)/(left-right)
        for t,(_,mean) in zip(points,values):
            if t>=self.window:self.minimum=mean if self.minimum is None else min(self.minimum,mean)
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
    if not hasattr(a,'expect_gpus'):a.expect_gpus=None
    if getattr(a,'startup_grace',None) is None:a.startup_grace=a.window
    holder_slice=getattr(a,'holder_slice',False) or os.environ.get('UTIL_HOLDER_SLICE')=='1'
    a.out=Path(a.out).resolve();a.out.parent.mkdir(parents=True,exist_ok=True)
    summary_path=a.out.with_suffix('.summary.json');phase_path=a.out.with_suffix('.phases.jsonl');attempt_path=a.out.with_suffix('.attempt.json')
    # Never overwrite an earlier attempt or merge UUIDs across invocations.
    if summary_path.exists() or phase_path.exists() or attempt_path.exists():raise ValueError('use a fresh utilisation path')
    stream=a.out.open('x',buffering=1)
    with attempt_path.open('x') as attempt:
        json.dump(dict(schema='kit-util-watch-attempt.v1',expected_gpus=a.expect_gpus,holder_slice=holder_slice),attempt)
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
    selector=None;scope_env=env;holder_queries=0
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
        expected=getattr(a,'expect_gpus',None)
        try:
            ids=[] if expected==0 else visible(env)
            if holder_slice and expected!=0:
                selector,scope_env=holder_scope(env,expected)
                ids=[selector]
        except ValueError:errors+=1;scope_valid=False
        if ids==[]:
            uuids=[]
            try:
                if expected!=0:allocated([],[],env,expected)
            except ValueError:errors+=1
        stopped=threading.Event();finalized=threading.Event()
        query_timeout=max(1.,min(5.,a.interval))
        def sample():
            nonlocal uuids,errors,last,holder_queries
            while True:
                if ids!=[] and scope_valid:
                    try:
                        try:measured=query(query_timeout)
                        except Exception:
                            if selector is None:raise
                            measured=[]
                        if not measured and selector is not None:
                            measured=query(query_timeout,selector)
                            holder_queries+=1
                            if len(measured)!=1:raise ValueError('holder GPU measurement unavailable')
                        if finalized.is_set():return
                        rows=allocated(measured,ids,scope_env,expected,holder_slice=holder_slice)
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
            stopped.set();sampler.join(timeout=query_timeout+1.)
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
        result=dict(schema='kit-util-watch.v1',expected_gpus=a.expect_gpus,job=os.environ.get('UTIL_JOB',os.environ.get('SLURM_JOB_ID')),gpus=len(uuids) if uuids is not None else len(ids) if ids is not None else None,
            gpu_uuids=uuids,wall_s=wall,mean_util=min(100.,max(0.,stats.total/wall)) if complete and last is not None else None,
            min_window_mean=min(100.,max(0.,stats.minimum)) if complete and stats.minimum is not None else None,
            seconds_below_70=min(wall,max(0.,stats.below)) if complete and last is not None else None,
            startup_wall_s=min(wall,a.window),startup_mean_util=stats.startup_total/min(wall,a.window) if complete and last is not None else None,
            startup_seconds_below_70=stats.startup_below if complete and last is not None else None,
            phases=phases(phase_path),telemetry_complete=complete,telemetry_errors=errors,
            started_at=started_at,returncode=code,interval_s=a.interval,window_s=a.window,startup_grace_seconds=a.startup_grace,
            holder_slice=holder_slice,holder_selector_queries=holder_queries,
            holder_slices=list(uuids or []) if holder_slice else [],
            holder_slice_telemetry_complete=bool(holder_slice and complete))
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
    p.add_argument('--expect-gpus',type=int,choices=range(9))
    p.add_argument('--holder-slice',action='store_true',help='one-GPU holder slice scoped by CUDA_VISIBLE_DEVICES')
    p.add_argument('--startup-grace',type=float,help='recorded startup grace in seconds (default: --window)')
    p.add_argument('--window',type=float,default=300);p.add_argument('--warn',type=float,default=75)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args(argv)
    if a.startup_grace is None:a.startup_grace=a.window
    if a.command[:1]==['--']:a.command=a.command[1:]
    if not a.command or not all(math.isfinite(v) and v>0 for v in (a.interval,a.window)) or not math.isfinite(a.startup_grace) or a.startup_grace<0 or not math.isfinite(a.warn) or not 0<=a.warn<=100:
        p.error('command, positive interval/window, nonnegative startup grace and warn in 0..100 required')
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
