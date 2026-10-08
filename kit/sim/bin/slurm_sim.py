#!/usr/bin/env python3
"""Process-only Slurm stand-in, NOT cgroups, device isolation, multi-node scheduling,
Slurm RPC latency or proof of a real cluster's delay bound. A detached timer owns
and repeatedly walks the launched process tree, retaining discovered membership
across reparenting/setsid/exec. Sampling can miss an instantaneous orphan fork.
SIM_SLURM_SECONDS_PER_MINUTE speeds tests ONLY; production arithmetic stays 60.
Config: $SIM_SLURM_STATE/config.json (KillWait, ProctrackType, TaskPlugin,
OverTimeLimit, PartitionOverTimeLimit, StartDelay, StartOffset, KillDetached).
"""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid

# Safe to import during the real trainer's CPU module smoke. Executables
# require SIM_SLURM_STATE; module imports never create synthetic state.
ROOT = Path(os.environ['SIM_SLURM_STATE']) if os.environ.get('SIM_SLURM_STATE') else None
if ROOT is not None:ROOT.mkdir(parents=True, exist_ok=True)
TERMINAL = ('COMPLETED', 'CANCELLED', 'TIMEOUT', 'FAILED', 'PREEMPTED')
EXPECTED = {'ProctrackType': 'proctrack/cgroup', 'TaskPlugin': 'task/cgroup,task/affinity',
            'JobAcctGatherType': 'jobacct_gather/cgroup', 'SignalChildrenProcesses': 'no',
            'KillWait': 40, 'OverTimeLimit': 0, 'PreemptType': 'preempt/none', 'PreemptMode': 'OFF'}
PS = shutil.which('ps', path=os.defpath)


def config():
    try:
        return json.loads((ROOT / 'config.json').read_text())
    except FileNotFoundError:
        return {}


@contextlib.contextmanager
def locked():
    with (ROOT / 'lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def read(sid):
    try:
        return json.loads((ROOT / (sid + '.json')).read_text())
    except FileNotFoundError:
        return None


def write(d):
    p = ROOT / (d['StepId'] + '.json')
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d))
    os.replace(tmp, p)


def patch(sid, **kw):
    with locked():
        d = read(sid)
        d.update(kw)
        write(d)
        return d


def tree():
    # Birth identity prevents a reused pid from becoming a step member.
    found = {}
    if Path('/proc/self/stat').exists():
        for p in Path('/proc').glob('[0-9]*/stat'):
            try:
                v = p.read_text().rsplit(')', 1)[1].split()
                if v[0] != 'Z':
                    found[int(p.parent.name)] = (int(v[1]), v[19])
            except (OSError, ValueError, IndexError):
                pass
    else:
        # Scheduler discovery is independent of the client's PATH scan stand-ins.
        # This remains a process-tree sampler, not kernel containment.
        text = subprocess.run([PS, '-axo', 'pid=,ppid=,stat=,lstart='], capture_output=True, text=True, check=True, timeout=1).stdout
        for line in text.splitlines():
            v = line.split()
            if len(v) >= 4 and not v[2].startswith('Z'):
                found[int(v[0])] = (int(v[1]), ' '.join(v[3:]))
    return found


def assign_resources(d, cfg):
    """Slurm owns the job bitmap: lowest free GPUs, independent of client CVD."""
    bitmap=str(cfg.get('JobGPUIds','0,1,2,3,4,5,6,7')).split(',')
    busy=set(); used_cpu=0; used_mem=0
    for path in ROOT.glob('*.json'):
        if path.name=='config.json': continue
        other=json.loads(path.read_text())
        if (other.get('StepId')!=d['StepId'] and other.get('State') in ('RUNNING','COMPLETING') and
                other.get('StepId','').split('.')[0]==d['StepId'].split('.')[0]):
            busy.update(other.get('AssignedGPUIds',[]))
            used_cpu+=other.get('CPUs',1); used_mem+=other.get('MemoryMB',0)
    free=[i for i in bitmap if i not in busy]
    if (len(free)<d['GPUCount'] or used_cpu+d.get('CPUs',1)>int(cfg.get('NumCPUs',64)) or
            used_mem+d.get('MemoryMB',0)>int(cfg.get('MemoryMB',512000))): return None
    # Explicit config is fault injection, never a client selector preference.
    return str(cfg['StepDevices']).split(',')[:d['GPUCount']] if cfg.get('StepDevices') is not None else free[:d['GPUCount']]


def timer(sid, command):
    cfg = config()
    # The daemon owns the launcher; neither client nor wrapper owns enforcement.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    patch(sid, TimerPid=os.getpid())
    until = time.monotonic() + float(cfg.get('StartDelay', 0))
    while time.monotonic() < until:
        if read(sid).get('Cancel'):
            patch(sid, State='CANCELLED', EndTime=time.time(), ExitCode=125)
            return
        time.sleep(.02)
    while True:
        with locked():
            d=read(sid)
            if d.get('Cancel') or d['State'] in TERMINAL:
                d.update(State='CANCELLED',EndTime=time.time(),ExitCode=125); write(d); return
            selected=assign_resources(d,cfg)
            if selected is not None:
                d['AssignedGPUIds']=selected
                log=(ROOT/(sid+'.log')).open('ab',buffering=0)
                env={**os.environ,'SLURM_STEP_ID':sid.split('.')[1],'SLURM_JOB_ID':sid.split('.')[0],
                    'SLURM_STEP_NODELIST':cfg.get('NodeList','sim-node'),'SLURM_JOB_GPUS':str(cfg.get('JobGPUIds','0,1,2,3,4,5,6,7')),
                    'SLURM_STEP_GPUS':','.join(selected),'SIM_VISIBLE_GPU_IDS':','.join(selected),
                    'CUDA_VISIBLE_DEVICES':','.join(map(str,range(d['GPUCount'])))}
                proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                start=time.time()+float(cfg.get('StartOffset',0))
                d.update(State='RUNNING',StartTime=start,LauncherPid=proc.pid,Command=command); write(d); break
        time.sleep(.02)
    period = 60.0 if d['GPUCount'] == 0 else float(os.environ.get('SIM_SLURM_SECONDS_PER_MINUTE', '60'))
    limit = time.monotonic() + d['TimeLimit'] * period
    members = {}
    started_clock = time.monotonic()
    why, kill_at, cancelled_at = None, None, None
    # slurmctld checks step limits in its 30-second periodic pass. The
    # phase/jitter is independent of the requested minute, even on fast clocks.
    check_period = float(cfg.get('PeriodicTimeout', 30))*period/60
    next_check = started_clock + float(cfg.get('TimeoutJitter', 27))*period/60
    while True:
        snapshot = tree()
        if proc.pid in snapshot:
            members.setdefault(proc.pid, snapshot[proc.pid][1])
        changed = True
        while changed:
            changed = False
            for pid, (ppid, born) in snapshot.items():
                if ppid in members and pid not in members:
                    members[pid] = born
                    changed = True
        living = {p: b for p, b in members.items() if p in snapshot and snapshot[p][1] == b}
        d = read(sid)
        due = time.monotonic() >= next_check
        if due:
            next_check += check_period
        if why is None:
            if cfg.get('PreemptAfter') is not None and time.monotonic()-started_clock >= float(cfg['PreemptAfter']):
                why = 'PREEMPTED'
                patch(sid, PreemptedAt=time.time(), GraceTimeSeconds=int(cfg.get('GraceTime', 600)))
            elif due and time.monotonic() >= limit:
                why = 'TIMEOUT'
            elif d.get('Cancel'):
                why, cancelled_at = 'CANCELLED', d.get('CancelTime')
            elif proc.poll() is not None:
                why = 'COMPLETED' if proc.returncode == 0 else 'FAILED'
            if why:
                grace = int(cfg.get('GraceTime', 600)) if why == 'PREEMPTED' else 0
                kill_at = time.monotonic() + grace + max(0, int(cfg.get('KillWait', 40)))
                patch(sid, KillWaitSeconds=int(cfg.get('KillWait', 40)), GraceTimeSeconds=grace,
                      TerminationStartedAt=time.time(), KillBy=time.time()+grace+int(cfg.get('KillWait', 40)))
        if why:
            # CANCEL's grace does not suspend the step's independent time limit.
            # Preserve the preemption cause while shortening enforcement to the
            # first periodic timeout pass plus KillWait.
            if due and time.monotonic() >= limit and why in ('TIMEOUT', 'PREEMPTED') and not d.get('StepTimeoutAt'):
                kill_at = min(kill_at, time.monotonic() + max(0, int(cfg.get('KillWait', 40))))
                patch(sid, StepTimeoutAt=time.time(), KillBy=time.time()+max(0, kill_at-time.monotonic()))
            sig = signal.SIGKILL if time.monotonic() >= kill_at else signal.SIGTERM
            targets = living if cfg.get('KillDetached', True) else {p: b for p, b in living.items() if _group(p) == proc.pid}
            for pid in targets:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            # Bad-containment mode deliberately lies about emptiness: selftest must find escapees.
            if not targets:
                proc.poll()
                patch(sid, State=why, EndTime=time.time(), CancelTime=cancelled_at,
                      ExitCode=(143 if why == 'PREEMPTED' else proc.returncode if proc.returncode is not None else 137),
                      Members={str(p): b for p, b in members.items()}, TerminationReason=why)
                return
        # Keep an external, inspectable membership archive for teardown, not for the kit's proof.
        patch(sid, Members={str(p): b for p, b in members.items()})
        time.sleep(.02)


def _group(pid):
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None


def srun(args):
    opts = {}
    while args and args[0].startswith('--'):
        key, sep, value = args.pop(0).partition('=')
        opts[key] = value if sep else True
    minutes = int(opts.get('--time', 0))
    if minutes < 1 or not args:
        return 2
    sid = os.environ['SLURM_JOB_ID'] + '.' + uuid.uuid4().hex[:8]
    d = {'StepId': sid, 'Name': opts.get('--job-name', ''), 'State': 'PENDING', 'TimeLimit': minutes,
         'NodeList': config().get('NodeList', 'sim-node'), 'GPUCount': int(str(opts.get('--gres', 'gpu:1')).split(':')[-1]),
         'CPUs':int(opts.get('--cpus-per-task',1)),
         'MemoryMB':int(str(opts.get('--mem',config().get('MemoryMB',512000))).rstrip('M')),
         'StartTime': None, 'EndTime': None, 'ClientPid': os.getpid(), 'SubmitTime': time.time()}
    with locked():
        write(d)
    first = subprocess.Popen([sys.executable, __file__, 'timer', sid] + args, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    patch(sid, TimerPid=first.pid)
    while True:
        d = read(sid)
        if d['State'] in TERMINAL:
            first.wait(timeout=5)
            log = ROOT / (sid + '.log')
            if log.exists():
                sys.stdout.buffer.write(log.read_bytes())
            return int(d.get('ExitCode', 1)) % 256
        time.sleep(.03)


def main(name, args):
    cfg = config()
    if name == 'timer':
        try:
            timer(args[0], args[1:])
        except BaseException as e:
            d = read(args[0]) or {}
            for pid in list(map(int, d.get('Members', {}))) + [d.get('LauncherPid')]:
                if isinstance(pid, int):
                    try: os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError: pass
            patch(args[0], State='FAILED', EndTime=time.time(), ExitCode=2, Error=str(e))
        return 0
    if name == 'srun':
        return srun(args)
    if name == 'scancel':
        if len(args) != 1 or '.' not in args[0]:
            return 2  # No allocation cancellation, even in the stand-in.
        with locked():
            d = read(args[0])
            if d is None:
                return 1
            if d['State'] not in TERMINAL:
                d.update(Cancel=True, CancelTime=time.time())
                if d['State'] == 'PENDING':
                    d.update(State='CANCELLED', EndTime=time.time(), ExitCode=125)
                write(d)
        return 0
    steps = [json.loads(p.read_text()) for p in ROOT.glob('*.json') if p.name != 'config.json']
    if name == 'squeue':
        for d in steps:
            if d['State'] not in TERMINAL and ('--steps=' + d['StepId'] in args or not any(a.startswith('--steps=') for a in args)):
                print(d['StepId'])
        return 0
    if name == 'sacct':
        ref = args[args.index('-j')+1]
        jobcfg={**cfg,**cfg.get('Jobs',{}).get(ref,{})}
        start_path=ROOT/('allocation-'+ref+'.start')
        if '.' not in ref and start_path.exists():
            state=jobcfg.get('JobState','RUNNING'); end=jobcfg.get('JobEndTime')
            lag=float(jobcfg.get('AccountingLag',10))*float(os.environ.get('SIM_SLURM_SECONDS_PER_MINUTE','60'))/60
            if state not in TERMINAL or end and time.time()>=float(end)+lag:
                stamp=lambda t:time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(float(t))) if t else 'Unknown'
                tres='cpu=%s,mem=%sM,gres/gpu=8'%(jobcfg.get('NumCPUs',64),jobcfg.get('MemoryMB',512000))
                print('|'.join([ref,state,stamp(start_path.read_text()),stamp(end),tres]))
        for d in steps:
            if d['StepId'] == ref and (not d.get('EndTime') or time.time() >= d['EndTime'] + float(cfg.get('AccountingLag', 10))*float(os.environ.get('SIM_SLURM_SECONDS_PER_MINUTE', '60'))/60):
                def stamp(t):
                    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t)) if t else 'Unknown'
                print('|'.join((d['StepId'], 'TRACKING_ERROR' if d.get('Error') else d['State'], stamp(d.get('StartTime')), stamp(d.get('EndTime')))))
        return 0
    if name == 'sacctmgr':
        if cfg.get('QOSVisible', True):
            print('%s|%s|%s|' % (cfg.get('QOS', 'normal'), cfg.get('QOSPreemptMode', 'OFF'), cfg.get('GraceTime', 600)))
        return 0
    if args == ['--version']:
        if cfg.get('Version', 'slurm-wlm 23.11.10 (stand-in)') is not None:
            print(cfg.get('Version', 'slurm-wlm 23.11.10 (stand-in)'))
        return 0
    if args[:2] == ['show', 'config']:
        entries = []
        for key, default in (('SignalChildrenProcesses', 'no'), ('ConstrainDevices', 'yes')):
            val = cfg.get(key, default)
            if val is not None:
                entries.append('%s=%s' % (key, 'yes' if val is True else 'no' if val is False else val))
        (ROOT / 'cgroup.conf').write_text('\n'.join(entries)+'\n')
        print('SLURM_CONF = '+str(ROOT / 'slurm.conf'))
        for key, default in EXPECTED.items():
            if key == 'SignalChildrenProcesses': continue
            if cfg.get(key, default) is not None:
                val = cfg.get(key, default)
                if key == 'KillWait' and str(val).isdigit(): val = str(val)+' sec'
                if key == 'OverTimeLimit' and str(val).isdigit(): val = str(val)+' min'
                print('%s = %s' % (key, val))
        print('Cgroup Support Configuration:')
        for entry in entries: print(entry.replace('=', ' = ', 1))
        return 0
    if args[:2] == ['show', 'job']:
        cfg={**cfg,**cfg.get('Jobs',{}).get(args[2],{})}
        if args[2] in cfg.get('ForgetJobs',[]): return 1
        job_state = cfg.get('JobState', 'PREEMPTED' if any(d.get('TerminationReason') == 'PREEMPTED' for d in steps) else 'RUNNING')
        start_path = ROOT/('allocation-'+args[2]+'.start')
        if not start_path.exists(): start_path.write_text(str(cfg.get('JobStartTime', time.time())))
        def stamp(t):
            return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(float(t))) if t else 'Unknown'
        print('JobId=%s JobState=%s Partition=sim NodeList=%s NumNodes=1 Shared=0 QOS=%s TimeLimit=%s StartTime=%s EndTime=%s NumCPUs=%s MinMemoryNode=%s TresPerNode=gres/gpu:8 AllocTRES=cpu=%s,mem=%sM,gres/gpu=8' % (args[2],job_state,cfg.get('NodeList','sim-node'),cfg.get('QOS','normal'),cfg.get('JobTimeLimit','2-00:00:00'),stamp(start_path.read_text()),stamp(cfg.get('JobEndTime')),cfg.get('NumCPUs',64),cfg.get('MemoryMB',512000),cfg.get('NumCPUs',64),cfg.get('MemoryMB',512000))); return 0
    if args[:2] == ['show', 'partition']:
        overtime = cfg.get('PartitionOverTimeLimit', 'NONE')
        print('PartitionName=sim' + ('' if overtime is None else ' OverTimeLimit=%s' % overtime) +
              ('' if cfg.get('PartitionPreemptMode', 'OFF') is None else ' PreemptMode=%s' % cfg.get('PartitionPreemptMode', 'OFF'))); return 0
    if args[:2] == ['show', 'step']:
        if cfg.get('ForgetTerminalSteps', True):
            steps = [d for d in steps if d['State'] not in TERMINAL]
        for d in steps:
            if len(args) <= 2 or args[2] in (d['StepId'], d['StepId'].split('.')[0]):
                def stamp(t):
                    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t)) if t else 'Unknown'
                print('StepId=%s Name=%s State=%s StartTime=%s EndTime=%s TimeLimit=%s NodeList=%s TRES=gres/gpu=%s' % (
                    d['StepId'], d['Name'], d['State'], stamp(d.get('StartTime')), stamp(d.get('EndTime')),
                    '%02d:%02d:00' % divmod(d['TimeLimit'], 60), d['NodeList'], d['GPUCount']))
                if d.get('GraceTimeSeconds') is not None: print('GraceTimeSeconds=%s KillWaitSeconds=%s' % (d['GraceTimeSeconds'], d['KillWaitSeconds']))
                if d.get('Error'): print('TrackingError=' + d['Error'].replace(' ', '_'))
        return 0
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2:]))
