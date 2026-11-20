"""CR-only process supervision, including descendants in independent sessions.

Linux child-subreaper adoption closes the reparenting race: an orphaned worker
returns to this wrapper, even when the utilisation watcher dies first. Other
platforms can perform best-effort cleanup but cannot certify the same guarantee.
"""
import ctypes
import os
import signal
import subprocess
import sys
import threading
import time


def processes():
    """Only process identity metadata; never read command lines or environments."""
    if sys.platform.startswith('linux'):
        from pathlib import Path
        result = {}; pending = [os.getpid()]; seen = set()
        while pending:
            parent = pending.pop()
            if parent in seen: continue
            seen.add(parent)
            try: tasks = list(Path('/proc', str(parent), 'task').iterdir())
            except FileNotFoundError:
                if parent == os.getpid(): raise RuntimeError('CR proc inventory unavailable')
                continue
            if parent == os.getpid() and not tasks:
                raise RuntimeError('CR proc inventory unavailable')
            inventoried = False
            for task in tasks:
                try: children = (task/'children').read_text().split()
                except FileNotFoundError: continue
                inventoried = True
                for child in children:
                    try:
                        # comm can contain spaces/parentheses; final ')' ends it.
                        fields = Path('/proc', child, 'stat').read_text().rsplit(')', 1)[1].split()
                        pid = int(child)
                        result[pid] = (int(fields[1]), fields[0]);pending.append(pid)
                    except (FileNotFoundError, ProcessLookupError):
                        continue
            if parent == os.getpid() and not inventoried:
                raise RuntimeError('CR proc children inventory unavailable')
        return result
    with subprocess.Popen(['ps', '-axo', 'pid=,ppid=,stat='],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as helper:
        output, _ = helper.communicate(timeout=5)
        if helper.returncode: raise RuntimeError('process inventory unavailable')
        return {int(p): (int(parent), status) for p, parent, status in
                (line.split()[:3] for line in output.splitlines() if line.strip())
                if int(p) != helper.pid}


def descendants(table, root):
    found = {root}
    while True:
        more = {pid for pid, (parent, _) in table.items() if parent in found}
        if more <= found: return found-{root}
        found |= more


def gpu_run(command, evidence, check=False, **kwargs):
    """Return/raise only after cleanup; evidence is also filled on SIGTERM."""
    evidence['teardown_confirmed'] = False
    libc = None; previous_subreaper = ctypes.c_int()
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(37, ctypes.byref(previous_subreaper), 0, 0, 0) != 0 or libc.prctl(36, 1, 0, 0, 0) != 0:
            raise RuntimeError('CR child-subreaper unavailable')
    process = None; tracked = set(); stopped = threading.Event(); scan_failed = []
    def scan():
        table = processes()
        owned = descendants(table, os.getpid())
        if libc is None:
            owned |= tracked.intersection(table)
        tracked.update(owned)
        # ps itself is a completed child by now; dead/zombie processes cannot
        # own GPU work and need not keep the reservation alive.
        return {p for p in owned if p in table and not table[p][1].startswith('Z')}
    def monitor():
        while not stopped.wait(.05):
            try: scan()
            except Exception: scan_failed.append(True)
    def term(signum, frame): raise SystemExit(128+signum)
    previous = signal.signal(signal.SIGTERM, term)
    thread = None
    try:
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        tracked.add(process.pid)
        if libc is None:
            thread = threading.Thread(target=monitor, daemon=True); thread.start()
        code = process.wait()
        if check and code: raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)
    finally:
        # Further TERM must not interrupt teardown and manufacture completion.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stopped.set()
        if thread is not None: thread.join(timeout=6)
        try:
            if process is not None:
                # Repeated scans include adopted or newly forked descendants.
                for sig, seconds in ((signal.SIGTERM, 1.), (signal.SIGKILL, 5.)):
                    deadline = time.monotonic()+seconds
                    while True:
                        try: alive = scan()
                        except Exception:
                            scan_failed.append(True)
                            alive = {process.pid}
                        try: os.killpg(process.pid, sig)
                        except (ProcessLookupError, PermissionError): pass
                        for pid in sorted(alive):
                            try: os.kill(pid, sig)
                            except ProcessLookupError: pass
                        if not alive or time.monotonic() >= deadline: break
                        time.sleep(.05)
                    if not scan(): break
                alive = scan()
                if not alive:
                    # Re-snapshot after reparenting/fork races have settled.
                    time.sleep(.05)
                    alive = scan()
                if not alive:
                    process.wait(timeout=1)
                    # Reap adopted zombies without blocking on live children.
                    for pid in sorted(tracked-{process.pid}):
                        try: os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError: pass
                evidence['teardown_confirmed'] = not alive and libc is not None and not scan_failed
            else:
                evidence['teardown_confirmed'] = True  # no process was launched
        except Exception as error:
            evidence['teardown_confirmed'] = False
            evidence['teardown_error'] = type(error).__name__
        finally:
            if libc is not None:
                libc.prctl(36, previous_subreaper.value, 0, 0, 0)
            signal.signal(signal.SIGTERM, previous)


def cpu_preflight():
    """Exercise Linux adoption and detached TERM-resistant cleanup without CUDA."""
    if not sys.platform.startswith('linux'):
        raise RuntimeError('CR Linux CPU teardown preflight unavailable')
    processes()  # Refuse missing inventory before launching even a CPU child.
    worker = ("import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
              "print('CR_READY',flush=True);time.sleep(60)")
    launcher = ("import signal,subprocess,sys;signal.alarm(10);"
                f"p=subprocess.Popen([sys.executable,'-I','-c',{worker!r}],"
                "start_new_session=True,stdout=subprocess.PIPE);"
                "assert p.stdout.readline()==b'CR_READY\\n'")
    evidence = {}
    gpu_run([sys.executable, '-I', '-c', launcher], evidence, check=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if evidence.get('teardown_confirmed') is not True:
        raise RuntimeError('CR Linux CPU teardown preflight unconfirmed')
    return evidence
