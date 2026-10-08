#!/usr/bin/env python3
"""The run-time limit that survives the wrapper: an independent watchdog for one package-4 GPU attempt, and the JOB
BOUNDARY that both it and kit/p4_run.py end (round-4 ruling G1, round-5 ruling H1, round-6 rulings J1 and J2).

    python p4_watchdog.py --record WORK/k8b4/budget/attempts/<row>-a<N>.json --out WORK/k8b4/budget/watchdog/<row>-a<N>.json \\
                          --pgid <the job's process group>

TIMES (H1). The attempt record holds `limit_until` (the moment the admitted run time ends: launch + the ledger's limit)
and `deadline` (= limit_until + the reserved shutdown allowance, 60 seconds). Shutdown BEGINS at limit_until: at that
moment the wrapper and this watchdog both send SIGTERM to the job; SIGKILL follows after the kill grace and in any
case no later than `kill_by` (= deadline - 15 seconds; a quarter of the allowance when a test lowers it); by `deadline`
termination is verified (the job re-scanned until it is empty) or the attempt is "workers not verified terminated" and
its block is stopped with its accounting unavailable. Nothing the campaign owns may be alive after the deadline.

THE SIGNAL SCHEDULE (J1: termination never waits on discovery). `JobEnd` ends a job on a schedule fixed in ABSOLUTE
MONOTONIC time when it is created (a wall-clock time is converted once, so a clock step cannot move it):
  - at the TERM time (limit_until for the watchdog; at once for the wrapper's own end of a run and for recovery) its
    FIRST action -- before any scan, and with no filesystem operation before it -- is killpg(group, SIGTERM) on the
    process group it was GIVEN (held in memory, never re-read; a signal-0 probe while it waits notes the group's
    disappearance, after which that id is never signalled again, so a reused id is safe);
  - at the KILL time (min(TERM time + kill grace, kill_by)) its action is killpg(group, SIGKILL), again before any scan,
    to the group and to every member discovery has found so far; then SIGKILL to the group every half second until the
    job is verified empty or the deadline passes.
These run in a thread of their own that does nothing else: no nvidia-smi, no /proc scan and no slow disk can delay
them. DISCOVERY -- the marker scan, the process-group scan, the GPU-holder query -- runs in a second thread after the
TERM; each of its calls is bounded by min(its own timeout, the time left to the next hard action (the KILL, then the
deadline)) and a call that overruns is killed (its whole session). Every additional member discovery finds is signalled
as found: SIGTERM before the KILL time, SIGKILL at or after it. Records (the watchdog's "acting" record, the block's
stop, `job_alive_at_limit`) are written by the caller AFTER the first TERM, while the schedule runs on.

THE JOB. Every process of the job carries the inherited environment marker KIT_P4_JOB=<row>-a<attempt>-<random nonce>,
set by the wrapper before it execs the launcher. "The job" is the UNION of
  - the launcher's process group (`pgid` of the attempt record, given to the watchdog with --pgid);
  - every process of this user whose ENVIRONMENT carries that marker (`marker_pids`: /proc/<pid>/environ on Linux;
    `ps eww -o pid=,command= -U <uid>` elsewhere, which prints each process's environment after its command; a
    self-test (`environment_scan_works`) starts a child with a probe variable and must find it, or the scan has failed
    and no launch is made);
  - every compute process this user owns on the row's GPUs (`nvidia-smi --query-compute-apps=pid`, `-i` the row's
    CUDA_VISIBLE_DEVICES) that STARTED within the attempt (J2): at or after the start of the attempt's process group
    (`job_window_start`, measured with the same clock as every other process's start, `process_start_time`) and not
    after the attempt ended (its record's verified end, `ended_precise`, when it has one). A GPU holder that started
    outside that window is another attempt's: it is never signalled (listed as `gpu_outside_window`), and the job can
    be verified empty beside it.
A watchdog therefore never signals a process that neither belongs to its group nor carries its marker, except a GPU
holder that started during its own attempt. DECLARED RESIDUAL: a process that left the group, cleared the marker from
its environment AND holds no GPU when the job is scanned is not found; it is outside the meaning of the GPU-hour
ceiling until it takes a GPU (persistent containment is the later `cgroup` / `slurm-step` seam of kit/p4_run.py
`start_contained`). (On macOS, used only for the dry run and the tests, the kernel hides the environment of Apple's own
platform binaries such as /bin/sleep: such a process is found only through its process group or a GPU. Production
nodes are Linux, where /proc shows every environment of the user's own processes.)

THE WATCHDOG. kit/p4_run.py starts it once the attempt record durably holds limit_until, the deadline, kill_by, the
marker and the job's process group (the launcher's shim exists and waits; the command runs only after the watchdog
reports that it watches). It detaches completely (a new session, forked once more, the first process exits at once,
stdio /dev/null, SIGHUP and SIGINT ignored), so it depends on neither the wrapper nor the runner. Then:
  - its signal schedule starts at once (TERM at limit_until, KILL at min(limit_until + kill grace, kill_by));
  - until the TERM it re-reads the record every half second; when the record carries `watchdog_stand_down` (the
    wrapper -- or a later recovery -- ended the job itself and verified it gone) it cancels the schedule, writes
    "stood down" durably (its ACKNOWLEDGEMENT, J2) and exits at once: it never scans again;
  - once the TERM is sent it records, durably, `job_alive_at_limit` (the group took the TERM, or the first scan found a
    member) with state "acting", and the block's durable stop (WORK/k8b4/budget/stop-<block>.json, exclusive and
    atomic: a spending-limit termination), then waits for the schedule and discovery to finish, and writes a DURABLE
    termination record (fsync, atomic rename; state "acted", its acknowledgement too): `job_alive_at_limit`,
    `term_at`, `kill_sent_at`, `job_gone_verified_at` (the time it verified the whole job gone) and its exact
    `..._precise` form, `verified_idle_at`; null when the job could not be shown empty by the deadline (then the stop
    says the block's accounting is unavailable).
IT NEVER LIVES FOREVER. A watchdog whose attempt record no longer exists (its folder deleted: nothing can account for
the job any more) for ABANDON_POLLS consecutive polls ends the job at once, on the same schedule (TERM now, KILL after
the kill grace), and exits ("abandoned"); and whatever happens, a guard timer ends the watchdog process at
deadline + EXIT_MARGIN (5 minutes; one minute after its start when it started later than that), writing "expired" if
it can. A watchdog started after its deadline acts at once and
exits.

kit/p4_run.py admits no GPU row while any watchdog of the same WORK is alive and has not acknowledged (state "starting",
"watching" or "acting"), and recovery charges an attempt whose wrapper vanished through this record's verified time; it
never infers one. Standard library only; no other kit module is imported, so it keeps working whatever happens to the
rest of the tree (kit/p4_run.py imports the job-boundary functions from here).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-p4-watchdog.v3"
BUDGET_SCHEMA = "kit-p4-budget.v1"           # the stop record's schema (kit/p4_budget.py SCHEMA)
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
PRECISE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
POLL = 0.5
SCAN_POLL = 0.1
SCAN_TIMEOUT = 60.0                          # one discovery call's own timeout (further bounded by the schedule)
KILL_GRACE = 30.0
STAND_DOWN = "watchdog_stand_down"
MARKER_VAR = "KIT_P4_JOB"
BUDGET_INCOMPLETE = "budget-incomplete"
COMPLIANCE_UNAVAILABLE = "budget compliance unavailable"
#: a watchdog record in one of these states has acknowledged (J2): it stood down, or recorded its own termination result
ACKNOWLEDGED = ("stood down", "acted", "refused", "abandoned", "expired")
UNACKNOWLEDGED = ("starting", "watching", "acting")
ABANDON_POLLS = 3                            # consecutive polls without the attempt record: the watchdog is abandoned
EXIT_MARGIN = 300.0                          # no watchdog process outlives its deadline by more than this


# ------------------------------------------------------------------------------------------------ time and files
def now_text(at: datetime | None = None) -> str:
    return (at or datetime.now(timezone.utc)).strftime(TIME_FORMAT)


def precise_text(at: datetime | None = None) -> str:
    return (at or datetime.now(timezone.utc)).strftime(PRECISE_FORMAT)


def from_epoch(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, timezone.utc)


def mono_of(epoch: float) -> float:
    """The monotonic-clock moment of a wall-clock epoch, converted ONCE (J1: a schedule is kept in monotonic time)."""
    return time.monotonic() + (float(epoch) - time.time())


def parse_time(text):
    """A recorded time in either format (whole seconds or microseconds), or None."""
    for fmt in (PRECISE_FORMAT, TIME_FORMAT):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def read_json(path: Path):
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_durably(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-%d-%d" % (os.getpid(), time.monotonic_ns()))     # one per writer, threads included
    with tmp.open("w") as handle:
        json.dump(data, handle, indent=1, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def create_once(path: Path, data: dict) -> bool:
    """Write `path` once, durably and atomically: the full content is fsynced under a temporary name and then LINKED to
    `path` (exclusive: an existing file is never replaced). True when this call created it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.tmp-%d-%d" % (path.name, os.getpid(), time.monotonic_ns()))
    with tmp.open("w") as handle:
        json.dump(data, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
        created = True
    except FileExistsError:
        created = False
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(path.parent)
    return created


def stop_path_of(record_path: Path, block: str) -> Path:
    """WORK/k8b4/budget/stop-<block>.json for an attempt record WORK/k8b4/budget/attempts/<row>-a<N>.json."""
    return Path(record_path).resolve().parent.parent / ("stop-%s.json" % block)


# ------------------------------------------------------------------------------------------------ bounded calls
class ScanTimeout(OSError):
    """A discovery call ran out of time: its own timeout, or the time left before the next hard action (J1)."""


def _run(command: list, until: float | None = None, own: float = SCAN_TIMEOUT, env: dict | None = None):
    """Run `command` bounded by min(`own` seconds, the monotonic moment `until`), in a session of its own; on overrun
    the whole session is killed and ScanTimeout raised (a pipe held open by a grandchild cannot hold the caller)."""
    timeout = own if until is None else min(own, until - time.monotonic())
    if timeout <= 0:
        raise ScanTimeout("no time left for %s before the next hard action" % command[0])
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True, env=env)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.communicate(timeout=1)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        raise ScanTimeout("%s did not answer within %.1f s" % (command[0], timeout)) from None
    return subprocess.CompletedProcess(command, proc.returncode, out, err)


# ------------------------------------------------------------------------------------------------ processes
def _alive_pid(pid: int, until: float | None = None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not _zombie(pid, until)


def _zombie(pid: int, until: float | None = None) -> bool:
    try:
        state = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()[0]
        return state == "Z"
    except (OSError, IndexError):
        pass
    try:
        done = _run(["ps", "-o", "stat=", "-p", str(pid)], until, 30)
    except (OSError, subprocess.SubprocessError):
        return False                                     # unknown: counted alive (it is never signalled for being dead)
    return done.stdout.strip().startswith("Z")


def group_alive(pgid, until: float | None = None) -> bool:
    """Whether the process group has a live member (a group of zombies only is not alive). Unknown counts as alive."""
    if not isinstance(pgid, int) or isinstance(pgid, bool) or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        done = _run(["ps", "-A", "-o", "pgid=,stat="], until, 30)
    except (OSError, subprocess.SubprocessError):
        return True
    if done.returncode != 0:
        return True
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == str(pgid) and not parts[1].startswith("Z"):
            return True
    return False


def owned_by_me(pid: int, until: float | None = None) -> bool:
    try:
        return os.stat("/proc/%d" % pid).st_uid == os.getuid()
    except OSError:
        pass
    try:
        done = _run(["ps", "-o", "uid=", "-p", str(pid)], until, 30)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.stdout.strip() == str(os.getuid())


def _boot_time() -> float | None:
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def process_start_time(pid: int, until: float | None = None):
    """When process `pid` started, as epoch seconds, or None when it cannot be read (J2). Linux: /proc/<pid>/stat's
    starttime after the boot time of /proc/stat (clock ticks); elsewhere `ps -o lstart=` (whole seconds, local time).
    Every start time compared with another comes from this one function, so both carry the same resolution."""
    try:
        fields = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
        boot = _boot_time()
        if boot is not None:
            return boot + int(fields[19]) / float(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        pass
    try:
        done = _run(["ps", "-o", "lstart=", "-p", str(pid)], until, 30, env={**os.environ, "LC_ALL": "C", "LANG": "C"})
        text = " ".join(done.stdout.split())
        if done.returncode != 0 or not text:
            return None
        return time.mktime(time.strptime(text, "%a %b %d %H:%M:%S %Y"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class EnvironmentScanError(Exception):
    """Neither /proc nor `ps eww` shows this user's process environments: the job boundary cannot be observed."""


def _proc_scan(token: str, until: float | None = None) -> set:
    """Linux: every live process of this user whose /proc/<pid>/environ holds `token` as one entry."""
    wanted, found, me = token.encode(), set(), os.getpid()
    entries = os.listdir("/proc")
    for number, name in enumerate(entries):
        if until is not None and number % 256 == 0 and time.monotonic() >= until:
            raise ScanTimeout("the /proc scan ran out of time before the next hard action")
        if not name.isdigit() or int(name) == me:
            continue
        pid = int(name)
        try:
            if os.stat("/proc/%d" % pid).st_uid != os.getuid():
                continue
            data = Path("/proc/%d/environ" % pid).read_bytes()
        except OSError:
            continue
        if wanted in data.split(b"\0") and not _zombie(pid, until):
            found.add(pid)
    return found


def _ps_scan(token: str, until: float | None = None) -> set:
    """Elsewhere (macOS): `ps eww` prints each process's environment after its command; `token` must be one word of it."""
    done = _run(["ps", "eww", "-o", "pid=,command=", "-U", str(os.getuid())], until, SCAN_TIMEOUT)
    if done.returncode != 0:
        raise OSError("ps eww exited %d: %s" % (done.returncode, done.stderr.strip()[:200]))
    found, me = set(), os.getpid()
    for line in done.stdout.splitlines():
        head, _sep, rest = line.strip().partition(" ")
        if head.isdigit() and int(head) != me and token in rest.split():
            found.add(int(head))
    return found


def _scanners() -> list:
    out = []
    if os.path.isfile("/proc/self/environ"):
        out.append(("/proc/<pid>/environ", _proc_scan))
    if shutil.which("ps"):
        out.append(("ps eww -o pid=,command= -U <uid>", _ps_scan))
    return out


_SCAN_METHOD: dict = {}


def environment_scan_works(until: float | None = None) -> tuple:
    """(True, method) when one of the two implementations is shown to read this user's process environments: a child
    started with a probe variable must be found by it; (False, why) when neither works (no launch is then made). Each
    scan is bounded by the monotonic moment `until`; only a success is remembered."""
    if _SCAN_METHOD.get("ok"):
        return True, _SCAN_METHOD["method"]
    probe = "probe-%d-%d" % (os.getpid(), time.monotonic_ns())
    token = "%s=%s" % (MARKER_VAR, probe)
    errors = []
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], env={**os.environ, MARKER_VAR: probe},
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.2)
        for method, scan in _scanners():
            try:
                if child.pid in scan(token, until):
                    _SCAN_METHOD.update({"ok": True, "method": method, "scan": scan})
                    return True, method
                errors.append("%s did not show a probe child's environment" % method)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                errors.append("%s failed: %s" % (method, error))
    finally:
        child.kill()
        child.wait()
    return False, "; ".join(errors) or "neither /proc nor ps is available"


def marker_pids(marker: str, until: float | None = None) -> set:
    """Every live process of this user whose environment carries KIT_P4_JOB=<marker>. EnvironmentScanError when
    neither implementation can read environments here; ScanTimeout when the call ran out of time."""
    if not marker:
        return set()
    ok, why = environment_scan_works(until)
    if not ok:
        raise EnvironmentScanError("the job boundary cannot be observed on this machine: %s" % why)
    return set(_SCAN_METHOD["scan"]("%s=%s" % (MARKER_VAR, marker), until))


def gpu_entries(devices, until: float | None = None):
    """The compute-process entries nvidia-smi lists on `devices` (None: every GPU); None without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"] + (["-i", devices] if devices else [])
    try:
        done = _run(command, until, SCAN_TIMEOUT)
    except ScanTimeout as error:
        return ["nvidia-smi did not answer (%s)" % error]
    except (OSError, subprocess.SubprocessError):
        return ["nvidia-smi did not answer"]
    if done.returncode != 0:
        return ["nvidia-smi exited %d" % done.returncode]
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def job_members(marker, pgid, devices, until: float | None = None, window=None, group_gone: bool = False) -> dict:
    """The job as it is now: {group, marker, gpu (this user's compute pids on the row's GPUs that started inside the
    attempt's `window` (start, end-or-None) when one is given), gpu_outside_window (this user's GPU holders that started
    outside it: another attempt's, never signalled), gpu_other (entries that are not this user's processes or not
    pids), errors, empty}. `empty` only when every part is verified empty (holders outside the window do not count).
    Every call is bounded by the monotonic moment `until` (J1)."""
    out = {"group": False if group_gone else group_alive(pgid, until), "marker": [], "gpu": [], "gpu_other": [],
           "gpu_outside_window": [], "errors": []}
    try:
        out["marker"] = sorted(marker_pids(marker, until))
    except (EnvironmentScanError, OSError, subprocess.SubprocessError) as error:
        out["errors"].append("marker scan: %s" % error)
    entries = gpu_entries(devices, until)
    if entries is None:
        if os.environ.get("KIT_ALLOW_NO_NVIDIA_SMI") != "1":
            out["errors"].append("no nvidia-smi on PATH: the GPUs cannot be shown idle")
    else:
        start, end = (window or (None, None))[:2] if window else (None, None)
        for entry in entries:
            if not (entry.isdigit() and int(entry) != os.getpid() and _alive_pid(int(entry), until) and owned_by_me(int(entry), until)):
                out["gpu_other"].append(entry)
                continue
            pid = int(entry)
            if window is not None and start is not None:
                began = process_start_time(pid, until)
                if began is None:
                    out["errors"].append("the start time of GPU compute process %d cannot be read" % pid)
                    continue
                if began < float(start) or (end is not None and began > float(end)):
                    out["gpu_outside_window"].append(pid)
                    continue
            out["gpu"].append(pid)
    out["alive"] = bool(out["group"] or out["marker"] or out["gpu"])
    out["empty"] = not (out["alive"] or out["gpu_other"] or out["errors"])
    return out


# ------------------------------------------------------------------------------------------------ J1: the schedule
class JobEnd:
    """End one job on a FIXED signal schedule (round-6 ruling J1; see the module's text). `term_at`, `kill_at` and
    `verify_by` are wall-clock epoch seconds, converted once to the monotonic clock; `term_at` None means now. The
    process group `pgid` is held in memory. `window_start` / `window_end_of()` bound the GPU-holder path (J2).

        job = JobEnd(...); job.start()          # the schedule runs in its own thread from here on
        ...records, written AFTER job.termed is set...
        result = job.wait(reap)                 # until verified or the deadline

    Nothing the caller does (a scan, a disk write) can delay the TERM or the KILL."""

    def __init__(self, marker, pgid, devices, kill_at: float, verify_by: float, term_at: float | None = None,
                 window_start=None, window_end_of=None, scheduler=None):
        self.scheduler = scheduler
        self.marker, self.devices = marker, devices
        self.pgid = pgid if isinstance(pgid, int) and not isinstance(pgid, bool) and pgid > 1 else None
        now = time.time()
        self.term_due = now if term_at is None else float(term_at)
        self.kill_due, self.verify_due = float(kill_at), max(float(verify_by), float(kill_at))
        self.term_mono, self.kill_mono, self.verify_mono = mono_of(self.term_due), mono_of(self.kill_due), mono_of(self.verify_due)
        self.window_start, self.window_end_of = window_start, window_end_of
        self.lock = threading.Lock()
        self.cancelled = False
        self.group_gone = self.pgid is None
        self.wake, self.termed, self.first_scan, self.verified, self.finished = (threading.Event() for _ in range(5))
        self.stop_discovery = threading.Event()
        self.found, self.signalled = set(), {}
        self.out = {"term": False, "kill": False, "term_at": None, "kill_sent_at": None, "kill_action_at": None,
                    "group_term_delivered": None, "group_kill_delivered": None, "verified": False, "verified_at": None,
                    "scans": 0, "first": None, "last": None, "scan_errors": 0}

    # -- the schedule thread: system calls only
    def start(self) -> "JobEnd":
        threading.Thread(target=self._schedule, name="p4-signal-schedule", daemon=True).start()
        return self

    def cancel_before_term(self) -> bool:
        """Cancel the schedule if its TERM has not been sent (a watchdog's stand-down); False once it has."""
        with self.lock:
            if self.termed.is_set():
                return False
            self.cancelled = True
        self.wake.set()
        return True

    def _probe_group(self) -> None:
        if self.group_gone:
            return
        try:
            os.killpg(self.pgid, 0)
        except ProcessLookupError:
            self.group_gone = True                         # never signal this id again: it may be reused
        except PermissionError:
            pass

    def _signal_group(self, sig):
        """True when the group took the signal; "EPERM" when the kernel answered EPERM -- on macOS that is also answered
        while a member is exiting, and the signal still reaches every member this user may signal (so it counts as
        sent, and is recorded as such); False when the group is gone (never signalled again)."""
        if self.group_gone:
            return False
        try:
            os.killpg(self.pgid, sig)
            return True
        except ProcessLookupError:
            self.group_gone = True
        except PermissionError:
            self.out.setdefault("group_signal_errors", []).append("%s EPERM at %s" % (signal.Signals(sig).name, precise_text()))
            return "EPERM"
        return False

    def _schedule(self) -> None:
        while True:                                        # until the TERM: wait, and note a vanished group
            left = self.term_mono - time.monotonic()
            if left <= 0:
                break
            self.wake.wait(min(left, POLL))
            if self.cancelled:
                return
            self._probe_group()
        with self.lock:
            if self.cancelled:
                return
            if self.scheduler:
                # Submit owning-step cancellation FIRST; don't let a stalled Slurm RPC
                # hold up the secondary group schedule. The armed --time is independent.
                try:
                    client = subprocess.Popen(['scancel', '%s.%s' % (self.scheduler['job_id'], self.scheduler['step_id'])],
                                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.out['scheduler_cancel_submitted_at'] = precise_text()
                    def reap_cancel():
                        try: client.wait(timeout=5)
                        except subprocess.TimeoutExpired: client.kill(); client.wait()
                    threading.Thread(target=reap_cancel, daemon=True).start()
                except OSError as e:
                    self.out['scheduler_cancel_error'] = str(e)
            delivered = self._signal_group(signal.SIGTERM)   # J1: the FIRST action; nothing precedes it
            self.out.update({"term_at": precise_text(), "group_term_delivered": delivered, "term": delivered})
            self.termed.set()
        threading.Thread(target=self._discover, name="p4-discovery", daemon=True).start()
        while not self.verified.is_set():                  # the KILL, whatever discovery is doing
            left = self.kill_mono - time.monotonic()
            if left <= 0:
                break
            self.verified.wait(left)
        if not self.verified.is_set():
            with self.lock:
                self.out["kill_action_at"] = precise_text()
                self._kill_group()
                if [pid for pid in sorted(self.found) if self._send(pid, signal.SIGKILL)]:
                    self.out["kill"] = True
                    self.out["kill_sent_at"] = self.out["kill_sent_at"] or self.out["kill_action_at"]
            while not self.verified.is_set() and time.monotonic() < self.verify_mono:
                self.verified.wait(min(POLL, max(0.0, self.verify_mono - time.monotonic())))
                if not self.verified.is_set() and not self.group_gone:
                    with self.lock:
                        self._kill_group()
        self.stop_discovery.set()
        self.finished.set()

    def _kill_group(self) -> None:
        """SIGKILL to the group (under the lock), recorded whichever thread sends it first: `group_kill_delivered` and
        `kill_sent_at` keep the FIRST delivery; a later attempt that finds the group gone does not erase it."""
        delivered = self._signal_group(signal.SIGKILL)
        if delivered:
            self.out["kill"] = True
            self.out["kill_sent_at"] = self.out["kill_sent_at"] or precise_text()
            if not self.out["group_kill_delivered"]:
                self.out["group_kill_delivered"] = delivered
        elif self.out["group_kill_delivered"] is None:
            self.out["group_kill_delivered"] = False

    def _send(self, pid: int, sig) -> bool:
        if pid in (os.getpid(), 1):
            return False
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            return False
        self.signalled[pid] = signal.Signals(sig).name
        return True

    # -- the discovery thread: bounded scans, every member found signalled as found
    def _window(self):
        if self.window_start is None:
            return None
        end = None
        if self.window_end_of is not None:
            try:
                end = self.window_end_of()
            except Exception:                              # noqa: BLE001 -- an unreadable end leaves the window open
                end = None
        return (float(self.window_start), end)

    def _discover(self) -> None:
        while not self.stop_discovery.is_set():
            now = time.monotonic()
            if now >= self.verify_mono:
                break
            until = self.kill_mono if now < self.kill_mono else self.verify_mono
            try:
                members = job_members(self.marker, self.pgid, self.devices, until=until, window=self._window(), group_gone=self.group_gone)
            except Exception as error:                     # noqa: BLE001 -- discovery never stops the schedule
                members = {"group": not self.group_gone, "marker": [], "gpu": [], "gpu_other": [], "gpu_outside_window": [],
                           "errors": ["discovery failed: %s" % error], "alive": True, "empty": False}
            if not members.get("group"):
                self._probe_group()
            with self.lock:
                self.out["scans"] += 1
                self.out["scan_errors"] += bool(members.get("errors"))
                if self.out["first"] is None:
                    self.out["first"] = members
                    self.first_scan.set()
                # `last` is the last scan that ANSWERED: a final scan cut short because no time was left before the
                # next hard action must not replace what the scans before it saw (it is kept beside it, and the job
                # is not verified empty either way).
                unanswered = bool(members.get("errors")) or any(str(e).startswith("nvidia-smi did not answer") for e in members.get("gpu_other") or [])
                if unanswered and self.out["last"] is not None:
                    self.out["last_unanswered"] = members
                else:
                    self.out["last"] = members
                after_kill = time.monotonic() >= self.kill_mono
                for pid in sorted(set(members["marker"]) | set(members["gpu"])):
                    self.found.add(pid)
                    if after_kill:
                        if self._send(pid, signal.SIGKILL):
                            self.out["kill"] = True
                            self.out["kill_sent_at"] = self.out["kill_sent_at"] or precise_text()
                    elif pid not in self.signalled and self._send(pid, signal.SIGTERM):
                        self.out["term"] = True
                if after_kill and members.get("group"):
                    self._kill_group()                       # due now: discovery may be the first to send it
            if self.scheduler:
                # Outside-step scheduler verification accompanies the fallback scans.
                import importlib.util
                spec = importlib.util.spec_from_file_location('watch_contain', Path(__file__).with_name('p4_contain.py'))
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                e = module.evidence(self.scheduler['job_id'], self.scheduler['step_id'], self.devices,
                                    time.time()+max(.01, self.verify_mono-time.monotonic()))
                self.out['scheduler_termination'] = e
                members['empty'] = e.get('verified', False)
            if members.get("empty"):
                self.out.update({"verified": True, "verified_at": time.time()})
                self.verified.set()
                break
            self.stop_discovery.wait(SCAN_POLL)

    # -- the caller
    def wait(self, reap=None) -> dict:
        while not self.finished.wait(0.05):
            if reap is not None:
                reap()
        if reap is not None:
            reap()
        return self.result()

    def result(self) -> dict:
        with self.lock:
            out = dict(self.out)
        first = out["first"]
        last = out["last"] or {"group": not self.group_gone, "marker": [], "gpu": [], "gpu_other": [], "gpu_outside_window": [],
                               "errors": ["no discovery scan completed before %s" % precise_text(from_epoch(self.verify_due))],
                               "alive": None, "empty": False}
        out.update({"alive_at_start": bool(out["group_term_delivered"] or (first or {}).get("alive")), "first": first or {},
                    "last": last, "term_due": precise_text(from_epoch(self.term_due)), "kill_at": precise_text(from_epoch(self.kill_due)),
                    "verify_by": precise_text(from_epoch(self.verify_due)), "signalled": {str(k): v for k, v in sorted(self.signalled.items())},
                    "window": list(self._window()) if self.window_start is not None else None})
        return out


def end_job(marker, pgid, devices, kill_at: float, verify_by: float, reap=None, window_start=None, window_end_of=None, scheduler=None) -> dict:
    """End the whole job NOW on the J1 schedule: SIGTERM to the group at once (the first action), SIGKILL at `kill_at`
    (wall-clock epoch seconds) to the group and every member found, discovery bounded alongside, re-scanned until the
    job is EMPTY (group gone, no marker-carrying process, no compute process of the attempt on the GPUs), at most until
    `verify_by`. {alive_at_start, term, kill, term_at, kill_sent_at, verified, verified_at, first, last, scans, ...}."""
    return JobEnd(marker, pgid, devices, kill_at, verify_by, window_start=window_start, window_end_of=window_end_of, scheduler=scheduler).start().wait(reap)


# ------------------------------------------------------------------------------------------------ the watchdog
def write_stop(record_path: Path, record: dict, reason: str, cause: str, unavailable: bool = False) -> bool:
    if record.get('containment')=='slurm-step':
        # Both supported record layouts live beneath WORK/k8b4.
        path=Path(record_path).resolve()
        root=next((parent.parent for parent in path.parents if parent.name=='k8b4'),None)
        if root is not None:
            # Keep the watchdog independent of imports from the rest of kit.
            stop={'schema':'kit-v4-containment-stop.v1','stop_class':'hard','ok':False,
                  'stop_id':os.urandom(16).hex(),'allocation_id':(record.get('slurm') or {}).get('job_id'),
                  'stopped_at':precise_text(),'failure_type':reason,'row_record':str(record_path)}
            stop['content_sha256']=hashlib.sha256(json.dumps(stop,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            create_once(root/'k8b4/containment/v4-stop.json',stop)
    block = record.get("block")
    if not block:
        return False
    entry = {"schema": BUDGET_SCHEMA, "block": block, "time": now_text(), "row": record.get("row"), "attempt": record.get("attempt"),
             "reason": "%s: %s" % (COMPLIANCE_UNAVAILABLE if unavailable else BUDGET_INCOMPLETE, reason), "cause": cause,
             "written_by": "kit/p4_watchdog.py"}
    if unavailable:
        entry["accounting"] = "unavailable"
    return create_once(stop_path_of(record_path, block), entry)


def _ended_of(record_path: Path):
    """The attempt's verified end (epoch seconds) from its record, if the wrapper or a recovery recorded one."""
    at = parse_time((read_json(record_path) or {}).get("ended_precise"))
    return at.timestamp() if at is not None else None


def watch(record_path: Path, out_path: Path, pgid_arg=None) -> int:
    record = read_json(record_path) or {}
    base = {"schema": SCHEMA, "row": record.get("row"), "attempt": record.get("attempt"), "pid": os.getpid(),
            "record": str(record_path), "started_at": now_text()}
    limit_until, deadline = parse_time(record.get("limit_until")), parse_time(record.get("deadline"))
    kill_by, marker = parse_time(record.get("kill_by")) or deadline, record.get("job_marker")
    pgid = pgid_arg if pgid_arg is not None else record.get("pgid")
    why = None
    if limit_until is None or deadline is None or not marker:
        why = "the attempt record holds no limit_until, deadline or job marker (round-5 ruling H1)"
    elif not isinstance(pgid, int) or isinstance(pgid, bool) or pgid <= 1:
        why = "no process group was given for the job (round-6 ruling J1: the group is held from the start)"
    elif pgid_arg is not None and record.get("pgid") not in (None, pgid_arg):
        why = "the process group given (%s) is not the attempt record's (%s)" % (pgid_arg, record.get("pgid"))
    if why:
        write_durably(out_path, {**base, "state": "refused", "why": why})
        return 2
    try:
        grace = max(0.5, min(KILL_GRACE, float(record.get("kill_grace_seconds") or KILL_GRACE)))
    except (TypeError, ValueError):
        grace = KILL_GRACE
    devices = record.get("cuda_visible_devices") or None
    kill_at = min(limit_until.timestamp() + grace, kill_by.timestamp())
    job = JobEnd(marker, pgid, devices, kill_at=kill_at, verify_by=deadline.timestamp(), term_at=limit_until.timestamp(),
                 window_start=record.get("job_window_start"), window_end_of=lambda: _ended_of(record_path), scheduler=record.get("slurm"))
    job.start()                                            # J1: from here the TERM and the KILL are on their schedule

    def expire():                                          # the guard: never alive past deadline + EXIT_MARGIN
        try:
            write_durably(out_path, {**base, "state": "expired", "expired_at": precise_text(),
                                     "why": "the watchdog was still running %d s after its deadline" % EXIT_MARGIN})
        except OSError:
            pass
        os._exit(3)
    guard = threading.Timer(max(60.0, deadline.timestamp() + EXIT_MARGIN - time.time()), expire)   # a late start still acts first
    guard.daemon = True
    guard.start()
    base.update({"limit_until": now_text(limit_until), "deadline": now_text(deadline), "kill_by": precise_text(kill_by),
                 "kill_grace_seconds": grace, "devices": devices or "all", "job_marker": marker, "pgid": pgid,
                 "term_due": precise_text(limit_until), "kill_due": precise_text(from_epoch(kill_at)),
                 "job_window_start": record.get("job_window_start")})
    write_durably(out_path, {**base, "state": "watching"})
    environment_scan_works(until=job.term_mono)            # the marker scan's self-test, bounded: before it is needed
    missing = 0
    while not job.termed.is_set():
        current = read_json(record_path)
        missing = missing + 1 if not Path(record_path).exists() else 0
        if missing >= ABANDON_POLLS and job.cancel_before_term():
            # the attempt record is gone (its folder deleted): nothing can account for this job any more, so it is ended
            # now on the same schedule, and the watchdog exits
            now = time.time()
            ended = JobEnd(marker, pgid, devices, kill_at=now + grace, verify_by=now + grace + 10.0,
                           window_start=record.get("job_window_start"), scheduler=record.get("slurm")).start().wait()
            try:
                write_durably(out_path, {**base, "state": "abandoned", "abandoned_at": precise_text(), "term_at": ended["term_at"],
                                         "kill_sent_at": ended["kill_sent_at"], "job_gone": ended["verified"],
                                         "why": "the attempt record %s no longer exists" % record_path})
            except OSError:
                pass
            return 4
        if isinstance(current, dict) and current.get(STAND_DOWN) and job.cancel_before_term():
            # J2: the acknowledgement; the schedule is cancelled and this watchdog never scans again
            write_durably(out_path, {**base, "state": "stood down", "stood_down_at": now_text(), "stood_down_at_precise": precise_text(),
                                     "stand_down": current.get(STAND_DOWN)})
            return 0
        job.termed.wait(POLL)
    # the TERM went out at limit_until; only now the evidence and the stop (J1)
    term_at = parse_time(job.out["term_at"]) or datetime.now(timezone.utc)
    alive = True if job.out["group_term_delivered"] else None
    if alive is None:
        job.first_scan.wait(max(0.0, job.verify_mono - time.monotonic()))
        first = job.out["first"]
        alive = bool(first.get("alive")) if isinstance(first, dict) else None
    current = read_json(record_path) or record
    fields = {**base, "acted_at": now_text(term_at), "acted_at_precise": precise_text(term_at), "pgid_recorded": isinstance(record.get("pgid"), int),
              "job_alive_at_limit": alive, "group_term_delivered": job.out["group_term_delivered"], "job_at_limit": job.out["first"],
              # the round-4 name, read by older recoveries
              "group_alive_at_deadline": alive}
    if alive:
        # the job ran to the end of its admitted time: the evidence and the block's stop, durable, while the schedule runs
        write_durably(out_path, {**fields, "state": "acting"})
        write_stop(record_path, current, "row %s attempt %s was still running when its admitted run time ended at %s; the watchdog "
                   "ended it" % (current.get("row"), current.get("attempt"), now_text(limit_until)),
                   "spending-limit termination (the watchdog ended the job at the end of its admitted run time)")
    ended = job.wait()
    if fields["job_at_limit"] is None:
        fields["job_at_limit"] = ended["first"] or None
    verified_at = from_epoch(ended["verified_at"]) if ended["verified"] else None
    fields.update({"scheduler_termination": ended.get("scheduler_termination"), "state": "acted", "term": ended["term"], "kill": ended["kill"], "term_at": ended["term_at"],
                   "kill_sent_at": ended["kill_sent_at"], "kill_action_at": ended["kill_action_at"],
                   "group_kill_delivered": ended["group_kill_delivered"], "scans": ended["scans"], "signalled": ended["signalled"],
                   "group_signal_errors": ended.get("group_signal_errors"),
                   "group_gone": not ended["last"].get("group"), "job_gone": ended["verified"], "job_last_scan": ended["last"],
                   "job_gone_verified_at": _ceil_text(verified_at) if verified_at else None,
                   "job_gone_verified_at_precise": precise_text(verified_at) if verified_at else None,
                   "group_gone_verified_at": _ceil_text(verified_at) if verified_at else None,
                   "gpus_idle": ended["verified"], "verified_idle_at": _ceil_text(verified_at) if verified_at else None,
                   "terminated_by_deadline": bool(verified_at is not None and verified_at <= deadline)})
    if not ended["verified"]:
        fields["idle_note"] = "the job could not be shown empty by the deadline: %s" % _describe(ended["last"])
        write_durably(out_path, fields)
        write_stop(record_path, current, "row %s attempt %s: the job could not be shown terminated by its deadline %s (%s)"
                   % (current.get("row"), current.get("attempt"), now_text(deadline), _describe(ended["last"])),
                   "workers not verified terminated", unavailable=True)
        return 1
    fields["idle_note"] = "the job (group, marker %s, GPUs %s) verified empty" % (marker, devices or "all")
    write_durably(out_path, fields)
    return 0


def _ceil_text(at: datetime) -> str:
    """Whole seconds, rounded UP: a charged time is never earlier than the verified one."""
    if at.microsecond:
        at = datetime.fromtimestamp(int(at.timestamp()) + 1, timezone.utc)
    return now_text(at)


def _describe(members: dict) -> str:
    parts = []
    if members.get("group"):
        parts.append("the process group is alive")
    if members.get("marker"):
        parts.append("marker-carrying processes %s" % ", ".join(map(str, members["marker"][:8])))
    if members.get("gpu"):
        parts.append("GPU compute processes %s" % ", ".join(map(str, members["gpu"][:8])))
    if members.get("gpu_other"):
        parts.append("GPU entries that are not this user's processes %s" % ", ".join(members["gpu_other"][:8]))
    parts += members.get("errors") or []
    return "; ".join(parts) or "nothing"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--record", required=True, help="the attempt record kit/p4_run.py writes (it holds limit_until and the deadline)")
    parser.add_argument("--out", required=True, help="the watchdog's own durable record")
    parser.add_argument("--pgid", type=int, help="the job's process group, held in memory from the start (J1); default: the record's")
    parser.add_argument("--foreground", action="store_true", help="do not fork (tests)")
    args = parser.parse_args(argv)
    record, out = Path(args.record), Path(args.out)
    for sig in (signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    if not args.foreground:
        child = os.fork()
        if child:
            # the first process: name the watcher durably, then exit at once (the wrapper reaps it); the watcher is
            # reparented to init and shares nothing with the wrapper but the files
            write_durably(out, {"schema": SCHEMA, "state": "starting", "pid": child, "record": str(record),
                                "started_at": now_text()})
            os._exit(0)
        try:
            os.setsid()                                        # its own session too: no terminal, no shared group
        except OSError:
            pass
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(devnull, fd)
        end = time.monotonic() + 10                            # the first process names us before we write
        while time.monotonic() < end and (read_json(out) or {}).get("pid") != os.getpid():
            time.sleep(0.05)
    return watch(record, out, args.pgid)


if __name__ == "__main__":
    sys.exit(main())
