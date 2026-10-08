#!/usr/bin/env python3
"""Package 4's one way to start GPU work: verify, recover, idle check, reconcile, gate, run and OWN the workers, record.

    python p4_run.py --work W --campaign C.yaml --reservation R.json --row ROW --attempt N [--train] [--baseline B.json]
                     [--idle-wait 120] -- COMMAND ...

Every GPU row of a package-4 campaign (training, qualification, qualification scoring, scoring, direct) runs its
command through this file, which implements docs/phase2/plan-v3-package4-amendment3-20261004.md B3, B5, B6 and B7 and
the round-3 rulings F1, F2 and F4 for one attempt of one row (the row's env, set by kit/runner.py, carries P4_BLOCK,
P4_KIND and the launcher's settings). In this order:

  0. THE RECORD WORK/k8b4/budget/attempts/<row>-a<N>.json is written durably (fsync, atomic rename) before anything else
     and at every step: `state` admitting -> launching -> running -> ended | unverified, or refused / busy / reconciled.
  1. the frozen launcher knobs (NGPU, TP, OFFLOAD and LORA or FEEDBACK, SOFT, TEMP) are read from the verified pilot
     baseline (--baseline, kit/p4_recipe.py's frozen baseline.json) FIRST, before anything reads the environment;
  2. THE FROZEN INPUTS (kit/p4_frozen.py; F2): the pilot's finalization manifest and every decision input in it, the
     selection, the reservation, the recipe check's frozen record and baseline are re-hashed; for a training row also
     the trainer checkout (its commit and an empty `git status --short`), the task data's sha256 and the incoming
     checkpoint's export file list against the frozen record. Any difference: "inputs changed after they were frozen:
     <which>", exit 2, nothing launched;
  3. RECOVERY (F1): every attempt record still `launching`/`running` whose wrapper process is gone is reconciled before
     anything is admitted: its process group is ended if alive (TERM, then KILL after the kill grace), the GPUs it used
     are polled until idle, and it is charged up to its recorded absolute deadline, or to now when that is earlier and
     the GPUs are verified idle now (to now as well if its workers were still alive past the deadline). It is classified
     `interrupted`; one past its deadline is a spending-limit termination and stops its block; one whose GPUs cannot be
     shown idle is `workers not verified terminated` and stops its block too;
  4. the GPUs the row will use show no compute process (`nvidia-smi --query-compute-apps=pid --format=csv,noheader`,
     `-i` the CUDA_VISIBLE_DEVICES list when it is set), polling up to --idle-wait seconds; else "GPU still busy", exit 3.
     No nvidia-smi at all fails closed unless KIT_ALLOW_NO_NVIDIA_SMI=1 (the dry run only);
  5. (training, --train) RECONCILE: if any earlier attempt of the run left a complete export (the launchers' own test),
     the run is NOT trained again: exit 0, "existing export of attempt k kept", class `reconciled` (B5). Its settings
     are verified from that attempt's run-summary.json with the FROZEN knobs (not inherited ones);
  6. the GATE (kit/p4_budget.py `decide`: the ledger line, and on a budget refusal the durable stop). Exit 1 or 2
     launches nothing. A training row's GPU count in the ledger is the verified baseline's n_gpus;
  7. THE LAUNCH (F1): the record first holds the limit (the ledger's, already less the 60-second grace) and the ABSOLUTE
     deadline (launch + limit + grace), then the command starts in its OWN session and process group, and the record
     gets its pid and group. When the command ends for ANY reason -- its exit, the limit, or a signal to this wrapper
     (SIGTERM, SIGINT, SIGHUP) -- the WHOLE group is ended (SIGTERM, then SIGKILL after the kill grace) and the GPUs
     it used are polled until they show no compute process, at most until the deadline (and at most --idle-wait
     seconds after the group was ended). The attempt is charged until that verified idle time. If the GPUs cannot be
     shown idle in time: state `unverified`, class "workers not verified terminated", the block's durable stop is
     written and nothing else of that block launches; the ledger keeps charging it until a later recovery verifies it;
  8. classify: the limit reached -> `budget` (exit 124, or 137 when SIGKILL was needed), the durable stop; a signal ->
     `interrupted` (exit 130); training: `valid` (exit 0, complete export, merged 1, settings verified), `settings`
     (exit 5), `invalid export` (exit 4, or exit 0 without a complete export), else "launcher failure, cause
     unclassified" (B5). Retry eligibility (kit/p4_budget.py `eligibility`, which also compares the attempt's actual
     command and resolved configuration with the frozen baseline: F4) is recorded with the export validation and the
     tail of the console log.

ROUND 4 (the manager's rulings G1 and G2), which governs where it differs from the text above:
  G1  THE DEADLINE SURVIVES THE WRAPPER. Before the launcher, once the record durably holds the absolute deadline, an
      independent watchdog (kit/p4_watchdog.py) is started: a new session, forked once more, stdio closed, depending
      on neither this wrapper nor the runner. The launcher is then started through a shim in its own process group,
      the group is recorded durably, and only then does the command run. The watchdog sleeps until the deadline and,
      unless told to stand down, ends the whole group (TERM, grace, KILL), verifies it gone and the GPUs idle, and
      writes a durable termination record (WORK/k8b4/budget/watchdog/<row>-a<N>.json). When the launcher ends while
      this wrapper lives, the wrapper ends the group, verifies the GPUs idle and only then sets `watchdog_stand_down`
      in the record (the watchdog exits without killing). No watchdog, no launch. RECOVERY never infers a termination
      time: an attempt whose wrapper vanished is charged through a TRUSTWORTHY timestamp -- the wrapper's verified
      idle time or the watchdog's record -- and, with neither, through the moment recovery itself verifies the group
      gone and the GPUs idle (now); if that cannot be verified the attempt is `unverified`, the block's accounting is
      "budget compliance unavailable" and its durable stop is written (nothing of it launches). A watchdog that ended
      a live group at the deadline, or a recovery at or past the deadline, is a spending-limit termination. Workers
      that escaped the group cannot be reached; they are seen as busy GPUs: block stop, accounting unavailable.
  G2  THE INCOMING CHECKPOINT. Before every training and qualification launch the incoming export is re-hashed (the
      sha256 of every file, streamed) and compared with the identity the recipe check recorded in the VERIFIED frozen
      document (k8b4/recipe-check/frozen.json, `comparison.incoming`); a missing identity, a changed file list, a
      changed content, or a MODEL_DIR that is not that folder refuses the launch ("incoming checkpoint changed since
      the recipe check"). KIT_P4_GRACE (seconds) can only LOWER the deadline's grace (the tests).

ROUND 5 (the manager's rulings H1, H2), which governs where it differs from the text above:
  H1  NO SPENDING PAST THE HARD DEADLINE; A JOB BOUNDARY THAT SURVIVES THE WRAPPER. The record holds `limit_until`
      (launch + the ledger's limit: the end of the admitted run time) and `deadline` (= limit_until + the reserved
      shutdown allowance, 60 s). Shutdown BEGINS at limit_until: this wrapper and the watchdog both send SIGTERM to the
      job; SIGKILL follows after the kill grace and no later than `kill_by` (= deadline - 15 s); by the deadline
      termination is verified or the attempt is "workers not verified terminated" and its block is stopped with its
      accounting unavailable. THE JOB is the launcher's process group UNION every process of this user whose environment
      carries the marker KIT_P4_JOB=<row>-a<attempt>-<nonce> (set here, inherited by everything the launcher starts)
      UNION every compute process this user owns on the row's GPUs (kit/p4_watchdog.py `end_job`, which re-scans that
      set until it is empty). No launch is made where process environments cannot be read (`environment_scan_works`).
  H2  THE BUDGET DISPOSITION IS DURABLE BEFORE THE TERMINAL STATE. When a run is ended by its limit (or is verified ended
      only at or after limit_until) the order is: (1) the attempt's `class: budget` / `ended_by: limit` and the block's
      stop file are written durably (fsync, atomic rename / exclusive link); (2) only then `state: ended`; (3) only then
      the watchdog is stood down. RECOVERY treats ANY attempt record with `ended_by: limit`, or whose watchdog says the job
      was alive at limit_until, or whose end time is at or after limit_until (kit/p4_budget.py
      `spending_limit_evidence`) as a budget termination and finishes the finalization -- `class: budget`, the stop if it
      is missing, the stand-down -- records already `ended` included. KIT_P4_TEST_PAUSE=<point> (tests only) makes the
      wrapper wait at a named point (after-budget-record, after-stop, after-termination, after-ended) so that a test can
      kill it there.

ROUND 6 (the manager's rulings J1, J2, J4), which governs where it differs from the text above:
  J1  TERMINATION DOES NOT WAIT ON DISCOVERY. Every end of a job -- the wrapper's at the limit, after the command's exit
      or on a signal, the watchdog's at limit_until, recovery's -- runs kit/p4_watchdog.py `JobEnd`: its FIRST action is
      killpg(group, SIGTERM) on the group id held in memory, before any scan and with no record written before it; at
      min(TERM + kill grace, kill_by) killpg(group, SIGKILL), again before any scan; both on a schedule in absolute
      monotonic time, in a thread that nothing else runs in, so no nvidia-smi, /proc or `ps` scan and no slow disk can
      delay them. Discovery (marker scan, group scan, GPU holders) runs alongside, each call bounded by the time left to
      the next hard action, and signals each member as it is found. At the limit the budget disposition and the stop
      (H2) are written AFTER the first TERM. The job's group exists BEFORE the watchdog starts: the command waits in the
      shim (start_contained) while the watchdog is started with that group (--pgid), and runs only once it watches.
  J2  ACKNOWLEDGEMENT. The wrapper does not return before the watchdog of its attempt has durably acknowledged the
      stand-down ("stood down") or recorded its own termination result ("acted"): it waits as long as the watchdog
      lives, up to the deadline + 20 s when the admitted time has ended (the watchdog acts by then) and otherwise
      KIT_P4_ACK_WAIT seconds (default 30: a stand-down is acknowledged within half a second). And no GPU row is
      admitted -- after recovery, before the idle check -- while any watchdog record of the same WORK is unacknowledged
      and its process alive: the admission waits within the same bounds, then refuses ("watchdog unacknowledged", exit
      3, nothing launched). Recovery stands a vanished wrapper's watchdog down after verifying its job gone, and the
      same gate then waits for that acknowledgement. A watchdog signals only its own group, its own marker's carriers,
      and GPU holders that started within its attempt (`job_window_start` .. the recorded end).
  J4  CONTAINMENT SEAM. The job is launched through ONE function, `start_contained(argv, env, limit_seconds, record)`,
      under KIT_P4_CONTAINMENT (`process-group` or `slurm-step`); the attempt
      record names the containment used; any other value (`cgroup` is an intended later one)
      refuses the launch before the gate, so nothing starts and no ledger line is written.

Exit: the command's status, 5 for wrong settings, 4 for an exit 0 without a complete export, 3 for busy GPUs or an
unacknowledged watchdog, 6 for workers not verified terminated, 1 or 2 for a refusal, 124 (137 when SIGKILL was needed)
for a run ended by its limit.
KIT_P4_KILL_GRACE (seconds) can only LOWER the kill grace (default 30); the tests use it. Standard library only;
kit/p4_budget.py, kit/p4_frozen.py, kit/p4_recipe.py and kit/p4_watchdog.py are loaded by file path, and
kit/p4_watchdog.py is also run by it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "kit-p4-attempt.v4"
EXIT_BUSY, EXIT_SETTINGS, EXIT_UNVERIFIED, EXIT_SIGNAL = 3, 5, 6, 130
IDLE_POLL = 1.0
KILL_GRACE = 30                          # SIGTERM, then SIGKILL after this many seconds; inside the 60-second reserve
KNOBS = {"grpo": ("NGPU", "TP", "OFFLOAD", "LORA"), "sdpo": ("NGPU", "TP", "OFFLOAD", "FEEDBACK", "SOFT", "TEMP")}
NOT_VERIFIED = "workers not verified terminated"
COMPLIANCE_UNAVAILABLE = "budget compliance unavailable"
LIVE_STATES = ("launching", "running")
WATCHDOG_DIR = Path("k8b4") / "budget" / "watchdog"
STAND_DOWN = "watchdog_stand_down"
KILL_MARGIN = 15.0                       # SIGKILL no later than deadline - 15 s (H1)
MIN_VERIFY = 10.0                        # after a normal end: at least this long to see the job empty after SIGKILL
PAUSE_POINTS = ("after-budget-record", "after-stop", "after-termination", "after-ended")
ACK_WAIT = 30.0                          # J2: how long a stand-down's acknowledgement is waited for (KIT_P4_ACK_WAIT lowers it)
ACK_MARGIN = 20.0                        # J2: a watchdog that acts at limit_until records its result by deadline + this
CONTAINMENT_VAR = "KIT_P4_CONTAINMENT"   # J4: how the job is contained
CONTAINMENTS = ("process-group", "slurm-step")  # explicit campaign choice; default retained for historical process tests
PLANNED_CONTAINMENTS = ("cgroup",)       # J4: the intended later values; refused until implemented
#: the launcher is started through this shim: its process group exists, and is recorded durably, BEFORE the command
#: runs; the command runs only after the wrapper writes "go" (a wrapper that dies first leaves nothing running)
EXEC_SHIM = ("import os, sys\n"
             "if sys.stdin.readline().strip() != 'go':\n"
             "    sys.exit(125)\n"
             "os.dup2(os.open(os.devnull, os.O_RDONLY), 0)\n"
             "try:\n"
             "    os.execvp(sys.argv[1], sys.argv[1:])\n"
             "except OSError as error:\n"
             "    sys.stderr.write('p4_run: cannot start %s: %s\\n' % (sys.argv[1], error))\n"
             "    sys.exit(127)\n")


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_p4_run_%s" % name, HERE / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pb = _load("p4_budget")
frozen_inputs = _load("p4_frozen")
wd = _load("p4_watchdog")
contain = _load("p4_contain")


def now_text(at: datetime | None = None) -> str:
    return (at or datetime.now(timezone.utc)).strftime(pb.STARTED_AT_FORMAT)


def parse_time(text):
    try:
        return datetime.strptime(text, pb.STARTED_AT_FORMAT).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def kill_grace() -> float:
    """The SIGTERM-to-SIGKILL grace: KILL_GRACE, or KIT_P4_KILL_GRACE when that is LOWER (the tests)."""
    try:
        lowered = float(os.environ.get("KIT_P4_KILL_GRACE", KILL_GRACE))
    except ValueError:
        lowered = KILL_GRACE
    return max(0.5, min(KILL_GRACE, lowered))


def deadline_grace() -> float:
    """The shutdown grace between the limit and the absolute deadline: the ledger's GRACE_SECONDS, or KIT_P4_GRACE when
    that is LOWER (the tests: an earlier deadline only charges less of the admitted allowance)."""
    try:
        lowered = float(os.environ.get("KIT_P4_GRACE", pb.GRACE_SECONDS))
    except ValueError:
        lowered = pb.GRACE_SECONDS
    return max(1.0, min(float(pb.GRACE_SECONDS), lowered))


def watchdog_record_path(work: Path, row: str, attempt: int) -> Path:
    return Path(work) / WATCHDOG_DIR / ("%s-a%d.json" % (row, attempt))


def read_json(path: Path):
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def pid_alive(pid) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        if Path('/proc/self/stat').exists():
            state = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[0]
            return state != 'Z'
        done = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    except OSError:
        return True  # kill(0) saw a process; unavailable discovery never proves death
    return bool(done.stdout.strip()) and not done.stdout.strip().startswith("Z")


def start_watchdog(record_path: Path, out: Path, pgid: int) -> tuple:
    """Round-4 ruling G1: the independent watchdog (kit/p4_watchdog.py), started in its own session (and forked once
    more, stdio closed) and GIVEN the job's process group (J1: held in memory from its start): (pid, None) once it
    watches, else (None, why)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        first = subprocess.Popen([sys.executable, str(HERE / "p4_watchdog.py"), "--record", str(record_path), "--out", str(out),
                                  "--pgid", str(pgid)],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True, close_fds=True, cwd="/")
        first.wait(timeout=30)                                 # the first process exits at once; the watcher is reparented
    except (OSError, subprocess.SubprocessError) as error:
        return None, "the watchdog could not be started: %s" % error
    end = time.monotonic() + 15
    while time.monotonic() < end:
        state = read_json(out) or {}
        if state.get("state") == "watching" and pid_alive(state.get("pid")):
            return state["pid"], None
        if state.get("state") == "refused":
            return None, "the watchdog refused: %s" % state.get("why")
        time.sleep(0.05)
    return None, "the watchdog did not report that it watches within 15 seconds (%s)" % (read_json(out) or {}).get("state")


def ack_wait() -> float:
    """J2: how long a stand-down's acknowledgement is waited for: ACK_WAIT, or KIT_P4_ACK_WAIT when that is LOWER."""
    try:
        lowered = float(os.environ.get("KIT_P4_ACK_WAIT", ACK_WAIT))
    except ValueError:
        lowered = ACK_WAIT
    return max(0.5, min(ACK_WAIT, lowered))


def watchdog_alive(pid) -> bool:
    """Whether the kit/p4_watchdog.py process `pid` still runs (a reused pid running something else is not it)."""
    if not pid_alive(pid):
        return False
    done = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    return "p4_watchdog.py" in done.stdout


def ack_bound(state: dict, first_seen: float) -> float:
    """J2: until when an unacknowledged watchdog is waited for: ack_wait() after it was first seen, or -- when it acts,
    or its admitted run time has ended -- until its deadline + ACK_MARGIN (it records its own result by then)."""
    bound = first_seen + ack_wait()
    limit_until, deadline = wd.parse_time(state.get("limit_until")), wd.parse_time(state.get("deadline"))
    if deadline is not None and (state.get("state") == "acting" or (limit_until is not None and time.time() + ack_wait() >= limit_until.timestamp())):
        bound = max(bound, deadline.timestamp() + ACK_MARGIN)
    return bound


def await_watchdog(out: Path, pid=None) -> dict:
    """J2: the watchdog's record once it has ACKNOWLEDGED (stood down, or recorded its own termination result), waiting
    as long as its process lives, within `ack_bound`. `acknowledged` False when it did not (it died, or the bound passed)."""
    first_seen = time.time()
    names = ("state", "pid", "stood_down_at", "acted_at", "job_alive_at_limit", "verified_idle_at", "term", "kill", "term_at", "kill_sent_at")
    while True:
        state = read_json(out) or {}
        pid = pid or state.get("pid")
        keys = {k: state.get(k) for k in names}
        if state.get("state") in wd.ACKNOWLEDGED:
            return {**keys, "acknowledged": True, "acknowledged_seen_at": wd.precise_text()}
        if not watchdog_alive(pid):
            state = read_json(out) or {}                         # it may have acknowledged, then exited, between the two reads
            keys = {k: state.get(k) for k in names}
            if state.get("state") in wd.ACKNOWLEDGED:
                return {**keys, "acknowledged": True, "acknowledged_seen_at": wd.precise_text()}
            return {**keys, "acknowledged": False, "why": "the watchdog %s is gone without acknowledging" % pid}
        if time.time() >= ack_bound(state, first_seen):
            return {**keys, "acknowledged": False, "why": "the watchdog %s did not acknowledge in time" % pid}
        time.sleep(0.1)


def unacknowledged_watchdogs(work: Path) -> list:
    """J2: every watchdog record of WORK whose state is not an acknowledgement and whose process is alive."""
    out = []
    for path in sorted((Path(work) / WATCHDOG_DIR).glob("*.json")):
        state = read_json(path) or {}
        if state.get("state") in wd.ACKNOWLEDGED or not watchdog_alive(state.get("pid")):
            continue
        out.append({"record": path.stem, "state": state.get("state"), "pid": state.get("pid"), "limit_until": state.get("limit_until"),
                    "deadline": state.get("deadline")})
    return out


def acknowledgement_gate(work: Path) -> tuple:
    """J2, before any admission: (True, [the watchdogs waited for]) once no watchdog of WORK is alive and
    unacknowledged; (False, why) when one stays so past its `ack_bound`."""
    first_seen, waited = {}, {}
    while True:
        pending = unacknowledged_watchdogs(work)
        if not pending:
            return True, list(waited.values())
        now, late = time.time(), []
        for entry in pending:
            first_seen.setdefault(entry["record"], now)
            waited.setdefault(entry["record"], {**entry, "first_seen": wd.precise_text(wd.from_epoch(first_seen[entry["record"]]))})
            if now >= ack_bound(entry, first_seen[entry["record"]]):
                late.append(entry)
        if late:
            return False, ("watchdog unacknowledged: the watchdog of %s (pid %s, state %s) is alive and has not acknowledged a stand-down "
                           "or recorded its own termination result, so no other GPU row is admitted (round-6 ruling J2)"
                           % (", ".join(e["record"] for e in late), ", ".join(str(e["pid"]) for e in late),
                              ", ".join(str(e["state"]) for e in late)))
        time.sleep(0.2)


# ------------------------------------------------------------------------------------------------ J4: containment
class ContainmentError(Exception):
    """The containment setting names no implemented containment: nothing is launched."""


def containment_setting() -> tuple:
    """J4: (the containment KIT_P4_CONTAINMENT names, None), or (None, why) for a value that is not implemented."""
    value = os.environ.get(CONTAINMENT_VAR, "").strip() or CONTAINMENTS[0]
    if value in CONTAINMENTS:
        return value, None
    later = " (an intended later value, not implemented yet)" if value in PLANNED_CONTAINMENTS else ""
    return None, ("unknown containment %s=%r%s: the implemented value is %s, so nothing is launched (round-6 ruling J4)"
                  % (CONTAINMENT_VAR, value, later, " or ".join(CONTAINMENTS)))


class ContainedJob:
    """A job started by `start_contained`: its command waits (parked in the shim) until `release()`."""

    def __init__(self, popen, containment: str, limit_seconds: int, window_start):
        self.popen, self.containment, self.limit_seconds, self.window_start = popen, containment, limit_seconds, window_start
        self.pid = popen.pid
        self.pgid = os.getpgid(popen.pid)

    def release(self) -> None:
        """The group is on record and watched: the command runs now."""
        self.popen.stdin.write(b"go\n")
        self.popen.stdin.close()

    def abort(self) -> None:
        """Nothing was released: end the parked group (the command never ran)."""
        try:
            os.killpg(self.pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.popen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def wait(self, timeout=None):
        return self.popen.wait(timeout=timeout)

    def poll(self):
        return self.popen.poll()

    @property
    def returncode(self):
        return self.popen.returncode


class SlurmJob(ContainedJob):
    def __init__(self, proc, release_file, record):
        self.popen, self.record, self.release_file = proc, record, release_file
        self.pid, self.pgid = proc.pid, record.data['pgid']
        self.containment, self.limit_seconds = 'slurm-step', record.data['limit_seconds']
        self.window_start = record.data.get('job_window_start')

    def release(self):
        self.release_file.touch()
        self.record.update(slurm={**self.record.data['slurm'], 'released_at': wd.precise_text()})

    def abort(self):
        s = self.record.data['slurm']
        c = contain.cancel(s['job_id'], s['step_id'])
        e = contain.evidence(s['job_id'], s['step_id'], contain.scheduler_devices(self.record.data), self.record.data['deadline'])
        self.record.update(slurm={**s, 'cancel': c, 'termination': e})
        if e.get('verified'):
            at = ceil_text(wd.parse_time(e['verified_at']))
            self.record.update(charged_until=at, verified_idle_at=at)
        else:
            self.record.update(accounting=COMPLIANCE_UNAVAILABLE)
            wd.write_stop(self.record.path, self.record.data, 'aborted parked step not verified gone', NOT_VERIFIED, unavailable=True)
        if self.popen.poll() is None:
            self.popen.kill()
        self.popen.wait(timeout=10)


def start_contained(argv: list, env: dict, limit_seconds: int, record) -> ContainedJob:
    """Round-6 ruling J4: the ONE way a package-4 job is launched, under the containment the attempt record names
    (`containment`, from KIT_P4_CONTAINMENT). Implemented:
      process-group  (the default; today's behaviour) the command runs through EXEC_SHIM in its OWN session and process
                     group, with the KIT_P4_JOB marker in `env`; its members are that group, the marker's carriers and
                     its GPU holders (kit/p4_watchdog.py); the group exists and is recorded durably -- pid, pgid,
                     job_window_start (the group's start, the GPU-holder window of J2), containment -- BEFORE the
                     command runs: it waits until `release()`. `limit_seconds` is enforced by this wrapper and the
                     watchdog (J1), not by the containment.
      slurm-step     the command parks inside one scheduler-timed step, whose start and GPU identities
                     are rechecked before release (p4_contain.start_step). Normal exit/abort cancels
                     only that step. The frozen cluster receipt and adversarial selftest are required.
    Intended later value, refused: cgroup (an independently owned dispatcher boundary).
    Any value but an implemented one raises ContainmentError before anything starts."""
    containment = record.data.get("containment")
    if containment not in CONTAINMENTS:
        raise ContainmentError("containment %r is not implemented (implemented: %s; intended later: %s)"
                               % (containment, ", ".join(CONTAINMENTS), ", ".join(PLANNED_CONTAINMENTS)))
    if containment == 'slurm-step':
        try:
            count = int((record.data.get('gate') or {}).get('gpus') or env.get('P4_GPUS', '1'))
            proc, release_file = contain.start_step(argv, env, record, count, require_selftest=env.get('P4_KIND') != 'selftest')
            return SlurmJob(proc, release_file, record)
        except contain.Refused as e:
            raise ContainmentError(str(e)) from None
    popen = subprocess.Popen([sys.executable, "-c", EXEC_SHIM] + list(argv), env=env, stdin=subprocess.PIPE, start_new_session=True)
    job = ContainedJob(popen, containment, limit_seconds, wd.process_start_time(popen.pid))
    record.update(pid=job.pid, pgid=job.pgid, job_window_start=job.window_start, containment=containment,
                  containment_detail={"limit_seconds": limit_seconds, "session_and_group": job.pgid,
                                      "members": "group, KIT_P4_JOB marker, GPU holders started within the attempt"})
    return job


class Record:
    """The attempt record, written durably at every step (a crash leaves the last step behind)."""

    def __init__(self, work: Path, row: str, attempt: int, fields: dict):
        self.path = pb.attempt_record_path(work, row, attempt)
        if self.path.exists():
            raise FileExistsError("%s exists: an attempt is recorded once" % self.path)
        self.data = {"schema": SCHEMA, "row": row, "attempt": attempt, "state": "admitting", "launched": False, "ok": 0,
                     "export_complete": 0, "settings_mismatch_count": 0, "started_at": now_text(), "ended_at": None,
                     "wrapper_pid": os.getpid(), **fields}
        self.save()

    def update(self, **fields):
        self.data.update(fields)
        self.save()

    def save(self):
        write_record(self.path, self.data)


def write_record(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(data, handle, indent=1, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ GPUs and processes
def busy_pids(devices):
    """The compute-process pids nvidia-smi lists on `devices` (None: every GPU); None when there is no nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"] + (["-i", devices] if devices else [])
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return ["nvidia-smi did not answer within 60 seconds"]
    if done.returncode != 0:
        return ["nvidia-smi exited %d: %s" % (done.returncode, (done.stderr or done.stdout).strip()[:200])]
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def wait_idle(devices, wait: float) -> tuple:
    """(True, note) once no compute process is listed; (False, why) after `wait` seconds or without nvidia-smi."""
    deadline = time.monotonic() + max(0.0, wait)
    while True:
        pids = busy_pids(devices)
        if pids is None:
            if os.environ.get("KIT_ALLOW_NO_NVIDIA_SMI") == "1":
                return True, "no nvidia-smi on PATH; KIT_ALLOW_NO_NVIDIA_SMI=1 (dry run only)"
            return False, "no nvidia-smi on PATH: the GPUs cannot be shown idle, so nothing is launched (fail closed)"
        if not pids:
            return True, "no compute process on GPUs %s" % (devices or "all")
        if time.monotonic() >= deadline:
            return False, "GPU still busy: compute processes %s on GPUs %s after %d seconds" % (", ".join(pids[:8]), devices or "all", wait)
        time.sleep(min(IDLE_POLL, max(0.1, deadline - time.monotonic())))


def group_alive(pgid) -> bool:
    if not isinstance(pgid, int) or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_margin(allowance: float) -> float:
    """SIGKILL comes no later than deadline - KILL_MARGIN (15 s); a quarter of the allowance when a test lowers it."""
    return min(KILL_MARGIN, max(0.25, allowance / 4.0))


def ceil_text(at: datetime) -> str:
    """Whole seconds, rounded UP: a charged or verified time is never written earlier than it was."""
    if at.microsecond:
        at = datetime.fromtimestamp(int(at.timestamp()) + 1, timezone.utc)
    return now_text(at)


def test_pause(path: Path, point: str) -> None:
    """Crash injection for the tests (round-5 ruling H2): with KIT_P4_TEST_PAUSE=<point> the wrapper announces the point
    (a file <record>.paused-<point>) and waits there, so that a test can SIGKILL it between two durable writes."""
    if os.environ.get("KIT_P4_TEST_PAUSE") != point:
        return
    flag = Path(str(path) + ".paused-" + point)
    flag.write_text(str(os.getpid()))
    time.sleep(float(os.environ.get("KIT_P4_TEST_PAUSE_SECONDS", "300")))
    flag.unlink(missing_ok=True)


def terminate_job(record: dict, kill_after: float, verify_within: float, reap=None, window_end=None) -> dict:
    """End the WHOLE job of an attempt record (kit/p4_watchdog.py `end_job`: its process group, every process carrying
    its marker, every compute process of this user on its GPUs): TERM now, KILL after `kill_after` seconds (never later
    than its kill_by), re-scanned until empty, at most `verify_within` seconds (never later than its deadline when that
    is still ahead)."""
    now = time.time()
    kill_at, verify_by = now + kill_after, now + verify_within
    kill_by = wd.parse_time(record.get("kill_by"))
    deadline = parse_time(record.get("deadline"))
    if kill_by is not None and kill_by.timestamp() > now:
        kill_at = min(kill_at, kill_by.timestamp())
    if deadline is not None and deadline.timestamp() > now:
        verify_by = min(verify_by, deadline.timestamp())
    return wd.end_job(record.get("job_marker"), record.get("pgid"), record.get("cuda_visible_devices") or None,
                      kill_at=kill_at, verify_by=max(verify_by, kill_at + 1.0), reap=reap,
                      window_start=record.get("job_window_start"), window_end_of=(lambda: window_end) if window_end is not None else None, scheduler=record.get("slurm"))


def job_summary(ended: dict) -> dict:
    last = ended.get("last") or {}
    keys = ("group", "marker", "gpu", "gpu_other", "gpu_outside_window", "errors")
    return {"alive_at_start": ended.get("alive_at_start"), "term": ended.get("term"), "kill": ended.get("kill"),
            "term_at": ended.get("term_at"), "kill_sent_at": ended.get("kill_sent_at"), "verified": ended.get("verified"),
            "verified_at": wd.precise_text(wd.from_epoch(ended["verified_at"])) if ended.get("verified") else None,
            # J1: the schedule (when the TERM and the KILL were due and when the KILL action ran) and what each reached
            "term_due": ended.get("term_due"), "kill_at": ended.get("kill_at"), "kill_action_at": ended.get("kill_action_at"),
            "group_term_delivered": ended.get("group_term_delivered"), "group_kill_delivered": ended.get("group_kill_delivered"),
            "signalled": ended.get("signalled"), "window": ended.get("window"), "group_signal_errors": ended.get("group_signal_errors"),
            "first_scan": {k: (ended.get("first") or {}).get(k) for k in keys},
            "last_scan": {k: last.get(k) for k in keys}, "scans": ended.get("scans")}


def wrapper_alive(pid) -> bool:
    """Whether the kit/p4_run.py process `pid` still runs (a reused pid running something else is not it)."""
    if not isinstance(pid, int) or pid <= 1 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    done = subprocess.run(["ps", "-o", "stat=,command=", "-p", str(pid)], capture_output=True, text=True)
    text = done.stdout.strip()
    return bool(text) and not text.startswith("Z") and "p4_run.py" in text


class WrapperSignal(Exception):
    """This wrapper was asked to stop (SIGTERM, SIGINT, SIGHUP): its workers are ended before it exits."""


def _raise(signum, _frame):
    raise WrapperSignal(signal.Signals(signum).name)


# ------------------------------------------------------------------------------------------------ 3. recovery
def _stop(work: Path, record: dict, at: datetime, reason: str, cause: str, unavailable: bool = False) -> bool:
    if record.get('containment')=='slurm-step':contain.hard_stop(work,reason)
    return pb.record_stop(work, record.get("block"), {"time": now_text(at), "row": record.get("row"), "attempt": record.get("attempt"),
                                                      "reason": "%s: %s" % (COMPLIANCE_UNAVAILABLE if unavailable else pb.BUDGET_INCOMPLETE, reason),
                                                      "cause": cause, **({"accounting": "unavailable"} if unavailable else {})})


def trusted_termination(watch) -> tuple:
    """(time, how) from the watchdog's durable record when it verified the WHOLE job gone (group, marker-carrying
    processes, GPU compute processes); else (None, why). The wrapper's own verified time needs no recovery."""
    if not isinstance(watch, dict):
        return None, "no watchdog record"
    if watch.get("state") != "acted":
        return None, "the watchdog record is %r, not a termination record" % watch.get("state")
    at = wd.parse_time(watch.get("job_gone_verified_at_precise")) or parse_time(watch.get("verified_idle_at"))
    gone = watch.get("job_gone") if "job_gone" in watch else watch.get("group_gone")
    if not gone or at is None or not watch.get("verified_idle_at"):
        return None, "the watchdog did not verify termination (job gone %r: %s)" % (gone, watch.get("idle_note"))
    alive = watch.get("job_alive_at_limit", watch.get("group_alive_at_deadline"))
    return at, "the watchdog %s the job at the end of its admitted run time %s and verified it gone (GPUs idle) at %s" % (
        "ended" if alive else "found nothing alive of", watch.get("limit_until") or watch.get("deadline"),
        watch.get("job_gone_verified_at_precise") or watch.get("verified_idle_at"))


def settle_budget(work: Path, path: Path, record: dict, fields: dict | None = None, at: datetime | None = None) -> dict | None:
    """Round-5 ruling H2, for one attempt record: when it holds spending-limit evidence (kit/p4_budget.py
    `spending_limit_evidence`, judged on the record as it will be after `fields`), the budget disposition is made durable
    in the order (1) `class: budget` written, then the block's stop if it is missing, (2) then `fields` (a terminal
    state, if any), (3) then the stand-down of the watchdog. Without evidence, `fields` are written as they are. Returns
    what was finished, or None."""
    fields = dict(fields or {})
    would = {**record, **fields}
    why = pb.spending_limit_evidence(work, would)
    finished = {}
    if why:
        if record.get("class") != "budget" or fields.get("class", "budget") != "budget":
            first = {"class": "budget", "ok": 0, "budget_evidence": why}
            if record.get("class") not in (None, "budget"):
                first["class_before_budget"] = record.get("class")
            record.update(first)
            write_record(path, record)                                              # (1) the disposition
            finished["class"] = "budget"
        fields.pop("class", None)
        fields["budget_evidence"] = why
        if pb.read_stop(work, record.get("block")) is None:
            finished["stop_written"] = _stop(work, record, at or datetime.now(timezone.utc),
                                             "row %s attempt %s was ended by its spending limit (%s)" % (record.get("row"), record.get("attempt"), why),
                                             "spending-limit termination (%s)" % why)
    stand_down = fields.pop(STAND_DOWN, None)
    if fields:
        record.update(fields)
        write_record(path, record)                                                  # (2) the terminal state
    if record.get("state") == "ended" and not record.get(STAND_DOWN):
        record[STAND_DOWN] = stand_down or now_text()
        write_record(path, record)                                                  # (3) the watchdog stands down
        finished["stood_down"] = True
    return finished or None


def recover(work: Path, me: Path, idle_wait: float) -> list:
    """Round-4 ruling G1 and round-5 rulings H1, H2: BEFORE anything is admitted,
      - every attempt record still launching/running whose wrapper is gone is reconciled. A termination time is never
        inferred: it is charged through a TRUSTWORTHY timestamp (the watchdog's record of the time it verified the whole
        job gone), or, with none, through the moment THIS recovery ends the whole job (group, marker, GPUs) and verifies
        it empty (now); if that cannot be verified the attempt is `unverified`, the block's accounting is "budget
        compliance unavailable" and its durable stop is written;
      - EVERY record (ended ones included) holding spending-limit evidence -- `ended_by: limit`, a watchdog that found the
        job alive at limit_until, an end at or after limit_until -- is finished as a budget termination: `class: budget`,
        the block's stop if missing, the stand-down (H2: incomplete finalizations are completed);
      - a record left `unverified` is closed when a later recovery verifies its job empty (its block's stop stands).
    Returns what was done, one entry per record."""
    done = []
    for path in sorted((Path(work) / pb.ATTEMPTS_DIR).glob("*.json")):
        if path.resolve() == Path(me).resolve():
            continue
        record = read_json(path)
        if record is None:
            continue
        if record.get('containment') == 'slurm-step' and record.get('state') in (*LIVE_STATES, 'unverified') and not wrapper_alive(record.get('wrapper_pid')):
            s = record.get('slurm') or {}
            if s.get('step_id'):
                c = contain.cancel(s['job_id'], s['step_id'])
                e = contain.evidence(s['job_id'], s['step_id'], contain.scheduler_devices(record), time.time()+max(10, idle_wait))
                record['slurm'] = {**s, 'cancel': c, 'termination': e}
                if e.get('verified'):
                    at = wd.parse_time(e['verified_at'])
                    watch = read_json(watchdog_record_path(work, record.get('row'), record.get('attempt'))) or {}
                    trusted = watch.get('scheduler_termination') or {}
                    if trusted.get('verified') and trusted.get('step_gone') and trusted.get('gpus_idle') and wd.parse_time(trusted.get('verified_at')):
                        prior = wd.parse_time(trusted['verified_at'])
                        candidate = {**record, 'charged_until': ceil_text(prior),
                                     'slurm': {**s, 'termination': trusted}}
                        if prior < at and not contain.scheduler_problems(work, candidate):
                            # Preserve the proof whose verified timestamp is charged;
                            # attaching the fresh observation to an earlier charge
                            # would make the archive internally inconsistent.
                            at = prior
                            record['slurm'].update(termination=trusted, recovery_observation=e)
                    fields = {'state': 'ended', 'class': e.get('failure_type') or ('preempted' if e.get('state') == 'PREEMPTED' else 'interrupted'),
                              'failure_type': e.get('failure_type') or ('preempted' if e.get('state') == 'PREEMPTED' else None), 'charged_until': ceil_text(at), 'ended_at': ceil_text(at),
                              'ended_precise': wd.precise_text(at), 'verified_idle_at': ceil_text(at), 'termination_source': 'slurm recovery', 'past_limit': e.get('state') == 'TIMEOUT' or at >= wd.parse_time(record['limit_until']),
                              'terminated_by_deadline': at <= wd.parse_time(record['deadline']), 'slurm': record['slurm']}
                    if fields.get('failure_type'):
                        fields.update(ok=0, status=2)
                        fields['stop_class']=contain.stop_class(fields['failure_type'],True,not fields['past_limit'])
                        if fields['stop_class']=='hard':
                            _stop(work, record, at, 'scheduler failure: '+fields['failure_type'], fields['failure_type'])
                            contain.hard_stop(work,fields['failure_type'],path)
                    settle_budget(work, path, record, fields, at)
                    done.append({'row': record.get('row'), 'attempt': record.get('attempt'), 'slurm': e})
                    continue
                record.update(state='unverified', accounting=COMPLIANCE_UNAVAILABLE, **{'class': NOT_VERIFIED})
                write_record(path, record)
                _stop(work, record, datetime.now(timezone.utc), 'owning step not verified gone', NOT_VERIFIED, unavailable=True)
                continue
        if record.get("state") not in LIVE_STATES or wrapper_alive(record.get("wrapper_pid")):
            if record.get("state") not in LIVE_STATES:
                finished = settle_budget(work, path, record)
                if finished:
                    done.append({"record": path.name, "state": record.get("state"), "class": record.get("class"), "finished": finished})
            continue
        row, attempt = record.get("row"), record.get("attempt")
        limit_until = pb.limit_until_of(record)
        watch_path = watchdog_record_path(work, row, attempt) if isinstance(attempt, int) else None
        watch = read_json(watch_path) if watch_path else None
        watcher = (watch or {}).get("pid") or (record.get("watchdog") or {}).get("pid")
        if limit_until is not None and datetime.now(timezone.utc) >= limit_until and (watch or {}).get("state") in ("starting", "watching", "acting") \
                and pid_alive(watcher):
            # the admitted run time has ended and the watchdog is at work: give it the time its shutdown takes (at most
            # until the deadline, and a little more)
            deadline = parse_time(record.get("deadline")) or (limit_until + timedelta(seconds=pb.GRACE_SECONDS))
            end = time.monotonic() + max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds()) + 20
            while time.monotonic() < end and (read_json(watch_path) or {}).get("state") in ("starting", "watching", "acting") and pid_alive(watcher):
                time.sleep(0.2)
            watch = read_json(watch_path)
        at, how = trusted_termination(watch)
        fields = {"recovered_at": now_text(), "recovered_by": {"row": Path(me).stem, "wrapper_pid": os.getpid()},
                  "watchdog_record": {k: (watch or {}).get(k) for k in ("state", "acted_at", "job_alive_at_limit", "job_gone",
                                                                         "job_gone_verified_at_precise", "gpus_idle", "verified_idle_at")}}
        if at is not None:
            leftover = terminate_job(record, kill_grace(), kill_grace() + MIN_VERIFY,      # not expected after a verified end
                                     window_end=at.timestamp())
            past = bool((watch or {}).get("job_alive_at_limit", (watch or {}).get("group_alive_at_deadline"))) or \
                (limit_until is not None and at >= limit_until)
            fields.update({"state": "ended", "class": "interrupted", "ok": 0, "charged_until": ceil_text(at), "verified_idle_at": ceil_text(at),
                           "ended_at": ceil_text(at), "ended_precise": wd.precise_text(at), "past_limit": past,
                           "past_deadline": bool(parse_time(record.get("deadline")) and at > parse_time(record.get("deadline"))),
                           "termination_source": "watchdog", STAND_DOWN: now_text(),
                           "recovery": {"trusted_termination": how, "leftover_scan": job_summary(leftover)},
                           "reason": "interrupted: the wrapper of attempt %s ended without recording an end; %s; charged until %s%s"
                                     % (attempt, how, ceil_text(at), " (a spending-limit termination)" if past else "")})
            settle_budget(work, path, record, fields, at)
        else:
            ended = terminate_job(record, kill_grace(), kill_grace() + max(idle_wait, MIN_VERIFY))
            now = wd.from_epoch(ended["verified_at"]) if ended["verified"] else datetime.now(timezone.utc)
            fields["recovery"] = {"trusted_termination": None, "why_none": how, "job": job_summary(ended)}
            if not ended["verified"]:
                reason = ("%s: the wrapper of attempt %s is gone, no trustworthy termination time was recorded (%s), and this recovery "
                          "cannot verify termination now (%s)" % (NOT_VERIFIED, attempt, how, wd._describe(ended["last"])))
                fields.update({"state": "unverified", "class": NOT_VERIFIED, "ok": 0, "accounting": COMPLIANCE_UNAVAILABLE, "reason": reason})
                record.update(fields)
                write_record(path, record)
                _stop(work, record, now, reason, NOT_VERIFIED, unavailable=True)
            else:
                past = limit_until is None or now >= limit_until
                fields.update({"state": "ended", "class": "interrupted", "ok": 0, "charged_until": ceil_text(now), "verified_idle_at": ceil_text(now),
                               "ended_at": ceil_text(now), "ended_precise": wd.precise_text(now), "past_limit": past,
                               "past_deadline": bool(parse_time(record.get("deadline")) is None or now > parse_time(record.get("deadline"))),
                               "termination_source": "recovery", STAND_DOWN: now_text(now),
                               "reason": "interrupted: the wrapper of attempt %s ended without recording an end and no trustworthy termination "
                                         "time was recorded (%s); this recovery verified the whole job gone at %s and charges it until "
                                         "then%s" % (attempt, how, ceil_text(now), " (at or after the end of its admitted run time %s: a "
                                                                                   "spending-limit termination)" % record.get("limit_until") if past else "")})
                settle_budget(work, path, record, fields, now)
        done.append({"record": path.name, "state": record["state"], "class": record["class"], "charged_until": record.get("charged_until"),
                     "termination_source": record.get("termination_source")})
    for path in sorted((Path(work) / pb.ATTEMPTS_DIR).glob("*.json")):
        record = read_json(path)
        if record is not None and record.get("state") == "unverified" and not record.get("verified_idle_at") and path.resolve() != Path(me).resolve():
            ended = terminate_job(record, kill_grace(), kill_grace() + MIN_VERIFY)
            if ended["verified"]:
                at = wd.from_epoch(ended["verified_at"])
                record.update({"verified_idle_at": ceil_text(at), "charged_until": ceil_text(at), "ended_at": record.get("ended_at") or ceil_text(at),
                               "termination_source": "a later recovery's verification", STAND_DOWN: now_text()})
                write_record(path, record)
                done.append({"record": path.name, "state": "unverified", "verified_idle_at": record["verified_idle_at"]})
    return done


# ------------------------------------------------------------------------------------------------ 5. reconcile
def run_key(row: str) -> str:
    return row


def earlier_runs(work: Path, key: str, attempt: int) -> list:
    """[(k, folder)] of runs/<key>-a<k> with k < attempt, lowest first."""
    out = []
    for folder in (Path(work) / "runs").glob("%s-a*" % key):
        tail = folder.name[len(key) + 2:]
        if folder.is_dir() and tail.isdigit() and int(tail) < attempt:
            out.append((int(tail), folder))
    return sorted(out)


def export_detail(folder: Path) -> dict:
    """The export-validation result of one merged-model folder, file by file (archived with the attempt)."""
    folder = Path(folder)
    detail = {"folder": folder.name, "exists": folder.is_dir(), "config": (folder / "config.json").is_file(),
              "tokenizer_config": (folder / "tokenizer_config.json").is_file(), "complete": pb.complete_export(folder)}
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = sorted(set(json.loads(index.read_text()).get("weight_map", {}).values()))
        except (OSError, ValueError, AttributeError):
            shards = []
        detail["shards"] = {s: ((folder / s).stat().st_size if (folder / s).is_file() else None) for s in shards}
    elif (folder / "model.safetensors").exists():
        detail["shards"] = {"model.safetensors": (folder / "model.safetensors").stat().st_size}
    return detail


def tail(path: Path, lines: int = 40) -> list:
    try:
        return Path(path).read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return []


# ------------------------------------------------------------------------------------------------ 1. knobs, 2. inputs
def baseline_knobs(path: Path, row: str) -> tuple:
    """({knob: value}, None) from the frozen, verified baseline of this row; (None, why) otherwise."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        return None, "the verified pilot baseline %s cannot be read: %s" % (path, error)
    entry = (doc.get("rows") or {}).get(row)
    if doc.get("ok") is not True or not isinstance(entry, dict) or entry.get("ok") is not True:
        return None, "the pilot baseline %s does not verify row %s (recipe-check did not pass for it)" % (path, row)
    knobs = entry.get("knobs")
    if not isinstance(knobs, dict) or not knobs:
        return None, "the pilot baseline %s records no launcher knobs for %s" % (path, row)
    return {k: str(v) for k, v in knobs.items()}, None


def incoming_changes(work: Path, entry: dict, recipe=None) -> list:
    """Round-4 ruling G2: the incoming checkpoint of a training or qualification launch is the one the recipe check
    identified, by CONTENT: the sha256 of every file of its export folder (config, tokenizer files, every weight shard),
    recorded in the FROZEN recipe-check document (WORK/k8b4/recipe-check/frozen.json, verified by kit/p4_frozen.py before
    this runs) and re-hashed now (streaming). A missing identity, a changed file list or any changed content refuses the
    launch, and so does a MODEL_DIR that is not that folder."""
    recipe = recipe or _load("p4_recipe")
    work = Path(work)
    incoming = entry.get("incoming") if isinstance(entry, dict) else None
    frozen = read_json(work / frozen_inputs.RECIPE_FROZEN)
    identity = ((((frozen or {}).get("comparison") or {}).get("incoming") or {}).get(str(Path(incoming).parent)) or {}) if incoming else {}
    recorded = identity.get("export_sha256")
    if not incoming or not isinstance(recorded, dict) or not recorded:
        return ["the incoming checkpoint's identity is not recorded: the frozen recipe-check document %s holds no content hashes of "
                "%s, so nothing is launched (round-4 ruling G2)" % (frozen_inputs.RECIPE_FROZEN, incoming or "the row's incoming export")]
    problems = []
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir or Path(model_dir).resolve() != (work / incoming).resolve():
        problems.append("MODEL_DIR %r is not the incoming checkpoint the recipe check identified (%s)" % (model_dir, work / incoming))
    now = recipe.export_identity(work / incoming)
    if now != recorded:
        added, removed = sorted(set(now) - set(recorded)), sorted(set(recorded) - set(now))
        changed = sorted(k for k in set(now) & set(recorded) if now[k] != recorded[k])
        problems.append("incoming checkpoint changed since the recipe check: %s (files added %s; removed %s; content changed %s)"
                        % (incoming, ", ".join(added[:5]) or "none", ", ".join(removed[:5]) or "none", ", ".join(changed[:5]) or "none"))
    return problems


def environment_changes(baseline: Path, row: str, work: Path | None = None) -> list:
    """For a training or qualification row (F2, F4, G2): the trainer checkout, the task data and the incoming
    checkpoint, against the frozen recipe-check records; empty when they are unchanged."""
    recipe = _load("p4_recipe")
    try:
        doc = json.loads(Path(baseline).read_text())
    except (OSError, ValueError):
        return ["the frozen baseline %s cannot be read" % baseline]
    entry = (doc.get("rows") or {}).get(row) or {}
    sdpo, problems = doc.get("sdpo_dir"), []
    if not sdpo or os.environ.get("SDPO_DIR") != sdpo:
        problems.append("SDPO_DIR is %r, the recipe check verified %r" % (os.environ.get("SDPO_DIR"), sdpo))
        return problems
    trainer, frozen = recipe.trainer_identity(sdpo), doc.get("trainer") or {}
    if trainer.get("commit") != frozen.get("commit"):
        problems.append("the trainer checkout is at commit %s, the recipe check verified %s" % (trainer.get("commit"), frozen.get("commit")))
    if not trainer.get("status_empty"):
        problems.append("the trainer checkout is not clean (git status --short: %s)" % "; ".join((trainer.get("status") or "").strip().splitlines()[:3]))
    if trainer.get("sources_combined_sha256") != frozen.get("sources_combined_sha256"):
        problems.append("the trainer's entry point and configuration sources changed since the recipe check")
    identity = entry.get("data_identity") or {}
    now = recipe.data_hashes(sdpo, identity.get("dataset") or "")
    if not identity.get("now") or now != identity.get("now"):
        problems.append("the task data %s changed since the recipe check (%s)" % (identity.get("dataset"), now))
    problems += incoming_changes(Path(work) if work is not None else Path(doc.get("work_root") or "."), entry, recipe)
    return problems


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--row", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--train", action="store_true", help="a training launch: reconcile first, classify the export")
    parser.add_argument("--baseline", help="kit/p4_recipe.py's frozen baseline.json (default WORK/k8b4/recipe-check/baseline.json)")
    parser.add_argument("--idle-wait", type=float, default=120.0)
    args = parser.parse_args(argv[:split])                   # --help prints and exits 0 here
    command = argv[split + 1:]
    if not command:
        print("p4_run: no command after --", file=sys.stderr)
        return pb.EXIT_UNAVAILABLE
    work, row, attempt = Path(args.work), args.row, args.attempt
    baseline_path = Path(args.baseline) if args.baseline else work / frozen_inputs.RECIPE_BASELINE
    block, kind = os.environ.get("P4_BLOCK"), os.environ.get("P4_KIND")
    if block not in pb.BLOCKS or kind not in pb.GPU_KINDS:
        print("p4_run: the row's env must carry P4_BLOCK (%s) and a GPU P4_KIND (%s); found %r, %r"
              % ("/".join(pb.BLOCKS), "/".join(pb.GPU_KINDS), block, kind), file=sys.stderr)
        return pb.EXIT_UNAVAILABLE
    devices = os.environ.get("CUDA_VISIBLE_DEVICES") or None
    try:
        record = Record(work, row, attempt, {"block": block, "kind": kind, "train": args.train, "command": command,
                                             "cuda_visible_devices": devices})
    except FileExistsError as error:
        print("p4_run refused: %s" % error, file=sys.stderr)
        return pb.EXIT_UNAVAILABLE

    def refuse(cls: str, why: str, code: int = pb.EXIT_UNAVAILABLE) -> int:
        record.update(**{"state": "refused", "class": cls, "reason": why, "ended_at": now_text(), "elapsed_seconds": 0.0, "status": None})
        print("p4_run refused: %s" % why, file=sys.stderr)
        return code

    # 0. J4: the containment the job will run under; an unknown one launches nothing (before the gate: no ledger line)
    containment, why = containment_setting()
    record.update(containment=containment)
    if containment is None:
        return refuse("refused", why)
    if containment == 'slurm-step':
        contain.track_allocation(work)
        problems, _docs = contain.verify_receipts(work, selftest=kind != 'selftest')
        if problems:
            return refuse('refused', '; '.join(problems))
    # 1. the frozen knobs, before anything reads the environment (should-fix of round 2)
    knobs = None
    if args.train:
        knobs, why = baseline_knobs(baseline_path, row)
        if knobs is None:
            return refuse("refused", why)
    # 2. the frozen decision inputs (F2), and for training the trainer, the data and the incoming checkpoint (F4)
    stale = frozen_inputs.verify(work, reservation=args.reservation, baseline=baseline_path,
                                 containment=containment == 'slurm-step' and kind != 'selftest')
    if args.train and not stale:
        stale = environment_changes(baseline_path, row, work)
    if stale:
        return refuse("refused", frozen_inputs.refusal(stale))
    # 3. recovery of interrupted attempts, before anything is admitted (F1)
    record.update(recovered=recover(work, record.path, args.idle_wait))
    # 3b. J2: no admission while a watchdog of this WORK is alive and has not acknowledged
    acknowledged, waited = acknowledgement_gate(work)
    if not acknowledged:
        record.update(**{"state": "busy", "class": "watchdog unacknowledged", "reason": waited, "ended_at": now_text(),
                         "elapsed_seconds": 0.0, "status": None})
        print(waited, file=sys.stderr)
        return EXIT_BUSY
    record.update(watchdogs_waited_for=waited)
    # 4. idle GPUs
    idle, note = wait_idle(devices, args.idle_wait)
    if not idle:
        record.update(**{"state": "busy", "class": "gpu busy", "reason": note, "ended_at": now_text(), "elapsed_seconds": 0.0, "status": None})
        print(note, file=sys.stderr)
        return EXIT_BUSY
    record.update(idle_check=note)

    env = dict(os.environ)
    steps = env.get("STEPS", "")
    campaign_name = os.environ.get("P4_CAMPAIGN_NAME") or Path(args.campaign).stem
    if args.train:
        env.update(knobs)
        env["DRY_RUN"] = "0"
        # 5. reconcile: a complete export from ANY earlier attempt is kept, never retrained; verified with the frozen knobs
        found = []
        for k, folder in earlier_runs(work, run_key(row), attempt):
            detail = export_detail(folder / ("hf-step%s" % steps))
            found.append({"attempt": k, "run": "runs/%s" % folder.name, "export": detail,
                          "run_summary": (folder / "run-summary.json").is_file()})
        record.update(reconciled=found)
        kept = next((f for f in found if f["export"]["complete"]), None)
        if kept is not None:
            summary = pb._json_file(work / kept["run"] / "run-summary.json")
            ok, wrong = pb.verify_settings(summary, {**env, "NAME": Path(kept["run"]).name})
            record.update(**{"state": "reconciled", "class": "reconciled", "kept_attempt": kept["attempt"], "export_complete": 1,
                             "settings_verified": ok, "settings": wrong, "settings_recorded": int(ok is not None),
                             "settings_mismatch_count": len(wrong) if ok is False else 0, "ok": int(ok is not False),
                             "status": 0, "ended_at": now_text(), "elapsed_seconds": 0.0,
                             "reason": "existing export of attempt %d kept: a complete export is never retrained%s"
                                       % (kept["attempt"], "" if ok else "; its settings are %s" % ("not recorded (the report "
                                          "marks the run not evaluated)" if ok is None else "wrong: " + "; ".join(wrong)))})
            print("existing export of attempt %d kept (runs/%s-a%d/hf-step%s): not trained again"
                  % (kept["attempt"], row, kept["attempt"], steps))
            return 0 if ok is not False else EXIT_SETTINGS
        record.update(launcher_env={**knobs, "MODEL_DIR": env.get("MODEL_DIR")})

    # 6. the gate
    result = pb.decide(work, Path(args.campaign), Path(args.reservation), block, row, workers_verified=True, attempt=attempt,
                       baseline=baseline_path)
    for check in result["inequalities"]:
        print(check["printed"])
    record.update(gate={k: result.get(k) for k in ("exit", "reason", "limit_seconds", "ledger_seconds", "limit_capped", "retry", "gpus")})
    if result["exit"] != pb.EXIT_OK:
        cls = "budget" if result["exit"] == pb.EXIT_NOT_ALLOWED else "refused"
        record.update(**{"state": "refused", "class": cls, "reason": result["reason"], "ended_at": now_text(), "elapsed_seconds": 0.0,
                         "status": None, "stop_written": result.get("stop_written")})
        print(result["reason"], file=sys.stderr)
        return result["exit"]
    limit = int(result["limit_seconds"])

    # 7. the launch (round-5 ruling H1; round-6 rulings J1, J2, J4): the job boundary must be observable; the durable
    # times and the job marker first; then the job's process group, PARKED (start_contained: the command waits in the
    # shim); then the independent watchdog, GIVEN that group; only then does the command run
    scan_ok, scan_how = wd.environment_scan_works()
    if not scan_ok and containment != "slurm-step":
        return refuse("refused", "the job boundary cannot be observed on this machine (no way to read process environments: %s), "
                                 "so nothing is launched (round-5 ruling H1)" % scan_how)
    launched = datetime.now(timezone.utc).replace(microsecond=0)
    allowance = deadline_grace()
    limit_until = launched + timedelta(seconds=limit)
    deadline = limit_until + timedelta(seconds=allowance)
    kill_by = deadline - timedelta(seconds=kill_margin(allowance))
    grace = kill_grace()
    marker = "%s-a%d-%s" % (row, attempt, secrets.token_hex(8))
    watch_path = watchdog_record_path(work, row, attempt)
    record.update(state="launching", launched=False, limit_seconds=limit, launched_at=now_text(launched), limit_until=now_text(limit_until),
                  deadline=now_text(deadline), kill_by=wd.precise_text(kill_by), deadline_grace_seconds=allowance, kill_grace_seconds=grace,
                  idle_wait_seconds=args.idle_wait, gpus=devices or "all", job_marker=marker, job_scan=scan_how,
                  watchdog={"record": str(watch_path.relative_to(work)) if watch_path.is_relative_to(work) else str(watch_path)})
    env[wd.MARKER_VAR] = marker
    try:
        job = start_contained(command, env, limit, record)
    except (OSError, ContainmentError) as error:
        return refuse("refused", "the job could not be started under containment %s, so nothing is launched: %s" % (containment, error))
    if containment == 'slurm-step':
        devices = contain.scheduler_devices(record.data)
    watcher, why = start_watchdog(record.path, watch_path, job.pgid)
    if watcher is None:
        job.abort()                                                     # the command never ran
        return refuse("refused", "the run-time limit's watchdog could not be started, so nothing is launched (round-4 ruling G1): %s" % why)
    record.update(watchdog={**record.data["watchdog"], "pid": watcher})
    print("runtime limit %d seconds, until %s; shutdown from then, SIGKILL by %s, verified by the deadline %s (+%d s); enforced by "
          "watchdog %d too; job marker %s=%s; containment %s; %s" % (limit, now_text(limit_until), wd.precise_text(kill_by), now_text(deadline),
                                                                     allowance, watcher, wd.MARKER_VAR, marker, containment, " ".join(command)), flush=True)
    limit_mono = wd.mono_of(limit_until.timestamp())                   # J1: the limit in monotonic time, converted once
    clock = time.monotonic()
    old = {s: signal.signal(s, _raise) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    ended_by = "exit"
    try:
        record.update(state="running", launched=True)
        job.release()                                                   # the group is on record and watched: the command runs
        try:
            job.wait(timeout=max(0.0, limit_mono - time.monotonic()))
        except subprocess.TimeoutExpired:
            ended_by = "limit"
    except WrapperSignal as caught:
        ended_by = "signal %s" % caught
    finally:
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):          # from here on the cleanup must finish
            signal.signal(s, signal.SIG_IGN)
    scheduler_end = None
    stopped_at = time.time()
    if ended_by == "exit" and stopped_at >= limit_until.timestamp():
        ended_by = "limit"                                              # it ran until its admitted time had ended
    pgid = job.pgid
    # J1: the shutdown starts NOW, on its fixed schedule; its TERM to the group is the first action, before any record
    kill_at = min(stopped_at + grace, kill_by.timestamp())
    verify_by = deadline.timestamp() if ended_by == "limit" else min(deadline.timestamp(), kill_at + max(args.idle_wait, MIN_VERIFY))
    ending = wd.JobEnd(marker, pgid, devices, kill_at=kill_at, verify_by=max(verify_by, kill_at), window_start=job.window_start, scheduler=record.data.get('slurm')).start()
    ending.termed.wait(5)
    if containment == 'slurm-step':
        s = record.data['slurm']
        observation = contain.command(['scontrol', 'show', 'step', contain.step_ref(s['job_id'], s['step_id'])], timeout=5)
        if contain.values(observation['stdout']).get('State') == 'TIMEOUT':
            ended_by = 'limit'
        record.update(slurm={**s, 'cancel_at_limit': ended_by == 'limit', 'end_observation': observation,
                             'cancel_submitted_at': ending.out.get('scheduler_cancel_submitted_at')})
    ran = round(time.monotonic() - clock, 1)
    limit_reason = None
    def mark_limit():
        nonlocal limit_reason
        # H2 (1): the budget disposition and the block's stop are durable before any terminal state (after the TERM: J1)
        archived = record.data.get('slurm') or {}
        scheduler_timeout = (archived.get('termination') or {}).get('state') == 'TIMEOUT' or contain.values((archived.get('end_observation') or {}).get('stdout', '')).get('State') == 'TIMEOUT'
        if scheduler_timeout:
            limit_reason = '%s: row %s attempt %d reached its scheduler TIMEOUT (%s whole minutes); admitted limit_until %s is unchanged' % (
                pb.BUDGET_INCOMPLETE, row, attempt, archived.get('effective_minutes'), now_text(limit_until))
        else:
            limit_reason = "%s: row %s attempt %d was still running when its admitted run time (%d s) ended at %s" % (
                pb.BUDGET_INCOMPLETE, row, attempt, limit, now_text(limit_until))
        record.update(**{"ended_by": "limit", "class": "budget", "ok": 0, "reason": limit_reason,
                         "limit_reached_at": wd.precise_text(wd.from_epoch(stopped_at))})
        test_pause(record.path, "after-budget-record")
        pb.record_stop(work, block, {"time": now_text(), "row": row, "attempt": attempt, "reason": limit_reason,
                                     "cause": "spending-limit termination", "limit_seconds": limit, "limit_until": now_text(limit_until)})
        test_pause(record.path, "after-stop")
    if ended_by == "limit":
        mark_limit()
    if containment == 'slurm-step':
        s = record.data['slurm']
        scheduler_end = contain.evidence(s['job_id'], s['step_id'], devices, deadline)
        record.update(slurm={**s, 'termination': scheduler_end})
        if scheduler_end.get('state') == 'TIMEOUT' and ended_by != 'limit':
            ended_by = 'limit'
            record.update(slurm={**record.data['slurm'], 'cancel_at_limit': True})
            mark_limit()  # accounting may be the first durable TIMEOUT observation
    ended = ending.wait(reap=job.poll)
    if job.poll() is None:
        try:
            job.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    status = job.returncode
    if ended_by == "limit":
        status = 137 if ended["kill"] else 124
    verified_at = wd.from_epoch(ended["verified_at"]) if ended["verified"] else None
    if scheduler_end is not None:
        if not scheduler_end.get('verified'):
            verified_at = None
        elif verified_at is not None:
            verified_at = max(verified_at, wd.parse_time(scheduler_end['verified_at']))
    elapsed = round(time.monotonic() - clock, 1)
    job_ended = job_summary(ended)
    fields = {"status": status, "ended_by": ended_by, "launcher_seconds": ran, "job_ended": job_ended,
              "group_ended": {"pgid": pgid, "term": ended["term"], "kill": ended["kill"], "gone": not ended["last"].get("group")},
              "idle_after": "the job verified empty at %s" % job_ended["verified_at"] if verified_at else wd._describe(ended["last"]),
              "elapsed_seconds": elapsed}
    for s, handler in old.items():
        signal.signal(s, handler)

    def acknowledged(stand_down: bool) -> None:
        """J2: (with `stand_down`, after writing it) wait for the watchdog's durable acknowledgement before returning. Both
        keep the same schedule (J1): a TERM or KILL its watchdog delivered first counts for the job (status 137 when a
        SIGKILL was needed, from either)."""
        if stand_down:
            record.update(**{STAND_DOWN: now_text()})
        watch = await_watchdog(watch_path, watcher)
        update = {"watchdog": {**record.data["watchdog"], **watch}}
        if watch.get("state") == "acted":
            group = dict(record.data.get("group_ended") or {})
            group.update({"term": bool(group.get("term") or watch.get("term")), "kill": bool(group.get("kill") or watch.get("kill")),
                          "by_watchdog": {"term": watch.get("term"), "kill": watch.get("kill")}})
            update["group_ended"] = group
            if ended_by == "limit" and watch.get("kill") and record.data.get("status") == 124:
                update["status"] = 137
        record.update(**update)

    if verified_at is None:
        # H1: the job (its group, its marker, its GPUs) could not be shown empty by the deadline: the block's accounting is
        # unavailable and its durable stop is written; the watchdog is NOT stood down (it records its own result)
        reason = "%s: row %s attempt %d: its job could not be shown terminated (%s)" % (NOT_VERIFIED, row, attempt, wd._describe(ended["last"]))
        record.update(**fields, **{"state": "unverified", "class": NOT_VERIFIED, "reason": reason, "ok": 0, "accounting": COMPLIANCE_UNAVAILABLE})
        pb.record_stop(work, block, {"time": now_text(), "row": row, "attempt": attempt, "cause": NOT_VERIFIED,
                                     "accounting": "unavailable", "reason": "%s: %s" % (COMPLIANCE_UNAVAILABLE, reason)})
        if record.data.get('containment')=='slurm-step':contain.hard_stop(work,reason,record.path)
        acknowledged(stand_down=False)
        print(reason, file=sys.stderr)
        return EXIT_UNVERIFIED
    test_pause(record.path, "after-termination")
    past_limit = verified_at >= limit_until
    fields.update({"state": "ended", "verified_idle_at": ceil_text(verified_at), "charged_until": ceil_text(verified_at),
                   "ended_at": ceil_text(verified_at), "ended_precise": wd.precise_text(verified_at), "termination_source": "wrapper",
                   "past_limit": past_limit, "past_deadline": verified_at > deadline,
                   "terminated_by_deadline": verified_at <= deadline})
    if scheduler_end is not None and (scheduler_end.get('failure_type') or scheduler_end.get('state') == 'PREEMPTED'):
        failure = scheduler_end.get('failure_type') or 'preempted'
        reason = failure+': owning Slurm allocation failed; no successful row outcome'
        record.update(**fields, **{'class': failure, 'failure_type': failure, 'ok': 0, 'status': 2, 'reason': reason})
        cls=contain.stop_class(failure,True,not past_limit)
        record.update(stop_class=cls,allocation_ended=cls=='allocation_preempted')
        if cls=='hard':
            pb.record_stop(work, block, {'time': now_text(), 'row': row, 'attempt': attempt, 'cause': failure, 'reason': reason})
            contain.hard_stop(work,reason,record.path)
        acknowledged(stand_down=True)
        print(reason, file=sys.stderr)
        return 2
    if ended_by == "limit" or past_limit:
        if record.data.get('containment')=='slurm-step':contain.hard_stop(work,'step past its cap',record.path)
        if ended_by != "limit":
            # verified only at or after limit_until: a spending-limit termination too (H2 (1) before (2))
            limit_reason = "%s: row %s attempt %d ended (%s) but its job was verified gone only at %s, at or after the end of its admitted " \
                           "run time %s" % (pb.BUDGET_INCOMPLETE, row, attempt, ended_by, wd.precise_text(verified_at), now_text(limit_until))
            record.update(**{"class": "budget", "ok": 0, "reason": limit_reason, "budget_evidence": "ended at or after limit_until"})
            pb.record_stop(work, block, {"time": now_text(), "row": row, "attempt": attempt, "reason": limit_reason,
                                         "cause": "spending-limit termination", "limit_seconds": limit, "limit_until": now_text(limit_until)})
        record.update(**fields)                                       # (2) the terminal state
        test_pause(record.path, "after-ended")
        acknowledged(stand_down=True)                                 # (3) the watchdog stands down, and acknowledges (J2)
        print(limit_reason, file=sys.stderr)
        if ended_by.startswith("signal"):
            return EXIT_SIGNAL
        return record.data.get("status", status) if ended_by == "limit" else 124
    if ended_by.startswith("signal"):
        reason = "interrupted: the wrapper received %s; its job was ended and verified gone" % ended_by[len("signal "):]
        record.update(**fields, **{"class": "interrupted", "reason": reason})
        test_pause(record.path, "after-ended")
        acknowledged(stand_down=True)
        print(reason, file=sys.stderr)
        return EXIT_SIGNAL
    record.update(**fields)
    test_pause(record.path, "after-ended")
    acknowledged(stand_down=True)                                     # the watchdog stands down: the wrapper verified the end

    # 8. classify
    if not args.train:
        record.update(**fields, **{"class": "complete" if status == 0 else pb.UNCLASSIFIED, "ok": int(status == 0)})
        return status
    name = env.get("NAME", "")
    run = work / "runs" / name
    summary = pb._json_file(run / "run-summary.json")
    detail = export_detail(run / ("hf-step%s" % steps))
    ok, wrong = pb.verify_settings(summary, env)
    merged = isinstance(summary, dict) and summary.get("merged") == 1 and summary.get("returncode") == 0
    if ok is False:
        cls, code = "settings", EXIT_SETTINGS
    elif status == 0 and detail["complete"] and merged and ok:
        cls, code = "valid", 0
    elif status == 4:
        cls, code = "invalid export (launcher exit 4: trained, no complete merged model)", 4
    elif status == 0:
        cls, code = "invalid export (exit 0 without a complete export)", 4
    else:
        cls, code = "%s (exit %d)" % (pb.UNCLASSIFIED, status), status
    fields.update({"class": cls, "export": detail, "export_complete": int(detail["complete"]), "settings_verified": ok,
                   "settings": wrong, "settings_recorded": int(ok is not None), "ok": int(cls == "valid"),
                   "settings_mismatch_count": len(wrong) if ok is False else 0,
                   "failure_evidence": None if cls == "valid" else {"console_tail": tail(run / "console.log"),
                                                                   "merge_log_tail": tail(run / "merge.log", 15)}})
    record.update(**fields)
    if cls != "valid":
        history = [h for h in pb.attempt_history(work, campaign_name, row) if h["attempt"] == attempt]
        if history:
            history[0]["record"] = record.data
            record.update(retry_eligibility=pb.eligibility(work, history[0], workers_verified=True, baseline=baseline_path))
    return code


if __name__ == "__main__":
    sys.exit(main())
