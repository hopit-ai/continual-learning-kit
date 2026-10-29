"""Container-safe progress and record writer shared by smoke launchers."""
import json
import time


class Steps:
    """Runs one step at a time and PRINTS PROGRESS AS IT HAPPENS, so `modal app logs <app>` (or
    scripts/modal_progress.py) shows which step is running, its trainer steps, and an ETA. Every line this class
    prints for that purpose starts with `[progress]`; the trainer's own per-step lines are passed through as they
    arrive. Before 29 September the output was held until each step ended, and a 20-step run was invisible."""

    PASS_THROUGH = ("step:", "training/global_step", "Saving model", " correct (", "done: ", "wrote ")

    def __init__(self, env):
        import os                                                            # noqa: PLC0415
        self.env = {**os.environ, **env}; self.record = []; self.clock = time.monotonic()

    def _say(self, text):
        print("[progress] t+%ds %s" % (time.monotonic() - self.clock, text), flush=True)

    def plan(self, names):
        """Declare the steps this function will run, in order, so progress can be read as n of N with an ETA."""
        self._say("PLAN %d %s" % (len(names), ",".join(names)))

    def run(self, name, argv, extra=None, cwd=None, keep=12, background=False):
        import collections, subprocess                                       # noqa: E401,PLC0415
        env = {**self.env, **(extra or {})}
        if background:
            return subprocess.Popen(argv, env=env, cwd=cwd, stdout=open("/tmp/%s.log" % name, "w"), stderr=subprocess.STDOUT)
        started = time.monotonic()
        self._say("START %s" % name)
        proc = subprocess.Popen(argv, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail, errors = collections.deque(maxlen=max(keep, 1)), collections.deque(maxlen=8)
        for line in proc.stdout:
            line = line.rstrip("\n"); tail.append(line[:400])
            if "Error" in line or "Traceback" in line: errors.append(line[:400])
            if any(k in line for k in self.PASS_THROUGH): print("[progress]   %s | %s" % (name, line[:240]), flush=True)
        rc = proc.wait(); seconds = round(time.monotonic() - started, 1)
        self.record.append({"name": name, "returncode": rc, "seconds": seconds,
                            "tail": list(tail), "errors": list(errors) if rc else []})
        self._say("END %s rc=%d %.0fs" % (name, rc, seconds))
        return rc == 0

    def done(self, **extra):
        record = {"steps": self.record, "seconds_in_function": round(time.monotonic() - self.clock, 1), **extra}
        # The whole record as ONE log line. `modal run --detach` returns the record through the local client; when that
        # client loses its connection the function still finishes, and without this line its result would be gone
        # (budget-audit smoke, attempt 1). Recover with: modal app logs <app-id> | grep "^\[record\] "
        print("[record] " + json.dumps(record, sort_keys=True, default=str), flush=True)
        return record

