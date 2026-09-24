#!/usr/bin/env python3
"""Run a model's Python program against a problem's tests, locally, without root and without network.

This is the executor behind `kit/beds/code.py`. It exists because a coding bed cannot be scored by
comparing strings: the only way to know whether a program is right is to run it. Everything here is
standard library, runs as an ordinary unprivileged user inside the partner's container, and needs no
Docker socket, no cgroup write access and no cloud sandbox service.

WHAT IT DOES, per test:

  1. makes a fresh temporary directory and writes `solution.py` (the model's program) and
     `_runner.py` (this file's `RUNNER_SOURCE`) into it;
  2. writes the test's input to `input.txt` -- the expected OUTPUT is never written anywhere the
     candidate program could read it, and never enters the child process at all;
  3. starts one subprocess, in its own session, with a minimal environment, its stdout and stderr
     redirected to files in that directory, and (for a stdin test) `input.txt` as its real stdin,
     so `sys.stdin.buffer` works -- phase 1 lost a scoring run to a `StringIO` stand-in that had no
     `.buffer`, and this bed does not repeat that;
  4. waits at most `timeout_s` seconds of wall clock and then kills the whole process group;
  5. compares, in the PARENT, what came back with what the test expects;
  6. deletes the directory, whatever happened.

The child sets its own limits before it runs a line of the model's code: address space
(`RLIMIT_AS`), CPU seconds (`RLIMIT_CPU`, a second line of defence behind the wall clock), and
output file size (`RLIMIT_FSIZE`, which is what bounds a program that prints forever). It then
replaces `socket.socket` and its neighbours with functions that raise, and the parent hands it an
environment with every proxy variable removed and `no_proxy=*`.

WHAT IT IS NOT. It is not a security boundary against hostile code. The child is an ordinary
process with the caller's own user id and the caller's own filesystem: a program that wants to read
`~/.ssh`, write outside its temporary directory, spawn a subprocess that has no socket patch, or
call a socket through `ctypes`, can. The limits above stop the accidents a wrong program actually
produces -- infinite loops, runaway memory, endless output, an incidental `pip`/`urllib` call -- and
nothing more. Run it on model output, in a container you are willing to lose. Do not run it on code
from someone who means you harm.

TWO TEST FORMS, as LiveCodeBench defines them.

  stdin      the whole test input goes to the program's standard input and its standard output is
             compared with the expected output line by line, each line right-stripped and trailing
             blank lines dropped on both sides.
  functional the test input is one JSON value per line, the arguments; the program is imported (not
             run as `__main__`), `Solution().<fn_name>` is called with them -- or a module-level
             `<fn_name>` if there is no `Solution` -- and its return value is JSON round-tripped and
             compared with the JSON-decoded expected value. Tuples become lists in that round trip,
             as they do in the reference harness, and the reference's `expected == [result]`
             fallback is kept.

    from kit.sandbox import run_tests
    verdict = run_tests(code, tests, testtype="stdin", fn_name="")
    verdict["kind"]   # passed | wrong_answer | timeout | error | network | output_limit
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# One test's wall-clock budget. LiveCodeBench problems state a one-second limit for C++; the
# reference harness gives Python six. Ten is this bed's default: generous enough that a correct
# Python solution is not failed for being Python, tight enough that a rollout of forty tests cannot
# hold a GPU for an hour.
DEFAULT_TIMEOUT_S = 10.0
# Address space, not resident memory: `RLIMIT_AS` counts what is mapped. Phase 1 set it to 1 GiB and
# a thread-pooled BLAS pre-mapped enough that a correct program could not even serialise its answer,
# so the child's environment caps every numeric thread pool at one and the default here is 4 GiB.
DEFAULT_MEMORY_MIB = 4096
# What one test may print. A program that prints in an infinite loop hits this as SIGXFSZ rather
# than filling the disk; a program whose answer is genuinely large is not truncated below it.
DEFAULT_MAX_OUTPUT_MIB = 32
# Sentinels the child prints on stderr so the parent can tell one refusal from an ordinary crash.
NETWORK_SENTINEL = "kit-sandbox: network access is disabled"
MISSING_FUNCTION_SENTINEL = "kit-sandbox: the program defines no callable named"
UNSERIALISABLE_SENTINEL = "kit-sandbox: the returned value is not JSON"

KINDS = ("passed", "wrong_answer", "timeout", "error", "network", "output_limit")
TESTTYPES = ("stdin", "functional")

RESULT_NAME = "__kit_result.json"
STDOUT_NAME = "__kit_stdout.txt"
STDERR_NAME = "__kit_stderr.txt"


class SandboxError(Exception):
    """The sandbox itself could not run the test. Never turned into a score of zero."""


# --------------------------------------------------------------------------------- the child process
RUNNER_SOURCE = r'''"""Runs one candidate program under limits. Written into a temporary directory and deleted after."""
import json
import os
import runpy
import sys

NETWORK_SENTINEL = "kit-sandbox: network access is disabled"
MISSING_FUNCTION_SENTINEL = "kit-sandbox: the program defines no callable named"
UNSERIALISABLE_SENTINEL = "kit-sandbox: the returned value is not JSON"
RESULT_NAME = "__kit_result.json"


def set_limits():
    """Address space, CPU seconds and output size, before any of the candidate's code runs."""
    try:
        import resource
    except ImportError:                       # not a POSIX platform: the wall clock is all there is
        return False
    memory_bytes = int(os.environ.get("KIT_SANDBOX_MEMORY_BYTES", "0"))
    cpu_seconds = int(os.environ.get("KIT_SANDBOX_CPU_SECONDS", "0"))
    file_bytes = int(os.environ.get("KIT_SANDBOX_FILE_BYTES", "0"))
    for name, value in (("RLIMIT_AS", memory_bytes), ("RLIMIT_CPU", cpu_seconds),
                        ("RLIMIT_FSIZE", file_bytes)):
        limit = getattr(resource, name, None)
        if limit is None or value <= 0:
            continue
        try:
            soft, hard = resource.getrlimit(limit)
            ceiling = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(limit, (ceiling, hard))
        except (ValueError, OSError):         # a limit this platform will not accept is reported, not fatal
            sys.stderr.write("kit-sandbox: could not set %s\n" % name)
    return True


def block_network():
    """Make every socket the standard library would open raise instead. Not a firewall; see the module docstring."""
    import socket

    try:
        import ssl                            # noqa: F401
    except ImportError:                       # `ssl` subclasses socket.socket at import time, so it has
        pass                                  # to be built against the real class before the swap

    class DeniedSocket:                       # a class, not a function: `class SSLSocket(socket)` needs one
        def __init__(self, *args, **kwargs):
            raise OSError(NETWORK_SENTINEL)

    def deny(*args, **kwargs):
        raise OSError(NETWORK_SENTINEL)

    socket.socket = DeniedSocket
    for name in ("socketpair", "create_connection", "create_server",
                 "getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        if hasattr(socket, name):
            setattr(socket, name, deny)


def callable_named(namespace, fn_name):
    """`Solution().<fn_name>` when the program defines a Solution, else a module-level `<fn_name>`."""
    solution = namespace.get("Solution")
    if isinstance(solution, type) and callable(getattr(solution, fn_name, None)):
        return getattr(solution(), fn_name)
    target = namespace.get(fn_name)
    if callable(target):
        return target
    raise SystemExit("%s %s\n" % (MISSING_FUNCTION_SENTINEL, fn_name))


def main():
    mode = sys.argv[1]
    fn_name = sys.argv[2] if len(sys.argv) > 2 else ""
    set_limits()
    block_network()
    if mode == "stdin":
        runpy.run_path("solution.py", run_name="__main__")
        return 0
    with open("input.txt", encoding="utf-8") as handle:
        text = handle.read()
    arguments = [json.loads(line) for line in text.split("\n") if line.strip()]
    namespace = runpy.run_path("solution.py")          # imported, not run as __main__
    result = callable_named(namespace, fn_name)(*arguments)
    try:
        encoded = json.dumps(result)
    except (TypeError, ValueError):
        sys.stderr.write("%s\n" % UNSERIALISABLE_SENTINEL)
        return 1
    with open("%s.part" % RESULT_NAME, "w", encoding="utf-8") as handle:
        handle.write(encoded)
    os.replace("%s.part" % RESULT_NAME, RESULT_NAME)   # a half-written result is never read
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def child_environment() -> dict:
    """A minimal environment with every proxy variable removed and thread pools capped at one."""
    keep = {}
    for name in ("PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT", "TMPDIR"):
        if name in os.environ:
            keep[name] = os.environ[name]
    keep.setdefault("PATH", "/usr/bin:/bin")
    keep.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        # Unset by omission: HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, FTP_PROXY and their lower-case
        # spellings are simply not in this dictionary. `no_proxy` is then belt and braces.
        "no_proxy": "*", "NO_PROXY": "*",
    })
    return keep


# ------------------------------------------------------------------------------------- comparison
def output_lines(text: str) -> list:
    """The comparison form for a stdin test: each line right-stripped, trailing blank lines dropped."""
    lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def stdout_matches(observed: str, expected: str) -> bool:
    return output_lines(observed) == output_lines(expected)


def call_matches(result, expected_text: str) -> bool:
    """The reference's call-based comparison: equality, or the expected value wrapping the result."""
    try:
        expected = json.loads(expected_text)
    except (TypeError, ValueError) as exc:
        raise SandboxError("a functional test's expected value is not JSON: %s" % exc) from exc
    if result == expected:
        return True
    return isinstance(expected, list) and len(expected) == 1 and expected[0] == result


# ----------------------------------------------------------------------------------- running a test
def _classify_exit(returncode: int, stderr: str, stdout_bytes: int, max_output_bytes: int) -> str:
    """One finished process to one outcome kind. Signals first: they say why it stopped.

    The output limit arrives two ways. On a platform that delivers SIGXFSZ the process dies of the
    signal; where Python has SIGXFSZ ignored -- macOS does -- the write instead fails with EFBIG and
    the program dies of the exception, so the file's own size is the second witness.
    """
    if NETWORK_SENTINEL in stderr:
        return "network"
    if stdout_bytes >= max_output_bytes > 0 and returncode != 0:
        return "output_limit"
    if returncode < 0:
        stopped_by = -returncode
        if stopped_by == getattr(signal, "SIGXCPU", None):
            return "timeout"
        if stopped_by == getattr(signal, "SIGXFSZ", None):
            return "output_limit"
        return "error"
    return "error" if returncode != 0 else "passed"


def _read_capped(path: Path, limit: int):
    """(text, overflowed). Never reads more than `limit` bytes into memory."""
    if not path.is_file():
        return "", False
    size = path.stat().st_size
    with path.open("rb") as handle:
        data = handle.read(limit)
    return data.decode("utf-8", "replace"), size > limit


def _read_tail(path: Path, limit: int) -> str:
    """The LAST `limit` bytes. Stderr is read from the end because that is where the traceback is:
    a program that prints a megabyte of noise and then fails on a socket must still be seen to."""
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > limit:
            handle.seek(size - limit)
        return handle.read(limit).decode("utf-8", "replace")


def run_test(code: str, test, *, testtype: str, fn_name: str = "",
             timeout_s: float = DEFAULT_TIMEOUT_S, memory_mib: int = DEFAULT_MEMORY_MIB,
             max_output_mib: int = DEFAULT_MAX_OUTPUT_MIB, python=None) -> dict:
    """Run one test in a fresh temporary directory and delete it. Returns the outcome, never raises on bad code.

    `test` is `{"input": ..., "output": ...}`. `SandboxError` is raised only when the sandbox itself
    is at fault -- an unreadable request, a directory that cannot be made -- never because the
    candidate program misbehaved.
    """
    if testtype not in TESTTYPES:
        raise SandboxError("unknown test form %r: expected one of %s" % (testtype, ", ".join(TESTTYPES)))
    if not isinstance(code, str) or not code.strip():
        raise SandboxError("there is no program to run")
    if not isinstance(test, dict) or "input" not in test or "output" not in test:
        raise SandboxError("a test must be a mapping with an input and an output")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= 600:
        raise SandboxError("the time limit must be positive and no greater than 600 seconds")
    if testtype == "functional" and not fn_name:
        raise SandboxError("a functional test needs the name of the function to call")

    max_output_bytes = int(max_output_mib) * 1024 * 1024
    interpreter = python or sys.executable
    workdir = Path(tempfile.mkdtemp(prefix="kit-sandbox-"))
    started = time.monotonic()
    process = None
    try:
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "_runner.py").write_text(RUNNER_SOURCE, encoding="utf-8")
        (workdir / "input.txt").write_text(str(test["input"]), encoding="utf-8")
        stdout_path, stderr_path = workdir / STDOUT_NAME, workdir / STDERR_NAME
        environment = child_environment()
        environment.update({
            "HOME": str(workdir), "TMPDIR": str(workdir),
            "KIT_SANDBOX_MEMORY_BYTES": str(int(memory_mib) * 1024 * 1024),
            "KIT_SANDBOX_CPU_SECONDS": str(int(timeout_s) + 1),
            "KIT_SANDBOX_FILE_BYTES": str(max_output_bytes),
        })
        with open(workdir / "input.txt", "rb") as stdin_handle, \
                stdout_path.open("wb") as out_handle, stderr_path.open("wb") as err_handle:
            process = subprocess.Popen(
                # -I: isolated, so the caller's PYTHON* variables and user site-packages cannot change
                # what the candidate sees. -X utf8 because -I also discards PYTHONIOENCODING.
                [interpreter, "-I", "-B", "-X", "utf8", "_runner.py", testtype, fn_name],
                cwd=str(workdir), env=environment,
                stdin=stdin_handle if testtype == "stdin" else subprocess.DEVNULL,
                stdout=out_handle, stderr=err_handle,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout_s)
                timed_out = False
            except subprocess.TimeoutExpired:
                _kill_group(process)
                returncode, timed_out = process.returncode or -9, True
        elapsed = time.monotonic() - started
        stderr = _read_tail(stderr_path, 64 * 1024)
        if timed_out:
            return {"kind": "timeout", "elapsed_s": round(elapsed, 3), "detail": ""}
        stdout_bytes = stdout_path.stat().st_size if stdout_path.is_file() else 0
        kind = _classify_exit(returncode, stderr, stdout_bytes, max_output_bytes)
        if kind != "passed":
            return {"kind": kind, "elapsed_s": round(elapsed, 3), "detail": stderr[-2000:]}
        if testtype == "stdin":
            observed, overflowed = _read_capped(stdout_path, max_output_bytes)
            if overflowed:
                return {"kind": "output_limit", "elapsed_s": round(elapsed, 3), "detail": ""}
            correct = stdout_matches(observed, str(test["output"]))
        else:
            result_path = workdir / RESULT_NAME
            if not result_path.is_file():
                return {"kind": "error", "elapsed_s": round(elapsed, 3),
                        "detail": stderr[-2000:] or "the program returned no value"}
            encoded, overflowed = _read_capped(result_path, max_output_bytes)
            if overflowed:
                return {"kind": "output_limit", "elapsed_s": round(elapsed, 3), "detail": ""}
            correct = call_matches(json.loads(encoded), str(test["output"]))
        return {"kind": "passed" if correct else "wrong_answer", "elapsed_s": round(elapsed, 3), "detail": ""}
    except SandboxError:
        raise
    except OSError as exc:
        raise SandboxError("the sandbox could not run a test: %s" % exc) from exc
    finally:
        if process is not None and process.poll() is None:
            _kill_group(process)
        shutil.rmtree(workdir, ignore_errors=True)


def _kill_group(process) -> None:
    """Kill the child and anything it started. A wall-clock limit that leaves orphans is not a limit."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, AttributeError):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_tests(code: str, tests, *, testtype: str, fn_name: str = "",
              timeout_s: float = DEFAULT_TIMEOUT_S, memory_mib: int = DEFAULT_MEMORY_MIB,
              max_output_mib: int = DEFAULT_MAX_OUTPUT_MIB, python=None) -> dict:
    """Run every test until one fails. `kind` is `passed` only when all of them passed.

    Stopping at the first failure is what makes this affordable inside a training loop: the reward
    is one only if every test passes, so a program that fails test three is worth the same as one
    that fails test three and the thirty-seven after it.
    """
    if not tests:
        raise SandboxError("a problem with no tests cannot be scored")
    started = time.monotonic()
    for index, test in enumerate(tests):
        outcome = run_test(code, test, testtype=testtype, fn_name=fn_name, timeout_s=timeout_s,
                           memory_mib=memory_mib, max_output_mib=max_output_mib, python=python)
        if outcome["kind"] != "passed":
            return {"kind": outcome["kind"], "passed": index, "n": len(tests),
                    "first_failure_index": index, "detail": outcome["detail"],
                    "elapsed_s": round(time.monotonic() - started, 3)}
    return {"kind": "passed", "passed": len(tests), "n": len(tests), "first_failure_index": None,
            "detail": "", "elapsed_s": round(time.monotonic() - started, 3)}


def limits(timeout_s: float = DEFAULT_TIMEOUT_S, memory_mib: int = DEFAULT_MEMORY_MIB,
           max_output_mib: int = DEFAULT_MAX_OUTPUT_MIB) -> dict:
    """What this sandbox enforces, for a manifest to record. The honest version, not the hopeful one."""
    try:
        import resource                                                        # noqa: PLC0415, F401
        has_resource = True
    except ImportError:
        has_resource = False
    return {
        "per_test_wall_clock_s": timeout_s,
        "per_test_cpu_seconds": int(timeout_s) + 1 if has_resource else None,
        "address_space_mib": memory_mib if has_resource else None,
        # Asked for is not the same as enforced: Linux honours RLIMIT_AS, macOS commonly does not.
        # `python3 kit/sandbox.py` allocates past the limit on this machine and reports what happened.
        "address_space_limit_enforced": "run python3 kit/sandbox.py on this machine to find out",
        "output_bytes_per_test": max_output_mib * 1024 * 1024,
        "resource_limits_available": has_resource,
        "process_group_killed_on_timeout": hasattr(os, "killpg"),
        "network": "socket.socket and its neighbours raise; proxy variables are unset; no_proxy=*",
        "temporary_directory": "one per test, deleted afterwards",
        "expected_output_visible_to_the_program": False,
        "is_a_security_boundary": False,
        "not_prevented": ["reading and writing the caller's own files outside the temporary directory",
                          "spawning a subprocess, which does not inherit the socket patch",
                          "calling the operating system directly through ctypes"],
    }


def memory_limit_check(memory_mib: int = DEFAULT_MEMORY_MIB, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Ask for twice the address space the limit allows and report whether the kernel stopped it."""
    code = "x = bytearray(%d)\nprint(len(x))" % (int(memory_mib) * 2 * 1024 * 1024)
    verdict = run_tests(code, [{"input": "", "output": str(int(memory_mib) * 2 * 1024 * 1024)}],
                        testtype="stdin", timeout_s=timeout_s, memory_mib=memory_mib)
    return {"asked_for_mib": memory_mib * 2, "limit_mib": memory_mib, "outcome": verdict["kind"],
            "enforced": verdict["kind"] != "passed"}


def main(argv=None) -> int:
    """Print what the sandbox enforces on this machine, and run one trivial program end to end."""
    import argparse                                                            # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Report the code sandbox's limits on this machine.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--memory-mib", type=int, default=DEFAULT_MEMORY_MIB)
    args = parser.parse_args(argv)
    report = {"limits": limits(args.timeout, args.memory_mib),
              "self_check": run_tests("print(int(input()) + 1)", [{"input": "1\n", "output": "2\n"}],
                                      testtype="stdin", timeout_s=args.timeout,
                                      memory_mib=args.memory_mib),
              "memory_limit": memory_limit_check(args.memory_mib, args.timeout)}
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["self_check"]["kind"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
