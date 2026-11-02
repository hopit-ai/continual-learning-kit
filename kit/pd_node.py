"""PD job entry point: CPU prep -> 1-GPU pre -> 8-GPU train -> CPU HF
exports -> 1-GPU post -> CPU collect. GPU phases always use util_watch.py.
The legacy whole-seed/eight-GPU scoring entry point is retired.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.pd_contract import require


def _gpu_uuid(value):
    """CUDA accepts GPU-UUID selectors; Torch may report the UUID without GPU-."""
    require(isinstance(value, str), "GPU UUID must be a string")
    raw = value.removeprefix("GPU-")
    try:
        normalized = str(uuid.UUID(raw))
    except (ValueError, AttributeError):
        raise ValueError("whole-GPU UUID required; MIG allocations are not supported") from None
    return "GPU-"+normalized


def run_phase(phase, plan, output):
    from kit import pd_jobs as jobs
    require(phase in jobs.PHASES, "PENDING: explicit split-job phase required")
    if not jobs.GPUS[phase]:
        return jobs.worker(phase, plan, output)
    # Preconditions are cheap and fail before the watcher launches any GPU code.
    config, root, prepared = jobs.gate(plan, output)
    jobs.allocation_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"), jobs.GPUS[phase])
    jobs.prerequisites(phase, config, root, prepared)
    watcher = Path(__file__).with_name("util_watch.py")
    require(watcher.is_file(), "PENDING: required utilisation watcher is unavailable")
    child = [sys.executable, "-I", str(Path(jobs.__file__).resolve()), "--phase", phase,
        "--plan", str(Path(plan).resolve()), "--out", str(root)]
    command = child if os.environ.get('UTIL_WATCH_ACTIVE') else [sys.executable, "-I", str(watcher),
        "--out", str(root/(phase+"-util.jsonl")), "--", *child]
    return subprocess.run(command).returncode


def main():
    from kit import pd_jobs as jobs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=jobs.PHASES, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        result = run_phase(args.phase, args.plan, args.out)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps(dict(status="PENDING", reason=str(error))))
        return 3
    return result if type(result) is int else 0


if __name__ == "__main__":
    raise SystemExit(main())
