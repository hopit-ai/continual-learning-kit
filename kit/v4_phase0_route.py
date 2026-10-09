#!/usr/bin/env python3
"""Zero-GPU route diagnosis only; never certifies an unknown Pyxis containment route."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.v4_phase0_timing import scale
from kit.p4_watchdog import write_durably
QUESTION="Does `k8b_pilot_run.sbatch` launch its container with `srun --container-image`, or directly as the batch process, and can that same image run on the login node without an allocation?"


def probe_route(work,pilot_wrapper):
    """Inspect the original launch lines without executing the pilot or creating any step."""
    raw=Path(pilot_wrapper).read_bytes()
    text='\n'.join(line for line in raw.decode().splitlines() if not line.lstrip().startswith('#')).replace('\\\n',' ')
    if re.search(r'\bsrun\b[^\n]*--container-image',text):route='pyxis_step'
    elif re.search(r'\b(?:apptainer|singularity)\s+exec\b|\benroot\s+start\b|\bdocker\s+run\b',text):route='batch_process'
    elif re.search(r'\b(?:source|\.)\s+[^\n]*activate\b',text) and not re.search(r'\bsrun\b',text):route='host_venv'
    else:route='unknown'
    result={'schema':'v4-phase0-route-probe.v1','route':route,'pilot_wrapper_sha256':hashlib.sha256(raw).hexdigest(),
        'gpu_submission_allowed':False,'question':QUESTION,
        'message':('Pyxis step route detected: this package does not qualify nested container steps. Return this probe to the owner; no GPU job.' if route=='pyxis_step' else
                   'Batch-process route identified; separately confirm the same allocation-free CPU environment and mounts before using the runbook.' if route in ('batch_process','host_venv') else
                   'Route unknown: return this probe and answer the owner question before any GPU job.'),
        'cpu_environment_on_login_node':'must be confirmed separately; a node-only route is insufficient'}
    write_durably(Path(work)/'v4/report-phase0/route-probe.json',result)
    return result


def header(work,pilot_wrapper):
    """A one-minute zero-GPU sbatch executes only dependency-light wrapper inspection."""
    kit=Path(__file__).resolve().parent
    return '#!/bin/bash\n#SBATCH --no-requeue\n#SBATCH --time=00:'+str(scale(60)//60)+':00\n#SBATCH --gpus=0\nset -eu\npython3 '+shlex.quote(str(kit/'v4_phase0_route.py'))+' --work '+shlex.quote(str(Path(work).resolve()))+' --pilot-wrapper '+shlex.quote(str(Path(pilot_wrapper).resolve()))+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--pilot-wrapper',type=Path,required=True)
    p.add_argument('--write-probe',type=Path);a=p.parse_args(argv)
    if a.write_probe:
        (a.work/"v4/report-phase0").mkdir(parents=True,exist_ok=True)
        with a.write_probe.open('x') as handle:handle.write(header(a.work,a.pilot_wrapper))
    else:print(json.dumps(probe_route(a.work,a.pilot_wrapper),indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
