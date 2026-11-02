#!/usr/bin/env python3
"""First tier-node job: import stacks, readable cgroups, CPU checks only. UUIDs are checked in productive GPU jobs."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import os
import subprocess
from kit import combined_send as combined

TRAINER_IMPORTS = 'import torch, transformers, numpy, pyarrow, datasets; import verl.utils.dataset.rl_dataset; import kit.m3_entry, kit.m3_runner, kit.pd_node'
INFERENCE_IMPORTS = 'import torch, vllm; import kit.m3_score, kit.sp_score, kit.sp_prepare'


def check(settings, node):
    from kit.m2_preflight import check_runtime
    from kit.m2_node_preflight import check_dataset_imports
    from kit import x_memory
    for python in (settings['trainer_python'],settings['inference_python']): check_runtime(python)
    check_dataset_imports(settings['trainer_python'])
    env=dict(os.environ,PYTHONPATH=str(combined.KIT.parent),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for python,imports in ((settings['trainer_python'],TRAINER_IMPORTS),(settings['inference_python'],INFERENCE_IMPORTS)):
        subprocess.run([python,'-c',imports],env=env,check=True,timeout=120,capture_output=True,text=True)
    memory=x_memory.sample(node=True)
    combined.need(memory['raw_current_bytes'] is not None and memory['raw_peak_bytes'] is not None and memory['oom_kill'] is not None,
                  'cgroup current/peak/OOM counters unavailable')
    return dict(status='PASS',technical_synthetic=False,trainer_imports='PASS',inference_imports='PASS',
                cgroup=memory,gpus=[], gpu_check='deferred to allocated productive job')


def main():
    p=argparse.ArgumentParser();p.add_argument('--settings-json',required=True);p.add_argument('--node',required=True);a=p.parse_args()
    settings=json.loads(a.settings_json)
    receipt=dict(settings_sha256=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest())
    try: receipt.update(check(settings,a.node)); receipt['hardware_sha256']=combined.sha(settings['tier_hardware']); code=0
    except Exception as exc: receipt.update(status='PENDING',technical_synthetic=False,error_type=type(exc).__name__);code=3
    combined.write(Path(settings['control_root'])/'preflight'/(a.node+'.json'),receipt)
    print(receipt['status'],'tier CPU dependency/cgroup preflight');return code
if __name__=='__main__':raise SystemExit(main())
