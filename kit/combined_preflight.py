#!/usr/bin/env python3
"""First tier-node job: import stacks, readable cgroups, all GPU UUIDs. No training."""
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
    from kit import x_memory, m2_assignment, m2_gpu
    for python in (settings['trainer_python'],settings['inference_python']): check_runtime(python)
    check_dataset_imports(settings['trainer_python'])
    env=dict(os.environ,PYTHONPATH=str(combined.KIT.parent),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for python,imports in ((settings['trainer_python'],TRAINER_IMPORTS),(settings['inference_python'],INFERENCE_IMPORTS)):
        subprocess.run([python,'-c',imports],env=env,check=True,timeout=120,capture_output=True,text=True)
    memory=x_memory.sample(node=True)
    combined.need(memory['raw_current_bytes'] is not None and memory['raw_peak_bytes'] is not None and memory['oom_kill'] is not None,
                  'cgroup current/peak/OOM counters unavailable')
    # Use the inference interpreter for UUID/environment assertions, exactly as scoring does.
    code='''import json, os
from kit import m2_gpu, m2_assignment
visible=os.environ.get('CUDA_VISIBLE_DEVICES'); result=[]
for index,uuid in m2_gpu.devices().items():
 if visible is not None and index not in visible.split(',') and uuid not in visible.split(','): continue
 if visible is None: os.environ.pop('CUDA_VISIBLE_DEVICES',None)
 else: os.environ['CUDA_VISIBLE_DEVICES']=visible
 mapped=m2_gpu.prepare(index,uuid); m2_gpu.assert_current()
 result.append(m2_assignment.inventory(mapped['index']))
print(json.dumps(result))
'''
    result=subprocess.run([settings['inference_python'],'-c',code],env=env,check=True,timeout=600,capture_output=True,text=True)
    observed=json.loads(result.stdout.strip().split('\n')[-1])
    expected=combined.load(settings['tier_hardware'])[node]
    combined.need(expected.get('complete_inventory') is True, 'complete allocated tier GPU inventory required')
    combined.need(sorted(observed,key=lambda x:x['uuid'])==sorted(expected['gpus'],key=lambda x:x['uuid']), 'tier UUID/environment inventory differs')
    combined.need(all(combined.host(g['node'])==node for g in observed), 'tier hostname differs')
    return dict(status='PASS',technical_synthetic=False,trainer_imports='PASS',inference_imports='PASS',
                cgroup=memory,gpus=observed)


def main():
    p=argparse.ArgumentParser();p.add_argument('--settings-json',required=True);p.add_argument('--node',required=True);a=p.parse_args()
    settings=json.loads(a.settings_json)
    receipt=dict(settings_sha256=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest())
    try: receipt.update(check(settings,a.node)); receipt['hardware_sha256']=combined.sha(settings['tier_hardware']); code=0
    except Exception as exc: receipt.update(status='PENDING',technical_synthetic=False,error_type=type(exc).__name__);code=3
    combined.write(Path(settings['control_root'])/'preflight'/(a.node+'.json'),receipt)
    print(receipt['status'],'tier dependency/cgroup/UUID preflight');return code
if __name__=='__main__':raise SystemExit(main())
