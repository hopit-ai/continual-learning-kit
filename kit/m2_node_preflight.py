#!/usr/bin/env python3
"""First fail-closed node job: durable locks, local NVMe and assigned GPUs."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
import subprocess
from kit.m2_common import entry,atomic
from kit import m2_assignment as assignment,m2_snapshot as snapshots


DATASET_IMPORT_CHECK = 'import numpy, pyarrow, datasets; import verl.utils.dataset.rl_dataset'


def check_dataset_imports(python):
    """Import the trainer dataset stack without imposing upgrades on working nodes."""
    repair = f"{python} -m pip install 'pyarrow==21.0.0' 'datasets==4.8.5'"
    try:
        result = subprocess.run([str(python), '-c', DATASET_IMPORT_CHECK],
                                capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'PF-node trainer dataset import check failed: {error}; '
                           f'for a fresh environment, after SDPO install run: {repair}') from error
    if result.returncode:
        raise RuntimeError(f'PF-node trainer dataset import check failed (exit {result.returncode}): '
                           f'{result.stderr.strip()}; for a fresh environment, after SDPO install run: {repair}')


def probe(local_root,durable,node,standin):
    authority=assignment.load(durable/'scoring-assignment.json')
    launch=__import__('json').loads((durable/'launch.json').read_text())
    assignment.require(node in launch['nodes'],'preflight node absent from frozen launch')
    if standin:
        return dict(status='PASS',node=node,technical_synthetic=True,assignment_sha256=assignment.sha(durable/'scoring-assignment.json'),
            filesystem=dict(functional_flock='SYNTHETIC'),nvme=dict(all_transports_pcie='SYNTHETIC'),gpus=[])
    assignment.require(assignment.platform.node()==node,'preflight hostname differs')
    locks=snapshots.flock_preflight(durable,nodes=len(launch['nodes']))
    from kit.next_blend import local_nvme
    nvme=local_nvme(local_root,False)
    from kit.m2_gpu import devices,prepare,assert_current
    observed=[];allocated=os.environ.get('CUDA_VISIBLE_DEVICES');old_uuid=os.environ.get('M2_ASSIGNED_GPU_UUID')
    try:
        inventory=devices()
        # Check all GPUs the allocation exposes, including train-only GPUs.
        for index,uuid in inventory.items():
            if allocated is not None and index not in allocated.split(',') and uuid not in allocated.split(','):continue
            if allocated is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=allocated
            mapping=prepare(index,uuid);assert_current()
            hardware=assignment.inventory(mapping['index'])
            for expected in authority['assignments'].values():
                if expected['node']==node and expected['uuid']==uuid:assignment.require(hardware==expected,'node preflight assignment GPU/environment differs')
            observed.append(dict(index=index,**hardware,uuid_asserted=True))
        assignment.require(observed,'no allocated GPUs')
        wanted={r['uuid'] for r in authority['assignments'].values() if r['node']==node}
        assignment.require(wanted<={r['uuid'] for r in observed},'assigned UUID absent from node allocation')
    finally:
        for name,value in (('CUDA_VISIBLE_DEVICES',allocated),('M2_ASSIGNED_GPU_UUID',old_uuid)):
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
    return dict(status='PASS',node=node,technical_synthetic=False,assignment_sha256=assignment.sha(durable/'scoring-assignment.json'),
        filesystem=locks,nvme=nvme,gpus=observed)


def job(a):
    try:
        if not a.standin:
            from kit.m2_preflight import check_runtime
            for python in (a.trainer_python,a.inference_python):check_runtime(python)
            check_dataset_imports(a.trainer_python)
        receipt=probe(a.local_root,a.durable,a.node,a.standin)
        receipt['trainer_dataset_imports'] = 'SYNTHETIC' if a.standin else 'PASS'
    except Exception as error:
        atomic(a.out/'preflight/node.json',dict(status='PENDING',node=a.node,error=type(error).__name__+': '+str(error)))
        raise
    atomic(a.out/'preflight/node.json',receipt)


def configure(p):
    p.add_argument('--trainer-python',default=sys.executable);p.add_argument('--inference-python',default=sys.executable)
    p.add_argument('--durable',type=Path,required=True);p.add_argument('--local-root',type=Path,required=True);p.add_argument('--node',required=True)
if __name__=='__main__':raise SystemExit(entry(job,configure,'node-preflight'))
