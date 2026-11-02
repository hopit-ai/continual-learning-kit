"""Translate frozen physical UUIDs to PCI-ordered CUDA indices before CUDA init."""
import os
import subprocess
import sys
import json
from kit.m2_assignment import require


def devices():
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    rows=[tuple(v.strip() for v in line.split(',')) for line in text.splitlines() if line.strip()]
    require(rows and all(len(row)==2 and row[0].isdigit() for row in rows),'invalid GPU index/UUID inventory')
    require(len({r[1] for r in rows})==len(rows),'duplicate GPU UUID')
    return dict(rows)


def gpu_index(uuid):
    rows=allocated_devices();matches=[index for index,value in rows.items() if value==uuid]
    require(len(matches)==1,'assigned GPU UUID not visible before staging')
    index=matches[0]
    allowed=os.environ.get('CUDA_VISIBLE_DEVICES')
    if allowed is not None:
        tokens=allowed.split(',')
        require(index in tokens or uuid in tokens,'assigned GPU outside allocation CUDA_VISIBLE_DEVICES')
    return index


def allocated_devices(rows=None):
    """CUDA tokens mapped to measured NVML UUIDs, including cgroup ordinals."""
    rows=devices() if rows is None else rows
    allowed=os.environ.get('CUDA_VISIBLE_DEVICES')
    if allowed is None:return rows
    tokens=allowed.split(',')
    scheduler=os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS')
    if scheduler:
        physical=scheduler.split(',')
        if len(physical)==len(tokens):
            mapped=[rows.get(v,v if v in rows.values() else None) for v in physical]
            if all(mapped):return dict(zip(tokens,mapped))
            # NVML may itself be restricted and renumbered by ConstrainDevices.
            if len(rows)==len(tokens):return dict(zip(tokens,rows.values()))
    result={token:rows.get(token,token if token in rows.values() else None) for token in tokens}
    if all(result.values()):return result
    if len(rows)==len(tokens) and all(token.isdigit() for token in tokens):
        return dict(zip(tokens,rows.values()))
    require(False,'allocated GPU UUIDs unavailable')


def prepare(gpu, expected_uuid=None):
    rows=allocated_devices()
    uuid=expected_uuid or (gpu if str(gpu).startswith('GPU-') else rows.get(str(gpu)))
    require(uuid is not None,'unknown requested GPU')
    index=gpu_index(uuid)
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=index,M2_ASSIGNED_GPU_UUID=uuid)
    return dict(uuid=uuid,index=index)


def normalized_uuid(value):
    if isinstance(value,bytes):value=value.decode()
    return str(value).lower().removeprefix('gpu-')


def current_uuid():
    """Query CUDA in a short child; vLLM's parent must remain CUDA-uninitialised."""
    code = "import json,torch; u=torch.cuda.get_device_properties(0).uuid; print(json.dumps(u.decode() if isinstance(u,bytes) else str(u)))"
    try:
        return json.loads(subprocess.check_output([sys.executable,'-c',code],text=True,timeout=300))
    except (subprocess.CalledProcessError,OSError,subprocess.TimeoutExpired,json.JSONDecodeError) as error:
        raise RuntimeError('retryable CUDA UUID subprocess failure: '+str(error)) from error


def assert_current():
    uuid=os.environ.get('M2_ASSIGNED_GPU_UUID')
    if not uuid:return
    require(normalized_uuid(current_uuid())==normalized_uuid(uuid),
            'in-process torch CUDA device UUID differs from assigned physical GPU')


def prepared_index(gpu,standin=False):
    """Leaf scorers preserve the physical assignment's prepared numeric index."""
    if standin:return str(gpu)
    return prepare(gpu,expected_uuid=os.environ.get('M2_ASSIGNED_GPU_UUID'))['index']
