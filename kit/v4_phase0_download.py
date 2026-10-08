#!/usr/bin/env python3
"""Pinned full-repository HF staging on a zero-GPU compute batch host, stdlib entry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.v4_phase0_site import storage,free_space
from kit.runner import bounded_command,write_durably
MODELS={'initial':('Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218'),
        'teacher':('Qwen/Qwen3.6-27B','6a9e13bd6fc8f0983b9b99948120bc37f49c13e9')}
DOWNLOAD_GIB=100
RECEIPT='v4/report-phase0/download/receipt.json'


def snapshot_path(hf_home,repo,revision):return Path(hf_home)/'hub'/('models--'+repo.replace('/','--'))/'snapshots'/revision


def inventory(path,hf_home):
    records={}
    for file in sorted(Path(path).rglob('*')):
        if not file.is_file():continue
        if not file.resolve().is_relative_to(Path(hf_home).resolve()):raise ValueError('download snapshot symlink escapes owned HF cache')
        records[str(file.relative_to(path))]={'bytes':file.stat().st_size}
        if file.suffix=='.json':records[str(file.relative_to(path))]['sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    if not records or 'config.json' not in records:raise ValueError('download snapshot is incomplete: config.json absent')
    index=Path(path)/'model.safetensors.index.json'
    shards=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if not shards or any(not records.get(name,{}).get('bytes') for name in shards):raise ValueError('download snapshot weight shards absent or empty')
    return records


def make_receipt(work,hf_home,paths,slurm):
    hf_home=Path(hf_home).resolve();work=Path(work).resolve()
    if hf_home==work or hf_home.is_relative_to(work):raise ValueError('download HF_HOME must be outside WORK')
    if slurm.get('gpus')!=0 or not str(slurm.get('job_id','')).isdecimal():raise ValueError('download requires a scheduler-verified zero-GPU batch job')
    models={}
    for role,(repo,revision) in MODELS.items():
        path=Path(paths[role]).resolve()
        if path!=snapshot_path(hf_home,repo,revision):raise ValueError('download revision directory differs: '+role)
        files=inventory(path,hf_home)
        models[role]={'repo':repo,'revision':revision,'path':str(path),'revision_dir':path.name,'files':files,'bytes':sum(r['bytes'] for r in files.values())}
    return {'schema':'v4-phase0-download.v1','hf_home':str(hf_home),'models':models,'slurm':slurm,'allocation_gpu_hours':0,'gpu_block_charge':0}


def verify_download_receipt(work,doc=None,paths=None):
    if doc is None:
        try:doc=json.loads((Path(work)/RECEIPT).read_text())
        except (OSError,ValueError) as exc:raise ValueError('compute-node download receipt missing/invalid; wait for the zero-GPU job, then prepare') from exc
    paths=paths or {'initial':Path(os.environ['MODEL_DIR']),'teacher':Path(os.environ['TEACHER_MODEL_DIR'])}
    observed=make_receipt(work,doc['hf_home'],paths,doc['slurm'])
    for key in ('schema','hf_home','models','slurm','allocation_gpu_hours','gpu_block_charge'):
        if observed[key]!=doc.get(key):raise ValueError('download receipt or snapshot bytes/sizes/revision changed: '+key)
    return doc


def scheduler_job():
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdecimal() or os.environ.get('SLURM_STEP_ID','').isdecimal():raise ValueError('download must run directly in the zero-GPU batch job on a compute node')
    raw=subprocess.check_output(['scontrol','show','job',job],text=True,timeout=15)
    fields=dict(re.findall(r'(\w+)=([^\s]+)',raw))
    if fields.get('JobId')!=job or 'AllocTRES' not in fields:raise ValueError('download job identity/allocation not scheduler-verified')
    counts=[int(n) for n in re.findall(r'(?:^|,)gres/gpu(?:[:][^=,]+)?=(\d+)',fields['AllocTRES'])]
    if any(counts):raise ValueError('download job allocated GPUs; refuse model staging in a GPU allocation')
    return {'job_id':job,'gpus':0,'alloc_tres':fields['AllocTRES']}


def download_snapshot(repo,revision,hf_home,seconds,log):
    code='from huggingface_hub import snapshot_download; import sys; print(snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],cache_dir=sys.argv[3]))'
    result=bounded_command([sys.executable,'-c',code,repo,revision,str(Path(hf_home)/'hub')],timeout=seconds,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'0','HF_HUB_DOWNLOAD_TIMEOUT':'30','HF_HUB_ETAG_TIMEOUT':'30'},log=log)
    if result['returncode']:raise ValueError('compute HF download failed: '+repo+'; '+str(result.get('failure_type'))+'; see '+str(log))
    expected=snapshot_path(hf_home,repo,revision)
    lines=log.read_text(errors='replace').strip().splitlines()
    # Merged stderr can follow stdout: record the cross-check, never trust it.
    write_durably(log.with_suffix('.path-check.json'),{'snapshot_path':str(expected),
        'printed_last_line_matches':bool(lines and lines[-1]==str(expected))})
    return expected


def download(work,hf_home):
    work=Path(work).resolve();hf_home=Path(hf_home).resolve();folder=work/'v4/report-phase0/download';folder.mkdir(parents=True,exist_ok=True)
    if (work/RECEIPT).exists():raise ValueError('download receipt already exists; preserve it; do not overwrite this WORK')
    clock=time.monotonic()
    try:
        slurm=scheduler_job();free_space(hf_home,DOWNLOAD_GIB)
        with storage(work,task_root=hf_home.parent):
            os.environ.update(HF_HOME=str(hf_home),HF_HUB_CACHE=str(hf_home/'hub'),HF_ASSETS_CACHE=str(hf_home/'assets'),
                HUGGINGFACE_HUB_CACHE=str(hf_home/'hub'),TRANSFORMERS_CACHE=str(hf_home/'hub'))
            paths={}
            for role,(repo,revision) in MODELS.items():
                left=7000-(time.monotonic()-clock)
                if left<=0:raise ValueError('download job deadline reached before '+role)
                paths[role]=download_snapshot(repo,revision,hf_home,left,folder/(role+'.log'))
            receipt=make_receipt(work,hf_home,paths,slurm);receipt['seconds']=time.monotonic()-clock
            write_durably(work/RECEIPT,receipt)
        return receipt
    except (ValueError,OSError,subprocess.SubprocessError) as exc:
        write_durably(folder/'failure.json',{'ok':False,'message':str(exc),'seconds':time.monotonic()-clock})
        raise


def download_script(work,task_root,kit,account,partition,qos,python):
    for value in (account,partition,qos):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',value):raise ValueError('invalid download site assignment')
    q=lambda p:shlex.quote(str(Path(p).resolve()))
    python_q=shlex.quote(str(Path(python).absolute()))  # Preserve the venv executable symlink and its pyvenv.cfg.
    return f'''#!/bin/bash
#SBATCH --no-requeue
#SBATCH --gpus=0
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
set -euo pipefail
export WORK={q(work)} TASK_ROOT={q(task_root)} KIT={q(kit)}
export PYTHONPATH="$KIT/.."
# Host-venv batch process, no enclosing srun/container.
CACHE_EXPORTS="$({python_q} "$KIT/v4_phase0_site.py" storage --work "$WORK" --task-root "$TASK_ROOT")"
eval "$CACHE_EXPORTS"
{python_q} "$KIT/v4_phase0_download.py" download --work "$WORK" --hf-home "$HF_HOME"
'''


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('download','script','verify'))
    p.add_argument('--work',type=Path,required=True);p.add_argument('--hf-home',type=Path);p.add_argument('--task-root',type=Path)
    p.add_argument('--python',type=Path);p.add_argument('--out',type=Path)
    for key in ('account','partition','qos'):p.add_argument('--'+key)
    a=p.parse_args(argv)
    try:
        if a.operation=='script':
            if not all((a.task_root,a.python,a.out,a.account,a.partition,a.qos)):raise ValueError('script needs task-root, python, out and your own account/partition/QOS')
            a.out.parent.mkdir(parents=True,exist_ok=True)
            with a.out.open('x') as f:f.write(download_script(a.work,a.task_root,Path(__file__).resolve().parent,a.account,a.partition,a.qos,a.python))
            print(a.out)
        elif a.operation=='download':
            if not a.hf_home:raise ValueError('download requires --hf-home outside WORK')
            print(json.dumps(download(a.work,a.hf_home),indent=2))
        else:print(json.dumps(verify_download_receipt(a.work),indent=2))
    except (ValueError,OSError,KeyError,subprocess.SubprocessError) as exc:print('STOP: '+str(exc));return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
