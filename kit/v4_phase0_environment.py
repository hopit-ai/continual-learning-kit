#!/usr/bin/env python3
"""Measured-deadline host-venv environment admission before phase-0 containment probes."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
if __package__ in (None, ''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.v4_phase0_timing import scale
from kit.runner import bounded_command, write_durably
from kit.v4_phase0_timing import relaxed_timeout


class AdmissionFailure(ValueError):
    pass

def cuda_probe(work,role,torch):
    """Small real CUDA operations in each admitted runtime, within the parent deadline."""
    doc={'role':role,'ok':False,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'devices':[]}
    path=Path(work)/('v4/report-phase0/environment-check-'+role+'-cuda.json')
    try:
        version,cuda=('2.9.0+cu128','12.8') if role=='trainer' else ('2.10.0+cu129','12.9')
        if (torch.__version__ not in ('2.9.0','2.9.0+cu128') if role=='trainer' else torch.__version__!=version) or torch.version.cuda!=cuda:raise AdmissionFailure(role+' torch/CUDA runtime pins differ')
        if not torch.cuda.is_available():raise ValueError(role+' CUDA unavailable')
        count=torch.cuda.device_count();doc['device_count']=count
        if count!=8:raise ValueError(role+' GPU count: expected allocated 8, found '+str(count))
        getter=getattr(getattr(torch,'_C',None),'_cuda_getDriverVersion',None)
        if getter:
            value=getter()
            if not isinstance(value,int) or value<=0:raise ValueError('invalid CUDA driver version')
            doc['driver']={'source':'torch._C._cuda_getDriverVersion','version':value}
        else:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=scale(600)).strip().splitlines()
            if len(lines)!=8 or len(set(lines))!=1 or not all(__import__('re').fullmatch(r'[0-9]+(?:\.[0-9]+)+',x) for x in lines):raise ValueError('cannot record a consistent driver version for eight GPUs')
            doc['driver']={'source':'nvidia-smi','version':lines[0]}
        for index in range(8):
            try:
                value=torch.ones(1,device='cuda:'+str(index));value.add_(1)
                torch.cuda.synchronize(index)
                if value.item()!=2:raise ValueError('one-element result differs')
            except Exception as exc:raise ValueError(role+' CUDA op failed on device '+str(index)+': '+str(exc)) from exc
            doc['devices'].append({'index':index,'one_element_result':2})
        doc['ok']=True
    except Exception as exc:
        doc['cause']=str(exc)
        if isinstance(exc,AdmissionFailure):raise
    finally:write_durably(path,doc)
    return doc


def probe(work, role):
    """Import prepared environments and check paths/devices without loading models."""
    work=Path(work)
    launch=json.loads((work/'v4/report-inputs/launch-inputs-phase0.json').read_text())
    expected=work/'phase0-envs'/role
    if Path(sys.prefix).resolve()!=expected.resolve():raise AdmissionFailure('prepared '+role+' Python is not active')
    if role=='inference':
        import torch, vllm
        if vllm.__version__!='0.18.0' or torch.__version__!='2.10.0+cu129':raise AdmissionFailure('separate inference package pins differ')
        from vllm.engine.arg_utils import EngineArgs  # noqa: F401
        return cuda_probe(work,role,torch)
    for key in ('MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER','SDPO_DIR','FINQA_ROOT','V4_TEACHER_PYTHON'):
        if not Path(launch['environment'][key]).exists():raise FileNotFoundError('prepared path missing: '+key)
    for command in ('scontrol','srun','sacct','squeue','scancel','nvidia-smi'):
        if not shutil.which(command):print('WARNING: host cannot find site command: '+command)
    import torch, verl, vllm, qwen_vl_utils  # noqa: F401
    from kit.v4_teacher import runtime_versions
    from kit.v4_phase0_site import validate_base_versions,installed_versions,torch_runtime_identity
    identity=torch_runtime_identity()
    try:validate_base_versions(installed_versions(),identity)
    except ValueError as exc:raise AdmissionFailure(str(exc)) from exc
    frozen=json.loads((work/'v4/report-phase0/environment.json').read_text())
    if installed_versions()!=frozen['trainer_dependency_versions']:raise AdmissionFailure('trainer dependencies differ from CPU prepare')
    if runtime_versions()!=frozen['runtime_versions']:raise AdmissionFailure('trainer versions differ from CPU prepare')
    if frozen.get('trainer_torch_runtime',identity)!=identity:raise AdmissionFailure('trainer torch runtime differs from CPU prepare')
    return cuda_probe(work,role,torch)


def environment_timing(import_seconds):
    """Derive the full measured requirement, including 60 seconds for CUDA startup."""
    if set(import_seconds)!= {'trainer','inference'} or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in import_seconds.values()):
        raise ValueError('prepare import timings must contain finite trainer and inference seconds')
    return {'import_seconds':dict(import_seconds),'requirement_seconds':max(120,3*max(import_seconds.values()))+60,
            'estimated_seconds':sum(import_seconds.values())+60,
            'deadline_seconds':scale(relaxed_timeout(sum(import_seconds.values())+60)), 'timing_expectation':'record-only'}


def admitted_environment_timing(import_seconds):
    """Record import timing without a performance gate; the planner records the complete graph."""
    return environment_timing(import_seconds)


def prepared_deadline(work):
    """Consume the measured timing registration stored in the prepare receipt."""
    try:timing=json.loads((Path(work)/'v4/report-inputs/prepare-receipt-phase0.json').read_text())['environment_check']
    except (OSError,KeyError) as exc:raise ValueError('CPU prepare environment timing receipt is missing; use a fresh WORK and prepare again') from exc
    if timing!=admitted_environment_timing(timing['import_seconds']):raise ValueError('prepare environment timing derivation differs')
    return timing['deadline_seconds']


def check_environment(work, seconds=None):
    """Check batch-process placement and step creation before both measured import probes."""
    work=Path(work).resolve();folder=work/'v4/report-phase0';folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();observations=[];job_tmp=None
    try:
        step=os.environ.get('SLURM_STEP_ID','')
        if step.isdecimal():observations.append({'warning':'SLURM_STEP_ID='+step+' is a numeric Slurm step; informational route observation'})
        if not os.environ.get('SLURM_JOB_ID'):raise AdmissionFailure('SLURM_JOB_ID is missing; environment check requires the existing batch allocation')
        if seconds is None:seconds=prepared_deadline(work)
        launch=json.loads((work/'v4/report-inputs/launch-inputs-phase0.json').read_text())
        from kit.v4_phase0_site import job_tmp_check
        try:job_tmp=job_tmp_check(launch['environment'])
        except ValueError as exc:raise AdmissionFailure(str(exc)) from exc
        def run(argv,log,role):
            left=seconds-(time.monotonic()-started)
            if left<=0:
                observations.append({'role':role,'warning':'environment-check deadline reached','informational':True});return
            observed=bounded_command(argv,timeout=left,env=os.environ.copy(),log=log)
            observations.append({'role':role,**observed})
            if observed['returncode']:
                detail=log.read_text(errors='replace').strip()[:500] if log.exists() else ''
                failure=folder/('environment-check-'+role.split()[0]+'-failure.json')
                if failure.is_file() and json.loads(failure.read_text()).get('hard_stop'):raise AdmissionFailure(detail)
                observations[-1]['warning']=role+' failed: '+detail
        run(['srun','--overlap','-n1','true'],folder/'environment-check-step.log','srun --overlap -n1 true step creation')
        for role,python in (('trainer',str(work/'phase0-envs/trainer/bin/python')),
                            ('inference',launch['environment']['V4_TEACHER_PYTHON'])):
            if not Path(python).is_file():raise FileNotFoundError('prepared '+role+' Python is missing from the prepared paths')
            run([python,str(Path(__file__).resolve()),'--work',str(work),'--probe',role],
                folder/('environment-check-'+role+'.log'),role+' import/device check')
        warnings=[o.get('warning') for o in observations if o.get('warning')]
        result={'ok':True,'message':'Host-venv environment check completed before the self-test.'+(' WARNING (informational): '+'; '.join(warnings) if warnings else '')}
    except (ValueError,KeyError,OSError) as exc:
        result={'ok':False,'hard_stop':isinstance(exc,(AdmissionFailure,FileNotFoundError)), 'message':('STOP before the self-test: ' if isinstance(exc,(AdmissionFailure,FileNotFoundError)) else 'WARNING (informational): ')+ ' phase-0 host-venv environment unavailable or different from CPU prepare. '+str(exc)+'. Keep the native base env lines and run directly in the batch process from k8b_pilot_run.sbatch; return this setup blocker to the owner.'}
    result.update(seconds=time.monotonic()-started,deadline_seconds=seconds,observations=observations,job_tmp=job_tmp)
    write_durably(folder/'environment-check.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--probe',choices=('trainer','inference'))
    a=p.parse_args(argv)
    if a.probe:
        try:result=probe(a.work,a.probe)
        except (AdmissionFailure,FileNotFoundError) as exc:
            result={'ok':False,'hard_stop':True,'cause':str(exc)}
            write_durably(a.work/('v4/report-phase0/environment-check-'+a.probe+'-failure.json'),result)
            print(json.dumps(result));return 2
        except (ValueError,ImportError,OSError) as exc:result={'ok':False,'informational':True,'cause':str(exc)}
        print(json.dumps(result));return 0
    result=check_environment(a.work);print(result['message']);return 2 if result.get('hard_stop') else 0

if __name__=='__main__':raise SystemExit(main())
