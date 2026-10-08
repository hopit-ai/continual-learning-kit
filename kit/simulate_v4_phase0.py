#!/usr/bin/env python3
"""CPU PHASE-0 rehearsal with real runner, bars, dependencies, timeouts and collector."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kit.v4_campaign import build
from kit.v4_phase0 import read_archive,required_seconds,BASE
CASES=('success','missing_paste','failed_selftest','determinism_mismatch','out_of_memory','wall_time','pyxis_route','pre_selftest_refusal','host_venv_success')


def synthetic_presend(work):
    """Owned fictional raw settings exercise the real CPU parser, never node probes."""
    from kit.p4_contain import check
    from kit.runner import write_durably
    from kit.v4_prepare import file_receipt
    folder=Path(work)/BASE/'presend';paste=folder/'paste';paste.mkdir(parents=True)
    raw={'version.txt':'slurm 25.05.3\n',
         'config.txt':'ProctrackType = proctrack/cgroup\nTaskPlugin = task/cgroup,task/affinity\nJobAcctGatherType = jobacct_gather/cgroup\nKillWait = 40 sec\nOverTimeLimit = 0 min\nPreemptType = preempt/none\nPreemptMode = OFF\n',
         'partition.txt':'PartitionName=cpu-rehearsal OverTimeLimit=NONE PreemptMode=OFF\n',
         'qos.txt':'cpu-rehearsal|OFF|0|\n','cgroup.conf':'ConstrainDevices=yes\nSignalChildrenProcesses=no\n'}
    for name,text in raw.items():(paste/name).write_text(text)
    result=check(work,folder,from_file=paste,qos='cpu-rehearsal')
    if not result['ok']:raise ValueError('synthetic paste parser refused: '+str(result['problems']))
    write_durably(folder/'paste-files.json',file_receipt(paste.iterdir()))


def rehearse_pre_selftest(out,work,case):
    """Execute refusal, host reconciliation and collection with owned scheduler executables."""
    from kit.v4_phase0_submission import record_submission
    from kit.v4_phase0_route import probe_route
    folder=work/BASE;allocation=folder/'allocation';allocation.mkdir(parents=True)
    plan=allocation/'phase0.json'
    plan.write_text(json.dumps({'phase':'phase0','block':'qualification','block_limit':100,'ceiling':560,'gpus':8,
        'sbatch_time':'01:50:00','phase0_allocation_cap_gpu_hours':22,'prior_allocations':{}}))
    wrapper=allocation/'phase0.sbatch';wrapper.write_text('#!/bin/bash\n# CPU stand-in; never submitted to real Slurm.\n')
    record_submission(work,'123',plan,wrapper)
    inputs=work/'v4/report-inputs';inputs.mkdir(parents=True)
    (inputs/'prepare-receipt-phase0.json').write_text(json.dumps({'environment_check':{'import_seconds':{'trainer':1,'inference':1},'deadline_seconds':180}}))
    (inputs/'launch-inputs-phase0.json').write_text(json.dumps({'environment':{'V4_TEACHER_PYTHON':str(work/'unstarted-python'),'TMPDIR':'/tmp'}}))
    bindir=out/'scheduler-standins';bindir.mkdir()
    control='JobId=123 JobState=FAILED StartTime=2026-10-08T00:00:00+00:00 EndTime=2026-10-08T00:02:00+00:00 TimeLimit=01:50:00 AllocTRES=gres/gpu=8 NumCPUs=32'
    for name,body in {'scontrol':'print('+repr(control)+')', 'srun':'import sys; print("stand-in step creation refused",file=sys.stderr); sys.exit(1)'}.items():
        file=bindir/name;file.write_text('#!'+sys.executable+'\n'+body+'\n');file.chmod(0o755)
    if case=='pyxis_route':
        pilot=out/'standin-pilot.sbatch';pilot.write_text('srun --container-image=standin-image python payload.py\n')
        probe_route(work,pilot)
    env={**os.environ,'SLURM_JOB_ID':'123','SLURM_STEP_ID':'0' if case=='pyxis_route' else 'batch',
        'PATH':str(bindir)+os.pathsep+os.environ['PATH'],'TMPDIR':'/tmp'}
    done=subprocess.run([sys.executable,str(ROOT/'kit/v4_phase0_environment.py'),'--work',str(work)],env=env,capture_output=True,text=True,timeout=10)
    (out/'environment.log').write_text(done.stdout+done.stderr)
    env.pop('SLURM_JOB_ID',None);env.pop('SLURM_STEP_ID',None)
    reconciled=subprocess.run([sys.executable,'-S',str(ROOT/'kit/p4_contain.py'),'reconcile','--work',str(work),'--out',str(work/'reconcile'),'--seconds','1'],env=env,capture_output=True,text=True,timeout=10)
    (out/'reconcile.log').write_text(reconciled.stdout+reconciled.stderr)
    if reconciled.returncode:raise ValueError('stand-in pre-selftest reconciliation failed: '+reconciled.stderr+reconciled.stdout)
    archive=out/'phase0.tar.gz'
    collected=subprocess.run([sys.executable,'-S',str(ROOT/'kit/collect.py'),'--work',str(work),'--out',str(archive)],env=env,capture_output=True,text=True,timeout=30)
    (out/'collect.log').write_text(collected.stdout+collected.stderr)
    if collected.returncode:raise ValueError('stand-in pre-selftest collection failed')
    report={'scope':'CPU stand-ins; no GPU or containment certification','case':case,'runner_exit':done.returncode,
        'training_started':False,'reader':read_archive(archive),'planned_seconds':required_seconds(),'archive':str(archive)}
    (out/'simulation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def synthetic_download(out,work):
    """Owned tiny snapshots exercise the real compute-receipt verifier, never HF."""
    from kit.v4_phase0_download import MODELS,snapshot_path,make_receipt,verify_download_receipt
    from kit.runner import write_durably
    hf=out/'task/hf-cache';paths={}
    for role,(repo,revision) in MODELS.items():
        path=snapshot_path(hf,repo,revision);path.mkdir(parents=True)
        (path/'config.json').write_text(json.dumps({'_commit_hash':revision}))
        (path/'model.safetensors').write_bytes(b'CPU host-route stand-in, never loaded')
        paths[role]=path
    doc=make_receipt(work,hf,paths,{'job_id':'122','gpus':0})
    verify_download_receipt(work,doc,paths)
    write_durably(work/BASE/'download/receipt.json',doc)


def synthetic_submission(work):
    """Fictional running allocation exercises the real login receipt and live gate."""
    import time
    from kit import p4_contain as pc
    from kit.v4_phase0_submission import record_submission
    from kit.v4_teacher import sha
    inputs=work/'v4/report-inputs';inputs.mkdir(parents=True)
    prepare=inputs/'prepare-receipt-phase0.json';prepare.write_text(json.dumps({'files':{}}))
    (inputs/'launch-inputs-phase0.json').write_text(json.dumps({'paths':{},'trainer_prefix':sys.prefix}))
    allocation=work/BASE/'allocation';allocation.mkdir()
    plan=allocation/'phase0.json';plan.write_text(json.dumps({'phase':'phase0','block':'qualification','block_limit':100,'ceiling':560,'gpus':8,
        'sbatch_time':'01:50:00','phase0_allocation_cap_gpu_hours':22,'prior_allocations':{},
        'phase0_prepare_sha256':sha(prepare.read_bytes()),'phase0_presend_sha256':sha((work/BASE/'presend/containment-presend.json').read_bytes())}))
    wrapper=plan.with_suffix('.sbatch');wrapper.write_text('# CPU stand-in; never submitted.\n')
    record_submission(work,'123',plan,wrapper)
    ledger=pc.record_allocation(work,{'job_id':'123','state':'RUNNING','start':pc.wd.precise_text(pc.wd.from_epoch(time.time()-120)),
        'end':None,'width':8,'time_limit_seconds':6600},'qualification',100,560)
    ledger['allocations']['123']['phase0']=True
    pc.wd.write_durably(work/pc.DIRECTORY/'allocation-ledger.json',ledger)


def reconcile_success(out,work):
    """Close the fictional submitted job through the real stdlib-only return tool."""
    from kit import p4_contain as pc
    entry=json.loads((work/pc.DIRECTORY/'allocation-ledger.json').read_text())['allocations']['123']
    control='JobId=123 JobState=COMPLETED StartTime='+entry['start']+' EndTime='+pc.wd.precise_text(pc.wd.from_epoch(time.time()))+' TimeLimit=01:50:00 AllocTRES=gres/gpu=8 NumCPUs=32'
    bindir=out/'scheduler-standins';bindir.mkdir()
    script=bindir/'scontrol';script.write_text('#!'+sys.executable+'\nprint('+repr(control)+')\n');script.chmod(0o755)
    env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH']}
    env.pop('SLURM_JOB_ID',None);env.pop('SLURM_STEP_ID',None)
    done=subprocess.run([sys.executable,'-S',str(ROOT/'kit/p4_contain.py'),'reconcile','--work',str(work),'--out',str(work/'reconcile'),'--seconds','1'],env=env,capture_output=True,text=True,timeout=10)
    (out/'reconcile.log').write_text(done.stdout+done.stderr)
    if done.returncode:raise ValueError('CPU success reconciliation failed: '+done.stdout+done.stderr)


def rehearse(out,case='success'):
    if case not in CASES:raise ValueError('unknown CPU case')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);work=out/'work';work.mkdir()
    if case in ('pyxis_route','pre_selftest_refusal'):return rehearse_pre_selftest(out,work,case)
    doc=build('phase0');doc['v4'].pop('allocation_stages')
    doc['v4']['phase']='CPU-phase0-rehearsal'
    for row in doc['rows']:
        row['command']=[sys.executable,str(ROOT/'kit/sim/v4_phase0_row.py'),'--work','{work}','--row',row['id']]
        if case=='wall_time' and row['id']=='train-q-S-finqa':row['timeout_seconds']=.2
    path=out/'standin-campaign.json';path.write_text(json.dumps(doc,indent=2)+'\n')
    env={**os.environ,'WORK':str(work),'SIM_PHASE0_CASE':case}
    env.pop('SLURM_JOB_ID',None)
    # The diagnostic reader checks the frozen stand-in graph and real runner
    # records. It never promotes these synthetic records into a hardware pass.
    folder=work/'v4/report-simulation';folder.mkdir(parents=True)
    (folder/'campaign.json').write_text(json.dumps(doc,indent=2)+'\n')
    if case!='missing_paste':synthetic_presend(work)
    if case in ('success','host_venv_success'):
        if case=='host_venv_success':synthetic_download(out,work)
        synthetic_submission(work)
        for row in doc['rows'][-2:]:row['env']['SLURM_JOB_ID']='123'
        path.write_text(json.dumps(doc,indent=2)+'\n')
        (folder/'campaign.json').write_text(json.dumps(doc,indent=2)+'\n')
    done=subprocess.run([sys.executable,str(ROOT/'kit/runner.py'),'batch',str(path)],env=env,capture_output=True,text=True,timeout=120)
    (out/'runner.log').write_text(done.stdout+done.stderr)
    if case in ('success','host_venv_success'):reconcile_success(out,work)
    archive=out/'phase0.tar.gz'
    collected=subprocess.run([sys.executable,str(ROOT/'kit/collect.py'),'--work',str(work),'--out',str(archive)],env=env,capture_output=True,text=True,timeout=120)
    (out/'collect.log').write_text(collected.stdout+collected.stderr)
    if collected.returncode:raise ValueError('CPU rehearsal collector failed')
    report={'scope':'CPU stand-ins; no GPU or containment certification','case':case,'runner_exit':done.returncode,
            'training_started':any((work/'campaign/v4-phase0').glob('train-*/attempt-*/start.json')),
            'reader':read_archive(archive),'planned_seconds':required_seconds(),'archive':str(archive)}
    if case in ('success','host_venv_success') and done.returncode==0:
        report['live_report']=json.loads((work/BASE/'reading.json').read_text())
        report['pause']=json.loads((work/BASE/'pause.json').read_text())
        if case=='host_venv_success':report['download']=json.loads((work/BASE/'download/receipt.json').read_text())
    (out/'simulation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',choices=(*CASES,'all'),default='all');a=p.parse_args(argv)
    results=[rehearse(a.out/case,case) for case in CASES] if a.case=='all' else [rehearse(a.out,a.case)]
    print(json.dumps(results,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
