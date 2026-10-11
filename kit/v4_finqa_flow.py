#!/usr/bin/env python3
"""Self-monitored FinQA flow: qualification first, main follows, no owner pause."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit import runner
from kit.v4_finqa_policy import CorrectnessStop,ceiling,warn,confine
from kit.v4_phase0_timing import scale
KIT=Path(__file__).resolve().parent


def continuation(work):
    """Repeat the copied native wrapper only after this allocation ends."""
    wrapper=confine(work,Path(os.environ['V4_FLOW_WRAPPER']))
    if not wrapper.is_file():
        raise FileNotFoundError('native continuation wrapper missing: '+str(wrapper))
    command=['sbatch','--parsable','--dependency=afterany:'+os.environ['SLURM_JOB_ID'],str(wrapper)]
    done=subprocess.run(command,capture_output=True,text=True,timeout=scale(60),check=True)
    folder=confine(work,Path(work)/'v4/report-flow/submissions');folder.mkdir(parents=True,exist_ok=True)
    record={'command':command,'job_id':done.stdout.strip().split(';')[0], 'stdout':done.stdout,'stderr':done.stderr}
    if not record['job_id'].isdigit():raise RuntimeError('continuation sbatch did not return a job ID: '+done.stdout)
    runner.write_durably(folder/(str(time.time_ns())+'.json'),record)
    print('CONTINUATION: '+record['job_id']);return record


def flow_environment(function):
    def wrapped(work,*args,**kwargs):
        old={key:os.environ.get(key) for key in ('WORK','V4_FINQA_CAMPAIGN','V4_PHASE0_MODE')}
        try:return function(work,*args,**kwargs)
        finally:
            for key,value in old.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
    return wrapped

@flow_environment
def run(work,*,campaigns=None):
    work=Path(work).resolve();os.environ.update(WORK=str(work),V4_FINQA_CAMPAIGN='1',V4_PHASE0_MODE='0')
    campaigns=Path(campaigns or KIT/'campaigns')
    # CPU prepared both phases before sbatch. Each row reuses bound model/data
    # identity; plan projections never authorize or prohibit scientific work.
    results={};code=0
    for phase in ('qualification','main'):
        started_spend=ceiling(work)
        graph=runner.load_campaign(campaigns/f'v4-{phase}.yaml')
        envelope=sum(r.get('timeout_seconds',3600) for r in graph['rows'])*8/3600
        if phase=='qualification' and envelope>100:
            warn('qualification projection',f'{envelope:.6f} GPU-hour timeout envelope exceeds planned 100; continuing within actual 1000 ceiling',work)
        rows,registration=runner.allocation_rows(graph,'all')
        runner.allocation_selftest(graph,rows,registration)
        result=runner.cmd_run(graph,rows,True,None,parallel=False);results[phase]=result
        ended_spend=ceiling(work)
        progress=confine(work,work/'v4/report-flow/progress');progress.mkdir(parents=True,exist_ok=True)
        runner.write_durably(progress/(phase+'-'+str(time.time_ns())+'.json'),dict(phase=phase,result=result,planned_qualification_gpu_hours=100,timeout_envelope_gpu_hours=envelope,actual_spend_at_start=started_spend,actual_spend_at_end=ended_spend,observed_allocation_gpu_hours_delta=max(0,ended_spend-started_spend),owner_proceed_required=False,scope='Observed cumulative allocation-spend boundaries; idle outside this phase remains in the total ledger.'))
        print(f'PROGRESS: {phase}: cumulative actual {ended_spend:.6f} / 1000 GPU-hours')
        if result==75:
            if os.environ.get('V4_FLOW_WRAPPER'):continuation(work)
            else:warn('flow continuation','allocation ended; set V4_FLOW_WRAPPER to the retained native wrapper for automatic continuation',work)
            return 75
        if result==runner.EXIT_REFUSED:
            # Correctness/ceiling stops are the only FinQA admission refusals.
            return result
        if result:warn('flow '+phase,'failed rows retained; independent work continues; fix and rerun in SAME WORK',work)
        code=code or result
    folder=confine(work,work/'v4/report-flow');folder.mkdir(parents=True,exist_ok=True)
    runner.write_durably(folder/(str(time.time_ns())+'.json'),{'phases':results,'owner_proceed_required':False})
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args(argv)
    try:return run(a.work)
    except CorrectnessStop as exc:print('STOP: '+str(exc));return 2
    except (OSError,ValueError,subprocess.SubprocessError) as exc:print('FAILED: flow: '+str(exc));return 1
if __name__=='__main__':raise SystemExit(main())
