#!/usr/bin/env python3
"""CPU row stand-in for orchestration rehearsal, never a model or Slurm proof."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--row',required=True)
    parser.add_argument('--phase',required=True);parser.add_argument('--attempt',type=int,required=True)
    args=parser.parse_args();out=args.work/'v4/report-simulation';out.mkdir(parents=True,exist_ok=True)
    failure=os.environ.get('SIM_V4_FAILURE','')
    selected=os.environ.get('SIM_V4_FAIL_ROW','')
    failed=failure and args.row==selected and args.attempt==1
    event={'stand_in':True,'scientific_phase_allowed':False,'row':args.row,'phase':args.phase,
           'attempt':args.attempt,'offload':os.environ.get('OFFLOAD','0'),'failure':failure if failed else None,
           'containment_executed':False,'row_operation_executed':False}
    (out/(args.phase+'-'+args.row+'-a'+str(args.attempt)+'.json')).write_text(json.dumps(event)+'\n')
    if failed:
        print('CPU STAND-IN injected '+failure,flush=True)
        if failure=='ceiling':
            ledger=args.work/'k8b4/containment/allocation-ledger.json';ledger.parent.mkdir(parents=True,exist_ok=True)
            ledger.write_text(json.dumps({'allocations':{'CPU-ceiling':dict(start=1,end=450001,width=8,segments=[dict(block='scientific',start=1,end=450001)])}}))
        if failure=='killed_row':
            import signal
            os.kill(os.getpid(),signal.SIGKILL)
        return 1
    from kit.sim.v4_merge_standin import evaluation_and_sweep
    evaluation_and_sweep(args.work,args.phase,args.row)
    if args.row.startswith(('train-','merge-')):
        from kit.sim.v4_merge_standin import prepare_retained_run,recover,evaluation_and_sweep
        from kit.v4_campaign_ops import job_for
        ident=args.row.split('-',1)[1]
        if args.row.startswith('train-'):prepare_retained_run(args.work,args.phase,ident)
        else:
            recover(args.work,args.phase,ident)
            event.update(row_operation_executed=True,arm=job_for(ident,args.phase)['arm'],
                         external_model_merger='CPU stand-in',real_merge_recovery=True)
            (out/(args.phase+'-'+args.row+'-a'+str(args.attempt)+'.json')).write_text(json.dumps(event)+'\n')
    status=args.work/'v4/report-status';status.mkdir(parents=True,exist_ok=True)
    (status/(args.row+'.json')).write_text(json.dumps({'ok':1,'stand_in':True,'scientific_phase_allowed':False})+'\n')
    return 0


if __name__=='__main__':sys.exit(main())
