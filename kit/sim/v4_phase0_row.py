#!/usr/bin/env python3
"""PHASE-0 CPU stand-in row. Its records can never certify real hardware."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def cpu_report_tables(evidence,report):
    """Stand in for model/training tables only; real live accounting runs first."""
    from kit import v4_phase0 as p
    from kit.v4_qualification import compare_pairs
    p.archived_presend(evidence)
    graph=evidence.json('v4/report-simulation/campaign.json')
    for row in graph['rows'][:-2]:
        root='campaign/v4-phase0/'+row['id']+'/attempt-1/'
        p.require(evidence.json(root+'start.json')['campaign_sha256']==p.t.sha(evidence.files['v4/report-simulation/campaign.json']) and
            evidence.json(root+'verdict.json')['verdict']=='PASS','CPU prior runner row differs')
    report['tables']['determinism']=compare_pairs(evidence.json('v4/report-simulation/scoring-pairs.json'),('finqa',))
    report.update(status='technical pass',scope='CPU stand-ins; no GPU or containment certification')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--row',required=True)
    a=p.parse_args();case=os.environ.get('SIM_PHASE0_CASE','success')
    failure={'missing_paste':'verify-prepare','failed_selftest':'containment-selftest',
             'determinism_mismatch':'scoring-agreement','out_of_memory':'train-q-S-finqa','wall_time':'train-q-S-finqa'}
    folder=a.work/'v4/report-simulation';folder.mkdir(parents=True,exist_ok=True)
    (folder/(a.row+'.json')).write_text(json.dumps({'stand_in':True,'row':a.row,'case':case,'scientific_phase_allowed':False})+'\n')
    if case in ('success','host_venv_success') and a.row in ('report','PAUSE'):
        from kit import v4_phase0 as p
        # Only GPU/model table validation is a CPU stand-in. The production CLI,
        # operation, DirectoryEvidence and live submission/budget gate are real.
        p.rehearsal_report=cpu_report_tables
        return p.main(['report' if a.row=='report' else 'pause','--work',str(a.work),'--row',a.row])
    if a.row=='scoring-agreement':
        from kit.v4_qualification import compare_pairs
        runs=[{'engine_ok':True,'configuration_ok':True,'fingerprint':'CPU-STANDIN',
               'answers':['Answer: 1']*50,'verdicts':[1]*50} for _ in range(4)]
        if case=='determinism_mismatch':runs[3]['answers'][0]='Answer: 2'
        (folder/'scoring-pairs.json').write_text(json.dumps({'finqa':runs})+'\n')
        compare_pairs({'finqa':runs},('finqa',))
        if case in ('success','host_venv_success'):(a.work/'v4/report-phase0/scoring-agreement.json').write_text('{}')
    if a.row==failure.get(case):
        if case=='wall_time':time.sleep(10)
        else:print({'missing_paste':'partner Slurm paste is absent','failed_selftest':'selftest refused',
                    'out_of_memory':'CUDA out of memory'}[case],flush=True)
        return 1
    status=a.work/'v4/report-status';status.mkdir(parents=True,exist_ok=True)
    (status/(a.row+'.json')).write_text(json.dumps({'ok':1,'stand_in':True})+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
