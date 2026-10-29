#!/usr/bin/env python3
"""CPU native-host rehearsal; adapters cannot certify Slurm/GPU admission or authorize science."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.v4_campaign import build
from kit.v4_allocation import script,header
ROOT=Path(__file__).resolve().parents[1]


def rehearse(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    work=out/'shared-storage/shared-storage/operator/partner/v4-finqa/work.ABCDEF';work.mkdir(parents=True)
    shim=out/'cpu-adapter-kit';(shim/'campaigns').mkdir(parents=True)
    ledger=work/'k8b4/containment/allocation-ledger.json';ledger.parent.mkdir(parents=True)
    import time
    from kit.p4_watchdog import precise_text,from_epoch
    start=precise_text(from_epoch(time.time()));end=precise_text(from_epoch(time.time()+86400))
    ledger.write_text(json.dumps({'allocations':{'123':{'start':start,'planned_end':end,'observed_until':start,'width':8,'segments':[{'start':start,'block':'scientific'}]}},'CPU_standin':True}))
    inputs=work/'v4/report-inputs';inputs.mkdir(parents=True);(inputs/'phase0-binding.json').write_text('{}')
    activate=out/'activate';activate.write_text('# CPU stand-in for the retained trainer venv; no model installed.\n')
    adapter='import sys,os,json\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(ROOT))+')\n'
    (shim/'v4_phase0_carry.py').write_text(adapter+'print("CPU source-admission stand-in; no scientific authorization")\n')
    (shim/'runner.py').write_text(adapter+'from kit import runner\nrunner.__file__=__file__  # Resolve only kit entry paths to CPU adapters.\nraise SystemExit(runner.main())\n')
    (shim/'v4_finqa_site.py').write_text(adapter+'''from kit import v4_finqa_site as s
s.JOB_TMP=os.environ['CPU_HOST_JOB_TMP'];s.RAY_JOB_TMP=s.JOB_TMP+'/r'
s.frozen_environment=lambda work: json.loads((Path(work)/s.LAUNCH).read_text())['environment']
raise SystemExit(s.main())
''')
    (shim/'p4_contain.py').write_text(adapter+'''from kit import p4_contain as p,v4_finqa_site as s
import tempfile
s.JOB_TMP=os.environ['CPU_HOST_JOB_TMP'];s.RAY_JOB_TMP=s.JOB_TMP+'/r'
def external_slurm_standin(argv):
    assert os.environ['TMPDIR']==s.JOB_TMP and tempfile.gettempdir()==s.JOB_TMP
    assert os.environ['VLLM_RPC_BASE_PATH']=='/tmp'
    assert Path(os.environ['PIP_CACHE_DIR']).is_relative_to(Path(os.environ['WORK']).parent/'cache')
    file=Path(os.environ['WORK'])/'v4/report-host-rehearsal/containment-entry.jsonl'
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('a') as handle:handle.write(json.dumps(dict(stage=os.environ['CPU_HOST_STAGE'],tmpdir=os.environ['TMPDIR'],ray=os.environ['RAY_TMPDIR'],external_slurm_standin=True,containment_certified=False))+'\\n')
    return 0
p._main=external_slurm_standin
raise SystemExit(p.main())
''')
    (shim/'v4_finqa_flow.py').write_text(adapter+'''from kit import v4_finqa_flow as f,runner
f.KIT=Path(__file__).parent;runner.__file__=str(Path(__file__).with_name('runner.py'))
raise SystemExit(f.main())
''')
    rows={};codes={}
    for phase in ('qualification','main'):
        graph=build(phase);graph['v4']['CPU_native_host_rehearsal']=True
        for row in graph['rows']:
            if not row['id'].endswith('selftest'):
                row['command']=[sys.executable,str(ROOT/'kit/sim/v4_finqa_row.py'),'--work','{work}','--phase',phase,'--row',row['id'],'--attempt','{attempt}']
        (shim/f'campaigns/v4-{phase}.yaml').write_text(json.dumps(graph,indent=2)+'\n')
        rows[phase]=len(graph['rows'])
    with tempfile.TemporaryDirectory(prefix='fq-',dir='/tmp') as short:
        from kit import v4_finqa_site as site
        launch=site.storage_environment(work,allocation=True)
        launch.update(TMPDIR=short,RAY_TMPDIR=short+'/r',WORK=str(work),KIT=str(shim))
        (inputs/'launch-inputs-finqa.json').write_text(json.dumps({'environment':launch}))
        phase=stage='qualification'
        doc=dict(phase=phase,stage=stage,block='scientific',block_limit=1000,ceiling=1000,gpus=8,
                 sbatch_time='24:00:00',launch_environment=launch,site={'activate':str(activate)},
                 plan_path=str(work/'flow.json'))
        payload=work/'flow.sh';payload.write_text(script(doc,work,shim))
        (work/'flow.json').write_text(json.dumps(doc))
        wrapper=payload.with_suffix('.sbatch');wrapper.write_text(header(doc,work)+'# CPU stand-in of native wrapper.\nexport PATH="'+str(Path(sys.executable).parent)+':$PATH"\nbash '+str(payload)+'\n')
        env={**os.environ,'PYTHONPATH':str(ROOT),'SLURM_JOB_ID':'123','SLURM_STEP_ID':'batch',
             'CPU_HOST_STAGE':'automatic-flow','CPU_HOST_JOB_TMP':short}
        done=subprocess.run(['bash',str(wrapper)],env=env,capture_output=True,text=True,timeout=180)
        (out/'automatic-flow.log').write_text(done.stdout+done.stderr)
        codes['automatic-flow']=done.returncode
        if done.returncode:raise ValueError('CPU native flow failed; see '+str(out))
    # Real collector/reader reject adapters as scientific evidence.
    collected=subprocess.run([sys.executable,str(ROOT/'kit/collect.py'),'--work',str(work),'--out',str(out/'return.tar.gz')],capture_output=True,text=True,timeout=60)
    if collected.returncode:raise ValueError('real collector failed: '+collected.stderr)
    from kit.v4_readers import read_campaign
    result=dict(scope='CPU native-host route only; no scientific or containment certification',rows=rows,exit_codes=codes,
                qualification_pause=False,fictional_proceed=False,automatic_main=True,containment_entry_executed=True,
                short_job_tmp_checked=True,reader=read_campaign(out/'return.tar.gz'))
    (out/'simulation.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    print(json.dumps(rehearse(a.out),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
