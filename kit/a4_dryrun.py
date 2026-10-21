#!/usr/bin/env python3
"""Every 4A row via real jobs, deliberate failures, durable job handoff and return readback."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
from kit.a4_common import _account_allocations
from kit import a4_sim, a4_run, message_handoff
import subprocess
import shutil
from types import SimpleNamespace as NS


def job(a):
    require(a.standin,'dry run is stand-in only')
    require(not (a.out/'fixtures').exists(),'fresh dry-run root required')
    a.out.mkdir(parents=True,exist_ok=True)
    fixture=a.out/'fixtures';slice_dir,b,replay=a4_sim.fixture(fixture,a.recipe)
    parent=a.out/'message';out=parent/'A4';durable=a.out/'durable';failures=[];invocations=[]
    # Model the combined message: return 1 consumed 80 h in this same allocation.
    job_starts={1:time.time()-80*3600}
    atomic(parent/'return-1.json',dict(complete=True,preceding_wall_hours=80,technical_synthetic=True))
    job_number=1
    def call(name,args=(),expect=0):
        cmd=[sys.executable,str(KIT/(name+'.py')),'--out',str(out),'--standin',*[str(x) for x in args]]
        environment=dict(os.environ,SLURM_JOB_ID=f'standin-{job_number}',
                         SLURM_JOB_START_TIME=str(job_starts[job_number]), SLURM_RESTART_COUNT='0')
        result=subprocess.run(cmd,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        with (a.out/'transcript.log').open('a') as f:f.write('$ '+' '.join(cmd)+'\n'+result.stdout)
        require(result.returncode==expect,f'{name} expected {expect}, got {result.returncode}: '+result.stdout[-2000:])
        invocations.append(dict(script=name,returncode=result.returncode))
    call('a4_run',expect=3);failures.append('unbound W')
    call('a4_allocation',['--start','--parent-out',parent,'--durable',durable])
    marker=out/'4a-starts/standin-1-r0.json'
    check(message_handoff.committed(durable)/'tree/A4/4a-starts/standin-1-r0.json',s.file_hash(marker))
    call('a4_bind',['--b-source',b,'--slice',slice_dir,'--base','synthetic-base','--replay',replay,
        '--stage-gpu-hours',3,'--approved-ceiling',600 if a.recipe in ('F','F+replay') else 800,
        '--approval-reference',APPROVAL_REFERENCE])
    call('a4_ledger',['--month',MONTHS[0]],expect=3);failures.append('feedback before complete seal')
    call('a4_run',['--through-month',1,'--fail-once'],expect=1);failures.append('interrupted stage before export')
    call('a4_handoff',['--parent-out',parent,'--durable',durable])
    end_allocation(out,binding(out,True),'standin-1',time.time())
    _account_allocations(out,binding(out,True),time.time(),('standin-1',0))
    idle=json.loads((out/'costs/allocation-idle-standin-1-r0.json').read_text())
    require(idle['a4_started_at']-idle['allocation_started_at']>=80*3600,'return-1 setup missing')
    require(abs(idle['held_seconds']-(idle['allocation_ended_at']-idle['a4_started_at']))<1e-6,
            'return-1 work charged to 4A')
    message_handoff.save(parent,durable)
    job_number=2
    job_starts[2]=time.time()-3600
    snapshot=message_handoff.committed(durable)/'tree'
    before={str(p.relative_to(snapshot)):s.file_hash(p) for p in snapshot.rglob('*') if p.is_file()}
    shutil.rmtree(parent)
    message_handoff.restore(durable,parent,move_aside=True)
    require(all(s.file_hash(parent/n)==sha for n,sha in before.items()),'job-boundary restore differs')
    call('a4_allocation',['--start','--parent-out',parent,'--durable',durable])
    call('a4_run',['--through-month',3])
    call('a4_handoff',['--parent-out',parent,'--durable',durable])
    end_allocation(out,binding(out,True),'standin-2',time.time())
    message_handoff.save(parent,durable)
    job_number=3
    job_starts[3]=time.time()
    shutil.rmtree(parent);message_handoff.restore(durable,parent,move_aside=True)
    call('a4_allocation',['--start','--parent-out',parent,'--durable',durable])
    call('a4_run')
    from kit.a4_read import load
    records,missing=load(out)
    require(not missing and len(records)==len(list(a4_run.score_plan())),'missing 4A row')
    runhash={str(p.relative_to(out)):s.file_hash(p) for p in (out/'runs').rglob('*') if p.is_file()}
    call('a4_run')
    require(runhash=={str(p.relative_to(out)):s.file_hash(p) for p in (out/'runs').rglob('*') if p.is_file()},'completed stages changed on resume')
    # Frozen identities reject a changed physical GPU and tampered admission evidence.
    physical=out/'next-scoring-machine.json';old=physical.read_bytes();atomic(physical,dict(uuid='different',environment='CPU-STANDIN'))
    call('a4_run',expect=2);physical.write_bytes(old);failures.append('different physical scoring GPU')
    trace=out/'ledger'/MONTHS[0]/'traces.jsonl';old=trace.read_bytes();trace.write_bytes(old+b'\n')
    call('a4_run',expect=2);trace.write_bytes(old);failures.append('ledger hash changed')
    # Nonzero synthetic tail cost proves collection refreshes pre-handoff readouts.
    atomic(out/'costs/standin-handoff-accounting.json',dict(kind='handoff',gpu_hours=.25,technical_synthetic=True))
    call('a4_handoff',['--parent-out',parent,'--durable',durable])
    committed=message_handoff.committed(durable)/'tree/A4'
    archive=a.out/'4a-return.tar.gz'
    call('a4_collect',['--archive',archive,'--durable-root',committed])
    # Exercise the real collector's durable bulk-hash gate even in a stand-in.
    import tarfile
    with tarfile.open(archive,'r:gz') as tar:
        compact=json.load(tar.extractfile('collect-manifest.json'))
    for record in compact['omitted']:check(committed/record['path'],record['sha256'])
    call('a4_read',['--source',archive,'--bootstrap',2000])
    original=json.loads((out/'reading/summary.json').read_text());returned=json.loads((out/'returned-reading/summary.json').read_text())
    require(original==returned,'compact archive readout differs')
    require(original['shared_study_gpu_hours']==.25,'collector omitted the nonzero synthetic tail cost')
    nll_files=[p for p in (out/'runs').glob('*/nll-*.json') if not p.name.endswith('.identity.json')]
    require(len(nll_files)==90 and all(json.loads(p.read_text())['response_tokens']>0 for p in nll_files),'before/after NLL missing')
    summary=dict(synthetic_tail_gpu_hours=.25,durable_omitted_hashes_checked=len(compact['omitted']),nll_readouts=len(nll_files),technical_synthetic=True,all_runbook_rows=True,score_cells=len(records),
        stages=len(list((out/'runs').glob('*/summary.json'))),question_rows=sum(len(r) for r in records.values()),
        failures=failures,resume_across_jobs=True,completed_stage_hashes_unchanged=True,archive_bytes=archive.stat().st_size,
        archive_readback_equal=True,invocations=invocations,recipe=a.recipe,allocation_jobs=3,
        return_1_precedes_4a_wall_hours=80,return_1_excluded_from_4a=True,
        per_job_4a_markers=len(list((out/'4a-starts').glob('*.json'))),
        approved_ceiling_gpu_hours=original['approved_ceiling_gpu_hours'])
    atomic(a.out/'dryrun-summary.json',summary);print(json.dumps(summary,indent=2))


def configure(p):p.add_argument('--recipe',choices=('F','D','F+replay','D+replay'),default='F')
if __name__=='__main__':raise SystemExit(entry(job,configure,'a4-dryrun'))
