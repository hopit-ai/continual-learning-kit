#!/usr/bin/env python3
"""All message-2 rows and 15 binding branches, CPU only, with durable requeue/readback."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit import m2_bind, m2_prepare, m2_run, m2_admit, m2_read, m2_collect, message_handoff
from kit.m2_sim import fixtures
from kit.next_dryrun import invoke
from types import SimpleNamespace as NS
from collections import Counter
from contextlib import redirect_stdout
from unittest.mock import patch
import shutil
import time


def branch(root, nomination, hard, counts):
    supplied,assets=fixtures(root/'fixtures',nomination,hard)
    out=root/'out'; common=dict(out=out,standin=True)
    expected,_,_,_=m2_bind.read_return1(NS(**common,**vars(supplied)))
    supplied.expect=root/'send-binding.json'; atomic(supplied.expect,expected)
    m2_bind.job(NS(**common,**vars(supplied)))
    m2_bind.job(NS(**common,**vars(supplied)))
    m2_prepare.job(NS(**common,**vars(assets)))
    def dispatch(cmd,**kwargs):
        invoke(Path(cmd[1]).stem,cmd[2:],counts)
        return NS(returncode=0)
    durable=root/'durable'
    def row(action,**kw):
        a=NS(**common,action=action,durable=durable,model=None,tokenizer=None,sdpo=None,panel_root=None,gpu='0,1,2,3,4,5,6,7',
             score_gpu='0',teacher_gpu='0,1',trainer_python=sys.executable,inference_python=sys.executable,
             bootstrap=20,standin_pending=nomination=='pending')
        for k,v in kw.items():setattr(a,k,v)
        with patch.object(m2_run.subprocess,'run',side_effect=dispatch):m2_run.job(a)
    row('boundaries');row('targets');row('targets')
    # Plain D never waits for admission, including a missing sibling confirmation.
    row('math-plain')
    row('legal-generic')
    row('admit')
    row('legal-broad');row('admit');row('legal-next');row('admit')
    message_handoff.save(out,durable)
    row('math-sibling')
    row('math-sibling')  # finished rows SKIP; pending rows still wait
    m2_read.job(NS(out=out/'reading',source=out,standin=True,bootstrap=20))
    result=json.loads((out/'reading/reading.json').read_text())
    admitted=json.loads((out/'admission.json').read_text())
    expected='PENDING' if nomination=='pending' else 'COMPLETE'
    require(admitted['status']==expected,'unexpected admission branch')
    if expected=='COMPLETE':require(admitted['configuration']==('generic' if nomination=='none' else nomination),'wrong admitted configuration')
    with patch.dict(os.environ,SLURM_JOB_ID='m2-standin',SLURM_RESTART_COUNT='0',SLURM_JOB_START_TIME=str(time.time()-1)):
        message_handoff.record_job_start(durable,True)
    message_handoff.save(out,durable)
    # Simulate a requeue on the same allocation: erase local OUT, restore last committed generation,
    # repeat an unfinished/completed row, then verify immutable training receipts.
    before={str(p.relative_to(out)):s.file_hash(p) for p in (out/'runs').rglob('*') if p.is_file()}
    shutil.rmtree(out)
    with patch.dict(os.environ,SLURM_JOB_ID='m2-standin',SLURM_RESTART_COUNT='1',SLURM_JOB_START_TIME=str(time.time()-1)):
        requeued=message_handoff.record_job_start(durable,True)
        require(requeued['slurm_restart_count']==1,'requeue restart count lost')
    message_handoff.restore(durable,out)
    row('math-sibling')
    require(all(s.file_hash(out/name)==sha for name,sha in before.items()),'requeue changed a completed stage')
    archive=root/'message2-return.tar.gz'
    m2_collect.job(NS(**common,archive=archive));m2_collect.job(NS(**common,archive=archive))
    m2_read.job(NS(out=root/'returned',source=archive,standin=True,bootstrap=20))
    require(json.loads((root/'returned/reading.json').read_text())==result,'archive readback differs')
    return dict(nomination=nomination,hard_math=hard,admission=admitted['status'],configuration=admitted.get('configuration'),
                stages=len(list((out/'runs').glob('*/complete.json'))),score_jobs=len(list((out/'scores').glob('*/manifest.json'))),
                missing_cells=len(result['missing']),archive_bytes=archive.stat().st_size,requeue=True)


def job(a):
    require(a.standin,'dry run requires --standin')
    require(not (a.out/'dryrun.log').exists(),'fresh dry-run OUT required')
    a.out.mkdir(parents=True,exist_ok=True); counts=Counter();results=[]
    with (a.out/'dryrun.log').open('w') as log,redirect_stdout(log):
        for nomination in ('broad','earlier','both','none','pending'):
            for hard in ('QUALIFIES','DOES_NOT_QUALIFY','INCOMPLETE'):
                results.append(branch(a.out/(nomination+'-'+hard),nomination,hard,counts))
    summary=dict(technical_synthetic=True,branches=len(results),results=results,invocations=dict(counts),gpu_hours=0,
                 rows='M1–M13',archive_readbacks=len(results),requeues=len(results))
    atomic(a.out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


def configure(p): pass
if __name__=='__main__':
    if '--out' not in sys.argv and not os.environ.get('OUT'):
        import tempfile
        sys.argv += ['--out',tempfile.mkdtemp(prefix='message2-standin-')]
    raise SystemExit(entry(job,configure,'message2-dryrun'))
