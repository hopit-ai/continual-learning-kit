#!/usr/bin/env python3
"""Prepared 4A leaf: unchanged generation/NLL numerics, measured allocation evidence."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
from kit import a4_score,a4_nll
from kit.p3b_score import write_csv
import argparse


def physical(standin):
    from kit.lane_pool import gpu_uuid
    return dict(uuid=gpu_uuid(),environment=s.digest(dict(python=sys.executable,versions=t.runtime_versions())),
        allocation_nonce=os.environ['A4_ALLOCATION_NONCE'],host=socket.gethostname(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),technical_synthetic=standin)


def job(a):
    spec=json.loads(a.spec.read_text());standin=spec['technical_synthetic'];measured=physical(standin)
    require(t.runtime_versions()==spec['software'],'bound inference software changed')
    require(not a.out.exists(),'fresh per-attempt score output required')
    a.out.mkdir(parents=True)
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    require(',' not in gpu,'one allocated scoring GPU required')
    if a.nll:
        check(Path(spec['schedule']),spec['schedule_sha256'])
        rows=t.read_jsonl(Path(spec['schedule']))
        clock=time.monotonic();result=a4_nll.measure(spec['record']['model'],rows,standin)
        require(physical(standin)==measured,'NLL GPU changed')
        atomic(a.out/'result.json',dict(result,seconds=time.monotonic()-clock,identity=dict(model=spec['model'],schedule=spec['schedule_sha256'],physical=measured),technical_synthetic=standin))
        freeze(a.out,['result.json']);return
    kind,arm,seed,n,p,mode=spec['row'];data=spec['data']
    clock=time.monotonic()
    try:raw=a4_score.generate(data,spec['record'],gpu,standin,arm,kind)
    finally:
        from kit.inference_lifecycle import shutdown_since
        shutdown_since()
    require(physical(standin)==measured,'scoring GPU changed')
    generation=s.digest(dict(model=spec['model'],prompts=[dict(id=r['id'],panel=r['panel'],prompt=r['prompt']) for r in data],physical=measured))
    ident=dict(model=spec['model'],data=s.digest(data),physical=measured,binding=spec['binding'],technical_synthetic=standin)
    elapsed=time.monotonic()-clock
    for index,entry in enumerate([dict(index=spec['index'],row=spec['row']),*spec['aliases']]):
        kind,arm,seed,n,p,mode=entry['row']
        root=a.out if index==0 else a.out.parent/str(entry['index'])
        root.mkdir(parents=True,exist_ok=index==0)
        rawfile=root/'raw/responses.jsonl';rawfile.parent.mkdir();t.write_jsonl(rawfile,raw)
        result=a4_score.graded(data,raw,dict(kind=kind,arm=arm,seed=seed,month=n,mode=mode,generation_id=generation),standin)
        write_csv(root/'scores.csv',result,a4_score.COLUMNS);atomic(root/'identity.json',ident)
        atomic(root/'generated.json',dict(cache_key=generation,reused=index!=0,sha256=s.file_hash(rawfile),seconds=elapsed if index==0 else 0))
        freeze(root,['scores.csv','identity.json','generated.json','raw/responses.jsonl'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--nll',action='store_true');a=p.parse_args()
    try:
        with job_deadline(json.loads(a.spec.read_text())['max_seconds']):job(a)
    except NotReady as exc:print('PENDING:',exc);return 75
    except CorrectnessStop as exc:print('STOP:',exc);return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
