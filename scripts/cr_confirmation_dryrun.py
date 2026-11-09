#!/usr/bin/env python3
"""Every confirmation row through the real shell/controller/watcher, CPU only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse
import os
import subprocess
from kit.nm_common import atomic,load,KIT
from kit.cr_jobs import inventory
from kit.nm_jobs import current
from kit.v6_infra import authority_sha256


def run(directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    config=directory/'config.json';out=directory/'confirmation'
    atomic(config,dict(out=str(directory/'six-rows'),job_python=sys.executable,
        technical_synthetic=True,infrastructure_update=authority_sha256(),
        cr_confirmation=dict(out=str(out),six_row_root=str(directory/'root-A'),
            v6b_root=str(directory/'v6b'),v6a_root=str(directory/'v6a'),hosts=['CPU-STANDIN'])))
    env=dict(os.environ,KIT=str(KIT),NM_CONFIG=str(config),JOB_PY=sys.executable,PYTHONDONTWRITEBYTECODE='1')
    env.pop('UTIL_WATCH_ACTIVE',None)
    calls=[]
    for row in inventory():
        done=subprocess.run(['bash',str(KIT/'jobs/nm/run-task.sh'),row['id']],env=env,capture_output=True,text=True,timeout=45)
        expected=0 if row['phase'] in ('account','read','collect') else 75
        assert done.returncode==expected,(row['id'],done.returncode,done.stdout,done.stderr)
        record=current(out,row['id'],complete=False)
        if row['phase']=='bind':
            binding=load(out/'attempts'/row['id']/record['nonce']/'input-binding.json')
            assert binding['status']=='PENDING' and binding['escalation']=='owner'
        calls.append(dict(row=row['id'],returncode=done.returncode,status=record['status'],nonce=record['nonce']))
    # A malformed source inventory exercises a distinct failure attempt in the
    # actual child/watcher; restore the original missing-input config afterwards.
    raw=directory/'malformed.json';raw.write_text('not-json')
    from kit.nm_common import sha
    cfg=load(config);cfg['cr_confirmation']['realised_inputs']=dict(path=str(raw),sha256=sha(raw));atomic(config,cfg)
    done=subprocess.run(['bash',str(KIT/'jobs/nm/run-task.sh'),'NM-cr-bind'],env=env,capture_output=True,text=True,timeout=45)
    assert done.returncode==75,(done.stdout,done.stderr)
    malformed=current(out,'NM-cr-bind',complete=False)
    assert malformed['status']=='PENDING' and malformed['nonce']!=calls[0]['nonce']
    binding=load(out/'attempts/NM-cr-bind'/malformed['nonce']/'input-binding.json')
    assert binding['status']=='PENDING' and binding['escalation']=='owner'
    assert len({r['nonce'] for r in calls})==len(calls)
    from kit.nm_text import parse_parts
    collection=current(out,'NM-cr-collect');parts=load(out/'attempts/NM-cr-collect'/collection['nonce']/'parts.json')
    parsed=parse_parts([Path(p).read_text() for p in parts])
    assert parsed['families']['cr']['status']=='PENDING/INCOMPLETE' and len(parts)<=4
    result=dict(rows=len(calls),wrapper_calls=len(calls)+1,pending=sum(c['returncode']==75 for c in calls),
                failed=0,malformed_pending=1,parts=len(parts),technical_synthetic=True,calls=calls)
    atomic(directory/'result.json',result);return result


def runnable(directory):
    """All 78 real wrapper rows with native B regeneration and CPU model work."""
    from scripts.cr_confirmation_fixture import build
    from kit.nm_common import sha
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    cr,inventory_path=build(directory/'fixture');out=Path(cr['out'])
    # Preserve the bound original MATH source, separate from fresh task schedules.
    out.mkdir();(out/'targets-math').symlink_to(directory/'fixture/targets-math',target_is_directory=True)
    config=directory/'config.json'
    atomic(config,dict(out=str(directory/'six-rows'),job_python=sys.executable,
        technical_synthetic=True,infrastructure_update=authority_sha256(),cr_confirmation=cr))
    authority_before=sha(KIT/'resources/v6/infrastructure-update.json')
    env=dict(os.environ,KIT=str(KIT),NM_CONFIG=str(config),JOB_PY=sys.executable,PYTHONDONTWRITEBYTECODE='1')
    env.pop('UTIL_WATCH_ACTIVE',None)
    calls=[]
    for row in inventory():
        done=subprocess.run(['bash',str(KIT/'jobs/nm/run-task.sh'),row['id']],env=env,capture_output=True,text=True,timeout=240)
        assert done.returncode==0,(row['id'],done.returncode,done.stdout[-4000:],done.stderr[-4000:])
        record=current(out,row['id']);calls.append(dict(row=row['id'],returncode=done.returncode,status=record['status'],nonce=record['nonce']))
        atomic(directory/'progress.json',dict(rows=len(calls),calls=calls))
        print(row['id']+' COMPLETE',flush=True)
    binding=load(out/'confirmation-binding.json');proof=binding['finqa_reproduction']
    assert proof['status']=='COMPLETE' and all(c['byte_equal'] for c in proof['checks'])
    for seed in (104,105,106):
        assert sha(out/'targets-finqa'/f'schedule-{seed}.jsonl')==proof['regenerated'][str(seed)]['schedule_sha256']
        for arm in ('G','CR'):assert (out/'runs'/f'{arm}-s{seed}-finqa/complete.json').is_file()
    from kit.nm_text import parse_parts
    collection=current(out,'NM-cr-collect');parts=load(out/'attempts/NM-cr-collect'/collection['nonce']/'parts.json')
    parsed=parse_parts([Path(p).read_text() for p in parts])
    assert parsed['families']['cr']['status']=='TECHNICAL-ONLY'
    assert parsed['families']['cr']['metadata']['finqa']['reproduction']==proof
    assert sha(KIT/'resources/v6/infrastructure-update.json')==authority_before
    result=dict(rows=len(calls),complete=len(calls),parts=len(parts),technical_synthetic=True,
        finqa_runnable=True,finqa_reproduction=proof,authority_unchanged=True,calls=calls)
    atomic(directory/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--runnable',action='store_true');a=p.parse_args()
    result=(runnable if a.runnable else run)(a.out);print({k:v for k,v in result.items() if k not in ('calls','finqa_reproduction')})
