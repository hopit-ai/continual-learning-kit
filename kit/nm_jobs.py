#!/usr/bin/env python3
"""Explicit next-message rows and real allocation wrapper; operator owns sbatch."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.nm_common import *
from contextlib import nullcontext,contextmanager
import argparse
import fcntl
import socket
import time
import uuid


def inventory(*, confirmation=False):
    rows=[]
    def add(family,name,phase,deps=(),gpus=0,**kwargs):
        ident='NM-'+family+'-'+name
        row=dict(id=ident,family=family,group='nm',phase=phase,gpus=gpus,nodes=1,cpus=32 if gpus==8 else 8,
            may_share_node=gpus!=8,depends_on=[dict(id=d,type='afterany' if phase in ('read','account') else 'afterok') for d in deps],
            dependency_options=['--kill-on-invalid-dep=yes'],
            command=['{job_python}','-I','{kit}/nm_jobs.py','run','--config','{nm_config}','--task',ident],
            lane_concurrency=3 if phase=='score' and kwargs.get('panel') not in ('knowledge','hellaswag') else 1,**kwargs)
        if phase=='score':row['expected_score_cells']=7 if family=='v2b' and kwargs.get('panel') not in (*GENERAL,'hellaswag') else 12 if kwargs.get('panel')=='legal' or (family=='grpo' and kwargs.get('panel')=='finqa') else 6
        if phase in ('cohort','arrival'):row['expected_score_cells']=2
        rows.append(row);return ident
    for family in FAMILIES:
        binding=add(family,'bind','bind')
        prep=add(family,'prepare','prepare',[binding])
        frontier=[prep]
        if family=='v2b':
            for month in ('july','august'):
                context=add(family,month+'-contexts','contexts',frontier,month=month)
                arrival=add(family,month+'-arrival','arrival',[context],1,month=month)
                frontier=[add(family,month+'-seal','ledger',[arrival],month=month)]
            frontier=[add(family,'schedule','schedule',frontier)]
        if family=='opr':
            frontier=[]
            for seed in SEEDS:
                q=add(family,f'generation-qualify-s{seed}','generation-qualify',[prep],1,seed=seed)
                gen=add(family,f'generate-s{seed}','generate',[q],1,seed=seed)
                cohort=add(family,f'cohort-s{seed}','cohort',[prep],1,seed=seed)
                frontier.append(add(family,f'refresh-s{seed}','refresh',[gen,cohort],seed=seed))
        if family=='fisher':
            pre=add(family,'estimator-preflight','estimator-preflight',[prep],8)
            estimator=add(family,'estimate','estimate',[pre],8)
            estimator=add(family,'estimator-publish','estimator-publish',[estimator])
            cals=[]
            for seed in SEEDS:
                cp=add(family,f'calibrate-s{seed}-prepare','calibrate-prepare',[estimator],seed=seed,arm='C')
                cals.append(add(family,f'calibrate-s{seed}','calibrate',[cp],8,seed=seed,arm='C'))
            frontier=[add(family,'seal-calibration','seal-calibration',cals)]
        qualification=[]
        for arm in ('C','T'):
            qprep=add(family,'qualify-'+arm+'-prepare','qualify-prepare',frontier,arm=arm,seed=101)
            q=add(family,'qualify-'+arm,'qualify',[qprep],8,arm=arm,seed=101)
            qualification.append(add(family,'qualify-'+arm+'-read','qualify-read',[q],arm=arm,seed=101))
        exports=[]
        for seed in SEEDS:
            for arm in ('C','T'):
                name=f'{arm}-s{seed}'
                prepared=add(family,name+'-prepare','train-prepare',[*frontier,*qualification],seed=seed,arm=arm)
                trained=add(family,name+'-train','train',[prepared],8,seed=seed,arm=arm)
                exports.append(add(family,name+'-export','export',[trained],seed=seed,arm=arm))
        scores=[]
        panels=PANELS if family not in ('v2b','grpo') else (*GENERAL,'hellaswag','finqa') if family=='grpo' else (
            *GENERAL,'hellaswag','july-off','july-on','august-off','august-on','september-off','september-on','september-on-holdout')
        context=add(family,'september-contexts','contexts',frontier,month='september') if family=='v2b' else None
        for seed in SEEDS:
            seed_exports=[e for e in exports if e.endswith(f'-s{seed}-export')]
            for panel in panels:
                deps=seed_exports+([context] if context and panel not in (*GENERAL,'hellaswag') else [])
                p=add(family,f's{seed}-{panel}-prepare','score-prepare',deps,seed=seed,panel=panel)
                scores.append(add(family,f's{seed}-{panel}','score',[p],1,seed=seed,panel=panel))
        account=add(family,'account','account',scores)
        reading=add(family,'read','read',[account,*scores])
        add(family,'collect','collect',[reading])
    if confirmation:
        from kit.cr_jobs import inventory as cr_inventory
        rows.extend(cr_inventory())
    result=dict(schema=SCHEMA,rows=rows,seeds=SEEDS,one_node=True,
        authority='bound launch infrastructure_update; manager regenerated last',
        planning_allowances=dict(K_CR=1362,K_core=1307,K_archived=1028,CR=1565,I0=1270))
    if confirmation:result['confirmation']=dict(seeds=[104,105,106],arms=['G','CR'],comparison='CR-G',planning_allowance_gpu_hours=830,source_kit='059d6293')
    return result


def row_for(task):
    matches=[r for r in inventory(confirmation=task.startswith('NM-cr-'))['rows'] if r['id']==task];need(len(matches)==1,'unregistered row');return matches[0]


def current(out,task,*,complete=True):
    path=out/'state'/(task+'.json');ready(path.is_file(),'dependency absent: '+task)
    pointer=load(path);record=checked(dict(path=str(out/pointer['receipt']),sha256=pointer['sha256']))
    if complete:ready(record['status'] in ('COMPLETE','ARCHIVED UNREAD'),'dependency not complete: '+task)
    for name,hash_value in record.get('artifacts',{}).items():
        ready((out/name).is_file(),'dependency artifact absent: '+name)
        need(sha(out/name)==hash_value,'dependency artifact changed: '+name)
    return record


@contextmanager
def row_lock(out,task):
    path=out/'locks'/(task+'.lock');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise Pending('same row already running')
        yield


def run(config,task):
    cfg=load(config)
    if task.startswith('NM-cr-'):
        from kit.cr_jobs import settings
        cfg=settings(cfg)
    out=Path(cfg['out']);need(out.is_absolute(),'absolute output root required')
    row=row_for(task);synthetic=cfg.get('technical_synthetic') is True
    # Always start a new attempt. No historical assignment is an admission pin.
    nonce=uuid.uuid4().hex;attempt=out/'attempts'/task/nonce;attempt.mkdir(parents=True,exist_ok=False)
    started=time.time();record=dict(schema=SCHEMA,task=task,family=row['family'],phase=row['phase'],gpus=row['gpus'],
        nonce=nonce,hostname=socket.gethostname(),status='RUNNING',started=started,ended=None,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),technical_synthetic=synthetic,artifacts={})
    if row['family']=='cr':record.update({k:row[k] for k in ('arm','seed','stage','lane') if k in row})
    atomic(attempt/'receipt.json',record);code=1
    try:
        with row_lock(out,task):
            if not synthetic:
                ready(record['slurm_job_id'],'one scheduler allocation per row required')
                authority=cfg.get('infrastructure_update');ready(authority,'bound launch authority absent')
                from kit.v6_infra import verify
                verify(authority)
                need(os.environ.get('KIT_V6_AUTHORITY_SHA') in (None,authority),'conflicting launch authority')
                os.environ['KIT_V6_AUTHORITY_SHA']=authority
                # Tiny metadata only: bound CPU receipts contain model/data audits.
                if row['gpus']:
                    ready((out/'launch'/ (row['family']+'.json')).is_file(),'CPU family launch binding absent')
                    ready(cfg.get('send_approval'),'separate bound send approval absent')
                if (out/'launch'/(row['family']+'.json')).is_file():
                    bound=load(out/'launch'/(row['family']+'.json'));need(bound['config_sha256']==digest(launch_config(cfg,row['family'])),'family launch config changed')
                    need(bound['code']==source_seal(),'NM implementation changed after launch binding')
                    if row['family']=='cr':
                        from kit.cr_confirmation import code_seal
                        need(bound.get('cr_code')==code_seal(),'confirmation implementation changed after pre-data binding')
                    allowed=bound.get('hosts',[])
                    need(not allowed or socket.gethostname() in allowed,'node-local source hostname differs')
            for dep in row['depends_on']:
                if dep['type']=='afterok':current(out,dep['id'])
            archived=out/'bindings'/(row['family']+'.json')
            if row['phase']!='bind' and archived.is_file() and load(archived)['status']=='ARCHIVED UNREAD':
                record['status']='ARCHIVED UNREAD';code=0
            else:
                from kit.m3_manifest import allocated_gpus,allocation_reservation
                limit=None
                if row['gpus'] and not synthetic:
                    from kit.nm_budget import reserve
                    limit=reserve(out,cfg,row,record,attempt)
                ids=allocated_gpus(row['gpus']) if row['gpus'] and not synthetic else []
                env=dict(os.environ,NM_ATTEMPT=str(attempt),NM_ALLOCATION_NONCE=nonce,KIT_LANE_JOB=nonce,
                    NM_SYNTHETIC='1' if synthetic else '0',CUDA_DEVICE_ORDER='PCI_BUS_ID')
                if row['family']=='cr':env.update(KIT_M3_MODE='cr-confirmation',KIT_V6A='1',INFERENCE_PY=cfg['job_python'])
                else:env.pop('KIT_M3_MODE',None)
                for key in ('M2_ASSIGNMENT','M2_ASSIGNED_GPU_UUID','M2_FAMILY','M2_FAMILY_SEED','M3_GPU_LOCK_FDS','M3_NODE_POLICY'):
                    env.pop(key,None)
                if row['family']=='cr':env['M3_NODE_POLICY']=json.dumps(dict(nodes=cfg.get('hosts',[])))
                if not row['gpus'] or synthetic:env['CUDA_VISIBLE_DEVICES']=''
                env['UTIL_JOB']=task
                with allocation_reservation(ids) if ids else nullcontext(()) as fds:
                    env['M3_GPU_LOCK_FDS']=json.dumps(fds)
                    command=[cfg.get('job_python',sys.executable),'-I',str(KIT/'nm_execute.py'),'--config',str(Path(config).resolve()),'--task',task,'--attempt',str(attempt)]
                    need(env.get('UTIL_WATCH_ACTIVE')!='1','NM wrapper must own one fresh watcher per allocation')
                    command=[cfg.get('job_python',sys.executable),'-I',str(KIT/'util_watch.py'),'--out',str(attempt/'util.jsonl'),
                        '--expect-gpus',str(row['gpus']),'--',*command]
                    if synthetic:
                        # The real watcher uses its public simulator fixture, no NVML/CUDA.
                        env['PATH']=str(KIT/'jobs/nm/sim')+os.pathsep+env.get('PATH','')
                        env['CUDA_VISIBLE_DEVICES']=','.join(str(i) for i in range(row['gpus']))
                    from kit.a4_common import gpu_run,job_deadline
                    with job_deadline(limit) if limit else nullcontext():
                        code=gpu_run(command,env=env,pass_fds=fds).returncode
                record['status']='COMPLETE' if code==0 else 'PENDING' if code in (3,75,124,137,143,-9,-15) else 'FAILED'
                if record['status']=='PENDING':code=PENDING
                if (attempt/'output.json').is_file():
                    output=load(attempt/'output.json')
                    if output.get('status')=='ARCHIVED UNREAD':record['status']='ARCHIVED UNREAD'
                    record['artifacts']={str(p.relative_to(out)):sha(p) for p in [attempt/'output.json',*[
                        out/name for name in output.get('published',[])]]}
    except (Pending,FileNotFoundError) as error:
        record.update(status='PENDING',error=str(error));code=PENDING;print('PENDING:',error)
    except (KeyboardInterrupt,SystemExit) as error:
        record.update(status='PENDING',error='allocation interrupted: '+str(error));code=PENDING
    except Exception as error:
        record.update(status='FAILED',error=type(error).__name__+': '+str(error));code=2;print('STOP:',record['error'])
    finally:
        record.update(ended=time.time(),exit_code=code)
        # CPU collector reconciles actual scheduler end. No inferred GPU cost ceiling.
        record['process_gpu_hours']=0 if synthetic else row['gpus']*(record['ended']-started)/3600
        record['allocation_gpu_hours']=0 if synthetic or not row['gpus'] else None
        record['cost_status']='TECHNICAL-ONLY' if synthetic else 'PENDING scheduler end' if row['gpus'] else 'COMPLETE'
        atomic(attempt/'receipt.json',record)
        atomic(out/'state'/(task+'.json'),dict(receipt=str((attempt/'receipt.json').relative_to(out)),sha256=sha(attempt/'receipt.json')))
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('inventory','run'))
    p.add_argument('--confirmation',action='store_true',help='include the isolated seventh CR-G comparison');p.add_argument('--config',type=Path);p.add_argument('--task');p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.action=='inventory':
        if a.out:atomic(a.out,inventory(confirmation=a.confirmation))
        else:print(json.dumps(inventory(confirmation=a.confirmation),indent=1))
        return 0
    need(a.config and a.task,'config and task required');return run(a.config,a.task)
if __name__=='__main__':raise SystemExit(main())
