"""One dedicated eight-GPU first-update D pass, with no optimization step."""
from pathlib import Path
import json,os,subprocess,time
from types import SimpleNamespace as NS
from kit.x_common import m,atomic,bind,sha,require,ROOT,incoming,digest
from kit.p_determinism import DETERMINISTIC_ENV


def deterministic_runtime(env,cmd,seed):
    """Compatibility wrapper: X retains the unchanged core numerical policy."""
    return list(cmd)


def runtime(env,cmd,*,mode,batch,run,anchor,seed,value=0.):
    cmd=deterministic_runtime(env,cmd,seed)
    from kit.m2_paths import local
    anchor=local(anchor)
    values=dict(X_P2_MODE=mode,X_P2_BATCH=str(batch),X_P2_RUN=str(run),X_P2_ANCHOR=str(anchor),
        X_P2_LAMBDA=str(value),X_P2_SEED=str(seed))
    if mode not in ('measure','prepare'):
        values['X_P2_CALIBRATION']=str(Path(batch).parent/'calibration.json')
        values['X_P2_CALIBRATION_SHA']=sha(values['X_P2_CALIBRATION'])
    env.update(values)
    cmd=[v.replace('c_sdft_entry.py','x_sdft_entry.py') for v in cmd]
    for key,val in values.items():cmd.append('+ray_kwargs.ray_init.runtime_env.env_vars.'+key+'='+json.dumps(val))
    return cmd


def measure(a,doc,seed):
    from kit.x_train import targets,retain_interrupted
    from kit import m2_train
    run=a.out/'p2'/f'measure-s{seed}';batch=a.out/'p2'/f's{seed}'
    if a.standin:
        import torch
        scheduled,_=targets(doc,seed,None)
        batch.mkdir(parents=True,exist_ok=True)
        torch.save(dict(technical_synthetic=True,ids=[r['id'] for r in scheduled[:32]],
            response_mask=torch.ones(32,2),old_log_probs=torch.zeros(32,2),rollout_log_probs=torch.zeros(32,2)),batch/'actor-input.pt')
        atomic(batch/'batch.json',dict(schema='rung-p2-first-batch.v1',variant='p2-l2-init',sha256=sha(batch/'actor-input.pt'),
            schedule_sha256=digest(scheduled[:32]),technical_synthetic=True))
        for rank in range(1):
            torch.save(torch.get_rng_state(),batch/f'rng-r{rank}.pt')
            atomic(batch/f'measure-r{rank}.json',dict(schema='rung-p2-measure.v1',variant='p2-l2-init',
                task_grad_norm=2.+(seed-101)/10,incoming_distance=4.,rank=rank,
                state_before={'model':'unchanged','optimizer':'fresh','teacher':'incoming','scheduler':0},
                state_after={'model':'unchanged','optimizer':'fresh','teacher':'incoming','scheduler':0},
                rng_file_sha256=sha(batch/f'rng-r{rank}.pt'),optimizer_steps=0,scheduler_steps=0,ema_steps=0,
                after_reduction_unscaling=True,before_clipping=True,technical_synthetic=True))
        return
    require(a.sdpo and a.tokenizer and len(set(a.gpus.split(',')))==8,'P2 X4 needs pinned tokenizer/SDPO and eight distinct GPUs')
    require(subprocess.check_output(['git','-C',str(a.sdpo),'rev-parse','HEAD'],text=True).strip()==
        '7c457fc1b1f636ae794eb0362ba37d4743b06fbc','P2 SDPO pin changed')
    from transformers import AutoTokenizer
    m.identity(a.tokenizer,'student')
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    scheduled,audit=targets(doc,seed,tokenizer)
    converted,_=m2_train.convert(scheduled,tokenizer,'legal')
    data=a.out/'p2-data'/f's{seed}'
    if not (data/'manifest.json').exists():
        import pyarrow as pa,pyarrow.parquet as pq
        data.mkdir(parents=True,exist_ok=True)
        for part,rows in [('train',converted),('val',converted[:32]),('test',converted[:1])]:pq.write_table(pa.Table.from_pylist(rows),data/(part+'.parquet'))
        atomic(data/'manifest.json',m.files_manifest(data,['train.parquet','val.parquet','test.parquet'],rows=1280))
    m.frozen_files(data)
    bind(batch/'schedule.json',dict(first_batch_sha256=digest(scheduled[:32]),all_rows_sha256=digest(scheduled),slots=audit,
        incoming=m.checked_model(incoming(doc,seed),False),data_manifest_sha256=sha(data/'manifest.json')))
    args=NS(**dict(vars(a),stage='legal',seed=seed,gpu=a.gpus))
    cmd,_,env=m2_train.command(args,'D+replay',incoming(doc,seed)['model'],data,run)
    cmd=runtime(env,cmd,mode='measure',batch=batch,run=run,anchor=doc['p2_anchor']['model'],seed=seed)
    from kit.x_p2_feasibility import runtime_identity,identity
    cmd=runtime_identity(env,cmd,run/'node-identity.json',identity(a.out,doc,seed))
    bind(run/'command.json',cmd);retain_interrupted(run,'full')
    atomic(run/'started.json',dict(mode='measurement',seed=seed))
    start=time.monotonic()
    try:subprocess.run(cmd,cwd=a.sdpo,env=env,check=True)
    finally:atomic(run/'cost.json',dict(gpu_hours=8*(time.monotonic()-start)/3600,seconds=time.monotonic()-start))
