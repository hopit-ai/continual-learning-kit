#!/usr/bin/env python3
"""Registered P2 mechanics inside the development-only real verl integration exercise.

No import-time execution. Download, two-GPU training and full-state resumption
only happen when the explicitly prepared Modal script is launched by the owner.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import json,os,shutil,subprocess,sys
from kit.s_common import atomic,sha,require,ROOT
from kit import m2_train
from kit.x_p2_measure import runtime
from kit.x_smoke_checkpoint import dispose

# Already used by scripts/p4_sec_readouts_modal.py and phase-4 evidence.
MODEL_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def development_rows(tokenizer,finqa=None):
    from kit.x_smoke_inputs import converted_rows
    return converted_rows(tokenizer,finqa)


def build_command(model,data,run,steps=2,*,preparation=False,resume=False):
    a=NS(stage='legal',seed=101,gpu='0,1',sdpo=Path('/root/SDPO'),tokenizer=str(model),trainer_python=sys.executable)
    cmd,_,env=m2_train.command(a,'D+replay',str(model),data,run)
    changes={'trainer.n_gpus_per_node':'2','trainer.total_training_steps':str(steps),'trainer.save_freq':str(4 if preparation else -1 if resume else 2),
        'actor_rollout_ref.rollout.gpu_memory_utilization':'.35'}
    if preparation:changes['custom_reward_function.path']=str(ROOT/'kit/p3b_reward.py')
    cmd=[v.replace('c_sdft_entry.py','x_sdft_entry.py') for v in cmd]
    cmd=[v.split('=',1)[0]+'='+changes.get(v.split('=',1)[0].lstrip('+'),v.split('=',1)[1]) if '=' in v else v for v in cmd]
    # This key is absent from the recipe list, so append rather than replacing.
    cmd.append('actor_rollout_ref.rollout.response_length=128')
    cmd=[v for v in cmd if not v.startswith('actor_rollout_ref.rollout.enforce_eager=')]+['actor_rollout_ref.rollout.enforce_eager=True']
    # Same production validators/loss/worker/actor; only the existing short smoke
    # profile and an X-only development boundary (2 of 3) are selected.
    env.update(X_P2_DETERMINISTIC_SMOKE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='101',V4_PROFILE='technical-smoke',V4_TELEMETRY='0',X_INTEGRATION_SMOKE='development-only')
    for key in ('V4_PROFILE','V4_TELEMETRY','X_INTEGRATION_SMOKE','X_P2_DETERMINISTIC_SMOKE','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED'):
        prefix='+ray_kwargs.ray_init.runtime_env.env_vars.'+key+'='
        cmd=[v for v in cmd if not v.startswith(prefix)]+[prefix+json.dumps(env[key])]
    cmd.append('+actor_rollout_ref.model.override_config.attn_implementation=eager')
    return cmd,env


def stage_command(stage,model,data,run,*,batch=None,anchor=None,value=0.,boundary=None):
    require(stage in ('preparation','measure','control','p2','prefix','resume'),'unknown smoke stage')
    cmd,env=build_command(model,data,run,steps=3 if stage in ('prefix','resume') else 2 if stage=='measure' else 4,
        preparation=stage=='preparation',resume=stage=='resume')
    if stage in ('preparation','measure','control','p2'):
        if stage in ('control','p2'):cmd=[v if not v.startswith('trainer.save_freq=') else 'trainer.save_freq=4' for v in cmd]
        if stage=='measure':cmd=[v if not v.startswith('trainer.save_freq=') else 'trainer.save_freq=-1' for v in cmd]
        mode={'preparation':'prepare','p2':'train'}.get(stage,stage)
        cmd=runtime(env,cmd,mode=mode,batch=batch,run=run,anchor=anchor,seed=101,value=value)
        if stage=='p2':
            env['X_P2_EARLY_VERDICT']='1'
            cmd.append('+ray_kwargs.ray_init.runtime_env.env_vars.X_P2_EARLY_VERDICT='+json.dumps('1'))
        if stage!='measure':cmd=smoke_role(env,cmd,{'preparation':'prepare'}.get(stage,stage))
    else:
        values=dict(M2_ADDON_BOUNDARY=str(boundary))
        if stage=='prefix':values['M2_ADDON_PREFIX']='1'
        env.update(values)
        cmd += ['+ray_kwargs.ray_init.runtime_env.env_vars.'+key+'='+json.dumps(val) for key,val in values.items()]
        if stage=='resume':
            cmd=[v for v in cmd if not v.startswith(('trainer.resume_mode=','trainer.resume_from_path='))]
            cmd += ['trainer.resume_mode=resume_path','trainer.resume_from_path='+str(run/'sdft/global_step_2')]
    return cmd,env


def execute(cmd,env,run):
    run.mkdir(parents=True,exist_ok=True);atomic(run/'command.json',cmd)
    subprocess.run(cmd,cwd='/root/SDPO',env=env,check=True)


def run(root,part,monitor=None):
    from kit.x_smoke_settings import PART_STEMS
    require(part in PART_STEMS,"part must be p2 or resume")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import pyarrow as pa,pyarrow.parquet as pq
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    base=root/'base';snapshot_download('Qwen/Qwen3-1.7B',revision=MODEL_REVISION,local_dir=base)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    from kit.p_smoke import frozen_inputs
    _,finqa,_=frozen_inputs(tokenizer,ROOT/'kit/resources/p/smoke3-settings.json')
    # Separate containers share only pinned inputs, never mutable checkpoints.
    # Resume tests restoration mechanics from the untouched base; no P2 cache,
    # calibration, preparation or regulariser extension is needed for that part.
    incoming=prepare(root,base,tokenizer,finqa) if part=='p2' else base
    if part=='resume':
        atomic(root/'frozen-inputs.json',dict(base_revision=MODEL_REVISION,seed=101,
            settings_sha256=sha(ROOT/'kit/resources/p2/smoke.json'),finqa_ids=[r['id'] for r in finqa[:128]]))
    from kit.x_smoke_inputs import write_data
    rows=development_rows(tokenizer,finqa);data=write_data(root/'data',rows)
    frozen=read_json(root/'frozen-inputs.json');frozen['paired_manifest_sha256']=sha(data/'manifest.json')
    atomic(root/'frozen-inputs.json',frozen)
    partial(root,'development-inputs',dict(model_revision=MODEL_REVISION,model_config_sha256=sha(base/'config.json'),frozen_inputs=frozen))
    if part=='resume':return run_resume(root,incoming,data,rows,tokenizer,base)
    return run_p2(root,incoming,data,base,monitor)


def run_p2(root,incoming,data,base,monitor):
    # Real 1280-row development schedule; only the explicit short dose is run.
    batch=root/'p2/s101';calibration=root/'p2/calibration.json'
    cmd,env=stage_command('measure',incoming,data,root/'measure',batch=batch,anchor=base)
    execute(cmd,env,root/'measure')
    measured=[json.loads((batch/f'measure-r{rank}.json').read_text()) for rank in range(2)]
    require(all(r['state_before']==r['state_after'] and r['optimizer_steps']==r['scheduler_steps']==r['ema_steps']==0 for r in measured),'X4 mutated state')
    g,d=measured[0]['task_grad_norm'],measured[0]['incoming_distance'];require(g>0 and d>0,'development calibration norms invalid')
    partial(root,'X4-measure',dict(measurement=measured))
    # NOT the registered five-seed calibration; cannot pass x_p/calibration/read.
    atomic(calibration,dict(schema='development-only-one-seed',variant='p2-l2-init',technical_synthetic=True,
        seeds={'101':dict(task_grad_norm=g,incoming_distance=d)}))
    coefficient=dict(lambda_value=.1*g/d,task_grad_norm=g,incoming_distance=d,frozen_before_paired_updates=True)
    exports={};live={}
    for mode,name in (('control','control'),('train','p2-train')):
        target=root/name;cmd,env=stage_command('control' if mode=='control' else 'p2',incoming,data,target,
            batch=batch,anchor=base,value=.1*g/d if mode=='train' else 0.)
        execute(cmd,env,target)
        checkpoint=read_json(target/'checkpoint-memory-step4.json')
        require(checkpoint['status']=='PASS','missing checked final checkpoint')
        if mode=='train':
            p2=[read_json(target/f'p2-rank{rank}.json') for rank in range(2)]
            require(all(r['completed_updates']==4 and len(r['updates'])==4 for r in p2),'P2 hook did not run four times')
            p2_smoke=read_json(root/'stage-parts/P2-smoke-result.json')['p2_smoke']
            require(p2_smoke['verdict_boundary']=='after-update-4-before-native-checkpoint','missing pre-save P2 verdict')
            partial(root,'X6-P2',dict(checkpoint=checkpoint,updates=4,exported=p2_smoke['exported_distances']['p2']))
            dispose(target/'sdft/global_step_4');dispose(target/'live-step4');dispose(root/'export-p2-live')
            continue
        from kit.x_p2_export import merge,distance as export_distance
        live[name]=[read_json(target/f'smoke-rank{rank}.json') for rank in range(2)]
        exported=merge(target,root/('export-'+name),env)
        exports['control']=export_distance(exported,base,live[name][0]['geometry']['inventory'])
        exports['incoming']=export_distance(incoming,base,live[name][0]['geometry']['inventory'])
        dispose(target/'sdft/global_step_4');dispose(exported)
        partial(root,'X5-control',dict(checkpoint=checkpoint,updates=4,exported=exports['control']))
        from kit.p2_smoke import code_hashes
        from kit.x_smoke_memory import sample
        if monitor is not None:include_checkpoint_samples(root,monitor)
        atomic(root/'p2/pre-save-context.json',dict(schema='p2-pre-save-context.v1',base=str(base),exports=exports,
            preparation=[read_json(root/'preparation'/f'smoke-rank{r}.json') for r in range(2)],control=live['control'],
            coefficient=coefficient,container_host=monitor.sample() if monitor is not None else sample(),code_sha256=code_hashes()))
    return dict(passed=p2_smoke['passed'],mechanics_passed=p2_smoke['mechanics_passed'],resources_passed=p2_smoke['resources_passed'],
        part='p2',p2_smoke=p2_smoke,technical_synthetic=True,scientific_evidence=False,
        model='Qwen/Qwen3-1.7B',model_revision=MODEL_REVISION,model_config_sha256=sha(base/'config.json'),
        cases=['four-update FinQA D+replay preparation','X4 no-state-mutation','X5 cached control, four updates',
            'X6 P2, four updates','distributed P2 mechanical gates; verdict before native final save'],measurement=measured,p2_training=p2)


def resume_rows(rows,tokenizer):
    for row in rows[64:]:
        if row['extra_info']['p3b_replay']:
            row['extra_info']['demonstration']='two'
            row['extra_info']['p3b_target_ids']=tokenizer.encode('two',add_special_tokens=False)+[tokenizer.eos_token_id]
    return rows


def run_resume(root,incoming,data,rows,tokenizer,base):
    from kit.x_smoke_inputs import write_data
    drift=root/'drift';boundary=drift/'prefix.json'
    cmd,env=stage_command('prefix',incoming,data,drift,boundary=boundary)
    execute(cmd,env,drift)
    require(json.loads(boundary.read_text())['update']==2,'drift prefix did not stop at update 2')
    partial(root,'A-drift-prefix2',dict(prefix=read_json(boundary),checkpoint=read_json(drift/'checkpoint-memory-step2.json')))
    # Change only the unconsumed replay targets, retaining IDs/task positions and
    # data-loader geometry; real checked_resume qualifies all saved live state.
    resume_rows(rows,tokenizer)
    write_data(data,rows)
    cmd,env=stage_command('resume',incoming,data,drift,boundary=boundary)
    execute(cmd,env,drift)
    restored=json.loads((drift/'restore-qualified.json').read_text())
    require(restored['status']=='PASS' and restored['before_update']==3,'drift resume unqualified')
    dispose(drift/'sdft/global_step_2');dispose(drift/'sdft/global_step_3')
    partial(root,'A-drift-resume3',dict(restore=restored,final_update=3,final_checkpoint='not saved; live update-2 restoration qualification retained'))
    return dict(passed=True,mechanics_passed=True,part='resume',a_resume_status='VERIFIED',technical_synthetic=True,scientific_evidence=False,
        model='Qwen/Qwen3-1.7B',model_revision=MODEL_REVISION,model_config_sha256=sha(base/'config.json'),
        cases=['A drift prefix 2 / new replay / checked resume 3'],restore=restored)



def smoke_role(env,cmd,role):
    env['X_P2_SMOKE_ROLE']=role
    return [*cmd,'+ray_kwargs.ray_init.runtime_env.env_vars.X_P2_SMOKE_ROLE='+json.dumps(role)]


def prepare(root,base,tokenizer,finqa):
    from kit.x_smoke_inputs import converted_rows,write_data
    from kit.x_p2_export import merge
    data=write_data(root/'preparation-data',converted_rows(tokenizer,finqa,preparation=True))
    atomic(root/'frozen-inputs.json',dict(preparation_manifest_sha256=sha(data/'manifest.json'),settings_sha256=sha(ROOT/'kit/resources/p2/smoke.json'),
        base_revision=MODEL_REVISION,seed=101,finqa_ids=[r['id'] for r in finqa[:128]]))
    run=root/'preparation';cmd,env=stage_command('preparation',base,data,run,batch=root/'p2/s101',anchor=base)
    execute(cmd,env,run)
    incoming=merge(run,root/'development-incoming',env);dispose(run/'sdft/global_step_4')
    partial(root,'four-update-preparation',dict(checkpoint=read_json(run/'checkpoint-memory-step4.json'),incoming=str(incoming)))
    return incoming


def read_json(path):return json.loads(Path(path).read_text())


def include_checkpoint_samples(root,monitor):
    for path in Path(root).glob('*/checkpoint-memory-step*.json'):
        checkpoint=read_json(path)
        for key in ('host_before','host_after'):
            if key in checkpoint:monitor.include(checkpoint[key])


def partial(root,stage,record):
    value=dict(schema='x-integration-stage-partial.v1',stage=stage,**record)
    atomic(Path(root)/'stage-parts'/(stage+'.json'),value)
    print('[record-partial] '+json.dumps(value,sort_keys=True),flush=True)


def reserve_attempt(directory,part,*,ledger=None):
    """Atomically consume per-part authority, including lost-client launches."""
    import re
    from kit.x_smoke_settings import PART_STEMS,AUTHORISED_ATTEMPTS
    require(part in PART_STEMS,'part must be p2 or resume')
    stem=PART_STEMS[part];directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    recorded=set()
    for path in directory.glob(stem+'*.json'):
        match=re.fullmatch(re.escape(stem)+r'\.attempt(\d+)(?:\.claim)?\.json',path.name)
        if match:recorded.add(int(match[1]))
        elif path.name==stem+'.claim.json':recorded.add(1)  # original lost-client claim
        else:raise ValueError('unrecognised integration attempt evidence: '+path.name)
    require(recorded==set(range(1,len(recorded)+1)),'integration attempt history has gaps')
    n=len(recorded)+1;authorised=AUTHORISED_ATTEMPTS[part]
    require(n<=authorised,'authorised integration attempts for '+part+' exhausted; separate authority required')
    if part=='p2' and n==5:
        require(ledger is not None,'attempt 5 requires its exact ledger table row with the ≤$15 ceiling')
    if ledger is not None:
        require(re.search(re.escape(f'{stem} attempt {n}')+r'(?!\d)',ledger) is not None,
            f'separate single-attempt ledger bound required for {stem} attempt {n} before launch')
        if part=='p2' and n in (4,5):
            rows=[list(map(str.strip,line.strip().strip('|').split('|'))) for line in ledger.splitlines() if line.startswith('|')]
            from kit.x_smoke_settings import ATTEMPT4_LEDGER_ROW,ATTEMPT5_LEDGER_ROW
            expected=ATTEMPT4_LEDGER_ROW if n==4 else ATTEMPT5_LEDGER_ROW
            require(any(tuple(row)==expected for row in rows),
                f'attempt {n} requires its exact ledger table row with the ≤$15 ceiling')
        from kit.x_smoke_settings import forecast
        forecast(part,n)  # qualify the prospective bound BEFORE consuming the claim
        if part=='p2' and n>=3:
            from kit.x_smoke_settings import check_stage_audit
            if n==5:
                from kit.x_smoke_settings import ATTEMPT5_STAGE_AUDIT
                check_stage_audit(ATTEMPT5_STAGE_AUDIT)
            else:check_stage_audit()
    claim=directory/(stem+f'.attempt{n}.claim.json')
    try:
        with claim.open('x') as f:
            json.dump(dict(attempt=n,part=part,stem=stem,status='RESERVED',total_attempt_cap=authorised),f)
            f.flush();os.fsync(f.fileno())
    except FileExistsError:raise ValueError(f'integration attempt {n} for {part} already reserved')
    return n,stem


def verdict_scope(record,partials,part):
    """Only this part's P2 result decides P2; the resume part cannot qualify it."""
    from kit.x_smoke_settings import PART_STEMS
    require(part in PART_STEMS,'part must be p2 or resume')
    record.update(part=part,partial_records=partials,p2_qualified=False,
        a_resume_status='VERIFIED' if part=='resume' and record.get('passed') and record.get('restore',{}).get('status')=='PASS' else 'UNVERIFIED')
    if part=='resume':
        record.pop('p2_smoke',None)
        return record
    passed=[p['p2_smoke'] for p in partials if p.get('stage')=='P2-smoke-result' and p.get('p2_smoke',{}).get('passed') is True]
    if passed:
        proof=passed[-1]
        require(not record.get('p2_smoke') or record['p2_smoke']==proof,'P2 result changed after publication')
        record.update(p2_smoke=proof,p2_qualified=True)
        if not record.get('passed'):record['failure_scope']='after-P2-verdict'
    return record


def import_check(part):
    """Check CPU imports and shipped inputs before any download/model/GPU work.

    Trainer construction (ray/verl/vLLM) is beyond that boundary. Import its
    kit hooks here where possible; no worker, model or trainer is constructed.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import pyarrow as pa,pyarrow.parquet as pq
    import torch
    from kit.c_train import training_rows
    from kit.p_smoke import frozen_inputs
    from kit import c_data,c_reward,x_sdft_entry,x_resume
    from kit import x_p2_runtime,x_p2_smoke_runtime,sdft_actor,p3b_loss
    from kit import x_ema,x_p2_actor,x_p2_resources,v4_run,v4_state,v4_retry,v4_timing
    from kit.x_smoke_settings import PART_STEMS
    require(part in PART_STEMS,'part must be p2 or resume')
    frozen_inputs(settings_path=ROOT/'kit/resources/p/smoke3-settings.json')
    # Command construction reads the shipped recipe and resolves kit entry,
    # dataset and reward paths. It does not execute the trainer.
    cmd,_=build_command(ROOT/'base',ROOT/'data',ROOT/'import-check',resume=part=='resume')
    require(Path(cmd[1]).is_file(),'container trainer entry missing')
    for value in cmd:
        if value.startswith(('data.custom_cls.path=','custom_reward_function.path=')):
            require(Path(value.split('=',1)[1]).is_file(),'container dataset/reward module missing')
    json.loads((ROOT/'kit/resources/p2/smoke.json').read_text())
    if part=='p2':
        from torch.distributed.tensor import DTensor
        from safetensors import safe_open
        from safetensors.torch import save_file
        from kit.x_p2_export import merge,distance
        from kit.x_p2_smoke_read import evaluate
        from kit.x_p2_verdict import publish_before_checkpoint
        from kit.p2_smoke import code_hashes
        code_hashes()  # Includes the launcher copied for provenance, not import.
    print('[import-check] '+part+' OK',flush=True)
    return 0


def main(argv=None):
    import argparse
    from kit.smoke_steps import Steps
    from kit.x_smoke_memory import Monitor,gate
    from kit.x_smoke_settings import MEMORY_GIB
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=('p2','resume'),required=True)
    parser.add_argument('--import-check',action='store_true',help='check CPU imports and shipped inputs without downloads or model/GPU work')
    args=parser.parse_args(argv);part=args.part;root=Path('/tmp/x-integration-'+part)
    if args.import_check:return import_check(part)
    monitor=Monitor();monitor.__enter__()
    print('[record-partial] '+json.dumps(dict(stage='container-start',part=part,host=monitor.initial)),flush=True)
    steps=Steps({});steps.plan(['real-X-integration-'+part])
    try:value=run(root,part,monitor)
    except Exception as error:value=dict(passed=False,technical_synthetic=True,error=type(error).__name__+': '+str(error))
    monitor.__exit__()
    # Checkpoint boundary samples supplement the continuous container sampler,
    # including sub-interval save spikes observed in the trainer process.
    include_checkpoint_samples(root,monitor)
    value['container_host_peak']=monitor.peak
    value['host_gate']=gate(monitor.peak,MEMORY_GIB*1024**3)
    value['mechanics_passed']=value.get('mechanics_passed',bool(value['passed']))
    value['resources_passed']=bool(value.get('resources_passed',True) and value['host_gate']['passed'])
    value['passed']=bool(value['passed'] and value['resources_passed'])
    parts=[read_json(p) for p in sorted((root/'stage-parts').glob('*.json'))]
    value=verdict_scope(value,parts,part)
    steps.done(**value);return 0 if value['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
