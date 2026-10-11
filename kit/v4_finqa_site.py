#!/usr/bin/env python3
"""Main-only native-host storage and retained phase-0 launch inputs; no downloads."""
from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit import v4_phase0_site as site
from kit.v4_phase0_storage import GIB,qwen3_parameters,snapshot_bytes

JOB_TMP='/tmp/finqa-tmp'
RAY_JOB_TMP='/tmp/finqa-ray'
LAUNCH='v4/report-inputs/launch-inputs-finqa.json'


def storage_environment(work,*,allocation=False):
    env=site.storage_environment(work,allocation=False)
    # Reuse reviewed cache routing; only short IPC scratch moves into job /tmp.
    env.update(RAY_TMPDIR=RAY_JOB_TMP,VLLM_RPC_BASE_PATH='/tmp')
    if allocation:env['TMPDIR']=JOB_TMP
    path=Path(work)/LAUNCH
    if path.exists():
        frozen=json.loads(path.read_text())['environment']
        for key in ('HF_HOME','HF_HUB_CACHE','HF_ASSETS_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE'):env[key]=frozen[key]
    return env


@contextmanager
def storage(work,*,allocation=False):
    env=storage_environment(work,allocation=allocation)
    old={k:os.environ.get(k) for k in env};old_temp=tempfile.tempdir
    os.environ.update(env);tempfile.tempdir=env['TMPDIR']
    try:yield env
    finally:
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        tempfile.tempdir=old_temp


def disk_requirement(launch):
    """Serial main retention peaks; restoration retires successful arms in order.

    Native rotation is disabled only in main. Slot saves block training until
    old state is retired; before-save admission bounds old/new coexistence at
    two full states, including interruption and incomplete replacement. Invalid
    failed directories cannot accumulate beside another slot. Restoration is
    first and retains at most one arm's legs (six PPO states with EMA or four
    SFT states), including killed and resumed directories. Failed/unverified
    arms stop before another starts. It retains ten exports and all 26 metadata
    paths across per-arm retirements. All 61
    exports (46 evaluated main, five smokes, ten incidental restoration) survive.
    BF16 merger writes directly to its target; shard reload needs RAM, not a disk
    copy. Reserve one GiB of metadata/extra/serialization per historical save
    path as well: protected checkpoint metadata survives tensor retirement.
    """
    from kit.v4_campaign import slots,ALL_ARMS
    from kit.v4_arms import ARM_TABLE
    initial=Path(launch['MODEL_DIR']);teacher=Path(launch['TEACHER_MODEL_DIR'])
    raw=(initial/'config.json').read_bytes();parameters=qwen3_parameters(json.loads(raw))
    runs=[]
    for job in slots():runs.append(dict(id=job['id'],arm=job['arm'],exports=2,retained_checkpoints=1))
    for arm in ALL_ARMS:
        runs.append(dict(id='q-'+arm+'-finqa',arm=arm,exports=1,retained_checkpoints=1))
        ppo=bool(ARM_TABLE[arm]['actor_subpath'])
        for leg,count,exports in (('uninterrupted',3 if ppo else 2,1),('killed',2 if ppo else 1,0),('resumed',1,1)):
            runs.append(dict(id='restoration-'+arm+'-'+leg,arm=arm,retained_checkpoints=count,exports=exports))
    restoration=[r for r in runs if r['id'].startswith('restoration-')]
    state_count=sum(r['retained_checkpoints'] for r in restoration)
    retained_exports=sum(r['exports'] for r in runs)
    smoke_paths=sum(2 if ARM_TABLE[a]['actor_subpath'] else 1 for a in ALL_ARMS)
    main_paths=sum(40 if ARM_TABLE[j['arm']]['actor_subpath'] else 2 for j in slots())
    # PPO can force a capacity-expiry checkpoint at any of its 40 updates.
    # Paths are step-numbered even across retries; failed state cannot collect
    # beside another slot. Keep protected metadata's full former 1-GiB reserve.
    historical_paths=state_count+smoke_paths+main_paths
    def peak(states,emas,exports,extra_paths):
        model=4*parameters*states;adam=8*parameters*states;ema=4*parameters*emas
        merged=2*parameters*exports;extra=GIB*extra_paths
        return dict(state_count=states,ema_state_count=emas,export_count=exports,
                    metadata_and_extra_path_count=extra_paths,
                    model_bytes=model,adam_moments_bytes=adam,ema_bytes=ema,
                    merged_export_bytes=merged,extra_and_serialization_bytes=extra,
                    total_bytes=model+adam+ema+merged+extra)
    restoration_arms={};exports_so_far=0;paths_so_far=0
    for arm in ALL_ARMS: # actual serial driver order; prior arms' exports/metadata remain
        legs=[r for r in restoration if r['arm']==arm]
        states=sum(r['retained_checkpoints'] for r in legs)
        exports_so_far+=sum(r['exports'] for r in legs);paths_so_far+=states
        restoration_arms[arm]=peak(states,states if ARM_TABLE[arm]['actor_subpath'] else 0,exports_so_far,paths_so_far)
    restoration_peak=max(restoration_arms.values(),key=lambda row:row['total_bytes'])
    peaks={'restoration':restoration_peak,
           'qualification':peak(2,2,len(ALL_ARMS)+sum(r['exports'] for r in restoration),state_count+smoke_paths),
           'scientific':peak(2,2,retained_exports,historical_paths)}
    dominant=max(peaks,key=lambda key:peaks[key]['total_bytes']);coexist=peaks[dominant]
    snapshots={role:{'path':str(path),'bytes':snapshot_bytes(path)} for role,path in (('initial',initial),('teacher',teacher))}
    models=max(100*GIB,sum(row['bytes'] for row in snapshots.values()));caches=30*GIB
    required=coexist['total_bytes']+models+caches
    return dict(schema='v4-finqa-disk.v2',initial_parameters=parameters,initial_config_sha256=hashlib.sha256(raw).hexdigest(),
                snapshots=snapshots,runs=runs,checkpoint_policy='evidence-bound main state retirement; exports retained',
                peaks=peaks,restoration_arms=restoration_arms,dominant_peak=dominant,retained_export_count=retained_exports,historical_checkpoint_path_count=historical_paths,
                peak_merge_bytes=coexist['total_bytes']-coexist['extra_and_serialization_bytes'],
                extra_and_serialization_bytes=coexist['extra_and_serialization_bytes'],model_cache_bytes=models,
                environment_and_compile_cache_bytes=caches,required_bytes=required,required_gib=(required+GIB-1)//GIB)


def frozen_environment(work):
    doc=json.loads((Path(work)/LAUNCH).read_text())
    from kit.v4_phase0_carry import verify_work
    ledger=verify_work(work)
    if doc['phase0_archive_sha256']!=ledger['phase0_import']['archive_sha256']:raise ValueError('host launch phase0 archive binding differs')
    from kit.v4_archive import Archive
    from kit.v4_phase0_carry import ARCHIVE_PATH
    with Archive(Path(work)/ARCHIVE_PATH) as source:
        original=source.json('v4/report-inputs/launch-inputs-phase0.json')
        for key in ('MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER','SDPO_DIR','FINQA_ROOT','V4_TEACHER_PYTHON'):
            if doc['environment'][key]!=original['environment'][key]:
                from kit.v4_finqa_policy import warn
                warn('retained input paths','path changed: '+key+'; byte identities remain verified',work)
        if doc['trainer_activate']!=str(Path(original['trainer_prefix'])/'bin/activate'):
            from kit.v4_finqa_policy import warn
            warn('trainer activation','retained trainer path differs',work)
    env=doc['environment']
    if env['WORK']!=str(Path(work).resolve()):raise ValueError('frozen destination outside WORK')
    if env['TMPDIR']!=JOB_TMP:
        from kit.v4_finqa_policy import warn
        warn('job temporary paths','frozen job TMPDIR differs',work)
    return {**env,**({'V4_KIT_TAG':os.environ['V4_KIT_TAG']} if 'V4_KIT_TAG' in os.environ else {})}


def planning_storage(work,launch=None):
    launch=launch or frozen_environment(work)
    from kit.v4_finqa_policy import observations,observe
    with observations(work):sockets=observe(site.socket_path_check,launch,site='socket paths') or {}
    disk=disk_requirement(launch)
    # Later allocations retain earlier state. Check additional free capacity,
    # avoiding a second reserve for bytes already occupying this same WORK.
    retained=sum(p.stat().st_size for row in disk['runs'] for p in (Path(work)/'runs'/row['id']).rglob('*') if p.is_file() and not p.is_symlink())
    additional=max(0,disk['required_bytes']-retained)
    minimum=(additional+GIB-1)//GIB
    return {**disk,**sockets,'tmpdir':launch['TMPDIR'],'retained_run_bytes':retained,
            'additional_required_gib':minimum,'advisory_only':True,'free_bytes':site.free_space(work,0)}


def inherit(work):
    """Reuse the bound return and live retained snapshots/venvs, never stage twice."""
    from kit.v4_phase0_carry import verify_work,ARCHIVE_PATH
    from kit.v4_archive import Archive
    from kit.v4_phase0_download import verify_download_receipt
    work=Path(work).resolve();ledger=verify_work(work)
    with Archive(work/ARCHIVE_PATH) as source:
        original=source.json('v4/report-inputs/launch-inputs-phase0.json')
        env=dict(original['environment'])
        download=source.json('v4/report-phase0/download/receipt.json')
        identity=source.json('v4/report-phase0/environment.json')
        models=source.json('v4/report-inputs/model-identities.json') if 'v4/report-inputs/model-identities.json' in source.files else None
    verify_download_receipt(work,doc=download,paths={role:Path(env[key]) for role,key in (('initial','MODEL_DIR'),('teacher','TEACHER_MODEL_DIR'))})
    for key in ('SDPO_DIR','FINQA_ROOT','QWEN3_8B_TOKENIZER','V4_TEACHER_PYTHON'):
        if not Path(env[key]).exists():raise ValueError('retained phase0 input missing: '+key)
    trainer=Path(original['trainer_prefix'])
    if not (trainer/'bin/activate').is_file():raise ValueError('retained phase0 trainer activation missing')
    env.update(storage_environment(work,allocation=True))
    # HF cache is the verified zero-GPU job's original cache even if main TASK_ROOT differs.
    hf=download['hf_home'];env.update(HF_HOME=hf,HF_HUB_CACHE=hf+'/hub',HF_ASSETS_CACHE=hf+'/assets',
        HUGGINGFACE_HUB_CACHE=hf+'/hub',TRANSFORMERS_CACHE=hf+'/hub')
    kit=Path(__file__).resolve().parent
    env.update(TASK_ROOT=os.environ.get('TASK_ROOT',str(work.parent)),WORK=str(work),KIT=str(kit),PYTHONPATH=str(kit.parent)+os.pathsep+env['SDPO_DIR'],
               V4_KIT_TAG=os.environ['V4_KIT_TAG'],V4_FINQA_CAMPAIGN='1',V4_PHASE0_MODE='0',
               V4_INFERENCE_COMMIT='phase0-package-sha256:'+identity['inference']['package_sha256'])
    doc=dict(schema='v4-finqa-host-launch.v1',phase0_archive_sha256=ledger['phase0_import']['archive_sha256'],
             trainer_activate=str(trainer/'bin/activate'),environment=env,download_receipt=download)
    registration=work/'v4/report-inputs/frozen-model-registration.json';registration.parent.mkdir(parents=True,exist_ok=True)
    from kit.runner import write_durably
    if models is not None and registration.exists() and json.loads(registration.read_text())!=models:raise ValueError('frozen model identity differs from phase0')
    if models is not None and not registration.exists():write_durably(registration,models)
    disk=planning_storage(work,env) # CPU refusal before preparation or GPU planning.
    path=work/LAUNCH;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        def comparable(value):
            value={**value,'environment':dict(value['environment'])}
            value['environment'].pop('V4_KIT_TAG',None)
            return value
        if comparable(json.loads(path.read_text()))!=comparable(doc):
            from kit.v4_finqa_policy import warn
            warn('host launch inputs','infrastructure environment differs from initial receipt; original evidence retained',work)
    else:
        with path.open('x') as handle:json.dump(doc,handle,indent=2);handle.write('\n')
    return doc,disk


def exports(work):
    """Recover retained login inputs in a fresh shell, preserving its owner tag."""
    env=dict(frozen_environment(work))
    if Path(env['KIT']).resolve()!=Path(__file__).resolve().parent:
        from kit.v4_finqa_policy import warn
        warn('host kit path','kit moved from frozen path; current source and original evidence retained',work)
        env['KIT']=str(Path(__file__).resolve().parent)
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env['SDPO_DIR']
    env.update(storage_environment(work,allocation=False))
    env['V4_PHASE0_MODE']='0'
    env.pop('V4_KIT_TAG',None) # supplied by the owner for this phase, never replay an old tag
    doc=json.loads((Path(work)/LAUNCH).read_text())
    env['V4_TRAINER_ACTIVATE']=doc['trainer_activate']
    return env


def check(work):
    """Host findings are warnings; frozen model/data bytes are verified by prepare."""
    from kit.v4_finqa_policy import observations,require,observe,confine
    with observations(work):
        require(not os.environ.get('SLURM_STEP_ID','').isdecimal(),'main payload should run in native batch process','host entry')
        job=os.environ.get('SLURM_JOB_ID','unknown')
        require(job.isdecimal(),'main host scheduler job ID unavailable','scheduler telemetry')
        launch=frozen_environment(work)
        for key,value in launch.items():
            if key not in ('PATH','PYTHONPATH','V4_KIT_TAG'):require(os.environ.get(key)==value,'main frozen host environment differs: '+key,'host environment')
        result=observe(site.job_tmp_check,launch,site='job temporary paths') or {'warning':'job temporary path observation incomplete'}
        from kit.runner import write_durably
        write_durably(confine(work,Path(work)/'k8b4/containment/allocations'/job/'host-check.json'),result)
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('inherit','exports','storage','check','disk'))
    p.add_argument('--work',type=Path,required=True);p.add_argument('--allocation',action='store_true');a=p.parse_args(argv)
    try:
        if a.operation=='inherit':
            doc,disk=inherit(a.work)
            print('# CPU disk requirement: '+str(disk['required_gib'])+' GiB; also check your quota')
            env=dict(doc['environment']);env.update(storage_environment(a.work,allocation=False))
            # Reuse the verified model cache, no second download.
            for key in ('HF_HOME','HF_HUB_CACHE','HF_ASSETS_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE'):env[key]=doc['environment'][key]
            env['V4_TRAINER_ACTIVATE']=doc['trainer_activate']
            print('\n'.join('export '+k+'='+shlex.quote(v) for k,v in sorted(env.items())))
        elif a.operation=='exports':print('\n'.join('export '+k+'='+shlex.quote(v) for k,v in sorted(exports(a.work).items())))
        elif a.operation=='storage':print('\n'.join('export '+k+'='+shlex.quote(v) for k,v in sorted(storage_environment(a.work,allocation=a.allocation).items())))
        else:print(json.dumps(check(a.work) if a.operation=='check' else planning_storage(a.work),indent=2))
    except (ValueError,KeyError,OSError) as exc:print('STOP: '+str(exc),file=sys.stderr);return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
