#!/usr/bin/env python3
"""Row operations for the existing runner. Never a campaign dispatcher."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit import v4_teacher as t
from kit.v4_campaign import TASKS, PANELS, ALL_ARMS, slots
from kit.v4_resources import resource

KIT=Path(__file__).resolve().parent
ROOT=KIT.parent


def require(value,message):
    from kit.v4_finqa_policy import require as check
    return check(value,message,'campaign operation')


def load(path): return json.loads(Path(path).read_text())


from kit.v4_phase0_timing import shared_limit

def run(argv,env=None):
    """Run exactly one existing tool, inheriting containment's device selection."""
    subprocess.run(argv,env={**os.environ,**(env or {})},check=True,timeout=int(os.environ.get('V4_COMMAND_TIMEOUT',str(shared_limit(1800)))))


def paths(work,phase):
    return Path(work)/'v4'/('report-'+phase)


def job_for(ident,phase):
    if phase=='main': return next(job for job in slots() if job['id']==ident)
    arm=ident[2:].rsplit('-',1)[0];task=ident.rsplit('-',1)[1]
    return {'id':ident,'arm':arm,'task':task,'seed':101,'incoming':None}


def prepare(work,phase):
    """Freeze original pools, panels and probes before any coverage filtering."""
    require(not (Path(os.environ['MODEL_DIR'])/'phase0-technical-only.json').exists(),'phase0 technical checkpoint cannot be an initialisation')
    if phase!='phase0':
        from kit.v4_finqa_data import prepare as finqa_prepare
        return finqa_prepare(work,phase)
    import pyarrow.parquet as pq
    from kit.v4_datasets import make_datasets
    base=paths(work,phase); base.mkdir(parents=True,exist_ok=True)
    shipping=Path(work)/'v4'/'datasets'
    if not shipping.exists(): make_datasets(Path(os.environ['SDPO_DIR']),shipping,Path(os.environ['FINQA_ROOT']))
    inputs=Path(work)/'v4'/'report-inputs'; inputs.mkdir(parents=True,exist_ok=True)
    source_map={
        'SDPO/data/preprocess.py':Path(os.environ['SDPO_DIR'])/'data/preprocess.py',
        'k8b-chemistry-templates.json':resource('k8b-chemistry-templates.json'),
    }
    for split in ('train','test'):
        source_map[f'SDPO/datasets/sciknoweval/chemistry/{split}.json']=Path(os.environ['SDPO_DIR'])/f'datasets/sciknoweval/chemistry/{split}.json'
        source_map[f'FinQA/{split}.json']=Path(os.environ['FINQA_ROOT'])/(split+'.json')
    for name,src in source_map.items():
        target=inputs/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(target.read_bytes()==src.read_bytes(),'frozen source changed: '+name)
        else:shutil.copyfile(src,target)
    shutil.copyfile(resource('frontier-yardstick-chemistry-openai/responses-chemistry.jsonl'),inputs/'frontier-chemistry.jsonl')
    for name in ('TOKEN80','CLEAN','REACTION','NONREACTION'):
        src=resource('v4-data-audit-'+name+'.ids')
        shutil.copyfile(src,inputs/src.name)
    shutil.copyfile(Path(os.environ['SDPO_DIR'])/'verl/utils/reward_score/feedback/mcq.py',inputs/'mcq.py')
    shutil.copyfile(resource('v4-heldout-prompt-identity.json'),inputs/'v4-heldout-prompt-identity.json')
    for task in ('chemistry','finqa'):  # Tagged phase-0 preparation; main returns above.
        rows=t.authors_rows(task)
        pool=[]
        for row in rows:
            info=row['extra_info']; index=str(info['index'])
            item={'id':'sciknoweval-train-'+index if task=='chemistry' else index,
                  'task':task,'question':info['problem'],'messages':row['prompt'],
                  'description':info['description'],'idx':index,'split':'train',
                  'gold':row['reward_model']['ground_truth']}
            item['family']=t.family_of(item); pool.append(item)
        t.check_pool_provenance(pool)
        frozen=sorted(pool,key=lambda i:t.sha('v4-fixed-probe|'+task+'|'+i['id']))[:20]
        t.write_jsonl(inputs/(task+'-pool.jsonl'),pool)
        t.write_jsonl(inputs/(task+'-probe.jsonl'),frozen)
        selected=pool[:20] if phase=='phase0' else pool
        t.write_jsonl(base/(task+'-pool.jsonl'),selected)
    t.atomic_json(inputs/'model-identities.json',{'teacher':t.local_identity(os.environ['TEACHER_MODEL_DIR']),
                  'initial':t.local_identity(os.environ['MODEL_DIR'],'Qwen/Qwen3-8B')})
    tokenizer=inputs/'tokenizer'; tokenizer.mkdir(exist_ok=True)
    from kit.v4_contract import TOKENIZER_FILES
    for name in TOKENIZER_FILES:
        shutil.copyfile(Path(os.environ['QWEN3_8B_TOKENIZER'])/name,tokenizer/name)
    for task,folder in (('chemistry','v4_chem'),('finqa','v4_finqa'),('chemistry_probe','v4_chem_probe')):
        rows=pq.read_table(shipping/'datasets'/folder/'test.parquet').to_pylist()
        t.write_jsonl(inputs/(task+'-panel.jsonl'),rows)


def identity_gate(work):
    """Verify the unchanged registered model revisions and weight shards."""
    t.register_finqa_models()
    identities=load(Path(work)/'v4/report-inputs/model-identities.json')
    from kit.v4_contract import INITIAL_8B_REVISION
    require(t.REGISTERED_TEACHER_REVISION and t.REGISTERED_TEACHER_WEIGHT_HASHES and t.INITIAL_8B_WEIGHT_HASHES,
            'registered model identity missing')
    for name,revision,hashes in (('teacher',t.REGISTERED_TEACHER_REVISION,t.REGISTERED_TEACHER_WEIGHT_HASHES),('initial',INITIAL_8B_REVISION,t.INITIAL_8B_WEIGHT_HASHES)):
        doc=identities[name]
        actual={k:v for k,v in doc['model_file_hashes'].items() if k.endswith(('.safetensors','.bin'))}
        require(doc['model_revision']==revision and actual==hashes,'registered model identity differs: '+name)
    t.atomic_json(paths(work,'main')/'model-identity-check.json', {'ok':True,'self_monitored':True})


def teacher(work,phase,task,rewriting=False):
    """Use the durable bounded teacher/rewrite CLIs with unchanged prompts/verifier."""
    base=paths(work,phase)
    directory=base/('rewrite' if rewriting else 'teacher')/task
    if phase=='main':reuse_sample(work,task,rewriting)
    model=os.environ['MODEL_DIR' if rewriting else 'TEACHER_MODEL_DIR']
    argv=[sys.executable,str(KIT/'v4_teacher.py'),'rewrite' if rewriting else 'generate',
          '--task',task,'--pool',str(base/(task+'-pool.jsonl')),'--out',str(directory),'--model',model,
          '--tokenizer-8b',os.environ['QWEN3_8B_TOKENIZER']]
    if rewriting: argv+=['--demonstrations',str(base/'teacher'/task)]
    from kit.v4_generate_shards import generate
    generate(work,phase,task,rewriting=rewriting)


def reuse_sample(work,task,rewriting=False):
    """Extend the owner's hash-bound qualification journal; never generate its attempts again."""
    role='rewrite' if rewriting else 'teacher'
    source=paths(work,'qualification')/role/task
    if not source.exists():
        from kit.v4_finqa_policy import warn
        warn('corpus reuse','qualification corpus unavailable; main generation will fill missing work',work)
        return
    require(t.read_jsonl(source/'pool.jsonl')==t.read_jsonl(paths(work,'main')/(task+'-pool.jsonl')),'qualification/full original pool changed')
    identity=load(Path(work)/'v4/report-inputs/model-identities.json')['initial' if rewriting else 'teacher']
    manifest=load(source/'manifest.json')
    require(manifest['model_revision']==identity['model_revision'] and manifest['model_file_hashes']==identity['model_file_hashes'],'qualification generator model identity changed')
    destination=paths(work,'main')/role/task
    copied={}
    for original in sorted(source.rglob('*')):
        if not original.is_file():continue
        relative=original.relative_to(source)
        if relative.parts[0] not in ('raw','timings') or original.suffix!='.json':continue
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(target.read_bytes()==original.read_bytes(),'qualification source journal changed')
        else:shutil.copyfile(original,target)
        copied[str(relative)]=t.sha(original.read_bytes())
    destination.mkdir(parents=True,exist_ok=True)
    t.atomic_json(destination/'qualification-journal.json', {'source':str(source.relative_to(work)), 'files':copied,'owner_proceed_required':False})


def coverage(work,phase):
    """Record full-pool coverage; low coverage warns and surviving questions train."""
    base=paths(work,phase); items=[]; demos=[]; rewrites=[]
    for task in TASKS:
        items+=t.load_pool(base/(task+'-pool.jsonl'))
        demos+=t.read_jsonl(base/'teacher'/task/'demonstrations.jsonl')
        rewrites+=t.read_jsonl(base/'rewrite'/task/'rewrites.jsonl')
    report=t.coverage(items,demos,rewrites=rewrites)
    if phase=='main':
        t.atomic_json(base/'coverage.json',report)
        require(report['gate_pass'],'coverage gate failed')
        return
    else:
        original=t.load_pool(Path(work)/'v4/report-inputs/finqa-pool.jsonl')
        from kit.v4_finqa_policy import enabled
        if enabled():require(items==original,'qualification/full original pool changed')
        elif items!=original:
            sample=report;report=t.coverage(original,demos,rewrites=rewrites);report['sample']=sample
        report.update(qualification_pool_size=len(items), scientific_gate_applicable=False)
        for task in TASKS:
            for corpus in ('teacher','rewrite'):
                raw=t.read_jsonl(base/corpus/task/'raw_outputs.jsonl')
                require(all(not r.get('input_over_cap') for r in raw), 'sample silent input truncation')
    t.atomic_json(base/'coverage.json',report)


def training_set(work,phase):
    base=paths(work,phase)
    target=base/'common'
    if not target.exists(): t.training_set(base/'teacher',base/'rewrite',target,t.load_student_tokenizer(os.environ['QWEN3_8B_TOKENIZER']))
    if phase=='main': require(load(target/'manifest.json')['coverage']['scientific_phase_allowed'],'unqualified common training set')


def schedule(work,phase,ident):
    """Shared seeded exposure order for every arm, task and order, with recorded smoke repetition."""
    import pyarrow.parquet as pq
    from kit import v4_sft
    job=job_for(ident,phase); base=paths(work,phase)
    task=job['task']; seed=job['seed']; target=base/'scheduled'/f'{task}-s{seed}'
    if target.exists(): return
    rows=pq.read_table(base/'common'/(task+'.parquet')).to_pylist()
    if not rows:raise RuntimeError('no surviving verified questions; cannot execute training schedule')
    ids=[str(r['extra_info']['index']) for r in rows]
    random.Random(seed).shuffle(ids)
    count=64 if phase in ('qualification','phase0') else 1280
    exposure=ids[:count] if len(ids)>=count else [ids[i%len(ids)] for i in range(count)]
    doc={'seed':seed,'ids':exposure,'steps':2 if phase in ('qualification','phase0') else 40,
         'profile':'technical-smoke' if phase in ('qualification','phase0') else 'scientific',
         'technical_padding':{'smoke_only':phase in ('qualification','phase0'),'accepted_unique_rows':len(ids),'repeated_exposures':max(0,count-len(ids))}}
    path=base/f'{task}-s{seed}-schedule.json';t.atomic_json(path,doc)
    v4_sft.main(['--training-set',str(base/'common'),'--task',task,'--schedule',str(path),
                 '--out',str(target),'--tokenizer-8b',os.environ['QWEN3_8B_TOKENIZER']])
    dataset=target/'dataset'; dataset.mkdir()
    shutil.copyfile(target/'train.parquet',dataset/'train.parquet')
    folder='v4_chem' if task=='chemistry' else 'v4_finqa'
    shutil.copyfile(Path(work)/'v4/datasets/datasets'/folder/'test.parquet',dataset/'test.parquet')


def train(work,phase,ident):
    """Launch the registered entry; never retrain an already complete valid result."""
    if phase!='phase0':
        from kit.v4_retention import sweep
        from kit.v4_finqa_policy import observe
        observe(sweep,work,starting_slot=ident,site='retention before training')
    job=job_for(ident,phase); base=paths(work,phase); scheduled=base/'scheduled'/f"{job['task']}-s{job['seed']}"
    out=Path(work)/'runs'/ident
    if (out/'run-summary.json').exists():
        doc=load(out/'run-summary.json')
        if doc.get('dose_complete'):
            require(doc['seed']==job['seed'] and doc['arm']==job['arm'] and doc['train_sha256']==t.sha((scheduled/'train.parquet').read_bytes()),'saved result changed')
            if doc.get('merged')!=1:
                merge(work,phase,ident)
            return
    retry=None
    if out.exists():
        from kit.v4_retry import local_retry, saved_state
        attempt=int(os.environ.get('V4_RUNNER_ATTEMPT','1'))
        require(attempt>1,'failed training needs a new append-only attempt')
        root=Path(work)/'campaign'/('v4-'+phase)/('train-'+ident)
        current=load(root/f'attempt-{attempt}/start.json')
        previous=load(root/f'attempt-{attempt-1}/start.json')
        retry=local_retry(work,'v4-'+phase,'train-'+ident,attempt,previous)
        valid=[]
        for checkpoint in out.glob('*/global_step_*'):
            try:valid.append((int(checkpoint.name.removeprefix('global_step_')),checkpoint,saved_state(checkpoint,job['arm'])))
            except ValueError:continue
        backup=base/'failed-training'/ident/f'attempt-{attempt-1}'
        backup.parent.mkdir(parents=True,exist_ok=True)
        if retry['state_policy']=='recover_completed_operation':
            from kit.v4_receipts import completed_updates
            final=2 if phase in ('qualification','phase0') else 40
            require(completed_updates(out)==final and valid and max(valid)[0]==final,'completed-state recovery lacks observed optimizer dose')
            backup.mkdir(parents=True,exist_ok=True)
            for path in out.iterdir():
                if path.is_file():shutil.copyfile(path,backup/path.name)
                elif path.name=='env':shutil.copytree(path,backup/'env',dirs_exist_ok=True)
            if not (out/'run-summary.json').exists():t.atomic_json(out/'run-summary.json',{'arm':job['arm'],'seed':job['seed'],'returncode':0})
            from kit.v4_run import summarize
            summarize({'WORK':str(work),'NAME':ident,'ARM':job['arm'],'SEED':str(job['seed']),
                'STEPS':str(final),'V4_PROFILE':'technical-smoke' if phase in ('qualification','phase0') else 'scientific',
                'DATA_MANIFEST':str(scheduled/'manifest.json'),'TRAIN_FILE':str(scheduled/'train.parquet')})
            merge(work,phase,ident)
            return
        if retry['state_policy']=='resume_valid_state':
            require(valid and max(valid)[2]==retry['saved_state_sha256'],'retry must use latest complete saved state')
            backup.mkdir(parents=True,exist_ok=True)
            for path in out.iterdir():
                if path.is_file():shutil.copyfile(path,backup/path.name)
                elif path.name=='env':shutil.copytree(path,backup/'env',dirs_exist_ok=True)
            # Worker telemetry appends rather than truncating. Observations after
            # the restored step belong to the preserved failed attempt, not the
            # resumed forty-update result.
            for name in ('training-tokens.jsonl','export-timings.jsonl'):
                path=out/'env'/name
                if path.exists():
                    t.write_jsonl(path,[r for r in t.read_jsonl(path) if r.get('step',0)<=retry['completed_updates']])
        else:
            require(retry['state_policy']=='no_valid_saved_state' and not valid,'valid saved state must be resumed, never reset')
            # Partial state stays in the append-only failed attempt archive.
            backup.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out),str(backup))
    model=str(Path(work)/'runs'/job['incoming']/'hf-step40') if job['incoming'] else os.environ['MODEL_DIR']
    require(not (Path(model)/'phase0-technical-only.json').exists(),'phase0 technical checkpoint cannot be an initialisation')
    incoming=t.local_identity(model,'Qwen/Qwen3-8B')
    identity_path=out/'env/incoming-checkpoint.json'
    if identity_path.exists():require(load(identity_path)==incoming,'incoming checkpoint changed')
    # Launchers refuse an existing run directory. Freeze the input separately
    # before launch, then copy its identity into the launcher's own env directory.
    frozen=base/'incoming-identities'/(ident+'.json')
    frozen.parent.mkdir(parents=True,exist_ok=True)
    if frozen.exists():require(load(frozen)==incoming,'incoming checkpoint changed')
    else:t.atomic_json(frozen,incoming)
    env={'ARM':job['arm'],'SEED':str(job['seed']),'NAME':ident,'WORK':str(work),'MODEL_DIR':model,
         'STEPS':'2' if phase in ('qualification','phase0') else '40','V4_PROFILE':'technical-smoke' if phase in ('qualification','phase0') else 'scientific',
         'V4_SAVE_FREQ':'2' if phase in ('qualification','phase0') else '20','NGPU':'8','TP':'2',
         'DATA_MANIFEST':str(scheduled/'manifest.json'),'TRAIN_FILE':str(scheduled/'train.parquet'),
         'VAL_FILE':str(scheduled/'val.parquet'),'DATASET':str(scheduled/'dataset'),
         'REWARD_FILE':str(KIT/'beds/v4_reward.py'),'V4_TELEMETRY':'1'}
    env.update(V4_CHECKPOINT_STATE='1',V4_RESUME_PATH=os.environ.get('V4_RESUME_PATH','') if retry else '',
               V4_RETRY_ADMITTED='1' if retry else '0')
    if phase!='phase0':env.update(V4_MAIN_RETENTION='slot',KEEP_TRAINER_CKPT='1')
    try:run(['bash',str(KIT/'run_v4.sh')],env)
    except subprocess.CalledProcessError:
        # Export failure cannot authorize a second optimizer dose. The launcher
        # keeps its completed checkpoint; recover only the missing merge.
        if (out/'run-summary.json').exists() and load(out/'run-summary.json').get('dose_complete'):
            merge(work,phase,ident)
        raise
    finally:
        if out.is_dir():t.atomic_json(identity_path,incoming)
        if retry and retry['state_policy']=='resume_valid_state' and (out/'metrics.jsonl').exists():
            # The pinned FileLogger truncates on initialization. Retain prior
            # steps through the saved point and append the resumed observations.
            step=retry['completed_updates'];old=t.read_jsonl(backup/'metrics.jsonl')
            new=t.read_jsonl(out/'metrics.jsonl')
            t.write_jsonl(out/'metrics.jsonl',[r for r in old if r.get('step',0)<=step]+[r for r in new if r.get('step',0)>step])


def merge(work,phase,ident):
    from kit.v4_finqa_policy import enabled,disk
    if enabled():
        from kit.v4_finqa_policy import writer_bytes
        disk(work,writer_bytes(work),'BF16 export')
    """Validate final merge and preserve/merge the step-20 scientific checkpoint."""
    from kit.v4_sft import complete_export
    job=job_for(ident,phase);out=Path(work)/'runs'/ident
    final=2 if phase in ('qualification','phase0') else 40
    for step in ((final,20) if phase=='main' else (final,)):
        if complete_export(out/f'hf-step{step}'):continue
        require(load(out/'env/optimizer-updates.json')['completed_optimizer_updates']==final,'cannot recover an incomplete training dose')
        from kit.v4_arms import checkpoint_path
        source=checkpoint_path(out,job['arm'],step,actor=True)
        require(source.is_dir(),f'step-{step} checkpoint missing')
        run([sys.executable,str(KIT/'v4_timing.py'),'--out',str(out/'env'/f'recovered-merge-step{step}.json'),'--',sys.executable,'-m','verl.model_merger','merge','--backend','fsdp','--local_dir',str(source),'--target_dir',str(out/f'hf-step{step}')])
        require(complete_export(out/f'hf-step{step}'),f'step-{step} merge incomplete')
    summary=load(out/'run-summary.json')
    if summary.get('merged')!=1:
        t.atomic_json(out/'run-summary.json',{**summary,'merged':1,'merge_returncode':0,'export_recovered':True})
    for step in (20,40) if phase=='main' else (2,):
        if phase=='phase0':
            from kit.v4_phase0 import label_checkpoint
            label_checkpoint(out/f'hf-step{step}')
        t.atomic_json(out/'env'/f'export-identity-step{step}.json',t.local_identity(out/f'hf-step{step}','Qwen/Qwen3-8B'))
        if phase!='phase0':
            from kit.v4_retention import verify_export
            verify_export(out,step)


def score(work,phase,checkpoint,task,cap):
    """Reload and score an entire fixed panel at B or H on the first scoring GPU."""
    base=paths(work,phase); out=base/'eval'/f'{checkpoint}-{task}-{cap}'
    model=os.environ['MODEL_DIR'] if checkpoint=='baseline' or checkpoint.startswith('agreement') else str(Path(work)/'runs'/checkpoint/('hf-step2' if phase in ('qualification','phase0') else 'hf-step40'))
    final=out/('forgetting.json' if task=='general-v1' else 'bed-score.json')
    if final.exists():
        # Reuse a completed score only when it parses and belongs to this checkpoint; otherwise it is moved
        # aside below as failed-attempt evidence and scored again. The report reader validates every score in
        # full afterwards; running that reader here needs scheduler evidence that does not exist mid-run.
        identity=t.local_identity(model,'Qwen/Qwen3-8B');stored=out/'checkpoint-identity.json'
        try:
            load(final)
            if not stored.exists():t.atomic_json(stored,identity)
            if load(stored)==identity:return
        except ValueError:pass
    if out.exists():
        # Scorers require a fresh destination. Keep all incomplete bytes with
        # the failed runner attempt, including engine-start diagnostics.
        from kit.v4_finqa_policy import confine
        previous=max(1,int(os.environ.get('V4_RUNNER_ATTEMPT','1'))-1)
        row=os.environ.get('V4_CURRENT_ROW') or f'score-{checkpoint}-{task}-{cap}'
        failed=confine(work,Path(work)/'campaign'/('v4-'+phase)/row/f'attempt-{previous}')
        saved=failed/'scoring-output';suffix=1
        while saved.exists():
            saved=failed/f'scoring-output-{suffix}';suffix+=1
        confine(work,out);failed.mkdir(parents=True,exist_ok=True)
        out.rename(saved)
    if task=='general-v1':
        argv=[sys.executable,str(KIT/'score_forgetting.py'),'generate','--model',model,'--out',str(out),'--max-new-tokens',str(cap),'--max-model-len','12288']
        run(argv,{'V4_TELEMETRY':'1'})
        t.atomic_json(out/'checkpoint-identity.json',t.local_identity(model,'Qwen/Qwen3-8B'))
        return
    root=Path(work)/'v4/datasets/datasets'/({'chemistry':'v4_chem','finqa':'v4_finqa','chemistry_probe':'v4_chem_probe'}[task])
    argv=[sys.executable,str(KIT/'eval_bed.py'),'generate','--bed','chemistry' if task=='chemistry_probe' else task,
          '--root',str(root),'--split','test','--model',model,'--out',str(out),
          '--max-new-tokens',str(cap),'--max-model-len','12288']
    if checkpoint.startswith('agreement'):argv+=['--limit','50']
    elif phase in ('qualification','phase0') and checkpoint!='baseline': argv+=['--limit','20']
    try:run(argv,{'V4_TELEMETRY':'1'})
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
        out.mkdir(parents=True,exist_ok=True)
        if not (out/'engine-status.json').exists():t.atomic_json(out/'engine-status.json',
            {'engine_ok':False,'configuration_ok':False,'failure_type':'scoring_engine_start','error':str(error)})
        raise
    t.atomic_json(out/'checkpoint-identity.json',t.local_identity(model,'Qwen/Qwen3-8B'))


def probe(work,phase,ident,task,step,conditioned=False):
    """Eight unaided responses per frozen training-only question at each saved point."""
    job=job_for(ident,phase)
    model=(str(Path(work)/'runs'/job['incoming']/'hf-step40') if job['incoming'] else os.environ['MODEL_DIR']) if step==0 else str(Path(work)/'runs'/ident/f'hf-step{step}')
    items=t.load_pool(paths(work,phase)/(task+'-pool.jsonl') if conditioned and phase in ('qualification','phase0') else Path(work)/'v4/report-inputs'/(task+'-probe.jsonl'))
    t.check_pool_provenance(items)
    require(not conditioned or step==0 and not job['incoming'],'conditioned probe must use the frozen initial 8B')
    suffix=f'conditioned-{task}' if conditioned else f'{ident}-{task}-{step}'
    out=paths(work,phase)/'probes'/suffix;out.mkdir(parents=True,exist_ok=True)
    demos={r['id']:r for r in t.read_jsonl(paths(work,phase)/'teacher'/task/'demonstrations.jsonl')} if conditioned else {}
    if conditioned and phase=='qualification':items=items[:20]
    original=items
    if conditioned:items=[i for i in items if i['id'] in demos]
    checkpoint=t.local_identity(model,'Qwen/Qwen3-8B')
    cache_key=t.sha(json.dumps({'weights':{k:v for k,v in checkpoint['model_file_hashes'].items() if k.endswith(('.safetensors','.bin'))},
        'pool':original,'demonstrations':demos,'conditioned':conditioned,'cap':2048,'temperature':.7,'top_p':.95,
        'attempts':8,'seed_rule':'v4-mechanism','tokenizer':t.tokenizer_identity(os.environ['QWEN3_8B_TOKENIZER'])},sort_keys=True))
    cache=Path(work)/'v4/probe-cache'/cache_key;cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists() and not (out/'raw_outputs.jsonl').exists():
        entry=load(cache);source=Path(work)/entry['source']
        require(source.resolve().is_relative_to(Path(work).resolve()),'probe cache source outside WORK')
        for name,digest in entry['files'].items():
            require(t.sha((source/name).read_bytes())==digest,'cached probe changed')
            shutil.copyfile(source/name,out/name)
        t.atomic_json(out/'reuse.json',{'cache_key':cache_key,**entry})
        return
    pending=[];keys=[]
    for item in items:
        for attempt in range(1,9):
            path=out/(t.sha(item['id'])+f'-{attempt}.json')
            if not path.exists():
                seed=int(t.sha('v4-mechanism|'+item['id']+'|'+str(attempt))[:8],16)%(2**31)
                messages=t.rewrite_messages(item,demos[item['id']]) if conditioned else t.student_messages(item)
                pending.append({'messages':messages,'seed':seed});keys.append((item,attempt,path,seed))
    if pending:
        engine=t.VLLMGenerator(model,2048,input_cap=6144 if conditioned else 4096,max_num_seqs=128,probe_mode=True)
        outputs=engine.generate_many(pending)
        for (item,attempt,path,seed),raw in zip(keys,outputs):
            t.atomic_json(path,{'id':item['id'],'task':task,'family':item['family'],'attempt':attempt,
                'seed':seed,'temperature':.7,'top_p':.95,'conditioned':conditioned,**raw})
        t.atomic_json(out/'timing.json',{'reload':engine.reload_timing})
    elif not (out/'timing.json').exists():
        t.atomic_json(out/'timing.json',{'reload':None,'empty_conditioned_probe':conditioned and not items})
    raw=[load(out/(t.sha(item['id'])+f'-{attempt}.json')) for item in items for attempt in range(1,9)]
    t.write_jsonl(out/'raw_outputs.jsonl',raw)
    t.atomic_json(out/'checkpoint-identity.json',checkpoint)
    from kit.eval_bed import machine_fingerprint
    t.atomic_json(out/'probe-identity.json',{'pool_sha256':t.sha(json.dumps(original,sort_keys=True)),
                  'conditioned':conditioned,'unavailable_ids':sorted(i['id'] for i in original if conditioned and i['id'] not in demos),
                  'batch_size':160,'max_num_seqs':128,'max_new_tokens':2048,'machine':machine_fingerprint(True)})
    if not cache.exists():
        from kit.runner import write_durably
        write_durably(cache,{'source':str(out.relative_to(Path(work))),
            'files':{name:t.sha((out/name).read_bytes()) for name in ('raw_outputs.jsonl','timing.json','checkpoint-identity.json','probe-identity.json')}})


def standalone(work):
    """Stop before any sequence slot when either actual D-101 acquisition/cost fails."""
    from kit.v4_archive import DirectoryEvidence
    from kit.v4_readers import scorer_context, scoring, run_record
    from kit import v4_analysis as analysis
    import tempfile
    archive=DirectoryEvidence(work)
    table={};fingerprints=set()
    from kit.v4_evidence import scheduler_rows, data_context, common_data, training_provenance
    from kit.v4_finqa_policy import observe
    observe(scheduler_rows,archive,'qualification');observe(scheduler_rows,archive,'main')
    with tempfile.TemporaryDirectory(prefix='v4-standalone-') as temp:
        tokenizer,mcq=scorer_context(archive,temp)
        with data_context(archive,tokenizer,mcq):
            common_data(archive,'main',tokenizer)
            for task in TASKS:
                ident='D-'+task+'-alone-s101'
                summary=run_record(archive,ident,'main')
                training_provenance(archive,'main',job_for(ident,'main'),summary,tokenizer)
                require(summary['arm']=='D' and summary['seed']==101,'wrong standalone identity')
                before=scoring(archive,'qualification','baseline',task,2048,mcq)
                after=scoring(archive,'main',ident,task,2048,mcq)
                fingerprints.update((before['fingerprint'],after['fingerprint']))
                transition=analysis.paired(before['verdicts'],after['verdicts'])
                table[task]={'baseline':before['strict'],'standalone':after['strict'],
                             'gain':transition['net'],'transitions':transition,
                             'acquisition_pass':transition['net']>=analysis.GAIN[task],
                             'cost_pass':analysis.cost_pass(after,before)}
                table[task]['pass']=table[task]['acquisition_pass'] and table[task]['cost_pass']
    report={'tables':table,'pass':len(fingerprints)==1 and all(row['pass'] for row in table.values()),
            'reason':None if len(fingerprints)==1 else 'cross-GPU/machine scoring'}
    t.atomic_json(paths(work,'main')/'D-standalone.json',report)
    require(report['pass'],'seed-101 D standalone acquisition or deployment cost failed')
    return report


def report(work,phase):
    """Write the shared raw-evidence reading before the collector's final archive.

    This row cannot attest its own runner verdict. The archive reader performs
    that final chronology check after collection, including PAUSE/FINISHED.
    Missing measurement/evidence remains an explicit technical/incomplete report.
    """
    from kit.v4_archive import DirectoryEvidence
    from kit.v4_readers import qualification_evidence, campaign_evidence
    paths(work,phase).mkdir(parents=True,exist_ok=True)
    evidence=DirectoryEvidence(work)
    if phase in ('qualification','phase0'):
        try: result=qualification_evidence(evidence)
        except (ValueError,KeyError,TypeError,OSError) as error:
            result={'status':'stop: technical','reasons':[str(error)],'tables':{},'owner_proceed_required':False}
    else:
        result=campaign_evidence(evidence)
    t.atomic_json(paths(work,phase)/'reading.json',result)
    from kit.v4_finqa_policy import warn,warnings,correctness,CorrectnessStop
    for reason in result.get('reasons',[]):
        if correctness(reason):raise CorrectnessStop(reason)
        warn('progress report',reason,work)
    result.update(status='progress report',owner_proceed_required=False,warnings=warnings(work))
    t.atomic_json(paths(work,phase)/'reading.json',result)
    return result


def launcher_env(work,arm,ident,scheduled,steps=2):
    return {'ARM':arm,'SEED':'101','NAME':ident,'WORK':str(work),'MODEL_DIR':os.environ['MODEL_DIR'],
        'STEPS':str(steps),'V4_PROFILE':'technical-smoke','V4_SAVE_FREQ':'2','NGPU':'8','TP':'2',
        'DATA_MANIFEST':str(scheduled/'manifest.json'),'TRAIN_FILE':str(scheduled/'train.parquet'),
        'VAL_FILE':str(scheduled/'val.parquet'),'DATASET':str(scheduled/'dataset'),
        'REWARD_FILE':str(KIT/'beds/v4_reward.py'),'V4_TELEMETRY':'1','V4_CHECKPOINT_STATE':'1',
        'PYTHONPATH':str(ROOT)+os.pathsep+os.environ['SDPO_DIR'],'KIT_FINISH_GATE':'1'}


def qualification_configs(work,phase="qualification",task="finqa"):
    """Resolve the actual launch argv in the pinned trainer, before all training."""
    import yaml
    from kit.v4_qualification import validate_configs
    base=paths(work,phase);out=base/'configs';out.mkdir(parents=True,exist_ok=True)
    configs={}
    for arm in ('SFRD' if phase=='phase0' else ALL_ARMS):
        ident=f'q-{arm}-{task}';scheduled=base/f'scheduled/{task}-s101'
        if not scheduled.exists():schedule(work,phase,ident)
        env={**os.environ,**launcher_env(work,arm,ident,scheduled),'DRY_RUN':'1'}
        dry=subprocess.run(['bash',str(KIT/'run_v4.sh')],env=env,capture_output=True,text=True,timeout=shared_limit(120),check=True)
        argv=dry.stdout.strip().split('\n')
        if arm in 'FR':
            # Remove only the distributed process launcher. Keep the exact script,
            # Hydra config name and every override emitted by the real launcher.
            entry=argv.index(str(KIT/'sft_entry.py'));command=[sys.executable]+argv[entry:]
            env.update(KIT_SFT_ARM_F='1',KIT_SFT_ARM=arm,KIT_SFT_SEED='101')
        else:command=[sys.executable]+argv[1:]
        command+=['--cfg','job','--resolve']
        resolved=subprocess.run(command,env=env,cwd=os.environ['SDPO_DIR'],capture_output=True,text=True,timeout=shared_limit(180),check=True)
        from kit.runner import write_durably
        target=out/(arm+'.yaml')
        if target.exists():require(target.read_text()==resolved.stdout,'resolved qualification config changed')
        else:target.write_text(resolved.stdout)
        t.atomic_json(out/(arm+'-command.json'),{'launcher':'run_v4.sh','arm':arm,'argv':argv,'resolve_command':command,
            'returncode':resolved.returncode,'resolved_sha256':t.sha(resolved.stdout),'trainer_commit':subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=os.environ['SDPO_DIR'],text=True,timeout=shared_limit(30)).strip()})
        configs[arm]=yaml.safe_load(resolved.stdout)
    from kit.v4_finqa_policy import observe
    observe(validate_configs,configs)
    t.atomic_json(base/'resolved-configs.json',{'ok':True,'arms':list(configs)})


def scoring_environment_gate(work):
    """Record changed physical scoring GPU/environment before generation."""
    from kit.eval_bed import machine_fingerprint
    from kit.v4_finqa_policy import warn
    path=paths(work,'qualification')/'eval/baseline-finqa-2048/bed-score.json'
    if not path.exists():warn('scoring environment','qualification baseline missing',work);return None
    baseline=load(path)['machine'];current=machine_fingerprint(True)
    require(current['versions']==baseline['versions'] and current['gpus']==baseline['gpus'],'scoring GPU or environment differs from qualification')
    return current


def registered_caps(work):
    """Generous static backstops; no owner receipt or measurement admission."""
    from kit.v4_campaign import build
    rows=build('main')['rows'];caps={key:3000 for key in ('teacher','rewrite','train','merge','score','probe','cpu')}
    for row in rows:
        key=row['id'].split('-')[0]
        if key in caps:caps[key]=max(caps[key],row.get('timeout_seconds',3000))
        elif key=='baseline':caps['score']=max(caps['score'],row['timeout_seconds'])
        elif 'allocation_cpu_cap_seconds' in row:caps['cpu']=max(caps['cpu'],row['allocation_cpu_cap_seconds'])
    return caps


def teacher_determinism(work):
    """Independent same prompts/seeds, reversed and differently composed batches."""
    from kit.v4_generate_shards import determinism
    return determinism(work)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('environment','prepare','verify-prepare','runner-profile','agreement','configs','determinism','profile-waves','restoration','archive-pack','gate','teacher','rewrite','coverage','training-set','schedule','train','merge','score','probe','standalone','report','pause','finished'))
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--phase',choices=('qualification','main'),required=True)
    parser.add_argument('--row',required=True);parser.add_argument('--task');parser.add_argument('--slot');parser.add_argument('--checkpoint')
    parser.add_argument('--cap',type=int);parser.add_argument('--step',type=int);parser.add_argument('--conditioned',action='store_true')
    args=parser.parse_args(argv); base=paths(args.work,args.phase)
    old_environment={key:os.environ.get(key) for key in ('WORK','V4_FINQA_CAMPAIGN','V4_CURRENT_ROW')}
    try:
        from kit.v4_timing import utc_now, receipt
        start_utc=utc_now();started=time.monotonic()
        from kit.v4_finqa_policy import confine
        confine(args.work,base)
        base.mkdir(parents=True,exist_ok=True)
        operation=args.operation
        os.environ.update(WORK=str(args.work),V4_FINQA_CAMPAIGN='1',V4_CURRENT_ROW=args.row)
        from kit.v4_finqa_policy import observe,ceiling
        ceiling(args.work)
        if os.environ.get('V4_FINQA_CAMPAIGN')=='1':
            from kit.v4_retention import sweep
            observe(sweep,args.work,starting_slot=args.slot if operation=='train' else None,running_row=args.row,site='retention sweep')
        if operation in ('teacher','rewrite','train','score','probe','determinism','profile-waves','restoration'):
            from kit.v4_prepare import verify
            verify(args.work,args.phase)
        if args.phase=='main' and operation in ('teacher','rewrite','score','probe','train'):
            observe(scoring_environment_gate,args.work)
            registered_caps(args.work)
        if operation=='environment':
            tag=os.environ.get('V4_KIT_TAG','local-untagged')
            t.atomic_json(base/'environment.json',{'kit_tag':tag,'runtime_versions':t.runtime_versions(),
                 'trainer_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=os.environ['SDPO_DIR'],text=True,timeout=shared_limit(30)).strip(),
                 'inference_commit':os.environ.get('V4_INFERENCE_COMMIT')})
        elif operation=='prepare':
            from kit.v4_prepare import prepare as cpu_prepare
            cpu_prepare(args.work,args.phase)
        elif operation=='verify-prepare':
            from kit.v4_prepare import verify
            verify(args.work,args.phase)
        elif operation=='runner-profile':
            from kit.runner import bounded_command
            destination=base/'runner-profile';destination.mkdir(parents=True,exist_ok=True)
            result=bounded_command([sys.executable,str(KIT/'v4_runner_profile.py'),'--out',str(destination/'profile.json')],
                timeout=shared_limit(900),log=destination/'output.log')
            if result['returncode']:raise ValueError(result['failure_type']+': runner profile')
        elif operation=='agreement':
            from kit.v4_archive import DirectoryEvidence
            from kit.v4_readers import scorer_context
            from kit.v4_qualification import archive_agreement
            import tempfile
            evidence=DirectoryEvidence(args.work)
            from kit.v4_evidence import scheduler_rows
            observe(scheduler_rows,evidence,'qualification')
            with tempfile.TemporaryDirectory() as temp:
                tokenizer,mcq=scorer_context(evidence,temp)
                from kit.v4_evidence import data_context
                with data_context(evidence,tokenizer,mcq):
                    result=observe(archive_agreement,evidence,mcq,tasks=TASKS)
                    t.atomic_json(base/'scoring-agreement.json',result or {'warning':'agreement unavailable'})
        elif operation=='configs': qualification_configs(args.work)
        elif operation=='determinism': teacher_determinism(args.work)
        elif operation=='profile-waves':
            from kit.v4_generate_shards import profile_waves
            profile_waves(args.work)
        elif operation=='archive-pack':
            destination=Path(args.work)/'archive-previews';destination.mkdir(exist_ok=True)
            target=destination/(args.phase+'-'+str(os.environ.get('SLURM_JOB_ID','cpu'))+'-'+str(os.environ.get('V4_RUNNER_ATTEMPT','1'))+'-'+str(time.time_ns())+'.tar.gz')
            run([sys.executable,str(KIT/'collect.py'),'--work',str(args.work),'--out',str(target)])
        elif operation=='restoration':
            from kit.v4_restore_check import run_check
            run_check(args.work)
        elif operation=='gate': identity_gate(args.work)
        elif operation in ('teacher','rewrite'): teacher(args.work,args.phase,args.task,operation=='rewrite')
        elif operation=='coverage': coverage(args.work,args.phase)
        elif operation=='training-set': training_set(args.work,args.phase)
        elif operation=='schedule': schedule(args.work,args.phase,args.slot)
        elif operation=='train': train(args.work,args.phase,args.slot)
        elif operation=='merge': merge(args.work,args.phase,args.slot)
        elif operation=='score': score(args.work,args.phase,args.checkpoint,args.task,args.cap)
        elif operation=='probe': probe(args.work,args.phase,args.slot,args.task,args.step,args.conditioned)
        elif operation=='standalone': observe(standalone,args.work,site='standalone analysis')
        elif operation=='report': report(args.work,args.phase)
        else:
            t.atomic_json(base/(operation+'.json'),{'state':'FINISHED','owner_proceed_required':False})
            print('FINISHED: progress archive ready; continue automatically to main')
        status=Path(args.work)/'v4/report-status'; status.mkdir(parents=True,exist_ok=True)
        (base/'timings').mkdir(parents=True,exist_ok=True)
        measurement=receipt(start_utc,time.monotonic()-started,row=args.row,phase=args.phase,operation=operation)
        t.atomic_json(base/'timings'/(args.row+'.json'),measurement)
        t.atomic_json(status/(args.row+'.json'),{'ok':1,'wall_seconds':measurement['wall_seconds']})
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        from kit.v4_finqa_policy import CorrectnessStop
        try:confine(args.work,base)
        except CorrectnessStop:
            print('STOP: correctness: '+str(exc));return 2
        (base/'failures').mkdir(parents=True,exist_ok=True)
        t.atomic_json(base/'failures'/(args.row+'.json'), {'row':args.row,'operation':args.operation,'cause':str(exc)})
        print('FAILED: '+args.operation+': '+str(exc))
        return 1
    finally:
        for key,value in old_environment.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


if __name__=='__main__': raise SystemExit(main())
