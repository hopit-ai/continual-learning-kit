#!/usr/bin/env python3
"""CPU preparation and an immutable hash receipt, before any allocation exists."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import math
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.v4_arms import LAUNCHERS
KIT=Path(__file__).resolve().parent


from kit.v4_phase0_timing import scale

def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def input_kind(path):
    path=Path(path).resolve()
    for key in ('MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER'):
        if os.environ.get(key) and path.is_relative_to(Path(os.environ[key]).resolve()):return 'model/data'
    if path.suffix in ('.py','.sh','.yaml','.so') or path.name in ('python','python3'):return 'software'
    if any(marker in str(path) for marker in ('/sources/','/tokenizer/','-pool.','-panel.','-probe.','model-identities','frozen-model-registration')) or path.suffix in ('.parquet','.safetensors','.bin','.pt'):return 'model/data'
    return 'observation'


def file_receipt(paths):
    return {'files':{str(Path(p).absolute()):{'sha256':digest(p),'bytes':Path(p).stat().st_size,'kind':input_kind(p)}
                     for p in sorted(map(Path,paths))}}


def verify_files(receipt):
    for name, item in receipt['files'].items():
        path=Path(name)
        if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            from kit.v4_finqa_policy import enabled,warn
            kind=item.get('kind',input_kind(path))
            if enabled() and kind!='model/data':warn('local source/observation change','prepared '+kind+' input changed: '+name)
            else:raise ValueError('prepared input changed: '+name)
    return True


def verify_selected_models(work):
    """A changed environment cannot select unverified model bytes after prepare."""
    from kit.v4_finqa_site import LAUNCH
    launch=Path(work)/LAUNCH
    if not launch.exists():return # Legacy CPU fixtures have no live host route.
    frozen=json.loads(launch.read_text())['environment']
    from kit.v4_teacher import local_identity,tokenizer_identity
    for role,key in (('initial','MODEL_DIR'),('teacher','TEACHER_MODEL_DIR')):
        selected=Path(os.environ[key]).resolve();original=Path(frozen[key]).resolve()
        if selected==original:continue # verify_files just hashed this exact tree.
        expected=json.loads((Path(work)/'v4/report-inputs/model-identities.json').read_text())[role]
        actual=local_identity(selected,'Qwen/Qwen3-8B' if role=='initial' else 'Qwen/Qwen3.6-27B')
        if actual['model_revision']!=expected['model_revision'] or actual['model_file_hashes']!=expected['model_file_hashes']:
            raise ValueError('selected model identity differs: '+role)
    if Path(os.environ['QWEN3_8B_TOKENIZER']).resolve()!=Path(frozen['QWEN3_8B_TOKENIZER']).resolve():
        if tokenizer_identity(os.environ['QWEN3_8B_TOKENIZER'])!=tokenizer_identity(frozen['QWEN3_8B_TOKENIZER']):raise ValueError('selected tokenizer identity differs')


def verification_allowance(measured):
    """Measured complete byte rehash x 1.5, with the original 60-second floor."""
    if isinstance(measured,bool) or not isinstance(measured,(int,float)) or not math.isfinite(measured) or measured<0:
        raise ValueError('prepared integrity verification measurement must be finite nonnegative seconds')
    scaled=1.5*measured
    if not math.isfinite(scaled):raise ValueError('prepared integrity verification measurement allowance overflows')
    return max(60,math.ceil(scaled))


def main_verification_allowance(measured):
    try:return scale(verification_allowance(measured))
    except ValueError as exc:
        from kit.v4_finqa_policy import enabled,warn
        if not enabled():raise
        warn('prepared verification timing',str(exc)+'; using the original generous floor')
        return scale(60)


def main_verification_row_cap(work,phase,default):
    """A hash-bound CPU measurement can enlarge verification patience, never the job."""
    receipt_path=Path(work)/'v4/report-inputs'/('prepare-receipt-main.json' if phase=='main' else 'prepare-receipt.json')
    if not receipt_path.exists():return default
    receipt=json.loads(receipt_path.read_text())
    if 'verification_seconds' not in receipt:return default
    allowance=main_verification_allowance(receipt['verification_seconds'])
    return max(default,allowance)


def measured_verification(work,phase):
    """Freeze a CPU measurement of the exact main-package verifier once."""
    path=Path(work)/'v4/report-inputs'/('prepare-receipt-main.json' if phase=='main' else 'prepare-receipt.json')
    doc=json.loads(path.read_text())
    if 'verification_seconds' not in doc:
        started=time.monotonic();verify(work,phase)
        doc['verification_seconds']=time.monotonic()-started
        doc['verification_allowance_seconds']=main_verification_allowance(doc['verification_seconds'])
        from kit.p4_watchdog import write_durably
        write_durably(path,doc)
    if doc.get('verification_allowance_seconds')!=main_verification_allowance(doc['verification_seconds']):
        from kit.v4_finqa_policy import require
        require(False,'prepared integrity verification allowance differs from CPU measurement','prepared verification timing')
    return doc


def smoke_inventory(kit=KIT):
    kit=Path(kit)
    # Manager-side archive ingest is never part of the partner launch path.
    return {'modules':['kit.'+'.'.join(p.relative_to(kit).with_suffix('').parts) for p in sorted(kit.rglob('*.py'))
                       if p.relative_to(kit).as_posix()!='v4_finqa_ingest.py'],
            'dependencies':['qwen_vl_utils','torch','torchdata.stateful_dataloader','transformers','pyarrow','verl'],
            'launchers':LAUNCHERS,
            'shells':[str(p) for p in sorted(kit.rglob('*.sh'))]}



def archived_smoke_inventory(archive):
    """Phase-0 predates the main tag: compare imports to its collected source.

    Collector manifest hashes every snapshot byte. Read only its filenames; do
    not execute archived code or require modules added at the package boundary.
    Thin local test evidence still uses the current inventory.
    """
    inventory=smoke_inventory();prefix='kit-snapshot/kit/'
    sources=sorted(name[len(prefix):] for name in archive.files if name.startswith(prefix) and name.endswith('.py'))
    if sources:inventory['modules']=['kit.'+'.'.join(Path(name).with_suffix('').parts) for name in sources
                                    if name!='v4_finqa_ingest.py']
    return inventory

def import_smoke(work,phase="qualification"):
    """Run actual imports in independent CPU processes, including patched loader."""
    from kit.runner import bounded_command, write_durably
    work=Path(work);out=work/('v4/report-inputs/import-smoke' if phase=='qualification' else 'v4/report-inputs/import-smoke-'+phase);out.mkdir(parents=True,exist_ok=True)
    inventory=smoke_inventory();observations=[]
    if phase=='phase0':inventory['launchers']={a:LAUNCHERS[a] for a in 'SFRD'}
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(KIT.parent)+os.pathsep+str(KIT)+os.pathsep+os.environ['SDPO_DIR'],
         'KIT_SFT_ARM_F':'0','V4_TELEMETRY':'0',**({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'} if phase=='phase0' else {})}
    commands=[('import-'+module,[sys.executable,'-c','import importlib; importlib.import_module('+repr(module)+')'])
              for module in inventory['dependencies']+inventory['modules']]
    if phase=='phase0':commands.append(('phase0-trainer-imports',[sys.executable,'-c','import torch, verl, vllm, qwen_vl_utils']))
    commands += [('syntax-'+Path(path).name,['bash','-n',path]) for path in inventory['shells']]
    commands += [('entry-help-'+name,[sys.executable,str(KIT/name),'--help']) for name in ('sft_entry.py','sdft_entry.py')]
    commands += [('entry-help-S',[sys.executable,'-m','kit.v4_ppo_entry','--help'])]
    for arm in ('SFRD' if phase=='phase0' else ('S','F','R','D','D-R')):
        commands.append(('launcher-'+arm,['bash',str(KIT/'run_v4.sh')]))
    # Importing the actual entry installs its patch. Read its first CPU batch;
    # this reaches the pinned loader implementation instead of inspecting source.
    code='''import kit.sft_entry
from kit.v4_sft import V4SFTDataset
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader
import os, torch
model=AutoTokenizer.from_pretrained(os.environ['QWEN3_8B_TOKENIZER'], local_files_only=True)
import tempfile, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
os.environ.update(V4_PROFILE='technical-smoke',STEPS='2')
root=Path(tempfile.mkdtemp());path=root/'first-batch.parquet'
pq.write_table(pa.Table.from_pylist([{'prompt':[{'role':'user','content':'What is 1 + 1?'}],'response':'Answer: 2','technical_synthetic':True}]*2),path)
dataset=V4SFTDataset(str(path),model,{'max_length':6144,'truncation':'error','response_key':'response'})
# Login-node first-batch admission checks the patched loader without sockets.
# The actual training loader retains its eight workers in the GPU allocation.
loader=StatefulDataLoader(dataset,batch_size=2,pin_memory=True,pin_memory_device='cuda',num_workers=0)
batch=next(iter(loader))
assert loader.pin_memory is False and batch['input_ids'].device.type=='cpu'
assert batch['loss_mask'].sum()>0
print('patched first CPU batch passed')
'''
    commands.append(('first-sft-cpu-batch',[sys.executable,'-c',code]))
    for index,(name,command) in enumerate(commands):
        command_env=dict(env)
        if name.startswith('launcher-'):
            command_env.update(ARM=name.removeprefix('launcher-'),SEED='101',NAME='cpu-import-smoke-'+name[-1],WORK=str(work),
                DRY_RUN='1',V4_PROFILE='technical-smoke',STEPS='2',DATA_MANIFEST=str(out/'future-manifest.json'),
                TRAIN_FILE=str(out/'future-train.parquet'),VAL_FILE=str(out/'future-val.parquet'),
                DATASET='datasets/v4_chem',REWARD_FILE=str(KIT/'beds/v4_reward.py'))
        result=bounded_command(command,timeout=scale(600) if phase=='phase0' else scale(120),env=command_env,log=out/f'{index}.log')
        observations.append({'name':name,'command':command,**result,'log_sha256':digest(out/f'{index}.log')})
        if result['returncode']:
            write_durably(out/'failure.json',{'failure_type':'cpu_import_smoke','observations':observations})
            raise ValueError('cpu_import_smoke: '+name)
    from kit.v4_teacher import runtime_versions
    write_durably(out/'receipt.json',{'schema':'v4-import-smoke.v1','ok':True,'inventory':inventory,
        'observations':observations,'environment':{'runtime_versions':runtime_versions(),'python':sys.version,
        'trainer_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=os.environ['SDPO_DIR'],text=True,timeout=scale(600) if phase=='phase0' else scale(30)).strip()}})
    return out/'receipt.json'


def prepare(work,phase):
    from kit.runner import bounded_command, write_durably
    if os.environ.get('SLURM_JOB_ID'):
        from kit.v4_finqa_policy import require
        require(False,'CPU prepare must precede any GPU allocation','prepare timing')
    from kit.v4_phase0_carry import verify_work
    verify_work(work)
    from kit.v4_finqa_site import planning_storage
    planning_storage(work)
    work=Path(work);receipt_path=work/('v4/report-inputs/prepare-receipt.json' if phase=='qualification' else 'v4/report-inputs/prepare-receipt-main.json')
    if receipt_path.exists():
        verify_files(json.loads(receipt_path.read_text()))
        from kit.v4_campaign_ops import prepare as data_prepare
        data_prepare(work,phase)
        measured_verification(work,phase)
        return receipt_path
    smoke=import_smoke(work,phase)  # FIRST qualification operation in the partner environment.
    # The partner stages pinned dependencies/models using explicit bounded CPU
    # commands. Optional downloads are supplied in a frozen plan, never guessed.
    if phase!='phase0' and os.environ.get('V4_STATIC_PREPARE_PLAN'):
        from kit.v4_finqa_policy import warn
        warn('CPU preparation','main reuses retained snapshots; ignoring legacy static preparation plan',work)
    if phase=='phase0' and os.environ.get('V4_STATIC_PREPARE_PLAN'):
        plan=json.loads(Path(os.environ['V4_STATIC_PREPARE_PLAN']).read_text())
        for entry in plan['commands']:
            result=bounded_command(entry['argv'],timeout=scale(max(600,entry['timeout_seconds'])) if phase=='phase0' else entry['timeout_seconds'])
            if result['returncode']:raise ValueError(result['failure_type']+': static preparation')
    from kit.v4_campaign_ops import prepare as data_prepare
    data_prepare(work,phase)
    from kit import v4_teacher as t
    tokenizer=t.load_student_tokenizer(os.environ['QWEN3_8B_TOKENIZER'])
    lengths={}
    for task in ('finqa',):
        items=t.read_jsonl(work/f'v4/report-{phase}/{task}-pool.jsonl')
        lengths[task]=[{'id':item['id'],'input_tokens':len(tokenizer.encode(tokenizer.apply_chat_template(
            t.student_messages(item),tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))} for item in items]
    inputs=work/'v4/report-inputs';inputs.mkdir(parents=True,exist_ok=True)
    if phase=='qualification':
        write_durably(inputs/'runner-main-campaign.json',json.loads((KIT/'campaigns/v4-main.yaml').read_text()))
        site=int(os.environ.get('V4_SITE_MINUTES','360'))
        if site<15:
            from kit.v4_finqa_policy import warn
            warn('site window','short allocation; progress may need continuation',work)
        write_durably(inputs/'allocation-profile.json',{'schema':'v4-allocation-profile.v1','site_minutes':site,
            'windows':{stage:min(requested,site) for stage,requested in {'teacher':3000,'rewrite':3000,'scientific':7200}.items()}})
        teacher_tokenizer=t.load_student_tokenizer(os.environ['TEACHER_MODEL_DIR'])
        ratios={task:t.measure_pool_token_ratio(t.read_jsonl(work/f'v4/report-qualification/{task}-pool.jsonl'),teacher_tokenizer,tokenizer) for task in ('finqa',)}
        write_durably(inputs/'teacher-token-ratio.json',{'registered_teacher_cap':t.TEACHER_NEW_TOKENS,'tasks':ratios})
    write_durably(inputs/('static-tokenisation.json' if phase=='qualification' else 'static-tokenisation-main.json'),lengths)
    if (inputs/'v4_departures.json').exists():
        if (inputs/'v4_departures.json').read_bytes()!=(KIT/'v4_departures.json').read_bytes():
            from kit.v4_finqa_policy import warn
            warn('local science setting','departures changed since qualification; flag in return note',work)
    else:(inputs/'v4_departures.json').write_bytes((KIT/'v4_departures.json').read_bytes())
    if phase!='phase0':
        from kit.v4_retention import POLICY
        policy=inputs/'retention-policy.json'
        if policy.exists() and json.loads(policy.read_text())!=POLICY:
            from kit.v4_finqa_policy import warn
            warn('retention configuration','frozen retention policy differs; existing proofs remain authoritative',work)
        if not policy.exists():write_durably(policy,POLICY)
    paths=list((work/'v4/datasets').rglob('*'))+list(inputs.rglob('*'))
    paths += [p for p in KIT.rglob('*') if p.suffix in ('.py','.sh','.yaml','.json') and '__pycache__' not in p.parts]
    for key in ('MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER'):
        paths += list(Path(os.environ[key]).rglob('*'))
    inference=os.environ.get('V4_TEACHER_PYTHON')
    if not inference:raise ValueError('separate pinned teacher inference Python required')
    result=subprocess.run([inference,'-c','import json,vllm; print(json.dumps({"vllm":vllm.__version__}))'],capture_output=True,text=True,timeout=scale(120))
    if result.returncode:raise ValueError('cpu_inference_environment: '+result.stderr)
    version=json.loads(result.stdout.strip().split('\n')[-1])['vllm']
    inspection=subprocess.run([inference,'-c',
        'import hashlib,json,pathlib,vllm; root=pathlib.Path(vllm.__file__).parent; files={str(p):{"sha256":hashlib.file_digest(p.open("rb"),"sha256").hexdigest(),"bytes":p.stat().st_size} for p in root.rglob("*") if p.is_file() and p.suffix in (".py",".so")}; print(json.dumps(files,sort_keys=True))'],capture_output=True,text=True,timeout=scale(180))
    if inspection.returncode:raise ValueError('cpu_inference_identity')
    package_files=json.loads(inspection.stdout)
    if not package_files:raise ValueError('teacher inference package files absent')
    if tuple(map(int,version.split('.')[:2]))<(0,17):
        from kit.v4_finqa_policy import warn
        warn('inference version','teacher inference vLLM is below registered >=0.17',work)
    inference_identity={'python':str(Path(inference).resolve()),'python_sha256':digest(inference),'vllm':version,
                        'commit':os.environ['V4_INFERENCE_COMMIT'],'package_sha256':hashlib.sha256(json.dumps(package_files,sort_keys=True).encode()).hexdigest()}
    inference_path=inputs/('inference-environment.json' if phase=='qualification' else 'inference-environment-main.json')
    write_durably(inference_path,inference_identity)
    paths += [Path(inference),inference_path,*map(Path,package_files)]
    archived={str(p.relative_to(work)):digest(p) for p in inputs.rglob('*') if p.is_file()}
    receipt={'archived_files':archived,'schema':'v4-cpu-prepare.v1','phase':phase,'before_allocation':not bool(os.environ.get('SLURM_JOB_ID')),
        'import_smoke_sha256':digest(smoke),'inference_environment':inference_identity,
        **file_receipt({p.resolve() for p in paths if p.is_file()})}
    write_durably(receipt_path,receipt)
    measured_verification(work,phase)
    return receipt_path


def verify(work,phase='qualification'):
    work=Path(work);name='prepare-receipt-main.json' if phase=='main' else 'prepare-receipt.json'
    receipt=json.loads((work/'v4/report-inputs'/name).read_text())
    from kit.v4_finqa_policy import require
    require(receipt.get('before_allocation') is True,'prepare not completed before allocation','prepare timing')
    verify_files(receipt)
    verify_selected_models(work)
    return True


def verify_archived(archive):
    from kit.v4_finqa_policy import require, enabled, warn
    receipt=archive.json('v4/report-inputs/prepare-receipt.json')
    if 'verification_seconds' in receipt:
        if receipt.get('verification_allowance_seconds')!=main_verification_allowance(receipt['verification_seconds']):
            require(False,'prepared verification allowance differs from CPU measurement','archived verification timing')
    smoke=archive.json('v4/report-inputs/import-smoke/receipt.json')
    if receipt.get('before_allocation') is not True or smoke.get('ok') is not True:require(False,'missing partner CPU prepare/import smoke','archived CPU smoke')
    raw=archive.files['v4/report-inputs/import-smoke/receipt.json']
    if hashlib.sha256(raw).hexdigest()!=receipt['import_smoke_sha256']:require(False,'CPU smoke receipt changed','archived CPU smoke')
    if not receipt.get('archived_files'):raise ValueError('CPU prepare archived input hashes missing')
    for name,digest_value in receipt['archived_files'].items():
        if name not in archive.files and enabled() and input_kind(name)!='model/data':
            warn('archived source/observation change','prepared observation missing: '+name);continue
        if name not in archive.files:
            if name.startswith('v4/report-inputs/v4-data-audit-') and name.endswith('.ids'):
                raise ValueError('missing diagnostic: '+repr(name))
            raise ValueError('missing prepared archive input: '+name)
        if hashlib.sha256(archive.files[name]).hexdigest()!=digest_value:
            if enabled() and input_kind(name)!='model/data':warn('archived source/observation change','prepared software/observation changed: '+name)
            else:raise ValueError('prepared archive input changed: '+name)
    env=archive.json('v4/report-qualification/environment.json')
    if smoke.get('environment',{}).get('trainer_commit')!=env['trainer_commit'] or smoke.get('environment',{}).get('runtime_versions')!=env.get('runtime_versions'):
        require(False,'CPU smoke environment differs from qualification','archived CPU smoke')
    inventory=smoke_inventory()
    if smoke['inventory']['modules']!=inventory['modules'] or smoke['inventory']['dependencies']!=inventory['dependencies']:
        require(False,'CPU import inventory incomplete','archived CPU smoke')
    names={r['name'] for r in smoke['observations'] if r.get('returncode')==0}
    if not {'import-'+name for name in inventory['modules']+inventory['dependencies']} <= names or not {'first-sft-cpu-batch',*('launcher-'+a for a in 'SFRD'),'entry-help-sft_entry.py','entry-help-sdft_entry.py','entry-help-S'} <= names:
        require(False,'CPU module/first-batch smoke incomplete','archived CPU smoke')
    for index,observation in enumerate(smoke['observations']):
        raw=archive.files[f'v4/report-inputs/import-smoke/{index}.log']
        if hashlib.sha256(raw).hexdigest()!=observation['log_sha256']:require(False,'CPU smoke log changed','archived CPU smoke')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--phase',choices=('qualification','main'),default='qualification')
    args=parser.parse_args(argv);print(prepare(args.work,args.phase));return 0


if __name__=='__main__':raise SystemExit(main())
