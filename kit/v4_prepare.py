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
KIT=Path(__file__).resolve().parent


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def file_receipt(paths):
    return {'files':{str(Path(p).absolute()):{'sha256':digest(p),'bytes':Path(p).stat().st_size}
                     for p in sorted(map(Path,paths))}}


def verify_files(receipt):
    for name, item in receipt['files'].items():
        path=Path(name)
        if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError('prepared input changed: '+name)
    return True


def smoke_inventory(kit=KIT):
    kit=Path(kit)
    return {'modules':['kit.'+'.'.join(p.relative_to(kit).with_suffix('').parts) for p in sorted(kit.rglob('*.py'))],
            'dependencies':['qwen_vl_utils','torch','torchdata.stateful_dataloader','transformers','pyarrow','verl'],
            'launchers':{'S':'run_sdpo_toolalpaca.sh','D':'run_sdft.sh','F':'run_sft_f.sh','R':'run_sft_r.sh'},
            'shells':[str(p) for p in sorted(kit.rglob('*.sh'))]}


def import_smoke(work,phase="qualification"):
    """Run actual imports in independent CPU processes, including patched loader."""
    from kit.runner import bounded_command, write_durably
    work=Path(work);out=work/('v4/report-inputs/import-smoke' if phase=='qualification' else 'v4/report-inputs/import-smoke-'+phase);out.mkdir(parents=True,exist_ok=True)
    inventory=smoke_inventory();observations=[]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(KIT.parent)+os.pathsep+str(KIT)+os.pathsep+os.environ['SDPO_DIR'],
         'KIT_SFT_ARM_F':'0','V4_TELEMETRY':'0',**({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'} if phase=='phase0' else {})}
    commands=[('import-'+module,[sys.executable,'-c','import importlib; importlib.import_module('+repr(module)+')'])
              for module in inventory['dependencies']+inventory['modules']]
    if phase=='phase0':commands.append(('phase0-trainer-imports',[sys.executable,'-c','import torch, verl, vllm, qwen_vl_utils']))
    commands += [('syntax-'+Path(path).name,['bash','-n',path]) for path in inventory['shells']]
    commands += [('entry-help-'+name,[sys.executable,str(KIT/name),'--help']) for name in ('sft_entry.py','sdft_entry.py')]
    commands += [('entry-help-S',[sys.executable,'-m','kit.v4_ppo_entry','--help'])]
    for arm in 'SFRD':
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
            command_env.update(ARM=name[-1],SEED='101',NAME='cpu-import-smoke-'+name[-1],WORK=str(work),
                DRY_RUN='1',V4_PROFILE='technical-smoke',STEPS='2',DATA_MANIFEST=str(out/'future-manifest.json'),
                TRAIN_FILE=str(out/'future-train.parquet'),VAL_FILE=str(out/'future-val.parquet'),
                DATASET='datasets/v4_chem',REWARD_FILE=str(KIT/'beds/v4_reward.py'))
        result=bounded_command(command,timeout=600 if phase=='phase0' else 120,env=command_env,log=out/f'{index}.log')
        observations.append({'name':name,'command':command,**result,'log_sha256':digest(out/f'{index}.log')})
        if result['returncode']:
            write_durably(out/'failure.json',{'failure_type':'cpu_import_smoke','observations':observations})
            raise ValueError('cpu_import_smoke: '+name)
    from kit.v4_teacher import runtime_versions
    write_durably(out/'receipt.json',{'schema':'v4-import-smoke.v1','ok':True,'inventory':inventory,
        'observations':observations,'environment':{'runtime_versions':runtime_versions(),'python':sys.version,
        'trainer_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=os.environ['SDPO_DIR'],text=True,timeout=600 if phase=='phase0' else 30).strip()}})
    return out/'receipt.json'


def prepare(work,phase):
    from kit.runner import bounded_command, write_durably
    if os.environ.get('SLURM_JOB_ID'):raise ValueError('CPU prepare must precede any GPU allocation')
    work=Path(work);receipt_path=work/('v4/report-inputs/prepare-receipt.json' if phase=='qualification' else 'v4/report-inputs/prepare-receipt-main.json')
    if receipt_path.exists():
        verify_files(json.loads(receipt_path.read_text()))
        if phase=='main':
            from kit.v4_campaign_ops import prepare as data_prepare
            data_prepare(work,phase)
        return receipt_path
    smoke=import_smoke(work,phase)  # FIRST qualification operation in the partner environment.
    # The partner stages pinned dependencies/models using explicit bounded CPU
    # commands. Optional downloads are supplied in a frozen plan, never guessed.
    if os.environ.get('V4_STATIC_PREPARE_PLAN'):
        plan=json.loads(Path(os.environ['V4_STATIC_PREPARE_PLAN']).read_text())
        for entry in plan['commands']:
            result=bounded_command(entry['argv'],timeout=max(600,entry['timeout_seconds']) if phase=='phase0' else entry['timeout_seconds'])
            if result['returncode']:raise ValueError(result['failure_type']+': static preparation')
    from kit.v4_campaign_ops import prepare as data_prepare
    data_prepare(work,phase)
    from kit import v4_teacher as t
    tokenizer=t.load_student_tokenizer(os.environ['QWEN3_8B_TOKENIZER'])
    lengths={}
    for task in ('chemistry','finqa'):
        items=t.read_jsonl(work/f'v4/report-{phase}/{task}-pool.jsonl')
        lengths[task]=[{'id':item['id'],'input_tokens':len(tokenizer.encode(tokenizer.apply_chat_template(
            t.student_messages(item),tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))} for item in items]
    inputs=work/'v4/report-inputs';inputs.mkdir(parents=True,exist_ok=True)
    if phase=='qualification':
        write_durably(inputs/'runner-main-campaign.json',json.loads((KIT/'campaigns/v4-main.yaml').read_text()))
        site=int(os.environ.get('V4_SITE_MINUTES','360'))
        if site<15:raise ValueError('site window cannot fit containment selftest and a minimum-cap row')
        write_durably(inputs/'allocation-profile.json',{'schema':'v4-allocation-profile.v1','site_minutes':site,
            'windows':{stage:min(requested,site) for stage,requested in {'teacher':300,'rewrite':300,'scientific':720}.items()}})
        teacher_tokenizer=t.load_student_tokenizer(os.environ['TEACHER_MODEL_DIR'])
        ratios={task:t.measure_pool_token_ratio(t.read_jsonl(work/f'v4/report-qualification/{task}-pool.jsonl'),teacher_tokenizer,tokenizer) for task in ('chemistry','finqa')}
        write_durably(inputs/'teacher-token-ratio.json',{'registered_teacher_cap':t.TEACHER_NEW_TOKENS,'tasks':ratios})
    write_durably(inputs/('static-tokenisation.json' if phase=='qualification' else 'static-tokenisation-main.json'),lengths)
    if (inputs/'v4_departures.json').exists():
        if (inputs/'v4_departures.json').read_bytes()!=(KIT/'v4_departures.json').read_bytes():raise ValueError('frozen departures changed since qualification')
    else:(inputs/'v4_departures.json').write_bytes((KIT/'v4_departures.json').read_bytes())
    paths=list((work/'v4/datasets').rglob('*'))+list(inputs.rglob('*'))
    paths += [p for p in KIT.rglob('*') if p.suffix in ('.py','.sh','.yaml','.json') and '__pycache__' not in p.parts]
    for key in ('MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER'):
        paths += list(Path(os.environ[key]).rglob('*'))
    inference=os.environ.get('V4_TEACHER_PYTHON')
    if not inference:raise ValueError('separate pinned teacher inference Python required')
    result=subprocess.run([inference,'-c','import json,vllm; print(json.dumps({"vllm":vllm.__version__}))'],capture_output=True,text=True,timeout=120)
    if result.returncode:raise ValueError('cpu_inference_environment: '+result.stderr)
    version=json.loads(result.stdout.strip().split('\n')[-1])['vllm']
    inspection=subprocess.run([inference,'-c',
        'import hashlib,json,pathlib,vllm; root=pathlib.Path(vllm.__file__).parent; files={str(p):{"sha256":hashlib.file_digest(p.open("rb"),"sha256").hexdigest(),"bytes":p.stat().st_size} for p in root.rglob("*") if p.is_file() and p.suffix in (".py",".so")}; print(json.dumps(files,sort_keys=True))'],capture_output=True,text=True,timeout=180)
    if inspection.returncode:raise ValueError('cpu_inference_identity')
    package_files=json.loads(inspection.stdout)
    if not package_files:raise ValueError('teacher inference package files absent')
    if tuple(map(int,version.split('.')[:2]))<(0,17):raise ValueError('teacher inference vLLM needs >=0.17')
    inference_identity={'python':str(Path(inference).resolve()),'python_sha256':digest(inference),'vllm':version,
                        'commit':os.environ['V4_INFERENCE_COMMIT'],'package_sha256':hashlib.sha256(json.dumps(package_files,sort_keys=True).encode()).hexdigest()}
    inference_path=inputs/('inference-environment.json' if phase=='qualification' else 'inference-environment-main.json')
    write_durably(inference_path,inference_identity)
    paths += [Path(inference),inference_path,*map(Path,package_files)]
    archived={str(p.relative_to(work)):digest(p) for p in inputs.rglob('*') if p.is_file()}
    receipt={'archived_files':archived,'schema':'v4-cpu-prepare.v1','phase':phase,'before_allocation':True,
        'import_smoke_sha256':digest(smoke),'inference_environment':inference_identity,
        **file_receipt({p.resolve() for p in paths if p.is_file()})}
    write_durably(receipt_path,receipt)
    return receipt_path


def verify(work,phase='qualification'):
    work=Path(work);name='prepare-receipt-main.json' if phase=='main' else 'prepare-receipt.json'
    receipt=json.loads((work/'v4/report-inputs'/name).read_text())
    if receipt.get('before_allocation') is not True:raise ValueError('prepare not completed before allocation')
    return verify_files(receipt)


def verify_archived(archive):
    receipt=archive.json('v4/report-inputs/prepare-receipt.json')
    smoke=archive.json('v4/report-inputs/import-smoke/receipt.json')
    if receipt.get('before_allocation') is not True or smoke.get('ok') is not True:raise ValueError('missing partner CPU prepare/import smoke')
    raw=archive.files['v4/report-inputs/import-smoke/receipt.json']
    if hashlib.sha256(raw).hexdigest()!=receipt['import_smoke_sha256']:raise ValueError('CPU smoke receipt changed')
    if not receipt.get('archived_files'):raise ValueError('CPU prepare archived input hashes missing')
    for name,digest_value in receipt['archived_files'].items():
        if name not in archive.files:
            if name.startswith('v4/report-inputs/v4-data-audit-') and name.endswith('.ids'):
                raise ValueError('missing diagnostic: '+repr(name))
            raise ValueError('missing prepared archive input: '+name)
        if hashlib.sha256(archive.files[name]).hexdigest()!=digest_value:raise ValueError('prepared archive input changed: '+name)
    env=archive.json('v4/report-qualification/environment.json')
    if smoke.get('environment',{}).get('trainer_commit')!=env['trainer_commit'] or smoke.get('environment',{}).get('runtime_versions')!=env.get('runtime_versions'):
        raise ValueError('CPU smoke environment differs from qualification')
    inventory=smoke_inventory()
    if smoke['inventory']['modules']!=inventory['modules'] or smoke['inventory']['dependencies']!=inventory['dependencies']:
        raise ValueError('CPU import inventory incomplete')
    names={r['name'] for r in smoke['observations'] if r.get('returncode')==0}
    if not {'import-'+name for name in inventory['modules']+inventory['dependencies']} <= names or not {'first-sft-cpu-batch',*('launcher-'+a for a in 'SFRD'),'entry-help-sft_entry.py','entry-help-sdft_entry.py','entry-help-S'} <= names:
        raise ValueError('CPU module/first-batch smoke incomplete')
    for index,observation in enumerate(smoke['observations']):
        raw=archive.files[f'v4/report-inputs/import-smoke/{index}.log']
        if hashlib.sha256(raw).hexdigest()!=observation['log_sha256']:raise ValueError('CPU smoke log changed')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--phase',choices=('qualification','main'),default='qualification')
    args=parser.parse_args(argv);print(prepare(args.work,args.phase));return 0


def verification_allowance(measured):
    """Measured complete byte rehash x 1.5, with the original 60-second floor."""
    if isinstance(measured,bool) or not isinstance(measured,(int,float)) or not math.isfinite(measured) or measured<0:
        raise ValueError('prepared integrity verification measurement must be finite nonnegative seconds')
    scaled=1.5*measured
    if not math.isfinite(scaled):raise ValueError('prepared integrity verification measurement allowance overflows')
    return max(60,math.ceil(scaled))

if __name__=='__main__':raise SystemExit(main())
