#!/usr/bin/env python3
"""Technical FinQA PHASE-0 adapter for the existing runner, planner and readers."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit import v4_teacher as t,v4_phase0_site as site
from kit.runner import bounded_command,write_durably
from kit.v4_prepare import digest,file_receipt,verify_files as base_verify_files,verification_allowance as base_verification_allowance
from kit.v4_phase0_timing import relaxed_timeout,scale
KIT=Path(__file__).resolve().parent
BASE='v4/report-phase0/'
DISPATCH_SECONDS=scale(15)
ALLOCATION_CAP_GPU_HOURS=192
OUTPUT_POLICY={'schema':'v4-phase0-output-policy.v1','scope':'technical',
               'scientific_phase_allowed':False,'usable_as_scientific_initialisations':False}
INFERENCE_PACKAGES=['vllm==0.18.0','torch==2.10.0+cu129','torchaudio==2.10.0+cu129',
                    'torchvision==0.25.0+cu129','datasets==4.0.0','pyarrow==21.0.0']
TRAINER_PACKAGES=['antlr4-python3-runtime==4.9.3','math-verify==0.8.0','ray==2.53.0',
                  'torchdata==0.11.0','transformers==4.57.1','typing_extensions>=4.14','qwen-vl-utils']


class HardStop(ValueError):
    pass

def verify_files(receipt):
    try:return base_verify_files(receipt)
    except ValueError as exc:raise HardStop(str(exc)) from exc


def verification_allowance(measured):
    return scale(base_verification_allowance(measured))


def require(ok,reason):
    if not ok:raise HardStop(reason)


def campaign():
    """Select existing qualification operations; no second dispatch or training implementation."""
    from kit.v4_campaign import build
    original=build('qualification')
    wanted=['containment-selftest','prepare',*[f'score-agreement{i}-finqa-2048' for i in range(1,5)],
            'scoring-agreement','teacher-finqa','rewrite-finqa','training-set','schedule-q-S-finqa','resolved-configs']
    for arm in 'SFRD':wanted += [f'train-q-{arm}-finqa',f'merge-q-{arm}-finqa',f'score-q-{arm}-finqa-finqa-2048']
    wanted+=['report','PAUSE']
    indexed={r['id']:r for r in original['rows']};rows=[]
    for name in wanted:
        row=copy.deepcopy(indexed[name]);row['needs']=[rows[-1]['id']] if rows else []
        row.pop('wants',None);row.pop('order',None)
        if name=='prepare':row['id']='verify-prepare'
        if name.startswith('train-'):row['needs']+=['scoring-agreement','resolved-configs']
        pilots=[previous['id'] for previous in rows if previous.get('pilot')]
        if pilots:row['needs']=list(dict.fromkeys([*row['needs'],pilots[-1]]))
        command=row['command'];gpu='--time-cap' in command
        if gpu:
            cap=scale(600 if name=='teacher-finqa' else relaxed_timeout(300))
            command[command.index('--time-cap')+1]=str(cap)
            # The Slurm row cap already includes teardown; do not add a duplicate wrapper allowance.
            row['timeout_seconds']=cap
        else:
            cap=600 if name=='containment-selftest' else 120 if name in ('scoring-agreement','training-set','resolved-configs') else 30 if name=='PAUSE' else 60
            if name not in ('containment-selftest','prepare'):cap=relaxed_timeout(cap)
            cap=scale(cap)
            row['timeout_seconds']=row['allocation_cpu_cap_seconds']=cap
        if name!='containment-selftest':
            command[command.index('--row')+1]=row['id']
            for bar in row.get('bars',[]):bar['source']=bar['source'].replace('/prepare.json','/verify-prepare.json')
            command[command.index('{kit}/v4_campaign_ops.py')]='{kit}/v4_phase0.py'
            command[command.index('--phase')+1]='phase0'
        row['pilot']=False
        if name=='containment-selftest':row['bars']=[]
        row['env'].update(V4_PHASE0_MODE='1',V4_COMMAND_TIMEOUT=str(cap),V4_PHASE0_INFERENCE='1' if name=='teacher-finqa' else '0')
        rows.append(row)
    # Every row names its serial predecessor; this is the complete one-allocation unit.
    return {'schema':'kit-campaign.v1','name':'v4-phase0','workdir_env':'WORK','rows':rows,
            'v4':{'phase':'phase0','steps':2,'seeds':[101],'parallel_jobs':1,'scientific_phase_allowed':False,
                  'cpu_prepare':{'command':['python','{kit}/v4_phase0.py','prepare','--work','{work}']},
                  'allocation_stages':{'phase0':{'first':'containment-selftest','last':'PAUSE','block':'qualification','block_limit':100}},
                  'maximum_gpu_hours':ALLOCATION_CAP_GPU_HOURS,
                  'allocation_allowance':1.0, 'outputs':copy.deepcopy(OUTPUT_POLICY),'dispatch_seconds_per_row':DISPATCH_SECONDS,'environment_check_seconds':scale(600),
                  'scoring_gpu':'one stable Slurm-assigned physical GPU','training_questions':20,
                  'scoring_pairs':2,'scoring_questions':50,'post_training_questions':20,
                  'slot_order':[{'id':f'q-{arm}-finqa','arm':arm,'task':'finqa','seed':101,'incoming':None} for arm in 'SFRD']}}


def required_seconds(environment_seconds=scale(600),verification_seconds=scale(60)):
    rows=campaign()['rows']
    return sum(int(r['command'][r['command'].index('--time-cap')+1]) if '--time-cap' in r['command']
               else verification_seconds if r['id']=='verify-prepare' else r['allocation_cpu_cap_seconds'] for r in rows)+len(rows)*DISPATCH_SECONDS+environment_seconds


def label_checkpoint(path):
    """A portable label accompanies every PHASE-0 export, even outside WORK."""
    marker=Path(path)/'phase0-technical-only.json'
    if marker.exists():
        require(json.loads(marker.read_text())==OUTPUT_POLICY,'phase0 checkpoint label changed')
    else:write_durably(marker,OUTPUT_POLICY)


def output_policy_sha256():
    # Match the runner's durable JSON serializer exactly; replay without weights.
    return t.sha(json.dumps(OUTPUT_POLICY,indent=1,sort_keys=True,allow_nan=False)+'\n')


def allocation_cap(ledger, now=None):
    """Full-width actual charge or Start+TimeLimit reservation, including refused jobs."""
    from kit.v4_budget import stamp
    now = time.time() if now is None else now
    used = 0
    for entry in ledger['allocations'].values():
        if not entry.get('phase0'): continue
        require(entry.get('start') and entry.get('width') == 8, 'phase0 allocation cap: unknown exposure')
        end = entry.get('end') or entry.get('planned_end')
        require(end, 'phase0 allocation cap: unknown allocation end')
        start, finish = float(stamp(entry['start'])), float(stamp(end))
        require(finish >= start, 'phase0 allocation cap: invalid allocation interval')
        if not entry.get('end'): finish = max(finish, now)
        used += (finish-start)*entry['width']/3600
    require(used <= ALLOCATION_CAP_GPU_HOURS, 'phase0 allocation cap of 192 GPU-hours reached')
    return used


def verify_presend(work):
    from kit.p4_frozen import seal
    from kit.v4_slurm_capture import verify_capture
    verify_capture(work)
    root=Path(work)/BASE/'presend'
    path=root/'containment-presend.json'
    require(path.is_file(),'partner Slurm paste/check --from-file is absent')
    doc=json.loads(path.read_text());require(doc==seal(doc),'partner Slurm paste check changed')
    binding=json.loads((root/'paste-files.json').read_text());verify_files(binding)
    return doc


def presend(work,paste,qos):
    require(paste is not None and Path(paste).is_dir(),'partner Slurm paste is required; refuse to start')
    require(not os.environ.get('SLURM_JOB_ID'),'check --from-file must run on CPU before allocation')
    folder=Path(work)/BASE/'presend';folder.mkdir(parents=True,exist_ok=True)
    copied=folder/'paste';copied.mkdir(exist_ok=False)
    for source in sorted(Path(paste).iterdir()):
        require(source.is_file() and not source.is_symlink(),'paste must contain ordinary files')
        shutil.copyfile(source,copied/source.name)
    argv=[sys.executable,str(KIT/'p4_contain.py'),'check','--work',str(work),'--out',str(folder),'--from-file',str(copied)]
    if qos:argv+=['--qos',qos]
    observed=bounded_command(argv,timeout=scale(600),log=folder/'check.log')
    write_durably(folder/'check-command.json',{'argv':argv,**observed,'log_sha256':digest(folder/'check.log')})
    result=json.loads((folder/'containment-presend.json').read_text())
    write_durably(folder/'paste-files.json',file_receipt(copied.iterdir()))
    return result


def environment_build(work):
    """Bounded CPU builds: separate venvs; trainer uses the staged CUDA trainer base."""
    require(not os.environ.get('SLURM_JOB_ID'),'environment build must precede GPU allocation')
    cache=site.storage_environment(work,min_gib=30)
    base=site.base_environment(work)
    try:site.check_network(('https://pypi.org/simple/','https://download.pytorch.org/whl/cu129/'))
    except ValueError as exc:
        (Path(work)/BASE/'setup-blocker.txt').write_text(str(exc)+'\n')
        raise
    constraints=Path(work)/BASE/'environment-build/base-constraints.txt'
    constraints.write_text(''.join(name+'=='+version+'\n' for name,version in site.base_constraints(base['inventory']).items()))
    root=Path(work)/'phase0-envs';root.mkdir(parents=True,exist_ok=True)
    trainer=root/'trainer';inference=root/'inference'
    owned=Path(work)/'phase0-source/SDPO';owned.parent.mkdir(parents=True,exist_ok=True)
    from kit.v4_datasets import SDPO_COMMIT
    commands=[('git',['clone','--no-hardlinks',os.environ['SDPO_DIR'],str(owned)]),
              ('git',['-C',str(owned),'checkout','--detach',SDPO_COMMIT]),
              (sys.executable,['-m','venv','--system-site-packages','--without-pip',str(trainer)]),
              (str(trainer/'bin/python'),['-m','pip','install','--timeout',str(scale(600)),'--retries','1','-c',str(constraints),'-e',str(owned),*TRAINER_PACKAGES]),
              (str(trainer/'bin/python'),[str(KIT/'v4_phase0_site.py'),'audit-trainer','--work',str(work)]),
              (str(trainer/'bin/python'),['-m','pip','check']),
              (sys.executable,['-m','venv',str(inference)]),
              (str(inference/'bin/python'),['-m','pip','install','--timeout',str(scale(600)),'--retries','1',*INFERENCE_PACKAGES,'--extra-index-url',
                                          'https://download.pytorch.org/whl/cu129','--only-binary=vllm'])]
    observations=[]
    for i,(python,args) in enumerate(commands):
        if i==3:site.bind_trainer_base(work,trainer,base)
        log=Path(work)/BASE/'environment-build'/f'{i}.log';log.parent.mkdir(parents=True,exist_ok=True)
        result=bounded_command([python,*args],timeout=scale(600 if i in (4,5) else 1800),env={**os.environ,**cache,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},log=log)
        observations.append({'argv':[python,*args],**result,'log_sha256':digest(log)})
        pip_refusal=None
        if i==5:
            output=site.pip_check_output(log.read_bytes())
            delta=site.pip_check_difference(base['pip_check']['output'],output)
            write_durably(log.parent/'trainer-pip-check.json',{'schema':'v4-trainer-pip-check.v1',**result,
                'output':output,'new_lines':delta,'log_sha256':digest(log)})
            if delta:pip_refusal='new pip check problems: '+'; '.join(delta)
        accepted_pip=i==5 and result['returncode'] in (0,1) and result.get('failure_type') in (None,'cpu_exit') and not pip_refusal
        if (result['returncode'] and not accepted_pip) or pip_refusal:
            write_durably(log.parent/'failure.json',{'failure_type':result['failure_type'],'observations':observations})
            message='CPU environment build '+('pip check' if i==5 else 'step '+str(i))+' failed: '+(pip_refusal or str(result['failure_type']))+'; see '+str(log)+'. If login network is blocked, retain the setup blocker and return it. Base environment was not modified.'
            (Path(work)/BASE/'setup-blocker.txt').write_text(message+'\n')
            raise ValueError(message)
    write_durably(Path(work)/BASE/'environment-build/commands.json',{'observations':observations})
    return trainer/'bin/python',inference/'bin/python',owned


def inference_check(work):
    python=os.environ['V4_TEACHER_PYTHON']
    require(Path(python).absolute()!=Path(sys.executable).absolute(),'27B inference environment must be separate from trainer')
    code='''import json,sys,hashlib,time
clock=time.monotonic()
import torch,vllm
from pathlib import Path
from vllm.engine.arg_utils import EngineArgs
import_seconds=time.monotonic()-clock
assert vllm.__version__=='0.18.0'
assert torch.__version__=='2.10.0+cu129' and torch.version.cuda=='12.9'
assert 'gdn_prefill_backend' in EngineArgs.__dataclass_fields__
root=Path(vllm.__file__).parent
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py','.so')}
assert files
identity=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
print(json.dumps({'import_seconds':import_seconds,'package_sha256':identity,'python':sys.executable,'prefix':sys.prefix,'vllm':vllm.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'gdn_prefill_backend':'triton','attention_backend':'FLASH_ATTN'}))
'''
    log=Path(work)/BASE/'environment-build/inference-check.log';log.parent.mkdir(parents=True,exist_ok=True)
    result=bounded_command([python,'-c',code],timeout=scale(600),env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},log=log)
    require(result['returncode']==0,'cpu_inference_environment: '+str(result['failure_type']))
    doc=json.loads(log.read_text().strip().split('\n')[-1])
    require(doc['prefix']!=sys.prefix,'trainer and inference sys.prefix must differ')
    doc.update(check=result,log_sha256=digest(log))
    return doc


def validate_prepare_inputs(work,paste,qos,verified_environments=False):
    """Read-only preflight: report every absent/invalid input before touching WORK."""
    import re
    from kit.v4_contract import INITIAL_8B_REVISION,TOKENIZER_FILES
    from kit.v4_pool import FINQA_HASHES
    from kit.v4_datasets import SDPO_COMMIT
    errors=[]
    keys=('MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER','SDPO_DIR','FINQA_ROOT','V4_KIT_TAG')
    if verified_environments:keys+=('V4_TEACHER_PYTHON',)
    for key in keys:
        value=os.environ.get(key,'').strip()
        if not value:errors.append('missing '+key);continue
        if key=='V4_KIT_TAG':continue
        path=Path(value)
        if not path.is_absolute():errors.append(key+' must be an absolute path')
        if not (path.is_file() if key=='V4_TEACHER_PYTHON' else path.is_dir()):
            errors.append(key+' input is absent: '+value);continue
        if key=='V4_TEACHER_PYTHON' and not os.access(path,os.X_OK):errors.append(key+' is not executable')
        if key in ('MODEL_DIR','TEACHER_MODEL_DIR'):
            revision=INITIAL_8B_REVISION if key=='MODEL_DIR' else '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
            try:
                config=json.loads((path/'config.json').read_text())
                if (config.get('_commit_hash') or path.name)!=revision:errors.append(key+' revision differs: expected '+revision)
                if key=='TEACHER_MODEL_DIR' and (config.get('text_config') or config).get('vocab_size')!=248320:errors.append(key+' is not the registered 27B architecture')
                index=path/'model.safetensors.index.json'
                shards=set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
                if not shards or any(not (path/name).is_file() or not (path/name).stat().st_size for name in shards):errors.append(key+' has missing/empty weight shards')
            except (OSError,ValueError,KeyError,TypeError) as error:errors.append(key+' model input invalid: '+str(error))
        if key in ('QWEN3_8B_TOKENIZER','FINQA_ROOT'):
            expected=TOKENIZER_FILES if key=='QWEN3_8B_TOKENIZER' else {split+'.json':h for split,h in FINQA_HASHES.items()}
            for name,h in expected.items():
                try:
                    if digest(path/name)!=h:errors.append(key+' pinned bytes differ: '+name)
                except OSError:errors.append(key+' missing input: '+name)
        if key=='SDPO_DIR':
            for name in ('pyproject.toml','verl/utils/reward_score/feedback/mcq.py'):
                if not (path/name).is_file():errors.append(key+' missing input: '+name)
            try:
                commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True,timeout=scale(600),stderr=subprocess.DEVNULL).strip()
                if commit!=SDPO_COMMIT:errors.append(key+' commit differs: expected '+SDPO_COMMIT)
            except (OSError,subprocess.SubprocessError):errors.append(key+' is not a readable pinned git checkout')
    if not verified_environments and paste is not None:
        if not qos or not re.fullmatch(r'[A-Za-z0-9_.-]+',qos):errors.append('missing/invalid qos (partner own QOS)')
        if paste is None or not Path(paste).is_dir():errors.append('missing paste directory')
        else:
            for name in ('version.txt','config.txt','partition.txt','qos.txt','cgroup.conf'):
                path=Path(paste)/name
                if not path.is_file() or path.is_symlink() or not path.stat().st_size:errors.append('paste missing/invalid '+name)
            if {p.name for p in Path(paste).iterdir()}!={'version.txt','config.txt','partition.txt','qos.txt','cgroup.conf'}:errors.append('paste must contain exactly the five named raw files')
            for path in Path(paste).iterdir():
                if not path.is_file() or path.is_symlink():errors.append('paste must contain ordinary files: '+path.name)
        required_paste=('version.txt','config.txt','partition.txt','qos.txt','cgroup.conf')
        if paste is not None and qos and all((Path(paste)/n).is_file() and not (Path(paste)/n).is_symlink() for n in required_paste):
            from kit.p4_contain import _cluster_settings
            checked=_cluster_settings(from_file=Path(paste),qos_choice=qos)
            errors.extend('paste settings: '+reason for reason in checked['problems'])
        errors.append('manual paste preparation is retired; use the first CPU capture command')
        if Path(work).exists() and (not Path(work).is_dir() or any(Path(work).iterdir())):errors.append('WORK is not empty; preserve it and choose a fresh WORK after any failed prepare')
    if not verified_environments and paste is None:
        try:
            verify_presend(work)
        except (ValueError,OSError,KeyError) as error:errors.append(str(error))
        if any((Path(work)/name).exists() for name in ('phase0-envs','phase0-source','v4/report-inputs','v4/report-phase0/environment-build')):
            errors.append('preparation already started; preserve it and choose a fresh WORK, then capture again')
    if os.environ.get('V4_TELEMETRY','1')!='1':errors.append('V4_TELEMETRY must equal 1')
    if os.environ.get('WORK') and Path(os.environ['WORK']).resolve()!=Path(work).resolve():errors.append('WORK environment differs from --work')
    if errors:raise ValueError('CPU prepare input validation refused:\n- '+'\n- '.join(errors))


def frozen_launch_environment(work):
    """Planning consumes preparation's frozen paths, never login-shell defaults."""
    frozen=json.loads((Path(work)/'v4/report-inputs/launch-inputs-phase0.json').read_text())
    env=frozen['environment']
    required={'MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER','V4_TEACHER_PYTHON','SDPO_DIR','FINQA_ROOT','V4_TELEMETRY','WORK','KIT','PYTHONPATH','V4_KIT_TAG'}
    require(required<=env.keys() and all(isinstance(v,str) and v for v in env.values()),'prepared launch environment incomplete')
    require(env['WORK']==str(Path(work).resolve()) and env['KIT']==str(KIT) and env['V4_TELEMETRY']=='1','prepared launch environment identity differs')
    return env


def prepare(work,paste=None,qos=None,verified_environments=False):
    require(not os.environ.get('SLURM_JOB_ID'),'CPU prepare must precede any GPU allocation')
    work=Path(work).resolve()
    validate_prepare_inputs(work,paste,qos,verified_environments)
    from kit.v4_phase0_download import verify_download_receipt
    verify_download_receipt(work)
    with site.storage(work):
        folder=work/BASE;folder.mkdir(parents=True,exist_ok=True)
        try:disk=site.planning_storage(work,{**os.environ,'TMPDIR':site.JOB_TMP})
        except (ValueError,OSError) as exc:
            (folder/'setup-blocker.txt').write_text(str(exc)+'\n')
            raise
        disk['login_tmpdir']=os.environ['TMPDIR']
        original=folder/'preparation-storage.json'
        storage_receipt=folder/'preparation-storage-trainer.json' if verified_environments and original.exists() else original
        write_durably(storage_receipt,disk)
        return _prepare_owned(work,verified_environments)


def _prepare_owned(work,verified_environments):
    work.mkdir(parents=True,exist_ok=True)
    if not verified_environments:
        verify_presend(work)
        trainer,inference,owned=environment_build(work)
        result=bounded_command([str(trainer),str(KIT/'v4_phase0.py'),'prepare','--work',str(work),'--verified-environments'],
            timeout=scale(3600),env={**os.environ,'WORK':str(work),'V4_TEACHER_PYTHON':str(inference),'SDPO_DIR':str(owned),'PYTHONPATH':str(KIT.parent)+os.pathsep+str(owned),'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
            log=work/BASE/'environment-build/trainer-prepare.log')
        require(result['returncode']==0,'cpu_prepare: '+str(result['failure_type']))
        return work/'v4/report-inputs/prepare-receipt-phase0.json'
    verify_presend(work)
    trainer_torch_runtime=site.torch_runtime_identity()
    site.validate_base_versions(site.installed_versions(),trainer_torch_runtime)
    from kit.v4_prepare import import_smoke
    smoke=import_smoke(work,'phase0')  # FIRST operation in the built trainer environment.
    inference=inference_check(work)
    from kit.v4_phase0_environment import admitted_environment_timing
    imports=json.loads(Path(smoke).read_text())['observations']
    measured=next(o['seconds'] for o in imports if o['name']=='phase0-trainer-imports' and o['returncode']==0)
    try:timing=admitted_environment_timing({'trainer':measured,'inference':inference['import_seconds']})
    except ValueError as exc:
        blocker=work/BASE/'setup-blocker.txt';blocker.parent.mkdir(parents=True,exist_ok=True)
        blocker.write_text(str(exc)+'\n')
        raise
    from kit.v4_pool import FINQA_HASHES
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kit.v4_evidence import pool_items
    inputs=work/'v4/report-inputs';inputs.mkdir(parents=True,exist_ok=True)
    source_root=inputs/'sources/FinQA';source_root.mkdir(parents=True,exist_ok=True)
    for split in ('train','test'):
        source=Path(os.environ['FINQA_ROOT'])/(split+'.json')
        require(digest(source)==FINQA_HASHES[split],'pinned FinQA source changed: '+split)
        shutil.copyfile(source,source_root/source.name)
    rows=t.authors_rows('finqa');pool=pool_items(rows,'finqa')
    panel=t.finqa.rows_for_trainer(t.finqa.load(source_root,'test'),'test')
    t.write_jsonl(inputs/'finqa-pool.jsonl',pool);t.write_jsonl(inputs/'finqa-panel.jsonl',panel)
    base=work/BASE;base.mkdir(parents=True,exist_ok=True);t.write_jsonl(base/'finqa-pool.jsonl',pool[:20])
    dataset=work/'v4/datasets/datasets/v4_finqa';dataset.mkdir(parents=True,exist_ok=True)
    for split,data in (('train',rows),('test',panel)):pq.write_table(pa.Table.from_pylist(data),dataset/(split+'.parquet'))
    token_dir=inputs/'tokenizer';token_dir.mkdir(exist_ok=True)
    from kit.v4_contract import TOKENIZER_FILES,pinned_tokenizer
    for name in TOKENIZER_FILES:shutil.copyfile(Path(os.environ['QWEN3_8B_TOKENIZER'])/name,token_dir/name)
    tokenizer=pinned_tokenizer()
    lengths=[{'id':i['id'],'tokens':len(tokenizer.apply_chat_template(i['messages'],add_generation_prompt=True,enable_thinking=False))} for i in pool[:20]]
    write_durably(inputs/'static-tokenisation-phase0.json',lengths)
    shutil.copyfile(Path(os.environ['SDPO_DIR'])/'verl/utils/reward_score/feedback/mcq.py',inputs/'mcq.py')
    shutil.copyfile(KIT/'v4_departures.json',inputs/'v4_departures.json')
    write_durably(inputs/'model-identities.json',{'initial':t.local_identity(os.environ['MODEL_DIR'],'Qwen/Qwen3-8B'),
                                               'teacher':t.local_identity(os.environ['TEACHER_MODEL_DIR'])})
    trainer_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=os.environ['SDPO_DIR'],text=True,timeout=scale(600)).strip()
    from kit.v4_datasets import SDPO_COMMIT
    require(trainer_commit==SDPO_COMMIT,'pinned trainer commit differs')
    from kit.v4_archive import DirectoryEvidence
    from kit.v4_evidence import environment
    host_base=json.loads((base/'environment-build/base-environment.json').read_text())
    site.validate_base_versions(host_base['versions'],host_base.get('torch_runtime'))
    trainer_inventory=json.loads((base/'environment-build/trainer-inventory.json').read_text())
    trainer_pip_check=json.loads((base/'environment-build/trainer-pip-check.json').read_text())
    write_durably(base/'environment.json',{'trainer_pip_check':trainer_pip_check,'trainer_inventory':trainer_inventory['visible'],'trainer_own_inventory':trainer_inventory['own'],'host_base':host_base,'kit_tag':os.environ.get('V4_KIT_TAG') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=KIT,text=True,timeout=scale(600)).strip(),
        'trainer_dependency_versions':site.installed_versions(),'trainer_torch_runtime':trainer_torch_runtime,'runtime_versions':t.runtime_versions(),'trainer_commit':trainer_commit,
        'inference_package_identity_sha256':inference['package_sha256'],'inference':inference,'trainer_prefix':sys.prefix})
    environment(DirectoryEvidence(work),'phase0')
    write_durably(inputs/'phase0-campaign.json',campaign())
    launch={key:str(Path(os.environ[key]).absolute()) for key in ('MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER','V4_TEACHER_PYTHON','SDPO_DIR','FINQA_ROOT')}
    env={**launch,'WORK':str(work),'KIT':str(KIT),'V4_TELEMETRY':'1','V4_KIT_TAG':os.environ['V4_KIT_TAG'],
         'PYTHONPATH':str(KIT.parent)+':'+launch['SDPO_DIR'],'TASK_ROOT':str(Path(os.environ.get('TASK_ROOT',work.parent)).resolve()),'V4_PHASE0_MODE':'1',
         **site.storage_environment(work,allocation=True),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    write_durably(inputs/'launch-inputs-phase0.json',{'paths':launch,'environment':env,'trainer_prefix':sys.prefix})
    paths=[base/'preparation-storage.json',base/'environment-build/base-pip-check.log',base/'environment-build/trainer-pip-check.json',base/'environment-build/5.log',base/'environment-build/trainer-inventory.json',base/'environment-build/base-inheritance.json',base/'environment-build/base-environment.json',base/'environment-build/base-constraints.txt',base/'download/receipt.json',base/'finqa-pool.jsonl',*[p for p in (work/'v4/datasets').rglob('*') if p.is_file()]]
    if (base/'preparation-storage-trainer.json').exists():paths.append(base/'preparation-storage-trainer.json')
    paths += [p for p in inputs.rglob('*') if p.is_file()]+[p for p in KIT.rglob('*') if p.suffix in ('.py','.sh','.yaml','.json') and '__pycache__' not in p.parts]
    paths += [p for p in Path(os.environ['SDPO_DIR']).rglob('*.py') if '__pycache__' not in p.parts]
    paths += [p for p in (work/'phase0-envs').rglob('*') if p.is_file() and p.suffix in ('.py','.so','.pth') and '__pycache__' not in p.parts]
    for key in ('MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER'):paths += [p for p in Path(os.environ[key]).rglob('*') if p.is_file()]
    archived={str(p.relative_to(work)):digest(p) for p in inputs.rglob('*') if p.is_file()}
    archived.update({str(p.relative_to(work)):digest(p) for p in (base/'preparation-storage.json',base/'preparation-storage-trainer.json') if p.exists()})
    write_durably(inputs/'prepare-receipt-phase0.json',{'schema':'v4-cpu-prepare.v1','phase':'phase0','before_allocation':True,
        'environment_check':timing,'import_smoke_sha256':digest(smoke),'archived_files':archived,**file_receipt(set(paths))})
    measure_prepare_verification(work)
    return inputs/'prepare-receipt-phase0.json'


def measure_prepare_verification(work):
    """Time the exact immutable-byte verifier on CPU before its allocation row."""
    started=time.monotonic()
    doc=verify_prepare(work)
    require('verification_seconds' not in doc,'prepare verification measurement is already frozen')
    doc['verification_seconds']=time.monotonic()-started
    doc['verification_allowance_seconds']=verification_allowance(doc['verification_seconds'])
    from kit.p4_watchdog import write_durably as finalize_prepare
    finalize_prepare(Path(work)/'v4/report-inputs/prepare-receipt-phase0.json',doc)
    return doc['verification_seconds']


def verify_prepare(work,*,rehash=True):
    verify_presend(work)
    path=Path(work)/'v4/report-inputs/prepare-receipt-phase0.json'
    require(path.is_file(),'phase0 CPU prepare is absent')
    doc=json.loads(path.read_text());require(doc['phase']=='phase0' and doc['before_allocation'] is True,'phase0 prepare identity differs')
    from kit.v4_phase0_environment import prepared_deadline
    prepared_deadline(work)
    if rehash:verify_files(doc)
    return doc


def operation(work,name,**args):
    """Delegate real bodies to the reviewed campaign operations."""
    from kit import v4_campaign_ops as ops
    base=Path(work)/BASE;base.mkdir(parents=True,exist_ok=True)
    if name=='verify-prepare':return verify_prepare(work)
    # Every row rechecks frozen data/code hashes; initial verify also hashes models/envs.
    receipt=json.loads((Path(work)/'v4/report-inputs/prepare-receipt-phase0.json').read_text())
    frozen=json.loads((Path(work)/'v4/report-inputs/launch-inputs-phase0.json').read_text())
    require(all(str(Path(os.environ[key]).absolute())==value for key,value in frozen['paths'].items()) and sys.prefix==frozen['trainer_prefix'],'phase0 launch environment/path differs from CPU preparation')
    subset={name:record for name,record in receipt['files'].items() if Path(name).is_relative_to(Path(work)/'v4/report-inputs') or Path(name).is_relative_to(KIT) or Path(name).is_relative_to(Path(work)/'v4/datasets') or Path(name)==Path(work)/BASE/'finqa-pool.jsonl'}
    verify_files({'files':subset})
    if name in ('teacher','rewrite'):
        return ops.teacher(work,'phase0','finqa',name=='rewrite')
    if name=='training-set':return ops.training_set(work,'phase0')
    if name=='schedule':return ops.schedule(work,'phase0',args['slot'])
    if name=='configs':return ops.qualification_configs(work,'phase0','finqa')
    if name=='train':return ops.train(work,'phase0',args['slot'])
    if name=='merge':return ops.merge(work,'phase0',args['slot'])
    if name=='score':return ops.score(work,'phase0',args['checkpoint'],'finqa',2048)
    if name in ('agreement','report'):
        from kit.v4_archive import DirectoryEvidence
        evidence=DirectoryEvidence(work)
        result=agreement(evidence) if name=='agreement' else evidence_report(evidence,final=False)
        write_durably(base/('scoring-agreement.json' if name=='agreement' else 'reading.json'),result)
        return result
    if name=='pause':return write_durably(base/'pause.json',{'state':'PAUSE','scientific_phase_allowed':False,'owner_review_required':True})
    raise ValueError('unknown phase0 operation: '+name)


def paste_problems(pasted,observed):
    """CPU paste has no running-job fields; compare every setting it actually names."""
    from kit.p4_contain import SAFETY_FIELDS
    def changed(expected,actual):
        if expected is None:return False
        if isinstance(expected,dict):return not isinstance(actual,dict) or any(changed(v,actual.get(k)) for k,v in expected.items())
        return expected!=actual
    return [key for key in SAFETY_FIELDS if key not in ('gpu_count','expected','simulation') and changed(pasted.get(key),observed.get(key))]


def verify_allocation_plan(work):
    """After selftest: require the exact CPU plan and actual 8-GPU time reservation."""
    from kit.v4_budget import stamp
    path=os.environ.get('V4_ALLOCATION_PLAN')
    require(path,'phase0 allocation plan is absent')
    raw=Path(path).read_bytes()
    require(t.sha(raw)==os.environ.get('V4_ALLOCATION_PLAN_SHA256'),'phase0 plan hash differs')
    doc=json.loads(raw)
    require(doc['phase']=='phase0' and doc['stage']=='phase0' and doc['block']=='qualification' and doc['block_limit']==100 and doc['ceiling']==560,'phase0 plan registration differs')
    seconds=sum(int(n)*unit for n,unit in zip(doc['sbatch_time'].split(':'),(3600,60,1)))
    require(seconds<=86400 and seconds*8/3600<=ALLOCATION_CAP_GPU_HOURS and doc.get('phase0_allocation_cap_gpu_hours')==ALLOCATION_CAP_GPU_HOURS and doc['gpus']==8,'phase0 allocation window differs')
    verification_plan(doc,json.loads((Path(work)/'v4/report-inputs/prepare-receipt-phase0.json').read_text()))
    pasted=verify_presend(work)
    for key,path in {'phase0_prepare_sha256':'v4/report-inputs/prepare-receipt-phase0.json','phase0_presend_sha256':BASE+'presend/containment-presend.json'}.items():
        require(doc[key]==digest(Path(work)/path),'phase0 CPU evidence changed since planning')
    from kit.p4_contain import hard_stop
    ledger=json.loads((Path(work)/'k8b4/containment/allocation-ledger.json').read_text())
    job=os.environ['SLURM_JOB_ID'];entry=ledger['allocations'][job]
    entry['phase0']=True;allocation_cap(ledger)
    require(entry['width']==8 and stamp(entry['planned_end'])-stamp(entry['start'])==seconds and entry['segments'][0]['block']=='qualification','phase0 actual allocation differs from plan')
    prior={key:{field:value[field] for field in ('start','end','width','segments')} for key,value in ledger['allocations'].items() if key!=job}
    require(prior==doc['prior_allocations'],'phase0 accounting changed since planning')
    return doc


@contextmanager
def finqa_context(evidence,tokenizer):
    """Rebuild FinQA from pinned raw source bytes, with no Chemistry prerequisite."""
    from kit import v4_pool as pool,v4_contract as contract
    from kit.v4_evidence import pool_items
    from kit.v4_readers import MCQ_SHA256
    for split,digest_value in pool.FINQA_HASHES.items():
        require(t.sha(evidence.files['v4/report-inputs/sources/FinQA/'+split+'.json'])==digest_value,'pinned FinQA source changed: '+split)
    old=contract.pinned_tokenizer;old_authors=t.authors_rows
    with tempfile.TemporaryDirectory(prefix='phase0-finqa-') as temp:
        root=Path(temp)
        for split in ('train','test'):(root/(split+'.json')).write_bytes(evidence.files['v4/report-inputs/sources/FinQA/'+split+'.json'])
        try:
            contract.pinned_tokenizer=lambda:tokenizer;contract._cached_prompt_length.cache_clear()
            rows=t.authors_rows('finqa',finqa_root=root)
            panels=t.finqa.rows_for_trainer(t.finqa.load(root,'test'),'test')
            items=pool_items(rows,'finqa')
            require(evidence.jsonl('v4/report-inputs/finqa-pool.jsonl')==items,'original FinQA pool differs')
            require(evidence.jsonl('v4/report-inputs/finqa-panel.jsonl')==panels,'FinQA heldout panel differs')
            evidence.expected_pools={'finqa':items};evidence.expected_panels={'finqa':panels}
            t.authors_rows=lambda task,*args,**kwargs:copy.deepcopy(rows) if task=='finqa' else old_authors(task,*args,**kwargs)
            yield
        finally:
            contract.pinned_tokenizer=old;t.authors_rows=old_authors;contract._cached_prompt_length.cache_clear()


def agreement(evidence):
    from kit.v4_readers import scorer_context
    from kit.v4_qualification import archive_agreement
    from kit.v4_evidence import scheduler_rows
    with tempfile.TemporaryDirectory(prefix='phase0-score-') as temp:
        tokenizer,mcq=scorer_context(evidence,temp)
        try:scheduler_rows(evidence,'phase0')
        except (ValueError,KeyError) as exc:print('WARNING: informational scheduler/containment observation: '+str(exc),file=sys.stderr)
        with finqa_context(evidence,tokenizer):return archive_agreement(evidence,mcq,'phase0',('finqa',))


def archived_presend(evidence):
    from kit.p4_frozen import seal
    from kit.v4_evidence import reparse_settings
    doc=evidence.json(BASE+'presend/containment-presend.json')
    require(doc==seal(doc) and doc.get('schema')=='kit-v4-containment-presend.v1','missing/changed partner paste check')
    reparse_settings(doc)
    binding=evidence.json(BASE+'presend/paste-files.json')
    require({Path(n).name for n in binding.get('files',{})}=={'version.txt','config.txt','partition.txt','qos.txt','cgroup.conf'},'missing partner paste bytes')
    for name,record in binding['files'].items():
        raw=evidence.files[BASE+'presend/paste/'+Path(name).name]
        require(t.sha(raw)==record['sha256'] and len(raw)==record['bytes'],'partner paste bytes changed')
        matched=[c for c in doc['commands'] if (c['argv'][0]=='read-paste' and Path(c['argv'][-1]).name==Path(name).name)
                 or (Path(name).name=='version.txt' and c['argv'][:2]==['scontrol','--version'])
                 or (Path(name).name=='config.txt' and c['argv'][:3]==['scontrol','show','config'])
                 or (Path(name).name=='partition.txt' and c['argv'][:3]==['scontrol','show','partition'])
                 or (Path(name).name=='qos.txt' and c['argv'][:1]==['sacctmgr'])
                 or (Path(name).name=='cgroup.conf' and c['argv'][:1]==['read-cgroup-conf'])]
        require(len(matched)==1 and matched[0]['stdout'].encode()==raw,'partner paste bytes differ from checked observations')
    return doc


def archived_prepare(evidence):
    archived_presend(evidence)
    from kit.v4_slurm_capture import validate_capture,NAMES
    captured=validate_capture(evidence.json(BASE+'presend/capture.json'),{name:evidence.files[BASE+'presend/paste/'+name] for name in NAMES},evidence.files[BASE+'presend/containment-presend.json'])
    require(not captured.get('synthetic'),'Slurm capture is synthetic')

    receipt=evidence.json('v4/report-inputs/prepare-receipt-phase0.json')
    require(receipt['phase']=='phase0' and receipt['before_allocation'] is True,'missing CPU prepare before allocation')
    measured=receipt.get('verification_seconds')
    allowance=verification_allowance(measured)
    require(receipt.get('verification_allowance_seconds',allowance)==allowance,'prepared verification allowance differs from CPU measurement')
    for name,digest_value in receipt['archived_files'].items():require(t.sha(evidence.files[name])==digest_value,'prepared archive input changed: '+name)
    require(evidence.json('v4/report-inputs/phase0-campaign.json')==campaign(),'frozen phase0 graph changed')
    raw=evidence.files['v4/report-inputs/import-smoke-phase0/receipt.json']
    require(t.sha(raw)==receipt['import_smoke_sha256'],'CPU smoke hash differs')
    smoke=json.loads(raw)
    from kit.v4_prepare import smoke_inventory
    inv=smoke_inventory()
    require(smoke['ok'] is True and smoke['inventory']['modules']==inv['modules'] and smoke['inventory']['dependencies']==inv['dependencies'],'CPU smoke inventory incomplete')
    required={'import-'+n for n in inv['modules']+inv['dependencies']}|{'launcher-'+a for a in 'SFRD'}|{'phase0-trainer-imports','first-sft-cpu-batch','entry-help-sft_entry.py','entry-help-sdft_entry.py','entry-help-S'}
    require(required<={o['name'] for o in smoke['observations'] if o['returncode']==0},'CPU entry/first-batch smoke incomplete')
    for i,o in enumerate(smoke['observations']):require(t.sha(evidence.files[f'v4/report-inputs/import-smoke-phase0/{i}.log'])==o['log_sha256'],'CPU smoke log changed')
    env=evidence.json(BASE+'environment.json')
    host_base=evidence.json(BASE+'environment-build/base-environment.json')
    site.validate_base_versions(host_base['versions'],host_base.get('torch_runtime'))
    site.validate_base_versions(env['trainer_dependency_versions'],env.get('trainer_torch_runtime'))
    require(env.get('host_base')==host_base and host_base.get('route')=='host-venv','host base receipt binding differs')
    bridge=evidence.json(BASE+'environment-build/base-inheritance.json')
    require(bridge['enabled']==host_base['is_venv'] and host_base['is_venv']==(host_base['prefix']!=host_base['base_prefix']) and bridge['base_site_packages']==host_base['site_packages'],'native base inheritance differs')
    if bridge['enabled']:
        require(bridge['sha256']==t.sha(site.base_bridge_text(host_base['site_packages']).encode()) and Path(bridge['path']).is_relative_to(Path(env['trainer_prefix'])) and Path(bridge['path']).name=='phase0-host-base.pth','native base bridge path differs')
    else:require(bridge['path'] is None and bridge['sha256'] is None,'unnecessary native base bridge')
    inventory=evidence.json(BASE+'environment-build/trainer-inventory.json')
    require(inventory['prefix']==env['trainer_prefix'] and inventory['visible']==env['trainer_inventory'] and inventory['own']==env['trainer_own_inventory'],'trainer inventory binding differs')
    site.validate_shadowing(host_base['inventory'],inventory['own'],Path(env['trainer_prefix']).parents[1]/'phase0-source/SDPO')
    download=evidence.json(BASE+'download/receipt.json')
    from kit.v4_phase0_download import MODELS
    require(download.get('schema')=='v4-phase0-download.v1' and download.get('allocation_gpu_hours')==0 and download['slurm']['gpus']==0,'zero-GPU download receipt differs')
    for role,(repo,revision) in MODELS.items():
        model=download['models'][role]
        require(model['repo']==repo and model['revision']==revision and model['revision_dir']==revision and model['bytes']>0 and model['bytes']==sum(f['bytes'] for f in model['files'].values()),'pinned download inventory differs')
    require(smoke['environment']['trainer_commit']==env['trainer_commit'] and smoke['environment']['runtime_versions']==env['runtime_versions'],'CPU smoke environment differs')
    inference=env['inference']
    require(inference['vllm']=='0.18.0' and inference['torch']=='2.10.0+cu129' and inference['cuda']=='12.9' and inference['gdn_prefill_backend']=='triton' and inference['attention_backend']=='FLASH_ATTN','27B environment pin/backend differs')
    require(inference['prefix']!=env['trainer_prefix'] and inference['check']['returncode']==0,'27B environment is not separate or verified')
    raw_check=evidence.files[BASE+'environment-build/inference-check.log']
    require(t.sha(raw_check)==inference['log_sha256'],'27B CPU check changed')
    observed=json.loads(raw_check.decode().strip().split('\n')[-1])
    require(all(inference.get(k)==v for k,v in observed.items()),'27B CPU check/receipt differs')
    from kit.v4_phase0_environment import admitted_environment_timing
    measured=next(o['seconds'] for o in smoke['observations'] if o['name']=='phase0-trainer-imports')
    require(receipt.get('environment_check')==admitted_environment_timing({'trainer':measured,'inference':inference['import_seconds']}),'prepare environment timing derivation differs')
    commands=evidence.json(BASE+'environment-build/commands.json')['observations']
    require(len(commands)==8 and all(o['returncode']==0 for i,o in enumerate(commands) if i!=5) and commands[5]['returncode'] in (0,1),'CPU environment builds incomplete')
    from kit.v4_datasets import SDPO_COMMIT
    owned=commands[0]['argv'][-1];trainer=env['trainer_prefix'];infer=inference['prefix']
    require(commands[0]['argv'][:3]==['git','clone','--no-hardlinks'] and commands[1]['argv']==['git','-C',owned,'checkout','--detach',SDPO_COMMIT],'owned trainer checkout recipe differs')
    require(commands[2]['argv'][1:]==['-m','venv','--system-site-packages','--without-pip',trainer] and commands[3]['argv']==[trainer+'/bin/python','-m','pip','install','--timeout',str(scale(600)),'--retries','1','-c',commands[3]['argv'][9],'-e',owned,*TRAINER_PACKAGES],'trainer build recipe differs')
    require(commands[6]['argv'][1:]==['-m','venv',infer] and commands[7]['argv']==[infer+'/bin/python','-m','pip','install','--timeout',str(scale(600)),'--retries','1',*INFERENCE_PACKAGES,'--extra-index-url','https://download.pytorch.org/whl/cu129','--only-binary=vllm'],'separate inference build recipe differs')
    require(commands[4]['argv'][0]==trainer+'/bin/python' and Path(commands[4]['argv'][1]).name=='v4_phase0_site.py' and commands[4]['argv'][2:]==['audit-trainer','--work',str(Path(trainer).parents[1])] and commands[5]['argv']==[trainer+'/bin/python','-m','pip','check'],'trainer audit/pip check recipe differs')
    base_check=host_base['pip_check'];trainer_check=evidence.json(BASE+'environment-build/trainer-pip-check.json')
    base_raw=evidence.files[BASE+'environment-build/base-pip-check.log'];trainer_raw=evidence.files[BASE+'environment-build/5.log']
    require(base_check['argv']==[host_base['python'],'-m','pip','check'] and base_check['returncode'] in (0,1) and base_check.get('failure_type') in (None,'cpu_exit'),'base pip check did not finish')
    require(t.sha(base_raw)==base_check['log_sha256'] and t.sha(trainer_raw)==trainer_check['log_sha256'] and trainer_check['returncode']==commands[5]['returncode'] and trainer_check.get('failure_type') in (None,'cpu_exit') and trainer_check==env['trainer_pip_check'],'pip check receipt binding differs')
    site.verify_pip_checks(base_check,trainer_check,base_raw,trainer_raw)
    constraints=site.base_constraints(host_base['inventory'])
    require(Path(commands[3]['argv'][9]).name=='base-constraints.txt','trainer pip constraint path differs')
    require(dict(line.split('==',1) for line in evidence.files[BASE+'environment-build/base-constraints.txt'].decode().splitlines())==constraints,'trainer constraints differ from native base')
    for i,o in enumerate(commands):require(t.sha(evidence.files[BASE+f'environment-build/{i}.log'])==o['log_sha256'],'environment build log changed')
    return {'before_allocation':True,'trainer':env['trainer_prefix'],'inference':inference}


def rehearsal_report(evidence,report):
    """Diagnose synthetic archives from actual parser/runner evidence; never PASS."""
    from kit.v4_qualification import compare_pairs
    prefix='v4/report-simulation/'
    graph=evidence.json(prefix+'campaign.json')
    require(graph.get('name')=='v4-phase0' and graph['v4']['phase']=='CPU-phase0-rehearsal','CPU rehearsal graph differs')
    rows=graph['rows']
    require([r['id'] for r in rows]==[r['id'] for r in campaign()['rows']],'CPU rehearsal row inventory differs')
    # Missing paste is a pre-allocation refusal even if the stand-in runner
    # archived its failed row. Other cases must pass raw paste reconstruction.
    archived_presend(evidence)
    report['tables']['presend']={'ok':True,'synthetic':True,'raw_files':5}
    graph_sha=t.sha(evidence.files[prefix+'campaign.json'])
    states={}
    for row in rows:
        root='campaign/v4-phase0/'+row['id']+'/attempt-1/'
        if root+'start.json' not in evidence.files:break
        start=evidence.json(root+'start.json');verdict=evidence.json(root+'verdict.json')
        require(start.get('campaign_sha256')==graph_sha and start.get('campaign')=='v4-phase0' and start.get('row')==row['id'] and verdict.get('row')==row['id'],'CPU rehearsal runner identity differs')
        states[row['id']]=verdict['verdict'];report['tables']['runner']=states
        if row['id']=='scoring-agreement':
            pairs=evidence.json(prefix+'scoring-pairs.json')
            report['tables']['determinism']=compare_pairs(pairs,('finqa',))
        if verdict['verdict']!='PASS':
            log=evidence.files.get(root+'output.log',b'').decode(errors='replace')
            if row['id']=='containment-selftest':
                report.setdefault('warnings',[]).append('WARNING: containment self-test FAILED; informational only; owner monitors the run.')
                continue
            if 'CUDA out of memory' in log:raise ValueError('out_of_memory: '+row['id'])
            if verdict.get('failure_type')=='cpu_deadline':raise ValueError('wall_time: bounded CPU deadline: '+row['id'])
            raise ValueError('CPU rehearsal row failure: '+row['id'])
    require(len(states)==len(rows) and all(v=='PASS' for k,v in states.items() if k!='containment-selftest'),'CPU rehearsal runner incomplete')
    raise ValueError('CPU stand-ins: runner and determinism checks reproduced; real GPU containment, models and training are absent')


def evidence_report(evidence,final=True):
    """Recompute the technical receipt from raw evidence, never a coverage/pass summary."""
    report={'schema':'v4-phase0-reading.v1','status':'incomplete','reasons':[],
            'scientific_phase_allowed':False,'owner_review_required':True,'usable_as_scientific_initialisations':False,'outputs_scope':'technical','tables':{}}
    report['warnings']=[]
    for name in evidence.files:
        if name.endswith('containment-selftest.json'):
            doc=evidence.json(name)
            if not doc.get('ok'):report['warnings'].append('WARNING: containment self-test FAILED; informational only; owner monitors the run.')
    try:
        from kit.v4_readers import verdicts,scorer_context,corpus,run_record,scoring
        from kit.v4_evidence import environment,containment,scheduler_rows,common_data,training_provenance
        from kit.v4_qualification import validate_configs,departures_hash
        from kit.v4_budget import allocation_spend,number
        from kit.v4_campaign_ops import job_for
        from kit.v4_phase0_submission import archived_submission_accounting,live_submission_accounting
        submission_job=None
        if final:archived_submission_accounting(evidence,report)
        else:submission_job=live_submission_accounting(evidence,report)
        # Failed engines precede raw output parsing and agreement arithmetic.
        for checkpoint in [*('agreement'+str(i) for i in range(1,5)),*(f'q-{a}-finqa' for a in 'SFRD')]:
            name=BASE+f'eval/{checkpoint}-finqa-2048/engine-status.json'
            if name in evidence.files:
                engine=evidence.json(name)
                require(engine.get('engine_ok') is True,'scoring_engine_start: finqa')
                require(engine.get('configuration_ok') is True,'scoring_engine_configuration: finqa')
        if 'v4/report-simulation/campaign.json' in evidence.files:return rehearsal_report(evidence,report)
        if BASE+'presend/capture.json' in evidence.files:
            from kit.v4_slurm_capture import validate_capture,NAMES
            captured=validate_capture(evidence.json(BASE+'presend/capture.json'),
                {name:evidence.files[BASE+'presend/paste/'+name] for name in NAMES},evidence.files[BASE+'presend/containment-presend.json'])
            doc=evidence.json(BASE+'presend/containment-presend.json')
            report['tables']['slurm_capture']={'ok':captured['ok'],'site':captured['site'],'problems':doc.get('problems',[]),'observations':captured['observations']}
            if not captured['ok']:report.setdefault('warnings',[]).append('Informational Slurm capture: '+'; '.join(captured.get('problems',[])))
        blocker=BASE+'setup-blocker.txt'
        if blocker in evidence.files:
            raise ValueError('Setup blocker: '+evidence.files[blocker].decode(errors='replace').strip())
        check=BASE+'environment-check.json'
        if check in evidence.files and evidence.json(check).get('hard_stop'):
            raise ValueError('Setup blocker: '+evidence.json(check).get('message','host-venv environment check refused'))
        if 'v4/report-inputs/prepare-receipt-phase0.json' not in evidence.files:
            if report['tables'].get('slurm_capture',{}).get('ok'):
                raise ValueError('Capture-only archive: Slurm capture passed; CPU prepare and GPU qualification have not completed. Return this evidence to the owner; no technical pass.')
            raise ValueError('CPU prepare is absent or incomplete; return the setup evidence to the owner.')
        report['tables']['prepare']=archived_prepare(evidence)
        report['tables']['environment']=environment(evidence,'phase0')
        if final:
            ledger=evidence.json('k8b4/containment/allocation-ledger.json')
            plan_names=[n for n in evidence.files if n.startswith('v4/report-phase0/allocation/') and n.endswith('.json')]
            require(len(plan_names)==1,'phase0 requires one retained allocation plan')
            plan=evidence.json(plan_names[0]);require(plan['phase']=='phase0' and plan['block']=='qualification' and plan['gpus']==8,'wrong phase0 allocation plan')
            new={job:entry for job,entry in ledger['allocations'].items() if job not in plan['prior_allocations']}
            require(len(new)==1,'phase0 must use exactly one new allocation')
            require(all(entry.get('end') for entry in new.values()),'phase0 allocation is still open')
            pass # Historical containment diagnostics do not veto phase 0.
            report['tables']['containment']={'informational':True,'selftests':[evidence.json(n) for n in evidence.files if n.endswith('containment-selftest.json')]}
            seconds=sum(int(n)*unit for n,unit in zip(plan['sbatch_time'].split(':'),(3600,60,1)))
            require(seconds<=86400 and plan.get('phase0_allocation_cap_gpu_hours')==ALLOCATION_CAP_GPU_HOURS,'phase0 plan window differs')
            verification_plan(plan,evidence.json('v4/report-inputs/prepare-receipt-phase0.json'))
            for key,path in {'phase0_prepare_sha256':'v4/report-inputs/prepare-receipt-phase0.json','phase0_presend_sha256':BASE+'presend/containment-presend.json'}.items():
                require(plan[key]==t.sha(evidence.files[path]),'phase0 planned CPU evidence binding differs')
            total,all_blocks=allocation_spend(ledger)
            report['qualification_gpu_hours']=float(all_blocks.get('qualification',0))
            used,blocks=allocation_spend({'allocations':new})
            require(0<used<=ALLOCATION_CAP_GPU_HOURS and set(blocks)=={'qualification'},'phase0 exceeds 192 allocation GPU-hours cap or wrong block')
            report['allocation_gpu_hours']=float(used)
        try:scheduler_rows(evidence,'phase0')
        except (ValueError,KeyError) as exc:report.setdefault('warnings',[]).append('Informational scheduler/containment observation: '+str(exc))
        if final or submission_job is not None:
            job=next(iter(new)) if final else submission_job
            require(all(record['allocation_id']==job for (phase,row),record in evidence.scheduler_rows.items() if phase=='phase0'),'phase0 GPU rows used different allocations')
        with tempfile.TemporaryDirectory(prefix='phase0-read-') as temp:
            tokenizer,mcq=scorer_context(evidence,temp)
            with finqa_context(evidence,tokenizer):
                from kit.v4_qualification import archive_agreement
                agree=archive_agreement(evidence,mcq,'phase0',('finqa',));report['tables']['determinism']=agree
                report['tables']['corpus']=corpus(evidence,'phase0','finqa',tokenizer,mcq)
                for raw_attempt in evidence.jsonl(BASE+'teacher/finqa/raw_outputs.jsonl'):
                    policy=raw_attempt['reload_timing'].get('batch_policy',{})
                    require(policy.get('vllm')=='0.18.0' and policy.get('gdn_prefill_backend')=='triton' and policy.get('attention_backend')=='FLASH_ATTN','teacher runtime backend/pin differs')
                generation={}
                from kit.v4_timing import sample
                ledger=evidence.json('k8b4/containment/allocation-ledger.json')
                for role in ('teacher','rewrite'):
                    root=BASE+role+'/finqa/'
                    journal=evidence.jsonl(root+'raw_outputs.jsonl');items=evidence.jsonl(root+'pool.jsonl')
                    serial=evidence.json(root+'serial-merge-timing.json')
                    require(serial['verified_attempts']==len(journal) and serial['raw_sha256']==t.sha(json.dumps(journal,sort_keys=True)) and serial['pool_sha256']==t.sha(json.dumps(items,sort_keys=True)),'serial merge journal/count binding differs')
                    sample(ledger,serial)
                    generation[role]={'attempts':len(journal),'serial_reverification_seconds':float(number(serial['wall_seconds'],'serial merge seconds'))}
                report['tables']['generation']=generation
                report['tables']['common']=common_data(evidence,'phase0',tokenizer,('finqa',))
                require(report['tables']['corpus']['intersection']>0,'no accepted demonstration/rewrite intersection for smoke')
                import yaml
                configs={a:yaml.safe_load(evidence.files[BASE+'configs/'+a+'.yaml']) for a in 'SFRD'}
                validate_configs(configs)
                for arm,config in configs.items():
                    command=evidence.json(BASE+'configs/'+arm+'-command.json')
                    require(command['returncode']==0 and command['resolved_sha256']==t.sha(evidence.files[BASE+'configs/'+arm+'.yaml']) and command['resolve_command'][-3:]==['--cfg','job','--resolve'],'resolved launcher evidence differs')
                report['departures_sha256']=departures_hash(evidence.files['v4/report-inputs/v4_departures.json'])
                trained={}
                for arm in 'SFRD':
                    ident=f'q-{arm}-finqa';summary=run_record(evidence,ident,'phase0')
                    for identity_path in ('runs/'+ident+'/env/export-identity-step2.json',BASE+f'eval/{ident}-finqa-2048/checkpoint-identity.json'):
                        require(evidence.json(identity_path)['model_file_hashes'].get('phase0-technical-only.json')==output_policy_sha256(),'missing or changed technical checkpoint label')
                    require(summary['technical_smoke'] is True and summary['scientific_phase_allowed'] is False,'phase0 training is technical only')
                    provenance=training_provenance(evidence,'phase0',job_for(ident,'phase0'),summary,tokenizer)
                    scores=scoring(evidence,'phase0',ident,'finqa',2048,mcq,limit=20)
                    require(scores['fingerprint']==agree['fingerprint'],'post-training scoring GPU/environment differs')
                    metrics=evidence.jsonl('runs/'+ident+'/metrics.jsonl')
                    key='train/time(s)' if arm in 'FR' else 'timing_s/step'
                    seconds=[float(number(m.get('data',m)[key],'seconds per step')) for m in metrics if m['step'] in (1,2)]
                    require(len(seconds)==2 and all(s>0 for s in seconds),'missing step speed')
                    exports=evidence.jsonl('runs/'+ident+'/env/export-timings.jsonl')
                    merged=evidence.json('runs/'+ident+'/env/merge-timing.json')
                    tokens=evidence.jsonl('runs/'+ident+'/env/training-tokens.jsonl')
                    require(exports and tokens and merged['ok'] is True,'missing export/token/merge telemetry')
                    token_totals={}
                    for key in ('generated_trajectories','generated_tokens','supervised_tokens'):
                        require(all(type(row.get(key)) is int and row[key]>=0 for row in tokens),'invalid training token telemetry')
                        token_totals[key]=sum(row[key] for row in tokens)
                    export_seconds=sum(float(number(row['wall_seconds'],'export seconds')) for row in exports)
                    physical=evidence.scheduler_rows.get(('phase0','train-'+ident),{}).get('slurm',{}).get('gpu_uuids',[])
                    require(set(physical)=={r['uuid'] for r in summary['peak_memory_per_gpu'].values()},'training memory physical GPU assignment differs')
                    from kit.v4_timing import sample
                    ledger=evidence.json('k8b4/containment/allocation-ledger.json')
                    sample(ledger,merged)
                    for export in exports:sample(ledger,export)
                    reload=evidence.json(BASE+f'eval/{ident}-finqa-2048/timing.json')['reload'];sample(ledger,reload)
                    trained[arm]={'steps':2,'seconds_per_step':seconds,'tokens':token_totals,'export_seconds':export_seconds,'memory':summary['peak_memory_per_gpu'],
                                  'merge_seconds':float(number(merged['wall_seconds'],'merge seconds')),
                                  'reload':evidence.json(BASE+f'eval/{ident}-finqa-2048/timing.json')['reload'],
                                  'score':scores,'provenance':provenance}
                report['tables']['training']=trained
        if final:
            states,errors=verdicts(evidence,'phase0');require(not errors,'; '.join(errors))
            from kit.v4_auxiliary import chronology
            trace=chronology(evidence,'phase0');require(not trace['errors'],'; '.join(trace['errors']))
            expected=t.sha((KIT/'campaigns/v4-phase0.yaml').read_bytes())
            for row in trace['rows'].values():
                for attempt in row['attempts']:
                    start=attempt['start']
                    require(start.get('campaign_sha256')==expected,'runner phase0 source hash differs')
                    require(start.get('env',{}).get('V4_TELEMETRY')=='1','phase0 telemetry was disabled')
            report['tables']['runner']=states
        report['status']='technical pass'
    except (ValueError,KeyError,TypeError,OSError,EOFError) as exc:report['reasons']=[str(exc)]
    return report


def read_archive(path):
    from kit.v4_archive import Archive
    import tarfile
    try:
        with Archive(path) as evidence:return evidence_report(evidence)
    except (ValueError,OSError,EOFError,tarfile.TarError) as exc:
        return {'schema':'v4-phase0-reading.v1','status':'incomplete','reasons':[str(exc)],'scientific_phase_allowed':False,'owner_review_required':True,'usable_as_scientific_initialisations':False,'outputs_scope':'technical','tables':{}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('capture','prepare','verify-prepare','score','agreement','teacher','rewrite','training-set','schedule','configs','train','merge','report','pause'))
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--paste',type=Path);parser.add_argument('--qos');parser.add_argument('--partition');parser.add_argument('--account')
    parser.add_argument('--verified-environments',action='store_true');parser.add_argument('--phase',default='phase0',choices=('phase0',))
    for key in ('row','slot','task','checkpoint'):parser.add_argument('--'+key)
    parser.add_argument('--cap',type=int)
    args=parser.parse_args(argv)
    if args.operation=='capture':
        from kit.v4_slurm_capture import capture
        captured=capture(args.work,args.partition,args.qos,args.account)
        print(json.dumps(captured,indent=2))
        if not captured['ok']:print('WARNING: informational Slurm capture failure; retain and return the evidence.')
        return 0
    if args.operation=='prepare':print(prepare(args.work,args.paste,args.qos,args.verified_environments));return 0
    from kit.v4_timing import receipt,utc_now
    started=utc_now();clock=time.monotonic()
    try:operation(args.work,args.operation,slot=args.slot,checkpoint=args.checkpoint)
    except (HardStop,FileNotFoundError) as exc:
        write_durably(args.work/BASE/'hard-stop.json',{'cause':str(exc),'row':args.row})
        print('STOP: '+str(exc));return 2
    except (ValueError,RuntimeError,KeyError,subprocess.SubprocessError) as exc:
        print('WARNING: phase-0 observation failed: '+str(exc))
        write_durably(args.work/BASE/'observations'/(args.row+'.json'),{'informational':True,'cause':str(exc)})
    base=args.work/BASE; (base/'timings').mkdir(exist_ok=True)
    write_durably(base/'timings'/(args.row+'.json'),receipt(started,time.monotonic()-clock,row=args.row,phase='phase0',operation=args.operation))
    write_durably(args.work/'v4/report-status'/(args.row+'.json'),{'ok':1})
    return 0



def verification_plan(doc,receipt):
    """Re-derive the row allowance and its fit; retain legacy <=60-second plans."""
    measured=receipt.get('verification_seconds');allowance=verification_allowance(measured)
    require(doc.get('verification_seconds',measured)==measured,'planned verification measurement differs from prepare')
    require(doc.get('verification_allowance_seconds',scale(60))==allowance,'planned verification allowance differs from CPU measurement')
    if 'verification_allowance_seconds' in receipt:
        require(receipt['verification_allowance_seconds']==allowance,'prepared verification allowance differs')
    environment=receipt.get('environment_check',{}).get('deadline_seconds',scale(600))
    require(doc.get('environment_check_deadline_seconds',environment)==environment,'planned environment-check deadline differs')
    required=required_seconds(environment,allowance)
    seconds=sum(int(n)*unit for n,unit in zip(doc['sbatch_time'].split(':'),(3600,60,1)))
    require(0<seconds<=86400,'phase0 reservation exceeds 24 hours')
    require(doc.get('required_seconds',required)==required and doc.get('reservation_slack_seconds',seconds-required)==seconds-required,'planned verification fit arithmetic differs')
    return allowance


def verification_row_cap(work):
    """Only verify-prepare gets the measured cap from the hash-bound CPU plan."""
    path=os.environ.get('V4_ALLOCATION_PLAN')
    if not path:return scale(60) # Unchanged tagged/rehearsal route without a new plan.
    raw=Path(path).read_bytes();require(t.sha(raw)==os.environ.get('V4_ALLOCATION_PLAN_SHA256'),'phase0 plan hash differs')
    doc=json.loads(raw);require(doc.get('phase')=='phase0','phase0 verification plan differs')
    receipt_path=Path(work)/'v4/report-inputs/prepare-receipt-phase0.json'
    require(doc['phase0_prepare_sha256']==digest(receipt_path),'phase0 CPU evidence changed since planning')
    return verification_plan(doc,json.loads(receipt_path.read_text()))

if __name__=='__main__':raise SystemExit(main())
