"""Main-package-only, evidence-bound trainer-state retirement (never exports).

The sole slot runs serially. Checkpoint replacement has an old/new overlap;
restoration retains an arm's legs until its own recorded verdict is reverified.
Deletion publishes a durable intent before unlinking; replay checks all remaining
bytes before touching any file. No shutil.rmtree or arbitrary caller paths.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import functools
import math
from contextvars import ContextVar
from kit import v4_teacher as t
from kit.v4_arms import ARM_TABLE, checkpoint_path
from kit.runner import write_durably
from kit.v4_phase0_timing import shared_limit

POLICY={'schema':'v4-finqa-retention.v2','scope':'main package only','exports':'retain all',
        'slots':'latest durable state until final export and all registered evaluations',
        'restoration':'per-arm legs until verified recorded arm verdict; retire before next arm','parallel_slots':1}

# Unmeasured Lustre planning allowance, inside the owning row deadline. Saves,
# initial export hashing and CPU merge/reload belong to the trainer/merge row.
# With fingerprint caches, retirement reads evidence, not model/Adam/EMA shards:
# reserve 4 GiB of JSON/JSONL (four score panels + three probes, including their
# per-attempt copies, token IDs and receipts), twice for validation and hashing.
# 64 MiB/s deliberately allows shared-Lustre contention instead of assuming the
# >1 GiB/s needed to hash a 122 GiB state in the former 120-second deadline.
# Another 32 seconds reserves parsing, stat/fsync and the small unlink journal.
# These are planning allowances, not measurements; the outer row cap also bounds
# a blocked kernel call, while io_check stops between reads and before unlink.
RETIREMENT_EVIDENCE_READ_BYTES=8*1024**3
RETIREMENT_READ_BYTES_PER_SECOND=64*1024**2
RETIREMENT_METADATA_SECONDS=32
RETIREMENT_IO_SECONDS=math.ceil(RETIREMENT_EVIDENCE_READ_BYTES/RETIREMENT_READ_BYTES_PER_SECOND)+RETIREMENT_METADATA_SECONDS
_deadline=ContextVar('retirement_io_deadline',default=None)


def io_check():
    deadline=_deadline.get()
    if deadline is not None and time.monotonic()>=deadline:raise OSError('retirement IO cap reached; retain/replay intent')


def bounded_retirement(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        token=_deadline.set(min(_deadline.get() or float('inf'),time.monotonic()+shared_limit(RETIREMENT_IO_SECONDS)))
        try:
            return function(*args,**kwargs)
        finally:_deadline.reset(token)
    return wrapped


def fingerprint(path):
    stat=Path(path).stat()
    return [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):io_check();digest.update(block)
    return digest.hexdigest()


def state_names(arm):
    actor=ARM_TABLE[arm]['actor_subpath'];prefix=actor+'/' if actor else ''
    names=['data.pt']
    for rank in range(8):
        names += [prefix+f'{kind}_world_size_8_rank_{rank}.pt' for kind in ('model','optim','extra_state')]
        if actor:names.append(prefix+f'v4-ema/model_world_size_8_rank_{rank}.pt')
    return names


def entries(root,arm,*,partial=False,cached=None,fingerprints=None):
    root=Path(root)
    if not root.is_dir() or root.is_symlink():raise ValueError('retention state missing/linked')
    names=set(state_names(arm));files=[]
    for path in sorted(root.rglob('*')):
        io_check()
        if path.is_symlink():raise ValueError('retention symlink refused')
        if not path.is_file():continue
        name=path.relative_to(root).as_posix()
        if path.suffix=='.pt' and name not in names:raise ValueError('unreviewed trainer state file: '+name)
        before=fingerprint(path);previous=(cached or {}).get(name,{})
        digest=previous['sha256'] if previous.get('fingerprint')==before else sha_file(path)
        if fingerprint(path)!=before:raise ValueError('retention state changed during hashing')
        if fingerprints is not None:fingerprints[name]=before
        files.append({'path':name,'bytes':before[2],'sha256':digest,'delete':name in names})
    actual={row['path'] for row in files if row['bytes']>0}
    if not partial and not names<=actual:raise ValueError('retention requires complete model/Adam/EMA/extra/loader state')
    return files


def state_entries(run,arm,step,*,partial=False):
    """Reuse a hashed manifest only while inode/size/mtime/ctime still match.

    Persist outside the tensor tree; verify its tree digest against the durable
    receipt when present. Changed files are read again and compared by callers.
    """
    run=Path(run);root=checkpoint_path(run,arm,step)
    cache=run/'env'/f'state-files-step{step}.json';receipt=run/'env'/f'saved-state-step{step}.json'
    if any(p.is_symlink() for p in (run,run/'env',cache,receipt)):raise ValueError('linked state hash receipt')
    previous=json.loads(cache.read_text()) if cache.exists() else {}
    if previous:
        digest=t.sha(json.dumps({name:row['sha256'] for name,row in previous.items()},sort_keys=True))
        if receipt.exists() and json.loads(receipt.read_text())['state_sha256']!=digest:
            raise ValueError('cached state manifest differs from durable receipt')
    signatures={}
    files=entries(root,arm,partial=partial,cached=previous,fingerprints=signatures)
    if not partial:
        cache.parent.mkdir(parents=True,exist_ok=True)
        t.atomic_json(cache,{row['path']:{'sha256':row['sha256'],'fingerprint':signatures[row['path']]} for row in files})
    return files


def durable_state(run,arm,step):
    root=checkpoint_path(run,arm,step);files=state_entries(run,arm,step)
    for row in files:
        with (root/row['path']).open('rb') as handle:os.fsync(handle.fileno())
    for directory in [p for p in root.rglob('*') if p.is_dir()]+[root,root.parent]:
        fd=os.open(directory,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    # Same tree digest as the reviewed retry reader; streaming avoids shard-sized RAM allocations.
    digest=t.sha(json.dumps({row['path']:row['sha256'] for row in files},sort_keys=True))
    receipt=Path(run)/'env'/f'saved-state-step{step}.json';receipt.parent.mkdir(parents=True,exist_ok=True)
    doc={'step':step,'arm':arm,'path':str(root),'state_sha256':digest,
         'contents':['model','optimizer','extra','dataloader']+(['EMA'] if ARM_TABLE[arm]['actor_subpath'] else [])}
    if receipt.exists() and json.loads(receipt.read_text())!=doc:raise ValueError('saved state receipt changed')
    if not receipt.exists():t.atomic_json(receipt,doc)
    return receipt


def reload_weights(path):
    """CPU reload of every BF16 tensor, without importing a trainer or RNG changes."""
    from safetensors import safe_open
    path=Path(path);index=path/'model.safetensors.index.json'
    mapping=json.loads(index.read_text())['weight_map'] if index.exists() else None
    shards=set(mapping.values()) if mapping else {'model.safetensors'};seen={}
    for name in sorted(shards):
        file=path/name
        if Path(name).name!=name or file.is_symlink():raise ValueError('unsafe export shard')
        with safe_open(str(file),framework='pt',device='cpu') as handle:
            for key in handle.keys():
                if key in seen:raise ValueError('duplicate exported tensor')
                tensor=handle.get_tensor(key)
                if str(tensor.dtype)!='torch.bfloat16' or tensor.numel()==0:raise ValueError('export tensor is not nonempty BF16')
                seen[key]=name
                del tensor
    if not seen or mapping is not None and mapping!=seen:raise ValueError('export index/reload tensor identity differs')
    return {'tensor_count':len(seen),'dtype':'BF16'}


def export_fingerprints(target):
    target=Path(target)
    if target.is_symlink():raise ValueError('linked export refused')
    signatures={}
    for path in sorted(target.rglob('*')):
        io_check()
        if path.is_symlink():raise ValueError('linked export refused')
        if path.is_file():signatures[path.relative_to(target).as_posix()]=fingerprint(path)
    return signatures


def export_identity(run,step,*,fingerprints=None):
    """Reuse an identity only with unchanged membership and file fingerprints.

    The cache must agree with the existing durable identity receipt. Capture
    signatures before and after the initial full hash; never bless a mutation
    by restatting later. Config/tokenizer changes invalidate reuse too.
    """
    run=Path(run);target=run/f'hf-step{step}';env=run/'env'
    cache=env/f'export-files-step{step}.json';receipt=env/f'export-identity-step{step}.json'
    if any(p.is_symlink() for p in (run,env,cache,receipt)):raise ValueError('linked export hash receipt')
    before=export_fingerprints(target)
    previous=json.loads(cache.read_text()) if cache.exists() else {}
    recorded=json.loads(receipt.read_text()) if receipt.exists() else None
    if previous and previous.get('fingerprints')==before and previous.get('identity')==recorded:
        if export_fingerprints(target)!=before:raise ValueError('export changed during identity verification')
        if fingerprints is not None:fingerprints.update(before)
        return recorded
    identity=t.local_identity(target,'Qwen/Qwen3-8B')
    if export_fingerprints(target)!=before:raise ValueError('export changed during hashing')
    if recorded is not None and recorded!=identity:raise ValueError('export identity changed')
    env.mkdir(exist_ok=True)
    if recorded is None:t.atomic_json(receipt,identity)
    t.atomic_json(cache,{'identity':identity,'fingerprints':before})
    if fingerprints is not None:fingerprints.update(before)
    return identity


def verify_export(run,step):
    from kit.v4_sft import complete_export
    run=Path(run);target=run/f'hf-step{step}'
    if not complete_export(target):raise ValueError('retention requires complete merged export')
    signatures={};identity=export_identity(run,step,fingerprints=signatures);env=run/'env'
    identity_path=env/f'export-identity-step{step}.json';reload_path=env/f'export-reload-step{step}.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('export identity changed')
    if not identity_path.exists():t.atomic_json(identity_path,identity)
    if reload_path.exists():
        receipt=json.loads(reload_path.read_text())
        if receipt.get('identity')!=identity or receipt.get('tensor_count',0)<=0 or receipt.get('dtype')!='BF16':raise ValueError('export reload identity changed')
    else:
        reloaded=reload_weights(target)
        if export_fingerprints(target)!=signatures:raise ValueError('export changed during reload')
        t.atomic_json(reload_path,{'identity':identity,**reloaded})
    for file in target.iterdir():
        if file.is_file():
            with file.open('rb') as handle:os.fsync(handle.fileno())
    fd=os.open(target,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    if export_fingerprints(target)!=signatures:raise ValueError('export changed during verification')
    return [identity_path,reload_path]


def ensure_mid_export(run,arm):
    from kit.v4_sft import complete_export
    run=Path(run);target=run/'hf-step20'
    if not complete_export(target):
        source=checkpoint_path(run,arm,20,actor=True)
        # Isolated CPU subprocess: no trainer state, CUDA context or RNG is modified.
        from kit.v4_campaign_ops import run as execute
        execute([sys.executable,'-m','verl.model_merger','merge','--backend','fsdp',
                 '--local_dir',str(source),'--target_dir',str(target)],{'CUDA_VISIBLE_DEVICES':''})
    return verify_export(run,20)


def proofs(work,paths):
    work=Path(work).resolve();out={}
    for path in paths:
        path=Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(work):raise ValueError('retention proof outside WORK/linked')
        out[path.resolve().relative_to(work).as_posix()]=sha_file(path)
    return out


@bounded_retirement
def retire(work,run,arm,step,condition):
    work=Path(work).resolve();run=Path(run)
    if any(p.is_symlink() for p in (run,run/'env',run/'env/retention',run/'env/retention'/f'state-step{step}.json')):
        raise ValueError('linked retirement evidence refused')
    run=run.resolve();root=checkpoint_path(run,arm,step)
    if run.parent!=work/'runs' or not root.resolve().is_relative_to(run):raise ValueError('unsafe retirement root')
    journal=run/'env/retention'/f'state-step{step}.json';journal.parent.mkdir(parents=True,exist_ok=True)
    if journal.exists():
        doc=json.loads(journal.read_text())
        if doc['condition']!=condition or doc['root']!=str(root):raise ValueError('retirement condition changed')
        if t.sha(json.dumps(doc['files'],sort_keys=True))!=doc['manifest_sha256']:raise ValueError('retirement manifest changed')
        allowed=set(state_names(arm))
        for row in doc['files']:
            if row['delete'] and row['path'] not in allowed:raise ValueError('unsafe retirement unlink target')
        existing=state_entries(run,arm,step,partial=True)
        expected={r['path']:r for r in doc['files']}
        for row in existing:
            if expected.get(row['path'])!=row:raise ValueError('retirement state changed after intent')
        # Protected metadata must remain, even across interrupted deletion.
        if any(not r['delete'] and not (root/r['path']).is_file() for r in doc['files']):raise ValueError('protected checkpoint metadata missing')
    else:
        files=state_entries(run,arm,step)
        doc={'schema':'v4-state-retirement.v1','root':str(root),'arm':arm,'step':step,'status':'intent',
             'reason':condition['kind'],'condition':condition,'files':files,
             'manifest_sha256':t.sha(json.dumps(files,sort_keys=True))}
        write_durably(journal,doc) # precedes EVERY unlink, including interrupted replay
    for row in doc['files']:
        if row['delete']:
            io_check()
            if row['path'] not in state_names(arm):raise ValueError('unsafe retirement unlink target')
            file=root/row['path']
            if file.exists():file.unlink()
    for directory in [p for p in root.rglob('*') if p.is_dir()]+[root]:
        fd=os.open(directory,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    t.atomic_json(journal,{**doc,'status':'deleted'})


def before_save(run,arm,step):
    """Bound interrupted/partial plus replacement states BEFORE any new write."""
    run=Path(run);target=checkpoint_path(run,arm,step)
    others=[p for p in (run/ARM_TABLE[arm]['checkpoint_dir']).glob('global_step_*')
            if p!=target and any(p.rglob('*.pt'))]
    if len(others)>1:
        from kit.v4_finqa_policy import require
        require(False,'retention save would exceed old/new state coexistence','retention capacity')
    from kit.v4_finqa_policy import enabled,disk
    if enabled():
        from kit.v4_finqa_policy import writer_bytes
        disk(run.parent.parent,writer_bytes(run.parent.parent,arm=arm),'checkpoint save')
    if target.is_symlink() or any(p.is_symlink() for p in target.rglob('*')):raise ValueError('linked save target refused')


def after_save(run,arm,step,final):
    """After all eight ranks save; hashing/merge/reload use the trainer row cap."""
    run=Path(run);work=run.parent.parent
    receipt=durable_state(run,arm,step)
    if final==40 and step==20:ensure_mid_export(run,arm)
    pending=[]
    for old in sorted((run/ARM_TABLE[arm]['checkpoint_dir']).glob('global_step_*')):
        prior=int(old.name.removeprefix('global_step_'))
        if prior>=step:continue
        journal=run/'env/retention'/f'state-step{prior}.json'
        if journal.exists():
            existing=json.loads(journal.read_text())
            if existing['status']=='deleted':continue
            for path,digest in existing['condition']['proofs'].items():
                if sha_file(work/path)!=digest:raise ValueError('interrupted retirement proof changed')
            pending.append((prior,existing['condition']));continue
        proof_paths=[receipt]
        if final==40 and prior==20:
            reload_path=run/'env/export-reload-step20.json'
            if not reload_path.exists():
                saved=run/'env/saved-state-step20.json'
                observed=t.sha(json.dumps({r['path']:r['sha256'] for r in state_entries(run,arm,20)},sort_keys=True))
                if not saved.exists() or json.loads(saved.read_text())['state_sha256']!=observed:
                    raise ValueError('step20 durable state missing/changed; cannot recover export')
                ensure_mid_export(run,arm)
            proof_paths+=verify_export(run,20)
        condition={'kind':'newer_durable_state','new_step':step,'proofs':proofs(work,proof_paths)}
        pending.append((prior,condition))
    # Validate all targets before the first deletion.
    for prior,condition in pending:
        if not (run/'env/retention'/f'state-step{prior}.json').exists():state_entries(run,arm,prior)
    for prior,condition in pending:retire(work,run,arm,prior,condition)


def passed(work,phase,row):
    directory=Path(work)/'campaign'/('v4-'+phase)/row
    attempts=sorted(directory.glob('attempt-*'),key=lambda p:int(p.name.split('-')[-1]))
    if not attempts:raise ValueError('retention requires registered evaluation verdict: '+row)
    latest=attempts[-1];path=latest/'verdict.json';start=latest/'start.json'
    doc=json.loads(path.read_text());started=json.loads(start.read_text())
    if doc.get('verdict')!='PASS' or doc.get('row')!=row or doc.get('attempt')!=started.get('attempt') or started.get('row')!=row or doc.get('inputs',{})!=started.get('inputs',{}):raise ValueError('retention requires latest PASS: '+row)
    # A superseded producer invalidates the dependent evaluation, as in runner.
    for other,binding in doc.get('inputs',{}).items():
        if binding is None:continue
        candidates=sorted((directory.parent/other).glob('attempt-*'),key=lambda p:int(p.name.split('-')[-1]))
        if not candidates or [int(candidates[-1].name.split('-')[-1]),json.loads((candidates[-1]/'verdict.json').read_text())['verdict']]!=binding:
            raise ValueError('retention verdict lineage changed')
    return [path,start]


def slot_condition(work,phase,ident):
    from kit.v4_campaign_ops import job_for
    from kit.v4_evidence import weights
    work=Path(work);run=work/'runs'/ident;job=job_for(ident,phase);final=40 if phase=='main' else 2
    summary_path=run/'run-summary.json';summary=json.loads(summary_path.read_text())
    if summary.get('merged')!=1 or summary.get('merge_returncode')!=0 or summary.get('completed_optimizer_updates')!=final or summary.get('dose_complete') is not True:
        raise ValueError('retention final export/dose not proven')
    proof_paths=[summary_path]+passed(work,phase,'train-'+ident)+passed(work,phase,'merge-'+ident)
    for step in ((20,40) if phase=='main' else (2,)):
        if not (run/'env'/f'export-reload-step{step}.json').exists():raise ValueError('export reload evidence missing')
        proof_paths+=verify_export(run,step)
    rows=[(f'score-{ident}-finqa-2048',f'eval/{ident}-finqa-2048',final)]
    if phase=='main':
        rows += [(f'score-{ident}-{task}-{cap}',f'eval/{ident}-{task}-{cap}',40) for task,cap in [('finqa',8192),('general-v1',2048),('general-v1',8192)]]
        rows += [(f'probe-{ident}-finqa-{step}',f'probes/{ident}-finqa-{step}',step) for step in (0,20,40)]
    for row,directory,step in rows:
        proof_paths+=passed(work,phase,row)
        output=work/('v4/report-'+phase)/directory
        expected=(json.loads((work/'v4/report-inputs/model-identities.json').read_text())['initial'] if step==0 else export_identity(run,step))
        actual=json.loads((output/'checkpoint-identity.json').read_text())
        if weights(actual)!=weights(expected):raise ValueError('evaluation reload differs from retained export')
        probe=row.startswith('probe-')
        raw=output/('raw_outputs.jsonl' if probe else 'responses.jsonl')
        task='finqa' if probe else directory.rsplit('-',1)[0].removeprefix('eval/'+ident+'-')
        panel=work/'v4/report-inputs'/(task+('-probe.jsonl' if probe else '-panel.jsonl'))
        count=len(t.read_jsonl(panel))*(8 if probe else 1)
        if phase=='qualification':count=20
        if count<=0 or len(t.read_jsonl(raw))!=count:raise ValueError('registered evaluation incomplete')
        required=['timing.json','probe-identity.json'] if probe else ['tokens.jsonl','forgetting.json' if task=='general-v1' else 'bed-score.json']
        for name in required:
            file=output/name
            if name.endswith('.jsonl'):
                if len(t.read_jsonl(file))!=count:raise ValueError('registered evaluation tokens incomplete')
            elif not isinstance(json.loads(file.read_text()),dict) or not json.loads(file.read_text()):raise ValueError('registered evaluation report missing')
        proof_paths.append(panel)
        proof_paths += [p for p in output.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')]
    return job['arm'],{'kind':'final_export_and_registered_evaluations','phase':phase,'slot':ident,'proofs':proofs(work,proof_paths)}


@bounded_retirement
def finish_slot(work,phase,ident):
    arm,condition=slot_condition(work,phase,ident);run=Path(work)/'runs'/ident
    for state in sorted((run/ARM_TABLE[arm]['checkpoint_dir']).glob('global_step_*')):
        step=int(state.name.removeprefix('global_step_'))
        journal=run/'env/retention'/f'state-step{step}.json'
        if journal.exists() and json.loads(journal.read_text())['reason']=='newer_durable_state':
            doc=json.loads(journal.read_text())
            if doc['status']=='intent':
                for path,digest in doc['condition']['proofs'].items():
                    if sha_file(Path(work)/path)!=digest:raise ValueError('interrupted retirement proof changed')
                retire(work,run,arm,step,doc['condition'])
            continue
        retire(work,run,arm,step,condition)


def finish_restoration_arm(work,arm):
    """Retire only a successful arm, after re-reading its durable bound verdict.

    The proof excludes later arms and the full verdict: interruption/replay must
    not rebind already recorded deletion conditions when those records arrive.
    Validate all targets before deleting any; failed/unverified arms keep states.
    """
    from kit.v4_archive import DirectoryEvidence
    from kit.v4_restore_check import verify_arm_archive
    work=Path(work);result=verify_arm_archive(DirectoryEvidence(work),arm)
    root=work/'v4/report-qualification/restoration'/arm
    proof_paths=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl')]
    proof_paths.append(root.parent.parent/'environment.json')
    condition={'kind':'recorded_verified_restoration_arm_verdict','arm':arm,'proofs':proofs(work,proof_paths)}
    pending=[]
    for leg in ('uninterrupted','killed','resumed'):
        run=work/'runs'/f'restoration-{arm}-{leg}'
        states=list((run/ARM_TABLE[arm]['checkpoint_dir']).glob('global_step_*'))
        if not states:raise ValueError('restoration arm leg states/evidence missing')
        for state in states:
            step=int(state.name.removeprefix('global_step_'));pending.append((run,step))
            if not (run/'env/retention'/f'state-step{step}.json').exists():state_entries(run,arm,step)
    for run,step in pending:retire(work,run,arm,step,condition)
    return result


def finish_restoration(work):
    from kit.v4_archive import DirectoryEvidence
    from kit.v4_restore_check import verify_archive
    work=Path(work);results=verify_archive(DirectoryEvidence(work))
    root=work/'v4/report-qualification/restoration'
    if json.loads((root/'qualification.json').read_text()).get('schema')=='v4-restoration-qualification.v2':
        for arm in results:finish_restoration_arm(work,arm)
        return results
    condition={'kind':'recorded_verified_restoration_verdict','proofs':proofs(work,[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl')])}
    pending=[]
    for arm in results:
        for leg in ('uninterrupted','killed','resumed'):
            run=work/'runs'/f'restoration-{arm}-{leg}'
            for state in (run/ARM_TABLE[arm]['checkpoint_dir']).glob('global_step_*'):
                step=int(state.name.removeprefix('global_step_'));pending.append((run,arm,step))
                if not (run/'env/retention'/f'state-step{step}.json').exists():state_entries(run,arm,step)
    for run,arm,step in pending:retire(work,run,arm,step,condition)
    return results


def sweep(work,*,starting_slot=None,running_row=None):
    """Inside each main-package row; retire completed slots before any new launch.

    Incomplete/failed slots keep their latest state. A new slot cannot accumulate
    another state beside one lacking its final evaluation/deletion conditions.
    """
    work=Path(work)
    from kit.v4_campaign import slots,ALL_ARMS
    jobs=[('qualification','q-'+arm+'-finqa') for arm in ALL_ARMS]+[('main',j['id']) for j in slots()]
    for phase,ident in jobs:
        run=work/'runs'/ident
        if not run.exists() or ident==starting_slot:continue
        if running_row and (running_row.endswith(ident) or any(running_row.startswith(op+'-'+ident+'-') for op in ('score','probe'))):
            continue # The current attempt has a start record but no verdict yet.
        states=list(run.glob('*/global_step_*/**/*.pt'))
        if not states:continue
        last=(f'probe-{ident}-finqa-40' if phase=='main' else f'score-{ident}-finqa-2048')
        try:
            passed(work,phase,last)
            finish_slot(work,phase,ident)
        except (ValueError,OSError,KeyError):
            if starting_slot is not None:
                from kit.v4_finqa_policy import enabled,warn
                if not enabled():raise ValueError('previous slot state cannot be retired; refuse concurrent retained slot: '+ident)
                warn('retention', 'previous slot state retained without deleting: '+ident,work)
            continue # exactly the evaluations still needed, no deletion attempted
    if starting_slot and any(p.suffix=='.pt' for p in (work/'v4').glob('report-*/failed-training/**/global_step_*/**/*')):
        from kit.v4_finqa_policy import require
        require(False,'failed-run trainer state retained; next slot may need more disk','retention')


def verify_evidence(archive):
    """Both readers retain auditable hashes/conditions, without requiring deleted weights."""
    marker='v4/report-inputs/retention-policy.json'
    if marker not in archive.files:
        if 'v4/report-inputs/phase0-binding.json' in archive.files:raise ValueError('main retention policy missing')
        return
    if archive.json(marker)!=POLICY:raise ValueError('retention policy differs')
    for name in archive.files:
        if '/env/retention/state-step' not in name or not name.endswith('.json'):continue
        doc=archive.json(name)
        if doc.get('status') not in ('intent','deleted') or doc.get('reason')!=doc['condition']['kind'] or t.sha(json.dumps(doc['files'],sort_keys=True))!=doc['manifest_sha256']:
            raise ValueError('retention evidence manifest differs')
        if doc['reason'] not in ('newer_durable_state','final_export_and_registered_evaluations','recorded_verified_restoration_verdict','recorded_verified_restoration_arm_verdict') or not doc['condition']['proofs']:raise ValueError('retirement condition missing/unknown')
        if doc['reason']=='recorded_verified_restoration_arm_verdict':
            from kit.v4_restore_check import verify_arm_archive
            if doc['condition'].get('arm')!=doc['arm']:raise ValueError('retirement arm differs')
            verify_arm_archive(archive,doc['arm'])
        arm=doc['arm'];allowed=set(state_names(arm))
        for row in doc['files']:
            if row['delete'] and row['path'] not in allowed or len(row['sha256'])!=64 or row['bytes']<0:raise ValueError('unsafe retention evidence file')
        for path,digest in doc['condition']['proofs'].items():
            if path not in archive.files or t.sha(archive.files[path])!=digest:raise ValueError('retention condition proof changed/missing')
