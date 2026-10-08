"""Reviewed infrastructure retry admission, independent of model outcomes."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

REASONS={'preempted','node_or_gpu_fault','out_of_memory','disk_full','download_or_io','containment_kill'}


def failure_record(work,command):
    """Read the original copy, or a durable row bound to the exact attempt path.

    The wrapper can die after the durable write but before its final copy. Never
    guess a row from argv or choose a better outcome among multiple candidates.
    """
    work=Path(work).resolve();out=Path(command[command.index('--out')+1]).resolve()
    if not out.is_relative_to(work):raise ValueError('failure record outside WORK')
    copied=out/'containment-row.json'
    if copied.exists():return copied,json.loads(copied.read_text())
    candidates=[]
    for path in (work/'k8b4/containment/rows').glob('*.json'):
        doc=json.loads(path.read_text())
        if doc.get('output_path') and Path(doc['output_path']).resolve()==out:candidates.append((path,doc))
    if len(candidates)!=1:raise ValueError('missing or ambiguous durable attempt containment record')
    return candidates[0]


def automatic(work,campaign,row,attempt,previous):
    """Apply the reviewed infrastructure rule to bounded, archived evidence only.

    This generates the same admission receipt the reader checks. A nonzero exit
    or an observed score alone never qualifies. No inference is made from a
    launcher log until the owning Slurm step has independent termination proof.
    """
    if attempt>3:raise ValueError('more than two retries')
    from kit.p4_contain import scheduler_problems
    work=Path(work);command=previous['command']
    if (work/'k8b4/containment/v4-stop.json').exists():raise ValueError('hard stop requires acknowledgement and reconcile')
    record_path,record=failure_record(work,command)
    if '--' not in command or record.get('command')!=command[command.index('--')+1:]:
        raise ValueError('failure record is not bound to the launched command')
    acknowledged=False
    if record.get('stop_class')=='hard' or record.get('state')=='unverified':
        from kit.v4_ack import verify
        def launch_identity(doc):
            # Reconcile can update termination and charges, never admission,
            # argv, allocation/step/GPU identity, inputs or safety receipts.
            mutable={'state','ok','status','failure_type','stop_class','charged_until',
                     'charged_gpu_hours_upper_bound','ledger_charge_gpu_hours'}
            stable={k:v for k,v in doc.items() if k not in mutable and k!='slurm'}
            stable['slurm']={k:v for k,v in doc.get('slurm',{}).items()
                             if k not in ('termination','cancel','recovery_observation')}
            return stable
        for stop in (work/'k8b4/containment/acknowledged-stops').glob('*/stop.json'):
            try:
                doc=json.loads(stop.read_text());path=Path(doc['row_record']).resolve()
                if not path.is_relative_to(work.resolve()):continue
                current=json.loads(path.read_text())
                if launch_identity(current)!=launch_identity(record):continue
                kind=doc['failure_type'].lower()
                if not any(word in kind for word in ('time cap','not verified','unverified','unknown exposure','pending step unresolved')):continue
                verify(stop,stop.parent/'ack.json')
                record=current;record_path=path;acknowledged=True;break
            except (OSError,ValueError,KeyError,TypeError):continue
        if not acknowledged:raise ValueError('hard row failure lacks verified manager acknowledgement')
    problems=scheduler_problems(work,record)
    if acknowledged:
        # Acknowledgement permits new bounded work after cleanup; prior late
        # exposure stays charged. Every other replay check still applies.
        problems=[p for p in problems if p not in ('scheduler termination outside admitted start/kill_by window','termination verified past deadline')]
    if record.get('ok') or record.get('state')!='ended' or problems:
        raise ValueError('failure lacks verified termination/exposure')
    log=work/'campaign'/campaign/row/f'attempt-{previous["attempt"]}'/'output.log'
    text=log.read_text();termination=record['slurm']['termination']
    reason=None;excerpt=None
    for code,pattern in (
        ('out_of_memory',r'(?:CUDA out of memory|torch\.OutOfMemoryError)[^\n]*'),
        ('disk_full',r'(?:No space left on device|\[Errno 28\])[^\n]*'),
        ('download_or_io',r'(?:ConnectionError|ReadTimeout|HTTPError|Input/output error)[^\n]*')):
        match=re.search(pattern,text)
        if match:reason=code;excerpt=match.group();break
    if termination.get('failure_type')=='preempted' or termination.get('state')=='PREEMPTED':reason='preempted'
    elif termination.get('state')=='NODE_FAIL':reason='node_or_gpu_fault'
    elif record.get('failure_type') in ('time cap','termination outside time cap'):reason='containment_kill'
    if not reason:raise ValueError('unqualified row failure')
    if reason=='out_of_memory' and any(json.loads(p.read_text()).get('reason_code')=='out_of_memory' for p in (work/'v4/retries'/campaign/row).glob('attempt-*.json')):
        raise ValueError('only one OOM retry')
    if not excerpt:
        # The evidence is the independently replayed scheduler record, rather
        # than a fabricated excerpt appended to the launcher's output.
        log=record_path;text=log.read_text();excerpt=record.get('failure_type') or termination['state']
    if not log.resolve().is_relative_to(work.resolve()):raise ValueError('failure evidence outside WORK')
    receipt={'reason_code':reason,'previous_attempt':previous['attempt'],'eligible':True,
        'termination_verified':True,'log_path':str(log.relative_to(work)),
        'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'excerpt':excerpt,
        'state_policy':'no_valid_saved_state','review_policy':'registered outcome-blind infrastructure admission'}
    if row.startswith('train-'):
        ident=row.removeprefix('train-');arm=ident.split('-')[1] if ident.startswith('q-') else ident[0]
        dose=2 if ident.startswith('q-') else 40
        candidates=sorted((work/'runs'/ident).glob('*/global_step_*'),key=lambda p:int(p.name.removeprefix('global_step_')),reverse=True)
        for candidate in candidates:
            try:digest=saved_state(candidate,arm)
            except ValueError:continue
            step=int(candidate.name.removeprefix('global_step_'))
            if step==dose:receipt['state_policy']='recover_completed_operation';break
            if 0<step<dose:
                restoration_gate(work,arm)
                receipt.update(state_policy='resume_valid_state',saved_state_path=str(candidate.resolve()),
                    saved_state_sha256=digest,completed_updates=step,arm=arm);break
    target=work/'v4/retries'/campaign/row/f'attempt-{attempt}.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as handle:json.dump(receipt,handle,sort_keys=True,indent=2);handle.write('\n')
    return receipt


def tree_hash(path):
    path=Path(path)
    if not path.is_dir() or path.is_symlink():raise ValueError('saved state directory missing or linked')
    files={}
    for item in sorted(path.rglob('*')):
        if item.is_symlink():raise ValueError('saved state contains symlink')
        if item.is_file():files[str(item.relative_to(path))]=hashlib.sha256(item.read_bytes()).hexdigest()
    if not files:raise ValueError('empty saved state')
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()


def saved_state(path,arm,width=8):
    """Only complete model/optimizer/RNG/scheduler/loader/EMA shards are resumable."""
    path=Path(path);actor=path/'actor' if arm in 'SD' else path
    expected=['data.pt']
    for rank in range(width):
        expected += [str((actor/pathname).relative_to(path)) for pathname in
            (f'model_world_size_{width}_rank_{rank}.pt',f'optim_world_size_{width}_rank_{rank}.pt',f'extra_state_world_size_{width}_rank_{rank}.pt')]
        if arm in 'SD':expected.append(f'actor/v4-ema/model_world_size_{width}_rank_{rank}.pt')
    if any(not (path/name).is_file() or (path/name).stat().st_size==0 for name in expected):raise ValueError('saved checkpoint lacks optimizer/RNG/dataloader/EMA state')
    return tree_hash(path)


def admit(work,campaign,row,attempt,current,previous):
    """Require a review bound to the preceding append-only attempt and its log."""
    if attempt>3:raise ValueError('more than two retries')
    work=Path(work);receipt_path=work/'v4/retries'/campaign/row/f'attempt-{attempt}.json'
    receipt=json.loads(receipt_path.read_text())
    if receipt['reason_code'] not in REASONS:raise ValueError('non-infrastructure retry reason')
    if receipt.get('eligible') is not True or receipt.get('termination_verified') is not True:raise ValueError('reviewed eligibility/termination missing')
    if receipt['previous_attempt']!=previous['attempt']:raise ValueError('retry not bound to preceding attempt')
    log_path=(work/receipt['log_path']).resolve()
    if not log_path.is_relative_to(work.resolve()):raise ValueError('supporting log outside WORK')
    raw=log_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=receipt['log_sha256'] or not receipt['excerpt'] or receipt['excerpt'] not in raw.decode():raise ValueError('missing supporting infrastructure log excerpt')
    def command(doc):return [arg.replace('attempt-'+str(doc['attempt']),'attempt-N') for arg in doc['command']]
    if command(current)!=command(previous) or current['inputs']!=previous['inputs']:raise ValueError('retry changed command or dependency lineage')
    def environment(doc):return {k:v for k,v in doc['env'].items() if k not in ('OFFLOAD','V4_RESUME_PATH','V4_RUNNER_ATTEMPT')}
    if environment(current)!=environment(previous):raise ValueError('retry changed seed/recipe/environment')
    prior=list(receipt_path.parent.glob('attempt-*.json'))
    oom=[json.loads(p.read_text()) for p in prior if json.loads(p.read_text()).get('reason_code')=='out_of_memory']
    if receipt['reason_code']=='out_of_memory':
        if len(oom)!=1 or current['env'].get('OFFLOAD')!='1':raise ValueError('OOM requires exactly one OFFLOAD=1 retry')
    elif current['env'].get('OFFLOAD','0')!=previous['env'].get('OFFLOAD','0'):raise ValueError('placement changed without OOM review')
    policy=receipt.get('state_policy')
    if policy not in ('resume_valid_state','no_valid_saved_state','recover_completed_operation'):raise ValueError('missing saved-state recovery policy')
    if policy=='resume_valid_state':
        path=Path(receipt['saved_state_path']).resolve()
        if not path.is_relative_to(work.resolve()):raise ValueError('saved state outside WORK')
        arm=receipt['arm'];restoration_gate(work,arm);digest=saved_state(path,arm)
        if digest!=receipt['saved_state_sha256'] or current['env'].get('V4_RESUME_PATH')!=str(path):raise ValueError('saved state changed')
        step=int(path.name.removeprefix('global_step_'))
        if step!=receipt['completed_updates'] or not 0<step<40:raise ValueError('saved optimizer dose differs')
    return receipt


def restoration_gate(work,arm):
    """No saved-state restoration until real-stack qualification is reverified."""
    from kit.v4_archive import DirectoryEvidence
    from kit.v4_restore_check import verify_archive
    try:result=verify_archive(DirectoryEvidence(work))
    except (OSError,ValueError,KeyError,TypeError) as error:
        raise ValueError('restoration qualification required; STOP for review: '+str(error)) from error
    if arm not in result or result[arm].get('passed') is not True:
        raise ValueError('restoration qualification missing for '+arm)
    return result[arm]
