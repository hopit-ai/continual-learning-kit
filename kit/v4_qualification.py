"""Qualification admission from raw evidence, shared by execution and both readers."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

TASKS = ('chemistry', 'finqa')


def departures_hash(raw=None):
    raw = (Path(__file__).with_name('v4_departures.json').read_bytes() if raw is None else raw)
    if raw != Path(__file__).with_name('v4_departures.json').read_bytes():
        raise ValueError('frozen departures changed')
    return hashlib.sha256(raw).hexdigest()


def compare_pairs(runs,tasks=TASKS):
    """Two independent engine reload pairs/task, first 50 fixed held-out IDs, B.

    Engine startup/configuration is checked for ALL observations before any
    agreement arithmetic. A passing pair cannot hide a later differing pair.
    """
    if set(runs) != set(tasks) or any(len(runs[t]) != 4 for t in tasks):
        raise ValueError('two independent scoring pairs required for each task')
    for task, observations in runs.items():
        for observation in observations:
            if observation.get('engine_ok') is not True: raise ValueError('scoring_engine_start: '+task)
            if observation.get('configuration_ok') is not True: raise ValueError('scoring_engine_configuration: '+task)
    machines = {r['fingerprint'] for values in runs.values() for r in values}
    if len(machines) != 1: raise ValueError('scoring GPU or environment differs')
    tables = {}
    for task, observations in runs.items():
        pairs = []
        for pair in range(2):
            left, right = observations[2*pair:2*pair+2]
            if any(len(r['answers']) != 50 or len(r['verdicts']) != 50 for r in (left,right)):
                raise ValueError('agreement requires 50 questions at B: '+task)
            equal = left['answers'] == right['answers'] and left['verdicts'] == right['verdicts']
            if not equal: raise ValueError(f'scoring disagreement: {task} pair {pair+1}')
            pairs.append({'pair': pair+1, 'answers_agree':50, 'verdicts_agree':50})
        tables[task] = pairs
    return {'pairs_per_task':2,'questions_per_task':50,'cap':2048,
            'scope':'first 50 fixed held-out IDs; two disjoint independent reload pairs per task',
            'fingerprint':next(iter(machines)), 'tables':tables}


def validate_configs(configs):
    if set(configs) != set('SFRD'): raise ValueError('all four launcher configurations required')
    from kit.v4_contract import check_resolved
    import os
    old=os.environ.get('KIT_FINISH_GATE');os.environ['KIT_FINISH_GATE']='1'
    try:
        for arm, config in configs.items():
            reward=config.get('custom_reward_function',{}).get('path') if arm=='S' else None
            if arm=='S' and not str(reward).endswith('/kit/beds/v4_reward.py'):
                raise ValueError('S reward path differs')
            check_resolved(config,arm,profile_name='technical-smoke',reward_path=reward)
    finally:
        if old is None:os.environ.pop('KIT_FINISH_GATE',None)
        else:os.environ['KIT_FINISH_GATE']=old
    return True


def check_metrics(metrics, arm, steps, *, movement_required=True):
    """Every update has finite loss and the registered *used* learning rate."""
    loss_key='train/loss' if arm in 'FR' else 'actor/pg_loss'
    lr_key='train/lr' if arm in 'FR' else 'actor/lr'
    rows={}
    for metric in metrics:
        step=metric.get('step')
        if step not in range(1,steps+1):continue
        data=metric.get('data',metric)
        # The pinned SFT trainer also logs a validation-only record for the last step (val/loss, no train/loss).
        if loss_key not in data:continue
        if step in rows:raise ValueError('duplicate optimizer step')
        rows[step]=data
        for key in (loss_key,lr_key):
            value=data.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('missing/nonfinite '+key)
        expected=1e-5*min((step-1)/10,1)
        if not math.isclose(data[lr_key],expected,rel_tol=1e-6,abs_tol=1e-12):
            raise ValueError('per-step learning rate differs')
        if arm=='D':
            for key in ('sdft/teacher_student_l2','sdft/teacher_change_l2'):
                value=data.get(key)
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                    raise ValueError('missing/nonfinite EMA movement')
    if set(rows)!=set(range(1,steps+1)):raise ValueError('missing optimizer metrics')
    if arm=='D' and movement_required and steps>=2:
        if rows[2]['sdft/teacher_student_l2']<=0 or rows[2]['sdft/teacher_change_l2']<=0:
            raise ValueError('EMA did not move after second update')
    return {'first_update_learning_rate':0, 'steps':steps,
            'ema_movement':[{ 'step':step, **{key:row[key] for key in ('sdft/teacher_student_l2','sdft/teacher_change_l2')}}
                            for step,row in sorted(rows.items())] if arm=='D' else None}


def archive_agreement(archive, mcq,phase="qualification",tasks=TASKS):
    """Re-score all 8 archived reloads; never trust an agreement summary."""
    from kit.v4_readers import scoring
    observations={task:[] for task in tasks}
    # First check every startup/config record, including failed engines whose raw
    # outputs cannot exist. This must precede scoring arithmetic.
    for task in tasks:
        for run in range(1,5):
            root=f'v4/report-{phase}/eval/agreement{run}-{task}-2048/'
            engine=archive.json(root+'engine-status.json')
            if engine.get('engine_ok') is not True:raise ValueError('scoring_engine_start: '+task)
            if engine.get('configuration_ok') is not True:raise ValueError('scoring_engine_configuration: '+task)
    for task in tasks:
        for run in range(1,5):
            checkpoint='agreement'+str(run)
            scored=scoring(archive,phase,checkpoint,task,2048,mcq,limit=50)
            raw=archive.jsonl(f'v4/report-{phase}/eval/{checkpoint}-{task}-2048/responses.jsonl')
            observations[task].append({'engine_ok':True,'configuration_ok':True,
                'fingerprint':scored['fingerprint'],'answers':[r['response'] for r in raw],
                'verdicts':scored['verdicts']})
    return compare_pairs(observations,tasks)


def archive_static(archive):
    import yaml
    from kit.v4_prepare import verify_archived
    verify_archived(archive)
    configs={arm:yaml.safe_load(archive.files[f'v4/report-qualification/configs/{arm}.yaml']) for arm in 'SFRD'}
    validate_configs(configs)
    environment=archive.json('v4/report-qualification/environment.json')
    for arm,config in configs.items():
        command=archive.json(f'v4/report-qualification/configs/{arm}-command.json')
        if (command.get('returncode')!=0 or not all(arg in command['resolve_command'] for arg in ('--cfg','job','--resolve'))
            or command['resolved_sha256']!=hashlib.sha256(archive.files[f'v4/report-qualification/configs/{arm}.yaml']).hexdigest()
            or command['trainer_commit']!=environment['trainer_commit']):raise ValueError('qualification config command/hash differs')
    return {'departures_sha256':departures_hash(archive.files['v4/report-inputs/v4_departures.json']),
            'configs_checked':list('SFRD')}


def measured_caps(archive,budget):
    """Freeze outcome-blind row caps from the raw qualification projection."""
    from kit.v4_budget import row_caps
    return row_caps(budget['components'])


def archive_determinism(archive):
    proof=archive.json('v4/report-qualification/teacher-determinism.json')
    policy=proof['batch_policy']
    if policy.get('attention_backend')!='FLASH_ATTN':raise ValueError('teacher attention backend differs')
    invariant=policy.get('batch_invariant')
    if type(invariant) is not bool:raise ValueError('batch invariance policy missing')
    if invariant:
        if policy.get('engine_starts')!=1:raise ValueError('unexpected invariant engine restart')
    elif (policy.get('engine_starts')!=2 or not policy.get('startup_failure') or policy.get('batch_invariance')!='unavailable for this model'
          or policy.get('serial_equivalence_guaranteed') is not False or policy.get('regeneration_guaranteed') is not False):
        raise ValueError('unlabelled batch invariance fallback')
    if proof['same_prompt_indices']!=[[0,2],[1,1]] or len(proof['left'])!=2 or len(proof['right'])!=3:
        raise ValueError('determinism batch composition differs')
    equal=all(proof['left'][i]['text']==proof['right'][j]['text'] and proof['left'][i]['token_ids']==proof['right'][j]['token_ids'] for i,j in ((0,2),(1,1)))
    if invariant and not equal:raise ValueError('batch-invariant determinism failed')
    return {'passed':equal,'batch_policy':policy,'serial_equivalence_guaranteed':invariant and equal}


def failed_engines_first(archive):
    """Preserve a typed engine failure even in a partial, pre-training archive."""
    for name in sorted(archive.files):
        if name.startswith('v4/report-qualification/eval/') and name.endswith('/engine-status.json'):
            status=archive.json(name)
            if status.get('engine_ok') is not True:raise ValueError('scoring_engine_start: '+name)
            if status.get('configuration_ok') is not True:raise ValueError('scoring_engine_configuration: '+name)


def archive_ratio(archive):
    """Recompute empirical sample ratios; never turn them into an output guarantee."""
    from kit import v4_teacher as t
    proof=archive.json('v4/report-inputs/teacher-token-ratio.json')
    if proof['registered_teacher_cap']!=t.TEACHER_NEW_TOKENS or set(proof['tasks'])!=set(TASKS):raise ValueError('teacher token-ratio registration differs')
    for task,receipt in proof['tasks'].items():
        pool=archive.jsonl('v4/report-qualification/'+task+'-pool.jsonl')
        if receipt['pool_hash']!=t.sha(json.dumps(pool,sort_keys=True)) or receipt['universal_response_bound_proven'] is not False:
            raise ValueError('teacher token ratio is not a universal response bound')
        counts=receipt['counts']
        if [r['id'] for r in counts]!=[i['id'] for i in pool]:raise ValueError('teacher token-ratio sample differs')
        for item,count in zip(pool,counts):
            student=len(archive.student_tokenizer.encode('\n'.join(m['content'] for m in t.student_messages(item)),add_special_tokens=False))
            if count['student_tokens']!=student or type(count['teacher_tokens']) is not int or count['teacher_tokens']<=0:raise ValueError('teacher token-ratio counts differ')
        ratio=max(r['teacher_tokens']/r['student_tokens'] for r in counts)
        if receipt['ratio_bound_on_pool_text']!=ratio or receipt['smallest_cap_under_observed_ratio']!=math.ceil((t.RESPONSE_CAP-1)*ratio)+1:
            raise ValueError('teacher token-ratio arithmetic differs')
    return proof
