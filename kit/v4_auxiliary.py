"""Archive-only mechanism, frontier, resource and append-only retry readings."""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
import json
import math

from kit import v4_analysis as a, v4_teacher as t
from kit.v4_evidence import require, weights
from kit.v4_campaign import TASKS, build, slots
from kit.v4_budget import allocation_spend, number, stamp, gpu_hours

INFRASTRUCTURE={'preempted','node_or_gpu_fault','out_of_memory','disk_full','download_or_io','containment_kill'}


def chronology(archive,phase):
    """Keep every launched attempt; count the first valid result and reject score-selected replacement.

    A retry receipt is an independent, archived review of the preceding failed
    attempt, bound to its full supporting log. It is never inferred from scores.
    Refused dependency rows have no launch and are omissions, not training retries.
    """
    result={};errors=[]
    for row in build(phase)['rows']:
        ident=row['id'];prefix=f'campaign/v4-{phase}/{ident}/'
        starts=sorted((name for name in archive.files if name.startswith(prefix) and name.endswith('/start.json')),
                      key=lambda name:int(name.split('/')[-2].split('-')[1]))
        attempts=[];first_valid=None;oom=0;reference=None
        for name in starts:
            start=archive.json(name);attempt=start['attempt'];parent=name.removesuffix('start.json')
            require(start['row']==ident and start['campaign']=='v4-'+phase,'chronology identity differs')
            verdict=archive.json(parent+'verdict.json') if parent+'verdict.json' in archive.files else None
            entry={'attempt':attempt,'start':start,'verdict':verdict,'reason_code':None}
            if attempt!=len(attempts)+1:
                errors.append('missing or unordered earlier attempt: '+ident)
            if attempts:
                if first_valid is not None:errors.append('replacement after first valid result: '+ident)
                receipt_path=f'v4/retries/v4-{phase}/{ident}/attempt-{attempt}.json'
                try:
                    receipt=archive.json(receipt_path)
                    reason=receipt['reason_code'];entry['reason_code']=reason
                    require(reason in INFRASTRUCTURE,'non-infrastructure retry reason')
                    require(receipt['previous_attempt']==attempts[-1]['attempt'],'retry not bound to preceding attempt')
                    log=archive.files[receipt['log_path']]
                    require(t.sha(log)==receipt['log_sha256'] and receipt['excerpt'] and receipt['excerpt'] in log.decode(),'missing supporting infrastructure log excerpt')
                    require(receipt.get('eligible') is True and receipt.get('termination_verified') is True,'retry has no reviewed termination/eligibility evidence')
                    previous=attempts[-1]['start']
                    if reason=='out_of_memory':
                        oom+=1
                        require(oom==1 and start['env'].get('OFFLOAD')=='1','OOM requires one OFFLOAD=1 retry')
                    else:
                        require(start['env'].get('OFFLOAD','0')==previous['env'].get('OFFLOAD','0'),'placement changed without OOM review')
                    # Wrapper attempt paths differ; computation and read inputs do not.
                    def command(doc):return [arg.replace('attempt-'+str(doc['attempt']),'attempt-N') for arg in doc['command']]
                    require(command(start)==command(previous),'retry changed command/input')
                    require(start['inputs']==previous['inputs'],'retry changed dependency lineage')
                    require({k:v for k,v in start['env'].items() if k not in ('OFFLOAD','V4_RESUME_PATH','V4_RUNNER_ATTEMPT')}=={k:v for k,v in previous['env'].items() if k not in ('OFFLOAD','V4_RESUME_PATH','V4_RUNNER_ATTEMPT')},'retry changed environment/seed/recipe')
                    require(receipt.get('state_policy') in ('resume_valid_state','no_valid_saved_state','recover_completed_operation'),'retry state policy missing')
                    if receipt['state_policy']=='resume_valid_state':
                        require(receipt.get('saved_state_sha256') and receipt.get('completed_updates') is not None,'saved state provenance missing')
                except (ValueError,KeyError,TypeError) as exc:errors.append('disqualified retry '+ident+': '+str(exc))
            if reference is None:reference=start
            if verdict is not None:
                require(verdict['row']==ident and verdict['attempt']==attempt,'attempt verdict identity differs')
                if verdict.get('verdict')=='PASS' and first_valid is None:first_valid=attempt
            attempts.append(entry)
        if len(attempts)>3:errors.append('more than two retries: '+ident)
        result[ident]={'attempts':attempts,'first_valid_attempt':first_valid}
    return {'rows':result,'errors':errors}


def probe_point(archive,phase,name,task,items,mcq,expected_weights,*,conditioned=False,demonstrations=None):
    """Recompute eight strict finished-answer rewards per fixed training identity."""
    root=f'v4/report-{phase}/probes/{name}/'
    from kit.v4_readers import decoded_response,rendered_prompt_hash
    identity=archive.json(root+'probe-identity.json')
    if root+'reuse.json' in archive.files:
        receipt=archive.json(root+'reuse.json');source=receipt['source'].rstrip('/')+'/'
        require(source!=root and source.startswith('v4/report-') and '/../' not in source,'invalid probe reuse source')
        for name,digest in receipt['files'].items():
            require(name in ('raw_outputs.jsonl','timing.json','checkpoint-identity.json','probe-identity.json'),'foreign reused probe file')
            require(t.sha(archive.files[source+name])==digest and archive.files[source+name]==archive.files[root+name],'reused probe hash differs')
    require(identity['pool_sha256']==t.sha(json.dumps(items,sort_keys=True)) and identity['conditioned'] is conditioned,'probe membership/mode changed')
    require(identity['max_new_tokens']==2048 and identity['max_num_seqs']==128,'probe decoding geometry differs')
    if hasattr(archive,'scheduler_rows'):
        row=('profile-conditioned-'+task if conditioned else 'profile-probe-'+task) if phase=='qualification' else ('conditioned-probe-'+task if conditioned else 'probe-'+name)
        record=archive.scheduler_rows.get((phase,row))
        require(record is not None and record['gpus']==1,'missing valid probe containment')
        fingerprint=identity['machine'];physical=record['slurm']['gpu_uuids']
        require(len(physical)==1 and physical==record['slurm']['expected_gpu_uuids'] and fingerprint['cuda_visible_devices'] and ',' not in fingerprint['cuda_visible_devices'] and any(physical[0] in line for line in fingerprint['gpus']),'probe physical scoring GPU differs')
        actual=json.dumps({k:fingerprint[k] for k in ('gpus','cuda_visible_devices','versions','deterministic')},sort_keys=True)
        require(fingerprint['id']==t.sha(actual)[:16] and fingerprint['deterministic'] is True,'probe fingerprint differs')
    require(weights(archive.json(root+'checkpoint-identity.json'))==expected_weights,'probe checkpoint lineage differs')
    demos=demonstrations or {}
    available=[i for i in items if not conditioned or i['id'] in demos]
    require(identity['unavailable_ids']==sorted(i['id'] for i in items if conditioned and i['id'] not in demos),'conditioned unavailable identities differ')
    rows=archive.jsonl(root+'raw_outputs.jsonl');lookup={i['id']:i for i in available}
    require(len(rows)==8*len(available) and {(r['id'],r['attempt']) for r in rows}=={(i['id'],k) for i in available for k in range(1,9)},'missing/duplicate eight-attempt probe records')
    by_id=defaultdict(list)
    for row in rows:
        item=lookup[row['id']];seed=int(t.sha('v4-mechanism|'+item['id']+'|'+str(row['attempt']))[:8],16)%(2**31)
        require(row['seed']==seed and row['temperature']==.7 and row['top_p']==.95 and row['conditioned'] is conditioned,'probe seed/decoder differs')
        ids=row['token_ids'];require(len(ids)==row['teacher_tokens'] and len(ids)<=2048,'missing/invalid probe token IDs')
        require(decoded_response(archive,ids)==row['text'],'probe tokens do not decode to response')
        messages=t.rewrite_messages(item,demos[item['id']]) if conditioned else t.student_messages(item)
        require(row['rendered_prompt_sha256']==rendered_prompt_hash(archive,messages),'probe prompt differs')
        finished=row['finish_reason'] in ('stop','eos')
        gold=str(t.gold_of(item))
        strict=int(mcq.compute_score(row['text'],gold)['acc']) if task=='chemistry' else int(t.finqa.compute_score('finqa',row['text'],gold)['acc'])
        by_id[item['id']].append({'attempt':row['attempt'],'reward':int(finished)*strict,'tokens':len(ids),'finished':finished})
    return dict(by_id)


def probe_summary(records,ids):
    """Include empty strata; variation is population variance of eight binary rewards."""
    ids=set(ids);known=ids&set(records);rows=[r for ident in known for r in records[ident]]
    successes=sum(r['reward'] for r in rows)
    means=[sum(r['reward'] for r in records[ident])/8 for ident in known]
    return {'questions':len(ids),'available_questions':len(known),'attempts':len(rows),'successes':successes,
            'success_rate':successes/len(rows) if rows else None,
            'zero_in_eight':sum(p==0 for p in means),'no_success_share':sum(p==0 for p in means)/len(means) if means else None,
            'mean_reward_variance':sum(p*(1-p) for p in means)/len(means) if means else None,
            'finished_answers':sum(r['finished'] for r in rows),'generated_tokens':sum(r['tokens'] for r in rows),
            'mean_length':sum(r['tokens'] for r in rows)/len(rows) if rows else None}


def mechanism(archive,coverage,mcq,valid_slots):
    """Freeze overlapping memberships from initial unaided success, then report all points/families."""
    initial=weights(archive.json('v4/report-inputs/model-identities.json')['initial'])
    result={'strata_memberships':{},'points':{},'initial_conditioned_comparison':{}};errors=[]
    for task in TASKS:
        pool=archive.expected_pools[task]
        frozen=sorted(pool,key=lambda i:t.sha('v4-fixed-probe|'+task+'|'+i['id']))[:20]
        require(archive.jsonl('v4/report-inputs/'+task+'-probe.jsonl')==frozen,'probe was not fixed before coverage filtering')
        zero=probe_point(archive,'qualification',f'q-S-chemistry-{task}-0',task,frozen,mcq,initial)
        ids={i['id'] for i in frozen}
        success={ident for ident,rows in zero.items() if any(r['reward'] for r in rows)}
        strata={'ALL':ids,'initially_unaided_successful':success,'zero_in_eight':ids-success,
                'teacher_uncovered':ids-set(coverage[task]['accepted_ids']['teacher']),
                'rewrite_uncovered':ids-set(coverage[task]['accepted_ids']['rewrite'])}
        families={family:{i['id'] for i in frozen if i['family']==family} for family in sorted({i['family'] for i in pool})}
        result['strata_memberships'][task]={k:sorted(v) for k,v in strata.items()}
        demos={r['id']:r for r in archive.jsonl(f'v4/report-main/teacher/{task}/demonstrations.jsonl')}
        conditioned=probe_point(archive,'main','conditioned-'+task,task,frozen,mcq,initial,conditioned=True,demonstrations=demos)
        result['initial_conditioned_comparison'][task]={'unaided':probe_summary(zero,ids),'conditioned':probe_summary(conditioned,ids),
            'same_teacher_covered_ids':{'unaided':probe_summary(zero,set(conditioned)),'conditioned':probe_summary(conditioned,set(conditioned))}}
        for job in slots():
            if job['id'] not in valid_slots:continue
            for step in (0,20,40):
                name=f"{job['id']}-{task}-{step}"
                incoming=weights(archive.json('runs/'+job['incoming']+'/env/export-identity-step40.json')) if job['incoming'] else initial
                expected=incoming if step==0 else weights(archive.json('runs/'+job['id']+f'/env/export-identity-step{step}.json'))
                try:
                    point=probe_point(archive,'main',name,task,frozen,mcq,expected)
                    result['points'][name]={'strata':{k:probe_summary(point,v) for k,v in strata.items()},
                                           'families':{k:probe_summary(point,v) for k,v in families.items()}}
                except (ValueError,KeyError,TypeError) as exc:errors.append(name+': '+str(exc))
    return result,errors


def frontier(archive,mcq):
    """Select exactly the registered primary Chemistry identities from saved frontier answers."""
    items=archive.expected_panels['chemistry']
    gold={'sciknoweval-test-'+str(i['extra_info']['index']):str(i['reward_model']['ground_truth']) for i in items}
    raw=archive.jsonl('v4/report-inputs/frontier-chemistry.jsonl')
    selected=[r for r in raw if r['id'] in gold]
    require(len(selected)==159 and {r['id'] for r in selected}==set(gold),'frontier primary identities missing/duplicate')
    strict={r['id']:int(mcq.compute_score(r['response'],gold[r['id']])['acc']) for r in selected}
    correct=sum(strict.values())
    return {'chemistry':{'n':159,'strict':correct,'verdicts':strict,'wilson':a.wilson(correct,159),
                         'label':'frontier reference on primary 159 identities'},
            'finqa':{'correct':948,'label':'historical reference'}}


def resource_costs(archive):
    """Components use raw elapsed allocation intervals; comparative shared charges do not alter the ledger.

    Corpus generation and verification are apportioned within the union of the
    concurrent task rows, in proportion to their measured component durations.
    All row admission/cleanup idle time is retained in generation/startup cost.
    """
    ledger=archive.json('k8b4/containment/allocation-ledger.json');actual,blocks=allocation_spend(ledger)
    components={arm:{k:Fraction(0) for k in ('generation','verification','training','export_merge_reload','evaluation','probes')} for arm in 'SFRD'}
    tokens={arm:{'generated_trajectories':0,'generated_tokens':0,'supervised_tokens':0} for arm in 'SFRD'}
    rows=getattr(archive,'scheduler_rows',{})
    def row_cost(ident,phase='main'):
        record=rows[(phase,ident)]
        return gpu_hours(record['allocation_width'],stamp(record['charged_until'])-stamp(record['started_at']))
    # The fresh baselines and frozen initial probe are reused scientific evidence;
    # two-step training smokes and measurement-only conditioned samples are excluded.
    for arm in 'SFRD':
        for task in (*TASKS,'chemistry_probe'):
            for cap in (2048,8192):components[arm]['evaluation']+=row_cost(f'baseline-{task}-{cap}','qualification')
        for task in TASKS:
            components[arm]['probes']+=row_cost('profile-probe-'+task,'qualification')+row_cost('conditioned-probe-'+task)
    for job in slots():
        arm=job['arm'];ident=job['id'];summary=archive.json('runs/'+ident+'/run-summary.json')
        train=row_cost('train-'+ident)
        merge=archive.json('runs/'+ident+'/env/merge-timing.json')
        width=ledger['allocations'][str(merge['allocation_id'])]['width']
        exports=archive.jsonl('runs/'+ident+'/env/export-timings.jsonl')
        export_cost=sum(gpu_hours(ledger['allocations'][str(e['allocation_id'])]['width'],e['wall_seconds']) for e in exports)
        merge_cost=gpu_hours(width,merge['wall_seconds'])
        require(export_cost+merge_cost<=train,'training/export cost accounting overlap')
        components[arm]['training']+=train-export_cost-merge_cost
        validation=archive.json('v4/report-main/timings/merge-'+ident+'.json')
        validation_cost=gpu_hours(ledger['allocations'][str(validation['allocation_id'])]['width'],validation['wall_seconds'])
        components[arm]['export_merge_reload']+=export_cost+merge_cost+validation_cost
        for task in (*TASKS,'chemistry_probe'):
            for cap in (2048,8192):
                components[arm]['evaluation']+=row_cost(f'score-{ident}-{task}-{cap}')
                raw=archive.jsonl(f'v4/report-main/eval/{ident}-{task}-{cap}/responses.jsonl')
                tokens[arm]['generated_trajectories']+=len(raw)
                tokens[arm]['generated_tokens']+=sum(r['output_tokens'] for r in raw)
        for task in TASKS:
            for step in (0,20,40):
                components[arm]['probes']+=row_cost(f'probe-{ident}-{task}-{step}')
                raw=archive.jsonl(f'v4/report-main/probes/{ident}-{task}-{step}/raw_outputs.jsonl')
                if f'v4/report-main/probes/{ident}-{task}-{step}/reuse.json' not in archive.files:
                    tokens[arm]['generated_trajectories']+=len(raw)
                    tokens[arm]['generated_tokens']+=sum(r['teacher_tokens'] for r in raw)
        # Tokens are measured trainer observations, never dose times a guessed length.
        metrics=archive.jsonl('runs/'+ident+'/env/training-tokens.jsonl')
        require(len(metrics)==40*8 and {(m['step'],m['rank']) for m in metrics}=={(step,rank) for step in range(1,41) for rank in range(8)},'missing/duplicate per-rank training token observations')
        for field in ('generated_trajectories','generated_tokens','supervised_tokens'):
            measurements=[m.get(field) for m in metrics]
            require(all(type(v) is int and v>=0 for v in measurements),'missing measured training '+field+': '+ident)
            tokens[arm][field]+=sum(measurements)
    shared={}
    for role in ('teacher','rewrite'):
        intervals=defaultdict(list);generation=Fraction(0);verification=Fraction(0);seen=set()
        for phase in ('qualification','main'):
            for task in TASKS:
                record=rows[(phase,role+'-'+task)]
                intervals[str(record['allocation_id'])].append((stamp(record['started_at']),stamp(record['charged_until'])))
                for entry in archive.jsonl(f'v4/report-{phase}/{role}/{task}/timings.jsonl'):
                    for component in ('generate','verify'):
                        timing=entry[component];identity=json.dumps(timing,sort_keys=True)
                        if identity in seen:continue
                        seen.add(identity)
                        width=ledger['allocations'][str(timing['allocation_id'])]['width']
                        cost=gpu_hours(width,timing['wall_seconds'])
                        if component=='generate':generation+=cost
                        else:verification+=cost
        total=Fraction(0)
        for allocation,spans in intervals.items():
            spans.sort();left,right=spans[0]
            for begin,end in spans[1:]:
                if begin<=right:right=max(right,end)
                else:total+=gpu_hours(ledger['allocations'][allocation]['width'],right-left);left,right=begin,end
            total+=gpu_hours(ledger['allocations'][allocation]['width'],right-left)
        fraction=verification/(generation+verification) if generation+verification else Fraction(0)
        shared[role]={'generation':total*(1-fraction),'verification':total*fraction}
        for arm in ('F','R','D') if role=='teacher' else ('R',):
            for component,value in shared[role].items():components[arm][component]+=value
            for task in TASKS:
                raw=archive.jsonl(f'v4/report-main/{role}/{task}/raw_outputs.jsonl')
                require(all(type(r.get('teacher_tokens')) is int and r['teacher_tokens']>=0 for r in raw),'missing corpus generated tokens')
                if f'v4/report-main/probes/{ident}-{task}-{step}/reuse.json' not in archive.files:
                    tokens[arm]['generated_trajectories']+=len(raw)
                    tokens[arm]['generated_tokens']+=sum(r['teacher_tokens'] for r in raw)
    totals={arm:sum(values.values()) for arm,values in components.items()}
    return {'actual_allocation_gpu_hours':float(actual),'actual_blocks':{k:float(v) for k,v in blocks.items()},
            'components_gpu_hours':{arm:{k:float(v) for k,v in values.items()} for arm,values in components.items()},
            'totals_gpu_hours':{arm:float(v) for arm,v in totals.items()},'tokens':tokens,
            'lower_total_cost_than_D':{arm:totals[arm]<totals['D'] for arm in ('F','R')},
            'shared_attribution':'27B generation/verification in full to F/R/D; rewrite generation/verification to R; ledger counts allocation intervals once'}
