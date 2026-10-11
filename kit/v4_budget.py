#!/usr/bin/env python3
"""Plan v4 1a.7 admission, using allocation wall time, never summed row exposure."""
from __future__ import annotations
import argparse
from datetime import datetime
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
from collections import Counter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kit.v4_arms import ALL_ARMS

ALLOWANCE = Fraction(3, 2)
CONTINGENCY = Fraction(45)
CEILING = Fraction(1000)
QUALIFICATION_LIMIT = Fraction(100)
from kit.v4_phase0_timing import scale
DEFAULT_WINDOWS = {key:scale(minutes) for key,minutes in {'teacher':300, 'rewrite':300, 'scientific':360}.items()}
TASKS = ('finqa',)
ARMS = ALL_ARMS


def number(value, name, *, positive=False):
    """Reject unknown, boolean, nonfinite and negative accounting inputs."""
    if isinstance(value, bool) or value is None:
        raise ValueError('missing/invalid timing or accounting: '+name)
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise ValueError('missing/invalid timing or accounting: '+name) from None
    if result < 0 or positive and result <= 0:
        raise ValueError('missing/invalid timing or accounting: '+name)
    return result


def gpu_hours(gpus, wall_seconds):
    """GPUs in the allocation times wall seconds divided by 3,600."""
    width = number(gpus, 'allocation GPUs', positive=True)
    if width.denominator != 1:
        raise ValueError('allocation GPU width must be an integer')
    return width * number(wall_seconds, 'wall seconds') / 3600


def stamp(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return number(value, 'allocation timestamp')
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('allocation timestamp needs timezone')
    return number(parsed.timestamp(), 'allocation timestamp')


def allocation_spend(ledger):
    """Recompute closed or observed allocation intervals, including idle/pause time."""
    spent, blocks = Fraction(0), {}
    from kit.v4_phase0_carry import accounting_allocations
    allocations = accounting_allocations(ledger)
    if not isinstance(allocations, dict) or not allocations:
        raise ValueError('missing allocation ledger')
    intervals = []
    for job, entry in allocations.items():
        start = stamp(entry['start'])
        end = stamp(entry.get('end') or entry.get('observed_until'))
        if end < start:
            raise ValueError('reversed allocation interval: '+job)
        spent += gpu_hours(entry['width'], end-start)
        intervals.append((start,end))
        cursor = start
        for segment in entry['segments']:
            begin = stamp(segment['start'])
            finish = min(end, stamp(segment['end'])) if segment.get('end') else end
            if begin != cursor or finish < begin:
                raise ValueError('gap/overlap in allocation block segments')
            block = segment['block']
            blocks[block] = blocks.get(block, Fraction(0))+gpu_hours(entry['width'],finish-begin)
            cursor = finish
        if cursor != end:
            raise ValueError('unaccounted allocation interval')
    # Multiple allocations may overlap: each is billed independently.
    return spent, blocks


def work_items(probe_n=20):
    """Price the entire fixed 23-slot graph, never infer slots from success."""
    if type(probe_n) is not int or probe_n!=20:raise ValueError('invalid frozen probe size')
    from kit.v4_campaign import slots, build, PROBE_STEPS
    jobs=slots();rows=[]
    count=len(jobs)
    def add(key,units,block='main'):rows.append(dict(key=key,units=units,block=block))
    for arm in ARMS:
        n=sum(j['arm']==arm for j in jobs)
        add(f'train:{arm}:finqa',n*40)
        for op in ('startup','export','merge'):add(f'{op}:{arm}:finqa',n*(1 if op=='startup' else 2))
    for task,n in (('finqa',1147),('general-v1',300)):
        for cap in (2048,8192):
            add(f'eval:{task}:{cap}',count*n);add(f'reload:{task}:{cap}',count)
    add('probe:finqa:2048',count*len(PROBE_STEPS)*8*probe_n);add('probe_reload:finqa',count*len(PROBE_STEPS))
    add('conditioned_probe:finqa',8*probe_n);add('conditioned_reload:finqa',1)
    for role in ('teacher','rewrite'):
        block='qualification' if role=='rewrite' else 'main'
        add(f'corpus_reload:{role}:finqa',1,block)
        add(f'serial_merge:{role}:finqa',4*1441,block)
    add('cpu:schedule:finqa',count)
    graph=build('main');ids=[r['id'] for r in graph['rows']]
    for stage,spec in graph['v4']['allocation_stages'].items():
        block='qualification' if spec['block']=='qualification' else 'main'
        add('runner:dispatch:'+stage,ids.index(spec['last'])-ids.index(spec['first'])+1,block)
        add('runner:prepare:'+stage,1,block)
    for op in ('coverage','training-set','gate','report','archive-pack'):add('cpu:'+op,1,'qualification' if op in ('coverage','training-set') else 'main')
    add('selftest',2);add('selftest:rewrite',1,'qualification')
    return rows


def admit(ledger, timings, completed=None, *, probe_n=20):
    """Componentized 1.5 projection plus actual spend and 45-hour contingency.

    timings is {key: [{wall_seconds, allocation_gpus, units}, ...]}; maximum
    *wall time per unit* is selected before converting with that observation's
    allocation width. completed names already reusable units, never smoke steps.
    Generation is four attempts on every remaining question regardless of acceptance.
    """
    completed = completed or {}
    try:
        spent, blocks = allocation_spend(ledger)
        rows = work_items(probe_n)
        for task in TASKS:
            for model in ('teacher', 'rewrite'):
                done = number(completed.get(f'{model}_questions:{task}',0), 'completed corpus questions')
                if done.denominator != 1 or done > 1441:
                    raise ValueError('invalid completed corpus question count')
                for operation in ('generate','verify'):
                    replay=number(completed.get(f'reuse_verify:{model}:{task}',0),'replayed verification count') if operation=='verify' else Fraction(0)
                    if replay.denominator!=1 or replay>4*done:raise ValueError('invalid replay verification count')
                    rows.append({'key':f'{operation}:{model}:{task}', 'units':4*(1441-int(done))+int(replay),
                                 'block':'qualification' if model=='rewrite' else 'main'})
        geometry=timings.get('batch_geometry')
        if geometry is not None:
            from kit.v4_teacher import batched_worst_case
            for row in rows:
                if row['key'].startswith(('generate:','verify:')):
                    operation,model,task=row['key'].split(':');spec=geometry[model+':'+task]
                    bound=batched_worst_case(spec['pool_size'],shard_count=spec['shards'],shard_sizes=spec['shard_sizes'])
                    row['units']=(0 if completed.get(f'{model}_questions:{task}',0)==1441 else bound['max_waves_per_shard']) if operation=='generate' else (row['units'] if completed.get(f'{model}_questions:{task}',0)==1441 else 4*bound['max_shard_size'])
                    row['batch_geometry']=bound
        components=[];exact_costs={}
        projections={'qualification':Fraction(0),'main':Fraction(0)}
        for row in rows:
            key=row['key']
            done = number(completed.get(key,0),'completed '+key)
            units = number(row['units'],'scheduled units')-done
            if units < 0 or done.denominator != 1:
                raise ValueError('completed work exceeds schedule: '+key)
            samples=timings.get(key)
            if not isinstance(samples,list) or not samples:
                raise ValueError('missing timing: '+key)
            measured=[]
            for sample in samples:
                wall = number(sample.get('wall_seconds'),'wall '+key)
                width = number(sample.get('allocation_gpus'),'allocation width '+key,positive=True)
                size = number(sample.get('units'),'measurement units '+key,positive=True)
                gpu_hours(width,wall)  # integer width check
                measured.append((wall/size,width))
            seconds,width = max(measured,key=lambda x:(x[0],x[1]))
            cost=ALLOWANCE*gpu_hours(width,seconds*units)
            exact_costs[key]=cost
            projections[row['block']]+=cost
            components.append({**row,'remaining_units':int(units),'slowest_seconds_per_unit':float(seconds),
                               'seconds_per_unit_exact':str(seconds),'allocation_gpus':int(width),'allowance':1.5,'projected_gpu_hours':float(cost)})
        overlap_groups=[];billable=dict(exact_costs)
        # Both task corpora now use the whole allocation and execute serially.
        # Shard overlap is represented by batched_worst_case, never discounted
        # a second time between tasks or by summing one-GPU exposure bounds.
        for component in components:component['billable_projected_gpu_hours']=float(billable[component['key']])
        windows=timings.get('allocation_windows',DEFAULT_WINDOWS)
        if set(windows)!=set(DEFAULT_WINDOWS) or any(type(v) is not int or v<15 for v in windows.values()):
            raise ValueError('invalid allocation window registration')
        counts,schedule,window_reasons=allocation_counts(components,windows,spent,blocks,
            caps=timings.get('allocation_caps',stage_caps(components) if 'allocation_windows' in timings else {}))
        for item in components:
            key=item['key'];count=None
            if key=='selftest':count=counts['teacher']+counts['scientific']
            elif key=='selftest:rewrite':count=counts['rewrite']
            elif key.startswith('runner:prepare:'):count=counts[key.split(':')[-1]]
            if count is not None:
                old=exact_costs[key]
                cost=ALLOWANCE*gpu_hours(item['allocation_gpus'],number(item['slowest_seconds_per_unit'],'setup seconds')*count)
                projections[item['block']]+=cost-old
                item.update(units=count,remaining_units=count,projected_gpu_hours=float(cost),billable_projected_gpu_hours=float(cost))
        total=spent+sum(projections.values())+CONTINGENCY
        qualification=blocks.get('qualification',Fraction(0))+projections['qualification']
        reasons=list(window_reasons)
        if qualification>QUALIFICATION_LIMIT and '100-hour qualification block exceeded' not in reasons:
            reasons.append('100-hour qualification block exceeded')
        if total>CEILING and '1000-hour campaign ceiling exceeded' not in reasons:
            reasons.append('1000-hour campaign ceiling exceeded')
        return {'status':'proceed','reasons':reasons,'warnings':reasons,
                'actual_gpu_hours':float(spent),'actual_blocks':{k:float(v) for k,v in blocks.items()},
                'remaining_qualification_gpu_hours':float(CEILING-spent),
                'planned_qualification_remaining_gpu_hours':float(QUALIFICATION_LIMIT-blocks.get('qualification',0)),
                'remaining_ceiling_gpu_hours':float(CEILING-spent),
                'allocation_windows':windows,'allocation_counts':counts,'allocation_schedule':schedule,
                'components':components,'overlap_groups':overlap_groups,'contingency_gpu_hours':45,
                'qualification_projected_gpu_hours':float(qualification),'total_projected_gpu_hours':float(total),
                'owner_proceed_required':False}
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        return {'status':'DEFERRED TO QUALIFICATION','reasons':[str(exc)],'owner_proceed_required':False}


def row_caps(components):
    """Measured complete command caps, including serial CPU/merge/reload work."""
    rows={r['key']:r for r in components}
    def seconds(key):
        r=rows[key]
        return ALLOWANCE*number(r.get('seconds_per_unit_exact',r['slowest_seconds_per_unit']),'rate')*r['units']
    caps={}
    for role in ('teacher','rewrite'):
        caps[role]=max(300,math.ceil(sum(seconds(f'{op}:{role}:finqa') for op in ('corpus_reload','generate','verify','serial_merge'))))
    from kit.v4_campaign import slots, PROBE_STEPS
    jobs=slots();counts=Counter(job['arm'] for job in jobs);count=len(jobs)
    caps['train']=max(300,math.ceil(max(sum(seconds(f'{op}:{arm}:finqa') for op in ('train','startup','export','merge'))/n for arm,n in counts.items())))
    caps['merge']=max(300,math.ceil(max(seconds(f'merge:{arm}:finqa')/n for arm,n in counts.items())))
    caps['score']=max(300,math.ceil(max((seconds(f'eval:{task}:{cap}')+seconds(f'reload:{task}:{cap}'))/count for task in ('finqa','general-v1') for cap in (2048,8192))))
    caps['probe']=max(300,math.ceil((seconds('probe:finqa:2048')+seconds('probe_reload:finqa'))/(count*len(PROBE_STEPS))))
    from kit.v4_retention import RETIREMENT_IO_SECONDS
    caps['cpu']=max(300+RETIREMENT_IO_SECONDS,math.ceil(max(seconds(k)/max(1,rows[k]['units']) for k in rows if k.startswith('cpu:'))))
    from kit.v4_phase0_timing import scale
    return {key:scale(value) for key,value in caps.items()}


def stage_caps(components):
    caps=row_caps(components)
    return {**caps,'teacher':max(caps['teacher'],caps['cpu']), 'rewrite':max(caps['rewrite'],caps['cpu']),
            'scientific':max(caps[key] for key in ('train','merge','score','probe','cpu'))}


def component_stage(key):
    if key.startswith(('runner:dispatch:','runner:prepare:')):return key.split(':')[-1]
    if key.startswith(('generate:teacher:','verify:teacher:','serial_merge:teacher:','corpus_reload:teacher:')) or key=='cpu:gate':return 'teacher'
    if key.startswith(('generate:rewrite:','verify:rewrite:','serial_merge:rewrite:','corpus_reload:rewrite:')) or key in ('cpu:coverage','cpu:training-set'):return 'rewrite'
    return 'scientific'


def allocation_counts(components,windows,spent,blocks,*,caps,passed=()):
    """Pack the registered serial rows; clip each new window to both budgets.

    A row starts only if its frozen cap fits. Completed commands consume no
    command cap; every resume still pays its full graph preparation and scan.
    """
    by_key={r['key']:r for r in components}
    remaining={key:number(row['remaining_units'],'remaining units') for key,row in by_key.items()}
    def take(key,units):
        count=min(remaining[key],number(units,'row units'));remaining[key]-=count
        row=by_key[key]
        return ALLOWANCE*number(row.get('seconds_per_unit_exact',row['slowest_seconds_per_unit']),'measured rate')*count
    from kit.v4_campaign import build, slots
    graph=build('main');ids=[r['id'] for r in graph['rows']]
    jobs={job['id']:job for job in slots()}
    rows_by_stage={stage:[] for stage in windows}
    for stage,spec in graph['v4']['allocation_stages'].items():
        for row in graph['rows'][ids.index(spec['first']):ids.index(spec['last'])+1]:
            ident=row['id'];operation=ident.split('-')[0];wall=Fraction(0);pending=False
            def charge(key,n):
                nonlocal wall,pending
                pending=pending or remaining[key]>0
                wall+=take(key,n)
            if ident in passed:pass
            elif operation in ('teacher','rewrite') and ident in [operation+'-'+t for t in TASKS]:
                task=ident[len(operation)+1:]
                for part in ('generate','verify','corpus_reload','serial_merge'):
                    key=f'{part}:{operation}:{task}';charge(key,remaining[key])
            elif operation in ('schedule','train','merge') and ident[len(operation)+1:] in jobs:
                job=jobs[ident[len(operation)+1:]];arm=job['arm'];task=job['task']
                if operation=='schedule':charge('cpu:schedule:'+task,1)
                elif operation=='merge':charge(f'merge:{arm}:{task}',1)
                else:
                    for part,n in (('train',40),('startup',1),('export',2),('merge',1)):charge(f'{part}:{arm}:{task}',n)
            elif operation=='score':
                cap=ident.rsplit('-',1)[1];panel=next(t for t in ('finqa','general-v1') if ident.endswith('-'+t+'-'+cap));n={'finqa':1147,'general-v1':300}[panel]
                charge(f'eval:{panel}:{cap}',n);charge(f'reload:{panel}:{cap}',1)
            elif operation=='probe':
                panel=ident.rsplit('-',2)[-2]
                charge('probe:'+panel+':2048',160);charge('probe_reload:'+panel,1)
            elif ident.startswith('conditioned-probe-'):
                panel=ident.removeprefix('conditioned-probe-')
                charge('conditioned_probe:'+panel,160);charge('conditioned_reload:'+panel,1);operation='probe'
            elif ident in ('coverage','training-set','owner-registration-gate','report','archive-pack'):
                charge('cpu:'+('gate' if ident=='owner-registration-gate' else ident),1)
            dispatch=take('runner:dispatch:'+stage,1)
            if pending or dispatch:
                cap=number(caps.get(operation,caps.get('cpu',caps.get(stage,0))),'row cap') if pending else Fraction(0)
                if ident in ('environment','prepare','owner-registration-gate') and (pending or dispatch):
                    cap=number(row['timeout_seconds'],'entry CPU deadline')
                # Dispatch is additional to the command cap, never hidden in it.
                rows_by_stage[stage].append((wall+dispatch,max(cap,wall)+dispatch))
    counts=dict.fromkeys(windows,0);schedule=[];reasons=[]
    remaining_total=CEILING-spent
    remaining_qualification=remaining_total
    registrations=graph['v4']['allocation_stages']
    for stage in ('rewrite','teacher','scientific'):
        todo=rows_by_stage[stage];cursor=0
        # Zero-rate synthetic fixtures still exercise a scheduled allocation.
        needed=any(r['remaining_units']>0 and component_stage(r['key'])==stage and not r['key'].startswith(('selftest','runner:prepare:')) for r in components)
        if not needed:continue
        selftest=ALLOWANCE*number(by_key['selftest:rewrite' if stage=='rewrite' else 'selftest']['slowest_seconds_per_unit'],'selftest seconds')
        startup=ALLOWANCE*number(by_key['runner:prepare:'+stage]['slowest_seconds_per_unit'],'runner startup seconds')
        setup=selftest+startup
        while cursor<len(todo) or counts[stage]==0:
            qualification=registrations[stage]['block']=='qualification'
            available=remaining_total
            window=number(windows[stage]*60,'site window')
            counts[stage]+=1
            schedule.append({'stage':stage,'window_seconds':float(window),'setup_seconds':float(setup),
                             'row_cap_headroom_seconds':float(max((cap for _,cap in todo[cursor:]),default=0))})
            used=setup;before=cursor
            while cursor<len(todo):
                wall,cap=todo[cursor]
                if window-used<wall:break
                used+=wall;cursor+=1
            cost=used*8/3600
            remaining_total-=cost
            if qualification:remaining_qualification-=cost
            if used>window or (cursor<len(todo) and cursor==before):
                reason='100-hour qualification block exceeded' if qualification and remaining_qualification<=remaining_total else '1000-hour campaign ceiling exceeded'
                # A site window too short to fit even the first pending row has
                # its own scope reason; budgets remain charged independently.
                if cursor<len(todo) and window==windows[stage]*60 and window-used<todo[cursor][0]:
                    reason='site window cannot pack a measured row; continuation required'
                reasons.append(reason);break
            if counts[stage]>10000:raise ValueError('unbounded allocation count')
    return counts,schedule,list(dict.fromkeys(reasons))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qualification_archive',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    from kit.v4_readers import read_qualification
    qualification=read_qualification(args.qualification_archive)
    report=qualification.get('budget') or {'status':'DEFERRED TO QUALIFICATION','reasons':qualification['reasons'],'owner_proceed_required':False}
    if qualification['status']=='stop: technical':report={**report,'status':'DEFERRED TO QUALIFICATION','reasons':qualification['reasons']}
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.out: args.out.write_text(text)
    else: print(text,end='')
    return 0 if report['status']=='proceed' else 2



def reproject(ledger,components,windows,completed=None,*,caps=None):
    """Re-price reader-verified frozen work at an allocation boundary.

    Completed units come from immutable, current PASS runner records. This path
    does not re-estimate rates, change dose or accept a supplied budget summary.
    """
    import copy
    completed=completed or {};rows=copy.deepcopy(components)
    spent,blocks=allocation_spend(ledger)
    for row in rows:
        key=row['key'];done=number(completed.get(key,0),'completed '+key)
        if done.denominator!=1 or done>row['units']:raise ValueError('completed work exceeds frozen schedule: '+key)
        remaining=row['units']-int(done)
        cost=ALLOWANCE*gpu_hours(row['allocation_gpus'],number(row.get('seconds_per_unit_exact',row['slowest_seconds_per_unit']),'frozen measured rate')*remaining)
        row.update(remaining_units=remaining,projected_gpu_hours=float(cost),billable_projected_gpu_hours=float(cost))
    counts,schedule,reasons=allocation_counts(rows,windows,spent,blocks,caps=caps or stage_caps(components),passed=set(completed.get('_passed_rows',())))
    for row in rows:
        key=row['key'];count=None
        if key=='selftest':count=counts['teacher']+counts['scientific']
        elif key=='selftest:rewrite':count=counts['rewrite']
        elif key.startswith('runner:prepare:'):count=counts[key.split(':')[-1]]
        if count is not None:
            cost=ALLOWANCE*gpu_hours(row['allocation_gpus'],number(row['slowest_seconds_per_unit'],'frozen setup')*count)
            row.update(remaining_units=count,projected_gpu_hours=float(cost),billable_projected_gpu_hours=float(cost))
    def cost(row):return ALLOWANCE*gpu_hours(row['allocation_gpus'],number(row.get('seconds_per_unit_exact',row['slowest_seconds_per_unit']),'measured rate')*row['remaining_units'])
    projected=sum((cost(r) for r in rows),Fraction(0))
    qualification=blocks.get('qualification',0)+sum((cost(r) for r in rows if r['block']=='qualification'),Fraction(0))
    total=spent+projected+CONTINGENCY
    if qualification>QUALIFICATION_LIMIT:reasons.append('100-hour qualification block exceeded')
    if total>CEILING:reasons.append('1000-hour campaign ceiling exceeded')
    return {'status':'proceed','reasons':list(dict.fromkeys(reasons)),'warnings':list(dict.fromkeys(reasons)),
        'actual_gpu_hours':float(spent),'actual_blocks':{k:float(v) for k,v in blocks.items()},
                'remaining_qualification_gpu_hours':float(CEILING-spent),
                'planned_qualification_remaining_gpu_hours':float(QUALIFICATION_LIMIT-blocks.get('qualification',0)),
                'remaining_ceiling_gpu_hours':float(CEILING-spent),
        'allocation_windows':windows,'allocation_counts':counts,'allocation_schedule':schedule,
        'components':rows,'qualification_projected_gpu_hours':float(qualification),
        'total_projected_gpu_hours':float(total),'contingency_gpu_hours':45,'owner_proceed_required':False}

if __name__=='__main__':raise SystemExit(main())
