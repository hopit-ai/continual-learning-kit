"""Per-comparison evidence and frozen, incumbent-bound tier-2 interpretation."""
from kit.x_common import *
from kit.x_tier2 import COMPARISONS, ARMS2, identity


def tables(values,doc,support,inherited,repeats,same_machine):
    from kit.x_read import _tables
    results, details = {}, {}
    for name,spec in COMPARISONS.items():
        own_pre=[dict(r,arm='x-pre') for r in values if r['arm']=='x-t2-'+name+'-pre']
        common=own_pre if isinstance(inherited,dict) else own_pre or [r for r in values if r['arm']=='x-pre']
        challenger='p' if spec['primary']=='gain_auc' else 'drift'
        mapped=[]
        for row in values:
            for side,new in (('control','control'),('challenger',challenger)):
                if row['arm']=='x-t2-'+name+'-'+side:
                    mapped.append(dict(row,arm='x-'+new))
        try:
            data=_tables([*common,*mapped],dict(doc,_score_family='x-tier2.'+name),support,inherited,repeats,same_machine)
            comparison=data['comparisons'][challenger]
            signals=('drift','plasticity') if spec['signal']=='both' else (spec['signal'],)
            if len(doc['domains'])==1:
                comparison['status']='DEGENERATE'
            elif any(support[s]['status']!='READY' for s in signals):
                comparison['status']='PENDING' if any(support[s]['status']=='PENDING' for s in signals) else 'UNINFORMATIVE'
            elif (spec['challenger_p2'] or spec['control_p2']) and doc.get('p_variant')!='p2-l2-init':
                comparison['status']='DEFERRED'
            comparison.update(control='t2-'+name+'-control',challenger='t2-'+name+'-challenger',
                interval_scope='per comparison; three separately executed matched seed pairs')
            results[name]=comparison
            details[name]=dict(per_seed={k:v for k,v in data['per_seed'].items() if k in ('control',challenger)},
                transitions=data['transitions'])
        except (ValueError,KeyError,TypeError,IndexError) as error:
            results[name]=dict(status='INCONCLUSIVE',evidence_error=str(error))
    return dict(comparisons=results,details=details)


def reading(result,data,qualification):
    """Only bound outcomes reach admitted(); archive other identities/statuses."""
    from kit.x_read import admitted
    comparisons=data.get('tier2',{}).get('comparisons',{})
    archive={name:dict(status=comparisons.get(name,{}).get('status','PENDING'),
        control='t2-'+name+'-control',challenger='t2-'+name+'-challenger',interpretation='UNBOUND') for name in COMPARISONS}
    result['tier2_archived']=archive
    result['tier2_admitted']={}
    result['tier2_eligible_comparisons']={}
    def read_bound(name):
        c=comparisons.get(name,dict(status='PENDING'))
        archive.pop(name,None)
        result['tier2_eligible_comparisons'][name]=c
        value=admitted(c)
        result['tier2_admitted'][name]=value
        return value
    def decision(name,value):
        status=comparisons.get(name,{}).get('status','PENDING')
        return ('PENDING' if value is None else 'ADMITTED' if value else 'NOT_ADMITTED') if status=='COMPLETE' else status
    first=dict(result['admitted'])
    if data.get('variant')=='disabled' or data.get('tier2',{}).get('p2_feasibility')=='DEFERRED':
        # A recorded feasibility disposition resolves P2 non-admission. Missing
        # calibration/evidence without that disposition remains pending.
        first['p']=False
        result['tier2_p2_feasibility']='DEFERRED'
        for name,spec in COMPARISONS.items():
            if spec['control_p2'] or spec['challenger_p2']:
                archive[name]['status']='SKIPPED_P2_FEASIBILITY'
    if any(first.get(k) is None for k in ('drift','plasticity','p')):
        result.update(incumbent=None,rung_b='PENDING_PREREQUISITE',incremental_plasticity='PENDING_PREREQUISITE')
        return result
    a='drift' if first['drift'] else 'plasticity' if first['plasticity'] else None
    p2=bool(first['p'])
    if a and p2:
        stack='stack-'+a
        stacked=read_bound(stack)
        result['stacking']=decision(stack,stacked)
        if stacked is None:
            result.update(incumbent=None,rung_b='PENDING_STACKING',incremental_plasticity='PENDING_STACKING')
            return result
        p2=stacked
    else:
        result['stacking']='NOT_TRIGGERED'
    incumbent=(a or 'control')+('+p' if p2 else '')
    result['incumbent']=incumbent
    result['bound_tier2_incumbent']=incumbent
    if not first['drift']:
        result.update(rung_b='NOT_OPEN',incremental_plasticity='NOT_OPEN')
        return result
    b='b-p2' if p2 else 'b'
    b_admitted=read_bound(b)
    result['rung_b']=decision(b,b_admitted)
    if b_admitted:
        result['incumbent']='adaptive-drift'+('+p' if p2 else '')
        result['incremental_plasticity']='DEFERRED_NEW_REGISTRATION'
    elif b_admitted is None:
        result['incremental_plasticity']='PENDING_B'
    elif qualification is None:
        result['incremental_plasticity']='PENDING_SCREEN_S'
    elif not qualification['qualifies']['plasticity']:
        result['incremental_plasticity']='NOT_QUALIFIED'
    else:
        inc='incremental-p2' if p2 else 'incremental'
        value=read_bound(inc)
        result['incremental_plasticity']=decision(inc,value)
        if value:
            result['incumbent']='incremental-plasticity'+('+p' if p2 else '')
    return result


def validate_allocation(evidence,doc,arm,seed,standin):
    """Re-derive the registered signals, shares, integer counts and schedules."""
    import json
    from collections import Counter
    from kit import x_slots,x_tier2_slots
    def load(name):return json.loads(evidence[name])
    allocation=identity(arm)['allocation']
    root=f'runs/x-{arm}-s{seed}/'
    signals=f'tier2/signals/{arm}/s{seed}/'
    ds=doc['domains'];ref=incoming(doc,seed)
    origin=dict(technical_synthetic=True,model=ref['model']) if standin else ref['identity']
    measured={}
    steps=(10,20,30,40) if allocation=='adaptive' else (10,) if allocation=='drift' else ()
    actual={name.removeprefix(signals) for name in evidence if name.startswith(signals) and
        name.endswith('.json') and not name.endswith('.inputs.json')}
    require(actual=={f'k{k}.json' for k in steps},'fixed allocation must use k10 only; unexpected/missing signal refresh')
    baseline=None;targets=None;previous=None
    for step in steps:
        record=load(signals+f'k{step}.json');inputs=record['inputs']
        exported=load(root+f'export-{step}.json')
        expected=dict(technical_synthetic=True,model=exported['model']) if standin else exported
        require(inputs==load(signals+f'k{step}.inputs.json') and inputs['baseline']=='stage-start' and
            inputs['step']==step and inputs['ema_coefficient']==(.5 if allocation=='adaptive' else None) and
            inputs['checkpoints']==[origin,expected],'drift checkpoint/baseline inputs changed')
        require(set(record['measurements'])==set(ds) and set(inputs['targets'])==set(ds),'drift domain set changed')
        before={d:record['measurements'][d][0] for d in ds}
        if step==10:baseline=before;targets=inputs['targets']
        require(before==baseline and inputs['targets']==targets,'stage-start baseline was not reused from k10')
        raw={d:record['measurements'][d][1]['token_mean']-baseline[d]['token_mean'] for d in ds}
        expected_signal=x_tier2_slots.ema(previous,raw) if allocation=='adaptive' else raw
        require(record['raw_drift']==raw and record['allocation_signal']==expected_signal,'drift/positive EMA recomputation differs')
        measured[step]=expected_signal;previous=expected_signal
    if allocation in ('standalone','incremental'):
        suffix=f'-x-tier2.{identity(arm)["comparison"]}-s{seed}' if 'scoring-assignment.json' in evidence else ''
        probe={d:load(f'probes/b-s{seed}/{d}/receipt{suffix}.json') for d in ds}
        for d,value in probe.items():
            require(value['status']=='COMPLETE' and value['inputs']['incoming']==doc['identities'][f'b-s{seed}'] and
                value['drop']==value['before']['token_mean']-value['after']['token_mean'],'plasticity must use the bound incoming probe')
        measured[0 if allocation=='standalone' else 10]={d:v['drop'] for d,v in probe.items()}
    ends=(20,30,40) if allocation=='adaptive' else (40,)
    expected_files={f'slots-{k}.json' for k in ends}|(set() if allocation=='standalone' else {'slots-10.json'})
    require({name.removeprefix(root) for name in evidence if name.startswith(root+'slots-')}==expected_files,
        'allocation windows changed')
    if allocation!='standalone':
        require(load(root+'slots-10.json')==dict(start=0,end=10,allocation='incumbent'),'common ten-update prefix changed')
    for end in ends:
        start=end-10 if allocation=='adaptive' else 0 if allocation=='standalone' else 10
        weights=x_slots.shares(measured[start],doc['configuration'])
        slots=(x_tier2_slots.window(weights,doc['configuration'],seed,start,end) if allocation=='adaptive'
            else x_slots.schedule(weights,doc['configuration'],seed,start))
        audit=load(root+f'slots-{end}.json')
        require(audit['shares']==weights and audit['slots']==slots and audit['counts']==dict(Counter(r['domain'] for r in slots)) and
            audit['salt']==('addon-tier2-slots-v1' if allocation=='adaptive' else 'addon-slots-v1'),
            'allocation shares/counts/schedule differ from registration')
    require(load(root+'summary.json')['slot_audit']==load(root+'slots-40.json'),'final slot audit changed')


def validate_resources(evidence,root,receipt,receipt_sha,allocation):
    """Every completed native window must retain its own resource qualification."""
    import json
    from kit.x_p2_feasibility import check_rank,SETTINGS
    expected=dict(receipt['identity'],preflight_receipt_sha256=receipt_sha)
    windows=([(0,40)] if allocation=='standalone' else
        [(0,10),(10,20),(20,30),(30,40)] if allocation=='adaptive' else [(0,10),(10,40)])
    for rank in range(8):
        full=json.loads(evidence[f'p2/feasibility/s{receipt["identity"]["seed"]}/run/resource-r{rank}.json'])
        for start,end in windows:
            resource=json.loads(evidence[root+f'segments/resource-{start}-{end}-r{rank}.json'])
            require(resource['rank']==rank and resource['geometry']==full['geometry'],'tier-2 segment shard accounting differs')
            check_rank(resource,expected,read(SETTINGS),completed_updates=end)
        resource=json.loads(evidence[root+f'resource-r{rank}.json'])
        require(resource['rank']==rank and resource['geometry']==full['geometry'],'tier-2 shard accounting differs')
        check_rank(resource,expected,read(SETTINGS))


def validate(evidence,doc,standin,values,p2_validation):
    """Validate every comparison independently; never invalidate tier 1."""
    import hashlib
    import json
    errors={}
    setup=json.loads(evidence['m2-scoring-machine.json'])
    binding=hashlib.sha256(evidence['binding.json']).hexdigest()
    for name,spec in COMPARISONS.items():
        try:
            if spec['control_p2'] or spec['challenger_p2']:
                require(p2_validation['status']=='VALID','X4 calibration/P2 prerequisite unavailable')
                cal=json.loads(evidence['p2/calibration.json'])
                cal_sha=hashlib.sha256(evidence['p2/calibration.json']).hexdigest()
            for side in ('control','challenger'):
                arm='t2-'+name+'-'+side
                for seed in SEEDS:
                    require(f'tier2/{name}/terminal-{arm}-s{seed}.json' not in evidence,'tier-2 arm has a terminal resource failure')
                    root=f'runs/x-{arm}-s{seed}/'
                    if root+'complete.json' in evidence:
                        validate_allocation(evidence,doc,arm,seed,standin)
                        ident=json.loads(evidence[root+'identity.json'])
                        require(all(ident[k]==v for k,v in identity(arm).items()) and ident['arm']=='x-'+arm and
                            ident['seed']==seed and ident['binding_sha256']==binding,'tier-2 training identity changed')
                        ref=incoming(doc,seed)
                        expected_incoming=dict(technical_synthetic=True,model=ref['model']) if standin else ref['identity']
                        require(ident['incoming']==expected_incoming and ident['steps']==40 and ident['batch']==32 and
                            ident['fresh_optimizer'] and ident['fresh_ema'],'tier-2 incoming or dose changed')
                        if identity(arm)['p2']:
                            require(ident['p2_calibration_sha256']==cal_sha,'tier-2 recalibrated lambda')
                            from kit.x_p2_feasibility import validate_seed_return
                            receipt=validate_seed_return(evidence,doc,seed,standin)
                            receipt_sha=hashlib.sha256(evidence[f'p2/feasibility/s{seed}/receipt.json']).hexdigest()
                            require(ident['p2_preflight_receipt_sha256']==receipt_sha,'tier-2 node receipt binding changed')
                            if not standin:
                                validate_resources(evidence,root,receipt,receipt_sha,identity(arm)['allocation'])
                            for rank in range(1 if standin else 8):
                                origin=json.loads(evidence[root+f'first-segment/ema-origin/rank{rank}.json'])
                                require(origin==json.loads(evidence[root+f'ema-origin/rank{rank}.json']) and
                                    origin['status']=='PASS' and origin['variant']=='p2-l2-init' and origin['before_first_update'] and
                                    origin['student_checkpoint']==origin['teacher_checkpoint']==ref['model'],
                                    'P2 first-segment EMA origin missing or changed')
                                dose=json.loads(evidence[root+f'p2-rank{rank}.json'])
                                require(dose['completed_updates']==40 and len(dose['updates'])==40 and
                                    not dose['teacher_regularised'] and all(r['lambda_value']==cal['coefficient']['value'] for r in dose['updates']),
                                    'tier-2 P2 dose differs from X4')
                        if not standin and identity(arm)['allocation']!='standalone':
                            for step in ((10,20,30) if identity(arm)['allocation']=='adaptive' else (10,)):
                                prefix=json.loads(evidence[root+f'boundaries/k{step}/prefix.json'])
                                restored=json.loads(evidence[root+f'boundaries/k{step}/restore-qualified.json'])
                                require(restored['status']=='PASS' and restored['before_update']==step+1 and
                                    all(restored[k]==prefix[k] for k in ('checkpoint_sha256','ranks','dataloader')),
                                    'tier-2 live restore qualification missing or changed')
                    scored={(int(r['step']),'general-v1' if r['panel'] in ('ifeval','math','knowledge') else r['panel'])
                        for r in values if r['arm']=='x-'+arm and int(r['seed'])==seed}
                    for step,panel in scored:
                        key=f'scores/x-{arm}-s{seed}-k{step}-{panel}/m2-identity.json'
                        meta=json.loads(evidence[key])
                        require(('scoring-assignment.json' in evidence or meta['gpu_identity']==setup) and meta['binding_sha256']==binding and
                            (meta['arm'],meta['seed'],meta['step'],meta['panel'])==('x-'+arm,seed,step,panel), 'tier-2 score identity changed')
                        ref=incoming(doc,seed)
                        expected=(dict(technical_synthetic=True,model=ref['model']) if standin else ref['identity']) if step==0 else json.loads(evidence[root+f'export-{step}.json'])
                        require(meta['model']==expected if standin else meta['model']['model_file_hashes']==expected['model_file_hashes'], 'tier-2 score checkpoint changed')
        except (ValueError,KeyError,TypeError,IndexError) as error:
            errors[name]=str(error)
    return errors
