#!/usr/bin/env python3
"""Node comparison tables and manager-only frozen, Screen-S-gated reading order."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *
import csv
import io
import gzip
import statistics
from collections import defaultdict
from kit.p3b_read import bound
from kit.c_read import accuracy, question_interval


def _tables(values, doc, support, inherited=(), repeats=10000, same_machine=True):
    if isinstance(inherited,dict):inherited=inherited.get(doc.get('_score_family','x-tier1'),[])
    groups = defaultdict(dict)
    from kit.m2_score import items_for
    expected = {p: {r['id'] for r in items_for(p, doc['technical_synthetic'])} for p in ('test','math_hard')}
    general = items_for('general-v1', doc['technical_synthetic'])
    expected.update({p: {r['id'] for r in general if r['panel'] == p} for p in ('ifeval','math','knowledge')})
    legal = {r['id']: r for r in items_for('test', doc['technical_synthetic'])}
    for row in values:
        key = (row['arm'].removeprefix('x-'), int(row['seed']), int(row['step']), row['panel'])
        require(key[0] in ('pre', *ARMS) and key[1] in SEEDS and key[2] in STEPS and int(row['cap']) == 2048, 'unregistered X score')
        require(row['question_id'] not in groups[key], 'duplicate X score')
        require(int(row['correct']) in (0,1), 'nonbinary X score')
        if key[-1] == 'test':
            item = legal.get(row['question_id'])
            require(item and row['gold'] == item['gold'] and row['group'] == item['group'] and row['correct'] == row['tolerant_correct'], 'ContractNLI parser/document binding changed')
        groups[key][row['question_id']] = row
    for (*_, panel), g in groups.items():
        if panel in expected: require(set(g) == expected[panel], 'partial/foreign X panel: '+panel)
        if panel == 'finqa': require(len(g) == 1147, 'full FinQA panel required')
        if panel == 'hellaswag': require(len(g) == 100, 'HellaSwag-100 required')
    old = defaultdict(dict)
    for row in inherited: old[row['arm'],int(row['seed']),int(row['step']),row['panel']][row['question_id']] = row
    def get(arm, seed, step, panel): return groups.get((arm,seed,step,panel))
    panel_names = ('ifeval','math','knowledge','finqa','math_hard','hellaswag')
    seed_tables, comparisons, transitions = {}, {}, []
    for arm in ARMS:
        records, missing = [], []
        for seed in SEEDS:
            required = [(arm,k,'test') for k in STEPS]+[(arm,k,p) for k in (0,40) for p in panel_names]+[('pre',0,p) for p in ('test',*panel_names)]
            absent = [dict(arm=a,seed=seed,step=k,panel=p) for a,k,p in required if not get(a,seed,k,p)]
            missing.extend(absent)
            if absent: continue
            pre = get('pre',seed,0,'test')
            points = [get(arm,seed,k,'test') for k in STEPS]
            require(all(set(g) == set(pre) for g in points), 'curve cohort changed')
            scores = [accuracy(g) for g in points]
            baseline = accuracy(pre)
            gain_auc = sum((x+y)/8 for x,y in zip(scores,scores[1:]))-baseline
            retentions = {p: accuracy(get(arm,seed,40,p))-accuracy(get('pre',seed,0,p)) for p in panel_names}
            final = {p: accuracy(get(arm,seed,40,p)) for p in panel_names}
            require(arm == 'p' and doc.get('p_variant')!='p2-l2-init' or scores[0] == baseline, 'non-P arm incoming score differs')
            # Secondary document-clustered bootstrap conditions on this seed.
            per_question = {q: sum((100*int(a[q]['correct'])+100*int(b[q]['correct']))/8 for a,b in zip(points,points[1:]))-100*int(pre[q]['correct']) for q in pre}
            record = dict(seed=seed, points=scores, incoming=baseline, final=scores[-1], gain_auc=gain_auc,
                recovery_auc=sum((x+y)/8 for x,y in zip(scores,scores[1:]))-scores[0], immediate_damage=scores[0]-baseline,
                retention=retentions, panel_final=final, protected_mean=statistics.mean(retentions[p] for p in doc['protected']),
                document_bootstrap=question_interval(per_question, pre, repeats, True))
            base_finqa = old.get(('base',seed,0,'finqa')) or (old.get(('base',101,0,'finqa')) if doc['technical_synthetic'] else None)
            start_finqa, end_finqa = get('pre',seed,0,'finqa'), get(arm,seed,40,'finqa')
            if base_finqa:
                require(set(base_finqa) == set(start_finqa), 'FinQA acquisition cohort changed')
                acquired = [q for q in start_finqa if not int(base_finqa[q]['correct']) and int(start_finqa[q]['correct'])]
                record['acquired_answer_survival'] = dict(n=len(acquired), survived=sum(int(end_finqa[q]['correct']) for q in acquired),
                    lost_ids=[q for q in acquired if not int(end_finqa[q]['correct'])])
            else: record['acquired_answer_survival'] = dict(status='PENDING_BASE')
            if not same_machine: record['acquired_answer_survival'] = dict(status='PENDING_CROSS_MACHINE')
            inherited_bars = {'finqa_retention': retentions['finqa']}
            for p in ('ifeval','math','knowledge'):
                base = old.get(('base',seed,0,p)) or (old.get(('base',101,0,p)) if doc['technical_synthetic'] else None)
                inherited_bars[p+'_cumulative'] = final[p]-accuracy(base) if base else None
            comparator = old.get(('standalone-legal-'+doc['configuration'],seed,40,'test'))
            inherited_bars['connected_vs_standalone'] = scores[-1]-accuracy(comparator) if comparator else None
            base_test=old.get(('base',seed,0,'test')) or (old.get(('base',101,0,'test')) if doc['technical_synthetic'] else None)
            record['learning_gap_context'] = (dict(status='DEFINED',
                finqa_start_minus_base_start_gain=(scores[-1]-baseline)-(accuracy(comparator)-accuracy(base_test)))
                if arm=='control' and comparator and base_test and same_machine else dict(status='PENDING_CROSS_MACHINE' if not same_machine else 'PENDING_NO_COMPATIBLE_COMPARATOR'))
            record['inherited'] = inherited_bars
            for p in ('test',*panel_names):
                before, after = get('pre',seed,0,p), get(arm,seed,40,p)
                require(set(before) == set(after), 'transition panel changed')
                transitions.append(dict(arm=arm,seed=seed,panel=p,
                    lost=sum(int(before[q]['correct']) and not int(after[q]['correct']) for q in before),
                    learned=sum(not int(before[q]['correct']) and int(after[q]['correct']) for q in before),
                    never_correct=sum(not int(before[q]['correct']) and not int(after[q]['correct']) for q in before)))
            records.append(record)
        seed_tables[arm] = dict(seeds=records, missing=missing)
    for challenger in ('drift','plasticity','p'):
        state = 'DEGENERATE' if len(doc['domains']) == 1 and challenger != 'p' else (
            support[challenger]['status'] if challenger in support else 'READY')
        candidate = {r['seed']: r for r in seed_tables[challenger]['seeds']}
        control = {r['seed']: r for r in seed_tables['control']['seeds']}
        endpoints = defaultdict(list)
        for seed in SEEDS:
            if seed not in candidate or seed not in control: continue
            a, b = candidate[seed], control[seed]
            for key in ('gain_auc','final','protected_mean'): endpoints[key].append(a[key]-b[key])
            for p in panel_names: endpoints[p].append(a['panel_final'][p]-b['panel_final'][p])
        inherited_bars = {}
        margins = dict(finqa_retention=-1, ifeval_cumulative=-3, math_cumulative=-3, knowledge_cumulative=-3, connected_vs_standalone=-2)
        for key, margin in margins.items():
            vals = [r['inherited'][key] for r in candidate.values() if r['inherited'][key] is not None]
            status = ('PENDING_NO_COMPATIBLE_COMPARATOR' if key == 'connected_vs_standalone' else
                      'PENDING_CROSS_MACHINE' if key.endswith('_cumulative') and not same_machine else
                      'COMPLETE' if len(vals) == 3 else 'PENDING')
            inherited_bars[key] = dict(interval=bound(vals,3), margin=margin, status=status,
                                       admission_gate=key != 'connected_vs_standalone')
        finals = [r['final'] for r in candidate.values()]
        gains = [r['final']-r['incoming'] for r in candidate.values()]
        inherited_bars['contract_nli_final'] = dict(interval=bound(finals,3), margin=69.7)
        inherited_bars['contract_nli_gain'] = dict(interval=bound(gains,3), margin=0, strict=True)
        comparisons[challenger] = dict(status=state if state != 'READY' else 'COMPLETE' if len(candidate)==len(control)==3 else 'PENDING',
            endpoints={k: bound(v,3) for k,v in endpoints.items()}, inherited=inherited_bars,
            positive_acquisition_seeds=sum(v>0 for v in gains),
            primary='gain_auc' if challenger == 'p' else 'protected_mean',
            seeds=sorted(set(candidate)&set(control)))
        from kit.m2_format import letter_valid,letter_cells
        inherited_values=[r for group in old.values() for r in group.values()]
        relevant=[r for r in values if r['arm'] in ('x-pre','x-control','x-'+challenger)]+[r for r in inherited_values if r['arm']=='base']
        if not letter_valid(relevant):comparisons[challenger].update(status='PENDING',reason='FORMAT-CONFOUNDED',format_cells=letter_cells(relevant))
        if challenger=='p':
            secondary=[]
            for seed in SEEDS:
                if seed not in candidate or seed not in control:continue
                left=[get('control',seed,k,'test') for k in STEPS]
                right=[get('p',seed,k,'test') for k in STEPS]
                delta={q:sum(100*(int(a[q]['correct'])+int(b[q]['correct'])-int(c[q]['correct'])-int(d[q]['correct']))/8
                    for a,b,c,d in zip(right,right[1:],left,left[1:])) for q in left[0]}
                secondary.append(dict(seed=seed,change=statistics.mean(delta.values()),
                    **question_interval(delta,left[0],repeats,True)))
            comparisons[challenger]['document_paired_secondary']=secondary
            comparisons[challenger]['learning_gap_context']=[dict(seed=r['seed'],**r['learning_gap_context']) for r in control.values()]
    return dict(comparisons=comparisons, per_seed=seed_tables, transitions=transitions,
        interval='Inherited package seed-t bounds: one-sided 95%, df=2; per comparison, not joint. Secondary bootstrap clustered by document.',
        variant=doc.get('p_variant','disabled'),technical_synthetic=doc['technical_synthetic'])


def tables(values,doc,support,inherited=(),repeats=10000,same_machine=True):
    values=list(values)
    first=[v for v in values if not v['arm'].startswith('x-t2-')]
    try:data=_tables(first,doc,support,inherited,repeats,same_machine)
    except (ValueError,KeyError,TypeError,IndexError) as error:
        # Revalidate all other arms independently; only P-specific defects are
        # isolated. Common/control/A defects still raise on this second call.
        data=_tables([v for v in first if v['arm']!='x-p'],doc,support,inherited,repeats,same_machine)
        data['comparisons']['p'].update(status='INCONCLUSIVE',evidence_error=str(error))
    if any(v['arm'].startswith('x-t2-') for v in values):
        from kit.x_tier2_read import tables as tier2_tables
        data['tier2']=tier2_tables(values,doc,support,inherited,repeats,same_machine)
    return data


def without_p(data):
    return dict(comparisons={k:v for k,v in data['comparisons'].items() if k!='p'},
        per_seed={k:v for k,v in data['per_seed'].items() if k!='p'},
        transitions=[v for v in data['transitions'] if v['arm']!='p'])


def admitted(comparison):
    if comparison['status'] != 'COMPLETE': return None if comparison['status'] in ('PENDING','INCONCLUSIVE','DEFERRED') else False
    primary = comparison['endpoints'][comparison['primary']]
    if primary['lower'] is None: return None
    guards = [v for k,v in comparison['endpoints'].items() if k != comparison['primary']]
    inherited = {k:v for k,v in comparison['inherited'].items() if v.get('admission_gate',True)}
    if any(v.get('status','COMPLETE') != 'COMPLETE' for v in inherited.values()): return None
    if any(v['interval']['lower'] is None for v in inherited.values()): return None
    return (primary['mean'] >= 2 and primary['lower'] > 0 and
        all(v['lower'] is not None and v['lower'] >= -1 for v in guards) and
        all(v['interval']['lower'] > v['margin'] if v.get('strict') else v['interval']['lower'] >= v['margin'] for v in inherited.values()) and
        comparison['positive_acquisition_seeds'] >= 3)


def _reading_tier1(data, qualification, standin):
    if qualification is None and data.get('variant')=='p2-l2-init':
        # Fixed-mix P2 is independent of S; known unrun A needs no signal gate.
        absent={key:data['comparisons'].get(key,{}).get('status') in ('DEGENERATE','UNINFORMATIVE','DEFERRED')
            for key in ('drift','plasticity')}
        no_a=all(absent.values());p=admitted(data['comparisons']['p'])
        return dict(order=['Screen S pending','A disposition','P2/stacking','remaining branches'],
            admitted=dict(drift=False if absent['drift'] else None,plasticity=False if absent['plasticity'] else None,p=p),
            status={**{key:data['comparisons'].get(key,{}).get('status','PENDING_SCREEN_S') for key in absent},'p':data['comparisons']['p']['status']},
            eligible_comparisons={'p':data['comparisons']['p']},
            stacking='PENDING_A' if not no_a else 'PENDING_P' if p is None else 'NOT_TRIGGERED',
            incumbent=('control'+('+p2-l2-init' if p else '')) if no_a and p is not None else None,
            rung_b='NOT_OPEN' if no_a else 'PENDING_A',incremental_plasticity='NOT_OPEN' if no_a else 'PENDING_A',
            qualification_sha256=None,variant='p2-l2-init',technical_synthetic=standin)
    require(qualification is not None, 'manager reading requires Screen S qualification file; A cannot be reported without it')
    require(qualification.get('technical_synthetic') is standin, 'Screen S scope mismatch')
    require(qualification.get('registration_sha256') == sha(RESOURCE), 'Screen S registration hash missing or changed')
    require(qualification.get('analysis_sha256') == sha(ROOT/'scripts/screen_s_analysis.py'), 'Screen S analysis hash missing or changed')
    require(isinstance(qualification.get('screen_return_sha256'),str) and len(qualification['screen_return_sha256']) == 64, 'Screen S return hash required')
    for signal, keys in (('drift',('drift_association','drift_forward')), ('plasticity',('plasticity',))):
        expected = all(qualification.get(k,{}).get('status') == 'PASS' for k in keys)
        require(qualification.get('qualifies',{}).get(signal) is expected, 'inconsistent Screen S qualification')
    results, reports = {}, {}
    drift = data['comparisons']['drift']
    if qualification['qualifies']['drift']:
        results['drift'] = admitted(drift); reports['drift'] = drift
    else: results['drift'] = False
    if results['drift'] is None:
        results['plasticity'] = None
    elif qualification['qualifies']['plasticity'] and not results['drift']:
        results['plasticity'] = admitted(data['comparisons']['plasticity'])
        reports['plasticity'] = data['comparisons']['plasticity']
    else: results['plasticity'] = False
    results['p'] = admitted(data['comparisons']['p']); reports['p'] = data['comparisons']['p']
    a = results['drift'] or results['plasticity']
    if results['drift'] is None or results['plasticity'] is None:
        stack = 'PENDING_A'; incumbent = None
    elif results['p'] is None:
        stack = 'PENDING_P'; incumbent = None
    elif a and results['p']:
        stack = 'MESSAGE3_A_PLUS_P_VS_FRESH_A'; incumbent = 'drift' if results['drift'] else 'plasticity'
    else:
        stack = 'NOT_TRIGGERED'; incumbent = ('drift' if results['drift'] else 'plasticity' if results['plasticity'] else 'control')+('+p' if results['p'] else '')
    return dict(order=['Screen S','eligible fixed drift','standalone plasticity only after resolved drift non-admission',
        'P2/stacking' if data.get('variant')=='p2-l2-init' else 'P/stacking','B against bound fixed-drift incumbent','incremental plasticity only after resolved B non-admission'],
        admitted=results, status={k:v['status'] for k,v in data['comparisons'].items()}, eligible_comparisons=reports, stacking=stack, incumbent=incumbent,
        rung_b='PENDING_FRESH_MATCHED_COMPARISON' if results['drift'] else 'NOT_OPEN',
        incremental_plasticity=('PENDING_B_AND_BOUND_INCUMBENT' if results['drift'] else
            'PENDING_DRIFT' if results['drift'] is None else 'NOT_OPEN') if qualification['qualifies']['plasticity'] else 'NOT_OPEN',
        qualification_sha256=digest(qualification),variant=data.get('variant','disabled'), technical_synthetic=standin)


def reading(data, qualification, standin):
    result=_reading_tier1(data,qualification,standin)
    if 'tier2' in data:
        from kit.x_tier2_read import reading as tier2_reading
        try:return tier2_reading(result,data,qualification)
        except (ValueError,KeyError,TypeError,AttributeError,IndexError) as error:
            result=_reading_tier1(data,qualification,standin)
            result.update(tier2_status='INCONCLUSIVE',tier2_error=str(error),
                tier2_eligible_comparisons={},bound_tier2_incumbent=None)
    return result


def _validate_common(evidence, doc, standin):
    from kit.m2_assignment import validate_files
    validate_files(evidence,require_assignment=not standin)
    require(doc['implementation'] == implementation(), 'returned add-on implementation changed')
    require(doc['technical_synthetic'] is standin, 'returned add-on scope mismatch')
    if 'registration_sha256' in doc:
        require(doc['registration_sha256'] == sha(RESOURCE), 'returned Screen S registration changed')
    setup = json.loads(evidence['m2-scoring-machine.json'])
    require(standin or doc.get('scoring_protocol')==m.protocol(),'UNVERIFIED: missing/incompatible scoring protocol')
    for name, raw in evidence.items():
        if not name.startswith('scores/') or not name.endswith('/m2-identity.json'): continue
        if name.startswith('scores/x-t2-'):continue
        meta = json.loads(raw)
        if doc.get('scoring_protocol'):require(meta['protocol']==doc['scoring_protocol'],'UNVERIFIED: changed scoring protocol')
        require(not meta['arm'].startswith('x-t2-'),'tier-1 identity cannot claim a tier-2 arm')
        require(('scoring-assignment.json' in evidence or meta['gpu_identity'] == setup) and meta['binding_sha256'] == hashlib.sha256(evidence['binding.json']).hexdigest(), 'returned score identity changed')
        arm, seed, step = meta['arm'].removeprefix('x-'), meta['seed'], meta['step']
        require(arm in ('pre',*ARMS) and seed in SEEDS and step in STEPS, 'unregistered returned score identity')
        if arm == 'pre' or step == 0 and (arm != 'p' or doc.get('p_variant')=='p2-l2-init'):
            ref = incoming(doc,seed)
            expected = dict(technical_synthetic=True,model=ref['model']) if standin else ref['identity']
        else:
            key = f'replacement/s{seed}/export.json' if step == 0 else f'runs/x-{arm}-s{seed}/export-{step}.json'
            require(key in evidence, 'returned score has no checkpoint receipt')
            expected = json.loads(evidence[key])
        require((meta['model'] == expected if standin else meta['model']['model_file_hashes'] == expected['model_file_hashes']), 'returned score model differs from bound checkpoint')
    support = json.loads(evidence['support.json'])
    for signal in ('drift','plasticity'):
        if support[signal]['status'] != 'READY': continue
        for seed in SEEDS:
            for domain in doc['domains']:
                key = f'own/b-s{seed}/{domain}/manifest.json'
                require(key in evidence, 'ready signal missing own-response receipt')
                own = json.loads(evidence[key])
                require(own['incoming'] == doc['identities'][f'b-s{seed}'] and own['monitor_n'] == 64 and
                        (signal == 'drift' or own['train_n'] == 128), 'ready signal lacks all-domain three-seed support')
    if not standin:
        for seed in SEEDS:
            prefix=f'runs/x-drift-s{seed}/'
            if prefix+'complete.json' not in evidence:continue
            expected=json.loads(evidence[prefix+'prefix.json'])
            qualified=json.loads(evidence[prefix+'restore-qualified.json'])
            require(qualified['status']=='PASS' and qualified['before_update']==11 and
                qualified['checkpoint_sha256']==expected['checkpoint_sha256'] and
                qualified['ranks']==expected['ranks'] and qualified['dataloader']==expected['dataloader'],
                'drift return lacks live restoration qualification')


def _validate_p2(evidence,doc,standin):
    from kit.x_p2_feasibility import validate_return
    validate_return(evidence,doc,standin)
    diagnostics=[]
    if doc.get('p_variant')=='p2-l2-init':
        from kit.p2_l2 import coefficient
        calibration=json.loads(evidence['p2/calibration.json'])
        require(calibration['schema']=='rung-p2-calibration.v1' and calibration['coefficient']==coefficient(calibration['seeds']),'P2 calibration unavailable/invalid')
    for seed in SEEDS:
        prefix=f'runs/x-p-s{seed}/'
        require(doc.get('p_variant')=='p2-l2-init','closed P variant cannot enable P2')
        from kit.p2_l2 import coefficient
        calibration=json.loads(evidence['p2/calibration.json'])
        require(calibration['schema']=='rung-p2-calibration.v1' and calibration['variant']=='p2-l2-init' and
            calibration['coefficient']==coefficient(calibration['seeds']) and calibration['anchor']==doc['p2_anchor'] and
            calibration['binding_sha256']==hashlib.sha256(evidence['binding.json']).hexdigest(),'returned P2 calibration differs')
        from kit.x_p import validate_smoke
        smoke_raw=evidence['p2/smoke-receipt.json']
        require(hashlib.sha256(smoke_raw).hexdigest()==calibration['smoke_sha256'],'returned P2 smoke binding changed')
        validate_smoke(json.loads(smoke_raw),standin)
        require(calibration['implementation']==doc['implementation'],'P2 calibration code differs')
        calibration_sha=hashlib.sha256(evidence['p2/calibration.json']).hexdigest()
        for name,expected in calibration['files'].items():
            if name in evidence:require(hashlib.sha256(evidence[name]).hexdigest()==expected,'returned P2 measured evidence changed')
        batch=json.loads(evidence[f'p2/s{seed}/batch.json'])
        require(batch['schema']=='rung-p2-first-batch.v1' and batch['variant']=='p2-l2-init' and
            batch['sha256']==calibration['files'][f'p2/s{seed}/actor-input.pt'],'P2 bound first-batch hash differs')
        for rank in range(1 if standin else 8):
            measured=json.loads(evidence[f'p2/s{seed}/measure-r{rank}.json'])
            require(measured['schema']=='rung-p2-measure.v1' and measured['variant']=='p2-l2-init' and measured['rank']==rank and
                measured['state_before']==measured['state_after'] and
                measured['task_grad_norm']==calibration['seeds'][str(seed)]['task_grad_norm'] and
                measured['incoming_distance']==calibration['seeds'][str(seed)]['incoming_distance'] and
                measured['optimizer_steps']==measured['scheduler_steps']==measured['ema_steps']==0 and
                measured['after_reduction_unscaling'] and measured['before_clipping'] and
                measured['rng_file_sha256']==calibration['files'][f'p2/s{seed}/rng-r{rank}.pt'],
                'P2 X4 no-step/gradient/RNG evidence differs')
        if prefix+'complete.json' not in evidence:continue
        identity=json.loads(evidence[prefix+'identity.json'])
        require(identity['variant']=='p2-l2-init' and identity['p2_calibration_sha256']==calibration_sha,
            'P2 calibration changed after stage identity was frozen')
        control_key=f'runs/x-control-s{seed}/identity.json'
        if control_key in evidence:
            require(json.loads(evidence[control_key])['p2_calibration_sha256']==calibration_sha,
                'P2/control first-batch calibration differs')
        ref=incoming(doc,seed)['model']
        for rank in range(1 if standin else 8):
            value=json.loads(evidence[prefix+f'ema-origin/rank{rank}.json'])
            require(value['status']=='PASS' and value.get('variant')=='p2-l2-init' and value['before_first_update'] and
                value['student_checkpoint']==value['teacher_checkpoint']==ref,'P2 EMA must originate at untouched incoming')
            training=json.loads(evidence[prefix+f'p2-rank{rank}.json'])
            require(training['schema']=='rung-p2-training.v1' and training['variant']=='p2-l2-init' and
                training['teacher_regularised'] is False and training['completed_updates']==40 and len(training['updates'])==40 and
                all(r['lambda_value']==calibration['coefficient']['value'] for r in training['updates']),'P2 training dose/identity differs')
            if not standin:
                from kit.p2_l2 import deviation
                first=training['updates'][0]
                diagnostics.append(dict(seed=seed,rank=rank,ratio=deviation(first['regulariser_to_task_ratio'],
                    calibration['coefficient']['realised_ratios'][str(seed)])))

    return diagnostics


def validate_compact(evidence,doc,standin):
    # Validate the common/A return independently. A bad P2 receipt, export or
    # score identity must not remove the shared control or any A comparison.
    common={name:raw for name,raw in evidence.items() if not name.startswith(('scores/x-p-','runs/x-p-'))}
    _validate_common(common,doc,standin)
    try:
        _validate_common(evidence,doc,standin)
        return dict(status='VALID',diagnostics=_validate_p2(evidence,doc,standin))
    except (ValueError,KeyError,TypeError,OSError,IndexError) as error:
        return dict(status='INCONCLUSIVE',reason=type(error).__name__+': '+str(error),diagnostics=[])


def isolate_p2(data,validation):
    if validation['status']=='INCONCLUSIVE':
        data['comparisons']['p']=dict(data['comparisons']['p'],status='INCONCLUSIVE',evidence_error=validation['reason'])
    data['p2_validation']=validation
    return data


def inherited_local(out):
    if not (out/'scoring-assignment.json').exists():return read(out/'inherited.json')['rows']
    from kit.x_references import gathered
    files={str(p.relative_to(out)):p.read_bytes() for p in (out/'references').rglob('*') if p.is_file() and p.suffix in ('.csv','.json')}
    files['scoring-assignment.json']=(out/'scoring-assignment.json').read_bytes()
    return gathered(files)


def inherited_return(evidence):
    if 'scoring-assignment.json' not in evidence:return json.loads(gzip.decompress(evidence['inherited.json.gz']))['rows']
    from kit.x_references import gathered
    return gathered(evidence)


def job(a):
    if a.action == 'tables':
        doc = context(a.out, a.standin)
        import shutil
        errors=a.out/'tier2/score-errors'
        if errors.exists():shutil.rmtree(errors)
        values = []
        p_errors=[]
        for path in sorted((a.out/'scores').glob('*/scores.csv')):
            try:
                m.frozen_files(path.parent)
                with path.open() as f: own=list(csv.DictReader(f))
                if path.parent.name.startswith('x-t2-'):
                    expected=path.parent.name.rsplit('-s',1)[0]
                    require(all(r['arm']==expected for r in own),'tier-2 CSV arm differs from its run directory')
                values.extend(own)
            except (ValueError,OSError,KeyError,TypeError) as error:
                if path.parent.name.startswith('x-t2-'):
                    atomic(a.out/'tier2/score-errors'/f'{path.parent.name}.json',dict(status='INCONCLUSIVE',reason=str(error)))
                elif path.parent.name.startswith('x-p-'):p_errors.append(str(error))
                else:raise
        data = tables(values, doc, read(a.out/'support.json'), inherited_local(a.out), a.bootstrap,
                      read(a.out/'m2-scoring-machine.json') == doc['core_scoring_machine']['identity'])
        if p_errors:data['comparisons']['p'].update(status='INCONCLUSIVE',evidence_error='; '.join(p_errors))
        if (a.out/'tier2/plan.json').exists() and 'tier2' not in data:
            from kit.x_tier2_read import tables as tier2_tables
            data['tier2']=tier2_tables(values,doc,read(a.out/'support.json'),inherited_local(a.out),a.bootstrap,
                read(a.out/'m2-scoring-machine.json') == doc['core_scoring_machine']['identity'])
        if 'tier2' in data:
            data['tier2']['p2_feasibility']=('DEFERRED' if doc.get('p_variant')=='disabled' or
                (a.out/'dispositions/p.json').exists() and read(a.out/'dispositions/p.json').get('status')=='DEFERRED' else 'PENDING_OR_FEASIBLE')
            for error in (a.out/'tier2/score-errors').glob('*.json'):
                from kit.x_tier2 import identity
                name=identity(error.stem.rsplit('-s',1)[0].removeprefix('x-'))['comparison']
                data['tier2']['comparisons'][name].update(status='INCONCLUSIVE',evidence_error=read(error)['reason'])
        atomic(a.out/'comparison-tables.json', data)
        print('X tables only; no node admission or Screen S efficacy interpretation')
    else:
        from kit.x_collect import load
        evidence = load(a.source, a.standin)
        data = json.loads(evidence['comparison-tables.json'])
        doc = json.loads(evidence['binding.json'])
        validation=validate_compact(evidence,doc,a.standin)
        import gzip
        values = list(csv.DictReader(io.StringIO(gzip.decompress(evidence['scores.csv.gz']).decode())))
        identities={}
        for name,raw in evidence.items():
            if name.startswith('scores/') and name.endswith('/m2-identity.json'):
                if name.startswith('scores/x-t2-'):continue
                if name.startswith('scores/x-p-') and validation['status']=='INCONCLUSIVE':continue
                meta=json.loads(raw);identities[(meta['arm'],meta['seed'],meta['step'],meta['panel'])]=meta
        for row in values:
            if row['arm'].startswith('x-t2-'):continue
            panel='general-v1' if row['panel'] in ('ifeval','math','knowledge') else row['panel']
            if row['arm']=='x-p' and validation['status']=='INCONCLUSIVE':continue
            found=(row['arm'],int(row['seed']),int(row['step']),panel) in identities
            if not found and row['arm']=='x-p':
                validation=dict(status='INCONCLUSIVE',reason='returned P2 score missing identity');continue
            require(found,'returned score missing identity')
        for name,raw in evidence.items():
            if not name.endswith('/alias.json'):continue
            if name.startswith('scores/x-t2-'):continue
            if name.startswith('scores/x-p-') and validation['status']=='INCONCLUSIVE':continue
            try:
                alias=json.loads(raw);source=alias['source']
                manifest=json.loads(evidence[source+'/manifest.json'])
                require(manifest['files']['scores.csv']==alias['scores_sha256'] and
                    manifest['files']['m2-identity.json']==alias['identity_sha256'],'incoming alias hash changed')
            except (ValueError,KeyError,TypeError) as error:
                if not name.startswith('scores/x-p-'):raise
                validation=dict(status='INCONCLUSIVE',reason='P2 incoming alias defective: '+str(error))
        recomputed = tables(values,doc,json.loads(evidence['support.json']),
            inherited_return(evidence),a.bootstrap,
            json.loads(evidence['m2-scoring-machine.json']) == doc['core_scoring_machine']['identity'])
        # Tables use a registered 10,000-draw secondary bootstrap in production;
        # fixtures explicitly pass their smaller count on both sides.
        if validation['status']=='INCONCLUSIVE' or recomputed['comparisons']['p']['status']=='INCONCLUSIVE':
            require(without_p(data)==without_p(recomputed),'returned A/control comparison tables disagree with score evidence')
            if validation['status']!='INCONCLUSIVE':validation=dict(status='INCONCLUSIVE',reason=recomputed['comparisons']['p']['evidence_error'])
        else:
            require(without_p(data)==without_p(recomputed),'returned A/control comparison tables disagree with score evidence')
            if {k:v for k,v in data.items() if k!='tier2'}!={k:v for k,v in recomputed.items() if k!='tier2'}:
                validation=dict(status='INCONCLUSIVE',reason='returned P2 tables disagree with score evidence')
        qualification=None
        if a.screen_qualification is not None:
            qualification = read(a.screen_qualification)
            require(a.screen_return is not None and qualification['screen_return_sha256'] == sha(a.screen_return), 'Screen S return hash mismatch')
            from kit.x_qualify import qualify
            require(digest(qualification) == digest(qualify(a.screen_return,a.standin)), 'Screen S qualification differs from frozen analysis of bound return')
        else:require(doc.get('p_variant')=='p2-l2-init','Screen S qualification file required')
        isolate_p2(data,validation)
        if 'tier2' in data:
            from kit.x_tier2_read import validate as validate_tier2, tables as tier2_tables
            try:
                second=recomputed.get('tier2') or tier2_tables(values,doc,json.loads(evidence['support.json']),
                    inherited_return(evidence),a.bootstrap,
                    json.loads(evidence['m2-scoring-machine.json']) == doc['core_scoring_machine']['identity'])
                errors=validate_tier2(evidence,doc,a.standin,values,validation)
                from kit.x_tier2 import COMPARISONS
                require(set(data['tier2']['comparisons'])==set(COMPARISONS),'tier-2 matrix identity changed')
                for name,c in data['tier2']['comparisons'].items():
                    if c != second['comparisons'].get(name):errors[name]='tier-2 table differs from scores'
                    if name in errors:c.update(status='INCONCLUSIVE',evidence_error=errors[name])
                data['tier2']['p2_feasibility']=('DEFERRED' if doc.get('p_variant')=='disabled' or
                    'dispositions/p.json' in evidence and json.loads(evidence['dispositions/p.json']).get('status')=='DEFERRED' else 'PENDING_OR_FEASIBLE')
            except (ValueError,KeyError,TypeError,AttributeError,IndexError) as error:
                from kit.x_tier2 import COMPARISONS
                data['tier2']=dict(comparisons={name:dict(status='INCONCLUSIVE',evidence_error=str(error)) for name in COMPARISONS})
        result = reading(data, qualification, a.standin)
        result['p2_validation']=validation
        result['cost'] = json.loads(evidence['costs.json'])
        atomic(a.out/'reading.json', result)
        print('Frozen add-on reading:', result['admitted'], result['stacking'])


def configure(p):
    p.add_argument('--action', choices=('tables','read'), required=True)
    p.add_argument('--source', type=Path); p.add_argument('--screen-qualification', type=Path)
    p.add_argument('--screen-return', type=Path)
    p.add_argument('--bootstrap', type=int, default=10000)
if __name__ == '__main__': raise SystemExit(entry(job, configure))
