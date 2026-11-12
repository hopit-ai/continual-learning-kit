"""Descriptive estimates and independent ASCII count/summary records."""
import json
import math
from kit.trace_common import *


def op_bwt(matrix):
    t = len(matrix)-1
    need(t > 0 and all(len(row)==t for row in matrix), 'incomplete score matrix')
    op = statistics_mean(matrix[-1])
    if t == 1:
        return dict(OP=op,BWT_TRACE=None,BWT_Tminus1=None)
    total = math.fsum(matrix[-1][i]-matrix[i+1][i] for i in range(t-1))
    return dict(OP=op,BWT_TRACE=total/t,BWT_Tminus1=total/(t-1))


def statistics_mean(values):
    return math.fsum(values)/len(values)


def descriptive(scores):
    """Input is study -> seed -> state -> task/panel mean, all on 0–100."""
    result = dict(reporting='DESCRIPTIVE',sources={})
    baseline = [op_bwt([[scores['s8'][str(seed)].get('base' if i==0 else 'S'+str(i),{}).get(task)
                        for task in TASKS] for i in range(9)]) for seed in SEEDS] if 's8' in scores else []
    result['baseline'] = {name:stats([r[name] for r in baseline])
                          for name in ('OP','BWT_TRACE','BWT_Tminus1')} if baseline else None
    for source in SOURCES:
        if source not in scores:
            continue
        deltas = []
        intervals = {}
        def collect(name, value):
            intervals.setdefault(name,[]).append(value)
        for seed in SEEDS:
            cells = scores[source][str(seed)]
            deltas.append(statistics_mean([cells['F3'][t]-cells['D3'][t] for t in SUBSET[:2]]))
            for task in SUBSET[:2]:
                collect('retained_'+task,cells['F3'][task]-cells['D3'][task])
                for arm in ('D','F'):
                    acquired = str(SUBSET.index(task)+1)
                    collect(arm+'_own_retention_'+task,cells[arm+'3'][task]-cells[arm+acquired][task])
            for i,task in enumerate(SUBSET,1):
                for arm in ('D','F'):
                    collect(f'{arm}_acquisition_{task}',cells[arm+str(i)][task]-cells[arm+str(i-1)][task])
                collect('acquisition_difference_'+task,
                    (cells['F'+str(i)][task]-cells['F'+str(i-1)][task])-
                    (cells['D'+str(i)][task]-cells['D'+str(i-1)][task]))
            for panel in ('instruction','maths','knowledge'):
                for i in range(4):
                    collect(f'{panel}_F_minus_D_{i}',cells['F'+str(i)][panel]-cells['D'+str(i)][panel])
                for arm in ('D','F'):
                    collect(arm+'_base_delta_'+panel,cells[arm+'3'][panel]-cells[arm+'0'][panel])
            for i,task in enumerate(SUBSET[1:],2):
                for arm in ('D','F'):
                    collect(f'{arm}_chain_standalone_{i}',cells[arm+str(i)][task]-cells['U'+arm+str(i)][task])
                    collect(f'U{arm}_acquisition_{i}',cells['U'+arm+str(i)][task]-cells[arm+'0'][task])
        secondary = {}
        for arm in ('D','F'):
            matrices = [op_bwt([[scores[source][str(seed)][arm+str(i)][task]
                         for task in SUBSET] for i in range(4)]) for seed in SEEDS]
            secondary[arm] = {name:stats([r[name] for r in matrices])
                              for name in ('OP','BWT_TRACE','BWT_Tminus1')}
        result['sources'][source] = dict(retained_effect=stats(deltas), secondary=secondary,
                                        descriptive_guards={k:stats(v) for k,v in intervals.items()})
    result['descriptive_source_effect_difference'] = stats([
        statistics_mean([scores['27b'][str(seed)]['F3'][task]-scores['27b'][str(seed)]['D3'][task]
                         -scores['gold'][str(seed)]['F3'][task]+scores['gold'][str(seed)]['D3'][task]
                         for task in SUBSET[:2]]) for seed in SEEDS]) if set(SOURCES) <= set(scores) else None
    return result


def record_line(key, value):
    return key+'='+json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def pack(records, key, values, prefix=()):
    """Independent chunks carry their own key, offset and denominator."""
    start, chunk = 0, []
    for value in values:
        trial = [*prefix, *chunk, value]
        if len(record_line(key+'.'+str(start), trial)) >= 90 and chunk:
            records[key+'.'+str(start)] = [*prefix, *chunk]
            start += len(chunk)
            chunk = []
        chunk.append(value)
    if chunk:
        records[key+'.'+str(start)] = [*prefix, *chunk]


def pages(payload, authority, kind='ESSENTIAL'):
    import hashlib
    lines = [(v if k=='judge-profile' else record_line(k,v))
             for k,v in sorted(payload['records'].items())]
    need(all(line.isascii() and len(line)<90 for line in lines),
         'return record too long '+str([line.split('=')[0] for line in lines if len(line)>=90]))
    chunks = [lines[i:i+69] for i in range(0,len(lines),69)] or [[]]
    result = []
    for index, chunk in enumerate(chunks,1):
        body = '\n'.join([kind+' DESCRIPTIVE', 'TRANSCRIPTION-UNVERIFIED',
                          'CHECKSUM ADVISORY', *chunk])+'\n'
        result.append(f'KIT TRACE {authority} PART {index}/{len(chunks)}\n'+body+
                      'SHA256 '+hashlib.sha256(body.encode('ascii')).hexdigest()+'\n')
    return result


def split_return(payload, authority):
    """Keep the pasted summary to three pages, preserving every spill locally."""
    records = payload['records']
    full_only = ('S.', 'SA.', 'J.', 'C.', 'A.', 'U.', 'lane.', 'env.', 'tokens_caps.',
                 'tokens.D.', 'calibration_ids')
    candidates = [(k,v) for k,v in records.items() if not k.startswith(full_only)]
    # Put requested results ahead of supporting hashes when unusual evidence
    # volumes need more than the pasted allowance.
    first = ('reporting', 'technical_synthetic', 'registration', 'binding', 'config',
             'seeds', 'departures.', 'licence.', 'audit', 'OP.', 'effect.', 'gpu_h.',
             'phase.', 'judge-profile', 'failures', 'EOS.', 'EOSr.', 'cap_rates.', 'utilisation', 'missing.')
    candidates.sort(key=lambda item: (not item[0].startswith(first), item[0]))
    selected = dict(candidates[:3*69-3])
    spill = len(candidates)-len(selected)
    selected.update({'missing.full_spill':spill, 'full.records':len(records),
                     'full.sha256':digest(payload)})
    essential = dict(payload, records=selected)
    return essential, pages(essential,authority), pages(payload,authority,'FULL')


def reject_nonfinite(value):
    raise ValueError('nonfinite record number')


def parse_pages(texts):
    """A bad line loses only its record. Digests never gate transcription."""
    records, invalid, checksums = {}, [], []
    import hashlib
    import re
    for page, text in enumerate(texts,1):
        lines = text.splitlines()
        if lines and lines[-1].startswith('SHA256 '):
            body = '\n'.join(lines[1:-1])+'\n'
            if lines[-1][7:] != hashlib.sha256(body.encode('ascii',errors='replace')).hexdigest():
                checksums.append(page)
        for number, line in enumerate(lines,1):
            if line.startswith(('KIT ', 'ESSENTIAL ', 'FULL ', 'TRANSCRIPTION-', 'CHECKSUM ', 'SHA256 ')):
                continue
            try:
                if line.startswith('TRACE|judge-profile|'):
                    records['judge-profile'] = line
                    continue
                key, raw = line.split('=',1)
                need(re.fullmatch(r'[A-Za-z0-9_.-]+',key) is not None, 'invalid record key')
                value = json.loads(raw, parse_constant=reject_nonfinite)
                if key in records:
                    need(records[key] == value, 'conflicting duplicate record')
                records[key] = value
            except (ValueError, TypeError):
                invalid.append([page,number])
    result = dict(schema='trace-return.v2', reporting='DESCRIPTIVE',
                  transcription='TRANSCRIPTION-UNVERIFIED', checksums='ADVISORY', records=records)
    if invalid:
        result['invalid_records'] = invalid
    if checksums:
        result['checksum_mismatches'] = checksums
    return result


def score_states(study, seed, metric, all_states=False):
    ordered = states(study,seed)
    if all_states:
        return ordered
    if study == 's8' and metric in TASKS:
        return [state for state in ordered if state['stage'] in {TASKS.index(metric)+1,8}]
    if study != 's8':
        return [state for state in ordered if not state['arm'].startswith('U') or
                metric == SUBSET[state['stage']-1]]
    return ordered


def score_records(records, numeric):
    for study, seeds in numeric.items():
        for seed, cells in seeds.items():
            for metric in next(iter(cells.values())):
                # FULL retains every state, including standalone panel diagnostics.
                ordered = score_states(study,int(seed),metric,all_states=True)
                values = [cells[state['id']][metric] for state in ordered]
                need(len({v[1] for v in values})==1, 'score population changed across states')
                pack(records,f'SA.{study}.{seed}.{metric}',[v[0] for v in values],
                     prefix=[values[0][1]])


def effect_records(records, scores):
    """Per-seed OP, both BWT conventions, and retained-task source effects."""
    for study, seeds in scores.items():
        tasks = TASKS if study == 's8' else SUBSET
        arms = ('S',) if study == 's8' else ('D','F')
        for seed, cells in seeds.items():
            for arm in arms:
                matrix = [[cells['base' if study == 's8' and i == 0 else arm+str(i)][task]
                           for task in tasks] for i in range(len(tasks)+1)]
                estimates = op_bwt(matrix)
                pack(records,f'OP.{study}.{seed}.{arm}',
                     [estimates[k] for k in ('OP','BWT_TRACE','BWT_Tminus1')])
            if study != 's8':
                retained = [cells['F3'][task]-cells['D3'][task] for task in SUBSET[:2]]
                pack(records,f'effect.{study}.{seed}',[statistics_mean(retained),*retained])
    if set(SOURCES) <= set(scores):
        for seed in map(str,SEEDS):
            effect = statistics_mean([
                scores['27b'][seed]['F3'][task]-scores['27b'][seed]['D3'][task]
                -scores['gold'][seed]['F3'][task]+scores['gold'][seed]['D3'][task]
                for task in SUBSET[:2]])
            records[f'effect.27b_minus_gold.{seed}'] = effect


def returned_scores(payload):
    """Reconstruct score means from independent S records and registered ordering."""
    scores = {}
    for key, values in payload['records'].items():
        if not key.startswith(('S.','SA.')):
            continue
        kind,study,seed,metric,offset = key.split('.')
        cells = scores.setdefault(study,{}).setdefault(seed,{})
        order = score_states(study,int(seed),metric,all_states=kind == 'SA')
        n, *sums = values
        need(n > 0, 'invalid score denominator')
        for index,total in enumerate(sums,int(offset)):
            cells.setdefault(order[index]['id'],{})[metric] = total/n
    return scores


def output_diagnostics(root):
    """Aggregate local token/cap evidence, preserving explicit unavailable fields."""
    groups = {}
    paths = [*((root/'scores').rglob('*.json')), *((root/'judge').rglob('*raw.json'))]
    for path in paths:
        data = load(path)
        outputs = data.get('outputs', []) if isinstance(data,dict) else data
        if not isinstance(outputs,list):
            continue
        role = 'judge' if 'judge' in path.relative_to(root).parts else 'score'
        for row in outputs:
            if not isinstance(row,dict) or 'tokens' not in row:
                continue
            acc = groups.setdefault(role,[0,0,0,0,0])
            # calls, input tokens, output tokens, prompt cap hits, response cap hits
            acc[0] += 1
            for position,key in ((1,'prompt_tokens'),(3,'prompt_cap_hit')):
                if key not in row:
                    acc[position] = None
                elif acc[position] is not None:
                    acc[position] += row[key]
            acc[2] += row['tokens']
            acc[4] += row.get('response_cap_hit',row.get('finish_reason')=='length')
    for path in (root/'target-journal').glob('*.json'):
        data = load(path)
        for row in (data['raw'],(data.get('verification') or {}).get('raw')):
            if row is None:
                continue
            acc = groups.setdefault('teacher',[0,0,0,0,0])
            acc[0] += 1
            acc[1] += row.get('input_tokens',0)
            acc[2] += row.get('teacher_tokens',0)
            acc[3] += row.get('input_over_cap',False)
            acc[4] += row.get('finish_reason')=='length'
    for path in (root/'training').glob('s8-*/token-counts.json'):
        counts = load(path)
        acc = groups.setdefault('training',[0]*5)
        groups['training'] = [a+b for a,b in zip(acc,counts)]
    return groups


def collect(cfg):
    root = Path(cfg['out'])
    prepared = load(root/'prepared.json')
    studies = prepared.get('active_studies', cfg.get('studies',['s8',*SOURCES]))
    scores, numeric, judged = {},{},{}
    records = dict(reporting='DESCRIPTIVE', comparisons=len(studies), seeds=list(SEEDS),
                   technical_synthetic=bool(cfg.get('technical_synthetic')),
                   registration=sha(REG),
                   binding=sha(root/'binding.json'), config=digest(cfg),
                   blocked=sorted(prepared.get('blocked_studies',{})))
    from kit.trace_sft import deepspeed_config
    records['recipe.s8'] = digest(deepspeed_config())
    records['recipe.D'] = sha(KIT/'resources/p3b/v4-recipes.json')
    pack(records,'departures',list(DEPARTURES))
    for study in studies:
        scores[study], numeric[study], judged[study] = {},{},{}
        for seed in SEEDS:
            base = root/'scores'/f'{study}-{seed}'
            lane = load(base/'lane.json')
            need(lane['complete'] and lane['config_sha256']==digest(cfg), 'score lane identity differs')
            need(lane['states']==states(study,seed), 'score lane population differs')
            for name,expected in lane.get('artifacts_sha256',{}).items():
                need(sha(root/name)==expected,'score lane artifact changed')
            cells, sufficient = {},{}
            for state in states(study,seed):
                values = load(base/state['id']/'result.json')
                sufficient[state['id']] = {k:[round(v['sum'],9),v['n']] for k,v in values.items()}
                cells[state['id']] = {k:total/n for k,(total,n) in sufficient[state['id']].items()}
            scores[study][str(seed)] = cells
            numeric[study][str(seed)] = sufficient
            judges = load(root/'judge'/f'{study}-{seed}'/'summary.json')
            for state,panels in judges.items():
                if state=='_profile':
                    continue
                # Panel and order counts stay independently recoverable.
                pack(records,f'J.{study}.{seed}.{state}',
                     [[v for order in panels[panel] for v in order]
                      for panel in ('Self-Instruct','LIMA','CoNa')])
            records[f'lane.{study}.{seed}'] = digest(lane)
            records[f'env.{study}.{seed}'] = lane['environment_sha256']
    summary = descriptive(scores)
    score_records(records,numeric)
    effect_records(records,scores)
    ledger = [dict(load(p), attempt_sha256=digest(str(p.relative_to(root))), receipt_sha256=sha(p))
              for p in sorted((root/'attempts').glob('*/*/receipt.json'))]
    ledger = [r for r in ledger if r['row']!='collect']
    phases = {}
    for receipt in ledger:
        phase = receipt['row'].split('-')[0]
        phases[phase] = phases.get(phase,0)+receipt.get('gpu_hours',0)
        key = 'A.'+receipt['attempt_sha256'][:12]
        records[key+'.sha256'] = receipt['receipt_sha256']
        records[key] = [receipt['row'], round(receipt.get('gpu_hours',0),6),
                        receipt['status'],len(receipt.get('warnings',[]))]
        pack(records,key+'.clock',[receipt.get(k) for k in ('gpus','started','ended')])
        records[key+'.error'] = receipt.get('error_type')
        if phase == 'qualify':
            records['Q.'+receipt['row'].split('-')[1]] = [receipt.get('result',{}).get(k,'PENDING')
                 for k in ('identity','leakage','freeze_bit_equality','scorer_fixtures','licences')]
    for phase, hours in phases.items():
        records['gpu_h.'+phase] = round(hours,6)
    if cfg.get('judge_throughput_probe'):
        from kit.trace_judge import probe_essential as essential, decision_receipt
        records['judge-profile'] = essential(decision_receipt(cfg))
        for study in studies:
            for seed in SEEDS:
                records[f'profile.{study}.{seed}'] = load(
                    root/'judge'/f'{study}-{seed}'/'profile.json')['profile']
    from kit.trace_jobs import timing_rows as phase_rows
    records.update(phase_rows(root))
    records['gpu_h.total'] = round(math.fsum(phases.values()),6)
    records['failures'] = sum(r['status']!='COMPLETE' for r in ledger)
    records['warnings'] = sum(len(r.get('warnings',[])) for r in ledger)
    records['budget'] = digest(ledger)
    labels_path = root/'judge'/'calibration-labels.json'
    if labels_path.exists():
        labels = load(labels_path)
        records['calibration_ids'] = digest([r['id'] for r in labels])
        pack(records,'C',[r['labels'] for r in labels])
        records['calibration'] = 'MEASURED'
    else:
        records['calibration'] = 'PENDING'
    for task, splits in prepared.get('counts',{}).items():
        records['N.'+task] = [splits[k] for k in ('train','eval','test')]
    for task, splits in prepared.get('l_eos',{}).items():
        for split,value in splits.items():
            records[f'EOS.{task}.{split}'] = [value['rows'],value['affected'],value['suffix_errors'],
                                            value.get('eos_position_errors')]
            pack(records,f'EOSr.{task}.{split}',[
                 value.get('share',value['affected']/value['rows'] if value['rows'] else 0.),
                 value.get('original_share'),value.get('excess_share')])
            if value.get('original_share') is None:
                records[f'missing.EOS.{task}.{split}'] = 'ORIGINAL_UNMEASURED'
    if not prepared.get('l_eos'):
        records['L-EOS'] = 'UNMEASURED'
    for study,status in prepared.get('licences', {'all':'UNMEASURED'}).items():
        records['licence.'+study] = status
    # Opaque hashes bind complete local populations and metric vectors.
    for name in ('counts','selected','audit','source_sha256'):
        records['H.'+name] = digest(prepared.get(name,{}))
    records['H.scores'] = digest(numeric)
    records['H.vectors'] = digest({str(p.relative_to(root)):sha(p)
        for p in (root/'scores').rglob('*.json') if p.name != 'lane.json'})
    records['H.models'] = digest({r:cfg.get(r,{}) for r in ('base','teacher','judge','original_tokenizer')})
    records['H.prompts'] = digest(prepared.get('datasets',{}))
    records['audit'] = prepared.get('audit',{}).get('unresolved','UNMEASURED')
    records['L-EOS.warn_excess'] = L_EOS_WARN_EXCESS
    records['L-EOS.position_bound'] = 1
    records['gradient_clipping'] = GRADIENT_CLIPPING
    records['H.environments'] = digest({k:v for k,v in records.items() if k.startswith('env.')})
    diagnostics = output_diagnostics(root)
    for role in ('score','judge','teacher','training'):
        counts = diagnostics.get(role)
        records['tokens_caps.'+role] = counts or 'UNMEASURED'
        records['cap_rates.'+role] = ([hits/counts[0] if hits is not None else 'UNMEASURED'
                                      for hits in counts[3:]] if counts else 'UNMEASURED')
        if not counts or any(value is None for value in counts):
            records['missing.tokens_caps.'+role] = 'UNMEASURED'
    telemetry = list((root/'utilisation').glob('*.summary.json'))
    measured, minimums, weighted_util, gpu_seconds, incomplete = [], [], 0., 0., 0
    for path in telemetry:
        value = load(path)
        key = 'U.'+digest(path.name)[:12]
        records[key] = [value.get('mean_util'),value.get('min_window_mean'),
                        value.get('telemetry_complete',False)]
        records[key+'.sha256'] = sha(path)
        for field in ('gpus','wall_s','seconds_below_70','startup_wall_s','startup_mean_util',
                      'startup_seconds_below_70','telemetry_errors','returncode','interval_s',
                      'window_s','startup_grace_seconds','holder_slice','holder_selector_queries'):
            records[key+'.'+field] = value.get(field)
        weight = (value.get('gpus') or 0)*(value.get('wall_s') or 0)
        if value.get('telemetry_complete') and value.get('mean_util') is not None and weight > 0:
            measured.append(value['mean_util'])
            weighted_util += value['mean_util']*weight
            gpu_seconds += weight
            if value.get('min_window_mean') is not None:
                minimums.append(value['min_window_mean'])
        else:
            incomplete += 1
    records['utilisation.allocations'] = [len(telemetry),len(measured),incomplete]
    records['utilisation.mean'] = weighted_util/gpu_seconds if gpu_seconds else 'UNMEASURED'
    records['utilisation.min_window'] = min(minimums) if minimums else 'UNMEASURED'
    records['utilisation.below70'] = sum(value < 70 for value in measured)
    if not telemetry or incomplete or not gpu_seconds:
        records['missing.utilisation'] = 'UNMEASURED_OR_INCOMPLETE'
    native_tokens = [0,0]
    for path in (root/'training').glob('*/*/native-receipt.json'):
        for update in load(path).get('history',[]):
            native_tokens[0] += update.get('nm/response_tokens',0)
            native_tokens[1] += update.get('nm/actor_padded_tokens',0)
    records['tokens.D.response_padded'] = native_tokens if any(native_tokens) else 'UNMEASURED'
    cost = cfg.get('serving_cost')
    records['serving_cost'] = cost if type(cost) in (int,float) else 'UNMEASURED'
    if records['serving_cost'] == 'UNMEASURED':
        records['missing.serving_cost'] = 'UNMEASURED'
    if records['calibration'] == 'PENDING':
        records['missing.calibration'] = 'PENDING'
    payload = dict(schema='trace-return.v2', reporting='DESCRIPTIVE',
                   transcription='TRANSCRIPTION-UNVERIFIED',checksums='ADVISORY', records=records)
    atomic(root/'descriptive.json',summary)
    essential, texts, full_texts = split_return(payload,sha(REG)[:12])
    need(parse_pages(texts)==essential, 'essential text return round trip differs')
    need(parse_pages(full_texts)==payload, 'full text return round trip differs')
    reconstructed = returned_scores(payload)
    need(descriptive(reconstructed)==summary, 'count return estimates differ')
    atomic(root/'return-numeric.json',essential)
    full_root = root/'full'
    full_root.mkdir(exist_ok=True)
    atomic(full_root/'return-numeric.json',payload)
    for old in root.glob('return-*.txt'):
        old.unlink()
    for index,text in enumerate(texts,1):
        (root/f'return-{index}.txt').write_text(text)
    for old in full_root.glob('return-*.txt'):
        old.unlink()
    for index,text in enumerate(full_texts,1):
        (full_root/f'return-{index}.txt').write_text(text)
    return essential
