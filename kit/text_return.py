"""ASCII text transport. Presentation only: scientific verdicts come from native readers.

Checksums cover the exact canonical body, including its final LF. The importer
restores record boundaries after chat wrapping, rather than correcting values.
"""
from __future__ import annotations
import ast
import hashlib
import hmac
import secrets
import json
import math
from pathlib import Path
import re
import statistics

SCHEMA = 'kit-text-return.v1'
LABELS = ('PASS', 'NOT MET', 'INCONCLUSIVE', 'PENDING', 'UNVERIFIED', 'UNVERIFIED-HARDWARE', 'DEFERRED')
FAMILIES = ('M', 'X', 'S', 'KCR', 'PD', 'SP', '4A')
MAX_CHARS = 12000


def canonical(value):
    # JSON object keys are strings. Normalize before sorting so an integer
    # update key (5,10) has exactly the same digest after JSON reload ('5','10').
    def normalize(v):
        if isinstance(v,dict):
            result={str(k):normalize(x) for k,x in v.items()}
            if len(result)!=len(v):raise ValueError('duplicate canonical JSON keys')
            return result
        if isinstance(v,(list,tuple)):return [normalize(x) for x in v]
        return v
    return json.dumps(normalize(value), sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode('ascii')).hexdigest()


def label(value):
    if value in LABELS: return value
    return {'FAIL': 'NOT MET', 'NOT ESTABLISHED': 'NOT MET', 'ESTABLISHED': 'PASS',
            'PLASTICITY LOSS SHOWN': 'PASS', 'INCOMPLETE': 'PENDING',
            'DESCRIPTIVE': 'DEFERRED', 'NOT_OPEN': 'DEFERRED', 'UNINFORMATIVE':'DEFERRED', 'DEGENERATE':'DEFERRED'}.get(value, 'PENDING')


# Only vocabulary from the shipped calculators may name a metric. Arbitrary
# result keys get opaque aliases; arbitrary text values are never transported.
# In particular, exception messages, provenance and identity/location maps do
# not pass through a regex-based "redactor" that might miss a private token.
READER_SOURCES = ('m2_read.py', 'x_read.py', 'x_tier2_read.py', 'm3_read.py',
                  'pd_read.py', 'sp_read.py', 'a4_read.py')
PRIVATE = re.compile(r'(?:path|uuid|hostname|username|job.?id|directory|environment|storage|tokenizer|identity|sha256|checkpoint|raw|predictions|covariates|lost_ids|gained_ids|membership_ids|acquired_ids|recovered_ids|panel_ids|evaluation_ids|model|settings|hardware|sources|reason|error|policy|rules|interpretation|note)', re.I)
OMIT = {'missing', 'missing_entries', 'omissions', 'transitions', 'cell_taus', 'generation', 'items'}


def vocabulary():
    root = Path(__file__).resolve().parent
    names = set(FAMILIES) | set(LABELS) | {'all', 'plain', 'sibling', 'base', 'control', 'drift', 'plasticity', 'p', 'K', 'CR', 'G', 'C0', 'finqa', 'contract_nli', 'math_hard', 'general-v1', 'required_word'}
    for source in [root / n for n in READER_SOURCES] + [root.parent / 'scripts/screen_s_analysis.py']:
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg: names.add(node.arg)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if len(value) < 160 and '\n' not in value:
                    names.add(value)
    # Registered, dynamically composed M3 guard/contrast labels.
    from kit.m3_read import Table, CONTRASTS
    names.update(Table({}).guards()); names.update(Table({}).diagnostics()); names.update(CONTRASTS)
    for trajectory in ('plain', 'sibling'):
        for category in ('acquisition', 'retention', 'sequence_acquisition'):
            keys = ('finqa', 'contract_nli', 'math_hard') if category != 'retention' else tuple(f'{p}_stage{s}' for p in ('finqa','contract_nli','ifeval','knowledge','math') for s in (1,2,3))
            names.update(keys)
            names.update(trajectory + '.' + category + '.' + k for k in keys)
    from kit.x_tier2 import COMPARISONS
    names.update(COMPARISONS)
    names.update('tier2.'+k for k in COMPARISONS)
    from kit.m2_common import CONFIGS
    from kit.x_common import ARMS
    names.update(CONFIGS); names.update(ARMS)
    names.update('x-'+k for k in ARMS)
    names.update('legal-'+k for k in ('plain',*CONFIGS))
    names.update('standalone-legal-'+k for k in ('plain',*CONFIGS))
    names.update(('finqa-plain','finqa-sibling','math-plain','math-sibling','standalone-math-plain','standalone-math-sibling','message2','message3','incoming','final','base','canonical','paraphrase','arrival','consolidation','direct','frozen','memory','off','on'))
    names.update(arm+'.'+stage for arm in ('K','C0','G','CR') for stage in ('finqa','contract_nli','math_hard'))
    names.update('2026-'+str(m).zfill(2) for m in range(1,7))
    names.update(('acquisition-positive','positive-months','process_wall_seconds','sum','max','broad-generic','earlier-generic','both-broad','both-generic','finqa-minus-C0','finqa-minus-G','change-strict','change-instruction','change-template_reliance_gap','tier2','standalone','admission-evaluated','READER_REFUSED','MISSING_INPUT','READER_ERROR','NOT_CONFIGURED','CONFIRMATION_NOT_ADMITTED','broad-confirmation','earlier-confirmation','stage2-base','consolidation-memory','descriptive','recorded_gpu_hours','tier_recorded_gpu_hours','tier_complete','total_recorded_gpu_hours','by_family','accounting','allocation_idle_reconciliation','budget_gate','allowance_gpu_hours','tier_over_allowance_warning','warning_summary','wall_time_risk','untimed','unverified_constituents','reader_unavailable','warning_count','failed_guard_count','guards_in_metrics'))
    return names


def alias_context(key=None):
    return dict(hmac_key=key or secrets.token_hex(32),aliases={},dropped_unknown_numeric={})


def known_key(value,names):
    return value in names or (value.isdigit() and 0<=int(value)<=128)


def safe_key(value, names, mapping=None):
    value=str(value)
    if known_key(value,names) and re.fullmatch(r'[A-Za-z0-9_.+/<>=:-]{1,160}',value):return value
    if mapping is None:mapping=alias_context()
    if 'hmac_key' not in mapping:mapping.update(alias_context())
    alias='field-'+hmac.new(bytes.fromhex(mapping['hmac_key']),value.encode(),hashlib.sha256).hexdigest()[:16]
    mapping['aliases'][alias]=value
    return alias


def project(value, names, mapping, location='', allow_numbers=True):
    if 'hmac_key' not in mapping:mapping.update(alias_context())
    if isinstance(value,dict):
        result={}
        for key,item in sorted(value.items(),key=lambda kv:str(kv[0])):
            key=str(key)
            if PRIVATE.search(key) or key in OMIT or key.endswith('_ids'):continue
            known=known_key(key,names)
            if not known and isinstance(item,(int,float)) and not isinstance(item,bool):
                mapping['dropped_unknown_numeric'][location+'/'+key]=1
                continue
            safe=safe_key(key,names,mapping)
            result[safe]=project(item,names,mapping,location+'/'+safe,known)
        return result
    if isinstance(value,(list,tuple)):
        return [project(v,names,mapping,location+'/'+str(i),allow_numbers) for i,v in enumerate(value)]
    if value is None or isinstance(value,bool):return value
    if isinstance(value,(int,float)):
        if not allow_numbers:
            mapping['dropped_unknown_numeric'][location]=1
            return None
        return value if math.isfinite(value) else None
    if isinstance(value,str) and value in names and re.fullmatch(r'(finqa|contract_nli|math_hard)/(final|retention|retention_absolute|task|step20)/[A-Za-z0-9_]+',value):return value
    if isinstance(value,str) and value in names and re.fullmatch(r'[A-Za-z0-9_.+-]{1,80}',value) and not PRIVATE.search(value):return value
    if isinstance(value,str) and value in set(LABELS)|{'FAIL','COMPLETE','INCOMPLETE','DESCRIPTIVE','AT_COST','WITHIN_CEILING','UNDEFINED','UNPAIRED-GPU','MEASURED','UNMEASURED','ESTABLISHED','NOT ESTABLISHED','PLASTICITY LOSS SHOWN'}:return value
    return None


def leaves(value, path=''):
    if isinstance(value, dict):
        for k, v in sorted(value.items()): yield from leaves(v, path + '/' + k)
    elif isinstance(value, list):
        for i, v in enumerate(value): yield from leaves(v, path + '/' + str(i))
    else: yield path, value


def metric_leaves(value, path='', seed='all'):
    if isinstance(value,dict) and set(value)=={'@columns','@rows','@offset'}:
        columns=value['@columns']
        if not isinstance(columns,list) or len(set(columns))!=len(columns):raise ValueError('invalid metric columns')
        for i,row in enumerate(value['@rows'],value['@offset']):
            if len(row)!=len(columns):raise ValueError('invalid metric row width')
            yield from metric_leaves(dict(zip(columns,row)),path+'/'+str(i),seed)
    elif isinstance(value, dict):
        declared=value.get('seed')
        if str(declared) in ('101','102','103','104','105'):seed=str(declared)
        if isinstance(value.get('row'),(list,tuple)) and len(value['row'])==6 and value['row'][2] in (101,102,103,104,105):seed=str(value['row'][2])
        for k, item in sorted(value.items()):
            own = k if k in ('101','102','103','104','105') else seed
            yield from metric_leaves(item, path+'/'+k, own)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            own = str(101+i) if path.endswith('/values') and len(value) in (3,5) else seed
            yield from metric_leaves(item,path+'/'+str(i),own)
    else:yield path,value,seed


def metric_rows(family, value):
    # One scalar per logical key=value record; never a JSON object/table blob.
    for path, item, seed in metric_leaves(value):
        encoded=canonical(item).replace(' ',r'\u0020')
        kind='z' if item is None else 'b' if isinstance(item,bool) else 'n' if isinstance(item,(int,float)) else 's'
        yield 'R|'+family+'|reader|'+seed+'|'+path+'='+kind+':'+encoded


def body_text(rows):
    """Deterministic <=90-column body, restored after arbitrary chat wrapping."""
    return ''.join('\n'.join(row[i:i+90] for i in range(0,len(row),90))+'\n' for row in rows)


def section_counts(rows):
    return {kind:sum(row.startswith(kind+'|') for row in rows) for kind in ('D','R','E','J','W','T')}


def registered_decisions(scope):
    result=set()
    for key in scope:
        family=re.sub(r'[0-9]+$','',key) if key!='4A' else key
        if family not in FAMILIES:raise ValueError('invalid decision family')
        result.update((key,d['comparison']) for d in _stub_decisions(family,{'_reader_status':'PENDING'}))
    return result


def decision(family, comparison, verdict, primary=None, positive=None, guards=()):
    primary = dict(primary or {})
    if family=='KCR' and primary.get('mean') is not None and primary.get('lower') is not None:
        primary.setdefault('upper',2*primary['mean']-primary['lower'])
    return dict(family=family, comparison=comparison, verdict=label(verdict),
                estimate=primary.get('mean'), lower=primary.get('lower'), upper=primary.get('upper'),
                seeds_positive=positive, guards=list(guards))


def x_guard_failures(c):
    """Name unmet bars already used by x_read.admitted; never override its verdict."""
    if not c:return ['NOT_ELIGIBLE']
    if c.get('status')!='COMPLETE':return ['INCOMPLETE_EVIDENCE']
    primary=c.get('primary');p=c.get('endpoints',{}).get(primary,{})
    failed=[]
    if p.get('mean') is None or p['mean']<2:failed.append('PRIMARY_MEAN')
    if p.get('lower') is None or p['lower']<=0:failed.append('PRIMARY_INTERVAL')
    for key,v in c.get('endpoints',{}).items():
        if key!=primary and (v.get('lower') is None or v['lower'] < -1):failed.append(key)
    for key,v in c.get('inherited',{}).items():
        if not v.get('admission_gate',True):continue
        lower=v.get('interval',{}).get('lower')
        if v.get('status','COMPLETE')!='COMPLETE' or lower is None or (lower<=v['margin'] if v.get('strict') else lower<v['margin']):failed.append(key)
    if c.get('positive_acquisition_seeds',0)<3:failed.append('ACQUISITION_SEEDS')
    return failed


def decisions(family, r):
    out = []
    def add(name, status, p=None, positive=None, guards=()):
        out.append(decision(family,name,r.get('_reader_status',status),p,positive,(r.get('_warning','MISSING_INPUT'),) if r.get('_reader_status') else guards))
    if r.get('_reader_status') and family not in ('M','X'):
        status = r['_reader_status']
        expected = {'M': ('plain', 'sibling', 'admission-evaluated'), 'X': ('drift', 'plasticity', 'p', 'rung_b', 'incremental_plasticity', 'stacking'), 'S': ('drift_association', 'drift_forward', 'plasticity'), 'KCR': ('K-C0', 'CR-G'), 'PD': ('stage2-base',), 'SP': ('descriptive',), '4A': ('consolidation-memory', 'retention')}[family]
        for name in expected: add(name, status, guards=(r.get('_warning', 'MISSING_INPUT'),))
    elif family == 'M':
        for t in r.get('trajectories', []):
            name = t['trajectory']
            if 'status' in t: continue  # Registered endpoint stubs below carry incomplete trajectories.
            for category in ('acquisition', 'retention', 'sequence_acquisition'):
                for key, v in t.get(category, {}).items():
                    p = v.get('from_incoming', v).get('seed_bound', {})
                    seed_rows=v.get('from_incoming',v).get('seeds',[])
                    positive=v.get('positive_seeds',sum(x['change']>0 for x in seed_rows) if seed_rows else None)
                    add(name + '.' + category + '.' + key, v.get('status'), p, positive,
                        ('BAR_NOT_PASS',) if v.get('status') != 'PASS' else ())
        present={d['comparison'] for d in out}
        for trajectory in ('plain','sibling'):
            keys={'acquisition':('finqa','contract_nli','math_hard'),
                  'retention':tuple(f'{p}_stage{s}' for p,s in [('finqa',2),('finqa',3),('contract_nli',3)] + [(p,s) for p in ('ifeval','knowledge','math') for s in (1,2,3)]),
                  'sequence_acquisition':('contract_nli','math_hard')}
            for category, endpoints in keys.items():
                for endpoint in endpoints:
                    name=trajectory+'.'+category+'.'+endpoint
                    if name not in present:
                        dropped=r.get('stage3')=='DROPPED' and ('math_hard' in endpoint or 'stage3' in endpoint)
                        add(name,'DEFERRED' if dropped else r.get('_reader_status','PENDING'),guards=('NOT_CONFIGURED',) if r.get('_reader_status') else ())
        admission = r.get('admission', {})
        add('admission-evaluated', 'PASS' if admission.get('status') == 'COMPLETE' else r.get('_reader_status','PENDING'))
        for branch in ('broad','earlier'):
            disposition=admission.get(branch+'_decision')
            state='PASS' if disposition=='ADMITTED' else 'DEFERRED' if disposition=='NOT_NOMINATED' else 'INCONCLUSIVE' if disposition=='INCONCLUSIVE' else r.get('_reader_status','PENDING')
            candidate='broad' if branch=='broad' else 'both' if admission.get('broad') else 'earlier'
            control='broad' if branch=='earlier' and admission.get('broad') else 'generic'
            c=next((c for c in admission.get('comparisons',[]) if (c['candidate'],c['control'])==(candidate,control)),{})
            primary='math' if branch=='broad' else 'survival'
            add(branch+'-confirmation',state,c.get('endpoints',{}).get(primary),guards=('CONFIRMATION_NOT_ADMITTED',) if state=='INCONCLUSIVE' else ())
    elif family == 'X':
        for key in ('drift', 'plasticity', 'p'):
            admitted = r.get('admitted', {}).get(key)
            c = r.get('eligible_comparisons', {}).get(key, {})
            status = r.get('status', {}).get(key)
            state = r.get('_reader_status') or ('PASS' if admitted is True else ('DEFERRED' if status in ('DEFERRED', 'DEGENERATE', 'UNINFORMATIVE') else 'NOT MET') if admitted is False else label(status))
            p = c.get('endpoints', {}).get(c.get('primary'), {})
            add(key, state, p, c.get('positive_acquisition_seeds'), x_guard_failures(c) if admitted is not True else ())
        for key in ('rung_b', 'incremental_plasticity', 'stacking'):
            value = r.get(key, 'PENDING')
            add(key, 'PASS' if value=='ADMITTED' else 'NOT MET' if value=='NOT_ADMITTED' else 'DEFERRED' if value in ('NOT_OPEN','NOT_TRIGGERED','NOT_QUALIFIED','DEFERRED_NEW_REGISTRATION','DEFERRED','UNINFORMATIVE','DEGENERATE') else 'PENDING')
        from kit.x_tier2 import COMPARISONS
        eligible=r.get('tier2_eligible_comparisons',{})
        for key in COMPARISONS:
            if key not in eligible:
                archived=r.get('tier2_archived',{}).get(key,{})
                add('tier2.'+key,r.get('_reader_status') or ('PENDING' if not archived else 'DEFERRED'),guards=('UNBOUND_COMPARISON',))
        for key, c in eligible.items():
            from kit.x_read import admitted
            ok = admitted(c)
            add('tier2.' + key, 'PASS' if ok else 'NOT MET' if ok is False else label(c.get('status')),
                c.get('endpoints', {}).get(c.get('primary'), {}), c.get('positive_acquisition_seeds'),x_guard_failures(c) if ok is not True else ())
    elif family == 'S':
        for key in ('drift_association', 'drift_forward', 'plasticity'):
            v = r[key]; p = v.get('primary', v)
            # Screen S test_cells stores the aggregate under result.
            if 'result' in v: p = v['result']
            add(key, v['status'], p, p.get('positive'), ('SIGNAL_NOT_PASS',) if v['status'] != 'PASS' else ())
    elif family == 'KCR':
        for key in ('K-C0', 'CR-G'):
            v = r['contrasts'][key]; p = v.get('primary') or {}
            guards = list(v.get('failed_guards', [])) + [k for k, met in v.get('bars', {}).items() if not met]
            add(key, v['label'], p, sum(x > 0 for x in p.get('values', [])) if p else None, guards)
    elif family == 'PD':
        p = r.get('primary') or {}
        diffs = r.get('seed_differences', {})
        add('stage2-base', r['status'], p, sum(x < 0 for x in diffs.values()) if diffs else None,
            ('NATIVE_EVIDENCE_UNAVAILABLE',) if not p else ('SENSITIVITY_NOT_SHOWN',) if not r.get('sensitivity') else ())
    elif family == 'SP':
        add('descriptive', 'UNVERIFIED' if r.get('status') == 'UNVERIFIED' else 'PENDING' if r.get('status') == 'INCOMPLETE' else 'DEFERRED', guards=('DESCRIPTIVE_ONLY',))
    elif family == '4A':
        p = r.get('primary', {})
        add('consolidation-memory', r.get('verdict'), p.get('interval'), sum(x > 0 for x in p.get('per_seed', {}).values()))
        add('retention', 'PASS' if r.get('retention_established') else 'NOT MET', guards=('SEE_REGISTERED_BARS',))
    return out


# Independent fallback registry: a failing family formatter cannot prevent its stub.
_stub_decisions=decisions


def drow(d):
    vals = [d['family'], d['comparison'], d['verdict']] + [canonical(d[k]) for k in ('estimate', 'lower', 'upper', 'seeds_positive')] + [','.join(d['guards']) or '-']
    return 'D|' + '|'.join(vals)


def job_inventory():
    """Stable, public registered labels; independent of private launch settings."""
    labels=[]
    for line in Path(__file__).with_name('combined-inventory.md').read_text().splitlines():
        cells=[x.strip() for x in line.split('|')]
        if len(cells)>=4 and cells[2] in ('core','preflight','kcr','pd','sp') and re.fullmatch(r'[A-Za-z0-9_.-]+',cells[1]):
            if cells[1] not in labels:labels.append(cells[1])
    for prefix in ('PF-node-','PF-tier-'):
        for i in range(128):
            if prefix+str(i) not in labels:labels.append(prefix+str(i))
    return labels,sha(canonical(labels))


def rounded(value):
    """Four significant figures for text only; canonical node JSON is untouched."""
    if isinstance(value, float): return float(format(value, '.4g'))
    if isinstance(value, dict): return {k:rounded(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [rounded(v) for v in value]
    return value


def essential_endpoints(family, r):
    """Select presentation evidence, never calculate or override admission.

    Bounds/bars come from the registered reader. SD is descriptive sample SD
    when the reader supplies only its seed vector and bound. Selection uses
    unrounded values: a guard is retained when failing, unknown, or <=0.5
    points inside its bar. Primary records are always retained. 4A retains
    its registered five seeds rather than pretending it is a three-seed rule.
    """
    records=[]
    def add(name, values, stat=None, kind='P', bar='-', method='seed-t', near=True):
        if not near:return
        if isinstance(values,dict): values=[values.get(str(s),values.get(s)) for s in range(101,106 if family=='4A' else 104)]
        values=list(values or [None]*(5 if family=='4A' else 3))
        stat=stat or {}; known=[v for v in values if isinstance(v,(int,float)) and not isinstance(v,bool)]
        records.append(dict(name=name,kind=kind,values=values,mean=stat.get('mean',statistics.mean(known) if known else None),
            sd=stat.get('sd',statistics.stdev(known) if len(known)>1 else None),
            lower=stat.get('lower'),upper=stat.get('upper'),positive=sum(v<0 if family=='PD' else v>0 for v in known) if known else None,
            n=stat.get('n',len(known)),bar=bar,method=method))
    def close(stat,bar,upper=False):
        v=(stat or {}).get('upper' if upper else 'lower')
        return v is None or (v>=bar-.5 if upper else v<=bar+.5)
    def pairs(rows):return {str(x['seed']):x.get('change') for x in rows if 'seed' in x}
    if r.get('_reader_status'):return records
    if family=='M':
        summaries=r.get('summaries',[])
        def scores(arm,panel):return {str(x['seed']):x['accuracy'] for x in summaries if x['arm']==arm and x['panel']==panel and x['step']==40}
        for t in r.get('trajectories',[]):
            for cat in ('acquisition','retention','sequence_acquisition'):
                for endpoint,v in t.get(cat,{}).items():
                    name=t['trajectory']+'.'+cat+'.'+endpoint
                    source=v.get('from_incoming',v); stat=source.get('seed_bound',{})
                    add(name,pairs(source.get('seeds',[])),stat,bar='mean>=6;lo>0;pos=3' if endpoint=='math_hard' and cat=='acquisition' else str(v.get('margin',0)),
                        method='seed-t REFERENCE-SCORED' if v.get('method') == 'REFERENCE-SCORED' else 'seed-t')
                    if endpoint=='contract_nli' and cat=='acquisition':
                        # Final legal accuracy is a separate absolute guard.
                        arm='legal-plain' if t['trajectory']=='plain' else 'legal-'+str(r.get('admission',{}).get('configuration'))
                        q=v.get('final_seed_bound',{})
                        add(name+'.final',scores(arm,'test'),q,'G','69.7',near=close(q,69.7))
                    if 'standalone_acquisition' in v:
                        q=v['standalone_acquisition'];add(name+'.standalone',pairs(q.get('seeds',[])),q.get('seed_bound'),bar='mean>=6;lo>0;pos=3')
        for c in r.get('admission',{}).get('comparisons',[]):
            primary='math' if c['candidate']=='broad' else 'survival'
            for endpoint,q in c.get('endpoints',{}).items():
                if endpoint not in (primary,'math','survival','hellaswag','gain','contract_nli'):continue
                kind='P' if endpoint==primary else 'G'
                if kind=='G' and not close(q,-1):continue
                key={'gain':'gain_question_pairs','gain_auc':'auc_question_pairs','survival':'survival_question_pairs'}.get(endpoint,'question_pairs')
                rows=[x for x in c.get(key,[]) if key!='question_pairs' or x.get('panel')==endpoint]
                add(c['candidate']+'-'+c['control']+'.'+endpoint,pairs(rows),q,kind,'mean>=2;lo>0' if kind=='P' else '-1')
    elif family=='X':
        tables=r.get('tables',{})
        comparisons=[(k,v,tables.get('per_seed',{}),k) for k,v in r.get('eligible_comparisons',{}).items()]
        for k,v in tables.get('tier2',{}).get('comparisons',{}).items():
            seed_tables=tables.get('tier2',{}).get('details',{}).get(k,{}).get('per_seed',{})
            challenger=next((x for x in seed_tables if x!='control'),'drift')
            comparisons.append(('tier2.'+k,v,seed_tables,challenger))
        for name,c,seed_tables,arm in comparisons:
            a={str(x['seed']):x for x in seed_tables.get(arm,{}).get('seeds',[])}
            b={str(x['seed']):x for x in seed_tables.get('control',{}).get('seeds',[])}
            primary=c.get('primary')
            for endpoint,q in c.get('endpoints',{}).items():
                if endpoint!=primary and not close(q,-1):continue
                vals={}
                for seed in a.keys() & b.keys():
                    x,y=a[seed],b[seed]
                    left=x.get(endpoint,x.get('panel_final',{}).get(endpoint));right=y.get(endpoint,y.get('panel_final',{}).get(endpoint))
                    if left is not None and right is not None:vals[seed]=left-right
                add(name+'.'+endpoint,vals,q,'P' if endpoint==primary else 'G','mean>=2;lo>0' if endpoint==primary else '-1',
                    method='seed-t REFERENCE-SCORED' if c.get('method') == 'REFERENCE-SCORED' and endpoint in ('knowledge', 'hellaswag', 'protected_mean') else 'seed-t')
            # The admission rule counts positive absolute acquisition, not
            # positive primary contrasts. Keep this distinct input explicitly.
            vals={s:x['final']-x['incoming'] for s,x in a.items()}
            add(name+'.acquisition-positive',vals,kind='I',bar='pos=3',method='per-seed')
            for endpoint,v in c.get('inherited',{}).items():
                q=v.get('interval',{});bar=v['margin']
                if not v.get('admission_gate',True) or not close(q,bar):continue
                vals={s:(x['final'] if endpoint=='contract_nli_final' else x['final']-x['incoming'] if endpoint=='contract_nli_gain' else x.get('inherited',{}).get(endpoint)) for s,x in a.items()}
                add(name+'.'+endpoint,vals,q,'G',str(bar))
    elif family=='S':
        for name in ('drift_association','drift_forward','plasticity'):
            v=r[name];q=v.get('result',v.get('primary',v))
            add(name,q.get('per_seed'),q,bar='mean>=0.5;lo>0;pos=3',method='bootstrap97.5')
            d=v.get('difficulty',{})
            if name!='plasticity' and (q.get('mean') is None or d.get('mean') is None or q['mean']-d['mean']<=.5):
                add(name+'.difficulty',d.get('per_seed'),d,'G','mean<signal',method='bootstrap97.5')
    elif family=='KCR':
        for name,c in r.get('contrasts',{}).items():
            if name not in ('K-C0','CR-G'):continue
            q=dict(c.get('primary') or {})
            if q.get('mean') is not None and q.get('lower') is not None:q.setdefault('upper',2*q['mean']-q['lower'])
            add(name+'.primary',q.get('values'),q,bar='mean>=2;lo>0;pos=3',method='seed-t-two-sided95 REFERENCE-SCORED' if name == 'CR-G' and c.get('method') == 'REFERENCE-SCORED' else 'seed-t-two-sided95')
            for endpoint,q in c.get('guards',{}).items():
                add(name+'.'+endpoint,q.get('values'),q,'G','-1',method='seed-t-two-sided95 REFERENCE-SCORED' if c.get('method') == 'REFERENCE-SCORED' and endpoint.rsplit('/', 1)[-1] in ('knowledge', 'hellaswag') else 'seed-t-two-sided95',near=close(q,-1) or endpoint in c.get('failed_guards',[]))
    elif family=='PD':
        q=r.get('primary',{});add('stage2-base',r.get('seed_differences'),q,bar='mean<=-2;hi<0;negative=3',method='seed-t-two-sided95')
        vals={s:x.get('base',{}).get('final') for s,x in r.get('curves',{}).items()}
        # This guard is applied per seed, not to a t interval.
        if not r.get('sensitivity') or any(v is None or v<=10.5 for v in vals.values()):
            add('base-final-sensitivity',vals,kind='G',bar='each>=10',method='per-seed')
    elif family=='SP':
        for name,f in r.get('families',{}).items():
            stats_to_print=list(f.get('contrasts',{}).items())
            for comparison,stats in f.get('intervals',{}).items():
                stats_to_print.append((comparison,{'template_reliance_gap':stats.get('template_reliance_gap',{}),**{'change-'+k:v for k,v in stats.get('change_from_incoming',{}).items()}}))
            for comparison,stats in stats_to_print:
                for metric,q in stats.items():
                    add(name+'.'+comparison+'.'+metric,q.get('per_seed',q.get('values')),q,bar='DESCRIPTIVE',method='descriptive-seed-t')
    elif family=='4A':
        q=r.get('primary',{});add('consolidation-memory',q.get('per_seed'),q.get('interval'),bar='mean>=5;lo>0;pos>=4;months>=3',method='seed-t-n5')
        add('positive-months',q.get('monthly'),kind='I',bar='positive-months>=3',method='months2-6')
        for q in r.get('retention',[]):
            if q.get('arm')!='consolidation' or not q.get('is_registered_bar') or not close(q['interval'],q['margin'],True):continue
            matching=[x for x in r.get('transitions',[]) if x.get('comparison')=='retention' and all(x.get(k)==q.get(k) for k in ('arm','month','panel','mode','reference'))]
            vals={str(x['seed']):-x['delta_points'] for x in matching}
            name='.'.join(str(q[k]) for k in ('arm','month','panel','mode','reference'))
            add(name,vals,q['interval'],'G',str(q['margin']),method='seed-t-upper-n5')
    return records


def erow(family,record,names,mapping):
    r=rounded(record)
    fields=[family,'.'.join(safe_key(k,names,mapping) for k in r['name'].split('.')),r['kind'],canonical(r['values'])]
    fields += [canonical(r[k]) for k in ('mean','sd','lower','upper','positive','n')]
    bar=r['bar']
    if re.fullmatch(r'-?[0-9]+(?:\.[0-9]+)?',bar):bar=canonical(rounded(float(bar)))
    elif bar not in ('-','mean>=6;lo>0;pos=3','mean>=2;lo>0','mean>=0.5;lo>0;pos=3','mean>=2;lo>0;pos=3','mean<=-2;hi<0;negative=3','each>=10','DESCRIPTIVE','mean>=5;lo>0;pos>=4;months>=3','positive-months>=3','mean<signal','pos=3'):bar='UNVERIFIED'
    fields += [bar,r['method'].replace(' ',':')]
    return 'E|'+'|'.join(fields)


def operation_rows(full,public,names,mapping):
    """Public measurements and coded warnings; never echo free-form reasons."""
    rows=['T|JOB-INVENTORY|'+job_inventory()[1], 'T|JOBS|family|public-job-index-or-u-local-ordinal|completion']+full.get('jobs',[])
    rows += ['T|COSTS|recorded-process-GPU-hours-only|missing-allocation-or-idle-is-unknown',
        'T|TIER-ALLOWANCE|1800|KCR<=1400|PD<=264|SP<=216|warning-only']
    rows += list(metric_rows('ACCOUNTING',rounded(public['ACCOUNTING'])))
    for family,results in sorted(full['readers'].items()):
        for i,r in enumerate(results):
            key=family if len(results)==1 else family+str(i+1)
            timing={k:r[k] for k in ('wall_time','accounting','timing_status','wall_time_warnings','warning_summary','warning_count') if k in r}
            secs=[x.get('seconds') for x in r.get('cost',{}).get('records',[]) if isinstance(x,dict)]
            secs=[v for v in secs if isinstance(v,(int,float)) and math.isfinite(v) and v>=0]
            if secs:timing['process_wall_seconds']={'sum':sum(secs),'max':max(secs),'measurements':len(secs)}
            rows += list(metric_rows(key,rounded(project(timing,names,mapping,key))))
            warnings={}
            def walk(v):
                if isinstance(v,dict):
                    state=v.get('status',v.get('_reader_status',''))
                    if isinstance(state,str) and any(w in state for w in ('PENDING','INCOMPLETE','UNVERIFIED','DEFERRED')):
                        code=safe_key(state,names,mapping);warnings[code]=warnings.get(code,0)+1
                    for k,x in v.items():
                        if k in ('reason','error','_warning') and x:
                            text=str(x)
                            code=re.sub(r'[^A-Za-z0-9_.<>=-]+','-',text).strip('-') if text in names and re.fullmatch(r'[A-Za-z0-9 ,;:_.()<>==-]{1,160}',text) else safe_key(text,names,mapping)
                            warnings[code]=warnings.get(code,0)+1
                        # Avoid traversing private metadata and potentially huge
                        # raw rows; all warning codes must be registered/aliased.
                        elif not PRIVATE.search(str(k)) and k not in OMIT:walk(x)
                elif isinstance(v,list):
                    for x in v:walk(x)
            walk(r)
            for code,count in sorted(warnings.items()):rows.append(f'W|{key}|{code}|{count}')
    return rows+['T|WARNINGS|PARTNER-COMPUTED|independent-reverification=UNVERIFIED',
        'T|WARNINGS|opaque-warning-codes-have-details-retained-on-node',
        'T|CHECKSUM-SCOPE|transcription-and-truncation-only|no-authentication']


def write_parts(out, full, tag, commit, jobs=(), max_chars=MAX_CHARS, *, full_detail=False):
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', tag) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('kit tag and full 40-character commit required')
    if not 1000 <= max_chars <= MAX_CHARS: raise ValueError('part limit must be 1000..12000')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if list(out.glob('RETURN-TEXT-*.md')) or list((out/'full').glob('RETURN-TEXT-*.md')): raise ValueError('use a fresh text output directory')
    names=vocabulary();mapping=alias_context();public={};verdicts=[];endpoint_by_key={};operations_by_key={}
    full=dict(full,readers={f:list(full['readers'].get(f) or [dict(_reader_status='DEFERRED' if f=='4A' else 'PENDING',_warning='NOT_CONFIGURED')]) for f in FAMILIES})
    serializable=dict(full,readers={f:list(rs) for f,rs in full['readers'].items()})
    def numeric_fields(values):
        if any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)) for v in values):
            raise ValueError('non-numeric presentation field')
    def render(family,r,key):
        canonical(r)
        ds=sorted(decisions(family,r),key=lambda d:d['comparison'])
        if {(family,d['comparison']) for d in ds}!=registered_decisions([family]):
            raise ValueError('decision presentation differs from registered scope')
        for d in ds:
            d['family']=key
            d['comparison']=safe_key(d['comparison'],names,mapping)
            d['guards']=[safe_key(g,names|{'BAR_NOT_PASS','NOT_ADMITTED','SIGNAL_NOT_PASS','SENSITIVITY_NOT_SHOWN','DESCRIPTIVE_ONLY','SEE_REGISTERED_BARS','UNBOUND_COMPARISON','NOT_ELIGIBLE','INCOMPLETE_EVIDENCE','PRIMARY_MEAN','PRIMARY_INTERVAL','ACQUISITION_SEEDS','NATIVE_EVIDENCE_UNAVAILABLE'},mapping) for g in d['guards']]
            if len(d['guards'])>4:d['guards']=d['guards'][:4]+['plus-'+str(len(d['guards'])-4)+'-guards-in-metrics']
            numeric_fields(d[k] for k in ('estimate','lower','upper','seeds_positive'))
            drow(rounded(d))
        projected=project(r,names,mapping,key)
        list(metric_rows(key,rounded(projected)))
        endpoint_records=essential_endpoints(family,r)
        for record in endpoint_records:numeric_fields([*record['values'],*[record[k] for k in ('mean','sd','lower','upper','positive','n')]])
        endpoints=[erow(key,x,names,mapping) for x in endpoint_records]
        operations=operation_rows(dict(readers={key:[r]}),{'ACCOUNTING':{}},names,mapping)
        operations=[row for row in operations if row.startswith(('R|'+key+'|','W|'+key+'|'))]
        return ds,projected,endpoints,operations
    for family, results in sorted(full['readers'].items()):
        for i, r in enumerate(results):
            key = family if len(results) == 1 else family + str(i + 1)
            try:ds,projected,endpoints,operations=render(family,r,key)
            except Exception as exc:
                stub=dict(_reader_status='UNVERIFIED',_warning='READER_ERROR')
                try:canonical(r)
                except (ValueError,TypeError):serializable['readers'][family][i]=dict(stub,_local_exception=type(exc).__name__+': '+str(exc))
                r=stub
                ds=[dict(d,family=key) for d in _stub_decisions(family,stub)]
                projected={'status':'UNVERIFIED','warning_count':1};endpoints=[]
                operations=[f'W|{key}|READER_ERROR|1']
            verdicts.extend(ds);public[key]=projected;endpoint_by_key[key]=endpoints;operations_by_key[key]=operations
    public['ACCOUNTING'] = project(full.get('accounting', {}), names, mapping, 'ACCOUNTING')
    digest = sha(canonical(serializable))
    first = ['T|PARTNER-COMPUTED|independent-reverification=UNVERIFIED',
             'T|TECHNICAL-STANDIN' if full.get('technical_synthetic') else 'T|NODE-READERS',
             'RESULTS-SHA256 ' + digest,
             'T|DECISIONS|family|comparison|verdict|estimate|lower|upper|seeds-in-registered-direction|guard-failures']
    first += ['T|SCOPE|'+','.join(sorted(k for k in public if k!='ACCOUNTING'))]
    first += [drow(rounded(d)) for d in verdicts]
    first += ['T|JOB-INVENTORY|'+job_inventory()[1], 'T|JOBS|family|public-job-index-or-u-local-ordinal|completion'] + list(jobs or full.get('jobs',[]))
    details = ['T|METRICS|family|comparison|seed|metric-pointer=value']
    for family, r in public.items(): details.extend(metric_rows(family, rounded(r)))
    details += ['T|COSTS|recorded-process-GPU-hours-only|missing-allocation-or-idle-is-unknown',
                'T|TIER-ALLOWANCE|1800|KCR<=1400|PD<=264|SP<=216|warning-only',
                'T|WARNINGS|wall-time-over-24h-and-unmeasured-timings-in-warning_summary',
                'T|WARNINGS|PARTNER-COMPUTED|independent-reverification=UNVERIFIED',
                'T|CHECKSUM-SCOPE|transcription-and-truncation-only|no-authentication']
    for family in sorted(public):
        count=sum(v for k,v in mapping['dropped_unknown_numeric'].items() if k.split('/')[0]==family)
        if count:details.append(f'W|{family}|UNKNOWN_NUMERIC_DROPPED|{count}')
    def split(rows,limit):
        chunks=[[]];size=0
        for row in rows:
            if len(body_text([row]))>limit:raise ValueError('metric record exceeds part limit')
            if size+len(body_text([row]))>limit:chunks.append([]);size=0
            chunks[-1].append(row);size+=len(body_text([row]))
        return chunks
    # Bind the ordered FULL content (not framing/set-reference records). This
    # avoids a self-referential digest and permits requested parts in any order.
    limit=max_chars-500
    if len(body_text(first))>limit:raise ValueError('all decisions/jobs exceed Part 1 capacity; never omit verdicts')
    full_chunks=[first]+split(details,limit)
    content_hashes=[sha(canonical(chunk)) for chunk in full_chunks]
    full_digest=sha(canonical(content_hashes));full_count=len(full_chunks)
    reference=f'T|FULL-SET|{full_count}|{full_digest}'
    def emit(directory,chunks,mode):
        directory.mkdir(parents=True,exist_ok=True);paths=[]
        for k,rows in enumerate(chunks,1):
            counts='T|COUNTS|'+','.join(key+'='+str(value) for key,value in section_counts(rows).items())
            body=body_text([f'T|KIT|{tag}|{commit}',f'T|PART|{k}|{len(chunks)}','T|MODE|'+mode,reference,counts]+rows)
            text=body_text([f'KIT {tag} {commit} PART {k}/{len(chunks)}'])+body+'SHA256 '+sha(body)+'\n'
            if not text.isascii() or '\t' in text or len(text)>max_chars:raise ValueError('part capacity exceeded')
            checked=parse_parts([text],part1_only=True,expect_commit=commit,_single_part=True)
            if any(value!='MATCH' for value in checked['part_checksums'].values()):raise ValueError('internal text checksum round-trip differs')
            path=directory/f'RETURN-TEXT-{k}-of-{len(chunks)}.md';path.write_text(text,encoding='ascii');paths.append(path)
        return paths
    # Always retain FULL on the node, with stable part numbers for requests.
    full_paths=emit(out/'full',full_chunks,'FULL')
    essential_first=first[:first.index('T|JOB-INVENTORY|'+job_inventory()[1])]
    endpoint_rows=['T|ENDPOINTS|family|endpoint|P-primary-G-guard-I-rule-input|seeds101-103|mean|sample-SD|lower|upper|seeds-in-registered-direction|n|bar|method',
        'T|ENDPOINTS|4A-keeps-five-seeds101-105|months2-6-kept-separately|KCR-upper-symmetric-derived']
    for rows in endpoint_by_key.values():endpoint_rows.extend(rows)
    ops=operation_rows(dict(readers={},jobs=full.get('jobs',[])),public,names,mapping)
    footer=ops[-3:];ops=ops[:-3]
    for rows in operations_by_key.values():ops.extend(rows)
    drops={}
    for location,count in mapping['dropped_unknown_numeric'].items():
        family=location.split('/')[0];drops[family]=drops.get(family,0)+count
    drop_rows=[f'W|{family}|UNKNOWN_NUMERIC_DROPPED|{count}' for family,count in sorted(drops.items())]
    # The FULL set uses the same drop counters in its detailed warning page.
    ops.extend(drop_rows);ops.extend(footer)
    # Fixed semantic pages 1/2/3; page 4 carries overflow from 2 and/or 3.
    endpoint_chunks=split(endpoint_rows,limit);ops_chunks=split(ops,limit)
    overflow=[row for chunk in endpoint_chunks[1:]+ops_chunks[1:] for row in chunk]
    essential_chunks=[essential_first,endpoint_chunks[0],ops_chunks[0]]
    if overflow:essential_chunks.append(overflow)
    oversized=bool(overflow) and len(body_text(overflow))>limit
    paths=[] if oversized else emit(out,essential_chunks,'ESSENTIAL')
    (out/'full-results.local.json').write_text(canonical(serializable)+'\n',encoding='ascii')
    (out/'metric-map.local.json').write_text(canonical(mapping)+'\n',encoding='ascii')
    (out/'full-set.local.json').write_text(canonical(dict(parts=full_count,sha256=full_digest,content_hashes=content_hashes))+'\n',encoding='ascii')
    if oversized and not full_detail:raise ValueError('ESSENTIAL exceeds four-part capacity; no evidence omitted; FULL retained on node')
    return full_paths if full_detail else paths


def numeric(value, *, integer=False):
    if value is None:return
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('invalid numeric field')
    if integer and (value<0 or value!=int(value)):raise ValueError('invalid numeric count')


def without_transcription(record):
    return {k:v for k,v in record.items() if k!='transcription_verification'}


def merge_record(first,second,kind):
    if without_transcription(first)!=without_transcription(second):raise ValueError('supplement '+kind+' conflict')
    if 'TRANSCRIPTION-UNVERIFIED' in (first['transcription_verification'],second['transcription_verification']):return dict(first,transcription_verification='TRANSCRIPTION-UNVERIFIED')
    return first


def parse_parts(texts, part1_only=False, base=None, *, expect_commit=None, _single_part=False):
    """Verify complete ESSENTIAL and/or FULL, allowing arbitrary FULL supplements.

    A base is a previously verified local reading, never partner-supplied JSON.
    Supplements retain the same transcription-only trust level as every part.
    """
    if expect_commit is not None and not re.fullmatch('[0-9a-f]{40}',expect_commit):raise ValueError('expected full 40-character commit required')
    sets={};counts={};identity=None;reference=None;checksum_status={}
    for text in texts:
        text=re.sub('[\u200b-\u200f\u2060\ufeff]','',text)
        text=text.translate(str.maketrans({'\u201c':'"','\u201d':'"','\u2018':"'",'\u2019':"'",'\u00a0':' '})).replace('\r','')
        text=re.sub(r'(?m)^[ \t]*(?:>[ \t]+)+','',text)
        text=re.sub(r'\s+','',text)
        header=r'KIT([A-Za-z0-9_.-]{1,64}?)([0-9a-f]{40})PART(\d+)/(\d+)'
        frames=list(re.finditer(header+r'(.*?)(?='+header+r'|$)',text,re.S))
        if not frames:
            hint=re.search(r'PART(\d+)/(\d+)',text)
            raise ValueError(('part '+hint[1]+': ' if hint else '')+'missing or truncated KIT frame')
        remainder=text
        for frame in reversed(frames):remainder=remainder[:frame.start()]+remainder[frame.end():]
        if remainder:raise ValueError('unframed pasted text')
        for frame in frames:
            tag,commit,k,n,raw=frame.groups()[:5];header_numbers=(k,n);k=int(k);n=int(n)
            footer=re.search(r'(?<!RESULTS-)SHA256(.*)$',raw)
            digest=None
            if footer:
                if not re.fullmatch('[0-9a-f]*',footer[1]):raise ValueError(f'part {k}: unframed text after checksum')
                digest=footer[1];raw=raw[:footer.start()]
            if not 1<=k<=n:raise ValueError('invalid part number')
            own=(tag,commit)
            if identity is not None and identity!=own:raise ValueError('mixed kit identities')
            identity=own
            prefix=r'(?:RESULTS-SHA256|(?:D|R|E|W|J)\|(?:M|X|S|KCR|PD|SP|4A|ACCOUNTING|core|preflight)[0-9]*\||T\|(?:KIT|PART|MODE|FULL-SET|COUNTS|SCOPE|ENDPOINTS|PARTNER-COMPUTED|TECHNICAL-STANDIN|NODE-READERS|DECISIONS|JOB-INVENTORY|JOBS|METRICS|COSTS|TIER-ALLOWANCE|WARNINGS|CHECKSUM-SCOPE))'
            rows=[v for v in re.split('(?='+prefix+')',raw) if v];restored=[]
            for row in rows:
                if row.startswith('RESULTS-SHA256'):row='RESULTS-SHA256 '+row[len('RESULTS-SHA256'):]
                if row.startswith('D|'):
                    fields=row.split('|')
                    if len(fields)!=9:raise ValueError(f'part {k}: invalid decision record')
                    if fields[3]=='NOTMET':fields[3]='NOT MET'
                    row='|'.join(fields)
                restored.append(row)
            body=body_text(restored)
            try:matches=sha(body)==digest
            except UnicodeEncodeError:matches=False
            # V5: a checksum is advisory; structural validation remains mandatory.
            kits=[row for row in restored if row.startswith('T|KIT|')]
            numbering=[row for row in restored if row.startswith('T|PART|')]
            if kits!=[f'T|KIT|{tag}|{commit}'] or numbering!=['T|PART|'+'|'.join(header_numbers)]:raise ValueError(f'part {k}: header/body identity mismatch')
            if expect_commit is not None and commit!=expect_commit:raise ValueError(f'part {k}: unexpected kit commit')
            modes=[row.split('|')[2] for row in restored if row.startswith('T|MODE|')]
            if len(modes)!=1 or modes[0] not in ('ESSENTIAL','FULL'):raise ValueError('missing or invalid mode')
            mode=modes[0]
            if mode=='ESSENTIAL' and n>4:raise ValueError('ESSENTIAL exceeds four parts')
            refs=[row.split('|')[2:] for row in restored if row.startswith('T|FULL-SET|')]
            if modes and len(refs)!=1:raise ValueError('missing FULL set reference')
            if refs:
                if len(refs[0])!=2 or not refs[0][0].isdigit() or not re.fullmatch('[0-9a-f]{64}',refs[0][1]):raise ValueError('invalid FULL set reference')
                ref=(int(refs[0][0]),refs[0][1])
                if reference is not None and reference!=ref:raise ValueError('mixed FULL set digests or counts')
                reference=ref
                if mode=='FULL' and n!=ref[0]:raise ValueError('FULL part count differs from reference')
            if mode in counts and counts[mode]!=n:raise ValueError('mixed kit part counts')
            counts[mode]=n;parts=sets.setdefault(mode,{})
            if k in parts:raise ValueError(f'duplicate part {k} ({mode})')
            counter=[row for row in restored if row.startswith('T|COUNTS|')]
            if len(counter)!=1:raise ValueError(f'part {k}: missing section row counts')
            if counter:
                payload=[row for row in restored if not row.startswith(('T|KIT|','T|PART|','T|MODE|','T|FULL-SET|','T|COUNTS|'))]
                wanted='T|COUNTS|'+','.join(key+'='+str(value) for key,value in section_counts(payload).items())
                if counter!=[wanted]:raise ValueError(f'part {k}: section row counts differ')
            parts[k]=restored
            checksum_status[mode+':'+str(k)]='MATCH' if matches else 'TRANSCRIPTION-UNVERIFIED'
    if base is not None:
        if base.get('provenance')!='PARTNER-COMPUTED' or base.get('independent_reverification')!='UNVERIFIED':raise ValueError('invalid local base reading')
        if identity!=(base['kit_tag'],base['kit_commit']):raise ValueError('mixed kit identities')
        if reference!=(base.get('full_part_count'),base.get('full_set_sha256')):raise ValueError('mixed FULL set digest with base')
    main='ESSENTIAL' if 'ESSENTIAL' in sets else 'FULL'
    supplement_only=base is not None and set(sets)=={'FULL'}
    parts=sets[main];expected=counts[main]
    missing=sorted(set(range(1,expected+1))-parts.keys())
    if _single_part and sum(len(p) for p in sets.values())!=1:raise ValueError('self-check requires one part')
    if not supplement_only and not _single_part:
        if 1 not in parts:raise ValueError('missing part 1')
        if missing and not(part1_only and set(parts)=={1}):raise ValueError('missing parts: '+','.join(map(str,missing)))
    verdicts=[];metrics={};jobs=[];endpoints={};warnings=[];results_hash=None;inventory_hash=None
    seen_metric=set();seen_job=set();seen_warning=set();seen_decisions={};scopes={}
    for mode in ('ESSENTIAL','FULL'):
        for k,rows in sorted(sets.get(mode,{}).items()):
            transcription=checksum_status[mode+':'+str(k)]
            def annotate(record):
                return dict(record,transcription_verification=transcription)
            for row in rows:
                f=row.split('|')
                if row.startswith('RESULTS-SHA256 '):
                    if k!=1:raise ValueError('results digest outside part 1')
                    h=row.split()[1]
                    if not re.fullmatch('[0-9a-f]{64}',h):raise ValueError('invalid RESULTS digest')
                    if results_hash is not None and h!=results_hash:raise ValueError('mixed RESULTS digests')
                    results_hash=h
                elif f[:2]==['T','SCOPE']:
                    if k!=1 or mode in scopes or len(f)!=3:raise ValueError('invalid decision scope')
                    scope=f[2].split(',')
                    if len(set(scope))!=len(scope) or {re.sub(r'[0-9]+$','',x) if x!='4A' else x for x in scope}!=set(FAMILIES):raise ValueError('incomplete decision family scope')
                    scopes[mode]=scope
                elif f[0]=='D':
                    if k!=1:raise ValueError('decision outside part 1')
                    if len(f)!=9:raise ValueError('invalid decision record')
                    if f[3] not in LABELS:raise ValueError('unknown verdict')
                    d=dict(family=f[1],comparison=f[2],verdict=f[3],estimate=json.loads(f[4]),lower=json.loads(f[5]),upper=json.loads(f[6]),seeds_in_direction=json.loads(f[7]),seed_direction='negative' if f[1].startswith('PD') else 'positive',guards=[] if f[8]=='-' else f[8].split(','),provenance='PARTNER-COMPUTED',independent_reverification='UNVERIFIED')
                    for name in ('estimate','lower','upper'):numeric(d[name])
                    numeric(d['seeds_in_direction'],integer=True)
                    if d['seeds_in_direction'] is not None and d['seeds_in_direction']>(5 if f[1].startswith('4A') else 3):raise ValueError('invalid numeric seed count')
                    if d['lower'] is not None and d['upper'] is not None and d['lower']>d['upper']:raise ValueError('invalid interval ordering')
                    decision_key=(f[1],f[2])
                    if decision_key in seen_decisions.setdefault(mode,set()):raise ValueError('duplicate decision line')
                    seen_decisions[mode].add(decision_key)
                    d=annotate(d)
                    if mode==main:verdicts.append(d)
                    elif not supplement_only and without_transcription(d) not in [without_transcription(v) for v in verdicts]:raise ValueError('FULL decisions differ from ESSENTIAL')
                elif f[0]=='R':
                    if len(f)==5 and '=' in f[4]:
                        pointer,raw_value=f[4].rsplit('=',1)
                        kind=None
                        if re.match(r'^[nbzs]:',raw_value):kind,raw_value=raw_value.split(':',1)
                        value=json.loads(raw_value)
                        if isinstance(value,(dict,list)):raise ValueError('metric must be scalar')
                        if kind=='n':numeric(value)
                        if kind=='n' and value is None or kind=='b' and not isinstance(value,bool) or kind=='z' and value is not None or kind=='s' and not isinstance(value,str):raise ValueError('invalid metric scalar type')
                        decoded=[(pointer,value,f[3])]
                    elif len(f)==6:decoded=metric_leaves(json.loads(f[5]),f[4],f[3])
                    else:raise ValueError('invalid metric record')
                    if f[3] not in ('all','101','102','103','104','105'):raise ValueError('invalid metric seed')
                    for pointer,value,seed in decoded:
                        if isinstance(value,(int,float)) and not isinstance(value,bool):numeric(value)
                        key=f[1]+pointer
                        if (mode,key) in seen_metric:raise ValueError('duplicate metric')
                        seen_metric.add((mode,key))
                        record=dict(family=f[1],comparison=f[2],seed=seed,metric=pointer,value=value,provenance='PARTNER-COMPUTED',independent_reverification='UNVERIFIED')
                        record=annotate(record)
                        if key in metrics:record=merge_record(metrics[key],record,'metric')
                        metrics[key]=record
                elif f[0]=='E':
                    if len(f)!=13 or mode!='ESSENTIAL':raise ValueError('invalid essential endpoint')
                    key=f[1]+':'+f[2]
                    if key in endpoints:raise ValueError('duplicate essential endpoint')
                    values=json.loads(f[4]);method=f[12]
                    endpoints[key]=dict(family=f[1],endpoint=f[2],kind=f[3],values=values,
                        seeds=list(range(101,101+len(values))) if method!='months2-6' else None,
                        mean=json.loads(f[5]),sd=json.loads(f[6]),lower=json.loads(f[7]),upper=json.loads(f[8]),seeds_in_direction=json.loads(f[9]),seed_direction='negative' if f[1].startswith('PD') else 'positive',n=json.loads(f[10]),bar=f[11],method=method,
                        provenance='PARTNER-COMPUTED',independent_reverification='UNVERIFIED')
                    if not isinstance(values,list):raise ValueError('invalid endpoint values')
                    for v in values:numeric(v)
                    for name in ('mean','sd','lower','upper'):numeric(endpoints[key][name])
                    for name in ('seeds_in_direction','n'):numeric(endpoints[key][name],integer=True)
                    endpoints[key]=annotate(endpoints[key])
                elif f[0]=='W':
                    if len(f)!=4:raise ValueError('invalid warning record')
                    numeric(int(f[3]),integer=True)
                    if row not in seen_warning:
                        warnings.append(annotate(dict(family=f[1],code=f[2],count=int(f[3]),provenance='PARTNER-COMPUTED',independent_reverification='UNVERIFIED')));seen_warning.add(row)
                elif f[:2]==['T','JOB-INVENTORY']:
                    if inventory_hash is not None and inventory_hash!=f[2]:raise ValueError('mixed job inventories')
                    inventory_hash=f[2]
                elif f[0]=='J':
                    if len(f)!=4 or f[3] not in ('DONE','FAILED','PENDING'):raise ValueError('invalid job record')
                    if row not in seen_job:
                        jobs.append(annotate(dict(family=f[1],job=f[2],status=f[3],provenance='PARTNER-COMPUTED',independent_reverification='UNVERIFIED')));seen_job.add(row)
                elif f[0]!='T' or f[1] not in {'KIT','PART','MODE','FULL-SET','COUNTS','ENDPOINTS','PARTNER-COMPUTED','TECHNICAL-STANDIN','NODE-READERS','DECISIONS','JOBS','METRICS','COSTS','TIER-ALLOWANCE','WARNINGS','CHECKSUM-SCOPE'}:
                    raise ValueError(f'part {k}: unknown or damaged record')
    for mode,scope in scopes.items():
        if seen_decisions.get(mode,set())!=registered_decisions(scope):raise ValueError('missing or unexpected registered decision line')
    for mode in sets:
        if 1 in sets[mode] and mode not in scopes and any(row.startswith('T|MODE|') for row in sets[mode][1]):raise ValueError('missing decision scope')
    labels,local_hash=job_inventory()
    for job in jobs:
        resolved=[]
        if inventory_hash==local_hash:
            for group in job['job'].split(','):
                if group.startswith('u') or group=='unknown':continue
                if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?',group):raise ValueError('invalid registered job range')
                bounds=list(map(int,group.split('-')));start=bounds[0];end=bounds[-1]
                if not 1<=start<=end<=len(labels):raise ValueError('registered job range outside inventory')
                resolved.extend(labels[i-1] for i in range(start,end+1))
        job.update(registered_tasks=resolved,inventory_matches=inventory_hash==local_hash)
    if not supplement_only and (not _single_part or 1 in parts) and (results_hash is None or not verdicts):raise ValueError('part 1 lacks decisions/results digest')
    full_parts=sets.get('FULL',{})
    if reference and len(full_parts)==reference[0]:
        content=[[row for row in full_parts[k] if not row.startswith(('T|KIT|','T|PART|','T|MODE|','T|FULL-SET|','T|COUNTS|'))] for k in range(1,reference[0]+1)]
        if sha(canonical([sha(canonical(chunk)) for chunk in content]))!=reference[1]:
            checksum_status['FULL-SET']='TRANSCRIPTION-UNVERIFIED'
    result=dict(schema=SCHEMA,kit_tag=identity[0],kit_commit=identity[1],mode=main,complete=not missing,missing_parts=missing,results_sha256=results_hash,job_inventory_sha256=inventory_hash,decisions=verdicts,metrics=metrics,jobs=jobs,endpoints=endpoints,warnings=warnings,
        full_part_count=reference[0] if reference else counts.get('FULL'),full_set_sha256=reference[1] if reference else None,
        received_full_parts=sorted(full_parts),part_checksums=checksum_status,
        # Retain verified part hashes to reject conflicting/repeated later text.
        full_part_hashes={str(k):sha('\n'.join(rows)+'\n') for k,rows in full_parts.items()},
        full_content_hashes={str(k):sha(canonical([row for row in rows if not row.startswith(('T|KIT|','T|PART|','T|MODE|','T|FULL-SET|','T|COUNTS|'))])) for k,rows in full_parts.items()},
        provenance='PARTNER-COMPUTED',independent_reverification='UNVERIFIED')
    if base is not None:
        if not supplement_only:raise ValueError('base accepts FULL supplements only')
        result=merge_supplement(base,result)
    result['detail_complete']=len(result['received_full_parts'])==result['full_part_count']
    if result['detail_complete'] and result['full_set_sha256']:
        hashes=[result['full_content_hashes'][str(k)] for k in range(1,result['full_part_count']+1)]
        if sha(canonical(hashes))!=result['full_set_sha256']:result['part_checksums']['FULL-SET']='TRANSCRIPTION-UNVERIFIED'
    result['missing_full_parts']=sorted(set(range(1,result['full_part_count']+1))-set(result['received_full_parts']))
    return result


def merge_supplement(base, supplement):
    """Merge verified later FULL pages into an owner-local ESSENTIAL reading."""
    result=dict(base)
    if supplement.get('results_sha256') and supplement['results_sha256']!=base['results_sha256']:raise ValueError('mixed RESULTS digests')
    if supplement['decisions'] and [without_transcription(v) for v in supplement['decisions']]!=[without_transcription(v) for v in base['decisions']]:raise ValueError('supplement decision conflict')
    hashes=dict(base.get('full_part_hashes',{}))
    for k,h in supplement['full_part_hashes'].items():
        if k in hashes:raise ValueError('duplicate FULL supplement part '+k)
        hashes[k]=h
    result['part_checksums']={**base.get('part_checksums',{}),**supplement.get('part_checksums',{})}
    result['full_content_hashes']={**base.get('full_content_hashes',{}),**supplement['full_content_hashes']}
    result['full_part_hashes']=hashes;result['received_full_parts']=sorted(map(int,hashes))
    metrics=dict(base['metrics'])
    for key,v in supplement['metrics'].items():
        metrics[key]=merge_record(metrics[key],v,'metric') if key in metrics else v
    result['metrics']=metrics
    result['warnings']=base['warnings']+[w for w in supplement['warnings'] if w not in base['warnings']]
    # FULL's job summary is identical to the ESSENTIAL summary, not extra jobs.
    if supplement['jobs'] and [without_transcription(v) for v in supplement['jobs']]!=[without_transcription(v) for v in base['jobs']]:raise ValueError('supplement job conflict')
    return result
