"""Next-message evidence adapter for the kit's KIT/T/D/E text transport.

Only registered numeric endpoints, coded states and hashed lane identities leave
the node. The CPU importer recomputes bars from exact integer correct-counts;
neither pasted bounds nor the partner's verdict decide the local verdict.
"""
import json
import math
from pathlib import Path
import re
import statistics
from kit.nm_common import FAMILIES,GENERAL,SEEDS,TWO,ONE,benefit,guard,stats,digest,need,ready,Pending,Invalid
from kit.text_return import canonical,sha,body_text,frame_text,section_counts


def encoded(value):
    return canonical(value).replace(' ',r'\u0020').replace('|',r'\u007c')


def trow(section,key,value):return 'T|NM|'+section+'|'+key+'|'+encoded(value)


def endpoint(family,name,kind,record,bar='DESCRIPTIVE'):
    fields=[family,name,kind,encoded(record['values'])]
    fields += [encoded(record[k]) for k in ('mean','sd','lower','upper','positive','n')]
    fields += [bar,'one' if record['t']==ONE else 'two']
    return 'E|'+'|'.join(fields)


def guard_names(family):
    if family=='grpo':return {'acquisition',*GENERAL,'hellaswag'}
    names={*GENERAL,'hellaswag'}
    if family=='v2b':return names|{'september-on-all52'}
    names|={'finqa-final','legal-final','finqa-retention','legal-gain'}
    return names|({'legal-gain-AUC'} if family!='opr' else set())


TEXT_FAMILIES=(*FAMILIES,'cr')


def payload(full,*,full_detail=False):
    if full.get('nm_family')=='cr':
        from kit.cr_text import payload as cr_payload
        return cr_payload(full,full_detail=full_detail)
    family=full['nm_family'];need(family in FAMILIES,'unknown NM family')
    r=full['reading'];authority=full.get('infrastructure_authority')
    need(isinstance(authority,str) and re.fullmatch('[0-9a-f]{64}',authority),'bound infrastructure authority required')
    technical=r.get('status')=='TECHNICAL-ONLY'
    status=r.get('simulated_status') if technical else r['status']
    rows=['T|AUTH|'+authority,trow('meta','family',family),
        trow('meta','seeds',list(SEEDS)),
        trow('meta','reported_status',status),trow('meta','technical',technical),
        trow('meta','benchmark',r.get('benchmark',False)),
        'T|PARTNER-COMPUTED|independent-reverification=UNVERIFIED',
        'RESULTS-SHA256 '+sha(canonical(r))]
    if family in ('opr','fisher') and r.get('benchmark'):rows.append(trow('meta','scope','NON-INCUMBENT BENCHMARK'))
    p=r.get('primary',{});key='NM-'+family
    rows.append('D|'+key+'|primary|'+status.replace(' ',':')+'|'+
        '|'.join(encoded(p.get(k)) for k in ('mean','lower','upper','positive'))+'|-')
    for seed,record in sorted(r.get('generation_qualification',{}).items()):
        rows.append(trow('utilisation','generation_qualification_s'+seed,record))
    for warning in r.get('utilisation_warnings',[]):rows.append('W|'+key+'|'+warning+'|1')
    for name in ('binding_sha256','spec_sha256'):
        if r.get(name):rows.append(trow('hash',name,r[name]))
    if p:
        need(r.get('correct_counts'),'computed endpoints require exact correct-counts')
        for cell,counts in sorted(r['correct_counts'].items()):rows.append(trow('count',cell,counts))
    diagnostics=r.get('diagnostics',{})
    if full_detail:
        if p:rows.append(endpoint(key,'primary','P',p,'LB>=-1' if family=='grpo' else 'mean>=2;LB>0;pos=3'))
        for name,v in sorted(r.get('guards',{}).items()):
            rows.append(endpoint(key,name,'G',v,'LB>0;pos=3' if name=='acquisition' else 'LB>='+str(v['margin'])))
        for arm,parts in sorted(r.get('package_summaries',{}).items()):
            for name,v in sorted(parts.items()):rows.append(endpoint(key,'package.'+arm+'.'+name,'S',v))
        for name in ('D_sensitivity','general_retention','legal-gain-AUC'):
            if diagnostics.get(name):rows.append(endpoint(key,name,'I',diagnostics[name]))
        for group in ('FinQA_gain_AUC','prior_months','pooled_prior_months'):
            for name,v in sorted(diagnostics.get(group,{}).items()):rows.append(endpoint(key,group+'.'+name,'I',v))
    for seed,v in sorted(diagnostics.get('cohorts',{}).items()):
        rows.append(trow('cohort',seed,[v[k] for k in ('before','final','intersection')]))
        rows.append(trow('hash','cohort.'+seed,v['membership_sha256']))
        for ref in ('base','incoming'):
            changes=v[ref]['disagreements'];need(len(changes)<=11,'OPR disagreements exceed registered limit')
            rows.append(trow('agreement',seed+'.'+ref,[v[ref]['agreement'],len(changes)]))
            for i,d in enumerate(changes):rows.append(trow('disagree',seed+'.'+ref+'.'+str(i),[d['id'],d['pre'],d['final']]))
    support=r.get('return_evidence',{})
    for seed,v in sorted(support.get('opr_refresh',{}).items()):
        rows.append(trow('coverage',seed,[v[k] for k in ('unique','accepted','coverage','fallback_fraction','exposure_substitution','exposure_fallback')]))
    calibration=support.get('fisher_calibration')
    if calibration:
        rows.append(trow('fisher','coefficients',[calibration['lambda_iso'],calibration['lambda_fisher'],
            statistics.median(calibration['ratios'].values()),calibration['median_ratio_relative_error']]))
        for seed,v in sorted(calibration['inputs'].items()):
            rows.append(trow('fisher',seed,[v[k] for k in ('task_grad_norm','incoming_distance','weighted_distance')]))
    persistence=support.get('persistence') or r.get('persistence')
    if persistence:rows.append(endpoint(key,'persistence','I',persistence))
    for lane,identity in sorted(r.get('lanes',{}).items()):
        # Never expose the allocation nonce, UUID or environment itself.
        match=re.fullmatch(r"\((101|102|103), '([a-z-]+)'\)",lane)
        need(match is not None,'unknown lane key')
        rows.append(trow('lane',match[1]+'.'+match[2],digest(identity)))
    accounting=full.get('accounting',r.get('accounting',{}))
    rows.append(trow('account','status',accounting.get('status','UNMEASURED')))
    rows.append(trow('account','gpu_hours',accounting.get('gpu_hours')))
    rows.append(trow('account','attempts',accounting.get('attempts')))
    if full_detail:
        for phase,v in sorted(accounting.get('by_phase',{}).items()):rows.append(trow('attempt',phase,v))
    return rows+['T|CHECKSUM-SCOPE|advisory-only|TRANSCRIPTION-UNVERIFIED-never-rejects']


# One return cycle covers all currently collected families. These limits apply
# to the combined dispatch, including framing, rather than separately per family.
PART_CHARS=11400
MAX_PARTS=4


def write_parts(out,full,tag,commit,*,max_chars=PART_CHARS,full_detail=False):
    need(re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',tag) and re.fullmatch('[0-9a-f]{40}',commit),'kit identity required')
    need(1000<=max_chars<=12000,'part capacity must be 1000..12000')
    max_chars=min(max_chars,PART_CHARS)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    need(not list(out.glob('RETURN-TEXT-*.md')) and not (out/'full').exists(),'use a fresh text output directory')
    families=full.get('nm_dispatch',[full]);names=[v['nm_family'] for v in families]
    need(names and len(set(names))==len(names) and set(names)<=set(TEXT_FAMILIES),'duplicate/unknown dispatch family')
    need(len({v['infrastructure_authority'] for v in families})==1,'mixed dispatch authorities')
    dispatch='nm_dispatch' in full
    def rows(detail):
        result=[trow('dispatch','families',names)] if dispatch else []
        for v in families:
            if dispatch:result.append(trow('family','begin',v['nm_family']))
            result.extend(payload(v,full_detail=detail))
        return result
    def chunks(records):
        result=[[]];size=0;limit=max_chars-600
        for row in records:
            length=len(body_text([row],89));need(length<=limit,'text record exceeds part capacity')
            if size+length>limit:result.append([]);size=0
            result[-1].append(row);size+=length
        return result
    def emit(directory,parts,mode):
        directory.mkdir(parents=True,exist_ok=True)
        reference=f'T|FULL-SET|{len(parts)}|'+sha(canonical([sha(canonical(c)) for c in parts]))
        paths=[]
        for k,chunk in enumerate(parts,1):
            text=frame_text(tag,commit,k,len(parts),mode,reference,chunk,width=89)
            need(text.isascii() and len(text)<=max_chars and all(len(line)<90 for line in text.splitlines()),'text capacity exceeded')
            path=directory/f'RETURN-TEXT-{k}-of-{len(parts)}.md';path.write_text(text,encoding='ascii');paths.append(path)
        return paths
    # Always retain the unabridged detail locally, even if ESSENTIAL overflows.
    full_paths=emit(out/'full',chunks(rows(True)),'FULL')
    essential=chunks(rows(False))
    need(full_detail or len(essential)<=MAX_PARTS,'dispatch ESSENTIAL exceeds four parts; FULL retained on node')
    paths=full_paths if full_detail else emit(out,essential,'ESSENTIAL')
    local=parse_parts([p.read_text() for p in paths],expect_commit=commit)
    need(all(v=='MATCH' for v in local['part_checksums'].values()),'text self-check differs')
    return paths


def count_endpoints(family,counts):
    """Repeat the native reader's arithmetic, including subtraction/AUC order."""
    def acc(seed,panel,arm,step=40):
        key=str(seed)+'.'+panel;ready(key in counts,'missing correct-counts: '+key)
        row=counts[key];curve=panel=='legal' or (family=='grpo' and panel=='finqa')
        need(len(row)==(13 if curve else 5),'invalid correct-count columns')
        need(all(type(v) is int for v in row) and row[0]>0 and all(0<=v<=row[0] for v in row[1:]),'invalid correct-counts')
        if family=='v2b' and panel in ('september-off','september-on'):
            need(row[0]==(13 if panel=='september-off' else 52),'v2b registered denominator differs')
        index={'base':1,'incoming':2}.get(arm)
        if index is None:index=3+(5 if curve else 1)*(arm=='T')+( (0,10,20,30,40).index(step) if curve else 0)
        return 100*row[index]/row[0]
    def delta(s,p):return acc(s,p,'T')-acc(s,p,'C')
    def gain(s,p,a):return acc(s,p,a)-acc(s,p,'incoming')
    def auc(s,p,a):
        vals=[acc(s,p,a,k) for k in (0,10,20,30,40)]
        return sum((a+b)*5 for a,b in zip(vals,vals[1:]))/40-acc(s,p,'incoming')
    result={}
    def put(name,values,kind='G',one=False):result[name]=dict(values=values,kind=kind,method='one' if one else 'two')
    if family=='v2b':primary=[delta(s,'september-off') for s in SEEDS]
    elif family=='grpo':primary=[delta(s,'finqa') for s in SEEDS]
    elif family=='isdft':primary=[delta(s,'instruction') for s in SEEDS]
    elif family in ('fisher','freeze'):
        panels=GENERAL if family=='fisher' else (*GENERAL,'finqa')
        primary=[sum(delta(s,p) for p in panels)/len(panels) for s in SEEDS]
    else:
        primary=[]
        for seed in SEEDS:
            key=str(seed)+'.finqa-cohort';ready(key in counts,'OPR cohort counts missing')
            row=counts[key];need(len(row)==3 and all(type(v) is int for v in row) and row[0]>0 and all(0<=v<=row[0] for v in row[1:]),'invalid cohort counts')
            primary.append(100*(row[2]-row[1])/row[0])
    put('primary',primary,'P')
    for p in (*GENERAL,'hellaswag'):put(p,[delta(s,p) for s in SEEDS],one=family=='grpo')
    if family=='grpo':
        put('acquisition',[acc(s,'finqa','T')-acc(s,'finqa','base') for s in SEEDS],one=True)
        put('D_sensitivity',[acc(s,'finqa','C')-acc(s,'finqa','base') for s in SEEDS],'I',True)
        put('general_retention',[sum(delta(s,p) for p in GENERAL)/3 for s in SEEDS],'I')
    elif family=='v2b':put('september-on-all52',[delta(s,'september-on') for s in SEEDS])
    else:
        for p in ('finqa','legal'):
            put(p+'-final',[delta(s,p) for s in SEEDS])
            put(p+'-retention' if p=='finqa' else 'legal-gain',[gain(s,p,'T')-gain(s,p,'C') for s in SEEDS])
        put('legal-gain-AUC',[auc(s,'legal','T')-auc(s,'legal','C') for s in SEEDS],'I' if family=='opr' else 'G')
    return result


def recompute(family,endpoints,metadata):
    """Use only seed values and frozen rules, never pasted summary/decision bits."""
    status=metadata['meta']['reported_status'];benchmark=metadata['meta']['benchmark'];technical=metadata['meta']['technical']
    if 'primary' not in endpoints:
        need(status in ('PENDING/INCOMPLETE','UNVERIFIED','UNVERIFIED-HARDWARE','ARCHIVED UNREAD'),'decision lacks endpoints')
        persistence=endpoints.get('persistence')
        if family=='isdft' and persistence:
            need(stats(persistence['values'],ONE)['mean']>=-3,'archive persistence bar differs')
        return dict(status=status,endpoint_verdict_rederived=False)
    need({n for n,v in endpoints.items() if v['kind']=='G'}==guard_names(family),'missing or unexpected guard endpoints')
    primary=(guard if family=='grpo' else benefit)(endpoints['primary']['values'])
    guards={}
    for name in guard_names(family):
        values=endpoints[name]['values']
        if name=='acquisition':
            g=stats(values,ONE);g['pass']=g['lower']>0 and g['positive']==3
        else:g=guard(values,-3 if family=='grpo' else -1,ONE if family=='grpo' else TWO)
        guards[name]=g
    passed=primary['pass'] and all(g['pass'] for g in guards.values())
    result_status=('QUALIFIED' if family=='grpo' else 'NON-INCUMBENT BENCHMARK' if benchmark and family!='v2b' else 'PASS') if passed else 'INCONCLUSIVE'
    if family=='grpo':
        ready('D_sensitivity' in endpoints,'D sensitivity missing')
        d=stats(endpoints['D_sensitivity']['values'],ONE)
        if not (d['lower']>0 and d['positive']==3):result_status='PENDING/INCOMPLETE'
    if family=='opr':
        for seed in map(str,SEEDS):
            ready(seed in metadata.get('cohort',{}) and seed in metadata.get('coverage',{}),'OPR cohort/coverage missing')
            before,final,intersection=metadata['cohort'][seed]
            if min(before,final)<100:result_status='PENDING/INCOMPLETE'
            need(0<=intersection<=min(before,final),'invalid cohort intersection')
            if metadata.get('count'):
                cohort=metadata['count'][seed+'.finqa-cohort'];panel=metadata['count'][seed+'.finqa']
                need(cohort[0]==final and panel[0]==1147,'OPR count/cohort denominator differs')
                need(all(0<=panel[3+i]-cohort[1+i]<=1147-final for i in (0,1)),'OPR cohort counts exceed panel counts')
            for ref in ('base','incoming'):
                key=seed+'.'+ref;ready(key in metadata.get('agreement',{}),'OPR agreement missing')
                agreement,count=metadata['agreement'][key]
                ids=[v for n,v in metadata.get('disagree',{}).items() if n.startswith(key+'.')]
                need(type(count) is int and count==len(ids) and len({v[0] for v in ids})==count,'disagreement count differs')
                need(abs(agreement-(1-count/1147))<1e-12,'agreement fraction differs')
                if count>11 or agreement<.99:result_status='UNVERIFIED'
            unique,accepted,coverage,fallback,substitution,exposure_fallback=metadata['coverage'][seed]
            need(type(unique) is int and type(accepted) is int and unique>0 and 0<=accepted<=unique,'invalid coverage counts')
            need(abs(coverage-accepted/unique)<1e-12 and abs(fallback+coverage-1)<1e-12,'coverage fractions differ')
            need(0<=substitution<=1 and abs(substitution+exposure_fallback-1)<1e-12,'exposure fractions differ')
            if coverage<.8:result_status='PENDING/INCOMPLETE'
    if family=='fisher':
        f=metadata.get('fisher',{});ready(set(f)>={'coefficients',*map(str,SEEDS)},'Fisher calibration missing')
        lam,lam_f,median,error=f['coefficients'];need(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (lam,lam_f)),'invalid Fisher coefficients')
        ratios=[]
        for seed in map(str,SEEDS):
            g,d,w=f[seed];need(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (g,d,w)),'invalid Fisher calibration norms')
            ratios.append(lam_f*w/(lam*d))
        calculated=statistics.median(ratios)
        need(abs(calculated-median)<1e-12 and abs(abs(calculated-1)-error)<1e-12,'Fisher median summary differs')
        if error>1e-4:result_status='PENDING/INCOMPLETE'
    if family=='isdft':
        ready('persistence' in endpoints,'iSDFT persistence missing')
        if stats(endpoints['persistence']['values'],ONE)['mean']>=-3:result_status='ARCHIVED UNREAD'
    expected_lanes=3*(11 if family=='v2b' else 5 if family=='grpo' else 6)
    ready(len(metadata.get('lane',{}))==expected_lanes,'lane identity hashes incomplete')
    result=dict(status='TECHNICAL-ONLY' if technical and result_status not in ('PENDING/INCOMPLETE','UNVERIFIED','ARCHIVED UNREAD') else result_status,
        primary=primary,guards=guards,endpoint_verdict_rederived=True,incumbent_changed=False)
    if technical:result['simulated_status']=result_status
    return result


def parse_parts(texts,*,expect_commit=None,part1_only=False):
    sets={};identity=None;reference=None;checks={};mode=None;total=None
    prefix=r'(?:RESULTS-SHA256|D\|NM-|E\|NM-|W\|NM-(?:cr|opr)\||T\|(?:KIT|PART|MODE|FULL-SET|COUNTS|AUTH|NM|PARTNER-COMPUTED|CHECKSUM-SCOPE)\|)'
    for text in texts:
        text=re.sub(r'[\u200b-\u200f\u2060\ufeff]','',text)
        text=text.translate(str.maketrans({'\u201c':'"','\u201d':'"','\u2018':"'",'\u2019':"'",'\u00a0':' '})).replace('\r','')
        text=re.sub(r'(?m)^[ \t]*(?:>[ \t]+)+','',text);text=re.sub(r'\s+','',text)
        header=r'KIT([A-Za-z0-9_.-]{1,64}?)([0-9a-f]{40})PART(\d+)/(\d+)'
        frames=list(re.finditer(header+r'(.*?)(?='+header+r'|$)',text,re.S));need(frames,'missing KIT frame')
        need(''.join(m.group() for m in frames)==text,'unframed pasted text')
        for frame in frames:
            tag,commit,k,n,raw=frame.groups()[:5];k,n=int(k),int(n)
            need(identity in (None,(tag,commit)),'mixed kit identities');identity=(tag,commit)
            need(expect_commit is None or commit==expect_commit,'unexpected kit commit')
            need(1<=k<=n and total in (None,n),'invalid part count');total=n
            footer=re.search(r'(?<!RESULTS-)SHA256(.*)$',raw);checksum=footer[1] if footer else None
            if footer:raw=raw[:footer.start()]
            rows=[v for v in re.split('(?='+prefix+')',raw) if v]
            rows=[r.replace('RESULTS-SHA256','RESULTS-SHA256 ',1) if r.startswith('RESULTS-SHA256') else r for r in rows]
            need(f'T|KIT|{tag}|{commit}' in rows and f'T|PART|{k}|{n}' in rows,'frame/body identity differs')
            modes=[r.split('|')[2] for r in rows if r.startswith('T|MODE|')]
            need(len(modes)==1 and modes[0] in ('ESSENTIAL','FULL') and mode in (None,modes[0]),'mixed/invalid modes');mode=modes[0]
            need(mode!='ESSENTIAL' or n<=4,'ESSENTIAL exceeds four parts')
            refs=[r for r in rows if r.startswith('T|FULL-SET|')];need(len(refs)==1 and reference in (None,refs[0]),'mixed set references');reference=refs[0]
            payload_rows=[r for r in rows if not r.startswith(('T|KIT|','T|PART|','T|MODE|','T|FULL-SET|','T|COUNTS|'))]
            counters=[r for r in rows if r.startswith('T|COUNTS|')]
            need(counters==['T|COUNTS|'+','.join(key+'='+str(v) for key,v in section_counts(payload_rows).items())],'section row counts differ')
            need(k not in sets,'duplicate part');sets[k]=payload_rows
            checks[str(k)]='MATCH' if sha(body_text(rows,89))==checksum else 'TRANSCRIPTION-UNVERIFIED'
    missing=sorted(set(range(1,total+1))-sets.keys())
    if missing:
        ready(part1_only and set(sets)=={1},'missing parts: '+str(missing))
        return dict(status='PENDING/INCOMPLETE',missing_parts=missing,part_checksums=checks,complete=False)
    rows=[row for k in range(1,total+1) for row in sets[k]]
    wanted=sha(canonical([sha(canonical(sets[k])) for k in range(1,total+1)]))
    if reference!=f'T|FULL-SET|{total}|'+wanted:checks['FULL-SET']='TRANSCRIPTION-UNVERIFIED'
    if rows[0].startswith('T|NM|dispatch|families|'):
        names=json.loads(rows[0].split('|',4)[4]);blocks={};current=None
        need(names and len(names)==len(set(names)) and set(names)<=set(TEXT_FAMILIES),'invalid dispatch families')
        for row in rows[1:]:
            if row.startswith('T|NM|family|begin|'):
                current=json.loads(row.split('|',4)[4]);need(current in names and current not in blocks,'duplicate/unknown family block');blocks[current]=[]
            else:
                need(current is not None,'missing family block');blocks[current].append(row)
        need(list(blocks)==names,'missing/reordered family blocks')
        results={name:parse_family(blocks[name],identity,checks) for name in names}
        need(all(r['family']==name for name,r in results.items()),'dispatch family differs')
        need(len({r['infrastructure_authority'] for r in results.values()})==1,'mixed dispatch authorities')
        return dict(schema='kit-text-return.v1',status='DISPATCH',families=results,complete=True,
            kit_tag=identity[0],kit_commit=identity[1],part_checksums=checks,missing_parts=[],
            endpoint_verdict_rederived=all(r['endpoint_verdict_rederived'] for r in results.values()))
    return parse_family(rows,identity,checks)


def parse_family(rows,identity,checks):
    endpoints={};metadata={};decision=None;authority=None;results_hash=None;warnings=[]
    for row in rows:
        f=row.split('|')
        if f[:2]==['T','NM']:
            need(len(f)==5 and f[3] not in metadata.setdefault(f[2],{}),'duplicate/invalid NM metadata')
            metadata[f[2]][f[3]]=json.loads(f[4])
        elif f[0]=='W':
            need(len(f)==4 and f[1] in ('NM-cr','NM-opr') and re.fullmatch('[A-Za-z0-9_-]+',f[2]) and f[3]=='1','invalid utilisation warning')
            warnings.append(f[2])
        elif f[0]=='E':
            need(len(f)==13 and f[2] not in endpoints and f[3] in ('P','G','I','S'),'invalid endpoint')
            values=json.loads(f[4]);stats(values)
            endpoints[f[2]]=dict(kind=f[3],values=values,method=f[12])
        elif f[0]=='D':
            need(len(f)==9 and decision is None,'invalid decision');decision=f
        elif f[:2]==['T','AUTH']:
            need(authority is None and len(f)==3 and re.fullmatch('[0-9a-f]{64}',f[2]),'invalid authority');authority=f[2]
        elif row.startswith('RESULTS-SHA256 '):
            need(results_hash is None,'duplicate results digest');results_hash=row.split()[1]
        else:need(f[:2] in (['T','PARTNER-COMPUTED'],['T','CHECKSUM-SCOPE']),'unknown/damaged text record')
    family=metadata.get('meta',{}).get('family');need(family in TEXT_FAMILIES and decision and authority,'missing decision identity')
    from kit.m3_mode import CONFIRMATION_SEEDS
    need(metadata['meta'].get('seeds')==list(CONFIRMATION_SEEDS if family=='cr' else SEEDS),'paired seed order differs')
    need(decision[1]=='NM-'+family and decision[2]=='primary','decision family differs')
    need(all(r.startswith(('T|','RESULTS-','D|NM-'+family+'|','E|NM-'+family+'|','W|NM-'+family+'|')) for r in rows),'mixed family payload')
    if family=='cr':
        from kit.cr_text import recompute as cr_recompute
        result=cr_recompute(metadata)
        result['utilisation_warnings']=warnings
        result.update(family='cr',kit_tag=identity[0],kit_commit=identity[1],infrastructure_authority=authority,
            complete=True,missing_parts=[],part_checksums=dict(checks),metadata=metadata,
            provenance='PARTNER-COMPUTED',independent_reverification='UNVERIFIED',reported_status=metadata['meta']['reported_status'])
        return result
    if metadata.get('count'):endpoints.update(count_endpoints(family,metadata['count']))
    try:result=recompute(family,endpoints,metadata)
    except Pending as error:result=dict(status='PENDING/INCOMPLETE',reason=str(error),endpoint_verdict_rederived=False)
    result.update(schema='kit-text-return.v1',family=family,kit_tag=identity[0],kit_commit=identity[1],
        infrastructure_authority=authority,complete=True,missing_parts=[],part_checksums=dict(checks),
        metadata=metadata,endpoints=endpoints,reported_status=metadata['meta']['reported_status'],
        provenance='PARTNER-COMPUTED',independent_reverification='UNVERIFIED')
    if warnings:result['utilisation_warnings']=warnings
    result['package_summaries']={};result['diagnostics']={}
    for name,v in endpoints.items():
        need(v['method'] in ('one','two'),'unknown interval convention')
        if v['kind']=='S':
            _,arm,panel=name.split('.',2);need(arm in ('C','T'),'invalid package arm')
            part=stats(v['values'],ONE) if panel=='legal_gain' else guard(v['values'],69.7 if panel=='legal_absolute' else -1 if panel=='finqa' else -3,ONE)
            result['package_summaries'].setdefault(arm,{})[panel]=part
        elif v['kind']=='I':result['diagnostics'][name]=stats(v['values'],ONE if v['method']=='one' else TWO)
    verdict=result.get('simulated_status',result['status'])
    result['reported_verdict_matches']=verdict==metadata['meta']['reported_status']
    result['decision_line_matches']=decision[3]==metadata['meta']['reported_status'].replace(' ',':')
    return result
