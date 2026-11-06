"""Budgeted seventh-family return: integer counts, frozen CR-G arithmetic."""
import os
import re
from contextlib import contextmanager
from kit.nm_common import need, ready, stats, digest
from kit.cr_confirmation import SEEDS, from_counts

METRICS=('ifeval','math','knowledge','finqa','test','math_hard','hellaswag')
STAGES=('finqa','contract_nli','math_hard')


@contextmanager
def mode():
    previous=os.environ.get('KIT_M3_MODE');os.environ['KIT_M3_MODE']='cr-confirmation'
    try:yield
    finally:
        if previous is None:os.environ.pop('KIT_M3_MODE',None)
        else:os.environ['KIT_M3_MODE']=previous


def compact(counts):
    result={}
    for seed in SEEDS:
        for metric in METRICS:
            keys=[f'base.{seed}.finqa.0.{metric}',
                  *[f'{arm}.{seed}.{stage}.{step}.{metric}' for arm in ('G','CR') for stage in STAGES for step in (0,40)]]
            # Counts absent from non-applicable incoming panels stay null. The
            # frozen reader decides which constituents are mandatory.
            present=[counts[k] for k in keys if k in counts]
            if not present:continue
            n=present[0][0];need(all(v[0]==n for v in present),'CR panel denominator changed')
            result[f'{seed}.{metric}']=[n,*[counts[k][1] if k in counts else None for k in keys]]
    return result


def expand(counts):
    result={}
    for key,row in counts.items():
        seed,metric=key.split('.');need(int(seed) in SEEDS and metric in METRICS,'foreign confirmation seed/metric')
        need(isinstance(row,list) and len(row)==14 and type(row[0]) is int and row[0]>0,'invalid CR count columns')
        keys=[f'base.{seed}.finqa.0.{metric}',
              *[f'{arm}.{seed}.{stage}.{step}.{metric}' for arm in ('G','CR') for stage in STAGES for step in (0,40)]]
        for name,count in zip(keys,row[1:]):
            if count is None:continue
            need(type(count) is int and 0<=count<=row[0],'invalid CR correct-count')
            result[name]=[row[0],count]
    return result


def payload(full,full_detail=False):
    from kit.nm_text import trow,endpoint,encoded
    r=full['reading'];status=r['status'];c=r.get('contrasts',{}).get('CR-G',{})
    p=c.get('primary',{})
    rows=['T|AUTH|'+full['infrastructure_authority'],trow('meta','family','cr'),
          trow('meta','seeds',list(SEEDS)),trow('meta','reported_status',status),
          trow('meta','technical',r.get('technical_synthetic',False)),trow('meta','benchmark',False),
          trow('meta','future_precedence','v8-only;v7-six-rows-I0'),
          'T|PARTNER-COMPUTED|independent-reverification=UNVERIFIED',
          'D|NM-cr|primary|'+status.replace(' ',':')+'|'+'|'.join(encoded(p.get(k)) for k in ('mean','lower','upper','positive'))+'|-']
    proof=r.get('finqa_reproduction',{})
    if proof:
        rows.append(trow('finqa','reproduction',proof))
        for warning in proof.get('errors',[]):rows.append('W|NM-cr|'+re.sub('[^A-Za-z0-9_-]+','_',warning).strip('_')+'|1')
    if r.get('qualification_pending'):
        rows.append('W|NM-cr|qualification_pending_'+re.sub('[^A-Za-z0-9_-]+','_',r['qualification_pending']).strip('_')+'|1')
    for warning in r.get('utilisation_warnings',[]):rows.append('W|NM-cr|'+warning+'|1')
    for key,value in compact(r.get('correct_counts',{})).items():rows.append(trow('count',key,value))
    for name,value in [('primary',p),*[(stage+'/final/ifeval',c.get('guards',{}).get(stage+'/final/ifeval',{})) for stage in STAGES]]:
        if value:
            v=stats(value['values']);rows.append(endpoint('NM-cr',name,'P' if name=='primary' else 'G',v,'mean>=2;LB>0;pos=3' if name=='primary' else 'LB>=-1'))
    rows.append(trow('meta','failed_guards',c.get('failed_guards',[])))
    if full_detail:
        for key,value in sorted(r.get('correct_counts',{}).items()):rows.append(trow('detail',key,value))
        for name,value in c.get('guards',{}).items():
            if not name.endswith('/final/ifeval'):rows.append(endpoint('NM-cr',name,'G',stats(value['values']),'LB>=-1'))
    for key,value in r.get('lane_hashes',{}).items():rows.append(trow('lane',key,value))
    rows.append(trow('account','status',full.get('accounting',{}).get('status','PENDING process/allocation reconciliation')))
    account=full.get('accounting',{})
    for field in ('gpu_hours','process_gpu_hours','attempts'):
        rows.append(trow('account',field,account.get(field)))
    rows.append(trow('account','reconciliation_request','per-arm/seed/stage:process/allocated-GPUh;train/prep/qualification/native/reference/export/idle/failed;v6a-298.3-coverage'))
    rows.append(trow('account','wall_time_risk_gt24',True))
    return rows+['T|CHECKSUM-SCOPE|advisory-only|TRANSCRIPTION-UNVERIFIED-never-rejects']


def recompute(metadata):
    status=metadata['meta']['reported_status']
    need(status in ('PASS','INCONCLUSIVE','PENDING/INCOMPLETE','UNVERIFIED','UNVERIFIED-HARDWARE'),'unknown CR status')
    if any(key.split('.')[0] in ('101','102','103') for key in metadata.get('count',{})):
        return dict(status='UNVERIFIED',reason='exploratory seed contamination',endpoint_verdict_rederived=False)
    with mode():result=from_counts(expand(metadata.get('count',{})))
    computed=result['status']
    # Pasted counts reproduce bars, while missing/invalid provenance stays an
    # explicit stop; transport cannot repair hardware/input provenance.
    result['endpoint_verdict_rederived']=computed in ('PASS','INCONCLUSIVE')
    result['scientific_count_verdict']=computed
    if status in ('PENDING/INCOMPLETE','UNVERIFIED','UNVERIFIED-HARDWARE'):result['status']=status
    result['reported_verdict_matches']=computed==status
    proof=metadata.get('finqa',{}).get('reproduction')
    if proof:
        result['finqa_reproduction']=proof
        if proof.get('status')!='COMPLETE' or proof.get('errors'):
            result.update(status='PENDING/INCOMPLETE',endpoint_verdict_rederived=False)
    if metadata['meta'].get('technical'):
        result.update(status='TECHNICAL-ONLY',simulated_status=computed)
    return result
