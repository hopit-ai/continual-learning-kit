#!/usr/bin/env python3
"""Add-on forecast; measured ContractNLI stages supersede D-family proxy timings."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *


def estimate(configuration, timings=None, tier2=False, dispositions=None, tier1_dispositions=None):
    k = len(domains(configuration)); stages = 6 if k == 1 else 12
    timings = timings or {}
    require(not timings.get('technical_synthetic', False), 'CPU stand-in latency cannot price GPU work')
    stage = timings.get('contract_nli_stage_gpu_hours', 58*8/60)
    require(stage > 0, 'invalid measured stage timing')
    probes = 0 if k == 1 else 3*k
    if tier1_dispositions is not None:
        candidates=('control','p') if k==1 else ('control','p','drift','plasticity')
        stages=3*sum(tier1_dispositions.get(arm,'PENDING') in ('READY','PENDING') for arm in candidates)
        if tier1_dispositions.get('plasticity') not in ('READY','PENDING'):probes=0
    extra = {name: timings.get(name) for name in ('generation_gpu_hours','scoring_gpu_hours','p2_measurement_gpu_hours','regulariser_gpu_hours','startup_allocation_gpu_hours')}
    training = stages*stage
    preflight=timings.get('p2_disposable_feasibility_gpu_hours',3*stage)
    result=dict(stages=stages, domains=k, probes=probes, training_gpu_hours=round(training,2),
        registered_full_training_gpu_hours=93,p2_disposable_feasibility_gpu_hours=round(preflight,2),p2_measurement_prior_gpu_hours=[3,4.2], full_proxy_exact=round(12*58*8/60,2),
        training_basis='measured ContractNLI' if 'contract_nli_stage_gpu_hours' in timings else '58 min x 8 GPUs D-family proxy',
        probe_gpu_hours=[probes*.75, probes*1.4], extra=extra,
        known_subtotal_gpu_hours=[round(training+preflight+probes*v+sum(x for x in extra.values() if x is not None),2) for v in (.75,1.4)],
        complete=all(x is not None for x in extra.values()),
        note='Reprice pending/uninformative arms, actual failures and wall time before send; missing costs are not zeros or approval.')
    if tier2:
        from kit.x_tier2 import COMPARISONS
        statuses={name:('DEGENERATE' if k==1 else (dispositions or {}).get(name,'READY')) for name in COMPARISONS}
        counted=[name for name,state in statuses.items() if state in ('READY','PENDING')]
        second_stages=6*len(counted)
        extras={key:timings.get(key) for key in ('tier2_generation_gpu_hours','tier2_scoring_gpu_hours',
            'tier2_repeated_drift_gpu_hours','tier2_regulariser_gpu_hours','tier2_startup_allocation_gpu_hours')}
        result['tier2']=dict(comparisons=statuses,priced_comparisons=counted,stages=second_stages,
            training_gpu_hours=round(second_stages*stage,2),maximum_executed_comparisons=6,maximum_executed_stages=36,
            maximum_bound_comparisons=5,maximum_bound_stages=30,registered_bound_training_gpu_hours=231.9,
            ordinary_training_scoring_startup_gpu_hours=[49.2*len(counted),52.8*len(counted)],
            extra=extras,complete=all(v is not None for v in extras.values()),
            additional_probes=0,probe_policy='Reuse frozen incoming CE measurements, never training controls; tier-1 3K probes charged once.',
            note='Two speculative stacking candidates; one is bound. Pending branches remain in the upper estimate; skipped branches cost zero future stages. Actual failures remain charged.')
        result['total_stages']=stages+second_stages
        result['total_training_gpu_hours']=round((stages+second_stages)*stage,2)
        result['complete']=result['complete'] and result['tier2']['complete']
    return result


def job(a):
    doc = context(a.out, a.standin)
    second=(a.out/'tier2/plan.json').exists() or getattr(a,'tier2',False)
    if second:
        from kit.x_tier2 import COMPARISONS,readiness
        states={name:readiness(a.out,doc,name) for name in COMPARISONS}
        support=read(a.out/'support.json') if (a.out/'support.json').exists() else {}
        deferred=(doc.get('p_variant')=='disabled' or
            (a.out/'dispositions/p.json').exists() and read(a.out/'dispositions/p.json').get('status')=='DEFERRED')
        first=dict(control='READY',p='DEFERRED' if deferred else 'READY' if (a.out/'p2/calibration.json').exists() else 'PENDING',
            **{s:('DEGENERATE' if len(doc['domains'])==1 else support.get(s,{}).get('status','PENDING')) for s in ('drift','plasticity')})
    else:states=None;first=None
    result = estimate(doc['configuration'], read(a.timings) if a.timings else None,second,states,first)
    if first is not None:result['tier1_dispositions']=first
    atomic(a.out/'pricing.json', result)
    print(json.dumps(result, sort_keys=True))

def configure(p):
    p.add_argument('--timings', type=Path)
    p.add_argument('--tier2', action='store_true')
if __name__ == '__main__': raise SystemExit(entry(job, configure))
