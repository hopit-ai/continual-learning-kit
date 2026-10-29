"""Mechanical gates on real two-rank P2 state; no efficacy interpretation."""
import math,statistics
from kit.s_common import require
from kit.x_p2_state import validate_geometry
from kit.p2_smoke import ASSERTIONS,SCHEMA,code_hashes,analytic_checks


def evaluate(preparation,control,challenger,coefficient,exports,model_revision,container_host=None):
    require(len(preparation)==len(control)==len(challenger)==2,'all smoke ranks required')
    for group in (preparation,control,challenger):
        require(sorted(v['rank'] for v in group)==[0,1],'smoke rank inventory changed')
        validate_geometry([v['geometry'] for v in group],2)
        require(all(v['precision']=='fp32-masters/fp64-accumulation' and len(v['updates'])==4 and len(v['distance_trajectory'])==5 for v in group),'registered smoke dose or precision unavailable')
        require(all((v['updates'][i]['lr']==0. if i==0 else math.isclose(v['updates'][i]['lr'],1e-5*i/10,rel_tol=1e-9)) for v in group for i in range(4)),'smoke learning-rate schedule changed')
    require(all(v['attention_implementation']=='eager' and v['deterministic_configuration']['deterministic'] is True
        and v['deterministic_configuration']['deterministic_warn_only'] is False for v in preparation+control+challenger),'actual smoke attention/deterministic mode unverified')
    from kit.x_integration_smoke import MODEL_REVISION
    require(model_revision==MODEL_REVISION,'smoke model revision changed')
    require(coefficient['frozen_before_paired_updates'] is True,'smoke coefficient not prospectively frozen')
    for group in (preparation,control,challenger):
        require(group[0]['distance_trajectory']==group[1]['distance_trajectory'],'distributed distance reductions disagree')
    require(all(r['lambda_value']==expected and r['after_reduction_unscaling'] and r['before_clipping']
        for group,expected in ((preparation,0.),(control,0.),(challenger,coefficient['lambda_value'])) for v in group for r in v['updates']),
        'regularizer coefficient or native boundary changed')
    require(all(v['distance_trajectory'][0]==v['distance_trajectory'][1]==0 for v in preparation),'preparation must start at the actual pinned base, with a zero-LR first update')
    require(math.isclose(preparation[0]['distance_trajectory'][4],control[0]['distance_trajectory'][0],rel_tol=1e-8,abs_tol=1e-10),'paired incoming differs from actual prepared master state')
    initial=all(a['initial']==b['initial'] for a,b in zip(control,challenger))
    d0=control[0]['distance_trajectory'][0];value=coefficient['lambda_value']
    require(math.isfinite(d0) and d0>0 and math.isfinite(value) and value>0,'smoke lambda inputs invalid')
    formula=math.isclose(coefficient['incoming_distance']**2,d0,rel_tol=1e-8) and math.isclose(value,.1*coefficient['task_grad_norm']/coefficient['incoming_distance'],rel_tol=1e-12) and all(math.isclose(value,.1*v['updates'][0]['task_grad_norm']/math.sqrt(d0),rel_tol=1e-6) for v in control+challenger)
    ratios=[b['updates'][i]['combined_preclip_grad_norm']/a['updates'][i]['combined_preclip_grad_norm'] for a,b in zip(control,challenger) for i in range(4)]
    times=lambda group:[max(v['updates'][i]['seconds'] for v in group) for i in range(4)]
    full_times=lambda group:[max(v['updates'][i]['full_seconds'] for v in group) for i in range(4)]
    full_timing=statistics.median(full_times(challenger)[1:])/statistics.median(full_times(control)[1:])
    timing=statistics.median(times(challenger)[1:])/statistics.median(times(control)[1:])
    finite=all(v['base_unchanged'] and v['teacher_gradients_absent'] and not v['teacher_regularised'] and
        all(r['finite'] and r['optimizer_materialised'] and all(math.isfinite(r[k]) for k in ('task_loss','regulariser','total_loss','combined_preclip_grad_norm')) for r in v['updates']) for v in preparation+control+challenger)
    analytic=analytic_checks()['passed'] and all(
        math.isclose(r['regulariser'],v['distance_trajectory'][i]/2,rel_tol=1e-8,abs_tol=1e-10) and
        math.isclose(r['analytic_combined_squared'],r['task_grad_norm']**2+2*r['lambda_value']*r['task_delta_dot']+r['lambda_value']**2*v['distance_trajectory'][i],rel_tol=1e-10,abs_tol=1e-10) and
        math.isclose(r['combined_squared'],r['analytic_combined_squared'],rel_tol=2e-5,abs_tol=1e-10) and
        r['analytic_gradient_error']<=1e-6 for v in control+challenger for i,r in enumerate(v['updates']))
    for name,group in (('incoming',control),('control',control),('p2',challenger)):
        point=0 if name=='incoming' else 4
        require(exports[name]['dtype']=='torch.float32' and math.isclose(exports[name]['distance2'],group[0]['distance_trajectory'][point],rel_tol=1e-8,abs_tol=1e-10),'actual exported weights lost registered precision')
    from kit.x_smoke_memory import qualified,gate
    from kit.x_smoke_settings import MEMORY_GIB
    gib=1024**3
    reservation=all(max(v['gpu']['reserved_bytes'],v['gpu']['device_used_bytes'])<=.95*v['gpu']['total_bytes'] and qualified(v['host'],MEMORY_GIB*gib) for v in preparation+control+challenger)
    if container_host is not None:reservation=reservation and qualified(container_host,MEMORY_GIB*gib)
    delta=control[0]['distance_trajectory'][4]-challenger[0]['distance_trajectory'][4]
    checks=dict(finite=finite,analytic=analytic,coefficient=formula,norms=all(.5<=r<=2 for r in ratios),timing=timing<=1.1 and full_timing<=1.1,
        reservation=reservation,contraction=delta>=1e-4*d0>0,identical_start=initial)
    host_gates=[gate(v['host'],MEMORY_GIB*gib) for v in preparation+control+challenger]
    if container_host is not None:host_gates.append(gate(container_host,MEMORY_GIB*gib))
    return dict(schema=SCHEMA,variant='p2-l2-init',placement='real-verl-fsdp2',world_size=2,
        model_revision=model_revision,precision='fp32-masters/fp64-accumulation',technical_synthetic=False,scientific_d_run=False,
        code_sha256=code_hashes(),assertions={k:dict(status='PASS' if checks[k] else 'FAIL',value=checks[k]) for k in ASSERTIONS},
        container_host=container_host,passed=all(checks.values()),
        mechanics_passed=all(v for k,v in checks.items() if k!='reservation'),resources_passed=reservation,
        host_gates=host_gates,coefficient=coefficient,preparation=preparation,control=control,challenger=challenger,
        full_step_time_ratio=full_timing,exported_distances=exports,gradient_ratios=ratios,step_time_ratio=timing,contraction=dict(delta=delta,D0=d0,threshold=1e-4*d0))
