#!/usr/bin/env python3
"""X4: registered P2 three-seed calibration; boundary replacement stays closed."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.x_common import *
from kit.p2_l2 import VARIANT,coefficient

from kit.p2_smoke import ASSERTIONS as SMOKE_CHECKS


def smoke(path,standin):
    require(path is not None and Path(path).is_file(),'P2 passing smoke receipt required')
    return validate_smoke(read(path),standin)


def validate_smoke(value,standin):
    if not standin:
        from kit.x_smoke_settings import AUTHORISED_ATTEMPTS
        require(value.get('schema')=='x-integration-smoke.v3' and value.get('part')=='p2' and
            value.get('attempt') in range(1,AUTHORISED_ATTEMPTS['p2']+1) and
            value.get('maximum_attempts') in range(value['attempt'],AUTHORISED_ATTEMPTS['p2']+1) and
            value.get('retries')==0 and value.get('gpu')=='H100:2' and value.get('timeout_seconds')==(3000 if value.get('attempt') in (4,5) else 1800) and
            0<value.get('total_ceiling_usd',float('inf'))<=(15 if value.get('attempt') in (4,5) else 10) and value.get('evidence_stem')=='x-integration-p2-smoke',
            'real P2 requires an authorised bounded split P2 integration receipt')
        if value['attempt']>=2:
            from kit.x_smoke_batches import validate_pairing
            validate_pairing(value.get('p2_smoke',{}).get('paired_batches',{}))
    if value.get('schema')=='x-integration-smoke.v3':
        require(value.get('part')=='p2','resume integration receipt cannot qualify P2')
        require(value.get('passed') is True or (value.get('p2_qualified') is True and
            value.get('failure_scope')=='after-P2-verdict' and
            any(p.get('stage')=='P2-smoke-result' and p.get('p2_smoke')==value.get('p2_smoke') and p.get('p2_smoke',{}).get('passed') is True for p in value.get('partial_records',[]))),
            'split P2 integration smoke failed before a passing P2 result')
        value=value['p2_smoke']
    from kit.p2_smoke import SCHEMA
    require(value.get('schema')==SCHEMA and value.get('variant')==VARIANT,'old P receipt cannot enable P2')
    require(value.get('placement')=='real-verl-fsdp2' and value.get('world_size')==2 and value.get('model_revision')=='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e' and value.get('precision')=='fp32-masters/fp64-accumulation','old/incorrect smoke placement or precision')
    require(value.get('technical_synthetic',False) is standin,'P2 smoke scope mismatch')
    require(value.get('passed') is True and all(value.get('assertions',{}).get(k,{}).get('status')=='PASS' for k in SMOKE_CHECKS),
        'P2 smoke failed or incomplete; defer without recalibration')
    from kit.p2_smoke import code_hashes
    require(value.get('code_sha256')==code_hashes(),'P2 smoke code changed')
    return value


def calibration(out,doc):
    path=out/'p2/calibration.json';value=read(path)
    require(value.get('schema')=='rung-p2-calibration.v1' and value.get('variant')==doc.get('p_variant')==VARIANT,'P2 calibration/variant required')
    require(value['binding_sha256']==sha(out/'binding.json') and value['anchor']==doc['p2_anchor'],'P2 calibration identity changed')
    require(value['node_limits_sha256']==doc['p2_node_limits_sha256']==sha(ROOT/'kit/resources/p2/node-limits.json'),'node feasibility settings changed')
    require(value['coefficient']==coefficient(value['seeds']),'P2 coefficient rule changed')
    require(value['implementation']==doc['implementation'],'P2 calibration implementation changed')
    require(sha(out/'p2/smoke-receipt.json')==value['smoke_sha256'],'P2 calibration smoke changed')
    validate_smoke(read(out/'p2/smoke-receipt.json'),doc['technical_synthetic'])
    for name,expected in value['files'].items():require(sha(out/name)==expected,'P2 measured evidence changed: '+name)
    return value


def job(a):
    doc=context(a.out,a.standin)
    if doc.get('p_variant')=='disabled':print('SKIP P2: no enabled variant');return
    require(doc.get('p_variant')==VARIANT and a.action in ('measure','reduce'),'boundary P utility/replacement is closed; use P2 measure')
    proof=smoke(a.smoke,a.standin)
    m.checked_model(doc['p2_anchor'],a.standin,base=True)
    root=a.out/'p2'
    root.mkdir(parents=True,exist_ok=True)
    smoke_copy=root/'smoke-receipt.json'
    if smoke_copy.exists():require(sha(smoke_copy)==sha(a.smoke),'P2 smoke receipt changed')
    else:smoke_copy.write_bytes(Path(a.smoke).read_bytes())
    if (root/'calibration.json').exists():
        value=calibration(a.out,doc);require(value['smoke_sha256']==sha(a.smoke),'P2 smoke binding changed')
        print('SKIP: frozen three-seed P2 coefficient');return
    require(not list((a.out/'scores').glob('*/manifest.json')) and not list((a.out/'runs').glob('*/started.json')),
        'P2 X4 must precede every X training/scored measurement')
    require(not (a.out/'replacement').exists(),'old replacement state cannot enable P2')
    from kit.x_p2_measure import measure
    seeds={};files={}
    for seed in SEEDS:
        folder=root/f's{seed}'
        width=1 if a.standin else 8
        if not all((folder/f'measure-r{rank}.json').exists() for rank in range(width)):
            if a.action=='reduce':
                atomic(a.out/'dispositions/p.json',dict(status='DEFERRED',reason='X4 three-seed measurement incomplete'));return
            if getattr(a,'seed',None) is not None and seed!=a.seed:continue
            measure(a,doc,seed)
        batch=read(folder/'batch.json')
        require(batch['schema']=='rung-p2-first-batch.v1' and batch['variant']==VARIANT and
            sha(folder/'actor-input.pt')==batch['sha256'],'P2 missing bound rollout batch')
        files[str((folder/'actor-input.pt').relative_to(a.out))]=batch['sha256']
        records=[read(folder/f'measure-r{rank}.json') for rank in range(width)]
        for rank,r in enumerate(records):
            require(r['schema']=='rung-p2-measure.v1' and r['variant']==VARIANT and r['rank']==rank and
                r['state_before']==r['state_after'] and r['optimizer_steps']==r['scheduler_steps']==r['ema_steps']==0 and
                r['after_reduction_unscaling'] and r['before_clipping'],'P2 measurement mutated state or wrong boundary')
            require(sha(folder/f'rng-r{rank}.pt')==r['rng_file_sha256'],'P2 pinned RNG changed')
            files[str((folder/f'rng-r{rank}.pt').relative_to(a.out))]=r['rng_file_sha256']
        if not a.standin:
            from kit.x_p2_feasibility import validate_measurement
            validate_measurement(a.out,doc,seed)
        require(all((r['task_grad_norm'],r['incoming_distance'])==(records[0]['task_grad_norm'],records[0]['incoming_distance']) for r in records),'P2 all-rank reduction disagreement')
        seeds[str(seed)]={k:records[0][k] for k in ('task_grad_norm','incoming_distance')}
        seeds[str(seed)]['first_batch_sha256']=(batch['schedule_sha256'] if a.standin else read(folder/'schedule.json')['first_batch_sha256'])
        for path in folder.iterdir():
            if path.is_file():files[str(path.relative_to(a.out))]=sha(path)
    if getattr(a,'seed',None) is not None:
        print('X4 seed measurement frozen; calibration reducer requires all three seeds');return
    bind(root/'calibration.json',dict(schema='rung-p2-calibration.v1',variant=VARIANT,
        binding_sha256=sha(a.out/'binding.json'),anchor=doc['p2_anchor'],smoke_sha256=sha(a.smoke),
        node_limits_sha256=doc['p2_node_limits_sha256'],coefficient=coefficient(seeds),seeds=seeds,files=files,implementation=implementation(),technical_synthetic=a.standin))
    print('P2 frozen lambda:',read(root/'calibration.json')['coefficient']['value'])


def configure(p):
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--action',choices=('measure','reduce','utility','replace'),required=True)
    p.add_argument('--smoke',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--tokenizer');p.add_argument('--sdpo',type=Path)
    p.add_argument('--trainer-python',default=sys.executable)
if __name__=='__main__':raise SystemExit(entry(job,configure))
