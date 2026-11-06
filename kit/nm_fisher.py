#!/usr/bin/env python3
"""Eight-rank per-example empirical Fisher and measured estimator preflight.

Launch via torch.distributed.run, only after CPU preparation. Shards are native
estimator checkpoints; CPU finalisation hashes/publishes them in a separate row.
"""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os
import time
from kit.nm_common import *


def estimate(plan,action,out):
    import torch
    from torch.distributed.device_mesh import init_device_mesh
    from kit.nm_model import prepared_reference
    from kit.nm_loss import fisher_estimate,distributed_trace
    from kit.m3_loss import replay_loss
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl');need(torch.distributed.get_world_size()==8,'Fisher production requires eight ranks')
    p=load(plan);out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    model=prepared_reference(p['base_model'],BASE,init_device_mesh('cuda',(8,)),binding=Path(p['base_binding']),
        binding_sha256=p['base_binding_sha256'],files_metadata=p['base_metadata'])
    model.requires_grad_(True);model.eval()
    rows=p['rows'];need(len(rows)==256,'Fisher quotas must total 256')
    if action=='preflight':rows=p['preflight_rows'];need(len(rows)==32,'Fisher preflight needs 16/8/8')
    timings={}
    def sequence_score(model,row):
        x={k:torch.tensor(v,device='cuda',dtype=torch.long) for k,v in row['inputs'].items()}
        x['response_mask']=x['response_mask'].float()
        clock=time.monotonic();value=-replay_loss(model,x)
        torch.cuda.synchronize();timings.setdefault(row['domain'],[]).append(time.monotonic()-clock)
        return value
    # Warmup and longest rows are memory-only, never substituted for the estimator set.
    if action=='preflight':
        for r in p['longest_rows']:
            model.zero_grad(set_to_none=True);sequence_score(model,r).backward();torch.cuda.synchronize()
        timings={}
    gradient_times=[]
    def timed(model,row):
        before=time.monotonic();value=sequence_score(model,row)
        # End-to-end per-example timing includes backward and device accumulation.
        value.register_hook(lambda g:g)
        gradient_times.append(before);return value
    raw=fisher_estimate(model,rows,timed);torch.cuda.synchronize();compute=time.monotonic()-gradient_times[0]
    weights,trace=distributed_trace(raw,torch.device('cuda',rank))
    # Record total per-example times from successive starts, including last tail.
    starts=gradient_times+[time.monotonic()]
    per_example=[b-a for a,b in zip(starts,starts[1:])]
    # One device-to-host publication per normalized shard, after all examples.
    torch.save({n:w.cpu() for n,w in weights.items()},out/f'rank-{rank}.pt')
    # Raw preflight values allow exact row reuse in a future implementation; this
    # version recomputes all 256 and charges preflight once, with no reuse credit.
    record=dict(action=action,base_revision=BASE,plan_sha256=sha(plan),rank=rank,trace=trace,
        examples=len(rows),compute_seconds=compute,wall_seconds=time.monotonic()-started,
        per_domain_seconds={d:[t for r,t in zip(rows,per_example) if r['domain']==d] for d in ('instruction','maths','knowledge')},
        estimator='sequence-score empirical proxy; response/EOS mean; individual gradient squares',
        gradients_reused=False,optimizer_mutations=0)
    atomic(out/f'rank-{rank}.json',record);torch.distributed.barrier();torch.distributed.destroy_process_group()


def price(preflight,calibrations,startup_export_hours,reserve):
    """Registered E formula, decimal rounding only at complete component boundary."""
    from decimal import Decimal
    need(.25<=reserve<=.50,'Fisher timing reserve must be declared 25–50%')
    counts={'instruction':128,'maths':64,'knowledge':64};seconds=Decimal(0)
    for domain,n in counts.items():
        values=preflight['per_domain_seconds'].get(domain,[]);ready(values,'Fisher domain timing absent')
        seconds+=Decimal(n)*sum(Decimal(str(v)) for v in values)/len(values)
    ready(set(calibrations)==set(map(str,SEEDS)),'three measured calibration timings required')
    seconds+=sum(Decimal(str(v)) for v in calibrations.values())
    compute=Decimal(8)*seconds/Decimal(3600)
    return dict(compute_gpu_hours=money(compute),reserve_gpu_hours=money(compute*Decimal(str(reserve))),
        startup_export_gpu_hours=money(startup_export_hours),
        total_gpu_hours=money(Decimal(money(compute))+Decimal(money(compute*Decimal(str(reserve))))+Decimal(money(startup_export_hours))),
        note='preflight and diagnostic/retry allocation charges are added once separately')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    p.add_argument('--action',choices=('preflight','estimate'),required=True);a=p.parse_args();estimate(a.plan,a.action,a.out)
if __name__=='__main__':main()
