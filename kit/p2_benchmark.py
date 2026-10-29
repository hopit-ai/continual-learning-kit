#!/usr/bin/env python3
"""CPU streaming preflight for the P2 smoke; no downloads or GPU operations.

Each of three distinct arrays spans >=4 GB. Fresh nonoverlapping resident
windows bound RAM; their pages are retired after timing, never a reused 64 MB
buffer. Setup/initialization/page retirement is outside each operation's timer.
"""
import argparse,gc,hashlib,json,math,mmap,statistics,time
from pathlib import Path
from kit.s_common import ROOT,sha,require

ATTEMPT3=ROOT/'docs/phase2/evidence/rung-p-smoke.attempt3.json'
TARGET_ELEMENTS=8_200_000_000


def corrected_forecast(measurements,elements,attempt3):
    factor=TARGET_ELEMENTS/elements
    local={k:v['median_seconds']*factor for k,v in measurements.items()}
    paired=[r['seconds'] for k in ('control','replaced') for r in attempt3[k]]
    train=next(v['seconds'] for v in attempt3['steps'] if v['name']=='assertions-and-paired-updates')
    download=next(v['seconds'] for v in attempt3['steps'] if v['name']=='model-8b')
    setup=(train-sum(paired))/2
    require(setup>0,'attempt-3 setup basis invalid')
    # Full observed Modal step / CPU-only local AdamW+EMA+copy+finite time is a
    # conservative slowdown proxy: GPU work remains in its numerator. Also keep
    # the review's attempt-3-informed 60--90 GB/s planning envelope; use its slow
    # end for the gate, never present it as directly measured memory bandwidth.
    local_gbps=16*TARGET_ELEMENTS/local['addition']/1e9
    recipe_factor=max(1.,statistics.median(paired)/local['adamw_ema'])
    envelope_factor=max(1.,local_gbps/60.)
    slowdown=max(recipe_factor,envelope_factor)
    corrected={k:v*slowdown for k,v in local.items()}
    timed=4*max(sum(r['seconds'] for r in attempt3[k]) for k in ('control','replaced'))
    audit_counts=dict(norm=24,distance=14,dot=8,finite=16,hash=5,addition=4)
    audit=sum(corrected[k]*count for k,count in audit_counts.items())
    # Four arms plus one separate CPU base load. Charge a full measured setup
    # allowance to the latter too. Include exported bf16 D2H copies at 12 GB/s.
    setup_total=5*setup
    export_copy=4*(TARGET_ELEMENTS*2)/12e9
    startup=download+30.
    lifetime=timed+setup_total+audit+export_copy+startup
    return dict(schema='rung-p2-corrected-forecast.v2',attempt3_training_seconds=train,
        attempt3_update_seconds=sum(paired),attempt3_per_arm_setup_seconds=setup,
        attempt3_median_update_seconds=statistics.median(paired),setup_allowances=5,
        setup_seconds=setup_total,startup_download_seconds=startup,export_copy_seconds=export_copy,
        native_update_seconds=timed,audit_pass_counts=audit_counts,
        local_projected_seconds=local,local_add_streaming_gbps=local_gbps,
        attempt3_recipe_slowdown_factor=recipe_factor,attempt3_bandwidth_envelope_gbps=[60,90],
        bandwidth_envelope_source='round-2 review estimate informed by attempt-3 CPU master AdamW/EMA/copy timings; not a direct bandwidth measurement',
        bandwidth_slowdown_factor=envelope_factor,applied_modal_slowdown_factor=slowdown,
        corrected_projected_seconds=corrected,corrected_timed_add_seconds=corrected['addition'],
        corrected_lifetime_seconds=lifetime,corrected_lifetime_minutes=lifetime/60,
        launch_allowed=lifetime<=25*60 and corrected['addition']<=1.5,
        launch_limits=dict(lifetime_seconds=1500,timed_add_seconds=1.5))


def launch_forecast(value):
    """Reject old/uncorrected/nonfinite forecasts; enforce both manager limits."""
    f=value['forecast']
    require(f['schema']=='rung-p2-corrected-forecast.v2','corrected P2 forecast required')
    require(all(isinstance(f[k],(int,float)) and math.isfinite(f[k]) and f[k]>=0 for k in
        ('corrected_lifetime_seconds','corrected_timed_add_seconds')),'nonfinite corrected forecast')
    require(f['corrected_lifetime_seconds']<=1500 and f['corrected_timed_add_seconds']<=1.5,
        'P2 corrected preflight exceeds 25 minutes or 1.5 seconds per timed add')
    return f


def run(elements=1_000_000_000,window=64*1024*1024,repeats=3):
    import torch,psutil
    from kit.p2_smoke import add_in_place,distance,gradient_dot,norm_squared,finite,tensor_hash
    torch.set_num_threads(8)
    require(elements*4>=4_000_000_000,'streaming benchmark requires >=4 GB per distinct array')
    require(psutil.virtual_memory().available>=4_500_000_000,'insufficient RAM for streaming resident windows')
    arrays=[mmap.mmap(-1,elements*4,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS) for _ in range(3)]
    retire=getattr(mmap,'MADV_FREE_REUSABLE',mmap.MADV_DONTNEED)
    peak=0;measurements={}
    try:
        for operation in ('addition','distance','dot','norm','finite','hash','adamw_ema'):
            times=[]
            for repeat in range(repeats):
                seconds=0.
                for start in range(0,elements,window):
                    size=min(window,elements-start)
                    values=[torch.frombuffer(a,dtype=torch.float32,count=size,offset=start*4) for a in arrays]
                    p,b,g=values;p.fill_(1.);b.fill_(.9);g.fill_(.01)
                    parameter=torch.nn.Parameter(p);parameter.grad=g
                    optimizer=None;teacher=None
                    if operation=='adamw_ema':
                        optimizer=torch.optim.AdamW([parameter],lr=1e-6,betas=(.9,.999),weight_decay=.01,foreach=False)
                        optimizer.state[parameter]=dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.ones_like(p))
                        teacher=p.clone()
                    peak=max(peak,psutil.Process().memory_info().rss)
                    require(psutil.virtual_memory().available>=2_000_000_000,'streaming benchmark available-memory floor reached')
                    clock=time.monotonic()
                    with torch.no_grad():
                        if operation=='addition':add_in_place([parameter],[b],.01)
                        elif operation=='distance':distance([p],[b])
                        elif operation=='dot':gradient_dot([parameter],[b])
                        elif operation=='norm':norm_squared([g])
                        elif operation=='finite':finite([g])
                        elif operation=='hash':tensor_hash([p])
                        else:
                            optimizer.step();teacher.lerp_(parameter,.01)
                            copied=parameter.detach().clone();bool(torch.isfinite(teacher).all());del copied
                    seconds+=time.monotonic()-clock
                    del optimizer,teacher,parameter,p,b,g,values
                    gc.collect()
                    for a in arrays:a.madvise(retire)
                times.append(seconds)
                print('[progress] '+json.dumps(dict(operation=operation,repeat=repeat+1,seconds=seconds)),flush=True)
            measurements[operation]=dict(seconds=times,median_seconds=statistics.median(times))
    finally:
        for a in arrays:a.close()
    attempt=json.loads(ATTEMPT3.read_text())
    return dict(schema='rung-p2-streaming-benchmark.v2',threads=8,measured_elements=elements,
        projected_elements=TARGET_ELEMENTS,distinct_arrays=3,bytes_per_array=elements*4,
        array_method='distinct anonymous mappings; sequential nonoverlapping 256 MiB windows; pages retired after timing',
        resident_peak_bytes=peak,window_elements=window,repeats=repeats,
        attempt3_sha256=sha(ATTEMPT3),source_sha256=sha(Path(__file__)),measurements=measurements,
        forecast=corrected_forecast(measurements,elements,attempt))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=run();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
