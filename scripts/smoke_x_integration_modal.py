"""Prepared only. Real X hooks on two H100s, two separate parts, one container per part, no automatic retries.

    modal run --detach scripts/smoke_x_integration_modal.py --part p2
    modal run --detach scripts/smoke_x_integration_modal.py --part resume

The embedded distributed P2 mechanical receipt can qualify its smoke only.
"""
from pathlib import Path
import json,sys,time
import modal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smoke_common_modal import ROOT,partner_image,write_record
from kit.x_integration_smoke import reserve_attempt,MODEL_REVISION,verdict_scope
from kit.p_determinism import DETERMINISTIC_ENV
from kit.x_smoke_memory import SCOPE

app=modal.App('x-integration-smoke')
image=partner_image({}).add_local_file(ROOT/'scripts/smoke_x_integration_modal.py','/root/scripts/smoke_x_integration_modal.py',copy=True)
from kit.x_smoke_settings import CPUS,MEMORY_GIB,TIMEOUT_SECONDS,check_probe,forecast,PART_STEMS,PROBE_SHA256,AUTHORISED_ATTEMPTS,resources
GPU='H100:2'
HOURLY_USD=3600*(2*.001097+CPUS*.00003942+MEMORY_GIB*.00000667)
TOTAL_CEILING_USD=round(1830/3600*HOURLY_USD*1.25+.05+3,2)
assert TOTAL_CEILING_USD<=10

@app.local_entrypoint()
def main(part: str):
    if part not in PART_STEMS:raise ValueError("part must be p2 or resume")

    probe=check_probe(ROOT/'docs/phase3/evidence/modal-memory-probe.attempt1.json')
    ledger=ROOT/'docs/phase2/ledger.md'
    attempt,stem=reserve_attempt(ROOT/'docs/phase2/evidence',part,ledger=ledger.read_text())
    plan=forecast(part,attempt);budget=resources(part,attempt)
    print(json.dumps(dict(part=part,attempt=attempt,forecast=plan,**budget),sort_keys=True))
    started=time.monotonic();sandbox=None;record=None;partials=[]
    try:
        sandbox=modal.Sandbox.create('python','-m','kit.x_integration_smoke','--part',part,app=app,image=image,gpu=GPU,
            cpu=(CPUS,CPUS),memory=(MEMORY_GIB*1024,MEMORY_GIB*1024),timeout=budget['timeout_seconds'],
            env=dict(DETERMINISTIC_ENV,PYTHONPATH='/root:/root/SDPO',OMP_NUM_THREADS='8',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false'))
        for line in sandbox.stdout:
            print(line,end='')
            if '[record-partial] ' in line:partials.append(json.JSONDecoder().raw_decode(line.split('[record-partial] ',1)[1].lstrip())[0])
            if line.startswith('[record] '):record=json.loads(line[len('[record] '):])
        sandbox.wait()
        if record is None:record=dict(passed=False,steps=[],error='no integration evidence')
    except Exception as error:record=dict(passed=False,steps=[],error=type(error).__name__+': '+str(error))
    finally:
        if sandbox is not None:sandbox.terminate()
    record=verdict_scope(record,partials,part)
    record.update(partial_records=partials,attempt=attempt,evidence_stem=stem,maximum_attempts=AUTHORISED_ATTEMPTS[part],
        cumulative_ceiling_usd=round((attempt-1)*TOTAL_CEILING_USD+budget['total_ceiling_usd'],2),model_revision=MODEL_REVISION,schema='x-integration-smoke.v3',part=part,forecast=plan,technical_synthetic=True,scientific_evidence=False,
        seconds_in_function=time.monotonic()-started,total_ceiling_usd=budget['total_ceiling_usd'],
        memory_probe=probe['decision'],memory_probe_sha256=PROBE_SHA256,registered_probe_resolution=dict(writes_count_as_memory=False,host_measure=SCOPE,memory_gib=64,authority='amendment P2 resource paragraph: Corrected after the resume smoke, 25 October'),gpu=GPU,cpu_cores=CPUS,memory_gib=MEMORY_GIB,timeout_seconds=budget['timeout_seconds'],retries=0,create_calls=1)
    write_record(record,stem,hourly_usd=HOURLY_USD,attempt=attempt)
    if not record['passed']:raise SystemExit(1)
