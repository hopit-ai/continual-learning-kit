"""Frozen split-smoke resources after the registered 25 October CPU probe."""
from pathlib import Path
import hashlib,json
from kit.s_common import ROOT,require

WRITES_COUNT_AS_MEMORY=False
MEMORY_GIB=64
PREPARATION_MODEL_ONLY=False
TIMEOUT_SECONDS=1800
CPUS=8
TOTAL_CEILING_USD=round(1830*(2*.001097+CPUS*.00003942+MEMORY_GIB*.00000667)*1.25+.05+3,2)
PROBE_SHA256='2ef76f1ea98cd9215ca80b3e251ddf55bb45e0b21db901a2428e2244427bfb5f'
PART_STEMS={'p2':'x-integration-p2-smoke','resume':'x-integration-resume-smoke'}
AUTHORISED_ATTEMPTS={'p2':5,'resume':1}
# Frozen owner row: matching a mention or a different authority is insufficient.
ATTEMPT4_LEDGER_ROW=('2026-10-26','A3','x-integration-p2-smoke attempt 4',
    'Final P2 smoke: attempt 3 repeated with a 50-min limit and no intermediate step-2 saves; paired batches; aggregate-cgroup gate; one attempt, no retries',
    'Owner, 26 Oct: \"Attempt 4: 50-min limit, ≤$15 (Recommended)\"','≤$15','pending')
ATTEMPT5_LEDGER_ROW=('2026-10-29','A3','x-integration-p2-smoke attempt 5',
    'Fresh part-P2 smoke on current combined-send code; same design as attempt 4; 50-min limit; paired batches; aggregate-cgroup gate; one attempt, no retries',
    'Owner, 29 Oct: "Yes, run it (<=$15)"','≤$15','pending')
ATTEMPT5_STAGE_AUDIT=ROOT/'docs/phase3/evidence/p2-cpu-stage-audit-attempt5-20261029.json'
ATTEMPT4_TIMEOUT_SECONDS=3000
ATTEMPT4_CEILING_USD=round(3030*(2*.001097+CPUS*.00003942+MEMORY_GIB*.00000667)*1.25+.05+3,2)
ATTEMPT5_TIMEOUT_SECONDS=ATTEMPT4_TIMEOUT_SECONDS
ATTEMPT5_CEILING_USD=ATTEMPT4_CEILING_USD
assert TOTAL_CEILING_USD<=10 and ATTEMPT4_CEILING_USD<=15 and ATTEMPT5_CEILING_USD<=15


def resources(part,attempt):
    require(part in PART_STEMS and 1<=attempt<=AUTHORISED_ATTEMPTS[part],'unauthorised smoke resources')
    if part=='p2' and attempt==4:return dict(timeout_seconds=ATTEMPT4_TIMEOUT_SECONDS,total_ceiling_usd=ATTEMPT4_CEILING_USD)
    if part=='p2' and attempt==5:return dict(timeout_seconds=ATTEMPT5_TIMEOUT_SECONDS,total_ceiling_usd=ATTEMPT5_CEILING_USD)
    return dict(timeout_seconds=TIMEOUT_SECONDS,total_ceiling_usd=TOTAL_CEILING_USD)


def check_probe(path):
    """Bind the actual probe, including its unresolved automatic classifier.

    The registration resolves that missing-counter result from two completed
    20 GB writes in an 8 GiB container. Preserve the raw evidence unchanged.
    """
    value=json.loads(path.read_text())
    require(hashlib.sha256(path.read_bytes()).hexdigest()==PROBE_SHA256,'registered CPU memory probe evidence changed or unresolved')
    require(value.get('schema')=='modal-memory-probe.v1' and value.get('cpu_only') is True and value.get('gpus')==0 and
        value.get('memory_gib')==8 and value.get('returncode')==0 and value.get('status')=='FINISHED','completed registered CPU memory probe required')
    cases=value.get('decision',{}).get('cases',[])
    require(len(cases)==2 and len({c['directory'] for c in cases})==2 and all(c.get('deleted') is True and
        c.get('written_bytes')==c.get('requested_bytes')==20_000_000_000 for c in cases),'probe must complete and delete both 20 GB files')
    require(WRITES_COUNT_AS_MEMORY is False and MEMORY_GIB==64 and PREPARATION_MODEL_ONLY is False,'probe result freezes 64 GiB with full-state preparation')
    return value


def forecast(part,attempt=1):
    require(part in PART_STEMS,'part must be p2 or resume')
    value=json.loads((ROOT/'kit/resources/p2/split-forecast.json').read_text())
    if part=='p2' and attempt in (4,5):value=json.loads((ROOT/'kit/resources/p2/attempt4-forecast.json').read_text())
    basis=value['measured_basis']
    require(hashlib.sha256((ROOT/basis['path']).read_bytes()).hexdigest()==basis['sha256'],'forecast timing basis changed')
    plan=value['parts'][part]
    limit=45*60 if part=='p2' and attempt in (4,5) else 27*60
    require(0<plan['seconds']['central']<=plan['seconds']['upper_scenario']<=limit,'split forecast exceeds '+str(limit//60)+' minutes; report rather than cut checks')
    return dict(plan,measured_basis=basis,assumptions=value['assumptions'],forecast_limit_seconds=limit)


def check_stage_audit(path=None):
    """Final attempt authority is conditional on this code's complete CPU audit."""
    from kit.p2_smoke import code_hashes
    path=path or ROOT/'docs/phase3/evidence/p2-cpu-stage-audit-20261025.json'
    require(path.is_file(),'attempt 3/4/5 requires passing CPU stage audit before reserving an attempt')
    value=json.loads(path.read_text())
    require(value.get('schema')=='p2-cpu-stage-audit.v1' and value.get('passed') is True,'CPU stage audit failed or missing')
    require(value.get('code_sha256')==code_hashes(),'CPU stage audit code changed; rerun before attempt 3/4/5')
    require(value.get('test_sha256')==hashlib.sha256((ROOT/'tests/test_x_smoke_stage_audit.py').read_bytes()).hexdigest(),'CPU stage audit tests changed')
    require(set(value.get('stages',{}))=={'preparation','measure','control','p2','prefix','resume'} and
        all(r.get('batches')==4 and r.get('task_per_batch')==28 and r.get('replay_positions')==[8,16,24,32] for r in value['stages'].values()),
        'CPU audit must qualify four batches of every dispatched smoke stage')
    return value
