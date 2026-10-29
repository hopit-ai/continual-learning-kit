"""Container-safe pre-send likelihood/reference probe and knowledge format audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from kit.smoke_steps import Steps
from kit.v4_contract import INITIAL_8B_REVISION

KIT=Path(__file__).resolve().parent
ITEMS=KIT/'resources/m2/hellaswag-validation-items.json'
LAUNCHER=KIT/'resources/m2/manager-format-launcher.txt'
MODEL='Qwen/Qwen3-8B'
SCHEMA='message2-presend-format-check.v2'
ROLES=('base',)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declaration():
    from kit.m2_format import policy,POLICY
    policy()
    return dict(policy_sha256=sha(POLICY),items_sha256=sha(ITEMS),
        knowledge_panel_sha256=sha(KIT/'panels/general-v1.jsonl'),model=MODEL,revision=INITIAL_8B_REVISION,
        incoming_checkpoints_validated='on-node-only',roles=list(ROLES),
        gpu_mapper_sha256=sha(KIT/'m2_gpu.py'),lifecycle_sha256=sha(KIT/'inference_lifecycle.py'),
        launcher_sha256=sha(LAUNCHER),subprocess_sha256=sha(KIT/'inference_subprocess.py'))


def _knowledge_audit(model):
    from kit.v4_teacher import VLLMGenerator
    from kit.scorers import extract_letter_strict,score
    from kit.inference_lifecycle import marker,shutdown_since
    rows=[json.loads(line) for line in (KIT/'panels/general-v1.jsonl').read_text().splitlines() if line.strip()]
    rows=[r for r in rows if r['panel']=='knowledge']
    if len(rows)!=100:raise ValueError('all frozen knowledge items required')
    records=[];mark=marker()
    try:
        generator=VLLMGenerator(str(model),2048,tensor_parallel_size=1,allow_batch_fallback=False)
        for row in rows:
            output=generator([dict(role='user',content=row['prompt'])],1,seed=0,temperature=0,top_p=1)
            text=output['text'];result=score('knowledge',text,row)
            records.append(dict(id=row['id'],response=text,finish_reason=output['finish_reason'],
                incorrect_format=int(extract_letter_strict(text) is None),correct=result['correct']))
    finally:shutdown_since(mark)
    failures=sum(r['incorrect_format'] for r in records)
    return dict(n=100,strict_format_failures=failures,incorrect_format_rate=failures/100,
        diagnostic_only=True,rows=records,complete=True)


def knowledge_audit(model):
    from kit.inference_subprocess import compute
    return compute('knowledge',model)


def run(work):
    from huggingface_hub import snapshot_download
    from kit.m2_format import likelihoods,validation,json_finite
    from kit import hellaswag_reference
    from kit.m2_gpu import prepare,assert_current
    from kit.p3b_common import identity
    declared=declaration();work=Path(work);work.mkdir(parents=True,exist_ok=False)
    model=Path(snapshot_download(repo_id=MODEL,revision=INITIAL_8B_REVISION,local_dir=str(work/INITIAL_8B_REVISION)))
    observed=identity(model,'student')
    mapping=prepare('0');assert_current()
    rows=json.loads(ITEMS.read_text())['rows'];checks=[];audits={}
    for role in ROLES:
        print('[progress] format validation '+role,flush=True)
        production=likelihoods(model,rows,model)
        reference=hellaswag_reference.score(str(model),rows,str(model))
        checks.append(dict(checkpoint=role,model_hashes=observed['model_file_hashes'],
            **validation(production,reference,[r['id'] for r in rows]),production=production,reference=reference))
        print('[progress] knowledge format audit '+role,flush=True)
        audits[role]=knowledge_audit(model)
    return json_finite(dict(schema=SCHEMA,declaration=declared,completed=True,passed=all(c['status']=='PASS' for c in checks),
        gpu_mapping=mapping,gpu_uuid_asserted=True,checks=checks,knowledge_audit=audits,
        diagnostic_only=True,scientific_evidence=False,incoming_checkpoints_validated='on-node-only',
        note='Frozen base only. Every real incoming checkpoint is validated on its assigned node; no incoming stand-in role is run.'))


def validate_receipt(path):
    from kit.p3b_common import require,NotReady
    from kit.m2_format import validation
    from kit.m2_paths import local
    if path is None or not local(path).is_file():raise NotReady('manager pre-send format-check receipt required')
    path=local(path);value=json.loads(path.read_text());expected=declaration()
    require(value.get('schema')==SCHEMA and value.get('declaration')==expected,'pre-send format-check declaration changed')
    if not value.get('completed') or not value.get('passed'):raise NotReady('pre-send likelihood technical validation PENDING')
    require(value.get('exitcode')==0 and value.get('max_containers')==value.get('create_calls')==1 and value.get('retries')==0 and value.get('timeout_cost_bound_usd',11)<=10,'pre-send container/price/exit receipt invalid')
    require(value.get('gpu_uuid_asserted') is True,'pre-send UUID assertion required')
    require([c['checkpoint'] for c in value['checks']]==list(ROLES),'frozen base-only coverage required')
    registered=json.loads((KIT/'resources/p3b/model-hashes.json').read_text())['student']['files']
    ids=[r['id'] for r in json.loads(ITEMS.read_text())['rows']]
    knowledge={r['id'] for r in (json.loads(l) for l in (KIT/'panels/general-v1.jsonl').read_text().splitlines() if l.strip()) if r['panel']=='knowledge'}
    for check in value['checks']:
        require(check['model_hashes']==registered,'pre-send model files differ')
        outcome=validation(check['production'],check['reference'],ids)
        require(outcome['status']=='PASS' and all(check.get(k)==v for k,v in outcome.items()),'pre-send likelihood technical outcome differs')
        audit=value['knowledge_audit'][check['checkpoint']];records=audit['rows']
        require(len(records)==100 and {r['id'] for r in records}==knowledge and audit.get('complete') is True,'knowledge audit coverage incomplete')
        from kit.scorers import extract_letter_strict
        failures=sum(extract_letter_strict(r['response']) is None for r in records)
        require(audit['strict_format_failures']==failures and audit['incorrect_format_rate']==failures/100,'knowledge audit diagnostics differ')
    return dict(record_sha256=sha(path),**expected,knowledge_audit='COMPLETE',source='manager-presend-technical-probe')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,default=Path('/tmp/hellaswag-format-check'))
    parser.add_argument('--import-check',action='store_true');args=parser.parse_args()
    if args.import_check:print('container import PASS');return 0
    steps=Steps({})
    try:result=run(args.work)
    except Exception as error:
        result=dict(schema=SCHEMA,completed=False,passed=False,error=type(error).__name__+': '+str(error))
    steps.done(**result)
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
