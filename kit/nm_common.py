"""Frozen next-message constants, immutable artifacts and compact receipts."""
from pathlib import Path
import hashlib
import json
import math
import os
import statistics
from decimal import Decimal, ROUND_HALF_UP

KIT=Path(__file__).resolve().parent
ROOT=KIT.parent
SEEDS=(101,102,103)
FAMILIES=('v2b','opr','fisher','isdft','freeze','grpo')
GENERAL=('instruction','maths','knowledge')
PANELS=(*GENERAL,'hellaswag','finqa','legal')
STEPS=(0,10,20,30,40)
BASE='b968826d9c46dd6066d109eabc6255188de91218'
TWO=4.302653
ONE=2.919986
PENDING=75
SCHEMA='next-message.v1'

class Pending(Exception):pass
class Invalid(ValueError):pass

def need(value,reason):
    if not value:raise Invalid(reason)

def ready(value,reason):
    if not value:raise Pending(reason)

def raw_hash(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):return raw_hash(Path(path).read_bytes())
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return raw_hash(canonical(value))
def load(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_bytes(canonical(value)+b'\n');temporary.replace(path)

def seal(path,value):
    path=Path(path)
    if path.exists():need(load(path)==value,'sealed artifact changed: '+str(path))
    else:atomic(path,value)
    return value

def checked(ref):
    need(isinstance(ref,dict) and set(ref)>={'path','sha256'},'hash-bound input required')
    path=Path(ref['path']);ready(path.is_file(),'missing input: '+str(path))
    need(sha(path)==ref['sha256'],'input digest changed: '+str(path))
    return load(path)

def stats(values,t=TWO):
    need(len(values)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in values),'exactly three finite paired seeds required')
    mean=statistics.mean(values);sd=statistics.stdev(values);radius=t*sd/math.sqrt(3)
    return dict(values=values,n=3,df=2,t=t,mean=mean,sd=sd,lower=mean-radius,upper=mean+radius,positive=sum(v>0 for v in values))

def benefit(values):
    row=stats(values);row['pass']=row['mean']>=2 and row['lower']>0 and row['positive']==3
    return row

def guard(values,margin=-1,t=TWO):
    row=stats(values,t);row.update(margin=margin,**{'pass':row['lower']>=margin});return row

def money(value):return str(Decimal(str(value)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))

INHERITED=('sdft_actor.py','sdft_objective.py','sdft_verl.py','v4_contract.py','v4_state.py',
    'v4_timing.py','v4_restore_check.py','m3_loss.py','m3_actor.py','m3_runner.py','p2_l2.py',
    'x_p2_actor.py','m3_manifest.py','m2_gpu.py','m2_assignment.py','lane_pool.py','util_watch.py',
    'inference_lifecycle.py','hellaswag_reference.py','m3_score.py','ref_score_job.py','score_forgetting.py',
    'scorers.py','c_common.py','c_score.py','c_reward.py','a4_memory.py','a4_ledger.py','a4_common.py','report_text.py','text_return.py',
    'a4_sec.py','a4_reward.py','eval_bed.py','beds/finqa.py','resources/p3b/v4-recipes.json')

def source_seal():
    paths=[*KIT.glob('nm_*.py'),*KIT.glob('cr_*.py'),KIT/'m3_mode.py',*(KIT/'jobs/nm').glob('*.sh'),*(KIT/'resources/nm').glob('*.json'),*(KIT/'resources/nm/registrations').glob('*.md')]
    paths.extend(KIT/name for name in INHERITED)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}

def launch_config(cfg,family):
    # Future families may receive new returns without reopening an already bound row.
    common={k:v for k,v in cfg.items() if k not in ('inputs','budget')}
    budget=dict(cfg.get('budget',{}));bounds=budget.get('row_max_seconds',{})
    budget['row_max_seconds']={k:v for k,v in bounds.items() if k.startswith('NM-'+family+'-')}
    common.update(input=cfg.get('inputs',{}).get(family),budget=budget,family=family)
    return common


def text_lines(family,result,authority):
    lines=['T|AUTH|'+str(authority),f"NM|{family}|{result['status']}"]
    if result.get('primary'):
        p=result['primary'];lines.append(f"NM|{family}|mean={p['mean']:.3f}|LB={p['lower']:.3f}|pos={p['positive']}/3")
    if result.get('interpretation'):lines.append(f"NM|{family}|{result['interpretation']}")
    need(all(len(line)<=90 for line in lines),'text return exceeds 90 characters')
    return lines
