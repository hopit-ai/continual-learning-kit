"""Mechanical pre-send validation of the frozen ARC prompt probe, never inference."""
import json
from pathlib import Path
from kit.arc_prompt_check import SCHEMA,MODEL,INITIAL_8B_REVISION,COUNT,code_hashes
from kit.next_pools import ARC_PROMPT,POOL_PROTOCOL
from kit.next_hellaswag import strict_label
from kit.m2_format import POLICY,policy
from kit.p3b_common import require,NotReady,s

FROZEN=Path(__file__).resolve().parent/'resources/m2/arc-prompt-FROZEN.json'
REVIEWED_FROZEN_SHA256='6ce5528c1f4cd2d75fca3f98aaadcb6f6f188dd21cfd8a912886d36e65ef8989'

ATTEMPT2=FROZEN.with_name('arc-prompt-attempt2.json')
REVIEWED_ATTEMPT2_SHA256='5512e14a95371c8864973b5a4a7b25686c1b8e2694421b5f2824fae36cb59138'


def validate_frozen(path=FROZEN):
    """Bind the manager-reviewed passing record merged from main, byte for byte.

    This is the explicitly authorised attempt-2 acceptance record. Its historical
    code/diagnostic metadata is not a claim that today's GPU mapper was executed.
    Subsequent training still regenerates exclusions and the pool prospectively.
    """
    from kit.m2_paths import local
    path=local(path)
    if not path.is_file():raise NotReady('reviewed FROZEN ARC acceptance record required before send')
    require(s.file_hash(path)==REVIEWED_FROZEN_SHA256,'ARC FROZEN record differs from reviewed passing record')
    record=json.loads(path.read_text());thresholds=policy()['arc']
    require(s.file_hash(ATTEMPT2)==REVIEWED_ATTEMPT2_SHA256,'ARC attempt2 payload differs from frozen acceptance')
    probe=json.loads(ATTEMPT2.read_text());ids=probe['candidate_ids']
    require(len(ids)==len(set(ids))==COUNT and ids==[r['id'] for r in probe['selection']['rows']], 'ARC attempt2 candidate identity/coverage differs')
    require(probe['passed'] and probe['completed'] and all(probe[k]==record[k] for k in ('n','parse_count','accepted_count','candidates_sha256','sample_sha256','model_revision')), 'ARC attempt2 differs from FROZEN')
    require(record['frozen_arc_prompt']==ARC_PROMPT and record['model_revision']==INITIAL_8B_REVISION,
            'ARC prompt/base differs from reviewed FROZEN acceptance')
    require(record['n']==thresholds['n'] and record['parse_count']>=thresholds['min_strict_parseable'] and
            record['accepted_count']>=thresholds['min_accepted'],'ARC FROZEN acceptance thresholds failed')
    return dict(record_sha256=REVIEWED_FROZEN_SHA256,format_validity_sha256=s.file_hash(POLICY),
                candidates_sha256=record['candidates_sha256'],sample_sha256=record['sample_sha256'],
                attempt2_sha256=REVIEWED_ATTEMPT2_SHA256,candidate_ids=ids,
                parse_count=record['parse_count'],accepted_count=record['accepted_count'],source='reviewed-FROZEN')


def validate(path):
    from kit.m2_paths import local
    if path is None or not local(path).is_file():raise NotReady('passing frozen ARC prompt check required before send')
    path=local(path);record=json.loads(path.read_text());plan=record['selection'];thresholds=policy()['arc']
    require(plan.get('kit_code_sha256')==code_hashes(),'ARC code identities changed after passing check')
    require(record.get('gpu_uuid_asserted') is True and record.get('gpu_mapping',{}).get('uuid','').startswith('GPU-'),'ARC in-process GPU UUID assertion missing')
    require(record.get('schema')==SCHEMA and record.get('model')==MODEL and record.get('model_revision')==INITIAL_8B_REVISION,'ARC base/generation identity differs')
    require(record.get('format_validity_sha256')==s.file_hash(POLICY),'ARC acceptance policy changed')
    require(plan['arc_prompt']==ARC_PROMPT and plan['pool_protocol']==POOL_PROTOCOL,'ARC prompt differs from frozen check')
    require(record.get('generation')==dict(seed=0,temperature=0,top_p=1,max_tokens=2048,thinking=False,generator='kit.v4_teacher.VLLMGenerator',tensor_parallel_size=1,allow_batch_fallback=False),'ARC generation path differs')
    rows=record.get('outcomes',[]);candidates=plan['rows']
    require(len(candidates)==COUNT and len({r['id'] for r in candidates})==COUNT,'ARC frozen candidate population changed')
    require([r['id'] for r in rows]==[r['id'] for r in candidates],'ARC responses missing/foreign/reordered')
    parse_count=accepted=0
    for response,candidate in zip(rows,candidates):
        parsed=strict_label(response['response'],candidate['labels'])
        finished=response['finish_reason'] in ('stop','eos')
        valid=finished and parsed is not None
        correct=parsed==candidate['gold']
        checked=finished and correct and response.get('encoding_valid') is True
        require(response['parsed']==parsed and response['correct']==correct and response['accepted']==checked,'ARC checks differ from unchanged strict verifier')
        parse_count+=valid;accepted+=checked
    require(record.get('n')==COUNT and record.get('parse_count')==parse_count and record.get('accepted_count')==accepted,'ARC counters differ')
    require(plan['inputs'].get('candidate_preparation',{}).get('fresh_exclusions_rerun') is True,'fresh registered exclusions required; inherited diagnostic candidates cannot qualify send')
    require(plan['inputs'].get('technical_synthetic') is False and all(k in plan['inputs'] for k in ('panels','generic','earlier_receipts','sources')),'ARC registered exclusion identities missing')
    if not record.get('completed') or not record.get('passed') or parse_count<thresholds['min_strict_parseable'] or accepted<thresholds['min_accepted']:
        raise NotReady('ARC acceptance failed; same 128 candidates required; no verifier relaxation/replenishment')
    return dict(record_sha256=s.file_hash(path),format_validity_sha256=s.file_hash(POLICY),candidates_sha256=plan['candidates_sha256'],
                candidate_ids=[r['id'] for r in candidates],parse_count=parse_count,accepted_count=accepted)


def validate_candidate_membership(path,acceptance=None):
    """Compare ARC IDs only; the real broad file begins with independent GSM8K rows."""
    acceptance=acceptance or validate_frozen()
    path=Path(path);inputs=json.loads(path.with_name('inputs.json').read_text())
    require(inputs.get('candidate_preparation',{}).get('fresh_exclusions_rerun') is True,
            'fresh registered exclusions must be rerun before broad generation')
    candidates=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    arc=[r for r in candidates if r['source']=='arc']
    if [r['id'] for r in arc[:COUNT]]!=acceptance['candidate_ids']:
        raise NotReady('fresh first 128 ARC candidates differ from frozen attempt2; broad PENDING, no replenishment')
    require(len({r['id'] for r in arc})==len(arc), 'duplicate ARC candidate IDs')
    return dict(arc_candidate_ids=acceptance['candidate_ids'],fresh_candidates_sha256=s.file_hash(path),
                attempt2_sha256=acceptance['attempt2_sha256'])
