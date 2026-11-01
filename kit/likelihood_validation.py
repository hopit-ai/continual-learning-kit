"""Validation-only near-tie rule (1 Nov); the frozen scorers remain unchanged.

Scorers store LL/character. Pinned character and conditional-token counts recover
LL/token solely for these gap diagnostics; choices still use the original scores.
"""
import hashlib
import json
import math
from pathlib import Path

KIT=Path(__file__).resolve().parent
RESOURCE=KIT/'resources/m2'
RULE='likelihood-near-tie-20261101'
THRESHOLD=0.001
LEGACY_FIELDS=('complete_item_choice_coverage','finite_likelihoods','selected_choice_agreement','required_agreement','items','choices_per_item')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def amendment():
    path=RESOURCE/'likelihood-near-tie.json'
    frozen=json.loads((RESOURCE/'freeze.json').read_text())['files']
    if sha(path)!=frozen[path.name]:raise ValueError('near-tie amendment pin differs')
    value=json.loads(path.read_text())
    if value['rule']!=RULE or value['strict_gap_threshold']!=THRESHOLD or value['gap_unit']!='nats/token':raise ValueError('near-tie rule differs')
    for name,checksum in value['implementations'].items():
        if sha(KIT/name)!=checksum:raise ValueError('near-tie validation implementation differs')
    return value


def lengths():
    path=RESOURCE/'likelihood-token-lengths.json'
    frozen=json.loads((RESOURCE/'freeze.json').read_text())['files']
    if sha(path)!=frozen[path.name]:raise ValueError('validation token lengths pin differs')
    value=json.loads(path.read_text());policy=json.loads((RESOURCE/'format-validity.json').read_text())
    if value['tokenizer_sha256']!=policy['tokenizer_files'] or value['hellaswag_items_sha256']!=sha(RESOURCE/'hellaswag-validation-items.json') or value['knowledge_panel_sha256']!=sha(KIT/'panels/general-v1.jsonl'):
        raise ValueError('validation length provenance differs')
    return value['lengths']


def diagnostics(production,reference):
    table=lengths();ref={r['id']:r for r in reference};out=[]
    for row in production:
        other=ref[row['id']]
        choose=lambda scores:max(range(4),key=lambda i:(scores[i],-i))
        production_choice,reference_choice=choose(row['scores']),choose(other['scores'])
        disagree=production_choice!=reference_choice
        counts=table.get(row['id']);a=b=None
        if counts:
            chars,tokens=counts['characters'],counts['tokens']
            if len(chars)!=4 or len(tokens)!=4 or any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in chars+tokens):raise ValueError('invalid validation lengths')
            def gap(scores):
                values=sorted((v*c/t for v,c,t in zip(scores,chars,tokens)),reverse=True)
                return values[0]-values[1]
            a,b=gap(row['scores']),gap(other['scores'])
            if not all(math.isfinite(v) for v in (a,b)):raise ValueError('nonfinite validation gap')
        out.append(dict(id=row['id'],production_choice=production_choice,reference_choice=reference_choice,
            disagreeing_pair=[production_choice,reference_choice] if disagree else None,production_gap=a,reference_gap=b,gap_unit='nats/token',
            disagree=disagree,near_tie=disagree and a is not None and b is not None and a<THRESHOLD and b<THRESHOLD))
    return out


def matches_check(check,result,*,legacy=False):
    if check.get('validation_rule')==RULE:
        return all(check.get(k)==v for k,v in result.items())
    if not legacy:return False
    # Legacy summaries are checked under their original unanimous-choice rule;
    # the new outcome is independently recomputed from all original score rows.
    old_status=('PASS',) if result['selected_choice_agreement']==100 else ('PENDING','FAIL')
    return check.get('status') in old_status and all(check.get(k)==result[k] for k in LEGACY_FIELDS)
