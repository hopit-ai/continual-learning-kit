#!/usr/bin/env python3
"""Seal whole-month predictions before releasing the frozen collector's checked traces."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *


def seal(out,month):
    n=MONTHS.index(month)+1; records={}
    for arm,seed,mode in arrival_specs():
        root=score_root(out,'arrival',arm,seed,n,month,mode)
        frozen_files(root)
        records[root.name]=s.file_hash(root/'manifest.json')
    value=dict(month=month,predictions=records,context_sha256=s.file_hash(out/'contexts'/month/'manifest.json'))
    bind(out/'seals'/(month+'.json'),value)
    return value


def admission(item,response,tok):
    reasons=[]; text=response['response']; sc=item['sec']
    if sc['issuer_holdout'] or not sc.get('training_allowed',False): reasons.append('holdout_or_training_excluded')
    if response['finish_reason'] not in ('stop','eos'): reasons.append('unfinished')
    if not 0<response['output_tokens']<=2048: reasons.append('response_cap')
    if len(text.strip())>2500: reasons.append('character_cap')
    if not finqa.is_correct(finqa.extract_answer(text),finqa.gold_of(item)): reasons.append('historical')
    if not sec.v4_explicit_correct(item,text): reasons.append('explicit_unit')
    # The immutable audited item's displayed cells, XBRL and arithmetic must still bind.
    try: sec.check_v4_item(item,finqa)
    except (AssertionError,ValueError,KeyError): reasons.append('source_unit_provenance')
    pids,tids=t.tokenize_student(tok,[dict(role='user',content=finqa.render_prompt(item))],text)
    if len(tids)>2048 or len(pids)>10240: reasons.append('eos_inclusive_cap')
    return reasons


def job(a):
    doc=binding(a.out,a.standin); month=a.month
    require(month in MONTHS,'unregistered month')
    previous=ledger(a.out,MONTHS.index(month))
    sealed=seal(a.out,month)
    root=a.out/'ledger'/month
    if (root/'manifest.json').exists(): frozen_files(root);print('SKIP: ledger',month);return
    frozen=score_root(a.out,'arrival','frozen',0,MONTHS.index(month)+1,month,'off')
    raw=t.read_jsonl(frozen/'raw/responses.jsonl'); lookup={r['id']:r for r in raw}
    batch=items(doc,month)
    require(len(raw)==len(lookup)==len(batch) and set(lookup)=={i['id'] for i in batch},'collector coverage')
    tok=tokenizer(doc,a.standin);traces=[];decisions=[]
    for i in batch:
        r=lookup[i['id']];reasons=admission(i,r,tok)
        decisions.append(dict(id=i['id'],admitted=not reasons,reasons=reasons))
        if reasons: continue
        traces.append(dict(id=i['id'],month=month,item=i,response=r['response'],
            source=i['filename'],date=i['sec']['filing_date'],collector_checkpoint=doc['base_identity'],
            item_sha256=s.digest(i),response_sha256=t.sha(r['response'].encode()),raw_sha256=s.digest(r),
            collector_manifest=s.file_hash(frozen/'manifest.json'),verifier=doc['protocol'],
            provenance=dict(issuer=i['sec']['issuer_ciks'],required_accessions=i['sec']['required_accessions'],
                required_filing_groups=i['sec']['required_filing_groups'],unedited=True)))
    root.mkdir(parents=True,exist_ok=True)
    t.write_jsonl(root/'traces.jsonl',traces);t.write_jsonl(root/'admission.jsonl',decisions)
    atomic(root/'sealed.json',sealed)
    freeze(root,['traces.jsonl','admission.jsonl','sealed.json'],admitted=len(traces),
           previous_ledger_hash=s.digest(previous),technical_synthetic=a.standin)
    print('ledger',month,':',len(traces),'admitted')


def configure(p): p.add_argument('--month',choices=MONTHS,required=True)
if __name__=='__main__': raise SystemExit(entry(job,configure,'a4-ledger-{month}'))
