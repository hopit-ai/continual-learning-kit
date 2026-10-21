"""Static incumbent retrieval and arrival-time recurrence; gold-blind rendered contexts."""
from kit.a4_common import *
from kit.eval_bed import render


def eligible(item, case):
    a,b = item['sec'], case['sec']
    return (not set(a['issuer_ciks']) & set(b['issuer_ciks']) and a['accession'] != b['accession']
        and a['filing_group'] != b['filing_group']
        and not set(a.get('required_accessions', [])) & set(b.get('required_accessions', []))
        and not set(a.get('required_filing_groups', [])) & set(b.get('required_filing_groups', [])))


def build(item, pool, tok):
    frozen = finqa.render_prompt(item)
    require(frozen.startswith(finqa.INSTRUCTION), 'unexpected SEC prompt')
    cases = sorted([r for r in pool if eligible(item,r['item'])],
        key=lambda r:(-sec.similarity(item,r['item']),r['id']))[:3]
    retrieved = len(cases)
    while True:
        block = '\n\n'.join(f'Example {n}\n'+sec.v4_case_text(r['item'],r['response'],finqa) for n,r in enumerate(cases,1))
        prompt = frozen if not cases else finqa.INSTRUCTION+'\n\n'+sec.MEMORY_HEADER+'\n\n'+block+frozen[len(finqa.INSTRUCTION):]
        length = len(tok.encode(render(tok,prompt),add_special_tokens=False))
        if length <= 10240: break
        require(bool(cases), 'current documents exceed input cap; never truncate')
        cases.pop()
    return dict(prompt=prompt, case_ids=[r['id'] for r in cases], cases_dropped=retrieved-len(cases), input_tokens=length)


def recurrence(item, pool):
    words = sec.label_words(item); months = set()
    for r in pool:
        other = sec.label_words(r['item'])
        j = len(words & other)/len(words | other) if words | other else 0
        if item['sec']['template'] == r['item']['sec']['template'] and j >= .5: months.add(r['month'])
    return dict(recurrence=len(months), stratum='high' if len(months)>=2 else 'low')


def archive(out, doc, month, standin):
    n = MONTHS.index(month)
    pool = ledger(out,n)
    root = out/'contexts'/month
    observed = dict(previous_ledgers={m:s.file_hash(out/'ledger'/m/'manifest.json') for m in MONTHS[:n]},
        batch=doc['slice_files'][f'batch-{month}.json'], protocol=doc['protocol'])
    bind(out/'context-identities'/(month+'.json'), observed)
    if (root/'manifest.json').exists(): frozen_files(root); return
    tok = tokenizer(doc,standin); rows=[]
    previous_issuers={cik for m in MONTHS[:n] for i in items(doc,m) for cik in i['sec']['issuer_ciks']}
    for i in items(doc,month):
        off=build(i,[],tok); on=build(i,pool,tok)
        rows.append(dict(id=i['id'], month=month, recurring_issuer=bool(previous_issuers & set(i['sec']['issuer_ciks'])),
            **recurrence(i,pool), off=off, on=on))
    root.mkdir(parents=True,exist_ok=True)
    t.write_jsonl(root/'prompts.jsonl',rows)
    freeze(root,['prompts.jsonl'],**observed)


def contexts(out, month, mode):
    root=out/'contexts'/month; frozen_files(root)
    return {r['id']:dict(r[mode],recurrence=r['recurrence'],stratum=r['stratum'],recurring_issuer=r['recurring_issuer']) for r in t.read_jsonl(root/'prompts.jsonl')}
