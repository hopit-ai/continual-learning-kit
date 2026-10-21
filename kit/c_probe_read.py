#!/usr/bin/env python3
"""C after-FinQA replay breadth; reuse next_read's factorial paired admission logic."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.c_common import *
from kit.p3b_read import load
from kit.next_read import analyse


def job(a):
    rows, files = load(a.source)
    docs = [json.loads(raw) for name,raw in files.items() if name.endswith('binding.json')]
    require(len(docs) == 1, 'exactly one C binding required')
    doc = docs[0]
    require(doc['protocol'] == protocol() and doc['technical_synthetic'] == a.standin, 'C protocol/type changed')
    mapped = []
    for row in rows:
        arm = row['arm']
        if row['panel'] == 'panel': continue
        if arm == 'base':
            mapped.append(dict(row,seed=0,panel='contract_nli' if row['panel']=='test' else row['panel']))
        else:
            matches = [p for p in ('generic','broad','earlier','both') if arm == probe_arm(doc,p)]
            if matches: mapped.append(dict(row,arm='v4-'+matches[0],panel='contract_nli' if row['panel']=='test' else row['panel']))
    from kit.c_score import COLUMNS
    mapped=[{key:str(r.get(key,'')) for key in COLUMNS} for r in mapped]
    mapped.sort(key=lambda r:(r['arm'],int(r['seed']),int(r['step']),r['panel'],r['question_id']))
    required={(f'v4-{arm}',seed,step,panel) for arm in ('generic','broad','earlier','both') for seed in SEEDS
        for panel in ('contract_nli','finqa','ifeval','knowledge','math','hellaswag')
        for step in (STEPS if panel=='contract_nli' else RETENTION_STEPS)}
    observed={(r['arm'],int(r['seed']),int(r['step']),r['panel']) for r in mapped}
    if not required <= observed: raise NotReady('C probe scoring incomplete: '+str(len(required-observed))+' cells')
    result = analyse(mapped,a.bootstrap,acquisition='contract_nli')
    result.update(technical_synthetic=a.standin, acquisition='ContractNLI test-reserve tolerant final and gain',
        start_policy=doc['start_policy'], source_sha256=s.digest(mapped))
    atomic(a.out/'reading-probe.json',result)
    print('C probe:',len(result['v4_curves']),'curves;',result['v4_nomination'])


def configure(p):
    p.add_argument('--source',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=10000)
if __name__ == '__main__': raise SystemExit(entry(job,configure,'c-probe-read'))
