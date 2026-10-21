"""Small, source-bound CPU fixtures, explicitly synthetic and independent of private data."""
from kit.a4_common import *
from kit.c_sim import b_rows
from kit.p3b_score import write_csv
from copy import deepcopy


def fixture(root,winner='F'):
    slice_dir=root/'slice';slice_dir.mkdir(parents=True,exist_ok=True)
    for n,month in enumerate(MONTHS,1):
        rows=[]
        for k in range(24):
            acc=f'fixture-{n}-{k}';page=acc+'/R1.htm';unit='US dollars'
            table=[['','Current','Prior'],['Operating cash flow','200','100'],['Operating income','100','100']]
            ops=[]
            for row,col in ((1,1),(2,1),(1,2),(2,2)):
                cell=table[row][col]
                ops.append(dict(label=table[row][0],shown_label=table[row][0],period=table[0][col],cell=cell,
                    value=float(cell),unit=unit,normalized_usd=cell,r_page=page,table_row=row,table_column=col,
                    accession=acc,concept='fixture_concept',context_id='fixture_context',context=dict(start='2025-01-01',end='2025-12-31')))
            formula='100*(a/b-c/d)';value,answer,program=sec.v4_compute(formula,ops)
            holdout=k>=12;issuer=f'issuer-{n}-{k}';family='cash_conversion_change' if k%2 else 'fcf_margin_change'
            sc=dict(version=4,template=family,operands=ops,excerpts=[dict(kind='cash_flow',r_page=page,accession=acc,
                filing_date=month+'-15',table=table,unit=unit,footnotes=[])],formula=formula,unrounded_gold=str(value),
                answer_unit='percentage points',issuer_holdout=holdout,training_allowed=not holdout,
                cik=issuer,issuer_ciks=[issuer],issuer_group=issuer,filing_group=acc,required_filing_groups=[acc],
                required_accessions=[acc],accession=acc,accepted=month+f'-{k+1:02} 00:00:00',filing_date=month+'-15',form='10-K')
            rows.append(dict(id=acc,filename=page,pre_text=[f'Synthetic company {issuer}.'],post_text=[],table=deepcopy(table),
                qa=dict(question=f'Synthetic {family} filing {n} row {k}: signed change in percentage points?',exe_ans=float(value),answer=answer,program=program),sec=sc))
        atomic(slice_dir/f'batch-{month}.json',rows)
    atomic(slice_dir/'manifest.json',dict(months=MONTHS,technical_synthetic=True))
    write_csv(root/'B/scores.csv',b_rows(winner))
    replay=root/'replay';replay.mkdir()
    t.write_jsonl(replay/'replay.jsonl',[dict(id=f'generic-{i}',prompt=f'Explain the unrelated botanical specimen genus {i}.',response='A generic checked response.') for i in range(12)])
    freeze(replay,['replay.jsonl'],technical_synthetic=True)
    return slice_dir,root/'B',replay
