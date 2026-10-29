"""Smoke-only inputs using the production generic D+replay slot composition."""
from pathlib import Path
from kit.s_common import atomic,require
from kit.p3b_common import mixed
from kit.next_slots import MAPS
from kit.v4_contract import BATCH_PROMPTS


def source_rows(finqa,preparation=False):
    require(bool(finqa),'smoke requires frozen eligible FinQA rows')
    if preparation:
        from kit.beds.finqa import ANSWER_LINE
        task=[]
        for i in range(1280):
            row=finqa[i%128];answers=list(ANSWER_LINE.finditer(row['response']))
            require(bool(answers),'frozen FinQA smoke target has no Answer line')
            # The frozen short target "Answer: yes/no" also occurs literally
            # in FinQA's answer-format instruction. Preserve it verbatim inside
            # a labelled demonstration so D's prompt-cleanliness guard holds.
            task.append(dict(row,id='prepare-'+str(i),p3b_replay=False,data_source='finqa',
                response='Frozen FinQA training target:\n'+row['response'],
                reward_model=dict(style='finqa',ground_truth=answers[-1].group(1).strip())))
    else:
        task=[dict(id='development-'+str(i),prompt=[dict(role='user',content=
            'Development legal stand-in. Clause: Either party may terminate with notice. Claim: notice is permitted. Is it entailed?')],
            response='Entailment',gold='entailment',p3b_replay=False) for i in range(1280)]
    source='finqa' if preparation else 'contract_nli'
    replay=[dict(r,p3b_replay=True,extra_info=dict(r.get('extra_info') or {},p3b_task_source=source)) for r in finqa[:128]]
    return task,replay


def converted_rows(tokenizer,finqa,preparation=False):
    task,replay=source_rows(finqa,preparation)
    from kit.c_d_prompt import d_prompt_eligible
    from kit.v4_teacher import tokenize_student
    skipped=[str(r['id']) for r in replay if not d_prompt_eligible(tokenize_student(tokenizer,r['prompt'],r['response'])[0])]
    # Native C D eligibility: shuffle the full pool, skip over-cap draws, then
    # fill the registered generic slots. Never shuffle just the survivors.
    scheduled=mixed(task,replay,101,skip_ids=skipped)
    if preparation:
        from kit.p3b_train import training_rows
        converted=training_rows(scheduled,tokenizer,'D')
    else:
        from kit.c_train import training_rows
        converted,_=training_rows(scheduled,tokenizer,'D+replay')
    for row in converted:
        row['extra_info'].update(technical_synthetic=True,p3b_task_source='finqa' if preparation else 'contract_nli')
    validate_composition(converted)
    return converted


def validate_composition(rows):
    positions=MAPS['generic']['generic']
    require(len(rows)==1280,'smoke schedule must retain 40 x 32 rows')
    for offset in range(0,len(rows),BATCH_PROMPTS):
        flags=[i+1 for i,r in enumerate(rows[offset:offset+BATCH_PROMPTS]) if r['extra_info'].get('p3b_replay')]
        require(flags==positions,'smoke D+replay batch must use production generic slots '+str(positions))
    return dict(slot_map=MAPS['generic'],task_per_batch=BATCH_PROMPTS-len(positions),replay_per_batch=len(positions),batches=40)


def write_data(path,rows):
    """The same exact parquet writer is used by the smoke and its CPU stage audit."""
    import pyarrow as pa,pyarrow.parquet as pq
    from kit import m2_common as m
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    composition=validate_composition(rows)
    for panel,values in (('train',rows),('test',rows[:1]),('val',rows[:32])):
        pq.write_table(pa.Table.from_pylist(values),path/(panel+'.parquet'))
    atomic(path/'manifest.json',m.files_manifest(path,['train.parquet','val.parquet','test.parquet'],
        rows=1280,technical_synthetic=True,composition=composition))
    return path
