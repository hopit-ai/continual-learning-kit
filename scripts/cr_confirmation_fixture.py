"""CPU-only, real-shaped inputs for the CR native FinQA regeneration tests."""
from pathlib import Path
import os
from unittest.mock import patch
from kit.nm_common import atomic, load, sha
from kit import cr_confirmation as cr, cr_finqa


def build(directory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kit import v4_campaign_ops as b, v4_sft, m3_pools
    from kit.p3b_common import t, cycle, files_manifest
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    work=directory/'b';base=b.paths(work,'main');(base/'common').mkdir(parents=True)
    response='The report provides the quantities and their arithmetic gives the required total.\nAnswer: 7'
    common=[dict(prompt=[dict(role='user',content=f'Report fixture {i}: compute its total.')],
        data_source='finqa',reward_model=dict(style='rule',ground_truth='7'),
        extra_info=dict(index=f'q{(i*37)%67}',split='train',demonstration=response,rewrite=response,
            problem='fixture arithmetic',description='fixture report',teacher_model='Qwen3.6-27B',stand_in_teacher=False)) for i in range(67)]
    corpus=base/'common/finqa.parquet';pq.write_table(pa.Table.from_pylist(common),corpus)
    atomic(base/'common/manifest.json',dict(schema='kit-v4-common-training.v1',teacher_models=['Qwen3.6-27B'],
        stand_in_teacher=False,coverage=dict(scientific_phase_allowed=True),files=dict(finqa=dict(sha256=sha(corpus)))))
    test=work/'v4/datasets/datasets/v4_finqa/test.parquet';test.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([dict(fixture=True)]),test)
    tokenizer=cr_finqa.StandinTokenizer()
    with patch.object(v4_sft,'load_student_tokenizer',lambda *args:tokenizer),patch.dict(os.environ,QWEN3_8B_TOKENIZER='CPU-STANDIN'):
        for seed in (101,102,103,104,105):b.schedule(work,'main',f'D-finqa-alone-s{seed}')
    # Original v6a copier, including its verifier and JSONL writer. Only the
    # release-manifest lookup is substituted for this explicitly local fixture.
    folders=[base/f'scheduled/finqa-s{seed}' for seed in (101,102,103)]
    with patch('kit.m3_manifest.validate_input'):
        schedules,_=m3_pools.finqa_task_schedules(folders,False,tokenizer)
    v6a=directory/'v6a';v6a.mkdir()
    old={}
    for seed,rows in schedules.items():
        path=v6a/f'schedule-{seed}.jsonl';t.write_jsonl(path,rows);old[str(seed)]=str(path)
    sources=[]
    def add(path,role='schedule-sources',**kw):
        path=Path(path)
        if not any(r['path']==str(path) for r in sources):sources.append(dict(path=str(path),role=role,sha256=sha(path),**kw))
        return str(path)
    for role in sorted(cr.ROLES):
        path=directory/(role+'.json');atomic(path,dict(path='m3-fallback') if role=='math-mode' else {})
        add(path,role)
    paths={r['role']:r['path'] for r in sources}
    finqa=dict(algorithm=cr_finqa.PROTOCOL,b_work=str(work),corpus=str(corpus),v6a_schedules=old)
    for path in (*cr_finqa.native_paths(finqa),*map(Path,old.values())):add(path)
    tasks=dict(finqa=finqa)
    for stage in ('contract_nli','math_hard'):
        path=directory/(stage+'-corpus.jsonl')
        rows=[dict(id=f'{stage}-{i}',prompt='Fixture task '+str(i),response=response,gold='7') for i in range(67)]
        t.write_jsonl(path,rows);add(path)
        old={}
        for seed in (101,102,103):
            ordered=directory/f'{stage}-{seed}.jsonl';t.write_jsonl(ordered,cycle(rows,seed,1280));old[str(seed)]=add(ordered)
        tasks[stage]=dict(corpus=str(path),v6a_schedules=old,algorithm='seed-shuffle-cycle-v1')
    # Valid replay minima; direct unique prompts keep the scheduling audit useful.
    pools=directory/'pools';pools.mkdir();names=[]
    from kit import m3_schedule
    for domain,n in dict(instruction=960,gsm8k=80,arc=80,finqa=320,contract_nli=320).items():
        name=domain+'.jsonl';t.write_jsonl(pools/name,[dict(id=f'{domain}-{i}',prompt=f'{domain} question {i}',response=response) for i in range(n)]);names.append(name)
    atomic(pools/'availability.json',{a:dict(status='READY') for a in ('G','CR')});names.append('availability.json')
    atomic(pools/'manifest.json',files_manifest(pools,names,technical_synthetic=True))
    for name in (*names,'manifest.json'):add(pools/name,'pools')
    model=directory/'model';atomic(model/'config.json',{})
    from kit import m3_train
    from types import SimpleNamespace as NS
    base_out=directory/'base-source';m3_train.bind_base(NS(out=base_out,model=model,receipt=None,standin=True))
    base_binding=add(base_out/'inputs/base.json','references')
    native=directory/'native.json';atomic(native,dict(nodes=[dict(node='CPU-STANDIN',gpus=[dict(uuid='CPU-STANDIN',environment='CPU-STANDIN')])],message2_nodes=[]));add(native,'scorers')
    # Retain the original hard-MATH training pool identity for the defence-in-depth check.
    from kit.m2_common import M,hard_data
    math=directory/'targets-math';math.mkdir();pool=hard_data()['train_pool.jsonl'][0]
    atomic(math/'inputs.json',dict(pool=sha(M/'train_pool.jsonl')));t.write_jsonl(math/'pool.jsonl',[pool]);math_names=['inputs.json','pool.jsonl']
    for seed in (101,102,103):
        name=f'schedule-{seed}.jsonl';t.write_jsonl(math/name,[pool]*1280);math_names.append(name)
    atomic(math/'manifest.json',files_manifest(math,math_names,population=500,technical_synthetic=True))
    for name in (*math_names,'manifest.json'):add(math/name,'math-targets')
    hellaswag=directory/'hellaswag';hellaswag.mkdir()
    t.write_jsonl(hellaswag/'panel.jsonl',[dict(id=f'hellaswag-fixture-{i}',prompt=f'Technical fixture {i}',gold='ABCD'[i%4]) for i in range(100)])
    atomic(hellaswag/'manifest.json',files_manifest(hellaswag,['panel.jsonl'],technical_synthetic=True,panel_sha256=sha(hellaswag/'panel.jsonl')))
    for name in ('panel.jsonl','manifest.json'):add(hellaswag/name,'panels')
    doc=dict(kit_commit=cr.BASELINE,hard_math_mode='m3-fallback',sources=sources,task_sources=tasks,
        pools_manifest=str(pools/'manifest.json'),base_binding=base_binding,hellaswag_manifest=str(hellaswag/'manifest.json'),
        native_inventory=str(native),math_source=paths['math-mode'])
    inventory=directory/'node-inventory.json';atomic(inventory,doc)
    return dict(out=str(directory/'confirmation'),v6a_root=str(v6a),
        six_row_root=str(directory/'root-A'),v6b_root=str(directory/'v6b'),hosts=['CPU-STANDIN'],
        realised_inputs=dict(path=str(inventory),sha256=sha(inventory)),technical_synthetic=True),inventory
