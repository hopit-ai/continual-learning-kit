"""FinQA-only preparation/replay using the registered pool selector and scorers."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
from kit import v4_teacher as t, v4_pool as pool, v4_contract as contract
from kit.v4_evidence import require, pool_items


def prepare(work,phase):
    import pyarrow as pa
    import pyarrow.parquet as pq
    work=Path(work);inputs=work/'v4/report-inputs';inputs.mkdir(parents=True,exist_ok=True)
    root=Path(os.environ['FINQA_ROOT']);freeze=pool.freeze_finqa(root)
    for split in ('train','test'):
        source=inputs/'sources/FinQA'/(split+'.json');source.parent.mkdir(parents=True,exist_ok=True)
        if source.exists():require(source.read_bytes()==(root/(split+'.json')).read_bytes(),'source changed')
        else:shutil.copyfile(root/(split+'.json'),source)
    items={s:{i['id']:i for i in t.finqa.load(root,s)} for s in ('train','test')}
    native={s:t.finqa.rows_for_trainer([items[s][i['id']] for i in freeze[s]],s) for s in items}
    shipping=work/'v4/datasets/datasets/v4_finqa';shipping.mkdir(parents=True,exist_ok=True)
    for split,rows in native.items():
        target=shipping/(split+'.parquet')
        if target.exists():require(pq.read_table(target).to_pylist()==rows,'shipping rows changed')
        else:pq.write_table(pa.Table.from_pylist(rows),target,compression='zstd',row_group_size=32)
    original=pool_items(native['train'],'finqa');t.check_pool_provenance(original)
    frozen=sorted(original,key=lambda i:t.sha('v4-fixed-probe|finqa|'+i['id']))[:20]
    for name,rows in (('finqa-pool',original),('finqa-probe',frozen),('finqa-panel',native['test'])):t.write_jsonl(inputs/(name+'.jsonl'),rows)
    # Keep the general panel's exact bytes, keyed by panel AND id.
    shutil.copyfile(Path(__file__).parent/'panels/general-v1.jsonl',inputs/'general-v1-panel.jsonl')
    t.atomic_json(inputs/'finqa-freeze.json',freeze)
    base=work/('v4/report-'+phase);base.mkdir(parents=True,exist_ok=True)
    t.write_jsonl(base/'finqa-pool.jsonl',original)
    t.atomic_json(inputs/'model-identities.json',{'teacher':t.local_identity(os.environ['TEACHER_MODEL_DIR']),
                  'initial':t.local_identity(os.environ['MODEL_DIR'],'Qwen/Qwen3-8B')})
    for name in contract.TOKENIZER_FILES:
        target=inputs/'tokenizer'/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(os.environ['QWEN3_8B_TOKENIZER'])/name,target)
    shutil.copyfile(Path(os.environ['SDPO_DIR'])/'verl/utils/reward_score/feedback/mcq.py',inputs/'mcq.py')


@contextmanager
def data_context(archive,tokenizer,mcq=None):
    old=contract.pinned_tokenizer;authors=t.authors_rows
    contract.pinned_tokenizer=lambda:tokenizer;contract._cached_prompt_length.cache_clear()
    try:
        with tempfile.TemporaryDirectory(prefix='finqa-source-') as tmp:
            root=Path(tmp)
            for split in ('train','test'):
                raw=archive.files['v4/report-inputs/sources/FinQA/'+split+'.json']
                require(t.sha(raw)==pool.FINQA_HASHES[split],'FinQA pinned source hash changed')
                (root/(split+'.json')).write_bytes(raw)
            freeze=pool.freeze_finqa(root)
            items={s:{i['id']:i for i in t.finqa.load(root,s)} for s in ('train','test')}
            native={s:t.finqa.rows_for_trainer([items[s][i['id']] for i in freeze[s]],s) for s in items}
            pools={'finqa':pool_items(native['train'],'finqa')};panels={'finqa':native['test']}
            require(archive.jsonl('v4/report-inputs/finqa-pool.jsonl')==pools['finqa'],'FinQA pool differs')
            require(archive.jsonl('v4/report-inputs/finqa-panel.jsonl')==panels['finqa'],'FinQA panel differs')
            raw=archive.files['v4/report-inputs/general-v1-panel.jsonl']
            require(t.sha(raw)=='ce5743ef03f59f57b18231ccd88aee66f0350c8fe3fce8a4abaeb816dd00a1e6','general-v1 panel hash differs')
            archive.expected_pools=pools;archive.expected_panels=panels;archive.expected_training_rows={'finqa':native['train']}
            t.authors_rows=lambda task,*args,**kwargs:copy.deepcopy(native['train']) if task=='finqa' else authors(task,*args,**kwargs)
            yield {'pools':pools,'panels':panels,'freeze':freeze}
    finally:
        contract.pinned_tokenizer=old;t.authors_rows=authors;contract._cached_prompt_length.cache_clear()
