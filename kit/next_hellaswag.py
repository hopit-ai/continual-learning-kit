#!/usr/bin/env python3
"""Freeze HellaSwag-100 before output, or score one checkpoint with its frozen strict parser."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.next_common import *
from kit.p3b_score import write_csv
import re
import time

REVISION='218ec52e09a7e7462a5400043bb9a69a41d06b76'
VALIDATION_FILES={'data/validation-00000-of-00001.parquet':'899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd'}
NOTICE='HellaSwag (Zellers et al., 2019), Rowan/hellaswag; MIT License; ActivityNet-derived items only; https://huggingface.co/datasets/Rowan/hellaswag'


def strict_label(text, labels='ABCD'):
    # The final line, including case and punctuation, must be exactly the requested form.
    last=text.strip().splitlines()[-1] if text.strip() else ''
    match=re.fullmatch(r'Answer: (['+re.escape(labels)+'])',last)
    return match[1] if match else None


def download(repo, revision, fragment, expected_files=None, *, root=None):
    import pyarrow.parquet as pq
    if root is None:
        from huggingface_hub import snapshot_download
        root = snapshot_download(repo, repo_type='dataset', revision=revision, local_files_only=True)
    root = Path(root)
    paths = sorted(p for p in root.rglob('*.parquet') if fragment in str(p.relative_to(root)))
    require(bool(paths), 'pinned parquet split must be staged in the local cache: '+repo+' '+revision)
    hashes={str(path.relative_to(root)):s.file_hash(path) for path in paths}
    if expected_files is not None:
        require(hashes==expected_files,'pinned HellaSwag validation parquet files/SHA256 mismatch')
    rows=[]
    for path in paths:rows+=pq.read_table(path).to_pylist()
    return rows,hashes


def panel_rows(raw):
    raw=[r for r in raw if r['source_id'].startswith('activitynet~')]
    require(len({str(r['ind']) for r in raw})==len(raw),'duplicate HellaSwag source IDs')
    require(len(raw)>=100,'HellaSwag panel requires 100 ActivityNet items')
    chosen=sorted(raw,key=lambda r:(t.sha('next-v4-holdout-v1|'+str(r['ind'])),str(r['ind'])))[:100]
    return [dict(id='hellaswag-val-'+str(r['ind']),prompt=r['ctx']+'\n'+
        '\n'.join(f'{label}. {ending}' for label,ending in zip('ABCD',r['endings']))+
        '\nEnd with Answer: <letter>',gold='ABCD'[int(r['label'])],licence_notice=NOTICE) for r in chosen]


def prepare(a):
    frozen_next(); root=a.out/'hellaswag'
    provenance=dict(revision=REVISION,scorer_sha256=s.file_hash(KIT/'next_hellaswag.py'),technical_synthetic=a.standin)
    if a.standin:
        require(a.rows is not None,'stand-in requires local HellaSwag fixture')
        raw=t.read_jsonl(a.rows);provenance['source_sha256']=s.file_hash(a.rows)
    else:
        require(a.rows is None,'real HellaSwag uses pinned validation source')
        raw,provenance['source_files']=download('Rowan/hellaswag',REVISION,'validation',VALIDATION_FILES)
    rows=panel_rows(raw);require(len(rows)==100,'HellaSwag panel requires 100 items')
    bind(root/'inputs.json',provenance)
    if (root/'manifest.json').exists(): frozen_files(root); print('SKIP HellaSwag freeze');return
    require(not list((a.out/'pools/raw').glob('*.json')) and not list((a.out/'scores').glob('*')) and not list((a.out/'sec').glob('*')),
            'holdout must freeze before any new output')
    write_rows(root/'panel.jsonl',rows)
    freeze(root,['panel.jsonl','inputs.json'],a.standin,notice=NOTICE,panel_sha256=s.file_hash(root/'panel.jsonl'))
    print('Frozen HellaSwag-100:',s.file_hash(root/'panel.jsonl'))


def score(a):
    frozen_next(); gpu_identity=machine(a.out,a.gpu,a.standin)
    manifest=frozen_files(a.out/'hellaswag');require(manifest['technical_synthetic']==a.standin,'panel type mismatch')
    observed=checkpoint(a.model,a.checkpoint_receipt,a.standin,base=a.arm=='base')
    rows=t.read_jsonl(a.out/'hellaswag/panel.jsonl')
    folder=a.out/'scores'/f'{a.arm}-s{a.seed}-k{a.step}-hellaswag'
    bind(a.out/'score-identities'/(folder.name+'.json'),dict(model=observed,panel=manifest['panel_sha256'],technical_synthetic=a.standin))
    if (folder/'manifest.json').exists():frozen_files(folder);print('SKIP',folder.name);return
    if folder.exists():retain_partial(folder)
    started=time.monotonic()
    if a.standin:
        raw=[dict(id=r['id'],response='Answer: '+('A' if i%3==0 else r['gold']),output_tokens=10,finish_reason='stop') for i,r in enumerate(rows)]
    else:
        # Use the same deterministic scorer engine and context as B without changing its parsers.
        from kit import eval_bed as bed
        from types import SimpleNamespace
        module=SimpleNamespace(DATA_SOURCE='hellaswag',
            compute_score=lambda source,text,gold,extra_info=None:dict(acc=int(strict_label(text)==gold),incorrect_format=int(strict_label(text) is None)))
        old=bed.load_bed
        bed.BED_FILES['hellaswag']=KIT/'next_hellaswag.py'
        bed.SPLITS['hellaswag']=('test',)
        bed.load_bed=lambda name:module if name=='hellaswag' else old(name)
        try:
            os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,KIT_DETERMINISTIC='1')
            bed.cmd_generate(SimpleNamespace(bed='hellaswag',root='',split='test',model=a.model,max_new_tokens=2048,
                max_model_len=12288,out=str(folder/'raw'),limit=None,heldout_n=None,allow_subset=False),frozen_items=[dict(id=r['id'],prompt=r['prompt'],ground_truth=r['gold']) for r in rows])
            raw=t.read_jsonl(folder/'raw/responses.jsonl')
        finally:
            bed.load_bed=old
            bed.BED_FILES.pop('hellaswag');bed.SPLITS.pop('hellaswag')
    require(len(raw)==100 and {r['id'] for r in raw}=={r['id'] for r in rows},'HellaSwag response coverage')
    byid={r['id']:r for r in rows}
    write_rows(folder/'responses.jsonl',raw)
    scores=[dict(arm=a.arm,seed=a.seed,step=a.step,task='hellaswag',panel='hellaswag',cap=2048,
        question_id=r['id'],correct=int(strict_label(r['response'])==byid[r['id']]['gold']),
        output_tokens=r['output_tokens'],finished=int(r['finish_reason'] in ('stop','eos'))) for r in raw]
    write_csv(folder/'scores.csv',scores)
    atomic(folder/'identity.json',dict(model=observed,gpu_identity=gpu_identity,panel_sha256=manifest['panel_sha256'],technical_synthetic=a.standin))
    atomic(folder/'cost.json',dict(kind='HellaSwag_scoring',gpu_hours=0 if a.standin else (time.monotonic()-started)/3600,technical_synthetic=a.standin))
    freeze(folder,['scores.csv','identity.json','responses.jsonl','cost.json'],a.standin)


def job(a):
    if a.action=='prepare': prepare(a)
    else: score(a)


def configure(p):
    p.add_argument('--action',choices=('prepare','score'),required=True);p.add_argument('--rows',type=Path)
    device(p);p.add_argument('--checkpoint-receipt',type=Path);p.add_argument('--arm');p.add_argument('--seed',type=int);p.add_argument('--step',type=int)
if __name__ == '__main__':raise SystemExit(entry(job,configure,'next-hellaswag-{action}'))
