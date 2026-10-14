#!/usr/bin/env python3
"""One checkpoint, one frozen panel set, one physical scoring GPU; v4 scorers unchanged."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_common import *
import csv
from types import SimpleNamespace
COLUMNS = 'arm seed step task panel cap question_id correct output_tokens finished'.split()


def write_csv(path, rows, columns=COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        writer=csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def job(a):
    from kit import eval_bed as bed, score_forgetting as general
    frozen_hashes(); require(',' not in a.gpu, 'use one scoring GPU')
    root = a.out / 'scores' / f'{a.arm}-s{a.seed}-k{a.step}-{a.panel_set}'
    args = SimpleNamespace(bed=a.panel_set, root=str(a.panel_root), split='test', model=a.model,
        max_new_tokens=2048, max_model_len=12288, out=str(root / 'raw'), limit=None, heldout_n=None, allow_subset=False)
    if a.panel_set == 'general-v1': items, _ = general.load_panel()
    elif a.panel_set in s.ORDER:
        from kit.p3a_common import frozen, INPUTS
        require(a.package_a is not None, 'task scoring requires frozen package A')
        frozen(a.package_a, a.standin); folder=a.package_a / INPUTS / 'prepared' / a.panel_set
        items=[dict(id=r['id'],prompt=r['prompt'],ground_truth=r['gold']) for r in t.read_jsonl(folder / 'heldout.jsonl')]
        args.split='heldout'; args.root=str(folder); os.environ['CODE_TESTS']=str(folder / 'tests.jsonl')
    elif a.standin: items=[dict(id=str(i),prompt='standin',ground_truth='0') for i in range(10)]
    else:
        require(a.panel_root is not None and a.panel_root.is_dir(),'missing FinQA panel root')
        items=bed.items_of(bed.load_bed(a.panel_set),args)
        from kit.v4_contract import check_eval_items
        check_eval_items(items,'finqa','test')
        require(len(items)==1147,'FinQA scoring requires all 1147 questions')
    receipt=a.checkpoint_receipt
    base_start=a.arm=='base' or (a.step==0 and a.arm.startswith('standalone-'))
    if not a.standin and not base_start:
        if receipt is None and a.model:
            parent=Path(a.model).parent
            guessed=parent / (('export-'+str(a.step)+'.json') if a.step>0 else 'export.json')
            if guessed.exists(): receipt=guessed
        require(receipt is not None,'checkpoint scoring requires its original export receipt')
    observed=dict(technical_synthetic=True) if a.standin else identity(a.model,'student' if base_start else None,receipt)
    bind(a.out / 'score-identities' / (root.name+'.json'), dict(model=observed,panel=s.digest(items),gpu=a.gpu,arm=a.arm,seed=a.seed,step=a.step))
    if (root / 'manifest.json').exists(): frozen_files(root); print('SKIP:', root.name); return
    retain_partial(root)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, KIT_DETERMINISTIC='1', V4_TELEMETRY='1', V4_PHASE0_INFERENCE='0')
    import time
    clock=time.monotonic()
    if a.standin:
        raw=[dict(id=r['id'],panel=r.get('panel',a.panel_set),output_tokens=10,finish_reason='stop',response='standin') for r in items]
        correct={(r.get('panel',a.panel_set),r['id']):int(i%3!=0) for i,r in enumerate(items)}
        root.mkdir(parents=True); machine='CPU-STANDIN'
    else:
        try:
            if a.panel_set == 'general-v1': general.cmd_generate(args)
            else: bed.cmd_generate(args, frozen_items=items)
        except bed.EvalBedError as exc: raise CorrectnessStop(str(exc)) from exc
        raw=t.read_jsonl(root / 'raw/responses.jsonl')
        doc=json.loads((root / 'raw' / ('forgetting.json' if a.panel_set=='general-v1' else 'bed-score.json')).read_text())
        machine=doc['machine']['id']
        correct={(p,q):v for p,x in doc['panels'].items() for q,v in x['per_member'].items()} if a.panel_set=='general-v1' else {(a.panel_set,q):v for q,v in doc['per_item'].items()}
    require(len(raw)==len(items) and {(r.get('panel',a.panel_set),r['id']) for r in raw}==set(correct)=={(r.get('panel',a.panel_set),r['id']) for r in items},'broken scoring coverage')
    bind(a.out / 'scoring-machine.json',dict(id=machine,gpu=a.gpu,technical_synthetic=a.standin))
    rows=[dict(arm=a.arm,seed=a.seed,step=a.step,task=a.panel_set,panel=r.get('panel',a.panel_set),cap=2048,question_id=r['id'],
        correct=int(correct[(r.get('panel',a.panel_set),r['id'])]),output_tokens=r['output_tokens'],finished=int(r['finish_reason'] in ('stop','eos'))) for r in raw]
    write_csv(root / 'scores.csv',rows)
    atomic(root / 'identity.json',dict(model=observed,machine=machine,panel_sha256=s.digest(items),technical_synthetic=a.standin))
    atomic(root / 'cost.json',dict(kind='scoring',gpu_hours=0 if a.standin else (time.monotonic()-clock)/3600,technical_synthetic=a.standin))
    atomic(root / 'manifest.json',files_manifest(root,['scores.csv','identity.json','cost.json']))


def configure(p):
    device(p); p.add_argument('--arm',required=True); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--step',type=int,required=True); p.add_argument('--panel-set',choices=('finqa','general-v1','code','math_screen'),required=True)
    p.add_argument('--checkpoint-receipt',type=Path); p.add_argument('--panel-root',type=Path); p.add_argument('--package-a',type=Path)
if __name__ == '__main__': raise SystemExit(entry(job,configure,'p3b-score-{arm}-{seed}-{step}-{panel_set}'))
