#!/usr/bin/env python3
"""One SEC checkpoint's complete July–September memory loop, with its own collector."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.next_common import *
from kit import next_sec_core as sec
from kit.p3b_score import write_csv
import shutil
import time


def job(a):
    frozen_next(); gpu_identity=machine(a.out, a.gpu, a.standin)
    inputs=read_inputs(a.out,a.standin)
    expected=inputs['base'] if a.seed==0 else inputs['sec'][SEEDS.index(a.seed)]
    require(Path(a.checkpoint_path).resolve()==Path(expected['path']), 'wrong replay sibling checkpoint for W')
    evaluator=checkpoint(a.checkpoint_path,a.checkpoint_receipt,a.standin,base=a.seed==0)
    require(evaluator['model_file_hashes']==expected['identity']['model_file_hashes'], 'bound SEC checkpoint changed')
    collector=checkpoint(a.collector_checkpoint_path,a.collector_checkpoint_receipt,a.standin,base=a.seed==0)
    require(evaluator['model_file_hashes']==collector['model_file_hashes'], 'collector must equal evaluator')
    root=a.out/'sec'/('base' if a.seed==0 else f's{a.seed}')
    bind(root/'identity.json',dict(checkpoint=evaluator,collector=collector,gpu_identity=gpu_identity,technical_synthetic=a.standin))
    if (root/'manifest.json').exists(): frozen_files(root); print('SKIP SEC',a.seed); return
    started=time.monotonic()
    slice_dir=N/'sec'; runs=root/'runs'; rows=[]; months=('2026-07','2026-08','2026-09')
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, KIT_DETERMINISTIC='1')
    for month in months:
        for arm in ('frozen','memory'):
            if arm=='memory' and month==months[0]: continue
            current=slice_dir
            if arm=='memory':
                current=root/'memory'
                sec.build_memory_batch(slice_dir,runs,month,current)
                earlier={m:dict(checkpoint=evaluator,batch=s.file_hash(slice_dir/f'batch-{m}.json'),
                    responses=s.file_hash(runs/f'frozen-{m}/responses.jsonl')) for m in months if m<month}
                bind(current/f'collector-{month}.json', earlier)
            folder=runs/f'{arm}-{month}'; batch=current/f'batch-{month}.json'
            items=json.loads(batch.read_text())
            if not (folder/'bed-score.json').exists():
                if folder.exists(): retain_partial(folder)
                if a.standin:
                    finqa=sec.load_kit('beds/finqa'); raw=[]
                    for i,item in enumerate(items):
                        response='Answer: '+str(item['qa']['exe_ans']) if i%3 else 'Answer: -999999'
                        raw.append(dict(id=item['id'],response=response,output_tokens=10,finish_reason='stop'))
                    write_rows(folder/'responses.jsonl',raw)
                    write_rows(folder/'tokens.jsonl',[dict(id=i['id'],prompt_tokens=20) for i in items])
                    atomic(folder/'bed-score.json',dict(items_sha256=sec.load_kit('eval_bed').items_digest(items),
                        responses_sha256=s.file_hash(folder/'responses.jsonl'),sec_v4_input_batch_sha256=s.file_hash(batch),
                        per_item={r['id']:int(finqa.compute_score(finqa.DATA_SOURCE,r['response'],finqa.gold_of(i))['acc']) for i,r in zip(items,raw)}))
                else:
                    code=sec.run_eval_bed(str(current),month,'generate',['--model',a.checkpoint_path,'--out',str(folder),
                        '--max-new-tokens','2048','--max-model-len','12288'])
                    require(code==0,'SEC generation failed')
            score=json.loads((folder/'bed-score.json').read_text())
            check(batch,score['sec_v4_input_batch_sha256']); check(folder/'responses.jsonl',score['responses_sha256'])
            require(score['items_sha256']==sec.load_kit('eval_bed').items_digest(items),'SEC question mismatch')
            raw=t.read_jsonl(folder/'responses.jsonl'); token_rows=t.read_jsonl(folder/'tokens.jsonl')
            metrics=sec.v4_metrics(items,raw,token_rows)
            atomic(folder/'explicit-metrics.json',metrics)
            byid={r['id']:r for r in raw};prompts={r['id']:r.get('prompt_tokens') for r in token_rows}
            for item in items:
                if not item['sec']['issuer_holdout']: continue
                r=byid[item['id']]; pred=metrics['predictions'][item['id']]
                rows.append(dict(arm='v0-'+('base' if a.seed==0 else 'trained')+'-'+('off' if arm=='frozen' else 'memory'),
                    seed=a.seed,step=0,task='sec',panel='sec-'+month,cap=2048,question_id=item['id'],correct=pred['explicit_unit'],
                    output_tokens=r['output_tokens'],finished=int(r['finish_reason'] in ('stop','eos')),
                    historical=pred['historical'],prompt_tokens=prompts[item['id']],issuer='|'.join(sorted(item['sec']['issuer_ciks']))))
    write_csv(root/'scores.csv',rows, [*__import__('kit.p3b_score',fromlist=['COLUMNS']).COLUMNS,'historical','prompt_tokens','issuer'])
    atomic(root/'cost.json',dict(kind='SEC_scoring',gpu_hours=0 if a.standin else (time.monotonic()-started)/3600,technical_synthetic=a.standin))
    freeze(root,['scores.csv','identity.json','cost.json']+[str(p.relative_to(root)) for p in sorted(runs.rglob('*')) if p.is_file()]+[str(p.relative_to(root)) for p in sorted((root/'memory').glob('*')) if p.is_file()],a.standin)
    print('SEC',a.seed,':',len(rows),'unseen-issuer score rows')


def configure(p):
    device(p);p.add_argument('--seed',type=int,choices=(0,*SEEDS),required=True)
    for name in ('checkpoint-path','checkpoint-receipt','collector-checkpoint-path','collector-checkpoint-receipt'):
        p.add_argument('--'+name,required=True)
if __name__ == '__main__': raise SystemExit(entry(job, configure, 'next-sec-{seed}'))
