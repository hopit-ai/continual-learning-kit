#!/usr/bin/env python3
"""Offline execution of v0+/v3 and pool rows of README-next through real --standin jobs; no scheduler."""
from pathlib import Path
import sys
if __package__ in (None, ''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.next_common import *
import argparse
from collections import Counter
from contextlib import redirect_stdout
import importlib
import io


def model(root,label):
    root.mkdir(parents=True,exist_ok=True)
    atomic(root/'config.json',dict(fixture=label))
    receipt=root.parent/(root.name+'-receipt.json')
    atomic(receipt,dict(model_dir=str(root),technical_synthetic=True,model_file_hashes={'config.json':s.file_hash(root/'config.json')}))
    return receipt


def fixture(root):
    from kit.p3a_common import INPUTS
    import pyarrow as pa
    import pyarrow.parquet as pq
    b=root/'B';v4=root/'v4';a=root/'A';exports=root/'exports';base=root/'base'
    base_receipt=model(base,'base')
    for recipe in RECIPES:
        for seed in SEEDS:
            if recipe in ('D','F'):
                path=exports/f'{recipe}-finqa-alone-s{seed}/hf-step40';receipt=model(path,f'{recipe}-{seed}')
                atomic(v4/f'runs/{recipe}-finqa-alone-s{seed}/env/export-identity-step40.json',json.loads(receipt.read_text()))
            else:
                path=b/f'runs/finqa-{recipe}-s{seed}/hf-step40';receipt=model(path,f'{recipe}-{seed}')
                atomic(path.parent/'export-40.json',json.loads(receipt.read_text()))
            for route in ('standalone','sequence'):
                path=b/f'runs/{route}-{recipe}-s{seed}/hf-step40';receipt=model(path,f'{route}-{recipe}-{seed}')
                atomic(path.parent/'export-40.json',json.loads(receipt.read_text()))
    generic=[dict(id='generic-'+str(i),prompt=q,response='A') for i,q in enumerate([
        'Describe a red velvet cake.', 'List five planets beyond Earth.', 'Translate good morning into French.',
        'Explain photosynthesis for a child.', 'Give directions to the nearest library.', 'Name the inventor of the telephone.'])]
    write_rows(b/'replay/replay.jsonl',generic)
    freeze(b/'replay',['replay.jsonl'],True)
    names=[]
    for seed in SEEDS:
        rows=[dict(id=f'math-train-{i%20}',prompt='Distinct MATH training computation '+str(i%20),response='Worked target. \\boxed{2}',task='math_screen',gold='2') for i in range(1280)]
        name=f'schedule-{seed}.jsonl';write_rows(b/'targets'/name,rows);names.append(name)
    atomic(b/'targets/inputs.json',dict(task='math_screen'))
    freeze(b/'targets',names+['inputs.json'],True)
    prepared=a/INPUTS/'prepared';task=prepared/'math_screen'
    write_rows(task/'heldout.jsonl',[dict(id=f'math-eval-{i}',prompt=f'Find unique synthetic MATH result {i}',gold='2') for i in range(10)])
    (a/INPUTS/'preregistration.md').write_text('Synthetic registration only\n')
    manifest=dict(technical_synthetic=True,preregistration_sha256=s.file_hash(a/INPUTS/'preregistration.md'),implementation={},
        candidates=dict(math_screen=dict(files={'heldout.jsonl':s.file_hash(task/'heldout.jsonl')})))
    atomic(prepared/'frozen.json',manifest);atomic(a/INPUTS/'frozen-annex.json',dict(manifest=manifest,manifest_sha256=s.digest(manifest)))
    earlier=root/'finqa-targets';earlier.mkdir()
    pq.write_table(pa.Table.from_pylist([dict(prompt=[dict(role='user',content='Calculate the synthetic corporate revenue ratio from unique filing '+str(i))],
        data_source='finqa',reward_model=dict(ground_truth='2'),extra_info=dict(split='train',index=f'finqa-train-{i}',demonstration='The verified teacher explains each calculation in enough words. Answer: 2')) for i in range(6)]),earlier/'train.parquet')
    atomic(earlier/'manifest.json',dict(train_sha256=s.file_hash(earlier/'train.parquet'),technical_synthetic=True))
    write_rows(root/'finqa-panel.jsonl',[dict(data_source='finqa',extra_info=dict(index='finqa-eval',split='test'),prompt=[dict(role='user',content='A different heldout FinQA financial table')])])
    write_rows(root/'gsm.jsonl',[dict(question=f'Zebra {i} has {i+1} marbles and buys {i+3} marbles. How many are there?',answer='solution #### '+str(2*i+4),fake_response='Answer: '+str(2*i+4)) for i in range(3)])
    write_rows(root/'arc.jsonl',[dict(id=f'ARC-{i}',question=f'Which specimen in experiment {i} is a conductor?',choices=dict(label=['A','B','C','D'],text=['copper','paper','wood','air']),answerKey='A',fake_response='Answer: A') for i in range(3)])
    write_rows(root/'hellaswag.jsonl',[dict(ind=i,ctx=f'Activity video {i}. Someone carefully starts a unique action sequence {i}.',endings=['They finish.','They vanish.','A fish speaks.','It snows.'],label=i%4) for i in range(120)])
    # Complete B evidence including stage-1s and all unreset curves; no recovered/reset rows.
    from kit.p3b_score import write_csv
    score_rows=[]
    def add(arm,seed,step,panel):
        for i in range(10):
            score_rows.append(dict(arm=arm,seed=seed,step=step,task=panel,panel=panel,cap=2048,question_id=f'{panel}-{i}',correct=int(i%3!=0),output_tokens=10,finished=1))
    for panel in ('finqa','ifeval','knowledge','math','math_screen'):add('base',0,0,panel)
    for r in RECIPES:
        for seed in SEEDS:
            for panel in ('finqa','ifeval','knowledge','math'):add(r if r in ('D','F') else 'finqa-'+r,seed,40,panel)
            for route in ('standalone','sequence'):
                for panel in ('finqa','ifeval','knowledge','math','math_screen'):
                    for step in ((0,10,20,30,40) if panel=='math_screen' else (0,20,40)):add(route+'-'+r,seed,step,panel)
    write_csv(root/'B-reading.csv',score_rows)
    return dict(b=b,v4=v4,a=a,exports=exports,base=base,base_receipt=base_receipt,earlier=earlier)


def invoke(module,args,counts):
    module=importlib.import_module('kit.'+module)
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--standin',action='store_true')
    module.configure(p);a=p.parse_args([str(v) for v in args]);a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    require(a.standin,'dry-run must be stand-in')
    counts[module.__name__.split('.')[-1]]+=1
    return (module.replay_job if hasattr(module,'replay_job') else module.job)(a)


def job(a):
    require(a.standin,'this rehearsal requires --standin')
    require(not (a.out/'dryrun-summary.json').exists(),'choose a fresh dry-run OUT')
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
    f=fixture(a.out/'fixtures');out=a.out/'NEXT';counts=Counter();failures=[]
    def call(name,*args):return invoke(name,['--out',out,'--standin',*args],counts)
    # Nested score calls run their real parsers and jobs inline, avoiding thousands of Python startups.
    from kit import next_blend,next_score
    def nested(command):
        name=Path(command[1]).stem
        return invoke(name,command[2:],counts)
    old_blend,old_score=next_blend.run,next_score.run
    next_blend.run=next_score.run=nested
    try:
        with (a.out/'dryrun.log').open('w') as log, redirect_stdout(log):
            call('next_w','--source',a.out/'fixtures/B-reading.csv','--bootstrap',20)
            call('next_bind','--w-identity',out/'W.json','--base',f['base'],'--base-receipt',f['base_receipt'],
                '--package-b',f['b'],'--v4-work',f['v4'],'--v4-exports',f['exports'])
            inputs=read_inputs(out,True)
            call('next_hellaswag','--action','prepare','--rows',a.out/'fixtures/hellaswag.jsonl')
            for seed,parent in [(0,inputs['base'])]+list(zip(SEEDS,inputs['sec'])):
                call('next_sec','--seed',seed,'--checkpoint-path',parent['path'],'--checkpoint-receipt',parent['receipt'],
                    '--collector-checkpoint-path',parent['path'],'--collector-checkpoint-receipt',parent['receipt'])
            for slot in inputs['finals']:
                for lam in (0,.25,.5,.75,1):
                    call('next_blend','--slot',slot['name'],'--lam',lam,'--nvme-root',a.out,'--package-a',f['a'])
            if not getattr(a,'defer_pools',False):
                call('next_pools','--package-a',f['a'],'--finqa-panel',a.out/'fixtures/finqa-panel.jsonl','--finqa-targets',f['earlier'],
                    '--gsm-rows',a.out/'fixtures/gsm.jsonl','--arc-rows',a.out/'fixtures/arc.jsonl')
            # Repeat identity-bound jobs and exercise correctness/not-ready paths without modifying freezes.
            def failure(label,expected,fn):
                try:fn()
                except expected:failures.append(label)
                else:raise AssertionError('failure path did not stop: '+label)
            failure('collector mismatch',CorrectnessStop,lambda:call('next_sec','--seed',101,'--checkpoint-path',inputs['sec'][0]['path'],
                '--checkpoint-receipt',inputs['sec'][0]['receipt'],'--collector-checkpoint-path',inputs['base']['path'],'--collector-checkpoint-receipt',inputs['base']['receipt']))
            failure('missing blend slot',CorrectnessStop,lambda:call('next_blend','--slot','missing','--lam',.5,'--nvme-root',a.out))
            failure('scoring GPU mismatch',CorrectnessStop,lambda:next_score.machine(out,'0,1',True))
            failure('changed checkpoint',CorrectnessStop,lambda:checkpoint(a.out/'missing',a.out/'missing-receipt',True))
            failure('pending W',NotReady,lambda:call('next_w','--source',out,'--bootstrap',20))
            failure('missing freeze',NotReady,lambda:frozen_files(a.out/'absent'))
            failure('empty replay pool',CorrectnessStop,lambda:cycle([],101,160))
            # Complete idempotent replay: the bound inputs and output hashes must remain unchanged.
            slot=inputs['finals'][0]
            call('next_blend','--slot',slot['name'],'--lam',.5,'--nvme-root',a.out,'--package-a',f['a'])
            call('next_read','--source',out,'--bootstrap',20)
            call('next_collect','--archive',a.out/'next-return.tar.gz')
            invoke('next_read',['--out',a.out/'returned','--standin','--source',a.out/'next-return.tar.gz','--bootstrap',20],counts)
            require(json.loads((out/'reading-next.json').read_text())==json.loads((a.out/'returned/reading-next.json').read_text()),'return reader differs')
    finally:next_blend.run,next_score.run=old_blend,old_score
    rows=sum(1 for p in list((out/'scores').glob('*/scores.csv'))+list((out/'sec').glob('*/scores.csv')) for _ in p.read_text().splitlines()[1:])
    result=dict(command_counts=dict(counts),failure_paths=failures,question_rows=rows,screen_finals=60,blend_points=300,
        probe_stages=0,probe_score_jobs=0,pools_deferred=getattr(a,'defer_pools',False),unique_score_jobs=len(list((out/'scores').glob('*/scores.csv'))),
        unique_shared_endpoints=len(list((out/'scores').glob('next-shared-*/scores.csv'))),sec_checkpoints=6,standin_only=True,all_runbook_rows=True)
    atomic(a.out/'dryrun-summary.json',result);print(json.dumps(result,sort_keys=True))


def configure(p):p.add_argument('--defer-pools',action='store_true',help='combined rehearsal runs N6 after C targets')
if __name__ == '__main__':raise SystemExit(entry(job,configure,'next-dryrun'))
