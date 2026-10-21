#!/usr/bin/env python3
"""One registered full-weight 40-update stage; fresh W optimizer and incoming EMA."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
from kit import c_train
from kit.p3b_train import completed_training
from types import SimpleNamespace as NS
from collections import Counter
import subprocess


def training_rows(rows,tok,recipe):
    converted,counts=c_train.training_rows([dict(r,gold=r.get('gold','')) for r in rows],tok,recipe)
    for r in converted:
        if r['data_source']!='p3b_replay':r['data_source']='a4_sec'
        r['extra_info']['p3b_task_source']='a4_sec'
    return converted,counts


def command(a,recipe,model,data,run):
    cmd,resolve,env=c_train.command(a,recipe,model,data,run)
    def change(args):
        result=[]
        for x in args:
            if x==str(KIT/'c_sdft_entry.py'):x=str(KIT/'a4_sdft_entry.py')
            if x.startswith('custom_reward_function.path='):x='custom_reward_function.path='+str(KIT/'a4_reward.py')
            # Nine registered full-weight finals per seed; no intermediate selection.
            if x.startswith('trainer.save_freq='):x='trainer.save_freq=40'
            result.append(x)
        return result
    env['TASK']='a4_sec'
    env['A4_TOKEN_LOG']=str(run/'token-journals')
    cmd,resolve=change(cmd),change(resolve)
    if recipe.startswith('D'):
        extra='+ray_kwargs.ray_init.runtime_env.env_vars.A4_TOKEN_LOG='+json.dumps(env['A4_TOKEN_LOG'])
        cmd+=[extra];resolve.insert(len(resolve)-3,extra)
    return cmd,resolve,env


def schedule(out,doc,arm,seed,n,tok):
    pool=ledger(out,n)
    if arm=='direct':pool=[r for r in pool if r['month']==MONTHS[n-1]]
    rows=[dict(id=r['id'],prompt=[dict(role='user',content=finqa.render_prompt(r['item']))],response=r['response'],
               gold=finqa.gold_of(r['item'])) for r in pool]
    if not rows:return [],dict(empty=True)
    task=cycle(rows,seed,1280);replay_audit=None
    if '+replay' in doc['W']:
        source=Path(doc['replay']['path']);check(source/'manifest.json',doc['replay']['manifest']);frozen_files(source)
        generic=[dict(r,p3b_replay=True) for r in t.read_jsonl(source/'replay.jsonl')]
        skipped=[];draws=[]
        if doc['W']=='D+replay':
            for r in generic:
                prompt=r['prompt'] if isinstance(r['prompt'],list) else [dict(role='user',content=r['prompt'])]
                if len(t.tokenize_student(tok,prompt,r['response'])[0])>2048:skipped.append(str(r['id']))
        task=mixed(task,generic,seed,skip_ids=skipped,skipped_draws=draws)
        replay_audit=dict(skipped_ids=skipped,skipped_draws=draws)
    counts=Counter(r['id'] for r in task if not r.get('p3b_replay'))
    return task,dict(empty=False,unique_traces=len(counts),repetitions=dict(counts),task_exposures=sum(counts.values()),
        replay_exposures=1280-sum(counts.values()),replay=replay_audit)


def _job(a):
    doc=binding(a.out,a.standin);n=a.month
    require(t.runtime_versions()==doc['software']['trainer'],'bound trainer environment changed')
    require(a.arm in ARMS and a.seed in SEEDS and 1<=n<=6,'unregistered training row')
    require(a.arm!='consolidation' or n in BOUNDARIES,'unregistered consolidation boundary')
    incoming=state(a.out,doc,a.arm,a.seed,n-1)
    # The registered general retention panel must be sealed for this exact incoming checkpoint.
    for panel in ('general-v1',):
        frozen_files(score_root(a.out,'incoming',a.arm,a.seed,n,panel,'off'))
    tok=tokenizer(doc,a.standin);rows,audit=schedule(a.out,doc,a.arm,a.seed,n,tok)
    name=stage_name(a.arm,a.seed,n);run=a.out/'runs'/name;data=a.out/'data'/name
    observed=dict(incoming=model_identity(incoming,a.standin),rows=s.digest(rows),seed=a.seed,recipe=doc['W'],
        binding=s.file_hash(a.out/'binding.json'),ledger={m:s.file_hash(a.out/'ledger'/m/'manifest.json') for m in MONTHS[:n]},audit=audit)
    with lock(run):
        bind(a.out/'stage-identities'/(name+'.json'),observed)
        if (run/'manifest.json').exists():frozen_files(run);print('SKIP:',name);return
        if not rows:
            atomic(run/'summary.json',dict(skipped=True,reason='empty_monthly_ledger' if a.arm=='direct' else 'empty_cumulative_ledger',**observed))
            freeze(run,['summary.json']);return
        budget(a.out,doc)
        converted,token_counts=training_rows(rows,tok,doc['W'])
        data.mkdir(parents=True,exist_ok=True)
        if not (data/'schedule.jsonl').exists():t.write_jsonl(data/'schedule.jsonl',rows)
        require(t.read_jsonl(data/'schedule.jsonl')==rows,'schedule changed')
        if not a.standin:
            require(a.sdpo is not None and a.tokenizer,'pinned SDPO/tokenizer required')
            require(subprocess.check_output(['git','-C',str(a.sdpo),'rev-parse','HEAD'],text=True).strip()=='7c457fc1b1f636ae794eb0362ba37d4743b06fbc','wrong SDPO revision')
            identity(a.tokenizer,'student')
            require(len(set(a.gpu.split(',')))==8,'eight distinct training GPUs required')
            import pyarrow as pa
            import pyarrow.parquet as pq
            if not (data/'manifest.json').exists():
                for part,rr in [('train',converted),('val',converted[:32]),('test',[dict(converted[0],extra_info=dict(converted[0]['extra_info'],demonstration=''))])]:
                    pq.write_table(pa.Table.from_pylist(rr),data/(part+'.parquet'))
                freeze(data,['schedule.jsonl','train.parquet','val.parquet','test.parquet'],rows=1280)
            frozen_files(data)
        else:freeze(data,['schedule.jsonl'],rows=1280)
        def nll(which,record):
            cmd=[a.inference_python,str(KIT/'a4_nll.py'),'--out',str(a.out),'--model',record['model'],
                '--schedule',str(data/'schedule.jsonl'),'--output',str(run/f'nll-{which}.json'),'--gpu',a.score_gpu]
            if record.get('receipt'):cmd+=['--receipt',record['receipt']]
            if a.standin:cmd+=['--standin']
            attempt=start_attempt(a.out,doc,'nll',a.standin,arm=a.arm,seed=a.seed,month=n,which=which)
            success=False
            try:
                with job_deadline(doc['job_limits']['nll']):gpu_run(cmd,check=True)
                success=True
            finally:finish_attempt(a.out,attempt,completed=success)
        nll('before',incoming)
        attempt=start_attempt(a.out,doc,'training',a.standin,arm=a.arm,seed=a.seed,month=n)
        success=False
        try:
            with job_deadline(doc['job_limits']['training']):
                if a.standin:
                    if a.fail_once and not (run/'failure-injected.json').exists():
                        atomic(run/'failure-injected.json',dict(technical_synthetic=True));raise RuntimeError('stand-in interrupted stage')
                    atomic(run/'export-40.json',dict(technical_synthetic=True,step=40))
                else:
                    cmd,resolve,env=command(a,doc['W'],incoming['model'],data,run)
                    bind(run/'command.json',cmd)
                    if not completed_training(run,doc['W'][0],40):
                        if (run/'started.json').exists():
                            partial=run/'partials'/str(time.time_ns());partial.mkdir(parents=True)
                            for name in ('sdft','train','metrics.jsonl','started.json','trainer-exit.json','resolved-config.yaml','resolved-config.err'):
                                if (run/name).exists():(run/name).rename(partial/name)
                        with (run/'resolved-config.yaml').open('w') as out,(run/'resolved-config.err').open('w') as err:
                            gpu_run(resolve,cwd=a.sdpo,env=env,stdout=out,stderr=err,check=True)
                        atomic(run/'started.json',observed);clock=time.monotonic()
                        done=gpu_run(cmd,cwd=a.sdpo,env=env)
                        atomic(run/'trainer-exit.json',dict(returncode=done.returncode,seconds=time.monotonic()-clock))
                        require(done.returncode==0,'trainer failed; retain partials and restart same seed')
                        if not completed_training(run,doc['W'][0],40):raise NotReady('40-update checkpoint missing')
                    from kit.v4_sft import complete_export
                    export=run/'hf-step40'
                    if not complete_export(export):
                        ckpt=run/('sdft' if doc['W'][0]=='D' else 'train')/'global_step_40'
                        if doc['W'][0]=='D':ckpt/='actor'
                        gpu_run([a.trainer_python,'-m','verl.model_merger','merge','--backend','fsdp','--local_dir',str(ckpt),'--target_dir',str(export)],cwd=a.sdpo,env=env,check=True)
                    require(complete_export(export),'incomplete merged checkpoint')
                    atomic(run/'export-40.json',identity(export))
                success=True
        finally:finish_attempt(a.out,attempt,completed=success)
        nll('after',dict(model=str(run/'hf-step40'),receipt=str(run/'export-40.json')))
        runtime_tokens=None
        if doc['W'].startswith('D') and not a.standin:
            journals=[json.loads(p.read_text()) for p in (run/'token-journals').glob('*.json')]
            require(bool(journals),'D token accounting missing')
            runtime_tokens=sum(r['student_tokens']+r['teacher_tokens']+r['replay_target_tokens'] for r in journals)
        elif not a.standin:
            metrics=[r.get('data',r) for p in run.rglob('metrics.jsonl') for r in t.read_jsonl(p)]
            require(any('p3b/total_tokens' in r for r in metrics),'F token accounting missing')
            runtime_tokens=sum(r.get('p3b/total_tokens',0) for r in metrics)
        atomic(run/'summary.json',dict(skipped=False,**observed,scheduled_tokens=token_counts,runtime_training_tokens=runtime_tokens))
        freeze(run,['summary.json','export-40.json','nll-before.json','nll-after.json'])


def job(a):
    require(not a.fail_once or a.standin,'failure injection is stand-in only')
    with lock(a.out/'node-writer'):return _job(a)


def configure(p):
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--month',type=int,required=True);p.add_argument('--gpu',default='0,1,2,3,4,5,6,7')
    p.add_argument('--score-gpu',default='0');p.add_argument('--tokenizer');p.add_argument('--sdpo',type=Path)
    p.add_argument('--trainer-python',default=sys.executable);p.add_argument('--inference-python',default=sys.executable)
    p.add_argument('--fail-once',action='store_true',help='stand-in failure rehearsal only')
if __name__=='__main__':raise SystemExit(entry(job,configure,'a4-train-{arm}-{seed}-{month}'))
