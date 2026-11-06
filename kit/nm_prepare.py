"""CPU-only audits, immutable schedules, native commands, export and lane plans."""
from pathlib import Path
import os
import subprocess
import sys
from kit.nm_common import *
from kit import nm_data


def verify_model(record):
    path=Path(record['model']);ready(path.is_dir(),'model not staged')
    files={str(p.relative_to(path)):sha(p) for p in sorted([*path.glob('*.safetensors'),*path.glob('*.safetensors.index.json')])}
    ready(files and (path/'config.json').is_file(),'model export incomplete')
    config=load(path/'config.json')
    need(config.get('model_type')=='qwen3' and config.get('hidden_size')==4096 and config.get('num_hidden_layers')==36,'wrong Qwen3-8B architecture')
    files['config.json']=sha(path/'config.json')
    need(digest(files)==record['sha256'],'model tensor/config identity differs')
    return dict(record,files=files)


def audit_bundle(bundle,binding):
    need(bundle['base']['revision']==BASE,'wrong pinned base')
    verify_model(bundle['base'])
    for parent in binding.get('recipe',{}).get('starts',{}).values():verify_model(parent)
    audit=nm_data.audit_candidates(bundle['candidate_superset'],bundle['excluded_superset'])
    for panel,items in bundle['panels'].items():
        authority=bundle.get('panel_authorities',{}).get(panel)
        ready(authority,'frozen panel authority absent: '+panel)
        need(checked(authority)==items,'panel source population differs: '+panel)
        need(len({r['id'] for r in items})==len(items),'duplicate panel IDs')
        if panel in GENERAL or panel in ('hellaswag','legal'):need(len(items)==100,'frozen panel size differs')
        if panel=='finqa':need(len(items)==1147,'FinQA panel size differs')
        if panel in ('instruction','maths'):
            from kit.score_forgetting import load_panel
            general,_=load_panel()
            expected=[r for r in general if r['panel']==('ifeval' if panel=='instruction' else 'math')]
            need(items==expected,'frozen strict general panel differs')
        if panel=='knowledge':
            from kit.m3_score import knowledge_items
            need(items==knowledge_items(),'frozen reference knowledge panel differs')
    ready(bundle.get('source_revisions') and bundle.get('licenses') and bundle.get('holdout_authority'),'source/licence/exclusion bindings absent')
    need(not bundle.get('reserved_outcomes_opened',False),'reserved stream outcomes opened')
    return dict(exclusions=audit,base_sha256=bundle['base']['sha256'],sources=bundle['source_revisions'])


def render_scoring(items,tokenizer,panel):
    from transformers import AutoTokenizer
    from kit.eval_bed import render,render_messages
    tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True)
    rendered=[render_messages(tok,r['prompt']) if isinstance(r['prompt'],list) else render(tok,r['prompt']) for r in items]
    cap=10240 if panel=='legal' or panel not in PANELS else 4096 if panel=='finqa' else 2048
    need(all(len(tok.encode(v,add_special_tokens=False))<=cap for v in rendered),'decision prompt exceeds frozen cap')
    return rendered


def converted_rows(rows,tok,family,arm,caps,global_rows):
    """No runtime filtering or truncation; reject an entire overlength schedule."""
    from kit.sdft_objective import teacher_messages
    out=[];maxima={};totals=dict(task=0,replay=0)
    for i,r in enumerate(rows):
        prompt=r['prompt']
        if isinstance(prompt,str):prompt=[dict(role='user',content=prompt)]
        replay=bool(r.get('m3_replay',r.get('p3b_replay',False)))
        target=r['response'];teacher_context=r.get('teacher_context') if family=='v2b' and arm=='T' else target
        if teacher_context is None:teacher_context=target
        def encode(messages):return tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        pids=encode(prompt);tids=tok.encode(target+tok.eos_token,add_special_tokens=False)
        teacher=encode(teacher_messages(prompt,teacher_context))
        need(0<len(pids)<=caps['prompt'] and 0<len(tids)<=2048,'admitted row exceeds cap; no filtering')
        need(tids[-1]==tok.eos_token_id and tok.eos_token_id not in tids[:-1],'target EOS convention differs')
        need(len(teacher)<=caps['teacher'],'teacher context over cap; no truncation')
        domain=r.get('domain','finqa' if family=='grpo' else 'sec' if family=='v2b' else 'contract_nli')
        extra=dict(r.get('extra_info',{}),index=str(r['id']),split='train',demonstration=teacher_context,
            m3_replay=replay,m3_domain=domain,m3_target_ids=tids,m3_eos_token_id=tok.eos_token_id,
            nm_update=i//global_rows+1,nm_position=i%global_rows+1)
        source='p3b_replay' if replay else 'finqa' if family=='grpo' else 'a4_sec' if family=='v2b' else 'contract_nli'
        if family=='v2b':extra.update(r.get('reward_extra',{}))
        out.append(dict(id=str(r['id']),prompt=prompt,data_source=source,reward_model=dict(style='rule',ground_truth='' if replay else r['gold']),extra_info=extra))
        totals['replay' if replay else 'task']+=1
        maxima[domain]=max(maxima.get(domain,0),len(pids)+2048,len(teacher)+2048)
    return out,dict(longest_by_domain=maxima,counts=totals,schedule_sha256=digest(rows),rows=len(rows),filtering='none at NM preparation; inherited admitted draws retained')


def training_command(bundle,binding,rows,root,family,arm,seed,*,qualification=False,measurement=False):
    """Compose the frozen D recipe, changing only registered geometry/components."""
    from transformers import AutoTokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rec=binding.get('recipe',{});global_rows=48 if rec.get('name')=='CR' else 32
    caps=rec.get('caps',dict(prompt=2048 if family=='grpo' else 10240,teacher=6144 if family=='grpo' else 14336,response=2048))
    tok=AutoTokenizer.from_pretrained(bundle['tokenizer'],local_files_only=True)
    count=8 if qualification else 40
    if qualification:rows=rows[:8*global_rows]
    need(len(rows)==count*global_rows,'frozen schedule dose differs')
    for start in range(0,len(rows),global_rows):
        batch=rows[start:start+global_rows]
        flags=[bool(r.get('m3_replay',r.get('p3b_replay',False))) for r in batch]
        need(sum(flags)==rec.get('replay_rows',0),'per-update task/replay dose differs')
        if rec.get('name')=='K':need(all(r.get('domain')=='instruction' for r,f in zip(batch,flags) if f),'K generic replay domain differs')
        if rec.get('name')=='CR':
            from collections import Counter
            aliases={'gsm8k':'maths','arc':'knowledge'}
            domains=Counter(aliases.get(r.get('domain'),r.get('domain')) for r,f in zip(batch,flags) if f)
            need(domains==dict(instruction=4,maths=2,knowledge=2,finqa=8),'CR replay domain geometry differs')
    converted,audit=converted_rows(rows,tok,family,arm,caps,global_rows)
    seal(root/'schedule.json',rows)
    # Native n=4 repeats prompts; dataset holds eight unique prompt slots/update.
    dataset=converted[::4] if family=='grpo' else converted
    if family=='grpo':
        need(all(len({str(r['id']) for r in rows[j:j+4]})==1 for j in range(0,len(rows),4)),'GRPO groups not contiguous')
        # The trainer expands each row four times; positions are filled before dispatch.
    data=root/'data';data.mkdir(exist_ok=True)
    for name,values in [('train',dataset),('test',[dict(dataset[0],extra_info=dict(dataset[0]['extra_info'],demonstration=''))])]:
        pq.write_table(pa.Table.from_pylist(values),data/(name+'.parquet'))
    native=root/'native';native.mkdir(exist_ok=True)
    incoming=bundle['base'] if family in ('v2b','grpo') else rec['starts'][str(seed)]
    incoming=verify_model(incoming)
    from kit.nm_model import metadata as model_metadata
    overrides=load(KIT/'resources/p3b/v4-recipes.json')['overrides']['D']
    overrides=[x.format(kit=KIT,sdpo=Path(bundle['sdpo']),model=incoming['model'],data=data,run=root,seed=seed) for x in overrides]
    changes={'data.train_batch_size':8 if family=='grpo' else global_rows,'actor_rollout_ref.actor.ppo_mini_batch_size':global_rows,
        'actor_rollout_ref.rollout.n':4 if family=='grpo' else 1,'actor_rollout_ref.rollout.temperature':.7 if family=='grpo' else 1.,
        'actor_rollout_ref.rollout.top_p':1.,'data.max_prompt_length':caps['prompt'],
        'actor_rollout_ref.actor.self_distillation.max_reprompt_len':caps['teacher'],
        'max_model_len':caps['teacher']+2048,'actor_rollout_ref.rollout.max_model_len':caps['teacher']+2048,
        'actor_rollout_ref.rollout.max_num_batched_tokens':caps['teacher']+2048,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu':caps['teacher']+2048,
        'trainer.balance_batch':rec.get('balance_batch',False),'trainer.default_local_dir':str(native),
        'trainer.save_freq':4 if qualification else 10,'trainer.total_training_steps':count,
        'custom_reward_function.path':str(KIT/'nm_entry.py'),'custom_reward_function.name':'compute_score'}
    overrides=[x for x in overrides if x.split('=',1)[0].lstrip('+') not in changes]
    overrides+=['++'+key+'='+json.dumps(value) for key,value in changes.items()]
    penalty=rec.get('p2_lambda')
    calibration=bundle.get('calibration')
    if family=='fisher':
        ready(calibration or measurement,'Fisher coefficients not sealed')
        penalty=0. if measurement else calibration['lambda_iso' if arm=='C' else 'lambda_fisher']
    n=dict(family=family,arm=arm,seed=seed,updates=count,rows_per_update=global_rows,caps=caps,
        base_model=bundle['base']['model'],base_revision=BASE,base_binding=bundle['base_binding'],base_binding_sha256=sha(bundle['base_binding']),base_metadata=bundle['base_metadata'],recipe=rec.get('name','D'),
        balance_batch=rec.get('balance_batch',False),penalty=penalty,measurement=measurement,incoming=incoming,incoming_metadata=model_metadata(incoming['model'],incoming['files']),
        fisher_dir=bundle.get('fisher_dir'),fisher_manifest_sha256=sha(Path(bundle['fisher_dir'])/'manifest.json') if bundle.get('fisher_dir') else None,receipt=str(root/'native-receipt.json'))
    overrides+=['+nm.'+key+'='+json.dumps(value) for key,value in n.items()]
    env=dict(PYTHONPATH=str(ROOT)+os.pathsep+str(Path(bundle['sdpo'])),KIT_FINISH_GATE='1',
        WANDB_MODE='disabled',V4_CHECKPOINT_STATE='1',V4_PROFILE='scientific',ARM='D',KIT_V4_ARM='D',
        SDPO_DIR=bundle['sdpo'],VERL_FILE_LOGGER_PATH=str(root/'metrics.jsonl'))
    for key,value in env.items():overrides.append('++ray_kwargs.ray_init.runtime_env.env_vars.'+key+'='+json.dumps(value))
    command=[bundle.get('trainer_python',sys.executable),'-I',str(KIT/'nm_entry.py'),'--config-name','sdpo',*overrides]
    metadata=dict(command=command,env=env,cwd=bundle['sdpo'],audit=audit,native=n,incoming=incoming,files={name+'.parquet':sha(data/(name+'.parquet')) for name in ('train','test')})
    seal(root/'prepared.json',metadata);return metadata


def export_stage(prepared,root,python):
    root=Path(root);need(load(root/'native-receipt.json')['status']=='COMPLETE','native stage incomplete')
    exports={};native=Path(prepared['native']['receipt']).parent/'native'
    for step in STEPS:
        if step==0:exports['0']=prepared['incoming'];continue
        source=native/f'global_step_{step}'/'actor';ready(source.is_dir(),'native export checkpoint absent')
        target=root/f'hf-step{step}'
        subprocess.run([python,'-m','verl.model_merger','merge','--backend','fsdp','--local_dir',str(source),'--target_dir',str(target)],
            cwd=prepared['cwd'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        files={p.name:sha(p) for p in sorted([*target.glob('*.safetensors'),*target.glob('*.safetensors.index.json')])};ready(files,'CPU export weights absent')
        files['config.json']=sha(target/'config.json')
        exports[str(step)]=dict(model=str(target),sha256=digest(files),files=files)
    seal(root/'exports.json',exports);return exports
