"""Real CPU/GPU phase operations for the six fixed next-message comparisons."""
from pathlib import Path
import os
import subprocess
import sys
import time
from kit.nm_common import *
from kit import nm_data,nm_prepare


def source(out,family):
    from kit.nm_jobs import current
    current(out,'NM-'+family+'-bind')
    return load(out/'sources'/(family+'.json'))
def bound(out,family):
    from kit.nm_jobs import current
    current(out,'NM-'+family+'-bind')
    return load(out/'bindings'/(family+'.json'))
def exports_for(out,family,arm,seed):
    from kit.nm_jobs import current
    current(out,f'NM-{family}-{arm}-s{seed}-export')
    return load(out/'runs'/family/f'{arm}-s{seed}'/'exports.json')
def stage_root(out,row,qualification=False):
    return out/('qualification' if qualification else 'runs')/row['family']/f"{row['arm']}-s{row['seed']}"


def row_result(out,task):
    from kit.nm_jobs import current
    record=current(out,task);return out/'attempts'/task/record['nonce']


def cpu_base_binding(bundle,out):
    record=nm_prepare.verify_model(bundle['base']);path=out/'base.json'
    seal(path,dict(model=record['model'],revision=BASE,identity=dict(model_file_hashes=record['files'])))
    return str(path)


def training_rows(out,row,bundle):
    family=row['family'];seed=row['seed']
    if family=='grpo':
        rows=nm_data.grpo_schedule(bundle['training_candidates'],seed)
        return [dict(r,response=r['demonstration']) for r in rows]
    if family=='v2b':
        row_result(out,'NM-v2b-august-seal')
        ledger=load(out/'v2b'/'ledger.json');rows,_=nm_data.v2b_schedule(ledger,seed)
        from transformers import AutoTokenizer
        from kit.beds import finqa
        tok=AutoTokenizer.from_pretrained(bundle['tokenizer'],local_files_only=True)
        result=[]
        for r in rows:
            item=dict(r['item'],month=r['month'])
            context=nm_data.v2b_context(item,ledger,tok,boundary=True)
            result.append(dict(r,prompt=finqa.render_prompt(item),gold=finqa.gold_of(item),teacher_context=context['context'],retrieval=context))
        ready(any(r['teacher_context'] for r in result),'zero context support')
        return result
    rows=bundle['training_rows'][str(seed)]
    rec=bound(out,family)['recipe'];need(digest(rows)==rec['schedule_hashes'][str(seed)],'bound schedule differs')
    if family=='opr' and row['arm']=='T':
        refreshed=load(out/'opr'/f'refresh-s{seed}.json')
        lookup={str(r['id']):r['response'] for r in refreshed['rows']}
        rows=[dict(r,response=lookup[str(r['id'])]) if r.get('domain')=='finqa' and r.get('m3_replay') else r for r in rows]
    return rows


def run_native(bundle,root,measurement=False):
    p=load(root/'prepared.json');env=dict(os.environ,**p['env']);started=time.monotonic()
    from kit.a4_common import gpu_run
    if (root/'native-receipt.json').is_file() and load(root/'native-receipt.json').get('status')==('CALIBRATED' if measurement else 'COMPLETE'):
        ready((root/'native-timing.json').is_file(),'original native timing receipt absent')
        return dict(load(root/'native-timing.json'),reused_complete=True)
    if (root/'started.json').exists():
        partial=root/'partials'/os.environ['NM_ALLOCATION_NONCE'];partial.mkdir(parents=True,exist_ok=False)
        for name in ('native','metrics.jsonl','started.json','native-receipt.json','native-timing.json'):
            if (root/name).exists():(root/name).rename(partial/name)
    atomic(root/'started.json',dict(incoming=p['incoming'],nonce=os.environ['NM_ALLOCATION_NONCE']))
    result=gpu_run(p['command'],cwd=p['cwd'],env=env,pass_fds=tuple(json.loads(os.environ.get('M3_GPU_LOCK_FDS','[]'))))
    receipt=root/'native-receipt.json'
    if measurement:
        ready(receipt.is_file(),'native calibration receipt absent')
        need(load(receipt).get('status')=='CALIBRATED','calibration mutated native state or failed')
        # Measurement intentionally exits from preclip before native optimizer/scheduler/EMA.
    else:
        need(result.returncode==0,'native trainer failed; retain attempt and restart/resume same seed')
        ready(receipt.is_file(),'native completion receipt absent')
    timing=dict(native_receipt=str(receipt),wall_seconds=time.monotonic()-started)
    seal(root/'native-timing.json',timing);return timing


def prepare_lane(out,family,seed,panel,bundle,*,cohort_only=False):
    from kit.nm_score import scorer_authority,file_stats
    binding=bound(out,family);rec=binding.get('recipe',{})
    incoming=bundle['base'] if family in ('grpo','v2b') else rec['starts'][str(seed)]
    sources=[('base',0,bundle['base']),('incoming',0,incoming)]
    if not cohort_only:
        if family=='v2b' and panel not in (*GENERAL,'hellaswag'):sources.append(('memory',0,bundle['base']))
        for arm in ('C','T'):
            exports=exports_for(out,family,arm,seed)
            for step in STEPS if panel=='legal' or (family=='grpo' and panel=='finqa') else (20,40):sources.append((arm,step,exports[str(step)]))
    path=out/'plans'/family/(f'cohort-s{seed}' if cohort_only else f's{seed}-{panel}')
    path.mkdir(parents=True,exist_ok=True);commands=[];cells=[]
    items=bundle['panels'][panel]
    for index,(arm,step,src) in enumerate(sources):
        if 'files' not in src:src=nm_prepare.verify_model(src)
        cell_items=items
        if family=='v2b' and panel not in (*GENERAL,'hellaswag') and ('on' in panel or arm=='memory'):
            row_result(out,'NM-v2b-'+panel.split('-')[0]+'-contexts')
            contexts=load(out/'v2b'/('contexts-'+panel.split('-')[0]+'.json'))
            cell_items=[dict(r,prompt=contexts[r['id']]['on']) for r in items]
        plan=dict(seed=seed,panel=panel,model=arm,step=step,source=src,items=cell_items,
            tokenizer=bundle['tokenizer'],metadata=file_stats(src),authority=scorer_authority(panel))
        if panel not in ('knowledge','hellaswag'):
            plan['rendered']=nm_prepare.render_scoring(cell_items,bundle['tokenizer'],panel)
        name=path/(str(index)+'.json');seal(name,plan)
        output_name=str(index)+'.score.json';cells.append(dict(plan=str(name),plan_sha256=sha(name),output=output_name))
        command=[bundle.get('inference_python',sys.executable),'-I',str(KIT/'nm_score.py'),'--plan',str(name),'--out','{attempt}/'+output_name]
        engine=dict(backend='hf',profile='reference-bf16') if panel in ('knowledge','hellaswag') else dict(backend='vllm',model=src['model'],max_model_len=12288 if panel not in ('instruction','maths') else 4096)
        # Full-precision references are serial unless that exact worker has measured approval.
        commands.append(dict(argv=command,requires=[str(Path(src['model'])/'config.json')],**({} if panel in ('knowledge','hellaswag') else dict(engine=engine))))
    qualification_binding=dict(family=family,panel=panel,cohort_only=cohort_only,
        panel_sha256=digest(items),scorer=scorer_authority(panel),base_revision=BASE,
        engines=[{k:v for k,v in c.get('engine',{}).items() if k!='model'} for c in commands],
        implementation_sha256=digest(source_seal()))
    qualification=bundle.get('lane_qualifications',{}).get(('cohort-' if cohort_only else '')+panel)
    concurrency=qualified_concurrency(3 if panel not in ('knowledge','hellaswag') and not cohort_only else 1,qualification,qualification_binding)
    seal(path/'lane.json',dict(commands=commands,cells=cells,lane_concurrency=concurrency,
        qualification=qualification if concurrency==3 else None,qualification_binding=qualification_binding,
        concurrency_reason='bound K=3 qualification' if concurrency==3 else 'K=1: no applicable K=3 qualification'))
    return path


def qualified_concurrency(requested,ref,binding):
    if requested!=3 or not ref:return 1
    try:q=checked(ref)
    except Pending:return 1
    need(q.get('schema')=='nm-lane-qualification.v1' and q.get('binding_sha256')==digest(binding),'scoring qualification binding differs')
    passed=(q.get('concurrency')==3 and q.get('telemetry_complete') is True and q.get('numeric_match') is True
        and q.get('memory_safe') is True and all(type(q.get(k)) in (int,float) and math.isfinite(q[k]) and 70<=q[k]<=100 for k in ('mean_util','min_window_mean')))
    return 3 if passed else 1


def run_lane(plan,attempt,*,concurrency=1):
    from kit.lane_pool import execute
    p=load(Path(plan)/'lane.json')
    for cell in p['cells']:need(sha(cell['plan'])==cell['plan_sha256'],'CPU scoring plan changed')
    commands=[dict(c,argv=[x.replace('{attempt}',str(attempt)) for x in c['argv']]) for c in p['commands']]
    missing=[name for c in commands for name in c.get('requires',[]) if not Path(name).is_file()]
    ready(not missing,'complete scoring frontier not ready: '+str(missing[:3]))
    actual=qualified_concurrency(concurrency,p.get('qualification'),p.get('qualification_binding'))
    need(actual<=min(concurrency,p.get('lane_concurrency',1)),'scoring concurrency exceeds CPU preparation')
    atomic(attempt/'lane-concurrency.json',dict(requested=concurrency,actual=actual,
        reason='bound K=3 qualification' if actual==3 else 'K=1: row cap or no applicable bound qualification',qualification=p.get('qualification') if actual==3 else None))
    code=execute(commands,attempt/'pool',env=dict(os.environ),pass_fds=tuple(json.loads(os.environ.get('M3_GPU_LOCK_FDS','[]'))),concurrency=actual,stagger_seconds=20)
    ready(code not in (3,75),'scoring lane prerequisites incomplete');need(code==0,'scoring lane failed')
    cells=[]
    for c in p['cells']:cells.append(load(attempt/c['output'])['cell'])
    atomic(attempt/'scores.json',dict(cells=cells));return cells


def estimator_plan(bundle,out):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(bundle['tokenizer'],local_files_only=True)
    rows=nm_data.fisher_rows(bundle['fisher_domains']);converted=[]
    from kit.eval_bed import render,render_messages
    for r in rows:
        prompt=render_messages(tok,r['prompt']) if isinstance(r['prompt'],list) else render(tok,r['prompt'])
        x=tok.encode(prompt,add_special_tokens=False);y=tok.encode(r['response']+tok.eos_token,add_special_tokens=False)
        need(x and 0<len(y)<=2048 and y[-1]==tok.eos_token_id and tok.eos_token_id not in y[:-1],'Fisher response/cap differs')
        need(len(x)<=10240,'Fisher proxy prompt exceeds cap')
        full=x+y;inputs=dict(input_ids=[full],attention_mask=[[1]*len(full)],position_ids=[list(range(len(full)))],responses=[y],response_mask=[[1]*len(y)])
        converted.append(dict(r,inputs=inputs))
    pre=[];longest=[]
    for d,n in [('instruction',16),('maths',8),('knowledge',8)]:
        domain=[r for r in converted if r['domain']==d];pre.extend(domain[:n]);longest.append(max(domain,key=lambda r:len(r['inputs']['input_ids'][0])))
    path=out/'fisher'/'plan.json';seal(path,dict(base_model=bundle['base']['model'],base_binding=bundle['base_binding'],base_binding_sha256=sha(bundle['base_binding']),base_metadata=bundle['base_metadata'],rows=converted,preflight_rows=pre,longest_rows=longest));return path


def operation(cfg,row,attempt,out,config_path):
    family=row['family'];phase=row['phase'];published=[];base=out/'prepared'/family
    def publish(path,value,immutable=True):
        (seal if immutable else atomic)(path,value);published.append(str(path.relative_to(out)));return value
    if phase=='bind':
        from kit.nm_bind import row_binding
        ready(family in cfg.get('inputs',{}),'bound source package absent for '+family)
        bundle=checked(cfg['inputs'][family]);binding_spec=checked(bundle['binding_spec']) if family not in ('grpo','v2b') else {}
        b=row_binding(family,binding_spec,incumbent_path=out/'incumbent.json')
        publish(out/'sources'/(family+'.json'),bundle)
        publish(out/'bindings'/(family+'.json'),b)
        publish(out/'launch'/(family+'.json'),dict(config_sha256=digest(launch_config(cfg,family)),code=source_seal(),hosts=cfg.get('hosts',[]),
            infrastructure_update=cfg['infrastructure_update'],owner_approval='2 November 2026: Approve all six + allocation'))
        publish(out/'launch.json',dict(code=source_seal(),hosts=cfg.get('hosts',[]),infrastructure_update=cfg['infrastructure_update']))
        return dict(status=b['status'],published=published)
    bundle=source(out,family);binding=bound(out,family)
    if phase=='prepare':
        audit=nm_prepare.audit_bundle(bundle,binding)
        from kit.nm_model import metadata
        base_binding=cpu_base_binding(bundle,base)
        bundle=dict(bundle,base_binding=base_binding,base_metadata=metadata(bundle['base']['model'],load(base_binding)['identity']['model_file_hashes']))
        publish(base/'bundle.json',bundle);publish(base/'audit.json',audit)
        if family=='fisher':estimator_plan(bundle,out)
        if family=='opr':
            for seed in SEEDS:
                rows=[r for r in bundle['training_rows'][str(seed)] if r.get('m3_replay') and r.get('domain')=='finqa']
                ready(rows,'bound earlier-task replay absent')
                unique={str(r['id']):r for r in rows};incoming=nm_prepare.verify_model(binding['recipe']['starts'][str(seed)])
                from transformers import AutoTokenizer
                from kit.eval_bed import render,render_messages
                tok=AutoTokenizer.from_pretrained(bundle['tokenizer'],local_files_only=True)
                rendered={i:render_messages(tok,r['prompt']) if isinstance(r['prompt'],list) else render(tok,r['prompt']) for i,r in unique.items()}
                need(all(len(tok.encode(v,add_special_tokens=False))<=10240 for v in rendered.values()),'generation prompt over cap')
                generation=dict(seed=seed,model=incoming['model'],generator_sha256=incoming['sha256'],model_metadata=metadata(incoming['model'],incoming['files']),rendered_prompts=rendered,tokenizer=bundle['tokenizer'],
                    prompts=list(unique.values()),prompt_cap=10240,memory_fraction=bundle['generation']['memory_fraction'],batch_size=bundle['generation']['batch_size'])
                publish(out/'opr'/f'generation-s{seed}.json',generation)
                # Bound a single ready batch for mechanics, not another whole-pool refresh.
                qids=bundle['generation'].get('qualification_ids',{}).get(str(seed))
                ready(qids and len(set(qids))==len(qids) and set(qids)<=set(unique),'generation qualification batch absent')
                need(len(qids)==min(len(unique),generation['batch_size']),'generation qualification must cover the actual batch size')
                publish(out/'opr'/f'generation-qualify-s{seed}.json',dict(generation,prompts=[unique[i] for i in qids]))
                prepare_lane(out,family,seed,'finqa',bundle,cohort_only=True)
    elif phase in ('train-prepare','qualify-prepare','calibrate-prepare'):
        bundle=load(base/'bundle.json');rows=training_rows(out,row,bundle)
        if family=='fisher':
            bundle['fisher_dir']=str(out/'fisher'/'estimate')
            if phase!='calibrate-prepare':bundle['calibration']=load(out/'fisher'/'calibration.json')
        root=stage_root(out,row,phase=='qualify-prepare')
        if phase=='calibrate-prepare':root=out/'fisher'/f'calibrate-s{row["seed"]}'
        nm_prepare.training_command(bundle,binding,rows,root,family,row['arm'],row['seed'],qualification=phase=='qualify-prepare',measurement=phase=='calibrate-prepare')
        published.append(str((root/'prepared.json').relative_to(out)))
    elif phase in ('train','qualify','calibrate'):
        bundle=load(base/'bundle.json');root=stage_root(out,row,phase=='qualify')
        if phase=='calibrate':root=out/'fisher'/f'calibrate-s{row["seed"]}'
        if phase=='train':
            # A bound send/qualification is required; building never fabricates one.
            ready(cfg.get('send_approval'),'separate send approval absent')
        result=run_native(bundle,root,phase=='calibrate')
        publish(attempt/'timing.json',result);published.append(str((root/'native-receipt.json').relative_to(out)))
    elif phase=='qualify-read':
        root=stage_root(out,row,True);native=load(root/'native-receipt.json')
        need(native['updates']==8,'qualification dose incomplete')
        # Independent mathematical, longest-row, nonzero-LR and native-resume
        # evidence is bound by the production qualification package, never inferred
        # from a successful trainer exit or a CPU stand-in.
        q=checked(bundle.get('qualification',{}).get(row['arm'],{}))
        required=('independent_gradient','loss_reference','world_1_2_8','longest_input','optimizer_materialized','native_resume_equal','export_reload_equal','zero_nonzero_LR','no_reference_gradients')
        ready(all(q.get(k) is True for k in required),'production qualification gates incomplete')
        need(q.get('source_sha256')==digest(source_seal()) and q.get('schedule_sha256')==load(root/'prepared.json')['audit']['schedule_sha256'],'qualification implementation/schedule differs')
        publish(root/'qualified.json',q)
    elif phase=='export':
        root=stage_root(out,row);p=load(root/'prepared.json');nm_prepare.export_stage(p,root,bundle.get('trainer_python',sys.executable))
        published.append(str((root/'exports.json').relative_to(out)))
    elif phase=='score-prepare':
        path=prepare_lane(out,family,row['seed'],row['panel'],bundle);published.append(str((path/'lane.json').relative_to(out)))
    elif phase=='score':
        plan=out/'plans'/family/f's{row["seed"]}-{row["panel"]}';run_lane(plan,attempt,concurrency=row['lane_concurrency']);published.append(str((attempt/'scores.json').relative_to(out)))
    elif phase=='cohort':
        cells=run_lane(out/'plans'/family/f'cohort-s{row["seed"]}',attempt,concurrency=row['lane_concurrency'])
        vectors={c['model']:c['items'] for c in cells};nm_data.cohort(vectors['base'],vectors['incoming'])
        publish(attempt/'cohort.json',dict(**vectors,before_training=True,identity=[{k:c[k] for k in ('nonce','uuid','environment','checkpoint_sha256')} for c in cells]))
    elif phase in ('generation-qualify','generate'):
        from kit.nm_generate import generate
        row_result(out,'NM-opr-prepare')
        if phase=='generate':
            q=row_result(out,f'NM-opr-generation-qualify-s{row["seed"]}')
            util=load(q/'util.summary.json')
            ready(util.get('telemetry_complete') and util.get('mean_util',0)>=70 and (util.get('min_window_mean') or 0)>=70,'generation utilisation qualification PENDING')
        plan_name='generation-qualify' if phase=='generation-qualify' else 'generation'
        generate(out/'opr'/f'{plan_name}-s{row["seed"]}.json',attempt/'generation.json',phase=='generation-qualify')
        published.append(str((attempt/'generation.json').relative_to(out)))
    elif phase=='refresh':
        gen=row_result(out,f'NM-opr-generate-s{row["seed"]}');cohort=row_result(out,f'NM-opr-cohort-s{row["seed"]}')
        rows=[r for r in bundle['training_rows'][str(row['seed'])] if r.get('m3_replay') and r.get('domain')=='finqa']
        refreshed,audit=nm_data.refresh(rows,load(gen/'generation.json')['attempts'],row['seed'])
        publish(out/'opr'/f'refresh-s{row["seed"]}.json',dict(rows=refreshed,audit=audit,cohort=load(cohort/'cohort.json')))
    elif phase in ('estimator-preflight','estimate'):
        root=out/'fisher'/('preflight' if phase=='estimator-preflight' else 'estimate');root.mkdir(parents=True,exist_ok=True)
        command=[bundle.get('trainer_python',sys.executable),'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',
            str(KIT/'nm_fisher.py'),'--plan',str(out/'fisher'/'plan.json'),'--action','preflight' if phase=='estimator-preflight' else 'estimate','--out',str(root)]
        result=subprocess.run(command,env=dict(os.environ,PYTHONPATH=str(ROOT)),pass_fds=tuple(json.loads(os.environ.get('M3_GPU_LOCK_FDS','[]'))))
        need(result.returncode==0,'Fisher native estimator failed')
        published.extend(str(p.relative_to(out)) for p in root.glob('rank-*.json'))
    elif phase=='estimator-publish':
        root=out/'fisher'/'estimate';records=[load(root/f'rank-{rank}.json') for rank in range(8)]
        need(all(r['examples']==256 and r['base_revision']==BASE for r in records),'Fisher native shard mismatch')
        from kit.nm_fisher_stats import summarize
        summary=summarize(sorted(root.glob('rank-*.pt')))
        need(abs(summary['mean']-1)<=1e-5,'global Fisher normalisation differs')
        publish(root/'manifest.json',dict(base_revision=BASE,files={p.name:sha(p) for p in root.glob('rank-*.pt')},records=records,summary=summary,metadata={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in root.glob('rank-*.pt')}))
    elif phase=='seal-calibration':
        from kit.nm_loss import calibrate
        from kit.nm_fisher import price
        records={str(s):load(out/'fisher'/f'calibrate-s{s}'/'native-receipt.json') for s in SEEDS}
        result=calibrate(records,binding.get('recipe',{}).get('p2_lambda') if not binding['benchmark'] else None)
        timings={str(s):load(row_result(out,f'NM-fisher-calibrate-s{s}')/'timing.json')['wall_seconds'] for s in SEEDS}
        pre=load(out/'fisher'/'preflight'/'rank-0.json')
        ready(bundle.get('fisher_pricing'),'measured startup/export/diagnostic/retry costs absent')
        pricing=bundle['fisher_pricing'];result['measured_E']=price(pre,timings,pricing['startup_export_gpu_hours'],pricing['reserve'])
        publish(out/'fisher'/'calibration.json',result)
    elif phase in ('contexts','arrival','ledger','schedule'):
        return development(cfg,row,attempt,out,bundle,published)
    elif phase=='account':
        from kit.nm_budget import reconcile
        result=reconcile(out,family=family)
        publish(attempt/'accounting.json',result)
        ready(result['status']=='COMPLETE','actual allocation ends pending')
    elif phase=='read':
        return reading(row,attempt,out,bundle,published)
    elif phase=='collect':
        return collect(row,attempt,out,bundle,published)
    else:raise Invalid('unimplemented explicit NM phase '+phase)
    return dict(status='COMPLETE',published=published)


def development(cfg,row,attempt,out,bundle,published):
    from kit.nm_development import operation
    return operation(row,attempt,out,bundle)


def reading(row,attempt,out,bundle,published):
    from kit.nm_jobs import inventory,current
    from kit.nm_read import read
    from kit.nm_score import scorer_authority
    family=row['family'];cells=[];serving=[]
    for r in inventory()['rows']:
        if r['family']==family and r['phase']=='score':
            path=row_result(out,r['id']);cells.extend(load(path/'scores.json')['cells'])
            for cell_path in sorted(path.glob('*.score.json')):
                scored=load(cell_path);cell=scored['cell'];raw=scored.get('raw',[])
                correct=sum(cell['items'].values());tokens=sum(v['output_tokens'] for v in raw)
                if raw:serving.append(dict(seed=cell['seed'],panel=cell['panel'],model=cell['model'],step=cell['step'],output_tokens=tokens,correct=correct,tokens_per_correct=tokens/correct if correct else None))
    spec=dict(panels={p:[i['id'] for i in items] for p,items in bundle['panels'].items()},
        scorers={p:scorer_authority(p) for p in bundle['panels']},checkpoints={},pre_cohort={})
        # Expected checkpoint identities come from CPU export receipts, never score claims.
    for seed in SEEDS:
        start=bundle['base'] if family in ('grpo','v2b') else bound(out,family)['recipe']['starts'][str(seed)]
        expected=dict(base={'0':bundle['base']['sha256']},incoming={'0':start['sha256']})
        if family=='v2b':expected['memory']={'0':bundle['base']['sha256']}
        for arm in ('C','T'):expected[arm]={k:v['sha256'] for k,v in exports_for(out,family,arm,seed).items()}
        spec['checkpoints'][str(seed)]=expected
        if family=='opr':
            current(out,f'NM-opr-refresh-s{seed}')
            refresh=load(out/'opr'/f'refresh-s{seed}.json')
            spec['pre_cohort'][str(seed)]=refresh['cohort']
            spec.setdefault('opr_refresh',{})[str(seed)]=refresh['audit']
    if family=='fisher':spec['fisher_calibration']=load(out/'fisher'/'calibration.json')
    for item in serving:
        bases=[v for v in serving if v['seed']==item['seed'] and v['panel']==item['panel'] and v['model']=='base']
        reference=bases[0]['tokens_per_correct'] if bases else None
        item['at_cost']=item['tokens_per_correct']>1.5*reference if reference and item['tokens_per_correct'] is not None else None
    binding=bound(out,family);scores=dict(cells=cells,serving=serving)
    atomic(attempt/'reader-spec.json',spec);atomic(attempt/'scores.json',scores);atomic(attempt/'binding.json',binding)
    result=read(family,scores,spec,binding);atomic(attempt/'reading.json',result)
    lines=text_lines(family,result,load(out/'launch.json')['infrastructure_update']);(attempt/'summary.txt').write_text('\n'.join(lines)+'\n')
    ready(not result['status'].startswith('PENDING') or result.get('primary'),result.get('reason','reader incomplete'))
    published.extend(str((attempt/name).relative_to(out)) for name in ('reading.json','summary.txt','reader-spec.json','scores.json','binding.json'))
    return dict(status='COMPLETE',published=published,reading_status=result['status'])


def collect(row,attempt,out,bundle,published):
    import tarfile
    from kit.nm_budget import reconcile
    accounting=reconcile(out,family=row['family'])
    atomic(attempt/'accounting.json',accounting)
    ready(accounting['status']=='COMPLETE','actual scheduler ends pending for cost accounting')
    family=row['family'];reading_root=row_result(out,'NM-'+family+'-read')
    from collections import Counter,defaultdict
    by_phase=defaultdict(lambda:dict(attempts=0,gpu_hours=0,statuses=Counter()))
    receipts=[load(p) for p in (out/'attempts').glob('NM-'+family+'-*/*/receipt.json')]
    # The collecting attempt is still running; label it explicitly rather than
    # claiming it has already completed. CPU attempts consume zero GPU-hours.
    for r in receipts:
        v=by_phase[r['phase']];v['attempts']+=1;v['statuses'][r['status']]+=1
        v['gpu_hours']+=r.get('allocation_gpu_hours') or 0
    accounting.update(attempts=len(receipts),by_phase=dict(by_phase))
    atomic(attempt/'accounting.json',accounting)
    reading=load(reading_root/'reading.json')
    from kit.report_text import report
    launch=load(out/'launch.json')
    full_paths=report(dict(_mode='nm',family=family,reading=str(reading_root/'reading.json'),
        accounting=str(attempt/'accounting.json'),infrastructure_update=launch['infrastructure_update']),attempt/'family-detail',
        standin=launch.get('technical_synthetic',False),full_detail=True)
    paths=dispatch_text(out,attempt,dict(nm_family=family,reading=reading,accounting=accounting,
        infrastructure_authority=launch['infrastructure_update']))
    text=''.join(p.read_text() for p in paths);(attempt/'return.txt').write_text(text,encoding='ascii')
    from kit.nm_text import parse_parts
    imported=parse_parts([text])['families'][family];need(imported['status']==reading['status'],'pasted text verdict does not reproduce')
    if reading.get('simulated_status'):need(imported.get('simulated_status')==reading['simulated_status'],'pasted simulated verdict differs')
    published.extend(str(p.relative_to(out)) for p in [attempt/'return.txt',attempt/'return-evidence.json',attempt/'accounting.json',*paths,*full_paths])
    if not bundle.get('return_archive'):
        return dict(status='COMPLETE',published=published,accounting=accounting,text_parts=len(paths))
    files=[reading_root/'reading.json',reading_root/'summary.txt',out/'bindings'/(family+'.json'),out/'launch.json']
    if (out/'launch'/(family+'.json')).is_file():files.append(out/'launch'/(family+'.json'))
    # Retain every attempt's telemetry, including failures; omit native weights.
    for root in (out/'attempts').glob('NM-'+family+'-*/*'):
        files.extend(p for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.csv') and p.name!='archive-manifest.json')
    files.extend(reading_root/name for name in ('reader-spec.json','scores.json','binding.json'))
    files.append(attempt/'accounting.json')
    manifest=dict(schema=SCHEMA,family=family,files={str(p.relative_to(out)):sha(p) for p in sorted(set(files))},
        **{key:str((reading_root/name).relative_to(out)) for key,name in [('scores','scores.json'),('spec','reader-spec.json'),('binding','binding.json'),('reading','reading.json')]})
    atomic(attempt/'archive-manifest.json',manifest);files.append(attempt/'archive-manifest.json')
    archive=attempt/'return.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(set(files)):tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
    with tarfile.open(archive) as tar:need(len(tar.getnames())==len(set(files)),'archive coverage differs')
    from kit.nm_return import reproduce
    reproduce(archive)
    published.append(str(archive.relative_to(out)))
    return dict(status='COMPLETE',published=published,archive_sha256=sha(archive),accounting=accounting)


def dispatch_text(out,attempt,full):
    """Replace this family's result in one cumulative, budgeted return cycle.

    The pointer commits only after a complete text set passes its self-check.
    Serialising this small CPU publication also covers concurrent collectors.
    Previous attempt artifacts are immutable; paste only the newest cycle.
    """
    import fcntl
    from kit.nm_text import write_parts
    from kit.report_text import kit_identity
    evidence=attempt/'return-evidence.json';atomic(evidence,full)
    with (out/'.return-cycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        pointer=out/'return-cycle.json'
        refs=load(pointer)['evidence'] if pointer.exists() else {}
        refs[full['nm_family']]=dict(path=str(evidence),sha256=sha(evidence))
        families=[checked(refs[name]) for name in (*FAMILIES,'cr') if name in refs]
        tag,commit=kit_identity()
        paths=write_parts(attempt/'text',dict(nm_dispatch=families),tag,commit)
        atomic(pointer,dict(schema='nm-return-cycle.v1',evidence=refs,families=[v['nm_family'] for v in families],
            parts=[str(p.relative_to(out)) if p.is_relative_to(out) else str(p) for p in paths],characters=sum(len(p.read_text()) for p in paths)))
        return paths
