"""Seventh v7 comparison: explicit CR/G rows using the real NM allocation wrapper."""
from pathlib import Path
import os
import json
import sys
from types import SimpleNamespace as NS
from kit.nm_common import *
from kit.cr_confirmation import SEEDS as CR_SEEDS, ARMS, FINQA_PENDING, isolated, bind_inputs, verify_diff, verify_standin
from kit.m3_schedule import STAGES


def settings(cfg):
    cr=dict(cfg.get('cr_confirmation',{}))
    ready(cr.get('out'),'separate cr_confirmation.out required')
    isolated(cr,cfg['out'])
    cr.update(budget=cr.get('budget',{}),job_python=cfg.get('job_python',sys.executable),
        infrastructure_update=cfg.get('infrastructure_update'),send_approval=cfg.get('send_approval'),
        technical_synthetic=cfg.get('technical_synthetic',False),dispatch_out=cfg['out'])
    verify_standin(cr)
    return cr


def inventory():
    rows=[]
    def add(name,phase,deps=(),gpus=0,**kw):
        ident='NM-cr-'+name
        rows.append(dict(id=ident,family='cr',group='cr-confirmation',phase=phase,gpus=gpus,nodes=1,
            cpus=32 if gpus==8 else 8,may_share_node=gpus!=8,
            depends_on=[dict(id=d,type='afterany' if phase in ('read','account','collect') else 'afterok') for d in deps],
            dependency_options=['--kill-on-invalid-dep=yes'],lane_concurrency=2 if phase=='score' else 1,
            command=['{job_python}','-I','{kit}/nm_jobs.py','run','--config','{nm_config}','--task',ident],**kw))
        return ident
    binding=add('bind','bind')
    prep=add('prepare','prepare',[binding])
    qualifications=[]
    for arm in ARMS:
        qp=add('qualify-'+arm+'-prepare','qualify-prepare',[prep],arm=arm,seed=104,stage='contract_nli')
        q=add('qualify-'+arm,'qualify',[qp],8,arm=arm,seed=104,stage='contract_nli')
        qualifications.append(add('qualify-'+arm+'-read','qualify-read',[q],arm=arm))
    exports={}
    for seed in CR_SEEDS:
        exports[seed]=[]
        for arm in ARMS:
            prior=prep
            for stage in STAGES:
                name=f'{arm}-s{seed}-{stage}'
                p=add(name+'-prepare','train-prepare',[prior,*qualifications],arm=arm,seed=seed,stage=stage)
                t=add(name+'-train','train',[p],8,arm=arm,seed=seed,stage=stage)
                prior=add(name+'-export','export',[t],arm=arm,seed=seed,stage=stage)
                exports[seed].append(prior)
    assign=add('assign','assign',[prep])
    scores=[]
    for seed in CR_SEEDS:
        for lane in ('native','reference'):
            p=add(f'{lane}-s{seed}-prepare','score-prepare',[assign,*exports[seed]],seed=seed,lane=lane)
            scores.append(add(f'{lane}-s{seed}','score',[p],1,seed=seed,lane=lane,
                assignment=f'm3|{seed}' if lane=='native' else f'm3.reference|{seed}',
                expected_score_cells=76 if lane=='native' else 26))
    account=add('account','account',scores)
    read=add('read','read',[account,*scores])
    add('collect','collect',[read])
    return rows


def args(cfg,row,out,action):
    stage=row.get('stage');root=out/('targets-'+stage) if stage else None
    return NS(out=out,action=action,standin=bool(cfg.get('technical_synthetic')),arm=row.get('arm'),seed=row.get('seed'),
        stage=stage,schedule_root=root,schedule_roots=[out/('targets-'+s) for s in STAGES],
        stages=list(STAGES),arms=list(ARMS),tokenizer=cfg.get('tokenizer'),panel=cfg.get('panels'),
        holdout=cfg.get('holdouts'),superset=None,sdpo=Path(cfg['sdpo']) if cfg.get('sdpo') else None,
        trainer_python=cfg.get('trainer_python',sys.executable),gpu=os.environ.get('CUDA_VISIBLE_DEVICES',''),
        host_reservation_bytes=cfg.get('host_reservation_bytes'))


def native_commands(cfg,out,seed):
    """Exact v6a generated-panel/curve layout, restricted to G and CR."""
    from kit.m3_read import TASK_METRIC
    commands=[]
    panels=('finqa','test','math_hard','general-v1')
    def put(arm,stage,step,panel):
        c=[cfg['job_python'],'-I',str(KIT/'m3_score.py'),'--action','score','--out',str(out),
            '--arm',arm,'--seed',str(seed),'--stage',stage,'--step',str(step),'--panel-set',panel,'--gpu','0']
        if cfg.get('finqa_panel'):c+=['--panel-root',cfg['finqa_panel']]
        if cfg.get('technical_synthetic'):c+=['--standin']
        commands.append(c)
    for panel in panels:put('base',STAGES[0],0,panel)
    for arm in ARMS:
        for n,stage in enumerate(STAGES):
            layout={0:panels,10:[TASK_METRIC[stage]],20:[TASK_METRIC[stage],*[TASK_METRIC[e] for e in STAGES[:n]]],30:[TASK_METRIC[stage]],40:panels}
            for step,selected in layout.items():
                for panel in selected:put(arm,stage,step,panel)
    return commands


def checkpoints(seed):
    return [dict(arm='base',seed=seed,stage=STAGES[0],step=0),
        *[dict(arm=arm,seed=seed,stage=stage,step=k) for arm in ARMS for stage in STAGES for k in (0,40)]]


def prepare(cfg,out):
    """Reuse bytes; derive new schedules only after proving old source/order identity."""
    from kit import m3_manifest, m3_train
    from kit.p3b_common import files_manifest, t, cycle
    bound=load(out/'confirmation-binding.json');doc=bound['realised']
    check=out/'input-checks'/__import__('uuid').uuid4().hex
    observed=bind_inputs(cfg,check)
    ready(observed==bound,'realised inputs changed; owner escalation')
    m3_manifest.init_root(out,core_submitted=True)
    for source,target in ((Path(doc['pools_manifest']).parent,out/'pools'),
                          (Path(doc['hellaswag_manifest']).parent,out/'hellaswag')):
        if not target.exists():target.symlink_to(source,target_is_directory=True)
        need(target.resolve()==source.resolve(),'realised source mount changed')
    for source,target in ((doc['base_binding'],out/'inputs/base.json'),(doc['math_source'],out/'targets-math-source.json')):
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():need(target.read_bytes()==Path(source).read_bytes(),'realised binding changed')
        else:target.write_bytes(Path(source).read_bytes())
    for stage,source in doc['task_sources'].items():
        if stage=='finqa':
            from kit.cr_finqa import PROTOCOL
            proof=observed['finqa_reproduction']
            root=out/'targets-finqa';root.mkdir(exist_ok=True);names=[]
            for seed in CR_SEEDS:
                name=f'schedule-{seed}.jsonl';raw=(check/'finqa-generated'/name).read_bytes()
                if (root/name).exists():ready((root/name).read_bytes()==raw,'FinQA fresh schedule changed; owner escalation')
                else:(root/name).write_bytes(raw)
                names.append(name)
            seal(root/'receipt.json',proof);names.append('receipt.json')
            seal(root/'manifest.json',files_manifest(root,names,schedule_protocol=PROTOCOL,seeds=list(CR_SEEDS)))
            continue
        corpus=t.read_jsonl(Path(source['corpus']))
        ready(bool(corpus),'recorded schedule corpus absent; owner escalation')
        # This proves the frozen order algorithm against every realised source
        # schedule; never infer target bytes/order from old output schedules.
        for seed,path in source['v6a_schedules'].items():
            ready(cycle(corpus,int(seed),1280)==t.read_jsonl(Path(path)), 'v6a source/algorithm mismatch; owner escalation')
        root=out/('targets-'+stage);root.mkdir(exist_ok=True);names=[]
        for seed in CR_SEEDS:
            name=f'schedule-{seed}.jsonl';expected=cycle(corpus,seed,1280)
            if (root/name).exists():need(t.read_jsonl(root/name)==expected,'fresh task schedule changed')
            else:t.write_jsonl(root/name,expected)
            names.append(name)
        seal(root/'manifest.json',files_manifest(root,names,source_sha256=sha(source['corpus']),seeds=list(CR_SEEDS)))
    m3_train.schedules_job(args(cfg,{},out,'schedules'))
    seal(out/'pre-data-freeze.json',dict(binding=sha(out/'confirmation-binding.json'),code=bound['code'],
        schedules={str(p.relative_to(out)):sha(p) for p in out.glob('schedules-coverage/*/manifest.json')},seeds=list(CR_SEEDS)))


def operation(cfg,row,attempt,out):
    """Allocation child; invoked by nm_execute through the normal watcher."""
    from kit import m3_train,m3_score,ref_score_job
    from kit.cr_confirmation import analyse
    verify_standin(cfg)
    need(os.environ.get('KIT_M3_MODE')=='cr-confirmation','confirmation mode required')
    phase=row['phase'];published=[]
    if cfg.get('technical_synthetic') and phase=='score':
        os.environ['KIT_V6_AUTHORITY_SHA']=cfg['infrastructure_update']
        # The outer watcher's fixture reports eight GPUs; lane_pool asks for
        # one allocated UUID. This adapter is confined to the CR CPU child.
        simulator=attempt/'sim/nvidia-smi';simulator.parent.mkdir()
        simulator.write_text('#!'+sys.executable+'\nimport sys,runpy\n'
            "if '--query-gpu=uuid' in sys.argv:print('GPU-CPU-0')\n"
            'else:runpy.run_path('+repr(str(KIT/'jobs/nm/sim/nvidia-smi'))+",run_name='__main__')\n")
        simulator.chmod(0o755)
        os.environ['PATH']=str(simulator.parent)+os.pathsep+os.environ.get('PATH','')
    if phase=='bind':
        binding=bind_inputs(cfg,attempt)
        if cfg.get('technical_synthetic'):
            atomic(attempt/'diff-review.json',dict(status='PENDING',technical_synthetic=True,
                reason='CPU stand-in does not establish baseline review or scientific qualification'))
        else:verify_diff(cfg.get('diff_receipt'))
        warnings=[]
        verify_operator(cfg,warnings)
        atomic(attempt/'operator-evidence.json',dict(utilisation_warnings=warnings))
        ready(cfg.get('hosts'),'explicit confirmation hosts missing')
        native=load(binding['realised']['native_inventory'])
        from kit.m3_manifest import host_name
        allowed={host_name(n['node'] if isinstance(n,dict) else n) for n in native['nodes']}
        excluded={host_name(n) for n in native.get('message2_nodes',[])}
        need({host_name(n) for n in cfg['hosts']}<=allowed and not {host_name(n) for n in cfg['hosts']}&excluded,'confirmation host violates inherited core-node exclusion')
        seal(out/'confirmation-binding.json',binding)
        from kit.nm_common import source_seal,launch_config
        seal(out/'launch/cr.json',dict(config_sha256=digest(launch_config(cfg,'cr')),code=source_seal(),cr_code=binding['code'],hosts=cfg.get('hosts',[])))
    elif phase=='prepare':prepare(cfg,out)
    elif phase in ('train-prepare','train','export','qualify-prepare','qualify'):
        action={'train-prepare':'prepare','train':'train','export':'export','qualify-prepare':'preflight-inputs','qualify':'preflight'}[phase]
        a=args(cfg,row,out,action)
        if cfg.get('technical_synthetic') and phase in ('qualify-prepare','qualify'):
            atomic(attempt/'qualification.json',dict(status='TECHNICAL-ONLY',technical_synthetic=True,
                reason='CPU fixture exercises scheduling only; eight-GPU preflight remains PENDING'))
        else:
            m3_train.job(a)
            if cfg.get('technical_synthetic') and phase=='export':
                for step in (10,20,30,40):
                    atomic(m3_train.run_root(out,row['arm'],row['seed'],row['stage'])/f'hf-step{step}/config.json',dict(technical_synthetic=True))
    elif phase=='qualify-read':
        from kit.m3_contract import load_preflight,PREFLIGHT_TEMPLATE
        if not cfg.get('technical_synthetic'):
            ready(load_preflight(out/PREFLIGHT_TEMPLATE.format(arm=row['arm']),row['arm']), 'current-code eight-GPU preflight PENDING')
    elif phase=='assign':
        doc=load(out/'confirmation-binding.json')['realised']
        m3_score.assign_job(NS(out=out,inventory=Path(doc['native_inventory']),standin=cfg.get('technical_synthetic',False)))
        native=load(out/'m3-scoring-assignment.json')['assignments']
        need(set(native)=={f'm3|{s}' for s in CR_SEEDS},'foreign native assignment seed')
        ref_score_job.assign(NS(out=out,family='m3',from_assignment=None,lane_assignment=Path(cfg['reference_assignment']) if cfg.get('reference_assignment') else None,schedule_checkpoints=False))
        reference=load(out/'ref-scoring-assignment.json')['assignments']
        for seed in CR_SEEDS:
            lanes=[reference[f'm3.{p}|{seed}'] for p in ('hellaswag','knowledge-ll')]
            need(lanes[0]==lanes[1],'UNVERIFIED-HARDWARE: reference panels split across assignments')
        atomic(out/'reference-checkpoints.local.json',[c for seed in CR_SEEDS for c in checkpoints(seed)])
    elif phase=='score-prepare':
        seed=row['seed']
        if row['lane']=='native':
            commands=native_commands(cfg,out,seed)
            units=list(dict.fromkeys((c[c.index('--arm')+1],seed,c[c.index('--stage')+1],int(c[c.index('--step')+1])) for c in commands))
            m3_score.prepare_scoring(NS(out=out,standin=cfg.get('technical_synthetic',False),checkpoints_json=json.dumps(units),
                ready=out/'scoring-ready'/f'M3-rescore-{seed}.local.json',workspace=None))
        else:
            for panel in ('hellaswag','knowledge-ll'):ref_score_job.prepare(reference_args(cfg,out,seed,panel))
    elif phase=='score':
        from kit.lane_pool import execute
        from kit.lane_evidence import complete_attempt
        seed=row['seed']
        if row['lane']=='native':
            workspace=m3_score.prepare_workspace(out,out/'scoring-jobs'/f'M3-rescore-{seed}'/os.environ['KIT_LANE_JOB'])
            source=out/'scoring-ready'/f'M3-rescore-{seed}.local.json';ready(source.is_file(),'CPU scoring preparation absent')
            snapshot=workspace/'scoring-ready.local.json';snapshot.write_bytes(source.read_bytes())
            commands=m3_score.prepared_commands(native_commands(cfg,out,seed),snapshot)
            for command in commands:command[command.index('--out')+1]=str(workspace)
            env=dict(os.environ,M3_SCORE_READY=str(snapshot),KIT_V6A='1')
            fds=tuple(json.loads(env.get('M3_GPU_LOCK_FDS','[]')))
            code=execute(commands,workspace/'pool',env=env,pass_fds=fds,concurrency=1)
            ready(code not in (3,75),'native lane incomplete');need(code==0,'native lane failed')
            identities=[]
            for c in commands:
                get=lambda flag:c[c.index(flag)+1]
                unit=m3_score.unit_root(workspace,get('--arm'),seed,get('--stage'),int(get('--step')),get('--panel-set'))
                identities.append(str((unit/'m3-identity.json').relative_to(workspace)))
            complete_attempt(workspace,os.environ['KIT_LANE_JOB'],identities,extra_files=['scoring-ready.local.json'])
        else:ref_score_job.run_lane(reference_args(cfg,out,seed,'both'))
    elif phase=='account':
        # No accounting discount and no fabricated allocated-hour estimate.
        receipts=[load(p) for p in (out/'attempts').glob('NM-cr-*/*/receipt.json')]
        from kit.nm_budget import reconcile
        accounting=reconcile(out,family='cr')
        atomic(attempt/'accounting.json',dict(accounting,attempts=len(receipts),process_gpu_hours=sum(r.get('process_gpu_hours',0) for r in receipts),
            stages=[{k:r.get(k) for k in ('task','phase','arm','seed','stage','lane','gpus','nonce','process_gpu_hours','allocation_gpu_hours','started','ended','status')} for r in receipts],
            requested_reconciliation='per arm/seed/stage; train/prep/qualification/native/reference/export/idle/failed attempts; v6a 298.3 coverage',
            critical_path_hours=[15.24,27.47],wall_time_risk_gt24=True,utilisation='monitoring only: actual window/startup grace and holder-slice evidence'))
    elif phase=='read':
        result=read_result(cfg,out);atomic(attempt/'reading.json',result)
    elif phase=='collect':
        from kit.nm_jobs import current
        reading_record=current(out,'NM-cr-read',complete=False)
        path=out/'attempts/NM-cr-read'/reading_record['nonce']/'reading.json'
        result=load(path) if path.is_file() else dict(status='PENDING/INCOMPLETE',reason='reader absent')
        from kit.nm_pipeline import dispatch_text
        account_record=current(out,'NM-cr-account',complete=False)
        account_path=out/'attempts/NM-cr-account'/account_record['nonce']/'accounting.json'
        accounting=load(account_path) if account_path.is_file() else dict(status='PENDING')
        Path(cfg['dispatch_out']).mkdir(parents=True,exist_ok=True)
        paths=dispatch_text(Path(cfg['dispatch_out']),attempt,dict(nm_family='cr',reading=result,
            infrastructure_authority=cfg['infrastructure_update'],accounting=accounting))
        atomic(attempt/'parts.json',[str(p) for p in paths])
    published.extend(str((attempt/name).relative_to(out)) for name in ('input-binding.json','finqa-reproduction.json','operator-evidence.json','diff-review.json','qualification.json','reading.json','accounting.json','parts.json') if (attempt/name).is_file())
    return dict(status='COMPLETE',published=published,technical_synthetic=bool(cfg.get('technical_synthetic')))


def reference_args(cfg,out,seed,panel):
    return NS(out=out,family='m3',seed=seed,panel=panel,checkpoints=out/'reference-checkpoints.local.json',
        gpu='0',standin=cfg.get('technical_synthetic',False))


def finqa_proof(out):
    """Prefer any failing proof; a newer source recheck cannot clear it."""
    paths=[out/'confirmation-binding.json',out/'finqa-task-check.json',
        *out.glob('attempts/NM-cr-bind/*/input-binding.json'),
        *out.glob('input-checks/*/input-binding.json'),
        *out.glob('read-input-checks/*/input-binding.json')]
    paths=[p for p in paths if p.is_file()]
    if not paths:return {}
    records=[]
    for path in paths:
        doc=load(path)
        proof=doc if path.name=='finqa-task-check.json' else doc.get('finqa_reproduction',{})
        records.append((path.stat().st_mtime_ns,proof))
    failures=[r for r in records if r[1].get('status')!='COMPLETE' or r[1].get('errors')]
    return max(failures or records,key=lambda r:r[0])[1]


def read_result(cfg,out):
    from kit import m3_read,m3_train
    from kit.cr_confirmation import analyse
    verify_standin(cfg)
    warnings=[]
    if (out/'targets-finqa/manifest.json').is_file():
        from kit.cr_confirmation import task_binding
        try:task_binding(out,'finqa',out/'targets-finqa')
        except Pending:pass  # Its failure proof is retained and printed below.
    proof=finqa_proof(out)
    proof_fields=dict(finqa_reproduction=proof)
    operator=verify_operator(cfg,warnings)
    # Scan score evidence before the inherited enumerator can ignore an old
    # seed. Include partial attempts; they cannot silently contaminate the tree.
    for parent in ('scores','scoring-jobs','reference-scores','reference-attempts'):
        for identity in (out/parent).rglob('*identity.json'):
            meta=load(identity)
            if meta.get('seed') not in (None,*CR_SEEDS) or meta.get('arm','base') not in ('base',*ARMS):
                return dict(status='UNVERIFIED',reason='old/foreign seed cell: '+str(identity),utilisation_warnings=warnings,**proof_fields)
    import csv
    for parent in ('scores','scoring-jobs','reference-scores','reference-attempts'):
        for scores in (out/parent).rglob('scores.csv'):
            with scores.open(newline='') as stream:
                if any(r.get('seed') in ('101','102','103') for r in csv.DictReader(stream)):
                    return dict(status='UNVERIFIED',reason='exploratory seed score cell: '+str(scores),utilisation_warnings=warnings,**proof_fields)
    if not (out/'m3-scoring-assignment.json').is_file():
        return dict(status='PENDING/INCOMPLETE',reason='native assignment absent',utilisation_warnings=warnings,**proof_fields)
    cells,assignment=m3_read.load_cells(out,bool(cfg.get('technical_synthetic')))
    receipts={}
    for arm in ARMS:
        for seed in CR_SEEDS:
            for stage in STAGES:
                try:receipts[arm,seed,stage]=m3_train.validate_stage_receipt(out,arm,seed,stage)['status']
                except FileNotFoundError:receipts[arm,seed,stage]='INCOMPLETE'
                except __import__('kit.p3b_common',fromlist=['NotReady']).NotReady:receipts[arm,seed,stage]='INCOMPLETE'
                except ValueError:receipts[arm,seed,stage]='FAIL'
    result=analyse(cells,assignment,receipts=None if cfg.get('technical_synthetic') else receipts)
    result['technical_synthetic']=bool(cfg.get('technical_synthetic'))
    result.update(proof_fields)
    result['utilisation_warnings']=warnings
    for seed in CR_SEEDS:
        for lane in ('native','reference'):
            jobs={c.get('allocation_job') for k,c in cells.items() if k[1]==seed and (c.get('method')=='REFERENCE-SCORED')==(lane=='reference')}
            if len(jobs)!=1 or None in jobs:continue  # Strict identity check below.
            selected=out/'attempts'/f'NM-cr-{lane}-s{seed}'/next(iter(jobs))
            path=selected/'util.summary.json'
            try:
                telemetry=load(path)
                if not isinstance(telemetry,dict):raise ValueError('invalid summary')
                lane_warnings=qualify_utilisation(telemetry,operator)
            except (OSError,ValueError):
                lane_warnings=['UTILISATION_MISSING_OR_UNREADABLE']
            warnings.extend(f'{seed}-{lane}-{warning}' for warning in lane_warnings)
    if result['status'] not in ('UNVERIFIED','UNVERIFIED-HARDWARE'):
        try:
            ready((out/'confirmation-binding.json').is_file(),'realised input binding absent; owner escalation')
            bound=load(out/'confirmation-binding.json')
            observed=bind_inputs(cfg,out/'read-input-checks'/__import__('uuid').uuid4().hex)
            ready(observed==bound,'realised inputs changed; owner escalation')
            ready(proof.get('status')=='COMPLETE' and not proof.get('errors'),FINQA_PENDING)
            ready(bound.get('code')==__import__('kit.cr_confirmation',fromlist=['code_seal']).code_seal(),'code changed after pre-data freeze; owner escalation')
            for seed in CR_SEEDS:
                for lane in ('native','reference'):
                    jobs={c.get('allocation_job') for k,c in cells.items() if k[1]==seed and (c.get('method')=='REFERENCE-SCORED')==(lane=='reference')}
                    ready(len(jobs)==1 and None not in jobs,'complete '+lane+' allocation missing')
        except Pending as error:
            result['finqa_reproduction']=finqa_proof(out)
            result.update(status='PENDING/INCOMPLETE',qualification_pending=str(error))
            result['contrasts']['CR-G']['label']='PENDING';result['decisions']['CR']='PENDING'
    result['lane_hashes']={str(seed)+'.'+lane:digest(sorted((str(k),c.get('uuid'),c.get('environment'),c.get('allocation_job')) for k,c in cells.items() if k[1]==seed and (c.get('method')=='REFERENCE-SCORED')==(lane=='reference'))) for seed in CR_SEEDS for lane in ('native','reference')}
    return result


def verify_operator(cfg,warnings=None):
    """Inspect optional operator evidence; qualification is monitoring only."""
    warnings=[] if warnings is None else warnings
    ref=cfg.get('operator_receipt')
    if not ref:
        warnings.append('OPERATOR_RECEIPT_ABSENT')
        return {}
    try:
        path=Path(ref['path'])
        if sha(path)!=ref.get('sha256'):warnings.append('OPERATOR_RECEIPT_HASH_MISMATCH')
        doc=load(path)
        if not isinstance(doc,dict):raise ValueError('invalid receipt')
        required={'3a','3b','3d','3g','3h','likelihood-K2','holder-slice'}
        changes=doc.get('qualified_changes',[])
        if doc.get('status')!='PASS' or not isinstance(changes,list) or not required<=set(changes):
            warnings.append('OPERATOR_CHANGES_UNQUALIFIED')
        if doc.get('scorer_numerics_unchanged') is not True or doc.get('likelihood_concurrency') not in (1,2):
            warnings.append('OPERATOR_SCORER_QUALIFICATION_ABSENT')
        if not operating_number(doc.get('rolling_window_seconds'),positive=True) or not operating_number(doc.get('startup_grace_seconds')):
            warnings.append('OPERATOR_WINDOW_GRACE_INVALID')
        files=('inference_lifecycle.py','lane_pool.py','util_watch.py','eval_bed.py')
        if doc.get('code')!={name:sha(KIT/name) for name in files}:
            warnings.append('OPERATOR_CODE_DIFFERS')
        return doc
    except (OSError,ValueError,TypeError,KeyError):
        warnings.append('OPERATOR_RECEIPT_UNREADABLE')
        return {}


def operating_number(value,*,positive=False):
    import math
    try:return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)
    except OverflowError:return False


def qualify_utilisation(telemetry,operator):
    """Report operating observations without changing the scientific verdict."""
    warnings=[]
    window=telemetry.get('window_s');grace=telemetry.get('startup_grace_seconds')
    if not operating_number(window,positive=True) or not operating_number(grace):
        warnings.append('UTILISATION_WINDOW_GRACE_ABSENT_OR_INVALID')
    if operator and (window!=operator.get('rolling_window_seconds') or grace!=operator.get('startup_grace_seconds')):
        warnings.append('UTILISATION_WINDOW_GRACE_DIFFERS')
    if telemetry.get('telemetry_complete') is not True:warnings.append('TELEMETRY_INCOMPLETE')
    for field in ('mean_util','min_window_mean'):
        value=telemetry.get(field)
        if not operating_number(value) or value<70 or value>100:
            warnings.append('UTILISATION_'+field.upper()+'_UNQUALIFIED')
    if telemetry.get('holder_slice_telemetry_complete') is not True or not telemetry.get('holder_slices'):
        warnings.append('HOLDER_SLICE_TELEMETRY_UNQUALIFIED')
    return warnings
