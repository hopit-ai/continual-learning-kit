"""Explicit CPU stand-ins and deterministic vectors; never scientific evidence."""
import os
from kit.nm_common import *
from kit.nm_score import scorer_authority


def fixture(family):
    panels={p:[f'{p}-{i:04d}' for i in range(1147 if p=='finqa' else 100)] for p in PANELS}
    if family=='grpo':panels.pop('legal')
    if family=='v2b':
        panels={p:panels[p] for p in (*GENERAL,'hellaswag')}
        for p,n in [('july-off',21),('july-on',21),('august-off',24),('august-on',24),('september-off',13),('september-on',52),('september-on-holdout',13)]:panels[p]=[f'{p}-{i:04d}' for i in range(n)]
    checkpoints={str(s):{model:{str(k):digest([family,s,model,k]) for k in (STEPS if model in ('C','T') else (0,))} for model in ('base','incoming','memory','C','T')} for s in SEEDS}
    spec=dict(panels=panels,checkpoints=checkpoints,scorers={p:scorer_authority(p) for p in panels},pre_cohort={})
    cells=[]
    for seed in SEEDS:
        for panel,ids in panels.items():
            for model in ('base','incoming','memory','C','T'):
                for step in STEPS if model in ('C','T') else (0,):
                    n=len(ids)
                    fraction=.20 if model=='base' else .80 if model=='incoming' else .70 if model=='memory' else .60 if model=='C' else .70
                    if family=='grpo' and panel=='finqa':fraction=.20 if model in ('base','incoming') else .60+step/400 if model in ('C','T') else .2
                    if model in ('C','T') and step==0:fraction=.80 if family!='grpo' else .20
                    vector={i:int(j<int(n*fraction)) for j,i in enumerate(ids)}
                    cells.append(dict(seed=seed,panel=panel,model=model,step=step,items=vector,
                        checkpoint_sha256=checkpoints[str(seed)][model][str(step)],nonce=f'cpu-{family}-{seed}-{panel}',
                        uuid=f'GPU-CPU-{seed}',environment='CPU-STANDIN',scorer_sha256=spec['scorers'][panel]['sha256'],
                        method=spec['scorers'][panel]['method'],precision='full',status='COMPLETE',technical_synthetic=True))
            if panel=='finqa':
                selected={c['model']:dict(c['items']) for c in cells if c['panel']==panel and c['seed']==seed and c['model'] in ('base','incoming')}
                spec['pre_cohort'][str(seed)]=dict(before_training=True,**selected)
    if family=='opr':spec['opr_refresh']={str(s):dict(unique=5,accepted=4,coverage=.8,fallback_fraction=.2,exposure_substitution=.8,exposure_fallback=.2) for s in SEEDS}
    if family=='fisher':spec['fisher_calibration']=dict(lambda_iso=.1,lambda_fisher=.05,ratios={str(s):1. for s in SEEDS},median_ratio_relative_error=0.,
        inputs={str(s):dict(task_grad_norm=1.,incoming_distance=1.,weighted_distance=2.) for s in SEEDS})
    binding=dict(family=family,status='READY',benchmark=False)
    if family=='isdft':binding['persistence']=stats([-4.,-4.,-4.],ONE)
    return dict(cells=cells,technical_synthetic=True),spec,binding


def text_stress_fixture(family):
    """Transport stress case: varying seed/curve counts, maximum disagreements.

    Prime-sized general/legal panels expose full-precision float expansion in
    the former format. Registered SEC and FinQA denominators remain untouched.
    IDs are 64 ASCII characters, including all 66 permitted OPR disagreements.
    This is a capacity fixture, never scientific or production panel evidence.
    """
    doc,spec,binding=fixture(family)
    for panel in (*GENERAL,'hellaswag',*(() if family in ('grpo','v2b') else ('legal',))):
        spec['panels'][panel]=[('stress-'+panel+'-'+str(i)).ljust(64,'x') for i in range(1009)]
    if family=='opr':
        old=spec['panels']['finqa'];new=[i.ljust(64,'x') for i in old]
        spec['panels']['finqa']=new
        for pre in spec['pre_cohort'].values():
            for ref in ('base','incoming'):pre[ref]={j:pre[ref][i] for i,j in zip(old,new)}
    for c in doc['cells']:
        ids=spec['panels'][c['panel']];n=len(ids)
        if family=='opr' and c['panel']=='finqa' and c['model'] in ('base','incoming'):
            c['items']=dict(spec['pre_cohort'][str(c['seed'])][c['model']]);continue
        # Full-digit counts, nonconstant seed effects, nonflat AUCs.
        target=int(n*({'base':.2,'incoming':.8,'memory':.7,'C':.6,'T':.7}[c['model']]))
        target=min(n,max(0,target+(c['seed']-101)*{'base':2,'incoming':-3,'memory':1,'C':5,'T':9}[c['model']]+c['step']//10*(2 if c['model']=='T' else 1)))
        c['items']={i:int(j<target) for j,i in enumerate(ids)}
    if family=='opr':
        for pre in spec['pre_cohort'].values():
            for ref in ('base','incoming'):
                for ident in list(pre[ref])[:11]:pre[ref][ident]^=1
    return doc,spec,binding


def operation(cfg,row,attempt,out):
    """Stand-in replaces only model/data work; wrapper, watcher and DAG are real."""
    family=row['family'];phase=row['phase'];published=[]
    failure=cfg.get('fail_task')==row['id']
    marker=out/'injected'/row['id']
    if failure and not marker.exists():
        marker.parent.mkdir(parents=True,exist_ok=True);marker.write_text('once')
        raise RuntimeError('CPU stand-in interrupted '+phase)
    if cfg.get('pending_task')==row['id']:raise Pending('CPU stand-in prerequisite pending')
    if phase=='bind':
        path=out/'bindings'/(family+'.json');binding=dict(family=family,status='ARCHIVED UNREAD' if family in cfg.get('archive',[]) else 'READY',benchmark=False,technical_synthetic=True,reason='CPU stand-in resolved trigger')
        seal(path,binding);published.append(str(path.relative_to(out)))
        seal(out/'launch.json',dict(technical_synthetic=True,infrastructure_update='0'*64))
        return dict(status=binding['status'],published=published,technical_synthetic=True)
    if family=='opr' and phase=='generate':
        from kit.nm_data import generation_seed
        seed=row['seed'];success=cfg.get('opr_successes',4)
        rows=[dict(id=str(i),prompt='training prompt '+str(i),response='Answer: 7',gold='7',stored_verified=True,target_sha256=raw_hash(b'Answer: 7')) for i in range(5)]
        outputs={r['id']:{str(a):dict(response='Answer: 7' if int(r['id'])<success else 'Answer: 8',output_tokens=4,
            finish_reason='stop',rng=generation_seed(seed,r['id'],a)) for a in range(1,2 if int(r['id'])<success else 5)} for r in rows}
        atomic(attempt/'generation.json',dict(rows=rows,attempts=outputs));published.append(str((attempt/'generation.json').relative_to(out)))
    if family=='opr' and phase=='cohort':
        from kit.nm_data import cohort
        _,spec,_=fixture(family);pre=spec['pre_cohort'][str(row['seed'])]
        if 'opr_cohort_size' in cfg:
            base_vector=pre['base'];n=sum(base_vector.values())+cfg['opr_cohort_size']
            pre['incoming']={i:int(j<n) for j,i in enumerate(pre['incoming'])}
        cohort(pre['base'],pre['incoming'])
        atomic(attempt/'cohort.json',pre);published.append(str((attempt/'cohort.json').relative_to(out)))
    if family=='opr' and phase=='refresh':
        from kit.nm_data import refresh
        from kit.nm_jobs import current
        seed=row['seed'];r=current(out,f'NM-opr-generate-s{seed}')
        generated=load(out/'attempts'/r['task']/r['nonce']/'generation.json')
        refreshed,audit=refresh(generated['rows'],generated['attempts'],seed)
        atomic(attempt/'refresh.json',dict(rows=refreshed,audit=audit));published.append(str((attempt/'refresh.json').relative_to(out)))
    if phase=='score':
        scores,spec,_=(text_stress_fixture if cfg.get('transport_stress') else fixture)(family);cells=[c for c in scores['cells'] if c['seed']==row['seed'] and c['panel']==row['panel']]
        steps=STEPS if row['panel']=='legal' or (family=='grpo' and row['panel']=='finqa') else (20,40)
        cells=[c for c in cells if (c['model'] in ('base','incoming') and c['step']==0) or (c['model']=='memory' and family=='v2b' and row['panel'] not in (*GENERAL,'hellaswag')) or (c['model'] in ('C','T') and c['step'] in steps)]
        need(len(cells)==row['expected_score_cells'],'stand-in scorer command inventory differs')
        from kit.m2_gpu import allocated_devices
        measured=allocated_devices()[os.environ['CUDA_VISIBLE_DEVICES']]
        for cell in cells:cell.update(nonce=attempt.name,uuid=measured)
        if cfg.get('grpo_D_insensitive') and family=='grpo' and row['panel']=='finqa':
            base=next(c['items'] for c in cells if c['model']=='base')
            for c in cells:
                if c['model']=='C' and c['step']==40:c['items']=dict(base)
        if cfg.get('invalid_identity_task')==row['id']:
            next(c for c in cells if c['model']=='T')['uuid']='GPU-CPU-other'
        if cfg.get('invalid_checkpoint_task')==row['id']:
            next(c for c in cells if c['model']=='T')['checkpoint_sha256']='0'*64
        atomic(attempt/'scores.json',dict(cells=cells,technical_synthetic=True));published.append(str((attempt/'scores.json').relative_to(out)))
    if phase=='read':
        from kit.nm_read import read
        from kit.nm_jobs import current,inventory
        _,spec,binding=(text_stress_fixture if cfg.get('transport_stress') else fixture)(family);cells=[]
        for r in inventory()['rows']:
            if r['family']==family and r['phase']=='score':
                receipt=current(out,r['id']);p=out/receipt['artifacts_path'] if 'artifacts_path' in receipt else out/'attempts'/r['id']/receipt['nonce']/'scores.json'
                ready(p.is_file(),'stand-in lane missing');cells.extend(load(p)['cells'])
        result=read(family,dict(cells=cells,technical_synthetic=True),spec,binding)
        ready(not result['status'].startswith('PENDING') or result.get('primary'),'reader incomplete')
        expected='PENDING/INCOMPLETE' if family=='grpo' and cfg.get('grpo_D_insensitive') else 'UNVERIFIED-HARDWARE' if cfg.get('invalid_identity_task') else 'UNVERIFIED' if cfg.get('invalid_checkpoint_task') else 'TECHNICAL-ONLY'
        need(result['status']==expected,'stand-in reader invalid: '+str(result))
        atomic(attempt/'reader-spec.json',spec);atomic(attempt/'scores.json',dict(cells=cells,technical_synthetic=True));atomic(attempt/'binding.json',binding)
        published.extend(str((attempt/name).relative_to(out)) for name in ('reader-spec.json','scores.json','binding.json'))
        atomic(attempt/'reading.json',result);published.append(str((attempt/'reading.json').relative_to(out)))
        (attempt/'summary.txt').write_text('\n'.join(text_lines(family,result,'0'*64))+'\n')
        published.append(str((attempt/'summary.txt').relative_to(out)))
    if phase=='collect':
        from kit.nm_pipeline import collect
        return collect(row,attempt,out,dict(return_archive=cfg.get('return_archive',False)),published)
    return dict(status='COMPLETE',technical_synthetic=True,row=row['id'],phase=phase,published=published)
