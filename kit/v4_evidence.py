"""Archive-only provenance and auxiliary evidence for plan-v4 readers."""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

from kit import v4_teacher as t, v4_analysis as a
from kit.v4_contract import INITIAL_8B_REVISION, TOKENIZER_FILES


def require(value,message):
    if not value:raise ValueError(message)


def weights(doc):
    result={name:digest for name,digest in doc.get('model_file_hashes',{}).items() if name.endswith(('.safetensors','.bin'))}
    require(result and all(re.fullmatch(r'[0-9a-f]{64}',digest) for digest in result.values()),'missing/invalid weight-shard hashes')
    return result


def environment(archive,phase):
    """Require named immutable code/model revisions and measured local shard identities."""
    from kit.v4_datasets import SDPO_COMMIT
    env=archive.json('v4/report-'+phase+'/environment.json')
    identities=archive.json('v4/report-inputs/model-identities.json')
    require(env.get('kit_tag') and env.get('trainer_commit')==SDPO_COMMIT,'environment provenance missing or changed')
    if phase=='phase0':
        identity=env.get('inference_package_identity_sha256')
        require(re.fullmatch(r'[0-9a-f]{64}',identity or '') and identity==(env.get('inference') or {}).get('package_sha256'),'inference installed-package identity missing or changed')
    else:
        require(re.fullmatch(r'[0-9a-f]{40}',env.get('inference_commit') or ''),'environment provenance missing or changed')
    for name,doc in identities.items():
        require(name in ('teacher','initial') and re.fullmatch(r'[0-9a-f]{40}',doc.get('model_revision') or ''),'unknown model revision: '+name)
        weights(doc)
    require(set(identities)=={'teacher','initial'},'missing model identity')
    require(identities['initial']['model_revision']==INITIAL_8B_REVISION,'initial 8B revision differs')
    require(identities['teacher'].get('model')=='Qwen/Qwen3.6-27B' and (identities['teacher']['model_config'].get('text_config') or identities['teacher']['model_config']).get('vocab_size')==248320,'not the registered 27B architecture')
    if phase=='main':
        require(t.REGISTERED_TEACHER_REVISION and t.REGISTERED_TEACHER_WEIGHT_HASHES and t.INITIAL_8B_WEIGHT_HASHES,'C-B2 model registration missing')
        require(identities['teacher']['model_revision']==t.REGISTERED_TEACHER_REVISION and weights(identities['teacher'])==t.REGISTERED_TEACHER_WEIGHT_HASHES,'27B registered revision/shards differ')
        require(weights(identities['initial'])==t.INITIAL_8B_WEIGHT_HASHES,'initial 8B registered shards differ')
    return {'environment':env,'models':identities}


def containment(archive):
    """Require sealed, matching safety settings and both probes for every allocation."""
    from kit.p4_frozen import seal
    from kit.p4_contain import SAFETY_FIELDS
    from kit.v4_budget import allocation_spend
    ledger=archive.json('k8b4/containment/allocation-ledger.json')
    actual,blocks=allocation_spend(ledger)
    first=archive.json('k8b4/containment/containment-receipt.json')
    result={}
    for job,allocation in ledger['allocations'].items():
        prefix='k8b4/containment/' if str(first['job_id'])==str(job) else 'k8b4/containment/allocations/'+str(job)+'/'
        receipt=archive.json(prefix+'containment-receipt.json');test=archive.json(prefix+'containment-selftest.json')
        require(receipt==seal(receipt) and test==seal(test),'containment: altered frozen receipt')
        require(receipt.get('ok') is True and test.get('ok') is True and test.get('v4_qualified') is True,'containment: failed/unknown selftest')
        require(receipt.get('simulation') is False and receipt['gpu_count']==allocation['width']==8,'containment: stand-in or wrong allocation width')
        require(str(receipt['job_id'])==str(job) and test['receipt_sha256']==receipt['content_sha256'],'containment: allocation binding differs')
        require(all(receipt.get(k)==first.get(k) for k in SAFETY_FIELDS),'containment: safety settings changed between allocations')
        require(receipt['ProctrackType']=='proctrack/cgroup' and set(receipt['TaskPlugin'].split(','))=={'task/cgroup','task/affinity'} and receipt['JobAcctGatherType']=='jobacct_gather/cgroup','containment: plugins differ')
        overtime=receipt['OverTimeLimit']
        require(receipt['ConstrainDevices'].lower()=='yes' and receipt['KillWait']==40 and overtime.get('global')=='0' and overtime.get('partition') in ('0','NONE') and overtime.get('job') in (None,'0'),'containment: unknown/failing device or timeout setting')
        require(receipt['SignalChildrenProcesses'].lower() in ('yes','no') and receipt['preemption']['effective_mode'] in ('OFF','CANCEL'),'containment: unknown signal/preemption setting')
        require(test['cpu_probe'].get('ok') is True and test['gpu_acquisition'].get('ok') is True and test['termination'].get('verified') is True,'containment: probes not exercised/terminated')
        require(test['elapsed_seconds']<=600,'containment: qualification exceeded ten minutes')
        require(all(command.get('returncode')==0 and hashlib.sha256(command['stdout'].encode()).hexdigest()==command['stdout_sha256'] and hashlib.sha256(command['stderr'].encode()).hexdigest()==command['stderr_sha256'] for command in receipt['commands']),'containment: raw Slurm observation differs')
        reparse_settings(receipt)
        result[str(job)]={'settings':receipt,'selftest':test}
    return {'allocations':result,'actual_gpu_hours':float(actual),'blocks':{k:float(v) for k,v in blocks.items()}}


def reparse_settings(receipt):
    """Run containment's existing paste parser on the archived raw observations."""
    from kit.p4_contain import _cluster_settings
    files={}
    for command in receipt['commands']:
        argv=command['argv']
        if argv[:1]==['read-paste'] and Path(argv[-1]).name in {'version.txt','config.txt','partition.txt','qos.txt','cgroup.conf'}:name=Path(argv[-1]).name
        elif argv[:2]==['scontrol','--version']:name='version.txt'
        elif argv[:3]==['scontrol','show','config']:name='config.txt'
        elif argv[:3]==['scontrol','show','partition']:name='partition.txt'
        elif argv[:1]==['sacctmgr']:name='qos.txt'
        elif argv[:1]==['read-cgroup-conf']:name='cgroup.conf'
        else:continue
        require(name not in files,'containment: duplicate raw settings observation')
        files[name]=command['stdout']
    require(set(files)=={'version.txt','config.txt','partition.txt','qos.txt','cgroup.conf'},'containment: missing raw Slurm setting')
    with tempfile.TemporaryDirectory(prefix='v4-slurm-pastes-') as temp:
        for name,raw in files.items():(Path(temp)/name).write_text(raw)
        recomputed=_cluster_settings(from_file=Path(temp),qos_choice=receipt['preemption']['qos_name'])
    require(recomputed['ok'],'containment: raw settings failed: '+'; '.join(recomputed['problems']))
    for key in ('slurm_version','ProctrackType','TaskPlugin','JobAcctGatherType','ConstrainDevices','SignalChildrenProcesses','KillWait','preemption'):
        require(recomputed[key]==receipt[key],'containment: setting differs from raw observations: '+key)


def pool_items(rows,task):
    result=[]
    for row in rows:
        info=row['extra_info'];index=str(info['index'])
        item={'id':'sciknoweval-train-'+index if task=='chemistry' else index,'task':task,
              'question':info['problem'],'messages':row['prompt'],'description':info['description'],
              'idx':index,'split':'train','gold':row['reward_model']['ground_truth']}
        item['family']=t.family_of(item);result.append(item)
    return result


@contextmanager
def data_context(archive,tokenizer,mcq=None):
    """Rebuild frozen pools/panels from archived, hash-pinned public source bytes."""
    from kit import v4_pool as pool, v4_datasets as datasets, v4_contract as contract
    prefix='v4/report-inputs/sources/'
    expected={'SDPO/data/preprocess.py':datasets.PREPROCESS_SHA256,
              'k8b-chemistry-templates.json':pool.TAXONOMY_SHA256,
              'FinQA/train.json':pool.FINQA_HASHES['train'],'FinQA/test.json':pool.FINQA_HASHES['test']}
    for name,digest in expected.items():
        require(hashlib.sha256(archive.files.get(prefix+name,b'')).hexdigest()==digest,'pinned source hash differs: '+name)
    with tempfile.TemporaryDirectory(prefix='v4-sources-') as temp:
        root=Path(temp)
        names=[*expected,'SDPO/datasets/sciknoweval/chemistry/train.json','SDPO/datasets/sciknoweval/chemistry/test.json']
        for name in names:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            require(prefix+name in archive.files,'missing pinned source: '+name)
            path.write_bytes(archive.files[prefix+name])
        old_tokenizer=contract.pinned_tokenizer
        old_authors=t.authors_rows
        old_mcq=t.chemistry.authors._SCORERS.get('mcq')
        if mcq is not None:t.chemistry.authors._SCORERS['mcq']=mcq
        contract.pinned_tokenizer=lambda:tokenizer
        contract._cached_prompt_length.cache_clear()
        try:
            frozen=pool.build_manifest(root/'SDPO',root/'FinQA',root/'k8b-chemistry-templates.json')
            # Only the exact pinned preprocessing code is ever executed.
            native=t.load_module('v4_archive_preprocess',root/'SDPO/data/preprocess.py')
            chem={};shipped={}
            for split in ('train','test'):
                source=pool.read_json(root/'SDPO/datasets/sciknoweval/chemistry'/(split+'.json'))
                mapper=native.make_map_fn(split)
                chem[split]={};shipped[split]={}
                for index,item in enumerate(source):
                    remaining=copy.deepcopy(item)
                    mapped=mapper(remaining,index)
                    chem[split][str(item['idx'])]=mapped
                    # Dataset.map retains columns not popped by the authors'
                    # mapper (notably system and embedding). Shipping preserves
                    # those bytes; common training uses the mapper's return.
                    shipped[split][str(item['idx'])]={**remaining,**mapped}
            chemistry=[chem['train'][str(i['idx'])] for i in frozen['chemistry']['train']]
            finqa_items={split:{i['id']:i for i in t.finqa.load(root/'FinQA',split)} for split in ('train','test')}
            finance=t.finqa.rows_for_trainer([finqa_items['train'][i['id']] for i in frozen['finqa']['train']],'train')
            panels={'chemistry':[shipped['test'][str(i['idx'])] for i in frozen['chemistry']['test']],
                    'chemistry_probe':[shipped['test'][str(i['idx'])] for i in frozen['chemistry']['probe']],
                    'finqa':t.finqa.rows_for_trainer([finqa_items['test'][i['id']] for i in frozen['finqa']['test']],'test')}
            pools={'chemistry':pool_items(chemistry,'chemistry'),'finqa':pool_items(finance,'finqa')}
            native_rows={'chemistry':chemistry,'finqa':finance}
            t.authors_rows=lambda task,*args,**kwargs:copy.deepcopy(native_rows[task])
            for task,items in pools.items():
                require(archive.jsonl('v4/report-inputs/'+task+'-pool.jsonl')==items,'original pool differs from pinned source: '+task)
            for task,items in panels.items():
                require(archive.jsonl('v4/report-inputs/'+task+'-panel.jsonl')==items,'held-out panel differs from pinned source: '+task)
            archive.expected_pools=pools;archive.expected_panels=panels
            archive.expected_training_rows=native_rows
            yield {'pools':pools,'panels':panels,'freeze':frozen}
        finally:
            contract.pinned_tokenizer=old_tokenizer
            t.authors_rows=old_authors
            if old_mcq is None:t.chemistry.authors._SCORERS.pop('mcq',None)
            else:t.chemistry.authors._SCORERS['mcq']=old_mcq
            contract._cached_prompt_length.cache_clear()


def peak_memory(archive,ident):
    """Recompute each of eight GPUs' allocator peak and read actual optimizer groups."""
    rows=archive.jsonl('runs/'+ident+'/env/gpu-memory.jsonl');by_rank=defaultdict(list)
    for row in rows:by_rank[row['rank']].append(row)
    require(set(by_rank)==set(range(8)),'missing peak memory per GPU: '+ident)
    result={}
    for rank,records in sorted(by_rank.items()):
        uuids={r['uuid'] for r in records}
        require(len(uuids)==1 and 'unknown' not in uuids,'unknown/changed physical training GPU')
        groups=[g for r in records for g in r['optimizer_parameter_groups']]
        require(groups and any(g['weight_decay']==.01 for g in groups) and all(g['weight_decay'] in (0.,.01) for g in groups),'optimizer parameter-group weight decay differs')
        require(all(type(r['peak_allocated_bytes']) is int and type(r['peak_reserved_bytes']) is int and 0<r['peak_allocated_bytes']<=r['peak_reserved_bytes'] for r in records),'unknown/invalid GPU peak memory')
        result[str(rank)]={'uuid':next(iter(uuids)),'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in records),
                           'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in records),'optimizer_parameter_groups':groups}
    require(len({r['uuid'] for r in result.values()})==8,'duplicate physical training GPU')
    return result


def common_data(archive,phase,tokenizer,tasks=("chemistry","finqa")):
    """Rebuild both verified targets on original prompts and compare actual parquet bytes/rows."""
    import pyarrow.parquet as pq
    base='v4/report-'+phase+'/'
    pools=[];demos=[];rewrites=[]
    for task in tasks:
        pools+=archive.jsonl(base+task+'-pool.jsonl')
        demos+=archive.jsonl(base+'teacher/'+task+'/demonstrations.jsonl')
        rewrites+=archive.jsonl(base+'rewrite/'+task+'/rewrites.jsonl')
    expected,proof=t.common_training_rows(pools,demos,rewrites,tokenizer)
    manifest=archive.json(base+'common/manifest.json')
    require(manifest['prompt_identity']==proof,'common prompt identity proof differs')
    require(manifest['targets']=={'F':'extra_info.demonstration','D':'extra_info.demonstration','R':'extra_info.rewrite','S':None},'offline corpus roles differ')
    require(len(manifest['sources'])==2*len(tasks),'missing/extra corpus source')
    used=set()
    for directory,hashes in manifest['sources'].items():
        path=Path(directory);task=path.name;role=path.parent.name
        require(task in ('chemistry','finqa') and role in ('teacher','rewrite') and (role,task) not in used,'corpus source identity differs')
        used.add((role,task))
        filename='demonstrations.jsonl' if role=='teacher' else 'rewrites.jsonl'
        require(set(hashes)=={filename,'pool.jsonl','manifest.json'},'missing corpus source hash')
        for name,digest in hashes.items():
            require(t.sha(archive.files[base+role+'/'+task+'/'+name])==digest,'shared corpus hash differs: '+role+':'+task)
    for task,rows in expected.items():
        raw=archive.files[base+'common/'+task+'.parquet']
        require(t.sha(raw)==manifest['files'][task]['sha256'],'common-training-set byte hash differs')
        actual=pq.read_table(io.BytesIO(raw)).to_pylist()
        require(actual==rows and len(actual)==manifest['files'][task]['rows'],'common rows/targets differ from reverified intersection')
    archive.common_rows=expected
    return {'manifest_sha256':t.sha(archive.files[base+'common/manifest.json']),
            'tasks':{task:{'rows':len(rows),'sha256':manifest['files'][task]['sha256']} for task,rows in expected.items()}}


def training_provenance(archive,phase,job,summary,tokenizer):
    """Bind seed, common corpus, ordered exposures and incoming shard lineage to a slot."""
    import random
    import pyarrow.parquet as pq
    from kit.v4_sft import schedule_common_rows
    base='v4/report-'+phase+'/'
    task=job['task'];seed=job['seed'];ident=job['id']
    require(summary['arm']==job['arm'] and summary['seed']==seed,'slot arm/seed differs')
    schedule_path=base+f'{task}-s{seed}-schedule.json'
    schedule=archive.json(schedule_path)
    count=64 if phase in ('qualification','phase0') else 1280
    ids=[str(row['extra_info']['index']) for row in archive.common_rows[task]]
    random.Random(seed).shuffle(ids)
    expected_ids=ids[:count] if len(ids)>=count else [ids[i%len(ids)] for i in range(count)] if ids else []
    require(schedule['seed']==seed and schedule['ids']==expected_ids,'registered seeded exposure schedule differs')
    scheduled=base+f'scheduled/{task}-s{seed}/'
    manifest=archive.json(scheduled+'manifest.json')
    raw=archive.files[scheduled+'train.parquet']
    require(t.sha(raw)==manifest['train_sha256']==summary['train_sha256'],'training dataset byte hash differs')
    require(t.sha(archive.files[base+'common/manifest.json'])==manifest['common_training_manifest_sha256']==summary['common_training_manifest_sha256'],'common corpus reused across seeds/orders differs')
    require(t.sha(archive.files[schedule_path])==manifest['schedule_sha256']==summary['schedule_sha256'],'training schedule hash differs')
    cache=getattr(archive,'_verified_schedules',None)
    if cache is None:cache=set();archive._verified_schedules=cache
    key=(task,manifest['common_training_manifest_sha256'],manifest['schedule_sha256'],t.sha(raw))
    if key not in cache:
        expected=schedule_common_rows(archive.common_rows[task],schedule,tokenizer)
        require(pq.read_table(io.BytesIO(raw)).to_pylist()==expected,'ordered training rows/targets differ')
        cache.add(key)
    require(manifest['rows']==count and list(map(str,manifest['ids']))==expected_ids,'scheduled exposure count differs')
    root='runs/'+ident+'/env/'
    require(archive.json(root+'data-manifest.json')==manifest,'launcher read different dataset manifest')
    identities=archive.json('v4/report-inputs/model-identities.json')
    incoming=archive.json(root+'incoming-checkpoint.json')
    expected_incoming=(archive.json('runs/'+job['incoming']+'/env/export-identity-step40.json') if job['incoming'] else identities['initial'])
    require(weights(incoming)==weights(expected_incoming),'incoming checkpoint shard lineage differs')
    final_step=2 if phase in ('qualification','phase0') else 40
    outgoing=archive.json(root+f'export-identity-step{final_step}.json')
    weights(outgoing)
    if phase=='main':weights(archive.json(root+'export-identity-step20.json'))
    return {'dataset_sha256':t.sha(raw),'common_training_manifest_sha256':manifest['common_training_manifest_sha256'],
            'schedule_sha256':manifest['schedule_sha256'],'incoming_weight_shards':weights(incoming),
            'outgoing_weight_shards':weights(outgoing),'seed':seed,'steps':final_step}


def scheduler_rows(archive,phase):
    """Reuse containment's archive replay checks, binding each successful GPU row."""
    from kit.p4_contain import scheduler_problems
    from kit.v4_campaign import build
    prefix=f'v4/containment/v4-{phase}/'
    result={}
    with tempfile.TemporaryDirectory(prefix='v4-contained-evidence-') as temp:
        root=Path(temp)
        for name,raw in archive.files.items():
            if name.startswith('k8b4/containment/') and name.endswith('.json'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        for row in build(phase)['rows']:
            if '{kit}/p4_contain.py' not in row['command'] or 'row' not in row['command']:continue
            names=[name for name in archive.files if name.startswith(prefix+row['id']+'/') and name.endswith('/containment-row.json')]
            if not names:continue
            names.sort(key=lambda name:int(name.split('/')[-2].split('-')[1]))
            valid=[]
            for name in names:
                record=archive.json(name)
                problems=[] if phase=='phase0' else scheduler_problems(root,record)
                if record.get('ok')==1 and not problems:valid.append((name,record))
            if valid:result[row['id']]=valid[0][1]
    archive.scheduler_rows={**getattr(archive,'scheduler_rows',{}),**{(phase,key):value for key,value in result.items()}}
    return result


def scoring_gpu(archive,phase,checkpoint,task,cap,fingerprint):
    """Compare the scorer's mode with its independently archived owning-step GPU."""
    row=(f'baseline-{task}-{cap}' if checkpoint=='baseline' else f'score-{checkpoint}-{task}-{cap}')
    record=archive.scheduler_rows.get((phase,row))
    if phase=='phase0':return (record or {}).get('slurm',{}).get('gpu_uuids',['informational'])[0]
    require(record is not None,'missing valid scoring containment: '+row)
    slurm=record['slurm'];uuids=slurm['gpu_uuids']
    require(record['gpus']==slurm['gpu_count']==1 and len(uuids)==1,'scoring used multiple physical GPUs')
    require(uuids==slurm['expected_gpu_uuids'],'physical scoring assignment differs')
    require(fingerprint.get('cuda_visible_devices') and ',' not in fingerprint['cuda_visible_devices'],'unknown/multiple scoring CUDA selection')
    require(any(uuids[0] in line for line in fingerprint['gpus']),'scoring fingerprint lacks owning physical UUID')
    return uuids[0]
