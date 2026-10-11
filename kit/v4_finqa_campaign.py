"""Registered 15.12 FinQA standalone graph; existing contained runner operations."""
from kit.v4_campaign import slots, ALL_ARMS, SEEDS, PANELS, PROBE_STEPS
import math
import json
import hashlib
from pathlib import Path
from kit.p4_contain import MIN_ROW_CAP
from kit.v4_phase0_timing import scale, TIMING_MULTIPLIER
from kit.v4_retention import RETIREMENT_IO_SECONDS

# Registered unmeasured defaults: 300 seconds for a two-step eight-GPU smoke;
# 450 seconds per three-step restoration leg, including state save/hash/reload.
# Restoration launches uninterrupted/killed/resumed legs for EVERY arm plus IO,
# kill verification and comparisons (five kills: wait 10 + ray stop 60 +
# idle check 60 + query slack 10 seconds each, plus 100 seconds controller IO).
# No target-node rate is claimed here.
QUALIFICATION_LAUNCH_SECONDS=300
RESTORATION_LEG_SECONDS=450
RESTORATION_LEGS=('uninterrupted','killed','resumed')
RESTORATION_CONTROLLER_SECONDS=800
RESTORATION_CAP_SECONDS=len(ALL_ARMS)*len(RESTORATION_LEGS)*RESTORATION_LEG_SECONDS+RESTORATION_CONTROLLER_SECONDS
# Conservative eager, batch-invariant H=8192 planning rate: 2 seconds/question,
# 1.5 margin. Retain the reviewed 3600-second cap until real measurements justify
# a reduction. C1 registers a separate 1800-second B bound: its response budget
# is one quarter of H, while the cap retains startup/reload headroom.
BASELINE_SECONDS_PER_QUESTION=2
BASELINE_MARGIN=1.5
BASELINE_REVIEWED_MINIMUM_SECONDS=3600
BASELINE_B_CAP_SECONDS=1800
def baseline_cap(panel,response_cap=8192):
    n={'finqa':1147,'general-v1':300}[panel]
    if response_cap==2048:return BASELINE_B_CAP_SECONDS
    if response_cap!=8192:raise ValueError('unregistered baseline response cap')
    return max(BASELINE_REVIEWED_MINIMUM_SECONDS,math.ceil(n*BASELINE_SECONDS_PER_QUESTION*BASELINE_MARGIN))


def phase0_measured_launch_allowances(archive_path=None):
    """C1 substitution hook: verified phase-0 whole two-step launch time x 1.5.

    No archive means the registered unmeasured defaults. A supplied archive must
    pass the real phase-0 reader; use the slowest S/F/R/D whole launch (startup,
    two updates, checkpoint/export), not a per-step time or synthetic summary.
    Freeze returned values/source into the campaign before its immutable tag.
    """
    if archive_path is None:
        return dict(qualification_launch_seconds=QUALIFICATION_LAUNCH_SECONDS,restoration_leg_seconds=RESTORATION_LEG_SECONDS,
                    source=dict(kind='registered_worst_case',measured=False))
    from kit.v4_phase0_carry import verified_source, public_binding
    from kit.v4_archive import Archive
    from kit.v4_timing import sample
    from kit.v4_budget import number
    archive_path=Path(archive_path)
    digest=hashlib.sha256(archive_path.read_bytes()).hexdigest()
    binding,report=verified_source(archive_path)
    if report['status']!='technical pass':raise ValueError('phase0 must pass before measured launch substitution')
    receipts=[]
    with Archive(archive_path) as archive:
        ledger=archive.json('k8b4/containment/allocation-ledger.json')
        models=archive.json('v4/report-inputs/model-identities.json') if 'v4/report-inputs/model-identities.json' in archive.files else None
        for arm in ('S','F','R','D'):
            name=f'v4/report-phase0/timings/train-q-{arm}-finqa.json';record=archive.json(name)
            from kit.v4_finqa_policy import warn
            if report['tables']['training'][arm]['steps']!=2 or record.get('row')!=f'train-q-{arm}-finqa' or record.get('phase')!='phase0' or record.get('operation')!='train':
                warn('C1 measurement','two-step timing labels differ; source bytes preserved, estimate is advisory')
            try:
                observed=sample(ledger,record)
                if observed['allocation_gpus']!=8:warn('C1 measurement','observed GPU width differs; main still requests eight GPUs')
                seconds=number(record.get('wall_seconds'),'phase0 two-step launch',positive=True)
                value=float(seconds);fallback=False
            except (ValueError,KeyError,TypeError) as exc:
                warn('C1 measurement',str(exc)+'; use the generous unmeasured launch floor')
                value=QUALIFICATION_LAUNCH_SECONDS/1.5;fallback=True
            receipts.append(dict(path=name,sha256=hashlib.sha256(archive.files[name]).hexdigest(),wall_seconds=value,
                                 reported_wall_seconds=str(record.get('wall_seconds')),measurement_warning=fallback,allocation_id=record.get('allocation_id')))
    if hashlib.sha256(archive_path.read_bytes()).hexdigest()!=digest:raise ValueError('phase0 launch archive changed during reading')
    slowest=max(number(r['wall_seconds'],'launch seconds') for r in receipts)
    allowance=math.ceil(slowest*number('1.5','launch allowance'))
    # A restoration leg (three steps plus full-state save, hash and reload) is dominated by startup, like the
    # measured two-step launch; the 1.5 margin covers the extra step, and the leg never drops below the
    # registered unmeasured leg allowance.
    restoration_leg=max(RESTORATION_LEG_SECONDS,allowance)
    return dict(qualification_launch_seconds=allowance,restoration_leg_seconds=restoration_leg,
                phase0_accounting=public_binding(binding),model_registration=models,
                source=dict(kind='verified_phase0_archive',measured=True,archive_name=archive_path.name,archive_sha256=digest,
                            raw_receipts=receipts,slowest_two_step_launch_seconds=float(slowest),allowance=1.5))

DEPTH_OMISSION='Omitted under 15.11.1: live 1.7B hook-on/off noninterference cannot be demonstrated offline; no hooks installed.'


def build(phase,*,phase0_launch_archive=None):
    if phase not in ('qualification','main'):raise ValueError('unknown campaign phase')
    source_allowances=phase0_measured_launch_allowances(phase0_launch_archive)
    allowances=source_allowances if phase=='qualification' else None
    launch_seconds=allowances['qualification_launch_seconds'] if allowances else QUALIFICATION_LAUNCH_SECONDS
    restoration_cap=len(ALL_ARMS)*len(RESTORATION_LEGS)*(allowances['restoration_leg_seconds'] if allowances else RESTORATION_LEG_SECONDS)+RESTORATION_CONTROLLER_SECONDS
    rows=[];training=False
    def add(ident,operation=None,*args,gpu=0,cap=900,pilot=False,needs=None,command=None):
        if phase=='qualification':
            cap=min(cap,{'profile-waves':1800,'restoration':restoration_cap,'train':launch_seconds,'score':900,
                         'teacher':14400,'rewrite':14400,'determinism':900,'probe':600,
                         'runner-profile':600}.get(operation,300)) if command is None and not ident.startswith('baseline-') else cap
        # The following schedule (or final report) owns completed-slot retirement.
        # This explicit unmeasured IO cap is included in its outer row budget.
        if operation in ('schedule','report'):cap+=RETIREMENT_IO_SECONDS
        cap=scale(max(MIN_ROW_CAP,cap))
        deps=([rows[-1]['id']] if rows and not training else []) if needs is None else list(needs)
        deps += [r['id'] for r in rows if r.get('pilot')][-1:]
        argv=command or ['python','{kit}/v4_campaign_ops.py',operation,'--work','{work}','--phase',phase,'--row',ident,*args]
        block='scientific'
        if gpu:
            argv=['python','{kit}/p4_contain.py','row','--work','{work}','--out','{work}/v4/containment/{campaign}/{row}/attempt-{attempt}',
                  '--gpus',str(gpu),'--time-cap',str(cap),'--block',block,'--block-limit','1000','--ceiling','1000','--',*argv]
        row=dict(id=ident,needs=list(dict.fromkeys(deps)),command=argv,env={'V4_TELEMETRY':'1','V4_FINQA_CAMPAIGN':'1'},timeout_seconds=cap+scale(120) if gpu else cap)
        if operation=='restoration':row['env']['V4_RESTORATION_LAUNCH_ALLOWANCES']=json.dumps(allowances,sort_keys=True)
        if not gpu:row['allocation_cpu_cap_seconds']=cap
        if training and rows:row.update(wants=[rows[-1]['id']],order=[rows[-1]['id']])
        if pilot:row['observation']=True
        rows.append(row)
    def selftest(ident):
        block='scientific'
        add(ident,command=['python','{kit}/p4_contain.py','selftest','--work','{work}','--out','{work}/k8b4/containment',
                         '--block',block,'--block-limit','1000','--ceiling','1000'],cap=600)
    selftest('containment-selftest')
    add('environment','environment',pilot=True)
    add('prepare','verify-prepare',pilot=True)
    if phase=='main':add('model-identity-check','gate')
    else:
        add('runner-profile','runner-profile',pilot=True)
        for task in PANELS:
            for cap in (2048,8192):add(f'baseline-{task}-{cap}','score','--checkpoint','baseline','--task',task,'--cap',str(cap),gpu=1,cap=baseline_cap(task,cap))
        for repeat in range(1,5):add(f'score-agreement{repeat}-finqa-2048','score','--checkpoint','agreement'+str(repeat),'--task','finqa','--cap','2048',gpu=1,cap=3600)
        add('scoring-agreement','agreement',pilot=True)
        add('teacher-determinism','determinism',gpu=8,cap=1800,pilot=True)
        add('full-wave-profile','profile-waves',gpu=8,cap=3600,pilot=True)
    # Qualification builds the full registered corpus before main. The main stage
    # admits the full-pool generation, reusing those immutable attempts.
    add('teacher-finqa','teacher','--task','finqa',gpu=8,cap=14400)
    
    if phase=='main':selftest('rewrite-allocation-selftest')
    add('rewrite-finqa','rewrite','--task','finqa',gpu=8,cap=14400)
    add('coverage','coverage',pilot=True)
    add('training-set','training-set',pilot=True)
    if phase=='qualification':
        add('resolved-configs','configs',pilot=True)
        add('restoration-check','restoration',gpu=8,cap=restoration_cap,pilot=True)
        add('profile-probe-finqa','probe','--slot','q-S-finqa','--task','finqa','--step','0',gpu=1,cap=1800)
        add('profile-conditioned-finqa','probe','--slot','q-S-finqa','--task','finqa','--step','0','--conditioned',gpu=1,cap=1800)
        jobs=[dict(id=f'q-{arm}-finqa',arm=arm,task='finqa',seed=101,incoming=None) for arm in ALL_ARMS]
    else:
        jobs=slots()
        selftest('scientific-allocation-selftest')
        add('conditioned-probe-finqa','probe','--slot','D-finqa-alone-s101','--task','finqa','--step','0','--conditioned',gpu=1,cap=1800)
    training=True
    for index,job in enumerate(jobs):
        ident=job['id']
        add('schedule-'+ident,'schedule','--slot',ident,needs=['training-set'])
        add('train-'+ident,'train','--slot',ident,gpu=8,cap=launch_seconds if phase=='qualification' else 10800,needs=['schedule-'+ident])
        add('merge-'+ident,'merge','--slot',ident,cap=1200,needs=['train-'+ident])
        panels=PANELS if phase=='main' else ('finqa',)
        caps=(2048,8192) if phase=='main' else (2048,)
        for task in panels:
            for cap in caps:add(f'score-{ident}-{task}-{cap}','score','--checkpoint',ident,'--task',task,'--cap',str(cap),gpu=1,cap=3600,needs=['merge-'+ident])
        if phase=='main':
            for step in PROBE_STEPS:add(f'probe-{ident}-finqa-{step}','probe','--slot',ident,'--task','finqa','--step',str(step),gpu=1,cap=1800,needs=['merge-'+ident])
            if index==0:add('D-standalone-gate','standalone',pilot=True,needs=[f'score-{ident}-finqa-2048'])
    add('report','report',pilot=True)
    add('archive-pack','archive-pack',cap=600)
    add('QUALIFICATION_FINISHED' if phase=='qualification' else 'FINISHED','finished')
    graph=dict(schema='kit-campaign.v1',name='v4-'+phase,workdir_env='WORK',rows=rows,v4=dict(
        phase=phase,timing_multiplier=TIMING_MULTIPLIER,containment_selftest="informational",scope='finqa-alone',parallel_jobs=1,retirement_io_seconds=RETIREMENT_IO_SECONDS,seeds=list(SEEDS),primary_seeds=[101,102,103],replication_seeds=[104,105],
        steps=2 if phase=='qualification' else 40,qualification_limit=1000,planned_qualification_gpu_hours=100,ceiling=1000,self_monitored=True,generation_shards={'teacher':4,'rewrite':8},
        teacher_depth={'status':'omitted','reason':DEPTH_OMISSION,'allowance_gpu_hours':5},probe_questions_per_task=20,
        cpu_prepare={'first':'import-smoke','command':['python','{kit}/v4_prepare.py','--work','{work}','--phase',phase]},
        scoring_gpu='one stable Slurm-assigned physical GPU',slot_order=jobs,
        allocation_stages=({'qualification':dict(first='containment-selftest',last='QUALIFICATION_FINISHED',block='scientific',block_limit=1000)} if phase=='qualification' else {
            'teacher':dict(first='containment-selftest',last='teacher-finqa',block='scientific',block_limit=1000),
            'rewrite':dict(first='rewrite-allocation-selftest',last='training-set',block='scientific',block_limit=1000),
            'scientific':dict(first='scientific-allocation-selftest',last='FINISHED',block='scientific',block_limit=1000)})))
    if source_allowances.get('phase0_accounting'):graph['v4']['phase0_accounting']=source_allowances['phase0_accounting']
    if source_allowances.get('model_registration'):graph['v4']['model_registration']=source_allowances['model_registration']
    if allowances:graph['v4']['launch_allowances']=allowances
    return graph
