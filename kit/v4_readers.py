"""Archive-only v4 evidence readers. Reports are recomputed, summaries are advisory."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import tarfile
import binascii
import zlib

from kit import v4_analysis as a, v4_teacher as t, canonical, tokens_io
from kit.v4_archive import Archive
from kit.v4_budget import admit
from kit.v4_campaign import slots, build, TASKS
from kit.v4_contract import check_resolved

# Pinned authors' mcq.py; copying this exact file into the return archive closes
# the scorer dependency without inventing a replacement scoring rule.
MCQ_SHA256='fbc2bc87b9fadc94b22bfc244af346936c3486d7dfab716a0ce2151abbe3fb9c'


def decoded_response(archive,ids):
    cache=getattr(archive,'_decoded_responses',None)
    if cache is None:cache={};archive._decoded_responses=cache
    key=tuple(ids)
    if key not in cache:cache[key]=archive.student_tokenizer.decode(ids,skip_special_tokens=True)
    return cache[key]


def rendered_prompt_hash(archive,messages):
    cache=getattr(archive,'_rendered_prompt_hashes',None)
    if cache is None:cache={};archive._rendered_prompt_hashes=cache
    key=json.dumps(messages,sort_keys=True)
    if key not in cache:
        cache[key]=t.sha(archive.student_tokenizer.apply_chat_template(messages,tokenize=False,
            add_generation_prompt=True,enable_thinking=False))
    return cache[key]


def verdicts(archive,phase):
    """Required real-runner chronology; start and verdict must identify this campaign."""
    campaign='v4-'+phase
    rows=build(phase)['rows']
    states={}; errors=[]
    for row in rows:
        prefix='campaign/'+campaign+'/'+row['id']+'/'
        entries=sorted([name for name in archive.files if name.startswith(prefix) and name.endswith('/verdict.json')],key=lambda name:int(name.split('/')[-2].split('-')[1]))
        if not entries:
            states[row['id']]='missing';errors.append('missing runner row: '+row['id']);continue
        for entry in entries:
            verdict=archive.json(entry);start=archive.json(entry.removesuffix('verdict.json')+'start.json')
            if start.get('campaign')!=campaign or start.get('row')!=row['id'] or verdict.get('row')!=row['id']:
                errors.append('runner identity differs: '+row['id'])
        state=archive.json(entries[-1])['verdict'];states[row['id']]=state
        if state!='PASS':errors.append('runner row '+row['id']+': '+state)
    return states,errors


def scorer_context(archive,temp):
    """Materialize ONLY verified tokenizer/scorer data, never weights or archive code."""
    base=Path(temp);tokenizer=base/'tokenizer';tokenizer.mkdir()
    prefix='v4/report-inputs/tokenizer/'
    for name,raw in archive.files.items():
        if name.startswith(prefix):
            if Path(name).name not in {'tokenizer.json','tokenizer_config.json','config.json'}:raise ValueError('unexpected frozen tokenizer file')
            target=tokenizer/Path(name).name
            target.write_bytes(raw)
    from kit.v4_contract import TOKENIZER_FILES
    for name,digest in TOKENIZER_FILES.items():
        if hashlib.sha256(archive.files.get(prefix+name,b'')).hexdigest()!=digest:raise ValueError('frozen tokenizer changed: '+name)
    raw=archive.files.get('v4/report-inputs/mcq.py')
    if raw is None or hashlib.sha256(raw).hexdigest()!=MCQ_SHA256:
        raise ValueError('pinned authors scorer missing/changed')
    source=base/'mcq.py';source.write_bytes(raw)
    module=t.load_module('v4_archive_mcq',source)
    student=t.load_student_tokenizer(tokenizer)
    archive.student_tokenizer=student
    return student,module


def scoring(archive,phase,checkpoint,task,cap,mcq,limit=None):
    """Score raw answers using the bed, including incorrect-answer token expenditure."""
    root=f'v4/report-{phase}/eval/{checkpoint}-{task}-{cap}/'
    rows=archive.jsonl(root+'responses.jsonl')
    token_rows=archive.jsonl(root+'tokens.jsonl')
    try:
        token_map={str(r['id']):tokens_io.unpack(r['ids']) for r in token_rows}
    except (ValueError,TypeError,AttributeError,zlib.error,binascii.Error) as exc:
        raise ValueError('invalid evaluation token payload') from exc
    if len(token_map)!=len(token_rows):raise ValueError('duplicate token identities')
    items=archive.jsonl('v4/report-inputs/'+task+'-panel.jsonl')
    if hasattr(archive,'expected_panels') and items!=archive.expected_panels[task]:raise ValueError('scoring panel differs from pinned source')
    items=items[:limit] if limit else items
    def key(item):
        index=str(item['extra_info']['index'])
        return 'sciknoweval-test-'+index if task.startswith('chemistry') else index
    gold={key(item):str(item['reward_model']['ground_truth']) for item in items}
    if set(token_map)!=set(gold):raise ValueError('missing or foreign evaluation token identities')
    if len(items)!=(limit or a.PANELS[task]):raise ValueError('wrong full-panel denominator: '+task)
    result=[]
    for row in rows:
        ident=str(row['id']);text=row['response']
        if ident not in gold or ident not in token_map:raise ValueError('foreign/missing scoring or token identity')
        ids=token_map[ident]
        if row.get('output_tokens')!=len(ids):raise ValueError('scoring token count differs')
        bed='chemistry' if task.startswith('chemistry') else 'finqa'
        strict=int(mcq.compute_score(text,gold[ident])['acc']) if bed=='chemistry' else int(t.finqa.compute_score('finqa',text,gold[ident])['acc'])
        candidate=canonical.extract(bed,text)
        correct=int(canonical.is_correct(bed,candidate,gold[ident]))
        result.append({'id':ident,'strict':strict,'canonical':correct,'tokens':len(ids)})
    report=a.panel(result,set(gold),cap)
    bed_score=archive.json(root+'bed-score.json')
    if hasattr(archive,'expected_panels'):
        from kit.eval_bed import DECODING, ENGINE, render_messages
        from kit.v4_evidence import weights, require
        require(bed_score.get('mode')=='generate' and bed_score.get('max_new_tokens')==cap,'wrong evaluation mode/cap')
        require(bed_score['decoding']=={**DECODING,'max_tokens':cap},'evaluation decoder differs')
        require(all(bed_score['engine'].get(k)==v for k,v in {**ENGINE,'max_model_len':12288,'eager':True,'deterministic':True,'batch_invariant':True}.items()),'evaluation engine differs')
        require(bed_score['generation']=={'n':1,'enable_lora':False,'enforce_eager':True,'seed':0,'chat_template':{'add_generation_prompt':True,'enable_thinking':False}},'evaluation generation differs')
        require(bed_score['responses_sha256']==t.sha(archive.files[root+'responses.jsonl']) and bed_score['tokens_sha256']==t.sha(archive.files[root+'tokens.jsonl']),'evaluation raw hash differs')
        indexed={key(i):i for i in items}
        for row,tokens in zip(rows,token_rows):
            ident=str(row['id']);require(str(tokens['id'])==ident,'evaluation token order differs')
            decoded=decoded_response(archive,token_map[ident])
            require(decoded==row['response'],'evaluation token IDs do not decode to response')
            require(tokens['prompt_sha256']==rendered_prompt_hash(archive,indexed[ident]['prompt']),'held-out rendered prompt differs')
        expected_identity=archive.json('v4/report-inputs/model-identities.json')['initial'] if checkpoint=='baseline' or checkpoint.startswith('agreement') else archive.json('runs/'+checkpoint+'/env/export-identity-step'+('2' if phase in ('qualification','phase0') else '40')+'.json')
        require(weights(archive.json(root+'checkpoint-identity.json'))==weights(expected_identity),'scored checkpoint shards differ')
    fingerprint=bed_score['machine']
    # Recompute the fingerprint digest; also bind the selected physical UUID from
    # containment instead of trusting the report's machine.id.
    identity={k:fingerprint[k] for k in ('gpus','cuda_visible_devices','versions','deterministic')}
    recomputed=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    if fingerprint.get('id')!=recomputed:raise ValueError('forged machine fingerprint')
    if not fingerprint['gpus'] or fingerprint['deterministic'] is not True:raise ValueError('unknown scoring machine')
    if hasattr(archive,'scheduler_rows'):
        from kit.v4_evidence import scoring_gpu
        physical=scoring_gpu(archive,phase,checkpoint,task,cap,fingerprint)
        report['physical_scoring_gpu']=physical
        recomputed+=':'+physical
    report['fingerprint']=recomputed
    return report


def corpus(archive,phase,task,tokenizer,mcq):
    """Reverify every raw attempt and rebuild first-acceptable corpora and coverage."""
    base=f'v4/report-{phase}/'
    pool=archive.jsonl(base+task+'-pool.jsonl')
    if hasattr(archive,'expected_pools') and pool!=(archive.expected_pools[task][:20] if phase in ('qualification','phase0') else archive.expected_pools[task]):raise ValueError('corpus pool differs from pinned source')
    if len(pool)!=(20 if phase in ('qualification','phase0') else 1441):raise ValueError('wrong original corpus denominator')
    lookup={str(i['id']):i for i in pool}
    if len(lookup)!=len(pool):raise ValueError('duplicate corpus pool identity')
    accepted={};tables={};docs={};verified={}
    old=t.chemistry.authors._SCORERS.get('mcq')
    t.chemistry.authors._SCORERS['mcq']=mcq
    try:
        for name,filename in (('teacher','demonstrations.jsonl'),('rewrite','rewrites.jsonl')):
            root=base+name+'/'+task+'/'
            doc=archive.json(root+'manifest.json');docs[name]=doc
            if doc.get('technical_synthetic') or doc.get('stand_in_teacher'):raise ValueError('scientific archive contains stand-in targets')
            items=archive.jsonl(root+'pool.jsonl')
            if items!=pool:raise ValueError('corpus original pools differ')
            identity=archive.json('v4/report-inputs/model-identities.json')['teacher' if name=='teacher' else 'initial']
            if doc.get('model_revision')!=identity['model_revision'] or doc.get('model_file_hashes')!=identity['model_file_hashes']:raise ValueError('generator revision/weight identity differs')
            expected_role='external_teacher' if name=='teacher' else 'frozen_initial_student'
            if doc.get('generator_role')!=expected_role or doc.get('attempts')!=4 or doc.get('thinking') is not False or doc.get('max_new_tokens')!=(t.TEACHER_NEW_TOKENS if name=='teacher' else t.REWRITE_NEW_TOKENS):raise ValueError('corpus decoding/role differs')
            if doc.get('pool_hash')!=t.sha(json.dumps(pool,sort_keys=True)):raise ValueError('original corpus hash differs')
            if name=='rewrite' and doc.get('demonstrations_hash')!=t.sha(json.dumps(archive.jsonl(base+'teacher/'+task+'/demonstrations.jsonl'),sort_keys=True)):raise ValueError('rewrite demonstration source differs')
            raw=archive.jsonl(root+'raw_outputs.jsonl')
            results=[];attempt_counts=defaultdict(int);finished=set()
            for row in raw:
                if str(row['id']) not in lookup:raise ValueError('foreign teacher question')
                if type(row.get('attempt')) is not int or not 1<=row['attempt']<=4:raise ValueError('corpus attempt outside 1..4')
                ident=str(row['id'])
                if ident in finished or row['attempt']!=attempt_counts[ident]+1:
                    raise ValueError('corpus journal omitted/repeated attempt or continued after first acceptable attempt')
                attempt_counts[ident]+=1
                if phase in ('qualification','phase0') and row.get('input_over_cap'):raise ValueError('silent input truncation or input allowance exceeded')
                result=t.verify(lookup[str(row['id'])],row,tokenizer,mapped_target=name=='rewrite')
                expected_seed,expected_temperature=(t.SCHEDULE[row['attempt']-1] if name=='teacher' else (t.rewrite_seed(row['id'],row['attempt']),.7))
                if row.get('seed')!=expected_seed or row.get('temperature')!=expected_temperature or row.get('top_p')!=(1 if name=='teacher' else .95):raise ValueError('attempt decoding schedule differs')
                results.append(result)
                if result['verified']:finished.add(ident)
            targets=archive.jsonl(root+filename)
            rejections=archive.jsonl(root+'rejections.jsonl')
            expected_rejections=[r for r in results if not r['verified']]
            if len(rejections)!=len(expected_rejections) or any(any(record.get(field)!=expected[field] for field in ('id','attempt','verified','primary_reason','reasons','verifier_detail','tokens_8b','context_tokens_8b')) for record,expected in zip(rejections,expected_rejections)):
                raise ValueError('typed rejection journal differs from reverified attempts')
            if any(r.get('stand_in_teacher') is not False or r.get('teacher_model')!=doc['teacher_model'] or r.get('verified') is not True for r in targets):raise ValueError('corpus target provenance/verdict differs')
            ids={str(r['id']) for r in targets}
            if len(ids)!=len(targets):raise ValueError('duplicate corpus target')
            tables[name]=a.rejection_table(pool if name=='teacher' else [i for i in pool if str(i['id']) in accepted['teacher']],results,ids)
            accepted[name]=ids;verified[name]=results
            for ident in lookup if name=='teacher' else accepted['teacher']:
                if ident not in ids and sum(str(r['id'])==ident for r in results)!=4:raise ValueError('exhausted question lacks four attempts')
            winners={str(r['id']):r for r in results if r['verified']}
            if any(r['text']!=winners[str(r['id'])]['text'] or r['attempt']!=winners[str(r['id'])]['attempt'] for r in targets):raise ValueError('target differs from first acceptable raw attempt')
            active=pool if name=='teacher' else [i for i in pool if str(i['id']) in accepted['teacher']]
            demos={str(r['id']):r for r in archive.jsonl(base+'teacher/'+task+'/demonstrations.jsonl')}
            prompts={str(i['id']):t.rewrite_messages(i,demos[str(i['id'])]) if name=='rewrite' else t.teacher_messages(i) for i in active}
            if doc['prompt_hashes']!={key:t.sha(json.dumps(value,sort_keys=True)) for key,value in prompts.items()}:raise ValueError('corpus generation prompts differ')
            expected_decoding={'temperature':.7 if name=='rewrite' else 'schedule','top_p':.95 if name=='rewrite' else 1,'top_k':-1,'repetition_penalty':1,'n':1,'tp':2 if name=='teacher' else 1,'executor_backend':'mp' if name=='teacher' else 'uni','v1_multiprocessing':False,'dtype':'bfloat16','max_model_len':8192}
            if doc['decoding']!=expected_decoding or doc['input_cap']!=(6144 if name=='rewrite' else 4096) or doc['response_cap']!=2048:raise ValueError('generator decoding/context differs')
            gate=archive.json(root+'coverage.json')
            if gate['tasks'][task]['original_pool']!=len(pool):raise ValueError('coverage gate uses filtered denominator')
        initial=archive.json('v4/report-inputs/model-identities.json')['initial']
        if docs['rewrite']['model_revision']!=initial['model_revision'] or docs['rewrite']['model_file_hashes']!=initial['model_file_hashes'] or docs['rewrite']['generator_role']!='frozen_initial_student':raise ValueError('R generator is not frozen initial 8B')
        intersection=accepted['teacher']&accepted['rewrite']
        families={}
        for family in sorted({t.family_of(i) for i in pool}):
            ids={str(i['id']) for i in pool if t.family_of(i)==family}
            required=(7*len(ids)+9)//10 if family.startswith('molar weight') else None
            families[family]={'original':len(ids),'teacher':len(ids&accepted['teacher']),'intersection':len(ids&intersection),'required':required,
                              'pass':required is None or len(ids&intersection)>=required}
        return {'task':task,'original':len(pool),'teacher':len(accepted['teacher']),'rewrite':len(accepted['rewrite']),
                'intersection':len(intersection),'required':(4*len(pool)+4)//5,'families':families,
                'pass':len(intersection)>=(4*len(pool)+4)//5 and all(f['pass'] for f in families.values()),
                'teacher_acceptance':tables['teacher'],'rewrite_acceptance':tables['rewrite'],
                'accepted_ids':{k:sorted(v) for k,v in accepted.items()},
                'lengths':{name:[{'id':r['id'],'attempt':r['attempt'],'student_response_tokens':r['tokens_8b'],
                                  'student_context_tokens':r['context_tokens_8b'],'response_characters':len(r['text']),
                                  'verified':r['verified'],'reasons':r['reasons']} for r in rows] for name,rows in verified.items()}}
    finally:
        if old is None:t.chemistry.authors._SCORERS.pop('mcq',None)
        else:t.chemistry.authors._SCORERS['mcq']=old


def run_record(archive,ident,phase):
    """Read actual resolved recipe and completed-update records, never registered summaries."""
    import yaml
    root='runs/'+ident+'/'
    summary=archive.json(root+'run-summary.json')
    from kit.v4_qualification import departures_hash
    if summary.get('departures_sha256')!=departures_hash(archive.files['v4/report-inputs/v4_departures.json']):
        raise ValueError('run departures hash differs')
    config=yaml.safe_load(archive.files[root+'env/resolved-config.yaml'])
    arm=summary['arm'];seed=summary['seed']
    resume_path=None
    if config['trainer'].get('resume_mode')!='disable':
        from kit.v4_auxiliary import chronology
        trace=chronology(archive,phase)
        row='train-'+ident;entry=trace['rows'][row]
        from kit.v4_evidence import require
        require(not any(row in error for error in trace['errors']),'disqualified infrastructure retry')
        attempt=next(a for a in entry['attempts'] if a['attempt']==entry['first_valid_attempt'])
        resume_path=attempt['start']['env'].get('V4_RESUME_PATH')
        require(resume_path,'unreviewed optimizer resume')
        review=archive.json(f'v4/retries/v4-{phase}/{row}/attempt-{attempt["attempt"]}.json')
        state=archive.json(root+f'env/saved-state-step{review["completed_updates"]}.json')
        require(review['state_policy']=='resume_valid_state' and state['state_sha256']==review['saved_state_sha256'] and state['path']==resume_path,'saved-state recovery identity differs')
    old=os.environ.get('KIT_FINISH_GATE');os.environ['KIT_FINISH_GATE']='1'
    reward_path=None
    if arm=='S':
        reward_path=config['custom_reward_function']['path']
        if not reward_path.endswith('/kit/beds/v4_reward.py'):raise ValueError('S reward path differs')
    try:check_resolved(config,arm,profile_name='technical-smoke' if phase in ('qualification','phase0') else 'scientific',reward_path=reward_path,resume_path=resume_path)
    finally:
        if old is None:os.environ.pop('KIT_FINISH_GATE',None)
        else:os.environ['KIT_FINISH_GATE']=old
    steps=2 if phase in ('qualification','phase0') else 40
    receipt=archive.json(root+'env/optimizer-updates.json')
    if receipt['completed_optimizer_updates']!=steps or summary['completed_optimizer_updates']!=steps:raise ValueError('incomplete optimizer dose: '+ident)
    if summary['merged']!=1 or summary['merge_returncode']!=0:raise ValueError('merge failed: '+ident)
    metrics=archive.jsonl(root+'metrics.jsonl')
    from kit.v4_qualification import check_metrics
    measured=check_metrics(metrics,arm,steps)
    if summary.get('ema_movement')!=measured['ema_movement']:raise ValueError('EMA summary differs from per-step metrics')
    if not set(range(1,steps+1))<={m.get('step') for m in metrics}:raise ValueError('missing optimizer steps: '+ident)
    loss_key='train/loss' if arm in 'FR' else 'actor/pg_loss'
    counts=[m.get('data',m).get('v4/completed_optimizer_updates') for m in metrics if loss_key in m.get('data',m)]
    if not counts or counts[-1]!=steps or any(value is None or value<0 or value>steps for value in counts):raise ValueError('optimizer metric dose differs: '+ident)
    if seed not in (101,102,103):raise ValueError('wrong seed: '+ident)
    resolved_seed=config['trainer']['seed'] if arm in 'FR' else config['data']['seed']
    if resolved_seed!=seed:raise ValueError('resolved and recorded seeds differ: '+ident)
    if phase in ('qualification','phase0'):
        from kit.v4_evidence import peak_memory
        summary={**summary,'peak_memory_per_gpu':peak_memory(archive,ident)}
    return summary


def qualification_evidence(archive):
    """Read raw qualification evidence locally or from a verified return archive."""
    report={'status':'stop: technical','reasons':[],'owner_proceed_required':True,'tables':{}}
    try:
        from kit.v4_qualification import failed_engines_first
        failed_engines_first(archive)
        with tempfile.TemporaryDirectory(prefix='v4-read-') as temp:
            from kit.v4_evidence import environment, containment, data_context, common_data, training_provenance, scheduler_rows
            report['tables']['environment']=environment(archive,'qualification')
            report['tables']['containment']=containment(archive)
            scheduler_rows(archive,'qualification')
            from kit.v4_qualification import archive_static, archive_agreement
            report['tables']['static_qualification']=archive_static(archive)
            report['departures_sha256']=report['tables']['static_qualification']['departures_sha256']
            tokenizer,mcq=scorer_context(archive,temp)
            from kit.v4_restore_check import verify_archive
            with data_context(archive,tokenizer,mcq):
                report['tables']['scoring_agreement']=archive_agreement(archive,mcq)
                report['tables']['restoration']=verify_archive(archive)
                from kit.v4_qualification import archive_determinism
                report['tables']['teacher_determinism']=archive_determinism(archive)
                from kit.v4_qualification import archive_ratio
                report['tables']['teacher_token_ratio']=archive_ratio(archive)
                baseline={};fingerprints=set()
                for task in (*TASKS,'chemistry_probe'):
                    baseline[task]={}
                    for cap in (2048,8192):
                        value=scoring(archive,'qualification','baseline',task,cap,mcq)
                        fingerprints.add(value.pop('fingerprint'));baseline[task][str(cap)]=value
                report['tables']['baseline']=baseline
                report['tables']['corpora']={task:corpus(archive,'qualification',task,tokenizer,mcq) for task in TASKS}
                report['tables']['common_data']=common_data(archive,'qualification',tokenizer)
                records={}
                for arm in ('S','F','R','D'):
                    for task in TASKS:
                        ident=f'q-{arm}-{task}'; records[ident]=run_record(archive,ident,'qualification')
                        from kit.v4_campaign_ops import job_for
                        records[ident]['provenance']=training_provenance(archive,'qualification',job_for(ident,'qualification'),records[ident],tokenizer)
                        for panel in TASKS:
                            value=scoring(archive,'qualification',ident,panel,2048,mcq,limit=20)
                            fingerprints.add(value['fingerprint'])
                from kit.v4_auxiliary import probe_point
                from kit.v4_evidence import weights
                initial=weights(archive.json('v4/report-inputs/model-identities.json')['initial'])
                for task in TASKS:
                    frozen=sorted(archive.expected_pools[task],key=lambda i:t.sha('v4-fixed-probe|'+task+'|'+i['id']))[:20]
                    if archive.jsonl('v4/report-inputs/'+task+'-probe.jsonl')!=frozen:raise ValueError('probe not fixed before coverage')
                    probe_point(archive,'qualification','q-S-chemistry-'+task+'-0',task,frozen,mcq,initial)
                    sample=archive.expected_pools[task][:20]
                    demos={r['id']:r for r in archive.jsonl(f'v4/report-qualification/teacher/{task}/demonstrations.jsonl')}
                    probe_point(archive,'qualification','conditioned-'+task,task,sample,mcq,initial,conditioned=True,demonstrations=demos)
                if len(fingerprints)!=1:raise ValueError('cross-GPU/machine scoring')
                report['tables']['training']=records
                from kit.v4_timing import qualification_timings
                completed={}
                for task in TASKS:
                    completed['teacher_questions:'+task]=20
                    completed['rewrite_questions:'+task]=report['tables']['corpora'][task]['teacher']
                    for role in ('teacher','rewrite'):
                        completed[f'reuse_verify:{role}:{task}']=len(archive.jsonl(f'v4/report-qualification/{role}/{task}/raw_outputs.jsonl'))
                inputs=qualification_timings(archive,completed=completed)
                budget=admit(archive.json('k8b4/containment/allocation-ledger.json'),inputs['timings'],inputs.get('completed'),probe_n=inputs['probe_n'])
            report['budget']=budget;report['status']=budget['status'];report['reasons']=budget['reasons']
            if budget['status']=='proceed':
                from kit.v4_qualification import measured_caps
                report['row_caps']=measured_caps(archive,budget)
    except (ValueError,KeyError,TypeError,OSError,tarfile.TarError,EOFError) as exc:
        report['reasons']=[str(exc)]
    return report


def read_qualification(path):
    """Technical pass and scope recommendation; owner proceed is always separate."""
    report={'status':'stop: technical','reasons':[],'owner_proceed_required':True,'tables':{}}
    try:
        with Archive(path) as archive:
            _,errors=verdicts(archive,'qualification')
            report=qualification_evidence(archive)
            if errors:
                report['status']='stop: technical';report['reasons']=errors+report['reasons']
    except (ValueError,KeyError,TypeError,OSError,tarfile.TarError,EOFError) as exc:report['reasons']=[str(exc)]
    return report


def campaign_evidence(archive):
    """Read registered primary arithmetic and all required auxiliary evidence independently.

    Missing diagnostics never erase already established failures. Prescribed stops
    identify expected unrun rows instead of treating them as successful evidence.
    """
    from kit.v4_evidence import environment, containment, data_context, common_data, training_provenance, scheduler_rows, require
    from kit.v4_auxiliary import chronology, mechanism, frontier, resource_costs
    report={'status':'incomplete','reasons':[],'tables':{},'expected_unrun':[],'known_failures':[],
            'selection_history':a.SELECTION_HISTORY,'historical_finqa_reference':{'correct':948,'label':'historical reference'}}
    try:
        from kit.v4_qualification import departures_hash,archive_static
        report['tables']['environment']=environment(archive,'main')
        report['departures_sha256']=departures_hash(archive.files['v4/report-inputs/v4_departures.json'])
        report['tables']['static_qualification']=archive_static(archive)
        report['tables']['containment']=containment(archive)
        owner=archive.json('v4/report-main/owner-registration.json')
        env=report['tables']['environment']['environment']
        require(owner['owner']['owner_proceed'] is True and owner['owner']['qualification_sha256']==owner['qualification_sha256'], 'owner proceed archive binding differs')
        require(owner['new_kit_tag']==env['kit_tag']==owner['owner']['new_kit_tag'] and env['kit_tag']!=owner['qualification_environment']['kit_tag'],'main did not use the new owner-approved kit tag')
        require(env['trainer_commit']==owner['qualification_environment']['trainer_commit'] and env['inference_commit']==owner['qualification_environment']['inference_commit'],'trainer/inference environment changed since qualification')
        scheduler_rows(archive,'qualification');scheduler_rows(archive,'main')
        trace=chronology(archive,'main');report['tables']['chronology']=trace
        report['reasons']+=trace['errors']
        with tempfile.TemporaryDirectory(prefix='v4-read-') as temp:
            tokenizer,mcq=scorer_context(archive,temp)
            from kit.v4_qualification import archive_agreement
            from kit.v4_restore_check import verify_archive
            with data_context(archive,tokenizer,mcq):
                report['tables']['scoring_agreement']=archive_agreement(archive,mcq)
                report['tables']['restoration']=verify_archive(archive)
                from kit.v4_qualification import archive_determinism
                report['tables']['teacher_determinism']=archive_determinism(archive)
                from kit.v4_qualification import archive_ratio
                report['tables']['teacher_token_ratio']=archive_ratio(archive)
                from kit.v4_timing import qualification_timings
                from kit.v4_qualification import measured_caps
                qinputs=qualification_timings(archive)
                qbudget=admit(archive.json('k8b4/containment/allocation-ledger.json'),qinputs['timings'],probe_n=qinputs['probe_n'])
                proposed=measured_caps(archive,qbudget)
                require(owner['owner'].get('row_caps')==proposed,'registered row caps differ from qualification measurements')
                report['row_caps']=proposed
                coverage={}
                for task in TASKS:
                    coverage[task]=corpus(archive,'main',task,tokenizer,mcq)
                    report['tables']['coverage']=coverage
                if not all(c['pass'] for c in coverage.values()):
                    report.update(status='stopped: coverage',reasons=['full-pool intersection or family coverage failed'],expected_unrun=[s['id'] for s in slots()])
                    return report
                # Registered code must have executed the gate, independently of its reported pass.
                gate=archive.json('v4/report-main/coverage.json')
                require(set(gate['tasks'])==set(TASKS),'full-pool gate code report missing task')
                report['tables']['common_data']=common_data(archive,'main',tokenizer)
                baseline={};baseline_diagnostics={};fingerprints=set()
                for task in (*TASKS,'chemistry_probe'):
                    baseline_diagnostics[task]={}
                    for cap in (2048,8192):
                        value=scoring(archive,'qualification','baseline',task,cap,mcq)
                        fingerprints.add(value['fingerprint']);baseline_diagnostics[task][str(cap)]=value
                    if task in TASKS:baseline[task]=baseline_diagnostics[task]['2048']
                report['tables']['baseline']=baseline_diagnostics
                evaluations={};records={};invalid={}
                retry_invalid={ident for ident,entry in trace['rows'].items() if any(ident in error for error in trace['errors'])}
                def read_job(job):
                    ident=job['id']
                    try:
                        require('train-'+ident not in retry_invalid,'disqualified infrastructure retry')
                        require(not job['incoming'] or job['incoming'] in records,'dependent incoming lineage disqualified')
                        summary=run_record(archive,ident,'main')
                        summary['provenance']=training_provenance(archive,'main',job,summary,tokenizer)
                        scores={}
                        for task in (*TASKS,'chemistry_probe'):
                            scores[task]={}
                            for cap in (2048,8192):
                                value=scoring(archive,'main',ident,task,cap,mcq)
                                fingerprints.add(value['fingerprint']);scores[task][str(cap)]=value
                        records[ident]=summary;evaluations[ident]=scores
                    except (ValueError,KeyError,TypeError) as exc:invalid[ident]=str(exc)
                for job in slots()[:2]:read_job(job)
                qualification={}
                for task in TASKS:
                    ident='D-'+task+'-alone-s101'
                    if ident not in evaluations:continue
                    score=evaluations[ident][task]['2048'];transition=a.paired(baseline[task]['verdicts'],score['verdicts'])
                    qualification[task]={'baseline':baseline[task]['strict'],'standalone':score['strict'],'gain':transition['net'],
                        'transitions':transition,'acquisition_pass':transition['net']>=a.GAIN[task], 'cost_pass':a.cost_pass(score,baseline[task])}
                    qualification[task]['pass']=qualification[task]['acquisition_pass'] and qualification[task]['cost_pass']
                report['tables']['D_qualification']=qualification
                require(len(fingerprints)==1,'cross-GPU/machine scoring before D standalone gate')
                if len(qualification)==2 and not all(r['pass'] for r in qualification.values()):
                    report.update(status='stopped: D standalone',reasons=['seed-101 D standalone acquisition or deployment cost failed'],expected_unrun=[s['id'] for s in slots()[2:]])
                    report['known_failures']=[{'arm':'D','standalone':task,'criteria':[key for key in ('acquisition_pass','cost_pass') if not row[key]]} for task,row in qualification.items() if not row['pass']]
                    return report
                for job in slots()[2:]:read_job(job)
                report['disqualified']=invalid
                report['tables']['scores']=evaluations;report['tables']['training']=records
                if len(fingerprints)!=1:report['reasons'].append('cross-GPU/machine scoring')
                if invalid:report['reasons'].append('missing or disqualified required slots')
                cells={};final_counts={arm:{} for arm in ('S','F','R','D')};lexical={}
                subsets=None
                try:subsets=a.frozen_subsets(archive.files,baseline['chemistry']['verdicts'])
                except (ValueError,KeyError,TypeError) as exc:report['reasons'].append('missing diagnostic: '+str(exc))
                for arm in ('S','F','R','D'):
                    cells[arm]={}
                    for seed in (101,102,103):
                        for first,second in (TASKS,TASKS[::-1]):
                            stage=f'{arm}-{first}-alone-s{seed}';final=f'{arm}-{first}-{second}-s{seed}';key=f'{seed}:{first}-{second}'
                            standalones={task:f'{arm}-{task}-alone-s{seed}' for task in TASKS}
                            if any(ident not in evaluations for ident in [stage,final,*standalones.values()]):continue
                            stage_scores={task:evaluations[stage][task]['2048'] for task in TASKS}
                            final_scores={task:evaluations[final][task]['2048'] for task in TASKS}
                            standalone_scores={task:evaluations[ident][task]['2048'] for task,ident in standalones.items()}
                            result=a.cell(first,second,baseline,stage_scores,final_scores,standalone=standalone_scores)
                            cells[arm][key]=result;final_counts[arm][key]=final_scores[second]['strict']
                            if not result['met']:report['known_failures'].append({'arm':arm,'cell':key,'criteria':result['failures']})
                            if subsets is not None:
                                comparisons={'first_acquisition':(baseline['chemistry'],stage_scores['chemistry'])} if first=='chemistry' else {'second_gain':(stage_scores['chemistry'],final_scores['chemistry'])}
                                comparisons['final_first_acquisition' if first=='chemistry' else 'final_second_acquisition']=(baseline['chemistry'],final_scores['chemistry'])
                                if first=='chemistry':comparisons['retention']=(stage_scores['chemistry'],final_scores['chemistry'])
                                acquisition={k for k in comparisons if k!='retention'}
                                lexical[arm+':'+key]={criterion:a.lexical(before['verdicts'],after['verdicts'],subsets,result['passes'][criterion] if criterion in acquisition else False) for criterion,(before,after) in comparisons.items()}
                report['tables']['cells']=cells;report['tables']['lexical']=lexical
                if subsets is not None:
                    report['tables']['lexical_checkpoints']={ident:{str(cap):{mode:{name:a.paired(baseline_diagnostics['chemistry'][str(cap)][mode+'_verdicts'],scores['chemistry'][str(cap)][mode+'_verdicts'],ids) for name,ids in subsets.items()} for mode in ('canonical',)} | {'strict':{name:a.paired(baseline_diagnostics['chemistry'][str(cap)]['verdicts'],scores['chemistry'][str(cap)]['verdicts'],ids) for name,ids in subsets.items()}} for cap in (2048,8192)} for ident,scores in evaluations.items()}
                required={f'{seed}:{order}' for seed in (101,102,103) for order in ('chemistry-finqa','finqa-chemistry')}
                comparison=None
                if all(set(final_counts[arm])==required for arm in ('D','S')):
                    comparison=a.advantage(final_counts['D'],final_counts['S']);report['tables']['advantage_over_S']=comparison
                    if not comparison['met']:report['known_failures'].append({'arm':'D','criterion':'advantage_over_S','differences_pp':comparison['differences_pp']})
                else:report['reasons'].append('missing primary comparison cells')
                report['tables']['matches_D']={arm:a.matches(final_counts['D'],final_counts[arm],cells[arm]) for arm in ('F','R')}
                primary=bool(comparison and comparison['met'] and len(cells['D'])==6 and all(c['met'] for c in cells['D'].values()) and len(qualification)==2 and all(q['pass'] for q in qualification.values()))
                report['primary_arithmetic_met']=primary
                report['primary_criterion']='D acquisition, retention and deployment cost in all six seed/order cells, plus advantage over S in every seed'
                report['failed_criteria_and_cells']=report['known_failures']
                for name,reader in (('mechanism',lambda:mechanism(archive,coverage,mcq,records)),('frontier',lambda:frontier(archive,mcq)),('resources',lambda:resource_costs(archive))):
                    try:
                        value=reader()
                        if name=='mechanism':value,errors=value;report['reasons']+=errors
                        report['tables'][name]=value
                    except (ValueError,KeyError,TypeError) as exc:report['reasons'].append(name+': '+str(exc))
                _,row_errors=verdicts(archive,'main');report['reasons']+=row_errors
                for phase in ('main',):
                    expected_gpu_rows={r['id'] for r in build(phase)['rows'] if '{kit}/p4_contain.py' in r['command'] and 'row' in r['command']}
                    missing=expected_gpu_rows-{key for (p,key) in archive.scheduler_rows if p==phase}
                    if missing:report['reasons'].append('missing valid GPU-row containment: '+', '.join(sorted(missing)))
                if 'k8b4/containment/v4-stop.json' in archive.files:report['reasons'].append('containment or allocation-ceiling stop remains recorded')
                if not report['reasons']:
                    report['status']='primary claim met' if primary else 'primary claim not met'
                    report['wording']=('Under these fixed recipes at 40 updates, demonstration-conditioned SDFT met the registered acquisition, retention and cost criteria in all six cells, on Chemistry without the identified shortcut and on FinQA, and exceeded self-only SDPO by at least 5 percentage points on the two-order mean in every seed, with no seed-and-order deficit exceeding 2.5 points.' if primary else 'Under these fixed recipes at 40 updates, demonstration-conditioned SDFT did not meet the registered criteria in the named cells at this dose.')
                    if not primary:report['wording']+=' Failures: '+json.dumps(report['known_failures'],sort_keys=True)+'.'
                    resources=report['tables']['resources']
                    report['additional_machinery_not_justified']=[arm for arm in ('F','R') if report['tables']['matches_D'][arm] and resources['lower_total_cost_than_D'][arm]]
                else:report['wording']='Incomplete; established criterion failures remain recorded and do not authorize rerunning them.'
    except (ValueError,KeyError,TypeError,OSError,tarfile.TarError,EOFError,ImportError) as exc:report['reasons'].append(str(exc))
    return report


def read_campaign(path):
    """Verify exhaustive archive integrity, then recompute the full registered reading."""
    try:
        with Archive(path) as archive:return campaign_evidence(archive)
    except (ValueError,KeyError,TypeError,OSError,tarfile.TarError,EOFError) as exc:
        return {'status':'incomplete','reasons':[str(exc)],'tables':{},'expected_unrun':[],'known_failures':[]}
