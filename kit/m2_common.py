"""Frozen connected continuation identities and deterministic branch rules."""
from kit.c_common import (KIT, STEPS, RETENTION_STEPS, lock, checked_model,
                          legal, panels as legal_panels, overlap)
from kit.p3b_common import *
from kit.m2_seed_policy import SEEDS

M = KIT / 'resources/m2'
PANELS = ('finqa', 'test', 'math_hard', 'ifeval', 'knowledge', 'math')
PANEL_SETS = ('finqa', 'test', 'math_hard', 'general-v1')
CONFIGS = ('generic', 'broad', 'earlier', 'both')


def protocol():
    frozen = json.loads((M / 'freeze.json').read_text())
    for name, digest in frozen['files'].items(): check(M / name, digest)
    from kit.c_common import protocol as c_protocol
    return dict(runtime_updates=s.file_hash(M/'runtime-updates.json'),resources=s.file_hash(M/'freeze.json'), c=c_protocol(),
                teacher_resources=s.file_hash(KIT/'resources/math-hard-teacher/SHA256SUMS'),
                implementation={str(p.relative_to(KIT)):s.file_hash(p) for p in
                                [*sorted(KIT.glob('m2_*.py')), *[KIT/name for name in
                                 ('message2_dryrun.py','math_hard_teacher.py','math_hard_teacher_read.py',
                                  'second_task_screen.py','message_handoff.py')]]})


def hard_data():
    manifest = json.loads((M/'manifest.json').read_text())
    result = {}
    for name, n in [('train_pool.jsonl',500), ('eval.jsonl',100)]:
        check(M/name, manifest['files'][name])
        rows = t.read_jsonl(M/name)
        require(len(rows)==len({r['id'] for r in rows})==n, 'hard-MATH membership/count changed')
        require(all(r['task']=='math_screen' and r['family']=='5' for r in rows), 'hard-MATH eligibility changed')
        result[name] = rows
    screen_path=KIT/'resources/math-hard-teacher/screen_panel.jsonl'
    check(screen_path,manifest['files']['screen_panel.jsonl'])
    screen = t.read_jsonl(screen_path)
    sets = [{r['id'] for r in rows} for rows in [*result.values(),screen]]
    require(all(not a & b for i,a in enumerate(sets) for b in sets[i+1:]), 'hard-MATH sets overlap')
    return result


def binding(out, standin):
    from kit.m2_paths import local
    out=local(out)
    path = out/'binding.json'
    if not path.exists(): raise NotReady('message-2 binding pending')
    doc = json.loads(path.read_text())
    require(doc.get('technical_synthetic') is standin, 'synthetic binding mismatch; use --standin for technical_synthetic')
    require(doc['protocol']==protocol(), 'message-2 protocol changed after binding')
    return doc


def admission(out, doc):
    path = out/'admission.json'
    if not path.exists(): raise NotReady('Pending connected finals')
    result = json.loads(path.read_text())
    require(result['binding_sha256']==s.file_hash(out/'binding.json'), 'admission binding changed')
    if result['status']!='COMPLETE': raise NotReady('Pending connected finals')
    return result['configuration']


def durable_admission(out, doc, durable):
    """No sibling MATH work until CURRENT commits this exact admission."""
    config=admission(out,doc)
    if durable is None: raise NotReady('sibling MATH requires durable admission CURRENT')
    if (Path(durable)/'barriers/M9.json').exists():
        from kit.m2_node import barrier
        from kit.m2_assignment import sha
        commit=barrier(durable,'M9',sha(out/'scoring-assignment.json'))
        if commit.get('admission_sha256'):
            require(commit['admission_sha256']==s.file_hash(out/'admission.json'),'durable admission differs')
            return config
        from kit.m2_snapshot import restore
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/'admission'
            restore(durable,commit['task'],target)
            require(s.file_hash(target/'admission.json')==s.file_hash(out/'admission.json'),'durable admission differs')
        return config
    from kit.message_handoff import locked, committed
    with locked(durable):
        generation=committed(durable,verify=False)
        if generation is None or not (generation/'tree/admission.json').is_file():
            raise NotReady('sibling MATH requires durable admission CURRENT')
        manifest=json.loads((generation/'manifest.json').read_text())
        require('admission.json' in manifest['entries'],'CURRENT has no committed admission')
        check(generation/'tree/admission.json',manifest['entries']['admission.json']['sha256'])
        require(s.file_hash(generation/'tree/admission.json')==s.file_hash(out/'admission.json'),
                'durable admission differs from admission.json')
    return config


def legal_arm(config):
    require(config in CONFIGS, 'unknown replay configuration')
    return 'legal-'+config


def math_arm(trajectory):
    require(trajectory in ('plain','sibling'), 'unknown trajectory')
    return 'math-'+trajectory


def nomination_from_files(doc,files):
    if doc.get('nomination_policy')!='on-node-four-arm-reprobe':
        require(doc.get('technical_synthetic') is True,'real nomination requires fresh on-node re-probe')
        return doc['nomination']
    if 'nomination.json' not in files or 'nomination-commit.json' not in files:
        raise NotReady('fresh four-arm re-probe nomination is not durably committed')
    from kit.m2_reprobe import verify_record
    record=json.loads(files['nomination.json']);commit=json.loads(files['nomination-commit.json'])
    require(record['reprobe_binding']['protocol']==doc['protocol'],'nomination uses another scoring protocol')
    require(commit['record_sha256']==t.sha(files['nomination.json']),'nomination commit hash differs')
    require(commit['assignment_sha256']==t.sha(files['scoring-assignment.json']),'nomination assignment differs')
    value=verify_record(record,files['scoring-assignment.json'],t.sha(files['binding.json']))
    if value['status'] not in ('COMPLETE','BROAD_COMPLETE'):raise NotReady('re-probe nomination pending; no earlier-only fallback')
    return value


def nomination(out,doc):
    names=('binding.json','nomination.json','nomination-commit.json','scoring-assignment.json')
    files={n:(out/n).read_bytes() for n in names if (out/n).exists()}
    if doc.get('nomination_policy')=='on-node-four-arm-reprobe' and os.environ.get('M2_DURABLE'):
        scope=json.loads(files['nomination.json']).get('scope','full') if 'nomination.json' in files else 'full'
        root=Path(os.environ['M2_DURABLE'])/('nomination-broad' if scope=='broad' else 'nomination')
        if not (root/'COMMIT.json').exists():raise NotReady('durable re-probe nomination pending')
        require((root/'COMMIT.json').read_bytes()==files.get('nomination-commit.json') and
                (root/'record.json').read_bytes()==files.get('nomination.json'),'durable nomination differs')
    return nomination_from_files(doc,files)


def permitted_legal(out, doc, config):
    """Broad first; earlier is tested against the surviving incumbent, never chosen by MATH."""
    if config=='generic': return True
    current = nomination(out,doc)
    if config=='broad': return current['broad'] is True
    if current['earlier'] is not True: return False
    path = out/'broad-decision.json'
    if not path.exists(): raise NotReady('broad confirmation must be read before earlier-task confirmation')
    decision = json.loads(path.read_text())
    require(decision['binding_sha256']==s.file_hash(out/'binding.json'), 'broad decision binding changed')
    if decision['status']=='PENDING': raise NotReady('broad confirmation pending')
    return config == ('both' if decision['admitted'] else 'earlier')


def record(out, doc, arm, seed, step):
    require(seed in SEEDS and step in STEPS, 'unregistered seed/step')
    if arm.startswith('standalone-'):
        require(step==40,'standalone final only')
        key=f'{arm}:{seed}'
        if key not in doc.get('comparators',{}): raise NotReady('compatible standalone comparator absent')
        return doc['comparators'][key]
    if arm=='base':
        require(step==0, 'base at step 0 only')
        return doc['starts']['base']
    if arm.startswith('finqa-'):
        require(step==40 and arm in ('finqa-plain','finqa-sibling'), 'FinQA boundary only')
        return doc['starts'][arm+':'+str(seed)]
    if arm=='legal-plain':
        return doc['starts'][('finqa-plain:' if step==0 else 'legal-plain:')+str(seed)] if step in (0,40) else doc['c_exports'][f'{seed}:{step}']
    if arm.startswith('legal-'):
        config=arm.removeprefix('legal-')
        require(permitted_legal(out,doc,config), 'configuration not nominated or not the surviving incumbent')
        if step==0: return doc['starts']['finqa-sibling:'+str(seed)]
    elif arm.startswith('math-'):
        require(doc['hard_math']=='QUALIFIES', 'third stage dropped or pending')
        trajectory=arm.removeprefix('math-')
        require(trajectory in ('plain','sibling'), 'unknown MATH trajectory')
        parent='legal-plain' if trajectory=='plain' else legal_arm(admission(out,doc))
        if step==0: return record(out,doc,parent,seed,40)
    else: raise CorrectnessStop('unknown message-2 arm')
    run=out/'runs'/f'{arm}-s{seed}'
    if not (run/'complete.json').exists(): raise NotReady('stage not complete: '+str(run))
    frozen_files(run)
    return dict(model=str(run/f'hf-step{step}'), receipt=str(run/f'export-{step}.json'))


def save_status(out, doc):
    hard=doc['hard_math']
    status=dict(hard_math=hard, stage3='ENABLED' if hard=='QUALIFIES' else 'DROPPED' if hard=='DOES_NOT_QUALIFY' else 'PENDING',
                consequence='Two-task readout only; third-task acquisition and two-task retention through stage 3 lost.' if hard=='DOES_NOT_QUALIFY' else 'No reinterpretation of incomplete evidence as a failed screen.',
                technical_synthetic=doc['technical_synthetic'])
    bind(out/'stage3-status.json',status)
    return status


def validate_evidence(doc, files, standin):
    """Validate available compact manifests, scoring setup and parent/output receipts without weights."""
    from kit.ref_guard import native_files
    files = native_files(files)
    require(doc.get('technical_synthetic') is standin,'technical_synthetic evidence requires --standin')
    for name,raw in files.items():
        if name.endswith('/manifest.json'):
            prefix=name.rsplit('/',1)[0]+'/'
            for child,digest in json.loads(raw).get('files',{}).items():
                if prefix+child in files: require(t.sha(files[prefix+child])==digest,'returned frozen evidence differs: '+prefix+child)
    from kit.m2_assignment import validate_files
    validate_files(files,require_assignment=not standin)
    has_assignment='scoring-assignment.json' in files
    setup=json.loads(files['m2-scoring-machine.json']) if 'm2-scoring-machine.json' in files else None
    admitted=json.loads(files.get('admission.json',b'{}')).get('configuration')
    def incoming(arm,seed):
        if arm.startswith('legal-'): return doc['starts']['finqa-sibling:'+str(seed)]
        if arm=='math-plain': return doc['starts']['legal-plain:'+str(seed)]
        require(arm=='math-sibling' and admitted in CONFIGS,'sibling MATH has no durable admission')
        receipt=f'runs/legal-{admitted}-s{seed}/export-40.json'
        require(receipt in files,'missing admitted legal export receipt')
        return dict(model=str(Path(doc['out'])/'runs'/f'legal-{admitted}-s{seed}'/'hf-step40'),
                    receipt=str(Path(doc['out'])/receipt),identity=json.loads(files[receipt]))
    for name,raw in files.items():
        if name.startswith('runs/') and name.endswith('/summary.json'):
            summary=json.loads(raw); arm_seed=Path(name).parent.name; arm,seed=arm_seed.rsplit('-s',1);seed=int(seed)
            require(arm in [legal_arm(c) for c in CONFIGS]+['math-plain','math-sibling'],'unregistered training arm')
            expected=incoming(arm,seed)
            actual=summary['incoming_record']
            if not standin:
                require(summary['incoming']['model_file_hashes']==expected['identity']['model_file_hashes'],'connected parent export changed')
            else:
                from kit.m2_paths import equivalent
                portable=expected.get('identity',{}).get('model') if arm=='math-sibling' else None
                if portable:
                    parent=f'legal-{admitted}-s{seed}'
                    require(summary['incoming']==dict(technical_synthetic=True,model=portable) and
                        Path(actual['model']).parts[-3:]==('runs',parent,'hf-step40') and
                        Path(actual.get('receipt','')).parts[-3:]==('runs',parent,'export-40.json'),
                        'connected parent chain changed')
                else:
                    require(equivalent(actual['model'],expected['model']) and equivalent(actual.get('receipt'),expected.get('receipt')),'connected parent chain changed')
            require(summary['steps']==40 and summary['batch']==32 and summary['lr']==1e-5 and summary['warmup']==10 and summary['fresh_optimizer'] and summary['fresh_ema'],'stage settings changed')
            if arm.startswith('math-'):
                require(doc['hard_math']=='QUALIFIES','MATH run with dropped/pending third stage')
                require(summary['configuration']==(None if arm=='math-plain' else admitted),'sibling configuration changed between stages')
    for name,raw in files.items():
        if not name.startswith('scores/') or not name.endswith('/m2-identity.json'): continue
        meta=json.loads(raw)
        if has_assignment:require(meta['assignment']['family']=='core','UNVERIFIED: foreign core contrast family')
        require(has_assignment or setup is not None and meta['gpu_identity']==setup,'UNVERIFIED: mixed scoring host/environment/physical GPU')
        require(meta['protocol']==doc['protocol'] and meta['technical_synthetic'] is standin,'mixed scoring protocol/type')
        arm=meta['arm']; seed=meta['seed']; step=meta['step']
        if arm=='base': expected=doc['starts']['base']
        elif arm.startswith('standalone-'): expected=doc['comparators'][f'{arm}:{seed}']
        elif arm.startswith('finqa-'): expected=doc['starts'][f'{arm}:{seed}']
        elif arm=='legal-plain':expected=doc['starts'][f'finqa-plain:{seed}'] if step==0 else doc['c_exports'][f'{seed}:{step}']
        elif step==0: expected=incoming(arm,seed)
        else:
            receipt=f'runs/{arm}-s{seed}/export-{step}.json'
            require(receipt in files,'missing scored export receipt')
            expected=dict(identity=json.loads(files[receipt]))
        if not standin:
            require(meta['model']['model_file_hashes']==expected['identity']['model_file_hashes'],'score model differs from connected export')
    if not standin:
        require(has_assignment or setup is not None,'missing scoring setup')
        for name in files:
            if name.startswith('scores/') and name.endswith('/scores.csv'):
                require(name.rsplit('/',1)[0]+'/m2-identity.json' in files,'score missing connected identity')
