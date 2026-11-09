"""CR-G confirmation binding and reader; native B order regeneration only."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import os
import subprocess
from kit.nm_common import atomic, load, sha, digest, need, ready, Pending, Invalid, seal, KIT
from kit.m3_mode import CONFIRMATION_SEEDS as SEEDS, CONFIRMATION_ARMS as ARMS

BASELINE = '059d6293'
FINQA_PENDING = 'FinQA native reproduction proof missing; owner escalation'
# The node inventory is itself pinned by the operator, and must enumerate ALL
# realised inputs, not merely pointers to the original generation protocols.
ROLES = {'candidate-lists', 'raw-responses', 'accepted-responses', 'task-demonstrations',
         'exclusions', 'verifiers', 'arc-frozen', 'arc-acceptance', 'contract-raw-targets',
         'contract-journal', 'contract-replay', 'math-mode', 'math-targets',
         'schedule-sources', 'pools', 'panels', 'references', 'scorers', 'licences'}
ALLOWED = {'seed-mode', 'finqa-native-order', 'technical-standin', 'namespace', '3a', '3b', '3d', '3g', '3h', 'likelihood-K2', 'holder-slice'}
ALLOWED.update(('owner-v6-howchange', 'owner-v7c-fix'))
OWNER_V7C_FIX_FILES = ('kit/cr_confirmation.py', 'kit/cr_jobs.py', 'scripts/publish_kit.py')
OWNER_V7C_FIX_REVIEWER = 'Opus v7c review round 3'
OWNER_V7C_FIX_BINDING = ('Exact bytes are bound by the receipt code seal; the owner supplies '
    'the receipt hash in launch JSON diff_receipt.sha256.')
OWNER_HUNKS = 'resources/cr-confirmation/owner-approved-hunks.json'
OWNER_SHA256 = '356c1035dca50a0dfb79b728c0583c994c4e9a51622d6033b856d271188e14fd'
OWNER_COVERAGE = 'resources/cr-confirmation/owner-approved-169.json'
OWNER_COVERAGE_SHA256 = 'c8580c3aa36ffe7ba30cac3a18eee924a6514b262ef88b50b83310407399fd93'
OWNER_RESOURCES = ((OWNER_HUNKS, OWNER_SHA256), (OWNER_COVERAGE, OWNER_COVERAGE_SHA256))
TECHNICAL_STANDIN = ('CPU-STANDIN only: fixture tokenizer/model/export/scoring; skip baseline diff approval, '
    'eight-GPU preflight and training receipts; simulated nvidia-smi; TECHNICAL-ONLY output')
IMPLEMENTATION = ('cr_confirmation.py', 'cr_jobs.py', 'cr_text.py', 'cr_finqa.py', 'm3_mode.py',
                  'm3_schedule.py', 'm3_contract.py', 'm3_entry.py', 'm3_train.py',
                  'm3_score.py', 'm3_read.py', 'ref_score_job.py', 'nm_jobs.py',
                  'nm_text.py', 'nm_pipeline.py', 'nm_common.py', 'nm_execute.py', 'nm_budget.py', 'ref_guard.py')


def code_seal():
    from kit.cr_dependencies import FROZEN_DEPENDENCIES
    names = set(FROZEN_DEPENDENCIES) | {path for path, _ in OWNER_RESOURCES}
    return {name: sha(KIT/name) for name in sorted(names)}


def owner_hunks():
    approved = set()
    for path, expected in OWNER_RESOURCES:
        ready(sha(KIT/path) == expected, 'owner approval resource changed; owner escalation')
        doc = load(KIT/path)
        ready(doc.get('baseline') == BASELINE, 'owner approval baseline differs')
        approved.update((h['file'], h['sha256']) for h in doc['hunks'])
    return approved


def check_categories(hunks):
    approved = owner_hunks()
    return bool(hunks) and all(h.get('category') in ALLOWED and h.get('sha256') and
        (h['category'] != 'owner-v6-howchange' or (h.get('file'), h['sha256']) in approved) and
        (h['category'] != 'owner-v7c-fix' or h.get('file') in OWNER_V7C_FIX_FILES)
        for h in hunks)


def isolated(cfg, dispatch_out=None):
    out = Path(cfg['out']).resolve()
    others = [cfg.get('v6a_root'), cfg.get('v6b_root'), cfg.get('six_row_root'),
              dispatch_out, *cfg.get('other_roots', [])]
    ready(cfg.get('v6a_root'), 'v6a realised root absent; owner escalation')
    for other in filter(None, others):
        path = Path(other).resolve()
        need(out != path and out not in path.parents and path not in out.parents,
             'confirmation requires separate non-nested output trees')
    return out


def isolated_parent(cfg, dispatch_out):
    """The CR-only dispatch must also be disjoint from all inherited roots."""
    parent = Path(dispatch_out).resolve()
    for other in filter(None, [cfg.get('v6a_root'), cfg.get('v6b_root'),
                              cfg.get('six_row_root'), *cfg.get('other_roots', [])]):
        path = Path(other).resolve()
        need(parent != path and parent not in path.parents and path not in parent.parents,
             'CR parent requires separate non-nested output trees')


class HostPolicy(Invalid):
    """An explicit host-policy refusal remains Invalid, not an input retry."""


def verify_standin(cfg):
    """Synthetic bypasses may address only the reserved CPU fixture host."""
    if cfg.get('technical_synthetic'):
        if not (cfg.get('technical_synthetic') is True and cfg.get('hosts') == ['CPU-STANDIN']):
            raise HostPolicy('technical_synthetic requires hosts exactly [CPU-STANDIN]; non-stand-in host refused')


def _check_inputs(cfg, attempt, report, checked):
    from kit.cr_finqa import PROTOCOL, native_paths, reproduce, receipt_value
    errors = report['errors']
    cfg = receipt_value(cfg, errors, 'configuration')
    ready(isinstance(cfg, dict), 'configuration must be an object')
    verify_standin(cfg)
    ref = cfg.get('realised_inputs', {})
    ready(isinstance(ref, dict), 'realised_inputs must be an object')
    report['inventory'].update(path=ref.get('path'), expected=ref.get('sha256'))
    path = checked.access('path', ref.get('path'), 'v6a inventory')
    observed = checked.access('sha', path, 'v6a inventory') if path is not None else None
    report['inventory']['observed'] = observed
    if not checked.valid_digest(ref.get('sha256')) or observed is None or observed != ref['sha256']:
        errors.append('v6a node inventory absent/hash mismatch; never reconstruct')
        return
    doc = checked.document(path, ref['sha256'], 'v6a inventory')
    if doc is None: return
    report['realised'] = doc
    report['hard_math_mode'] = doc.get('hard_math_mode')
    if doc.get('kit_commit') != BASELINE: errors.append('realised inventory must bind kit 059d6293')
    if doc.get('hard_math_mode') not in ('core-copy', 'm3-fallback'): errors.append('recorded hard-MATH mode absent')
    sources = doc.get('sources')
    ready(isinstance(sources, list), 'realised sources must be an array')
    roles = set()
    verified = {}
    paths = {}
    valid_rows = []
    members = []
    for index, row in enumerate(sources):
        label = 'source '+str(index)
        record = dict(expected=None, observed=None)
        report['sources'].append(record)
        if not isinstance(row, dict):
            errors.append(label+' must be an object')
            continue
        record.update(row, expected=row.get('sha256'), observed=None)
        valid_rows.append(row)
        if isinstance(row.get('role'), str): roles.add(row['role'])
        elif 'role' in row: errors.append(label+' role must be a string when present')
        source = checked.access('path', row.get('path'), label)
        if source is None: continue
        actual = checked.access('sha', source, label)
        record['observed'] = actual
        resolved = checked.access('resolve', source, label)
        if resolved is not None: paths[str(resolved)] = row.get('sha256')
        valid = checked.valid_digest(row.get('sha256')) and actual is not None and actual == row['sha256']
        if not valid:
            errors.append('missing/hash mismatch: '+str(source))
        elif resolved is not None:
            key = str(resolved)
            if key in verified and verified[key] != actual:
                errors.append('conflicting realised source: '+key)
            verified[key] = actual
        if 'manifest' in row and type(row['manifest']) is not bool:
            errors.append(label+' manifest flag must be boolean')
        if valid and row.get('manifest') is True:
            manifest = checked.document(source, row['sha256'], label+' manifest')
            if manifest is None: continue
            files = manifest.get('files', {})
            if not isinstance(files, dict):
                errors.append(label+' manifest files must be an object')
                continue
            for name, expected in files.items():
                if not name or not checked.valid_digest(expected):
                    errors.append(label+' manifest member name/digest invalid')
                    continue
                member = checked.access('resolve', source.parent/name, label+' manifest member')
                # Membership is checked after every source row has been inspected.
                if member is not None:
                    members.append((str(member), expected))
    if not sources or not ROLES <= roles: errors.append('realised source inventory incomplete')
    for member, expected in members:
        if paths.get(member) != expected or verified.get(member) != expected:
            errors.append('unbound realised manifest member: '+member)
    modes = [r for r in valid_rows if r.get('role') == 'math-mode']
    if len(modes) != 1:
        errors.append('recorded mode receipt missing')
    else:
        mode = modes[0]
        resolved = checked.access('resolve', mode.get('path'), 'recorded mode receipt')
        expected = mode.get('sha256')
        if resolved is None or not checked.valid_digest(expected) or verified.get(str(resolved)) != expected:
            errors.append('recorded mode receipt hash mismatch')
        else:
            receipt = checked.document(mode['path'], expected, 'recorded mode receipt')
            if receipt is None or receipt.get('path') != doc.get('hard_math_mode'):
                errors.append('recorded hard-MATH mode mismatch')
        if doc.get('math_source') != mode.get('path'):
            errors.append('hard-MATH source differs from recorded mode receipt')
    bound_paths = {r['path'] for r in valid_rows if isinstance(r.get('path'), str)}
    for name in ('pools_manifest', 'base_binding', 'hellaswag_manifest', 'native_inventory', 'math_source'):
        value = doc.get(name)
        if not isinstance(value, str) or value not in bound_paths: errors.append('unbound '+name)
    tasks = doc.get('task_sources')
    ready(isinstance(tasks, dict), 'task_sources must be an object')
    if set(tasks) != {'finqa', 'contract_nli', 'math_hard'}: errors.append('task schedule sources absent')
    for stage, task in tasks.items():
        if not isinstance(task, dict):
            errors.append('task source must be an object: '+stage)
            continue
        corpus = task.get('corpus')
        if not isinstance(corpus, str) or corpus not in bound_paths: errors.append('unbound task corpus: '+stage)
        schedules = task.get('v6a_schedules')
        if not isinstance(schedules, dict):
            errors.append('v6a_schedules must be an object: '+stage)
        else:
            if set(schedules) != {'101', '102', '103'}: errors.append('v6a task schedule identity absent: '+stage)
            for item in schedules.values():
                if not isinstance(item, str) or item not in bound_paths: errors.append('unbound task schedule: '+stage)
        if stage == 'finqa':
            if task.get('algorithm') != PROTOCOL: errors.append('FinQA requires B native procedure; owner escalation')
            try:
                for item in native_paths(task):
                    resolved = checked.access('resolve', item, 'FinQA native input')
                    if resolved is not None and str(resolved) not in verified:
                        errors.append('unbound FinQA native input: '+str(item))
            except Pending as error:
                errors.append('FinQA native inputs absent: '+str(error)+'; owner escalation')
            if not cfg.get('technical_synthetic'):
                members = checked.access('members', cfg.get('tokenizer'), 'FinQA tokenizer')
                if not members: errors.append('FinQA pinned tokenizer missing')
                for item in members or []:
                    resolved = checked.access('resolve', item, 'FinQA tokenizer member')
                    if resolved is not None and str(resolved) not in verified:
                        errors.append('unbound FinQA tokenizer member: '+str(item))
        elif task.get('algorithm') != 'seed-shuffle-cycle-v1': errors.append('schedule algorithm differs: '+stage)
    if not errors:
        proof = reproduce(tasks['finqa'], cfg, attempt, verified_inventory=verified)
        report['finqa_reproduction'] = proof
        errors.extend(proof['errors'])


def binding_report():
    return dict(schema='cr-input-binding.v1', baseline=BASELINE, seeds=list(SEEDS),
                status='PENDING', escalation='owner', inventory=dict(path=None, expected=None, observed=None),
                sources=[], errors=[])


def input_failure(attempt, error, *, observed=None):
    """Settings failures occur before inventory binding can start."""
    from kit.cr_finqa import receipt_value
    report = binding_report()
    report.update(expected='valid CR bind-row configuration', observed=observed)
    report['errors'].append(type(error).__name__+': '+str(error))
    errors=[]
    report=receipt_value(report, errors, 'input binding')
    report['errors'].extend(errors)
    atomic(Path(attempt)/'input-binding.json', report)
    return report


def bind_inputs(cfg, attempt, *, validate=None):
    """Always publish input failures as owner-escalated PENDING with diagnostics."""
    from kit.cr_finqa import PROTOCOL, InputChecks, receipt_value
    report = binding_report()
    refusal = None
    try:
        _check_inputs(cfg, attempt, report, InputChecks(report['errors']))
        if not report['errors']:
            report.update(code=code_seal())
            if validate is not None:
                validate(report, InputChecks(report['errors']))
    except HostPolicy as error:
        report['errors'].append(str(error))
        if validate is not None: refusal = error
    except (Pending, ValueError) as error:
        report['errors'].append(str(error))
    except Exception as error:
        # Last-resort input boundary. Expected malformed cases are validated above.
        report['errors'].append('input checking failed: '+type(error).__name__+' '+str(error))
    sanitization_errors = []
    report = receipt_value(report, sanitization_errors, 'input binding')
    report['errors'].extend(sanitization_errors)
    if report['errors'] and 'finqa_reproduction' not in report:
        report['finqa_reproduction'] = dict(protocol=PROTOCOL, status='PENDING', escalation='owner',
            checks=[], regenerated={}, errors=['FinQA input binding: '+r+'; owner escalation' for r in report['errors']])
    if not report['errors']:
        report.update(status='COMPLETE', escalation=None)
    # Output errors are deliberately outside the input boundary. atomic uses
    # allow_nan=False, after receipt_value has sanitised all operator content.
    atomic(Path(attempt)/'input-binding.json', report)
    if refusal is not None: raise refusal
    ready(not report['errors'], '; '.join(report['errors'])+'; owner escalation')
    return report


def diff_receipt(out, *, base=BASELINE):
    """Enumerate every baseline hunk; unknown baseline/hunks never become approval."""
    root=KIT.parent
    result=dict(schema='cr-diff.v1', baseline=base, status='PENDING', escalation='owner', hunks=[],
                allowed_categories=sorted(ALLOWED), category_definitions={'technical-standin':TECHNICAL_STANDIN,
                    'owner-v7c-fix':dict(files=list(OWNER_V7C_FIX_FILES), reviewer=OWNER_V7C_FIX_REVIEWER,
                        binding=OWNER_V7C_FIX_BINDING, quote=None, date=None, hunks=[]),
                    'owner-v6-howchange':dict(resources=[dict(path='kit/'+p,sha256=h) for p,h in OWNER_RESOURCES],
                        scope='exact file and hunk sha256 pairs only; owner-approved coverage closure')},
                code=code_seal())
    try:
        found=subprocess.run(['git','rev-parse','--verify',base+'^{commit}'],cwd=root,capture_output=True,text=True)
    except OSError as error:
        result['reason']='git unavailable: '+str(error)+'; diff review PENDING'
        atomic(out,result)
        return result
    comparison=base
    target='hunks'
    if found.returncode:
        result['reason']='baseline git object unavailable; diff review PENDING'
        # Reviewable local inventory is explicitly diagnostic. It cannot be
        # promoted to a receipt against the missing scientific baseline.
        try:
            head=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
        except OSError as error:
            result['reason']+='; diagnostic HEAD unavailable: '+str(error)
            atomic(out,result)
            return result
        if head.returncode or not head.stdout.strip():
            result['reason']+='; diagnostic HEAD unavailable'
            atomic(out,result)
            return result
        comparison=head.stdout.strip()
        result['diagnostic_base_commit']=comparison
        result['diagnostic_hunks']=[]
        target='diagnostic_hunks'
    else:
        result['baseline_commit']=found.stdout.strip()
        result['reason']='each hunk requires allowed category and pre-data review'
    import difflib
    for name in code_seal():
        file='kit/'+name
        file=os.path.normpath(file)
        previous=subprocess.run(['git','show',comparison+':'+file],cwd=root,capture_output=True)
        before=previous.stdout.decode().splitlines(True) if previous.returncode==0 else []
        suggested=None
        if found.returncode==0 and previous.returncode==0 and name in IMPLEMENTATION:
            suggested='namespace' if name.startswith(('nm_','cr_text')) else 'seed-mode'
        patch=list(difflib.unified_diff(before,(KIT/name).read_text().splitlines(True),fromfile=comparison+'/'+file,tofile=file))
        hunks=[]
        for line in patch:
            if line.startswith('@@'):hunks.append([])
            if hunks:hunks[-1].append(line)
        for hunk in hunks:
            category=suggested
            if category is not None and name.startswith('cr_') and any(
                    token in ''.join(hunk) for token in ('technical_synthetic','verify_standin','TECHNICAL_STANDIN')):
                category='technical-standin'
            result[target].append(dict(file=file,header=hunk[0].strip(),sha256=digest(hunk),
                category=None,suggested_category=category,
                patch=''.join(hunk)))
    atomic(out,result);return result


def verify_diff(ref):
    ready(isinstance(ref,dict), 'diff_receipt must be an object; owner escalation')
    ready(isinstance(ref.get('path'),str) and bool(ref['path']), 'diff receipt path required; owner escalation')
    ready(ref and Path(ref.get('path','')).is_file(), 'pre-data diff receipt absent; owner escalation')
    ready(sha(ref['path'])==ref.get('sha256'), 'diff receipt hash mismatch; owner escalation')
    doc=load(ref['path'])
    ready(isinstance(doc,dict), 'diff receipt must contain a JSON object; owner escalation')
    ready(doc.get('baseline')==BASELINE and doc.get('status')=='COMPLETE' and doc.get('review'), 'baseline diff review incomplete; owner escalation')
    ready(doc.get('code')==code_seal(), 'confirmation code changed since diff review; owner escalation')
    ready(check_categories(doc.get('hunks', [])), 'unapproved diff hunk; owner escalation')
    fixes = [{k:h[k] for k in ('file', 'header', 'sha256')}
             for h in doc['hunks'] if h['category'] == 'owner-v7c-fix']
    fixes.sort(key=lambda h: h['file'])
    if fixes:
        definition = doc.get('category_definitions', {}).get('owner-v7c-fix', {})
        ready(definition.get('files') == list(OWNER_V7C_FIX_FILES) and
              definition.get('reviewer') == OWNER_V7C_FIX_REVIEWER and
              definition.get('binding') == OWNER_V7C_FIX_BINDING and
              all(isinstance(definition.get(k), str) and definition[k].strip() for k in ('quote', 'date')) and
              definition.get('hunks') == fixes and {h['file'] for h in fixes} == set(OWNER_V7C_FIX_FILES),
              'owner-v7c-fix approval definition incomplete; owner escalation')
    # Regenerate baseline hunks, so an incomplete approved list cannot hide code.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='cr-diff-') as temporary:
        actual=diff_receipt(Path(temporary)/'diff.json')
    ready(actual.get('baseline_commit')==doc.get('baseline_commit') and
          [(h['file'],h['sha256']) for h in actual['hunks']]==[(h['file'],h['sha256']) for h in doc['hunks']],
          'baseline unavailable or diff hunk inventory changed; owner escalation')
    return doc


def task_binding(out, stage, root):
    path=Path(out)/'confirmation-binding.json';ready(path.is_file(),'confirmation input binding absent')
    doc=load(path);source=doc['realised']['task_sources'][stage]
    need(Path(root).resolve()==(Path(out)/('targets-'+stage)).resolve(),'foreign confirmation task schedule root')
    if stage == 'finqa':
        try:
            from kit.cr_finqa import PROTOCOL, FRESH, native_paths, B_CODE
            proof=doc.get('finqa_reproduction',{})
            paths=native_paths(source)
            for item in paths:
                matches=[r for r in doc['sources'] if Path(r['path']).resolve()==item.resolve()]
                ready(len(matches)==1 and item.is_file() and sha(item)==matches[0]['sha256'],
                      'FinQA native source changed/unbound; owner escalation')
            ready(proof.get('code')=={name:sha(KIT/name) for name in B_CODE},'FinQA native code changed; owner escalation')
            ready(proof.get('status')=='COMPLETE' and not proof.get('errors') and proof.get('protocol')==PROTOCOL, FINQA_PENDING)
            manifest=load(Path(root)/'manifest.json')
            ready(manifest.get('schedule_protocol')==PROTOCOL, FINQA_PENDING)
            ready(load(Path(root)/'receipt.json')==proof,'FinQA reproduction receipt changed; owner escalation')
            for seed in FRESH:
                name=f'schedule-{seed}.jsonl';expected=proof.get('regenerated',{}).get(str(seed),{}).get('schedule_sha256')
                ready(expected and manifest.get('files',{}).get(name)==expected and sha(Path(root)/name)==expected,
                      'FinQA regenerated schedule mismatch; owner escalation')
        except (Pending,OSError,KeyError,ValueError) as error:
            proof=dict(doc.get('finqa_reproduction',{}),status='PENDING',escalation='owner',
                errors=['FinQA task binding: '+str(error)+'; owner escalation'])
            atomic(Path(out)/'finqa-task-check.json',proof)
            raise Pending(proof['errors'][0]) from error
        atomic(Path(out)/'finqa-task-check.json',proof)
    return dict(confirmation_binding=sha(path), source=source, manifest_sha256=sha(Path(root)/'manifest.json'))


def counts_from_cells(cells):
    result={}
    for (arm,seed,stage,step,panel), cell in cells.items():
        if arm=='seed':continue
        for metric, items in cell.get('items',{}).items():
            if items: result['.'.join(map(str,(arm,seed,stage,step,metric)))]=[len(items),sum(items.values())]
    return result


def analyse(cells, assignment, *, require_curves=True, **kwargs):
    from kit import m3_read
    need(os.environ.get('KIT_M3_MODE')=='cr-confirmation','isolated confirmation mode required')
    cells={k:(dict(v,identity_error='UNVERIFIED-HARDWARE: '+str(v['hardware_error'])) if v.get('hardware_error') else v) for k,v in cells.items()}
    result=m3_read.analyse(cells,assignment,**kwargs)
    result['schema']='cr-confirmation-read.v1'
    result['status']=result['contrasts']['CR-G']['label']
    if result['identity_failures'] and result['status'] not in ('UNVERIFIED','UNVERIFIED-HARDWARE'):
        result['status']='UNVERIFIED-HARDWARE' if any('UNVERIFIED-HARDWARE' in f['reason'] for f in result['identity_failures']) else 'UNVERIFIED'
        result['contrasts']['CR-G'].update(label=result['status'],reason=result['identity_failures'])
    if result['status']=='PENDING':result['status']='PENDING/INCOMPLETE'
    if require_curves and result['status'] not in ('UNVERIFIED','UNVERIFIED-HARDWARE'):
        missing=sorted(required_cells()-set(cells))
        if missing:
            result['status']='PENDING/INCOMPLETE'
            result['missing_cells']=[list(k) for k in missing]
            result['contrasts']['CR-G']['label']='PENDING'
            result['decisions']['CR']='PENDING'
    result['correct_counts']=counts_from_cells(cells)
    result['incumbent_changed']=False
    result['future_precedence']='v8 only if admitted; other six v7 rows retain I0'
    c=result['contrasts']['CR-G']
    result['decisions']={'CR':c['label']}
    result['ifeval']={stage:c.get('guards',{}).get(stage+'/final/ifeval',dict(status='PENDING'))
                      for stage in m3_read.STAGES}
    return result


def from_counts(counts):
    """Re-derive the scientific endpoint from integer correct/total only."""
    cells={};assignment={}
    for name,value in counts.items():
        arm,seed,stage,step,metric=name.split('.');seed=int(seed);step=int(step)
        need(type(value) is list and len(value)==2 and all(type(v) is int for v in value) and 0<=value[1]<=value[0] and value[0]>0,'invalid CR correct-counts')
        from kit.m3_read import PANEL_OF
        panel=PANEL_OF[metric];key=(arm,seed,stage,step,panel)
        native=dict(uuid='count-native',environment='integer-return',allocation_job='native-'+str(seed))
        reference=dict(uuid='count-reference',environment='integer-return',allocation_job='reference-'+str(seed))
        identity=reference if metric in ('knowledge','hellaswag') else native
        cell=cells.setdefault(key,dict(metrics={},items={},**identity))
        if metric in ('knowledge','hellaswag'):cell.update(method='REFERENCE-SCORED',reference_assignment=reference)
        cell['metrics'][metric]=100*value[1]/value[0]
        assignment[f'm3|{seed}']=native
    from kit import m3_read
    if any(k[1] not in SEEDS or k[0] not in ('base',*ARMS) for k in cells):
        contrast=dict(label='UNVERIFIED',reason='foreign seed/arm contamination')
    else:
        table=m3_read.Table(cells)
        contrast=m3_read.contrast(table,'CR-G',m3_read.identity_failures(table,assignment),())
    return dict(status='PENDING/INCOMPLETE' if contrast['label']=='PENDING' else contrast['label'],
        contrasts={'CR-G':contrast},seeds=list(SEEDS),incumbent_changed=False,
        ifeval={stage:contrast.get('guards',{}).get(stage+'/final/ifeval',dict(status='PENDING')) for stage in m3_read.STAGES})


def required_cells():
    from kit.m3_read import STAGES, TASK_METRIC
    result=set()
    generated=('finqa','test','math_hard','general-v1')
    for seed in SEEDS:
        for panel in (*generated,'hellaswag','knowledge-ll'):result.add(('base',seed,'finqa',0,panel))
        for arm in ARMS:
            for n,stage in enumerate(STAGES):
                layout={0:generated,10:(TASK_METRIC[stage],),20:(TASK_METRIC[stage],*[TASK_METRIC[e] for e in STAGES[:n]]),30:(TASK_METRIC[stage],),40:generated}
                for step,panels in layout.items():
                    for panel in (*panels,*(('hellaswag','knowledge-ll') if step in (0,40) else ())):result.add((arm,seed,stage,step,panel))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diff-receipt',type=Path,required=True)
    args=parser.parse_args()
    print(diff_receipt(args.diff_receipt)['status'])
