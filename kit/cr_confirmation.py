"""CR-G confirmation binding and reader; native B order regeneration only."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import os
import re
import subprocess
from kit.nm_common import atomic, load, sha, digest, need, ready, Pending, seal, KIT
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
TECHNICAL_STANDIN = ('CPU-STANDIN only: fixture tokenizer/model/export/scoring; skip baseline diff approval, '
    'eight-GPU preflight and training receipts; simulated nvidia-smi; TECHNICAL-ONLY output')
IMPLEMENTATION = ('cr_confirmation.py', 'cr_jobs.py', 'cr_text.py', 'cr_finqa.py', 'm3_mode.py',
                  'm3_schedule.py', 'm3_contract.py', 'm3_entry.py', 'm3_train.py',
                  'm3_score.py', 'm3_read.py', 'ref_score_job.py', 'nm_jobs.py',
                  'nm_text.py', 'nm_pipeline.py', 'nm_common.py', 'nm_execute.py', 'nm_budget.py', 'ref_guard.py')


def code_seal():
    from kit.m3_contract import CODE_FILES
    from kit.cr_finqa import B_CODE
    inherited=(*CODE_FILES,'m3_manifest.py','m3_pools.py','p3b_train.py','p3b_score.py',
        'p3b_common.py','hellaswag_reference.py','likelihood_validation.py','score_forgetting.py',
        'scorers.py','c_score.py','next_hellaswag.py','inference_lifecycle.py','lane_pool.py',
        'util_watch.py','eval_bed.py','lane_evidence.py','m2_assignment.py','m2_gpu.py',
        'v6_infra.py','resources/p3b/v4-recipes.json')
    return {name: sha(KIT/name) for name in sorted(set((*IMPLEMENTATION,*inherited,*B_CODE)))}


def isolated(cfg, dispatch_out=None):
    out = Path(cfg['out']).resolve()
    others = [cfg.get('v6a_root'), dispatch_out, *cfg.get('other_roots', [])]
    ready(cfg.get('v6a_root'), 'v6a realised root absent; owner escalation')
    for other in filter(None, others):
        path = Path(other).resolve()
        need(out != path and out not in path.parents and path not in out.parents,
             'confirmation requires separate non-nested output trees')
    return out


def verify_standin(cfg):
    """Synthetic bypasses may address only the reserved CPU fixture host."""
    if cfg.get('technical_synthetic'):
        need(cfg.get('technical_synthetic') is True and cfg.get('hosts') == ['CPU-STANDIN'],
             'technical_synthetic requires hosts exactly [CPU-STANDIN]; non-stand-in host refused')


def bind_inputs(cfg, attempt):
    """Publish expected/observed even on missing/hash/mode mismatch. Fail closed."""
    verify_standin(cfg)
    report = dict(schema='cr-input-binding.v1', baseline=BASELINE, seeds=list(SEEDS),
                  status='PENDING', escalation='owner', sources=[], errors=[])
    def blocked(reason): report['errors'].append(reason)
    ref = cfg.get('realised_inputs', {})
    path = Path(ref.get('path', '/MISSING-v6a-realised-inputs'))
    observed = sha(path) if path.is_file() else None
    report['inventory'] = dict(path=str(path), expected=ref.get('sha256'), observed=observed)
    if not ref.get('sha256') or observed != ref['sha256']:
        blocked('v6a node inventory absent/hash mismatch; never reconstruct')
    else:
        doc = load(path)
        if doc.get('kit_commit') != BASELINE: blocked('realised inventory must bind kit 059d6293')
        if doc.get('hard_math_mode') not in ('core-copy', 'm3-fallback'): blocked('recorded hard-MATH mode absent')
        sources = doc.get('sources', [])
        if not sources or not ROLES <= {r.get('role') for r in sources}: blocked('realised source inventory incomplete')
        for row in sources:
            source = Path(row['path']); actual = sha(source) if source.is_file() else None
            report['sources'].append(dict(row, expected=row.get('sha256'), observed=actual))
            if not re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')) or actual != row['sha256']:
                blocked('missing/hash mismatch: '+str(source))
            # Every manifest member must also be enumerated with its recorded
            # expected hash; adding a newly generated expected hash is forbidden.
            if actual == row.get('sha256') and row.get('manifest'):
                for name, expected in load(source).get('files', {}).items():
                    member=(source.parent/name).resolve()
                    if not any(Path(r['path']).resolve()==member and r.get('sha256')==expected for r in sources):
                        blocked('unbound realised manifest member: '+str(member))
        modes=[r for r in sources if r.get('role')=='math-mode']
        if len(modes)!=1 or not Path(modes[0]['path']).is_file(): blocked('recorded mode receipt missing')
        elif sha(Path(modes[0]['path'])) != modes[0].get('sha256'): blocked('recorded mode receipt hash mismatch')
        else:
            # Only a hash-bound receipt is read; an unreadable one is a mismatch.
            try: recorded = load(modes[0]['path']).get('path')
            except (ValueError, OSError, AttributeError): recorded = None
            if recorded != doc.get('hard_math_mode'): blocked('recorded hard-MATH mode mismatch')
        # Paths consumed by preparation must be exact entries in this inventory.
        for name in ('pools_manifest','base_binding','hellaswag_manifest', 'native_inventory', 'math_source'):
            if not doc.get(name) or not any(r['path']==doc[name] for r in sources): blocked('unbound '+name)
        tasks=doc.get('task_sources', {})
        if set(tasks)!={'finqa','contract_nli','math_hard'}: blocked('task schedule sources absent')
        for stage, task in tasks.items():
            for key in ('corpus','v6a_schedules'):
                paths=[task.get(key)] if key=='corpus' else list(task.get(key,{}).values())
                if key=='v6a_schedules' and set(task.get(key,{}))!={'101','102','103'}: blocked('v6a task schedule identity absent: '+stage)
                for item in paths:
                    if not any(r['path']==item for r in sources):blocked('unbound task source '+str(item))
            if stage == 'finqa':
                from kit.cr_finqa import PROTOCOL, native_paths
                if task.get('algorithm') != PROTOCOL: blocked('FinQA requires B native procedure; owner escalation')
                try:
                    for item in native_paths(task):
                        if not any(Path(r['path']).resolve()==item.resolve() for r in sources):
                            blocked('unbound FinQA native input: '+str(item))
                    if not cfg.get('technical_synthetic'):
                        tokenizer=Path(cfg.get('tokenizer') or '/MISSING-tokenizer')
                        members=[p for p in tokenizer.rglob('*') if p.is_file()]
                        if not members: blocked('FinQA pinned tokenizer missing')
                        for item in members:
                            if not any(Path(r['path']).resolve()==item.resolve() for r in sources):
                                blocked('unbound FinQA tokenizer member: '+str(item))
                except (KeyError, Pending) as error: blocked('FinQA native inputs absent: '+str(error)+'; owner escalation')
            elif task.get('algorithm')!='seed-shuffle-cycle-v1':blocked('schedule algorithm differs: '+stage)
        if modes and doc.get('math_source')!=modes[0]['path']:blocked('hard-MATH source differs from recorded mode receipt')
        report['hard_math_mode']=doc.get('hard_math_mode')
        report['realised']=doc
        if not report['errors']:
            from kit.cr_finqa import reproduce
            proof=reproduce(tasks['finqa'],cfg,attempt)
            report['finqa_reproduction']=proof
            report['errors'].extend(proof['errors'])
    if report['errors'] and 'finqa_reproduction' not in report:
        from kit.cr_finqa import PROTOCOL
        report['finqa_reproduction']=dict(protocol=PROTOCOL,status='PENDING',escalation='owner',
            checks=[],regenerated={},errors=['FinQA input binding: '+r+'; owner escalation' for r in report['errors']])
    if not report['errors']:
        report.update(status='COMPLETE', escalation=None, code=code_seal())
    atomic(attempt/'input-binding.json', report)
    ready(not report['errors'], '; '.join(report['errors'])+'; owner escalation')
    return report


def diff_receipt(out, *, base=BASELINE):
    """Enumerate every baseline hunk; unknown baseline/hunks never become approval."""
    root=KIT.parent
    result=dict(schema='cr-diff.v1', baseline=base, status='PENDING', escalation='owner', hunks=[],
                allowed_categories=sorted(ALLOWED), category_definitions={'technical-standin':TECHNICAL_STANDIN},
                code=code_seal())
    found=subprocess.run(['git','rev-parse','--verify',base+'^{commit}'],cwd=root,capture_output=True,text=True)
    comparison=base
    target='hunks'
    if found.returncode:
        result['reason']='baseline git object unavailable; diff review PENDING'
        # Reviewable local inventory is explicitly diagnostic. It cannot be
        # promoted to a receipt against the missing scientific baseline.
        comparison=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        result['diagnostic_base_commit']=comparison
        result['diagnostic_hunks']=[]
        target='diagnostic_hunks'
    else:
        result['baseline_commit']=found.stdout.strip()
        result['reason']='each hunk requires allowed category and pre-data review'
    import difflib
    for name in code_seal():
        file='kit/'+name
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
    ready(ref and Path(ref.get('path','')).is_file(), 'pre-data diff receipt absent; owner escalation')
    ready(sha(ref['path'])==ref.get('sha256'), 'diff receipt hash mismatch; owner escalation')
    doc=load(ref['path'])
    ready(doc.get('baseline')==BASELINE and doc.get('status')=='COMPLETE' and doc.get('review'), 'baseline diff review incomplete; owner escalation')
    ready(doc.get('code')==code_seal(), 'confirmation code changed since diff review; owner escalation')
    ready(doc.get('hunks') and all(h.get('category') in ALLOWED and h.get('sha256') for h in doc['hunks']), 'unapproved diff hunk; owner escalation')
    # Regenerate baseline hunks, so an incomplete approved list cannot hide code.
    import tempfile
    with tempfile.TemporaryDirectory(dir=KIT.parent/'tmp') as temporary:
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
