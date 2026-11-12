#!/usr/bin/env python3
"""TRACE plain allocation rows. The partner owns sbatch and dependencies."""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from kit.trace_common import *


def configured_inventory(cfg):
    if cfg.get('efficiency'):
        from kit.trace_train import efficiency_inventory as efficient
        value = efficient(cfg.get('studies'))
    else:
        value = inventory(cfg.get('studies'))
    if cfg.get('judge_throughput_probe'):
        rows=[]
        for row in value['rows']:
            if row['phase']=='score':
                gate='admit-'+row['id']
                rows.append(dict(row,id=gate,phase='judge-admit',gpus=0,cpus=1,mem_mb=256))
                row=dict(row,depends=[gate])
            rows.append(row)
            if row['phase']=='score':
                rows.append(dict(id='finalize-'+row['id'],phase='judge-finalize',gpus=0,cpus=1,mem_mb=256,
                                 study=row['study'],seed=row['seed'],depends=[],
                                 depends_on=[dict(id=row['id'],type='afterany'),
                                             dict(id=gate,type='afterany')]))
        value=dict(value,rows=rows)
    return value


def scientific_config(cfg):
    return {k:v for k,v in cfg.items() if k!='_config_path'}


def row_receipt(out, row_id):
    pointer = load(Path(out)/'state'/(row_id+'.json'))
    path = Path(out)/pointer['path']
    need(sha(path)==pointer['sha256'], 'completed allocation receipt changed')
    receipt = load(path)
    need(receipt['status']=='COMPLETE', 'allocation incomplete')
    for name,expected in receipt.get('artifacts_sha256',{}).items():
        need(sha(Path(out)/name)==expected,'completed row artifact changed')
    return receipt


def execute(cfg,row,attempt,standin):
    if row['phase']=='judge-finalize':
        from kit.trace_judge import finalize_probe
        scorer=f'score-{row["study"]}-{row["seed"]}'
        # A scheduler finalizer also works after SIGKILL or cancellation without
        # a scorer receipt. The scorer lock protects direct invocations/retries.
        with lock(Path(cfg['out'])/'locks'/(scorer+'.lock')):
            return finalize_probe(cfg,f'{row["study"]}-{row["seed"]}')
    if standin:
        from kit.trace_rehearsal import standin_row
        return standin_row(cfg,row,attempt)
    if row['phase']=='judge-admit':
        from kit.trace_judge import admit
        return admit(cfg,f'{row["study"]}-{row["seed"]}')
    if row['phase'] in ('score-prep','prep-stage','train-stage','export-stage','assemble'):
        from kit.trace_train import execute_efficiency as efficient
        return efficient(cfg,row)
    if row['phase']=='prepare':
        from kit.trace_prepare import prepare
        return prepare(cfg)
    if row['phase']=='targets':
        from kit.trace_targets import run
        return run(cfg)
    if row['phase']=='bind':
        from kit.trace_prepare import bind
        return bind(cfg)
    if row['phase']=='qualify':
        # Commands bind the installed stack's native numerical probes. They
        # execute inside this allocation and publish correctness evidence.
        ref = cfg['qualification'][row['study']]
        command = bound_json(ref)
        for item in command['files']:
            bound_file(item)
        subprocess.run(command['argv'],check=True,env=dict(os.environ,
                       TRACE_CONFIG=cfg['_config_path'],TRACE_STUDY=row['study'],
                       TRACE_QUALIFICATION_OUT=str(attempt/'qualification.json')))
        result = load(attempt/'qualification.json')
        required = ('identity','leakage','freeze_bit_equality','scorer_fixtures','licences')
        need(all(result.get(k) is True for k in required), 'qualification correctness unresolved')
        need(result.get('evidence_sha256'), 'qualification numerical evidence absent')
        return result
    if row['phase']=='train':
        from kit.trace_train import run
        return run(cfg,row['study'],row['seed'])
    if row['phase']=='score':
        from kit.trace_score import lane
        from kit.trace_jobs import timing_phase as phase
        with phase('score'):
            lane(cfg,row['study'],row['seed'],attempt.name)
        os.environ.pop('TRACE_TOKEN_CACHE', None)
        from kit.trace_judge import run as judge
        with phase('judge'):
            return judge(cfg,row['study'],row['seed'])
    from kit.trace_read import collect
    return collect(cfg)


@contextmanager
def probe_completion(cfg,row):
    """Persist the claimed scorer's decision across preflight and execution exits."""
    error=None
    try:
        yield
    except BaseException as failure:
        error=failure
        raise
    finally:
        if cfg.get('judge_throughput_probe') and row['phase']=='score':
            from kit.trace_judge import finalize_probe
            finalize_probe(cfg,f'{row["study"]}-{row["seed"]}',error)


def run(config_path,row_id,standin=False):
    offline()
    original = load(config_path)
    need(bool(original.get('technical_synthetic'))==standin, 'synthetic/production mode differs')
    cfg = dict(original,_config_path=str(Path(config_path).resolve()))
    out = Path(cfg['out'])
    rows = configured_inventory(cfg)['rows']
    row = next((r for r in rows if r['id']==row_id),None)
    need(row is not None, 'unknown TRACE allocation row')
    with lock(out/'locks'/(row_id+'.lock')),probe_completion(cfg,row):
        # Check all dependencies before taking a GPU or publishing an attempt.
        dependencies = row['depends']
        if row['phase'] == 'collect' and not standin:
            active = load(out/'prepared.json')['active_studies']
            dependencies = [dep for dep in dependencies if any(
                dep.startswith('score-'+study+'-') for study in active)]
        for dep in dependencies:
            receipt = row_receipt(out,dep)
            need(receipt['config_sha256']==digest(cfg) and receipt['technical_synthetic']==standin,
                 'dependency binding differs')
        if not standin and row['gpus']:
            prepared = load(out/'prepared.json')
            study = row.get('study')
            need(prepared.get('active_studies'), 'all requested studies PENDING')
            need(prepared['registration_sha256'] == sha(REG), 'TRACE registration changed')
            need(prepared['source_sha256'] == {p.name:sha(p) for p in KIT.glob('trace_*.py')},
                 'TRACE implementation changed after preparation')
            need(not study or study not in prepared.get('blocked_studies', {}),
                 'affected study licence/census PENDING')
            need(prepared['audit']['unresolved'] == 0, 'overlap adjudication PENDING')
        state_path = out/'state'/(row_id+'.json')
        if state_path.exists():
            receipt = row_receipt(out,row_id)
            need(receipt['config_sha256']==digest(cfg), 'completed row config changed')
            # A previous afterany run may have preceded the scorer's claim.
            # Finalizers must recheck under its lock on every submission.
            if row['phase'] != 'judge-finalize':
                return receipt
        attempt = out/'attempts'/row_id/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        started = time.time()
        from kit.trace_jobs import timing_finish as finish
        phase_file = attempt/'phase-times.jsonl'
        os.environ.update(V7H_PHASE_FILE=str(phase_file),V7H_PHASE_GPUS=str(0 if standin else row['gpus']))
        value = dict(row=row_id,config_sha256=digest(cfg),technical_synthetic=standin,
                     row_type=row['phase'],status='RUNNING',gpus=0 if standin else row['gpus'],started=started,
                     gpu_hours=0.,warnings=[])
        atomic(attempt/'receipt.json',value)
        try:
            if cfg.get('efficiency') and row['phase']=='score':
                os.environ['TRACE_TOKEN_CACHE'] = str(out/'token-cache')
            from contextlib import nullcontext
            with timing_phase('score' if row['phase']=='targets' else 'prep') if row['phase'] in (
                    'prepare','targets','bind','qualify','score-prep','prep-stage','assemble') else nullcontext():
                result = execute(cfg,row,attempt,standin)
            value['result'] = result
            scope = []
            if row['phase'] in ('prepare','targets','bind'):
                names = {'prepare':'prepared','targets':'targets','bind':'binding'}
                scope = [p for p in [out/(names[row['phase']]+'.json'),
                    out/'audit'/'manual-review.json'] if p.exists()]
            elif row['phase'] in ('train','prep-stage','train-stage','export-stage','assemble'):
                scope = list((out/'training'/f'{row["study"]}-{row["seed"]}').rglob('*.json'))
            elif row['phase']=='score-prep':
                scope = list((out/'token-cache').glob('*.json'))
            elif row['phase']=='qualify':
                scope = [p for p in attempt.glob('*.json') if p.name != 'receipt.json']
            elif row['phase']=='score':
                scope = [*(out/'scores'/f'{row["study"]}-{row["seed"]}').rglob('*.json'),
                         *(out/'judge'/f'{row["study"]}-{row["seed"]}').rglob('*.json')]
            elif row['phase']=='collect':
                scope = [*out.glob('return-*.txt'),*out.glob('full/return-*.txt'),
                         out/'return-numeric.json',out/'full/return-numeric.json']
            value['artifacts_sha256'] = {str(p.relative_to(out)):sha(p) for p in scope}
            value['status'] = 'PENDING' if isinstance(result, dict) and result.get('status') == 'PENDING' else 'COMPLETE'
        except FileNotFoundError as error:
            value.update(status='PENDING', error_type=type(error).__name__)
        except BaseException as error:
            value.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'ERROR',
                         error_type=type(error).__name__)
            raise
        finally:
            value['ended'] = time.time()
            # In an enclosing row wrapper these clocks include all native stages,
            # loads, exports, failures and idle periods. Queue time is excluded.
            value['gpu_hours'] = value['gpus']*(value['ended']-started)/3600
            value['phase_gpu_seconds'] = finish(phase_file,value['ended']-started,value['gpus'])
            previous = [load(p) for p in (out/'attempts').glob('*/*/receipt.json')
                        if p != attempt/'receipt.json']
            if value['ended']-started > 86400:
                value['warnings'].append('allocation elapsed time exceeded 24 hours, continuing')
            if isinstance(value.get('result'),dict):
                warnings = value['result'].get('warnings',[])
                if isinstance(warnings,list):
                    value['warnings'].extend(warnings)
            total = sum(r.get('gpu_hours',0) for r in previous)+value['gpu_hours']
            if total > cfg.get('estimated_ceiling_gpu_h',3500):
                value['warnings'].append('estimated GPU ceiling exceeded, continuing')
            if not standin and row['gpus']:
                value['warnings'].append('utilisation evidence is in allocation wrapper telemetry')
            atomic(attempt/'receipt.json',value)
        if value['status'] == 'PENDING':
            return value
        # Keep every attempt immutable; finalizers publish their latest check.
        writer = atomic if row['phase']=='judge-finalize' else seal
        writer(state_path,dict(path=str((attempt/'receipt.json').relative_to(out)),
                               sha256=sha(attempt/'receipt.json')))
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('inventory','price','run'))
    parser.add_argument('--config')
    parser.add_argument('--row')
    parser.add_argument('--standin',action='store_true')
    args = parser.parse_args()
    if args.action in ('inventory','price'):
        import json
        scope = load(args.config).get('studies') if args.config else None
        print(json.dumps((configured_inventory(load(args.config)) if args.config else inventory(scope))
                         if args.action=='inventory' else estimates(),indent=2))
    else:
        need(args.config and args.row,'config and row required')
        cfg = load(args.config)
        row = next(r for r in configured_inventory(cfg)['rows'] if r['id']==args.row)
        if not args.standin and row['gpus'] and os.environ.get('UTIL_WATCH_ACTIVE')!='1':
            from kit.util_watch import use_holder_slice
            cfg = load(args.config)
            telemetry = Path(cfg['out'])/'utilisation'/(args.row+'-'+uuid.uuid4().hex+'.jsonl')
            command = [sys.executable,'-I',str(KIT/'util_watch.py'),'--out',str(telemetry),
                       '--expect-gpus',str(row['gpus']),'--warn','70']
            if use_holder_slice(os.environ,row['gpus']):
                command.append('--holder-slice')
            command += ['--',sys.executable,'-I',str(KIT/'trace_jobs.py'),'run',
                        '--config',args.config,'--row',args.row]
            raise SystemExit(subprocess.call(command))
        try:
            receipt = run(args.config,args.row,args.standin)
        except FileNotFoundError:
            return 75
        return 75 if receipt['status'] == 'PENDING' else 0




from pathlib import Path
import fcntl
import json
import os
import time

PHASES=('train','score','judge','export','prep','idle-in-allocation')


@contextmanager
def timing_phase(name):
    started=time.monotonic()
    try:
        yield
    finally:
        path=os.environ.get('V7H_PHASE_FILE')
        if path:
            seconds=(time.monotonic()-started)*int(os.environ['V7H_PHASE_GPUS'])
            with open(path,'a') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX)
                stream.write(json.dumps([name,seconds])+'\n')
                fcntl.flock(stream,fcntl.LOCK_UN)


def timing_finish(path,elapsed,gpus):
    totals=dict.fromkeys(PHASES,0.)
    if Path(path).exists():
        for line in Path(path).read_text().split('\n'):
            if not line:
                continue
            name,seconds=json.loads(line)
            totals[name]+=seconds
    totals['idle-in-allocation']=max(0.,elapsed*gpus-sum(totals.values()))
    return totals


def timing_rows(root):
    from kit.nm_common import load
    totals={}
    for path in Path(root).glob('attempts/*/*/receipt.json'):
        receipt=load(path)
        if 'phase_gpu_seconds' not in receipt:
            continue
        kind=receipt.get('row_type',receipt['row'].split('-')[0])
        values=totals.setdefault(kind,dict.fromkeys(PHASES,0.))
        for name,seconds in receipt['phase_gpu_seconds'].items():
            values[name]+=seconds
    # Order is fixed and documented; integer seconds keep lines below 90 chars.
    return {'phase.'+kind:[round(values[p],3) for p in PHASES]
            for kind,values in totals.items()}


if __name__=='__main__':
    raise SystemExit(main())
