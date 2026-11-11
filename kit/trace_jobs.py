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
from kit.trace_common import *


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
    if standin:
        from kit.trace_rehearsal import standin_row
        return standin_row(cfg,row,attempt)
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
        lane(cfg,row['study'],row['seed'],attempt.name)
        from kit.trace_judge import run as judge
        return judge(cfg,row['study'],row['seed'])
    from kit.trace_read import collect
    return collect(cfg)


def run(config_path,row_id,standin=False):
    offline()
    original = load(config_path)
    need(bool(original.get('technical_synthetic'))==standin, 'synthetic/production mode differs')
    cfg = dict(original,_config_path=str(Path(config_path).resolve()))
    out = Path(cfg['out'])
    rows = inventory(cfg.get('studies'))['rows']
    row = next((r for r in rows if r['id']==row_id),None)
    need(row is not None, 'unknown TRACE allocation row')
    with lock(out/'locks'/(row_id+'.lock')):
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
            return receipt
        attempt = out/'attempts'/row_id/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        started = time.time()
        value = dict(row=row_id,config_sha256=digest(cfg),technical_synthetic=standin,
                     status='RUNNING',gpus=0 if standin else row['gpus'],started=started,
                     gpu_hours=0.,warnings=[])
        atomic(attempt/'receipt.json',value)
        try:
            result = execute(cfg,row,attempt,standin)
            value['result'] = result
            scope = []
            if row['phase'] in ('prepare','targets','bind'):
                names = {'prepare':'prepared','targets':'targets','bind':'binding'}
                scope = [p for p in [out/(names[row['phase']]+'.json'),
                    out/'audit'/'manual-review.json'] if p.exists()]
            elif row['phase']=='train':
                scope = list((out/'training'/f'{row["study"]}-{row["seed"]}').rglob('*.json'))
            elif row['phase']=='score':
                scope = [*(out/'scores'/f'{row["study"]}-{row["seed"]}').rglob('*.json'),
                         *(out/'judge'/f'{row["study"]}-{row["seed"]}').rglob('*.json')]
            elif row['phase']=='collect':
                scope = [*out.glob('return-*.txt'),*out.glob('full/return-*.txt'),
                         out/'return-numeric.json',out/'full/return-numeric.json']
            value['artifacts_sha256'] = {str(p.relative_to(out)):sha(p) for p in scope}
            value['status'] = 'PENDING' if isinstance(result, dict) and result.get('status') == 'PENDING' else 'COMPLETE'
        except BaseException as error:
            value.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'ERROR',
                         error_type=type(error).__name__)
            raise
        finally:
            value['ended'] = time.time()
            # In an enclosing row wrapper these clocks include all native stages,
            # loads, exports, failures and idle periods. Queue time is excluded.
            value['gpu_hours'] = value['gpus']*(value['ended']-started)/3600
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
        seal(state_path,dict(path=str((attempt/'receipt.json').relative_to(out)),
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
        print(json.dumps(inventory(scope) if args.action=='inventory' else estimates(),indent=2))
    else:
        need(args.config and args.row,'config and row required')
        cfg = load(args.config)
        row = next(r for r in inventory(cfg.get('studies'))['rows'] if r['id']==args.row)
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
        run(args.config,args.row,args.standin)


if __name__=='__main__':
    main()
