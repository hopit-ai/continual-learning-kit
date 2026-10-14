#!/usr/bin/env python3
"""Bind the closed phase-0 return to the main package; never reset its spend."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import tarfile

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

ARCHIVE_PATH='v4/report-inputs/phase0-return.tar.gz'
BINDING_PATH='v4/report-inputs/phase0-binding.json'
LEDGER_PATH='k8b4/containment/allocation-ledger.json'
CAMPAIGNS=Path(__file__).with_name('campaigns')


def digest(raw):return hashlib.sha256(raw).hexdigest()

def canonical(doc):return json.dumps(doc,sort_keys=True,separators=(',',':')).encode()


def charge_record(entry):
    """Immutable accounting identity; derived summaries/rows may be refreshed."""
    return {key:entry[key] for key in ('start','end','width','segments')}


def verified_source(path):
    """Only the real reader can admit source records; no summary is trusted."""
    from kit.v4_phase0 import read_archive
    from kit.v4_archive import Archive
    from kit.v4_budget import allocation_spend
    path=Path(path);raw=path.read_bytes();sha=digest(raw)
    report=read_archive(path)
    if report['status']!='technical pass':raise ValueError('phase0 return must pass the real phase-0 reader: '+str(report.get('reasons',[])))
    try:
        with Archive(path) as archive:ledger=archive.json(LEDGER_PATH)
    except (tarfile.TarError,EOFError) as exc:raise ValueError('phase0 return is unreadable: '+str(exc)) from exc
    records=ledger['allocations']
    if not records or ledger.get('phase0_import') or ledger.get('accounting_unknown') or any(not e.get('end') or e.get('accounting_unknown') for e in records.values()):
        raise ValueError('phase0 allocation records must be closed and known, without a prior import')
    spent,blocks=allocation_spend(ledger)
    if spent>=1000:
        raise ValueError('actual 1000 GPU-hour ceiling reached')
    if path.read_bytes()!=raw:raise ValueError('phase0 return changed during verification')
    binding=dict(schema='v4-phase0-binding.v1',archive_sha256=sha,ledger_sha256=digest(canonical(ledger)),
                 records_sha256=digest(canonical(records)),allocations=records)
    return binding,report


def accounting_allocations(ledger):
    """Union by original allocation ID, rejecting contradictory duplicate records."""
    local=ledger.get('allocations')
    if not isinstance(local,dict):raise ValueError('missing allocation ledger')
    imported=ledger.get('phase0_import')
    if imported is None:return local
    if not isinstance(imported,dict):raise ValueError('phase0 imported allocation binding is invalid')
    records=imported.get('allocations')
    if (imported.get('schema')!='v4-phase0-binding.v1' or not isinstance(records,dict) or not records
        or digest(canonical(records))!=imported.get('records_sha256')
        or any(not e.get('end') or e.get('accounting_unknown') for e in records.values())):
        raise ValueError('phase0 imported allocation records altered or not closed')
    merged=dict(records)
    for job,entry in local.items():
        if job in merged and charge_record(entry)!=charge_record(merged[job]):
            from kit.v4_finqa_policy import enabled,warn
            if not enabled():raise ValueError('conflicting phase0 allocation record: '+job)
            warn('allocation accounting','conflicting phase0 allocation record: '+job+'; immutable imported charge retained')
            continue
        merged[job]=entry
    return merged


def expected_binding(campaigns=None):
    """Both immutable manifests must bind the same return used by the C1 hook."""
    directory=Path(campaigns or CAMPAIGNS)
    graphs=[json.loads((directory/f'v4-{phase}.yaml').read_text()) for phase in ('qualification','main')]
    return campaign_binding(graphs)


def campaign_binding(graphs):
    sources=[g['v4'].get('phase0_accounting') for g in graphs]
    if not sources or not sources[0] or any(s!=sources[0] for s in sources):raise ValueError('phase0 return is missing or differs in the frozen campaign binding')
    for graph in graphs:
        if graph['v4']['phase']=='qualification' and graph['v4']['launch_allowances']['source'].get('archive_sha256')!=sources[0]['archive_sha256']:
            raise ValueError('phase0 accounting differs from the C1 launch archive')
    return sources[0]


def public_binding(binding):
    return {key:binding[key] for key in ('schema','archive_sha256','ledger_sha256','records_sha256')}


def verify_records(path,binding,expected):
    if digest(Path(path).read_bytes())!=binding.get('archive_sha256'):raise ValueError('phase0 returned archive SHA-256 binding differs')
    try:verified,_=verified_source(path)
    except (tarfile.TarError,EOFError) as exc:raise ValueError('phase0 return is unreadable: '+str(exc)) from exc
    if verified!=binding or public_binding(binding)!=expected:
        raise ValueError('phase0 return or records differ from the campaign binding')
    return verified


def bind(work,archive,*,campaigns=None):
    """One explicit import before CPU prepare/admission; never overwrite/rebind."""
    work=Path(work)
    destinations=[work/ARCHIVE_PATH,work/BINDING_PATH]
    ledger_path=work/LEDGER_PATH
    if any(p.exists() for p in destinations):raise ValueError('phase0 return already bound; double binding refused')
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(schema='kit-v4-allocation-ledger.v2',allocations={})
    if 'phase0_import' in ledger:raise ValueError('phase0 return already imported; double binding refused')
    if any(not e.get('end') for e in ledger['allocations'].values()):raise ValueError('close local allocations before phase0 binding')
    binding,_=verified_source(archive)
    verify_records(archive,binding,expected_binding(campaigns))
    ledger['phase0_import']=binding
    accounting_allocations(ledger) # Fail on conflicts before writing anything.
    ledger.update(block_limits={'qualification':1000,'scientific':1000},ceiling=1000)
    raw=Path(archive).read_bytes()
    if digest(raw)!=binding['archive_sha256']:raise ValueError('phase0 return changed before import')
    for path in destinations:path.parent.mkdir(parents=True,exist_ok=True)
    with destinations[0].open('xb') as handle:handle.write(raw)
    with destinations[1].open('xb') as handle:handle.write(canonical(binding)+b'\n')
    from kit.runner import write_durably
    write_durably(ledger_path,ledger)
    return binding


def verify_work(work,*,campaigns=None,warning_work=None):
    work=Path(work)
    try:
        binding=json.loads((work/BINDING_PATH).read_text())
        verify_records(work/ARCHIVE_PATH,binding,expected_binding(campaigns))
        ledger=json.loads((work/LEDGER_PATH).read_text())
        from kit.v4_finqa_policy import enabled, warn
        if ledger.get('phase0_import')!=binding:
            if not enabled():raise ValueError('phase0 import missing or differs from its bound return')
            warn('allocation accounting','live import differs; immutable verified phase0 records remain authoritative',warning_work or work)
            ledger={**ledger,'phase0_import':binding}
        try:accounting_allocations(ledger)
        except ValueError as exc:
            if not enabled():raise
            warn('allocation accounting',str(exc),warning_work or work)
        return ledger
    except (OSError,KeyError) as exc:raise ValueError('phase0 bound return is missing/unreadable: '+str(exc)) from exc


def archive_ledger(archive,phase):
    """Verify source bytes and real phase-0 evidence again on archive replay."""
    from kit.v4_budget import allocation_spend
    graphs=[archive.json('kit-snapshot/campaigns/v4-'+p+'.yaml') for p in (('qualification','main') if phase=='main' else ('qualification',))]
    expected=campaign_binding(graphs);binding=archive.json(BINDING_PATH)
    raw=archive.files.get(ARCHIVE_PATH)
    if raw is None or digest(raw)!=binding['archive_sha256']:raise ValueError('phase0 returned archive missing or SHA-256 binding differs')
    with tempfile.TemporaryDirectory(prefix='v4-phase0-bound-') as temp:
        path=Path(temp)/'phase0.tar.gz';path.write_bytes(raw);verify_records(path,binding,expected)
    ledger=archive.json(LEDGER_PATH)
    if ledger.get('phase0_import')!=binding:raise ValueError('phase0 archive ledger import differs from its binding')
    actual,blocks=allocation_spend(ledger)
    archive.phase0_verified_allocations=set(binding['allocations'])
    return ledger,dict(actual_gpu_hours=float(actual),blocks={k:float(v) for k,v in blocks.items()},
                       phase0_gpu_hours=float(allocation_spend({'allocations':binding['allocations']})[0]),
                       phase0_archive_sha256=binding['archive_sha256'],
                       remaining_qualification_gpu_hours=float(1000-actual),planned_qualification_remaining_gpu_hours=float(100-blocks.get('qualification',0)),remaining_ceiling_gpu_hours=float(1000-actual))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--archive',type=Path);group.add_argument('--verify',action='store_true')
    args=parser.parse_args(argv)
    try:result=verify_work(args.work) if args.verify else bind(args.work,args.archive)
    except (ValueError,OSError) as exc:parser.error(str(exc))
    print(json.dumps(public_binding(result['phase0_import'] if args.verify else result),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
