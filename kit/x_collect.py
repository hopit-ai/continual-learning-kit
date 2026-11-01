#!/usr/bin/env python3
"""Separate compact X return, including partial failures; no core dependency or writer."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *
import io
import tarfile
import gzip
import csv

LIMIT = 28*1024*1024


def load(path, standin=False):
    require(Path(path).stat().st_size < LIMIT, 'compressed archive exceeds limit')
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        require(sum(m.size for m in members) < 256*1024*1024, 'archive expands beyond safety limit')
        require(all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members), 'invalid archive member')
        require(len({m.name for m in members}) == len(members), 'duplicate archive member')
        files = {m.name: archive.extractfile(m).read() for m in members}
    manifest = json.loads(files.pop('addon/manifest.json'))
    require(manifest['technical_synthetic'] is standin, 'technical_synthetic archive requires --standin')
    require(set(files) == set(manifest['files']), 'archive inventory mismatch')
    for name, raw in files.items():
        require(hashlib.sha256(raw).hexdigest() == manifest['files'][name], 'archive hash mismatch')
    return {name.removeprefix('addon/'): raw for name, raw in files.items()}


def job(a):
    # Collection is allowed after failed/unfinished rows too. Readiness is the
    # reader's responsibility; the durable trap remains available even pre-X1.
    document = read(a.out/'binding.json') if (a.out/'binding.json').exists() else None
    require(document is None or document['technical_synthetic'] is a.standin, 'collection scope mismatch')
    from kit.m2_assignment import validate_files
    checked={str(p.relative_to(a.out)):p.read_bytes() for p in a.out.rglob('*') if p.is_file() and (p.suffix in ('.json','.csv') or p == a.out/'hellaswag/panel.jsonl') and not p.is_symlink()}
    from kit.ref_guard import native_files
    validate_files(native_files(checked),require_assignment=not a.standin)
    evidence, bulk, costs, score_rows = {}, [], [], []
    def tier2_error(relative,error):
        evidence['addon/tier2/collection-errors/'+relative.replace('/','_')+'.json']=json.dumps(
            dict(status='INCONCLUSIVE',path=relative,reason=str(error)),sort_keys=True).encode()
    boundary_seconds=[]
    compact = {'scoring-assignment.json','binding.json','support.json','audit.json','pricing.json','inherited.json','comparison-tables.json','m2-scoring-machine.json','scoring-core-match.json','p2/calibration.json'}
    compact.add('ref-scoring-assignment.json')
    compact.add('hellaswag/panel.jsonl')
    prefixes = ('node-preflight/','execution/','scoring-core-match/','dispositions/','signals/','failures/','tier2/','reference-scores/')
    for path in sorted(a.out.rglob('*')):
        if not path.is_file() or path.is_symlink() or path == a.archive or path.name in ('.lock','collect-complete.json') or path.name.endswith('.tmp'): continue
        relative = str(path.relative_to(a.out))
        if relative.startswith('return/'): continue
        second=relative.startswith(('tier2/','runs/x-t2-','scores/x-t2-'))
        if (path.name == 'cost.json' or '/training-costs/' in relative or relative.startswith(('signals/','tier2/signals/')) or
                relative.startswith('probes/') and (path.name.startswith('receipt') or path.name=='training-cost.json')) and path.suffix == '.json':
            try:
                value = read(path)
                if 'gpu_hours' in value:
                    costs.append(dict(path=relative,gpu_hours=value['gpu_hours'],seconds=value.get('seconds')))
            except (ValueError,KeyError,TypeError) as error:
                if not second:raise
                tier2_error(relative,error)
        is_weight = path.suffix in ('.pt','.bin','.safetensors','.distcp') or any(p.startswith('hf-') for p in path.relative_to(a.out).parts)
        if is_weight:
            bulk.append(dict(path=relative, bytes=path.stat().st_size, identity='see checkpoint export receipt; durable state manifest hashes all bytes'))
            continue
        include = (relative.startswith('references/') and path.name in ('rows.json','binding.json','scoring-assignment.json','scores.csv','identity.json','m2-identity.json','assignment-identity.json','manifest.json','cost.json','generation-summary.json')) or (relative.startswith('p2/') and path.suffix=='.json') or relative in compact or relative.startswith(prefixes) or (
            relative.startswith(('own/','probes/','runs/','replacement/')) and
            ((path.name.startswith('receipt') and path.suffix=='.json') or path.name in ('inputs.json','identity.json','summary.json','manifest.json','receipt.json','complete.json','prefix.json','restore-qualified.json','saved-state-step10.json','utility-frozen.json','utility-eligibility.json','cost.json') or
             path.name.startswith(('export-','p2-rank','control-first-r','slots-','resource-r')) or path.name=='cached-first-batch.json' or path.name == 'export.json' or '/segments/' in relative or '/boundaries/' in relative or '/training-costs/' in relative or '/ema-origin/' in relative)) or (
            relative.startswith('scores/') and path.name in ('scores.csv','alias.json','m2-identity.json','manifest.json','cost.json','generation-summary.json','assignment-identity.json'))
        if path.name.startswith('p2-rank') and path.suffix=='.json' and '/partials/' not in relative:
            try:
                training=read(path)
                if not training.get('technical_synthetic',document and document.get('technical_synthetic',False)):
                    boundary_seconds.extend(r['boundary_seconds'] for r in training['updates'] if 'boundary_seconds' in r)
            except (ValueError,KeyError,TypeError,AttributeError) as error:
                if not second:raise
                tier2_error(relative,error)
        raw = path.read_bytes()
        if relative.startswith('scores/') and path.name == 'scores.csv':
            try:
                if second:m.frozen_files(path.parent)
                with path.open() as f: own=list(csv.DictReader(f))
                if second:
                    expected=path.parent.name.rsplit('-s',1)[0]
                    require(all(r['arm']==expected for r in own),'tier-2 CSV arm differs from its directory')
                    from kit.c_score import COLUMNS
                    require(all(set(r)==set(COLUMNS) for r in own),'unregistered tier-2 score columns')
                else:
                    require(all(not r['arm'].startswith('x-t2-') for r in own),'tier-1 CSV cannot claim a tier-2 arm')
                score_rows.extend(own)
                # Literal alias sources remain available to verify their hashes on readback.
                if (a.out/'scoring-assignment.json').exists():evidence['addon/'+relative]=raw
            except (ValueError,OSError,KeyError,TypeError) as error:
                if not second:raise
                tier2_error(relative,error)
        elif relative == 'inherited.json':
            if (a.out/'scoring-assignment.json').exists():
                families={}
                for reference in sorted((a.out/'references').glob('*/rows.json')):
                    value=read(reference);families.setdefault(value['assignment']['family'],[]).extend(value['rows'])
                raw=json.dumps(dict(rows=families),sort_keys=True).encode()
            evidence['addon/inherited.json.gz'] = gzip.compress(raw,mtime=0)
        elif include: evidence['addon/'+relative] = raw
        else: bulk.append(dict(path=relative, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    if score_rows:
        from kit.c_score import COLUMNS
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=COLUMNS)
        writer.writeheader();writer.writerows(score_rows)
        evidence['addon/scores.csv.gz']=gzip.compress(buffer.getvalue().encode(),mtime=0)
    else: evidence['addon/scores.csv.gz']=gzip.compress(b'',mtime=0)
    evidence['addon/bulk.json'] = json.dumps(bulk, sort_keys=True).encode()
    evidence['addon/costs.json'] = json.dumps(dict(records=costs,
        regulariser_gpu_hours=sum(boundary_seconds)/3600 if boundary_seconds else None,
        recorded_gpu_hours=sum(v['gpu_hours'] for v in costs if v['gpu_hours'] is not None),
        complete=False,note='Recorded process costs include failures. Missing replacement/allocation/idle timings are not zero; reconcile charged allocation separately.'),sort_keys=True).encode()
    evidence['addon/manifest.json'] = json.dumps(dict(files={name: hashlib.sha256(raw).hexdigest() for name,raw in evidence.items()},
        technical_synthetic=a.standin), sort_keys=True).encode()
    source = digest({k: hashlib.sha256(v).hexdigest() for k,v in evidence.items()})
    marker = a.out/'collect-complete.json'
    if marker.exists() and read(marker)['source'] == source and a.archive.exists():
        require(sha(a.archive) == read(marker)['archive_sha256'], 'collected archive changed')
        print('SKIP: X archive'); return
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.archive.with_suffix(a.archive.suffix+'.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for name, raw in sorted(evidence.items()):
            member = tarfile.TarInfo(name); member.size = len(raw); member.mtime = 0
            archive.addfile(member, io.BytesIO(raw))
    require(temporary.stat().st_size < LIMIT, 'compressed archive exceeds limit')
    temporary.replace(a.archive)
    atomic(marker, dict(source=source, archive_sha256=sha(a.archive),archive=str(a.archive.resolve())))
    load(a.archive, a.standin)
    print('X archive:', a.archive.stat().st_size, 'bytes')


def configure(p): p.add_argument('--archive', type=Path, required=True)
if __name__ == '__main__': raise SystemExit(entry(job, configure))
