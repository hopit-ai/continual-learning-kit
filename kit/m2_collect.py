#!/usr/bin/env python3
"""Compact C return below 28 MiB; hash-list bulk text evidence retained on partner storage."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit.p3b_read import load
from kit.p3b_score import write_csv
from kit.c_score import COLUMNS
import gzip
import tarfile
import io

EXCLUDE = {'data','partials','raw','rollouts','validation','sdft','train','sources','weights','models','cache','failed-v3'}


def job(a):
    doc = binding(a.out, a.standin)
    rows, files = load(a.out)
    from kit.m2_assignment import validate_files
    from kit.ref_guard import native_files
    validate_files(native_files(files),require_assignment=not a.standin)
    if not rows: raise NotReady('no completed message-2 scores to collect')
    exports = a.out / 'return'; exports.mkdir(exist_ok=True)
    # Heterogeneous legal/retention CSVs get a shared schema; no columns are lost.
    write_csv(exports / 'scores.csv', [{k:r.get(k,'') for k in COLUMNS} for r in rows], COLUMNS)
    metrics = []
    for path in sorted((a.out / 'runs').glob('*/metrics.jsonl')):
        for r in t.read_jsonl(path):
            for k,v in r.get('data',r).items():
                if isinstance(v,(int,float)): metrics.append(dict(run=path.parent.name,step=r.get('step'),metric=k,value=v))
    write_csv(exports / 'metrics.csv', metrics, ['run','step','metric','value'])
    for name in ('scores','metrics'):
        path = exports / (name+'.csv'); path.with_suffix('.csv.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0)); path.unlink()
    included = []; bulk = []; omitted_weights = 0
    for path in sorted(a.out.rglob('*')):
        if not path.is_file() or path.is_symlink(): continue
        name = str(path.relative_to(a.out)); parts = set(path.relative_to(a.out).parts)
        if path.is_relative_to(exports) or path == a.archive or path.name in ('.lock','collect-complete.json','m2-collect.log'): continue
        # Match B exclusions. Native/merged weights stay on storage; export receipts
        # carry merged model hashes. Bulk scientific text is individually hash-listed.
        is_weight = path.suffix in ('.pt','.bin','.safetensors','.distcp')
        if is_weight:
            omitted_weights += 1; continue
        if path.suffix not in ('.json','.jsonl','.csv','.md','.log','.yaml','.err','.txt'): continue
        size = path.stat().st_size; sha = s.file_hash(path)
        record = dict(path=name,bytes=size,sha256=sha)
        omit = bool(parts & EXCLUDE) or any(p.startswith('hf-') for p in parts) or path.name in ('scores.csv','metrics.jsonl','pool.jsonl') or 'logs' in parts or (size > 1_000_000 and path.name not in ('reading.json','binding.json','manifest.json','admission.json','broad-decision.json','stage3-status.json','nomination.json','nomination-commit.json'))
        if (a.out/'scoring-assignment.json').exists() and name.startswith(('scores/','packages/','references/','retained/','reference-scores/')) and path.name=='scores.csv':omit=False
        if name.startswith(('targets/','targets-math/')) and '/teacher/costs/' not in name and path.name not in ('manifest.json','coverage.json','inputs.json','teacher-identity.json','cost.json'): omit = True
        if omit: bulk.append(record)
        else: included.append((path,record))
    atomic(exports / 'bulk-evidence.json', dict(storage_root=str(a.out),files=bulk,
        weight_files_excluded=omitted_weights,weight_hashes='Merged HF weights are in the included runs/*/export-*.json receipts; native checkpoint weights stay on storage.',
        note='Keep this entire OUT durable, including raw teacher attempts/rejections, generated token IDs, schedules, trainer data, metrics and partials.'))
    for name in ('scores.csv.gz','metrics.csv.gz','bulk-evidence.json'):
        p = exports / name; included.append((p,dict(path='return/'+name,bytes=p.stat().st_size,sha256=s.file_hash(p))))
    source = s.digest([r for _,r in included])
    marker = a.out / 'collect-complete.json'
    if marker.exists() and json.loads(marker.read_text()).get('source') == source and a.archive.exists():
        check(a.archive,json.loads(marker.read_text())['archive_sha256']); print('SKIP: collected'); return
    a.archive.parent.mkdir(parents=True,exist_ok=True); temporary = a.archive.with_name(a.archive.name+'.tmp')
    with tarfile.open(temporary,'w:gz') as tar:
        for path,record in included:
            raw = path.read_bytes(); member = tarfile.TarInfo('return/'+record['path']); member.size=len(raw); tar.addfile(member,io.BytesIO(raw))
        raw = json.dumps(dict(files=[r for _,r in included],source=source,technical_synthetic=doc['technical_synthetic'])).encode()
        member=tarfile.TarInfo('return/collect-manifest.json'); member.size=len(raw); tar.addfile(member,io.BytesIO(raw))
    if temporary.stat().st_size >= 28*1024*1024:
        temporary.unlink(); raise CorrectnessStop('compact return exceeds 28 MiB; report sizes before sending')
    temporary.replace(a.archive)
    atomic(marker,dict(source=source,archive_sha256=s.file_hash(a.archive)))
    print(a.archive,a.archive.stat().st_size,'bytes; return only this archive; keep bulk storage durable')


def configure(p): p.add_argument('--archive',type=Path,required=True)
if __name__ == '__main__': raise SystemExit(entry(job,configure,'m2-collect'))
