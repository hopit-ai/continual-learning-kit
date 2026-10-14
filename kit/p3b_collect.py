#!/usr/bin/env python3
"""CPU return: compressed question/metric exports plus small evidence archive, no weights."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_common import *
from kit.p3b_read import load
from kit.p3b_score import write_csv
import gzip
import io
import tarfile


def job(a):
    rows,_=load(a.out)
    if not rows: raise NotReady('no completed scores to collect')
    sources=[p for p in sorted(a.out.rglob('*')) if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.csv','.md','.log')
        and not {'data','partials','raw','rollouts','validation','sdft','train','sources','weights','models','cache'} & set(p.relative_to(a.out).parts)
        and not any(x.startswith('hf-') for x in p.relative_to(a.out).parts) and not p.is_relative_to(a.out / 'return') and p.name not in ('tests.jsonl','scores.csv','metrics.csv','collect-complete.json','p3b-collect.log')]
    fingerprint=s.digest(dict(scores=rows,files={str(p.relative_to(a.out)):s.file_hash(p) for p in sources}))
    marker=a.out / 'collect-complete.json'
    if marker.exists() and json.loads(marker.read_text()).get('source')==fingerprint and a.archive.exists():
        check(a.archive,json.loads(marker.read_text())['archive_sha256']); print('SKIP: collected'); return
    exports=a.out / 'return'; exports.mkdir(exist_ok=True)
    write_csv(exports / 'scores.csv',rows)
    metrics=[]
    for path in sorted((a.out / 'runs').glob('*/metrics.jsonl')):
        for row in t.read_jsonl(path):
            for k,v in row.get('data',row).items():
                if isinstance(v,(float,int)): metrics.append(dict(run=path.parent.name,step=row.get('step'),metric=k,value=v))
    write_csv(exports / 'metrics.csv',metrics,['run','step','metric','value'])
    for name in ('scores','metrics'):
        raw=(exports / (name+'.csv')).read_bytes(); (exports / (name+'.csv.gz')).write_bytes(gzip.compress(raw,mtime=0)); (exports / (name+'.csv')).unlink()
    sources=[p for p in sources if not p.is_relative_to(exports) and p!=marker]
    sources += [exports / 'scores.csv.gz',exports / 'metrics.csv.gz']
    manifest=[]; skipped=[]; a.archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.archive,'w:gz') as tar:
        for p in sources:
            raw=p.read_bytes(); record=dict(path=str(p.relative_to(a.out)),bytes=len(raw),sha256=t.sha(raw))
            if len(raw)>2_000_000 and p.parent!=exports: skipped.append(record); continue
            manifest.append(record); member=tarfile.TarInfo('return/'+record['path']);member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
        raw=json.dumps(dict(files=manifest,excluded_large=skipped,source=fingerprint)).encode()
        member=tarfile.TarInfo('return/collect-manifest.json');member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
    require(a.archive.stat().st_size<=28*1024*1024,'return exceeds 28 MiB; flag evidence sizes before sending')
    atomic(marker,dict(source=fingerprint,archive_sha256=s.file_hash(a.archive)))
    print(a.archive, a.archive.stat().st_size, 'bytes; send with return/*.csv.gz')


def configure(p): p.add_argument('--archive',type=Path,required=True)
if __name__ == '__main__': raise SystemExit(entry(job,configure,'p3b-collect'))
