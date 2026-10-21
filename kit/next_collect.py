#!/usr/bin/env python3
"""Compact return: gzipped per-question scores, small summaries, durable evidence hashes."""
from pathlib import Path
import sys
if __package__ in (None, ''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.next_common import *
import gzip
import io
import tarfile

LIMIT=2_000_000
CAP=28*1024*1024
EXCLUDED={'data','partials','raw','rollouts','validation','sdft','train','sources','weights','models','cache','logs'}


def job(a):
    files={};omitted=[];scores=0
    durable=getattr(a,'durable_root',None)
    if not a.standin:require(durable is not None and durable.is_dir(),'copy full evidence to --durable-root before collection')
    durable=(durable or a.out).resolve()
    for manifest in a.out.rglob('manifest.json'):
        if 'partials' not in manifest.relative_to(a.out).parts and isinstance(json.loads(manifest.read_text()).get('files'),dict):
            frozen_files(manifest.parent)
    for p in sorted(a.out.rglob('*')):
        if not p.is_file() or p.is_symlink() or p.resolve()==a.archive.resolve() or p.name in ('next-collect.log','collect-complete.json') or p.name.endswith('.partial') or 'return' in p.relative_to(a.out).parts:continue
        name=str(p.relative_to(a.out));record=dict(path=name,bytes=p.stat().st_size,sha256=s.file_hash(p))
        parts=set(p.relative_to(a.out).parts)
        excluded=(p.suffix in ('.safetensors','.bin','.pt','.parquet') or bool(parts & EXCLUDED) or
                  any(x.startswith('hf-') for x in parts) or p.name in ('tests.jsonl','responses.jsonl','metrics.csv','collect-complete.json','next-collect.log') or
                  p.suffix not in ('.json','.jsonl','.csv','.md','.gz'))
        if not excluded and p.name=='scores.csv':
            raw=gzip.compress(p.read_bytes(),mtime=0);target=name+'.gz'
            require(len(raw)<=LIMIT,'compressed question CSV exceeds 2 MB; split this score export before collection')
            files[target]=raw;scores+=1;continue
        if excluded or p.stat().st_size>LIMIT:
            # The archive names the exact durable counterpart of every omitted bulk file.
            if not a.standin:check(durable/name,record['sha256'])
            omitted.append(record);continue
        files[name]=p.read_bytes()
    if not scores:raise NotReady('no scores to collect')
    manifest=dict(files=[dict(path=n,bytes=len(b),sha256=t.sha(b)) for n,b in files.items()],omitted=omitted,
                  durable_root=str(durable),technical_synthetic=a.standin)
    files['collect-manifest.json']=(json.dumps(manifest,sort_keys=True)+'\n').encode()
    a.archive.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.archive.with_suffix(a.archive.suffix+'.partial')
    with tarfile.open(tmp,'w:gz') as tar:
        for name,raw in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0;tar.addfile(info,io.BytesIO(raw))
    require(tmp.stat().st_size<=CAP,'compact return exceeds 28 MiB; retain archive.partial and report before considering multipart')
    tmp.replace(a.archive)
    print('Return:',len(files),'files;',len(omitted),'hash-listed durable omissions;',a.archive.stat().st_size,'bytes')


def configure(p):
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--durable-root',type=Path,help='full evidence copy, required outside stand-in mode')
if __name__ == '__main__':raise SystemExit(entry(job,configure,'next-collect'))
