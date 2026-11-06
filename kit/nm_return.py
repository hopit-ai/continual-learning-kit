#!/usr/bin/env python3
"""Reproduce a frozen NM reader from the compact return, on CPU only."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import tarfile
from kit.nm_common import *
from kit.nm_read import read


def reproduce(archive):
    with tarfile.open(archive,'r:gz') as tar:
        names=tar.getnames();need(len(names)==len(set(names)),'duplicate return members')
        need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names),'unsafe return member')
        manifests=[n for n in names if n.endswith('/archive-manifest.json')]
        need(len(manifests)==1,'return manifest absent/ambiguous')
        manifest=json.load(tar.extractfile(manifests[0]));files={}
        for name,expected in manifest['files'].items():
            ready(name in names,'archive evidence absent: '+name)
            raw=tar.extractfile(name).read();need(raw_hash(raw)==expected,'archive evidence digest differs: '+name);files[name]=raw
        def doc(key):return json.loads(files[manifest[key]])
        result=read(manifest['family'],doc('scores'),doc('spec'),doc('binding'))
        need(canonical(result)==canonical(doc('reading')),'archived reading does not reproduce')
        return result


def main():
    p=argparse.ArgumentParser();inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--archive');inputs.add_argument('--text',nargs='+',help='pasted ESSENTIAL part files; CPU only')
    p.add_argument('--out');a=p.parse_args()
    if a.text:
        from kit.text_return import parse_parts
        result=parse_parts([Path(name).read_text() for name in a.text])
    else:result=reproduce(a.archive)
    if a.out:atomic(a.out,result)
    print(result['status']);return 0
if __name__=='__main__':raise SystemExit(main())
