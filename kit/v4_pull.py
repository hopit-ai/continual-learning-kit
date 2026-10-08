#!/usr/bin/env python3
"""Verify evidence pulls before use, including Modal smoke and real-27B checks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.v4_archive import Archive


def verify(path,expected_sha256=None):
    path=Path(path)
    with path.open('rb') as handle:
        digest=hashlib.file_digest(handle,'sha256').hexdigest()
    if expected_sha256 and digest!=expected_sha256:raise ValueError('pulled archive hash differs from remote receipt')
    with Archive(path) as archive:
        return {'archive_sha256':digest,'manifest_sha256':hashlib.sha256(archive.files['collect.manifest.json']).hexdigest(),
                'files_checked':len(archive.manifest['files'])}



def verify_directory(path):
    """Refuse unmanifested or modified extracted evidence before local rehearsal."""
    root=Path(path).resolve()
    manifest=json.loads((root/'collect.manifest.json').read_text())
    if manifest.get('schema')!='kit-collect.v1' or manifest.get('symlinks') or manifest.get('truncated'):
        raise ValueError('incomplete pulled evidence manifest')
    listed=set()
    for entry in manifest['files']:
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe pulled evidence path')
        target=root/relative
        if any(p.is_symlink() for p in (target,*target.parents)) or not target.is_file():raise ValueError('symlinked/missing pulled evidence')
        if target.stat().st_size!=entry['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('pulled evidence manifest hash mismatch')
        if entry['path'] in listed:raise ValueError('duplicate pulled evidence path')
        listed.add(entry['path'])
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='collect.manifest.json'}
    if actual!=listed:raise ValueError('unmanifested pulled evidence')
    return manifest

def modal_pull(volume,remote,out,expected_sha256):
    """Download into an owned temporary directory, verify, then publish once."""
    import os
    import tempfile
    from kit.runner import bounded_command
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix='.v4-pull-') as temp:
        downloaded=Path(temp)/'archive.tar.gz'
        result=bounded_command(['modal','volume','get',volume,remote,str(downloaded)],timeout=600,log=Path(temp)/'pull.log')
        if result['returncode']:raise ValueError('evidence_pull: '+str(result['failure_type']))
        receipt=verify(downloaded,expected_sha256)
        os.link(downloaded,out)  # Complete verified archive; no overwrite.
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('four-arm-smoke','27b-check','partner'),required=True)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--sha256')
    parser.add_argument('--modal-volume');parser.add_argument('--remote')
    args=parser.parse_args(argv)
    if args.modal_volume:
        if not args.remote or not args.sha256:parser.error('Modal pull requires remote path and returned sha256')
        result=modal_pull(args.modal_volume,args.remote,args.archive,args.sha256)
    else:result=verify(args.archive,args.sha256)
    print(json.dumps({'kind':args.kind,**result},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
