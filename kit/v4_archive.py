"""Verified, read-only archive access. No tar extraction and no external paths."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


class Archive:
    """Read collector archives only after checking every listed byte and member."""
    def __init__(self, path):
        self.path=Path(path)
        self.files={}
        self.tar=None
        if not self.path.is_file():
            raise ValueError('missing archive: '+str(path))
        self.tar=tarfile.open(self.path,'r:gz')
        try:
            seen=set()
            tops=set()
            for member in self.tar.getmembers():
                name=PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or len(name.parts)<2:
                    raise ValueError('unsafe archive path')
                if member.name in seen: raise ValueError('duplicate archive member: '+member.name)
                seen.add(member.name); tops.add(name.parts[0])
                if not member.isfile(): raise ValueError('archive contains non-file member')
                self.files[str(PurePosixPath(*name.parts[1:]))]=self.tar.extractfile(member).read()
            if len(tops)!=1: raise ValueError('archive needs one root')
            manifest=self.json('collect.manifest.json')
            if manifest['schema']!='kit-collect.v1': raise ValueError('unknown archive manifest')
            if manifest.get('truncated') or manifest.get('symlinks'):
                raise ValueError('incomplete or symlinked archive evidence')
            entries=manifest['files']; listed=set()
            for item in entries:
                name=item['path']
                if name in listed: raise ValueError('duplicate manifest path: '+name)
                listed.add(name)
                raw=self.files.get(name)
                if raw is None: raise ValueError('missing manifest file: '+name)
                if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
                    raise ValueError('archive hash/size mismatch: '+name)
            if listed!=set(self.files)-{'collect.manifest.json'}:
                raise ValueError('unmanifested archive evidence')
            self.manifest=manifest
        except BaseException:
            self.tar.close()
            raise

    def json(self,name):
        try: return json.loads(self.files[name])
        except KeyError: raise ValueError('missing required artifact: '+name) from None

    def jsonl(self,name):
        try: raw=self.files[name].decode('utf-8')
        except KeyError: raise ValueError('missing required artifact: '+name) from None
        return [json.loads(line) for line in raw.split('\n') if line.strip()]

    def close(self):
        if self.tar: self.tar.close()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()


class DirectoryEvidence:
    """Collector-selected local evidence for pre-collection row gates.

    Uses exactly the collector's pruning and size policy. This does not certify a
    return archive: only Archive verifies its exhaustive manifest. Models are
    never read, and symlinked evidence refuses rather than following a target.
    """
    def __init__(self, work):
        from kit.collect import gather
        work=Path(work).resolve()
        self.work=work
        files,excluded,links=gather(work,[work])
        if links: raise ValueError('symlinked local evidence')
        large=[entry for entry in excluded if entry['reason']=='over 50 MB']
        if large: raise ValueError('local evidence exceeds collector size limit')
        self.files={str(rel):path.read_bytes() for path,rel,_ in files}

    json=Archive.json
    jsonl=Archive.jsonl
