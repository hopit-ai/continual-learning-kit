#!/usr/bin/env python3
"""Generate prospective code-only pin transitions from a reviewed Git base."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kit.v6_infra import SCHEMA,validate


def generate(base,root=ROOT):
    def git(*args):return subprocess.check_output(['git',*args],cwd=root)
    names=git('ls-tree','-r','--name-only',base,'kit','scripts','docs').decode().splitlines()
    tracked=set(names);pins={}
    def add(name,value,parent):
        if not name.endswith(('.py','.sh')) or not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):return
        for candidate in (name,'kit/'+name,str(parent/name)):
            if candidate in tracked:pins.setdefault(candidate,set()).add(value)
    def walk(value,parent):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str):
                for key in ('sha256','pinned_sha256','new_sha256'):add(value['path'],value.get(key),parent)
            for name,item in value.items():
                add(name,item,parent)
                if name.endswith(('.py','.sh')) and isinstance(item,dict):
                    for k in ('original_sha256','v2_sha256','sha256'):add(name,item.get(k),parent)
                walk(item,parent)
        elif isinstance(value,list):
            for item in value:walk(item,parent)
    pin_names=[name for name in names if name.endswith('.json') or Path(name).name=='SHA256SUMS']
    # Read base blobs in one Git process, rather than spawning a process for
    # every historical receipt. Length framing preserves arbitrary JSON bytes.
    batch=subprocess.run(['git','cat-file','--batch'],cwd=root,
        input=''.join(base+':'+name+'\n' for name in pin_names).encode(),stdout=subprocess.PIPE,check=True).stdout
    offset=0
    for name in pin_names:
        end=batch.index(b'\n',offset);header=batch[offset:end].split();size=int(header[-1])
        raw=batch[end+1:end+1+size].decode();offset=end+size+2
        if name.endswith('.json'):
            try:walk(json.loads(raw),Path(name).parent)
            except ValueError:continue
        else:
            for line in raw.splitlines():
                fields=line.split()
                if len(fields)==2:add(fields[1].lstrip('*'),fields[0],Path(name).parent)
    changed=set(git('diff','--name-only',base,'--','kit','scripts').decode().splitlines())
    files=[]
    for name,values in sorted(pins.items()):
        if name not in changed:continue
        path=root/name
        old=git('show',base+':'+name)
        if not path.is_file():raise ValueError('pinned implementation removed: '+name)
        current=path.read_bytes()
        if current==old:continue
        # Include each historical pin AND the base bytes. Different checkers can
        # pin different generations of the same infrastructure implementation.
        for pinned in sorted(values|{hashlib.sha256(old).hexdigest()}):
            files.append(dict(path=name,pinned_sha256=pinned,new_sha256=hashlib.sha256(current).hexdigest(),
                reason='Prospective v6 scheduling, resource, handoff and authority infrastructure; scientific definitions unchanged'))
    return validate(dict(schema=SCHEMA,statement='scientific definitions unchanged',files=files))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();doc=generate(a.base);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(doc,sort_keys=True,indent=2)+'\n')
    print(len(doc['files']),'exact prospective pin transitions generated')
if __name__=='__main__':main()
