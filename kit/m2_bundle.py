#!/usr/bin/env python3
"""Package staged partner inputs locally once; shared storage receives tar + SHA only."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,shutil
from kit import m2_snapshot as snapshots,m2_assignment as assignment


def pack(sources,local_root,durable,authority):
    out=Path(local_root)/'input-bundle'
    assignment.require(not out.exists(),'input bundle needs a fresh local directory')
    out.mkdir(parents=True);origins={}
    for name,value in sources.items():
        snapshots.safe_key(name);lexical=Path(value).absolute();source=lexical.resolve();target=out/name
        assignment.require(source.exists(),'input missing '+str(source))
        assignment.require(not source.is_symlink(),'input symlink must be resolved explicitly')
        if source.is_dir():shutil.copytree(source,target)
        else:shutil.copyfile(source,target)
        origins[str(source)]=name;origins[str(lexical)]=name;origins[str(Path(value))]=name
    snapshots.atomic(out/'input-origins.json',origins)
    return snapshots.publish(out,durable,'INPUTS',assignment=assignment.sha(authority),origins=origins)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','local-root','durable','assignment'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(pack(json.loads(a.sources.read_text()),a.local_root,a.durable,a.assignment),sort_keys=True))
if __name__=='__main__':main()
