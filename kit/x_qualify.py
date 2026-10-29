#!/usr/bin/env python3
"""Bind the current frozen Screen S analysis to its actual returned archive."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.s_common import sha, RESOURCE, ROOT, atomic, require, registration, SEEDS


def qualify(archive, standin=False):
    from scripts.screen_s_analysis import analyze, load_archive
    import tarfile,hashlib
    registered=registration()
    require(tuple(registered['analysis']['seeds']) == SEEDS == (101,102,103), 'Screen S three-seed registration required')
    require(registered['analysis']['positive_seeds_min'] == len(SEEDS), 'Screen S requires 3/3 positive seeds')
    document=load_archive(archive,standin)
    with tarfile.open(archive,'r:gz') as source:
        returned_registration=hashlib.sha256(source.extractfile('screen-s/registration.json').read()).hexdigest()
    require(returned_registration == sha(RESOURCE), 'Screen S returned registration hash changed')
    return dict(analyze(document,standin), registration_sha256=returned_registration,
        analysis_sha256=sha(ROOT/'scripts/screen_s_analysis.py'),screen_return_sha256=sha(archive))


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--standin',action='store_true');a=p.parse_args()
    atomic(a.output,qualify(a.archive,a.standin))

if __name__=='__main__':main()
