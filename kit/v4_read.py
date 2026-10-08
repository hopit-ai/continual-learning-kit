#!/usr/bin/env python3
"""Archive-only v4 reader entry, also shipped in kit-snapshot/kit/."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.v4_readers import read_campaign,read_qualification


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('qualification','main'))
    parser.add_argument('archive',type=Path)
    args=parser.parse_args(argv)
    result=(read_qualification if args.phase=='qualification' else read_campaign)(args.archive)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result['status'] in ('proceed','primary claim met','primary claim not met') else 2
if __name__=='__main__':raise SystemExit(main())
