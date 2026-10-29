#!/usr/bin/env python3
"""Manager command: freeze message-2 expectations from return-1 archives."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit.m2_bind import read_return1, return1_options


def job(a):
    for source in (a.b_source,a.c_source,a.teacher_source):
        require(source.is_file() and source.name.endswith('.tar.gz'),'manager send requires return-1 archives')
    expected,_,_,_=read_return1(a)
    bind(a.out/'send-binding.json',expected)
    print(a.out/'send-binding.json')


def configure(p):
    return1_options(p)
    for name in ('b-source','b-reading','c-source','teacher-source'):
        p.add_argument('--'+name,type=Path,required=True)
if __name__=='__main__': raise SystemExit(entry(job,configure,'m2-send'))
