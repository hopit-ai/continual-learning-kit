#!/usr/bin/env python3
"""Manager command: freeze message-2 expectations from return-1 archives."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit.m2_bind import read_return1, return1_options


def job(a):
    from kit.m2_expect import public_document, check_public
    require(not (getattr(a,'standin',False) and getattr(a,'decisions',None)), '--standin cannot be combined with --decisions')
    if a.public_from:
        from kit import m2_return1 as r
        require(a.c_source and a.decisions and a.b_source, 'public export requires original C/B archives and decisions')
        ruling=r.decisions(a.decisions)
        cdoc,_=r.binding(a.c_source,ruling)
        expected=json.loads(a.public_from.read_text())
        expected['b_archive_sha256']=s.file_hash(a.b_source)
        doc=public_document(expected,cdoc,ruling)
        atomic(a.out/'send-binding-expected.json',doc)
        print('public send expectation written (content identities only)')
        return
    require(all((a.b_source,a.b_reading,a.c_source,a.teacher_source)), 'B reading and original B/C/teacher archives required')
    for source in (a.b_source,a.c_source,a.teacher_source):
        require(source.is_file() and source.name.endswith('.tar.gz'),'manager send requires return-1 archives')
    expected,cdoc,_,_=read_return1(a)
    expected['b_archive_sha256']=s.file_hash(a.b_source)
    if a.public_expect and not getattr(a,'standin',False):
        from kit.m2_return1 import decisions
        check_public(a.public_expect,expected,cdoc,decisions(a.decisions))
    bind(a.out/'send-binding.json',expected)
    print('local send binding written; public content expectation '+('PASS' if a.public_expect else 'not requested'))


def configure(p):
    from kit.m2_expect import EXPECTED
    return1_options(p)
    for name in ('b-source','b-reading','c-source','teacher-source'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--public-from',type=Path,help='export a verified private binding as a path-free public expectation')
    p.add_argument('--public-expect',type=Path,default=EXPECTED,help='real sends default to the shipped public expectation; refuse any content mismatch before writing')
if __name__=='__main__': raise SystemExit(entry(job,configure,'m2-send'))
