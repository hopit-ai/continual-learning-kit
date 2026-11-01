#!/usr/bin/env python3
"""Read pasted WhatsApp return parts; checksums are advisory; no independent scientific verification.

python scripts/read_text_return.py pasted-*.txt --out reading.json
python scripts/read_text_return.py part1.txt --part1-only --out decisions.json
"""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from kit.text_return import canonical, parse_parts


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parts',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--part1-only',action='store_true')
    p.add_argument('--expect-commit',help='require the published kit full commit (40 lowercase hex digits)')
    p.add_argument('--base',type=Path,help='previously verified owner-local ESSENTIAL reading; accept later FULL parts')
    a=p.parse_args(argv)
    try:result=parse_parts([path.read_text() for path in a.parts],a.part1_only,json.loads(a.base.read_text()) if a.base else None,expect_commit=a.expect_commit)
    except (ValueError,OSError) as exc:
        print('REFUSED: '+str(exc),file=sys.stderr);return 2
    a.out.write_text(canonical(result)+'\n')
    print(f"{len(result['decisions'])} verdicts; complete={result['complete']}; PARTNER-COMPUTED; independent re-verification UNVERIFIED")
    bad=[k for k,v in result.get('part_checksums',{}).items() if v=='TRANSCRIPTION-UNVERIFIED']
    if bad:print('ADVISORY: TRANSCRIPTION-UNVERIFIED '+', '.join(bad))
    return 0


if __name__=='__main__':raise SystemExit(main())
