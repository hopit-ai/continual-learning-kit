#!/usr/bin/env python3
"""Every combined inventory row through the real wrapper; CPU stand-ins only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse
import json
from kit import combined_dryrun,text_return


def run(out, *, resume_core=False):
    result=combined_dryrun.run(out,resume_core=resume_core)
    root=Path(out)
    attempts=sorted((p for p in (root/'control/return-text').iterdir() if p.is_dir()),key=lambda p:p.stat().st_mtime) if (root/'control/return-text').is_dir() else []
    parts=sorted((attempts[-1] if attempts else root/'control/return-text').glob('RETURN-TEXT-*.md'))  # tier M2: one folder per REPORT attempt
    parsed=text_return.parse_parts([p.read_text() for p in parts])
    assert parsed['infrastructure_authority']
    assert any(row['family']=='4A' for row in parsed['decisions'])
    assert not any(row['family']=='4A' and row['verdict']=='DEFERRED' for row in parsed['decisions'])
    result.update(text_return_roundtrip=True,mandatory_4a_in_text=True,
        essential_parts=len(parts),all_parts_checksum_match=all(v=='MATCH' for v in parsed['part_checksums'].values()))
    assert result['all_parts_checksum_match']
    reading=json.loads((root/'A4/reading/summary.json').read_text())
    returned=json.loads((root/'A4/returned-reading/summary.json').read_text())
    assert reading==returned,'4A report and archive must use the same native bootstrap settings'
    result.update(a4_archive_readback_equal=True,a4_hardware_status=reading['hardware_status'],a4_seed_df=reading['primary']['interval']['df'])
    (root/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--standin',action='store_true',required=True)
    p.add_argument('--resume-core',action='store_true',help='reuse verified CPU core wrapper records before any 4A binding')
    a=p.parse_args();print(json.dumps(run(a.out,resume_core=a.resume_core),indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
