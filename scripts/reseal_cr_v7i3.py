#!/usr/bin/env python3
"""New PENDING CR receipt using the inherited exact-hunk/reseal machinery."""
from pathlib import Path
import difflib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kit.nm_common import atomic, digest, load, sha
from scripts import reseal_cr_v7i1 as previous

RECEIPT = ROOT/'kit/resources/cr-confirmation/diff-receipt-v7i3.json'
HUNKS = ROOT/'docs/send/v7i3-cr-hunks-for-owner-approval.md'
BASE = '67eca3bb'
REVIEW_FILES = ('kit/README.md','kit/README-next-message.md','kit/README-v7c.md',
                 'docs/phase4/partner-return-20261123/release-notes.md',
                 'tests/test_cr_confirmation.py','tests/test_cr_finqa.py','tests/test_cr_round3.py',
                 'tests/test_cr_v7i1.py','tests/test_cr_v7c_closure.py','tests/test_cr_v7c_review.py',
                 'tests/test_kitfix_v7i2.py','tests/test_kitfix_v7i3.py')


def reseal():
    additional = []
    for name in REVIEW_FILES:
        old = subprocess.run(['git','show',BASE+':'+name],cwd=ROOT,capture_output=True,text=True)
        path = ROOT/name
        if old.returncode and not any(line.startswith('fatal: path ') for line in old.stderr.splitlines()):
            raise RuntimeError('review base unavailable: '+old.stderr)
        if path.is_file() or old.returncode == 0:
            patch = ''.join(difflib.unified_diff(old.stdout.splitlines(True),
                path.read_text().splitlines(True) if path.is_file() else [],fromfile='v7i2/'+name,tofile=name))
            if patch:
                additional.append(dict(file=name, patch=patch, sha256=digest(patch)))
    # Keep the historical v7i1 receipt and its approvals untouched. The helper
    # regenerates the full execution closure and exact cumulative review delta.
    receipt, hunks = previous.RECEIPT, previous.HUNKS
    previous.RECEIPT, previous.HUNKS = RECEIPT, HUNKS
    try:
        previous.reseal(schema='cr-diff.v7i3', release='kit-message-v7i3',
                        packet_title='CR v7i3 hunks', packet_receipt=RECEIPT.name,
                        packet_settings='Declared v7i3 departure: matcher-flagged task rows are removed before scheduling, identically for every CR arm; 1,280 exposures remain. Unfiltered byte proofs remain required. Legacy owner-v7i1-fix category names are retained by the inherited hunk verifier.',
                        review_inputs=dict(base=BASE, additional_hunks=[
                            {key:hunk[key] for key in ('file','sha256')} for hunk in additional]))
    finally:
        previous.RECEIPT, previous.HUNKS = receipt, hunks
    doc = load(RECEIPT)
    doc['reason'] = 'v7i3 task exclusions and current conservative execution closure require new exact-hunk approval.'
    atomic(RECEIPT, doc)
    packet = HUNKS.read_text()
    packet += '\n## Additional CR documentation and regression hunks from v7i2\n\n'
    for hunk in additional:
        packet += '\n```diff\n'+hunk['patch']+'```\n'
    HUNKS.write_text(packet)
    resource = RECEIPT.parent
    (resource/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(resource))+'\n'
        for p in sorted(resource.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return dict(status=doc['status'],owner_approval=doc['owner_approval']['status'],
                hunks=len(doc['hunks']),receipt_sha256=sha(RECEIPT))


if __name__ == '__main__':
    print(json.dumps(reseal(),sort_keys=True))
