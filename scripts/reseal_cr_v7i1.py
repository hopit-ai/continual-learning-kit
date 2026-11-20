#!/usr/bin/env python3
"""Rebuild the v7i1 CR delta, owner hunk packet and resource checksums offline.

Run after every merge/edit to sealed files. Exact unchanged approvals survive;
any changed code or hunk resets both approval records to PENDING. This script
never supplies review or owner approval. The manager fills those after review.
"""
from pathlib import Path
import json
import argparse
import difflib
import os
import sys
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kit import cr_confirmation as cr
from kit.nm_common import atomic, load, sha

RESOURCE = ROOT/'kit/resources/cr-confirmation'
RECEIPT = RESOURCE/'diff-receipt-v7i1.json'
# Fixed source boundaries; inherited v7c content is never read or copied.
V7I_SOURCE = 'e9d517b6'
REVIEW_BASE = '1d6d5251'
TRACE_TIP = '2f4c7c0b'
CR_TIP = '5816c0ae'
BASELINE_TEXTS = RESOURCE/'v7i-baseline.json'
HUNKS = ROOT/'docs/phase4/partner-blockers-20261120/cr-hunks-for-owner-approval.md'


def summary(name, patch=''):
    additions = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+'))
    purposes = {
        'scripts/publish_kit.py': (
            ('test_cr_round2.py', 'Export CR CPU supervision and accounting regressions so the node-side preflight command works.'),
        ),
        'scripts/cr_confirmation_fixture.py': (
            ('import sys', 'Import the current CPU interpreter used by the bind fixture.'),),
        'kit/cr_confirmation.py': (
            ('V7C_BASELINE =', 'Pin the original v7c review baseline and immutable receipt digest.'),
            ('def v7c_delta', 'Generate exact new hunks with branch-line provenance and hash-only inherited transitions.'),
            ('def verify_v7i1_diff', 'Require current code, exact hunks and inherited hashes plus separate review and owner approvals.'),
            ('return v7c_delta', 'Select the portable v7i1 delta for the reviewed v7c baseline.'),
            ('return verify_v7i1_diff', 'Dispatch the v7i1 receipt schema to its separate approval gate.')),
        'kit/cr_dependencies.py': (
            ('reseal_cr_v7i1.py', 'Seal and export the offline reseal utility.'),
            ('cr_process.py', 'Seal the CR process-tree supervisor.')),
        'kit/cr_jobs.py': (
            ('def training_command', 'Adapt the CR M3 entry to isolated imports without changing authority-pinned shared driver bytes.'),
            ('cpu_preflight()', 'Exercise Linux child adoption and detached TERM-resistant cleanup during CPU binding before GPU rows become eligible.'),
            ('isolated_training_job(a)', 'Dispatch CR training and qualification through the isolated command adapter.'),
            ('bind_python', 'Require a separate nonempty bind-time training interpreter.')),
        'kit/nm_budget.py': (
            ('CR_WARN =', 'Define the exact 830 GPU-hour warn-only accounting message.'),
            ('def finish_cr_row', 'Reserve and charge CR per row, validate completion bindings, and recover matching historical holders across requeues.'),
            ('reports=[]', 'Report monotonic row costs, scheduler holder costs and unattributed holder time without a CR cap refusal.')),
        'kit/nm_jobs.py': (
            ('monotonic_start=', 'Start monotonic timing and preserve CR row identity and prospective reservation evidence.'),
            ('cr_gpu_run', 'Route only CR GPU rows directly to process-tree supervision; preserve shared wrapper bytes.'),
            ('elapsed=time.monotonic()', 'Bind terminal CR timing and teardown evidence, release exact charges and preserve non-CR accounting.')),
    }
    for marker, purpose in purposes.get(name, ()):
        if marker in additions:
            return purpose
    return {
        'kit/cr_process.py': 'Track, terminate and confirm all CR descendants, including independent sessions.',
        'kit/cr_text.py': 'Carry the 830 WARN and holder accounting into both pasted-return modes.',
        'kit/cr_finqa.py': 'Use the required training-environment interpreter for native FinQA regeneration.',
        'kit/cr_jobs.py': 'Require the bind interpreter, run Linux process-tree teardown on the CPU, and isolate CR M3 training and HF export launches without changing pinned shared drivers.',
        'kit/nm_budget.py': 'Account CR per row, match requeued holder history, release ended reservations, and warn above 830 GPU-hours.',
        'kit/nm_jobs.py': 'Route CR GPU rows directly to its supervisor; finalize monotonic charges through confirmed teardown.',
        'kit/cr_confirmation.py': 'Verify inherited hash provenance and new exact hunks only after review and approval.',
        'kit/cr_dependencies.py': 'Include the process supervisor and offline reseal utility in the frozen review closure.',
        'scripts/publish_kit.py': 'Name v7i1 and use a CR safety banner that stays true before and after approval.',
        'scripts/cr_confirmation_fixture.py': 'Give the CPU regeneration fixture an explicit bind interpreter.',
        'scripts/cr_confirmation_dryrun.py': 'Give the CPU wrapper rehearsal an explicit bind interpreter.',
        'scripts/reseal_cr_v7i1.py': 'Regenerate new hunks and inherited hash provenance, approval packet and checksums; invalidate approvals after code changes.',
        'kit/m2_privacy.py': 'Inherited since v7c: load optional external private tokens and retain structural scanning. This build does not edit this file.',
        'kit/ts_jobs.py': 'Inherited since v7c: score compared arms in one seed/panel allocation and report identity failures. This build does not edit this file.',
        'kit/ts_read.py': 'Inherited since v7c: compare evidence through the shared allocation identity checks. This build does not edit this file.',
    }.get(name, 'Review this new post-v7i/redaction hunk before approving the v7i1 seal.')


def source_text(commit, name):
    path = os.path.normpath('kit/'+name)
    result = subprocess.run(['git', 'show', commit+':'+path], cwd=ROOT, capture_output=True)
    return result.stdout.decode() if result.returncode == 0 else None


def provenance(send_docs, name, before, after):
    for path in sorted(send_docs.glob('published-kit-message-*.json')):
        record = load(path)
        source = record.get('private_source')
        if not source:
            continue
        text = source_text(source, name)
        if text is not None and cr.digest_text(text) == after:
            return dict(public_tag=record['tag'], public_commit=record['public_commit'], source_commit=source)
    return dict(public_tag=None, public_commit=None, source_commit=V7I_SOURCE)


def line_origins(baseline):
    """Hash changed lines by sign and branch provenance, with no source text."""
    origins = {}
    for name in baseline['code']:
        if name != '../scripts/publish_kit.py' and not name.startswith('trace_'):
            continue
        before = source_text(REVIEW_BASE, name) or ''
        branches = {}
        for label, tip in (('TRACE', TRACE_TIP), ('CR', CR_TIP)):
            after = source_text(tip, name)
            if after is None:
                raise RuntimeError('origin tip unavailable: '+tip)
            changes = list(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))[2:]
            branches[label] = {cr.digest_text(line) for line in changes if line[:1] in ('+', '-')}
        origins[name] = sorted(branches['TRACE'] - branches['CR'])
    return dict(trace_tip=TRACE_TIP, cr_tip=CR_TIP, trace_lines=origins)


def reseal(*, send_docs=None):
    old = load(RECEIPT) if RECEIPT.is_file() else {}
    inherited = RESOURCE/'diff-receipt-v7c.json'
    if sha(inherited) != cr.V7C_RECEIPT_SHA256:
        raise RuntimeError('inherited v7c receipt changed')
    expected = load(inherited)['code']
    current = cr.code_seal()
    if BASELINE_TEXTS.is_file():
        baseline = load(BASELINE_TEXTS)
    else:
        if send_docs is None:
            raise RuntimeError('baseline missing; explicit --send-docs directory required')
        send_docs = Path(send_docs)
        if not send_docs.is_dir():
            raise RuntimeError('--send-docs directory missing')
        public, base, transitions = {}, {}, []
        for name in sorted(set(expected) | set(current)):
            text = source_text(V7I_SOURCE, name)
            if text is not None:
                public[name] = cr.digest_text(text)
            text = source_text(REVIEW_BASE, name)
            if text is not None:
                base[name] = cr.digest_text(text)
        for name in sorted(set(expected) | set(public)):
            if expected.get(name) != public.get(name):
                transitions.append(dict(category='inherited-public-tag', seal_path=name,
                    path=os.path.normpath('kit/'+name), v7c_sha256=expected.get(name),
                    v7i_sha256=public.get(name), **provenance(send_docs, name, expected.get(name), public.get(name))))
        for name in sorted(set(public) | set(base)):
            if public.get(name) != base.get(name):
                transitions.append(dict(category='test-redaction-cherry-pick', seal_path=name,
                    path=os.path.normpath('kit/'+name), v7i_sha256=public.get(name),
                    review_base_sha256=base.get(name), source_commit=REVIEW_BASE,
                    public_tag=None, public_commit=None))
        baseline = dict(review_base_commit=REVIEW_BASE, code=base, v7i_code=public,
                        inherited=transitions, files={})
    stored = baseline.get('line_provenance', {})
    if stored.get('trace_tip') != TRACE_TIP or stored.get('cr_tip') != CR_TIP:
        baseline['line_provenance'] = line_origins(baseline)
    needed = {}
    for name in sorted(baseline['code']):
        if current.get(name) == baseline['code'][name]:
            continue
        before = baseline['files'].get(name)
        if before is None or cr.digest_text(before) != baseline['code'][name]:
            before = source_text(REVIEW_BASE, name)
        if before is None or cr.digest_text(before) != baseline['code'][name]:
            raise RuntimeError('review baseline seal mismatch: '+name)
        needed[name] = before
    baseline['files'] = needed
    atomic(BASELINE_TEXTS, baseline)
    # The superseded historical-text artifact must never ship.
    obsolete = RESOURCE/'v7c-baseline.json'
    if obsolete.exists():
        obsolete.unlink()
    with tempfile.TemporaryDirectory(prefix='cr-v7i1-reseal-') as folder:
        doc = cr.diff_receipt(Path(folder)/'delta.json', base=cr.V7C_BASELINE)
    if not doc.get('baseline_commit'):
        raise RuntimeError('v7c baseline unavailable; cannot reseal')
    doc.update(schema='cr-diff.v7i1', release='kit-message-v7i1',
        inherited_review=dict(path='kit/resources/cr-confirmation/diff-receipt-v7c.json',
                              sha256=cr.V7C_RECEIPT_SHA256),
        allowed_categories=['owner-v7i1-fix', 'owner-v7i1-fix (TRACE)', 'owner-v7i1-fix (CR+TRACE)'], category_definitions={},
        reason='Inherited public transitions bind hashes only; new hunks start at v7i plus test redaction; new approvals required.')
    doc['review'] = dict(status='PENDING', reviewer='PENDING', date='PENDING', evidence='PENDING')
    doc['owner_approval'] = dict(status='PENDING', quote='PENDING', date='PENDING')
    # Carry approval only for identical bytes and ordered exact hunk pairs.
    if all(old.get(k) == doc.get(k) for k in ('code', 'hunks', 'baseline_commit', 'baseline_sha256', 'inherited', 'inherited_review', 'release')):
        for key in ('review', 'owner_approval', 'status', 'escalation'):
            if key in old:
                doc[key] = old[key]
    atomic(RECEIPT, doc)
    lines = ['# CR v7i1 hunks for owner approval', '',
             'Review: '+doc['review']['status']+'. Owner approval: '+doc['owner_approval']['status']+'.', '',
             'No approval is supplied by this script. The manager fills the receipt review and owner_approval fields after the corresponding approvals, then sets status to COMPLETE and escalation to null.', '',
             'The original reviewed v7c receipt is unchanged. Only the new hunks below request owner approval. Inherited provenance follows, with hashes only. Registered settings, data, seeds, thresholds and endpoints are unchanged.', '',
             'New-hunk review base: v7i (`e9d517b6`) plus test redaction (`1d6d5251`).', '',
             'Inherited baseline: `'+doc['baseline_commit']+'`. Receipt: `kit/resources/cr-confirmation/diff-receipt-v7i1.json`.', '']
    trace_count = sum(h['origins'] == ['TRACE'] for h in doc['hunks'])
    mixed_count = sum(h['origins'] == ['CR', 'TRACE'] for h in doc['hunks'])
    inherited_count = sum(v['category']=='inherited-public-tag' for v in doc['inherited'])
    redaction_count = sum(v['category']=='test-redaction-cherry-pick' for v in doc['inherited'])
    origin_counts = ([f'{trace_count} TRACE-origin'] if trace_count else [])
    origin_counts += ([f'{mixed_count} mixed CR/TRACE-origin'] if mixed_count else [])
    counts = ', '.join(origin_counts) or '0 TRACE-origin'
    lines += [f"{len(doc['hunks'])} new hunks for approval ({counts}), {inherited_count} inherited files and {redaction_count} test-redaction transition, hashes only.", '']
    for n, h in enumerate(doc['hunks'], 1):
        lines += [f"## Hunk {n}: {h['file']} ({h['category']})", '', ((summary(h['file'], h['patch']) if 'CR' in h['origins'] else '') + (' TRACE-origin change in this shared file (line provenance against the recorded branch tips).' if 'TRACE' in h['origins'] else '')).strip(), '',
                  'Exact hunk SHA256: `'+h['sha256']+'`.', '', '```diff',
                  '--- v7i-plus-test-redaction/'+h['file'], '+++ '+h['file'], h['patch'].rstrip(), '```', '']
    lines += ['## Inherited provenance (hashes only)', '',
        'These inherited files shipped in public tags but were never CR-reviewed. None is on the CR training/scoring path. The m2_expect → m2_privacy import is inside the M2 public-send projection; CR does not call that projection or the M2 bind/rescore dispatch; these optional imports are not on its training/scoring path. The all-row CPU import trace checks their absence (production GPU execution remains UNVERIFIED). The publisher is build-time only; teacher-signal modules dispatch separately from CR.', '',
        '| File | Category | Before SHA256 | After SHA256 | First public tag / commit (or private source) |',
        '| --- | --- | --- | --- | --- |']
    for item in doc['inherited']:
        before = item.get('v7c_sha256', item.get('v7i_sha256'))
        after = item.get('review_base_sha256', item.get('v7i_sha256'))
        origin = ((str(item['public_tag'])+' / '+str(item['public_commit'])) if item.get('public_tag')
                  else 'private source '+item['source_commit'])
        lines.append('| '+ ' | '.join((item['path'], item['category'], str(before), str(after), origin))+' |')
    HUNKS.parent.mkdir(parents=True, exist_ok=True)
    HUNKS.write_text('\n'.join(lines))
    files = sorted(p for p in RESOURCE.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (RESOURCE/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(RESOURCE))+'\n' for p in files))
    return dict(status=doc['status'], sealed_files=len(doc['code']), hunks=len(doc['hunks']),
                receipt_sha256=sha(RECEIPT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--send-docs', type=Path, help='explicit provenance directory; required only if baseline is missing')
    args = parser.parse_args()
    print(json.dumps(reseal(send_docs=args.send_docs), sort_keys=True))
