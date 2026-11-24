#!/usr/bin/env python3
"""Offline CR fix receipt: ceiling arithmetic, focused tests and approval state."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def arithmetic():
    from kit.cr_jobs import inventory
    from kit.nm_common import load
    bounds = load(ROOT/'kit/resources/cr-confirmation/row-max-seconds.json')
    rows = [r for r in inventory() if r['gpus']]
    ceilings = sum(r['gpus']*bounds[r['id']]/3600 for r in rows)
    allowance = 830
    return dict(rows=len(rows), row_ceiling_gpu_hours=ceilings, allowance_gpu_hours=allowance,
        overage_gpu_hours=ceilings-allowance,
        reservation_slack_gpu_hours=sum(r['gpus']*30/3600 for r in rows),
        before=dict(reservation_rule='holder age + row ceiling + 30 seconds; one row per allocation',
                    result='REFUSE: even row ceilings alone exceed 830'),
        after=dict(reservation_rule='row age + row ceiling + 30 seconds; several CR rows per holder; exact charge uses wrapper MONOTONIC through confirmed teardown',
                   result='WARN: 896 ceiling GPU-hours permitted; confirmed actual row costs replace reservations; missing completion uses scheduler-end upper bounds'),
        estimate_note='Ceilings are prospective bounds, not measured runtime or a spend estimate. Holder-age inflation depends on actual scheduling and is UNVERIFIED.')


def test_files():
    """Explicit integrated set plus all infrastructure/wrapper importers."""
    files = {'tests/'+name+'.py' for name in (
        'test_kitfix_trace', 'test_trace_feedback_launch', 'test_trace_feedback_mechanism',
        'test_trace_general', 'test_trace_learning_inputs', 'test_trace_metrics',
        'test_trace_operator', 'test_trace_probe', 'test_trace_registration', 'test_kitfix_v7i3',
        'test_nm_prepare', 'test_kit_sdft', 'test_v7_operator_fixes', 'test_v7b_fixes',
        'test_check_a2_release')}
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_cr_*.py'))
    # Search without collecting fixtures or opening calibration LOCAL-KEY files.
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_*.py')
                 if 'LOCAL-KEY' not in p.name and p.name != 'test_trace_calibration_v7h_data.py'
                 and any(name in p.read_text() for name in ('v6_infra', 'a4_common')))
    return sorted(files)


def run_tests():
    env = dict(PATH=os.defpath, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               KIT_V6_AUTHORITY_SHA=hashlib.sha256(
                   (ROOT/'kit/resources/v6/infrastructure-update.json').read_bytes()).hexdigest())
    if 'HOME' in os.environ:
        env['HOME'] = os.environ['HOME']
    bootstrap = 'import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module("pytest",run_name="__main__")'
    with tempfile.TemporaryDirectory(prefix='kitfix-cr-tests-') as temporary:
        command = [sys.executable, '-I', '-c', bootstrap, str(ROOT), '-q', *test_files(),
                   '--basetemp='+str(Path(temporary)/'basetemp'), '-p', 'no:cacheprovider']
        return subprocess.run(command, cwd=ROOT, env=env).returncode


def approval_state():
    from kit.nm_common import load
    doc = load(ROOT/'kit/resources/cr-confirmation/diff-receipt-v7i3.json')
    return {k: doc[k] for k in ('status', 'review', 'owner_approval')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arithmetic-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(arithmetic(), indent=2), flush=True)
    if args.arithmetic_only:
        return 0
    result = run_tests()
    print(json.dumps(dict(python=sys.executable, hydra_interpreter=sys.executable,
                          hydra_validation='Required TRACE tests run the real parser when installed; skips mean UNVERIFIED',
                          test_exit_code=result, linux_execution='UNVERIFIED: macOS CPU checks only; run tests/test_cr_round2.py on the Linux node CPU before GPU rows', approval=approval_state()), indent=2))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
