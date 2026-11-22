#!/usr/bin/env python3
"""Offline v7i2 regression receipt; synthetic checkpoints and optional staged census."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TESTS = ['tests/test_kitfix_v7i2.py', 'tests/test_trace_repair.py']


def run_tests():
    with tempfile.TemporaryDirectory(prefix='kitfix-v7i2-') as temporary:
        command = [sys.executable, '-m', 'pytest', '-q', *TESTS,
                   '--basetemp=' + str(Path(temporary)/'basetemp'), '-p', 'no:cacheprovider']
        subprocess.run(command, cwd=ROOT, check=True)


def scheduler_demo():
    import torch
    from kit.nm_counters import install_counters
    def make():
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.01)
        return SimpleNamespace(actor_optimizer=optimizer,
            actor_lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: .99**n),
            actor=SimpleNamespace(_update_teacher=lambda: None))
    old = make()
    for owner in (old.actor_optimizer, old.actor_lr_scheduler):
        original = owner.step
        def counted(*args, _original=original, **kwargs):
            return _original(*args, **kwargs)
        counted._wrapped_by_lr_sched = True
        owner.step = counted
    torch.save(old.actor_optimizer.state_dict(), BytesIO())
    try:
        torch.save(old.actor_lr_scheduler.state_dict(), BytesIO())
    except (AttributeError, TypeError) as error:
        print('Old scheduler torch.save: expected failure ('+type(error).__name__+')')
    else:
        raise AssertionError('old scheduler patch unexpectedly serialised')
    new = make()
    install_counters(new)
    for _ in range(2):
        new.actor_optimizer.step()
        new.actor_lr_scheduler.step()
        new.actor._update_teacher()
    assert new._nm_steps == dict(optimizer=2, scheduler=2, ema=2)
    torch.save(new.actor_optimizer.state_dict(), BytesIO())
    torch.save(new.actor_lr_scheduler.state_dict(), BytesIO())
    print('New optimizer and scheduler torch.save: PASS; counters:', new._nm_steps)
    print('Native GPU/FSDP execution: UNVERIFIED by this CPU receipt.')


def crosscheck(source=None, converter=None, exclusions=None):
    if source is None or not Path(source).is_dir():
        print('T5 cross-check: SKIPPED (staged data absent; supply --staged-source, --converter, --exclusions).')
        return
    if not converter or not exclusions:
        raise ValueError('staged cross-check needs the shipped converter and bound exclusions')
    from kit import trace_repair as repair
    from kit.trace_common import SUBSET, sha
    with tempfile.TemporaryDirectory(prefix='kitfix-v7i2-data-') as temporary:
        work = Path(temporary)
        subprocess.run([sys.executable, '-B', str(converter), 'convert-tasks',
            '--source', str(source), '--out', str(work/'converted'),
            '--group-policy', 'singleton'], check=True)
        def ref(path):
            return dict(path=str(path), sha256=sha(path))
        cfg = dict(studies=['gold', '27b'],
            datasets={task: {split: ref(work/'converted'/task/(split+'.jsonl'))
                             for split in ('train', 'eval', 'test')} for task in SUBSET},
            exclusions=ref(Path(exclusions)))
        result = repair.repair(cfg, work/'repaired')
        manifest = result['manifest']
        print('Task | pairs before | drops by rule | distinct after | train SHA256')
        for task in SUBSET:
            train_hash = (result.get('config_edits', {}).get('datasets', {}).get(task, {})
                          .get('train', {}).get('sha256', 'PENDING'))
            print(task, manifest['pairs_before'].get(task, 0), manifest['drops_per_rule'][task],
                  manifest['counts'][task]['after']['distinct'], train_hash, sep=' | ')
        print('Cleared pairs:', manifest['cleared_pair_count'],
              '; remaining cleared:', manifest['remaining_cleared_pair_count'])
        print('audit_review SHA256:', result.get('config_edits', {}).get('audit_review', {}).get('sha256', 'PENDING'))
        print('Expected hashes match the node only if its own exclusion populations add no flags.')
        if result['status'] != 'COMPLETE':
            raise RuntimeError('T5 PENDING: '+json.dumps(result['problems'], sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-tests', action='store_true')
    parser.add_argument('--staged-source', type=Path, default=ROOT/'tmp/trace-download-20261118/hf')
    inputs = ROOT/'docs/phase4/trace-node-inputs/hf-node-inputs'
    parser.add_argument('--converter', type=Path, default=inputs/'trace_node_inputs.py')
    parser.add_argument('--exclusions', type=Path, default=inputs/'exclusions-kit-part.jsonl')
    args = parser.parse_args()
    if not args.skip_tests:
        run_tests()
    scheduler_demo()
    crosscheck(args.staged_source, args.converter, args.exclusions)


if __name__ == '__main__':
    main()
