#!/usr/bin/env python3
"""TS-only native entry, sharing the frozen stage checker semantics."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    from kit.nm_entry import compute_score as incumbent
    if data_source in ('finqa', 'contract_nli', 'p3b_replay'):
        return incumbent(data_source, solution_str, ground_truth, extra_info)
    if data_source == 'math_screen':
        from kit.beds.math_screen import compute_score as math_score
        return math_score(data_source, solution_str, ground_truth, extra_info)
    raise ValueError('unknown TS checker source')


def main():
    from kit.ts_native import TSTaskRunner, nm_native
    nm_native.check_pinned()
    nm_native.main_ppo.TaskRunner = TSTaskRunner
    nm_native.main_ppo.main()


if __name__ == '__main__':
    main()
