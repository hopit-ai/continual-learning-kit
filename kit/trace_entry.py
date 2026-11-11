#!/usr/bin/env python3
"""TRACE pure-D reward adapter. Native loss, EMA, mask and resume remain NM's."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source != 'trace':
        raise ValueError('unexpected TRACE adapter source')
    # D retains every trajectory and always receives its bound demonstration.
    # Rewards never filter, select or alter this pure distillation objective.
    return dict(score=0., acc=0.)


if __name__ == '__main__':
    from kit.trace_prompt import install
    from kit.trace_native import TraceTaskRunner, nm_native
    install()
    nm_native.check_pinned()
    nm_native.main_ppo.TaskRunner = TraceTaskRunner
    nm_native.main_ppo.main()
