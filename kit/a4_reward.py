"""Diagnostic self-trace reward; W's losses never depend on this correctness signal."""
from kit.beds import finqa


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    replay=bool((extra_info or {}).get('p3b_replay'))
    if data_source not in ('a4_sec','p3b_replay'): raise ValueError('wrong 4A source')
    result=finqa.compute_score('finqa',solution_str,ground_truth) if not replay else dict(score=0.,acc=0.,pred='',incorrect_format=0)
    return dict(result,p3b_replay=int(replay))
