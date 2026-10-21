"""C diagnostic reward schema; D task loss and B replay CE are unchanged."""
from kit.c_common import legal


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    replay = bool((extra_info or {}).get('p3b_replay'))
    if data_source not in ('contract_nli', 'p3b_replay'): raise ValueError('wrong C reward source')
    correct = False if replay else legal.parse_answer('contract_nli', solution_str) == ground_truth
    return dict(score=float(correct), acc=float(correct), pred='' if replay else legal.parse_answer('contract_nli', solution_str) or '',
                incorrect_format=float(not replay and legal.parse_answer('contract_nli', solution_str) is None), p3b_replay=int(replay))
