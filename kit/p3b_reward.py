"""Diagnostic rewards only: task scorers unchanged; replay rewards never enter loss."""
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'finqa':
        from kit.beds.v4_reward import compute_score as score
        return score(data_source, solution_str, ground_truth, extra_info)
    if data_source == 'p3b_replay': return {'score':0.}
    from kit.beds import code, math_screen
    module = code if data_source == code.DATA_SOURCE else math_screen
    return module.compute_score(data_source, solution_str, ground_truth, extra_info)
