"""Diagnostic rewards only: task scorers unchanged; replay rewards never enter loss."""
import os


def _task_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'finqa':
        from kit.beds.v4_reward import compute_score as score
        return score(data_source, solution_str, ground_truth, extra_info)
    if data_source == 'p3b_replay': return {'score':0.}
    from kit.beds import code, math_screen
    module = code if data_source == code.DATA_SOURCE else math_screen
    return module.compute_score(data_source, solution_str, ground_truth, extra_info)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if os.environ.get('P3B_REPLAY') != '1':
        return _task_score(data_source, solution_str, ground_truth, extra_info)
    if data_source != 'p3b_replay':
        return dict(_task_score(data_source, solution_str, ground_truth, extra_info), p3b_replay=0)
    source = (extra_info or {}).get('p3b_task_source')
    # These are the existing task schemas, including FinQA's finish-gate fields.
    # Do not call a task verifier on generic text (code would need private tests).
    result = dict(score=0., acc=0., pred='', incorrect_format=0.)
    if source in ('finqa', 'livecodebench'): result['feedback'] = ''
    if source == 'finqa':
        from kit.beds.v4_reward import finish_gate, FinishGateBlind
        finish_gate()  # Preserve validation of KIT_FINISH_GATE, including gate=0.
        if not isinstance(extra_info, dict) or 'truncated' not in extra_info:
            raise FinishGateBlind('replay reward has no truncated flag')
        result.update(truncated=int(bool(extra_info['truncated'])), score_before_gate=0.)
    elif source not in ('math_screen', 'livecodebench'):
        raise ValueError('replay reward needs a supported p3b_task_source')
    return dict(result, p3b_replay=1)


def install_metrics(trainer):
    """Keep verl's training reward summaries task-only; raw rollouts stay marked."""
    if os.environ.get('P3B_REPLAY') != '1': return
    original = trainer.compute_data_metrics
    if getattr(original, '_p3b_task_only', False): return
    def task_metrics(batch, use_critic=True):
        import numpy as np
        result = original(batch=batch, use_critic=use_critic)
        flags = np.array([bool(e.get('p3b_replay')) for e in batch.non_tensor_batch['extra_info']])
        task = original(batch=batch[np.flatnonzero(~flags)], use_critic=use_critic)
        result.update({k:v for k,v in task.items() if k.startswith('critic/')})
        result.update({'p3b/task_rows':int((~flags).sum()), 'p3b/replay_rows':int(flags.sum())})
        return result
    task_metrics._p3b_task_only = True
    trainer.compute_data_metrics = task_metrics
