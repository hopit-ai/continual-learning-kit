#!/usr/bin/env python3
"""Pinned native NM entry; precomposed immutable configuration only."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def compute_score(data_source,solution_str,ground_truth,extra_info=None):
    if data_source=='finqa':
        from kit.beds import finqa
        extra_info=extra_info or {}
        if 'truncated' not in extra_info:raise ValueError('FinQA reward requires finish-gate telemetry')
        score=int(not extra_info['truncated'] and finqa.is_correct(finqa.extract_answer(solution_str),ground_truth))
        return dict(score=score,acc=score,truncated=int(extra_info['truncated']))
    if data_source=='p3b_replay':return dict(score=0.,acc=0.)
    if data_source=='contract_nli':
        from kit.c_reward import compute_score as score
        return score(data_source,solution_str,ground_truth,extra_info)
    if data_source=='a4_sec':
        from kit.a4_reward import compute_score as score
        return score(data_source,solution_str,ground_truth,extra_info)
    raise ValueError('unregistered NM task reward source')


def main():
    from kit.nm_native import NMTaskRunner,main_ppo,check_pinned
    check_pinned();main_ppo.TaskRunner=NMTaskRunner;main_ppo.main()
if __name__=='__main__':main()
