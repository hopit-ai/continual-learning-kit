#!/usr/bin/env python3
"""Plan-v4 training dispatch, loaded by file path by verl.

Chemistry calls the pinned authors' mcq.compute_score (the sciknoweval branch of
their dispatcher); FinQA calls the kit bed. Both reuse authors_gate's finish
contract: require the trainer's truncated flag even with KIT_FINISH_GATE off,
record score_before_gate, and zero cut rollouts only when the gate is on.
Loading mcq directly avoids importing unrelated code/math checker dependencies.
"""
from __future__ import annotations

import importlib.util
import numbers
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GATE = _load("v4_finish_gate", HERE / "rewards.py")
FINQA = _load("v4_reward_finqa", HERE / "finqa.py")
FinishGateBlind = _GATE.FinishGateBlind
finish_gate = _GATE.finish_gate
apply_finish_gate = _GATE.apply_finish_gate

sdpo = os.environ.get("SDPO_DIR")
if not sdpo:
    raise RuntimeError("SDPO_DIR is not set; v4 Chemistry requires the pinned authors' mcq.py")
AUTHORS_FILE = Path(sdpo) / "verl/utils/reward_score/feedback/mcq.py"
if not AUTHORS_FILE.is_file():
    raise RuntimeError("the pinned authors' Chemistry scorer is missing: %s" % AUTHORS_FILE)
AUTHORS = _load("v4_reward_authors_mcq", AUTHORS_FILE)


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info: dict | None = None) -> dict:
    """Score one Chemistry/FinQA rollout and preserve the finish mark and ungated verdict."""
    gate = finish_gate()
    if not isinstance(extra_info, dict) or "truncated" not in extra_info:
        raise FinishGateBlind("KIT_FINISH_GATE=%d but extra_info has no truncated flag; v4 reward will not guess" % gate)
    if data_source == "sciknoweval":
        result = AUTHORS.compute_score(solution_str or "", ground_truth)
    elif data_source == FINQA.DATA_SOURCE:
        result = FINQA.compute_score(data_source, solution_str, ground_truth, extra_info)
    else:
        raise ValueError("v4 reward does not support data_source %r" % data_source)
    if isinstance(result, numbers.Number):
        result = {"score": result}
    if not isinstance(result, dict):
        raise TypeError("v4 scorer returned neither a dict nor a number: %r" % (result,))
    if gate:
        return apply_finish_gate(result, extra_info)
    return {**result, "truncated": int(bool(extra_info["truncated"])), "score_before_gate": result.get("score")}
