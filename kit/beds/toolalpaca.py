#!/usr/bin/env python3
"""The SDPO authors' tool-use task (ToolAlpaca, their `datasets/tooluse`), as a scored bed.

The task K0, K4a and K1c trained on. Until plan v3 it was scored only from the trainer's own validation dumps
(sampled, sixteen answers a question); this bed lets kit/eval_bed.py score it the way every other bed is scored:
one greedy answer a question, deterministically, on one GPU, with the token ids kept. 4,046 training and 68 test
questions; the answer format (`Thought:` / `Action:` / `Action Input:` with a JSON object) is given inside the user
prompt, and there is no system message. `--root` is `<SDPO_DIR>/datasets/tooluse`.

SCORING is the authors' `verl/utils/reward_score/feedback/tooluse.py::compute_score`, unchanged: every
`Action: <name>` in the answer is collected, every `Action Input: {...}` is parsed and merged into one dictionary,
and the answer is right only if the multiset of action names and the merged inputs both equal the gold's. Two
properties of that rule matter when reading a score. Its pattern for the input object is non-greedy
(`Action Input:\\s*({.*?})`), so an object with a nested object is cut at the first closing brace, fails to parse and
is silently dropped: a correct nested call is scored wrong. And all `Action:` lines count, so an answer that
revises itself is scored on every action it wrote. `incorrect_format` is the authors' own (1 when the answer has no
`Action:` followed by `Action Input:`).

Sixty-eight questions resolve a difference of about twelve points at best; this bed is a stress test of the
answer format, not a fine retention measurement.

This bed is NOT registered in kit/beds/rewards.py: training uses the authors' own reward dispatcher through the two
ToolAlpaca launchers, as it always has.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("kit_beds_authors", Path(__file__).resolve().parent / "_authors.py")
authors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(authors)

DATA_SOURCE = "tooluse"
SPLITS = ("train", "test")


def load(root, split: str) -> list:
    if split not in SPLITS:
        raise authors.AuthorsBedError("the toolalpaca bed has no %r split (it has %s)" % (split, ", ".join(SPLITS)))
    items = authors.load(root, split, DATA_SOURCE)
    for item in items:
        try:
            gold = json.loads(item["gold"])
        except ValueError:
            gold = None
        if not isinstance(gold, list) or not gold or not all(isinstance(call, dict) and "Action" in call for call in gold):
            raise authors.AuthorsBedError("%s has a ground truth that is not a JSON list of tool calls" % item["id"])
    return items


def render_prompt(item: dict) -> list:
    """The authors' chat messages (a single user turn). eval_bed renders them with the trainer's template call."""
    return item["messages"]


def gold_of(item: dict) -> str:
    return item["gold"]


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    return dict(authors.scorer("tooluse").compute_score(solution_str or "", ground_truth))
