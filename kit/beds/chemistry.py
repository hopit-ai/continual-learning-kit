#!/usr/bin/env python3
"""The SDPO authors' Science Q&A task (SciKnowEval level 3; Chemistry is the domain plan v3 uses), as a scored bed.

Four-option multiple choice. The data are the authors' own, inside the pinned checkout
(`datasets/sciknoweval/<domain>/`, 1,890 training and 210 test questions for chemistry, a disjoint split made by
their `data/split_tasks.py`). The system prompt, which is in the data, asks for

    <reasoning> ... </reasoning>
    <answer> ... </answer>

with the answer a single letter. `--root` is the domain directory after `data/preprocess.py` has written its
parquet files. The prompt is the authors' message list, rendered with the same chat template call the trainer uses
(thinking off), so this bed asks exactly what training asked.

SCORING is the authors' `verl/utils/reward_score/feedback/mcq.py::compute_score`, unchanged: the prediction is the
text after the LAST `<answer>` up to the next `</answer>`, stripped, and it is right only if it equals the gold
letter exactly. An answer with no `<answer>` tag is scored on its whole text, which is never a letter, so it is
wrong. Two things this bed adds around that call, neither of which changes a score:

- `incorrect_format` here is 1 when the answer contains no `<answer>` tag at all. The authors' own
  `incorrect_format` is inverted (their `is_correct_format` is true for a well-formed answer and the value is
  passed through as "incorrect"); it is kept beside ours as `authors_incorrect_format` so their number can still
  be reproduced.
- `DATA_SOURCE` is the dataset's own `data_source` field, "sciknoweval", for every domain.

This bed is NOT registered in kit/beds/rewards.py: training on this task uses the authors' own reward dispatcher
through kit/run_grpo_toolalpaca.sh and kit/run_sdpo_toolalpaca.sh with DATASET set, exactly as their sweep does.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("kit_beds_authors", Path(__file__).resolve().parent / "_authors.py")
authors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(authors)

DATA_SOURCE = "sciknoweval"
SPLITS = ("train", "test")
LETTERS = ("A", "B", "C", "D")


def load(root, split: str) -> list:
    if split not in SPLITS:
        raise authors.AuthorsBedError("the chemistry bed has no %r split (it has %s)" % (split, ", ".join(SPLITS)))
    items = authors.load(root, split, DATA_SOURCE)
    bad = [item["id"] for item in items if item["gold"] not in LETTERS]
    if bad:
        raise authors.AuthorsBedError("%d questions have a gold answer that is not one of %s (e.g. %s)" % (len(bad), "/".join(LETTERS), bad[0]))
    return items


def render_prompt(item: dict) -> list:
    """The authors' chat messages (system, then user). eval_bed renders a message list with the trainer's template call."""
    return item["messages"]


def gold_of(item: dict) -> str:
    return item["gold"]


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    result = dict(authors.scorer("mcq").compute_score(solution_str or "", ground_truth))
    result["authors_incorrect_format"] = result.get("incorrect_format")
    result["incorrect_format"] = 0 if "<answer>" in (solution_str or "") else 1
    return result
