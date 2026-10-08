#!/usr/bin/env python3
"""The FINISH GATE on the authors' own tasks (Chemistry, ToolAlpaca): plan v3 package 4's `cap-gate` intervention.

WHAT IT DOES
------------
The authors' reward function (`verl/utils/reward_score/feedback/__init__.py` of the pinned lasgroup/SDPO checkout)
never looks at whether an answer ended: the trainer cuts a rollout at `data.max_response_length` and the authors'
checker scores whatever the cut text happens to contain. This file wraps that function, unchanged, and:

  * ALWAYS returns a dict carrying `truncated` (0 or 1, the trainer's own mark) and `score_before_gate` (the authors'
    verdict), so every training rollout records whether it ended and what the authors would have paid;
  * with KIT_FINISH_GATE=1, pays 0 for an answer the trainer marks as cut: `score` 0.0, `acc` 0.0 when present, and
    a feedback line saying why. An answer that ended keeps every key and value of the authors' result unchanged,
    with only the two keys above added.

The gate logic itself is kit/beds/rewards.py (`finish_gate`, `apply_finish_gate`, `FinishGateBlind`), loaded by path
and reused, so the Spider/GSM8K gate and this one are the same code.

WHERE THE MARK COMES FROM
-------------------------
The pinned trainer's reward manager (verl/workers/reward_manager/naive.py:84) sets `extra_info["truncated"]` to True
when the kept response ids hold no end-of-sequence token, and passes `extra_info` to this function. The gate reads that
key and nothing else; it never guesses from the length of the text. If the key is absent this file RAISES
`FinishGateBlind` -- with the gate on because a blind gate would pay every cut answer, and with the gate off too,
because the one thing the file is then for is recording that flag as 0 or 1, and it will not invent one.

A BARE NUMBER
-------------
The pinned authors' function returns a dict for every task it knows. If it ever returns a bare number x, this file
wraps it as {"score": x} first and then adds the two keys, so the trainer always receives a dict and the dumped
rollouts always carry `truncated` and `score_before_gate`.

HOW IT FINDS THE AUTHORS' FUNCTION
----------------------------------
verl loads this file by path (custom_reward_function.path), not as a package, so the authors' function is loaded by
path too: `$SDPO_DIR/verl/utils/reward_score/feedback/__init__.py`. If SDPO_DIR is unset or the file is not there it
RAISES at import, which is when the trainer loads the reward function, before any GPU work. It never falls back to a
copy of its own. The authors' file imports `verl.utils.reward_score.feedback.*` absolutely, which the launchers make
importable by putting $SDPO_DIR on PYTHONPATH.

The launchers (kit/run_grpo_toolalpaca.sh, kit/run_sdpo_toolalpaca.sh) point the trainer here ONLY with FINISH_GATE=1,
so a control's command is the pilot's, unchanged.
"""
from __future__ import annotations

import importlib.util
import numbers
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUTHORS_RELATIVE = Path("verl") / "utils" / "reward_score" / "feedback" / "__init__.py"


class AuthorsRewardMissing(RuntimeError):
    """The pinned checkout's own reward function could not be found; nothing may be scored."""


def _load_by_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authors_file() -> Path:
    """`$SDPO_DIR/verl/utils/reward_score/feedback/__init__.py`, or a refusal saying what is missing."""
    sdpo_dir = os.environ.get("SDPO_DIR")
    if not sdpo_dir:
        raise AuthorsRewardMissing("SDPO_DIR is not set, so the authors' reward function cannot be found. Set it to the "
                                   "pinned lasgroup/SDPO checkout. This file never scores with a copy of its own.")
    path = Path(sdpo_dir) / AUTHORS_RELATIVE
    if not path.is_file():
        raise AuthorsRewardMissing("the authors' reward function is not at %s (SDPO_DIR=%s): is that the pinned "
                                   "lasgroup/SDPO checkout? This file never scores with a copy of its own." % (path, sdpo_dir))
    return path


_GATE = _load_by_path("kit_beds_rewards_for_authors_gate", HERE / "rewards.py")
FinishGateBlind = _GATE.FinishGateBlind
FINISH_ENV = _GATE.FINISH_ENV
finish_gate = _GATE.finish_gate
apply_finish_gate = _GATE.apply_finish_gate

AUTHORS_FILE = authors_file()
AUTHORS = _load_by_path("sdpo_reference_feedback_for_authors_gate", AUTHORS_FILE)


def _as_dict(result):
    """The authors' result as a dict: a dict as it is, a bare number x as {"score": x}. Anything else is a refusal."""
    if isinstance(result, dict):
        return result
    if isinstance(result, numbers.Number):
        return {"score": result}
    raise TypeError("the authors' compute_score returned %r (%s), neither a dict nor a number"
                    % (result, type(result).__name__))


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info: dict | None = None) -> dict:
    """verl's entry point: the authors' verdict, unchanged for an answer that ended, plus `truncated` and
    `score_before_gate`; with KIT_FINISH_GATE=1, zero for an answer the trainer cut."""
    gate = finish_gate()                       # a bad KIT_FINISH_GATE value is refused before anything is scored
    if not isinstance(extra_info, dict) or "truncated" not in extra_info:
        raise FinishGateBlind("%s=%s but the trainer passed no `truncated` flag in extra_info, so a cut answer cannot be "
                              "told from a finished one; this file records that flag and will not guess it from the text"
                              % (FINISH_ENV, "1" if gate else "0"))
    result = _as_dict(AUTHORS.compute_score(data_source, solution_str, ground_truth, extra_info))
    if gate:
        return apply_finish_gate(result, extra_info)
    out = dict(result)
    out["truncated"] = int(bool(extra_info["truncated"]))
    out["score_before_gate"] = result.get("score")
    return out


if __name__ == "__main__":
    print("kit/beds/authors_gate.py wraps %s; gate %s" % (AUTHORS_FILE, "ON" if finish_gate() else "off"))
