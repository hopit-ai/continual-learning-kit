#!/usr/bin/env python3
"""Partial credit on ToolAlpaca, for TRAINING ONLY. The K4a `soft` arm.

THE RULE, in plain words
------------------------
The authors' checker is all-or-nothing: you score 1 if you call exactly the right tools with exactly
the right arguments, and 0 otherwise. This file keeps that verdict and adds a second, softer number
beside it.

    exactly right, by the authors' own checker      ->  1.0
    otherwise, EVERY tool right but not every argument
                                                    ->  the share of the arguments you got right
    otherwise (any tool wrong, missing or spurious) ->  0.0

so that

  * an answer the authors' checker calls CORRECT scores exactly 1.0, as before;
  * the right tools with three of four arguments right scores 0.75;
  * the right tools with half the arguments right scores 0.50;
  * the right tools with one of three arguments right scores 0.33;
  * the right tools with no argument right scores 0.0;
  * calling half the right tools, however perfectly, scores 0.0;
  * no `Action:` / `Action Input:` pair at all scores 0.0.

WHAT THAT NUMBER ACTUALLY DECIDES
---------------------------------
In SDPO the reward does exactly one thing: it decides which of a question's 8 attempts is eligible
to be shown to the teacher as a worked example. The eligibility line is `score >= 0.5`
(`success_reward_threshold`, verl/trainer/config/actor/actor.yaml:99 of the pinned reference). The
SDPO loss itself has no reward-weighted term at all -- see docs/phase2/k4a/feasibility.md (b).

So this rule is not "partial reward for partial work". It is a rule for one decision, and the rule
above states that decision directly:

    an attempt is shown to the teacher when it calls EXACTLY the right tools
    and gets AT LEAST HALF of the arguments right.

Calling only some of the right tools is not a near-miss -- it is a different answer -- so it never
qualifies, whatever its arguments look like. Anything else stays below 0.5 and is ignored, exactly
as in K0.

Two consequences, declared rather than hidden:

  * an attempt promoted this way is a WRONG answer being shown as a worked example;
  * SDPO takes the FIRST eligible sibling, not the best one (ray_trainer.py:665 of the reference),
    so on a question that already had a correct attempt, a near-miss with a lower index can be shown
    instead. This arm therefore perturbs the questions K0 already solves as well as the stuck ones.

Both are the point of the measurement, not a defect of the file.

THE VALIDATION SCORE IS NOT TOUCHED
-----------------------------------
verl builds the training and the validation reward function from the same config key
(main_ppo.py:328-333), so there is no way to soften one without the other -- and no need. The
number K0 reports, `val-core/tooluse/acc/mean@16`, is built from the `acc` key, not from the reward
(ray_trainer.py:946). This file returns the authors' `acc`, `pred`, `incorrect_format` and
`feedback` **verbatim** and changes only `score`. The strict all-or-nothing validation score is
therefore bit-for-bit K0's. tests/test_kit_k4a.py asserts that against the real checker.

HOW IT FINDS THE AUTHORS' CHECKER
---------------------------------
It never reimplements it. verl loads this file by path, not as a package, so the checker is loaded
by path too: `$SDPO_TOOLUSE`, else `$SDPO_DIR/verl/utils/reward_score/feedback/tooluse.py`, else an
importable `verl`. If none is there it RAISES. A soft reward that silently fell back to a private
copy of the parser would be the quietest way imaginable to make K4a incomparable with K0.

    self-test (no GPU, no trainer):
        SDPO_DIR=/work/SDPO python kit/beds/tooluse_soft.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

DATA_SOURCE = "tooluse"
#: an attempt scoring at least this is eligible as a demonstration; the reference's own default,
#: repeated here only so the rule above can be read without opening the trainer's config. It is what
#: makes "at least half the arguments" the line: the score IS the share of arguments right.
SUCCESS_REWARD_THRESHOLD = 0.5
PASSTHROUGH_KEYS = ("acc", "pred", "incorrect_format", "feedback")


class AuthorsCheckerMissing(RuntimeError):
    """The reference's own tooluse checker could not be found; nothing may be scored."""


def _candidate_paths() -> list:
    paths = []
    explicit = os.environ.get("SDPO_TOOLUSE")
    if explicit:
        paths.append(Path(explicit))
    sdpo_dir = os.environ.get("SDPO_DIR")
    if sdpo_dir:
        paths.append(Path(sdpo_dir) / "verl" / "utils" / "reward_score" / "feedback" / "tooluse.py")
    return paths


def load_authors_checker():
    """The reference's `verl/utils/reward_score/feedback/tooluse.py`, loaded by path or imported."""
    for path in _candidate_paths():
        if path.is_file():
            spec = importlib.util.spec_from_file_location("sdpo_reference_tooluse", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    try:
        from verl.utils.reward_score.feedback import tooluse                 # noqa: PLC0415
        return tooluse
    except ImportError as exc:
        raise AuthorsCheckerMissing(
            "cannot find the reference's tooluse checker. Set SDPO_DIR to the pinned lasgroup/SDPO "
            "checkout (or SDPO_TOOLUSE to the file itself), or run inside the trainer, where verl is "
            "importable. Tried: %s. This file never scores with a copy of its own."
            % (", ".join(str(p) for p in _candidate_paths()) or "nothing: neither variable is set")
        ) from exc


AUTHORS = load_authors_checker()


# ------------------------------------------------------------------------------------ the rule
def _canonical(value) -> str:
    """One string per argument value, so {'a': 1} and {'a': 1} match and {'a': '1'} does not."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def every_tool_right(predicted: list, expected: list) -> bool:
    """The authors' own test for the tool calls (tooluse.py:86): the same names, the same number of
    times, in any order. Nothing missing, nothing spurious."""
    return Counter(predicted) == Counter(expected)


def input_share(predicted: dict, expected: dict) -> float:
    """How much of the argument dict was right. Both empty is nothing to get wrong, so 1.0."""
    longest = max(len(predicted), len(expected))
    if longest == 0:
        return 1.0
    matched = sum(1 for key, value in expected.items()
                  if key in predicted and _canonical(predicted[key]) == _canonical(value))
    return matched / longest


def partial_credit(solution_str: str, strict: dict, expected_actions: list, expected_inputs: dict) -> float:
    """The number at the top of this file. `strict` is the authors' own verdict for this answer."""
    if strict["acc"] >= 1.0:
        return 1.0
    if not every_tool_right(AUTHORS.extract_actions(solution_str), expected_actions):
        return 0.0
    score = input_share(AUTHORS.extract_action_inputs(solution_str), expected_inputs)
    # 1.0 is reserved for answers the authors' checker calls correct. Reaching it here would mean
    # every tool AND every argument matched, which IS that case, so this clamp should never fire. It
    # is here so that a future change to the share cannot quietly invent a second kind of perfect
    # answer -- one that SDPO would then treat as a demonstration on equal terms with a real one.
    return min(round(score, 6), 0.99)


def expected_from_ground_truth(ground_truth) -> tuple:
    """The reference's own parse of a row's ground truth: (actions, merged argument dict)."""
    try:
        gt_list = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except json.JSONDecodeError:
        return [], {}
    if not isinstance(gt_list, list):
        return [], {}
    actions, inputs = [], []
    for item in gt_list:
        actions.append(item.get("Action"))
        raw = item.get("Action_Input")
        try:
            inputs.append(json.loads(raw) if isinstance(raw, str) else raw)
        except (json.JSONDecodeError, TypeError):
            inputs.append({})
    return actions, AUTHORS.merge_action_inputs([i for i in inputs if isinstance(i, dict)])


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info: dict = None) -> dict:
    """verl's entry point. Same five keys as the reference; only `score` differs."""
    if data_source != DATA_SOURCE:
        raise ValueError(
            "kit/beds/tooluse_soft.py was given data_source %r, but it only scores %r. A run whose "
            "rows silently scored zero would look exactly like a soft reward that did not help."
            % (data_source, DATA_SOURCE))
    strict = AUTHORS.compute_score(solution_str, ground_truth)
    expected_actions, expected_inputs = expected_from_ground_truth(ground_truth)
    result = {key: strict[key] for key in PASSTHROUGH_KEYS}
    result["score"] = partial_credit(solution_str, strict, expected_actions, expected_inputs)
    return result


# ------------------------------------------------------------------------------------ self-test
def _selftest() -> int:
    gt = json.dumps([
        {"Action": "get_weather", "Action_Input": json.dumps({"city": "Zurich", "unit": "C"})},
        {"Action": "send_email", "Action_Input": json.dumps({"to": "ada@example.com"})},
    ])
    right = [("get_weather", {"city": "Zurich", "unit": "C"}), ("send_email", {"to": "ada@example.com"})]

    def answer(actions_and_inputs) -> str:
        return "\n".join("Action: %s\nAction Input: %s" % (a, json.dumps(i)) for a, i in actions_and_inputs)

    cases = [
        ("exactly right", answer(right), 1.0, 1.0),
        ("right tools, two of three arguments",
         answer([("get_weather", {"city": "Zurich", "unit": "F"}), ("send_email", {"to": "ada@example.com"})]),
         0.0, 2 / 3),
        ("right tools, no argument right",
         answer([("get_weather", {"city": "Bern", "unit": "F"}), ("send_email", {"to": "bob@example.com"})]),
         0.0, 0.0),
        ("one of the two tools, its arguments right",
         answer([("get_weather", {"city": "Zurich", "unit": "C"})]), 0.0, 0.0),
        ("a spurious extra tool call",
         answer(right + [("delete_all", {})]), 0.0, 0.0),
        ("no tool call at all", "I think it is sunny in Zurich.", 0.0, 0.0),
    ]
    bad = 0
    for name, solution, want_acc, want_score in cases:
        got = compute_score(DATA_SOURCE, solution, gt)
        ok = abs(got["acc"] - want_acc) < 1e-6 and abs(got["score"] - want_score) < 1e-6
        bad += not ok
        shown = "shown to the teacher" if got["score"] >= SUCCESS_REWARD_THRESHOLD else "ignored"
        print("%-38s acc %.1f score %.2f  %-20s %s"
              % (name, got["acc"], got["score"], shown,
                 "ok" if ok else "WANTED acc %.1f score %.2f" % (want_acc, want_score)))
    print("checker loaded from: %s" % getattr(AUTHORS, "__file__", AUTHORS.__name__))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_selftest())
