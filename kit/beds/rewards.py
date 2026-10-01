#!/usr/bin/env python3
"""ONE reward entry point for a training file whose rows come from more than one bed.

A rehearsal run trains on maths questions and SQL questions in the same batch, so one process has to
reward both. verl takes a single custom reward function, given as a FILE PATH and a function name:

    custom_reward_function.path=/work/kit/beds/rewards.py
    custom_reward_function.name=compute_score

It imports that file directly, not as a package, so `from . import spider` would fail and
`import spider` would depend on the working directory. The beds are therefore loaded the same way the
trainer loads this file: by path, from this file's own directory, at import time. Nothing here needs
the kit to be installed, on `sys.path`, or on any particular working directory.

Dispatch is on `data_source`, the field every bed writes into its own trainer rows, read from each
bed's own `DATA_SOURCE` constant rather than repeated here -- a bed that renames its data source
renames it everywhere at once. An unknown `data_source` RAISES: a mixed run in which one half of the
rows silently scored zero would look exactly like a rehearsal that did not work.

The return value is whatever the bed returned, unchanged: this file adds no key, drops none, and
rewrites no feedback, so the reward a row gets is the reward its own bed defines.

ONE exception, off by default: a LENGTH BUDGET. The dose probe (receipt 232) showed plain GRPO on Spider
drifting to 900-token answers after one pass over the questions, with the training reward flat and the
held-out score unmoved, so the count alone hid it. With the environment variable KIT_LENGTH_BUDGET_CHARS
set to an integer, an answer longer than that many characters scores 0 whatever the bed said, and the
returned dict carries `over_budget: 1` and a feedback line saying so. The bed's own verdict is kept in
`score_before_budget` so a readout can count how often the budget bit. Unset, nothing here changes.

A SECOND exception, also off by default: the FINISH GATE. K1c (receipts 241, 242) showed answers growing
from about 250 to 1,500-8,000 tokens during plain GRPO while the training reward held: the trainer cuts a
rollout at `data.max_response_length` and still rewards whatever answer the cut text happens to contain,
so the policy is never told that an answer must END. With KIT_FINISH_GATE=1, a rollout the trainer marks
as cut scores 0 whatever the bed said. The trainer supplies the mark itself: its reward manager sets
`extra_info["truncated"]` to True when the kept response ids contain no end-of-sequence token
(verl/workers/reward_manager/naive.py). The gate reads that key and nothing else, so it is exact on token
ids: an answer whose last kept token is the end token is finished, however long it is. The returned dict
then carries `truncated` (0 or 1) and the bed's own verdict in `score_before_gate`. If the gate is on and
the trainer did NOT supply the key, this file RAISES: a gate that cannot see whether the answer finished
would pay every cut answer and look like a control that did nothing. Unset, nothing here changes.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

BUDGET_ENV = "KIT_LENGTH_BUDGET_CHARS"
FINISH_ENV = "KIT_FINISH_GATE"

HERE = Path(__file__).resolve().parent
BED_FILES = {"spider": HERE / "spider.py", "gsm8k": HERE / "gsm8k.py", "finqa": HERE / "finqa.py", "code": HERE / "code.py"}


class UnknownDataSource(ValueError):
    """A row carries a `data_source` no bed in this directory claims."""


def _load_bed(name: str, path: Path):
    """Import one bed from its file, without a package and without touching sys.path."""
    if not path.is_file():
        raise FileNotFoundError("the %s bed is missing from %s (expected %s)" % (name, HERE, path.name))
    spec = importlib.util.spec_from_file_location("kit_bed_%s" % name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_beds() -> dict:
    """{data_source: module} for every bed beside this file. Two beds claiming one source is a refusal."""
    beds: dict = {}
    for name, path in sorted(BED_FILES.items()):
        module = _load_bed(name, path)
        source = getattr(module, "DATA_SOURCE", None)
        if not isinstance(source, str) or not source:
            raise ValueError("the %s bed has no DATA_SOURCE constant" % name)
        if source in beds:
            raise ValueError("two beds claim data_source %r: %s and %s" % (source, beds[source].__name__, name))
        beds[source] = module
    return beds


BEDS = _load_beds()
DATA_SOURCES = tuple(sorted(BEDS))


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None):
    """Reward one rollout with the bed that owns its `data_source`, and return exactly what that bed returns."""
    bed = BEDS.get(data_source) if isinstance(data_source, str) else None
    if bed is None:
        raise UnknownDataSource("no bed rewards data_source %r; this file dispatches %s"
                                % (data_source, ", ".join(DATA_SOURCES)))
    result = bed.compute_score(data_source, solution_str, ground_truth, extra_info)
    budget = length_budget()
    if budget is not None:
        result = apply_budget(result, solution_str, budget)
    if finish_gate():
        result = apply_finish_gate(result, extra_info)
    return result


class FinishGateBlind(ValueError):
    """The finish gate is on and the trainer did not say whether the answer was cut."""


def finish_gate() -> bool:
    """Whether KIT_FINISH_GATE asks for the gate. Only "1" turns it on; any other non-empty value is a refusal."""
    raw = os.environ.get(FINISH_ENV)
    if raw is None or raw == "" or raw == "0":
        return False
    if raw != "1":
        raise ValueError("%s must be 0 or 1, not %r" % (FINISH_ENV, raw))
    return True


def apply_finish_gate(result, extra_info):
    """Zero the score of a rollout the trainer cut before it ended; keep the bed's verdict beside it.

    Reads `extra_info["truncated"]`, which the trainer's reward manager sets from the response's token ids.
    Never mutates `result`. Raises when the key is absent, because then the gate would be blind."""
    if not isinstance(extra_info, dict) or "truncated" not in extra_info:
        raise FinishGateBlind("%s=1 but the trainer passed no `truncated` flag in extra_info, so a cut answer cannot "
                              "be told from a finished one; this trainer build does not support the gate" % FINISH_ENV)
    cut = bool(extra_info["truncated"])
    if not isinstance(result, dict):
        return 0.0 if cut else result
    out = dict(result)
    out["truncated"] = int(cut)
    out["score_before_gate"] = result.get("score")
    if not cut:
        return out
    out["score"] = 0.0
    if "acc" in out:
        out["acc"] = 0.0
    note = "The answer was cut off before it ended; an answer must finish within the response limit to count."
    out["feedback"] = (str(out.get("feedback") or "").strip() + " " + note).strip()
    return out


def length_budget():
    """The character budget from the environment, or None when unset. A bad value is a refusal."""
    raw = os.environ.get(BUDGET_ENV)
    if raw is None or raw == "":
        return None
    try:
        budget = int(raw)
    except ValueError as exc:
        raise ValueError("%s must be a positive integer, not %r" % (BUDGET_ENV, raw)) from exc
    if budget <= 0:
        raise ValueError("%s must be a positive integer, not %r" % (BUDGET_ENV, raw))
    return budget


def apply_budget(result, solution_str: str, budget: int):
    """Zero the score of an answer over the budget; keep the bed's verdict beside it. Never mutates `result`."""
    length = len(solution_str or "")
    if not isinstance(result, dict):
        return result if length <= budget else 0.0
    out = dict(result)
    out["answer_chars"] = length
    out["score_before_budget"] = result.get("score")
    if length <= budget:
        out["over_budget"] = 0
        return out
    out["over_budget"] = 1
    out["score"] = 0.0
    if "acc" in out:
        out["acc"] = 0.0
    note = "The answer is %d characters, over the budget of %d; a correct answer must be shorter." % (length, budget)
    out["feedback"] = (str(out.get("feedback") or "").strip() + " " + note).strip()
    return out


if __name__ == "__main__":
    print("kit/beds/rewards.py dispatches: %s" % ", ".join(DATA_SOURCES))
