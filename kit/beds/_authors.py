"""Shared by the two beds that score the SDPO authors' own tasks (kit/beds/chemistry.py, kit/beds/toolalpaca.py).

These beds add no task of ours. They read the trainer-format files the authors' `data/preprocess.py` writes inside
the pinned checkout (`<task>/train.parquet`, `<task>/test.parquet`: a `prompt` that is a list of chat messages, a
`reward_model.ground_truth`, an `extra_info.index`), and they score an answer by calling the authors' own scoring
function, loaded by file path from `$SDPO_DIR`. So a greedy, deterministic, one-GPU scoring by kit/eval_bed.py asks
the model exactly what the trainer asked it and marks the answer exactly as the trainer marked it.

`SDPO_DIR` is read when a score is computed, never at import: importing this file must not need the checkout,
because tests and the report tools import the beds without it.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

SCORER_DIR = "verl/utils/reward_score/feedback"
_SCORERS: dict = {}


class AuthorsBedError(ValueError):
    """The authors' data or scorer is not where this bed needs it."""


def scorer(name: str):
    """The authors' `<name>.py` scoring module (`mcq` or `tooluse`), loaded once from the pinned checkout."""
    if name not in _SCORERS:
        root = os.environ.get("SDPO_DIR")
        if not root:
            raise AuthorsBedError("set SDPO_DIR to the pinned lasgroup/SDPO checkout: this bed scores with the authors' own "
                                  "%s/%s.py and has no scorer of its own" % (SCORER_DIR, name))
        path = Path(root) / SCORER_DIR / ("%s.py" % name)
        if not path.is_file():
            raise AuthorsBedError("the authors' scorer is missing: %s (is SDPO_DIR the pinned checkout?)" % path)
        spec = importlib.util.spec_from_file_location("kit_authors_%s" % name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SCORERS[name] = module
    return _SCORERS[name]


def _rows(root: Path, split: str) -> list:
    """The rows of `<root>/<split>.parquet`, or of `<root>/<split>.jsonl` in the same schema (used by tests)."""
    parquet, jsonl = root / ("%s.parquet" % split), root / ("%s.jsonl" % split)
    if parquet.is_file():
        try:
            import pyarrow.parquet as pq                                          # noqa: PLC0415
        except ImportError:
            raise AuthorsBedError("%s is a parquet file and pyarrow is not installed" % parquet)
        return pq.read_table(parquet).to_pylist()
    if jsonl.is_file():
        return [json.loads(line) for line in jsonl.read_text(encoding="utf-8").split("\n") if line.strip()]
    raise AuthorsBedError("no %s.parquet under %s: run `python data/preprocess.py --data_source <task directory>` in the "
                          "pinned checkout first (the campaign's prepare step does it)" % (split, root))


def load(root, split: str, data_source: str) -> list:
    """[{'id', 'messages', 'gold'}] for one split of one task, in file order.

    Refuses a file that belongs to another task, a row without a user message or a ground truth, and repeated ids:
    each would otherwise be scored as an ordinary wrong answer."""
    root = Path(root)
    rows = _rows(root, split)
    if not rows:
        raise AuthorsBedError("%s %s holds no rows" % (root, split))
    items, seen = [], set()
    for number, row in enumerate(rows):
        source = row.get("data_source")
        if source != data_source:
            raise AuthorsBedError("%s %s row %d has data_source %r, not %r: this is another task's file"
                                  % (root, split, number, source, data_source))
        messages = [{"role": str(m["role"]), "content": str(m["content"])} for m in (row.get("prompt") or [])]
        if not messages or messages[-1]["role"] != "user" or not messages[-1]["content"].strip():
            raise AuthorsBedError("%s %s row %d has no user message to answer" % (root, split, number))
        gold = (row.get("reward_model") or {}).get("ground_truth")
        if gold is None or str(gold) == "":
            raise AuthorsBedError("%s %s row %d has no ground truth" % (root, split, number))
        index = (row.get("extra_info") or {}).get("index")
        key = "%s-%s-%s" % (data_source, split, index if index not in (None, "") else number)
        if key in seen:
            raise AuthorsBedError("%s %s repeats the id %s" % (root, split, key))
        seen.add(key)
        items.append({"id": key, "messages": messages, "gold": str(gold)})
    return items
