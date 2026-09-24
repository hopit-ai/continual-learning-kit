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
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

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
    return bed.compute_score(data_source, solution_str, ground_truth, extra_info)


if __name__ == "__main__":
    print("kit/beds/rewards.py dispatches: %s" % ", ".join(DATA_SOURCES))
