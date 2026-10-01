#!/usr/bin/env python3
"""What the trainer's own training rollouts looked like, step by step: rewards, their spread, lengths, cuts, loops.

    python rollout_stats.py --run $WORK/runs/g8-chem-r1-a1 --model /work/models/qwen3-8b \\
                            --out $WORK/k8b/report-rollouts/g8-chem-r1-a1

The trainer dumps every training rollout to `<run>/rollouts/<step>.jsonl` (with `trainer.rollout_data_dir` set, as
the pilot launchers always set it): one JSON object a line, 256 lines a step (32 prompts x 8 attempts), each with at
least `input` (the prompt text), `output` (the answer text), `score` (the reward) and `step`, plus whatever keys the
reward function returned (`acc`, `incorrect_format`, sometimes `truncated`). This file reads those dumps and nothing
else, and says, for every step in step order:

    rollouts, prompts          lines in the file; distinct `input` texts
    score mean                 the mean reward
    groups                     of the prompt groups (the attempts at one prompt): the share whose attempts all got
                               the same score, all scored zero, all scored above zero. A group with no spread gives a
                               group-relative method nothing to learn from, so these say whether the step had signal
    tokens                     answer length in TOKENS (the model's tokenizer, `encode` without special tokens):
                               mean, median, p90 (nearest rank) and max, over all answers and separately over answers
                               with score > 0 and with score == 0 (a negative score, if a reward gives one, is in
                               neither split and is counted in `negative_scores`)
    at cap                     the share of answers whose length is at or over the cap, overall and by the same split
    truncated                  the share of answers whose `truncated` key is true, among the answers that have one;
                               absent (null) when no answer has the key
    loops                      the median over answers of len(zlib.compress(text)) / len(text in UTF-8 bytes), lower
                               meaning more repetitive (empty answers are left out); and the share of answers in which
                               the most common non-empty line (compared after stripping) occurs at least twice AND
                               makes up at least 30 percent of the non-empty lines. The "at least twice" is there
                               because without it every answer of three lines or fewer would count as a loop, and a
                               tool call (Thought / Action / Action Input) is exactly three lines

`totals` gives the same numbers over every rollout of the run, with a prompt group being one prompt at one step.

THE CAP is `--cap` when given, else `max_response_length` from the run's train-summary.json or run-summary.json
(whichever is present first and records it), else 8192; where it came from is written as `cap_source`. An answer at
the cap was very probably cut by the trainer; the share says how often.

Descriptive only: nothing here is a score of the model, and the trainer's rewards are those of the training reward
function, whatever it was. Only `load_tokenizer` imports transformers (tests replace it); everything else is the
standard library. Nothing is overwritten: an existing --out is refused, and so is a run with no rollouts folder.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-rollout-stats.v1"
DEFAULT_CAP = 8192
SUMMARIES = ("train-summary.json", "run-summary.json")
LOOP_LINE_SHARE = (3, 10)          # the most common non-empty line is at least 3/10 of the non-empty lines ...
LOOP_MIN_REPEATS = 2               # ... and occurs at least twice


class RolloutStatsError(ValueError):
    """The run cannot be summarised; nothing is written."""


def load_tokenizer(model_dir):
    """The model's own tokenizer. The ONLY place this file imports transformers (tests replace it)."""
    from transformers import AutoTokenizer                                  # noqa: PLC0415
    return AutoTokenizer.from_pretrained(str(model_dir))


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------------------- reading
def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_cap(run: Path, given) -> tuple:
    """(cap, where it came from): --cap, else a summary's `max_response_length`, else DEFAULT_CAP."""
    if given is not None:
        return int(given), "--cap"
    for name in SUMMARIES:
        path = run / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("max_response_length")
        except (ValueError, AttributeError):
            value = None
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, "%s max_response_length" % name
    return DEFAULT_CAP, "default %d (no summary records max_response_length)" % DEFAULT_CAP


def step_files(run: Path) -> list:
    """[(step, path)] for every `<run>/rollouts/<step>.jsonl`, in step order."""
    folder = run / "rollouts"
    if not folder.is_dir():
        raise RolloutStatsError("%s has no rollouts folder: the trainer writes one only with trainer.rollout_data_dir "
                                "set, so there is nothing to summarise" % run)
    files = sorted((int(path.stem), path) for path in folder.glob("*.jsonl") if path.stem.isdigit())
    if not files:
        raise RolloutStatsError("%s holds no <step>.jsonl file: there is nothing to summarise" % folder)
    return files


def read_rows(path: Path) -> list:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise RolloutStatsError("%s line %d is not JSON: %s" % (path, number, exc)) from exc
        if not isinstance(row, dict) or "input" not in row or "output" not in row or not _number(row.get("score")):
            raise RolloutStatsError("%s line %d lacks an `input`, an `output` or a numeric `score`: it is not a "
                                    "trainer rollout line" % (path, number))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------- measures
def share(count: int, total: int):
    return round(count / total, 6) if total else None


def distribution(values: list) -> dict:
    """mean, median, p90 (nearest rank, in integers) and max; None throughout when there is nothing."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    rank = -((-9 * len(ordered)) // 10)                                       # ceil(0.9 n), no float rounding
    return {"n": len(ordered), "mean": round(statistics.fmean(ordered), 3), "median": statistics.median(ordered),
            "p90": ordered[rank - 1], "max": ordered[-1]}


def compress_ratio(text: str):
    """len(zlib.compress(bytes)) / len(bytes); None for an empty answer. Lower is more repetitive."""
    raw = text.encode("utf-8")
    return len(zlib.compress(raw)) / len(raw) if raw else None


def is_looping(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False
    count = Counter(lines).most_common(1)[0][1]
    return count >= LOOP_MIN_REPEATS and count * LOOP_LINE_SHARE[1] >= len(lines) * LOOP_LINE_SHARE[0]


def _text(row: dict) -> str:
    value = row.get("output")
    return "" if value is None else str(value)


def summarise(rows: list, lengths: list, groups_of: list, cap: int) -> dict:
    """Every number above for one set of rollouts; `groups_of[i]` names the prompt group of rows[i]."""
    scores = [float(row["score"]) for row in rows]
    groups: dict = defaultdict(list)
    for key, score in zip(groups_of, scores):
        groups[key].append(score)
    split = {"all": list(range(len(rows))),
             "positive": [i for i, s in enumerate(scores) if s > 0],
             "zero": [i for i, s in enumerate(scores) if s == 0]}
    flags = [bool(row["truncated"]) for row in rows if "truncated" in row]
    ratios = [r for r in (compress_ratio(_text(row)) for row in rows) if r is not None]
    return {"rollouts": len(rows), "prompts": len({str(row["input"]) for row in rows}), "prompt_groups": len(groups),
            "score_mean": round(statistics.fmean(scores), 6) if scores else None,
            "negative_scores": sum(1 for s in scores if s < 0),
            "groups": {"n": len(groups),
                       "same_score_share": share(sum(1 for g in groups.values() if len(set(g)) == 1), len(groups)),
                       "all_zero_share": share(sum(1 for g in groups.values() if all(s == 0 for s in g)), len(groups)),
                       "all_positive_share": share(sum(1 for g in groups.values() if all(s > 0 for s in g)), len(groups))},
            "tokens": {name: distribution([lengths[i] for i in index]) for name, index in split.items()},
            "at_cap_share": {name: share(sum(1 for i in index if lengths[i] >= cap), len(index)) for name, index in split.items()},
            "truncated_share": share(sum(flags), len(flags)) if flags else None, "truncated_known": len(flags),
            "compress_ratio_median": round(statistics.median(ratios), 6) if ratios else None,
            "looping_share": share(sum(1 for row in rows if is_looping(_text(row))), len(rows))}


def build(args) -> dict:
    run = Path(args.run)
    if not run.is_dir():
        raise RolloutStatsError("no such run folder: %s" % run)
    files = step_files(run)
    cap, cap_source = read_cap(run, args.cap)
    if cap <= 0:
        raise RolloutStatsError("--cap must be a positive whole number, got %d" % cap)
    tokenizer = load_tokenizer(args.model)
    steps, every_row, every_length, every_group = [], [], [], []
    for step, path in files:
        rows = read_rows(path)
        if not rows:
            steps.append({"step": step, "file": path.name, "rollouts": 0})
            continue
        lengths = [count_tokens(tokenizer, _text(row)) for row in rows]
        steps.append({"step": step, "file": path.name,
                      **summarise(rows, lengths, [str(row["input"]) for row in rows], cap)})
        every_row += rows
        every_length += lengths
        every_group += [(step, str(row["input"])) for row in rows]
    totals = {"steps": len(steps), **summarise(every_row, every_length, every_group, cap)}
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run": str(run.resolve()), "model": str(args.model), "cap": cap, "cap_source": cap_source,
            "loop_rule": {"line_share_at_least": LOOP_LINE_SHARE[0] / LOOP_LINE_SHARE[1], "min_repeats": LOOP_MIN_REPEATS},
            "steps_read": len(steps),          # a plain number at the top level, for a campaign bar
            "steps": steps, "totals": totals}


# ------------------------------------------------------------------------------------------ render
def _pct(value) -> str:
    return "-" if value is None else "%.1f%%" % (100 * value)


def _num(value, pattern="%.1f") -> str:
    return "-" if value is None else pattern % value


def _row(label, s: dict) -> str:
    if not s.get("rollouts"):
        return "| %s | 0 |" % label + " - |" * 13
    g, t, c = s["groups"], s["tokens"], s["at_cap_share"]
    return "| %s | %d | %d | %s | %s / %s / %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        label, s["rollouts"], s["prompts"], _num(s["score_mean"], "%.3f"),
        _pct(g["same_score_share"]), _pct(g["all_zero_share"]), _pct(g["all_positive_share"]),
        _num(t["all"]["mean"]), _num(t["all"]["median"]), _num(t["all"]["p90"], "%d"), _num(t["all"]["max"], "%d"),
        _num(t["positive"]["mean"]), _num(t["zero"]["mean"]), _pct(c["all"]), _pct(s["truncated_share"]),
        _num(s["compress_ratio_median"], "%.3f"), _pct(s["looping_share"]))


def render(report: dict) -> str:
    lines = ["# Training rollouts: %s" % report["run"], "",
             "One row per training step, read from the trainer's own rollout dumps. Groups: of the prompt groups (the "
             "attempts at one prompt), the share whose attempts all got the same score / all scored zero / all scored "
             "above zero. Lengths are in tokens. At cap: answers at or over %d tokens (%s). zlib: the median compressed "
             "size over raw size, lower being more repetitive. Looping: answers whose most common non-empty line occurs "
             "at least twice and is at least 30 percent of their non-empty lines." % (report["cap"], report["cap_source"]), "",
             "| step | rollouts | prompts | score mean | groups same / all 0 / all >0 | tokens mean | median | p90 | max "
             "| mean, score > 0 | mean, score = 0 | at cap | truncated | zlib | looping |",
             "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [_row(str(s["step"]), s) for s in report["steps"]]
    lines.append(_row("all", report["totals"]))
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="What the trainer's training rollouts looked like, step by step.")
    parser.add_argument("--run", required=True, help="a run folder holding rollouts/<step>.jsonl")
    parser.add_argument("--model", required=True, help="a directory holding the trained model's tokenizer")
    parser.add_argument("--cap", type=int, default=None,
                        help="the training answer cap; default: max_response_length from the run's summary, else %d" % DEFAULT_CAP)
    parser.add_argument("--out", required=True, help="a new directory")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: choose a new --out" % out)
    try:
        report = build(args)
    except RolloutStatsError as exc:
        raise SystemExit(str(exc))
    out.mkdir(parents=True)
    (out / "rollout-stats.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "rollout-stats.md").write_text(render(report), encoding="utf-8")
    totals = report["totals"]
    print("%d steps, %d rollouts; cap %d (%s); answers at the cap %s" % (totals["steps"], totals["rollouts"], report["cap"],
                                                                        report["cap_source"], _pct(totals["at_cap_share"]["all"])))
    print("wrote", out / "rollout-stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
