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

THE GATE AND THE REDUCED RECORDS (package 4; send-5 review round 2, finding 9; amendment 3 B8). Each step and `totals`
also carry `gate`: the mean reward after the gate (`score`, what the trainer used), the mean `score_before_gate` (the
authors' own verdict), the cut share (`truncated`), and the prompt groups with no positive reward after and before the
gate. A run whose rollouts lack `score_before_gate` / `truncated` gets "not recorded" there; but a GATED run (its
summary says finish_gate 1) must carry both on EVERY rollout: `gate_protocol.ok` is false otherwise, and the package-4
report marks the run "not evaluated" (a protocol failure, not optional missingness). Beside the report goes
`rollout-rows.jsonl.gz`, one line per rollout: attempt, step, group (sha256 of the prompt text: one prompt at one step),
rollout (its line in the step file), rollout_in_group, tokens, score, score_before_gate, truncated. Every summary here
can be recomputed from it; it holds no text. Written with mtime 0, so equal rollouts give equal bytes.

    python rollout_stats.py --runs-of KEY --work WORK --model M --out-root WORK/k8b4/report-rollouts --summary S.json

summarises EVERY attempt WORK/runs/KEY-a<N> not yet summarised, each into OUT-ROOT/KEY-a<N>/ (an attempt that dumped no
rollouts gets a record saying so, steps_read 0), and writes S.json: the attempts found and whether each has records.

RECONCILED WITH THE TRAINER (round-4 ruling G5). Every record carries `rollout_evidence` (`reconcile`): for a run that
completed (a complete export, or run-summary returncode 0) each step 1..steps of the trainer's own metrics.jsonl must
hold exactly prompts x attempts rollouts (data.train_batch_size x actor_rollout_ref.rollout.n, from the run's recorded
command or resolved configuration); missing or partial dumps are "rollout evidence incomplete", and `complete` is 0.
An empty dump of a completed run is never complete.

POSITIVE, DURABLE, ATTEMPT-BOUND EVIDENCE OF "NO ROLLOUT" (round-6 ruling J3, replacing round 5's H3). A failed attempt
with zero reduced rows is "no rollout before the failure (evidenced)" ONLY if
  (i)  the wrapper's attempt record shows the launcher was never started (refused by the gate, the frozen check, the
       idle-GPU check, the containment setting or the watchdog start: `launcher_started` False); or
  (ii) the launcher's own progress record `env/stage.json` exists, is bound to this attempt (its `attempt` is the run's
       name and, when the wrapper recorded one, its `job` is the attempt's KIT_P4_JOB marker), its last stage is before
       `trainer-invoked` (`started` or `config-resolved`), AND it holds the `exited` entry the launcher's EXIT trap
       appends: the launcher itself recorded that it ended before invoking the trainer.
In EVERY other case -- no env/stage.json, no `exited` entry, a record of another attempt, the trainer invoked (with or
without metrics), metrics empty or recording no step, the run folder missing although the launcher started -- the
attempt's completeness is "rollout evidence unavailable": `complete` is 0 and the package-4 report gives the block no
registered label. A trainer that was invoked and crashed before its first dumped step therefore costs its block the
labels (amendment 3 B8). Absence is never evidence: a lost start marker does not prove non-invocation, and no logged
step does not prove that no rollout happened before the failure.

Descriptive only: nothing here is a score of the model, and the trainer's rewards are those of the training reward
function, whatever it was. Only `load_tokenizer` imports transformers (tests replace it); everything else is the
standard library. Nothing is overwritten: an existing --out is refused, and so is a run with no rollouts folder.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
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


def gate_summary(rows: list, groups_of: list) -> dict:
    """Rewards after and before the finish gate, the cut share, and the groups with no positive reward on either side."""
    after = [float(row["score"]) for row in rows]
    before = [row.get("score_before_gate") for row in rows]
    cut = [row.get("truncated") for row in rows]
    has_before = bool(rows) and all(_number(v) for v in before)
    has_cut = bool(rows) and all(v is not None and not isinstance(v, str) for v in cut)
    groups_after, groups_before = defaultdict(list), defaultdict(list)
    for key, a, b in zip(groups_of, after, before):
        groups_after[key].append(a)
        groups_before[key].append(b)
    n = len(groups_after)
    none_after = sum(1 for g in groups_after.values() if all(v <= 0 for v in g))
    out = {"score_after_gate_mean": round(statistics.fmean(after), 6) if after else None,
           "groups": n, "groups_no_reward_after_gate": {"count": none_after, "share": share(none_after, n)}}
    if has_before:
        none_before = sum(1 for g in groups_before.values() if all(float(v) <= 0 for v in g))
        out.update({"score_before_gate_mean": round(statistics.fmean(float(v) for v in before), 6),
                    "groups_no_reward_before_gate": {"count": none_before, "share": share(none_before, n)}})
    else:
        out.update({"score_before_gate_mean": "not recorded", "groups_no_reward_before_gate": "not recorded"})
    out["cut_share"] = share(sum(1 for v in cut if v), len(cut)) if has_cut else "not recorded"
    return out


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
            "looping_share": share(sum(1 for row in rows if is_looping(_text(row))), len(rows)),
            "gate": gate_summary(rows, groups_of)}


ROWS_FILE = "rollout-rows.jsonl.gz"


def _attempt_of(run: Path):
    match = re.search(r"-a(\d+)$", run.name)
    return int(match.group(1)) if match else None


def _gated(run: Path) -> bool:
    for name in SUMMARIES:
        try:
            if json.loads((run / name).read_text(encoding="utf-8")).get("finish_gate") in (1, "1", True):
                return True
        except (OSError, ValueError, AttributeError):
            continue
    return False


def reduced_row(attempt, step: int, index: int, in_group: int, row: dict, tokens: int) -> dict:
    """One rollout as the archive keeps it: identities, length and rewards, no text (amendment 3 B8)."""
    flag = row.get("truncated")
    before = row.get("score_before_gate")
    return {"attempt": attempt, "step": step, "group": hashlib.sha256(str(row["input"]).encode("utf-8")).hexdigest(),
            "rollout": index, "rollout_in_group": in_group, "tokens": tokens, "score": float(row["score"]),
            "score_before_gate": float(before) if _number(before) else None,
            "truncated": (int(bool(flag)) if flag is not None and not isinstance(flag, str) else None)}


def write_rows(path: Path, rows: list) -> str:
    """rollout-rows.jsonl.gz with mtime 0 (equal rollouts, equal bytes); returns its sha256."""
    raw = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(raw)
    Path(path).write_bytes(buffer.getvalue())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


# ------------------------------------------------------------------- round-4 ruling G5: reconciled with the trainer
def _export_complete(folder: Path) -> bool:
    """The launchers' own test of a merged model: config.json, tokenizer_config.json, every shard (or one
    model.safetensors), none empty (kit/p4_budget.py `complete_export`, repeated here: this file imports no kit module)."""
    if not ((folder / "config.json").is_file() and (folder / "tokenizer_config.json").is_file()):
        return False
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError, AttributeError):
            return False
        return bool(shards) and all((folder / s).is_file() and (folder / s).stat().st_size > 0 for s in shards)
    single = folder / "model.safetensors"
    return single.is_file() and single.stat().st_size > 0


def run_completed(run: Path) -> tuple:
    """(True, why) when the training run completed: a complete export (hf-step*) or run-summary.json returncode 0."""
    run = Path(run)
    exports = sorted(p.name for p in run.glob("hf-step*") if p.is_dir() and _export_complete(p))
    if exports:
        return True, "a complete export (%s)" % ", ".join(exports)
    try:
        summary = json.loads((run / "run-summary.json").read_text())
    except (OSError, ValueError):
        summary = None
    if isinstance(summary, dict) and summary.get("returncode") == 0 and not isinstance(summary.get("returncode"), bool):
        return True, "run-summary.json returncode 0"
    return False, "no complete export and run-summary.json returncode %r" % ((summary or {}).get("returncode") if isinstance(summary, dict) else "absent")


def metrics_steps(run: Path):
    """The training steps the trainer's own metrics.jsonl records (file-logger lines {"step": N, "data": ...}, N >= 1;
    step 0 is the validation before training), sorted and distinct; None when there is no metrics file."""
    path = Path(run) / "metrics.jsonl"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):              # unreadable counts as missing (it used to raise)
        return None
    steps = set()
    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            step = json.loads(line).get("step")
        except (ValueError, AttributeError):
            continue
        if isinstance(step, int) and not isinstance(step, bool) and step >= 1:
            steps.add(step)
    return sorted(steps)


def expected_rollouts(run: Path) -> tuple:
    """(prompts per step x attempts per prompt, source) from the run's recorded command (env/argv.txt:
    data.train_batch_size and actor_rollout_ref.rollout.n), else its resolved configuration; (None, why) otherwise."""
    run, keys = Path(run), {}
    try:
        for line in (run / "env" / "argv.txt").read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("-"):
                key, _sep, value = line.partition("=")
                keys[key] = value
    except OSError:
        pass
    batch, n = keys.get("data.train_batch_size"), keys.get("actor_rollout_ref.rollout.n")
    source = "env/argv.txt"
    if batch is None or n is None:
        try:
            import yaml                                                      # noqa: PLC0415
            doc = yaml.safe_load((run / "env" / "resolved-config.yaml").read_text(encoding="utf-8"))
            batch = batch if batch is not None else ((doc or {}).get("data") or {}).get("train_batch_size")
            n = n if n is not None else (((doc or {}).get("actor_rollout_ref") or {}).get("rollout") or {}).get("n")
            source = "env/argv.txt and env/resolved-config.yaml"
        except Exception:                                                    # noqa: BLE001 (no file, no PyYAML, bad YAML)
            pass
    try:
        batch, n = int(str(batch)), int(str(n))
    except (TypeError, ValueError):
        return None, "the run's recorded command and resolved configuration name no data.train_batch_size and rollout n"
    if batch < 1 or n < 1:
        return None, "data.train_batch_size %d and rollout n %d are not positive" % (batch, n)
    return batch * n, "%s: data.train_batch_size %d x actor_rollout_ref.rollout.n %d" % (source, batch, n)


EVIDENCE_COMPLETE, EVIDENCE_INCOMPLETE, EVIDENCE_NONE_BEFORE_FAILURE, EVIDENCE_UNAVAILABLE = (
    "complete", "rollout evidence incomplete", "no rollout before the failure (evidenced)", "rollout evidence unavailable")
#: the statuses under which an attempt's reduced records are complete
EVIDENCE_OK = (EVIDENCE_COMPLETE, EVIDENCE_NONE_BEFORE_FAILURE)
#: both launchers write it immediately before invoking the trainer (round-5 ruling H3; kept, no longer evidence by itself)
TRAINER_STARTED = Path("env") / "trainer-started-at.txt"
#: both launchers' own durable progress record (round-6 ruling J3), packed by kit/collect.py
STAGE_FILE = Path("env") / "stage.json"
STAGES_BEFORE_TRAINER = ("started", "config-resolved")
STAGES_OF_THE_TRAINER = ("trainer-invoked", "trainer-exited", "merged")


def stage_evidence(run: Path, job_marker=None) -> tuple:
    """Round-6 ruling J3 (ii): (True, why) when the launcher's own env/stage.json is bound to this attempt (its
    `attempt` is the run folder's name; its `job` is `job_marker` when the wrapper recorded one), its last stage is
    before `trainer-invoked`, and its EXIT trap's `exited` entry closes it; else (False, what is missing)."""
    path = Path(run) / STAGE_FILE
    try:
        doc = json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        return False, "no %s: the launcher left no record of its own progress" % STAGE_FILE
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return False, "%s is unreadable (%s)" % (STAGE_FILE, exc)
    entries = doc.get("stages") if isinstance(doc, dict) else None
    if not isinstance(entries, list) or not all(isinstance(e, dict) and isinstance(e.get("stage"), str) for e in entries):
        return False, "%s holds no list of stages" % STAGE_FILE
    if doc.get("attempt") != Path(run).name:
        return False, "%s belongs to attempt %r, not %s" % (STAGE_FILE, doc.get("attempt"), Path(run).name)
    if job_marker and doc.get("job") != job_marker:
        return False, "%s was written under job marker %r, not this attempt's %s" % (STAGE_FILE, doc.get("job"), job_marker)
    reached = [e["stage"] for e in entries if e["stage"] != "exited"]
    exited = [e for e in entries if e["stage"] == "exited"]
    if any(s in STAGES_OF_THE_TRAINER for s in reached):
        return False, "the launcher recorded that it invoked the trainer (stages %s)" % ", ".join(reached)
    if not exited:
        return False, "%s has no `exited` entry: the launcher did not record its own end (stages %s)" % (STAGE_FILE, ", ".join(reached) or "none")
    if not reached or reached[-1] not in STAGES_BEFORE_TRAINER or exited[-1].get("last_stage") != reached[-1] \
            or entries[-1].get("stage") != "exited":
        return False, "%s is not the record of a launcher that ended before the trainer (stages %s; exited after %r)" % (
            STAGE_FILE, ", ".join(reached) or "none", exited[-1].get("last_stage"))
    return True, "the launcher recorded its stages %s and then its own exit (status %r) before invoking the trainer" % (
        ", ".join(reached), exited[-1].get("returncode"))


def reconcile(run: Path, steps_of_rows: list, export_complete: bool = False, launcher_started=None, job_marker=None) -> dict:
    """Round-4 ruling G5: the reduced rollout records of one attempt against what the trainer did. `steps_of_rows` is
    the `step` of every reduced row. For a run that COMPLETED (a complete export, or run-summary returncode 0) every
    step 1..steps recorded in metrics.jsonl must hold exactly prompts x attempts rollouts, the metrics must record
    steps 1..its last step without a gap (the run-summary's `steps` when it records them), and no rollout may be
    dumped at an unrecorded step. A run that did NOT complete may hold no rollout at all only with the positive,
    attempt-bound evidence of round-6 ruling J3 (see the module's text), else its status is "rollout evidence
    unavailable"; otherwise each recorded step needs its full count, and only the step after the last recorded one may
    hold a (full) dump. `status` is "complete" only when all of that holds. `export_complete`: kit/p4_run.py's attempt
    record says the export was complete (an extracted archive holds no hf-step folder). `launcher_started`: what the
    wrapper's attempt record says (False: the launcher was never started; None: no record). `job_marker`: the
    attempt's KIT_P4_JOB marker from that record, which a bound env/stage.json must carry."""
    run = Path(run)
    if launcher_started is False and not steps_of_rows:
        return {"completed": False, "completed_why": "the launcher was never started", "metrics_steps": None, "expected_per_step": None,
                "expected_source": None, "rows": 0, "problems": [], "status": EVIDENCE_NONE_BEFORE_FAILURE,
                "why": "(i) the wrapper's attempt record shows the launcher was never started", "evidence": "launcher never started"}
    if not run.is_dir():
        return {"completed": False, "completed_why": "no run folder", "metrics_steps": None, "expected_per_step": None,
                "expected_source": None, "rows": len(steps_of_rows), "status": EVIDENCE_UNAVAILABLE,
                "problems": ["the run folder %s is missing although the launcher started: the attempt's completeness is unavailable" % run.name],
                "evidence": "run folder missing"}
    completed, completed_why = run_completed(run)
    if not completed and export_complete:              # an archive holds no weights: the wrapper's record of the export
        completed, completed_why = True, "the wrapper's attempt record: a complete export"
    steps = metrics_steps(run)
    expected, source = expected_rollouts(run)
    counts = Counter(steps_of_rows)
    out = {"completed": completed, "completed_why": completed_why, "metrics_steps": len(steps) if steps is not None else None,
           "expected_per_step": expected, "expected_source": source, "rows": sum(counts.values()), "problems": []}
    problems = out["problems"]
    if not completed and not counts and not steps:
        # round-6 ruling J3: zero rows from a failed attempt need the launcher's own, attempt-bound record that it ENDED
        # before invoking the trainer; the absence of anything (a start marker, metrics, a logged step) is no evidence.
        # Metrics that record a training step contradict every such claim: those fall through to "incomplete" below.
        bound, why = stage_evidence(run, job_marker)
        if bound:
            out.update({"status": EVIDENCE_NONE_BEFORE_FAILURE, "evidence": "launcher ended before the trainer",
                        "why": "(ii) the attempt did not complete (%s); %s" % (completed_why, why)})
            return out
        out.update({"status": EVIDENCE_UNAVAILABLE, "evidence": "no positive evidence",
                    "problems": ["the attempt failed (%s) with no rollout dumped and no training step recorded, and nothing evidences that "
                                 "it rolled nothing out before the failure (%s): the attempt's completeness is unavailable"
                                 % (completed_why, why)]})
        return out
    if steps is None:
        problems.append("the trainer's metrics.jsonl is missing: the dumped rollouts cannot be reconciled with the steps it ran")
        steps = []
    if expected is None:
        problems.append(source)
    if completed:
        if not steps:
            problems.append("a completed run whose metrics.jsonl records no training step")
        elif steps != list(range(1, steps[-1] + 1)):
            problems.append("metrics.jsonl records steps %s..%s with gaps" % (steps[0], steps[-1]))
        try:
            declared = json.loads((run / "run-summary.json").read_text()).get("steps")
        except (OSError, ValueError, AttributeError):
            declared = None
        if isinstance(declared, int) and steps and steps[-1] != declared:
            problems.append("metrics.jsonl's last step is %d, the run-summary says %d steps" % (steps[-1], declared))
    allowed_extra = set() if completed or not steps else {steps[-1] + 1}
    if not completed and not steps:
        allowed_extra = {1}
    if expected is not None:
        missing = [st for st in steps if counts.get(st, 0) == 0]
        partial = [{"step": st, "rows": counts[st], "expected": expected} for st in steps if 0 < counts.get(st, 0) != expected]
        unlogged = sorted(st for st in counts if st not in set(steps) and (st not in allowed_extra or counts[st] != expected))
        if missing:
            problems.append("no dumped rollout for %d recorded step(s) (%s)" % (len(missing), ", ".join(map(str, missing[:8]))))
        if partial:
            problems.append("partial dumps: %s" % ", ".join("step %(step)d %(rows)d of %(expected)d" % p for p in partial[:8]))
        if unlogged:
            problems.append("rollouts dumped at step(s) the metrics do not record: %s" % ", ".join(map(str, unlogged[:8])))
        out.update({"steps_missing": missing, "steps_partial": partial, "steps_unlogged": unlogged})
    out["status"] = EVIDENCE_INCOMPLETE if problems else EVIDENCE_COMPLETE
    return out


def build(args, rows_out: list | None = None) -> dict:
    run = Path(args.run)
    if not run.is_dir():
        raise RolloutStatsError("no such run folder: %s" % run)
    attempt, gated = _attempt_of(run), _gated(run)
    cap, cap_source = read_cap(run, args.cap)
    if cap <= 0:
        raise RolloutStatsError("--cap must be a positive whole number, got %d" % cap)
    dumped = (run / "rollouts").is_dir() and any(p.stem.isdigit() for p in (run / "rollouts").glob("*.jsonl"))
    if getattr(args, "allow_empty", False) and not dumped:
        empty = summarise([], [], [], cap)
        evidence = reconcile(run, [])
        return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "run": str(run.resolve()), "attempt": attempt, "model": str(args.model), "cap": cap, "cap_source": cap_source,
                "no_rollouts": "the attempt dumped no rollouts (%s holds no <step>.jsonl): there is nothing to reduce" % (run / "rollouts"),
                "steps_read": 0, "steps": [], "totals": {"steps": 0, **empty}, "rows_file": ROWS_FILE, "rows": 0,
                # G5: an empty dump is never "complete" for a run that completed or that recorded training steps
                "rollout_evidence": evidence, "complete": int(evidence["status"] in EVIDENCE_OK),
                "gate_protocol": {"gated": gated, "rows_missing_gate_fields": 0,
                                  "ok": evidence["status"] == EVIDENCE_NONE_BEFORE_FAILURE if gated else True}}
    files = step_files(run)
    tokenizer = load_tokenizer(args.model)
    known: dict = {}                                     # equal answers have equal lengths: each text is tokenized once

    def tokens_of(text: str) -> int:
        if text not in known:
            known[text] = count_tokens(tokenizer, text)
        return known[text]
    steps, every_row, every_length, every_group, reduced = [], [], [], [], []
    for step, path in files:
        rows = read_rows(path)
        if not rows:
            steps.append({"step": step, "file": path.name, "rollouts": 0})
            continue
        lengths = [tokens_of(_text(row)) for row in rows]
        steps.append({"step": step, "file": path.name,
                      **summarise(rows, lengths, [str(row["input"]) for row in rows], cap)})
        seen = Counter()
        for index, (row, length) in enumerate(zip(rows, lengths)):
            reduced.append(reduced_row(attempt, step, index, seen[str(row["input"])], row, length))
            seen[str(row["input"])] += 1
        every_row += rows
        every_length += lengths
        every_group += [(step, str(row["input"])) for row in rows]
    totals = {"steps": len(steps), **summarise(every_row, every_length, every_group, cap)}
    missing = sum(1 for r in reduced if r["truncated"] is None or r["score_before_gate"] is None)
    evidence = reconcile(run, [r["step"] for r in reduced])
    if rows_out is not None:
        rows_out.extend(reduced)
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run": str(run.resolve()), "attempt": attempt, "model": str(args.model), "cap": cap, "cap_source": cap_source,
            "loop_rule": {"line_share_at_least": LOOP_LINE_SHARE[0] / LOOP_LINE_SHARE[1], "min_repeats": LOOP_MIN_REPEATS},
            "steps_read": len(steps),          # a plain number at the top level, for a campaign bar
            "steps": steps, "totals": totals, "rows_file": ROWS_FILE, "rows": len(reduced),
            "rollout_evidence": evidence, "complete": int(evidence["status"] in EVIDENCE_OK),
            "gate_protocol": {"gated": gated, "rows_missing_gate_fields": missing if gated else None,
                              "ok": (missing == 0) if gated else True,
                              "rule": "a gated run's every rollout carries `truncated` and `score_before_gate` (amendment 3 B8)"}}


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
def write_one(args, out: Path) -> dict:
    rows = []
    report = build(args, rows)
    out.mkdir(parents=True)
    report["rows_sha256"] = write_rows(out / ROWS_FILE, rows)
    (out / "rollout-stats.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "rollout-stats.md").write_text(render(report), encoding="utf-8")
    return report


def all_attempts(args) -> int:
    """--runs-of KEY: every attempt WORK/runs/KEY-a<N> summarised once into OUT-ROOT/KEY-a<N>; the summary file says
    which attempts exist and whether each has its reduced records."""
    runs = sorted((int(p.name[len(args.runs_of) + 2:]), p) for p in (Path(args.work) / "runs").glob("%s-a*" % args.runs_of)
                  if p.is_dir() and p.name[len(args.runs_of) + 2:].isdigit())
    found = []
    for attempt, run in runs:
        out = Path(args.out_root) / run.name
        entry = {"attempt": attempt, "run": "runs/%s" % run.name, "records": "%s/%s" % (out.name, ROWS_FILE)}
        if not out.exists():
            ns = argparse.Namespace(run=str(run), model=args.model, cap=args.cap, allow_empty=True)
            try:
                write_one(ns, out)
                entry["written"] = True
            except RolloutStatsError as exc:
                entry["error"] = str(exc)
        entry["has_records"] = (out / ROWS_FILE).is_file() and (out / "rollout-stats.json").is_file()
        try:
            entry["rollout_evidence"] = (json.loads((out / "rollout-stats.json").read_text()).get("rollout_evidence") or {}).get("status")
        except (OSError, ValueError):
            entry["rollout_evidence"] = None
        found.append(entry)
    summary = {"schema": SCHEMA + "/attempts", "key": args.runs_of, "attempts": found, "attempts_found": len(found),
               "attempts_without_records": sum(1 for e in found if not e["has_records"]),
               "attempts_with_incomplete_evidence": sum(1 for e in found if e.get("rollout_evidence") != EVIDENCE_COMPLETE
                                                        and e.get("rollout_evidence") != EVIDENCE_NONE_BEFORE_FAILURE)}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("%s: %d attempt(s), %d without reduced records; wrote %s" % (args.runs_of, len(found), summary["attempts_without_records"], args.summary))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="What the trainer's training rollouts looked like, step by step.")
    parser.add_argument("--run", help="a run folder holding rollouts/<step>.jsonl")
    parser.add_argument("--model", required=True, help="a directory holding the trained model's tokenizer")
    parser.add_argument("--cap", type=int, default=None,
                        help="the training answer cap; default: max_response_length from the run's summary, else %d" % DEFAULT_CAP)
    parser.add_argument("--out", help="a new directory")
    parser.add_argument("--allow-empty", action="store_true", help="a run that dumped no rollouts gets a record saying so")
    parser.add_argument("--runs-of", help="a run key: summarise every attempt WORK/runs/KEY-a<N> not yet summarised")
    parser.add_argument("--work")
    parser.add_argument("--out-root")
    parser.add_argument("--summary")
    args = parser.parse_args(argv)
    if args.runs_of:
        if not (args.work and args.out_root and args.summary):
            raise SystemExit("--runs-of needs --work, --out-root and --summary")
        return all_attempts(args)
    if not (args.run and args.out):
        raise SystemExit("give --run and --out (or --runs-of)")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: choose a new --out" % out)
    try:
        report = write_one(args, out)
    except RolloutStatsError as exc:
        raise SystemExit(str(exc))
    totals = report["totals"]
    print("%d steps, %d rollouts; cap %d (%s); answers at the cap %s" % (totals["steps"], totals["rollouts"], report["cap"],
                                                                        report["cap_source"], _pct(totals["at_cap_share"]["all"])))
    print("wrote", out / "rollout-stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
