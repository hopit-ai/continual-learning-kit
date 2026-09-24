#!/usr/bin/env python3
"""Hints for the questions a model never solves: find them, ask a bigger model for a plan, check the
plan does not contain the answer, and put it where exactly one of the student and the teacher sees it.

    python hints.py stuck    --rows TRAIN.jsonl --model MODEL_DIR --out DIR      # ONE GPU, 8 attempts at T=1.0
    python hints.py generate --stuck DIR/stuck.json --rows TRAIN.jsonl --out DIR \\
                             --base-url http://localhost:8000/v1 --model Qwen/Qwen3-235B-A22B
    python hints.py filter   --hints DIR/hints.jsonl --rows TRAIN.jsonl --out DIR       # no GPU, no network
    python hints.py apply    --rows TRAIN.jsonl --hints DIR/hints-filtered.jsonl \\
                             --out DIR --fade-at-step 20 --tokenizer MODEL_DIR         # no GPU
    python hints.py feedback --hints DIR/hints-filtered.jsonl --rows TRAIN.jsonl       # no GPU: the reward file's self-check

WHY THIS FILE EXISTS. On a bed whose answers are checkable from the question (Spider, FinQA) the
untrained Qwen3-1.7B is stuck -- right in none of 8 attempts -- on a quarter to a third of the
training questions, while a frontier model answers most of them from the prompt alone
(`docs/phase2/evidence/k4-gate/gate.json`, receipt 227). K4 asks whether a short plan from a large
open model, given during training, lifts the student's UNAIDED held-out accuracy. Design note:
`docs/phase2/k4-design-20260924.md`. Feasibility, with the trainer's own line numbers:
`docs/phase2/k4/feasibility.md`.

THE FOUR STAGES ARE SEPARATE COMMANDS ON PURPOSE. `stuck` needs one GPU; `generate` needs a served
model and no GPU of its own; `filter` and `apply` need neither. Each writes a small JSON that the next
one reads and a campaign row can gate on, so a package that went wrong went wrong at a named stage
with its evidence on disk. Nothing is ever overwritten: an existing --out is refused.

WHERE THE HINT GOES, AND WHO SEES IT.

    `apply`     appends the hint to the PROMPT of the stuck rows and writes a new training file. The
                student reads it during training; at test time `kit/eval_bed.py` renders prompts from
                the bed's own data, so no measurement ever contains a hint. This is the `hint` and
                `hint-faded` arms, and `kit/run_grpo.sh` runs them unchanged.
    `feedback`  is a REWARD FUNCTION. It scores the attempt with the bed that owns the row, exactly as
                `kit/beds/rewards.py` does, and then returns the hint as the `feedback` string. In the
                pinned SDPO trainer that string reaches the teacher's prompt and nothing else
                (`verl/trainer/ppo/ray_trainer.py:629,728-745,762`), so the student never reads it.
                This is the `teacher-hint` arm, and `kit/run_sdpo_bed.sh` runs it.

LEAKAGE IS THE ONE THING THAT WOULD INVALIDATE K4, so `filter` is strict and counts what it drops:
on Spider a hint may name the gold query's tables and columns but may not contain the query, and may
not contain a complete SELECT statement of its own; on a numeric bed it may not contain the gold
number, nor any number within 1 percent of it at any of the three scales the bed accepts, nor state a
yes/no answer, nor write the bed's own `Answer:` line. Dropped hints are counted by reason, and the
campaign gates on the coverage that survives.

Standard library only, plus: pyarrow to write the parquet the trainer reads (`apply`), vLLM and
transformers inside `stuck`, and `transformers` inside `apply` only when a --tokenizer is given.
Nothing here imports anything from this programme's private packages.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "kit-hints.v1"
BED_FILES = {"spider": HERE / "beds" / "spider.py", "gsm8k": HERE / "beds" / "gsm8k.py",
             "finqa": HERE / "beds" / "finqa.py"}

# ---- the stuck set: the trainer's own sampling ------------------------------------------------------
# 8 attempts at temperature 1.0 is what `actor_rollout_ref.rollout.n=8` and verl's default temperature
# make the trainer do, and it is what the gate measured. A different number here would define a
# different stuck set from the one the trainer actually gets stuck on.
ATTEMPTS, TEMPERATURE, MAX_NEW = 8, 1.0, 1536
SAMPLING_ENGINE = {"gpu_memory_utilization": 0.85, "max_model_len": 8192, "seed": 0}

# ---- the hint-giver's instructions, fixed here so the request is auditable --------------------------
# It is asked for a PLAN, in a fixed number of lines, and forbidden the final answer. The filter does
# not trust this instruction: it checks. Both are recorded (this text's sha256 is in every manifest).
HINT_SYSTEM = (
    "You are helping a much smaller model learn to solve this kind of question on its own. "
    "You write a short plan, never an answer."
)
HINT_INSTRUCTION = (
    "Below is a question a small model gets wrong every time. Write a plan it could follow to reach "
    "the answer itself.\n"
    "\n"
    "Rules for your reply:\n"
    "- Three to five lines, one step a line, each line a short imperative sentence.\n"
    "- Say WHICH parts of the question to use (which table, which columns, which rows, which years) "
    "and IN WHAT ORDER to combine them.\n"
    "- Do NOT state the final answer, or any number, query or value that is the answer or is close "
    "to it.\n"
    "- Do not write a SQL statement, and do not write an `Answer:` line.\n"
    "- Reply with the plan only: no preamble, no heading, no explanation of these rules.\n"
    "\n"
    "The question:\n"
    "\n"
)
#: How the hint is joined to the question in the student's prompt. Fixed, and its sha256 is recorded:
#: a hinted run whose header moved is a different experiment from the one this header was measured in.
HINT_HEADER = "\n\nA hint from a stronger model. It is a plan, not the answer:\n"
HINT_MIN_LINES, HINT_MAX_LINES = 3, 5
GENERATE_DEFAULTS = {"max_tokens": 400, "temperature": 0.0, "timeout": 180.0, "retries": 3}
#: Reasoning models (Qwen3, DeepSeek-R1 servings) return their thinking inside `content` unless the chat
#: template is told not to think. Asked for through vLLM's `chat_template_kwargs`; a server that rejects
#: the field (HTTP 400) is asked again without it, and whatever thinking still comes back is stripped by
#: split_thinking() so that only the plan reaches the filter. Smoke attempt 2 lost 147 of 171 hints to
#: this: Qwen3-8B's thinking made every hint 10 to 40 lines.
THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
_THINK = re.compile(r"<think>[\s\S]*?</think>", re.I)
_THINK_OPEN = re.compile(r"<think>", re.I)
#: A plan written as one paragraph (smoke attempt 3: 141 of 171 Qwen3-8B hints were one line of three to five
#: sentences) is reshaped to one sentence a line before the line rules see it. Only a reply with fewer than
#: HINT_MIN_LINES lines is reshaped, so a plan already written as lines is never changed.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`'\"(])")

# ---- the leakage filter ----------------------------------------------------------------------------
NUMBER_TOLERANCE = 0.01                      # "within 1 percent", the design note's rule
_NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*|[-+]?\.\d+")
#: A complete SQL statement of the hint's own: `SELECT ... FROM <table>` on one line, either written in SQL's
#: upper case, or in code (backticks), or in the shape of code (nothing English between SELECT and FROM, no
#: determiner after FROM). The English sentence "Select the titles from the publication table" is a plan
#: step, not a statement: smoke attempt 4 lost 51 of 172 hints to a looser version of this rule.
_SELECT = re.compile(r"\bselect\b([^\n]{0,200}?)\bfrom\b[ \t]+([`'\"]?\w+)", re.I)
_SELECT_UPPER = re.compile(r"\bSELECT\b[^\n]{0,200}?\bFROM\b[ \t]+\S")
_SELECT_CODE = re.compile(r"`[^`\n]*\bselect\b[^`\n]{0,200}?\bfrom\b[^`\n]*`", re.I)
_ENGLISH = {"the", "a", "an", "each", "every", "all", "any", "this", "that", "these", "those", "its", "their",
            "of", "to", "for", "with", "in", "on", "by", "which", "whose", "who", "and", "or", "then", "only"}
_ANSWER_LINE = re.compile(r"\banswer\s*[:=]\s*\S", re.I)
_YES_NO = re.compile(r"\b(?:the\s+)?answer\s+(?:is|would\s+be)\s+(?:yes|no)\b", re.I)
_WHITESPACE = re.compile(r"\s+")
#: Every reason a hint can be dropped. Counted separately so a readout can say WHY coverage fell, and
#: so a bed-inappropriate rule (a yes/no check on Spider) is visibly zero rather than invisible.
DROP_REASONS = ("empty", "too_few_lines", "too_many_lines", "contains_gold_query",
                "contains_sql_statement", "contains_gold_number", "states_yes_no",
                "writes_answer_line", "no_such_question")


class HintsError(ValueError):
    """The request, the inputs or a hint is wrong; nothing is written."""


# ------------------------------------------------------------------------------------ small helpers
def load_bed(name: str):
    """Import one bed from its file, without a package and without touching sys.path."""
    path = BED_FILES[name]
    if not path.is_file():
        raise HintsError("the %s bed is missing (expected %s)" % (name, path))
    spec = importlib.util.spec_from_file_location("kit_bed_%s" % name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path) -> list:
    """Rows from a .jsonl (one object a line) or a .parquet file. `split` on newlines, never splitlines."""
    path = Path(path)
    if not path.is_file():
        raise HintsError("no such file: %s" % path)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq                                      # noqa: PLC0415
        except ImportError:
            raise HintsError("%s is parquet and pyarrow is not installed; pass the jsonl instead" % path) from None
        rows = pq.read_table(path).to_pylist()
    else:
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise HintsError("%s line %d is not JSON: %s" % (path, number, exc)) from exc
    if not rows:
        raise HintsError("%s has no rows" % path)
    if any(not isinstance(row, dict) for row in rows):
        raise HintsError("%s has a row that is not a JSON object" % path)
    return rows


def as_jsonl(rows: list) -> str:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_dir(path) -> Path:
    path = Path(path)
    if path.exists():
        raise HintsError("refusing to overwrite %s: an output is never replaced, choose a new --out" % path)
    path.mkdir(parents=True)
    return path


def write_new(path: Path, text: str) -> str:
    if path.exists():
        raise HintsError("refusing to overwrite %s" % path)
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def write_json(path: Path, payload: dict) -> None:
    write_new(path, json.dumps(payload, indent=1, sort_keys=True) + "\n")


def question_id(row: dict) -> str:
    """How a training row names its question: the bed's own id, in `extra_info.index`."""
    index = (row.get("extra_info") or {}).get("index")
    if not isinstance(index, str) or not index:
        raise HintsError("a training row has no `extra_info.index`: this is not a row one of this "
                         "kit's beds wrote, and a hint could not be attached to a question")
    return index


def prompt_of(row: dict) -> str:
    """The row's single user message, which is what the beds write and what the trainer reads."""
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], dict):
        raise HintsError("row %s does not carry exactly one prompt message" % question_id(row))
    content = prompt[0].get("content")
    if not isinstance(content, str) or not content:
        raise HintsError("row %s has an empty prompt" % question_id(row))
    return content


def data_source_of(rows: list) -> str:
    sources = sorted({str(row.get("data_source")) for row in rows})
    if len(sources) != 1:
        raise HintsError("these rows carry %d data sources (%s); one call is one bed"
                         % (len(sources), ", ".join(sources)))
    return sources[0]


def bed_named_by(data_source: str) -> str:
    for name in sorted(BED_FILES):
        if load_bed(name).DATA_SOURCE == data_source:
            return name
    raise HintsError("no bed in %s claims data_source %r" % (HERE / "beds", data_source))


def ground_truth_of(row: dict) -> str:
    truth = (row.get("reward_model") or {}).get("ground_truth")
    if truth is None:
        raise HintsError("row %s has no reward_model.ground_truth" % question_id(row))
    return str(truth)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rows_by_id(rows: list) -> dict:
    found: dict = {}
    for row in rows:
        key = question_id(row)
        if key in found:
            raise HintsError("two training rows carry the question id %s" % key)
        found[key] = row
    return found


# ================================================================================ 1. the stuck set
def wins_of(bed, row: dict, texts: list) -> int:
    """How many of this question's attempts the bed's own checker calls right."""
    truth, wins = ground_truth_of(row), 0
    for text in texts:
        try:
            result = bed.compute_score(row["data_source"], text, truth, row.get("extra_info"))
        except Exception:                                                    # noqa: BLE001
            # A checker that cannot score an attempt (unparseable SQL, a database error) means that
            # attempt was not right. It must not stop the sampling of 640 other questions.
            continue
        if float(result.get("score", 0.0)) == 1.0:
            wins += 1
    return wins


def stuck_record(rows: list, per_question: list, *, attempts: int, temperature: float, model: str,
                 seconds: float, extra: dict) -> dict:
    """Everything a campaign bar or a readout needs about one bed's stuck set."""
    stuck = [entry["index"] for entry in per_question if entry["wins"] == 0]
    total = len(per_question)
    return {"schema": SCHEMA, "stage": "stuck", "generated_at": now(),
            "data_source": data_source_of(rows), "model": str(model),
            "attempts": int(attempts), "temperature": float(temperature),
            "questions": total, "stuck": len(stuck),
            "stuck_share": round(len(stuck) / total, 6) if total else 0.0,
            "solved_every_time": sum(entry["wins"] == attempts for entry in per_question),
            "mean_win_rate": round(sum(entry["wins"] for entry in per_question) / (attempts * total), 6)
                             if total else 0.0,
            "wins_histogram": [sum(entry["wins"] == k for entry in per_question) for k in range(attempts + 1)],
            "stuck_ids": stuck, "seconds": round(seconds, 1), **extra}


def cmd_stuck(args) -> int:
    rows = read_jsonl(args.rows)
    if args.limit:
        rows = rows[: args.limit]
    bed = load_bed(bed_named_by(data_source_of(rows)))
    model = Path(args.model).resolve()
    if not (model / "config.json").is_file():
        raise HintsError("not a HuggingFace model directory: %s" % model)
    if Path(args.out).exists():                    # refused here, and the directory is made at the end,
        raise HintsError("refusing to overwrite %s: an output is never replaced, choose a new --out"
                         % args.out)               # so a crash in vLLM leaves nothing to block a retry
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer                                   # noqa: PLC0415
    from vllm import LLM, SamplingParams                                     # noqa: PLC0415
    import vllm                                                              # noqa: PLC0415
    tokenizer = AutoTokenizer.from_pretrained(str(model))
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt_of(row)}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for row in rows]
    engine = {**SAMPLING_ENGINE, "max_model_len": args.max_model_len or SAMPLING_ENGINE["max_model_len"]}
    llm = LLM(model=str(model), **engine)
    started = time.monotonic()
    outputs = llm.generate(prompts, SamplingParams(n=args.attempts, temperature=args.temperature,
                                                   max_tokens=args.max_new, seed=engine["seed"]))
    per_question, attempt_rows = [], []
    for row, output in zip(rows, outputs):
        texts = [completion.text for completion in output.outputs]
        wins = wins_of(bed, row, texts)
        per_question.append({"index": question_id(row), "wins": wins, "attempts": len(texts)})
        attempt_rows.append({"index": question_id(row), "wins": wins,
                             "attempts": [{"text": text[-2000:]} for text in texts]})
    record = stuck_record(rows, per_question, attempts=args.attempts, temperature=args.temperature,
                          model=model, seconds=time.monotonic() - started,
                          extra={"rows_file": str(Path(args.rows).resolve()),
                                 "rows_sha256": sha256_file(args.rows), "limit": int(args.limit or 0),
                                 "max_new_tokens": args.max_new, "engine": {**engine, "vllm": vllm.__version__},
                                 "per_question": per_question})
    out = fresh_dir(args.out)
    write_json(out / "stuck.json", record)
    write_new(out / "attempts.jsonl", as_jsonl(attempt_rows))
    print("%s: %d of %d questions stuck (0 of %d right), %d solved every time, mean win rate %.4f"
          % (record["data_source"], record["stuck"], record["questions"], record["attempts"],
             record["solved_every_time"], record["mean_win_rate"]))
    print("wrote", out / "stuck.json")
    return 0


# =============================================================== 2. asking the hint-giver for a plan
def hint_request(prompt: str) -> dict:
    """The chat request body, in the OpenAI-compatible shape every vLLM server serves."""
    return {"messages": [{"role": "system", "content": HINT_SYSTEM},
                         {"role": "user", "content": HINT_INSTRUCTION + prompt}]}


def post_chat(base_url: str, model: str, prompt: str, *, api_key: str | None, max_tokens: int,
              temperature: float, timeout: float, retries: int) -> dict:
    """One completion from an OpenAI-compatible /chat/completions endpoint. Standard library only."""
    import urllib.error                                                      # noqa: PLC0415
    import urllib.request                                                    # noqa: PLC0415
    url = base_url.rstrip("/") + "/chat/completions"
    extras = dict(THINKING_OFF)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    last = None
    for attempt in range(max(1, retries)):
        body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": temperature,
                           **extras, **hint_request(prompt)}).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:   # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
            choice = (payload.get("choices") or [{}])[0]
            return {"text": ((choice.get("message") or {}).get("content") or ""),
                    "finish_reason": choice.get("finish_reason"), "usage": payload.get("usage") or {},
                    "thinking_switch": "sent" if extras else "rejected"}
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 400 and extras:
                extras = {}                     # the server does not know the switch: ask plainly, once
                continue
            if attempt + 1 < max(1, retries):
                time.sleep(5.0 * (attempt + 1))
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            last = exc
            if attempt + 1 < max(1, retries):
                time.sleep(5.0 * (attempt + 1))
    return {"text": "", "finish_reason": "error", "usage": {}, "error": "%s: %s" % (type(last).__name__, last),
            "thinking_switch": "sent" if extras else "rejected"}


def split_thinking(text: str) -> tuple:
    """(thinking, plan). A closed <think>...</think> block is the thinking and everything else is the plan;
    an unclosed <think> means the model ran out of tokens before it wrote a plan, so the plan is empty."""
    text = text or ""
    blocks = _THINK.findall(text)
    rest = _THINK.sub("", text)
    open_at = _THINK_OPEN.search(rest)
    if open_at:
        blocks.append(rest[open_at.start():])
        rest = ""                               # nothing around an unclosed <think> is a plan we can trust
    return "\n".join(blocks), rest


def tidy(text: str) -> str:
    """The plan as lines: any thinking block removed, blank lines and surrounding whitespace dropped,
    nothing else changed."""
    _, plan = split_thinking(text)
    lines = [line.strip() for line in plan.replace("\r", "\n").split("\n") if line.strip()]
    if 0 < len(lines) < HINT_MIN_LINES:
        lines = [s.strip() for line in lines for s in _SENTENCE_END.split(line) if s.strip()]
    return "\n".join(lines)


def was_reshaped(text: str) -> bool:
    """True when tidy() turned a one-paragraph reply into lines (counted in the generate manifest)."""
    _, plan = split_thinking(text)
    raw = [line for line in plan.replace("\r", "\n").split("\n") if line.strip()]
    return 0 < len(raw) < HINT_MIN_LINES and len(tidy(text).split("\n")) > len(raw)


def cmd_generate(args) -> int:
    rows = read_jsonl(args.rows)
    by_id = rows_by_id(rows)
    stuck = json.loads(Path(args.stuck).read_text(encoding="utf-8"))
    if stuck.get("stage") != "stuck":
        raise HintsError("%s is not a stuck.json this file wrote" % args.stuck)
    wanted = [i for i in stuck["stuck_ids"] if i in by_id]
    unknown = [i for i in stuck["stuck_ids"] if i not in by_id]
    if unknown:
        raise HintsError("%d stuck ids are not in %s (e.g. %s): the stuck set and the training file "
                         "must be the same rows" % (len(unknown), args.rows, ", ".join(unknown[:3])))
    if args.limit:
        wanted = wanted[: args.limit]
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    out = fresh_dir(args.out)
    written, usage, errors = [], {"prompt_tokens": 0, "completion_tokens": 0}, 0
    thought, unclosed, reshaped, switches = 0, 0, 0, {"sent": 0, "rejected": 0}
    started = time.monotonic()
    handle = (out / "hints.jsonl").open("w", encoding="utf-8")
    try:
        for number, index in enumerate(wanted, start=1):
            prompt = prompt_of(by_id[index])
            answer = post_chat(args.base_url, args.model, prompt, api_key=api_key,
                               max_tokens=args.max_tokens, temperature=args.temperature,
                               timeout=args.timeout, retries=args.retries)
            thinking, _ = split_thinking(answer["text"])
            hint = tidy(answer["text"])
            errors += int(bool(answer.get("error")))
            thought += int(bool(thinking))
            reshaped += int(was_reshaped(answer["text"]))
            unclosed += int(bool(thinking) and not hint and "</think>" not in answer["text"].lower())
            switches[answer.get("thinking_switch") or "sent"] += 1
            for key in usage:
                usage[key] += int((answer.get("usage") or {}).get(key) or 0)
            row = {"index": index, "hint": hint, "lines": len(hint.split("\n")) if hint else 0,
                   "finish_reason": answer.get("finish_reason"), "prompt_sha256": sha256_text(prompt),
                   "thinking_chars": len(thinking), "error": answer.get("error")}
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()                       # a crash at question 400 must not lose questions 1 to 399
            written.append(row)
            if number % 25 == 0 or number == len(wanted):
                print("%d of %d asked (%d errors)" % (number, len(wanted), errors), flush=True)
    finally:
        handle.close()
    manifest = {"schema": SCHEMA, "stage": "generate", "generated_at": now(),
                "data_source": stuck["data_source"], "hint_giver": args.model, "base_url": args.base_url,
                "stuck_file": str(Path(args.stuck).resolve()), "stuck_questions": stuck["stuck"],
                "asked": len(written), "with_text": sum(1 for row in written if row["hint"]),
                "errors": errors, "usage": usage, "seconds": round(time.monotonic() - started, 1),
                "request": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                            **THINKING_OFF},
                "thinking": {"replies_with_a_thinking_block": thought,
                             "thinking_unclosed_so_no_plan": unclosed, "switch": switches},
                "reshaped_from_one_paragraph": reshaped,
                "system_sha256": sha256_text(HINT_SYSTEM),
                "instruction_sha256": sha256_text(HINT_INSTRUCTION),
                "hints_sha256": sha256_file(out / "hints.jsonl")}
    write_json(out / "hints.manifest.json", manifest)
    print("asked %d questions, %d came back with text, %d errors" % (len(written), manifest["with_text"], errors))
    print("wrote", out / "hints.jsonl")
    return 0


# ============================================================================== 3. the leakage filter
def normalise_sql(text: str) -> str:
    """Whitespace-free, lower-case, single-quoted, no trailing semicolon: `a = 1` and `a=1` are the same query,
    so a hint quoting the gold query with different spacing is still caught."""
    return _WHITESPACE.sub("", (text or "").replace('"', "'").strip().rstrip(";")).lower()


def numbers_in(text: str) -> list:
    out = []
    for token in _NUMBER.findall(text or ""):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def leaks_number(hint: str, gold: str) -> bool:
    """Does the hint contain the gold number, or any number within 1 percent of it?

    The three scales are the bed's own: FinQA accepts the executed value, 100 times it and one
    hundredth of it (kit/beds/finqa.py), so a hint that names any of them has named the answer.
    """
    try:
        value = float(str(gold).strip())
    except (TypeError, ValueError):
        return False
    targets = [value, value * 100.0, value / 100.0]
    for found in numbers_in(hint):
        for target in targets:
            if abs(found - target) <= NUMBER_TOLERANCE * max(abs(target), 1e-9):
                return True
    return False


def gold_query_of(ground_truth: str):
    """Spider's ground truth is the JSON `{"db": ..., "gold_sql": ...}`; anything else has no query."""
    try:
        reference = json.loads(ground_truth)
    except (TypeError, ValueError):
        return None
    return reference.get("gold_sql") if isinstance(reference, dict) else None


def looks_like_sql_statement(text: str) -> bool:
    """True when the text holds a SELECT ... FROM <table> statement of its own (see _SELECT above)."""
    text = text or ""
    if _SELECT_UPPER.search(text) or _SELECT_CODE.search(text):
        return True
    for match in _SELECT.finditer(text):
        middle = {w.lower() for w in re.findall(r"[A-Za-z]+", match.group(1))}
        after = match.group(2).strip("`'\"").lower()
        if not (middle & _ENGLISH) and after not in _ENGLISH:
            return True
    return False


def drop_reason(hint: str, row: dict, *, allow_sql_statements: bool = False):
    """Why this hint may not be used, or None. The rules are the design note's, in its own order."""
    if not hint or not hint.strip():
        return "empty"
    lines = [line for line in hint.split("\n") if line.strip()]
    if len(lines) < HINT_MIN_LINES:
        return "too_few_lines"
    if len(lines) > HINT_MAX_LINES:
        return "too_many_lines"
    truth = ground_truth_of(row)
    gold_query = gold_query_of(truth)
    if gold_query:
        if normalise_sql(gold_query) in normalise_sql(hint):
            return "contains_gold_query"
        if not allow_sql_statements and looks_like_sql_statement(hint):
            # Stricter than the note's letter, in service of the note's intent ("never the answer"): a
            # hint carrying a complete SELECT hands over a query whether or not it is the gold one.
            # --allow-sql-statements reproduces the literal rule; the count is reported either way.
            return "contains_sql_statement"
    else:
        if str(truth).strip().lower() in ("yes", "no"):
            if _YES_NO.search(hint):
                return "states_yes_no"
        elif leaks_number(hint, truth):
            return "contains_gold_number"
    if _ANSWER_LINE.search(hint):
        return "writes_answer_line"
    return None


def filter_hints(hints: list, rows: list, *, allow_sql_statements: bool = False) -> tuple:
    """(kept, dropped, counts). One hint a question; a second hint for one question is a refusal."""
    by_id, seen = rows_by_id(rows), set()
    kept, dropped = [], []
    counts = {reason: 0 for reason in DROP_REASONS}
    for entry in hints:
        index = str(entry.get("index") or "")
        if index in seen:
            raise HintsError("two hints for question %s: one question is one hint" % index)
        seen.add(index)
        row = by_id.get(index)
        if row is None:
            counts["no_such_question"] += 1
            dropped.append({"index": index, "reason": "no_such_question"})
            continue
        hint = tidy(str(entry.get("hint") or ""))
        reason = drop_reason(hint, row, allow_sql_statements=allow_sql_statements)
        if reason is None:
            kept.append({"index": index, "hint": hint, "lines": len(hint.split("\n"))})
        else:
            counts[reason] += 1
            dropped.append({"index": index, "reason": reason, "hint": hint})
    return kept, dropped, counts


def stuck_count(args) -> int:
    """How many questions the coverage is a share OF: the stuck set, read from the stuck.json when
    one is given. Defaulting to the number of hints read would flatter the coverage of a `generate`
    that stopped early, so the campaign always passes --stuck."""
    if args.stuck:
        record = json.loads(Path(args.stuck).read_text(encoding="utf-8"))
        if record.get("stage") != "stuck":
            raise HintsError("%s is not a stuck.json this file wrote" % args.stuck)
        return int(record["stuck"])
    return int(args.stuck_questions or 0)


def cmd_filter(args) -> int:
    rows = read_jsonl(args.rows)
    hints = read_jsonl(args.hints)
    kept, dropped, counts = filter_hints(hints, rows, allow_sql_statements=args.allow_sql_statements)
    stuck_total = stuck_count(args) or len(hints)
    out = fresh_dir(args.out)
    kept_sha = write_new(out / "hints-filtered.jsonl", as_jsonl(sorted(kept, key=lambda row: row["index"])))
    write_new(out / "hints-dropped.jsonl", as_jsonl(sorted(dropped, key=lambda row: row["index"])))
    record = {"schema": SCHEMA, "stage": "filter", "generated_at": now(),
              "data_source": data_source_of(rows), "hints_file": str(Path(args.hints).resolve()),
              "hints_read": len(hints), "stuck_questions": stuck_total,
              "kept": len(kept), "dropped": len(dropped), "dropped_by_reason": counts,
              "coverage": round(len(kept) / stuck_total, 6) if stuck_total else 0.0,
              "allow_sql_statements": int(bool(args.allow_sql_statements)),
              "mean_lines": round(sum(row["lines"] for row in kept) / len(kept), 3) if kept else 0.0,
              "hints_filtered_sha256": kept_sha,
              "header_sha256": sha256_text(HINT_HEADER),
              "instruction_sha256": sha256_text(HINT_INSTRUCTION)}
    write_json(out / "filter.json", record)
    print("kept %d of %d hints (coverage %.4f of %d stuck questions); dropped: %s"
          % (len(kept), len(hints), record["coverage"], stuck_total,
             ", ".join("%s %d" % item for item in sorted(counts.items()) if item[1]) or "none"))
    print("wrote", out / "hints-filtered.jsonl")
    return 0


# ============================================================== 4. the hinted training file(s)
def hinted_prompt(prompt: str, hint: str) -> str:
    return prompt + HINT_HEADER + hint


def apply_hints(rows: list, hints: list) -> tuple:
    """(new rows, hinted ids). A row with no hint is returned UNCHANGED, object for object."""
    by_hint = {str(entry["index"]): str(entry["hint"]) for entry in hints}
    unknown = sorted(set(by_hint) - {question_id(row) for row in rows})
    if unknown:
        raise HintsError("%d hints name questions that are not in this training file (e.g. %s)"
                         % (len(unknown), ", ".join(unknown[:3])))
    out, hinted = [], []
    for row in rows:
        index = question_id(row)
        hint = by_hint.get(index)
        if not hint:
            out.append(row)
            continue
        new = json.loads(json.dumps(row))                    # a copy: the input row is never mutated
        new["prompt"] = [{**row["prompt"][0], "content": hinted_prompt(prompt_of(row), hint)}]
        out.append(new)
        hinted.append(index)
    return out, hinted


def token_counter(tokenizer_dir):
    """A function from prompt text to token count, or None when no tokenizer was asked for."""
    if not tokenizer_dir:
        return None
    from transformers import AutoTokenizer                                   # noqa: PLC0415
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


#: When no tokenizer is available, prompt length is estimated at this many characters a token. It is
#: deliberately LOW (a pessimistic estimate overstates the token count), and every manifest says which
#: of the two was used, because verl drops a prompt over `data.max_prompt_length` silently and a
#: shorter dose than the campaign asked for reaches no last step, no checkpoint and no merge.
CHARS_PER_TOKEN = 3.0


def prompt_lengths(rows: list, counter) -> list:
    if counter is None:
        return [int(len(prompt_of(row)) / CHARS_PER_TOKEN) + 1 for row in rows]
    return [counter(prompt_of(row)) for row in rows]


def split_index(lengths: list, keep: int, budget: int) -> int:
    """The number of rows that holds exactly `keep` rows the trainer will not drop as overlong."""
    surviving = 0
    for position, length in enumerate(lengths, start=1):
        if length <= budget:
            surviving += 1
        if surviving == keep:
            return position
    raise HintsError("only %d of %d rows are within %d tokens, so the first %d steps cannot be filled: "
                     "prepare a longer training file" % (surviving, len(lengths), budget, keep))


def write_rows(out: Path, name: str, rows: list) -> dict:
    """One training file as jsonl and (when pyarrow is installed) as the parquet the trainer reads."""
    text = as_jsonl(rows)
    record = {"rows": len(rows), "jsonl": "%s.jsonl" % name, "jsonl_sha256": sha256_text(text),
              "parquet": None}
    try:
        import pyarrow as pa                                                 # noqa: PLC0415
        import pyarrow.parquet as pq                                         # noqa: PLC0415
    except ImportError:
        pa = pq = None
    table = pa.Table.from_pylist(rows) if pa is not None else None            # built before anything is written
    write_new(out / ("%s.jsonl" % name), text)
    if table is not None:
        pq.write_table(table, out / ("%s.parquet" % name))
        record["parquet"] = "%s.parquet" % name
    return record


def cmd_apply(args) -> int:
    rows = read_jsonl(args.rows)
    hints = read_jsonl(args.hints)
    hinted_rows, hinted_ids = apply_hints(rows, hints)
    if not hinted_ids:
        raise HintsError("no hint matched any row: the hinted file would equal the control's and the "
                         "arm would measure nothing")
    counter = token_counter(args.tokenizer)
    budget = args.max_prompt_tokens
    plain_lengths = prompt_lengths(rows, counter)
    hinted_lengths = prompt_lengths(hinted_rows, counter)
    # A hint that pushes its prompt over verl's limit costs the trainer that ROW, which shortens the
    # dose and leaves no checkpoint. The default is to refuse (below, where the dose is checked) so the
    # decision is ours and not a file's; --drop-hints-over-budget is the declared fallback: that
    # question keeps its plain prompt, the hint is counted here, and the coverage it costs is visible.
    reverted = []
    if args.drop_hints_over_budget:
        marked_all = set(hinted_ids)
        kept_rows = []
        for original, hinted, length in zip(rows, hinted_rows, hinted_lengths):
            if question_id(original) in marked_all and length > budget:
                kept_rows.append(original)
                reverted.append(question_id(original))
            else:
                kept_rows.append(hinted)
        if reverted:
            hinted_rows = kept_rows
            hinted_ids = [index for index in hinted_ids if index not in set(reverted)]
            hinted_lengths = prompt_lengths(hinted_rows, counter)
            if not hinted_ids:
                raise HintsError("every hint would push its prompt over %d tokens, so nothing is left "
                                 "to measure" % budget)
    # The rows with no hint must come out unchanged: that is the one claim the `hint` arm rests on, so
    # it is PROVED, against a fresh parse of the source file rather than against the objects in memory.
    marked = set(hinted_ids)
    unchanged = as_jsonl([row for row in read_jsonl(args.rows) if question_id(row) not in marked])
    if sha256_text(as_jsonl([row for row in hinted_rows if question_id(row) not in marked])) != \
            sha256_text(unchanged):
        raise HintsError("a row with no hint changed: the hinted file may differ from the control's "
                         "training file only on the hinted questions")
    # Every refusal above and below happens BEFORE the output directory exists, so a corrected re-run
    # is not blocked by the never-overwrite rule.
    if args.steps:
        within = sum(1 for length in hinted_lengths if length <= budget)
        if within < args.steps * args.batch:
            raise HintsError("the hinted file has %d rows within %d tokens and %d steps x %d need %d: "
                             "the trainer would run out of data, reach no last step and leave no "
                             "checkpoint. Shorten the hints, raise the budget on purpose, or pass "
                             "--drop-hints-over-budget (declared, counted, and it costs coverage)."
                             % (within, budget, args.steps, args.batch, args.steps * args.batch))
    parts, fade = {}, {}
    if args.fade_at_step:
        keep = args.fade_at_step * args.batch
        position = split_index(hinted_lengths, keep, budget)
        parts = {"train-part1": hinted_rows[:position], "train-part2": rows[position:]}
        remaining = args.steps - args.fade_at_step if args.steps else args.fade_at_step
        surviving2 = sum(1 for length in plain_lengths[position:] if length <= budget)
        if surviving2 < remaining * args.batch:
            raise HintsError("after the fade at step %d only %d rows are left within %d tokens, and "
                             "%d steps x %d need %d: prepare a longer training file"
                             % (args.fade_at_step, surviving2, budget, remaining, args.batch,
                                remaining * args.batch))
        fade = {"fade_at_step": args.fade_at_step, "batch": args.batch, "steps": args.steps,
                "split_after_rows": position,
                "part1_rows_within_budget": sum(1 for length in hinted_lengths[:position] if length <= budget),
                "part2_rows_within_budget": surviving2,
                "part1_hinted_rows": sum(1 for row in parts["train-part1"] if question_id(row) in marked)}
    out = fresh_dir(args.out)
    files = {"train": write_rows(out, "train", hinted_rows)}
    for name in sorted(parts):
        files[name] = write_rows(out, name, parts[name])
    manifest = {"schema": SCHEMA, "stage": "apply", "generated_at": now(),
                "data_source": data_source_of(rows),
                "rows_file": str(Path(args.rows).resolve()), "rows_sha256": sha256_file(args.rows),
                "hints_file": str(Path(args.hints).resolve()), "hints_sha256": sha256_file(args.hints),
                "rows": len(rows), "hints_read": len(hints), "hinted_rows": len(hinted_ids),
                "unhinted_rows": len(rows) - len(hinted_ids),
                "hints_dropped_over_budget": len(reverted),
                "drop_hints_over_budget": int(bool(args.drop_hints_over_budget)),
                "unchanged_rows_sha256": sha256_text(unchanged),
                "header_sha256": sha256_text(HINT_HEADER),
                "max_prompt_tokens": budget,
                "prompt_tokens_from": "tokenizer" if counter is not None else "characters/%.1f" % CHARS_PER_TOKEN,
                "tokenizer": str(args.tokenizer) if args.tokenizer else None,
                "longest_prompt_before": max(plain_lengths), "longest_prompt_after": max(hinted_lengths),
                "rows_over_budget_before": sum(1 for length in plain_lengths if length > budget),
                "rows_over_budget_after": sum(1 for length in hinted_lengths if length > budget),
                "rows_within_budget_after": sum(1 for length in hinted_lengths if length <= budget),
                "files": files, **fade}
    manifest["rows_pushed_over_budget"] = (manifest["rows_over_budget_after"]
                                           - manifest["rows_over_budget_before"])
    write_json(out / "apply.manifest.json", manifest)
    print("hinted %d of %d rows; longest prompt %d -> %d tokens (%s), %d rows over %d%s"
          % (len(hinted_ids), len(rows), manifest["longest_prompt_before"],
             manifest["longest_prompt_after"], manifest["prompt_tokens_from"],
             manifest["rows_over_budget_after"], budget,
             "; split after row %d" % fade["split_after_rows"] if fade else ""))
    print("wrote", out / "train.jsonl")
    return 0


# ======================================================= 5. the reward function for `teacher-hint`
#: {question id: hint}, loaded once per process from HINTS_FILE. A reward function is called once per
#: rollout inside a ray worker, so the file is read on the first call and never again.
_HINTS: dict | None = None
_BEDS: dict | None = None
HINTS_FILE_ENV = "HINTS_FILE"


def load_hints(path=None) -> dict:
    """{question id: hint} from a filtered hints file. A missing file RAISES.

    A `teacher-hint` run whose hints did not load would be an ordinary SDPO run reporting itself as a
    hint arm -- the quietest way for this package to produce a wrong answer. So there is no default
    and no empty fallback.
    """
    global _HINTS                                                            # noqa: PLW0603
    if _HINTS is not None and path is None:
        return _HINTS
    where = path or os.environ.get(HINTS_FILE_ENV)
    if not where:
        raise HintsError("set %s to the hints-filtered.jsonl this arm serves to the teacher; a run "
                         "with no hints is not the arm it is named for" % HINTS_FILE_ENV)
    hints = {}
    for entry in read_jsonl(where):
        index, hint = str(entry.get("index") or ""), tidy(str(entry.get("hint") or ""))
        if not index or not hint:
            raise HintsError("%s has an entry with no index or no hint" % where)
        if index in hints:
            raise HintsError("%s has two hints for question %s" % (where, index))
        hints[index] = hint
    if not hints:
        raise HintsError("%s holds no hints" % where)
    if path is None:
        _HINTS = hints
    return hints


def beds() -> dict:
    """{data_source: bed module}, loaded the way kit/beds/rewards.py loads them: by path, once."""
    global _BEDS                                                             # noqa: PLW0603
    if _BEDS is None:
        loaded = {}
        for name in sorted(BED_FILES):
            module = load_bed(name)
            source = getattr(module, "DATA_SOURCE", None)
            if not isinstance(source, str) or not source:
                raise HintsError("the %s bed has no DATA_SOURCE constant" % name)
            if source in loaded:
                raise HintsError("two beds claim data_source %r" % source)
            loaded[source] = module
        _BEDS = loaded
    return _BEDS


def feedback(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None):
    """The `teacher-hint` arm's reward function: the bed's own score, with the hint as `feedback`.

    Dispatch is on `data_source`, from each bed's own DATA_SOURCE constant, exactly as
    kit/beds/rewards.py does; an unknown data source RAISES rather than scoring zero.

    The `feedback` string is THE HINT for a stuck question answered wrongly, and EMPTY otherwise --
    empty for a question with no hint, and empty for an attempt the bed calls right, which is the
    beds' own convention. In the pinned SDPO trainer an empty string is dropped
    (ray_trainer.py:632) and a non-empty one is placed in the teacher's prompt and nowhere else
    (ray_trainer.py:728-745,762). The bed's own feedback message is REPLACED, not appended to: this
    arm's one declared change is that the teacher sees the hint, and passing the bed's generic
    complaint as well would be a second change.
    """
    bed = beds().get(data_source) if isinstance(data_source, str) else None
    if bed is None:
        raise HintsError("no bed rewards data_source %r; this file dispatches %s"
                         % (data_source, ", ".join(sorted(beds()))))
    result = dict(bed.compute_score(data_source, solution_str, ground_truth, extra_info))
    index = str((extra_info or {}).get("index") or "")
    hint = load_hints().get(index, "")
    result["feedback"] = "" if float(result.get("acc", 0.0)) >= 1.0 else hint
    return result


#: verl takes `custom_reward_function.name`, and its default in every launcher of this kit is
#: `compute_score`. Both names are this one function, so the launcher needs no special case.
compute_score = feedback


def cmd_feedback(args) -> int:
    """The reward file's self-check: does it load its hints, and does it serve them to the right rows?"""
    rows = read_jsonl(args.rows)
    hints = load_hints(args.hints)
    by_id = rows_by_id(rows)
    unknown = sorted(set(hints) - set(by_id))
    if unknown:
        raise HintsError("%d hints name questions that are not in %s (e.g. %s)"
                         % (len(unknown), args.rows, ", ".join(unknown[:3])))
    source = data_source_of(rows)
    if source not in beds():
        raise HintsError("no bed rewards data_source %r" % source)
    os.environ[HINTS_FILE_ENV] = str(Path(args.hints).resolve())
    # Served, one stuck question at a time: the hint must come back for every question that has one.
    served = 0
    for index in sorted(hints):
        sample = by_id[index]
        result = feedback(source, "this answer is deliberately wrong", ground_truth_of(sample),
                          sample.get("extra_info"))
        if result["feedback"] != hints[index]:
            raise HintsError("the reward function did not serve the hint for %s" % index)
        served += 1
    keys = sorted(result)
    # And empty everywhere else: a reward function that fed the teacher a hint on every question
    # would be a different experiment, and one whose lists were misaligned would feed the WRONG hint.
    empty_elsewhere = 1
    for plain in [row for row in rows if question_id(row) not in hints][:50]:
        quiet = feedback(source, "this answer is deliberately wrong", ground_truth_of(plain),
                         plain.get("extra_info"))
        if quiet["feedback"] != "":
            empty_elsewhere = 0
            raise HintsError("the reward function returned feedback for question %s, which has no hint"
                             % question_id(plain))
    if args.out:
        out = fresh_dir(args.out)
        write_json(out / "feedback-check.json", {
            "schema": SCHEMA, "stage": "feedback-check", "generated_at": now(),
            "data_source": source, "hints_file": str(Path(args.hints).resolve()),
            "rows": len(rows), "hints": len(hints), "served": served,
            "empty_elsewhere": empty_elsewhere, "keys": keys,
            "hints_sha256": sha256_file(args.hints)})
    print("%s: %d hints load and %d of %d rows are served theirs; every other row gets an empty "
          "feedback string" % (source, len(hints), served, len(rows)))
    print("the keys the trainer dumps: %s" % ", ".join(keys))
    return 0


# ------------------------------------------------------------------------------------ the command line
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)

    one = sub.add_parser("stuck", help="which questions the model never solves (ONE GPU)")
    one.add_argument("--rows", required=True, help="a bed's train.jsonl or train.parquet")
    one.add_argument("--model", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--attempts", type=int, default=ATTEMPTS)
    one.add_argument("--temperature", type=float, default=TEMPERATURE)
    one.add_argument("--max-new", type=int, default=MAX_NEW)
    one.add_argument("--max-model-len", type=int, default=None)
    one.add_argument("--limit", type=int, help="the first N rows only; recorded in stuck.json")

    two = sub.add_parser("generate", help="ask an OpenAI-compatible endpoint for a plan per stuck question")
    two.add_argument("--stuck", required=True, help="the stuck.json written by `stuck`")
    two.add_argument("--rows", required=True)
    two.add_argument("--out", required=True)
    two.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    two.add_argument("--model", required=True, help="the served model's name, as the endpoint calls it")
    two.add_argument("--api-key-env", default="OPENAI_API_KEY",
                     help="the environment variable holding a bearer token; ignored when unset")
    two.add_argument("--max-tokens", type=int, default=GENERATE_DEFAULTS["max_tokens"])
    two.add_argument("--temperature", type=float, default=GENERATE_DEFAULTS["temperature"])
    two.add_argument("--timeout", type=float, default=GENERATE_DEFAULTS["timeout"])
    two.add_argument("--retries", type=int, default=GENERATE_DEFAULTS["retries"])
    two.add_argument("--limit", type=int)

    three = sub.add_parser("filter", help="drop every hint that could contain the answer, and count them")
    three.add_argument("--hints", required=True)
    three.add_argument("--rows", required=True)
    three.add_argument("--out", required=True)
    three.add_argument("--stuck", default=None,
                       help="the stuck.json written by `stuck`: `coverage` is a share of ITS stuck set")
    three.add_argument("--stuck-questions", type=int, default=None,
                       help="the size of the stuck set as a number, when there is no stuck.json to hand")
    three.add_argument("--allow-sql-statements", action="store_true",
                       help="keep a hint containing a SELECT that is not the gold query (the design "
                            "note's literal rule); the count is reported either way")

    four = sub.add_parser("apply", help="write the hinted training file(s)")
    four.add_argument("--rows", required=True)
    four.add_argument("--hints", required=True, help="the hints-filtered.jsonl written by `filter`")
    four.add_argument("--out", required=True)
    four.add_argument("--fade-at-step", type=int, default=None,
                      help="also write train-part1 (hinted) and train-part2 (plain) for `hint-faded`")
    four.add_argument("--steps", type=int, default=None, help="the arm's total steps (default: twice the fade)")
    four.add_argument("--batch", type=int, default=32, help="data.train_batch_size (default 32)")
    four.add_argument("--tokenizer", default=None,
                      help="a model directory: count prompt tokens exactly instead of estimating")
    four.add_argument("--max-prompt-tokens", type=int, default=2048,
                      help="verl's data.max_prompt_length; a longer prompt is dropped by the trainer")
    four.add_argument("--drop-hints-over-budget", action="store_true",
                      help="keep the plain prompt for a question whose hint would exceed the budget, "
                           "instead of refusing; the count is recorded and costs coverage")

    five = sub.add_parser("feedback", help="the reward file's self-check: it loads and serves its hints")
    five.add_argument("--hints", required=True)
    five.add_argument("--rows", required=True)
    five.add_argument("--out", default=None, help="a new directory for feedback-check.json, so a bar can read it")

    args = parser.parse_args(argv)
    actions = {"stuck": cmd_stuck, "generate": cmd_generate, "filter": cmd_filter,
               "apply": cmd_apply, "feedback": cmd_feedback}
    try:
        return actions[args.action](args)
    except HintsError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    sys.exit(main())
