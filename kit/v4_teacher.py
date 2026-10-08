#!/usr/bin/env python3
"""Bounded, restartable text demonstrations from Qwen3.6-27B (plan v4 §4).

Attempt numbers are ONE based. Fixed (seed, temperature) schedule:
1=(27101, 0.0), 2=(27102, 0.3), 3=(27103, 0.6), 4=(27104, 0.8).
All use top_p=1, top_k=-1, repetition_penalty=1, n=1; thinking is disabled
with the TEACHER'S own chat template. Explicit worked reasoning is requested.
The process owns one vLLM engine, TP=1, bf16, up to 32 concurrent sequences on one H100. Seeds
make restart scheduling stable, not a promise of cross-hardware bit parity.

POOL is training-only JSONL: id, task, question, optionally messages/prompt,
family, options (list or letter dictionary), gold/answer/reward_model. Missing
gold/options cannot be verified and is recorded as uncovered, never inferred.
Absent split means a caller-declared training pool; explicit non-train splits
are refused. Use Unit A's frozen pool/manifest in production.

Numeric tolerance: molar weight max(0.1 g/mol, 0.001*abs(option)); integer
counts 1e-6; other numeric families max(1e-6, 0.001*abs(option)). Nearest
option must be in tolerance and uniquely nearest (tie epsilon 1e-9); multiple
options within their respective tolerances are also refused as ambiguous.
Completeness is an explicit structural screen, not a proof of reasoning:
closed reasoning/answer tags, >=8 reasoning words, no dangling think tag,
and an EOS/stop completion. FinQA requires >=8 words before its last Answer:
line. The response + EOS must fit 2,048 tokens. Unit A question eligibility is 2,048
chat tokens; the SFT allocation remains 4,096 + 2,048 without truncation. Rewrite
and D demonstration contexts both admit 6,144 input tokens. Any think tag in an
output is rejected; rendered generation-prompt tails are retained in raw records.

Each returned raw attempt is atomically saved BEFORE verification. Derived
JSONL files are rebuilt from that journal. Interrupted inference with no
returned output retries the same numbered seed; durable attempts never run
again. A lock excludes concurrent writers. Configuration/prompt changes
refuse resumption. No extra attempts, gold-answer prompts or performance retries.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
from collections import Counter, defaultdict
import fcntl
import hashlib
from importlib import metadata
import json
import math
import time
import os
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.beds import finqa, chemistry
from kit.v4_contract import INITIAL_8B_REVISION, DEMONSTRATION_CONTEXT_CAP, check_schedule, input_path
from kit.v4_families import family as classify_family, normalize
from kit.v4_resources import resource

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_8B = Path(os.environ['QWEN3_8B_TOKENIZER']) if os.environ.get('QWEN3_8B_TOKENIZER') else None
RESPONSE_CAP = 2048
INPUT_CAP = 4096
SFT_CONTEXT_CAP = INPUT_CAP + RESPONSE_CAP
REWRITE_INPUT_CAP = DEMONSTRATION_CONTEXT_CAP
# Qualification changes these checked-in constants, never a CLI flag or environment.
REGISTERED_TEACHER_REVISION = None
REGISTERED_TEACHER_WEIGHT_HASHES = None
# No weight receipt is present locally. An unset registry fails scientific admission.
INITIAL_8B_WEIGHT_HASHES = None
TEACHER_NEW_TOKENS = 2560
H100_MAX_NUM_SEQS = 32
TEACHER_TOKEN_LIMIT_RECEIPT = {
    "status": "27b_tokenizer_unavailable_locally", "max_new_tokens": 2560,
    "response_cap_8b_including_eos": RESPONSE_CAP,
    "ratio_bound": None,
    "reason": "No pinned Qwen3.6-27B tokenizer is available locally. Use the requested "
              "2,560 fallback (25 percent headroom); this is not a proven response-token bound. "
              "Measure and register the two-tokenizer bound at real-model qualification.",
}
REWRITE_NEW_TOKENS = 2048
REWRITE_INSTRUCTION = ("Write a self-contained worked answer to the question in your own words. "
                       "Preserve the valid reasoning and calculations, explain the necessary steps, "
                       "and give the final answer in the required format. Do not refer to the reference answer.")
REFERENCES = Path(os.environ['V4_REFERENCE_ROOT']) if os.environ.get('V4_REFERENCE_ROOT') else None
SCHEDULE = [(27101, 0.0), (27102, 0.3), (27103, 0.6), (27104, 0.8)]
NUMERIC_FAMILIES = {"molar weight from SMILES", "molar weight from IUPAC name",
                    "hydrogen-bond donors or acceptors", "rotatable bonds",
                    "distribution coefficient logD", "aqueous solubility"}
EXCLUDED = {"distribution coefficient logD", "aqueous solubility"}
OPTION = re.compile(r"(?:^|\n)[ \t]*([A-D])[ \t]*[:.)][ \t]*(.*?)(?=\n[ \t]*[A-D][ \t]*[:.)]|\nPlease reason|\Z)", re.S)
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NUMERIC_VALUE = re.compile(rf"^\s*({NUM})\s*(?:g/mol|g mol-1|%|bonds?|donors?|acceptors?)?\s*$", re.I)
CHEM_SYSTEM = "Work through the chemistry explicitly. Give a complete worked response in <reasoning>...</reasoning>, then <answer>...</answer>. "


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def atomic_json(path, value):
    """Durable atomic replace; a killed write cannot become a completed attempt."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def load_pool(path, task=None):
    return validate_pool(read_jsonl(path), task)


def validate_pool(items, task=None):
    """Apply training-only, task, exclusion and duplicate checks to Python inputs too."""
    seen = set()
    for item in items:
        if not {"id", "task", "question"} <= item.keys() or not isinstance(item["question"], str):
            raise ValueError("pool rows need id, task and a string question")
        if str(item["id"]) in seen:
            raise ValueError("duplicate pool id: " + str(item["id"]))
        seen.add(str(item["id"]))
        split = item.get("split", item.get("extra_info", {}).get("split", "train"))
        if split != "train":
            raise ValueError("teacher pools/probes must be training-only")
        if item["task"] not in ("chemistry", "finqa") or (task and item["task"] != task):
            raise ValueError("pool task does not match command")
        if item["task"] == "chemistry" and family_of(item) in EXCLUDED:
            raise ValueError("plan v4 excludes logD and aqueous solubility from training")
    if not items:
        raise ValueError("empty pool")
    return items


def family_of(item):
    return item.get("family") or (classify_family(normalize(item.get("description", item["question"]))) if item["task"] == "chemistry" else "finqa")


def student_messages(item):
    """Unaided scoring/training prompt, with no demonstration or gold attached."""
    messages = item.get("messages", item.get("prompt"))
    if isinstance(messages, list):
        if any(m["role"] not in ("system", "user") for m in messages):
            raise ValueError("pool prompt must contain only system/user messages")
        return [{"role": m["role"], "content": m["content"]} for m in messages]
    content = messages if isinstance(messages, str) else item["question"]
    if item["task"] == "finqa" and "qa" in item:
        content = finqa.render_prompt(item)
    if item["task"] == "chemistry":
        opts = options_of(item, content)
        if opts and not OPTION.search(content):
            content += "\n\n" + "\n".join(f"{k}: {v}" for k, v in opts.items())
        return [{"role": "system", "content": item.get("system", CHEM_SYSTEM + "The final answer must be the option letter alone.")}, {"role": "user", "content": content}]
    return [{"role": "user", "content": content}]


def options_of(item, content=None):
    opts = item.get("options")
    if isinstance(opts, list):
        return {chr(65 + i): str(v) for i, v in enumerate(opts)}
    if isinstance(opts, dict):
        return {str(k): str(v) for k, v in opts.items()}
    if content is None:
        content = "\n".join(m["content"] for m in item.get("messages", []) if m["role"] == "user") or item.get("prompt", item["question"])
    if not isinstance(content, str):
        content = "\n".join(m["content"] for m in content if m["role"] == "user")
    return {m.group(1): m.group(2).strip() for m in OPTION.finditer(content)}


def numeric_value(text):
    match = NUMERIC_VALUE.fullmatch(str(text))
    return float(match.group(1)) if match and math.isfinite(float(match.group(1))) else None


def is_numeric(item):
    opts = options_of(item)
    return family_of(item) in NUMERIC_FAMILIES or bool(opts) and all(numeric_value(v) is not None for v in opts.values())


def teacher_messages(item, *, visible_options=False, permutation=None):
    """Replace letter instructions AND remove options for numeric generation."""
    if item["task"] == "finqa":
        messages = student_messages(item)
        return [{"role": "system", "content": "Give a complete worked calculation. Finish with Answer: <number> (or yes/no)."}, *messages]
    messages = student_messages(item)
    content = "\n".join(m["content"] for m in messages if m["role"] == "user")
    opts = options_of(item, content)
    blind = is_numeric(item) and not visible_options
    if blind:
        content = OPTION.sub("", content).strip()
    elif permutation is not None:
        if sorted(permutation) != list(range(len(opts))):
            raise ValueError("invalid option permutation")
        values = list(opts.values())
        replacements = {chr(65+i): values[j] for i, j in enumerate(permutation)}
        # Replace values IN PLACE: moving an instruction around would confound
        # the option-permutation consistency probe.
        for match in reversed(list(OPTION.finditer(content))):
            start, end = match.span(2)
            content = content[:start] + replacements[match.group(1)] + content[end:]
    system = CHEM_SYSTEM + ("The final answer must be the numeric value alone, with no option letter." if blind else "The final answer must be the option letter alone.")
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def gold_of(item):
    return item.get("gold", item.get("answer", item.get("reward_model", {}).get("ground_truth", item.get("qa", {}).get("exe_ans"))))


def map_numeric(value, options, family):
    """Return uniquely nearest letter only when within the frozen tolerance."""
    detail = {"prediction": value, "family": family, "rule": "nearest; ties within 1e-9 or multiple in-tolerance options refused"}
    parsed = {k: numeric_value(v) for k, v in options.items()}
    if value is None or not parsed or any(v is None for v in parsed.values()):
        return None, {**detail, "reason": "missing_numeric_prediction_or_options"}
    distances = sorted((abs(value - v), k, v) for k, v in parsed.items())
    delta, letter, target = distances[0]
    def tolerance_for(target):
        return (max(0.1, .001 * abs(target)) if family.startswith("molar weight") else
                1e-6 if family in {"hydrogen-bond donors or acceptors", "rotatable bonds"} else max(1e-6, .001 * abs(target)))
    tolerance = tolerance_for(target)
    detail.update(distance=delta, tolerance=tolerance, nearest_option=letter, option_value=target)
    if len(distances) > 1 and abs(distances[1][0] - delta) <= 1e-9:
        return None, {**detail, "reason": "ambiguous_nearest_options"}
    candidates = [k for distance, k, v in distances if distance <= tolerance_for(v) + 1e-12]
    if len(candidates) > 1:
        return None, {**detail, "reason": "ambiguous_options_within_tolerance", "candidates": candidates}
    if delta > tolerance + 1e-12:
        return None, {**detail, "reason": "outside_tolerance"}
    return letter, detail


def chemistry_choice(item, text, *, blind=True, options=None):
    matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, re.S))
    answer = matches[-1].group(1).strip() if matches else None
    if blind and is_numeric(item):
        choice, detail = map_numeric(numeric_value(answer), options or options_of(item), family_of(item))
    else:
        choice = answer if answer in (options or options_of(item)) else None
        detail = {"prediction": answer, "reason": None if choice else "missing_or_invalid_option_letter"}
    return choice, detail


def tokenize_student(tokenizer, messages, text):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(text + tokenizer.eos_token, add_special_tokens=False)
    return prompt_ids, response_ids


def verify(item, raw, tokenizer, *, mapped_target=False):
    """Return all rejection reasons, a mapped target, and real student lengths."""
    text = raw["text"]
    reasons = []
    if raw.get("finish_reason") not in ("stop", "eos"):
        reasons.append("unfinished_generation")
    if re.search(r"</?think\b", text, re.I):
        reasons.append("thinking_content_present")
    if text.count("<think>") != text.count("</think>"):
        reasons.append("unclosed_thinking")
    if item["task"] == "chemistry":
        choice, detail = chemistry_choice(item, text, blind=not mapped_target)
        correct = choice is not None and choice == gold_of(item)
        match = re.fullmatch(r"\s*(?:<think>.*?</think>\s*)?<reasoning>(.*?)</reasoning>\s*<answer>.*?</answer>\s*", text, re.S)
        worked = bool(match) and len(re.findall(r"\b\w+\b", match.group(1))) >= 8 and "..." not in match.group(1)
        target = text
        if choice and is_numeric(item) and not mapped_target:
            final = list(re.finditer(r"<answer>.*?</answer>", text, re.S))[-1]
            target = text[:final.start()] + f"<answer>{choice}</answer>" + text[final.end():]
    else:
        prediction = finqa.extract_answer(text)
        correct = finqa.is_correct(prediction, str(gold_of(item)))
        detail = {"prediction": prediction, "scorer": "kit.beds.finqa.is_correct", "tolerance": finqa.TOLERANCE}
        lines = list(finqa.ANSWER_LINE.finditer(text))
        worked = bool(lines) and len(text[:lines[-1].start()].split()) >= 8 and not text[lines[-1].end():].strip()
        target = text
    if item["task"] == "chemistry" and choice:
        # Score the mapped target through the authors' actual scorer as well.
        input_path("SDPO_DIR")
        authors_score = chemistry.compute_score("sciknoweval", target, str(gold_of(item)))["score"]
        detail["authors_score"] = authors_score
        correct = correct and authors_score == 1
    if gold_of(item) is None:
        reasons.append("missing_gold")
    if not correct:
        reasons.append("incorrect_final_answer")
        if detail.get("reason"):
            reasons.append(detail["reason"])
    if not worked:
        reasons.append("incomplete_worked_response")
    prompt_ids, response_ids = tokenize_student(tokenizer, student_messages(item), target)
    if len(response_ids) > RESPONSE_CAP:
        reasons.append("student_response_over_2048")
    if len(prompt_ids) > INPUT_CAP:
        reasons.append("student_input_over_4096")
    if raw.get("input_over_cap"):
        reasons.insert(0, "generation_input_over_cap")
    detail.update(correct=correct, worked_response=worked)
    return {"id": item["id"], "task": item["task"], "family": family_of(item), "attempt": raw["attempt"],
            "text": target, "verified": not reasons, "verifier_detail": detail, "tokens_8b": len(response_ids),
            "context_tokens_8b": len(prompt_ids) + len(response_ids), "primary_reason": reasons[0] if reasons else None, "reasons": reasons}


def verified_lookup(items, records, tokenizer=None):
    """Reject foreign/duplicate targets and optionally recheck mapped final answers."""
    pool = {str(i["id"]): i for i in items}
    result = {}
    for record in records:
        key = str(record["id"])
        if key in result or key not in pool or record.get("verified") is not True:
            raise ValueError("duplicate, foreign or unverified target")
        if record.get("task", pool[key]["task"]) != pool[key]["task"]:
            raise ValueError("target task mismatch")
        if tokenizer is not None:
            checked = verify(pool[key], {"text": record["text"], "attempt": record.get("attempt", 1),
                                        "finish_reason": "stop"}, tokenizer, mapped_target=True)
            if not checked["verified"]:
                raise ValueError("target fails verification: " + ", ".join(checked["reasons"]))
        result[key] = record
    return result


def rewrite_messages(item, demonstration):
    """Append the plan's literal rewrite instruction to the original question prompt."""
    messages = student_messages(item)
    messages[-1]["content"] += "\n\nReference worked answer:\n" + demonstration["text"] + "\n\n" + REWRITE_INSTRUCTION
    return messages


def rewrite_seed(ident, attempt):
    """Stable question/attempt seed; corpus reused across campaign seeds and orders."""
    return int(sha(f"v4-rewrite|{ident}|{attempt}")[:8], 16) % (2**31)


def coverage(items, demonstrations, exhausted=(), *, rewrites=None, rewrite_exhausted=(), stand_in_teacher=False, model_identity_qualified=False):
    """Gate the intersection against the unfiltered original pool, never the teacher subset."""
    from kit.v4_contract import technical_synthetic
    synthetic = technical_synthetic(demonstrations) or technical_synthetic(rewrites)
    teacher = set(verified_lookup(items, demonstrations))
    rewritten = set(verified_lookup(items, rewrites)) if rewrites is not None else set()
    intersection = teacher & rewritten if rewrites is not None else teacher
    groups = defaultdict(list)
    for item in items:
        for key in ((item["task"], None), (item["task"], family_of(item))):
            groups[key].append(str(item["id"]))
    tasks, families = {}, {}
    for (task, family), ids in sorted(groups.items(), key=lambda kv: str(kv[0])):
        total, n = len(ids), len(set(ids) & intersection)
        threshold = 80 if family is None else 70 if family.startswith("molar weight") else None
        required = (threshold * total + 99) // 100 if threshold else None
        nt = len(set(ids) & teacher)
        entry = {"covered": n, "total": total, "original_pool": total, "fraction": n / total,
                 "teacher_covered": nt, "teacher_uncovered": total - nt,
                 "rewrite_covered": len(set(ids) & rewritten),
                 "rewrite_attrition": nt - n if rewrites is not None else None,
                 "intersection": n if rewrites is not None else None,
                 "threshold_percent": threshold, "required": required,
                 "pass": n >= required if required is not None else None}
        if family is None:
            tasks[task] = entry
        else:
            families.setdefault(task, {})[family] = entry
    checks = [e["pass"] for e in tasks.values()] + [e["pass"] for fs in families.values() for e in fs.values() if e["pass"] is not None]
    complete = (teacher <= rewritten | set(rewrite_exhausted)) if rewrites is not None else len(teacher | set(exhausted)) == len(items)
    scientific = not synthetic and rewrites is not None and not stand_in_teacher and all(e["total"] == 1441 for e in tasks.values())
    return {"technical_synthetic": synthetic, "tasks": tasks, "families": families, "gate_basis": "intersection" if rewrites is not None else "teacher_only_provisional",
            "gate_pass": all(checks), "complete": complete, "qualified": all(checks) and complete,
            "stand_in_teacher": stand_in_teacher, "scope": "original_1441_pool" if scientific else "engineering_or_smoke_pool",
            "coverage_phase_allowed": scientific and all(checks) and complete,
            "scientific_phase_allowed": scientific and all(checks) and complete and model_identity_qualified,
            "teacher_uncovered_ids": [i["id"] for i in items if str(i["id"]) not in teacher],
            "rewrite_uncovered_ids": [i["id"] for i in items if str(i["id"]) in teacher & set(rewrite_exhausted)],
            "rewrite_pending_ids": [i["id"] for i in items if rewrites is not None and str(i["id"]) in teacher - rewritten - set(rewrite_exhausted)],
            "uncovered_ids": [i["id"] for i in items if str(i["id"]) not in intersection]}


def shard_directory(index, count):
    """Stable, distinct directory name for each zero-based shard."""
    return f"shard-{index:05d}-of-{count:05d}"


def select_shard(items, index, count):
    """Select by ID hash while retaining original pool order, independent of workers."""
    if not isinstance(count, int) or not isinstance(index, int) or count < 1 or not 0 <= index < count:
        raise ValueError("shard index must be in 0..count-1")
    return [i for i in items if int(sha(str(i["id"])), 16) % count == index]


def batched_worst_case(pool_size, *, attempts=4, shard_count=1, max_num_seqs=H100_MAX_NUM_SEQS,
                       max_new_tokens=TEACHER_NEW_TOKENS, input_cap=INPUT_CAP,
                       seconds_per_wave=None, reload_seconds=0, verify_seconds_per_attempt=0, shard_sizes=None):
    """Budget geometry; optional measured full-wave latency yields a wall-time bound.

    A wave must be measured with input_cap and max_new_tokens, including prefill.
    Shards run concurrently, one H100 each. Hash shards are not assumed balanced:
    without actual shard sizes all questions may land on a single shard. Pass counts
    from select_shard (use the full pool for conservative rewrite sizing). Allocation
    GPU-hours include idle shards. This deliberately supplies no invented throughput.
    """
    if (not isinstance(pool_size, int) or pool_size < 0 or not 1 <= attempts <= 4
            or shard_count < 1 or max_num_seqs < 1 or max_new_tokens < 1 or input_cap < 1
            or reload_seconds < 0 or verify_seconds_per_attempt < 0
            or seconds_per_wave is not None and seconds_per_wave < 0):
        raise ValueError("invalid batched worst-case geometry")
    if shard_sizes is not None and (len(shard_sizes) != shard_count or sum(shard_sizes) != pool_size
            or any(not isinstance(n, int) or n < 0 for n in shard_sizes)):
        raise ValueError("invalid shard sizes")
    largest = max(shard_sizes, default=0) if shard_sizes is not None else pool_size
    waves = attempts * math.ceil(largest / max_num_seqs)
    wall = None if seconds_per_wave is None else (reload_seconds + waves * seconds_per_wave
            + largest * attempts * verify_seconds_per_attempt)
    return {"pool_size": pool_size, "attempts": attempts, "shard_count": shard_count,
            "max_num_seqs": max_num_seqs, "max_new_tokens": max_new_tokens, "input_cap": input_cap, "max_shard_size": largest,
            "max_requests": pool_size * attempts, "max_generate_calls_per_shard": attempts if pool_size else 0,
            "max_waves_per_shard": waves, "max_output_tokens": pool_size * attempts * max_new_tokens,
            "max_input_tokens": pool_size * attempts * input_cap, "wall_seconds": wall,
            "allocation_gpu_hours": None if wall is None else wall * shard_count / 3600}


def measure_pool_token_ratio(items, teacher_tokenizer, student_tokenizer):
    """Record an empirical teacher/student ratio on real prompt text, not a universal output bound."""
    counts = []
    for item in items:
        text = "\n".join(m["content"] for m in student_messages(item))
        teacher = len(teacher_tokenizer.encode(text, add_special_tokens=False))
        student = len(student_tokenizer.encode(text, add_special_tokens=False))
        if student == 0:
            raise ValueError("empty pool text cannot establish a token ratio")
        counts.append({"id": item["id"], "teacher_tokens": teacher, "student_tokens": student})
    if not counts:
        raise ValueError("empty pool cannot establish a token ratio")
    ratio = max(r["teacher_tokens"] / r["student_tokens"] for r in counts)
    return {"status": "empirical_pool_measurement", "pool_hash": sha(json.dumps(items, sort_keys=True)),
            "ratio_bound_on_pool_text": ratio, "counts": counts,
            "smallest_cap_under_observed_ratio": math.ceil((RESPONSE_CAP - 1) * ratio) + 1,
            "universal_response_bound_proven": False}


def merge_shards(directories, out, tokenizer):
    """Validate a complete homogeneous shard set and replay its durable journals without inference."""
    from contextlib import ExitStack
    directories = [Path(d) for d in directories]
    if not directories:
        raise ValueError("missing shards")
    items = load_pool(directories[0] / "pool.jsonl")
    check_pool_provenance(items)
    with ExitStack() as locks:
        documents = []
        for directory in directories:
            lock = locks.enter_context((directory / ".lock").open("r"))
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            documents.append(json.loads((directory / "manifest.json").read_text()))
        if any("shard" not in d for d in documents):
            raise ValueError("foreign shard: no shard identity")
        count = documents[0]["shard"]["count"]
        indices = [d["shard"]["index"] for d in documents]
        if len(set(indices)) != len(indices):
            raise ValueError("duplicated shard")
        if set(indices) != set(range(count)):
            raise ValueError("missing or foreign shard")
        base = {k: v for k, v in documents[0].items() if k != "shard"}
        rewriting = base["action"] == "rewrite"
        demonstrations = read_jsonl(directories[0] / "demonstrations_source.jsonl") if rewriting else None
        teachers = verified_lookup(items, demonstrations, tokenizer) if rewriting else {}
        active_pool = [i for i in items if str(i["id"]) in teachers] if rewriting else items
        expected_prompts = {str(i["id"]): sha(json.dumps(rewrite_messages(i, teachers[str(i["id"])])
                            if rewriting else teacher_messages(i), sort_keys=True)) for i in active_pool}
        if (base["prompt_hashes"] != expected_prompts or base["demonstrations_hash"] !=
                (sha(json.dumps(demonstrations, sort_keys=True)) if rewriting else None)):
            raise ValueError("foreign shard prompt or demonstration source")
        journals, timings = {}, {}
        for directory, doc in zip(directories, documents):
            if (doc["shard"] != {"index": doc["shard"]["index"], "count": count, "rule": "sha256(id) modulo count"}
                    or {k: v for k, v in doc.items() if k != "shard"} != base
                    or load_pool(directory / "pool.jsonl") != items
                    or doc["pool_hash"] != sha(json.dumps(items, sort_keys=True))):
                raise ValueError("foreign shard configuration or pool")
            if rewriting and read_jsonl(directory / "demonstrations_source.jsonl") != demonstrations:
                raise ValueError("foreign shard demonstrations")
            subset = select_shard(items, doc["shard"]["index"], count)
            active = [i for i in subset if str(i["id"]) in teachers] if rewriting else subset
            expected = set()
            for item in active:
                for attempt in range(1, base["attempts"] + 1):
                    name = f"{sha(str(item['id']))}-{attempt}.json"
                    path = directory / "raw" / name
                    if not path.is_file():
                        raise ValueError("missing shard attempt: " + name)
                    raw = json.loads(path.read_text())
                    seed, temperature = (rewrite_seed(item["id"], attempt), .7) if rewriting else SCHEDULE[attempt-1]
                    if any(raw.get(k) != v for k, v in {"id": item["id"], "attempt": attempt,
                            "seed": seed, "temperature": temperature, "top_p": .95 if rewriting else 1,
                            "stand_in_teacher": base["stand_in_teacher"], "teacher_model": base["teacher_model"]}.items()):
                        raise ValueError("foreign shard attempt")
                    expected.add(name)
                    journals[name] = raw
                    timing = directory / "timings" / name
                    if timing.is_file():
                        timings[name] = json.loads(timing.read_text())
                    checked = verify(item, raw, tokenizer, mapped_target=True) if rewriting else verify(item, raw, tokenizer)
                    if checked["verified"]:
                        break
            if {p.name for p in (directory / "raw").glob("*.json")} != expected:
                raise ValueError("foreign or duplicated shard attempt")
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        (out / "raw").mkdir()
        (out / "timings").mkdir()
        for name, raw in journals.items():
            atomic_json(out / "raw" / name, raw)
        for name, timing in timings.items():
            atomic_json(out / "timings" / name, timing)
        def forbidden(*args, **kwargs):
            raise ValueError("missing shard attempt during replay")
        return _run_pool(items, out, forbidden, tokenizer, base, base["attempts"], demonstrations)


def generate_pool(items, out, generator, tokenizer, manifest, attempts=4, *, sdpo=None, finqa_root=None, shard_index=None, shard_count=1):
    """Generate bounded teacher attempts using the common durable journal."""
    check_pool_provenance(items, sdpo, finqa_root)
    return _run_pool(items, out, generator, tokenizer, manifest, attempts, shard_index=shard_index, shard_count=shard_count)


def rewrite_pool(items, demonstrations, out, generator, tokenizer, manifest, *, sdpo=None, finqa_root=None, shard_index=None, shard_count=1):
    """Frozen initial-student rewrites: four attempts, first acceptable, no fallback."""
    check_pool_provenance(items, sdpo, finqa_root)
    verified_lookup(items, demonstrations, tokenizer)
    return _run_pool(items, out, generator, tokenizer, manifest, 4, demonstrations=demonstrations, shard_index=shard_index, shard_count=shard_count)


def _run_pool(items, out, generator, tokenizer, manifest, attempts, demonstrations=None, *, shard_index=None, shard_count=1):
    if not 1 <= attempts <= 4:
        raise ValueError("attempts must be 1..4")
    from kit.v4_contract import technical_synthetic
    synthetic = technical_synthetic(manifest) or technical_synthetic(demonstrations) or technical_synthetic(items)
    if synthetic and manifest.get("stand_in_teacher") is not True:
        raise ValueError("technical_synthetic inference inputs are smoke only")
    if synthetic:
        manifest = {**manifest, "technical_synthetic": True, "scientific_phase_allowed": False}
    rewriting = demonstrations is not None
    teachers = verified_lookup(items, demonstrations) if rewriting else {}
    active = [i for i in items if str(i["id"]) in teachers] if rewriting else items
    prompts = {str(i["id"]): rewrite_messages(i, teachers[str(i["id"])]) if rewriting else teacher_messages(i) for i in active}
    provenance = {"stand_in_teacher": manifest.get("stand_in_teacher", False),
                  "teacher_model": manifest.get("teacher_model", manifest.get("model", "unspecified"))}
    manifest = json.loads(json.dumps({**manifest, **provenance, "action": "rewrite" if rewriting else "generate",
                "attempts": attempts, "pool_hash": sha(json.dumps(items, sort_keys=True)),
                "demonstrations_hash": sha(json.dumps(demonstrations, sort_keys=True)) if rewriting else None,
                "prompt_hashes": {k: sha(json.dumps(v, sort_keys=True)) for k, v in prompts.items()}}))
    shard_items = select_shard(items, shard_index, shard_count) if shard_index is not None else items
    if shard_index is None and shard_count != 1:
        raise ValueError("shard_count requires shard_index")
    shard_ids = {str(i["id"]) for i in shard_items}
    active = [i for i in active if str(i["id"]) in shard_ids]
    out = Path(out)
    if shard_index is not None:
        manifest["shard"] = {"index": shard_index, "count": shard_count, "rule": "sha256(id) modulo count"}
        out = out / shard_directory(shard_index, shard_count)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mpath = out / "manifest.json"
        if mpath.exists() and json.loads(mpath.read_text()) != manifest:
            raise ValueError("resume manifest changed; use a fresh output directory")
        atomic_json(mpath, manifest)
        write_jsonl(out / "pool.jsonl", items)
        if rewriting:
            write_jsonl(out / "demonstrations_source.jsonl", demonstrations)
        journal = out / "raw"
        journal.mkdir(exist_ok=True)
        timing_journal=out/'timings';timing_journal.mkdir(exist_ok=True)
        accepted, rejects, exhausted, raw_rows = [], [], set(), []
        checked = {}
        raw_by_key = {}
        try:
            remaining = list(active)
            for attempt in range(1, attempts + 1):
                pending = []
                for item in remaining:
                    key = (str(item["id"]), attempt)
                    path = journal / f"{sha(key[0])}-{attempt}.json"
                    if path.exists():
                        raw_by_key[key] = json.loads(path.read_text())
                    else:
                        seed, temperature = (rewrite_seed(item["id"], attempt), .7) if rewriting else SCHEDULE[attempt-1]
                        pending.append({"id": item["id"], "messages": prompts[key[0]], "attempt": attempt,
                                        "seed": seed, "temperature": temperature, "top_p": .95 if rewriting else 1})
                if pending:
                    from kit.v4_timing import utc_now, receipt
                    started, clock = utc_now(), time.monotonic()
                    batch = getattr(generator, "generate_many", None)
                    outputs = batch(pending) if batch else (
                        generator(r["messages"], attempt, seed=r["seed"], temperature=r["temperature"], top_p=r["top_p"])
                        if rewriting else generator(r["messages"], attempt) for r in pending)
                    if batch and len(outputs) != len(pending):
                        raise ValueError("incomplete generation batch")
                    # Persist the WHOLE returned round before the first verifier is called.
                    for request, output in zip(pending, outputs):
                        key = (str(request["id"]), attempt)
                        raw = {**output, **provenance, **{k: request[k] for k in ("id", "attempt", "seed", "temperature", "top_p")}}
                        if rewriting and teachers[key[0]].get("technical_synthetic"):
                            raw.update(technical_synthetic=True, scientific_phase_allowed=False)
                        atomic_json(journal / f"{sha(key[0])}-{attempt}.json", raw)
                        raw_by_key[key] = raw
                        measured = output.get("generate_timing") or receipt(started, (time.monotonic()-clock)/len(pending), batch_requests=len(pending))
                        atomic_json(timing_journal / f"{sha(key[0])}-{attempt}.json", {
                            "id": request["id"], "attempt": attempt, "raw_sha256": sha(json.dumps(raw, sort_keys=True)),
                            "generate": measured, "verify": None})
                rejected = []
                for item in remaining:
                    key = (str(item["id"]), attempt)
                    raw = raw_by_key[key]
                    from kit.v4_timing import utc_now, receipt
                    started, clock = utc_now(), time.monotonic()
                    result = verify(item, raw, tokenizer, mapped_target=True) if rewriting else verify(item, raw, tokenizer)
                    timing_path = timing_journal / f"{sha(key[0])}-{attempt}.json"
                    timing_doc = json.loads(timing_path.read_text()) if timing_path.exists() else None
                    if timing_doc is None and raw.get("generate_timing"):
                        timing_doc = {"id": item["id"], "attempt": attempt,
                                      "raw_sha256": sha(json.dumps(raw, sort_keys=True)),
                                      "generate": raw["generate_timing"], "verify": None}
                    if timing_doc is not None and timing_doc.get("verify") is None:
                        atomic_json(timing_path, {**timing_doc, "verify": receipt(started, time.monotonic()-clock)})
                    result.update(provenance)
                    if raw.get("technical_synthetic"):
                        result.update(technical_synthetic=True, scientific_phase_allowed=False)
                    checked[key] = result
                    if not result["verified"]:
                        rejected.append(item)
                remaining = rejected
                if not remaining:
                    break
            exhausted = {str(i["id"]) for i in remaining}
        finally:
            # Corpus order stays item-major, identical to the serial algorithm.
            for item in active:
                for attempt in range(1, attempts + 1):
                    key = (str(item["id"]), attempt)
                    if key in raw_by_key:
                        raw_rows.append(raw_by_key[key])
                    result = checked.get(key)
                    if result is None:
                        continue
                    if result["verified"]:
                        accepted.append(result)
                        break
                    rejects.append({**result, "raw_text": raw_by_key[key]["text"]})
            write_jsonl(out / ("rewrites.jsonl" if rewriting else "demonstrations.jsonl"), accepted)
            write_jsonl(out / "rejections.jsonl", rejects)
            write_jsonl(out / "raw_outputs.jsonl", raw_rows)
            write_jsonl(out / "timings.jsonl", [json.loads((timing_journal/f"{sha(str(row['id']))}-{row['attempt']}.json").read_text())
                        for row in raw_rows if (timing_journal/f"{sha(str(row['id']))}-{row['attempt']}.json").exists()])
            shard_teachers = [d for d in demonstrations if str(d["id"]) in shard_ids] if rewriting else None
            report = coverage(shard_items, shard_teachers, rewrites=accepted, rewrite_exhausted=exhausted,
                              stand_in_teacher=provenance["stand_in_teacher"]) if rewriting else coverage(shard_items, accepted, exhausted, stand_in_teacher=provenance["stand_in_teacher"])
            atomic_json(out / "coverage.json", {**report, **provenance,
                "reason_count_unit": "rejected_attempt", "primary_reason_counts": dict(Counter(r["primary_reason"] for r in rejects)),
                "reason_counts": dict(Counter(reason for r in rejects for reason in r["reasons"]))})
        return accepted



def audit_outputs(items, records):
    """Count chosen numeric ranks and VALUE consistency; missing answers never agree."""
    from kit.v4_contract import check_synthetic_rows
    check_synthetic_rows(records)
    lookup = {str(i["id"]): i for i in items}
    ranks, positions, family_ranks = Counter(), Counter(), defaultdict(Counter)
    pairs = defaultdict(dict)
    for record in records:
        item = lookup[str(record["id"])]
        if item["task"] != "chemistry":
            raise ValueError("option audit is Chemistry-only")
        options = options_of(item)
        perm = record.get("permutation")
        if perm is not None:
            if sorted(perm) != list(range(len(options))):
                raise ValueError("invalid option permutation")
            options = {chr(65+i): list(options.values())[j] for i, j in enumerate(perm)}
        choice, _ = chemistry_choice(item, record["text"], blind=False, options=options)
        value = options.get(choice)
        key = "permuted" if perm is not None else "original"
        if key in pairs[str(item["id"])]:
            raise ValueError("duplicate audit output")
        pairs[str(item["id"])][key] = value
        if key == "original" and choice:
            positions[choice] += 1
            numeric = numeric_value(value)
            if numeric is not None and all(numeric_value(v) is not None for v in options.values()):
                rank = str(1 + sum(numeric_value(v) < numeric for v in options.values()))
                ranks[rank] += 1
                family_ranks[family_of(item)][rank] += 1
    details = [{"id": i["id"], **pairs[str(i["id"])], "complete": all(pairs[str(i["id"])].get(k) is not None for k in ("original", "permuted")),
                "consistent": all(pairs[str(i["id"])].get(k) is not None for k in ("original", "permuted")) and pairs[str(i["id"])]["original"] == pairs[str(i["id"])]["permuted"]} for i in items]
    complete = sum(d["complete"] for d in details)
    consistent = sum(d["consistent"] for d in details)
    return {"schema": "kit-v4-teacher-audit.v1", "training_only": True, "options_visible": True,
            "answer_rank_counts": dict(ranks), "answer_position_counts": dict(positions),
            "rank_counts_by_family": {k: dict(v) for k, v in family_ranks.items()},
            "total": len(items), "complete_pairs": complete, "consistent_pairs": consistent,
            "consistency": consistent / complete if complete else None, "pairs": details}


def load_student_tokenizer(path=None):
    path=input_path("QWEN3_8B_TOKENIZER",path)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def tokenizer_identity(path):
    """Chat-template changes must invalidate restart just like vocabulary changes."""
    return {f.name: sha(f.read_bytes()) for f in sorted(Path(path).iterdir()) if f.is_file() and f.suffix in (".json", ".jinja")}


def runtime_versions():
    versions = {"python": sys.version}
    for package in ("vllm", "transformers", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def local_identity(path, model_name="Qwen/Qwen3.6-27B"):
    """Snapshot revision when known, plus hashes of all local model/tokenizer files."""
    path = Path(path).resolve()
    config = json.loads((path / "config.json").read_text())
    candidate = path.name if re.fullmatch(r"[a-f0-9]{40}", path.name) else config.get("_commit_hash", "")
    revision = candidate if re.fullmatch(r"[a-f0-9]{40}", candidate or "") else None
    hashes = {}
    for file in sorted(path.iterdir()):
        if file.is_file() and file.suffix in (".json", ".jinja", ".safetensors", ".bin"):
            digest = hashlib.sha256()
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            hashes[file.name] = digest.hexdigest()
    return {"model": model_name, "model_dir": str(path), "model_revision": revision,
            "immutable_revision_known": bool(revision), "model_config": config, "model_file_hashes": hashes}


class VLLMGenerator:
    def __init__(self, model, max_new_tokens, *, input_cap=INPUT_CAP, max_num_seqs=H100_MAX_NUM_SEQS, probe_mode=False, tensor_parallel_size=1, allow_batch_fallback=False):
        from kit.v4_timing import utc_now, receipt
        started=utc_now();clock=time.monotonic()
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ['VLLM_BATCH_INVARIANT']='1'
        os.environ['VLLM_ATTENTION_BACKEND']='FLASH_ATTN'
        from vllm import LLM, SamplingParams
        self.params = SamplingParams
        self.max_new_tokens = max_new_tokens
        self.input_cap = input_cap
        if tensor_parallel_size not in (1,2):raise ValueError('registered teacher/rewrite TP is 2/1')
        kwargs=dict(model=str(model),tokenizer=str(model),tensor_parallel_size=tensor_parallel_size,
                    distributed_executor_backend='mp' if tensor_parallel_size==2 else 'uni',dtype='bfloat16',
                    max_model_len=8192,max_num_seqs=max_num_seqs,gpu_memory_utilization=.90,seed=27101)
        if os.environ.get('V4_PHASE0_INFERENCE')=='1':
            import vllm
            if vllm.__version__!='0.18.0':raise ValueError('phase0 teacher requires vLLM 0.18.0')
            kwargs.update(attention_config={'backend':'FLASH_ATTN'},gdn_prefill_backend='triton')
        if probe_mode:kwargs['enforce_eager']=True
        self.batch_policy={'attention_backend':'FLASH_ATTN','batch_invariant':True,'engine_starts':1,
                           'serial_equivalence_guaranteed':True,'regeneration_guaranteed':True}
        if os.environ.get('V4_PHASE0_INFERENCE')=='1':
            self.batch_policy.update(vllm='0.18.0',gdn_prefill_backend='triton')
        try:self.engine=LLM(**kwargs)
        except Exception as error:
            if not allow_batch_fallback:raise
            self.batch_policy.update(batch_invariant=False,engine_starts=2,
                batch_invariance='unavailable for this model',startup_failure=type(error).__name__+': '+str(error),
                serial_equivalence_guaranteed=False,regeneration_guaranteed=False)
            os.environ['VLLM_BATCH_INVARIANT']='0'
            self.engine=LLM(**kwargs)
        self.tokenizer = self.engine.get_tokenizer()
        self.reload_timing=receipt(started,time.monotonic()-clock,batch_policy=self.batch_policy)

    def __call__(self, messages, attempt, *, seed=None, temperature=None, top_p=1):
        default_seed, default_temperature = SCHEDULE[attempt - 1]
        seed = default_seed if seed is None else seed
        temperature = default_temperature if temperature is None else temperature
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_tokens=len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if input_tokens > self.input_cap:
            return {"rendered_prompt_tail": prompt[-1024:], "input_tokens":input_tokens,"input_cap": self.input_cap, "text": "", "finish_reason": "input_cap", "input_over_cap": True, "teacher_tokens": 0, "rendered_prompt_sha256": sha(prompt)}
        params = self.params(seed=seed, temperature=temperature, top_p=top_p, top_k=-1, repetition_penalty=1,
                             n=1, max_tokens=self.max_new_tokens)
        from kit.v4_timing import utc_now, receipt
        started=utc_now();clock=time.monotonic()
        result = self.engine.generate([prompt], params, use_tqdm=False)[0].outputs[0]
        return {"text": result.text, "finish_reason": result.finish_reason, "stop_reason": result.stop_reason,
                "generate_timing":receipt(started,time.monotonic()-clock),"reload_timing":self.reload_timing,
                "input_tokens":input_tokens,"teacher_tokens": len(result.token_ids), "rendered_prompt_sha256": sha(prompt), "rendered_prompt_tail": prompt[-1024:], "input_cap": self.input_cap}

    def generate_many(self, requests):
        """One engine call per round, with independent registered request parameters."""
        from kit.v4_timing import utc_now, receipt
        prompts, params, positions, rows = [], [], [], []
        for index, request in enumerate(requests):
            prompt = self.tokenizer.apply_chat_template(request["messages"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            count = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            row = {"input_tokens": count, "input_cap": self.input_cap,
                   "rendered_prompt_sha256": sha(prompt), "rendered_prompt_tail": prompt[-1024:]}
            rows.append(row)
            if count > self.input_cap:
                row.update(text="", finish_reason="input_cap", input_over_cap=True, teacher_tokens=0)
                continue
            positions.append(index)
            prompts.append(prompt)
            params.append(self.params(seed=request["seed"], temperature=request.get("temperature", .7),
                top_p=request.get("top_p", .95), top_k=-1, repetition_penalty=1, n=1, max_tokens=self.max_new_tokens))
        if not prompts:
            return rows
        started, clock = utc_now(), time.monotonic()
        outputs = self.engine.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise ValueError("incomplete generation batch")
        timing = receipt(started, (time.monotonic()-clock)/len(outputs), batch_requests=len(outputs))
        for index, output in zip(positions, outputs):
            result = output.outputs[0]
            rows[index].update(text=result.text, finish_reason=result.finish_reason, stop_reason=result.stop_reason,
                token_ids=list(result.token_ids), teacher_tokens=len(result.token_ids),
                generate_timing=timing, reload_timing=self.reload_timing)
        return rows


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authors_rows(task, sdpo=None, finqa_root=None):
    """Unit A's exact source mappers, loaded read-only; no copied prompt construction."""
    if task == "finqa":
        root = input_path("FINQA_ROOT",finqa_root)
        from kit.v4_pool import select_finqa_pool, FINQA_HASHES
        if sha((root / "train.json").read_bytes()) != FINQA_HASHES["train"]:
            raise ValueError("Unit A FinQA source hash changed")
        items = select_finqa_pool(finqa.load(root, "train"))
        if len(items) != 1441:
            raise ValueError("Unit A FinQA original pool must contain 1,441 IDs")
        return finqa.rows_for_trainer(items, "train")
    root = input_path("SDPO_DIR",sdpo)
    from kit.v4_datasets import check_preprocess
    check_preprocess(root)
    from kit.v4_pool import freeze_chemistry
    freeze_chemistry(root)  # validate frozen source bytes, not just the current row mapper
    mapper = load_module("v4_teacher_authors_preprocess", root / "data/preprocess.py").make_map_fn("train")
    text = (root / "datasets/sciknoweval/chemistry/train.json").read_text()
    items = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.split("\n") if line.strip()]
    taxonomy = json.loads(resource("k8b-chemistry-templates.json").read_text())
    eligible = {str(q["idx"]) for q in taxonomy["questions"] if q["split"] == "train" and q["family"] not in EXCLUDED}
    rows = [mapper(copy.deepcopy(item), n) for n, item in enumerate(items) if str(item["idx"]) in eligible]
    if len(rows) != 1441 or {str(r["extra_info"]["index"]) for r in rows} != eligible:
        raise ValueError("Unit A Chemistry original pool must contain its frozen 1,441 IDs")
    return rows


def prompt_bytes(messages):
    """Canonical UTF-8 serialization, preserving every role and content byte."""
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def check_pool_provenance(items, sdpo=None, finqa_root=None):
    """Refuse foreign IDs, prompt bytes, golds and families before any inference."""
    validate_pool(items)
    sources = {task: {str(r["extra_info"]["index"]): r for r in authors_rows(task, sdpo, finqa_root)} for task in {i["task"] for i in items}}
    seen_indices = defaultdict(set)
    checked = {}
    for item in items:
        key = str(item["id"])
        index = str(item.get("idx", item.get("extra_info", {}).get("index", key.removeprefix("sciknoweval-train-") if item["task"] == "chemistry" else key)))
        valid_ids = {index, "sciknoweval-train-" + index} if item["task"] == "chemistry" else {index}
        if key not in valid_ids:
            raise ValueError("pool id differs from Unit A training identity: " + key)
        if index in seen_indices[item["task"]]:
            raise ValueError("duplicate Unit A pool index: " + index)
        seen_indices[item["task"]].add(index)
        if index not in sources[item["task"]]:
            raise ValueError("pool id absent from Unit A source: " + key)
        row = copy.deepcopy(sources[item["task"]][index])
        expected, actual = prompt_bytes(row["prompt"]), prompt_bytes(student_messages(item))
        if expected != actual or str(row["reward_model"]["ground_truth"]) != str(gold_of(item)):
            raise ValueError("pool prompt/gold differs from Unit A row: " + key)
        if item["task"] == "chemistry":
            if "options" in item and options_of(item) != options_of({"messages": row["prompt"], "question": row["extra_info"]["problem"]}):
                raise ValueError("pool options differ from Unit A row: " + key)
            expected_family = classify_family(normalize(row["extra_info"]["description"]))
            if family_of(item) != expected_family:
                raise ValueError("pool family differs from Unit A taxonomy: " + key)
        checked[key] = (row, index, expected, actual)
    return checked


def common_training_rows(items, demonstrations, rewrites, tokenizer, sdpo=None, finqa_root=None):
    """Materialize both targets on the intersection and prove original prompt identity."""
    demos = verified_lookup(items, demonstrations, tokenizer)
    own = verified_lookup(items, rewrites, tokenizer)
    if not set(own) <= set(demos):
        raise ValueError("rewrite has no verified demonstration")
    checked = check_pool_provenance(items, sdpo, finqa_root)
    tables = {task: [] for task in {i["task"] for i in items}}
    proof = {}
    for item in items:
        key = str(item["id"])
        row, index, expected, actual = checked[key]
        if key not in demos or key not in own:
            continue
        row["extra_info"].update(demonstration=demos[key]["text"], rewrite=own[key]["text"],
                                  teacher_model=demos[key].get("teacher_model", "unspecified"),
                                  stand_in_teacher=demos[key].get("stand_in_teacher", False),
                                  technical_synthetic=bool(demos[key].get("technical_synthetic") or own[key].get("technical_synthetic")))
        tables[item["task"]].append(row)
        proof[key] = {"source_index": index, "authors_prompt_sha256": sha(expected), "training_prompt_sha256": sha(actual), "byte_identical": True}
    return tables, proof


def corpus_directories(root):
    root = Path(root)
    return [root] if (root / "pool.jsonl").is_file() else sorted(p.parent for p in root.rglob("pool.jsonl"))


def scientific_identity_allowed(teachers, rewriters):
    """Flags and model names cannot qualify unpinned weights or changed decoding."""
    if not REGISTERED_TEACHER_REVISION or not REGISTERED_TEACHER_WEIGHT_HASHES or not INITIAL_8B_WEIGHT_HASHES or not teachers or not rewriters:
        return False
    from kit.v4_contract import technical_synthetic
    if technical_synthetic(teachers) or technical_synthetic(rewriters):
        return False
    for docs, role, cap in ((teachers, "external_teacher", TEACHER_NEW_TOKENS), (rewriters, "frozen_initial_student", REWRITE_NEW_TOKENS)):
        for doc in docs:
            if (doc.get("stand_in_teacher") is not False or doc.get("generator_role") != role
                    or doc.get("attempts") != 4 or doc.get("max_new_tokens") != cap
                    or doc.get("thinking") is not False or doc.get("response_cap") != RESPONSE_CAP
                    or doc.get("immutable_revision_known") is not True):
                return False
            if role == "external_teacher":
                weights = {name: digest for name, digest in doc.get("model_file_hashes", {}).items() if name.endswith((".safetensors", ".bin"))}
                if (doc.get("model_revision") != REGISTERED_TEACHER_REVISION
                        or weights != REGISTERED_TEACHER_WEIGHT_HASHES
                        or (doc.get("model_config", {}).get("text_config") or doc.get("model_config", {})).get("vocab_size") != 248320):
                    return False
            else:
                weights = {name: digest for name, digest in doc.get("model_file_hashes", {}).items() if name.endswith((".safetensors", ".bin"))}
                if doc.get("model_revision") != INITIAL_8B_REVISION or weights != INITIAL_8B_WEIGHT_HASHES:
                    return False
    return True


def training_set(demonstrations, rewrites, out, tokenizer, sdpo=None, finqa_root=None):
    """Build shippable common task parquets with immutable source and prompt receipts."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    items, demos, own, teacher_manifests, rewrite_manifests, sources = [], [], [], [], [], {}
    for dirs, filename, records, manifests in ((corpus_directories(demonstrations), "demonstrations.jsonl", demos, teacher_manifests),
                                              (corpus_directories(rewrites), "rewrites.jsonl", own, rewrite_manifests)):
        if not dirs:
            raise ValueError("corpus directory has no saved original pool")
        for directory in dirs:
            pool = load_pool(directory / "pool.jsonl")
            doc = json.loads((directory / "manifest.json").read_text())
            if doc["pool_hash"] != sha(json.dumps(pool, sort_keys=True)):
                raise ValueError("saved original pool changed")
            if filename == "demonstrations.jsonl":
                items.extend(pool)
            else:
                # Rewrite must carry the very same original pool, including teacher-uncovered IDs.
                teacher_pools = [load_pool(d / "pool.jsonl") for d in corpus_directories(demonstrations)]
                if pool not in teacher_pools:
                    raise ValueError("rewrite and teacher original pools differ")
                matched = next(d for d in corpus_directories(demonstrations) if load_pool(d / "pool.jsonl") == pool)
                if doc["demonstrations_hash"] != sha(json.dumps(read_jsonl(matched / "demonstrations.jsonl"), sort_keys=True)):
                    raise ValueError("rewrite demonstration source changed")
            loaded = read_jsonl(directory / filename)
            if any(r.get("stand_in_teacher") != doc["stand_in_teacher"] or r.get("teacher_model") != doc["teacher_model"] for r in loaded):
                raise ValueError("target provenance differs from corpus manifest")
            records.extend(loaded)
            manifests.append(doc)
            sources[str(directory.resolve())] = {name: sha((directory / name).read_bytes()) for name in (filename, "pool.jsonl", "manifest.json")}
    if len({str(i["id"]) for i in items}) != len(items):
        raise ValueError("duplicate original pool id")
    if {m["stand_in_teacher"] for m in teacher_manifests + rewrite_manifests} not in ({False}, {True}):
        raise ValueError("mixed stand-in and production corpora")
    stand_in = teacher_manifests[0]["stand_in_teacher"]
    tables, proof = common_training_rows(items, demos, own, tokenizer, sdpo, finqa_root)
    rewrite_reports = [json.loads((d / "coverage.json").read_text()) for d in corpus_directories(rewrites)]
    report = coverage(items, demos, rewrites=own,
                      rewrite_exhausted={str(i) for r in rewrite_reports for i in r["rewrite_uncovered_ids"]}, stand_in_teacher=stand_in)
    # Pending source work cannot masquerade as a completed corpus.
    report["complete"] = all(json.loads((d / "coverage.json").read_text())["complete"] for d in corpus_directories(demonstrations) + corpus_directories(rewrites))
    report["qualified"] = report["gate_pass"] and report["complete"]
    from kit.v4_contract import technical_synthetic
    synthetic = technical_synthetic(demos + own + teacher_manifests + rewrite_manifests)
    report["technical_synthetic"] = synthetic
    report["model_identity_qualified"] = scientific_identity_allowed(teacher_manifests, rewrite_manifests)
    report["scientific_phase_allowed"] = not synthetic and report["coverage_phase_allowed"] and report["complete"] and set(tables) == {"chemistry", "finqa"} and report["model_identity_qualified"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for task, rows in tables.items():
        path = out / (task + ".parquet")
        if rows:
            table = pa.Table.from_pylist(rows)
        else:
            sample = copy.deepcopy(authors_rows(task, sdpo, finqa_root)[0])
            sample["extra_info"].update(demonstration="", rewrite="", teacher_model="", stand_in_teacher=stand_in)
            table = pa.Table.from_pylist([sample]).slice(0, 0)
        pq.write_table(table, path, compression="zstd", row_group_size=32)
        if pq.read_table(path).to_pylist() != rows:
            raise ValueError("parquet round-trip changed common rows")
        files[task] = {"rows": len(rows), "sha256": sha(path.read_bytes()),
                       "technical_synthetic": any(r["extra_info"].get("technical_synthetic") for r in rows),
                       "technical_synthetic_rows": sum(bool(r["extra_info"].get("technical_synthetic")) for r in rows)}
    atomic_json(out / "coverage.json", report)
    sdpo_path = input_path("SDPO_DIR",sdpo) if "chemistry" in tables else None
    finance_path = input_path("FINQA_ROOT",finqa_root) if "finqa" in tables else None
    author_files = ([sdpo_path / "data/preprocess.py", sdpo_path / "datasets/sciknoweval/chemistry/train.json"] if "chemistry" in tables else [])
    author_files += [ROOT / "kit/beds/finqa.py", finance_path / "train.json"] if "finqa" in tables else []
    atomic_json(out / "manifest.json", {"schema": "kit-v4-common-training.v1",
                "authors_source_hashes": {str(p.resolve()): sha(p.read_bytes()) for p in author_files}, "stand_in_teacher": stand_in,
                "teacher_models": sorted({m["teacher_model"] for m in teacher_manifests}), "sources": sources,
                "files": files, "prompt_identity": proof, "coverage": report, "technical_synthetic": synthetic,
                "targets": {"F": "extra_info.demonstration", "D": "extra_info.demonstration", "R": "extra_info.rewrite", "S": None}})
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    merge = subs.add_parser("merge-shards")
    merge.add_argument("--shards", nargs="+", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument("--tokenizer-8b", default=os.environ.get("QWEN3_8B_TOKENIZER"))
    ts = subs.add_parser("training-set")
    ts.add_argument("--demonstrations", required=True)
    ts.add_argument("--rewrites", required=True)
    ts.add_argument("--out", required=True)
    ts.add_argument("--sdpo")
    ts.add_argument("--finqa-root")
    ts.add_argument("--tokenizer-8b", default=os.environ.get("QWEN3_8B_TOKENIZER"))
    r = subs.add_parser("rewrite")
    r.add_argument("--demonstrations", required=True)
    g = subs.add_parser("generate")
    for sub in (g, r):
        sub.add_argument("--shard-index", type=int, help="zero-based hash shard; writes a distinct child of --out")
        sub.add_argument("--shard-count", type=int, default=1)
        sub.add_argument("--tp",type=int,choices=(1,2),default=1)
        sub.add_argument("--model", required=True)
        sub.add_argument("--task", choices=("chemistry", "finqa"), required=True)
        sub.add_argument("--tokenizer-8b", default=os.environ.get("QWEN3_8B_TOKENIZER"))
        sub.add_argument("--revision", help="immutable upstream commit for a copied local model directory")
        sub.add_argument("--stand-in-teacher", action="store_true", help="1.7B smoke only; never a scientific qualification")
    g.add_argument("--attempts", type=int, default=4, choices=range(1, 5))
    a = subs.add_parser("audit")
    a.add_argument("--model")
    a.add_argument("--responses", help="saved original/permuted outputs; no GPU")
    a.add_argument("--limit", type=int, default=20)
    for sub in (g, r, a):
        sub.add_argument("--pool", required=True)
        sub.add_argument("--out", required=True)
        sub.add_argument("--max-new-tokens", type=int, default=REWRITE_NEW_TOKENS if sub is r else TEACHER_NEW_TOKENS)
    args = parser.parse_args(argv)
    if args.action == "merge-shards":
        items=load_pool(Path(args.shards[0])/"pool.jsonl")
        check_pool_provenance(items)
        merge_shards(args.shards, args.out, load_student_tokenizer(args.tokenizer_8b))
        return 0
    if args.action == "training-set":
        training_set(args.demonstrations, args.rewrites, args.out, load_student_tokenizer(args.tokenizer_8b), args.sdpo, args.finqa_root)
        return 0
    if not 1 <= args.max_new_tokens <= 4096:
        parser.error("max-new-tokens must be 1..4096 (8,192 teacher context)")
    items = load_pool(args.pool, getattr(args, "task", None))
    check_pool_provenance(items)
    if args.action in ("generate", "rewrite"):
        if args.revision and not re.fullmatch(r"[a-f0-9]{40}", args.revision):
            parser.error("revision must be an immutable 40-hex commit")
        tokenizer = load_student_tokenizer(args.tokenizer_8b)
        rewriting = args.action == "rewrite"
        teacher_doc = json.loads((Path(args.demonstrations) / "manifest.json").read_text()) if rewriting else None
        if teacher_doc and teacher_doc["pool_hash"] != sha(json.dumps(items, sort_keys=True)):
            parser.error("rewrite requires the original unfiltered teacher pool")
        if rewriting and not json.loads((Path(args.demonstrations) / "coverage.json").read_text())["complete"]:
            parser.error("finish teacher generation before rewriting")
        if rewriting and args.stand_in_teacher and not teacher_doc["stand_in_teacher"]:
            parser.error("cannot relabel production demonstrations as stand-ins")
        stand_in = args.stand_in_teacher or bool(teacher_doc and teacher_doc["stand_in_teacher"])
        model_name = "Qwen/Qwen3-1.7B (stand-in)" if stand_in else "Qwen/Qwen3-8B (frozen initial student)" if rewriting else "Qwen/Qwen3.6-27B"
        manifest = {"schema": "kit-v4-teacher.v2", **local_identity(args.model, model_name),
                    "stand_in_teacher": stand_in, "teacher_model": teacher_doc["teacher_model"] if rewriting else model_name,
                    "generator_role": "frozen_initial_student" if rewriting else "external_teacher",
                    "module_sha256": sha(Path(__file__).read_bytes()), "taxonomy_sha256": sha(resource("k8b-chemistry-templates.json").read_bytes()) if args.task=="chemistry" else None,
                    "pool_file_sha256": sha(Path(args.pool).read_bytes()), "student_tokenizer_dir": str(Path(args.tokenizer_8b).resolve()),
                    "student_tokenizer_sha256": sha((Path(args.tokenizer_8b) / "tokenizer.json").read_bytes()),
                    "student_tokenizer_files": tokenizer_identity(args.tokenizer_8b), "runtime_versions": runtime_versions(),
                    "verifier_source_hashes": {name: sha((ROOT / name).read_bytes()) for name in ("kit/beds/finqa.py", "kit/beds/chemistry.py", "kit/beds/_authors.py", "kit/v4_families.py")},
                    "thinking": False, "schedule": "sha256(v4-rewrite|id|attempt) first 32 bits modulo 2**31" if rewriting else SCHEDULE,
                    "teacher_token_limit_receipt": TEACHER_TOKEN_LIMIT_RECEIPT if not rewriting else None,
                    "batch_execution": {"max_num_seqs": H100_MAX_NUM_SEQS, "batch_invariant_requested": True, "actual_policy": "raw reload_timing.batch_policy", "rounds": "all uncovered at attempt k"},
                    "max_new_tokens": args.max_new_tokens, "response_cap": RESPONSE_CAP, "input_cap": REWRITE_INPUT_CAP if rewriting else INPUT_CAP,
                    "decoding": {"temperature": .7 if rewriting else "schedule", "top_p": .95 if rewriting else 1, "top_k": -1, "repetition_penalty": 1, "n": 1, "tp": args.tp, "executor_backend": "mp" if args.tp==2 else "uni", "v1_multiprocessing": False, "dtype": "bfloat16", "max_model_len": 8192}}
        if args.revision:
            known = manifest["model_revision"]
            if known and known != args.revision:
                parser.error("revision disagrees with model snapshot")
            manifest.update(model_revision=args.revision, immutable_revision_known=True)
        # Normalize tuples exactly as they will be read back from JSON.
        manifest = json.loads(json.dumps(manifest))
        generator = None
        def lazy_generate(messages, attempt, **kwargs):
            nonlocal generator
            if generator is None:
                generator = VLLMGenerator(args.model, args.max_new_tokens, input_cap=REWRITE_INPUT_CAP if rewriting else INPUT_CAP,
                    **({"tensor_parallel_size":args.tp,"allow_batch_fallback":not rewriting} if args.tp!=1 else {}))
            return generator(messages, attempt, **kwargs)
        def lazy_many(requests):
            nonlocal generator
            if generator is None:
                generator = VLLMGenerator(args.model, args.max_new_tokens, input_cap=REWRITE_INPUT_CAP if rewriting else INPUT_CAP,
                    **({"tensor_parallel_size":args.tp,"allow_batch_fallback":not rewriting} if args.tp!=1 else {}))
            if hasattr(generator, "generate_many"):
                return generator.generate_many(requests)
            return [generator(r["messages"], r["attempt"], seed=r["seed"], temperature=r["temperature"], top_p=r["top_p"])
                    if rewriting else generator(r["messages"], r["attempt"]) for r in requests]
        lazy_generate.generate_many = lazy_many
        if rewriting:
            rewrite_pool(items, read_jsonl(Path(args.demonstrations) / "demonstrations.jsonl"), args.out, lazy_generate, tokenizer, manifest, shard_index=args.shard_index, shard_count=args.shard_count)
        else:
            generate_pool(items, args.out, lazy_generate, tokenizer, manifest, args.attempts, shard_index=args.shard_index, shard_count=args.shard_count)
    else:
        if any(i["task"] != "chemistry" for i in items):
            parser.error("option audit is Chemistry-only")
        if args.responses:
            records = read_jsonl(args.responses)
        else:
            if not args.model or args.limit < 1:
                parser.error("audit needs --responses or --model and positive --limit")
            items = sorted(items, key=lambda i: sha(str(i["id"])))[:args.limit]
            generator = VLLMGenerator(args.model, args.max_new_tokens)
            records = []
            for item in items:
                n = len(options_of(item))
                perm = list(range(1, n)) + [0]
                for permutation in (None, perm):
                    records.append({"id": item["id"], "permutation": permutation,
                                    **generator(teacher_messages(item, visible_options=True, permutation=permutation), 1)})
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        audit = audit_outputs(items, records)
        audit.update(pool_sha256=sha(Path(args.pool).read_bytes()), raw_outputs=records)
        if args.model:
            audit.update(local_identity(args.model), thinking=False, schedule=SCHEDULE, max_new_tokens=args.max_new_tokens)
        atomic_json(out / "audit.json", audit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
