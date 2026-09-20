#!/usr/bin/env python3
"""Score a saved model on ONE bed's held-out questions, in the same deterministic mode as the forgetting panel.

    python eval_bed.py generate --bed spider --root /data/spider_data --model /work/models/stage-a --out /work/k3/a-spider-a1
    python eval_bed.py score    --bed spider --root /data/spider_data --responses RESPONSES.jsonl --out DIR   # no GPU

`kit/score_forgetting.py` measures what a model has LOST on three general panels. This file measures
what it has on the job itself -- the Spider held-out 100, the GSM8K held-out 300, the FinQA test split
-- and it must be measured the same way, because the K3 readout subtracts one of these numbers from
another (`kit/delta.py`). So `generate` uses the SAME decoding constants, the SAME engine settings,
the SAME chat rendering with thinking disabled, and records the SAME machine-and-mode fingerprint.

Those lines are DUPLICATED from score_forgetting.py on purpose: that file is the kit's reference for
what a comparable scoring is, and a shared helper would let a change there move a number here
silently. `tests/test_kit_k3_tools.py` compares the two files' decoding constants, engine settings,
determinism switches and fingerprint function and fails if either side moves.

Read the numbers from `bed-score.json`: `n`, `correct`, `accuracy`, `incorrect_format` and
`total_correct` are top-level numbers, so a campaign bar can gate on them, and `delta.py` subtracts
two of these files. `responses.jsonl` holds every answer, so `score` can re-score them on a CPU after
the scoring rule is examined -- the K3 pilot's first act is to read the answers, not the number.

`generate` needs vLLM and one GPU. `score` needs the bed's own data (Spider's databases, GSM8K's
test split, FinQA's json) and no GPU at all. Nothing is overwritten: an existing --out is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "kit-bed-score.v1"

# ---- duplicated from score_forgetting.py, deliberately; the drift test compares them --------------
DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "max_tokens": 2048}
ENGINE = {"dtype": "bfloat16", "tensor_parallel_size": 1, "gpu_memory_utilization": 0.85, "max_model_len": 4096}
# --------------------------------------------------------------------------------------------------

# One bed's answer budget. All three are 2,048 new tokens, the same cap the forgetting panel uses, so
# a truncated answer here means the same thing it means there. Kept per bed because a bed whose
# answers are longer would need its own number, and that change must be visible.
MAX_NEW_TOKENS = {"spider": 2048, "gsm8k": 2048, "finqa": 2048}
BED_FILES = {"spider": HERE / "beds" / "spider.py", "gsm8k": HERE / "beds" / "gsm8k.py",
             "finqa": HERE / "beds" / "finqa.py"}
DEFAULT_SPLIT = {"spider": "heldout", "gsm8k": "heldout", "finqa": "test"}
SPLITS = {"spider": ("train", "heldout"), "gsm8k": ("train", "test", "heldout"), "finqa": ("train", "dev", "test")}


class EvalBedError(ValueError):
    """The request, the bed's data or the responses are wrong; no number is produced."""


def load_bed(name: str):
    path = BED_FILES[name]
    spec = importlib.util.spec_from_file_location("kit_bed_%s" % name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_dir(path: Path) -> Path:
    if path.exists():
        raise EvalBedError("refusing to overwrite %s: choose a new --out" % path)
    path.mkdir(parents=True)
    return path


def render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def machine_fingerprint(deterministic: bool = True) -> dict:
    """What has to be equal for two scorings to be comparable: the physical GPU, the software, and the decoding mode."""
    import platform, socket, subprocess                                      # noqa: E401,PLC0415
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        smi = []
    versions = {}
    for name in ("torch", "vllm", "transformers"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:                                                    # noqa: BLE001
            versions[name] = None
    visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
    fingerprint = {"hostname": socket.gethostname(), "gpus": sorted(line.strip() for line in smi), "cuda_visible_devices": visible,
                   "versions": versions, "python": platform.python_version(), "deterministic": deterministic}
    fingerprint["id"] = hashlib.sha256(json.dumps({k: fingerprint[k] for k in ("gpus", "cuda_visible_devices", "versions", "deterministic")},
                                                  sort_keys=True).encode()).hexdigest()[:16]
    return fingerprint


# ------------------------------------------------------------------------------- the three beds
def bed_root(args) -> str:
    """The directory holding the bed's own data. Spider may take it from SPIDER_ROOT, as its bed does."""
    root = args.root or (os.environ.get("SPIDER_ROOT") if args.bed == "spider" else None)
    if not root:
        raise EvalBedError("--root is required: the directory holding the %s data%s"
                           % (args.bed, " (or set SPIDER_ROOT)" if args.bed == "spider" else ""))
    if not Path(root).is_dir():
        raise EvalBedError("--root %s is not a directory" % root)
    return str(root)


def split_of(args) -> str:
    split = args.split or DEFAULT_SPLIT[args.bed]
    if split not in SPLITS[args.bed]:
        raise EvalBedError("the %s bed has no %r split (it has %s)" % (args.bed, split, ", ".join(SPLITS[args.bed])))
    return split


def items_of(module, args) -> list:
    """[{'id', 'prompt', 'ground_truth'}] for the bed's split, built from the partner's own copy of the data."""
    root, split = bed_root(args), split_of(args)
    if args.bed == "spider":
        members = module.load_members(root, split, module.load_split())
        items = [{"id": m["id"], "prompt": m["prompt"], "ground_truth": module.ground_truth_for(m)} for m in members]
    elif args.bed == "gsm8k":
        if split == "heldout":
            source = module.held_out(module.load(root, "test", expect_size=not args.allow_subset),
                                     module.panel_questions(), args.heldout_n)
        else:
            source = module.load(root, split, expect_size=not args.allow_subset)
        items = [{"id": item["id"], "prompt": module.render_prompt(item), "ground_truth": item["gold"]} for item in source]
    else:
        items = [{"id": item["id"], "prompt": module.render_prompt(item), "ground_truth": module.gold_of(item)}
                 for item in module.load(Path(root), split)]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise EvalBedError("the %s %s split produced no items" % (args.bed, split))
    return items


def score_one(module, args, response: str, ground_truth: str) -> dict:
    """One answer through the bed's own reward function: the same rule that rewarded it during training."""
    if args.bed == "spider":
        return module.compute_score(module.DATA_SOURCE, response, ground_truth, None, spider_root=bed_root(args))
    return module.compute_score(module.DATA_SOURCE, response, ground_truth, None)


# ------------------------------------------------------------------------------------- scoring
def read_responses(path: Path) -> dict:
    """{id: response} from a responses file. A duplicate answer is a refusal, never the last one silently."""
    answers: dict = {}
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise EvalBedError("%s has a line that is not JSON: %s" % (path, exc)) from exc
        if not isinstance(row, dict):
            raise EvalBedError("%s has a line that is not a JSON object" % path)
        key = str(row.get("id") or row.get("member_id") or "")
        if not key:
            raise EvalBedError("a response row has neither `id` nor `member_id`: %s" % json.dumps(row)[:120])
        if key in answers:
            raise EvalBedError("the response file has two answers for %s" % key)
        answers[key] = str(row.get("response") or "")
    if not answers:
        raise EvalBedError("no responses in %s" % path)
    return answers


def grade(module, args, items: list, answers: dict) -> dict:
    """Every item's answer scored by the bed. Missing or foreign ids are refused, never counted as wrong."""
    wanted = [item["id"] for item in items]
    missing = [key for key in wanted if key not in answers]
    extra = sorted(set(answers) - set(wanted))
    if missing or extra:
        raise EvalBedError("the responses do not match the %s %s split: %d missing (e.g. %s), %d unexpected (e.g. %s). "
                           "Filter the file to this bed's answers before scoring."
                           % (args.bed, split_of(args), len(missing), ", ".join(missing[:3]) or "-",
                              len(extra), ", ".join(extra[:3]) or "-"))
    correct = incorrect_format = 0
    per_item: dict = {}
    for item in items:
        result = score_one(module, args, answers[item["id"]], item["ground_truth"])
        per_item[item["id"]] = int(result["acc"])
        correct += int(result["acc"])
        incorrect_format += int(result["incorrect_format"])
    n = len(items)
    return {"n": n, "correct": correct, "total_correct": correct, "incorrect_format": incorrect_format,
            "accuracy": round(correct / n, 6), "per_item": per_item}


def write_result(out: Path, *, bed: str, split: str, graded: dict, extra: dict) -> dict:
    result = {"schema": SCHEMA, "bed": bed, "split": split,
              "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "decoding": DECODING, **graded, **extra}
    (out / "bed-score.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("%s %s: %d of %d correct (%.4f), %d answers in the wrong format"
          % (bed, split, graded["correct"], graded["n"], graded["accuracy"], graded["incorrect_format"]))
    print("wrote", out / "bed-score.json")
    return result


def items_digest(items: list) -> str:
    return hashlib.sha256("".join(item["id"] + "\n" for item in items).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------ subcommands
def cmd_generate(args) -> int:
    module = load_bed(args.bed)
    items = items_of(module, args)
    model = Path(args.model).resolve()
    if not (model / "config.json").is_file():
        raise EvalBedError("not a HuggingFace model directory: %s" % model)
    out = fresh_dir(Path(args.out))
    # Deterministic by default (KIT_DETERMINISTIC=0 turns it off, and the result then refuses comparison with a deterministic one).
    # Without it, two scorings of ONE model on ONE machine agreed on as few as 98 of 300 answers and flipped up to 7 verdicts; with it, 300 of 300.
    deterministic = os.environ.get("KIT_DETERMINISTIC", "1") == "1"
    eager = batch_invariant = deterministic     # eager: no torch.compile, no CUDA graphs, so cold and warm runs execute the same kernels
    if batch_invariant:                      # must be set before vLLM is imported; asks for kernels whose result does not depend on batching
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")       # the mode refuses to start without a named backend
    from transformers import AutoTokenizer                                  # noqa: PLC0415
    from vllm import LLM, SamplingParams                                    # noqa: PLC0415
    import vllm                                                             # noqa: PLC0415
    tokenizer = AutoTokenizer.from_pretrained(str(model))
    prompts = [render(tokenizer, item["prompt"]) for item in items]
    decoding = {**DECODING, "max_tokens": MAX_NEW_TOKENS[args.bed]}
    llm = LLM(model=str(model), enable_lora=False, **ENGINE, **({"enforce_eager": True, "seed": 0} if eager else {}))
    outputs = llm.generate(prompts, SamplingParams(n=1, **decoding))
    answers, rows = {}, []
    for item, output in zip(items, outputs):
        completion = output.outputs[0]
        answers[item["id"]] = completion.text
        rows.append({"bed": args.bed, "split": split_of(args), "id": item["id"], "response": completion.text,
                     "output_tokens": len(completion.token_ids), "finish_reason": completion.finish_reason})
    (out / "responses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    truncated = sum(r["finish_reason"] == "length" for r in rows)
    write_result(out, bed=args.bed, split=split_of(args), graded=grade(module, args, items, answers),
                 extra={"mode": "generate", "model": str(model), "items_sha256": items_digest(items),
                        "limit": int(args.limit or 0),
                        "max_new_tokens": MAX_NEW_TOKENS[args.bed], "truncated_at_max_tokens": truncated,
                        "engine": {**ENGINE, "vllm": vllm.__version__, "batch_invariant": batch_invariant,
                                   "eager": eager, "deterministic": deterministic},
                        "machine": machine_fingerprint(deterministic)})
    return 0


def cmd_score(args) -> int:
    out = Path(args.out)
    if out.exists():                                   # refuse before any work, and again before writing
        raise EvalBedError("refusing to overwrite %s: choose a new --out" % out)
    module = load_bed(args.bed)
    items = items_of(module, args)
    answers = read_responses(Path(args.responses))
    # Graded BEFORE the directory is made: a refused scoring must leave no directory behind, or the
    # never-overwrite rule would make the corrected re-run refuse itself.
    graded = grade(module, args, items, answers)
    out = fresh_dir(out)
    write_result(out, bed=args.bed, split=split_of(args), graded=graded,
                 extra={"mode": "score", "items_sha256": items_digest(items), "limit": int(args.limit or 0),
                        "responses_file": str(Path(args.responses).resolve()),
                        "responses_sha256": hashlib.sha256(Path(args.responses).read_bytes()).hexdigest()})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a model on one bed's held-out questions.")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("generate", "score"):
        p = sub.add_parser(name)
        p.add_argument("--bed", required=True, choices=sorted(BED_FILES))
        p.add_argument("--root", default=None, help="the directory holding the bed's data (Spider: or SPIDER_ROOT)")
        p.add_argument("--split", default=None,
                       help="default: " + ", ".join("%s=%s" % pair for pair in sorted(DEFAULT_SPLIT.items())))
        p.add_argument("--out", required=True)
        p.add_argument("--limit", type=int, help="score the first N items only; recorded in the result")
        p.add_argument("--heldout-n", type=int, default=300, help="GSM8K only: the size of the held-out set")
        p.add_argument("--allow-subset", action="store_true", help="GSM8K only: the copy is not the published dataset")
        if name == "generate":
            p.add_argument("--model", required=True)
        else:
            p.add_argument("--responses", required=True)
    args = parser.parse_args(argv)
    try:
        return {"generate": cmd_generate, "score": cmd_score}[args.action](args)
    except EvalBedError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    sys.exit(main())
