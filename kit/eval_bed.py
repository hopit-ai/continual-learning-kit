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
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
_tokens_spec = importlib.util.spec_from_file_location("kit_tokens_io", HERE / "tokens_io.py")
tokens_io = importlib.util.module_from_spec(_tokens_spec)          # generated token ids beside responses.jsonl
_tokens_spec.loader.exec_module(tokens_io)
SCHEMA = "kit-bed-score.v1"

# ---- duplicated from score_forgetting.py, deliberately; the drift test compares them --------------
DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "max_tokens": 2048}
ENGINE = {"dtype": "bfloat16", "tensor_parallel_size": 1, "gpu_memory_utilization": 0.85, "max_model_len": 4096}
# --------------------------------------------------------------------------------------------------

# One bed's answer budget. All three are 2,048 new tokens, the same cap the forgetting panel uses, so
# a truncated answer here means the same thing it means there. Kept per bed because a bed whose
# answers are longer would need its own number, and that change must be visible.
MAX_NEW_TOKENS = {"spider": 2048, "gsm8k": 2048, "finqa": 2048, "code": 2048, "chemistry": 2048, "toolalpaca": 2048}
PROMPT_BUDGET = 4096        # with `--max-new-tokens` the context grows to this plus the cap. Not the trainer's 2,048 prompt limit: 18 FinQA test prompts are longer (up to 2,558 tokens, docs/phase2/k1c/feasibility.md), and a prompt over the budget would be given less than the cap and still be counted as cut at it
# `chemistry` and `toolalpaca` (plan v3) are the SDPO authors' own two tasks: their data, their chat messages, their
# scorer, read from the pinned checkout (kit/beds/_authors.py). They are scored here and trained through the two
# ToolAlpaca launchers; they are NOT in kit/beds/rewards.py, which would need the checkout at import time.
BED_FILES = {"spider": HERE / "beds" / "spider.py", "gsm8k": HERE / "beds" / "gsm8k.py",
             "finqa": HERE / "beds" / "finqa.py", "code": HERE / "beds" / "code.py",
             "chemistry": HERE / "beds" / "chemistry.py", "toolalpaca": HERE / "beds" / "toolalpaca.py"}
DEFAULT_SPLIT = {"spider": "heldout", "gsm8k": "heldout", "finqa": "test", "code": "heldout", "chemistry": "test", "toolalpaca": "test"}
SPLITS = {"spider": ("train", "heldout"), "gsm8k": ("train", "test", "heldout"), "finqa": ("train", "dev", "test"), "code": ("train", "heldout"),
          "chemistry": ("train", "test"), "toolalpaca": ("train", "test")}


class EvalBedError(ValueError):
    """The request, the bed's data or the responses are wrong; no number is produced."""


def load_bed(name: str):
    path = BED_FILES[name]
    spec = importlib.util.spec_from_file_location("kit_bed_%s" % name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: the authors' scoring module each bed loads from the pinned checkout (kit/beds/_authors.py `scorer`)
AUTHORS_SCORERS = {"chemistry": "mcq", "toolalpaca": "tooluse"}


def _file_sha256(path: Path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def scorer_identity(bed: str) -> dict:
    """The implementation identity of this bed's scorer (package-4 review round 3, F5): the sha256 of every source file
    that decides a verdict -- this scoring module, the bed module (with kit/beds/_authors.py for the authors' beds), the
    authors' reward source it loads from $SDPO_DIR, and kit/canonical.py -- keyed by a name that holds no absolute path,
    and `scorer_sha256`, the sha256 of that map (sorted keys). A file that cannot be read is recorded as null, so the
    hash differs from any complete one."""
    files = {"kit/eval_bed.py": HERE / "eval_bed.py", "kit/beds/%s.py" % bed: BED_FILES[bed], "kit/canonical.py": HERE / "canonical.py"}
    if bed == "finqa":
        files["kit/beds/_authors.py"] = HERE / "beds" / "_authors.py"
    if bed in AUTHORS_SCORERS:
        files["kit/beds/_authors.py"] = HERE / "beds" / "_authors.py"
        root = os.environ.get("SDPO_DIR")
        name = "sdpo/verl/utils/reward_score/feedback/%s.py" % AUTHORS_SCORERS[bed]
        files[name] = (Path(root) / name[len("sdpo/"):]) if root else Path("/nonexistent")
    hashes = {name: _file_sha256(path) for name, path in sorted(files.items())}
    return {"scorer_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), "scorer_files": hashes}


def fresh_dir(path: Path) -> Path:
    if path.exists():
        raise EvalBedError("refusing to overwrite %s: choose a new --out" % path)
    path.mkdir(parents=True)
    return path


def render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def render_messages(tokenizer, messages: list) -> str:
    """A bed whose prompt is already a list of chat messages (the authors' tasks carry a system message): the same
    template call as `render`, which is the trainer's own (agent_loop: add_generation_prompt, thinking off), on the
    messages as given. `render` itself is left exactly as it is: it is pinned against score_forgetting.py's."""
    return tokenizer.apply_chat_template([{"role": str(m["role"]), "content": str(m["content"])} for m in messages],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)


def machine_fingerprint(deterministic: bool = True) -> dict:
    """What has to be equal for two scorings to be comparable: the physical GPU, the software, and the decoding mode."""
    import platform, socket, subprocess                                      # noqa: E401,PLC0415
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=300).stdout.strip().splitlines()
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
    root = args.root or {"spider": os.environ.get("SPIDER_ROOT"), "code": os.environ.get("LCB_ROOT")}.get(args.bed)
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
    if args.bed == "code":
        # the coding bed builds its own items; scoring needs the tests file `prepare` wrote, named by CODE_TESTS
        items = module.eval_items(root, split)
    elif args.bed == "spider":
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
    if os.environ.get("KIT_V4_ARM") or "v4_" in str(root):
        from kit.v4_contract import check_eval_items
        check_eval_items(items, args.bed, split)
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
    if os.environ.get("KIT_V4_ARM"):
        from kit.v4_contract import check_eval_items
        check_eval_items(items, args.bed, split_of(args))
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
              "decoding": DECODING, **graded, **scorer_identity(bed), **extra}      # `extra` may carry its own `decoding` (a cap override)
    (out / "bed-score.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("%s %s: %d of %d correct (%.4f), %d answers in the wrong format"
          % (bed, split, graded["correct"], graded["n"], graded["accuracy"], graded["incorrect_format"]))
    print("wrote", out / "bed-score.json")
    return result


def items_digest(items: list) -> str:
    return hashlib.sha256("".join(item["id"] + "\n" for item in items).encode("utf-8")).hexdigest()


def panel_labels(module, args) -> dict:
    """Carry a shortened FinQA smoke panel's identity into generated and rescored results."""
    if args.bed != "finqa":
        return {}
    root, split = Path(bed_root(args)), split_of(args)
    marker = root / "technical-smoke-panel.json"
    if not marker.is_file() or not (root / (split + ".parquet")).is_file():
        return {}
    labels = json.loads(marker.read_text())
    declared, published = labels["rows"][split], module.SPLIT_SIZES[split]
    if declared >= published:
        return {}
    return {"profile": "technical-smoke", "technical_smoke": True, "stand_in": labels["stand_in"],
            "scientific_phase_allowed": False,
            "technical_smoke_panel": {"declared_rows": declared, "published_rows": published,
                                      "marker_sha256": _file_sha256(marker)}}


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
    reload_started=datetime.now(timezone.utc).isoformat();reload_clock=time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(model))
    prompts = [render(tokenizer, item["prompt"]) if isinstance(item["prompt"], str) else render_messages(tokenizer, item["prompt"])
               for item in items]
    cap = int(args.max_new_tokens or MAX_NEW_TOKENS[args.bed])
    decoding = {**DECODING, "max_tokens": cap}
    # A longer cap than the bed's needs a longer context: room for the longest prompt (PROMPT_BUDGET) plus the cap.
    engine = {**ENGINE, "max_model_len": PROMPT_BUDGET + cap} if args.max_new_tokens else dict(ENGINE)   # the default scoring is untouched
    if args.max_model_len:                    # a budget audit scores two caps in ONE context, so the cap is the only thing that differs
        engine["max_model_len"] = int(args.max_model_len)
    from kit.v4_teacher import atomic_json
    expected={**ENGINE,'max_model_len':12288}
    configuration_ok=all(engine.get(k)==v for k,v in expected.items()) and eager and batch_invariant
    try:
        llm = LLM(model=str(model), enable_lora=False, **engine, **({'enforce_eager': True, 'seed': 0} if eager else {}))
    except Exception as error:
        atomic_json(out/'engine-status.json',{'engine_ok':False,'configuration_ok':bool(configuration_ok),
            'failure_type':'scoring_engine_start','error_type':type(error).__name__,'error':str(error)})
        raise
    atomic_json(out/'engine-status.json',{'engine_ok':True,'configuration_ok':bool(configuration_ok),
        'engine':engine,'attention_backend':os.environ.get('VLLM_ATTENTION_BACKEND'),'machine':machine_fingerprint(deterministic)})
    reload_seconds=time.monotonic()-reload_clock
    generate_started=datetime.now(timezone.utc).isoformat();generate_clock=time.monotonic()
    outputs = llm.generate(prompts, SamplingParams(n=1, **decoding))
    generate_seconds=time.monotonic()-generate_clock
    if os.environ.get('V4_TELEMETRY')=='1':
        sys.path.insert(0,str(HERE.parent))
        from kit.v4_timing import receipt
        (out/'timing.json').write_text(json.dumps({'reload':receipt(reload_started,reload_seconds),
            'generate':receipt(generate_started,generate_seconds)},sort_keys=True)+'\n')
    answers, rows, token_rows = {}, [], []
    for item, prompt, output in zip(items, prompts, outputs):
        completion = output.outputs[0]
        answers[item["id"]] = completion.text
        rows.append({"bed": args.bed, "split": split_of(args), "id": item["id"], "response": completion.text,
                     "output_tokens": len(completion.token_ids), "finish_reason": completion.finish_reason})
        prompt_ids = getattr(output, "prompt_token_ids", None)
        token_rows.append({"id": item["id"], "ids": list(completion.token_ids),          # the generated ids, for prefix budgets (kit/tokens_io.py)
                           "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                           "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None})
    (out / "responses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tokens_io.write(out / tokens_io.FILE_NAME, token_rows)
    # the raw files bound to this scoring attempt by hash (package-4 review round 3, finding 4: a scoring with no sweep,
    # the qualification scorings, is bound by these)
    raw_hashes = {"responses_sha256": _file_sha256(out / "responses.jsonl"), "tokens_sha256": _file_sha256(out / tokens_io.FILE_NAME)}
    truncated = sum(r["finish_reason"] == "length" for r in rows)
    write_result(out, bed=args.bed, split=split_of(args), graded=grade(module, args, items, answers),
                 extra={**panel_labels(module, args), "mode": "generate", "model": str(model), "items_sha256": items_digest(items), **raw_hashes,
                        "limit": int(args.limit or 0),
                        "max_new_tokens": cap, "truncated_at_max_tokens": truncated,
                        "decoding": decoding,          # overrides the bed default written by write_result
                        "tokens_file": tokens_io.FILE_NAME, "token_encoding": tokens_io.ENCODING,
                        "prompts_sha256": hashlib.sha256("".join(r["prompt_sha256"] + "\n" for r in token_rows).encode("utf-8")).hexdigest(),
                        "generation": {"n": 1, "enable_lora": False, "enforce_eager": eager, "seed": 0 if eager else None,
                                       "chat_template": {"add_generation_prompt": True, "enable_thinking": False}},
                        # what a correct answer costs (kit/density.py): the whole set's output tokens
                        "output_tokens_total": sum(r["output_tokens"] for r in rows),
                        "output_tokens_mean": round(sum(r["output_tokens"] for r in rows) / max(1, len(rows)), 3),
                        "engine": {**engine, "vllm": vllm.__version__, "batch_invariant": batch_invariant,
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
                 extra={**panel_labels(module, args), "mode": "score", "items_sha256": items_digest(items), "limit": int(args.limit or 0),
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
            p.add_argument("--max-model-len", type=int, default=None,
                           help="the engine's context length, overriding the default (4,096, or 4,096 plus the cap when "
                                "--max-new-tokens is given). A budget audit passes the SAME value to a short-cap and a "
                                "long-cap scoring so that the cap is the only difference between them; recorded in `engine`")
            p.add_argument("--max-new-tokens", type=int, default=None,
                           help="answer cap instead of the bed's %s. A DEPARTURE from the bar of record, for diagnosis "
                                "(e.g. 8192, the trainer's own cap, to see whether answers cut at 2,048 were right); "
                                "recorded as `max_new_tokens` and `decoding.max_tokens`, and a result scored at another "
                                "cap is not comparable with this one" % MAX_NEW_TOKENS["gsm8k"])
        else:
            p.add_argument("--responses", required=True)
    args = parser.parse_args(argv)
    try:
        return {"generate": cmd_generate, "score": cmd_score}[args.action](args)
    except EvalBedError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    sys.exit(main())
