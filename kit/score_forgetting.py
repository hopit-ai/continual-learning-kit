#!/usr/bin/env python3
"""Score a saved model for forgetting on the three general panels, on any machine with vLLM.

    python score_forgetting.py generate --model /path/to/hf-model --out /work/forgetting/NAME
    python score_forgetting.py score    --responses responses.jsonl --out DIR      # no GPU: re-score saved answers
    python score_forgetting.py compare  --base BASE/forgetting.json --after RUN/forgetting.json --out DIR
    python score_forgetting.py agree    --a DIR1 --b DIR2 --out DIR      # two scorings of ONE model: how repeatable is this machine?
    python score_forgetting.py summarize --root DIR --base base-1 --out DIR   # every scored model against the untrained one

What it measures. 300 questions fixed in advance: 100 grade-school maths (GSM8K), 100 general
knowledge multiple choice (MMLU), 100 instruction-following prompts (an IFEval-style subset). One
greedy answer each, thinking disabled, at most 2,048 new tokens: the same prompts, chat template,
decoding and scoring code as the nine-panel matrix in our own harness, of which these are the three
panels under the registered floor (a trained model may lose at most 3 per 100 on each).

**Two rules for a comparison that means anything** (receipts 204 and 209).

1. *Deterministic decoding.* "Greedy" decoding under vLLM's batched, compiled engine is NOT repeatable,
   even on one machine in one container: three pairs of scorings of one untrained model agreed on 297,
   179 and 98 of 300 answers, with 0, 5 and 7 verdicts changed. With compilation and CUDA graphs off, a
   fixed seed and batch-invariant kernels, two pairs agreed on 300 of 300. That mode is the default here.
2. *Same machine.* Across machines, even with the same GPU model, 10 to 13 verdicts of 300 changed and
   a panel moved by up to 3 points, which is the size of the floor being checked.

`generate` records a fingerprint of the machine AND the decoding mode; `compare` and `summarize` answer
NOT_COMPARABLE unless both results carry the same one. Score the untrained model, in the same mode,
on every machine you score trained models on. The `repeatable` pilot of the K1a campaign proves it
each time by scoring the untrained model twice.

`generate` needs vLLM and one GPU. `score` and `compare` need only the Python standard library.
Nothing is overwritten: an existing output directory is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "kit-forgetting.v1"
PANEL_FILE = HERE / "panels" / "general-v1.jsonl"
FLOOR_PER_100 = -3
DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "max_tokens": 2048}
ENGINE = {"dtype": "bfloat16", "tensor_parallel_size": 1, "gpu_memory_utilization": 0.85, "max_model_len": 4096}

_spec = importlib.util.spec_from_file_location("kit_scorers", HERE / "scorers.py")
scorers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorers)


def load_panel(path: Path = PANEL_FILE) -> tuple[list, str]:
    text = path.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if manifest["sha256"] != digest:
        raise SystemExit("panel file does not match its manifest: %s" % path)
    return [json.loads(line) for line in text.split("\n") if line.strip()], digest


def fresh_dir(path: Path) -> Path:
    if path.exists():
        raise SystemExit("refusing to overwrite %s: choose a new --out" % path)
    path.mkdir(parents=True)
    return path


def model_identity(model: Path) -> dict:
    files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        f = model / name
        if f.is_file():
            files[name] = hashlib.sha256(f.read_bytes()).hexdigest()
    weights = sorted(p for p in model.glob("*.safetensors"))
    return {"path": str(model), "files": files, "weight_files": len(weights), "weight_bytes": sum(p.stat().st_size for p in weights)}


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


def render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def grade(members: list, responses: dict) -> dict:
    """responses: {(panel, id): text}. Every member must have exactly one answer."""
    missing = [(m["panel"], m["id"]) for m in members if (m["panel"], m["id"]) not in responses]
    if missing:
        raise SystemExit("%d members have no response, e.g. %s" % (len(missing), missing[:3]))
    panels: dict = {}
    for member in members:
        text = responses[(member["panel"], member["id"])]
        result = scorers.score(member["panel"], text, member)
        slot = panels.setdefault(member["panel"], {"n": 0, "correct": 0, "parsed": 0, "per_member": {}, "_chars": []})
        slot["n"] += 1
        slot["correct"] += int(result["correct"])
        slot["parsed"] += int(result["parsed"])
        slot["per_member"][member["id"]] = int(result["correct"])
        slot["_chars"].append(len(text))
    for slot in panels.values():
        chars = slot.pop("_chars")
        slot["median_output_chars"] = statistics.median(chars)
        slot["empty_outputs"] = sum(c == 0 for c in chars)
    return panels


def write_result(out: Path, *, panels: dict, panel_sha: str, extra: dict) -> dict:
    result = {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "panel_file_sha256": panel_sha, "decoding": DECODING, "panels": panels,
              "total": {"n": sum(p["n"] for p in panels.values()), "correct": sum(p["correct"] for p in panels.values())},
              "total_correct": sum(p["correct"] for p in panels.values()), **extra}
    (out / "forgetting.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    for name, p in panels.items():
        print("%-10s %3d of %3d correct | %3d parsed | median %d chars" % (name, p["correct"], p["n"], p["parsed"], p["median_output_chars"]))
    print("wrote", out / "forgetting.json")
    return result


def cmd_generate(args) -> int:
    members, digest = load_panel()
    model = Path(args.model).resolve()
    if not (model / "config.json").is_file():
        raise SystemExit("not a HuggingFace model directory: %s" % model)
    out = fresh_dir(Path(args.out))
    import os                                                               # noqa: PLC0415
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
    prompts = [render(tokenizer, m["prompt"]) for m in members]
    llm = LLM(model=str(model), enable_lora=False, **ENGINE, **({"enforce_eager": True, "seed": 0} if eager else {}))
    outputs = llm.generate(prompts, SamplingParams(n=1, **DECODING))
    responses, rows = {}, []
    for member, output in zip(members, outputs):
        completion = output.outputs[0]
        responses[(member["panel"], member["id"])] = completion.text
        rows.append({"panel": member["panel"], "id": member["id"], "response": completion.text,
                     "output_tokens": len(completion.token_ids), "finish_reason": completion.finish_reason})
    (out / "responses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    truncated = sum(r["finish_reason"] == "length" for r in rows)
    panels = grade(members, responses)
    for name, slot in panels.items():                      # what a correct answer costs (kit/density.py)
        tokens = [r["output_tokens"] for r in rows if r["panel"] == name]
        slot["output_tokens_total"] = sum(tokens)
        slot["output_tokens_mean"] = round(sum(tokens) / max(1, len(tokens)), 3)
    write_result(out, panels=panels, panel_sha=digest,
                 extra={"mode": "generate", "model": model_identity(model), "engine": {**ENGINE, "vllm": vllm.__version__, "batch_invariant": batch_invariant, "eager": eager, "deterministic": deterministic},
                        "machine": machine_fingerprint(deterministic), "truncated_at_max_tokens": truncated})
    return 0


def cmd_score(args) -> int:
    members, digest = load_panel()
    responses = {}
    for line in Path(args.responses).read_text().split("\n"):
        if line.strip():
            row = json.loads(line)
            key = (row["panel"], row.get("id") or row.get("member_id"))
            if key in responses:
                raise SystemExit("duplicate response for %s" % (key,))
            responses[key] = row["response"]
    wanted = {(m["panel"], m["id"]) for m in members}
    responses = {k: v for k, v in responses.items() if k in wanted}
    out = fresh_dir(Path(args.out))
    write_result(out, panels=grade(members, responses), panel_sha=digest,
                 extra={"mode": "score", "responses_file": str(Path(args.responses).resolve())})
    return 0


def cmd_compare(args) -> int:
    base, after = (json.loads(Path(p).read_text()) for p in (args.base, args.after))
    if base["panel_file_sha256"] != after["panel_file_sha256"]:
        raise SystemExit("the two results were scored on different panel files")
    out = fresh_dir(Path(args.out))
    machines = [(r.get("machine") or {}).get("id") for r in (base, after)]
    same_machine = machines[0] is not None and machines[0] == machines[1]
    rows, worst = [], 0
    for name in base["panels"]:
        b, a = base["panels"][name]["per_member"], after["panels"][name]["per_member"]
        if set(a) != set(b):
            raise SystemExit("panel %s: the two results cover different members" % name)
        change = sum(a.values()) - sum(b.values())
        worst = min(worst, change)
        rows.append({"panel": name, "n": len(b), "base": sum(b.values()), "after": sum(a.values()), "change": change,
                     "lost": sum(b[m] and not a[m] for m in b), "gained": sum(a[m] and not b[m] for m in b),
                     "ok": change >= FLOOR_PER_100})
    if not same_machine and not args.allow_different_machines:
        verdict = "NOT_COMPARABLE"
    else:
        verdict = "PASS" if all(r["ok"] for r in rows) else "FAIL"
    result = {"schema": SCHEMA, "mode": "compare", "floor_per_100": FLOOR_PER_100, "rows": rows, "worst_change": worst,
              "verdict": verdict, "same_machine": same_machine, "machine_ids": machines,
              "different_machines_allowed": bool(args.allow_different_machines)}
    (out / "comparison.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    lines = ["| panel | base | after | change | lost | gained | within floor of %d |" % FLOOR_PER_100, "|---|---|---|---|---|---|---|"]
    lines += ["| %s | %d | %d | %+d | %d | %d | %s |" % (r["panel"], r["base"], r["after"], r["change"], r["lost"], r["gained"], "yes" if r["ok"] else "NO") for r in rows]
    note = "" if same_machine else ("\n\nThe two results were NOT scored on the same machine (%s vs %s). Cross-machine noise is up to 3 points a panel, "
                                    "the size of the floor itself; score the untrained model on this machine and compare again." % tuple(machines))
    (out / "comparison.md").write_text("\n".join(lines) + "\n\nVerdict: %s%s\n" % (verdict, note))
    print("\n".join(lines)); print("Verdict:", verdict + note)
    return {"PASS": 0, "FAIL": 1, "NOT_COMPARABLE": 2}[verdict]


OUR_UNTRAINED_TOTAL = 251            # Qwen3-8B on our harness (91 / 72 / 88); two other machines gave 250 and 251 (receipt 204)


def _responses(directory: Path) -> dict:
    return {(r["panel"], r["id"]): r["response"] for r in (json.loads(line) for line in (directory / "responses.jsonl").read_text().split("\n") if line.strip())}


def cmd_agree(args) -> int:
    a_dir, b_dir = Path(args.a), Path(args.b)
    a, b = (json.loads((d / "forgetting.json").read_text()) for d in (a_dir, b_dir))
    ra, rb = _responses(a_dir), _responses(b_dir)
    if set(ra) != set(rb) or a["panel_file_sha256"] != b["panel_file_sha256"]:
        raise SystemExit("the two scorings do not cover the same questions")
    out = fresh_dir(Path(args.out))
    changed = sum(a["panels"][p]["per_member"][m] != b["panels"][p]["per_member"][m] for p in a["panels"] for m in a["panels"][p]["per_member"])
    result = {"schema": SCHEMA, "mode": "agree", "questions": len(ra), "identical_answers": sum(ra[k] == rb[k] for k in ra), "changed_verdicts": changed,
              "total_a": a["total_correct"], "total_b": b["total_correct"], "distance_from_our_untrained_total": abs(a["total_correct"] - OUR_UNTRAINED_TOTAL),
              "same_machine": (a.get("machine") or {}).get("id") is not None and (a.get("machine") or {}).get("id") == (b.get("machine") or {}).get("id"),
              "panels_a": {p: v["correct"] for p, v in a["panels"].items()}, "panels_b": {p: v["correct"] for p, v in b["panels"].items()}}
    result["same_machine_flag"] = int(result["same_machine"])
    (out / "agreement.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1))
    return 0


def _latest(root: Path) -> dict:
    """{'base-1': Path('.../base-1-a2'), ...}: the highest attempt of every scored directory under root."""
    found: dict = {}
    for directory in sorted(p for p in root.iterdir() if (p / "forgetting.json").is_file()):
        name, _, attempt = directory.name.rpartition("-a")
        key, number = (name, int(attempt)) if name and attempt.isdigit() else (directory.name, 0)
        if key not in found or number > found[key][0]:
            found[key] = (number, directory)
    return {key: directory for key, (_, directory) in found.items()}


def cmd_summarize(args) -> int:
    scored = _latest(Path(args.root))
    if args.base not in scored:
        raise SystemExit("no scoring named %s under %s (found %s)" % (args.base, args.root, sorted(scored)))
    base = json.loads((scored[args.base] / "forgetting.json").read_text())
    out = fresh_dir(Path(args.out))
    rows = []
    for name, directory in sorted(scored.items()):
        if name.startswith("base"):
            continue
        after = json.loads((directory / "forgetting.json").read_text())
        same = (base.get("machine") or {}).get("id") is not None and (base.get("machine") or {}).get("id") == (after.get("machine") or {}).get("id")
        changes = {p: after["panels"][p]["correct"] - base["panels"][p]["correct"] for p in base["panels"]}
        lost = {p: sum(base["panels"][p]["per_member"][m] and not after["panels"][p]["per_member"][m] for m in base["panels"][p]["per_member"]) for p in base["panels"]}
        verdict = "NOT_COMPARABLE" if not same else ("PASS" if min(changes.values()) >= FLOOR_PER_100 else "FAIL")
        rows.append({"model": name, "same_machine": same, "scores": {p: after["panels"][p]["correct"] for p in after["panels"]}, "changes": changes, "lost": lost,
                     "median_output_chars": {p: after["panels"][p]["median_output_chars"] for p in after["panels"]}, "verdict": verdict})
    report = {"schema": SCHEMA, "mode": "summarize", "floor_per_100": FLOOR_PER_100, "base": {"name": args.base, "scores": {p: v["correct"] for p, v in base["panels"].items()},
              "median_output_chars": {p: v["median_output_chars"] for p, v in base["panels"].items()}, "machine": base.get("machine")}, "models": rows}
    (out / "forgetting-report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    panels = list(base["panels"])
    lines = ["# Forgetting report", "", "Untrained model (%s): %s. Floor: a trained model may lose at most %d per 100 on each panel, judged only against an untrained model scored on the same machine."
             % (args.base, ", ".join("%s %d" % (p, base["panels"][p]["correct"]) for p in panels), -FLOOR_PER_100), "",
             "| model | " + " | ".join(panels) + " | verdict |", "|---|" + "---|" * (len(panels) + 1)]
    lines += ["| %s | %s | %s |" % (r["model"], " | ".join("%d (%+d)" % (r["scores"][p], r["changes"][p]) for p in panels), r["verdict"]) for r in rows]
    (out / "forgetting-report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a saved model for forgetting on three general panels.")
    sub = parser.add_subparsers(dest="action", required=True)
    g = sub.add_parser("generate"); g.add_argument("--model", required=True); g.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--responses", required=True); s.add_argument("--out", required=True)
    c = sub.add_parser("compare"); c.add_argument("--base", required=True); c.add_argument("--after", required=True); c.add_argument("--out", required=True)
    c.add_argument("--allow-different-machines", action="store_true", help="judge anyway; recorded in the result")
    a = sub.add_parser("agree"); a.add_argument("--a", required=True); a.add_argument("--b", required=True); a.add_argument("--out", required=True)
    z = sub.add_parser("summarize"); z.add_argument("--root", required=True); z.add_argument("--base", default="base-1"); z.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"generate": cmd_generate, "score": cmd_score, "compare": cmd_compare, "agree": cmd_agree, "summarize": cmd_summarize}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
