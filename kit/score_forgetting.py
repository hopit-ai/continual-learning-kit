#!/usr/bin/env python3
"""Score a saved model for forgetting on the three general panels, on any machine with vLLM.

    python score_forgetting.py generate --model /path/to/hf-model --out /work/forgetting/NAME
    python score_forgetting.py score    --responses responses.jsonl --out DIR      # no GPU: re-score saved answers
    python score_forgetting.py compare  --base BASE/forgetting.json --after RUN/forgetting.json --out DIR

What it measures. 300 questions fixed in advance: 100 grade-school maths (GSM8K), 100 general
knowledge multiple choice (MMLU), 100 instruction-following prompts (an IFEval-style subset). One
greedy answer each, thinking disabled, at most 2,048 new tokens: the same prompts, chat template,
decoding and scoring code as the nine-panel matrix in our own harness, of which these are the three
panels under the registered floor (a trained model may lose at most 3 per 100 on each).

**Compare only results scored on the same machine.** Greedy decoding under a batched engine is
repeatable on one machine and NOT across machines: scoring the same untrained model twice on one
GPU gave 297 of 300 identical answers and no changed verdict, while two different machines with the
same GPU model agreed on about 90 answers and flipped 10 to 13 verdicts, moving a panel by up to 3
points (receipt 204). The floor is 3 points, so a cross-machine comparison cannot resolve it.
`generate` therefore records a machine fingerprint, and `compare` answers NOT_COMPARABLE unless both
results carry the same one. Score the untrained model on every machine you score trained models on.

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
    return [json.loads(line) for line in text.splitlines() if line.strip()], digest


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


def machine_fingerprint() -> dict:
    """What has to be equal for two scorings to be comparable: the physical GPU and the software."""
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
                   "versions": versions, "python": platform.python_version()}
    fingerprint["id"] = hashlib.sha256(json.dumps({k: fingerprint[k] for k in ("gpus", "cuda_visible_devices", "versions")},
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
              "total": {"n": sum(p["n"] for p in panels.values()), "correct": sum(p["correct"] for p in panels.values())}, **extra}
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
    from transformers import AutoTokenizer                                  # noqa: PLC0415
    from vllm import LLM, SamplingParams                                    # noqa: PLC0415
    import vllm                                                             # noqa: PLC0415
    tokenizer = AutoTokenizer.from_pretrained(str(model))
    prompts = [render(tokenizer, m["prompt"]) for m in members]
    llm = LLM(model=str(model), enable_lora=False, **ENGINE)
    outputs = llm.generate(prompts, SamplingParams(n=1, **DECODING))
    responses, rows = {}, []
    for member, output in zip(members, outputs):
        completion = output.outputs[0]
        responses[(member["panel"], member["id"])] = completion.text
        rows.append({"panel": member["panel"], "id": member["id"], "response": completion.text,
                     "output_tokens": len(completion.token_ids), "finish_reason": completion.finish_reason})
    (out / "responses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    truncated = sum(r["finish_reason"] == "length" for r in rows)
    write_result(out, panels=grade(members, responses), panel_sha=digest,
                 extra={"mode": "generate", "model": model_identity(model), "engine": {**ENGINE, "vllm": vllm.__version__},
                        "machine": machine_fingerprint(), "truncated_at_max_tokens": truncated})
    return 0


def cmd_score(args) -> int:
    members, digest = load_panel()
    responses = {}
    for line in Path(args.responses).read_text().splitlines():
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a saved model for forgetting on three general panels.")
    sub = parser.add_subparsers(dest="action", required=True)
    g = sub.add_parser("generate"); g.add_argument("--model", required=True); g.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--responses", required=True); s.add_argument("--out", required=True)
    c = sub.add_parser("compare"); c.add_argument("--base", required=True); c.add_argument("--after", required=True); c.add_argument("--out", required=True)
    c.add_argument("--allow-different-machines", action="store_true", help="judge anyway; recorded in the result")
    args = parser.parse_args(argv)
    return {"generate": cmd_generate, "score": cmd_score, "compare": cmd_compare}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
