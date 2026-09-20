#!/usr/bin/env python3
"""Ask the ORIGINAL model to answer the repair prompts; its answers become the repair targets.

The repair trains a damaged model on how its undamaged ancestor answers everyday requests. Nothing here
comes from a larger model, a benchmark or a human: it is the model's own former behaviour.

    python make_targets.py --model /path/to/original --prompts repair-prompts.jsonl --out targets.jsonl
    python make_targets.py ... --backend hf        # CPU or a single small GPU, no vLLM (toy and pilot runs)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate repair targets from the original model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    rows = [json.loads(line) for line in Path(args.prompts).read_text().split("\n") if line.strip()]
    from transformers import AutoTokenizer                                  # noqa: PLC0415
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rendered = [render(tokenizer, row["prompt"]) for row in rows]
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams                                # noqa: PLC0415
        llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048)
        answers = [o.outputs[0].text for o in llm.generate(rendered, SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_tokens))]
    else:
        import torch                                                        # noqa: PLC0415
        from transformers import AutoModelForCausalLM                       # noqa: PLC0415
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        answers = []
        for text in rendered:
            ids = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(**ids, max_new_tokens=args.max_tokens, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
            answers.append(tokenizer.decode(generated[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({**row, "target": answer}, sort_keys=True, ensure_ascii=False) + "\n" for row, answer in zip(rows, answers)))
    empty = sum(not a.strip() for a in answers)
    stats = {"schema": "kit-repair-targets.v1", "targets": len(answers), "empty": empty, "median_chars": sorted(len(a) for a in answers)[len(answers) // 2],
             "model": args.model, "backend": args.backend}
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True))      # numbers a campaign bar can read
    print("wrote %d targets to %s (%d empty, median %d chars)" % (len(answers), out, empty, sorted(len(a) for a in answers)[len(answers) // 2]))
    return 0 if empty < len(answers) // 10 else 1


if __name__ == "__main__":
    sys.exit(main())
