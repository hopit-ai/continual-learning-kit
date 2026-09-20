#!/usr/bin/env python3
"""Train a small LoRA adapter on (prompt, target) pairs, saving it at chosen steps. One GPU, or CPU for a toy run.

    python train_lora.py --model DAMAGED --data targets.jsonl --out DIR --steps 300 --save-at 50,100,200,300

Used for the repair (targets are the original model's own answers) and, in the pilot only, to damage a
small model on purpose. Loss is taken on the answer tokens only. Settings follow "LoRA Without Regret":
every linear layer, alpha = 2 x rank, a learning rate about ten times a full fine-tune's, small batches.
Nothing is overwritten; a log line is written per optimizer step so a short run can be read afterwards.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def encode(tokenizer, prompt: str, target: str, max_len: int):
    head = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    tail_ids = tokenizer(target + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = (head_ids + tail_ids)[:max_len]
    labels = ([-100] * len(head_ids) + tail_ids)[:max_len]
    return ids, labels


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on prompt/target pairs with checkpoints at chosen steps.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--save-at", default="50,100,200,300")
    parser.add_argument("--batch", type=int, default=8, help="sequences per optimizer step (accumulated one at a time)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    save_at = sorted({int(x) for x in args.save_at.split(",") if x.strip()})
    if not save_at or save_at[-1] > args.steps:
        raise SystemExit("--save-at must name steps no later than --steps")

    import torch                                                            # noqa: PLC0415
    from peft import LoraConfig, get_peft_model                             # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer            # noqa: PLC0415
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if str(r.get("target", "")).strip()]
    if len(rows) < 2:
        raise SystemExit("need at least two examples with a non-empty target")
    examples = [encode(tokenizer, r["prompt"], r["target"], args.max_len) for r in rows]
    examples = [e for e in examples if any(label != -100 for label in e[1])]

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    if device == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "examples": len(examples), "device": device, "save_at": save_at}, indent=1, sort_keys=True))
    order, cursor, clock = list(range(len(examples))), 0, time.monotonic()
    random.shuffle(order)
    model.train()
    with (out / "train_log.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            total = 0.0
            for _ in range(args.batch):
                if cursor == len(order):
                    random.shuffle(order)
                    cursor = 0
                ids, labels = examples[order[cursor]]
                cursor += 1
                loss = model(input_ids=torch.tensor([ids], device=device), labels=torch.tensor([labels], device=device)).loss / args.batch
                loss.backward()
                total += float(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            log.write(json.dumps({"step": step, "loss": round(total, 5), "seconds": round(time.monotonic() - clock, 1)}) + "\n")
            log.flush()
            if step in save_at:
                model.save_pretrained(out / ("adapter-step%d" % step))
                print("step %d loss %.4f: saved adapter-step%d" % (step, total, step))
    return 0


if __name__ == "__main__":
    sys.exit(main())
