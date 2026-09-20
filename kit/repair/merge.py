#!/usr/bin/env python3
"""Fold a LoRA adapter into its base model and write a plain HuggingFace model a scorer can load. CPU is enough.

    python merge.py --base MODEL_DIR --adapter ADAPTER_DIR --out MERGED_DIR

The tokenizer, chat template and generation config are COPIED byte for byte from the base directory,
never re-saved through the library: re-serialising a tokenizer can change it silently, and a dropped
`chat_template.jinja` once cost this programme a day. The copy is verified by hash.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

CARRY = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
         "chat_template.jinja", "chat_template.json", "generation_config.json", "tokenizer.model")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    import torch                                                            # noqa: PLC0415
    from peft import PeftModel                                              # noqa: PLC0415
    from transformers import AutoModelForCausalLM                           # noqa: PLC0415
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    carried = []
    for name in CARRY:
        source = Path(args.base) / name
        if source.is_file():
            shutil.copyfile(source, out / name)             # overwrites the library's own generation_config.json with the base's bytes
            assert hashlib.sha256((out / name).read_bytes()).hexdigest() == hashlib.sha256(source.read_bytes()).hexdigest(), name
            carried.append(name)
    if "tokenizer.json" not in carried and "tokenizer.model" not in carried:
        raise SystemExit("the base directory has no tokenizer file to carry over: %s" % args.base)
    print("carried byte for byte:", ", ".join(carried))
    print("merged %s into %s -> %s" % (args.adapter, args.base, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
