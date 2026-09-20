#!/usr/bin/env python3
"""How far has a model moved from another, in weight space? Streams safetensors shards; CPU, low memory.

    python drift.py --reference ORIGINAL_DIR --model OTHER_DIR --out drift.json

Reports the relative distance ||W - W_ref|| / ||W_ref|| overall and by group (attention, MLP, embeddings,
norms). The recovery test uses it to tell "ability came back because the model was pulled back to where
it started" (distance to the original shrinks a lot) from "ability came back after a nudge" (it barely moves).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def group_of(name: str) -> str:
    if "embed" in name or "lm_head" in name:
        return "embeddings"
    if "norm" in name:
        return "norms"
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "attn")):
        return "attention"
    if any(k in name for k in ("gate_proj", "up_proj", "down_proj", "mlp")):
        return "mlp"
    return "other"


def tensors(directory: Path):
    from safetensors import safe_open                                       # noqa: PLC0415
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Relative weight distance between two HuggingFace model directories.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    import torch                                                            # noqa: PLC0415
    reference = {}
    for name, tensor in tensors(Path(args.reference)):
        reference[name] = tensor.to(torch.float32)
    sums: dict = {}
    seen = 0
    for name, tensor in tensors(Path(args.model)):
        if name not in reference:
            continue
        ref = reference.pop(name)
        if ref.shape != tensor.shape:
            raise SystemExit("shape differs for %s: not the same architecture" % name)
        delta = float((tensor.to(torch.float32) - ref).pow(2).sum())
        size = float(ref.pow(2).sum())
        for key in ("all", group_of(name)):
            slot = sums.setdefault(key, [0.0, 0.0])
            slot[0] += delta
            slot[1] += size
        seen += 1
    if not seen:
        raise SystemExit("the two directories share no tensor names")
    result = {"schema": "kit-drift.v1", "reference": args.reference, "model": args.model, "tensors_compared": seen, "tensors_only_in_reference": len(reference),
              "relative_distance": {key: (delta / size) ** 0.5 if size else None for key, (delta, size) in sorted(sums.items())}}
    result["relative_distance_all"] = result["relative_distance"]["all"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result["relative_distance"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
