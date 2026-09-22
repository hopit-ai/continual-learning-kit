#!/usr/bin/env python3
"""Fold a LoRA adapter into a full HuggingFace model, and REFUSE if it changed nothing.

    python fold_lora.py --merged $RUN/merged-step40 --out $RUN/hf-step40 --alpha 32 \\
                        --checkpoint-adapter $RUN/tool-grpo/global_step_40/actor/lora_adapter

WHY THIS FILE EXISTS. `python -m verl.model_merger merge` does NOT fold a LoRA adapter. It pops every
`lora_*` tensor out of the state dict into `<target_dir>/lora_adapter/` and saves the survivors, which
are the frozen base weights, as the model (`verl/model_merger/base_model_merger.py:301-306`, and
`save_lora_adapter` at `:235-290`). So `<target_dir>` is the UNTRAINED model with the adapter beside
it. Worse, the adapter it writes is a no-op as saved: `:268` hard-codes

    "lora_alpha": 0,  # lora_alpha is not set. An error should be raised to inform the user to set it manually.

and no error is raised. `alpha / r = 0` scales every delta to nothing.

If that were missed, `kit/score_forgetting.py` would score the untrained base model, every panel would
land within noise of the untrained scoring, and the package would report "LoRA forgets nothing" -- the
answer it exists to test -- with no LoRA weight ever scored. `config.json` and the safetensors are all
present, so a `merged model present` bar passes on it. That is why this is a step with its own checks
and its own exit code, and not a line inside a launcher.

WHAT IT CHECKS, and which check is allowed to be soft:

1. THE ADAPTER IS NOT A NO-OP (hard). For every adapted module this recomputes the delta itself, in
   float32, as `(alpha / r) * B @ A` -- the same arithmetic peft does, from the adapter file, with the
   alpha this run declared. EVERY adapted module must come out non-zero. Under the merger's
   `lora_alpha: 0` all of them are zero, so the trap fails here, before anything is written. This
   check is exact: it involves no rounding, because it never touches the saved bfloat16 weights.
2. THE FOLD REACHED THE MODEL (hard). At least one base weight must differ from the merger's copy
   after the fold. Zero means peft loaded the adapter and merged nothing.
3. NOTHING ELSE MOVED (hard). Every tensor that is NOT an adapted module -- embeddings, norms,
   `lm_head`, which peft's `all-linear` excludes -- must be bit-identical to the merger's copy. A
   change there means something other than the adapter moved.
4. HOW MANY adapted weights visibly changed (soft, RECORDED). The saved weights are bfloat16, whose
   relative resolution is about 0.008, so a delta smaller than that on some row is absorbed by the
   write and that tensor compares equal. `changed_tensors` vs `expected_changed` is therefore
   reported, and a shortfall is a FLAG in `fold.json`, not a refusal -- check 1 has already proved the
   adapter is live. Check 2 keeps the case "nothing at all reached the model" hard.

Nothing is overwritten and a failure writes NOTHING: `--out` is created only after every hard check
has passed, so a campaign bar cannot read a pass out of a no-op. Exit 0 on success, 3 on a failed
check, 2 on a bad request.

Needs torch, transformers, peft and safetensors -- the same install the trainer has. No GPU: the fold
runs on the CPU. It holds one copy of the model in memory (about 16 GB for an 8B in bfloat16) and
reads the merger's copy one tensor at a time from disk, so peak memory is roughly one model, and
`--out` costs another model on disk beside `--merged`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-lora-fold.v1"
ADAPTER_DIR = "lora_adapter"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
#: what peft prefixes every adapter key with; `save_lora_adapter` keeps it (base_model_merger.py:262)
PEFT_PREFIX = "base_model.model."
EXIT_OK, EXIT_REQUEST, EXIT_CHECK = 0, 2, 3


class FoldError(ValueError):
    """The request or the adapter is wrong; nothing is written."""


class FoldCheckFailed(ValueError):
    """The fold changed nothing, or changed something it must not. Nothing is written."""


# ------------------------------------------------------------------------------------ the adapter
def module_of(key: str) -> str | None:
    """`base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight` -> `model.layers.0.self_attn.q_proj`.

    None for a key that is not a LoRA factor. The `.default.` infix is handled because peft writes it
    and the merger strips only `.default.weight` (base_model_merger.py:261).
    """
    name = key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key
    for factor in (".lora_A.", ".lora_B."):
        if factor in name:
            return name.split(factor)[0]
    return None


def read_adapter(directory: Path) -> dict:
    """{'r', 'lora_alpha', 'target_modules', 'config'} from an adapter directory."""
    config_path = directory / ADAPTER_CONFIG
    if not config_path.is_file():
        raise FoldError("no %s in %s" % (ADAPTER_CONFIG, directory))
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise FoldError("%s is not JSON: %s" % (config_path, exc)) from exc
    rank = config.get("r")
    if not isinstance(rank, int) or rank <= 0:
        raise FoldError("%s has r=%r; a LoRA adapter needs a positive rank" % (config_path, rank))
    return {"r": rank, "lora_alpha": config.get("lora_alpha"),
            "target_modules": sorted(config.get("target_modules") or []), "config": config}


def factors(weights_path: Path) -> dict:
    """{module: {'A': tensor, 'B': tensor}} from an adapter's safetensors file, on the CPU."""
    from safetensors.torch import load_file                                    # noqa: PLC0415

    tensors = load_file(str(weights_path), device="cpu")
    found: dict = {}
    for key, tensor in tensors.items():
        module = module_of(key)
        if module is None:
            continue
        side = "A" if ".lora_A." in key else "B"
        if side in found.setdefault(module, {}):
            raise FoldError("%s has two lora_%s tensors for %s" % (weights_path, side, module))
        found[module][side] = tensor
    if not found:
        raise FoldError("%s holds no lora_A/lora_B tensors: this is not a LoRA adapter" % weights_path)
    incomplete = sorted(m for m, pair in found.items() if set(pair) != {"A", "B"})
    if incomplete:
        raise FoldError("%d adapted modules have only one factor, e.g. %s"
                        % (len(incomplete), ", ".join(incomplete[:3])))
    return found


def analytic_deltas(pairs: dict, *, rank: int, alpha: float) -> dict:
    """{module: max |(alpha/r) * B @ A|} in float32. The check that no rounding can hide."""
    import torch                                                              # noqa: PLC0415

    scaling = float(alpha) / float(rank)
    out = {}
    for module, pair in sorted(pairs.items()):
        with torch.no_grad():
            delta = (pair["B"].to(torch.float32) @ pair["A"].to(torch.float32)) * scaling
            out[module] = float(delta.abs().max())
        del delta
    return out


# ------------------------------------------------------------------------------- the merger's copy
def base_tensors(directory: Path):
    """A lazy {key: tensor} reader over a HuggingFace safetensors directory. Keys first, bytes on demand."""
    from safetensors import safe_open                                         # noqa: PLC0415

    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = {name: directory / shard for name, shard in weight_map.items()}
    else:
        single = directory / "model.safetensors"
        if not single.is_file():
            raise FoldError("no safetensors in %s: expected model.safetensors or an index" % directory)
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            files = {name: single for name in handle.keys()}

    class Reader:
        keys = frozenset(files)

        @staticmethod
        def get(key: str):
            with safe_open(str(files[key]), framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)

    return Reader


# ------------------------------------------------------------------------------------- the fold
def stage_adapter(source: Path, staging: Path, alpha: int) -> dict:
    """A copy of the adapter with `lora_alpha` set to what this run declared. The input is never touched."""
    adapter = read_adapter(source)
    if adapter["lora_alpha"] not in (0, None, alpha):
        raise FoldError(
            "the adapter at %s declares lora_alpha=%r and --alpha says %d. One of them is wrong, and "
            "guessing would silently rescale every weight in the model: pass the alpha the run was "
            "launched with, or fix the adapter."
            % (source, adapter["lora_alpha"], alpha))
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / ADAPTER_WEIGHTS, staging / ADAPTER_WEIGHTS)
    config = dict(adapter["config"])
    config["lora_alpha"] = alpha
    (staging / ADAPTER_CONFIG).write_text(json.dumps(config, indent=1, sort_keys=True))
    return adapter


def cross_check(adapter: dict, checkpoint_adapter: Path | None, alpha: int) -> dict | None:
    """The trainer's OWN adapter config (fsdp_workers.py:1127-1144) carries the real lora_alpha when it
    was written -- that save sits in a try/except that only logs, so it may be absent. When it is
    there, a disagreement is a refusal rather than a quiet preference for either side."""
    if checkpoint_adapter is None or not (checkpoint_adapter / ADAPTER_CONFIG).is_file():
        return None
    theirs = read_adapter(checkpoint_adapter)
    if theirs["r"] != adapter["r"]:
        raise FoldError("the trainer's adapter config says r=%d and the merger's says r=%d"
                        % (theirs["r"], adapter["r"]))
    if theirs["lora_alpha"] not in (0, None) and theirs["lora_alpha"] != alpha:
        raise FoldError(
            "the trainer wrote lora_alpha=%r at %s and --alpha says %d. The trainer's value is the one "
            "the run actually trained with." % (theirs["lora_alpha"], checkpoint_adapter, alpha))
    if theirs["target_modules"] and adapter["target_modules"] and \
            set(theirs["target_modules"]) != set(adapter["target_modules"]):
        raise FoldError("the two adapter configs name different target modules: %s vs %s"
                        % (theirs["target_modules"], adapter["target_modules"]))
    return {"path": str(checkpoint_adapter), "r": theirs["r"], "lora_alpha": theirs["lora_alpha"],
            "target_modules": theirs["target_modules"]}


def fold(merged: Path, out: Path, *, alpha: int, checkpoint_adapter: Path | None = None) -> dict:
    """Fold `merged/lora_adapter` into `merged`, check it, then write `out`. Returns the fold report."""
    if not (merged / "config.json").is_file():
        raise FoldError("not a HuggingFace model directory: %s" % merged)
    source = merged / ADAPTER_DIR
    if not source.is_dir():
        raise FoldError(
            "no %s in %s. `verl.model_merger merge` writes it beside the model whenever the checkpoint "
            "holds LoRA tensors; without it this directory is simply the model, and folding is not "
            "what it needs." % (ADAPTER_DIR, merged))
    if out.exists():
        raise FoldError("refusing to overwrite %s: an output is never replaced" % out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch                                                              # noqa: PLC0415
    from peft import PeftModel                                                # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer              # noqa: PLC0415

    staging = Path(tempfile.mkdtemp(prefix=".fold-lora-", dir=str(out.parent)))
    try:
        adapter = stage_adapter(source, staging, alpha)
        trainer_side = cross_check(adapter, checkpoint_adapter, alpha)
        pairs = factors(staging / ADAPTER_WEIGHTS)
        deltas = analytic_deltas(pairs, rank=adapter["r"], alpha=alpha)
        dead = sorted(module for module, size in deltas.items() if size == 0.0)
        if dead:
            raise FoldCheckFailed(
                "%d of %d adapted modules have an all-zero delta at alpha=%d (e.g. %s), so folding this "
                "adapter would produce the base model. This is what `verl.model_merger`'s hard-coded "
                "\"lora_alpha\": 0 does to every module at once (base_model_merger.py:268)."
                % (len(dead), len(deltas), alpha, ", ".join(dead[:3])))

        base = base_tensors(merged)
        # bfloat16, which is what the merger saved (base_model_merger.py:256): the LoRA arm's model has
        # to be scored in the same dtype as the full arm's or the comparison is between two precisions.
        model = AutoModelForCausalLM.from_pretrained(str(merged), dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, str(staging), is_trainable=False)
        model = model.merge_and_unload()

        adapted = set(deltas)
        scaling = float(alpha) / float(adapter["r"])
        survival, moved_elements, adapted_elements = [], 0, 0
        changed, unchanged, moved, not_compared = [], [], [], []
        max_abs, worst = 0.0, None
        with torch.no_grad():
            for key, tensor in model.state_dict().items():
                module = key[: -len(".weight")] if key.endswith(".weight") else None
                if key not in base.keys:
                    not_compared.append(key)
                    continue
                before = base.get(key).to(tensor.dtype)
                if module in adapted:
                    difference = float((tensor - before).abs().max())
                    (changed if difference > 0.0 else unchanged).append(module)
                    if difference > max_abs:
                        max_abs, worst = difference, module
                    # How much of the intended change survived the bfloat16 save: the realised change
                    # against the float32 one, as the relative error of the whole tensor, and the share
                    # of its elements that moved at all.
                    intended = (pairs[module]["B"].to(torch.float32) @ pairs[module]["A"].to(torch.float32)) * scaling
                    realised = tensor.to(torch.float32) - before.to(torch.float32)
                    norm = float(intended.norm())
                    if norm > 0.0:
                        survival.append(float((realised - intended).norm()) / norm)
                    moved_elements += int((realised != 0).sum()); adapted_elements += realised.numel()
                    del intended, realised
                elif not torch.equal(tensor, before):
                    moved.append(key)
                del before
        if moved:
            raise FoldCheckFailed(
                "%d tensors that are NOT adapted modules changed (e.g. %s). Only the LoRA modules may "
                "move; a change anywhere else means this is not the base model the adapter was trained "
                "on." % (len(moved), ", ".join(sorted(moved)[:3])))
        if not changed:
            raise FoldCheckFailed(
                "the folded model is identical to %s in every one of the %d adapted modules, so nothing "
                "reached it. Scoring it would score the UNTRAINED model." % (merged, len(adapted)))

        report = {
            "schema": SCHEMA,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "merged_dir": str(merged.resolve()), "out_dir": str(out.resolve()),
            "adapter_dir": str(source.resolve()),
            "r": adapter["r"], "alpha_declared": alpha,
            "alpha_in_merged_adapter": adapter["lora_alpha"],
            "scaling": round(float(alpha) / float(adapter["r"]), 6),
            "target_modules": adapter["target_modules"],
            "checkpoint_adapter": trainer_side,
            "expected_changed": len(adapted),
            "changed_tensors": len(changed),
            "unchanged_tensors": len(unchanged),
            "unchanged_modules": sorted(unchanged)[:20],
            "not_compared": sorted(not_compared),
            "differs_from_base": 1,
            "max_abs_delta": round(max_abs, 8),
            "max_abs_delta_module": worst,
            "min_analytic_delta": round(min(deltas.values()), 10),
            "max_analytic_delta": round(max(deltas.values()), 10),
            "dtype": "bfloat16",
            "rounding_error_median": round(statistics.median(survival), 4) if survival else None,
            "rounding_error_max": round(max(survival), 4) if survival else None,
            "share_of_adapted_elements_changed": round(moved_elements / adapted_elements, 4) if adapted_elements else None,
            "flags": [],
        }
        if survival and statistics.median(survival) > 0.5:
            report["flags"].append(
                "the bfloat16 save lost most of the adapter's change: in the median adapted tensor the saved "
                "change differs from the float32 one by %.0f%% of its size, and only %.1f%% of adapted weights "
                "moved at all. A score of this model measures a model much closer to the base than the one "
                "that was trained, so compare LoRA and full training with care."
                % (100 * statistics.median(survival), 100 * moved_elements / adapted_elements))
        if unchanged:
            report["flags"].append(
                "%d of %d adapted modules compare equal after the bfloat16 write, although every one "
                "of them has a non-zero float32 delta (smallest %.3g). bfloat16 has about 3 decimal "
                "digits, so a small delta is absorbed by the save. The adapter is live; this is "
                "recorded, not a failure." % (len(unchanged), len(adapted), min(deltas.values())))
        model.save_pretrained(str(out))
        AutoTokenizer.from_pretrained(str(merged)).save_pretrained(str(out))
        (out / "fold.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fold a LoRA adapter into a full HuggingFace model.")
    parser.add_argument("--merged", type=Path, required=True,
                        help="verl.model_merger's target_dir: the base model plus lora_adapter/")
    parser.add_argument("--out", type=Path, required=True, help="a new directory; never overwritten")
    parser.add_argument("--alpha", type=int, required=True,
                        help="the lora_alpha the run was LAUNCHED with; the merger writes 0")
    parser.add_argument("--checkpoint-adapter", type=Path, default=None,
                        help="the trainer's own <checkpoint>/actor/lora_adapter, cross-checked when present")
    args = parser.parse_args(argv)
    if args.alpha <= 0:
        print("--alpha must be positive: alpha 0 scales every LoRA delta to nothing, which is exactly "
              "the bug this step exists to catch", file=sys.stderr)
        return EXIT_REQUEST
    try:
        report = fold(args.merged, args.out, alpha=args.alpha, checkpoint_adapter=args.checkpoint_adapter)
    except FoldCheckFailed as exc:
        print("FOLD FAILED: %s" % exc, file=sys.stderr)
        return EXIT_CHECK
    except FoldError as exc:
        print("cannot fold: %s" % exc, file=sys.stderr)
        return EXIT_REQUEST
    print("folded %d of %d adapted modules into %s (r %d, alpha %d, largest weight change %.3g)"
          % (report["changed_tensors"], report["expected_changed"], args.out, report["r"],
             report["alpha_declared"], report["max_abs_delta"]))
    for flag in report["flags"]:
        print("flag: %s" % flag)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
