#!/usr/bin/env python3
"""Measure how much room a saved model has left to learn, and whether a sequence has used it up.

    python plasticity.py probe   --model /work/k5/stage4 --base /work/models/Qwen3-8B \\
                                 --reference /work/k5/plasticity/untrained/plasticity.json \\
                                 --out /work/k5/plasticity/stage4
    python plasticity.py compare --chain untrained/plasticity.json stage1/plasticity.json ... \\
                                 --metrics $WORK/runs --out /work/k5/plasticity/compare-a1

THE QUESTION (research target Q11, plan 4b). After a model has learned three jobs in a row, is it
slower to learn the fourth? "Plasticity loss" is the usual name. This file measures it and nothing
else: it trains nothing, changes no weight, and answers no question about how to fix it.

WHAT `probe` MEASURES, on one saved checkpoint, from 200 prompts and forward passes only:

  dormant share       the share of MLP hidden units whose |activation| never once reaches 0.001, on
                      any token of any of the 200 prompts. A dormant unit is carrying nothing, so it
                      has nothing left to give back.
  saturated share     the share of measured |activation| values above the 99.9th percentile of the
                      UNTRAINED model's own activations (`--reference` is that model's plasticity.json;
                      without it the percentile is taken from the model in front of us, which makes
                      the number 0.001 by construction, to the resolution of the sample -- that is how
                      the untrained reference is probed, and the result records which of the two
                      happened).
  effective rank      per layer, of the residual stream: the singular values of the collected
                      activations, normalised to sum to 1, their entropy, exponentiated. A layer
                      whose activations have collapsed onto a few directions has a low one.
  weight norm         the Frobenius norm of every floating-point weight, and -- with `--base` -- how
                      far the weights have moved from the untrained model, relative to its own norm.

WHAT `compare` DOES. It reads a chain of those files in stage order and applies the bar written down
in plan 4b before any of these numbers existed:

    plasticity loss is PRESENT only if a job takes at least 25 percent more steps to reach half its
    final gain in position 4 than in position 1, on at least 2 of 3 seeds, AND at least one internal
    signal (dormant share, saturated share, effective rank) moves the same way.

The two halves are reported separately and the tool NEVER answers PRESENT on one of them: a
slowdown with no internal movement is a training-curve fact, and an internal movement with no
slowdown is a measurement with no consequence. The learning half is read from each run's
`metrics.jsonl` -- the validation series the trainer logged -- and needs the runs to be named
`<job>-pos<POSITION>-seed<SEED>` (an `-aN` attempt suffix is allowed and the highest attempt wins),
because a job has to be found at both positions under one seed before anything can be subtracted.

ONE MACHINE, ONE DTYPE. An activation is not a count: it moves with the GPU, the kernel and the
dtype. Every probe records a machine-and-mode fingerprint and `compare` REFUSES a chain that mixes
two of them unless `--allow-different-machines`, which is recorded in the report as the departure it
is. This is the same rule the scoring tools enforce (receipts 204 and 209), for the same reason.

WHAT THIS IS NOT. It is a measurement, not a fix, and not a gate: `compare` exits 0 whatever the
verdict. Nothing here is evidence that resetting, re-initialising or regularising anything would
help; plan 4b says a fix is only designed if this bar is met.

`probe` needs torch, transformers and safetensors, one GPU for a real model (a tiny one runs on a
CPU), and no vLLM. `compare` needs only the Python standard library. Nothing is overwritten: an
existing --out is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "kit-plasticity.v1"
PANEL_FILE = HERE / "panels" / "general-v1.jsonl"

#: The fixed probe set: 200 of the 300 forgetting-panel prompts, taken round-robin from the panels in
#: sorted order so the share of each panel is fixed too (67 math, 67 knowledge, 66 ifeval). No random
#: seed is involved: the same 200 prompts come out of any machine, in the same order, for ever.
PROMPTS = 200
DORMANT_THRESHOLD = 1e-3            # |activation| below this, on every token of every prompt, is dormant
SATURATION_QUANTILE = 0.999         # the untrained model's own 99.9th percentile of |activation|
TOKENS_PER_PROMPT = 8               # residual-stream rows kept per prompt, evenly spaced (see select_positions)
VALUES_PER_PROMPT = 500             # |activation| values kept per layer per prompt, for the percentile
MAX_PROMPT_TOKENS = 2048            # a prompt longer than this is truncated, and the result says so
SEED = 0                            # the only randomness: which activation values are kept for the percentile

#: plan 4b's bar, written before any of these numbers existed. Neither half alone is enough.
POSITION_EARLY = 1
POSITION_LATE = 4
SLOWER_BY = 1.25                    # "at least 25 percent slower" in position 4 than in position 1
SEEDS_REQUIRED = 2                  # "on at least 2 of 3 seeds"
SEEDS_EXPECTED = 3                  # fewer than three seeds is INSUFFICIENT, not a pass and not a fail

#: The module whose INPUT is the MLP's hidden layer -- the activations after the non-linearity. Qwen
#: and Llama call it down_proj; the other three names cover the GPT-2, OPT and Mixtral shapes.
MLP_HIDDEN_SUFFIXES = ("mlp.down_proj", "mlp.c_proj", "mlp.fc2", "mlp.w2")
VAL_KEY = re.compile(r"^val-core/.+/acc/mean@\d+$")
POINT = re.compile(r"^(?P<job>[a-z0-9]+)-pos(?P<position>\d+)-seed(?P<seed>\d+)$")
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")

#: Which way each internal signal moves when a model has LESS room left to learn.
LESS_PLASTIC = {"dormant_share": "up", "saturated_share": "up", "effective_rank": "down"}


class PlasticityError(ValueError):
    """The checkpoint, the chain or the runs cannot support a measurement; nothing is written."""


# ------------------------------------------------------------------------------- the fixed prompts
def load_panel(path: Path = PANEL_FILE) -> tuple:
    """The 300 panel members and the sha256 of the file, checked against its manifest.

    Duplicated from score_forgetting.py on purpose, as eval_bed.py duplicates its decoding constants:
    this file must keep reading the frozen panel even if that one is re-specified."""
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest["sha256"] != digest:
        raise PlasticityError("panel file does not match its manifest: %s" % path)
    return [json.loads(line) for line in text.split("\n") if line.strip()], digest


def select_prompts(members: list, count: int = PROMPTS) -> list:
    """`count` members, taken one panel at a time in a fixed rotation over the panels in sorted order.

    Deterministic with no seed and no shuffle: sort each panel by id, then take the first of each
    panel, the second of each, and so on until `count` are held. Every machine draws the same 200."""
    if count > len(members):
        raise PlasticityError("asked for %d prompts but the panel holds %d" % (count, len(members)))
    by_panel = {}
    for member in members:
        by_panel.setdefault(member["panel"], []).append(member)
    for panel in by_panel:
        by_panel[panel].sort(key=lambda m: m["id"])
    order = sorted(by_panel)
    chosen, depth = [], 0
    while len(chosen) < count:
        took = False
        for panel in order:
            if depth < len(by_panel[panel]) and len(chosen) < count:
                chosen.append(by_panel[panel][depth])
                took = True
        if not took:
            break
        depth += 1
    return chosen


def selection_digest(chosen: list) -> str:
    """A fingerprint of WHICH prompts were measured, so two probes can be proved to share them."""
    return hashlib.sha256("\n".join("%s/%s" % (m["panel"], m["id"]) for m in chosen).encode()).hexdigest()


def select_positions(length: int, count: int = TOKENS_PER_PROMPT) -> list:
    """Evenly spaced token positions, from index 1 to the last.

    Position 0 is left out whenever there is more than one token: the first position is an attention
    sink whose residual norm dwarfs every other one, and a single row that large decides the
    effective rank of the whole layer by itself."""
    if length <= 0 or count <= 0:
        return []
    first = 1 if length > 1 else 0
    available = length - first
    if count == 1:
        return [length - 1]
    if available <= count:
        return list(range(first, length))
    return [first + round(i * (available - 1) / (count - 1)) for i in range(count)]


def sample_indices(total: int, count: int, *, layer: int, prompt: int) -> list:
    """`count` distinct flat indices into one prompt's activation tensor, fixed by SEED, layer and prompt."""
    if total <= count:
        return list(range(total))
    return random.Random("%d-%d-%d" % (SEED, layer, prompt)).sample(range(total), count)


# --------------------------------------------------------------------------------- what was probed
def fresh_dir(path: Path) -> Path:
    if path.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % path)
    path.mkdir(parents=True)
    return path


def model_identity(model: Path) -> dict:
    """Enough of the checkpoint to tell two of them apart without reading the weights twice."""
    files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        path = model / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    weights = sorted(model.glob("*.safetensors"))
    return {"path": str(model), "files": files, "weight_files": len(weights),
            "weight_bytes": sum(p.stat().st_size for p in weights)}


def machine_fingerprint(device: str, dtype: str) -> dict:
    """What has to be equal for two probes to be comparable: the GPU, the software, the device and the dtype.

    An activation is not a count. It moves with the kernel it was computed by, so the dtype and the
    device are part of the fingerprint here, where the scoring tools instead carry their decoding mode."""
    import os, platform, socket, subprocess                                  # noqa: E401,PLC0415
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        smi = []
    versions = {}
    for name in ("torch", "transformers"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:                                                    # noqa: BLE001
            versions[name] = None
    fingerprint = {"hostname": socket.gethostname(), "gpus": sorted(line.strip() for line in smi),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "versions": versions,
                   "python": platform.python_version(), "device": device, "dtype": dtype}
    fingerprint["id"] = hashlib.sha256(json.dumps(
        {k: fingerprint[k] for k in ("gpus", "cuda_visible_devices", "versions", "device", "dtype")},
        sort_keys=True).encode()).hexdigest()[:16]
    return fingerprint


# --------------------------------------------------------------------------------------- the weights
def _layer_of(key: str) -> str:
    found = re.search(r"layers\.(\d+)\.", key)
    return found.group(1) if found else "other"


def weight_stats(model: Path, base: Path | None) -> dict:
    """Frobenius norm of every floating-point weight, and how far it has moved from --base.

    Read a tensor at a time straight from the safetensors files, so a second whole model never has to
    fit anywhere: an 8B comparison costs one tensor of memory, not 16 GB."""
    import torch                                                             # noqa: PLC0415
    from safetensors import safe_open                                        # noqa: PLC0415

    def shards(directory: Path) -> dict:
        found = sorted(directory.glob("*.safetensors"))
        if not found:
            raise PlasticityError("no *.safetensors in %s: this tool reads weights from safetensors only" % directory)
        index = {}
        for path in found:
            with safe_open(str(path), framework="pt") as handle:
                index.update({key: path for key in handle.keys()})
        return index

    here, there = shards(model), (shards(base) if base is not None else {})
    total_square, delta_square, base_square = 0.0, 0.0, 0.0
    per_layer: dict = {}
    compared, skipped, missing, mismatched = 0, 0, 0, 0
    for key in sorted(here):
        with safe_open(str(here[key]), framework="pt") as handle:
            tensor = handle.get_tensor(key)
        if not tensor.is_floating_point():
            skipped += 1
            continue
        tensor = tensor.to(torch.float32)
        square = float(tensor.pow(2).sum())
        total_square += square
        if base is None:
            continue
        if key not in there:
            missing += 1
            continue
        with safe_open(str(there[key]), framework="pt") as handle:
            other = handle.get_tensor(key)
        if tuple(other.shape) != tuple(tensor.shape):
            mismatched += 1
            continue
        other = other.to(torch.float32)
        moved = float((tensor - other).pow(2).sum())
        reference = float(other.pow(2).sum())
        delta_square += moved
        base_square += reference
        compared += 1
        slot = per_layer.setdefault(_layer_of(key), [0.0, 0.0])
        slot[0] += moved
        slot[1] += reference
    result = {"frobenius_norm": round(math.sqrt(total_square), 6), "float_tensors": len(here) - skipped,
              "non_float_tensors_skipped": skipped, "tensors_compared": compared,
              "tensors_missing_from_base": missing, "tensors_with_a_different_shape": mismatched,
              "distance_from_base": None, "relative_distance_from_base": None,
              "relative_distance_per_layer": {}}
    if base is not None and compared:
        result["distance_from_base"] = round(math.sqrt(delta_square), 6)
        result["relative_distance_from_base"] = round(math.sqrt(delta_square / base_square), 8) if base_square else None
        result["relative_distance_per_layer"] = {
            layer: (round(math.sqrt(moved / reference), 8) if reference else None)
            for layer, (moved, reference) in sorted(per_layer.items(), key=lambda kv: (kv[0] == "other", kv[0]))}
    return result


# ---------------------------------------------------------------------------------- the activations
def effective_rank(rows) -> float:
    """exp(entropy of the singular values, normalised to sum to 1). A collapsed layer scores near 1."""
    import torch                                                             # noqa: PLC0415

    values = torch.linalg.svdvals(rows.to(torch.float32))
    total = float(values.sum())
    if total <= 0:
        return 0.0
    share = values / total
    share = share[share > 0]
    return float(torch.exp(-(share * share.log()).sum()))


def render_prompt(tokenizer, prompt: str) -> tuple:
    """(text, used_chat_template). The chat template when the tokenizer has one, the bare prompt otherwise."""
    try:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False), True
    except (ValueError, TypeError, AttributeError):
        return prompt, False


def mlp_hidden_modules(model) -> list:
    """[(name, module)] for every block whose INPUT is that block's MLP hidden activation, in depth order."""
    found = [(name, module) for name, module in model.named_modules()
             if any(name.endswith(suffix) for suffix in MLP_HIDDEN_SUFFIXES)]
    if not found:
        raise PlasticityError(
            "no MLP hidden layer found in this model: expected modules named %s. Dormant and saturated "
            "shares cannot be measured on an architecture this file does not know."
            % ", ".join("*." + s for s in MLP_HIDDEN_SUFFIXES))
    return found


def probe(model_dir: Path, *, base: Path | None = None, reference: dict | None = None,
          device: str = "cpu", dtype: str = "float32", count: int = PROMPTS,
          tokens_per_prompt: int = TOKENS_PER_PROMPT, values_per_prompt: int = VALUES_PER_PROMPT,
          max_prompt_tokens: int = MAX_PROMPT_TOKENS) -> dict:
    """One checkpoint -> the plasticity.json body. Forward passes only; no weight is written to."""
    import torch                                                             # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer             # noqa: PLC0415

    if not (model_dir / "config.json").is_file():
        raise PlasticityError("not a HuggingFace model directory: %s" % model_dir)
    members, panel_sha = load_panel()
    chosen = select_prompts(members, count)
    digest = selection_digest(chosen)
    if reference is not None:
        check_reference(reference, digest)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=getattr(torch, dtype))
    model.to(device)
    model.eval()
    blocks = mlp_hidden_modules(model)
    if reference is not None and len(reference["_thresholds"]) != len(blocks):
        raise PlasticityError("--reference has %d layers and this model has %d: its per-layer thresholds "
                              "are not thresholds for this one" % (len(reference["_thresholds"]), len(blocks)))
    width = [None] * len(blocks)
    peak = [None] * len(blocks)                       # per unit, the largest |activation| ever seen
    kept: list = [[] for _ in blocks]                 # the |activation| values kept for the percentile
    captured: dict = {}

    def capture(index):
        def hook(_module, inputs):
            captured[index] = inputs[0].detach()
        return hook

    handles = [module.register_forward_pre_hook(capture(i)) for i, (_name, module) in enumerate(blocks)]
    rows: list = []                                   # per residual layer, the kept rows
    truncated, tokens_seen, used_template = 0, 0, None
    try:
        for order, member in enumerate(chosen):
            text, with_template = render_prompt(tokenizer, member["prompt"])
            used_template = with_template if used_template is None else (used_template and with_template)
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
            if ids.shape[1] > max_prompt_tokens:
                ids = ids[:, :max_prompt_tokens]
                truncated += 1
            tokens_seen += int(ids.shape[1])
            captured.clear()
            with torch.no_grad():
                out = model(input_ids=ids.to(device), output_hidden_states=True, use_cache=False)
            for index in range(len(blocks)):
                activation = captured[index][0].to("cpu", torch.float32).abs()
                if width[index] is None:
                    width[index] = int(activation.shape[-1])
                    peak[index] = torch.zeros(width[index])
                peak[index] = torch.maximum(peak[index], activation.amax(dim=0))
                flat = activation.reshape(-1)
                picks = sample_indices(int(flat.numel()), values_per_prompt, layer=index, prompt=order)
                kept[index].append(flat[torch.tensor(picks, dtype=torch.long)])
            hidden = out.hidden_states[1:]             # [0] is the embedding, before any layer ran
            if not rows:
                rows = [[] for _ in hidden]
            positions = select_positions(int(ids.shape[1]), tokens_per_prompt)
            index_tensor = torch.tensor(positions, dtype=torch.long)
            for index, state in enumerate(hidden):
                rows[index].append(state[0].to("cpu", torch.float32).index_select(0, index_tensor))
            del out, hidden
    finally:
        for handle in handles:
            handle.remove()

    thresholds = (reference or {}).get("_thresholds")
    layers, dormant_total, units_total = [], 0, 0
    above_total, values_total, saturated_units = 0, 0, 0
    for index in range(len(blocks)):
        values = torch.cat(kept[index]).sort().values
        own = float(values[min(len(values) - 1, int(SATURATION_QUANTILE * len(values)))])
        threshold = float(thresholds[index]) if thresholds is not None else own
        dormant = int((peak[index] < DORMANT_THRESHOLD).sum())
        above = int((values > threshold).sum())
        units = width[index]
        residual = torch.cat(rows[index]) if index < len(rows) else None
        centred = residual - residual.mean(dim=0, keepdim=True) if residual is not None else None
        layers.append({
            "layer": index, "module": blocks[index][0], "mlp_hidden_size": units,
            "dormant_units": dormant, "dormant_share": round(dormant / units, 8),
            "saturation_threshold": round(threshold, 8), "own_saturation_threshold": round(own, 8),
            "saturated_values": above, "values_sampled": int(len(values)),
            "saturated_share": round(above / len(values), 8),
            "saturated_units": int((peak[index] > threshold).sum()),
            "saturated_units_share": round(float((peak[index] > threshold).sum()) / units, 8),
            "max_abs_activation": round(float(peak[index].max()), 8),
            "median_abs_activation": round(float(values[len(values) // 2]), 8),
            "effective_rank": round(effective_rank(residual), 6) if residual is not None else None,
            "effective_rank_centred": round(effective_rank(centred), 6) if residual is not None else None,
            "residual_rows": int(residual.shape[0]) if residual is not None else 0,
            "residual_width": int(residual.shape[1]) if residual is not None else None,
        })
        dormant_total += dormant
        units_total += units
        above_total += above
        values_total += len(values)
        saturated_units += layers[-1]["saturated_units"]

    ranks = [row["effective_rank"] for row in layers if row["effective_rank"] is not None]
    config = model.config.to_dict()
    result = {
        "schema": SCHEMA, "mode": "probe",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model_identity(model_dir), "base": model_identity(base) if base is not None else None,
        "panel_file_sha256": panel_sha,
        "prompt_selection": {"n": len(chosen), "sha256": digest,
                             "rule": "round-robin over the panels in sorted order, each sorted by id",
                             "per_panel": {panel: sum(m["panel"] == panel for m in chosen)
                                           for panel in sorted({m["panel"] for m in chosen})}},
        "settings": {"dormant_threshold": DORMANT_THRESHOLD, "saturation_quantile": SATURATION_QUANTILE,
                     "tokens_per_prompt": tokens_per_prompt, "values_per_prompt": values_per_prompt,
                     "max_prompt_tokens": max_prompt_tokens, "seed": SEED, "device": device, "dtype": dtype,
                     "chat_template": bool(used_template)},
        "architecture": {"model_type": config.get("model_type"), "layers": len(blocks),
                         "hidden_size": config.get("hidden_size"), "mlp_hidden_size": width[0],
                         "residual_layers": len(rows)},
        "saturation_reference": {"source": "reference" if thresholds is not None else "self",
                                 "path": (reference or {}).get("_path"),
                                 "model": (reference or {}).get("model")},
        "layers": layers,
        "dormant_share": round(dormant_total / units_total, 8),
        "saturated_share": round(above_total / values_total, 8),
        "saturated_units_share": round(saturated_units / units_total, 8),
        "effective_rank": round(statistics.fmean(ranks), 6) if ranks else None,
        "effective_rank_centred": round(statistics.fmean(
            [row["effective_rank_centred"] for row in layers if row["effective_rank_centred"] is not None]), 6) if ranks else None,
        "weights": weight_stats(model_dir, base),
        "prompts_measured": len(chosen), "prompts_truncated": truncated, "tokens_measured": tokens_seen,
        "machine": machine_fingerprint(device, dtype),
    }
    return result


def check_reference(reference: dict, digest: str) -> None:
    """A percentile from another model is only a threshold if it was taken on the same prompts."""
    if reference.get("schema") != SCHEMA:
        raise PlasticityError("--reference is not a %s file" % SCHEMA)
    theirs = (reference.get("prompt_selection") or {}).get("sha256")
    if theirs != digest:
        raise PlasticityError("--reference measured a different set of prompts (%s, not %s): its 99.9th "
                              "percentile is not a threshold for this one" % (theirs, digest))


def reference_thresholds(path: Path) -> dict:
    """The reference's own per-layer 99.9th percentile, carried under a private key for `probe`."""
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("saturation_reference", {}).get("source") != "self":
        raise PlasticityError(
            "%s was itself measured against another model's percentile, so it is not the untrained "
            "reference. Probe the untrained model with no --reference first." % path)
    reference["_thresholds"] = [row["own_saturation_threshold"] for row in reference["layers"]]
    reference["_path"] = str(path.resolve())
    return reference


def cmd_probe(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % out)
    try:
        reference = reference_thresholds(Path(args.reference)) if args.reference else None
        result = probe(Path(args.model).resolve(),
                       base=Path(args.base).resolve() if args.base else None, reference=reference,
                       device=args.device, dtype=args.dtype, count=args.prompts,
                       tokens_per_prompt=args.tokens_per_prompt, values_per_prompt=args.values_per_prompt,
                       max_prompt_tokens=args.max_prompt_tokens)
    except PlasticityError as exc:
        raise SystemExit(str(exc))
    fresh_dir(out)
    (out / "plasticity.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("%d prompts, %d layers, %d tokens" % (result["prompts_measured"], result["architecture"]["layers"],
                                                result["tokens_measured"]))
    print("dormant %.4f | saturated %.4f (threshold from the %s) | effective rank %s | weight norm %g%s"
          % (result["dormant_share"], result["saturated_share"], result["saturation_reference"]["source"],
             result["effective_rank"], result["weights"]["frobenius_norm"],
             "" if result["weights"]["relative_distance_from_base"] is None else
             " | moved %.6f of the untrained norm" % result["weights"]["relative_distance_from_base"]))
    print("wrote", out / "plasticity.json")
    return 0


# ------------------------------------------------------------------- the learning half of the bar
def _jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def validation_series(run: Path, key: str | None = None) -> tuple:
    """([(step, accuracy)], key): the validation curve the trainer logged, earliest step first."""
    metrics = run / "metrics.jsonl"
    if not metrics.is_file():
        raise PlasticityError("no metrics.jsonl in %s: the learning half of the bar is read from the "
                              "validation series the trainer logged" % run)
    records = _jsonl(metrics)
    if key is None:
        keys = sorted({k for record in records for k in (record.get("data") or {}) if VAL_KEY.match(k)})
        if not keys:
            raise PlasticityError("no validation key matching %s in %s: name one with --val-key"
                                  % (VAL_KEY.pattern, metrics))
        if len(keys) > 1:
            raise PlasticityError("%s logs %d validation keys (%s): name the one to use with --val-key"
                                  % (metrics, len(keys), ", ".join(keys)))
        key = keys[0]
    series = sorted({record["step"]: float((record.get("data") or {})[key])
                     for record in records if key in (record.get("data") or {})}.items())
    return series, key


def steps_to_half_gain(series: list) -> dict:
    """The first logged step at which the run reached half of its final gain, and how close a call it was.

    The gain is the last validation minus the first. A run that ended no higher than it started has no
    gain to halve, and the answer is null -- never zero, which would read as "learned instantly"."""
    if len(series) < 2:
        return {"steps": None, "steps_interpolated": None, "first": None, "final": None, "gain": None,
                "validations": len(series), "note": "fewer than two validations"}
    (first_step, first), (last_step, final) = series[0], series[-1]
    gain = final - first
    out = {"steps": None, "steps_interpolated": None, "first": round(first, 6), "final": round(final, 6),
           "gain": round(gain, 6), "validations": len(series), "first_step": first_step,
           "last_step": last_step, "note": None}
    if gain <= 0:
        out["note"] = "the run ended no higher than it started, so there is no gain to halve"
        return out
    target = first + gain / 2
    previous = series[0]
    for step, value in series[1:]:
        if value >= target:
            out["steps"] = step
            span = value - previous[1]
            out["steps_interpolated"] = round(
                previous[0] + (step - previous[0]) * ((target - previous[1]) / span), 4) if span > 0 else float(step)
            return out
        previous = (step, value)
    out["note"] = "no logged validation reached half the final gain"
    return out


def find_runs(entries: list) -> dict:
    """{(job, position, seed): directory} over the run directories named under --metrics.

    An entry is either a run directory (it holds metrics.jsonl) or a root holding some; the highest
    `-aN` attempt of a name wins, as everywhere else in the kit."""
    found: dict = {}
    for entry in entries:
        path = Path(entry)
        if not path.is_dir():
            raise PlasticityError("no such directory: %s" % path)
        children = [path] if (path / "metrics.jsonl").is_file() else sorted(p for p in path.iterdir() if p.is_dir())
        for child in children:
            match = ATTEMPT.match(child.name)
            stem, attempt = (match["stem"], int(match["attempt"])) if match else (child.name, 0)
            point = POINT.match(stem)
            if not point or not (child / "metrics.jsonl").is_file():
                continue
            key = (point["job"], int(point["position"]), int(point["seed"]))
            if key not in found or attempt > found[key][0]:
                found[key] = (attempt, child)
    if not found:
        raise PlasticityError(
            "no runs named <job>-pos<POSITION>-seed<SEED> (with an optional -aN attempt) holding a "
            "metrics.jsonl under %s. The learning half of the bar compares ONE job at position %d with "
            "the same job at position %d under the same seed, so the position and the seed have to be "
            "in the name." % (", ".join(str(e) for e in entries), POSITION_LATE, POSITION_EARLY))
    return {key: directory for key, (_attempt, directory) in found.items()}


def learning_half(runs: dict, key: str | None = None) -> dict:
    """plan 4b's first half: is a job at least 25 percent slower in position 4 than in position 1?"""
    measured, val_key = {}, key
    for (job, position, seed), directory in sorted(runs.items()):
        series, val_key = validation_series(directory, val_key)
        measured[(job, position, seed)] = {"run": directory.name, **steps_to_half_gain(series)}
    jobs: dict = {}
    for (job, position, seed) in measured:
        jobs.setdefault(job, set()).add(seed)
    rows, per_job = [], {}
    for job in sorted(jobs):
        slower, compared, unusable = 0, 0, 0
        for seed in sorted(jobs[job]):
            early = measured.get((job, POSITION_EARLY, seed))
            late = measured.get((job, POSITION_LATE, seed))
            if early is None or late is None:
                continue
            row = {"job": job, "seed": seed,
                   "early_run": early["run"], "late_run": late["run"],
                   "early_steps": early["steps"], "late_steps": late["steps"],
                   "early_gain": early["gain"], "late_gain": late["gain"],
                   "early_steps_interpolated": early["steps_interpolated"],
                   "late_steps_interpolated": late["steps_interpolated"],
                   "ratio": None, "slower": None, "note": early["note"] or late["note"]}
            if early["steps"] is not None and early["steps"] > 0 and late["steps"] is not None:
                row["ratio"] = round(late["steps"] / early["steps"], 4)
                row["slower"] = bool(row["ratio"] >= SLOWER_BY)
                slower += int(row["slower"])
                compared += 1
            else:
                unusable += 1
                row["note"] = row["note"] or "one side has no steps-to-half-gain"
            rows.append(row)
        per_job[job] = {"seeds_compared": compared, "seeds_slower": slower, "seeds_unusable": unusable,
                        "met": bool(compared >= SEEDS_EXPECTED and slower >= SEEDS_REQUIRED),
                        "sufficient": bool(compared >= SEEDS_EXPECTED),
                        # plan 4b wrote "2 of 3". It did not say what 2 of 5 means, and this tool does
                        # not decide that for it: it applies the number it was given and says so.
                        "note": ("the bar was written for %d seeds and %d were compared: %d of %d is "
                                 "being read as the same bar, which was never stated"
                                 % (SEEDS_EXPECTED, compared, SEEDS_REQUIRED, compared))
                                if compared > SEEDS_EXPECTED else None}
    sufficient = any(job["sufficient"] for job in per_job.values())
    return {"positions": {"early": POSITION_EARLY, "late": POSITION_LATE}, "slower_by": SLOWER_BY,
            "seeds_required": SEEDS_REQUIRED, "seeds_expected": SEEDS_EXPECTED, "val_key": val_key,
            "runs_read": len(measured), "per_seed": rows, "per_job": per_job,
            "sufficient": sufficient, "met": bool(sufficient and any(job["met"] for job in per_job.values())),
            "per_run": {"%s-pos%d-seed%d" % point: measured[point] for point in sorted(measured)}}


# ------------------------------------------------------------------- the internal half of the bar
def internal_half(chain: list) -> dict:
    """plan 4b's second half: did any internal signal move the way less plasticity would move it?

    Direction only. plan 4b set a size for the learning half (25 percent) and none for this one, so a
    move of any size counts here -- which is exactly why one half is never allowed to answer alone."""
    first, last = chain[0], chain[-1]
    signals = []
    for name, direction in sorted(LESS_PLASTIC.items()):
        before, after = first.get(name), last.get(name)
        if before is None or after is None:
            signals.append({"signal": name, "less_plastic_is": direction, "from": before, "to": after,
                            "change": None, "relative_change": None, "moved": None})
            continue
        change = after - before
        signals.append({"signal": name, "less_plastic_is": direction, "from": before, "to": after,
                        "change": round(change, 8),
                        "relative_change": round(change / before, 6) if before else None,
                        "moved": bool(change > 0 if direction == "up" else change < 0)})
    moving = [s for s in signals if s["moved"]]
    usable = [s for s in signals if s["moved"] is not None]
    return {"stages": len(chain), "first": first.get("_name"), "last": last.get("_name"),
            "signals": signals, "signals_moving": len(moving),
            "moving": [s["signal"] for s in moving],
            "sufficient": bool(len(usable) and len(chain) >= 2), "met": bool(moving),
            "size_of_move_was_never_pre_set": 1}


def read_chain(paths: list) -> list:
    """The probe files in the order given, which is stage order, earliest first."""
    chain = []
    for path in paths:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        if body.get("schema") != SCHEMA:
            raise PlasticityError("%s is not a %s file" % (path, SCHEMA))
        body["_name"] = Path(path).parent.name if Path(path).name == "plasticity.json" else Path(path).name
        body["_path"] = str(Path(path).resolve())
        chain.append(body)
    if len(chain) < 2:
        raise PlasticityError("--chain needs at least two probes: a first stage and a later one")
    selections = {(body.get("prompt_selection") or {}).get("sha256") for body in chain}
    if len(selections) > 1:
        raise PlasticityError("the probes in --chain measured different prompt sets (%s)"
                              % ", ".join(str(s) for s in sorted(selections, key=str)))
    shapes = {(body["architecture"]["layers"], body["architecture"]["mlp_hidden_size"]) for body in chain}
    if len(shapes) > 1:
        raise PlasticityError("the probes in --chain are of different architectures (%s): a dormant "
                              "share over one width is not a dormant share over another"
                              % ", ".join(str(s) for s in sorted(shapes)))
    return chain


def build(chain_paths: list, metrics: list | None, *, val_key: str | None = None,
          allow_different_machines: bool = False) -> dict:
    chain = read_chain(chain_paths)
    machines = sorted({(body.get("machine") or {}).get("id") for body in chain}, key=lambda m: (m is None, m))
    comparable = len(machines) == 1 and machines[0] is not None
    if not comparable and not allow_different_machines:
        raise PlasticityError(
            "the probes in --chain carry %d different machine-and-mode fingerprints (%s). An activation "
            "moves with the GPU, the kernel and the dtype, so a dormant share measured on one machine "
            "cannot be subtracted from one measured on another. Re-probe on one machine, or pass "
            "--allow-different-machines and say so in the readout." % (len(machines), ", ".join(str(m) for m in machines)))
    internal = internal_half(chain)
    learning = learning_half(find_runs(metrics), val_key) if metrics else {
        "met": False, "sufficient": False, "runs_read": 0,
        "note": "no --metrics given, so the learning half of the bar was not measured"}
    if not (learning["sufficient"] and internal["sufficient"]):
        verdict = "INCOMPLETE"
    elif learning["met"] and internal["met"]:
        verdict = "PRESENT"
    else:
        verdict = "NOT_PRESENT"
    return {"schema": SCHEMA, "mode": "compare",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bar": ("plasticity loss is PRESENT only if a job takes at least %g times as many steps to "
                    "reach half its final gain in position %d as in position %d, on at least %d of %d "
                    "seeds, AND at least one internal signal moves the same way. Written in plan 4b "
                    "before any of these numbers existed." % (SLOWER_BY, POSITION_LATE, POSITION_EARLY,
                                                              SEEDS_REQUIRED, SEEDS_EXPECTED)),
            "learning_half": learning, "internal_half": internal,
            "learning_half_met": int(bool(learning["met"])), "internal_half_met": int(bool(internal["met"])),
            "verdict": verdict, "plasticity_loss_present": int(verdict == "PRESENT"),
            "chain": [{"name": body["_name"], "path": body["_path"],
                       "dormant_share": body.get("dormant_share"), "saturated_share": body.get("saturated_share"),
                       "effective_rank": body.get("effective_rank"),
                       "relative_distance_from_base": (body.get("weights") or {}).get("relative_distance_from_base"),
                       "saturation_reference": (body.get("saturation_reference") or {}).get("source")}
                      for body in chain],
            "machine_ids": machines, "comparable": int(comparable),
            "different_machines_allowed": int(bool(allow_different_machines))}


def render(report: dict) -> str:
    learning, internal = report["learning_half"], report["internal_half"]
    lines = ["# Plasticity: does the model lose its room to learn across a sequence?", "",
             "The bar, from plan 4b: " + report["bar"], "",
             "**Each half is reported on its own, and one half never answers.** A slowdown with no "
             "internal movement is a fact about a training curve; an internal movement with no "
             "slowdown is a measurement with no consequence.", ""]
    if not report["comparable"]:
        lines += ["**The probes do not share one machine-and-mode fingerprint (%s), so these differences "
                  "are not comparable.**" % ", ".join(str(m) for m in report["machine_ids"]), ""]
    lines += ["## The chain", "",
              "| stage | dormant share | saturated share | effective rank | moved from untrained |",
              "|---|---|---|---|---|"]
    for row in report["chain"]:
        lines.append("| %s | %s | %s | %s | %s |" % (row["name"], row["dormant_share"], row["saturated_share"],
                                                     row["effective_rank"],
                                                     "-" if row["relative_distance_from_base"] is None
                                                     else "%.6f" % row["relative_distance_from_base"]))
    lines += ["", "## Half one: is the job slower in position %d?" % report["learning_half"].get(
        "positions", {}).get("late", POSITION_LATE), ""]
    if not learning.get("per_seed"):
        lines += [learning.get("note", "No runs were read, so this half was not measured."), ""]
    else:
        lines += ["| job | seed | steps to half gain, position %d | position %d | ratio | at least %g? |"
                  % (POSITION_EARLY, POSITION_LATE, SLOWER_BY), "|---|---|---|---|---|---|"]
        for row in learning["per_seed"]:
            lines.append("| %s | %d | %s | %s | %s | %s |" % (
                row["job"], row["seed"], row["early_steps"], row["late_steps"],
                "-" if row["ratio"] is None else "%.2f" % row["ratio"],
                "-" if row["slower"] is None else ("yes" if row["slower"] else "no")))
        for job, summary in sorted(learning["per_job"].items()):
            lines.append("| **%s** | | | | | **%d of %d seeds slower%s** |"
                         % (job, summary["seeds_slower"], summary["seeds_compared"],
                            "" if summary["sufficient"] else ", fewer than %d seeds compared" % SEEDS_EXPECTED))
        lines.append("")
    lines += ["Half one: %s." % ("MET" if learning["met"] else
                                 ("NOT MEASURED" if not learning["sufficient"] else "not met")), "",
              "## Half two: did an internal signal move the same way?", "",
              "| signal | less plasticity is | first stage | last stage | change | moved that way? |",
              "|---|---|---|---|---|---|"]
    for signal in internal["signals"]:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            signal["signal"], signal["less_plastic_is"], signal["from"], signal["to"],
            "-" if signal["change"] is None else "%+g" % signal["change"],
            "-" if signal["moved"] is None else ("yes" if signal["moved"] else "no")))
    lines += ["", "Half two: %s. plan 4b set no minimum size for this half, so any movement in the right "
              "direction counts here -- which is why it can never answer on its own."
              % ("MET" if internal["met"] else "not met"), "",
              "## Verdict", "", "**%s.**" % report["verdict"],
              "", "This is a measurement, not a fix: nothing here says that resetting, re-initialising "
              "or regularising anything would help. plan 4b designs a fix only if this bar is met.", ""]
    return "\n".join(lines)


def cmd_compare(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % out)
    try:
        report = build(args.chain, args.metrics, val_key=args.val_key,
                       allow_different_machines=args.allow_different_machines)
    except PlasticityError as exc:
        raise SystemExit(str(exc))
    fresh_dir(out)
    (out / "plasticity-compare.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    text = render(report)
    (out / "plasticity-compare.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure a checkpoint's room to learn, and judge a chain "
                                                 "of them against plan 4b's bar.")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("probe", help="one checkpoint -> plasticity.json")
    p.add_argument("--model", required=True, help="the HuggingFace checkpoint to probe")
    p.add_argument("--out", required=True, help="a new directory; never overwritten")
    p.add_argument("--base", help="the untrained model, for the weight distance")
    p.add_argument("--reference", help="the untrained model's plasticity.json, for the saturation threshold")
    p.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    p.add_argument("--dtype", default="float32", help="a torch dtype (default: float32; use bfloat16 on a GPU)")
    p.add_argument("--prompts", type=int, default=PROMPTS, help="how many panel prompts (default: %d)" % PROMPTS)
    p.add_argument("--tokens-per-prompt", type=int, default=TOKENS_PER_PROMPT)
    p.add_argument("--values-per-prompt", type=int, default=VALUES_PER_PROMPT)
    p.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    c = sub.add_parser("compare", help="a chain of probes plus the runs' metrics.jsonl -> the plan 4b bar")
    c.add_argument("--chain", nargs="+", required=True, help="plasticity.json files in stage order, earliest first")
    c.add_argument("--metrics", nargs="*", help="run directories named <job>-pos<N>-seed<S>, or roots holding them")
    c.add_argument("--val-key", help="the validation metric to read (default: the one val-core key in the file)")
    c.add_argument("--out", required=True, help="a new directory; never overwritten")
    c.add_argument("--allow-different-machines", action="store_true",
                   help="judge anyway; recorded in the report as a departure")
    args = parser.parse_args(argv)
    return {"probe": cmd_probe, "compare": cmd_compare}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
