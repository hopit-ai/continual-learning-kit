"""Independent PD label-CE forks, exact-label likelihoods and SwiGLU hooks.

All model/tokenizer loads are local-only. Nothing in this module invokes a
teacher, replay, EMA, a chain trainer, or a network service.
"""
from __future__ import annotations

import argparse
import copy
import functools
from contextlib import nullcontext
import json
import math
import random
import sys
import os
import time
import shutil
import subprocess
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kit.pd_contract import (SEEDS, STARTS, UPDATES, TRAINING_SPEC, digest, require, seal,
                             sha256_file, validate_checkpoint, verify_live_export,
                             verify_seal)
from kit.pd_prepare import taxonomy, write_frozen

BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SMOKE_STARTUP_SECONDS = 60


def code_identity():
    root = Path(__file__).parent
    names = ("pd_contract.py", "pd_prepare.py", "pd_probe.py", "pd_read.py", "pd_smoke_modal.py")
    return digest({name: sha256_file(root/name) for name in names} | {"taxonomy": digest(taxonomy())})


def tokenizer_identity(directory):
    root = Path(directory)
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
             "vocab.json", "merges.txt", "chat_template.jinja")
    files = {name: sha256_file(root/name) for name in names if (root/name).is_file()}
    require("tokenizer.json" in files and "tokenizer_config.json" in files, "pinned tokenizer artifacts missing")
    return digest(files)


def gpu_identity():
    import torch
    import transformers
    require(torch.cuda.is_available(), "live PD requires CUDA")
    devices = []
    for index in range(torch.cuda.device_count()):
        value = getattr(torch.cuda.get_device_properties(index), "uuid", None)
        require(value is not None, "physical GPU UUID unavailable")
        devices.append(str(value))
    return dict(uuid=devices[0], devices=devices,
                environment=digest(dict(torch=torch.__version__, cuda=torch.version.cuda,
                                        transformers=transformers.__version__, precision="bf16/fp32-masters",
                                        attention="eager", deterministic=True)))


def configure_determinism():
    import os
    import torch
    require(not torch.cuda.is_initialized() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            "deterministic workspace must be set before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def export_state(state):
    """BF16 export, retaining nonfloating buffers; FP32 masters stay untouched."""
    import torch
    return {name: value.to(torch.bfloat16) if value.is_floating_point() else value
            for name, value in state.items()}


def export_probe(model, destination, parent, update, *, state=None):
    require(update in UPDATES[1:], "unregistered export update")
    root = Path(destination)
    require(not root.exists(), "probe export already exists; never resume or overwrite")
    import torch
    state = export_state(model.state_dict()) if state is None else state
    require(all(not value.is_floating_point() or value.dtype == torch.bfloat16 for value in state.values()),
            "export state must contain BF16 floating tensors")
    model.save_pretrained(root, state_dict=state, safe_serialization=True, max_shard_size="2GB")
    tensors = {p.name: sha256_file(p) for p in root.iterdir() if p.suffix in (".safetensors", ".bin")}
    require(tensors, "no exported tensors")
    receipt = seal(dict(schema="pd-probe-export-v2", tensor_files=tensors, weights_sha256=digest(tensors),
                        parent_sha256=parent["identity_sha256"], step=update, recipe="label-CE",
                        export_dtype="bfloat16", master_dtype="float32"))
    write_frozen(root/"pd-export.json", receipt)
    checkpoint = seal(dict(checkpoint_id=f"pd-{parent['seed']}-{parent['start']}-{update}",
                           role="probe", start=parent["start"], seed=parent["seed"], step=update,
                           recipe="label-CE", parent_id=parent["checkpoint_id"],
                           base_revision=parent["base_revision"], tokenizer_sha256=parent["tokenizer_sha256"],
                           send_binding_sha256=parent["send_binding_sha256"],
                           export_receipt_sha256=sha256_file(root/"pd-export.json"),
                           tensor_files=tensors, weights_sha256=digest(tensors), export_dtype="bfloat16"))
    validate_checkpoint(checkpoint)
    return checkpoint


def prompt_text(question):
    definitions = "\n".join(f"{label}: {definition}" for label, definition in sorted(taxonomy().items()))
    return ("Classify the expected answer type of the question. Do not answer the question. "
            "Return one exact fine-grained label from these definitions:\n" + definitions +
            "\n\nQuestion:\n" + question)


def encode_row(tokenizer, row):
    definition_prompt = tokenizer.apply_chat_template([dict(role="user", content=prompt_text(""))],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    definition_tokens = len(tokenizer(definition_prompt, add_special_tokens=False,
                                      return_offsets_mapping=True)["input_ids"])
    require(definition_tokens <= 1024, "PENDING: definition prompt alone overflows cap")
    rendered = tokenizer.apply_chat_template([dict(role="user", content=prompt_text(row["text"]))],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    content_start = rendered.index("\n\nQuestion:\n") + len("\n\nQuestion:\n")
    content_end = content_start + len(row["text"])
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    prefix = list(enc["input_ids"])
    positions = [i for i, (a, b) in enumerate(enc["offset_mapping"])
                 if b > a and a < content_end and b > content_start]
    require(positions, "missing question-content token mask")
    suffixes = {label: list(tokenizer.encode(label, add_special_tokens=False)) for label in sorted(taxonomy())}
    eos = tokenizer.eos_token_id
    require(isinstance(eos, int) and all(tokens and eos not in tokens for tokens in suffixes.values()),
            "invalid suffix or EOS boundary")
    target = suffixes[row["label"]] + [eos]
    suffixes = {label: tokens+[eos] for label, tokens in suffixes.items()}
    return dict(id=row["id"], prefix_ids=prefix, suffixes=suffixes, eos_token_id=eos,
                definition_prompt_tokens=definition_tokens,
                suffix_sha256=digest(suffixes), input_ids=prefix+target,
                labels=[-100]*len(prefix)+target, content_positions=positions,
                candidate_lengths=[len(prefix)+len(s) for s in suffixes.values()],
                rendered_prompt_sha256=digest(rendered))


def seed_batches(training_ids, seed):
    require(seed in SEEDS, "only seeds 101-103 are registered")
    require(len(training_ids) == 1280 and len(set(training_ids)) == 1280, "one pass requires 1,280 distinct rows")
    order = list(training_ids)
    random.Random(seed).shuffle(order)
    return [order[i:i+32] for i in range(0, 1280, 32)]


def exposure_receipt(batches, tokenized):
    examples = []
    for batch in batches:
        for row_id in batch:
            row = tokenized[row_id]
            examples.append(dict(id=row_id, input_ids=row["input_ids"], labels=row["labels"]))
    return seal(dict(schema="pd-exposure-v1", batches=batches, examples_sha256=digest(examples),
                     examples=len(examples), input_tokens=sum(len(r["input_ids"]) for r in examples),
                     response_tokens=sum(sum(x != -100 for x in r["labels"]) for r in examples)))


def response_ce(logits, labels):
    """Equal mean EOS-inclusive response-token CE for each example."""
    import torch
    losses = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                                               labels[:, 1:], reduction="none", ignore_index=-100)
    mask = labels[:, 1:] != -100
    require(bool(mask.any(dim=1).all()), "empty response mask")
    return ((losses*mask).sum(1)/mask.sum(1)).mean()


def _device(model):
    return model.get_input_embeddings().weight.device


def _compute_context(model):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if _device(model).type == "cuda" else nullcontext()


def label_log_probs(model, prefix_ids, suffixes):
    """Summed conditional log-probability of frozen exact label-plus-EOS tokens."""
    import torch
    require(prefix_ids and suffixes, "empty prefix or label set")
    output = {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _compute_context(model):
            # One prefix prefill, then independent cache forks. This implements
            # the same conditional probabilities without repeating the 50-label
            # definition prompt for every candidate. Cache branches never mix.
            cache = None
            first_logp = None
            if hasattr(model, "config"):
                prefix = torch.tensor([prefix_ids], device=_device(model), dtype=torch.long)
                prefetched = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
                cache = prefetched.past_key_values
                require(cache is not None, "deterministic scorer requires an actual prefix cache")
                first_logp = prefetched.logits[0, -1].float().log_softmax(-1)
                del prefetched
            for label, suffix in sorted(suffixes.items()):
                require(suffix, "empty label suffix")
                if cache is not None:
                    value = first_logp[suffix[0]]
                    if len(suffix) > 1:
                        ids = torch.tensor([suffix[:-1]], device=_device(model), dtype=torch.long)
                        attention = torch.ones((1, len(prefix_ids)+len(suffix)-1), device=ids.device, dtype=torch.long)
                        branch = copy.deepcopy(cache)
                        logits = model(input_ids=ids, attention_mask=attention,
                                       past_key_values=branch, use_cache=True).logits[0].float()
                        targets = torch.tensor(suffix[1:], device=logits.device)
                        value = value+logits.log_softmax(-1).gather(1, targets[:, None]).sum()
                        del branch, logits
                    require(math.isfinite(value.item()), "nonfinite label likelihood")
                    output[label] = value.item()
                    continue
                ids = torch.tensor([prefix_ids+suffix], device=_device(model), dtype=torch.long)
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
                selected = logits[0, len(prefix_ids)-1:len(prefix_ids)+len(suffix)-1].float()
                targets = torch.tensor(suffix, device=selected.device)
                value = selected.log_softmax(-1).gather(1, targets[:, None]).sum().item()
                require(math.isfinite(value), "nonfinite label likelihood")
                output[label] = value
    finally:
        model.train(was_training)
    return output


def predict_label(scores):
    require(scores and all(math.isfinite(v) for v in scores.values()), "nonfinite or empty label scores")
    return min(scores, key=lambda label: (-scores[label], label))


def score_rows(model, rows, tokenized, *, generated=False, tokenizer=None):
    predictions, generated_predictions, labels, generated_labels = {}, {}, {}, {}
    score_vectors, mean_predictions, covariates = {}, {}, {}
    for row in rows:
        encoded = tokenized[row["id"]]
        values = label_log_probs(model, encoded["prefix_ids"], encoded["suffixes"])
        prediction = predict_label(values)
        score_vectors[row["id"]] = values
        means = {label: value/len(encoded["suffixes"][label]) for label, value in values.items()}
        mean_predictions[row["id"]] = predict_label(means) == row["label"]
        ordered = sorted(values, key=lambda label: (-values[label], label))
        covariates[row["id"]] = dict(gold_log_probability=values[row["label"]], gold_rank=ordered.index(row["label"])+1)
        labels[row["id"]] = prediction
        predictions[row["id"]] = prediction == row["label"]
        if generated:
            import torch
            ids = torch.tensor([encoded["prefix_ids"]], device=_device(model))
            was_training = model.training
            model.eval()
            with torch.no_grad(), _compute_context(model):
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        do_sample=False, max_new_tokens=32,
                                        pad_token_id=tokenizer.eos_token_id, use_cache=True)
            model.train(was_training)
            text = tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True)
            generated_labels[row["id"]] = text
            generated_predictions[row["id"]] = text == row["label"]
    return dict(predictions=predictions, labels=labels, summed_scores=score_vectors,
                mean_token_predictions=mean_predictions, gold_covariates=covariates,
                generated_predictions=generated_predictions, generated_labels=generated_labels)


def position_sample(calibration_rows, tokenized):
    """Exactly the same hash-selected content-token positions at every start."""
    import hashlib
    positions = [(row["id"], pos) for row in calibration_rows
                 for pos in tokenized[row["id"]]["content_positions"]]
    positions.sort(key=lambda pair: (hashlib.sha256(("pd-rank-v1|"+pair[0]+"|"+str(pair[1])).encode()).hexdigest(), pair))
    return positions[:1024]


def activation_statistics(z, selected_z=None):
    """z is the actual SwiGLU down-projection input, question content only."""
    import torch
    require(z.ndim == 2 and z.shape[0] > 0 and z.shape[1] > 0, "empty activation matrix")
    require(bool(torch.isfinite(z).all()), "nonfinite SwiGLU activations")
    utility = z.double().abs().mean(0)
    denominator = utility.mean().item()
    dormant = 1.0 if denominator == 0 else (utility/denominator <= .01).double().mean().item()
    rank_z = z if selected_z is None else selected_z
    require(rank_z.ndim == 2 and rank_z.shape[1] == z.shape[1], "rank matrix width mismatch")
    rank, normalized = None, None
    n, width = rank_z.shape
    if n >= 2:
        centered = rank_z.double()-rank_z.double().mean(0)
        # N <= 1024: the Gram matrix avoids a full MLP-width SVD.
        eigenvalues = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0)
        total = eigenvalues.sum().item()
        if total > 0:
            probabilities = eigenvalues[eigenvalues > 0]/total
            rank = torch.exp(-(probabilities*probabilities.log()).sum()).item()
            normalized = rank/min(n-1, width)
    return dict(dormant_fraction=dormant, zero_layer=denominator == 0,
                effective_rank=rank, normalized_effective_rank=normalized, rank_positions=n)


class SwiGLUHooks:
    """Capture down_proj pre-inputs, never gate_proj zeros or a ReLU proxy."""
    def __init__(self, model):
        self.values, self.handles = {}, []
        self.positions = None
        for name, module in model.named_modules():
            if name.endswith(".mlp.down_proj") or name == "mlp.down_proj":
                self.handles.append(module.register_forward_pre_hook(self._hook(name)))
        require(self.handles, "model has no supported SwiGLU down projection")

    def _hook(self, name):
        def capture(module, args):
            import torch
            source = args[0] if self.positions is None else args[0][:, self.positions]
            value = source.detach().float().cpu()
            require(bool(torch.isfinite(value).all()), "nonfinite diagnostic tensor")
            self.values[name] = value
        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()


def representation_diagnostics(model, calibration_rows, tokenized):
    import torch
    sample = set(position_sample(calibration_rows, tokenized))
    all_values, rank_values = {}, {}
    hooks = SwiGLUHooks(model)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _compute_context(model):
            for row in calibration_rows:
                encoded = tokenized[row["id"]]
                ids = torch.tensor([encoded["prefix_ids"]], device=_device(model))
                hooks.values.clear()
                hooks.positions = encoded["content_positions"]
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
                positions = encoded["content_positions"]
                selected = [i for i, p in enumerate(positions) if (row["id"], p) in sample]
                for name, value in hooks.values.items():
                    all_values.setdefault(name, []).append(value[0])
                    if selected:
                        rank_values.setdefault(name, []).append(value[0, selected])
        layers = {name: activation_statistics(torch.cat(values), torch.cat(rank_values[name]))
                  for name, values in all_values.items()}
        return dict(layers=layers, dormant_layer_mean=sum(v["dormant_fraction"] for v in layers.values())/len(layers),
                    positions_sha256=digest(sorted(sample)))
    finally:
        hooks.close()
        model.train(was_training)


def fresh_optimizer(model):
    import torch
    params = list(model.parameters())
    require(params and all(p.requires_grad and p.dtype == torch.float32 for p in params),
            "full-weight CE requires FP32 master parameters")
    optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=0., foreach=False)
    require(not optimizer.state, "optimizer state is not fresh")
    return optimizer


def move_optimizer_moments(optimizer, *, to_cpu):
    """Byte-preserving storage transfer; AdamW arithmetic always stays on GPU.

    Two-rank smoke stages moments on host between steps to leave room for a
    full single-GPU scoring model. Eight-rank node moments stay resident.
    """
    moved = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for key in ("exp_avg", "exp_avg_sq"):
                state = optimizer.state.get(parameter, {})
                if key in state:
                    value = state[key]
                    moved += value.numel()*value.element_size()
                    state[key] = value.to("cpu" if to_cpu else parameter.device)
    return moved


def train_fork(model_factory, batches, tokenized, *, seed, evaluate=None, on_update=None,
               max_updates=40):
    """Factory MUST load untouched bound weights; fresh optimizer per call.

    Microbatch size one gives the same per-example weighting on each start.
    Prefix/evaluation callbacks run with no optimizer mutation. No resume input
    is accepted. The optional two-update prefix is solely for smoke repetition.
    """
    import torch
    distributed = torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    require(world in (1, 2, 8), "unsupported training world size")
    require(seed in SEEDS and len(batches) == 40 and all(len(b) == 32 for b in batches), "invalid probe schedule")
    require(max_updates in (2, 40), "only full probe or smoke prefix allowed")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model_factory()
    model.train()
    optimizer = fresh_optimizer(model)
    receipt = dict(seed=seed, training_world_size=world, training_spec=TRAINING_SPEC, fresh_optimizer=True, initial_optimizer_states=len(optimizer.state),
                   moment_storage="host-between-steps" if world == 2 else "parameter-device",
                   scheduler="constant-no-state", exposure=exposure_receipt(batches[:max_updates], tokenized),
                   losses=[], updates=[], scores={}, diagnostics={})
    if evaluate:
        receipt["scores"][0] = evaluate(model, 0)
    for update, batch in enumerate(batches[:max_updates], 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        summed = 0.
        for row_id in batch[rank::world]:
            encoded = tokenized[row_id]
            require(len(encoded["input_ids"]) <= 1024, "oversize frozen row")
            ids = torch.tensor([encoded["input_ids"]], device=_device(model))
            labels = torch.tensor([encoded["labels"]], device=_device(model))
            with _compute_context(model):
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
                loss = response_ce(logits, labels.to(logits.device))
            require(bool(torch.isfinite(loss)), "nonfinite loss")
            (loss*global_loss_scale(world)).backward()
            summed += loss.item()*global_loss_scale(world)
        if distributed:
            total = torch.tensor(summed, dtype=torch.float64, device=_device(model))
            torch.distributed.all_reduce(total)
            summed = total.item()/world
        norm = (model.clip_grad_norm_(1.) if distributed else
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        require(math.isfinite(float(norm)), "nonfinite global gradient norm")
        if world == 2:
            move_optimizer_moments(optimizer, to_cpu=False)
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "nonfinite model weights")
        require(all(bool(torch.isfinite(v).all()) for state in optimizer.state.values()
                    for v in state.values() if torch.is_tensor(v)), "nonfinite optimizer state")
        require(all(state[key].dtype == torch.float32 for state in optimizer.state.values()
                    for key in ("exp_avg", "exp_avg_sq") if key in state), "optimizer moments must remain FP32")
        optimizer.zero_grad(set_to_none=True)
        if world == 2:
            move_optimizer_moments(optimizer, to_cpu=True)
            torch.cuda.empty_cache()
        receipt["losses"].append(summed)
        receipt["updates"].append(dict(update=update, lr=1e-5, preclip_norm=float(norm), examples=32,
                                        training_seconds=time.monotonic()-started))
        if on_update:
            on_update(model, optimizer, update, receipt)
        if evaluate and update in UPDATES:
            receipt["scores"][update] = evaluate(model, update)
    return model, optimizer, receipt


def global_loss_scale(world_size):
    """FSDP averages rank gradients: local sum * world/32 yields global mean.

    No extra microbatch or world divisor is applied anywhere else.
    """
    require(world_size in (1, 2, 8), "unsupported global loss world size")
    return world_size/32


def gradient_reference_check():
    """Independent CPU autograd reference, including unequal response lengths."""
    import torch
    features = torch.linspace(-.7, .9, 32*6*3, dtype=torch.float64).reshape(32, 6, 3)
    labels = torch.tensor([[-100, -100, 1, 2, 3, -100] if i % 2 else
                           [-100, 2, 1, 2, 3, -100] for i in range(32)])
    initial = torch.linspace(-.5, .6, 12, dtype=torch.float64).reshape(3, 4)
    weight = initial.clone().requires_grad_()
    logits = (features @ weight).float()
    independent = []
    for i in range(32):
        terms = [-logits[i, t-1].log_softmax(0)[labels[i, t]]
                 for t in range(1, 6) if labels[i, t] != -100]
        independent.append(torch.stack(terms).mean())
    (torch.stack(independent).sum()/32).backward()
    expected = weight.grad.clone()
    errors = {}
    for world in (1, 2, 8):
        reduced = torch.zeros_like(expected)
        for rank in range(world):
            candidate = initial.clone().requires_grad_()
            local = torch.arange(rank, 32, world)
            objective = sum(response_ce((features[i:i+1] @ candidate), labels[i:i+1])
                            for i in local)*global_loss_scale(world)
            objective.backward()
            reduced += candidate.grad/world
        errors[str(world)] = (reduced-expected).abs().max().item()
    require(all(v <= 1e-6 for v in errors.values()), "global loss gradient reference mismatch")
    return seal(dict(status="PASS", worlds=[1, 2, 8], maximum_absolute_gradient_errors=errors,
                     tolerance=1e-6, batch=32, unequal_response_lengths=True))


class PDStorage:
    """Isolated 256-GiB working tree, >=320-GiB initial free-space check.

    Only owned pending/scoring exports may be deleted. Source checkpoints must live outside this state tree.
    """
    QUOTA = 256*1024**3
    MIN_FREE = 320*1024**3

    def __init__(self, root, source_roots=(), *, disk_usage=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for source in source_roots:
            source = Path(source).resolve()
            require(source != self.root and self.root not in source.parents and source not in self.root.parents,
                    "PD storage must be separate from source checkpoints")
        require(not list(self.root.iterdir()), "failed fork restarts from update 0 in a clean PD tree")
        usage = (disk_usage or shutil.disk_usage)(self.root)
        require(usage.free >= self.MIN_FREE, "PENDING: insufficient PD free-space preflight")
        self.preflight = seal(dict(schema="pd-storage-preflight-v2", quota_bytes=self.QUOTA,
                                   required_free_bytes=self.MIN_FREE, observed_free_bytes=usage.free))
        write_frozen(self.root/"pd-owned-state.json", dict(schema="pd-owned-state-v2"))
        self.check()

    def check(self, extra_bytes=0):
        used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        require(used+extra_bytes <= self.QUOTA, "PENDING: PD working quota exceeded")
        return used

    def delete_owned(self, target):
        target = Path(target).resolve()
        require(target.parent == self.root and target.name in ("pending-export", "scoring-export"), "refusing deletion outside PD-owned transient state")
        if target.exists():
            shutil.rmtree(target)

    def begin_export(self):
        require(not (self.root/"pending-export").exists() and not (self.root/"scoring-export").exists(),
                "PENDING: a scoring export is already pending")
        self.check(40*1024**3)
        return self.root/"pending-export"

    def transfer(self, checkpoint):
        source, target = self.root/"pending-export", self.root/"scoring-export"
        require(source.is_dir() and not target.exists(), "only one export may await scoring")
        validate_checkpoint(checkpoint)
        started = time.monotonic()
        source.rename(target)  # Atomic within the dedicated PD filesystem; no tensor copy/hash.
        self.check()
        return target, time.monotonic()-started

    def complete(self, checkpoint, scored):
        require(scored["checkpoint_id"] == checkpoint["checkpoint_id"], "wrong scored export")
        verify_seal(scored)
        proof = scored["live_load_verification"]
        verify_seal(proof)
        require(proof == live_load_verification(checkpoint), "scorer did not verify the bound export")
        verification = seal(dict(schema="pd-stream-verification-v2", checkpoint_id=checkpoint["checkpoint_id"],
                                 tensor_files=checkpoint["tensor_files"], weights_sha256=checkpoint["weights_sha256"],
                                 export_receipt_sha256=checkpoint["export_receipt_sha256"], score_sha256=digest(scored),
                                 predictions_sha256=digest(scored["predictions"]),
                                 summed_scores_sha256=digest(scored["summed_scores"]), covariates_sha256=digest(scored["gold_covariates"]),
                                 gpu_identity=scored["gpu_identity"], disposition="verified-scored-deleted"))
        write_frozen(self.root/"retained"/(checkpoint["checkpoint_id"]+".json"),
                     dict(checkpoint=checkpoint, verification=verification, metrics=scored))
        self.delete_owned(self.root/"scoring-export")
        self.check()
        return verification


def _master_call(function):
    """All ranks see master I/O failures; no rank silently advances a curve."""
    import torch
    if not torch.distributed.is_initialized():
        return function()
    value = [None]
    if torch.distributed.get_rank() == 0:
        try:
            value[0] = dict(ok=True, result=function())
        except Exception as error:
            value[0] = dict(ok=False, reason=str(error) if isinstance(error, ValueError) else type(error).__name__,
                            failure_kind=failure_kind(error) if isinstance(error, RuntimeError) else None)
    torch.distributed.broadcast_object_list(value, src=0)
    if not value[0]["ok"] and value[0].get("failure_kind"):
        raise QualificationFailure(value[0]["failure_kind"])
    require(value[0]["ok"], "PENDING: isolated PD operation: "+value[0].get("reason", "unknown"))
    return value[0]["result"]


class QualificationFailure(RuntimeError):
    pass


def failure_kind(error):
    if isinstance(error, QualificationFailure):
        return str(error)
    text = str(error).lower()
    return "oom" if "out of memory" in text else "determinism" if "determin" in text else "runtime"


def failure_receipt(error):
    kind = failure_kind(error)
    return seal(dict(schema="pd-smoke-receipt-v2", failure_kind=kind,
                     oom=kind == "oom", determinism_failed=kind == "determinism",
                     receipts_complete=False, spending_cutoff=False))


def live_load_verification(checkpoint):
    return seal(dict(schema="pd-live-load-verification-v2", checkpoint_id=checkpoint["checkpoint_id"],
                     tensor_files=checkpoint["tensor_files"], weights_sha256=checkpoint["weights_sha256"],
                     export_receipt_sha256=checkpoint["export_receipt_sha256"],
                     export_dtype=checkpoint.get("export_dtype"), verified=True))


def scoring_replica(config, bf16_state, device):
    """Score exactly the serialized BF16 state using the reload numerical setup.

    A forward in summon_full_params is forbidden by FSDP. This independent
    replica also avoids nested FSDP wrappers and never mutates training masters.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(copy.deepcopy(config), torch_dtype=torch.float32,
                                                attn_implementation="eager")
    model.load_state_dict(bf16_state, strict=True, assign=True)
    model.tie_weights()
    return model.to(device=device, dtype=torch.float32).eval()


def load_sharded_model(directory):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    rank = int(os.environ["LOCAL_RANK"])
    module = load_local_model(directory, device_map={"": rank})
    classes = {type(m) for m in module.modules() if type(m).__name__ in module._no_split_modules}
    require(classes, "transformer block auto-wrap classes missing")
    return FSDP(module, device_id=rank, use_orig_params=True,
                auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=classes),
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                               buffer_dtype=torch.float32))


def gather_export(model, storage, parent, update, *, calibration=None, tokenized=None, scoring_device=None):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    started = time.monotonic()
    _master_call(storage.begin_export)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = model.state_dict()
    original = None
    def export():
        nonlocal original
        bf16_state = export_state(state)
        state.clear()  # Release the full FP32 CPU gather before constructing the replica.
        if calibration is not None:
            replica = scoring_replica(model.module.config, bf16_state, scoring_device)
            original = score_rows(replica, calibration, tokenized)
            del replica
            torch.cuda.empty_cache()
        # Supplied full CPU state avoids reading another rank's local flattened storage.
        checkpoint = export_probe(model.module, storage.root/"pending-export", parent, update, state=bf16_state)
        return dict(checkpoint=checkpoint, original=original)
    exported = _master_call(export)
    del state
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    return exported["checkpoint"], time.monotonic()-started, exported["original"]


def bind_chain(checkpoints, send_binding):
    """Use the exact pre-bound v4 FinQA and C sequence-D parents."""
    verify_seal(send_binding)
    require(set(checkpoints) == set(STARTS), "PENDING: missing required chain exports")
    for start, checkpoint in checkpoints.items():
        validate_checkpoint(checkpoint)
        require(checkpoint.get("role", "chain_start") == "chain_start", "probe export cannot substitute for a chain start")
        require(checkpoint["start"] == start, "start identity mismatch")
        require(checkpoint["base_revision"] == BASE_REVISION, "wrong original base revision")
        require(checkpoint["send_binding_sha256"] == send_binding["identity_sha256"], "send binding mismatch")
        expected = send_binding["pd_starts"][str(checkpoint["seed"])][start]
        # Bind immutable content fields, avoiding a circular hash between the
        # send binding and a checkpoint receipt which itself names that binding.
        required = set(("checkpoint_id", "weights_sha256", "export_receipt_sha256", "parent_id",
                        "recipe", "step", "base_revision", "tokenizer_sha256"))
        require(set(expected) == required and all(checkpoint[k] == expected[k] for k in required),
                "checkpoint is not the exact bound plain-D start")
    base, first, second = [checkpoints[s] for s in STARTS]
    require(len({c["seed"] for c in checkpoints.values()}) == 1, "chain seed mismatch")
    require(base["checkpoint_id"].endswith(f"@seed{base['seed']}"), "base checkpoint ID must be logical per-seed ID")
    require(first["parent_id"] == base["checkpoint_id"] and second["parent_id"] == first["checkpoint_id"],
            "wrong sequence parent links")
    require(len({c["tokenizer_sha256"] for c in checkpoints.values()}) == 1, "chain tokenizer mismatch")
    return checkpoints


def load_local_model(directory, *, device_map="balanced"):
    import torch
    from transformers import AutoModelForCausalLM
    model, loading = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
                                                torch_dtype=torch.float32, device_map=device_map,
                                                attn_implementation="eager", output_loading_info=True)
    require(not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            "checkpoint did not load all and only its bound tensors")
    require(model.config._attn_implementation == "eager", "deterministic attention implementation changed")
    require(model.config.model_type == "qwen3", "PD requires actual Qwen3")
    require(model.config.hidden_act == "silu", "PD diagnostics require actual SwiGLU")
    require(7_000_000_000 <= sum(p.numel() for p in model.parameters()) <= 9_000_000_000,
            "PD requires actual 8B weights")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def score_worker(config, checkpoint_directory, start, update, output, *, smoke=False):
    """A fresh process on exactly one assigned physical scoring GPU."""
    configure_determinism()
    import torch
    from transformers import AutoTokenizer
    require(torch.cuda.device_count() == 1, "one physical scoring GPU required")
    identity = gpu_identity()
    require(identity == config["scoring_gpu_identity"], "assigned scoring GPU/environment changed")
    preparation = json.loads(Path(config["inputs"] if smoke else config["preparation"]).read_text())
    verify_seal(preparation)
    require(preparation["code_sha256"] == code_identity(), "scoring code changed")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    require(tokenizer_identity(config["tokenizer"]) == preparation["tokenizer_sha256"], "scoring tokenizer changed")
    require(all(encode_row(tokenizer, row) == preparation["tokenized"][row["id"]] for row in preparation["rows"]),
            "scoring tokenization/masks changed")
    directory = Path(checkpoint_directory)
    checkpoint = (json.loads((directory/"checkpoint.json").read_text()) if update else
                  (config["base_checkpoint"] if smoke else config["checkpoints"][start]))
    tensors = {p.name: p for p in directory.iterdir() if p.suffix in (".safetensors", ".bin")}
    receipt = directory/"pd-export.json" if update else Path(
        config["base_receipt"] if smoke else config["exports"][start]["receipt"])
    verify_live_export(checkpoint, tensors, receipt, directory=directory)
    loaded = time.monotonic()
    model = load_local_model(directory, device_map={"": 0})
    torch.cuda.synchronize()
    load_seconds = time.monotonic()-loaded
    scoring = time.monotonic()
    ids = preparation["partitions"]["calibration" if smoke else "evaluation"]
    rows = [r for r in preparation["rows"] if r["id"] in ids]
    result = score_rows(model, rows, preparation["tokenized"], generated=True, tokenizer=tokenizer)
    if update in (0, 40):
        calibration = [r for r in preparation["rows"] if r["id"] in preparation["partitions"]["calibration"]]
        result["diagnostics"] = representation_diagnostics(model, calibration, preparation["tokenized"])
    torch.cuda.synchronize()
    result.update(checkpoint_id=checkpoint["checkpoint_id"], gpu_identity=identity,
                  live_load_verification=live_load_verification(checkpoint),
                  load_seconds=load_seconds, score_seconds=time.monotonic()-scoring)
    receipt_started = time.monotonic()
    write_frozen(output, seal(result))
    write_frozen(str(output)+".timing.json", seal(dict(receipt_seconds=time.monotonic()-receipt_started)))


def score_on_gpu(config, plan_filename, directory, start, update, storage, *, smoke=False):
    destination = storage.root/f"score-{start}-{update}.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config["score_device"]),
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command = [sys.executable, str(Path(__file__).resolve()), "--plan", str(plan_filename),
               "--out", str(destination), "--score-worker", "--checkpoint-dir", str(directory),
               "--start", start, "--update", str(update)]
    if smoke:
        command.append("--smoke-worker")
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if completed.returncode:
        failures = list(storage.root.glob("failure-*.json"))
        if failures:
            observed = json.loads(failures[0].read_text())
            verify_seal(observed)
            raise QualificationFailure(observed["failure_kind"])
        raise ValueError("PENDING: isolated single-GPU scorer did not complete")
    scored = json.loads(destination.read_text())
    verify_seal(scored)
    io = json.loads(Path(str(destination)+".timing.json").read_text())
    verify_seal(io)
    scored = seal({**scored, "receipt_seconds": io["receipt_seconds"]})
    require(scored["gpu_identity"] == config["scoring_gpu_identity"], "scorer assignment changed")
    storage.check()
    return scored


def _local_weights_digest(model):
    import torch
    h = __import__("hashlib").sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().float().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    value = h.hexdigest()
    values = [None]*torch.distributed.get_world_size()
    torch.distributed.all_gather_object(values, value)
    return digest(values)


def distributed_worker(config, plan_filename, output, *, smoke=False):
    started = time.monotonic()
    import torch
    from datetime import timedelta, date
    from kit.pd_contract import make_manifest, validate_score
    from kit.pd_read import freeze_common_errors
    from kit.pd_prepare import SOURCES, verify_returned_pools
    from kit.pd_smoke_modal import (validate_inputs, smoke_gates, export_agreement,
                                    RESERVATION, validate_price, PeakSampler)
    configure_determinism()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    # The smoke supervisor bounds bootstrap to 60 s; later scoring collectives
    # retain their long timeout while the assigned GPU scores an entire panel.
    torch.distributed.init_process_group("nccl", timeout=timedelta(hours=24))
    if smoke:
        # Force lazy NCCL communicator setup before declaring startup complete.
        torch.distributed.barrier(device_ids=[rank])
        write_frozen(Path(output).parent/f"distributed-ready-rank-{rank}.json", seal(dict(
            rank=rank, master_addr=os.environ["MASTER_ADDR"],
            nccl_socket_ifname=os.environ["NCCL_SOCKET_IFNAME"],
            gloo_socket_ifname=os.environ["GLOO_SOCKET_IFNAME"])))
    world = torch.distributed.get_world_size()
    require(world == (2 if smoke else 8), "smoke requires two ranks; node requires eight")
    require(len(config["scoring_gpu_identity"]["devices"]) == 1 and
            config["scoring_gpu_identity"]["uuid"] in gpu_identity()["devices"],
            "the single assigned scoring GPU must belong to the reserved worker GPUs")
    gradient_reference = gradient_reference_check()
    require(all("H100" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())),
            "registered runtime requires H100s")
    preparation = json.loads(Path(config["inputs"] if smoke else config["preparation"]).read_text())
    require(config.get("gpu_rate_per_hour") is None or
            (math.isfinite(config["gpu_rate_per_hour"]) and config["gpu_rate_per_hour"] >= 0), "invalid measured GPU rate")
    verify_seal(preparation)
    require(preparation["code_sha256"] == code_identity(), "frozen preparation code changed")
    seed = 101 if smoke else config["seed"]
    if smoke:
        validate_inputs(preparation)
        bindings = {"base": config["base_checkpoint"]}
        source_roots = [config["base"]]
        ceiling = validate_price(config["quote"])
    else:
        require(preparation["schema"] == "pd-preparation-v2" and
                preparation["source_hashes"] == {k: v["sha256"] for k, v in SOURCES.items()},
                "wrong source/registration preparation")
        bindings = bind_chain(config["checkpoints"], config["send_binding"])
        qualification = config["smoke_receipt"]
        require(smoke_gates(qualification)["status"] == "QUALIFIED"
                and qualification["code_sha256"] == code_identity(), "PENDING: actual two-rank smoke qualification required")
        source_roots = [config["exports"][s]["directory"] for s in STARTS]
    require(seed in SEEDS, "unregistered seed")
    require(all(c["base_revision"] == BASE_REVISION and c["tokenizer_sha256"] == preparation["tokenizer_sha256"]
                for c in bindings.values()), "wrong pinned start/tokenizer")
    # Each rank reads the dedicated shared PD tree, never a source/core tree.
    storage = object.__new__(PDStorage)
    storage.root = Path(output).resolve()
    storage.preflight = _master_call(lambda: PDStorage(output, source_roots).preflight)
    if torch.distributed.get_rank() == 0:
        write_frozen(storage.root/"storage-preflight.json", storage.preflight)
    batches = seed_batches(preparation["partitions"]["training"], seed)
    exposure = exposure_receipt(batches, preparation["tokenized"])
    initial = {}
    for start, checkpoint in bindings.items():
        directory = config["base"] if smoke else config["exports"][start]["directory"]
        initial[start] = _master_call(lambda start=start, directory=directory:
            score_on_gpu(config, plan_filename, directory, start, 0, storage, smoke=smoke))
    cohort = None if smoke else freeze_common_errors({s: initial[s]["predictions"] for s in STARTS})
    if not smoke:
        _master_call(lambda: write_frozen(storage.root/"common-errors.json", cohort))
    checkpoints, identities, runs, stream = list(bindings.values()), [], [], {}
    smoke_original = None
    first_initial_hash = None
    final_hash = None
    for start in bindings:
        fork_started = time.monotonic()
        directory = config["base"] if smoke else config["exports"][start]["directory"]
        initial_hashes = []
        def factory():
            model = load_sharded_model(directory)
            # Empty optimizer state is asserted independently inside train_fork.
            if smoke:
                initial_hashes.append(_local_weights_digest(model))
            return model
        def evaluate(model, update):
            nonlocal smoke_original, final_hash
            if update == 0:
                scored = initial[start]
                checkpoint = bindings[start]
            elif smoke and update != 40:
                return {}
            else:
                if smoke:
                    calibration = [r for r in preparation["rows"] if r["id"] in preparation["partitions"]["calibration"]]
                    final_hash = _local_weights_digest(model)
                checkpoint, export_seconds, smoke_original = gather_export(
                    model, storage, bindings[start], update,
                    calibration=calibration if smoke else None, tokenized=preparation["tokenized"],
                    scoring_device=gpu_identity()["devices"].index(config["scoring_gpu_identity"]["uuid"]))
                def transfer_score_delete():
                    write_frozen(storage.root/"pending-export"/"checkpoint.json", checkpoint)
                    copied, transfer_seconds = storage.transfer(checkpoint)
                    scored = score_on_gpu(config, plan_filename, copied, start, update, storage, smoke=smoke)
                    verification = storage.complete(checkpoint, scored)
                    return dict(scored=scored, verification=verification, transfer_seconds=transfer_seconds)
                transferred = _master_call(transfer_score_delete)
                scored = transferred["scored"]
                verification = transferred["verification"]
                checkpoint = seal({**checkpoint, "stream_verification_sha256": verification["identity_sha256"]})
                stream[checkpoint["checkpoint_id"]] = verification
                checkpoints.append(checkpoint)
                scored = seal(dict(scored, export_seconds=export_seconds, transfer_seconds=transferred["transfer_seconds"],
                              gather_export_gpu_hours=world*export_seconds/3600,
                              transfer_score_idle_gpu_hours=world*(transferred["transfer_seconds"]+scored["load_seconds"]+scored["score_seconds"]+scored["receipt_seconds"])/3600))
            if smoke:
                return scored
            score_id = f"pd-{seed}-{start}-{update}"
            identity = seal(dict(score_id=score_id, checkpoint_id=checkpoint["checkpoint_id"], seed=seed, update=update,
                panel_sha256=preparation["identity_sha256"], prompt_sha256=preparation["prompt_sha256"],
                scorer_sha256=sha256_file(__file__), decoding_sha256=digest(dict(thinking=False, generated_cap=32,
                label_rule="sum-label-plus-eos-lexical-ties", generated_greedy=True)),
                tokenizer_sha256=preparation["tokenizer_sha256"],
                suffix_sha256=digest(next(iter(preparation["tokenized"].values()))["suffixes"]),
                gpu_identity=scored["gpu_identity"], predictions_sha256=digest(scored["predictions"]),
                summed_scores_sha256=digest(scored["summed_scores"]), covariates_sha256=digest(scored["gold_covariates"]),
                exposure_sha256=exposure["identity_sha256"], scoring_rule="sum-label-plus-eos"))
            identities.append(validate_score(identity))
            return seal(dict(scored, score_id=score_id))
        model, optimizer, receipt = train_fork(factory, batches, preparation["tokenized"], seed=seed,
                                               evaluate=evaluate, on_update=None)
        if smoke:
            first_initial_hash = initial_hashes[0]
            first = receipt
            reloaded = receipt["scores"][40]
            del model, optimizer
            torch.cuda.empty_cache()
            repeat_model, repeat_optimizer, repeat = train_fork(factory, batches, preparation["tokenized"], seed=101, max_updates=2)
            base_after_hash = initial_hashes[1]
            del repeat_model, repeat_optimizer
            torch.cuda.empty_cache()
        else:
            score_wall = sum(s.get("load_seconds", 0)+s.get("score_seconds", 0)+s.get("receipt_seconds", 0)
                             for s in receipt["scores"].values())
            wall = time.monotonic()-fork_started+sum(initial[start].get(k, 0) for k in ("load_seconds", "score_seconds", "receipt_seconds"))
            receipt["timing"] = dict(wall_seconds=wall, serial_scoring_wall_seconds=score_wall,
                charged_gpu_count=world, gpu_hours=world*wall/3600,
                export_gpu_hours=sum(s.get("gather_export_gpu_hours", 0) for s in receipt["scores"].values()),
                wall_time_risk=score_wall > 24*3600,
                h100_cost=world*wall/3600*config["gpu_rate_per_hour"] if config.get("gpu_rate_per_hour") is not None else None)
            receipt.update(start=start, checkpoint_sha256=bindings[start]["identity_sha256"],
                           preparation_sha256=preparation["identity_sha256"], cohort_sha256=cohort["identity_sha256"])
            runs.append(seal(receipt))
            _master_call(lambda: write_frozen(storage.root/(start+"-probe.json"), runs[-1]))
            del model, optimizer
            torch.cuda.empty_cache()
    elapsed = time.monotonic()-started
    timing = dict(wall_seconds=elapsed, charged_gpu_count=world, gpu_hours=world*elapsed/3600,
                  h100_cost=(world*elapsed/3600*config["gpu_rate_per_hour"] if config.get("gpu_rate_per_hour") is not None else None),
                  serial_score_seconds=sum(s.get("load_seconds", 0)+s.get("score_seconds", 0)+s.get("receipt_seconds", 0) for r in runs for s in r["scores"].values()),
                  wall_time_risk=elapsed > 24*3600)
    if smoke:
        before, after = initial["base"]["predictions"], reloaded["predictions"]
        wrong = [i for i, correct in before.items() if not correct]
        agreement = export_agreement(smoke_original["summed_scores"], reloaded["summed_scores"])
        first_two = exposure_receipt(first["exposure"]["batches"][:2], preparation["tokenized"])
        result = dict(schema="pd-smoke-receipt-v2", date=date.today().isoformat(), quote=config["quote"],
            reservation=RESERVATION, scope=config.get("scope", "registered-smoke"),
            scientific_preparation="PENDING" if config.get("scope") == "SMOKE-SCOPE" else "not-applicable",
            scoring_gpu_identity=config["scoring_gpu_identity"],
            export_dtype="bfloat16", export_scoring_setup="unsharded-fp32-autocast-bf16", model="Qwen3-8B", seed=101, code_sha256=code_identity(), training_world_size=2,
            calibration_before=100*sum(before.values())/len(before), calibration_after=100*sum(after.values())/len(after),
            initially_wrong_improvement=100*sum(after[i] for i in wrong)/len(wrong) if wrong else None,
            losses=first["losses"], repeated_losses=repeat["losses"], updates=40,
            gradient_reference=gradient_reference,
            determinism=dict(algorithms=torch.are_deterministic_algorithms_enabled(),
                             workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"), attention="eager"),
            initial_optimizer_states=[first["initial_optimizer_states"], repeat["initial_optimizer_states"]],
            first_two_exposure_sha256=first_two["identity_sha256"], repeat_exposure_sha256=repeat["exposure"]["identity_sha256"],
            full_exposure_sha256=first["exposure"]["identity_sha256"], expected_exposure_sha256=exposure["identity_sha256"],
            first_two_batch_sha256=digest(first["exposure"]["batches"][:2]), repeat_batch_sha256=digest(repeat["exposure"]["batches"]),
            finite_gradients_weights_diagnostics=True,
            diagnostics={0: initial["base"]["diagnostics"], 40: reloaded["diagnostics"]},
            initial_weights_sha256=first_initial_hash, final_weights_sha256=final_hash,
            base_before_sha256=first_initial_hash, base_after_sha256=base_after_hash,
            export_agreement=agreement, original_export_scores=smoke_original["summed_scores"],
            reloaded_export_scores=reloaded["summed_scores"], timing=timing,
            oom=False, receipts_complete=True, reservation_cost_ceiling=ceiling,
            seconds_including_receipts=elapsed, spending_cutoff=False)
        # Device and aggregate host peaks are injected by the live sampler on the
        # master rank; they are measurements/warnings, never correctness gates.
        return result
    manifest = make_manifest(checkpoints, identities, preparation_sha256=preparation["identity_sha256"], code_sha256=code_identity())
    returned = verify_returned_pools(preparation, config["returned_pools"]) if config.get("returned_pools") else None
    result = dict(manifest=manifest, runs=runs, cohorts={str(seed): cohort}, send_binding=config["send_binding"],
                  stream_verifications=stream, timing=timing, return_verification=returned,
                  smoke_receipt=config["smoke_receipt"])
    _master_call(lambda: write_frozen(storage.root/"manifest.json", manifest))
    _master_call(lambda: write_frozen(storage.root/"seed-evidence.json", result))
    return result


def distributed_launch(plan, output, *, smoke=False):
    """Pure launch specification; only the smoke pins all local TCP addresses."""
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    if isinstance(plan, dict):
        env["PD_OPERATOR_PLAN"] = json.dumps(plan)
        plan = "__environment__"
    rendezvous = ["--standalone"]
    if smoke:
        # c10d chooses an unused loopback port. local-addr also pins the separate
        # rank bootstrap store; endpoint alone still permits hostname discovery.
        rendezvous = ["--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0", "--rdzv-id=pd-smoke",
                      "--local-addr=127.0.0.1",
                      "--rdzv-conf=is_host=true,read_timeout=15,join_timeout=30,last_call_timeout=1,close_timeout=5",
                      "--max-restarts=0", "--monitor-interval=0.1"]
        env.update(MASTER_ADDR="127.0.0.1", NCCL_SOCKET_IFNAME="lo", GLOO_SOCKET_IFNAME="lo",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                   TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    command = [sys.executable, "-m", "torch.distributed.run", *rendezvous, "--nnodes=1",
               "--nproc-per-node="+("2" if smoke else "8"), str(Path(__file__).resolve()),
               "--plan", str(plan), "--out", str(output), "--train-worker"]
    if smoke:
        command.append("--smoke-worker")
    return command, env


def wait_for_smoke_startup(process, output, started):
    """Fail locally by 60 seconds even if a library ignores its socket timeout."""
    markers = [Path(output).parent/f"distributed-ready-rank-{rank}.json" for rank in range(2)]
    while not all(p.is_file() for p in markers):
        if process.poll() is not None:
            raise ValueError("PENDING: distributed startup failed before both ranks were ready")
        remaining = SMOKE_STARTUP_SECONDS-(time.monotonic()-started)
        if remaining <= 0:
            raise ValueError("PENDING: distributed startup exceeded 60 seconds")
        try:
            process.wait(timeout=min(.1, remaining))
        except subprocess.TimeoutExpired:
            pass


def launch_ranks(plan, output, *, smoke=False, timeout=None):
    """Explicit operator/runtime launch, with bounded smoke bootstrap and timing."""
    command, env = distributed_launch(plan, output, smoke=smoke)
    started = time.monotonic()
    process = subprocess.Popen(command, env=env, start_new_session=True)
    phase = "distributed_startup" if smoke else "worker"
    try:
        if smoke:
            wait_for_smoke_startup(process, output, started)
            phase = "worker"
        remaining = None if timeout is None else max(0, timeout-(time.monotonic()-started))
        code = process.wait(timeout=remaining)
        require(code == 0, "PENDING: distributed PD worker did not complete")
    except BaseException:
        import signal
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        if smoke and phase == "distributed_startup":
            write_frozen(Path(output)/"distributed-startup.json", seal(dict(
                status="PENDING", failure_kind="smoke_environment", phase=phase,
                startup_limit_seconds=SMOKE_STARTUP_SECONDS, rendezvous_read_timeout_seconds=15,
                rendezvous="loopback-c10d", seconds=time.monotonic()-started)))
        raise
    finally:
        elapsed = time.monotonic()-started
        write_frozen(Path(output)/"launcher-timing.json", seal(dict(wall_seconds=elapsed,
                     charged_gpu_count=2 if smoke else 8, gpu_hours=(2 if smoke else 8)*elapsed/3600,
                     includes_worker_startup=True, last_phase=phase)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--out")
    parser.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--score-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--start", choices=STARTS)
    parser.add_argument("--update", type=int, choices=UPDATES)
    parser.add_argument("--gpu-identity", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpu_identity:
        configure_determinism()
        identity = gpu_identity()
        require(len(identity["devices"]) == 1, "single scoring GPU discovery required")
        print(json.dumps(identity))
        return 0
    require(args.plan and args.out, "--plan and --out required")
    config = json.loads(os.environ["PD_OPERATOR_PLAN"] if args.plan == "__environment__" else Path(args.plan).read_text())
    try:
        if args.score_worker:
            score_worker(config, args.checkpoint_dir, args.start, args.update, args.out, smoke=args.smoke_worker)
        elif args.train_worker:
            if args.smoke_worker:
                from kit.pd_smoke_modal import PeakSampler, smoke_gates
                from contextlib import nullcontext
                import torch
                configure_determinism()
                # The sampler spans initial scoring, training, export and repeat.
                context = PeakSampler() if int(os.environ["RANK"]) == 0 else nullcontext()
                with context as peaks:
                    result = distributed_worker(config, args.plan, args.out, smoke=True)
                if torch.distributed.get_rank() == 0:
                    result.update(gpu_peak_fractions=peaks.gpu, aggregate_host_peak_bytes=peaks.host)
                    result = seal(result)
                    write_frozen(Path(args.out)/"smoke.json", result)
                    write_frozen(Path(args.out)/"gates.json", smoke_gates(result))
            else:
                distributed_worker(config, args.plan, args.out)
        else:
            launch_ranks(args.plan, args.out, smoke=args.smoke_worker)
    except RuntimeError as error:
        if args.smoke_worker:
            from kit.pd_smoke_modal import smoke_gates
            root = Path(args.out).parent if args.score_worker else Path(args.out)
            result = failure_receipt(error)
            write_frozen(root/("failure-"+("score" if args.score_worker else os.environ.get("RANK", "0"))+".json"), result)
            if not args.score_worker and int(os.environ.get("RANK", "0")) == 0:
                write_frozen(root/"smoke.json", result)
                write_frozen(root/"gates.json", smoke_gates(result))
        return 3
    except (OSError, ValueError, subprocess.SubprocessError):
        import torch
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            root = Path(args.out).parent if args.score_worker else Path(args.out)
            write_frozen(root/"pending.json", dict(status="PENDING", reason="isolated PD runtime did not finish",
                                                   core_untouched=True))
        return 3
    finally:
        import torch
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
