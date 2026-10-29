"""Unlaunched, explicit PD Modal smoke definition and fail-closed gates.

Importing this file or invoking its CLI never launches Modal. An operator may
construct the app only after binding an all-reservation price quote <= $10.
Model/tokenizer/preparation must already exist on the supplied mounted volume.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import threading
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kit.pd_contract import digest, require, seal, sha256_file, verify_live_export, verify_seal

RESERVATION = dict(gpus=2, gpu="H100", memory_gib=128, cpus=4, vcpus=8, seconds=2700, retries=0)


def validate_price(quote, *, today=None):
    """Price the complete reservation; never substitute measured busy time."""
    today = today or date.today()
    verify_seal(quote)
    require(quote["source"] == "https://modal.com/pricing", "current Modal price source required")
    age = (today-date.fromisoformat(quote["date"])).days
    require(0 <= age <= 7, "current bound pricing quote required")
    rates = quote["rates_per_second"]
    require(set(rates) == {"H100", "cpu", "memory_gib"}, "incomplete reservation rates")
    require(all(math.isfinite(v) and v > 0 for v in rates.values()), "invalid reservation rates")
    overhead = quote["ceilings"]
    require(set(overhead) == {"startup", "storage", "egress"} and
            all(math.isfinite(v) and v >= 0 for v in overhead.values()), "explicit overhead ceilings required")
    total = RESERVATION["seconds"]*(RESERVATION["gpus"]*rates["H100"]+RESERVATION["cpus"]*rates["cpu"]+RESERVATION["memory_gib"]*rates["memory_gib"])+sum(overhead.values())
    require(total <= 10, "DEFERRED: complete reservation ceiling exceeds $10")
    if "reservation" in quote:
        require(quote["reservation"] == RESERVATION, "quoted reservation differs from registered resources")
        require(abs(quote["total_usd"]-total) <= 1e-8, "quoted total differs from full reservation")
    return total


def smoke_inputs(preparation):
    verify_seal(preparation)
    ids = set(preparation["partitions"]["calibration"]+preparation["partitions"]["training"])
    return seal(dict(schema="pd-smoke-inputs-v2", preparation_sha256=preparation["identity_sha256"],
                     partitions={k: preparation["partitions"][k] for k in ("calibration", "training")},
                     rows=[r for r in preparation["rows"] if r["id"] in ids],
                     tokenized={i: preparation["tokenized"][i] for i in sorted(ids)},
                     tokenizer_sha256=preparation["tokenizer_sha256"], code_sha256=preparation["code_sha256"]))


def validate_inputs(inputs):
    verify_seal(inputs)
    require(inputs["schema"] == "pd-smoke-inputs-v2" and
            set(inputs["partitions"]) == {"calibration", "training"}, "smoke cannot access evaluation/chain outcomes")
    cal, train = inputs["partitions"]["calibration"], inputs["partitions"]["training"]
    require(len(cal) == 128 and len(train) == 1280 and len(set(cal+train)) == 1408, "wrong smoke inputs")
    require(set(inputs["tokenized"]) == set(cal+train) == {r["id"] for r in inputs["rows"]}, "smoke input coverage mismatch")
    require(all(r["split"] == "train" for r in inputs["rows"]), "smoke requires training-source questions only")
    return inputs


def export_agreement(original, reloaded):
    require(original and set(original) == set(reloaded), "export/reload score coverage mismatch")
    disagreements = []
    for item_id in sorted(original):
        before, after = original[item_id], reloaded[item_id]
        require(len(before) >= 2 and set(before) == set(after), "export/reload label coverage mismatch")
        require(all(math.isfinite(v) for v in list(before.values())+list(after.values())), "nonfinite export likelihood")
        order_before = sorted(before, key=lambda label: (-before[label], label))
        order_after = sorted(after, key=lambda label: (-after[label], label))
        if order_before[0] != order_after[0]:
            disagreements.append(dict(id=item_id, original_label=order_before[0], reloaded_label=order_after[0],
                original_margin=before[order_before[0]]-before[order_before[1]],
                reloaded_margin=after[order_after[0]]-after[order_after[1]]))
    fraction = 1-len(disagreements)/len(original)
    return seal(dict(coverage=len(original), agreement_fraction=fraction, disagreements=disagreements,
                     original_sha256=digest(original), reloaded_sha256=digest(reloaded),
                     qualified=fraction >= .99 and all(d["original_margin"] < 1e-3 and
                                                       d["reloaded_margin"] < 1e-3 for d in disagreements)))


def smoke_gates(receipt):
    """Correctness/headroom gates; timing and 90% utilization are warnings only."""
    try:
        verify_seal(receipt)
        if receipt.get("spending_cutoff"):
            return dict(status="PENDING", gates={}, failed=[], scientific_seed_result=False,
                        reason="spending cutoff: unfinished checks require revised budget", warnings=["spending cutoff"])
        if receipt.get("failure_kind") in ("oom", "determinism", "runtime"):
            kind = receipt["failure_kind"]
            return dict(status="DEFERRED", gates={kind: False}, failed=[kind],
                        scientific_seed_result=False, warnings=[])
        bound = validate_price(receipt["quote"], today=date.fromisoformat(receipt["date"]))
        losses, repeated = receipt["losses"], receipt["repeated_losses"]
        agreement = export_agreement(receipt["original_export_scores"], receipt["reloaded_export_scores"])
        reference = receipt["gradient_reference"]
        verify_seal(reference)
        repeat_ok = (len(losses) == 40 and len(repeated) == 2 and
                     all(math.isfinite(v) for v in losses+repeated) and losses[0] == repeated[0] and
                     abs(losses[1]-repeated[1])/max(abs(losses[1]), 1e-12) <= 1e-3)
        gates = dict(
            reservation=receipt["reservation"] == RESERVATION,
            actual_model=receipt["model"] == "Qwen3-8B" and receipt["seed"] == 101 and receipt["training_world_size"] == 2,
            headroom=0 <= receipt["calibration_before"] <= 70 and 0 <= receipt["calibration_after"] <= 100,
            improvement=receipt["calibration_after"]-receipt["calibration_before"] >= 5,
            initially_wrong=receipt["initially_wrong_improvement"] is not None and receipt["initially_wrong_improvement"] >= 10,
            forty_updates=receipt["updates"] == 40 and len(losses) == 40,
            fresh_optimizers=receipt["initial_optimizer_states"] == [0, 0],
            repeated_loss=repeat_ok,
            exposure=receipt["first_two_exposure_sha256"] == receipt["repeat_exposure_sha256"],
            full_exposure=receipt["full_exposure_sha256"] == receipt["expected_exposure_sha256"],
            export_precision=receipt["export_dtype"] == "bfloat16" and
                             receipt["export_scoring_setup"] == "unsharded-fp32-autocast-bf16",
            scoring_gpu=len(receipt["scoring_gpu_identity"]["devices"]) == 1 and
                        receipt["scoring_gpu_identity"]["devices"] == [receipt["scoring_gpu_identity"]["uuid"]],
            batch=receipt["first_two_batch_sha256"] == receipt["repeat_batch_sha256"],
            finite=receipt["finite_gradients_weights_diagnostics"] is True and
                   all(math.isfinite(v) for v in losses+repeated) and finite_diagnostics(receipt["diagnostics"]),
            gradient_reference=reference["status"] == "PASS" and reference["worlds"] == [1, 2, 8] and
                               set(reference["maximum_absolute_gradient_errors"]) == {"1", "2", "8"} and
                               all(0 <= v <= 1e-6 for v in reference["maximum_absolute_gradient_errors"].values()),
            determinism=receipt["determinism"] == dict(algorithms=True, workspace=":4096:8", attention="eager"),
            changed=receipt["initial_weights_sha256"] != receipt["final_weights_sha256"],
            export_reload=agreement == receipt["export_agreement"] and agreement["coverage"] == 128 and agreement["qualified"],
            untouched_base=receipt["base_before_sha256"] == receipt["base_after_sha256"],
            no_oom=receipt["oom"] is False,
            completed=receipt["receipts_complete"] is True,
            budget=receipt["reservation_cost_ceiling"] == bound and bound <= 10,
        )
        warnings = []
        if receipt.get("seconds_including_receipts", 0) > 2700:
            warnings.append("45-minute spending-plan duration exceeded; review cost")
        if any(v > .9 for v in receipt.get("gpu_peak_fractions", [])):
            warnings.append("GPU utilization above 90%")
        if receipt.get("aggregate_host_peak_bytes", 0) > .9*128*1024**3:
            warnings.append("host utilization above 90%")
        return dict(status="QUALIFIED" if all(gates.values()) else "DEFERRED", gates=gates,
                    failed=[k for k, v in gates.items() if not v], warnings=warnings, scientific_seed_result=False)
    except (KeyError, ValueError, TypeError):
        return dict(status="PENDING", gates={}, failed=[], reason="incomplete or unverified smoke receipts",
                    warnings=[], scientific_seed_result=False)

def finite_diagnostics(diagnostics):
    """Undefined effective rank is valid; nonfinite values or missing layers fail."""
    try:
        converted = {int(k): v for k, v in diagnostics.items()}
        require(set(converted) == {0, 40}, "diagnostics required at updates 0 and 40")
        for value in converted.values():
            require(value["layers"] and math.isfinite(value["dormant_layer_mean"]), "missing or nonfinite layers")
            for layer in value["layers"].values():
                require(0 <= layer["dormant_fraction"] <= 1 and isinstance(layer["zero_layer"], bool), "invalid dormant layer")
                for key in ("effective_rank", "normalized_effective_rank"):
                    require(layer[key] is None or math.isfinite(layer[key]), "nonfinite effective rank")
        return True
    except (KeyError, ValueError, TypeError):
        return False


class PeakSampler:
    """Aggregate container host usage plus device-wide GPU use, including reload."""
    def __init__(self):
        import torch
        self.torch = torch
        self.gpu = [0., 0.]
        self.host = 0
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        for i in range(2):
            free, total = self.torch.cuda.mem_get_info(i)
            peak = max(total-free, self.torch.cuda.max_memory_reserved(i))
            self.gpu[i] = max(self.gpu[i], peak/total)
        candidates = [Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")]
        current = next((p for p in candidates if p.exists()), None)
        require(current is not None, "aggregate host memory counter unavailable")
        self.host = max(self.host, int(current.read_text()))

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception as error:
                self.error = error
                return
            self.stop.wait(.05)

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
        require(self.error is None, "memory sampler failed")


def model_digest(model):
    """Stream one tensor at a time, including its name/dtype/shape."""
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(digest(dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype))).encode())
        h.update(tensor.detach().contiguous().cpu().view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def discover_scoring_gpu(score_device):
    """Discover the runtime-assigned UUID in a child, leaving parent CUDA cold."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(score_device), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("pd_probe.py")), "--gpu-identity"],
                            env=env, capture_output=True, text=True, check=True, timeout=60)
    identity = json.loads(result.stdout)
    require(identity["devices"] == [identity["uuid"]] and identity["environment"],
            "single physical scoring GPU identity required")
    return identity


def run_smoke(config):
    """Budget supervisor for the actual two-rank runtime; never called by build."""
    started = time.monotonic()
    import subprocess
    from kit.pd_probe import launch_ranks
    from kit.pd_prepare import write_frozen
    validate_price(config["quote"])
    root = Path(config["out"])
    root.mkdir(parents=True, exist_ok=True)
    require(not list(root.iterdir()), "failed fork restarts from update 0 in a clean PD tree")
    discovered = None
    try:
        config = dict(config)
        config.setdefault("score_device", "0")
        config["scoring_gpu_identity"] = discover_scoring_gpu(config["score_device"])
        discovered = seal(config["scoring_gpu_identity"])
        # Reserve a few seconds for the PENDING receipt before the container's
        # hard spending cutoff. This is never a scientific speed threshold.
        remaining = max(1, config.get("runtime_seconds", 2580)-(time.monotonic()-started)-10)
        launch_ranks(config, root, smoke=True, timeout=remaining)
        return json.loads((root/"gates.json").read_text())
    except subprocess.TimeoutExpired:
        pending = seal(dict(schema="pd-smoke-receipt-v2", spending_cutoff=True))
        write_frozen(root/"spending-cutoff.json", pending)
        return smoke_gates(pending)
    except (OSError, ValueError, subprocess.SubprocessError):
        if (root/"distributed-startup.json").exists():
            return dict(status="PENDING", reason="smoke environment distributed startup failed",
                        scientific_seed_result=False)
        failures = sorted(root.glob("failure-*.json"))
        if failures:
            receipt = json.loads(failures[0].read_text())
            result = smoke_gates(receipt)
            write_frozen(root/"qualification-failure.json", receipt)
            return result
        return dict(status="PENDING", reason="isolated smoke did not complete", scientific_seed_result=False)
    finally:
        if discovered is not None:
            write_frozen(root/"scoring-gpu.json", discovered)


PUBLIC_MODEL = "Qwen/Qwen3-8B"
STARTUP_SECONDS = 120
IDLE_SECONDS = 2
BODY_SECONDS = RESERVATION["seconds"]-STARTUP_SECONDS
INPUT_LIMIT = 18*1024**3
RETURN_LIMIT = 64*1024**2


def validate_runtime(runtime):
    import re
    require(set(runtime) == {"image_registry", "image_digest", "packages"}, "incomplete pinned runtime")
    require(re.fullmatch(r"[A-Za-z0-9./:_-]+", runtime["image_registry"] or "") is not None,
            "public prebuilt runtime registry required")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["image_digest"] or "") is not None,
            "immutable prebuilt runtime digest required")
    require(set(runtime["packages"]) == {"torch", "transformers", "accelerate", "safetensors"},
            "runtime package versions required")
    require(all(isinstance(v, str) and re.fullmatch(r"\d+\.\d+\.\d+(?:[+.-][0-9A-Za-z.]+)?", v)
                for v in runtime["packages"].values()), "unfrozen runtime versions")
    return runtime


def acquire_public_cache(cache):
    """Explicit local download only. Never authenticate, upload or launch."""
    import functools
    import os
    from huggingface_hub import hf_hub_download, snapshot_download
    from kit.pd_prepare import REVISION, SOURCES, fetch_provenance, fetch_source
    from kit.pd_probe import BASE_REVISION
    cache = Path(cache).resolve()
    repository = Path(__file__).resolve().parents[1]
    require(cache != repository and repository not in cache.parents, "public cache must be outside git tree")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    snapshot_download(repo_id=PUBLIC_MODEL, revision=BASE_REVISION, token=False, local_dir=cache/"model",
                      allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json", "chat_template.jinja"])
    fetch_provenance(cache/"trec", downloader=functools.partial(hf_hub_download, token=False))
    fetch_source("train", cache/"trec"/SOURCES["train"]["name"])
    # Test data is deliberately never acquired by this smoke preparation path.
    return dict(model_revision=BASE_REVISION, dataset_revision=REVISION, test_downloaded=False)


def assemble_bundle(cache, preparation_file, runtime_file, destination):
    """Freeze an approved preparation and public bytes; no invented audit decisions."""
    import os
    from kit.pd_contract import validate_checkpoint
    from kit.pd_prepare import REVISION, SOURCES, pinned_provenance, verify_source, write_frozen
    from kit.pd_probe import BASE_REVISION, code_identity, tokenizer_identity
    cache, destination = Path(cache).resolve(), Path(destination).resolve()
    repository = Path(__file__).resolve().parents[1]
    require(destination != repository and repository not in destination.parents, "bundle must be outside git tree")
    require(not destination.exists(), "frozen bundle destination must be new")
    preparation = json.loads(Path(preparation_file).read_text())
    verify_seal(preparation)
    require(preparation["schema"] == "pd-preparation-v2" and preparation["source_revision"] == REVISION,
            "approved final-v2 preparation required")
    verify_seal(preparation["exclusion"])
    require(all(len(preparation["exclusion"].get(k, "")) == 64 for k in
                ("inventory_sha256", "custodian_sha256", "topic_review_sha256")), "approved exclusion evidence missing")
    require(preparation["code_sha256"] == code_identity(), "preparation code changed; rebuild before freezing")
    source_hashes = {k: v["sha256"] for k, v in SOURCES.items()}
    require(preparation["source_hashes"] == source_hashes, "source hashes changed")
    provenance = pinned_provenance(cache/"trec"/"trec.py", cache/"trec"/"dataset_infos.json")
    require(preparation["source_provenance"] == provenance, "preparation provenance mismatch")
    train = verify_source((cache/"trec"/SOURCES["train"]["name"]).read_bytes(), "train")
    selected = {r["id"]: r for r in train}
    inputs = validate_inputs(smoke_inputs(preparation))
    require(all(row == selected[row["id"]] for row in inputs["rows"]), "frozen rows differ from pinned train bytes")
    require(preparation["tokenizer_sha256"] == tokenizer_identity(cache/"model"), "tokenizer changed")
    runtime = validate_runtime(json.loads(Path(runtime_file).read_text()))
    model_index = json.loads((cache/"model"/"model.safetensors.index.json").read_text())
    shard_names = set(model_index["weight_map"].values())
    require(shard_names and all(Path(name).name == name and name.endswith(".safetensors") for name in shard_names),
            "nonportable model tensor index")
    require(shard_names == {p.name for p in (cache/"model").glob("*.safetensors")}, "model shard inventory mismatch")
    destination.mkdir(parents=True)
    def copy_file(source, target, *, hardlink=False):
        target.parent.mkdir(parents=True, exist_ok=True)
        # Hard links avoid a second 16.4-GB local copy on the same scratch filesystem.
        import shutil
        if hardlink:
            try:
                os.link(source, target)
                return
            except OSError:
                pass
        shutil.copyfile(source, target)
    for source in (cache/"model").iterdir():
        if source.is_file() and (source.name in shard_names or source.suffix == ".json"
                                 or source.name in ("merges.txt", "vocab.json", "chat_template.jinja")):
            copy_file(source, destination/"model"/source.name, hardlink=True)
    write_frozen(destination/"inputs.json", inputs)
    write_frozen(destination/"runtime.json", runtime)
    write_frozen(destination/"source-provenance.json", provenance)
    files = {name: sha256_file(destination/"model"/name) for name in sorted(shard_names)}
    receipt = seal(dict(schema="pd-public-base-export-v1", repository=PUBLIC_MODEL, revision=BASE_REVISION,
                        tensor_files=files, weights_sha256=digest(files),
                        tokenizer_sha256=preparation["tokenizer_sha256"]))
    write_frozen(destination/"base-receipt.json", receipt)
    checkpoint = seal(dict(checkpoint_id=PUBLIC_MODEL+"@seed101", start="base", seed=101, step=0,
                           role="chain_start", recipe="base", parent_id=None, base_revision=BASE_REVISION,
                           tokenizer_sha256=preparation["tokenizer_sha256"],
                           send_binding_sha256=digest(dict(scope="pd-smoke-only", preparation=preparation["identity_sha256"])),
                           export_receipt_sha256=sha256_file(destination/"base-receipt.json"),
                           tensor_files=files, weights_sha256=digest(files)))
    validate_checkpoint(checkpoint)
    for name in ("pd_contract.py", "pd_prepare.py", "pd_probe.py", "pd_read.py", "pd_smoke_modal.py"):
        copy_file(Path(__file__).with_name(name), destination/"code"/"kit"/name)
    copy_file(Path(__file__).with_name("resources")/"pd"/"taxonomy.json",
              destination/"code"/"kit"/"resources"/"pd"/"taxonomy.json")
    (destination/"code"/"kit"/"__init__.py").write_text("")
    inventory = {p.relative_to(destination).as_posix(): dict(bytes=p.stat().st_size, sha256=sha256_file(p))
                 for p in sorted(destination.rglob("*")) if p.is_file()}
    require(sum(v["bytes"] for v in inventory.values()) <= INPUT_LIMIT, "smoke bundle exceeds priced input allowance")
    manifest = seal(dict(schema="pd-smoke-bundle-v1", scope="smoke-only-no-scientific-seed-result",
                         model=dict(repository=PUBLIC_MODEL, revision=BASE_REVISION),
                         dataset=dict(repository="CogComp/trec", revision=REVISION, splits=["train"]),
                         preparation_sha256=preparation["identity_sha256"], code_sha256=code_identity(),
                         checkpoint=checkpoint, files=inventory))
    write_frozen(destination/"bundle.json", manifest)
    return manifest


def plan_bundle(bundle, quote_file, spec_file):
    """Read/hash/price only. No Modal import, authentication, upload or GPU calls."""
    from kit.pd_prepare import REVISION
    from kit.pd_probe import BASE_REVISION, code_identity
    quote = json.loads(Path(quote_file).read_text())
    total = validate_price(quote)
    spec = json.loads(Path(spec_file).read_text())
    verify_seal(spec)
    require(spec["schema"] == "pd-smoke-bundle-spec-v1" and spec["reservation"] == RESERVATION
            and spec["model"] == dict(repository=PUBLIC_MODEL, revision=BASE_REVISION)
            and spec["dataset"] == dict(repository="CogComp/trec", revision=REVISION, splits=["train"])
            and spec["code_sha256"] == code_identity(), "frozen smoke specification changed")
    blockers = []
    checks = dict(price=True, specification=True, bundle=False, tensors=False, audited_inputs=False, runtime=False)
    root = Path(bundle).resolve()
    if not (root/"bundle.json").is_file():
        blockers = ["Pinned public model/tokenizer and dataset bytes are not locally cached.",
                    "Approved exclusion/topic/custodian preparation is missing.",
                    "Immutable prebuilt runtime image and package versions are missing.",
                    "Frozen bundle has not been assembled."]
    else:
        manifest = json.loads((root/"bundle.json").read_text())
        verify_seal(manifest)
        require(manifest["schema"] == "pd-smoke-bundle-v1" and
                manifest["scope"] == "smoke-only-no-scientific-seed-result", "wrong smoke bundle")
        require(manifest["model"] == spec["model"] and manifest["dataset"] == spec["dataset"] and
                manifest["code_sha256"] == spec["code_sha256"], "bundle pin/code mismatch")
        expected = manifest["files"]
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                  and p != root/"bundle.json"}
        require(actual == set(expected) and all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in expected),
                "bundle file inventory changed")
        require(sum(v["bytes"] for v in expected.values()) <= INPUT_LIMIT, "unpriced bundle size")
        for name, value in expected.items():
            require((root/name).stat().st_size == value["bytes"] and sha256_file(root/name) == value["sha256"],
                    "bundle bytes changed")
        code_names = ("pd_contract.py", "pd_prepare.py", "pd_probe.py", "pd_read.py", "pd_smoke_modal.py")
        snapshot = {name: expected["code/kit/"+name]["sha256"] for name in code_names}
        snapshot["taxonomy"] = digest(json.loads((root/"code"/"kit"/"resources"/"pd"/"taxonomy.json").read_text()))
        require(digest(snapshot) == manifest["code_sha256"], "bundled code differs from frozen code identity")
        inputs = validate_inputs(json.loads((root/"inputs.json").read_text()))
        require(inputs["preparation_sha256"] == manifest["preparation_sha256"] and
                inputs["code_sha256"] == manifest["code_sha256"], "audited preparation identity changed")
        require(inputs["tokenizer_sha256"] == manifest["checkpoint"]["tokenizer_sha256"], "bundle tokenizer identity mismatch")
        require(all(max(t["candidate_lengths"]) <= 1024 and len(t["input_ids"]) <= 1024
                    for t in inputs["tokenized"].values()), "bundle contains oversized sequence")
        runtime = validate_runtime(json.loads((root/"runtime.json").read_text()))
        checkpoint = manifest["checkpoint"]
        tensors = {p.name: p for p in (root/"model").glob("*.safetensors")}
        # Reuse the byte hashes just checked instead of re-reading 16.4 GB.
        require(set(tensors) == set(checkpoint["tensor_files"]) and all(
            expected["model/"+n]["sha256"] == value for n, value in checkpoint["tensor_files"].items()),
            "bound base tensor inventory mismatch")
        from kit.pd_contract import validate_checkpoint
        validate_checkpoint(checkpoint)
        require(expected["base-receipt.json"]["sha256"] == checkpoint["export_receipt_sha256"],
                "base receipt mismatch")
        require(checkpoint["start"] == "base" and checkpoint["seed"] == 101 and checkpoint["step"] == 0
                and checkpoint["base_revision"] == spec["model"]["revision"]
                and checkpoint["checkpoint_id"].endswith("@seed101"), "smoke requires pinned seed-101 base")
        index = json.loads((root/"model"/"model.safetensors.index.json").read_text())
        require(set(index["weight_map"].values()) == set(tensors), "model tensor index mismatch")
        checks = dict.fromkeys(checks, True)
    return seal(dict(schema="pd-smoke-plan-v1", status="READY" if not blockers else "BLOCKED",
                     checks=checks, blockers=blockers, quote_total_usd=round(total, 10),
                     quote_sha256=quote["identity_sha256"], specification_sha256=spec["identity_sha256"],
                     reservation=RESERVATION, uploaded=False, launched=False, credentials_read=False,
                     scientific_seed_result=False))



def partner_runtime_spec():
    """Hash the exact shared helper/spec without importing Modal or resolving an image."""
    import ast
    helper = Path(__file__).resolve().parents[1]/"scripts"/"smoke_common_modal.py"
    tree = ast.parse(helper.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("IMAGE", "SDPO_COMMIT"):
                values[node.targets[0].id] = ast.literal_eval(node.value)
    require(set(values) == {"IMAGE", "SDPO_COMMIT"} and
            any(isinstance(n, ast.FunctionDef) and n.name == "partner_image" for n in tree.body),
            "shared image helper spec unavailable")
    return seal(dict(schema="pd-partner-image-spec-v1", factory="partner_image({})",
                     helper_sha256=sha256_file(helper), base_image=values["IMAGE"],
                     sdpo_revision=values["SDPO_COMMIT"]))


def smoke_assets():
    from kit.pd_prepare import RESOURCE
    return {name: RESOURCE/name for name in
            ("smoke-public-pins.json", "smoke-rules.json", "smoke-development-rows.json")}


def validate_public_pins(pins):
    from kit.pd_contract import is_sha
    from kit.pd_prepare import REVISION, SOURCES
    from kit.pd_probe import BASE_REVISION
    verify_seal(pins)
    require(pins["schema"] == "pd-smoke-public-pins-v1" and
            pins["model"]["repository"] == PUBLIC_MODEL and pins["model"]["revision"] == BASE_REVISION and
            pins["dataset"]["repository"] == "CogComp/trec" and pins["dataset"]["revision"] == REVISION and
            pins["dataset"]["train"] == SOURCES["train"] and pins["test_download"] is False,
            "wrong pinned public smoke inputs")
    files = pins["model"]["files"]
    names = {"config.json", "generation_config.json", "model.safetensors.index.json",
             "tokenizer.json", "tokenizer_config.json", "vocab.json"} | {
             f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)}
    require(set(files) == names and all(is_sha(v) for v in files.values()), "incomplete public model byte pins")
    require(is_sha(pins["dataset"]["metadata_contract_sha256"]), "missing dataset metadata contract pin")
    return pins


def plan_smoke(quote_file, spec_file):
    """READY means the pinned fetch/SMOKE-SCOPE recipe is complete, never qualified."""
    from kit.pd_prepare import REVISION
    from kit.pd_probe import BASE_REVISION, code_identity
    require(2 <= IDLE_SECONDS <= 3600, "Modal scaledown_window must be between 2 and 3600")
    quote = json.loads(Path(quote_file).read_text())
    total = validate_price(quote)
    spec = json.loads(Path(spec_file).read_text())
    verify_seal(spec)
    require(spec["schema"] == "pd-smoke-container-plan-v2" and spec["scope"] == "SMOKE-SCOPE"
            and spec["scientific_preparation"] == "PENDING" and spec["reservation"] == RESERVATION
            and spec["model"] == dict(repository=PUBLIC_MODEL, revision=BASE_REVISION)
            and spec["dataset"] == dict(repository="CogComp/trec", revision=REVISION, splits=["train"])
            and spec["code_sha256"] == code_identity(), "frozen container smoke specification changed")
    require(spec["runtime"] == partner_runtime_spec(), "shared partner image spec changed")
    require(spec["network_policy"] == smoke_network_policy(), "smoke loopback/offline policy changed")
    bounds = quote["bounds"]
    rate = sum(quote["rates_per_second"][k]*n for k, n in (("H100", 2), ("cpu", 4), ("memory_gib", 128)))
    require(bounds["download_container_seconds"] == 600 and bounds["idle_seconds"] == IDLE_SECONDS
            and bounds["startup_seconds"] == STARTUP_SECONDS and bounds["local_weights_upload"] is False
            and bounds["training_block_network"] is False and bounds["offline_mode"] is True and
            quote["ceilings"]["startup"] >= (600+STARTUP_SECONDS+IDLE_SECONDS)*rate+1-1e-10, "download/image startup is not fully priced")
    assets = smoke_assets()
    require(spec["assets"] == {name: sha256_file(path) for name, path in assets.items()}, "smoke assets changed")
    validate_public_pins(json.loads(assets["smoke-public-pins.json"].read_text()))
    rules = json.loads(assets["smoke-rules.json"].read_text())
    require(rules["scope"] == "SMOKE-SCOPE" and rules["no_replenishment"] is True, "wrong smoke-only rules")
    require(spec["fetch"] == dict(location="image-build-container", seconds=600, token=False,
                                 verify_before_use=True, local_weights_upload=False,
                                 training_block_network=False, offline_mode=True), "unregistered fetch/network plan")
    require(spec["schedule"] == dict(seed=101, calibration=128, training=1280, updates=40, batch=32,
                                     repeat_updates=2), "smoke schedule changed")
    return seal(dict(schema="pd-smoke-plan-v2", status="READY", scope="SMOKE-SCOPE",
                     scientific_preparation="PENDING", scientific_seed_result=False,
                     checks=dict(price=True, specification=True, runtime_spec=True, public_byte_pins=True,
                                 local_fixtures=True, schedule=True, no_local_weights_upload=True,
                                 loopback_distributed=True, network_open_offline_mode=True),
                     runtime_image_id="RESOLVED_ONLY_ON_LAUNCH",
                     runtime_spec_sha256=spec["runtime"]["identity_sha256"],
                     input_validation="REQUIRED_IN_CONTAINER_BEFORE_USE",
                     topic_flags="GENERATED_FROM_PINNED_TRAIN_BEFORE_INFERENCE",
                     quote_total_usd=round(total, 10), quote_sha256=quote["identity_sha256"],
                     specification_sha256=spec["identity_sha256"], blockers=[],
                     uploaded=False, launched=False, credentials_read=False))


def verify_pinned_model(directory, pins):
    """Byte hashes are frozen before download, never learned from the live download."""
    directory = Path(directory)
    files = pins["model"]["files"]
    require({p.name for p in directory.iterdir() if p.is_file()} == set(files), "public model file inventory changed")
    for name, expected in files.items():
        require(sha256_file(directory/name) == expected, "pinned public model file hash mismatch")
    index = json.loads((directory/"model.safetensors.index.json").read_text())
    require(set(index["weight_map"].values()) == {n for n in files if n.endswith(".safetensors")},
            "public tensor index differs from pinned inventory")
    return seal(dict(schema="pd-verified-public-model-v1", revision=pins["model"]["revision"],
                     files=files, public_pins_sha256=pins["identity_sha256"], verified=True))


def container_prepare(spec, *, root=None):
    """Data-only image layer on partner_image(); no GPU, no inference, no auth."""
    import logging
    import os
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoTokenizer
    from kit.pd_prepare import (SOURCES, REVISION, taxonomy, digest, fetch_source, smoke_scope_prepare, write_frozen)
    from kit.pd_probe import code_identity, encode_row, tokenizer_identity
    require(spec["code_sha256"] == code_identity(), "image PD code changed")
    assets = smoke_assets()
    require(spec["assets"] == {n: sha256_file(p) for n, p in assets.items()}, "image smoke assets changed")
    pins = validate_public_pins(json.loads(assets["smoke-public-pins.json"].read_text()))
    root = Path(root) if root is not None else Path("/root/pd-smoke-data")
    root.mkdir(exist_ok=False)
    os.environ.update(HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
    old_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        model = root/"model"
        snapshot_download(repo_id=pins["model"]["repository"], revision=pins["model"]["revision"],
                          allow_patterns=sorted(pins["model"]["files"]), local_dir=model, token=False, max_workers=4)
        verified = verify_pinned_model(model, pins)
        metadata = Path(hf_hub_download(repo_id="CogComp/trec", repo_type="dataset", revision=REVISION,
                                       filename="dataset_infos.json", local_dir=root/"metadata", token=False))
        info = json.loads(metadata.read_text())["default"]
        def source_view(split):
            source = SOURCES[split]
            value = info["download_checksums"]["https://cogcomp.seas.upenn.edu/Data/QA/QC/"+source["name"]]
            return dict(name=source["name"], bytes=value["num_bytes"], rows=info["splits"][split]["num_examples"],
                        sha256=value["checksum"])
        metadata_view = dict(train=source_view("train"), test=source_view("test"),
                             labels=sorted(info["features"]["fine_label"]["names"]))
        require(digest(metadata_view) == pins["dataset"]["metadata_contract_sha256"],
                "pinned revision metadata contract changed")
        rows = fetch_source("train", root/SOURCES["train"]["name"])
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False, use_fast=True)
        require(tokenizer.is_fast, "frozen fast tokenizer required")
        preparation, inputs, flags = smoke_scope_prepare(rows, lambda row: encode_row(tokenizer, row),
            tokenizer_sha256=tokenizer_identity(model), code_sha256=code_identity(),
            public_pins_sha256=pins["identity_sha256"], rules=json.loads(assets["smoke-rules.json"].read_text()),
            development_rows=json.loads(assets["smoke-development-rows.json"].read_text()))
        validate_inputs(inputs)
        write_frozen(root/"smoke-preparation.json", preparation)
        write_frozen(root/"inputs.json", inputs)
        write_frozen(root/"topic-flags.json", flags)
        weights = {n: v for n, v in pins["model"]["files"].items() if n.endswith(".safetensors")}
        base_receipt = seal(dict(schema="pd-smoke-public-base-v2", scope="SMOKE-SCOPE",
                                 revision=pins["model"]["revision"], tensor_files=weights,
                                 weights_sha256=digest(weights), public_verification=verified))
        write_frozen(root/"base-receipt.json", base_receipt)
        checkpoint = seal(dict(checkpoint_id=PUBLIC_MODEL+"@seed101", start="base", seed=101, step=0,
            role="chain_start", recipe="base", parent_id=None, base_revision=pins["model"]["revision"],
            tokenizer_sha256=inputs["tokenizer_sha256"], send_binding_sha256=preparation["identity_sha256"],
            tensor_files=weights, weights_sha256=digest(weights),
            export_receipt_sha256=sha256_file(root/"base-receipt.json")))
        from kit.pd_contract import validate_checkpoint
        validate_checkpoint(checkpoint)
        write_frozen(root/"prepared.json", seal(dict(schema="pd-container-preparation-v1", scope="SMOKE-SCOPE",
            scientific_preparation="PENDING", checkpoint=checkpoint, inputs_sha256=sha256_file(root/"inputs.json"),
            topic_flags_sha256=sha256_file(root/"topic-flags.json"), specification_sha256=spec["identity_sha256"],
            public_verification=verified, metadata_raw_sha256=sha256_file(metadata),
            metadata_contract_sha256=digest(metadata_view), test_downloaded=False)))
    except Exception as error:
        # Do not propagate third-party URLs/headers or implicit credential logs.
        raise ValueError("Pinned public fetch/preparation failed: "+type(error).__name__) from None
    finally:
        logging.disable(old_level)


def smoke_network_policy():
    return dict(worker_network="OPEN_OFFLINE_MODE", block_network=False,
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", pinned_hash_verified_inputs=True,
                rendezvous="127.0.0.1", NCCL_SOCKET_IFNAME="lo", GLOO_SOCKET_IFNAME="lo",
                distributed_startup_limit_seconds=60, rendezvous_read_timeout_seconds=15, join_timeout_seconds=30,
                ray_used=False, implementation="direct-FSDP")


def reservation_cost(quote, seconds):
    """List-price reserved compute from measured wall time, excluding other costs."""
    require(math.isfinite(seconds) and seconds >= 0, "invalid measured reservation seconds")
    rates = quote["rates_per_second"]
    components = dict(gpu=2*rates["H100"]*seconds, cpu=4*rates["cpu"]*seconds,
                      memory=128*rates["memory_gib"]*seconds)
    return seal(dict(schema="pd-measured-reservation-cost-v1", currency="USD", seconds=seconds,
                     reserved_gpus=2, rates_per_second=rates, components_usd=components,
                     total_usd=sum(components.values()), gpu_hours=2*seconds/3600,
                     basis="measured worker execution at reserved list rates; not an invoice",
                     excludes=["platform startup", "image build", "storage", "egress"]))


def prepared_worker(spec, quote, runtime):
    """One network-open, offline-mode two-GPU worker with frozen local inputs."""
    started = time.monotonic()
    import importlib.metadata
    from kit.pd_probe import code_identity
    root = Path("/root/pd-smoke-data")
    require(runtime["modal_image_id"].startswith("im-") and
            runtime["pinned_spec_sha256"] == spec["runtime"]["identity_sha256"], "resolved image identity required")
    require(spec["code_sha256"] == code_identity(), "offline-mode worker code changed")
    prepared = json.loads((root/"prepared.json").read_text())
    verify_seal(prepared)
    require(prepared["specification_sha256"] == spec["identity_sha256"] and
            prepared["inputs_sha256"] == sha256_file(root/"inputs.json") and
            prepared["topic_flags_sha256"] == sha256_file(root/"topic-flags.json"), "frozen image inputs changed")
    runtime = dict(runtime, network_policy=smoke_network_policy(), packages={k: importlib.metadata.version(k) for k in
                                      ("torch", "transformers", "accelerate", "safetensors")})
    result = run_smoke(dict(inputs=str(root/"inputs.json"), base=str(root/"model"), tokenizer=str(root/"model"),
                            base_checkpoint=prepared["checkpoint"], base_receipt=str(root/"base-receipt.json"),
                            quote=quote, score_device="0", out="/pd-state/run", runtime_seconds=BODY_SECONDS,
                            scope="SMOKE-SCOPE"))
    output = Path("/pd-state/run")
    receipts = {p.relative_to(output).as_posix(): json.loads(p.read_text()) for p in output.rglob("*.json")
                if p.is_file() and p.relative_to(output).parts[0] not in ("pending-export", "scoring-export")}
    flags = json.loads((root/"topic-flags.json").read_text())
    require(len(json.dumps(receipts).encode()) <= RETURN_LIMIT, "receipt egress allowance exceeded")
    return seal(dict(schema="pd-smoke-return-v2", scope="SMOKE-SCOPE", scientific_preparation="PENDING",
                     scientific_seed_result=False, gates=result, receipts=receipts, topic_flags=flags,
                     prepared=prepared, runtime=runtime, measured_cost=reservation_cost(quote, time.monotonic()-started),
                     specification_sha256=spec["identity_sha256"],
                     quote_sha256=quote["identity_sha256"]))


def modal_worker_options(image, state_volume):
    """Unhydrated worker specification, also constructed by the offline SDK test.

    Modal 1.5.3 checks only >0 locally; the service requires 2..3600.
    Validate that stricter bound before opening any remote application.
    """
    require(2 <= IDLE_SECONDS <= 3600, "Modal scaledown_window must be between 2 and 3600")
    return dict(image=image, gpu="H100:2", cpu=(4, 4), memory=(131072, 131072),
                ephemeral_disk=524288, timeout=BODY_SECONDS, startup_timeout=STARTUP_SECONDS,
                retries=0, max_containers=1, min_containers=0, buffer_containers=0,
                scaledown_window=IDLE_SECONDS, single_use_containers=True,
                env=dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                         MASTER_ADDR="127.0.0.1", NCCL_SOCKET_IFNAME="lo", GLOO_SOCKET_IFNAME="lo",
                         PYTHONPATH="/root:/root/SDPO",
                         CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1"),
                volumes={"/pd-state": state_volume})


def build_modal_app(config):
    """Exactly the shared runtime factory; only data/code layers are added."""
    validate_price(config.get("quote", {}))
    plan_smoke(config["quote_file"], config["spec_file"])
    import modal
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    sys.path.insert(0, str(scripts))
    from smoke_common_modal import partner_image
    spec, quote = config["spec"], config["quote"]
    app = modal.App("pd-smoke")
    image = (partner_image({})
             .env(dict(PYTHONPATH="/root:/root/SDPO", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="1"))
             .run_function(container_prepare, kwargs=dict(spec=spec), cpu=1, memory=16384, timeout=600))
    worker = app.function(**modal_worker_options(image, config["state_volume"]))(prepared_worker)
    return app, worker, image


def launch_smoke(quote_file, spec_file, receipt_file, flags_file):
    """Called only by the manager's explicit Modal command, never by dry plan."""
    from kit.pd_prepare import write_frozen
    plan = plan_smoke(quote_file, spec_file)
    require(plan["status"] == "READY" and not Path(receipt_file).exists(), "fresh READY launch/receipt required")
    if Path(flags_file).exists():
        require(json.loads(Path(flags_file).read_text())["status"] == "PENDING_CONTAINER_FETCH",
                "fresh pending topic-flag destination required before launch")
    spec, quote = json.loads(Path(spec_file).read_text()), json.loads(Path(quote_file).read_text())
    import modal
    returned = None
    runtime = dict(modal_image_id=None, pinned_spec_sha256=spec["runtime"]["identity_sha256"],
                   network_policy=smoke_network_policy())
    try:
        with modal.Volume.ephemeral() as state:
            app, worker, image = build_modal_app(dict(spec=spec, quote=quote, quote_file=quote_file,
                                                     spec_file=spec_file, state_volume=state))
            with app.run():
                runtime["modal_image_id"] = image.object_id
                require(runtime["modal_image_id"].startswith("im-"), "Modal image did not resolve")
                returned = worker.remote(spec, quote, runtime)
    except Exception as error:
        returned = seal(dict(schema="pd-smoke-return-v2", scope="SMOKE-SCOPE", scientific_preparation="PENDING",
                             scientific_seed_result=False, gates=dict(status="PENDING",
                                 reason="image preparation or worker did not return complete evidence"),
                             receipts={}, runtime=runtime, error_type=type(error).__name__,
                             specification_sha256=spec["identity_sha256"], quote_sha256=quote["identity_sha256"]))
    verify_seal(returned)
    require(returned["specification_sha256"] == spec["identity_sha256"] and
            returned["quote_sha256"] == quote["identity_sha256"] and
            all(returned["runtime"][k] == value for k, value in runtime.items()), "returned launch binding changed")
    write_frozen(receipt_file, returned)
    if "topic_flags" in returned:
        verify_seal(returned["topic_flags"])
        target = Path(flags_file)
        if target.exists():
            require(json.loads(target.read_text())["status"] == "PENDING_CONTAINER_FETCH",
                    "refusing to overwrite completed topic flags")
        target.write_text(json.dumps(returned["topic_flags"], sort_keys=True, indent=2)+"\n")
    return returned["gates"]



def main():
    parser = argparse.ArgumentParser(description="Offline smoke receipt validation; never launches Modal")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from kit.pd_prepare import write_frozen
    result = smoke_gates(json.loads(Path(args.receipt).read_text()))
    write_frozen(args.out, result)
    return 0 if result["status"] == "QUALIFIED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
