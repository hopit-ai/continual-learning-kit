"""PD v2 identity, score and streamed-export evidence contract.

The final registration excludes PD forks from SP's checkpoint inventory.

Manifests contain portable artifact IDs and content digests, never machine-local
locations. A resolver supplied by the caller maps artifact IDs to local files.
Every digest is SHA256 of canonical JSON, except byte/tensor file digests.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SCHEMA = "pd-checkpoint-score-manifest-v2"
SEEDS = (101, 102, 103)
STARTS = ("base", "stage1", "stage2")
UPDATES = (0, 5, 10, 20, 30, 40)
TRAINING_SPEC = dict(optimizer="AdamW", betas=[.9, .95], eps=1e-8, weight_decay=0.,
                     gradient_clip=1., lr=1e-5, precision="bf16/fp32-masters-moments",
                     loss="equal-example-mean-response-token-CE-with-EOS", replay=False,
                     teacher=False, ema=False, input_cap=1024, batch=32, updates=40)
CHECKPOINT_FIELDS = (
    "checkpoint_id", "start", "seed", "step", "recipe", "parent_id",
    "base_revision", "tokenizer_sha256", "send_binding_sha256",
    "export_receipt_sha256", "tensor_files", "weights_sha256",
)
SCORE_FIELDS = (
    "score_id", "checkpoint_id", "seed", "update", "panel_sha256",
    "prompt_sha256", "scorer_sha256", "decoding_sha256", "tokenizer_sha256",
    "suffix_sha256", "gpu_identity", "predictions_sha256", "exposure_sha256",
    "summed_scores_sha256", "covariates_sha256", "scoring_rule",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    # JSON object keys are strings on disk. Normalize before sorting so a curve
    # keyed by integer updates has the same digest after an export/reload.
    def unique_pairs(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), "duplicate canonical JSON keys")
        return result
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False), object_pairs_hook=unique_pairs)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(filename):
    h = hashlib.sha256()
    with open(filename, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def seal(value):
    """Bind the entire payload; receipt edits invalidate identity_sha256."""
    value = dict(value)
    value.pop("identity_sha256", None)
    return {**value, "identity_sha256": digest(value)}


def verify_seal(value):
    payload = {k: v for k, v in value.items() if k != "identity_sha256"}
    require(value.get("identity_sha256") == digest(payload), "identity digest mismatch")


def validate_checkpoint(checkpoint):
    verify_seal(checkpoint)
    require(all(k in checkpoint for k in CHECKPOINT_FIELDS), "incomplete checkpoint identity")
    require(checkpoint["seed"] in SEEDS, "unregistered seed")
    require(checkpoint["start"] in STARTS, "unregistered start")
    probe = checkpoint.get("role", "chain_start") == "probe"
    require(checkpoint.get("role", "chain_start") in ("probe", "chain_start"), "unknown checkpoint role")
    if probe:
        require(checkpoint.get("export_dtype") == "bfloat16", "probe export dtype must be bfloat16")
        require(checkpoint["step"] in UPDATES[1:] and checkpoint["recipe"] == "label-CE", "wrong probe export")
    else:
        require(checkpoint["step"] == (0 if checkpoint["start"] == "base" else 40), "wrong chain step")
        require(checkpoint["recipe"] == ("base" if checkpoint["start"] == "base" else "plain-D"),
                "replay or alternate arm cannot substitute for plain-D")
    require(bool(checkpoint["base_revision"]), "missing pinned base revision")
    require(checkpoint["parent_id"] is None if checkpoint["start"] == "base" and not probe
            else bool(checkpoint["parent_id"]), "missing parent link")
    files = checkpoint["tensor_files"]
    require(bool(files) and all(Path(k).name == k and is_sha(v) for k, v in files.items()),
            "tensor inventory must contain portable names and full hashes")
    require(checkpoint["weights_sha256"] == digest(files), "tensor inventory digest mismatch")
    for key in ("tokenizer_sha256", "send_binding_sha256", "export_receipt_sha256"):
        require(is_sha(checkpoint[key]), "missing checkpoint digest: " + key)
    return checkpoint


def verify_live_export(checkpoint, tensor_resolver, receipt_file, *, directory=None):
    """Verify complete live tensor inventory, including added/removed shards.

    tensor_resolver is a mapping from every live tensor basename to its local
    filename. Include all model tensor files, not a subset supplied by a receipt.
    """
    validate_checkpoint(checkpoint)
    if directory is not None:
        live = {p.name: p for p in Path(directory).iterdir() if p.suffix in (".safetensors", ".bin")}
        require(set(live) == set(tensor_resolver) and all(Path(tensor_resolver[k]).resolve() == live[k].resolve() for k in live),
                "export tensor inventory changed")
    require(set(tensor_resolver) == set(checkpoint["tensor_files"]), "export tensor inventory changed")
    for name, filename in tensor_resolver.items():
        require(sha256_file(filename) == checkpoint["tensor_files"][name], "live tensor hash mismatch")
    require(sha256_file(receipt_file) == checkpoint["export_receipt_sha256"], "export receipt changed")
    return checkpoint


def validate_score(score):
    verify_seal(score)
    require(all(k in score for k in SCORE_FIELDS), "incomplete score identity")
    require(score["seed"] in SEEDS and score["update"] in UPDATES, "unregistered score cell")
    require(score["scoring_rule"] == "sum-label-plus-eos", "wrong primary label scoring rule")
    require(bool(score["gpu_identity"].get("uuid")) and bool(score["gpu_identity"].get("environment")),
            "physical GPU UUID and environment required")
    for key in SCORE_FIELDS:
        if key.endswith("_sha256"):
            require(is_sha(score[key]), "invalid score digest: " + key)
    return score


def verify_stream_receipt(checkpoint, verification):
    """Retained live verification after an owned probe export has been deleted."""
    validate_checkpoint(checkpoint)
    verify_seal(verification)
    require(checkpoint.get("role") == "probe" and verification.get("schema") == "pd-stream-verification-v2",
            "stream receipt cannot replace verification of a source checkpoint")
    require(checkpoint.get("stream_verification_sha256") == verification["identity_sha256"], "stream receipt changed")
    for field in ("checkpoint_id", "tensor_files", "weights_sha256", "export_receipt_sha256"):
        require(checkpoint[field] == verification[field], "stream export verification mismatch")
    require(verification["disposition"] == "verified-scored-deleted" and is_sha(verification["score_sha256"]),
            "stream export has not been verified, scored and deleted")
    return verification


def make_manifest(checkpoints, scores, *, preparation_sha256, code_sha256):
    doc = seal(dict(schema=SCHEMA, checkpoints=sorted(checkpoints, key=lambda c: c["checkpoint_id"]),
                    scores=sorted(scores, key=lambda s: s["score_id"]),
                    preparation_sha256=preparation_sha256, code_sha256=code_sha256))
    return validate_manifest(doc)


def validate_manifest(doc):
    verify_seal(doc)
    require(doc.get("schema") == SCHEMA, "wrong PD manifest schema")
    require(is_sha(doc.get("preparation_sha256")) and is_sha(doc.get("code_sha256")), "missing manifest hashes")
    checkpoints = {c["checkpoint_id"]: validate_checkpoint(c) for c in doc["checkpoints"]}
    require(len(checkpoints) == len(doc["checkpoints"]), "duplicate checkpoint ID")
    require(len({s["score_id"] for s in doc["scores"]}) == len(doc["scores"]), "duplicate score ID")
    for score in doc["scores"]:
        validate_score(score)
        require(score["checkpoint_id"] in checkpoints, "score references missing checkpoint")
        checkpoint = checkpoints[score["checkpoint_id"]]
        require(score["seed"] == checkpoint["seed"], "score seed mismatch")
        require(score["tokenizer_sha256"] == checkpoint["tokenizer_sha256"], "score tokenizer mismatch")
        require(score["update"] == (checkpoint["step"] if checkpoint.get("role") == "probe" else 0),
                "score does not identify its exported curve checkpoint")
    return doc
