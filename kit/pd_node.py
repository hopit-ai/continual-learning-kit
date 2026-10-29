"""Node supervisor outside the qualified PD core's code_identity() hash set.

Run once inside an eight-GPU allocation. GPU discovery happens only in a child;
the supervisor never initializes CUDA. No Modal, Ray, model or data acquisition.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kit import pd_probe as probe
from kit import pd_smoke_modal as smoke
from kit import pd_scientific as scientific
from kit import pd_materialize as materialize
from kit.pd_contract import SEEDS, digest, is_sha, require, seal, sha256_file
from kit.pd_prepare import write_frozen


def _gpu_uuid(value):
    """CUDA accepts GPU-UUID selectors; Torch may report the UUID without GPU-."""
    require(isinstance(value, str), "GPU UUID must be a string")
    raw = value.removeprefix("GPU-")
    try:
        normalized = str(uuid.UUID(raw))
    except (ValueError, AttributeError):
        raise ValueError("whole-GPU UUID required; MIG allocations are not supported") from None
    return "GPU-"+normalized


def allocation_tokens(visible_devices):
    require(isinstance(visible_devices, str) and visible_devices.strip(),
            "CUDA_VISIBLE_DEVICES must explicitly describe the eight-GPU allocation")
    tokens = [token.strip() for token in visible_devices.split(",")]
    normalized = [str(int(token)) if token.isdecimal() else _gpu_uuid(token) for token in tokens]
    require(len(normalized) == 8 and len(set(normalized)) == 8,
            "PD node requires exactly eight distinct allocated GPUs")
    return normalized


def bind_plan(plan, score_visible_index=0):
    """Translate a visible slot through the allocation, then bind a stable UUID."""
    require(plan.get("seed") in SEEDS, "PD node requires one registered seed 101..103")
    scientific.require_plan(plan)
    preparation_pin = materialize.require_plan_pin(plan)
    qualification = plan.get("smoke_receipt", {})
    require(smoke.smoke_gates(qualification)["status"] == "QUALIFIED" and
            qualification.get("code_sha256") == probe.code_identity(),
            "qualified smoke receipt must match the unchanged PD core")
    tokens = allocation_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"))
    require(type(score_visible_index) is int and 0 <= score_visible_index < len(tokens),
            "score-visible-index must select a visible slot 0..7")
    # discover_scoring_gpu rewrites CUDA_VISIBLE_DEVICES in its child. Passing
    # the inherited allocation token, rather than the visible slot, preserves
    # Slurm subset/reordered allocations. Its environment digest is core-owned.
    selected = tokens[score_visible_index]
    identity = smoke.discover_scoring_gpu(selected)
    require(identity.get("devices") == [identity.get("uuid")] and
            is_sha(identity.get("environment")), "invalid discovered scoring GPU identity")
    selector = _gpu_uuid(identity["uuid"])
    if selected.startswith("GPU-"):
        require(selector == selected, "discovery returned a different allocated GPU UUID")
    binding = seal(dict(schema="pd-node-runtime-binding-v1", seed=plan["seed"],
                        input_plan_sha256=digest(plan), code_sha256=probe.code_identity(),
                        wrapper_sha256=sha256_file(__file__),
                        preparation_pin_sha256=preparation_pin["identity_sha256"],
                        allocation_sha256=digest(tokens), score_visible_index=score_visible_index,
                        score_device=selector, scoring_gpu_identity=identity))
    return seal({**plan, "score_device": selector, "scoring_gpu_identity": identity,
                 "node_binding": binding})


class PreparationError(ValueError):
    """A pre-launch gate failed; no distributed launch was attempted."""


def _prepare_launch(plan_file, bound_plan_file, output, *, score_visible_index=0, receipt_file=None):
    """Check gates and freeze the runtime plan outside the clean PD tree."""
    source, bound, state = (Path(p).resolve() for p in (plan_file, bound_plan_file, output))
    receipt = Path(receipt_file).resolve() if receipt_file else bound.with_name(bound.name+".receipt.json")
    require(source != bound and receipt not in (source, bound), "plan and receipt destinations must be distinct")
    require(not bound.exists() and not receipt.exists(), "fresh bound plan and node receipt required")
    require(all(p != state and state not in p.parents for p in (source, bound, receipt)),
            "plans and supervisor receipts must stay outside the clean PD state tree")
    require(not state.exists() or (state.is_dir() and not any(state.iterdir())),
            "a failed seed restarts from update 0 in a clean PD tree")
    plan = json.loads(source.read_text())
    # Do not write operational plans into any original source checkpoint tree.
    for export in plan.get("exports", {}).values():
        original = Path(export["directory"]).resolve()
        require(all(p != original and original not in p.parents for p in (bound, receipt)),
                "bound plan and receipt must stay outside source exports")
    bound_plan = bind_plan(plan, score_visible_index)
    write_frozen(bound, bound_plan)
    result = dict(schema="pd-node-launch-receipt-v1", seed=bound_plan["seed"],
                  bound_plan_sha256=sha256_file(bound),
                  binding_sha256=bound_plan["node_binding"]["identity_sha256"],
                  code_sha256=probe.code_identity(), wrapper_sha256=sha256_file(__file__),
                  scientific_result=False)
    return bound_plan, bound, state, receipt, result


def run_node(plan_file, bound_plan_file, output, *, score_visible_index=0, receipt_file=None):
    """Only preparation failures are deferred; launch failures retain their type."""
    try:
        bound_plan, bound, state, receipt, result = _prepare_launch(
            plan_file, bound_plan_file, output,
            score_visible_index=score_visible_index, receipt_file=receipt_file)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise PreparationError(str(error)) from error
    try:
        # A real filename lets every isolated scorer load this same bound plan.
        probe.launch_ranks(bound, state, smoke=False)
        if "scientific" in bound_plan:
            scientific.retain_verified_evidence(state, bound_plan)
    except BaseException as error:
        write_frozen(receipt, seal(dict(result, status="PENDING", error_type=type(error).__name__,
            reason="node supervisor observed a failed or interrupted distributed launch",
            rank_failure_classification="UNAVAILABLE_IN_QUALIFIED_CORE")))
        raise
    write_frozen(receipt, seal(dict(result, status="PROCESS_COMPLETED")))
    return bound_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--bound-plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-visible-index", type=int, default=0)
    parser.add_argument("--receipt", help="optional supervisor receipt outside PD state")
    args = parser.parse_args()
    try:
        bound = run_node(args.plan, args.bound_plan, args.out,
                         score_visible_index=args.score_visible_index, receipt_file=args.receipt)
    except PreparationError as error:
        print(json.dumps(dict(status="DEFERRED", error_type=type(error.__cause__).__name__,
                              reason="PD scientific preparation/approval, node verification or source binding is missing or invalid")))
        return 3
    except Exception as error:
        print(json.dumps(dict(status="FAILED", error_type=type(error).__name__,
                              reason="PD node runtime failed after preparation")))
        return 1
    print(json.dumps(dict(status="PROCESS_COMPLETED", seed=bound["seed"],
                          binding_sha256=bound["node_binding"]["identity_sha256"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
