"""PD's common-error acquisition estimand and exact three-seed reader."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kit.pd_contract import (SEEDS, STARTS, UPDATES, TRAINING_SPEC, digest, require, seal,
                             validate_manifest, verify_live_export, verify_seal, verify_stream_receipt)

T_TWO_SIDED = 4.302653


def collect_seed_evidence(records):
    """Merge exactly three independently collected seed receipts, without filling cells."""
    from kit.pd_contract import make_manifest
    require(len(records) == 3, "INCOMPLETE: three seed receipts required")
    for record in records:
        validate_manifest(record["manifest"])
    manifests = [r["manifest"] for r in records]
    require(len({digest(r["send_binding"]) for r in records}) == 1, "UNVERIFIED: send binding mismatch")
    require(len({m["preparation_sha256"] for m in manifests}) == 1 and
            len({m["code_sha256"] for m in manifests}) == 1, "UNVERIFIED: preparation/code mismatch")
    cohorts = {}
    for record in records:
        require(not set(cohorts) & set(record["cohorts"]), "duplicate seed receipt")
        cohorts.update(record["cohorts"])
    require(set(cohorts) == {str(s) for s in SEEDS}, "INCOMPLETE: missing registered seed")
    for key in ("smoke_receipt", "return_verification"):
        require(len({digest(r[key]) for r in records}) == 1, "UNVERIFIED: qualification/return mismatch")
    return dict(manifest=make_manifest([c for m in manifests for c in m["checkpoints"]],
                                      [s for m in manifests for s in m["scores"]],
                                      preparation_sha256=manifests[0]["preparation_sha256"],
                                      code_sha256=manifests[0]["code_sha256"]),
                runs=[r for record in records for r in record["runs"]], cohorts=cohorts,
                send_binding=records[0]["send_binding"], smoke_receipt=records[0]["smoke_receipt"],
                return_verification=records[0]["return_verification"],
                stream_verifications={k: v for r in records for k, v in r["stream_verifications"].items()})


def freeze_common_errors(initial_predictions, *, minimum=64):
    require(set(initial_predictions) == set(STARTS), "missing start for common-error cohort")
    ids = set(initial_predictions["base"])
    require(ids and all(set(values) == ids for values in initial_predictions.values()), "initial panel coverage mismatch")
    require(all(isinstance(v, bool) for values in initial_predictions.values() for v in values.values()),
            "correctness must be Boolean")
    errors = sorted(i for i in ids if not any(initial_predictions[s][i] for s in STARTS))
    require(len(errors) >= minimum, "PENDING: fewer than 64 common errors")
    return seal(dict(schema="pd-common-errors-v1", ids=errors, panel_ids=sorted(ids),
                     initial_predictions_sha256=digest(initial_predictions), frozen_before_updates=True))


def _curve(curve):
    converted = {int(k): v for k, v in curve.items()}
    require(len(converted) == len(curve) and set(converted) == set(UPDATES), "INCOMPLETE: missing or duplicate curve cells")
    return converted


def auc(accuracies):
    values = _curve(accuracies)
    require(all(math.isfinite(v) for v in values.values()), "nonfinite curve")
    return sum((b-a)*(values[a]+values[b])/2 for a, b in zip(UPDATES, UPDATES[1:]))/40


def corrected_auc(prediction_curve, cohort):
    """Common-unsolved cohort U in percentage points; never a start correction
    on independently selected subsets, nor a switch to final-score differences.
    """
    verify_seal(cohort)
    require(cohort["frozen_before_updates"] is True and cohort["schema"] == "pd-common-errors-v1", "unfrozen cohort")
    curve = _curve(prediction_curve)
    ids = cohort["ids"]
    require(len(ids) >= 64 and len(set(ids)) == len(ids), "insufficient frozen common errors")
    for predictions in curve.values():
        require(set(predictions) == set(cohort["panel_ids"]), "curve panel coverage mismatch")
        require(all(isinstance(v, bool) for v in predictions.values()), "invalid prediction flags")
    require(not any(curve[0][i] for i in ids), "confounded start: common errors not initially unsolved")
    accuracies = {t: 100*sum(predictions[i] for i in ids)/len(ids) for t, predictions in curve.items()}
    return dict(auc=auc(accuracies), accuracy=accuracies, final=accuracies[40], initial=accuracies[0])


def full_panel_metrics(prediction_curve):
    curve = _curve(prediction_curve)
    ids = set(curve[0])
    require(ids and all(set(p) == ids for p in curve.values()), "incomplete full panel")
    accuracy = {t: 100*sum(predictions.values())/len(ids) for t, predictions in curve.items()}
    initial = accuracy[0]
    headroom = None if initial == 100 else {t: (a-initial)/(100-initial) for t, a in accuracy.items()}
    return dict(raw_auc=auc(accuracy), starting_accuracy=initial, gain_auc=auc(accuracy)-initial,
                final_accuracy=accuracy[40], headroom_normalized_gain=headroom)


def paired_interval(values):
    require(set(values) == set(SEEDS), "exactly all three registered seeds required")
    xs = [values[s] for s in SEEDS]
    require(all(math.isfinite(x) for x in xs), "nonfinite seed difference")
    mean = statistics.mean(xs)
    radius = T_TWO_SIDED*statistics.stdev(xs)/math.sqrt(3)
    return dict(n=3, mean=mean, lower=mean-radius, upper=mean+radius,
                negative_seeds=sum(x < 0 for x in xs), t=T_TWO_SIDED)


def _read(evidence, export_resolvers, trusted):
    validate_manifest(evidence["manifest"])
    manifest = evidence["manifest"]
    require(trusted is not None, "UNVERIFIED: independent frozen digest anchors required")
    verify_seal(evidence["send_binding"])
    require(evidence["send_binding"]["identity_sha256"] == trusted["send_binding_sha256"] and
            manifest["preparation_sha256"] == trusted["preparation_sha256"] and
            manifest["code_sha256"] == trusted["code_sha256"], "UNVERIFIED: frozen anchor mismatch")
    from kit.pd_smoke_modal import smoke_gates
    qualification = evidence.get("smoke_receipt")
    require(qualification is not None, "INCOMPLETE: smoke qualification missing")
    require(qualification["identity_sha256"] == trusted["smoke_receipt_sha256"] and
            qualification["code_sha256"] == manifest["code_sha256"] and
            smoke_gates(qualification)["status"] == "QUALIFIED", "UNVERIFIED: smoke qualification failed")
    returned = evidence.get("return_verification")
    require(returned is not None, "INCOMPLETE: rebuilt-pool verification missing")
    verify_seal(returned)
    require(returned["status"] == "VERIFIED" and returned["preparation_sha256"] == manifest["preparation_sha256"] and
            returned["candidate_supersets_sha256"] == trusted["candidate_supersets_sha256"], "UNVERIFIED: returned pool mismatch")
    require(set(evidence["cohorts"]) == {str(s) for s in SEEDS}, "INCOMPLETE: missing seed cohort")
    runs = {(r["seed"], r["start"]): r for r in evidence["runs"]}
    require(len(runs) == len(evidence["runs"]), "duplicate run cells")
    require(set(runs) == {(s, j) for s in SEEDS for j in STARTS}, "INCOMPLETE: nine runs required")
    checkpoints = {c["checkpoint_id"]: c for c in manifest["checkpoints"]}
    require(len(checkpoints) == 54, "INCOMPLETE: nine original and 45 streamed checkpoints required")
    from kit.pd_probe import bind_chain
    for seed in SEEDS:
        chain = {c["start"]: c for c in checkpoints.values() if c["seed"] == seed and c.get("role", "chain_start") == "chain_start"}
        require(len([c for c in checkpoints.values() if c["seed"] == seed and c.get("role", "chain_start") == "chain_start"]) == 3,
                "INCOMPLETE: three original starts per seed required")
        bind_chain(chain, evidence["send_binding"])
    for checkpoint in checkpoints.values():
        if checkpoint.get("role") == "probe":
            parent = checkpoints.get(checkpoint["parent_id"])
            require(parent is not None and parent.get("role", "chain_start") == "chain_start" and
                    all(checkpoint[k] == parent[k] for k in ("seed", "start", "base_revision", "tokenizer_sha256", "send_binding_sha256")),
                    "UNVERIFIED: probe derived from wrong start")
    # The offline reader inspects actual files; a caller's Boolean 'verified'
    # or a receipt that has not been compared with live weights is insufficient.
    source_ids = {k for k, c in checkpoints.items() if c.get("role", "chain_start") == "chain_start"}
    require(export_resolvers is not None and set(export_resolvers) == source_ids, "UNVERIFIED: original live exports required; probe exports must be streamed")
    for checkpoint_id, checkpoint in checkpoints.items():
        if checkpoint.get("role") == "probe":
            require(checkpoint_id in evidence.get("stream_verifications", {}), "INCOMPLETE: streamed export proof missing")
            verify_stream_receipt(checkpoint, evidence["stream_verifications"][checkpoint_id])
            continue
        resolver = export_resolvers[checkpoint_id]
        require(resolver.get("directory") is not None, "UNVERIFIED: complete live export directory required")
        verify_live_export(checkpoint, resolver["tensors"], resolver["receipt"], directory=resolver["directory"])
    scores = {s["score_id"]: s for s in manifest["scores"]}
    require(len(scores) == 54, "INCOMPLETE: all 54 score cells required")
    require({s["checkpoint_id"] for s in scores.values()} == set(checkpoints), "INCOMPLETE: unscored checkpoint exports")
    primary, descriptive, summaries = {}, {}, {}
    for seed in SEEDS:
        cohort = evidence["cohorts"][str(seed)]
        initial, seed_scores, exposures, result = {}, [], [], {}
        for start in STARTS:
            run = runs[seed, start]
            verify_seal(run)
            require(run.get("training_spec") == TRAINING_SPEC, "UNVERIFIED: training recipe changed")
            require(run.get("training_world_size") == 8, "UNVERIFIED: node requires eight training ranks")
            require(run.get("preparation_sha256") == manifest["preparation_sha256"] and
                    run.get("cohort_sha256") == cohort["identity_sha256"], "UNVERIFIED: run inputs/cohort changed")
            require(run.get("checkpoint_sha256") == next(c["identity_sha256"] for c in checkpoints.values()
                    if c["seed"] == seed and c["start"] == start and c.get("role", "chain_start") == "chain_start"),
                    "UNVERIFIED: run start changed")
            require(len(run["losses"]) == 40 and all(math.isfinite(v) for v in run["losses"]), "UNVERIFIED: nonfinite or missing losses")
            require(run.get("fresh_optimizer") is True and run.get("initial_optimizer_states") == 0,
                    "UNVERIFIED: non-fresh optimizer")
            require(run.get("scheduler") == "constant-no-state", "UNVERIFIED: wrong schedule")
            updates = run["updates"]
            require(len(updates) == 40 and [u["update"] for u in updates] == list(range(1, 41))
                    and all(u["lr"] == 1e-5 and u["examples"] == 32 and math.isfinite(u["preclip_norm"]) for u in updates),
                    "INCOMPLETE: exactly 40 finite nonzero updates required")
            exposure = run["exposure"]
            verify_seal(exposure)
            require(exposure["examples"] == 1280 and len(exposure["batches"]) == 40
                    and all(len(b) == 32 for b in exposure["batches"]), "INCOMPLETE: wrong exposure")
            exposures.append(exposure["identity_sha256"])
            require(exposure["identity_sha256"] == trusted["exposure_sha256"][str(seed)], "UNVERIFIED: exposure differs from frozen seed plan")
            curve = _curve(run["scores"])
            predictions = {}
            for t, cell in curve.items():
                require(cell["score_id"] in scores, "INCOMPLETE: score absent from manifest")
                identity = scores[cell["score_id"]]
                require(identity["predictions_sha256"] == digest(cell["predictions"]), "UNVERIFIED: score content changed")
                require(identity["summed_scores_sha256"] == digest(cell["summed_scores"]) and
                        identity["covariates_sha256"] == digest(cell["gold_covariates"]), "UNVERIFIED: score/covariates changed")
                require(identity["seed"] == seed and identity["update"] == t, "UNVERIFIED: wrong score cell")
                checkpoint = checkpoints[identity["checkpoint_id"]]
                if checkpoint.get("role") == "probe":
                    proof = evidence["stream_verifications"][checkpoint["checkpoint_id"]]
                    require(proof["gpu_identity"] == identity["gpu_identity"] and all(proof[k] == identity[k] for k in
                            ("predictions_sha256", "summed_scores_sha256", "covariates_sha256")), "UNVERIFIED: streamed score mismatch")
                require(checkpoint["start"] == start, "UNVERIFIED: wrong score start")
                require(identity["exposure_sha256"] == exposure["identity_sha256"], "UNVERIFIED: score exposure changed")
                predictions[t] = cell["predictions"]
                seed_scores.append(identity)
            initial[start] = predictions[0]
            result[start] = corrected_auc(predictions, cohort)
            result[start]["full_panel"] = full_panel_metrics(predictions)
            result[start]["timing"] = run.get("timing")
            initial_covariates = curve[0]["gold_covariates"]
            require(set(initial_covariates) == set(cohort["panel_ids"]), "INCOMPLETE: initial covariate coverage missing")
            require(all(math.isfinite(v["gold_log_probability"]) and 1 <= v["gold_rank"] <= 50
                        for v in initial_covariates.values()), "UNVERIFIED: invalid initial covariates")
            result[start]["initial_gold_covariates_on_common_errors"] = {i: initial_covariates[i] for i in cohort["ids"]}
            result[start]["representation_diagnostics"] = {t: curve[t].get("diagnostics") for t in (0, 40)}
            result[start]["generated_label_accuracy"] = {
                t: (100*sum(cell["generated_predictions"].values())/len(cell["generated_predictions"])
                    if cell.get("generated_predictions") else None) for t, cell in curve.items()}
            result[start]["mean_token_label_accuracy"] = {
                t: (100*sum(cell["mean_token_predictions"].values())/len(cell["mean_token_predictions"])
                    if cell.get("mean_token_predictions") else None) for t, cell in curve.items()}
        require(len(set(exposures)) == 1, "UNVERIFIED: unequal example/token/batch exposure")
        frozen = freeze_common_errors(initial)
        require(frozen == cohort, "UNVERIFIED: common-error cohort changed or selected after training")
        require(0 < len(trusted["evaluation_ids"]) <= 500 and len(set(trusted["evaluation_ids"])) == len(trusted["evaluation_ids"]) and
                set(cohort["panel_ids"]) == set(trusted["evaluation_ids"]),
                "INCOMPLETE: all eligible original test rows required")
        # Every curve cell, control and start shares one physical assignment.
        for field in ("gpu_identity", "panel_sha256", "prompt_sha256", "scorer_sha256", "decoding_sha256",
                      "tokenizer_sha256", "suffix_sha256"):
            require(len({digest(s[field]) for s in seed_scores}) == 1, "UNVERIFIED: paired score identity mismatch: "+field)
        require(all(len(s["gpu_identity"].get("devices", [])) == 1 and
                    s["gpu_identity"]["devices"][0] == s["gpu_identity"]["uuid"] for s in seed_scores),
                "UNVERIFIED: one physical scoring GPU per seed required")
        require(all(s["panel_sha256"] == manifest["preparation_sha256"] for s in seed_scores),
                "UNVERIFIED: score preparation changed")
        primary[seed] = result["stage2"]["auc"]-result["base"]["auc"]
        descriptive[seed] = dict(stage1_minus_base=result["stage1"]["auc"]-result["base"]["auc"],
                                 stage2_minus_stage1=result["stage2"]["auc"]-result["stage1"]["auc"])
        summaries[seed] = result
    interval = paired_interval(primary)
    sensitivity = all(summaries[s]["base"]["final"] >= 10 for s in SEEDS)
    shown = sensitivity and interval["mean"] <= -2 and interval["upper"] < 0 and interval["negative_seeds"] == 3
    return dict(status="PLASTICITY LOSS SHOWN" if shown else "INCONCLUSIVE", sensitivity=sensitivity,
                reason=None if sensitivity else "probe lacks demonstrated sensitivity",
                primary=interval, seed_differences=primary, descriptive=descriptive, curves=summaries,
                material_deficit_excluded_at_this_dose_cohort=sensitivity and interval["lower"] >= -1,
                preserved_plasticity_established=False,
                timing_status="MEASURED" if all(r.get("timing") is not None for r in runs.values()) else "UNMEASURED",
                wall_time_warnings=[dict(seed=s, start=j, warning="WALL-TIME RISK")
                    for (s, j), run in runs.items() if (run.get("timing") or {}).get("serial_scoring_wall_seconds", 0) > 24*3600])


def read_evidence(evidence, export_resolvers=None, *, trusted=None):
    try:
        return _read(evidence, export_resolvers, trusted)
    except (ValueError, KeyError, TypeError, OSError) as error:
        reason = "UNVERIFIED: local export artifact unavailable" if isinstance(error, OSError) else str(error)
        status = "INCOMPLETE" if "INCOMPLETE:" in reason else "PENDING" if "PENDING:" in reason else "UNVERIFIED"
        return dict(status=status, reason=reason, primary=None, preserved_plasticity_established=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--exports", required=True, help="local artifact resolver; never copied into receipts")
    parser.add_argument("--out", required=True)
    parser.add_argument("--trusted", required=True, help="independently frozen send/preparation/code/exposure digests")
    args = parser.parse_args()
    evidence = json.loads(Path(args.evidence).read_text())
    if "seed_records" in evidence:
        evidence = collect_seed_evidence(evidence["seed_records"])
    exports = json.loads(Path(args.exports).read_text())
    from kit.pd_prepare import write_frozen
    result = read_evidence(evidence, exports, trusted=json.loads(Path(args.trusted).read_text()))
    write_frozen(args.out, result)
    return 0 if result["primary"] is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
