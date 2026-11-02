"""Prospective split-job PD reader; the qualified v2 reader remains frozen.

The v2 evidence checks below are retained, with pre-training cohort selection
and cross-job update-0 agreement replacing the former single-job cohort check.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import math
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.pd_contract import (SEEDS, STARTS, UPDATES, TRAINING_SPEC, digest, require, seal,
    validate_manifest, verify_live_export, verify_seal, verify_stream_receipt)
from kit.pd_read import (auc, _curve, full_panel_metrics, paired_interval)


def corrected_auc(prediction_curve, cohort):
    verify_seal(cohort)
    require(cohort.get("schema") == "pd-common-errors-v1" and cohort.get("frozen_before_updates") is True,
            "UNVERIFIED: unfrozen pre-training cohort")
    curve = _curve(prediction_curve)
    ids = cohort["ids"]
    require(len(ids) >= 64 and len(set(ids)) == len(ids), "PENDING: fewer than 64 common errors")
    for predictions in curve.values():
        require(set(predictions) == set(cohort["panel_ids"]) and
                all(isinstance(v, bool) for v in predictions.values()), "INCOMPLETE: curve panel coverage mismatch")
    accuracies = {t: 100*sum(p[i] for i in ids)/len(ids) for t, p in curve.items()}
    return dict(auc=auc(accuracies), accuracy=accuracies, final=accuracies[40], initial=accuracies[0])


def _read(evidence, export_resolvers, trusted):
    from kit.pd_jobs import validate_amendment
    agreements = validate_amendment(evidence, trusted)
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
        # Cohort selection belongs exclusively to the pre-training job.
        # Post-job update 0 may differ only under the approved reload rule.
        verify_seal(cohort)
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
    sensitivity = all(summaries[s]["base"]["final"]-summaries[s]["base"]["initial"] >= 10 for s in SEEDS)
    shown = sensitivity and interval["mean"] <= -2 and interval["upper"] < 0 and interval["negative_seeds"] == 3
    return dict(reload_agreement=agreements, status="PLASTICITY LOSS SHOWN" if shown else "INCONCLUSIVE", sensitivity=sensitivity,
                reason=None if sensitivity else "probe lacks demonstrated sensitivity",
                primary=interval, seed_differences=primary, descriptive=descriptive, curves=summaries,
                material_deficit_excluded_at_this_dose_cohort=sensitivity and interval["lower"] >= -1,
                preserved_plasticity_established=False,
                timing_status="MEASURED" if all(r.get("timing") is not None for r in runs.values()) else "UNMEASURED",
                wall_time_warnings=[dict(seed=s, start=j, warning="WALL-TIME RISK")
                    for (s, j), run in runs.items() if (run.get("timing") or {}).get("serial_scoring_wall_seconds", 0) > 24*3600])



def collect_seed_evidence(records):
    from kit.pd_read import collect_seed_evidence as collect_v2
    from kit.pd_jobs import validate_seed_amendment
    require(len(records)==3, "INCOMPLETE: three seed receipts required; failed seed is UNVERIFIED")
    for record in records:
        validate_seed_amendment(record)
    merged = collect_v2(records)
    merged.update(split_job_sha256=records[0]["split_job_sha256"],
        pre_training={k:v for r in records for k,v in r["pre_training"].items()},
        training_receipts={k:v for r in records for k,v in r["training_receipts"].items()},
        post_jobs={k:v for r in records for k,v in r["post_jobs"].items()})
    require(len({r["split_job_sha256"] for r in records}) == 1, "UNVERIFIED: split-job implementation changed")
    return merged


def read_evidence(evidence, export_resolvers=None, *, trusted=None):
    try:
        if "seed_records" in evidence:
            evidence=collect_seed_evidence(evidence["seed_records"])
        return _read(evidence, export_resolvers, trusted)
    except (ValueError, KeyError, TypeError, OSError) as error:
        reason = "UNVERIFIED: local export artifact unavailable" if isinstance(error, OSError) else str(error)
        status = ("UNVERIFIED-HARDWARE" if "UNVERIFIED-HARDWARE:" in reason else
                  "INCOMPLETE" if "INCOMPLETE:" in reason else "PENDING" if "PENDING:" in reason else "UNVERIFIED")
        return dict(status=status, reason=reason, primary=None, preserved_plasticity_established=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence", "exports", "trusted", "out"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    evidence = json.loads(Path(args.evidence).read_text())
    result = read_evidence(evidence, json.loads(Path(args.exports).read_text()) if Path(args.exports).is_file() else None,
                           trusted=json.loads(Path(args.trusted).read_text()) if Path(args.trusted).is_file() else None)
    from kit.pd_prepare import write_frozen
    write_frozen(args.out, result)
    return 0 if result["primary"] is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
