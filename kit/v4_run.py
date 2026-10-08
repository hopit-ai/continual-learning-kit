"""Shared data admission and receipts for the plan-v4 launch entry."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.v4_teacher import atomic_json, sha
from kit.v4_contract import check_schedule, check_heldout_rows, profile, check_synthetic_rows, technical_synthetic

REGISTERED = {"learning_rate": 1e-5, "lr_warmup_steps": 10, "lr_scheduler": "constant",
              "weight_decay": .01, "batch_prompts": 32, "max_response_length": 2048,
              "optimizer_reset": True, "shuffle": False, "steps": 40, "finish_gate": 1}


def require(condition, message="v4 gate failed"):
    if not condition:
        raise ValueError(message)


def check_data(env):
    """Require one hashed ordered schedule and qualified full-dose data for every arm."""
    import pyarrow.parquet as pq
    doc = json.loads(Path(env["DATA_MANIFEST"]).read_text())
    train = Path(env["TRAIN_FILE"])
    steps = int(env.get("STEPS", 40))
    name = profile(steps, env.get("V4_PROFILE", "scientific"))
    require(doc["schema"] == "kit-v4-sft-data.v1" and doc["arm"] == "COMMON", "all arms require common data")
    require(doc["seed"] == int(env["SEED"]) and doc["train_sha256"] == sha(train.read_bytes()), "schedule seed/hash mismatch")
    rows = pq.read_table(train).to_pylist()
    check_synthetic_rows([doc, *rows], steps, name)
    require(bool(doc.get("technical_synthetic")) == technical_synthetic(rows), "technical_synthetic manifest/row labels disagree")
    require(not technical_synthetic(rows) or doc.get("scientific_phase_allowed") is False, "synthetic manifest cannot claim scientific admission")
    require(len(rows) == doc["rows"] >= steps * 32, "insufficient scheduled exposures")
    require([r["extra_info"]["index"] for r in rows] == doc["ids"], "schedule order changed")
    require(all(r["extra_info"].get("split") == "train" and r["extra_info"].get("demonstration") and r["extra_info"].get("rewrite") for r in rows), "v4 gate failed")
    require(name == "technical-smoke" or (doc["scientific_phase_allowed"] and not doc["stand_in_teacher"]), "unqualified or stand-in data is smoke only")
    check_schedule(rows, steps)
    if env["ARM"] in ("S", "D"):
        data = Path(env["DATASET"])
        if not data.is_absolute():
            data = Path(env["SDPO_DIR"]) / data
        require(sha((data / "train.parquet").read_bytes()) == doc["train_sha256"], "RL and SFT schedules differ")
        heldout = pq.read_table(data / "test.parquet").to_pylist()
        check_heldout_rows(heldout)
    return doc


def summarize(env):
    """Normalize fields while retaining each launcher's original receipts and settings."""
    out = Path(env["WORK"]) / "runs" / env["NAME"]
    doc = json.loads((out / "run-summary.json").read_text())
    data = json.loads(Path(env["DATA_MANIFEST"]).read_text())
    arm = env["ARM"]
    registered = {**REGISTERED, "rollout_n": 8 if arm == "S" else 1,
                  "teacher_update_rate": .05 if arm == "S" else .01 if arm == "D" else None}
    doc.update({k:v for k,v in registered.items() if k != "steps"}, registered=registered, arm=arm, seed=int(env["SEED"]),
               technical_smoke=env.get("V4_PROFILE", "scientific") == "technical-smoke", profile=env.get("V4_PROFILE", "scientific"),
               scientific_phase_allowed=env.get("V4_PROFILE", "scientific") == "scientific" and data.get("scientific_phase_allowed") is True, stand_in_teacher=data["stand_in_teacher"],
               teacher_models=data["teacher_models"], target_field={"F": "extra_info.demonstration", "R": "extra_info.rewrite", "D": "extra_info.demonstration", "S": None}[arm],
               data_manifest_sha256=sha(Path(env["DATA_MANIFEST"]).read_bytes()),
               common_training_manifest_sha256=data["common_training_manifest_sha256"],
               schedule_sha256=data["schedule_sha256"], train_sha256=data["train_sha256"],
               train_file=env["TRAIN_FILE"])
    from kit.v4_receipts import completed_updates
    requested = int(env.get("STEPS", 40))
    completed = completed_updates(out)
    doc.update(requested_optimizer_updates=requested, completed_optimizer_updates=completed, dose_complete=completed == requested)
    doc["technical_synthetic"] = technical_synthetic(data)
    doc["scientific_phase_allowed"] &= doc["dose_complete"] and not doc["technical_synthetic"]
    from kit.v4_qualification import departures_hash, check_metrics
    doc['departures_sha256']=departures_hash()
    if doc.get('dose_complete') and not os.environ.get('V4_RESTORE_PROBE_OUT'):
        from kit.v4_teacher import read_jsonl
        doc.update(check_metrics(read_jsonl(out/'metrics.jsonl'), arm, int(requested)))
    atomic_json(out / "run-summary.json", doc)
    atomic_json(out / "env/v4-settings.json", {**registered, "technical_synthetic":doc["technical_synthetic"]})
    return doc


if __name__ == "__main__":
    if sys.argv[1:] in (["--help"], ["-h"]):
        print("usage: v4_run.py {check,summarize} (registered settings from environment)")
        raise SystemExit(0)
    if sys.argv[1] == "check":
        check_data(os.environ)
    elif sys.argv[1] == "summarize":
        summarize(os.environ)
    else:
        raise SystemExit("use check or summarize")
