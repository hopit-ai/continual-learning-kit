#!/usr/bin/env python3
"""Arms F/R data: an explicit shared 40x32 ID schedule, question -> verified text.

The schedule is JSON {"seed": 101, "ids": [id, ...]} (1,280 entries, repetitions allowed),
supplied by the common campaign/data unit, never a new arm-specific shuffle.
Validation uses training examples only, for loss diagnostics; held-out targets
must not enter this file. No demonstration is ever placed in a prompt column.
LR=1e-5 is a declared choice: pinned SDFT d775732 exposes its distillation LR
(2e-5 default/5e-5 README), but provides no SFT baseline launcher or SFT LR.
We retain K2b's AdamW, betas (0.9,0.999), decay .01, clip 1, 10-step warmup
then constant; reset optimizer each stage. Coordinate this frozen choice with D.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.v4_teacher import (TOKENIZER_8B, INPUT_CAP, RESPONSE_CAP, SFT_CONTEXT_CAP, atomic_json, load_pool, load_student_tokenizer,
                            read_jsonl, sha, student_messages, tokenize_student, verify)


def registered_scheduler(optimizer, num_warmup_steps, num_training_steps, **kwargs):
    """Ten absolute warmup updates then constant, even for the two-step smoke."""
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lambda step: min(step / 10, 1.0))


def encode_example(tokenizer, messages, response, max_length=SFT_CONTEXT_CAP):
    """Token-aligned mask: verl shifts loss_mask[:, 1:] alongside next-token labels.

    Response tokens INCLUDING EOS are labels; the last prompt logit predicts
    the first response token. No prompt token is a supervised label.
    """
    prompt_ids, response_ids = tokenize_student(tokenizer, messages, response)
    ids = prompt_ids + response_ids
    if len(prompt_ids) > INPUT_CAP or len(response_ids) > RESPONSE_CAP or len(ids) > max_length:
        raise ValueError("SFT example exceeds separate input/response allowances")
    if not prompt_ids or not response_ids:
        raise ValueError("empty SFT prompt/response")
    mask = [int(len(prompt_ids) <= i < len(ids)) for i in range(max_length)]
    attention = [1] * len(ids) + [0] * (max_length - len(ids))
    return {"input_ids": ids + [tokenizer.pad_token_id] * (max_length - len(ids)),
            "attention_mask": attention, "position_ids": list(range(len(ids))) + [0] * (max_length - len(ids)),
            "loss_mask": mask}


def build_rows(items, demonstrations, schedule, tokenizer):
    """Refuse unverified/unknown IDs rather than stuffing gold or dropping exposures."""
    from kit.v4_contract import check_synthetic_rows
    check_synthetic_rows(demonstrations)
    lookup = {str(i["id"]): i for i in items}
    demos = {}
    for d in demonstrations:
        key = str(d["id"])
        if key in demos or key not in lookup or d.get("verified") is not True or not d.get("text"):
            raise ValueError("duplicate, unknown or unverified demonstration")
        item = lookup[key]
        checked = verify(item, {"text": d["text"], "attempt": d.get("attempt", 1), "finish_reason": "stop"}, tokenizer, mapped_target=True)
        if not checked["verified"]:
            raise ValueError("demonstration fails verification: " + ", ".join(checked["reasons"]))
        demos[key] = d
    ids = schedule["ids"] if isinstance(schedule, dict) else schedule
    if len(ids) != 40 * 32:
        raise ValueError("common schedule must contain exactly 40 x 32 IDs")
    rows = []
    for ident in ids:
        key = str(ident)
        if key not in demos:
            raise ValueError("schedule contains uncovered question: " + key)
        messages = student_messages(lookup[key])
        response = demos[key]["text"]
        encoded = encode_example(tokenizer, messages, response)
        rows.append({"id": ident, "prompt": json.dumps(messages, ensure_ascii=False), "response": response,
                     "extra_info": {"demonstration": response, "stand_in_teacher": demos[key].get("stand_in_teacher", False), "teacher_model": demos[key].get("teacher_model", "unspecified")},
                     "context_tokens_8b": sum(encoded["attention_mask"])})
    return rows


def schedule_common_rows(rows, schedule, tokenizer):
    """One shared exposure file for F/R; validate both targets without changing prompts."""
    by_id = {}
    for row in rows:
        key = str(row["extra_info"]["index"])
        if key in by_id or row["extra_info"].get("split") != "train":
            raise ValueError("duplicate/non-training common row")
        item = {"id": key, "task": "finqa" if row["data_source"] == "finqa" else "chemistry",
                "question": row["extra_info"]["problem"], "messages": row["prompt"],
                "gold": row["reward_model"]["ground_truth"], "description": row["extra_info"]["description"]}
        for field in ("demonstration", "rewrite"):
            response = row["extra_info"][field]
            checked = verify(item, {"text": response, "attempt": 1, "finish_reason": "stop"}, tokenizer, mapped_target=True)
            if not checked["verified"]:
                raise ValueError("common " + field + " fails verifier")
            encode_example(tokenizer, row["prompt"], response)
        by_id[key] = {**row, "id": row["extra_info"]["index"]}
    from kit.v4_contract import check_synthetic_rows
    steps = schedule.get("steps", 40)
    check_synthetic_rows(rows, steps, schedule.get("profile", "scientific"))
    if not isinstance(steps, int) or not 1 <= steps <= 40 or len(schedule["ids"]) != steps * 32:
        raise ValueError("common schedule must contain exactly steps x 32 IDs")
    if steps != 40 and not all(r["extra_info"].get("stand_in_teacher") for r in rows) and schedule.get("profile") != "technical-smoke":
        raise ValueError("short schedules are smoke only; real models require an explicit technical-smoke profile")
    result = []
    for ident in schedule["ids"]:
        key = str(ident)
        if key not in by_id:
            raise ValueError("schedule contains uncovered question: " + key)
        result.append(by_id[key])
    return result


class V4SFTDataset:
    """verl custom dataset retaining original system/user messages and schedule order."""
    def __init__(self, parquet_files, tokenizer, config, max_samples=-1):
        import pyarrow.parquet as pq
        paths = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        self.rows = [r for path in paths for r in pq.read_table(path).to_pylist()]
        from kit.v4_contract import check_synthetic_rows
        check_synthetic_rows(self.rows, os.environ.get("STEPS", 40), os.environ.get("V4_PROFILE", "scientific"))
        self.tokenizer = tokenizer
        self.max_length = config.get("max_length", SFT_CONTEXT_CAP)
        self.response_key = config.get("response_key", "response")
        if self.response_key not in ("response", "extra_info.demonstration", "extra_info.rewrite"):
            raise ValueError("unknown SFT target field")
        if self.max_length != SFT_CONTEXT_CAP or config.get("truncation", "error") != "error":
            raise ValueError("arms F/R forbid truncation or a changed context cap")
        if max_samples > 0:
            raise ValueError("arms F/R must use the entire shared prompt schedule")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        response = row
        for field in self.response_key.split("."):
            response = response[field]
        messages = json.loads(row["prompt"]) if isinstance(row["prompt"], str) else row["prompt"]
        result = encode_example(self.tokenizer, messages, response, self.max_length)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}


def complete_export(path):
    """A config and some shards do not certify a merged HF checkpoint."""
    path = Path(path)
    if not all((path / f).is_file() and (path / f).stat().st_size for f in ("config.json", "tokenizer_config.json", "tokenizer.json")):
        return False
    index = path / "model.safetensors.index.json"
    try:
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values()) if index.exists() else {"model.safetensors"}
        return bool(shards) and all((path / s).is_file() and (path / s).stat().st_size for s in shards)
    except (ValueError, OSError):
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool")
    p.add_argument("--demonstrations")
    p.add_argument("--training-set", help="common training-set directory carrying both targets")
    p.add_argument("--task", choices=("chemistry", "finqa"))
    p.add_argument("--schedule", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer-8b", default=os.environ.get("QWEN3_8B_TOKENIZER"))
    args = p.parse_args(argv)
    import pyarrow as pa
    import pyarrow.parquet as pq
    schedule = json.loads(Path(args.schedule).read_text())
    if not isinstance(schedule, dict) or not isinstance(schedule.get("seed"), int):
        p.error("shared schedule must contain integer seed and ids")
    tokenizer = load_student_tokenizer(args.tokenizer_8b)
    common = None
    if args.training_set:
        if not args.task or args.pool or args.demonstrations:
            p.error("--training-set requires --task and replaces --pool/--demonstrations")
        root = Path(args.training_set)
        common = json.loads((root / "manifest.json").read_text())
        path = root / (args.task + ".parquet")
        if common["schema"] != "kit-v4-common-training.v1" or common["files"][args.task]["sha256"] != sha(path.read_bytes()):
            raise ValueError("common training data changed")
        rows = schedule_common_rows(pq.read_table(path).to_pylist(), schedule, tokenizer)
    else:
        if not args.pool or not args.demonstrations:
            p.error("provide --training-set/--task or --pool/--demonstrations")
        rows = build_rows(load_pool(args.pool), read_jsonl(args.demonstrations), schedule, tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(rows), out / "train.parquet")
    # Diagnostic loss on the first training batch; never a held-out demonstration.
    pq.write_table(pa.Table.from_pylist(rows[:32]), out / "val.parquet")
    from kit.v4_contract import technical_synthetic
    marked = technical_synthetic(rows)
    atomic_json(out / "manifest.json", {"schema": "kit-v4-sft-data.v1", "arm": "COMMON" if common else "F", "rows": len(rows),
                "teacher_models": common["teacher_models"] if common else sorted({r["extra_info"]["teacher_model"] for r in rows}),
                "stand_in_teacher": common["stand_in_teacher"] if common else any(r["extra_info"]["stand_in_teacher"] for r in rows),
                "scientific_phase_allowed": not marked and common["coverage"]["scientific_phase_allowed"] if common else False,
                "technical_synthetic": marked,
                "target_fields": ["extra_info.demonstration", "extra_info.rewrite"] if common else ["extra_info.demonstration"],
                "technical_padding": schedule.get("technical_padding", {}), "schedule_steps": schedule.get("steps", 40),
                "ids": [r["id"] for r in rows], "seed": schedule.get("seed") if isinstance(schedule, dict) else None,
                "common_training_manifest_sha256": sha((Path(args.training_set) / "manifest.json").read_bytes()) if common else None,
                "pool_sha256": sha(Path(args.pool).read_bytes()) if args.pool else None,
                "demonstrations_sha256": sha(Path(args.demonstrations).read_bytes()) if args.demonstrations else None,
                "schedule_sha256": sha(Path(args.schedule).read_bytes()), "response_only": True, "thinking": False,
                "train_sha256": sha((out / "train.parquet").read_bytes()), "val_sha256": sha((out / "val.parquet").read_bytes())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
