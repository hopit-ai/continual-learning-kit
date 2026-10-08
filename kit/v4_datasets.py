#!/usr/bin/env python3
"""Produce shippable plan-v4 parquet folders without writing into reference checkouts.

python scripts/v4_make_datasets.py --sdpo DIR --out OUT [--finqa FinQA/dataset]
Ship OUT/datasets into the partner's writable SDPO_DIR; use DATASET=datasets/v4_chem
or datasets/v4_finqa, REWARD_FILE=/absolute/kit/beds/v4_reward.py, FEEDBACK=0, SOFT=0.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import tarfile
from pathlib import Path

from kit import v4_pool as data

SDPO_COMMIT = "7c457fc1b1f636ae794eb0362ba37d4743b06fbc"
PREPROCESS_SHA256 = "5c39bf32e93f4f14ff012a1c8dadb2ad359dd610145ead3f0008479c8d8739c8"


def check_preprocess(sdpo, check_commit=True):
    data.require(data.sha((sdpo / "data/preprocess.py").read_bytes()) == PREPROCESS_SHA256, "pinned Chemistry preprocess hash changed")
    if not check_commit:
        return
    commit = subprocess.check_output(["git", "-C", str(sdpo), "rev-parse", "HEAD"], text=True).strip()
    data.require(commit == SDPO_COMMIT, "pinned SDPO commit changed")


def pilot_prompt_parity(rows):
    if not os.environ.get("V4_PILOT_ARCHIVE"):return {"available": False, "checked": 0}
    archive = Path(os.environ["V4_PILOT_ARCHIVE"])
    if not archive.is_file():
        return {"available": False, "checked": 0}
    from kit.v4_contract import pinned_tokenizer
    tokenizer = pinned_tokenizer()
    with tarfile.open(archive) as tar:
        records = [json.loads(line) for line in tar.extractfile("send4-k8b/k8b/eval/base8b-chemistry-a1/tokens.jsonl")]
    expected = {r["id"]: r["prompt_sha256"] for r in records}
    matches = 0
    for row in rows:
        key = "sciknoweval-test-" + str(row["extra_info"]["index"])
        rendered = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        data.require(expected.get(key) == data.sha(rendered.encode()), "pilot prompt hash differs: " + key)
        matches += 1
    data.require(matches == len(expected) == 210, "pilot prompt panel incomplete")
    return {"available": True, "checked": matches, "matches": matches, "archive_sha256": data.sha(archive.read_bytes())}


def chemistry_rows(sdpo: Path, split: str) -> list:
    """Reuse authors' parquet when present; otherwise run their actual preprocessing in memory on pinned JSON.

    The local read-only reference has JSON only. No copy of the authors' prompt-building logic lives here.
    extra_info.index remains the original global id; no prompt-building logic is copied here.
    """
    import pyarrow.parquet as pq
    check_preprocess(sdpo, check_commit=False)
    root = sdpo / data.CHEM_DIR
    items = data.read_json(root / (split + ".json"))
    preprocess = data.load_module("v4_authors_preprocess", sdpo / "data/preprocess.py")
    mapper = preprocess.make_map_fn(split)
    expected = {str(item["idx"]): mapper(copy.deepcopy(item), n) for n, item in enumerate(items)}
    path = root / (split + ".parquet")
    if path.is_file():
        rows = pq.read_table(path).to_pylist()
    else:
        from datasets import Dataset
        # The reference's load_dataset uses on-disk caches. Keep the input in memory
        # and call its same sharding/mapping function, with no reference/cache writes.
        train_size = len(data.read_json(root / "train.json")) if (root / "train.json").is_file() else len(items)
        num_shards = min(4, train_size // 1000 + 1)
        processed = preprocess._map_in_shards(Dataset.from_list(items), split, num_shards=num_shards, num_proc=None)
        rows = processed.data.table.to_pylist()
    seen = set()
    for row in rows:
        key = str(row["extra_info"]["index"])
        data.require(key in expected and key not in seen, "authors' parquet has foreign or repeated Chemistry id")
        seen.add(key)
        for field in ("data_source", "prompt", "reward_model", "extra_info"):
            data.require(row[field] == expected[key][field], "authors' parquet differs from pinned preprocessing: " + key)
    data.require(seen == set(expected), "authors' parquet is missing Chemistry ids")
    return rows


def filter_rows(rows: list, members: list) -> list:
    """Filter by global identity, in manifest order, never renumbering held-out/probe questions."""
    by_id = {str(row["extra_info"]["index"]): row for row in rows}
    return [by_id[str(member["idx"])] for member in members]


def make_datasets(sdpo: Path, out: Path, finqa_root: Path | None = None) -> dict:
    """Validate the freeze, then write five trainer files and their content/provenance manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    sdpo, out = Path(sdpo).resolve(), Path(out).resolve()
    finqa_root = Path(finqa_root).resolve() if finqa_root else sdpo.parent / "FinQA/dataset"
    data.require("references" not in out.parts and not out.is_relative_to(sdpo) and not out.is_relative_to(finqa_root),
                 "OUT must be a shipping folder outside reference/source checkouts")
    check_preprocess(sdpo)
    frozen = data.build_manifest(sdpo, finqa_root)
    chem_train, chem_test = chemistry_rows(sdpo, "train"), chemistry_rows(sdpo, "test")
    chem = frozen["chemistry"]
    finqa_items = {split: {i["id"]: i for i in data.finqa.load(finqa_root, split)} for split in ("train", "test")}
    tables = {"v4_chem/train": filter_rows(chem_train, chem["train"]),
              "v4_chem/test": filter_rows(chem_test, chem["test"]),
              "v4_chem_probe/test": filter_rows(chem_test, chem["probe"])}
    for split in ("train", "test"):
        items = [finqa_items[split][q["id"]] for q in frozen["finqa"][split]]
        tables["v4_finqa/" + split] = data.finqa.rows_for_trainer(items, split)
    from kit.v4_contract import prompt_length, INITIAL_8B_REVISION
    lengths = {}
    for name, rows in tables.items():
        if name.endswith("/train"):
            lengths[name] = [prompt_length(row["prompt"]) for row in rows]
            data.require(max(lengths[name]) <= 2048, "shipped training prompt exceeds registered 2048-token allowance")
    files = [out / "datasets" / (name + ".parquet") for name in tables]
    data.require(all(p.resolve().is_relative_to(out) for p in files), "dataset output symlink escapes shipping folder")
    data.require(not any(p.exists() for p in files + [out / "v4-datasets-manifest.json", out / "v4-data-manifest.json"]),
                 "refusing to overwrite existing v4 dataset files")
    provenance = {"schema": "v4-datasets.v1", "data_manifest_sha256": data.content_hash(frozen),
                  "chemistry_preprocess_sha256": data.sha((sdpo / "data/preprocess.py").read_bytes()),
                  "chemistry_rows": "Authors' preprocess output filtered by extra_info.index; JSON-only checkouts use the authors' _map_in_shards in memory.",
                  "sdpo_commit": SDPO_COMMIT, "pilot_prompt_parity": pilot_prompt_parity(chem_test),
                  "prompt_eligibility": {"tokenizer_revision": INITIAL_8B_REVISION, "cap": 2048,
                      "training_max_tokens": {name: max(values) for name, values in lengths.items()}}, "files": {}}
    for name, rows in tables.items():
        path = out / "datasets" / (name + ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd", row_group_size=32)
        data.require(pq.read_table(path).to_pylist() == rows, "parquet round-trip changed trainer rows")
        provenance["files"]["datasets/" + name + ".parquet"] = {
            "rows": len(rows), "sha256": data.sha(path.read_bytes()),
            "rows_sha256": data.sha(json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())}
    data.write_manifest(frozen, out / "v4-data-manifest.json")
    data.write_manifest(provenance, out / "v4-datasets-manifest.json")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdpo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--finqa", type=Path)
    args = parser.parse_args(argv)
    result = make_datasets(args.sdpo, args.out, args.finqa)
    print({name: record["rows"] for name, record in result["files"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
