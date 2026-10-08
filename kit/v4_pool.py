#!/usr/bin/env python3
"""Freeze plan-v4 identities from pinned public data, never from model results.

python scripts/v4_data.py --sdpo SDPO --finqa FinQA/dataset
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from kit.v4_contract import INITIAL_8B_REVISION, PROMPT_CAP, prompt_length

OUT = ROOT / "docs/phase2/evidence/v4-data-manifest.json"
TAXONOMY = Path(__file__).resolve().parent / "resources/k8b-chemistry-templates.json"
TAXONOMY_SHA256 = "ade0870f8facd139924ead94c293c5655698383637f088e0b805cdf89011d17d"
CHEM_DIR = "datasets/sciknoweval/chemistry"
REMOVED_FAMILIES = ("distribution coefficient logD", "aqueous solubility")
FINQA_HASHES = {"train": "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
                "test": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc"}
POOL_RULE = 'Keep gold_consistent train items whose pinned Qwen3-8B chat-rendered prompt is at most 2,048 tokens; sort by sha256(UTF-8("v4-finqa-pool|" + id)), then id to break ties; take the first 1,441.'


def load_module(name: str, path: Path):
    """Load kit/reference code by file path, without requiring package installation."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finqa = load_module("v4_finqa_bed", ROOT / "kit/beds/finqa.py")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(item: dict) -> str:
    """Hash canonical JSON of the complete source row; no paths or timestamps enter it."""
    return sha(json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def read_json(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.split("\n") if line.strip()]


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def shortcut_evidence(item: dict) -> dict:
    """Recompute numeric rank from every option, rather than trusting the taxonomy's label."""
    pairs = re.findall(r"^([A-D]):\s*([-+]?\d+(?:\.\d+)?)\s*$", item["prompt"], re.M)
    require(len(pairs) == 4 and {p[0] for p in pairs} == set("ABCD"), "removed question lacks four numeric options")
    values = {letter: Decimal(value) for letter, value in pairs}
    require(len(set(values.values())) == 4, "removed question has tied numeric options")
    ranked = sorted(values, key=values.get)
    return {"options": dict(pairs), "ascending_letters": ranked, "third_smallest_letter": ranked[2],
            "gold": item["answer"], "matches_gold": ranked[2] == item["answer"]}


def freeze_chemistry(sdpo: Path, taxonomy: Path = TAXONOMY) -> dict:
    """Validate the frozen per-question mapping and exclude exactly the two declared families."""
    require(sha(taxonomy.read_bytes()) == TAXONOMY_SHA256, "frozen Chemistry taxonomy hash changed")
    tax = json.loads(taxonomy.read_text())
    result = {"source_commit": tax["source_commit"], "taxonomy_sha256": sha(taxonomy.read_bytes()),
              "removed_families": list(REMOVED_FAMILIES), "sources": {}, "train": [], "test": [],
              "probe": [], "removed_train": []}
    expected = {"train": 1890, "test": 210}
    require(len(tax["questions"]) == sum(expected.values()), "taxonomy does not cover both original splits")
    all_ids = set()
    for split in ("train", "test"):
        path = sdpo / CHEM_DIR / (split + ".json")
        digest = sha(path.read_bytes())
        require(digest == tax["sources"][split]["sha256"], "Chemistry source hash differs from frozen taxonomy: " + split)
        result["sources"][split] = {"file": CHEM_DIR + "/" + split + ".json", "sha256": digest}
        items = read_json(path)
        mapping = [q for q in tax["questions"] if q["split"] == split]
        require(len(items) == len(mapping) == expected[split], "Chemistry original count changed: " + split)
        require({q["position"] for q in mapping} == set(range(len(items))), "taxonomy repeats or omits positions")
        for q in sorted(mapping, key=lambda q: q["position"]):
            item = items[q["position"]]
            taxonomy_content = sha(json.dumps({"prompt": item.get("prompt"), "answer": item.get("answer")}, sort_keys=True).encode())
            require(item["idx"] == q["idx"] and taxonomy_content == q["content_sha256"], "taxonomy identity/content mismatch")
            require(str(item["idx"]) not in all_ids, "Chemistry train/test id overlap or duplicate")
            all_ids.add(str(item["idx"]))
            row = {"id": "sciknoweval-%s-%s" % (split, item["idx"]), "idx": item["idx"], "split": split,
                   "position": q["position"], "family": q["family"], "content_sha256": content_hash(item),
                   "taxonomy_content_sha256": taxonomy_content}
            if q["family"] in REMOVED_FAMILIES:
                row["positional_rule"] = shortcut_evidence(item)
                require(row["positional_rule"]["matches_gold"], "third-smallest rule failed: " + row["id"])
                result["removed_train" if split == "train" else "probe"].append(row)
            else:
                result[split].append(row)
    require((len(result["train"]), len(result["test"]), len(result["probe"]), len(result["removed_train"])) == (1441, 159, 51, 449),
            "Chemistry plan-v4 exclusion counts changed")
    result["counts"] = {k: len(result[k]) for k in ("train", "test", "probe", "removed_train")}
    result["removed_by_family"] = {split: dict(sorted(collections.Counter(q["family"] for q in result[key]).items()))
                                   for split, key in (("train", "removed_train"), ("test", "probe"))}
    train_content = {q["taxonomy_content_sha256"] for q in result["train"]}
    heldout_content = {q["taxonomy_content_sha256"] for key in ("test", "probe") for q in result[key]}
    result["separation"] = {"train_heldout_id_overlap": len({q["idx"] for q in result["train"]} & {q["idx"] for key in ("test", "probe") for q in result[key]}), "train_heldout_content_overlap": len(train_content & heldout_content)}
    require(not train_content & heldout_content, "Chemistry training/held-out content overlap")
    result["shortcut_rule"] = {"rule": "Sort four distinct numeric option values ascending; the third-smallest letter equals gold.",
                               "checked": len(result["removed_train"] + result["probe"]), "matches": sum(q["positional_rule"]["matches_gold"] for q in result["removed_train"] + result["probe"]), "heldout_only_probe": True}
    return result


def pool_key(item: dict) -> tuple:
    """The pool selection is independent of input order and any model result."""
    return sha(("v4-finqa-pool|" + item["id"]).encode()), item["id"]


def eligible_finqa(items):
    """Apply the pinned verl chat-rendered limit BEFORE the unchanged hash selection."""
    rows = finqa.rows_for_trainer(items, "train")
    return [(item, prompt_length(row["prompt"])) for item, row in zip(items, rows)]


def select_finqa_pool(items: list, n: int = 1441) -> list:
    eligible = [item for item, length in eligible_finqa(items) if finqa.gold_consistent(item) and length <= PROMPT_CAP]
    require(len(eligible) >= n, "too few gold-consistent FinQA training items")
    return sorted(eligible, key=pool_key)[:n]


def report_page(item: dict) -> str:
    """FinQA ids end in .pdf-<question number>; retain the company/year/page.pdf identity."""
    require(".pdf-" in item["id"], "FinQA id has no report-page identity")
    return item["id"].rsplit(".pdf-", 1)[0] + ".pdf"


def freeze_finqa(root: Path) -> dict:
    """Receipt 221's full held-out panel, with the training-only gold validity filter audited separately."""
    raw, loaded, sources = {}, {}, {}
    for split in ("train", "test"):
        path = root / (split + ".json")
        digest = sha(path.read_bytes())
        require(digest == FINQA_HASHES[split], "FinQA pinned source hash changed: " + split)
        raw[split], loaded[split] = read_json(path), finqa.load(root, split)
        require(raw[split] == loaded[split], "FinQA load changed the raw panel")
        require(len({i["id"] for i in loaded[split]}) == len(loaded[split]), "FinQA repeated ids")
        sources[split] = {"file": "dataset/" + split + ".json", "sha256": digest, "raw_count": len(raw[split]),
                          "load_count": len(loaded[split]), "gold_consistent_count": sum(finqa.gold_consistent(i) for i in loaded[split])}
    lengths = eligible_finqa(loaded["train"])
    pool = sorted([i for i, length in lengths if finqa.gold_consistent(i) and length <= PROMPT_CAP], key=pool_key)[:1441]
    require(len(pool) == 1441, "too few eligible FinQA training questions")
    test = loaded["test"]
    require(len(test) == 1147, "receipt 221 FinQA held-out size changed")
    original_overlap = sorted({report_page(i) for i in loaded["train"]} & {report_page(i) for i in test})
    overlap = sorted({report_page(i) for i in pool} & {report_page(i) for i in test})
    ids_overlap = sorted({i["id"] for i in pool} & {i["id"] for i in test})
    require(not ids_overlap and not overlap and not original_overlap, "FinQA pool or source leaks held-out ids or report pages")
    def record(item):
        return {"id": item["id"], "report_page": report_page(item), "content_sha256": content_hash(item),
                "gold_consistent": finqa.gold_consistent(item)}
    return {"source_commit": finqa.DATASET_COMMIT, "sources": sources, "pool_rule": POOL_RULE,
            "prompt_eligibility": {"tokenizer_revision": INITIAL_8B_REVISION, "max_prompt_length": PROMPT_CAP,
                "rendering": "rl_dataset.doc2len: apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
                "excluded_ids": [i["id"] for i, length in lengths if length > PROMPT_CAP],
                "gold_consistent_excluded_ids": [i["id"] for i, length in lengths if length > PROMPT_CAP and finqa.gold_consistent(i)],
                "eligible_gold_consistent": sum(length <= PROMPT_CAP and finqa.gold_consistent(i) for i, length in lengths)},
            "counts": {"train": len(pool), "test": len(test)}, "train": [{**record(i), "selection_sha256": pool_key(i)[0]} for i in pool],
            "test": [record(i) for i in test],
            "heldout_rule": "All test.json items, exactly as finqa.load and receipt 221 / kit.eval_bed; gold_consistent is a flag, not a held-out filter.",
            "gold_inconsistent_ids": {s: [i["id"] for i in loaded[s] if not finqa.gold_consistent(i)] for s in loaded},
            "separation": {"id_overlap": ids_overlap, "pool_test_report_pages": overlap,
                           "full_train_test_report_pages": original_overlap,
                           "documented_full_train_test_report_page_overlap": len(original_overlap)}}


def build_manifest(sdpo: Path, finqa_root: Path, taxonomy: Path = TAXONOMY) -> dict:
    """A deterministic freeze: only source bytes and fixed code/rules enter the record."""
    code = [Path(__file__), ROOT / "kit/v4_families.py", ROOT / "kit/beds/finqa.py"]
    return {"schema": "v4-data-manifest.v1", "content_hash_rule": content_hash.__doc__,
            "code_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in code},
            "chemistry": freeze_chemistry(Path(sdpo), taxonomy), "finqa": freeze_finqa(Path(finqa_root))}


def write_manifest(manifest: dict, out: Path):
    require("references" not in out.resolve().parts, "refusing to write a manifest inside references")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdpo", required=True)
    parser.add_argument("--finqa", help="FinQA dataset folder; default: sibling of SDPO, FinQA/dataset")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sdpo = Path(args.sdpo)
    manifest = build_manifest(sdpo, Path(args.finqa) if args.finqa else sdpo.parent / "FinQA/dataset")
    write_manifest(manifest, args.out)
    print("Chemistry", manifest["chemistry"]["counts"], "FinQA", manifest["finqa"]["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
