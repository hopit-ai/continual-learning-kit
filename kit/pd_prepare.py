"""Pinned, offline-first TREC preparation and fail-closed exclusion audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from urllib.request import urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kit.pd_contract import digest, require, seal, sha256_file, verify_seal

REVISION = "1f97567bdd2adedefe8abdaa9bd6ee0e6725b458"
SALT = "pd-trec-v2|"
SOURCE_ROOT = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"
SOURCES = {
    "train": {"name": "train_5500.label", "bytes": 335858, "rows": 5452,
              "sha256": "9e4c8bdcaffb96ed61041bd64b564183d52793a8e91d84fc3a8646885f466ec3"},
    "test": {"name": "TREC_10.label", "bytes": 23354, "rows": 500,
             "sha256": "033f22c028c2bbba9ca682f68ffe204dc1aa6e1cf35dd6207f2d4ca67f0d0e8e"},
}
REQUIRED_COVERAGE = frozenset({"programme_sources", "panels", "generic_candidates",
                              "generic_rejected", "accepted_responses", "broad",
                              "earlier_task", "K", "CR", "chemistry_goal1", "reserved"})
RESOURCE = Path(__file__).with_name("resources") / "pd"


def taxonomy():
    return json.loads((RESOURCE / "taxonomy.json").read_text())


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class TextProfile:
    """Reusable normalized text features; constructed once per indexed row."""
    def __init__(self, text, *, normalized=False):
        self.text = text if normalized else normalize(text)
        self.tokens = tuple(self.text.split())
        self.grams = frozenset(zip(*(self.tokens[i:] for i in range(5)))) if len(self.tokens) >= 5 else frozenset()
        self.length = len(self.text)
        self.characters = Counter(self.text)


def _edit_match(a, b):
    maximum = max(a.length, b.length)
    budget = maximum//10
    if abs(a.length-b.length) > budget:
        return False
    common = sum((a.characters & b.characters).values())
    if maximum-common > budget:
        return False
    # Banded exact Levenshtein: only paths that can meet >=0.9 survive.
    previous = {j: j for j in range(min(b.length, budget)+1)}
    outside = budget+1
    for i, left in enumerate(a.text, 1):
        current = {}
        for j in range(max(0, i-budget), min(b.length, i+budget)+1):
            if j == 0:
                current[j] = i
            else:
                current[j] = min(previous.get(j, outside)+1,
                                 current.get(j-1, outside)+1,
                                 previous.get(j-1, outside)+(left != b.text[j-1]))
        if not current or min(current.values()) > budget:
            return False
        previous = current
    return previous.get(b.length, outside) <= budget


def compare_profiles(a, b):
    if a.text == b.text:
        return "exact"
    if a.grams and b.grams and 5*min(len(a.grams), len(b.grams)) >= 4*max(len(a.grams), len(b.grams)):
        intersection = len(a.grams & b.grams)
        if 5*intersection >= 4*len(a.grams | b.grams):
            return "token5_jaccard"
    if _edit_match(a, b):
        return "edit_similarity"
    return None


def duplicate(a, b):
    """Public pair helper; audits reuse indexed profiles instead."""
    return compare_profiles(TextProfile(a), TextProfile(b))


class DuplicateIndex:
    """Exact lookup, shared-gram postings and length buckets avoid full scans."""
    def __init__(self, rows=()):
        self.profiles, self.ids, self.exact = [], [], {}
        self.by_length = defaultdict(set)
        self.by_gram = defaultdict(set)
        for row in rows:
            self.add(row)

    def add(self, row, profile=None):
        text = profile.text if profile is not None else normalize(row["text"])
        if text in self.exact:
            return  # Repeated ancestry prompts need one feature record.
        profile = profile or TextProfile(text, normalized=True)
        index = len(self.profiles)
        self.profiles.append(profile)
        self.ids.append(row["id"])
        self.exact[text] = index
        self.by_length[profile.length].add(index)
        for gram in profile.grams:
            self.by_gram[gram].add(index)

    def find(self, query):
        query = query if isinstance(query, TextProfile) else TextProfile(query)
        if query.text in self.exact:
            return "exact", self.ids[self.exact[query.text]]
        candidates = set()
        if query.grams:
            for gram in query.grams:
                for index in self.by_gram.get(gram, ()):
                    count = len(self.profiles[index].grams)
                    if 5*min(count, len(query.grams)) >= 4*max(count, len(query.grams)):
                        candidates.add(index)
        # All >=0.9 edit matches must lie in this inclusive length interval.
        low = math.ceil(.9*query.length-1e-12)
        high = math.floor(query.length/.9+1e-12)
        for length in range(low, high+1):
            candidates.update(self.by_length.get(length, ()))
        for index in sorted(candidates):
            reason = compare_profiles(query, self.profiles[index])
            if reason:
                return reason, self.ids[index]
        return None, None


def parse_trec(raw, split):
    require(split in SOURCES, "unknown source split")
    labels = taxonomy()
    rows = []
    # Mirror the pinned CogComp loader's one-byte correction and original IDs.
    for index, line in enumerate(raw.splitlines()):
        label, sep, text = line.replace(b"\xf0", b" ").strip().decode("utf-8").partition(" ")
        require(sep and text and label in labels, "invalid corrected TREC label row")
        rows.append(dict(id=f"{split}:{index}", original_id=index, split=split,
                         text=text, label=label))
    return rows


def verify_source(raw, split):
    source = SOURCES[split]
    require(len(raw) == source["bytes"] and hashlib.sha256(raw).hexdigest() == source["sha256"],
            "pinned CogComp source bytes mismatch")
    rows = parse_trec(raw, split)
    require(len(rows) == source["rows"], "pinned CogComp row count mismatch")
    return rows


def fetch_source(split, destination, *, fetcher=None):
    """Explicit opt-in acquisition; never called by preparation or imports."""
    if fetcher is None:
        def fetcher(url):
            with urlopen(url, timeout=60) as response:
                return response.read()
    raw = fetcher(SOURCE_ROOT + SOURCES[split]["name"])
    rows = verify_source(raw, split)
    destination = Path(destination)
    if destination.exists():
        require(destination.read_bytes() == raw, "refusing to replace frozen source")
    else:
        destination.write_bytes(raw)
    return rows


def pinned_provenance(loader, metadata):
    """Tie external byte pins to the loader and metadata at the HF revision."""
    loader, metadata = Path(loader), Path(metadata)
    script = loader.read_text()
    info = json.loads(metadata.read_text())["default"]
    checksums = info["download_checksums"]
    for source in SOURCES.values():
        url = SOURCE_ROOT+source["name"]
        require(SOURCE_ROOT in script and source["name"] in script, "pinned loader source URL mismatch")
        require(checksums[url] == {"num_bytes": source["bytes"], "checksum": source["sha256"]},
                "pinned loader metadata source hash mismatch")
    labels = info["features"]["fine_label"]["names"]
    require(set(labels) == set(taxonomy()) and len(labels) == 50, "pinned loader taxonomy mismatch")
    return seal(dict(schema="pd-source-provenance-v2", repository="CogComp/trec", revision=REVISION,
                     loader_sha256=sha256_file(loader), metadata_sha256=sha256_file(metadata),
                     source_hashes={k: v["sha256"] for k, v in SOURCES.items()}))


def fetch_provenance(destination, *, downloader=None):
    """Explicit future acquisition only; build/tests supply local fixture files."""
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    paths = {name: downloader(repo_id="CogComp/trec", filename=name, repo_type="dataset", revision=REVISION)
             for name in ("trec.py", "dataset_infos.json")}
    receipt = pinned_provenance(paths["trec.py"], paths["dataset_infos.json"])
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        target = root/name
        raw = Path(path).read_bytes()
        require(not target.exists() or target.read_bytes() == raw, "frozen provenance changed")
        if not target.exists():
            target.write_bytes(raw)
    write_frozen(root/"provenance.json", receipt)
    return receipt


def hash_order(rows):
    def key(row):
        split = row.get("split", row["id"].split(":", 1)[0])
        original_id = row.get("original_id", row["id"].split(":", 1)[-1])
        return hashlib.sha256((SALT+split+"|"+str(original_id)).encode()).hexdigest(), row["id"]
    return sorted(rows, key=key)


def audit(rows, inventory, topic_reviews, custodian, *, expected_inventory_sha256):
    """Inventory closure is external evidence, not inferred from supplied pools.

    Each source has a content digest even if its rows are legitimately empty.
    Reserved comparisons are custodian decisions bound to the candidate and
    inventory digests; reserved text/outcomes are never accepted by this API.
    """
    verify_seal(inventory)
    require(inventory["identity_sha256"] == expected_inventory_sha256, "ancestry inventory changed")
    require(inventory.get("closed") is True, "PENDING: unclosed ancestry")
    require(set(inventory["coverage"]) >= REQUIRED_COVERAGE, "PENDING: incomplete programme coverage")
    require(inventory.get("source_ids") == sorted(source["id"] for source in inventory["sources"]),
            "source inventory is not completely enumerated")
    require(len(set(inventory["source_ids"])) == len(inventory["source_ids"]), "duplicate ancestry source")
    corpus = []
    for source in inventory["sources"]:
        require(source.get("kind") != "reserved", "reserved text must stay with custodian")
        require(source["rows_sha256"] == digest(source["rows"]), "ancestry rows changed")
        corpus.extend(source["rows"])
    require(len({row["id"] for row in rows}) == len(rows), "duplicate original ID")
    candidate_sha = digest(rows)
    verify_seal(custodian)
    require(custodian.get("closed") is True and custodian.get("candidate_sha256") == candidate_sha
            and custodian.get("inventory_sha256") == expected_inventory_sha256,
            "PENDING: custodian exclusions not bound to complete inventory/candidates")
    require(custodian.get("thresholds") == {"token5_jaccard": .8, "edit_similarity": .9},
            "custodian duplicate thresholds mismatch")
    require(set(topic_reviews) == {r["id"] for r in rows}, "PENDING: complete human topic review required")
    require(set(custodian["decisions"]) == set(topic_reviews), "PENDING: missing reserved exclusions")
    accepted, decisions = [], []
    programme_index, accepted_index = DuplicateIndex(corpus), DuplicateIndex()
    # Test priority prevents unused late training rows from shrinking evaluation.
    ordered = hash_order([r for r in rows if r.get("split", r["id"].split(":", 1)[0]) == "test"])
    ordered += hash_order([r for r in rows if r.get("split", r["id"].split(":", 1)[0]) == "train"])
    require(len(ordered) == len(rows), "unknown original TREC split")
    for row in ordered:
        review = topic_reviews[row["id"]]
        require(review.get("reviewed") is True and isinstance(review.get("exclude"), bool),
                "PENDING: human chemistry/topic review incomplete")
        require(isinstance(custodian["decisions"][row["id"]], bool), "invalid custodian decision")
        reason, match = None, None
        if review["exclude"]:
            reason = "human_topic"
        elif custodian["decisions"][row["id"]]:
            reason = "custodian_reserved"
        else:
            profile = TextProfile(row["text"])
            reason, match = programme_index.find(profile)
        # Deduplicate across all eligible rows before assigning partitions.
        if not reason:
            reason, match = accepted_index.find(profile)
        decisions.append(dict(id=row["id"], reason=reason, match=match))
        if not reason:
            accepted.append(row)
            accepted_index.add(row, profile)
    return accepted, seal(dict(candidate_sha256=candidate_sha,
                              inventory_sha256=expected_inventory_sha256,
                              custodian_sha256=custodian["identity_sha256"],
                              topic_review_sha256=digest(topic_reviews), decisions=decisions))


def prepare(rows, inventory, topic_reviews, custodian, encode, *, expected_inventory_sha256,
            tokenizer_sha256, code_sha256, source_hashes=None, provenance=None):
    require(provenance is not None, "pinned loader provenance required")
    verify_seal(provenance)
    require(provenance["revision"] == REVISION and provenance["repository"] == "CogComp/trec"
            and provenance["source_hashes"] == {k: v["sha256"] for k, v in SOURCES.items()},
            "pinned source provenance mismatch")
    eligible, exclusion = audit(rows, inventory, topic_reviews, custodian,
                                expected_inventory_sha256=expected_inventory_sha256)
    encoded, oversize, measurements = {}, [], []
    for row in eligible:
        tokenized = encode(row)
        require(tokenized["id"] == row["id"], "tokenization ID mismatch")
        measurements.append(dict(id=row["id"], prompt_tokens=len(tokenized["prefix_ids"]),
                                 maximum_label_eos_sequence_tokens=max(tokenized["candidate_lengths"]),
                                 training_sequence_tokens=len(tokenized["input_ids"])))
        if max(tokenized["candidate_lengths"]) > 1024 or len(tokenized["input_ids"]) > 1024:
            oversize.append(row["id"])
        else:
            encoded[row["id"]] = tokenized
    train = hash_order([r for r in eligible if r["split"] == "train" and r["id"] in encoded])
    test = hash_order([r for r in eligible if r["split"] == "test" and r["id"] in encoded])
    require(len(train) >= 1408, "PENDING: insufficient eligible training/calibration rows; defer PD")
    require(len(test) <= 500, "invalid original test coverage")
    partitions = dict(calibration=[r["id"] for r in train[:128]],
                      training=[r["id"] for r in train[128:1408]],
                      evaluation=[r["id"] for r in test])
    selected = {i for part in partitions.values() for i in part}
    return seal(dict(schema="pd-preparation-v2", source_revision=REVISION, source_provenance=provenance,
                     source_hashes=source_hashes or {k: v["sha256"] for k, v in SOURCES.items()},
                     exclusion=exclusion, oversize=oversize, partitions=partitions,
                     rows=[r for r in rows if r["id"] in selected],
                     tokenized={i: encoded[i] for i in sorted(selected)},
                     tokenizer_sha256=tokenizer_sha256, code_sha256=code_sha256,
                     prompt_sha256=digest(taxonomy()), no_replenishment=True,
                     eligible_test_ids=[r["id"] for r in test], encoding_measurements=measurements,
                     candidate_supersets=freeze_candidate_supersets(inventory),
                     sequence_cap_measurements=dict(max_sequence_tokens=max((max(t["candidate_lengths"]) for t in encoded.values()), default=0),
                                                    max_prompt_tokens=max((len(t["prefix_ids"]) for t in encoded.values()), default=0))))


def freeze_candidate_supersets(inventory):
    """Freeze prompts independently of future response acceptance decisions."""
    verify_seal(inventory)
    return seal(dict(schema="pd-candidate-supersets-v2", inventory_sha256=inventory["identity_sha256"],
                     sources={s["id"]: sorted({normalize(r["text"]) for r in s["rows"]}) for s in inventory["sources"]}))


def verify_returned_pools(preparation, returned):
    """Verification only: never mutate partitions, delete items or re-score."""
    verify_seal(preparation)
    frozen = preparation["candidate_supersets"]
    verify_seal(frozen)
    try:
        verify_seal(returned)
        require(returned["candidate_supersets_sha256"] == frozen["identity_sha256"], "candidate superset changed")
        require(set(returned["sources"]) == set(frozen["sources"]), "missing or added ancestry source")
        pd_index = DuplicateIndex(preparation["rows"])
        for source, rows in returned["sources"].items():
            superset = set(frozen["sources"][source])
            for row in rows:
                prompt = TextProfile(row["text"])
                require(prompt.text in superset, "returned prompt outside frozen superset")
                require(not pd_index.find(prompt)[0], "returned prompt/response overlaps frozen PD")
                if row.get("response"):
                    require(not pd_index.find(TextProfile(row["response"]))[0], "returned response overlaps frozen PD")
        return seal(dict(status="VERIFIED", preparation_sha256=preparation["identity_sha256"],
                         returned_sha256=returned["identity_sha256"], candidate_supersets_sha256=frozen["identity_sha256"]))
    except (KeyError, ValueError, TypeError) as error:
        return seal(dict(status="UNVERIFIED", reason=str(error), preparation_sha256=preparation["identity_sha256"],
                         candidate_supersets_sha256=frozen["identity_sha256"]))


def write_frozen(filename, value):
    target = Path(filename)
    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if target.exists():
        require(target.read_text() == rendered, "frozen artifact changed")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered)



def smoke_scope_prepare(rows, encode, *, tokenizer_sha256, code_sha256, public_pins_sha256, rules, development_rows):
    """Manager-authorized SMOKE-SCOPE only; never impersonates scientific audit."""
    require(rows and all(row["split"] == "train" for row in rows), "smoke scope requires training source only")
    require(len({r["id"] for r in rows}) == len(rows), "duplicate original smoke ID")
    patterns = {name: re.compile(pattern) for name, pattern in rules["topic_patterns"].items()}
    flags, decisions, eligible, tokenized = [], [], [], {}
    index = DuplicateIndex(development_rows)
    for row in hash_order(rows):
        reasons = [name for name, pattern in patterns.items() if pattern.search(normalize(row["text"]))]
        if reasons:
            flags.append(dict(id=row["id"], original_id=row["original_id"], text=row["text"],
                              label=row["label"], reasons=reasons, automated_excluded=True))
            decisions.append(dict(id=row["id"], reason="topic_flag", match=None))
            continue
        reason, match = index.find(row["text"])
        if reason:
            decisions.append(dict(id=row["id"], reason=reason, match=match))
            continue
        value = encode(row)
        require(value["id"] == row["id"], "smoke encoding identity changed")
        if max(value["candidate_lengths"]) > 1024 or len(value["input_ids"]) > 1024:
            decisions.append(dict(id=row["id"], reason="sequence_cap", match=None))
            continue
        eligible.append(row)
        tokenized[row["id"]] = value
        index.add(row)
        decisions.append(dict(id=row["id"], reason=None, match=None))
    require(len(eligible) >= 1408, "PENDING: insufficient smoke-scope training/calibration rows")
    selected = eligible[:1408]
    ids = [r["id"] for r in selected]
    preparation = seal(dict(schema="pd-smoke-scope-preparation-v1", scope="SMOKE-SCOPE",
                            scientific_preparation="PENDING", topic_review="PENDING",
                            custodian_decisions="PENDING", programme_exclusions="PENDING",
                            source_revision=REVISION, public_pins_sha256=public_pins_sha256,
                            rules_sha256=digest(rules), development_rows_sha256=digest(development_rows),
                            code_sha256=code_sha256, tokenizer_sha256=tokenizer_sha256,
                            decisions=decisions, selected_ids=ids, no_replenishment=True))
    inputs = seal(dict(schema="pd-smoke-inputs-v2", scope="SMOKE-SCOPE",
                       preparation_sha256=preparation["identity_sha256"], scientific_preparation="PENDING",
                       code_sha256=code_sha256, tokenizer_sha256=tokenizer_sha256,
                       partitions=dict(calibration=ids[:128], training=ids[128:]),
                       rows=selected, tokenized={i: tokenized[i] for i in ids}))
    flag_receipt = seal(dict(schema="pd-smoke-topic-flags-v1", scope="SMOKE-SCOPE",
                             status="GENERATED_AUTOMATICALLY_REVIEW_PENDING", review="PENDING",
                             source_revision=REVISION, rules_sha256=digest(rules), items=flags,
                             candidate_rows=len(rows), flagged_rows=len(flags), scientific_preparation="PENDING"))
    return preparation, inputs, flag_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="explicit acquisition; never run during build")
    fetch.add_argument("--split", choices=SOURCES, required=True)
    fetch.add_argument("--out", required=True)
    provenance = sub.add_parser("fetch-provenance", help="explicit pinned loader acquisition; never run during build")
    provenance.add_argument("--out", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--train", required=True)
    verify.add_argument("--test", required=True)
    verify.add_argument("--out", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--plan", required=True, help="local source, audit and tokenizer inputs")
    build.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_source(args.split, args.out)
    elif args.command == "fetch-provenance":
        fetch_provenance(args.out)
    elif args.command == "verify":
        rows = verify_source(Path(args.train).read_bytes(), "train") + verify_source(Path(args.test).read_bytes(), "test")
        write_frozen(args.out, rows)
    else:
        from transformers import AutoTokenizer
        from kit.pd_probe import code_identity, encode_row, tokenizer_identity
        plan = json.loads(Path(args.plan).read_text())
        rows = []
        for split in SOURCES:
            rows.extend(verify_source(Path(plan[split]).read_bytes(), split))
        tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer"], local_files_only=True)
        document = prepare(rows, plan["inventory"], plan["topic_reviews"], plan["custodian"],
                           lambda row: encode_row(tokenizer, row),
                           expected_inventory_sha256=plan["expected_inventory_sha256"],
                           tokenizer_sha256=tokenizer_identity(plan["tokenizer"]), code_sha256=code_identity(),
                           provenance=pinned_provenance(plan["loader"], plan["metadata"]))
        write_frozen(args.out, document)


if __name__ == "__main__":
    main()
