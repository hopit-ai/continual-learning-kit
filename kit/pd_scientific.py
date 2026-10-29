"""Scientific PD preparation and verify-only closure, outside the qualified core.

CPU/data-only commands. No model inference, training, reserved outcomes or remote
execution. Human approval remains explicit; absent evidence always defers PD.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import lzma
from pathlib import Path
import re
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit import pd_prepare as p, pd_probe as probe
from kit.pd_contract import SEEDS, STARTS, digest, require, seal, sha256_file, verify_seal, is_sha

KIT = Path(__file__).resolve().parent
TEXT_FIELDS = ("content", "prompt", "response", "completion", "target", "answer", "output", "chosen", "question", "text", "demonstration", "response_as_generated", "refusal_text")
THRESHOLDS = dict(token5_jaccard=.8, edit_similarity=.9)


# Manager ruling, 29 October 2026. This is a send-time source separation
# commitment, never an empirical comparison against a nonexistent slice.
CONFIRMATION_STATEMENT = (
    "no confirmation slice exists at send time; TREC (public question-type "
    "classification data) is disjoint from the SEC-filing/legal/maths "
    "confirmation sources by construction; the custodian re-checks PD prompts "
    "against the slice when it is built"
)


def confirmation_ruling():
    return seal(dict(schema="pd-confirmation-send-ruling-v1", authority="manager",
        date="2026-10-29", statement=CONFIRMATION_STATEMENT,
        confirmation_months=["2026-10", "2026-11"], slice_exists_at_send=False,
        disjoint_by_construction=True, comparison_performed=False,
        future_recheck_required=True, recheck_status="PENDING_SLICE_BUILD",
        slice_build_window=dict(start="2026-12-15", end="2026-12-31"),
        recheck_scope="All frozen PD candidate prompts; custodian compares against the slice when built; no reserved outcomes accessed.",
        thresholds=THRESHOLDS, reserved_outcomes_accessed=False,
        verification_only=True, deletion_allowed=False, rescoring_allowed=False))


def read(path):
    path = Path(path)
    raw=path.read_bytes()
    return json.loads(lzma.decompress(raw) if path.suffix==".xz" else gzip.decompress(raw) if path.suffix==".gz" else raw)


def write(path, obj):
    path = Path(path)
    if path.suffix not in (".gz",".xz"):
        p.write_frozen(path, obj)
        return
    raw=json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    raw=lzma.compress(raw,preset=6) if path.suffix==".xz" else gzip.compress(raw,mtime=0)
    require(not path.exists() or path.read_bytes() == raw, "frozen compressed artifact changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def jsonl(path):
    # Unicode paragraph/line separators inside a JSON string are not row breaks.
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(v.get("content", "") for v in value if isinstance(v, dict) and isinstance(v.get("content"), str))
    return ""


def row_texts(row):
    result = [(field, text(row[field])) for field in TEXT_FIELDS if field in row and text(row[field]).strip()]
    for i, message in enumerate(row.get("messages", [])):
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            result.append(("message-"+str(i), message["content"]))
    extra = row.get("extra_info", {})
    if isinstance(extra, dict) and isinstance(extra.get("demonstration"), str):
        result.append(("demonstration", extra["demonstration"]))
    return result


def source_from_rows(name, role, rows, byte_sha256):
    values = []
    for i, row in enumerate(rows):
        for field, value in row_texts(row):
            values.append(dict(id=f"{name}:{i}:{field}", text=p.normalize(value)))
    require(values, "DEFERRED: inventory contains no auditable text")
    return dict(id=name, kind=role, byte_sha256=byte_sha256, raw_rows=len(rows), rows=values, rows_sha256=digest(values))


def pack_corpus(sources):
    texts,packed={},[]
    for source in sources:
        refs=[]
        for row in source["rows"]:
            key=hashlib.sha256(row["text"].encode()).hexdigest()
            texts[key]=row["text"]
            refs.append(dict(id=row["id"],text_sha256=key))
        packed.append({**source,"rows":refs})
    return dict(schema="pd-exclusion-corpus-v1",texts=texts,sources=packed)


def unpack_corpus(value):
    require(value["schema"]=="pd-exclusion-corpus-v1","unknown exclusion corpus")
    require(all(hashlib.sha256(text.encode()).hexdigest()==key for key,text in value["texts"].items()),"reference text hashes changed")
    sources=[]
    for source in value["sources"]:
        rows=[dict(id=row["id"],text=value["texts"][row["text_sha256"]]) for row in source["rows"]]
        require(digest(rows)==source["rows_sha256"],"source reference membership changed")
        sources.append({**source,"rows":rows})
    return sources


def programme_audit(rows, sources):
    # Keep exact registered thresholds. Very long references can be discarded
    # only if both necessary edit-length and *unique* gram-count bounds fail.
    profiles = [p.TextProfile(r["text"]) for r in rows]
    maximum_length = max(v.length for v in profiles)
    maximum_grams = max(len(v.grams) for v in profiles)
    index, seen = p.DuplicateIndex(), set()
    for source in sources:
        for row in source["rows"]:
            normalized = p.normalize(row["text"])
            key = hashlib.sha256(normalized.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            profile = p.TextProfile(normalized, normalized=True)
            if profile.length > maximum_length/.9 and len(profile.grams) > maximum_grams/.8:
                continue
            index.add(row, profile)
    hits = []
    for row, profile in zip(rows, profiles):
        reason, match = index.find(profile)
        if reason:
            hits.append(dict(id=row["id"], reason=reason, match=match))
    return seal(dict(schema="pd-available-exclusions-v1", candidate_sha256=digest(rows),
                     source_count=len(sources), source_rows=sum(s["raw_rows"] for s in sources),
                     thresholds=THRESHOLDS, hits=hits, indexed_unique_references=len(index.ids),
                     scope="ALL_AVAILABLE_PROGRAMME_INPUTS", global_inventory_complete=False))


def topic_flags(rows):
    rules = read(KIT/"resources/pd/smoke-rules.json")["topic_patterns"]
    rules = dict(rules, chemistry_terms=r"\b(?:atoms?|molecules?|atomic|chemistry|chemical|polymers?|chloride|sodium|phosphorus|nitrogen|oxygen|carbon|hydrogen|radium|quicksilver|cryogenics|litmus|electrolysis|caffeine|cholesterol|nicotine|triglycerides|phenylalanine|propylene|acetaminophen|amoxicillin|amitriptyline|naproxen|antacids|quaaludes|enzymes?|carcinogen|acid rain|ph scale)\b")
    patterns = {k:re.compile(v) for k,v in rules.items()}
    flags = []
    for row in rows:
        reasons = [name for name, pattern in patterns.items() if pattern.search(p.normalize(row["text"]))]
        if row["label"] == "ENTY:substance":
            reasons.append("substance-answer-label")
        if reasons:
            flags.append({**row, "automated_reasons":sorted(reasons)})
    return seal(dict(schema="pd-scientific-topic-flags-v1", candidate_sha256=digest(rows),
                     patterns=rules, substance_label_rule=True, items=flags,
                     unflagged_policy="Human approval explicitly accepts the complete automated screening, including unflagged rows."))


def partition(rows, encoded, programme_hits, exclusions):
    require(set(encoded) == {r["id"] for r in rows}, "encoding coverage changed")
    blocked = {r["id"]:dict(reason=r["reason"], match=r["match"]) for r in programme_hits}
    accepted, decisions, index = [], [], p.DuplicateIndex()
    ordered = p.hash_order([r for r in rows if r["split"] == "test"])+p.hash_order([r for r in rows if r["split"] == "train"])
    for row in ordered:
        item_id = row["id"]
        value = encoded[item_id]
        reason, match = None, None
        if item_id in blocked:
            reason, match = blocked[item_id]["reason"], blocked[item_id]["match"]
        elif item_id in exclusions:
            reason = "approved_topic"
        elif max(value["candidate_lengths"]) > 1024 or len(value["input_ids"]) > 1024:
            reason = "sequence_cap"
        else:
            reason, match = index.find(row["text"])
        decisions.append(dict(id=item_id, reason=reason, match=match))
        if not reason:
            accepted.append(row)
            index.add(row)
    train = p.hash_order([r for r in accepted if r["split"] == "train"])
    test = p.hash_order([r for r in accepted if r["split"] == "test"])
    require(len(train) >= 1408, "DEFERRED: insufficient eligible train/calibration rows")
    parts = dict(calibration=[r["id"] for r in train[:128]], training=[r["id"] for r in train[128:1408]], evaluation=[r["id"] for r in test])
    return parts, decisions, accepted


def approval_exclusions(package, rows, flags, approval):
    require(approval.get("schema") == "pd-topic-approval-v1" and approval.get("status") == "APPROVED"
            and approval.get("package_sha256") == package["identity_sha256"]
            and approval.get("candidate_sha256") == digest(rows)
            and approval.get("topic_flags_sha256") == flags["identity_sha256"]
            and approval.get("complete_automated_screen_reviewed") is True,
            "DEFERRED: owner topic approval is absent, pending or bound to different inputs")
    require(approval.get("decision") in ("approve_automated_exclusions", "specific_overrides"), "DEFERRED: explicit owner decision required")
    overrides = approval.get("overrides", {})
    require(isinstance(overrides,dict) and set(overrides) <= {r["id"] for r in rows}
            and all(type(v) is bool for v in overrides.values()), "invalid topic overrides")
    require(approval["decision"] == "specific_overrides" or not overrides, "overrides need explicit specific_overrides decision")
    excluded = {r["id"] for r in flags["items"]}
    for item_id, exclude in overrides.items():
        excluded.add(item_id) if exclude else excluded.discard(item_id)
    return excluded


def candidate_supersets(sources, inventory_sha256):
    # References are normalized by source_from_rows. Stream the same canonical
    # JSON used by pd_contract.digest, avoiding several copies of a large corpus.
    payload=dict(schema="pd-candidate-supersets-v2",inventory_sha256=inventory_sha256,
        sources={s["id"]:sorted({r["text"] for r in s["rows"]}) for s in sources})
    h=hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).iterencode(payload):
        h.update(chunk.encode())
    return {**payload,"identity_sha256":h.hexdigest()}


def validate_prepared(root, package, prepared, approval):
    rows,encoded,flags,audit=(read(Path(root)/name) for name in ("rows.json.gz","encoded.json.gz","topic-flags.json","audit.json"))
    exclusions=approval_exclusions(package,rows,flags,approval)
    parts,decisions,accepted=partition(rows,encoded,audit["hits"],exclusions)
    selected={i for ids in parts.values() for i in ids}
    verify_seal(prepared);verify_seal(prepared["candidate_supersets"])
    require(prepared.get("schema")=="pd-preparation-v2" and prepared.get("scientific_status")=="FROZEN_APPROVED_NODE_VERIFY_REQUIRED"
        and prepared.get("scientific_package_sha256")==package["identity_sha256"]
        and prepared.get("scientific_approval_sha256")==digest(approval)
        and prepared["partitions"]==parts and prepared["rows"]==[r for r in rows if r["id"] in selected]
        and prepared["tokenized"]=={i:encoded[i] for i in sorted(selected)}
        and prepared["candidate_supersets"]["identity_sha256"]==package["candidate_supersets_sha256"]
        and prepared["source_hashes"]=={k:v["sha256"] for k,v in p.SOURCES.items()}
        and prepared["tokenizer_sha256"]==package["tokenizer_sha256"] and prepared["code_sha256"]==probe.code_identity(),
        "approved preparation differs from frozen source/encoding/partition inputs")


def load_package(root):
    root = Path(root)
    package = read(root/"preparation.pending.json")
    verify_seal(package)
    require(package["builder_sha256"]==sha256_file(__file__), "DEFERRED: scientific builder changed")
    require(package["code_sha256"] == probe.code_identity(), "DEFERRED: qualified core changed")
    for name, expected in package["files"].items():
        require(sha256_file(root/name) == expected, "scientific package bytes changed: "+name)
    return package


def build_package(cache, sp_cache, m3, output, review):
    from transformers import AutoTokenizer
    cache, sp_cache, m3, output = map(Path, (cache, sp_cache, m3, output))
    require(not output.exists(), "scientific package output must be fresh")
    rows = sum((p.verify_source((cache/"trec"/pin["name"]).read_bytes(), split) for split,pin in p.SOURCES.items()), [])
    provenance = p.pinned_provenance(cache/"trec/trec.py",cache/"trec/dataset_infos.json")
    pins = read(KIT/"resources/pd/smoke-public-pins.json")["model"]["files"]
    for name in ("tokenizer.json","tokenizer_config.json","vocab.json"):
        require(sha256_file(cache/"tokenizer"/name)==pins[name], "pinned tokenizer bytes changed")
    tokenizer = AutoTokenizer.from_pretrained(cache/"tokenizer",local_files_only=True)
    frozen_path = KIT/"resources/sp/FROZEN.json"
    frozen = read(frozen_path)
    specification = read(sp_cache/"pools.json")
    metadata = {pool["name"]:pool for pool in frozen["pools"]}
    require({x["name"] for x in specification["pools"]} == set(metadata), "SP available inventory coverage changed")
    sources, prompt_lookup = [], {}
    for item in specification["pools"]:
        path = sp_cache/item["path"]
        require(sha256_file(path)==metadata[item["name"]]["sha256"]==item["sha256"], "SP exclusion pool byte mismatch")
        data = jsonl(path)
        require(len(data)==metadata[item["name"]]["rows"], "SP pool row count changed")
        sources.append(source_from_rows("sp/"+item["name"],item["role"],data,item["sha256"]))
        for row in data:
            normalized = p.normalize(text(row.get("prompt",row.get("messages",""))))
            if normalized:
                prompt_lookup[hashlib.sha256(normalized.encode()).hexdigest()] = normalized
    superset_path=m3/"sp-candidate-superset-v1.json"
    superset=read(superset_path)
    for name, members in sorted(superset["pools"].items()):
        require(set(members)<=set(prompt_lookup), "K/CR candidate text unavailable: "+name)
        values=[dict(id=str(i),text=prompt_lookup[h]) for i,h in enumerate(members)]
        sources.append(source_from_rows("m3/"+name,"candidate-superset",values,digest(members)))
    for name in ("panel.jsonl","fixtures.jsonl","requests.jsonl"):
        path=KIT/"resources/sp"/name
        require(sha256_file(path)==frozen["files"][name], "SP frozen panel bytes changed")
        sources.append(source_from_rows("sp-panel/"+name,"panels",jsonl(path),sha256_file(path)))
    # Enumerate remaining versioned public programme panels. Already covered
    # files need no second audit, but their byte identities remain enumerated.
    known={s["byte_sha256"] for s in sources}
    for path in sorted((KIT/"panels").glob("*.jsonl")):
        sha=sha256_file(path)
        if sha not in known:
            sources.append(source_from_rows("core-panel/"+path.name,"panels",jsonl(path),sha));known.add(sha)
    # Locally available Chemistry Goal-1/probe/training material also belongs
    # to the exclusion union, even though it was absent from the SP staging set.
    chemistry=list(sorted((KIT.parent/"data").glob("chem*.jsonl")))
    chemistry += sorted((KIT/"resources").glob("*chemistry*/responses*.jsonl"))
    for path in chemistry:
        sources.append(source_from_rows("chemistry/"+path.relative_to(KIT.parent).as_posix(),"chemistry_goal1",jsonl(path),sha256_file(path)))
    audit=programme_audit(rows,sources)
    encoded={row["id"]:probe.encode_row(tokenizer,row) for row in rows}
    flags=topic_flags(rows)
    exclusions={r["id"] for r in flags["items"]}
    parts,decisions,accepted=partition(rows,encoded,audit["hits"],exclusions)
    output.mkdir(parents=True)
    for split,pin in p.SOURCES.items():
        dest=output/"sources"/pin["name"];dest.parent.mkdir(exist_ok=True);dest.write_bytes((cache/"trec"/pin["name"]).read_bytes())
    for name in ("trec.py","dataset_infos.json"):(output/"sources"/name).write_bytes((cache/"trec"/name).read_bytes())
    write(output/"rows.json.gz",rows);write(output/"encoded.json.gz",encoded)
    write(output/"available-corpus.json.xz",pack_corpus(sources))
    write(output/"audit.json",audit);write(output/"topic-flags.json",flags)
    missing=list(frozen["node_verification"]["unavailable_inventories"])
    missing.append(dict(inventory_id="chemistry-goal1",role="chemistry_goal1",format="jsonl",sha256=None,
                        description="Complete chemistry Goal-1 prompts/responses; explicit closed inventory required."))
    policy=seal(dict(schema="pd-scientific-node-policy-v1",required_inventories=missing,
        required_coverage=sorted(p.REQUIRED_COVERAGE),verification_only=True,deletion_allowed=False,rescoring_allowed=False,
        missing="DEFERRED before GPU discovery",overlap="UNVERIFIED; original frozen IDs retained; no measurement launch",
        confirmation=dict(mode="SEND_TIME_MANAGER_RULING", ruling_sha256=confirmation_ruling()["identity_sha256"],
                          fingerprint_file_required_at_send=False, future_recheck_required=True)))
    write(output/"confirmation-send-ruling.json",confirmation_ruling())
    write(output/"node-policy.json",policy)
    starts=seal(dict(schema="pd-send-chain-sources-v1",seeds=list(SEEDS),base_revision=probe.BASE_REVISION,
        roles=dict(base="pinned-Qwen3-8B",stage1="core/plain-D/FinQA/final",stage2="core/plain-D/FinQA/ContractNLI/final"),
        stage_steps=dict(base=0,stage1=40,stage2=40),recipe="plain-D",binding="send.pd_starts and send.pd_chain_sources_sha256; verify bind_chain and live tensor/receipt hashes on node"))
    write(output/"chain-starts.json",starts)
    inventory=seal(dict(schema="pd-available-inventory-v1",closed_available=True,closed_global=False,
        sources=[{k:v for k,v in s.items() if k!="rows"} for s in sources],
        bindings=dict(sp_frozen_sha256=sha256_file(frozen_path),sp_available_inventory_sha256=sha256_file(KIT/"resources/sp/available-inventory.json"),
                      m3_superset_sha256=sha256_file(superset_path),m3_candidate_sources_sha256=sha256_file(m3/"candidate-sources.json"),m3_input_binding_sha256=sha256_file(m3/"input-binding.json")),
        node_policy_sha256=policy["identity_sha256"]))
    write(output/"inventory.json",inventory)
    counts=dict(source_train=5452,source_test=500,programme_hits=len(audit["hits"]),topic_flags=dict(Counter(r["split"] for r in flags["items"])),
                proposed_eligible=dict(Counter(r["split"] for r in accepted)),proposed_partitions={k:len(v) for k,v in parts.items()},
                local_source_count=len(sources),local_source_rows=sum(s["raw_rows"] for s in sources),node_held_inventories=len(missing))
    package=seal(dict(schema="pd-scientific-package-v1",status="PENDING_OWNER_TOPIC_APPROVAL",scope="SCIENTIFIC_LOCAL_FREEZE_NODE_VERIFY_REQUIRED",
        code_sha256=probe.code_identity(),builder_sha256=sha256_file(__file__),source_provenance=provenance,tokenizer_sha256=probe.tokenizer_identity(cache/"tokenizer"),
        inventory_sha256=inventory["identity_sha256"],candidate_supersets_sha256=candidate_supersets(sources,inventory["identity_sha256"])["identity_sha256"],counts=counts,proposed_partitions=parts,
        encoding_measurements=dict(max_whole_sequence_tokens=max(max(v["candidate_lengths"]) for v in encoded.values()),
            max_prompt_tokens=max(len(v["prefix_ids"]) for v in encoded.values()),
            sequence_cap_exclusions=sum(v["reason"]=="sequence_cap" for v in decisions),
            internal_duplicate_exclusions=sum(v["reason"] in ("exact","edit_similarity","token5_jaccard") and v["id"] not in {x["id"] for x in audit["hits"]} for v in decisions)),
        files={f.relative_to(output).as_posix():sha256_file(f) for f in sorted(output.rglob("*")) if f.is_file()}))
    write(output/"preparation.pending.json",package)
    approval=dict(schema="pd-topic-approval-v1",status="PENDING",decision=None,package_sha256=package["identity_sha256"],candidate_sha256=digest(rows),
        topic_flags_sha256=flags["identity_sha256"],complete_automated_screen_reviewed=False,overrides={},
        instructions="Owner sets APPROVED, complete_automated_screen_reviewed=true, decision=approve_automated_exclusions or specific_overrides; overrides map original IDs to exclude booleans. No default is approval.")
    write(output/"topic-approval.template.json",approval)
    node_spec=dict(schema="pd-node-inventories-v1",package_sha256=package["identity_sha256"],ancestry_closed=False,complete_inventory=False,
                   coverage=[],inventories=[dict(inventory_id=x["inventory_id"],format=x["format"],path=None,sha256=x["sha256"],**({"producer_manifest":None} if x.get("producer_manifest_sha256") else {})) for x in missing])
    write(output/"node-inventories.template.json",node_spec)
    write(output/"PD-ROWS.json",seal(dict(schema="pd-scientific-rows-v1",package_sha256=package["identity_sha256"],
        rows=[dict(id="PD-"+str(s),seed=s,status="DEFERRED",phase="after core-submitted + PF-tier on node",requires=["sealed preparation","owner topic approval","node inventories/custodian verify-only receipt","send-bound original plain-D starts"]) for s in SEEDS])))
    write(output/"plan-inputs.template.json",dict(preparation="APPROVED_PREPARATION_JSON",
        scientific=dict(package="kit/resources/pd/scientific",approval="OWNER_APPROVAL_JSON",verification="NODE_VERIFY_JSON",send_manifest="SEALED_SEND_MANIFEST_JSON",inventories="NODE_INVENTORIES_JSON",custodian="CUSTODIAN_RECEIPT_JSON"),
        note="Merge into each source plan; keep existing checkpoints, exports, tokenizer and send_binding. Missing approval/verification defers before GPU discovery."))
    write(output/"confirmation-reference.template.json",dict(schema="pd-confirmation-prompts-v1",closed=False,months=["2026-10","2026-11"],rows=[],
        note="FUTURE CUSTODIAN RE-CHECK ONLY, after the 15-31 December slice build. Not required at send time. Custodian supplies complete id/text references without outcomes when the slice exists; independently bind its hash in the later re-check receipt."))
    review=Path(review);review.parent.mkdir(parents=True,exist_ok=True)
    lines=["# PD topic review — 29 October 2026", "", "Owner decision required; no approval or reserved-source closure is inferred.",
           "Scientific train AND test candidates are screened. The table includes every automated flag before topic exclusion; programme/duplicate/cap exclusions remain independent.",
           "", f"Candidates: {len(rows)} (5,452 train + 500 test). Automated flags: {len(flags['items'])}. Package: `{package['identity_sha256']}`.",
           "", "| ID | Text | Label | Automated reason |", "|---|---|---|---|"]
    for row in flags["items"]:
        clean=row["text"].replace("|","\\|").replace("\n"," ")
        lines.append(f"| {row['id']} | {clean} | {row['label']} | {', '.join(row['automated_reasons'])} |")
    lines += ["", "Fill a copy of topic-approval.template.json: approve all automated exclusions, or provide explicit per-ID exclude/retain overrides. Unflagged candidates are covered by the complete-screen approval. Finalize before inference; no post-outcome item changes."]
    review.write_text("\n".join(lines)+"\n")
    return package


def finalize(root, approval_file, output):
    root=Path(root);package=load_package(root)
    rows,encoded,flags,audit=(read(root/n) for n in ("rows.json.gz","encoded.json.gz","topic-flags.json","audit.json"))
    approval=read(approval_file)
    exclusions=approval_exclusions(package,rows,flags,approval)
    sources=unpack_corpus(read(root/"available-corpus.json.xz"))
    parts,decisions,accepted=partition(rows,encoded,audit["hits"],exclusions)
    selected={i for ids in parts.values() for i in ids}
    supersets=candidate_supersets(sources,package["inventory_sha256"])
    require(supersets["identity_sha256"]==package["candidate_supersets_sha256"],"frozen candidate superset changed")
    measurements=[dict(id=r["id"],prompt_tokens=len(encoded[r["id"]]["prefix_ids"]),maximum_label_eos_sequence_tokens=max(encoded[r["id"]]["candidate_lengths"]),training_sequence_tokens=len(encoded[r["id"]]["input_ids"])) for r in accepted]
    prepared=seal(dict(schema="pd-preparation-v2",source_revision=p.REVISION,source_provenance=package["source_provenance"],
        source_hashes={k:v["sha256"] for k,v in p.SOURCES.items()},code_sha256=probe.code_identity(),tokenizer_sha256=package["tokenizer_sha256"],
        prompt_sha256=digest(p.taxonomy()),partitions=parts,rows=[r for r in rows if r["id"] in selected],tokenized={i:encoded[i] for i in sorted(selected)},
        eligible_test_ids=parts["evaluation"],oversize=[r["id"] for r in decisions if r["reason"]=="sequence_cap"],encoding_measurements=measurements,
        exclusion=seal(dict(candidate_sha256=digest(rows),inventory_sha256=package["inventory_sha256"],topic_review_sha256=digest(approval),
                            custodian_status="NODE_VERIFY_ONLY_REQUIRED",global_inventory_complete=False,decisions=decisions)),
        candidate_supersets=supersets,no_replenishment=True,
        sequence_cap_measurements=dict(max_sequence_tokens=max(m["maximum_label_eos_sequence_tokens"] for m in measurements),max_prompt_tokens=max(m["prompt_tokens"] for m in measurements)),
        scientific_package_sha256=package["identity_sha256"],scientific_approval_sha256=digest(approval),scientific_status="FROZEN_APPROVED_NODE_VERIFY_REQUIRED"))
    write(output,prepared)
    return prepared


def _node_inventory(entry, requirement, base):
    path=Path(base)/entry["path"]
    require(entry.get("format")==requirement["format"],"node inventory format changed")
    expected=requirement.get("sha256")
    observed=sha256_file(path)
    require(is_sha(entry.get("sha256")) and entry["sha256"]==observed and (not expected or expected==observed),"node inventory hash changed")
    if requirement.get("producer_manifest_sha256"):
        manifest=Path(base)/entry["producer_manifest"]
        require(sha256_file(manifest)==requirement["producer_manifest_sha256"],"producer manifest binding changed")
        require(read(manifest).get("train_sha256")==observed,"producer train hash changed")
    if requirement["format"]=="parquet":
        import pyarrow.parquet as pq
        values=pq.read_table(path).to_pylist()
        for row in values:
            if isinstance(row.get("prompt"),str) and row["prompt"].lstrip().startswith("["):
                row["prompt"]=json.loads(row["prompt"])
    elif requirement["format"]=="journal":
        manifest=read(path);files=manifest["files"]
        names=sorted(n for n in files if n.startswith("teacher/raw/") and n.endswith(".json"))
        require(names and set(names)=={v.relative_to(path.parent).as_posix() for v in (path.parent/"teacher/raw").glob("*.json")},"journal coverage differs from producer manifest")
        values=[]
        for name in names:
            require(not Path(name).is_absolute() and ".." not in Path(name).parts,"unsafe journal member")
            member=path.parent/name
            require(sha256_file(member)==files[name],"journal member bytes changed")
            raw=read(member)
            # Producer journals commonly nest messages inside request.
            if isinstance(raw.get("request"),dict):
                raw={**raw, "messages":raw["request"].get("messages",raw.get("messages",[]))}
            values.append(raw)
    else:
        values=jsonl(path)
    require(values,"empty node inventory cannot establish closure")
    return source_from_rows(requirement["inventory_id"],requirement["role"],values,observed)


def custodian_receipt(root, package, prepared):
    """Bind the manager's ruling; explicitly retain the future comparison duty."""
    root=Path(root)
    ruling=read(root/"confirmation-send-ruling.json");verify_seal(ruling)
    policy=read(root/"node-policy.json");verify_seal(policy)
    require(ruling==confirmation_ruling() and policy.get("confirmation")==dict(
        mode="SEND_TIME_MANAGER_RULING",ruling_sha256=ruling["identity_sha256"],
        fingerprint_file_required_at_send=False,future_recheck_required=True),
        "DEFERRED: bound manager confirmation ruling required")
    verify_seal(prepared)
    require(prepared.get("scientific_package_sha256")==package["identity_sha256"]
            and prepared.get("scientific_status")=="FROZEN_APPROVED_NODE_VERIFY_REQUIRED",
            "DEFERRED: custodian receipt requires approved preparation binding")
    rows=read(root/"rows.json.gz")
    return seal(dict(schema="pd-custodian-send-receipt-v2",
        package_sha256=package["identity_sha256"],preparation_sha256=prepared["identity_sha256"],
        candidate_sha256=digest(rows),candidate_ids_sha256=digest(sorted(r["id"] for r in rows)),
        candidate_count=len(rows),node_policy_sha256=policy["identity_sha256"],
        ruling_sha256=ruling["identity_sha256"],statement=ruling["statement"],
        status="SEND_TIME_RULING_BOUND",confirmation_months=ruling["confirmation_months"],
        slice_exists_at_send=False,disjoint_by_construction=True,comparison_performed=False,
        future_recheck_required=True,recheck_status="PENDING_SLICE_BUILD",
        slice_build_window=ruling["slice_build_window"],thresholds=THRESHOLDS,
        reserved_outcomes_accessed=False,verification_only=True))


def validate_custodian(root, package, prepared, receipt):
    verify_seal(receipt)
    require(receipt==custodian_receipt(root,package,prepared),
            "DEFERRED: preparation-bound send-time custodian ruling receipt required")


def custodian_check(root, preparation_file, output):
    """No fingerprint input exists at send time; no comparison is fabricated."""
    root=Path(root);package=load_package(root)
    result=custodian_receipt(root,package,read(preparation_file))
    write(output,result);return result


def verify_node(root, preparation_file, approval_file, inventories_file, custodian_file, send_file, plan_file, output):
    root=Path(root);package=load_package(root)
    prepared,approval,spec,custodian,send,plan=map(read,(preparation_file,approval_file,inventories_file,custodian_file,send_file,plan_file))
    verify_seal(prepared);verify_seal(custodian);verify_seal(send)
    rows=read(root/"rows.json.gz");flags=read(root/"topic-flags.json");approval_exclusions(package,rows,flags,approval)
    require(prepared.get("scientific_package_sha256")==package["identity_sha256"] and prepared.get("scientific_approval_sha256")==digest(approval),"approved preparation binding changed")
    validate_prepared(root,package,prepared,approval)
    policy=read(root/"node-policy.json");required={x["inventory_id"]:x for x in policy["required_inventories"]}
    require(spec.get("schema")=="pd-node-inventories-v1" and spec.get("package_sha256")==package["identity_sha256"]
            and spec.get("complete_inventory") is True and spec.get("ancestry_closed") is True
            and set(spec.get("coverage",[]))>=p.REQUIRED_COVERAGE,"DEFERRED: complete node inventory/ancestry attestation required")
    entries=spec.get("inventories",[]);provided={x["inventory_id"]:x for x in entries}
    require(len(provided)==len(entries) and set(provided)==set(required),"DEFERRED: missing or extra node-held inventories")
    sources=[_node_inventory(provided[n],required[n],Path(inventories_file).resolve().parent) for n in sorted(required)]
    validate_custodian(root,package,prepared,custodian)
    starts=read(root/"chain-starts.json")
    require(set(send.get("pd_starts",{}))=={str(s) for s in SEEDS},"send must bind all three registered seeds")
    require(send.get("pd_chain_sources_sha256")==starts["identity_sha256"]
            and send.get("pd_custodian_receipt_sha256")==custodian["identity_sha256"],"DEFERRED: send-time chain/custodian source bindings required")
    require(plan.get("seed") in SEEDS and plan.get("send_binding")==send,"node plan does not carry the frozen send binding")
    chain=probe.bind_chain(plan["checkpoints"],send)
    require(all(v["seed"]==plan["seed"] and v["tokenizer_sha256"]==prepared["tokenizer_sha256"] for v in chain.values()),"chain seed/tokenizer mismatch")
    require(chain["base"]["recipe"]=="base" and all(chain[s]["recipe"]=="plain-D" and chain[s]["step"]==40 for s in ("stage1","stage2")),"only original plain-D final starts are allowed")
    # Tensor bytes are checked before GPU allocation as well as by the scorer.
    from kit.pd_contract import verify_live_export
    for start,checkpoint in chain.items():
        export=plan["exports"][start];directory=Path(export["directory"])
        verify_live_export(checkpoint,{f.name:f for f in directory.iterdir() if f.suffix in (".safetensors",".bin")},export["receipt"],directory=directory)
    audit=programme_audit(prepared["rows"],sources)
    affected=sorted({x["id"] for x in audit["hits"]})
    common=seal(dict(schema="pd-scientific-return-verification-v1",status="UNVERIFIED" if affected else "VERIFIED",
        preparation_sha256=prepared["identity_sha256"],candidate_supersets_sha256=prepared["candidate_supersets"]["identity_sha256"],
        package_sha256=package["identity_sha256"],approval_sha256=digest(approval),
        custodian_sha256=custodian["identity_sha256"],node_reference_sha256=digest([{k:x[k] for k in ("id","byte_sha256","rows_sha256")} for x in sources]),
        send_sha256=sha256_file(send_file),affected_ids=affected,verification_only=True,
        confirmation_comparison_performed=False,confirmation_recheck_status="PENDING_SLICE_BUILD"))
    result=seal(dict(schema="pd-scientific-node-verification-v1",status="UNVERIFIED" if affected else "VERIFIED",
        verification_complete=True,scoring_allowed=not affected,seed=plan["seed"],package_sha256=package["identity_sha256"],
        preparation_sha256=prepared["identity_sha256"],approval_sha256=digest(approval),send_sha256=sha256_file(send_file),
        checkpoints_sha256=digest(plan["checkpoints"]),exports_sha256=digest(plan["exports"]),
        custodian_sha256=custodian["identity_sha256"],node_inventory_sha256=sha256_file(inventories_file),
        affected_ids=affected,programme_audit=audit,return_verification=common,verification_only=True,deletion_allowed=False,rescoring_allowed=False))
    write(output,result);return result


def require_plan(plan):
    """Node entry gate: never discover a GPU for an unapproved scientific run."""
    try:
        config=plan["scientific"]
        root=Path(config["package"]);package=load_package(root)
        prepared=read(plan["preparation"]);verify_seal(prepared)
        approval=read(config["approval"])
        approval_exclusions(package,read(root/"rows.json.gz"),read(root/"topic-flags.json"),approval)
        validate_prepared(root,package,prepared,approval)
        verified=read(config["verification"]);verify_seal(verified)
        require(verified.get("schema")=="pd-scientific-node-verification-v1" and verified.get("status")=="VERIFIED"
                and verified.get("scoring_allowed") is True and verified.get("verification_only") is True
                and verified.get("verification_complete") is True and not verified.get("affected_ids")
                and verified.get("package_sha256")==package["identity_sha256"]
                and verified.get("preparation_sha256")==prepared["identity_sha256"]
                and verified.get("approval_sha256")==digest(approval)
                and verified.get("send_sha256")==sha256_file(config["send_manifest"])
                and verified.get("checkpoints_sha256")==digest(plan["checkpoints"])
                and verified.get("exports_sha256")==digest(plan["exports"])
                and verified.get("seed")==plan["seed"],"verification binding changed")
        require(prepared.get("scientific_package_sha256")==package["identity_sha256"]
                and prepared.get("scientific_approval_sha256")==digest(approval),"approved scientific preparation changed")
        require(read(config["send_manifest"])==plan["send_binding"],"send binding changed")
        require(sha256_file(config["inventories"])==verified["node_inventory_sha256"],"node inventory manifest changed")
        custodian=read(config["custodian"])
        validate_custodian(root,package,prepared,custodian)
        require(custodian["identity_sha256"]==verified["custodian_sha256"]
                and plan["send_binding"].get("pd_custodian_receipt_sha256")==custodian["identity_sha256"],"custodian receipt changed")
        spec=read(config["inventories"])
        requirements={x["inventory_id"]:x for x in read(root/"node-policy.json")["required_inventories"]}
        for entry in spec["inventories"]:
            _node_inventory(entry,requirements[entry["inventory_id"]],Path(config["inventories"]).resolve().parent)
    except (KeyError,TypeError,ValueError,OSError) as error:
        raise ValueError("DEFERRED: PD needs its sealed scientific preparation, explicit owner topic approval, complete node/custodian verification and send-bound plain-D starts") from None


def retain_verified_evidence(state, plan):
    """CPU sidecar adds the common closure proof; raw core evidence is retained."""
    state=Path(state);raw=state/"seed-evidence.json"
    require(raw.is_file(),"DEFERRED: completed core did not return seed evidence")
    require_plan(plan)
    evidence=read(raw);verified=read(plan["scientific"]["verification"])
    common=verified["return_verification"];verify_seal(common)
    prior=evidence.get("return_verification")
    if prior is not None:
        verify_seal(prior)
        require(prior.get("status")=="VERIFIED" and prior.get("preparation_sha256")==common["preparation_sha256"]
                and prior.get("candidate_supersets_sha256")==common["candidate_supersets_sha256"],
                "UNVERIFIED core returned-pool evidence cannot be promoted by the scientific sidecar")
    require(common["status"]=="VERIFIED" and common["preparation_sha256"]==read(plan["preparation"])["identity_sha256"],"return closure proof changed")
    evidence={**evidence,"return_verification":common,"raw_core_evidence_sha256":sha256_file(raw),
              "scientific_node_verification_sha256":verified["identity_sha256"]}
    write(state/"seed-evidence.verified.json",evidence)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    build=sub.add_parser("build")
    for flag in ("cache","sp-cache","m3","out","review"):build.add_argument("--"+flag,required=True)
    final=sub.add_parser("finalize")
    for flag in ("package","approval","out"):final.add_argument("--"+flag,required=True)
    cust=sub.add_parser("custodian")
    for flag in ("package","preparation","out"):cust.add_argument("--"+flag,required=True)
    verify=sub.add_parser("verify-node")
    for flag in ("package","preparation","approval","inventories","custodian","send","plan","out"):verify.add_argument("--"+flag,required=True)
    args=parser.parse_args()
    try:
        if args.command=="build":result=build_package(args.cache,args.sp_cache,args.m3,args.out,args.review)
        elif args.command=="finalize":result=finalize(args.package,args.approval,args.out)
        elif args.command=="custodian":result=custodian_check(args.package,args.preparation,args.out)
        else:result=verify_node(args.package,args.preparation,args.approval,args.inventories,args.custodian,args.send,args.plan,args.out)
        print(json.dumps({k:result[k] for k in ("schema","status","identity_sha256","counts") if k in result},sort_keys=True))
        return 0 if result.get("scoring_allowed",True) else 3
    except (ValueError,KeyError,OSError,TypeError):
        print(json.dumps(dict(status="DEFERRED",reason="scientific preparation/approval or bound verify-only evidence is missing or invalid")))
        return 3


if __name__=="__main__":raise SystemExit(main())
