# Scientific freeze and node VERIFY-ONLY contract

The manager authorizes freezing the complete locally available set: 23 pools,
98,273 rows, zero exclusion hits and 200 inspected five-nearest lists. The
historical available audit and author receipts retain their truthful global
incompleteness labels. `FROZEN.json` records the authorized local scope and embeds
`node_verification`, with every unavailable inventory identified explicitly.
The frozen panel has 200 requests and 400 prompts. Every pair stays in the panel.

The 12 deferred inventories are three B generic replay files, five historically
bound B FinQA target files, ContractNLI prepared rows and its teacher journal,
and complete programme ancestry and panel inventories. Historical target-file
coverage adds no scientific seeds; scoring remains seeds 101–103 only. Known
byte hashes are mandatory. Unknown node-held inventory hashes are bound by the
node-local manifest before scoring; ancestry completeness must be attested.

Create a node-local copy of `node-pools.template.json`. Set all local locations,
file hashes, `complete_inventory: true`, `ancestry_closed: true`, the byte SHA256
of `FROZEN.json`, and `policy_sha256 = sp_prepare.digest(frozen['node_verification'])`.
For a journal, supply the producer's manifest; all `teacher/raw/*.json` members
must be present, hash-matched and exactly enumerated. FinQA parquet rows are read
as data, extracting prompts and `extra_info.demonstration`; no source code or
model is loaded. Include any rebuilt broad/K/CR returned manifest through
`returned_pools_spec`; its frozen-parent containment is also verified.

Before each SP scoring task, use this owned wrapper (all capitalized arguments
are node-local placeholders):

```sh
python -I kit/resources/sp/node_verification.py run \
  --frozen FROZEN_DIRECTORY --pools NODE_INVENTORIES_JSON \
  --receipt SP_ROOT/pool-verification.json -- \
  python kit/sp_manifest.py run SIDECAR_TASK_ARGUMENTS
```

The wrapper calls the same `sp_prepare.recheck()` API already used by
`sp_manifest.py verify-pools`. It runs the identical frozen exclusion functions
before invoking the supplied scoring command. Missing, malformed or mismatched
inventory evidence prevents that command from running. A completed check with
an overlap still permits the original panel to be scored once: both wording
arms of only the implicated request IDs are marked UNVERIFIED in the receipt.
No pair is deleted, rewritten, replenished or rescored.

`sp-recheck-v2` retains the legacy top-level `status` (`CLEAN` or `UNVERIFIED`)
and adds `verification_complete`, `scoring_allowed`, `affected_request_ids`,
`unverified_request_ids`, `unverified_prompt_ids` and `pair_status`. Consumers
must use `scoring_allowed`, not top-level `status`, for the pre-scoring gate.
`affected_request_ids` contains localizable hits only; unresolved identity or
coverage prevents verification of all pairs without inventing pair-specific hits.

For the returned report, preserve all original scores and denominators and call:

```sh
python -I kit/resources/sp/node_verification.py report \
  --frozen FROZEN_DIRECTORY --receipt SP_ROOT/pool-verification.json \
  --sidecar SP_ROOT --out REPORT_DIRECTORY
```

This wrapper uses `sp_read`'s existing table functions, emits per-pair labels,
and explicitly reports whether full-panel statistics include unverified pairs.
It does not mark unrelated pairs or every comparison family UNVERIFIED because
of a single overlap. `annotate(scored, verification)` is the corresponding
in-memory adapter and preserves score values, row order and membership.

Launcher/reader ownership remains with m3b2. The existing direct `sp_read` CLI
has a family-wide fallback and must be routed through this adapter or updated
by its owner. The existing direct runner must invoke the pre-scoring wrapper.
The scoring implementation need not change, preserving its frozen code hash.
