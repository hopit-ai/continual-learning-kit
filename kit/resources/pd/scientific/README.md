# Scientific PD preparation: owner approval plus node verification

This package is SCIENTIFIC preparation data, separate from SMOKE-SCOPE.
`preparation.pending.json` seals all pinned sources, encodings, local exclusion
references, source inventories, source provenance, node policy and proposed
partitions. Every data-file byte SHA256 is in its `files` map. Its core hash is
the passed smoke hash; the scientific helper has a separate builder hash.
`finalization-pin.json` separately binds the approved output and materializer; it
stays outside that original manifest to avoid changing the approved identity.
No model inference or scientific outcomes were used to build these inputs.

`PD-ROWS.json` declares PD-101/102/103 **DEFERRED** until their final sealed
preparation matching `finalization-pin.json`, committed owner approval,
node/custodian verification and send-bound
original chain starts are present. The node wrapper enforces this before GPU
discovery. The combined inventory owner must preserve these prerequisites;
absence is not permission to launch a smoke-scope or unreviewed scientific job.

## Available inventory and source bytes

The local exclusion union includes every staged SP exclusion pool, all eight
K/CR candidate groups reconstructed from the same text with exact membership
against the frozen K/CR superset, the SP scoring panel/requests/fixtures, all
other versioned public panel files and locally available chemistry training,
heldout/probe and yardstick response inputs. `inventory.json` enumerates each
source, raw row count, normalized text digest and byte identity. Aliased source
rows can repeat across inventories; the auditor indexes unique text once.
The compressed corpus is lossless, with shared text objects referenced by hashes.

The TREC sources are the exact registered CogComp training and test bytes:
5,452 train and 500 test rows. `sources/trec.py` and `dataset_infos.json` bind
these checksums to HF revision `1f97567bdd2adedefe8abdaa9bd6ee0e6725b458`.
The original-byte correction, taxonomy, threshold functions, tokenization and
hash ordering come from the unchanged qualified PD core. Encodings for every
candidate are already frozen; finalization needs no model, tokenizer or network.
The tokenizer bytes match the smoke-bound Qwen3-8B tokenizer identity.

No unavailable ancestor inventory is labelled globally closed. The 13 node-held
requirements in `node-policy.json` carry all known independent byte hashes:
three B replay files, five historical FinQA target files, ContractNLI prepared
rows and teacher journal, complete programme ancestry and panel inventories,
and any additional chemistry Goal-1 material outside the available local set.
Historical target artifacts do not add scientific seeds: only 101–103 run.

## Approved topic decision and deterministic node finalization

The owner approved all automated topic exclusions on 29 October 2026. The manager
committed the decision in `topic-approval.json`: 130 exclusions (85 train, 45 test),
no overrides, and explicit approval of the complete automated screen. The table
in `docs/phase3/pd-topic-review-20261029.md` is an our-side audit document recording
the reviewed candidates; it is not shipped. The committed approval ships.
`topic-approval.template.json` remains an unapproved template for audit/tests;
use the committed approval, not that template. Different or pending approvals fail.

The approved final preparation identity is:

`d3ad575f52b1203e210098c37584cb4a80db4f3a44508a5babd52fd121a1a7fa`

The full JSON is 508,919,629 bytes and is deliberately absent from the kit.
`finalization-pin.json` binds that identity, the exact output byte SHA256
`a7e50f1c2ebc4ac918c7704748e0d777d74783002715d52ae58b9b4fac13463c`,
the original package/builder, approval and materializer. Two independent CPU
finalizations produced identical complete bytes; the small sealed
`finalization-verification.json` records both runs.

Before node inventory verification or GPU discovery, materialize it once into
node-local storage outside the shipped package:

```sh
python kit/pd_materialize.py \
  --package kit/resources/pd/scientific \
  --approval kit/resources/pd/scientific/topic-approval.json \
  --out APPROVED_PREPARATION_JSON
```

The wrapper invokes the unchanged frozen builder with exactly this command:

```sh
python kit/pd_scientific.py finalize \
  --package kit/resources/pd/scientific \
  --approval kit/resources/pd/scientific/topic-approval.json \
  --out STAGED_NODE_PREPARATION_JSON
```

It publishes APPROVED_PREPARATION_JSON only when the returned `identity_sha256`,
complete output SHA256 and byte count match the pin. Success returns
`status: "PIN_MATCH"`; mismatch/failure returns exit 3/DEFERRED and leaves no
published preparation. Use a fresh destination; no existing file is overwritten.
All inputs are shipped; no tokenizer, model, network or GPU is needed. Allow
node-local CPU memory and disk for the full JSON plus temporary finalization
copies. This unmeasured CPU cost is not a timing gate. Do not transfer the large
JSON in the kit or place it inside the scientific package.

The original builder is deliberately unchanged: its hash is already in the
approved package and changing it would change the manager's approved preparation
identity. Pin enforcement lives outside both the frozen builder and the smoke's
hashed core. `pd_node.py` independently rejects a plan whose fully validated
preparation identity differs from the pin, before GPU discovery; its bound plan
records the pin identity. Directly invoking the builder does not replace this gate.

The approved preparation remains `FROZEN_APPROVED_NODE_VERIFY_REQUIRED`, with
truthful `global_inventory_complete: false` until node verification succeeds.
Use the same materialized preparation for all three seeds. Never delete, replace,
replenish or rescore rows after outcomes. Finalization introduces no new decision.

## Send-time custodian ruling and December re-check

Manager ruling, 29 October 2026, sealed in `confirmation-send-ruling.json`:

> no confirmation slice exists at send time; TREC (public question-type classification data) is disjoint from the SEC-filing/legal/maths confirmation sources by construction; the custodian re-checks PD prompts against the slice when it is built

The custodian builds the October–November slice on 15–31 December. No
confirmation fingerprint file is required at send time. This source separation
ruling permits the send-time custodian gate; it does not assert that a comparison
has occurred. `confirmation-reference.template.json` is retained solely for the
future custodian re-check, without reserved answers, scores or outcomes.

The committed approval and pinned preparation make the custodian receipt
deterministic. `custodian-send-receipt.json` is already shipped, generated on CPU
from the actual matched preparation. Its sealed identity is
`5e86c73c37af273a6f8380c140703cf13b3f4fac7ffd11417bbc5c4842d2efdc`.
Our-side send assembly binds that value as `pd_custodian_receipt_sha256`.
The node can use the shipped receipt or recreate it after materialization:

```sh
python kit/pd_scientific.py custodian \
  --package kit/resources/pd/scientific --preparation APPROVED_PREPARATION_JSON \
  --out CUSTODIAN_RECEIPT_JSON
```

The sealed receipt binds the scientific package, approved preparation, all 5,952
candidate prompts/IDs, node policy and exact manager ruling. It records
`comparison_performed: false`, `future_recheck_required: true` and
`recheck_status: "PENDING_SLICE_BUILD"`. Bind its `identity_sha256` in the send
manifest as `pd_custodian_receipt_sha256`. The node accepts this receipt without
fingerprints and rejects changed commitments or a receipt from another preparation.
The returned verification sidecar also records the pending December comparison.
No all-false empirical match decisions are invented. The custodian re-checks all
frozen PD candidate prompts against the slice when built, using the registered
thresholds. No post-outcome deletion, replenishment or automatic rescoring follows.

The committed owner approval and all 13 node-held inventory requirements remain mandatory.

## Send-time original chain bindings

`chain-starts.json` freezes symbolic source roles: pinned Qwen3-8B; the exact
original plain-D FinQA final used as C's sequence-D parent; and that same chain's
plain-D ContractNLI step-40 final. No standalone C, replay sibling or alternate
route is allowed. This is a binding to core exports, not chain retraining.

The shipped sealed `send-binding.json` includes:

- `pd_chain_sources_sha256`: the sealed identity of `chain-starts.json`;
- `pd_custodian_receipt_sha256`: the sealed identity of the preparation-bound
  send-time custodian ruling receipt (generated after owner topic approval);
- `pd_starts["101"|"102"|"103"]["base"|"stage1"|"stage2"]`: the existing core
  PD contract's eight immutable start fields. Bind exact checkpoint IDs,
  tensor/export-receipt hashes, parent links, recipe, step, base revision and
  tokenizer. Base logical IDs are seed-specific; both descendants use `plain-D`.

The identities come from `package-c-return.tar.gz` (archive SHA256 and member
hashes are in the binding). Its `return/binding.json` records the original v4
plain-D FinQA parents, their complete tensor hashes and original export receipt
hashes for seeds 101–103. Each sequence-D summary confirms that exact incoming
parent, recipe D, sequence route, 40 steps and zero replay. The ContractNLI
`export-40.json` bytes match their returned run manifests. Omitted tensors count
only as recorded identities: no live verification has occurred here.

For base, the existing shipped `kit/resources/p3b/model-hashes.json` is the
receipt: use that file unchanged as `exports.base.receipt`. Its student tensor
hashes match the returned base registration. Base has no training export receipt;
the PD contract accepts this exact registration file by byte SHA256. For stage1,
resolve the original `export-identity-step40.json` with the recorded SHA256;
for stage2, resolve the original sequence-D `export-40.json` bytes. Redacting or
reserializing any receipt changes its identity. Original private receipts stay
node-local and are not copied into this package.

Copy the complete `send-binding.json` object into each plan's `send_binding`.
Build sealed chain-start checkpoints from its eight fields plus `seed`, `start`,
`tensor_files` from the binding, and `send_binding_sha256` equal to the binding's
`identity_sha256`. Checkpoint IDs are portable logical aliases; retain them and
their parent links exactly. Use the pinned base tokenizer for all three starts,
whose tokenizer identity matches the approved preparation. Point
`scientific.send_manifest` at this same file using `plan-inputs.template.json`.

Node verification checks the send binding and every live tensor/receipt byte
before any scientific run; the unchanged scorer checks them again before loading.
An absent exact export is DEFERRED. The partner supplies local resolvers for
already-bound artifacts, not new endpoint or ancestry decisions.

## Node inventories: verify only, before each seed

Fill a node-local copy of `node-inventories.template.json`: all local locations,
independent hashes, required producer manifests, exact inventory IDs,
`complete_inventory: true`, `ancestry_closed: true`, and every `required_coverage`
category listed in `node-policy.json`. Known hashes cannot be overridden.
For a journal, every producer-declared `teacher/raw/*.json` must exist with the
right hash and no undeclared members. Parquet is read as data. No archive code
or pickle is executed. Public and private prompt/response inventory closure
must be complete, including accepted/rejected pools and ancestry; reserved
outcomes stay in the custodian scope.

Merge `plan-inputs.template.json` into each original per-seed plan. Preserve its
core checkpoints, export resolvers, tokenizer and exact `send_binding`. Supply
all paths in `scientific`, including inventories and custodian receipt. Then:

```sh
python kit/pd_scientific.py verify-node \
  --package kit/resources/pd/scientific --preparation APPROVED_PREPARATION_JSON \
  --approval kit/resources/pd/scientific/topic-approval.json --inventories NODE_INVENTORIES_JSON \
  --custodian kit/resources/pd/scientific/custodian-send-receipt.json \
  --send kit/resources/pd/scientific/send-binding.json \
  --plan SEED_PLAN_JSON --out NODE_VERIFY_JSON
```

Missing, changed or incomplete evidence blocks the run with exit 3/DEFERRED.
An overlap returns UNVERIFIED with implicated frozen IDs and does not permit
scientific launch. No item is removed and no split is rebuilt. If outcomes
already exist, preserve them as unverified; do not automatically rescore.
The gate rechecks inventory bytes and custodian binding before GPU discovery.
The additional source hashing has unmeasured CPU/I/O cost; it is not a timing gate.

Once every gate is complete, the operator's ordinary plain-script step is:

```sh
python kit/pd_node.py --plan SEED_PLAN_JSON --bound-plan BOUND_PLAN_JSON \
  --out PD_STATE --score-visible-index 0
```

No new sbatch wrapper, hostname, account or local source location is shipped.
This document does not authorize a launch. It describes the final gated step.

Preserve the complete PD tree and both `seed-evidence.json` (raw qualified-core
output) and `seed-evidence.verified.json` (CPU sidecar carrying the common closed
inventory/send-time custodian proof, with December re-check pending), bound plan
and supervisor receipts. The reader
consumes the three verified sidecars; all three must bind the same common
closure inputs. No score, metric or denominator is changed by the sidecar.

## Optional our-side rebuild

The source cache and staged SP pools stay outside Git. This rebuild is CPU-only;
the existing frozen package makes a rebuild unnecessary after topic approval:

```sh
python kit/pd_scientific.py build --cache PD_CACHE --sp-cache SP_POOL_CACHE \
  --m3 M3_RESOURCES --out FRESH_PD_PACKAGE --review TOPIC_REVIEW_MD
```

Caches must contain the exact pinned TREC/provenance and tokenizer bytes and
SP pools bound by `FROZEN.json`; the K/CR superset must have complete materialized
prompt coverage. Build rejects any mismatch. It never downloads or infers.
