# SP descriptive scoring sidecar

Plain scripts, submitted with the operator's own `sbatch`. No script submits a job.
Only seeds 101–103 are supported. SP never blocks parents, copies checkpoints, pins exports,
changes retention, trains, admits or vetoes an arm.

1. Run `plan.sh` on CPU with `KIT`, `SP_ROOT`, `LAUNCH` and optionally `M3_LAUNCH`.
   The immutable inventory includes future base/incoming/final coverage even before exports exist.
   SP(a) reuses existing raw outputs at every registered scored step. Single-cell core families
   (`finqa-plain`, `finqa-sibling`, `standalone-legal-plain`) contribute SP(b)/base descriptions only.
   S, PD forks and 4A are excluded. HellaSwag and knowledge-ll lengths are N/A.
2. Freeze SP hardware separately before scoring. Supply `SP_GPUS`, a local JSON mapping each
   inventory assignment to `{uuid, environment}`, and `SP_PARENT_HARDWARE`, the complete local
   parent hardware inventories, covering training GPUs as well as scoring GPUs, each attesting `complete_inventory: true`. `plan.sh` writes
   the write-once `hardware.json`. UUIDs must be distinct across SP assignments and free of parent holds while SP runs.
   Historical parent GPUs may be reused only after their inventory records `held: false`,
   `released: true` and a bound `release_receipt_sha256`. Supply current `SP_PARENT_HARDWARE`
   receipts on each run; reuse fails if a parent holds the GPU again or occupancy is unknown.
   GPU allocation must remain exclusive to the SP assignment for its whole scoring task.
   Missing environments or inventories prevent SP GPU work.
3. Submit each `run-task.sh` with `TASK` and `--dependency=afterany:...` for the parents listed in
   `sidecar.json`. Supply `SP_INPUTS` with all parent output roots and `SP_MODELS` with the local
   owner export map below. Incoming/base entries run before finals. The scorer resolves the frozen
   UUID before CUDA initialization and verifies the observed UUID and environment. `SCORE_GPU`
   is only a legacy CLI option; physical assignment determines the device.
4. Supply the complete, bound `sp-pools-v2` return specification as
   `returned-pools.json` under the SP root. `SP-verify-pools` invokes `sp_prepare.recheck()` on the
   frozen scientific panel. This specification is runtime evidence, not a shipped
   scientific input; its absence makes SP UNVERIFIED. Missing broad/K/CR roles, outside candidates or accepted-response
   overlaps make reports UNVERIFIED. Verification never deletes panel rows or requests rescoring.
5. Collect SP receipts/results independently, excluding `models.local.json`,
   `owner-exports.local.json` and all other local location maps. `SP-read` verifies returned manifests, reports every missing
   planned entry, strict/instruction/family/wording summaries, stage-specific gap changes,
   three-seed intervals, registered K−C0/CR−G and X contrasts, pair bootstraps and length metrics.
   Missing seeds remain INCOMPLETE; GPU mismatches remain UNPAIRED-GPU. All results are descriptive.

`plan.sh` optionally accepts `SP_RETURN1`, the core return-1 binding or reviewed decisions.
Otherwise it reads the core's pinned decisions. Only packages present in the launch are expanded,
including retained calls. Missing retained raw text remains missing. A programmatic inventory
without a package binding retains an unresolved runtime entry and an unknown upper price;
it must be rebuilt with exact calls before freezing for execution.

SP(b) is priced at 0.25–1 GPU-h per planned entry plus startup, without dropping unresolved cells.
Runtime identity binding writes separate immutable entry receipts and never changes frozen coverage.
Identical weights/panel/prompts/decoding/hardware reuse verified SP outputs. SP(a) uses zero new
GPU work and freezes incoming-correct populations, including initially incorrect items in the full
panel denominator. Missing response text, finish metadata or grades cannot shorten that denominator.

The CPU-only `durable_adapter.py` now produces `sp-durable-export-v1` receipts from native
owner artifacts. Set `SP_OWNER_EXPORTS` to a local JSON source map keyed by inventory entry ID.
`plan.sh` adapts currently available exports; every GPU task reruns the adapter for its entry
before binding, so future exports are resolved after their parents finish. `SP_MODELS` defaults
to the generated `models.local.json`; existing manually supplied adapter maps remain supported.

A core source object has `kind: "m2-snapshot"`, `durable: true`, `durable_root`, `model`,
`archive`, `export_prefix` (its model directory inside the snapshot file inventory), and
`owner_receipt_member` (the existing export receipt inside that inventory). The adapter reads the
native complete `message2-stage-snapshot.v2` header, existing `.tar.sha256` and
`.tar.manifest.sha256` companions and the original export receipt. It writes metadata only;
it never extracts the archive or makes a live export. A live owner export must already survive.

A K/CR source object has `kind: "m3-export"`, `durable: true`, `durable_root`, `model`, and
`owner_receipt` identifying the existing `export-40.json`. The adapter requires the owner's
unchanged run `manifest.json` to bind that receipt, `complete.json` to report `complete: true`
and `receipt_status: "PASS"`, and the original `hf-step40` export to survive in place. No tar
snapshot is invented for this native completed-export format.

`durable: true` attests the supplied location is the owner's existing durable storage, including
for native K/CR exports. Adapter receipts bind the original immutable metadata and model hashes.
Retention remains owner-controlled: adapters read surviving completed artifacts only, honor every
existing declared expiry and never ask the owner to keep anything longer. An optional `expires_at`
in a local source may shorten the window; it cannot extend an artifact's deadline. Null means
no expiry was declared in the owner artifacts, and never guarantees continued availability.

Only runtime `entry` binding fully hashes a snapshot archive. Model substitution and `panel_job`
recheck its sealed hash proof, file fingerprint (including ctime), native checksum/header/owner
receipts and continued availability. A changed fingerprint invalidates the entry instead of
rehashing on the GPU. Live model bytes are still checked before inference. All returned adapter
receipts contain portable hashes only; generated source/location maps stay local and must be
excluded from returns. No adapter writes any owner artifact, restores or copies a checkpoint,
creates a lease, pins an export or changes retention.

Planned aliases share an SP entry and price with their registered target: K/CR stage-N incoming
uses the preceding final (stage 1 uses base), and guaranteed X incoming aliases use pre. Every
arm/stage comparison use remains present in the inventory and reports.

Wall-time totals include planned missing SP cells, parent scoring cells and other registered
scoring work, by assignment and arm/seed. Missing load/score/receipt timings are UNMEASURED.
Measured lower totals exceeding 24 hours carry WALL-TIME RISK even if other timings are missing.
Timing never gates a result. PD timing is separate. No paid or remote execution is performed by
these wrappers.
