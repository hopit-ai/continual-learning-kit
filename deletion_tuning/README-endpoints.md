# Deletion endpoints and returns

These adapters import the unchanged v7b freeze scorer, item reader, count reader,
and return arithmetic. The fixed registration remains authoritative. No trainer,
export merger, incumbent selection, protocol list, or pinned kit file changes.
`queued-insert.txt` contains the replacement queued-message lines, all below 90
characters. Public kit identity is supplied explicitly; no private identity is
inferred from the working checkout.

## Run the independent rows

Run from the checkout with offline model/data assets already available:

```sh
python -m deletion_tuning.deletion_jobs inventory --out rows.json
export DELETION_ROOT="$PWD"
export DELETION_CONFIG=deletion.json
# Submit using your own sbatch options. CPU preparation requests zero GPUs.
deletion_tuning/jobs/run-task.sh prepare-101-legal
# After that preparation succeeds, request exactly one GPU for this row.
deletion_tuning/jobs/run-task.sh score-101-legal
```

There are 21 complete score lanes: seeds 101/102/103 times instruction, maths,
knowledge, hellaswag, finqa, legal, and legal-diagnostic. Each has an independent
CPU preparation row. Submit all ready preparations concurrently, followed by
all ready scoring rows. There is no whole-study, cross-seed or cross-panel
barrier. A lane requires only its own base/incoming and C/T export frontier;
legal and its separate diagnostic use 0/10/20/30/40, other panels 0/20/40.
Export with the existing deletion hard-mask merger; T step zero is hard masked.
C step zero must have the untouched incoming tensor/config digest.

CPU preparation verifies export bytes with `nm_prepare.verify_model`, renders
prompts with `nm_prepare.render_scoring`, audits the bound panel sources and
exclusions with `nm_prepare.audit_bundle`, and freezes the scoring plans. No
checkpoint hashing, merging, prompt tokenization or export waiting runs on a
score GPU. Export paths must already refer to CPU-staged immutable HF folders.
No source lookup, download, scheduler submission or scheduler polling is added.
Prepare/score failures exit 75 for pending work and 2 for unverified evidence.

`nm_score.py` scores every cell. Legal diagnostics use the same tolerant legal
parser, with their own bound items and separate lane. The inherited lease and
`lane_pool.execute` enforce one physical GPU. Every cell must retain one nonce,
UUID and environment for that seed/panel. A retry publishes a fresh complete
attempt; no resume/stitching across allocations. The latest failed attempt
hides earlier successes. The pool adds zero startup stagger. Within a lane,
default concurrency is one until that exact worker has numerical and memory
qualification. The cap is two for knowledge/HellaSwag and three otherwise.
Independent lanes can always run on separate GPUs in parallel. Utilisation and
wall time are observations, never endpoint gates.

## Input contract

The JSON config has `out` (absolute local output directory), `kit_commit` (40
lowercase hex public commit), and four `{path, sha256}` input references:

- `bundle`: the inherited NM bundle with base, tokenizer, panels,
  panel_authorities, candidate_superset, excluded_superset, source_revisions,
  licenses and holdout_authority. Include the separate `legal-diagnostic`
  panel and its hash-bound authority. General, HellaSwag and legal test contain
  100 items each; FinQA contains 1,147. Training exclusions include both legal
  document groups. `audit_bundle` performs its original source/data checks.
- `binding`: the READY, non-benchmark `nm_bind.row_binding('freeze', ...)`
  projection of the entire already resolved incumbent, including exact
  `recipe.starts` and `recipe.schedule_hashes`. This adapter does not resolve
  or authorize a different incumbent.
- `lr_receipt`: the frozen production `deletion-lr.v1` receipt. Its existing
  `deletion_lr.frozen_rate` validator checks selection and SHA; no endpoint
  participates in selection. Qualification must freeze it before science.
- `evidence`: the hash-bound production qualification/measurement document
  described below. Missing qualifications produce PENDING/INCOMPLETE.

The bundle adds `exports[seed][arm]`, each a `{path, sha256}` reference to a JSON
manifest mapping string steps to HF model records. Each record contains
`model`, `sha256` (the inherited tensor/config digest), `seed`, `arm`, `step`;
T records also require `hard_masked: true`. They are standard dense exports,
not native resumable checkpoints. Manifests may be supplied as soon as a seed's
exports exist, without waiting for other seeds. Keep config references fixed
for all attempts being collected; start a fresh output root if they change.

Optional `bundle.score_qualifications[panel]` is a `{path, sha256}` reference.
It contains `binding_sha256` equal to the CPU lane's `identity` digest,
`numeric_match: true`, `memory_safe: true`, and integer `concurrency` within
that panel's cap, plus measured `uuid` and `environment`. Scoring checks those
against the actual reserved worker before enabling concurrency. The identity
includes the seed, panel population, scorer,
whole incumbent binding, and deletion implementation bytes. A qualification
for a different seed/worker binding cannot enable concurrency. Create the
identity with a serial CPU preparation, qualify the worker externally, then
prepare a fresh run root with that qualification. Input bundle location/hash
is not part of the qualification identity, preventing a circular hash.

The evidence document contains:

- `qualification: "COMPLETE"`, `recipe_sha256`, `parents_sha256`, and
  `schedules_sha256` (canonical `nm_common.digest` of the entire recipe, its
  starts, and its schedule_hashes), plus `lr_sha256` and
  `lr_frozen_before_science: true`.
- `seeds[seed]` with true fields `paired_schedule`, `dense_immutable`,
  `step0_hard`, `resume_equal`, `component_qualification`, `data_isolation`,
  and `rate` equal to the frozen finder selection. These are externally
  measured production qualification results, never inferred from trainer exit.
- Each seed's `canary`: `n`, `ids_sha256`, and accuracy percentages `incoming`,
  `initial_soft`, `initial_hard`, `teacher`, on the same frozen training or
  development canary. Additional losses and intermediate measurements remain
  in FULL. Nothing in these measurements changes LR selection or the bars.
- Each seed's `masks[step][layer]`: integer `weights`, `hard_zeros`,
  `keep_to_delete`, `delete_to_keep`, for every layer at all five exports.
  Flips are relative to initialized hard bits. Layer inventories, populations,
  and zero/flip arithmetic must agree; initialized flip counts are zero.
  Additional soft attenuation measurements are retained unchanged in FULL.
- `accounting`: `status`, `gpu_hours`, `all_attempts: true` and the complete
  allocation ledger. Use actual scheduler allocation starts/ends, including
  failed/retried training, qualification/LR finder, scoring and reserved GPUs.
  Process time in row receipts is diagnostic and is never substituted for
  allocated time. Missing actual allocation accounting is reported PENDING.
- `changes`: nonempty list of privacy-safe uppercase codes, e.g. `["NONE"]`.
  Codes use A-Z, digits, underscore/hyphen, at most 48 characters each.
  Put detailed forced changes in the local document, without pasting paths,
  machine identities or execution identifiers.

The shipped LR validator requires model_geometry `[4096,36]`, matching the
Qwen3-8B hidden size and layer count checked by the inherited HF verifier.
The endpoint adapters reuse both validators with this consistent geometry.

## Collect and read on CPU

```sh
python -m deletion_tuning.deletion_return collect \
  --config deletion.json --out deletion-return
python -m deletion_tuning.deletion_return read \
  --text deletion-return/return-{1,2,3}.txt --out manager-reading.json
```

Use a fresh collection directory for every collection. Paste all three top-level
`return-N.txt` files verbatim. Each has at most 74 lines, each line below 90
ASCII characters. Part 3 has the queued DELETION keys: kit, recipe, parents,
schedules, LR receipt SHA, selected rate/fallback, actual allocation cost,
and forced-change codes. SHA checks are advisory; every reading is marked
TRANSCRIPTION-UNVERIFIED. Changing only a checksum does not reject a result.
Missing pages/cells/qualification are pending, identity errors unverified,
mixed hardware unverified-hardware, and non-finite scientific evidence triggers
a correctness stop with no decision. Valid complete evidence is PASS or
INCONCLUSIVE. Technical stand-ins cannot produce a scientific PASS.

The manager recomputes every bar using `nm_text.count_endpoints` and `recompute`
for freeze: df=2, t=4.302653, mean >=2, all three seed effects positive and lower
bound >0; all individual guards lower bound >=-1. Exact correct counts preserve
the untouched incoming reference and T's damaged step-zero trapezoid. General
absolute-final and retention contrasts cancel the same common base reference,
so their identical guards are computed once, exactly as in v7b. Likewise FinQA
and legal final/gain guards retain v7b's named duplicate contrasts. No composite
can replace an individual guard. Package summaries remain separate and use the
inherited one-sided arithmetic where applicable; no incumbent changes here.

The local FULL set includes item vectors, spec/binding, complete qualification
and per-layer masks, LR receipt, item-based reading (including package summaries,
answer transitions and acquired-answer survival), and a manifest of every raw
attempt file, including retries. Original raw scorer responses and receipts stay
in the attempt tree. Tokens/correct and the >1.5x base "at cost" diagnostic are
included where generation token observations exist; reference-only outputs
without token observations remain UNMEASURED. Initial full-panel scores and
step-20 diagnostics are retained. If the supporting step-20 transport records
exceed the ESSENTIAL allowance they remain local in FULL. The full LR receipt
and per-layer records stay local; ESSENTIAL carries its SHA and trial summaries.
The manager's pasted-count arithmetic does not independently verify raw model,
qualification or item provenance; FULL supports that separate audit.
