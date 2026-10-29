# PD v2 — final-registration implementation

Seeds are exactly 101–103. This isolated tier never trains a chain, writes to
replay, extends core checkpoint retention, or blocks core execution. SP excludes
PD forks. No model/dataset acquisition or remote execution occurred during build.
The final owner-approved registration governs. Attempt 3 qualified the technical
smoke; scientific preparation and scientific measurement remain pending.


## Scientific preparation and combined-send B4

PD-101/102/103 are **DEFERRED** unless node-local finalization matches the pinned
approved preparation, the committed owner topic approval is present, and
node-side inventory, custodian and original-chain checks succeed. `pd_node.py`
enforces these gates before GPU discovery with a clear DEFERRED message/exit 3.
Failures during or after distributed launch return exit 1 with `error_type`.
Do not substitute SMOKE-SCOPE inputs or ask
the operator to make our-side topic/exclusion decisions.

The shipped package is `kit/resources/pd/scientific/`. It contains pinned TREC
train AND test source/provenance bytes, complete frozen encodings, the audited
local inventory/corpus, proposed partitions, node-only verification policy,
chain-source commitments, approval template and three deferred row records.
The owner approved all 130 automated topic exclusions; the committed decision is
scientific/topic-approval.json. The pinned preparation identity is
`d3ad575f52b1203e210098c37584cb4a80db4f3a44508a5babd52fd121a1a7fa`.
The 508,919,629-byte final JSON is not shipped. On the node run
`python kit/pd_materialize.py --package kit/resources/pd/scientific --approval kit/resources/pd/scientific/topic-approval.json --out APPROVED_PREPARATION_JSON`.
This CPU wrapper invokes `pd_scientific.py finalize` from the shipped inputs,
checks the pin and publishes only a matching output. The node launch gate checks
the pinned identity again before GPU discovery; mismatches are DEFERRED/exit 3.
See the package README for exact custodian/verify-node/plain-script commands.
The reviewed human decision table, docs/phase3/pd-topic-review-20261029.md, is
an our-side audit document and is not shipped; the committed approval ships.

Available audit: 40 sources / 154,180 source rows, including the frozen SP set,
every K/CR candidate group and all locally available chemistry inputs/responses;
zero programme overlap hits. The 5,952 TREC rows produce 130 automated topic flags
(85 train, 45 test). Proposed approval of all flags yields 128 calibration,
1,280 training, 455 evaluation rows and 3,828 unused eligible training rows.
Within-TREC deduplication excludes 131 further train rows. No sequence exceeds
1,024 tokens (measured maximum 498); no post-outcome replenishment is allowed.
These counts are fixed by the committed owner approval and pinned preparation.

Thirteen additional node-held inventories remain verify-only requirements.
The October–November confirmation slice does not exist at send time; its custodian
builds it on 15–31 December. The manager's sealed source-separation ruling replaces
the fingerprint prerequisite: public TREC question classification is disjoint
from SEC-filing/legal/maths confirmation sources by construction. The custodian
re-checks PD prompts when the slice is built. A preparation-bound receipt records
this commitment and the pending future comparison; the send manifest binds it as
`pd_custodian_receipt_sha256`. No empirical clean comparison is claimed.
A node overlap makes the result UNVERIFIED without changing or rescoring items.
Send assembly binds the exact core plain-D FinQA/ContractNLI source roles and
original export identities; node verification checks the send and live tensors.
The core's five hashed modules and taxonomy remain unchanged, preserving the
passed technical smoke. The low-level preparation API below is implementation
reference; its legacy custodian-fingerprint input is superseded for the scientific
send by the manager ruling above. The scientific package is the send workflow.

## Pinned source, exclusions and partitions

The 50-label mapping and source byte hashes are from
[the pinned CogComp metadata](https://huggingface.co/datasets/CogComp/trec/blob/1f97567bdd2adedefe8abdaa9bd6ee0e6725b458/dataset_infos.json).
The [pinned loader](https://huggingface.co/datasets/CogComp/trec/blob/1f97567bdd2adedefe8abdaa9bd6ee0e6725b458/trec.py)
defines original numeric IDs and the single invalid-byte correction.
Definitions in `taxonomy.json` follow the
[official taxonomy](https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html).
Training has 5,452 rows and test 500 before exclusions. TREC is external non-stream
data: the programme's month split does not apply. Pretraining exposure is unknown.

`python kit/pd_prepare.py verify --train TRAIN --test TEST --out VERIFIED`
checks actual byte lengths, SHA256 and row counts before parsing. The fetch
subcommands are explicit acquisition code for later authorized use; never run
them as part of an offline build. `fetch-provenance --out PROVENANCE` obtains
`trec.py` and `dataset_infos.json` through `hf_hub_download` with repository
`CogComp/trec`, dataset type and the exact pinned revision. It checks loader
source names, metadata byte hashes/counts and all 50 labels, then freezes both
file hashes in a provenance receipt. Preparation includes that receipt.
The scientific B4 package now includes byte-verified train/test sources and
pinned provenance; tests retain CPU fixtures and never download weights.

`python kit/pd_prepare.py prepare --plan PLAN --out FROZEN` accepts operator-local
locations for `train`, `test`, `tokenizer`, pinned `loader` and `metadata`,
plus sealed `inventory`, complete
`topic_reviews`, `custodian` decisions, and `expected_inventory_sha256`.
The tokenizer loads with `local_files_only=True`.

The manager independently freezes the complete inventory digest. It requires
`closed: true`, every `REQUIRED_COVERAGE` category, sorted complete `source_ids`,
and `sources`. Each source has `id`, `kind`, `rows`,
`rows_sha256=digest(rows)`; rows contain `id` and `text`. Include complete
programme sources/panels, generic/rejected/accepted-response material, chemistry,
earlier-task ancestry and broad/K/CR candidate-prompt supersets before send.
An incomplete supplied subset cannot establish closure.

Human decisions map every original TREC ID to
`{reviewed: true, exclude: BOOLEAN}`. The custodian supplies a sealed receipt
with `closed: true`, `candidate_sha256=digest(rows)`, `inventory_sha256`,
thresholds `{token5_jaccard: 0.8, edit_similarity: 0.9}`, and Boolean exclusion
`decisions` for every candidate. Reserved text/outcomes never enter this API.

Duplicate checks use NFKC/casefold/collapsed whitespace; normalized Levenshtein
similarity is one minus distance divided by maximum character length. Five-gram
Jaccard uses sets of whitespace tokens. Empty five-gram sets do not trigger
Jaccard. Equality/edit tests still apply. Profiles cache normalized text, tokens,
five-gram sets, lengths and character counts once per source row. Exact lookup,
shared-gram postings and length buckets prune candidates; Jaccard is evaluated
only at compatible gram counts and banded exact Levenshtein only within the
registered 10% length bound. Returned-pool verification uses the same index.
Among TREC duplicates accept test rows first in salted order, then drop training
rows matching any retained test row or earlier accepted training row. Unused
training rows therefore cannot discard an eligible test row.

Hash original training IDs with `pd-trec-v2|train|ID`: calibration 128, then
training 1,280, remainder unused. Hash test IDs with `pd-trec-v2|test|ID` and
evaluate **all eligible original test rows**, up to 500. No fixed 384-row slice.
Insufficient training/calibration defers PD; fewer than 64 common errors later
makes the probe PENDING. No replenishment or outcome-driven item deletion.

Measure the actual empty-question definition prompt before encoding; prompt-alone
overflow defers PD. For every question measure the whole rendered prompt plus
every exact label and EOS, and its training target. Exclude sequences over 1,024
before partitioning. Preparation retains all cap measurements and exclusion IDs.
Code, source bytes, token IDs, masks, suffixes, taxonomy and partitions are sealed.

`freeze_candidate_supersets()` binds normalized prompt supersets independently
of eventual response acceptance. `verify_returned_pools()` accepts sealed
`{candidate_supersets_sha256, sources: {SOURCE_ID: ROWS}}`, with optional
`response` per returned row. All source keys must be covered; prompts must be
contained in frozen supersets and prompts/responses must still exclude PD.
Mismatch is UNVERIFIED. Verification never changes partitions or reruns scoring.

## Primary scoring and descriptive diagnostics

Primary prediction maximizes **summed conditional log-probability of exact label
plus EOS**, with lexical ties. The tokenizer appends EOS exactly once to every
candidate. Frozen independent suffix boundaries avoid token-merge ambiguity.
A single prefix prefill is reused through independent KV-cache forks; CPU tests
compare this with an independent full causal forward, including EOS. No label
branch may modify another label's cache. Mean-token likelihood accuracy and
greedy generated exact-label accuracy are descriptive.

Every cell retains complete summed score vectors and gold log-probability/rank.
The reader reports each start's initial covariates on its frozen common-error
cohort. Zero initial accuracy reduces, but does not eliminate, competence
confounding. Common errors may concentrate label noise; short-label transfer
from ContractNLI may mask loss. Stage 1 remains descriptive.

At updates 0/40, content-only calibration tokens feed actual SwiGLU down-projection
pre-input hooks. Report normalized mean-absolute utility <=0.01 dormant fractions,
zero-layer flags and unweighted layer mean. Effective rank uses identical
hash-selected positions, centering and squared singular-value probabilities,
normalized by min(N-1,width); zero variance/N<2 is undefined. No representation
statistic gates a scientific rung.

## Bound starts and distributed training

The send binding's `pd_starts[SEED][START]` contains `checkpoint_id`,
`weights_sha256`, `export_receipt_sha256`, `parent_id`, `recipe`, `step`,
`base_revision`, `tokenizer_sha256`. These content fields avoid a circular hash
with the send-binding receipt. Each seed requires its own logical base ID ending
in `@seed101`, `@seed102` or `@seed103`; underlying base bytes may be shared.
Stage 1's `parent_id` must equal that seed's base ID, and stage 2 must name its
stage-1 parent. Starts are original pinned Qwen3-8B, exact v4
plain-D FinQA final used as C sequence-D parent, and its C ContractNLI step-40
final. Never substitute another route or replay arm. Verify every live tensor
and original receipt before loading. Reject missing/unexpected/mismatched loaded
weights and require the prescribed eager attention and SiLU architecture.

Inside the operator's eight-H100 allocation, use this plain-script step:

```sh
python kit/pd_node.py --plan PLAN --bound-plan BOUND_PLAN --out PD_STATE --score-visible-index 0
```

`CUDA_VISIBLE_DEVICES` must contain the allocation's eight distinct device tokens,
as supplied by Slurm. `--score-visible-index` selects a visible slot 0–7 in that
list. The wrapper translates it to the inherited allocation token, discovers the
physical UUID and environment digest in a CUDA-isolated subprocess, then freezes
`score_device=GPU-<UUID>` and `scoring_gpu_identity` into BOUND_PLAN before calling
`launch_ranks`. Using the stable UUID keeps every scorer on that same physical
GPU even with a reordered/subset allocation. Do not type a UUID or environment
digest by hand. The parent stays CUDA cold and preserves the full allocation
for all eight training ranks.

BOUND_PLAN and its default BOUND_PLAN.receipt.json supervisor receipt must be
fresh destinations outside PD_STATE and original source exports. The wrapper
refuses an existing nonempty state tree and never changes PLAN or source weights.
The wrapper is outside the five Python modules and taxonomy hashed by
`code_identity()`; its own SHA256 is bound separately in the runtime plan.
The qualified core remains byte-for-byte unchanged.

The input plan supplies the `scientific` paths from scientific/plan-inputs.template.json, plus:
- `preparation`, `seed`, `checkpoints`, `send_binding`, local `tokenizer`;
- `exports[START]` with original local `directory` and `receipt`;
- a qualified bound `smoke_receipt` (the sealed `receipts["smoke.json"]` from
  the passed smoke return, with the same core code hash), optional `returned_pools`, and optional
  measured `gpu_rate_per_hour` for explicit cost calculation.

The dedicated scorer is a fresh process restricted to that one physical GPU.
All starts and all curve points for a seed use the same assignment/environment.
Other seeds may use different assigned GPUs. Original source snapshots stay
outside PD state and are never deleted.

Each fork freshly reloads its bound source, RNG and AdamW: beta=(0.9,0.95),
epsilon=1e-8, decay zero, clipping 1, constant LR 1e-5, 40 nonzero updates.
FSDP auto-wraps the registered transformer block classes from
`_no_split_modules`, maintaining FP32 parameter/moment shards, FP32 buffers
(including RoPE frequencies) and BF16 compute. Each seed permutes
1,280 rows once into 40 global batches of 32, identical for all starts.
Loss is the global mean of per-example mean label-plus-EOS CE; prompt/padding
are excluded. With FSDP's rank-average reduction, each local example contributes
`world_size/32`, without another microbatch/world divisor. An independent CPU
autograd reference verifies simulated world sizes 1/2/8 and detects an extra
world divisor. Actual smoke uses two ranks; scientific node jobs require eight. In the two-rank
smoke only, FP32 optimizer moments are stored on host between updates and copied
back to the parameter GPU before AdamW arithmetic. Gradients are released after
stepping and unused allocator blocks before scoring. The eight-rank node path
keeps moments resident. These fixed storage choices change no loss or optimizer
arithmetic and make room for the separate single-GPU scorer.

Deterministic algorithms, eager attention, disabled TF32, frozen environment,
and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization are mandatory.
Unsupported kernels fail correctness qualification. There is no teacher, EMA,
replay, LR search or automatic retry.

## Streaming exports and restart rule

Use a separate PD state tree/volume with 256-GiB working quota and >=320-GiB
initial free-space preflight. Every run requires an empty PD tree. Source
snapshots cannot overlap it.

At updates 5/10/20/30/40 gather full FP32 state to CPU, convert floating tensors
to BF16 exactly as the core export helper does, retain nonfloating tensors and
release the FP32 gather. Training masters remain FP32. Export into one pending
slot, hash tensor bytes once and record `export_dtype=bfloat16` and
`master_dtype=float32` in the export receipt. Atomically rename the pending
directory to the scoring slot on the same PD filesystem. The isolated scorer
hashes actual tensors/receipt once before loading, then returns a sealed live-load
proof bound to the checkpoint. Completion checks that proof and retains sealed
tensor/receipt/score verification before removing the scoring slot. There are
45 exports across nine forks; no copy or repeated completion hashes are needed.
A failed live verification preserves the scoring slot and blocks advancement.
Deletion guards reject sources and every non-owned target.

**A failed fork restarts its whole seed from update 0 in a clean PD tree.** No resume state,
optimizer/RNG generation, resume API or automatic retry remains. An operator
starts a fresh tree after failure; the implementation never deletes an old run
or source to make room. Disk/full-job failure leaves unfinished evidence PENDING
and never touches core. A late failure therefore repeats all three forks and
all 18 scoring cells for that same registered seed, at additional GPU-hour cost.
The final review's PD M1 correction supersedes the stale resume-state wording;
the manager must update the canonical registration outside this builder's files.

The node supervisor now retains a generic PENDING launch-failure receipt outside
the clean PD tree. The qualified core swallows node rank RuntimeError details,
so the wrapper cannot honestly distinguish a rank OOM from a deterministic-op
failure. Detailed M4 rank classification stays deferred until an explicitly
authorized core change and requalification; no cause is invented.

Schema `pd-checkpoint-score-manifest-v2` contains nine logical original starts
and 45 curve-export identities, plus 54 score identities. Scores bind summed
vectors, gold covariates, correctness, suffixes, tokenizer, decoding, environment
and ordered batch/token/mask exposure. Derived checkpoint identities link retained
`stream_verifications`; the reader verifies these after model deletion. Original
nine source identities still require full live tensor checks. Seals are invariant
under JSON export/reload of integer update keys.

## Reader and reporting

Merge exactly three `seed-evidence.json` records with
`pd_read.collect_seed_evidence()`, or pass `{seed_records: [...]}` to:
`python kit/pd_read.py --evidence EVIDENCE --exports ORIGINAL_RESOLVERS --trusted ANCHORS --out REPORT`.

Resolvers cover only the nine preserved source identities, each with complete
`directory`, `tensors` and `receipt`. Independently frozen anchors supply
`send_binding_sha256`, `preparation_sha256`, `code_sha256`,
`smoke_receipt_sha256`, `candidate_supersets_sha256`, all eligible
`evaluation_ids`, and `exposure_sha256[SEED]`. Compute expected exposure before
measurement using the frozen rows, `seed_batches()` and `exposure_receipt()`.
Missing return verification is INCOMPLETE; mismatches are UNVERIFIED.

Freeze E_s before training from questions wrong at all three starts. Primary is
stage2-minus-base trapezoidal accuracy AUC on E_s, divided by 40. Require base
final E_s improvement >=10 points in every seed. Mean <=-2, upper two-sided 95%
bound <0 using t=4.302653, and 3/3 negative shows loss. Complete non-affirmative
evidence is INCONCLUSIVE, never preserved plasticity. Missing cells are INCOMPLETE;
invalid identity is UNVERIFIED. Final catch-up does not veto slower acquisition.
Lower bound >=-1 separately excludes a material deficit at this dose/cohort.
Keep raw/gain AUC, full-panel starting/final accuracy, undefined-safe normalized
headroom, stage-1 comparisons, generated/mean accuracy and representations descriptive.

## Unlaunched smoke, spending and measured costs

`smoke_inputs()` strips all test rows and seals calibration/training only.
`build_modal_app()` defines one container: two H100s, 128 GiB, eight vCPUs (four physical cores),
no retries and a 45-minute spending reservation. The pinned prebuilt runtime
must already contain dependencies/code; input and PD state volumes must be
distinct and prepopulated. All model/tokenizer loads are offline.

A sealed current Modal quote supplies published H100/CPU/memory-second rates
and startup/storage/egress ceilings. Price the entire reservation; ceiling <=$10.
No rate is invented and no app is run during build. The supervisor passes its
local operator configuration through environment, not returned artifact files.
Before launch, a subprocess restricted by `CUDA_VISIBLE_DEVICES=score_device`
discovers the runtime-assigned single-GPU UUID/environment. The supervisor parent
never initializes CUDA. That identity is injected into the plan and retained in
the smoke receipt and a sealed GPU receipt after worker execution.
It leaves time to write a PENDING spending-cutoff receipt. Additional execution
requires revised budget.

The actual 8B seed-101 smoke trains 40 updates with two ranks, reloads untouched
base and independently repeats updates 1–2. Require exact first-loss equality;
second-loss relative error <=1e-3 using max(abs(original),1e-12). Export/reload
label agreement >=99% is allowed only if every disagreement has top-two summed
margin <1e-3 in both executions; retain full vectors and disagreement receipts.
The original is an independent unsharded replica of the exact live BF16 export
state, scored before serialization using FP32 weights/buffers and BF16 autocast,
identically to the reload. FSDP explicitly forbids a forward inside
`summon_full_params`; no scoring forward runs there or through nested wrappers.
A tiny local Qwen3 CPU fixture verifies identical pre-serialization/reloaded
logits, BF16 file contents and unchanged FP32 masters.
Require base calibration <=70%, overall improvement >=5 points and >=10 points
on calibration questions initially wrong, fresh states, equal exposure/batches,
and the first fork's actual full exposure digest equal to the planned digest.
The two-update comparison derives from the first fork's recorded batches.
Require finite weights/gradients/defined diagnostics, nonzero weight change, no OOM and
valid exports. Observed OOM, deterministic-operation and other runtime failures
produce sealed failure receipts and DEFERRED qualification, including failures
in scoring subprocesses or a non-master rank. A spending cutoff or missing
unfinished evidence remains PENDING, never a scientific seed result.

Timing and 90% GPU/host utilization are warning-only. Record cgroup aggregate
host/device peaks, load/score/receipt durations, startup-inclusive wall time and
GPU-hours. Gather/export and transfer/scoring idle charges include all reserved
ranks. Per-fork serial scoring >24 hours is WALL-TIME RISK, never a pass gate.
Optional rate inputs compute monetary costs; absence stays unmeasured.

The measured attempt-3 scoring rate supersedes the historical 41–84 H100 GPU-h
forecast: the revised three-seed request is 222–264 GPU-h and 9.23–10.98 wall
hours per seed, including all eight reserved ranks during serial scoring.
See docs/phase3/pd-reprice-20261029.md for source timings, formulas, idle-rank
charges and remaining training/export allowances. The approved 1,800 GPU-h tier
allowance includes PD <=264 GPU-h; it is not launch authority. No scientific outcome is
claimed by technical smoke or CPU fixtures.

## Manager-authorized SMOKE-SCOPE handoff

Use the single command and offline plan in docs/phase3/smoke-pd-20261029.
The manager's later ruling replaces local weight uploads and a registry digest:
call the shared partner_image({}) factory exactly. A bounded CPU-only data image
layer fetches exact pinned public inputs with token disabled and verifies model
and training-byte SHA256 values before use. Record the resolved Modal image ID
and the pinned helper-spec hash. For the smoke, the GPU worker has an open
network with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1; inputs remain pinned,
hash-verified and loaded locally. Loopback c10d plus a 60-second bootstrap
supervisor avoids container-address rendezvous. PD uses direct FSDP without Ray.

For this smoke only, compute automated train duplicate, development-fixture
overlap, conservative chemistry-topic and cap exclusions. Freeze a distinct
SMOKE-SCOPE preparation and topic-flag items before inference. The manager's
launcher writes returned items for later review; attempt 2 retains its
topic-flags.json and attempt 3 uses topic-flags-attempt3.json. Absent or initial
items=null means not yet fetched, not zero flags. The scientific owner topic
approval is now committed and its node preparation is pinned as documented
above. Node-held programme verification remains required; the later custodian
comparison is pending the December slice build.

The quote includes the bounded 600-second fetch/preparation stage plus cold
startup and the shared image build allowance. The eight registered vCPUs map
to four physical cores. No model is fetched, image built, input uploaded or GPU
started during dry plan or CPU tests.
