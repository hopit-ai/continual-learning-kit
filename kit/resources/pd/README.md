# PD — right-sized jobs and the prospective scorer amendment

Seeds are exactly 101–103. This isolated tier never trains a chain, writes to
replay, extends core checkpoint retention, or blocks core execution. SP excludes
PD forks. No model/dataset acquisition or remote execution occurred during build.
The final owner-approved registration governs. Attempt 3 qualified the technical
smoke; scientific preparation and scientific measurement remain pending.


## Scientific preparation and combined-send B4

PD-101/102/103 are **DEFERRED** unless node-local finalization matches the pinned
approved preparation, the committed owner topic approval is present, and
node-side inventory, custodian and original-chain checks succeed. `pd_node.py`
enforces these gates in the CPU `prep` phase before any GPU job. Missing or invalid prerequisites return PENDING/exit 3; distributed failures leave incomplete evidence and require a clean whole-seed restart.
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

The input plan supplies the `scientific` paths from scientific/plan-inputs.template.json, plus
`preparation`, `seed`, `checkpoints`, `send_binding`, local `tokenizer`,
`exports[START]` with original `directory` and `receipt`, and the qualified
`smoke_receipt`. Keep the plan and all source checkpoints outside PD_STATE.
The numerical core and its preparation/smoke hashes remain unchanged; the new
orchestration has a separate `split_job_sha256` frozen in the CPU receipt and
independent reader anchors.

Submit the following phases as **separate jobs**, in order, for each seed.
`PLAN` and `PD_STATE` are operator-local placeholders. The same node-local state
must remain accessible: every phase checks the preparation hostname. CPU jobs
may share that host with GPU jobs. Schedule dependencies without an allocation;
never run these commands consecutively inside one GPU allocation.

| Task suffix (each seed 101–103) | nodes | GPUs | CPUs | May share node | Dependency |
|---|---:|---:|---:|---|---|
| prep | 1 | 0 | 8 | yes | completed scientific preparation/verification |
| pre | 1 | 1 | 8 | yes | afterok prep |
| train | 1 | 8 | 32 | yes, CPU work only; no shared GPUs | afterok pre |
| finalize | 1 | 0 | 16 | yes | afterok train |
| post | 1 | 1 | 8 | yes | afterok finalize |
| collect | 1 | 0 | 4 | yes | afterany post |
| read (all seeds) | 1 | 0 | 4 | yes | afterany all collect jobs |

Portable rows are in `job-resources.json` and `pd_jobs.resource_inventory()`;
`resource_rows(seed)` supplies each seed's six phases.
Use these sbatch resource arguments (add the site's partition/account, node
constraint and dependency yourself):

```sh
# One independent submission per line; append the matching phase command.
sbatch --nodes=1 --gpus=0 --cpus-per-task=8  ... # prep
sbatch --nodes=1 --gpus=1 --cpus-per-task=8  ... # pre
sbatch --nodes=1 --gpus=8 --cpus-per-task=32 ... # train
sbatch --nodes=1 --gpus=0 --cpus-per-task=16 ... # finalize
sbatch --nodes=1 --gpus=1 --cpus-per-task=8  ... # post
sbatch --nodes=1 --gpus=0 --cpus-per-task=4  ... # collect / all-seed read
```

Each submitted script invokes exactly one phase:

```sh
python -I kit/pd_node.py --phase prep     --plan PLAN --out PD_STATE
python -I kit/pd_node.py --phase pre      --plan PLAN --out PD_STATE
python -I kit/pd_node.py --phase train    --plan PLAN --out PD_STATE
python -I kit/pd_node.py --phase finalize --plan PLAN --out PD_STATE
python -I kit/pd_node.py --phase post     --plan PLAN --out PD_STATE
python -I kit/pd_node.py --phase collect  --plan PLAN --out PD_STATE
```

The GPU phases require exactly 1/8/1 allocated whole GPUs in
`CUDA_VISIBLE_DEVICES`, respectively; reordered/subset numeric selectors and
whole-GPU UUIDs are supported, MIG is refused. They run through
`python -I kit/util_watch.py --out PD_STATE/PHASE-util.jsonl -- COMMAND`;
missing watcher or prerequisites means immediate PENDING. Only allocated GPUs
are leased, nonblocking, by UUID in the host-wide `/tmp/kit-gpu-leases`
directory; there is no node-wide PD lock. Training owns all eight of its GPUs.
Use `pd_node.py` as the public entry; `pd_jobs.py` worker/rank modes are internal.
The old `--bound-plan` whole-seed entry is retired and fails closed.

The pre job scores all three starts and freezes E_s with a hash receipt before
training (at least 64 common errors). Training never invokes a scorer. The post
job scores update 0 again and all 15 curve points serially on its one physical
GPU/environment. **Only its 18 cells define the reported curve.** The pre job
may have used a different physical GPU. Compare each start's two update-0
summed score vectors on the frozen E_s: agreement >=99%, with every disagreement's
top-two summed-score margin strictly <1e-3 in both executions. The receipt and
reader recompute this rule and report full agreement/disagreement details.
Failure marks that seed UNVERIFIED; all registered cells remain visible.
A near-tie reload may produce nonzero post-job update-0 accuracy on E_s; retain
it in the curve and require base final minus initial improvement >=10 points.
Never reselect E_s from the post-job scores or combine scores across post jobs.

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

## Immutable spool, CPU export and restart rule

A seed has three forks and five post-update checkpoints per fork: **15 exports**
(45 across three seeds). All sources stay outside the fresh, isolated PD tree.
CPU prep checks all scientific gates, byte hashes, token IDs/masks and tokenizer,
then freezes compact training/scoring inputs. No dataset acquisition occurs.

At updates 5/10/20/30/40, gather full FP32 state to CPU and make a detached BF16
copy of every floating tensor, copying nonfloating buffers too. This snapshot
is never mutated. A bounded asynchronous writer saves a native `state.pt` and
configuration, hashes them and atomically renames the complete snapshot. Ranks
continue training while the previous snapshot saves; only checkpoint boundaries
can wait for that job's own writer to bound host memory. There is no scoring
callback, scorer subprocess or wait for another job. The final native save is
part of training; no HF conversion, publication or collection runs there.

The 0-GPU finalize job verifies all 15 native snapshots, constructs models on
CPU/meta, and calls the unchanged BF16 HF export helper with those exact states.
It writes tensor hashes, the original live-load proof inputs and checkpoint
identities. A spool completion receipt is published only after every export
exists. The post job refuses missing/incomplete spools before CUDA discovery.
Its unchanged likelihood/diagnostic functions hash and live-load each export,
then retain a sealed proof and scores **before** deleting that HF export and
its redundant native snapshot. A missing/wrong live-load proof preserves both.
Original checkpoints are never deleted. CPU collect rechecks scientific closure
and publishes `seed-evidence.verified.json`; failed reload agreement remains
UNVERIFIED there and in the amended reader.

Storage is deliberately conservative: reserve `15 * checkpoint_bound` native
bytes plus the same HF bytes, plus 64 GiB headroom. `checkpoint_bound` is the
larger of 18 GiB and the largest bound source tensor inventory's actual byte
size. Thus the usual per-seed spool bound is **540 GiB**, requiring **604 GiB
free at CPU preparation** (1,812 GiB for three simultaneous seed trees on one
filesystem). Each written checkpoint is checked against its bound. The old
256-GiB quota/320-GiB free check belonged to the retired one-slot streaming path.
At roughly 8B BF16 parameters the weights alone are about 15 GiB per checkpoint,
or about 450 GiB for both copies of all 15 points; actual sizes/config/shards
are measured on the node. Reserve disk capacity for concurrent seeds at the
scheduler/storage level: a free-space observation does not reserve disk blocks.
Host RAM must fit the existing full FP32 gather plus one BF16 snapshot; a new
snapshot waits for the previous writer rather than growing a queue.

**A failed phase restarts its whole seed from update 0 in a fresh tree.**
No optimizer/RNG resume, phase retry or automatic retry is introduced. Keep
failed trees, watcher logs and timing receipts for cost accounting. Never remove
a failed tree or original source automatically to make room. The pre/training/
export/post directories and completion receipts reject reuse. Disk or job
failure leaves unfinished PD PENDING and never touches core.

Schema `pd-checkpoint-score-manifest-v2` contains nine logical original starts
and 45 curve-export identities, plus 54 score identities. Scores bind summed
vectors, gold covariates, correctness, suffixes, tokenizer, decoding, environment
and ordered batch/token/mask exposure. Derived checkpoint identities link retained
`stream_verifications`; the reader verifies these after model deletion. Original
nine source identities still require full live tensor checks. Seals are invariant
under JSON export/reload of integer update keys.

## Reader and reporting

Merge exactly three `seed-evidence.json` records with
`pd_jobs_read.collect_seed_evidence()`, or pass `{seed_records: [...]}` to:
`python -I kit/pd_jobs_read.py --evidence EVIDENCE --exports ORIGINAL_RESOLVERS --trusted ANCHORS --out REPORT`.

Resolvers cover only the nine preserved source identities, each with complete
`directory`, `tensors` and `receipt`. Independently frozen anchors supply
`send_binding_sha256`, `preparation_sha256`, `code_sha256`,
`smoke_receipt_sha256`, `candidate_supersets_sha256`, `split_job_sha256`, all eligible
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

The right-sized estimate is **63–98 H100 GPU-h for three seeds**, replacing
the previous 222–264 GPU-h allowance. It includes 21 one-GPU scores per seed
(three pre-training plus 18 post-training), eight-GPU training/native saves and
the retained startup allowances. Forecast per-seed GPU wall time is about
10.53–12.34 hours across its separate allocations, plus unmeasured CPU phases
and queue time. See docs/phase3/pd-reprice-20261029.md for formulas and limits. The approved 1,800 GPU-h tier
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
