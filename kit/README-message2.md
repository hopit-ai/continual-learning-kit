# Message 2: connected continuation

Desk-built under the declared 24 October exception and clarification, before
return 1 and the hard-MATH result. This kit grants no launch authority. The
partner supplies sbatch; no network, API call, or new standalone training control
is part of these commands. W, exact parents, original archive hashes, the
three-seed decisions hash and hard-MATH qualification bind at send. A fresh
four-arm C-route re-probe supplies nomination on the node. Connected admission
is then computed broad first and earlier-task against the surviving incumbent,
before sibling MATH. No person chooses a configuration after results.

The 21 October amendment specifies W=D (provisional in measured B). The binder
recomputes B's unchanged W rule and requires agreement with C. Another W stops
for prospective clarification instead of extending the amendment by analogy.
C's **sequence-D core stage from v4 D** is the plain connected legal stage,
reused with its exact export identities. The sibling starts from B's
**prevention-trained D+replay FinQA final**, never C's v4-start replay final.
Both connected legal and MATH stages preserve each sibling's one admitted
configuration, with no reset, recovery, checkpoint selection or seed replacement.

## MESSAGE 2 v2 preflight and pool repair

Before M1 in **both** venvs (install on the staging host):

```bash
"$TRAINER_PY" -m pip install rapidfuzz==3.14.6
"$INFERENCE_PY" -m pip install rapidfuzz==3.14.6
python3 "$KIT/m2_preflight.py" --trainer-python "$TRAINER_PY" --inference-python "$INFERENCE_PY"
```

The near-duplicate checker requires rapidfuzz and fails immediately if missing;
there is no slow Python fallback. The distance rule is unchanged.

**M3b — regenerate and freeze broad pools before M-pools or any broad training.**
Use a new NEXT-v2 preparation tree. Copy the verified `next-inputs/`, `replay/`
and `hellaswag/` snapshots from NEXT into it, preserving original bound checkpoint
paths. Do not copy old `pools/`, raw answers or scores. Then run:

```bash
export NEXT_V2=/local/nvme/next-v2
run_stage "$INFERENCE_PY" "$KIT/next_pools.py" --out "$NEXT_V2" \
  --package-a "$A_OUT" --finqa-panel "$FINQA_PANEL_JSONL" \
  --finqa-targets "$B_TARGETS" --gpu "$SCORE_GPU" || exit $?
```

`A_OUT`, `FINQA_PANEL_JSONL` and `B_TARGETS` are the original checked Package A
panels, full 1,147-question FinQA canonical JSONL, and B's verified target directory.
Point M-pools' `--pool-root` at `NEXT_V2/pools`. Freeze its manifest and tar+SHA256
handoff before training. The frozen ARC instruction is `Do not use Markdown (no bold, italics or headings) anywhere in your reply. End with a final line that is exactly: Answer: X (where X is the correct choice label).`. The registered checker
(`resources/next/registration.md`: strict final choice label equals gold)
and evaluation prompts/parsers are unchanged. M-pools rejects old/relaxed pools.
A zero accepted ARC pool remains incomplete; no candidate replenishment or parser
relaxation. Return-1 format-confounded evidence is retained, never relabelled as
repaired evidence. Tests use synthetic base answers; real base journals are
verified and frozen by M3b on the partner's hardware before any broad training.

Inference engines shut down inside the logging context, including error exits,
before tee's pipe closes. NVMe validation follows the filesystem device's sysfs
partition/RAID ancestry and needs no /dev nodes or PATH shims for bind-mounted /tmp.

## Three-seed send, re-probe and independent return-1 rescoring

The 29 October rules supersede the historical send-time nomination prose in the
serial reference below. Real `m2_send`/`m2_bind` require `--decisions`,
`--return1-archives` and `--rescore-paths`, in addition to their existing inputs.
Use the original `package-c-return.tar.gz`, `next-return-1.tar.gz` and
`math-hard-teacher-return.tar.gz` in that directory. Their exact original hashes
and the reviewed decisions-file hash are frozen in `send-binding.json`. The
reading repack that removed a `.` directory member is never an input identity.
B's completed evidence still selects **D, provisional**; C identities use
101–103 and hard-MATH remains **QUALIFIES**. No new W selection is performed.

`rescore-paths.json` maps every Tier A call's `checkpoint` from the decisions
file to `{"model":"/partner/export/path","receipt":"/partner/original/export.json"}`.
The pinned frozen base may omit its receipt. Keep the original paths when they
are available; otherwise explicitly bind the relocated paths. Hashes of live
export tensors must equal the recorded call's hashes. All exports stay on the
partner's storage; there is no download or regenerated replacement checkpoint.

For example, add these arguments to the existing manager send command and node
bind command:

```bash
--decisions "$REPO/kit/resources/m2/return1-decisions.json" \
--return1-archives /absolute/original-return1 \
--rescore-paths /absolute/send/rescore-paths.json \
--format-check /absolute/send/hellaswag-format-check.attempt1.json
```

**RE-PROBE runs first.** R0 freezes the fixed ARC pool/prompt/strict parser and
C's original D after-FinQA starts. The four generic/broad/earlier/both arms each
train fresh on seeds 101–103 at the registered dose. Each seed's assigned
`reprobe` GPU scores its frozen base, incoming controls, every legal curve and
diagnostic-panel point, and every FinQA/general/HellaSwag guard. The strict
letter parser stays unchanged. Original format-confounded scores are never used.

`R-nominate` recomputes the registered three-seed paired two-sided bounds and
writes `nomination.json`, then verified durable `nomination/record.json` and
`COMMIT.json`. Broad nominated selects both versus broad, including acquisition
protection versus generic. A complete valid broad miss is INCONCLUSIVE/resolved
non-nomination and selects earlier versus generic. Missing or unverified evidence
selects neither branch. M8/M9 consume this record while preserving their
connected confirmation endpoints and distinct B prevention-trained starts.
Sibling MATH and all X remain pending until durable M9; plain D, S and core
collection have independent jobs. A partial collector can return available
receipts with exit 3; requeue refreshes only previously incomplete dependencies.
Completed inputs, snapshots, admissions and nominations stay immutable.

**Tier A is independent.** `RA0`, `RA101`, `RA102`, `RA103`, `RA-collect` execute
exactly the decisions file's 31 calls (10/8/13) in three packages. The original
checkpoint/panel/step and GPU UUID/environment are frozen and checked on scoring
and readback. Unavailable original UUIDs remain pending. `--enable-tier-b` is an
explicit freeze-time option in both assignment and manifest generation; Tier B
is off by default. Rescoring never supplies the fresh replay nomination.

## Static production manifest and input handoff

Create an operator settings JSON with these fields. Values are paths/CLI inputs;
the generator owns the reviewed row definitions and dependency graph:

```json
{
  "nodes": ["node0"],
  "trainer_python": "/venv-trainer/bin/python",
  "inference_python": "/venv-inference/bin/python",
  "runtime": {
    "sdpo": "/staged/SDPO", "tokenizer": "/staged/student",
    "model": "/staged/teacher", "panel_root": "/staged/finqa",
    "finqa": "/staged/finqa"
  },
  "bind": {
    "b_source": "/staged/B", "b_reading": "/send/b-reading.json",
    "c_source": "/staged/C", "checkpoints": "/send/checkpoints.json",
    "teacher_source": "/staged/math-hard-teacher",
    "expect": "/send/send-binding.json",
    "decisions": "/repo/kit/resources/m2/return1-decisions.json",
    "return1_archives": "/staged/original-return1",
    "rescore_paths": "/send/rescore-paths.json",
    "format_check": "/send/hellaswag-format-check.attempt1.json"
  },
  "next_source": "/staged/NEXT",
  "pools": {
    "package_a": "/staged/A", "finqa_panel": "/staged/finqa-panel.jsonl",
    "finqa_targets": ["/staged/B/targets/finqa"]
  },
  "prepare": {
    "c_targets": "/staged/C/targets", "replay": "/staged/B/replay",
    "panel_root": "/staged/finqa", "hellaswag": "/staged/NEXT/hellaswag"
  },
  "screen": {"spec": "/send/s-spec.json", "sources": "/send/s-sources.json"},
  "addon": {"spec": "/send/x-spec.json", "smoke": "/send/p2-smoke.json"}
}
```

Use the actual previously prepared S spec/sources, X spec and PASS smoke receipt;
the example paths do not supply data or scientific approvals. The generator reads
the S spec to enumerate its incoming/domain jobs and 15 inventory scores. Freeze
all settings, source specs, original receipts and exports with the input bundle.

On local storage, `input-sources.json` maps safe names to the staged files/trees
needed by these settings, including export roots, models/tokenizers, data, source
specs, original archives and send bindings. Package those inputs once. Its origin
map preserves the original absolute paths inside immutable JSON receipts:

```bash
"$INFERENCE_PY" "$KIT/m2_manifest.py" --settings /absolute/send/settings.json --output /absolute/send/launch.json
# Freeze launch/assignment using prepare.sh as shown below, then package locally:
"$INFERENCE_PY" "$KIT/m2_bundle.py" --sources /absolute/send/input-sources.json \
  --local-root /local/nvme/m2-input-staging --durable "$DURABLE" \
  --assignment /absolute/send/scoring-assignment.json
```

Every stage restores exact tar+SHA objects locally; immutable ancestral replicas
are cached by SHA under `LOCAL_ROOT/.replicas`, separate from writable stage OUT.
Snapshot receipts preserve references to exact historical objects, so a later
CURRENT cannot substitute a different export. Only tar files and small checksum,
manifest-pointer and barrier receipts touch durable shared storage. Keep native
optimizer/scheduler/RNG/EMA/dataloader state in the full snapshots. The return-1
restore took about three hours: give jobs sufficient TERM lead time and budget
measured transfer/restore wall time separately from training GPU-hours.

The manifest is a static dependency list, not a launcher. Inspect `tasks` and
submit the selected IDs with the partner's sbatch dependencies. M7 and M8 fan out
by seed after nomination. X15–X20 contain 36 fresh stages, each with its own
incoming/measurement/continuation/final jobs; no fresh control is reused.
R-nominate, M-core-collect, S12 and X collections can retain partial evidence.
An interrupted upload does not publish a completed pointer; a missing prerequisite
exits pending instead of polling. Requeue the same ID with the same frozen files.

## Concurrent 4A on separate nodes

Run the already-sent 4A runbook concurrently only on its own dedicated nodes and
state/durable roots. Bind **W=D, provisional** under its existing five-seed
protocol. Give its base and all controls their own dedicated physical scoring
GPU/environment, recorded by that kit's original scoring-machine receipts. Do
not set MESSAGE 2's `M2_ASSIGNMENT`, `M2_FAMILY` or replica-path environment in
4A jobs. Do not share scoring jobs, scores, caches, generated data or state roots;
common retention panels are scored independently. January–June data stays in 4A.

The note's bugs 3 and 5 apply: install/check rapidfuzz in both 4A interpreters
using the preflight commands above, and use the supplied engine-lifecycle fix
in `a4_score.py`/the logging context. Those changes release inference before tee
closes; 4A's doses, data, five seeds, controls and outcome rules are unchanged.
4A is grandfathered from message 1; the 29 October three-seed cut does not apply to it.

## Running on N nodes

The v2 commands in this section supersede the serial shared-OUT job boundary
examples below. The partner owns sbatch submissions, resource requests and
Slurm dependencies. There is no submitter, background dispatcher, polling loop
or automatic retry in the kit. A job executes one explicit manifest task and
exits 3 when a prerequisite is pending; identity refusal is UNVERIFIED (exit 2).

Use one local NVMe `LOCAL_ROOT` per node, a common durable root, and isolated
`LOCAL_ROOT/<task>/out` trees. M, S and X tasks never share a writable tree.
Reducers assemble verified snapshots; only the M9 reducer may commit admission.
Never train and score concurrently on the same full-node allocation. Different
scoring families may use separate assigned GPUs concurrently; each family is
serial on its assigned GPU. In particular, distributing one contrast's panels
across GPUs is forbidden even when all GPUs reproduce the same aggregate FinQA.

The first send job group is mandatory, using the same send and frozen manifest:

1. After INPUTS and launch/assignment freezing, submit **PF-node-0 ... PF-node-(N-1)**
   as exclusive whole-node jobs on their matching slots, before any other job on
   that node. Each runs both-runtime rapidfuzz checks and the trainer dataset import
   check from README-partner (NumPy/PyArrow/datasets/verl), then actual filesystem/options
   and functional flock, sysfs NVMe controller transports (all PCIe), all allocated
   GPU UUID-to-index mappings and child torch UUID assertions. Environment
   fingerprints and all assigned GPUs must match the frozen authority. This is
   also the required partner-stack mapping smoke, not an extra partner message.
2. Submit **PF-ARC** as the first preparation job on its deterministic slot after
   that node's preflight. It invokes `next_pools.py --prepare-candidates-only` on
   the real staged A/B/FinQA/replay/ancestry inputs and pinned cached sources.
   It performs all fresh exclusions and binds the first 128 ARC IDs to the packed
   passing attempt-2 receipt. Only then may M3b generate and freeze the broad pool.
3. Use partner-owned sbatch `afterok` dependencies. The node runner independently
   refuses to start any other task until its own PF-node snapshot is complete;
   missing/failed preflight exits PENDING. PF-ARC failure keeps M3b/re-probe and
   broad-dependent admission/sibling/add-on work waiting. Pool-free M3, plain-D
   work, Screen S and independent core collection proceed on healthy nodes.
   No alternate candidates, replenishment, lower threshold or reassignment.
4. Every independent collector includes `node-preflight/` with verified node and
   ARC receipts, snapshot identities, partial failures or explicit missing/PENDING
   status. Preserve these in the return. Repair infrastructure and requeue the
   identical task; do not discard a failed preflight to select another cohort.

Eight-node first jobs (separate partner-owned sbatch allocations):
```bash
bash "$KIT/jobs/message2-v2/prepare.sh" --task PF-node-0 --node-slot 0/8
# Repeat explicitly for PF-node-1 on 1/8 through PF-node-7 on 7/8.
# Then PF-ARC on its frozen slot; use launch.json node_slot, not an ordinal guess.
```
For one node use `--task PF-node-0 --node-slot 0/1`, followed by PF-ARC on 0/1.
Unknown shared filesystem types require the operator's explicit
`M2_DURABLE_FLOCK_ATTEST=<fstype>`; the attestation is printed and returned.
This never overrides known-bad local storage for multiple nodes, `noflock`,
`localflock`, `nolock` or `local_lock=all`; the functional probe always runs.
Before replica/continuation restore or local tar construction, the node pends
unless free space is at least **2 × selected bytes + 64 GiB**. CPU --standin
rehearsals reserve their selected bytes without a real-model reserve.

Collect an inventory from the **inference Python** on each allocated node,
using `m2_assignment.py --gpu 0` through `--gpu 7`. A node entry has
`{"node":"physical-hostname","gpus":[{"uuid":"GPU-...","environment":"SHA256"},...]}`.
Combine the entries into an array in `hardware.json`, then build the authority
once before scoring:

```bash
"$INFERENCE_PY" "$KIT/m2_assignment.py" --inventory hardware.json \
  --screen-spec /absolute/send/s-spec.json \
  --return1-decisions "$REPO/kit/resources/m2/return1-decisions.json" \
  --output scoring-assignment.json
```

The builder assigns core, X tier 1, each of the six X tier-2 comparisons,
Screen S seeds 101–103, and the additional v4 inventory scores. It freezes the
node, physical UUID and exact inference-environment fingerprint. Training may
occur elsewhere. All inherited base/incoming/standalone controls, curve points,
question-level survival cohorts, paired NLL and token-cost denominators must be
rescored inside the appropriate family. Shared base probes are one trained
probe, with separate assigned measurements; keep their exports until every
required scoring snapshot is verified on durable storage. Off-assignment or
identity-less evidence is UNVERIFIED, never an n=2 decision.

Generate the complete static task list using `m2_manifest.py` and the settings
below. The generated `launch.json` has schema `message2-launch.v2`, `nodes` in
slot order, and `tasks`. Each task contains:

- `id`: a unique safe stage/arm/seed identifier; `row`: its M/S/X row.
- `kind`: `prepare`, `train-stage`, `score-family`, `reduce-admission` or `collect`.
- `commands`: a fixed list of argv arrays for one selected row, executed serially.
  A single `command` argv is also accepted for hand-written diagnostic jobs.
- `requires`: completed snapshot task IDs; `overlay`: dependencies to assemble
  into this stage's writable view (identical files only; conflicts stop).
- `family`, `seed`: required for scoring. The node wrapper maps the assignment's
  GPU UUID to its PCI-ordered numeric index before staging. Pass `{score_gpu}`
  to leaf `--gpu`/`--score-gpu`; leaf scorers preserve that prepared index.
- `barriers`: durable prerequisite names; `commit_barrier` on the owning reducer.
  M10 needs M9. X1 needs M9, M10 disposition and S6. Tier 2 needs X12.

Command placeholders are `{python}`, `{kit}`, `{out}`, `{local}`, `{durable}`
and `{input_TASKID}` (the verified local replica of a dependency). Use original
logical checkpoint identities; snapshots record their origin paths and parent
hashes, and replica mappings resolve those paths locally without rewriting the
frozen receipts. External models/data must be staged locally with their original
registered receipts. No command downloads or substitutes inputs.

For example, a single selected training job after an assigned incoming-score
snapshot uses this task shape (fill the staged SDPO/tokenizer paths):

```json
{
  "id": "M7-generic-s101-train", "row": "M7", "kind": "train-stage",
  "requires": ["M7-generic-s101-incoming"],
  "overlay": ["M7-generic-s101-incoming"],
  "command": ["{python}", "{kit}/m2_run.py", "--out", "{out}",
    "--action", "legal-generic", "--seed", "101", "--mode", "train",
    "--tokenizer", "/staged/student", "--sdpo", "/staged/SDPO"]
}
```

`m2_run`, `x_run` and `x_tier2` accept `--seed`, `--select-stage`,
`--select-arm`, `--mode train|score|all` and deterministic `--node-slot i/N`.
M also has `--score-phase incoming|final|all`. Direct trainers/scorers retain
explicit stage/arm/seed arguments. Screen S has its own three-seed selectors;
probes offer `--phase train|score|all`. X4 offers per-seed measurement before
one all-three-seed calibration reduction. Do not reduce calibration from a
single measurement shard. Freeze local-support receipts only after all three
incoming generations are assembled.

The static dependency graph is: M3 → M4/M5 and fresh RE-PROBE; nomination
commits before M7/M8; plain MATH waits
only for its targets/exact plain legal parent. Broad reduction precedes earlier
confirmation. M9 waits for all required three-seed contrasts, publishes a tar,
verifies it and commits immutable admission before M10/X1. After S1→S2,
S3 inventory, S4 scoring and S5 audit→S6 freeze proceed independently; S6
then enables generation/drift/probe rows. Boundary accuracy is independent;
S never blocks core.
X1 requires M9, M10 disposition, M3 and S6. X4 calibration precedes fresh controls
and P2; controls and A survive P2 failures. X7 prefix→X8 measurement→full-state
continuation stays ordered. X12 is a verified tier-1 recovery point before
X14–X20; X23 adds an extended return. Scientific reading order is unchanged.

Freeze the manifests (this performs no submission or GPU work):

```bash
export MANIFEST=/absolute/send/launch.json DURABLE=/absolute/durable/message2-v2
export LOCAL_ROOT=/local/nvme/message2-v2 JOB_PY="$INFERENCE_PY"
bash "$KIT/jobs/message2-v2/prepare.sh" --assignment /absolute/send/scoring-assignment.json
```

**Eight-node example:** use `nodes` with eight hostnames. The default inventory builder puts core seed 101 on slot 3. The training hash
puts `M7-101-train` on slot 2; M9 and core collection belong to slot 7. In
separate partner-owned sbatch jobs, execute the selected IDs on their slots:

```bash
bash "$KIT/jobs/message2-v2/train-stage.sh" --task M7-101-train --node-slot 2/8
bash "$KIT/jobs/message2-v2/score-family.sh" --task M7-101-score --node-slot 3/8
bash "$KIT/jobs/message2-v2/reduce-admission.sh" --task M9 --node-slot 7/8
bash "$KIT/jobs/message2-v2/collect.sh" --task M-core-collect --node-slot 7/8
```

These are separate jobs with the partner's prerequisites, not a single shell
sequence. Non-owning training slots return NOT_THIS_SLOT without doing work;
score tasks execute only on the frozen assigned node. Use the same task/argv
for a requeue. No GPU UUID is reassigned when a node is unavailable.

**One-node example:** freeze one hostname and its eight physical GPUs, then
use the same task commands with `--node-slot 0/1`. The dependency graph,
assignments, three-seed completeness and distinct fresh controls are identical.
The partner serialises training against scoring in that allocation.

Each job copies each required tar once, verifies SHA256 and every member,
unpacks locally, and publishes a full-state tar plus checksum. A verified tar
is installed before its atomic CURRENT pointer; interrupted uploads leave no
new pointer. Completed snapshots and admission commits are immutable. TERM is
forwarded to the child's process group, followed by a partial full-state
snapshot after the child stops. Requeue uses identical inputs; native
model/optimizer/scheduler/RNG/EMA/dataloader state is retained. Preserve the
existing full-state restore qualification before resumed drift update 11.
Per-task `execution/*.json` and durable `timings/<task>/<snapshot>.json` record
input staging, child, snapshot and total wall time separately. Replica maps are
cached local files, not large environment variables. Record measured wall time
independently of GPU-hours; snapshots, duplicated
reference scoring, startup and four-update probes require measured repricing.

CPU-only validation (delete the basetemp afterwards):

```bash
df -h /
python3 -m pytest -q tests/test_message2_v2_bugs.py tests/test_message2_v2_seeds.py tests/test_message2_v2_multinode.py --basetemp=/tmp/message2-v2-tests
python3 kit/m2_v2_dryrun.py --out /tmp/message2-v2-transport --standin
```

Stop if `/` has under 6 GiB free. The transport rehearsal covers separate N=1
and N=3 local roots, none/broad/earlier/both/pending, every transport-row requeue,
verified snapshots, assignment refusal and admission barriers. The existing
`addon_dryrun --tier2` remains the four-configuration scientific-plumbing
rehearsal; stand-in outcomes never constitute scientific evidence.

## Stage before job 1

Reuse the entire [message-1 staging inventory](README-message.md#stage-before-job-1)
and its recorded dataset file hashes. These inputs add no new remote repository:

| Input | Immutable source / files | Use |
|---|---|---|
| Hard-MATH train pool | `EleutherAI/hendrycks_math` @ `21a5633873b6a120296cce3e2df9d5550074f4a3`; bundled `resources/m2/train_pool.jsonl`, 500 Level-5 rational-answer items; SHA256 `a35cecee96920a6a657b526477fbffc75cc09ba221f108919ed10bbca6ae830d` | Targets only when QUALIFIES; no redraw, rewrites or replacement IDs |
| Hard-MATH eval | Same pin; bundled `resources/m2/eval.jsonl`, 100 items; SHA256 `0bee96d6576a5144a0150d26170b08b6a04644bffcdc31db7e1f7e58f2c21b98` | All connected boundaries and MATH 0/10/20/30/40; never targets |
| Hard-MATH manifest | `resources/m2/manifest.json`, SHA256 `ee2417d5b1b772e994b80b70ea8866b5f49e79e70d241559e96145c55e1a0b8f` | Source provenance, largest-remainder subject quotas, salts `hmath-v1-pool` / `hmath-v1-eval`, rational eligibility, exact/near-duplicate exclusions and screen/tier-2 disjointness |
| FinQA | Original pinned v4 FinQA root from message 1 (`FINQA`); all **1,147** evaluation questions | Score at base, all stage boundaries, legal midpoint and every MATH curve point |
| ContractNLI | Approved original source and C's frozen 1,024-item membership, accepted targets/schedules; frozen `resources/c/contract_nli/test.jsonl` **100-pair reserve**, not diagnostic panel | Reuse C targets; original strict/tolerant parsers; document groups preserved |
| Replay | Returned B checked `replay/`; returned NEXT `pools/` (GSM8K `740312add88f781978c0658806c59bc2815b9866`, ARC `210d026faf9955653af8916fad021475a3f00453`, frozen verified earlier FinQA answers); NEXT `hellaswag/` (`Rowan/hellaswag` @ `218ec52e09a7e7462a5400043bb9a69a41d06b76`) | Exact pools and base responses; prospective overlap audit against all evaluations and general-v1; HellaSwag is transfer only |
| General-v1 | Existing `kit/panels/general-v1.jsonl` and manifest, unchanged | IFEval, knowledge, maths separately; overlapping IFEval families and lost/recovered IDs |
| Models | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`; `Qwen/Qwen3.6-27B` @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | Reuse full staged snapshots and pinned metadata; checker never downloads model bytes |
| Returned evidence | Complete unreset B scores/reading; original B D+replay exports/receipts; C binding, core/probe scores, C target manifests and original stage exports; `math-hard-teacher/` journal/manifest/decision | Binder recomputes B ranking, C completeness, `c_probe_read` nomination and `math_hard_teacher_read` outcome; it never accepts a manually supplied outcome |
| Frontier / implementation | C's already-archived matched test frontier **82/100**; pinned SDPO `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`; existing trainer/inference environments | Exact legal acquisition threshold **69.7**, no new inference; fresh stage optimizer/schedule/EMA |

`scripts/check_pinned_inputs.py` now checks the three derived hard-MATH files
under the existing MATH dataset pin, including repository/kit byte equality,
manifest membership and counts. Models still receive revision/listing metadata
checks only. Its network pre-tag check remains a separate manager operation;
**do not run it on the offline GPU node or during this build**. Keep its receipt
from the staging host with the send review. All job commands use:

```bash
export CHECKOUT=/absolute/reviewed-checkout
export KIT="$CHECKOUT/kit"
export OUT=/local/nvme/message2 DURABLE=/absolute/durable/message2
export B=/absolute/durable/B C_OUT=/absolute/durable/message1/C
export NEXT=/absolute/durable/message1/NEXT
export BASE=/absolute/cache/student-snapshot TEACHER=/absolute/cache/teacher-snapshot
export FINQA=/absolute/durable/finqa SDPO=/absolute/pinned/SDPO
export TRAINER_PY=/venvs/trainer/bin/python INFERENCE_PY=/venvs/inference/bin/python
export SCORE_GPU=0 TEACHER_GPUS=6,7
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

`C_OUT` must be the full durable C tree; compact returns contain receipts, not
model weights. Before send, the manager runs the following CPU command on the
return-1 archives and the returned B reading. It recomputes W, provisional,
the on-node nomination requirement (old probe outcomes never nominate), the frozen
`math_hard_teacher_read` outcome and both score digests. Copy the resulting
`send-binding.json` with the kit. The node must match every field with `--expect`;
any mismatch stops with exit 2, including a probe that finishes after send.

```bash
python3 kit/m2_send.py --out /absolute/send \
  --b-source /absolute/returns/package-b-return.tar.gz \
  --b-reading /absolute/returns/B/reading/reading.json \
  --c-source /absolute/returns/package-c-return.tar.gz \
  --teacher-source /absolute/returns/math-hard-teacher-return.tar.gz
```

Bind against the same verified full trees on the node. Never move paths after binding.
The checkpoints JSON maps `D+replay:101` through `D+replay:103` to
`{"model":"/absolute/B/runs/finqa-D+replay-s101/hf-step40",
"receipt":"/absolute/B/runs/finqa-D+replay-s101/export-40.json"}`.
Other seeds use their own original paths. C supplies base and exact v4 D parents.
The kit records hashes, run IDs (original run paths), parents and every output.

## Ordered runbook

| Row | Work | Command below |
|---|---|---|
| M1 | Job start, restore durable OUT, physical scoring GPU identity, TERM wrapper | Boundary block |
| M2 | Bind W, C identities, pending on-node nomination and hard-MATH outcome at send | Binding block |
| M3 | Pool-free C/evaluation prepare; gates independent M4/M5/M6 | Prepare block |
| M-pools | Verified replay/pool audit; gates R0 and M7/M8 | Pool prepare block |
| M4 | Base and both FinQA boundaries; reused plain-D legal boundaries/curve; compatible C standalone reference | `row boundaries` |
| M5 | Only QUALIFIES: 27B targets on all 500 training items, shared across trajectories | `row targets` |
| M6 | Plain-D MATH independently of replay admission | `row math-plain` |
| M7 | Three missing sibling generic connected ContractNLI stages from B's own finals | `row legal-generic` |
| M8 | If nominated, broad connected confirmations; frozen reader writes broad decision | `row legal-broad`; `row admit` |
| M9 | If nominated, earlier-task versus surviving incumbent; durable final admission | `row legal-next`; `row admit` |
| M10 | Sibling MATH from exact admitted connected legal finals; pending returns without running | `row math-sibling` |
| M11 | Registered bars, identity chain, acquired-answer survival and costs; no performance stop | Read block |
| M12 | Save, verify, collect compact return; job-end save including failures/unfinished work | Return block |
| M13 | Manager archive readback; next job/requeue starts at M1 and repeats unfinished rows | Readback block |

M1 and every job end use the existing [handoff contract](README-message.md).
First executable command: record allocation start, restore verified CURRENT,
identify host/environment/physical UUID, then source the process-group wrapper.

```bash
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action job-start || exit $?
readonly PARENT_OUT="$OUT" DURABLE_ROOT="$DURABLE"
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action restore --move-aside || exit $?
"$TRAINER_PY" "$KIT/m2_handoff.py" --out "$OUT" --durable "$DURABLE" --action identity --gpu "$SCORE_GPU" || exit $?
export TERM_LEAD_SECONDS=1800 SAVE_ALLOWANCE_SECONDS=900
source "$KIT/message_job.sh"
```

Keep writers serial. Measure save/shutdown allowances on actual storage. Same
host/environment/physical UUID is required for comparisons. The message-2
identity guard stops and returns evidence if any changes, before comparing
or altering a durable admission. It cannot silently re-admit a different
configuration after sibling MATH has begun. C/next's existing scoring-epoch
behavior is unchanged.

Binding block (M2; filenames are the actual returned evidence locations):

```bash
run_stage "$TRAINER_PY" "$KIT/m2_bind.py" --out "$OUT" \
  --b-source "$B" --b-reading "$B/reading/reading.json" \
  --c-source "$C_OUT" --checkpoints /absolute/send/checkpoints-message2.json \
  --teacher-source /absolute/durable/message1/math-hard-teacher \
  --expect /absolute/send/send-binding.json || exit $?
```

Prepare block (M3):

```bash
run_stage "$TRAINER_PY" "$KIT/m2_prepare.py" --out "$OUT" \
  --c-targets "$C_OUT/targets" --replay "$B/replay" --pool-root "$NEXT/pools" \
  --hellaswag "$NEXT/hellaswag" --panel-root "$FINQA" || exit $?
row() {
  run_stage "$TRAINER_PY" "$KIT/m2_run.py" --out "$OUT" --durable "$DURABLE" --action "$1" \
    --tokenizer "$BASE" --model "$TEACHER" --sdpo "$SDPO" --panel-root "$FINQA" \
    --score-gpu "$SCORE_GPU" --teacher-gpu "$TEACHER_GPUS" \
    --trainer-python "$TRAINER_PY" --inference-python "$INFERENCE_PY"
}
row boundaries || exit $?
row targets || exit $?
row math-plain || exit $?
row legal-generic || exit $?
row legal-broad || exit $?
row admit || exit $?
row legal-next || exit $?
row admit || exit $?
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action save || exit $?
row math-sibling || exit $?
```

The M9 save commits the admission before any sibling MATH scoring or training.
The sibling row and direct sibling trainer require durable CURRENT to hold the
same `admission.json` SHA256 as local OUT; missing or different evidence stops.

Full-node stages use eight GPUs serially. Teacher targets use TP2 in a gap;
scoring is deterministic, eager, batch invariant, greedy, thinking off, 2,048
response tokens, on one physical GPU. Raw answers/token IDs/finish reasons,
software/GPU identities, attempts, exclusions and actual costs stay durable.

M5 reuses `second_task_screen.run_phase` / `v4_teacher._run_pool`: at most four
attempts with seeds **27101–27104**, temperatures **0/.3/.6/.8**, top-p 1,
thinking off, 2,560 teacher output tokens, 4,096 input tokens. First verified
complete reasoning-plus-box target wins under the frozen rational verifier and
EOS-inclusive 2,048-token student limit. Gold is verifier-only. Archive every
attempt and rejection; cycle accepted IDs in seed-shuffled schedules. Zero
accepted targets leaves training undefined, exit 3, with evidence retained.

Legal and MATH stages use full weights, **40 updates**, batch **32**, LR
**1e-5**, **ten-step warmup including the zero-LR first update**, fresh
optimizer/schedule and fresh EMA, seeds **101–103**. The D task objective and
replay CE are inherited unchanged. Generic positions **8/16/24/32**, 28 task +
4 generic; one addition **24/4/4**, both **20/4/4/4**, exactly the frozen
`next_slots` maps. Broad then earlier confirmations always start from the same
B prevention-trained FinQA final, preserving the factorial comparison. The
admitted legal final and its matching MATH stage use the **same configuration**.

M4 scores all tasks at base and every boundary. New legal acquisition curves
use 0/10/20/30/40, with all earlier/general panels at 0/20/40. MATH scores
**all three task panels and each general panel at 0/10/20/30/40**. FinQA uses
1,147 questions, legal the 100-pair reserve and MATH the hard 100-item eval.
HellaSwag transfer is additionally scored for replay confirmation. The frozen
reader reuses `next_read` factorial contrasts: broad's primary is general
maths retention, earlier's primary is acquired-FinQA survival; mean benefit
≥2 points with positive paired 95% interval, other endpoint/HellaSwag/legal
final/legal gain no worse than −1 by their bounds. If broad survives, earlier
uses both vs broad, with the existing both-vs-generic acquisition safeguard.
Return-1 success only nominates; connected confirmation is required.

All scripts support identical-input resume/SKIP. Identity changes stop (2).
Missing input/unfinished stage stops (3); the orchestration's pending sibling
row prints PENDING and returns for collection. Interrupted training preserves
partials and restarts the identical seed/input from step 0 if no durable final
exists. Finished-dose exports recover without retraining. Weak performance is
an analysis result and never changes stage dose, seeds, targets or configuration.

Read block (M11):

```bash
run_stage "$TRAINER_PY" "$KIT/m2_read.py" --out "$OUT/reading" --source "$OUT" || exit $?
```

The reader reports every seed and one-sided 95% seed-t bounds: FinQA positive
gain lower bound; legal tolerant mean and lower bound ≥69.7, positive gain in
3/3 seeds and positive gain lower bound (strict/format separately); hard-MATH
incoming mean gain ≥6 points, 3 positive seeds and positive gain lower bound.
MATH is called **acquisition only if final mean also beats measured base**;
otherwise successful incoming gain is recovery. Screen 69% is never the eval
baseline. Sequence legal/MATH must be within 2 points of a compatible same-recipe
base-start standalone control by paired bound; absent/incompatible controls
stay pending. Only plain legal has the reused C standalone D comparator here;
easy-MATH controls cannot supply a hard-MATH comparison. No new controls are
silently trained or budgeted.

FinQA retention is cumulative ≤1 point from its **stage-1 final** after stages
2 and 3; legal ≤1 point from its stage-2 final after MATH. Every general panel
stays within 3 points of original base after each stage. Crossing intervals
are INCONCLUSIVE. Acquired-answer cohorts are wrong at base/right at their own
acquisition boundary, followed through every later boundary with denominators
and lost IDs. Question-paired bootstrap is separate, legal document-clustered;
questions/checkpoints do not create seeds. Serving tokens per correct includes
failed-answer tokens; each task ≤1.5× frozen base, otherwise AT_COST; zero
correct is undefined. Report finish rates, all costs, and IFEval overlapping
families (case, punctuation, length, keywords, format, content, start/end).

**DOES_NOT_QUALIFY:** M5/M6/M10 SKIP; `stage3-status.json` records DROPPED.
Complete the connected legal confirmations and **two-task readout only**:
legal acquisition, FinQA/general retention and acquired-answer survival/cost.
Third-task acquisition and retention of both earlier tasks through stage 3
are lost. No hardest-types variant is selected here.

**INCOMPLETE:** stage 3 is PENDING; targets and both MATH trajectories SKIP.
Return completed two-task evidence and the incomplete screen journal. It is
neither screen outcome. **Replay pending/incomplete:** sibling MATH never runs;
plain MATH still runs if QUALIFIES. The next message can continue from the
returned durable decision/evidence, with no intervening human recipe choice.

Return block (M12):

```bash
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action save || exit $?
run_stage "$TRAINER_PY" "$KIT/m2_collect.py" --out "$OUT" --archive "$OUT/message2-return.tar.gz" || exit $?
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action save || exit $?
```

Return only the compact archive (<28 MiB), plus job IDs/restarts, actual timings,
charged allocation hours, pinned-input receipt, durable path and unfinished
inputs. Weights/raw journals/schedules remain on durable storage and are
hash-indexed by the archive; exports and the entire connected identity chain
are included. Verify CURRENT before ending/requeueing each job. The manager
reads the archive without extracting or requiring weight files (M13):

```bash
python3 kit/m2_read.py --out /absolute/reading-message2 --source /absolute/message2-return.tar.gz
python3 kit/message2_dryrun.py --out /tmp/message2-standin --standin
```

The dry run uses CPU fixtures only: all 15 nomination/outcome combinations,
W=D, broad/earlier/both/none/pending admission, resume/SKIP, durable requeue and
archive readback. Technical archives require `--standin`, never scientific use.

## Pricing at send

The older 65–90 h design allowance predates the twelve fresh re-probe stages and
the independent 31-call Tier A rescore. Include those jobs and their transfers in
measured repricing; a planning allowance is not a measured forecast or approval.

29 October v2 supersedes the historical five-seed prices below: use 3/5 of
training/probe/calibration rows; scoring/transfer costs require measurements.
Core maximum: 9 legal + 6 MATH stages. X tier 1: 12 stages; tier 2: 36
stages (30 bound), plus 3 disposable P2 feasibility passes and up to 12 probes.
Screen S uses 27 probes and three seeds under the owner’s later 29 October
amendment, with its whole-chain bootstrap below. `x_price.py` emits the scaled X counts.
M/X one-sided seed-t uses df=2, t=2.919986; two-sided uses t=4.302653. All
three seed contrasts are required; ≥80% positive means 3/3. Unmet bars
are INCONCLUSIVE for admission and preserve the incumbent.


Measured tier-2 D/D+replay proxy: **58 minutes × 8 GPUs = 7.73 GPU-h/stage**
(D uses D+replay's proxy). Reused B FinQA, C plain-D legal stages, targets and
frontier inference cost zero incremental training/inference. No double charge.

| Component | Incremental GPU-h |
|---|---:|
| Six MATH stages, three seeds × two trajectories | 46.4 |
| 500-item bounded 27B targets allowance | 10.0 |
| All-task scoring/export allowance | 30.0 |
| Subtotal before missing legal stages | 86.4 |
| Three missing generic sibling legal stages | 23.2 |
| **QUALIFIES floor** | **109.6** |
| Each nominated component's three connected legal confirmation stages | **+23.2**, plus its scoring |
| Both nominated components (broad, then earlier against survivor) | **+46.4**, plus confirmation scoring |
| QUALIFIES, one / both component sets before extra scoring/overhead | 132.8 / 156.0 |
| DOES_NOT_QUALIFY: three sibling legal stages + scoring allowance | **53.2**, before confirmations |
| DOES_NOT_QUALIFY, one / both component sets before extra scoring/overhead | 76.4 / 99.6 |
| INCOMPLETE | Two-task work only; stage-3 56.4 remains pending, never charged as completed |
| Confirmation scoring, startup/export and allocation idle overhead | Add measured forecast from return 1 explicitly; not included in the floor |

109.6 is a **floor**, not a ceiling or allocation approval. Reprice from return
1's measured legal/teacher/panel timings before send, including confirmations,
their scoring, failures and allocation overhead; long Level-5 targets can
exceed the FinQA-based proxy. Record GPU-hours and wall time separately. A
pending sibling reduces completed work, not seeds. Missing compatible standalone
controls remain pending instead of creating an unpriced experiment.

## Screen S

Approved 25 October under plan v6 amendment §2. This add-on runs **after M2
binding**, independently of M3–M13 and their continuation training rows. It
uses separate `S_OUT`/`S_DURABLE` trees, both outside message 2's OUT and durable
trees. Never share or nest the Screen S and message-2 state trees.
No paid run is part of the build/dry run. Round 3 allowed only the pinned
FinQA `train.json` download for the count-only feasibility check; GPU rows
remain offline. The owner supplies the
allocation and send review. Existing M rows and admission readers are unchanged.

Stage these additional **local** source bytes before S1. Reuse the original
immutable revisions and record source SHA256s from the staging receipt; do not
resolve a branch or download on the GPU node.

| Domain | Pin and staged file | Selection / checker |
|---|---|---|
| IFEval | `allenai/RLVR-IFeval` @ `47c03c73621c4aab2b824b7818681117d662770e`, original train parquet, 14,973 rows | Outside B's entire 2,048-candidate pool; source executable constraints using unchanged open-instruct checker @ `11826255077617a46919ce75cadf1f3d53f30dac` (vendored SHA256 `afea345e3bc100e6d5588ca14e7269141e8810e1489e60ad64a59cd57360201c`); ODC-BY, checker Apache-2.0 |
| GSM8K | `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866`, `main/train-00000-of-00001.parquet`, 7,473 rows | Outside complete 1,024-candidate broad list; unchanged exact-number checker; MIT |
| ARC | `allenai/ai2_arc` @ `210d026faf9955653af8916fad021475a3f00453`, `ARC-Challenge/validation-00000-of-00001.parquet`, 299 rows | **Validation**, not train; unchanged strict label checker; knowledge proxy; CC-BY-SA-4.0 |
| FinQA | `czyssrs/FinQA` @ `0f16e2867befa6840783e58be38c9efb9229d742`, original `dataset/dev.json`, 883 rows, prompts **with gold** | Uninformative if any v4 exposure used dev; unchanged executed-value/rounding verifier; MIT |

Also stage original v4 export receipts for seeds 101–103 (15 step-20 exports; step-10
inventory, including missing entries), B unreset sequence D/F exports at
10/20/40, v4 plain D/F exports at 20/40 and 10 where available, base snapshot,
all original ancestry training/candidate pools and receipts, and all evaluation
panels. Missing step-10 exports are inventoried, never reconstructed by training.
FinQA test scoring still uses all 1,147 original questions.

`S_SOURCES` is JSON keyed `ifeval`, `gsm8k`, `arc`, `finqa`; each value is
`{"path":"/absolute/staged/file","sha256":"<staging SHA256>","repo":"<above repo>","revision":"<above revision>","split":"train|validation|dev"}`.
The checker adds an explicit offline mode without altering its existing
message-2/network checks:

```bash
python3 scripts/check_pinned_inputs.py --screen-s-sources "$S_SOURCES" \
  --receipt /absolute/send/screen-s-pinned-inputs.json
```

S1 repeats the same source loaders and writes a local receipt. The binding
checks immutable code hashes in `resources/s/registration.json` **before any
measurement**, including `scripts/screen_s_analysis.py`, a4 NLL, checkers and
handoff. Do not edit or refreeze after real measurements. Technical fixtures
are labelled `technical_synthetic` throughout.

`S_SPEC` is a prospective, receipt-backed JSON inventory, not a manual result:

- `nodes`: safe IDs mapped to `{parent, root, step, checkpoint, exposure}`.
  `checkpoint` is `{model, receipt}` using the original export receipt, including
  the staged base's original identity receipt. `parent` is the exact incoming
  node for the stage, not a different probe or an intermediate checkpoint.
  The root has `parent:null`, `root:true`, `step:0`. Other roots are refused.
- `exposure`: `{path, sha256}` of a closed ancestry receipt containing
  `{complete:true, parent, checkpoint_identity, pools}`. `checkpoint_identity`
  is `s_common.digest(checkpoint)`. Each pool reference contains
  `{role,path,sha256}` and points to original, canonical `{id,prompt}` JSONL
  membership, **including candidates rejected by verification**. Retain the
  original source manifests alongside these records. Roles include
  `b-ifeval-candidates`, `broad-gsm8k-candidates`, `v4-finqa-training`, all other
  ancestor pools, and `v4-finqa-dev` if used. The complete receipt attests the
  entire adaptation pool inventory, including the base's lack of adaptation.
  Each non-root v4 node must declare a nonempty `v4-finqa-training` pool in
  its own receipt. Each tier-2 node must additionally declare nonempty
  `b-ifeval-candidates`, `broad-gsm8k-candidates`, and `b-second-task-training`
  pools (the complete second-task training membership); another node's receipt
  cannot supply a missing role. The unadapted base requires no adaptation roles.
  Missing, mismatched or incomplete proof makes domains uninformative. A declared
  `v4-finqa-dev` pool, or an exact ID or exact normalized-prompt match to dev in
  any ancestry pool, makes the entire FinQA domain uninformative. Normalization
  uses B's NFKC, casefold and whitespace rule. Near-duplicates only drop rows
  through the unchanged exclusion rule; panel-only matches also only drop rows.
- `stages`: the 12 exact primary entries, one per tier × D/F × seed 101–103:
  `{id,tier,recipe,seed,stage,incoming,checkpoints}`. Stage names are
  `base-to-finqa` and `finqa-to-second-task`. `checkpoints` maps string steps
  `20`, `40` and available `10` to node IDs. All tier-1 starts share base; each
  tier-2 start must be the matching tier-1 final. Each tier-2 entry also has
  `summary:{path,sha256}` for B's original training summary; route, recipe,
  seed and incoming model hashes are checked. Original v4/B export paths in
  receipts must establish their registered slot/step identities.
- `panels`: named `{path,sha256}` canonical `{id,prompt}` JSONL references for
  every panel listed in the registration's `panel_inventory`: general-v1,
  FinQA test, ContractNLI reserve and diagnostic, hard-MATH eval and screen,
  both Package-A panels and HellaSwag. General-v1 must match the bundled bytes.
  All B exclusion rules (shared IDs, NFKC equality, 5-shingle Jaccard ≥.8,
  edit similarity ≥.9) apply against every panel and every ancestry pool.
- `v4_step20` and `v4_step10`: lists with exactly the 15 original IDs with seeds 101–103 from
  `v4_campaign.slots()`, each `{slot,checkpoint}`. Step-10 `checkpoint:null`
  records absence. Step-20 references are mandatory. Available tier-1 step-10
  nodes must correspond to these same original exports. Other recipes are
  scored for the added job but never enter the primary Screen S statistic.

Run S rows serially within their own job shell. S3/S4 may proceed alongside
the independent continuation when a separate scoring GPU is allocated.
S5–S10 require their own allocation and must not share the continuation's
allocation or GPUs. S10 requires an exclusive eight-GPU node, including its
scoring and cleanup. Use the same inference environment, host and
physical scoring GPU for every Screen S scoring operation; changes stop.
Job start/TERM/end use the existing durable handoff and process-group wrapper.
Use a fresh shell for this block (the M-row wrapper has its own readonly roots).
Set `KIT` to the kit directory, `FINQA` to the original FinQA scoring data,
`SCORE_GPU` to the physical scoring GPU, `TRAINER_PY` to the trainer Python,
and `INFERENCE_PY` to the inference Python; all five are required:

```bash
export KIT=/absolute/send/kit FINQA=/absolute/data/finqa
export SCORE_GPU=0 TRAINER_PY=/absolute/trainer/bin/python INFERENCE_PY=/absolute/inference/bin/python
export M2_OUT=/local/nvme/message2
export S_OUT=/local/nvme/screen-s S_DURABLE=/absolute/durable/screen-s
export S_SOURCES=/absolute/send/screen-s-sources.json
export S_SPEC=/absolute/send/screen-s-spec.json
export OUT="$S_OUT" DURABLE="$S_DURABLE"
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action job-start || exit $?
readonly PARENT_OUT="$OUT" DURABLE_ROOT="$DURABLE"
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action restore --move-aside || exit $?
"$INFERENCE_PY" "$KIT/m2_handoff.py" --out "$OUT" --durable "$DURABLE" --action identity --gpu "$SCORE_GPU" || exit $?
export TERM_LEAD_SECONDS=1800 SAVE_ALLOWANCE_SECONDS=900
source "$KIT/message_job.sh"
srow() { run_stage "$INFERENCE_PY" "$KIT/$1" --out "$S_OUT" "${@:2}"; }
```

| Row | Work / resource | Command (in order) |
|---|---|---|
| S1 | Check four local pins/splits/populations; CPU | `srow s_inputs.py --sources "$S_SOURCES"` |
| S2 | Bind cohort, ancestry, code and M2 identity; CPU | `srow s_bind.py --message2 "$M2_OUT" --sources "$S_SOURCES" --spec "$S_SPEC"` |
| S3 | Inventory the 15 original v4 step-10 slots with seeds 101–103; CPU | `srow s_inventory.py` |
| S4 | Score all 15 v4 step-20 exports, FinQA + general-v1; one GPU | `srow s_score.py --action v4-step20 --finqa "$FINQA" --gpu "$SCORE_GPU"` |
| S5 | Prove held-out membership against closed ancestry and all panels; CPU | `srow s_audit.py` |
| S6 | Freeze hashed, disjoint train/monitor partitions; CPU | `srow s_freeze.py` |
| S7 | Incoming own-response generation and frozen verification, 7 incoming checkpoints; one GPU | `srow s_run.py --action generate --gpu "$SCORE_GPU" --inference-python "$INFERENCE_PY"` |
| S8 | EOS-inclusive NLL at stage start and available 10/20/40; one GPU | `srow s_run.py --action drift --gpu "$SCORE_GPU" --inference-python "$INFERENCE_PY"` |
| S9 | Same-GPU incoming/20/40 panel accuracy, reusing identical scoring from S4; one GPU | `srow s_score.py --action boundaries --finqa "$FINQA" --gpu "$SCORE_GPU"` |
| S10 | 27 independent four-update probes (uninformative domains get receipts); eight GPUs, scoring on one | `srow s_run.py --action probes --gpus 0,1,2,3,4,5,6,7 --score-gpu "$SCORE_GPU" --trainer-python "$TRAINER_PY" --inference-python "$INFERENCE_PY"` |
| S11 | Check amendment pricing arithmetic; CPU | `srow s_price.py` |
| S12 | Durable save, compact archive, verified save at job end; CPU | Return commands below |
| S13 | Manager archive-only analysis/readback; CPU | Analysis command below |

Append `|| exit $?` to each S-row invocation. At every requeue, repeat the job
boundary block, then unfinished rows: identical completed rows SKIP. Never run
concurrent writers in one S tree. GPU workers run in separate processes so
CUDA allocations are released between checkpoints and before full-node probes.
Interrupted probes restart the same four updates from the bound incoming;
scored receipts resume cleanup without extending the intervention.

Selection is `sha256("screen-s-v1|"+domain+"|"+id)`, then ID. Count from one:
every third audited row monitors, others train. The partitions are shared across
checkpoints. At each incoming checkpoint, freeze the **first 128 verified
training and first 64 verified monitoring** responses, greedy once, no new pool,
redraw or replenishment. Wrong/truncated outputs do not become targets. Fewer
than 64 monitors is uninformative; probes also require all 128 training rows.
Only incoming-verified behaviour earns extra replay credit; wrong-to-right
panel transitions are reported separately as learning.

Before any model measurement, the 25 October count-only feasibility check
applied S5's unchanged exclusions to FinQA dev against all nine real panels
plus all **6,251** pinned `train.json` rows (a superset of v4's FinQA training
pool). It kept **279/883** rows and excluded **604**, projecting **186 training
and 93 monitoring candidates** under the frozen 2:1 split. No exact dev ID or
normalized prompt matched train. Other real ancestry remains to be audited;
this count is not an S5 exposure proof. The fresh download was unavailable;
identical bytes were staged from a clean checkout at the pinned commit and
verified against its Git blob, as recorded in the
[feasibility receipt](../docs/phase3/evidence/screen-s/finqa-arc-feasibility-20261025.json).

The preregistered count-only decision is: **kept < 230** registers FinQA as
expected-uninformative, descriptive only, with tier 2 requiring IFEval, GSM8K
and ARC, all three; otherwise retain the current rule. **279 ≥ 230 selects the
current-rule branch**: tier 1 requires all three of IFEval/GSM8K/ARC; tier 2
requires at least three of four domains. At assumed 70% verified-correct FinQA
accuracy, expected correct candidates are **130.2 training / 65.1 monitoring**;
per-checkpoint shortfall probabilities are **32.98% training, 35.39% monitoring,
56.69% either**. The monitoring shortfall applies to drift; either shortfall
prevents the complete probe. This substantial risk is recorded without changing
the 230-row decision cutoff, sources, 128/64 requirements or split. With FinQA
uninformative in a cell, all three remaining domains are required.

ARC's projection assumes **299 rows, no ancestry pools**, hence **200 training /
99 monitoring candidates**. Exact binomial projections assume independent rows
and partitions at the stated common verified-correct accuracy; they are not
measured model results and do not assume independent checkpoints:

| Assumed 8B ARC-Challenge accuracy | Expected correct train / monitor | Training shortfall | Monitoring shortfall | Either shortfall per checkpoint |
|---|---|---|---|---|
| 60% | 120 / 59.4 | 86.07% | 79.92% | 97.20% |
| 65% | 130 / 64.35 | 35.31% | 42.50% | 62.80% |
| 70% | 140 / 69.3 | 2.84% | 10.30% | 12.86% |
| 75% | 150 / 74.25 | 0.02% | 0.79% | 0.81% |

Drift uses `a4_nll.measure` unchanged: EOS-inclusive `token_mean` primary and
`sequence_mean` descriptive, own frozen text and exact stage-start baseline.
Probes use independent incoming forks per domain, four global batches of 32,
equal sequence losses each normalised over response tokens including EOS,
AdamW .9/.999, decay .01, clip 1, LR **0/1e-6/2e-6/3e-6**. Probes
use fp32 master weights and buffers with bf16 compute and fp32 reduction, as
v4 does; exports are cast to bf16 at save. Shared base probes
are reused; domains never chain. After scoring drop/final NLL, remove all probe
model output, keeping its hash-bearing receipt. Monitoring never trains.
Each of the eight ranks loads the full fp32 model in host RAM: about 33 GB per
rank and roughly 260 GB per node.

The manager reader computes within-cell raw-signal Kendall tau-b with exact
ties, equal cells within seed, then equal seeds. Pass requires mean ≥.5,
positive means in 3/3 seeds and strictly positive 97.5% one-sided lower bound:
10,000 draws resample three entire seed chains, fixed RNG seed 20261025,
empirical inverse CDF rank 250. Constant marginals/undefined support are
inconclusive. Tier 1 needs all three domains, tier 2 at least three of four.

The owner's 29 October amendment supersedes the earlier five-seed S exception:
only seeds **101–103** enter new S jobs or its inventory. Domains and checkpoints
are never resampled as seeds. With three chains, the lower bootstrap tail
approaches the minimum seed statistic; S is a weaker experiment-opening screen.
Unmet or missing support stays INCONCLUSIVE for opening the experiment.

Missing primary cells are inconclusive. If any tier-1 forward cell is missing,
all tier-1 forward cells are descriptive only and the forward test uses tier 2,
requiring both recipes and all three seeds. Drift must beat
the stage-start NLL difficulty-only tau mean in both association and forward
prediction. Plasticity independently predicts cross-task damage. Drift must
pass both association and step-10→20/40 forward tests; drift goes first if
both qualify. There is no combined score or C admission gate.

```bash
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$S_OUT" --durable "$S_DURABLE" --action save || exit $?
srow s_collect.py --archive "$S_OUT/screen-s-return.tar.gz" || exit $?
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$S_OUT" --durable "$S_DURABLE" --action save || exit $?
# Manager, after return; no model tree or extraction required:
python3 scripts/screen_s_analysis.py --archive /absolute/returns/screen-s-return.tar.gz
# CPU rehearsal only; its archives require --standin when read:
python3 kit/screen_s_dryrun.py --out /tmp/screen-s-standin --standin
```

Return the separate compact archive (<28 MiB), source-pin receipt, allocation
job/restart records, real timings and durable location with message 2. Keep
raw responses/token IDs, accepted targets, exclusions and all ancestry source
proofs durable; the archive hash-indexes bulk files and includes compact
measurements, accuracy transitions, audits, registration and probe receipts.

Pricing: **27 × .75–1.4 = 20.25–37.8 GPU-h**, plus drift **3–5**, generation
**5**, other scoring/startup **5** = **33.25–52.8**, the owner’s approximate
**30–50 GPU-h** envelope (rounded planning figures). C replication is excluded. The 15-export step-20 scoring and
step-10 inventory are added message-2 jobs and need a separate measured price
before send. CPU stand-in latency is not a real generation forecast: preserve
its counts, then reprice using actual staged hardware/timing before the send
review. No real GPU-hour or export-size measurement is claimed by this build.

## Message-2 add-on: rung P2 and speculative rung A

Amendment §2/§3 registers a **separate job after M10**, with separate, unnested
`X_OUT`/`X_DURABLE` trees outside both M trees and `S_OUT`. M11–M13 proceed
independently. X scripts only read M evidence; an X failure cannot alter
`admission.json` or prevent the core return. No GPU/Modal launch is part of this
build. Boundary unit replacement is closed after deterministic smoke attempt 3.
Its disabled X-P slot now accepts only the registered **`p2-l2-init`** variant.
Old P utility/replacement/smoke receipts cannot enable P2.
Reviewed `m2_train.py` and `m2_score.py` remain byte-identical. X imports their
recipe/conversion and scoring function through private adapters.

Stage M through M10, M9's durable CURRENT, Screen S through S6 (not its result),
complete exposure proofs for all
three B D+replay finals. ContractNLI targets/schedules are reused from **M3**.
`X_SPEC` is prospective JSON containing `p_variant:"p2-l2-init"` to enable P2
(or `"disabled"` to leave X4/X6 deferred), plus:

- `incomings`: exactly keys `101` … `103`, each `{path,sha256}` of a closed
  receipt `{complete:true, checkpoint_identity, parent_identity, pools:[…]}`.
  `checkpoint_identity` is `s_common.digest` of the exact M2 B reference copied
  to X1's binding; `parent_identity` is M2's checked base identity. Each receipt
  attests the entire adaptation ancestry. Pool references are
  `{role,path,sha256}` canonical JSONL membership, including all rejected
  candidates. Required nonempty roles are `b-ifeval-candidates` (2,048),
  `broad-gsm8k-candidates` (1,024), `b-finqa-training`, plus every other ancestor
  pool. Declare `v4-finqa-dev` if used.
The B FinQA training membership stays inside the closed ancestry proofs.
P2 requires no separate FinQA utility-target file, no 256-row utility selection
and no replacement receipt.

X2 extends S's audit against the new closed ancestry and actual admitted pools.
It retains S's disjoint partitions and hash order, applying the unchanged S/B
exclusions without repartitioning or changing S files/registration/analysis.
Declared/exact FinQA dev exposure invalidates that domain. Incomplete proof
invalidates support. X3 uses S's unchanged generator/checkers anew at all three
incoming checkpoints: greedy once, frozen raw journals, first **64 verified
monitors / 128 verified probe-training rows**, no replenishment. Every allocated
domain and all three seeds are required. Shortfalls yield an UNINFORMATIVE
comparison receipt, never dropped domains/seeds or zero signals. Generic-only
is DEGENERATE for both A arms; X3 generation skips, controls and P still run.
Incoming scoring runs once as `pre`; control/drift/plasticity use hash-bound
aliases of those rows. `reading.json` preserves DEGENERATE/UNINFORMATIVE status.
Interrupted attempts move partial output/metrics aside; drift retains its
immutable `global_step_10` prefix. Three fresh controls
(`x-control-s101` … `x-control-s103`) are shared by all challengers.

Use a fresh shell and the existing job-start/restore/identity/process-group
wrapper. Keep writers serial; the eight-GPU training/probe node is exclusive.
X1 binds the core `m2-scoring-machine.json`. Every scoring operation uses the
same inference environment, host and physical scoring GPU as the core. A mismatch
leaves every mixed X/core bar `PENDING_CROSS_MACHINE`. X-only paired bars retain
their own machine binding.

**Same-node presence check on every start/requeue:** before the boundary block,
restore M and S with their existing handoff scripts if needed, then verify that
`M2_OUT/binding.json`, `M2_OUT/admission.json`,
`M2_OUT/m2-scoring-machine.json`, `M2_DURABLE/CURRENT.json`,
`S_OUT/binding.json`, `S_OUT/audit.json` and `S_OUT/frozen/manifest.json` exist
on this node, together with every path in X1's bound assets, sources, panels and
incoming checkpoints. The X restore restores only X; it does not copy M/S or
their external assets. `x_common.context` rechecks the core hashes on every row.
Do not proceed with missing local inputs.



Set the paths, check local presence, then run the boundary block:

```bash
export KIT=/absolute/reviewed-checkout/kit
export M2_OUT=/local/nvme/message2 M2_DURABLE=/absolute/durable/message2
export S_OUT=/local/nvme/screen-s S_DURABLE=/absolute/durable/screen-s
export X_OUT=/local/nvme/addon X_DURABLE=/absolute/durable/addon
export X_SPEC=/absolute/send/addon-spec.json P_SMOKE=/absolute/send/x-integration-p2-smoke.attempt1.json
export FINQA=/absolute/data/finqa SDPO=/absolute/pinned/SDPO BASE=/absolute/cache/student
export TRAINER_PY=/venvs/trainer/bin/python INFERENCE_PY=/venvs/inference/bin/python SCORE_GPU=0
export OUT="$X_OUT" DURABLE="$X_DURABLE"
for f in "$M2_OUT/binding.json" "$M2_OUT/admission.json" \
  "$M2_OUT/m2-scoring-machine.json" "$M2_DURABLE/CURRENT.json" \
  "$S_OUT/binding.json" "$S_OUT/audit.json" "$S_OUT/frozen/manifest.json"; do
  test -f "$f" || { echo "Missing same-node input: $f" >&2; exit 1; }
done
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action job-start || exit $?
readonly PARENT_OUT="$OUT" DURABLE_ROOT="$DURABLE"
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action restore --move-aside || exit $?
"$INFERENCE_PY" "$KIT/m2_handoff.py" --out "$OUT" --durable "$DURABLE" --action identity --gpu "$SCORE_GPU" || exit $?
export TERM_LEAD_SECONDS=1800 SAVE_ALLOWANCE_SECONDS=900
source "$KIT/message_job.sh"
xrow() { run_stage "$INFERENCE_PY" "$KIT/$1" --out "$X_OUT" "${@:2}"; }
xrun() {
  xrow x_run.py --action "$1" --gpus 0,1,2,3,4,5,6,7 --score-gpu "$SCORE_GPU" \
    --panel-root "$FINQA" --sdpo "$SDPO" --tokenizer "$BASE" --smoke "$P_SMOKE" \
    --trainer-python "$TRAINER_PY" --inference-python "$INFERENCE_PY"
}
```

| Row | Work / resource | Command (in order) |
|---|---|---|
| X1 | Bind resolved durable M9 hash, M10 disposition, B finals, M3 targets, panels/ancestry/code; CPU | `xrow x_bind.py --durable "$X_DURABLE" --message2 "$M2_OUT" --message2-durable "$M2_DURABLE" --screen-s "$S_OUT" --screen-s-durable "$S_DURABLE" --spec "$X_SPEC"` |
| X2 | Extend exposure proof and retain partitions; CPU | `xrow x_support.py --action audit` |
| X3 | New verified own responses and local support receipts; one GPU | `xrow x_support.py --action generate --gpu "$SCORE_GPU"` |
| X4 | Dedicated D measurement on three untouched incomings; freeze median λ, first inputs and identity-bound eight-rank shard/tie/padding receipts; eight GPUs | `xrun measure` |
| X5 | Common incoming scoring, three fresh fixed-mix controls using X4’s first-update cache when valid (native batch otherwise), and scoring; eight GPUs | `xrun controls` |
| X6 | P2: disposable full-path node feasibility, restart untouched incoming/EMA, analytic L2 for 40 scored updates; eight GPUs | `xrun p` |
| X7 | A drift fixed-mix updates 1–10, retain full state; eight GPUs | `xrun drift-prefix` |
| X8 | Incoming/update10 monitor NLL, freeze shares/slots, resume updates 11–40, score; eight GPUs | `xrun drift-continue` |
| X9 | Independent incoming CE probes/cleanup; standalone fixed plasticity stage/scoring; eight GPUs | `xrun plasticity` |
| X10 | Node per-comparison tables only; CPU | `xrow x_read.py --action tables` |
| X11 | Pricing arithmetic and measured overrides; CPU | `xrow x_price.py` (add `--timings /absolute/measured-timings.json` when available) |
| X12 | Durable save, separate archive and job-end save including failures; CPU | Return block below |
| X13 | Manager frozen reader with our-side Screen S qualification; CPU | Read command below |

### Speculative tier 2: X14–X24

These rows extend the same X job and durable tree after X1–X12 finish. X13 is
an optional early manager read, never a node prerequisite: the node does not
wait for our-side Screen S qualification or interpret speculative efficacy.
Finish the tier-1 return before starting tier 2, so its archive survives any
tier-2 failure. The final X23 archive adds the tier-2 evidence. Keep M/S/C files
and their jobs untouched. Requeue the same commands with the same binding.

```bash
x2run() {
  xrow x_tier2.py --action "$1" --gpus 0,1,2,3,4,5,6,7 --score-gpu "$SCORE_GPU" \
    --panel-root "$FINQA" --sdpo "$SDPO" --tokenizer "$BASE" \
    --trainer-python "$TRAINER_PY" --inference-python "$INFERENCE_PY"
}
```

| Row | Work / resource | Command |
|---|---|---|
| X14 | Verify X12 marker/archive hash; freeze candidate identities and per-signal/P2 technical dispositions; CPU | `x2run plan` |
| X15 | Fixed-drift A+P2 versus fresh fixed-drift A; 3+3 stages | `x2run stack-drift` |
| X16 | Standalone-plasticity A+P2 versus fresh standalone-plasticity A; 3+3 stages | `x2run stack-plasticity` |
| X17 | Adaptive drift B versus fresh fixed-drift A; 3+3 stages | `x2run b` |
| X18 | B+P2 versus fresh fixed-drift A+P2; 3+3 stages | `x2run b-p2` |
| X19 | Incremental plasticity versus fresh fixed drift, common 10-update prefix; 3+3 stages | `x2run incremental` |
| X20 | Incremental plasticity+P2 versus fresh fixed drift+P2, common prefix; 3+3 stages | `x2run incremental-p2` |
| X21 | Refresh separate per-comparison tables without admission; CPU | `xrow x_read.py --action tables` |
| X22 | Price executed matrix, skipped branches and unpriced additions; CPU | `xrow x_price.py --tier2` (add measured `--timings`) |
| X23 | Save durable state, collect extended X archive, save again; CPU | Repeat the return block below |
| X24 | Manager validates/recomputes both tiers, then binds and reads; CPU | Repeat the manager read command below |

Append `|| exit $?` to each invocation. Each comparison owns two distinct run
identities and three seeds per identity. Neither another comparison nor tier 1
supplies a training control. Incoming accuracy may be aliased by checkpoint
identity; the independent optimiser/EMA trajectory is always executed afresh.
P2 uses X4's frozen coefficient/anchor without recalibration. Tier 2 uses native
rollouts for every arm; it never substitutes X4's first batch (which would not
match standalone plasticity's replay allocation).

Drift measurements use each arm's own stage-start checkpoint and frozen own
responses. Bind the stage-start NLL at k10 once and reuse that literal baseline
at k20/k30/k40; never re-measure it at a later refresh. Fixed drift freezes shares after update 10. Adaptive drift refreshes
at 10/20/30 for the following windows: initialise with positive step-10 drift,
then EMA `0.5 * previous + 0.5 * positive current drift`, always relative to stage
start. Retain the update-40 refresh descriptively, with no subsequent allocation.
Apply A's floor/normalisation, largest remainder with domain-name ties,
fixed task positions and total replay slots. Only B uses per-window largest remainder and absolute hash-selected reservation
phases to preserve every rolling eight-update window across refreshes. Fixed drift,
standalone plasticity and incremental plasticity call `x_train.targets`, exactly
as X8/X9 do, with tier 1's `addon-slots-v1` schedule (start 10/0/10 respectively). Native
full-state saves/restores retain model, optimiser, scheduler, RNG, EMA and
dataloader position; training remains 40 updates with its original warmup.
Incremental plasticity uses the independent incoming-checkpoint probe signal
after the ten-update incumbent prefix, identically in both P2 statuses.

Generic-only skips all tier-2 allocation comparisons while standalone P2 stays
eligible. Support is checked per signal over every allocated domain and seed:
plasticity shortfalls leave drift comparisons open; drift shortfalls leave
standalone-plasticity stacking open. P2 infeasibility closes only comparisons
containing P2. Missing prerequisites stay pending; resolved technical failures
have skip dispositions. Runtime failures are recorded per arm/seed, retaining
other comparisons and all tier-1 evidence. No efficacy outcome changes execution.

The manager first resolves tier 1, reads only `stack-drift` or `stack-plasticity`
when both its components pass, then binds the downstream P2 status. B can be read
only against admitted fixed drift. Incremental plasticity can be read only
after the bound B comparison's resolved non-admission and qualified plasticity;
B admission defers it to a new registration. Pending stacking/B blocks dependent
reads. All other outcomes are archived by identity/status without efficacy
interpretation. Intervals are per comparison over the three matched seed pairs.

The registration's **five comparisons / 30 stages after binding** become up to
**six speculative comparisons / 36 executed stages before our-side reads**,
because both possible stacking candidates must be available. At the D proxy,
tier 2 costs **278.4 training GPU-h**, or **295.2–316.8 GPU-h including ordinary
scoring/startup**, before repeated drift scoring, generation and P2 overhead.
The bound five-comparison registration remains **231.9 / 246–264 GPU-h**.
Reduce future stage counts for recorded skips; retain pending branches in the
upper estimate and charge actual failed work. Incoming plasticity probes are
frozen measurements shared across runs and charged once at tier 1's `3K`; they
are never reused training controls. Measured timing keys additionally include
`tier2_generation_gpu_hours`, `tier2_scoring_gpu_hours`,
`tier2_repeated_drift_gpu_hours`, `tier2_regulariser_gpu_hours`, and
`tier2_startup_allocation_gpu_hours`; missing values remain unpriced.

The expanded full-cohort forecast covers **444,015 score rows** and **63,145
inherited rows**: **23,337,430 bytes**, including twice the compressed text and
a run-scaled 11.75 MiB receipt reserve, **20.5% below the 28 MiB limit**. The reserve adds 64 KiB per extra run to tier 1's 8 MiB for the 60 extra runs. The rehearsal also measures compressed metadata
for every configuration and checks it against the reserve. Collection checks the
actual compressed size and readback. For the full CPU rehearsal use
`python3 kit/addon_dryrun.py --out /absolute/fresh-temp --standin --tier2`;
it runs X1–X24 for all four pool configurations and requeues every row. Delete
the temporary directory afterwards and stop if free disk falls below 12 GiB.

Append `|| exit $?` to every X invocation. On requeue repeat the boundary block
and unfinished rows. Identical completed training, generation, probes, P and
scoring SKIP after identity checks. EXIT/TERM saves failures too. X12 can collect
partial evidence; keep every failed attempt on durable storage.

P2 is a separate registration, not a replacement fallback selected at runtime.
X1 binds its identity and the M2 base checkpoint, verifying the registered student
hashes/revision `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`.
Use a fresh X tree; an old replacement tree cannot be upgraded in place.

X4 runs three dedicated first-update D+admitted-replay passes on eight GPUs,
from the exact untouched incoming FinQA finals and exact first scheduled batch.
The core D numerical policy is unchanged for every X arm (including controls, A
and tier arms): no X-specific strict deterministic algorithms, cuBLAS/hash
overrides or `FLASH_ATTENTION_DETERMINISTIC` flag. This is the binding manager
decision after review. The pinned vLLM sampler defaults remain seed 0,
temperature 1, top-p 1, top-k -1; no unsupported `rollout.seed` override is added. Each full actor-input DataProto is frozen: generated tokens, masks,
teacher contexts, replay targets, old/rollout probabilities and importance
inputs. Each rank's pre-actor RNG state is bound too. X5/X6 consume that exact
cache at update 1 and restore its actor RNG; the normal first sampler invocation
is consumed before substituting the cache, preserving the stream position.

X4 intercepts D after accumulation/reduction/unscaling, before clipping. It
performs **no optimizer, scheduler or EMA step**; per-rank semantic state hashes
must agree before/after (RNG restored). Each phase then starts fresh from the
untouched incoming. All three positive finite task-gradient and base-distance
norms are required. `p2/calibration.json` freezes
`lambda = .10 * median_s(task_grad_norm_s / incoming_distance_s)`, all three
realised ratios, exact schedules/cache/RNG hashes, anchor, code and passing
`rung-p2-distributed-smoke.v2` receipt from the P2 part of the split integration smoke. The separately launched A-resume part cannot qualify or defer P2. No recalibration or smoke λ transfer is allowed.

X6 adds `lambda * (student - base)` once per optimizer update, through the X-only
worker/actor extension between D's unscale and existing clip/optimizer/EMA
sequence. The base is a frozen **fp32** matching local shard, not a full second
model per GPU. The pinned FSDP2 DTensor Shard(0) layout and legacy FSDP1 flat
shards are supported; padding is excluded, tied parameters count once, and
bounded 65,536-element differences avoid full-model temporaries. There is no
parameter-count, microbatch or world-size multiplier. Existing weight decay
stays active. Every trainable student parameter is included. The teacher is
never regularised; it follows D's ordinary .99/.01 EMA after optimization.
Controls and A bypass the regularizer; controls share only the first-batch
transport. A has no P2 actor hook, anchor or cache substitution. Initial
student/teacher identities remain the untouched incoming. Floating-point gradient
reproduction is diagnostic: X5/X6 and the reader record relative deviations from
X4, warn above `1e-3`, never fail for that deviation, and never change λ.
Per-rank receipts carry distances, gradient ratios, λ, `boundary_seconds`,
`max_memory_reserved` and the full 40-update dose. X4 records memory peaks too.

If X4 or calibration validation fails, X5 runs all controls with the ordinary
native first batch, records `CONTROL_WITHOUT_CACHE`/`NATIVE_NO_CACHE`, and defers
P2. X6 failures retain per-seed receipts and do not stop A. The reader marks bad
P2 evidence **INCONCLUSIVE**, retains its reason, and continues A and the remaining
reader order. It never treats missing P2 evidence as a zero or a negative result.
Step-1 `rollouts/` logs contain the freshly generated batch discarded by cache
substitution; `cached-first-batch.json` explicitly labels that log and binds the
actual trained X4 DataProto by hash.

First run the prepared **CPU-only memory probe** (the build does not launch it):

```bash
python scripts/probe_modal_memory.py
```

It reserves 2 CPUs/8 GiB for at most 600 seconds, no GPU, one Sandbox and no
retries, at a printed **$0.41 ceiling** including margin/build allowance. It
records cgroup v1/v2 memory files and visible ancestors, then attempts real
20 GB writes and deletion in /tmp and the working directory with live samples.
If non-reclaimable memory reaches 85% of the effective limit, it stops that
write early, labels the byte count and proceeds to investigate the other
directory, rather than losing all evidence to an OOM. No 20 GB write is run
by the local tests. The evidence path is
`docs/phase3/evidence/modal-memory-probe.attempt1.json`.

The 25 October probe is complete: both 20 GB files were written and deleted
in an 8 GiB sandbox. The amendment's **Probe result** resolves disk writes as
not memory-accounted. Its raw automatic classifier remains `null` because
cgroup stat/peak/OOM counters were unavailable; the launcher binds the exact
probe SHA and records that registered resolution separately. The Modal host
gate is now the labelled `/proc/*/status` **per-process resident-memory sum**,
sampled at 100 ms and boundaries, with shared pages conservatively counted per
process and 5% headroom against **64 GiB**. Missing cgroup counters remain
unknown, never fabricated as zero. Preparation retains its full-state update-4
save; the former 78 GiB/model-only contingency is disabled. The Slurm node gate
below continues to require its own cgroup accounting and known zero OOM kills.

The revised 25 October smoke runs **inside the real X integration smoke**, split
into two separately launched containers. This build has **not launched either**.
The retired `scripts/smoke_rung_p2_modal.py` refuses execution and points here:

```bash
modal run --detach scripts/smoke_x_integration_modal.py --part p2
modal run --detach scripts/smoke_x_integration_modal.py --part resume
```

Each authorised attempt has exactly **one** explicitly launched Sandbox, no retries or
recreation, two H100s, hard 8 CPUs/64 GiB and the historical 1800-second lifetime for attempts 1–3.
At rates GPU .001097 USD/s, CPU .00003942 USD/core/s and memory .00000667
USD/GiB/s, 30-second termination slack, 25% resource margin, $.05 writer allowance
and $3 build reserve give **$9.77 per part** (up to **$19.54 for both distinct
checks**). Each original launch needed its own manager-written ledger row containing its
evidence stem followed by `attempt 1`; attempt 4 has the stricter row check below. This build changes no ledger row.
Atomic per-part claims count lost-client launches; launching the other part is
independent, and a second launch of the same part requires separate authority.
The shared writer produces `x-integration-p2-smoke.attempt1.json` and
`x-integration-resume-smoke.attempt1.json`. Schema `x-integration-smoke.v3`
records the part, forecast, probe hash/resolution and its own ceiling. Only
`--part p2` evidence may enable P2; resume evidence can never enable it.

The owner authorises **P2 attempt 4** (amendment's 26 October Attempt 4
sentence): **3000 seconds**, no retries, **$14.17** priced ceiling (<= $15).
The same rates and margins above use 3030 seconds including termination slack.
Attempts 1–3 and the consumed resume attempt remain immutable with their old
limits. Attempt 4 requires the exact manager-written `x-integration-p2-smoke
attempt 4` ledger row with ceiling `≤$15`
and a current `docs/phase3/evidence/p2-cpu-stage-audit-20261025.json` **before**
its atomic claim. The receipt binds the input/command code and audit-test hashes;
a missing, failed or stale audit refuses launch without consuming the attempt.

`tests/test_x_smoke_stage_audit.py` writes the exact preparation and development
parquet with the smoke's input helpers. For preparation, X4, X5, X6, drift prefix
and resume it runs four batches through pinned verl's real RLHFDataset and
collation, the real mixed teacher builder and the production teacher-entry
function. Resume's audit starts at batch 3 after the two consumed prefix batches.
It checks the production generic map **28 task +4 replay at 8/16/24/32**, native
full-pool D eligibility, EOS and input/context limits, reward schemas, resolved
objective/contract settings, every override against pinned YAML defaults and
all required Ray-forwarded environment variables. No trainer/model/GPU is run.
Preparation's short `Answer: yes/no` labels are wrapped as a labelled frozen
demonstration because those literal strings also occur in FinQA's format
instruction; the original question and replay targets remain unchanged.
The P2/control actor-input pairing introduced for attempt 2 remains mandatory
for attempts 3 and 4. No registered norm, timing, precision or contraction gate changes.

Use Qwen/Qwen3-1.7B pinned at
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (the existing phase-4 pin), two-rank
FSDP2, production D builders, worker, actor, clipping, optimizer and EMA.
The smoke alone uses attempt 3's strict deterministic mode and eager attention;
node arms retain the unchanged D numerical policy. Preparation uses four
unregularised native D+replay updates on the frozen FinQA inputs: 28 FinQA task
contexts and four FinQA CE replay rows per batch; the prepared incoming is
exported from real fp32 checkpoint shards. Fresh paired controls/P2 each run
four D+replay updates at `[0,1e-6,2e-6,3e-6]`. The smoke anchor is the same pinned
1.7B **fp32** base, never the 8B node anchor. Seed-101
`lambda_smoke=.1*first_task_gradient_norm/incoming_distance` freezes before the
paired updates. The native X4 measurement supplies that gradient after
accumulation/reduction/unscaling without an optimizer/scheduler/EMA step.

Every rank records the actual actor `config._attn_implementation`, the complete
returned strict-determinism configuration, and identical starting student/teacher hashes, finite
losses/gradients/Adam/EMA, analytic value and gradient identities, zero-at-base,
unique tied-parameter and shard coverage, unchanged anchor, absent teacher
gradients, and no extra microbatch/world-size factor. Require all four pre-clip
P2/control norm ratios in [.5,2], median updates 2–4 time <=1.1x, GPU/host peaks
leaving 5% headroom, and
`D_control_4-D_p2_4 >= 1e-4*D_0 > 0`. Both full update time and time excluding
diagnostic audits must pass; the latter retains the synchronised chunked delta/add wall time (including host dispatch) and GPU
operations, clipping, optimizer and EMA. All three live distance points use
actual fp32 masters and fp64 accumulation. Actual start/end HF exports retain
fp32 checkpoint precision through an X-only exporter (verl's ordinary exporter
casts to bf16). Their distances must match the live values; incompatible
precision refuses the smoke. Per-rank trajectories and exported distances
remain in the shared return. No coefficient search or efficacy claim follows.

Preparation saves only at update 4; rollout `enforce_eager=True` avoids CUDA
graph warmup. The **P2 part** retains X4 no-state-mutation and X5/X6 checked
final update-4 saves only (X4 has no save); no P2 check uses update-2
checkpoints. Paired inputs are captured at the actor boundary, independently
of saves; full-state restoration belongs to the resume part. After X6 update 4,
workers export the actual fp32 student DTensor shards before native checkpoint
save; the driver merges them and runs every registered gate, including actual
exported-distance agreement. It durably publishes `P2-smoke-result` before
returning to the native final optimizer/EMA checkpoint save. A later final-save
timeout preserves that complete result via the existing verdict-scope rule.
It ends without running A. A published
PASS qualifies P2 independently of the separately launched resume part.

The **resume part** starts independently from the untouched pinned 1.7B base
on the same frozen development FinQA/legal inputs. It checks A prefix 2 →
full-state resume → update 3 with changed unconsumed replay; it performs no
P2 preparation, measurement, cached-batch substitution or regularisation.
It uses save_freq=-1 on resume: no update-3 checkpoint or checkpoint-memory-step3
requirement. A failed or missing resume part is **UNVERIFIED**; the node's
fail-closed restore qualification before update 11 remains its guard.

Development legal inputs are labelled stand-ins and frozen alongside the FinQA
rows. The appended `rollout.response_length=128` bounds sampling while
`data.max_response_length=2048` remains registered. Every stage emits
`[record-partial]`. Checkpoint hooks retain raw cgroup readings where available;
checkpoints are deleted after checks. Update-4 shards survive only until the
fp32 export, and the A prefix survives only until live restore qualifies.
Each container supervisor records the registered aggregate-cgroup peaks from startup
(RSS is diagnostic only; missing aggregate telemetry cannot pass).
Missing production precision or any P2 mechanical gate defers P2. Failure
before P2's own PASS never qualifies it; resume evidence is never a substitute.

`kit/resources/p2/split-forecast.json` freezes per-part forecasts from the
measured **292.907 s** one-GPU/two-update D launch, with no credited two-GPU
speedup or cheaper merges/saves. Each part charges its own 120 s cold startup.
For P2, four launches / 13 actor calls, planning 30 s per extra update gives
**24.03 min**, with the 0–60 s/update range **21.53–26.53 min**. For resume,
two launches / three actor calls give **11.76 min**, conservatively retaining
the two-launch/four-update basis without subtracting the unused update.
The original launches required each upper forecast **<=27 minutes** before
reserving an attempt; if exceeded, report rather than cut checks. Eager/deterministic
slowdown is unmeasured: these are forecasts, not completion guarantees. A
further 15% slowdown takes the upper P2 scenario to 30.21 min; the hard timeout
remains 1800 seconds. Every launch retains the registered checks.
Attempt 4 supersedes the P2 forecast with
`kit/resources/p2/attempt4-forecast.json`, parsed from attempt 3's measured
preparation (426 s), X4 (102 s), control (401 s), and truncated X6
(305 s through update 3) progress and its **1831.055 s** total elapsed evidence.
Completed update metrics span 89.86–113.10 s; X6 update 4 completed but its
final timing row/save did not return. No unobserved completion is claimed.
Central forecast is **33.52 min** (observed total +120 s live export +60 s
final-save tail); upper is **41.62 min** (1.2x observed total +180 s export
+120 s final-save tail), required **<=45 minutes** before the atomic claim.
No time credit is taken for removed intermediate saves. The resume forecast
and limits remain unchanged; the new P2 hard timeout is 50 minutes.

The old CPU-offload forecast (5.45 s/add, 37.9 minutes) is historical rationale
for the new placement, not a GPU-path performance claim or launch gate.

The small smoke **cannot qualify the node**. Frozen prospective limits are in
`kit/resources/p2/node-limits.json`, bound by SHA through X1, X4, X6 and the
reader. Measured D source files and their hashes are listed there: NOSKIPFWD
peaked at 61.778 GiB allocated / 70.045 GiB reserved per GPU; SWITCHREV reached
293.207 GiB host used. Allowing approximately 3.83 GiB/rank for the fp32 anchor
plus transients, freeze allocated <=72 GiB, reserved <=actual total minus
2.5 GiB, actual GPU total >=79 GiB and device free >=max(1.5 GiB,2%).
Use the smallest finite cgroup/ancestor memory.max as the effective host limit,
requiring >=512 GiB, sampled non-reclaimable peak <=448 GiB, and free
>=max(64 GiB,12.5%). The non-cache D basis 293.207 GiB +30.55 GiB total fp32
anchor +91.65 GiB serialization allowance gives 415.407 GiB; rounding to
448 GiB leaves about 32.6 GiB additional margin. Linux psutil.used excludes
buffers/cache, so raw page-cache-inclusive cgroup peaks are not this basis. These are prospective caps, not
claims that an unmeasured configuration fits.

X4 binds actual eight-rank parameter/anchor ranges, tied aliases, fp32 dtype,
zero padding and complete nonoverlapping coverage. Before scored P2 training,
X6 runs a **disposable full 40-update** pass on the actual 8B/eight-rank recipe,
same untouched incoming, frozen λ, first rollout/RNG and admitted replay. It
observes all materialised Adam states, teacher, anchor, rollout/optimizer/EMA
transients and checkpoint GPU/host peaks. Distinct physical GPU identities,
full-path finite state and frozen headroom limits must pass. GPU receipts also
include device-used memory sampled every 100 ms and at boundaries, alongside
allocator lifetime peaks; headroom uses the larger recorded device-used or
reserved value. Non-Torch allocations cannot hide behind allocator accounting. Scored training
then starts a **new process from untouched inputs**, never the preflight output.
Host gates use sampled anon+shmem+kernel; smoke falls back to a labelled /proc
RSS sum, while node old-kernel telemetry can use memory.current-file first.
Record raw memory.peak/current separately, plus all ancestor limits and OOM
events. The relevant job cgroup's oom_kill must be known and zero; unknown
counters refuse qualification. Missing memory.peak alone does not fail a gate.

Resource receipts bind incoming, seed, admission/configuration, anchor,
calibration, code and limits; scored-stage receipts bind the preflight receipt
and must satisfy the same limits. Missing/stale/failed receipts defer only P2,
leave controls/A/core collection usable, and make P2 INCONCLUSIVE at readback.
The disposable pass saves only at update 40. A Slurm-interrupted, receipt-less
run is preserved as run.interrupted-1 and gets at most one clean restart from
untouched inputs; no checkpoint resume. Completed failed passes remain failed.
Every P2-containing arm, including a future named/declared stack, routes through
this gate; ordinary controls and A bypass it. Its extra three stages
cost a D-family proxy **38.67 GPU-h**, separately itemised by `x_price.py` and
replaced by actual cost records when measured.

X4 has an a-priori allowance of **5–7 GPU-hours**, including three eight-GPU
initializations and gradient-only passes. Replace it with measured process costs
when available. The collector sums per-rank `boundary_seconds` as regularizer
GPU-hours; these are already part of stage process cost and must not be added
twice to charged totals. No strict-determinism slowdown is priced on X because
its numerical policy is unchanged.

Shares use the registered floor/normalisation; all-zero positive signals retain
incumbent shares. GSM8K/ARC remain separate: K=3/2/4 for broad/earlier/both,
with 8/8/12 replay positions of 32. Largest remainder ties use domain name.
`addon-slots-v1` hash-selects periodic reservations for every rolling eight-update
window, then hash-orders the remaining slots. Totals differ from ideal shares
by less than one slot. Task positions, replay totals, pool contents, prompt-cap
skipping and losses are preserved. Drift retains the consumed 320 examples and
replay draw positions. Its full 1,280-row schedule and **40-update scheduler**
exist from the start. The opt-in checkpoint hook stops only after update-10
model/optimizer/extra, EMA and dataloader saves. Native verl plus `v4_state`
restore these states for updates 11–40; HF weights-only restart is refused.
Before stopping, `prefix.json` records the full `global_step_10` file-tree hash,
each rank's live model/optimizer/RNG/scheduler/EMA hashes and rank-0 loader hash.
The continuation verifies the tree before loading, then recomputes every live
hash after loading, requiring trainer/actor/scheduler step 10 and ten yielded
batches. It writes `restore-qualified.json` before update 11 and refuses any
mismatch. These on-node checks qualify this restoration without a return cycle.
The vLLM sampler RNG is **not checkpointed**: the sampler is reseeded at this
process restart. Model/optimizer/scheduler/EMA, Python/NumPy/torch worker RNG
and data position remain subject to exact live-state hash equality; sampler
continuity is not claimed.
Interrupted continuation reloads the same full prefix. Standalone plasticity
uses independent fp32 four-update incoming forks through S10's machinery;
probe weights never initialise a training arm and exports are deleted after NLL.

M graders/decoding score ContractNLI at 0/10/20/30/40 and FinQA, general-v1,
hard-MATH retention and HellaSwag-100 at 0/20/40. P2 gain-AUC subtracts the common unchanged incoming accuracy in both arms;
its step-0 scores alias the common incoming by identity. Retention is final minus untouched incoming. Its protected
equal-weight mean uses exactly the admission-dependent instruction/maths/
knowledge/FinQA set registered in §2. Intervals preserve the inherited package's
three-paired-seed t bounds (one-sided 95%, df=2), separately from secondary
ContractNLI document bootstrap. Each comparison has its own interval. The core structurally produces only `standalone-legal-plain`, not a
replay-compatible standalone: `connected_vs_standalone` is reported as
`PENDING_NO_COMPATIBLE_COMPARATOR` and is excluded from admission. The
learning-gap context is likewise report-only. All other inherited admission
bars are structurally available: FinQA retention, ContractNLI final/gain and
base-relative instruction/maths/knowledge cumulative bars. Missing base rows
leave those bars pending; they are never silently waived. Transitions
report wrong-to-right learning separately from acquired-answer survival.

The node produces per-comparison tables only. Our-side Screen S analysis
uses `x_qualify.py` below to run the unchanged frozen
`scripts/screen_s_analysis.py` and bind its result to the Screen S registration,
analysis-code and actual return-archive SHA256 hashes. The reader verifies and
recomputes that qualification from `--screen-return`.
The manager reader requires it, checks drift association **and** forward PASS,
then reads eligible drift; standalone plasticity is read only after resolved
drift non-admission and plasticity qualification. Unqualified/unbound A tables
remain archived without efficacy interpretation. P2 follows independently of signal qualification; if its Screen S return is
missing, only P2 is interpreted and A/stacking remain pending. If A and P2 pass,
retain A provisionally until the bound X15/X16 stacking comparison resolves.
X24 then reads the eligible B and incremental comparisons in the frozen order
above. An early X13 return without tier 2 retains the message-3 fallback.
All eligible comparisons are reported together.

```bash
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$X_OUT" --durable "$X_DURABLE" --action save || exit $?
xrow x_collect.py --archive "$X_OUT/addon-return.tar.gz" || exit $?
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$X_OUT" --durable "$X_DURABLE" --action save || exit $?
# Manager, after receiving Screen S:
python3 kit/x_qualify.py --archive /absolute/returns/screen-s-return.tar.gz \
  --output /absolute/returns/screen-s-qualification.json
python3 kit/x_read.py --out /absolute/reading-addon --action read \
  --source /absolute/returns/addon-return.tar.gz \
  --screen-qualification /absolute/returns/screen-s-qualification.json \
  --screen-return /absolute/returns/screen-s-return.tar.gz
# CPU only, fresh directory; delete afterwards:
ADDON_TMP=$(mktemp -d "${TMPDIR:-/tmp}/addon-standin.XXXXXX")
python3 kit/addon_dryrun.py --out "$ADDON_TMP" --standin
rm -rf -- "$ADDON_TMP"
```

Return the separate <28 MiB archive, allocation/restart records, measured timings,
charged allocation hours and durable location. Raw responses, schedules,
utilities and weights remain durable with compact receipts/hash indexes.
The return contains one merged `scores.csv.gz` and `inherited.json.gz`; the
28 MiB limit applies to the compressed archive. The dry run forecasts the full
registered cohorts (ContractNLI 100, FinQA 1,147, hard-MATH 100, three general
panels of 100 each, HellaSwag 100), with twice the compressed score/inherited
text plus an 8 MiB tier-1 receipt reserve (tier 2 scales it as above), requiring
at least 20% margin. The dry run
checks free space on its `--out` filesystem and stops below 12 GiB.
Pricing: **20 × 58 minutes × 8 GPUs = 154.67 ≈155 training GPU-h**, plus
**3K × .75–1.4** probe GPU-h (**15–28** with both pools), plus separately priced
generation/scoring/X4 measurement/regularisation/startup/allocation overhead. Generic-only plans
ten stages, no allocation probes. Timings JSON may provide
`contract_nli_stage_gpu_hours`, `generation_gpu_hours`, `scoring_gpu_hours`,
`p2_measurement_gpu_hours`, `regulariser_gpu_hours`, `startup_allocation_gpu_hours`. Missing terms remain
unpriced; CPU timing cannot price GPU work. Reprice shortfalls/failures and
ContractNLI wall time before SEND.


Tier-2 integration with final rung P2 (25 October)

The six registered P2-containing tier-2 arms (the two stacking challengers,
B+P2 and its fixed-A+P2 control, incremental plasticity+P2 and its fixed-A+P2
control) reuse the **same seed's** scored-P2 disposable 8B/eight-rank node
feasibility receipt. This is a memory qualification, not a shared efficacy
control: every comparison still executes its own three fresh controls.
The incoming checkpoint, model revision, fp32 anchor, Adam optimiser, EMA
teacher, batch/sequence limits, FSDP2 width and frozen X4 coefficient are the
same. Allocation changes replay identities/shares, not this memory profile;
native first rollouts remain arm-specific. No extra disposable pass is priced
or launched for these six arms. Other future named/declared P2 arms remain
unregistered and deferred.

Tier 2 requires an existing valid seed-specific receipt and never manufactures
one or starts a new preflight. Missing receipts leave the P2 comparisons
pending; failed/stale receipts close only P2-containing comparisons. Each P2
stage binds the receipt hash, and production stages record their own eight-rank
resource measurements against the same limits (including intermediate native
checkpoint boundaries). The return reader independently validates the reused
receipt, binding and final resource records for every P2 comparison. Ordinary
allocation comparisons and tier-1/core returns remain independent of tier-2
failures.

Both local execution and returned reading use the same X4 calibration validator
and **part `p2` PASS** smoke validator as tier 1. A resume-only receipt cannot
qualify P2. The final verdict scope rule still accepts a recorded P2 PASS when
an unrelated later wrapper failure is explicitly `after-P2-verdict`; failure
before P2's PASS, absent receipts and changed receipt hashes block P2.


## Fix round 1: assignments, validation and failure handling

Use whole-node scoring allocations, e.g. `sbatch --exclusive --nodelist=<assigned-node>`;
alternatively run that node's family processes inside its one exclusive allocation.
A one-GPU Slurm allocation may give the wrong physical GPU. Before any restore,
the runner checks hostname and allocation-visible UUID, maps UUID to its physical
index with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and passes the index to the scorers.
Each inference/NLL process asserts its actual torch device-0 UUID. The manager's
one-H100 ARC Modal check exercises this same mapping/assertion path; run the
prepared one-GPU smoke on the partner stack before send as well.

Submit each batch script with `--signal=B:TERM@<restore+snapshot seconds>` using
measured worst-case restore plus snapshot/teardown time (return 1 restore took
about three hours). The executable templates use exec, so Python receives TERM
during staging as well as training. Children get a process-group TERM, a bounded
grace period, then KILL. Durable storage must mount with cross-node `flock`,
never `noflock` or `localflock`; LOCAL_ROOT must be PCIe NVMe. No PATH shims.
Exits: 0 completed, 2 UNVERIFIED/refusal, 3 pending, 4 child crash, 5 wrong slot.
Only the partner supplies submission/dependency policy; wrong-slot is not afterok.

M3 is pool-free. M4/M5/M6 proceed without M3b; M-pools gates R0 and replay work.
R-broad commits only the broad nomination so M8 can proceed while earlier-task
nomination is pending. R-nominate commits the final nomination before M9;
M7 needs neither nomination. R-collect independently returns pending or refused
re-probe attachments, labelled for review with no efficacy interpretation.

`resources/m2/format-validity.json` freezes the format-free HellaSwag likelihood
scorer, independent HF reference, preprocessing, fixed base tokenizer hashes,
character normalisation and lowest-choice-index tie rule. FV pre-gate jobs cover
the frozen base and incoming checkpoints on each decision family's assignment:
100 items, four finite likelihoods each, 100% selected-choice agreement. Failure
leaves affected decisions PENDING. Every candidate/reference then uses that same
format-free guard; strict HellaSwag accuracy and incorrect_format are diagnostics.
Knowledge and other unchanged strict-letter endpoints require zero strict-format
failures in every required cell. A failure is FORMAT-CONFOUNDED → PENDING.

Before send, `m2_send` and `m2_bind` require the reviewed acceptance record
`kit/resources/m2/arc-prompt-FROZEN.json` (default, or pass the identical
`docs/phase3/evidence/arc-prompt/FROZEN.json` with `--arc-prompt-check`). Its
reviewed SHA256 is `6ce5528c1f4cd2d75fca3f98aaadcb6f6f188dd21cfd8a912886d36e65ef8989`.
Attempt 2 passed with 128/128 strict parses and 120 accepted rows. The exact
passing wording is frozen in `next_pools.ARC_PROMPT`. Binding verifies this
record and the frozen >=127/128 parse / >=80 accepted thresholds; it never
loosens the parser or replenishes the sample. The pool is still regenerated
with registered exclusions and its first 128 ARC IDs must match that record.
The historical acceptance record does not substitute for the new partner-stack
UUID-mapping smoke, which the manager/partner must run before send.

The kit ships the redacted `resources/m2/return1-decisions.json`, pinned by a
constant. Its private original is referenced only by SHA256. Rescore path maps
use the neutral public checkpoint keys and the partner's actual local paths.
Run `python -m kit.m2_privacy` before publishing kit/docs. Original identifying
archives and unredacted evidence stay private. Scientific numbers/hashes remain.

**4A must be bound fresh with this kit**, since infrastructure fingerprints
changed. Its scientific registration, five seeds, separate nodes/state/assignment
and data isolation stay unchanged; use the required rapidfuzz and shutdown fixes.


Prepare the pre-send candidate freeze on CPU on the staged partner inputs,
before any base-answer generation (no GPU or network is used by this flag):

```bash
"$INFERENCE_PY" "$KIT/next_pools.py" --out "$NEXT_V2" \
  --package-a "$A_OUT" --finqa-panel "$FINQA_PANEL_JSONL" \
  --finqa-targets "$B_TARGETS" --gpu 0 --prepare-candidates-only
```

The first send jobs described above run this command on the staged inputs;
its receipts return automatically with core/S/X collection. This runs the complete registered exclusions, not inherited return-1
membership. M3b verifies that the first 128 ARC IDs after fresh exclusions
match the packed, hash-bound passing attempt-2 payload before generation. The
real broad file also contains GSM8K rows: its whole-file hash is recorded for
that preparation, and is not compared to the diagnostic ARC-only file hash.
M-pools repeats the ARC identity check. A changed ARC population keeps broad PENDING before generation/training; never
replenish it. This is an on-node dependency gate in the send, not a separate
partner message or a claim of locally available complete A/B inputs.


### Round-3 pre-send technical check (manager only; attempt 2 unlaunched)

This one-container check uses frozen Qwen3-8B at the pinned revision for the
base only. Incoming checkpoints are validated only on their assigned nodes. It runs the exact format-free HellaSwag scorer and independent
HF reference on the same frozen 100 ActivityNet items, with complete finite
choice coverage and 100/100 selected-choice agreement. It also archives a full
100-item general-v1 knowledge strict-format audit for the base; rates are
diagnostics. Knowledge failures veto only contrasts reading that strict endpoint,
through the existing core/S/X readers; they do not veto replay nomination or
M8/M9's math/survival/transfer/acquisition contrasts. Real incoming checkpoints
still receive assigned on-node validation and knowledge audits.

```bash
modal run --detach scripts/hellaswag_format_check_modal.py --attempt 2
```

The command prints and asserts its bound before creating exactly one H100
container: 8 CPU cores, 200 GiB RAM, 30-minute timeout, zero retries;
$5.926/hour × 0.5 hour × 1.25 + $0.05 = **$3.76 maximum planning bound**, rounded upward to a full cent.
Attempt 1 remains archived as an infrastructure failure (GPU memory retained
between engines); no agreement failure was observed. Every engine phase now
runs in its own subprocess, in the manager check and on-node likelihood jobs.
The shared writer archives the next run as
`docs/phase2/evidence/hellaswag-format-check.attempt2.json`.
Copy the passing receipt into the frozen send-input bundle. Pass that exact
copy with `--format-check` to both `m2_send.py` and `m2_bind.py`;
the binding checks its full declaration, frozen items/code/model hashes,
coverage/finiteness/agreement and knowledge audit, then freezes its SHA256.
The launcher source hash is carried in `resources/m2/manager-format-launcher.txt`;
the host checks its actual source against those frozen bytes before launching,
and partner receipt validation needs no manager scripts directory.
A missing/failing receipt keeps the real send pending. This build has not
launched attempt 2; the manager runs it after the code is frozen. Infrastructure
failures in assigned format-validation rows remain retryable; only observed
technical outcomes are bound permanently.

Durable preflight prints the actual filesystem type/options and performs a
held-lock exclusion plus released-lock acquisition in independent processes.
For multiple nodes, use a supported shared filesystem with remote flock enabled;
NFS `local_lock=none`, GPFS and Lustre are supported. A one-node run may use local
storage. The node cannot be queried from this build: retain the printed preflight
receipt with the partner's run. NVMe native multipath checks every controller's
transport and accepts only PCIe, including bind-mounted /tmp without device nodes.

Old partial snapshots receive a 24-hour prune grace period, active readers hold
shared archive leases, and optional pruning cannot invalidate a completed upload.
Budget local disk for selected native continuation state plus cached replicas;
ancestor-state deduplication/replica garbage collection remains a separate
infrastructure change requiring continuation qualification (optional review item 12).

Screen S keeps the registered 10,000 draws of three whole seed chains, RNG
20261025, empirical rank 250. With this fixed RNG/cohort, the lower bound selects
the minimum seed mean, so its positivity condition is redundant with 3/3
positive seed means. The mean >=0.5, difficulty baseline, tau-b cells and
association-plus-forward conjunction remain required.

The public export includes four byte-identical pinned sources in `scripts/`:
`check_pinned_inputs.py`, `screen_s_analysis.py`, `smoke_rung_p2_modal.py` and
`smoke_x_integration_modal.py`. Screen S/P2 require these for their on-node
implementation hashes and the S reader. The source files grant no launch
authority. The publisher privacy gate scans them and every exported file.
