# Send 6: v4 qualification and campaign

This engineering tree requires the manager's frozen commit, independent gates,
a live containment receipt and owner proceed before scientific GPU work.
[The single frozen-commit check list](../docs/phase2/v4-frozen-gate-commands.md)
contains unit suites, G1 identity, G5 rehearsals and G6 fresh-clone commands.
CPU stand-ins and synthetic G0 archives do not replace live qualification.

Stage the pinned trainer environment and a separate pinned vLLM >=0.17 inference
environment on CPU. Set `WORK`, `KIT`, `SDPO_DIR`, `FINQA_ROOT`, `MODEL_DIR`,
`TEACHER_MODEL_DIR`, `QWEN3_8B_TOKENIZER`, `V4_TEACHER_PYTHON`,
`V4_INFERENCE_COMMIT`, `V4_KIT_TAG` and `V4_ACTIVATE`. Model paths must be real HF
snapshot directories with a 40-hex revision and registered config, tokenizer and
weight hashes. The manager registers the exact 27B revision and shards; a name
or alias alone is insufficient. `V4_ACTIVATE` is an existing absolute activation
script for the pinned trainer environment. Set the actual site's `PARTITION`,
`QOS`, `ACCOUNT` and wall limit; do not submit placeholders.

Before **any GPU allocation**, run CPU preparation. Its first operation is the
CPU import/entry/first-batch smoke; stage the pinned environment and 8B tokenizer
needed for that smoke beforehand. Optional static downloads
are explicit argv/deadline entries in `V4_STATIC_PREPARE_PLAN`. Preparation imports
every kit module and entry point, including qwen-vl-utils, smoke-checks all four
launchers and the patched SFT loader's first CPU batch, preprocesses/tokenizes
static data and freezes byte hashes for data, code, models and environments.
GPU rows verify those hashes before use. Dependent scheduling, coverage,
common-set creation, merging, gates and archive packing have explicit deadlines;
they run inside the allocation and their measured allocation cost enters admission.

```sh
export V4_SITE_MINUTES="$SITE_MINUTES"  # freeze site windows before preparation
python "$KIT/v4_prepare.py" --work "$WORK" --phase qualification
# CPU site settings check and required archived paste files:
# follow README-contain.md, outside SLURM_JOB_ID.
```

**In every new allocation the FIRST containment command is `selftest` with an
explicit block. Never reconcile, check or runner tracking first.** Qualification
and rewrite use `--block qualification --block-limit 100 --ceiling 560`;
teacher and scientific use `--block scientific --block-limit 560 --ceiling 560`.
The generator enforces this order, including allocation-boundary resume.
Each allocation is one exclusive eight-GPU node. Reservations charge the full
StartTime+TimeLimit, including idle, CPU commands and selftests. Refused jobs are
charged and frozen time advice cannot extend them. Slurm assigns row GPUs and
containment verifies disjoint UUIDs. Only one-GPU rows can overlap under the
containment API; this campaign runs rows serially. Teacher and rewrite tasks each
occupy all eight GPUs and run sequentially. Teacher generation uses four TP2
replicas in the separate inference environment; rewrite uses eight TP1 replicas.
Scoring stays on the designated physical GPU and environment from qualification.
A mismatch refuses **before** generation. Every row sets `V4_TELEMETRY=1`.

Generate each sbatch **on the login node after prior allocations are terminal and
reconciled**. The output and advice JSON must be inside WORK so collection archives
them. The planner prints the exact numeric `sbatch --no-requeue --account=PARTNER_ACCOUNT --qos=PARTNER_QOS --time=HH:MM:00 FILE`: submit
that command only. It rounds down the minimum of requested time, site wall limit,
remaining block and remaining ceiling. It archives the ledger hash. A two-day
request refuses. No requested duration is a measured runtime estimate.

```sh
python "$KIT/v4_allocation.py" --work "$WORK" --phase qualification --stage qualification \
  --partition "$PARTITION" --qos "$QOS" --account "$ACCOUNT" --activate "$V4_ACTIVATE" \
  --minutes 360 --site-minutes "$SITE_MINUTES" --out "$WORK/report-allocation/qualification-1.sbatch"
# With no prior spend and a site limit >=360 minutes, exact time is --time=06:00:00.
# FIRST containment command in that script:
# python "$KIT/p4_contain.py" selftest --work "$WORK" --out "$WORK/k8b4/containment" \
#   --block qualification --block-limit 100 --ceiling 560
```

Qualification checks engine start and configuration **before** agreement arithmetic.
There are two independent same-task scoring pairs per task (four fresh reloads,
50 Chemistry and 50 FinQA questions at B=2048). A passing first pair followed by
any differing second pair blocks every training row. The archive reader regrades
all answers. Qualification resolves `--cfg job --resolve` for all four actual
launchers in the pinned trainer, checks registered constants before training,
measures full-cap waves and probes batch invariance under different batch
composition. The teacher explicitly requests FLASH_ATTN and batch invariance;
engine startup can take one labelled restart without invariance. That fallback
reports that serial equivalence and regeneration equivalence are unavailable.
The frozen corpus remains the experimental input. The teacher's 2560-token cap
is a registered conservative allowance, not a proven cross-tokenizer bound;
CPU preparation archives the empirical teacher/student tokenizer ratio on qualification pool text; both readers recompute its arithmetic and student counts. It is not a universal generated-response bound. Verification rejects student responses beyond B. Input truncation refuses;
`unfinished_generation` at a response cap is normal sample rejection.

Qualification also runs each arm for two updates, saves, kills the trainer,
resumes one update, and compares restored live model/optimizer/RNG/scheduler,
dataloader and EMA state hashes and next-step loss against an uninterrupted run.
**Saved-state restoration is DISABLED until this real-stack qualification passes.**
An interrupted slot with valid state stops for review until then. Incoming-checkpoint
restart is allowed only with no valid saved state and the registered typed
infrastructure retry conditions. Completed training is recovered by export and
scoring, never retrained. Failed attempts remain immutable. At most two retries
are allowed, including one OOM placement retry with `OFFLOAD=1`.

The departures from SDFT/SDPO references are frozen in `v4_departures.json`, cited
by SHA256 in every run summary and both readers. F/R supervise response tokens
including EOS; the last prompt logit predicts the first response token. Every
update archives finite loss and its used learning rate (first update zero, fixed
40-update dose). D archives teacher/student L2 and teacher change at every update;
qualification requires nonzero movement after update two. Memory, tokens and
export times are collected for qualification and main.

Admission uses full-width allocation cost, measured batched worst-case geometry,
a 1.5 allowance, four-attempt caps, in-allocation CPU commands, selftests and
45 GPU-hours contingency. CPU prepare freezes `allocation-profile.json` with
teacher/rewrite requests of 300 minutes and the scientific request of 720 minutes,
each clipped to `V4_SITE_MINUTES` (default 360). Qualification calibrates a fresh
Python process over the entire main runner graph; actual dispatch receipts also
measure command import/Slurm wrapper time outside operation bodies. Admission
prices graph preparation/import for every allocation and dispatch for every row.
The serial shard merge has its own raw-journal/pool-bound timing receipt, projected
for four attempts on all 1,441 questions, without a shard-concurrency discount.

Before **every** main sbatch the CPU planner re-reads the qualification archive,
checks its owner binding/caps, deducts only current PASS units, packs the serial
rows against their caps and recomputes selftest/startup counts. Every proposed
window is clipped again to the site limit and remaining 100/560 budgets. A window
that cannot fit a pending row refuses before submission. The JSON and generated
script bind this projection by `V4_ALLOCATION_PLAN_SHA256` to the prior closed accounting, owner receipt and
qualification archive. `V4_ALLOCATION_PLAN` is exported by the script; after the
FIRST explicit-block selftest, the runner verifies the actual Slurm duration,
width and block against it before tracking any row. Direct main runner entries
under Slurm require that same plan and its generated-script SHA256. The owner gate reuses the CPU planner's hash-bound qualification result instead of re-running the raw reader inside the GPU allocation. Qualification-derived row caps (minimum 300 seconds)
are registered **before** main in the owner receipt's `row_caps`, exactly matching
the reader's proposal. Main cannot substitute unmeasured defaults. Coverage
requires 1153/1441 per original pool plus the registered family bar. Failed coverage
blocks training; D-101 standalone acquisition/cost blocks subsequent sequences.
The report names failed primary criteria and cells. Identical probes reuse
immutable results by checkpoint/pool/decoder/tokenizer hash; reused tokens are
not charged as new generation. The 20 fixed pre-filter questions/task and eight
registered attempts at temperature .7/top-p .95 remain unchanged.

**The qualification job ends before the owner's pause.** PAUSE writes a return
request and the sbatch exits. Wait outside compute allocations for terminal
scheduler accounting, reconcile on the login node, then collect and read:

```sh
python "$KIT/p4_contain.py" reconcile --work "$WORK" --out "$WORK/reconcile-qualification" --seconds 60
python "$KIT/collect.py" --work "$WORK" --out send6-qualification.tar.gz
python "$KIT/v4_pull.py" --kind partner --archive send6-qualification.tar.gz
python "$KIT/v4_read.py" qualification send6-qualification.tar.gz
```

Manager registers actual model shards and a new immutable kit tag, then obtains
owner proceed. Set `V4_QUALIFICATION_ARCHIVE` to that verified archive and
`V4_OWNER_PROCEED` to its receipt (`owner_proceed: true`, `qualification_sha256`,
`new_kit_tag`, `row_caps`). The reader's recommendation is separate from approval.
Run `v4_prepare.py --work "$WORK" --phase main` on CPU before main allocations.
Main preserves one DAG across three stages; budget blocks cannot change in a job:

| Stage | FIRST selftest block/limit | Requested exact time before remainder clipping | Operations |
|---|---|---|---|
| teacher | scientific/560 | 05:00:00 | owner gate; extend approved teacher journals |
| rewrite | qualification/100 | 05:00:00 | extend rewrites; coverage/common set |
| scientific | scientific/560 | 12:00:00 | D-101 first; fixed 48 slots; scoring/probes/report |

For **each** allocation substitute one stage and its requested minutes (300,
300,720), use a new numbered output, and submit only the planner's printed exact
time. Clipped actual times are in its JSON; do not submit the table's request
against a smaller remainder. Script activation exports `PYTHONPATH` from KIT's
parent; direct runner invocation uses the same import root.

```sh
python "$KIT/v4_allocation.py" --work "$WORK" --phase main --stage "$STAGE" \
  --partition "$PARTITION" --qos "$QOS" --account "$ACCOUNT" --activate "$V4_ACTIVATE" \
  --minutes "$REQUESTED_MINUTES" --site-minutes "$SITE_MINUTES" \
  --out "$WORK/report-allocation/main-$STAGE-$ALLOCATION_NUMBER.sbatch"
```

Before an attempt is written, the runner checks its cap fits the current
allocation. If it cannot fit, the job ends with exit 75 without consuming an
attempt. After terminal accounting reconcile on the login node and submit a new
allocation for the same stage. Its first containment command is its own explicit
selftest. Passed rows are reused by hash; interrupted rows use reviewed retry
admission. Never shrink row caps or scientific dose to fit a remainder.

Three section-13 stop classes apply: (1) verified bounded row failures get no
`v4-stop`; typed evidence automatically admits registered retries; unqualified
failures stay failed; (2) CANCEL with verified termination ends the allocation,
with no manager file; (3) unknown exposure, cap breach, overlap, changed safety,
block or ceiling exhaustion writes `v4-stop.json`. A hard stop needs manager
acknowledgement bound to its exact SHA256. Manager uses `v4_ack.py make STOP.json`;
the campaign uses `verify`. Owner delivery establishes provenance. There are no
signatures, keys or cryptography dependencies. Reconcile alone cannot clear a
hard stop. Original stop and acknowledgement inodes/bytes remain in append-only
`acknowledged-stops`; history is never replaced or deleted.

After main terminal accounting, reconcile, collect and read `v4_read.py main`.
Every Modal/partner pull is manifest-hash-checked before use. For smoke/27B pulls,
use the returned remote archive SHA256 and path, never a copied directory:

```sh
python "$KIT/v4_pull.py" --kind four-arm-smoke --modal-volume "$VOLUME" \
  --remote "$REMOTE_ARCHIVE" --sha256 "$RETURNED_SHA256" --archive smoke-return.tar.gz
python scripts/pull_v4_27b_check.py --modal-volume "$VOLUME" \
  --remote "$REMOTE_ARCHIVE" --sha256 "$RETURNED_SHA256" --archive 27b-return.tar.gz
```

Readers recompute native scoring, provenance, criteria/cells, lexical diagnostics,
mechanism strata and costs from raw archives. Missing evidence stays incomplete,
preserving established failures. `kit-snapshot/kit/` supports fresh-source archive-only
reproduction; the complete synthetic tests explicitly use fictional model
registrations and do not claim GPU qualification. Orchestration dry runs replace
row bodies and are not G5 receipts. Follow the frozen check list for external G5/G6.

Every planner-generated sbatch includes `#SBATCH --no-requeue`. Supply the partner’s own `--account` and `--qos`; never reuse the CPU format capture’s midpri defaults. REQUEUE/preempt-qos admission requires live `Requeue=0` and a QOS absent from every other QOS Preempt list. All allocation wall time, including a pathological UnkillableStepTimeout tail of 500 seconds, remains charged; row caps and hard-stop review are unchanged.
