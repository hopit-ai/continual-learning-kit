# Message 3 tier: K (frozen-base soft replay) and CR (cumulative replay) — registration v2

Optional tier inside the combined message 2+3 send. It grants no launch authority and never
blocks message-2 core: its own root, manifest schema, task IDs, scoring assignment, nodes and
output tree. The partner submits each plain job with his own sbatch. Seeds 101-103 only.

## Arms (fresh chains from pinned Qwen3-8B: FinQA -> ContractNLI -> hard MATH)

| Arm | Rows/update by stage | Replay loss |
|---|---|---|
| C0 | 28 task + 4 instruction (positions 8/16/24/32) | token-mean CE |
| K | as C0 | forward KL(base‖student), full vocabulary, T=1 |
| G | 32 task + 8/16/24 instruction | CE |
| CR | 32 task + instruction 4, GSM8K 2, ARC 2 (+ FinQA 8) (+ ContractNLI 8) | CE |

* One shared instruction stream for every arm: message-2's seed-only generic cycling
  (`random.Random(seed)` shuffle) without length filtering, so the same 160 IDs recur each stage
  and the four IDs of every update match across arms.
* G's extra rows use the hash order `m3-cr-v2|seed|stage|generic-extra|id`, excluding the
  stage's shared IDs; other CR domains use `m3-cr-v2|seed|stage|<domain>|id`. No row repeats
  within a stage; cross-stage repeats are published in `schedules/audit.json`.
* No filtered draws: every arm and stage uses prompt 10,240, response 2,048 (incl. EOS),
  reprompt 14,336 and `trainer.balance_batch=False`. An overlength row fails preparation.
* Global objective `(Σ task d + Σ replay)/32`: each local row loss is scaled by 1/4 on eight
  ranks (1/16 on two). Batch/mini-batch 32 for C0/K, 40/48/56 for G/CR.

## Submission order

1. Submit **every message-2 core job first**. The tier never uses core nodes or GPUs: list them in
   `message2_nodes` (settings) and in the inventory (`message2_nodes`, `message2_assignment`).
2. `bash $KIT/jobs/message3/build.sh` (CPU): initialise the separate M3 root, build the manifest,
   print the isolation proof (pass `M2_MANIFEST` to check task and node disjointness).
3. `bash $KIT/jobs/message3/run-task.sh M3-init` records that core jobs are submitted. Every other
   task stays PENDING (exit 75) until this record exists.
4. `M3-bind-inputs` (CPU) binds the same B and C directories message 2 uses, read in place:
   B's `replay/` and FinQA target folders by manifest sha256 (as `m2_prepare` / `next_pools` bind
   them), C's ContractNLI journal also against `package-c-return.tar.gz`. Nothing is copied or vendored.
   Run `M3-verify-superset` (CPU) on the actual bound B/C files before any shared-pool freeze.
   `M3-hellaswag` freezes the tier's own HellaSwag-100; `M3-finqa-targets` preserves B's seed-matched scheduled COMMON `train.parquet`
   for each of seeds 101-103, checking its manifest `ids`, seed and hashes. These are already
   1,280-row exposure schedules: no merged task pool, sorting, de-duplication or second shuffle.
   C0/K use B's `mixed()` replacement at positions 8/16/24/32; G/CR retain all 32 native task
   positions. Earlier-task replay continues using core's unique union through `earlier_rows`.
   Missing seed schedules or obsolete M3 union/shuffle freezes fail preparation; use a fresh M3 root.
5. `M3-pools-shared` freezes instruction inputs independently; `M3-schedules` freezes C0/K's FinQA
   and ContractNLI schedules. Run `M3-bind-base`, `M3-assign`, then per seed `M3-base-<seed>`.
   Separately run `M3-contract` and `M3-finqa-replay` (CPU), then `M3-broad` (one GPU),
   `M3-pools` and `M3-schedules-coverage` (CPU). Broad preparation failures leave G/CR PENDING;
   C0/K continue. The FinQA replay materialization reverifies bound B targets and excludes panels
   before broad candidate selection; the broad job requires this materialized input.
6. Set `host_reservation_bytes` to the actual
   whole-node RAM reservation, then submit `M3-preflight-C0`, `M3-preflight-K`, `M3-preflight-G`
   and `M3-preflight-CR` on allowed idle whole nodes. Each produces its own contract-bound
   `preflight-inputs/<arm>/manifest.json` and node plan from the frozen stage-2 legal schedules.
   Four complete converted updates retain replay masks/positions and use the audited longest
   legal row as task input on every task position. Length comes from the hash-frozen schedule
   audit. No hard-MATH targets or earlier scientific training are needed for qualification.
   A passing `preflight/<arm>.json` gates only that arm; failed work is archived by the preflight
   runner. Timing is recorded, never gated. Do not weaken resume equality after a failed check.
7. Chains: `M3-<arm>-<seed>-<stage>-train` (8 GPUs, a node of its own) then
   `M3-<arm>-<seed>-<stage>-score` (the seed's assigned GPU). Stages are serial; independent
   arm/seed chains may run concurrently on separate M3 nodes. No training/scoring overlap on a node.
   **Stages 1-2 never wait for hard-MATH targets.**
8. When the first chain reaches stage 3, submit `M3-math-targets` then `M3-schedules-math`.
   If core's M5 has durably committed `targets-math` (the frozen 500-item Level-5 pool under the
   registered 3D bounded protocol), it is copied read-only with its snapshot hash; otherwise the
   identical protocol runs here on two GPUs (seeds 27101-27104, temperatures 0/.3/.6/.8, 2,560
   output / 4,096 input tokens, frozen verifier). These are target-generation attempt
   seeds, not training seeds; scientific training seeds remain 101–103. `targets-math-source.json` records which path ran.
   It never waits on core. C0/K stage-3 trains require `M3-schedules-math`; G/CR require
   `M3-schedules-coverage-math`. Shared and coverage schedules have independent immutable freezes.
9. After `M3-C0-101-finqa-train`: `M3-fv-<seed>` and `M3-kv-<seed>` validate the HellaSwag and
   knowledge likelihood scorers on the frozen base and C0's stage-1 final (seed 101), 100/100
   agreement with the independent reference on each seed's GPU. Until a seed passes, its
   likelihood scores are PENDING in the reader.
10. `M3-read` writes `m3-read.json` / `m3-read.md`; `m3_read.py --action sp-tiers` writes the bound
   SP checkpoint lists (planned lists ship as `kit/resources/m3/sp-tier-checkpoints-{K,CR}.json`).

### Our side, before send

* Commit/tag the tested kit before partner qualification.

* `m3_pools.py --out <scratch> --action superset --replay <B replay/> --finqa-targets ... --source <ContractNLI>`
  writes `kit/resources/m3/sp-candidate-superset-v1.json` (m3b2's format) from the pinned sources.
  Alternatively supply `--instruction-source <PINNED_RLVR_TRAIN_PARQUET>` and
  `--finqa-source <PINNED_FINQA_TRAIN_JSON>` for conservative source supersets. Both files
  are SHA256-checked against `candidate-sources.json`; broad snapshots must already be cached
  locally at the registered revisions and match the pinned parquet hashes. No download occurs.
  The shipped artifact contains all possible training prompts, including the 500-item hard-MATH
  pool. Runtime replay pools and task schedules must verify membership and frozen holdout
  exclusions; missing/mismatched evidence blocks affected training. PD/SP must bind this universe.

`run-task.sh` exits 0 (DONE), 75 (PENDING) or 1 (FAILED) and records state under `$M3_OUT/state/` only.
Every GPU job checks the executing hostname against allowed/core-held nodes and obtains a
node-local exclusive reservation spanning all commands. Hostnames are compared by the short
name before the first dot, for both socket and inherited platform inventories. Request a
**whole-node allocation**: the `/tmp` reservation coordinates M3 jobs only; it cannot reserve
resources against unrelated scheduler jobs sharing the node. The inherited lock survives launcher
termination while its child remains alive. A fresh launch also refuses nodes with active GPU work.
Direct scientific CLI execution enforces the same manifest and reservation.

Submission/start/end timestamps and every failed attempt are retained. In a scheduler job,
submission time is read from the scheduler; `M3_SUBMITTED_AT` may instead carry the epoch
recorded before submission. Direct runs have no measured scheduler queue and are marked accordingly.

## Pools (frozen before any chain trains)

| Domain | Source | Unique rows required |
|---|---|---:|
| Instruction | B's checked RLVR-IFeval `replay.jsonl` exactly, the directory message 2 binds (refresher-64 reserved) | 160 (C0/K); 960 (G) |
| FinQA task (stage 1) | B's verified 27B seed-specific COMMON scheduled parquet, native 40x32 order and repetitions; no resampling | one native schedule per seed 101-103 |
| Hard-MATH task (stage 3) | frozen 500-item Level-5 training pool, registered 3D bounded targets (core's committed copy or identical M3 run); scored on the frozen 100-item panel; never the 200-item screen | all accepted |
| GSM8K / ARC | message-2's frozen recipe in CR's own namespace, frozen strict ARC prompt | 80 each |
| FinQA | reverified B 27B training demonstrations (all 1,147 panel questions excluded) | 320 |
| ContractNLI | `contract-replay.jsonl` from the verified 27B training journal | 320 |

Exclusions: the three registered prompt rules only (normalised equality, word 5-shingle Jaccard
>= .8, edit similarity >= .9) against panels, frozen PD/SP/core holdouts and earlier pools.
Bare-ID collisions are not a rule. ContractNLI is not de-duplicated within itself (17 hypotheses
share each agreement); its per-agreement counts are in `pools/contract-document-counts.json`.
Short pools leave CR and G **PENDING**; C0 and K proceed.

## Scoring and reading

Scientific assignment requires at least three distinct physical GPUs (fixture sharing requires
explicit stand-in mode). One dedicated physical GPU per seed (`m3-scoring-assignment.json`, family `m3`) scores base,
every incoming checkpoint (aliases of the preceding final) and every final for all four arms.
Panels: FinQA-1,147, legal test 100, hard-MATH 100, general-v1 (instruction and maths), and two
format-free likelihood panels: HellaSwag-100 and knowledge (`kit/resources/m3/knowledge-likelihood.json`).
Each likelihood panel needs its own reference-agreement validation per seed before any
outcome. Generated-letter knowledge is diagnostic only.

`m3_read.py`: **K−C0** and **CR−G** are the decisions (mean >= 2, two-sided t=4.302653 lower
bound > 0, 3/3 positive, every guard's lower bound >= -1). CR−C0 and G−C0 are descriptive.
Gain-AUC, step-20 retention and acquired-answer survival are diagnostics. Package bars
(one-sided t=2.919986) are report-only. The read also reports each guard's power sensitivity
(required mean −1 + 2.48414 × SD), tokens per correct against the 1.5× ceiling, the repriced
budget (proxies until telemetry replaces them) and **flags any arm/seed or scoring queue over 24 h**.

Scientific retries preserve native state. The driver validates the latest complete eight-rank
model/optimizer/RNG/scheduler/dataloader/EMA checkpoint and binds its immutable digest before
resuming. Partial newer checkpoints stay in place when a complete checkpoint exists. Interruptions before
the first complete native checkpoint archive `sdft/`, rollouts, validation, start/exit records and
metrics with their byte hashes, then restart the identical stage. Training-cost attempts are retained.
A completed step-40 checkpoint with its receipt permits export recovery without training.
If the final receipt is missing, the state stays PENDING for a receipt-only native resume; a
trainer contract for that recovery remains a handoff requirement, and the driver never repeats updates.
Every scientific run manifest freezes `m3-receipt.json`; the reader opens and validates it against
arm, seed, stage and the bound base. Missing or invalid receipts cannot authorize admission.

Likelihood validation is recomputed on reading: assignment, checkpoint weights, item and policy
hashes, complete finite 100-by-4 production/reference results, and both checkpoints' full choice
agreement must match. Synthetic validation cannot authorize a scientific read. Scorers retain the
assignment's resolved physical GPU index and assert its UUID at inference startup.

Cost reports include every scoring attempt, both validation phases per checkpoint, the HellaSwag
likelihood pass, broad preparation attempts and M3-generated hard-MATH targets. Read-only core
target copies are not charged again. Failed attempts and queue delay contribute to the 24-hour flags.

Before send, the manager can independently verify actual returned inputs on CPU:

```sh
python -m kit.m3_pools --action verify-returns --out "$M3_OUT" \
  --finqa-targets "$B_FINQA_101" "$B_FINQA_102" "$B_FINQA_103" \
  --replay "$B_REPLAY" --targets "$C_TARGETS"
```

Required inputs: B's original seed-specific COMMON folders with manifest, `train.parquet` and
`val.parquet`; B's frozen replay directory with `replay.jsonl` and its manifest-listed files;
C's frozen target journal with all three `schedule-<seed>.jsonl` files and its manifest-listed
files; and the published `resources/m3/sp-candidate-superset-v1.json`. Files are read in place,
hashes verified and prompt membership checked for instruction, FinQA/task-FinQA and legal/task-legal.
The report is `returned-superset-verification.json`. Missing actual files are PENDING; any
missing prompt blocks send and affected training through the existing task/pool evidence gates.
No public-source superset can substitute for this actual-returned-file check.

Likelihood validations freeze only PASS or a complete finite reference-agreement FAIL. Technical
exceptions create attempt records and exit nonzero, allowing an identical retry. An agreement
FAIL remains immutable and cannot be rerun; old PENDING records are archived before retry.
Coverage schedules depend on shared schedules and audit all four arms for every seed, including
computed draw-index gaps per stream/domain.

Manager repricing API: `m3_read.padded_token_cost(updates, gpu_hours_per_million_tokens)` accepts
measured `m3/{actor,ema,base,rollout}_padded_tokens` and four independently calibrated phase
rates. It returns update-phase GPU-hours without changing registered budgets. Add startup,
exports, scoring and pool preparation separately; oracle work already counted in telemetry
must not be multiplied again.


The owner/manager's 1 November [Likelihood-scorer validation: near-tie ruling](resources/m2/likelihood-near-tie.md)
amends only the technical unanimous-choice validation gate. Both scorers' top-two
gaps must be strictly below 0.001 nats/token on every disagreement, otherwise
PENDING. Frozen character-normalised scores/choices stay unchanged; pinned CPU
length vectors supply the token-unit validation diagnostics. No automatic retry.
