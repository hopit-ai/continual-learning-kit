# Package B: plain sequence and recovery jobs

From the existing kit repository, migrate before running any job:

```bash
export CHECKOUT=/absolute/checkout
cd "$CHECKOUT" || exit $?
git fetch --tags && git checkout kit-p3b-v4
git describe --tags --exact-match
```

The verification must print `kit-p3b-v4`. Use Bash for the blocks below, in order;
loops execute jobs sequentially. Supply your own sbatch allocations.

**v4 changes (18 October, owner approved):** D+replay fixes prompt eligibility
and mixed task/replay reward schemas. Only D skips over-cap replay draws in
place; F+replay retains the full 1,328-row pool. F+replay now includes FinQA
seeds 104–105 and five standalone plus five matched sequence MATH stages.
All registered recipes, schedules, targets, scoring and bars remain unchanged.

Science: `resources/p3b/registration.md`, with the owner-approved 17/18 October
[amendments](resources/p3b/amendments-20261018.md).
One command is one job; supply your own sbatch. No campaign, runner, submission or automatic continuation. Keep one
absolute OUT on node-local NVMe and copy the complete OUT to shared-storage throughout
and before releasing the node. Training data and checkpoints are written beneath
OUT; there is no separate scratch argument.

```bash
export KIT="$CHECKOUT/kit"
export OUT=/absolute/node-local/package-B-OUT
export TRAINER_PY=/absolute/trainer-venv/bin/python
export INFERENCE_PY=/absolute/inference-venv/bin/python
export SDPO=/absolute/pinned-SDPO
export BASE=/absolute/Qwen3-8B-snapshot TEACHER=/absolute/Qwen3.6-27B-snapshot
export A=/absolute/package-A-OUT                     # the complete package-A OUT, not its archive
# The retained FinQA evidence tree on shared-storage: the directory holding v4/ and runs/.
export V4_WORK=/absolute/shared-storage/path/to/preserved-finqa-WORK
export V4_EXPORTS=/absolute/shared-storage/v4-finqa-keep/exports
export ARCHIVE=/absolute/p3b-return.tar.gz
export V4_FINQA_PANEL="$V4_WORK/v4/report-inputs/finqa-panel.jsonl"   # replay exclusions only
export FINQA="$V4_WORK/v4/datasets/datasets/v4_finqa"                  # scoring: holds test.parquet
```

Models live under V4_EXPORTS; the commands locate the unique directory ending
in `ARM-finqa-alone-sSEED/hf-stepSTEP` within that tree (including any snapshot
wrapper directories). Each one's receipt comes from its
original run, `$V4_WORK/runs/<ARM>-finqa-alone-s<SEED>/env/export-identity-step40.json`
(step 20: `...-step20.json`). Do not edit receipts after relocating a model.

Use SDPO `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`, base revision
`b968826d9c46dd6066d109eabc6255188de91218`, teacher revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. The trainer venv runs all training,
8B generation and scoring; the inference venv (vLLM 0.18) runs only the 27B.
Replay checking also needs `langdetect` (the pinned checker imports it), `rapidfuzz` (fast de-duplication),
pyarrow and huggingface_hub. The local reader needs numpy.

Run corrected D+replay FinQA seeds 101–105 and F+replay FinQA seeds 104–105
first. Prepare the common MATH targets in free-GPU gaps, then run tier 2 once
its inputs and tier 1 are complete. Retain completed F+replay 101–103 and
recovery outputs; finish any missing tier-1 recovery/scoring rows. Only overlap
with A on disjoint GPUs and with sufficient host memory/IO; every full training/recovery job uses all eight GPUs. Teacher
preparation uses two GPUs; replay generation and each scoring job use one.
**Every scoring, including re-scored base/incoming checkpoints, uses the same
one physical GPU and software environment.** Record durable paths and GPU IDs.

1. Preserve the five failed **v3** D+replay slots in the **same OUT** before
retrying. Run this block once, before any corrected D+replay run. It moves each
identity/data/run only if present and stops before moving anything if a prior
backup would be overwritten. Keep OUT/logs and all failed evidence intact.

```bash
mkdir -p "$OUT/failed-v3"
for SEED in 101 102 103 104 105; do
  SLOT="finqa-D+replay-s${SEED}"
  for ITEM in "identities/${SLOT}.json" "data/${SLOT}" "runs/${SLOT}"; do
    if [ -e "$OUT/failed-v3/$ITEM" ]; then
      printf 'STOP: backup already exists: %s\n' "$OUT/failed-v3/$ITEM"
      exit 2
    fi
  done
done
mkdir -p "$OUT/failed-v3/identities" "$OUT/failed-v3/data" "$OUT/failed-v3/runs"
for SEED in 101 102 103 104 105; do
  SLOT="finqa-D+replay-s${SEED}"
  if [ -e "$OUT/identities/${SLOT}.json" ]; then
    mv "$OUT/identities/${SLOT}.json" "$OUT/failed-v3/identities/${SLOT}.json" || exit $?
  fi
  if [ -e "$OUT/data/${SLOT}" ]; then
    mv "$OUT/data/${SLOT}" "$OUT/failed-v3/data/${SLOT}" || exit $?
  fi
  if [ -e "$OUT/runs/${SLOT}" ]; then
    mv "$OUT/runs/${SLOT}" "$OUT/failed-v3/runs/${SLOT}" || exit $?
  fi
done
```

Reuse the frozen replay pool, schedules and targets unchanged; leave completed
F+replay and recovery slots in place. Do not edit old receipts. The corrected
rows change identity, so cached v3 identities/parquet must be moved first.

2. Check/reuse replay on the partner's network-enabled compute node. Existing
completed replay skips; a fresh OUT prepares it here. A's frozen inputs must
exist, but its task decision need not be complete. `--finqa-panel` is v4's frozen
`v4/report-inputs/finqa-panel.jsonl` (all 1,147 native rows). No reference answers
are sent to generation. This records the pinned source parquet hash on first
use, exclusions, candidate IDs and a durable raw journal.

```bash
export REPLAY_GPU=5   # any free allocation-visible GPU, 5-7, while A runs
"$TRAINER_PY" "$KIT/p3b_replay_data.py" --out "$OUT" --package-a "$A" --finqa-panel "$V4_FINQA_PANEL" --model "$BASE" --gpu "$REPLAY_GPU" || exit $?
```

Wait for replay generation and all A GPU jobs to finish before any eight-GPU job.
`--parquet /cached/train.parquet` permits offline reuse of that pinned download.
First 64 accepted examples form `replay/refresher-64.jsonl`; first 16 are nested.
Remaining examples form replay. Fewer than 65 is exit 2; no additional pool.
Optional base refresher controls are **excluded by default**; decide to include
both sizes, seed 101, before observing any refreshed outcomes.

3. Retry D+replay FinQA from BASE, seeds 101–105. Each uses the original v4
seed-matched scheduled `train.parquet` and MANIFEST (1,280 verified targets).

```bash
for SEED in 101 102 103 104 105; do
  SCHEDULE="$V4_WORK/v4/report-main/scheduled/finqa-s${SEED}/train.parquet"
  MANIFEST="$V4_WORK/v4/report-main/scheduled/finqa-s${SEED}/manifest.json"
  "$TRAINER_PY" "$KIT/p3b_replay_train.py" --out "$OUT" --recipe D+replay --route finqa --seed "$SEED" --model "$BASE" --tokenizer "$BASE" --task-schedule "$SCHEDULE" --schedule-manifest "$MANIFEST" --trainer-python "$TRAINER_PY" --sdpo "$SDPO" --gpu 0,1,2,3,4,5,6,7 || exit $?
done
```

Then add F+replay FinQA seeds 104–105 with the unchanged recipe. Retain completed
101–103; report those original three and all five separately.

```bash
for SEED in 104 105; do
  SCHEDULE="$V4_WORK/v4/report-main/scheduled/finqa-s${SEED}/train.parquet"
  MANIFEST="$V4_WORK/v4/report-main/scheduled/finqa-s${SEED}/manifest.json"
  "$TRAINER_PY" "$KIT/p3b_replay_train.py" --out "$OUT" --recipe F+replay --route finqa --seed "$SEED" --model "$BASE" --tokenizer "$BASE" --task-schedule "$SCHEDULE" --schedule-manifest "$MANIFEST" --trainer-python "$TRAINER_PY" --sdpo "$SDPO" --gpu 0,1,2,3,4,5,6,7 || exit $?
done
```

Score all ten prevention finals, including retained F+replay 101–103. Every
scoring below uses the same physical GPU 0 and trainer environment.

```bash
for RECIPE in D+replay F+replay; do
  for SEED in 101 102 103 104 105; do
    MODEL="$OUT/runs/finqa-${RECIPE}-s${SEED}/hf-step40"
    RECEIPT="$OUT/runs/finqa-${RECIPE}-s${SEED}/export-40.json"
    "$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" --checkpoint-receipt "$RECEIPT" --arm "finqa-${RECIPE}" --seed "$SEED" --step 40 --panel-set finqa --panel-root "$FINQA" --gpu 0 || exit $?
    "$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" --checkpoint-receipt "$RECEIPT" --arm "finqa-${RECIPE}" --seed "$SEED" --step 40 --panel-set general-v1 --gpu 0 || exit $?
  done
done
```

Frozen v4 arguments are in `resources/p3b/v4-recipes.json`, with source notes.
These programs call `sdft_entry.py` / `sft_entry.py` directly and merge saved
models. Existing v4 jobs can still use `run_sdft.sh` / `run_sft_f.sh` unchanged
with their original MODEL_DIR/DATASET or TRAIN_FILE/VAL_FILE/DATA_MANIFEST/SEED.
Manual export recovery uses the same command as v4:
`$TRAINER_PY -m verl.model_merger merge --backend fsdp --local_dir CHECKPOINT --target_dir EXPORT`.
D checkpoints: `sdft/global_step_K/actor`; F: `train/global_step_K`.

v4 fixes D+replay's prompt eligibility and mixed reward schema. For
**D+replay only**, replay draws over the unchanged 2,048 student-prompt-token cap
are skipped in place in the full pool's original seed shuffle. The following
eligible draw fills the same replay slot; no truncation, cap increase or new
shuffle occurs. This applies to finqa, standalone, sequence and reset. F+replay
continues to use the full pool; recovery is unchanged. Task prompts still stop
if over D's cap.

The console logs eligibility and per-seed skipped draws. Each real D+replay run
writes `runs/NAME/replay-selection.json`: eligible count and IDs, all excluded
IDs with measured student-prompt token lengths, and skipped draws (`draw_index`
and `replay_slot` are zero-based, measured in the full cyclic shuffle and the
160 accepted replay slots respectively).
The identity, summary and final run manifest also carry this audit; the manifest
hashes the audit file. Technical stand-ins do not measure token eligibility.

In D+replay rewards, `p3b_replay=0/1` marks every task/replay row, and replay
returns the active task's exact diagnostic key set with neutral scores. FinQA's
finish gate still applies only as before; MATH/code retain their existing
scorers. Training reward/group summaries exclude replay, while rollout length
statistics include all generated work. Token telemetry separates task loss-mask
tokens (excluding D's first three positions) and frozen replay CE labels
(including EOS). Actor diagnostic keys are consistent even on replay-only GPU
shards; entropy and importance-weight diagnostics average task rows across ranks.
Endpoint bars and reset decisions still use held-out scores.

4. Finish tier-1 recovery and scoring. These loops safely skip completed jobs.
The twelve recovery outputs use the exact original same-seed D/F FinQA finals,
never another recovery copy. Relocation is allowed; every model-file hash must
match its original v4 step-40 receipt. All recovery jobs take four updates,
batch 16, fresh F AdamW, LR 1e-5 from update one.

```bash
for ARM in D F; do
  for SEED in 101 102 103; do
    MODEL="$(find "$V4_EXPORTS" -type d -path "*/${ARM}-finqa-alone-s${SEED}/hf-step40" -print)"
    [ -d "$MODEL" ] || { printf 'STOP: missing or ambiguous v4 final: %s s%s\n' "$ARM" "$SEED"; exit 2; }
    RECEIPT="$V4_WORK/runs/${ARM}-finqa-alone-s${SEED}/env/export-identity-step40.json"
    for SIZE in 16 64; do
      "$TRAINER_PY" "$KIT/p3b_recover.py" --out "$OUT" --arm "$ARM" --size "$SIZE" --seed "$SEED" --model "$MODEL" --incoming-receipt "$RECEIPT" --tokenizer "$BASE" --trainer-python "$TRAINER_PY" --sdpo "$SDPO" --gpu 0,1,2,3,4,5,6,7 || exit $?
    done
  done
done
```

Score BASE once (shared by prevention and recovery), all ten original v4 D/F
finals, then all twelve recovery outputs. Non-base scoring always supplies the
checkpoint's own receipt; BASE has no receipt and uses seed 0.

```bash
"$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$BASE" --arm base --seed 0 --step 0 --panel-set finqa --panel-root "$FINQA" --gpu 0 || exit $?
"$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$BASE" --arm base --seed 0 --step 0 --panel-set general-v1 --gpu 0 || exit $?
for ARM in D F; do
  for SEED in 101 102 103 104 105; do
    MODEL="$(find "$V4_EXPORTS" -type d -path "*/${ARM}-finqa-alone-s${SEED}/hf-step40" -print)"
    [ -d "$MODEL" ] || { printf 'STOP: missing or ambiguous v4 final: %s s%s\n' "$ARM" "$SEED"; exit 2; }
    RECEIPT="$V4_WORK/runs/${ARM}-finqa-alone-s${SEED}/env/export-identity-step40.json"
    "$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" --checkpoint-receipt "$RECEIPT" --arm "$ARM" --seed "$SEED" --step 40 --panel-set finqa --panel-root "$FINQA" --gpu 0 || exit $?
    "$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" --checkpoint-receipt "$RECEIPT" --arm "$ARM" --seed "$SEED" --step 40 --panel-set general-v1 --gpu 0 || exit $?
  done
  for SEED in 101 102 103; do
    for SIZE in 16 64; do
      MODEL="$OUT/runs/recover-${ARM}-n${SIZE}-s${SEED}/hf-step4"
      RECEIPT="$OUT/runs/recover-${ARM}-n${SIZE}-s${SEED}/export-4.json"
      "$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" --checkpoint-receipt "$RECEIPT" --arm "recover-${ARM}-n${SIZE}" --seed "$SEED" --step 4 --panel-set finqa --panel-root "$FINQA" --gpu 0 || exit $?
      "$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" --checkpoint-receipt "$RECEIPT" --arm "recover-${ARM}-n${SIZE}" --seed "$SEED" --step 4 --panel-set general-v1 --gpu 0 || exit $?
    done
  done
done
```

5. Prepare tier-2 common targets. Tier 2 waits for A's complete ordered PASS
selection; neither qualifying means SKIP. Pending means exit 3. Extend full-pool targets, reusing A's raw requests:

```bash
"$INFERENCE_PY" "$KIT/p3b_targets.py" --out "$OUT" --package-a "$A" --model "$TEACHER" --tokenizer "$BASE" --gpu 0,1 || exit $?
```

6. Train all 40 tier-2 MATH stages: four recipes × two routes × five seeds.
Every recipe uses the same frozen `targets/schedule-SEED.jsonl` and manifest.
Standalone starts at BASE with no incoming receipt. Sequence D/F starts at the
original matched v4 FinQA final with its original step-40 receipt; sequence
D+replay/F+replay starts at its own matched tier-1 prevention final with
`export-40.json`. Never use recovery exports. All five finals of each recipe
proceed regardless of tier-1 scores.

```bash
for RECIPE in D F D+replay F+replay; do
  for SEED in 101 102 103 104 105; do
    SCHEDULE="$OUT/targets/schedule-${SEED}.jsonl"
    "$TRAINER_PY" "$KIT/p3b_replay_train.py" --out "$OUT" --recipe "$RECIPE" --route standalone --seed "$SEED" --model "$BASE" --tokenizer "$BASE" --task-schedule "$SCHEDULE" --schedule-manifest "$OUT/targets/manifest.json" --trainer-python "$TRAINER_PY" --sdpo "$SDPO" --gpu 0,1,2,3,4,5,6,7 || exit $?
    case "$RECIPE" in
      D|F)
        MODEL="$(find "$V4_EXPORTS" -type d -path "*/${RECIPE}-finqa-alone-s${SEED}/hf-step40" -print)"
        [ -d "$MODEL" ] || { printf 'STOP: missing or ambiguous v4 final: %s s%s\n' "$RECIPE" "$SEED"; exit 2; }
        RECEIPT="$V4_WORK/runs/${RECIPE}-finqa-alone-s${SEED}/env/export-identity-step40.json"
        ;;
      D+replay|F+replay)
        MODEL="$OUT/runs/finqa-${RECIPE}-s${SEED}/hf-step40"
        RECEIPT="$OUT/runs/finqa-${RECIPE}-s${SEED}/export-40.json"
        ;;
    esac
    "$TRAINER_PY" "$KIT/p3b_replay_train.py" --out "$OUT" --recipe "$RECIPE" --route sequence --seed "$SEED" --model "$MODEL" --incoming-receipt "$RECEIPT" --tokenizer "$BASE" --task-schedule "$SCHEDULE" --schedule-manifest "$OUT/targets/manifest.json" --trainer-python "$TRAINER_PY" --sdpo "$SDPO" --gpu 0,1,2,3,4,5,6,7 || exit $?
  done
done
```

7. Score MATH at 0/10/20/30/40 for every tier-2 run, and BASE once for the
serving-token denominator. Score FinQA and general-v1 at 0/20/40. At standalone
step 0, MODEL is BASE and RECEIPT_ARGS is empty. At sequence step 0, MODEL is
the matched FinQA final and RECEIPT_ARGS contains its **step-40 receipt** even
though `--step` is 0. At later steps, use that tier-2 run's own export/receipt.
The Bash array prevents a missing receipt from becoming an empty CLI argument.

```bash
"$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$BASE" --arm base --seed 0 --step 0 --panel-set math_screen --package-a "$A" --gpu 0 || exit $?
for RECIPE in D F D+replay F+replay; do
  for ROUTE in standalone sequence; do
    for SEED in 101 102 103 104 105; do
      for STEP in 0 10 20 30 40; do
        RECEIPT_ARGS=()
        if [ "$STEP" -gt 0 ]; then
          MODEL="$OUT/runs/${ROUTE}-${RECIPE}-s${SEED}/hf-step${STEP}"
          RECEIPT_ARGS=(--checkpoint-receipt "$OUT/runs/${ROUTE}-${RECIPE}-s${SEED}/export-${STEP}.json")
        elif [ "$ROUTE" = standalone ]; then
          MODEL="$BASE"
        else
          case "$RECIPE" in
            D|F)
              MODEL="$(find "$V4_EXPORTS" -type d -path "*/${RECIPE}-finqa-alone-s${SEED}/hf-step40" -print)"
              [ -d "$MODEL" ] || { printf 'STOP: missing or ambiguous v4 final: %s s%s\n' "$RECIPE" "$SEED"; exit 2; }
              RECEIPT_ARGS=(--checkpoint-receipt "$V4_WORK/runs/${RECIPE}-finqa-alone-s${SEED}/env/export-identity-step40.json")
              ;;
            D+replay|F+replay)
              MODEL="$OUT/runs/finqa-${RECIPE}-s${SEED}/hf-step40"
              RECEIPT_ARGS=(--checkpoint-receipt "$OUT/runs/finqa-${RECIPE}-s${SEED}/export-40.json")
              ;;
          esac
        fi
        "$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" "${RECEIPT_ARGS[@]}" --arm "${ROUTE}-${RECIPE}" --seed "$SEED" --step "$STEP" --panel-set math_screen --package-a "$A" --gpu 0 || exit $?
        case "$STEP" in
          0|20|40)
            "$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" "${RECEIPT_ARGS[@]}" --arm "${ROUTE}-${RECIPE}" --seed "$SEED" --step "$STEP" --panel-set finqa --panel-root "$FINQA" --gpu 0 || exit $?
            "$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" "${RECEIPT_ARGS[@]}" --arm "${ROUTE}-${RECIPE}" --seed "$SEED" --step "$STEP" --panel-set general-v1 --gpu 0 || exit $?
            ;;
        esac
      done
    done
  done
done
```

Report gain-AUC, all five curve points, boundary/step-20 panels and survival of
FinQA answers acquired in stage 1 with per-seed denominators. Step 40 is primary;
curves never select a winner.

Our-side `p3b_read.py --source OUT_OR_CSV_OR_ARCHIVE --out READING` writes every
bar, per-seed/question interval, family/churn table, curve and reset trigger.
After its written trigger only, run `p3b_reset.py --out "$OUT" --trigger TRIGGER
--recipe RECIPE --seed SEED --model ORIGINAL_FINQA_FINAL --incoming-receipt RECEIPT
--base "$BASE"` for all five seeds. Use priority D+replay, D, F, F+replay
(last), at most one five-seed branch.
Export is `resets/RECIPE-sSEED/hf-reset`, receipt `export.json`. Score immediately
post-reset as `reset-RECIPE`, step 0, against `sequence-RECIPE`, step 0 pre-reset.
Then train `--route reset` with reset export/receipt and the identical schedule;
score 10/20/30/40 plus FinQA/general at 0/20/40. No trigger means no reset.

Stage counts: 12 recovery runs, 10 FinQA prevention stages and 40 tier-2 stages;
at most 5 conditional reset stages, plus 2 optional base recovery controls.
Preparation and scoring are additional work. Timing: about 20 min per F stage
and 6 min per recovery run on the partner's node; please report actual D and
MATH timings. Report actual GPU timings separately for preparation, training,
scoring and failed/restarted attempts. The owner ceiling remains 1,000 GPU-hours;
flag revised estimates before exceeding it and reduce no comparison-bearing
setting. No budget gate is implemented.

Rerun the identical command/OUT: completed inputs, jobs, scores and journals
skip. Identity changes stop. A training stage that crashed is restarted from
step 0 with the same seed; its partial state is kept under `partials/` (as for
an interrupted FinQA slot). Note every restart in your return note.
Logs append to OUT/logs. Exit 0 done/SKIP, 2 correctness, 3 NOT READY; other
failures propagate a traceback. Never run duplicate copies of the same job.

Do not change the science: panels, prompts, schedules, targets, decoding,
verifiers, seeds, attempts, objectives, caps or decision rules. If a scientific
change is unavoidable, preserve original evidence and flag the exact departure
for the owner; it is outside this registered run.

```bash
"$TRAINER_PY" "$KIT/p3b_collect.py" --out "$OUT" --archive "$ARCHIVE" || exit $?
```

Send the archive plus `OUT/return/scores.csv.gz`, `metrics.csv.gz`, source diff,
job IDs and a short note on fixes, provenance, missing work and costs. No weights,
caches, private tests or rollouts return. Large evidence omissions are hash-listed;
archive is capped at 28 MiB. Keep all full raw evidence/checkpoints durably.

## Step-20 scoring of the v4 exports (separate OUT)

Score the 23 step-20 exports one after another on one fixed GPU (trainer venv),
FinQA and general-v1 at 2,048 only, with each export's own step-20 receipt.
Do this when that GPU is not needed by A or an eight-GPU job.

```bash
export OUT20=/absolute/node-local/step20-results
export ARCHIVE20=/absolute/step20-return.tar.gz
for ARM in D F R S D-R; do
  SEEDS="101 102 103 104 105"
  if [ "$ARM" = D-R ]; then SEEDS="101 102 103"; fi
  for SEED in $SEEDS; do
    MODEL20="$(find "$V4_EXPORTS" -type d -path "*/${ARM}-finqa-alone-s${SEED}/hf-step20" -print)"
    [ -d "$MODEL20" ] || { printf 'STOP: missing or ambiguous v4 step 20: %s s%s\n' "$ARM" "$SEED"; exit 2; }
    RECEIPT20="$V4_WORK/runs/${ARM}-finqa-alone-s${SEED}/env/export-identity-step20.json"
    for PANEL in finqa general-v1; do
      "$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT20" --model "$MODEL20" --checkpoint-receipt "$RECEIPT20" --arm "$ARM" --seed "$SEED" --step 20 --panel-set "$PANEL" --panel-root "$FINQA" --gpu 0 || exit $?
    done
  done
done
"$TRAINER_PY" "$KIT/p3b_collect.py" --out "$OUT20" --archive "$ARCHIVE20" || exit $?
```

This covers all 23 exports: D/F/R/S seeds 101–105 and D-R seeds 101–103.
`OUT20/return/scores.csv.gz` contains the scores; its metrics export is empty
because no training runs in OUT20.

## Next partner message

The v0+, interpolation-screen and replay-breadth jobs are in [README-next.md](README-next.md). Existing Package B commands and frozen inputs remain unchanged.
