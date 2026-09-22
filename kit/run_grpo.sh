#!/usr/bin/env bash
# GRPO on a bed's prepared parquet files, at phase 1's SQL dose, with the kit's own reward function.
#
# This is phase 1's composed GRPO command -- continual/reference_runtime.py::reference_command
# ('sql_grpo', ...), the `baseline_grpo` config of the pinned lasgroup/SDPO fork -- with the
# filesystem paths, the run's name, the data files and the reward function made parameters, and with
# the hardware knobs the reference command leaves to its caller appended at the end.
# tests/test_kit_run_grpo.py runs that function and compares it with this script's DRY_RUN argv, key
# for key and value for value, and lists every difference it had to allow.
#
# Nothing here belongs to our project: it runs lasgroup/SDPO at its pinned commit, unmodified. The
# reward function is kit/beds/rewards.py, which dispatches on each row's `data_source` to the bed
# that owns it, so ONE run can train on SQL questions and maths questions at once.
#
# Required:  SDPO_DIR    checkout of https://github.com/lasgroup/SDPO at 7c457fc1b1f6...
#            MODEL_DIR   local model directory to start from (a bed's stage-A output is one)
#            NAME        a fresh name per run (never reuse one; outputs are never overwritten)
#            TRAIN_FILE  absolute .parquet of trainer rows (a bed's prepare, or kit/mix.py)
#            VAL_FILE    absolute .parquet the reader can load; validation itself is off by default
# Optional:  STEPS=60 SAVE_FREQ=$STEPS TEST_FREQ=-1 SEED= NGPU=8 TP=2 OFFLOAD=0 KL=0 KL_COEF=0.001
#            TASK=datasets/spider_sql WORK=$PWD/k3-work DRY_RUN=0 FILE_LOG=0
#
# FILE_LOG=1 adds `file` to trainer.logger, so the per-step metrics land in $OUT/metrics.jsonl (the
# path this script already exports as VERL_FILE_LOGGER_PATH) and a campaign can put a bar on a
# per-step number -- K1c gates its 1.7B rows on a response-length floor, which K3 had no file to read.
# It changes ONE argv value and only when set; with it unset the command below is byte for byte what
# K3 ran, which tests/test_kit_run_grpo.py and tests/test_kit_k1c.py both pin.
#
# SPIDER_ROOT, if set, is exported so the reward workers can reach the Spider databases: ray starts
# its workers from this process, so they inherit this environment. If you run against an EXTERNAL
# ray cluster, export SPIDER_ROOT where that cluster was started -- a reward worker that cannot find
# the databases raises and stops the run, which is the intended failure (it never scores zero).
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SDPO_DIR:?set SDPO_DIR to the pinned lasgroup/SDPO checkout}"
: "${MODEL_DIR:?set MODEL_DIR to the model directory this stage starts from}"
: "${NAME:?set NAME to a fresh run name}"
: "${TRAIN_FILE:?set TRAIN_FILE to the training parquet}"
: "${VAL_FILE:?set VAL_FILE to the validation parquet}"
STEPS="${STEPS:-60}"
SAVE_FREQ="${SAVE_FREQ:-$STEPS}"
TEST_FREQ="${TEST_FREQ:--1}"
SEED="${SEED:-}"
NGPU="${NGPU:-8}"
TP="${TP:-2}"
OFFLOAD="${OFFLOAD:-0}"
KL="${KL:-0}"
KL_COEF="${KL_COEF:-0.001}"
TASK="${TASK:-datasets/spider_sql}"
WORK="${WORK:-$PWD/k3-work}"
DRY_RUN="${DRY_RUN:-0}"
FILE_LOG="${FILE_LOG:-0}"
REWARD="${REWARD:-$KIT/beds/rewards.py}"

[[ "$FILE_LOG" == "0" || "$FILE_LOG" == "1" ]] || { echo "FILE_LOG must be 0 or 1, not $FILE_LOG" >&2; exit 2; }
if [[ "$FILE_LOG" == "1" ]]; then LOGGER="[console,file]"; else LOGGER="[console]"; fi

OUT="$WORK/runs/$NAME"
if [[ "$OFFLOAD" == "1" ]]; then OFF=True; else OFF=False; fi
# Keep every checkpoint this run writes: upstream keeps 1, which would discard an earlier boundary.
# At phase 1's dose (60 steps, saved every 20) this is 3, which is what the reference command asks for.
KEEP=$(( STEPS / SAVE_FREQ )); (( KEEP > 0 )) || KEEP=1

# The order below is the reference command's order, so the two can be compared token by token.
ARGV=(
  python -m verl.trainer.main_ppo --config-name baseline_grpo
  "data.train_files=[$TRAIN_FILE]"
  "data.val_files=[$VAL_FILE]"
  "actor_rollout_ref.model.path=$MODEL_DIR"
  "actor_rollout_ref.actor.strategy=fsdp2"
  "actor_rollout_ref.actor.optim.lr=1e-5"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.rollout.n=8"
  "data.train_batch_size=32"
  # The reference's SQL arms train on a fixed ordered schedule. kit/mix.py has already shuffled the
  # mixture deterministically by its own seed, so the order is fixed AND recorded in its manifest.
  "data.shuffle=False"
  "trainer.experiment_name=$NAME"
  "trainer.default_local_dir=$OUT/train"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.max_actor_ckpt_to_keep=$KEEP"
  "trainer.total_epochs=1"
  # No in-trainer validation: the beds are scored afterwards by kit/eval_bed.py, deterministically,
  # on one GPU of one machine, which is the only comparison this programme trusts.
  "trainer.val_before_train=False"
  "actor_rollout_ref.actor.checkpoint.save_contents=[model]"
  "trainer.resume_mode=disable"
  "trainer.logger=$LOGGER"
  "trainer.project_name=r99-reference-runtime"
  "trainer.group_name=$NAME"
  "vars.dir=$SDPO_DIR"
  "vars.task=$TASK"
  "vars.log_dir=$OUT/train/logs"
  "vars.ckpt_dir=$OUT/train"
  "custom_reward_function.path=$REWARD"
  "custom_reward_function.name=compute_score"
  "trainer.total_training_steps=$STEPS"
  "trainer.test_freq=$TEST_FREQ"
)
# data.seed is passed because the reference command passes it, but with data.shuffle=False the trainer
# never reads it: the file's written order is the order. vLLM sampling is unseeded there and here, so
# runs differ in their rollouts only. Say so in any readout that compares seeds.
if [[ -n "$SEED" ]]; then ARGV+=("data.seed=$SEED"); fi

# Hardware, which the reference command leaves to its caller (it takes them as `extra`).
ARGV+=(
  "trainer.n_gpus_per_node=$NGPU"
  "trainer.nnodes=1"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP"
)
if [[ "$OFFLOAD" == "1" ]]; then
  ARGV+=(
    "actor_rollout_ref.actor.fsdp_config.param_offload=$OFF"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFF"
    "actor_rollout_ref.ref.fsdp_config.param_offload=$OFF"
  )
fi
# The `kl` arm: the trainer's own KL-to-reference loss, with the three keys the pinned config really
# has (verl/trainer/config/actor/actor.yaml:167,177,180; user.yaml:25 sets use_kl_loss False). Turning
# it on makes verl hold a frozen reference copy of the model as well as the actor: budget the memory.
if [[ "$KL" == "1" ]]; then
  ARGV+=(
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=$KL_COEF"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  )
fi

if [[ "$DRY_RUN" == "1" ]]; then printf '%s\n' "${ARGV[@]}"; exit 0; fi

if [[ -e "$OUT" ]]; then echo "refusing to overwrite $OUT: pick a fresh NAME" >&2; exit 2; fi
for f in "$TRAIN_FILE" "$VAL_FILE"; do
  [[ -f "$f" ]] || { echo "missing data file $f: run the bed's prepare (and kit/mix.py) first" >&2; exit 2; }
done
[[ -f "$MODEL_DIR/config.json" ]] || { echo "MODEL_DIR=$MODEL_DIR is not a HuggingFace model directory" >&2; exit 2; }
[[ -f "$REWARD" ]] || { echo "missing reward function $REWARD" >&2; exit 2; }
mkdir -p "$OUT/env" "$OUT/train/logs"

# What the run was: recorded and flushed before it starts, so a crash still leaves an identity behind.
printf '%s\n' "${ARGV[@]}" > "$OUT/env/argv.txt"
# The trainer's commit is part of that identity, so an SDPO_DIR that is not a checkout is a refusal
# rather than a blank file: a downloaded tarball cannot say which commit it is.
git -C "$SDPO_DIR" rev-parse --git-dir > /dev/null 2>&1 || {
  echo "cannot read the commit of $SDPO_DIR: clone it with git as README-partner.md section 2 says" >&2; exit 2; }
git -C "$SDPO_DIR" rev-parse HEAD > "$OUT/env/sdpo-commit.txt" 2>/dev/null || \
  echo "no commit on HEAD" > "$OUT/env/sdpo-commit.txt"
git -C "$SDPO_DIR" status --short > "$OUT/env/sdpo-dirty.txt" 2>/dev/null || true
pip freeze > "$OUT/env/pip-freeze.txt" 2>/dev/null || true
nvidia-smi > "$OUT/env/nvidia-smi.txt" 2>/dev/null || true
ls "$MODEL_DIR" > "$OUT/env/model-files.txt"
sha256sum "$TRAIN_FILE" "$VAL_FILE" > "$OUT/env/data-sha256.txt" 2>/dev/null || \
  shasum -a 256 "$TRAIN_FILE" "$VAL_FILE" > "$OUT/env/data-sha256.txt" 2>/dev/null || true
date -u +%FT%TZ > "$OUT/env/started-at.txt"

export USER="${USER:-$(whoami)}" TASK EXPERIMENT="$NAME"
export VLLM_USE_V1=1 WANDB_MODE=disabled
export PYTHONPATH="$SDPO_DIR:${PYTHONPATH:-}"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"
# Ray starts its workers from this process, so SPIDER_ROOT reaches the Spider reward function through
# this environment. Check it here: a worker that cannot open the databases raises deep inside a step,
# a quarter of an hour after the GPUs were taken, and this is the same failure a minute earlier.
if [[ -n "${SPIDER_ROOT:-}" ]]; then
  [[ -d "$SPIDER_ROOT" ]] || { echo "SPIDER_ROOT=$SPIDER_ROOT is not a directory: the Spider reward workers would fail" >&2; exit 2; }
  export SPIDER_ROOT
  echo "$SPIDER_ROOT" > "$OUT/env/spider-root.txt"
fi

STARTED=$(date -u +%s)
cd "$SDPO_DIR"
set +e
"${ARGV[@]}" 2>&1 | tee "$OUT/console.log"
STATUS=${PIPESTATUS[0]}
set -e
date -u +%FT%TZ > "$OUT/env/finished-at.txt"

# actor save -> a HuggingFace model the beds and the forgetting panel can be scored on, through the
# reference's own merger (continual/reference_runtime.py::conversion_command is this argv).
MERGED=0
CKPT="$OUT/train/global_step_$STEPS/actor"
if [[ "$STATUS" == "0" && -d "$CKPT" ]]; then
  set +e
  python -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$OUT/hf-step$STEPS" \
    2>&1 | tee "$OUT/merge.log"
  set -e
  # A merge that produced nothing must still reach the summary below: `merged: 0` is what the
  # campaign's bar reads, and a run that stopped here would leave no summary at all.
  if [[ -f "$OUT/hf-step$STEPS/config.json" ]]; then
    MERGED=1
  else
    echo "the merge left no model at $OUT/hf-step$STEPS: see $OUT/merge.log" >&2
  fi
fi

# One small JSON of numbers, so a campaign row can gate on what this run actually produced.
cat > "$OUT/train-summary.json" <<JSON
{
 "schema": "kit-grpo-run.v1",
 "name": "$NAME",
 "steps": $STEPS,
 "save_freq": $SAVE_FREQ,
 "returncode": $STATUS,
 "merged": $MERGED,
 "kl": $KL,
 "n_gpus": $NGPU,
 "seconds": $(( $(date -u +%s) - STARTED )),
 "model_dir": "$MODEL_DIR",
 "train_file": "$TRAIN_FILE",
 "merged_dir": "$OUT/hf-step$STEPS"
}
JSON
echo "done: $OUT (returncode $STATUS, merged $MERGED)"
exit "$STATUS"
