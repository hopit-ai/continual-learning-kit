#!/usr/bin/env bash
# Supervised fine-tuning on a bed's exported (prompt, response) parquet, on the pinned lasgroup/SDPO fork's
# own SFT trainer, matched to kit/run_grpo.sh's dose. Package K2b's `sft` route; kit/run_grpo.sh is its `rl`.
#
# The two routes share one stack, one model, the same 1,280 GSM8K questions, 40 optimiser steps of 32
# examples, lr 1e-5, full parameters, fp32 master weights with bf16 compute, AdamW (0.9, 0.999), weight
# decay 0.01, gradient clip 1.0, and the same learning-rate shape: a linear warm-up over 10 steps, then
# constant. Only the loss differs. Where the SFT trainer's defaults differ from the GRPO actor's, this
# script overrides them, and says so beside each key:
#   - betas: sft_trainer.yaml sets (0.9, 0.95); the GRPO actor uses optim/fsdp.yaml's (0.9, 0.999).
#   - schedule: the SFT trainer has only `cosine` and `wsd`, and sizes the warm-up as a RATIO of the steps it
#     plans for (steps per pass x trainer.total_epochs), never of trainer.total_training_steps. So it is
#     planned for TWO passes (80 steps) with `wsd` (warm-up, then constant for 90 percent of the rest, then
#     decay) and a ratio of 0.125 = 10 warm-up steps, and stopped at STEPS=40 by trainer.total_training_steps:
#     steps 1-10 warm up and 11-40 are constant, the GRPO actor's schedule exactly. The decay would start at
#     step 74, which is never reached, and the second pass is never read. That holds only for a pass of 40
#     steps, so the run refuses a training file that is not 40 x 32 rows.
#   - the chat template: the SFT trainer does not read user.yaml, so thinking is turned off here, as
#     user.yaml turns it off for GRPO and kit/eval_bed.py for every scoring: all three see one prompt text.
#
# Required:  SDPO_DIR    checkout of https://github.com/lasgroup/SDPO at 7c457fc1b1f6...
#            MODEL_DIR   local model directory to start from
#            NAME        a fresh name per run (never reuse one; outputs are never overwritten)
#            TRAIN_FILE  absolute .parquet with `prompt` and `response` columns (kit/beds/gsm8k.py sft-export)
#            VAL_FILE    absolute .parquet of the same shape; the trainer computes one loss on it at the end
# Optional:  STEPS=40 SAVE_FREQ=$STEPS SEED= NGPU=8 LR=1e-5 WORK=$PWD/k3-work DRY_RUN=0 FILE_LOG=0
#
# SEED is a label, as it is for GRPO: it is passed as trainer.seed, which this trainer never reads. The data
# order is the DistributedSampler's fixed shuffle (its own seed 0), so it is the same in every repeat, and
# nothing else in supervised training is sampled: repeats differ only by GPU arithmetic.
#
# The trainer writes sharded FSDP checkpoints (model_world_size_<N>_rank_<r>.pt, fsdp_config.json and a
# huggingface/ folder with the config and tokenizer, but no weights) straight into global_step_<N>/, with no
# actor/ folder. They are merged into $OUT/hf-step$STEPS by the same merger run_grpo.sh uses, and
# train-summary.json carries run_grpo.sh's keys plus "route": "sft".
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SDPO_DIR:?set SDPO_DIR to the pinned lasgroup/SDPO checkout}"
: "${MODEL_DIR:?set MODEL_DIR to the model directory this stage starts from}"
: "${NAME:?set NAME to a fresh run name}"
: "${TRAIN_FILE:?set TRAIN_FILE to the SFT training parquet}"
: "${VAL_FILE:?set VAL_FILE to the SFT validation parquet}"
STEPS="${STEPS:-40}"
SAVE_FREQ="${SAVE_FREQ:-$STEPS}"
SEED="${SEED:-}"
NGPU="${NGPU:-8}"
WORK="${WORK:-$PWD/k3-work}"
DRY_RUN="${DRY_RUN:-0}"
FILE_LOG="${FILE_LOG:-0}"
LR="${LR:-1e-5}"
BATCH=32
PASS_STEPS=40              # the pass the schedule below is planned for: 1,280 rows at 32
EPOCHS=2                   # planned, never run: STEPS stops the trainer inside the first pass
WARMUP_RATIO=0.125         # x (PASS_STEPS x EPOCHS) = 10 warm-up steps, the GRPO actor's lr_warmup_steps

[[ "$FILE_LOG" == "0" || "$FILE_LOG" == "1" ]] || { echo "FILE_LOG must be 0 or 1, not $FILE_LOG" >&2; exit 2; }
[[ "$LR" =~ ^[0-9]+(\.[0-9]+)?(e-?[0-9]+)?$ ]] || { echo "LR must be a number like 1e-5, not $LR" >&2; exit 2; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ && "$SAVE_FREQ" =~ ^[1-9][0-9]*$ ]] || { echo "STEPS and SAVE_FREQ must be positive integers" >&2; exit 2; }
(( STEPS <= PASS_STEPS )) || { echo "STEPS=$STEPS is more than one pass ($PASS_STEPS): the schedule is planned for one" >&2; exit 2; }
[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] && (( BATCH % NGPU == 0 )) || { echo "NGPU=$NGPU must divide the batch of $BATCH" >&2; exit 2; }
if [[ "$FILE_LOG" == "1" ]]; then LOGGER="[console,file]"; else LOGGER="[console]"; fi

OUT="$WORK/runs/$NAME"

ARGV=(
  python -m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=$NGPU"
  "$KIT/sft_entry.py" --config-name sft_trainer
  "data.train_files=[$TRAIN_FILE]"
  "data.val_files=[$VAL_FILE]"
  "data.prompt_key=prompt"
  "data.response_key=response"
  # 2,048 tokens holds any GSM8K question and worked solution with room to spare; `error` (the default) stops
  # the run on a longer one rather than training on a cut answer.
  "data.max_length=2048"
  "data.truncation=error"
  "data.train_batch_size=$BATCH"
  "data.micro_batch_size_per_gpu=$(( BATCH / NGPU < 4 ? BATCH / NGPU : 4 ))"
  # sft_trainer.yaml has `apply_chat_template_kwargs: {}`, so the key is new and takes `+`.
  "+data.apply_chat_template_kwargs.enable_thinking=False"
  "model.partial_pretrain=$MODEL_DIR"
  "model.strategy=fsdp2"
  "model.lora_rank=0"
  "optim.lr=$LR"
  "optim.betas=[0.9,0.999]"
  "optim.weight_decay=0.01"
  "optim.clip_grad=1.0"
  "optim.lr_scheduler=wsd"
  "optim.lr_warmup_steps_ratio=$WARMUP_RATIO"
  "trainer.total_epochs=$EPOCHS"
  "trainer.total_training_steps=$STEPS"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.test_freq=-1"
  "trainer.checkpoint.save_contents=[model]"
  "trainer.resume_mode=disable"
  "trainer.default_local_dir=$OUT/train"
  "trainer.project_name=k2b-sft"
  "trainer.experiment_name=$NAME"
  "trainer.logger=$LOGGER"
  "trainer.nnodes=1"
  "trainer.n_gpus_per_node=$NGPU"
)
if [[ -n "$SEED" ]]; then ARGV+=("trainer.seed=$SEED"); fi

if [[ "$DRY_RUN" == "1" ]]; then printf '%s\n' "${ARGV[@]}"; exit 0; fi

if [[ -e "$OUT" ]]; then echo "refusing to overwrite $OUT: pick a fresh NAME" >&2; exit 2; fi
for f in "$TRAIN_FILE" "$VAL_FILE"; do
  [[ -f "$f" ]] || { echo "missing data file $f: run kit/beds/gsm8k.py sft-export first" >&2; exit 2; }
done
[[ -f "$MODEL_DIR/config.json" ]] || { echo "MODEL_DIR=$MODEL_DIR is not a HuggingFace model directory" >&2; exit 2; }
ROWS=$(python -c 'import sys, pyarrow.parquet as pq; t = pq.read_schema(sys.argv[1]); assert {"prompt", "response"} <= set(t.names), t.names; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' "$TRAIN_FILE") || {
  echo "$TRAIN_FILE is not a parquet file with prompt and response columns" >&2; exit 2; }
(( ROWS == PASS_STEPS * BATCH )) || {
  echo "$TRAIN_FILE has $ROWS rows; the schedule is planned for a pass of $PASS_STEPS x $BATCH = $(( PASS_STEPS * BATCH ))" >&2; exit 2; }
mkdir -p "$OUT/env" "$OUT/train"

# What the run was: recorded and flushed before it starts, so a crash still leaves an identity behind.
printf '%s\n' "${ARGV[@]}" > "$OUT/env/argv.txt"
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

export USER="${USER:-$(whoami)}" EXPERIMENT="$NAME"
export WANDB_MODE=disabled
export PYTHONPATH="$SDPO_DIR:${PYTHONPATH:-}"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"

STARTED=$(date -u +%s)
cd "$SDPO_DIR"
set +e
"${ARGV[@]}" 2>&1 | tee "$OUT/console.log"
STATUS=${PIPESTATUS[0]}
set -e
date -u +%FT%TZ > "$OUT/env/finished-at.txt"

# sharded save -> a HuggingFace model the beds and the forgetting panel can be scored on, through the same
# merger as run_grpo.sh. The SFT trainer saves into global_step_<N>/ itself; GRPO's actor saves into .../actor.
MERGED=0
CKPT="$OUT/train/global_step_$STEPS"
if [[ "$STATUS" == "0" && -d "$CKPT" ]]; then
  set +e
  python -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$OUT/hf-step$STEPS" \
    2>&1 | tee "$OUT/merge.log"
  set -e
  if [[ -f "$OUT/hf-step$STEPS/config.json" ]]; then
    MERGED=1
  else
    echo "the merge left no model at $OUT/hf-step$STEPS: see $OUT/merge.log" >&2
  fi
fi

# run_grpo.sh's keys, so one campaign bar reads either route, plus the route. There is no KL, entropy bonus
# or length budget in supervised training: they are written as the values that mean "off".
cat > "$OUT/train-summary.json" <<JSON
{
 "schema": "kit-sft-run.v1",
 "route": "sft",
 "name": "$NAME",
 "steps": $STEPS,
 "save_freq": $SAVE_FREQ,
 "returncode": $STATUS,
 "merged": $MERGED,
 "kl": 0,
 "lr": "$LR",
 "entropy_coeff": "0",
 "kl_coef": "0",
 "length_budget_chars": null,
 "n_gpus": $NGPU,
 "seconds": $(( $(date -u +%s) - STARTED )),
 "model_dir": "$MODEL_DIR",
 "train_file": "$TRAIN_FILE",
 "merged_dir": "$OUT/hf-step$STEPS"
}
JSON
echo "done: $OUT (returncode $STATUS, merged $MERGED)"
exit "$STATUS"
