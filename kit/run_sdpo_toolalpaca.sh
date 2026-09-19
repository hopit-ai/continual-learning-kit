#!/usr/bin/env bash
# SDPO on ToolAlpaca (the reference's `datasets/tooluse`), Qwen3-8B, at the paper's geometry.
#
# This is the exact trainer command of our run 3 (docs/replication/run3-record.json, 12 Sep 2026:
# 57.9 -> 66.1 avg@16 in 17 steps on 4 x H200), with Modal removed and paths made parameters.
# Receipt 201 proves the arguments below equal that record, key for key, apart from paths.
#
# Nothing here belongs to our project: it runs lasgroup/SDPO at its pinned commit, unmodified.
#
# Required:  SDPO_DIR   checkout of https://github.com/lasgroup/SDPO at 7c457fc1b1f6...
#            MODEL_DIR  local snapshot of Qwen/Qwen3-8B at revision b968826d9c46...
#            NAME       a fresh name per run (never reuse one; outputs are never overwritten)
# Optional:  STEPS=17 TEST_FREQ=17 SEED= NGPU=4 (8 on 80 GB cards) TP=2 OFFLOAD=0 WORK=$PWD/sdpo-work DRY_RUN=0
set -euo pipefail

: "${SDPO_DIR:?set SDPO_DIR to the pinned lasgroup/SDPO checkout}"
: "${MODEL_DIR:?set MODEL_DIR to the Qwen3-8B snapshot directory}"
: "${NAME:?set NAME to a fresh run name}"
STEPS="${STEPS:-17}"
TEST_FREQ="${TEST_FREQ:-$STEPS}"
SEED="${SEED:-}"
NGPU="${NGPU:-4}"
TP="${TP:-2}"
OFFLOAD="${OFFLOAD:-0}"
WORK="${WORK:-$PWD/sdpo-work}"
DRY_RUN="${DRY_RUN:-0}"

OUT="$WORK/runs/$NAME"
if [[ "$OFFLOAD" == "1" ]]; then OFF=True; else OFF=False; fi

ARGV=(
  python -m verl.trainer.main_ppo --config-name sdpo
  "data.train_files=[$SDPO_DIR/datasets/tooluse/train.parquet]"
  "data.val_files=[$SDPO_DIR/datasets/tooluse/test.parquet]"
  "actor_rollout_ref.model.path=$MODEL_DIR"
  "actor_rollout_ref.actor.strategy=fsdp2"
  "actor_rollout_ref.actor.optim.lr=1e-5"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.rollout.n=8"
  "data.train_batch_size=32"
  "data.shuffle=True"
  "trainer.experiment_name=$NAME"
  "trainer.default_local_dir=$OUT/tool-sdpo"
  "trainer.save_freq=$STEPS"
  "trainer.max_actor_ckpt_to_keep=4"
  "trainer.total_epochs=1"
  "trainer.val_before_train=True"
  "actor_rollout_ref.actor.checkpoint.save_contents=[model]"
  "trainer.resume_mode=disable"
  "trainer.logger=[console,file]"
  "trainer.project_name=r99-reference-runtime"
  "trainer.group_name=rep-sdpo-toolalpaca"
  "vars.dir=$SDPO_DIR"
  "vars.task=datasets/tooluse"
  "vars.log_dir=$OUT/tool-sdpo/logs"
  "vars.ckpt_dir=$OUT/tool-sdpo"
  "custom_reward_function.path=$SDPO_DIR/verl/utils/reward_score/feedback/__init__.py"
  "custom_reward_function.name=compute_score"
  "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema"
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05"
  "actor_rollout_ref.actor.self_distillation.alpha=0.5"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
  "actor_rollout_ref.actor.self_distillation.distillation_add_tail=True"
  "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
  "trainer.total_training_steps=$STEPS"
  "actor_rollout_ref.rollout.val_kwargs.n=16"
  "trainer.test_freq=$TEST_FREQ"
  "actor_rollout_ref.actor.calculate_entropy=True"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.55"
  "actor_rollout_ref.actor.fsdp_config.param_offload=$OFF"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFF"
  "actor_rollout_ref.ref.fsdp_config.param_offload=$OFF"
  "trainer.rollout_data_dir=$OUT/rollouts"
  "trainer.validation_data_dir=$OUT/validation"
  "trainer.n_gpus_per_node=$NGPU"
  "trainer.nnodes=1"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP"
)
# A seed is three overrides. Run 3 set none of them (defaults: 42, 42, null), so an empty SEED
# reproduces run 3's command exactly. vLLM sampling is unseeded either way.
if [[ -n "$SEED" ]]; then
  ARGV+=(
    "data.seed=$SEED"
    "actor_rollout_ref.actor.data_loader_seed=$SEED"
    "actor_rollout_ref.actor.fsdp_config.seed=$SEED"
  )
fi

if [[ "$DRY_RUN" == "1" ]]; then printf '%s\n' "${ARGV[@]}"; exit 0; fi

if [[ -e "$OUT" ]]; then echo "refusing to overwrite $OUT: pick a fresh NAME" >&2; exit 2; fi
for f in train.parquet test.parquet; do
  [[ -f "$SDPO_DIR/datasets/tooluse/$f" ]] || { echo "missing $f: run data/preprocess.py first" >&2; exit 2; }
done
mkdir -p "$OUT/env" "$OUT/tool-sdpo/logs"

# What the run was: recorded before it starts, so a crash still leaves an identity behind.
printf '%s\n' "${ARGV[@]}" > "$OUT/env/argv.txt"
git -C "$SDPO_DIR" rev-parse HEAD > "$OUT/env/sdpo-commit.txt"
git -C "$SDPO_DIR" status --short > "$OUT/env/sdpo-dirty.txt"
pip freeze > "$OUT/env/pip-freeze.txt" 2>/dev/null || true
nvidia-smi > "$OUT/env/nvidia-smi.txt" 2>/dev/null || true
ls "$MODEL_DIR" > "$OUT/env/model-files.txt"
date -u +%FT%TZ > "$OUT/env/started-at.txt"

export USER="${USER:-$(whoami)}" TASK="datasets/tooluse" EXPERIMENT="$NAME"
export VLLM_USE_V1=1 WANDB_MODE=disabled
export PYTHONPATH="$SDPO_DIR:${PYTHONPATH:-}"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"

cd "$SDPO_DIR"
"${ARGV[@]}" 2>&1 | tee "$OUT/console.log"
date -u +%FT%TZ > "$OUT/env/finished-at.txt"

# actor save -> a HuggingFace model we can score for retention; the reference's own merger.
CKPT="$OUT/tool-sdpo/global_step_$STEPS/actor"
if [[ -d "$CKPT" ]]; then
  python -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$OUT/hf-step$STEPS" \
    2>&1 | tee "$OUT/merge.log"
fi
echo "done: $OUT"
