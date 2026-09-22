#!/usr/bin/env bash
# The authors' own GRPO on ToolAlpaca (the reference's `datasets/tooluse`), Qwen3-8B, at K0's geometry.
#
# This is `--config-name baseline_grpo` -- the SAME config the reference's own GRPO launchers use on
# the SAME data: run_local_grpo.sh:9-12 and experiments/generalization/run_baseline_grpo_all.sh:16-24
# both run `baseline_grpo` on `datasets/tooluse`, and their sweep
# (run_baseline_grpo_all.sh:39-46, 110-118) includes exactly this point: batch 32, 8 attempts,
# warm-up 10, val_kwargs.n=16, lr 1e-5, ppo_mini_batch_size 32, Qwen3-8B. It is NOT SDPO with pieces
# switched off: the seven `self_distillation.*` overrides K0 passes are dead under
# policy_loss.loss_mode=vanilla (actor/actor.yaml:61) and are therefore not passed at all.
#
# Everything else is kit/run_sdpo_toolalpaca.sh's argv, key for key and value for value, so a GRPO run
# here is paired with a K0 SDPO run seed for seed. tests/test_kit_k1c.py diffs the two dry runs and
# lists every difference it had to allow. The differences are:
#
#   --config-name baseline_grpo   the method (K0: sdpo)
#   trainer.group_name            rep-grpo-toolalpaca (K0: rep-sdpo-toolalpaca)
#   the 7 self_distillation keys  absent: dead under vanilla
#   ref.fsdp_config.param_offload absent: under GRPO with use_kl_loss False NO reference model is
#                                 built (main_ppo.py:130-131,177 -> Role.ActorRollout;
#                                 fsdp_workers.py:196 _is_ref False; utils.py:72-75), so there is
#                                 nothing to offload. This is also why LoRA is safe here and was not
#                                 safe under SDPO: docs/phase2/k1c/feasibility.md (b).
#
# Nothing here belongs to our project: it runs lasgroup/SDPO at its pinned commit, unmodified.
#
# Required:  SDPO_DIR   checkout of https://github.com/lasgroup/SDPO at 7c457fc1b1f6...
#            MODEL_DIR  local snapshot of Qwen/Qwen3-8B at revision b968826d9c46...
#            NAME       a fresh name per run (never reuse one; outputs are never overwritten)
# Optional:  STEPS=17 TEST_FREQ=17 SEED= NGPU=4 (8 on 80 GB cards) TP=2 OFFLOAD=0 WORK=$PWD/sdpo-work DRY_RUN=0
#
# LORA=1 is the one arm this launcher adds (docs/phase2/k1c/feasibility.md (b)). It appends the three
# declared model keys and RAISES the learning rate tenfold, and nothing else:
#
#   actor_rollout_ref.model.lora_rank=64
#   actor_rollout_ref.model.lora_alpha=32
#   actor_rollout_ref.model.target_modules=all-linear
#   actor_rollout_ref.actor.optim.lr=1e-4      (instead of 1e-5)
#
# rank 64 on all linear layers with alpha 32 and a 10x learning rate is the setting of "LoRA Without
# Regret" (Schulman et al., Thinking Machines, 2025). Rank 64 is an exact vLLM rank
# (rollout/vllm_rollout/utils.py:32), so it is not rounded up silently. `target_modules` is passed
# even though `all-linear` is already the default (_generated_ppo_trainer.yaml:303) because it is the
# arm's identity, and the LoRA arm's argv is a different argv anyway.
#
# THE FOLD. `python -m verl.model_merger merge` does NOT fold a LoRA adapter into the model: it saves
# the UNMODIFIED BASE and writes the adapter beside it with `"lora_alpha": 0` hard-coded
# (base_model_merger.py:268, 301-306). Loading that adapter scales every delta to nothing. So with
# LORA=1 this launcher merges to `merged-step$STEPS` and then folds with kit/fold_lora.py, whose
# output at `hf-step$STEPS` is a full HuggingFace model that kit/score_forgetting.py can score exactly
# like K1a's. `hf-step$STEPS` therefore means the same thing in both arms. A fold that changed nothing
# writes no model and exits 3, because an unnoticed no-op would report "LoRA forgets nothing" without
# a single LoRA weight ever being scored.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
LORA="${LORA:-0}"

# The LoRA arm's three numbers, fixed here rather than taken from the environment: they are a settled
# decision, not a knob, and tests/test_kit_k1c.py pins them.
LORA_RANK=64
LORA_ALPHA=32
LORA_TARGETS=all-linear
LR_FULL=1e-5
LR_LORA=1e-4

OUT="$WORK/runs/$NAME"
if [[ "$OFFLOAD" == "1" ]]; then OFF=True; else OFF=False; fi

# Counted BEFORE the dry run, so a bad value is caught without a GPU.
[[ "$LORA" == "0" || "$LORA" == "1" ]] || { echo "LORA must be 0 or 1, not $LORA" >&2; exit 2; }
if [[ "$LORA" == "1" ]]; then ARM="lora"; LR="$LR_LORA"; else ARM="full"; LR="$LR_FULL"; fi

ARGV=(
  python -m verl.trainer.main_ppo --config-name baseline_grpo
  "data.train_files=[$SDPO_DIR/datasets/tooluse/train.parquet]"
  "data.val_files=[$SDPO_DIR/datasets/tooluse/test.parquet]"
  "actor_rollout_ref.model.path=$MODEL_DIR"
  "actor_rollout_ref.actor.strategy=fsdp2"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.rollout.n=8"
  "data.train_batch_size=32"
  "data.shuffle=True"
  "trainer.experiment_name=$NAME"
  "trainer.default_local_dir=$OUT/tool-grpo"
  "trainer.save_freq=$STEPS"
  "trainer.max_actor_ckpt_to_keep=4"
  "trainer.total_epochs=1"
  "trainer.val_before_train=True"
  "actor_rollout_ref.actor.checkpoint.save_contents=[model]"
  "trainer.resume_mode=disable"
  "trainer.logger=[console,file]"
  "trainer.project_name=r99-reference-runtime"
  "trainer.group_name=rep-grpo-toolalpaca"
  "vars.dir=$SDPO_DIR"
  "vars.task=datasets/tooluse"
  "vars.log_dir=$OUT/tool-grpo/logs"
  "vars.ckpt_dir=$OUT/tool-grpo"
  "custom_reward_function.path=$SDPO_DIR/verl/utils/reward_score/feedback/__init__.py"
  "custom_reward_function.name=compute_score"
  "trainer.total_training_steps=$STEPS"
  "actor_rollout_ref.rollout.val_kwargs.n=16"
  "trainer.test_freq=$TEST_FREQ"
  "actor_rollout_ref.actor.calculate_entropy=True"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.55"
  "actor_rollout_ref.actor.fsdp_config.param_offload=$OFF"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFF"
  "trainer.rollout_data_dir=$OUT/rollouts"
  "trainer.validation_data_dir=$OUT/validation"
  "trainer.n_gpus_per_node=$NGPU"
  "trainer.nnodes=1"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP"
)
# A seed is three overrides, the same three K0 passes. vLLM sampling is unseeded either way.
if [[ -n "$SEED" ]]; then
  ARGV+=(
    "data.seed=$SEED"
    "actor_rollout_ref.actor.data_loader_seed=$SEED"
    "actor_rollout_ref.actor.fsdp_config.seed=$SEED"
  )
fi

# The LoRA arm. Appended last, so with LORA unset the command above is K0's argv with the
# self-distillation block and the reference offload removed and nothing else.
# `rollout.load_format` stays at its default `dummy` (_generated_ppo_trainer.yaml:217): that is what
# makes the FIRST rollout ship the full base weights to vLLM and every later one ship only the
# adapter (fsdp_workers.py:670,689-697,756; fsdp_utils.py:630-651). `layered_summon` is NOT set,
# because fsdp_utils.py:621-626 raises when it is set under `dummy`.
if [[ "$LORA" == "1" ]]; then
  ARGV+=(
    "actor_rollout_ref.model.lora_rank=$LORA_RANK"
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
    "actor_rollout_ref.model.target_modules=$LORA_TARGETS"
  )
fi

if [[ "$DRY_RUN" == "1" ]]; then printf '%s\n' "${ARGV[@]}"; exit 0; fi

if [[ -e "$OUT" ]]; then echo "refusing to overwrite $OUT: pick a fresh NAME" >&2; exit 2; fi
for f in train.parquet test.parquet; do
  [[ -f "$SDPO_DIR/datasets/tooluse/$f" ]] || { echo "missing $f: run data/preprocess.py first" >&2; exit 2; }
done
[[ -f "$MODEL_DIR/config.json" ]] || { echo "MODEL_DIR=$MODEL_DIR is not a HuggingFace model directory" >&2; exit 2; }
[[ "$LORA" == "0" || -f "$KIT/fold_lora.py" ]] || { echo "LORA=1 needs $KIT/fold_lora.py" >&2; exit 2; }
mkdir -p "$OUT/env" "$OUT/tool-grpo/logs"

# What the run was: recorded before it starts, so a crash still leaves an identity behind.
printf '%s\n' "${ARGV[@]}" > "$OUT/env/argv.txt"
git -C "$SDPO_DIR" rev-parse --git-dir > /dev/null 2>&1 || {
  echo "cannot read the commit of $SDPO_DIR: clone it with git as README-partner.md section 2 says" >&2; exit 2; }
git -C "$SDPO_DIR" rev-parse HEAD > "$OUT/env/sdpo-commit.txt" 2>/dev/null || \
  echo "no commit on HEAD" > "$OUT/env/sdpo-commit.txt"
git -C "$SDPO_DIR" status --short > "$OUT/env/sdpo-dirty.txt" 2>/dev/null || true
pip freeze > "$OUT/env/pip-freeze.txt" 2>/dev/null || true
nvidia-smi > "$OUT/env/nvidia-smi.txt" 2>/dev/null || true
ls "$MODEL_DIR" > "$OUT/env/model-files.txt"
date -u +%FT%TZ > "$OUT/env/started-at.txt"

export USER="${USER:-$(whoami)}" TASK="datasets/tooluse" EXPERIMENT="$NAME"
export VLLM_USE_V1=1 WANDB_MODE=disabled
export PYTHONPATH="$SDPO_DIR:${PYTHONPATH:-}"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"

STARTED=$(date -u +%s)
cd "$SDPO_DIR"
set +e
"${ARGV[@]}" 2>&1 | tee "$OUT/console.log"
STATUS=${PIPESTATUS[0]}
set -e
date -u +%FT%TZ > "$OUT/env/finished-at.txt"

# actor save -> a HuggingFace model we can score for retention; the reference's own merger. On the
# LoRA arm the merger's output is the BASE model plus an adapter with lora_alpha 0, so it goes to a
# directory of its own and hf-step$STEPS is written by the fold instead.
MERGED=0
FOLDED=0
FOLD_CHANGED=0
EXPECTED_CHANGED=0
CKPT="$OUT/tool-grpo/global_step_$STEPS/actor"
FINAL="$OUT/hf-step$STEPS"
if [[ "$LORA" == "1" ]]; then MERGE_TARGET="$OUT/merged-step$STEPS"; else MERGE_TARGET="$FINAL"; fi
if [[ "$STATUS" == "0" && -d "$CKPT" ]]; then
  set +e
  python -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$MERGE_TARGET" \
    2>&1 | tee "$OUT/merge.log"
  set -e
  if [[ "$LORA" == "1" && -f "$MERGE_TARGET/config.json" ]]; then
    # The fold, and the check that a silent no-op cannot pass. --checkpoint-adapter is the trainer's
    # OWN adapter config (fsdp_workers.py:1127-1144), which carries the real lora_alpha when it was
    # written; fold_lora.py refuses a disagreement rather than preferring one side quietly.
    set +e
    python "$KIT/fold_lora.py" --merged "$MERGE_TARGET" --out "$FINAL" --alpha "$LORA_ALPHA" \
      --checkpoint-adapter "$CKPT/lora_adapter" 2>&1 | tee "$OUT/fold.log"
    set -e
  fi
  # A merge or fold that produced nothing must still reach the summary below: `merged: 0` is what a
  # campaign's bar reads, and a run that stopped here would leave no summary at all.
  if [[ -f "$FINAL/config.json" ]]; then
    MERGED=1
  elif [[ "$LORA" == "1" ]]; then
    echo "no scoreable model at $FINAL: see $OUT/merge.log and $OUT/fold.log" >&2
  else
    echo "the merge left no model at $FINAL: see $OUT/merge.log" >&2
  fi
  if [[ "$LORA" == "1" && -f "$FINAL/fold.json" ]]; then
    FOLDED=1
    FOLD_CHANGED=$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["changed_tensors"]))' "$FINAL/fold.json" || echo 0)
    EXPECTED_CHANGED=$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["expected_changed"]))' "$FINAL/fold.json" || echo 0)
  fi
fi

# One small JSON of numbers, so a campaign row can gate on what this run actually produced. It says
# which arm ran, so a report cannot silently attribute an arm's result to the wrong command.
cat > "$OUT/run-summary.json" <<JSON
{
 "schema": "kit-grpo-toolalpaca-run.v1",
 "name": "$NAME",
 "arm": "$ARM",
 "steps": $STEPS,
 "test_freq": $TEST_FREQ,
 "returncode": $STATUS,
 "merged": $MERGED,
 "lora": $LORA,
 "lora_rank": $LORA_RANK,
 "lora_alpha": $LORA_ALPHA,
 "learning_rate": "$LR",
 "folded": $FOLDED,
 "fold_changed_tensors": $FOLD_CHANGED,
 "fold_expected_changed": $EXPECTED_CHANGED,
 "n_gpus": $NGPU,
 "seconds": $(( $(date -u +%s) - STARTED )),
 "merged_dir": "$FINAL"
}
JSON
echo "done: $OUT (arm $ARM, returncode $STATUS, merged $MERGED, folded $FOLDED)"
# A LoRA run whose adapter did not reach the model would otherwise be scored as the untrained model
# and read as "LoRA forgets nothing". That is the loudest failure this launcher has.
if [[ "$STATUS" == "0" && "$LORA" == "1" && "$FOLDED" == "0" ]]; then
  echo "the LoRA adapter was NOT folded into a model: $FINAL holds no fold.json. Scoring anything in" >&2
  echo "$OUT would score the UNTRAINED base model. See $OUT/fold.log" >&2
  exit 3
fi
exit "$STATUS"
