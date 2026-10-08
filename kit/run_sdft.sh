#!/usr/bin/env bash
# Plan v4, arm D: demonstration-conditioned SDFT in the pinned verl trainer (docs/plan-v4-teacher-20261006.md, section 3).
#
# The student samples ONE answer a prompt from the QUESTION ALONE, as SDFT's reference does; a teacher with the student's own weights (an EMA
# copy at the reference's rate 0.01, every step) sees the question PLUS the row's verified demonstration (`extra_info.demonstration`,
# put there by the data step) and scores the student's tokens; the loss is per-token FORWARD KL from that teacher to the
# student over the FULL vocabulary, teacher stop-gradient, the SDFT reference's reduction (references/Self-Distillation
# at d775732). No successful-sibling condition: every sample has a target. kit/sdft_objective.py is the objective,
# kit/sdft_verl.py puts it into verl at run time (the pinned checkout's files are never edited, and a verl whose files
# differ from the pinned bytes is refused), kit/sdft_entry.py is the entry.
#
# Modelled on kit/run_sdpo_toolalpaca.sh (arm S's trainer command), with the same records: env/argv.txt, the resolved
# configuration, env/stage.json, env/trainer-started-at.txt, the dumped rollouts, the merged model and run-summary.json.
# tests/test_kit_sdft.py compares the two commands key for key and lists every difference it allows.
#
# Required:  SDPO_DIR   checkout of https://github.com/lasgroup/SDPO at 7c457fc1b1f6...
#            MODEL_DIR  the model this stage starts from (Qwen3-8B, or the previous stage's hf-step folder)
#            NAME       a fresh name per run (never reuse one; outputs are never overwritten)
#            DATASET    a folder holding train.parquet (every row with extra_info.demonstration) and test.parquet (no row
#                       with one): `datasets/<task>` inside SDPO_DIR, as the other launchers, or an absolute path
# REGISTERED for arm D (plan v4 section 3, frozen 7 October). These are NOT knobs: each is a constant below, written
# into the argv and into run-summary.json, checked again by the trainer (kit/sdft_objective.py REGISTERED), and an
# environment that sets LR, TEACHER_RATE, ROLLOUT_N, WARMUP_STEPS or WEIGHT_DECAY to any other value is refused:
#            ROLLOUT_N=1      one student sample a prompt (SDFT's reference: num_generations=1, main.py:119)
#            TEACHER_RATE=0.01 EMA teacher rate after every optimizer step (the reference's ref_model_mixup_alpha with
#                             ref_model_sync_steps=1, main.py:16,125-127)
#            LR=1e-5          actor learning rate (arm S's and the pinned sdpo.yaml's; the reference used 2e-5 / 5e-5)
#            WARMUP_STEPS=10  linear warmup, then a CONSTANT rate (lr_scheduler_type=constant), as arm S; the reference
#                             used warmup ratio 0.1 and cosine decay
#            WEIGHT_DECAY=0.01 verl's AdamW default, which arm S also runs with, stated; the reference (HF default) used 0.0
# Knobs (each recorded in the argv and in run-summary.json):
#            KL=forward       the only value accepted: the reference's paper results used forward KL (README 04/07/26)
#            TOPK=            must stay EMPTY: arm D distils the full vocabulary (distillation_topk=null). Any value is
#                             refused here, and again by the trainer and by the loss
#            MAX_RESPONSE=2048 the plan's common training-response cap (data.max_response_length)
#            STEPS=40         optimizer steps (32 prompts each, one sample a prompt); the training file needs at least
#                             32 x STEPS rows, or the single epoch ends early
#            SEED=            empty = verl's defaults; a value sets data.seed, the actor's data_loader_seed and fsdp seed
#            REWARD_FILE=     the reward function verl loads (logged and dumped only: arm D's loss does not read rewards);
#                             default the authors' own, $SDPO_DIR/verl/utils/reward_score/feedback/__init__.py
# Optional:  NGPU=8 TP=2 OFFLOAD=0 TEST_FREQ=-1 VAL_BEFORE=0 KEEP_TRAINER_CKPT=1 WORK=$PWD/sdft-work DRY_RUN=0
#            One H100 (plan section 11's 1.7B smoke): NGPU=1 TP=1, OFFLOAD=0 (as the K4a and pilot-chain SDPO smokes ran
#            the 1.7B); the micro-batch is one sequence per GPU (user.yaml), not a knob.
#            TEST_FREQ=-1 and VAL_BEFORE=0: no in-trainer validation; scores come from the campaign's own scorer.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export KIT_V4_ARM=D
source "$KIT/v4_profile.sh"
: "${SDPO_DIR:?set SDPO_DIR to the pinned lasgroup/SDPO checkout}"
: "${MODEL_DIR:?set MODEL_DIR to the model directory this stage starts from}"
: "${NAME:?set NAME to a fresh run name}"
: "${DATASET:?set DATASET to the folder holding train.parquet (with demonstrations) and test.parquet}"
# The registered constants (kit/sdft_objective.py REGISTERED holds the same values; tests/test_kit_sdft.py compares them).
readonly D_ROLLOUT_N=1 D_TEACHER_RATE=0.01 D_LR=1e-5 D_WARMUP_STEPS=10 D_WEIGHT_DECAY=0.01 D_LR_SCHEDULER=constant
for pair in "ROLLOUT_N:$D_ROLLOUT_N" "TEACHER_RATE:$D_TEACHER_RATE" "LR:$D_LR" "WARMUP_STEPS:$D_WARMUP_STEPS" "WEIGHT_DECAY:$D_WEIGHT_DECAY"; do
  var="${pair%%:*}"; registered="${pair#*:}"
  if [[ -n "${!var:-}" && "${!var}" != "$registered" ]]; then
    echo "$var is registered for arm D at $registered and must be unset or exactly $registered, not ${!var}" >&2; exit 2
  fi
done
STEPS="${STEPS:-40}"
KL="${KL:-forward}"
TOPK="${TOPK:-}"
MAX_RESPONSE="${MAX_RESPONSE:-2048}"
SEED="${SEED:-}"
REWARD_FILE="${REWARD_FILE:-$KIT/beds/v4_reward.py}"
NGPU="${NGPU:-8}"
TP="${TP:-2}"
OFFLOAD="${OFFLOAD:-0}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE="${VAL_BEFORE:-0}"
KEEP_TRAINER_CKPT="${KEEP_TRAINER_CKPT:-1}"
WORK="${WORK:-$PWD/sdft-work}"
DRY_RUN="${DRY_RUN:-0}"

# Every value is checked BEFORE the dry run, so a bad one is caught without a GPU.
[[ -z "$TOPK" ]] || { echo "TOPK must be empty, not $TOPK: arm D is full-vocabulary forward KL, never top-k" >&2; exit 2; }
[[ "$KL" == "forward" ]] || { echo "KL must be forward, not $KL: the reference's paper results used forward KL" >&2; exit 2; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "STEPS must be a positive integer, not $STEPS" >&2; exit 2; }
[[ "$MAX_RESPONSE" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_RESPONSE must be a positive integer of tokens, not $MAX_RESPONSE" >&2; exit 2; }
[[ -z "$SEED" || "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be empty or a non-negative integer, not $SEED" >&2; exit 2; }
[[ "$NGPU" =~ ^[1-9][0-9]*$ && "$TP" =~ ^[1-9][0-9]*$ ]] || { echo "NGPU and TP must be positive integers" >&2; exit 2; }
[[ "$OFFLOAD" == "0" || "$OFFLOAD" == "1" ]] || { echo "OFFLOAD must be 0 or 1, not $OFFLOAD" >&2; exit 2; }
[[ "$TEST_FREQ" =~ ^(-1|[1-9][0-9]*)$ ]] || { echo "TEST_FREQ must be -1 or a positive integer, not $TEST_FREQ" >&2; exit 2; }
[[ "$VAL_BEFORE" == "0" || "$VAL_BEFORE" == "1" ]] || { echo "VAL_BEFORE must be 0 or 1, not $VAL_BEFORE" >&2; exit 2; }
[[ "$KEEP_TRAINER_CKPT" == "0" || "$KEEP_TRAINER_CKPT" == "1" ]] || { echo "KEEP_TRAINER_CKPT must be 0 or 1, not $KEEP_TRAINER_CKPT" >&2; exit 2; }
if [[ "$DATASET" =~ ^datasets/[A-Za-z0-9_/-]+$ && "$DATASET" != *//* ]]; then
  DATA_DIR="$SDPO_DIR/$DATASET"
elif [[ "$DATASET" =~ ^/[A-Za-z0-9_./-]+$ && "$DATASET" != *..* ]]; then
  DATA_DIR="${DATASET%/}"
else
  echo "DATASET must be datasets/<task> inside the checkout or an absolute folder, not $DATASET" >&2; exit 2
fi
[[ "$REWARD_FILE" == /* ]] || { echo "REWARD_FILE must be an absolute path, not $REWARD_FILE" >&2; exit 2; }

OUT="$WORK/runs/$NAME"
if [[ "$OFFLOAD" == "1" ]]; then OFF=True; else OFF=False; fi
if [[ "$VAL_BEFORE" == "1" ]]; then VB=True; else VB=False; fi

ARGV=(
  python "$KIT/sdft_entry.py" --config-name sdpo
  "data.train_files=[$DATA_DIR/train.parquet]"
  "data.val_files=[$DATA_DIR/test.parquet]"
  "actor_rollout_ref.model.path=$MODEL_DIR"
  "actor_rollout_ref.actor.strategy=fsdp2"
  "actor_rollout_ref.model.use_remove_padding=False"
  "actor_rollout_ref.model.use_fused_kernels=False"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
  "actor_rollout_ref.actor.optim.lr=$D_LR"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=$D_WARMUP_STEPS"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.rollout.n=$D_ROLLOUT_N"
  "data.train_batch_size=32"
  "data.shuffle=False"
  "trainer.experiment_name=$NAME"
  "trainer.default_local_dir=$OUT/sdft"
  "trainer.save_freq=${V4_SAVE_FREQ:-$STEPS}"
  "trainer.max_actor_ckpt_to_keep=4"
  "trainer.total_epochs=1"
  "trainer.val_before_train=$VB"
  "actor_rollout_ref.actor.checkpoint.save_contents=[model]"
  "trainer.resume_mode=disable"
  "trainer.logger=[console,file]"
  "trainer.project_name=plan-v4-teacher"
  "trainer.group_name=arm-d-sdft"
  "vars.dir=$SDPO_DIR"
  "vars.task=$DATASET"
  "vars.log_dir=$OUT/sdft/logs"
  "vars.ckpt_dir=$OUT/sdft"
  "custom_reward_function.path=$REWARD_FILE"
  "custom_reward_function.name=compute_score"
  "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema"
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$D_TEACHER_RATE"
  "actor_rollout_ref.actor.self_distillation.alpha=0.0"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=null"
  "actor_rollout_ref.actor.self_distillation.distillation_add_tail=False"
  "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=False"
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
  # Arm D's own settings, each one a reference value (kit/sdft_objective.py says where each comes from):
  "actor_rollout_ref.actor.self_distillation.full_logit_distillation=True"   # full vocabulary
  "actor_rollout_ref.actor.self_distillation.is_clip=null"                    # no student/old ratio clip in the reference
  "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"                 # token mean per sequence, then sequence mean
  "algorithm.rollout_correction.rollout_is=token"                             # the reference's truncated sampler ratio,
  "algorithm.rollout_correction.rollout_is_threshold=2.0"                     # cap 2.0 (sdpo.yaml's own values, stated)
  "actor_rollout_ref.rollout.calculate_log_probs=True"
  "actor_rollout_ref.actor.optim.lr_scheduler_type=$D_LR_SCHEDULER"           # registered: warmup, then constant
  "actor_rollout_ref.actor.optim.weight_decay=$D_WEIGHT_DECAY"                # registered: verl's (and arm S's) value, stated
)
if [[ -n "$SEED" ]]; then
  ARGV+=(
    "data.seed=$SEED"
    "actor_rollout_ref.actor.data_loader_seed=$SEED"
    "actor_rollout_ref.actor.fsdp_config.seed=$SEED"
  )
fi
ARGV+=("data.max_response_length=$MAX_RESPONSE")

# The common v4 entry supplies the identical ordered schedule to all four arms.
if [[ -n "${KIT_V4_ARM:-}" ]]; then
  ARGV+=("data.max_prompt_length=2048" "actor_rollout_ref.actor.self_distillation.max_reprompt_len=6144" "data.filter_overlong_prompts=False" "data.truncation=error")
fi

[[ -f "$REWARD_FILE" ]] || { echo "missing reward file: $REWARD_FILE" >&2; exit 2; }
[[ "$REWARD_FILE" == "$KIT/beds/v4_reward.py" ]] || { echo "v4 FINISH_GATE requires the registered gate-aware reward" >&2; exit 2; }
v4_check_synthetic_data "$DATA_DIR/train.parquet" "${DATA_MANIFEST:-}"
if [[ "${V4_CHECKPOINT_STATE:-0}" == 1 ]]; then
  ARGV+=("actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]" "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]")
fi
if [[ -n "${V4_RESUME_PATH:-}" ]]; then
  [[ "${V4_RETRY_ADMITTED:-0}" == 1 ]] || { echo 'resume requires reviewed infrastructure admission' >&2; exit 2; }
  ARGV+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=$V4_RESUME_PATH")
fi
if [[ "$DRY_RUN" == "1" ]]; then printf '%s\n' "${ARGV[@]}"; exit 0; fi

# env/stage.json, exactly as kit/run_sdpo_toolalpaca.sh writes it (package 4, round-6 ruling J3).
STAGE_FILE=""
LAST_STAGE=""
stage() {                                                         # stage NAME [STATUS]
  [[ -n "$STAGE_FILE" ]] || return 0
  [[ "$1" == "exited" ]] || LAST_STAGE="$1"
  python - "$STAGE_FILE" "$NAME" "${KIT_P4_JOB:-}" "$1" "${2:-}" "$LAST_STAGE" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, attempt, job, name, status, last = sys.argv[1:7]
try:
    doc = json.load(open(path))
except (OSError, ValueError):
    doc = {"schema": "kit-launcher-stage.v1", "attempt": attempt, "job": job, "stages": []}
entry = {"stage": name, "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}
if status != "":
    entry["returncode"] = int(status)
if name == "exited":
    entry["last_stage"] = last or None
doc["stages"].append(entry)
tmp = "%s.tmp-%d" % (path, os.getpid())
with open(tmp, "w") as handle:
    json.dump(doc, handle, indent=1)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, path)
folder = os.open(os.path.dirname(path), os.O_RDONLY)
os.fsync(folder)
os.close(folder)
PY
}
on_exit() {
  local status=$1
  trap - EXIT
  stage exited "$status" || true
  exit "$status"
}

if [[ -e "$OUT" && "${V4_RETRY_ADMITTED:-0}" != 1 ]]; then echo "refusing to overwrite $OUT: pick a fresh NAME" >&2; exit 2; fi
mkdir -p "$OUT/env"
STAGE_FILE="$OUT/env/stage.json"
trap 'on_exit $?' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
stage started
for f in train.parquet test.parquet; do
  [[ -f "$DATA_DIR/$f" ]] || { echo "missing $DATA_DIR/$f" >&2; exit 2; }
done
[[ -f "$REWARD_FILE" ]] || { echo "missing reward function $REWARD_FILE" >&2; exit 2; }
mkdir -p "$OUT/sdft/logs"

# What the run was: recorded before it starts, so a crash still leaves an identity behind.
printf '%s\n' "${ARGV[@]}" > "$OUT/env/argv.txt"
git -C "$SDPO_DIR" rev-parse HEAD > "$OUT/env/sdpo-commit.txt"
git -C "$SDPO_DIR" status --short > "$OUT/env/sdpo-dirty.txt"
pip freeze > "$OUT/env/pip-freeze.txt" 2>/dev/null || true
nvidia-smi > "$OUT/env/nvidia-smi.txt" 2>/dev/null || true
ls "$MODEL_DIR" > "$OUT/env/model-files.txt"
sha256sum "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet" > "$OUT/env/data-sha256.txt" 2>/dev/null || \
  shasum -a 256 "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet" > "$OUT/env/data-sha256.txt" 2>/dev/null || true
# The kit code that IS arm D, by content: the objective, the verl glue, the entry and this launcher.
( cd "$KIT" && { sha256sum sdft_objective.py sdft_actor.py sdft_verl.py sdft_entry.py run_sdft.sh v4_contract.py v4_profile.sh v4_receipts.py 2>/dev/null || \
  shasum -a 256 sdft_objective.py sdft_actor.py sdft_verl.py sdft_entry.py run_sdft.sh v4_contract.py v4_profile.sh v4_receipts.py; } ) > "$OUT/env/kit-sha256.txt" 2>/dev/null || true
date -u +%FT%TZ > "$OUT/env/started-at.txt"

export PYTHONPATH="$KIT/..:$KIT:$SDPO_DIR:${PYTHONPATH:-}"
if [[ "$V4_PROFILE" == scientific || -n "${DATA_MANIFEST:-}" ]]; then
  export ARM=D
  python "$KIT/v4_run.py" check
fi
# The data, before any GPU: every training row has a demonstration that is not in its own prompt, no held-out row has one.
if ! python "$KIT/sdft_objective.py" check-data "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet" "$OUT/env/data-check.json"; then
  echo "the data is not arm D's: nothing is trained" >&2
  exit 2
fi
stage data-checked

export USER="${USER:-$(whoami)}" TASK="$DATASET" EXPERIMENT="$NAME"
export VLLM_USE_V1=1 WANDB_MODE=disabled
export PYTHONPATH="$KIT/..:$SDPO_DIR:$KIT:${PYTHONPATH:-}"     # kit/ so that Ray's workers import sdft_verl and sdft_objective
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"

STARTED=$(date -u +%s)
cd "$SDPO_DIR"
# The trainer's own resolved configuration of exactly this command, before any training (package 4, amendment 3 B6).
if ! "${ARGV[@]:0:4}" --cfg job --resolve "${ARGV[@]:4}" > "$OUT/env/resolved-config.yaml" 2> "$OUT/env/resolved-config.err" \
   || [[ ! -s "$OUT/env/resolved-config.yaml" ]]; then
  echo "the trainer could not resolve its configuration (see $OUT/env/resolved-config.err): nothing is trained" >&2
  exit 2
fi
stage config-resolved
stage trainer-invoked
date -u +%FT%TZ > "$OUT/env/trainer-started-at.txt"
set +e
"${ARGV[@]}" 2>&1 | tee "$OUT/console.log"
STATUS=${PIPESTATUS[0]}
set -e
date -u +%FT%TZ > "$OUT/env/finished-at.txt"
stage trainer-exited "$STATUS"

complete_export() {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
ok = (d / "config.json").is_file() and (d / "tokenizer_config.json").is_file()
index = d / "model.safetensors.index.json"
if index.is_file():
    shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
    ok = ok and bool(shards) and all((d / s).is_file() and (d / s).stat().st_size > 0 for s in shards)
else:
    ok = ok and (d / "model.safetensors").is_file() and (d / "model.safetensors").stat().st_size > 0
sys.exit(0 if ok else 1)
PY
}

MERGED=0
MERGE_STATUS=-1
CKPT="$OUT/sdft/global_step_$STEPS/actor"
if [[ "$STATUS" == "0" && -d "$CKPT" ]]; then
  set +e
  MERGE_LAUNCH=(python)
  if [[ "${V4_TELEMETRY:-0}" == 1 ]]; then
    MERGE_LAUNCH=(python "$KIT/v4_timing.py" --out "$OUT/env/merge-timing.json" -- python)
  fi
  "${MERGE_LAUNCH[@]}" -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$OUT/hf-step$STEPS" \
    2>&1 | tee "$OUT/merge.log"
  MERGE_STATUS=${PIPESTATUS[0]}
  set -e
  if [[ "$MERGE_STATUS" == "0" ]] && complete_export "$OUT/hf-step$STEPS"; then
    MERGED=1
  else
    echo "the merge (exit $MERGE_STATUS) left no complete model at $OUT/hf-step$STEPS: see $OUT/merge.log. The trainer's checkpoint is kept." >&2
  fi
fi
stage merged "$MERGE_STATUS"

CKPT_KEPT=1
if [[ "$KEEP_TRAINER_CKPT" == "0" && "$MERGED" == "1" ]]; then
  rm -rf "$OUT/sdft/global_step_$STEPS"
  CKPT_KEPT=0
fi

cat > "$OUT/run-summary.json" <<JSON
{
 "schema": "kit-sdft-run.v1",
 "name": "$NAME",
 "arm": "D",
 "objective": "demonstration-conditioned SDFT: per-token forward KL(teacher || student), full vocabulary, teacher stop-gradient",
 "steps": $STEPS,
 "profile": "$V4_PROFILE",
 "scientific_phase_allowed": false,
 "batch_prompts": 32,
 "finish_gate": 1,
 "returncode": $STATUS,
 "merged": $MERGED,
 "registered": {"rollout_n": $D_ROLLOUT_N, "teacher_update_rate": $D_TEACHER_RATE, "learning_rate": $D_LR, "lr_warmup_steps": $D_WARMUP_STEPS, "lr_scheduler": "$D_LR_SCHEDULER", "weight_decay": $D_WEIGHT_DECAY, "steps": 40, "max_response_length": 2048, "batch_prompts": 32, "finish_gate": 1, "teacher_input_cap": 6144},
 "rollout_n": $D_ROLLOUT_N,
 "learning_rate": "$D_LR",
 "lr_warmup_steps": $D_WARMUP_STEPS,
 "lr_scheduler": "$D_LR_SCHEDULER",
 "weight_decay": "$D_WEIGHT_DECAY",
 "teacher_update_rate": "$D_TEACHER_RATE",
 "kl": "$KL",
 "alpha": 0.0,
 "distillation_topk": null,
 "full_vocabulary": true,
 "loss_tokens_skipped": 3,
 "importance_sampling_cap": 2.0,
 "loss_aggregation": "seq-mean-token-mean",
 "successful_sibling_condition": false,
 "dataset": "$DATASET",
 "data_dir": "$DATA_DIR",
 "reward_function": "$REWARD_FILE",
 "model_dir": "$MODEL_DIR",
 "max_response_length": $MAX_RESPONSE,
 "seed": "$SEED",
 "n_gpus": $NGPU,
 "test_freq": $TEST_FREQ,
 "merge_returncode": $MERGE_STATUS,
 "trainer_checkpoint_kept": $CKPT_KEPT,
 "seconds": $(( $(date -u +%s) - STARTED )),
 "merged_dir": "$OUT/hf-step$STEPS"
}
JSON
python "$KIT/v4_receipts.py" "$OUT" D "$STEPS" "$V4_PROFILE"
echo "done: $OUT (arm D, returncode $STATUS, merged $MERGED)"
if [[ "$STATUS" == "0" && "$MERGED" == "0" ]]; then exit 4; fi
exit "$STATUS"
