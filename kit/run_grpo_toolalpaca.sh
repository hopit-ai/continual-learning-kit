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
# Three knobs added for plan v3's 8B pilot (1 October 2026). Left unset, the command is byte for byte the one K1c ran.
#   DATASET=datasets/tooluse   a task directory inside the pinned checkout holding train.parquet and test.parquet,
#                              e.g. datasets/sciknoweval/chemistry (run `python data/preprocess.py --data_source
#                              datasets/sciknoweval/chemistry` there once; the campaign's prepare row does it).
#   LR=                        empty = the arm's own rate (1e-5 full, 1e-4 LoRA). The SDPO paper's tuned GRPO
#                              baseline is LR=1e-6 with MINI_BATCH=8 (its Table 13); 1e-5 with 32 is its
#                              "on-policy" variant, which is what K1c ran.
#   MINI_BATCH=32              `ppo_mini_batch_size`, in prompts: 8 means four optimizer steps per 32-prompt batch.
#   KEEP_TRAINER_CKPT=1        0 deletes the trainer's own checkpoint (`global_step_$STEPS`, full-precision shards,
#                              1.7 times the merged model: about 33 GB for the 8B) once the merged model is on disk. It
#                              is only needed to resume training, which no campaign does. `run-summary.json` records it.
# MODEL_DIR may be a previous run's `hf-step$STEPS` folder: the merger saves the tokenizer with it, so a second
# task can be trained on top of a first.
#
# Two knobs added for plan v3's package 4 (`cap-gate`, 4 October 2026), with kit/run_grpo.sh's semantics. Left at their
# defaults, the command is byte for byte the pilot's (tests/test_kit_p4_launchers.py compares it with HEAD's launcher).
#   MAX_RESPONSE=              empty = the pinned config's 8192 new tokens, nothing passed; a positive integer sets
#                              `data.max_response_length`, the trainer's one cap for training rollouts and its own
#                              validation (user.yaml:15). Package 4's intervention trains at 2048.
#   FINISH_GATE=0              1 exports KIT_FINISH_GATE=1 AND points `custom_reward_function.path` at
#                              kit/beds/authors_gate.py, which calls the authors' own reward function unchanged and pays 0
#                              for a rollout the trainer cut at the cap (naive.py:84 marks it); see that file. 0 unsets
#                              KIT_FINISH_GATE and leaves the reward path the authors' own, exactly as before.
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
DATASET="${DATASET:-datasets/tooluse}"
LR="${LR:-}"
MINI_BATCH="${MINI_BATCH:-32}"
KEEP_TRAINER_CKPT="${KEEP_TRAINER_CKPT:-1}"
MAX_RESPONSE="${MAX_RESPONSE:-}"
FINISH_GATE="${FINISH_GATE:-0}"
# Optional absolute custom reward path. Unset preserves every pilot argv token.
REWARD_FILE="${REWARD_FILE:-}"

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
[[ "$DATASET" =~ ^datasets/[A-Za-z0-9_/-]+$ && "$DATASET" != *//* ]] || { echo "DATASET must be a task directory inside the checkout like datasets/sciknoweval/chemistry, not $DATASET" >&2; exit 2; }
[[ -z "$LR" || "$LR" =~ ^[0-9]+(\.[0-9]+)?(e-?[0-9]+)?$ ]] || { echo "LR must be empty or a number like 1e-6, not $LR" >&2; exit 2; }
[[ "$MINI_BATCH" =~ ^[1-9][0-9]*$ ]] && (( 32 % MINI_BATCH == 0 )) || { echo "MINI_BATCH must divide the 32-prompt batch (1, 2, 4, 8, 16 or 32), not $MINI_BATCH" >&2; exit 2; }
[[ "$KEEP_TRAINER_CKPT" == "0" || "$KEEP_TRAINER_CKPT" == "1" ]] || { echo "KEEP_TRAINER_CKPT must be 0 or 1, not $KEEP_TRAINER_CKPT" >&2; exit 2; }
[[ "$KEEP_TRAINER_CKPT" == "1" || "$LORA" == "0" ]] || { echo "KEEP_TRAINER_CKPT=0 is for full training only: the LoRA arm's adapter lives in the trainer's checkpoint" >&2; exit 2; }
if [[ "$LORA" == "1" ]]; then ARM="lora"; DEFAULT_LR="$LR_LORA"; else ARM="full"; DEFAULT_LR="$LR_FULL"; fi
LR="${LR:-$DEFAULT_LR}"
[[ -z "$MAX_RESPONSE" || "$MAX_RESPONSE" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_RESPONSE must be empty or a positive integer of tokens, not $MAX_RESPONSE" >&2; exit 2; }
[[ "$FINISH_GATE" == "0" || "$FINISH_GATE" == "1" ]] || { echo "FINISH_GATE must be 0 or 1, not $FINISH_GATE" >&2; exit 2; }
# The reward function verl loads. The pilot's is the authors' own, inside the pinned checkout; the gate wraps it.
if [[ "$FINISH_GATE" == "1" ]]; then
  export KIT_FINISH_GATE=1
  REWARD="$KIT/beds/authors_gate.py"
else
  unset KIT_FINISH_GATE                       # an inherited KIT_FINISH_GATE=1 must not reach a control's reward workers
  REWARD="$SDPO_DIR/verl/utils/reward_score/feedback/__init__.py"
fi

if [[ -n "$REWARD_FILE" ]]; then
  [[ "$REWARD_FILE" == /* ]] || { echo "REWARD_FILE must be an absolute path" >&2; exit 2; }
  [[ -f "$REWARD_FILE" ]] || { echo "missing reward file: $REWARD_FILE" >&2; exit 2; }
  [[ "$FINISH_GATE" != 1 || "$REWARD_FILE" == "$KIT/beds/v4_reward.py" || "$REWARD_FILE" == "$KIT/beds/authors_gate.py" ]] || { echo 'custom reward is not verified finish-gate aware' >&2; exit 2; }
  [[ "${FEEDBACK:-0}" == 0 ]] || { echo 'refusing REWARD_FILE with FEEDBACK=1' >&2; exit 2; }
  REWARD="$REWARD_FILE"
fi

ARGV=(
  python -m verl.trainer.main_ppo --config-name baseline_grpo
  "data.train_files=[$SDPO_DIR/$DATASET/train.parquet]"
  "data.val_files=[$SDPO_DIR/$DATASET/test.parquet]"
  "actor_rollout_ref.model.path=$MODEL_DIR"
  "actor_rollout_ref.actor.strategy=fsdp2"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH"
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
  "vars.task=$DATASET"
  "vars.log_dir=$OUT/tool-grpo/logs"
  "vars.ckpt_dir=$OUT/tool-grpo"
  "custom_reward_function.path=$REWARD"
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

# The training cap (package 4). Appended last, so with MAX_RESPONSE unset nothing is added. The trainer keeps the first
# MAX_RESPONSE generated tokens of a rollout and marks it cut when no end token is among them (naive.py:84);
# FINISH_GATE is what makes the reward depend on that mark.
if [[ -n "$MAX_RESPONSE" ]]; then
  ARGV+=("data.max_response_length=$MAX_RESPONSE")
fi

if [[ "$DRY_RUN" == "1" ]]; then printf '%s\n' "${ARGV[@]}"; exit 0; fi

# The launcher's own progress, durably (package 4, round-6 ruling J3): env/stage.json names this attempt (NAME, and the
# KIT_P4_JOB marker the wrapper set) and lists every stage reached -- started, config-resolved, trainer-invoked,
# trainer-exited (with its status), merged -- each written to disk (fsync, atomic rename) before the launcher goes on.
# On ANY exit the EXIT trap appends `exited` with the launcher's status and the last stage reached (a TERM, INT or HUP
# exits through it too). Only a record that ends before `trainer-invoked` WITH that `exited` entry shows that a failed
# attempt rolled nothing out (kit/rollout_stats.py); a record that stops without it shows nothing.
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

if [[ -e "$OUT" ]]; then echo "refusing to overwrite $OUT: pick a fresh NAME" >&2; exit 2; fi
mkdir -p "$OUT/env"
STAGE_FILE="$OUT/env/stage.json"
trap 'on_exit $?' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
stage started
for f in train.parquet test.parquet; do
  [[ -f "$SDPO_DIR/$DATASET/$f" ]] || { echo "missing $SDPO_DIR/$DATASET/$f: run \`python data/preprocess.py --data_source $DATASET\` in $SDPO_DIR first" >&2; exit 2; }
done
[[ -f "$MODEL_DIR/config.json" ]] || { echo "MODEL_DIR=$MODEL_DIR is not a HuggingFace model directory" >&2; exit 2; }
[[ "$LORA" == "0" || -f "$KIT/fold_lora.py" ]] || { echo "LORA=1 needs $KIT/fold_lora.py" >&2; exit 2; }
[[ -f "$REWARD" || ( "$FINISH_GATE" == "0" && -z "$REWARD_FILE" ) ]] || { echo "missing reward function $REWARD" >&2; exit 2; }
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
# The task data the run read, by content (package 4, round 3, F4: a later run proves the same data by these hashes).
sha256sum "$SDPO_DIR/$DATASET/train.parquet" "$SDPO_DIR/$DATASET/test.parquet" > "$OUT/env/data-sha256.txt" 2>/dev/null || \
  shasum -a 256 "$SDPO_DIR/$DATASET/train.parquet" "$SDPO_DIR/$DATASET/test.parquet" > "$OUT/env/data-sha256.txt" 2>/dev/null || true
date -u +%FT%TZ > "$OUT/env/started-at.txt"

export USER="${USER:-$(whoami)}" TASK="$DATASET" EXPERIMENT="$NAME"
export VLLM_USE_V1=1 WANDB_MODE=disabled
export PYTHONPATH="$SDPO_DIR:${PYTHONPATH:-}"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"

STARTED=$(date -u +%s)
cd "$SDPO_DIR"
# The trainer's own resolved configuration of exactly this command (package 4, amendment 3 B6), before any training:
# Hydra's `--cfg job --resolve` prints the job configuration with every interpolation resolved, and exits. A command
# whose configuration cannot be resolved does not train (exit 2, before any GPU work, and no run-summary.json).
if ! "${ARGV[@]:0:5}" --cfg job --resolve "${ARGV[@]:5}" > "$OUT/env/resolved-config.yaml" 2> "$OUT/env/resolved-config.err" \
   || [[ ! -s "$OUT/env/resolved-config.yaml" ]]; then
  echo "the trainer could not resolve its configuration (see $OUT/env/resolved-config.err): nothing is trained" >&2
  exit 2
fi
stage config-resolved
stage trainer-invoked
# The trainer is invoked NOW (package 4, round-5 ruling H3). Kept as a timestamp; since round 6 (J3) its ABSENCE is no
# evidence of anything: only env/stage.json, ending before `trainer-invoked` with its `exited` entry, shows no rollout.
date -u +%FT%TZ > "$OUT/env/trainer-started-at.txt"
set +e
"${ARGV[@]}" 2>&1 | tee "$OUT/console.log"
STATUS=${PIPESTATUS[0]}
set -e
date -u +%FT%TZ > "$OUT/env/finished-at.txt"
stage trainer-exited "$STATUS"

# actor save -> a HuggingFace model we can score for retention; the reference's own merger. On the
# LoRA arm the merger's output is the BASE model plus an adapter with lora_alpha 0, so it goes to a
# directory of its own and hf-step$STEPS is written by the fold instead.
# A merged model is COMPLETE when the merger exited 0 and left a config, a tokenizer (it saves the tokenizer last) and
# every weight file its own index names (or the single model.safetensors), none empty. A merger that died half-way
# leaves a config and some shards: that is not a model, `merged` stays 0, and nothing is deleted.
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
  MERGE_STATUS=${PIPESTATUS[0]}
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
  if [[ "$LORA" == "1" && -f "$FINAL/config.json" ]]; then
    MERGED=1                                    # the LoRA arm's model is the fold's, judged by the fold check below
  elif [[ "$LORA" == "0" && "$MERGE_STATUS" == "0" ]] && complete_export "$FINAL"; then
    MERGED=1
  elif [[ "$LORA" == "1" ]]; then
    echo "no scoreable model at $FINAL: see $OUT/merge.log and $OUT/fold.log" >&2
  else
    echo "the merge (exit $MERGE_STATUS) left no complete model at $FINAL: see $OUT/merge.log. The trainer's checkpoint is kept." >&2
  fi
  if [[ "$LORA" == "1" && -f "$FINAL/fold.json" ]]; then
    FOLDED=1
    FOLD_CHANGED=$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["changed_tensors"]))' "$FINAL/fold.json" || echo 0)
    EXPECTED_CHANGED=$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["expected_changed"]))' "$FINAL/fold.json" || echo 0)
  fi
fi

stage merged "$MERGE_STATUS"                                     # -1: no merge was attempted

# The trainer's checkpoint goes only after a merged model with weights is on disk, and only when asked.
CKPT_KEPT=1
if [[ "$KEEP_TRAINER_CKPT" == "0" && "$LORA" == "0" && "$MERGED" == "1" ]]; then   # MERGED=1 means exit 0 and a complete export
  rm -rf "$OUT/tool-grpo/global_step_$STEPS"
  CKPT_KEPT=0
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
 "mini_batch": $MINI_BATCH,
 "dataset": "$DATASET",
 "model_dir": "$MODEL_DIR",
 "merge_returncode": $MERGE_STATUS,
 "trainer_checkpoint_kept": $CKPT_KEPT,
 "folded": $FOLDED,
 "fold_changed_tensors": $FOLD_CHANGED,
 "fold_expected_changed": $EXPECTED_CHANGED,
 "n_gpus": $NGPU,
 "max_response_length": ${MAX_RESPONSE:-8192},
 "finish_gate": $FINISH_GATE,
 "reward_function": "$REWARD",
 "seed": "$SEED",
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
# A run that trained and has no complete merged model is a failed run: nothing can be scored from it.
if [[ "$STATUS" == "0" && "$MERGED" == "0" ]]; then exit 4; fi
exit "$STATUS"
