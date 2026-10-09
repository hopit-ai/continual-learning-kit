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
#
# K4a arms (docs/phase2/k4a/feasibility.md). Each is ONE declared override and nothing else; with
# none of them set the argv below is byte for byte what K0 ran. AT MOST ONE PER RUN -- two is a
# refusal (exit 2), because a run that changed two things answers neither question:
#            FEEDBACK=1  the authors' own switch: the teacher is shown the checker's mismatch
#                        message, which names the expected actions and arguments. The student's own
#                        prompt and the validation prompt never see it.
#            SOFT=1      a partial-credit TRAINING reward (kit/beds/tooluse_soft.py). In SDPO the
#                        reward only picks which sibling attempt is shown to the teacher as a worked
#                        example, so this means "when nothing is exactly right, show the closest
#                        near-miss". The validation score stays the authors' strict all-or-nothing
#                        one, because that metric reads `acc`, which the wrapper copies unchanged.
#            TEMP=1.2    hotter training sampling. Validation samples at val_kwargs.temperature=0.6
#                        whatever this is set to. It also rescales the logits in every log-prob
#                        forward, teacher included: see the feasibility note before using it.
#
# Two knobs added for plan v3's package 4 (`cap-gate`, 4 October 2026), with kit/run_grpo.sh's semantics. Like the
# recipe knobs they are not arms. Left at their defaults, the command is byte for byte the pilot's
# (tests/test_kit_p4_launchers.py compares it with HEAD's launcher).
#            MAX_RESPONSE=  empty = the pinned config's 8192 new tokens, nothing passed; a positive integer sets
#                        `data.max_response_length`, the trainer's one cap for training rollouts and its own
#                        validation (user.yaml:15). Package 4's intervention trains at 2048.
#            FINISH_GATE=0  1 exports KIT_FINISH_GATE=1 AND points `custom_reward_function.path` at
#                        kit/beds/authors_gate.py, which calls the authors' own reward function unchanged and pays 0
#                        for a rollout the trainer cut at the cap (naive.py:84 marks it). Under SDPO the reward picks
#                        which sibling is shown to the teacher (score >= 0.5), so a cut answer is never a worked
#                        example. 0 unsets KIT_FINISH_GATE and leaves the reward path as before. FINISH_GATE=1 with
#                        SOFT=1 is refused: the two are different reward functions and verl takes one.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SDPO_DIR:?set SDPO_DIR to the pinned lasgroup/SDPO checkout}"
: "${MODEL_DIR:?set MODEL_DIR to the Qwen3-8B snapshot directory}"
: "${NAME:?set NAME to a fresh run name}"
if [[ -n "${KIT_V4_ARM:-}" || "${DATASET:-}" == datasets/v4_* ]]; then
  export KIT_V4_ARM=S
  source "$KIT/v4_profile.sh"
  [[ "${ROLLOUT_N:-8}" == 8 ]] || { echo "S ROLLOUT_N is registered at 8" >&2; exit 2; }
  [[ "${TEACHER_RATE:-0.05}" == 0.05 ]] || { echo 'TEACHER_RATE is registered at 0.05' >&2; exit 2; }
fi
STEPS="${STEPS:-17}"
TEST_FREQ="${TEST_FREQ:-$STEPS}"
SEED="${SEED:-}"
NGPU="${NGPU:-4}"
TP="${TP:-2}"
OFFLOAD="${OFFLOAD:-0}"
WORK="${WORK:-$PWD/sdpo-work}"
DRY_RUN="${DRY_RUN:-0}"
FEEDBACK="${FEEDBACK:-0}"
SOFT="${SOFT:-0}"
TEMP="${TEMP:-}"
# Three knobs added for plan v3's 8B pilot (1 October 2026). They are a RECIPE, not an arm: the one-arm rule below
# still counts only FEEDBACK, SOFT and TEMP. Left unset, the command is byte for byte K0's.
#   DATASET=datasets/tooluse   a task directory inside the pinned checkout holding train.parquet and test.parquet
#   LR=1e-5                    the actor learning rate (the pinned sdpo.yaml's own value)
#   TEACHER_RATE=0.05          `self_distillation.teacher_update_rate`: the teacher moves this fraction towards the
#                              student after every training step. 0 freezes the teacher at the weights the run
#                              STARTED from (MODEL_DIR), which is how a teacher "frozen within a stage" is run:
#                              start the next stage from the previous stage's hf-step folder and the teacher is
#                              that checkpoint.
#   KEEP_TRAINER_CKPT=1        0 deletes the trainer's own checkpoint (`global_step_$STEPS`, full-precision shards,
#                              1.7 times the merged model: about 33 GB for the 8B) once the merged model is on disk. It
#                              is only needed to resume training, which no campaign does. `run-summary.json` records it.
DATASET="${DATASET:-datasets/tooluse}"
LR="${LR:-1e-5}"
TEACHER_RATE="${TEACHER_RATE:-0.05}"
KEEP_TRAINER_CKPT="${KEEP_TRAINER_CKPT:-1}"
MAX_RESPONSE="${MAX_RESPONSE:-}"
FINISH_GATE="${FINISH_GATE:-0}"
# Optional absolute custom reward path. Unset preserves every pilot argv token.
REWARD_FILE="${REWARD_FILE:-}"

OUT="$WORK/runs/$NAME"
if [[ "$OFFLOAD" == "1" ]]; then OFF=True; else OFF=False; fi

# One run is one arm. Counted BEFORE the dry run, so a bad combination is caught without a GPU.
[[ "$FEEDBACK" == "0" || "$FEEDBACK" == "1" ]] || { echo "FEEDBACK must be 0 or 1, not $FEEDBACK" >&2; exit 2; }
[[ "$SOFT" == "0" || "$SOFT" == "1" ]] || { echo "SOFT must be 0 or 1, not $SOFT" >&2; exit 2; }
if [[ -n "$TEMP" ]]; then
  [[ "$TEMP" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "TEMP must be a positive number, not $TEMP" >&2; exit 2; }
fi
if [[ "$DATASET" =~ ^datasets/[A-Za-z0-9_/-]+$ && "$DATASET" != *//* ]]; then
  DATA_DIR="$SDPO_DIR/$DATASET"
elif [[ "${KIT_V4_ARM:-}" == S && "$DATASET" =~ ^/[A-Za-z0-9_./-]+$ && "$DATASET" != *..* ]]; then
  DATA_DIR="${DATASET%/}"
else
  echo "DATASET must be datasets/<task>, or an absolute scheduled folder for v4 S, not $DATASET" >&2; exit 2
fi
[[ "$LR" =~ ^[0-9]+(\.[0-9]+)?(e-?[0-9]+)?$ ]] || { echo "LR must be a number like 1e-5, not $LR" >&2; exit 2; }
[[ "$TEACHER_RATE" =~ ^(0|1|0?\.[0-9]+|1\.0+|0\.0+)$ ]] || { echo "TEACHER_RATE must be a number from 0 to 1 like 0.05, not $TEACHER_RATE" >&2; exit 2; }
[[ "$KEEP_TRAINER_CKPT" == "0" || "$KEEP_TRAINER_CKPT" == "1" ]] || { echo "KEEP_TRAINER_CKPT must be 0 or 1, not $KEEP_TRAINER_CKPT" >&2; exit 2; }
[[ -z "$MAX_RESPONSE" || "$MAX_RESPONSE" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_RESPONSE must be empty or a positive integer of tokens, not $MAX_RESPONSE" >&2; exit 2; }
[[ "$FINISH_GATE" == "0" || "$FINISH_GATE" == "1" ]] || { echo "FINISH_GATE must be 0 or 1, not $FINISH_GATE" >&2; exit 2; }
if [[ "$FINISH_GATE" == "1" && "$SOFT" == "1" ]]; then
  echo "refusing FINISH_GATE=1 with SOFT=1: each is its own reward function and verl loads one. Run them separately." >&2
  exit 2
fi
if [[ "$FINISH_GATE" == "1" && "$FEEDBACK" == "1" ]]; then
  echo "refusing FINISH_GATE=1 with FEEDBACK=1: the teacher would be shown the gate's own note as feedback, a combination nobody registered." >&2
  exit 2
fi
if [[ "$FINISH_GATE" == "1" ]]; then export KIT_FINISH_GATE=1; else unset KIT_FINISH_GATE; fi   # an inherited 1 must not leak into a control
ARMS=()
if [[ "$FEEDBACK" == "1" ]]; then ARMS+=("FEEDBACK=1"); fi
if [[ "$SOFT" == "1" ]]; then ARMS+=("SOFT=1"); fi
if [[ -n "$TEMP" ]]; then ARMS+=("TEMP=$TEMP"); fi
if (( ${#ARMS[@]} > 1 )); then
  echo "refusing to run ${#ARMS[@]} arms at once (${ARMS[*]}): one run is one arm, so that a difference" >&2
  echo "in the result can be attributed to one change. Run them separately." >&2
  exit 2
fi
ARM="${ARMS[0]:-none}"

if [[ "$FEEDBACK" == "1" ]]; then FB=True; else FB=False; fi
# The reward function verl loads. K0's is the authors' own, inside the pinned checkout.
if [[ "$SOFT" == "1" ]]; then REWARD="$KIT/beds/tooluse_soft.py"
elif [[ "$FINISH_GATE" == "1" ]]; then REWARD="$KIT/beds/authors_gate.py"
else REWARD="$SDPO_DIR/verl/utils/reward_score/feedback/__init__.py"; fi

if [[ -n "$REWARD_FILE" ]]; then
  [[ "$REWARD_FILE" == /* ]] || { echo "REWARD_FILE must be an absolute path" >&2; exit 2; }
  [[ "$SOFT" == "0" ]] || { echo "refusing REWARD_FILE with SOFT=1: each supplies a reward function" >&2; exit 2; }
  [[ -f "$REWARD_FILE" ]] || { echo "missing reward file: $REWARD_FILE" >&2; exit 2; }
  [[ "$FINISH_GATE" != 1 || "$REWARD_FILE" == "$KIT/beds/v4_reward.py" || "$REWARD_FILE" == "$KIT/beds/authors_gate.py" ]] || { echo 'custom reward is not verified finish-gate aware' >&2; exit 2; }
  [[ "${FEEDBACK:-0}" == 0 ]] || { echo 'refusing REWARD_FILE with FEEDBACK=1' >&2; exit 2; }
  REWARD="$REWARD_FILE"
fi

ARGV=(
  python -m verl.trainer.main_ppo --config-name sdpo
  "data.train_files=[$DATA_DIR/train.parquet]"
  "data.val_files=[$DATA_DIR/test.parquet]"
  "actor_rollout_ref.model.path=$MODEL_DIR"
  "actor_rollout_ref.actor.strategy=fsdp2"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.rollout.n=8"
  "data.train_batch_size=32"
  "data.shuffle=True"
  "trainer.experiment_name=$NAME"
  "trainer.default_local_dir=$OUT/tool-sdpo"
  "trainer.save_freq=${V4_SAVE_FREQ:-$STEPS}"
  "trainer.max_actor_ckpt_to_keep=4"
  "trainer.total_epochs=1"
  "trainer.val_before_train=True"
  "actor_rollout_ref.actor.checkpoint.save_contents=[model]"
  "trainer.resume_mode=disable"
  "trainer.logger=[console,file]"
  "trainer.project_name=r99-reference-runtime"
  "trainer.group_name=rep-sdpo-toolalpaca"
  "vars.dir=$SDPO_DIR"
  "vars.task=$DATASET"
  "vars.log_dir=$OUT/tool-sdpo/logs"
  "vars.ckpt_dir=$OUT/tool-sdpo"
  "custom_reward_function.path=$REWARD"
  "custom_reward_function.name=compute_score"
  "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema"
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_RATE"
  "actor_rollout_ref.actor.self_distillation.alpha=0.5"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
  "actor_rollout_ref.actor.self_distillation.distillation_add_tail=True"
  "actor_rollout_ref.actor.self_distillation.include_environment_feedback=$FB"
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

# The VARIATION arm. K0's argv does not mention temperature at all (verl's default is 1.0), so an
# unset TEMP appends nothing and the command above stays byte for byte K0's.
if [[ -n "$TEMP" ]]; then ARGV+=("actor_rollout_ref.rollout.temperature=$TEMP"); fi

# The training cap (package 4). Appended last, so with MAX_RESPONSE unset nothing is added. The trainer keeps the first
# MAX_RESPONSE generated tokens of a rollout and marks it cut when no end token is among them (naive.py:84);
# FINISH_GATE is what makes the reward depend on that mark.
if [[ -n "$MAX_RESPONSE" ]]; then ARGV+=("data.max_response_length=$MAX_RESPONSE"); fi

# The common v4 entry supplies the identical ordered schedule to all four arms.
if [[ -n "${KIT_V4_ARM:-}" ]]; then
  ARGV[2]=kit.v4_ppo_entry
  ARGV+=("data.shuffle=False" "trainer.val_before_train=False"
         "actor_rollout_ref.actor.optim.lr_scheduler_type=constant"
         "actor_rollout_ref.actor.optim.weight_decay=0.01"
         "data.max_prompt_length=2048" "data.filter_overlong_prompts=False" "data.truncation=error")
fi

if [[ -n "${KIT_V4_ARM:-}" ]]; then v4_check_synthetic_data "$DATA_DIR/train.parquet" "${DATA_MANIFEST:-}"; fi
if [[ "${V4_CHECKPOINT_STATE:-0}" == 1 ]]; then
  ARGV+=("actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]" "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]")
fi
if [[ -n "${V4_RESUME_PATH:-}" ]]; then
  [[ "${V4_RETRY_ADMITTED:-0}" == 1 ]] || { echo 'resume requires reviewed infrastructure admission' >&2; exit 2; }
  ARGV+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=$V4_RESUME_PATH")
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

if [[ -e "$OUT" && "${V4_RETRY_ADMITTED:-0}" != 1 ]]; then echo "refusing to overwrite $OUT: pick a fresh NAME" >&2; exit 2; fi
mkdir -p "$OUT/env"
STAGE_FILE="$OUT/env/stage.json"
trap 'on_exit $?' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
stage started
for f in train.parquet test.parquet; do
  [[ -f "$DATA_DIR/$f" ]] || { echo "missing $DATA_DIR/$f: for datasets/<task>, run \`python data/preprocess.py --data_source $DATASET\` in $SDPO_DIR first" >&2; exit 2; }
done
[[ -f "$REWARD" ]] || { echo "missing reward function $REWARD" >&2; exit 2; }
mkdir -p "$OUT/env" "$OUT/tool-sdpo/logs"

# What the run was: recorded before it starts, so a crash still leaves an identity behind.
printf '%s\n' "${ARGV[@]}" > "$OUT/env/argv.txt"
git -C "$SDPO_DIR" rev-parse HEAD > "$OUT/env/sdpo-commit.txt"
git -C "$SDPO_DIR" status --short > "$OUT/env/sdpo-dirty.txt"
pip freeze > "$OUT/env/pip-freeze.txt" 2>/dev/null || true
nvidia-smi > "$OUT/env/nvidia-smi.txt" 2>/dev/null || true
ls "$MODEL_DIR" > "$OUT/env/model-files.txt"
# The task data the run read, by content (package 4, round 3, F4: a later run proves the same data by these hashes).
sha256sum "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet" > "$OUT/env/data-sha256.txt" 2>/dev/null || \
  shasum -a 256 "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet" > "$OUT/env/data-sha256.txt" 2>/dev/null || true
date -u +%FT%TZ > "$OUT/env/started-at.txt"

export USER="${USER:-$(whoami)}" TASK="$DATASET" EXPERIMENT="$NAME"
export VLLM_USE_V1=1 WANDB_MODE=disabled
export PYTHONPATH="$KIT/..:$SDPO_DIR:${PYTHONPATH:-}"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"

if [[ -n "${KIT_V4_ARM:-}" ]]; then
  export ARM=S
  python "$KIT/v4_run.py" check
fi
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

# actor save -> a HuggingFace model we can score for retention; the reference's own merger.
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
CKPT="$OUT/tool-sdpo/global_step_$STEPS/actor"
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
  # A merge that produced nothing must still reach the summary below: `merged: 0` is what a
  # campaign's bar reads, and a run that stopped here would leave no summary at all.
  if [[ "$MERGE_STATUS" == "0" ]] && complete_export "$OUT/hf-step$STEPS"; then
    MERGED=1
  else
    echo "the merge (exit $MERGE_STATUS) left no complete model at $OUT/hf-step$STEPS: see $OUT/merge.log. The trainer's checkpoint is kept." >&2
  fi
fi

stage merged "$MERGE_STATUS"                                     # -1: no merge was attempted

# The trainer's checkpoint goes only after a merged model with weights is on disk, and only when asked.
CKPT_KEPT=1
if [[ "$KEEP_TRAINER_CKPT" == "0" && "$MERGED" == "1" ]]; then                    # MERGED=1 means exit 0 and a complete export
  rm -rf "$OUT/tool-sdpo/global_step_$STEPS"
  CKPT_KEPT=0
fi

# One small JSON of numbers, so a campaign row can gate on what this run actually produced. It says
# which arm ran, so a report cannot silently attribute an arm's result to the wrong command.
cat > "$OUT/run-summary.json" <<JSON
{
 "schema": "kit-sdpo-run.v1",
 "name": "$NAME",
 "arm": "$ARM",
 "steps": $STEPS,
 "test_freq": $TEST_FREQ,
 "returncode": $STATUS,
 "merged": $MERGED,
 "feedback": $FEEDBACK,
 "soft": $SOFT,
 "temperature": "${TEMP:-default}",
 "learning_rate": "$LR",
 "teacher_update_rate": "$TEACHER_RATE",
 "dataset": "$DATASET",
 "model_dir": "$MODEL_DIR",
 "merge_returncode": $MERGE_STATUS,
 "trainer_checkpoint_kept": $CKPT_KEPT,
 "n_gpus": $NGPU,
 "max_response_length": ${MAX_RESPONSE:-8192},
 "finish_gate": $FINISH_GATE,
 "seed": "$SEED",
 "seconds": $(( $(date -u +%s) - STARTED )),
 "reward_function": "$REWARD",
 "merged_dir": "$OUT/hf-step$STEPS"
}
JSON
if [[ -n "${KIT_V4_ARM:-}" ]]; then
  python "$KIT/v4_receipts.py" "$OUT" S "$STEPS" "$V4_PROFILE"
fi
echo "done: $OUT (arm $ARM, returncode $STATUS, merged $MERGED)"
# A run that trained and has no complete merged model is a failed run: nothing can be scored from it.
if [[ "$STATUS" == "0" && "$MERGED" == "0" ]]; then exit 4; fi
exit "$STATUS"
