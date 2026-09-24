#!/usr/bin/env bash
# SDPO on ONE OF THIS KIT'S BEDS (Spider, GSM8K, FinQA), at the paper's geometry.
#
# This is kit/run_sdpo_toolalpaca.sh's command -- the exact trainer command of our run 3, which
# receipt 201 proved equals that record key for key -- with the DATA FILES, the REWARD FUNCTION, the
# OUTPUT directory and the authors' own FEEDBACK switch made parameters, and nothing else changed.
# tests/test_kit_k4.py runs both scripts with DRY_RUN=1 and the same NAME, WORK and STEPS, compares
# the two argvs key for key AT BOTH SWITCH POSITIONS, and lists every difference it allows: the two
# data files, the reward file, `vars.task`, the group name, and the output directory stem. Any other
# difference fails that test.
#
# Nothing here belongs to our project: it runs lasgroup/SDPO at its pinned commit, unmodified.
#
# WHAT IT IS FOR. K4's TWO SDPO ARMS, which differ in this one key and nothing else:
#
#   FEEDBACK=1  `teacher-hint`. The custom reward function returns the hint as its `feedback` string,
#   (default)   and in this trainer that string reaches the TEACHER's prompt and nothing else
#               (verl/trainer/ppo/ray_trainer.py:629,728-745,762 -- docs/phase2/k4/feasibility.md
#               (a)). So the teacher sees the hint and the student does not.
#   FEEDBACK=0  `teacher-none`: the SDPO-route CONTROL, and run_sdpo_toolalpaca.sh's own default. The
#               reward function, the training file, the dose and the seeds are the hint arm's; the
#               trainer simply never puts the `feedback` string into the teacher's prompt. The two
#               arms therefore differ in exactly one key, and `teacher-hint` minus `teacher-none` is
#               the hint's effect on this route -- which `teacher-hint` minus `none` is not, because
#               that also changes GRPO to SDPO.
#
# The reward function is kit/hints.py, which serves a hint for the questions named in $HINTS_FILE and
# an empty string for every other question. IT IS REQUIRED FOR BOTH ARMS: scoring must be identical
# on both sides of the comparison, and only the trainer's use of the string may differ.
#
# Required:  SDPO_DIR    checkout of https://github.com/lasgroup/SDPO at 7c457fc1b1f6...
#            MODEL_DIR   local model directory to start from
#            NAME        a fresh name per run (never reuse one; outputs are never overwritten)
#            TRAIN_FILE  absolute .parquet of trainer rows (a bed's prepare)
#            VAL_FILE    absolute .parquet the reader can load; in-trainer validation is off by default
#            HINTS_FILE  the hints-filtered.jsonl this arm serves -- required ONLY when REWARD is
#                        kit/hints.py, which is the default
# Optional:  STEPS=40 TEST_FREQ=-1 VAL_BEFORE=0 SEED= NGPU=8 TP=2 OFFLOAD=0 SHUFFLE=0 FEEDBACK=1
#            TASK=datasets/spider_sql REWARD=$KIT/hints.py REWARD_NAME=compute_score
#            WORK=$PWD/k4-work DRY_RUN=0
#
# THREE SETTINGS THAT ARE NOT PATHS, and that a bed run gets wrong if it only changes the paths:
#
#   TASK          user.yaml:6 is `task: ${oc.env:TASK}`, so the config needs the variable even though
#                 every key built from it is overridden here. Exported below, as kit/run_grpo.sh does.
#   SHUFFLE=0     the reused control runs (K3 stage A, K1c part B) were launched by kit/run_grpo.sh
#                 with data.shuffle=False, so the file's order IS the schedule. An arm that shuffled
#                 would see different questions in different steps. Default 1 for argv parity with
#                 run_sdpo_toolalpaca.sh, which is what K0 ran; K4's campaign sets 0.
#   TEST_FREQ=-1  and VAL_BEFORE=0: no in-trainer validation. Every K4 number comes from
#     (default)   kit/eval_bed.py on one GPU of one machine, which is the same measurement the reused
#                 control runs got. A validation here would also cost 16 samples of a 1,147-question
#                 FinQA test file, twice a run, for a number nothing reads.
#
# SPIDER_ROOT, if set, is exported so the reward workers can reach the Spider databases: ray starts
# its workers from this process, so they inherit this environment. Checked here, because a worker that
# cannot open the databases raises deep inside a step a quarter of an hour after the GPUs were taken.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SDPO_DIR:?set SDPO_DIR to the pinned lasgroup/SDPO checkout}"
: "${MODEL_DIR:?set MODEL_DIR to the model directory this run starts from}"
: "${NAME:?set NAME to a fresh run name}"
: "${TRAIN_FILE:?set TRAIN_FILE to the training parquet}"
: "${VAL_FILE:?set VAL_FILE to the validation parquet}"
STEPS="${STEPS:-40}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE="${VAL_BEFORE:-0}"
SEED="${SEED:-}"
NGPU="${NGPU:-8}"
TP="${TP:-2}"
OFFLOAD="${OFFLOAD:-0}"
SHUFFLE="${SHUFFLE:-1}"
TASK="${TASK:-datasets/spider_sql}"
WORK="${WORK:-$PWD/k4-work}"
DRY_RUN="${DRY_RUN:-0}"
REWARD="${REWARD:-$KIT/hints.py}"
REWARD_NAME="${REWARD_NAME:-compute_score}"
FEEDBACK="${FEEDBACK:-1}"

OUT="$WORK/runs/$NAME"
if [[ "$OFFLOAD" == "1" ]]; then OFF=True; else OFF=False; fi
[[ "$SHUFFLE" == "0" || "$SHUFFLE" == "1" ]] || { echo "SHUFFLE must be 0 or 1, not $SHUFFLE" >&2; exit 2; }
[[ "$VAL_BEFORE" == "0" || "$VAL_BEFORE" == "1" ]] || { echo "VAL_BEFORE must be 0 or 1, not $VAL_BEFORE" >&2; exit 2; }
[[ "$FEEDBACK" == "0" || "$FEEDBACK" == "1" ]] || { echo "FEEDBACK must be 0 or 1, not $FEEDBACK" >&2; exit 2; }
if [[ "$SHUFFLE" == "1" ]]; then SHUF=True; else SHUF=False; fi
if [[ "$VAL_BEFORE" == "1" ]]; then VALB=True; else VALB=False; fi
# The switch, the arm it names, and nothing else between them: the run records which arm it was, so a
# report can never attribute one arm's numbers to the other.
if [[ "$FEEDBACK" == "1" ]]; then FB=True; ARM=teacher-hint; else FB=False; ARM=teacher-none; fi

# The order below is run_sdpo_toolalpaca.sh's order, so the two can be compared token by token.
ARGV=(
  python -m verl.trainer.main_ppo --config-name sdpo
  "data.train_files=[$TRAIN_FILE]"
  "data.val_files=[$VAL_FILE]"
  "actor_rollout_ref.model.path=$MODEL_DIR"
  "actor_rollout_ref.actor.strategy=fsdp2"
  "actor_rollout_ref.actor.optim.lr=1e-5"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.rollout.n=8"
  "data.train_batch_size=32"
  "data.shuffle=$SHUF"
  "trainer.experiment_name=$NAME"
  "trainer.default_local_dir=$OUT/train"
  "trainer.save_freq=$STEPS"
  "trainer.max_actor_ckpt_to_keep=4"
  "trainer.total_epochs=1"
  "trainer.val_before_train=$VALB"
  "actor_rollout_ref.actor.checkpoint.save_contents=[model]"
  "trainer.resume_mode=disable"
  "trainer.logger=[console,file]"
  "trainer.project_name=r99-reference-runtime"
  "trainer.group_name=k4-sdpo-bed"
  "vars.dir=$SDPO_DIR"
  "vars.task=$TASK"
  "vars.log_dir=$OUT/train/logs"
  "vars.ckpt_dir=$OUT/train"
  "custom_reward_function.path=$REWARD"
  "custom_reward_function.name=$REWARD_NAME"
  "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema"
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05"
  "actor_rollout_ref.actor.self_distillation.alpha=0.5"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
  "actor_rollout_ref.actor.self_distillation.distillation_add_tail=True"
  # The hint reaches the teacher through the reward function's `feedback` string, so this key IS the
  # difference between K4's two SDPO arms, and the only one. It is also the ONE self-distillation key
  # whose value differs from K0's default command when it is on.
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
# A seed is three overrides, as in run_sdpo_toolalpaca.sh. vLLM sampling is unseeded either way, and
# the GRPO control runs set data.seed alone (kit/run_grpo.sh:106): say so in any readout.
if [[ -n "$SEED" ]]; then
  ARGV+=(
    "data.seed=$SEED"
    "actor_rollout_ref.actor.data_loader_seed=$SEED"
    "actor_rollout_ref.actor.fsdp_config.seed=$SEED"
  )
fi

if [[ "$DRY_RUN" == "1" ]]; then printf '%s\n' "${ARGV[@]}"; exit 0; fi

if [[ -e "$OUT" ]]; then echo "refusing to overwrite $OUT: pick a fresh NAME" >&2; exit 2; fi
for f in "$TRAIN_FILE" "$VAL_FILE"; do
  [[ -f "$f" ]] || { echo "missing data file $f: run the bed's prepare (and kit/hints.py apply) first" >&2; exit 2; }
done
[[ -f "$MODEL_DIR/config.json" ]] || { echo "MODEL_DIR=$MODEL_DIR is not a HuggingFace model directory" >&2; exit 2; }
[[ -f "$REWARD" ]] || { echo "missing reward function $REWARD" >&2; exit 2; }
# The hints are what the `teacher-hint` arm IS, and `teacher-none` is only its control if it scored
# the attempts the same way, so BOTH arms load them and the file is checked before the GPUs are taken.
# A hint run whose reward function cannot find them would be an ordinary SDPO run reporting itself as
# a hint arm; a control run without them would differ from the treatment in two things, not one.
if [[ "$REWARD" == "$KIT/hints.py" ]]; then
  : "${HINTS_FILE:?set HINTS_FILE to the hints-filtered.jsonl this arm serves to the teacher}"
  [[ -s "$HINTS_FILE" ]] || { echo "HINTS_FILE=$HINTS_FILE is missing or empty: the teacher would be shown nothing" >&2; exit 2; }
  export HINTS_FILE
fi
mkdir -p "$OUT/env" "$OUT/train/logs"

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
sha256sum "$TRAIN_FILE" "$VAL_FILE" ${HINTS_FILE:+"$HINTS_FILE"} > "$OUT/env/data-sha256.txt" 2>/dev/null || \
  shasum -a 256 "$TRAIN_FILE" "$VAL_FILE" ${HINTS_FILE:+"$HINTS_FILE"} > "$OUT/env/data-sha256.txt" 2>/dev/null || true
date -u +%FT%TZ > "$OUT/env/started-at.txt"

export USER="${USER:-$(whoami)}" TASK EXPERIMENT="$NAME"
export VLLM_USE_V1=1 WANDB_MODE=disabled
export PYTHONPATH="$SDPO_DIR:${PYTHONPATH:-}"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"
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
# reference's own merger.
MERGED=0
CKPT="$OUT/train/global_step_$STEPS/actor"
if [[ "$STATUS" == "0" && -d "$CKPT" ]]; then
  set +e
  python -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$OUT/hf-step$STEPS" \
    2>&1 | tee "$OUT/merge.log"
  set -e
  # A merge that produced nothing must still reach the summary below: `merged: 0` is what a campaign's
  # bar reads, and a run that stopped here would leave no summary at all.
  if [[ -f "$OUT/hf-step$STEPS/config.json" ]]; then
    MERGED=1
  else
    echo "the merge left no model at $OUT/hf-step$STEPS: see $OUT/merge.log" >&2
  fi
fi

# One small JSON of numbers, so a campaign row can gate on what this run actually produced. `hints`
# is in it because a report must never attribute a hint arm's result to a run that served no hints,
# and `feedback` with `arm` because the two SDPO arms are one key apart and a control that ran with
# the switch on would look like a treatment.
cat > "$OUT/train-summary.json" <<JSON
{
 "schema": "kit-sdpo-bed-run.v1",
 "name": "$NAME",
 "arm": "$ARM",
 "steps": $STEPS,
 "test_freq": $TEST_FREQ,
 "returncode": $STATUS,
 "merged": $MERGED,
 "feedback": $FEEDBACK,
 "shuffle": $SHUFFLE,
 "n_gpus": $NGPU,
 "seconds": $(( $(date -u +%s) - STARTED )),
 "model_dir": "$MODEL_DIR",
 "train_file": "$TRAIN_FILE",
 "reward_function": "$REWARD",
 "reward_name": "$REWARD_NAME",
 "hints_file": "${HINTS_FILE:-}",
 "hints": $(if [[ -s "${HINTS_FILE:-}" ]]; then wc -l < "$HINTS_FILE" | tr -d ' '; else echo 0; fi),
 "merged_dir": "$OUT/hf-step$STEPS"
}
JSON
echo "done: $OUT (arm $ARM, returncode $STATUS, merged $MERGED)"
exit "$STATUS"
