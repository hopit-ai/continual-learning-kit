#!/usr/bin/env bash
# Plan v4 arms F/R, one configuration. Invoke directly or ARM=F bash kit/run_sft.sh.
# Required: SDPO_DIR MODEL_DIR NAME TRAIN_FILE VAL_FILE DATA_MANIFEST SEED
# Data is built by v4_sft.py --training-set from the COMMON 1,280-ID schedule. No sampler shuffle.
# V4_PROFILE determines scientific vs technical-smoke; scientific dose is 40x32.
# LR is frozen at 1e-5 (SDFT d775732 has no SFT baseline LR); see v4_sft.py.
set -euo pipefail
SFT_ARM="${SFT_ARM:-F}"
[[ "$SFT_ARM" == F || "$SFT_ARM" == R ]] || { echo "SFT_ARM must be F or R" >&2; exit 2; }
if [[ "$SFT_ARM" == F ]]; then TARGET_FIELD=extra_info.demonstration; else TARGET_FIELD=extra_info.rewrite; fi
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SDPO_DIR:?set SDPO_DIR to pinned SDPO checkout}"
: "${MODEL_DIR:?set MODEL_DIR to incoming HF model}"
: "${NAME:?set NAME to a fresh run name}"
: "${TRAIN_FILE:?set TRAIN_FILE to v4_sft.py train.parquet}"
: "${VAL_FILE:?set VAL_FILE to training-only validation parquet}"
: "${DATA_MANIFEST:?set DATA_MANIFEST to v4_sft.py manifest.json}"
: "${SEED:?set SEED to common prompt schedule seed}"
source "$KIT/v4_profile.sh"
STEPS="${STEPS:-40}"
NGPU="${NGPU:-8}"
WORK="${WORK:-$PWD/v4-work}"
DRY_RUN="${DRY_RUN:-0}"
FILE_LOG="${FILE_LOG:-1}"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] && (( STEPS <= 40 )) || { echo 'STEPS must be 1..40; <40 is smoke only' >&2; exit 2; }
[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] && (( 32 % NGPU == 0 )) || { echo 'NGPU must divide 32' >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo 'SEED must be an integer' >&2; exit 2; }
[[ "${LR:-1e-5}" == '1e-5' ]] || { echo 'arm F LR is frozen at 1e-5' >&2; exit 2; }
[[ "$FILE_LOG" == 0 || "$FILE_LOG" == 1 ]] || { echo 'FILE_LOG must be 0 or 1' >&2; exit 2; }
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ && "$NAME" != '.' && "$NAME" != '..' ]] || { echo 'NAME must be a simple run name' >&2; exit 2; }
# The trainer changes cwd to SDPO_DIR; every path in argv must survive it.
eval_paths=$(python - "$SDPO_DIR" "$MODEL_DIR" "$TRAIN_FILE" "$VAL_FILE" "$DATA_MANIFEST" "$WORK" <<'PY'
import os, shlex, sys
for name, value in zip(('SDPO_DIR', 'MODEL_DIR', 'TRAIN_FILE', 'VAL_FILE', 'DATA_MANIFEST', 'WORK'), sys.argv[1:]):
    print(name + '=' + shlex.quote(os.path.abspath(value)))
PY
)
eval "$eval_paths"
OUT="$WORK/runs/$NAME"
if [[ "$FILE_LOG" == 1 ]]; then LOGGER='[console,file]'; else LOGGER='[console]'; fi
ARGV=(python -m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=$NGPU"
  "$KIT/sft_entry.py" --config-name sft_trainer
  "data.train_files=[$TRAIN_FILE]" "data.val_files=[$VAL_FILE]"
  "data.prompt_key=prompt" "data.response_key=$TARGET_FIELD"
  "data.custom_cls.path=$KIT/v4_sft.py" "data.custom_cls.name=V4SFTDataset"
  "+data.max_prompt_length=4096" "+data.max_response_length=2048" "data.max_length=6144" "data.truncation=error" "data.train_batch_size=32"
  "data.micro_batch_size_per_gpu=$(( 32 / NGPU < 1 ? 32 / NGPU : 1 ))"
  "+data.apply_chat_template_kwargs.enable_thinking=False"
  "model.partial_pretrain=$MODEL_DIR" "model.strategy=fsdp2" "model.lora_rank=0"
  "model.enable_gradient_checkpointing=True" "model.fsdp_config.model_dtype=fp32"
  "optim.lr=1e-5" "optim.betas=[0.9,0.999]" "optim.weight_decay=0.01" "optim.clip_grad=1.0"
  "optim.lr_scheduler=wsd" "optim.lr_warmup_steps_ratio=0.125"
  "+optim.v4_warmup_steps=10" "+optim.v4_scheduler=constant"
  "trainer.total_epochs=2" "trainer.total_training_steps=$STEPS" "trainer.save_freq=${V4_SAVE_FREQ:-$STEPS}"
  "trainer.test_freq=-1" "trainer.checkpoint.save_contents=[model]" "trainer.resume_mode=disable"
  "trainer.seed=$SEED" "trainer.default_local_dir=$OUT/train"
  "trainer.project_name=v4-arm-$SFT_ARM" "trainer.experiment_name=$NAME" "trainer.logger=$LOGGER"
  "trainer.nnodes=1" "trainer.n_gpus_per_node=$NGPU")
if [[ "${OFFLOAD:-0}" == 1 ]]; then
  ARGV+=("model.fsdp_config.cpu_offload=True" "model.fsdp_config.offload_params=True")
fi
v4_check_synthetic_data "$TRAIN_FILE" "$DATA_MANIFEST"
if [[ "${V4_CHECKPOINT_STATE:-0}" == 1 ]]; then
  ARGV+=("trainer.checkpoint.save_contents=[model,optimizer,extra]" "trainer.checkpoint.load_contents=[model,optimizer,extra]")
fi
if [[ -n "${V4_RESUME_PATH:-}" ]]; then
  [[ "${V4_RETRY_ADMITTED:-0}" == 1 ]] || { echo 'resume requires reviewed infrastructure admission' >&2; exit 2; }
  ARGV+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=$V4_RESUME_PATH")
fi
if [[ "$DRY_RUN" == 1 ]]; then printf '%s\n' "${ARGV[@]}"; exit 0; fi

[[ ! -e "$OUT" || "${V4_RETRY_ADMITTED:-0}" == 1 ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT/env"
STAGE_FILE="$OUT/env/stage.json"
LAST_STAGE=''
stage() {
  [[ "$1" == exited ]] || LAST_STAGE="$1"
  python - "$STAGE_FILE" "$NAME" "${KIT_P4_JOB:-}" "$1" "${2:-}" "$LAST_STAGE" "$SFT_ARM" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, attempt, job, name, status, last, arm = sys.argv[1:]
doc = json.load(open(path)) if os.path.exists(path) else {"schema": "kit-launcher-stage.v1", "attempt": attempt, "job": job, "arm": arm, "stages": []}
labels = os.path.join(os.path.dirname(path), "data-labels.json")
if os.path.exists(labels): doc.update(json.load(open(labels)))
entry = {"stage": name, "time": datetime.now(timezone.utc).isoformat()}
if status: entry["returncode"] = int(status)
if name == "exited": entry["last_stage"] = last or None
doc["stages"].append(entry)
tmp = path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(doc, f, indent=2); f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
fd = os.open(os.path.dirname(path), os.O_RDONLY); os.fsync(fd); os.close(fd)
PY
}
on_exit() { local status=$1; trap - EXIT; stage exited "$status" || true; exit "$status"; }
trap 'on_exit $?' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
stage started
printf '%s\n' "${ARGV[@]}" > "$OUT/env/argv.txt"
export PYTHONPATH="$KIT/..:$SDPO_DIR:${PYTHONPATH:-}"
export KIT_SFT_ARM_F=1 KIT_SFT_ARM="$SFT_ARM" KIT_SFT_SEED="$SEED" WANDB_MODE=disabled
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"
export USER="${USER:-$(whoami)}" EXPERIMENT="$NAME"
python - "$TRAIN_FILE" "$VAL_FILE" "$DATA_MANIFEST" "$SEED" "$MODEL_DIR" "$SFT_ARM" "$STEPS" "$OUT/env/data-labels.json" <<'PY'
import json, os, sys
from pathlib import Path
import pyarrow.parquet as pq
from kit.v4_teacher import sha
from kit.v4_contract import check_schedule, profile, technical_synthetic
train, val, manifest, seed, model = map(Path, sys.argv[1:6])
arm, steps, labels = sys.argv[6:]
profile_name = profile(steps, os.environ['V4_PROFILE'])
doc = json.loads(manifest.read_text())
if not (doc['schema'] == 'kit-v4-sft-data.v1' and doc['arm'] in (arm, 'COMMON') and doc['response_only'] is True): sys.exit("v4 admission gate failed")
if not (32 * int(steps) <= doc['rows'] <= 1280 and doc['seed'] == int(str(seed))): sys.exit('shared schedule seed mismatch')
if not (doc['train_sha256'] == sha(train.read_bytes()) and doc['val_sha256'] == sha(val.read_bytes())): sys.exit('SFT data changed')
t = pq.read_table(train).to_pylist(); check_schedule(t, int(steps)); v = pq.read_table(val).to_pylist()
if bool(doc.get('technical_synthetic')) != technical_synthetic(t): sys.exit('technical_synthetic manifest/row labels disagree')
if technical_synthetic(t) and doc.get('scientific_phase_allowed') is not False: sys.exit('synthetic manifest cannot claim scientific admission')
if not (len(t) == doc['rows'] and [r['id'] for r in t] == doc['ids']): sys.exit('shared schedule order mismatch')
if not (v == t[:32]): sys.exit('validation must be training-only diagnostics')
if not ((model / 'config.json').is_file()): sys.exit('incoming HF model missing')
if not (arm != 'R' or doc['arm'] == 'COMMON'): sys.exit('R requires the common file with both targets')
if not (profile_name == 'technical-smoke' or (doc['arm'] == 'COMMON' and doc.get('scientific_phase_allowed') is True and not doc.get('stand_in_teacher') and doc['rows'] == 1280)): sys.exit('unqualified or stand-in data is smoke only')
json.dump({'technical_synthetic': doc.get('technical_synthetic', False), 'profile': profile_name, 'technical_smoke': profile_name == 'technical-smoke', 'stand_in_teacher': doc.get('stand_in_teacher', False), 'teacher_models': doc.get('teacher_models', []), 'target_field': 'extra_info.' + ('rewrite' if arm == 'R' else 'demonstration')}, open(labels, 'w'))
PY
git -C "$SDPO_DIR" rev-parse HEAD > "$OUT/env/sdpo-commit.txt"
[[ "$(cat "$OUT/env/sdpo-commit.txt")" == '7c457fc1b1f636ae794eb0362ba37d4743b06fbc' ]] || { echo 'SDPO revision mismatch' >&2; exit 2; }
git -C "$SDPO_DIR" status --short > "$OUT/env/sdpo-dirty.txt"
cp "$DATA_MANIFEST" "$OUT/env/data-manifest.json"
pip freeze > "$OUT/env/pip-freeze.txt" 2>/dev/null || true
nvidia-smi > "$OUT/env/nvidia-smi.txt" 2>/dev/null || true
ls "$MODEL_DIR" > "$OUT/env/model-files.txt"
date -u +%FT%TZ > "$OUT/env/started-at.txt"
python - "$OUT/env/settings.json" "$STEPS" "$NGPU" "$SEED" "$SFT_ARM" "$OUT/env/data-labels.json" <<'PY'
import json, sys
labels = json.load(open(sys.argv[6]))
json.dump({**labels, 'arm': sys.argv[5], 'steps': int(sys.argv[2]),
           'batch_prompts': 32, 'lr': 1e-5, 'lr_source': 'declared choice; pinned SDFT has no SFT baseline LR',
           'optimizer_reset': True, 'betas': [.9,.999], 'weight_decay': .01, 'clip_grad': 1,
           'warmup_steps': 10, 'scheduler': 'warmup then constant through step 40', 'max_input_tokens': 4096, 'max_context_tokens': 6144,
           'max_response_tokens': 2048, 'response_only_loss': True, 'thinking': False, 'sampler_shuffle': False,
           'loss_normalization': 'pinned verl mean over response tokens per microbatch, then mean over microbatches',
           'seed': int(sys.argv[4]), 'n_gpus': int(sys.argv[3])}, open(sys.argv[1], 'w'), indent=2)
PY
STARTED=$(date -u +%s)
cd "$SDPO_DIR"
# Resolve on ONE process, before torchrun can launch any GPU workers.
if ! python "$KIT/sft_entry.py" --config-name sft_trainer --cfg job --resolve "${ARGV[@]:9}" \
  > "$OUT/env/resolved-config.yaml" 2> "$OUT/env/resolved-config.err" || [[ ! -s "$OUT/env/resolved-config.yaml" ]]; then
  echo 'SFT config could not resolve; no trainer invoked' >&2; exit 2
fi
stage config-resolved
stage trainer-invoked
date -u +%FT%TZ > "$OUT/env/trainer-started-at.txt"
set +e
"${ARGV[@]}" 2>&1 | tee "$OUT/console.log"
STATUS=${PIPESTATUS[0]}
set -e
stage trainer-exited "$STATUS"
date -u +%FT%TZ > "$OUT/env/finished-at.txt"
MERGED=0
MERGE_STATUS=-1
CKPT="$OUT/train/global_step_$STEPS"
if [[ "$STATUS" == 0 && -d "$CKPT" ]]; then
  set +e
  MERGE_LAUNCH=(python)
  if [[ "${V4_TELEMETRY:-0}" == 1 ]]; then
    MERGE_LAUNCH=(python "$KIT/v4_timing.py" --out "$OUT/env/merge-timing.json" -- python)
  fi
  "${MERGE_LAUNCH[@]}" -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$OUT/hf-step$STEPS" 2>&1 | tee "$OUT/merge.log"
  MERGE_STATUS=${PIPESTATUS[0]}
  set -e
  if [[ "$MERGE_STATUS" == 0 ]] && python - "$OUT/hf-step$STEPS" <<'PY'
import sys
from kit.v4_sft import complete_export
sys.exit(0 if complete_export(sys.argv[1]) else 1)
PY
  then MERGED=1; fi
fi
stage merged "$MERGE_STATUS"
python - "$OUT" "$NAME" "$STATUS" "$MERGED" "$MERGE_STATUS" "$STEPS" "$NGPU" "$SEED" "$MODEL_DIR" "$TRAIN_FILE" "$(( $(date -u +%s) - STARTED ))" "$SFT_ARM" <<'PY'
import json, sys
from pathlib import Path
out, name, status, merged, merge_status, steps, ngpu, seed, model, train, seconds, arm = sys.argv[1:]
labels = json.load(open(Path(out) / "env/data-labels.json"))
doc = {**labels, 'schema': 'kit-sft-run.v2', 'route': 'sft', 'arm': arm, 'name': name,
       'returncode': int(status), 'merged': int(merged), 'merge_returncode': int(merge_status), 'steps': int(steps),
       'batch_prompts': 32, 'lr': '1e-5', 'seed': int(seed),
       'max_response_length': 2048, 'max_input_length': 4096, 'max_context_length': 6144, 'n_gpus': int(ngpu), 'seconds': int(seconds),
       'model_dir': model, 'train_file': train, 'merged_dir': str(Path(out) / ('hf-step' + steps)),
       'response_only_loss': True, 'optimizer_reset': True}
for filename in ('train-summary.json', 'run-summary.json'):
    json.dump(doc, open(Path(out) / filename, 'w'), indent=2)
PY
python "$KIT/v4_receipts.py" "$OUT" ${SFT_ARM} "$STEPS" "$V4_PROFILE"
if [[ "$STATUS" != 0 ]]; then exit "$STATUS"; fi
[[ "$MERGED" == 1 ]] || { echo 'arm F has no complete merged model' >&2; exit 2; }
