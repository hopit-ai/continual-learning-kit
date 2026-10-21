#!/usr/bin/env bash
# Entire payload runs inside the combined message's run_stage process group.
set -euo pipefail
: "${PARENT_OUT:?}" "${DURABLE_ROOT:?}" "${KIT:?}" "${BASE:?}"
: "${B_SOURCE:?}" "${B_REPLAY:?}" "${SDPO:?}" "${SCORE_GPU:?}"
: "${TRAINER_PY:?}" "${INFERENCE_PY:?}" "${FOUR_A_SLICE:?}"
: "${FOUR_A_TIMING:?}" "${FOUR_A_STAGE_GPU_HOURS:?}"
A4_OUT="$PARENT_OUT/A4"
"$TRAINER_PY" "$KIT/a4_allocation.py" --out "$A4_OUT" --start \
  --parent-out "$PARENT_OUT" --durable "$DURABLE_ROOT"
"$INFERENCE_PY" "$KIT/a4_bind.py" --out "$A4_OUT" \
  --b-source "$B_SOURCE" --slice "$FOUR_A_SLICE" --base "$BASE" \
  --replay "$B_REPLAY" --gpu "$SCORE_GPU" \
  --timing-receipt "$FOUR_A_TIMING" --stage-gpu-hours "$FOUR_A_STAGE_GPU_HOURS" \
  --trainer-python "$TRAINER_PY" \
  --allocation-gpus "${FOUR_A_ALLOCATION_GPUS:-8}" \
  --score-max-seconds "${FOUR_A_SCORE_MAX_SECONDS:-600}" \
  --nll-max-seconds "${FOUR_A_NLL_MAX_SECONDS:-600}" \
  --handoff-seconds "${FOUR_A_HANDOFF_SECONDS:-300}"
for month in 1 2 3 4 5 6; do
  "$TRAINER_PY" "$KIT/a4_run.py" --out "$A4_OUT" --through-month "$month" \
    --score-gpu "$SCORE_GPU" --gpu 0,1,2,3,4,5,6,7 --tokenizer "$BASE" \
    --sdpo "$SDPO" --trainer-python "$TRAINER_PY" --inference-python "$INFERENCE_PY"
  "$TRAINER_PY" "$KIT/a4_handoff.py" --out "$A4_OUT" --parent-out "$PARENT_OUT" --durable "$DURABLE_ROOT" --trainer-python "$TRAINER_PY"
done
"$TRAINER_PY" "$KIT/a4_read.py" --out "$A4_OUT"
"$TRAINER_PY" "$KIT/a4_handoff.py" --out "$A4_OUT" --parent-out "$PARENT_OUT" --durable "$DURABLE_ROOT" --trainer-python "$TRAINER_PY"
committed=$("$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action path)
"$TRAINER_PY" "$KIT/a4_collect.py" --out "$A4_OUT" --durable-root "$committed/A4" --archive "$PARENT_OUT/4a-return.tar.gz"
"$TRAINER_PY" "$KIT/a4_read.py" --out "$A4_OUT" --source "$PARENT_OUT/4a-return.tar.gz"
"$TRAINER_PY" "$KIT/a4_handoff.py" --out "$A4_OUT" --parent-out "$PARENT_OUT" --durable "$DURABLE_ROOT" --trainer-python "$TRAINER_PY"
