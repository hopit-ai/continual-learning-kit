#!/usr/bin/env bash
# Optional plain launch inside your own allocation. OUT and models are explicit.
set -u
KIT=${KIT:-$(cd "$(dirname "$0")" && pwd)}
: "${OUT:?set OUT}" "${CHEM3A_BASE:?set CHEM3A_BASE}" "${CHEM3A_R1:?set CHEM3A_R1}" "${CHEM3A_R2:?set CHEM3A_R2}"
read -r -a devices <<< "${CHEM3A_GPUS:-0 1 2}"
models=("$CHEM3A_BASE" "$CHEM3A_R1" "$CHEM3A_R2")
lanes=(base r1 r2)
pids=()
for lane in 0 1 2; do
  "${TRAINER_PY:-python3}" "$KIT/p3a_3a.py" --out "$OUT" --lane "${lanes[$lane]}" --model "${models[$lane]}" --gpu "${devices[$lane]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
