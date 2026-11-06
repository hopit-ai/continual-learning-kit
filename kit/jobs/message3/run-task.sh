#!/usr/bin/env bash
# One message-3 (K/CR) task from the M3 manifest. The partner submits it with his own sbatch.
# Isolation: writes only under $M3_OUT; exit 0 DONE, 75 PENDING, 1 FAILED. Never touches message-2 state.
set -uo pipefail
: "${KIT:?}" "${M3_OUT:?}" "${M3_MANIFEST:?}"
TASK="${1:?task id, e.g. M3-K-101-finqa-train}"
# Keep the argv array nonempty: bash 3 treats an empty array as unset under -u.
ARGS=(--action run --manifest "$M3_MANIFEST" --task "$TASK" --out "$M3_OUT")
if [ -n "${M3_SUBMITTED_AT:-}" ]; then ARGS+=(--submitted-at "$M3_SUBMITTED_AT"); fi
# The combined wrapper already observes the complete allocation/child lifetime.
if [ "${UTIL_WATCH_ACTIVE:-}" = 1 ]; then
  exec "${JOB_PY:-python3}" -I "$KIT/m3_manifest.py" "${ARGS[@]}" --score-gpu "${SCORE_GPU:-0}"
fi
UTIL_ROW=$("${JOB_PY:-python3}" -I -c 'import json,sys,time;doc=json.load(open(sys.argv[1]));print(next(t["gpus"] for t in doc["tasks"] if t["id"]==sys.argv[2]));print(time.time_ns())' "$M3_MANIFEST" "$TASK") || exit 1
UTIL_GPUS=${UTIL_ROW%%$'\n'*}
UTIL_ATTEMPT=${UTIL_ROW#*$'\n'}
UTIL_SCOPE=(--expect-gpus "$UTIL_GPUS")
if [ "$UTIL_GPUS" = 1 ]; then UTIL_SCOPE+=(--holder-slice); fi
export UTIL_JOB="$TASK"
exec "${JOB_PY:-python3}" -I "$KIT/util_watch.py" --out "$M3_OUT/state/utilisation/$UTIL_ATTEMPT.jsonl" "${UTIL_SCOPE[@]}" -- \
  "${JOB_PY:-python3}" -I "$KIT/m3_manifest.py" "${ARGS[@]}" --score-gpu "${SCORE_GPU:-0}"
