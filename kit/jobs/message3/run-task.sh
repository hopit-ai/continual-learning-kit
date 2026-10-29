#!/usr/bin/env bash
# One message-3 (K/CR) task from the M3 manifest. The partner submits it with his own sbatch.
# Isolation: writes only under $M3_OUT; exit 0 DONE, 75 PENDING, 1 FAILED. Never touches message-2 state.
set -uo pipefail
: "${KIT:?}" "${M3_OUT:?}" "${M3_MANIFEST:?}"
TASK="${1:?task id, e.g. M3-K-101-finqa-train}"
# Keep the argv array nonempty: bash 3 treats an empty array as unset under -u.
ARGS=(--action run --manifest "$M3_MANIFEST" --task "$TASK" --out "$M3_OUT")
if [ -n "${M3_SUBMITTED_AT:-}" ]; then ARGS+=(--submitted-at "$M3_SUBMITTED_AT"); fi
exec "${JOB_PY:-python3}" "$KIT/m3_manifest.py" "${ARGS[@]}" --score-gpu "${SCORE_GPU:-0}"
