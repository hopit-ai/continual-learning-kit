#!/usr/bin/env bash
# The operator supplies sbatch. No submission or scheduler polling happens here.
set -euo pipefail
: "${KIT:?}" "${TRACE_CONFIG:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec "${JOB_PY:-python3}" -I "$KIT/trace_jobs.py" run \
  --config "$TRACE_CONFIG" --row "${1:?TRACE row ID}"
