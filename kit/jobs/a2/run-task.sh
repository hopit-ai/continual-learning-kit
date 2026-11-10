#!/usr/bin/env bash
set -euo pipefail
: "${KIT:?}" "${A2_CONFIG:?}"
exec "${JOB_PY:-python3}" -I "$KIT/a2_jobs.py" run --config "$A2_CONFIG" --task "${1:?A2 row ID}"
