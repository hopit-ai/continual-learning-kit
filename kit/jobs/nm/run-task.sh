#!/usr/bin/env bash
# One right-sized allocation, one watcher/lease; no scheduler submissions.
set -euo pipefail
: "${KIT:?}" "${NM_CONFIG:?}"
exec "${JOB_PY:-python3}" -I "$KIT/nm_jobs.py" run --config "$NM_CONFIG" --task "${1:?NM row ID}"
