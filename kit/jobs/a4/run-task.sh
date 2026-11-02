#!/usr/bin/env bash
# Equivalent to the combined wrapper, with the same watcher/lease lifetime.
set -euo pipefail
: "${KIT:?}" "${A4_CONFIG:?}"
exec "${JOB_PY:-python3}" -I "$KIT/a4_jobs.py" run --config "$A4_CONFIG" --task "${1:?4A row ID}"
