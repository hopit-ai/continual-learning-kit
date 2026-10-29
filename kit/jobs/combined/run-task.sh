#!/usr/bin/env bash
# One explicit job. The operator owns sbatch, allocation and all dependencies.
set -euo pipefail
: "${KIT:?}" "${COMBINED_MANIFEST:?}"
exec "${JOB_PY:-python3}" "$KIT/combined_send.py" run --manifest "$COMBINED_MANIFEST" --task "${1:?combined task ID}"
