#!/usr/bin/env bash
set -euo pipefail
: "${KIT:?}" "${LOCAL_ROOT:?}" "${DURABLE:?}" "${MANIFEST:?}"
# Submit with --signal=B:TERM@<measured restore+snapshot seconds>.
# exec makes Python the batch process; TERM/KILL and snapshot publication are bounded.
exec "${JOB_PY:-python3}" "$KIT/m2_node.py" --action train-stage \
  --manifest "$MANIFEST" --local-root "$LOCAL_ROOT" --durable "$DURABLE" "$@"
