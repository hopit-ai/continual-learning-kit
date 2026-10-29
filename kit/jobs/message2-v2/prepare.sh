#!/usr/bin/env bash
set -euo pipefail
: "${KIT:?}" "${LOCAL_ROOT:?}" "${DURABLE:?}" "${MANIFEST:?}"
# Submit with --signal=B:TERM@<measured restore+snapshot seconds>.
# exec makes Python the batch process; TERM/KILL and snapshot publication are bounded.
: "${TRAINER_PY:?}" "${INFERENCE_PY:?}"
# First PF-node jobs record both-runtime failures inside their snapshot.
# Later preparation jobs keep the original fail-fast runtime check.
case " $* " in
  *" --task PF-node-"*) ;;
  *) "${JOB_PY:-python3}" "$KIT/m2_preflight.py" --trainer-python "$TRAINER_PY" --inference-python "$INFERENCE_PY" ;;
esac
exec "${JOB_PY:-python3}" "$KIT/m2_node.py" --action prepare \
  --manifest "$MANIFEST" --local-root "$LOCAL_ROOT" --durable "$DURABLE" "$@"
