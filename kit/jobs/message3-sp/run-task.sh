#!/usr/bin/env bash
# One SP sidecar task (message-3 tier, scoring-only, descriptive).
#
# Submit AFTER its parents with afterany semantics, e.g. --dependency=afterany:<parent job>.
# Never make a message-2 job depend on this script. It always exits 0: a failed or missing input is
# recorded in "$SP_ROOT/state/$TASK.json" (DONE / FAILED / MISSING-INPUT) and blocks nothing.
#
# Required: KIT (the kit directory), SP_ROOT (a directory separate from every message-2 output
# root), SIDECAR (the SP sidecar manifest), TASK (an SP-* task ID).
# Optional: SP_MODELS (local owner durable-export map), SP_HARDWARE (frozen hardware authority),
# SCORE_GPU (legacy only), SP_INPUTS (space-separated parent output roots),
# JOB_PY (default python3). The two local map files stay on the node and are never returned.
set -uo pipefail
: "${KIT:?}" "${SP_ROOT:?}" "${SIDECAR:?}" "${TASK:?}"
args=(run --sidecar "$SIDECAR" --task "$TASK" --kit "$KIT" --sp "$SP_ROOT" --score-gpu "${SCORE_GPU:-0}")
args+=(--models "${SP_MODELS:-$SP_ROOT/models.local.json}")
args+=(--owner-sources "${SP_OWNER_EXPORTS:-$SP_ROOT/owner-exports.local.json}")
args+=(--hardware "${SP_HARDWARE:-$SP_ROOT/hardware.json}")
for parent in ${SP_PARENT_HARDWARE:-}; do args+=(--parent-hardware "$parent"); done
for root in ${SP_INPUTS:-}; do args+=(--input "$root"); done
"${JOB_PY:-python3}" "$KIT/sp_manifest.py" "${args[@]}" || echo "SP sidecar task $TASK did not complete; see $SP_ROOT/state" >&2
exit 0
