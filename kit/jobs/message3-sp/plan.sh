#!/usr/bin/env bash
# Build the SP checkpoint inventory and sidecar manifest from the launch manifests (CPU).
# Reads the launch manifests only; never edits them and never submits anything. Always exits 0.
#
# Required: KIT, SP_ROOT, LAUNCH (the message-2 launch manifest JSON).
# Optional: M3_LAUNCH (the K/CR launch manifest JSON), PD_MANIFEST (PD checkpoint/score manifest),
# SP_TIERS (space-separated extra checkpoint lists), SP_IDENTITIES (space-separated parent output
# roots holding scorer identities).
set -uo pipefail
: "${KIT:?}" "${SP_ROOT:?}" "${LAUNCH:?}"
mkdir -p "$SP_ROOT"
launches=(--launch "$LAUNCH")
[[ -n "${M3_LAUNCH:-}" ]] && launches+=(--launch "$M3_LAUNCH")
args=(inventory "${launches[@]}" --out "$SP_ROOT/inventory.json")
[[ -n "${SP_RETURN1:-}" ]] && args+=(--return1 "$SP_RETURN1")
[[ -n "${PD_MANIFEST:-}" ]] && args+=(--pd "$PD_MANIFEST")
for tier in ${SP_TIERS:-}; do args+=(--tier "$tier"); done
for root in ${SP_IDENTITIES:-}; do args+=(--identities "$root"); done
py="${JOB_PY:-python3}"
"$py" "$KIT/sp_manifest.py" "${args[@]}" \
  && "$py" "$KIT/sp_manifest.py" sidecar "${launches[@]}" --inventory "$SP_ROOT/inventory.json" --out "$SP_ROOT/sidecar.json" \
  || echo "SP plan did not complete; message-2 and K/CR jobs are unaffected" >&2
if [[ -n "${SP_GPUS:-}" ]]; then
  hardware=(hardware --gpu-identities "$SP_GPUS" --out "$SP_ROOT/hardware.json")
  for parent in ${SP_PARENT_HARDWARE:-}; do hardware+=(--parent-hardware "$parent"); done
  "$py" "$KIT/sp_manifest.py" "${hardware[@]}" || echo "SP hardware is unverified; parents are unaffected" >&2
fi
if [[ -n "${SP_OWNER_EXPORTS:-}" ]]; then
  "$py" "$KIT/jobs/message3-sp/durable_adapter.py" --sources "$SP_OWNER_EXPORTS" --out "$SP_ROOT" \
    || echo "SP owner adapter incomplete; parents are unaffected" >&2
fi
exit 0
