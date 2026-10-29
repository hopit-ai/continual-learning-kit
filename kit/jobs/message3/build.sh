#!/usr/bin/env bash
# Materialise the M3 manifest from settings and prove isolation against the message-2 manifest. CPU only.
set -euo pipefail
: "${KIT:?}" "${M3_OUT:?}" "${M3_SETTINGS:?}" "${M3_MANIFEST:?}"
PY="${JOB_PY:-python3}"
"$PY" "$KIT/m3_manifest.py" --action init --out "$M3_OUT"
"$PY" "$KIT/m3_manifest.py" --action build --settings "$M3_SETTINGS" --output "$M3_MANIFEST"
if [ -n "${M2_MANIFEST:-}" ]; then
  "$PY" "$KIT/m3_manifest.py" --action isolation --manifest "$M3_MANIFEST" --message2-manifest "$M2_MANIFEST"
else
  "$PY" "$KIT/m3_manifest.py" --action isolation --manifest "$M3_MANIFEST"
fi
