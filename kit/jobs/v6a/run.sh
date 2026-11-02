#!/usr/bin/env bash
# Operator submits exactly one v6a row; resource/dependency requests stay explicit.
set -euo pipefail
: "${KIT:?}" "${COMBINED_MANIFEST:?}"
TASK="${1:?v6a row ID}"
read -r GROUP GPUS < <("${JOB_PY:-python3}" -I -c '
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d.get("mode")=="v6a", "v6a manifest required"
r=next(r for r in d["tasks"] if r["id"]==sys.argv[2])
print(r["group"],r["gpus"])
' "$COMBINED_MANIFEST" "$TASK")
if [ "${UTIL_WATCH_ACTIVE:-}" = 1 ]; then
  exec "${JOB_PY:-python3}" -I "$KIT/combined_send.py" run \
    --manifest "$COMBINED_MANIFEST" --task "$TASK"
fi
if ! UTIL_OUT=$("${JOB_PY:-python3}" -I -c '
import json,sys,uuid
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text())["settings"]
print(Path(s["control_root"])/"utilisation"/sys.argv[2]/sys.argv[3]/(uuid.uuid4().hex+".jsonl"))
' "$COMBINED_MANIFEST" "$GROUP" "$TASK"); then
  echo "WARN watcher path unavailable; running child unwatched" >&2
  export UTIL_WATCH_ACTIVE=1
  exec "${JOB_PY:-python3}" -I "$KIT/combined_send.py" run \
    --manifest "$COMBINED_MANIFEST" --task "$TASK"
fi
export UTIL_JOB="$TASK"
exec "${JOB_PY:-python3}" -I "$KIT/util_watch.py" --out "$UTIL_OUT" \
  --expect-gpus "$GPUS" -- "${JOB_PY:-python3}" -I "$KIT/combined_send.py" run \
  --manifest "$COMBINED_MANIFEST" --task "$TASK"
