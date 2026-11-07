#!/usr/bin/env bash
# One explicit job. The operator owns sbatch, allocation and all dependencies.
set -euo pipefail
: "${KIT:?}" "${COMBINED_MANIFEST:?}"
TASK="${1:?combined task ID}"
if [ "${UTIL_WATCH_ACTIVE:-}" = 1 ]; then
  exec "${JOB_PY:-python3}" -I "$KIT/combined_send.py" run --manifest "$COMBINED_MANIFEST" --task "$TASK"
fi
if ! UTIL_ROW=$("${JOB_PY:-python3}" -I -c '
import json,sys,uuid
from pathlib import Path
doc=json.loads(Path(sys.argv[1]).read_text());task=next(t for t in doc["tasks"] if t["id"]==sys.argv[2]);s=doc["settings"]
root=Path(s["control_root"])/"utilisation"/task["group"]
path=root/(uuid.uuid4().hex+".jsonl")
root.mkdir(parents=True,exist_ok=True)
if task["gpus"]:
    with path.with_suffix(".started.json").open("x") as stream:json.dump(dict(task=task["id"],gpus=task["gpus"]),stream)
print(task["gpus"])
print(path)
' "$COMBINED_MANIFEST" "$TASK"); then
  echo "WARN watcher setup unavailable; running child unwatched" >&2
  export UTIL_WATCH_ACTIVE=1
  exec "${JOB_PY:-python3}" -I "$KIT/combined_send.py" run --manifest "$COMBINED_MANIFEST" --task "$TASK"
fi
UTIL_GPUS=${UTIL_ROW%%$'\n'*}
UTIL_OUT=${UTIL_ROW#*$'\n'}
if [ "$UTIL_GPUS" != 1 ]; then unset UTIL_HOLDER_SLICE; fi
UTIL_SCOPE=(--expect-gpus "$UTIL_GPUS")
if [ "$UTIL_GPUS" = 1 ] && "${JOB_PY:-python3}" -I -c '
import os,runpy,sys
sys.exit(0 if runpy.run_path(sys.argv[1])["use_holder_slice"](os.environ,1) else 1)
' "$KIT/util_watch.py"; then UTIL_SCOPE+=(--holder-slice); fi
export UTIL_JOB="$TASK"
exec "${JOB_PY:-python3}" -I "$KIT/util_watch.py" --out "$UTIL_OUT" "${UTIL_SCOPE[@]}" -- \
  "${JOB_PY:-python3}" -I "$KIT/combined_send.py" run --manifest "$COMBINED_MANIFEST" --task "$TASK"
