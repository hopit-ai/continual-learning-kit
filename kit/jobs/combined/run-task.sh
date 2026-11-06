#!/usr/bin/env bash
# One explicit job. The operator owns sbatch, allocation and all dependencies.
set -euo pipefail
: "${KIT:?}" "${COMBINED_MANIFEST:?}"
TASK="${1:?combined task ID}"
if [ "${UTIL_WATCH_ACTIVE:-}" = 1 ]; then
  exec "${JOB_PY:-python3}" -I "$KIT/combined_send.py" run --manifest "$COMBINED_MANIFEST" --task "$TASK"
fi
UTIL_ROW=$("${JOB_PY:-python3}" -I -c '
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
' "$COMBINED_MANIFEST" "$TASK")
UTIL_GPUS=${UTIL_ROW%%$'\n'*}
UTIL_OUT=${UTIL_ROW#*$'\n'}
UTIL_SCOPE=(--expect-gpus "$UTIL_GPUS")
if [ "$UTIL_GPUS" = 1 ]; then UTIL_SCOPE+=(--holder-slice); fi
export UTIL_JOB="$TASK"
exec "${JOB_PY:-python3}" -I "$KIT/util_watch.py" --out "$UTIL_OUT" "${UTIL_SCOPE[@]}" -- \
  "${JOB_PY:-python3}" -I "$KIT/combined_send.py" run --manifest "$COMBINED_MANIFEST" --task "$TASK"
