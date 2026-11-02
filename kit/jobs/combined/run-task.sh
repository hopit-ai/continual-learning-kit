#!/usr/bin/env bash
# One explicit job. The operator owns sbatch, allocation and all dependencies.
set -euo pipefail
: "${KIT:?}" "${COMBINED_MANIFEST:?}"
TASK="${1:?combined task ID}"
IFS=$'\t' read -r UTIL_OUT EXPECT_GPUS < <("${JOB_PY:-python3}" -I -c '
import json,sys,time
from pathlib import Path
doc=json.loads(Path(sys.argv[1]).read_text());task=next(t for t in doc["tasks"] if t["id"]==sys.argv[2]);s=doc["settings"]
group=task["group"]
if group=="core":root=Path(s["core_local_root"])/"execution"
elif group=="kcr":root=Path(s["kcr_root"])/"state"
elif group=="pd":root=Path(s["pd_root"])/("seed-"+str(task.get("seed","")))
elif group=="sp":root=Path(s["sp_root"])/"state"
else:root=Path(s["control_root"])/group
print(str(root/"utilisation"/(str(time.time_ns())+".jsonl"))+"\t"+str(task.get("gpus",0)))
' "$COMBINED_MANIFEST" "$TASK")
if [ "${UTIL_WATCH_ACTIVE:-}" = 1 ]; then
  exec "${JOB_PY:-python3}" -I "$KIT/combined_send.py" run --manifest "$COMBINED_MANIFEST" --task "$TASK"
fi
export UTIL_JOB="$TASK"
exec "${JOB_PY:-python3}" -I "$KIT/util_watch.py" --out "$UTIL_OUT" --expect-gpus "$EXPECT_GPUS" -- \
  "${JOB_PY:-python3}" -I "$KIT/combined_send.py" run --manifest "$COMBINED_MANIFEST" --task "$TASK"
