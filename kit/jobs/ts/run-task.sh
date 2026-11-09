#!/usr/bin/env bash
# One operator-owned allocation per row. Dependencies return promptly on error.
set -euo pipefail
: "${KIT:?}" "${TS_CONFIG:?}" "${TS_WORK:?}"
row="${1:?TS row ID}"
expected="$("${JOB_PY:-python3}" -I "$KIT/ts_jobs.py" resources --task "$row")"
nonce="$("${JOB_PY:-python3}" -I -c 'import uuid; print(uuid.uuid4().hex)')"
mkdir -p "$TS_WORK/util/$row"
exec "${JOB_PY:-python3}" -I "$KIT/util_watch.py" --out "$TS_WORK/util/$row/$nonce.jsonl" \
  --expect-gpus "$expected" -- "${JOB_PY:-python3}" -I "$KIT/ts_jobs.py" run --config "$TS_CONFIG" --task "$row"
