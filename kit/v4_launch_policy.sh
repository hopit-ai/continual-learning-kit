# Main FinQA checks warn; tagged phase-0 and legacy launchers retain their policy.
v4_validation_stop() {
  if [[ "${V4_FINQA_CAMPAIGN:-0}" == 1 && "${V4_PHASE0_MODE:-0}" != 1 ]]; then
    PYTHONPATH="$KIT/..:${PYTHONPATH:-}" python - "$1" <<'PYTHON'
import os,sys
from kit.v4_finqa_policy import warn
warn('launcher check at line '+sys.argv[1], 'See the preceding diagnostic in this attempt log; continuing with the local configuration. Flag any science change in the return note.',os.environ.get('WORK'))
PYTHON
  else
    exit 2
  fi
}
v4_guard_write() {
  if [[ "${V4_FINQA_CAMPAIGN:-0}" == 1 && "${V4_PHASE0_MODE:-0}" != 1 ]]; then
    PYTHONPATH="$KIT/..:${PYTHONPATH:-}" python - "$1" <<'PYTHON'
import os,sys
from kit.v4_finqa_policy import confine
confine(os.environ['WORK'],sys.argv[1])
PYTHON
  fi
}
