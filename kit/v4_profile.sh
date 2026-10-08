# Registered v4 recipe. Source before defaults so explicit drift cannot be hidden.
export V4_PROFILE="${V4_PROFILE:-scientific}"
export SEED="${SEED:-101}"
[[ "$SEED" == 101 || "$SEED" == 102 || "$SEED" == 103 ]] || { echo "registered v4 seeds are 101, 102, 103" >&2; exit 2; }
export STEPS="${STEPS:-40}" MAX_RESPONSE="${MAX_RESPONSE:-2048}"
[[ "$V4_PROFILE" == scientific || "$V4_PROFILE" == technical-smoke ]] || { echo 'unknown v4 profile' >&2; exit 2; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] && (( STEPS <= 40 )) || { echo 'v4 profile requires steps 1..40' >&2; exit 2; }
[[ "$V4_PROFILE" != scientific || "$STEPS" == 40 ]] || { echo 'scientific profile requires 40 steps; select technical-smoke profile explicitly' >&2; exit 2; }
for pair in MAX_RESPONSE:2048 LR:1e-5 WARMUP_STEPS:10 WEIGHT_DECAY:0.01 LR_SCHEDULER:constant TEST_FREQ:-1 VAL_BEFORE:0 FINISH_GATE:1 FEEDBACK:0 SOFT:0 SHUFFLE:0; do
  var="${pair%%:*}"; value="${pair#*:}"
  [[ -z "${!var:-}" || "${!var}" == "$value" ]] || { echo "$var is registered at $value for v4" >&2; exit 2; }
  export "$var=$value"
done
export KIT_FINISH_GATE=1 SHUFFLE=0

# Fail closed on gold-derived smoke targets even when only printing commands.
v4_check_synthetic_data() {
  [[ "$V4_PROFILE" != technical-smoke || "$STEPS" == 40 ]] || return 0
  local v4_existing_input=0 v4_input
  for v4_input in "$@"; do [[ ! -f "$v4_input" ]] || v4_existing_input=1; done
  [[ "$v4_existing_input" == 1 ]] || return 0
  PYTHONPATH="$KIT/..:${PYTHONPATH:-}" python - "$@" <<'PYTHON'
import json, os, sys
from pathlib import Path
from kit.v4_contract import check_synthetic_rows
values = []
for name in sys.argv[1:]:
    path = Path(name)
    if not path.is_file(): continue  # historical dry plans can name future inputs
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        values.extend(pq.read_table(path).to_pylist())
    elif path.suffix == '.json': values.append(json.loads(path.read_text()))
check_synthetic_rows(values, os.environ['STEPS'], os.environ['V4_PROFILE'])
PYTHON
}
