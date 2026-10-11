#!/usr/bin/env bash
# Common registered entry for S/F/R/D; legacy launchers retain their historical defaults.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$KIT/v4_launch_policy.sh"
export WORK="${WORK:-$PWD/v4-work}" NGPU="${NGPU:-8}" TP="${TP:-2}" OFFLOAD="${OFFLOAD:-0}"
: "${ARM:?set ARM to S, F, R or D}"
: "${DATA_MANIFEST:?set DATA_MANIFEST to the shared scheduled-data manifest}"
: "${SEED:?set SEED to 101, 102 or 103}"
source "$KIT/v4_profile.sh"
[[ "$SEED" =~ ^10[1-5]$ ]] || { echo 'v4 seeds are 101-105' >&2; v4_validation_stop "$LINENO"; }
export KIT_V4_ARM="$ARM" TEST_FREQ=-1 VAL_BEFORE=0
case "$ARM" in
 S) [[ "${TEACHER_RATE:-0.05}" == 0.05 && "${ROLLOUT_N:-8}" == 8 ]] || { echo "S teacher rate and rollout n are registered at 0.05 and 8" >&2; v4_validation_stop "$LINENO"; }
    launcher=run_sdpo_toolalpaca.sh; export TEACHER_RATE=0.05; : "${REWARD_FILE:?set REWARD_FILE to kit/beds/v4_reward.py}"
    [[ "$REWARD_FILE" == "$KIT/beds/v4_reward.py" ]] || { echo 'S reward file is registered at kit/beds/v4_reward.py' >&2; v4_validation_stop "$LINENO"; } ;;
 D|D-R) launcher=run_sdft.sh; : "${REWARD_FILE:?set REWARD_FILE to kit/beds/v4_reward.py}" ;;
 F|R) launcher=run_sft.sh ;;
 *) echo 'ARM must be S, F, R or D' >&2; v4_validation_stop "$LINENO" ;;
esac
v4_check_synthetic_data "${TRAIN_FILE:-}" "$DATA_MANIFEST"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  export PYTHONPATH="$KIT/..:${PYTHONPATH:-}"
  python "$KIT/v4_run.py" check
fi
set +e
bash "$KIT/$launcher" "$@"
status=$?
set -e
if [[ "${DRY_RUN:-0}" != 1 && -d "$WORK/runs/$NAME/env" ]]; then
  cp "$DATA_MANIFEST" "$WORK/runs/$NAME/env/data-manifest.json"
fi
if [[ "${DRY_RUN:-0}" != 1 && -f "$WORK/runs/$NAME/run-summary.json" ]]; then
  python "$KIT/v4_run.py" summarize
fi
exit "$status"
