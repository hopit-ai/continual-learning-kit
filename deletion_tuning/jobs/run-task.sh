#!/usr/bin/env bash
# Operator-owned sbatch; independent lanes have no cross-seed dependencies.
set -euo pipefail
: "${DELETION_ROOT:?}" "${DELETION_CONFIG:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
row="${1:?deletion row ID}"
case "$row" in
  prepare-*|collect) export CUDA_VISIBLE_DEVICES='' ;;
esac
exec "${JOB_PY:-python3}" -I "$DELETION_ROOT/deletion_tuning/deletion_jobs.py" run \
  --config "$DELETION_CONFIG" --row "$row"
