#!/usr/bin/env bash
# Arm R uses the exact F launcher/configuration and common parquet, selecting the rewrite.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SFT_ARM=R
exec bash "$KIT/run_sft_f.sh" "$@"
