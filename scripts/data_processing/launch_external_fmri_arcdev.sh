#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/data_processing/_external_arcdev_common.sh"
external_init
DATASETS="${DATASETS:-EHBS,LA5C,ADHD_Suijing,PKU_ADHD,Olin_ASD_SZ,BLSA,Shile_Nanjing,PK_MPRC}"
MAX_WORKERS="${MAX_WORKERS:-8}"
export DATASETS MAX_WORKERS
external_deploy
external_start fmri
