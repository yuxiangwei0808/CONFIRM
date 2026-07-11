#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/data_processing/_external_arcdev_common.sh"
external_init
DATASETS="${DATASETS:-all}"
export DATASETS
external_deploy
external_start audit
