#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/data_processing/_external_arcdev_common.sh"
external_init

DATASETS="${DATASETS:-EHBS,ADHD_Suijing,PKU_ADHD,Olin_ASD_SZ,BLSA,AIBL,Shile_Nanjing,PK_MPRC}"
RECON_ENGINE="${RECON_ENGINE:-${ENGINE:-fastsurfer}}"
GPU_IDS="${GPU_IDS:-0,1}"
THREADS_PER_JOB="${THREADS_PER_JOB:-4}"
LOAD_FREESURFER_MODULE="${LOAD_FREESURFER_MODULE:-1}"
FREESURFER_MODULE="${FREESURFER_MODULE:-freesurfer/7.4.1}"
EXPECTED_FREESURFER_VERSION="${EXPECTED_FREESURFER_VERSION:-7.4.1}"
# Set this environment variable to the FastSurfer checkout on the remote host.
REMOTE_FASTSURFER_HOME="${REMOTE_FASTSURFER_HOME:-}"
REMOTE_FASTSURFER_PYTHON="${REMOTE_FASTSURFER_PYTHON:-${REMOTE_FASTSURFER_HOME:+$REMOTE_FASTSURFER_HOME/.venv/bin/python}}"
if [[ "$LOAD_FREESURFER_MODULE" != "1" && -z "${REMOTE_FREESURFER_HOME:-}" ]]; then
  echo "REMOTE_FREESURFER_HOME is required when LOAD_FREESURFER_MODULE=0" >&2
  exit 2
fi
if [[ "$RECON_ENGINE" == "fastsurfer" ]]; then
  [[ -n "$REMOTE_FASTSURFER_HOME" ]] || {
    echo "REMOTE_FASTSURFER_HOME must point to the remote FastSurfer checkout" >&2
    exit 2
  }
fi
export DATASETS RECON_ENGINE GPU_IDS THREADS_PER_JOB LOAD_FREESURFER_MODULE FREESURFER_MODULE EXPECTED_FREESURFER_VERSION REMOTE_FASTSURFER_HOME REMOTE_FASTSURFER_PYTHON

if [[ "$LOAD_FREESURFER_MODULE" == "1" ]]; then
  echo "remote bootstrap: module load $FREESURFER_MODULE"
fi
echo "required FreeSurfer version: $EXPECTED_FREESURFER_VERSION"
if [[ -n "${REMOTE_FREESURFER_HOME:-}" ]]; then
  echo "FreeSurfer home override: $REMOTE_FREESURFER_HOME"
fi
echo "FastSurfer home: ${REMOTE_FASTSURFER_HOME:-not used}"
echo "FastSurfer Python: ${REMOTE_FASTSURFER_PYTHON:-not used}"

external_deploy
external_start freesurfer
