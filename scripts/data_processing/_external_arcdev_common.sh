#!/usr/bin/env bash

external_init() {
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  SSH_HOST="${SSH_HOST:-${ARCDEV_HOST:-arcdev}}"
  ARCDEV_HOST="$SSH_HOST"
  : "${REMOTE_ROOT:?Set REMOTE_ROOT to the writable lab run directory}"
  : "${REMOTE_PYTHON:?Set REMOTE_PYTHON to the Python executable on the remote host}"
  EXTERNAL_DATASET_CONFIG="${EXTERNAL_DATASET_CONFIG:-configs/external_datasets.local.yml}"
  RUN_ID="${RUN_ID:-external-$(date +%Y%m%dT%H%M%S)}"
  REMOTE_RUN_DIR="$REMOTE_ROOT/runs/$RUN_ID"
  REMOTE_CODE_DIR="$REMOTE_RUN_DIR/code"
  FOLLOW="${FOLLOW:-1}"
  DEPLOY="${DEPLOY:-1}"
  export REPO_ROOT SSH_HOST ARCDEV_HOST REMOTE_ROOT REMOTE_PYTHON EXTERNAL_DATASET_CONFIG RUN_ID REMOTE_RUN_DIR REMOTE_CODE_DIR FOLLOW DEPLOY
}

external_deploy() {
  if [[ "$DEPLOY" != "1" ]]; then
    return
  fi
  ssh "$SSH_HOST" mkdir -p "$REMOTE_CODE_DIR"
  rsync -a \
    "$REPO_ROOT/src" \
    "$REPO_ROOT/nbs_data" \
    "$REPO_ROOT/configs" \
    "$REPO_ROOT/scripts" \
    "$REPO_ROOT/pyproject.toml" \
    "$SSH_HOST:$REMOTE_CODE_DIR/"
}

external_start() {
  local stage="${1:?stage is required}"
  ssh "$SSH_HOST" env \
    SSH_HOST="$SSH_HOST" \
    ARCDEV_HOST="$SSH_HOST" \
    RUN_ID="$RUN_ID" \
    REMOTE_ROOT="$REMOTE_ROOT" \
    REMOTE_RUN_DIR="$REMOTE_RUN_DIR" \
    REMOTE_CODE_DIR="$REMOTE_CODE_DIR" \
    REMOTE_PYTHON="$REMOTE_PYTHON" \
    EXTERNAL_DATASET_CONFIG="$EXTERNAL_DATASET_CONFIG" \
    DATASETS="${DATASETS:-all}" \
    MAX_WORKERS="${MAX_WORKERS:-8}" \
    FOLLOW="$FOLLOW" \
    RECON_ENGINE="${RECON_ENGINE:-${ENGINE:-fastsurfer}}" \
    GPU_IDS="${GPU_IDS:-0,1}" \
    THREADS_PER_JOB="${THREADS_PER_JOB:-4}" \
    LIMIT="${LIMIT:-}" \
    RETRY_FAILED="${RETRY_FAILED:-0}" \
    ALLOW_RUN_ID_REUSE="${ALLOW_RUN_ID_REUSE:-0}" \
    LOAD_FREESURFER_MODULE="${LOAD_FREESURFER_MODULE:-1}" \
    FREESURFER_MODULE="${FREESURFER_MODULE:-freesurfer/7.4.1}" \
    EXPECTED_FREESURFER_VERSION="${EXPECTED_FREESURFER_VERSION:-7.4.1}" \
    REMOTE_FREESURFER_HOME="${REMOTE_FREESURFER_HOME:-}" \
    REMOTE_FASTSURFER_HOME="${REMOTE_FASTSURFER_HOME:-}" \
    REMOTE_FASTSURFER_PYTHON="${REMOTE_FASTSURFER_PYTHON:-}" \
    REMOTE_FS_LICENSE="${REMOTE_FS_LICENSE:-}" \
    bash -l "$REMOTE_CODE_DIR/scripts/data_processing/run_external_data_remote.sh" "$stage"
}
