#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?usage: run_external_data_remote.sh audit|fmri|freesurfer}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/users1/ywei/confirm_external_prep}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-$REMOTE_ROOT/runs/$RUN_ID}"
REMOTE_CODE_DIR="${REMOTE_CODE_DIR:-$REMOTE_RUN_DIR/code}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/users/ywei13/.conda/envs/playground/bin/python}"
DATASETS="${DATASETS:-all}"
FOLLOW="${FOLLOW:-1}"
LOG="$REMOTE_RUN_DIR/logs/${STAGE}.log"

mkdir -p "$REMOTE_RUN_DIR/logs"

if [[ "${REMOTE_BACKGROUND_CHILD:-0}" != "1" ]]; then
  if [[ -s "$LOG" && "${ALLOW_RUN_ID_REUSE:-0}" != "1" ]]; then
    echo "RUN_ID=$RUN_ID already has a $STAGE log; use a new RUN_ID or set ALLOW_RUN_ID_REUSE=1 explicitly" >&2
    exit 2
  fi
  exec 9>"$REMOTE_RUN_DIR/logs/${STAGE}.lock"
  if ! flock -n 9; then
    echo "RUN_ID=$RUN_ID already has an active $STAGE worker" >&2
    exit 2
  fi
  nohup env REMOTE_BACKGROUND_CHILD=1 "$0" "$STAGE" >"$LOG" 2>&1 </dev/null &
  PID=$!
  printf 'started stage=%s run_id=%s pid=%s\n' "$STAGE" "$RUN_ID" "$PID"
  printf 'log: %s:%s\n' "$(hostname)" "$LOG"
  printf 'monitor: ssh %s tail -f %q\n' "${SSH_HOST:-${ARCDEV_HOST:-arcdev}}" "$LOG"
  if [[ "$FOLLOW" == "1" ]]; then
    tail -n 20 -F "$LOG" &
    TAIL_PID=$!
    while kill -0 "$PID" 2>/dev/null; do sleep 5; done
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$PID"
  fi
  exit 0
fi

cd "$REMOTE_CODE_DIR"
export PYTHONPATH="$REMOTE_CODE_DIR/src:$REMOTE_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

CONFIG="$REMOTE_CODE_DIR/configs/external_datasets.yml"
SUBJECTS_ROOT="$REMOTE_ROOT/subjects"

load_freesurfer_module() {
  local module_name="${FREESURFER_MODULE:-freesurfer/7.4.1}"
  local init_script

  if [[ "${LOAD_FREESURFER_MODULE:-1}" != "1" ]]; then
    return
  fi

  if [[ -z "${MODULEPATH:-}" && -r /etc/profile ]]; then
    set +u
    # A direct SSH login receives the site module paths from /etc/profile.
    source /etc/profile
    set -u
  fi

  if ! type module >/dev/null 2>&1; then
    for init_script in \
      /etc/profile.d/lmod.sh \
      /etc/profile.d/modules.sh \
      /usr/share/lmod/lmod/init/bash
    do
      if [[ -r "$init_script" ]]; then
        # shellcheck disable=SC1090
        set +u
        source "$init_script"
        set -u
        break
      fi
    done
  fi

  if ! type module >/dev/null 2>&1; then
    echo "The remote module command is unavailable; set LOAD_FREESURFER_MODULE=0 and REMOTE_FREESURFER_HOME explicitly" >&2
    exit 2
  fi

  echo "loading FreeSurfer environment: module load $module_name"
  set +u
  module load "$module_name"
  set -u
}

case "$STAGE" in
  audit)
    "$REMOTE_PYTHON" -u -m nbs_data.prepare_external_evidence audit \
      --config "$CONFIG" \
      --datasets "$DATASETS" \
      --out-root "$REMOTE_RUN_DIR"
    "$REMOTE_PYTHON" -u -m nbs_data.prepare_external_evidence manifest \
      --config "$CONFIG" \
      --datasets "$DATASETS" \
      --out-root "$REMOTE_RUN_DIR"
    ;;
  fmri)
    "$REMOTE_PYTHON" -u -m nbs_data.prepare_external_evidence fmri \
      --config "$CONFIG" \
      --datasets "$DATASETS" \
      --out-root "$REMOTE_RUN_DIR" \
      --max-workers "${MAX_WORKERS:-8}"
    ;;
  freesurfer)
    load_freesurfer_module
    FREESURFER_HOME_TO_USE="${REMOTE_FREESURFER_HOME:-${FREESURFER_HOME:-}}"
    [[ -n "$FREESURFER_HOME_TO_USE" ]] || {
      echo "FreeSurfer home was not set by the module and REMOTE_FREESURFER_HOME was not supplied" >&2
      exit 2
    }
    if [[ "${RECON_ENGINE:-fastsurfer}" == "fastsurfer" ]]; then
      [[ -n "${REMOTE_FASTSURFER_HOME:-}" ]] || { echo "REMOTE_FASTSURFER_HOME is required"; exit 2; }
      [[ -n "${REMOTE_FASTSURFER_PYTHON:-}" ]] || { echo "REMOTE_FASTSURFER_PYTHON is required"; exit 2; }
    fi
    echo "FreeSurfer home used for version validation: $FREESURFER_HOME_TO_USE"
    "$REMOTE_PYTHON" -u -m nbs_data.prepare_external_evidence manifest \
      --config "$CONFIG" \
      --datasets "$DATASETS" \
      --out-root "$REMOTE_RUN_DIR"

    limit_args=()
    retry_args=()
    fs_license_args=()
    fastsurfer_args=()
    if [[ -n "${LIMIT:-}" ]]; then limit_args=(--limit "$LIMIT"); fi
    if [[ "${RETRY_FAILED:-0}" == "1" ]]; then retry_args=(--retry-failed); fi
    if [[ -n "${REMOTE_FS_LICENSE:-}" ]]; then fs_license_args=(--fs-license "$REMOTE_FS_LICENSE"); fi
    if [[ -n "${REMOTE_FASTSURFER_HOME:-}" ]]; then
      fastsurfer_args=(
        --fastsurfer-home "$REMOTE_FASTSURFER_HOME"
        --fastsurfer-python "$REMOTE_FASTSURFER_PYTHON"
      )
    fi

    "$REMOTE_PYTHON" -u -m nbs_data.run_external_freesurfer \
      --config "$CONFIG" \
      --datasets "$DATASETS" \
      --out-root "$REMOTE_RUN_DIR" \
      --subjects-root "$SUBJECTS_ROOT" \
      --engine "${RECON_ENGINE:-fastsurfer}" \
      --freesurfer-home "$FREESURFER_HOME_TO_USE" \
      --expected-freesurfer-version "${EXPECTED_FREESURFER_VERSION:-7.4.1}" \
      "${fastsurfer_args[@]}" \
      "${fs_license_args[@]}" \
      --gpu-ids "${GPU_IDS:-0,1}" \
      --threads "${THREADS_PER_JOB:-4}" \
      "${limit_args[@]}" \
      "${retry_args[@]}"

    "$REMOTE_PYTHON" -u -m nbs_data.prepare_external_evidence aggregate \
      --config "$CONFIG" \
      --datasets "$DATASETS" \
      --out-root "$REMOTE_RUN_DIR" \
      --subjects-root "$SUBJECTS_ROOT"
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

echo "completed stage=$STAGE run_id=$RUN_ID"
