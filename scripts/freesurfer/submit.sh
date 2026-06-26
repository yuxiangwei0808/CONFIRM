#!/usr/bin/env bash
# One-command local driver: build list -> run recon-all/FastSurfer on this
# machine -> aggregate finished outputs. No SLURM/sbatch dependency.
# Usage:   bash submit.sh <COHORT> [more cohorts...]
# Knobs:   LOCAL_JOBS=2      bash submit.sh Shile_Nanjing      (default 1)
#          RECON=fastsurfer  bash submit.sh Shile_Nanjing      (default recon-all)
#          NImax=10          bash submit.sh Shile_Nanjing      (cap #subjects, for a test run)
#          START_INDEX=101   bash submit.sh Shile_Nanjing      (resume from list row)
#          THREADS=4         bash submit.sh Shile_Nanjing      (threads per subject)
#          OUT_DIR=/scratch/ywei/fs bash submit.sh Shile_Nanjing
set -euo pipefail
cd "$(dirname "$0")"

[ "$#" -gt 0 ] || { echo "usage: bash submit.sh <COHORT> [more cohorts...]"; exit 1; }

LOCAL_JOBS="${LOCAL_JOBS:-1}"
START_INDEX="${START_INDEX:-1}"
THREADS_WAS_SET="${THREADS+x}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-2}}"
RECON="${RECON:-recon-all}"
export THREADS RECON
if [ -n "${OUT_DIR:-}" ] && [ -n "${CONFIRM_SUBJECTS_DIR:-}" ]; then
  echo "ERROR: set only one of OUT_DIR or CONFIRM_SUBJECTS_DIR"
  exit 1
fi
if [ -n "${OUT_DIR:-}" ]; then
  export OUT_DIR
fi

case "$LOCAL_JOBS" in (*[!0-9]*|"") echo "ERROR: LOCAL_JOBS must be a positive integer"; exit 1 ;; esac
case "$START_INDEX" in (*[!0-9]*|"") echo "ERROR: START_INDEX must be a positive integer"; exit 1 ;; esac
[ "$LOCAL_JOBS" -ge 1 ] || { echo "ERROR: LOCAL_JOBS must be >= 1"; exit 1; }
[ "$START_INDEX" -ge 1 ] || { echo "ERROR: START_INDEX must be >= 1"; exit 1; }

if [ -n "${MAX_CONCURRENT:-}" ]; then
  echo "NOTE: MAX_CONCURRENT is ignored by local submit.sh; use LOCAL_JOBS instead."
fi
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ -z "$THREADS_WAS_SET" ]; then
  echo "NOTE: SLURM_CPUS_PER_TASK is deprecated for local submit.sh; use THREADS instead."
fi

wait_for_batch() {
  local failed=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

for COHORT in "$@"; do
  echo "=== $COHORT ==="
  bash 01_make_subject_list.sh "$COHORT"
  LIST="lists/${COHORT}_subjects.tsv"
  N=$(wc -l < "$LIST" | tr -d ' ')
  [ "$N" -gt 0 ] || { echo "  skip — no subjects"; continue; }
  [ -n "${NImax:-}" ] && [ "$N" -gt "$NImax" ] && N="$NImax"   # optional test cap
  [ "$START_INDEX" -le "$N" ] || { echo "  skip — START_INDEX=$START_INDEX > N=$N"; continue; }

  echo "  local run: rows ${START_INDEX}-${N}, LOCAL_JOBS=$LOCAL_JOBS, RECON=$RECON, threads/subject=$THREADS"
  if [ -n "${OUT_DIR:-${CONFIRM_SUBJECTS_DIR:-}}" ]; then
    echo "  output root override: ${OUT_DIR:-$CONFIRM_SUBJECTS_DIR}"
  fi
  echo "  logs stream to this terminal; use tmux/screen/nohup for long runs."

  pids=()
  active=0
  for TASK in $(seq "$START_INDEX" "$N"); do
    echo "  launch $COHORT row $TASK/$N"
    TASK_ID="$TASK" bash 02_reconall.slurm "$COHORT" &
    pids+=("$!")
    active=$((active + 1))
    if [ "$active" -ge "$LOCAL_JOBS" ]; then
      wait_for_batch "${pids[@]}" || { echo "ERROR: one or more recon jobs failed for $COHORT"; exit 1; }
      pids=()
      active=0
    fi
  done
  if [ "$active" -gt 0 ]; then
    wait_for_batch "${pids[@]}" || { echo "ERROR: one or more recon jobs failed for $COHORT"; exit 1; }
  fi

  echo "  aggregate $COHORT"
  bash 03_aggregate.slurm "$COHORT"
done
