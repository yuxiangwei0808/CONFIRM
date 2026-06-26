#!/usr/bin/env bash
# 02 — Run FreeSurfer recon-all (or FastSurfer) for one row of a cohort subject list.
# `submit.sh` runs this locally by setting TASK_ID for each row.
# Manual example:
#   TASK_ID=1 THREADS=2 OUT_DIR=/scratch/ywei/fs bash 02_reconall.slurm <COHORT>
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

COHORT="${1:?usage: TASK_ID=<row> bash 02_reconall.slurm <COHORT>}"
LIST="lists/${COHORT}_subjects.tsv"
[ -f "$LIST" ] || { echo "missing $LIST — run 01_make_subject_list.sh first"; exit 1; }
if [ -n "${OUT_DIR:-}" ] && [ -n "${CONFIRM_SUBJECTS_DIR:-}" ]; then
  echo "ERROR: set only one of OUT_DIR or CONFIRM_SUBJECTS_DIR"
  exit 1
fi
REQUESTED_SUBJECTS_DIR="${OUT_DIR:-${CONFIRM_SUBJECTS_DIR:-$PWD/freesurfer/${COHORT}}}"

# ---- FreeSurfer on arcdev (7.4.1 default; override with FREESURFER_MODULE) ----
if [ -n "${FREESURFER_HOME:-}" ] && [ -x "$FREESURFER_HOME/bin/recon-all" ]; then
  export PATH="$FREESURFER_HOME/bin:$PATH"
else
  FREESURFER_MODULE="${FREESURFER_MODULE:-freesurfer/7.4.1}"
  module load "$FREESURFER_MODULE" 2>/dev/null || true
fi
# source "$FREESURFER_HOME/SetUpFreeSurfer.sh" 2>/dev/null || true
command -v recon-all >/dev/null || { echo "ERROR: recon-all not on PATH — fix the module/source line above"; exit 1; }
if [ -z "${FS_LICENSE:-}" ] && [ -n "${FREESURFER_HOME:-}" ] && [ -f "$FREESURFER_HOME/license.txt" ]; then
  export FS_LICENSE="$FREESURFER_HOME/license.txt"
fi

# Output root (one SUBJECTS_DIR per cohort). Set this after module load because
# the FreeSurfer module exports its own read-only SUBJECTS_DIR. Do not inherit
# SUBJECTS_DIR from the shell; use OUT_DIR for an explicit override.
export SUBJECTS_DIR="$REQUESTED_SUBJECTS_DIR"
mkdir -p "$SUBJECTS_DIR"

# Which engine: recon-all (default, portable) or fastsurfer (faster, needs GPU + FASTSURFER_HOME)
RECON="${RECON:-recon-all}"

TASK="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-1}}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-2}}"
LINE=$(sed -n "${TASK}p" "$LIST")
SID=$(printf '%s' "$LINE" | cut -f1)
T1=$(printf '%s' "$LINE" | cut -f2)
[ -n "$SID" ] && [ -n "$T1" ] || { echo "no subject for array index $TASK"; exit 0; }

# Idempotent: skip if already finished.
if [ -f "$SUBJECTS_DIR/$SID/stats/aseg.stats" ]; then
  echo "[$SID] already complete — skipping"; exit 0
fi
# Clear a half-finished run so recon-all can restart cleanly.
[ -d "$SUBJECTS_DIR/$SID" ] && rm -rf "$SUBJECTS_DIR/$SID"

echo "[$SID] SUBJECTS_DIR=$SUBJECTS_DIR"
echo "[$SID] FREESURFER_HOME=${FREESURFER_HOME:-unset}"
echo "[$SID] FS_LICENSE=${FS_LICENSE:-unset}"
echo "[$SID] $RECON  <-  $T1"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[$SID] DRY_RUN=1; not running $RECON"
  exit 0
fi
if [ "$RECON" = "fastsurfer" ]; then
  # FastSurfer alternative (GPU). Needs FASTSURFER_HOME set.
  "$FASTSURFER_HOME/run_fastsurfer.sh" --t1 "$T1" --sid "$SID" --sd "$SUBJECTS_DIR" \
    --threads "$THREADS"
else
  recon-all -subjid "$SID" -i "$T1" -all -sd "$SUBJECTS_DIR" -no-isrunning \
    -threads "$THREADS"
fi
echo "[$SID] done"
