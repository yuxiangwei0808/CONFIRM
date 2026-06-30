#!/usr/bin/env bash
# 03 — Aggregate finished recon-all outputs into per-cohort tables CONFIRM ingests.
# Runs automatically after local `submit.sh`, or by hand:
#   bash 03_aggregate.slurm <COHORT>
set -euo pipefail
cd "$(dirname "$0")"

COHORT="${1:?usage: 03_aggregate.slurm <COHORT>}"
OUT="tables/${COHORT}"; mkdir -p "$OUT" logs
if [ -n "${OUT_DIR:-}" ] && [ -n "${CONFIRM_SUBJECTS_DIR:-}" ]; then
  echo "ERROR: set only one of OUT_DIR or CONFIRM_SUBJECTS_DIR"
  exit 1
fi
REQUESTED_SUBJECTS_DIR="${OUT_DIR:-${CONFIRM_SUBJECTS_DIR:-$PWD/freesurfer/${COHORT}}}"

# FreeSurfer on arcdev (7.4.1 default; override with FREESURFER_MODULE).
if [ -n "${FREESURFER_HOME:-}" ] && [ -x "$FREESURFER_HOME/bin/asegstats2table" ]; then
  export PATH="$FREESURFER_HOME/bin:$PATH"
else
  FREESURFER_MODULE="${FREESURFER_MODULE:-freesurfer/7.4.1}"
  module load "$FREESURFER_MODULE" 2>/dev/null || true
fi
# source "$FREESURFER_HOME/SetUpFreeSurfer.sh" 2>/dev/null || true
command -v asegstats2table >/dev/null || { echo "ERROR: asegstats2table not on PATH"; exit 1; }

# Set after module load because the FreeSurfer module exports its own read-only
# SUBJECTS_DIR. Do not inherit SUBJECTS_DIR from the shell; use OUT_DIR for an
# explicit override.
export SUBJECTS_DIR="$REQUESTED_SUBJECTS_DIR"
echo "[$COHORT] SUBJECTS_DIR=$SUBJECTS_DIR"

# Subjects whose recon-all actually finished (aseg.stats present).
COMPLETED="$OUT/completed_subjects.txt"
: > "$COMPLETED"
for d in "$SUBJECTS_DIR"/*/; do
  s=$(basename "$d")
  [ -f "$SUBJECTS_DIR/$s/stats/aseg.stats" ] && echo "$s" >> "$COMPLETED"
done
NC=$(wc -l < "$COMPLETED" | tr -d ' ')
echo "[$COHORT] $NC subjects completed -> $COMPLETED"
[ "$NC" -gt 0 ] || { echo "nothing to aggregate yet"; exit 0; }

mapfile -t SUBJS < "$COMPLETED"

# Subcortical volumes (includes Left/Right Hippocampus, Amygdala, Thalamus-Proper, Caudate,
# Putamen, Pallidum, Accumbens-area, Lateral-Ventricle, and EstimatedTotalIntraCranialVol).
asegstats2table --subjects "${SUBJS[@]}" --meas volume --tablefile "$OUT/aseg_volume.tsv" --skip

# Cortical thickness per hemisphere (Desikan-Killiany; includes entorhinal, fusiform, etc.).
aparcstats2table --subjects "${SUBJS[@]}" --hemi lh --meas thickness --tablefile "$OUT/aparc_lh_thickness.tsv" --skip
aparcstats2table --subjects "${SUBJS[@]}" --hemi rh --meas thickness --tablefile "$OUT/aparc_rh_thickness.tsv" --skip

echo "[$COHORT] wrote:"
ls -la "$OUT"/*.tsv
echo "Send the whole $OUT/ directory back for ingestion."
