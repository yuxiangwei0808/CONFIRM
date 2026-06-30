#!/usr/bin/env bash
# 04 — Aggregate ALREADY-COMPUTED FreeSurfer into the same tables as 03, WITHOUT
# re-running recon-all. Use this for cohorts that ship recon-all output inline
# (e.g. MDD_China: Data/<site>/T1/<subj>/fs_7.3.2/<subj>/stats/aseg.stats).
# Usage:  bash 04_collect_precomputed.sh <COHORT> [search_root_override]
set -euo pipefail
cd "$(dirname "$0")"

COHORT="${1:?usage: 04_collect_precomputed.sh <COHORT> [search_root]}"
ROOT="${2:-$(awk -F'\t' -v c="$COHORT" '$1==c{print $2}' cohorts.tsv)}"
[ -n "$ROOT" ] || { echo "ERROR: '$COHORT' not in cohorts.tsv and no root given"; exit 1; }
[ -d "$ROOT" ] || { echo "ERROR: root does not exist: $ROOT"; exit 1; }

SD="$PWD/freesurfer/${COHORT}"; OUT="tables/${COHORT}"; mkdir -p "$SD" "$OUT" logs
FREESURFER_MODULE="${FREESURFER_MODULE:-freesurfer/7.3.2}"
module load "$FREESURFER_MODULE" 2>/dev/null || true
# source "$FREESURFER_HOME/SetUpFreeSurfer.sh" 2>/dev/null || true
command -v asegstats2table >/dev/null || { echo "ERROR: asegstats2table not on PATH"; exit 1; }

# Locate every completed recon-all subject (has stats/aseg.stats); drop the fsaverage template.
# MDD_China has a known layout; use a direct glob because broad `find` scans on
# this shared filesystem can stall before yielding any results.
STATS=()
if [ "$COHORT" = "MDD_China" ]; then
  echo "[$COHORT] scanning known MDD_China layout: */T1/*/fs_7.3.2/*/stats/aseg.stats"
  for st in "$ROOT"/*/T1/*/fs_7.3.2/*/stats/aseg.stats; do
    case "$st" in */fsaverage/*|*/BCKUP/*) continue ;; esac
    [ -e "$st" ] && STATS+=("$st")
  done
else
  STAT_FIND_MAXDEPTH="${STAT_FIND_MAXDEPTH:-8}"
  echo "[$COHORT] scanning $ROOT for */stats/aseg.stats (maxdepth=$STAT_FIND_MAXDEPTH)"
  mapfile -t STATS < <(find "$ROOT" -maxdepth "$STAT_FIND_MAXDEPTH" -path '*/stats/aseg.stats' 2>/dev/null \
                         | grep -Ev '/(fsaverage|BCKUP)/' | sort)
fi
echo "[$COHORT] found ${#STATS[@]} precomputed recon-all subjects under $ROOT"
[ "${#STATS[@]}" -gt 0 ] || { echo "  none found — is this cohort actually FreeSurfer-processed?"; exit 1; }

# Symlink each FreeSurfer subject dir into a flat SUBJECTS_DIR with a unique id
# (id = path before '/fs_*', matching 01's native-T1 ids).
: > "$OUT/completed_subjects.txt"
declare -A seen
for st in "${STATS[@]}"; do
  subjdir=$(dirname "$(dirname "$st")")                 # .../<id>/fs_x.y.z/<id>
  rel="${st#"$ROOT"/}"; sid="${rel%%/fs_*}"; sid="${sid//\//_}"
  [ -n "$sid" ] || sid=$(basename "$subjdir")
  [ -n "${seen[$sid]:-}" ] && continue                  # dedup (e.g. stray nested fs dirs)
  seen[$sid]=1
  ln -sfn "$subjdir" "$SD/$sid"
  echo "$sid" >> "$OUT/completed_subjects.txt"
done
NC=$(wc -l < "$OUT/completed_subjects.txt" | tr -d ' ')
echo "[$COHORT] linked $NC unique subjects -> $SD"

export SUBJECTS_DIR="$SD"
mapfile -t SUBJS < "$OUT/completed_subjects.txt"
asegstats2table  --subjects "${SUBJS[@]}" --meas volume    --tablefile "$OUT/aseg_volume.tsv"          --skip
aparcstats2table --subjects "${SUBJS[@]}" --hemi lh --meas thickness --tablefile "$OUT/aparc_lh_thickness.tsv" --skip
aparcstats2table --subjects "${SUBJS[@]}" --hemi rh --meas thickness --tablefile "$OUT/aparc_rh_thickness.tsv" --skip
echo "[$COHORT] wrote (from precomputed FreeSurfer, no recon-all):"
ls -la "$OUT"/*.tsv
echo "Send the $OUT/ directory back for ingestion."
