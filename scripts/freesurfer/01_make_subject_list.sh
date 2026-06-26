#!/usr/bin/env bash
# 01 — Build a subject -> native-T1 list for one cohort.
# Usage:  bash 01_make_subject_list.sh <COHORT> [t1_root_override] [glob_override]
# Output: lists/<COHORT>_subjects.tsv   (col1 = subject id, col2 = absolute native-T1 path)
#
# NeuroMark cohorts are NOT standard BIDS and the structural folder holds SPM/VBM
# outputs (wc1*, mwc1*, Smwc1*, y_*) next to the *native* T1. We match ONLY the
# native image by exact basename (glob, e.g. T1.nii or anat.nii), which excludes
# every SPM-prefixed variant automatically. Subject id = the subject directory path
# relative to t1_root (sessions collapsed, one T1 per subject) -> guaranteed unique.
set -euo pipefail
cd "$(dirname "$0")"

COHORT="${1:?usage: 01_make_subject_list.sh <COHORT> [t1_root] [glob]}"
ROOT="${2:-$(awk -F'\t' -v c="$COHORT" '$1==c{print $2}' cohorts.tsv)}"
GLOB="${3:-$(awk -F'\t' -v c="$COHORT" '$1==c{print $3}' cohorts.tsv)}"
GLOB="${GLOB:-T1.nii}"
[ -n "$ROOT" ] || { echo "ERROR: '$COHORT' not in cohorts.tsv and no t1_root given"; exit 1; }
[ -d "$ROOT" ] || { echo "ERROR: t1_root does not exist: $ROOT"; exit 1; }

mkdir -p lists
OUT="lists/${COHORT}_subjects.tsv"

# Prune functional/diffusion and any existing FreeSurfer/derivative dirs: the native
# T1 never lives under them, and pruning keeps the find fast on networked storage.
# (Pruning fs_* also avoids picking up FreeSurfer's internal mri/T1.nii.)
find -L "$ROOT" -maxdepth 8 \
      \( -type d \( -iname 'func' -o -iname 'fMRI' -o -iname 'dwi' -o -iname 'dMRI' \
                    -o -iname 'DTI' -o -iname 'fs_*' -o -iname 'freesurfer' \
                    -o -iname 'fastsurfer' -o -iname 'derivatives' \) -prune \) -o \
      \( -type f -name "$GLOB" -print \) 2>/dev/null \
  | grep -viE 'ZN_Prep|ZN_Neuromark|Results|/work/|sourcedata|Temp/|/mri/|/surf/' \
  | sort \
  | awk -v root="$ROOT" '
      { p=$0; sub("^"root"/","",p);                # path relative to root
        d=p; sub("/[^/]*$","",d);                  # drop filename -> subject/.../dir
        sub("/(anat|func)$","",d);                 # drop trailing /anat or /func
        sub("/ses[-_]?[0-9A-Za-z]+$","",d);        # drop trailing /ses_xx (collapse sessions)
        gsub("[/ ]","_",d);                        # remaining path -> flat unique id
        if(d=="") d=p;
        if(!(d in seen)){ seen[d]=1; print d"\t"$0 } }' > "$OUT"

N=$(wc -l < "$OUT" | tr -d ' ')
echo "[$COHORT] root=$ROOT  glob=$GLOB"
echo "[$COHORT] wrote $OUT  ($N subjects, one native T1 each)"
if [ "$N" -gt 0 ]; then echo "  sample:"; head -2 "$OUT" | sed 's/^/    /'
else echo "  WARNING: 0 matches — check t1_root and glob (native T1 basename, e.g. T1.nii / anat.nii)."; fi
