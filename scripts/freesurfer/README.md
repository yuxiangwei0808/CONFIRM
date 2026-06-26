# FreeSurfer post-processing pipeline (CONFIRM external benchmark)

Turn the **native T1** of an unseen cohort into the FreeSurfer ROI tables CONFIRM
ingests (subcortical volumes + cortical thickness) — same format as the AOMIC /
NACC tables already used. recon-all runs on the native T1 only; the NeuroMark
`wc1*/mwc1*/Smwc1*` files are SPM/VBM outputs and are **not** usable here.

> Full state, findings, remaining blockers, and forward plan: **see `STATUS.md`**.

## Confirmed cohorts (verified on arcdev 2026-06-20 — native T1 present + counts)
| cohort | disorder | n | native T1 | FreeSurfer | how to run |
|---|---|---|---|---|---|
| Shile_Nanjing | SZ | **834** | `T1.nii` | none → recon-all | `submit.sh` |
| PK_MPRC | SZ | **444** | `anat.nii` | none → recon-all | `submit.sh` |
| MDD_China | MDD | **600 native / 599 usable FS** | `T1.nii` | **already done (fs_7.3.2)** | completed by `04_collect_precomputed.sh` (no recon-all) |

`cohorts.tsv` holds roots + globs. Empty mounts (`AMP_SCZ`, `HCP_EarlyP`, `EMBARC`,
`MDD_DIRECT`) are commented out — skip unless remounted.

FreeSurfer is resolved on arcdev: `module avail` exposes `freesurfer/7.3.2`
and `freesurfer/7.4.1`. Recon/FastSurfer defaults to `freesurfer/7.4.1`
unless you already sourced `FREESURFER_HOME`; the MDD_China precomputed
collector defaults to `freesurfer/7.3.2` to match its `fs_7.3.2` outputs.

**Why this benchmark works:** SZ gets a true cross-cohort pair (Shile ↔ PK_MPRC, also
3 PK scanners); MDD_China's 4 sites (anding/huaxi/xinxiang/zhejiang) give within-cohort
cross-site replication + unlock the 10 ENIGMA-MDD subcortical literature-nulls.

## Run it (on arcdev directly, no SLURM)
```bash
cd <where-you-put>/scripts/freesurfer
# 1) MDD_China is ALREADY FreeSurfer'd; aggregation has been run.
#    Re-run only if the remote source changes:
bash 04_collect_precomputed.sh MDD_China
# 2) SZ cohorts need recon-all. Smoke-test first:
NImax=2 LOCAL_JOBS=1 bash submit.sh Shile_Nanjing
# 3) Full local-machine run. Keep LOCAL_JOBS small on the dev node:
LOCAL_JOBS=1 bash submit.sh Shile_Nanjing PK_MPRC
# Optional: put FreeSurfer subject directories somewhere else:
OUT_DIR=/scratch/ywei/confirm_freesurfer bash submit.sh Shile_Nanjing
```
`submit.sh` does: `01` build subject list → local `02` recon-all/FastSurfer loop →
`03` aggregate. It has no `sbatch`/SLURM dependency. Use `tmux`, `screen`, or
`nohup` for long runs because output streams to the terminal.

## recon-all vs FastSurfer — pick FastSurfer
1878 subjects total. recon-all is ~6–10 h/subject (CPU); FastSurfer is ~1 h (GPU).
Set `RECON=fastsurfer` if `FASTSURFER_HOME` and GPU access are available on the
remote machine. Output stats are identical in format either way. Use `NImax=N`
for a test batch, `LOCAL_JOBS=N` for local parallelism (default 1), and
`THREADS=N` to control threads per subject.
The recon/aggregate scripts prefer an already sourced `FREESURFER_HOME`;
otherwise they load `${FREESURFER_MODULE:-freesurfer/7.4.1}`. Set
`FREESURFER_MODULE=freesurfer/7.3.2` only if you intentionally need 7.3.2.
Use `DRY_RUN=1` with `02_reconall.slurm` to verify paths without starting
recon-all.
The scripts intentionally ignore inherited `SUBJECTS_DIR` values from loaded
FreeSurfer modules; use `OUT_DIR=/path/to/output` if you need to override the
default `freesurfer/<COHORT>` output directory. `CONFIRM_SUBJECTS_DIR` remains
as a backward-compatible alias; do not set both.

## Output to send back
Per cohort, after `03`:
```
scripts/freesurfer/tables/<COHORT>/aseg_volume.tsv          # subcortical vols + eTIV, 1 row/subject
scripts/freesurfer/tables/<COHORT>/aparc_lh_thickness.tsv
scripts/freesurfer/tables/<COHORT>/aparc_rh_thickness.tsv
scripts/freesurfer/tables/<COHORT>/completed_subjects.txt
```
MDD_China is already copied back locally at `scripts/freesurfer/tables/MDD_China/`.
Send me those `tables/<COHORT>/` dirs. I handle diagnosis/age/sex at ingestion:
- **PK_MPRC** — `Data_info/Subj_info_MPRC_merge_5.xlsx` (match the `Z…` ids).
- **MDD_China** — dx is in the subject id (`DP_`=depression, `NC_`=control) + `Data_Info/MDD_China_organized.csv`.
- **Shile_Nanjing** — `Data_info/` has only `.docx`; the group/subject codes don't obviously encode dx. **I'll need the SZ/control label source for Shile** (NeuroMark summary, a separate sheet, or you point me to it).

## Notes
- Subject ids are path-derived & unique (e.g. `Scanner1_Z03060`, `anding_T1_DP_A1_…`); sessions collapse to the earliest. Verified against the real filesystem.
- `01` is idempotent; `02` skips subjects whose `aseg.stats` already exists, so re-running resumes.
- To resume from a particular row in `lists/<COHORT>_subjects.tsv`, use `START_INDEX=N`.
- There is no SLURM scheduling now; this uses the remote machine directly.
