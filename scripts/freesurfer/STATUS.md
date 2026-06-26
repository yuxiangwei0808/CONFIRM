# FreeSurfer external-cohort pipeline — STATUS / HANDOFF (2026-06-20)

Purpose: produce FreeSurfer structural ROI tables for **unseen** SZ/MDD cohorts, to add
literature-anchored external NULLS (ENIGMA SZ caudate/putamen, 10 ENIGMA-MDD subcortical
d≈0) and cross-cohort structural POSITIVES to the CONFIRM paper's external benchmark.

---

## 1. Scripts written (this folder)
| file | role |
|---|---|
| `README.md` | how to run |
| `cohorts.tsv` | per-cohort config: `cohort  t1_root  t1_glob  disorder  n  notes` |
| `01_make_subject_list.sh` | find native T1s → `lists/<COHORT>_subjects.tsv` (subj_id ⇥ T1 path) |
| `02_reconall.slurm` | local per-subject recon-all/FastSurfer runner |
| `03_aggregate.slurm` | `asegstats2table`+`aparcstats2table` over recon-all output → `tables/<COHORT>/` |
| `04_collect_precomputed.sh` | for cohorts ALREADY FreeSurfer'd (MDD_China): symlink existing recon dirs → aggregate, **no recon-all** |
| `submit.sh` | one-command driver: 01 → submit 02 array → submit 03 (afterany) |

**Deployed to cluster:** `arcdev:/data/users1/ywei/confirm_freesurfer/` (all 7 files, chmod +x). 
Cluster copy of `01`/`04` = the fixed versions (see §4).

---

## 2. Key findings (verified on arcdev = arctrdgndev101, 2026-06-20)

**Feature provenance.** NeuroMark "sMRI" = SPM/VBM outputs (`wc1*/mwc1*/Smwc1*/y_*`,
tissue-segment maps) — NOT usable for recon-all. The **native T1** (recon-all input) sits
beside them, named `T1.nii` (Shile, MDD_China) or `anat.nii` (PK_MPRC). `01` matches the
native by **exact basename**, which auto-excludes every SPM variant.

**Confirmed runnable cohorts (native-T1 counts verified):**
| cohort | disorder | n (native T1) | t1_root | glob | FreeSurfer status |
|---|---|---|---|---|---|
| Shile_Nanjing | SZ | **834** | `…/Shile_Nanjing/Data_BIDS/Raw_Data` | `T1.nii` | **none → run recon-all** |
| PK_MPRC | SZ | **444** | `…/PK_MPRC/sMRI` | `anat.nii` | **none → run recon-all** |
| MDD_China | MDD | **600 native / 599 usable FS** | `…/Depression/MDD_China/Data` | `T1.nii` | **ALREADY DONE (fs_7.3.2); `04` completed** |

- **MDD_China is already FreeSurfer-processed.** Usable output is 599 unique
  subjects after excluding one `BCKUP` duplicate directory. Each subject has
  `…/T1/<subj>/fs_7.3.2/<subj>/stats/{aseg.stats, lh.aparc.stats, rh.aparc.stats}` (+ a2009s,
  DKTatlas, BA atlases, and pre-tabulated `*_aseg_stats.txt`). → **no recon-all**; `04` is complete.
- **Shile_Nanjing & PK_MPRC have NO precomputed FreeSurfer** (deep find = 0 `aseg.stats`).
  → recon-all required. 834 + 444 = **1278 recon-all subjects**.
- MDD_China has **4 sites** (anding/huaxi/xinxiang/zhejiang) → within-cohort cross-site
  replication. SZ cross-cohort pair = **Shile ↔ PK_MPRC**.

**Unavailable on arcdev (do NOT target):** `AMP_SCZ`, `HCP_EarlyP`, `EMBARC`, `MDD_DIRECT`
— empty mount points (root-owned, 0 bytes) or no native T1. Commented out in `cohorts.tsv`.

**Diagnosis (dx) sources — for the ingestion step, NOT for these scripts:**
- PK_MPRC: `…/PK_MPRC/Data_info/Subj_info_MPRC_merge_5.xlsx` (+ `Demographics_S123_*.xlsx`); match the `Z…` ids.
- MDD_China: dx encoded in subject id (`DP_`=depression, `NC_`=control) + `…/MDD_China/Data_Info/MDD_China_organized.csv`.
- **Shile_Nanjing: dx UNKNOWN.** `Data_info/` holds only `.docx` (磁共振序列/说明); group codes
  XA/XB/… don't obviously encode SZ-vs-control. **NEED the label source from user** (NeuroMark
  summary CSV? a separate sheet?). Blocks the Shile claim build, not the recon-all.

---

## 3. Verifications done
- `01` subject-id logic unit-tested locally on 3 mock layouts → unique, collision-free,
  sessions collapsed, SPM files excluded.
- Same logic run against the **real** filesystem → **834 / 444 / 600** clean unique ids
  (e.g. `XA_XA100`, `Scanner1_Z03060`, `anding_T1_DP_A1_…`).
- `cohorts.tsv` lookup (root+glob, comment lines skipped, commented cohorts disabled) → OK.
- Deployed scripts run end-to-end on cluster: `lists/PK_MPRC_subjects.tsv` (444),
  `lists/Shile_Nanjing_subjects.tsv` (834) written correctly.
- MDD_China `fs_7.3.2/<subj>/stats/` confirmed to contain a complete recon-all output set.
- MDD_China aggregation completed on arcdev with `module load freesurfer/7.3.2`:
  `tables/MDD_China/{aseg_volume.tsv,aparc_lh_thickness.tsv,aparc_rh_thickness.tsv,completed_subjects.txt}`.
  Copied back locally under `scripts/freesurfer/tables/MDD_China/`; each TSV has
  599 subject rows plus header.

---

## 4. Pending issues / blockers
1. **RESOLVED — FreeSurfer location.** On arcdev, `module avail` exposes
   `freesurfer/7.3.2` and `freesurfer/7.4.1`. Recon/FastSurfer scripts now
   prefer an already sourced `FREESURFER_HOME`; otherwise they load
   `${FREESURFER_MODULE:-freesurfer/7.4.1}`. MDD_China precomputed aggregation
   can still use 7.3.2 via `FREESURFER_MODULE=freesurfer/7.3.2` if needed.
2. **No SLURM scheduling.** `submit.sh` now runs directly on the remote machine
   and does not call `sbatch`.
3. **MDD_China `01` returns 1200, not 600**: each subject has both `<subj>/T1.nii` and a
   duplicate `<subj>/T1/T1.nii`. Harmless — MDD_China uses `04` (keyed on `stats/aseg.stats`,
   unambiguous = 600), not `01`. Only relevant if someone tries to recon-all MDD (don't).
4. **MDD_China precomputed duplicate excluded**: the remote tree has both
   `.../NC_A2_sub_070_shijingping_NC19/fs_7.3.2/BCKUP/stats/aseg.stats` and the
   real subject stats directory. `04` skips `BCKUP`, yielding 599 usable rows.
5. **Shile dx label source unknown** (see §2) — needed before building Shile SZ claims.
6. Runtime: 1278 recon-all subjects @ ~6–10 h CPU each. On the dev machine, keep
   `LOCAL_JOBS` small (default 1). Use FastSurfer (`RECON=fastsurfer`, ~1 h/subj
   GPU) only if GPU/FastSurfer are configured, and always run a `NImax=N` smoke test first.

---

## 5. Pending tasks (forward plan)
1. **Ingest MDD_China tables:** local tables are ready in `scripts/freesurfer/tables/MDD_China/`.
   Convert to cohort parquet `[subject_id,site,dx,age,sex,smri_icv,smri_<region>]`
   using dx from `NC_`/`DP_` prefix and `Data_Info/MDD_China_organized.csv`.
2. **Shile_Nanjing + PK_MPRC:** smoke-test with
   `NImax=2 LOCAL_JOBS=1 bash submit.sh Shile_Nanjing`, then run
   `LOCAL_JOBS=1 bash submit.sh Shile_Nanjing PK_MPRC` (or `RECON=fastsurfer`
   if FastSurfer is configured). Use `OUT_DIR=/path/to/output` to place
   FreeSurfer subject directories outside the default `freesurfer/<COHORT>`.
   → `tables/<COHORT>/`.
3. **Send `tables/<COHORT>/` back** (aseg_volume.tsv, aparc_{lh,rh}_thickness.tsv, completed_subjects.txt).
4. **Ingest** (my side): tables → cohort parquets `[subject_id,site,dx,age,sex,smri_icv,smri_<region>]`
   (join dx per §2; resolve Shile labels). Reuse `prepare_ds000030_external.py` aseg parser /
   ROI_MAP (hippocampus/amygdala/thalamus/accumbens/pallidum/caudate/putamen/lateralventricle).
5. **Build claims CSVs** from `data/external_benchmark/enigma_label_table.csv`:
   SZ structural positives + SZ literature-nulls (caudate/putamen) for Shile/PK; 10 ENIGMA-MDD
   subcortical literature-nulls + any MDD positives for MDD_China.
6. **Run** `src/bench/run_external_generic.py` (cross-cohort SZ Shile↔PK; cross-site MDD within
   MDD_China) → fold structural SZ/MDD positives + literature-nulls into the paper's external
   benchmark (currently NACC AD + ds000030 abstention).

---

## 6. Output contract (what every cohort produces)
```
tables/<COHORT>/aseg_volume.tsv          # subcortical volumes + EstimatedTotalIntraCranialVol, 1 row/subject
tables/<COHORT>/aparc_lh_thickness.tsv   # left  Desikan-Killiany cortical thickness
tables/<COHORT>/aparc_rh_thickness.tsv   # right
tables/<COHORT>/completed_subjects.txt
```
Same format `src/bench/prepare_ds000030_external.py` already parses.
