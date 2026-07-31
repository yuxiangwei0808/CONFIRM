# Remote Data Discovery Report

Date: 2026-06-16  
Remote host: `arcdev`  
Remote policy used: read-only inspection only. No remote files were created, edited, moved, or deleted.

## Summary

The remote machine already contains several directly useful derivative datasets for scaling CONFIRM. The strongest immediate opportunity is fMRI derivative benchmarking across ABCD, UKB, HCP, HCP-Aging, ADNI, OASIS3, ABIDE2, ADHD200, and BLSA. These include metadata CSVs, network-level FC descriptors, ICA/FNC/fALFF descriptors, dynamic descriptors, and large HDF5 derivative matrices.

For sMRI, ADNI/OASIS3 are already usable locally and UKB has many preprocessed sMRI subject folders plus a gathered metadata table, but UKB sMRI will need a dedicated adapter/postprocessing step. Shared NeuroMark folders contain many additional disease cohorts and tables, especially schizophrenia/psychosis cohorts (COBRE, FBIRN, SZ_JH, BSNIP) and AD cohorts (AIBL, NACC, MIRIAD).

## Highest-Value Data We Can Use Next

### 1. fMRI derivative cohorts under `/data/users1/ywei/data`

These are the fastest to integrate because they are already table-like and generally share a common descriptor format.

| Cohort | Useful paths | Apparent scale | Modality | Direct use |
|---|---|---:|---|---|
| ABCD | `/data/users1/ywei/data/ABCD/fmri/metadata.csv`; `/fmri/descriptors/fc_descriptors.csv`; `/ica_descriptors.csv`; `/region_self_descriptors.csv`; `/TianS3/data_resampled.h5`; `/fc/data_fc.h5` | metadata 10,392 rows; descriptor files up to ~97k lines | rsfMRI | Yes, with adapter |
| HCP | `/data/users1/ywei/data/HCP/fmri/metadata_with_text_medical.csv`; `/fmri/descriptors/*`; `/TianS3/data_resampled.h5`; `/fc/data_fc.h5` | ~1,080 subject/session rows in self descriptors; expanded descriptors larger | rsfMRI | Yes, with adapter |
| HCP_Aging | `/data/users1/ywei/data/HCP_Aging/fmri/metadata.csv`; `/descriptors/*`; `/fc/data_fc.h5` | ~633 metadata rows; ~726 self-descriptor rows | rsfMRI + aging/cognition | Yes, with adapter |
| UKB | `/data/users1/ywei/data/UKB/fmri/metadata.csv`; `/descriptors/*`; `/TianS3/data_resampled.h5`; `/fc/data_fc.h5` | metadata 39,293 rows; fc descriptors 196,526 lines | rsfMRI + cognition | Yes, but large; use sampled/dev adapter first |
| ADNI | `/data/users1/ywei/data/ADNI/fmri/metadata.csv`; `/descriptors/*`; `/TianS3/data_resampled.h5`; `/fc/data_fc.h5` | metadata 1,281 rows | rsfMRI + AD clinical | Yes, with adapter |
| OASIS3 | `/data/users1/ywei/data/OASIS3/fmri/metadata.csv`; `/descriptors/*`; `/TianS3/data_resampled.h5`; `/fc` | metadata 914 rows | rsfMRI + AD clinical | Yes, with adapter |
| ABIDE2 | `/data/users1/ywei/data/ABIDE2/fmri/metadata.csv`; `/descriptors/*`; `/fc/data_fc.h5` | metadata 1,115 rows | rsfMRI + ASD | Yes, with adapter |
| ADHD200 | `/data/users1/ywei/data/ADHD200/fmri/metadata_with_text_medical_all.csv`; `/descriptors/*`; `/fc/data_fc.h5` | metadata 3,121 expanded rows; 620 FC runs | rsfMRI + ADHD | Yes, with adapter |
| BLSA | `/data/users1/ywei/data/BLSA/fmri/metadata.csv`; `/TianS3`; `/fc` | metadata 2,103 rows | rsfMRI + aging | Yes, with adapter |

Direct-use strategy: start with descriptor CSVs rather than HDF5. The CSVs are already near-canonical: metadata contains subject/session/age/sex/diagnosis/cognition; descriptors contain network and region measures. HDF5 can be added later for richer region/edge-level benchmarks.

### 2. Existing local ADNI/OASIS3 sMRI/PET

Already used locally:

- `data/raw/ADNIMERGE.xlsx`
- `data/raw/oasis3_data_info.tar.gz`
- extracted OASIS3 FreeSurfer and clinical tables

These support the current ADNI -> OASIS3 AD atrophy replication and should remain the main sMRI/PET anchor.

### 3. UKB sMRI

Paths:

- `/data/users1/ywei/data/UKB/smri/preprocessed/` has many subject directories.
- `/data/users1/ywei/data/UKB/gathered/UKBiobank_sMRI_metadata.csv`
- `/data/users1/ywei/data/UKB/gathered/raw_smri.zarr`
- `/data/users1/ywei/data/UKB/gathered/brain_mask_smri.npy`

Assessment: not directly usable by CONFIRM yet unless the gathered table contains usable regional IDPs. Needs a targeted adapter/postprocessor to inspect `UKBiobank_sMRI_metadata.csv` and the zarr layout. Because the subject directory count is very large, avoid recursive walks; use exact-path scripts.

### 4. Shared NeuroMark datasets under `/data/qneuromark/Data`

Top-level data include AD/aging, schizophrenia/psychosis, autism, depression, Huntington's, HCP, UKB, and other disease cohorts.

Most promising directly table-like assets:

- AD/aging:
  - `/data/qneuromark/Data/ADNI/Data_info/ADNIMERGE.xlsx`
  - `/data/qneuromark/Data/AIBL/demos/*` including diagnosis, CDR, MMSE, PET/MRI metadata
  - `/data/qneuromark/Data/NACC/data/investigator_mri_nacc65.csv` with MRI-derived measures including `NACCICV`, `NACCBRNV`, regional volumes
  - `/data/qneuromark/Data/MIRIAD/files/*.csv`
- Schizophrenia/psychosis:
  - `/data/qneuromark/Data/COBRE/Data_info/pheno_comb_cobre_all.csv`
  - `/data/qneuromark/Data/COBRE/Data_info/PANSS_COBRE.csv`
  - `/data/qneuromark/Data/FBIRN/Data_info/fBIRN_CMINDS_4rsfMRI2_G.csv`
  - `/data/qneuromark/Data/SZ_JH/Data_info/demo.csv`
- Other:
  - `/data/qneuromark/Data/PREDICT-HD/phenotype/*.tsv`
  - `/data/qneuromark/Data/HBN/Data_Info/HBN_R*_Pheno.csv`
  - `/data/qneuromark/Data/UKBiobank/Data_info/my_ukb_rfMRI.csv`
  - `/data/qneuromark/Data/HCP/Data_info/HCP_demo.csv`

Assessment: this folder can substantially expand the benchmark beyond AD, especially with schizophrenia/psychosis clinical anchors. It needs cohort-specific adapters, but many files are already CSV/TSV.

### 5. Shared NeuroMark2 datasets under `/data/neuromark2/Data`

Useful candidates:

- `ABCD`, `HCP_Aging`, `HCP_Development`
- `BSNIP2`, `AMP_SCZ`, `Olin_ASD_SZ` for psychiatric/neurodevelopmental claims
- `HIV_Duke/Data_Info/Neuromark_2022-0901_redux.csv`
- `MexicoCUD/participants.tsv`
- `Autism_baby/Data_Info/Demo_baby.csv`

Assessment: useful for benchmark breadth, but likely second wave after the user-owned fMRI derivative cohorts and qNeuroMark COBRE/FBIRN/SZ_JH anchors.

## Proposed Postprocessing Plan

### A. Direct descriptor-CSV adapter

Build one reusable adapter for folders like:

```text
<cohort>/fmri/metadata.csv
<cohort>/fmri/descriptors/fc_descriptors.csv
<cohort>/fmri/descriptors/ica_descriptors.csv
<cohort>/fmri/descriptors/region_self_descriptors.csv
```

The adapter should:

1. read metadata;
2. read selected descriptor table;
3. align rows by `subject_id`, `session_id` when present;
4. prefix features as `fc_`, `ica_`, `dyno_`, or `fmri_`;
5. normalize canonical columns: `subject_id`, `session`, `cohort`, `site`, `age`, `sex`, `dx`, `meanFD` if present;
6. emit local canonical parquet.

This single adapter should cover ABCD, HCP, HCP_Aging, UKB-fMRI, ADNI-fMRI, OASIS3-fMRI, ABIDE2, ADHD200, and BLSA with small mapping YAMLs.

### B. HDF5/zarr adapter, later

Use only after descriptor CSV claims are working. HDF5 files are large:

- UKB FC HDF5 ~21 GB
- ABCD FC HDF5 ~11 GB
- ABCD TianS3 HDF5 ~5.2 GB

These should not be pulled locally unless needed. Prefer running lightweight remote sampling or derived summary extraction after explicit user approval, because that would create derived files.

### C. Disease-specific table adapters

Build small adapters for high-value non-AD disease claims:

- COBRE/FBIRN/SZ_JH: clinical + fMRI/sMRI if derivative files can be located.
- AIBL/NACC/MIRIAD: AD/aging replication breadth.

## Recommended Next Benchmark Expansion

Use the remote data in this order:

1. **fMRI benchmark v0.1**: ABCD, HCP, HCP_Aging, ABIDE2, ADHD200, ADNI-fMRI, OASIS3-fMRI.
   - Claims: age effects, sex effects, cognition associations, diagnosis effects, injected site/confound traps.
   - Modality: `fMRI-FC`, `fMRI-ICA`, `fMRI-dynamics`.
2. **AD multi-modal expansion**: ADNI + OASIS3 + AIBL + NACC/MIRIAD.
   - Claims: sMRI atrophy, PET if available, fMRI connectivity.
3. **Psychosis expansion**: COBRE + FBIRN + SZ_JH, after imaging-feature alignment.
   - Claims: schizophrenia/control differences in anatomy, rsfMRI networks, DWI measures.
4. **UKB scale stress test**:
   - Claims: age, sex, fluid intelligence, imaging-derived phenotype associations.
   - Use descriptor CSVs first; defer 21 GB HDF5.

## Data Needed From User

No new downloads are needed for the next implementation step. The existing remote folders are sufficient to scale the benchmark.

I do need one decision before creating derived local files from remote data:

- Should I copy selected CSV metadata/descriptor files from `arcdev` into the local workspace under `data/raw_remote/` and build canonical parquet locally?

This would leave the remote read-only, but it will create local copies and derived local canonical files. For the first wave, only CSVs are needed; no large HDF5 files.

## Important Limits

- I did not inspect every file in UKB/sMRI or all NeuroMark datasets because recursive traversal is too expensive.
- Remote Python lacks pandas, so column inspection was done via headers and file sizes.
- No remote files were modified.
- Some datasets may have access constraints or de-identification/DUA constraints even if technically readable.
