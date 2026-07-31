# Remote Peek Summary

Date: 2026-06-16  
Host: `arcdev`  
Mode: read-only inventory/header inspection. No remote files were modified or copied.

This report has been sanitized after the dataset scope was corrected. The complete-pipeline path should use fMRI descriptor cohorts that are already copied locally or clearly available under `/data/users1/ywei/data`.

## Useful Findings

### fMRI Descriptor Cohorts

The strongest immediate assets are metadata and fMRI descriptor CSVs for:

- ABCD
- HCP
- HCP-Aging
- UKB
- ADNI-fMRI
- OASIS3-fMRI
- ABIDE2
- ADHD200
- BLSA metadata

These are table-like and suitable for CPU-only benchmark construction.

### Psychosis Data

COBRE has richer metadata than copied in wave 1:

- diagnosis, PANSS, cognitive, medication spreadsheets
- copied CSVs already include `Diagnosis` and PANSS anchor tables

FBIRN has richer metadata than copied in wave 1:

- diagnosis/cognition CSV already copied
- additional PANSS, medication, clinical spreadsheets are available

SZ_JH has:

- demo, neuropsychological, SANS/SAPS, medication tables
- copied wave 1 tables already cover the main clinical anchors

Interpretation:

- Psychosis clinical labels are available.
- The missing piece is aligned imaging-derived feature tables for those subject IDs.

### AD / Aging

AIBL has additional small metadata tables not copied in wave 1:

- `AIBL_MRI_Demos.csv`
- `aibl_av45meta_01-Jun-2018.csv`
- `aibl_flutemeta_01-Jun-2018.csv`
- `aibl_mri3meta_01-Jun-2018.csv`
- `aibl_apoeres_01-Jun-2018.csv`
- `aibl_neurobat_01-Jun-2018.csv`
- other clinical/visit metadata

NACC MRI/PET/CSF tables are already copied and prepared locally, but NACC disease claims need the clinical diagnosis/demographic table that is not currently available locally.

MIRIAD has raw-ish sMRI subject folders plus assessment/scans metadata. It may be usable only if we either derive features from images later or find precomputed imaging measures elsewhere.

## Recommended Remote Copy Wave 2

Small table-only copy, no raw images:

1. AIBL metadata expansion, when access is available:
   - `/data/qneuromark/Data/AIBL/demos/AIBL_MRI_Demos.csv`
   - `/data/qneuromark/Data/AIBL/demos/aibl_av45meta_01-Jun-2018.csv`
   - `/data/qneuromark/Data/AIBL/demos/aibl_flutemeta_01-Jun-2018.csv`
   - `/data/qneuromark/Data/AIBL/demos/aibl_mri3meta_01-Jun-2018.csv`
   - `/data/qneuromark/Data/AIBL/demos/aibl_apoeres_01-Jun-2018.csv`
   - `/data/qneuromark/Data/AIBL/demos/aibl_neurobat_01-Jun-2018.csv`

2. Psychosis metadata expansion, if needed:
   - COBRE cognitive/diagnosis/PANSS/medication spreadsheets
   - FBIRN PANSS/clinical/medication spreadsheets
   - SZ_JH medication table
   - ChineseSZ/UCSF prodrome demo tables only after deciding whether to integrate those cohorts

## Next Technical Implication

The best immediate adapter target is the local fMRI descriptor backbone:

- metadata + FC descriptors
- ICA/dynamics descriptors
- region-level descriptors
- label-aware gate evaluation
- synthetic artifact/null tasks
