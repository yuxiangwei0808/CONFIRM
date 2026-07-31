# Next Data Requirements

Generated: 2026-06-17

This note separates what we can create from local files now, what appears to be available from the prior remote-copy wave, and what still needs user preparation.

## Can Create Locally Now

### fMRI descriptor pipeline

Available local inputs:

- `data/raw_remote/arcdev/data/users1/ywei/data/ABCD/fmri/`
- `data/raw_remote/arcdev/data/users1/ywei/data/ABIDE2/fmri/`
- `data/raw_remote/arcdev/data/users1/ywei/data/ADHD200/fmri/`
- `data/raw_remote/arcdev/data/users1/ywei/data/ADNI/fmri/`
- `data/raw_remote/arcdev/data/users1/ywei/data/HCP/fmri/`
- `data/raw_remote/arcdev/data/users1/ywei/data/HCP_Aging/fmri/`
- `data/raw_remote/arcdev/data/users1/ywei/data/OASIS3/fmri/`
- `data/raw_remote/arcdev/data/users1/ywei/data/UKB/fmri/`

This is the strongest immediate complete-pipeline path. These cohorts already have metadata plus fMRI descriptor CSVs, so we can build a complete derivative-table agentic benchmark without raw image preprocessing.

Use for:

- Age, sex, cognition, and diagnosis claims where metadata labels are explicit.
- fMRI-FC, ICA/dynamics, and region-level descriptor claims.
- Injected artifact/null benchmarks.
- Cross-cohort replication and harmonization tests.

### ADNI adapter

Available local inputs:

- `data/raw/ADNIMERGE.xlsx`
- `data/raw/ADNI_document-20260617T002101Z-3-001.zip`
- `data/raw/fc_ADNI.tar.gz`
- `data/raw/adni_pet_summary_tables_info_20260616.tar.gz`

The new ADNI document zip is useful. It contains current ADNIMERGE, diagnosis, demographics, neuropsychological/assessment, and FDG-PET summary CSVs. We can build a stronger ADNI table adapter from these without raw images.

Likely adapter outputs:

- One subject/visit-level ADNI diagnosis-cognition table.
- Optional FDG-PET summary claims if regional PET fields are complete enough.
- Optional ADNI functional-connectivity claims if `fc_ADNI.tar.gz` has subject IDs and diagnosis/visit keys that can be joined to ADNIMERGE.

### Limited AIBL adapter

Available local prepared tables:

- `data_qneuromark_Data_AIBL_demos_AIBL_ALL_2_11_2022.parquet`
- `data_qneuromark_Data_AIBL_demos_aibl_cdr_01_Jun_2018.parquet`
- `data_qneuromark_Data_AIBL_demos_aibl_mmse_01_Jun_2018.parquet`
- `data_qneuromark_Data_AIBL_demos_aibl_mrimeta_01_Jun_2018.parquet`
- `data_qneuromark_Data_AIBL_demos_aibl_pdxconv_01_Jun_2018.parquet`
- `data_qneuromark_Data_AIBL_demos_aibl_ptdemog_01_Jun_2018.parquet`

This can support a limited AIBL metadata/cognition adapter. It is not enough for a strong amyloid-PET or APOE benchmark. Since AIBL access is not available now, we should not block on it.

### Existing fMRI pilot

Already runnable from `data/prepared_data/benchmark_ready/`:

- fMRI benchmark-ready claims.
- Label-aware scoring and combined report generation.

## Maybe Available From Prior Remote History

The 2026-06-17 remote recheck verified the remote cluster is reachable and confirmed several paths. See `docs/data_manifests/remote_recheck_20260617.md`.

| Need | Local status | If remote has it, copy only |
|---|---|---|
| NACC clinical diagnosis/demographics | Deferred; user does not currently have the needed clinical data. | Subject-level diagnosis, age, sex, visit/date table when available later. |
| Rich COBRE/FBIRN medication/PANSS/cognition | Some clinical tables local. | Extra symptom/cognition/medication sheets only if they join to imaging derivatives. |
| AIBL AV45/flutemetamol/APOE/neuropsych/MRI metadata | Not local. | These can wait until access is available. Prior remote paths included `aibl_av45meta_01-Jun-2018.csv`, `aibl_flutemeta_01-Jun-2018.csv`, `aibl_apoeres_01-Jun-2018.csv`, `aibl_neurobat_01-Jun-2018.csv`, `AIBL_MRI_Demos.csv`, and `aibl_mri3meta_01-Jun-2018.csv`. |

## Data Still Needed From You

### NACC disease claims

Deferred for now. Needed later if we want NACC AD/aging disease claims instead of only MRI/PET/CSF table associations:

- Subject ID: `NACCID` or equivalent.
- Visit/date key.
- Diagnosis or cognitive status.
- Age at visit.
- Sex.
- Optional: education, APOE, race/ethnicity, scanner/site, CSF amyloid/tau status.

Preferred format: CSV or Parquet, one row per subject visit.

### Psychosis multimodal claims

Current local COBRE/FBIRN/SZ_JH tables look mostly clinical. To make neuroimaging claims, we need aligned imaging-derived features:

- Subject ID matching clinical tables.
- Diagnosis/group.
- Age and sex.
- Site/scanner.
- Imaging features from sMRI, rsfMRI connectivity, DWI, or other derivatives.

Preferred format: CSV or Parquet, one row per subject/session. Wide feature tables are fine.

Local clinical anchors already present:

- COBRE diagnosis/PANSS-like tables.
- FBIRN diagnosis and cognition table.
- SZ_JH demographics, neuropsychology, and SANS/SAPS tables.

The missing piece is aligned imaging derivatives for those same subject IDs.

### AIBL later

Since access is not available now, this can wait. When available, the most useful AIBL files are:

- Diagnosis/conversion table.
- Demographics.
- MMSE/CDR/neuropsych.
- MRI-derived measures.
- Amyloid PET summary measures, especially AV45/flutemetamol if available.
- APOE, if permitted.

## Required Table Shape

Any new dataset is easiest to adapt if it has:

- `dataset`: dataset name.
- `subject_id`: stable subject ID.
- `session_id` or `visit`: visit/session key.
- `age`: numeric age at visit/session.
- `sex`: coded consistently or documented.
- `diagnosis` or `group`: if disease claims are intended.
- `site` or `scanner`: if multi-site or harmonization claims are intended.
- Imaging-derived feature columns: region volumes, cortical thickness, PET SUVRs, connectivity edges, ASL perfusion, DWI metrics, or similar.

If column names differ, that is fine. Include a README/data dictionary or the original documentation inside the archive.

## Compute Requirements

The current project is CPU-only because it consumes tables and imaging-derived features. GPU is not required for the current benchmark.

GPU becomes useful only if we add:

- Raw-image preprocessing, such as FreeSurfer/FastSurfer/fMRIPrep/QSIPrep.
- A neuroimaging vision-language model or 3D image model.
- Large representation learning over raw volumes or connectivity matrices.

For the next full-scale table/derivative experiments, CPU machines with enough RAM are the right resource.
