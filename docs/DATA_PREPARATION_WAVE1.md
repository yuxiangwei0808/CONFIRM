# Data Preparation Wave 1

Date: 2026-06-16  
Remote host: `arcdev`  
Remote policy used: read-only. Existing remote files were not modified.  
Local output roots:

- raw copied files: `data/raw_remote/arcdev/`
- prepared fMRI descriptor bundles: `data/prepared_data/fmri_descriptors/`
- prepared miscellaneous disease/multimodal tables: `data/prepared_data/misc_tables/`
- manifests: `docs/data_manifests/`

## What Was Copied

Copied selected CSV/TSV/XLSX-style table files only. No raw images, no HDF5, no zarr, and no large subject folders were copied.

Copy list:

- `docs/data_manifests/remote_copy_list_wave1.txt`

Raw copy manifest:

- `docs/data_manifests/raw_remote_wave1_manifest.csv`

The copied files total about 226 MB and cover:

- user-owned fMRI descriptor cohorts: ABCD, HCP, HCP-Aging, UKB, ADNI-fMRI, OASIS3-fMRI, ABIDE2, ADHD200, BLSA metadata
- shared NeuroMark disease/multimodal tables: COBRE, FBIRN, SZ_JH, AIBL, NACC, MIRIAD

## Prepared fMRI Descriptor Bundles

These are local parquet files built by `scripts/prepare_remote_tables.py`. They join metadata with descriptor CSVs and preserve features as `fc_*` or `raw_fmri_*` columns.

| Cohort | Rows | Columns | Feature Columns | Rows With Any Feature | Output |
|---|---:|---:|---:|---:|---|
| ABCD | 10,391 | 203 | 190 | 10,391 | `data/prepared_data/fmri_descriptors/ABCD.parquet` |
| HCP | 1,079 | 94 | 84 | 1,079 | `data/prepared_data/fmri_descriptors/HCP.parquet` |
| HCP_Aging | 632 | 96 | 84 | 632 | `data/prepared_data/fmri_descriptors/HCP_Aging.parquet` |
| UKB | 39,291 | 201 | 190 | 39,291 | `data/prepared_data/fmri_descriptors/UKB.parquet` |
| ADNI_fMRI | 1,280 | 201 | 190 | 1,280 | `data/prepared_data/fmri_descriptors/ADNI_fMRI.parquet` |
| OASIS3_fMRI | 913 | 202 | 190 | 913 | `data/prepared_data/fmri_descriptors/OASIS3_fMRI.parquet` |
| ABIDE2 | 1,114 | 94 | 84 | 772 | `data/prepared_data/fmri_descriptors/ABIDE2.parquet` |
| ADHD200 | 624 | 199 | 190 | 619 | `data/prepared_data/fmri_descriptors/ADHD200.parquet` |

Summary file:

- `docs/data_manifests/prepared_fmri_wave1_summary.csv`

Notes:

- ADNI-fMRI needed session reconstruction from metadata paths before descriptor rows aligned correctly.
- OASIS3-fMRI had an empty `Age` column; preparation now uses `ageAtEntry` because it has coverage.
- ABIDE2 has 1,114 metadata rows but 772 rows with descriptor features. This is usable, but downstream claims should filter to rows with non-missing features.

## Prepared Miscellaneous Tables

Miscellaneous CSV/TSV tables were converted to parquet for later disease/multimodal adapters.

Summary file:

- `docs/data_manifests/prepared_misc_wave1_summary.csv`

Most useful tables:

| Dataset | Prepared Content | Approximate Rows | Why It Matters |
|---|---|---:|---|
| NACC | MRI, amyloid PET, CSF tables | 455-11,730 | AD/aging multi-modal expansion |
| AIBL | demographics, diagnosis, CDR, MMSE, MRI/PET metadata | 655-5,586 | ADNI/OASIS replication extension |
| COBRE | diagnosis + PANSS tables | 174-252 | schizophrenia claims |
| FBIRN | cognition/diagnosis table | 332 | psychosis/cognition claims |
| SZ_JH | demo, neuropsych, SANS/SAPS | 88-185 | schizophrenia symptom/cognition claims |
| MIRIAD | assessments, counts, scans | 69-708 | AD/aging replication extension |

## What Is Ready Now

These data are ready for benchmark construction, without modifying CONFIRM internals:

1. **fMRI-FC / fMRI-dynamics benchmark claims**
   - ABCD, HCP, HCP-Aging, UKB, ADNI-fMRI, OASIS3-fMRI, ABIDE2, ADHD200
   - Candidate claims: age effects, sex effects, diagnosis effects, cognition associations, injected site/confound/null traps

2. **AD multi-modal expansion**
   - existing local ADNI/OASIS3 sMRI/PET anchor
   - copied AIBL, NACC, MIRIAD tables for later adapter work

3. **Psychosis expansion**
   - COBRE/FBIRN/SZ_JH can provide clinical labels/symptoms, but need imaging-feature matching from additional files before use as complete benchmark claims.

## What Still Needs Postprocessing

1. **CONFIRM-compatible ingest mappings**
   - The prepared fMRI parquet files are close to canonical, but not yet registered as CONFIRM cohort adapters.
   - Next step: either teach CONFIRM to ingest `data/prepared_data/fmri_descriptors/*.parquet`, or copy them into a new `data/canonical_remote/` layout after a final schema pass.

2. **Feature family dictionaries**
   - Current feature names are preserved and sanitized.
   - Need a dictionary that maps each feature to modality/family:
     - `fMRI-FC`
     - `fMRI-region`
     - `fMRI-dynamics`

3. **Disease-specific joins**
   - AIBL/NACC/MIRIAD need disease/visit harmonization before use in replication claims.
   - COBRE/FBIRN/SZ_JH need imaging-feature table matching before use as complete benchmark claims.

4. **Large derivative arrays**
   - HDF5/zarr files were not copied.
   - Use descriptor CSVs first. Copy or process large HDF5 only if a specific claim needs it.

## Recommended Next Step

Before returning to CONFIRM experiments, build a data-only `remote_benchmark_claim_inventory.csv` from these prepared tables:

- candidate claim name
- modality
- cohort(s)
- available outcome/features
- available predictor/label
- expected class: positive, null, fragile, injected-null, underpowered
- required adapter status

That inventory should drive the next scaled benchmark rather than creating claims ad hoc.
