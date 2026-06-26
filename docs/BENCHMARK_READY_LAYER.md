# Benchmark-Ready Data Layer

Date built: 2026-06-16  
Builder: `scripts/build_benchmark_ready_layer.py`

## Purpose

This layer is the stable data interface between the copied/prepared remote tables
and future CONFIRM benchmark runs. It is generated locally and can be deleted and
rebuilt without touching remote data or copied raw tables.

## Location

```text
data/prepared_data/benchmark_ready/
```

## Core Outputs

| File | Purpose |
|---|---|
| `cohorts/*.parquet` | Canonical-ish cohort tables for benchmark construction |
| `cohort_manifest.csv` | Row counts, feature counts, covariate coverage by cohort |
| `feature_dictionary.csv` | Feature modality/family/source/missingness |
| `claim_inventory_ready.csv` | Candidate claims annotated with readiness |
| `misc_table_manifest.csv` | Non-canonical disease/multimodal tables staged for adapters |
| `README.md` | Regeneration notes |

## Prepared Cohort Bundles

| Cohort | Rows | Feature Columns | Rows With Any Feature | Modality |
|---|---:|---:|---:|---|
| ABCD | 10,391 | 190 | 10,391 | fMRI |
| ABIDE2 | 1,114 | 84 | 772 | fMRI |
| ADHD200 | 624 | 190 | 619 | fMRI |
| ADNI_fMRI | 1,280 | 190 | 1,280 | fMRI |
| HCP | 1,079 | 84 | 1,079 | fMRI |
| HCP_Aging | 632 | 84 | 632 | fMRI |
| OASIS3_fMRI | 913 | 190 | 913 | fMRI |
| UKB | 39,291 | 190 | 39,291 | fMRI |

Total feature dictionary entries: 1,234

Feature families:

- fMRI FC-network: 672
- fMRI region: 380
- fMRI dynamics: 150
- phenotype/covariate labels: 32

## Claim Inventory Readiness

`claim_inventory_ready.csv` contains 36 candidate claims:

- 26 benchmark-ready now:
  - 23 fMRI claims from the remote descriptor cohorts
  - 2 already-working sMRI AD claims
  - 1 already-working within-ADNI PET claim
- 7 need disease-specific adapters
- 3 need imaging-feature matching

Ready claims by modality:

- fMRI-FC: 16
- fMRI-region: 4
- fMRI-dynamics: 3
- sMRI: 2
- PET: 1

## Cleanup Performed

The builder cleans and recreates only:

```text
data/prepared_data/benchmark_ready/
```

It does not modify:

- remote files on `arcdev`
- copied raw files in `data/raw_remote/`
- intermediate prepared files in `data/prepared_data/fmri_descriptors/`
- miscellaneous staged tables in `data/prepared_data/misc_tables/`
- existing CONFIRM experiment outputs

## Notes Before Experiment Integration

This layer is intentionally data-only. The next step is to teach the benchmark
runner to consume these cohort parquet files and claim inventory entries. No
benchmark statistics were run as part of this data-preparation step.

