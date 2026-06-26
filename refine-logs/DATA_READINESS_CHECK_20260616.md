# Data Readiness Check

Date: 2026-06-16

Scope: inspect the newly prepared `data/prepared_data/benchmark_ready/` layer and decide whether CONFIRM can proceed to scaled benchmark integration. Raw images remain postponed.

## Verdict

Proceed to the next implementation step.

The new data layer is good enough to build the next benchmark runner/adapter. It is not yet directly plug-and-play with the current `confirm run` canonical loader because a few cohorts contain small amounts of missing required covariates and because benchmark feature families should be selected from `feature_dictionary.csv` rather than from hard-coded prefixes.

## Ready Now

- 8 fMRI descriptor cohort parquet files:
  - ABCD: 10,391 rows, 190 feature columns
  - ABIDE2: 1,114 rows, 84 feature columns, 772 rows with non-missing features
  - ADHD200: 624 rows, 190 feature columns
  - ADNI_fMRI: 1,280 rows, 190 feature columns
  - HCP: 1,079 rows, 84 feature columns
  - HCP_Aging: 632 rows, 84 feature columns
  - OASIS3_fMRI: 913 rows, 190 feature columns
  - UKB: 39,291 rows, 190 feature columns
- Feature dictionary with 1,234 entries:
  - 672 FC-network features
  - 380 regional fMRI descriptor features
  - 150 dynamics/autocorrelation features
  - 32 phenotype/covariate entries
- Claim inventory with 36 candidates:
  - 23 fMRI benchmark-ready claims
  - 2 already-working sMRI AD claims
  - 1 already-working within-ADNI PET claim
  - 7 disease/multimodal claims needing adapters
  - 3 psychosis clinical-anchor claims needing imaging-feature matching

## Compatibility Findings

Current strict canonical validation passes as-is for:

- ABIDE2
- HCP
- HCP_Aging
- OASIS3_fMRI
- UKB

Current strict canonical validation fails as-is for:

- ABCD: 1 row has missing age
- ADHD200: 1 row has missing sex
- ADNI_fMRI: 5 rows have missing age

These are small row-level issues. The benchmark adapter should drop rows missing claim-required variables before calling the analysis engine rather than rejecting the whole cohort.

## Important Caveats

- `site` is currently often equal to cohort name, not true scanner/site. Site-confounded injected-null claims are still possible, but the strongest version needs real scanner/site/session variables when available.
- HCP and HCP_Aging have no disease label, which is fine for age, sex, and cognition claims.
- ABIDE2 has many phenotype rows without features, so ASD claims must filter to rows with complete feature and covariate data.
- Functional claims are fMRI-derivative claims, not raw-image workflows. This still counts as neuroimaging, but raw preprocessing agents remain a later module.
- Miscellaneous AIBL/NACC/MIRIAD/COBRE/FBIRN/SZ_JH tables are staged but not all benchmark-ready.

## Best Next Step

Implement a `benchmark_ready` adapter/runner that:

1. Loads `cohort_manifest.csv`, `feature_dictionary.csv`, and `claim_inventory_ready.csv`.
2. Loads cohort parquet files from `data/prepared_data/benchmark_ready/cohorts/`.
3. Selects feature families using `feature_dictionary.source_kind`:
   - `fc_self_descriptors`
   - `region_self_descriptors`
   - `ica_dyno_descriptors`
4. Drops rows missing required variables for each claim.
5. Generates scalar or brain-wide CONFIRM contracts from the inventory.
6. Runs the existing gate ladder and writes results under `review-stage/`.

Recommended first benchmark wave:

- Age FC: UKB -> HCP_Aging
- Sex FC: UKB -> HCP
- Cognition FC: UKB -> ABCD
- ADHD FC: ADHD200 -> ABCD
- ASD FC: ABIDE2 -> ABCD
- AD fMRI-FC: ADNI_fMRI -> OASIS3_fMRI
- Injected nulls: ABCD, UKB, HCP, ABIDE2, ADHD200

Recommended second benchmark wave:

- Psychosis adapters should wait until imaging-derived features are aligned with COBRE/FBIRN/SZ_JH clinical labels.
- AIBL/NACC/MIRIAD AD adapters after visit/diagnosis harmonization.
- COBRE/FBIRN/SZ_JH after imaging-feature matching.
