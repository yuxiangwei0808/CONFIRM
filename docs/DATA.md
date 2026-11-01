# Data inputs

CONFIRM analyzes tables of imaging-derived measurements. Prepare one Parquet
file per cohort or evidence partition, named `<cohort>.parquet`. Each row
describes one participant observation. Assign participants to separate
discovery, replication, and holdout partitions before analysis.

## Columns

| Required column | Meaning |
|---|---|
| `subject_id` | Participant identifier within the cohort |
| `cohort` | Cohort or partition name matching the filename |
| `site` | Acquisition site or scanner batch |
| `age` | Numeric age |
| `sex` | Sex encoded as `M` or `F`; common text and numeric encodings are normalized |

Include at least one numeric imaging measurement. Structural MRI, PET,
functional-connectivity, and behavioral columns use the prefixes `smri_`,
`pet_`, `fc_`, and `beh_`, respectively. The schema also accepts imaging
columns with the `raw_` prefix.

Optional metadata include `session`, `dx`, `eTIV`, `field_strength`, and
`fs_version`. Include every outcome, predictor, covariate, and subgroup
variable named in the analysis plan. The CSV importer rejects invalid age
or sex values. During analysis, the cohort loader drops rows with unusable
age or sex; inspect the sample sizes in the result receipt.

## Convert a CSV

For a CSV named `cohort.csv` with columns `participant`, `scanner`,
`age_years`, `sex`, and `mean_thickness`, save this mapping as `mapping.yaml`:

```yaml
cohort: MY_COHORT
columns:
  subject_id: participant
  site: scanner
  age: age_years
  sex: sex
idps:
  smri_meanthickness: mean_thickness
```

Run from the directory containing those two files:

```bash
confirm ingest --cohort generic_csv --raw-csv cohort.csv \
  --mapping-yaml mapping.yaml --out outputs/cohorts
```

This writes `outputs/cohorts/MY_COHORT.parquet` and
`outputs/cohorts/MY_COHORT.dict.csv`. Use `MY_COHORT` as the cohort name in
the analysis plan. Repeat with a separate mapping for each cohort.

The [mapping template](../configs/cohort_maps/_TEMPLATE.yaml) also shows
optional metadata, value mappings, and behavioral columns. Map each
measurement explicitly to preserve its scientific meaning.

The `imaging` extra supplies dependencies for the built-in ABIDE, ADHD200,
and OASIS1 import commands. Those importers compute summary features from
the downloaded images. Use the generic CSV importer for prepared
measurements, including your own preprocessing outputs.

## NeuroClaimBench inputs

The [benchmark data guide](../benchmark/neuroclaimbench-v2.1/DATA_ACCESS.md)
lists the source cohorts and the files required to execute frozen tasks.
Acquire cohort data through the relevant provider under its access terms.

Use `confirm run` for analyses of newly prepared data. An exact benchmark
rerun requires files matching the published SHA-256 hashes, which identify
the participants, preprocessing outputs, partitions, and file serialization
used for each task.
