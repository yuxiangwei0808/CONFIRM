# Cohort data and task execution

The metadata package supports inspection and scoring of reporting decisions.
Executing a frozen task requires its prepared cohort files. Obtain source
cohort data through the relevant provider or study team under their access
terms. The CC BY 4.0 license covers this package's metadata and documentation.

## Source cohorts

The task inventory names ABIDE1, ABIDE2, ADHD200, ABCD, ADNI, ADNI_fMRI,
BSNIP2, COBRE, ChineseSZ, FBIRN, GSP, HCP, HCP_Aging, JH, NACC, OASIS3,
OASIS3_fMRI, UKB, and OpenNeuro ds000030. It also includes constructed stress
datasets and evidence partitions derived from those sources.

Use `dataset_id` in `tasks.jsonl` and `provenance` in `cases.jsonl` to identify
the source of each task. The metadata contains no participant tables,
participant-level control assignments, or preprocessing outputs.

## Exact benchmark inputs

Each task names its discovery and replication cohorts. Store their prepared
tables as `<cohort>.parquet`. The runner verifies the file bytes against the
published SHA-256 hashes before computing results.

| Task type | Required prepared files | Hash field |
|---|---|---|
| Task with no `generator_spec`, including fixed stress datasets | Named discovery and replication cohorts | `tasks.jsonl`: `partition_hashes` |
| Task with a `generator_spec`, including random-label controls | Frozen generated discovery and replication cohorts | `outcomes.jsonl`: `generated_cohort_hashes` |

Place all cohorts for a task together in one directory and pass that
directory as `--data-root`. Additional `--data-root` arguments search other
directories. Generator specifications record the construction rules, seeds,
and assignment hashes; executing these tasks requires the generated tables.

Exact reruns require the original prepared artifacts, including partition
membership and file serialization. Raw provider downloads require further
preparation and do not supply those frozen files. Missing or mismatched
inputs produce explicit error rows. The scorer counts errors as withheld
and reports them separately in its coverage counts.

## Analyze newly prepared data

Use `confirm run` with a new analysis plan for newly prepared cohorts.
Prepare one Parquet table per cohort with these columns:

| Column | Content |
|---|---|
| `subject_id` | Participant identifier |
| `cohort` | Cohort name matching the filename |
| `site` | Acquisition site or scanner batch |
| `age` | Numeric age |
| `sex` | `M` or `F` |
| Imaging measurements | Numeric columns with `smri_`, `pet_`, `fc_`, or `raw_` prefixes |

Include all additional variables named in the plan, such as `eTIV`,
diagnosis, or behavioral scores. Keep discovery, replication, and holdout
participants separate. `confirm demo --out outputs/demo` creates complete
synthetic tables and plans illustrating this format. The CONFIRM source
package also includes a CSV mapping template and conversion guide.

Report analyses of new cohorts with their own input identities and results.
The frozen benchmark records describe the published task inputs and outcomes.
