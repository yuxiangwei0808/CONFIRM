# NeuroClaimBench data dictionary

JSONL files contain one JSON object per line. IDs are stable strings;
SHA-256 fields identify file contents or serialized records.

## Core records

| File | Records | Contents |
|---|---:|---|
| `cases.jsonl` | 544 | Scientific questions, analysis plans, source identities, and record links |
| `references.jsonl` | 544 | Reference labels, evidence basis, agreement, and scoring eligibility |
| `tasks.jsonl` | 489 | Executable plans and identities of their prepared input cohorts |
| `outcomes.jsonl` | 489 | Recorded CONFIRM gate results and execution status |

### Cases

- `benchmark_case_id` identifies the case; `claim_uid` identifies its claim.
- `semantic_cluster_id` groups related claims for clustered analyses.
- `benchmark_track` is `scientific`, `synthetic_stress`, or `external_transfer`.
- `target_family`, `modality`, and `question` describe the scientific target.
- `contract` contains the analysis plan. `reference_id` links to its reference;
  `task_ids` lists its executable tasks and is empty for excluded cases.
- `provenance`, `aliases`, and alignment fields identify source records and
  the match between each question and its analysis plan.

### References

- `disposition` is `confirm` (reference supports the claim), `abstain`
  (reference withholds support), or `unresolved` (insufficient evidence).
- `basis` is `literature`, `constructed_control`, or `unresolved`.
- `strength` separates literature references into `strict` and `provisional`
  groups. The provisional group contains additional labels admitted under
  two-model direct-evidence rules. Neuroimaging experts finalized the
  literature labels. Controls use `constructed`; unresolved references use
  `evidence_gap`.
- `score_eligible` selects references for agreement metrics.
- `evidence_ids`, `derivation_rule`, `agreeing_models`, `agreement_pattern`,
  and `vote_counts` record the evidence and rule supporting each label.

`confirm` in a reference is a literature or control label. CONFIRM's reporting
decision is computed independently from the gate results in `outcomes.jsonl`.

### Tasks

- `task_id` identifies the task; `benchmark_case_id` links to its case.
- `dataset_id`, `discovery_cohort`, and `replication_cohorts` identify inputs.
- `contract` is the executable analysis plan; `contract_sha256` identifies it.
- `partition_hashes` identifies the required prepared files for tasks without
  a generator specification. `partition_paths` records their source paths.
- `generator_spec` describes control construction; `generator_artifact_hashes`
  identifies the associated construction artifacts.
- `evidence_role` and `evidence_freshness` describe the evidence partition.
- `code_sha`, `schema_version`, and `gate_policy_version` identify the
  implementation and policy used to define the task.

### Outcomes

- `task_id` joins the outcome to its task; `status` records execution status.
- `gate_verdict` contains the gate checks, diagnostic label, and rationale.
- `confirm_outcome` and `raw_final_label` record the Full-standard result.
- `error` describes a failed execution when present.
- `cohort_content_hashes` identifies analyzed input files.
  `generated_cohort_hashes` identifies generated files required by tasks
  with a `generator_spec`.
- `task_fingerprint` and `detailed_result_sha256` identify the task and its
  detailed result artifact; the package distributes the compact outcome.

## Results and supporting records

| File | Contents |
|---|---|
| `benchmark_splits.json` | Track memberships, adjudication candidates, and excluded-case IDs |
| `benchmark_results.csv` | Case-level references, outcomes, and evaluation coverage |
| `benchmark_summary.json` | Metrics by track, target, reference basis, and strength |
| `benchmark_strata_summary.csv` | Counts, rates, Wilson intervals, and cluster-bootstrap intervals by evaluation group |
| `target_reference_summary.csv` | The same summaries grouped by target family |
| `cluster_bootstrap_sensitivity.csv` | Wilson and semantic-cluster-bootstrap interval comparison |
| `reference_agreement_audit.csv` | Counts by evidence basis, strength, and model-agreement pattern |
| `unresolved_case_summary.csv` | Unresolved cases, agreement patterns, and recorded outcomes |
| `alignment_policy.json` | Rules for question-plan alignment and execution eligibility |
| `analysis_audit.json` | Alignment, evidence, execution-coverage, and result-consistency checks |
| `analysis_manifest.json` | Analysis input/output hashes and bootstrap settings |
| `v2_to_v2.1_crosswalk.csv` | Source-to-benchmark ID and contract-hash mappings |
| `feedback_parent_crosswalk.csv` | Links from follow-up experiment parent claims to benchmark cases |
| `feedback_reference_summary.csv` | Follow-up parent support grouped by reference category and search budget |
| `feedback_crosswalk_manifest.json` | Hashes and matching counts for the follow-up parent links |

Source paths and artifact hashes in these records describe the benchmark's
scientific provenance. The files needed for scoring are included in this
package. [Task execution](DATA_ACCESS.md) uses separately obtained prepared
cohort files.

## Package files

| File | Contents |
|---|---|
| `README.md` | Installation and quickstart |
| `USAGE.md` | Loading, metrics, scoring, and task execution |
| `DATA_ACCESS.md` | Source cohorts, table format, and exact-input requirements |
| `DATA_DICTIONARY.md` | File and field definitions |
| `benchmark_manifest.json` | Record filenames, hashes, schema version, counts, and source identities |
| `RELEASE_MANIFEST.json` | Benchmark identity, record counts, and distributed file inventory |
| `PACKAGE_MANIFEST.json` | Version, license, framework dependency, and core record hashes |
| `CITATION.cff` | Authors and paper citation |
| `LICENSE` | CC BY 4.0 license text |
| `SHA256SUMS` | Checksums of every distributed file except this checksum file |
