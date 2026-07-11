# Implementation Notes

Updated: 2026-07-10

## Current Pipeline

The active CONFIRM workflow has three completed stages and one optional
feedback-loop stage.

1. Stage 0 creates literature-grounded claim seeds.
2. Stage 1 creates initial claim questions and drafts frozen `ClaimContract`s.
3. Stage 2 evaluates those contracts with unchanged CONFIRM gates.
4. Stage 3 optionally diagnoses failed claims and asks an LLM to propose
   connected follow-up candidates.

Final scientific verdicts are owned only by `confirm.verdict`. LLMs draft
questions, contracts, diagnoses, and candidate proposals; deterministic code
validates schema, evidence compatibility, anti-hacking constraints, and gate
outcomes.

## Stage 0: Literature Grounding

Launcher:

```bash
scripts/launch_literature_claim_grounding.sh
```

Default output:

```text
review-stage/literature-grounding-gpt55/
data/claims/literature_grounded_claims.csv
```

Stage 0 queries PubMed for the five target families, extracts structured
paper-derived claim seeds with GPT-5.5, and performs deterministic local
feasibility matching. A seed is emitted as executable only when its predictor,
outcome family, cohort pair, covariates, and group labels map to prepared local
columns. Invalid or unsupported seeds remain in feasibility artifacts and are
not treated as pipeline failures.

Current Stage 0 output:

| Item | Count |
|---|---:|
| PubMed records | 99 |
| extracted seeds | 274 |
| executable literature-grounded questions | 41 |

## Stage 1: Initial Claim Drafting

Launcher:

```bash
MAX_WORKERS=16 PARALLEL_BACKEND=thread scripts/launch_initial_claim_drafting.sh
```

Default output:

```text
review-stage/initial-claims-all-gpt55/
```

Inputs:

- fixed literature-grounded questions from
  `data/claims/literature_grounded_claims.csv`;
- GPT-5.5 proposed questions, defaulting to `NUM_CLAIMS=50` per target family.

Stage 1 writes:

- `claim_questions.jsonl`;
- `llm_question_prompts.jsonl`;
- `llm_question_responses.jsonl`;
- `llm_contract_prompts.jsonl`;
- `llm_contract_responses.jsonl`;
- `drafted_contracts.jsonl`;
- `draft_validation.json`;
- `draft_summary.json`;
- `accepted_contract_audit.json`.

Current full Stage 1 result:

| Source | Questions | Drafted contracts | Draft errors |
|---|---:|---:|---:|
| literature_grounded | 41 | 41 | 0 |
| llm_proposed | 250 | 248 | 2 |
| total | 291 | 289 | 2 |

The two remaining draft failures are non-draftable seeds: one ADHD subtype
contrast loses a control group after complete-case filtering, and one AD/aging
question asks a multi-level diagnosis-code contrast outside the binary
`ClaimContract` schema.

Important Stage 1 guardrails:

- GPT-5.5 structured output is parsed into Pydantic models and retried on
  schema or semantic-preflight failure.
- Disease case/control contrasts use runtime-derived `confirm_dx` when
  available.
- fMRI/FC contracts do not use `eTIV`; structural MRI/PET contracts may use it
  when shared by all selected cohorts.
- Literature-grounded contracts preserve their Stage 0 discovery and
  replication cohorts.
- Accepted contracts are audited for unsupported modalities and unsupported
  local measurements.

## Stage 2: CONFIRM Gate Evaluation

Launcher:

```bash
MAX_WORKERS=16 PARALLEL_BACKEND=process scripts/launch_confirm_gate_evaluation.sh
```

Default output:

```text
review-stage/confirm-gates-all-gpt55/
```

Stage 2 reads `drafted_contracts.jsonl` and evaluates each frozen contract once.
It writes:

- `combined_benchmark_results.json`;
- `claim_gate_audit.csv`;
- `claims.csv`;
- `summary.json`.

Current full Stage 2 result:

| Final label | Count |
|---|---:|
| confirmed | 74 |
| fragile | 169 |
| non_replicated | 38 |
| under_powered | 8 |
| execution errors | 0 |

By target family:

| Target | confirmed | fragile | non_replicated | under_powered |
|---|---:|---:|---:|---:|
| normative_fmri | 17 | 27 | 9 | 0 |
| adhd | 16 | 30 | 13 | 0 |
| asd | 0 | 59 | 0 | 0 |
| ad_aging | 38 | 19 | 4 | 0 |
| psychosis | 3 | 34 | 12 | 8 |

By source:

| Source | confirmed | fragile | non_replicated | under_powered |
|---|---:|---:|---:|---:|
| literature_grounded | 12 | 27 | 2 | 0 |
| llm_proposed | 62 | 142 | 36 | 8 |

## Target Families And Data

The active full pipeline uses target families:

- `normative_fmri`;
- `adhd`;
- `asd`;
- `ad_aging`;
- `psychosis`.

Prepared data roots:

```text
data/prepared_data/evidence_partitions/benchmark_ready/cohorts/
data/prepared_data/evidence_partitions/cohorts/
```

Stage 0, Stage 1, and Stage 2 use only
`evidence_partitions/benchmark_ready/cohorts/`, which contains the main
discovery/replication pairs. Catalog construction filters `_HOLDOUT` and
`_EXTERNAL` partitions even if a broader root is supplied, and Stage 2 rejects
contracts that reference those reserved roles. The general `cohorts/` root is
used by the feedback-loop evaluator for excluded evidence.

Current evidence families include:

| Target | Main evidence families | Main modalities |
|---|---|---|
| normative_fmri | UKB, HCP, HCP_Aging, ABCD | fMRI/functional connectivity and phenotypes |
| adhd | ADHD200, ABCD | fMRI/functional connectivity |
| asd | ABIDE1, ABIDE2 | fMRI/functional connectivity |
| ad_aging | ADNI, OASIS3, ADNI_fMRI, OASIS3_fMRI | structural MRI, PET, fMRI/FC |
| psychosis | COBRE, FBIRN, BSNIP, BSNIP2, ChineseSZ, JH, Olin_SZ | fMRI/functional connectivity |

Runtime-derived `confirm_dx` normalizes disease labels for ADHD, ASD,
AD/dementia, and psychosis contrasts without rewriting parquet files.

## Gate Flow

Each claim contract goes through the same deterministic gate ladder:

- contract/schema and semantic preflight;
- search provenance and multiplicity;
- confound and confound-completeness checks;
- power;
- multiverse consistency;
- discovery-to-replication agreement.

Labels mean:

- `confirmed`: all required gates passed;
- `fragile`: effect is not stable enough across multiplicity/multiverse and
  related robustness checks;
- `non_replicated`: discovery signal did not replicate;
- `under_powered`: power gate failed.

## Optional Feedback Loop

The feedback loop starts from Stage 2 outputs:

```bash
MAX_ROUNDS=1 MAX_CANDIDATES=2 scripts/launch_claim_search_fullscale.sh
```

Default input:

```text
review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json
```

Claim search uses failed-claim evidence for diagnosis and prompt context. LLM
candidates must remain connected to the original claim and pass deterministic
validation before evaluation. Same-data adaptive support is labeled separately
from holdout or external confirmation when those evidence paths are supplied.

Evidence labels are explicit:

- `exploratory_confirmed`: adaptive candidate passed on reused source data;
- `holdout_confirmed`: candidate passed on excluded internal holdout evidence;
- `external_confirmed`: candidate passed on a predeclared primary external set;
- secondary external sets are robustness checks and cannot upgrade the label.

## Evidence Partitions And External Data

`configs/evidence_partitions.yml` defines discovery, replication, holdout, and
external roles. Materialized records and schema metadata are written to:

```text
data/prepared_data/evidence_partitions/manifest.json
data/prepared_data/evidence_partitions/cohorts/
```

Internal holdouts are split again into distinct evaluation discovery and
replication files. The current manifest has 104 records and no subject overlap.
External evidence is selected by `ClaimContract`, not target name alone. A set
must match target, modality, feature family, predictor/group support, required
columns, outcome family, and observed group levels on both partitions. The
lowest-priority-number compatible primary set is selected before results are
read. A secondary set remains a robustness evaluation.

The current active external tables are NACC sMRI and ds000030/CNP sMRI. NACC
uses canonical FreeSurfer-style names and mm3 volumes. In the current 289-claim
Stage 2 inventory, schema-only primary external coverage is 35 contracts, all
in AD/aging; this is recorded in:

```text
data/prepared_data/evidence_partitions/audits/stage2_contract_coverage.json
data/prepared_data/evidence_partitions/audits/stage2_external_contract_coverage.csv
```

The zero current external coverage for normative fMRI, ADHD, ASD, and psychosis
fMRI is a data-preparation state, not a confirmation result. Candidate external
datasets are defined in `configs/external_datasets.yml`:

| Dataset | Prepared modality | Status/use |
|---|---|---|
| EHBS | fMRI network descriptors; sMRI | normative/aging candidate |
| LA5C/CNP | fMRI global FC | primary psychosis candidate |
| ADHD-Suijing, PKU | fMRI global FC; sMRI | quarantined for phenotype/age and network-parity issues |
| Olin ASD/SZ | fMRI global FC; sMRI | quarantined until authoritative labels exist |
| BLSA | existing static FC; sMRI | aging candidate |
| AIBL | sMRI | AD/aging secondary robustness set |
| Shile/Nanjing | sMRI | existing outputs reused but metadata-quarantined |
| PK_MPRC | fMRI global FC; sMRI | psychosis robustness candidate |

Prepared filenames include modality (`*_fMRI.parquet`, `*_sMRI.parquet`) so
syncing one modality cannot overwrite the other. Quarantined files are never
promoted automatically.

## External Preparation On Arcdev

The shell-only launchers deploy versioned code under
`/data/users1/ywei/confirm_external_prep/runs/$RUN_ID`, start work with `nohup`,
print the remote PID/log, and follow progress by default:

Set `SSH_HOST` to any host or alias from `~/.ssh/config`. It defaults to
`arcdev`; the older `ARCDEV_HOST` variable remains a backward-compatible
alias. For example, `SSH_HOST=arcdev-gpu ...` uses that alias for deployment,
launch, monitoring instructions, and result synchronization.
The detached worker starts through a login Bash shell so site-specific Lmod
paths match a direct SSH login.

```bash
SSH_HOST=arcdev RUN_ID=external-audit-20260710 DATASETS=all \
scripts/data_processing/audit_external_datasets_arcdev.sh

RUN_ID=external-fmri-20260710 \
DATASETS=EHBS,LA5C,ADHD_Suijing,PKU_ADHD,Olin_ASD_SZ,BLSA,Shile_Nanjing,PK_MPRC \
MAX_WORKERS=8 scripts/data_processing/launch_external_fmri_arcdev.sh

RUN_ID=external-fs-canary-20260710 \
REMOTE_FASTSURFER_HOME=/path/to/FastSurfer \
DATASETS=EHBS,ADHD_Suijing,PKU_ADHD,Olin_ASD_SZ,BLSA,AIBL,Shile_Nanjing,PK_MPRC \
GPU_IDS=0,1 THREADS_PER_JOB=4 LIMIT=2 \
scripts/data_processing/launch_external_freesurfer_arcdev.sh

RUN_ID=external-fmri-20260710 scripts/data_processing/sync_external_evidence_arcdev.sh
```

FastSurfer is the default and runs one sequential queue per GPU. Set
`RECON_ENGINE=recon-all` for the explicit CPU override. The remote worker runs
`module load freesurfer/7.4.1` by default (`FREESURFER_MODULE` changes the
module name). `REMOTE_FREESURFER_HOME` may override the home set by the module,
and the selected installation hard-fails unless `recon-all -version` reports
7.4.1. FastSurfer runs also require the manually supplied
`REMOTE_FASTSURFER_HOME`. Partial outputs are preserved.
Retrying requires `RETRY_FAILED=1`, which archives the previous attempt before
a new one starts.

FastSurfer uses `REMOTE_FASTSURFER_PYTHON`, defaulting to
`$REMOTE_FASTSURFER_HOME/.venv/bin/python`. Preflight requires Python 3.10 or
newer and imports `nibabel`, `torch`, and `yacs` before any subject starts. The
virtual environment's `bin` directory is placed first on `PATH` because
`run_fastsurfer.sh` invokes `python3` internally.

Run IDs are immutable by default. Reusing a run ID with an existing stage log,
or launching two workers for the same run and stage, is rejected before log
redirection. `ALLOW_RUN_ID_REUSE=1` is an explicit debugging override. Any
failed subject writes `status=completed_with_failures` and causes a nonzero
stage exit; the launcher no longer prints a successful completion for a failed
canary.

The canonical default is `EXPECTED_FREESURFER_VERSION=7.4.1`. A different
installed version can be selected explicitly for a separate, consistently
processed run. The selected
version is validated before processing and recorded in status artifacts and
per-subject completion receipts; versions must not be mixed within an evidence
set.

Completion requires a zero exit code, a receipt, `aseg.stats`, bilateral DKT
cortical stats, and bilateral white surfaces. Aggregation reconciles exactly to
strict completion receipts and emits canonical volume features in mm3 plus
separately named thickness measures. Existing Shile outputs are adopted only
after the same artifact audit and receive an explicit legacy-adoption receipt.

Sync first writes candidates under
`data/prepared_data/external_candidates/$RUN_ID/`. Promotion is explicit:

```bash
RUN_ID=external-fmri-20260710 PROMOTE=1 \
scripts/data_processing/sync_external_evidence_arcdev.sh
```

Promotion validates canonical schemas, archives replaced active files, rebuilds
the evidence manifest with overlap checks, and regenerates the 289-contract
external coverage report.

## Active Result Hygiene

Active result folders are:

```text
review-stage/literature-grounding-gpt55/
review-stage/initial-claims-all-gpt55/
review-stage/confirm-gates-all-gpt55/
```

Superseded literature-only and earlier feedback-loop runs were moved to:

```text
review-stage/_archive_20260702_pipeline_cleanup/
```

Older historical runs remain under earlier `_archive_*` folders. Archive folders
are ignored by git and are recoverable local history, not active evidence.
