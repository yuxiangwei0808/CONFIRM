# Implementation Notes

Updated: 2026-07-23

## Current Pipeline

The active CONFIRM workflow has three completed core stages, an initial-claim
excluded-evidence audit, one optional feedback-loop search stage, and a frozen
feedback-output excluded-evidence audit.

1. Stage 0 creates literature-grounded claim seeds.
2. Stage 1 creates initial claim questions and drafts frozen `ClaimContract`s.
3. Stage 2 evaluates those contracts with unchanged CONFIRM gates.
4. Stage 2B freezes the initial claims and evaluates them on compatible
   internal holdout and external evidence.
5. Stage 3 optionally diagnoses failed claims and asks an LLM to propose
   connected follow-up candidates.
6. Stage 4 freezes every final internally supported Stage 3 candidate and
   evaluates each compatible excluded-evidence pair without outcome-dependent
   routing.

Final scientific verdicts are owned only by `confirm.verdict`. LLMs draft
questions, contracts, diagnoses, and candidate proposals; deterministic code
validates schema, evidence compatibility, anti-hacking constraints, and gate
outcomes.

## Claim-Evaluation And Feedback Baselines

The comparison layer is separate from benchmark construction and does not
modify NeuroClaimBench v2.1 or feedback sweep v7.

Claim-evaluation baseline launcher:

```bash
PHASE=protocol scripts/launch_claim_evaluation_baselines.sh
PHASE=significance scripts/launch_claim_evaluation_baselines.sh
PHASE=llm_judge MODEL=openai:gpt-5.5 MAX_WORKERS=8 \
  scripts/launch_claim_evaluation_baselines.sh
PHASE=finalize scripts/launch_claim_evaluation_baselines.sh
PHASE=analyze scripts/launch_claim_evaluation_baselines.sh
```

The conventional baseline requires unadjusted significance in discovery and
every replication cohort with matching direction. The direct LLM judge sees
an anonymized frozen contract and numerical evidence but not CONFIRM gate
decisions or benchmark references. Benchmark references are joined only in
`PHASE=analyze`, after both baseline decision files are frozen.

Self-Refine feedback-control launcher:

```bash
TRACK=all scripts/launch_claim_search_self_refine.sh
scripts/launch_feedback_baseline_analysis.sh
```

Self-Refine uses one critique call and one refinement call per active round.
Its critique receives contracts, immutable constraints, executable source-data
metadata, and only a binary unsupported status. Candidate outputs use the
frozen v7 structured proposal schema and the same deterministic validation,
execution, deduplication, and final multiplicity policy as the existing
failure-specific and failure-blind arms. Exact provider token metadata is
recorded for this new arm. Monetary cost is reported only when the provider
returns it; the code does not infer a price from a mutable pricing table. The
reused v7 arms predate usage instrumentation, so their exact call counts are
reported but token and cost totals are marked unavailable.

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

The optional reporting threshold does not rerun or weaken any gate:

```bash
MINIMUM_EVIDENCE_TIER=discovery scripts/launch_confirm_gate_evaluation.sh
MINIMUM_EVIDENCE_TIER=replicated scripts/launch_confirm_gate_evaluation.sh
MINIMUM_EVIDENCE_TIER=confirmed scripts/launch_confirm_gate_evaluation.sh
```

`confirmed` is the default and reproduces the original strict verdict. The
other choices add a separate `support_decision`: `discovery` requires valid
search provenance, measured-confound checks, and multiplicity; `replicated`
also requires replication. `final_label` and `gate_verdict_label` always retain
the strict CONFIRM result used by failure diagnosis and feedback search.

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

## Stage 2B: Initial-Claim Retrospective Evidence Audit

Launcher:

```bash
scripts/launch_initial_claim_evidence.sh
```

Default output:

```text
review-stage/initial-claims-gpt55-retrospective-evidence-v1/
```

This audit freezes all 289 Stage 2 contracts before consulting excluded
evidence. `preflight` selects compatible internal holdout and external pairs
without reading outcomes, hashes the query plan, and deduplicates identical
executable contracts. `evaluate` runs only frozen tasks; `summarize` maps each
deduplicated result back to every original claim and reports matched
internal-versus-excluded outcomes. No LLM/API call is reachable from this
launcher.

```bash
OUT=review-stage/initial-claims-gpt55-retrospective-evidence-v1 \
PHASE=freeze scripts/launch_initial_claim_evidence.sh

OUT=review-stage/initial-claims-gpt55-retrospective-evidence-v1 \
PHASE=preflight MAX_WORKERS=8 scripts/launch_initial_claim_evidence.sh

OUT=review-stage/initial-claims-gpt55-retrospective-evidence-v1 \
PHASE=evaluate MAX_WORKERS=8 scripts/launch_initial_claim_evidence.sh

OUT=review-stage/initial-claims-gpt55-retrospective-evidence-v1 \
PHASE=summarize scripts/launch_initial_claim_evidence.sh
```

Current claim-level results:

| Evidence scope | Eligible/evaluated claims | Supported claims | Support rate |
|---|---:|---:|---:|
| Stage 2 discovery/replication | 289 | 74 | 25.6% |
| internal holdout | 209 | 21 | 10.0% |
| external NACC | 35 | 24 | 68.6% |

Among the 74 Stage 2-confirmed claims, 65 were holdout-compatible and 16 passed
holdout; 27 were external-compatible and 18 passed external evaluation. Among
initially non-confirmed claims, 5 of 144 holdout-compatible claims and 6 of 8
external-compatible claims passed the corresponding excluded evaluation.

Exact execution deduplication reduced the 209 holdout claim references to 204
tasks and the 35 external references to 22 tasks. There were 21 supported
holdout tasks and 12 supported external tasks. External coverage is limited to
NACC-compatible AD/aging structural-MRI contracts; it does not cover the other
four target families in this inventory.

All current holdout and NACC evidence is marked
`evidence_freshness=previously_queried` and
`final_confirmation_eligible=false`. These counts measure retrospective
concordance, not fresh or prospective confirmation.

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

The feedback loop starts from all 215 failed Stage 2 contracts. Search inclusion
does not depend on holdout or external availability, and sweep arms receive
source discovery/replication data only. The five ASD parents whose Stage 2
source already used `ABIDE1_HOLDOUT` remain searchable, but that partition is
recorded in their source-evidence ledger and cannot later count as excluded
evidence.

Default input:

```text
review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json
```

For each parent and round, GPT-5.5 returns up to the configured candidate count
as Pydantic-validated executable contracts. A partially valid response is not
backfilled; malformed or wholly invalid responses may be retried. Deterministic
code rejects duplicates, no-op rewrites, unrelated cohorts/modalities, changed
predictors or group contrasts, reversed directions, weaker gates, removed
covariates, unsupported fields, and invalid inclusion filters. Added confounds
must exist in every source cohort and include a scientific justification.
Declared transform labels are descriptive. The executable contract delta is
inferred independently, and transform disagreement is an audit warning rather
than an eligibility rule.

Outcome identity follows the exact executable column because `_z`-suffixed
columns in the prepared datasets are not consistently deterministic transforms
of their unsuffixed names. The LLM may propose creative alternative outcomes or
`multivariate_pattern` contracts within the original modality, including a
scalar-to-brainwide transition. A brainwide candidate must resolve to at least
three outcomes shared across every source cohort; two-point Pearson pattern
correlations are otherwise mechanically `-1` or `1`. For a scalar-to-brainwide
transition, the pipeline raises the effective `pattern_corr_min` to at least
`0.5` and records that normalization in `policy_adjustments`; the LLM is not
allowed to modify gates itself. Inclusion changes are judged against feasible
parent-data predicates independently of the declared or inferred transform.

Every valid unique candidate is evaluated on source data in deterministic
round/candidate order. All provisional passes are retained while failed
candidates drive the next round. Round 2 and later proposals must reference
failed candidates from the immediately preceding round through
`responds_to_candidate_ids`. Search stops when the round has no failed
candidate, no valid proposal remains, generation fails, or the round budget is
exhausted. Passing is not forced.

Multiplicity is finalized after each round. Every provisional pass is
re-adjudicated with:

```text
final_family_size = max(
    candidate_declared_family_size,
    parent_declared_family_size + cumulative_unique_hypotheses_tested,
)
```

A scalar candidate contributes one adaptive hypothesis. A brainwide candidate
contributes its distinct regional hypotheses plus one pattern hypothesis;
overlapping exact hypotheses are counted once within a lineage. Artifacts retain
candidate counts and hypothesis counts separately.

An early pass that fails under the realized burden is retracted and becomes
failure context for the next round. Rechecks do not add hypotheses. Only passes
under the final realized family size enter
`internally_supported_candidate_ids`, and that exact effective contract is
frozen for excluded-evidence evaluation.

Search artifacts checkpoint complete parent states atomically. Resume verifies
source, config, model, prompt, schema, implementation-file,
evidence-manifest, and partition hashes before reusing a checkpoint, verifies
each parent-checkpoint content hash, and does not repeat completed LLM calls.
Each state records a hashed source-evidence ledger reconstructed from the
frozen contract when Stage 2 did not serialize execution paths. LLM records
separate schema-attempt and deterministic-validation-retry indices and a typed
`retry_kind`; parsed candidates discarded before a whole-response retry remain
in `unretained_candidate_attempts` with their deterministic validation result.
Source ledgers prefer parquet content hashes and record the hash kind; run
provenance also reports duplicate logical partition IDs and rejects conflicting
duplicates. Aggregate JSON/CSV artifacts are rewritten every 10 completed
parents by default; per-parent durability is immediate. Each arm must finish
exactly 215 parent states and must contain zero excluded-evidence queries.

Current-data and retrospective evidence labels are explicit:

- `exploratory_confirmed`: adaptive candidate passed on reused source data;
- `retrospective_holdout_supported`: frozen parent/candidate contract passed on
  a compatible internal holdout pair;
- `retrospective_external_supported`: frozen parent/candidate contract passed
  on a compatible NACC/CNP pair;
- `excluded_evidence_unavailable`: no outcome-blind compatible excluded pair;
- `final_confirmation_eligible=false` for every retrospective audit result.

The current internal holdouts, NACC, and CNP have been queried in prior
development runs. Audit artifacts therefore set
`evidence_freshness=previously_queried`; they are retrospective benchmark
evidence, not pristine prospective confirmation.

The completed v7 failure-specific-versus-failure-blind control uses the exact
completed sweep source and structured `R3/C5` artifact. The launcher runs only
the missing failure-blind arm (stored under the legacy path name
`generic_retry`):

```bash
OUT=review-stage/claim-search-gpt55-control-r3-c5-v7 \
SWEEP=review-stage/claim-search-gpt55-sweep-v7 \
MAX_WORKERS=24 scripts/launch_claim_search_control.sh
```

The control withholds gate-specific localization and evidence from the
failure-blind retry; both arms retain the same source, catalog, model, budget,
validator, and multiplicity policy. `control_summary.json` and
`control_parent_pairs.csv` verify those invariants and report paired descriptive
differences. One realization is not a causal estimate of diagnosis benefit.
The older v6 control is not interchangeable because its source and
search-implementation hashes differ. The control summary compares the recorded
search-relevant implementation files and separately reports the complete hashes;
`confirm.frozen_evidence` is analysis-only and does not force a search rerun.
The completed control has 215 matched parents and zero execution errors or
excluded-evidence queries. Structured diagnosis uses 653 calls and supports 70
candidates across 24 parents; failure-blind retry's completed trace uses 693
calls and supports 49 candidates across 20 parents. Failure-blind retry also records 72
superseded transport-failed attempts, for 765 total API attempts. Paired support
cells are 17 both, seven structured-only, three generic-only, and 188 neither.
The legacy structured artifact lacks the newer explicit search-only fingerprint
field, so the result is reported with that provenance warning and remains
descriptive.

Known-negative smoke or stress run:

```bash
OUT=review-stage/claim-search-safety-gpt55-r1-c2-v7 \
MAX_ROUNDS=1 MAX_CANDIDATES=2 MAX_WORKERS=8 \
scripts/launch_claim_search_safety.sh

OUT=review-stage/claim-search-safety-gpt55-r10-c10-v7 \
MAX_ROUNDS=10 MAX_CANDIDATES=10 MAX_WORKERS=8 \
scripts/launch_claim_search_safety.sh
```

Synthetic p-fishing inventories use exact prepared feature columns. Suffixes
alone are not used to collapse hypotheses because those columns can contain
different measurements.

Twelve-arm descriptive sweep:

```bash
OUT=review-stage/claim-search-gpt55-sweep-v7 \
ROUNDS="1 3 5 10" CANDIDATES="2 5 10" \
BUILD_EVIDENCE_PARTITIONS=off MAX_WORKERS=8 \
scripts/launch_claim_search_sweep.sh
```

The matrix summary reports each arm independently and does not select a winner,
write `selected_config.json`, pool repeated claims across arms, or use excluded
outcomes. Budget differences are descriptive for one GPT-5.5 realization.

## Stage 4: Frozen Retrospective Evidence Audit

The audit launcher contains no LLM/API path:

```bash
scripts/launch_claim_search_retrospective_evidence.sh
```

It runs four explicit phases. `freeze` reads the 215 hash-checked atomic parent
checkpoints plus `run_provenance.json` and the finalized matrix summary. It
streams the monolithic artifact hash without loading the multi-gigabyte JSON
tail, reconciles every arm's complete adaptive search funnel, and writes each
unmodified LLM response once to
`frozen_llm_responses.jsonl`. Every final internally supported candidate is
frozen; no `selected_candidate_id` is used. `preflight` checks schema, complete
cases, group counts, rank, condition number, modality, outcome family, units,
and partition overlap without reading effect or gate outcomes. It schedules a
compatible internal holdout pair unless that evidence was already used by the
lineage, and schedules every compatible external evidence set independently.
It then hashes the complete outcome-blind query plan. `evaluate` executes only
tasks in that plan with atomic checkpoints and a deterministic hash-chained
query ledger. `summarize` reports each arm, target, source, transform, and
external dataset separately.

```bash
OUT=review-stage/claim-search-gpt55-retrospective-evidence-v3 \
SWEEP=review-stage/claim-search-gpt55-sweep-v7 \
PHASE=freeze scripts/launch_claim_search_retrospective_evidence.sh

OUT=review-stage/claim-search-gpt55-retrospective-evidence-v3 \
SWEEP=review-stage/claim-search-gpt55-sweep-v7 \
PHASE=preflight MAX_WORKERS=8 \
scripts/launch_claim_search_retrospective_evidence.sh
```

After reviewing `preflight_summary.json` and `evidence_query_plan.jsonl`:

```bash
OUT=review-stage/claim-search-gpt55-retrospective-evidence-v3 \
SWEEP=review-stage/claim-search-gpt55-sweep-v7 \
PHASE=evaluate MAX_WORKERS=8 \
scripts/launch_claim_search_retrospective_evidence.sh

OUT=review-stage/claim-search-gpt55-retrospective-evidence-v3 \
SWEEP=review-stage/claim-search-gpt55-sweep-v7 \
PHASE=summarize scripts/launch_claim_search_retrospective_evidence.sh
```

Every later phase verifies the sweep hashes, implementation hashes, evidence
manifest hash, and all partition hashes. Exact executable tasks are deduplicated
only when the effective contract, inclusion, evidence pair, family size, and
partition hashes are identical; each result maps back to every originating arm
event. Effective family size is never changed for deduplication. Failed parents
are linked from the separately frozen Stage 2B audit and are not rerun here.

The audit may report retrospective survival, candidate-only support, and
candidate-generation yield. It may not claim fresh confirmation, causal
feedback-loop improvement, a winning budget, or general external validity.
Known-negative safety is assessed only by the separate synthetic search and
frozen-evidence path; one GPT-5.5 realization does not establish general safety.

## Paper Analysis Layer

All statistical, tabular, and figure generation remains outside production
code in `nbs/`. The predeclared reference is `R3/C5`; all 12 arms are
descriptive robustness analyses. Sweep, deterministic novelty, and evidence
analyses can run after the sweep and freeze are complete:

```bash
PHASE=sweep scripts/launch_claim_search_paper_analysis.sh
PHASE=novelty-metrics scripts/launch_claim_search_paper_analysis.sh
PHASE=evidence scripts/launch_claim_search_paper_analysis.sh
```

The matched packet phase uses the completed v7 generic-retry control:

```bash
PHASE=novelty scripts/launch_claim_search_paper_analysis.sh
```

After three reviewers complete the 600-row rating template:

```bash
PHASE=ratings \
RATINGS=review-stage/claim-search-gpt55-paper-analysis-v1/rating_template.csv \
scripts/launch_claim_search_paper_analysis.sh
```

Outputs stay under
`review-stage/claim-search-gpt55-paper-analysis-v1/` until reviewed. The
analysis manifest hashes every input and output and records that budget effects
are not causal, internal support is adaptive, excluded evidence is previously
queried, and novelty is parent-relative rather than literature-wide.

The safety replay uses the same freezer with its own generated manifest:

```bash
SAFETY=review-stage/claim-search-safety-gpt55-r10-c10-v7
OUT=$SAFETY/retrospective-evidence \
SWEEP=$SAFETY EVIDENCE_MANIFEST=$SAFETY/data/manifest.json \
EVIDENCE_ROOTS=$SAFETY/data/cohorts SOURCE_ROOTS=$SAFETY/data/cohorts \
ALLOW_NONREFERENCE_COUNTS=on PHASE=freeze \
scripts/launch_claim_search_retrospective_evidence.sh
```

Run the same command with `PHASE=preflight`, inspect the frozen query plan, then
run `PHASE=evaluate` and `PHASE=summarize`. Safety summaries are stratified by
synthetic failure family and report internal, holdout, external, unavailable,
non-identifiable, and execution-error risk counts separately.

## Evidence Partitions And External Data

`configs/evidence_partitions.yml` defines discovery, replication, holdout, and
external roles. Materialized records and schema metadata are written to:

```text
data/prepared_data/evidence_partitions/manifest.json
data/prepared_data/evidence_partitions/cohorts/
```

Internal holdouts are split again into distinct evaluation discovery and
replication files. The current manifest has 104 records and no subject overlap.
Excluded evaluation requires the exact materialized partition named by the
manifest; it cannot fall back to a base-cohort alias. A source contract that
already used a holdout partition cannot reuse that internal holdout, although a
separate compatible external set may still be used. This excludes the five
current ASD Stage 2 contracts that used `ABIDE1_HOLDOUT` from internal-holdout
confirmation.
External evidence is selected by `ClaimContract`, not target name alone. A set
must match target, modality, feature family, predictor/group support, required
columns, outcome family, and observed group levels on both partitions. The
frozen Stage 4 audit schedules every compatible set before results are read and
reports each external dataset separately. Primary/secondary roles remain
descriptive manifest metadata and do not cause outcome-dependent routing.

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
datasets are defined in the ignored local file
`configs/external_datasets.local.yml`, initialized from
`configs/external_datasets.example.yml`:

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
`$REMOTE_ROOT/runs/$RUN_ID`, start work with `nohup`, print the remote PID/log,
and follow progress by default. Set `REMOTE_ROOT` and `REMOTE_PYTHON` for the
target lab environment before launching:

Set `SSH_HOST` to any host or alias from `~/.ssh/config`. It defaults to
`arcdev`; the older `ARCDEV_HOST` variable remains a backward-compatible
alias. For example, `SSH_HOST=arcdev-gpu ...` uses that alias for deployment,
launch, monitoring instructions, and result synchronization.
The detached worker starts through a login Bash shell so site-specific Lmod
paths match a direct SSH login.

```bash
SSH_HOST=arcdev REMOTE_ROOT=/shared/confirm_external_prep \
REMOTE_PYTHON=/shared/envs/confirm/bin/python RUN_ID=external-audit-20260710 DATASETS=all \
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

## NeuroClaimBench v2.1 Outcome-Blind Repair

NeuroClaimBench v2.1 is the only active benchmark version. Its frozen
pre-alignment input is stored as `data/neuroclaimbench/v2.1-source/`; this
source snapshot is a construction input, not a second benchmark release.
Question-contract alignment does not consult CONFIRM outcomes, labels,
p-values, effect estimates, adjudication votes, or feedback-search results:

1. `launch_neuroclaimbench_build.sh` with `PHASE=source` assembles the frozen
   source snapshot. The same launcher with `PHASE=alignment` then freezes the repair policy and
   runs deterministic alignment over every source item. Low sample size is not
   an exclusion; requested groups/outcomes, full rank, required columns, and
   positive residual degrees of freedom determine executability.
2. Gemini 3.5 Flash acts as an outcome-blind eligibility adjudicator for every
   literature-derived executable item. It receives only the canonical
   question, executable contract, cohort schemas, category levels, and sample
   counts. Deterministic rules remain the only contract-repair authority, but
   Gemini can classify an otherwise aligned item as ambiguous. This affected
   40 items; they remain unresolved and are not scored.
3. `launch_neuroclaimbench_build.sh` with `PHASE=package` applies the frozen deterministic repair
   manifest, hashes canonical questions separately from executable contracts,
   hashes every evidence parquet, and records assignment and selected-row hashes
   for generated controls. Changed literature contracts lose stale
   adjudications; unchanged contract-plus-question identities retain them.
4. The PubMed cache is rebuilt only for changed literature items. Exact old
   query results may be copied from the audited v1 cache, but a changed query is
   a cache miss. Offline GPT-5.5, Gemini 3.5 Flash, and Claude Opus 4.8 voting
   then refreshes only invalidated adjudications.
5. `launch_neuroclaimbench_build.sh` with `PHASE=reference` derives orthogonal
   `reference_basis` and `reference_strength` fields. Constructed controls use
   `constructed_control/constructed`, never a strict literature label.
6. `launch_neuroclaimbench_evaluate.sh` reruns every executable v2.1 task
   under one current CONFIRM revision. Checkpoints are reusable only when the
   complete contract, question, evidence, generator, schema, and gate-policy
   fingerprint matches.
7. `launch_neuroclaimbench_analyze.sh` produces tiered
   literature/constructed-control summaries, clustered uncertainty,
   exact pre-repair feedback joins, and a checksummed metadata-only release.

The primary scientific result combines strict and provisional literature
references within each split. Strict-only and provisional-only results are
sensitivity analyses. NACC and CNP are reported separately, constructed random
controls are never pooled with external literature references, and unresolved
claims receive verdict distributions but no accuracy score. Version 2.1 is a
retrospective benchmark revision whose policy is frozen before the v2.1 gate
rerun; it is not prospective validation.

The local result-preserving launch order is:

```bash
PHASE=package scripts/launch_neuroclaimbench_build.sh
PHASE=reference scripts/launch_neuroclaimbench_build.sh
PHASE=all MAX_WORKERS=8 scripts/launch_neuroclaimbench_evaluate.sh
PHASE=all scripts/launch_neuroclaimbench_analyze.sh
```

API and PubMed phases require `ALLOW_API=1`. The v2.1 cache workflow does not
reuse an older cache by default. An explicitly
provided reuse cache may contribute only exact query-result keys; packets and
votes are never reused for changed identities.

## Active Result Hygiene

Active result folders are:

```text
review-stage/literature-grounding-gpt55/
review-stage/initial-claims-all-gpt55/
review-stage/confirm-gates-all-gpt55/
review-stage/initial-claims-gpt55-retrospective-evidence-v1/
review-stage/claim-search-gpt55-control-r3-c5-v7/
review-stage/claim-search-gpt55-sweep-v7/
review-stage/claim-search-safety-gpt55-r10-c10-v7/
review-stage/claim-search-gpt55-retrospective-evidence-v3/
review-stage/claim-search-gpt55-paper-analysis-v1/
review-stage/neuroclaimbench-v2.1/alignment/
review-stage/neuroclaimbench-v2.1/adjudication/
review-stage/neuroclaimbench-v2.1/reference/
review-stage/neuroclaimbench-v2.1/results/
review-stage/neuroclaimbench-v2.1/analysis/
review-stage/neuroclaimbench-v2.1/feedback-crosswalk/
```

Superseded NeuroClaimBench versions and development audits are not active
workspace artifacts. Primary reusable artifact hashes are frozen in
`RESULTS_SHA256SUMS`.
