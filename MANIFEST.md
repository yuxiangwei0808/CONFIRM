# Research Output Manifest

Updated: 2026-07-23

## Active Documents

| File | Purpose |
|---|---|
| `IMPLEMENTATION_NOTES.md` | Full Stage 0-4 implementation and evidence policy. |
| `RESULTS_MANIFEST.md` | Reusable result layout, counts, and next commands. |
| `RESULTS_SHA256SUMS` | Integrity hashes for reusable result inputs. |
| `RESEARCH_BRIEF.md` | Short project framing and claim limits. |
| `EXTERNAL_BENCHMARK_RESULTS.md` | External-evidence coverage and limitations. |
| `review-stage/README.md` | Generated result-directory index. |
| `src/bench/README.md` | Active benchmark runner index. |

Integrity reports are stored with the run they audit; there is no root-level
audit that ambiguously applies to every run.

## Active Results

| Directory | Contents |
|---|---|
| `review-stage/literature-grounding-gpt55/` | Stage 0 PubMed grounding and feasibility artifacts. |
| `review-stage/initial-claims-all-gpt55/` | Stage 1 initial questions and frozen contracts. |
| `review-stage/confirm-gates-all-gpt55/` | Stage 2 deterministic gate results. |
| `review-stage/initial-claims-gpt55-retrospective-evidence-v1/` | Frozen retrospective evidence audit for initial claims. |
| `review-stage/claim-search-gpt55-sweep-v7/` | Completed 12-arm source-only feedback-search sweep. |
| `review-stage/claim-search-gpt55-control-r3-c5-v7/` | Completed matched structured-diagnosis versus generic-retry control. |
| `review-stage/claim-search-safety-gpt55-r1-c2-v7/` | Completed known-negative smoke run. |
| `review-stage/claim-search-safety-gpt55-r10-c10-v7/` | Completed maximum-budget known-negative stress run. |
| `review-stage/claim-search-gpt55-retrospective-evidence-v3/` | Completed frozen follow-up holdout/NACC evidence audit. |
| `review-stage/claim-search-gpt55-paper-analysis-v1/` | Derived paper tables, figures, case studies, audit, and hashes. |
| `data/neuroclaimbench/v2.1-source/` | Frozen source snapshot for the outcome-blind alignment pass. |
| `data/neuroclaimbench/v2.1/` | Canonical repaired NeuroClaimBench package. |
| `data/neuroclaimbench/pubmed-cache-v2.1/` | Audited immutable PubMed cache for invalidated v2.1 adjudications. |
| `review-stage/neuroclaimbench-v2.1/` | Current alignment, adjudication, reference, evaluation, analysis, and release artifacts. |
| `benchmark/neuroclaimbench-v2.1/` | Lean checksummed publishable benchmark release. |

Superseded results are recoverable under the ignored local directory
`review-stage/_archive_20260719_pre_v7/`. They are not active evidence.

## Active Launchers

| Script | Purpose |
|---|---|
| `scripts/launch_literature_claim_grounding.sh` | Stage 0 literature-grounded source generation. |
| `scripts/launch_initial_claim_drafting.sh` | Stage 1 LLM question and `ClaimContract` drafting. |
| `scripts/launch_confirm_gate_evaluation.sh` | Stage 2 deterministic gate evaluation. |
| `scripts/launch_initial_claim_evidence.sh` | Stage 2B frozen initial-claim evidence audit. |
| `scripts/launch_claim_search_control.sh` | Matched Stage 3 structured/generic control. |
| `scripts/launch_claim_search_sweep.sh` | Stage 3 source-data candidate-search matrix. |
| `scripts/launch_claim_search_safety.sh` | Stage 3 synthetic known-negative search. |
| `scripts/launch_claim_search_retrospective_evidence.sh` | Stage 4 frozen holdout/external audit. |
| `scripts/launch_claim_search_paper_analysis.sh` | Deterministic paper analyses and blinded-review workflow. |
| `scripts/launch_neuroclaimbench_build.sh` | Build/resume frozen source, alignment, cache, adjudication, package, and reference phases; API phases require `ALLOW_API=1`. |
| `scripts/launch_neuroclaimbench_evaluate.sh` | Evaluate every executable task locally. |
| `scripts/launch_neuroclaimbench_analyze.sh` | Build the compact schema, analyses, crosswalk, release, and audit payload locally. |

The canonical Stage 3 input is:

```text
review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json
```

Generated `review-stage/` outputs are ignored by git except for its index file.
External preparation remains isolated under `scripts/data_processing/`, with
dataset policy in `configs/external_datasets.yml` and evidence roles in
`configs/evidence_partitions.yml`.
