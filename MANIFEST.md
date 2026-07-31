# Research Output Manifest

Updated: 2026-07-30

This is the active-project map for the shared lab workspace. Every directory
belongs to one of four classes: **active** (used by the current code or
paper), **archived** (retained provenance, not current evidence), **generated**
(recreated locally and ignored), or **external baseline** (independent
comparison checkout).

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
| `README.md` | Lab entry point, setup, and navigation. |
| `docs/LAB_HANDOFF.md` | Local layout, archive, configuration, and baseline policy. |
| `docs/ARCHIVE_INDEX.md` | Archived-material index and restoration procedure. |

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

Superseded results are recoverable under ignored local directories matching
`review-stage/_archive_*/`. They are not active evidence; see
`docs/ARCHIVE_INDEX.md` for their replacements.

## Active source and data layout

| Path | Class | Purpose |
|---|---|---|
| `src/confirm/`, `tests/`, `scripts/`, `nbs/`, `nbs_data/` | active | CONFIRM implementation, validation, launchers, analysis, and preparation. |
| `configs/evidence_partitions.yml` | active | Portable prepared-cohort evidence layout. |
| `configs/external_datasets.example.yml` | active template | Copy to the ignored local configuration before external preparation. |
| `data/` | generated/local active | Restricted raw data, prepared cohorts, caches, and benchmark source packages. |
| `review-stage/` | generated/local active | Frozen experiment outputs listed above. |
| `benchmark/neuroclaimbench-v2.1/` | active | Lean checksummed benchmark metadata. |
| `external/NeuroClaw/`, `external/veritas/` | external baseline | Pinned comparison checkouts; not part of the default test suite. |
| `paper/` | active submodule | Manuscript source, active tables, and figures. |
| `docs/archive/`, `review-stage/_archive_*/` | archived | Recoverable historical material, excluded from active pipelines. |
| `.matplotlib/`, `build/`, `viz/` | generated | Caches or generated render output; not versioned. |

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
dataset policy in the ignored `configs/external_datasets.local.yml` (created
from the committed example) and evidence roles in `configs/evidence_partitions.yml`.
