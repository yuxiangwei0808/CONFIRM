# Research Output Manifest

Updated: 2026-07-16

## Active Documents

| File | Purpose |
|---|---|
| `IMPLEMENTATION_NOTES.md` | Current Stage 0-3 implementation and evidence policy. |
| `RESULTS_MANIFEST.md` | Active result layout, counts, and launch commands. |
| `RESEARCH_BRIEF.md` | Short project framing. |
| `EXTERNAL_BENCHMARK_RESULTS.md` | External-evidence coverage and limitations. |
| `EXPERIMENT_AUDIT.md` | Current independent integrity and full-sweep readiness audit. |
| `review-stage/README.md` | Active result-directory index. |
| `src/bench/README.md` | Active benchmark runner index. |

## Active Results

| Directory | Contents |
|---|---|
| `review-stage/literature-grounding-gpt55/` | Stage 0 PubMed grounding and feasibility artifacts. |
| `review-stage/initial-claims-all-gpt55/` | Stage 1 initial questions and frozen contracts. |
| `review-stage/confirm-gates-all-gpt55/` | Stage 2 CONFIRM gate results. |
| `review-stage/claim-search-gpt55-sweep-smoke-v5/` | Latest Stage 3 readiness smoke test; not a canonical sweep. |

## Active Launchers

| Script | Purpose |
|---|---|
| `scripts/launch_literature_claim_grounding.sh` | Stage 0 literature-grounded source generation. |
| `scripts/launch_initial_claim_drafting.sh` | Stage 1 LLM question and `ClaimContract` drafting. |
| `scripts/launch_confirm_gate_evaluation.sh` | Stage 2 deterministic gate evaluation. |
| `scripts/launch_claim_search_sweep.sh` | Stage 3 same-data candidate-search matrix. |
| `scripts/launch_claim_search_safety.sh` | Stage 3 synthetic known-negative safety benchmark. |
| `scripts/launch_claim_search_fullscale.sh` | Canonical selected-arm excluded-evidence evaluation. |

The canonical Stage 3 input is:

```text
review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json
```

External preparation remains isolated under `scripts/data_processing/`, with
dataset policy in `configs/external_datasets.yml` and evidence roles in
`configs/evidence_partitions.yml`.
