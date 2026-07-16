# Benchmark Code Index

The active CONFIRM workflow is implemented by these runners:

| File | Purpose |
|---|---|
| `run_literature_claim_grounding.py` | Stage 0 PubMed grounding and executable seed creation. |
| `run_initial_claim_drafting.py` | Stage 1 LLM question generation and frozen contract drafting. |
| `run_drafted_contract_gates.py` | Stage 2 deterministic CONFIRM gate evaluation. |
| `run_iterative_claim_search_replay.py` | Stage 3 iterative LLM candidate search. |
| `run_claim_search_case_studies.py` | Canonical claim-search trace tables. |
| `run_negatives_expansion.py` | Synthetic known-negative safety source and gate evaluation. |
| `injected_nulls.py` | Synthetic/adversarial null construction. |
| `progress.py` | Shared progress reporting. |

`labels.py` and `metrics.py` support the synthetic safety benchmark and legacy
paper tables. External evidence preparation is kept separate under `nbs_data/`;
the active scientific pipeline does not run external datasets as standalone
Stage 2 benchmarks.
