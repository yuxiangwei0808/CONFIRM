# Benchmark Code Index

This package contains the active CONFIRM benchmark runners and helper modules.

| File | Purpose |
|---|---|
| `run_literature_claim_grounding.py` | Stage 0: query/use PubMed records, extract literature-grounded seeds, and write executable claim questions. |
| `run_initial_claim_drafting.py` | Stage 1: load fixed claim questions or ask an LLM to propose questions, then draft frozen `ClaimContract`s. |
| `run_drafted_contract_gates.py` | Stage 2: evaluate drafted contracts with unchanged CONFIRM gates. |
| `run_iterative_claim_search_replay.py` | Feedback-loop replay over failed or mismatched gate results. |
| `run_claim_search_case_studies.py` | Case-study table generation for claim-search artifacts. |
| `run_expanded_benchmark.py` | Legacy curated deterministic gate benchmark used only through `scripts/launch_curated_gate_benchmark.sh`. |
| `combine_benchmark_results.py` | Combines curated benchmark result JSON files into one label-aware summary. |
| `run_negatives_expansion.py` | Auxiliary synthetic/null stress evaluation. |
| `run_nacc_external.py`, `run_external_generic.py` | Auxiliary external-evaluation runners. |
| `labels.py`, `metrics.py`, `claim_library.py` | Shared scoring labels, metric summaries, and claim defaults. |
| `injected_nulls.py` | Synthetic/adversarial null construction helpers. |

The active main pipeline is: literature grounding -> initial claim drafting ->
CONFIRM gate evaluation -> optional feedback-loop claim search.

External candidate preparation lives under `nbs_data/`; external datasets do
not enter Stage 0, Stage 1, or Stage 2. The feedback-loop evaluator selects
only predeclared, contract-compatible primary evidence sets from
`configs/evidence_partitions.yml`.
