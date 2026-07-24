# Benchmark Code Index

The active CONFIRM workflow is implemented by these runners:

| File | Purpose |
|---|---|
| `run_literature_claim_grounding.py` | Stage 0 PubMed grounding and executable seed creation. |
| `run_initial_claim_drafting.py` | Stage 1 LLM question generation and frozen contract drafting. |
| `run_drafted_contract_gates.py` | Stage 2 deterministic CONFIRM gate evaluation. |
| `run_initial_claim_evidence.py` | Stage 2B frozen retrospective evidence audit for initial claims. |
| `run_iterative_claim_search_replay.py` | Stage 3 iterative LLM candidate search. |
| `run_frozen_claim_evidence.py` | Stage 4 outcome-blind freeze, preflight, evaluation, and summary. |
| `run_known_negative_safety.py` | Synthetic known-negative safety source and gate evaluation. |
| `run_neuroclaimbench_build.py` | Build the frozen source snapshot used by the v2.1 alignment pass. |
| `run_neuroclaimbench_pubmed_cache.py` | Plan, fetch, freeze, and audit the local PubMed snapshot. |
| `run_neuroclaimbench_adjudication.py` | Run cache-backed multi-model literature adjudication. |
| `run_neuroclaimbench_reference_expansion.py` | Derive strict, provisional, and evidence-gap triage references without new LLM calls. |
| `run_neuroclaimbench_alignment.py` | Run the outcome-blind deterministic and Gemini question-contract alignment audit. |
| `run_neuroclaimbench_v21_build.py` | Apply frozen repairs to the source snapshot and build the content-hashed v2.1 package. |
| `run_neuroclaimbench_finalize.py` | Rerun frozen tasks and produce tiered and semantic-cluster uncertainty summaries. |
| `run_neuroclaimbench_v21_release.py` | Build the lean checksummed release and abstract-free external archive. |
| `benchmark.py` | Compact public case/reference/task/outcome schema. |
| `neuroclaimbench_v21_compat.py` | Frozen v2.1 construction schemas and policies used only for exact historical reproduction. |
| `io.py` | Shared atomic JSONL I/O. |
| `pubmed.py` | Shared PubMed retrieval boundary. |
| `injected_nulls.py` | Synthetic/adversarial null construction. |
| `progress.py` | Shared progress reporting. |

`labels.py` and `metrics.py` support the synthetic safety benchmark and legacy
paper tables. External evidence preparation is kept separate under `nbs_data/`;
the active scientific pipeline does not run external datasets as standalone
Stage 2 benchmarks. Result builders, summarizers, and plotting utilities live
under `nbs/`.

Public runtime code uses `confirm.execution.evaluate_contract`; runners do not
import private helpers from other runners. The Stage 0 feasibility rules,
Stage 1 contract compiler, v7 candidate wire schema, and v2.1 Gemini
eligibility behavior are frozen method implementations rather than general
reusable APIs.
