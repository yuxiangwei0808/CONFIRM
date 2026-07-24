# Analysis Utilities

This directory contains result-source builders and deterministic summarizers.
It does not contain experiment launchers or LLM clients.

| File | Purpose |
|---|---|
| `build_claim_search_source_from_results.py` | Convert Stage 2 failures into the Stage 3 source payload. |
| `summarize_claim_search_control.py` | Validate and summarize the matched feedback control. |
| `summarize_claim_search_matrix.py` | Validate and summarize a completed search matrix. |
| `analyze_claim_search_sweep.py` | Generate sweep budget, funnel, strata, transform, multiplicity, stability, and safety analyses. |
| `analyze_claim_search_evidence.py` | Generate excluded-evidence survival, matched parent/candidate, and deterministic case-study analyses. |
| `candidate_novelty_review.py` | Compute deterministic parent-relative novelty and run the 50-pair blinded structured-versus-generic forced-choice review. |
| `claim_search_analysis_common.py` | Shared bounded readers, hashing, intervals, and atomic output helpers for the analysis scripts. |
| `analyze_neuroclaimbench_v21.py` | Audit task reconciliation, alignment, reference expansion, PubMed retrieval, and unresolved verdicts. |
| `analyze_neuroclaimbench_gate_attribution.py` | Attribute frozen benchmark abstentions to individual gates and compute leave-one-gate-out sensitivity. |
| `analyze_neuroclaimbench_v21_feedback_crosswalk.py` | Join frozen feedback parents to v2.1 references by exact pre-v2 contract identity. |
| `plot_coverage_fcr.py` | Generate coverage/FCR paper tables and figures from frozen result JSON. |
| `freeze_scientific_runs.py` | Register hashes for the frozen v2.1 benchmark, v7 sweep, source snapshot, and paper tables. |
| `normalize_claim_search_artifacts.py` | Convert nested v7 parent checkpoints into compact, reconciled JSONL tables without rerunning search. |
| `compact_neuroclaimbench.py` | Build release schema 2 directly from the frozen package, reference decisions, and local task outcomes. |
| `audit_result_simplification.py` | Verify normalized sweep reconciliation, archives, compact release counts, checksums, and storage reduction. |

The claim-search paper analyses write only to
`review-stage/claim-search-gpt55-paper-analysis-v1/`. They never modify sweep,
checkpoint, evidence, or paper artifacts.
