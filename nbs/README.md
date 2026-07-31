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
| `analyze_confirm_evidence_tiers.py` | Derive discovery, replicated, and strict confirmation reporting decisions from frozen gate vectors. |
| `analyze_neuroclaimbench_v21_feedback_crosswalk.py` | Join frozen feedback parents to v2.1 references by exact pre-v2 contract identity. |
| `plot_coverage_fcr.py` | Generate coverage/FCR paper tables and figures from frozen result JSON. |
| `freeze_scientific_runs.py` | Register hashes for the frozen v2.1 benchmark, v7 sweep, source snapshot, and paper tables. |
| `normalize_claim_search_artifacts.py` | Convert nested v7 parent checkpoints into compact, reconciled JSONL tables without rerunning search. |
| `compact_neuroclaimbench.py` | Build release schema 2 directly from the frozen package, reference decisions, and local task outcomes. |
| `audit_result_simplification.py` | Verify normalized sweep reconciliation, archives, compact release counts, checksums, and storage reduction. |
| `analyze_claim_evaluation_baselines.py` | Freeze label-blind direct-LLM and discovery-plus-replication significance decisions, then compare them with CONFIRM. |
| `analyze_feedback_method_baselines.py` | Compare failure-specific diagnosis, failure-blind retry, and Self-Refine at the matched R3/C5 scientific budget, with separately labeled safety runs. |
| `freeze_feedback_baseline_manifest.py` | Freeze source, prompt, schema, and artifact identities for the feedback-method evidence audit. |
| `plot_main_confirm_results.py` | Generate the paper's baseline and evidence-tier figure, feedback table, and supplementary comparison tables from frozen CSV summaries. |
| `analyze_reference_bar_sensitivity.py` | Derive the active reference-bar sensitivity summary from frozen benchmark decisions. |
| `plot_reference_bar_frontier.py` | Render the active reference-bar sensitivity figure from its summary. |

The claim-search paper analyses write only to
`review-stage/claim-search-gpt55-paper-analysis-v1/`. They never modify sweep,
checkpoint, evidence, or paper artifacts.

Inactive multi-LLM and power-reference probes are retained under
`_archive_20260730_inactive_probes/`; they are not current paper inputs.
