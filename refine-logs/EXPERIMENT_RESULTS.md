# Current Experiment Results

Generated: 2026-06-17

The current canonical results are staged in `review-stage/combined-label-aware-*` and indexed in `docs/STAGED_RESULTS_INDEX.md`.

## Primary Result

- Primary run: `review-stage/combined-label-aware-combat/combined_benchmark_results.json`
- Audit table: `review-stage/combined-label-aware-combat/combined_benchmark_audit.csv`
- No-harmonization comparison: `review-stage/combined-label-aware-none/combined_benchmark_results.json`

## Scope

- 24 total claims
- 20 fMRI descriptor claims
- 4 NACC table-probe claims

## Full-Gate Metrics

- TPR: `5/5 = 1.000`
- FCR on known/synthetic null and stress claims: `0/7 = 0.000`
- Small-positive recovery: `2/6 = 0.333`
- Candidate confirmation: `0/6 = 0.000`

Interpretation boundary: this is a pilot-scale benchmark of the gate behavior, not a broad claim that the system validates neuroimaging biomarkers across disease domains.
