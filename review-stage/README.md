# Review Stage Index

Use this directory as the staged output area for current experiment results, audit tables, and review state.

## Current Canonical Outputs

- `combined-label-aware-combat/`: primary cleaned result.
- `combined-label-aware-none/`: no-harmonization comparison for the cleaned result.
- `benchmark-ready-label-aware-combat/`: fMRI descriptor component result.
- `benchmark-ready-label-aware-none/`: fMRI descriptor no-harmonization comparison.
- `multimodal-label-aware-combat/`: NACC table-probe component result.
- `multimodal-label-aware-none/`: NACC table-probe no-harmonization comparison.
- `hardening-controls/`: targeted FDR and power/MDE control checks.
- `REVIEW_STATE.json`: current machine-readable review status.

## Reading Order

1. Start with `../docs/STAGED_RESULTS_INDEX.md`.
2. Inspect `combined-label-aware-combat/combined_benchmark_audit.csv` for claim-level details.
3. Use `combined-label-aware-none/` and component result directories for ablations.
4. Read `../CLAIMS_FROM_RESULTS.md` for the narrow claims supported by the current pilot.
