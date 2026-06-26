# Implementation Notes

Generated: 2026-06-17

This file reflects the current staged workflow. Older pilot commands have been superseded by the label-aware benchmark runners.

## Current Entry Points

```bash
PYTHONPATH=src ./.venv/bin/python -m bench.run_benchmark_ready --harmonize combat --out-dir review-stage/benchmark-ready-label-aware-combat
PYTHONPATH=src ./.venv/bin/python -m bench.run_multimodal_benchmark --harmonize combat --out-dir review-stage/multimodal-label-aware-combat
PYTHONPATH=src ./.venv/bin/python -m bench.combine_benchmark_results --input review-stage/benchmark-ready-label-aware-combat/benchmark_ready_results.json --input review-stage/multimodal-label-aware-combat/multimodal_benchmark_results.json --out-dir review-stage/combined-label-aware-combat
PYTHONPATH=src ./.venv/bin/python -m pytest -q
```

## Current Artifact Index

- `docs/STAGED_RESULTS_INDEX.md`
- `review-stage/README.md`
- `review-stage/REVIEW_STATE.json`
- `CLAIMS_FROM_RESULTS.md`

## Scope

The staged workflow is CPU-only and uses precomputed neuroimaging-derived tables. Raw image preprocessing and raw-image LLM interpretation are intentionally deferred.
