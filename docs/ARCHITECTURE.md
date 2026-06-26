# Current Architecture

Generated: 2026-06-17

CONFIRM is staged as a CPU-only, tabular neuroimaging-derivative framework. It does not currently process raw images. The current pipeline consumes prepared public/cohort derivative tables, executes claim contracts with deterministic statistical gates, and reports whether claims may be labeled `confirmed`, `non_replicated`, `under_powered`, or `fragile`.

## Core Layers

| Layer | Files | Role |
|---|---|---|
| Claim engine | `src/confirm/contract.py`, `src/confirm/agent.py`, `src/confirm/verdict.py` | Validate claim contracts, orchestrate execution, and assign gated verdicts. |
| Statistics | `src/confirm/analysis.py`, `src/confirm/power.py`, `src/confirm/multiverse.py`, `src/confirm/replication.py`, `src/confirm/brainwide.py` | Run primary models, power checks, multiverse checks, scalar replication, and brain-wide replication utilities. |
| Harmonization | `src/confirm/harmonize.py` | Optional ComBat harmonization for site/scanner effects. |
| Provenance | `src/confirm/provenance.py`, `src/confirm/results.py` | Serialize numeric outputs and receipts. |
| Benchmark runners | `src/bench/run_benchmark_ready.py`, `src/bench/run_multimodal_benchmark.py`, `src/bench/combine_benchmark_results.py` | Run the current fMRI descriptor and NACC table-probe benchmark. |
| Label-aware scoring | `src/bench/labels.py` | Separate known positives, known/synthetic nulls, stress tests, fragile candidates, and small-positive expected claims. |

## Current Data Shape

The current staged experiments use:

- `data/prepared_data/benchmark_ready/cohorts/*.parquet` for fMRI descriptor cohorts.
- `data/prepared_data/benchmark_ready/misc_tables/*.parquet` for copied disease/multimodal tables.
- `data/prepared_data/benchmark_ready/claim_inventory_ready.csv` for fMRI claim inventory.
- `docs/literature_labels/fmri_claim_label_ledger.md` for human-readable claim labels.

Rows are subject/session-level table records. Imaging variables are derived measures such as `fc_*`, regional descriptors, sMRI-derived measures, PET/CSF/table measures when available, and demographic/clinical covariates.

## Current Experiment

The headline staged result is:

- primary: `review-stage/combined-label-aware-combat/combined_benchmark_results.json`
- audit table: `review-stage/combined-label-aware-combat/combined_benchmark_audit.csv`
- no-harmonization comparison: `review-stage/combined-label-aware-none/combined_benchmark_results.json`

It combines 20 fMRI descriptor claims with 4 NACC table-probe claims. At the full gate, the current pilot has TPR `5/5`, FCR `0/7`, small-positive recovery `2/6`, and candidate confirmation `0/6`.

## Legacy Boundary

Some earlier exploratory pilot code remains in the repository for reference and backward compatibility, but it is not part of the staged artifact set. The current entry points are the `src/bench/*label-aware*` runners listed above.

## Verification

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q
```

The cleanup-stage verification passed with 13 tests.
