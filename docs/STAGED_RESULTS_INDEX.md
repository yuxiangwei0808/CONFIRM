# Staged Results Index

> **SUPERSEDED (2026-06-20).** Describes an earlier 24-claim pilot. Authoritative result map: [../RESULTS_MANIFEST_20260620.md](../RESULTS_MANIFEST_20260620.md).

Generated: 2026-06-17

This workspace is not currently a Git repository, so "staged" here means organized, indexed, and ready to hand off or rerun. The current canonical experiment state is the cleaned label-aware fMRI descriptor benchmark plus NACC table probes under `review-stage/combined-label-aware-*`.

## Current Status

- Current benchmark scope: 24 claims, combining 20 fMRI benchmark-ready descriptor claims with 4 NACC table probes.
- Best current run: `review-stage/combined-label-aware-combat/combined_benchmark_results.json`.
- Main pilot signal: loose execution-only scoring confirms several null/stress claims, while the full gate reduces labeled false confirmations to zero.
- Current review boundary: ready to scale experiments, not ready for broad biomarker claims.
- Verification status: `pytest -q` passed with 13 tests after cleanup.

## Canonical Result Artifacts

| Artifact | Use |
|---|---|
| `review-stage/combined-label-aware-combat/combined_benchmark_results.json` | Primary cleaned combined ComBat result. |
| `review-stage/combined-label-aware-combat/combined_benchmark_audit.csv` | Per-claim audit table for the primary combined run. |
| `review-stage/combined-label-aware-none/combined_benchmark_results.json` | No-harmonization comparison after cleanup. |
| `review-stage/combined-label-aware-none/combined_benchmark_audit.csv` | Per-claim audit table for the no-harmonization combined run. |
| `review-stage/benchmark-ready-label-aware-combat/benchmark_ready_results.json` | fMRI-only label-aware ComBat component result. |
| `review-stage/benchmark-ready-label-aware-combat/benchmark_ready_audit.csv` | Per-claim audit table for the primary fMRI component. |
| `review-stage/benchmark-ready-label-aware-none/benchmark_ready_results.json` | fMRI-only no-harmonization comparison. |
| `review-stage/multimodal-label-aware-combat/multimodal_benchmark_results.json` | NACC table-probe ComBat component result. |
| `review-stage/multimodal-label-aware-combat/multimodal_benchmark_audit.csv` | NACC table-probe audit table. |
| `review-stage/hardening-controls/hardening_controls_results.json` | Targeted FDR and power/MDE control checks. |
| `review-stage/REVIEW_STATE.json` | Current machine-readable review state. |
| `CLAIMS_FROM_RESULTS.md` | Narrow claims supported by the current experiments. |
| `EXPERIMENT_AUDIT.md` | Integrity audit and post-audit fixes. |

Removed stale outputs: pre-label-aware hardened result folders, pre-hardening archive outputs, obsolete timestamped review-state files, and the obsolete auto-review sidecar.

## Current Primary Metrics

Primary result: `review-stage/combined-label-aware-combat/combined_benchmark_results.json`.

| Gate | TPR | FCR on null/stress claims | Small-positive recovery | Candidate confirmation |
|---|---:|---:|---:|---:|
| exec only | 5/5 = 1.000 | 4/7 = 0.571 | 4/6 = 0.667 | 3/6 = 0.500 |
| + confound | 5/5 = 1.000 | 2/7 = 0.286 | 4/6 = 0.667 | 3/6 = 0.500 |
| + power | 5/5 = 1.000 | 2/7 = 0.286 | 4/6 = 0.667 | 3/6 = 0.500 |
| + multiverse | 5/5 = 1.000 | 2/7 = 0.286 | 3/6 = 0.500 | 3/6 = 0.500 |
| + replication | 5/5 = 1.000 | 0/7 = 0.000 | 2/6 = 0.333 | 0/6 = 0.000 |

Interpretation boundary: these are pilot-scale results. The important result is the gate-ladder behavior, not a broad claim about neuroimaging biomarkers yet.

## Code Entry Points

| File | Role |
|---|---|
| `src/bench/labels.py` | Claim label taxonomy and label-aware scoring helpers. |
| `src/bench/run_benchmark_ready.py` | fMRI benchmark-ready runner. |
| `src/bench/run_multimodal_benchmark.py` | NACC table-probe runner. |
| `src/bench/combine_benchmark_results.py` | Combines fMRI and multimodal outputs into one label-aware report. |
| `src/bench/run_hardening_controls.py` | FDR and power/MDE control checks. |
| `src/confirm/agent.py` | Confirmation agent orchestration. |
| `src/confirm/contract.py` | Claim-contract data structures. |
| `src/confirm/analysis.py` | Statistical execution helpers used by claims. |
| `src/confirm/verdict.py` | Gate verdict logic. |

## Data Inputs Currently Used

| Path | Content |
|---|---|
| `data/prepared_data/benchmark_ready/claim_inventory_ready.csv` | Current fMRI claim inventory. |
| `data/prepared_data/benchmark_ready/*.parquet` | Prepared public-cohort fMRI/cognition/metadata tables. |
| `data/prepared_data/benchmark_ready/misc_tables/*.parquet` | Copied metadata/derivative tables for NACC, AIBL, MIRIAD, COBRE, FBIRN, and SZ_JH. |
| `docs/literature_labels/fmri_claim_label_ledger.md` | First-pass fMRI claim label ledger. |
| `docs/data_manifests/remote_peek_20260616.md` | Remote-cluster inventory report from the prior peek. |

## Re-run Commands

```bash
PYTHONPATH=src ./.venv/bin/python -m bench.run_benchmark_ready \
  --harmonize combat \
  --out-dir review-stage/benchmark-ready-label-aware-combat

PYTHONPATH=src ./.venv/bin/python -m bench.run_benchmark_ready \
  --harmonize none \
  --out-dir review-stage/benchmark-ready-label-aware-none

PYTHONPATH=src ./.venv/bin/python -m bench.run_multimodal_benchmark \
  --harmonize combat \
  --out-dir review-stage/multimodal-label-aware-combat

PYTHONPATH=src ./.venv/bin/python -m bench.run_multimodal_benchmark \
  --harmonize none \
  --out-dir review-stage/multimodal-label-aware-none

PYTHONPATH=src ./.venv/bin/python -m bench.combine_benchmark_results \
  --input review-stage/benchmark-ready-label-aware-combat/benchmark_ready_results.json \
  --input review-stage/multimodal-label-aware-combat/multimodal_benchmark_results.json \
  --out-dir review-stage/combined-label-aware-combat

PYTHONPATH=src ./.venv/bin/python -m bench.combine_benchmark_results \
  --input review-stage/benchmark-ready-label-aware-none/benchmark_ready_results.json \
  --input review-stage/multimodal-label-aware-none/multimodal_benchmark_results.json \
  --out-dir review-stage/combined-label-aware-none

PYTHONPATH=src ./.venv/bin/python -m pytest -q
```

## Immediate Next Implementation Targets

1. Extract and adapt the useful small CSVs from `data/raw/ADNI_document-20260617T002101Z-3-001.zip`.
2. Add a stronger ADNI disease/cognition/PET table adapter.
3. Add a formal machine-readable label table generated from `docs/literature_labels/fmri_claim_label_ledger.md`.
4. Expand NACC only if a clinical diagnosis/demographic table is added locally.
