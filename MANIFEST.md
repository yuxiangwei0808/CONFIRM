# Research Output Manifest

Current staged artifact set after cleanup on 2026-06-17.

| File | Stage | Description |
|---|---|---|
| `docs/STAGED_RESULTS_INDEX.md` | organization | Canonical index of current code, result artifacts, metrics, and rerun commands. |
| `review-stage/README.md` | organization | Review-stage directory guide and current artifact reading order. |
| `review-stage/REVIEW_STATE.json` | organization | Machine-readable status for the cleaned 24-claim pilot. |
| `CLAIMS_FROM_RESULTS.md` | review | Narrow claims supported by the cleaned label-aware results. |
| `EXPERIMENT_AUDIT.md` | review | Independent experiment integrity audit and post-audit fixes. |
| `review-stage/combined-label-aware-combat/combined_benchmark_results.json` | result | Primary cleaned combined ComBat result. |
| `review-stage/combined-label-aware-combat/combined_benchmark_audit.csv` | result | Per-claim audit table for the primary combined result. |
| `review-stage/combined-label-aware-none/combined_benchmark_results.json` | result | No-harmonization comparison for the cleaned combined result. |
| `review-stage/benchmark-ready-label-aware-combat/benchmark_ready_results.json` | result | fMRI descriptor component result with ComBat harmonization. |
| `review-stage/benchmark-ready-label-aware-none/benchmark_ready_results.json` | result | fMRI descriptor component result without harmonization. |
| `review-stage/multimodal-label-aware-combat/multimodal_benchmark_results.json` | result | NACC table-probe component result with ComBat harmonization. |
| `review-stage/multimodal-label-aware-none/multimodal_benchmark_results.json` | result | NACC table-probe component result without harmonization. |
| `review-stage/hardening-controls/hardening_controls_results.json` | result | Targeted FDR and power/MDE control checks. |
| `src/bench/labels.py` | code | Claim label taxonomy and label-aware scoring helpers. |
| `src/bench/run_benchmark_ready.py` | code | fMRI benchmark-ready runner. |
| `src/bench/run_multimodal_benchmark.py` | code | NACC table-probe runner. |
| `src/bench/combine_benchmark_results.py` | code | Combined fMRI plus NACC result summarizer. |
| `src/bench/run_hardening_controls.py` | code | FDR and power/MDE control checks. |
| `src/bench/README.md` | code | Benchmark runner code index. |
| `docs/literature_labels/fmri_claim_label_ledger.md` | data | First-pass fMRI claim label ledger. |
| `docs/data_manifests/adni_document_20260617_zip_inventory.md` | data | Inventory of useful files visible in the new ADNI document zip. |
| `docs/DATA_REQUIREMENTS_NEXT.md` | data | Local/remote/needed data requirements for benchmark expansion. |
| `docs/data_manifests/nacc_access_notes.md` | data | Access notes for NACC data needed later. |
| `docs/data_manifests/remote_recheck_20260617.md` | data | Read-only remote recheck and revised immediate scope. |

Removed stale artifacts:

- old archive outputs
- pre-label-aware hardened result folders
- obsolete timestamped review-state files under `review-stage/`
- obsolete auto-review render sidecar under `review-stage/`
- superseded hardening-control timestamp `hardening_controls_results_20260616_152518.json`
