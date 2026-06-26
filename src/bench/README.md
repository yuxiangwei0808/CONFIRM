# Benchmark Code Index

This folder contains the benchmark runners and result combiners used by the current pilot.

| File | Purpose |
|---|---|
| `labels.py` | Claim label taxonomy and label-aware scoring. |
| `run_benchmark_ready.py` | Runs the fMRI benchmark-ready claims over prepared cohort tables. |
| `run_multimodal_benchmark.py` | Runs table-derived NACC probe claims. |
| `combine_benchmark_results.py` | Combines fMRI and multimodal result JSON files into one label-aware summary. |
| `run_hardening_controls.py` | Runs targeted controls for FDR and power/MDE behavior. |
| `claim_library.py` | Shared claim defaults and effect-size references. |
| `injected_nulls.py` | Synthetic/adversarial null construction helpers. |

The current headline experiment is not raw-image processing. It consumes public/cohort derivative tables and asks whether the agentic confirmation gates prevent invalid neuroimaging-related claims from being labeled confirmed.
