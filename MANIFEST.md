# Research Output Manifest

Updated: 2026-07-10

## Active Documents

| File | Purpose |
|---|---|
| `IMPLEMENTATION_NOTES.md` | Detailed description of the current Stage 0/1/2 pipeline and feedback-loop hook. |
| `RESULTS_MANIFEST.md` | Current result layout, counts, and launch commands. |
| `RESEARCH_BRIEF.md` | Short project framing. |
| `EXTERNAL_BENCHMARK_RESULTS.md` | Auxiliary external-evaluation note. |
| `review-stage/README.md` | Active result-directory index. |
| `src/bench/README.md` | Benchmark runner/code index. |

## Active Results

| Directory | Contents |
|---|---|
| `review-stage/literature-grounding-gpt55/` | Stage 0 PubMed/literature grounding artifacts. |
| `review-stage/initial-claims-all-gpt55/` | Stage 1 full initial questions and drafted contracts. |
| `review-stage/confirm-gates-all-gpt55/` | Stage 2 CONFIRM gate results for drafted contracts. |
| `review-stage/_archive_20260702_pipeline_cleanup/` | Superseded literature-only and pre-cleanup feedback-loop outputs. |

## Main CONFIRM Launchers

| Script | Purpose |
|---|---|
| `scripts/launch_literature_claim_grounding.sh` | Stage 0 literature-grounded claim source generation. |
| `scripts/launch_initial_claim_drafting.sh` | Stage 1 LLM question generation and `ClaimContract` drafting. |
| `scripts/launch_confirm_gate_evaluation.sh` | Stage 2 CONFIRM gate evaluation. |
| `scripts/launch_claim_search_fullscale.sh` | One feedback-loop claim-search run from Stage 2 outputs. |
| `scripts/launch_claim_search_sweep.sh` | Feedback-loop claim-search parameter sweep. |
| `scripts/launch_curated_gate_benchmark.sh` | Legacy curated/auxiliary gate benchmark launcher. |

## External Data-Processing Launchers

These are isolated from the main workflow under `scripts/data_processing/`:

| Script | Purpose |
|---|---|
| `scripts/data_processing/audit_external_datasets_arcdev.sh` | Remote external registry, metadata, and baseline-scan audit. |
| `scripts/data_processing/launch_external_fmri_arcdev.sh` | Remote external fMRI descriptor preparation. |
| `scripts/data_processing/launch_external_freesurfer_arcdev.sh` | Receipt-backed FastSurfer/FreeSurfer preparation. |
| `scripts/data_processing/sync_external_evidence_arcdev.sh` | Versioned candidate sync and explicit promotion. |

External dataset policy and routing are defined in:

```text
configs/external_datasets.yml
configs/evidence_partitions.yml
```

The canonical feedback-loop input is the full Stage 2 artifact:

```text
review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json
```
