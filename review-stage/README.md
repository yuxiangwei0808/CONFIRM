# Review Stage Index

Updated: 2026-07-23

| Directory | Role |
|---|---|
| `literature-grounding-gpt55/` | Reusable Stage 0 PubMed grounding and local feasibility audit. |
| `initial-claims-all-gpt55/` | Reusable Stage 1 prompts, responses, frozen contracts, and validation. |
| `confirm-gates-all-gpt55/` | Reusable Stage 2 deterministic CONFIRM gate outputs. |
| `initial-claims-gpt55-retrospective-evidence-v1/` | Completed Stage 2B retrospective holdout/external audit. |
| `claim-search-gpt55-sweep-v7/normalized/` | Active compact, reconciled representation of the completed 12-arm Stage 3 sweep. |
| `claim-search-safety-gpt55-r1-c2-v7/` | Completed bounded known-negative smoke run. |
| `claim-search-safety-gpt55-r10-c10-v7/` | Completed maximum-budget known-negative stress run. |
| `claim-search-gpt55-control-r3-c5-v7/` | Completed matched structured-diagnosis versus generic-retry control. |
| `claim-search-gpt55-retrospective-evidence-v3/` | Completed frozen retrospective holdout/external audit of final internally supported candidates. |
| `claim-search-gpt55-paper-analysis-v1/` | Derived tables, figures, case studies, audit, and a completed blinded-review packet; human ratings remain pending. |
| `neuroclaimbench-v2.1/alignment/` | Frozen outcome-blind question-contract alignment audit. |
| `neuroclaimbench-v2.1/adjudication/` | Completed cache-backed adjudication for invalidated literature identities. |
| `neuroclaimbench-v2.1/reference/` | Frozen reference basis and strength profiles. |
| `neuroclaimbench-v2.1/results/` | Complete v2.1 task outcomes and clustered sensitivity summaries. |
| `neuroclaimbench-v2.1/analysis/` | Final benchmark tables and audits. |
| `neuroclaimbench-v2.1/feedback-crosswalk/` | Exact pre-repair feedback-parent crosswalk. |
| `neuroclaimbench-v2.1/compact/` | Compact case/reference/task/outcome representation used by active analyses. |
| `neuroclaimbench-v2.1/external-archive/` | Checksummed detailed v2.1 audit payload; not the default public model. |
| `_archive_20260719_pre_v7/` | Ignored, recoverable superseded/debug outputs; never active evidence. |
| `_archive_20260723_claim_search_v7_original/` | Checksummed compressed original v7 arm artifacts. |
| `_archive_20260723_neuroclaimbench_release_v1/` | Superseded release-schema-1 package and audit payload. |

Stage 3, Stage 4, and paper-analysis outputs use the v7/v3/v1 paths documented
in `../IMPLEMENTATION_NOTES.md`. Generated result directories are ignored by git; use
`../RESULTS_SHA256SUMS` to verify reusable inputs.
