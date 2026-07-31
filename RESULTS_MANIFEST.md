# CONFIRM Results Manifest

Updated: 2026-07-23

## Reusable Outputs

| Stage | Directory | Primary artifact |
|---|---|---|
| Stage 0 literature grounding | `review-stage/literature-grounding-gpt55/` | `literature_grounding_summary.json` |
| Stage 1 initial claim drafting | `review-stage/initial-claims-all-gpt55/` | `drafted_contracts.jsonl` |
| Stage 2 CONFIRM gates | `review-stage/confirm-gates-all-gpt55/` | `combined_benchmark_results.json` |
| Stage 2B initial-claim evidence audit | `review-stage/initial-claims-gpt55-retrospective-evidence-v1/` | `summary.json` |
| Stage 3 scientific sweep | `review-stage/claim-search-gpt55-sweep-v7/` | `matrix_summary.json` |
| Stage 3 matched control | `review-stage/claim-search-gpt55-control-r3-c5-v7/` | `control_summary.json` |
| Stage 3 known-negative safety | `review-stage/claim-search-safety-gpt55-r10-c10-v7/` | `replay/iterative_candidate_replay.json` |
| Stage 4 follow-up evidence audit | `review-stage/claim-search-gpt55-retrospective-evidence-v3/` | `summary.json` |
| Paper analysis | `review-stage/claim-search-gpt55-paper-analysis-v1/` | `analysis_manifest.json` |
| NeuroClaimBench v2.1 | `review-stage/neuroclaimbench-v2.1/` | `results/benchmark_summary.json` |
| Publishable benchmark release | `benchmark/neuroclaimbench-v2.1/` | `SHA256SUMS` |
| Paper claim audit | `review-stage/paper-claim-audit-v21/` | `PAPER_CLAIM_AUDIT.md` |

## Archived, not active evidence

| Directory | Replacement |
|---|---|
| `review-stage/_archive_20260730_inactive_probes/multillm-probe-v2/` | None; restore only for a renewed model-comparison study. |
| `review-stage/_archive_20260730_superseded_runs/claim-search-gpt55-control-r3-c5-v6-fixed/` | `claim-search-gpt55-control-r3-c5-v7/` |
| `review-stage/_archive_20260730_superseded_runs/simplification-audit-20260723/` | v2.1 compact release and audit |

These directories are immutable inputs or completed descriptive analyses for
the next feedback-loop run. Their primary hashes are recorded in
`RESULTS_SHA256SUMS`.

## NeuroClaimBench v2.1

The outcome-blind v2.1 source inventory contains 544 canonical items. Forty
ambiguous question-contract mappings and 15 non-executable items remain in the
crosswalk, leaving 489 executable claims and 268 score-eligible references.
All 489 tasks completed without an execution error.

The primary scientific literature result is 21/51 confirmable-claim recall and
2/19 unsafe confirmations. External literature is reported separately:
0/7 NACC and 0/11 ds000030 confirmable claims are recovered, while 0/8
ds000030 literature abstention references are confirmed. Constructed controls
also remain separate: 0/14 NACC, 0/8 ds000030, and 0/150 synthetic controls are
confirmed. The other 221 executable claims are unresolved and do not enter
accuracy denominators.

The canonical metadata package is `data/neuroclaimbench/v2.1/`; the lean
publishable release is `benchmark/neuroclaimbench-v2.1/`. Both release and
external-audit checksum manifests verify. NeuroClaimBench v2.1 is a
retrospective benchmark revision, not prospective external validation.

## Stage 0-2 Counts

| Item | Count |
|---|---:|
| PubMed records | 99 |
| extracted literature seeds | 274 |
| executable literature-grounded questions | 41 |
| LLM-proposed questions | 250 |
| total Stage 1 questions | 291 |
| drafted and Stage 2-evaluated contracts | 289 |
| Stage 1 non-draftable questions | 2 |
| Stage 2 execution errors | 0 |

Stage 2 labels:

| Label | Count |
|---|---:|
| confirmed | 74 |
| fragile | 169 |
| non_replicated | 38 |
| under_powered | 8 |

The 215 non-confirmed contracts are the fixed Stage 3 parent set.

## Retrospective Evidence

The Stage 2B audit made no LLM calls. It evaluated 209 holdout-compatible
initial claims and 35 externally compatible initial claims. At the claim level,
21 passed internal holdout evaluation and 24 passed NACC external evaluation.
All such evidence is marked `previously_queried` and
`final_confirmation_eligible=false`; these are retrospective concordance
results, not fresh confirmations.

## Feedback Search

All 12 v7 scientific arms completed 215 parents with zero source-execution
errors and zero excluded-evidence queries during search. Across arms, 63,598
proposals yielded 57,793 source evaluations, 1,492 provisional passes, and
1,333 final multiplicity-adjusted passes. The predeclared `R=3, C=5` reference
arm has:

| Generated | Retained | Source-evaluated | Provisional | Final support | Supported parents |
|---|---:|---:|---:|---:|---:|
| 3,187 | 3,059 | 2,971 | 86 | 70 | 24 |

The v7 synthetic stress run evaluates 10,716 candidates from 150 known-negative
parents and supports none. The frozen evidence audit executes all 1,275 planned
tasks without error. At `R=3, C=5`, 10/60 holdout-evaluable candidates survive
across six parents and 14/25 NACC-evaluable candidates survive across four
AD/aging parents. All excluded evidence is retrospective and previously
queried.

## Matched Feedback Control

The matched v7 generic-retry control completed all 215 parents with no source
execution errors and no excluded-evidence queries. At `R=3, C=5`, structured
diagnosis used 653 LLM calls and produced 70 final source-supported candidates
across 24 parents; generic retry's completed trace used 693 calls and produced
49 candidates across 20 parents. Generic retry additionally records 72
superseded transport-failed attempts. Paired parent cells are 17 both, 7
structured-only, 3 generic-only, and 188 neither. This is one descriptive
GPT-5.5 realization, not a causal feedback-effect estimate.

The 200-item blinded parent-relative novelty packet and 600-row template for
three reviewers are complete under the paper-analysis directory. Human ratings
remain pending; no expert-novelty or usefulness result is reported yet.
