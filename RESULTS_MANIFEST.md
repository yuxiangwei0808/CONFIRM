# CONFIRM Results Manifest

Updated: 2026-07-16

## Active Outputs

| Stage | Directory | Primary artifact |
|---|---|---|
| Stage 0 literature grounding | `review-stage/literature-grounding-gpt55/` | `literature_grounding_summary.json` |
| Stage 1 initial claim drafting | `review-stage/initial-claims-all-gpt55/` | `drafted_contracts.jsonl` |
| Stage 2 CONFIRM gates | `review-stage/confirm-gates-all-gpt55/` | `combined_benchmark_results.json` |
| Stage 3 readiness smoke | `review-stage/claim-search-gpt55-sweep-smoke-v5/` | `matrix_summary.json` |

## Stage 0-2 Counts

| Item | Count |
|---|---:|
| executable literature-grounded questions | 41 |
| LLM-proposed questions | 250 |
| total Stage 1 questions | 291 |
| drafted and Stage 2-evaluated contracts | 289 |
| Stage 2 execution errors | 0 |

Stage 2 labels:

| Label | Count |
|---|---:|
| confirmed | 74 |
| fragile | 169 |
| non_replicated | 38 |
| under_powered | 8 |

## Stage 3 Smoke Status

The GPT-5.5 `R=1, C=2` smoke searched all 194 evidence-eligible failed Stage 2
claims. It generated 382 candidates, retained 356 unique candidates, evaluated
351 valid connected candidates, produced one same-data
`exploratory_confirmed` result, and made zero excluded-evidence queries. It had
zero execution errors and zero non-identifiable analyses. This validates the
launcher and accounting only; it is not the full sweep or canonical result.

## Launch Commands

```bash
scripts/launch_literature_claim_grounding.sh
MAX_WORKERS=16 PARALLEL_BACKEND=thread scripts/launch_initial_claim_drafting.sh
MAX_WORKERS=16 PARALLEL_BACKEND=process scripts/launch_confirm_gate_evaluation.sh
```

Next full Stage 3 sweep:

```bash
OUT=review-stage/claim-search-gpt55-sweep-v5 \
MODEL=openai:gpt-5.5 \
ROUNDS="1 3 5 10" CANDIDATES="2 5 10" \
BUILD_EVIDENCE_PARTITIONS=off MAX_WORKERS=8 \
scripts/launch_claim_search_sweep.sh
```

Synthetic safety and canonical excluded-evidence evaluation remain separate
runs. Same-data support is never counted as holdout or external confirmation.
