# CONFIRM Results Manifest

Updated: 2026-07-02

This manifest lists the active Stage 0/1/2 outputs for the current full
initial-claim pipeline. Archived historical/debug runs are not active evidence.

## Active Outputs

| Stage | Directory | Primary artifact |
|---|---|---|
| Stage 0 literature grounding | `review-stage/literature-grounding-gpt55/` | `data/claims/literature_grounded_claims.csv` |
| Stage 1 full claim drafting | `review-stage/initial-claims-all-gpt55/` | `drafted_contracts.jsonl` |
| Stage 2 CONFIRM gates | `review-stage/confirm-gates-all-gpt55/` | `combined_benchmark_results.json` |

## Current Counts

| Stage | Count |
|---|---:|
| executable literature-grounded questions | 41 |
| LLM-proposed questions | 250 |
| total Stage 1 questions | 291 |
| drafted contracts | 289 |
| Stage 2 evaluated contracts | 289 |
| Stage 2 execution errors | 0 |

Stage 2 label distribution:

| Label | Count |
|---|---:|
| confirmed | 74 |
| fragile | 169 |
| non_replicated | 38 |
| under_powered | 8 |

## Launch Commands

Stage 0:

```bash
scripts/launch_literature_claim_grounding.sh
```

Stage 1:

```bash
MAX_WORKERS=16 PARALLEL_BACKEND=thread scripts/launch_initial_claim_drafting.sh
```

Stage 2:

```bash
MAX_WORKERS=16 PARALLEL_BACKEND=process scripts/launch_confirm_gate_evaluation.sh
```

Optional feedback loop:

```bash
MAX_ROUNDS=1 MAX_CANDIDATES=2 scripts/launch_claim_search_fullscale.sh
```

## Scope Rules

- Stage 0 literature seeds are part of initial claim creation, not final claims.
- Stage 1 LLM-proposed questions are generated separately from literature
  grounding and default to 50 per target family.
- Stage 2 runs unchanged CONFIRM gates; LLMs do not assign final labels.
- Feedback-loop outputs, synthetic stress runs, and external-only debug runs are
  separate auxiliary evidence until explicitly included in an experiment.
- Archived folders under `review-stage/_archive_*/` are local history only.
