# Feedback Loop Full-Scale Summary

Run date: 2026-06-26

## What Changed

- Added deterministic post-verdict feedback in `src/confirm/feedback.py`.
- Added revision response and policy validation for one-step repairs.
- Added configurable pipeline feedback output: `confirm run --feedback on` and `confirm ask --feedback on`; default remains off.
- Added replay and one-step repair experiments:
  - `src/bench/run_feedback_replay.py`
  - `src/bench/run_agentic_feedback_benchmark.py`

## E7 Replay

Artifact: `review-stage/feedback-loop-fullscale-20260626/replay/feedback_replay.json`

- Feedback rows: 289
- Abstentions: 257
- Feedback coverage over abstentions: 1.00
- Actionable abstentions: 257
- Primary failures: multiplicity 140, confound 54, search_provenance 43, replication 10, multiverse 7, power 3
- Repairability: contract_repairable 88, downgrade_only 147, needs_new_data 22

Interpretation: every abstention is mapped to a deterministic scientific/governance instruction.

## E8 Live Full-Scale

Artifact: `review-stage/feedback-loop-fullscale-20260626/agentic-feedback-live-v2/agentic_feedback_benchmark_full_scale_live_v2.json`

Models: `openai:gpt-5-mini`, `openai:gpt-4o`, `anthropic:claude-haiku-4-5`, `anthropic:claude-sonnet-4-5`, `openrouter:deepseek/deepseek-chat`, `openrouter:qwen/qwen-2.5-72b-instruct`.

| Arm | Attempted revisions | Valid revisions | Appropriate resolutions | Estimand-match improvement | Policy violations | False confirmations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generic retry | 24 | 3 (0.125) | 3 (0.125) | 2/10 (0.20) | 15 (0.625) | 0 |
| Structured feedback | 24 | 16 (0.667) | 14 (0.583) | 5/10 (0.50) | 5 (0.208) | 0 |

Interpretation: structured CONFIRM feedback materially improves valid/appropriate one-step revisions and lowers policy-violation/gaming rate without increasing false confirmations.

## Controlled Feedback-Following Baseline

Artifact: `review-stage/feedback-loop-fullscale-20260626/agentic-feedback-controlled-v4/agentic_feedback_benchmark_full_scale_controlled_v4.json`

| Arm | Attempted revisions | Valid revisions | Appropriate resolutions | Estimand-match improvement | Policy violations | False confirmations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generic retry | 8 | 0 (0.000) | 0 (0.000) | 0/6 (0.000) | 8 (1.000) | 0 |
| Structured feedback | 8 | 8 (1.000) | 8 (1.000) | 5/6 (0.833) | 0 (0.000) | 0 |

Interpretation: this is not evidence that all LLMs will follow feedback. It validates that the feedback schema plus policy validator can produce safe repair/triage when followed.

## Local Review

External secondary Codex review was attempted but blocked by data-export policy. A local adversarial review scored this extension 7/10, verdict almost.

Remaining limitations:

- E7 actionability is template-derived; no human-rating study yet.
- Live E8 still has 5/24 structured policy violations and 3/24 revision draft errors, so the feedback prompt and parser remain important engineering surfaces.
- The policy validator rejects structural gaming but cannot prove that a newly selected replication cohort was not chosen after seeing results.
- The controlled feedback-following baseline should be framed as a sanity check, not as live-agent evidence.

Final claim supported by current results: CONFIRM feedback improves safe one-step repair/triage over generic retry in the live multi-LLM benchmark, while preserving the main safety target of zero observed false confirmations in retry arms.
