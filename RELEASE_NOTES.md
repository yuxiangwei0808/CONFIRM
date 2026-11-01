# CONFIRM 0.1.1

CONFIRM provides an agentic workflow for planning neuroimaging analyses,
evaluating reporting requirements, and proposing follow-up analyses.

This release includes:

- Command-line and Python interfaces for executing analysis plans.
- Five statistical gates and three reporting standards: Screen, Replicate,
  and Full.
- Result receipts containing the plan, input hashes, computed statistics,
  and reporting decision.
- LLM-assisted drafting and bounded follow-up search.
- Synthetic examples covering support, confounding, and replication failure.
- NeuroClaimBench v2.1: 544 inventory cases, 489 executable tasks, reference
  labels, recorded outcomes, and commands for validation and scoring.

Python 3.10 or newer is required. The demo and scoring of recorded outcomes
run locally. LLM workflows require a provider account; benchmark execution
requires the prepared cohort files described in the
[data-access guide](benchmark/neuroclaimbench-v2.1/DATA_ACCESS.md).
