# Experiment Readiness Audit

Date: 2026-07-16
Auditor: independent GPT-5.5 xhigh, read-only
Scope: active Stage 0-3 code and retained GPT-5.5 result artifacts

## Overall Verdict: READY_WITH_WARNINGS

No code or result blocker was found before the full v5 claim-search sweep.

## Checks

### Data And Evidence Provenance: PASS

Stage 0 literature seeds are identified as source material rather than CONFIRM
results. Stage 1 and Stage 2 reject excluded-evidence cohorts. The v5 smoke used
only current discovery/replication data: holdout and external roots were empty,
the excluded-query count was zero, and the excluded-query ledger was empty.

### Metric Integrity: PASS

Multiplicity uses the declared, observed, and cumulative candidate-search family
sizes. Each tested candidate increases the effective family size. Same-data
adaptive support is labeled `exploratory_confirmed` and does not increment final
confirmation. Sweep configuration selection uses valid-connected-executable
lineage coverage and does not use support counts.

### Artifact Consistency: PASS

- Stage 0: 99 PubMed records, 274 extracted seeds, 41 executable questions.
- Stage 1: 291 questions, 289 drafted contracts, 2 explicit draft errors.
- Stage 2: 289 evaluated contracts, 0 execution errors; 74 `confirmed`, 169
  `fragile`, 38 `non_replicated`, and 8 `under_powered`.
- Stage 3 v5 smoke: 194 of 194 eligible failed claims completed; 356 unique
  retained candidates, 351 valid connected executable candidates, 1
  `exploratory_confirmed`, 0 final confirmations, and 0 execution errors.

### Active Code Paths: PASS

All active launchers resolve to existing runners. No active reference remains to
the removed curated benchmark, one-shot proposal, external-only benchmark, or
old audit runners.

### Full-Sweep Readiness: PASS

The `R=1, C=2` smoke completed every eligible lineage with no skipped search,
execution error, or non-identifiable analysis. This validates execution
readiness for the 12-arm sweep; it is not evidence of scientific efficacy.

### Evaluation Classification: PASS

- Stage 0: literature-derived source generation.
- Stage 1: LLM drafting with deterministic schema and semantic preflight.
- Stage 2: observational discovery/replication evaluation.
- Stage 3 sweep: adaptive same-data observational candidate search.
- Safety experiment: synthetic known-negative evaluation.
- Canonical selected arm: retrospective excluded holdout/external evaluation.

### Reproducibility: WARN

The smoke provenance records the command, Git SHA and dirty flag, model, prompt
and schema hashes, source and evidence-manifest hashes, partition hashes, random
seeds, rendered-prompt hash, and excluded-query ledger. Two limitations remain:

1. The smoke was run from a dirty worktree; provenance records the dirty count
   but not the full dirty-file list.
2. Some evidence-manifest records have null stored `content_sha256` values. The
   smoke recomputed content hashes for resolved unique partition IDs, so this is
   a metadata-granularity warning rather than missing evidence integrity.

## Readiness Decision

The full v5 sweep may proceed. Run it from a committed or otherwise snapshotted
worktree if stronger byte-level reproducibility is required. Do not interpret
the smoke's single exploratory pass as a final confirmation.
