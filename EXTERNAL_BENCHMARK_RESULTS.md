# External Evidence Results

Updated: 2026-07-23

NACC and ds000030/CNP have two distinct roles and must not be pooled.

## NeuroClaimBench v2.1 External Transfer

The v2.1 external-transfer split is retrospective because both datasets were
queried during method development. Literature references and constructed
random controls use separate denominators:

| Dataset | Reference basis | Result |
|---|---|---|
| NACC | literature confirm | 0/7 recovered |
| NACC | constructed abstain | 0/14 confirmed |
| ds000030 | literature confirm | 0/11 recovered |
| ds000030 | literature abstain | 0/8 confirmed |
| ds000030 | constructed abstain | 0/8 confirmed |

Twelve additional executable external claims are unresolved and all abstain;
they do not enter accuracy denominators. All 60 external-transfer tasks
completed without an execution error. These results show conservative
abstention with low positive recovery, not broad external sensitivity or fresh
validation.

## Frozen Stage 2 Evidence Audit

The separate 289-contract Stage 2 evidence audit gives 35 contracts a
schema-compatible NACC mapping for AD/aging structural MRI. Twenty-four receive
retrospective support. The current psychosis contracts are primarily fMRI and
therefore do not map to the ds000030 structural-MRI evidence. These mappings
are not NeuroClaimBench accuracy results and are not pooled with the v2.1
external-transfer split.

The active Stage 2 artifact is:

```text
review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json
```

Internal holdouts, external evidence, synthetic stress tests, and adaptive
same-data support always remain separate. All current holdout, NACC, and
ds000030 results are labeled retrospective; genuinely fresh confirmation
requires untouched evidence.
