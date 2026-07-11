# Auxiliary External Evaluation Note

Updated: 2026-07-10

NACC and ds000030/CNP remain reserved as auxiliary external evaluation sources.
They are not part of the active Stage 0/1/2 full initial-claim result unless a
later experiment explicitly supplies them as validation evidence.

Current active Stage 2 artifact:

```text
review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json
```

Potential external roles:

| Dataset | Intended role |
|---|---|
| NACC | External AD/aging evaluation where prepared labels/features match a generated or original claim. |
| CNP/ds000030 | External psychosis evaluation where schema compatibility permits execution. |

The active evidence manifest currently gives 35 of 289 Stage 2 contracts a
schema-compatible primary external set, all through NACC for AD/aging sMRI.
The CNP sMRI table does not cover the current psychosis contracts, which are
primarily fMRI. EHBS, LA5C, BLSA, AIBL, PK_MPRC, ADHD-Suijing/PKU, Olin, and
Shile are staged through the external-preparation workflow described in
`IMPLEMENTATION_NOTES.md`; none is active until canonical validation and
explicit promotion succeed.

External evaluation should be reported separately from:

- original Stage 2 discovery/replication gate labels;
- internal holdout evaluation for feedback-loop follow-up candidates;
- synthetic/null stress tests;
- same-data exploratory support from adaptive follow-up claims.

Any future pooled benchmark must name the exact artifacts and evidence roles it
includes.
