# Current Architecture

Updated: 2026-07-16

CONFIRM is organized as a frozen-contract pipeline with an optional iterative
follow-up search.

```text
PubMed records --------------------------+
                                          +-> Stage 1 questions -> frozen ClaimContracts
LLM-proposed questions ------------------+
                                                     |
                                                     v
                                      Stage 2 deterministic CONFIRM gates
                                                     |
                                             failed contracts only
                                                     v
                            Stage 3 diagnosis -> LLM candidates -> validation
                                                     |
                                      same-data exploratory screening
                                                     |
                                  one selected candidate, at most
                                                     v
                                holdout/external evaluation (canonical only)
```

## Ownership

- `src/bench/run_literature_claim_grounding.py`: Stage 0.
- `src/bench/run_initial_claim_drafting.py`: Stage 1.
- `src/bench/run_drafted_contract_gates.py`: Stage 2.
- `src/bench/run_iterative_claim_search_replay.py`: Stage 3 replay.
- `src/confirm/claim_search.py`: candidate schema, validation, lineage, and loop policy.
- `src/confirm/verdict.py`: final CONFIRM gate verdict ownership.
- `src/confirm/evidence_partitions.py`: discovery, replication, holdout, and external evidence roles.

The LLM may draft questions, contracts, and connected candidate proposals. It
cannot assign gate outcomes, weaken frozen thresholds, reverse direction, drop
required confounds, or upgrade same-data support to final confirmation.

## Evidence Roots

```text
data/prepared_data/evidence_partitions/benchmark_ready/cohorts/  # Stage 0-2 DISC/REP
data/prepared_data/evidence_partitions/cohorts/                  # HOLDOUT/external roles
data/prepared_data/evidence_partitions/manifest.json             # role and checksum ledger
```

Sweep runs query only the first root. The canonical selected-arm run may query
one predeclared excluded evidence pair after a candidate passes same-data
screening.
