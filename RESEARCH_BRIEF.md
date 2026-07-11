# Research Brief

Updated: 2026-07-02

CONFIRM is a claim-governance pipeline for agentic neuroimaging analysis. It
turns LLM- or literature-derived scientific questions into frozen structured
claim contracts, evaluates those contracts with deterministic statistical
gates, and abstains or downgrades when evidence is insufficient.

The current full initial-claim run has three active stages:

- Stage 0: GPT-5.5 extracts literature-grounded claim seeds from PubMed and
  keeps only locally executable questions for drafting.
- Stage 1: GPT-5.5 proposes 50 additional questions per target family and
  drafts all initial questions into Pydantic-validated `ClaimContract`s.
- Stage 2: unchanged CONFIRM gates evaluate the frozen contracts.

Current full Stage 2 result:

- `289` evaluated contracts;
- `74` confirmed;
- `169` fragile;
- `38` non-replicated;
- `8` under-powered;
- `0` execution errors.

The optional feedback-loop layer starts after Stage 2. It diagnoses failed
claims and asks an LLM to propose connected follow-up candidates. Those
candidates are separately validated for provenance, anti-hacking constraints,
and evidence eligibility before any evaluation.

The project framing should distinguish:

- initial claim creation from gate evaluation;
- original confirmed claims from fragile/non-replicated/under-powered outcomes;
- same-data exploratory support from holdout or external confirmation;
- main scientific claims from auxiliary synthetic or external stress tests.
