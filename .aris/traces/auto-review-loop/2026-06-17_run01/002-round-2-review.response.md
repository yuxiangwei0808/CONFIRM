# Round 2 Review — Codex gpt-5.5 (xhigh)

threadId: 019ed7ba-b621-7ae3-9df8-44caf875aae3 (fresh thread; round-1 thread lost to MCP restart, round-1 context re-supplied)
Date: 2026-06-17

**Score: 5.8/10. Verdict: No, but closer to Almost.**

## Raw response (verbatim)

Score: 5.8/10. Verdict: No, but closer to Almost. Would not accept yet at NeurIPS/ICML. The expansion helps, but the core claim is still undercut by exactly the failure mode you are trying to govern: two injected fishing nulls reach `confirmed`. Also 31/57 labels remain provisional, so the benchmark still partly grades itself. Promising paper with two hard blockers.

Ranked remaining weaknesses:
1. False-confirmed fishing nulls are the central failure case. If agentic fishing passes multiplicity/confound/power/multiverse/replication, reviewers will say CONFIRM governs polished claims, not search-induced claims.
2. Label authority still only partially solved. Primary result cannot rest on author-assigned labels. Use only externally sourced/adjudicated labels for the main benchmark; move provisional to supplementary.
3. AD headline not cleanly confirmed. I²≈95% = direction-replicated but not transportable in magnitude. Mismatched clinical definitions = construct-validity problem.
4. Known-positive recall too narrow. 5/5 CI [0.48,1.0], AD-dominated, not enough to show the layer preserves real discoveries broadly.
5. FCR uncertainty still large (2/23=0.087, upper CI 0.28).
6. Data gaps limit the benchmark claim (sMRI only ADNI/OASIS3; fMRI cohorts lack usable site labels).

Principled fixes:
- Fishing nulls = search-provenance failure; do NOT tune alpha. Add a claim lineage / search family field (every hypothesis the agent touched). Correct multiplicity over the full searched family, or abstain with `unverifiable_search_provenance`. Rebuild fishing-null controls with discovery/replication separation (fishing uses only discovery; replication untouched). Run within-cohort within-confound-stratum permutation nulls. Audit leakage (covariate-only prediction, site/scanner/age/sex leakage, ComBat leakage, subject overlap, whether the null tracks cohort structure). Keep the failures in the main table.
- AD heterogeneity = claim scoping, not another p-value gate. Split `confirmed` into direction_confirmed / magnitude_confirmed / transportable_confirmed. With I²≈95% AD should not be a headline magnitude/transportability claim. Harmonize ADNI/OASIS3 estimand (case definition, severity, covariates, eTIV/scanner, decision target). If impossible, mark replicated_but_heterogeneous / construct_mismatch. Report random-effects + prediction intervals; with 2 cohorts treat I² as a warning flag.
- Data gaps: get real sMRI derivative tables from more cohorts (age/sex/eTIV/site) or stop claiming sMRI aging/sex coverage. Recover true site IDs from source phenotypic files; if you cannot, use cohort-level replication clearly named as such (do not simulate site).
- Defensible current reframing: "CONFIRM is evaluated on a partially externally adjudicated cross-cohort derivative benchmark, strongest for AD-related claims and fMRI descriptor associations. It reduces false confirmations relative to execution-only baselines, but current results expose unresolved adversarial fishing and heterogeneity failures."

To become top-tier ready, next round must fix label authority and the fishing-null failure in a principled, protocol-wide way.

## Executor diagnosis of the fishing-null confirmations (post-review)
Construction (src/bench/injected_nulls.py + label table): "select discovery null with random replication" — a feature is selected in a discovery split, then tested in a replication split of the SAME cohort, with multiplicity family_size=1. Root cause: the multiplicity gate accounts only for the final reported claim, NOT the family of features searched to find it. CONFIRM governs the final claim's statistics but is blind to the search that produced it (garden of forking paths). Fix direction = search-provenance / claim-lineage gate (family-wise correction or `unverifiable_search_provenance` abstention) + discovery/replication separation in the null builder. This matches the reviewer's W1-central diagnosis and converts the bug into a headline mechanism.
